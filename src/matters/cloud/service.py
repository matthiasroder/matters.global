"""The local MCP graph contract backed by authenticated hosted storage.

The principal callback is supplied by the transport, never by tool arguments.
It must freshly validate the bearer token/grant on every call. A grant only
narrows access: the repository still checks current graph membership.
"""

from collections.abc import Mapping

from .. import rules
from ..engine import dependents, prerequisites
from ..local_service import LocalServiceError, MAX_PAGE_SIZE
from .repository import CloudGraphError


class HostedGraphService:
    def __init__(self, repository, principal_provider):
        if not callable(principal_provider):
            raise ValueError("A fresh authenticated principal provider is required.")
        self.repository = repository
        self.principal_provider = principal_provider

    def _principal(self, *, write=False):
        principal = self.principal_provider()
        if not isinstance(principal, Mapping):
            raise LocalServiceError("Authentication is required.", "unauthorized")
        actor = principal.get("actor_id")
        graph_ids = principal.get("graph_ids")
        scopes = principal.get("scopes")
        if (
            not isinstance(actor, str) or not actor
            or not isinstance(graph_ids, (list, tuple, set, frozenset))
            or any(not isinstance(graph, str) or not graph for graph in graph_ids)
            or not isinstance(scopes, (list, tuple, set, frozenset))
            or any(not isinstance(scope, str) for scope in scopes)
        ):
            raise LocalServiceError("Authentication is required.", "unauthorized")
        if "graphs:read" not in scopes or (write and "graphs:write" not in scopes):
            raise LocalServiceError("This connection does not permit that operation.", "forbidden")
        return actor, frozenset(graph_ids), frozenset(scopes)

    def _access(self, graph_id, *, write=False):
        principal = self._principal(write=write)
        if not isinstance(graph_id, str) or graph_id not in principal[1]:
            raise LocalServiceError("Graph access is not authorized.", "forbidden")
        return principal

    @staticmethod
    def _translate(error):
        return LocalServiceError(error.message, error.code, error.cycle)

    def catalog(self):
        actor, allowed, scopes = self._principal()
        try:
            listed = self.repository.list_graphs(actor)
        except CloudGraphError as error:
            raise self._translate(error) from error
        return {
            "schema_version": 1,
            "graphs": [
                {**graph, "read_only": graph["role"] == "viewer" or "graphs:write" not in scopes}
                for graph in listed["graphs"] if graph["graph_id"] in allowed
            ],
        }

    def read(self, graph_id, *, view="universe", matter_id=None, offset=0, limit=100, include_sources=False):
        actor, _, _ = self._access(graph_id)
        if type(include_sources) is not bool:
            raise LocalServiceError("include_sources must be a boolean.")
        try:
            if view == "export":
                return self.repository.export_graph(actor, graph_id)
            if view == "matter" and include_sources:
                graph = self.repository.get_graph_with_sources(actor, graph_id)
                state = graph["state"]
                if not isinstance(matter_id, str):
                    raise LocalServiceError("matter_id must be a string.")
                rules.require_matter_id_syntax(matter_id)
                rules.require_matter(matter_id, set(state["matters"]))
                edges = {tuple(edge) for edge in state["dependencies"]}
                return {
                    **{key: value for key, value in graph.items() if key not in {"state", "sources"}},
                    "id": matter_id,
                    "conditions": [{"index": index, **condition} for index, condition in enumerate(state["conditions"].get(matter_id, []))],
                    "prerequisites": sorted(prerequisites(matter_id, edges)),
                    "dependents": sorted(dependents(matter_id, edges)),
                    "sources": graph["sources"]["matter_sources"].get(matter_id, []),
                }
            result = self.repository.query_graph(actor, graph_id, view, matter_id=matter_id)
            if view == "matters":
                if type(offset) is not int or offset < 0:
                    raise LocalServiceError("offset must be a nonnegative integer.")
                if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
                    raise LocalServiceError(f"limit must be an integer from 1 to {MAX_PAGE_SIZE}.")
                result = {**result, "matters": result["matters"][offset:offset + limit], "offset": offset, "limit": limit}
            return result
        except CloudGraphError as error:
            raise self._translate(error) from error
        except rules.RuleError as error:
            raise LocalServiceError(str(error), error.code, error.cycle) from error

    def mutate(self, graph_id, operation, payload, *, expected_revision):
        actor, _, _ = self._access(graph_id, write=True)
        try:
            return self.repository.mutate_graph(actor, graph_id, operation, payload, expected_revision=expected_revision)
        except CloudGraphError as error:
            raise self._translate(error) from error
