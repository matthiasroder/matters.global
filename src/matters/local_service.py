"""Account-free, explicitly scoped operations for local Matters clients.

The registry is supplied by the operator, never by a tool argument. This
module owns graph selection and the local access policy; ``rules`` remains
the owner of graph validation, mutation semantics and write transactions.
No MCP SDK or web server is imported here.
"""

import re
from collections.abc import Mapping
from pathlib import Path

from . import rules
from .engine import dependents, prerequisites
from .reports import unlock_report
from .sources import sources_for_matter
from .storage import load_state, state_revision


SCHEMA_VERSION = 1
GRAPH_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
MAX_PAGE_SIZE = 500


class LocalServiceError(ValueError):
    """A client-readable error independent of its transport."""

    def __init__(self, message, code="invalid", cycle=()):
        super().__init__(message)
        self.message = message
        self.code = code
        self.cycle = tuple(cycle)


class LocalGraphService:
    """Use only existing files registered by the local process owner.

    The process's operating-system identity and filesystem permissions are
    its authority. ``read_only`` further limits that authority. These are
    local-process boundaries, not a remote-user authentication system.
    """

    def __init__(self, graphs: Mapping, *, read_only=False):
        if not isinstance(graphs, Mapping) or not graphs:
            raise LocalServiceError("At least one configured graph is required.")
        if type(read_only) is not bool:
            raise LocalServiceError("read_only must be a boolean.")
        self.read_only = read_only
        self._graphs = {}
        for graph_id, state_path in graphs.items():
            if not isinstance(graph_id, str) or not GRAPH_ID_PATTERN.fullmatch(graph_id):
                raise LocalServiceError(
                    "Configured graph IDs must start with a letter or number, contain only "
                    "letters, numbers, underscores, hyphens or dots, and have at most 128 characters."
                )
            if not isinstance(state_path, (str, Path)) or not str(state_path).strip():
                raise LocalServiceError("Configured graphs require an existing state file.")
            path = Path(state_path).expanduser().resolve()
            if not path.is_file():
                raise LocalServiceError(
                    f"Configured graph {graph_id!r} does not have an existing state file.",
                    "not_found",
                )
            self._graphs[graph_id] = path

    def _path(self, graph_id):
        # Identical refusal for malformed and unknown IDs; never interpret an
        # ID as a path or reveal the names/paths of other configured graphs.
        if not isinstance(graph_id, str) or graph_id not in self._graphs:
            raise LocalServiceError("Graph access is not configured.", "forbidden")
        path = self._graphs[graph_id]
        if not path.is_file():
            raise LocalServiceError("The configured graph file is unavailable.", "not_found")
        return path

    @staticmethod
    def _load(path):
        # load_state normally treats a missing path as an empty graph. A
        # configured MCP graph must remain an existing file instead.
        def existing_file(state_path):
            if not Path(state_path).is_file():
                raise rules.RuleError("The configured graph file is unavailable.", "not_found")
            return load_state(state_path)

        return rules.load_state_or_rule_error(path, existing_file)

    @staticmethod
    def _translate(error):
        return LocalServiceError(str(error), error.code, error.cycle)

    @staticmethod
    def _envelope(graph_id, snapshot):
        return {
            "schema_version": SCHEMA_VERSION,
            "graph_id": graph_id,
            "revision": state_revision(*snapshot),
        }

    @staticmethod
    def _state(snapshot):
        matters, conditions, dependencies = snapshot
        return {
            "schema_version": 2,
            "matters": sorted(matters),
            "conditions": conditions,
            "dependencies": [list(edge) for edge in sorted(dependencies)],
        }

    def catalog(self):
        """Discover permitted graph IDs; filesystem paths are not published."""

        entries = []
        try:
            for graph_id in sorted(self._graphs):
                snapshot = self._load(self._path(graph_id))
                entries.append({
                    "graph_id": graph_id,
                    "revision": state_revision(*snapshot),
                    "read_only": self.read_only,
                })
        except rules.RuleError as error:
            raise self._translate(error) from error
        return {"schema_version": SCHEMA_VERSION, "graphs": entries}

    def read(
        self, graph_id, *, view="universe", matter_id=None, offset=0,
        limit=100, include_sources=False,
    ):
        """Read a single graph snapshot with a revision of those exact facts."""

        path = self._path(graph_id)
        if type(include_sources) is not bool:
            raise LocalServiceError("include_sources must be a boolean.")
        try:
            if view == "export":
                # The exporter owns its snapshot. A second file read could
                # attach another writer's revision to the exported facts.
                from .portability import export_bundle

                bundle = export_bundle(path)
                state = bundle["graph"]
                snapshot = (
                    set(state["matters"]), state["conditions"],
                    {tuple(edge) for edge in state["dependencies"]},
                )
                return {**self._envelope(graph_id, snapshot), "bundle": bundle}

            snapshot = self._load(path)
            matters, conditions, dependencies = snapshot
            result = self._envelope(graph_id, snapshot)
            if view == "state":
                return {**result, "state": self._state(snapshot)}
            if view == "matters":
                if type(offset) is not int or offset < 0:
                    raise LocalServiceError("offset must be a nonnegative integer.")
                if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
                    raise LocalServiceError(f"limit must be an integer from 1 to {MAX_PAGE_SIZE}.")
                ordered = sorted(matters)
                return {
                    **result, "matters": ordered[offset:offset + limit],
                    "total": len(ordered), "offset": offset, "limit": limit,
                }
            if view == "matter":
                self._matter_id(matter_id)
                rules.require_matter(matter_id, matters)
                result.update({
                    "id": matter_id,
                    "conditions": [
                        {"index": index, **condition}
                        for index, condition in enumerate(conditions.get(matter_id, []))
                    ],
                    "prerequisites": sorted(prerequisites(matter_id, dependencies)),
                    "dependents": sorted(dependents(matter_id, dependencies)),
                })
                if include_sources:
                    result["sources"] = sources_for_matter(path, matter_id)
                return result

            if view not in {"universe", "frontier", "horizon", "unlock"}:
                raise LocalServiceError("Unknown graph read view.")
            index = rules.build_index(*snapshot)
            if view == "unlock":
                return {**result, **unlock_report(*snapshot, index=index)}
            if view == "universe":
                return {**result, "matters": sorted(index.universe)}
            self._matter_id(matter_id)
            rules.require_matter(matter_id, matters)
            selected = getattr(index, view)(matter_id)
            return {**result, "root": matter_id, "matters": sorted(selected)}
        except rules.RuleError as error:
            raise self._translate(error) from error
        except (OSError, ValueError) as error:
            if isinstance(error, LocalServiceError):
                raise
            raise LocalServiceError(str(error)) from error

    @staticmethod
    def _matter_id(value):
        if not isinstance(value, str):
            raise LocalServiceError("matter_id must be a string.")
        return rules.require_matter_id_syntax(value)

    @staticmethod
    def _index(payload):
        # This is transport shape validation only. The rules layer checks
        # bounds against the current, revision-checked condition list.
        value = payload.get("index")
        if type(value) is not int or value < 0:
            raise LocalServiceError("index must be a nonnegative integer.")
        return value

    @staticmethod
    def _boolean(payload, name, *, default=None):
        value = payload.get(name, default)
        if type(value) is not bool:
            raise LocalServiceError(f"{name} must be a boolean.")
        return value

    def mutate(self, graph_id, operation, payload, *, expected_revision):
        """Run one revision-checked engine operation against an allowed graph."""

        path = self._path(graph_id)
        if self.read_only:
            raise LocalServiceError("This graph connection is read-only.", "forbidden")
        if not isinstance(expected_revision, str) or not expected_revision.strip():
            raise LocalServiceError("A nonempty expected_revision is required.", "conflict")
        if not isinstance(payload, dict):
            raise LocalServiceError("Mutation payload must be an object.")
        shapes = {
            "create_matter": {"id", "conditions"},
            "set_condition": {"matter_id", "index", "truth"},
            "add_condition": {"matter_id", "label", "truth"},
            "edit_condition": {"matter_id", "index", "label"},
            "delete_condition": {"matter_id", "index", "confirmed"},
            "delete_matter": {"matter_id", "cascade", "confirmed"},
            "add_dependency": {"source", "target"},
            "remove_dependency": {"source", "target"},
        }
        if not isinstance(operation, str) or operation not in shapes:
            raise LocalServiceError("Unknown graph mutation operation.")
        if set(payload) - shapes[operation]:
            raise LocalServiceError("Mutation payload contains unsupported fields.")

        # Retain the objects loaded under the rule transaction. Rules mutate
        # these objects in place, so their final revision describes this
        # operation's committed facts even if a second writer immediately
        # replaces the on-disk file. No derived graph reads run after save.
        captured = []

        def capture(state_path):
            snapshot = self._load(state_path)
            captured.append(snapshot)
            return snapshot

        kwargs = {"load": capture, "expected_revision": expected_revision}
        result = None
        try:
            if operation == "create_matter":
                self._matter_id(payload.get("id"))
                rules.create_matter(path, payload, **kwargs)
            elif operation in {"add_dependency", "remove_dependency"}:
                self._matter_id(payload.get("source"))
                self._matter_id(payload.get("target"))
                getattr(rules, operation)(path, payload, **kwargs)
            else:
                matter_id = self._matter_id(payload.get("matter_id"))
                if operation == "delete_matter":
                    result = rules.delete_matter(
                        path, matter_id,
                        cascade=self._boolean(payload, "cascade", default=False),
                        confirmed=self._boolean(payload, "confirmed", default=False),
                        **kwargs,
                    )
                elif operation == "delete_condition":
                    result = rules.delete_condition(
                        path, matter_id, str(self._index(payload) + 1),
                        confirmed=self._boolean(payload, "confirmed", default=False),
                        **kwargs,
                    )
                else:
                    update = {}
                    if operation in {"set_condition", "edit_condition"}:
                        update["index"] = self._index(payload)
                    if operation in {"set_condition", "add_condition"}:
                        update["truth"] = self._boolean(
                            payload, "truth", default=False if operation == "add_condition" else None
                        )
                    if operation in {"add_condition", "edit_condition"}:
                        if not isinstance(payload.get("label"), str):
                            raise LocalServiceError("label must be a string.")
                        update["label"] = payload["label"]
                    rules.update_conditions(path, matter_id, update, **kwargs)
        except rules.RuleError as error:
            raise self._translate(error) from error
        except OSError as error:
            raise LocalServiceError("The graph could not be saved.", "unavailable") from error

        return {
            **self._envelope(graph_id, captured[0]),
            "result": {"operation": operation, **(result or {})},
        }
