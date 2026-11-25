"""Authorized visual payloads using the same local graph model and layout."""

from fastapi import Depends

from ..storage import state_from_data
from ..web import ApiError, graph_payload_from_state
from .repository import CloudGraphError


def visual_graph(repository, actor_id, graph_id):
    """Return one authorized snapshot; never create a temporary state file."""

    graph = repository.get_graph(actor_id, graph_id)
    try:
        payload = graph_payload_from_state(
            *state_from_data(graph["state"]), graph_id=graph["graph_id"]
        )
    except ApiError as error:
        raise CloudGraphError(str(error), "state_cycle") from error
    if payload["revision"] != graph["revision"]:
        raise CloudGraphError("The graph snapshot is inconsistent.", "unavailable")
    return {
        **payload, "schema_version": 1, "title": graph["title"],
        "kind": graph["kind"], "role": graph["role"], "version": graph["version"],
    }


def register_visual_routes(app, repository, authenticated_user):
    @app.get("/api/graphs/{graph_id}/visual")
    def get_visual_graph(graph_id: str, user=Depends(authenticated_user)):
        return visual_graph(repository, user["id"], graph_id)
