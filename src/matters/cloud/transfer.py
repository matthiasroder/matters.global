"""Authenticated graph-package transfer without server filesystem access."""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, StrictStr


class GraphImport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: Annotated[StrictStr, Field(min_length=1, max_length=200)]
    kind: Literal["personal", "project"] = "project"
    bundle: dict[str, Any]


def create_transfer_router(repository, authenticated_user):
    """Use the application's cookie identity and normal request protections.

    Imports always create new graph identities. No route takes an acting user,
    destination graph ID, server path or installation permission record.
    """
    router = APIRouter()

    @router.post("/api/graphs/import", status_code=201)
    def import_graph(body: GraphImport, user=Depends(authenticated_user)):
        return repository.import_graph(user["id"], body.title, body.bundle, body.kind)

    @router.get("/api/graphs/{graph_id}/export")
    def export_graph(graph_id: str, user=Depends(authenticated_user)):
        return repository.export_graph(user["id"], graph_id)

    return router
