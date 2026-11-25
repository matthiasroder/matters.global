"""Portable packages and the hosted implementation of the local MCP service."""

import copy
import os
import uuid
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import func, select
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateSchema, DropSchema

from matters.cloud.db import Database, graph_events, graph_members, graph_sources, graphs, users
from matters.cloud.repository import CloudGraphError, GraphRepository
from matters.cloud.service import HostedGraphService
from matters.local_service import LocalGraphService, LocalServiceError
from matters.portability import build_bundle, export_bundle, import_bundle


EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.fixture(params=["sqlite", "postgresql"])
def database(request, tmp_path):
    if request.param == "sqlite":
        db = Database("sqlite:///" + str(tmp_path / "transfer.db")).initialize()
        try:
            yield db
        finally:
            db.close()
        return
    configured = os.environ.get("MATTERS_TEST_DATABASE_URL")
    if not configured:
        pytest.skip("MATTERS_TEST_DATABASE_URL is required for PostgreSQL checks")
    administrative = Database(configured)
    schema = "transfer_" + uuid.uuid4().hex
    with administrative.engine.begin() as connection:
        connection.execute(CreateSchema(schema))
    db = Database(make_url(configured).update_query_dict({"options": "-csearch_path=" + schema})).initialize()
    try:
        yield db
    finally:
        db.close()
        with administrative.engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        administrative.close()


@pytest.fixture
def actors(database):
    result = {name: str(uuid.uuid4()) for name in ("alice", "bob", "carol")}
    with database.transaction(write=True) as connection:
        connection.execute(users.insert(), [
            {"id": identity, "email": name + "@example.org", "password_hash": "test-hash", "created_at": 1}
            for name, identity in result.items()
        ])
    return result


@pytest.fixture
def repository(database):
    return GraphRepository(database)


@pytest.fixture
def bundle():
    return build_bundle({
        "schema_version": 2,
        "matters": ["plan", "ship"],
        "conditions": {
            "plan": [{"label": "Scope written", "truth": True}],
            "ship": [{"label": "Someone can use it", "truth": False}],
        },
        "dependencies": [["plan", "ship"]],
    }, {"schema_version": 1, "matter_sources": {"ship": [{
        "title": "Useful evidence", "url": "https://example.org/evidence.pdf",
        "authors": ["Léa Example"], "pdf_pages": [[2, 3]],
        "private_installation_key": "must-not-transfer",
    }]}})


def counts(database):
    with database.transaction() as connection:
        return tuple(connection.execute(select(func.count()).select_from(table)).scalar_one() for table in (graphs, graph_sources, graph_events, graph_members))


def service(repository, actor, graph_ids, scopes=("graphs:read", "graphs:write")):
    principal = {"actor_id": actor, "graph_ids": list(graph_ids), "scopes": list(scopes)}
    return HostedGraphService(repository, lambda: principal)


@pytest.mark.parametrize("relative", ["matters.example.json", "creativity_matters_graph/creativity_graph.json"])
def test_repository_example_file_cloud_file_roundtrip(repository, actors, tmp_path, relative):
    original = export_bundle(EXAMPLES / relative)
    cloud = repository.import_graph(actors["alice"], "Imported example", original, "personal")
    exported = repository.export_graph(actors["alice"], cloud["graph_id"])
    assert exported["bundle"] == original
    assert exported["revision"] == cloud["revision"]
    destination = tmp_path / "new-local-graph.json"
    import_bundle(exported["bundle"], destination)
    assert export_bundle(destination) == original
    # A second installation creates a fresh owner and identity, never inherited ACL.
    second = repository.import_graph(actors["bob"], "Moved graph", exported["bundle"])
    assert second["graph_id"] != cloud["graph_id"]
    assert second["role"] == "owner"
    with pytest.raises(CloudGraphError) as denied:
        repository.get_graph(actors["alice"], second["graph_id"])
    assert denied.value.code == "not_found"


def test_legacy_native_import_and_permission_metadata_is_not_transferred(repository, actors, database, bundle):
    package = copy.deepcopy(bundle)
    package.update(owner_id=actors["bob"], graph_id="old-id", members=[actors["bob"]])
    package["graph"].update(owner_id=actors["bob"], permissions={actors["bob"]: "owner"})
    package["sources"]["matter_sources"]["ship"][0]["private_installation_key"] = "must-not-transfer"
    imported = repository.import_graph(actors["alice"], "Fresh graph", package)
    exported = repository.export_graph(actors["alice"], imported["graph_id"])["bundle"]
    assert exported == bundle
    assert "private_installation_key" not in str(exported)
    with database.transaction() as connection:
        assert connection.execute(select(graphs.c.owner_id).where(graphs.c.id == imported["graph_id"])).scalar_one() == actors["alice"]
        assert connection.execute(select(func.count()).select_from(graph_members)).scalar_one() == 0
    native = {"matters": ["legacy"], "conditions": {"legacy": [True]}, "dependencies": []}
    legacy = repository.import_graph(actors["alice"], "Legacy", native)
    assert legacy["state"]["conditions"]["legacy"] == [{"label": "Unlabeled legacy condition 1", "truth": True}]


@pytest.mark.parametrize("case", ["cycle", "orphan", "unknown-source", "version", "bad-citation"])
def test_invalid_imports_leave_no_partial_rows(repository, database, actors, bundle, case):
    invalid = copy.deepcopy(bundle)
    if case == "cycle":
        invalid["graph"]["dependencies"].append(["ship", "plan"])
    elif case == "orphan":
        invalid["graph"]["dependencies"].append(["missing", "plan"])
    elif case == "unknown-source":
        invalid["sources"]["matter_sources"]["missing"] = []
    elif case == "version":
        invalid["schema_version"] = True
    else:
        invalid["sources"]["matter_sources"]["ship"][0]["pdf_pages"] = [[3, 2]]
    before = counts(database)
    with pytest.raises(CloudGraphError):
        repository.import_graph(actors["alice"], "Bad package", invalid)
    assert counts(database) == before


def test_import_rolls_back_graph_and_citations_if_final_audit_write_fails(repository, database, actors, bundle, monkeypatch):
    before = counts(database)
    def fail(*_):
        raise RuntimeError("simulated audit failure")
    monkeypatch.setattr(repository, "_event", fail)
    with pytest.raises(RuntimeError, match="simulated audit failure"):
        repository.import_graph(actors["alice"], "Must roll back", bundle)
    assert counts(database) == before


def test_sources_survive_restart_and_are_removed_with_deleted_matter(repository, database, actors, bundle):
    imported = repository.import_graph(actors["alice"], "Sources", bundle)
    restarted = GraphRepository(database)
    graph = restarted.get_graph_with_sources(actors["alice"], imported["graph_id"])
    assert graph["sources"] == bundle["sources"]
    changed = restarted.mutate_graph(actors["alice"], imported["graph_id"], "delete_matter", {
        "matter_id": "ship", "confirmed": True,
    }, expected_revision=imported["revision"])
    exported = restarted.export_graph(actors["alice"], imported["graph_id"])
    assert exported["revision"] == changed["revision"]
    assert exported["bundle"]["graph"]["matters"] == ["plan"]
    assert exported["bundle"]["sources"]["matter_sources"] == {}


def test_export_and_source_reads_check_current_membership(repository, actors, bundle):
    graph = repository.import_graph(actors["alice"], "Source graph", bundle)
    for method in (repository.export_graph, repository.get_graph_with_sources):
        with pytest.raises(CloudGraphError) as denied:
            method(actors["bob"], graph["graph_id"])
        assert denied.value.code == "not_found"
    repository.set_member(actors["alice"], graph["graph_id"], actors["bob"], "viewer")
    assert repository.export_graph(actors["bob"], graph["graph_id"])["bundle"] == bundle
    repository.remove_member(actors["alice"], graph["graph_id"], actors["bob"])
    with pytest.raises(CloudGraphError):
        repository.export_graph(actors["bob"], graph["graph_id"])


def test_rejected_and_rolled_back_deletion_preserve_citations(repository, actors, bundle, monkeypatch):
    graph = repository.import_graph(actors["alice"], "Citations are transactional", bundle)
    payload = {"matter_id": "ship", "confirmed": True}
    with pytest.raises(CloudGraphError) as stale:
        repository.mutate_graph(actors["alice"], graph["graph_id"], "delete_matter", payload, expected_revision="stale")
    assert stale.value.code == "conflict"
    assert repository.export_graph(actors["alice"], graph["graph_id"])["bundle"] == bundle
    def fail(*_):
        raise RuntimeError("simulated final audit failure")
    monkeypatch.setattr(repository, "_event", fail)
    with pytest.raises(RuntimeError, match="simulated final audit failure"):
        repository.mutate_graph(actors["alice"], graph["graph_id"], "delete_matter", payload, expected_revision=graph["revision"])
    assert repository.export_graph(actors["alice"], graph["graph_id"])["bundle"] == bundle


def test_hosted_reads_match_local_contract_and_do_not_expose_other_graphs(repository, actors, bundle, tmp_path):
    graph = repository.import_graph(actors["alice"], "Granted", bundle)
    other = repository.import_graph(actors["alice"], "Not granted", bundle)
    path = tmp_path / "local.json"
    import_bundle(bundle, path)
    local = LocalGraphService({graph["graph_id"]: path})
    hosted = service(repository, actors["alice"], [graph["graph_id"]])
    assert [g["graph_id"] for g in hosted.catalog()["graphs"]] == [graph["graph_id"]]
    for view, matter in [("state", None), ("matters", None), ("matter", "ship"), ("universe", None), ("frontier", "plan"), ("horizon", "plan"), ("unlock", None), ("export", None)]:
        local_read = local.read(graph["graph_id"], view=view, matter_id=matter, include_sources=True)
        hosted_read = hosted.read(graph["graph_id"], view=view, matter_id=matter, include_sources=True)
        assert {k: hosted_read[k] for k in local_read} == local_read
    page = hosted.read(graph["graph_id"], view="matters", offset=1, limit=1)
    assert page["matters"] == ["ship"] and page["total"] == 2
    with pytest.raises(LocalServiceError) as denied:
        hosted.read(other["graph_id"])
    assert denied.value.code == "forbidden"


def test_every_hosted_call_revalidates_principal_and_current_acl(repository, actors, bundle):
    graph = repository.import_graph(actors["alice"], "Shared", bundle)
    repository.set_member(actors["alice"], graph["graph_id"], actors["bob"], "editor")
    calls = []
    active = True
    def principal():
        calls.append(True)
        if not active:
            raise LocalServiceError("Grant revoked.", "unauthorized")
        return {"actor_id": actors["bob"], "graph_ids": [graph["graph_id"]], "scopes": ["graphs:read", "graphs:write"]}
    hosted = HostedGraphService(repository, principal)
    assert hosted.catalog()["graphs"][0]["role"] == "editor"
    hosted.read(graph["graph_id"])
    changed = hosted.mutate(graph["graph_id"], "set_condition", {"matter_id": "ship", "index": 0, "truth": True}, expected_revision=graph["revision"])
    assert len(calls) == 3
    repository.remove_member(actors["alice"], graph["graph_id"], actors["bob"])
    with pytest.raises(LocalServiceError) as denied:
        hosted.read(graph["graph_id"], view="export")
    assert denied.value.code == "not_found"
    assert hosted.catalog()["graphs"] == []
    active = False
    with pytest.raises(LocalServiceError) as revoked:
        hosted.mutate(graph["graph_id"], "set_condition", {"matter_id": "ship", "index": 0, "truth": False}, expected_revision=changed["revision"])
    assert revoked.value.code == "unauthorized"
    assert repository.get_graph(actors["alice"], graph["graph_id"])["revision"] == changed["revision"]


def test_hosted_read_scope_and_viewer_roles_both_refuse_writes(repository, actors, bundle):
    graph = repository.import_graph(actors["alice"], "Read only", bundle)
    scoped = service(repository, actors["alice"], [graph["graph_id"]], scopes=["graphs:read"])
    assert scoped.catalog()["graphs"][0]["read_only"] is True
    payload = {"matter_id": "ship", "index": 0, "truth": True}
    with pytest.raises(LocalServiceError) as denied:
        scoped.mutate(graph["graph_id"], "set_condition", payload, expected_revision=graph["revision"])
    assert denied.value.code == "forbidden"
    repository.set_member(actors["alice"], graph["graph_id"], actors["bob"], "viewer")
    viewer = service(repository, actors["bob"], [graph["graph_id"]])
    assert viewer.catalog()["graphs"][0]["read_only"] is True
    with pytest.raises(LocalServiceError) as denied:
        viewer.mutate(graph["graph_id"], "set_condition", payload, expected_revision=graph["revision"])
    assert denied.value.code == "forbidden"
    assert repository.get_graph(actors["alice"], graph["graph_id"])["revision"] == graph["revision"]


@pytest.mark.parametrize("options", [{"offset": True}, {"offset": -1}, {"limit": True}, {"limit": 501}, {"include_sources": "true"}])
def test_hosted_pagination_and_source_flags_are_strict(repository, actors, bundle, options):
    graph = repository.import_graph(actors["alice"], "Strict", bundle)
    hosted = service(repository, actors["alice"], [graph["graph_id"]])
    with pytest.raises(LocalServiceError) as invalid:
        hosted.read(graph["graph_id"], view="matters", **options)
    assert invalid.value.code == "invalid"


def test_transfer_http_uses_session_identity_and_refuses_paths_and_foreign_exports(tmp_path, bundle):
    pytest.importorskip("fastapi")
    pytest.importorskip("argon2")
    pytest.importorskip("email_validator")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from matters.cloud.app import create_app
    app = create_app("sqlite:///" + str(tmp_path / "http.db"), secure_cookies=False)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.post("/api/graphs/import", json={"title": "Anonymous", "bundle": bundle}).status_code == 401
        credentials = {"email": "alice.transfer@example.org", "password": "Repository-example-password"}
        assert client.post("/api/auth/register", json=credentials).status_code == 201
        assert client.post("/api/auth/login", json=credentials).status_code == 200
        imported = client.post("/api/graphs/import", json={"title": "Alice import", "kind": "personal", "bundle": bundle})
        assert imported.status_code == 201, imported.text
        graph_id = imported.json()["graph_id"]
        assert client.get(f"/api/graphs/{graph_id}/export").json()["bundle"] == bundle
        for extra in ({"owner_id": "bob"}, {"path": "/tmp/cloud.json"}, {"graph_id": graph_id}):
            assert client.post("/api/graphs/import", json={"title": "Must refuse", "bundle": bundle, **extra}).status_code == 422
        assert client.get(f"/api/graphs/{graph_id}/export").headers["cache-control"] == "no-store"
        client.post("/api/auth/logout")
        credentials["email"] = "bob.transfer@example.org"
        assert client.post("/api/auth/register", json=credentials).status_code == 201
        assert client.post("/api/auth/login", json=credentials).status_code == 200
        assert client.get(f"/api/graphs/{graph_id}/export").status_code == 404
