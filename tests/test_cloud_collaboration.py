"""Whole-graph invitations grant only the intended authenticated account."""

import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("argon2")

from fastapi.testclient import TestClient
from sqlalchemy import event, select
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateSchema, DropSchema

from matters.cloud.app import create_app
from matters.cloud.collaboration import CloudCollaboration
from matters.cloud.db import Database, graph_invitations, graphs, users
from matters.cloud.repository import CloudGraphError, GraphRepository


@pytest.fixture(params=["sqlite", "postgresql"])
def database(request, tmp_path):
    if request.param == "sqlite":
        db = Database(f"sqlite:///{tmp_path / 'collaboration.db'}").initialize()
        yield db
        db.close()
        return
    configured = os.environ.get("MATTERS_TEST_DATABASE_URL")
    if not configured:
        pytest.skip("MATTERS_TEST_DATABASE_URL is required for PostgreSQL checks")
    administration = Database(configured)
    schema = "collaboration_" + uuid.uuid4().hex
    with administration.engine.begin() as connection:
        connection.execute(CreateSchema(schema))
    db = Database(make_url(configured).update_query_dict({"options": "-csearch_path=" + schema})).initialize()
    try:
        yield db
    finally:
        db.close()
        with administration.engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        administration.close()


@pytest.fixture
def setup(database):
    people = {name: str(uuid.uuid4()) for name in ("alice", "bob", "carol")}
    with database.transaction(write=True) as connection:
        connection.execute(users.insert(), [
            {"id": user_id, "email": name + "@example.com", "password_hash": "test-only", "created_at": 1}
            for name, user_id in people.items()
        ])
    repository = GraphRepository(database)
    graph = repository.create_graph(people["alice"], "Shared project")
    clock = [1_800_000_000]
    return CloudCollaboration(repository, now=lambda: clock[0]), repository, people, graph, clock


def invite(setup, role="viewer", **options):
    service, _, people, graph, _ = setup
    return service.create_invitation(people["alice"], graph["graph_id"], " Bob@EXAMPLE.COM ", role, **options)


def test_invitation_token_is_hashed_and_not_listed(setup, database):
    service, _, people, graph, _ = setup
    created = invite(setup)
    assert created["invitation"]["email"] == "bob@example.com"
    assert created["invitation"]["status"] == "pending"
    assert len(created["token"]) == 43
    with database.transaction() as connection:
        stored = connection.execute(select(graph_invitations)).mappings().one()
    assert stored["token_hash"] == hashlib.sha256(created["token"].encode()).hexdigest()
    assert created["token"] not in stored.values()
    listing = service.list_access(people["alice"], graph["graph_id"])
    assert listing["owner"]["user_id"] == people["alice"]
    assert listing["members"] == []
    assert created["token"] not in json.dumps(listing)
    assert "token_hash" not in json.dumps(listing)


def test_acceptance_is_bound_to_email_and_one_graph(setup):
    service, repository, people, graph, _ = setup
    personal = repository.create_graph(people["alice"], "Alice personal", "personal")
    created = invite(setup, "editor")
    with pytest.raises(CloudGraphError) as wrong:
        service.accept_invitation(people["carol"], created["token"])
    assert wrong.value.code == "not_found"
    with pytest.raises(CloudGraphError):
        repository.get_graph(people["bob"], graph["graph_id"])
    accepted = service.accept_invitation(people["bob"], created["token"])
    assert accepted["graph_id"] == graph["graph_id"]
    assert accepted["role"] == "editor"
    assert accepted["invitation"]["status"] == "accepted"
    assert repository.get_graph(people["bob"], graph["graph_id"])["role"] == "editor"
    with pytest.raises(CloudGraphError):
        repository.get_graph(people["bob"], personal["graph_id"])
    changed = repository.mutate_graph(
        people["bob"], graph["graph_id"], "create_matter", {"id": "review_result"},
        expected_revision=graph["revision"],
    )
    assert changed["state"]["matters"] == ["review_result"]
    with pytest.raises(CloudGraphError) as replay:
        service.accept_invitation(people["bob"], created["token"])
    assert replay.value.code == "conflict"


def test_viewer_role_changes_and_revocation_are_enforced(setup):
    service, repository, people, graph, _ = setup
    created = invite(setup)
    service.accept_invitation(people["bob"], created["token"])
    with pytest.raises(CloudGraphError) as readonly:
        repository.mutate_graph(people["bob"], graph["graph_id"], "create_matter", {"id": "blocked"}, expected_revision=graph["revision"])
    assert readonly.value.code == "forbidden"
    with pytest.raises(CloudGraphError):
        service.set_member_role(people["alice"], graph["graph_id"], people["carol"], "editor")
    service.set_member_role(people["alice"], graph["graph_id"], people["bob"], "editor")
    assert repository.get_graph(people["bob"], graph["graph_id"])["role"] == "editor"
    upgrade = invite(setup, "editor")
    removed = service.remove_member(people["alice"], graph["graph_id"], people["bob"])
    assert removed["removed"] is True
    assert removed["revoked_invitations"] == 1
    with pytest.raises(CloudGraphError):
        repository.get_graph(people["bob"], graph["graph_id"])
    with pytest.raises(CloudGraphError):
        service.accept_invitation(people["bob"], upgrade["token"])
    assert repository.list_graphs(people["bob"])["graphs"] == []


def test_owner_only_access_management(setup):
    service, repository, people, graph, _ = setup
    created = invite(setup)
    service.accept_invitation(people["bob"], created["token"])
    another = service.create_invitation(people["alice"], graph["graph_id"], "carol@example.com")
    for action in (
        lambda: service.list_access(people["bob"], graph["graph_id"]),
        lambda: service.create_invitation(people["bob"], graph["graph_id"], "carol@example.com"),
        lambda: service.revoke_invitation(people["bob"], graph["graph_id"], another["invitation"]["id"]),
        lambda: service.set_member_role(people["bob"], graph["graph_id"], people["bob"], "editor"),
        lambda: service.remove_member(people["bob"], graph["graph_id"], people["alice"]),
    ):
        with pytest.raises(CloudGraphError) as denied:
            action()
        assert denied.value.code == "forbidden"
    assert repository.get_graph(people["bob"], graph["graph_id"])["role"] == "viewer"


def test_expiry_revocation_and_superseded_links(setup):
    service, _, people, graph, clock = setup
    expired = invite(setup, ttl_seconds=10)
    clock[0] += 10
    assert service.list_access(people["alice"], graph["graph_id"])["invitations"][0]["status"] == "expired"
    with pytest.raises(CloudGraphError) as error:
        service.accept_invitation(people["bob"], expired["token"])
    assert error.value.code == "conflict"
    old = invite(setup)
    replacement = invite(setup, "editor")
    with pytest.raises(CloudGraphError):
        service.accept_invitation(people["bob"], old["token"])
    service.revoke_invitation(people["alice"], graph["graph_id"], replacement["invitation"]["id"])
    with pytest.raises(CloudGraphError):
        service.accept_invitation(people["bob"], replacement["token"])


def test_expiration_is_checked_after_transaction_ownership(setup, monkeypatch):
    service, repository, people, _, clock = setup
    created = invite(setup, ttl_seconds=10)
    original = repository._transaction

    @contextmanager
    def delayed(**options):
        with original(**options) as connection:
            clock[0] += 10
            yield connection

    monkeypatch.setattr(repository, "_transaction", delayed)
    with pytest.raises(CloudGraphError) as error:
        service.accept_invitation(people["bob"], created["token"])
    assert error.value.code == "conflict"


@pytest.mark.parametrize("email,role,ttl", [
    ("alice@example.com", "viewer", 60),
    ("not an email", "viewer", 60),
    ("bob@example.com", "owner", 60),
    ("bob@example.com", "viewer", True),
    ("bob@example.com", "viewer", 0),
    ("bob@example.com", "viewer", 31 * 86400),
])
def test_invalid_invitations_grant_nothing(setup, email, role, ttl):
    service, _, people, graph, _ = setup
    with pytest.raises(CloudGraphError):
        service.create_invitation(people["alice"], graph["graph_id"], email, role, ttl_seconds=ttl)
    assert service.list_access(people["alice"], graph["graph_id"])["invitations"] == []


def test_concurrent_acceptance_has_one_success(setup):
    service, _, people, _, _ = setup
    created = invite(setup)
    barrier = threading.Barrier(2)

    def accept():
        barrier.wait(timeout=10)
        try:
            return service.accept_invitation(people["bob"], created["token"])
        except CloudGraphError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [executor.submit(accept), executor.submit(accept)]
        results = [future.result(timeout=15) for future in results]
    assert sum(isinstance(result, dict) for result in results) == 1
    assert results.count("conflict") == 1


def test_waiting_acceptance_rechecks_revocation_after_graph_lock(setup, database):
    if database.engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row-lock interleaving check")
    service, _, people, graph, clock = setup
    created = invite(setup)
    waiting = threading.Event()

    def observed(_connection, _cursor, statement, _parameters, _context, _many):
        if "FOR UPDATE" in statement and "graphs" in statement:
            waiting.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with database.transaction(write=True) as connection:
            connection.execute(select(graphs.c.id).where(graphs.c.id == graph["graph_id"]).with_for_update()).one()
            event.listen(database.engine, "before_cursor_execute", observed)
            future = executor.submit(service.accept_invitation, people["bob"], created["token"])
            assert waiting.wait(timeout=10)
            connection.execute(graph_invitations.update().where(
                graph_invitations.c.id == created["invitation"]["id"]
            ).values(revoked_at=clock[0]))
        try:
            with pytest.raises(CloudGraphError) as error:
                future.result(timeout=15)
            assert error.value.code == "conflict"
        finally:
            event.remove(database.engine, "before_cursor_execute", observed)


def test_http_invitation_flow_uses_authenticated_identity(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'http.db'}", secure_cookies=False)
    with TestClient(app, base_url="http://localhost") as client:
        tokens = {}
        for name in ("alice", "bob"):
            body = {"email": name + "@example.com", "password": "Example-only-password-42!"}
            assert client.post("/api/auth/register", json=body).status_code == 201
            login = client.post("/api/auth/login", json=body)
            tokens[name] = login.cookies.get("matters_session")
            client.cookies.clear()
        owner = {"Cookie": "matters_session=" + tokens["alice"]}
        invited = {"Cookie": "matters_session=" + tokens["bob"]}
        graph = client.post("/api/graphs", json={"title": "Shared project"}, headers=owner).json()
        endpoint = f"/api/graphs/{graph['graph_id']}/invitations"
        created = client.post(endpoint, json={"email": "bob@example.com", "role": "viewer"}, headers=owner)
        assert created.status_code == 201, created.text
        token = created.json()["token"]
        assert client.get(f"/api/graphs/{graph['graph_id']}/sharing", headers=invited).status_code == 404
        assert client.post("/api/invitations/accept", json={"token": token, "actor_id": "alice"}, headers=invited).status_code == 422
        accepted = client.post("/api/invitations/accept", json={"token": token}, headers=invited)
        assert accepted.status_code == 200, accepted.text
        assert client.get(f"/api/graphs/{graph['graph_id']}", headers=invited).json()["role"] == "viewer"
        assert client.post("/api/invitations/accept", json={"token": token}, headers=invited).status_code == 409
        listing = client.get(f"/api/graphs/{graph['graph_id']}/sharing", headers=owner).json()
        assert token not in json.dumps(listing)
        member_id = listing["members"][0]["user_id"]
        assert client.delete(f"/api/graphs/{graph['graph_id']}/members/{member_id}", headers=owner).status_code == 200
        assert client.get(f"/api/graphs/{graph['graph_id']}", headers=invited).status_code == 404
