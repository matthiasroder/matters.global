"""Independent HTTP, persistence and isolation checks for Cloud foundations.

SQLite files always run. Set MATTERS_TEST_DATABASE_URL to an explicitly chosen
PostgreSQL test database to run the same checks there; every test gets its own
temporary schema and never reads or resets another test's tables.
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")
pytest.importorskip("email_validator")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.engine import make_url

from matters.cloud.app import create_app
from matters.cloud.db import graph_events, sessions, users
from matters.web import graph_payload


BASE_URL = "http://localhost"
COOKIE = "matters_session"
PASSWORD = "Repository-example-password-42!"
EXAMPLE = Path(__file__).parents[1] / "examples" / "matters.example.json"
BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("MATTERS_TEST_DATABASE_URL") else [])


@pytest.fixture(params=BACKENDS)
def database_url(request, tmp_path):
    if request.param == "sqlite":
        yield f"sqlite+pysqlite:///{tmp_path / 'cloud-integration.sqlite'}"
        return

    pytest.importorskip("psycopg")
    configured = make_url(os.environ["MATTERS_TEST_DATABASE_URL"])
    assert configured.get_backend_name() == "postgresql", "Expected a dedicated PostgreSQL test database"
    schema = "matters_integration_" + uuid.uuid4().hex
    admin = create_engine(configured)
    try:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        isolated = configured.update_query_dict({"options": f"-csearch_path={schema}"})
        yield isolated.render_as_string(hide_password=False)
    finally:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        admin.dispose()


def app_for(database_url, **options):
    return create_app(database_url, secure_cookies=False, public_origin=BASE_URL, **options)


def headers(token=None, **extra):
    result = {"Origin": BASE_URL}
    if token is not None:
        result["Cookie"] = f"{COOKIE}={token}"
    return {**result, **extra}


def body(response, expected_status=200):
    assert response.status_code == expected_status, response.text
    assert "no-store" in response.headers.get("cache-control", "").lower()
    result = response.json()
    assert result["schema_version"] == 1
    return result


def register_and_login(client, email):
    registered = body(client.post(
        "/api/auth/register", headers=headers(), json={"email": email, "password": PASSWORD},
    ), 201)
    response = client.post(
        "/api/auth/login", headers=headers(), json={"email": email, "password": PASSWORD},
    )
    logged_in = body(response)
    assert logged_in["user"]["id"] == registered["user"]["id"]
    serialized = json.dumps(logged_in)
    assert PASSWORD not in serialized
    assert not {"token", "password", "password_hash", "token_hash"}.intersection(logged_in["user"])
    token = response.cookies.get(COOKIE)
    assert token and len(token) >= 32
    assert token not in serialized
    cookie_headers = response.headers.get_list("set-cookie")
    assert any("httponly" in value.lower() and "samesite=strict" in value.lower() for value in cookie_headers)
    # Every later call chooses its own session, so no accidental cookie-jar
    # sharing can make a two-user isolation assertion pass as the wrong actor.
    client.cookies.clear()
    return registered["user"], token


def create_graph(client, token, title="Project example", kind="project"):
    graph = body(client.post(
        "/api/graphs", headers=headers(token),
        json={"title": title, "kind": kind},
    ), 201)
    # Seed through the public operations API; initial graph upload/import is
    # outside this foundation API. Every intermediate write is revision-checked.
    example = json.loads(EXAMPLE.read_text())
    for matter_id in example["matters"]:
        graph = body(mutate(client, token, graph, "create_matter", {
            "id": matter_id, "conditions": example["conditions"][matter_id],
        }))
    for source, target in example["dependencies"]:
        graph = body(mutate(client, token, graph, "add_dependency", {
            "source": source, "target": target,
        }))
    return get_graph(client, token, graph["graph_id"])


def graph_snapshot(result):
    """Mutation responses additionally carry an operation result."""
    return {key: value for key, value in result.items() if key != "result"}


def get_graph(client, token, graph_id):
    return body(client.get(f"/api/graphs/{graph_id}", headers=headers(token)))


def mutate(client, token, graph, operation, payload):
    return client.post(
        f"/api/graphs/{graph['graph_id']}/operations", headers=headers(token),
        json={"operation": operation, "payload": payload, "expected_revision": graph["revision"]},
    )


def test_two_users_own_separate_personal_and_project_graphs(database_url):
    app = app_for(database_url)
    with TestClient(app, base_url=BASE_URL) as client:
        alice, alice_token = register_and_login(client, "alice@example.com")
        bob, bob_token = register_and_login(client, "bob@example.com")
        assert alice["id"] != bob["id"]
        assert body(client.get("/api/graphs", headers=headers(alice_token)))["graphs"] == []
        personal = create_graph(client, alice_token, "Alice personal", "personal")
        project = create_graph(client, alice_token, "Alice project")
        bob_project = create_graph(client, bob_token, "Bob project")
        assert len({personal["graph_id"], project["graph_id"], bob_project["graph_id"]}) == 3
        for graph, kind in ((personal, "personal"), (project, "project")):
            assert graph["kind"] == kind
            assert graph["role"] == "owner"
            assert get_graph(client, alice_token, graph["graph_id"])["state"] == graph["state"]
        alice_ids = {graph["graph_id"] for graph in body(client.get("/api/graphs", headers=headers(alice_token)))["graphs"]}
        assert alice_ids == {personal["graph_id"], project["graph_id"]}
        # A claimed actor in a query cannot override the session identity.
        spoofed = body(client.get(
            "/api/graphs", headers=headers(bob_token), params={"user_id": alice["id"]},
        ), 400)
        assert spoofed["error"]["code"] == "invalid"
        bob_list = body(client.get("/api/graphs", headers=headers(bob_token)))
        assert {graph["graph_id"] for graph in bob_list["graphs"]} == {bob_project["graph_id"]}

        unknown = str(uuid.uuid4())
        forbidden = body(client.get(f"/api/graphs/{project['graph_id']}", headers=headers(bob_token)), 404)
        absent = body(client.get(f"/api/graphs/{unknown}", headers=headers(bob_token)), 404)
        assert forbidden == absent
        assert forbidden["error"]["code"] == "not_found"
        for view in ("state", "matters", "matter", "universe", "frontier", "horizon", "unlock"):
            response = client.get(
                f"/api/graphs/{project['graph_id']}/views/{view}", headers=headers(bob_token),
                params={"matter_id": "define_project_goal"},
            )
            assert body(response, 404) == forbidden
        body(client.get(f"/api/graphs/{project['graph_id']}"), 401)


def test_foreign_graph_mutations_and_actor_spoofs_are_rejected(database_url):
    app = app_for(database_url)
    with TestClient(app, base_url=BASE_URL) as client:
        alice, alice_token = register_and_login(client, "alice@example.com")
        _, bob_token = register_and_login(client, "bob@example.com")
        graph = create_graph(client, alice_token)
        before = get_graph(client, alice_token, graph["graph_id"])
        operations = [
            ("create_matter", {"id": "intrusion", "conditions": [{"label": "Intrusion done", "truth": False}]}),
            ("set_condition", {"matter_id": "ship_first_version", "index": 0, "truth": True}),
            ("add_condition", {"matter_id": "ship_first_version", "label": "Intrusion done", "truth": False}),
            ("edit_condition", {"matter_id": "ship_first_version", "index": 0, "label": "Changed"}),
            ("delete_condition", {"matter_id": "ship_first_version", "index": 0, "confirmed": True}),
            ("delete_matter", {"matter_id": "ship_first_version", "confirmed": True, "cascade": True}),
            ("add_dependency", {"source": "ship_first_version", "target": "define_project_goal"}),
            ("remove_dependency", {"source": "define_project_goal", "target": "ship_first_version"}),
        ]
        for operation, payload in operations:
            rejected = body(mutate(client, bob_token, graph, operation, payload), 404)
            assert rejected["error"]["code"] == "not_found"
            assert get_graph(client, alice_token, graph["graph_id"]) == before
        for spoof in ("owner_id", "user_id", "actor_id"):
            response = client.post(
                "/api/graphs", headers=headers(bob_token),
                json={"title": "Spoofed owner", "kind": "project", spoof: alice["id"]},
            )
            body(response, 422)
        response = client.post(
            f"/api/graphs/{graph['graph_id']}/operations", headers=headers(bob_token),
            json={"operation": "set_condition", "payload": operations[1][1],
                  "expected_revision": graph["revision"], "actor_id": alice["id"]},
        )
        body(response, 422)
        assert get_graph(client, alice_token, graph["graph_id"]) == before
        assert body(client.get("/api/graphs", headers=headers(bob_token)))["graphs"] == []


def test_accounts_sessions_and_graph_edits_survive_application_restart(database_url, tmp_path):
    first_app = app_for(database_url)
    with TestClient(first_app, base_url=BASE_URL) as client:
        alice, token = register_and_login(client, "alice@example.com")
        graph = create_graph(client, token)
        changed = graph_snapshot(body(mutate(client, token, graph, "set_condition", {
            "matter_id": "ship_first_version", "index": 0, "truth": True,
        })))

        local = tmp_path / "local-parity.json"
        local.write_bytes(EXAMPLE.read_bytes())
        result = subprocess.run(
            [sys.executable, "-m", "matters.cli", "mark", "ship_first_version", "1", "true",
             "--state", str(local), "--expected-revision", graph["revision"]],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert changed["state"] == json.loads(local.read_text())
        assert changed["revision"] == graph_payload(local)["revision"]
        universe = body(client.get(
            f"/api/graphs/{graph['graph_id']}/views/universe", headers=headers(token),
        ))
        assert universe["matters"] == graph_payload(local)["universe"] == []

    # New app, new services, disposed old engine and new database connections.
    second_app = app_for(database_url)
    with TestClient(second_app, base_url=BASE_URL) as client:
        assert body(client.get("/api/auth/me", headers=headers(token)))["user"]["id"] == alice["id"]
        assert get_graph(client, token, graph["graph_id"]) == changed
        response = client.post(
            "/api/auth/login", headers=headers(),
            json={"email": "ALICE@example.com", "password": PASSWORD},
        )
        assert body(response)["user"]["id"] == alice["id"]


def test_bad_mutations_and_stale_revisions_rollback_without_history(database_url):
    app = app_for(database_url)
    with TestClient(app, base_url=BASE_URL) as client:
        _, token = register_and_login(client, "alice@example.com")
        graph = create_graph(client, token)

        def events():
            with app.state.db.transaction() as connection:
                return connection.execute(select(graph_events).where(
                    graph_events.c.graph_id == graph["graph_id"]
                )).mappings().all()

        original_events = events()
        invalid = [
            ("set_condition", {"matter_id": "ship_first_version", "index": 0, "truth": "false"}),
            ("set_condition", {"matter_id": "ship_first_version", "index": True, "truth": True}),
            ("set_condition", {"matter_id": "ship_first_version", "index": "0", "truth": True}),
            ("set_condition", {"matter_id": "ship_first_version", "index": 0.0, "truth": True}),
            ("add_dependency", {"source": "ship_first_version", "target": "define_project_goal"}),
            ("create_matter", {"id": "bad", "conditions": '[{"label":"Done","truth":false}]'}),
            ("create_matter", {"id": "bad", "conditions": [{"label": "Done", "truth": "false"}]}),
        ]
        for operation, payload in invalid:
            response = mutate(client, token, graph, operation, payload)
            assert response.status_code in (400, 409, 422), response.text
            assert body(response, response.status_code)["error"]["code"]
            assert get_graph(client, token, graph["graph_id"]) == graph
            assert events() == original_events

        missing_precondition = client.post(
            f"/api/graphs/{graph['graph_id']}/operations", headers=headers(token),
            json={"operation": "set_condition", "payload": {"matter_id": "ship_first_version", "index": 0, "truth": True}},
        )
        body(missing_precondition, 422)
        changed = graph_snapshot(body(mutate(client, token, graph, "set_condition", {
            "matter_id": "ship_first_version", "index": 0, "truth": True,
        })))
        stale = mutate(client, token, graph, "set_condition", {
            "matter_id": "ship_first_version", "index": 0, "truth": False,
        })
        assert body(stale, 409)["error"]["code"] == "conflict"
        assert get_graph(client, token, graph["graph_id"]) == changed
        assert len(events()) == len(original_events) + 1


def test_membership_roles_and_revocation_control_http_access(database_url):
    app = app_for(database_url)
    with TestClient(app, base_url=BASE_URL) as client:
        alice, alice_token = register_and_login(client, "alice@example.com")
        bob, bob_token = register_and_login(client, "bob@example.com")
        graph = create_graph(client, alice_token)
        app.state.graphs.set_member(alice["id"], graph["graph_id"], bob["id"], role="viewer")
        viewed = get_graph(client, bob_token, graph["graph_id"])
        assert viewed["role"] == "viewer"
        assert viewed["state"] == graph["state"]
        body(mutate(client, bob_token, viewed, "set_condition", {
            "matter_id": "ship_first_version", "index": 0, "truth": True,
        }), 403)
        assert get_graph(client, alice_token, graph["graph_id"])["state"] == graph["state"]

        app.state.graphs.set_member(alice["id"], graph["graph_id"], bob["id"], role="editor")
        editable = get_graph(client, bob_token, graph["graph_id"])
        assert editable["role"] == "editor"
        changed = body(mutate(client, bob_token, editable, "set_condition", {
            "matter_id": "ship_first_version", "index": 0, "truth": True,
        }))
        assert changed["state"]["conditions"]["ship_first_version"][0]["truth"] is True
        with app.state.db.transaction() as connection:
            actors = connection.execute(select(graph_events.c.actor_id).where(
                graph_events.c.graph_id == graph["graph_id"],
                graph_events.c.operation == "set_condition",
            )).scalars().all()
        assert actors == [bob["id"]]

        app.state.graphs.remove_member(alice["id"], graph["graph_id"], bob["id"])
        body(client.get(f"/api/graphs/{graph['graph_id']}", headers=headers(bob_token)), 404)
        assert body(client.get("/api/graphs", headers=headers(bob_token)))["graphs"] == []
        body(mutate(client, bob_token, changed, "set_condition", {
            "matter_id": "ship_first_version", "index": 0, "truth": False,
        }), 404)


def test_simultaneous_http_edits_commit_once_for_one_revision(database_url):
    app = app_for(database_url)
    with TestClient(app, base_url=BASE_URL) as client:
        _, token = register_and_login(client, "alice@example.com")
        graph = create_graph(client, token)
        start = threading.Barrier(2)

        def edit(matter_id):
            start.wait(timeout=10)
            return mutate(client, token, graph, "create_matter", {
                "id": matter_id, "conditions": [{"label": "Reviewed", "truth": False}],
            })

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(edit, "first_edit")
            second = pool.submit(edit, "second_edit")
            results = [first.result(timeout=20), second.result(timeout=20)]
        assert sorted(result.status_code for result in results) == [200, 409]
        winning = graph_snapshot(body(next(result for result in results if result.status_code == 200)))
        assert body(next(result for result in results if result.status_code == 409), 409)["error"]["code"] == "conflict"
        current = get_graph(client, token, graph["graph_id"])
        assert current == winning
        assert current["version"] == graph["version"] + 1
        assert len({"first_edit", "second_edit"} & set(current["state"]["matters"])) == 1


def test_sessions_are_hashed_expire_and_logout_revokes_access(database_url):
    app = app_for(database_url)
    with TestClient(app, base_url=BASE_URL) as client:
        user, token = register_and_login(client, "alice@example.com")
        digest = hashlib.sha256(token.encode()).hexdigest()
        with app.state.db.transaction() as connection:
            stored_user = connection.execute(select(users).where(users.c.id == user["id"])).mappings().one()
            stored_session = connection.execute(select(sessions).where(sessions.c.token_hash == digest)).mappings().one()
        assert stored_user["password_hash"].startswith("$argon2id$")
        assert PASSWORD not in stored_user["password_hash"]
        assert stored_session["token_hash"] != token
        assert body(client.get("/api/auth/me", headers=headers(token)))["user"] == user

        app.state.accounts._now = lambda: stored_session["expires_at"]
        body(client.get("/api/auth/me", headers=headers(token)), 401)
        body(client.get("/api/graphs", headers=headers(token)), 401)
        # Create a fresh session at the now-advanced clock, then revoke it.
        login = client.post("/api/auth/login", headers=headers(), json={"email": user["email"], "password": PASSWORD})
        body(login)
        fresh = login.cookies.get(COOKIE)
        client.cookies.clear()
        body(client.post("/api/auth/logout", headers=headers(fresh), json={}))
        body(client.get("/api/auth/me", headers=headers(fresh)), 401)
        with app.state.db.transaction() as connection:
            assert connection.execute(select(sessions).where(
                sessions.c.token_hash == hashlib.sha256(fresh.encode()).hexdigest()
            )).first() is None


def test_cross_site_mutations_and_schema_echoes_are_blocked(database_url):
    app = app_for(database_url)
    with TestClient(app, base_url=BASE_URL) as client:
        _, token = register_and_login(client, "alice@example.com")
        for extra in ({"Origin": "https://attacker.example"}, {"Sec-Fetch-Site": "cross-site"}):
            response = client.post(
                "/api/graphs", headers=headers(token, **extra),
                json={"title": "Rejected cross-site graph", "kind": "project"},
            )
            body(response, 403)
        assert body(client.get("/api/graphs", headers=headers(token)))["graphs"] == []
        for path, data in (
            ("/api/auth/login", {"email": ["private-input-marker"], "password": PASSWORD}),
            ("/api/auth/register", {"email": "alice@example.com", "password": PASSWORD, "session_token": token}),
        ):
            response = client.post(path, headers=headers(token), json=data)
            rejected = body(response, 422)
            encoded = json.dumps(rejected)
            assert PASSWORD not in encoded
            assert token not in encoded
            assert "private-input-marker" not in encoded


def test_production_cookie_defaults_are_secure(database_url):
    app = create_app(database_url, public_origin="https://localhost")
    with TestClient(app, base_url="https://localhost") as client:
        credentials = {"email": "alice@example.com", "password": PASSWORD}
        response = client.post("/api/auth/register", headers={"Origin": "https://localhost"}, json=credentials)
        body(response, 201)
        login = client.post("/api/auth/login", headers={"Origin": "https://localhost"}, json=credentials)
        body(login)
        cookie_headers = login.headers.get_list("set-cookie")
        assert any("secure" in value.lower() and "httponly" in value.lower() for value in cookie_headers)
        body(client.get("/api/auth/me"))
