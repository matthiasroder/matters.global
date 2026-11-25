"""Hosted visuals reuse the local graph payload without local file access."""

import json
import uuid
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from matters import web
from matters.cloud.app import create_app
from matters.cloud.db import Database, users
from matters.cloud.repository import CloudGraphError, GraphRepository
from matters.cloud.visual import visual_graph


EXAMPLE = Path(__file__).parents[1] / "examples" / "matters.example.json"


def test_visual_model_reuses_local_payload_without_filesystem(tmp_path, monkeypatch):
    db = Database(f"sqlite:///{tmp_path / 'visual.db'}").initialize()
    actor = str(uuid.uuid4())
    stranger = str(uuid.uuid4())
    with db.transaction(write=True) as connection:
        connection.execute(users.insert(), [
            {"id": identity, "email": name + "@example.com", "password_hash": "test-only", "created_at": 1}
            for identity, name in ((actor, "alice"), (stranger, "bob"))
        ])
    repo = GraphRepository(db)
    expected = web.graph_payload(EXAMPLE)
    created = repo.create_graph(actor, "Visual example", state=json.loads(EXAMPLE.read_text()))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Hosted visual tried to open a local graph or shell")

    monkeypatch.setattr(web, "load_state", forbidden)
    monkeypatch.setattr(web.TerminalManager, "create", forbidden)
    actual = visual_graph(repo, actor, created["graph_id"])
    for key in ("nodes", "edges", "universe", "unlock", "overview_layout", "revision"):
        assert actual[key] == expected[key]
    assert actual["graph_id"] == created["graph_id"]
    assert actual["role"] == "owner"
    assert "state_path" not in actual
    with pytest.raises(CloudGraphError) as error:
        visual_graph(repo, stranger, created["graph_id"])
    assert error.value.code == "not_found"
    db.close()


def test_visual_http_auth_and_reused_renderer_asset(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'http.db'}", secure_cookies=False)
    with TestClient(app, base_url="http://localhost") as client:
        body = {"email": "alice@example.com", "password": "Example-only-password-42!"}
        assert client.post("/api/auth/register", json=body).status_code == 201
        assert client.post("/api/auth/login", json=body).status_code == 200
        graph = client.post("/api/graphs", json={"title": "Visual project"}).json()
        response = client.get(f"/api/graphs/{graph['graph_id']}/visual")
        assert response.status_code == 200, response.text
        assert response.json()["graph_id"] == graph["graph_id"]
        assert "state_path" not in response.json()
        renderer = client.get("/assets/local/map-renderer.js")
        assert renderer.status_code == 200
        original = Path(web.web_assets_path()).joinpath("map-renderer.js").read_bytes()
        assert renderer.content == original
        assert "createOverviewRenderer" in renderer.text
        assert client.post("/api/terminal/sessions", json={}).status_code == 404
        assert client.post("/api/state", json={"path": "/tmp/private.json"}).status_code == 404
        client.post("/api/auth/logout")
        assert client.get(f"/api/graphs/{graph['graph_id']}/visual").status_code == 401
