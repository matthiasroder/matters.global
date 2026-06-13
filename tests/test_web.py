import http.client
import json
import threading
from functools import partial
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import matters.web as web
from matters.cli import main
from matters.web import (
    ApiError,
    add_dependency,
    create_matter,
    graph_payload,
    MattersWebHandler,
    remove_dependency,
    resolve_terminal_workspace,
    run_command,
    StatePathStore,
    switch_state_path,
    TerminalManager,
    update_conditions,
)


ASSETS = Path(__file__).parents[1] / "src" / "matters" / "web_assets"


def write_state(path, data=None):
    path.write_text(
        json.dumps(
            data
            or {
                "matters": ["a", "b"],
                "conditions": {
                    "a": [{"label": "a done", "truth": True}],
                    "b": [{"label": "b done", "truth": False}],
                },
                "dependencies": [["a", "b"]],
            }
        )
    )


def api_request(state_path, method, path, body="", headers=None):
    handler = partial(
        MattersWebHandler,
        state_paths=StatePathStore(state_path),
        terminal_manager=TerminalManager(default_workspace=state_path.parent),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request_headers = headers(server.server_port) if callable(headers) else headers
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request(method, path, body=body, headers=request_headers or {})
        response = conn.getresponse()
        response_body = response.read()
        return response.status, response_body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_graph_payload_includes_derived_status(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    payload = graph_payload(state_path)

    nodes = {node["id"]: node for node in payload["nodes"]}
    assert nodes["a"]["resolved"] is True
    assert nodes["b"]["actionable"] is True
    assert nodes["b"]["prerequisites"] == ["a"]
    assert payload["edges"] == [{"source": "a", "target": "b"}]


def test_create_matter_persists_conditions(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})

    create_matter(
        state_path,
        {
            "title": "Ship web UI",
            "conditions": [{"label": "UI opens in browser", "truth": False}],
        },
    )

    data = json.loads(state_path.read_text())
    assert data["matters"] == ["ship_web_ui"]
    assert data["conditions"]["ship_web_ui"] == [
        {"label": "UI opens in browser", "truth": False}
    ]


def test_api_rejects_cross_origin_text_plain_mutation(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})

    status, body = api_request(
        state_path,
        "POST",
        "/api/matters",
        body=json.dumps({"title": "Cross origin write", "conditions": ["done"]}),
        headers={
            "Origin": "https://attacker.example",
            "Content-Type": "text/plain",
        },
    )

    assert status == HTTPStatus.FORBIDDEN
    assert b"cross-origin API request rejected" in body
    assert json.loads(state_path.read_text())["matters"] == []


def test_api_accepts_same_origin_json_mutation(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})

    status, _ = api_request(
        state_path,
        "POST",
        "/api/matters",
        body=json.dumps({"title": "Same origin write", "conditions": ["done"]}),
        headers=lambda port: {
            "Origin": f"http://127.0.0.1:{port}",
            "Content-Type": "application/json",
        },
    )

    assert status == HTTPStatus.CREATED
    assert json.loads(state_path.read_text())["matters"] == ["same_origin_write"]


def test_api_rejects_non_json_mutation(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})

    status, body = api_request(
        state_path,
        "POST",
        "/api/matters",
        body=json.dumps({"title": "Wrong media type"}),
        headers={"Content-Type": "text/plain"},
    )

    assert status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    assert b"application/json" in body
    assert json.loads(state_path.read_text())["matters"] == []


def test_concurrent_create_matter_preserves_all_writes(tmp_path, monkeypatch):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})
    original_load_state = web.load_state
    barrier = threading.Barrier(2)
    barrier_threads = set()
    barrier_lock = threading.Lock()

    def racing_load_state(path):
        result = original_load_state(path)
        thread_name = threading.current_thread().name
        with barrier_lock:
            should_wait = thread_name.startswith("writer-") and thread_name not in barrier_threads
            if should_wait:
                barrier_threads.add(thread_name)
        if should_wait:
            try:
                barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
        return result

    monkeypatch.setattr(web, "load_state", racing_load_state)
    errors = []

    def write(title):
        try:
            create_matter(state_path, {"title": title, "conditions": ["done"]})
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [
        threading.Thread(target=write, name="writer-a", args=("First write",)),
        threading.Thread(target=write, name="writer-b", args=("Second write",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert set(json.loads(state_path.read_text())["matters"]) == {
        "first_write",
        "second_write",
    }


def test_update_conditions_can_add_and_toggle(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    update_conditions(state_path, "b", {"label": "reviewed", "truth": False})
    update_conditions(state_path, "b", {"action": "toggle", "index": 1})

    data = json.loads(state_path.read_text())
    assert data["conditions"]["b"][1] == {"label": "reviewed", "truth": True}


def test_update_conditions_can_edit_label(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    update_conditions(state_path, "b", {"index": 0, "label": "b shipped", "truth": False})

    data = json.loads(state_path.read_text())
    assert data["conditions"]["b"][0] == {"label": "b shipped", "truth": False}


def test_add_and_remove_dependency(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(
        state_path,
        {
            "matters": ["a", "b"],
            "conditions": {"a": [], "b": []},
            "dependencies": [],
        },
    )

    add_dependency(state_path, {"source": "a", "target": "b"})
    assert json.loads(state_path.read_text())["dependencies"] == [["a", "b"]]

    remove_dependency(state_path, {"source": "a", "target": "b"})
    assert json.loads(state_path.read_text())["dependencies"] == []


def test_add_dependency_rejects_cycles(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    with pytest.raises(ApiError, match="cycle"):
        add_dependency(state_path, {"source": "b", "target": "a"})

    assert json.loads(state_path.read_text())["dependencies"] == [["a", "b"]]


def test_command_endpoint_runs_graph_operations(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    assert run_command(state_path, {"text": "universe"}) == {
        "type": "universe",
        "items": ["b"],
    }
    assert run_command(state_path, {"text": "frontier a"}) == {
        "type": "frontier",
        "matter": "a",
        "items": ["b"],
    }
    assert run_command(state_path, {"text": "horizon a"}) == {
        "type": "horizon",
        "matter": "a",
        "items": ["b"],
    }
    assert run_command(state_path, {"text": "unlock"})["report"]["universe"] == ["b"]


def test_command_endpoint_can_create_from_expression(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})

    result = run_command(state_path, {"text": "create goal (done) > prerequisite"})

    assert result["type"] == "create"
    assert json.loads(state_path.read_text())["dependencies"] == [["prerequisite", "goal"]]


def test_switch_state_path_changes_active_graph(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_state(first, {"matters": ["first"], "conditions": {"first": []}, "dependencies": []})
    write_state(second, {"matters": ["second"], "conditions": {"second": []}, "dependencies": []})
    state_paths = StatePathStore(first)

    payload = switch_state_path(state_paths, {"state_path": str(second)})

    assert state_paths.current() == second
    assert payload["state_path"] == str(second)
    assert [node["id"] for node in payload["nodes"]] == ["second"]


def test_switch_state_path_rejects_missing_file(tmp_path):
    state_paths = StatePathStore(tmp_path / "first.json")

    with pytest.raises(ApiError, match="state file does not exist"):
        switch_state_path(state_paths, {"state_path": str(tmp_path / "missing.json")})


def test_cli_registers_web_command(monkeypatch):
    called = {}

    def fake_serve(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("matters.web.serve", fake_serve)

    assert main(["web", "--state", "example.json", "--port", "0", "--no-open"]) == 0
    assert called == {
        "state_path": "example.json",
        "host": "127.0.0.1",
        "port": 0,
        "open_browser": False,
        "terminal_workspace": None,
        "terminal_shell": None,
    }


def test_cli_registers_terminal_options(monkeypatch):
    called = {}

    def fake_serve(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("matters.web.serve", fake_serve)

    assert (
        main(
            [
                "web",
                "--state",
                "example.json",
                "--terminal-workspace",
                "/tmp/matters-terminal",
                "--terminal-shell",
                "/bin/sh",
                "--no-open",
            ]
        )
        == 0
    )
    assert called["terminal_workspace"] == "/tmp/matters-terminal"
    assert called["terminal_shell"] == "/bin/sh"


def test_resolve_terminal_workspace_defaults_to_state_parent(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    assert resolve_terminal_workspace(state_path) == tmp_path


def test_terminal_manager_rejects_missing_workspace(tmp_path):
    manager = TerminalManager()

    with pytest.raises(ApiError, match="terminal workspace does not exist"):
        manager.create(workspace=tmp_path / "missing")


def test_web_assets_use_three_dimensional_canvas():
    html = (ASSETS / "index.html").read_text()
    app = (ASSETS / "app.js").read_text()

    assert '<div id="graph"' in html
    assert '<script type="module" src="app.js?v=api-hardening"></script>' in html
    assert '<details class="panel-section disclosure">' in html
    assert "<summary>Create Matter</summary>" in html
    assert "<summary>Dependencies</summary>" in html
    assert "Chat / Commands" not in html
    assert 'id="terminal-drawer"' in html
    assert 'id="toggle-terminal"' in html
    assert 'id="state-form"' in html
    assert "/Users/matthias/.openclaw/workspace" not in html
    assert "starting shell..." in app
    assert 'api("/api/state"' in app
    assert "@xterm/xterm@5.5.0" in html
    assert 'api("/api/terminal/sessions"' in app
    assert "new Terminal" in app
    assert "3d-force-graph@1.78.0" in app
    assert 'href="styles.css?v=switch-graph"' in html
    assert "[hidden]" in (ASSETS / "styles.css").read_text()
    assert "ForceGraph3D()(graphElement)" in app
    assert ".enableNodeDrag(false)" in app
    assert 'id="zoom-in"' in html
    assert 'id="zoom-out"' in html
    assert "function zoomCamera(factor)" in app
    assert "TorusGeometry" not in app
    assert "linkDirectionalArrowLength(1.7)" in app
    assert "d3Force(\"charge\")" in app
    assert "ORGANIC_LAYOUT" in app
    assert "function configureOrganicLayout()" in app
    assert "function graphViewportRect()" in app
    assert "async function responsePayload(response)" in app
    assert "function switchGraphStateErrorMessage(error)" in app
    assert "Restart the matters web server" in app
    assert "chargeForce" in app
    assert ".strength(ORGANIC_LAYOUT.chargeStrength)" in app
    assert 'd3Force("organicGravity", organicGravityForce(ORGANIC_LAYOUT.gravityStrength))' in app
    assert 'd3Force("statusDrift", statusDriftForce(ORGANIC_LAYOUT.statusDriftStrength))' in app
    assert 'd3Force("nodeCollision", nodeCollisionForce(ORGANIC_LAYOUT.collisionPadding))' in app
    assert "function organicSeedPosition" in app
    assert "webgl-fallback" in html
