import contextlib
import http.client
import json
import re
import threading
from functools import partial
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import matters.web as web
from matters.cli import main
from matters.web import (
    ApiError,
    add_dependency,
    api_tokens_match,
    create_matter,
    graph_payload,
    is_remote_bind_host,
    launch_url,
    MattersWebHandler,
    remove_dependency,
    request_api_token,
    resolve_terminal_workspace,
    run_command,
    StatePathStore,
    switch_state_path,
    TerminalManager,
    update_conditions,
)


ASSETS = Path(__file__).parents[1] / "src" / "matters" / "web_assets"

# Stands in for the random token `serve` mints. Tests that exercise
# authentication pass `token=` explicitly; every other test gets a valid one
# from the helper, because the token requirement is not what they are about.
VALID_TOKEN = "test-token-0123456789"


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


def api_request(state_path, method, path, body="", headers=None, token=VALID_TOKEN):
    """Make one request against a throwaway server.

    ``token`` defaults to the server's own token, so callers that are testing
    something other than authentication keep asserting exactly what they
    asserted before. Pass ``token=None`` to send no Authorization header.
    """

    handler = partial(
        MattersWebHandler,
        state_paths=StatePathStore(state_path),
        terminal_manager=TerminalManager(default_workspace=state_path.parent),
        api_token=VALID_TOKEN,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request_headers = dict(
            (headers(server.server_port) if callable(headers) else headers) or {}
        )
        if token is not None:
            request_headers.setdefault("Authorization", f"Bearer {token}")
        request_headers.setdefault(web.GRAPH_ID_HEADER, web.graph_identity(state_path))
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request(method, path, body=body, headers=request_headers)
        response = conn.getresponse()
        response_body = response.read()
        return response.status, response_body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def api_response(state_path, method, path, body="", headers=None, token=VALID_TOKEN):
    """Same as ``api_request`` but keeps the response headers too."""

    handler = partial(
        MattersWebHandler,
        state_paths=StatePathStore(state_path),
        terminal_manager=TerminalManager(default_workspace=state_path.parent),
        api_token=VALID_TOKEN,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request_headers = dict(
            (headers(server.server_port) if callable(headers) else headers) or {}
        )
        if token is not None:
            request_headers.setdefault("Authorization", f"Bearer {token}")
        request_headers.setdefault(web.GRAPH_ID_HEADER, web.graph_identity(state_path))
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request(method, path, body=body, headers=request_headers)
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@contextlib.contextmanager
def running_server(monkeypatch, state_path, **kwargs):
    """Run the real ``serve`` entry point and yield its launch URL.

    The launch URL is the only place the token is published, so the tests
    read it the same way a person does.
    """

    opened = []
    ready = threading.Event()
    captured = {}
    real_server_class = web.ThreadingHTTPServer

    def capture_server(address, handler):
        server = real_server_class(address, handler)
        captured["server"] = server
        return server

    def capture_open(url):
        opened.append(url)
        ready.set()
        return True

    monkeypatch.setattr(web, "ThreadingHTTPServer", capture_server)
    monkeypatch.setattr(web.webbrowser, "open", capture_open)
    thread = threading.Thread(
        target=web.serve,
        kwargs={"state_path": state_path, "host": "127.0.0.1", "port": 0, **kwargs},
        daemon=True,
    )
    thread.start()
    assert ready.wait(5), "serve did not reach the browser launch"
    try:
        yield opened[0]
    finally:
        captured["server"].shutdown()
        thread.join(timeout=5)


def launch_url_parts(url):
    parsed = urlparse(url)
    token = parse_qs(parsed.query).get("token", [""])[0]
    return parsed.port, token


def live_request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def test_graph_payload_includes_derived_status(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    payload = graph_payload(state_path)

    nodes = {node["id"]: node for node in payload["nodes"]}
    assert nodes["a"]["resolved"] is True
    assert nodes["b"]["actionable"] is True
    assert nodes["b"]["prerequisites"] == ["a"]
    assert payload["edges"] == [{"source": "a", "target": "b"}]
    assert set(payload) == {
        "graph_id",
        "state_path",
        "nodes",
        "edges",
        "universe",
        "unlock",
        "overview_layout",
    }
    assert set(nodes["a"]) == {
        "id",
        "label",
        "conditions",
        "prerequisites",
        "dependents",
        "resolved",
        "actionable",
        "blocked",
        "overview",
    }
    assert payload["overview_layout"] == {
        "version": 3,
        "algorithm": "solar-systems-v1",
        "max_depth": 1,
        "system_count": 1,
        "bounds": {
            "min_x": min(
                nodes["a"]["overview"]["x"], nodes["b"]["overview"]["x"]
            ),
            "max_x": max(
                nodes["a"]["overview"]["x"], nodes["b"]["overview"]["x"]
            ),
            "min_y": 0.0,
            "max_y": 8.0,
            "min_z": min(
                nodes["a"]["overview"]["z"], nodes["b"]["overview"]["z"]
            ),
            "max_z": max(
                nodes["a"]["overview"]["z"], nodes["b"]["overview"]["z"]
            ),
        },
    }
    assert nodes["a"]["overview"]["depth"] == 0
    assert nodes["a"]["overview"]["downstream_impact"] == 1
    assert nodes["a"]["overview"]["system"] == "b"
    assert nodes["a"]["overview"]["system_count"] == 1
    assert nodes["a"]["overview"]["system_population"] == 2
    assert nodes["a"]["overview"]["orbit_radius"] > 0
    assert nodes["b"]["overview"]["orbit_radius"] == 0


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
    assert "overview_layout" not in data
    assert all("overview" not in matter for matter in data["matters"])


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


def test_api_rejects_forged_host_and_origin_mutation(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})

    status, body = api_request(
        state_path,
        "POST",
        "/api/matters",
        body=json.dumps({"title": "Rebound write", "conditions": ["done"]}),
        headers=lambda port: {
            "Host": f"attacker.test:{port}",
            "Origin": f"http://attacker.test:{port}",
            "Content-Type": "application/json",
        },
    )

    assert status == HTTPStatus.FORBIDDEN
    assert b"invalid API request host" in body
    assert json.loads(state_path.read_text())["matters"] == []


def test_api_rejects_forged_host_and_origin_terminal_start(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})

    status, body = api_request(
        state_path,
        "POST",
        "/api/terminal/sessions",
        body=json.dumps({"rows": 24, "cols": 100}),
        headers=lambda port: {
            "Host": f"attacker.test:{port}",
            "Origin": f"http://attacker.test:{port}",
            "Content-Type": "application/json",
        },
    )

    assert status == HTTPStatus.FORBIDDEN
    assert b"invalid API request host" in body


def test_api_rejects_forged_host_and_origin_terminal_output(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path, {"matters": [], "conditions": {}, "dependencies": []})

    status, body = api_request(
        state_path,
        "GET",
        "/api/terminal/sessions/missing/output?seq=0",
        headers=lambda port: {
            "Host": f"attacker.test:{port}",
            "Origin": f"http://attacker.test:{port}",
        },
    )

    assert status == HTTPStatus.FORBIDDEN
    assert b"invalid API request host" in body


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


def test_graph_payload_rejects_dependency_cycle_with_422(tmp_path):
    state_path = tmp_path / "cyclic.json"
    write_state(
        state_path,
        {
            "matters": ["a", "b"],
            "conditions": {"a": [], "b": []},
            "dependencies": [["a", "b"], ["b", "a"]],
        },
    )

    with pytest.raises(ApiError, match="state dependency graph contains a cycle") as error:
        graph_payload(state_path)

    assert error.value.status == HTTPStatus.UNPROCESSABLE_ENTITY


def test_api_state_returns_422_for_dependency_cycle(tmp_path):
    state_path = tmp_path / "cyclic.json"
    write_state(
        state_path,
        {
            "matters": ["a", "b"],
            "conditions": {"a": [], "b": []},
            "dependencies": [["a", "b"], ["b", "a"]],
        },
    )

    status, body = api_request(state_path, "GET", "/api/state")

    assert status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert json.loads(body) == {"error": "state dependency graph contains a cycle"}


def test_switch_state_path_rejects_cycle_without_replacing_active_path(tmp_path):
    first = tmp_path / "first.json"
    cyclic = tmp_path / "cyclic.json"
    write_state(first, {"matters": ["first"], "conditions": {"first": []}, "dependencies": []})
    write_state(
        cyclic,
        {
            "matters": ["a", "b"],
            "conditions": {"a": [], "b": []},
            "dependencies": [["a", "b"], ["b", "a"]],
        },
    )
    state_paths = StatePathStore(first)

    with pytest.raises(ApiError, match="state dependency graph contains a cycle") as error:
        switch_state_path(state_paths, {"state_path": str(cyclic)})

    assert error.value.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert state_paths.current() == first


def test_switch_state_path_commits_only_after_candidate_payload_builds(
    tmp_path, monkeypatch
):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_state(first, {"matters": ["first"], "conditions": {"first": []}, "dependencies": []})
    write_state(second, {"matters": ["second"], "conditions": {"second": []}, "dependencies": []})
    state_paths = StatePathStore(first)

    def fail_candidate_payload(state_path):
        assert state_path == second
        raise ApiError("candidate payload failed")

    monkeypatch.setattr(web, "graph_payload", fail_candidate_payload)

    with pytest.raises(ApiError, match="candidate payload failed"):
        switch_state_path(state_paths, {"state_path": str(second)})

    assert state_paths.current() == first


def test_mutation_rejects_preexisting_cycle_without_writing(tmp_path):
    state_path = tmp_path / "cyclic.json"
    write_state(
        state_path,
        {
            "matters": ["a", "b"],
            "conditions": {"a": [], "b": []},
            "dependencies": [["a", "b"], ["b", "a"]],
        },
    )
    before = state_path.read_bytes()

    with pytest.raises(ApiError, match="state dependency graph contains a cycle") as error:
        create_matter(state_path, {"title": "must not be saved"})

    assert error.value.status == HTTPStatus.UNPROCESSABLE_ENTITY
    assert state_path.read_bytes() == before


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
        "allow_remote_access": False,
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


def test_api_request_without_token_is_unauthorized(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    status, body = api_request(state_path, "GET", "/api/state", token=None)

    assert status == HTTPStatus.UNAUTHORIZED
    assert json.loads(body) == {"error": "unauthorized"}


def test_terminal_session_cannot_be_created_without_token(tmp_path):
    """The defect itself: a socket to the port used to be a shell."""

    state_path = tmp_path / "matters.json"
    write_state(state_path)
    terminal_manager = TerminalManager(default_workspace=tmp_path)
    handler = partial(
        MattersWebHandler,
        state_paths=StatePathStore(state_path),
        terminal_manager=terminal_manager,
        api_token=VALID_TOKEN,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _headers, body = live_request(
            server.server_port,
            "POST",
            "/api/terminal/sessions",
            body=json.dumps({"rows": 24, "cols": 100}),
            headers={"Content-Type": "application/json"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
        terminal_manager.close_all()

    assert status == HTTPStatus.UNAUTHORIZED
    assert json.loads(body) == {"error": "unauthorized"}
    assert terminal_manager.sessions == {}


def test_api_request_with_wrong_token_is_unauthorized(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    missing_status, missing_body = api_request(
        state_path, "GET", "/api/state", token=None
    )
    wrong_status, wrong_body = api_request(
        state_path, "GET", "/api/state", token="not-the-token"
    )

    assert wrong_status == missing_status == HTTPStatus.UNAUTHORIZED
    # Identical response: a prober must not learn which of the two it sent.
    assert wrong_body == missing_body


def test_api_request_with_wrong_length_token_is_unauthorized(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    status, body = api_request(
        state_path, "GET", "/api/state", token=VALID_TOKEN + "x"
    )

    assert status == HTTPStatus.UNAUTHORIZED
    assert json.loads(body) == {"error": "unauthorized"}


def test_api_request_with_valid_token_succeeds(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    status, body = api_request(state_path, "GET", "/api/state", token=VALID_TOKEN)

    assert status == HTTPStatus.OK
    assert [node["id"] for node in json.loads(body)["nodes"]] == ["a", "b"]


def test_terminal_write_and_output_require_a_token(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    input_status, input_body = api_request(
        state_path,
        "POST",
        "/api/terminal/sessions/whatever/input",
        body=json.dumps({"data": "id\n"}),
        headers={"Content-Type": "application/json"},
        token=None,
    )
    output_status, output_body = api_request(
        state_path, "GET", "/api/terminal/sessions/whatever/output?seq=0", token=None
    )

    assert input_status == output_status == HTTPStatus.UNAUTHORIZED
    assert json.loads(input_body) == json.loads(output_body) == {"error": "unauthorized"}


def test_query_parameter_token_is_rejected_on_api_paths(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    status, body = api_request(
        state_path, "GET", f"/api/state?token={VALID_TOKEN}", token=None
    )

    assert status == HTTPStatus.UNAUTHORIZED
    assert json.loads(body) == {"error": "unauthorized"}


def test_request_api_token_reads_header_and_page_load_query_only():
    assert request_api_token("Bearer abc123", "/api/state") == "abc123"
    assert request_api_token("bearer abc123", "/api/state") == "abc123"
    assert request_api_token(None, "/?token=abc123") == "abc123"
    assert request_api_token(None, "/index.html?token=abc123") == "abc123"
    # Anywhere else the query parameter is ignored, so the token never has to
    # appear in an /api/ URL and cannot leak through Referer.
    assert request_api_token(None, "/api/state?token=abc123") is None
    assert request_api_token(None, "/app.js?token=abc123") is None
    assert request_api_token(None, "/api/state") is None
    assert request_api_token("Basic abc123", "/api/state") is None


def test_api_tokens_match_rejects_absent_and_non_ascii_tokens():
    assert api_tokens_match("abc", "abc") is True
    assert api_tokens_match("abc", "abd") is False
    assert api_tokens_match(None, "abc") is False
    assert api_tokens_match("", "abc") is False
    assert api_tokens_match("abc", None) is False
    # Headers arrive latin-1 decoded; a high-byte token must answer False and
    # not raise out of the request handler.
    assert api_tokens_match("\xff\xfe", "abc") is False


def test_static_assets_load_without_a_token(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    for path, needle in (("/", b"<title>matters graph</title>"), ("/app.js", b"async function api(")):
        status, body = api_request(state_path, "GET", path, token=None)
        assert status == HTTPStatus.OK
        assert needle in body


def test_content_security_policy_is_sent_on_assets_and_api_responses(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    asset_status, asset_headers, _ = api_response(state_path, "GET", "/", token=None)
    api_status, api_headers, _ = api_response(state_path, "GET", "/api/state")

    assert asset_status == HTTPStatus.OK
    assert api_status == HTTPStatus.OK
    for headers in (asset_headers, api_headers):
        policy = headers["Content-Security-Policy"]
        assert "default-src 'none'" in policy
        assert "script-src 'self' https://cdn.jsdelivr.net" in policy
        assert "connect-src 'self'" in policy
        assert "frame-ancestors 'none'" in policy
        assert "object-src 'none'" in policy
        assert "unsafe-eval" not in policy


def test_content_security_policy_is_sent_on_unauthorized_responses(tmp_path):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    status, headers, _ = api_response(state_path, "GET", "/api/state", token=None)

    assert status == HTTPStatus.UNAUTHORIZED
    assert "default-src 'none'" in headers["Content-Security-Policy"]


def test_launch_url_carries_the_token_that_unlocks_the_api(tmp_path, monkeypatch):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    with running_server(monkeypatch, state_path) as url:
        port, token = launch_url_parts(url)

        assert token
        assert token not in ("", "None")
        assert launch_url("127.0.0.1", port, token) == url

        asset_status, _headers, asset_body = live_request(port, "GET", "/")
        assert asset_status == HTTPStatus.OK
        assert b"<title>matters graph</title>" in asset_body

        unauthorized, _headers, _body = live_request(port, "GET", "/api/state")
        assert unauthorized == HTTPStatus.UNAUTHORIZED

        authorized, _headers, body = live_request(
            port, "GET", "/api/state", headers={"Authorization": f"Bearer {token}"}
        )
        assert authorized == HTTPStatus.OK
        assert [node["id"] for node in json.loads(body)["nodes"]] == ["a", "b"]


def test_serve_never_prints_or_writes_the_token_outside_the_launch_url(
    tmp_path, monkeypatch, capsys
):
    state_path = tmp_path / "matters.json"
    write_state(state_path)
    before = state_path.read_bytes()

    with running_server(monkeypatch, state_path) as url:
        port, token = launch_url_parts(url)
        live_request(port, "GET", "/")

    printed = capsys.readouterr()
    lines_with_token = [line for line in printed.out.splitlines() if token in line]

    assert lines_with_token == [f"Serving matters web UI at {url}"]
    assert token not in printed.err
    assert state_path.read_bytes() == before
    written = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and token.encode() in path.read_bytes()
    ]
    assert written == []


def test_serve_mints_a_fresh_token_per_run(tmp_path, monkeypatch):
    state_path = tmp_path / "matters.json"
    write_state(state_path)

    with running_server(monkeypatch, state_path) as first_url:
        _port, first_token = launch_url_parts(first_url)
    with running_server(monkeypatch, state_path) as second_url:
        _port, second_token = launch_url_parts(second_url)

    assert first_token != second_token
    assert re.fullmatch(r"[A-Za-z0-9_-]{32,}", first_token)


def test_is_remote_bind_host_flags_non_loopback_and_wildcard_addresses():
    assert is_remote_bind_host("192.168.1.10") is True
    assert is_remote_bind_host("example.test") is True
    assert is_remote_bind_host("127.0.0.1") is False
    assert is_remote_bind_host("127.0.0.2") is False
    assert is_remote_bind_host("localhost") is False
    assert is_remote_bind_host("::1") is False
    assert is_remote_bind_host("0.0.0.0") is True
    assert is_remote_bind_host("::") is True


@pytest.mark.parametrize("host", ["192.168.1.10", "0.0.0.0", "::"])
def test_cli_refuses_non_loopback_host_without_opt_in(monkeypatch, capsys, host):
    called = {}

    def fake_serve(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("matters.web.serve", fake_serve)

    with pytest.raises(SystemExit) as error:
        main(["web", "--host", host, "--port", "0", "--no-open"])

    assert error.value.code == 2
    assert called == {}
    message = capsys.readouterr().err
    assert "--allow-remote-access" in message
    assert host in message


@pytest.mark.parametrize("host", ["192.168.1.10", "0.0.0.0"])
def test_cli_allows_non_loopback_host_with_opt_in_and_warns(monkeypatch, capsys, host):
    called = {}

    def fake_serve(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("matters.web.serve", fake_serve)

    assert (
        main(
            [
                "web",
                "--host",
                host,
                "--port",
                "0",
                "--no-open",
                "--allow-remote-access",
            ]
        )
        == 0
    )
    assert called["host"] == host
    assert called["allow_remote_access"] is True
    warning = capsys.readouterr().err
    assert "WARNING" in warning
    assert host in warning
    assert "shell" in warning


def test_cli_keeps_loopback_hosts_without_opt_in(monkeypatch, capsys):
    called = {}

    def fake_serve(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("matters.web.serve", fake_serve)

    for host in ("127.0.0.1", "localhost"):
        called.clear()
        assert main(["web", "--host", host, "--port", "0", "--no-open"]) == 0
        assert called["host"] == host
    assert capsys.readouterr().err == ""


def test_web_assets_send_the_token_from_one_place_and_strip_it_from_the_url():
    app = (ASSETS / "app.js").read_text()

    assert "function readApiToken()" in app
    assert 'params.get("token")' in app
    assert "window.history.replaceState(" in app
    assert 'headers.Authorization = `Bearer ${apiToken}`' in app
    # One wrapper, not eighteen call sites.
    assert app.count("Bearer ${apiToken}") == 1
    assert app.count("async function api(path, options = {})") == 1


def test_web_assets_offer_focus_and_deterministic_overview():
    html = (ASSETS / "index.html").read_text()
    app = (ASSETS / "app.js").read_text()
    renderer = (ASSETS / "map-renderer.js").read_text()

    assert '<div id="graph"' in html
    assert '<script type="module" src="app.js?v=cloud-v1"></script>' in html
    assert 'from "./map-renderer.js?v=overview-v4"' in app
    assert '<details class="panel-section disclosure">' in html
    assert "<summary>Create Matter</summary>" in html
    assert "<summary>Dependencies</summary>" in html
    assert 'id="scope-filter"' in html
    assert 'id="overview-graph"' in html
    assert 'id="show-overview"' in html
    assert 'id="back-overview"' in html
    assert 'id="search-results"' in html
    assert 'id="view-live-region"' in html
    assert '<option value="attention">Attention</option>' in html
    assert '<option value="universe">Universe</option>' in html
    assert '<option value="all">All graph</option>' in html
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
    assert "cytoscape@3.34.0" in app
    assert "cytoscape-dagre@4.0.0" in app
    assert 'import("./cloud-renderer.js?v=cloud-v1")' in app
    assert 'id="show-focus"' in html
    assert 'href="styles.css?v=cloud-v1"' in html
    assert "[hidden]" in (ASSETS / "styles.css").read_text()
    assert "cytoscape({" in app
    assert "cytoscape.use(dagre)" in app
    assert "createOverviewRenderer" in app
    assert "LARGE_GRAPH_THRESHOLD" in renderer
    assert 'view: "focus"' in app
    assert "function completeAncestorIds(id)" in app
    assert "filterActive: !derivedActive && overviewFiltersActive()" in app
    assert "__no_matches__" not in app
    assert "function focusHereFromOverview()" in app
    assert "function returnToOverview()" in app
    assert "Overview unavailable" in app
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in app
    assert "slice(0, 50)" in app
    assert 'canvas.setAttribute("role", "img")' in renderer
    assert "ResizeObserver" in renderer
    assert "emphasis.filterActive && !match" in renderer
    assert "function classifyNodeOverlaps()" in renderer
    assert "const OVERLAP_GRID_SIZE = 64" in renderer
    assert "function drawMovingNodes()" in renderer
    assert "const NODE_DEPTH_BANDS = 8" in renderer
    assert "projectedNodes.forEach((projected)" in renderer
    assert "attentionMaxNodes" in app
    assert 'scope: "attention"' in app
    assert "function attentionScopeIds()" in app
    assert "function universeContextIds()" in app
    assert "function focusNode(id)" in app
    assert "function showDerivedScope(kind, matterId)" in app
    assert 'name: "cose"' in app
    assert 'name: "dagre"' in app
    assert 'rankDir: "LR"' in app
    assert 'id="zoom-in"' in html
    assert 'id="zoom-out"' in html
    assert "function zoomGraph(factor)" in app
    assert '"target-arrow-shape": "triangle"' in app
    assert "function runGraphLayout()" in app
    assert "async function responsePayload(response)" in app
    assert "function switchGraphStateErrorMessage(error)" in app
    assert "Restart the matters web server" in app
    assert "webgl-fallback" not in html


@pytest.mark.parametrize("method, endpoint, payload", [
    ("POST", "/api/matters", {"title": "new matter"}),
    ("POST", "/api/dependencies", {"source": "b", "target": "a"}),
    ("DELETE", "/api/dependencies", {"source": "a", "target": "b"}),
    ("PATCH", "/api/matters/b/conditions", {"action": "toggle", "index": 0}),
    ("POST", "/api/command", {"text": "create new matter (done)"}),
    ("POST", "/api/command", {"text": "universe"}),
])
def test_stale_tab_cannot_edit_or_query_another_graph(
    tmp_path, monkeypatch, method, endpoint, payload
):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_state(first)
    write_state(second)
    before = {path: path.read_bytes() for path in (first, second)}
    with running_server(monkeypatch, first) as url:
        port, token = launch_url_parts(url)
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        status, _, body = live_request(port, "GET", "/api/state", headers=headers)
        assert status == 200
        first_id = json.loads(body)["graph_id"]
        status, _, body = live_request(
            port, "POST", "/api/state", json.dumps({"state_path": str(second)}), headers
        )
        assert status == 200
        second_id = json.loads(body)["graph_id"]
        assert first_id != second_id

        status, _, body = live_request(
            port, method, endpoint, json.dumps(payload),
            {**headers, web.GRAPH_ID_HEADER: first_id},
        )
        assert status == 409
        assert "active graph changed" in json.loads(body)["error"]
        assert {path: path.read_bytes() for path in (first, second)} == before

        status, _, body = live_request(
            port, "PATCH", "/api/matters/b/conditions",
            json.dumps({"action": "toggle", "index": 0}),
            {**headers, web.GRAPH_ID_HEADER: second_id},
        )
        assert status == 200
        assert json.loads(body)["graph_id"] == second_id
    assert first.read_bytes() == before[first]
    assert json.loads(second.read_text())["conditions"]["b"][0]["truth"] is True


def test_graph_mutation_requires_explicit_graph_identity(tmp_path, monkeypatch):
    path = tmp_path / "matters.json"
    write_state(path)
    before = path.read_bytes()
    with running_server(monkeypatch, path) as url:
        port, token = launch_url_parts(url)
        status, _, _ = live_request(
            port, "POST", "/api/matters", json.dumps({"title": "unbound edit"}),
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
    assert status == 409
    assert path.read_bytes() == before


def test_graph_request_keeps_its_validated_path_during_a_switch(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_state(first)
    write_state(second)
    store = StatePathStore(first)
    selected = store.for_request(graph_payload(first)["graph_id"])
    store.switch(second)

    result = update_conditions(selected, "b", {"action": "toggle", "index": 0})

    assert result["graph_id"] == graph_payload(first)["graph_id"]
    assert json.loads(first.read_text())["conditions"]["b"][0]["truth"] is True
    assert json.loads(second.read_text())["conditions"]["b"][0]["truth"] is False


@pytest.mark.parametrize("allow_remote_access", [False, True])
def test_wildcard_remote_access_uses_the_actual_socket_destination(
    tmp_path, monkeypatch, allow_remote_access
):
    path = tmp_path / "matters.json"
    write_state(path)
    # Exclude the usual loopback names so a successful HTTP request must
    # have been admitted through its actual socket destination.
    monkeypatch.setattr(web, "api_host_allowlist", lambda *_args: frozenset())
    with running_server(
        monkeypatch, path, host="0.0.0.0", allow_remote_access=allow_remote_access
    ) as url:
        port, token = launch_url_parts(url)
        for host, origin, bearer, expected in [
            ("127.0.0.1", "127.0.0.1", token, 200 if allow_remote_access else 403),
            ("attacker.test", "attacker.test", token, 403),
            ("127.0.0.1", "attacker.test", token, 403),
            ("192.0.2.10", "192.0.2.10", token, 403),
            ("127.0.0.1", "127.0.0.1", "wrong-token", 401),
        ]:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("GET", "/api/state", headers={
                "Host": f"{host}:{port}",
                "Origin": f"http://{origin}:{port}",
                "Authorization": f"Bearer {bearer}",
            })
            response = conn.getresponse()
            response.read()
            conn.close()
            assert response.status == expected
