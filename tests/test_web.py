import json
from pathlib import Path

import pytest

from matters.cli import main
from matters.web import (
    ApiError,
    add_dependency,
    create_matter,
    graph_payload,
    remove_dependency,
    run_command,
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
    }


def test_web_assets_use_three_dimensional_canvas():
    html = (ASSETS / "index.html").read_text()
    app = (ASSETS / "app.js").read_text()

    assert '<div id="graph"' in html
    assert '<script type="module" src="app.js"></script>' in html
    assert "3d-force-graph@1.78.0" in app
    assert "ForceGraph3D()(graphElement)" in app
    assert ".enableNodeDrag(false)" in app
    assert "linkDirectionalArrowLength" in app
    assert "d3Force(\"charge\")" in app
    assert "webgl-fallback" in html
