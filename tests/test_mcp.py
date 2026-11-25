"""Adapter checks complement the real stdio integration tests."""

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from matters.cli import main
from matters.mcp_server import configured_graphs, create_mcp_server


EXAMPLE = Path(__file__).parents[1] / "examples" / "matters.example.json"


def example_copy(tmp_path):
    path = tmp_path / "example.json"
    path.write_bytes(EXAMPLE.read_bytes())
    return path


def call(server, name, arguments):
    return asyncio.run(server.call_tool(name, arguments))


def test_default_graph_identity_matches_existing_browser_and_canonical_path(tmp_path):
    from matters.web import graph_identity

    state = example_copy(tmp_path)
    alias = tmp_path / "alias.json"
    alias.symlink_to(state)

    graphs = configured_graphs(alias)

    assert graphs == {graph_identity(state): state.resolve()}
    assert graph_identity(state) == hashlib.sha256(str(state.resolve()).encode()).hexdigest()


def test_explicit_graphs_are_a_fixed_allowlist(tmp_path):
    state = example_copy(tmp_path)
    other = tmp_path / "other.json"
    other.write_bytes(EXAMPLE.read_bytes())

    graphs = configured_graphs(graph_specs=[f"personal={state}", f"project={other}"])

    assert graphs == {"personal": state.resolve(), "project": other.resolve()}


@pytest.mark.parametrize("kwargs", [
    {"graph_specs": ["missing-separator"]},
    {"graph_specs": ["project="]},
    {"graph_specs": ["../escape=/tmp/graph.json"]},
    {"graph_id": ""},
    {"graph_id": "space name"},
])
def test_invalid_startup_graph_configuration_is_rejected(tmp_path, kwargs):
    state = example_copy(tmp_path)
    with pytest.raises(ValueError):
        if "graph_specs" in kwargs:
            configured_graphs(**kwargs)
        else:
            configured_graphs(state, **kwargs)


def test_startup_and_service_share_alias_validation(tmp_path):
    state = example_copy(tmp_path)
    server = create_mcp_server(configured_graphs(state, graph_id="work.project-1"))

    result = call(server, "list_graphs", {})

    assert result.structuredContent["graphs"][0]["graph_id"] == "work.project-1"
    for invalid in ("_starts_with_underscore", "x" * 129, "path/name"):
        with pytest.raises(ValueError):
            configured_graphs(state, graph_id=invalid)


def test_duplicate_ids_and_missing_files_do_not_create_graphs(tmp_path):
    state = example_copy(tmp_path)
    missing = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="duplicate graph ID"):
        configured_graphs(graph_specs=[f"same={state}", f"same={state}"])
    with pytest.raises(ValueError, match="existing file"):
        configured_graphs(missing)

    assert not missing.exists()


@pytest.mark.parametrize("before_command", [False, True])
def test_cli_mcp_passes_explicit_graph_and_read_only_without_stdout(tmp_path, monkeypatch, capsys, before_command):
    from matters import mcp_server

    state = example_copy(tmp_path)
    invoked = []
    monkeypatch.setattr(mcp_server, "run_mcp", lambda graphs, **options: invoked.append((graphs, options)))
    args = ["mcp", "--graph-id", "project", "--read-only"]
    if before_command:
        args = ["--state", str(state), *args]
    else:
        args += ["--state", str(state)]

    assert main(args) == 0

    assert invoked == [({"project": state.resolve()}, {"read_only": True})]
    assert capsys.readouterr().out == ""


def test_cli_mcp_named_graphs_reject_conflicting_selection(tmp_path, capsys):
    state = example_copy(tmp_path)

    with pytest.raises(SystemExit) as error:
        main(["mcp", "--graph", f"project={state}", "--state", str(state)])

    assert error.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_core_cli_works_without_optional_sdk_and_mcp_has_install_hint(tmp_path):
    state = example_copy(tmp_path)
    script = """
import builtins, sys
original = builtins.__import__
def no_mcp(name, *args, **kwargs):
    if name == 'mcp' or name.startswith('mcp.'):
        raise ModuleNotFoundError("No module named 'mcp'", name='mcp')
    return original(name, *args, **kwargs)
builtins.__import__ = no_mcp
from matters.cli import main
assert main(['universe', '--state', sys.argv[1]]) == 0
try:
    main(['mcp', '--state', sys.argv[1]])
except SystemExit as error:
    assert error.code == 2
else:
    raise AssertionError('Missing MCP dependency was not reported')
"""
    result = subprocess.run([sys.executable, "-c", script, str(state)], text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ship_first_version\n"
    assert "install matters-global[mcp]" in result.stderr
    assert "Traceback" not in result.stderr


def test_tools_advertise_closed_schemas_versioned_outputs_and_annotations(tmp_path):
    server = create_mcp_server({"project": example_copy(tmp_path)})
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    assert set(tools) == {
        "list_graphs", "list_matters", "get_matter", "universe", "frontier", "horizon", "unlock",
        "create_matter", "set_condition", "add_condition", "edit_condition", "delete_condition",
        "delete_matter", "add_dependency", "remove_dependency", "export_graph",
    }
    for name, tool in tools.items():
        assert tool.inputSchema["additionalProperties"] is False
        assert tool.outputSchema["type"] == "object"
        assert "schema_version" in tool.outputSchema["required"]
        assert tool.annotations.openWorldHint is False
        if name != "list_graphs":
            assert "graph_id" in tool.inputSchema["required"]
    assert tools["set_condition"].inputSchema["properties"]["truth"]["type"] == "boolean"
    assert tools["set_condition"].inputSchema["properties"]["index"]["type"] == "integer"
    assert "expected_revision" in tools["set_condition"].inputSchema["required"]
    assert tools["get_matter"].annotations.readOnlyHint is True
    assert tools["delete_matter"].annotations.destructiveHint is True
    assert tools["remove_dependency"].annotations.destructiveHint is True


@pytest.mark.parametrize("name,arguments", [
    ("set_condition", {"graph_id": "project", "matter_id": "ship_first_version", "index": 0, "truth": "false", "expected_revision": "old"}),
    ("set_condition", {"graph_id": "project", "matter_id": "ship_first_version", "index": True, "truth": True, "expected_revision": "old"}),
    ("set_condition", {"graph_id": "project", "matter_id": "ship_first_version", "index": 0.0, "truth": True, "expected_revision": "old"}),
    ("create_matter", {"graph_id": "project", "matter_id": "new", "conditions": '[{"label":"Done","truth":false}]', "expected_revision": "old"}),
    ("universe", {"graph_id": "project", "path": "/tmp/other.json"}),
    ("universe", {}),
])
def test_schema_errors_are_structured_and_do_not_touch_the_file(tmp_path, name, arguments):
    state = example_copy(tmp_path)
    original = state.read_bytes()
    server = create_mcp_server({"project": state})

    result = call(server, name, arguments)

    assert result.isError is True
    assert result.structuredContent["schema_version"] == 1
    assert result.structuredContent["error"]["code"] == "invalid"
    assert state.read_bytes() == original


def test_unknown_tools_and_unconfigured_graphs_have_semantic_errors(tmp_path):
    server = create_mcp_server({"project": example_copy(tmp_path)})

    unknown = call(server, "run_shell", {"command": "whoami"})
    forbidden = call(server, "universe", {"graph_id": "../../private"})

    assert unknown.isError and unknown.structuredContent["error"]["code"] == "not_found"
    assert forbidden.isError and forbidden.structuredContent["error"]["code"] == "forbidden"
    assert "private" not in forbidden.content[0].text


def test_export_import_cli_round_trip_and_refuse_overwrite(tmp_path, capsys):
    pytest.importorskip("matters.portability")
    state = example_copy(tmp_path)
    output = tmp_path / "bundle.json"
    destination = tmp_path / "imported.json"
    original = state.read_bytes()

    assert main(["export", "--state", str(state), "--output", str(output)]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text())["format"] == "matters-graph"
    assert main(["import", str(output), "--state", str(destination)]) == 0
    assert "imported matters graph" in capsys.readouterr().out
    assert json.loads(destination.read_text()) == json.loads(state.read_text())
    assert state.read_bytes() == original

    with pytest.raises(SystemExit):
        main(["import", str(output), "--state", str(destination)])
    with pytest.raises(SystemExit):
        main(["export", "--state", str(state), "--output", str(state)])
    assert state.read_bytes() == original
