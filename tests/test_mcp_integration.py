"""Exercise the local MCP server through its real stdio protocol boundary.

These checks use temporary copies of repository examples. They verify what an
external client can do, including the relationship to the existing CLI and
browser API, rather than calling registered tool functions directly.
"""

import json
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from matters.web import graph_payload


EXAMPLE = Path(__file__).parents[1] / "examples" / "matters.example.json"
READ_TOOLS = {
    "list_graphs", "list_matters", "get_matter", "universe", "frontier",
    "horizon", "unlock", "export_graph",
}
WRITE_TOOLS = {
    "create_matter", "set_condition", "add_condition", "edit_condition",
    "delete_condition", "delete_matter", "add_dependency", "remove_dependency",
}


@asynccontextmanager
async def session_for(tmp_path, *arguments):
    """Launch the same command a local client configures, without user settings."""
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "matters.cli", "mcp", *map(str, arguments)],
        cwd=tmp_path,
        env={"MATTERS_CONFIG": str(tmp_path / "absent-model-config.toml")},
    )
    with tempfile.TemporaryFile(mode="w+") as errors:
        with anyio.fail_after(20):
            async with stdio_client(parameters, errlog=errors) as (reader, writer):
                async with ClientSession(
                    reader, writer, read_timeout_seconds=timedelta(seconds=10)
                ) as client:
                    await client.initialize()
                    yield client


def copied_example(tmp_path, name="project.json"):
    path = tmp_path / name
    path.write_bytes(EXAMPLE.read_bytes())
    return path


def payload(result):
    assert not result.isError, result.content
    assert isinstance(result.structuredContent, dict), result.content
    assert result.structuredContent["schema_version"] == 1
    return result.structuredContent


def error_payload(result, code=None):
    assert result.isError, result
    assert isinstance(result.structuredContent, dict), result.content
    assert result.structuredContent["schema_version"] == 1
    error = result.structuredContent["error"]
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error["code"], str) and error["code"]
    if code is not None:
        assert error["code"] == code
    return error


def cli_json(state, *arguments):
    result = subprocess.run(
        [sys.executable, "-m", "matters.cli", *arguments, "--state", str(state)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(result.stdout)


def test_stdio_discovery_and_write_match_cli_and_browser(tmp_path):
    state = copied_example(tmp_path)

    async def exercise():
        async with session_for(tmp_path, "--state", state, "--graph-id", "project") as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert set(tools) == READ_TOOLS | WRITE_TOOLS
            for name, tool in tools.items():
                required = tool.inputSchema.get("required", [])
                assert tool.annotations is not None
                assert tool.annotations.readOnlyHint is (name in READ_TOOLS)
                assert tool.annotations.openWorldHint is False
                if name != "list_graphs":
                    assert "graph_id" in required
                if name in WRITE_TOOLS:
                    assert "expected_revision" in required
                assert not {"path", "state_path", "shell", "command"}.intersection(
                    tool.inputSchema.get("properties", {})
                )

            catalogue = payload(await client.call_tool("list_graphs", {}))
            assert len(catalogue["graphs"]) == 1
            assert catalogue["graphs"][0]["graph_id"] == "project"
            assert catalogue["graphs"][0]["read_only"] is False
            current = payload(await client.call_tool("universe", {"graph_id": "project"}))
            assert current["matters"] == ["ship_first_version"]
            assert current["revision"] == catalogue["graphs"][0]["revision"]
            for view in ("frontier", "horizon"):
                result = payload(await client.call_tool(view, {
                    "graph_id": "project", "matter_id": "define_project_goal",
                }))
                assert result["matters"] == ["ship_first_version"]
                assert result["revision"] == current["revision"]
            report = payload(await client.call_tool("unlock", {"graph_id": "project"}))
            assert report["universe"] == ["ship_first_version"]
            page = payload(await client.call_tool("list_matters", {
                "graph_id": "project", "offset": 1, "limit": 1,
            }))
            assert page["matters"] == ["ship_first_version"]
            assert page["total"] == 2

            observed = payload(await client.call_tool("get_matter", {
                "graph_id": "project", "matter_id": "ship_first_version",
            }))
            assert observed["conditions"] == [{
                "index": 0,
                "label": "First version is usable by one real user",
                "truth": False,
            }]
            changed = payload(await client.call_tool("set_condition", {
                "graph_id": "project", "matter_id": "ship_first_version",
                "index": 0, "truth": True, "expected_revision": observed["revision"],
            }))
            assert changed["graph_id"] == "project"
            assert changed["revision"] != observed["revision"]

            cli = cli_json(state, "show", "ship_first_version", "--json")
            browser = graph_payload(state)
            node = next(node for node in browser["nodes"] if node["id"] == "ship_first_version")
            assert cli["conditions"][0]["truth"] is True
            assert node["conditions"][0]["truth"] is True
            assert node["resolved"] is True
            assert cli["revision"] == browser["revision"] == changed["revision"]
            assert payload(await client.call_tool("universe", {"graph_id": "project"}))["matters"] == []

    anyio.run(exercise)


def test_stdio_multiple_graphs_and_unconfigured_paths_are_isolated(tmp_path):
    personal = copied_example(tmp_path, "personal.json")
    project = copied_example(tmp_path, "project.json")
    unconfigured = copied_example(tmp_path, "not-configured.json")
    personal_before, unconfigured_before = personal.read_bytes(), unconfigured.read_bytes()
    project_id = "project.alpha"

    async def exercise():
        async with session_for(
            tmp_path, "--graph", f"personal={personal}", "--graph", f"{project_id}={project}"
        ) as client:
            catalogue = payload(await client.call_tool("list_graphs", {}))
            assert {graph["graph_id"] for graph in catalogue["graphs"]} == {"personal", project_id}
            observed = payload(await client.call_tool("get_matter", {
                "graph_id": project_id, "matter_id": "ship_first_version",
            }))
            arguments = {
                "matter_id": "ship_first_version", "index": 0, "truth": True,
                "expected_revision": observed["revision"],
            }
            for graph_id in ("unconfigured", str(unconfigured), "../not-configured.json"):
                error = error_payload(await client.call_tool("set_condition", {
                    **arguments, "graph_id": graph_id,
                }), "forbidden")
                assert str(unconfigured) not in error["message"]
                assert project.read_bytes() == EXAMPLE.read_bytes()
                assert personal.read_bytes() == personal_before
                assert unconfigured.read_bytes() == unconfigured_before

            payload(await client.call_tool("set_condition", {**arguments, "graph_id": project_id}))
            assert personal.read_bytes() == personal_before
            assert unconfigured.read_bytes() == unconfigured_before
            assert cli_json(project, "show", "ship_first_version", "--json")["conditions"][0]["truth"] is True
            assert cli_json(personal, "show", "ship_first_version", "--json")["conditions"][0]["truth"] is False

    anyio.run(exercise)


def test_stdio_editing_uses_current_revisions_and_requires_deletion_confirmation(tmp_path):
    state = copied_example(tmp_path)

    async def exercise():
        async with session_for(tmp_path, "--state", state, "--graph-id", "project") as client:
            revision = payload(await client.call_tool("universe", {"graph_id": "project"}))["revision"]

            async def change(name, **arguments):
                nonlocal revision
                result = payload(await client.call_tool(name, {
                    "graph_id": "project", "expected_revision": revision, **arguments,
                }))
                revision = result["revision"]
                return result

            await change("create_matter", matter_id="follow_up", conditions=[{
                "label": "Review completed", "truth": False,
            }])
            await change("add_condition", matter_id="follow_up", label="Report attached")
            await change("edit_condition", matter_id="follow_up", index=1, label="Evidence attached")
            observed = payload(await client.call_tool("get_matter", {
                "graph_id": "project", "matter_id": "follow_up",
            }))
            assert observed["conditions"] == [
                {"index": 0, "label": "Review completed", "truth": False},
                {"index": 1, "label": "Evidence attached", "truth": False},
            ]
            await change("add_dependency", source="follow_up", target="ship_first_version")
            assert payload(await client.call_tool("universe", {"graph_id": "project"}))["matters"] == ["follow_up"]
            await change("remove_dependency", source="follow_up", target="ship_first_version")
            await change("delete_condition", matter_id="follow_up", index=0)

            before_last_deletion = state.read_bytes()
            error_payload(await client.call_tool("delete_condition", {
                "graph_id": "project", "matter_id": "follow_up", "index": 0,
                "expected_revision": revision,
            }), "conflict")
            assert state.read_bytes() == before_last_deletion
            await change("delete_condition", matter_id="follow_up", index=0, confirmed=True)

            before_matter_deletion = state.read_bytes()
            error_payload(await client.call_tool("delete_matter", {
                "graph_id": "project", "matter_id": "follow_up", "expected_revision": revision,
            }), "conflict")
            assert state.read_bytes() == before_matter_deletion
            await change("delete_matter", matter_id="follow_up", confirmed=True)
            remaining = payload(await client.call_tool("list_matters", {"graph_id": "project"}))
            assert remaining["matters"] == ["define_project_goal", "ship_first_version"]

    anyio.run(exercise)


def test_stdio_stale_condition_position_after_cli_delete_is_rejected(tmp_path):
    state = tmp_path / "project.json"
    state.write_text(json.dumps({
        "schema_version": 2,
        "matters": ["goal"],
        "conditions": {"goal": [
            {"label": "first condition", "truth": False},
            {"label": "second condition", "truth": False},
        ]},
        "dependencies": [],
    }))

    async def exercise():
        async with session_for(tmp_path, "--state", state, "--graph-id", "project") as client:
            observed = payload(await client.call_tool("get_matter", {
                "graph_id": "project", "matter_id": "goal",
            }))
            subprocess.run(
                [sys.executable, "-m", "matters.cli", "delete-condition", "goal", "1", "--state", str(state)],
                check=True, capture_output=True, text=True, timeout=10,
            )
            after_cli = state.read_bytes()
            error_payload(await client.call_tool("set_condition", {
                "graph_id": "project", "matter_id": "goal", "index": 0,
                "truth": True, "expected_revision": observed["revision"],
            }), "conflict")
            assert state.read_bytes() == after_cli
            current = payload(await client.call_tool("get_matter", {
                "graph_id": "project", "matter_id": "goal",
            }))
            assert current["conditions"] == [{"index": 0, "label": "second condition", "truth": False}]
            assert current["revision"] != observed["revision"]

    anyio.run(exercise)


def test_stdio_read_only_and_cycle_rejection_preserve_graph(tmp_path):
    state = copied_example(tmp_path)
    original = state.read_bytes()

    async def exercise():
        async with session_for(
            tmp_path, "--state", state, "--graph-id", "project", "--read-only"
        ) as client:
            catalogue = payload(await client.call_tool("list_graphs", {}))
            assert catalogue["graphs"][0]["read_only"] is True
            revision = catalogue["graphs"][0]["revision"]
            error_payload(await client.call_tool("set_condition", {
                "graph_id": "project", "matter_id": "ship_first_version", "index": 0,
                "truth": True, "expected_revision": revision,
            }), "forbidden")
            assert state.read_bytes() == original

        async with session_for(tmp_path, "--state", state, "--graph-id", "project") as client:
            current = payload(await client.call_tool("universe", {"graph_id": "project"}))
            error = error_payload(await client.call_tool("add_dependency", {
                "graph_id": "project", "source": "ship_first_version",
                "target": "define_project_goal", "expected_revision": current["revision"],
            }))
            assert "cycle" in error["message"].lower()
            assert state.read_bytes() == original

    anyio.run(exercise)


def test_stdio_does_not_coerce_truth_indexes_or_missing_preconditions(tmp_path):
    state = copied_example(tmp_path)
    original = state.read_bytes()

    async def exercise():
        async with session_for(tmp_path, "--state", state, "--graph-id", "project") as client:
            observed = payload(await client.call_tool("get_matter", {
                "graph_id": "project", "matter_id": "ship_first_version",
            }))
            valid = {
                "graph_id": "project", "matter_id": "ship_first_version", "index": 0,
                "truth": True, "expected_revision": observed["revision"],
            }
            invalid = [
                {**valid, "truth": "false"}, {**valid, "truth": 1},
                {**valid, "index": True}, {**valid, "index": "0"},
                {**valid, "index": 0.0}, {**valid, "index": -1},
                {key: value for key, value in valid.items() if key != "graph_id"},
                {key: value for key, value in valid.items() if key != "expected_revision"},
            ]
            for arguments in invalid:
                error_payload(await client.call_tool("set_condition", arguments))
                assert state.read_bytes() == original, arguments
            for conditions in (
                '[{"label":"Review completed","truth":false}]',
                {"label": "Review completed", "truth": False},
                [{"label": "Review completed", "truth": "false"}],
                [{"label": "Review completed", "truth": False, "state_path": "other.json"}],
            ):
                error_payload(await client.call_tool("create_matter", {
                    "graph_id": "project", "matter_id": "follow_up",
                    "conditions": conditions, "expected_revision": observed["revision"],
                }), "invalid")
                assert state.read_bytes() == original
            error_payload(await client.call_tool("set_condition", {
                **valid, "state_path": str(tmp_path / "not-configured.json"),
            }), "invalid")
            assert state.read_bytes() == original

    anyio.run(exercise)


def test_stdio_sources_and_export_stay_with_selected_graph(tmp_path):
    state = copied_example(tmp_path)
    source = {"title": "Repository example evidence", "url": "https://example.org/evidence"}
    state.with_suffix(".sources.json").write_text(json.dumps({
        "schema_version": 1, "matter_sources": {"ship_first_version": [source]},
    }))

    async def exercise():
        async with session_for(tmp_path, "--state", state, "--graph-id", "project") as client:
            matter = payload(await client.call_tool("get_matter", {
                "graph_id": "project", "matter_id": "ship_first_version", "include_sources": True,
            }))
            assert matter["sources"] == [source]
            exported = payload(await client.call_tool("export_graph", {"graph_id": "project"}))
            assert exported["graph_id"] == "project"
            assert exported["revision"] == matter["revision"]
            # The bundle must carry the citation; the exact versioned package
            # shape is checked by the independent portability tests.
            assert "Repository example evidence" in json.dumps(exported)
            assert "https://example.org/evidence" in json.dumps(exported)

    anyio.run(exercise)
