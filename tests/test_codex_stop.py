import io
import json
import shlex
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from matters import codex_stop
from matters.cli import main


@pytest.fixture
def graph(tmp_path, capsys):
    path = tmp_path / "release graph.json"
    assert main([
        "--state", str(path), "create",
        "publish (release is approved) > build (build verified)",
        "publish > notes (checklist written)",
        "unrelated (something else done)",
    ]) == 0
    capsys.readouterr()
    return path


@pytest.fixture
def stop_event(graph, tmp_path, monkeypatch, capsys):
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({
        "target": "publish", "state": str(graph), "limit": 12,
        "continuations": 0, "verified": 0,
    }))
    monkeypatch.setenv(codex_stop.RUN_FILE_ENV, str(run_path))

    def run(**fields):
        event = {"hook_event_name": "Stop", **fields}
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
        capsys.readouterr()
        assert main(["codex-stop"]) == 0
        return json.loads(capsys.readouterr().out)

    return run_path, run


def test_continues_through_prerequisites_and_yields_for_approval(graph, stop_event):
    _, run = stop_event
    original = graph.read_bytes()
    first = run(stop_hook_active=False)
    assert first["decision"] == "block"
    assert "build, condition 1" in first["reason"]
    assert graph.read_bytes() == original

    assert main(["--state", str(graph), "mark", "build", "1", "true"]) == 0
    second = run(stop_hook_active=True)
    assert second["decision"] == "block"
    assert "notes, condition 1" in second["reason"]

    assert main(["--state", str(graph), "mark", "notes", "1", "true"]) == 0
    assert run(stop_hook_active=True) == {
        "systemMessage": "Matters is waiting for publish, condition 1: release is approved",
    }
    assert main(["--state", str(graph), "mark", "publish", "1", "true"]) == 0
    assert run() == {"systemMessage": "Matters target 'publish' is resolved."}


def test_unrelated_progress_does_not_restart_the_loop(graph, stop_event):
    _, run = stop_event
    assert run()["decision"] == "block"
    assert main(["--state", str(graph), "mark", "unrelated", "1", "true"]) == 0
    assert run(stop_hook_active=True) == {
        "systemMessage": "Matters stopped without a newly verified condition. Next: build, condition 1.",
    }


def test_limit_and_reported_blocker_yield_the_turn(graph, stop_event):
    run_path, run = stop_event
    data = json.loads(run_path.read_text())
    run_path.write_text(json.dumps({**data, "limit": 1}))
    assert run(last_assistant_message="Need access.\n[MATTERS_WAITING]\n") == {}
    assert run(permission_mode="plan") == {}
    assert run(last_assistant_message="The [MATTERS_WAITING] marker stays unused.")["decision"] == "block"
    assert main(["--state", str(graph), "mark", "build", "1", "true"]) == 0
    assert run() == {
        "systemMessage": "Matters continuation limit reached. Next: notes, condition 1.",
    }


def test_explicit_external_wait_yields_with_condition(graph, stop_event):
    _, run = stop_event
    assert main(["--state", str(graph), "mark", "notes", "1", "true"]) == 0
    assert main([
        "--state", str(graph), "edit-condition", "build", "1", "[external] CI returns its result",
    ]) == 0
    assert run() == {
        "systemMessage": "Matters is waiting for build, condition 1: [external] CI returns its result",
    }


def test_follow_passes_only_session_configuration_to_codex(graph, monkeypatch):
    monkeypatch.setattr(codex_stop.shutil, "which", lambda _: "/opt/bin/codex")
    launched = []

    def run(command, *, env, check):
        assert command[:2] == ["/opt/bin/codex", "-c"]
        spec = tomllib.loads(command[2])["hooks"]["Stop"][0]["hooks"][0]
        assert shlex.split(spec["command"])[1:] == ["-m", "matters.cli", "codex-stop"]
        assert "publish" in command[-1] and str(graph) in command[-1]
        assert env["MATTERS_STATE"] == str(graph.resolve())
        run_path = Path(env[codex_stop.RUN_FILE_ENV])
        config = json.loads(run_path.read_text())
        assert config["target"] == "publish"
        assert config["state"] == str(graph.resolve())
        assert config["limit"] == 2
        launched.append(run_path)
        return codex_stop.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(codex_stop.subprocess, "run", run)
    assert main(["follow", "publish", "--state", str(graph), "--max-continuations", "2"]) == 0
    assert len(launched) == 1
    assert not launched[0].exists()
