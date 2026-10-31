"""Exercise the actual skill helper across independent processes."""

import json
from pathlib import Path
import selectors
import subprocess
import sys

import pytest


HELPER = Path(__file__).resolve().parents[1] / "skills/matters-triage/scripts/coordinator_lock.py"


def command(state, run_id="test-run"):
    return [sys.executable, str(HELPER), "--state", str(state), "--run-id", run_id]


def acquire(state):
    process = subprocess.Popen(command(state), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=5), "coordinator never reported acquisition"
        acquired = json.loads(process.stdout.readline())
        assert acquired["status"] == "acquired"
        assert process.poll() is None
        return process, acquired
    except BaseException:
        process.kill()
        process.communicate(timeout=5)
        raise


@pytest.mark.parametrize("use_symlink", [False, True])
def test_same_graph_excludes_another_coordinator_and_releases_on_eof(tmp_path, use_symlink):
    state = tmp_path / "graph.json"
    state.write_text("unchanged graph bytes")
    alias = tmp_path / "alias.json"
    alias.symlink_to(state)
    first, acquired = acquire(state)
    lock = Path(acquired["lock_path"])
    inode = lock.stat().st_ino
    try:
        rejected = subprocess.run(command(alias if use_symlink else state, "second"),
                                  input="", capture_output=True, text=True, timeout=5)
        assert rejected.returncode != 0
        assert "another local triage coordinator" in rejected.stderr
        assert not rejected.stdout
    finally:
        output, errors = first.communicate(timeout=5)
    assert first.returncode == 0, errors
    assert json.loads(output)["status"] == "released"
    second, _ = acquire(alias)
    second.communicate(timeout=5)
    assert second.returncode == 0
    assert lock.stat().st_ino == inode
    assert state.read_text() == "unchanged graph bytes"


def test_process_death_releases_ownership_without_removing_sidecar(tmp_path):
    state = tmp_path / "graph.json"
    state.touch()
    first, acquired = acquire(state)
    first.kill()
    first.communicate(timeout=5)
    second, _ = acquire(state)
    second.communicate(timeout=5)
    assert second.returncode == 0
    assert Path(acquired["lock_path"]).is_file()


def test_different_graphs_allow_independent_coordinators(tmp_path):
    states = [tmp_path / "a.json", tmp_path / "b.json"]
    for state in states:
        state.touch()
    first, _ = acquire(states[0])
    try:
        second, _ = acquire(states[1])
        second.communicate(timeout=5)
        assert second.returncode == 0
    finally:
        first.communicate(timeout=5)


def test_missing_graph_fails_without_creating_files(tmp_path):
    state = tmp_path / "missing.json"
    rejected = subprocess.run(command(state), input="", capture_output=True,
                              text=True, timeout=5)
    assert rejected.returncode != 0
    assert not rejected.stdout
    assert list(tmp_path.iterdir()) == []
