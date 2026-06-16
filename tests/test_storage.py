import json

import pytest

from matters import load_state, resolve_state_path, save_state
import matters.storage as storage


def test_load_state_normalizes_legacy_conditions(tmp_path):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": {"legacy condition": True}},
                "dependencies": [],
            }
        )
    )

    matters, conditions, dependencies = load_state(state_path)

    assert matters == {"a"}
    assert conditions == {"a": [{"label": "legacy condition", "truth": True}]}
    assert dependencies == set()


def test_save_state_creates_parent_directory(tmp_path):
    state_path = tmp_path / "nested" / "matters.json"

    save_state(
        {"a"},
        {"a": [{"label": "done", "truth": False}]},
        set(),
        path=state_path,
    )

    assert json.loads(state_path.read_text()) == {
        "schema_version": 2,
        "matters": ["a"],
        "conditions": {"a": [{"label": "done", "truth": False}]},
        "dependencies": [],
    }


def test_resolve_state_path_uses_default_state_with_project_state(tmp_path, monkeypatch):
    project_state = tmp_path / ".matters" / "matters.json"
    project_state.parent.mkdir()
    project_state.write_text("{}")
    default_state = tmp_path / "personal" / "matters.json"
    monkeypatch.delenv("MATTERS_STATE", raising=False)
    monkeypatch.setattr(storage, "DEFAULT_STATE_PATH", default_state)

    assert resolve_state_path(cwd=tmp_path) == default_state


def test_resolve_state_path_prefers_env_over_default(tmp_path, monkeypatch):
    default_state = tmp_path / "personal" / "matters.json"
    env_state = tmp_path / "env" / "matters.json"
    monkeypatch.setenv("MATTERS_STATE", str(env_state))
    monkeypatch.setattr(storage, "DEFAULT_STATE_PATH", default_state)

    assert resolve_state_path(cwd=tmp_path) == env_state


def test_resolve_state_path_prefers_explicit_path_over_env(tmp_path, monkeypatch):
    explicit_state = tmp_path / "explicit" / "matters.json"
    env_state = tmp_path / "env" / "matters.json"
    monkeypatch.setenv("MATTERS_STATE", str(env_state))

    assert resolve_state_path(explicit_state, cwd=tmp_path) == explicit_state


def test_load_state_rejects_unknown_dependency_endpoint(tmp_path):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": [{"label": "done", "truth": False}]},
                "dependencies": [["missing", "a"]],
            }
        )
    )

    with pytest.raises(ValueError, match="unknown source: missing"):
        load_state(state_path)


def test_load_state_rejects_malformed_dependency(tmp_path):
    state_path = tmp_path / "matters.json"
    state_path.write_text(
        json.dumps(
            {
                "matters": ["a"],
                "conditions": {"a": []},
                "dependencies": [["a"]],
            }
        )
    )

    with pytest.raises(ValueError, match="must have two endpoints"):
        load_state(state_path)
