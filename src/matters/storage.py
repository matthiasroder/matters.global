"""JSON persistence for matters state."""

import json
import os
from pathlib import Path

from .engine import as_condition_list, normalize_conditions, serialize_condition


DEFAULT_STATE_PATH = Path.home() / ".local" / "share" / "matters" / "matters.json"


def resolve_state_path(path=None, cwd=None):
    if path:
        return Path(path).expanduser()

    env_path = os.environ.get("MATTERS_STATE")
    if env_path:
        return Path(env_path).expanduser()

    base = Path(cwd) if cwd is not None else Path.cwd()
    project_state = base / ".matters" / "matters.json"
    if project_state.exists():
        return project_state

    return DEFAULT_STATE_PATH


def load_state(path=None):
    state_path = resolve_state_path(path)
    try:
        with state_path.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"matters": [], "conditions": {}, "dependencies": []}

    return (
        set(data["matters"]),
        normalize_conditions(data["conditions"]),
        {tuple(dependency) for dependency in data["dependencies"]},
    )


def save_state(*args, path=None):
    matters, conditions, dependencies, state_path = resolve_save_args(args, path)
    state_path = resolve_state_path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "schema_version": 2,
        "matters": sorted(matters),
        "conditions": {
            matter: [
                serialize_condition(condition, index)
                for index, condition in enumerate(
                    as_condition_list(matter_conditions), start=1
                )
            ]
            for matter, matter_conditions in conditions.items()
        },
        "dependencies": sorted([list(dependency) for dependency in dependencies]),
    }

    with state_path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def resolve_save_args(args, path):
    if len(args) == 3:
        matters, conditions, dependencies = args
        return matters, conditions, dependencies, path

    if len(args) == 4:
        first, second, third, fourth = args
        if isinstance(first, (str, Path)):
            return second, third, fourth, first
        return first, second, third, fourth

    raise TypeError(
        "save_state expects (matters, conditions, dependencies[, path]) "
        "or legacy (path, matters, conditions, dependencies)"
    )
