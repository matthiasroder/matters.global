"""JSON persistence for matters state."""

import json
import os
import tempfile
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

    validate_state_data(data)
    matters = set(data["matters"])
    return (
        matters,
        normalize_conditions(data["conditions"]),
        normalize_dependency_records(data["dependencies"], matters),
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
    validate_state_data(data)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{state_path.name}.",
        suffix=".tmp",
        dir=state_path.parent,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, state_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


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


def validate_state_data(data):
    if not isinstance(data, dict):
        raise ValueError("state must be a JSON object")

    matters = data.get("matters")
    if not isinstance(matters, list):
        raise ValueError("state matters must be a list")
    matter_ids = set()
    for index, matter in enumerate(matters):
        if not isinstance(matter, str) or not matter:
            raise ValueError(f"state matters[{index}] must be a non-empty string")
        matter_ids.add(matter)

    conditions = data.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("state conditions must be an object")
    for matter in conditions:
        if matter not in matter_ids:
            raise ValueError(f"conditions contain unknown matter: {matter}")

    dependencies = data.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("state dependencies must be a list")
    normalize_dependency_records(dependencies, matter_ids)


def normalize_dependency_records(dependencies, matters, context="state"):
    matter_ids = set(matters)
    normalized = set()
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, (list, tuple)) or len(dependency) != 2:
            raise ValueError(f"{context} dependency {index} must have two endpoints")
        source, target = dependency
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"{context} dependency {index} endpoints must be strings")
        if source not in matter_ids:
            raise ValueError(f"{context} dependency {index} has unknown source: {source}")
        if target not in matter_ids:
            raise ValueError(f"{context} dependency {index} has unknown target: {target}")
        normalized.add((source, target))
    return normalized
