# System definitions:
#
# Matter:
#   A node of concern. It may be resolved or unresolved.
#
# Condition:
#   An atomic truth criterion attached to a matter. A condition is a
#   collapsed concern: it is treated as directly true or false at the
#   current level of attention.
#
# Dependency:
#   A directed relation between matters. A dependency (a, b) means that
#   matter a must be resolved before matter b can be resolved.
#
# Resolution:
#   A matter is resolved exactly when all of its conditions are true and
#   all of its prerequisite matters are resolved.
#
# Universe:
#   All unresolved matters in the whole graph whose prerequisites are
#   resolved. These are globally actionable.
#
# Frontier:
#   Given a matter r, the unresolved direct dependents of r whose
#   prerequisites are resolved. These are the level-1 matters made
#   actionable by r.
#
# Horizon:
#   Given a matter r, the farthest unresolved descendants reachable from r.
#
# Persistent state:
#   JSON stores only the primitives: matters, condition labels and truth
#   values, and dependencies. Everything else is computed after loading.


import json
from pathlib import Path


STATE_PATH = Path(__file__).with_name("matters.json")


def load_state(path=STATE_PATH):
    path = Path(path)
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"matters": [], "conditions": {}, "dependencies": []}

    return (
        set(data["matters"]),
        normalize_conditions(data["conditions"]),
        {tuple(d) for d in data["dependencies"]},
    )


def save_state(*args, path=STATE_PATH):
    matters, conditions, dependencies, path = resolve_save_args(args, path)
    path = Path(path)
    data = {
        "schema_version": 2,
        "matters": sorted(matters),
        "conditions": {
            m: [
                serialize_condition(c, index)
                for index, c in enumerate(as_condition_list(cs), start=1)
            ]
            for m, cs in conditions.items()
        },
        "dependencies": sorted([list(d) for d in dependencies]),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


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


def create_condition(label, truth_value=False):
    return {"label": str(label), "truth": truth(truth_value)}


def as_condition_list(cs):
    if cs is None:
        return []
    if isinstance(cs, dict) and ("label" in cs or "truth" in cs or "value" in cs):
        return [cs]
    if isinstance(cs, dict):
        return [
            {"label": label, "truth": value}
            for label, value in cs.items()
        ]
    return list(cs)


def normalize_conditions(conditions):
    return {
        matter: [
            serialize_condition(c, index)
            for index, c in enumerate(as_condition_list(cs), start=1)
        ]
        for matter, cs in conditions.items()
    }


def condition_label(c, index=None):
    if isinstance(c, dict):
        label = c.get("label") or c.get("name")
        if label and str(label).strip():
            return str(label).strip()

    if index is None:
        return "Unlabeled legacy condition"
    return f"Unlabeled legacy condition {index}"


def serialize_condition(c, index=None):
    return {
        "label": condition_label(c, index),
        "truth": truth(c),
    }


def truth(c):
    if isinstance(c, dict):
        if "truth" in c:
            return truth(c["truth"])
        if "value" in c:
            return truth(c["value"])
        return False
    return c() if callable(c) else bool(c)


def prerequisites(m, dependencies):
    return {a for a, b in dependencies if b == m}


def dependents(m, dependencies):
    return {b for a, b in dependencies if a == m}


def resolved(m, conditions, dependencies, seen=None):
    if seen is None:
        seen = set()
    if m in seen:
        raise ValueError("dependency cycle")
    return (
        all(truth(c) for c in conditions.get(m, ()))
        and all(
            resolved(p, conditions, dependencies, seen | {m})
            for p in prerequisites(m, dependencies)
        )
    )


def unresolved(m, conditions, dependencies):
    return not resolved(m, conditions, dependencies)


def universe(matters, conditions, dependencies):
    return {
        m
        for m in matters
        if unresolved(m, conditions, dependencies)
        and all(
            resolved(p, conditions, dependencies)
            for p in prerequisites(m, dependencies)
        )
    }


def frontier(r, conditions, dependencies):
    return {
        m
        for m in dependents(r, dependencies)
        if unresolved(m, conditions, dependencies)
        and all(
            resolved(p, conditions, dependencies)
            for p in prerequisites(m, dependencies)
        )
    }


def descendants(r, dependencies):
    out = set()
    stack = list(dependents(r, dependencies))
    while stack:
        m = stack.pop()
        if m not in out:
            out.add(m)
            stack += list(dependents(m, dependencies))
    return out


def horizon(r, conditions, dependencies):
    ds = descendants(r, dependencies)
    return {
        m
        for m in ds
        if unresolved(m, conditions, dependencies)
        and not any(
            d in ds and unresolved(d, conditions, dependencies)
            for d in dependents(m, dependencies)
        )
    }
