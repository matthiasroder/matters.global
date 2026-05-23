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
#   JSON stores only the primitives: matters, condition truth values, and
#   dependencies. Everything else is computed after loading.


import json


def load_state(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"matters": [], "conditions": {}, "dependencies": []}

    return (
        set(data["matters"]),
        data["conditions"],
        {tuple(d) for d in data["dependencies"]},
    )


def save_state(path, matters, conditions, dependencies):
    data = {
        "matters": sorted(matters),
        "conditions": {
            m: [truth(c) for c in cs]
            for m, cs in conditions.items()
        },
        "dependencies": sorted([list(d) for d in dependencies]),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def truth(c):
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
