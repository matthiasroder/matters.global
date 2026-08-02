"""Condition plumbing and the two cycle-safe adjacency reads.

Matter:
    A node of concern. It may be resolved or unresolved.

Condition:
    An atomic truth criterion attached to a matter. A condition is a
    collapsed concern: it is treated as directly true or false at the
    current level of attention.

Dependency:
    A directed relation between matters. A dependency ``(a, b)`` means that
    matter ``a`` must be resolved before matter ``b`` can be resolved.

Resolution:
    A matter is resolved exactly when all of its conditions are true and all
    of its prerequisite matters are resolved.

This module owns condition handling and nothing derived from graph *shape*.
``resolved``, ``unresolved``, ``universe``, ``frontier``, ``horizon`` and
``descendants`` used to live here as a second, recursive implementation of
what :mod:`matters.graph_index` already computed; they now live there only
(D5/D7). ``graph_index`` imports :func:`truth` from here, so this module must
never import ``graph_index`` back (D8) -- which is why those functions moved
instead of staying here as wrappers.

:func:`prerequisites` and :func:`dependents` stay. They are pure set
comprehensions over the edge set: they never traverse, so they answer on a
state file that contains a cycle. That is exactly why ``rules.describe_matter``
can serve ``matters show`` on a broken file, which is what keeps such a file
repairable.
"""


def create_condition(label, truth_value=False):
    return {"label": str(label), "truth": truth(truth_value)}


def as_condition_list(conditions):
    if conditions is None:
        return []
    if isinstance(conditions, dict) and (
        "label" in conditions or "truth" in conditions or "value" in conditions
    ):
        return [conditions]
    if isinstance(conditions, dict):
        return [
            {"label": label, "truth": value}
            for label, value in conditions.items()
        ]
    return list(conditions)


def normalize_conditions(conditions):
    return {
        matter: [
            serialize_condition(condition, index)
            for index, condition in enumerate(as_condition_list(matter_conditions), start=1)
        ]
        for matter, matter_conditions in conditions.items()
    }


def condition_label(condition, index=None):
    if isinstance(condition, dict):
        label = condition.get("label") or condition.get("name")
        if label and str(label).strip():
            return str(label).strip()

    if index is None:
        return "Unlabeled legacy condition"
    return f"Unlabeled legacy condition {index}"


def serialize_condition(condition, index=None):
    return {
        "label": condition_label(condition, index),
        "truth": truth(condition),
    }


def truth(condition):
    if isinstance(condition, dict):
        if "truth" in condition:
            return truth(condition["truth"])
        if "value" in condition:
            return truth(condition["value"])
        return False
    return condition() if callable(condition) else bool(condition)


def prerequisites(matter, dependencies):
    return {a for a, b in dependencies if b == matter}


def dependents(matter, dependencies):
    return {b for a, b in dependencies if a == matter}


def has_dependency_cycle(dependencies):
    """Report whether ``dependencies`` contains a cycle. Scheduled to go.

    The second of two cycle detectors left in the codebase, down from three:
    ``rules.has_cycle`` (via :class:`~matters.graph_index.GraphIndex`) is the
    one every other caller uses, and it names the offending cycle instead of
    answering yes/no. This one survives for exactly one caller,
    ``sharing.merge_public_state``, which takes a bare edge list and has no
    matters set to build an index from. Rewriting ``sharing`` is separately
    scheduled work and takes the count to one; until then this is a known
    duplicate, not an oversight. Do not add callers.
    """

    outgoing = {}
    for source, target in dependencies:
        outgoing.setdefault(source, set()).add(target)

    visiting = set()
    visited = set()

    def visit(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for target in outgoing.get(node, ()):
            if visit(target):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in outgoing)
