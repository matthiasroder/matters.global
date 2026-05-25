"""Pure matters graph operations.

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


def resolved(matter, conditions, dependencies, seen=None):
    if seen is None:
        seen = set()
    if matter in seen:
        raise ValueError("dependency cycle")
    return (
        all(truth(condition) for condition in conditions.get(matter, ()))
        and all(
            resolved(prerequisite, conditions, dependencies, seen | {matter})
            for prerequisite in prerequisites(matter, dependencies)
        )
    )


def unresolved(matter, conditions, dependencies):
    return not resolved(matter, conditions, dependencies)


def universe(matters, conditions, dependencies):
    return {
        matter
        for matter in matters
        if unresolved(matter, conditions, dependencies)
        and all(
            resolved(prerequisite, conditions, dependencies)
            for prerequisite in prerequisites(matter, dependencies)
        )
    }


def frontier(root, conditions, dependencies):
    return {
        matter
        for matter in dependents(root, dependencies)
        if unresolved(matter, conditions, dependencies)
        and all(
            resolved(prerequisite, conditions, dependencies)
            for prerequisite in prerequisites(matter, dependencies)
        )
    }


def descendants(root, dependencies):
    out = set()
    stack = list(dependents(root, dependencies))
    while stack:
        matter = stack.pop()
        if matter not in out:
            out.add(matter)
            stack += list(dependents(matter, dependencies))
    return out


def horizon(root, conditions, dependencies):
    root_descendants = descendants(root, dependencies)
    return {
        matter
        for matter in root_descendants
        if unresolved(matter, conditions, dependencies)
        and not any(
            dependent in root_descendants
            and unresolved(dependent, conditions, dependencies)
            for dependent in dependents(matter, dependencies)
        )
    }
