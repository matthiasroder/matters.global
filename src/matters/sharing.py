"""Helpers for exporting shareable matter states."""

from .engine import as_condition_list, has_dependency_cycle, serialize_condition
from .storage import normalize_dependency_records


PUBLIC = "public"


def public_state(matters, conditions, dependencies, visibility):
    public_matters = {matter for matter in matters if visibility.get(matter) == PUBLIC}
    public_conditions = {
        matter: [
            serialize_condition(condition, index)
            for index, condition in enumerate(
                as_condition_list(conditions.get(matter, ())), start=1
            )
        ]
        for matter in sorted(public_matters)
    }
    public_dependencies = sorted(
        [
            [prerequisite, dependent]
            for prerequisite, dependent in dependencies
            if prerequisite in public_matters and dependent in public_matters
        ]
    )

    return {
        "schema_version": 2,
        "matters": sorted(public_matters),
        "conditions": public_conditions,
        "dependencies": public_dependencies,
    }


def merge_public_state(matters, conditions, dependencies, visibility, incoming_state):
    public_matters = {matter for matter in matters if visibility.get(matter) == PUBLIC}
    incoming_matters = set(incoming_state.get("matters", ()))
    non_public = incoming_matters - public_matters
    if non_public:
        raise ValueError(
            "incoming public state contains non-public matters: "
            + ", ".join(sorted(non_public))
        )

    merged_conditions = {
        matter: [
            serialize_condition(condition, index)
            for index, condition in enumerate(
                as_condition_list(matter_conditions), start=1
            )
        ]
        for matter, matter_conditions in conditions.items()
    }
    for matter in incoming_matters:
        merged_conditions[matter] = [
            serialize_condition(condition, index)
            for index, condition in enumerate(
                as_condition_list(incoming_state.get("conditions", {}).get(matter, ())),
                start=1,
            )
        ]

    private_dependencies = {
        (prerequisite, dependent)
        for prerequisite, dependent in dependencies
        if prerequisite not in public_matters or dependent not in public_matters
    }
    incoming_dependencies = normalize_dependency_records(
        incoming_state.get("dependencies", []),
        public_matters,
        context="incoming public state",
    )
    merged_dependencies = private_dependencies | incoming_dependencies
    if has_dependency_cycle(merged_dependencies):
        raise ValueError("incoming public state would create a dependency cycle")

    return {
        "schema_version": 2,
        "matters": sorted(matters),
        "conditions": {
            matter: merged_conditions[matter] for matter in sorted(merged_conditions)
        },
        "dependencies": sorted([list(dependency) for dependency in merged_dependencies]),
    }
