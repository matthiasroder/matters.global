"""Derived reports for deciding what matters can move next."""

from .engine import truth
from .graph_index import GraphIndex


HUMAN_INPUT_MARKERS = (
    "confirmed",
    "emailed",
    "invited",
    "paid",
    "payment",
    "published",
    "sent",
    "signed",
    "won",
    "lost",
)


def false_condition_labels(matter, conditions):
    return [
        condition.get("label", "Unlabeled condition")
        for condition in conditions.get(matter, ())
        if not truth(condition)
    ]


def downstream_impact(matter, index):
    """Return how many matters wait on ``matter``, directly or not.

    Takes the index rather than an edge list. The unindexed fallback that
    used to live here rescanned the edges per matter and, being a second
    implementation, could disagree with the index on a graph containing a
    cycle -- where it answered instead of refusing. There is one traversal
    now, and it is :class:`~matters.graph_index.GraphIndex`'s.
    """

    return index.downstream_impact[matter]


def propose_action(matter, condition_label):
    lower_label = condition_label.lower()
    needs_human_input = "unless explicitly confirmed" not in lower_label and any(
        marker in lower_label for marker in HUMAN_INPUT_MARKERS
    )
    mode = "needs_human_input" if needs_human_input else "agent_can_start"

    if needs_human_input:
        action = (
            f"Prepare the smallest request or draft needed for a human to verify: "
            f"{condition_label}."
        )
    else:
        action = (
            f"Draft or implement the next concrete artifact for {matter}: "
            f"{condition_label}."
        )

    return {"mode": mode, "condition": condition_label, "action": action}


def unlock_items(matters, conditions, dependencies, index=None):
    index = index or GraphIndex(matters, conditions, dependencies)
    actionable = index.universe
    items = []

    for matter in actionable:
        false_conditions = false_condition_labels(matter, conditions)
        items.append(
            {
                "matter": matter,
                "impact": downstream_impact(matter, index),
                "false_conditions": false_conditions,
                "actions": [
                    propose_action(matter, condition) for condition in false_conditions
                ],
            }
        )

    return sorted(items, key=lambda item: (-item["impact"], item["matter"]))


def unlock_report(matters, conditions, dependencies, index=None):
    index = index or GraphIndex(matters, conditions, dependencies)
    return {
        "universe": sorted(index.universe),
        "items": unlock_items(
            matters, conditions, dependencies, index=index
        ),
        "blocked": sorted(index.blocked),
    }


def format_unlock_report(report):
    lines = ["Actionable matters"]
    if report["items"]:
        for item in report["items"]:
            lines.append(f"- {item['matter']} (downstream impact: {item['impact']})")
            for action in item["actions"]:
                lines.append(
                    f"  - [{action['mode']}] {action['action']}"
                )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Blocked matters")
    if report["blocked"]:
        lines.extend(f"- {matter}" for matter in report["blocked"])
    else:
        lines.append("- none")

    return "\n".join(lines)
