"""Derived reports for deciding what matters can move next."""

from .engine import descendants, prerequisites, resolved, truth, universe


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


def downstream_impact(matter, dependencies):
    return len(descendants(matter, dependencies))


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


def unlock_items(matters, conditions, dependencies):
    actionable = universe(matters, conditions, dependencies)
    items = []

    for matter in actionable:
        false_conditions = false_condition_labels(matter, conditions)
        items.append(
            {
                "matter": matter,
                "impact": downstream_impact(matter, dependencies),
                "false_conditions": false_conditions,
                "actions": [
                    propose_action(matter, condition) for condition in false_conditions
                ],
            }
        )

    return sorted(items, key=lambda item: (-item["impact"], item["matter"]))


def unlock_report(matters, conditions, dependencies):
    return {
        "universe": sorted(universe(matters, conditions, dependencies)),
        "items": unlock_items(matters, conditions, dependencies),
        "blocked": sorted(
            matter
            for matter in matters
            if not resolved(matter, conditions, dependencies)
            and any(
                not resolved(prerequisite, conditions, dependencies)
                for prerequisite in prerequisites(matter, dependencies)
            )
        ),
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
