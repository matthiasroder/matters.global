from matters import format_unlock_report, unlock_report


def test_unlock_report_prioritizes_actionable_matters_by_downstream_impact():
    matters = {"root", "quick", "blocked", "later", "last"}
    conditions = {
        "root": [{"label": "artifact is drafted", "truth": False}],
        "quick": [{"label": "decision is confirmed", "truth": False}],
        "blocked": [{"label": "blocked work is done", "truth": False}],
        "later": [{"label": "later work is done", "truth": False}],
        "last": [{"label": "last work is done", "truth": False}],
    }
    dependencies = {("root", "later"), ("later", "last"), ("blocked", "quick")}

    report = unlock_report(matters, conditions, dependencies)

    assert report["universe"] == ["blocked", "root"]
    assert [item["matter"] for item in report["items"]] == ["root", "blocked"]
    assert report["items"][0]["impact"] == 2
    assert report["items"][0]["actions"] == [
        {
            "mode": "agent_can_start",
            "condition": "artifact is drafted",
            "action": (
                "Draft or implement the next concrete artifact for root: "
                "artifact is drafted."
            ),
        }
    ]
    assert report["blocked"] == ["last", "later", "quick"]


def test_unlock_report_marks_confirmation_actions_as_human_input():
    report = unlock_report(
        {"proposal"},
        {"proposal": [{"label": "client has confirmed scope", "truth": False}]},
        set(),
    )

    assert report["items"][0]["actions"][0]["mode"] == "needs_human_input"


def test_format_unlock_report_is_short_and_readable():
    report = unlock_report(
        {"proposal"},
        {"proposal": [{"label": "offer is drafted", "truth": False}]},
        set(),
    )

    assert format_unlock_report(report).splitlines() == [
        "Actionable matters",
        "- proposal (downstream impact: 0)",
        "  - [agent_can_start] Draft or implement the next concrete artifact for "
        "proposal: offer is drafted.",
        "",
        "Blocked matters",
        "- none",
    ]
