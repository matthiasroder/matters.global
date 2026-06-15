import pytest

from matters import merge_public_state, public_state


def test_public_state_exports_only_public_matters_and_edges():
    state = public_state(
        {"public_goal", "private_goal", "public_child"},
        {
            "public_goal": [{"label": "ready", "truth": True}],
            "private_goal": [{"label": "secret", "truth": False}],
            "public_child": [{"label": "child ready", "truth": False}],
        },
        {
            ("public_goal", "public_child"),
            ("private_goal", "public_child"),
        },
        {
            "public_goal": "public",
            "private_goal": "private",
            "public_child": "public",
        },
    )

    assert state == {
        "schema_version": 2,
        "matters": ["public_child", "public_goal"],
        "conditions": {
            "public_child": [{"label": "child ready", "truth": False}],
            "public_goal": [{"label": "ready", "truth": True}],
        },
        "dependencies": [["public_goal", "public_child"]],
    }


def test_merge_public_state_updates_public_conditions_and_preserves_private_state():
    state = merge_public_state(
        {"public_goal", "private_goal"},
        {
            "public_goal": [{"label": "ready", "truth": False}],
            "private_goal": [{"label": "secret", "truth": False}],
        },
        {("private_goal", "public_goal")},
        {"public_goal": "public", "private_goal": "private"},
        {
            "matters": ["public_goal"],
            "conditions": {"public_goal": [{"label": "ready", "truth": True}]},
            "dependencies": [],
        },
    )

    assert state["conditions"] == {
        "private_goal": [{"label": "secret", "truth": False}],
        "public_goal": [{"label": "ready", "truth": True}],
    }
    assert state["dependencies"] == [["private_goal", "public_goal"]]


def test_merge_public_state_rejects_non_public_matter_ids():
    with pytest.raises(ValueError, match="non-public matters: private_goal"):
        merge_public_state(
            {"public_goal", "private_goal"},
            {"public_goal": [], "private_goal": []},
            set(),
            {"public_goal": "public", "private_goal": "private"},
            {
                "matters": ["private_goal"],
                "conditions": {"private_goal": []},
                "dependencies": [],
            },
        )


def test_merge_public_state_rejects_malformed_dependency():
    with pytest.raises(ValueError, match="dependency 0 must have two endpoints"):
        merge_public_state(
            {"public_goal"},
            {"public_goal": []},
            set(),
            {"public_goal": "public"},
            {
                "matters": ["public_goal"],
                "conditions": {"public_goal": []},
                "dependencies": [["public_goal"]],
            },
        )


def test_merge_public_state_rejects_private_dependency_endpoint():
    with pytest.raises(ValueError, match="unknown target: private_goal"):
        merge_public_state(
            {"public_goal", "private_goal"},
            {"public_goal": [], "private_goal": []},
            set(),
            {"public_goal": "public", "private_goal": "private"},
            {
                "matters": ["public_goal"],
                "conditions": {"public_goal": []},
                "dependencies": [["public_goal", "private_goal"]],
            },
        )


def test_merge_public_state_rejects_dependency_cycle():
    with pytest.raises(ValueError, match="dependency cycle"):
        merge_public_state(
            {"public_goal", "public_child"},
            {
                "public_goal": [{"label": "ready", "truth": True}],
                "public_child": [{"label": "child ready", "truth": True}],
            },
            set(),
            {"public_goal": "public", "public_child": "public"},
            {
                "matters": ["public_goal", "public_child"],
                "conditions": {
                    "public_goal": [{"label": "ready", "truth": True}],
                    "public_child": [{"label": "child ready", "truth": True}],
                },
                "dependencies": [
                    ["public_goal", "public_child"],
                    ["public_child", "public_goal"],
                ],
            },
        )
