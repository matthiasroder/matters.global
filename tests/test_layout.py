import math

from matters.graph_index import GraphIndex
from matters.layout import build_overview_layout


def layout(matters, dependencies, conditions=None):
    return build_overview_layout(
        GraphIndex(matters, conditions or {}, dependencies)
    )


def test_empty_layout_has_zero_bounds():
    metadata, coordinates = layout(set(), set())

    assert metadata == {
        "version": 2,
        "algorithm": "layered-archipelago-v1",
        "max_depth": 0,
        "island_count": 0,
        "bounds": {
            "min_x": 0.0,
            "max_x": 0.0,
            "min_y": 0.0,
            "max_y": 0.0,
            "min_z": 0.0,
            "max_z": 0.0,
        },
    }
    assert coordinates == {}


def test_layout_is_deterministic_and_rounded_to_three_decimals():
    matters = {"root", "left", "right", "tip"}
    dependencies = {
        ("root", "right"),
        ("left", "tip"),
        ("right", "tip"),
        ("root", "left"),
    }

    first = layout(matters, dependencies)
    second = layout(reversed(sorted(matters)), reversed(sorted(dependencies)))

    assert first == second
    for coordinate in first[1].values():
        for axis in ("x", "y", "z"):
            assert coordinate[axis] == round(coordinate[axis], 3)


def test_every_dependency_moves_up_in_depth_and_y():
    matters = {"a", "b", "c", "d"}
    dependencies = {("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")}
    metadata, coordinates = layout(matters, dependencies)

    assert metadata["max_depth"] == 2
    for source, target in dependencies:
        assert coordinates[target]["depth"] > coordinates[source]["depth"]
        assert coordinates[target]["y"] > coordinates[source]["y"]


def test_root_families_form_islands_and_join_between_them():
    matters = {"root_a", "root_b", "branch_a", "branch_b", "join", "after_join"}
    dependencies = {
        ("root_a", "branch_a"),
        ("root_b", "branch_b"),
        ("branch_a", "join"),
        ("branch_b", "join"),
        ("join", "after_join"),
    }
    metadata, coordinates = layout(matters, dependencies)

    assert metadata["island_count"] == 2
    assert coordinates["root_a"]["island"] == "root_a"
    assert coordinates["branch_a"]["island"] == "root_a"
    assert coordinates["branch_a"]["root_count"] == 1
    assert coordinates["join"]["island"] == "root_a"
    assert coordinates["join"]["root_count"] == 2
    assert coordinates["after_join"]["island"] == "root_a"
    assert coordinates["after_join"]["root_count"] == 1

    root_midpoint = {
        axis: (coordinates["root_a"][axis] + coordinates["root_b"][axis]) / 2
        for axis in ("x", "z")
    }
    join_offset = math.hypot(
        coordinates["join"]["x"] - root_midpoint["x"],
        coordinates["join"]["z"] - root_midpoint["z"],
    )
    assert join_offset <= 78.001


def test_condition_changes_do_not_move_nodes():
    matters = {"a", "b"}
    dependencies = {("a", "b")}
    _, before = layout(
        matters,
        dependencies,
        {"a": [{"label": "done", "truth": False}]},
    )
    _, after = layout(
        matters,
        dependencies,
        {"a": [{"label": "done", "truth": True}]},
    )

    assert before == after


def test_dependency_change_leaves_unrelated_branch_coordinates_unchanged():
    matters = {"a", "b", "c", "x", "y"}
    _, before = layout(matters, {("a", "b"), ("x", "y")})
    _, after = layout(matters, {("a", "b"), ("b", "c"), ("x", "y")})

    assert after["c"] != before["c"]
    for matter in {"a", "b", "x", "y"}:
        assert {
            axis: after[matter][axis] for axis in ("x", "y", "z", "depth")
        } == {
            axis: before[matter][axis] for axis in ("x", "y", "z", "depth")
        }


def test_layout_includes_exact_unique_downstream_impact():
    matters = {"root", "left", "right", "tip"}
    dependencies = {
        ("root", "left"),
        ("root", "right"),
        ("left", "tip"),
        ("right", "tip"),
    }
    _, coordinates = layout(matters, dependencies)

    assert coordinates["root"]["downstream_impact"] == 3
