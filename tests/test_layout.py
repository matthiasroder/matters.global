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
        "version": 3,
        "algorithm": "solar-systems-v1",
        "max_depth": 0,
        "system_count": 0,
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


def test_every_dependency_moves_inward_toward_its_goal():
    matters = {"a", "b", "c", "d"}
    dependencies = {("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")}
    metadata, coordinates = layout(matters, dependencies)

    assert metadata["max_depth"] == 2
    for source, target in dependencies:
        assert coordinates[target]["depth"] > coordinates[source]["depth"]
        assert coordinates[target]["orbit_level"] < coordinates[source]["orbit_level"]
        assert coordinates[target]["y"] < coordinates[source]["y"]


def test_dependency_depths_form_orbits_and_join_between_systems():
    matters = {
        "root_a", "a1", "a2", "a3", "goal_a",
        "root_b", "b1", "b2", "b3", "goal_b",
        "bridge", "z_bridge_goal",
    }
    dependencies = {
        ("root_a", "a1"),
        ("a1", "a2"),
        ("a2", "a3"),
        ("a3", "goal_a"),
        ("root_b", "b1"),
        ("b1", "b2"),
        ("b2", "b3"),
        ("b3", "goal_b"),
        ("a1", "bridge"),
        ("b1", "bridge"),
        ("bridge", "z_bridge_goal"),
    }
    metadata, coordinates = layout(matters, dependencies)

    assert metadata["system_count"] == 2
    assert coordinates["goal_a"]["system"] == "goal_a"
    assert coordinates["goal_a"]["orbit_radius"] == 0
    assert coordinates["goal_a"]["system_population"] > 1
    assert coordinates["goal_a"]["system_radius"] > coordinates["a1"]["orbit_radius"]
    assert coordinates["a1"]["system"] == "goal_a"
    assert coordinates["a1"]["system_count"] == 1
    assert coordinates["a1"]["orbit_radius"] > 0
    assert coordinates["root_a"]["system"] == "goal_a"
    assert coordinates["root_a"]["system_count"] == 1
    assert coordinates["root_a"]["orbit_radius"] > coordinates["a1"]["orbit_radius"]
    assert coordinates["bridge"]["system"] == "goal_a"
    assert coordinates["bridge"]["system_count"] == 2
    assert coordinates["bridge"]["orbit_radius"] == 0

    branch_distance = math.hypot(
        coordinates["a1"]["x"] - coordinates["goal_a"]["x"],
        coordinates["a1"]["z"] - coordinates["goal_a"]["z"],
    )
    assert abs(branch_distance - coordinates["a1"]["orbit_radius"]) <= 5.001

    root_midpoint = {
        axis: (coordinates["goal_a"][axis] + coordinates["goal_b"][axis]) / 2
        for axis in ("x", "z")
    }
    join_offset = math.hypot(
        coordinates["bridge"]["x"] - root_midpoint["x"],
        coordinates["bridge"]["z"] - root_midpoint["z"],
    )
    assert join_offset <= 24.001

    system_distance = math.hypot(
        coordinates["goal_a"]["x"] - coordinates["goal_b"]["x"],
        coordinates["goal_a"]["z"] - coordinates["goal_b"]["z"],
    )
    assert system_distance >= 336


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
