import pytest

from matters import frontier, horizon, resolved, universe


def test_resolution_uses_conditions_and_prerequisites():
    conditions = {
        "a": [{"label": "done", "truth": True}],
        "b": [{"label": "done", "truth": False}],
    }
    dependencies = {("a", "b")}

    assert resolved("a", conditions, dependencies)
    assert not resolved("b", conditions, dependencies)


def test_universe_contains_unresolved_matters_with_resolved_prerequisites():
    matters = {"a", "b", "c"}
    conditions = {
        "a": [{"label": "done", "truth": True}],
        "b": [{"label": "done", "truth": False}],
        "c": [{"label": "done", "truth": False}],
    }
    dependencies = {("a", "b"), ("b", "c")}

    assert universe(matters, conditions, dependencies) == {"b"}


def test_frontier_and_horizon_are_computed_from_dependencies():
    conditions = {
        "root": [{"label": "done", "truth": True}],
        "child": [{"label": "done", "truth": False}],
        "grandchild": [{"label": "done", "truth": False}],
    }
    dependencies = {("root", "child"), ("child", "grandchild")}

    assert frontier("root", conditions, dependencies) == {"child"}
    assert horizon("root", conditions, dependencies) == {"grandchild"}


def test_dependency_cycle_raises_value_error():
    with pytest.raises(ValueError, match="dependency cycle"):
        resolved("a", {}, {("a", "b"), ("b", "a")})
