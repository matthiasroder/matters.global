import pytest

from matters import (
    DependencyCycleError,
    dependents,
    frontier,
    has_dependency_cycle,
    horizon,
    prerequisites,
    resolved,
    universe,
)


def test_resolution_uses_conditions_and_prerequisites():
    matters = {"a", "b"}
    conditions = {
        "a": [{"label": "done", "truth": True}],
        "b": [{"label": "done", "truth": False}],
    }
    dependencies = {("a", "b")}

    assert resolved("a", matters, conditions, dependencies)
    assert not resolved("b", matters, conditions, dependencies)


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
    matters = {"root", "child", "grandchild"}
    conditions = {
        "root": [{"label": "done", "truth": True}],
        "child": [{"label": "done", "truth": False}],
        "grandchild": [{"label": "done", "truth": False}],
    }
    dependencies = {("root", "child"), ("child", "grandchild")}

    assert frontier("root", matters, conditions, dependencies) == {"child"}
    assert horizon("root", matters, conditions, dependencies) == {"grandchild"}


def test_dependency_cycle_raises_a_named_cycle():
    """The public names refuse a loop, and say which one (D7).

    They used to raise ``ValueError("dependency cycle")`` -- when they raised
    at all, which depended on whether a false condition short-circuited the
    recursion before it came back around. The refusal is now unconditional
    and carries the cycle as data. ``DependencyCycleError`` still subclasses
    ``ValueError``, so a caller catching that keeps working.
    """

    matters = {"a", "b"}
    dependencies = {("a", "b"), ("b", "a")}

    with pytest.raises(DependencyCycleError) as error:
        resolved("a", matters, {}, dependencies)

    assert str(error.value) == "dependency graph contains a cycle"
    assert error.value.cycle == ("a", "b")
    assert isinstance(error.value, ValueError)

    # Not conditional on the conditions: the same file refuses whether or not
    # a false condition sits in front of the loop.
    with pytest.raises(DependencyCycleError):
        resolved("a", matters, {"a": [{"label": "no", "truth": False}]}, dependencies)


def test_prerequisites_and_dependents_read_a_cyclic_edge_set():
    """The two reads that stay in ``engine`` never traverse (D5).

    This is the property ``rules.describe_matter`` leans on, and through it
    ``matters show`` on a state file that contains a loop.
    """

    dependencies = {("a", "b"), ("b", "a")}

    assert prerequisites("a", dependencies) == {"b"}
    assert dependents("a", dependencies) == {"b"}


def test_has_dependency_cycle_detects_cycles():
    assert has_dependency_cycle({("a", "b"), ("b", "c"), ("c", "a")})
    assert not has_dependency_cycle({("a", "b"), ("b", "c")})


def test_engine_does_not_import_graph_index():
    """D8: the import runs one way and must keep running one way.

    ``graph_index`` imports ``engine.truth``, so an import back would be a
    cycle -- which is why the derived functions moved out of ``engine``
    rather than staying there as wrappers.
    """

    import ast
    import inspect

    import matters.engine as engine

    tree = ast.parse(inspect.getsource(engine))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    # Prose mentioning graph_index is fine and wanted; an import is not, so
    # this reads the import statements rather than the file's text.
    assert "graph_index" not in {name.split(".")[0] for name in imported}
    assert not hasattr(engine, "resolved")
    assert not hasattr(engine, "unresolved")
    assert not hasattr(engine, "universe")
    assert not hasattr(engine, "frontier")
    assert not hasattr(engine, "horizon")
    assert not hasattr(engine, "descendants")
