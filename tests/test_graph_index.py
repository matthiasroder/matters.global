import pytest

from matters.engine import descendants, frontier, horizon, resolved, universe
from matters.graph_index import DependencyCycleError, GraphIndex


GRAPH_CASES = [
    (set(), set()),
    ({"only"}, set()),
    ({"a", "b", "c"}, {("a", "b"), ("b", "c")}),
    (
        {"root", "left", "right", "tip"},
        {("root", "left"), ("root", "right"), ("left", "tip"), ("right", "tip")},
    ),
    ({"a", "b", "x", "y"}, {("a", "b"), ("x", "y")}),
    ({"a", "z", "join"}, {("a", "join"), ("z", "join")}),
]


@pytest.mark.parametrize(("matters", "dependencies"), GRAPH_CASES)
def test_graph_index_matches_engine_derived_semantics(matters, dependencies):
    conditions = {
        matter: [{"label": f"{matter} done", "truth": matter in {"a", "root"}}]
        for matter in matters
    }
    index = GraphIndex(matters, conditions, dependencies)

    assert index.universe == universe(matters, conditions, dependencies)
    for matter in matters:
        assert index.resolved[matter] == resolved(matter, conditions, dependencies)
        assert index.descendants(matter) == descendants(matter, dependencies)
        assert index.downstream_impact[matter] == len(descendants(matter, dependencies))
        assert index.frontier(matter) == frontier(matter, conditions, dependencies)
        assert index.horizon(matter) == horizon(matter, conditions, dependencies)


def test_graph_index_uses_lexicographic_topological_order_and_longest_depth():
    index = GraphIndex(
        {"root", "alpha", "beta", "join", "tip"},
        {},
        {
            ("root", "alpha"),
            ("root", "beta"),
            ("alpha", "join"),
            ("beta", "join"),
            ("join", "tip"),
        },
    )

    assert index.topological_order == ("root", "alpha", "beta", "join", "tip")
    assert index.depth == {"root": 0, "alpha": 1, "beta": 1, "join": 2, "tip": 3}


def test_graph_index_counts_diamond_descendants_once():
    index = GraphIndex(
        {"root", "left", "right", "tip"},
        {},
        {("root", "left"), ("root", "right"), ("left", "tip"), ("right", "tip")},
    )

    assert index.descendants("root") == {"left", "right", "tip"}
    assert index.downstream_impact["root"] == 3


def test_graph_index_rejects_dependency_cycles():
    with pytest.raises(DependencyCycleError, match="dependency graph contains a cycle"):
        GraphIndex({"a", "b", "c"}, {}, {("a", "b"), ("b", "c"), ("c", "a")})


def test_graph_index_rejects_unknown_dependency_endpoints():
    with pytest.raises(ValueError, match="unknown target: missing"):
        GraphIndex({"a"}, {}, {("a", "missing")})
