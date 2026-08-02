import pytest

from matters.graph_index import (
    DependencyCycleError,
    GraphIndex,
    descendants,
    frontier,
    horizon,
    induced_subgraph,
    resolved,
    universe,
    unresolved,
)


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


def case_conditions(matters):
    return {
        matter: [{"label": f"{matter} done", "truth": matter in {"a", "root"}}]
        for matter in matters
    }


@pytest.mark.parametrize(("matters", "dependencies"), GRAPH_CASES)
def test_module_level_functions_answer_what_the_index_answers(matters, dependencies):
    """The historical names are one-shot wrappers, not a second traversal.

    This test used to compare ``GraphIndex`` against ``engine``'s recursive
    twin. There is no twin now, so it pins the wrappers to the index instead
    -- which is the whole point of the move: one implementation, reached two
    ways.
    """

    conditions = case_conditions(matters)
    index = GraphIndex(matters, conditions, dependencies)

    assert universe(matters, conditions, dependencies) == set(index.universe)
    for matter in matters:
        assert resolved(matter, matters, conditions, dependencies) is index.resolved[
            matter
        ]
        assert unresolved(matter, matters, conditions, dependencies) is not index.resolved[
            matter
        ]
        assert descendants(matter, matters, conditions, dependencies) == index.descendants(
            matter
        )
        assert index.downstream_impact[matter] == len(index.descendants(matter))
        assert frontier(matter, matters, conditions, dependencies) == index.frontier(matter)
        assert horizon(matter, matters, conditions, dependencies) == index.horizon(matter)


@pytest.mark.parametrize(("matters", "dependencies"), GRAPH_CASES)
def test_ancestors_and_descendants_are_mirrors(matters, dependencies):
    index = GraphIndex(matters, case_conditions(matters), dependencies)

    for matter in matters:
        for reached in index.descendants(matter):
            assert matter in index.ancestors(reached)
        for reached in index.ancestors(matter):
            assert matter in index.descendants(reached)
        # Acyclic by construction, so nothing reaches itself either way.
        assert matter not in index.ancestors(matter)
        assert matter not in index.descendants(matter)


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


def test_graph_index_counts_diamond_ancestors_once():
    index = GraphIndex(
        {"root", "left", "right", "tip"},
        {},
        {("root", "left"), ("root", "right"), ("left", "tip"), ("right", "tip")},
    )

    assert index.ancestors("tip") == {"root", "left", "right"}
    assert index.ancestors("root") == set()


# ---------------------------------------------------------------------------
# neighborhood
# ---------------------------------------------------------------------------


CHAIN = ("a", "b", "c", "d", "e")
CHAIN_EDGES = {("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")}


def chain_index():
    return GraphIndex(set(CHAIN), {}, CHAIN_EDGES)


def test_neighborhood_is_unlimited_in_both_directions_by_default():
    index = chain_index()

    assert index.neighborhood("c") == {"a", "b", "d", "e"}
    assert index.neighborhood("c") == index.ancestors("c") | index.descendants("c")


def test_neighborhood_limits_each_direction_independently():
    index = chain_index()

    assert index.neighborhood("c", up=1, down=1) == {"b", "d"}
    assert index.neighborhood("c", up=0, down=None) == index.descendants("c")
    assert index.neighborhood("c", up=None, down=0) == index.ancestors("c")
    assert index.neighborhood("c", up=0, down=0) == set()
    # A limit past the end of the graph is not an error, it just stops.
    assert index.neighborhood("c", up=99, down=99) == {"a", "b", "d", "e"}


def test_neighborhood_measures_the_shortest_path_not_the_longest():
    """A shortcut edge pulls ``tip`` inside ``down=1`` even though the long
    way round is two hops."""

    index = GraphIndex(
        {"root", "mid", "tip"},
        {},
        {("root", "mid"), ("mid", "tip"), ("root", "tip")},
    )

    assert index.neighborhood("root", down=1) == {"mid", "tip"}


def test_neighborhood_rejects_a_negative_hop_limit():
    with pytest.raises(ValueError, match="hop limit must not be negative"):
        chain_index().neighborhood("c", down=-1)


def test_neighborhood_and_ancestors_reject_an_unknown_matter():
    index = chain_index()

    with pytest.raises(KeyError):
        index.ancestors("ghost")
    with pytest.raises(KeyError):
        index.neighborhood("ghost")


# ---------------------------------------------------------------------------
# The shapes the old parity test never covered
# ---------------------------------------------------------------------------


def test_a_cycle_is_refused_and_named():
    with pytest.raises(DependencyCycleError) as error:
        GraphIndex({"a", "b", "c"}, {}, {("a", "b"), ("b", "c"), ("c", "a")})

    assert str(error.value) == "dependency graph contains a cycle"
    assert error.value.cycle == ("a", "b", "c")


def test_a_self_loop_is_refused_and_named_as_one_element():
    with pytest.raises(DependencyCycleError) as error:
        GraphIndex({"a"}, {}, {("a", "a")})

    assert error.value.cycle == ("a",)


def test_a_cycle_off_to_one_side_still_refuses_the_whole_graph():
    """The refusal is about the file, not about the matter being asked for.

    ``clean`` is in no cycle and depends on nothing that is, but the index is
    of one graph and there is no partial index. This is what makes the CLI's
    three derived verbs report a loop they do not touch -- and why ``show``
    and ``list``, which build no index, must keep answering.
    """

    matters = {"clean", "a", "b"}
    dependencies = {("a", "b"), ("b", "a")}

    with pytest.raises(DependencyCycleError) as error:
        GraphIndex(matters, {}, dependencies)

    assert error.value.cycle == ("a", "b")
    with pytest.raises(DependencyCycleError):
        resolved("clean", matters, {}, dependencies)


@pytest.mark.parametrize(
    "call",
    [
        lambda: resolved("ghost", {"a"}, {}, set()),
        lambda: unresolved("ghost", {"a"}, {}, set()),
        lambda: descendants("ghost", {"a"}, {}, set()),
        lambda: frontier("ghost", {"a"}, {}, set()),
        lambda: horizon("ghost", {"a"}, {}, set()),
    ],
)
def test_probing_an_id_that_is_not_a_matter_raises(call):
    """The old engine answered for an id it had never heard of.

    ``descendants("ghost", ...)`` returned an empty set and
    ``resolved("ghost", ...)`` returned ``True``, because a matter with no
    conditions and no prerequisites is vacuously resolved. A typo therefore
    read as a real answer. It now raises.
    """

    with pytest.raises(KeyError, match="ghost"):
        call()


def test_a_matter_with_an_empty_condition_list_is_resolved():
    matters = {"empty", "waiting"}
    conditions = {"empty": [], "waiting": [{"label": "no", "truth": False}]}
    dependencies = {("empty", "waiting")}

    assert resolved("empty", matters, conditions, dependencies)
    assert universe(matters, conditions, dependencies) == {"waiting"}
    assert frontier("empty", matters, conditions, dependencies) == {"waiting"}


def test_a_matter_missing_from_conditions_entirely_is_resolved():
    matters = {"empty", "waiting"}
    conditions = {"waiting": [{"label": "no", "truth": False}]}
    dependencies = {("empty", "waiting")}

    assert resolved("empty", matters, conditions, dependencies)


def test_a_long_chain_does_not_exhaust_the_stack():
    """The recursive engine raised ``RecursionError`` around 331 edges."""

    ids = [f"m{number:05d}" for number in range(3000)]
    dependencies = {(ids[position], ids[position + 1]) for position in range(len(ids) - 1)}
    conditions = {ids[0]: [{"label": "no", "truth": False}]}

    assert universe(set(ids), conditions, dependencies) == {ids[0]}
    assert len(descendants(ids[0], set(ids), conditions, dependencies)) == len(ids) - 1


def test_graph_index_rejects_unknown_dependency_endpoints():
    with pytest.raises(ValueError, match="unknown target: missing"):
        GraphIndex({"a"}, {}, {("a", "missing")})


# ---------------------------------------------------------------------------
# induced_subgraph
# ---------------------------------------------------------------------------


def test_induced_subgraph_keeps_an_edge_only_when_both_endpoints_survive():
    matters = {"a", "b", "c", "loose"}
    conditions = {matter: [{"label": f"{matter} done", "truth": False}] for matter in matters}
    dependencies = {("a", "b"), ("b", "c")}

    kept_matters, kept_conditions, kept_dependencies = induced_subgraph(
        matters, conditions, dependencies, {"a", "b", "loose"}
    )

    assert kept_matters == {"a", "b", "loose"}
    assert set(kept_conditions) == {"a", "b", "loose"}
    assert kept_dependencies == {("a", "b")}
    # The result stands on its own: no edge points at a matter it dropped.
    GraphIndex(kept_matters, kept_conditions, kept_dependencies)


def test_induced_subgraph_returns_load_state_shapes_and_copies_conditions():
    matters = {"a", "b"}
    conditions = {"a": [{"label": "a done", "truth": False}], "b": []}
    dependencies = {("a", "b")}

    kept_matters, kept_conditions, kept_dependencies = induced_subgraph(
        matters, conditions, dependencies, {"a", "b"}
    )

    assert isinstance(kept_matters, set)
    assert isinstance(kept_conditions, dict)
    assert isinstance(kept_dependencies, set)

    kept_conditions["a"].append({"label": "added", "truth": True})
    assert conditions["a"] == [{"label": "a done", "truth": False}]


def test_induced_subgraph_drops_ids_that_are_not_matters():
    kept_matters, kept_conditions, kept_dependencies = induced_subgraph(
        {"a"}, {"a": []}, set(), {"a", "ghost"}
    )

    assert kept_matters == {"a"}
    assert kept_conditions == {"a": []}
    assert kept_dependencies == set()


def test_induced_subgraph_of_nothing_is_an_empty_graph():
    assert induced_subgraph({"a"}, {"a": []}, set(), set()) == (set(), {}, set())


def test_induced_subgraph_of_the_whole_graph_is_the_whole_graph():
    """Including for a matter that carries no conditions entry at all."""

    matters = {"a", "b", "silent"}
    conditions = {"a": [{"label": "a done", "truth": False}], "b": []}
    dependencies = {("a", "b")}

    assert induced_subgraph(matters, conditions, dependencies, matters) == (
        matters,
        conditions,
        dependencies,
    )
