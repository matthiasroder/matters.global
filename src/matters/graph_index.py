"""Indexed, deterministic derived values for a matters dependency graph.

The single implementation of every value derived from graph shape. The
module-level functions at the bottom are the historical ``engine`` names
(``resolved``, ``unresolved``, ``universe``, ``frontier``, ``horizon``,
``descendants``); they moved here and now mean what :class:`GraphIndex`
means, which is a deliberate behaviour change at 0.1.0 (D5/D7), not an
accident. The visible differences on the graphs where the two used to
disagree:

============================  =========================  ===================
Input                         Old ``engine`` behaviour   Now
============================  =========================  ===================
graph with a cycle            ``ValueError("dependency   ``DependencyCycle
                              cycle")`` *or* a silent    Error`` naming one
                              wrong answer, depending    concrete cycle,
                              on which conditions were   always
                              false first
chain of ~331 edges or more   ``RecursionError``         answered; the
                                                         traversal is
                                                         iterative
probe id absent from          answered as if it existed  ``KeyError``
``matters``
============================  =========================  ===================

``engine`` imports nothing from here and this module imports only
:func:`matters.engine.truth`; the direction is fixed, because the reverse is
a circular import (D8).
"""

from __future__ import annotations

import heapq

from .engine import truth


class DependencyCycleError(ValueError):
    """Raised when a dependency graph cannot be topologically ordered.

    ``cycle`` is one concrete cycle, as a tuple of matter ids in edge
    direction (``prerequisite`` first), each id appearing exactly once: the
    closing edge from the last id back to the first is implied, so a
    self-loop is the one-element tuple ``("a",)``. A graph can hold several
    cycles; this names one of them, which is all a caller needs to remove an
    edge and try again.

    The argument is optional so that ``DependencyCycleError()`` keeps
    working, and the message is unchanged: callers that render the cycle for
    a person read ``.cycle`` and format it themselves (``rules.format_cycle``
    is the one renderer).
    """

    def __init__(self, cycle=()):
        super().__init__("dependency graph contains a cycle")
        self.cycle = tuple(cycle)


def extract_cycle(prerequisite_sets, remaining):
    """Return one cycle drawn from ``remaining``, in edge direction.

    ``remaining`` is exactly the set Kahn's algorithm could not order: every
    matter in it is in a cycle or downstream of one, and every one of them
    therefore still has at least one prerequisite inside ``remaining`` --
    otherwise its indegree would have reached zero. Walking prerequisites can
    only stay inside the set, so a finite walk must revisit a matter, and the
    revisited prefix is a cycle.

    Deterministic twice over, because an error message that changes between
    two runs on the same file is not a message a person can act on: the walk
    starts at the smallest id and always takes the smallest prerequisite, and
    the result is rotated to begin at its own smallest id, so which node the
    walk happened to enter the cycle from cannot show up in the output.

    Returns matter ids in ``source -> target`` order (prerequisite first),
    the same direction the dependency tuples and ``create`` output use, with
    no repeated closing element.
    """

    if not remaining:
        return ()

    path = []
    position = {}
    matter = min(remaining)
    while matter not in position:
        position[matter] = len(path)
        path.append(matter)
        matter = min(
            prerequisite
            for prerequisite in prerequisite_sets[matter]
            if prerequisite in remaining
        )

    # The walk followed prerequisites, which is against edge direction.
    cycle = tuple(reversed(path[position[matter] :]))
    start = cycle.index(min(cycle))
    return cycle[start:] + cycle[:start]


class GraphIndex:
    """Precompute adjacency, status, depth, and reachability for one graph.

    The index is intentionally derived from the in-memory graph only. It does
    not add fields to, or otherwise alter, the persisted matters state.
    """

    def __init__(self, matters, conditions, dependencies):
        self.matters = tuple(sorted(matters))
        matter_set = set(self.matters)

        prerequisite_sets = {matter: set() for matter in self.matters}
        dependent_sets = {matter: set() for matter in self.matters}
        for source, target in dependencies:
            if source not in matter_set:
                raise ValueError(f"dependency has unknown source: {source}")
            if target not in matter_set:
                raise ValueError(f"dependency has unknown target: {target}")
            prerequisite_sets[target].add(source)
            dependent_sets[source].add(target)

        self.prerequisites = {
            matter: tuple(sorted(prerequisite_sets[matter]))
            for matter in self.matters
        }
        self.dependents = {
            matter: tuple(sorted(dependent_sets[matter]))
            for matter in self.matters
        }
        self.topological_order = self._topological_order(prerequisite_sets)

        depth = {}
        resolved = {}
        actionable = set()
        blocked = set()
        for matter in self.topological_order:
            matter_prerequisites = self.prerequisites[matter]
            depth[matter] = (
                max(depth[prerequisite] for prerequisite in matter_prerequisites) + 1
                if matter_prerequisites
                else 0
            )
            prerequisites_resolved = all(
                resolved[prerequisite] for prerequisite in matter_prerequisites
            )
            resolved[matter] = prerequisites_resolved and all(
                truth(condition) for condition in conditions.get(matter, ())
            )
            if not resolved[matter]:
                if prerequisites_resolved:
                    actionable.add(matter)
                else:
                    blocked.add(matter)

        self.depth = depth
        self.resolved = resolved
        self.universe = frozenset(actionable)
        self.blocked = frozenset(blocked)

        # Reachability as one big int per matter, so a union of two reachable
        # sets is one ``|`` and a count is one ``bit_count()``. The two passes
        # are mirror images and must stay that way: descendants close over
        # ``dependents`` walking the topological order backwards, ancestors
        # close over ``prerequisites`` walking it forwards. Each pass reads
        # only entries it has already filled, which is what the order buys.
        bit_index = {
            matter: index for index, matter in enumerate(self.topological_order)
        }
        descendant_bits = {}
        for matter in reversed(self.topological_order):
            bits = 0
            for dependent in self.dependents[matter]:
                bits |= 1 << bit_index[dependent]
                bits |= descendant_bits[dependent]
            descendant_bits[matter] = bits
        ancestor_bits = {}
        for matter in self.topological_order:
            bits = 0
            for prerequisite in self.prerequisites[matter]:
                bits |= 1 << bit_index[prerequisite]
                bits |= ancestor_bits[prerequisite]
            ancestor_bits[matter] = bits
        self._bit_index = bit_index
        self._descendant_bits = descendant_bits
        self._ancestor_bits = ancestor_bits
        self.downstream_impact = {
            matter: bits.bit_count() for matter, bits in descendant_bits.items()
        }

    def _topological_order(self, prerequisite_sets):
        indegree = {
            matter: len(prerequisite_sets[matter]) for matter in self.matters
        }
        ready = [matter for matter in self.matters if indegree[matter] == 0]
        heapq.heapify(ready)
        order = []

        while ready:
            matter = heapq.heappop(ready)
            order.append(matter)
            for dependent in self.dependents[matter]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heapq.heappush(ready, dependent)

        if len(order) != len(self.matters):
            remaining = frozenset(self.matters) - frozenset(order)
            raise DependencyCycleError(
                extract_cycle(prerequisite_sets, remaining)
            )
        return tuple(order)

    def descendants(self, matter):
        """Return the exact transitive dependent set for ``matter``.

        Excludes ``matter`` itself, which is not a special case: the graph is
        acyclic by construction, so no matter reaches itself.
        """

        self._require_matter(matter)
        return self._decode(self._descendant_bits[matter])

    def ancestors(self, matter):
        """Return the exact transitive prerequisite set for ``matter``.

        The mirror of :meth:`descendants`: everything this matter waits on,
        at any remove, where ``self.prerequisites[matter]`` is the same
        question one hop out.
        """

        self._require_matter(matter)
        return self._decode(self._ancestor_bits[matter])

    def neighborhood(self, matter, *, up=None, down=None):
        """Return everything within ``up`` hops back and ``down`` hops on.

        ``None`` means unlimited, so ``neighborhood(m)`` is
        ``ancestors(m) | descendants(m)``, ``neighborhood(m, up=0)`` is
        ``descendants(m)`` and ``neighborhood(m, down=0)`` is
        ``ancestors(m)``. ``matter`` itself is never in the result, matching
        both unlimited cases; a caller that wants the closed slice adds it.

        A hop count is the *shortest* path length, not the longest: a matter
        two hops away by one route and five by another is inside
        ``down=2``. That is the useful reading for "show me a bit more
        context", which is what the limit exists for.
        """

        self._require_matter(matter)
        return self._reachable(
            matter, self.prerequisites, self._ancestor_bits, up
        ) | self._reachable(
            matter, self.dependents, self._descendant_bits, down
        )

    def _reachable(self, matter, adjacency, unlimited_bits, hops):
        """Walk ``adjacency`` from ``matter``, at most ``hops`` steps.

        The unlimited case reads the precomputed bitset instead of walking,
        which is the whole reason the bitsets exist.
        """

        if hops is None:
            return self._decode(unlimited_bits[matter])
        if hops < 0:
            raise ValueError(f"hop limit must not be negative: {hops}")

        reached = set()
        current = {matter}
        for _ in range(hops):
            if not current:
                break
            current = {
                neighbour
                for node in current
                for neighbour in adjacency[node]
                if neighbour not in reached
            }
            reached |= current
        return reached

    def _decode(self, bits):
        """Turn one reachability bitset back into a set of matter ids."""

        return {
            candidate
            for candidate, index in self._bit_index.items()
            if bits & (1 << index)
        }

    def frontier(self, matter):
        """Return the actionable direct dependents of ``matter``."""

        self._require_matter(matter)
        return {
            dependent
            for dependent in self.dependents[matter]
            if dependent in self.universe
        }

    def horizon(self, matter):
        """Return the unresolved terminal descendants of ``matter``.

        Terminal *within the descendant set*: a matter counts when nothing
        downstream of it is both a descendant of ``matter`` and unresolved.
        """

        descendants = self.descendants(matter)
        return {
            descendant
            for descendant in descendants
            if not self.resolved[descendant]
            and not any(
                dependent in descendants and not self.resolved[dependent]
                for dependent in self.dependents[descendant]
            )
        }

    def _require_matter(self, matter):
        if matter not in self.resolved:
            raise KeyError(matter)


# ---------------------------------------------------------------------------
# The historical engine names
#
# One-shot wrappers: each builds an index over the whole graph and asks it one
# question, so a caller with several questions should build one
# :class:`GraphIndex` and keep it rather than paying for the index per call.
# They exist so ``matters.resolved`` and friends keep resolving to something
# after the move (D7), and because a one-line question deserves a one-line
# call.
#
# Every one of them takes the ``matters`` set. ``frontier``, ``horizon`` and
# ``descendants`` used to infer their nodes from the edge list, which is the
# habit this module exists to end: an edge list cannot see an isolated matter,
# and cannot tell a typo'd id from a real one. The added parameter is part of
# the deliberate 0.1.0 break.
# ---------------------------------------------------------------------------


def resolved(matter, matters, conditions, dependencies):
    """Return whether ``matter`` is resolved."""

    # Indexing the dict raises ``KeyError(matter)`` for an id that is not in
    # ``matters``, which is the same refusal the index's own methods make.
    return GraphIndex(matters, conditions, dependencies).resolved[matter]


def unresolved(matter, matters, conditions, dependencies):
    """Return whether ``matter`` is not resolved."""

    return not resolved(matter, matters, conditions, dependencies)


def universe(matters, conditions, dependencies):
    """Return every actionable matter: unresolved, prerequisites resolved."""

    return set(GraphIndex(matters, conditions, dependencies).universe)


def frontier(root, matters, conditions, dependencies):
    """Return the actionable direct dependents of ``root``."""

    return GraphIndex(matters, conditions, dependencies).frontier(root)


def horizon(root, matters, conditions, dependencies):
    """Return the unresolved terminal descendants of ``root``."""

    return GraphIndex(matters, conditions, dependencies).horizon(root)


def descendants(root, matters, conditions, dependencies):
    """Return the transitive dependent set of ``root``."""

    return GraphIndex(matters, conditions, dependencies).descendants(root)


def induced_subgraph(matters, conditions, dependencies, ids):
    """Return the ``(matters, conditions, dependencies)`` triple over ``ids``.

    An edge survives only when **both** endpoints are kept, which is what
    makes the result loadable on its own: a dangling endpoint is rejected by
    ``storage.normalize_dependency_records`` and by :class:`GraphIndex`.
    ``sharing.public_state`` filters a visibility slice by the same rule; this
    is the same shape of operation over an arbitrary id set, and deliberately
    does not import ``sharing``, whose job is a wire format and which owns
    condition re-serialisation and sort order that a subgraph must not
    inherit.

    ``ids`` may name matters that are not in ``matters``; they are dropped
    rather than raising, so a caller can intersect two slices without
    checking first. The returned types match ``storage.load_state`` --
    ``set``, ``dict`` of ``list``, ``set`` of pairs -- and condition lists are
    copied one level deep, so appending to the result cannot reach back into
    the source state.

    A matter carrying no conditions entry keeps no conditions entry, rather
    than gaining an empty one, which makes the whole-graph case an identity:
    ``induced_subgraph(m, c, d, m) == (m, c, d)``.
    """

    kept = {matter for matter in ids if matter in matters}
    return (
        kept,
        {
            matter: list(conditions[matter])
            for matter in kept
            if matter in conditions
        },
        {
            (source, target)
            for source, target in dependencies
            if source in kept and target in kept
        },
    )
