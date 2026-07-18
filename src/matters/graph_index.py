"""Indexed, deterministic derived values for a matters dependency graph."""

from __future__ import annotations

import heapq

from .engine import truth


class DependencyCycleError(ValueError):
    """Raised when a dependency graph cannot be topologically ordered."""

    def __init__(self):
        super().__init__("dependency graph contains a cycle")


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
        self._bit_index = bit_index
        self._descendant_bits = descendant_bits
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
            raise DependencyCycleError()
        return tuple(order)

    def descendants(self, matter):
        """Return the exact transitive dependent set for ``matter``."""

        self._require_matter(matter)
        bits = self._descendant_bits[matter]
        return {
            candidate
            for candidate, index in self._bit_index.items()
            if bits & (1 << index)
        }

    def frontier(self, matter):
        """Return actionable direct dependents, matching engine.frontier."""

        self._require_matter(matter)
        return {
            dependent
            for dependent in self.dependents[matter]
            if dependent in self.universe
        }

    def horizon(self, matter):
        """Return unresolved terminal descendants, matching engine.horizon."""

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
