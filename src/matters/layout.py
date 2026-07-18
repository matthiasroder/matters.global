"""Deterministic spatial layouts for matters dependency graphs."""

from __future__ import annotations

import hashlib
import math


LAYOUT_VERSION = 2
LAYOUT_ALGORITHM = "layered-archipelago-v1"
ROOT_GRID_MIN_SIZE = 16
ROOT_GRID_TARGET_LOAD = 0.72
ISLAND_SPACING = 520.0
ISLAND_JITTER = 46.0
TERRACE_RADIUS_MIN = 22.0
TERRACE_RADIUS_STEP = 8.0
TERRACE_RADIUS_MAX = 78.0
VERTICAL_STEP = 120.0


def build_overview_layout(index):
    """Return overview metadata and stable coordinates for ``index``.

    Every dependency root owns an island anchor. Matters that inherit one root
    form vertical dependency terraces around that anchor; matters that directly
    join several root families sit between the contributing islands and pass a
    stable primary island to their descendants. Coordinates depend only on
    stable matter IDs and dependency structure, never on conditions.
    """

    island_memberships = _island_memberships(index)
    roots = tuple(
        matter for matter in index.topological_order if not index.prerequisites[matter]
    )
    root_anchors = _root_anchors(roots)
    coordinates = {}

    for matter in index.topological_order:
        depth = index.depth[matter]
        primary_island, contributing_islands = island_memberships[matter]
        anchor_x = sum(
            root_anchors[root][0] for root in contributing_islands
        ) / len(contributing_islands)
        anchor_z = sum(
            root_anchors[root][1] for root in contributing_islands
        ) / len(contributing_islands)

        if depth == 0:
            x, z = anchor_x, anchor_z
        else:
            angle = _unit_interval(matter, "terrace-angle") * math.tau
            max_radius = min(
                TERRACE_RADIUS_MAX,
                TERRACE_RADIUS_MIN + depth * TERRACE_RADIUS_STEP,
            )
            radius = TERRACE_RADIUS_MIN + _unit_interval(
                matter, "terrace-radius"
            ) * (max_radius - TERRACE_RADIUS_MIN)
            x = anchor_x + radius * math.cos(angle)
            z = anchor_z + radius * math.sin(angle)

        coordinates[matter] = {
            "x": round(x, 3),
            "y": round(depth * VERTICAL_STEP, 3),
            "z": round(z, 3),
            "depth": depth,
            "downstream_impact": index.downstream_impact[matter],
            "island": primary_island,
            "root_count": len(contributing_islands),
        }

    metadata = {
        "version": LAYOUT_VERSION,
        "algorithm": LAYOUT_ALGORITHM,
        "max_depth": max(index.depth.values(), default=0),
        "island_count": len(roots),
        "bounds": _bounds(coordinates),
    }
    return metadata, coordinates


def _island_memberships(index):
    memberships = {}
    for matter in index.topological_order:
        prerequisites = index.prerequisites[matter]
        contributing_islands = (
            tuple(sorted({memberships[item][0] for item in prerequisites}))
            if prerequisites
            else (matter,)
        )
        memberships[matter] = (min(contributing_islands), contributing_islands)
    return memberships


def _root_anchors(roots):
    if not roots:
        return {}

    grid_size = max(
        ROOT_GRID_MIN_SIZE,
        math.ceil(math.sqrt(len(roots) / ROOT_GRID_TARGET_LOAD)),
    )
    slot_count = grid_size * grid_size
    occupied = set()
    anchors = {}

    for root in sorted(roots):
        start = int(_unit_interval(root, "island-slot") * slot_count) % slot_count
        step = _coprime_step(root, slot_count)
        for attempt in range(slot_count):
            slot = (start + attempt * step) % slot_count
            if slot not in occupied:
                occupied.add(slot)
                break
        else:  # pragma: no cover - grid sizing guarantees a free slot
            raise RuntimeError("overview island grid is unexpectedly full")

        column = slot % grid_size
        row = slot // grid_size
        jitter_x = (
            _unit_interval(root, "island-jitter-x") - 0.5
        ) * 2.0 * ISLAND_JITTER
        jitter_z = (
            _unit_interval(root, "island-jitter-z") - 0.5
        ) * 2.0 * ISLAND_JITTER
        anchors[root] = (
            (column - (grid_size - 1) / 2.0) * ISLAND_SPACING + jitter_x,
            (row - (grid_size - 1) / 2.0) * ISLAND_SPACING + jitter_z,
        )

    return anchors


def _coprime_step(root, slot_count):
    step = 1 + int(_unit_interval(root, "island-step") * (slot_count - 1))
    while math.gcd(step, slot_count) != 1:
        step = step % (slot_count - 1) + 1
    return step


def _unit_interval(matter, purpose):
    digest = hashlib.sha256(
        f"{LAYOUT_ALGORITHM}\0{purpose}\0{matter}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)


def _bounds(coordinates):
    if not coordinates:
        return {
            "min_x": 0.0,
            "max_x": 0.0,
            "min_y": 0.0,
            "max_y": 0.0,
            "min_z": 0.0,
            "max_z": 0.0,
        }

    return {
        f"{bound}_{axis}": aggregate(
            coordinate[axis] for coordinate in coordinates.values()
        )
        for axis in ("x", "y", "z")
        for bound, aggregate in (("min", min), ("max", max))
    }
