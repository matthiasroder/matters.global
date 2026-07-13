"""Deterministic spatial layouts for matters dependency graphs."""

from __future__ import annotations

import hashlib
import math


LAYOUT_VERSION = 1
LAYOUT_ALGORITHM = "dependency-cone-v1"
ROOT_RADIUS = 120.0
DEPTH_RADIUS_STEP = 72.0
RADIAL_JITTER = 34.0
ANGULAR_JITTER = 0.52
VERTICAL_STEP = 120.0


def build_overview_layout(index):
    """Return overview metadata and stable coordinates for ``index``.

    Coordinates depend only on stable matter IDs and dependency structure.
    In particular, conditions and their truth values never affect placement.
    """

    primary_roots = _primary_roots(index)
    coordinates = {}

    for matter in index.topological_order:
        depth = index.depth[matter]
        root = primary_roots[matter]
        root_angle = _unit_interval(root, "root-angle") * math.tau

        if depth == 0:
            angle = root_angle
            radius = ROOT_RADIUS
        else:
            angular_offset = (
                _unit_interval(matter, "angular-jitter") - 0.5
            ) * 2.0 * ANGULAR_JITTER
            radial_offset = (
                _unit_interval(matter, "radial-jitter") - 0.5
            ) * 2.0 * RADIAL_JITTER
            angle = root_angle + angular_offset
            radius = ROOT_RADIUS + depth * DEPTH_RADIUS_STEP + radial_offset

        coordinates[matter] = {
            "x": round(radius * math.cos(angle), 3),
            "y": round(depth * VERTICAL_STEP, 3),
            "z": round(radius * math.sin(angle), 3),
            "depth": depth,
            "downstream_impact": index.downstream_impact[matter],
        }

    metadata = {
        "version": LAYOUT_VERSION,
        "algorithm": LAYOUT_ALGORITHM,
        "max_depth": max(index.depth.values(), default=0),
        "bounds": _bounds(coordinates),
    }
    return metadata, coordinates


def _primary_roots(index):
    primary_roots = {}
    for matter in index.topological_order:
        matter_prerequisites = index.prerequisites[matter]
        primary_roots[matter] = (
            min(primary_roots[prerequisite] for prerequisite in matter_prerequisites)
            if matter_prerequisites
            else matter
        )
    return primary_roots


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
