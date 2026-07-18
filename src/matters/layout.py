"""Deterministic spatial layouts for matters dependency graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math


LAYOUT_VERSION = 3
LAYOUT_ALGORITHM = "solar-systems-v1"
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
ORBIT_STEP = 68.0
ORBIT_MIN_GAP = 52.0
PLANET_ARC_GAP = 24.0
ORBIT_NUDGE = 5.0
SYSTEM_MIN_RADIUS = 88.0
SYSTEM_PADDING = 54.0
SYSTEM_GAP = 92.0
PACK_NUDGE = 36.0
BRIDGE_JITTER = 24.0
ELEVATION_STEP = 8.0


def build_overview_layout(index):
    """Return a deterministic atlas of goal-centered dependency systems.

    Each independent prerequisite family selects its deepest terminal goal as
    a sun. Related branches share that system, their matters occupy orbital
    rings by distance from the goal, and cross-system joins sit between suns.
    Conditions and truth values never affect placement.
    """

    systems, root_systems = _system_definitions(index)
    memberships = _system_memberships(index, systems, root_systems)
    orbit_levels = _orbit_levels(index, memberships)
    orbit_members = _orbit_members(memberships, systems, orbit_levels)
    orbit_radii = _orbit_radii(orbit_members, systems)
    system_radii = {
        system: max(
            SYSTEM_MIN_RADIUS,
            max(orbit_radii[system].values(), default=0.0) + SYSTEM_PADDING,
        )
        for system in systems
    }
    system_centers = _pack_systems(system_radii)
    system_populations = Counter(
        primary_system for primary_system, _ in memberships.values()
    )
    orbit_angles = _orbit_angles(orbit_members)
    coordinates = {}

    for matter in index.topological_order:
        depth = index.depth[matter]
        orbit_level = orbit_levels[matter]
        primary_system, contributing_systems = memberships[matter]
        center_x = sum(system_centers[root][0] for root in contributing_systems)
        center_z = sum(system_centers[root][1] for root in contributing_systems)
        center_x /= len(contributing_systems)
        center_z /= len(contributing_systems)

        if orbit_level == 0:
            x, z = center_x, center_z
            orbit_radius = 0.0
        elif len(contributing_systems) > 1:
            angle = _unit_interval(matter, "bridge-angle") * math.tau
            jitter = _unit_interval(matter, "bridge-radius") * BRIDGE_JITTER
            x = center_x + jitter * math.cos(angle)
            z = center_z + jitter * math.sin(angle)
            orbit_radius = 0.0
        else:
            orbit_radius = orbit_radii[primary_system][orbit_level]
            angle = orbit_angles[matter]
            nudge = (
                _unit_interval(matter, "orbit-nudge") - 0.5
            ) * 2.0 * ORBIT_NUDGE
            radius = orbit_radius + nudge
            x = center_x + radius * math.cos(angle)
            z = center_z + radius * math.sin(angle)

        coordinates[matter] = {
            "x": round(x, 3),
            "y": round(orbit_level * ELEVATION_STEP, 3),
            "z": round(z, 3),
            "depth": depth,
            "orbit_level": orbit_level,
            "downstream_impact": index.downstream_impact[matter],
            "system": primary_system,
            "system_count": len(contributing_systems),
            "system_population": system_populations[primary_system],
            "system_radius": round(system_radii[primary_system], 3),
            "orbit_radius": round(orbit_radius, 3),
        }

    metadata = {
        "version": LAYOUT_VERSION,
        "algorithm": LAYOUT_ALGORITHM,
        "max_depth": max(index.depth.values(), default=0),
        "system_count": len(systems),
        "bounds": _bounds(coordinates),
    }
    return metadata, coordinates


def _system_definitions(index):
    reachable_goals = {}
    for matter in reversed(index.topological_order):
        dependents = index.dependents[matter]
        reachable_goals[matter] = (
            {matter}
            if not dependents
            else set().union(*(reachable_goals[item] for item in dependents))
        )

    roots = (
        matter for matter in index.topological_order if not index.prerequisites[matter]
    )
    root_systems = {
        root: min(
            reachable_goals[root],
            key=lambda goal: (-index.depth[goal], goal),
        )
        for root in roots
    }
    systems = tuple(sorted(set(root_systems.values())))
    return systems, root_systems


def _system_memberships(index, systems, root_systems):
    system_set = set(systems)
    memberships = {}
    for matter in index.topological_order:
        prerequisites = index.prerequisites[matter]
        if matter in system_set:
            contributing_systems = (matter,)
        elif prerequisites:
            contributing_systems = tuple(
                sorted({memberships[item][0] for item in prerequisites})
            )
        else:
            contributing_systems = (root_systems[matter],)
        memberships[matter] = (
            min(contributing_systems),
            contributing_systems,
        )
    return memberships


def _orbit_levels(index, memberships):
    return {
        matter: (
            0
            if matter == primary_system
            else max(1, index.depth[primary_system] - index.depth[matter])
        )
        for matter, (primary_system, _) in memberships.items()
    }


def _orbit_members(memberships, systems, orbit_levels):
    members = {system: defaultdict(list) for system in systems}
    for matter, orbit_level in orbit_levels.items():
        primary_system, contributing_systems = memberships[matter]
        if orbit_level and len(contributing_systems) == 1:
            members[primary_system][orbit_level].append(matter)
    return members


def _orbit_radii(orbit_members, roots):
    radii = {root: {} for root in roots}
    for root in roots:
        previous_radius = 0.0
        for depth in sorted(orbit_members[root]):
            population = len(orbit_members[root][depth])
            population_radius = population * PLANET_ARC_GAP / math.tau
            radius = max(
                depth * ORBIT_STEP,
                population_radius,
                previous_radius + ORBIT_MIN_GAP,
            )
            radii[root][depth] = radius
            previous_radius = radius
    return radii


def _orbit_angles(orbit_members):
    angles = {}
    for root, depths in orbit_members.items():
        for depth, matters in depths.items():
            phase = _unit_interval(f"{root}:{depth}", "orbit-phase") * math.tau
            for position, matter in enumerate(sorted(matters)):
                angles[matter] = (phase + position * GOLDEN_ANGLE) % math.tau
    return angles


def _pack_systems(system_radii):
    if not system_radii:
        return {}

    ordered = sorted(system_radii, key=lambda root: (-system_radii[root], root))
    average_diameter = 2.0 * math.sqrt(
        sum(radius * radius for radius in system_radii.values()) / len(ordered)
    )
    spiral_step = max(SYSTEM_MIN_RADIUS * 2.0, average_diameter * 0.82)
    centers = {}

    for position, root in enumerate(ordered):
        radius = system_radii[root]
        if not centers:
            centers[root] = (0.0, 0.0)
            continue

        angle = position * GOLDEN_ANGLE
        distance = spiral_step * math.sqrt(position)
        while True:
            x = distance * math.cos(angle)
            z = distance * math.sin(angle)
            if all(
                math.hypot(x - other_x, z - other_z)
                >= radius + system_radii[other] + SYSTEM_GAP
                for other, (other_x, other_z) in centers.items()
            ):
                centers[root] = (x, z)
                break
            distance += PACK_NUDGE

    min_x = min(x - system_radii[root] for root, (x, _) in centers.items())
    max_x = max(x + system_radii[root] for root, (x, _) in centers.items())
    min_z = min(z - system_radii[root] for root, (_, z) in centers.items())
    max_z = max(z + system_radii[root] for root, (_, z) in centers.items())
    offset_x = (min_x + max_x) / 2.0
    offset_z = (min_z + max_z) / 2.0
    return {
        root: (x - offset_x, z - offset_z)
        for root, (x, z) in centers.items()
    }


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
