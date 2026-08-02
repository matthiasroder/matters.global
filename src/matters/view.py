"""One matter, one self-contained HTML page.

``matters view X`` answers "what is stopping X, and what waits on it" with a
picture. The page is a single file with no network access of any kind: the
canvas renderer that ``matters web`` serves is inlined verbatim, and the
graph travels with it as literal JSON. Opening it needs no server, and a
copy mailed to someone else still works.

Three decisions shape this module.

**The slice is everything connected to X.** ``ancestors(X) | descendants(X)
| {X}``, because the question is about X's whole predicament, not its
immediate surroundings. ``--depth N`` narrows both directions through
:meth:`~matters.graph_index.GraphIndex.neighborhood`; it exists because one
matter in a real 1057-matter graph can pull in several hundred.

**The slice is laid out on its own.** :func:`~matters.layout.build_overview_layout`
runs over the induced subgraph, not the whole graph, so the picture uses
every pixel it has. The coordinates are therefore not the ones ``matters
web`` shows for the same matters, and are not meant to be.

**A state file with a loop still gets a picture.** ``show`` and ``list``
already guarantee that a broken file stays inspectable, and a picture is
exactly what someone holding a tangled graph wants. ``view`` degrades to
structure only -- no resolved/actionable status, ring placement instead of
the solar-system layout -- and says so on the page. It does not become a
fourth read verb that refuses.

Status is always read from the index over the **whole** graph, never from
the slice's own index. Under ``--depth`` a matter can lose the prerequisite
that blocks it, and an index of the slice alone would then call it
actionable; only the layout is allowed to be slice-local.
"""

from __future__ import annotations

import html
import json
import math
import webbrowser
from importlib import resources
from pathlib import Path

from . import rules
from .graph_index import induced_subgraph
from .layout import build_overview_layout
from .storage import resolve_state_path


RENDERER_ASSET = "map-renderer.js"
RENDERER_EXPORT = "export function createOverviewRenderer("

# The degraded placement used when the graph cannot be indexed. Deliberately
# dull: concentric rings by hop distance from the matter, evenly spaced,
# no hashing and no packing. It has to be readable by whoever is repairing a
# cyclic file, which means it has to be obvious rather than good.
FALLBACK_ALGORITHM = "hop-rings-v1"
FALLBACK_VERSION = 1
RING_STEP = 96.0
RING_ARC_GAP = 30.0
RING_ELEVATION = 8.0
RING_PADDING = 54.0


def load_view_payload(matter, *, state_path=None, depth=None):
    """Load the state file and build the payload for ``matter``."""

    path = resolve_state_path(state_path)
    matters, conditions, dependencies = rules.load_state_or_rule_error(path)
    return build_view_payload(
        matter, matters, conditions, dependencies, depth=depth, state_path=path
    )


def build_view_payload(
    matter, matters, conditions, dependencies, *, depth=None, state_path=None
):
    """Return everything the page needs, as plain JSON-serialisable data.

    The node shape mirrors ``web.graph_payload``'s, because the renderer
    reading it is the same one: ``id``, ``label``, ``resolved``,
    ``actionable`` and an ``overview`` coordinate block. ``label`` is the id
    with underscores opened out, the way the web payload does it -- there is
    still no display-name field to read.

    ``prerequisites`` and ``dependents`` on each node are the edges **inside
    the slice**. An edge leaving the slice is not in the picture, so listing
    it in the panel would point at something the reader cannot see.

    When the graph cannot be indexed, ``status_available`` is ``False``,
    ``cycle`` names one concrete loop, and every ``resolved``/``actionable``/
    ``blocked`` field is ``None`` -- unknown, not false.
    """

    rules.require_matter(matter, matters)
    require_depth(depth)

    index, cycle = index_or_cycle(matters, conditions, dependencies)
    if index is None:
        ids, hops = structural_slice(matter, matters, dependencies, depth)
    else:
        ids = index.neighborhood(matter, up=depth, down=depth) | {matter}
        hops = None

    slice_matters, slice_conditions, slice_dependencies = induced_subgraph(
        matters, conditions, dependencies, ids
    )
    prerequisite_map, dependent_map = adjacency(slice_matters, slice_dependencies)

    if index is None:
        overview_layout, coordinates = fallback_layout(matter, hops, dependent_map)
        ancestors = sorted(upstream_of(matter, prerequisite_map))
    else:
        overview_layout, coordinates = build_overview_layout(
            rules.build_index(slice_matters, slice_conditions, slice_dependencies)
        )
        ancestors = sorted(index.ancestors(matter) & slice_matters)

    nodes = [
        {
            "id": candidate,
            "label": candidate.replace("_", " "),
            "conditions": slice_conditions.get(candidate, []),
            "prerequisites": sorted(prerequisite_map[candidate]),
            "dependents": sorted(dependent_map[candidate]),
            "resolved": None if index is None else index.resolved[candidate],
            "actionable": None if index is None else candidate in index.universe,
            "blocked": (
                None
                if index is None
                else not index.resolved[candidate]
                and candidate not in index.universe
            ),
            "overview": coordinates[candidate],
        }
        for candidate in sorted(slice_matters)
    ]

    return {
        "matter": matter,
        "state_path": str(resolve_state_path(state_path)),
        "depth": depth,
        "status_available": index is not None,
        "cycle": list(cycle),
        "nodes": nodes,
        "edges": [
            {"source": source, "target": target}
            for source, target in sorted(slice_dependencies)
        ],
        "ancestors": ancestors,
        "direct_dependents": sorted(dependent_map[matter]),
        # What is stopping this matter: the part of its ancestry that is not
        # resolved yet. Empty when the status is unknown, which is honest --
        # a cyclic file cannot say what blocks what.
        "blocking": (
            []
            if index is None
            else [
                ancestor for ancestor in ancestors if not index.resolved[ancestor]
            ]
        ),
        "overview_layout": overview_layout,
    }


def index_or_cycle(matters, conditions, dependencies):
    """Return ``(index, ())`` or ``(None, cycle)`` for a graph with a loop.

    The refusal comes from ``rules.build_index``, so the loop this page
    names is the same loop every other surface names.
    """

    try:
        return rules.build_index(matters, conditions, dependencies), ()
    except rules.RuleError as error:
        if error.code != "state_cycle":
            raise
        return None, error.cycle


def require_depth(depth):
    """Reject a negative hop limit the way ``neighborhood`` does.

    Checked up front so both the indexed and the degraded path refuse it
    identically, instead of one raising and the other quietly walking zero
    hops.
    """

    if depth is not None and depth < 0:
        raise ValueError(f"depth must not be negative: {depth}")


def adjacency(matters, dependencies):
    """Return ``(prerequisites, dependents)`` maps over an edge set."""

    prerequisite_map = {matter: set() for matter in matters}
    dependent_map = {matter: set() for matter in matters}
    for source, target in dependencies:
        dependent_map[source].add(target)
        prerequisite_map[target].add(source)
    return prerequisite_map, dependent_map


# ---------------------------------------------------------------------------
# The degraded path: a graph with a loop cannot be indexed
# ---------------------------------------------------------------------------


def structural_slice(matter, matters, dependencies, depth):
    """Return ``(ids, hops)`` for a graph that has no index.

    Mirrors :meth:`~matters.graph_index.GraphIndex.neighborhood`: the two
    directions are walked separately, so ``depth=1`` reaches direct
    prerequisites and direct dependents but not a sibling one hop up and one
    back down. ``hops`` is the shortest distance in either direction and is
    what the ring placement reads.
    """

    prerequisite_map, dependent_map = adjacency(matters, dependencies)
    upward = walk(matter, prerequisite_map, depth)
    downward = walk(matter, dependent_map, depth)
    hops = {
        candidate: min(
            upward.get(candidate, math.inf), downward.get(candidate, math.inf)
        )
        for candidate in upward.keys() | downward.keys()
    }
    return set(hops), hops


def walk(matter, adjacency_map, limit):
    """Breadth-first hop counts along one direction, at most ``limit`` hops.

    Cycle-safe because a matter keeps the first hop count it is reached
    with and is never queued twice; ``limit=None`` therefore terminates on a
    looped graph, which is the whole point of this function existing next to
    the indexed one.
    """

    distances = {matter: 0}
    frontier = [matter]
    hops = 0
    while frontier and (limit is None or hops < limit):
        hops += 1
        following = []
        for node in frontier:
            for neighbour in adjacency_map.get(node, ()):
                if neighbour not in distances:
                    distances[neighbour] = hops
                    following.append(neighbour)
        frontier = following
    return distances


def upstream_of(matter, prerequisite_map):
    """Everything reachable backwards from ``matter``, excluding itself."""

    reached = walk(matter, prerequisite_map, None)
    reached.pop(matter, None)
    return set(reached)


def fallback_layout(matter, hops, dependent_map):
    """Place the slice on concentric rings by hop distance from ``matter``.

    The matter sits at the origin and is the only sun, so the renderer's
    level-of-detail rule and its system boundary keep working. Ring radii
    grow with distance and with how many matters share the ring; angles are
    an even division of the circle in id order. No hashing, no packing, no
    prerequisite direction -- a graph with a loop has no depth to encode, and
    pretending otherwise would draw a picture that is wrong rather than
    plain.

    The metadata deliberately carries no ``bounds``: the renderer derives its
    own from the coordinates, and copying :mod:`matters.layout`'s private
    bounds helper here would be a second implementation of it.
    """

    rings = {}
    for candidate, hop in sorted(hops.items()):
        rings.setdefault(hop, []).append(candidate)

    radii = {}
    previous = 0.0
    for hop in sorted(rings):
        if hop == 0:
            radii[hop] = 0.0
            continue
        radius = max(
            hop * RING_STEP,
            len(rings[hop]) * RING_ARC_GAP / math.tau,
            previous + RING_STEP / 2,
        )
        radii[hop] = radius
        previous = radius

    outer_radius = max(radii.values(), default=0.0)
    coordinates = {}
    for hop, members in rings.items():
        radius = radii[hop]
        for position, candidate in enumerate(members):
            angle = position * math.tau / len(members)
            coordinates[candidate] = {
                "x": round(radius * math.cos(angle), 3),
                "y": round(hop * RING_ELEVATION, 3),
                "z": round(radius * math.sin(angle), 3),
                "depth": hop,
                "orbit_level": hop,
                # Direct dependents only. The exact transitive count is what
                # an index is for, and there is none.
                "downstream_impact": len(dependent_map[candidate]),
                "system": matter,
                "system_count": 1,
                "system_population": len(hops),
                "system_radius": round(outer_radius + RING_PADDING, 3),
                "orbit_radius": round(radius, 3),
            }

    metadata = {
        "version": FALLBACK_VERSION,
        "algorithm": FALLBACK_ALGORITHM,
        "max_depth": max(rings, default=0),
        "system_count": 1,
    }
    return metadata, coordinates


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def read_renderer_source():
    """Return ``map-renderer.js`` as an inlinable script.

    The file is inlined verbatim except for the one ``export`` keyword: it
    has no imports and makes no network call of any kind, which is what lets
    the page stand on its own. ``app.js`` is deliberately not touched -- it
    fetches from the local API and pulls modules from a CDN, and none of it
    is needed to draw a graph that is already in the file.

    Both checks below are load-bearing rather than defensive: a renderer
    that grew a second ``export`` would make the page a syntax error, and
    silently emitting that is worse than refusing here.
    """

    source = (
        resources.files("matters")
        .joinpath("web_assets")
        .joinpath(RENDERER_ASSET)
        .read_text(encoding="utf-8")
    )
    if RENDERER_EXPORT not in source:
        raise ValueError(
            f"{RENDERER_ASSET} no longer exports createOverviewRenderer"
        )
    inlined = source.replace(
        RENDERER_EXPORT, RENDERER_EXPORT.removeprefix("export "), 1
    )
    if "\nexport " in inlined or inlined.startswith("export "):
        raise ValueError(f"{RENDERER_ASSET} has an export this page cannot inline")
    return inlined


def render_view_html(payload):
    """Return the whole page as one string of HTML."""

    matter = payload["matter"]
    heading = matter.replace("_", " ")
    slice_summary = (
        f"{count_label(len(payload['nodes']), 'matter', 'matters')}, "
        f"{count_label(len(payload['edges']), 'dependency', 'dependencies')}"
    )
    scope = (
        "everything connected"
        if payload["depth"] is None
        else f"within {count_label(payload['depth'], 'hop', 'hops')}"
    )
    meta = f"{matter} · {slice_summary} · {scope} · {payload['state_path']}"

    if payload["status_available"]:
        banner = ""
    else:
        banner = (
            '<p class="banner">Structure only. This state file contains a '
            f"cycle ({html.escape(rules.format_cycle(payload['cycle']))}), so "
            "no matter can be called resolved, actionable or blocked, and the "
            "matters are placed on plain hop rings instead of the usual "
            "layout. Break the cycle and this page regains both.</p>"
        )

    return (
        PAGE_TEMPLATE.replace("__TITLE__", html.escape(f"matters view: {matter}"))
        .replace("__HEADING__", html.escape(heading))
        .replace("__META__", html.escape(meta))
        .replace("__BANNER__", banner)
        .replace("__PAYLOAD__", embedded_json(payload))
        .replace("__RENDERER__", read_renderer_source())
    )


def embedded_json(payload):
    """Serialise ``payload`` so it cannot end the script element early.

    ``json.dumps`` escapes nothing HTML cares about, and a condition label is
    free text: one ``</script>`` in one label would otherwise close the
    element and spill the rest of the graph into the document as markup.
    """

    return (
        json.dumps(payload)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def count_label(count, singular, plural):
    return f"{count} {singular if count == 1 else plural}"


def view_output_path(matter, output=None):
    """Where the page goes: ``--output`` if given, else the working directory.

    The working directory rather than a temporary one on purpose. A file
    under ``/tmp`` with a predictable name is both hard to find again and a
    thing another user on the machine can plant a symlink for; a file the
    caller can see, and whose path the command prints, is neither.
    """

    if output:
        return Path(output).expanduser()
    return Path.cwd() / f"matters-view-{matter}.html"


def write_view(matter, *, state_path=None, depth=None, output=None):
    """Build the page for ``matter`` and write it. Returns ``(path, payload)``.

    Writes exactly one file and takes no lock. ``view`` is a read: it must
    not go through ``rules.state_transaction``, must not leave a ``.lock``
    sidecar next to the state file, and must not change a byte of it.
    """

    payload = load_view_payload(matter, state_path=state_path, depth=depth)
    path = view_output_path(matter, output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_view_html(payload), encoding="utf-8")
    return path, payload


def open_view(path):
    """Open a written page in the browser as a ``file://`` URL."""

    return webbrowser.open(Path(path).resolve().as_uri())


# The page itself. Placeholders are ``__NAME__`` and substituted by
# ``str.replace``: the document holds CSS and JavaScript, both full of
# braces, so ``str.format`` and f-strings are not usable here.
#
# Nothing in this template addresses the network. No stylesheet link, no
# script src, no font, no image: the only two script elements are the
# inlined renderer and the code below, and the graph is a literal. That
# property is pinned by a test, because it is the whole promise of the file.
PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --paper: #f7f1e4;
    --ink: #28302f;
    --muted: #68736e;
    --line: rgba(104, 96, 81, .3);
    --gold: #c2933e;
    --actionable: #2f7f5a;
    --blocked: #b85f49;
    --resolved: #687b82;
    --derived: #54737c;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--paper);
    color: var(--ink);
    display: flex;
    flex-direction: column;
    font: 15px/1.5 ui-serif, Georgia, serif;
  }
  header {
    border-bottom: 1px solid var(--line);
    flex: none;
    padding: 14px 20px;
  }
  h1 { font-size: 20px; font-weight: 600; margin: 0; }
  .meta {
    color: var(--muted);
    font: 12px/1.6 ui-sans-serif, system-ui, sans-serif;
    margin: 4px 0 0;
    word-break: break-word;
  }
  .banner {
    background: rgba(184, 95, 73, .1);
    border-left: 3px solid var(--blocked);
    font: 13px/1.5 ui-sans-serif, system-ui, sans-serif;
    margin: 10px 0 0;
    padding: 8px 12px;
  }
  main { display: flex; flex: 1; min-height: 0; }
  #map { flex: 1; min-width: 0; position: relative; }
  aside {
    border-left: 1px solid var(--line);
    flex: none;
    font: 13px/1.55 ui-sans-serif, system-ui, sans-serif;
    overflow-y: auto;
    padding: 14px 16px;
    width: 310px;
  }
  aside h2 {
    font: 600 12px/1.4 ui-sans-serif, system-ui, sans-serif;
    letter-spacing: .07em;
    margin: 18px 0 6px;
    text-transform: uppercase;
    color: var(--muted);
  }
  aside h2:first-of-type { margin-top: 0; }
  .controls { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
  button {
    background: #fffdf6;
    border: 1px solid var(--line);
    border-radius: 4px;
    color: inherit;
    cursor: pointer;
    font: inherit;
    padding: 4px 10px;
  }
  button[aria-pressed="true"] { background: var(--gold); color: #fffdf6; }
  ul { list-style: none; margin: 0; padding: 0; }
  li { margin: 0 0 3px; overflow-wrap: anywhere; }
  .swatch {
    border-radius: 50%;
    display: inline-block;
    height: 9px;
    margin-right: 7px;
    vertical-align: baseline;
    width: 9px;
  }
  .empty { color: var(--muted); }
  .count { color: var(--muted); font-variant-numeric: tabular-nums; }
  #detail dt {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: .06em;
    text-transform: uppercase;
  }
  #detail dd { margin: 2px 0 10px; overflow-wrap: anywhere; }
  #detail dl { margin: 0; }
  .error { color: var(--blocked); }
</style>
</head>
<body>
<header>
  <h1>__HEADING__</h1>
  <p class="meta">__META__</p>
  __BANNER__
</header>
<main>
  <section id="map" aria-label="Dependency map"></section>
  <aside>
    <div class="controls">
      <button id="mode-blockers" type="button" aria-pressed="true">Blockers</button>
      <button id="mode-plain" type="button" aria-pressed="false">Whole slice</button>
      <button id="reset" type="button">Reset view</button>
    </div>
    <div id="detail"></div>
    <h2>Legend</h2>
    <ul id="legend"></ul>
  </aside>
</main>
<script>
__RENDERER__
</script>
<script>
const VIEW = __PAYLOAD__;
const byId = new Map(VIEW.nodes.map((node) => [node.id, node]));
const detail = document.getElementById("detail");
const legend = document.getElementById("legend");
const buttons = {
  blockers: document.getElementById("mode-blockers"),
  plain: document.getElementById("mode-plain")
};
// The renderer has three status colours and no fourth for "unknown". On a
// state file with a loop there is no status to show, so every matter is drawn
// in the quiet resolved grey and the legend says why. The payload keeps
// telling the truth: resolved is null there, not false.
const modelNodes = VIEW.status_available
  ? VIEW.nodes
  : VIEW.nodes.map((node) => ({ ...node, resolved: true }));
let mode = "blockers";
let renderer = null;

function model() {
  const base = { nodes: modelNodes, edges: VIEW.edges, matchIds: [], filterActive: false };
  if (mode === "plain") {
    // No selection at all: with nothing emphasised the renderer dims nothing,
    // which is the only way to see the slice evenly.
    return { ...base, selectedId: null, ancestorIds: [], directDependentIds: [], derivedHighlightIds: [] };
  }
  return {
    ...base,
    selectedId: VIEW.matter,
    ancestorIds: VIEW.ancestors,
    directDependentIds: VIEW.direct_dependents,
    derivedHighlightIds: VIEW.blocking
  };
}

function setMode(next) {
  mode = next;
  buttons.blockers.setAttribute("aria-pressed", String(next === "blockers"));
  buttons.plain.setAttribute("aria-pressed", String(next === "plain"));
  if (renderer) renderer.setModel(model());
}

// Not `status`: a top-level `function status` in a classic script collides
// with the window property of that name, and the failure is silent.
function statusText(node) {
  if (!VIEW.status_available) return "status unavailable";
  if (node.resolved) return "resolved";
  if (node.actionable) return "actionable";
  return "blocked";
}

function line(text, className) {
  const item = document.createElement("li");
  if (className) item.className = className;
  item.textContent = text;
  return item;
}

function idList(ids) {
  const list = document.createElement("ul");
  if (!ids.length) {
    list.append(line("none", "empty"));
    return list;
  }
  ids.forEach((id) => list.append(line(id)));
  return list;
}

function field(term, value) {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  if (typeof value === "string") {
    dd.textContent = value;
  } else {
    dd.append(value);
  }
  return [dt, dd];
}

function showDetail(id) {
  detail.replaceChildren();
  const node = id === null ? byId.get(VIEW.matter) : byId.get(id);
  if (!node) return;
  const heading = document.createElement("h2");
  heading.textContent = node.id === VIEW.matter ? "This matter" : "Selected";
  const list = document.createElement("dl");
  const conditions = document.createElement("ul");
  if (!node.conditions.length) {
    conditions.append(line("none", "empty"));
  } else {
    node.conditions.forEach((condition) => {
      conditions.append(line((condition.truth ? "[x] " : "[ ] ") + condition.label));
    });
  }
  list.append(
    ...field("matter", node.label),
    ...field("id", node.id),
    ...field("status", statusText(node)),
    ...field("conditions", conditions),
    ...field("requires", idList(node.prerequisites)),
    ...field("required by", idList(node.dependents))
  );
  detail.append(heading, list);
}

function swatchLine(color, text) {
  const item = document.createElement("li");
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = color;
  item.append(swatch, document.createTextNode(text));
  return item;
}

function drawLegend() {
  if (VIEW.status_available) {
    legend.append(
      swatchLine("#2f7f5a", "actionable"),
      swatchLine("#b85f49", "blocked"),
      swatchLine("#687b82", "resolved"),
      swatchLine("#54737c", VIEW.blocking.length + " blocking this matter")
    );
  } else {
    legend.append(swatchLine("#687b82", "status unavailable: the graph has a cycle"));
  }
  legend.append(
    swatchLine("#c2933e", VIEW.ancestors.length + " upstream of this matter"),
    swatchLine("#d8e7d9", VIEW.direct_dependents.length + " waiting directly on it")
  );
}

function fail(error) {
  detail.replaceChildren();
  const message = document.createElement("p");
  message.className = "error";
  message.textContent = "The map could not be drawn: " + (error && error.message ? error.message : error);
  detail.append(message);
}

buttons.blockers.addEventListener("click", () => setMode("blockers"));
buttons.plain.addEventListener("click", () => setMode("plain"));
document.getElementById("reset").addEventListener("click", () => {
  if (renderer) renderer.resetCamera();
});

drawLegend();
showDetail(null);
try {
  renderer = createOverviewRenderer({
    container: document.getElementById("map"),
    onSelect: showDetail,
    onError: fail,
    reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches
  });
  renderer.setModel(model());
} catch (error) {
  fail(error);
}
</script>
</body>
</html>
"""
