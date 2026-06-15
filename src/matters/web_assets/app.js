import cytoscape from "https://cdn.jsdelivr.net/npm/cytoscape@3.34.0/+esm";
import dagre from "https://cdn.jsdelivr.net/npm/cytoscape-dagre@4.0.0/+esm";
import { FitAddon } from "https://cdn.jsdelivr.net/npm/@xterm/addon-fit@0.10.0/+esm";
import { Terminal } from "https://cdn.jsdelivr.net/npm/@xterm/xterm@5.5.0/+esm";

cytoscape.use(dagre);

const STATUS_COLORS = {
  actionable: "#27835b",
  blocked: "#cc654c",
  resolved: "#687985",
  selected: "#2f68c5",
  edge: "#8c9aa0",
  faded: "#d8d5ca"
};

const GRAPH_VIEW = {
  attentionMaxNodes: 150,
  attentionSeeds: 18,
  fitPadding: 42,
  largeGraphThreshold: 220,
  maxAnimatedNodes: 220,
  minZoom: 0.08,
  maxZoom: 1.35
};

const state = {
  graph: null,
  cy: null,
  scope: "attention",
  scopeIds: null,
  selectedId: null,
  visibleIds: new Set(),
  terminal: null,
  fitAddon: null,
  terminalSessionId: null,
  terminalSeq: 0,
  terminalPollTimer: null,
  terminalPolling: false
};

const graphElement = document.querySelector("#graph");
const inspector = document.querySelector("#inspector");
const operationOutput = document.querySelector("#operation-output");
const statePath = document.querySelector("#state-path");
const emptyState = document.querySelector("#empty-state");
const searchInput = document.querySelector("#search");
const statusFilter = document.querySelector("#status-filter");
const scopeFilter = document.querySelector("#scope-filter");
const stateForm = document.querySelector("#state-form");
const dependencyForm = document.querySelector("#dependency-form");
const terminalDrawer = document.querySelector("#terminal-drawer");
const terminalElement = document.querySelector("#terminal");
const terminalStatus = document.querySelector("#terminal-status");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  const payload = await responsePayload(response);
  if (!response.ok) {
    const error = new Error(apiErrorMessage(response, payload));
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

async function responsePayload(response) {
  const contentType = response.headers.get("Content-Type") || "";
  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }
  return response.text();
}

function apiErrorMessage(response, payload) {
  if (payload && typeof payload === "object" && payload.error) {
    return payload.error;
  }

  const status = [response.status, response.statusText].filter(Boolean).join(" ");
  return `Request failed: ${status || "unknown error"}`;
}

async function loadGraph() {
  state.graph = await api("/api/state");
  render();
}

function initGraph() {
  state.cy = cytoscape({
    container: graphElement,
    elements: [],
    minZoom: GRAPH_VIEW.minZoom,
    maxZoom: GRAPH_VIEW.maxZoom,
    boxSelectionEnabled: false,
    autoungrabify: false,
    style: graphStyles()
  });

  state.cy.on("tap", "node", (event) => {
    focusNode(event.target.id());
    renderInspector();
    updateOperationButtons();
  });

  state.cy.on("tap", (event) => {
    if (event.target !== state.cy) return;
    state.selectedId = null;
    setScope("attention");
    renderInspector();
    updateOperationButtons();
  });

  window.addEventListener("resize", resizeGraph);
}

function graphStyles() {
  return [
    {
      selector: "node",
      style: {
        "background-color": STATUS_COLORS.blocked,
        "border-color": "#fffef9",
        "border-opacity": 0.96,
        "border-width": 2,
        color: "#24292f",
        "font-family": "Inter, system-ui, sans-serif",
        "font-size": 11,
        "font-weight": 700,
        height: "data(size)",
        label: "data(labelText)",
        "min-zoomed-font-size": 7,
        opacity: 0.96,
        "overlay-opacity": 0,
        shape: "ellipse",
        "text-background-color": "#fffdf8",
        "text-background-opacity": 0.86,
        "text-background-padding": 3,
        "text-border-color": "#e8dfd0",
        "text-border-opacity": 0.9,
        "text-border-width": 1,
        "text-halign": "right",
        "text-margin-x": 7,
        "text-max-width": 150,
        "text-wrap": "wrap",
        "text-valign": "center",
        width: "data(size)",
        "z-index": 10
      }
    },
    {
      selector: "node.actionable",
      style: {
        "background-color": STATUS_COLORS.actionable
      }
    },
    {
      selector: "node.resolved",
      style: {
        "background-color": STATUS_COLORS.resolved
      }
    },
    {
      selector: "node.blocked",
      style: {
        "background-color": STATUS_COLORS.blocked
      }
    },
    {
      selector: "node.selected",
      style: {
        "background-color": STATUS_COLORS.selected,
        "border-color": "#153f91",
        "border-width": 4,
        "font-size": 12,
        height: "data(selectedSize)",
        "text-background-color": "#eef6ff",
        "text-border-color": "#bdd3f7",
        width: "data(selectedSize)",
        "z-index": 40
      }
    },
    {
      selector: "node.dimmed",
      style: {
        opacity: 0.16,
        "text-opacity": 0.18
      }
    },
    {
      selector: "edge",
      style: {
        "curve-style": "bezier",
        "line-color": STATUS_COLORS.edge,
        opacity: 0.58,
        "overlay-opacity": 0,
        "target-arrow-color": STATUS_COLORS.edge,
        "target-arrow-shape": "triangle",
        "target-distance-from-node": 2,
        width: 1.35,
        "z-index": 1
      }
    },
    {
      selector: "edge.focused",
      style: {
        "line-color": STATUS_COLORS.selected,
        opacity: 0.92,
        "target-arrow-color": STATUS_COLORS.selected,
        width: 2.25,
        "z-index": 30
      }
    },
    {
      selector: "edge.dimmed",
      style: {
        "line-color": STATUS_COLORS.faded,
        opacity: 0.1,
        "target-arrow-color": STATUS_COLORS.faded
      }
    },
    {
      selector: ".filtered",
      style: {
        display: "none"
      }
    }
  ];
}

function resizeGraph() {
  if (!state.cy) return;
  state.cy.resize();
  window.requestAnimationFrame(() => {
    fitGraphToVisible();
  });
}

function render() {
  syncStatePathControl();
  if (state.graph.nodes.length <= GRAPH_VIEW.largeGraphThreshold && state.scope === "attention") {
    state.scope = "all";
  }
  syncScopeControl();
  recomputeVisibleIds();
  renderFiltersAndSelectors();
  renderGraph();
  renderInspector();
  updateOperationButtons();
  emptyState.hidden = state.graph.nodes.length > 0;
}

function syncStatePathControl() {
  statePath.textContent = state.graph.state_path;
  const pathInput = stateForm.elements.state_path;
  if (document.activeElement !== pathInput) {
    pathInput.value = state.graph.state_path;
  }
}

function syncScopeControl() {
  if (document.activeElement !== scopeFilter) {
    scopeFilter.value = state.scope;
  }
}

function setScope(scope, ids = null) {
  state.scope = normalizedScope(scope);
  state.scopeIds = ids ? new Set(ids) : null;
  syncScopeControl();
  recomputeVisibleIds();
  refreshGraphStyles({ layout: true });
}

function normalizedScope(scope) {
  if (scope === "attention" && state.graph?.nodes.length <= GRAPH_VIEW.largeGraphThreshold) {
    return "all";
  }
  return scope;
}

function recomputeVisibleIds() {
  state.visibleIds = new Set(filteredNodes().map((node) => node.id));
}

function renderGraph() {
  if (!state.cy) return;
  state.cy.batch(() => {
    state.cy.elements().remove();
    state.cy.add(toCytoscapeElements());
  });
  refreshGraphStyles({ layout: true });
}

function toCytoscapeElements() {
  const nodes = state.graph.nodes.map((node) => ({
    data: nodeData(node),
    classes: statusClass(node)
  }));
  const edges = state.graph.edges.map((edge, index) => ({
    data: {
      id: `edge-${index}`,
      source: edge.source,
      target: edge.target
    }
  }));
  return [...nodes, ...edges];
}

function nodeData(node) {
  const size = nodeSize(node);
  return {
    ...node,
    degree: graphDegree(node.id),
    falseConditions: node.conditions.filter((condition) => !condition.truth).length,
    labelText: graphLabelForNode(node),
    selectedSize: Math.max(size + 10, 30),
    size
  };
}

function refreshGraphStyles(options = {}) {
  if (!state.cy || !state.graph) return;
  const connected = connectedSet();
  const selected = Boolean(state.selectedId);

  state.cy.batch(() => {
    state.cy.nodes().forEach((element) => {
      const node = nodeById(element.id());
      if (!node) return;
      const visible = nodeVisible(node.id);
      element.data(nodeData(node));
      element.toggleClass("filtered", !visible);
      element.toggleClass("selected", node.id === state.selectedId);
      element.toggleClass("dimmed", selected && !connected.has(node.id));
      element.toggleClass("actionable", statusClass(node) === "actionable");
      element.toggleClass("blocked", statusClass(node) === "blocked");
      element.toggleClass("resolved", statusClass(node) === "resolved");
    });

    state.cy.edges().forEach((element) => {
      const link = { source: element.data("source"), target: element.data("target") };
      const visible = linkVisible(link);
      const focused = linkTouchesSelection(link);
      element.toggleClass("filtered", !visible);
      element.toggleClass("focused", selected && focused);
      element.toggleClass("dimmed", selected && !focused);
    });
  });

  if (options.layout) {
    runGraphLayout();
  }
}

function runGraphLayout() {
  if (!state.cy) return;
  const visible = state.cy.elements(":visible");
  if (!visible.length) return;
  const layout = visible.layout(graphLayoutOptions());
  layout.run();
}

function graphLayoutOptions() {
  const visibleNodeCount = state.cy.nodes(":visible").length;
  if (shouldUseOverviewLayout(visibleNodeCount)) {
    return {
      name: "cose",
      animate: false,
      componentSpacing: 56,
      coolingFactor: 0.96,
      edgeElasticity: 88,
      fit: true,
      gravity: 0.42,
      idealEdgeLength: 46,
      initialTemp: 180,
      minTemp: 1,
      nestingFactor: 1.15,
      nodeOverlap: 10,
      nodeRepulsion: 5200,
      numIter: 900,
      padding: GRAPH_VIEW.fitPadding,
      randomize: true
    };
  }

  return {
    name: "dagre",
    rankDir: "LR",
    ranker: "network-simplex",
    nodeSep: visibleNodeCount > 260 ? 16 : 44,
    edgeSep: visibleNodeCount > 260 ? 6 : 12,
    rankSep: visibleNodeCount > 260 ? 40 : 86,
    avoidOverlap: true,
    nodeDimensionsIncludeLabels: true,
    animate: visibleNodeCount > 0 && visibleNodeCount <= GRAPH_VIEW.maxAnimatedNodes,
    animationDuration: 240,
    fit: true,
    padding: GRAPH_VIEW.fitPadding
  };
}

function shouldUseOverviewLayout(visibleNodeCount) {
  if (visibleNodeCount > 260) return true;
  if (state.scope === "custom" && visibleNodeCount > 8) return true;
  return state.scope !== "custom" && visibleNodeCount > 48;
}

function fitGraphToVisible() {
  if (!state.cy) return;
  const visible = state.cy.elements(":visible");
  if (visible.length) {
    state.cy.fit(visible, GRAPH_VIEW.fitPadding);
  }
}

function zoomGraph(factor) {
  if (!state.cy) return;
  const rect = graphElement.getBoundingClientRect();
  const renderedPosition = { x: rect.width / 2, y: rect.height / 2 };
  const level = Math.min(
    GRAPH_VIEW.maxZoom,
    Math.max(GRAPH_VIEW.minZoom, state.cy.zoom() * factor)
  );
  state.cy.animate(
    {
      zoom: { level, renderedPosition }
    },
    { duration: 140 }
  );
}

function nodeVisible(id) {
  return state.visibleIds.has(id);
}

function linkVisible(link) {
  return state.visibleIds.has(link.source) && state.visibleIds.has(link.target);
}

function linkTouchesSelection(link) {
  return link.source === state.selectedId || link.target === state.selectedId;
}

function renderFiltersAndSelectors() {
  const selectedValues = {
    source: dependencyForm.elements.source.value,
    target: dependencyForm.elements.target.value
  };

  ["source", "target"].forEach((name) => {
    const select = dependencyForm.elements[name];
    select.replaceChildren();
    state.graph.nodes.forEach((node) => {
      const option = document.createElement("option");
      option.value = node.id;
      option.textContent = node.id;
      select.append(option);
    });
  });

  dependencyForm.elements.source.value =
    selectedValues.source || state.selectedId || state.graph.nodes[0]?.id || "";
  dependencyForm.elements.target.value =
    selectedValues.target || state.graph.nodes.find((node) => node.id !== dependencyForm.elements.source.value)?.id || "";
}

function renderInspector() {
  const node = currentNode();
  if (!node) {
    inspector.className = "inspector muted";
    inspector.textContent = "Select a node to inspect it.";
    return;
  }

  inspector.className = "inspector";
  inspector.replaceChildren();

  const title = document.createElement("h3");
  title.textContent = node.id;

  const badges = document.createElement("div");
  badges.className = "badge-row";
  const badge = document.createElement("span");
  badge.className = `badge ${statusClass(node)}`;
  badge.textContent = statusLabel(node);
  badges.append(badge);

  const conditionTitle = smallHeading("Conditions");
  const conditionList = document.createElement("div");
  conditionList.className = "condition-list";
  node.conditions.forEach((condition, index) => {
    const item = document.createElement("div");
    item.className = "condition-item";
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.textContent = condition.truth ? "True" : "False";
    toggle.className = condition.truth ? "secondary" : "";
    toggle.addEventListener("click", () => toggleCondition(node.id, index));
    const label = document.createElement("input");
    label.value = condition.label;
    label.setAttribute("aria-label", "Condition label");
    const save = document.createElement("button");
    save.type = "button";
    save.textContent = "Save";
    save.addEventListener("click", () => {
      updateCondition(node.id, index, label.value, condition.truth);
    });
    item.append(toggle, label, save);
    conditionList.append(item);
  });

  const addForm = document.createElement("form");
  addForm.className = "stack";
  addForm.innerHTML = `
    <input name="label" placeholder="New condition">
    <button type="submit">Add condition</button>
  `;
  addForm.addEventListener("submit", (event) => {
    event.preventDefault();
    addCondition(node.id, addForm.elements.label.value);
    addForm.reset();
  });

  const links = document.createElement("div");
  links.className = "link-list";
  links.append(smallHeading("Prerequisites"));
  links.append(...linkSpans(node.prerequisites));
  links.append(smallHeading("Dependents"));
  links.append(...linkSpans(node.dependents));

  inspector.append(title, badges, conditionTitle, conditionList, addForm, links);
}

function smallHeading(text) {
  const heading = document.createElement("strong");
  heading.textContent = text;
  return heading;
}

function linkSpans(ids) {
  if (!ids.length) {
    const empty = document.createElement("span");
    empty.textContent = "none";
    return [empty];
  }
  return ids.map((id) => {
    const span = document.createElement("span");
    span.textContent = id;
    return span;
  });
}

function filteredNodes() {
  const query = searchInput.value.trim().toLowerCase();
  const status = statusFilter.value;
  const scopedIds = baseScopeIds();
  return state.graph.nodes.filter((node) => {
    const matchesScope = scopedIds.has(node.id);
    const matchesText =
      !query ||
      node.id.toLowerCase().includes(query) ||
      node.conditions.some((condition) => condition.label.toLowerCase().includes(query));
    const matchesStatus =
      status === "all" ||
      (status === "actionable" && node.actionable) ||
      (status === "blocked" && node.blocked) ||
      (status === "resolved" && node.resolved) ||
      (status === "unresolved" && !node.resolved);
    return matchesScope && matchesText && matchesStatus;
  });
}

function baseScopeIds() {
  if (!state.graph) return new Set();
  if (state.scope === "custom" && state.scopeIds) return new Set(state.scopeIds);
  if (state.scope === "all") return allMatterIds();
  if (state.scope === "universe") return universeContextIds();
  return attentionScopeIds();
}

function allMatterIds() {
  return new Set(state.graph.nodes.map((node) => node.id));
}

function attentionScopeIds() {
  if (state.graph.nodes.length <= GRAPH_VIEW.largeGraphThreshold) return allMatterIds();

  const ids = new Set();
  const seeds = state.graph.nodes
    .filter((node) => node.actionable)
    .sort((a, b) => downstreamCount(b.id) - downstreamCount(a.id) || a.id.localeCompare(b.id))
    .slice(0, GRAPH_VIEW.attentionSeeds);

  seeds.forEach((node) => {
    ids.add(node.id);
    sortedDependents(node.id).forEach((id) => {
      if (ids.size < GRAPH_VIEW.attentionMaxNodes) ids.add(id);
    });
  });

  if (!ids.size) {
    state.graph.nodes
      .slice(0, Math.min(GRAPH_VIEW.attentionMaxNodes, state.graph.nodes.length))
      .forEach((node) => ids.add(node.id));
  }

  return ids;
}

function universeContextIds() {
  const ids = new Set(state.graph.universe || []);
  (state.graph.universe || []).forEach((id) => {
    sortedDependents(id).forEach((dependent) => ids.add(dependent));
  });
  return ids.size ? ids : attentionScopeIds();
}

function focusContextIds(id) {
  const ids = new Set([id]);
  const node = nodeById(id);
  if (!node) return ids;
  node.prerequisites.forEach((matterId) => ids.add(matterId));
  sortedDependents(id).forEach((matterId) => ids.add(matterId));
  return ids;
}

function downstreamCount(id) {
  const node = nodeById(id);
  return node ? node.dependents.length : 0;
}

function sortedDependents(id) {
  const node = nodeById(id);
  if (!node) return [];
  return [...node.dependents].sort((a, b) => downstreamCount(b) - downstreamCount(a) || a.localeCompare(b));
}

function focusNode(id) {
  state.selectedId = id;
  setScope("custom", focusContextIds(id));
}

function connectedSet() {
  if (!state.selectedId) return new Set(state.graph.nodes.map((node) => node.id));
  const connected = new Set([state.selectedId]);
  state.graph.edges.forEach((edge) => {
    if (edge.source === state.selectedId) connected.add(edge.target);
    if (edge.target === state.selectedId) connected.add(edge.source);
  });
  return connected;
}

function currentNode() {
  return nodeById(state.selectedId);
}

function nodeById(id) {
  return state.graph?.nodes.find((node) => node.id === id);
}

function nodeSize(node) {
  const base = node.resolved ? 16 : 18;
  const actionableBoost = node.actionable ? 3 : 0;
  const degreeBoost = Math.min(11, Math.sqrt(graphDegree(node.id)) * 3.2);
  return Math.round(base + actionableBoost + degreeBoost);
}

function graphDegree(id) {
  if (!id || !state.graph) return 0;
  return state.graph.edges.reduce((total, edge) => {
    return total + (edge.source === id || edge.target === id ? 1 : 0);
  }, 0);
}

function statusClass(node) {
  if (node.resolved) return "resolved";
  if (node.actionable) return "actionable";
  return "blocked";
}

function statusLabel(node) {
  if (node.resolved) return "resolved";
  if (node.actionable) return "actionable";
  return "blocked";
}

function graphLabelForNode(node) {
  if (!node) return "";
  const connected = connectedSet();
  if (node.id === state.selectedId) return truncateLabel(node.label, 68);
  if (state.selectedId && connected.has(node.id)) return truncateLabel(node.label, 48);
  if (searchInput.value.trim()) return truncateLabel(node.label, 48);
  if (state.visibleIds.size <= 80) return truncateLabel(node.label, 42);
  if (node.actionable || graphDegree(node.id) >= 10) return truncateLabel(node.label, 36);
  return "";
}

function truncateLabel(label, maxLength) {
  return label.length > maxLength ? `${label.slice(0, maxLength - 3)}...` : label;
}

function updateOperationButtons() {
  document.querySelector("#show-frontier").disabled = !state.selectedId;
  document.querySelector("#show-horizon").disabled = !state.selectedId;
}

async function toggleCondition(matterId, index) {
  await api(`/api/matters/${encodeURIComponent(matterId)}/conditions`, {
    method: "PATCH",
    body: JSON.stringify({ action: "toggle", index })
  });
  await loadGraph();
}

async function addCondition(matterId, label) {
  if (!label.trim()) return;
  await api(`/api/matters/${encodeURIComponent(matterId)}/conditions`, {
    method: "PATCH",
    body: JSON.stringify({ label: label.trim(), truth: false })
  });
  await loadGraph();
}

async function updateCondition(matterId, index, label, conditionTruth) {
  if (!label.trim()) return;
  await api(`/api/matters/${encodeURIComponent(matterId)}/conditions`, {
    method: "PATCH",
    body: JSON.stringify({ index, label: label.trim(), truth: conditionTruth })
  });
  await loadGraph();
}

async function runCommand(text) {
  try {
    const result = await api("/api/command", {
      method: "POST",
      body: JSON.stringify({ text })
    });
    setOperationOutput(result.type || "matters", formatCommandResult(result));
    if (result.state) {
      state.graph = result.state;
      render();
    } else {
      await loadGraph();
    }
  } catch (error) {
    setOperationOutput("error", error.message);
  }
}

async function showDerivedScope(kind, matterId) {
  try {
    const result = await api("/api/command", {
      method: "POST",
      body: JSON.stringify({ text: `${kind} ${matterId}` })
    });
    const ids = kind === "horizon"
      ? downstreamContextIds(matterId, result.items)
      : new Set([matterId, ...result.items]);
    state.selectedId = matterId;
    setScope("custom", ids);
    renderInspector();
    updateOperationButtons();
    setOperationOutput(result.type || kind, formatCommandResult(result));
  } catch (error) {
    setOperationOutput("error", error.message);
  }
}

function downstreamContextIds(rootId, targetIds) {
  const targets = new Set(targetIds);
  const ids = new Set([rootId, ...targetIds]);
  const queue = [rootId];
  const seen = new Set([rootId]);
  while (queue.length && ids.size < GRAPH_VIEW.attentionMaxNodes) {
    const current = queue.shift();
    sortedDependents(current).forEach((id) => {
      if (seen.has(id)) return;
      seen.add(id);
      ids.add(id);
      if (!targets.has(id)) queue.push(id);
    });
  }
  return ids;
}

async function switchGraphState(statePathValue) {
  const nextStatePath = statePathValue.trim();
  if (!nextStatePath) return;
  try {
    state.graph = await api("/api/state", {
      method: "POST",
      body: JSON.stringify({ state_path: nextStatePath })
    });
    state.selectedId = null;
    state.scope = "attention";
    state.scopeIds = null;
    searchInput.value = "";
    statusFilter.value = "all";
    render();
    setOperationOutput("graph", `Switched to:\n${state.graph.state_path}`);
  } catch (error) {
    setOperationOutput("error", switchGraphStateErrorMessage(error));
  }
}

function switchGraphStateErrorMessage(error) {
  if (error.status === 404) {
    return [
      "This matters web server was started before graph switching was available.",
      "Restart the matters web server, reload this window, and switch again."
    ].join("\n");
  }
  return error.message;
}

function formatCommandResult(result) {
  if (result.type === "universe" || result.type === "frontier" || result.type === "horizon") {
    return result.items.length ? result.items.join("\n") : "none";
  }
  if (result.type === "unlock") {
    return JSON.stringify(result.report, null, 2);
  }
  if (result.type === "create") {
    return `Created:\n${result.created.map((matter) => `- ${matter.id}`).join("\n")}`;
  }
  return JSON.stringify(result, null, 2);
}

function setOperationOutput(role, text) {
  operationOutput.hidden = false;
  operationOutput.replaceChildren();
  const heading = document.createElement("strong");
  heading.textContent = role;
  const body = document.createElement("pre");
  body.textContent = text;
  operationOutput.append(heading, body);
  return body;
}

function ensureTerminal() {
  if (state.terminal) return;
  state.terminal = new Terminal({
    cursorBlink: true,
    fontFamily: "Menlo, Monaco, Consolas, monospace",
    fontSize: 13,
    convertEol: true,
    theme: {
      background: "#111318",
      foreground: "#e8edf2",
      cursor: "#e8edf2"
    }
  });
  state.fitAddon = new FitAddon();
  state.terminal.loadAddon(state.fitAddon);
  state.terminal.open(terminalElement);
  state.terminal.onData((data) => {
    sendTerminalInput(data);
  });
  state.terminal.onResize(({ cols, rows }) => {
    resizeTerminal(rows, cols);
  });
}

async function openTerminal() {
  terminalDrawer.hidden = false;
  ensureTerminal();
  refreshLayout();
  document.querySelector("#toggle-terminal").textContent = "Hide Terminal";
  if (!state.terminalSessionId) {
    await createTerminalSession();
  }
  startTerminalPolling();
  state.terminal.focus();
}

function hideTerminal() {
  terminalDrawer.hidden = true;
  document.querySelector("#toggle-terminal").textContent = "Terminal";
  refreshLayout();
}

function fitTerminal() {
  if (!state.fitAddon || terminalDrawer.hidden) return;
  state.fitAddon.fit();
}

function refreshLayout() {
  window.requestAnimationFrame(() => {
    resizeGraph();
    fitTerminal();
  });
}

async function createTerminalSession() {
  terminalStatus.textContent = "starting shell...";
  const rows = state.terminal?.rows || 24;
  const cols = state.terminal?.cols || 100;
  const session = await api("/api/terminal/sessions", {
    method: "POST",
    body: JSON.stringify({ rows, cols })
  });
  state.terminalSessionId = session.id;
  state.terminalSeq = 0;
  terminalStatus.textContent = session.workspace;
}

async function restartTerminal() {
  const previousSessionId = state.terminalSessionId;
  state.terminalSessionId = null;
  state.terminalSeq = 0;
  if (state.terminal) state.terminal.clear();
  if (previousSessionId) {
    await api(`/api/terminal/sessions/${encodeURIComponent(previousSessionId)}`, {
      method: "DELETE"
    }).catch(() => {});
  }
  await createTerminalSession();
}

function startTerminalPolling() {
  if (state.terminalPollTimer) return;
  state.terminalPollTimer = window.setInterval(pollTerminal, 120);
  pollTerminal();
}

async function pollTerminal() {
  if (!state.terminalSessionId || state.terminalPolling) return;
  state.terminalPolling = true;
  try {
    const payload = await api(
      `/api/terminal/sessions/${encodeURIComponent(state.terminalSessionId)}/output?seq=${state.terminalSeq}`
    );
    payload.chunks.forEach((chunk) => {
      state.terminal.write(chunk.data);
      state.terminalSeq = chunk.seq;
    });
    if (payload.closed) {
      terminalStatus.textContent = "terminal exited";
      window.clearInterval(state.terminalPollTimer);
      state.terminalPollTimer = null;
    }
  } catch (error) {
    terminalStatus.textContent = error.message;
    window.clearInterval(state.terminalPollTimer);
    state.terminalPollTimer = null;
  } finally {
    state.terminalPolling = false;
  }
}

async function sendTerminalInput(data) {
  if (!state.terminalSessionId) return;
  try {
    await api(`/api/terminal/sessions/${encodeURIComponent(state.terminalSessionId)}/input`, {
      method: "POST",
      body: JSON.stringify({ data })
    });
  } catch (error) {
    terminalStatus.textContent = error.message;
  }
}

async function resizeTerminal(rows, cols) {
  if (!state.terminalSessionId) return;
  await api(`/api/terminal/sessions/${encodeURIComponent(state.terminalSessionId)}/resize`, {
    method: "POST",
    body: JSON.stringify({ rows, cols })
  }).catch(() => {});
}

searchInput.addEventListener("input", () => {
  recomputeVisibleIds();
  refreshGraphStyles({ layout: true });
});
statusFilter.addEventListener("change", () => {
  recomputeVisibleIds();
  refreshGraphStyles({ layout: true });
});
scopeFilter.addEventListener("change", () => {
  state.selectedId = null;
  setScope(scopeFilter.value);
  renderInspector();
  updateOperationButtons();
});
window.addEventListener("resize", fitTerminal);
document.querySelector("#zoom-in").addEventListener("click", () => zoomGraph(1.22));
document.querySelector("#zoom-out").addEventListener("click", () => zoomGraph(0.82));
document.querySelector("#reset-view").addEventListener("click", () => {
  state.selectedId = null;
  searchInput.value = "";
  statusFilter.value = "all";
  setScope("attention");
  renderInspector();
  updateOperationButtons();
});
document.querySelector("#toggle-terminal").addEventListener("click", () => {
  if (terminalDrawer.hidden) {
    openTerminal().catch((error) => {
      terminalStatus.textContent = error.message;
      terminalDrawer.hidden = false;
    });
  } else {
    hideTerminal();
  }
});
document.querySelector("#hide-terminal").addEventListener("click", hideTerminal);
document.querySelector("#restart-terminal").addEventListener("click", () => {
  restartTerminal().catch((error) => {
    terminalStatus.textContent = error.message;
  });
});
document.querySelector("#show-universe").addEventListener("click", () => {
  state.selectedId = null;
  setScope("universe");
  renderInspector();
  updateOperationButtons();
  setOperationOutput("universe", (state.graph.universe || []).join("\n") || "none");
});
document.querySelector("#show-unlock").addEventListener("click", () => {
  state.selectedId = null;
  setScope("universe");
  renderInspector();
  updateOperationButtons();
  runCommand("unlock");
});
document.querySelector("#show-frontier").addEventListener("click", () => {
  if (state.selectedId) showDerivedScope("frontier", state.selectedId);
});
document.querySelector("#show-horizon").addEventListener("click", () => {
  if (state.selectedId) showDerivedScope("horizon", state.selectedId);
});

stateForm.addEventListener("submit", (event) => {
  event.preventDefault();
  switchGraphState(stateForm.elements.state_path.value);
});

document.querySelector("#create-matter-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const conditions = form.elements.conditions.value
    .split("\n")
    .map((label) => label.trim())
    .filter(Boolean)
    .map((label) => ({ label, truth: false }));
  try {
    const payload = {
      title: form.elements.title.value,
      id: form.elements.id.value,
      conditions
    };
    const createdId = payload.id.trim() || slugify(payload.title);
    state.graph = await api("/api/matters", {
      method: "POST",
      body: JSON.stringify(payload)
    });
    state.selectedId = state.graph.nodes.find((node) => node.id === createdId)?.id || state.graph.nodes.at(-1)?.id;
    state.scope = "custom";
    state.scopeIds = focusContextIds(state.selectedId);
    form.reset();
    render();
    setOperationOutput("matters", "Matter created.");
  } catch (error) {
    setOperationOutput("error", error.message);
  }
});

dependencyForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    source: dependencyForm.elements.source.value,
    target: dependencyForm.elements.target.value
  };
  try {
    state.graph = await api("/api/dependencies", {
      method: "POST",
      body: JSON.stringify(payload)
    });
    render();
    setOperationOutput("matters", `Added ${payload.source} -> ${payload.target}`);
  } catch (error) {
    setOperationOutput("error", error.message);
  }
});

document.querySelector("#remove-dependency").addEventListener("click", async () => {
  const payload = {
    source: dependencyForm.elements.source.value,
    target: dependencyForm.elements.target.value
  };
  try {
    state.graph = await api("/api/dependencies", {
      method: "DELETE",
      body: JSON.stringify(payload)
    });
    render();
    setOperationOutput("matters", `Removed ${payload.source} -> ${payload.target}`);
  } catch (error) {
    setOperationOutput("error", error.message);
  }
});

initGraph();
loadGraph().catch((error) => {
  setOperationOutput("error", error.message);
});
if (new URLSearchParams(window.location.search).has("terminal")) {
  openTerminal().catch((error) => {
    terminalStatus.textContent = error.message;
    terminalDrawer.hidden = false;
  });
}

function slugify(value) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}
