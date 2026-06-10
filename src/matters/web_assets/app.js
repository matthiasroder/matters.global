import ForceGraph3D from "https://cdn.jsdelivr.net/npm/3d-force-graph@1.78.0/+esm";
import { FitAddon } from "https://cdn.jsdelivr.net/npm/@xterm/addon-fit@0.10.0/+esm";
import { Terminal } from "https://cdn.jsdelivr.net/npm/@xterm/xterm@5.5.0/+esm";
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.178.0/+esm";

const STATUS_COLORS = {
  actionable: "#27835b",
  blocked: "#cc654c",
  resolved: "#687985",
  selected: "#2f68c5",
  edge: "#9aa4aa",
  faded: "#d8d5ca"
};

const ORGANIC_LAYOUT = {
  chargeStrength: -46,
  linkDistance: 28,
  linkStrength: 0.34,
  statusDriftStrength: 0.012,
  gravityStrength: 0.032,
  collisionPadding: 4.2,
  velocityDecay: 0.26,
  warmupTicks: 90,
  cooldownTicks: 420
};

const state = {
  graph: null,
  forceGraph: null,
  selectedId: null,
  visibleIds: new Set(),
  nodeObjects: new Map(),
  webgl: false,
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
const webglFallback = document.querySelector("#webgl-fallback");
const searchInput = document.querySelector("#search");
const statusFilter = document.querySelector("#status-filter");
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
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

async function loadGraph() {
  state.graph = await api("/api/state");
  render();
}

function initGraph() {
  if (!hasWebGL()) {
    webglFallback.hidden = false;
    graphElement.hidden = true;
    return;
  }

  state.webgl = true;
  state.forceGraph = ForceGraph3D()(graphElement)
    .backgroundColor("#fbfaf4")
    .showNavInfo(false)
    .enableNodeDrag(false)
    .warmupTicks(ORGANIC_LAYOUT.warmupTicks)
    .cooldownTicks(ORGANIC_LAYOUT.cooldownTicks)
    .nodeThreeObject(createNodeObject)
    .nodeVisibility((node) => nodeVisible(node.id))
    .linkVisibility((link) => linkVisible(link))
    .linkMaterial(createLinkMaterial)
    .linkDirectionalArrowLength(1.7)
    .linkDirectionalArrowRelPos(0.92)
    .linkDirectionalArrowColor((link) => linkColor(link))
    .linkDirectionalParticles(0)
    .onNodeClick((node) => {
      state.selectedId = node.id;
      renderInspector();
      updateOperationButtons();
      refreshGraphStyles();
    })
    .onBackgroundClick(() => {
      state.selectedId = null;
      renderInspector();
      updateOperationButtons();
      refreshGraphStyles();
  });

  configureOrganicLayout();
  resizeGraph();
  window.addEventListener("resize", resizeGraph);
}

function configureOrganicLayout() {
  const chargeForce = state.forceGraph.d3Force("charge");
  if (chargeForce) {
    chargeForce
      .strength(ORGANIC_LAYOUT.chargeStrength)
      .distanceMin(5)
      .distanceMax(260);
  }

  const linkForce = state.forceGraph.d3Force("link");
  if (linkForce) {
    linkForce
      .distance(organicLinkDistance)
      .strength(organicLinkStrength)
      .iterations(2);
  }

  state.forceGraph.d3Force("organicGravity", organicGravityForce(ORGANIC_LAYOUT.gravityStrength));
  state.forceGraph.d3Force("statusDrift", statusDriftForce(ORGANIC_LAYOUT.statusDriftStrength));
  state.forceGraph.d3Force("nodeCollision", nodeCollisionForce(ORGANIC_LAYOUT.collisionPadding));
  state.forceGraph.d3VelocityDecay(ORGANIC_LAYOUT.velocityDecay);
}

function hasWebGL() {
  try {
    const probe = document.createElement("canvas");
    return Boolean(
      window.WebGLRenderingContext &&
        (probe.getContext("webgl2") || probe.getContext("webgl"))
    );
  } catch {
    return false;
  }
}

function organicGravityForce(strength) {
  let nodes = [];

  function force(alpha) {
    for (const node of nodes) {
      node.vx = (node.vx || 0) - (node.x || 0) * strength * alpha;
      node.vy = (node.vy || 0) - (node.y || 0) * strength * alpha;
      node.vz = (node.vz || 0) - (node.z || 0) * strength * alpha;
    }
  }

  force.initialize = (nextNodes) => {
    nodes = nextNodes;
  };

  return force;
}

function statusDriftForce(strength) {
  let nodes = [];

  function force(alpha) {
    for (const node of nodes) {
      const anchor = statusAnchor(node);
      node.vx = (node.vx || 0) + (anchor.x - (node.x || 0)) * strength * alpha;
      node.vy = (node.vy || 0) + (anchor.y - (node.y || 0)) * strength * alpha;
      node.vz = (node.vz || 0) + (anchor.z - (node.z || 0)) * strength * alpha;
    }
  }

  force.initialize = (nextNodes) => {
    nodes = nextNodes;
  };

  return force;
}

function nodeCollisionForce(padding) {
  let nodes = [];

  function force(alpha) {
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i];
        const b = nodes[j];
        const minDistance = nodeRadius(a) + nodeRadius(b) + padding;
        const dx = ((b.x || 0) - (a.x || 0)) || 0.01;
        const dy = ((b.y || 0) - (a.y || 0)) || -0.01;
        const dz = ((b.z || 0) - (a.z || 0)) || 0.01;
        const distance = Math.hypot(dx, dy, dz);

        if (distance >= minDistance) continue;

        const push = ((minDistance - distance) / distance) * alpha * 0.48;
        const x = dx * push;
        const y = dy * push;
        const z = dz * push;
        a.vx = (a.vx || 0) - x;
        a.vy = (a.vy || 0) - y;
        a.vz = (a.vz || 0) - z;
        b.vx = (b.vx || 0) + x;
        b.vy = (b.vy || 0) + y;
        b.vz = (b.vz || 0) + z;
      }
    }
  }

  force.initialize = (nextNodes) => {
    nodes = nextNodes;
  };

  return force;
}

function organicLinkDistance(link) {
  const sourceId = linkId(link.source);
  const targetId = linkId(link.target);
  const degreeSpread = Math.min(22, (graphDegree(sourceId) + graphDegree(targetId)) * 1.7);
  const statusSpread = sameStatus(link) ? -4 : 8;
  return ORGANIC_LAYOUT.linkDistance + degreeSpread + statusSpread;
}

function organicLinkStrength(link) {
  return sameStatus(link) ? ORGANIC_LAYOUT.linkStrength + 0.08 : ORGANIC_LAYOUT.linkStrength;
}

function resizeGraph() {
  if (!state.forceGraph) return;
  const viewport = graphViewportRect();
  state.forceGraph
    .width(Math.max(Math.floor(viewport.width), 1))
    .height(Math.max(Math.floor(viewport.height), 1));
}

function graphViewportRect() {
  return graphElement.parentElement?.getBoundingClientRect() || graphElement.getBoundingClientRect();
}

function render() {
  syncStatePathControl();
  state.visibleIds = new Set(filteredNodes().map((node) => node.id));
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

function renderGraph() {
  if (!state.forceGraph) return;
  state.nodeObjects.clear();
  state.forceGraph.graphData(toForceGraphData());
  refreshGraphStyles();
  state.forceGraph.d3ReheatSimulation?.();
  window.setTimeout(() => {
    resetCamera();
  }, 550);
}

function toForceGraphData() {
  const previousNodes = currentGraphNodesById();
  const total = state.graph.nodes.length || 1;
  const nodes = state.graph.nodes.map((node, index) => {
    const previous = previousNodes.get(node.id);
    return {
      ...node,
      ...organicSeedPosition(node, index, total, previous),
      name: node.label,
      val: nodeRadius(node)
    };
  });
  const links = state.graph.edges.map((edge) => ({
    source: edge.source,
    target: edge.target
  }));
  return { nodes, links };
}

function createNodeObject(node) {
  const group = new THREE.Group();
  const selected = node.id === state.selectedId;
  const radius = nodeRadius(node);
  const color = selected ? STATUS_COLORS.selected : statusColor(node);

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 28, 18),
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.46,
      metalness: 0.07,
      transparent: true
    })
  );
  group.add(sphere);

  const label = makeLabelSprite(displayLabel(node), selected);
  label.position.set(radius + 1.4, radius + 0.7, 0);
  group.add(label);

  group.userData = { id: node.id };
  state.nodeObjects.set(node.id, group);
  applyNodeObjectStyle(group, node);
  return group;
}

function currentGraphNodesById() {
  const graphData = state.forceGraph?.graphData?.();
  if (!graphData?.nodes) return new Map();
  return new Map(graphData.nodes.map((node) => [node.id, node]));
}

function organicSeedPosition(node, index, total, previous) {
  if (
    Number.isFinite(previous?.x) &&
    Number.isFinite(previous?.y) &&
    Number.isFinite(previous?.z)
  ) {
    return {
      x: previous.x,
      y: previous.y,
      z: previous.z,
      vx: previous.vx || 0,
      vy: previous.vy || 0,
      vz: previous.vz || 0
    };
  }

  const radius = Math.min(130, Math.max(28, 18 + Math.sqrt(total) * 11 + graphDegree(node.id) * 1.5));
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const denominator = Math.max(total - 1, 1);
  const y = 1 - (index / denominator) * 2;
  const spread = Math.sqrt(Math.max(0, 1 - y * y));
  const theta = index * goldenAngle;
  const anchor = statusAnchor(node);

  return {
    x: Math.cos(theta) * spread * radius + anchor.x * 0.26,
    y: y * radius * 0.72 + anchor.y * 0.26,
    z: Math.sin(theta) * spread * radius + anchor.z * 0.26
  };
}

function nodeRadius(node) {
  if (node.id === state.selectedId) return 1.85;
  if (node.actionable) return 1.55;
  return 1.38;
}

function createLinkMaterial(link) {
  return new THREE.LineBasicMaterial({
    color: linkColor(link),
    transparent: true,
    opacity: linkOpacity(link)
  });
}

function refreshGraphStyles() {
  if (!state.forceGraph) return;
  state.forceGraph
    .nodeVisibility((node) => nodeVisible(node.id))
    .linkVisibility((link) => linkVisible(link))
    .linkMaterial(createLinkMaterial)
    .linkDirectionalArrowColor((link) => linkColor(link))
    .nodeThreeObject(createNodeObject);
  state.forceGraph.refresh();
}

function applyNodeObjectStyle(group, node) {
  const opacity = nodeOpacity(node);
  group.traverse((child) => {
    if (!child.material) return;
    child.material.transparent = true;
    child.material.opacity = opacity;
  });
}

function makeLabelSprite(text, selected) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = 512;
  canvas.height = 128;
  context.font = "700 34px Inter, system-ui, sans-serif";
  context.textBaseline = "middle";
  context.fillStyle = selected ? "rgba(238, 246, 255, 0.96)" : "rgba(255, 253, 248, 0.88)";
  roundRect(context, 8, 22, 496, 82, 18);
  context.fill();
  context.fillStyle = selected ? STATUS_COLORS.selected : "#24292f";
  context.fillText(text, 28, 64, 456);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(14, 3.5, 1);
  return sprite;
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + width, y, x + width, y + height, radius);
  context.arcTo(x + width, y + height, x, y + height, radius);
  context.arcTo(x, y + height, x, y, radius);
  context.arcTo(x, y, x + width, y, radius);
  context.closePath();
}

function nodeVisible(id) {
  return state.visibleIds.has(id);
}

function linkVisible(link) {
  return state.visibleIds.has(linkId(link.source)) && state.visibleIds.has(linkId(link.target));
}

function nodeOpacity(node) {
  if (!state.selectedId) return 1;
  return connectedSet().has(node.id) ? 1 : 0.14;
}

function linkOpacity(link) {
  if (!state.selectedId) return 0.62;
  return linkTouchesSelection(link) ? 0.92 : 0.08;
}

function linkColor(link) {
  if (!state.selectedId) return STATUS_COLORS.edge;
  return linkTouchesSelection(link) ? STATUS_COLORS.selected : STATUS_COLORS.faded;
}

function linkTouchesSelection(link) {
  return linkId(link.source) === state.selectedId || linkId(link.target) === state.selectedId;
}

function linkId(endpoint) {
  return typeof endpoint === "object" ? endpoint.id : endpoint;
}

function sameStatus(link) {
  const source = nodeById(linkId(link.source));
  const target = nodeById(linkId(link.target));
  return source && target && statusClass(source) === statusClass(target);
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
  return state.graph.nodes.filter((node) => {
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
    return matchesText && matchesStatus;
  });
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

function statusColor(node) {
  if (node.resolved) return STATUS_COLORS.resolved;
  if (node.actionable) return STATUS_COLORS.actionable;
  return STATUS_COLORS.blocked;
}

function statusAnchor(node) {
  if (node.resolved) return { x: 14, y: -10, z: 16 };
  if (node.actionable) return { x: -18, y: 12, z: 4 };
  return { x: 8, y: 4, z: -14 };
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

function displayLabel(node) {
  return node.label.length > 34 ? `${node.label.slice(0, 31)}...` : node.label;
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

async function switchGraphState(statePathValue) {
  const nextStatePath = statePathValue.trim();
  if (!nextStatePath) return;
  try {
    state.graph = await api("/api/state", {
      method: "POST",
      body: JSON.stringify({ state_path: nextStatePath })
    });
    state.selectedId = null;
    searchInput.value = "";
    statusFilter.value = "all";
    render();
    setOperationOutput("graph", `Switched to:\n${state.graph.state_path}`);
  } catch (error) {
    setOperationOutput("error", error.message);
  }
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
  terminalStatus.textContent = "starting zsh...";
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

function resetCamera() {
  if (!state.forceGraph) return;
  const distance = graphViewportRect().width < 600 ? 82 : 92;
  state.forceGraph.cameraPosition({ x: 0, y: 0, z: distance }, { x: 0, y: 0, z: 0 }, 700);
}

function zoomCamera(factor) {
  if (!state.forceGraph) return;
  const camera = state.forceGraph.camera();
  const { x, y, z } = camera.position;
  const currentDistance = Math.hypot(x, y, z) || 1;
  const nextDistance = Math.min(600, Math.max(12, currentDistance * factor));
  const scale = nextDistance / currentDistance;
  state.forceGraph.cameraPosition(
    { x: x * scale, y: y * scale, z: z * scale },
    { x: 0, y: 0, z: 0 },
    260
  );
}

searchInput.addEventListener("input", () => {
  state.visibleIds = new Set(filteredNodes().map((node) => node.id));
  refreshGraphStyles();
});
statusFilter.addEventListener("change", () => {
  state.visibleIds = new Set(filteredNodes().map((node) => node.id));
  refreshGraphStyles();
});
window.addEventListener("resize", fitTerminal);
document.querySelector("#zoom-in").addEventListener("click", () => zoomCamera(0.72));
document.querySelector("#zoom-out").addEventListener("click", () => zoomCamera(1.38));
document.querySelector("#reset-view").addEventListener("click", resetCamera);
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
document.querySelector("#show-universe").addEventListener("click", () => runCommand("universe"));
document.querySelector("#show-unlock").addEventListener("click", () => runCommand("unlock"));
document.querySelector("#show-frontier").addEventListener("click", () => {
  if (state.selectedId) runCommand(`frontier ${state.selectedId}`);
});
document.querySelector("#show-horizon").addEventListener("click", () => {
  if (state.selectedId) runCommand(`horizon ${state.selectedId}`);
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
