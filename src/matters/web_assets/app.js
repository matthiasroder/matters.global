import ForceGraph3D from "https://cdn.jsdelivr.net/npm/3d-force-graph@1.78.0/+esm";
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.178.0/+esm";

const STATUS_COLORS = {
  actionable: "#2f7d55",
  blocked: "#b8542f",
  resolved: "#5b6d7a",
  selected: "#274f9f",
  edge: "#8f9aa3",
  faded: "#c6ccc2"
};

const state = {
  graph: null,
  forceGraph: null,
  selectedId: null,
  visibleIds: new Set(),
  nodeObjects: new Map(),
  webgl: false
};

const graphElement = document.querySelector("#graph");
const inspector = document.querySelector("#inspector");
const messages = document.querySelector("#messages");
const statePath = document.querySelector("#state-path");
const emptyState = document.querySelector("#empty-state");
const webglFallback = document.querySelector("#webgl-fallback");
const searchInput = document.querySelector("#search");
const statusFilter = document.querySelector("#status-filter");
const dependencyForm = document.querySelector("#dependency-form");

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
  statePath.textContent = state.graph.state_path;
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
    .backgroundColor("#f7f8f5")
    .showNavInfo(false)
    .enableNodeDrag(false)
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

  const chargeForce = state.forceGraph.d3Force("charge");
  if (chargeForce) chargeForce.strength(-44);
  const linkForce = state.forceGraph.d3Force("link");
  if (linkForce) {
    linkForce.distance((link) => (sameStatus(link) ? 10 : 14));
    linkForce.strength(1.05);
  }
  state.forceGraph.d3VelocityDecay(0.36);

  resizeGraph();
  window.addEventListener("resize", resizeGraph);
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

function resizeGraph() {
  if (!state.forceGraph) return;
  state.forceGraph
    .width(Math.max(graphElement.clientWidth, 1))
    .height(Math.max(graphElement.clientHeight, 1));
}

function render() {
  state.visibleIds = new Set(filteredNodes().map((node) => node.id));
  renderFiltersAndSelectors();
  renderGraph();
  renderInspector();
  updateOperationButtons();
  emptyState.hidden = state.graph.nodes.length > 0;
}

function renderGraph() {
  if (!state.forceGraph) return;
  state.nodeObjects.clear();
  state.forceGraph.graphData(toForceGraphData());
  refreshGraphStyles();
  window.setTimeout(() => {
    resetCamera();
  }, 350);
}

function toForceGraphData() {
  const nodes = state.graph.nodes.map((node) => ({
    ...node,
    name: node.label,
    val: node.id === state.selectedId ? 3.3 : node.actionable ? 2.7 : 2.4
  }));
  const links = state.graph.edges.map((edge) => ({
    source: edge.source,
    target: edge.target
  }));
  return { nodes, links };
}

function createNodeObject(node) {
  const group = new THREE.Group();
  const selected = node.id === state.selectedId;
  const radius = selected ? 1.85 : node.actionable ? 1.55 : 1.38;
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
  context.fillStyle = selected ? "rgba(238, 244, 255, 0.96)" : "rgba(255, 255, 255, 0.86)";
  roundRect(context, 8, 22, 496, 82, 18);
  context.fill();
  context.fillStyle = selected ? STATUS_COLORS.selected : "#1d2025";
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
  appendMessage("you", text);
  try {
    const result = await api("/api/command", {
      method: "POST",
      body: JSON.stringify({ text })
    });
    appendMessage("matters", formatCommandResult(result));
    if (result.state) {
      state.graph = result.state;
      render();
    } else {
      await loadGraph();
    }
  } catch (error) {
    appendMessage("error", error.message);
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

function appendMessage(role, text) {
  const message = document.createElement("div");
  message.className = "message";
  const heading = document.createElement("strong");
  heading.textContent = role;
  const body = document.createElement("pre");
  body.textContent = text;
  message.append(heading, body);
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
}

function resetCamera() {
  if (!state.forceGraph) return;
  const distance = graphElement.clientWidth < 600 ? 82 : 92;
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
document.querySelector("#zoom-in").addEventListener("click", () => zoomCamera(0.72));
document.querySelector("#zoom-out").addEventListener("click", () => zoomCamera(1.38));
document.querySelector("#reset-view").addEventListener("click", resetCamera);
document.querySelector("#show-universe").addEventListener("click", () => runCommand("universe"));
document.querySelector("#show-unlock").addEventListener("click", () => runCommand("unlock"));
document.querySelector("#show-frontier").addEventListener("click", () => {
  if (state.selectedId) runCommand(`frontier ${state.selectedId}`);
});
document.querySelector("#show-horizon").addEventListener("click", () => {
  if (state.selectedId) runCommand(`horizon ${state.selectedId}`);
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
    appendMessage("matters", "Matter created.");
  } catch (error) {
    appendMessage("error", error.message);
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
    appendMessage("matters", `Added ${payload.source} -> ${payload.target}`);
  } catch (error) {
    appendMessage("error", error.message);
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
    appendMessage("matters", `Removed ${payload.source} -> ${payload.target}`);
  } catch (error) {
    appendMessage("error", error.message);
  }
});

document.querySelector("#command-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = event.currentTarget.elements.command;
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  runCommand(text);
});

initGraph();
loadGraph().catch((error) => {
  appendMessage("error", error.message);
});

function slugify(value) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}
