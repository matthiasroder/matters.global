import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const STATUS_COLORS = {
  actionable: 0x2f7d55,
  blocked: 0xb8542f,
  resolved: 0x5b6d7a,
  selected: 0x274f9f,
  edge: 0x66717c,
  dimmed: 0xc3c9c0
};

const state = {
  graph: null,
  selectedId: null,
  positions: new Map(),
  nodeObjects: new Map(),
  edgeObjects: [],
  visibleIds: new Set(),
  scene: null,
  camera: null,
  renderer: null,
  graphGroup: null,
  raycaster: new THREE.Raycaster(),
  pointer: new THREE.Vector2(),
  interaction: null,
  animationFrame: null,
  webgl: false
};

const canvas = document.querySelector("#graph");
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
  ensurePositions();
  render();
}

function initThree() {
  if (!hasWebGL()) {
    webglFallback.hidden = false;
    canvas.hidden = true;
    return;
  }

  state.webgl = true;
  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color(0xf7f8f5);

  state.camera = new THREE.PerspectiveCamera(48, 1, 0.1, 2000);
  resetCamera();

  state.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const ambient = new THREE.AmbientLight(0xffffff, 0.72);
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(9, 12, 14);
  const fill = new THREE.DirectionalLight(0xffffff, 0.45);
  fill.position.set(-12, -8, 8);
  state.scene.add(ambient, key, fill);

  state.graphGroup = new THREE.Group();
  state.scene.add(state.graphGroup);

  resizeRenderer();
  window.addEventListener("resize", resizeRenderer);
  animate();
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

function resizeRenderer() {
  if (!state.webgl) return;
  const width = Math.max(canvas.clientWidth, 1);
  const height = Math.max(canvas.clientHeight, 1);
  state.camera.aspect = width / height;
  state.camera.updateProjectionMatrix();
  state.renderer.setSize(width, height, false);
}

function ensurePositions() {
  const nodes = state.graph.nodes;
  const count = Math.max(nodes.length, 1);
  const radius = Math.max(7, Math.cbrt(count) * 5.2);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  nodes.forEach((node, index) => {
    if (state.positions.has(node.id)) return;
    const y = count === 1 ? 0 : 1 - (index / (count - 1)) * 2;
    const ring = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = index * goldenAngle;
    const depth = dependencyDepth(node.id) * 2.2;
    state.positions.set(
      node.id,
      new THREE.Vector3(
        Math.cos(theta) * ring * radius,
        y * radius * 0.72,
        Math.sin(theta) * ring * radius + depth
      )
    );
  });
}

function dependencyDepth(nodeId) {
  const incoming = new Map();
  state.graph.edges.forEach((edge) => {
    if (!incoming.has(edge.target)) incoming.set(edge.target, []);
    incoming.get(edge.target).push(edge.source);
  });

  const visit = (id, seen = new Set()) => {
    if (seen.has(id)) return 0;
    const prerequisites = incoming.get(id) || [];
    if (!prerequisites.length) return 0;
    return 1 + Math.max(...prerequisites.map((source) => visit(source, new Set([...seen, id]))));
  };

  return visit(nodeId);
}

function render() {
  renderFiltersAndSelectors();
  renderGraph();
  renderInspector();
  updateOperationButtons();
  emptyState.hidden = state.graph.nodes.length > 0;
}

function renderGraph() {
  state.visibleIds = new Set(filteredNodes().map((node) => node.id));
  if (!state.webgl) return;

  state.graphGroup.clear();
  state.nodeObjects.clear();
  state.edgeObjects = [];
  const connected = connectedSet();

  state.graph.edges.forEach((edge) => {
    const source = state.positions.get(edge.source);
    const target = state.positions.get(edge.target);
    if (!source || !target) return;

    const edgeGroup = createEdge(source, target);
    edgeGroup.userData = { source: edge.source, target: edge.target, kind: "edge" };
    applyEdgeVisibility(edgeGroup, edge, connected);
    state.graphGroup.add(edgeGroup);
    state.edgeObjects.push(edgeGroup);
  });

  state.graph.nodes.forEach((node) => {
    const point = state.positions.get(node.id);
    if (!point) return;

    const nodeGroup = createNode(node);
    nodeGroup.position.copy(point);
    nodeGroup.userData = { id: node.id, kind: "node" };
    applyNodeVisibility(nodeGroup, node, connected);
    state.graphGroup.add(nodeGroup);
    state.nodeObjects.set(node.id, nodeGroup);
  });
}

function createNode(node) {
  const group = new THREE.Group();
  const radius = node.id === state.selectedId ? 0.54 : 0.44;
  const geometry = new THREE.SphereGeometry(radius, 32, 20);
  const material = new THREE.MeshStandardMaterial({
    color: node.id === state.selectedId ? STATUS_COLORS.selected : statusColor(node),
    roughness: 0.42,
    metalness: 0.08,
    transparent: true
  });
  const sphere = new THREE.Mesh(geometry, material);
  sphere.userData = { id: node.id, kind: "node-hit-target" };
  group.add(sphere);

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(radius * 1.18, 0.035, 10, 36),
    new THREE.MeshBasicMaterial({
      color: node.id === state.selectedId ? STATUS_COLORS.selected : statusColor(node),
      transparent: true,
      opacity: node.id === state.selectedId ? 0.86 : 0.42
    })
  );
  ring.rotation.x = Math.PI / 2;
  group.add(ring);

  const label = makeLabelSprite(displayLabel(node), node.id === state.selectedId);
  label.position.set(0.72, 0.54, 0);
  group.add(label);

  return group;
}

function createEdge(source, target) {
  const group = new THREE.Group();
  const direction = new THREE.Vector3().subVectors(target, source);
  const length = direction.length();
  const lineEnd = new THREE.Vector3().copy(source).add(direction.clone().multiplyScalar(0.9));

  const geometry = new THREE.BufferGeometry().setFromPoints([source, lineEnd]);
  const line = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({
      color: STATUS_COLORS.edge,
      transparent: true,
      opacity: 0.64
    })
  );

  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.18, 0.52, 18),
    new THREE.MeshBasicMaterial({
      color: STATUS_COLORS.edge,
      transparent: true,
      opacity: 0.7
    })
  );
  cone.position.copy(source).add(direction.multiplyScalar(0.92));
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3().subVectors(target, source).normalize());
  if (length < 0.001) cone.visible = false;

  group.add(line, cone);
  return group;
}

function makeLabelSprite(text, selected) {
  const canvasLabel = document.createElement("canvas");
  const context = canvasLabel.getContext("2d");
  canvasLabel.width = 512;
  canvasLabel.height = 128;
  context.font = "700 34px Inter, system-ui, sans-serif";
  context.textBaseline = "middle";
  context.fillStyle = selected ? "rgba(238, 244, 255, 0.96)" : "rgba(255, 255, 255, 0.86)";
  roundRect(context, 8, 22, 496, 82, 18);
  context.fill();
  context.fillStyle = selected ? "#274f9f" : "#1d2025";
  context.fillText(text, 28, 64, 456);

  const texture = new THREE.CanvasTexture(canvasLabel);
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(4.2, 1.05, 1);
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

function applyNodeVisibility(group, node, connected) {
  const visible = state.visibleIds.has(node.id);
  const focusedOut = state.selectedId && !connected.has(node.id);
  group.visible = visible;
  setObjectOpacity(group, focusedOut ? 0.16 : 1);
}

function applyEdgeVisibility(group, edge, connected) {
  const visible = state.visibleIds.has(edge.source) && state.visibleIds.has(edge.target);
  const connectedEdge = edge.source === state.selectedId || edge.target === state.selectedId;
  const focusedOut = state.selectedId && !connectedEdge;
  group.visible = visible;
  setObjectOpacity(group, connectedEdge ? 0.96 : focusedOut ? 0.1 : 0.64);
  if (connectedEdge) {
    group.children.forEach((child) => {
      child.material.color.setHex(STATUS_COLORS.selected);
    });
  }
}

function setObjectOpacity(object, opacity) {
  object.traverse((child) => {
    if (!child.material) return;
    child.material.transparent = true;
    child.material.opacity = opacity;
  });
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
  return state.graph?.nodes.find((node) => node.id === state.selectedId);
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
      ensurePositions();
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
  if (!state.camera) return;
  state.camera.position.set(0, 0, 24);
  state.camera.lookAt(0, 0, 0);
  if (state.graphGroup) {
    state.graphGroup.rotation.set(-0.2, 0.35, 0);
    state.graphGroup.position.set(0, 0, 0);
  }
}

function animate() {
  state.animationFrame = requestAnimationFrame(animate);
  if (!state.webgl) return;
  state.renderer.render(state.scene, state.camera);
}

function pointerToNdc(event) {
  const rect = canvas.getBoundingClientRect();
  state.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  state.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function hitNode(event) {
  if (!state.webgl) return null;
  pointerToNdc(event);
  state.raycaster.setFromCamera(state.pointer, state.camera);
  const targets = Array.from(state.nodeObjects.values()).flatMap((group) => group.children);
  const hits = state.raycaster.intersectObjects(targets, true);
  const hit = hits.find((item) => item.object.userData.kind === "node-hit-target");
  return hit?.object.userData.id || null;
}

function startInteraction(event) {
  if (!state.webgl) return;
  const nodeId = hitNode(event);
  state.interaction = {
    mode: nodeId ? "node-rotate" : "orbit",
    nodeId,
    startX: event.clientX,
    startY: event.clientY,
    moved: false,
    rotation: state.graphGroup.rotation.clone(),
    position: state.graphGroup.position.clone()
  };
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
}

function moveInteraction(event) {
  if (!state.interaction) return;
  const dx = event.clientX - state.interaction.startX;
  const dy = event.clientY - state.interaction.startY;
  if (Math.hypot(dx, dy) > 4) state.interaction.moved = true;

  if (event.shiftKey && state.interaction.mode === "orbit") {
    state.graphGroup.position.x = state.interaction.position.x + dx * 0.018;
    state.graphGroup.position.y = state.interaction.position.y - dy * 0.018;
    return;
  }

  state.graphGroup.rotation.y = state.interaction.rotation.y + dx * 0.012;
  state.graphGroup.rotation.x = state.interaction.rotation.x + dy * 0.012;
  const nodeRoll = state.interaction.mode === "node-rotate" ? (dx - dy) * 0.003 : 0;
  state.graphGroup.rotation.z =
    state.interaction.rotation.z + nodeRoll + (event.altKey ? dx * 0.006 : 0);
}

function endInteraction(event) {
  if (!state.interaction) return;
  const interaction = state.interaction;
  state.interaction = null;
  canvas.classList.remove("dragging");
  canvas.releasePointerCapture(event.pointerId);

  if (!interaction.moved && interaction.nodeId) {
    state.selectedId = interaction.nodeId;
    render();
  } else if (!interaction.moved) {
    state.selectedId = null;
    render();
  }
}

canvas.addEventListener("pointerdown", startInteraction);
canvas.addEventListener("pointermove", moveInteraction);
canvas.addEventListener("pointerup", endInteraction);
canvas.addEventListener("pointercancel", endInteraction);
canvas.addEventListener("wheel", (event) => {
  if (!state.webgl) return;
  event.preventDefault();
  const nextZ = state.camera.position.z + event.deltaY * 0.016;
  state.camera.position.z = Math.min(80, Math.max(4, nextZ));
}, { passive: false });

searchInput.addEventListener("input", render);
statusFilter.addEventListener("change", render);
document.querySelector("#reset-view").addEventListener("click", () => {
  resetCamera();
  resizeRenderer();
});
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
    ensurePositions();
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

initThree();
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
