const state = {
  graph: null,
  selectedId: null,
  transform: { x: 0, y: 0, k: 1 },
  positions: new Map(),
  dragging: false,
  dragStart: null
};

const svg = document.querySelector("#graph");
const viewport = document.querySelector("#viewport");
const edgeLayer = document.querySelector("#edges");
const nodeLayer = document.querySelector("#nodes");
const inspector = document.querySelector("#inspector");
const messages = document.querySelector("#messages");
const statePath = document.querySelector("#state-path");
const emptyState = document.querySelector("#empty-state");
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

function ensurePositions() {
  const nodes = state.graph.nodes;
  const width = Math.max(svg.clientWidth, 900);
  const height = Math.max(svg.clientHeight, 640);
  const radius = Math.min(width, height) * 0.34;
  const centerX = width / 2;
  const centerY = height / 2;

  nodes.forEach((node, index) => {
    if (state.positions.has(node.id)) return;
    const angle = (Math.PI * 2 * index) / Math.max(nodes.length, 1) - Math.PI / 2;
    const ringOffset = index % 3 === 0 ? 0.72 : index % 3 === 1 ? 1 : 1.2;
    state.positions.set(node.id, {
      x: centerX + Math.cos(angle) * radius * ringOffset,
      y: centerY + Math.sin(angle) * radius * ringOffset
    });
  });
}

function render() {
  renderFiltersAndSelectors();
  renderGraph();
  renderInspector();
  updateOperationButtons();
  emptyState.hidden = state.graph.nodes.length > 0;
}

function renderGraph() {
  edgeLayer.replaceChildren();
  nodeLayer.replaceChildren();

  const visible = new Set(filteredNodes().map((node) => node.id));
  const connected = connectedSet();

  state.graph.edges.forEach((edge) => {
    const source = state.positions.get(edge.source);
    const target = state.positions.get(edge.target);
    if (!source || !target) return;

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.classList.add("edge");
    line.dataset.source = edge.source;
    line.dataset.target = edge.target;
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    if (!visible.has(edge.source) || !visible.has(edge.target)) {
      line.classList.add("hidden");
    }
    if (state.selectedId) {
      if (edge.source === state.selectedId || edge.target === state.selectedId) {
        line.classList.add("connected");
      } else {
        line.classList.add("dimmed");
      }
    }
    edgeLayer.append(line);
  });

  state.graph.nodes.forEach((node) => {
    const point = state.positions.get(node.id);
    if (!point) return;

    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.classList.add("node", statusClass(node));
    group.dataset.id = node.id;
    group.setAttribute("transform", `translate(${point.x}, ${point.y})`);
    if (node.id === state.selectedId) group.classList.add("selected");
    if (!visible.has(node.id)) group.classList.add("hidden");
    if (state.selectedId && !connected.has(node.id)) group.classList.add("dimmed");

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("r", node.id === state.selectedId ? 18 : 15);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", 23);
    label.setAttribute("y", 5);
    label.textContent = displayLabel(node);

    group.append(circle, label);
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      state.selectedId = node.id;
      render();
    });
    nodeLayer.append(group);
  });

  applyTransform();
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

function applyTransform() {
  viewport.setAttribute(
    "transform",
    `translate(${state.transform.x}, ${state.transform.y}) scale(${state.transform.k})`
  );
}

function resetView() {
  state.transform = { x: 0, y: 0, k: 1 };
  applyTransform();
}

svg.addEventListener("wheel", (event) => {
  event.preventDefault();
  const direction = event.deltaY > 0 ? -1 : 1;
  const scale = direction > 0 ? 1.12 : 0.88;
  const nextK = Math.min(3.4, Math.max(0.22, state.transform.k * scale));
  const rect = svg.getBoundingClientRect();
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;
  state.transform.x = px - (px - state.transform.x) * (nextK / state.transform.k);
  state.transform.y = py - (py - state.transform.y) * (nextK / state.transform.k);
  state.transform.k = nextK;
  applyTransform();
}, { passive: false });

svg.addEventListener("pointerdown", (event) => {
  if (event.target.closest(".node")) return;
  state.dragging = true;
  state.dragStart = {
    x: event.clientX,
    y: event.clientY,
    tx: state.transform.x,
    ty: state.transform.y
  };
  svg.classList.add("dragging");
  svg.setPointerCapture(event.pointerId);
});

svg.addEventListener("pointermove", (event) => {
  if (!state.dragging) return;
  state.transform.x = state.dragStart.tx + event.clientX - state.dragStart.x;
  state.transform.y = state.dragStart.ty + event.clientY - state.dragStart.y;
  applyTransform();
});

svg.addEventListener("pointerup", (event) => {
  state.dragging = false;
  svg.classList.remove("dragging");
  svg.releasePointerCapture(event.pointerId);
});

svg.addEventListener("click", () => {
  state.selectedId = null;
  render();
});

searchInput.addEventListener("input", render);
statusFilter.addEventListener("change", render);
document.querySelector("#reset-view").addEventListener("click", resetView);
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
