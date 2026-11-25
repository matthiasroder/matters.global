"use strict";

const state = { user: null, graphs: [], graph: null, report: null, visual: null, selectedMatter: null, detailsMode: null, visualMode: "overview", sharingRequest: 0, accountSecurity: null, securityRequest: 0, recoveryCapabilities: null, recoveryRequest: 0, register: false, generation: 0, authEpoch: 0, authAttempt: 0, requestedGraph: null, catalogRequest: 0 };
const byId = id => document.getElementById(id);
const authForm = byId("auth-form");
const graphForm = byId("new-graph-form");
const matterForm = byId("new-matter-form");
const addConditionForm = byId("add-condition-form");
const invitationForm = byId("invitation-form");
const joinForm = byId("join-graph-form");
const dependencyForm = byId("dependency-form");
const recoveryForm = byId("recovery-request-form");
const passwordChangeForm = byId("password-change-form");
const signupEnabled = document.body?.dataset?.signupEnabled !== "false";
let overviewRenderer = null;
let rendererLoading = null;
let lastVisualMode = null;
let incomingInvitation = initialInvitation();
let inspectorReturnFocus = null;
const menuIds = ["graph-menu", "new-matter-panel", "account-menu"];
const maxMatterConditions = 1000;
let matterConditionRows = [];
let matterDraftSerial = 0;
let matterCreatePending = false;
let addConditionSerial = 0;
let addConditionScopeKey = null;
let addConditionPending = false;

function updateMatterConditionControls() {
  matterConditionRows.forEach((row, index) => {
    row.label.textContent = `Condition ${index + 1}`;
    row.input.id = `matter-condition-${index + 1}`;
    row.remove.setAttribute("aria-label", `Remove condition ${index + 1}`);
    row.remove.disabled = matterCreatePending || matterConditionRows.length === 1;
  });
  byId("add-matter-condition").disabled = matterCreatePending || matterConditionRows.length >= maxMatterConditions;
}

function appendMatterCondition(focus = false) {
  if (matterCreatePending || matterConditionRows.length >= maxMatterConditions) return;
  const container = element("div", "matter-condition-row");
  const label = element("label");
  const caption = element("span");
  const input = element("input");
  input.name = "condition";
  input.maxLength = 2000;
  input.placeholder = "What would make it resolved?";
  input.required = true;
  input.addEventListener("input", () => input.setCustomValidity?.(""));
  label.append(caption, input);
  const remove = element("button", "text-button condition-remove", "Remove");
  remove.type = "button";
  const row = { container, label: caption, input, remove };
  remove.addEventListener("click", () => {
    if (matterCreatePending || matterConditionRows.length === 1) return;
    const index = matterConditionRows.indexOf(row);
    matterConditionRows.splice(index, 1);
    container.remove();
    updateMatterConditionControls();
    matterConditionRows[Math.min(index, matterConditionRows.length - 1)].input.focus?.();
  });
  container.append(label, remove);
  matterConditionRows.push(row);
  byId("matter-condition-fields").append(container);
  updateMatterConditionControls();
  if (focus) input.focus?.();
}

function resetMatterDraft() {
  matterDraftSerial += 1;
  matterCreatePending = false;
  matterForm.reset();
  formBusy(matterForm, false);
  matterConditionRows = [];
  const label = element("p", "matter-conditions-label", "Resolved when");
  label.id = "matter-conditions-label";
  byId("matter-condition-fields").replaceChildren(label);
  appendMatterCondition();
}

function resetAddConditionDraft() {
  addConditionSerial += 1;
  addConditionPending = false;
  addConditionForm.reset();
  for (const control of addConditionForm.elements) control.setCustomValidity?.("");
  formBusy(addConditionForm, false);
  byId("add-condition-panel").open = false;
  byId("add-condition-message").textContent = "";
}

function clearConditionDraft() {
  addConditionScopeKey = null;
  resetAddConditionDraft();
  byId("add-condition-panel").hidden = true;
}

function syncAddConditionForm() {
  const graph = state.graph;
  const writableSelection = graph && graph.role !== "viewer" && state.detailsMode === "matter"
    && graph.state.matters.includes(state.selectedMatter) && state.requestedGraph === graph.graph_id;
  const key = writableSelection ? JSON.stringify([state.authEpoch, state.user?.id, graph.graph_id, state.selectedMatter]) : null;
  if (key !== addConditionScopeKey) resetAddConditionDraft();
  addConditionScopeKey = key;
  byId("add-condition-panel").hidden = !key;
}

function closeMenus(restoreFocus = false) {
  const opened = menuIds.map(byId).find(menu => menu.open);
  for (const id of menuIds) byId(id).open = false;
  if (restoreFocus) opened?.querySelector?.("summary")?.focus();
}

function openMenu(id, field) {
  closeMenus();
  byId(id).open = true;
  if (field) field.focus?.();
}

function closeInspector(restoreFocus = false) {
  state.detailsMode = null;
  renderMatters();
  if (restoreFocus) {
    if (inspectorReturnFocus?.isConnected) inspectorReturnFocus.focus?.({ preventScroll: true });
    else byId("toggle-list").focus?.({ preventScroll: true });
  }
  inspectorReturnFocus = null;
}

function openInspector(mode) {
  if (!state.graph) return;
  inspectorReturnFocus = document.activeElement || null;
  state.detailsMode = mode;
  renderMatters();
  byId("matter-inspector").focus?.({ preventScroll: true });
}

function initialInvitation() {
  if (typeof window === "undefined" || !window.location) return null;
  const token = new URLSearchParams(window.location.hash.slice(1)).get("invite");
  if (token) window.history.replaceState(null, "", window.location.pathname + window.location.search);
  return token && /^[A-Za-z0-9_-]{43}$/.test(token) ? token : null;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function message(text, auth = false) {
  byId(auth ? "auth-message" : "workspace-message").textContent = text || "";
}

function formBusy(form, busy) {
  for (const control of form.elements) control.disabled = busy;
}

function staleRequest() {
  const error = new Error("");
  error.stale = true;
  return error;
}

function sameSession(epoch, userId) {
  return state.authEpoch === epoch && state.user?.id === userId;
}

function clearGraphDisplay() {
  closeMenus();
  graphForm.reset();
  resetMatterDraft();
  clearConditionDraft();
  formBusy(graphForm, false);
  for (const form of [invitationForm, joinForm, dependencyForm]) { form.reset(); formBusy(form, false); }
  byId("invitation-link").value = "";
  byId("visual-search").value = "";
  byId("visual-status").value = "all";
  byId("sharing-owner").textContent = "";
  byId("sharing-members").replaceChildren();
  byId("sharing-invitations").replaceChildren();
  byId("sharing-panel").hidden = true;
  byId("sharing-panel").open = false;
  byId("visual-selection").textContent = "";
  byId("visual-selection").hidden = true;
  state.visual = null;
  state.selectedMatter = null;
  state.detailsMode = null;
  inspectorReturnFocus = null;
  state.visualMode = "overview";
  state.sharingRequest += 1;
  lastVisualMode = null;
  overviewRenderer?.setModel({ nodes: [], edges: [] });
  byId("matters-list").replaceChildren();
  byId("graph-list").replaceChildren();
  byId("graph-title").textContent = "Graphs";
  byId("account-email").textContent = "";
  byId("view-only").hidden = true;
  byId("matter-inspector").hidden = true;
  byId("empty-matters-state").hidden = true;
  byId("toggle-list").setAttribute("aria-pressed", "false");
  byId("new-graph-panel").open = false;
  byId("new-matter-panel").open = false;
  byId("graph-detail").hidden = true;
  byId("no-graph").hidden = false;
}

function resetAccountControls() {
  state.accountSecurity = null;
  state.securityRequest += 1;
  state.recoveryCapabilities = null;
  state.recoveryRequest += 1;
  recoveryForm.reset(); passwordChangeForm.reset();
  formBusy(recoveryForm, false); formBusy(passwordChangeForm, false);
  for (const id of ["recovery-email", "current-password", "new-password", "confirm-new-password"]) byId(id).value = "";
  byId("account-security-panel").open = false;
  byId("recovery-panel").open = false;
  byId("security-verified-status").textContent = "Open this panel to check email verification.";
  byId("security-delivery").textContent = "";
  byId("security-message").textContent = "";
  byId("recovery-message").textContent = "";
  byId("recovery-capability").textContent = "Checking whether account recovery is available…";
  byId("request-verification").hidden = true;
  byId("request-verification").disabled = false;
  passwordChangeForm.hidden = true;
  recoveryForm.hidden = true;
}

function showAuth() {
  if (state.user) incomingInvitation = null;
  state.authEpoch += 1;
  state.authAttempt += 1;
  state.user = null;
  state.graph = null;
  state.report = null;
  state.graphs = [];
  state.requestedGraph = null;
  state.catalogRequest += 1;
  state.generation += 1;
  byId("workspace-view").hidden = true;
  byId("graph-menu").hidden = true;
  byId("new-matter-panel").hidden = true;
  byId("account-controls").hidden = true;
  byId("auth-view").hidden = false;
  clearGraphDisplay();
  resetAccountControls();
  formBusy(authForm, false);
  byId("auth-switch").disabled = !signupEnabled;
  byId("password").value = "";
  state.register = false;
  renderAuthMode();
  if (incomingInvitation) byId("join-token").value = incomingInvitation;
}

function showWorkspace(user) {
  state.authEpoch += 1;
  state.user = user;
  state.graphs = [];
  state.graph = null;
  state.report = null;
  state.requestedGraph = null;
  state.catalogRequest += 1;
  state.generation += 1;
  clearGraphDisplay();
  resetAccountControls();
  if (incomingInvitation) byId("join-token").value = incomingInvitation;
  byId("account-email").textContent = user.email;
  byId("auth-view").hidden = true;
  byId("account-controls").hidden = false;
  byId("graph-menu").hidden = false;
  byId("new-matter-panel").hidden = true;
  byId("workspace-view").hidden = false;
  message("");
}

async function api(path, options = {}) {
  const epoch = state.authEpoch;
  const headers = { Accept: "application/json" };
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  let payload;
  try { payload = await response.json(); }
  catch { payload = {}; }
  if (epoch !== state.authEpoch) throw staleRequest();
  if (!response.ok) {
    const error = new Error(payload.error?.message || "The request could not be completed. Please try again.");
    error.status = response.status;
    if (response.status === 401 && !path.endsWith("/login") && !path.endsWith("/register")) showAuth();
    throw error;
  }
  return payload;
}

function renderGraphs() {
  const list = byId("graph-list");
  list.replaceChildren();
  for (const graph of state.graphs) {
    const button = element("button", "graph-link");
    button.type = "button";
    if (state.graph?.graph_id === graph.graph_id) button.setAttribute("aria-current", "page");
    button.append(element("strong", "", graph.title), element("small", "", graph.kind === "personal" ? "Personal space" : "Project space"));
    button.addEventListener("click", () => {
      closeMenus();
      return loadGraph(graph.graph_id).catch(error => { if (!error.stale) message(error.message); });
    });
    list.append(button);
  }
}

function forgetCurrentGraph() {
  state.graph = null;
  state.report = null;
  state.visual = null;
  state.selectedMatter = null;
  state.detailsMode = null;
  inspectorReturnFocus = null;
  state.requestedGraph = null;
  state.generation += 1;
  state.sharingRequest += 1;
  lastVisualMode = null;
  overviewRenderer?.setModel({ nodes: [], edges: [] });
  resetMatterDraft(); clearConditionDraft(); dependencyForm.reset(); invitationForm.reset();
  formBusy(dependencyForm, false); formBusy(invitationForm, false);
  byId("invitation-link").value = "";
  byId("sharing-owner").textContent = "";
  byId("sharing-members").replaceChildren(); byId("sharing-invitations").replaceChildren();
  byId("sharing-panel").hidden = true; byId("sharing-panel").open = false;
  byId("matters-list").replaceChildren();
  byId("graph-title").textContent = "Graphs";
  byId("visual-selection").textContent = "";
  byId("visual-selection").hidden = true;
  byId("matter-inspector").hidden = true;
  byId("empty-matters-state").hidden = true;
  byId("new-matter-panel").hidden = true;
  byId("view-only").hidden = true;
  byId("toggle-list").setAttribute("aria-pressed", "false");
  closeMenus();
  byId("graph-detail").hidden = true;
  byId("no-graph").hidden = false;
}

async function loadGraphs(preferred) {
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  if (!userId) throw staleRequest();
  const request = ++state.catalogRequest;
  const navigation = state.generation;
  const data = await api("/api/graphs");
  if (!sameSession(epoch, userId) || request !== state.catalogRequest) throw staleRequest();
  state.graphs = data.graphs;
  let selected = navigation === state.generation
    ? preferred || state.requestedGraph || state.graph?.graph_id || state.graphs[0]?.graph_id
    : state.requestedGraph || state.graph?.graph_id;
  const permitted = new Set(state.graphs.map(graph => graph.graph_id));
  if (state.graph && !permitted.has(state.graph.graph_id)) {
    forgetCurrentGraph();
    message("The previous graph is no longer available to your account.");
  }
  if (!permitted.has(selected)) selected = state.graphs[0]?.graph_id;
  renderGraphs();
  if (selected) await loadGraph(selected);
  else {
    byId("no-graph").hidden = false;
    byId("graph-detail").hidden = true;
  }
}

async function loadGraph(id) {
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  if (!userId) throw staleRequest();
  if (state.requestedGraph !== id) {
    resetMatterDraft();
    clearConditionDraft();
  }
  const generation = ++state.generation;
  state.requestedGraph = id;
  let graph;
  let report;
  let visual;
  try {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      [graph, report, visual] = await Promise.all([
        api(`/api/graphs/${encodeURIComponent(id)}`),
        api(`/api/graphs/${encodeURIComponent(id)}/views/unlock`),
        api(`/api/graphs/${encodeURIComponent(id)}/visual`)
      ]);
      if (!sameSession(epoch, userId) || generation !== state.generation) throw staleRequest();
      if (graph.revision === report.revision && graph.revision === visual.revision) break;
    }
    if (!sameSession(epoch, userId) || generation !== state.generation) throw staleRequest();
    if (graph.revision !== report.revision || graph.revision !== visual.revision) throw new Error("This graph is changing. Refresh it in a moment.");
    const changedGraph = state.graph?.graph_id !== graph.graph_id;
    if (changedGraph) {
      state.selectedMatter = null;
      state.detailsMode = null;
      inspectorReturnFocus = null;
      closeMenus();
      resetMatterDraft();
      clearConditionDraft();
      state.visualMode = "overview";
      state.sharingRequest += 1;
      byId("invitation-link").value = "";
      byId("sharing-members").replaceChildren();
      byId("sharing-invitations").replaceChildren();
      byId("sharing-panel").open = false;
      invitationForm.reset();
      formBusy(invitationForm, false);
      lastVisualMode = null;
    }
    state.graph = graph;
    state.report = report;
    state.visual = visual;
    renderGraphs();
    renderGraph();
  } catch (error) {
    if (sameSession(epoch, userId) && generation === state.generation) {
      if (error.status === 403 || error.status === 404) {
        if (state.graph?.graph_id === id) forgetCurrentGraph();
        state.graphs = state.graphs.filter(graph => graph.graph_id !== id);
        renderGraphs();
      }
      state.requestedGraph = state.graph?.graph_id || null;
    }
    throw error;
  }
}

function renderGraph() {
  const graph = state.graph;
  const writable = graph.role !== "viewer";
  byId("no-graph").hidden = true;
  byId("graph-detail").hidden = false;
  byId("graph-title").textContent = graph.title;
  byId("view-only").hidden = writable;
  matterForm.hidden = !writable;
  byId("new-matter-panel").hidden = !writable;
  if (!writable) byId("new-matter-panel").open = false;
  byId("empty-matters-state").hidden = graph.state.matters.length !== 0;
  byId("add-first-matter").hidden = !writable;
  byId("dependency-editor").hidden = !writable;
  byId("sharing-panel").hidden = graph.role !== "owner";
  for (const id of ["dependency-source", "dependency-target"]) {
    const select = byId(id);
    const previous = select.value;
    select.replaceChildren();
    for (const matterId of graph.state.matters) {
      const option = element("option", "", matterId.replaceAll("_", " "));
      option.value = matterId;
      select.append(option);
    }
    if (graph.state.matters.includes(previous)) select.value = previous;
  }
  if (state.selectedMatter && !graph.state.matters.includes(state.selectedMatter)) {
    state.selectedMatter = null;
    if (state.detailsMode === "matter") state.detailsMode = null;
  }
  renderMatters();
  renderVisual().catch(error => { if (!error.stale) message(error.message); });
  if (graph.role === "owner" && byId("sharing-panel").open) {
    loadSharing().catch(error => { if (!error.stale) message(error.message); });
  }
}

function renderMatters() {
  const graph = state.graph;
  const list = byId("matters-list");
  list.replaceChildren();
  const visible = Boolean(graph && state.detailsMode);
  byId("matter-inspector").hidden = !visible;
  byId("toggle-list").setAttribute("aria-pressed", String(state.detailsMode === "list"));
  byId("inspector-title").hidden = state.detailsMode !== "list";
  byId("visual-selection").hidden = state.detailsMode !== "matter";
  syncAddConditionForm();
  if (!visible) return;
  const ready = new Set(state.report.universe);
  const blocked = new Set(state.report.blocked);
  const writable = graph.role !== "viewer";
  const search = byId("visual-search").value.trim().toLowerCase();
  const statusFilter = byId("visual-status").value || "all";
  const ids = state.detailsMode === "matter" ? [state.selectedMatter] : graph.state.matters.filter(id => {
    const text = `${id} ${id.replaceAll("_", " ")} ${(graph.state.conditions[id] || []).map(condition => condition.label).join(" ")}`.toLowerCase();
    const status = ready.has(id) ? "actionable" : blocked.has(id) ? "blocked" : "resolved";
    return (!search || text.includes(search)) && (statusFilter === "all" || statusFilter === status);
  });
  if (!ids.length) list.append(element("p", "empty-matters", graph.state.matters.length ? "No matching matters." : "No matters yet."));
  for (const id of ids) {
    if (!id || !graph.state.matters.includes(id)) continue;
    const card = element("article", "matter-card");
    card.setAttribute("data-matter-id", id);
    const top = element("div", "matter-top");
    const status = ready.has(id) ? "Ready" : blocked.has(id) ? "Waiting" : "Resolved";
    const title = element("button", "matter-select", id.replaceAll("_", " "));
    title.type = "button";
    title.addEventListener("click", () => selectMatter(id));
    top.append(title, element("span", `status-label status-${status.toLowerCase()}`, status));
    card.append(top);
    for (const [index, condition] of (graph.state.conditions[id] || []).entries()) {
      const label = element("label", "condition-label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = condition.truth;
      checkbox.disabled = !writable;
      checkbox.setAttribute("data-condition-index", String(index));
      checkbox.addEventListener("change", async () => {
        const epoch = state.authEpoch;
        const userId = state.user?.id;
        const graphId = graph.graph_id;
        const hadFocus = document.activeElement === checkbox;
        checkbox.disabled = true;
        try { await mutate("set_condition", { matter_id: id, index, truth: checkbox.checked }); }
        catch (error) { checkbox.checked = condition.truth; if (!error.stale) message(error.message); }
        finally {
          checkbox.disabled = !writable;
          if (hadFocus && sameSession(epoch, userId) && state.graph?.graph_id === graphId && (!document.activeElement || document.activeElement === document.body)) {
            for (const candidate of byId("matters-list").querySelectorAll?.("input[type=checkbox]") || []) {
              if (candidate.closest(".matter-card")?.getAttribute("data-matter-id") === id && candidate.getAttribute("data-condition-index") === String(index)) candidate.focus({ preventScroll: true });
            }
          }
        }
      });
      label.append(checkbox, element("span", "", condition.label));
      card.append(label);
    }
    list.append(card);
  }
}

function selectMatter(id) {
  if (!state.graph?.state.matters.includes(id) || state.requestedGraph !== state.graph.graph_id) return;
  state.selectedMatter = id;
  closeMenus();
  openInspector("matter");
  renderVisual().catch(error => { if (!error.stale) message(error.message); });
}

async function renderer() {
  if (overviewRenderer) return overviewRenderer;
  if (typeof window === "undefined" || typeof HTMLElement === "undefined") return null;
  if (!rendererLoading) {
    rendererLoading = import("/assets/local/map-renderer.js").then(module => {
      overviewRenderer = module.createOverviewRenderer({
        container: byId("visual-canvas"),
        onSelect: id => selectMatter(id),
        onError: () => {
          if (!state.graph) return;
          message("The map could not be displayed. Your matters are available in the list.");
          state.detailsMode = "list";
          renderMatters();
        },
        reducedMotion: Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches),
        minimal: true
      });
      return overviewRenderer;
    }).catch(error => { rendererLoading = null; throw error; });
  }
  return rendererLoading;
}

async function renderVisual() {
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  const graphId = state.graph?.graph_id;
  const visual = state.visual;
  if (!visual || !graphId) return;
  const nodes = visual.nodes || [];
  if (state.selectedMatter && !nodes.some(node => node.id === state.selectedMatter)) state.selectedMatter = null;
  const selected = nodes.find(node => node.id === state.selectedMatter);
  byId("visual-focus").disabled = !selected;
  byId("visual-overview").setAttribute("aria-pressed", String(state.visualMode === "overview"));
  byId("visual-focus").setAttribute("aria-pressed", String(state.visualMode === "focus"));
  byId("visual-selection").textContent = selected
    ? `Requires: ${selected.prerequisites.map(id => id.replaceAll("_", " ")).join(", ") || "none"}. Unlocks: ${selected.dependents.map(id => id.replaceAll("_", " ")).join(", ") || "none"}.`
    : "";
  byId("visual-selection").hidden = !selected || state.detailsMode !== "matter";
  const scene = await renderer();
  if (!scene || !sameSession(epoch, userId) || state.graph?.graph_id !== graphId || state.visual !== visual) return;
  const search = byId("visual-search").value.trim().toLowerCase();
  const status = byId("visual-status").value || "all";
  const matches = nodes.filter(node => (
    (!search || `${node.id} ${node.label} ${node.conditions.map(condition => condition.label).join(" ")}`.toLowerCase().includes(search))
    && (status === "all" || Boolean(node[status]))
  )).map(node => node.id);
  const context = state.visualMode === "focus" && selected
    ? new Set([selected.id, ...selected.prerequisites, ...selected.dependents]) : null;
  scene.setModel({
    graphId, nodes: context ? nodes.filter(node => context.has(node.id)) : nodes,
    edges: context ? (visual.edges || []).filter(edge => context.has(edge.source) && context.has(edge.target)) : visual.edges || [],
    selectedId: state.selectedMatter,
    ancestorIds: selected?.prerequisites || [], directDependentIds: selected?.dependents || [],
    matchIds: matches, filterActive: Boolean(search) || status !== "all"
  });
  if (lastVisualMode !== state.visualMode) {
    scene.resetCamera();
    if (state.visualMode === "focus") scene.setCamera({ pitch: -1.25, yaw: 0 });
    lastVisualMode = state.visualMode;
  }
  scene.resize();
}

function sharingScope() {
  return { epoch: state.authEpoch, userId: state.user?.id, graphId: state.graph?.graph_id };
}

function requireSharingScope(scope) {
  if (!scope.graphId || !sameSession(scope.epoch, scope.userId) || state.graph?.graph_id !== scope.graphId || state.requestedGraph !== scope.graphId) throw staleRequest();
}

async function loadSharing() {
  const scope = sharingScope();
  if (state.graph?.role !== "owner") throw staleRequest();
  const request = ++state.sharingRequest;
  const data = await api(`/api/graphs/${encodeURIComponent(scope.graphId)}/sharing`);
  requireSharingScope(scope);
  if (request !== state.sharingRequest) throw staleRequest();
  byId("sharing-owner").textContent = `Owned by ${data.owner.email}. Viewers can read; editors can change graph contents. Only you can manage access.`;
  const members = byId("sharing-members");
  members.replaceChildren();
  if (!data.members.length) members.append(element("p", "field-help", "No other people have access."));
  for (const member of data.members) {
    const row = element("div", "sharing-row");
    const role = document.createElement("select");
    role.setAttribute("aria-label", `Access for ${member.email}`);
    for (const [value, label] of [["viewer", "Can view"], ["editor", "Can edit"]]) {
      const option = element("option", "", label); option.value = value; role.append(option);
    }
    role.value = member.role;
    role.addEventListener("change", async () => {
      role.disabled = true;
      try {
        requireSharingScope(scope);
        await api(`/api/graphs/${encodeURIComponent(scope.graphId)}/members/${encodeURIComponent(member.user_id)}`, { method: "PUT", body: JSON.stringify({ role: role.value }) });
        requireSharingScope(scope);
        await loadSharing();
      } catch (error) { role.value = member.role; if (!error.stale) message(error.message); }
      finally { role.disabled = false; }
    });
    const remove = element("button", "text-button", "Remove access");
    remove.type = "button";
    remove.addEventListener("click", async () => {
      remove.disabled = true;
      try {
        requireSharingScope(scope);
        await api(`/api/graphs/${encodeURIComponent(scope.graphId)}/members/${encodeURIComponent(member.user_id)}`, { method: "DELETE" });
        requireSharingScope(scope);
        await loadSharing();
      } catch (error) { if (!error.stale) message(error.message); }
      finally { remove.disabled = false; }
    });
    row.append(element("span", "", member.email), role, remove);
    members.append(row);
  }
  const invitations = byId("sharing-invitations");
  invitations.replaceChildren();
  if (!data.invitations.length) invitations.append(element("p", "field-help", "No invitation links yet."));
  for (const invitation of data.invitations) {
    const row = element("div", "sharing-row");
    row.append(element("span", "", invitation.email), element("small", "", `${invitation.role} · ${invitation.status}`));
    if (invitation.status === "pending") {
      const revoke = element("button", "text-button", "Revoke link");
      revoke.type = "button";
      revoke.addEventListener("click", async () => {
        revoke.disabled = true;
        try {
          requireSharingScope(scope);
          await api(`/api/graphs/${encodeURIComponent(scope.graphId)}/invitations/${encodeURIComponent(invitation.id)}`, { method: "DELETE" });
          requireSharingScope(scope);
          await loadSharing();
        } catch (error) { if (!error.stale) message(error.message); }
        finally { revoke.disabled = false; }
      });
      row.append(revoke);
    }
    invitations.append(row);
  }
}

async function mutate(operation, payload) {
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  message("");
  const graph = state.graph;
  try {
    await api(`/api/graphs/${encodeURIComponent(graph.graph_id)}/operations`, {
      method: "POST", body: JSON.stringify({ operation, payload, expected_revision: graph.revision })
    });
    if (!sameSession(epoch, userId) || state.requestedGraph !== graph.graph_id) throw staleRequest();
    await loadGraph(graph.graph_id);
  } catch (error) {
    if (!sameSession(epoch, userId) || state.requestedGraph !== graph.graph_id) throw staleRequest();
    if (error.status === 409) {
      if (state.graph?.graph_id === graph.graph_id) await loadGraph(graph.graph_id);
      throw new Error("This graph changed. Review the refreshed graph and try your change again.");
    }
    throw error;
  }
}

function renderAuthMode() {
  if (!signupEnabled) state.register = false;
  byId("signup-controls").hidden = !signupEnabled;
  byId("signup-message").hidden = signupEnabled;
  byId("auth-title").textContent = state.register ? "Create account" : "Sign in";
  byId("auth-subtitle").textContent = "";
  byId("auth-subtitle").hidden = true;
  byId("auth-submit").textContent = state.register ? "Create account" : "Sign in";
  byId("auth-switch-label").textContent = state.register ? "Already have an account?" : "New to Matters?";
  byId("auth-switch").textContent = state.register ? "Sign in" : "Create an account";
  byId("password").autocomplete = state.register ? "new-password" : "current-password";
  byId("password").minLength = state.register ? 12 : 1;
  byId("password-help").hidden = !state.register;
  byId("recovery-panel").hidden = state.register;
  if (state.register) {
    byId("recovery-panel").open = false;
    state.recoveryRequest += 1;
    recoveryForm.reset();
    byId("recovery-email").value = "";
    byId("recovery-message").textContent = "";
  }
}

byId("auth-switch").addEventListener("click", () => {
  if (!signupEnabled) return;
  state.register = !state.register;
  renderAuthMode();
  message("", true);
});

authForm.addEventListener("submit", async event => {
  event.preventDefault();
  const register = state.register;
  const attempt = ++state.authAttempt;
  state.authEpoch += 1;
  state.generation += 1;
  const epoch = state.authEpoch;
  const credentials = { email: byId("email").value.trim(), password: byId("password").value };
  message("", true);
  formBusy(authForm, true);
  byId("auth-switch").disabled = true;
  try {
    if (register) await api("/api/auth/register", { method: "POST", body: JSON.stringify(credentials) });
    const result = await api("/api/auth/login", { method: "POST", body: JSON.stringify(credentials) });
    if (epoch !== state.authEpoch) throw staleRequest();
    byId("password").value = "";
    showWorkspace(result.user);
    await loadGraphs();
  } catch (error) { if (!error.stale) message(error.message, !state.user); }
  finally {
    if (attempt === state.authAttempt) {
      formBusy(authForm, false);
      byId("auth-switch").disabled = !signupEnabled;
    }
  }
});

byId("sign-out").addEventListener("click", async () => {
  state.authEpoch += 1;
  state.generation += 1;
  const epoch = state.authEpoch;
  byId("sign-out").disabled = true;
  try {
    await api("/api/auth/logout", { method: "POST" });
    if (epoch !== state.authEpoch) throw staleRequest();
    showAuth();
    message("", true);
  } catch (error) { if (!error.stale) message(error.message); }
  finally { byId("sign-out").disabled = false; }
});

graphForm.addEventListener("submit", async event => {
  event.preventDefault();
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  const navigation = state.generation;
  const values = new FormData(graphForm);
  formBusy(graphForm, true);
  message("");
  try {
    const graph = await api("/api/graphs", { method: "POST", body: JSON.stringify({ title: values.get("title").trim(), kind: values.get("kind") }) });
    if (!sameSession(epoch, userId)) throw staleRequest();
    graphForm.reset();
    await loadGraphs(navigation === state.generation ? graph.graph_id : state.requestedGraph);
    if (state.graph?.graph_id === graph.graph_id) {
      byId("new-graph-panel").open = false;
    }
  } catch (error) { if (!error.stale) message(error.message); }
  finally { formBusy(graphForm, false); }
});

matterForm.addEventListener("submit", async event => {
  event.preventDefault();
  if (matterCreatePending || !state.user || !state.graph || state.graph.role === "viewer" || state.requestedGraph !== state.graph.graph_id) return;
  const values = new FormData(matterForm);
  const id = values.get("name").normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  if (!id) { message("Use a matter name containing a letter or number."); return; }
  const conditions = values.getAll("condition").map(value => ({ label: value.trim(), truth: false }));
  if (!conditions.length || conditions.length > maxMatterConditions) {
    message("Use between 1 and 1000 conditions.");
    return;
  }
  const invalid = conditions.findIndex(condition => !condition.label || condition.label.length > 2000);
  if (invalid !== -1) {
    const input = matterConditionRows[invalid]?.input;
    const error = conditions[invalid].label ? "Use at most 2000 characters for a condition." : "Enter a condition or remove this row.";
    input?.setCustomValidity?.(error);
    input?.reportValidity?.();
    input?.focus?.();
    message(error);
    return;
  }
  const epoch = state.authEpoch;
  const userId = state.user.id;
  const graphId = state.graph.graph_id;
  const serial = matterDraftSerial;
  const selected = state.selectedMatter;
  const current = () => serial === matterDraftSerial && sameSession(epoch, userId)
    && state.graph?.graph_id === graphId && state.requestedGraph === graphId;
  matterCreatePending = true;
  formBusy(matterForm, true);
  try {
    await mutate("create_matter", { id, conditions });
    if (!current()) throw staleRequest();
    resetMatterDraft();
    if (state.selectedMatter === selected) selectMatter(id);
  } catch (error) { if (!error.stale && current()) message(error.message); }
  finally {
    if (current()) {
      matterCreatePending = false;
      formBusy(matterForm, false);
      updateMatterConditionControls();
    }
  }
});

byId("add-matter-condition").addEventListener("click", () => appendMatterCondition(true));

addConditionForm.addEventListener("input", event => event.target?.setCustomValidity?.(""));
addConditionForm.addEventListener("submit", async event => {
  event.preventDefault();
  if (addConditionPending || !addConditionScopeKey || !state.user || !state.graph || state.graph.role === "viewer") return;
  const label = String(new FormData(addConditionForm).get("condition") || "").trim();
  if (!label || label.length > 2000) {
    const input = addConditionForm.querySelector?.('input[name="condition"]');
    const error = label ? "Use at most 2000 characters for a condition." : "Enter a condition.";
    input?.setCustomValidity?.(error);
    input?.reportValidity?.();
    input?.focus?.();
    byId("add-condition-message").textContent = error;
    return;
  }
  const epoch = state.authEpoch;
  const userId = state.user.id;
  const graphId = state.graph.graph_id;
  const matterId = state.selectedMatter;
  const scope = addConditionScopeKey;
  const serial = addConditionSerial;
  const current = () => serial === addConditionSerial && scope === addConditionScopeKey && sameSession(epoch, userId)
    && state.graph?.graph_id === graphId && state.requestedGraph === graphId
    && state.selectedMatter === matterId && state.detailsMode === "matter" && state.graph.role !== "viewer";
  if (!current()) return;
  addConditionPending = true;
  byId("add-condition-message").textContent = "";
  formBusy(addConditionForm, true);
  try {
    await mutate("add_condition", { matter_id: matterId, label, truth: false });
    if (!current()) throw staleRequest();
    resetAddConditionDraft();
  } catch (error) {
    if (!error.stale && current()) byId("add-condition-message").textContent = error.message;
  } finally {
    if (current()) {
      addConditionPending = false;
      formBusy(addConditionForm, false);
    }
  }
});

byId("refresh-graph").addEventListener("click", () => {
  if (state.graph) loadGraphs(state.requestedGraph || state.graph.graph_id).catch(error => { if (!error.stale) message(error.message); });
});

byId("visual-overview").addEventListener("click", () => { state.visualMode = "overview"; renderVisual().catch(error => { if (!error.stale) message(error.message); }); });
byId("visual-focus").addEventListener("click", () => { state.visualMode = "focus"; renderVisual().catch(error => { if (!error.stale) message(error.message); }); });
byId("visual-search").addEventListener("input", () => {
  renderMatters();
  renderVisual().catch(error => { if (!error.stale) message(error.message); });
});
byId("visual-status").addEventListener("change", () => {
  renderMatters();
  renderVisual().catch(error => { if (!error.stale) message(error.message); });
});
byId("visual-zoom-in").addEventListener("click", () => overviewRenderer?.zoom(1.2));
byId("visual-zoom-out").addEventListener("click", () => overviewRenderer?.zoom(1 / 1.2));
byId("visual-reset").addEventListener("click", () => overviewRenderer?.resetCamera());
byId("toggle-list").addEventListener("click", () => {
  closeMenus();
  if (state.detailsMode === "list") closeInspector(true);
  else openInspector("list");
});
byId("close-inspector").addEventListener("click", () => closeInspector(true));
byId("create-first-graph").addEventListener("click", () => {
  openMenu("graph-menu");
  byId("new-graph-panel").open = true;
  graphForm.querySelector?.('input[name="title"]')?.focus();
});
byId("add-first-matter").addEventListener("click", () => openMenu("new-matter-panel", matterForm.querySelector?.('input[name="name"]')));

for (const id of menuIds) {
  byId(id).addEventListener("toggle", () => {
    if (!byId(id).open) return;
    for (const other of menuIds) if (other !== id) byId(other).open = false;
  });
}
document.addEventListener?.("pointerdown", event => {
  for (const id of menuIds) {
    const menu = byId(id);
    if (menu.open && !menu.contains(event.target)) menu.open = false;
  }
});
document.addEventListener?.("keydown", event => {
  if (event.key !== "Escape") return;
  if (menuIds.some(id => byId(id).open)) {
    closeMenus(true);
    event.preventDefault();
  } else if (state.detailsMode) {
    closeInspector(true);
    event.preventDefault();
  }
});
byId("sharing-panel").addEventListener("toggle", () => {
  if (byId("sharing-panel").open && state.graph?.role === "owner") loadSharing().catch(error => { if (!error.stale) message(error.message); });
});

invitationForm.addEventListener("submit", async event => {
  event.preventDefault();
  const scope = sharingScope();
  formBusy(invitationForm, true);
  try {
    requireSharingScope(scope);
    const created = await api(`/api/graphs/${encodeURIComponent(scope.graphId)}/invitations`, {
      method: "POST", body: JSON.stringify({ email: byId("invite-email").value.trim(), role: byId("invite-role").value })
    });
    requireSharingScope(scope);
    const origin = typeof window === "undefined" ? "" : window.location.origin;
    byId("invitation-link").value = `${origin}/#invite=${created.token}`;
    invitationForm.reset();
    await loadSharing();
    message("Invitation link created. Copy it for the invited person; no email was sent.");
  } catch (error) { if (!error.stale) message(error.message); }
  finally { if (sameSession(scope.epoch, scope.userId)) formBusy(invitationForm, false); }
});

joinForm.addEventListener("submit", async event => {
  event.preventDefault();
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  const navigation = state.generation;
  let token = byId("join-token").value.trim();
  if (token.includes("#")) token = new URLSearchParams(token.split("#", 2)[1]).get("invite") || "";
  if (!/^[A-Za-z0-9_-]{43}$/.test(token)) { message("Paste the invitation link or token the owner gave you."); return; }
  formBusy(joinForm, true);
  try {
    const accepted = await api("/api/invitations/accept", { method: "POST", body: JSON.stringify({ token }) });
    if (!sameSession(epoch, userId)) throw staleRequest();
    incomingInvitation = null;
    joinForm.reset();
    await loadGraphs(navigation === state.generation ? accepted.graph_id : state.requestedGraph || state.graph?.graph_id);
    message("You now have access to this graph.");
  } catch (error) { if (!error.stale) message(error.message); }
  finally { if (sameSession(epoch, userId)) formBusy(joinForm, false); }
});

dependencyForm.addEventListener("submit", async event => {
  event.preventDefault();
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  formBusy(dependencyForm, true);
  try {
    await mutate(byId("dependency-operation").value, { source: byId("dependency-source").value, target: byId("dependency-target").value });
  } catch (error) { if (!error.stale) message(error.message); }
  finally { if (sameSession(epoch, userId)) formBusy(dependencyForm, false); }
});

function securityMessage(text) {
  byId("security-message").textContent = text || "";
}

function hasAccountCsrf(data) {
  return typeof data?.csrf_token === "string" && data.csrf_token.length > 0 && data.csrf_token.length <= 512;
}

async function loadRecoveryCapabilities() {
  const epoch = state.authEpoch;
  const request = ++state.recoveryRequest;
  recoveryForm.hidden = true;
  byId("recovery-message").textContent = "";
  byId("recovery-capability").textContent = "Checking whether account recovery is available…";
  try {
    const data = await api("/api/auth/capabilities");
    if (epoch !== state.authEpoch || request !== state.recoveryRequest || state.user) throw staleRequest();
    if (typeof data.email_delivery !== "boolean" || typeof data.password_reset !== "boolean") throw new Error("Account recovery status is unavailable. Try again later.");
    state.recoveryCapabilities = { email_delivery: data.email_delivery, password_reset: data.password_reset };
    const available = data.email_delivery && data.password_reset;
    recoveryForm.hidden = !available;
    byId("recovery-capability").textContent = available
      ? "Request a single-use link to choose a new password."
      : "Email recovery is not configured for this installation. Contact the installation operator for help.";
    if (available && !byId("recovery-email").value) byId("recovery-email").value = byId("email").value.trim();
  } catch (error) {
    if (!error.stale && epoch === state.authEpoch && request === state.recoveryRequest) {
      state.recoveryCapabilities = null;
      byId("recovery-capability").textContent = "Account recovery status is unavailable. Try again later.";
    }
  }
}

async function loadAccountSecurity() {
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  const request = ++state.securityRequest;
  if (!userId) throw staleRequest();
  state.accountSecurity = null;
  byId("security-verified-status").textContent = "Checking account security…";
  byId("request-verification").hidden = true;
  passwordChangeForm.hidden = true;
  try {
    const data = await api("/api/auth/security");
    if (!sameSession(epoch, userId) || request !== state.securityRequest) throw staleRequest();
    if (typeof data.email_verified !== "boolean") throw new Error("Account security status is unavailable. Try again later.");
    state.accountSecurity = data;
    byId("security-verified-status").textContent = data.email_verified ? "Email verified." : "Your email has not been verified yet.";
    byId("security-delivery").textContent = data.email_delivery === true
      ? "Verification and recovery emails are available."
      : "Email verification and recovery delivery are not configured for this installation. Password changes remain available.";
    byId("request-verification").hidden = !(data.email_delivery === true && data.email_verification === true && !data.email_verified && hasAccountCsrf(data));
    passwordChangeForm.hidden = !(data.password_change === true && hasAccountCsrf(data));
    return data;
  } catch (error) {
    if (sameSession(epoch, userId) && request === state.securityRequest) {
      state.accountSecurity = null;
      byId("security-verified-status").textContent = "Account security status is unavailable.";
      byId("security-delivery").textContent = "";
      securityMessage(error.message || "Try again later.");
    }
    throw error;
  }
}

byId("recovery-panel").addEventListener("toggle", () => {
  if (byId("recovery-panel").open && !state.user && !state.register) loadRecoveryCapabilities();
});

recoveryForm.addEventListener("submit", async event => {
  event.preventDefault();
  const epoch = state.authEpoch;
  const request = ++state.recoveryRequest;
  if (!state.recoveryCapabilities?.email_delivery || !state.recoveryCapabilities?.password_reset) {
    byId("recovery-message").textContent = "Email recovery is not configured for this installation.";
    return;
  }
  const email = byId("recovery-email").value.trim();
  formBusy(recoveryForm, true);
  try {
    const result = await api("/api/auth/password/reset/request", { method: "POST", body: JSON.stringify({ email }) });
    if (epoch !== state.authEpoch || request !== state.recoveryRequest || state.user) throw staleRequest();
    recoveryForm.reset();
    byId("recovery-email").value = "";
    byId("recovery-message").textContent = result.message || "If recovery is available for this address, instructions may arrive shortly. Check your inbox or contact the installation operator.";
  } catch (error) {
    if (!error.stale && epoch === state.authEpoch && request === state.recoveryRequest && !state.user) byId("recovery-message").textContent = error.message;
  } finally {
    if (epoch === state.authEpoch && request === state.recoveryRequest) formBusy(recoveryForm, false);
  }
});

byId("account-security-panel").addEventListener("toggle", () => {
  if (byId("account-security-panel").open && state.user) {
    securityMessage("");
    loadAccountSecurity().catch(() => {});
  }
});

byId("request-verification").addEventListener("click", async () => {
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  const security = state.accountSecurity;
  if (!userId || !hasAccountCsrf(security) || security.email_delivery !== true || security.email_verification !== true || security.email_verified) return;
  byId("request-verification").disabled = true;
  securityMessage("");
  try {
    const result = await api("/api/auth/verification/request", { method: "POST", body: JSON.stringify({ csrf_token: security.csrf_token }) });
    if (!sameSession(epoch, userId)) throw staleRequest();
    await loadAccountSecurity();
    securityMessage(result.message || "If verification is available, instructions may arrive shortly. Check your inbox or contact the installation operator.");
  } catch (error) { if (!error.stale && sameSession(epoch, userId)) securityMessage(error.message); }
  finally { if (sameSession(epoch, userId)) byId("request-verification").disabled = false; }
});

passwordChangeForm.addEventListener("submit", async event => {
  event.preventDefault();
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  const security = state.accountSecurity;
  securityMessage("");
  if (!userId || !hasAccountCsrf(security) || security.password_change !== true) {
    securityMessage("Open Account security again to refresh this form.");
    return;
  }
  const currentPassword = byId("current-password").value;
  const newPassword = byId("new-password").value;
  if (!currentPassword) { securityMessage("Enter your current password."); return; }
  if (newPassword.length < 12 || newPassword.length > 1024) { securityMessage("Use between 12 and 1024 characters for the new password."); return; }
  if (newPassword !== byId("confirm-new-password").value) { securityMessage("The new passwords do not match."); return; }
  const email = state.user.email;
  formBusy(passwordChangeForm, true);
  for (const id of ["current-password", "new-password", "confirm-new-password"]) byId(id).value = "";
  try {
    const result = await api("/api/auth/password/change", {
      method: "POST", body: JSON.stringify({ current_password: currentPassword, new_password: newPassword, csrf_token: security.csrf_token })
    });
    if (!sameSession(epoch, userId)) throw staleRequest();
    showAuth();
    byId("email").value = email;
    message(result.password_changed === true && result.sign_in_required === true
      ? "Password changed. Sign in with your new password."
      : "Sign in again to check your account.", true);
  } catch (error) { if (!error.stale && sameSession(epoch, userId)) securityMessage(error.message); }
  finally { if (sameSession(epoch, userId)) formBusy(passwordChangeForm, false); }
});

resetMatterDraft();

(async () => {
  const epoch = state.authEpoch;
  try {
    const result = await api("/api/auth/me");
    if (epoch !== state.authEpoch) throw staleRequest();
    showWorkspace(result.user);
    await loadGraphs();
  } catch (error) {
    if (error.stale || (epoch !== state.authEpoch && error.status !== 401)) return;
    if (error.status === 401) return;
    showAuth();
    if (error.status !== 401) message(error.message, true);
  }
})();
