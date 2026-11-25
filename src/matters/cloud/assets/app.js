"use strict";

const state = { user: null, graphs: [], graph: null, report: null, visual: null, selectedMatter: null, visualMode: "overview", sharingRequest: 0, accountSecurity: null, securityRequest: 0, recoveryCapabilities: null, recoveryRequest: 0, register: false, generation: 0, authEpoch: 0, authAttempt: 0, requestedGraph: null, catalogRequest: 0 };
const byId = id => document.getElementById(id);
const authForm = byId("auth-form");
const graphForm = byId("new-graph-form");
const matterForm = byId("new-matter-form");
const invitationForm = byId("invitation-form");
const joinForm = byId("join-graph-form");
const dependencyForm = byId("dependency-form");
const recoveryForm = byId("recovery-request-form");
const passwordChangeForm = byId("password-change-form");
let overviewRenderer = null;
let rendererLoading = null;
let lastVisualMode = null;
let incomingInvitation = initialInvitation();

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
  graphForm.reset();
  matterForm.reset();
  formBusy(graphForm, false);
  formBusy(matterForm, false);
  for (const form of [invitationForm, joinForm, dependencyForm]) { form.reset(); formBusy(form, false); }
  byId("invitation-link").value = "";
  byId("visual-search").value = "";
  byId("visual-status").value = "all";
  byId("sharing-owner").textContent = "";
  byId("sharing-members").replaceChildren();
  byId("sharing-invitations").replaceChildren();
  byId("sharing-panel").hidden = true;
  byId("sharing-panel").open = false;
  byId("visual-selection").textContent = "Select a matter in the overview or list.";
  state.visual = null;
  state.selectedMatter = null;
  state.visualMode = "overview";
  state.sharingRequest += 1;
  lastVisualMode = null;
  overviewRenderer?.setModel({ nodes: [], edges: [] });
  byId("matters-list").replaceChildren();
  byId("graph-list").replaceChildren();
  for (const id of ["graph-title", "graph-kind", "account-email"]) byId(id).textContent = "";
  for (const id of ["graph-count", "ready-count", "open-count", "resolved-count"]) byId(id).textContent = "0";
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
  byId("account-controls").hidden = true;
  byId("auth-view").hidden = false;
  clearGraphDisplay();
  resetAccountControls();
  formBusy(authForm, false);
  byId("auth-switch").disabled = false;
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
  byId("graph-count").textContent = state.graphs.length;
  for (const graph of state.graphs) {
    const button = element("button", "graph-link");
    button.type = "button";
    if (state.graph?.graph_id === graph.graph_id) button.setAttribute("aria-current", "page");
    button.append(element("strong", "", graph.title), element("small", "", graph.kind === "personal" ? "Personal space" : "Project space"));
    button.addEventListener("click", () => loadGraph(graph.graph_id).catch(error => { if (!error.stale) message(error.message); }));
    list.append(button);
  }
}

function forgetCurrentGraph() {
  state.graph = null;
  state.report = null;
  state.visual = null;
  state.selectedMatter = null;
  state.requestedGraph = null;
  state.generation += 1;
  state.sharingRequest += 1;
  lastVisualMode = null;
  overviewRenderer?.setModel({ nodes: [], edges: [] });
  matterForm.reset(); dependencyForm.reset(); invitationForm.reset();
  formBusy(matterForm, false); formBusy(dependencyForm, false); formBusy(invitationForm, false);
  byId("invitation-link").value = "";
  byId("sharing-owner").textContent = "";
  byId("sharing-members").replaceChildren(); byId("sharing-invitations").replaceChildren();
  byId("sharing-panel").hidden = true; byId("sharing-panel").open = false;
  byId("matters-list").replaceChildren();
  byId("graph-title").textContent = ""; byId("graph-kind").textContent = "";
  byId("visual-selection").textContent = "Select a matter in the overview or list.";
  for (const id of ["ready-count", "open-count", "resolved-count"]) byId(id).textContent = "0";
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
  const ready = new Set(state.report.universe);
  const blocked = new Set(state.report.blocked);
  const writable = graph.role !== "viewer";
  byId("no-graph").hidden = true;
  byId("graph-detail").hidden = false;
  byId("graph-title").textContent = graph.title;
  byId("graph-kind").textContent = `${graph.role === "owner" ? "Your" : "Shared"} ${graph.kind === "personal" ? "personal" : "project"} space`;
  byId("ready-count").textContent = ready.size;
  byId("open-count").textContent = ready.size + blocked.size;
  byId("resolved-count").textContent = graph.state.matters.length - ready.size - blocked.size;
  byId("view-only").hidden = writable;
  matterForm.hidden = !writable;
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
  const list = byId("matters-list");
  list.replaceChildren();
  if (!graph.state.matters.length) {
    list.append(element("p", "empty-matters", "No matters yet. Add the first concern you would like to move forward."));
  }
  for (const id of graph.state.matters) {
    const card = element("article", "matter-card");
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
      checkbox.addEventListener("change", async () => {
        checkbox.disabled = true;
        try { await mutate("set_condition", { matter_id: id, index, truth: checkbox.checked }); }
        catch (error) { checkbox.checked = condition.truth; if (!error.stale) message(error.message); }
        finally { checkbox.disabled = !writable; }
      });
      label.append(checkbox, element("span", "", condition.label));
      card.append(label);
    }
    list.append(card);
  }
  renderVisual().catch(error => { if (!error.stale) message(error.message); });
  if (graph.role === "owner" && byId("sharing-panel").open) {
    loadSharing().catch(error => { if (!error.stale) message(error.message); });
  }
}

function selectMatter(id) {
  state.selectedMatter = id;
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
        onError: () => message("The visual overview could not be displayed. Your matters are available below."),
        reducedMotion: Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches)
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
    ? `${selected.label}. Prerequisites: ${selected.prerequisites.map(id => id.replaceAll("_", " ")).join(", ") || "none"}. Dependents: ${selected.dependents.map(id => id.replaceAll("_", " ")).join(", ") || "none"}.`
    : "Select a matter in the overview or list.";
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
  byId("auth-title").textContent = state.register ? "A space of your own" : "Welcome back";
  byId("auth-subtitle").textContent = state.register ? "Create an account to start your first graph." : "Sign in to pick up where you left off.";
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
      byId("auth-switch").disabled = false;
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
  } catch (error) { if (!error.stale) message(error.message); }
  finally { formBusy(graphForm, false); }
});

matterForm.addEventListener("submit", async event => {
  event.preventDefault();
  const values = new FormData(matterForm);
  const id = values.get("name").normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  if (!id) { message("Use a matter name containing a letter or number."); return; }
  formBusy(matterForm, true);
  try {
    await mutate("create_matter", { id, conditions: [{ label: values.get("condition").trim(), truth: false }] });
    matterForm.reset();
  } catch (error) { if (!error.stale) message(error.message); }
  finally { formBusy(matterForm, false); }
});

byId("refresh-graph").addEventListener("click", () => {
  if (state.graph) loadGraphs(state.requestedGraph || state.graph.graph_id).catch(error => { if (!error.stale) message(error.message); });
});

byId("visual-overview").addEventListener("click", () => { state.visualMode = "overview"; renderVisual().catch(error => { if (!error.stale) message(error.message); }); });
byId("visual-focus").addEventListener("click", () => { state.visualMode = "focus"; renderVisual().catch(error => { if (!error.stale) message(error.message); }); });
byId("visual-search").addEventListener("input", () => renderVisual().catch(error => { if (!error.stale) message(error.message); }));
byId("visual-status").addEventListener("change", () => renderVisual().catch(error => { if (!error.stale) message(error.message); }));
byId("visual-zoom-in").addEventListener("click", () => overviewRenderer?.zoom(1.2));
byId("visual-zoom-out").addEventListener("click", () => overviewRenderer?.zoom(1 / 1.2));
byId("visual-reset").addEventListener("click", () => overviewRenderer?.resetCamera());
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
