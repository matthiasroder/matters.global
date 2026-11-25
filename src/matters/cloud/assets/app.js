"use strict";

const state = { user: null, graphs: [], graph: null, report: null, register: false, generation: 0, authEpoch: 0, authAttempt: 0, requestedGraph: null, catalogRequest: 0 };
const byId = id => document.getElementById(id);
const authForm = byId("auth-form");
const graphForm = byId("new-graph-form");
const matterForm = byId("new-matter-form");

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
  byId("matters-list").replaceChildren();
  byId("graph-list").replaceChildren();
  for (const id of ["graph-title", "graph-kind", "account-email"]) byId(id).textContent = "";
  for (const id of ["graph-count", "ready-count", "open-count", "resolved-count"]) byId(id).textContent = "0";
  byId("graph-detail").hidden = true;
  byId("no-graph").hidden = false;
}

function showAuth() {
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
  formBusy(authForm, false);
  byId("auth-switch").disabled = false;
  byId("password").value = "";
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

async function loadGraphs(preferred) {
  const epoch = state.authEpoch;
  const userId = state.user?.id;
  if (!userId) throw staleRequest();
  const request = ++state.catalogRequest;
  const navigation = state.generation;
  const data = await api("/api/graphs");
  if (!sameSession(epoch, userId) || request !== state.catalogRequest) throw staleRequest();
  state.graphs = data.graphs;
  renderGraphs();
  const selected = navigation === state.generation
    ? preferred || state.requestedGraph || state.graph?.graph_id || state.graphs[0]?.graph_id
    : state.requestedGraph || state.graph?.graph_id;
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
  try {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      [graph, report] = await Promise.all([
        api(`/api/graphs/${encodeURIComponent(id)}`),
        api(`/api/graphs/${encodeURIComponent(id)}/views/unlock`)
      ]);
      if (!sameSession(epoch, userId) || generation !== state.generation) throw staleRequest();
      if (graph.revision === report.revision) break;
    }
    if (!sameSession(epoch, userId) || generation !== state.generation) throw staleRequest();
    if (graph.revision !== report.revision) throw new Error("This graph is changing. Refresh it in a moment.");
    state.graph = graph;
    state.report = report;
    renderGraphs();
    renderGraph();
  } catch (error) {
    if (sameSession(epoch, userId) && generation === state.generation) state.requestedGraph = state.graph?.graph_id || null;
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
  byId("graph-kind").textContent = graph.kind === "personal" ? "Your personal space" : "Your project space";
  byId("ready-count").textContent = ready.size;
  byId("open-count").textContent = ready.size + blocked.size;
  byId("resolved-count").textContent = graph.state.matters.length - ready.size - blocked.size;
  byId("view-only").hidden = writable;
  matterForm.hidden = !writable;
  const list = byId("matters-list");
  list.replaceChildren();
  if (!graph.state.matters.length) {
    list.append(element("p", "empty-matters", "No matters yet. Add the first concern you would like to move forward."));
  }
  for (const id of graph.state.matters) {
    const card = element("article", "matter-card");
    const top = element("div", "matter-top");
    const status = ready.has(id) ? "Ready" : blocked.has(id) ? "Waiting" : "Resolved";
    top.append(element("h3", "", id.replaceAll("_", " ")), element("span", `status-label status-${status.toLowerCase()}`, status));
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

byId("auth-switch").addEventListener("click", () => {
  state.register = !state.register;
  byId("auth-title").textContent = state.register ? "A space of your own" : "Welcome back";
  byId("auth-subtitle").textContent = state.register ? "Create an account to start your first graph." : "Sign in to pick up where you left off.";
  byId("auth-submit").textContent = state.register ? "Create account" : "Sign in";
  byId("auth-switch-label").textContent = state.register ? "Already have an account?" : "New to Matters?";
  byId("auth-switch").textContent = state.register ? "Sign in" : "Create an account";
  byId("password").autocomplete = state.register ? "new-password" : "current-password";
  byId("password").minLength = state.register ? 12 : 1;
  byId("password-help").hidden = !state.register;
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
