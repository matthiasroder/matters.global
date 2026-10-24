const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "../src/matters/web_assets/app.js"), "utf8"
).replace(/^import .*;\n/gm, "").replace(/^initOverview\(\);$/m, "");

function response(payload) {
  return {
    ok: true,
    headers: { get: () => "application/json" },
    json: async () => payload
  };
}

function loadApp() {
  const intervals = new Map();
  const intervalDelays = new Map();
  const elements = new Map();
  const element = () => ({
    value: "", children: [], addEventListener() {},
    replaceChildren() { this.children = []; },
    append(...children) { this.children.push(...children); }
  });
  let nextTimer = 1;
  const context = vm.createContext({
    URLSearchParams,
    cytoscape: Object.assign(() => ({ on() {} }), { use() {} }),
    dagre: {},
    createOverviewRenderer: () => ({}),
    document: {
      createElement: element,
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, element());
        return elements.get(selector);
      }
    },
    window: {
      location: { search: "?token=test-token", pathname: "/", hash: "" },
      history: { replaceState() {} },
      matchMedia: () => ({ matches: false }),
      addEventListener() {},
      requestAnimationFrame() {},
      setInterval(callback, delay) {
        const id = nextTimer++;
        intervals.set(id, callback);
        intervalDelays.set(id, delay);
        return id;
      },
      clearInterval(id) { intervals.delete(id); intervalDelays.delete(id); }
    },
    // Startup's graph load stays pending; these tests do not render a graph.
    fetch: () => new Promise(() => {})
  });
  vm.runInContext(source + `
    render = () => {};
    globalThis.app = {state, api, pollTerminal, restartTerminal, startTerminalPolling,
      loadGraph, toggleCondition, updateCondition, switchGraphState, showGraphError,
      sendTerminalInput};
  `, context);
  const output = [];
  context.app.state.terminal = { rows: 24, cols: 100, clear() {}, write(text) { output.push(text); } };
  context.app.state.terminalSessionId = "old";
  return { context, app: context.app, intervals, intervalDelays, output, elements };
}

async function settle() {
  await new Promise((resolve) => setImmediate(resolve));
}

for (const failure of ["exit", "read error"]) {
  test(`restarting after ${failure} resumes output`, async () => {
    const { context, app, intervals, output } = loadApp();
    context.fetch = async () => {
      if (failure === "read error") throw new Error("connection lost");
      return response({ chunks: [], closed: true });
    };
    app.startTerminalPolling();
    await settle();
    assert.equal(intervals.size, 0);

    context.fetch = async (url, options) => {
      if (options.method === "DELETE") return response({ closed: true });
      if (options.method === "POST") return response({ id: "new", workspace: "/tmp" });
      assert.match(url, /sessions\/new\/output/);
      return response({ chunks: [{ seq: 1, data: "new shell prompt" }], closed: false });
    };
    await app.restartTerminal();
    await settle();

    assert.equal(intervals.size, 1);
    assert.equal(app.state.terminalSessionId, "new");
    assert.deepEqual(output, ["new shell prompt"]);
  });
}

for (const staleResult of ["exit", "read error"]) {
  test(`an old session's delayed ${staleResult} cannot stop its replacement`, async () => {
    const { context, app, intervals, output } = loadApp();
    let finishOld, failOld;
    context.fetch = () => new Promise((resolve, reject) => { finishOld = resolve; failOld = reject; });
    app.startTerminalPolling();
    context.fetch = async (_url, options) => {
      if (options.method === "DELETE") return response({ closed: true });
      if (options.method === "POST") return response({ id: "new", workspace: "/tmp" });
      return response({ chunks: [{ seq: 1, data: "new output" }], closed: false });
    };
    await app.restartTerminal();
    if (staleResult === "exit") {
      finishOld(response({ chunks: [{ seq: 99, data: "old output" }], closed: true }));
    } else {
      failOld(new Error("old read failed"));
    }
    await settle();
    assert.equal(intervals.size, 1);
    assert.equal(app.state.terminalSeq, 0);
    for (const callback of intervals.values()) await callback();
    assert.deepEqual(output, ["new output"]);
    assert.equal(app.state.terminalSeq, 1);
  });
}

test("graph requests carry the identity captured when they start", async () => {
  const { context, app } = loadApp();
  app.state.graph = { graph_id: "first", revision: "revision-one" };
  let request, finish;
  context.fetch = (_url, options) => {
    request = options;
    return new Promise((resolve) => { finish = resolve; });
  };
  const pending = app.api("/api/matters/b/conditions", { method: "PATCH" });
  app.state.graph = { graph_id: "second" };
  assert.equal(request.headers["X-Matters-Graph"], "first");
  assert.equal(request.headers["X-Matters-Revision"], "revision-one");
  assert.equal(request.headers.Authorization, "Bearer test-token");
  finish(response({ graph_id: "first" }));
  await pending;
});

function graph(id, revision = "r1") {
  return { graph_id: id, revision, state_path: `/${id}.json`, edges: [], universe: [],
    nodes: [{ id: "a", conditions: [], prerequisites: [], dependents: [] }] };
}

test("a delayed edit response cannot replace a switched graph", async () => {
  const { context, app } = loadApp();
  app.state.graph = graph("first");
  let finishEdit;
  context.fetch = () => new Promise((resolve) => { finishEdit = resolve; });
  const edit = app.toggleCondition("a", 0);
  context.fetch = async () => response(graph("second"));
  await app.switchGraphState("/second.json");
  finishEdit(response(graph("first", "r2")));
  await edit;
  assert.equal(app.state.graph.graph_id, "second");
});

test("a delayed response cannot overwrite a newer graph revision", async () => {
  const { context, app } = loadApp();
  app.state.graph = graph("first");
  let finishOlder;
  context.fetch = () => new Promise((resolve) => { finishOlder = resolve; });
  const older = app.toggleCondition("a", 0);
  context.fetch = async () => response(graph("first", "r3"));
  await app.toggleCondition("a", 1);
  finishOlder(response(graph("first", "r2")));
  await older;
  assert.equal(app.state.graph.revision, "r3");
});

test("renaming a condition sends no captured truth value", async () => {
  const { context, app } = loadApp();
  app.state.graph = graph("first");
  let body;
  context.fetch = async (_url, options) => {
    body = JSON.parse(options.body);
    return response(graph("first", "r2"));
  };
  await app.updateCondition("a", 0, " New label ");
  assert.deepEqual(body, { index: 0, label: "New label" });
});

test("a conflict offers a refresh instead of silently retrying the edit", () => {
  const { app, elements } = loadApp();
  app.showGraphError({ status: 409, message: "Graph changed" });
  const output = elements.get("#operation-output");
  assert.equal(output.children.at(-1).textContent, "Refresh graph");
});

test("terminal input is sent in order even when the first request is delayed", async () => {
  const { context, app } = loadApp();
  let finishFirst;
  const calls = [];
  context.fetch = (_url, options) => {
    calls.push(JSON.parse(options.body).data);
    if (calls.length === 1) return new Promise((resolve) => { finishFirst = resolve; });
    return Promise.resolve(response({ accepted_bytes: 6 }));
  };
  const first = app.sendTerminalInput("first");
  const second = app.sendTerminalInput("second");
  await settle();
  assert.deepEqual(calls, ["first"]);
  finishFirst(response({ accepted_bytes: 5 }));
  await Promise.all([first, second]);
  assert.deepEqual(calls, ["first", "second"]);
});

test("queued input cannot be redirected into a replacement terminal", async () => {
  const { context, app } = loadApp();
  let finishFirst;
  const calls = [];
  context.fetch = (url) => {
    calls.push(url);
    return new Promise((resolve) => { finishFirst = resolve; });
  };
  const first = app.sendTerminalInput("first");
  const second = app.sendTerminalInput("second");
  await settle();
  app.state.terminalSessionId = "replacement";
  finishFirst(response({ accepted_bytes: 5 }));
  await Promise.all([first, second]);
  assert.deepEqual(calls, ["/api/terminal/sessions/old/input"]);
});

test("a hidden terminal keeps a slower heartbeat", async () => {
  const { context, app, elements, intervalDelays } = loadApp();
  elements.get("#terminal-drawer").hidden = true;
  context.fetch = async () => response({ chunks: [], closed: false });
  app.startTerminalPolling();
  await settle();
  assert.deepEqual([...intervalDelays.values()], [2000]);
});

test("failed input cancels its queued suffix without retrying uncertain delivery", async () => {
  const { context, app } = loadApp();
  let failFirst;
  const calls = [];
  context.fetch = (_url, options) => {
    calls.push(JSON.parse(options.body).data);
    return new Promise((_resolve, reject) => { failFirst = reject; });
  };
  const first = app.sendTerminalInput("prefix");
  const second = app.sendTerminalInput("suffix\n");
  await settle();
  failFirst(new Error("connection lost"));
  await Promise.all([first, second]);
  assert.deepEqual(calls, ["prefix"]);
  context.fetch = async (_url, options) => {
    calls.push(JSON.parse(options.body).data);
    return response({ accepted_bytes: 1 });
  };
  await app.sendTerminalInput("\u0003");
  assert.deepEqual(calls, ["prefix", "\u0003"]);
});
