const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "../src/matters/web_assets/app.js"), "utf8"
).replace(/^import .*;\n/gm, "");

function response(payload) {
  return {
    ok: true,
    headers: { get: () => "application/json" },
    json: async () => payload
  };
}

function loadApp() {
  const intervals = new Map();
  const elements = new Map();
  let nextTimer = 1;
  const context = vm.createContext({
    URLSearchParams,
    cytoscape: Object.assign(() => ({ on() {} }), { use() {} }),
    dagre: {},
    createOverviewRenderer: () => ({}),
    document: {
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, { addEventListener() {} });
        return elements.get(selector);
      }
    },
    window: {
      location: { search: "?token=test-token", pathname: "/", hash: "" },
      history: { replaceState() {} },
      matchMedia: () => ({ matches: false }),
      addEventListener() {},
      setInterval(callback) {
        const id = nextTimer++;
        intervals.set(id, callback);
        return id;
      },
      clearInterval(id) { intervals.delete(id); }
    },
    // Startup's graph load stays pending; these tests do not render a graph.
    fetch: () => new Promise(() => {})
  });
  vm.runInContext(source + `
    globalThis.app = {state, api, pollTerminal, restartTerminal, startTerminalPolling};
  `, context);
  const output = [];
  context.app.state.terminal = { rows: 24, cols: 100, clear() {}, write(text) { output.push(text); } };
  context.app.state.terminalSessionId = "old";
  return { context, app: context.app, intervals, output, elements };
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
  app.state.graph = { graph_id: "first" };
  let request, finish;
  context.fetch = (_url, options) => {
    request = options;
    return new Promise((resolve) => { finish = resolve; });
  };
  const pending = app.api("/api/matters/b/conditions", { method: "PATCH" });
  app.state.graph = { graph_id: "second" };
  assert.equal(request.headers["X-Matters-Graph"], "first");
  assert.equal(request.headers.Authorization, "Bearer test-token");
  finish(response({ graph_id: "first" }));
  await pending;
});
