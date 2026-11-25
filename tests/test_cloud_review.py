"""Independent regressions for account and graph changes during pending UI work.

Run the shipped browser script with controlled network promises. The scenarios
use the real sign-out/sign-in/refresh handlers, so an earlier account's response
cannot supply graph metadata to a later account's rendered workspace.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


UI = Path(__file__).parents[1] / "src" / "matters" / "cloud" / "assets" / "app.js"

HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor() { this.children = []; this.elements = []; this.events = {}; this.value = ''; }
  addEventListener(name, handler) { this.events[name] = handler; }
  setAttribute() {}
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  reset() { for (const control of this.elements) control.value = control.defaultValue || ''; }
}
const nodes = new Map();
const document = {
  getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); },
  createElement() { return new Element(); }
};
const response = (data, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => data });
const people = {
  alice: { id: 'alice', email: 'alice@example.com' },
  bob: { id: 'bob', email: 'bob@example.com' }
};
const makeGraph = (id, owner, title) => ({
  schema_version: 1, graph_id: id, owner, title, kind: 'project', role: 'owner',
  revision: 'revision-' + id, version: 1,
  state: { schema_version: 2, matters: [], conditions: {}, dependencies: [] }
});
const graphs = [
  makeGraph('alice-a', 'alice', 'Alice private project A'),
  makeGraph('alice-b', 'alice', 'Alice private project B'),
  makeGraph('bob-a', 'bob', 'Bob project')
];
let session = 'alice';
let holdCatalog = false;
let holdMutation = false;
let pendingCatalog = null;
let pendingMutation = null;
let holdNavigation = false;
let pendingNavigation = null;
let releaseCatalog = null;
let releaseMutation = null;
let releaseNavigation = null;
const fetch = async (path, options = {}) => {
  if (path === '/api/auth/me') return session ? response({ user: people[session] }) : response({}, 401);
  if (path === '/api/auth/logout') { session = null; return response({ logged_out: true }); }
  if (path === '/api/auth/login') {
    session = JSON.parse(options.body).email.startsWith('bob') ? 'bob' : 'alice';
    return response({ user: people[session] });
  }
  if (path === '/api/graphs') {
    if (!session) return response({}, 401);
    const result = response({ graphs: graphs.filter(graph => graph.owner === session) });
    if (holdCatalog && session === 'alice') {
      holdCatalog = false;
      pendingCatalog = result;
      return new Promise(resolve => { releaseCatalog = resolve; });
    }
    return result;
  }
  const match = path.match(/^\/api\/graphs\/([^/]+)(.*)$/);
  const graph = match && graphs.find(candidate => candidate.graph_id === match[1] && candidate.owner === session);
  if (!graph) return response({ error: { message: 'Graph not found.' } }, 404);
  if (match[2] === '/views/unlock') return response({ revision: graph.revision, universe: [], blocked: [] });
  if (match[2] === '' && graph.graph_id === 'alice-b' && holdNavigation) {
    holdNavigation = false;
    pendingNavigation = response({ ...graph });
    return new Promise(resolve => { releaseNavigation = resolve; });
  }
  if (match[2] === '/operations' && holdMutation) {
    holdMutation = false;
    pendingMutation = response({ ...graph });
    return new Promise(resolve => { releaseMutation = resolve; });
  }
  return response({ ...graph });
};
const context = vm.createContext({ document, fetch, console, FormData: class {} });
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const drafts = { graph: new Element(), matter: new Element(), condition: new Element() };
document.getElementById('new-graph-form').elements.push(drafts.graph);
document.getElementById('new-matter-form').elements.push(drafts.matter, drafts.condition);
const evaluate = code => vm.runInContext(code, context);
const flush = async () => { for (let i = 0; i < 12; i += 1) await new Promise(resolve => setImmediate(resolve)); };
async function loginBob() {
  await nodes.get('sign-out').events.click();
  document.getElementById('email').value = 'bob@example.com';
  document.getElementById('password').value = 'Example-only-password-42!';
  await nodes.get('auth-form').events.submit({ preventDefault() {} });
}
(async () => {
  await flush();
  if (evaluate('state.user.id') !== 'alice' || evaluate('state.graph.graph_id') !== 'alice-a') {
    throw new Error('Fixture failed to finish the first signed-in workspace');
  }
  const scenario = process.argv[2];
  if (scenario === 'private-form-drafts') {
    drafts.graph.value = 'Alice private project draft';
    drafts.matter.value = 'Alice private concern draft';
    drafts.condition.value = 'Alice private evidence draft';
    for (const control of Object.values(drafts)) control.disabled = true;
    await loginBob();
  } else if (scenario === 'stale-catalog' || scenario === 'stale-unauthorized') {
    holdCatalog = true;
    nodes.get('refresh-graph').events.click();
    await flush();
    if (!releaseCatalog) throw new Error('Refresh did not issue the pending catalog request');
    await loginBob();
    if (evaluate('state.user.id') !== 'bob') throw new Error('Second account did not sign in');
    releaseCatalog(scenario === 'stale-catalog' ? pendingCatalog : response({ error: { message: 'Sign in.' } }, 401));
  } else if (scenario === 'stale-mutation' || scenario === 'pending-navigation') {
    holdMutation = true;
    const pending = evaluate("mutate('set_condition', {matter_id:'example',index:0,truth:true})").catch(() => {});
    await flush();
    if (!releaseMutation) throw new Error('Mutation was not held');
    const graphB = nodes.get('graph-list').children[1];
    holdNavigation = scenario === 'pending-navigation';
    const navigating = graphB.events.click();
    if (scenario === 'stale-mutation') {
      await navigating;
      if (evaluate('state.graph.graph_id') !== 'alice-b') throw new Error('Second graph did not load');
    } else {
      await flush();
      if (!releaseNavigation) throw new Error('Second graph request was not held');
    }
    releaseMutation(pendingMutation);
    await pending;
    if (scenario === 'pending-navigation') {
      releaseNavigation(pendingNavigation);
      await navigating;
    }
  } else {
    throw new Error('Unknown scenario');
  }
  await flush();
  const observed = evaluate('({ user: state.user?.id ?? null, graph: state.graph?.graph_id ?? null, titles: state.graphs.map(graph => graph.title) })');
  if (scenario === 'private-form-drafts') {
    observed.drafts = Object.values(drafts).map(control => control.value);
    observed.enabled = Object.values(drafts).every(control => !control.disabled);
  }
  console.log(JSON.stringify(observed));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("scenario", [
    "stale-catalog", "stale-unauthorized", "stale-mutation", "pending-navigation", "private-form-drafts",
])
def test_pending_ui_work_cannot_cross_account_or_graph_selection(scenario):
    result = subprocess.run(
        ["node", "-e", HARNESS, str(UI), scenario],
        text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    if scenario in {"stale-mutation", "pending-navigation"}:
        assert observed["user"] == "alice"
        assert observed["graph"] == "alice-b"
    else:
        assert observed["user"] == "bob"
        assert observed["graph"] == "bob-a"
        assert observed["titles"] == ["Bob project"]
        if scenario == "private-form-drafts":
            assert observed["drafts"] == ["", "", ""]
            assert observed["enabled"] is True
