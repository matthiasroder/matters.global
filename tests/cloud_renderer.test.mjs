import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import vm from 'node:vm';
import { buildCloudLayout, readableLabel } from '../src/matters/web_assets/cloud-layout.js';

const rendererSource = fs.readFileSync(new URL('../src/matters/web_assets/cloud-renderer.js', import.meta.url), 'utf8');

export function rendererHarness(source = rendererSource) {
  const metrics = {attributes: 0, measurements: 0, hidden: 0, storage: 0, frames: 0};
  class Events {
    listeners = new Map();
    addEventListener(name, listener) {
      if (!this.listeners.has(name)) this.listeners.set(name, new Set());
      this.listeners.get(name).add(listener);
    }
    removeEventListener(name, listener) { this.listeners.get(name)?.delete(listener); }
    emit(name) { for (const listener of this.listeners.get(name) || []) listener(); }
  }
  class Element extends Events {
    children = []; className = ''; textContent = ''; style = {}; dataset = {};
    attributes = new Map(); _hidden = false;
    classList = {
      contains: name => this.className.split(' ').includes(name),
      add: name => { if (!this.classList.contains(name)) this.className += ` ${name}`; },
      remove: name => { this.className = this.className.split(' ').filter(value => value !== name).join(' '); },
      toggle: (name, enabled) => enabled ? this.classList.add(name) : this.classList.remove(name)
    };
    setAttribute(name, value) { metrics.attributes++; this.attributes.set(name, value); }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    get hidden() { return this._hidden; }
    set hidden(value) { metrics.hidden++; this._hidden = value; }
    get offsetWidth() { metrics.measurements++; return this.hidden ? 0 : 180; }
    get offsetHeight() { metrics.measurements++; return this.hidden ? 0 : 36; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    querySelector() { return new Element(); }
    getBoundingClientRect() { return {width: 1000, height: 700}; }
  }
  class Vector {
    constructor(x = 0, y = 0, z = 500) { Object.assign(this, {x, y, z}); }
    clone() { return new Vector(this.x, this.y, this.z); }
    sub(point) { this.x -= point.x; this.y -= point.y; this.z -= point.z; return this; }
    normalize() { const d = Math.hypot(this.x, this.y, this.z) || 1; this.x /= d; this.y /= d; this.z /= d; return this; }
    set(x, y, z) { Object.assign(this, {x, y, z}); return this; }
    project() { this.x /= 1000; this.y /= 1000; this.z = 0; return this; }
    distanceTo(point) { return Math.hypot(this.x - point.x, this.y - point.y, this.z - point.z); }
  }
  const controls = new Events();
  controls.target = new Vector(0, 0, 0);
  controls.update = () => controls.emit('change');
  const camera = {position: new Vector(), fov: 50};
  let graph;
  function ForceGraph3D() {
    graph = new Proxy({
      controls: () => controls,
      renderer: () => ({setPixelRatio() {}}),
      camera: () => camera,
      cameraPosition(position, target) { Object.assign(camera.position, position); Object.assign(controls.target, target); return graph; }
    }, {
      get(target, key) {
        if (key in target) return target[key];
        return (...args) => { target[`_${key}`] = args[0]; return graph; };
      }
    });
    return graph;
  }
  const frames = new Map(); let nextFrame = 0, now = 0;
  const window = new Events(), fonts = new Events();
  const context = vm.createContext({
    ForceGraph3D, buildCloudLayout, readableLabel,
    document: {createElement: () => new Element(), fonts}, window,
    ResizeObserver: class { constructor(listener) { this.listener = listener; } observe() {} disconnect() {} },
    localStorage: {getItem: () => null, setItem() { metrics.storage++; }},
    performance: {now: () => now},
    requestAnimationFrame(callback) { const id = ++nextFrame; frames.set(id, callback); return id; },
    cancelAnimationFrame(id) { frames.delete(id); }
  });
  vm.runInContext(source.replace(/^import .*;\n/gm, '').replace('export function createCloudRenderer', 'function createCloudRenderer'), context);
  const container = new Element();
  const renderer = context.createCloudRenderer({container, onSelect() {}, onError(error) { throw error; }});
  function advance(milliseconds = 16) {
    now += milliseconds;
    const callbacks = [...frames.values()]; frames.clear();
    for (const callback of callbacks) { metrics.frames++; callback(now); }
  }
  return {renderer, container, metrics, controls, camera, fonts, window, graph, advance, pending: () => frames.size};
}

const nodes = [
  {id: 'first_matter', resolved: false, actionable: true, prerequisites: [], dependents: ['second_matter']},
  {id: 'second_matter', resolved: false, actionable: false, prerequisites: ['first_matter'], dependents: []}
];
function model(patch = {}) {
  return {graphId: 'test', nodes, edges, selectedId: null, filterActive: false, ...patch};
}
const edges = [{source: 'first_matter', target: 'second_matter'}];
function settle(harness) { harness.advance(1000); harness.advance(1000); }

test('idle clouds stop scheduling label work after placement', () => {
  const h = rendererHarness();
  h.renderer.setModel(model()); h.renderer.setVisible(true); settle(h);
  const before = {...h.metrics};
  for (let i = 0; i < 60; i++) h.advance();
  assert.deepEqual(h.metrics, before);
  assert.equal(h.pending(), 0);
  h.renderer.destroy();
});

test('camera and hover changes reposition labels without measuring or rewriting metadata', () => {
  const h = rendererHarness();
  h.renderer.setModel(model()); h.renderer.setVisible(true); settle(h);
  const before = {...h.metrics};
  h.camera.position.x += 25; h.controls.emit('change'); h.advance();
  h.graph._onNodeHover(nodes[0]); h.advance();
  assert.equal(h.metrics.frames, before.frames + 2);
  assert.equal(h.metrics.measurements, before.measurements);
  assert.equal(h.metrics.attributes, before.attributes);
  assert.equal(h.pending(), 0);
  h.renderer.destroy();
});

test('filters reuse payload topology and do not persist layout or touch label metadata', () => {
  const h = rendererHarness();
  h.renderer.setModel(model()); h.renderer.setVisible(true); settle(h);
  const before = {...h.metrics};
  // Payload arrays are reused by app.js for filter and selection changes.
  // A map trap proves this path never recalculates the topology signature.
  nodes.map = () => { throw new Error('unnecessary topology scan'); };
  try { h.renderer.setModel(model({filterActive: true, matchIds: ['first_matter']})); }
  finally { delete nodes.map; }
  h.advance();
  assert.equal(h.metrics.storage, before.storage);
  assert.equal(h.metrics.attributes, before.attributes);
  assert.equal(h.metrics.measurements, before.measurements);
  h.renderer.destroy();
});

test('camera animation finishes, refreshed text is measured, and fonts and resize invalidate measurements', () => {
  const h = rendererHarness();
  h.renderer.setModel(model()); h.renderer.setVisible(true); settle(h);
  const stored = h.metrics.storage;
  h.renderer.zoom(1.1); h.advance(30);
  assert.ok(h.pending() > 0);
  h.advance(200);
  assert.equal(h.pending(), 0);
  assert.equal(h.metrics.storage, stored + 1);
  const measured = h.metrics.measurements;
  h.renderer.setModel(model({nodes: nodes.map(node => ({...node, label: `changed ${node.id}`}))}));
  h.advance();
  assert.ok(h.metrics.measurements > measured);
  const beforeFonts = h.metrics.measurements;
  h.fonts.emit('loadingdone'); h.advance();
  assert.ok(h.metrics.measurements > beforeFonts);
  const beforeResize = h.metrics.measurements;
  h.window.emit('resize'); h.advance();
  assert.ok(h.metrics.measurements > beforeResize);
  h.renderer.setVisible(false);
  const hiddenFrames = h.metrics.frames;
  h.controls.emit('change'); h.advance();
  assert.equal(h.metrics.frames, hiddenFrames);
  h.renderer.setVisible(true); h.advance();
  assert.ok(h.metrics.frames > hiddenFrames);
  h.renderer.destroy();
  h.controls.emit('change'); h.fonts.emit('loadingdone'); h.window.emit('resize');
  assert.equal(h.pending(), 0);
});
