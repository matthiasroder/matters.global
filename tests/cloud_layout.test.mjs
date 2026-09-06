import assert from 'node:assert/strict';
import test from 'node:test';
import { buildCloudLayout } from '../src/matters/web_assets/cloud-layout.js';

const nodes = [
  {id:'build_mygain', prerequisites:['setup_mygain'], dependents:[], overview:{depth:1}},
  {id:'setup_mygain', prerequisites:[], dependents:['build_mygain'], overview:{depth:0}},
  {id:'publish_mygain_ad', prerequisites:[], dependents:[], overview:{depth:0}},
  {id:'build_mozart', prerequisites:['launch_mozart'], dependents:[], overview:{depth:1}},
  {id:'launch_mozart', prerequisites:[], dependents:['build_mozart'], overview:{depth:0}},
  {id:'resolve_insurance', prerequisites:[], dependents:[], overview:{depth:0}}
];

test('topic neighborhoods include isolated related matters without adding dependencies', () => {
  const before=structuredClone(nodes), layout=buildCloudLayout(nodes);
  assert.equal(layout.nodes.publish_mygain_ad.topic,layout.nodes.build_mygain.topic);
  assert.notEqual(layout.nodes.build_mygain.topic,layout.nodes.build_mozart.topic);
  assert.deepEqual(nodes,before);
});

test('ordering and truth changes cannot shuffle the cloud', () => {
  const initial=buildCloudLayout(nodes);
  const updated=buildCloudLayout([...nodes].reverse().map(n=>({...n,resolved:true,actionable:false})));
  assert.deepEqual(updated,initial);
});

test('saved positions and topics survive new matters and dependency changes', () => {
  const initial=buildCloudLayout(nodes);
  initial.nodes.build_mygain.x+=20;
  const changed=[...nodes.map(n=>({...n,prerequisites:[],dependents:[]})),{id:'setup_mygain_community',prerequisites:[],dependents:[]}];
  const updated=buildCloudLayout(changed,JSON.parse(JSON.stringify(initial)));
  for (const node of nodes) assert.deepEqual(updated.nodes[node.id],initial.nodes[node.id]);
  for (const [topic,point] of Object.entries(initial.topics)) assert.deepEqual(updated.topics[topic],point);
  assert.equal(updated.nodes.setup_mygain_community.topic,'mygain');
});

test('malformed saved coordinates do not poison the scene', () => {
  const saved={nodes:{build_mygain:{x:'bad',y:0,z:0,topic:'mygain'}},topics:{mygain:{x:Infinity,y:0,z:0}}};
  const result=buildCloudLayout(nodes,saved);
  for (const node of Object.values(result.nodes)) for(const axis of ['x','y','z']) assert.ok(Number.isFinite(node[axis]));
  assert.deepEqual(buildCloudLayout([]),{nodes:{},topics:{}});
});
