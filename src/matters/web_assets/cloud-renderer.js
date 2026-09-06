import ForceGraph3D from 'https://cdn.jsdelivr.net/npm/3d-force-graph@1.80.0/+esm';
import { buildCloudLayout, readableLabel } from './cloud-layout.js?v=cloud-v1';

const COLORS = {actionable:'#76c8a3', blocked:'#efad86', resolved:'#869ba7', selected:'#fff0d4', edge:'#72888f'};

export function createCloudRenderer({container, onSelect, onError, reducedMotion=false}) {
  const stage = document.createElement('div');
  stage.className = 'cloud-stage';
  const labels = document.createElement('div');
  labels.className = 'cloud-labels';
  labels.setAttribute('aria-label', 'Matter neighborhoods');
  const hud = document.createElement('div');
  hud.className = 'cloud-hud';
  const caption = document.createElement('div');
  caption.className = 'cloud-caption';
  const title = document.createElement('strong');
  title.textContent = 'Your matters, in space';
  const counts = document.createElement('span');
  caption.append(title, counts);
  const legend = document.createElement('div');
  legend.className = 'cloud-legend';
  for (const status of ['actionable','blocked','resolved']) {
    const item = document.createElement('span');
    item.className = status;
    item.textContent = readableLabel(status);
    legend.append(item);
  }
  hud.append(caption,legend);
  const navigation = document.createElement('div');
  navigation.className = 'cloud-navigation';
  navigation.textContent = 'Drag to orbit · Scroll to zoom · Right-drag to pan';
  const fit = document.createElement('button');
  fit.type = 'button';
  fit.textContent = 'Whole cloud';
  navigation.append(fit);
  container.append(stage,labels,hud,navigation);
  container.classList.add('cloud-view');

  let graph;
  try {
    graph = new ForceGraph3D(stage, {controlType:'orbit', rendererConfig:{antialias:true, alpha:true}})
      .backgroundColor('#152126')
      .showNavInfo(false)
      .nodeRelSize(5)
      .nodeVal(1)
      .nodeResolution(16)
      .nodeOpacity(1)
      .enableNodeDrag(false)
      .cooldownTicks(0)
      .linkDirectionalArrowLength(3)
      .linkDirectionalArrowRelPos(0.83)
      .linkOpacity(0.4);
  } catch (error) {
    container.replaceChildren();
    container.classList.remove('cloud-view');
    throw error;
  }
  const canvas = stage.querySelector('canvas');
  canvas.setAttribute('aria-label', 'Interactive 3D cloud. Drag to orbit, scroll to zoom. Use search to select any matter.');
  canvas.addEventListener('webglcontextlost', contextLost);
  const controls = graph.controls();
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.minDistance = 45;
  controls.maxDistance = 8000;
  graph.renderer().setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  let graphNodes = [], nodeById = new Map();
  let layout = {nodes:{},topics:{}};
  let storageKey = '', signature = '', selection = null, hovered = null;
  let active = false, destroyed = false, frame = 0, width = 1, height = 1;
  let nodeLabels = new Map(), topicLabels = new Map();
  let emphasized = new Set(), filtering = false, matches = new Set();

  function status(node) { return node.resolved ? 'resolved' : node.actionable ? 'actionable' : 'blocked'; }
  function linkId(endpoint) { return typeof endpoint === 'object' ? endpoint.id : endpoint; }
  function nodeEmphasized(node) {
    if (emphasized.size) return emphasized.has(node.id);
    return !filtering || matches.has(node.id);
  }
  function color(node) {
    if (node.id === selection) return COLORS.selected;
    return nodeEmphasized(node) ? COLORS[status(node)] : '#35484e';
  }
  function focusedLink(link) {
    return emphasized.has(linkId(link.source)) && emphasized.has(linkId(link.target));
  }
  graph.nodeColor(color)
    .linkColor(link => focusedLink(link) ? '#eed9b2' : '#4b626c')
    .linkWidth(link => focusedLink(link) ? 0.9 : 0.25)
    .linkDirectionalArrowColor(link => focusedLink(link) ? '#eed9b2' : '#72888f')
    .nodeLabel(node => {
      const element = document.createElement('div');
      element.className = 'cloud-tooltip';
      element.textContent = `${readableLabel(node.label || node.id)} · ${readableLabel(status(node))}`;
      return element;
    })
    .onNodeClick(node => onSelect(node.id))
    .onBackgroundClick(() => onSelect(null))
    .onNodeHover(node => { hovered = node?.id || null; });

  function readSaved() {
    try { return JSON.parse(localStorage.getItem(storageKey) || '{}') || {}; }
    catch { return {}; }
  }
  function save() {
    if (!storageKey || !graphNodes.length || destroyed) return;
    try { localStorage.setItem(storageKey, JSON.stringify({...layout,camera:getCamera()})); }
    catch { /* Storage may be disabled. The current scene remains usable. */ }
  }
  controls.addEventListener('end', save);

  function setModel(next) {
    const nextKey = `matters-cloud:v1:${next.graphId}`;
    const graphChanged = storageKey !== nextKey;
    if (graphChanged) { save(); storageKey = nextKey; signature = ''; selection = null; }
    const nextSignature = JSON.stringify([next.nodes.map(n=>n.id).sort(),next.edges.map(e=>[e.source,e.target]).sort()]);
    const structuralChange = signature !== nextSignature;
    emphasized = new Set(next.derivedHighlightIds || []);
    if (!emphasized.size && next.selectedId) emphasized = new Set([next.selectedId,...(next.ancestorIds || []),...(next.directDependentIds || [])]);
    filtering = next.filterActive;
    matches = new Set(next.matchIds || []);
    if (structuralChange) {
      const saved = graphChanged ? readSaved() : layout;
      layout = buildCloudLayout(next.nodes, saved);
      graphNodes = next.nodes.map(node => {
        const point = layout.nodes[node.id];
        return {...node,...point,fx:point.x,fy:point.y,fz:point.z};
      });
      nodeById = new Map(graphNodes.map(node=>[node.id,node]));
      const graphLinks = next.edges.map(edge=>({...edge}));
      graph.graphData({nodes:graphNodes,links:graphLinks});
      signature = nextSignature;
      buildLabels();
      if (graphChanged) {
        resize();
        if (validCamera(saved.camera)) setCamera(saved.camera);
        else resetCamera(false);
      }
    } else {
      for (const node of next.nodes) Object.assign(nodeById.get(node.id), node);
    }
    const previous = selection;
    selection = next.selectedId || null;
    graph.nodeColor(color).linkColor(link=>focusedLink(link)?'#eed9b2':'#4b626c')
      .linkWidth(link=>focusedLink(link)?0.9:0.25)
      .linkDirectionalArrowColor(link=>focusedLink(link)?'#eed9b2':'#72888f');
    counts.textContent = `${next.nodes.length} matters · ${Object.keys(layout.topics).length} neighborhoods`;
    if (selection && selection !== previous && !graphChanged) {
      const related=graphNodes.filter(node=>emphasized.has(node.id));
      framePoints(related.length ? related : [nodeById.get(selection)]);
    }
    save();
  }

  function buildLabels() {
    labels.replaceChildren();
    nodeLabels = new Map(); topicLabels = new Map();
    for (const node of graphNodes) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'cloud-node-label';
      button.textContent = readableLabel(node.label || node.id);
      button.dataset.matterId = node.id;
      button.addEventListener('click',()=>onSelect(node.id));
      labels.append(button); nodeLabels.set(node.id,button);
    }
    for (const [topic,point] of Object.entries(layout.topics)) {
      const button=document.createElement('button');
      button.type='button'; button.className='cloud-topic-label';
      button.textContent=readableLabel(topic);
      button.dataset.topic=topic;
      button.addEventListener('click',()=>flyTo(point,350));
      labels.append(button); topicLabels.set(topic,button);
    }
  }

  function validCamera(camera) {
    return camera && ['position','target'].every(part=>camera[part] && ['x','y','z'].every(axis=>Number.isFinite(camera[part][axis]) && Math.abs(camera[part][axis])<100000));
  }
  function getCamera() {
    const position=graph.camera().position,target=controls.target;
    return {position:{x:position.x,y:position.y,z:position.z},target:{x:target.x,y:target.y,z:target.z}};
  }
  function setCamera(camera) {
    if (!validCamera(camera)) return;
    graph.cameraPosition(camera.position,camera.target,0);
    controls.update();
  }
  function flyTo(point,distance=450) {
    if (!point) return;
    const direction=graph.camera().position.clone().sub(controls.target).normalize();
    graph.cameraPosition({x:point.x+direction.x*distance,y:point.y+direction.y*distance,z:point.z+direction.z*distance},point,reducedMotion?0:650);
  }
  function resetCamera(animate=true) {
    if (!graphNodes.length) return;
    framePoints(graphNodes,animate);
  }
  function framePoints(points,animate=true) {
    const center={x:0,y:0,z:0};
    for(const axis of ['x','y','z'])center[axis]=(Math.min(...points.map(n=>n[axis]))+Math.max(...points.map(n=>n[axis])))/2;
    const radius=Math.max(100,...points.map(n=>Math.hypot(n.x-center.x,n.y-center.y,n.z-center.z)));
    const distance=radius/Math.sin(graph.camera().fov*Math.PI/360)*Math.max(1,height/width)*1.18;
    const direction=graph.camera().position.clone().sub(controls.target).normalize();
    graph.cameraPosition({x:center.x+distance*direction.x,y:center.y+distance*direction.y,z:center.z+distance*direction.z},center,animate&&!reducedMotion?650:0);
  }
  function zoom(factor) {
    const camera=getCamera(),target=camera.target;
    const position=Object.fromEntries(['x','y','z'].map(axis=>[axis,target[axis]+(camera.position[axis]-target[axis])/factor]));
    graph.cameraPosition(position,target,reducedMotion?0:180);
  }
  fit.addEventListener('click',()=>resetCamera());

  function project(point) {
    const vector=graph.camera().position.clone().set(point.x,point.y,point.z).project(graph.camera());
    return {x:(vector.x+1)*width/2,y:(1-vector.y)*height/2,visible:vector.z>-1&&vector.z<1};
  }
  function displayLabel(element,point,occupied,priority=false) {
    const p=project(point), w=Math.min(element.offsetWidth||170,width-24),h=element.offsetHeight||36;
    const x=Math.max(12,Math.min(width-w-12,p.x+9)),y=p.y-h/2;
    const bounds={x,y,w,h};
    const outside=!p.visible||p.x<0||p.x>width||y<90||y+h>height-65;
    const collides=occupied.some(b=>x<b.x+b.w+5&&x+w+5>b.x&&y<b.y+b.h+4&&y+h+4>b.y);
    element.hidden=outside||(!priority&&collides);
    if(!element.hidden){element.style.transform=`translate(${x}px,${y}px)`;occupied.push(bounds);}
  }
  function renderLabels() {
    if(destroyed||!active)return;
    const occupied=[];
    const camera=graph.camera().position;
    const selected=nodeById.get(selection);
    const ranked=[...graphNodes].sort((a,b)=>{
      const score=n=>n.id===selection?10000:n.id===hovered?9000:emphasized.has(n.id)?5000:0;
      return score(b)-score(a)||camera.distanceTo(a)-camera.distanceTo(b);
    });
    for(const node of ranked){
      const element=nodeLabels.get(node.id);
      const important=node.id===selection||node.id===hovered;
      const nearby=camera.distanceTo(node)<750;
      const show=important||(nodeEmphasized(node)&&(nearby||emphasized.has(node.id)||filtering));
      element.hidden=!show;
      element.classList.toggle('selected',node.id===selection);
      element.dataset.status=status(node);
      element.setAttribute('aria-label',`${readableLabel(node.label||node.id)}, ${status(node)}`);
      if(show)displayLabel(element,node,occupied,important);
    }
    for(const [topic,element] of topicLabels){
      const point=layout.topics[topic];
      element.hidden=Boolean(selected&&selected.topic!==topic)||filtering;
      if(!element.hidden)displayLabel(element,{...point,y:point.y+115},occupied);
    }
    frame=requestAnimationFrame(renderLabels);
  }
  function resize() {
    const rect=container.getBoundingClientRect();
    if(!rect.width||!rect.height)return;
    width=rect.width;height=rect.height;
    graph.width(width).height(height);
  }
  const observer=new ResizeObserver(resize);
  observer.observe(container);
  function setVisible(visible) {
    if(active===visible)return;
    active=visible;
    cancelAnimationFrame(frame);
    if(visible){resize();graph.resumeAnimation();frame=requestAnimationFrame(renderLabels);}
    else {save();graph.pauseAnimation();}
  }
  function contextLost(event) { event.preventDefault();onError(new Error('3D rendering was interrupted. The 2D view is still available.')); }
  function destroy() {
    save();destroyed=true;cancelAnimationFrame(frame);observer.disconnect();
    controls.removeEventListener('end',save);
    canvas.removeEventListener('webglcontextlost',contextLost);
    window.removeEventListener('pagehide',save);
    graph._destructor();
    container.replaceChildren();container.classList.remove('cloud-view');
  }
  window.addEventListener('pagehide',save);
  graph.pauseAnimation();
  return {setModel,setVisible,resize,zoom,resetCamera,getCamera,setCamera,destroy,cloud:true};
}
