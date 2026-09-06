const STOP_WORDS = new Set(`a an and as at be build bring capture clarify close co define develop do figure findings for general get global graph home in increase into issue launch make of on online out paid pipeline portfolio profitable proof public publish purpose realize recruit resolve secure seed set setup skill slide style system the to transformation turn validate vehicle verify web website with write again account ad builder catalogue communication composition concept contract deal feedback intention junior musicians offer paper proposal quality research revenue sponsorship`.split(/\s+/));

function hash(text) {
  let value = 2166136261;
  for (const char of text) value = Math.imul(value ^ char.codePointAt(0), 16777619);
  return (value >>> 0) / 4294967296;
}

export function readableLabel(value) {
  const words = value.replaceAll('_', ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function validPoint(point) {
  return point && ['x', 'y', 'z'].every(axis => Number.isFinite(point[axis]) && Math.abs(point[axis]) < 100000);
}

function topicWords(node) {
  return [...new Set(node.id.toLowerCase().split(/[^\p{L}\p{N}]+/u)
    .filter(word => word.length > 3 && !STOP_WORDS.has(word)))];
}

function topicAssignments(nodes) {
  const counts = new Map();
  const words = new Map(nodes.map(node => [node.id, topicWords(node)]));
  for (const terms of words.values()) for (const term of terms) counts.set(term, (counts.get(term) || 0) + 1);
  const result = new Map();
  for (const node of nodes) {
    const candidates = words.get(node.id).filter(word => counts.get(word) > 1);
    candidates.sort((a, b) => counts.get(b) - counts.get(a) || a.localeCompare(b));
    if (candidates.length) result.set(node.id, candidates[0]);
  }
  // Unnamed branches can use a nearby named topic without adding a graph edge.
  for (let pass = 0; pass < nodes.length; pass++) {
    const additions = [];
    for (const node of nodes) {
      if (result.has(node.id)) continue;
      const adjacent = [...(node.dependents || []), ...(node.prerequisites || [])];
      const topic = adjacent.map(id => result.get(id)).find(Boolean);
      if (topic) additions.push([node.id, topic]);
    }
    if (!additions.length) break;
    for (const [id, topic] of additions) result.set(id, topic);
  }
  return result;
}

function centerFor(topic, existing) {
  let best, bestDistance = -1;
  for (let attempt = 0; attempt < 80; attempt++) {
    const azimuth = hash(`${topic}:a:${attempt}`) * Math.PI * 2;
    const elevation = (hash(`${topic}:e:${attempt}`) - 0.5) * 1.5;
    const radius = 250 + Math.sqrt(existing.length) * 32;
    const point = {
      x: Math.cos(azimuth) * Math.cos(elevation) * radius,
      y: Math.sin(elevation) * radius,
      z: Math.sin(azimuth) * Math.cos(elevation) * radius
    };
    const distance = Math.min(...existing.map(p => Math.hypot(p.x - point.x, p.y - point.y, p.z - point.z)));
    if (distance > bestDistance) { best = point; bestDistance = distance; }
    if (distance > 260) break;
  }
  return best;
}

export function buildCloudLayout(nodes, saved = {}) {
  const ordered = [...nodes].sort((a, b) => a.id.localeCompare(b.id));
  const inferred = topicAssignments(ordered);
  const points = new Map();
  const topics = new Map();
  for (const node of ordered) {
    const point = saved.nodes?.[node.id];
    if (validPoint(point) && typeof point.topic === 'string' && point.topic.length < 200) points.set(node.id, point);
  }
  const memberships = new Map(ordered.map(node => [node.id, points.get(node.id)?.topic || inferred.get(node.id) || 'other matters']));
  for (const topic of [...new Set(memberships.values())].sort()) {
    const savedCenter = saved.topics?.[topic];
    if (validPoint(savedCenter)) topics.set(topic, {x:savedCenter.x, y:savedCenter.y, z:savedCenter.z});
  }
  for (const topic of [...new Set(memberships.values())].sort()) {
    if (!topics.has(topic)) topics.set(topic, centerFor(topic, [...topics.values()]));
  }
  for (const node of ordered) {
    if (points.has(node.id)) continue;
    const topic = memberships.get(node.id), center = topics.get(topic);
    const angle = hash(`${node.id}:angle`) * Math.PI * 2;
    const radius = 42 + hash(`${node.id}:radius`) * 65;
    let point = {
      topic,
      x: center.x + Math.cos(angle) * radius,
      y: center.y + (hash(`${node.id}:height`) - 0.5) * 90 + Math.min(node.overview?.depth || 0, 5) * 12,
      z: center.z + Math.sin(angle) * radius
    };
    for (let attempt = 0; attempt < 30; attempt++) {
      if ([...points.values()].every(p => Math.hypot(p.x-point.x,p.y-point.y,p.z-point.z) > 28)) break;
      point = {...point, x:point.x+Math.cos(angle+attempt)*24, y:point.y+16, z:point.z+Math.sin(angle+attempt)*24};
    }
    points.set(node.id, point);
  }
  return {nodes:Object.fromEntries(points), topics:Object.fromEntries(topics)};
}
