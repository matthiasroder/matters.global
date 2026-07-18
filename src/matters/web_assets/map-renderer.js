const PALETTE = {
  paper: "#f7f1e4",
  paperLight: "#fffdf6",
  ink: "#28302f",
  mutedInk: "#68736e",
  line: "#9b9587",
  actionable: "#2f7f5a",
  blocked: "#b85f49",
  resolved: "#687b82",
  gold: "#c2933e",
  ivory: "#fff9e8",
  derived: "#54737c"
};

const CAMERA_LIMITS = {
  minPitch: -1.25,
  maxPitch: 1.25,
  minZoom: 0.18,
  maxZoom: 6
};

const DEFAULT_CAMERA = Object.freeze({
  yaw: -0.62,
  pitch: -0.62,
  zoom: 1,
  panX: 0,
  panY: 0
});

const EMPTY_MODEL = Object.freeze({
  nodes: [],
  edges: [],
  selectedId: null,
  ancestorIds: [],
  directDependentIds: [],
  matchIds: [],
  filterActive: false,
  derivedHighlightIds: []
});

const EDGE_STYLES = Object.freeze({
  faded: { color: PALETTE.line, alpha: 0.035, width: 0.75, arrows: false, arrowSize: 0 },
  normal: { color: PALETTE.line, alpha: 0.14, width: 0.75, arrows: true, arrowSize: 3.2 },
  focused: { color: PALETTE.gold, alpha: 0.76, width: 1.45, arrows: true, arrowSize: 4.5 },
  dependent: { color: PALETTE.actionable, alpha: 0.76, width: 1.45, arrows: true, arrowSize: 4.5 },
  derived: { color: PALETTE.derived, alpha: 0.76, width: 1.45, arrows: true, arrowSize: 4.5 }
});

const MOTION_SETTLE_MS = 110;
const OVERLAP_GRID_SIZE = 64;
const NODE_DEPTH_BANDS = 8;

/**
 * A small, dependency-free canvas renderer for the stable overview coordinates.
 * Graph traversal and application state deliberately remain outside this module.
 */
export function createOverviewRenderer({
  container,
  onSelect = () => {},
  onError = () => {},
  reducedMotion = false
} = {}) {
  if (!(container instanceof HTMLElement)) {
    const error = new TypeError("Overview renderer requires a container element");
    notifyInitializationError(onError, error);
    throw error;
  }

  const canvas = document.createElement("canvas");
  canvas.className = "overview-canvas";
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    "Matters overview. Drag to rotate, Shift-drag to pan, and use the mouse wheel to zoom."
  );
  canvas.tabIndex = 0;
  canvas.style.display = "block";
  canvas.style.height = "100%";
  canvas.style.touchAction = "none";
  canvas.style.width = "100%";
  container.append(canvas);

  const context = canvas.getContext("2d", { alpha: false });
  if (!context) {
    canvas.remove();
    const error = new Error("This browser does not support the canvas overview");
    notifyInitializationError(onError, error);
    throw error;
  }
  const backgroundCanvas = document.createElement("canvas");
  const backgroundContext = backgroundCanvas.getContext("2d", { alpha: false });

  let width = 1;
  let height = 1;
  let dpr = 1;
  let destroyed = false;
  let frameRequest = 0;
  let model = EMPTY_MODEL;
  let nodes = [];
  let edges = [];
  let nodeLookup = new Map();
  let projectedNodes = [];
  let emphasis = emphasisState(EMPTY_MODEL);
  let nodeStyles = new Map();
  let hoverNode = null;
  let camera = { ...DEFAULT_CAMERA };
  let world = emptyWorld();
  let movingUntil = 0;
  let settleTimer = 0;
  const pointers = new Map();
  let gesture = null;

  const resizeObserver = typeof ResizeObserver === "undefined"
    ? null
    : new ResizeObserver(() => resize());
  resizeObserver?.observe(container);

  function report(error) {
    try {
      onError(error instanceof Error ? error : new Error(String(error)));
    } catch {
      // An application error reporter must not break rendering.
    }
  }

  function guarded(action) {
    return (...args) => {
      if (destroyed) return undefined;
      try {
        return action(...args);
      } catch (error) {
        report(error);
        return undefined;
      }
    };
  }

  function setModel(nextModel = EMPTY_MODEL) {
    model = nextModel && typeof nextModel === "object" ? nextModel : EMPTY_MODEL;
    const rawNodes = Array.isArray(model.nodes) ? model.nodes : [];
    const rawEdges = Array.isArray(model.edges) ? model.edges : [];

    nodes = rawNodes.map((node, index) => normalizeNode(node, index, rawNodes.length));
    nodeLookup = new Map(nodes.map((node) => [node.id, node]));
    projectedNodes = nodes.map((node) => {
      const projected = { id: node.id, node, x: 0, y: 0, depth: 0, radius: 3 };
      node.projected = projected;
      return projected;
    });
    edges = rawEdges
      .map((edge) => ({
        source: nodeLookup.get(String(edge?.source ?? "")),
        target: nodeLookup.get(String(edge?.target ?? ""))
      }))
      .filter((edge) => edge.source && edge.target);

    emphasis = emphasisState(model);
    cacheVisualStyles();
    world = worldBounds(nodes);
    hoverNode = hoverNode ? nodeLookup.get(hoverNode.id) || null : null;
    scheduleDraw();
  }

  function resize() {
    const rect = container.getBoundingClientRect();
    const nextWidth = Math.max(1, Math.round(rect.width));
    const nextHeight = Math.max(1, Math.round(rect.height));
    const nextDpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    if (nextWidth === width && nextHeight === height && nextDpr === dpr) return;

    width = nextWidth;
    height = nextHeight;
    dpr = nextDpr;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    rebuildBackground();
    scheduleDraw();
  }

  function zoom(factor) {
    const safeFactor = finiteNumber(factor, 1);
    camera.zoom = clamp(camera.zoom * safeFactor, CAMERA_LIMITS.minZoom, CAMERA_LIMITS.maxZoom);
    markCameraMotion();
    scheduleDraw();
  }

  function resetCamera() {
    camera = { ...DEFAULT_CAMERA };
    scheduleDraw();
  }

  function getCamera() {
    return { ...camera };
  }

  function setCamera(nextCamera = {}) {
    camera = sanitizeCamera({ ...camera, ...nextCamera });
    markCameraMotion();
    scheduleDraw();
  }

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    if (frameRequest) window.cancelAnimationFrame(frameRequest);
    if (settleTimer) window.clearTimeout(settleTimer);
    resizeObserver?.disconnect();
    window.removeEventListener("resize", resize);
    canvas.removeEventListener("contextmenu", preventContextMenu);
    canvas.removeEventListener("pointerdown", pointerDown);
    canvas.removeEventListener("pointermove", pointerMove);
    canvas.removeEventListener("pointerup", pointerUp);
    canvas.removeEventListener("pointercancel", pointerUp);
    canvas.removeEventListener("wheel", wheel);
    canvas.remove();
    pointers.clear();
    projectedNodes = [];
    nodeLookup.clear();
  }

  function scheduleDraw() {
    if (destroyed || frameRequest) return;
    frameRequest = window.requestAnimationFrame(() => {
      frameRequest = 0;
      try {
        draw();
      } catch (error) {
        report(error);
      }
    });
  }

  function draw() {
    const moving = performance.now() < movingUntil;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawAtmosphere(moving);
    if (!nodes.length) {
      return;
    }

    projectNodes();
    if (!moving) {
      classifyNodeOverlaps();
      drawDepthGuides(projectedNodes);
    }
    drawEdges(moving);
    if (moving) {
      drawMovingNodes();
    } else {
      drawSettledNodes();
    }
    if (!moving) drawLabels(projectedNodes, emphasis);
    if (hoverNode && !moving) drawTooltip(hoverNode.projected);
  }

  function drawAtmosphere(moving) {
    if (moving) {
      context.fillStyle = PALETTE.paper;
      context.fillRect(0, 0, width, height);
      return;
    }
    if (backgroundContext && backgroundCanvas.width && backgroundCanvas.height) {
      context.drawImage(backgroundCanvas, 0, 0, width, height);
    } else {
      context.fillStyle = PALETTE.paper;
      context.fillRect(0, 0, width, height);
    }
  }

  function rebuildBackground() {
    if (!backgroundContext) return;
    backgroundCanvas.width = Math.round(width * dpr);
    backgroundCanvas.height = Math.round(height * dpr);
    backgroundContext.setTransform(dpr, 0, 0, dpr, 0, 0);
    const gradient = backgroundContext.createRadialGradient(
      width * 0.46,
      height * 0.34,
      Math.min(width, height) * 0.06,
      width * 0.52,
      height * 0.48,
      Math.max(width, height) * 0.82
    );
    gradient.addColorStop(0, PALETTE.paperLight);
    gradient.addColorStop(0.58, PALETTE.paper);
    gradient.addColorStop(1, "#e8dec9");
    backgroundContext.fillStyle = gradient;
    backgroundContext.fillRect(0, 0, width, height);
    backgroundContext.save();
    backgroundContext.globalAlpha = 0.08;
    backgroundContext.strokeStyle = "#796f5f";
    backgroundContext.lineWidth = 0.6;
    backgroundContext.beginPath();
    const gap = 42;
    for (let x = -height; x < width + height; x += gap) {
      backgroundContext.moveTo(x, 0);
      backgroundContext.lineTo(x + height, height);
    }
    backgroundContext.stroke();
    backgroundContext.restore();
  }

  function drawDepthGuides(projected) {
    const byTerrace = new Map();
    projected.forEach((node) => {
      if (node.node.rootCount !== 1) return;
      const layer = node.node.layer;
      const key = `${node.node.islandId}\0${layer}`;
      const current = byTerrace.get(key);
      if (!current) {
        byTerrace.set(key, {
          layer,
          minX: node.x,
          maxX: node.x,
          totalX: node.x,
          totalY: node.y,
          count: 1
        });
      } else {
        current.minX = Math.min(current.minX, node.x);
        current.maxX = Math.max(current.maxX, node.x);
        current.totalX += node.x;
        current.totalY += node.y;
        current.count += 1;
      }
    });

    context.save();
    context.setLineDash([2, 7]);
    context.lineWidth = 0.75;
    context.strokeStyle = "rgba(96, 91, 80, 0.16)";
    [...byTerrace.values()].sort((a, b) => a.layer - b.layer).forEach((terrace) => {
      const x = terrace.totalX / terrace.count;
      const y = terrace.totalY / terrace.count;
      const radiusX = Math.max(18, (terrace.maxX - terrace.minX) / 2 + 16);
      context.beginPath();
      context.ellipse(x, y, radiusX, Math.max(3, radiusX * 0.12), 0, 0, Math.PI * 2);
      context.stroke();
    });
    context.restore();
  }

  function drawEdges(moving) {
    context.save();
    context.lineCap = "round";
    Object.entries(EDGE_STYLES).forEach(([kind, style]) => drawEdgeGroup(kind, style, moving));
    context.restore();
  }

  function drawEdgeGroup(kind, style, moving) {
    let count = 0;
    context.globalAlpha = style.alpha;
    context.strokeStyle = style.color;
    context.fillStyle = style.color;
    context.lineWidth = style.width;
    context.beginPath();
    if (moving) {
      edges.forEach((edge) => {
        if (edge.renderKind !== kind) return;
        context.moveTo(edge.source.projected.x, edge.source.projected.y);
        context.lineTo(edge.target.projected.x, edge.target.projected.y);
        count += 1;
      });
      if (count) context.stroke();
      return;
    }
    edges.forEach((edge) => {
      if (edge.renderKind !== kind) return;
      const source = edge.source.projected;
      const target = edge.target.projected;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const length = Math.max(1, Math.hypot(dx, dy));
      const unitX = dx / length;
      const unitY = dy / length;
      const endX = target.x - unitX * (target.radius + 1.5);
      const endY = target.y - unitY * (target.radius + 1.5);
      context.moveTo(source.x, source.y);
      context.lineTo(endX, endY);
      if (style.arrows && length > 18) {
        const size = style.arrowSize;
        const backX = endX - unitX * size;
        const backY = endY - unitY * size;
        const sideX = -unitY * size * 0.58;
        const sideY = unitX * size * 0.58;
        context.moveTo(endX, endY);
        context.lineTo(backX + sideX, backY + sideY);
        context.lineTo(backX - sideX, backY - sideY);
        context.closePath();
      }
      count += 1;
    });
    if (!count) return;
    context.stroke();
    if (style.arrows) context.fill();
  }

  function drawMovingNodes() {
    const bandSize = Math.max(1, Math.ceil(projectedNodes.length / NODE_DEPTH_BANDS));
    for (let start = 0; start < projectedNodes.length; start += bandSize) {
      const end = Math.min(projectedNodes.length, start + bandSize);
      nodeStyles.forEach((style, key) => {
        if (!style.selected) drawMovingStyleBand(key, style, start, end);
      });
    }
    const selected = emphasis.selectedId ? nodeLookup.get(emphasis.selectedId)?.projected : null;
    if (selected) drawNodeDirect(selected);
    context.shadowBlur = 0;
    context.globalAlpha = 1;
  }

  function drawMovingStyleBand(key, style, start, end) {
    let count = 0;
    context.save();
    applyNodeStyle(style);
    context.beginPath();
    for (let index = start; index < end; index += 1) {
      const projected = projectedNodes[index];
      if (projected.node.renderStyleKey !== key) continue;
      context.moveTo(projected.x + projected.radius, projected.y);
      context.arc(projected.x, projected.y, projected.radius, 0, Math.PI * 2);
      count += 1;
    }
    if (count) {
      context.fill();
      context.stroke();
    }
    context.restore();
  }

  function drawSettledNodes() {
    nodeStyles.forEach((style, key) => drawNonOverlappingStyle(key, style));
    projectedNodes.forEach((projected) => {
      if (projected.overlapping) drawNodeDirect(projected);
    });
    context.shadowBlur = 0;
    context.globalAlpha = 1;
  }

  function drawNonOverlappingStyle(key, style) {
    if (style.selected) {
      const selected = nodeLookup.get(emphasis.selectedId)?.projected;
      if (selected && !selected.overlapping) drawNodeDirect(selected);
      return;
    }
    let count = 0;
    context.save();
    applyNodeStyle(style);
    context.beginPath();
    projectedNodes.forEach((projected) => {
      if (projected.overlapping || projected.node.renderStyleKey !== key) return;
      context.moveTo(projected.x + projected.radius, projected.y);
      context.arc(projected.x, projected.y, projected.radius, 0, Math.PI * 2);
      count += 1;
    });
    if (count) {
      context.fill();
      context.stroke();
    }
    context.restore();
  }

  function drawNodeDirect(projected) {
    const style = projected.node.renderStyle;
    applyNodeStyle(style);
    context.beginPath();
    context.arc(
      projected.x,
      projected.y,
      style.selected ? projected.radius + 1.7 : projected.radius,
      0,
      Math.PI * 2
    );
    context.fill();
    context.stroke();
    if (style.selected) {
      context.shadowBlur = 0;
      context.globalAlpha = style.outerAlpha;
      context.strokeStyle = PALETTE.gold;
      context.lineWidth = 1;
      context.beginPath();
      context.arc(projected.x, projected.y, projected.radius + 5.2, 0, Math.PI * 2);
      context.stroke();
    }
  }

  function applyNodeStyle(style) {
    context.globalAlpha = style.alpha;
    context.fillStyle = style.fill;
    context.strokeStyle = style.stroke;
    context.lineWidth = style.lineWidth;
    context.shadowBlur = style.shadowBlur;
    context.shadowColor = style.shadowColor;
  }

  function drawLabels(projected, emphasis) {
    const labelLimit = clamp(Math.floor((width * height) / 17000), 18, 92);
    const priority = [...projected].sort((a, b) => {
      return labelPriority(b.node, emphasis) - labelPriority(a.node, emphasis)
        || a.node.id.localeCompare(b.node.id);
    });
    const occupied = [];
    let labelCount = 0;

    context.save();
    context.font = "600 11px ui-serif, Georgia, serif";
    context.textBaseline = "middle";
    priority.some((projectedNode) => {
      if (labelCount >= labelLimit) return true;
      const priorityScore = labelPriority(projectedNode.node, emphasis);
      if (priorityScore <= 0) return false;
      const label = truncate(projectedNode.node.label || projectedNode.node.id, 34);
      const textWidth = Math.ceil(context.measureText(label).width);
      const x = projectedNode.x + projectedNode.radius + 6;
      const y = projectedNode.y;
      const box = { x: x - 3, y: y - 9, width: textWidth + 7, height: 18 };
      if (occupied.some((other) => intersects(box, other))) return false;
      if (box.x + box.width > width - 5 || box.y < 4 || box.y + box.height > height - 4) return false;

      const filterMatch = !emphasis.filterActive || emphasis.matches.has(projectedNode.node.id);
      const related = filterMatch && (!emphasis.active || emphasis.all.has(projectedNode.node.id));
      context.globalAlpha = related ? 0.94 : 0.16;
      context.fillStyle = "rgba(255,253,246,.86)";
      context.fillRect(box.x, box.y, box.width, box.height);
      context.fillStyle = projectedNode.node.id === emphasis.selectedId ? "#76551e" : PALETTE.ink;
      context.fillText(label, x, y + 0.5);
      occupied.push(box);
      labelCount += 1;
      return false;
    });
    context.restore();
  }

  function drawTooltip(projected) {
    if (!projected) return;
    const label = truncate(projected.node.label || projected.node.id, 46);
    const status = projected.node.resolved
      ? "resolved"
      : projected.node.actionable
        ? "actionable"
        : "blocked";
    context.save();
    context.font = "600 12px ui-serif, Georgia, serif";
    const widthNeeded = Math.max(context.measureText(label).width, context.measureText(status).width) + 22;
    const panelWidth = clamp(widthNeeded, 112, 292);
    const panelHeight = 58;
    const x = clamp(projected.x + 14, 8, width - panelWidth - 8);
    const y = clamp(projected.y - panelHeight - 10, 8, height - panelHeight - 8);
    context.shadowBlur = 14;
    context.shadowColor = "rgba(43,45,39,.17)";
    context.fillStyle = "rgba(255,253,246,.97)";
    context.strokeStyle = "rgba(104,96,81,.35)";
    context.lineWidth = 1;
    roundedRect(context, x, y, panelWidth, panelHeight, 5);
    context.fill();
    context.shadowBlur = 0;
    context.stroke();
    context.fillStyle = PALETTE.ink;
    context.fillText(label, x + 11, y + 18);
    context.font = "500 10px ui-sans-serif, system-ui, sans-serif";
    context.fillStyle = PALETTE.mutedInk;
    context.fillText(`${status} · depth ${projected.node.layer}`, x + 11, y + 35);
    context.fillText(`${projected.node.impact} downstream`, x + 11, y + 49);
    context.restore();
  }

  function projectNodes() {
    const cosYaw = Math.cos(camera.yaw);
    const sinYaw = Math.sin(camera.yaw);
    const cosPitch = Math.cos(camera.pitch);
    const sinPitch = Math.sin(camera.pitch);
    const focal = Math.max(480, world.span * 2.7);
    const scale = fitScale() * camera.zoom;
    const zoomRadius = Math.sqrt(camera.zoom);
    const screenX = width / 2 + camera.panX;
    const screenY = height / 2 + camera.panY;
    projectedNodes.forEach((projected) => {
      const node = projected.node;
      const centerX = node.x - world.centerX;
      const centerY = node.y - world.centerY;
      const centerZ = node.z - world.centerZ;
      const rotatedX = centerX * cosYaw - centerZ * sinYaw;
      const yawDepth = centerX * sinYaw + centerZ * cosYaw;
      const rotatedY = centerY * cosPitch - yawDepth * sinPitch;
      const depth = centerY * sinPitch + yawDepth * cosPitch;
      const perspective = clamp(focal / (focal + depth), 0.38, 2.4);
      projected.x = screenX + rotatedX * scale * perspective;
      projected.y = screenY + rotatedY * scale * perspective;
      projected.depth = depth;
      projected.radius = clamp(node.baseRadius * perspective * zoomRadius, 2.6, 17);
    });
    projectedNodes.sort((a, b) => b.depth - a.depth);
  }

  function classifyNodeOverlaps() {
    const grid = new Map();
    let queryStamp = 0;
    projectedNodes.forEach((projected) => {
      const style = projected.node.renderStyle;
      const padding = style.selected
        ? Math.max(5.2, style.shadowBlur)
        : Math.max(style.lineWidth, style.shadowBlur);
      projected.visualRadius = projected.radius + padding;
      projected.overlapping = false;
      queryStamp += 1;

      const minCellX = Math.floor((projected.x - projected.visualRadius) / OVERLAP_GRID_SIZE);
      const maxCellX = Math.floor((projected.x + projected.visualRadius) / OVERLAP_GRID_SIZE);
      const minCellY = Math.floor((projected.y - projected.visualRadius) / OVERLAP_GRID_SIZE);
      const maxCellY = Math.floor((projected.y + projected.visualRadius) / OVERLAP_GRID_SIZE);

      for (let cellX = minCellX; cellX <= maxCellX; cellX += 1) {
        for (let cellY = minCellY; cellY <= maxCellY; cellY += 1) {
          const bucket = grid.get(overlapGridKey(cellX, cellY));
          if (!bucket) continue;
          bucket.forEach((candidate) => {
            if (candidate.queryStamp === queryStamp) return;
            candidate.queryStamp = queryStamp;
            const dx = candidate.x - projected.x;
            const dy = candidate.y - projected.y;
            const distance = candidate.visualRadius + projected.visualRadius;
            if (dx * dx + dy * dy >= distance * distance) return;
            candidate.overlapping = true;
            projected.overlapping = true;
          });
        }
      }

      for (let cellX = minCellX; cellX <= maxCellX; cellX += 1) {
        for (let cellY = minCellY; cellY <= maxCellY; cellY += 1) {
          const key = overlapGridKey(cellX, cellY);
          const bucket = grid.get(key);
          if (bucket) {
            bucket.push(projected);
          } else {
            grid.set(key, [projected]);
          }
        }
      }
    });
  }

  function fitScale() {
    if (!nodes.length) return 1;
    const cosYaw = Math.abs(Math.cos(camera.yaw));
    const sinYaw = Math.abs(Math.sin(camera.yaw));
    const cosPitch = Math.abs(Math.cos(camera.pitch));
    const sinPitch = Math.abs(Math.sin(camera.pitch));
    const horizontalSpan = world.spanX * cosYaw + world.spanZ * sinYaw;
    const yawDepthSpan = world.spanX * sinYaw + world.spanZ * cosYaw;
    const verticalSpan = world.spanY * cosPitch + yawDepthSpan * sinPitch;
    const horizontal = width / Math.max(1, horizontalSpan * 1.28);
    const vertical = height / Math.max(1, verticalSpan * 1.28);
    return clamp(Math.min(horizontal, vertical), 0.035, 4.5);
  }

  function cacheVisualStyles() {
    nodeStyles = new Map();
    nodes.forEach((node) => {
      const selected = node.id === emphasis.selectedId;
      const ancestor = emphasis.ancestors.has(node.id);
      const dependent = emphasis.dependents.has(node.id);
      const derived = emphasis.derived.has(node.id);
      const match = emphasis.matches.has(node.id);
      const related = selected || ancestor || dependent || derived || match;
      const filterMismatch = emphasis.filterActive && !match;
      const faded = filterMismatch || (emphasis.active && !related);
      if (selected) {
        const key = [statusKey(node), "selected", faded ? "faded" : "normal"].join("|");
        node.renderStyleKey = key;
        if (!nodeStyles.has(key)) {
          nodeStyles.set(key, {
            selected: true,
            alpha: faded ? 0.16 : 0.96,
            outerAlpha: 0.96,
            fill: statusColor(node),
            stroke: PALETTE.ivory,
            lineWidth: 3.1,
            shadowBlur: 15,
            shadowColor: "rgba(194,147,62,.58)"
          });
        }
        node.renderStyle = nodeStyles.get(key);
        return;
      }

      const strokeKind = ancestor
        ? "ancestor"
        : dependent
          ? "dependent"
          : derived
            ? "derived"
            : "ordinary";
      const key = [
        statusKey(node),
        strokeKind,
        faded ? "faded" : "normal",
        match ? "match" : "plain"
      ].join("|");
      node.renderStyleKey = key;
      if (!nodeStyles.has(key)) {
        nodeStyles.set(key, {
          selected: false,
          alpha: faded ? 0.16 : 0.96,
          outerAlpha: 0,
          fill: statusColor(node),
          stroke: strokeKind === "ancestor"
            ? PALETTE.gold
            : strokeKind === "dependent"
              ? "#d8e7d9"
              : strokeKind === "derived"
                ? PALETTE.derived
                : "rgba(255,253,246,.9)",
          lineWidth: related ? 2.1 : 1.15,
          shadowBlur: match ? 9 : 0,
          shadowColor: "rgba(255,249,232,.9)"
        });
      }
      node.renderStyle = nodeStyles.get(key);
    });

    edges.forEach((edge) => {
      const kind = edgeEmphasis(edge, emphasis);
      const filterMismatch = emphasis.filterActive
        && !(emphasis.matches.has(edge.source.id) && emphasis.matches.has(edge.target.id));
      edge.renderKind = filterMismatch || (emphasis.active && !kind)
        ? "faded"
        : kind === "dependent"
          ? "dependent"
          : kind === "derived"
            ? "derived"
            : kind
              ? "focused"
              : "normal";
    });
  }

  function markCameraMotion() {
    movingUntil = performance.now() + MOTION_SETTLE_MS;
    if (settleTimer) return;
    const settle = () => {
      const remaining = movingUntil - performance.now();
      if (remaining > 0) {
        settleTimer = window.setTimeout(settle, remaining + 4);
        return;
      }
      settleTimer = 0;
      scheduleDraw();
    };
    settleTimer = window.setTimeout(settle, MOTION_SETTLE_MS + 4);
  }

  function preventContextMenu(event) {
    event.preventDefault();
  }

  function handlePointerDown(event) {
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    const point = pointerPoint(event);
    pointers.set(event.pointerId, point);
    if (pointers.size === 1) {
      gesture = {
        kind: event.shiftKey || event.button === 1 || event.button === 2 ? "pan" : "rotate",
        startX: point.x,
        startY: point.y,
        lastX: point.x,
        lastY: point.y,
        moved: false,
        pointerId: event.pointerId
      };
    } else if (pointers.size === 2) {
      gesture = twoPointerGesture();
    }
    event.preventDefault();
  }

  function handlePointerMove(event) {
    const point = pointerPoint(event);
    if (!pointers.has(event.pointerId)) {
      const nextHover = hitTest(point.x, point.y);
      if (nextHover?.id !== hoverNode?.id) {
        hoverNode = nextHover;
        canvas.style.cursor = hoverNode ? "pointer" : "grab";
        scheduleDraw();
      }
      return;
    }

    pointers.set(event.pointerId, point);
    if (pointers.size >= 2) {
      const next = twoPointerGesture();
      if (gesture?.kind === "multi") {
        camera.panX += next.centerX - gesture.centerX;
        camera.panY += next.centerY - gesture.centerY;
        if (gesture.distance > 0) {
          camera.zoom = clamp(
            camera.zoom * (next.distance / gesture.distance),
            CAMERA_LIMITS.minZoom,
            CAMERA_LIMITS.maxZoom
          );
        }
      }
      gesture = next;
      markCameraMotion();
      scheduleDraw();
      event.preventDefault();
      return;
    }

    if (!gesture || gesture.pointerId !== event.pointerId) return;
    const dx = point.x - gesture.lastX;
    const dy = point.y - gesture.lastY;
    gesture.moved ||= Math.hypot(point.x - gesture.startX, point.y - gesture.startY) > 5;
    gesture.lastX = point.x;
    gesture.lastY = point.y;
    if (gesture.kind === "pan") {
      camera.panX += dx;
      camera.panY += dy;
      canvas.style.cursor = "grabbing";
    } else {
      camera.yaw = normalizeAngle(camera.yaw + dx * 0.0065);
      camera.pitch = clamp(camera.pitch + dy * 0.005, CAMERA_LIMITS.minPitch, CAMERA_LIMITS.maxPitch);
      canvas.style.cursor = "grabbing";
    }
    markCameraMotion();
    scheduleDraw();
    event.preventDefault();
  }

  function handlePointerUp(event) {
    const endingGesture = gesture;
    const point = pointerPoint(event);
    const wasSingle = pointers.size === 1 && endingGesture?.pointerId === event.pointerId;
    pointers.delete(event.pointerId);
    if (wasSingle && !endingGesture.moved && event.button === 0) {
      const selected = hitTest(point.x, point.y);
      try {
        onSelect(selected?.id ?? null);
      } catch (error) {
        report(error);
      }
    }
    if (pointers.size === 1) {
      const [remaining] = pointers.values();
      const [pointerId] = pointers.keys();
      gesture = {
        kind: "rotate",
        startX: remaining.x,
        startY: remaining.y,
        lastX: remaining.x,
        lastY: remaining.y,
        moved: true,
        pointerId
      };
    } else if (!pointers.size) {
      gesture = null;
      canvas.style.cursor = hoverNode ? "pointer" : "grab";
    }
    event.preventDefault();
  }

  function handleWheel(event) {
    const intensity = reducedMotion ? 0.0012 : 0.0018;
    zoom(Math.exp(-event.deltaY * intensity));
    event.preventDefault();
  }

  function twoPointerGesture() {
    const [first, second] = [...pointers.values()];
    return {
      kind: "multi",
      centerX: (first.x + second.x) / 2,
      centerY: (first.y + second.y) / 2,
      distance: Math.max(1, Math.hypot(second.x - first.x, second.y - first.y)),
      moved: true
    };
  }

  function pointerPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function hitTest(x, y) {
    for (let index = projectedNodes.length - 1; index >= 0; index -= 1) {
      const node = projectedNodes[index];
      if (Math.hypot(node.x - x, node.y - y) <= Math.max(8, node.radius + 4)) {
        return node.node;
      }
    }
    return null;
  }

  const pointerDown = guarded(handlePointerDown);
  const pointerMove = guarded(handlePointerMove);
  const pointerUp = guarded(handlePointerUp);
  const wheel = guarded(handleWheel);

  canvas.addEventListener("contextmenu", preventContextMenu);
  canvas.addEventListener("pointerdown", pointerDown);
  canvas.addEventListener("pointermove", pointerMove);
  canvas.addEventListener("pointerup", pointerUp);
  canvas.addEventListener("pointercancel", pointerUp);
  canvas.addEventListener("wheel", wheel, { passive: false });
  window.addEventListener("resize", resize);
  resize();

  return {
    setModel: guarded(setModel),
    resize: guarded(resize),
    zoom: guarded(zoom),
    resetCamera: guarded(resetCamera),
    getCamera,
    setCamera: guarded(setCamera),
    destroy
  };
}

function normalizeNode(node, index, count) {
  const overview = node?.overview || node?.position || {};
  const fallbackAngle = count ? (index / count) * Math.PI * 2 : 0;
  return {
    ...node,
    id: String(node?.id ?? `node-${index}`),
    label: String(node?.label ?? node?.id ?? `node-${index}`),
    x: finiteNumber(overview.x, Math.cos(fallbackAngle) * 100),
    y: finiteNumber(overview.y, 0),
    z: finiteNumber(overview.z, Math.sin(fallbackAngle) * 100),
    layer: Math.max(0, Math.round(finiteNumber(overview.depth, 0))),
    impact: Math.max(0, Math.round(finiteNumber(overview.downstream_impact, 0))),
    islandId: String(overview.island ?? node?.id ?? `node-${index}`),
    rootCount: Math.max(1, Math.round(finiteNumber(overview.root_count, 1))),
    baseRadius: 3.3 + Math.log2(Math.max(0, finiteNumber(overview.downstream_impact, 0)) + 1) * 1.15
  };
}

function worldBounds(nodes) {
  if (!nodes.length) return emptyWorld();
  const xs = nodes.map((node) => node.x);
  const ys = nodes.map((node) => node.y);
  const zs = nodes.map((node) => node.z);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  const spanX = Math.max(120, maxX - minX);
  const spanY = Math.max(120, maxY - minY);
  const spanZ = Math.max(120, maxZ - minZ);
  return {
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    centerZ: (minZ + maxZ) / 2,
    spanX,
    spanY,
    spanZ,
    span: Math.max(spanX, spanY, spanZ)
  };
}

function emptyWorld() {
  return { centerX: 0, centerY: 0, centerZ: 0, spanX: 120, spanY: 120, spanZ: 120, span: 120 };
}

function emphasisState(model) {
  const selectedId = model.selectedId == null ? null : String(model.selectedId);
  const filterActive = Boolean(model.filterActive);
  const ancestors = idSet(model.ancestorIds);
  const dependents = idSet(model.directDependentIds);
  const matches = idSet(model.matchIds);
  const derived = idSet(model.derivedHighlightIds);
  const all = new Set([...ancestors, ...dependents, ...matches, ...derived]);
  if (selectedId) all.add(selectedId);
  return {
    selectedId,
    ancestors,
    dependents,
    matches,
    derived,
    filterActive,
    all,
    active: filterActive || all.size > 0
  };
}

function edgeEmphasis(edge, emphasis) {
  if (!emphasis.active) return null;
  if (edge.source.id === emphasis.selectedId && emphasis.dependents.has(edge.target.id)) {
    return "dependent";
  }
  if (
    emphasis.ancestors.has(edge.source.id)
    && (emphasis.ancestors.has(edge.target.id) || edge.target.id === emphasis.selectedId)
  ) {
    return "ancestry";
  }
  if (emphasis.derived.has(edge.source.id) && emphasis.derived.has(edge.target.id)) {
    return "derived";
  }
  return emphasis.all.has(edge.source.id) && emphasis.all.has(edge.target.id) ? "related" : null;
}

function labelPriority(node, emphasis) {
  if (node.id === emphasis.selectedId) return 100000;
  if (emphasis.matches.has(node.id)) return 90000;
  if (emphasis.ancestors.has(node.id)) return 80000 + node.impact;
  if (emphasis.dependents.has(node.id)) return 70000 + node.impact;
  if (emphasis.derived.has(node.id)) return 60000 + node.impact;
  if (node.actionable) return 10000 + node.impact;
  return node.impact >= 6 ? node.impact : 0;
}

function statusColor(node) {
  if (node.resolved) return PALETTE.resolved;
  if (node.actionable) return PALETTE.actionable;
  return PALETTE.blocked;
}

function statusKey(node) {
  if (node.resolved) return "resolved";
  if (node.actionable) return "actionable";
  return "blocked";
}

function overlapGridKey(cellX, cellY) {
  return (cellX + 32768) * 65536 + cellY + 32768;
}

function idSet(values) {
  return new Set(Array.isArray(values) || values instanceof Set ? [...values].map(String) : []);
}

function sanitizeCamera(camera) {
  return {
    yaw: normalizeAngle(finiteNumber(camera.yaw, DEFAULT_CAMERA.yaw)),
    pitch: clamp(finiteNumber(camera.pitch, DEFAULT_CAMERA.pitch), CAMERA_LIMITS.minPitch, CAMERA_LIMITS.maxPitch),
    zoom: clamp(finiteNumber(camera.zoom, DEFAULT_CAMERA.zoom), CAMERA_LIMITS.minZoom, CAMERA_LIMITS.maxZoom),
    panX: clamp(finiteNumber(camera.panX, 0), -100000, 100000),
    panY: clamp(finiteNumber(camera.panY, 0), -100000, 100000)
  };
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalizeAngle(value) {
  const turn = Math.PI * 2;
  return ((value + Math.PI) % turn + turn) % turn - Math.PI;
}

function truncate(value, maximum) {
  return value.length > maximum ? `${value.slice(0, maximum - 1)}…` : value;
}

function intersects(a, b) {
  return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
}

function roundedRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.roundRect(x, y, width, height, radius);
}

function notifyInitializationError(onError, error) {
  try {
    if (typeof onError === "function") onError(error);
  } catch {
    // Preserve the renderer's original initialization error.
  }
}
