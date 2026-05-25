const canvas = document.querySelector("#matter-map");
const ctx = canvas.getContext("2d");

const colors = ["#c7433a", "#245c92", "#4c7a55", "#b47b26", "#171717"];
const labels = [
  "public demo",
  "shared view",
  "clear conditions",
  "next move",
  "open question",
  "resolved blocker",
  "outside collaborators",
  "world-readable map"
];

let nodes = [];
let pointer = { x: 0, y: 0, active: false };

function resize() {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = window.innerWidth * ratio;
  canvas.height = canvas.parentElement.offsetHeight * ratio;
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${canvas.parentElement.offsetHeight}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  const count = window.innerWidth < 700 ? 18 : 28;
  nodes = Array.from({ length: count }, (_, index) => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * canvas.parentElement.offsetHeight,
    vx: (Math.random() - 0.5) * 0.34,
    vy: (Math.random() - 0.5) * 0.34,
    radius: index % 5 === 0 ? 9 : 5,
    color: colors[index % colors.length],
    label: labels[index % labels.length],
    showLabel: index % 7 === 0
  }));
}

function drawNode(node) {
  ctx.beginPath();
  ctx.fillStyle = node.color;
  ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
  ctx.fill();

  const clearOfHeadline =
    node.x > window.innerWidth * 0.58 ||
    (node.y > canvas.parentElement.offsetHeight * 0.72 && node.x > window.innerWidth * 0.28);
  if (node.showLabel && clearOfHeadline && window.innerWidth > 760) {
    ctx.font = "600 13px Inter, system-ui, sans-serif";
    ctx.fillStyle = "rgba(23, 23, 23, 0.72)";
    ctx.fillText(node.label, node.x + 12, node.y + 5);
  }
}

function drawLink(a, b, opacity) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = `rgba(23, 23, 23, ${opacity})`;
  ctx.lineWidth = 1;
  ctx.stroke();
}

function animate() {
  const width = window.innerWidth;
  const height = canvas.parentElement.offsetHeight;
  ctx.clearRect(0, 0, width, height);

  nodes.forEach((node) => {
    if (pointer.active) {
      const dx = pointer.x - node.x;
      const dy = pointer.y - node.y;
      const distance = Math.hypot(dx, dy);
      if (distance < 180) {
        node.vx -= dx * 0.000025;
        node.vy -= dy * 0.000025;
      }
    }

    node.x += node.vx;
    node.y += node.vy;

    if (node.x < 20 || node.x > width - 20) node.vx *= -1;
    if (node.y < 20 || node.y > height - 20) node.vy *= -1;
  });

  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const a = nodes[i];
      const b = nodes[j];
      const distance = Math.hypot(a.x - b.x, a.y - b.y);
      if (distance < 190) {
        drawLink(a, b, Math.max(0.03, 0.16 - distance / 1300));
      }
    }
  }

  nodes.forEach(drawNode);
  requestAnimationFrame(animate);
}

window.addEventListener("resize", resize);
window.addEventListener("pointermove", (event) => {
  pointer = { x: event.clientX, y: event.clientY, active: true };
});
window.addEventListener("pointerleave", () => {
  pointer.active = false;
});

resize();
animate();
