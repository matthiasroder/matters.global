"""PNG snapshot of a ``matters view`` payload.

``matters view`` writes a self-contained HTML page and, unless ``--no-open``
is passed, asks the desktop to open it. A chat assistant has no desktop.
``--png`` writes this image of the same slice so the assistant can attach a
picture: overview ``x``/``z`` positions, an edge from each prerequisite to
the matter that waits on it, and the focus matter ringed in gold.

The drawing uses the standard library only. Colours match the HTML page.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path


PNG_WIDTH = 1100
PNG_HEIGHT = 720

PAPER = (247, 241, 228)
INK = (40, 48, 47)
EDGE = (104, 96, 81)
ACTIONABLE = (47, 127, 90)
BLOCKED = (184, 95, 73)
RESOLVED = (104, 123, 130)
FOCUS = (194, 147, 62)

_NODE_RADIUS = 14
_FOCUS_RADIUS = 18
_FONT_SCALE = 2
_UNKNOWN_GLYPH = (0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F)


def render_view_png(payload, *, width=PNG_WIDTH, height=PNG_HEIGHT):
    """Return a PNG of ``payload``, the dict ``build_view_payload`` builds."""

    rgb = rasterize_view(payload, width=width, height=height)
    return encode_png(width, height, rgb)


def write_view_png(payload, path):
    """Write ``render_view_png(payload)`` to ``path`` and return that path."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(render_view_png(payload))
    return destination


def rasterize_view(payload, *, width=PNG_WIDTH, height=PNG_HEIGHT):
    """Return packed RGB bytes, ``width * height * 3`` long."""

    if width < 200 or height < 160:
        raise ValueError(f"png canvas is too small: {width}x{height}")

    pixels = bytearray(bytes(PAPER) * (width * height))
    nodes = list(payload.get("nodes") or [])
    by_id = {node["id"]: node for node in nodes}
    placed = _place(nodes, width, height)

    for edge in payload.get("edges") or []:
        source = placed.get(edge["source"])
        target = placed.get(edge["target"])
        if source is None or target is None:
            continue
        x0, y0, x1, y1 = _shorten(
            source[0], source[1], target[0], target[1], _NODE_RADIUS + 2, _FOCUS_RADIUS + 6
        )
        _stroke(pixels, width, height, x0, y0, x1, y1, EDGE, radius=1.6)
        _arrow(pixels, width, height, x0, y0, x1, y1, INK)

    focus_id = payload.get("matter")
    for node in nodes:
        spot = placed.get(node["id"])
        if spot is None:
            continue
        color = _status_color(node)
        if node["id"] == focus_id:
            _fill_circle(pixels, width, height, spot[0], spot[1], _FOCUS_RADIUS, FOCUS)
            _fill_circle(pixels, width, height, spot[0], spot[1], _NODE_RADIUS - 3, color)
        else:
            _fill_circle(pixels, width, height, spot[0], spot[1], _NODE_RADIUS, color)

    for node in nodes:
        spot = placed.get(node["id"])
        if spot is None:
            continue
        label = _fit_label(str(node.get("label") or node["id"]))
        _draw_text(
            pixels,
            width,
            height,
            int(spot[0] - _text_width(label, _FONT_SCALE) / 2),
            int(spot[1] + _FOCUS_RADIUS + 6),
            label,
            _FONT_SCALE,
            INK,
        )

    _draw_chrome(pixels, width, height, payload, by_id)
    return pixels


def encode_png(width, height, rgb):
    """Encode packed RGB bytes as an 8-bit PNG with no extra dependencies."""

    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError(f"rgb length {len(rgb)} does not match {width}x{height}")

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(rgb[start : start + stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _status_color(node):
    if node.get("resolved") is True:
        return RESOLVED
    if node.get("actionable") is True:
        return ACTIONABLE
    if node.get("blocked") is True:
        return BLOCKED
    return RESOLVED


def _place(nodes, width, height):
    plot_left, plot_top = 40, 56
    plot_right, plot_bottom = width - 40, height - 72
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    coords = []
    for node in nodes:
        overview = node.get("overview") or {}
        if "x" not in overview or "z" not in overview:
            continue
        coords.append((node["id"], float(overview["x"]), float(overview["z"])))
    if not coords:
        return {}

    xs = [item[1] for item in coords]
    zs = [item[2] for item in coords]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    span_x = max_x - min_x
    span_z = max_z - min_z
    center_x = (plot_left + plot_right) / 2
    center_y = (plot_top + plot_bottom) / 2
    if span_x == 0 and span_z == 0:
        return {item[0]: (center_x, center_y) for item in coords}

    scale_x = plot_w if span_x == 0 else plot_w / span_x
    scale_z = plot_h if span_z == 0 else plot_h / span_z
    scale = min(scale_x, scale_z)
    used_w = span_x * scale
    used_h = span_z * scale
    origin_x = plot_left + (plot_w - used_w) / 2
    origin_y = plot_top + (plot_h - used_h) / 2
    placed = {}
    for matter_id, x_value, z_value in coords:
        placed[matter_id] = (
            origin_x + (x_value - min_x) * scale,
            origin_y + (max_z - z_value) * scale,
        )
    if span_x == 0:
        for matter_id in placed:
            placed[matter_id] = (center_x, placed[matter_id][1])
    if span_z == 0:
        for matter_id in placed:
            placed[matter_id] = (placed[matter_id][0], center_y)
    return placed


def _draw_chrome(pixels, width, height, payload, by_id):
    focus = by_id.get(payload.get("matter"))
    title = str(focus["label"]) if focus else str(payload.get("matter") or "matters")
    counted = len(payload.get("nodes") or [])
    matter_word = "matter" if counted == 1 else "matters"
    subtitle = f"{counted} {matter_word}"
    if payload.get("status_available") is False:
        subtitle += ", structure only"
    _draw_text(pixels, width, height, 36, 16, _fit_label(title, limit=48), 3, INK)
    _draw_text(pixels, width, height, 36, 42, subtitle, 2, EDGE)

    legend_y = height - 36
    cursor = 36
    if payload.get("status_available") is False:
        cursor = _legend_item(pixels, width, height, cursor, legend_y, RESOLVED, "status unavailable")
    else:
        cursor = _legend_item(pixels, width, height, cursor, legend_y, ACTIONABLE, "actionable")
        cursor = _legend_item(pixels, width, height, cursor, legend_y, BLOCKED, "blocked")
        cursor = _legend_item(pixels, width, height, cursor, legend_y, RESOLVED, "resolved")
    _legend_item(pixels, width, height, cursor, legend_y, FOCUS, "this matter", ring=True)


def _legend_item(pixels, width, height, x, y, color, text, ring=False):
    if ring:
        _fill_circle(pixels, width, height, x + 8, y + 7, 8, color)
        _fill_circle(pixels, width, height, x + 8, y + 7, 4, PAPER)
    else:
        _fill_rect(pixels, width, height, x, y, 16, 14, color)
    label_x = x + 22
    _draw_text(pixels, width, height, label_x, y, text, 2, INK)
    return label_x + _text_width(text, 2) + 28


def _fit_label(text, limit=22):
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 2].rstrip() + ".."


def _text_width(text, scale):
    if not text:
        return 0
    advance = (5 + 1) * scale
    return len(text) * advance - scale


def _draw_text(pixels, width, height, x, y, text, scale, color):
    cursor = x
    advance = (5 + 1) * scale
    for character in text:
        glyph = _GLYPHS.get(character, _UNKNOWN_GLYPH)
        if character != " ":
            for row, bits in enumerate(glyph):
                for col in range(5):
                    if bits & (1 << (4 - col)):
                        _fill_rect(
                            pixels,
                            width,
                            height,
                            cursor + col * scale,
                            y + row * scale,
                            scale,
                            scale,
                            color,
                        )
        cursor += advance


def _fill_rect(pixels, width, height, x, y, rect_w, rect_h, color):
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x + rect_w))
    y1 = min(height, int(y + rect_h))
    for py in range(y0, y1):
        row = py * width
        for px in range(x0, x1):
            start = (row + px) * 3
            pixels[start : start + 3] = color


def _fill_circle(pixels, width, height, cx, cy, radius, color):
    r2 = radius * radius
    x0 = max(0, int(cx - radius))
    x1 = min(width - 1, int(cx + radius))
    y0 = max(0, int(cy - radius))
    y1 = min(height - 1, int(cy + radius))
    for py in range(y0, y1 + 1):
        dy = py + 0.5 - cy
        row = py * width
        for px in range(x0, x1 + 1):
            dx = px + 0.5 - cx
            if dx * dx + dy * dy <= r2:
                start = (row + px) * 3
                pixels[start : start + 3] = color


def _stroke(pixels, width, height, x0, y0, x1, y1, color, radius):
    length = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(length))
    for step in range(steps + 1):
        t = step / steps
        _fill_circle(
            pixels,
            width,
            height,
            x0 + (x1 - x0) * t,
            y0 + (y1 - y0) * t,
            radius,
            color,
        )


def _arrow(pixels, width, height, x0, y0, x1, y1, color):
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 8:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = (x1, y1)
    left = (x1 - ux * 22 + px * 9, y1 - uy * 22 + py * 9)
    right = (x1 - ux * 22 - px * 9, y1 - uy * 22 - py * 9)
    _fill_triangle(pixels, width, height, tip, left, right, color)


def _fill_triangle(pixels, width, height, a, b, c, color):
    xs = [a[0], b[0], c[0]]
    ys = [a[1], b[1], c[1]]
    min_x = max(0, int(min(xs)))
    max_x = min(width - 1, int(max(xs)) + 1)
    min_y = max(0, int(min(ys)))
    max_y = min(height - 1, int(max(ys)) + 1)
    area = _cross(a, b, c)
    if area == 0:
        return
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            point = (px + 0.5, py + 0.5)
            w0 = _cross(b, c, point) / area
            w1 = _cross(c, a, point) / area
            w2 = _cross(a, b, point) / area
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                start = (py * width + px) * 3
                pixels[start : start + 3] = color


def _cross(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _shorten(x0, y0, x1, y1, trim_start, trim_end):
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length <= trim_start + trim_end + 1:
        return x0, y0, x1, y1
    ux, uy = dx / length, dy / length
    return (
        x0 + ux * trim_start,
        y0 + uy * trim_start,
        x1 - ux * trim_end,
        y1 - uy * trim_end,
    )


def _parse_font(source):
    glyphs = {}
    blocks = [block for block in source.strip().split("\n\n") if block.strip()]
    for block in blocks:
        rows = block.split("\n")
        name, art = rows[0], rows[1:]
        if len(art) != 7 or any(len(row) != 5 for row in art):
            raise ValueError(f"font glyph {name!r} must be 7 rows of 5")
        bits = []
        for row in art:
            value = 0
            for col, mark in enumerate(row):
                if mark == "0":
                    value |= 1 << (4 - col)
                elif mark != ".":
                    raise ValueError(f"font glyph {name!r} has {mark!r}")
            bits.append(value)
        glyphs[" " if name == "sp" else name] = tuple(bits)
    return glyphs


_FONT = """
sp
.....
.....
.....
.....
.....
.....
.....

a
.....
.000.
....0
.0000
0...0
0...0
.0000

b
0....
0....
0.00.
00..0
0...0
00..0
0.00.

c
.....
.000.
0...0
0....
0....
0...0
.000.

d
....0
....0
.00.0
0..00
0...0
0..00
.00.0

e
.....
.000.
0...0
00000
0....
0...0
.000.

f
.000.
0...0
0....
0000.
0....
0....
0....

g
.000.
0...0
0...0
.0000
....0
0...0
.000.

h
0....
0....
0.00.
00..0
0...0
0...0
0...0

i
..0..
.....
..0..
..0..
..0..
..0..
..0..

j
...0.
.....
...0.
...0.
...0.
0..0.
.00..

k
0....
0..0.
0.0..
00...
0.0..
0..0.
0...0

l
.0...
.0...
.0...
.0...
.0...
.0...
.000.

m
.....
00.00
0.0.0
0.0.0
0.0.0
0.0.0
0...0

n
.....
0.00.
00..0
0...0
0...0
0...0
0...0

o
.....
.000.
0...0
0...0
0...0
0...0
.000.

p
0.00.
00..0
0...0
0.00.
0....
0....
0....

q
.00.0
0..00
0...0
.00.0
....0
....0
....0

r
.....
0.00.
00..0
0....
0....
0....
0....

s
.....
.0000
0....
.000.
....0
0...0
.000.

t
.0...
.0...
0000.
.0...
.0...
.0...
..00.

u
.....
0...0
0...0
0...0
0...0
0..00
.00..

v
.....
0...0
0...0
0...0
0...0
.0.0.
..0..

w
.....
0...0
0...0
0.0.0
0.0.0
0.0.0
.0.0.

x
.....
0...0
.0.0.
..0..
..0..
.0.0.
0...0

y
.....
0...0
0...0
.0000
....0
....0
.000.

z
.....
00000
...0.
..0..
.0...
0....
00000

0
.000.
0...0
0..00
0.0.0
00..0
0...0
.000.

1
..0..
.00..
..0..
..0..
..0..
..0..
.000.

2
.000.
0...0
....0
..00.
.0...
0....
00000

3
.000.
0...0
....0
..00.
....0
0...0
.000.

4
...0.
..00.
.0.0.
0..0.
00000
...0.
...0.

5
00000
0....
0000.
....0
....0
0...0
.000.

6
.000.
0....
0....
0000.
0...0
0...0
.000.

7
00000
....0
...0.
..0..
.0...
.0...
.0...

8
.000.
0...0
0...0
.000.
0...0
0...0
.000.

9
.000.
0...0
0...0
.0000
....0
....0
.000.

-
.....
.....
.....
00000
.....
.....
.....

_
.....
.....
.....
.....
.....
.....
00000

.
.....
.....
.....
.....
.....
..00.
..00.

:
.....
..0..
.....
.....
.....
..0..
.....

,
.....
.....
.....
.....
..0..
..0..
.0...

?
.000.
0...0
...0.
..0..
.....
..0..
.....

/
....0
...0.
..0..
..0..
.0...
0....
.....
"""

_GLYPHS = _parse_font(_FONT)
