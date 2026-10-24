"""PNG snapshot for chat clients that cannot open the HTML view."""

import struct
import zlib

import pytest

from matters.view import build_view_payload
from matters.view_png import (
    ACTIONABLE,
    BLOCKED,
    FOCUS,
    INK,
    PAPER,
    PNG_HEIGHT,
    PNG_WIDTH,
    RESOLVED,
    _LEGEND_HEIGHT,
    _LabelOcclusion,
    _arrow,
    _edge_endpoints,
    _layout_labels,
    _node_radius,
    _place,
    _plot_box,
    _point_in_rect,
    _segment_hits_rect,
    _stroke,
    _wrap_label,
    encode_png,
    rasterize_view,
    render_view_png,
)
from test_view import payload_for


def decode_png(data):
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    width = height = None
    idat = b""
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if tag == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", chunk[:10])
            assert (depth, color_type) == (8, 2)
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    raw = zlib.decompress(idat)
    stride = width * 3
    rows = []
    cursor = 0
    for _ in range(height):
        assert raw[cursor] == 0
        cursor += 1
        rows.append(raw[cursor : cursor + stride])
        cursor += stride
    assert cursor == len(raw)
    return width, height, b"".join(rows)


def color_set(rgb):
    return {tuple(rgb[index : index + 3]) for index in range(0, len(rgb), 3)}


def test_the_png_is_a_real_image_of_the_slice():
    png = render_view_png(payload_for("c"))
    width, height, rgb = decode_png(png)

    assert (width, height) == (PNG_WIDTH, PNG_HEIGHT)
    present = color_set(rgb)
    # a is resolved, b is actionable, c and d are blocked, c wears the focus ring.
    assert {PAPER, INK, ACTIONABLE, BLOCKED, RESOLVED, FOCUS} <= present


def test_short_titles_wrap_to_two_lines_instead_of_truncating():
    assert _wrap_label("launch a small newsletter") == ("launch a small", "newsletter")
    assert _wrap_label("pick a platform") == ("pick a platform",)
    wrapped = _wrap_label("one two three four five six seven eight nine ten")
    assert len(wrapped) == 2
    assert wrapped[-1].endswith("..")


def test_newsletter_labels_clear_the_arrows_and_the_legend():
    payload = build_view_payload(
        "launch_a_small_newsletter",
        {
            "launch_a_small_newsletter",
            "pick_a_platform",
            "write_issue_one",
        },
        {
            "launch_a_small_newsletter": [
                {"label": "first issue sent to 20 subscribers", "truth": False}
            ],
            "pick_a_platform": [{"label": "Resolved: pick a platform", "truth": True}],
            "write_issue_one": [{"label": "Resolved: write issue one", "truth": False}],
        },
        {
            ("pick_a_platform", "launch_a_small_newsletter"),
            ("write_issue_one", "launch_a_small_newsletter"),
        },
    )
    placed = _place(payload["nodes"], PNG_WIDTH, PNG_HEIGHT)
    labels = _layout_labels(
        payload["nodes"], placed, payload["edges"], payload["matter"], PNG_WIDTH, PNG_HEIGHT
    )
    legend_top = PNG_HEIGHT - _LEGEND_HEIGHT
    focus = payload["matter"]

    assert labels[focus]["lines"] == ("launch a small", "newsletter")
    for layout in labels.values():
        assert layout["rect"][3] <= legend_top
        assert ".." not in " ".join(layout["lines"])

    for edge in payload["edges"]:
        source, target = edge["source"], edge["target"]
        x0, y0, x1, y1 = _edge_endpoints(
            placed[source],
            placed[target],
            _node_radius(source, focus),
            _node_radius(target, focus),
        )
        target_x, target_y = placed[target]
        distance = ((x1 - target_x) ** 2 + (y1 - target_y) ** 2) ** 0.5
        assert abs(distance - (_node_radius(target, focus) + 1)) < 0.2
        for layout in labels.values():
            assert not _point_in_rect(x1, y1, layout["rect"])
            assert not _segment_hits_rect(x0, y0, x1, y1, layout["rect"])

    # The picture itself still draws the focus ring and the three status colours.
    present = color_set(rasterize_view(payload))
    assert {INK, ACTIONABLE, BLOCKED, RESOLVED, FOCUS} <= present


def test_a_lone_matter_is_drawn_at_the_centre():
    rgb = rasterize_view(payload_for("lonely"))
    left, top, right, bottom = _plot_box(PNG_WIDTH, PNG_HEIGHT)
    cx = int((left + right) / 2)
    cy = int((top + bottom) / 2)
    found = False
    for y in range(cy - 16, cy + 16):
        for x in range(cx - 16, cx + 16):
            start = (y * PNG_WIDTH + x) * 3
            if tuple(rgb[start : start + 3]) != PAPER:
                found = True
                break
    assert found


def test_a_cycle_still_draws_structure_without_status_colours():
    png = render_view_png(
        payload_for("c", dependencies={("a", "b"), ("b", "c"), ("c", "a")})
    )
    _, _, rgb = decode_png(png)
    present = color_set(rgb)

    assert FOCUS in present
    assert RESOLVED in present
    assert ACTIONABLE not in present
    assert BLOCKED not in present


def test_encode_png_rejects_a_buffer_of_the_wrong_length():
    with pytest.raises(ValueError, match="does not match"):
        encode_png(2, 2, b"\x00" * 3)


@pytest.mark.parametrize(
    "segment, rect, expected",
    [
        ((-2, 1, 4, 1), (0, 0, 2, 2), True),
        ((1, 4, 1, -2), (0, 0, 2, 2), True),
        ((-2, 0, 4, 0), (0, 0, 2, 2), True),
        ((-2, -2, 0, 0), (0, 0, 2, 2), True),
        ((-2, -0.01, 4, -0.01), (0, 0, 2, 2), False),
        ((1, 1, 1, 1), (0, 0, 2, 2), True),
        ((3, 3, 3, 3), (0, 0, 2, 2), False),
        # A narrow intersection between raster samples must still be detected.
        ((0, 0, 10, 0), (0.4, -0.1, 0.6, 0.1), True),
    ],
)
def test_segment_rectangle_intersection_includes_tangencies_and_subpixel_crossings(
    segment, rect, expected
):
    assert _segment_hits_rect(*segment, rect) is expected
    assert _segment_hits_rect(*segment[2:], *segment[:2], rect) is expected


def test_label_occlusion_preserves_fractional_and_half_open_rectangle_boundaries():
    rectangles = [(-40.5, -32, 70.25, 66), (90.1, 12.5, 144, 32.5)]
    indexed = _LabelOcclusion(rectangles)
    for x in [-41, -40.5, -32, -0.1, 0, 31.999, 32, 64, 70.25, 90.1, 143.999, 144]:
        for y in [-33, -32, -0.1, 0, 12.5, 31.999, 32, 32.5, 65.999, 66]:
            assert indexed.contains(x, y) is any(
                _point_in_rect(x, y, rect) for rect in rectangles
            )


def test_indexed_occlusion_draws_the_same_strokes_and_arrows_as_rectangle_scanning():
    width, height = 240, 180
    rectangles = [(50.25, 40.5, 110.75, 120.25), (140, 70, 200, 95)]
    original = bytearray(bytes(PAPER) * width * height)
    indexed = bytearray(original)
    for pixels, avoid in ((original, rectangles), (indexed, _LabelOcclusion(rectangles))):
        _stroke(pixels, width, height, 10.2, 12.4, 230.9, 160.3, INK, 1.6, avoid)
        _arrow(pixels, width, height, 20.3, 30.7, 160.2, 90.8, INK, avoid)
    assert indexed == original
