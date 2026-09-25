"""PNG snapshot for chat clients that cannot open the HTML view."""

import struct
import zlib

import pytest

from matters.view_png import (
    ACTIONABLE,
    BLOCKED,
    FOCUS,
    INK,
    PAPER,
    PNG_HEIGHT,
    PNG_WIDTH,
    RESOLVED,
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


def test_a_lone_matter_is_drawn_at_the_centre():
    rgb = rasterize_view(payload_for("lonely"))
    cx = PNG_WIDTH // 2
    cy = (56 + (PNG_HEIGHT - 72)) // 2
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
