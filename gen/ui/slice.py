"""The nine-slice contract: insets, the nine rectangles, and the resample.

An asset drawn at one size and displayed at another is the normal case in a 2-D
game, and a nine-slice is the only resampling rule that survives it without
mangling the art: keep the four corners, stretch the four edges along one axis,
stretch the middle along both.  Everything in this module exists to make that
rule precise and testable.

Two consequences worth stating, because both are easy to get wrong:

* The corners are **never resampled**.  Not "resampled to the same size" -- the
  blocks are not passed through the resampler at all, so they come out
  bit-identical, which is what the tests assert.
* A nine-slice can always grow, but it cannot shrink below ``l + r`` by
  ``t + b``: below that the corners would have to overlap.  :func:`minimum` says
  how small the asset can be displayed, and :func:`stretch` refuses to guess.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from core import palette
from core.canvas import Canvas

__all__ = [
    "insets",
    "cuts",
    "minimum",
    "stretch_pixels",
    "stretch",
    "from_rgba",
]


def insets(panel) -> tuple:
    """The panel's four corner insets in *output* pixels: ``(l, t, r, b)``.

    Design pixels become pixels here and nowhere else, because this is the number
    an engine has to be told: it is the only part of the design that is not
    allowed to scale.  Four numbers rather than one, since a window with a title
    bar has a deeper top than sides.
    """
    return tuple(int(round(v * panel.scale)) for v in panel.insets)


def minimum(ins) -> tuple:
    """The smallest size that still has room for both corners."""
    l, t, r, b = ins
    return (l + r, t + b)


def cuts(size, ins) -> list:
    """The nine rectangles, as ``(x0, y0, x1, y1)`` pixel bounds, row-major.

    Index 0 is the top-left corner, 4 the middle, 8 the bottom-right.  The nine
    rectangles tile the image exactly -- no gap, no overlap, no rounding -- which
    is the arithmetic the whole convention leans on.
    """
    w, h = int(size[0]), int(size[1])
    l, t, r, b = (int(v) for v in ins)
    if l + r > w or t + b > h:
        raise ValueError(f"nine-slice insets {ins} do not fit in a {w}x{h} image")
    if l < 0 or t < 0 or r < 0 or b < 0:
        raise ValueError("nine-slice insets cannot be negative")
    xs = (0, l, w - r, w)
    ys = (0, t, h - b, h)
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(3) for i in range(3)]


def _resize(block: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resample one block, and hand it straight back when nothing would change.

    The early return is not an optimisation.  It is the guarantee that a corner
    block leaves this module byte for byte as it arrived.
    """
    if block.shape[1] == w and block.shape[0] == h:
        return block
    img = Image.fromarray(np.ascontiguousarray(block), "RGBA")
    return np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.uint8)


def stretch_pixels(rgba, size, ins) -> np.ndarray:
    """Resample an 8-bit RGBA array by the nine-slice rule."""
    rgba = np.asarray(rgba, dtype=np.uint8)
    src = (rgba.shape[1], rgba.shape[0])
    dst_rects = cuts(size, ins)
    src_rects = cuts(src, ins)
    out = np.zeros((int(size[1]), int(size[0]), 4), dtype=np.uint8)
    for (sx0, sy0, sx1, sy1), (dx0, dy0, dx1, dy1) in zip(src_rects, dst_rects):
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        block = _resize(rgba[sy0:sy1, sx0:sx1], dx1 - dx0, dy1 - dy0)
        out[dy0:dy1, dx0:dx1] = block
    return out


def stretch(canvas: Canvas, size, ins) -> Canvas:
    """Nine-slice one rendered asset to a new pixel size.

    The resample happens in 8-bit sRGB -- in the PNG's own space -- because that
    is the asset.  Compositing wants linear light, but *stretching* is what the
    engine will do to the file, so doing it anywhere else would be modelling a
    step that never happens.  The world frame keeps its scale, so the design is
    genuinely wider rather than zoomed.
    """
    dst = (int(size[0]), int(size[1]))
    lo = minimum(ins)
    if dst[0] < lo[0] or dst[1] < lo[1]:
        raise ValueError(
            f"cannot nine-slice to {dst}: the corners alone need at least {lo}"
        )
    return from_rgba(stretch_pixels(canvas.to_uint8(), dst, ins), canvas.scale, canvas.center)


def from_rgba(rgba, scale: float, center=(0.0, 0.0)) -> Canvas:
    """A :class:`Canvas` from an 8-bit RGBA array, un-premultiplying nothing.

    Straight alpha is what both PNG and :meth:`Canvas.over` use, so the round trip
    is lossless apart from the 8-bit quantisation it was always going to have.
    """
    rgba = np.asarray(rgba, dtype=np.float64)
    h, w = rgba.shape[:2]
    c = Canvas(w, h, scale=scale, center=center)
    c.rgb = palette.srgb_to_linear(rgba[..., :3] / 255.0)
    c.a = rgba[..., 3] / 255.0
    return c
