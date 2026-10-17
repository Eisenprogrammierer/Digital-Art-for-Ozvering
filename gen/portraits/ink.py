"""Line art: a stroke that rides a distance field and thickens where it likes.

Anime is a line-art medium first and a shading medium second.  The drawing reads
from its contour, and the contour is not a uniform wire: it is heavy along the
jaw and the outer silhouette, light inside the face, and it tapers to a point at
an eyelash and at the tip of a brow.  Because every shape here *is* a distance
field, its contour is the zero set and a stroke is a band around zero -- so the
line redraws itself when the head gets wider, with no authoring per pose.

Line colour matters as much as line weight.  Pure black flattens a drawing; a
line that is a very dark, slightly saturated relative of the colour it outlines
keeps the skin warm and the hair cool.  :func:`line_colour` derives that from the
local hue.
"""
from __future__ import annotations

import numpy as np

from core import palette

from . import brush
from .shape import Shape, as_shape, pad_bbox

__all__ = ["INK", "stroke", "paint", "line_colour", "shade_line"]

#: Default ink for lashes and pupils: near-black with a violet cast.
INK = (0.105, 0.090, 0.115)


def stroke(shape: Shape, width: float, inset: float = 0.0, taper=None) -> Shape:
    """Band of width ``width`` around the contour of ``shape``.

    ``inset`` slides the band inwards, so a stroke clipped to its *own*
    silhouette still draws its whole thickness instead of losing the outer half
    to the clip -- the usual way to ink an outline that must stay inside the
    shape.

    ``taper`` is a callable ``(x, y) -> factor`` modulating the width across the
    canvas, which is how the outer contour gets heavy at the jaw and thin at the
    temple.  Accept factors up to about 2: the bounds are padded for that.
    """
    shape = as_shape(shape)

    def f(x, y):
        d = shape.f(x, y) + inset
        w = width if taper is None else width * taper(x, y)
        return np.abs(d) - 0.5 * np.maximum(w, 1e-6)

    return Shape(f, pad_bbox(shape.bbox, abs(inset) + 2.2 * abs(width)))


def paint(canvas, shape, width: float, colour=INK, inset: float = 0.0, taper=None,
          clip=None, alpha: float = 1.0):
    """Draw the outline of ``shape`` with the ink conventions above."""
    brush.paint(canvas, stroke(shape, width, inset, taper), colour, clip=clip, alpha=alpha)


def line_colour(base, depth: float = 0.60, chroma: float = 1.30, hue: float = 0.0,
                floor: float = 0.075):
    """A dark, saturated relative of ``base``: the coloured-line-art trick.

    ``depth`` hugely outruns what a shadow would take (a shadow is 0.08-0.12 of
    OKLCH lightness; a line is 0.5-0.7), which is what keeps a line dark enough to
    read as a line while still carrying the hue underneath.
    """
    L, C, h = palette.srgb_to_oklch(base)
    return palette.oklch_to_srgb(
        (max(float(L) - depth, floor), min(float(C) * chroma, 0.16), (float(h) + hue) % 1.0)
    )


def shade_line(base, mul: float = 0.45, floor: float = 0.07, chroma: float = 1.15,
               hue: float = -0.020, cap: float = 0.14):
    """A line colour keeping its distance *below* ``base``, whatever the base is.

    :func:`line_colour` subtracts a fixed amount of lightness, which turns mid-tone
    hair black.  A line drawn on hair should instead be a fraction of the base's
    own lightness, so blonde hair gets a mid-brown line and black hair a charcoal
    one -- the same rule an illustrator applies by picking a darker shade rather
    than a different colour.
    """
    L, C, h = palette.srgb_to_oklch(base)
    return palette.oklch_to_srgb(
        (max(float(L) * mul, floor), min(float(C) * chroma, cap), (float(h) + hue) % 1.0)
    )
