"""Signed distance fields: analytic primitives that compose into any shape.

Convention
----------
A field is a plain callable ``f(x, y) -> ndarray`` that accepts broadcastable
coordinate arrays (or the full meshgrids produced by
:meth:`core.canvas.Canvas.grid`) and returns the **Euclidean distance in world
units**: negative inside the shape, positive outside, zero on the boundary.

Because fields are ordinary functions, shapes are composed with operators,
transformed with affine matrices, and evaluated at any resolution.  There is no
resolution-independent geometry to keep in sync -- the same field feeds a 24 px
icon and a 512 px illustration.
"""
from __future__ import annotations

import numpy as np

from . import mat as _mat

__all__ = [
    "circle",
    "box",
    "segment",
    "ring",
    "polygon",
    "regular_polygon",
    "star",
    "rounded",
    "outline",
    "offset",
    "union",
    "intersection",
    "difference",
    "smooth_union",
    "smooth_intersection",
    "smooth_difference",
    "transformed",
    "moved",
    "rotated",
    "scaled",
    "mirrored",
    "repeated",
]


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def circle(x, y, r: float = 1.0, cx: float = 0.0, cy: float = 0.0):
    """Disc of radius ``r`` centred at ``(cx, cy)``."""
    return np.hypot(x - cx, y - cy) - r


def box(x, y, hw: float = 1.0, hh: float = 1.0, r: float = 0.0, cx: float = 0.0, cy: float = 0.0):
    """Axis-aligned box with half-extents ``(hw, hh)`` and corner radius ``r``.

    Exact for the rounded variant: the corner radius is subtracted from the
    half-extents first, then re-added outside the clamp.
    """
    qx = np.abs(x - cx) - hw + r
    qy = np.abs(y - cy) - hh + r
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - r


def segment(x, y, a, b, r: float = 0.0):
    """Capsule from ``a=(ax, ay)`` to ``b=(bx, by)`` with radius ``r``."""
    ax, ay = a
    bx, by = b
    px, py = x - ax, y - ay
    ex, ey = bx - ax, by - ay
    denom = ex * ex + ey * ey
    if denom == 0.0:
        h = 0.0
    else:
        h = np.clip((px * ex + py * ey) / denom, 0.0, 1.0)
    return np.hypot(px - ex * h, py - ey * h) - r


def ring(x, y, r: float = 1.0, thickness: float = 0.2, cx: float = 0.0, cy: float = 0.0):
    """Annulus: the region between ``r - thickness`` and ``r + thickness``."""
    return np.abs(circle(x, y, r, cx, cy)) - thickness


def polygon(x, y, pts):
    """Exact signed distance to an arbitrary simple polygon.

    Vectorised port of the classic per-edge Voronoi/crossing test: each edge
    contributes its closest-point distance, and the sign is toggled by the
    ray-crossing condition.  Correct for concave and convex outlines alike.

    The crossing parity is a property of the geometry, not of the vertex order,
    so clockwise and counter-clockwise input give the same sign -- no winding
    normalisation is needed.  (Self-intersecting input follows the even-odd
    rule, which is the sane degenerate behaviour.)
    """
    pts = np.asarray(pts, dtype=np.float64)
    px = np.asarray(x, dtype=np.float64)
    py = np.asarray(y, dtype=np.float64)
    n = len(pts)

    d2 = (px - pts[0, 0]) ** 2 + (py - pts[0, 1]) ** 2
    sign = np.ones(np.broadcast_shapes(px.shape, py.shape), dtype=np.float64)

    for i in range(n):
        j = (i - 1) % n
        vx, vy = pts[i]
        wx, wy = pts[j]
        ex, ey = wx - vx, wy - vy
        rx, ry = px - vx, py - vy

        denom = ex * ex + ey * ey
        h = 0.0 if denom == 0.0 else np.clip((rx * ex + ry * ey) / denom, 0.0, 1.0)
        bx, by = rx - ex * h, ry - ey * h
        d2 = np.minimum(d2, bx * bx + by * by)

        above_i = py >= vy
        below_j = py < wy
        right_of_edge = ex * ry > ey * rx
        flip = (above_i & below_j & right_of_edge) | (~above_i & ~below_j & ~right_of_edge)
        sign = np.where(flip, -sign, sign)

    return sign * np.sqrt(d2)


def regular_polygon(x, y, n: int = 6, circumradius: float = 1.0, rotation: float = 0.0):
    """Regular ``n``-gon inscribed in a circle of radius ``circumradius``."""
    ang = np.radians(rotation) + np.pi / 2.0
    pts = [
        (
            circumradius * np.cos(ang + 2.0 * np.pi * i / n),
            circumradius * np.sin(ang + 2.0 * np.pi * i / n),
        )
        for i in range(n)
    ]
    return polygon(x, y, pts)


def star(x, y, n: int = 5, outer: float = 1.0, inner: float = 0.45, rotation: float = 0.0):
    """``n``-pointed star alternating between ``outer`` and ``inner`` radius."""
    ang = np.radians(rotation) + np.pi / 2.0
    pts = []
    for i in range(2 * n):
        r = outer if i % 2 == 0 else inner
        a = ang + np.pi * i / n
        pts.append((r * np.cos(a), r * np.sin(a)))
    return polygon(x, y, pts)


# --------------------------------------------------------------------------
# operators (all take and return fields)
# --------------------------------------------------------------------------
def rounded(field, r: float):
    """Round every corner of ``field`` by ``r`` (offset outward by ``r``)."""
    return lambda x, y: field(x, y) - r


def offset(field, d: float):
    """Alias of :func:`rounded`: grow (d>0) or shrink (d<0) the shape."""
    return lambda x, y: field(x, y) - d


def outline(field, width: float):
    """Stroke of thickness ``2*width`` centred on the boundary of ``field``."""
    return lambda x, y: np.abs(field(x, y)) - width


def union(d1, d2):
    """Hard union: the closest of the two surfaces."""
    return np.minimum(d1, d2)


def intersection(d1, d2):
    """Hard intersection: the farthest of the two surfaces."""
    return np.maximum(d1, d2)


def difference(d1, d2):
    """Hard subtraction of ``d2`` from ``d1``."""
    return np.maximum(d1, -d2)


def smooth_union(d1, d2, k: float = 0.1):
    """Union with a fillet of radius ``k`` welded into the seam.

    The polynomial ``k*h*(1-h)`` shaves ``k/4`` off the result near the joint,
    which is exactly the material a fillet needs.  ``k=0`` degrades to the hard
    union.
    """
    if k <= 0.0:
        return np.minimum(d1, d2)
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)


def smooth_intersection(d1, d2, k: float = 0.1):
    """Intersection with a concave fillet of radius ``k``."""
    if k <= 0.0:
        return np.maximum(d1, d2)
    h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h)


def smooth_difference(d1, d2, k: float = 0.1):
    """Subtraction whose inner corners are filleted by ``k``."""
    return smooth_intersection(d1, -d2, k)


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def transformed(field, m):
    """Apply an affine matrix ``m`` to the *shape* (coordinates are inverted)."""
    mi = _mat.invert(m)
    s = _mat.uniform_scale(m)

    def g(x, y):
        x2, y2 = _mat.apply_grid(mi, x, y)
        return field(x2, y2) * s

    return g


def moved(field, tx: float, ty: float):
    """Translate the shape."""
    return lambda x, y: field(x - tx, y - ty)


def rotated(field, deg: float):
    """Rotate the shape counter-clockwise by ``deg`` degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return lambda x, y: field(c * x + s * y, -s * x + c * y)


def scaled(field, sx: float, sy: float | None = None):
    """Scale the shape; distances stay metric for uniform scaling."""
    sy = sx if sy is None else sy
    k = min(abs(sx), abs(sy)) or 1.0
    return lambda x, y: field(x / sx, y / sy) * k


def mirrored(field, axis: str = "x"):
    """Make the shape symmetric by folding one coordinate through zero."""
    if axis == "x":
        return lambda x, y: field(np.abs(x), y)
    if axis == "y":
        return lambda x, y: field(x, np.abs(y))
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def repeated(field, px: float, py: float | None = None):
    """Infinite lattice of the shape, one copy per ``(px, py)`` cell."""
    py = px if py is None else py

    def g(x, y):
        return field(np.mod(x + px / 2.0, px) - px / 2.0, np.mod(y + py / 2.0, py) - py / 2.0)

    return g
