"""Portrait geometry: shapes that know their own bounds.

A face is a hundred small shapes -- hair clumps, eyelids, lashes, lips -- and
:func:`core.sdf.polygon` costs one pass *per edge*.  Evaluating every shape over
the whole canvas is what makes a naive portrait take a minute; a
:class:`Shape` pairs a field with the rectangle it can possibly be non-zero in,
so the renderer evaluates it only there.  Measured on a 512 px canvas, windowing
turns a hair clump from 250 ms into 20 ms, which is the difference between a
preview and a contact sheet.

The bounds are kept exact by construction: unions widen them, intersections
narrow them, offsets pad them, and a transform maps the corners.  Every field
below is therefore guaranteed positive outside its ``bbox``, which is the one
invariant :mod:`gen.portraits.brush` relies on.

Two builders do most of the work:

* :func:`lock` -- a spine with a width profile.  A hair clump, an eyelash, an
  eyebrow and a mouth line are the same object pointed at different places.
* :func:`lens` -- two arcs meeting at two corners.  An eye, a lip, a nose wedge.
"""
from __future__ import annotations

import numpy as np

from core import curve
from core import mat as _mat
from core import sdf

__all__ = [
    "Shape",
    "Box",
    "as_shape",
    "bbox_of",
    "pad_bbox",
    "union_bbox",
    "intersect_bbox",
    "window",
    "poly",
    "ellipse",
    "circle",
    "box",
    "union_all",
    "intersection_all",
    "lock",
    "arc",
    "lens",
    "mirror_x",
    "smooth_loop",
    "resample_loop",
]

#: A world-space rectangle ``(x0, y0, x1, y1)``.
Box = "tuple[float, float, float, float]"

_INF = float("inf")


# --------------------------------------------------------------------------
# bounds
# --------------------------------------------------------------------------
def bbox_of(points) -> tuple:
    """Bounding box of a point array shaped ``(n, 2)``."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))


def pad_bbox(b, d: float) -> tuple:
    """Grow a box by ``d`` on every side (``d < 0`` shrinks it)."""
    return (b[0] - d, b[1] - d, b[2] + d, b[3] + d)


def union_bbox(a, b) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def intersect_bbox(a, b) -> tuple:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def _corners(b) -> np.ndarray:
    x0, y0, x1, y1 = b
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def window(canvas, bbox, pad: float | None = None):
    """Sub-grid of ``canvas`` covering ``bbox``.

    Returns ``(X, Y, row0, col0)`` for a rectangular slice of the full meshgrid,
    or ``None`` when the box falls outside the frame.  Two pixels of slack are
    added by default so the anti-aliased edge of a shape is never cut off by its
    own bounds.
    """
    X, Y = canvas.grid()
    if pad is None:
        pad = 2.0 * canvas.dx
    x0, y0, x1, y1 = bbox
    xs = X[0]
    ys = Y[:, 0]
    cols = np.nonzero((xs >= x0 - pad) & (xs <= x1 + pad))[0]
    rows = np.nonzero((ys >= y0 - pad) & (ys <= y1 + pad))[0]
    if cols.size == 0 or rows.size == 0:
        return None
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    return X[r0:r1, c0:c1], Y[r0:r1, c0:c1], r0, c0


# --------------------------------------------------------------------------
# the shape type
# --------------------------------------------------------------------------
class Shape:
    """A distance field together with the box below which it is positive."""

    __slots__ = ("f", "bbox")

    def __init__(self, f, bbox):
        self.f = f
        self.bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    def __call__(self, x, y):
        return self.f(x, y)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        b = self.bbox
        return f"Shape(bbox=({b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}))"

    # -- composition ------------------------------------------------------
    def __or__(self, other) -> "Shape":
        other = as_shape(other)
        return Shape(
            lambda x, y: np.minimum(self.f(x, y), other.f(x, y)),
            union_bbox(self.bbox, other.bbox),
        )

    def __and__(self, other) -> "Shape":
        other = as_shape(other)
        return Shape(
            lambda x, y: np.maximum(self.f(x, y), other.f(x, y)),
            intersect_bbox(self.bbox, other.bbox),
        )

    def __sub__(self, other) -> "Shape":
        other = as_shape(other)
        return Shape(lambda x, y: np.maximum(self.f(x, y), -other.f(x, y)), self.bbox)

    def offset(self, d: float) -> "Shape":
        """Grow (``d > 0``) or shrink (``d < 0``) the shape."""
        return Shape(lambda x, y: self.f(x, y) - d, pad_bbox(self.bbox, abs(d)))

    def outline(self, width: float) -> "Shape":
        """Band of half-width ``width`` centred on the contour."""
        return Shape(lambda x, y: np.abs(self.f(x, y)) - width, pad_bbox(self.bbox, abs(width)))

    def place(self, cx: float = 0.0, cy: float = 0.0, sx: float = 1.0, sy: float | None = None,
              rot: float = 0.0, shear: float = 0.0, shear_y: float = 0.0) -> "Shape":
        """Place the shape on the canvas: scale, shear, rotate, translate.

        A negative ``sx`` mirrors, which is how the left eye and the left brow
        are built from the right ones -- exactly symmetric, no duplicated
        authoring.  Distances stay metric because the field is multiplied by the
        transform's equivalent isotropic scale.
        """
        sy = sx if sy is None else sy
        m = _mat.compose(
            _mat.translate(cx, cy),
            _mat.rotate(rot),
            _mat.shear(shear, shear_y),
            _mat.scale(sx, sy),
        )
        k = _mat.uniform_scale(m)
        mi = _mat.invert(m)
        b = bbox_of(_mat.apply(m, _corners(self.bbox)))

        def g(x, y):
            x2, y2 = _mat.apply_grid(mi, x, y)
            return self.f(x2, y2) * k

        return Shape(g, b)

    def smooth(self, r: float, other) -> "Shape":
        """Union with this shape welding a fillet of radius ``r`` into the seam.

        The soft joint is what makes an anatomical transition read as one body
        rather than two overlapping cut-outs -- the same trick the heraldic
        charges use, and the reason a neck does not look pasted onto a jaw.
        """
        other = as_shape(other)

        def g(x, y):
            return sdf.smooth_union(self.f(x, y), other.f(x, y), r)

        return Shape(g, union_bbox(self.bbox, other.bbox))


def as_shape(o, bbox=None) -> Shape:
    """Coerce a field or a ``Shape`` into a ``Shape`` (infinite bounds if bare)."""
    if isinstance(o, Shape):
        return o
    if callable(o):
        return Shape(o, bbox if bbox is not None else (-_INF, -_INF, _INF, _INF))
    raise TypeError(f"expected a Shape or a field, got {type(o).__name__}")


# --------------------------------------------------------------------------
# primitives (field + known bounds)
# --------------------------------------------------------------------------
def poly(points, resample: int | None = None) -> Shape:
    """Exact field of a polygon, optionally resampled to ``resample`` vertices.

    Even vertex spacing is worth a resample: :func:`core.sdf.polygon` does the
    same work per edge whether the edge is 0.1 or 0.001 units long, so throwing
    away the clustered vertices of a dense spline buys accuracy-preserving speed.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        raise ValueError("a polygon needs at least three points")
    if resample is not None and len(p) > resample:
        p = curve.resample_closed(p, int(resample))
    return Shape(lambda x, y: sdf.polygon(x, y, p), bbox_of(p))


def ellipse(a: float, b: float, cx: float = 0.0, cy: float = 0.0) -> Shape:
    """Axis-aligned ellipse with half-axes ``(a, b)``."""
    return Shape(
        lambda x, y: sdf.ellipse(x, y, a, b, cx, cy),
        (cx - a, cy - b, cx + a, cy + b),
    )


def circle(r: float, cx: float = 0.0, cy: float = 0.0) -> Shape:
    return ellipse(r, r, cx, cy)


def box(hw: float, hh: float, r: float = 0.0, cx: float = 0.0, cy: float = 0.0) -> Shape:
    return Shape(
        lambda x, y: sdf.box(x, y, hw, hh, r, cx, cy),
        (cx - hw, cy - hh, cx + hw, cy + hh),
    )


def union_all(shapes, pad: float = 0.035) -> Shape:
    """Union of many shapes, evaluated as one field with one set of bounds.

    The union's field is the minimum of its parts, and a part is only ever
    *inside* its own bounds -- so each part is evaluated only over the pixels that
    fall in its box, and reported as ``+inf`` everywhere else.  Nothing that reads
    the result cares: coverage and the shading bevel both clip a positive distance
    to zero, so an over-estimate outside a part is indistinguishable from the
    truth.  What it buys is the difference between evaluating thirty hair locks
    over the whole head and evaluating each over its own streak -- which is most
    of a portrait's render time.

    ``pad`` keeps the anti-aliased edge of a part inside its own box, so the
    outermost contour of the union still fades rather than cutting off.
    """
    shapes = [as_shape(s) for s in shapes]
    if not shapes:
        raise ValueError("union_all() needs at least one shape")
    b = shapes[0].bbox
    for s in shapes[1:]:
        b = union_bbox(b, s.bbox)
    boxes = [pad_bbox(s.bbox, pad) for s in shapes]

    def g(x, y):
        out = np.full(np.shape(x), _INF)
        for s, bx in zip(shapes, boxes):
            m = (x >= bx[0]) & (x <= bx[2]) & (y >= bx[1]) & (y <= bx[3])
            if np.ndim(m) == 0:
                # A scalar query (a test, a hairline sample) must not be indexed
                # with a 0-d mask: numpy turns the boolean into an index, not a
                # window, and ``out[m]`` then fails or drops a dimension.
                if bool(m):
                    out = np.minimum(out, s.f(x, y))
                continue
            if not m.any():
                continue
            out[m] = np.minimum(out[m], s.f(x[m], y[m]))
        return out

    return Shape(g, b)


def intersection_all(shapes) -> Shape:
    shapes = [as_shape(s) for s in shapes]
    if not shapes:
        raise ValueError("intersection_all() needs at least one shape")
    b = shapes[0].bbox
    for s in shapes[1:]:
        b = intersect_bbox(b, s.bbox)
    rest = shapes[1:]

    def g(x, y):
        d = shapes[0].f(x, y)
        for s in rest:
            d = np.maximum(d, s.f(x, y))
        return d

    return Shape(g, b)


# --------------------------------------------------------------------------
# curve builders
# --------------------------------------------------------------------------
def arc(a, b, apex: float, bias: float = 0.5, n: int = 24) -> np.ndarray:
    """Quadratic arc from ``a`` to ``b`` bowing ``apex`` towards the left normal.

    The left normal of ``a -> b`` is ``(-dy, dx)``, so a positive ``apex`` bows
    upwards when ``a`` is to the left of ``b`` -- the convention every eyelid and
    smile in this package is written against.  ``bias`` slides the apex along the
    chord, because an anime upper lid peaks past the middle, not on it.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    chord = b - a
    nrm = np.array([-chord[1], chord[0]], dtype=np.float64)
    ln = float(np.linalg.norm(nrm))
    nrm = nrm / ln if ln > 1e-12 else np.array([0.0, 1.0])
    ctrl = a + chord * float(bias) + nrm * (2.0 * float(apex))
    return curve.bezier_quadratic(a, ctrl, b, n)


def lens(inner, outer, up: float, down: float, up_bias: float = 0.45,
         down_bias: float = 0.50, n: int = 22, resample: int | None = 48) -> Shape:
    """Closed almond between two corners: an upper arc and a lower arc."""
    inner = np.asarray(inner, dtype=np.float64)
    outer = np.asarray(outer, dtype=np.float64)
    upper = arc(inner, outer, up, up_bias, n)
    lower = arc(inner, outer, -down, down_bias, n)
    pts = np.vstack([upper, lower[-2:0:-1]])
    return poly(pts, resample=resample)


def lock(spine, w0: float = 0.10, w1: float = 0.0, bulge: float = 0.0,
         per_segment: int = 6, resample: int | None = 56) -> Shape:
    """Tapered lock along a smooth spine through ``spine``.

    Width runs linearly from ``w0`` at the root to ``w1`` at the tip, with an
    optional ``bulge`` added as a half-sine in between.  ``w1 = 0`` ends in a
    point, which is what makes a hair clump a clump and an eyelash a lash; a
    non-zero ``w1`` can then be rounded off by :meth:`Shape.offset`.
    """
    pts = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError("a lock needs at least two spine points")
    sp = curve.catmull_rom(pts, per_segment=per_segment)
    m = len(sp)
    t = np.linspace(0.0, 1.0, m)
    w = np.maximum(w0 + (w1 - w0) * t + bulge * np.sin(np.pi * t), 1e-6)
    d = np.gradient(sp, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    left = sp + nrm * (w * 0.5)[:, None]
    right = sp - nrm * (w * 0.5)[:, None]
    return poly(np.vstack([left, right[::-1]]), resample=resample)


def mirror_x(points) -> np.ndarray:
    """Mirror a point array through the vertical axis."""
    p = np.asarray(points, dtype=np.float64).copy()
    p[:, 0] *= -1.0
    return p


def smooth_loop(points, per_segment: int = 6, resample: int | None = None) -> np.ndarray:
    """Close a coarse control polygon into a smooth loop of sample points."""
    sp = curve.catmull_rom(points, per_segment=per_segment, closed=True)
    return curve.resample_closed(sp, resample) if resample else sp


def resample_loop(points, n: int = 48) -> np.ndarray:
    return curve.resample_closed(points, int(n))
