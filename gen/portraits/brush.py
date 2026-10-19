"""Painting: how pixels get written, and how a shape is evaluated only once.

Three ideas carry the whole renderer.

**Paint on the shape's own window.**  :func:`core.sdf.polygon` costs one pass per
edge, so a shape evaluated over the whole canvas pays for pixels it cannot reach.
Every shape here carries its bounds and is evaluated only inside them
(:func:`gen.portraits.shape.window`).  On a 512 px canvas that is the difference
between 250 ms and 20 ms for a hair clump.

**Filling, shading and inking one shape is one evaluation.**  A union of ten hair
clumps filled, cel-shaded and outlined would otherwise evaluate the union three
times -- and thirty polygons.  :class:`Surface` evaluates it once and derives the
ink band and the shading bands from the same distance array.

**A ``Shape`` clip is a field intersection; an array clip is a multiply.**  The
first keeps an anti-aliased edge crisp; the second lets a mask computed once (the
whole fringe, say) be reused by a dozen small shapes for free.  The soft-edge
blur always works in *pixels*, because a blush that scaled with the canvas would
stop being a blush at icon size.
"""
from __future__ import annotations

import numpy as np

from core import palette, render
from core.render import LIGHT

from .shape import Shape, as_shape, intersect_bbox, window

__all__ = [
    "LIGHT", "paint", "paint_lin", "paint_cov", "multiply", "mask", "shift_cov",
    "Surface",
]




# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def mask(canvas, shape) -> np.ndarray:
    """Full-frame coverage array of a shape, for reuse as a cheap clip."""
    out = np.zeros((canvas.h, canvas.w), dtype=np.float64)
    shape = as_shape(shape)
    win = window(canvas, shape.bbox, pad=0.0)
    if win is None:
        return out
    Xs, Ys, r0, c0 = win
    out[r0 : r0 + Xs.shape[0], c0 : c0 + Xs.shape[1]] = render.coverage(shape.f(Xs, Ys), canvas.dx)
    return out


def shift_cov(cov, dy: float, scale: float) -> np.ndarray:
    """Shift a coverage array by a world-space vertical offset (``dy`` is up)."""
    return render.translate_array(np.asarray(cov, dtype=np.float64), 0.0, -float(dy) * scale)


def paint_cov(canvas, cov, colour, soft: float = 0.0, alpha: float = 1.0):
    """Composite a full-frame coverage array in a flat colour."""
    a = np.asarray(cov, dtype=np.float64)
    if soft > 0.0:
        a = render.gaussian_blur(a, soft)
    if alpha != 1.0:
        a = a * float(alpha)
    canvas.over(palette.srgb_to_linear(colour), a)


def _tile(canvas, rgb, a, r0, c0):
    canvas.over(rgb, a, c0, r0)


# --------------------------------------------------------------------------
# one-shot painting
# --------------------------------------------------------------------------
def paint(canvas, shape, colour=None, clip=None, alpha: float = 1.0, soft: float = 0.0,
          ramp=None, premultiplied: bool = False):
    """Composite ``shape`` onto ``canvas``.

    ``colour`` is an sRGB triple, or ``None`` when ``ramp`` is a callable
    ``(X, Y) -> (h, w, 3)`` sRGB evaluated on the shape's own window.  ``clip`` is
    a ``Shape`` (exact intersection), a full-frame coverage array
    (multiplication), or ``None``.  ``soft`` blurs the coverage by that many
    *pixels*.
    """
    shape = as_shape(shape)
    bbox = shape.bbox
    if isinstance(clip, Shape):
        bbox = intersect_bbox(bbox, clip.bbox)
    win = window(canvas, bbox)
    if win is None:
        return
    Xs, Ys, r0, c0 = win
    d = shape.f(Xs, Ys)
    if isinstance(clip, Shape):
        d = np.maximum(d, clip.f(Xs, Ys))
    a = render.coverage(d, canvas.dx)
    if isinstance(clip, np.ndarray):
        a = a * clip[r0 : r0 + Xs.shape[0], c0 : c0 + Xs.shape[1]]
    if soft > 0.0:
        a = render.gaussian_blur(a, soft)
    if alpha != 1.0:
        a = a * float(alpha)
    if ramp is not None:
        rgb = palette.srgb_to_linear(ramp(Xs, Ys))
    elif premultiplied:
        rgb = np.asarray(colour, dtype=np.float64)
    else:
        rgb = palette.srgb_to_linear(colour)
    canvas.over(rgb, a, c0, r0)


def paint_lin(canvas, shape, rgb_linear, clip=None, alpha: float = 1.0, soft: float = 0.0):
    """As :func:`paint`, with an already-linear-light colour."""
    paint(canvas, shape, np.asarray(rgb_linear, dtype=np.float64), clip=clip, alpha=alpha,
          soft=soft, premultiplied=True)


def multiply(canvas, shape, factor, clip=None, alpha: float = 1.0):
    """Scale the lit pixels under ``shape`` -- bright for a shine, dark for a shade.

    Multiplying in linear light is what makes a hair shine read as a reflection
    rather than a lighter coat of paint: it brightens what is already there
    (strand shadows included) instead of replacing it.
    """
    shape = as_shape(shape)
    bbox = shape.bbox
    if isinstance(clip, Shape):
        bbox = intersect_bbox(bbox, clip.bbox)
    win = window(canvas, bbox)
    if win is None:
        return
    Xs, Ys, r0, c0 = win
    d = shape.f(Xs, Ys)
    if isinstance(clip, Shape):
        d = np.maximum(d, clip.f(Xs, Ys))
    a = render.coverage(d, canvas.dx)
    if isinstance(clip, np.ndarray):
        a = a * clip[r0 : r0 + Xs.shape[0], c0 : c0 + Xs.shape[1]]
    if alpha != 1.0:
        a = a * float(alpha)
    h, w = a.shape
    f = np.asarray(factor, dtype=np.float64)
    f = f if f.ndim == 3 else np.full((h, w, 1), float(f))
    tile = canvas.rgb[r0 : r0 + h, c0 : c0 + w]
    canvas.rgb[r0 : r0 + h, c0 : c0 + w] = np.clip(
        tile * (1.0 + a[..., None] * (f - 1.0)), 0.0, 1.0
    )


# --------------------------------------------------------------------------
# the multi-pass surface
# --------------------------------------------------------------------------
class Surface:
    """A shape's distance field, evaluated once and then painted many ways.

    Filling a union of ten hair clumps, cel-shading it and inking its contour is
    three passes over the same geometry.  A ``Surface`` pays for the geometry
    once: :meth:`paint`, :meth:`stroke` and :meth:`bands` all derive their masks
    from the stored array, and the ink band is arithmetic on it
    (``|d + inset| - w/2``), not a new distance field.
    """

    __slots__ = ("d", "x", "y", "rows", "cols", "dx", "scale", "y_up", "empty")

    def __init__(self, canvas, shape, clip=None, pad: float | None = None):
        s = as_shape(shape)
        bbox = s.bbox
        if isinstance(clip, Shape):
            bbox = intersect_bbox(bbox, clip.bbox)
        win = window(canvas, bbox, pad=pad)
        self.empty = win is None
        if self.empty:
            self.d = self.x = self.y = None
            return
        Xs, Ys, r0, c0 = win
        d = s.f(Xs, Ys)
        if isinstance(clip, Shape):
            d = np.maximum(d, clip.f(Xs, Ys))
        self.d = d
        self.x = Xs
        self.y = Ys
        self.rows = r0
        self.cols = c0
        self.dx = canvas.dx
        self.scale = canvas.scale
        self.y_up = canvas.y_up

    # -- derivations -------------------------------------------------------
    def coverage(self) -> np.ndarray | None:
        return None if self.empty else render.coverage(self.d, self.dx)

    def full_coverage(self, canvas) -> np.ndarray:
        """The coverage placed back into a full-frame array (for cheap reuse)."""
        out = np.zeros((canvas.h, canvas.w), dtype=np.float64)
        if not self.empty:
            h, w = self.d.shape
            out[self.rows : self.rows + h, self.cols : self.cols + w] = render.coverage(self.d, self.dx)
        return out

    def _emit(self, canvas, a, colour, ramp, soft, alpha):
        if soft > 0.0:
            a = render.gaussian_blur(a, soft)
        if alpha != 1.0:
            a = a * float(alpha)
        rgb = palette.srgb_to_linear(ramp(self.x, self.y)) if ramp is not None \
            else palette.srgb_to_linear(colour)
        canvas.over(rgb, a, self.cols, self.rows)

    # -- passes ------------------------------------------------------------
    def paint(self, canvas, colour=None, alpha: float = 1.0, soft: float = 0.0,
              ramp=None, mask=None):
        if self.empty:
            return
        a = self.coverage()
        if mask is not None:
            a = a * mask
        self._emit(canvas, a, colour, ramp, soft, alpha)

    def stroke(self, canvas, width: float, colour, inset: float = 0.0, alpha: float = 1.0,
               taper=None, mask=None, soft: float = 0.0):
        """Ink the contour, clipped to the shape itself (no spilling to the outside)."""
        if self.empty:
            return
        w = width if taper is None else width * taper(self.x, self.y)
        band = np.abs(self.d + inset) - 0.5 * np.maximum(w, 1e-6)
        a = render.coverage(band, self.dx) * self.coverage()
        if mask is not None:
            a = a * mask
        self._emit(canvas, a, colour, None, soft, alpha)

    def bands(self, canvas, colours, edges, volume=None, light=LIGHT, bevel: float = 0.05,
              soft: float = 0.7, alpha: float = 1.0):
        """Quantised Lambert bands over a volume: the cel pass, from the stored field."""
        if self.empty:
            return
        h = np.clip(-self.d, 0.0, bevel)
        if volume is not None:
            h = h + volume(self.x, self.y)
        n = render.height_normals(h, self.dx, self.y_up)
        L = np.array((light[0], light[1], 0.60), dtype=np.float64)
        L = L / max(float(np.linalg.norm(L)), 1e-12)
        lam = np.clip(n @ L, 0.0, 1.0)
        base = self.coverage()
        for i in range(min(len(colours), len(edges)) - 1, -1, -1):
            m = ((lam < float(edges[i])) & (self.d < 0.0)).astype(np.float64)
            if soft > 0.0:
                m = render.gaussian_blur(m, soft)
            self._emit(canvas, base * m, colours[i], None, 0.0, alpha)
