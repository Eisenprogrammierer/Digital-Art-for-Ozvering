"""Painting: paint lists into pixels, and finishing the shield.

Everything a blazon paints -- the field, the ordinary, the charges -- is a
distance field paired with a tincture, so one loop paints all of it and the
anti-aliasing comes from :func:`core.render.coverage`.  The only per-kind
difference is whether a layer is painted flat (heraldry is flat by tradition) or
embossed (:func:`core.render.shade`, for game-friendly mobile charges).

The shield itself is finished with a rim light and a soft shadow, both derived
from the silhouette's own distance field -- no authoring, and the same light
direction as every other asset in the project.
"""
from __future__ import annotations

import numpy as np

from core import palette, render
from core.canvas import Canvas

from . import tinctures

#: The single light direction shared by every generated asset, in world units.
LIGHT = (-0.45, 0.65)

__all__ = [
    "LIGHT",
    "field_canvas",
    "mask_alpha",
    "paint_layers",
    "paint_flat",
    "paint_shadow",
    "paint_rim",
]


def field_canvas(size, hw: float, hh: float, margin: float = 0.06, y_up: bool = True, background=None) -> Canvas:
    """A canvas whose world frame fits the box ``[-hw, hw] x [-hh, hh]``.

    Because the frame is derived from the field's extent, the same blazon can be
    rendered onto a heater shield or a 3:2 flag without touching the geometry.
    """
    if isinstance(size, (tuple, list)):
        w, h = int(size[0]), int(size[1])
    else:
        w = h = int(size)
    usable = max(1.0 - 2.0 * margin, 0.05)
    scale = min(w / (2.0 * hw), h / (2.0 * hh)) * usable
    return Canvas(w, h, scale=scale, center=(0.0, 0.0), y_up=y_up, background=background)


def mask_alpha(canvas: Canvas, field) -> np.ndarray:
    """Coverage of a silhouette field over the canvas grid, as an array.

    Computed once and multiplied into every layer, which is both faster and
    cleaner than re-evaluating the silhouette per layer.
    """
    X, Y = canvas.grid()
    return render.coverage(field(X, Y), canvas.dx)


def paint_layers(canvas: Canvas, layers, clip=None, emboss: float = 0.0, bevel: float = 0.15, light=LIGHT):
    """Paint a list of ``(field, tincture)`` layers in order.

    ``clip`` is a precomputed coverage array (see :func:`mask_alpha`) that keeps
    the paint inside the silhouette.  Painting a partition in order gives exact
    anti-aliased seams: adjacent regions are both ~0.5 at their shared boundary,
    so source-over produces the correct 50/50 mix.
    """
    X, Y = canvas.grid()
    dx = canvas.dx
    for region, tincture in layers:
        t = tinctures.get(tincture)
        d = region(X, Y)
        a = render.coverage(d, dx)
        if clip is not None:
            a = a * clip
        if emboss > 0.0:
            canvas.over(render.shade(t.rgb, d, dx, bevel=emboss, light=light), a)
        else:
            canvas.over(palette.srgb_to_linear(t.rgb), a)


def paint_flat(canvas: Canvas, field, tincture, clip=None):
    """Convenience: paint one field in one tincture."""
    paint_layers(canvas, [(field, tinctures.get(tincture))], clip=clip)


def paint_shadow(canvas: Canvas, field, offset=(-0.02, -0.05), sigma: float = 0.030, opacity: float = 0.34, color=(0.0, 0.0, 0.0)):
    """Soft contact shadow under a silhouette, painted before anything else."""
    a = mask_alpha(canvas, field)
    sh = render.drop_shadow(a, canvas.scale, offset, sigma, opacity)
    canvas.over_srgb(color, sh)


def paint_rim(canvas: Canvas, field, width: float = 0.045, strength: float = 0.30, light=LIGHT):
    """A rim light along the inside of the silhouette edge.

    The silhouette's distance field doubles as a height field, so the lit and
    shadowed sides of the rim fall out of its gradient.  Multiplying the buffer
    darkens in linear light, which is what makes painted metal read as metal.
    """
    X, Y = canvas.grid()
    d = field(X, Y)
    b = max(float(width), 1e-9)
    h = np.clip(-d, 0.0, b)
    n = render.height_normals(h, canvas.dx, canvas.y_up)
    L = np.array([light[0], light[1], 0.65])
    L = L / max(float(np.linalg.norm(L)), 1e-12)
    lambert = np.clip(n @ L, 0.0, 1.0)
    edge = np.clip(1.0 - (-d) / b, 0.0, 1.0) * (d < 0.0)
    factor = 1.0 + strength * (lambert - 0.5) * 2.0 * edge
    canvas.rgb *= factor[..., None]
    np.clip(canvas.rgb, 0.0, 1.0, out=canvas.rgb)
