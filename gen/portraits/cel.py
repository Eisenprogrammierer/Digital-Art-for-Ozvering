"""Cel shading: two or three flat bands where a renderer would put a gradient.

Anime shading is a *quantised* Lambert term -- hard-edged bands, a shadow colour
that is a different colour rather than a darker one, and a highlight that is a
shape rather than a falloff.  To place the terminator somewhere anatomically
sensible the face needs a volume, and a flat silhouette has no normals to light.
So the head is given a cheap ellipsoidal dome standing in for the skull, the
silhouette's own bevel is added to it, and the sum is lit and then quantised.
That is enough to put a crescent of shadow down one side of the face and a lit
plane on the other, in any pose, with nothing authored per pose.

The two colour rules that make a cel shadow read as anime rather than as a
darkened photograph:

* **Chroma goes up as lightness goes down.**  A pure lightness ramp desaturates
  into grey mud; a shadow keeps or gains chroma.
* **The hue rolls down through magenta towards violet.**  That is the direction
  every illustrator takes a shadow -- it reads as cool ambient light on warm
  skin -- whereas rolling the other way through yellow reads as jaundice.  The
  same rule applied to blue hair lands it on deep indigo, which is also right.

Shading itself is done by :meth:`gen.portraits.brush.Surface.bands`, so a shape
that has already been evaluated for its fill is not evaluated again to be lit.
"""
from __future__ import annotations

import numpy as np

from core import palette

from .brush import LIGHT, Surface

__all__ = ["LIGHT", "dome", "shade_shape", "shade_surface", "shadow", "highlight"]


def dome(rx: float, ry: float, height: float, cx: float = 0.0, cy: float = 0.0,
         power: float = 0.5):
    """Height field of an ellipsoidal cap, used as the skull under the face.

    ``power = 0.5`` is a true hemispheroid (normals roll off like a sphere).
    Smaller powers flatten the middle and concentrate the curvature at the rim,
    which is the knob for a rounder or a more planar face.
    """
    def h(x, y):
        q = 1.0 - ((x - cx) / rx) ** 2 - ((y - cy) / ry) ** 2
        return height * np.clip(q, 0.0, 1.0) ** power

    return h


def shade_surface(canvas, surface: Surface, colours, edges, volume=None, light=LIGHT,
                  bevel: float = 0.05, soft: float = 0.7, alpha: float = 1.0):
    """Quantise an already-evaluated surface into flat bands.

    ``colours[i]`` paints ``lambert < edges[i]``; band 0 is the deepest shadow,
    and the caller paints the base colour underneath.  Because the masks nest,
    the widest band is painted first and the deepest last, which lets a band be a
    thin sliver along the terminator without any per-band geometry.
    """
    surface.bands(canvas, colours, edges, volume=volume, light=light, bevel=bevel,
                  soft=soft, alpha=alpha)


def shade_shape(canvas, shape, colours, edges, volume=None, light=LIGHT,
                bevel: float = 0.05, soft: float = 0.7, alpha: float = 1.0):
    """As :func:`shade_surface`, for a shape that has not been evaluated yet."""
    shade_surface(canvas, Surface(canvas, shape), colours, edges, volume=volume,
                  light=light, bevel=bevel, soft=soft, alpha=alpha)


def shadow(base, depth: float = 0.085, chroma: float = 1.18, hue: float = -0.030):
    """An anime shadow colour: darker, more saturated, rolled towards violet."""
    L, C, h = palette.srgb_to_oklch(base)
    return palette.oklch_to_srgb(
        (max(float(L) - depth, 0.0), max(float(C) * chroma, 0.0), (float(h) + hue) % 1.0)
    )


def highlight(base, lift: float = 0.10, chroma: float = 0.85, hue: float = 0.008):
    """An anime highlight: lighter and slightly *less* saturated than the base."""
    L, C, h = palette.srgb_to_oklch(base)
    return palette.oklch_to_srgb(
        (min(float(L) + lift, 1.0), max(float(C) * chroma, 0.0), (float(h) + hue) % 1.0)
    )
