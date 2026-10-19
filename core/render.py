"""Turn distance fields into pixels: coverage, bevel lighting, shadows, sheets.

The interesting trick here is that a signed distance field is not just an
outline -- it is a *height field* waited to happen.  Its gradient gives surface
normals for free, so a bevel, a rim light and a specular highlight cost a couple
of ``np.gradient`` calls and no authoring at all.  That is why every asset in
this project is described as a distance field first and shaded afterwards.
"""
from __future__ import annotations

import numpy as np

from . import palette
from .canvas import Canvas

__all__ = [
    "coverage",
    "height_normals",
    "shade",
    "box_blur",
    "gaussian_blur",
    "translate_array",
    "drop_shadow",
    "cells",
    "sheet",
    "LIGHT",
]

#: The one light direction this project has, in world units and with y up.
#:
#: It lives in the core rather than in the packages because "every asset here is
#: lit the same way" is a property of the toolkit and not of any one generator --
#: and eight copies of a literal kept in step by a test are not one light, they
#: are eight lights that agree today.  Every shading call defaults to it, so the
#: packages that do not name a light get this one rather than a ninth.
LIGHT = (-0.45, 0.65)


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------
def coverage(d, dx: float = 1.0):
    """Anti-aliased coverage of a distance field, in ``[0, 1]``.

    A linear ramp one pixel wide.  It is exact for straight edges and a good
    approximation everywhere else, which is all a 2-D icon needs.
    """
    return np.clip(0.5 - np.asarray(d, dtype=np.float64) / dx, 0.0, 1.0)


# --------------------------------------------------------------------------
# lighting from the field
# --------------------------------------------------------------------------
def height_normals(h, dx: float = 1.0, y_up: bool = True) -> np.ndarray:
    """Normals of a height field ``h(x, y)``, shaped ``(..., 3)``.

    For a height map the normal is proportional to ``(-dh/dx, -dh/dy, 1)``.
    When the raster's rows run downwards while world ``y`` points up, the
    vertical derivative flips sign -- the one easy mistake to make here.
    """
    h = np.asarray(h, dtype=np.float64)
    gy = np.gradient(h, dx, axis=0)
    if y_up:
        gy = -gy
    gx = np.gradient(h, dx, axis=1)
    nz = np.ones_like(h)
    n = np.stack([-gx, -gy, nz], axis=-1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def _unit3(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def shade(
    base_srgb,
    d,
    dx: float = 1.0,
    bevel: float = 0.15,
    light=LIGHT,
    ambient: float = 0.45,
    diffuse: float = 0.6,
    spec: float = 0.25,
    shininess: float = 28.0,
    y_up: bool = True,
    rim: float = 0.0,
    inset: bool = False,
) -> np.ndarray:
    """Shade a distance field as a raised, beveled solid.

    ``bevel`` is the width (in world units) of the sloping band along the edge:
    zero gives a flat sticker, ~0.15 gives an embossed icon.  ``light`` is the
    in-plane direction towards the light source; the z-component is fixed, which
    keeps the look consistent across an entire asset set.

    ``inset`` turns the same ramp the other way: the height falls towards the
    edge, so the moulding reads as *pressed into* the surface instead of raised
    out of it.  A raised and a recessed bevel are not two effects -- they are one
    height field with the slope negated, which is why they share every other
    parameter.  Note that for a band (a ring of material) ``inset`` gives a
    valley with ridges along both boundaries, which is what a knocked-in button
    frame looks like.

    Returns linear-light RGB; ``coverage(d, dx)`` is still the matching alpha.
    """
    base = palette.srgb_to_linear(np.asarray(base_srgb, dtype=np.float64))
    d = np.asarray(d, dtype=np.float64)

    b = max(bevel, 1e-9)
    h = np.clip(-d, 0.0, b)
    if inset:
        h = b - h
    n = height_normals(h, dx, y_up)

    L = _unit3((light[0], light[1], 0.65))
    lambert = np.clip(n @ L, 0.0, 1.0)
    H = _unit3(L + np.array([0.0, 0.0, 1.0]))
    specular = np.clip(n @ H, 0.0, 1.0) ** shininess

    rgb = base * (ambient + diffuse * lambert)[..., None] + spec * specular[..., None]

    if rim > 0.0:
        # Fresnel-ish edge light: bright where the surface turns away from us.
        edge = np.clip(1.0 - n[..., 2], 0.0, 1.0) ** 2.0
        rgb = rgb + rim * edge[..., None]

    return np.clip(rgb, 0.0, 1.0)


# --------------------------------------------------------------------------
# blur and shadows
# --------------------------------------------------------------------------
def box_blur(a, radius: int) -> np.ndarray:
    """Separable box blur via cumulative sums: O(n) per axis, any radius."""
    a = np.asarray(a, dtype=np.float64)
    if radius < 1:
        return a.copy()
    k = 2 * radius + 1

    p = np.pad(a, ((radius, radius), (0, 0)), mode="edge")
    cs = np.cumsum(p, axis=0)
    cs = np.concatenate([np.zeros((1, cs.shape[1])), cs], axis=0)
    a = (cs[k:] - cs[:-k]) / k

    p = np.pad(a, ((0, 0), (radius, radius)), mode="edge")
    cs = np.cumsum(p, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1)), cs], axis=1)
    return (cs[:, k:] - cs[:, :-k]) / k


def gaussian_blur(a, sigma_px: float) -> np.ndarray:
    """Approximate Gaussian blur by three box passes.

    Three boxes have variance ``3*r*(r+1)/3 = r*(r+1)``, so ``r ~ sigma`` is the
    matching radius.  Visually indistinguishable from a true Gaussian kernel for
    the soft shadows and glows this project needs.
    """
    sigma_px = float(sigma_px)
    if sigma_px <= 0.0:
        return np.asarray(a, dtype=np.float64).copy()
    r = max(1, int(round(sigma_px)))
    out = np.asarray(a, dtype=np.float64)
    for _ in range(3):
        out = box_blur(out, r)
    return out


def translate_array(a, dx_px: float, dy_px: float, fill: float = 0.0) -> np.ndarray:
    """Shift an array by a (rounded) pixel offset, padding with ``fill``."""
    a = np.asarray(a, dtype=np.float64)
    ix = int(round(dx_px))
    iy = int(round(dy_px))
    if ix == 0 and iy == 0:
        return a.copy()
    h, w = a.shape
    out = np.full_like(a, fill)
    sx0, sx1 = max(0, -ix), min(w, w - ix)
    sy0, sy1 = max(0, -iy), min(h, h - iy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + iy : sy1 + iy, sx0 + ix : sx1 + ix] = a[sy0:sy1, sx0:sx1]
    return out


def drop_shadow(alpha, scale: float = 1.0, offset=(0.0, -0.06), sigma: float = 0.03, opacity: float = 0.4) -> np.ndarray:
    """Soft shadow alpha from a coverage mask.

    ``offset`` and ``sigma`` are in *world* units (converted with ``scale``), so
    a shadow keeps its proportions when the asset is re-rendered larger.
    """
    dx_px = offset[0] * scale
    dy_px = -offset[1] * scale  # world y up -> pixel rows down
    a = gaussian_blur(np.clip(alpha, 0.0, 1.0), sigma * scale)
    a = translate_array(a, dx_px, dy_px)
    return np.clip(a * opacity, 0.0, 1.0)


# --------------------------------------------------------------------------
# contact sheets / atlases
# --------------------------------------------------------------------------
def cells(count: int, cols: int, cw: int, ch: int, gap: int = 0) -> list:
    """Where the ``count`` cells of a ``cols``-wide grid begin, in pixels.

    Top-left corners, y down -- the convention the exported file has and the one
    ``export.atlas`` already writes its rectangles in, because a rectangle an engine
    reads is a rectangle in the image it reads it from.  :func:`sheet` draws on these
    and :func:`export.sheet.build` writes them into its sidecar, so the picture and the
    table of where each frame is cannot disagree about a single pixel.  That is the
    whole difference between a sprite sheet and a picture of one.
    """
    if cols < 1:
        raise ValueError("a grid is at least one column wide")
    if cw < 1 or ch < 1:
        raise ValueError("a cell has pixels in it")
    return [(gap + (i % cols) * (cw + gap), gap + (i // cols) * (ch + gap))
            for i in range(int(count))]


def sheet(canvases, cols: int = 4, gap: int = 8, background=(0.94, 0.94, 0.96), scale: float | None = None) -> Canvas:
    """Lay canvases out on a grid, centred, on a flat background.

    Doubles as the first step towards a texture atlas: the same tiling logic
    with pixel-exact packing is all a sprite sheet needs.  :func:`cells` is that
    tiling, in one place, and a sheet of *frames* rather than of samples is
    :func:`export.sheet.build` -- a grid, a table and a timeline.

    ``background=None`` gives clear margins instead of a paper colour, which is what a
    caller cutting the sheet up wants: a padding pixel the file says is not there is a
    pixel no bilinear filter can pull into a frame.
    """
    canvases = list(canvases)
    if not canvases:
        raise ValueError("sheet() needs at least one canvas")
    cw = max(c.w for c in canvases)
    ch = max(c.h for c in canvases)
    rows = (len(canvases) + cols - 1) // cols
    W = cols * cw + (cols + 1) * gap
    H = rows * ch + (rows + 1) * gap
    out = Canvas(W, H, scale=scale or canvases[0].scale, background=background)
    for c, (x, y) in zip(canvases, cells(len(canvases), cols, cw, ch, gap)):
        out.blit(c.rgb, c.a, x + (cw - c.w) // 2, y + (ch - c.h) // 2)
    return out
