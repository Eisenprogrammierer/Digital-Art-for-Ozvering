"""Tiles: the arithmetic that lets a texture repeat without a seam.

A tiling texture is not a picture that happens to be square.  It is a square with
a *wrap*: the pixel on its right edge is the neighbour of the pixel on its left
edge, and the same for top and bottom.  Anything built out of a tile has to be
built with that in mind, because a seam is not a place where two pictures meet --
it is a place where the eye finds a discontinuity, and it finds one wherever the
statistics change, even if the colours almost match.

Three things live here.

:func:`seam_error`
    How much larger the step across the wrap is than a step inside the tile.  It
    is about 1 for a field that was built to wrap and about 10 for one that was
    not, and it turns "seamless" from a claim about how a rendered picture reads
    into a number a test can hold the generator to.

:func:`tile_coords`
    Per-pixel tile index and offset inside the tile, which is the only way a
    vectorised painter can know which tile it is in *without* a loop over tiles:
    the same trick the map uses to know which hex a pixel landed in.

:func:`lit`
    Lambert shading of a colour field by a height field, using the same light
    direction as heraldry, portraits, UI and maps.  A texture lit from the other
    side is the fastest way to make an asset look borrowed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["LIGHT", "lit", "luminance", "periodic_noise", "repeat", "seam_error",
           "smooth", "tile_coords"]

#: The project's single light direction, in pixels/points with y pointing up.
#: Every generator in this repository uses it; a texture lit from anywhere else
#: would look like a visitor from another project.
LIGHT = (-0.45, 0.65)


def periodic_noise(shape, seed: int = 0, scale: float = 6.0):
    """Smooth noise that wraps, in ``[0, 1]``, with features about ``scale`` px wide.

    Made by damping the spectrum of seeded white noise with a Gaussian.  That is
    band-limited, which is what makes it smooth, and -- because every basis function
    of a DFT is itself periodic on the canvas -- it is *exactly* periodic, which is
    what makes it a texture rather than a picture of one.

    The obvious alternative, evaluating ``core.noise`` in pixel coordinates, is not
    periodic and cannot be made so by choosing a round cell size: the value two
    tiles apart hashes to a different lattice cell, so a texture whose *pattern*
    wraps perfectly still shows a line where the grain changes character.
    """
    h, w = int(shape[0]), int(shape[1])
    rng = np.random.default_rng(int(seed))
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    kernel = np.exp(-2.0 * np.pi ** 2 * float(scale) ** 2 * (fy ** 2 + fx ** 2))
    field = np.fft.ifft2(np.fft.fft2(white) * kernel).real
    field -= float(field.min())
    return field / max(float(field.max()), 1e-12)


def tile_coords(shape, tile: int):
    """Per-pixel ``(tile row, tile col, u, v)`` for a canvas of whole tiles.

    ``u`` and ``v`` are in ``[0, 1)`` inside their tile, so a painter can evaluate
    a tile's own geometry in its own coordinates and never think about the grid.
    Values are integer indices and float offsets, all broadcast to ``shape``
    without a Python loop, because a texture is evaluated on every pixel.
    """
    h, w = int(shape[0]), int(shape[1])
    if tile < 1 or w % tile or h % tile:
        raise ValueError(f"a {w}x{h} canvas is not a whole number of {tile} px tiles")
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    return tuple(np.broadcast_to(f, (h, w)) for f in (
        ys // tile, xs // tile, (xs % tile) / tile, (ys % tile) / tile,
    ))


def repeat(field, n: int):
    """Tile a field ``n`` by ``n``, for the picture that *shows* it wraps."""
    return np.tile(np.asarray(field), (int(n), int(n)) + (1,) * (np.ndim(field) - 2))


def _smooth_plane(field, sigma: float):
    """The blur itself, on one two-dimensional plane."""
    h, w = field.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    kernel = np.exp(-2.0 * np.pi ** 2 * float(sigma) ** 2 * (fy ** 2 + fx ** 2))
    return np.fft.ifft2(np.fft.fft2(field) * kernel).real


def smooth(field, sigma: float):
    """Gaussian blur that wraps, via the spectrum.

    Wrapping matters for the same reason it matters everywhere else here: a blurred
    tile with a clamped edge has a border that differs from its interior, and the
    seam comes back exactly where the pattern had gotten rid of it.  Doing it in the
    frequency domain gets the wrap for nothing, because that is what a DFT's basis
    functions already are.

    A colour field is ``(height, width, channels)`` and is smoothed one channel at a
    time.  That is written out here rather than done in one batched transform on
    purpose.  The first version reshaped the field so that the channels led, which
    *looks* like the same thing and is not: ``reshape`` reinterprets the memory
    order, so the planes it produced were interleaved slices of all three channels
    at once.  It raised nothing and it ran at the same speed -- it just quietly
    blended the colours into one another, which reads as a slightly grey texture
    rather than as a bug.  Per-channel is one line longer and cannot do that.
    """
    f = np.asarray(field, dtype=np.float64)
    if sigma <= 0.0:
        return f.copy()
    if f.ndim > 2:
        return np.stack([_smooth_plane(f[..., c], sigma) for c in range(f.shape[-1])],
                        axis=-1)
    return _smooth_plane(f, sigma)


def luminance(rgba) -> np.ndarray:
    """The brightness of a straight-alpha RGBA array, composited over white.

    A seam is a property of what was drawn, not of the field the drawing came from,
    so this is what :func:`seam_error` is pointed at when the question is whether the
    *texture* tiles.  Over white because that is what a texture with holes in it
    looks like on a page.
    """
    px = np.asarray(rgba, dtype=np.float64)
    if px.ndim != 3 or px.shape[-1] not in (3, 4):
        raise ValueError("expected an (h, w, 3) or (h, w, 4) array")
    if px.max() > 1.5:
        px = px / 255.0
    rgb = px[..., :3]
    if px.shape[-1] == 4:
        a = px[..., 3:4]
        rgb = rgb * a + (1.0 - a)
    return rgb.mean(axis=-1)


def seam_error(field) -> float:
    """The step across the wrap, over the mean step inside the tile.

    For a field that wraps, the last row really is one pixel away from the first,
    so the ratio is about 1.  For a field that does not, the wrap is a cliff and
    the ratio is large -- and it is the *statistics* that matter, not the colours:
    a seam is visible wherever the pattern changes character, which is why this is
    measured on the field itself and not on a difference of averaged brightness.
    """
    f = np.asarray(field, dtype=np.float64)
    if f.ndim < 2:
        raise ValueError("a tile is at least two dimensional")
    dy = np.abs(np.diff(f, axis=0)).mean()
    dx = np.abs(np.diff(f, axis=1)).mean()
    wrap_y = np.abs(f[0] - f[-1]).mean()
    wrap_x = np.abs(f[:, 0] - f[:, -1]).mean()
    inside = max(dy + dx, 1e-12)
    return float((wrap_x + wrap_y) / inside)


def lit(srgb, height, ambient: float = 0.62, diffuse: float = 0.55,
        light=LIGHT, dx: float = 1.0, y_up: bool = False):
    """Shade a colour field by a height field, with the project's one light.

    A multiplier rather than a mix, for the same reason the map does it that way:
    the colours of a texture are the point of it, and shading that pulls them
    towards grey loses the thing the texture was for.
    """
    from core import render  # local: keeps the module's import graph a leaf

    colour = np.asarray(srgb, dtype=np.float64)
    n = render.height_normals(np.asarray(height, dtype=np.float64), dx, y_up=y_up)
    L = np.array([light[0], light[1], 0.65], dtype=np.float64)
    L /= max(float(np.linalg.norm(L)), 1e-12)
    lam = np.clip(n @ L, 0.0, 1.0)
    return colour * (ambient + diffuse * lam)[..., None]
