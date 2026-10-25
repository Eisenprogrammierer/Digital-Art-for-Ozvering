"""Gray-Scott reaction-diffusion: patterns that are grown rather than drawn.

Two chemicals, one reaction, one diffusion each, and the whole zoo of organic
textures comes out of it.  The model is

    du/dt = Du * lap(u) - u*v**2 + F * (1 - u)
    dv/dt = Dv * lap(v) + u*v**2 - (F + k) * v

and the only thing that matters here is that it is *simulated on a torus*: the
Laplacian wraps, so the field is periodic by construction and the texture tiles
without a seam to check, fix or apologise for.  That is the reason this module
belongs in a package about tiling, and it is the same reason the map's terrain is
evaluated in normalised coordinates -- the boundary condition is the asset.

Determinism is not an accident either.  The initial condition comes from a seeded
generator, the integrator is a fixed explicit Euler step, and the run stops either
after a fixed number of steps or when the field stops changing by more than a
tolerance.  Both decisions are functions of the field alone, so the same seed gives
the same bytes on any machine -- which for a *simulation* means the float arithmetic
has to be exactly reproducible, and it is: one fixed order of operations over numpy
arrays, no threading and no reductions that could reorder.

A pattern has no detail finer than a few cells, so it is simulated on a small grid
and sampled up on the torus.  Simulating 384 by 384 directly would cost sixteen
times as much for the same picture.
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.canvas import Canvas

from . import tile as tiling

__all__ = ["LAPLACIAN", "PRESETS", "preset", "simulate", "texture", "upsample"]

LAPLACIAN = np.array(
    [[0.05, 0.20, 0.05],
     [0.20, -1.00, 0.20],
     [0.05, 0.20, 0.05]], dtype=np.float64,
)

DU, DV = 0.16, 0.08
DT = 1.0

PRESETS = {
    "coral": ((0.0545, 0.0620), "branching fingers, like coral or lichen"),
    "mitosis": ((0.0367, 0.0649), "cells that divide and separate"),
    "worms": ((0.0220, 0.0510), "long wandering stripes"),
    "spots": ((0.0350, 0.0650), "round spots on a plain ground"),
    "holes": ((0.0390, 0.0580), "spongy, with holes in it"),
}


def preset(name: str):
    """The ``(F, k)`` pair behind a preset name."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; try one of {sorted(PRESETS)}")
    return PRESETS[name][0]


def _seed_field(size: int, seed: int, blobs: int | None = None, radius: int = 3,
                coverage: float = 0.22):
    """``u`` and ``v``: an empty field with a few seeded lumps in it.

    The lumps matter twice over.  The model is symmetric and a perfectly uniform
    start is a fixed point, so without them nothing happens at all; and their
    density is what decides how much of the tile the pattern has covered by the time
    the run ends, because the pattern grows outwards from wherever it started.  A
    sparse start is a picture of dots however long it runs; a generous one is a
    material.

    They are round, and that is not decoration.  A square lump is a square shape at
    the length scale the instability selects, so the pattern grows *from* it and the
    squares come out the other side -- which is exactly what the first version of
    this module did, and it looked like a low-resolution voxel game.
    """
    u = np.ones((size, size), dtype=np.float64)
    v = np.zeros((size, size), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    if blobs is None:

        area = np.pi * (radius + 0.5) ** 2
        blobs = max(4, int(float(coverage) * size * size / area))
    span = np.arange(-radius, radius + 1)
    disc = np.hypot(span[:, None], span[None, :]) <= radius + 0.25
    for _ in range(int(blobs)):
        cy, cx = (int(rng.integers(0, size)) for _ in range(2))
        ys = (np.arange(cy - radius, cy + radius + 1) % size)
        xs = (np.arange(cx - radius, cx + radius + 1) % size)
        block_v = v[np.ix_(ys, xs)]
        block_v[disc] = 0.25
        v[np.ix_(ys, xs)] = block_v
        block_u = u[np.ix_(ys, xs)]
        block_u[disc] = 0.50
        u[np.ix_(ys, xs)] = block_u
    return u, v


def _laplacian(field):
    """The 9-point Laplacian *with wrap*, which is the whole point of the module.

    Padded once and then sliced, rather than rolled nine times: the slices are
    views, so the step costs one array per chemical instead of eighteen, and this
    is the innermost loop of the only heavy simulation in the repository.
    """
    p = np.pad(field, 1, mode="wrap")
    return (
        0.05 * (p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:])
        + 0.20 * (p[:-2, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + p[2:, 1:-1])
        - 1.00 * p[1:-1, 1:-1]
    )


def simulate(seed: int = 0, size: int = 96, f: float | None = None, k: float | None = None,
             preset_name: str = "coral", steps: int = 4000, settle: bool = True,
             tol: float = 2e-5, check: int = 25):
    """Run the model on a torus and return the ``v`` field.

    ``settle`` stops early once the field stops moving, which is what a pattern
    does: it reaches a state where nothing changes but the shapes stay.  The check
    is the largest change anywhere, sampled every ``check`` steps -- deterministic,
    and cheap enough to be worth having instead of a step count nobody can guess.
    """
    size = int(size)
    if size < 4:
        raise ValueError("a reaction-diffusion grid needs at least 4 cells")
    F, kk = (preset(preset_name) if f is None or k is None else (float(f), float(k)))
    u, v = _seed_field(size, seed)
    prev = v.copy()
    for step in range(int(steps)):
        lu = _laplacian(u)
        lv = _laplacian(v)
        uvv = u * v * v
        u = u + DT * (DU * lu - uvv + F * (1.0 - u))
        v = v + DT * (DV * lv + uvv - (F + kk) * v)
        if settle and check and (step + 1) % check == 0:

            if float(np.abs(v - prev).max()) < tol:
                break
            prev = v.copy()
    return v


def upsample(field, size: int):
    """Bilinear magnification to ``size``, interpolating on the torus.

    Wrapping at the edges is not tidiness: a pattern sampled up with a clamped edge
    would have a different border from its interior, and the seam would come back
    exactly where the simulation had removed it.
    """
    field = np.asarray(field, dtype=np.float64)
    h, w = field.shape
    ys = (np.arange(int(size)) + 0.5) * h / int(size) - 0.5
    xs = (np.arange(int(size)) + 0.5) * w / int(size) - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    y0 %= h
    x0 %= w
    y1 = (y0 + 1) % h
    x1 = (x0 + 1) % w
    top = field[np.ix_(y0, x0)] * (1.0 - fx) + field[np.ix_(y0, x1)] * fx
    bot = field[np.ix_(y1, x0)] * (1.0 - fx) + field[np.ix_(y1, x1)] * fx
    return top * (1.0 - fy) + bot * fy


def texture(seed: int = 0, size: int = 384, sim: int = 96, preset_name: str = "coral",
            steps: int = 4000, ramp=("#2a1018", "#7d2f2c", "#d97b45", "#f6e2c0"),
            relief: float = 2.6, ambient: float = 0.55, diffuse: float = 0.75):
    """A finished reaction-diffusion texture: a colour ramp over the field, lit.

    The relief is the field itself, so the pattern is not a picture of something
    organic -- it is a surface whose height *is* the concentration, lit by the
    project's one light.  That is why it reads as coral and not as camouflage.
    """
    size = int(size)
    v = simulate(seed=seed, size=sim, preset_name=preset_name, steps=steps)
    big = upsample(v, size)

    lo = float(np.quantile(big, 0.10))
    hi = float(np.quantile(big, 0.92))
    t = np.clip((big - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    colours = np.array([palette.hex_to_srgb(c) for c in ramp])
    srgb = palette.ramp_map(colours, t)
    canvas = Canvas(size, size, scale=1.0, y_up=False)

    height = (tiling.smooth(t, float(size) / float(sim) * 0.9) - 0.5) * float(relief)
    canvas.over_srgb(np.clip(tiling.lit(srgb, height, ambient=ambient, diffuse=diffuse,
                                        y_up=False), 0.0, 1.0),
                     np.ones((size, size), dtype=np.float64))
    return canvas
