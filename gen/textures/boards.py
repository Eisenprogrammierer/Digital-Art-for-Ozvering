"""Boards: the first texture here whose material has a direction.

Truchet, Wang and the reaction are all isotropic.  Turn any of them a quarter turn and
you have a different picture, but you have not drawn a different *kind* of thing.  A
floor is not like that: a board is long and thin, its grain runs the length of it, and
both of those are directions the tile has to carry across the seam.

Most of what that needs is familiar.  A board is a rectangle on a grid, so the grid has
to divide the tile -- a whole number of boards across and a whole number of bands down
-- and the stagger has to come back to a joint it started from by the time it reaches
the top, or the floor would jog sideways at every horizontal seam.  The grain is
:func:`gen.textures.tile.periodic_noise` with ``stretch``, so it is long along the
board and short across it, and periodic, so it wraps.

What is new is what a board *is* when it crosses the seam.  The stagger moves each
band's joints along, so in most bands a joint falls somewhere inside the tile and the
board it ends runs off the left edge and comes back on the right.  That is one board,
and its grain and its tone have to say so.  An identity computed from the tile would
give its two halves two different names, and the seam would then show up as a board
that changed colour halfway along it -- a worse defect than a visible line, because it
reads as a mistake rather than as a joint.  So the identity is computed from the
board's position on the *torus*: the same expression on both sides of the edge, taken
modulo the number of boards, gives the same answer -- or, where the edge itself lands
on a joint, the next board along and no other, which is what a joint is.  The test
asserts both, on every row, and asserts the same two things across the other seam.

The surface is a bevel and nothing else.  A sanded board is flat, and its grain is
*in* the wood rather than on it -- so the light finds the shape of the boards, in the
groove where two of them meet, and the grain is colour.  A grain with relief of its
own makes corrugated iron, which is a different material and a worse one.
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.canvas import Canvas

from . import tile as tiling

__all__ = ["layout", "texture"]

WOOD = (0.55, 0.058, 0.155)

BOARD_L = 0.075
BOARD_H = 0.018

GRAIN = 0.11

KNOTS = 0.32
KNOT_R = 0.34
KNOT_W = 0.65
KNOT_DARK = 0.22
KNOT_H = 0.6


def layout(size: int = 256, across: int = 1, down: int = 6, stagger: int = 2) -> dict:
    """The grid of boards: which band, which board, and where inside it a pixel is.

    ``across`` is how many board *ends* a band shows, ``down`` how many bands the tile
    is high, and ``stagger`` how many board-lengths the joints advance over the whole
    stack -- so that after ``down`` bands the joints are back on a grid they have been
    on before, which is the whole of why a floor tiles.  All three are whole numbers.

    Returns fields on the pixel grid: the band index, the board's position along its
    band *on the torus*, its index in the table of boards, its own coordinates ``u``
    along and ``v`` across, and the distance in pixels from the pixel to the nearest
    edge of its board.  The index is the band and the position together, because two
    boards a band apart are two boards: it is what a caller varies a board's tone, its
    grain and its knot by, and with ``across`` of 1 the position alone is always zero.
    """
    size, across, down, stagger = int(size), int(across), int(down), int(stagger)
    if across < 1 or down < 1:
        raise ValueError("a floor has at least one board across and one band down")
    if size < 1:
        raise ValueError("a floor is drawn at least one pixel across")
    ys = ((np.arange(size)[:, None] + 0.5) / size)
    xs = ((np.arange(size)[None, :] + 0.5) / size)
    band = np.broadcast_to(np.floor(ys * down).astype(np.int64), (size, size))
    v = np.broadcast_to(ys * down - np.floor(ys * down), (size, size))

    offset = band * float(stagger) / (across * down)
    grid = np.broadcast_to((xs - offset) * across, (size, size))
    floor = np.floor(grid)

    board = ((floor.astype(np.int64) % across) + across) % across
    u = grid - floor

    plank_px, band_px = size / across, size / down
    edge = np.minimum(np.minimum(u, 1.0 - u) * plank_px,
                      np.minimum(v, 1.0 - v) * band_px)
    return {"band": band, "board": board, "id": band * across + board,
            "u": u, "v": v, "edge": edge}


def texture(seed: int = 0, size: int = 256, across: int = 1, down: int = 6,
            stagger: int = 2, grain_scale: float = 5.0, stretch: float = 8.0,
            bevel: float = 3.0, dome: float = 1.5, knots: float = KNOTS,
            wood: tuple = WOOD) -> Canvas:
    """A finished floor: boards with their grain, bevelled into a groove between them.

    ``bevel`` is how many pixels the surface takes to come up from a board's edge to
    its flat top, and ``dome`` how many pixels it comes up by -- the two together are
    the shape of the groove, and they are the only places the light has to work with.
    ``grain_scale`` and ``stretch`` are the grain's width and how many times longer it
    is along the board than across it.  ``wood`` is the colour at the middle of the
    range the boards vary over, in OKLCH, so that a floor of something other than oak
    is a different tuple rather than a different function.
    """
    size, across, down = int(size), int(across), int(down)
    fields = layout(size=size, across=across, down=down, stagger=stagger)
    board_id, u, v, edge = fields["id"], fields["u"], fields["v"], fields["edge"]

    count = across * down
    rng = np.random.default_rng(int(seed) ^ 0x424F4152)
    tone = rng.uniform(-1.0, 1.0, count)
    turn = rng.uniform(-1.0, 1.0, count)
    shift = rng.integers(0, size, (count, 2))
    knot_at = rng.uniform(0.18, 0.82, (count, 2))
    knot_r = rng.uniform(0.4, 0.85, count) * KNOT_R * (size / max(down, 1))
    knot_on = rng.random(count) < float(knots)

    grain = tiling.periodic_noise((size, size), seed=int(seed) + 3067,
                                 scale=float(grain_scale), stretch=float(stretch))
    ys = np.arange(size)[:, None]
    xs = np.arange(size)[None, :]
    grain = grain[(ys - shift[board_id, 0]) % size, (xs - shift[board_id, 1]) % size]

    du = (u - knot_at[board_id, 0]) * (size / across) * KNOT_W
    dv = (v - knot_at[board_id, 1]) * (size / max(down, 1))
    radius = knot_r[board_id] * knot_on[board_id]
    knot = np.clip(1.0 - (np.hypot(du, dv) / np.maximum(radius, 1e-6)) ** 2, 0.0, 1.0)

    colour = np.array([palette.color(wood[0] + BOARD_L * t, wood[1], wood[2] + BOARD_H * h)
                       for t, h in zip(tone, turn)])
    srgb = colour[board_id]
    srgb = srgb * (1.0 - GRAIN + 2.0 * GRAIN * grain - knot * KNOT_DARK)[..., None]

    height = dome * np.clip(edge / max(float(bevel), 1e-6), 0.0, 1.0) + knot * KNOT_H
    lit = tiling.lit(srgb, height, y_up=False)

    canvas = Canvas(size, size, scale=1.0, y_up=False)
    canvas.over_srgb(np.clip(lit, 0.0, 1.0), np.ones((size, size), dtype=np.float64))
    return canvas
