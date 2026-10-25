"""Edge tiles: a border belongs to the edge between two hexes, not to either of them.

The coastline is the last thing on a map that was still *drawn* rather than
*chosen*.  Everything else -- the ground, the relief, the provinces -- is a
per-cell decision that a raster pass happens to be a convenient way to make; the
coast was a polyline walked round the land and stroked, which produces a picture
and nothing else.  A game cannot use a picture.  What it can use is a tileset: a
small number of sprites, and the rule that says which hex wears which.

So the coast is reduced to its decision, and the decision is local:

    a hex wears the six-bit number of which of its six edges is a border.

Both hexes across an edge set the bit -- the relation is symmetric, which is the
fact the whole module rests on -- so "different from my neighbour" is enough and
no walk round anything is needed.  A tile is then one of the six-bit numbers, and
:func:`sheet` draws all sixty-four of them, because a tileset is an *asset*: it
has to be complete to be usable on somebody else's map, not only on the map that
happened to produce it.

The one decision the rest follows from is that **a tile's ink lies inside its own
ground**.  A stroke centred on the shared edge would belong half to each
neighbour, and then the picture could not be cut into tiles at all.  Ink inside
the hexagon keeps the tiles disjoint -- hexagons already tile the plane, so tiles
that stay inside their hexagon compose with no rule and no seam -- and it is also
what the picture wants, since a border is not a line lying between two things but
the rim of each of them.

What is deliberately not here: the ground.  A tile carries the border and not the
terrain under it, because the terrain is *not* a function of the tile.  Relief
comes from the height of a hex and of its neighbours, so two hexes wearing the
same tile can be a plain and a mountain, and a sheet of ground tiles would be a
sheet of lies.  The border is the part that is genuinely local, so it is the part
that becomes tiles.

Nothing above is about water.  A border is where two hexes disagree, so the same
sixty-four tiles draw a coastline, a biome border, a frontier between two realms
or a line between two countries; the labelling is the caller's and so is the ink.
The one thing a coast needs that a biome border does not is a boundary to stop
at -- which is what ``within`` is for.
"""
from __future__ import annotations

import numpy as np

from core import render
from core.canvas import Canvas

from .grid import HexGrid

__all__ = ["EDGES", "MASKS", "SHEET_COLS", "borders", "tiles", "border_layer",
           "tile_size", "tile_image", "sheet"]

EDGES = 6
MASKS = 1 << EDGES

SHEET_COLS = 8

_COS = tuple(float(np.cos(np.radians(60.0 * k - 30.0))) for k in range(6))
_SIN = tuple(float(np.sin(np.radians(60.0 * k - 30.0))) for k in range(6))


def _segment_distance(x, y, ax, ay, bx, by):
    """Distance to the segment ``a``-``b``, when ``a`` and ``b`` are themselves arrays.

    :func:`core.sdf.segment` does this for fixed endpoints, and cannot do it here:
    a pixel's edge is an edge of *that pixel's* hexagon, so the endpoints move
    with it and arrive as arrays.  The arithmetic is the same, minus the guard
    against a zero-length edge, which a hexagon does not have.
    """
    px, py = x - ax, y - ay
    ex, ey = bx - ax, by - ay
    h = np.clip((px * ex + py * ey) / (ex * ex + ey * ey), 0.0, 1.0)
    return np.hypot(px - ex * h, py - ey * h)


def borders(labels, grid: HexGrid, outside=None, within=None) -> np.ndarray:
    """Which of each hex's six edges is a border, as a six-bit number.

    Bit ``d`` is set when the neighbour across edge ``d`` is of a different
    class, which is a *local* question -- there is no walk and no set of
    connected components anywhere in here -- and the answer is symmetric, so the
    two hexes across an edge always agree that there is a border between them.

    ``outside`` is what lies beyond the map's edge, or ``None`` to treat the rim
    as a class of its own, in which case every edge along the rim is a border.
    The maps here pass the sea, because that is what their edge already looks
    like: a frame of coastline all the way round would be a picture frame rather
    than a coast.

    ``within`` asks the same question of a *sub*set: an edge is a border only
    where both of the hexes that meet on it are in that set.  A biome border is
    asked with ``within=~water``, because which ground is on the other side of
    the water is a question the coastline already answers, and a border drawn
    between a forest and the sea would only make the coast thicker.  A hex off
    the map is never within, so the rim drops out on its own and ``outside`` has
    nothing left to say about it.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (grid.rows, grid.cols):
        raise ValueError(f"labels must be {grid.rows}x{grid.cols}, not {labels.shape}")
    if within is not None:
        within = np.asarray(within, dtype=bool)
        if within.shape != labels.shape:
            raise ValueError(f"within must be {grid.rows}x{grid.cols}, not {within.shape}")
    beyond = -1 if outside is None else int(outside)
    out = np.zeros((grid.rows, grid.cols), dtype=np.int64)
    for d in range(EDGES):
        nr, nc = grid.nbr[..., d, 0], grid.nbr[..., d, 1]
        here = nr >= 0
        rr = np.clip(nr, 0, grid.rows - 1)
        cc = np.clip(nc, 0, grid.cols - 1)
        different = np.where(here, labels[rr, cc], beyond) != labels
        if within is not None:
            different &= within & np.where(here, within[rr, cc], False)
        out |= np.where(different, 1 << d, 0)
    return out


def tiles(mask) -> tuple:
    """The tile each hex wears, and the tiles that occur, in the order first seen.

    Rows then columns, so the answer is a function of the map and not of any
    hash: two runs give the same list, and the list is short for the same reason
    a coastline is a small idea -- a hex in the middle of the sea has no borders
    at all, and one in the middle of the land has none either.
    """
    mask = np.asarray(mask, dtype=np.int64)
    seen: dict = {}
    index = np.zeros(mask.shape, dtype=np.int64)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            m = int(mask[r, c])
            if m not in seen:
                seen[m] = len(seen)
            index[r, c] = seen[m]
    return index, list(seen)


def border_layer(canvas: Canvas, grid: HexGrid, mask, cell, srgb,
                 width: float = 0.24, alpha: float = 0.80) -> None:
    """The border ink: the edges the key marks, inked inside their own hexagon.

    ``cell`` is the raster lookup the rest of the painter already has, so every
    pixel knows which hexagon it is in; the only other thing needed is which of
    *that* hexagon's six edges the key marks.  Three hexes meet at a corner, so
    the temptation is to look at the neighbours' edges as well -- and it is not
    needed.  If a hex agrees with both of the hexes it meets at a corner then
    those two agree with each other, so the far edge there is no border either,
    and a hexagon's own six edges are the only ones that can be within reach of
    the pixels inside it.  A border drawn this way is therefore the same picture
    the boundary walk drew, with one difference that is the point of the module:
    the ink stays on its own side, so nothing spills off the edge of the map.

    ``width`` is the band's full width in hex sizes, so the ink reaches
    ``width / 2`` into each hexagon and the two sides of a border meet at the
    edge, which is where the border actually is.
    """
    r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
    c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
    key = mask[r, c]
    size = grid.size
    hw = 0.5 * float(width) * size
    cx = grid.centers[r, c, 0]
    cy = grid.centers[r, c, 1]
    X, Y = canvas.grid()
    a = np.zeros(X.shape, dtype=np.float64)
    for d in range(EDGES):
        live = ((key >> d) & 1).astype(bool)
        if not live.any():
            continue
        k = (d + 1) % EDGES
        dist = _segment_distance(X, Y,
                                 cx + size * _COS[d], cy + size * _SIN[d],
                                 cx + size * _COS[k], cy + size * _SIN[k])
        a = np.maximum(a, np.where(live, render.coverage(dist - hw, canvas.dx), 0.0))
    a = a * cell.ok
    if a.any():
        canvas.over_srgb(srgb, a * float(alpha))


def tile_size(size: float) -> tuple:
    """The ``(width, height)`` in pixels of a tile for hexes of ``size``.

    Odd in both directions, so the hexagon's centre falls on a pixel centre and
    its corners fall a pixel clear of the frame: a tile has to be the same bytes
    every time it is drawn, and it must not be cropped by its own edge.
    """
    return (2 * int(np.ceil(0.5 * np.sqrt(3.0) * size)) + 1,
            2 * int(np.ceil(size)) + 1)


def tile_image(key: int, size: float, srgb, width: float = 0.24,
               alpha: float = 0.80, background=None) -> Canvas:
    """One tile, on its own: a hexagon of ``size`` with the marked edges inked.

    The tile is the hexagon's bounding box, and the hexagon sits on the canvas
    with its centre on a pixel centre and its corners a pixel clear of the edge,
    so the same key always renders to the same bytes and no ink is clipped by the
    frame.  That is the whole of the tile's specification: it is the border layer
    evaluated for one key, which is why the sheet and the map cannot disagree.
    """
    w, h = tile_size(size)
    m = int(key)
    if not 0 <= m < MASKS:
        raise ValueError(f"a tile key is one of {MASKS} edge patterns, not {key!r}")
    canvas = Canvas(w, h, scale=1.0, center=(0.0, 0.0), background=background)
    one = HexGrid(1, 1, size)
    mask = np.full((1, 1), m, dtype=np.int64)
    border_layer(canvas, one, mask, one.cell(*canvas.grid()), srgb,
                 width=width, alpha=alpha)
    return canvas


def sheet(srgb, size: float, width: float = 0.24, alpha: float = 0.80,
          gap: int = 8, background=None) -> Canvas:
    """All sixty-four tiles, in one eight-by-eight sheet, in key order.

    The sheet is an asset rather than a report: a game asks which of a hex's six
    edges is a border, takes that number as the tile, and blits it, so the sheet
    has to hold every key and not only the keys this generator's own maps happen
    to use.  :func:`tiles` is what measures the other thing -- how few of the
    sixty-four a real map wears.

    Left transparent, because a sheet is a container: a game lifts tile ``m``
    straight out of the ``(m // 8, m % 8)`` cell, and a background would bake
    itself into every pixel the tile does not cover.
    """
    return render.sheet([tile_image(m, size, srgb, width=width, alpha=alpha)
                         for m in range(MASKS)],
                        cols=SHEET_COLS, gap=gap, background=background)
