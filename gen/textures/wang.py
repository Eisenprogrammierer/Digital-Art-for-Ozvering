"""Wang tiles: the same seam, but this time it is the problem.

A Wang tile is a square that carries its edges.  Put tiles next to each other and
the shared edges have to agree; if they do not, the picture has a seam by
construction, and no amount of colour-matching will hide it.  So this module is the
mirror image of :mod:`gen.textures.truchet`: there, an arrangement was seamless for
*any* choice, and here the choice at every cell is constrained by the cells before
it.  The interesting part is therefore not the drawing.  It is the search.

The tiles here carry their *corners* rather than their edges, which is the form that
reads as terrain instead of as plumbing:

* a tile is four colours, one per corner;
* two neighbours agree when the two corners they share are equal -- so a left and a
  right neighbour agree about the top and the bottom of the edge between them;
* and a colour field that is the *bilinear blend* of the four corner colours is then
  continuous across every edge, at every corner of the grid, whatever the tiling is.
  The blend at the right-hand edge of a tile is the average of its two right-hand
  corners; the blend at the left-hand edge of the tile beside it is the average of
  its two left-hand corners; and those are the same two numbers, because that is
  what agreeing means.  Nothing else has to be checked.

The wrap is part of the problem too.  A texture has to tile, so the tiling is
generated on a torus: the last column has to agree with the first, and the last row
with the first.  A scanline search sees that for free -- the tile it must agree with
was laid down a whole row ago -- so the search is a depth-first walk with a stack,
and the generator can say honestly that a set either tiles or does not.

The ``complete`` set is all ``colours**4`` tiles, which tiles everything and never
backtracks: after the two neighbours have pinned two corners, the other two are free.
The ``diagonal`` set is the four tiles whose opposite corners match, and it *does*
dead-end, which is why the search has a stack at all.
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.canvas import Canvas

from . import tile as tiling

__all__ = ["PALETTES", "fill", "tile_set", "texture"]

#: Corner colours and the height each stands for, which is what gives the blobs
#: their relief.  Named for what they look like, because a texture is chosen by eye.
PALETTES = {
    "meadow": {
        "colours": ("#5d7f3c", "#87a24a", "#c9b46b", "#8c6f46"),
        "heights": (0.55, 0.70, 0.35, 0.80),
    },
    "stone": {
        "colours": ("#6b6f76", "#8b8f94", "#4e5259", "#a9a49b"),
        "heights": (0.5, 0.65, 0.3, 0.75),
    },
    "marsh": {
        "colours": ("#3f6b57", "#6f9163", "#8e9a5a", "#3b5570"),
        "heights": (0.5, 0.6, 0.65, 0.15),
    },
}


def tile_set(colours: int = 2, kind: str = "complete"):
    """The tiles of a set, as ``(top-left, top-right, bottom-left, bottom-right)``."""
    k = int(colours)
    if k < 2:
        raise ValueError("a Wang set needs at least two colours")
    if kind == "complete":
        return [(tl, tr, bl, br) for tl in range(k) for tr in range(k)
                for bl in range(k) for br in range(k)]
    if kind == "diagonal":
        return [(tl, tr, bl, br) for tl in range(k) for tr in range(k)
                for bl in range(k) for br in range(k)
                if tl == br and tr == bl]
    raise ValueError(f"unknown tile set {kind!r}")


def _agrees(tiles, grid, cols, rows, r: int, c: int, idx: int) -> bool:
    """Does tile ``idx`` fit at ``(r, c)``, including across the wrap?

    The left-hand neighbour is the one case that cannot be checked when it is met:
    at the start of a row it lives on the far side of the torus and has not been
    laid down yet.  It does not need to be -- the last column of that row checks the
    same two corners from the other side, and the constraint is the same one.
    """
    tl, tr, bl, br = tiles[idx]
    if c > 0:
        left = tiles[grid[r, c - 1]]
        if tl != left[1] or bl != left[3]:
            return False
    if r > 0:
        up = tiles[grid[r - 1, c]]
        if tl != up[2] or tr != up[3]:
            return False
    if c == cols - 1 and cols > 1:
        right = tiles[grid[r, 0]]
        if tr != right[0] or br != right[2]:
            return False
    if r == rows - 1 and rows > 1:
        down = tiles[grid[0, c]]
        if bl != down[0] or br != down[1]:
            return False
    return True


def fill(cols: int, rows: int, tiles, seed: int = 0, budget: int = 200000):
    """Lay tiles so every shared corner agrees, including across the wrap.

    A depth-first walk with a stack, in scanline order, trying the tiles in a
    seeded order at each cell.  It raises if the budget runs out rather than
    looping: a set that cannot tile a torus of this size should say so, not hang.
    """
    cols, rows = int(cols), int(rows)
    if cols < 1 or rows < 1:
        raise ValueError("a tiling needs at least one tile")
    if not tiles:
        raise ValueError("an empty tile set cannot tile anything")
    rng = np.random.default_rng(int(seed))
    grid = np.full((rows, cols), -1, dtype=np.int64)
    stacks = [None] * (rows * cols)
    pos = 0
    tries = 0
    while pos < cols * rows:
        r, c = divmod(pos, cols)
        if stacks[pos] is None:
            stacks[pos] = list(rng.permutation(len(tiles)))
        chosen = -1
        while stacks[pos]:
            tries += 1
            if tries > budget:
                raise ValueError(f"gave up after {budget} tries at {cols}x{rows} tiles")
            idx = int(stacks[pos].pop())
            if _agrees(tiles, grid, cols, rows, r, c, idx):
                chosen = idx
                break
        if chosen < 0:
            grid[r, c] = -1
            stacks[pos] = None
            pos -= 1
            if pos < 0:
                raise ValueError(f"this set of {len(tiles)} tiles cannot tile "
                                 f"{cols}x{rows} even going backwards")
            continue
        grid[r, c] = chosen
        pos += 1
    return grid


def texture(seed: int = 0, size: int = 256, cells: int = 16, colours: int = 3,
            kind: str = "complete", palette_name: str = "meadow", margin: float = 0.42,
            relief: float = 7.0, grain: float = 0.05) -> Canvas:
    """A finished Wang tiling: corner colours blended into blobs, with relief.

    ``margin`` is how much of each tile is spent blending between its corners.  At
    a small margin the corners are big flat patches and the texture reads as a map;
    at a large one it is all gradient and reads as a wash.  Either way it is
    continuous, which is the one thing this module can promise by construction.
    The grid wants to be fine -- sixteen cells across rather than four -- because
    what makes a Wang tiling look like terrain is that the blobs are *smaller* than
    the eye expects tiles to be.
    """
    size, cells = int(size), int(cells)
    if size % cells:
        raise ValueError(f"a {size} px tile is not a whole number of {cells} cells")
    if not 0.0 < margin <= 0.5:
        # Above half a tile the blend reaches the opposite corner, and then agreeing
        # about the shared corners is no longer enough to make the field continuous;
        # at zero there is no blend at all and the partition is a hard Voronoi edge,
        # which is the discontinuity this whole module exists to avoid.
        raise ValueError("margin is a fraction of a tile, above 0 and at most 0.5")
    pal = PALETTES[palette_name]
    if colours > len(pal["colours"]):
        raise ValueError(f"{palette_name} has only {len(pal['colours'])} colours")
    tset = tile_set(colours, kind)
    grid = fill(cells, cells, tset, seed=seed)

    colour_of = np.array([palette.hex_to_srgb(c) for c in pal["colours"]][:colours])
    height_of = np.array(pal["heights"], dtype=np.float64)[:colours]

    i, j, u, v = tiling.tile_coords((size, size), size // cells)

    # The four corner colours of the tile under every pixel, in a fixed order:
    # top-left, top-right, bottom-left, bottom-right.  One gather per corner, over
    # the whole canvas at once, because a texture is evaluated on every pixel.
    codes = np.array(tset, dtype=np.int64)          # (tiles, 4)
    idx = grid[i, j]
    tl, tr, bl, br = (codes[idx, k] for k in range(4))

    # A partition of unity that depends on nothing but the offset inside the tile,
    # so it is the same on both sides of every edge and takes the same value at
    # both ends of it.  Flat in the middle, so corners become patches and not cones.
    def edge(t):
        return render_smoothstep(np.clip((t - 0.5 + margin) / (2.0 * margin), 0.0, 1.0))

    a = edge(u)
    b = edge(v)
    weights = ((1.0 - a) * (1.0 - b), a * (1.0 - b), (1.0 - a) * b, a * b)

    canvas = Canvas(size, size, scale=1.0, y_up=False)
    srgb = np.zeros((size, size, 3), dtype=np.float64)
    height = np.zeros((size, size), dtype=np.float64)
    for w, code in zip(weights, (tl, tr, bl, br)):
        srgb += w[..., None] * colour_of[code]
        height += w * height_of[code]

    speck = tiling.periodic_noise((size, size), seed=int(seed) + 913, scale=2.5)
    # A hair of blur on the colour as well as on the relief.  The blend is
    # continuous but its *slope* jumps where a flat cell meets a graded one, and the
    # eye reads that kink as an edge however equal the two sides are; a pixel of
    # blur is enough to stop it reading as one, and being periodic it costs nothing.
    srgb = tiling.smooth(srgb, 1.0)
    srgb *= (1.0 - grain + 2.0 * grain * speck)[..., None]
    # The relief is smoothed before it is lit, and the grain on it is gentle.  A
    # bilinear blend is continuous but its *slope* jumps at every cell edge, and a
    # normal computed from an unsmoothed height turns that jump into a visible edge
    # -- which is how a continuous field ends up looking tiled anyway.
    height = tiling.smooth(height, 1.4) * float(relief) + (speck - 0.5) * 0.12
    canvas.over_srgb(np.clip(tiling.lit(srgb, height, y_up=False), 0.0, 1.0),
                     np.ones((size, size), dtype=np.float64))
    return canvas


def render_smoothstep(t):
    """The usual cubic ease, kept local so the drawing has no hidden imports."""
    return t * t * (3.0 - 2.0 * t)
