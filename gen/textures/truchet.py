"""Truchet tiles: the seamless texture with nothing to check.

A Truchet tile is a square with two arcs drawn on it, each joining two *midpoints*
of the square's edges.  There are two ways to place them -- arcs bulging out of one
diagonal, or out of the other -- and that is the whole alphabet.  The reason it is
the classic seamless texture is one sentence long:

> every arc of a Truchet tile ends at the midpoint of an edge, and the tile on the
> other side of that edge sends one of its own arcs to the same point.  The paths
> join whatever the neighbour is, and whatever the arrangement is.

So there is nothing to search, nothing to match and nothing to backtrack over: any
arrangement of the two orientations is seamless, and the pattern that comes out is
a set of paths that wander across the whole plane and either close on themselves or
run off the edge.  The generator picks an orientation for each grain with a seeded
random generator, and the seamlessness is a property of the geometry rather than of
the choice -- which is exactly the opposite of :mod:`gen.textures.wang`, where the
choice *is* the problem.

Two kinds are offered.  ``arcs`` is the classical one: quarter circles centred on
the tile's corners.  ``lines`` replaces each arc with two straight segments meeting
at the tile's centre, which gives the maze look; the connections are the same, so
the tiles are equally seamless, and the two kinds can even be mixed in one
arrangement without breaking anything.
"""
from __future__ import annotations

import numpy as np

from core import palette, render, sdf
from core.canvas import Canvas

from . import tile as tiling

__all__ = ["arrangement", "pattern", "texture"]

#: The two orientations, as the corner each arc is centred on, in tile units with
#: y pointing up: the first bulges out of the top-left and bottom-right corners,
#: the second out of the other diagonal.  Neighbouring tiles in an arrangement
#: meet along an edge, and both arcs that reach a shared edge meet at its middle.
_ORIENTATIONS = (((0.0, 1.0), (1.0, 0.0)), ((1.0, 1.0), (0.0, 0.0)))
#: Which two edge midpoints each orientation joins, for the straight-line kind.
_MATCHING = (((0.5, 1.0), (0.0, 0.5)), ((1.0, 0.5), (0.5, 0.0)))
#: The kinds of path a grain may draw.  Both connect the same edge midpoints, which
#: is what lets them be mixed in one arrangement without breaking the wrap.
_KINDS = ("arcs", "lines")


def arrangement(seed: int, grains: int, kinds=("arcs",), mix: float = 0.0):
    """A ``grains x grains`` grid of ``(orientation, kind index)``.

    ``mix`` is the chance that a grain switches to the next kind, so a texture can
    be all arcs, all lines, or a blend of both.
    """
    rng = np.random.default_rng(int(seed))
    grains = max(int(grains), 1)
    out = []
    for _ in range(grains):
        row = []
        for _ in range(grains):
            kind = 0
            if len(kinds) > 1 and rng.random() < float(mix):
                kind = int(rng.integers(1, len(kinds)))
            row.append((int(rng.integers(0, 2)), kind))
        out.append(row)
    return out


def pattern(seed: int = 0, size: int = 256, grains: int = 4, kinds=("arcs", "lines"),
            mix: float = 0.0, width: float = 0.16) -> dict:
    """The drawn geometry of a Truchet texture, as fields on a pixel grid.

    Returns the signed distance to the paths (negative inside them), the height of
    the carved relief, and the arrangement it came from -- everything the painter
    and the tests need, and no colour.
    """
    size = int(size)
    if size % grains:
        raise ValueError(f"a {size} px tile is not a whole number of {grains} grains")
    kinds = tuple(kinds) or ("arcs",)
    for name in kinds:
        if name not in _KINDS:
            raise ValueError(f"unknown Truchet kind {name!r}; try one of {sorted(_KINDS)}")
    cell = size // grains
    arr = arrangement(seed, grains, kinds=kinds, mix=mix)
    d = np.full((size, size), np.inf, dtype=np.float64)
    for i in range(grains):
        for j in range(grains):
            y0, x0 = i * cell, j * cell
            v, u = np.meshgrid(
                (np.arange(cell) + 0.5) / cell, (np.arange(cell) + 0.5) / cell,
                indexing="ij",
            )
            orientation, kind = arr[i][j]
            if kinds[kind % len(kinds)] == "arcs":
                part = None
                for cx, cy in _ORIENTATIONS[orientation]:
                    arc = np.abs(np.hypot(u - cx, v - cy) - 0.5)
                    part = arc if part is None else np.minimum(part, arc)
            else:
                a, b = _MATCHING[orientation]
                mid = (0.5, 0.5)
                part = np.minimum(sdf.segment(u, v, a, mid), sdf.segment(u, v, mid, b))
            d[y0:y0 + cell, x0:x0 + cell] = part
    # The paths are carved *in*: a groove of half-width ``width`` is pressed into
    # the surface, so the relief is negative inside it -- a ridge would be a
    # different material, and the walls of a groove light each other.
    shape = d - 0.5 * float(width)
    return {"distance": shape, "height": np.clip(shape, None, 0.0), "arrangement": arr}


def texture(seed: int = 0, size: int = 256, grains: int = 4, kinds=("arcs", "lines"),
            mix: float = 0.0, width: float = 0.16, relief: float = 0.55,
            background: str = "#6c6f78", ink: str = "#39404b", grain: float = 0.06,
            bevel: float = 0.5) -> Canvas:
    """A finished Truchet tile: a carved pattern on a grained surface.

    ``bevel`` is the softness of the carve in pixels: at 0 the paths are cut
    straight down, and larger values round the edges over, which is what stone
    looks like when it has been walked on.
    """

    size = int(size)
    cell = size // int(grains)
    fields = pattern(seed=seed, size=size, grains=grains, kinds=kinds, mix=mix, width=width)
    canvas = Canvas(size, size, scale=1.0, y_up=False)

    base = palette.hex_to_srgb(background)
    ink = palette.hex_to_srgb(ink)
    srgb = np.broadcast_to(base, (size, size, 3)).copy()

    # The surface first, so the carve is cut out of something rather than floated
    # over it.  The grain is periodic noise, not noise evaluated in pixel
    # coordinates: a grain that does not wrap breaks the seam that the pattern
    # itself keeps, which is the one way this texture could still fail.
    speck = tiling.periodic_noise((size, size), seed=int(seed) + 71, scale=2.5)
    srgb *= (1.0 - grain + 2.0 * grain * speck)[..., None]

    # The fields come back in tile units, where a nice round number like 0.16 is a
    # path width.  Pixels are what the painter wants, so scale once -- and then
    # ``relief`` is exactly the slope of the groove's wall, which is the number
    # worth choosing by eye.
    distance = fields["distance"] * cell
    height = fields["height"] * cell * float(relief)
    mask = render.coverage(distance - float(bevel), 1.0)
    # Both sides are shaded by the same groove height: the rim where the tool went
    # in and the floor it cut, lit by their own slopes.  That, and not the colour,
    # is what makes the paths read as grooves instead of as painted lines.
    floor = np.broadcast_to(ink, (size, size, 3))
    base_lit = tiling.lit(srgb, height, y_up=False)
    floor_lit = tiling.lit(floor, height, y_up=False, ambient=0.55, diffuse=0.75)
    canvas.over_srgb(np.clip(base_lit * (1.0 - mask)[..., None] + floor_lit * mask[..., None],
                             0.0, 1.0), np.ones((size, size), dtype=np.float64))
    return canvas
