"""A map at close range: the hex wears a ground, not a biome colour.

A strategic map answers *what is this hex*, and at fourteen pixels the honest answer
is a colour.  The moment the camera comes down -- a battle map, a site screen, a
tactical view -- a hex becomes a place, and what the eye reads as a place is not its
hue but the way the light breaks across it.  ``gen.textures.ground`` paints six
materials as height fields with a colour on them; this module puts them under the
hexes.

Two decisions hold the picture together.

**One light.**  The material's height is added to the height the hex relief
interpolates, and the single Lambert pass in ``paint.terrain_layer`` sees both.  A
material lit by its own light and then laid on a map lit by the project's light is two
lights, and two lights make a collage.

**Colour is modulated, never replaced.**  The biome colour is the map's claim about
what the hex *is*; the material's tone is the claim about what it is *like*.  The tone
moves the colour by ``mix`` either way, so a grass hex is still the map's green and a
rock hex is still the map's grey -- the difference between them is now where the light
lands.

The material is sampled in **world** coordinates, one texel to the pixel the site is
drawn at, so the pattern belongs to the map and not to the canvas.  Its features are
scaled by ``grid.size / REF``: a tuft is the same fraction of a hex at every size,
which is what makes a site drawn twice as large the same site rather than a magnified
one.  ``REF`` is the hex size the table in ``gen.textures.ground`` was tuned at.

Water has no ground.  ``ground.FOR_BIOME`` leaves the water biomes out on purpose, so
the sea is the depth ramp it is on the strategic map, pixel for pixel -- a site that
painted its water would be a second, quieter sea.

The dependency points one way: this module imports ``gen.textures``, and
``gen.textures`` knows nothing about maps.  It is the one place the two meet.
"""
from __future__ import annotations

import numpy as np

from core import render
from core.canvas import Canvas

from ..textures import ground as _ground
from . import autotile, paint, terrain
from .grid import HexGrid

__all__ = ["MIX", "REF", "RELIEF", "SPAN", "TERRAIN_RELIEF", "site_image", "worn"]

#: The hex size, in pixels, the ground table's feature sizes were tuned at.  A material
#: says ``lump_scale = 20`` texels, and at this zoom a texel is a pixel.
REF = 56.0

#: How many hexes one tile of material covers.  This is the material's period, not its
#: feature size: at six hexes the eye cannot find the repeat before it finds the hexes.
SPAN = 6.0

#: How far the material's tone may move the biome colour, either way.
MIX = 0.40

#: The material's height added to the hex relief, in texels per pixel of the tile.
RELIEF = 1.15

#: The map's own relief, turned down from the strategic map's 4.0 for the reason the
#: docstring gives: at this zoom the six-triangle crease is a folded sheet.
TERRAIN_RELIEF = 0.9


def worn(seed: int = 0, span: float = SPAN, mix: float = MIX,
         relief: float = RELIEF):
    """A ground hook for ``paint.terrain_layer``: ``(shade, height)`` per pixel.

    ``span`` is the material's period in hexes, ``mix`` how far its tone may move the
    biome colour, and ``relief`` how many pixels one texel of its height becomes.  The
    hook is a closure so the demo can hand the same numbers to the picture and to the
    sidecar; the sampler itself is a pure function of the seed, the grid and the
    canvas, which is what keeps a site reproducible.
    """
    mix = float(mix)
    relief = float(relief)

    def sampler(canvas: Canvas, grid: HexGrid, f: dict):
        X, Y = canvas.grid()
        cell = grid.cell(X, Y)
        r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
        c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
        biome = f["biome"][r, c]

        # One field per material the map can wear, generated once.  The order is the
        # sorted set of kinds, so the seeds -- and the picture -- do not depend on the
        # order of the ``FOR_BIOME`` table.
        kinds = sorted(set(_ground.FOR_BIOME.values()))
        slot = {kind: i for i, kind in enumerate(kinds)}
        pick = np.zeros(len(terrain.BIOMES), dtype=np.int64)
        for name, kind in _ground.FOR_BIOME.items():
            pick[terrain.INDEX[name]] = slot[kind]

        # The tile is ``span`` hexes wide, so its size follows the zoom: one texel is
        # one pixel, and nothing is minified or magnified on the way to the screen.
        size = max(int(round(float(span) * grid.size)), 16)
        tone = np.ones((len(kinds), size, size), dtype=np.float64)
        height = np.zeros_like(tone)
        for i, kind in enumerate(kinds):
            field = _ground.fields(kind, seed=int(seed) + 977 * i, size=size,
                                   zoom=grid.size / REF)
            tone[i] = field["tone"]
            height[i] = field["height"]

        # World coordinates, wrapped: the phase belongs to the map's origin and not to
        # the canvas, so a wider margin or a taller site does not slide the material.
        u = np.mod(np.floor(X).astype(np.int64), size)
        v = np.mod(np.floor(Y).astype(np.int64), size)
        which = pick[biome]
        shade = 1.0 - mix + 2.0 * mix * tone[which, v, u]
        # A texel is a pixel at ``REF``, and the material's features are scaled with
        # the hex; the height has to scale with them, or a crack that is a cliff at one
        # zoom becomes a wrinkle at twice the size.
        return shade, relief * (grid.size / REF) * height[which, v, u]

    return sampler


def site_image(seed: int = 0, cols: int = 8, rows: int = 6, size: float = REF,
               margin: float = 14.0, span: float = SPAN, mix: float = MIX,
               relief: float = RELIEF, terrain_relief: float = TERRAIN_RELIEF,
               coast: float = 0.10, inland: float = 0.045,
               lattice: float = 0.012, grounds: bool = True, **kw) -> Canvas:
    """One site: terrain, its borders, and the ground every hex wears.

    ``grounds=False`` is the same site without the material -- the flat biome colour the
    strategic map draws -- which is the comparison the demo makes and the test uses to
    show that the sea does not move.

    The ink widths are fractions of the hex: a border is a length in hexes, so a site at
    four times the hex size gets four times the ink and reads the same.  ``coast`` is the
    land's edge, ``inland`` the biome border inside the land, and ``lattice`` the grid,
    which is thin because on a site it is a ruler rather than a subject.

    ``terrain_relief`` is the map's own relief, and it is turned down from the strategic
    map's 4.0.  At fourteen pixels the six-triangle crease of a hex is a hint of a slope;
    at fifty-six it is a folded sheet, and the subject of a site is the ground, not the
    interpolant.  The hex still has its height -- it is simply no longer shouting.
    """
    grid = HexGrid(cols, rows, size)
    f = terrain.fields(grid, seed=seed, **kw)
    w, h = grid.extent
    canvas = Canvas(int(np.ceil(w + 2.0 * margin)), int(np.ceil(h + 2.0 * margin)),
                    scale=1.0)
    cell = grid.cell(*canvas.grid())

    hook = worn(seed=seed, span=span, mix=mix, relief=relief) if grounds else None
    paint.terrain_layer(canvas, grid, f, seed=seed, relief=terrain_relief,
                        ground=hook)

    # The coast is the land's edge, and the biome border is the same tileset asked of
    # the land alone -- `within` is what keeps the shore the coast's business.
    edge = autotile.borders((~f["water"]).astype(np.int64), grid, outside=0)
    autotile.border_layer(canvas, grid, edge, cell, paint.INK, width=coast, alpha=0.85)
    if inland > 0.0:
        rim = autotile.borders(f["biome"], grid, within=~f["water"])
        autotile.border_layer(canvas, grid, rim, cell, paint.INK, width=inland,
                              alpha=0.40)

    if lattice > 0.0:
        r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
        c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
        weight = np.where(f["water"][r, c], 0.08, 0.22)
        a = render.coverage(cell.edge_distance() - 0.5 * lattice * grid.size, canvas.dx)
        canvas.over_srgb(paint.INK, a * weight * cell.ok)

    return canvas
