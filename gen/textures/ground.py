"""Grounds: the six materials a map tile is made of, as a table of numbers.

A map draws ground as a flat biome colour, and at a hex or two that is right: the
question a map answers is *what is this hex*, not *what does it feel like*.  The moment
the camera comes down -- a battle map, a site screen, a tactical view -- the colour stops
being enough, because what the eye reads as a material is not its hue but the way the
light breaks across it.  Sand and dry grass are both a pale olive in a satellite
photograph; what separates them is that sand has ripples and grass has tufts, and both of
those are *relief*.

So a ground here is a height field with a colour on it, and the six are one painter over
a table.  The table mixes four primitive layers by weight:

* ``lump`` -- isotropic mottling at a chosen feature size: clods, tufts, drifts;
* ``ribbon`` -- the same noise stretched along one axis: sand ripples, grass laid over,
  the grain of anything soft;
* ``facet`` -- periodic Voronoi cells: slabs of rock, pebbles on a shore, each with its
  own tone, a top that is a plateau or a dome, and a bevel where the light catches the
  crack;
* ``grit`` -- one random value per pixel, which is what a close-up material has and a
  painter's flat fill does not.

The weights *are* the material.  Rock is ``facet`` with almost no colour variation; sand
is ``ribbon`` and a little ``lump``; snow is a big soft ``lump`` with the relief turned
down so the light barely finds it.  A seventh ground is a row, which is the test the
table has to pass: render every row and hold it to the seam like any other tile.

``FOR_BIOME`` says which material a map biome wears, and it is a separate table on
purpose: the map says what a hex *is*, and this says what it is made of.  Two biomes can
share a material -- plains, steppe, forest and marsh are all grass -- and one material can
be two biomes at different colours, because hills and mountains are both rock.  ``dirt``
is the one material no biome wears: it is the ground under a road, a camp or a ploughed
field, and a map that wants it knows why.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from core import palette
from core.canvas import Canvas

from . import tile as tiling

__all__ = ["FOR_BIOME", "GROUNDS", "Ground", "facets", "fields", "names", "texture"]


class Ground(NamedTuple):
    """One material: a colour, and the weights of the layers that make its surface.

    ``base`` is OKLCH with the hue in turns, exactly like the boards' wood, so a material
    of a different colour is a different tuple rather than a different function.
    ``drop`` and ``turn`` are how far the tone moves the lightness and the hue at either
    end of the layer mix.  ``relief`` is how tall the whole thing stands, in texels, and
    ``bevel`` is how much of a cell's width the drop to a crack takes -- a slab has a
    narrow one and a cobble a wide one.  ``dome`` is the last of the stone's three
    numbers: 0 is a plate that stays flat until its bevel, and 1 is a pebble that is
    round all the way to the crack, so the same facets make flagstones or gravel.
    """

    key: str
    base: tuple
    drop: float
    turn: float
    lump: float
    lump_scale: float
    ribbon: float
    ribbon_scale: float
    ribbon_stretch: float
    facet: float
    cells: int
    bevel: float
    dome: float
    grit: float
    relief: float


#: The six, in the order a sheet shows them: the soft grounds first, the hard ones after.
#: Every number is a *measurement of what the material looks like at a tile*: a feature
#: size of 4 px on a 64 px tile is a clod, and 20 px is a snowdrift, which is the whole
#: difference between the two rows that are otherwise the same soft noise.  The hues are
#: turns of OKLCH and sit where the map's own biomes sit -- plains is 0.353, desert 0.262,
#: mountain 0.296, snow 0.663 -- so a tile of ground and the hex it covers are the same
#: family of colour.  They are darker than the map's fills on purpose: a map is read from
#: above at a distance, and a hex a figure stands on is read from a metre away.
GROUNDS = (
    Ground("grass", base=(0.55, 0.100, 0.350), drop=0.13, turn=0.030,
           lump=1.00, lump_scale=6.0, ribbon=0.70, ribbon_scale=2.6, ribbon_stretch=3.5,
           facet=0.00, cells=1, bevel=0.00, dome=0.0, grit=0.22, relief=4.2),
    Ground("dirt", base=(0.38, 0.050, 0.140), drop=0.10, turn=0.020,
           lump=1.00, lump_scale=5.0, ribbon=0.00, ribbon_scale=3.0, ribbon_stretch=1.0,
           facet=0.25, cells=8, bevel=0.16, dome=0.5, grit=0.22, relief=3.6),
    Ground("sand", base=(0.78, 0.062, 0.260), drop=0.07, turn=0.012,
           lump=0.40, lump_scale=16.0, ribbon=1.00, ribbon_scale=10.0, ribbon_stretch=6.0,
           facet=0.00, cells=1, bevel=0.00, dome=0.0, grit=0.10, relief=3.2),
    Ground("rock", base=(0.56, 0.008, 0.290), drop=0.19, turn=0.010,
           lump=0.30, lump_scale=20.0, ribbon=0.00, ribbon_scale=3.0, ribbon_stretch=1.0,
           facet=1.00, cells=3, bevel=0.06, dome=0.0, grit=0.10, relief=7.0),
    Ground("snow", base=(0.90, 0.014, 0.630), drop=0.10, turn=0.020,
           lump=1.00, lump_scale=20.0, ribbon=0.30, ribbon_scale=8.0, ribbon_stretch=3.0,
           facet=0.00, cells=1, bevel=0.00, dome=0.0, grit=0.10, relief=3.5),
    Ground("shore", base=(0.64, 0.050, 0.250), drop=0.14, turn=0.020,
           lump=0.80, lump_scale=7.0, ribbon=0.20, ribbon_scale=4.0, ribbon_stretch=2.0,
           facet=0.60, cells=12, bevel=0.20, dome=1.0, grit=0.16, relief=3.0),
)

#: Which ground a map biome wears.  The water biomes are absent because water is not
#: ground, and ``dirt`` is absent because no biome is made of it -- see the module.
FOR_BIOME = {
    "shore": "shore",
    "desert": "sand",
    "steppe": "grass",
    "plains": "grass",
    "marsh": "grass",
    "forest": "grass",
    "hills": "rock",
    "mountain": "rock",
    "snow": "snow",
}

_BY_KEY = {g.key: g for g in GROUNDS}


def names() -> tuple:
    """The materials this table knows, in sheet order."""
    return tuple(g.key for g in GROUNDS)


def _get(kind) -> Ground:
    """The row for a kind -- or the row itself, because a material *is* a row.

    Passing a row is how a game defines a seventh material without editing this file:
    ``rock._replace(key="slate", drop=0.08)`` is a slate, and it is painted by the same
    painter, held to the same seam and read by the same sidecar.
    """
    if isinstance(kind, Ground):
        return kind
    try:
        return _BY_KEY[str(kind)]
    except KeyError:
        raise ValueError(f"unknown ground {kind!r}; try one of {list(names())}") from None


def facets(size: int = 128, cells: int = 4, seed: int = 0, jitter: float = 0.7) -> dict:
    """A Voronoi diagram of ``cells x cells`` jittered points **on a torus**.

    Returns the ``id`` of each pixel's nearest cell, the distance in pixels to the
    nearest *border* (half the gap to the second-nearest cell), and a ``tone`` per cell,
    already looked up per pixel.

    The torus is the whole point.  A Voronoi drawn on a square and cropped in the middle
    of a cell repeats the same pattern tile after tile, and the eye finds that faster
    than it finds a line -- which is a worse seam than the one it set out to avoid.  The
    distance here is computed on the wrap, so a slab that leaves the right edge is the
    same slab coming back on the left, and the crack between them is where the crack
    would be.  ``edge`` is what gives a slab its flat top: half the gap to the second
    cell is zero on the border and grows inward, so clipping it at a bevel is a plate of
    stone and not a bump.  It is the distance to the bisector with the *second-nearest*
    site, which near a vertex can be a pixel early -- the second site's cell need not
    reach this one -- and is never late, and early is what a plateau wants.

    ``jitter`` is how far a point may sit from the middle of its cell, as a fraction of
    the cell: at 0 the diagram is a grid of squares and at 1 a point can reach a corner.
    One cell has no bisector and therefore no crack, so its ``edge`` is infinite rather
    than a distance; a caller that clips it gets the whole tile as one flat plate, which
    is the honest answer to "where is the border of the only cell".

    Every cell is measured against every pixel, ``cells ** 2`` passes over the tile.  The
    obvious speed-up is the nine cells around the one a pixel falls in -- and it is wrong:
    a point jittered towards a corner can be nearer, across a border, than the pixel's own
    point is, so the search has to know which cells to look at and being one cell out is a
    kink in a crack that nobody can find later.  At the sizes a ground tile is made in,
    the honest loop is also the fast one.
    """
    size, cells = int(size), int(cells)
    if size < 2:
        raise ValueError("a ground is drawn at least two pixels across")
    if cells < 1:
        raise ValueError("a Voronoi needs at least one cell")
    if not 0.0 <= float(jitter) < 1.0:
        raise ValueError("jitter is a fraction of a cell, so it stays below one")

    rng = np.random.default_rng(int(seed) ^ 0x46414345)      # 'FACE'
    rows, cols = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
    cx = (cols + 0.5 + float(jitter) * (rng.random((cells, cells)) - 0.5)) / cells
    cy = (rows + 0.5 + float(jitter) * (rng.random((cells, cells)) - 0.5)) / cells
    tone = rng.random((cells, cells))

    ys = (np.arange(size)[:, None] + 0.5) / size
    xs = (np.arange(size)[None, :] + 0.5) / size

    d1 = np.full((size, size), np.inf)
    d2 = np.full((size, size), np.inf)
    near = np.zeros((size, size), dtype=np.int64)
    for i in range(cells):
        for j in range(cells):
            ux = xs - cx[i, j]
            uy = ys - cy[i, j]
            # The torus: the point is also its own image a whole tile away, so the
            # displacement is the shortest one of the three on each axis.
            ux -= np.round(ux)
            uy -= np.round(uy)
            d = np.hypot(ux, uy) * size
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            near = np.where(closer, i * cells + j, near)
            d1 = np.where(closer, d, d1)

    edge = np.maximum(0.5 * (d2 - d1), 0.0)
    return {"id": near, "edge": edge, "tone": tone.reshape(-1)[near], "table": tone}


def _noise(size: int, seed: int, feature: float, stretch: float = 1.0):
    """Periodic noise whose features read as ``feature`` pixels across.

    ``tile.periodic_noise`` takes a Gaussian *sigma*, so a table written in sigmas is a
    table of blurs.  What a material is written in is the size the eye sees -- a clod is
    four pixels, a snowdrift is twenty -- and the two are about a factor apart: a field
    smoothed by a Gaussian of sigma ``s`` decorrelates over roughly ``2 s``, and reads as
    lumps of about ``3 s``.  The conversion lives here, once, so the table can stay in the
    units of the thing it describes.
    """
    return tiling.periodic_noise((size, size), seed=seed,
                                 scale=max(float(feature) / 3.0, 0.35), stretch=stretch)


def fields(kind, seed: int = 0, size: int = 128, zoom: float = 1.0) -> dict:
    """The material's tone and height on the pixel grid, before anything is lit.

    ``kind`` is a name from the table or a row of your own.  ``tone`` is ``0..1`` across
    the material -- the number the colour is built from -- and ``height`` is in texels,
    which is the number the light is built from.  Kept apart from the painting so a test
    can ask what makes two materials different without going through a raster and a light.

    ``zoom`` scales the material's *features* and leaves the tile alone: at 2 the tufts,
    the ripples and the plates are twice as wide in texels, which is what drawing the same
    ground under a hex twice as large needs.  It multiplies the two feature sizes; the
    facets are counted per tile and therefore already scale with the canvas, and the grit
    stays one value per pixel by definition.  The default is the table as written, bit for
    bit, because multiplying by one is not a computation.
    """
    g = _get(kind)
    size = int(size)
    zoom = float(zoom)
    if not zoom > 0.0:
        raise ValueError(f"a ground is drawn at a positive zoom, not {zoom!r}")
    rng = np.random.default_rng((int(seed) * 2654435761 + 0x47524E44) & 0xFFFFFFFF)  # 'GRND'
    lump = (_noise(size, int(seed) + 101, g.lump_scale * zoom)
            if g.lump > 0 else np.full((size, size), 0.5))
    ribbon = (_noise(size, int(seed) + 211, g.ribbon_scale * zoom,
                     stretch=g.ribbon_stretch)
              if g.ribbon > 0 else np.full((size, size), 0.5))
    grit = rng.random((size, size))
    fac = facets(size=size, cells=g.cells, seed=int(seed) + 307) if g.facet > 0 else None
    facet_tone = fac["tone"] if fac is not None else 0.5

    total = max(g.lump + g.ribbon + g.facet, 1e-9)
    tone = (lump * g.lump + ribbon * g.ribbon + facet_tone * g.facet) / total
    tone = tone * (1.0 - g.grit) + grit * g.grit

    height = g.relief * ((lump - 0.5) * g.lump + (ribbon - 0.5) * g.ribbon) / total
    if fac is not None:
        # A slab is flat until its bevel; a pebble is round all the way to the crack.
        # ``sin`` is the dome -- flat at the top, steep at the rim -- and ``dome`` is the
        # material's own answer to "is this stone or is this gravel".  The bevel is a
        # fraction of a cell and never under a pixel, so the crack stays a crack at dpi.
        bevel = max(size / max(g.cells, 1) * g.bevel, 0.75)
        r = np.clip(fac["edge"] / bevel, 0.0, 1.0)
        top = (1.0 - g.dome) * r + g.dome * np.sin(0.5 * np.pi * r)
        height = height + g.relief * g.facet / total * (top - 0.5)
    height = height + (grit - 0.5) * (2.0 * g.grit)
    return {"tone": tone, "height": height, "facets": fac, "ground": g}


def texture(kind, seed: int = 0, size: int = 128, zoom: float = 1.0) -> Canvas:
    """A finished ground tile: the material lit by the project's one light.

    ``kind`` is a name from the table or a row of your own; a row is how a seventh
    material is added without touching this file.  Opaque, like a floor: a map tile with
    holes in it is a map tile that will show the thing behind it, and there is no thing
    behind it.  ``zoom`` is the feature scale :func:`fields` documents.
    """
    f = fields(kind, seed=seed, size=size, zoom=zoom)
    g = f["ground"]
    size = int(size)

    t = f["tone"]
    light = g.base[0] + g.drop * (t - 0.5)
    chroma = np.clip(g.base[1] * (0.75 + 0.5 * t), 0.0, 0.4)
    hue = g.base[2] + g.turn * (t - 0.5) * 2.0
    # ``fit_gamut`` walks a list of colours, so the field goes through it one row at a
    # time: flattened, fitted, put back -- which is also how the boards build a table.
    flat = np.stack([light.ravel(), chroma.ravel(), hue.ravel()], axis=-1)
    srgb = palette.oklch_to_srgb(flat).reshape(size, size, 3)

    lit = tiling.lit(np.clip(srgb, 0.0, 1.0), f["height"], y_up=False)
    canvas = Canvas(size, size, scale=1.0, y_up=False)
    canvas.over_srgb(np.clip(lit, 0.0, 1.0), np.ones((size, size), dtype=np.float64))
    return canvas
