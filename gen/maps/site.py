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

from core import noise, render
from core.canvas import Canvas

from ..textures import ground as _ground
from . import autotile, paint, terrain
from .grid import HexGrid

__all__ = ["MIX", "REF", "RELIEF", "SPAN", "TERRAIN_RELIEF", "site_image", "worn"]

REF = 56.0

SPAN = 6.0

MIX = 0.40

RELIEF = 1.15

TERRAIN_RELIEF = 0.9


def material(seed: int = 0, size: int = 128, zoom: float = 1.0,
             mix: float = MIX):
    """The ground table as two samplers: ``(shade, lift)``, both ``f(X, Y, biome)``.

    ``shade`` is the multiplier the material's tone puts on the biome colour and
    ``lift`` is its height in texels.  They are kept apart because a caller that only
    wants the colour -- an isometric board, a minimap -- should not pay for a height
    it will not light, and a caller that wants both should get both from one table.

    ``size`` is the material's tile in texels.  The caller makes it ``span`` hexes
    wide, because only the caller knows the hex size; at that width one texel is one
    pixel and nothing is minified on the way to the screen.  ``zoom`` is the feature
    scale :mod:`gen.textures.ground` documents.  The pattern is indexed by **world**
    coordinates, wrapped, so it belongs to the map's origin and not to the canvas.

    One field per material the map can wear is generated once.  The order is the
    sorted set of kinds, so the seeds -- and the picture -- do not depend on the order
    of the ``FOR_BIOME`` table.
    """
    size = int(size)
    zoom = float(zoom)
    mix = float(mix)
    kinds = sorted(set(_ground.FOR_BIOME.values()))
    slot = {kind: i for i, kind in enumerate(kinds)}
    pick = np.zeros(len(terrain.BIOMES), dtype=np.int64)
    for name, kind in _ground.FOR_BIOME.items():
        pick[terrain.INDEX[name]] = slot[kind]

    tone = np.ones((len(kinds), size, size), dtype=np.float64)
    height = np.zeros_like(tone)
    for i, kind in enumerate(kinds):
        field = _ground.fields(kind, seed=int(seed) + 977 * i, size=size, zoom=zoom)
        tone[i] = field["tone"]
        height[i] = field["height"]

    def pick_texel(X, Y, biome):
        u = np.mod(np.floor(X).astype(np.int64), size)
        v = np.mod(np.floor(Y).astype(np.int64), size)
        which = pick[biome]
        return which, v, u

    def shade(X, Y, biome):
        which, v, u = pick_texel(X, Y, biome)
        return 1.0 - mix + 2.0 * mix * tone[which, v, u]

    def lift(X, Y, biome):
        which, v, u = pick_texel(X, Y, biome)
        return height[which, v, u]

    return shade, lift


def worn(seed: int = 0, span: float = SPAN, mix: float = MIX,
         relief: float = RELIEF):
    """A ground hook for ``paint.terrain_layer``: ``(shade, height)`` per pixel.

    ``span`` is the material's period in hexes, ``mix`` how far its tone may move the
    biome colour, and ``relief`` how many pixels one texel of its height becomes.  The
    hook is a closure so the demo can hand the same numbers to the picture and to the
    sidecar; the sampler itself is a pure function of the seed, the grid and the
    canvas, which is what keeps a site reproducible.

    The material's *height* is scaled with the hex: a texel is a pixel at ``REF``, and
    the features are scaled with the zoom, so a crack that is a cliff at one zoom
    becomes a wrinkle at twice the size unless the height follows it.
    """
    mix = float(mix)
    relief = float(relief)

    def sampler(canvas: Canvas, grid: HexGrid, f: dict):
        X, Y = canvas.grid()
        cell = grid.cell(X, Y)
        r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
        c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
        biome = f["biome"][r, c]

        size = max(int(round(float(span) * grid.size)), 16)
        shade, lift = material(seed=seed, size=size, zoom=grid.size / REF, mix=mix)
        return shade(X, Y, biome), relief * (grid.size / REF) * lift(X, Y, biome)

    return sampler


def _river_field(canvas: Canvas, grid: HexGrid, f: dict, courses,
                 threshold: float, width: float = 0.17,
                 depth: float = 0.12):
    """A river as a channel: the carved height and the water surface, per pixel.

    The strategic map draws a river as a stroke, because at fourteen pixels a stroke is
    all a river can be.  At fifty-six a stroke is a blue line lying *on* the ground, and
    the one thing a close-up cannot afford is a map symbol pretending to be a place.  So
    the course is cut into the height field the light is cast on -- a shallow groove with
    soft banks -- and the water is painted in the groove, below the lip.

    Returns ``(carve, water, bank)``: a height in pixels to add to the terrain
    (negative), the coverage of the water surface, and the coverage of the lip that
    separates them.  All three are zero over the sea: a river ends at the coast it
    reaches, and the sea is drawn by the depth ramp, not by this.
    """
    X, Y = canvas.grid()
    cell = grid.cell(X, Y)
    r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
    c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
    dry = cell.ok & ~f["water"][r, c]

    carve = np.zeros(X.shape, dtype=np.float64)
    water = np.zeros(X.shape, dtype=np.float64)
    bank = np.zeros(X.shape, dtype=np.float64)

    for course in courses:
        grew = max(np.sqrt(float(course["flux"]) / max(float(threshold), 1e-9)), 0.55)
        half = max(0.5 * float(width) * grid.size * grew, 1.4)
        pts = [tuple(map(float, p)) for p in np.asarray(course["points"],
                                                       dtype=np.float64)]
        if len(pts) < 2:
            continue

        mouth = course.get("mouth")
        if mouth is not None:
            mr, mc = int(np.rint(mouth[0])), int(np.rint(mouth[1]))
            if 0 <= mr < grid.rows and 0 <= mc < grid.cols:
                for dirc in range(6):
                    nr, nc = int(grid.nbr[mr, mc, dirc, 0]), int(grid.nbr[mr, mc, dirc, 1])
                    if nr >= 0 and nc >= 0 and f["water"][nr, nc]:
                        edge = ((grid.centers[mr, mc] + grid.centers[nr, nc]) * 0.5)
                        pts.append((float(edge[0]), float(edge[1])))
                        break
        d = np.full(X.shape, np.inf)
        for a, b in zip(pts[:-1], pts[1:]):
            vx, vy = b[0] - a[0], b[1] - a[1]
            L2 = vx * vx + vy * vy
            if L2 < 1e-12:
                seg = np.hypot(X - a[0], Y - a[1])
            else:
                t = np.clip(((X - a[0]) * vx + (Y - a[1]) * vy) / L2, 0.0, 1.0)
                seg = np.hypot(X - (a[0] + t * vx), Y - (a[1] + t * vy))
            d = np.minimum(d, seg)

        source = np.hypot(X - pts[0][0], Y - pts[0][1])
        fade = np.clip(source / (1.3 * half), 0.0, 1.0)

        profile = np.clip(1.0 - (d / (half * 1.6)) ** 2, 0.0, 1.0)
        carve = np.minimum(carve, -float(depth) * grid.size * profile)
        water = np.maximum(water, np.clip((half - d) / (0.40 * half), 0.0, 1.0) * fade)
        bank = np.maximum(bank, np.clip(1.0 - np.abs(d - 1.3 * half) / (0.35 * half),
                                        0.0, 1.0) * fade)

    keep = dry.astype(np.float64)
    return carve * keep, water * keep, bank * keep


def site_image(seed: int = 0, cols: int = 8, rows: int = 6, size: float = REF,
               margin: float = 14.0, span: float = SPAN, mix: float = MIX,
               relief: float = RELIEF, terrain_relief: float = TERRAIN_RELIEF,
               coast: float = 0.10, inland: float = 0.045,
               lattice: float = 0.012, cliff: float = 0.55,
               river: float = 0.92, river_fraction: float | None = None,
               river_width: float = 0.17, river_depth: float = 0.06,
               shore: float = 0.052, grounds: bool = True, **kw) -> Canvas:
    """One site: terrain, its borders, and the ground every hex wears.

    ``grounds=False`` is the same site without the material -- the flat biome colour the
    strategic map draws -- which is the comparison the demo makes and the test uses to
    show that the sea does not move.

    The ink widths are fractions of the hex: a border is a length in hexes, so a site at
    four times the hex size gets four times the ink and reads the same.  ``coast`` is the
    land's edge, ``inland`` the biome border inside the land, and ``lattice`` the grid,
    which is thin because on a site it is a ruler rather than a subject.

    ``cliff`` is the crease the strategic map draws where the ground drops: at close
    range a slope is a surface with a normal, but a *step* is still a step, and the
    quantile threshold is what makes it visible without inventing a contour line.
    ``shore`` is the wet band on the land side of the coast -- ground that the sea has
    only just left -- and it is deliberately narrow, because a beach is read from its
    colour and a wet stripe is what says the water is close.

    ``terrain_relief`` is the map's own relief, and it is turned down from the strategic
    map's 4.0.  At fourteen pixels the six-triangle crease of a hex is a hint of a slope;
    at fifty-six it is a folded sheet, and the subject of a site is the ground, not the
    interpolant.  The hex still has its height -- it is simply no longer shouting.

    Rivers are the same hydrology the strategic map draws, at the same seed, but not the
    same *stroke*: at close range the course is carved into the height field and the
    water is painted in the groove, because a stroke would be a map symbol lying on the
    ground.  ``river`` is the water's opacity, ``river_fraction`` the drainage quantile
    that decides which courses exist, ``river_width`` the half-width in hexes and
    ``river_depth`` how deep the groove is cut, also in hexes.  A site is a *piece* of
    the map, so a river that reaches its edge continues off it.
    """
    grid = HexGrid(cols, rows, size)
    f = terrain.fields(grid, seed=seed, **kw)
    w, h = grid.extent
    canvas = Canvas(int(np.ceil(w + 2.0 * margin)), int(np.ceil(h + 2.0 * margin)),
                    scale=1.0)
    cell = grid.cell(*canvas.grid())
    r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
    c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
    water = f["water"][r, c]

    courses = threshold = None
    carve = water_a = bank_a = None
    if river > 0.0 and (~f["water"]).any():
        courses, threshold = paint.river_courses(grid, f, river_fraction)
        if courses:
            carve, water_a, bank_a = _river_field(
                canvas, grid, f, courses, threshold,
                width=river_width, depth=river_depth)

    material_hook = (worn(seed=seed, span=span, mix=mix, relief=relief)
                     if grounds else None)
    if carve is not None:
        def hook(cnv, g, ff):
            if material_hook is None:
                return None, carve
            shade, height = material_hook(cnv, g, ff)
            return shade, height + carve
    else:
        hook = material_hook
    paint.terrain_layer(canvas, grid, f, seed=seed, relief=terrain_relief,
                        ground=hook)

    if cliff > 0.0:
        a = paint.cliff_alpha(canvas, grid, f, cell, strength=cliff)
        canvas.over_srgb(paint.INK, a * (~water) * cell.ok)

    if shore > 0.0:
        d = cell.sector
        nr = grid.nbr[r, c, d, 0]
        nc = grid.nbr[r, c, d, 1]
        across = np.where(nr >= 0,
                          f["water"][np.clip(nr, 0, grid.rows - 1),
                                     np.clip(nc, 0, grid.cols - 1)], False)
        wet = render.coverage(cell.edge_distance() - shore * grid.size, canvas.dx)
        canvas.over_srgb((0.13, 0.16, 0.14),
                         0.26 * wet * (~water) * across * cell.ok)

    edge = autotile.borders((~f["water"]).astype(np.int64), grid, outside=0)
    autotile.border_layer(canvas, grid, edge, cell, paint.INK, width=coast, alpha=0.85)
    if inland > 0.0:
        rim = autotile.borders(f["biome"], grid, within=~f["water"])
        autotile.border_layer(canvas, grid, rim, cell, paint.INK, width=inland,
                              alpha=0.40)

    if water_a is not None:

        canvas.over_srgb((0.30, 0.25, 0.18), 0.34 * bank_a)
        X, Y = canvas.grid()
        ripple = noise.fbm(X / 9.0, Y / 9.0, seed=seed + 991, freq=1.0, octaves=3)
        water_rgb = (np.array([0.20, 0.42, 0.52])[None, None, :]
                     * (0.86 + 0.26 * ripple)[..., None]
                     * (0.92 + 0.18 * water_a)[..., None])
        canvas.over_srgb(np.clip(water_rgb, 0.0, 1.0), float(river) * water_a)

    if lattice > 0.0:
        weight = np.where(water, 0.08, 0.22)
        a = render.coverage(cell.edge_distance() - 0.5 * lattice * grid.size, canvas.dx)
        canvas.over_srgb(paint.INK, a * weight * cell.ok)

    return canvas
