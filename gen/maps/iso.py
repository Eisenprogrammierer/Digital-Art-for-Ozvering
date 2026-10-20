"""A board seen from the south: the same hexes, standing up.

The strategic map looks straight down because that is how a map is read, and a site
looks straight down because that is how a hex is inspected.  Neither is how a player
looks at a *place*: a game camera stands at the edge of the board and looks across it,
which is what this module draws.  The projection is orthographic and axonometric -- no
vanishing point, because a vanishing point is a perspective and a perspective makes two
equal hexes unequal -- with the ground plane squashed vertically and the height lifted
out of it::

    screen_x = x
    screen_y = -y * squash - height * rise

That is a camera looking from the south and above.  ``squash`` is the cosine of its
pitch and ``rise`` is the vertical scale of the relief; they are separate numbers
because a strategy board exaggerates height on purpose, and pretending the two are one
would make the exaggeration a perspective error instead of a choice.

**The tile is the unit.**  A top-down map is one raster pass over the whole canvas, so
its material can be sampled in world coordinates and the hex creases disappear.  A
board cannot be one pass: a hex has sides, and a side is a fact about that hex.  So the
hexes are drawn one at a time, far to near, and what keeps them from looking like
stamps is that the *material* under them is still sampled in world coordinates --
every top asks the same table the same question at the same place.  The seam is the
hex edge, which the lattice draws anyway; the pattern does not restart on it.

**Height is a side, not a tint.**  A hill on the strategic map is a lighter hex; on a
board a hill is a hex whose edge has a face.  Only the faces that turn towards the
camera are drawn -- the two southern ones -- and off the south edge of the map a face
drops to sea level, so the board is a slab with a cut edge and not a sheet of paper
with a picture on it.

The height is **terraced** before the faces are drawn.  A continuous elevation field
gave every edge a face a few pixels high, and a board where every hex is a plate on a
plate is a layer cake, not a landscape; quantising the land into a handful of levels
means two neighbours are usually on the *same* level and have no face at all, and the
faces that remain are steps the eye can name.  ``levels`` is the count, and one level is
the unit a cliff is measured in.  The tops stay flat for the same reason: a tile is a
tile, and the ground under it is the material.

The face is where ``dirt`` finally has a job.  The material is the one no biome wears,
and a cliff is what it is for: the ground's own tone is sampled along the face, so the
strata belong to the map's seed and not to a second noise field.  The colour is earth
rather than the biome's, because what a cliff *exposes* is not what grows on it.

The hero is a figure sprite from :mod:`gen.figures`, standing on the middle of a hex.
It is drawn after the row it stands in and before the rows in front of it, which is
the whole of the depth rule: a hill nearer the camera may hide it, a hill behind it
may not.  The sprite is a billboard with four cardinal facings -- profile, mirror,
front and back -- which is what a board with no yaw can honestly promise; a
three-quarter figure would promise a camera the projection does not have.
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.canvas import Canvas

from ..figures import figure
from ..textures import ground as _ground
from . import paint, site, terrain
from .grid import HexGrid

__all__ = ["HERO_SIZE", "RISE", "SQUASH", "hero_cell", "iso_image", "terraces"]

#: The vertical squash of the ground plane: the cosine of a camera pitch of about 52
#: degrees.  A board is read at a glance, so the hexes have to stay wide enough to
#: count and tall enough to stand a figure on.
SQUASH = 0.62

#: How far one unit of terrain height lifts the board, as a fraction of the hex size.
#: The map's land rises only about 0.2 above the sea level it was cut at, so a board
#: needs a strong vertical scale to show a hill at all: at 4.0 the highest ground stands
#: about 0.8 of a hex proud of the water, which is the exaggeration a strategy board has
#: always used.
RISE = 4.0

#: The sprite box of the hero, as a fraction of the hex size.  The drawn figure fills
#: about 0.92 of its box, so the hero is about one and a half hexes tall.
HERO_SIZE = 1.8

#: The lattice's ink and width, as fractions of the hex.  The board shows its hexes;
#: the line is a ruler, not a cage.
LATTICE = 0.016

#: How wide the foam is on the water side of a board's coast, as a fraction of the hex.
FOAM = 0.055


def _project(x, y, z, ox, oy, squash, rise):
    """World ``(x, y)`` and a height fraction to screen ``(sx, sy)``.

    ``rise`` is in pixels per unit of height and has already been scaled by the hex
    size; the height itself is the board's fraction above the sea.
    """
    return x + ox, -y * squash - z * rise + oy


def _cover(poly, x0, y0, x1, y1):
    """Coverage of a convex polygon over a pixel window.

    Returns the window's origin, its pixel-centre meshes, the signed distance from
    the polygon's edge (positive inside, in pixels), and the one-pixel-wide coverage.
    The half-plane distance is exact for a convex polygon, which every face here is.
    """
    poly = np.asarray(poly, dtype=np.float64)
    h, w = y1 - y0, x1 - x0
    px = np.arange(w, dtype=np.float64)[None, :] + x0 + 0.5
    py = np.arange(h, dtype=np.float64)[:, None] + y0 + 0.5
    px, py = np.broadcast_to(px, (h, w)), np.broadcast_to(py, (h, w))
    c = poly.mean(axis=0)
    d = np.full((h, w), np.inf)
    for k in range(len(poly)):
        ax, ay = poly[k]
        bx, by = poly[(k + 1) % len(poly)]
        nx, ny = by - ay, -(bx - ax)
        if nx * (c[0] - ax) + ny * (c[1] - ay) < 0.0:
            nx, ny = -nx, -ny
        length = float(np.hypot(nx, ny))
        if length < 1e-12:
            continue
        d = np.minimum(d, ((px - ax) * nx + (py - ay) * ny) / length)
    cov = np.clip(d + 1.0, 0.0, 1.0)
    return x0, y0, px, py, d, cov


def _window(poly, canvas_w, canvas_h):
    """The clipped integer pixel window around a projected polygon."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0 = max(int(np.floor(min(xs))) - 1, 0)
    y0 = max(int(np.floor(min(ys))) - 1, 0)
    x1 = min(int(np.ceil(max(xs))) + 1, canvas_w)
    y1 = min(int(np.ceil(max(ys))) + 1, canvas_h)
    return x0, y0, x1, y1


def terraces(f: dict, levels: int = 7) -> np.ndarray:
    """The board's height above the sea, quantised into ``levels`` steps.

    ``levels=0`` keeps the continuous field -- the height a map sees from above -- and
    ``levels=1`` is the flat sea plane with every land hex on it.  Anything higher
    rounds the land to that many terraces: two neighbours on the same terrace have no
    face between them, which is what keeps a board from being a pile of plates.  Water
    is always on the sea plane, whatever the count.
    """
    z = np.maximum(np.asarray(f["height"], dtype=np.float64) - float(f["sea"]), 0.0)
    levels = int(levels)
    if levels <= 0:
        return z
    if levels == 1:
        return np.zeros_like(z)
    if z.max() > 0.0:
        step = float(z.max()) / (levels - 1)
        z = np.round(z / step) * step
    return np.where(f["water"], 0.0, z)


def hero_cell(grid: HexGrid, f: dict, prefer: bool = True):
    """The ``(row, col)`` the demo's hero stands on, or ``None`` for a water world.

    The middle of the map, or the land nearest it -- and *not* the highest hex, because
    a hero on a summit is a postcard and a hero on the road is a game.  ``prefer``
    adds a small penalty to the dramatic biomes, so the nearest plain wins over a
    mountain the same distance away.  The choice is a pure function of the geography,
    which is what lets the sidecar name the hex the picture put him on.
    """
    if not (~f["water"]).any():
        return None
    dist = np.hypot(grid.centers[..., 0], grid.centers[..., 1]) / max(grid.size, 1e-9)
    penalty = np.zeros_like(dist)
    if prefer:
        cost = {
            "plains": 0.00, "steppe": 0.04, "forest": 0.08, "desert": 0.14,
            "hills": 0.20, "shore": 0.28, "marsh": 0.34, "mountain": 0.55,
            "snow": 0.70,
        }
        for name, value in cost.items():
            penalty = np.where(f["biome"] == terrain.INDEX[name], value, penalty)
    score = np.where(~f["water"], dist + penalty, np.inf)
    r, c = np.unravel_index(int(np.argmin(score)), score.shape)
    return int(r), int(c)


def _cliff_colour(seed, biome_table, zoom, lattice_size):
    """The earth a cliff exposes, as a function of world ``(x, y)`` and the biome.

    The dirt field is the material no biome wears, and the face is where it is worn;
    the biome's own colour is mixed in at a fifth, so a grass cliff is a greenish earth
    and a snow cliff is a cool one, without either becoming a wall of the top's colour.
    """
    field = _ground.fields("dirt", seed=int(seed) + 0x0C11F, size=lattice_size,
                           zoom=zoom)
    tone = field["tone"]
    earth = palette.hex_to_srgb("#8a7054")

    def at(x, y, biome):
        u = np.mod(np.floor(x).astype(np.int64), lattice_size)
        v = np.mod(np.floor(y).astype(np.int64), lattice_size)
        t = tone[v, u]
        rgb = 0.20 * biome_table[biome] + 0.80 * earth
        return rgb * (0.72 + 0.45 * t)[..., None]

    return at


def iso_image(
    seed: int = 0,
    cols: int = 12,
    rows: int = 9,
    size: float = 40.0,
    margin: float = 14.0,
    span: float = site.SPAN,
    mix: float = site.MIX,
    squash: float = SQUASH,
    rise: float = RISE,
    hero_size: float = HERO_SIZE,
    hero: tuple | None = None,
    hero_seed: int | None = None,
    cycle: str = "idle",
    phase: float = 0.0,
    facing: str = "right",
    lattice: float = LATTICE,
    foam: float = FOAM,
    levels: int = 7,
    **kw,
) -> Canvas:
    """One isometric board, with a hero standing on it.

    ``kw`` goes to :func:`gen.maps.terrain.fields`, so the geography is the same
    geography ``map`` and ``site`` draw at the same seed and shape.  ``hero`` is a
    ``(row, col)`` to stand the figure on; ``None`` asks :func:`hero_cell` for the
    nearest plain to the middle, and a named hex must be on the map and on land.
    ``hero_seed`` is the figure's own seed, so changing the map without changing the
    hero is a different number and not a different call.  ``facing`` is ``right`` or
    ``left`` across the screen, ``down`` towards the camera and ``up`` away from it;
    :data:`gen.figures.figure.FACINGS` is the whole set.

    ``squash`` and ``rise`` are the camera: the first is dimensionless and the second
    is a fraction of the hex size per unit of height, which is why the same board can
    be re-drawn at a different size and be the same board.  ``levels`` is the number of
    terraces the land is quantised into (0 keeps the continuous field, which is useful
    for a comparison and for a game with real slopes under its tiles; 1 is the flat sea
    plane).  ``lattice`` and ``foam`` are widths as fractions of the hex.

    Returns the board on a transparent canvas.  The caller puts it on a page, which is
    where a shadow belongs; a board that casts its own shadow into a transparent
    margin is a board with a grey halo in a game.
    """
    grid = HexGrid(cols, rows, size)
    f = terrain.fields(grid, seed=seed, **kw)
    z = terraces(f, levels)
    # ``rise`` is a fraction of the hex size at the API, and a number of pixels per
    # unit of height inside the projection.  The distinction is the whole difference
    # between a hill and a wrinkle: the map's heights are fractions, and a fraction
    # means nothing to a raster until the hex size says what it is a fraction of.
    rise = float(rise) * float(size)

    if hero is None:
        hero = hero_cell(grid, f)
    if hero is not None:
        hero = (int(hero[0]), int(hero[1]))
        if not (0 <= hero[0] < grid.rows and 0 <= hero[1] < grid.cols):
            raise ValueError(f"the hero's hex {hero} is off a {grid.rows}x{grid.cols} map")
        if f["water"][hero]:
            raise ValueError(f"the hero's hex {hero} is water; a figure needs land")
    if facing not in figure.FACINGS:
        raise ValueError(f"a hero faces {', '.join(figure.FACINGS)}; not {facing!r}")
    hero_seed = int(seed) if hero_seed is None else int(hero_seed)
    sprite = (figure.render(seed=hero_seed, cycle=cycle, phase=phase, facing=facing,
                            size=max(int(round(hero_size * size)), 16))
              if hero is not None else None)
    sprite_box = None
    if sprite is not None:
        ys, xs = np.nonzero(sprite.a > 0.01)
        if len(xs):
            sprite_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    # The board's own bounds, plus the hero's head: a hero on the back row must not
    # lose it to the margin.  Raw screen coordinates first; the offset comes after.
    raw = []
    for r in range(rows):
        for c in range(cols):
            for x, y in grid.corners_of(r, c):
                raw.append((x, -y * squash - z[r, c] * rise))
                raw.append((x, -y * squash))
    if hero is not None and sprite_box is not None:
        r, c = hero
        sx = grid.centers[r, c, 0]
        sy = -grid.centers[r, c, 1] * squash - z[r, c] * rise
        x0, y0, x1, y1 = sprite_box
        left = sx - (x0 + x1) * 0.5
        right = sx + (sprite.w - (x0 + x1) * 0.5)
        top = sy - (y1 - y0) - 2.0
        raw.extend([(left, top), (right, top), (left, sy + 2.0), (right, sy + 2.0)])
    minx = min(p[0] for p in raw)
    maxx = max(p[0] for p in raw)
    miny = min(p[1] for p in raw)
    maxy = max(p[1] for p in raw)
    ox = margin - minx
    oy = margin - miny
    canvas = Canvas(int(np.ceil(maxx - minx + 2.0 * margin)),
                    int(np.ceil(maxy - miny + 2.0 * margin)),
                    scale=1.0, y_up=False)

    material_size = max(int(round(float(span) * size)), 16)
    shade, _lift = site.material(seed=seed, size=material_size,
                                 zoom=size / site.REF, mix=mix)
    colours = terrain.colour_table()
    cliff_at = _cliff_colour(seed, colours, size / site.REF, material_size)
    width = max(float(lattice) * size, 0.7)
    foam_w = max(float(foam) * size, 1.0)

    def corners(r, c, zz):
        return [_project(x, y, zz, ox, oy, squash, rise)
                for x, y in grid.corners_of(r, c)]

    def draw_sides(r, c):
        own = z[r, c]
        for d in range(6):
            nr, nc = int(grid.nbr[r, c, d, 0]), int(grid.nbr[r, c, d, 1])
            if nr < 0 or nc < 0:
                # Off the board, the slab is cut down to sea level: the south and
                # west/east edges show a face, the north edge is behind the board.
                nz = 0.0
            else:
                nz = z[nr, nc]
            drop = (own - nz) * rise
            if drop < 0.55:
                continue
            # Only the faces that turn towards the camera: a hex edge whose
            # neighbour is nearer (smaller world y) is one of the two southern ones.
            if nr >= 0 and nc >= 0 and grid.centers[nr, nc, 1] >= grid.centers[r, c, 1]:
                continue
            if nr < 0 or nc < 0:
                # The east and west cut edges are seen exactly edge-on with no yaw;
                # their faces have no width on screen and are not worth drawing.
                (x0, y0) = grid.corners_of(r, c)[d]
                (x1, y1) = grid.corners_of(r, c)[(d + 1) % 6]
                if abs(x1 - x0) < 0.5:
                    continue
            top = corners(r, c, own)[d: d + 2] if d < 5 else [
                corners(r, c, own)[5], corners(r, c, own)[0]]
            bot = corners(r, c, nz)[d: d + 2] if d < 5 else [
                corners(r, c, nz)[5], corners(r, c, nz)[0]]
            poly = [top[0], top[1], bot[1], bot[0]]
            x0, y0, x1, y1 = _window(poly, canvas.w, canvas.h)
            if x1 <= x0 or y1 <= y0:
                continue
            _x, _y, px, py, dinside, cov = _cover(poly, x0, y0, x1, y1)
            # The material is sampled along the edge: x alone recovers the world x,
            # because the projection does not turn x; the edge gives the world y.
            (ax, ay), (bx, by) = grid.corners_of(r, c)[d], grid.corners_of(r, c)[(d + 1) % 6]
            wx = np.clip(px - ox, min(ax, bx), max(ax, bx))
            wy = np.where(abs(bx - ax) > 1e-9, ay + (wx - ax) * (by - ay) / (bx - ax),
                          (ay + by) * 0.5)
            srgb = cliff_at(wx, wy, f["biome"][r, c])
            # A vertical gradient: the top lip catches the sky, the base is in its own
            # shadow.  This is a second light in the sense that it is a gradient, and it
            # is the only one: the face's horizontal normal is turned away from the
            # project's light, so it would otherwise be uniformly black.
            ytop = np.minimum(top[0][1], top[1][1])
            ybot = np.maximum(bot[0][1], bot[1][1])
            frac = np.clip((py - ytop) / max(ybot - ytop, 1e-9), 0.0, 1.0)
            srgb = srgb * (1.02 - 0.25 * frac)[..., None]
            canvas.over_srgb(np.clip(srgb, 0.0, 1.0), cov, x0, y0)

    def draw_top(r, c):
        own = z[r, c]
        poly = corners(r, c, own)
        x0, y0, x1, y1 = _window(poly, canvas.w, canvas.h)
        if x1 <= x0 or y1 <= y0:
            return
        _x, _y, px, py, dinside, cov = _cover(poly, x0, y0, x1, y1)
        # The inverse of the projection at this hex's height: the world point under
        # every pixel of this top.  This is what makes the material world-sampled
        # instead of per-tile.
        wx = px - ox
        wy = -(py - oy + own * rise) / squash
        if f["water"][r, c]:
            depth = float(np.clip((f["sea"] - f["elev"][r, c]) / 0.30, 0.0, 1.0))
            base = palette.ramp_map(paint.SEA_RAMP, depth)
            srgb = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float64)
            srgb[:] = base
            ink = 0.10
        else:
            base = colours[f["biome"][r, c]]
            s = shade(wx, wy, f["biome"][r, c])
            srgb = base[None, None, :] * s[..., None]
            ink = 0.20
        # One line at the rim, drawn by darkening the top rather than as a second
        # pass: the distance is already here, and a stroke would have to be clipped
        # to the same polygon twice.
        edge = np.clip(1.0 - np.maximum(dinside, 0.0) / max(width * 1.4, 1e-9), 0.0, 1.0)
        srgb = srgb * (1.0 - ink * edge)[..., None]
        canvas.over_srgb(np.clip(srgb, 0.0, 1.0), cov, x0, y0)

    def draw_foam(r, c):
        if not f["water"][r, c]:
            return
        own = z[r, c]
        for d in range(6):
            nr, nc = int(grid.nbr[r, c, d, 0]), int(grid.nbr[r, c, d, 1])
            if nr < 0 or nc < 0 or f["water"][nr, nc]:
                continue
            cs = grid.corners_of(r, c)
            a, b = np.array(cs[d]), np.array(cs[(d + 1) % 6])
            centre = np.array(grid.centre_of((r, c)))
            mid = (a + b) * 0.5
            inward = centre - mid
            n = float(np.hypot(*inward))
            if n < 1e-9:
                continue
            inward = inward / n * foam_w
            q = [a, b, b + inward, a + inward]
            poly = [_project(x, y, own, ox, oy, squash, rise) for x, y in q]
            x0, y0, x1, y1 = _window(poly, canvas.w, canvas.h)
            if x1 <= x0 or y1 <= y0:
                continue
            _x, _y, px, py, _d, cov = _cover(poly, x0, y0, x1, y1)
            canvas.over_srgb(np.array([0.72, 0.84, 0.87]), cov * 0.55, x0, y0)

    def draw_ring(r, c):
        own = z[r, c]
        cx = grid.centers[r, c, 0]
        cy = grid.centers[r, c, 1]
        sx, sy = _project(cx, cy, own, ox, oy, squash, rise)
        rad = 0.56 * size
        a, b = rad, rad * squash
        x0 = max(int(np.floor(sx - a - 3)), 0)
        y0 = max(int(np.floor(sy - b - 3)), 0)
        x1 = min(int(np.ceil(sx + a + 3)), canvas.w)
        y1 = min(int(np.ceil(sy + b + 3)), canvas.h)
        if x1 <= x0 or y1 <= y0:
            return
        px = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
        py = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
        rho = np.hypot((px - sx) / a, (py - sy) / b)
        dist = np.abs(rho - 1.0) * min(a, b)
        w = max(1.4, 0.05 * size)
        dark = np.clip(w + 1.2 - dist, 0.0, 1.0)
        gold = np.clip(0.5 * w + 0.5 - dist, 0.0, 1.0)
        canvas.over_srgb(np.array([0.12, 0.10, 0.08]), dark * 0.55, x0, y0)
        canvas.over_srgb(np.array([0.93, 0.78, 0.42]), gold * 0.95, x0, y0)

    hero_row = None if hero is None else hero[0]
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            draw_sides(r, c)
            draw_top(r, c)
            draw_foam(r, c)
        if hero_row is not None and r == hero_row:
            draw_ring(*hero)
            if sprite is not None and sprite_box is not None:
                r, c = hero
                sx, sy = _project(grid.centers[r, c, 0], grid.centers[r, c, 1],
                                 z[r, c], ox, oy, squash, rise)
                x0, y0, x1, y1 = sprite_box
                canvas.blit(sprite.rgb, sprite.a,
                            int(round(sx - (x0 + x1) * 0.5)), int(round(sy - y1)))
    return canvas
