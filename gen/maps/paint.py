"""Painting the map: one per-pixel pass, then strokes on top.

The whole terrain is drawn in a single raster pass rather than hex by hex, and
that is the decision this module is built on.  A hexagon is the Voronoi cell of
its centre, so "which hex is this pixel in" is arithmetic, and everything that
depends on the cell -- its biome colour, its altitude, its distance to its own
border, its position within its six triangular sectors -- becomes an array
indexed by the cell the pixel landed in.  No seams, no anti-aliased cracks between
tiles, no per-hex canvas, and a map costs one pass over the pixels whatever the
hex size.

The relief is the same trick twice.  The height of the ground is not stored per
pixel; it is interpolated *within* the pixel from its hex's centre height out to
the midpoint of the sector's edge, i.e. towards the neighbour across it.  That
makes the surface piecewise-linear with creases on every hex edge, which is
exactly what a hex strategy map looks like: hills catch the light on one flank and
shadow the other, and the joins line up with the grid instead of fighting it.

Strokes -- the coastline, the hex lattice, the border of the map -- are the only
things drawn the traditional way, as distance fields over a window.  They are one
dimensional objects and there are far fewer of them than there are pixels.
"""
from __future__ import annotations

import numpy as np

from core import noise, palette, render, sdf
from core.canvas import Canvas
from core.render import LIGHT

from . import autotile, coast, politics, region, rivers, roads, terrain
from . import labels as _labels
from .grid import HexGrid

__all__ = [
    "INK",
    "LIGHT",
    "RIVER",
    "ROAD",
    "ROAD_EDGE",
    "SEA_BORDER",
    "SEA_RAMP",
    "map_image",
    "province_layer",
    "province_tints",
    "river_layer",
    "road_layer",
    "stroke_paths",
    "stroke_polyline",
    "terrain_layer",
]

INK = palette.hex_to_srgb("#16222e")

SEA_RAMP = tuple(palette.hex_to_srgb(h) for h in ("#59a3bd", "#2a6a8c", "#122c44"))

RIVER = palette.hex_to_srgb("#5f9cc0")

ROAD = palette.hex_to_srgb("#ecdcb0")
ROAD_EDGE = palette.hex_to_srgb("#4a3826")

SEA_BORDER = palette.hex_to_srgb("#86b4cd")

_AMBIENT = 0.66
_DIFFUSE = 0.52


def _window(canvas: Canvas, lo, hi) -> tuple:
    """Pixel bounds of a world box on a canvas, clipped to it."""
    sx = float(canvas.scale)
    cx, cy = canvas.center
    x0 = int(np.floor((lo[0] - cx) * sx - 0.5 + canvas.w / 2.0)) - 1
    x1 = int(np.ceil((hi[0] - cx) * sx + 0.5 + canvas.w / 2.0)) + 1
    if canvas.y_up:
        y0 = int(np.floor((cy - hi[1]) * sx - 0.5 + canvas.h / 2.0)) - 1
        y1 = int(np.ceil((cy - lo[1]) * sx + 0.5 + canvas.h / 2.0)) + 1
    else:
        y0 = int(np.floor((lo[1] - cy) * sx - 0.5 + canvas.h / 2.0)) - 1
        y1 = int(np.ceil((hi[1] - cy) * sx + 0.5 + canvas.h / 2.0)) + 1
    return (max(x0, 0), max(y0, 0), min(x1, canvas.w), min(y1, canvas.h))


def stroke_paths(canvas: Canvas, paths, srgb, alpha: float = 1.0, chunk: int = 12) -> None:
    """Stroke several polylines into one mask, and composite once.

    ``paths`` is a sequence of ``(points, width)`` pairs.  Merging them into a
    single mask before compositing is what makes a river *system* work: sixty
    short strokes at 80% alpha would double-darken at every junction, and a river
    that gets darker wherever it splits reads as a mistake rather than as water.

    The field is the minimum distance to any segment of a polyline, which is exact
    rather than an approximation -- but only if it is not evaluated where the line
    is not.  A coastline runs the length of the map, so one window round the whole
    of it would measure every segment against every pixel; instead the segments
    are taken in runs, each run is measured over its own small window, and the
    runs are merged with ``max``.
    """
    X, Y = canvas.grid()
    acc = np.zeros((canvas.h, canvas.w), dtype=np.float64)
    for points, width in paths:
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            continue
        hw = 0.5 * float(width)
        segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        for i in range(0, len(segs), chunk):
            part = segs[i:i + chunk]
            ends = np.array([p for seg in part for p in seg])
            x0, y0, x1, y1 = _window(canvas, ends.min(axis=0) - hw, ends.max(axis=0) + hw)
            if x1 <= x0 or y1 <= y0:
                continue
            Xs = X[y0:y1, x0:x1]
            Ys = Y[y0:y1, x0:x1]
            d = np.full(Xs.shape, np.inf)
            for a, b in part:
                d = np.minimum(d, sdf.segment(Xs, Ys, a, b))
            cov = render.coverage(np.abs(d) - hw, canvas.dx)
            acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1], cov)
    if acc.any():
        canvas.over(palette.srgb_to_linear(srgb), acc * float(alpha))


def stroke_polyline(canvas: Canvas, points, width: float, srgb, alpha: float = 1.0,
                    closed: bool = False, chunk: int = 12) -> None:
    """One polyline, as a union of capsules (see :func:`stroke_paths`)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return
    if closed:
        pts = np.vstack([pts, pts[:1]])
    stroke_paths(canvas, [(pts, width)], srgb, alpha=alpha, chunk=chunk)


def relief_height(grid: HexGrid, f: dict, cell, relief: float = 1.4):
    """The height field the light is cast on, per pixel.

    One hex is six triangles: the centre, and the midpoint of each edge.  Inside a
    triangle the height runs linearly from the cell's own to its neighbour's
    across that edge, so two hexes agree exactly at the edge they share and the
    ground creases there and only there.  Scaled by the hex size, because a 1.0
    of elevation across a 14 px hex has to be a slope in *pixels* to be a slope at
    all.
    """
    r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
    c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
    centre_h = f["height"][r, c]
    d = cell.sector
    nr = grid.nbr[r, c, d, 0]
    nc = grid.nbr[r, c, d, 1]
    ok = nr >= 0
    across = f["height"][np.clip(nr, 0, grid.rows - 1), np.clip(nc, 0, grid.cols - 1)]
    edge_h = np.where(ok, across, centre_h)
    apothem = 0.5 * np.sqrt(3.0) * grid.size
    t = np.clip(cell.rho / apothem, 0.0, 1.0)
    return (centre_h + (edge_h - centre_h) * t) * relief * grid.size


def terrain_layer(canvas: Canvas, grid: HexGrid, f: dict, seed: int = 0,
                  relief: float = 4.0, texture: float = 0.07, ground=None) -> None:
    """The single raster pass: biome colour, depth ramp, relief and grain.

    ``ground`` is an optional hook, called as ``ground(canvas, grid, f)`` and returning
    ``(shade, height)``: a per-pixel multiplier for the biome colour and a per-pixel
    height in pixels added to the hex relief *before* the one Lambert pass.  It is a
    hook rather than an import so this module stays about maps; :mod:`gen.maps.site` is
    the caller that knows about materials.  Water is not painted by it: the sea has no
    ground, so the hook's answer is applied to land only and the water is the ramp it
    always was, to the pixel.
    """
    shade = height_extra = None
    if ground is not None:
        shade, height_extra = ground(canvas, grid, f)
    X, Y = canvas.grid()
    cell = grid.cell(X, Y)
    r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
    c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
    water = f["water"][r, c]

    srgb = terrain.colour_table()[f["biome"][r, c]]

    depth = np.clip((f["sea"] - f["elev"][r, c]) / 0.30, 0.0, 1.0)
    sea = palette.ramp_map(SEA_RAMP, depth)
    srgb = np.where(water[..., None], sea, srgb)

    h = relief_height(grid, f, cell, relief)
    base_n = render.height_normals(h, canvas.dx, y_up=canvas.y_up)
    n = base_n
    if height_extra is not None:

        raised = h + np.where(water, 0.0, height_extra)
        n = render.height_normals(raised, canvas.dx, y_up=canvas.y_up)

        n = np.where(water[..., None], base_n, n)
    L = np.array([LIGHT[0], LIGHT[1], 0.65])
    L = L / np.linalg.norm(L)
    lam = np.clip(n @ L, 0.0, 1.0)
    srgb = srgb * (_AMBIENT + _DIFFUSE * lam)[..., None]

    grain = noise.fbm(X / 5.0, Y / 5.0, seed=seed + 404, freq=1.0, octaves=3)
    srgb = srgb * (1.0 - texture + 2.0 * texture * grain)[..., None]

    if shade is not None:
        srgb = srgb * np.where(water[..., None], 1.0, shade[..., None])

    canvas.over(palette.srgb_to_linear(np.clip(srgb, 0.0, 1.0)), cell.ok.astype(np.float64))


def cliff_alpha(canvas: Canvas, grid: HexGrid, f: dict, cell, strength: float = 0.55,
                width: float = 0.09, lo_q: float = 0.80, hi_q: float = 0.97):
    """A dark band along every edge the ground drops steeply across.

    This is the one place the map stops being a height field and becomes a map.
    A continuous surface shaded from its own gradient is honest and nearly
    invisible, because the flatlands really are flat; what makes hills read as
    hills at a glance is the crease, so the crease is drawn as a line.  The
    threshold is taken from the map's own distribution of drops rather than from
    a constant, so it works for a smooth continent and a shattered one, and each
    hex draws its own half of the band -- the two halves meet on the edge, which
    is where the crease actually is.
    """
    r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
    c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
    own = f["height"][r, c]
    d = cell.sector
    nr = grid.nbr[r, c, d, 0]
    nc = grid.nbr[r, c, d, 1]
    across = np.where(nr >= 0, f["height"][np.clip(nr, 0, grid.rows - 1),
                                            np.clip(nc, 0, grid.cols - 1)], own)
    drop = np.abs(own - across)

    inside = drop[cell.ok]
    lo = float(np.quantile(inside, lo_q)) if inside.size else 0.0
    hi = float(np.quantile(inside, hi_q)) if inside.size else 0.0
    weight = np.clip((drop - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    band = render.coverage(cell.edge_distance() - 0.5 * width * grid.size, canvas.dx)
    return band * weight * float(strength) * cell.ok.astype(np.float64)


def river_courses(grid: HexGrid, f: dict, fraction: float | None = None):
    """The rivers of a map: the courses, and the drainage a cell needed to be one.

    Keeping the threshold with the courses is what lets the painter scale a river's
    width by how much it drains without knowing anything about the hydrology.
    """
    frac = rivers.RIVER_FRACTION if fraction is None else float(fraction)
    filled = rivers.fill_depressions(grid, f["elev"], f["water"])
    down = rivers.downstream(grid, filled, f["water"])
    flux = rivers.drainage(grid, filled, down)
    thr = rivers.river_threshold(flux, f["water"], frac)
    return rivers.courses(grid, f, down, flux, frac), thr


def river_layer(canvas: Canvas, grid: HexGrid, courses, threshold: float,
                width: float = 0.17, alpha: float = 0.92) -> None:
    """The rivers, drawn widest where they have drained the most.

    A river's width goes with the square root of its drainage -- the standard
    hydraulic relation -- and that is why a river visibly fattens towards its
    mouth instead of being a constant line.  A course at the threshold is one
    ``width``; a course draining a hundred times as much is ten times wider, so
    the parameter is a scale rather than a thickness.
    """
    paths = []
    for course in courses:
        grew = max(np.sqrt(float(course["flux"]) / max(float(threshold), 1e-9)), 0.55)
        paths.append((course["points"], float(width) * grid.size * grew))
    stroke_paths(canvas, paths, RIVER, alpha=alpha)


def road_layer(canvas: Canvas, grid: HexGrid, roads, width: float = 0.15,
               edge: float = 0.11, alpha: float = 0.85) -> None:
    """The roads, drawn as a pale line with a darker one under it.

    Two strokes over the same path, merged the same way the rivers are: everything
    at one width goes down in one pass, so a road junction is a junction and not a
    darker spot where two roads happen to overlap.
    """
    if not roads:
        return
    paths = [(road["points"], float(width + edge) * grid.size) for road in roads]
    stroke_paths(canvas, paths, ROAD_EDGE, alpha=0.55 * float(alpha))
    paths = [(road["points"], float(width) * grid.size) for road in roads]
    stroke_paths(canvas, paths, ROAD, alpha=alpha)


def province_tints(n: int, L: float = 0.73, C: float = 0.13, hue0: float = 0.03) -> np.ndarray:
    """``n`` pale washes, spaced round the hue circle by the golden angle.

    Evenly spaced hues would do if the count were always the same, but a map asks
    for three provinces as often as for eleven, and ``k/n`` spacing only looks even
    when every hue is used.  The golden angle spreads *any prefix* of its sequence
    evenly, which is the property that is actually needed, and it is why two
    provinces next to each other are never two shades of the same colour.

    High lightness and low chroma on purpose: this is a wash over terrain that
    already carries the map's meaning, so its only job is to say where one
    province ends.
    """
    n = max(int(n), 1)
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        out[i] = palette.color(L, C, (hue0 + 0.6180339887498949 * i) % 1.0)
    return out


def province_layer(canvas: Canvas, grid: HexGrid, res: dict, cell,
                   fill: float = 0.22, frontier: float = 1.0, tints=None,
                   owners=None, internal: float = 0.35, line: float = 0.26,
                   ink=INK, ink_tints=None) -> None:
    """The political map: a wash per province, and a line where two meet.

    Both are drawn per pixel, and the frontier is drawn from the labels on either
    side of the edge a pixel is nearest to -- so the boundary of a province is not
    a polygon that was fitted to the cells, it is the cells.

    ``owners`` (one realm index per province) makes the frontier *political*: a
    border between two provinces of the same realm is damped to ``internal``,
    because a realm that is drawn as one block is a realm a player can see.  The
    provinces are still there -- faintly, which is what a province is on a map of
    realms.

    ``line`` and ``ink`` are how the same function draws a sea province: no wash,
    because the water is a depth ramp and a wash over it would be mud, a thinner
    line than a land frontier, because the sea is not divided by walls, and a pale
    ink, because the sea is dark.

    ``ink_tints`` is one colour per province for that line when a caller wants the
    line *worn* rather than drawn: territorial waters are the realm's, so the
    boundary of a sea province is painted in the colour of whoever holds it, and a
    zone nobody holds keeps the caller's ``ink``.
    """
    labels = np.asarray(res["labels"])
    if not res["count"]:
        return
    lab = None
    if fill > 0.0 or ink_tints is not None:
        r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
        c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
        lab = labels[r, c]
    if fill > 0.0:
        table = province_tints(res["count"]) if tints is None else np.asarray(tints)
        colour = table[np.clip(lab, 0, len(table) - 1)]
        canvas.over_srgb(colour, float(fill) * (lab >= 0) * cell.ok)
    if frontier > 0.0:
        a = float(frontier) * region.frontier_alpha(grid, labels, cell, width=float(line),
                                                    dx=canvas.dx)
        if owners is not None and internal < 1.0:
            a = a * _border_weight(grid, labels, cell, owners, internal)
        if ink_tints is None:
            canvas.over_srgb(ink, a)
        else:
            table = np.asarray(ink_tints)
            canvas.over_srgb(table[np.clip(lab, 0, len(table) - 1)], a * (lab >= 0))


def _border_weight(grid: HexGrid, labels, cell, owners, internal: float):
    """Per pixel: ``internal`` where both sides of the border are one realm.

    A cell is *internal* when every province bordering it is its own realm.  That
    is a property of the cell rather than of the edge a pixel is nearest to, and
    the difference is invisible: the extra half-hex of a full frontier at the last
    internal cell is a pixel wide at the width a frontier is drawn.
    """
    labels = np.asarray(labels)
    n = len(owners)
    weight = np.ones(labels.shape, dtype=np.float64)
    for r in range(labels.shape[0]):
        for c in range(labels.shape[1]):
            lab = int(labels[r, c])
            if lab < 0 or lab >= n:
                continue
            mine = int(owners[lab])
            same = True
            for rr, cc in grid.nbr[r, c]:
                rr, cc = int(rr), int(cc)
                if rr < 0 or cc < 0:
                    continue
                other = int(labels[rr, cc])
                if other >= 0 and other < n and int(owners[other]) != mine:
                    same = False
                    break
            if same:
                weight[r, c] = internal
    r = cell.rc[..., 0].astype(np.int64)
    c = cell.rc[..., 1].astype(np.int64)
    return weight[np.clip(r, 0, labels.shape[0] - 1),
                  np.clip(c, 0, labels.shape[1] - 1)] * cell.ok


def capital_marks(canvas: Canvas, grid: HexGrid, res: dict, size: float | None = None,
                  alpha: float = 0.95, colours=None) -> None:
    """A small dark boss with a pale centre, at every province capital.

    A filled dot rather than a star: at a 13 px hex a star is a smudge, and the
    point that has to survive is only "here is where the province is run from".

    ``colours`` gives a province's centre its own colour -- which a political map
    spends: the boss is the ink, the centre is whose province it is.  The boss
    keeps its own colour whatever it holds, because the mark has to be findable on
    the pale washes as well as on the dark ones.
    """
    if not res["count"]:
        return
    if colours is not None:
        colours = np.asarray(colours, dtype=np.float64)
    rad = float(0.30 * grid.size if size is None else size)
    X, Y = canvas.grid()
    for i, p in enumerate(res["provinces"]):
        cx, cy = grid.centre_of(p["capital"])
        x0, y0, x1, y1 = _window(canvas, (cx - rad, cy - rad), (cx + rad, cy + rad))
        if x1 <= x0 or y1 <= y0:
            continue
        d = np.hypot(X[y0:y1, x0:x1] - cx, Y[y0:y1, x0:x1] - cy)
        canvas.over_srgb(INK, render.coverage(d - rad, canvas.dx) * float(alpha), x0, y0)
        cap = (0.94, 0.91, 0.85)
        if colours is not None and i < len(colours):
            cap = colours[i]
        canvas.over_srgb(cap, render.coverage(d - rad * 0.42, canvas.dx), x0, y0)


def map_image(
    seed: int = 0,
    cols: int = 26,
    rows: int = 17,
    size: float = 14.0,
    margin: float = 7.0,
    relief: float = 4.0,
    texture: float = 0.07,
    ground=None,
    lattice: float = 0.085,
    cliff: float = 0.55,
    inland: float = 0.0,
    river: float = 0.92,
    river_fraction: float | None = None,
    road: float = 0.0,
    provinces: int = 0,
    province_fill: float = 0.22,
    province_tints=None,
    province_owners=None,
    province_internal: float = 0.35,
    frontier: float = 1.0,
    seas: int = 0,
    capitals: bool = True,
    names: bool = False,
    shadow: float = 0.34,
    **kw,
) -> Canvas:
    """A finished map at one seed: terrain, rivers, roads, provinces, coast and frame.

    ``seed`` and the grid shape are the whole specification of the *geography*;
    ``size`` and ``margin`` only decide how many pixels it is drawn in, so a map
    can be re-rendered at any scale and is still the same map.  Everything derived
    from the geography obeys the same rule: turning rivers off or asking for eight
    provinces changes what is drawn on the map and not one pixel of where the land
    is.

    The roads need capitals to run between, so they exist only with provinces --
    and they are routed by the biome table's own movement costs, which is why they
    belong to the geography rather than to the picture of it.

    ``seas`` divides the *water* into sea provinces, which are drawn as a maritime
    boundary and nothing else: the sea's colour is the depth of the sea and a
    political wash over it would be mud.  A map with sea provinces and a map without
    are the same map on land, down to the pixel.

    ``inland`` draws the *biome* border -- where two land hexes are of different
    ground -- with the same edge tiles the coast is drawn with, thinner and at
    whatever opacity is asked for.  It is the tileset's other use and not a second
    system: a border is where two hexes disagree, the disagreement is between two
    biomes here and between land and sea there, and everything downstream of that
    is the caller's ink.  A tile on water is off the land by construction, so the
    two borders never overlap and neither can be mistaken for the other.

    ``names`` sets the provinces' names into the map, in the project's own face and
    at the size each province can hold; :mod:`gen.maps.labels` argues the placement
    and is where a caller who wants the layout rather than the picture should look.
    The label belongs to the *geography*, like the rivers: it is placed by the same
    seed and at the same cell whatever the map is drawn at, and it is not a caption.
    """
    grid = HexGrid(cols, rows, size)
    f = terrain.fields(grid, seed=seed, **kw)
    w, h = grid.extent
    canvas = Canvas(int(np.ceil(w + 2.0 * margin)), int(np.ceil(h + 2.0 * margin)), scale=1.0)

    block = np.ones((rows, cols), dtype=bool)
    cell = grid.cell(*canvas.grid())

    if shadow > 0.0:
        canvas.over_srgb((0.04, 0.04, 0.06),
                         render.drop_shadow(cell.ok.astype(np.float64), canvas.scale,
                                            offset=(0.0, -3.0), sigma=3.2, opacity=shadow))

    terrain_layer(canvas, grid, f, seed=seed, relief=relief, texture=texture,
                  ground=ground)

    if cliff > 0.0:
        canvas.over_srgb(INK, cliff_alpha(canvas, grid, f, cell, strength=cliff))

    if inland > 0.0:
        edge = autotile.borders(f["biome"], grid, within=~f["water"])
        autotile.border_layer(canvas, grid, edge, cell, INK, width=0.11, alpha=inland)

    res = None
    if provinces > 0 and (~f["water"]).any():
        res = region.provinces(grid, f, count=int(provinces), seed=seed)
        province_layer(canvas, grid, res, cell, fill=province_fill,
                       tints=province_tints, owners=province_owners,
                       internal=province_internal, frontier=frontier)

    if seas > 0 and f["water"].any():
        charted = region.seas(grid, f, count=int(seas), seed=seed)
        ink_tints = None
        if province_owners is not None and province_tints is not None:
            held = politics.waters(grid, res, charted, province_owners)

            realm = {}
            for i, o in enumerate(province_owners):
                if int(o) >= 0 and int(o) not in realm:
                    realm[int(o)] = province_tints[i]
            ink_tints = np.array([realm.get(int(o), SEA_BORDER) for o in held])
        province_layer(canvas, grid, charted, cell, fill=0.0, frontier=0.58,
                       line=0.19, ink=SEA_BORDER, ink_tints=ink_tints)

    if river > 0.0 and (~f["water"]).any():
        courses, threshold = river_courses(grid, f, river_fraction)
        river_layer(canvas, grid, courses, threshold, alpha=river)

    if road > 0.0 and res is not None:
        road_layer(canvas, grid, roads.network(grid, f, res), alpha=road)

    if res is not None and names:
        _labels.draw(canvas, grid, _labels.layout(grid, res))

    if res is not None and capitals:
        marks = province_tints if province_owners is not None else None
        capital_marks(canvas, grid, res, size=0.30 * size, colours=marks)

    if lattice > 0.0:
        d_edge = cell.edge_distance()
        a = render.coverage(d_edge - 0.5 * lattice * size, canvas.dx)

        weight = np.where(f["water"][np.clip(cell.rc[..., 0], 0, rows - 1),
                                     np.clip(cell.rc[..., 1], 0, cols - 1)], 0.08, 0.20)
        canvas.over_srgb(INK, a * weight * cell.ok)

    edge = autotile.borders((~f["water"]).astype(np.int64), grid, outside=0)
    autotile.border_layer(canvas, grid, edge, cell, INK, width=0.24, alpha=0.80)
    for loop in coast.boundary_loops(grid, block):
        stroke_polyline(canvas, loop, 0.14 * size, INK, alpha=0.95, closed=True)
    return canvas
