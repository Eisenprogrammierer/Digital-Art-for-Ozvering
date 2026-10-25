"""A continuous isometric voxel world rather than a board of hexes.

The strategic map used to be a field of hexagons: every cell was a unit of
movement, every boundary was a line, and an icon stood *in* a cell.  This module
replaces that board with a quantised height field drawn as square voxel columns.
There is still a grid underneath -- a game needs one for terrain lookup and path
cost -- but no cell boundary is drawn, so the eye reads country rather than a
lattice.  Movement is continuous: positions are floats in world cells, and the
sidecar exposes the height and biome tables together with the projection needed to
place a sprite at any point.

The projection is the same one used by :mod:`gen.voxel`::

    screen_x = (x - y) * tile / 2
    screen_y = (x + y) * tile / 4 - z * tile * rise

with ``x`` east, ``y`` south and ``z`` up.  The top of a column is a diamond, and
only the two faces turned towards the camera are painted, so a hill is a stack of
voxels and a cliff is the side of one.  Water is flat; land is quantised into
``levels`` steps.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from core import noise, palette, render as renderlib, sdf
from core.canvas import Canvas
from core.render import LIGHT

from . import terrain

__all__ = [
    "SEA_RAMP",
    "fields",
    "heightmap",
    "world_to_screen",
    "screen_to_world",
    "height_at",
    "walkable_at",
    "biome_at",
    "render",
]

SEA_RAMP = tuple(palette.hex_to_srgb(h) for h in ("#5d9fb8", "#2d6c8c", "#123149"))

_EARTH = palette.hex_to_srgb("#514334")
_EARTH_DARK = palette.hex_to_srgb("#2b2822")
_ROAD = palette.hex_to_srgb("#d8c08a")
_ROAD_EDGE = palette.hex_to_srgb("#4a3926")
_RIVER = palette.hex_to_srgb("#5797ba")


def _shift(a, dx: int, dy: int, fill=0.0):
    out = np.full_like(a, fill)
    h, w = a.shape
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs, ys = max(0, -dx), max(0, -dy)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0:ys1, xs0:xs1] = a[ys:ys + (ys1 - ys0), xs:xs + (xs1 - xs0)]
    return out


def _smooth(a, passes: int = 1, weight: float = 0.45):
    out = np.asarray(a, dtype=np.float64)
    for _ in range(int(passes)):
        acc = out.copy()
        n = np.ones_like(out)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            acc += _shift(out, dx, dy)
            n += _shift(np.ones_like(out), dx, dy)
        out = (1.0 - weight) * out + weight * acc / n
    return out


def fields(seed: int = 7, cols: int = 42, rows: int = 30,
           sea_fraction: float = 0.34, features: float = 2.3,
           ridges: float = 0.26, edge: float = 0.92,
           smooth_passes: int = 1) -> dict:
    """Generate elevation, moisture, biome and a quantised voxel height table."""
    cols, rows = int(cols), int(rows)
    if cols < 3 or rows < 3:
        raise ValueError("a voxel world needs at least a 3 by 3 field")
    if not 0.0 <= float(sea_fraction) <= 0.95:
        raise ValueError("sea fraction is between 0 and 0.95")
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    cx = (x + 0.5) / max(cols, 1) * 2.0 - 1.0
    cy = (y + 0.5) / max(rows, 1) * 2.0 - 1.0
    ux, uy = noise.warp(cx * features, cy * features, seed=seed + 13,
                        amp=0.48, freq=1.0, octaves=3)
    elev = noise.fbm(ux, uy, seed=seed + 29, freq=1.0, octaves=6, gain=0.55)
    if ridges > 0.0:
        elev += float(ridges) * noise.ridged(ux, uy, seed=seed + 41,
                                             freq=1.7, octaves=5, gain=0.5)
    span = math.hypot(1.0, rows / max(cols, 1))
    radius = np.hypot(ux / features, uy / features) / span
    falloff = np.clip((radius - 0.46) / 0.44, 0.0, 1.0)
    elev -= float(edge) * falloff * falloff
    lo, hi = float(elev.min()), float(elev.max())
    elev = (elev - lo) / max(hi - lo, 1e-9)
    elev = _smooth(elev, smooth_passes)

    sea = float(np.quantile(elev, float(sea_fraction)))
    water = elev < sea
    alt = (elev - sea) / max(1.0 - sea, 1e-6)

    vx, vy = noise.warp(cx * 2.7, cy * 2.7, seed=seed + 101,
                        amp=0.44, freq=1.0, octaves=3)
    moist = noise.fbm(vx, vy, seed=seed + 67, freq=1.0, octaves=4, gain=0.55)
    moist = (moist - float(moist.min())) / max(float(moist.max() - moist.min()), 1e-9)

    nbr_water = np.zeros(water.shape, dtype=bool)
    nbr_land = np.zeros(water.shape, dtype=bool)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        shifted_water = _shift(water, dx, dy, 1.0).astype(bool)
        shifted_land = _shift(~water, dx, dy, 1.0).astype(bool)
        nbr_water |= shifted_water
        nbr_land |= shifted_land

    f = {
        "elev": elev,
        "alt": alt,
        "moist": moist,
        "sea": sea,
        "water": water,
        "near_sea": (~water) & nbr_water,
        "near_land": water & nbr_land,
    }
    f["biome"] = terrain.classify(f)
    f["height"] = np.where(water, 0.0, elev)
    f["cost"] = np.array([b.cost for b in terrain.BIOMES], dtype=np.float64)[f["biome"]]
    f["walkable"] = ~water
    return f


def heightmap(f: dict, levels: int = 8) -> np.ndarray:
    """Quantise land into voxel columns; water is level zero."""
    levels = max(1, int(levels))
    land = f["walkable"]
    z = 1.0 + np.floor(np.clip(f["alt"], 0.0, 1.0) * (levels - 1e-6))
    z = np.clip(z, 1.0, float(levels))
    return np.where(land, z, 0.0).astype(np.int32)


def _tone(colour, factor):
    return np.clip(np.asarray(colour, dtype=np.float64) * float(factor), 0.0, 1.0)


def _window(canvas: Canvas, pts, pad: float = 2.0):
    arr = np.asarray(pts, dtype=np.float64)
    x0 = max(0, int(np.floor(arr[:, 0].min() - pad)) - 1)
    y0 = max(0, int(np.floor(arr[:, 1].min() - pad)) - 1)
    x1 = min(canvas.w, int(np.ceil(arr[:, 0].max() + pad)) + 2)
    y1 = min(canvas.h, int(np.ceil(arr[:, 1].max() + pad)) + 2)
    return x0, y0, x1, y1


def _poly(canvas: Canvas, pts, colour, alpha: float = 1.0):
    if alpha <= 0.0:
        return
    x0, y0, x1, y1 = _window(canvas, pts)
    if x1 <= x0 or y1 <= y0:
        return
    X, Y = canvas.grid()
    d = sdf.polygon(X[y0:y1, x0:x1], Y[y0:y1, x0:x1], np.asarray(pts))
    cov = renderlib.coverage(d, canvas.dx)
    if cov.any():
        canvas.over_srgb(colour, cov * float(alpha), x0, y0)


def _stroke(canvas: Canvas, paths, colour, width: float, alpha: float = 1.0):
    X, Y = canvas.grid()
    acc = np.zeros((canvas.h, canvas.w), dtype=np.float64)
    for points in paths:
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            pad = width + 2.0
            x0 = max(0, int(math.floor(min(a[0], b[0]) - pad)) - 1)
            x1 = min(canvas.w, int(math.ceil(max(a[0], b[0]) + pad)) + 2)
            y0 = max(0, int(math.floor(min(a[1], b[1]) - pad)) - 1)
            y1 = min(canvas.h, int(math.ceil(max(a[1], b[1]) + pad)) + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            d = sdf.segment(X[y0:y1, x0:x1], Y[y0:y1, x0:x1], a, b, width * 0.5)
            cov = renderlib.coverage(d, canvas.dx)
            acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1], cov)
    if acc.any():
        canvas.over_srgb(colour, acc * float(alpha))


def world_to_screen(scene: dict, x: float, y: float, z: float = 0.0):
    """Project a continuous world position to pixels for this rendered scene."""
    tile = float(scene["tile_px"])
    ox, oy = scene["screen"]["origin"]
    sx = (float(x) - float(y)) * tile * 0.5 + ox
    sy = (float(x) + float(y)) * tile * 0.25 - float(z) * tile * scene["screen"]["rise"] + oy
    return float(sx), float(sy)


def screen_to_world(scene: dict, sx: float, sy: float, z: float = 0.0):
    """Invert :func:`world_to_screen`; the result is a continuous world point."""
    tile = float(scene["tile_px"])
    ox, oy = scene["screen"]["origin"]
    u = (float(sx) - ox) / (tile * 0.5)
    v = ((float(sy) - oy) + float(z) * tile * scene["screen"]["rise"]) / (tile * 0.25)
    return (0.5 * (u + v), 0.5 * (v - u))


def height_at(scene: dict, x: float, y: float) -> float:
    c = int(math.floor(float(x)))
    r = int(math.floor(float(y)))
    if r < 0 or c < 0 or r >= int(scene["rows"]) or c >= int(scene["cols"]):
        return 0.0
    return float(scene["heightmap"][r][c])


def walkable_at(scene: dict, x: float, y: float) -> bool:
    c, r = int(math.floor(float(x))), int(math.floor(float(y)))
    if r < 0 or c < 0 or r >= int(scene["rows"]) or c >= int(scene["cols"]):
        return False
    return bool(scene["walkable"][r][c])


def biome_at(scene: dict, x: float, y: float) -> str:
    c, r = int(math.floor(float(x))), int(math.floor(float(y)))
    if r < 0 or c < 0 or r >= int(scene["rows"]) or c >= int(scene["cols"]):
        return "ocean"
    return str(scene["biome"][r][c])


def _spawn(f: dict):
    rows, cols = f["water"].shape
    land = np.argwhere(~f["water"])
    if not len(land):
        return (cols * 0.5, rows * 0.5)
    centre = np.array([rows * 0.5, cols * 0.5])
    score = np.hypot(land[:, 0] - centre[0], land[:, 1] - centre[1])
    low = f["alt"][land[:, 0], land[:, 1]]
    score = score + np.abs(low - 0.18) * 5.0
    r, c = land[int(np.argmin(score))]
    return (float(c) + 0.5, float(r) + 0.5)


def _rivers(f: dict, count: int, seed: int):
    rows, cols = f["water"].shape
    rng = np.random.default_rng(int(seed) + 0x51A7)
    candidates = np.argwhere((~f["water"]) & (f["alt"] > 0.28))
    if not len(candidates):
        return []
    order = np.argsort(f["elev"][candidates[:, 0], candidates[:, 1]])[::-1]
    chosen = candidates[order]
    paths = []
    used = set()
    for r, c in chosen:
        if len(paths) >= int(count):
            break
        if (int(r), int(c)) in used:
            continue
        path = []
        seen = set()
        rr, cc = int(r), int(c)
        for _ in range(rows * cols):
            if (rr, cc) in seen:
                break
            seen.add((rr, cc))
            path.append((cc + 0.5, rr + 0.5))
            used.add((rr, cc))
            if f["water"][rr, cc]:
                break
            best = None
            best_h = f["elev"][rr, cc]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nr, nc = rr + dy, cc + dx
                    if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                        continue
                    h = f["elev"][nr, nc]
                    step = 1.0 if dx == 0 or dy == 0 else 1.414
                    score = h + 0.012 * step + 0.002 * rng.random()
                    if score < best_h or (best is None and h < best_h + 0.02):
                        best = (nr, nc)
                        best_h = h
            if best is None or best_h >= f["elev"][rr, cc] - 1e-7:
                break
            rr, cc = best
        if len(path) >= 4:
            paths.append(path)
    return paths


def _road_path(f: dict, start, goal):
    rows, cols = f["water"].shape
    sr, sc = int(start[1]), int(start[0])
    gr, gc = int(goal[1]), int(goal[0])

    def cost(r, c):
        return 1.0 + 8.0 * float(f["alt"][r, c]) + (3.0 if f["biome"][r, c] in
                                                     (terrain.INDEX["marsh"], terrain.INDEX["mountain"]) else 0.0)

    start_node = (sr, sc)
    frontier = [(0.0, start_node)]
    came = {}
    dist = {start_node: 0.0}
    while frontier:
        d0, (r, c) = heapq.heappop(frontier)
        if (r, c) == (gr, gc):
            break
        if d0 > dist.get((r, c), 1e30):
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nr, nc = r + dy, c + dx
                if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                    continue
                if f["water"][nr, nc]:
                    continue
                step = 1.0 if dx == 0 or dy == 0 else 1.414
                nd = d0 + step * cost(nr, nc)
                if nd < dist.get((nr, nc), 1e30):
                    dist[(nr, nc)] = nd
                    came[(nr, nc)] = (r, c)
                    heapq.heappush(frontier, (nd, (nr, nc)))
    if (gr, gc) not in came and (gr, gc) != start_node:
        return None
    out = [(gr + 0.5, gc + 0.5)]
    cur = (gr, gc)
    while cur != start_node:
        cur = came[cur]
        out.append((cur[1] + 0.5, cur[0] + 0.5))
    return list(reversed(out))


def _roads(f: dict, count: int, seed: int):
    rows, cols = f["water"].shape
    rng = np.random.default_rng(int(seed) + 0x70AD)
    land = np.argwhere(~f["water"])
    if len(land) < 2:
        return []
    paths = []
    for _ in range(max(0, int(count))):
        for _attempt in range(18):
            i, j = rng.integers(0, len(land), size=2)
            a, b = land[i], land[j]
            if np.hypot(a[0] - b[0], a[1] - b[1]) < min(rows, cols) * 0.35:
                continue
            path = _road_path(f, (a[1], a[0]), (b[1], b[0]))
            if path and len(path) > 3:
                paths.append(path)
                break
    return paths


def _hero_canvas(seed: int, size: int):
    try:
        from gen.figures import figure as figurelib
        return figurelib.render(seed=seed, cycle="idle", phase=0.0, size=size,
                                facing="down")
    except Exception:
        return None


def render(seed: int = 7, cols: int = 42, rows: int = 30, tile: float = 14.0,
           levels: int = 8, sea_fraction: float = 0.34, rivers: int = 4,
           roads: int = 2, hero=None, hero_seed: int = 7,
           margin: int = 18, background="#0b1926") -> tuple:
    """Render a complete isometric voxel world and its free-movement sidecar."""
    tile = float(tile)
    if tile < 2.0:
        raise ValueError("a voxel tile is at least two pixels wide")
    f = fields(seed=seed, cols=cols, rows=rows, sea_fraction=sea_fraction)
    z = heightmap(f, levels)
    colours = terrain.colour_table()
    rows, cols = z.shape
    texture = noise.fbm(*np.meshgrid(np.arange(cols) * 0.23, np.arange(rows) * 0.23),
                        seed=seed + 404, freq=1.0, octaves=3)
    gx = np.gradient(z.astype(np.float64), axis=1)
    gy = np.gradient(z.astype(np.float64), axis=0)
    n = np.stack([-gx, -gy, np.full_like(z, 1.8)], axis=-1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    L = np.array([LIGHT[0], LIGHT[1], 0.65], dtype=np.float64)
    L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0.0, 1.0)

    depth = np.clip((f["sea"] - f["elev"]) / 0.30, 0.0, 1.0)
    sea_col = palette.ramp_map(SEA_RAMP, depth)
    top = colours[f["biome"]]
    top = np.where(f["water"][..., None], sea_col, top)
    top = top * (0.86 + 0.22 * lam[..., None])
    top = top * (0.94 + 0.12 * texture[..., None])
    top = np.clip(top, 0.0, 1.0)

    points = []
    for r in range(rows):
        for c in range(cols):
            h = float(z[r, c])
            for xx, yy in ((c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)):
                points.append(((xx - yy) * tile * 0.5,
                               (xx + yy) * tile * 0.25 - h * tile * 0.62))
    pts = np.asarray(points)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    ox = -x0 + margin
    oy = -y0 + margin
    W = int(math.ceil(x1 - x0 + 2 * margin))
    H = int(math.ceil(y1 - y0 + 2 * margin))
    canvas = Canvas(W, H, scale=1.0, center=(W * 0.5, H * 0.5), y_up=False)
    if background is not None:
        bg = palette.hex_to_srgb(background) if isinstance(background, str) else background
        canvas.fill(bg)

    def px(x, y, h=0.0):
        return ((x - y) * tile * 0.5 + ox,
                (x + y) * tile * 0.25 - h * tile * 0.62 + oy)

    def neighbour(r, c, dr, dc):
        rr, cc = r + dr, c + dc
        if rr < 0 or cc < 0 or rr >= rows or cc >= cols:
            return 0.0
        return float(z[rr, cc])

    order = sorted(((r, c) for r in range(rows) for c in range(cols)),
                   key=lambda rc: (rc[0] + rc[1], rc[0]))
    for r, c in order:
        h = float(z[r, c])
        if h <= 0.0:
            continue
        colour = top[r, c]
        east = neighbour(r, c, 0, 1)
        south = neighbour(r, c, 1, 0)
        if east < h:
            side = palette.mix(colour, _EARTH, 0.34)
            side = _tone(side, 0.72)
            _poly(canvas, [px(c + 1, r, h), px(c + 1, r + 1, h),
                           px(c + 1, r + 1, east), px(c + 1, r, east)], side)
        if south < h:
            side = palette.mix(colour, _EARTH_DARK, 0.30)
            side = _tone(side, 0.56)
            _poly(canvas, [px(c, r + 1, h), px(c + 1, r + 1, h),
                           px(c + 1, r + 1, south), px(c, r + 1, south)], side)
        _poly(canvas, [px(c, r, h), px(c + 1, r, h),
                       px(c + 1, r + 1, h), px(c, r + 1, h)], colour)

    for r, c in order:
        if z[r, c] > 0:
            continue
        colour = top[r, c]
        _poly(canvas, [px(c, r, 0.0), px(c + 1, r, 0.0),
                       px(c + 1, r + 1, 0.0), px(c, r + 1, 0.0)], colour)
        if f["near_land"][r, c]:
            shore = palette.mix(colour, (0.86, 0.87, 0.80), 0.25)
            _poly(canvas, [px(c, r, 0.02), px(c + 1, r, 0.02),
                           px(c + 1, r + 1, 0.02), px(c, r + 1, 0.02)], shore, 0.38)

    river_paths = _rivers(f, rivers, seed)
    road_paths = _roads(f, roads, seed)
    scene = {
        "seed": int(seed),
        "projection": "isometric-axonometric",
        "axes": {"x": "east", "y": "south", "z": "up"},
        "cols": cols,
        "rows": rows,
        "levels": int(levels),
        "tile_px": tile,
        "screen": {"origin": [float(ox), float(oy)], "rise": 0.62},
        "size": [int(W), int(H)],
        "movement": "continuous",
        "sea": float(f["sea"]),
        "heightmap": z.astype(int).tolist(),
        "walkable": f["walkable"].astype(bool).tolist(),
        "biome": [[terrain.BIOMES[int(code)].key for code in row] for row in f["biome"]],
        "spawn": list(_spawn(f)),
        "rivers": [[[float(x), float(y)] for x, y in path] for path in river_paths],
        "roads": [[[float(x), float(y)] for x, y in path] for path in road_paths],
        "stats": {},
    }
    if river_paths:
        for path in river_paths:
            _stroke(canvas, [[px(x, y, height_at(scene, x, y) + 0.03) for x, y in path]],
                    _RIVER, max(1.4, tile * 0.12), 0.88)
    if road_paths:
        for path in road_paths:
            line = [px(x, y, height_at(scene, x, y) + 0.04) for x, y in path]
            _stroke(canvas, [line], _ROAD_EDGE, max(1.8, tile * 0.17), 0.78)
            _stroke(canvas, [line], _ROAD, max(1.0, tile * 0.10), 0.90)

    chosen_hero = None
    hero_pos = _spawn(f) if hero == "spawn" else hero
    if hero_pos is not None:
        hx, hy = float(hero_pos[0]), float(hero_pos[1])
        chosen_hero = {"x": hx, "y": hy, "seed": int(hero_seed),
                       "height": height_at(scene, hx, hy),
                       "walkable": walkable_at(scene, hx, hy)}
        sprite = _hero_canvas(int(hero_seed), max(18, int(tile * 2.15)))
        if sprite is not None:
            sx, sy = px(hx, hy, chosen_hero["height"])
            canvas.blit(sprite.rgb, sprite.a,
                        int(round(sx - sprite.w * 0.5)),
                        int(round(sy - sprite.h * 0.86)))
    scene["hero"] = chosen_hero

    counts = {}
    for i, b in enumerate(terrain.BIOMES):
        n = int(np.count_nonzero(f["biome"] == i))
        if n:
            counts[b.name] = n
    total = int(cols * rows)
    counts["cells"] = total
    counts["water"] = float(np.count_nonzero(f["water"])) / max(total, 1)
    scene["stats"] = counts
    return canvas, scene
