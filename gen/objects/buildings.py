"""Deterministic isometric buildings, one five-view turnaround per seed.

A building is a stack of boxes, not a texture on a billboard.  The model is built
from a foundation, a wall mass, window bays, a roof and a short list of details
that belong to the building's role.  The same model is then projected from north,
west, south, east and above by :mod:`gen.voxel`, so the five pictures agree about
every corner and a game can choose the facing that matches its camera.

The vocabulary is deliberately modern and mixed: houses and schools, warehouses
and factories, clinics and stations, hangars and cooling towers.  Variation comes
from proportions, colour and a handful of seeded details rather than from a
different renderer per type.
"""
from __future__ import annotations

import numpy as np

from core import palette, render as renderlib
from gen import voxel

__all__ = ["KINDS", "plan", "render", "views", "sheet"]

KINDS = (
    "house", "apartment", "office", "shop", "warehouse", "factory",
    "clinic", "school", "station", "hangar", "barracks", "power_plant",
    "water_tower", "bunker", "tower", "market", "hotel", "fire_station",
)

_LABELS = {
    "house": "House",
    "apartment": "Apartment block",
    "office": "Office block",
    "shop": "Shop",
    "warehouse": "Warehouse",
    "factory": "Factory",
    "clinic": "Clinic",
    "school": "School",
    "station": "Station",
    "hangar": "Hangar",
    "barracks": "Barracks",
    "power_plant": "Power plant",
    "water_tower": "Water tower",
    "bunker": "Bunker",
    "tower": "Tower",
    "market": "Market hall",
    "hotel": "Hotel",
    "fire_station": "Fire station",
}


def _rgb(seed, rng, low=0.40, high=0.76, chroma=(0.02, 0.10)):
    hue = float(rng.random())
    light = float(rng.uniform(low, high))
    sat = float(rng.uniform(*chroma))
    return palette.oklch_to_srgb((light, sat, hue))


def _hex(rgb):
    return palette.srgb_to_hex(np.asarray(rgb, dtype=np.float64))


def _materials(rng, kind):
    wall = _rgb(0, rng, 0.54, 0.78, (0.015, 0.075))
    trim = palette.mix(wall, (0.10, 0.11, 0.13), 0.38)
    roof = palette.mix(wall, (0.13, 0.14, 0.16), 0.24)
    glass = palette.mix(palette.oklch_to_srgb((0.52, 0.08, 0.54)),
                        (0.04, 0.08, 0.11), 0.46)
    concrete = palette.oklch_to_srgb((0.66, 0.018, 0.10))
    dark = palette.oklch_to_srgb((0.20, 0.018, 0.12))
    accent = _rgb(0, rng, 0.48, 0.68, (0.10, 0.20))
    if kind in ("factory", "warehouse", "hangar", "power_plant", "bunker"):
        wall = palette.mix(wall, concrete, 0.34)
        roof = palette.mix(roof, concrete, 0.24)
    if kind == "clinic":
        wall = palette.mix(wall, (0.92, 0.93, 0.90), 0.55)
        accent = palette.hex_to_srgb("#c8403c")
    if kind == "fire_station":
        wall = palette.mix(wall, (0.88, 0.24, 0.18), 0.40)
        accent = palette.hex_to_srgb("#e8c84a")
    if kind == "market":
        accent = _rgb(0, rng, 0.58, 0.70, (0.13, 0.22))
    return {
        "wall": wall, "trim": trim, "roof": roof, "glass": glass,
        "concrete": concrete, "dark": dark, "accent": accent,
    }


def _windows(m, w, d, base, floor_h, floors, mats, lit=0.30, rng=None):
    rng = rng or np.random.default_rng(1)
    count_x = max(2, min(5, int(round(w / 1.25))))
    count_y = max(1, min(3, int(round(d / 1.7))))
    width = max(0.30, min(0.58, (w - 0.7) / max(count_x, 1) * 0.56))
    depth = max(0.30, min(0.58, (d - 0.7) / max(count_y, 1) * 0.56))
    for floor in range(floors):
        z0 = base + floor * floor_h + floor_h * 0.28
        z1 = z0 + floor_h * 0.40
        colour = mats["glass"] if rng.random() > lit else palette.mix(
            mats["glass"], mats["accent"], 0.35)
        for i in range(count_x):
            c = (i + 0.5) / count_x * w
            x0, x1 = c - width * 0.5, c + width * 0.5
            m.add(x0, -0.025, z0, x1, 0.01, z1, mats["trim"], outline=False)
            m.add(x0 + 0.035, -0.04, z0 + 0.035, x1 - 0.035, 0.015, z1 - 0.035,
                  colour, outline=False)
            m.add(x0, d - 0.01, z0, x1, d + 0.025, z1, mats["trim"], outline=False)
            m.add(x0 + 0.035, d - 0.015, z0 + 0.035, x1 - 0.035, d + 0.04, z1 - 0.035,
                  colour, outline=False)
        for j in range(count_y):
            c = (j + 0.5) / count_y * d
            y0, y1 = c - depth * 0.5, c + depth * 0.5
            m.add(-0.025, y0, z0, 0.01, y1, z1, mats["trim"], outline=False)
            m.add(-0.04, y0 + 0.035, z0 + 0.035, 0.015, y1 - 0.035, z1 - 0.035,
                  colour, outline=False)
            m.add(w - 0.01, y0, z0, w + 0.025, y1, z1, mats["trim"], outline=False)
            m.add(w - 0.015, y0 + 0.035, z0 + 0.035, w + 0.04, y1 - 0.035, z1 - 0.035,
                  colour, outline=False)


def _roof(m, w, d, z, mats, kind, rng):
    m.add(-0.16, -0.16, z, w + 0.16, d + 0.16, z + 0.16, mats["trim"])
    if kind in ("house", "barracks", "hotel"):
        steps = 4
        for i in range(steps):
            inset = 0.22 * i
            m.add(inset, -0.05 + 0.10 * i, z + 0.16 + i * 0.26,
                  w - inset, d + 0.05 - 0.10 * i, z + 0.42 + i * 0.26,
                  mats["roof"])
        if kind == "house" and rng.random() < 0.8:
            m.add(w * 0.68, d * 0.48, z + 1.1, w * 0.82, d * 0.64, z + 2.0,
                  mats["wall"])
    elif kind == "factory":
        for i in range(3):
            x0 = w * (0.12 + 0.28 * i)
            m.add(x0, d * 0.18, z + 0.16, x0 + w * 0.16, d * 0.82, z + 0.72,
                  mats["roof"])
    else:
        m.add(w * 0.14, d * 0.18, z + 0.16, w * 0.34, d * 0.40, z + 0.52,
              mats["concrete"])
        m.add(w * 0.58, d * 0.52, z + 0.16, w * 0.80, d * 0.74, z + 0.46,
              mats["concrete"])
        m.add(w * 0.40, d * 0.10, z + 0.16, w * 0.48, d * 0.22, z + 0.90,
              mats["dark"])


def _door(m, w, d, base, mats, side="south", width=1.0):
    if side == "south":
        x = (w - width) * 0.5
        m.add(x - 0.06, -0.05, base, x + width + 0.06, 0.04, base + 1.28, mats["trim"])
        m.add(x + 0.08, -0.08, base + 0.08, x + width - 0.08, 0.06, base + 1.16,
              mats["accent"])
    else:
        y = (d - width) * 0.5
        m.add(w - 0.04, y - 0.06, base, w + 0.05, y + width + 0.06, base + 1.28,
              mats["trim"])
        m.add(w + 0.05, y + 0.08, base + 0.08, w + 0.08, y + width - 0.08, base + 1.16,
              mats["accent"])


def _porch(m, w, d, base, mats):
    m.add(-0.32, -0.62, base + 1.02, w + 0.32, 0.02, base + 1.16, mats["roof"])
    for x in (0.18, w - 0.18):
        m.add(x - 0.06, -0.54, base, x + 0.06, -0.42, base + 1.02, mats["trim"])


def _balconies(m, w, d, base, floor_h, floors, mats):
    for floor in range(1, floors):
        z = base + floor * floor_h + floor_h * 0.10
        m.add(-0.30, d * 0.20, z, 0.02, d * 0.80, z + 0.07, mats["concrete"])
        m.add(-0.30, d * 0.20, z + 0.07, -0.22, d * 0.80, z + 0.48, mats["trim"])


def _roll_door(m, w, base, mats):
    m.add(w * 0.14, -0.05, base, w * 0.86, 0.03, base + 1.65, mats["trim"])
    for i in range(5):
        z = base + 0.16 + i * 0.29
        m.add(w * 0.17, -0.07, z, w * 0.83, 0.05, z + 0.19, mats["concrete"])


def _chimney(m, x, y, z, mats, height=2.2, radius=0.26):
    m.add(x - radius, y - radius, z, x + radius, y + radius, z + height, mats["concrete"])
    m.add(x - radius * 1.18, y - radius * 1.18, z + height - 0.18,
          x + radius * 1.18, y + radius * 1.18, z + height, mats["trim"])


def _cooling_tower(m, x, y, z, mats, scale=1.0):
    levels = 7
    for i in range(levels):
        t = i / (levels - 1)
        half = (0.80 - 0.34 * np.sin(np.pi * t)) * scale
        m.add(x - half, y - half, z + i * 0.55 * scale,
              x + half, y + half, z + (i + 1) * 0.55 * scale,
              mats["concrete"], outline=False)
        m.add(x - half * 0.72, y - half * 0.72, z + (i + 1) * 0.55 * scale,
              x + half * 0.72, y + half * 0.72, z + (i + 1) * 0.55 * scale + 0.04,
              mats["trim"], outline=False)


def _antenna(m, x, y, z, mats):
    m.add(x - 0.045, y - 0.045, z, x + 0.045, y + 0.045, z + 2.2, mats["trim"])
    for k in range(3):
        zz = z + 1.2 + 0.30 * k
        m.add(x - 0.35, y - 0.035, zz, x + 0.35, y + 0.035, zz + 0.06, mats["accent"])


def _entrance_canopy(m, w, d, base, mats):
    m.add(w * 0.18, -0.75, base + 1.38, w * 0.82, 0.02, base + 1.52, mats["roof"])
    m.add(w * 0.22, -0.66, base + 1.40, w * 0.78, -0.55, base + 1.44, mats["accent"])


def _flag(m, x, y, z, mats):
    m.add(x - 0.035, y - 0.035, z, x + 0.035, y + 0.035, z + 2.0, mats["trim"])
    m.add(x + 0.035, y - 0.035, z + 1.55, x + 0.78, y + 0.035, z + 1.92, mats["accent"])


def _market_stalls(m, w, d, base, mats, rng):
    for i in range(3):
        x = 0.35 + i * (w - 0.9) / 2.6
        m.add(x, d * 0.12, base, x + 0.9, d * 0.42, base + 0.72, mats["wall"])
        m.add(x - 0.10, d * 0.08, base + 0.72, x + 1.0, d * 0.46, base + 0.82,
              mats["accent"] if i % 2 else mats["trim"])
        m.add(x + 0.12, d * 0.16, base + 0.16, x + 0.72, d * 0.28, base + 0.30,
              mats["dark"], outline=False)


def plan(seed: int = 7, kind: str | None = None):
    """Build one seeded building model and return it with its metadata."""
    rng = np.random.default_rng(int(seed) + 0x0B17)
    if kind is None:
        weights = np.array([
            1.4, 1.2, 1.2, 1.0, 0.9, 0.9, 0.7, 0.7, 0.6, 0.5,
            0.7, 0.5, 0.4, 0.4, 0.7, 0.5, 0.7, 0.5,
        ])
        kind = str(rng.choice(KINDS, p=weights / weights.sum()))
    if kind not in KINDS:
        raise ValueError(f"unknown building {kind!r}")
    mats = _materials(rng, kind)
    m = voxel.Model()

    if kind == "house":
        w, d = rng.uniform(3.2, 5.0), rng.uniform(3.0, 4.4)
        floors = 1
        fh = rng.uniform(1.5, 1.9)
    elif kind == "apartment":
        w, d = rng.uniform(3.6, 5.4), rng.uniform(3.2, 4.4)
        floors = int(rng.integers(3, 7))
        fh = 1.15
    elif kind == "office":
        w, d = rng.uniform(3.4, 5.2), rng.uniform(3.2, 4.6)
        floors = int(rng.integers(4, 9))
        fh = 1.05
    elif kind == "shop":
        w, d = rng.uniform(3.0, 4.6), rng.uniform(2.8, 4.0)
        floors = int(rng.integers(1, 3))
        fh = 1.55
    elif kind == "warehouse":
        w, d = rng.uniform(5.0, 7.4), rng.uniform(4.0, 6.0)
        floors = 1
        fh = 2.0
    elif kind == "factory":
        w, d = rng.uniform(5.2, 7.4), rng.uniform(4.2, 6.2)
        floors = int(rng.integers(1, 3))
        fh = 1.85
    elif kind == "clinic":
        w, d = rng.uniform(3.6, 5.4), rng.uniform(3.0, 4.4)
        floors = int(rng.integers(2, 4))
        fh = 1.25
    elif kind == "school":
        w, d = rng.uniform(5.2, 7.6), rng.uniform(3.4, 4.8)
        floors = int(rng.integers(1, 3))
        fh = 1.55
    elif kind == "station":
        w, d = rng.uniform(5.4, 7.8), rng.uniform(3.0, 4.0)
        floors = 1
        fh = 1.75
    elif kind == "hangar":
        w, d = rng.uniform(5.4, 7.8), rng.uniform(4.2, 6.0)
        floors = 1
        fh = 2.2
    elif kind == "barracks":
        w, d = rng.uniform(4.4, 6.6), rng.uniform(3.0, 4.2)
        floors = 2
        fh = 1.30
    elif kind == "power_plant":
        w, d = rng.uniform(4.0, 6.4), rng.uniform(3.4, 5.0)
        floors = 1
        fh = 2.1
    elif kind == "water_tower":
        w, d = rng.uniform(2.8, 4.0), rng.uniform(2.8, 4.0)
        floors = 1
        fh = 0.7
    elif kind == "bunker":
        w, d = rng.uniform(4.0, 6.0), rng.uniform(3.4, 5.0)
        floors = 1
        fh = 1.2
    elif kind == "tower":
        w, d = rng.uniform(2.8, 4.2), rng.uniform(2.8, 4.2)
        floors = int(rng.integers(7, 13))
        fh = 0.95
    elif kind == "market":
        w, d = rng.uniform(4.4, 6.4), rng.uniform(3.0, 4.4)
        floors = 1
        fh = 1.7
    elif kind == "hotel":
        w, d = rng.uniform(3.8, 5.6), rng.uniform(3.2, 4.4)
        floors = int(rng.integers(4, 8))
        fh = 1.15
    else:
        w, d = rng.uniform(4.2, 6.0), rng.uniform(3.6, 5.0)
        floors = 2
        fh = 1.4

    base = 0.12
    height = base + floors * fh
    m.add(-0.34, -0.34, 0.0, w + 0.34, d + 0.34, base, mats["concrete"])
    if kind == "tower":
        for floor in range(floors):
            inset = 0.12 * (floor // 3)
            m.add(inset, inset, base + floor * fh,
                  w - inset, d - inset, base + (floor + 1) * fh,
                  mats["wall"] if floor % 2 == 0 else mats["glass"])
    elif kind == "water_tower":
        for x, y in ((0.55, 0.55), (w - 0.55, 0.55), (0.55, d - 0.55), (w - 0.55, d - 0.55)):
            m.add(x - 0.10, y - 0.10, base, x + 0.10, y + 0.10, base + 2.0, mats["trim"])
        for i in range(5):
            inset = 0.30 + i * 0.16
            m.add(inset, inset, base + 2.0 + i * 0.30,
                  w - inset, d - inset, base + 2.3 + i * 0.30,
                  mats["wall"])
        m.add(w * 0.45, d * 0.45, base + 3.5, w * 0.55, d * 0.55, base + 4.2, mats["dark"])
    elif kind == "bunker":
        m.add(0, 0, base, w, d, base + fh * 0.66, mats["concrete"])
        for i in range(3):
            inset = 0.16 * i
            m.add(inset, inset, base + fh * 0.66 + i * 0.24,
                  w - inset, d - inset, base + fh * 0.92 + i * 0.24,
                  mats["concrete"])
    else:
        m.add(0, 0, base, w, d, height, mats["wall"])
        m.add(-0.04, -0.04, base, w + 0.04, d + 0.04, base + 0.12, mats["trim"])

    if kind not in ("bunker", "water_tower"):
        _windows(m, w, d, base, fh, floors, mats, rng=rng)
    if kind in ("apartment", "hotel"):
        _balconies(m, w, d, base, fh, floors, mats)
    if kind in ("shop", "clinic", "office", "station", "fire_station", "market", "hotel"):
        _entrance_canopy(m, w, d, base, mats)
    if kind == "house":
        _porch(m, w, d, base, mats)
        _door(m, w, d, base, mats, width=0.9)
    elif kind in ("shop", "clinic", "station", "fire_station", "hotel"):
        _door(m, w, d, base, mats, width=1.25)
    elif kind == "warehouse":
        _roll_door(m, w, base, mats)
        _door(m, w, d, base, mats, width=0.85, side="east")
    elif kind == "hangar":
        _roll_door(m, w, base, mats)
    elif kind not in ("tower", "water_tower", "bunker"):
        _door(m, w, d, base, mats, width=0.9)

    if kind in ("house", "barracks", "hotel", "apartment"):
        _roof(m, w, d, height, mats, kind, rng)
    elif kind in ("factory", "warehouse", "hangar"):
        _roof(m, w, d, height, mats, kind, rng)
    elif kind == "power_plant":
        _roof(m, w, d, height, mats, kind, rng)
        _chimney(m, w * 0.25, d * 0.30, height, mats, height=3.0, radius=0.34)
        _chimney(m, w * 0.70, d * 0.34, height, mats, height=2.5, radius=0.28)
        _cooling_tower(m, w * 0.50, d * 0.76, base, mats, scale=0.8)
    elif kind == "tower":
        _roof(m, w, d, height, mats, kind, rng)
        _antenna(m, w * 0.5, d * 0.5, height + 0.2, mats)
    elif kind == "water_tower":
        pass
    elif kind == "bunker":
        pass
    elif kind == "market":
        _roof(m, w, d, height, mats, kind, rng)
        _market_stalls(m, w, d, base, mats, rng)
    elif kind == "school":
        _roof(m, w, d, height, mats, "office", rng)
        _flag(m, w * 0.88, d * 0.82, base, mats)
    elif kind == "station":
        _roof(m, w, d, height, mats, "office", rng)
        m.add(w * 0.12, -0.9, base + 1.72, w * 0.88, 0.10, base + 2.05, mats["roof"])
        for x in (w * 0.18, w * 0.82):
            m.add(x - 0.10, -0.82, base, x + 0.10, -0.62, base + 1.72, mats["trim"])
    elif kind == "fire_station":
        _roof(m, w, d, height, mats, "office", rng)
        m.add(w * 0.18, -0.08, base + 0.10, w * 0.82, 0.02, base + 1.45, mats["dark"])
    else:
        _roof(m, w, d, height, mats, kind, rng)

    if kind == "shop":
        m.add(w * 0.10, -0.05, base + 1.82, w * 0.90, 0.02, base + 2.18, mats["accent"])
        for i in range(5):
            m.add(w * 0.10 + i * w * 0.16, -0.22, base + 1.55,
                  w * 0.10 + (i + 1) * w * 0.16, 0.01, base + 1.82,
                  mats["accent"] if i % 2 == 0 else mats["wall"])
    if kind == "clinic":
        m.add(w * 0.40, -0.05, height + 0.18, w * 0.60, 0.01, height + 0.54, mats["accent"])
        m.add(w * 0.45, -0.06, height + 0.26, w * 0.55, 0.02, height + 0.46, mats["wall"])
    if kind == "hotel":
        m.add(w * 0.20, -0.05, height + 0.18, w * 0.80, 0.01, height + 0.48, mats["accent"])
    if kind == "market":
        for i in range(3):
            m.add(0.18 + i * 0.50, d * 0.82, 0.0, 0.50 + i * 0.50, d * 0.98, 0.22,
                  mats["concrete"])

    x0, y0, z0, x1, y1, z1 = m.bounds()
    meta = {
        "seed": int(seed),
        "kind": kind,
        "label": _LABELS[kind],
        "views": list(voxel.VIEWS),
        "footprint": [round(float(w), 4), round(float(d), 4)],
        "floors": int(floors),
        "height": round(float(z1), 4),
        "parts": len(m.parts),
        "palette": {key: _hex(value) for key, value in mats.items()},
        "size": [round(float(w), 4), round(float(d), 4), round(float(z1), 4)],
        "centre": [round(float((x0 + x1) * 0.5), 4),
                   round(float((y0 + y1) * 0.5), 4),
                   round(float((z0 + z1) * 0.5), 4)],
    }
    return m, meta


def render(seed: int = 7, kind: str | None = None, size: int = 192,
           view: str = "south", margin: int = 12) -> tuple:
    """One projection of one building, with its metadata."""
    m, meta = plan(seed, kind)
    return voxel.render(m.parts, size=size, view=view, margin=margin), meta


def views(seed: int = 7, kind: str | None = None, size: int = 192,
          margin: int = 12) -> tuple:
    """All five projections of one building, with common scale and metadata."""
    m, meta = plan(seed, kind)
    return voxel.views(m.parts, size=size, margin=margin), meta


def sheet(seed: int = 7, count: int = 4, size: int = 160, kinds=None,
          gap: int = 8) -> tuple:
    """A five-column turnaround sheet: one row per building."""
    rng = np.random.default_rng(int(seed) + 0x5EED)
    selected = list(kinds) if kinds else [
        str(rng.choice(KINDS)) for _ in range(max(1, int(count)))
    ]
    selected = selected[:max(1, int(count))]
    frames, records = [], []
    for i, kind in enumerate(selected):
        grid, meta = views(seed + 17 * i, kind, size=size)
        frames.extend(grid[v] for v in voxel.VIEWS)
        meta["index"] = i
        meta["row"] = i
        records.append(meta)
    out = renderlib.sheet(frames, cols=len(voxel.VIEWS), gap=gap,
                      background=(0.92, 0.93, 0.95))
    return out, records
