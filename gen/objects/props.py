"""A modern object library: machines, vehicles, aircraft, furniture and kit.

These are the things that stand on a map or in a room, not the map itself.  Every
object is assembled from the same box vocabulary as the buildings, so a lathe and
a tank obey the same axonometric camera and light.  The list is intentionally broad
rather than a single showcase model: a strategy game needs a bench, a server rack,
a truck and a fighter before it needs a hero prop.

Each kind has a seed, and the seed changes proportions, material colours and a few
details without changing what the object *is*.  A tank stays a tank; it does not
become a truck because the noise moved.
"""
from __future__ import annotations

import numpy as np

from core import palette, render as renderlib
from gen import voxel

__all__ = ["KINDS", "LABELS", "plan", "render", "views", "sheet"]

KINDS = (
    "workbench", "lathe", "milling_machine", "press", "conveyor", "welder",
    "generator", "compressor", "server_rack", "computer", "desk", "chair",
    "table", "sofa", "bookcase", "cabinet", "fridge", "stove", "bed", "sink",
    "car", "truck", "bus", "tank", "apc", "artillery", "plane", "helicopter",
    "drone", "forklift", "crane", "container", "trailer", "radar", "fuel_tank",
    "barrel", "pallet", "satellite", "wind_turbine",
)

LABELS = {
    "workbench": "Workbench",
    "lathe": "Lathe",
    "milling_machine": "Milling machine",
    "press": "Press",
    "conveyor": "Conveyor",
    "welder": "Welder",
    "generator": "Generator",
    "compressor": "Compressor",
    "server_rack": "Server rack",
    "computer": "Computer",
    "desk": "Desk",
    "chair": "Chair",
    "table": "Table",
    "sofa": "Sofa",
    "bookcase": "Bookcase",
    "cabinet": "Cabinet",
    "fridge": "Refrigerator",
    "stove": "Stove",
    "bed": "Bed",
    "sink": "Sink",
    "car": "Car",
    "truck": "Truck",
    "bus": "Bus",
    "tank": "Tank",
    "apc": "Armoured carrier",
    "artillery": "Artillery",
    "plane": "Aircraft",
    "helicopter": "Helicopter",
    "drone": "Drone",
    "forklift": "Forklift",
    "crane": "Crane",
    "container": "Container",
    "trailer": "Trailer",
    "radar": "Radar",
    "fuel_tank": "Fuel tank",
    "barrel": "Barrel",
    "pallet": "Pallet",
    "satellite": "Satellite",
    "wind_turbine": "Wind turbine",
}


def _rgb(rng, low=0.38, high=0.72, chroma=(0.02, 0.09)):
    return palette.oklch_to_srgb((float(rng.uniform(low, high)),
                                  float(rng.uniform(*chroma)),
                                  float(rng.random())))


def _hex(rgb):
    return palette.srgb_to_hex(np.asarray(rgb, dtype=np.float64))


def _materials(rng):
    metal = _rgb(rng, 0.46, 0.68, (0.01, 0.055))
    painted = _rgb(rng, 0.38, 0.62, (0.08, 0.18))
    dark = palette.oklch_to_srgb((0.18 + 0.06 * rng.random(), 0.012, 0.12))
    rubber = palette.oklch_to_srgb((0.11 + 0.04 * rng.random(), 0.01, 0.12))
    glass = palette.mix(palette.oklch_to_srgb((0.56, 0.08, 0.54)),
                        (0.04, 0.08, 0.11), 0.46)
    wood = palette.oklch_to_srgb((0.48 + 0.12 * rng.random(), 0.075, 0.08))
    fabric = _rgb(rng, 0.36, 0.58, (0.06, 0.16))
    hazard = palette.hex_to_srgb("#e0b23e")
    white = palette.oklch_to_srgb((0.88, 0.012, 0.10))
    accent = _rgb(rng, 0.48, 0.68, (0.10, 0.20))
    concrete = palette.oklch_to_srgb((0.66, 0.018, 0.10))
    return {
        "metal": metal, "painted": painted, "dark": dark, "rubber": rubber,
        "glass": glass, "wood": wood, "fabric": fabric, "hazard": hazard,
        "white": white, "concrete": concrete, "accent": accent,
    }


def _wheel(m, x, y, z, mats, width=0.26, radius=0.34):
    m.add(x - width, y - radius, z, x + width, y + radius, z + radius * 2,
          mats["rubber"], outline=False)
    m.add(x - width * 0.70, y - radius * 0.45, z + radius * 0.55,
          x + width * 0.70, y + radius * 0.45, z + radius * 1.45,
          mats["metal"], outline=False)


def _wheels(m, w, d, mats, axle=0.0):
    for x in (0.55, w - 0.55):
        for y in (0.55, d - 0.55):
            _wheel(m, x, y, axle, mats)


def _machine_body(m, w, d, h, mats):
    m.add(0, 0, 0, w, d, 0.15, mats["dark"])
    m.add(0.18, 0.18, 0.15, w - 0.18, d - 0.18, h, mats["painted"])
    m.add(w * 0.60, d * 0.14, h * 0.58, w * 0.92, d * 0.28, h * 0.86,
          mats["glass"], outline=False)
    m.add(w * 0.64, d * 0.10, h * 0.62, w * 0.88, d * 0.16, h * 0.82,
          mats["dark"], outline=False)


def _bench(m, w, d, mats):
    m.add(0, 0, 0, w, d, 0.18, mats["wood"])
    for x in (0.16, w - 0.28):
        for y in (0.18, d - 0.28):
            m.add(x, y, 0.0, x + 0.12, y + 0.12, 0.92, mats["metal"])
    m.add(0.10, 0.10, 0.78, w - 0.10, d - 0.10, 0.90, mats["wood"])
    m.add(w * 0.55, d * 0.15, 0.90, w * 0.78, d * 0.34, 1.15, mats["painted"])
    m.add(w * 0.20, d * 0.62, 0.90, w * 0.35, d * 0.74, 1.02, mats["hazard"])


def _lathe(m, mats, rng):
    w, d = rng.uniform(1.9, 2.5), rng.uniform(0.9, 1.25)
    m.add(0, 0, 0, w, d, 0.18, mats["dark"])
    m.add(0.18, 0.18, 0.18, w - 0.18, d - 0.18, 0.66, mats["metal"])
    m.add(0.18, d * 0.32, 0.66, w - 0.18, d * 0.68, 0.88, mats["painted"])
    m.add(w * 0.72, d * 0.38, 0.88, w * 0.96, d * 0.62, 1.05, mats["dark"])
    m.add(w * 0.82, d * 0.46, 0.62, w * 0.98, d * 0.54, 0.98, mats["metal"])
    for i in range(4):
        m.add(0.42 + i * 0.22, d * 0.70, 0.88, 0.54 + i * 0.22, d * 0.94, 0.98,
              mats["hazard"], outline=False)


def _milling(m, mats, rng):
    w, d = rng.uniform(1.5, 2.0), rng.uniform(1.2, 1.6)
    _machine_body(m, w, d, 1.8, mats)
    m.add(w * 0.26, d * 0.40, 1.18, w * 0.56, d * 0.60, 1.58, mats["metal"])
    m.add(w * 0.40, d * 0.46, 0.66, w * 0.46, d * 0.54, 1.20, mats["dark"])
    m.add(w * 0.08, d * 0.08, 1.42, w * 0.34, d * 0.30, 1.62, mats["hazard"])


def _press(m, mats, rng):
    w, d = rng.uniform(1.4, 1.9), rng.uniform(1.3, 1.7)
    m.add(0, 0, 0, w, d, 0.22, mats["dark"])
    for x in (0.10, w - 0.30):
        m.add(x, 0.10, 0.22, x + 0.20, d - 0.10, 2.3, mats["painted"])
    m.add(0.10, 0.10, 2.12, w - 0.10, d - 0.10, 2.45, mats["metal"])
    m.add(w * 0.36, d * 0.30, 0.22, w * 0.64, d * 0.70, 0.48, mats["metal"])
    m.add(w * 0.40, d * 0.34, 1.72, w * 0.60, d * 0.66, 2.12, mats["dark"])


def _conveyor(m, mats, rng):
    w, d = rng.uniform(3.0, 4.2), rng.uniform(0.8, 1.1)
    for x in (0.12, w - 0.25):
        for y in (0.12, d - 0.25):
            m.add(x, y, 0.0, x + 0.13, y + 0.13, 0.85, mats["metal"])
    m.add(0, 0, 0.82, w, d, 0.96, mats["dark"])
    for i in range(int(w / 0.55)):
        m.add(i * 0.55 + 0.08, 0.10, 0.96, i * 0.55 + 0.42, d - 0.10, 1.00,
              mats["rubber"], outline=False)
    m.add(w * 0.30, d * 0.20, 1.00, w * 0.44, d * 0.58, 1.32, mats["hazard"])
    m.add(w * 0.60, d * 0.20, 1.00, w * 0.74, d * 0.58, 1.32, mats["hazard"])


def _welder(m, mats, rng):
    _bench(m, 1.8, 0.9, mats)
    m.add(0.10, 0.10, 0.90, 0.42, 0.40, 1.28, mats["hazard"])
    m.add(0.16, 0.16, 0.96, 0.36, 0.34, 1.18, mats["dark"])
    m.add(1.05, 0.20, 0.90, 1.35, 0.42, 1.24, mats["painted"])
    m.add(0.45, 0.34, 0.90, 0.95, 0.48, 0.98, mats["metal"])


def _generator(m, mats, rng):
    w, d = rng.uniform(1.5, 2.1), rng.uniform(0.9, 1.2)
    m.add(0, 0, 0, w, d, 0.22, mats["dark"])
    m.add(0.14, 0.14, 0.22, w - 0.14, d - 0.14, 1.18, mats["painted"])
    for i in range(5):
        m.add(0.24 + i * 0.25, 0.08, 0.42, 0.38 + i * 0.25, 0.16, 1.02,
              mats["dark"], outline=False)
    m.add(w * 0.70, d * 0.16, 1.18, w * 0.90, d * 0.38, 1.40, mats["metal"])
    m.add(w * 0.20, d * 0.72, 0.22, w * 0.34, d * 0.88, 0.54, mats["hazard"])


def _compressor(m, mats, rng):
    w, d = rng.uniform(1.2, 1.7), rng.uniform(0.9, 1.2)
    m.add(0, 0, 0, w, d, 0.18, mats["dark"])
    for i in range(3):
        m.add(0.10 + i * 0.35, 0.14, 0.18, 0.32 + i * 0.35, d - 0.14, 1.15,
              mats["metal"])
    m.add(w * 0.70, 0.10, 0.18, w - 0.08, 0.30, 1.35, mats["painted"])
    m.add(0.14, d - 0.22, 0.18, w - 0.14, d - 0.12, 0.34, mats["hazard"])


def _server(m, mats, rng):
    w, d = 0.9, 0.75
    m.add(0, 0, 0, w, d, 2.2, mats["dark"])
    for i in range(8):
        m.add(0.08, 0.04, 0.10 + i * 0.25, w - 0.08, 0.08, 0.22 + i * 0.25,
              mats["metal"], outline=False)
        m.add(w - 0.22, 0.04, 0.16 + i * 0.25, w - 0.10, 0.06, 0.19 + i * 0.25,
              mats["hazard"], outline=False)


def _computer(m, mats, rng):
    m.add(0, 0, 0, 1.1, 0.55, 0.08, mats["dark"])
    m.add(0.12, 0.12, 0.08, 0.56, 0.16, 0.55, mats["glass"], outline=False)
    m.add(0.08, 0.08, 0.08, 0.62, 0.20, 0.62, mats["metal"])
    m.add(0.72, 0.10, 0.08, 0.98, 0.42, 0.16, mats["metal"])
    m.add(0.76, 0.14, 0.16, 0.94, 0.38, 0.22, mats["painted"])


def _desk(m, mats, rng):
    m.add(0, 0, 0.68, 1.8, 0.9, 0.82, mats["wood"])
    for x in (0.10, 1.62):
        for y in (0.10, 0.72):
            m.add(x, y, 0, x + 0.10, y + 0.10, 0.68, mats["metal"])
    m.add(0.20, 0.18, 0.82, 0.55, 0.42, 1.05, mats["dark"])
    m.add(0.22, 0.20, 1.00, 0.53, 0.22, 1.30, mats["glass"], outline=False)


def _chair(m, mats, rng):
    m.add(0, 0, 0, 0.62, 0.62, 0.12, mats["fabric"])
    m.add(0, 0.48, 0.12, 0.62, 0.60, 0.92, mats["fabric"])
    for x in (0.08, 0.46):
        for y in (0.08, 0.46):
            m.add(x, y, 0, x + 0.08, y + 0.08, 0.52, mats["metal"])
    m.add(0.24, 0.24, 0.0, 0.38, 0.38, 0.30, mats["metal"])


def _table(m, mats, rng):
    w, d = rng.uniform(1.5, 2.2), rng.uniform(0.9, 1.3)
    m.add(0, 0, 0.72, w, d, 0.84, mats["wood"])
    for x in (0.12, w - 0.24):
        for y in (0.12, d - 0.24):
            m.add(x, y, 0, x + 0.12, y + 0.12, 0.72, mats["wood"])
    m.add(w * 0.08, d * 0.15, 0.84, w * 0.30, d * 0.38, 1.10, mats["white"])


def _sofa(m, mats, rng):
    w, d = rng.uniform(1.7, 2.4), 0.85
    m.add(0, 0, 0.18, w, d, 0.52, mats["fabric"])
    m.add(0, d - 0.20, 0.52, w, d, 1.02, mats["fabric"])
    m.add(0.12, 0.08, 0.52, w - 0.12, d - 0.18, 0.64, mats["fabric"])
    for x in (0.10, w - 0.30):
        m.add(x, 0.10, 0.0, x + 0.20, d - 0.10, 0.24, mats["dark"])
    m.add(0.22, 0.14, 0.64, 0.55, 0.42, 0.82, mats["painted"])


def _bookcase(m, mats, rng):
    m.add(0, 0, 0, 1.2, 0.42, 2.0, mats["wood"])
    for i in range(4):
        m.add(0.06, 0.06, 0.14 + i * 0.45, 1.14, 0.38, 0.22 + i * 0.45,
              mats["dark"], outline=False)
        for j in range(4):
            m.add(0.10 + j * 0.25, 0.10, 0.23 + i * 0.45,
                  0.25 + j * 0.25, 0.36, 0.54 + i * 0.45,
                  [mats["painted"], mats["fabric"], mats["hazard"]][j % 3],
                  outline=False)


def _cabinet(m, mats, rng):
    m.add(0, 0, 0, 1.1, 0.48, 1.7, mats["metal"])
    m.add(0.08, 0.02, 0.10, 0.52, 0.05, 1.58, mats["painted"], outline=False)
    m.add(0.58, 0.02, 0.10, 1.02, 0.05, 1.58, mats["painted"], outline=False)
    m.add(0.48, 0.04, 0.82, 0.54, 0.08, 0.94, mats["dark"], outline=False)
    m.add(0.56, 0.04, 0.82, 0.62, 0.08, 0.94, mats["dark"], outline=False)


def _fridge(m, mats, rng):
    m.add(0, 0, 0, 0.85, 0.70, 1.9, mats["white"])
    m.add(0.45, 0.02, 0.04, 0.49, 0.06, 1.86, mats["metal"], outline=False)
    m.add(0.06, 0.02, 0.88, 0.80, 0.05, 0.92, mats["dark"], outline=False)


def _stove(m, mats, rng):
    m.add(0, 0, 0, 0.95, 0.65, 0.92, mats["metal"])
    m.add(0.05, 0.05, 0.92, 0.90, 0.60, 0.98, mats["dark"])
    for x in (0.25, 0.62):
        for y in (0.20, 0.42):
            m.add(x, y, 0.98, x + 0.16, y + 0.12, 1.01, mats["rubber"],
                  outline=False)
    m.add(0.10, 0.02, 0.28, 0.85, 0.05, 0.48, mats["glass"], outline=False)


def _bed(m, mats, rng):
    w, d = rng.uniform(1.7, 2.2), rng.uniform(1.1, 1.5)
    m.add(0, 0, 0.12, w, d, 0.48, mats["wood"])
    m.add(0.08, 0.08, 0.48, w - 0.08, d - 0.08, 0.66, mats["white"])
    m.add(0.10, 0.16, 0.66, w * 0.48, d - 0.10, 0.80, mats["fabric"])
    m.add(0, d - 0.12, 0.12, w, d, 1.05, mats["wood"])
    for x in (0.10, w - 0.22):
        m.add(x, 0.10, 0.0, x + 0.12, 0.22, 0.12, mats["wood"])


def _sink(m, mats, rng):
    m.add(0, 0, 0.62, 1.1, 0.62, 0.78, mats["white"])
    m.add(0.12, 0.12, 0.78, 0.98, 0.50, 0.84, mats["metal"])
    m.add(0.22, 0.22, 0.84, 0.88, 0.42, 0.87, mats["glass"], outline=False)
    m.add(0.70, 0.20, 0.84, 0.76, 0.26, 1.06, mats["metal"])


def _car(m, mats, rng):
    w, d = rng.uniform(2.0, 2.6), 1.15
    m.add(0.15, 0.12, 0.35, w - 0.15, d - 0.12, 0.75, mats["painted"])
    m.add(0.55, 0.20, 0.75, w - 0.55, d - 0.20, 1.20, mats["painted"])
    m.add(0.65, 0.18, 0.92, w - 0.65, 0.22, 1.12, mats["glass"], outline=False)
    m.add(0.55, d - 0.22, 0.82, w - 0.55, d - 0.16, 1.12, mats["glass"], outline=False)
    _wheels(m, w, d, mats, axle=0.10)
    m.add(0.05, 0.18, 0.50, 0.20, d - 0.18, 0.64, mats["hazard"], outline=False)
    m.add(w - 0.20, 0.18, 0.50, w - 0.05, d - 0.18, 0.64, mats["white"], outline=False)


def _truck(m, mats, rng):
    w, d = rng.uniform(3.0, 3.8), 1.25
    m.add(0.10, 0.14, 0.35, 1.15, d - 0.14, 1.30, mats["painted"])
    m.add(0.88, 0.16, 0.90, 1.10, d - 0.16, 1.22, mats["glass"], outline=False)
    m.add(1.25, 0.12, 0.45, w - 0.10, d - 0.12, 1.55, mats["white"])
    for i in range(5):
        m.add(1.32 + i * 0.25, 0.08, 0.55, 1.40 + i * 0.25, 0.12, 1.45,
              mats["painted"], outline=False)
    _wheels(m, w, d, mats, axle=0.08)
    m.add(w - 0.20, 0.20, 0.55, w - 0.08, d - 0.20, 0.95, mats["hazard"], outline=False)


def _bus(m, mats, rng):
    w, d = rng.uniform(3.4, 4.2), 1.3
    m.add(0.10, 0.12, 0.38, w - 0.10, d - 0.12, 1.65, mats["painted"])
    for i in range(6):
        m.add(0.35 + i * 0.52, 0.10, 0.95, 0.75 + i * 0.52, 0.15, 1.42,
              mats["glass"], outline=False)
        m.add(0.35 + i * 0.52, d - 0.15, 0.95, 0.75 + i * 0.52, d - 0.10, 1.42,
              mats["glass"], outline=False)
    _wheels(m, w, d, mats, axle=0.05)
    m.add(0.12, 0.14, 0.55, 0.24, d - 0.14, 0.72, mats["hazard"], outline=False)


def _tank(m, mats, rng):
    w, d = rng.uniform(2.9, 3.5), 1.55
    hull = mats["painted"]
    turret = palette.mix(hull, mats["white"], 0.14)
    for y in (0.02, d - 0.30):
        m.add(0.12, y, 0.04, w - 0.12, y + 0.28, 0.56, mats["rubber"])
        for j in range(4):
            m.add(0.28 + j * 0.62, y - 0.02, 0.12, 0.60 + j * 0.62, y + 0.02, 0.46,
                  mats["dark"], outline=False)
        m.add(0.12, y + 0.02, 0.52, w - 0.12, y + 0.26, 0.62, mats["metal"])
    m.add(0.20, 0.26, 0.30, w - 0.20, d - 0.26, 0.78, hull)
    m.add(0.34, 0.34, 0.78, w - 0.34, d - 0.34, 0.98, hull)
    m.add(0.38, 0.40, 0.98, w * 0.72, d - 0.40, 1.30, turret)
    m.add(w * 0.42, d * 0.38, 1.30, w * 0.66, d * 0.62, 1.42, mats["metal"])
    m.add(w * 0.66, d * 0.46, 1.18, w + 0.92, d * 0.54, 1.28, mats["metal"])
    m.add(w + 0.80, d * 0.44, 1.16, w + 1.02, d * 0.56, 1.30, mats["dark"])
    m.add(w * 0.48, d * 0.42, 1.42, w * 0.60, d * 0.58, 1.50, mats["dark"])
    m.add(w * 0.30, d * 0.46, 1.02, w * 0.40, d * 0.54, 1.16, mats["glass"],
          outline=False)
    m.add(0.18, 0.28, 0.62, 0.38, 0.40, 0.78, mats["hazard"], outline=False)
    m.add(w - 0.28, 0.28, 0.62, w - 0.12, 0.40, 0.78, mats["hazard"], outline=False)
    m.add(w * 0.58, 0.24, 0.82, w * 0.60, 0.30, 0.96, mats["dark"], outline=False)
    m.add(w * 0.58, 1.24, 0.82, w * 0.60, 1.30, 0.96, mats["dark"], outline=False)

def _apc(m, mats, rng):
    w, d = rng.uniform(2.6, 3.2), 1.5
    m.add(0.15, 0.12, 0.38, w - 0.15, d - 0.12, 0.95, mats["painted"])
    m.add(0.30, 0.22, 0.95, w - 0.30, d - 0.22, 1.18, mats["painted"])
    m.add(w * 0.42, d * 0.34, 1.18, w * 0.62, d * 0.66, 1.48, mats["dark"])
    m.add(w * 0.58, d * 0.46, 1.32, w * 0.92, d * 0.54, 1.40, mats["dark"])
    _wheels(m, w, d, mats, axle=0.08)
    m.add(0.18, 0.20, 0.55, 0.30, 0.34, 0.74, mats["hazard"], outline=False)


def _artillery(m, mats, rng):
    w, d = rng.uniform(2.1, 2.6), 1.1
    m.add(0.28, 0.12, 0.25, w - 0.28, d - 0.12, 0.62, mats["dark"])
    m.add(0.42, 0.22, 0.62, 1.05, d - 0.22, 1.05, mats["painted"])
    m.add(0.95, d * 0.40, 0.82, w + 0.95, d * 0.60, 1.02, mats["metal"])
    _wheels(m, w, d, mats, axle=0.08)
    m.add(0.10, 0.18, 0.46, 0.30, d - 0.18, 0.62, mats["hazard"], outline=False)


def _plane(m, mats, rng):
    length = rng.uniform(3.3, 4.2)
    wing = rng.uniform(1.55, 1.95)
    silver = palette.mix(mats["metal"], mats["white"], 0.30)
    m.add(0.0, 0.68, 0.50, length * 0.88, 0.96, 0.92, mats["painted"])
    m.add(length * 0.88, 0.74, 0.56, length, 0.90, 0.82, mats["dark"])
    m.add(length * 0.55, -wing * 0.50, 0.64, length * 0.84, wing * 1.50, 0.76,
          silver)
    m.add(length * 0.64, -wing * 0.50, 0.58, length * 0.76, -wing * 0.18, 0.80,
          mats["painted"])
    m.add(length * 0.64, wing * 1.18, 0.58, length * 0.76, wing * 1.50, 0.80,
          mats["painted"])
    m.add(length * 0.80, 0.54, 0.88, length * 0.98, 1.10, 0.96, silver)
    m.add(length * 0.84, 0.76, 0.96, length * 0.96, 0.90, 1.48, mats["painted"])
    m.add(length * 0.84, 0.78, 1.48, length * 0.96, 0.88, 1.62, mats["accent"])
    m.add(length * 0.42, 0.66, 0.92, length * 0.64, 0.98, 1.12, mats["glass"],
          outline=False)
    m.add(length * 0.44, 0.60, 0.94, length * 0.62, 1.04, 1.10, mats["glass"],
          outline=False)
    for x in (length * 0.50, length * 0.70):
        for y in (0.26, 1.34):
            m.add(x - 0.10, y - 0.14, 0.56, x + 0.10, y + 0.14, 0.86,
                  mats["metal"])
            m.add(x - 0.06, y - 0.10, 0.42, x + 0.06, y + 0.10, 0.56,
                  mats["dark"])
    m.add(length * 0.08, 0.80, 0.92, length * 0.24, 0.98, 0.98, mats["hazard"],
          outline=False)
    m.add(length * 0.52, 0.70, 0.76, length * 0.54, 1.00, 0.84, mats["white"],
          outline=False)

def _helicopter(m, mats, rng):
    length = rng.uniform(2.6, 3.2)
    m.add(0.25, 0.45, 0.62, length * 0.60, 1.12, 1.15, mats["painted"])
    m.add(length * 0.55, 0.68, 0.78, length, 0.90, 0.98, mats["painted"])
    m.add(length * 0.86, 0.60, 1.10, length, 0.98, 1.22, mats["painted"])
    m.add(length * 0.18, 0.35, 0.70, length * 0.42, 0.48, 0.95, mats["metal"])
    m.add(length * 0.34, 0.78, 1.15, length * 0.58, 1.04, 1.38, mats["glass"], outline=False)
    for i in range(3):
        m.add(0.20, 0.55 + i * 0.10, 1.45, length * 0.62, 0.65 + i * 0.10, 1.52,
              mats["dark"], outline=False)
    for x in (0.45, length * 0.48):
        m.add(x - 0.04, 0.30, 0.12, x + 0.04, 0.44, 0.62, mats["metal"])
        m.add(x - 0.04, 1.12, 0.12, x + 0.04, 1.26, 0.62, mats["metal"])
    m.add(0.10, 0.40, 0.12, length * 0.55, 0.50, 0.18, mats["metal"])


def _drone(m, mats, rng):
    m.add(0.55, 0.55, 0.20, 1.25, 1.05, 0.42, mats["painted"])
    for x, y in ((0.20, 0.20), (1.35, 0.20), (0.20, 1.30), (1.35, 1.30)):
        m.add(x, y, 0.30, x + 0.18, y + 0.18, 0.40, mats["dark"])
        m.add(x - 0.22, y + 0.02, 0.40, x + 0.40, y + 0.14, 0.44,
              mats["metal"], outline=False)
    m.add(0.72, 0.72, 0.42, 0.92, 0.88, 0.58, mats["glass"], outline=False)


def _forklift(m, mats, rng):
    w, d = rng.uniform(1.6, 2.0), 1.0
    m.add(0.15, 0.12, 0.20, w - 0.35, d - 0.12, 0.55, mats["painted"])
    m.add(0.18, 0.18, 0.55, w - 0.40, 0.80, 0.95, mats["dark"], outline=False)
    m.add(w - 0.42, 0.10, 0.20, w - 0.22, 0.28, 2.1, mats["metal"])
    m.add(w - 0.48, 0.10, 2.1, w - 0.18, 0.30, 2.2, mats["metal"])
    m.add(w - 0.10, 0.10, 0.18, w + 0.45, 0.24, 0.28, mats["metal"])
    m.add(w - 0.10, 0.36, 0.18, w + 0.45, 0.50, 0.28, mats["metal"])
    _wheels(m, w - 0.25, d, mats, axle=0.05)


def _crane(m, mats, rng):
    m.add(0.34, 0.34, 0.0, 0.66, 0.66, 3.2, mats["hazard"])
    m.add(0.20, 0.20, 3.0, 3.0, 0.50, 3.20, mats["hazard"])
    m.add(0.30, 0.30, 3.2, 0.45, 1.10, 4.4, mats["metal"])
    m.add(2.30, 0.30, 3.2, 2.55, 0.50, 3.6, mats["dark"])
    m.add(2.42, 0.34, 2.4, 2.46, 0.46, 3.2, mats["dark"])
    m.add(2.30, 0.30, 2.2, 2.58, 0.50, 2.34, mats["metal"])


def _container(m, mats, rng):
    w, d = rng.uniform(2.0, 2.8), 1.1
    m.add(0, 0, 0, w, d, 1.3, mats["painted"])
    for i in range(6):
        m.add(0.15 + i * 0.30, -0.02, 0.08, 0.24 + i * 0.30, 0.02, 1.22,
              mats["dark"], outline=False)
    m.add(w - 0.08, 0.08, 0.08, w - 0.02, d - 0.08, 1.22, mats["metal"], outline=False)


def _trailer(m, mats, rng):
    w, d = 2.4, 1.05
    m.add(0.10, 0.12, 0.38, w - 0.10, d - 0.12, 1.25, mats["white"])
    m.add(0.10, 0.12, 0.25, 0.30, d - 0.12, 0.42, mats["metal"])
    _wheels(m, w, d, mats, axle=0.08)
    m.add(w - 0.05, 0.22, 0.42, w + 0.45, 0.34, 0.56, mats["metal"])


def _radar(m, mats, rng):
    m.add(0.40, 0.40, 0.0, 0.70, 0.70, 0.30, mats["concrete"])
    m.add(0.50, 0.50, 0.30, 0.60, 0.60, 1.5, mats["metal"])
    for i in range(5):
        half = 0.25 + 0.18 * i
        m.add(0.55 - half, 0.50, 1.2 + i * 0.15, 0.55 + half, 0.56, 1.3 + i * 0.15,
              mats["white"], outline=False)
    m.add(0.84, 0.52, 1.72, 1.00, 0.58, 1.82, mats["hazard"])


def _fuel_tank(m, mats, rng):
    m.add(0.18, 0.18, 0.0, 0.28, 0.28, 1.9, mats["metal"])
    m.add(0.18, 1.02, 0.0, 0.28, 1.12, 1.9, mats["metal"])
    m.add(0.70, 0.18, 0.0, 0.80, 0.28, 1.9, mats["metal"])
    m.add(0.70, 1.02, 0.0, 0.80, 1.12, 1.9, mats["metal"])
    for i in range(5):
        m.add(0.12, 0.12, 0.10 + i * 0.38, 0.88, 1.18, 0.42 + i * 0.38,
              mats["white"], outline=False)
    m.add(0.18, 0.18, 1.9, 0.82, 1.12, 2.0, mats["metal"])


def _barrel(m, mats, rng):
    m.add(0, 0, 0, 0.55, 0.55, 0.85, mats["painted"])
    for i in range(2):
        m.add(-0.02, -0.02, 0.20 + i * 0.28, 0.57, 0.57, 0.28 + i * 0.28,
              mats["metal"], outline=False)
    m.add(0.18, 0.18, 0.85, 0.32, 0.32, 0.90, mats["dark"], outline=False)


def _pallet(m, mats, rng):
    for i in range(4):
        m.add(0.0, 0.12 + i * 0.28, 0.0, 1.3, 0.20 + i * 0.28, 0.08, mats["wood"])
    for x in (0.10, 0.60, 1.10):
        m.add(x, 0.08, 0.08, x + 0.10, 1.22, 0.18, mats["wood"])
    m.add(0.08, 0.14, 0.18, 1.22, 0.40, 0.82, mats["wood"])
    m.add(0.08, 0.52, 0.18, 1.22, 0.80, 0.82, mats["wood"])


def _satellite(m, mats, rng):
    m.add(0.55, 0.55, 0.55, 1.05, 1.05, 1.0, mats["painted"])
    for x, y in ((0.10, 0.55), (1.05, 0.55), (0.55, 0.10), (0.55, 1.05)):
        m.add(x, y, 0.65, x + 0.45, y + 0.45, 0.72, mats["glass"], outline=False)
    m.add(0.72, 0.72, 1.0, 0.88, 0.88, 1.35, mats["metal"])
    m.add(0.24, 0.24, 0.0, 1.36, 0.30, 1.8, mats["metal"], outline=False)


def _wind_turbine(m, mats, rng):
    m.add(0.58, 0.58, 0.0, 0.72, 0.72, 2.8, mats["white"])
    m.add(0.30, 0.62, 2.6, 1.00, 0.70, 2.9, mats["white"])
    for i in range(3):
        angle = i * np.pi * 2.0 / 3.0
        x = 0.65 + np.cos(angle) * 0.75
        z = 2.75 + np.sin(angle) * 0.75
        m.add(min(0.65, x), 0.62, min(2.75, z), max(0.65, x), 0.70, max(2.75, z),
              mats["white"], outline=False)


def plan(seed: int = 7, kind: str | None = None):
    """Build one seeded modern object and return it with its metadata."""
    rng = np.random.default_rng(int(seed) + 0x11A7)
    if kind is None:
        kind = str(rng.choice(KINDS))
    if kind not in KINDS:
        raise ValueError(f"unknown object {kind!r}")
    mats = _materials(rng)
    m = voxel.Model()
    if kind == "workbench":
        _bench(m, 2.2, 1.0, mats)
    elif kind == "lathe":
        _lathe(m, mats, rng)
    elif kind == "milling_machine":
        _milling(m, mats, rng)
    elif kind == "press":
        _press(m, mats, rng)
    elif kind == "conveyor":
        _conveyor(m, mats, rng)
    elif kind == "welder":
        _welder(m, mats, rng)
    elif kind == "generator":
        _generator(m, mats, rng)
    elif kind == "compressor":
        _compressor(m, mats, rng)
    elif kind == "server_rack":
        _server(m, mats, rng)
    elif kind == "computer":
        _computer(m, mats, rng)
    elif kind == "desk":
        _desk(m, mats, rng)
    elif kind == "chair":
        _chair(m, mats, rng)
    elif kind == "table":
        _table(m, mats, rng)
    elif kind == "sofa":
        _sofa(m, mats, rng)
    elif kind == "bookcase":
        _bookcase(m, mats, rng)
    elif kind == "cabinet":
        _cabinet(m, mats, rng)
    elif kind == "fridge":
        _fridge(m, mats, rng)
    elif kind == "stove":
        _stove(m, mats, rng)
    elif kind == "bed":
        _bed(m, mats, rng)
    elif kind == "sink":
        _sink(m, mats, rng)
    elif kind == "car":
        _car(m, mats, rng)
    elif kind == "truck":
        _truck(m, mats, rng)
    elif kind == "bus":
        _bus(m, mats, rng)
    elif kind == "tank":
        _tank(m, mats, rng)
    elif kind == "apc":
        _apc(m, mats, rng)
    elif kind == "artillery":
        _artillery(m, mats, rng)
    elif kind == "plane":
        _plane(m, mats, rng)
    elif kind == "helicopter":
        _helicopter(m, mats, rng)
    elif kind == "drone":
        _drone(m, mats, rng)
    elif kind == "forklift":
        _forklift(m, mats, rng)
    elif kind == "crane":
        _crane(m, mats, rng)
    elif kind == "container":
        _container(m, mats, rng)
    elif kind == "trailer":
        _trailer(m, mats, rng)
    elif kind == "radar":
        _radar(m, mats, rng)
    elif kind == "fuel_tank":
        _fuel_tank(m, mats, rng)
    elif kind == "barrel":
        _barrel(m, mats, rng)
    elif kind == "pallet":
        _pallet(m, mats, rng)
    elif kind == "satellite":
        _satellite(m, mats, rng)
    elif kind == "wind_turbine":
        _wind_turbine(m, mats, rng)

    x0, y0, z0, x1, y1, z1 = m.bounds()
    meta = {
        "seed": int(seed),
        "kind": kind,
        "label": LABELS[kind],
        "views": list(voxel.VIEWS),
        "parts": len(m.parts),
        "size": [round(float(x1 - x0), 4), round(float(y1 - y0), 4),
                 round(float(z1 - z0), 4)],
        "centre": [round(float((x0 + x1) * 0.5), 4),
                   round(float((y0 + y1) * 0.5), 4),
                   round(float((z0 + z1) * 0.5), 4)],
        "palette": {key: _hex(value) for key, value in mats.items()},
    }
    return m, meta


def render(seed: int = 7, kind: str | None = None, size: int = 160,
           view: str = "north", margin: int = 10) -> tuple:
    m, meta = plan(seed, kind)
    meta["view"] = view
    return voxel.render(m.parts, size=size, view=view, margin=margin), meta


def views(seed: int = 7, kind: str | None = None, size: int = 160,
          margin: int = 10) -> tuple:
    m, meta = plan(seed, kind)
    return voxel.views(m.parts, size=size, margin=margin), meta


def sheet(seed: int = 7, count: int = 12, size: int = 128, cols: int = 6,
          kinds=None, gap: int = 8) -> tuple:
    rng = np.random.default_rng(int(seed) + 0x2EED)
    selected = list(kinds) if kinds else [
        str(rng.choice(KINDS)) for _ in range(max(1, int(count)))
    ]
    selected = selected[:max(1, int(count))]
    frames, records = [], []
    for i, kind in enumerate(selected):
        c, meta = render(seed + 23 * i, kind, size=size)
        frames.append(c)
        meta["index"] = i
        records.append(meta)
    out = renderlib.sheet(frames, cols=max(1, int(cols)), gap=gap,
                          background=(0.92, 0.93, 0.95))
    return out, records
