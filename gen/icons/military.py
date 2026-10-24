"""Military and grand-strategy charges for the icon set.

The first nine charges are resources: what a province *has*.  These are what a
unit or a division *is* -- infantry, armour, artillery, air, naval, airborne,
signals, medical, logistics, command -- drawn in the same negative-inside SDF
vocabulary as the coin and the grain.  A NATO-style symbol and a Hearts-of-Iron
style counter want the same thing from a 32 px icon: one silhouette that survives
the plate, the bevel and the badge.

The shapes are deliberately canonical rather than ornate.  A tank is a hull, two
tracks, a turret and a barrel; a parachute is a canopy, suspension lines and a
payload.  The seed may lean the symbol, but it may not turn it into another kind.
"""
from __future__ import annotations

import numpy as np

from core import sdf

__all__ = ["MILITARY", "FAMILIES"]


def _u(*parts):
    out = None
    for part in parts:
        out = part if out is None else sdf.union(out, part)
    return out


def _rifle(x, y, s: dict):
    t = 0.026 + 0.011 * s["weight"]
    barrel = sdf.segment(x, y, (-0.20, -0.22), (0.34, 0.30), t)
    muzzle = sdf.circle(x, y, t * 1.55, 0.35, 0.31)
    stock = sdf.polygon(x, y, [(-0.40, -0.40), (-0.20, -0.20),
                               (-0.27, -0.11), (-0.46, -0.31)])
    grip = sdf.segment(x, y, (-0.05, -0.10), (-0.13, -0.27), t * 1.20)
    magazine = sdf.polygon(x, y, [(0.00, -0.02), (0.12, 0.03),
                                  (0.08, -0.19), (-0.03, -0.23)])
    sight = sdf.segment(x, y, (0.04, 0.02), (0.10, 0.09), t * 1.30)
    return _u(barrel, muzzle, stock, grip, magazine, sight)


def _tank(x, y, s: dict):
    # Tracks run up and down the tile, so this is a tank seen from above: the
    # arrangement a counter uses, and the one that stays readable when the plate
    # is only 32 px high.
    tw = 0.070
    tracks = _u(
        sdf.box(x, y, tw, 0.335, 0.045, -0.245, -0.005),
        sdf.box(x, y, tw, 0.335, 0.045, 0.245, -0.005),
    )
    hull = sdf.box(x, y, 0.205, 0.255, 0.075, 0.0, -0.010)
    turret = sdf.box(x, y, 0.125, 0.115, 0.060, 0.0, 0.020)
    barrel = sdf.segment(x, y, (0.0, 0.060), (0.0, 0.435), 0.030)
    return _u(tracks, hull, turret, barrel)


def _cannon(x, y, s: dict):
    wheel = _u(sdf.ring(x, y, 0.185, 0.052, -0.155, -0.205),
               sdf.circle(x, y, 0.052, -0.155, -0.205))
    barrel = sdf.segment(x, y, (-0.10, -0.055), (0.425, 0.195), 0.052 + 0.012 * s["weight"])
    muzzle = sdf.circle(x, y, 0.070, 0.425, 0.195)
    trail = sdf.segment(x, y, (-0.150, -0.170), (-0.365, -0.350), 0.040)
    shield = sdf.polygon(x, y, [(-0.255, -0.085), (0.035, -0.020),
                                (0.035, -0.295), (-0.205, -0.300)])
    return _u(wheel, barrel, muzzle, trail, shield)


def _plane(x, y, s: dict):
    fuselage = sdf.segment(x, y, (0.0, -0.395), (0.0, 0.360), 0.050)
    nose = sdf.polygon(x, y, [(-0.058, 0.310), (0.058, 0.310), (0.0, 0.465)])
    wings = sdf.polygon(x, y, [(-0.455, 0.035), (-0.170, 0.115),
                               (0.170, 0.115), (0.455, 0.035),
                               (0.270, -0.055), (0.170, -0.100),
                               (-0.170, -0.100), (-0.270, -0.055)])
    tail = sdf.polygon(x, y, [(-0.290, -0.310), (-0.100, -0.240),
                              (0.100, -0.240), (0.290, -0.310),
                              (0.170, -0.400), (-0.170, -0.400)])
    fin = sdf.segment(x, y, (0.0, -0.300), (0.0, -0.435), 0.040)
    return _u(fuselage, nose, wings, tail, fin)


def _anchor(x, y, s: dict):
    ring = sdf.ring(x, y, 0.105, 0.035, 0.0, 0.350)
    shank = sdf.segment(x, y, (0.0, 0.270), (0.0, -0.250), 0.046)
    stock = sdf.segment(x, y, (-0.200, 0.220), (0.200, 0.220), 0.040)
    arms = _u(
        sdf.segment(x, y, (0.0, -0.210), (-0.300, -0.330), 0.055),
        sdf.segment(x, y, (0.0, -0.210), (0.300, -0.330), 0.055),
        sdf.polygon(x, y, [(-0.300, -0.330), (-0.365, -0.220),
                           (-0.250, -0.380)]),
        sdf.polygon(x, y, [(0.300, -0.330), (0.365, -0.220),
                           (0.250, -0.380)]),
    )
    return _u(ring, shank, stock, arms)


def _helmet(x, y, s: dict):
    dome = sdf.polygon(x, y, [(-0.335, 0.015), (-0.300, 0.205),
                              (-0.170, 0.325), (0.170, 0.325),
                              (0.300, 0.205), (0.335, 0.015)])
    brim = sdf.box(x, y, 0.395, 0.046, 0.025, 0.0, -0.005)
    band = sdf.box(x, y, 0.345, 0.025, 0.012, 0.0, -0.060)
    return _u(dome, brim, band)


def _parachute(x, y, s: dict):
    canopy = sdf.polygon(x, y, [
        (-0.390, 0.070), (-0.330, 0.235), (-0.205, 0.345),
        (0.0, 0.385), (0.205, 0.345), (0.330, 0.235), (0.390, 0.070),
    ])
    lines = _u(
        sdf.segment(x, y, (-0.330, 0.095), (-0.090, -0.270), 0.020),
        sdf.segment(x, y, (-0.110, 0.090), (-0.045, -0.270), 0.020),
        sdf.segment(x, y, (0.110, 0.090), (0.045, -0.270), 0.020),
        sdf.segment(x, y, (0.330, 0.095), (0.090, -0.270), 0.020),
    )
    payload = sdf.box(x, y, 0.125, 0.090, 0.025, 0.0, -0.325)
    return _u(canopy, lines, payload)


def _crosshair(x, y, s: dict):
    ring = sdf.ring(x, y, 0.285, 0.025)
    ticks = _u(
        sdf.segment(x, y, (0.0, 0.300), (0.0, 0.440), 0.030),
        sdf.segment(x, y, (0.0, -0.300), (0.0, -0.440), 0.030),
        sdf.segment(x, y, (0.300, 0.0), (0.440, 0.0), 0.030),
        sdf.segment(x, y, (-0.300, 0.0), (-0.440, 0.0), 0.030),
    )
    dot = sdf.circle(x, y, 0.040)
    return _u(ring, ticks, dot)


def _chevron(x, y, s: dict):
    def band(dy):
        return sdf.polygon(x, y, [
            (-0.365, -0.085 + dy), (0.0, 0.205 + dy), (0.365, -0.085 + dy),
            (0.365, 0.035 + dy), (0.0, 0.325 + dy), (-0.365, 0.035 + dy),
        ])
    return _u(band(0.0), band(-0.235))


def _star(x, y, s: dict):
    pts = []
    for i in range(10):
        r = 0.445 if i % 2 == 0 else 0.190
        a = np.pi / 2.0 + i * np.pi / 5.0
        pts.append((r * np.cos(a), r * np.sin(a)))
    return sdf.polygon(x, y, pts)


def _crown(x, y, s: dict):
    body = sdf.polygon(x, y, [
        (-0.390, -0.220), (0.390, -0.220), (0.335, 0.115),
        (0.195, -0.030), (0.0, 0.310), (-0.195, -0.030), (-0.335, 0.115),
    ])
    base = sdf.box(x, y, 0.390, 0.055, 0.015, 0.0, -0.300)
    jewels = _u(sdf.circle(x, y, 0.050, 0.0, 0.040),
                sdf.circle(x, y, 0.042, -0.285, 0.115),
                sdf.circle(x, y, 0.042, 0.285, 0.115))
    return _u(body, base, jewels)


def _supply(x, y, s: dict):
    crate = sdf.box(x, y, 0.315, 0.315, 0.030)
    straps = _u(
        sdf.box(x, y, 0.315, 0.038, 0.0, 0.0, 0.205),
        sdf.box(x, y, 0.315, 0.038, 0.0, 0.0, -0.205),
        sdf.box(x, y, 0.038, 0.315, 0.0, 0.0, 0.0),
    )
    return sdf.smooth_difference(crate, straps, 0.018)


def _truck(x, y, s: dict):
    cargo = sdf.box(x, y, 0.235, 0.170, 0.020, -0.095, 0.090)
    cab = sdf.polygon(x, y, [(0.150, -0.080), (0.320, -0.080),
                             (0.365, 0.080), (0.150, 0.155)])
    chassis = sdf.box(x, y, 0.390, 0.040, 0.010, 0.0, -0.135)
    wheels = _u(sdf.circle(x, y, 0.078, -0.215, -0.190),
                sdf.circle(x, y, 0.078, 0.225, -0.190))
    hubs = _u(sdf.ring(x, y, 0.078, 0.022, -0.215, -0.190),
              sdf.ring(x, y, 0.078, 0.022, 0.225, -0.190))
    return _u(cargo, cab, chassis, sdf.smooth_difference(wheels, hubs, 0.008))


def _signal(x, y, s: dict):
    mast = sdf.segment(x, y, (0.0, -0.430), (0.0, 0.160), 0.060)
    waves = _u(
        sdf.segment(x, y, (0.0, 0.160), (-0.250, 0.360), 0.052),
        sdf.segment(x, y, (0.0, 0.160), (0.250, 0.360), 0.052),
        sdf.segment(x, y, (0.0, 0.160), (-0.405, 0.185), 0.046),
        sdf.segment(x, y, (0.0, 0.160), (0.405, 0.185), 0.046),
    )
    base = sdf.box(x, y, 0.205, 0.055, 0.018, 0.0, -0.410)
    return _u(mast, waves, base)


def _medical(x, y, s: dict):
    arm = 0.145
    length = 0.410
    return _u(sdf.box(x, y, arm, length, 0.030),
              sdf.box(x, y, length, arm, 0.030))


def _fort(x, y, s: dict):
    # Tall keep plus two lower towers.  A single wide block is a tank's top-down
    # hull at small size; the stepped skyline is what makes this a castle.
    keep = sdf.box(x, y, 0.170, 0.335, 0.015, 0.0, -0.035)
    tower = _u(sdf.box(x, y, 0.080, 0.245, 0.012, -0.290, -0.125),
               sdf.box(x, y, 0.080, 0.245, 0.012, 0.290, -0.125))
    crenels = _u(*[sdf.box(x, y, 0.034, 0.070, 0.006, dx, dy)
                   for dx, dy in ((-0.120, 0.335), (0.0, 0.335), (0.120, 0.335),
                                  (-0.290, 0.145), (0.290, 0.145))])
    gate = sdf.union(sdf.box(x, y, 0.072, 0.145, 0.008, 0.0, -0.230),
                     sdf.circle(x, y, 0.072, 0.0, -0.230))
    return sdf.smooth_difference(_u(keep, tower, crenels), gate, 0.014)


def _ship(x, y, s: dict):
    hull = sdf.polygon(x, y, [(-0.410, -0.210), (0.410, -0.210),
                              (0.315, -0.375), (-0.300, -0.375)])
    deck = sdf.box(x, y, 0.410, 0.048, 0.015, 0.0, -0.165)
    tower = sdf.box(x, y, 0.100, 0.150, 0.020, -0.025, 0.005)
    bridge = sdf.box(x, y, 0.185, 0.045, 0.010, -0.025, 0.100)
    mast = sdf.segment(x, y, (-0.025, 0.100), (-0.025, 0.430), 0.028)
    yard = sdf.segment(x, y, (-0.190, 0.335), (0.140, 0.335), 0.025)
    return _u(hull, deck, tower, bridge, mast, yard)


def _rocket(x, y, s: dict):
    nose = sdf.polygon(x, y, [(-0.115, 0.245), (0.115, 0.245), (0.0, 0.455)])
    body = sdf.box(x, y, 0.115, 0.260, 0.030, 0.0, 0.005)
    fins = _u(
        sdf.polygon(x, y, [(-0.105, -0.145), (-0.300, -0.350),
                           (-0.105, -0.360)]),
        sdf.polygon(x, y, [(0.105, -0.145), (0.300, -0.350),
                           (0.105, -0.360)]),
    )
    flame = sdf.polygon(x, y, [(-0.075, -0.250), (0.075, -0.250), (0.0, -0.440)])
    return _u(nose, body, fins, flame)


#: Military charge -> hue.  The hues stay in the same OKLCH script: olive and
#: steel for land, blue for air and sea, signal red for command and medicine.
MILITARY = {
    "rifle": (_rifle, 0.095),
    "tank": (_tank, 0.125),
    "cannon": (_cannon, 0.075),
    "plane": (_plane, 0.565),
    "anchor": (_anchor, 0.620),
    "helmet": (_helmet, 0.090),
    "parachute": (_parachute, 0.700),
    "crosshair": (_crosshair, 0.010),
    "chevron": (_chevron, 0.055),
    "star": (_star, 0.135),
    "crown": (_crown, 0.120),
    "supply": (_supply, 0.100),
    "truck": (_truck, 0.085),
    "signal": (_signal, 0.030),
    "medical": (_medical, 0.985),
    "fort": (_fort, 0.070),
    "ship": (_ship, 0.590),
    "rocket": (_rocket, 0.020),
}

#: The two families a game can ask for by name.  A charge may belong to both; the
#: difference is the *set* a caller draws, not a second shape.  This is the
#: metadata a division designer reads when it wants a NATO-style palette or a
#: HOI4-style counter.
FAMILIES = {
    "nato": ("rifle", "tank", "cannon", "plane", "anchor", "parachute",
             "crosshair", "chevron", "star", "signal", "medical", "fort"),
    "hoi4": ("rifle", "tank", "cannon", "plane", "ship", "truck", "supply",
             "helmet", "rocket", "crown", "star", "signal", "medical", "fort"),
}
