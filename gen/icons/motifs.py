"""The shapes an icon is made of: the plate it sits on, and the charge it carries.

An icon is a plate with a charge on it, and the charge is the whole content -- so
it has to be legible at 32 px, which rules out most of what a generator would like
to draw.  The vocabulary is therefore small, and every member is a *symbol* rather
than a depiction: a coin is a ring, water is a drop, a blade is a sword seen flat.
Two or three primitives and one boolean is enough for each of them.

The seed is free to move a symbol's proportions about -- how many grains on the
stalk, how wide the blade, which way it leans -- but never to move the symbol
itself, which is the difference between a set of icons and a set of blobs.  That
line is what ``tests/test_icons.py`` measures: a seeded grain still overlaps its
canonical self by most of its area.

Every field here is negative inside, the way :mod:`core.sdf` defines distance, so
the painter can fill it, ink it, carve it or emboss it without knowing what it is.
"""
from __future__ import annotations

import numpy as np

from core import sdf

__all__ = ["KINDS", "PLATES", "kind", "checked", "field", "plate_field", "canonical",
           "random_spec"]


# --------------------------------------------------------------- the charges
def _coin(x, y, s: dict):
    """A ring, with a second ring stamped inside it: money reads at any size."""
    t = 0.042 + 0.030 * s["weight"]
    body = sdf.ring(x, y, 0.33, t)
    if s["detail"] > 0.35:
        body = sdf.union(body, sdf.ring(x, y, 0.16, t * 0.62))
    return body


def _grain(x, y, s: dict):
    """An ear of wheat: one stalk, and a pair of fat grains at each step up it.

    The grains have to be fat and the stalk short, or the whole thing reads as an
    arrow -- which is what the first version of this drew, and the reason an icon
    vocabulary is worth looking at rather than trusting.
    """
    t = 0.020 + 0.014 * s["weight"]
    body = sdf.segment(x, y, (0.0, -0.44), (0.0, 0.10), t)
    levels = 2 + int(round(s["detail"] * 2))
    for j in range(levels):
        y0 = -0.36 + 0.165 * j
        for side in (-1.0, 1.0):
            body = sdf.union(body, sdf.segment(
                x, y, (0.0, y0), (side * (0.16 + 0.030 * j), y0 + 0.17),
                0.072 - 0.006 * j))
    return body


def _wood(x, y, s: dict):
    """A stack of planks leaning the same way -- timber, and not a cross.

    Two planks crossed at a right angle is a plus sign, which is what the first
    version of this drew: a symbol has to be read at 32 px, and an X reads as
    "close", not as "wood".
    """
    t = 0.055 + 0.020 * s["weight"]
    lean = 8.0 + 8.0 * s["weight"]
    body = None
    for k in (-1, 0, 1):
        bar = sdf.rotated(lambda a, b: sdf.box(a, b, 0.36, t, 0.5 * t), lean)
        f = sdf.moved(bar, 0.0, k * 2.35 * t)(x, y)
        body = f if body is None else sdf.smooth_union(body, f, 0.02)
    return body


def _ore(x, y, s: dict):
    """A gem: a brilliant's outline, or -- plainer -- a crystal with three faces.

    The six-sided girdle matters: a five-sided gem is a pentagon, and a pentagon
    sitting on a shield plate is a shield, which is the sort of collision a small
    vocabulary has to be checked for rather than assumed away.
    """
    scale = 0.88 + 0.24 * s["weight"]
    if s["detail"] > 0.6:
        pts = [(-0.30, 0.26), (0.30, 0.26), (0.0, -0.42)]
    else:
        pts = [(-0.24, 0.30), (0.24, 0.30), (0.39, 0.02),
               (0.20, -0.34), (-0.20, -0.34), (-0.39, 0.02)]
    return sdf.polygon(x, y, [(px * scale, py * scale) for px, py in pts])


def _water(x, y, s: dict):
    """A drop: a disc with a tapering spike above it, welded with a fillet."""
    r = 0.24 + 0.03 * s["weight"]
    tip = 0.34 + 0.12 * s["weight"]
    disc = sdf.circle(x, y, r, 0.0, -0.10)
    spike = sdf.polygon(x, y, [(-r * 0.80, -0.02), (r * 0.80, -0.02), (0.0, tip)])
    return sdf.smooth_union(disc, spike, 0.05)


def _tongue(x, y, height: float, width: float):
    """The shared shape the flame is made of: a wide base tapering to a point."""
    base = sdf.circle(x, y, width, 0.0, -0.12)
    spike = sdf.polygon(x, y, [(-width, 0.0), (width, 0.0), (0.0, height)])
    return sdf.smooth_union(base, spike, 0.06)


def _flame(x, y, s: dict):
    """Fire: one narrow tongue, and -- when the seed asks -- a second off it."""
    body = _tongue(x, y, 0.42 + 0.10 * s["weight"], 0.155 + 0.030 * s["weight"])
    if s["detail"] > 0.3:
        small = sdf.moved(sdf.rotated(lambda a, b: _tongue(a, b, 0.24, 0.105), 26.0),
                          0.14, -0.09)(x, y)
        return sdf.smooth_union(body, small, 0.035)
    return body


def _blade(x, y, s: dict):
    """A sword seen flat: taper, guard, grip, pommel -- four pieces, one symbol."""
    w = 0.082 + 0.022 * s["weight"]
    tip = 0.44 + 0.06 * s["weight"]
    blade = sdf.polygon(x, y, [(-w, 0.06), (w, 0.06), (0.0, tip)])
    guard = sdf.box(x, y, 0.235, 0.036, 0.02, 0.0, 0.02)
    grip = sdf.segment(x, y, (0.0, -0.02), (0.0, -0.28), 0.042)
    pommel = sdf.circle(x, y, 0.070, 0.0, -0.33)
    return sdf.union(sdf.union(blade, guard), sdf.union(grip, pommel))


def _banner(x, y, s: dict):
    """A pennant on a pole: the flag, reduced to what survives at 32 px."""
    t = 0.028 + 0.016 * s["weight"]
    pole = sdf.segment(x, y, (-0.30, -0.44), (-0.30, 0.46), t)
    fly = 0.34 + 0.10 * s["weight"]
    pennant = sdf.polygon(x, y, [(-0.28, 0.38), (-0.28 + fly, 0.16), (-0.28, -0.06)])
    return sdf.union(pole, pennant)


def _shield(x, y, s: dict):
    """A heater shield, with a bar across it once the seed asks for detail."""
    scale = 0.92 + 0.16 * s["weight"]
    pts = [(-0.34, 0.42), (0.34, 0.42), (0.34, 0.04), (0.0, -0.44), (-0.34, 0.04)]
    body = sdf.polygon(x, y, [(px * scale, py * scale) for px, py in pts])
    if s["detail"] > 0.5:
        bar = sdf.box(x, y, 0.5, 0.045, 0.0, 0.0, -0.02)
        body = sdf.smooth_difference(body, bar, 0.02)
    return body


#: The vocabulary.  Each entry is the charge and the hue that *means* it: a player
#: reads a resource icon by colour before shape, so the two agreeing is what makes
#: a set feel designed rather than generated.  Hues are in OKLCH turns.
KINDS = {
    "coin": (_coin, 0.110),
    "grain": (_grain, 0.135),
    "wood": (_wood, 0.068),
    "ore": (_ore, 0.780),
    "water": (_water, 0.560),
    "flame": (_flame, 0.030),
    "blade": (_blade, 0.620),
    "banner": (_banner, 0.980),
    "shield": (_shield, 0.440),
}


# ----------------------------------------------------------------- the plate
#: The grounds an icon can be set on.  A *set* of icons shares one of these -- the
#: plate is the frame, the charge is the icon -- which is why the canonical spec
#: uses the same plate for every kind.
PLATES = ("round", "hex", "circle", "shield")


def _poly_points(pts, scale: float):
    return [(px * scale, py * scale) for px, py in pts]


def plate_field(plate: str, X, Y, spec: dict):
    """The plate's signed field, sized by ``margin`` and rounded by ``radius``."""
    m = float(spec["margin"])
    if plate not in PLATES:
        raise ValueError(f"plate must be one of {PLATES}, not {plate!r}")
    if plate == "round":
        return sdf.box(X, Y, m, m, float(spec["radius"]) * m * 2.0)
    if plate == "hex":
        return sdf.regular_polygon(X, Y, 6, m * 1.02, 0.0)
    if plate == "circle":
        return sdf.circle(X, Y, m * 1.03)
    pts = [(-0.78, 0.72), (0.78, 0.72), (0.78, 0.02), (0.0, -0.84), (-0.78, 0.02)]
    return sdf.polygon(X, Y, _poly_points(pts, m * 1.20))


# ------------------------------------------------------------------- a spec
def checked(name) -> str:
    """A charge's name, checked.  Raises for a name that is not one.

    A name that is wrong is an identifier error rather than a number out of range,
    so it is refused rather than repaired -- the same line ``gen.ui.material``
    draws, and for the same reason: silently drawing a coin where a shield was
    asked for is worse than stopping.
    """
    name = str(name)
    if name not in KINDS:
        raise ValueError(f"kind must be one of {tuple(KINDS)}, not {name!r}")
    return name


def kind(spec: dict) -> str:
    """The charge a spec asks for, checked."""
    return checked(spec.get("kind", "coin"))


def field(spec: dict, X, Y):
    """The charge's signed field: the symbol, leaning and stretched by the seed.

    The tilt and the stretch are applied here, once, in the charge's own frame, so
    a recipe never has to know that it is being drawn at an angle -- and a lean of
    a few degrees is enough to make two coins from two seeds look like two
    different coins without making either of them look like something else.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    t = np.radians(-float(spec.get("rotation", 0.0)))
    aspect = max(float(spec.get("aspect", 1.0)), 0.5)
    x = X * np.cos(t) - Y * np.sin(t)
    y = (X * np.sin(t) + Y * np.cos(t)) / aspect
    return KINDS[kind(spec)][0](x, y, spec)


def canonical(kind_name: str) -> dict:
    """One kind's plain spec: the icon a set is designed around.

    Everything that a seed is allowed to move sits at the middle of its range, and
    the plate is the round one for every kind, because a set of icons is one plate
    and many charges -- swapping the plate per icon is what makes a sheet look like
    a bag of assets instead of a set.
    """
    name = checked(kind_name)
    return {
        "kind": name,
        "plate": "round",
        "hue": float(KINDS[name][1]),
        "chroma": 0.115,
        "light": 0.745,
        "plate_light": 0.310,
        "plate_chroma": 0.032,
        "detail": 0.5,
        "weight": 0.5,
        "aspect": 1.0,
        "rotation": 0.0,
        "margin": 0.75,
        "bevel": 0.070,
        "radius": 0.075,
    }


def random_spec(seed: int, kind_name: str | None = None) -> dict:
    """A whole icon from a seed: which charge, how it leans, and what colour."""
    rng = np.random.default_rng(int(seed))
    names = tuple(KINDS)
    chosen = kind_name if kind_name is not None else names[int(rng.integers(0, len(names)))]
    s = canonical(chosen)
    s.update(
        plate=PLATES[int(rng.integers(0, len(PLATES)))],
        hue=(s["hue"] + float(rng.uniform(-0.022, 0.022))) % 1.0,
        chroma=float(rng.uniform(0.085, 0.150)),
        light=float(rng.uniform(0.700, 0.800)),
        plate_light=float(rng.uniform(0.255, 0.350)),
        plate_chroma=float(rng.uniform(0.018, 0.048)),
        detail=float(rng.uniform(0.0, 1.0)),
        weight=float(rng.uniform(0.3, 0.7)),
        aspect=float(rng.uniform(0.92, 1.08)),
        rotation=float(rng.uniform(-8.0, 8.0)),
        margin=float(rng.uniform(0.70, 0.80)),
        bevel=float(rng.uniform(0.055, 0.085)),
        radius=float(rng.uniform(0.05, 0.14)),
    )
    return s
