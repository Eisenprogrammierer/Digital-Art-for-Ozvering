"""Field wear: scratches, gouges, chips, cracks and stains over an icon.

An icon that represents a unit should be able to show how much of that unit is
left.  The whole point of a *stage* is that it is cumulative: stage 4 contains
everything stage 3 had and one more mark, so a row of ten is a history rather
than ten unrelated dirt textures.  That property is built into the geometry --
:func:`field` is the union of the marks whose birth stage has arrived -- instead
of being an accident of how the alpha happens to be blended.

The marks are drawn in the icon's own world frame, so a 32 px copy and a 256 px
copy are the same damage.  They are clipped to the plate, because a scratch in
the transparent margin is not a scratch on the object.  The first stages are
scuffs and a chip; the middle adds dark gouges; the last stages crack the plate
from the rim inwards.  None of them moves the charge: the symbol is still the
symbol at stage 10, which is the same line every other part of the icon set holds.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from core import palette, render, sdf

from . import motifs

__all__ = ["MAX", "LEVELS", "clamp_level", "stage", "stages", "field",
           "chip_field", "paint"]

#: 1..10 are the ten wear stages; 0 is the undamaged icon.
MAX = 10
LEVELS = tuple(range(1, MAX + 1))

#: Spec keys that determine the damage.  The badge is deliberately not in this
#: list: changing a count on the corner must not redraw the scratches on the
#: plate, because the plate is the same plate.
_KEYS = ("kind", "plate", "hue", "chroma", "light", "plate_light",
         "plate_chroma", "detail", "weight", "aspect", "rotation", "margin",
         "bevel", "radius")


def clamp_level(value) -> int:
    """Clamp a requested stage to one of the ten integer levels."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(MAX, n))


def stage(spec: dict, level: int) -> dict:
    """The same icon at one wear stage, as a new spec."""
    return {**dict(spec), "wear": clamp_level(level)}


def stages(spec: dict) -> list:
    """The ten damaged specs, pristine stage excluded."""
    return [stage(spec, level) for level in LEVELS]


def _spec_key(spec: dict) -> tuple:
    return tuple((key, spec.get(key)) for key in _KEYS)


def _seed(spec: dict) -> int:
    raw = repr(_spec_key(spec)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _walk(rng, start, direction, steps, step, jitter) -> list:
    """A short jagged path, rooted at ``start`` and ending inside the plate."""
    pts = [tuple(start)]
    x, y = start
    for i in range(steps):
        direction += float(rng.uniform(-jitter, jitter))
        length = float(step * rng.uniform(0.72, 1.18))
        x += np.cos(direction) * length
        y += np.sin(direction) * length
        pts.append((float(x), float(y)))
    return pts


@lru_cache(maxsize=512)
def _plan(key: tuple) -> dict:
    spec = dict(key)
    rng = np.random.default_rng(_seed(spec))
    m = float(spec.get("margin", 0.75))
    # Damage starts on the plate, not in its transparent margin.  The round
    # plate is the square of half-extent m; the other plates are within that
    # bound, and the painter clips to the real field anyway.
    reach = m * 0.94
    scuffs, gouges, cracks, chips, stains = [], [], [], [], []

    # The first cut appears at stage 1.  A scuff is a groove plus the lit lip
    # beside it; a line alone reads as a decoration painted on the plate.
    for i in range(7):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius = reach * float(rng.uniform(0.18, 0.86))
        start = (np.cos(angle) * radius, np.sin(angle) * radius)
        heading = angle + float(rng.uniform(-1.0, 1.0)) + np.pi * 0.5
        length = float(rng.uniform(0.10, 0.27) * m / 0.75)
        path = _walk(rng, start, heading, 2, length * 0.5, 0.55)
        width = float(rng.uniform(0.018, 0.030))
        scuffs.append({"birth": i + 1, "path": path, "width": width, "kind": "scuff"})

    # A gouge is shorter, deeper and wider: the kind of wound that takes a piece
    # out of the paint rather than a hairline scratch across it.
    for i in range(4):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius = reach * float(rng.uniform(0.35, 0.90))
        start = (np.cos(angle) * radius, np.sin(angle) * radius)
        heading = angle + float(rng.uniform(-0.8, 0.8)) + np.pi * 0.5
        length = float(rng.uniform(0.08, 0.16) * m / 0.75)
        path = _walk(rng, start, heading, 2, length * 0.5, 0.35)
        width = float(rng.uniform(0.035, 0.055))
        gouges.append({"birth": 2 + 2 * i, "path": path, "width": width,
                       "kind": "gouge"})

    # Cracks arrive later and walk in from the rim.  Their first segment is
    # longest, so the eye reads a break starting at the edge.
    for i in range(6):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        start = (np.cos(angle) * reach, np.sin(angle) * reach)
        heading = angle + np.pi + float(rng.uniform(-0.35, 0.35))
        path = _walk(rng, start, heading, 5, reach * float(rng.uniform(0.16, 0.25)), 0.50)
        cracks.append({"birth": 4 + i, "path": path, "width": 0.022,
                       "kind": "crack"})
        if i >= 2:
            # One branch per later crack, leaving the main path partway down.
            root = path[2]
            branch_angle = heading + float(rng.choice((-1.0, 1.0))) * float(rng.uniform(0.45, 1.0))
            cracks.append({"birth": 5 + i,
                           "path": _walk(rng, root, branch_angle, 3,
                                         reach * 0.11, 0.65),
                           "width": 0.015, "kind": "crack"})

    # Chips are alpha erosion, not paint.  They take bites out of the plate rim.
    for i in range(6):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius = reach * float(rng.uniform(0.94, 1.02))
        chips.append({"birth": 3 + (i * 7) % 8,
                      "x": float(np.cos(angle) * radius),
                      "y": float(np.sin(angle) * radius),
                      "r": float(rng.uniform(0.055, 0.095))})

    # Stains are broad and soft; they suggest rust and old blood without drawing
    # a second symbol on the plate.
    for i in range(3):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius = reach * float(rng.uniform(0.20, 0.70))
        x, y = np.cos(angle) * radius, np.sin(angle) * radius
        blobs = []
        for _ in range(3):
            blobs.append((x + float(rng.uniform(-0.05, 0.05)),
                          y + float(rng.uniform(-0.05, 0.05)),
                          float(rng.uniform(0.035, 0.070))))
        stains.append({"birth": 3 + 3 * i, "blobs": blobs})

    return {"scuffs": scuffs, "gouges": gouges, "cracks": cracks,
            "chips": chips, "stains": stains}


def _plan_for(spec: dict) -> dict:
    return _plan(_spec_key(dict(spec)))


def _marks(plan: dict, level: int):
    """All non-chip marks active at ``level``, oldest first."""
    active = []
    for group in ("scuffs", "gouges", "cracks"):
        active.extend(item for item in plan[group] if item["birth"] <= level)
    return active


def field(spec: dict, level: int, X, Y):
    """Signed distance to the union of every mark active at ``level``.

    Negative inside a mark.  Because a later stage only adds marks, this field
    only ever decreases: ``field(s, n + 1) <= field(s, n)`` everywhere.  That is
    the cumulative contract, and :mod:`tests.test_icons` checks it directly.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = clamp_level(level)
    out = np.full(np.shape(X), np.inf, dtype=np.float64)
    if n <= 0:
        return out
    plan = _plan_for(spec)
    for item in _marks(plan, n):
        points = item["path"]
        for a, b in zip(points[:-1], points[1:]):
            out = np.minimum(out, sdf.segment(X, Y, a, b, item["width"]))
    for item in plan["stains"]:
        if item["birth"] <= n:
            for x, y, r in item["blobs"]:
                out = np.minimum(out, sdf.circle(X, Y, r, x, y))
    return out


def chip_field(spec: dict, level: int, X, Y):
    """Signed distance to the chips active at ``level`` (for alpha erosion)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = clamp_level(level)
    out = np.full(np.shape(X), np.inf, dtype=np.float64)
    if n <= 0:
        return out
    for item in _plan_for(spec)["chips"]:
        if item["birth"] <= n:
            out = np.minimum(out, sdf.circle(X, Y, item["r"], item["x"], item["y"]))
    return out


def paint(canvas, spec: dict, level: int, X, Y, dx: float, t: dict) -> None:
    """Paint the damage for ``level`` over a finished charge, before its badge.

    The marks are clipped to the plate, so a chip on the rim is a chip in the
    object and not a floating crumb.  Deep marks get a dark groove and a very
    narrow lit lip; stains are a low-alpha rusty wash.  The badge is deliberately
    drawn after this pass, because a count is information and should stay legible
    however badly the unit has been mauled.
    """
    n = clamp_level(level)
    if n <= 0:
        return
    d_plate = motifs.plate_field(str(spec.get("plate", "round")), X, Y, spec)
    plate = render.coverage(d_plate, dx)
    if float(plate.max()) <= 0.0:
        return

    # 1. chips -- remove alpha at the rim.  This is the only pass that changes the
    #    silhouette, and it is why a high-stage icon looks worn even in a contact
    #    sheet where the tiny scratches have blurred away.
    chips = render.coverage(chip_field(spec, n, X, Y), dx * 0.85) * plate
    if float(chips.max()) > 0.0:
        canvas.a *= 1.0 - np.clip(chips * 0.86, 0.0, 0.86)

    # 2. stains -- broad, soft, and under the hard marks so they do not tint the
    #    bright lips.
    d_stain = np.full(np.shape(X), np.inf, dtype=np.float64)
    for item in _plan_for(spec)["stains"]:
        if item["birth"] <= n:
            for x, y, r in item["blobs"]:
                d_stain = np.minimum(d_stain, sdf.circle(X, Y, r, x, y))
    stain = render.coverage(d_stain, dx) * plate
    if float(stain.max()) > 0.0:
        rust = palette.mix(t["recess"], (0.38, 0.14, 0.055), 0.58)
        canvas.over_srgb(rust, stain * 0.24)

    # 3. the grooves -- dark, then a narrow lit centre line.  A dark scratch on a
    #    dark recess disappears, and a bright scratch on a bright charge does the
    #    same; the pair is what makes the mark legible on both materials.  Both
    #    fields are grown by half a pixel so the smallest stages still survive a
    #    32 px render instead of becoming an invisible subpixel.
    plan = _plan_for(spec)
    marks = _marks(plan, n)
    dmg = render.coverage(field(spec, n, X, Y) - dx * 0.55, dx) * plate
    if float(dmg.max()) > 0.0:
        groove = palette.mix(t["recess"], (0.055, 0.042, 0.038), 0.62)
        canvas.over_srgb(groove, dmg * (0.52 + 0.030 * n))

    lip = np.full(np.shape(X), np.inf, dtype=np.float64)
    for item in marks:
        points = item["path"]
        width = item["width"] * 0.34
        for a, b in zip(points[:-1], points[1:]):
            lip = np.minimum(lip, sdf.segment(X, Y, a, b, width))
    lip = render.coverage(lip - dx * 0.40, dx) * plate
    if float(lip.max()) > 0.0:
        chip_lit = palette.mix(t["rim"], (0.95, 0.92, 0.84), 0.55)
        canvas.over_srgb(chip_lit, lip * (0.42 + 0.022 * n))
