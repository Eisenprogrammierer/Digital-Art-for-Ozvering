"""Panel styles: one set of numbers that makes a whole UI kit look like one thing.

A UI panel is the first asset in this project that has to survive being resized
to a size nobody chose.  An icon is rendered at the size it is asked for; a panel
is drawn once at a comfortable size and then stretched to whatever the layout
needs, which is what the nine-slice convention is for.  That inverts the usual
question: the generated PNG is not the asset -- ``(image, corner insets)`` is --
and the numbers below are what make the two agree.

Everything is in **design pixels**: the size the asset was drawn for, at scale 1.
Rendering at ``scale=2`` doubles every pixel and changes no number, so a style is
resolution-independent by construction, and the *pixel* corner insets of the
nine-slice are just ``inset * scale``.

The one hard geometric rule the whole package obeys::

    every bit of curvature in the design must fit inside the corner box.

A nine-slice resample keeps the corners and stretches the four edges and the
middle.  A straight edge survives that; a curve does not -- it would smear.  So
the corner radius, the bevel and the corner ornament all have to live within
``inset`` pixels of the corner, and the design has to be constant along the axis
it gets stretched on.  :func:`validate` states that, :func:`clamp` enforces it,
and ``tests/test_ui.py`` checks the resulting pixels directly.
"""
from __future__ import annotations

import numpy as np

from core import palette

from . import material

__all__ = [
    "GEOMETRY",
    "COLOURS",
    "SIDED",
    "MATERIAL",
    "DEFAULTS",
    "STYLES",
    "BOSS_R0",
    "BOSS_O",
    "BOSS_OUTLINE",
    "names",
    "get",
    "clamp",
    "validate",
    "random_style",
    "title_bar",
    "pad",
    "boss_reach",
    "boss_limit",
]

BOSS_R0 = 0.40
BOSS_O = 0.75
BOSS_OUTLINE = 0.10
BOSS_MAX = 3.5


def _c(L: float, C: float, hue: float) -> np.ndarray:
    """Short local alias: build one sRGB colour from OKLCH."""
    return palette.color(L, C, hue)

GEOMETRY = (
    "inset", "radius", "border", "bevel", "rule", "rule_gap", "gradient",
    "alpha", "boss", "boss_harmonics", "boss_amp",
)

SIDED = ("inset", "border")

MATERIAL = ("material", "material_scale", "material_seed", "material_relief",
            "material_tone")

COLOURS = ("body", "body_low", "edge", "ink", "accent")

DEFAULTS = {
    "inset": 18.0,
    "radius": 9.0,
    "border": 5.0,
    "bevel": 2.5,
    "rule": 1.2,
    "rule_gap": 2.6,
    "gradient": 0.55,
    "alpha": 1.0,
    "boss": 1.5,
    "boss_harmonics": 7,
    "boss_amp": 0.12,
    "body": _c(0.34, 0.030, 0.090),
    "body_low": _c(0.22, 0.035, 0.090),
    "edge": _c(0.56, 0.055, 0.095),
    "ink": _c(0.15, 0.030, 0.090),
    "accent": _c(0.74, 0.110, 0.110),
    **material.DEFAULTS,
}


def _preset(**over) -> dict:
    return {**DEFAULTS, **over}

STYLES = {
    "parchment": _preset(
        body=_c(0.80, 0.045, 0.130), body_low=_c(0.70, 0.060, 0.120),
        edge=_c(0.52, 0.070, 0.110), ink=_c(0.30, 0.060, 0.090),
        accent=_c(0.76, 0.125, 0.105),
        radius=10.0, border=6.0, bevel=3.0, gradient=0.45,
    ),
    "slate": _preset(
        body=_c(0.40, 0.030, 0.610), body_low=_c(0.29, 0.035, 0.620),
        edge=_c(0.63, 0.035, 0.615), ink=_c(0.19, 0.045, 0.620),
        accent=_c(0.74, 0.100, 0.550),
        radius=7.0, border=4.0, bevel=2.0, gradient=0.70,
    ),
    "brass": _preset(
        body=_c(0.27, 0.040, 0.100), body_low=_c(0.18, 0.045, 0.095),
        edge=_c(0.67, 0.100, 0.115), ink=_c(0.13, 0.030, 0.090),
        accent=_c(0.82, 0.130, 0.125),
        radius=6.0, border=5.0, bevel=2.5, gradient=0.75,
        boss_amp=0.16, boss_harmonics=6,
    ),
    "moss": _preset(
        body=_c(0.35, 0.045, 0.420), body_low=_c(0.25, 0.050, 0.430),
        edge=_c(0.58, 0.070, 0.400), ink=_c(0.17, 0.035, 0.420),
        accent=_c(0.72, 0.110, 0.330),
        radius=12.0, border=5.0, bevel=2.5, gradient=0.60,
    ),
    "glass": _preset(
        body=_c(0.56, 0.030, 0.580), body_low=_c(0.42, 0.035, 0.585),
        edge=_c(0.79, 0.025, 0.580), ink=_c(0.26, 0.040, 0.590),
        accent=_c(0.72, 0.090, 0.530),
        radius=9.0, border=3.0, bevel=1.6, gradient=0.80, alpha=0.42,
        boss=0.0,
    ),

    "timber": _preset(
        body=_c(0.46, 0.055, 0.075), body_low=_c(0.30, 0.060, 0.070),
        edge=_c(0.64, 0.075, 0.080), ink=_c(0.20, 0.045, 0.075),
        accent=_c(0.72, 0.110, 0.115),
        radius=6.0, border=6.0, bevel=3.0, gradient=0.30,
        material="wood", material_scale=26.0, material_seed=11,
        material_relief=0.55, material_tone=0.30,
    ),
    "granite": _preset(
        body=_c(0.32, 0.012, 0.250), body_low=_c(0.21, 0.014, 0.255),
        edge=_c(0.60, 0.018, 0.255), ink=_c(0.15, 0.020, 0.250),
        accent=_c(0.70, 0.055, 0.230),
        radius=5.0, border=6.0, bevel=3.0, gradient=0.25,
        material="stone", material_scale=20.0, material_seed=23,
        material_relief=0.60, material_tone=0.22, boss=1.2,
    ),
}


def names() -> list:
    """Style names, in a stable order."""
    return sorted(STYLES)


def get(name: str) -> dict:
    """A named style, clamped; raises ``KeyError`` for an unknown name."""
    if name not in STYLES:
        raise KeyError(f"unknown ui style {name!r}; try one of {names()}")
    return clamp(STYLES[name])


def _four(value) -> tuple:
    """One number, or four, as ``(left, top, right, bottom)``."""
    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
        v = float(value)
        return (v, v, v, v)
    v = tuple(float(x) for x in value)
    if len(v) != 4:
        raise ValueError(f"expected one number or four (left, top, right, bottom), got {value!r}")
    return v


def _clean(spec: dict) -> dict:
    """Coerce the numeric keys to floats and the colours to sRGB arrays."""
    s = {**DEFAULTS, **dict(spec)}
    for k in GEOMETRY:
        s[k] = _four(s[k]) if k in SIDED else float(s[k])
    for k in MATERIAL:
        s[k] = str(s[k]) if k == "material" else float(s[k])
    for k in COLOURS:
        s[k] = np.asarray(s[k], dtype=np.float64)[:3].copy()
    return s


def clamp(spec: dict, size=None) -> dict:
    """Force a spec onto the legal set of designs.

    ``inset`` is an *output* of this function, not really an input: the style
    states a floor, and the derived corner box is widened to whatever the look
    actually needs -- the frame and its rule, the radius plus its bevel, or the
    ornament's reach.  So the nine-slice cut can never be too small to hold the
    design, by construction rather than by check.  ``size`` runs the argument the
    other way: at a size too small for the corner box the *look* gives way instead,
    keeping the radius -- the shape -- and flattening the bevel -- the shading --
    first, and thinning the rule rather than letting it spill out of the box.

    Everything here is per side, because a title bar is nothing but a wider top
    band.  What is *not* per side is the curvature: there is one corner radius, one
    bevel and one ornament size for the whole kit, so the number that bounds them
    is the smallest of the four sides, and a bar that is thin on one axis limits
    the ornament on all four corners.
    """
    s = _clean(spec)

    floor = tuple(max(v, 4.0) for v in s["inset"])
    cap = None
    if size is not None:
        w, h = _size(size)

        lim_x = max(4.0, 0.5 * (w - 1.0))
        lim_y = max(4.0, 0.5 * (h - 1.0))
        cap = (lim_x, lim_y, lim_x, lim_y)
    limit = cap if cap is not None else floor
    border = tuple(float(np.clip(v, 0.5, lim)) for v, lim in zip(s["border"], limit))
    room = min(min(f, lim) for f, lim in zip(floor, limit))
    radius = float(np.clip(s["radius"], 0.0, max(room - 0.5, 0.0)))
    bevel = float(np.clip(s["bevel"], 0.0, max(room - radius, 0.0)))
    rule = float(np.clip(s["rule"], 0.0, room))
    rule_gap = float(np.clip(s["rule_gap"], 0.0, room))

    avail = min(lim - b for lim, b in zip(limit, border))
    span = rule_gap + rule
    if span > avail + 1e-9 and span > 1e-9:
        k = max(avail, 0.0) / span
        rule_gap *= k
        rule *= k
    request = tuple(max(min(f, lim), b + rule_gap + rule)
                    for f, lim, b in zip(floor, limit, border))
    amp = float(np.clip(s["boss_amp"], 0.0, 0.30))

    cornices = tuple(_corner(i, request, border) for i in range(4))
    reach = min(boss_limit(ixy, bxy, amp) for ixy, bxy in cornices)
    boss = float(np.clip(s["boss"], 0.0, min(BOSS_MAX, reach)))

    s["inset"] = tuple(max(req, radius + bevel) for req in request)
    s["border"] = border
    s["bevel"] = bevel
    s["radius"] = radius
    s["rule"] = rule
    s["rule_gap"] = rule_gap
    s["gradient"] = float(np.clip(s["gradient"], 0.0, 1.0))
    s["alpha"] = float(np.clip(s["alpha"], 0.05, 1.0))
    s["boss"] = boss
    s["boss_harmonics"] = int(np.clip(round(s["boss_harmonics"]), 3, 12))
    s["boss_amp"] = amp

    s["material"] = str(s["material"])
    s["material_scale"] = float(np.clip(s["material_scale"], 1.0, 512.0))
    s["material_seed"] = int(np.clip(round(s["material_seed"]), 0, 2 ** 31 - 1))
    s["material_relief"] = float(np.clip(s["material_relief"], 0.0, 2.0))
    s["material_tone"] = float(np.clip(s["material_tone"], 0.0, 1.0))
    return s


def _corner(i: int, inset, border) -> tuple:
    """Corner ``i``'s box and band, as the two numbers the ornament is bounded by.

    The box is an ``(x, y)`` pair now that the sides can differ, so the honest
    measure of "how much room is there" is the *smaller* of the two, taken with the
    smaller of the two band widths -- the ornament is a disc, and a disc has one
    radius.
    """
    ix, iy = _corner_pair(i, inset)
    bx, by = _corner_pair(i, border)
    return (min(ix, iy), min(bx, by))


def _corner_pair(i: int, sides) -> tuple:
    """``(horizontal, vertical)`` member of ``sides`` for corner ``i``."""
    left, top, right, bottom = sides
    return (left, top) if i == 0 else (right, top) if i == 1 else (left, bottom) if i == 2 else (right, bottom)


def boss_reach(inset: float, border: float, boss: float, amp: float) -> float:
    """How far the corner ornament reaches from the panel corner, along an axis.

    The farthest point of a disc is not on the diagonal, so this adds the
    centre's axis offset to the disc's outer radius rather than taking the norm
    of the two -- which is what makes the bound tight enough to be useful.
    """
    if boss <= 0.0:
        return 0.0
    r0 = BOSS_R0 * border * boss
    return BOSS_O * border / np.sqrt(2.0) + r0 * (1.0 + amp + BOSS_OUTLINE)


def boss_limit(inset: float, border: float, amp: float) -> float:
    """The largest ``boss`` whose disc still fits in the corner box.

    :func:`boss_reach` is linear in ``boss``, so the limit is one division.  It
    can come out negative, which means the corner box is too small for any
    ornament at all -- a legal answer, and the one ``clamp`` records as ``0``.
    """
    if border <= 0.0:
        return 0.0
    head = inset - BOSS_O * border / np.sqrt(2.0)
    step = BOSS_R0 * border * (1.0 + amp + BOSS_OUTLINE)
    return float(head / step) if step > 1e-9 else 0.0


def validate(spec: dict, size=None) -> list:
    """Return the design invariants ``spec`` breaks; an empty list means legal.

    These are not style opinions -- every one of them is a precondition for the
    nine-slice resample to be faithful, or for a shape to exist at all.  With the
    sides free to differ, each of those preconditions is per side, so the checks
    that used to compare one number with itself now walk the four: a band may be
    wide on the top and thin on the left, and only the left is then wrong.
    """
    s = _clean(spec)
    bad = []
    ins, bor = s["inset"], s["border"]
    if min(ins) < 4.0:
        bad.append("inset: a corner box narrower than 4 px cannot hold anything")
    for side, i, b in zip(("left", "top", "right", "bottom"), ins, bor):
        if b > i + 1e-9:
            bad.append(f"border ({side}): the inner rectangle's corner would leave the corner box")
        if b + s["rule_gap"] + s["rule"] > i + 1e-9:
            bad.append(f"rule ({side}): the inner moulding would leave the corner box")
    if s["bevel"] > min(bor) + 1e-9:
        bad.append("bevel: the moulding is wider than the narrowest band it moulds")
    if s["radius"] > min(ins) - s["bevel"] + 1e-9:
        bad.append("radius: the corner curvature leaks out of the corner box")
    if s["radius"] < 0.0 or min(bor) <= 0.0 or s["bevel"] < 0.0 or s["rule"] < 0.0:
        bad.append("negative thickness")
    if not (0.0 <= s["gradient"] <= 1.0) or not (0.05 <= s["alpha"] <= 1.0):
        bad.append("range: gradient or alpha out of bounds")
    if not (0.0 <= s["boss_amp"] <= 0.30):
        bad.append("boss_amp: outside the range the radius function is bounded on")
    if s["boss"] > BOSS_MAX + 1e-9:
        bad.append("boss: larger than the design allows")
    else:
        for i in range(4):
            room, band = _corner(i, ins, bor)
            if boss_reach(room, band, s["boss"], s["boss_amp"]) > room + 1e-9:
                bad.append(f"boss: the corner ornament leaves its corner box (corner {i})")
                break
    if size is not None:
        w, h = _size(size)
        if ins[0] + ins[2] > w + 1e-9 or ins[1] + ins[3] > h + 1e-9:
            bad.append("inset: the corner boxes would meet in the middle")
        if 2.0 * (min(bor) + s["rule_gap"] + s["rule"]) >= min(w, h):
            bad.append("rule: no body left inside the frame at this size")
    if s["material"] not in material.KINDS:
        bad.append(f"material: {s['material']!r} is not one of {material.KINDS}")
    elif s["material"] != "none":
        if s["material_scale"] < 1.0:
            bad.append("material_scale: a feature smaller than a design pixel cannot be drawn")
        if not (0.0 <= s["material_relief"] <= 2.0):
            bad.append("material_relief: outside the range the shading is bounded on")
        if not (0.0 <= s["material_tone"] <= 1.0):
            bad.append("material_tone: a stain is a mix, so it cannot exceed 1")
    return bad


def title_bar(spec: dict, height: float, edge: str = "top") -> dict:
    """The same design with a title bar ``height`` design pixels deep.

    Both numbers have to move together -- the band has to be wide enough for the
    bar, and the corner box deep enough for the band -- so this is one call rather
    than two, and the bar is a **border**, not a second panel.  That is the whole
    point: the frame's moulding then runs around the title, which is what makes it
    look carved out of the window instead of stuck onto it.

    The floor is raised only as far as the bar, never lowered: a style that already
    reserves more room for its corners keeps it.
    """
    if edge not in ("top", "bottom"):
        raise ValueError(f"a title bar goes top or bottom, not {edge!r}")
    i = 1 if edge == "top" else 3
    s = _clean(spec)
    bar = max(float(height), float(min(s["border"])))
    border = list(s["border"])
    border[i] = bar
    inset = list(s["inset"])
    inset[i] = max(inset[i], bar)
    return clamp({**dict(spec), "border": tuple(border), "inset": tuple(inset)})


def pad(spec: dict) -> tuple:
    """How far the frame eats into the panel, per side, in design px.

    The inner edge of the frame plus the rule: the rectangle a layout engine may
    put content in.  Per side, because a titled window's content starts lower than
    it starts to the right.
    """
    s = _clean(spec)
    return tuple(b + s["rule_gap"] + s["rule"] for b in s["border"])


def random_style(seed: int, name: str | None = None) -> dict:
    """Draw a coherent kit from a seed: one hue family, sane proportions.

    The geometry is jittered *inside* the legal set rather than drawn blindly and
    clamped afterwards, so a seed never lands on a design that has quietly been
    repaired into something bland.
    """
    rng = np.random.default_rng(int(seed))
    base = dict(STYLES[name] if name else STYLES[names()[int(rng.integers(0, len(STYLES)))]])
    s = _clean(base)

    hue = float(rng.random())
    chroma = float(rng.uniform(0.02, 0.09))
    dark = bool(rng.random() < 0.5)
    body_L = float(rng.uniform(0.52, 0.80) if not dark else rng.uniform(0.24, 0.42))
    swing = float(rng.uniform(0.08, 0.20))
    accent_hue = hue + float(rng.choice([0.0, 0.5, 0.083, -0.083]))

    inset = float(rng.uniform(13.0, 22.0))
    border = float(rng.uniform(3.5, 6.5))
    radius = float(rng.uniform(4.0, 11.0))

    bevel = float(rng.uniform(0.35, 0.55)) * border
    radius = min(radius, inset - bevel)

    s.update(
        body=palette.color(body_L, chroma, hue),
        body_low=palette.color(body_L - swing, chroma * 1.15, hue),
        edge=palette.color(min(body_L + float(rng.uniform(0.14, 0.26)), 0.88),
                           chroma * 1.30, hue),
        ink=palette.color(max(body_L - float(rng.uniform(0.12, 0.22)), 0.08),
                          chroma * 0.85, hue),
        accent=palette.color(min(body_L + 0.36, 0.84), min(chroma * 2.1, 0.15), accent_hue),
        inset=inset, border=border, bevel=bevel, radius=radius,
        rule=float(rng.uniform(0.8, 1.6)),
        rule_gap=float(rng.uniform(1.8, 3.4)),
        gradient=float(rng.uniform(0.35, 0.85)),
        alpha=1.0 if rng.random() < 0.85 else float(rng.uniform(0.45, 0.8)),
        boss=0.0 if rng.random() < 0.2 else float(rng.uniform(1.4, 3.2)),
        boss_amp=float(rng.uniform(0.06, 0.20)),
        boss_harmonics=int(rng.integers(5, 10)),
    )

    wants_material = rng.random() < 0.3
    if s["material"] == "none" and wants_material:
        s["material"] = ("wood", "stone")[int(rng.integers(0, 2))]
        s["material_scale"] = float(rng.uniform(16.0, 30.0))
        s["material_seed"] = int(rng.integers(0, 2 ** 30))
        s["material_relief"] = float(rng.uniform(0.35, 0.75))
        s["material_tone"] = float(rng.uniform(0.15, 0.40))
    s = clamp(s)

    if rng.random() < 0.2:
        edge = "top" if rng.random() < 0.75 else "bottom"
        s = title_bar(s, float(rng.uniform(1.8, 3.2)) * border, edge=edge)
    return s


def _size(size) -> tuple:
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    return int(size), int(size)
