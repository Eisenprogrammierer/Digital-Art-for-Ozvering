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

__all__ = [
    "GEOMETRY",
    "COLOURS",
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
    "pad",
    "boss_reach",
    "boss_limit",
]

#: The corner ornament's shape constants, shared by the geometry, the painter and
#: the validator so the box the design promises to stay inside is the box it is
#: actually measured against.  All are fractions of the frame width.
BOSS_R0 = 0.40       # base radius of the scalloped disc
BOSS_O = 0.75        # distance from the panel corner to its centre, along the diagonal
BOSS_OUTLINE = 0.10  # hairline outline width, as a fraction of the radius
BOSS_MAX = 3.5       # largest multiple of BOSS_R0 * border worth drawing


def _c(L: float, C: float, hue: float) -> np.ndarray:
    """Short local alias: build one sRGB colour from OKLCH."""
    return palette.color(L, C, hue)


#: Numeric keys, in design pixels.  Kept apart from the colours so a kit can be
#: recoloured for a faction or a biome without touching its proportions.
GEOMETRY = (
    "inset", "radius", "border", "bevel", "rule", "rule_gap", "gradient",
    "alpha", "boss", "boss_harmonics", "boss_amp",
)

#: Colour keys, all sRGB triples.
COLOURS = ("body", "body_low", "edge", "ink", "accent")

#: The neutral default: a dark panel in a light frame, which is the shape most
#: strategy-game chrome ends up taking.
DEFAULTS = {
    "inset": 18.0,      # nine-slice cut from each edge; a floor, not a fixed size
    "radius": 9.0,      # outer corner radius; the curvature that must fit
    "border": 5.0,      # width of the frame band
    "bevel": 2.5,       # width of the rounded moulding on that band
    "rule": 1.2,        # inner rule line width; 0 disables it
    "rule_gap": 2.6,    # gap between the band and the rule
    "gradient": 0.55,   # how far the body falls from `body` to `body_low`
    "alpha": 1.0,       # body opacity, so a style can be glass
    "boss": 1.5,        # corner ornament size, as a multiple of 0.40 * border
    "boss_harmonics": 7,
    "boss_amp": 0.12,   # how scalloped the ornament's radius function is
    "body": _c(0.34, 0.030, 0.090),
    "body_low": _c(0.22, 0.035, 0.090),
    "edge": _c(0.56, 0.055, 0.095),
    "ink": _c(0.15, 0.030, 0.090),
    "accent": _c(0.74, 0.110, 0.110),
}


def _preset(**over) -> dict:
    return {**DEFAULTS, **over}


#: Named kits.  Each is a whole look: colour scheme plus the proportions that go
#: with it -- a parchment window has a wider, softer frame than a brass one.
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
}


def names() -> list:
    """Style names, in a stable order."""
    return sorted(STYLES)


def get(name: str) -> dict:
    """A named style, clamped; raises ``KeyError`` for an unknown name."""
    if name not in STYLES:
        raise KeyError(f"unknown ui style {name!r}; try one of {names()}")
    return clamp(STYLES[name])


def _clean(spec: dict) -> dict:
    """Coerce the numeric keys to floats and the colours to sRGB arrays."""
    s = {**DEFAULTS, **dict(spec)}
    for k in GEOMETRY:
        s[k] = float(s[k])
    for k in COLOURS:
        s[k] = np.asarray(s[k], dtype=np.float64)[:3].copy()
    return s


def clamp(spec: dict, size=None) -> dict:
    """Force a spec onto the legal set of designs.

    ``inset`` is an *output* of this function, not really an input: the style
    states a floor, and the derived corner box is widened to whatever the look
    actually needs -- the radius plus its bevel, or the ornament's reach.  So the
    nine-slice cut can never be too small to hold the curvature, by construction
    rather than by check.  ``size`` runs the argument the other way: at a size too
    small for the corner box the *look* gives way instead, keeping the radius --
    the shape -- and flattening the bevel -- the shading -- first.
    """
    s = _clean(spec)

    request = max(float(s["inset"]), 4.0)
    if size is not None:
        w, h = _size(size)
        # Two corner boxes and at least one pixel of stretchable middle.
        request = min(request, max(4.0, 0.5 * min(w, h) - 0.5))
    border = float(np.clip(s["border"], 0.5, request))
    radius = float(np.clip(s["radius"], 0.0, max(request - 0.5, 0.0)))
    bevel = float(np.clip(s["bevel"], 0.0, max(request - radius, 0.0)))
    rule = float(np.clip(s["rule"], 0.0, request))
    rule_gap = float(np.clip(s["rule_gap"], 0.0, request))
    amp = float(np.clip(s["boss_amp"], 0.0, 0.30))
    # The ornament yields to the frame, not the other way round: a thinner frame
    # would change what the whole kit looks like, a smaller rivet would not.
    boss = float(np.clip(s["boss"], 0.0, min(BOSS_MAX, boss_limit(request, border, amp))))

    s["inset"] = max(request, radius + bevel)
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
    return s


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
    nine-slice resample to be faithful, or for a shape to exist at all.
    """
    s = _clean(spec)
    bad = []
    if s["inset"] < 4.0:
        bad.append("inset: a corner box narrower than 4 px cannot hold anything")
    if s["border"] > s["inset"] + 1e-9:
        bad.append("border: the inner rectangle's corner would leave the corner box")
    if s["bevel"] > s["border"] + 1e-9:
        bad.append("bevel: the moulding is wider than the band it moulds")
    if s["radius"] > s["inset"] - s["bevel"] + 1e-9:
        bad.append("radius: the corner curvature leaks out of the corner box")
    if s["radius"] < 0.0 or s["border"] <= 0.0 or s["bevel"] < 0.0 or s["rule"] < 0.0:
        bad.append("negative thickness")
    if not (0.0 <= s["gradient"] <= 1.0) or not (0.05 <= s["alpha"] <= 1.0):
        bad.append("range: gradient or alpha out of bounds")
    if not (0.0 <= s["boss_amp"] <= 0.30):
        bad.append("boss_amp: outside the range the radius function is bounded on")
    if s["boss"] > BOSS_MAX + 1e-9:
        bad.append("boss: larger than the design allows")
    elif boss_reach(s["inset"], s["border"], s["boss"], s["boss_amp"]) > s["inset"] + 1e-9:
        bad.append("boss: the corner ornament leaves its corner box")
    if size is not None:
        w, h = _size(size)
        if 2.0 * s["inset"] > min(w, h) + 1e-9:
            bad.append("inset: the two corner boxes would meet in the middle")
        if 2.0 * (s["border"] + s["rule_gap"] + s["rule"]) >= min(w, h):
            bad.append("rule: no body left inside the frame at this size")
    return bad


def pad(spec: dict) -> float:
    """How far the frame eats into the panel: the content inset, in design px."""
    s = _clean(spec)
    return float(s["border"] + s["rule_gap"] + s["rule"])


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
    # Order the jitter so the clamp never has to repair it: bevel <= border <=
    # inset, and radius inside the corner box with the bevel accounted for.
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
    return clamp(s)


def _size(size) -> tuple:
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    return int(size), int(size)
