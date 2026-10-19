"""Painting an icon: five passes over one light, plus a badge when there is one.

    shadow   the plate's own coverage, blurred and dropped
    plate    the ground, shaded as a rounded solid
    rim      a bright stroke along the plate's edge
    recess   the ground knocked back inside, *around* the charge
    charge   the symbol itself, embossed, in the accent colour
    badge    a count or a tier, on the corner -- see :mod:`gen.icons.badge`

The first five are the same five ``gen.ui`` uses, because an icon and a button are
both a plate with something raised on it.  The sixth is conditional and lands on
top, since a badge is the one part of an icon that is allowed to break its
silhouette.

The construction is why a set of these reads as a set: the plate is one shape with
one light, one bevel and one hue family, and everything that differs between two
icons is the charge and the hue.  An icon is designed in the unit square and
rendered at whatever size is asked for, so that the 32 px and the 256 px copies of
one spec are the *same drawing* and the smaller one is not a downsampled version
of the larger -- which is the whole reason a `--sizes` sweep is worth generating
instead of resizing.

Readability is treated as a measurement rather than a hope.  :func:`silhouette` and
:func:`contrast` are the two numbers `tests/test_icons.py` holds every kind to,
because the failure mode that matters for an icon is not an ugly one, it is one
that has quietly become a blob.
"""
from __future__ import annotations

import numpy as np

from core import palette, render, sdf
from core.canvas import Canvas
from core.render import LIGHT

from . import badge as _badge
from . import motifs

__all__ = ["LIGHT", "SHADOW", "MARGIN", "draw", "icon", "tints", "clamp", "validate",
           "silhouette", "contrast"]

#: What an icon throws on the ground.
SHADOW = (0.03, 0.03, 0.05)

#: The canvas is a little larger than the plate, so an icon never touches the edge
#: of its own tile and the drop shadow has somewhere to fall.  In world units, like
#: everything else, so a re-render at another size is the same design.
MARGIN = 0.80

#: The recess: how far inside the plate the ground is knocked back, how much
#: clearance the charge gets from it, and how wide the bright edge stroke is.
INSET = 0.055
GAP = 0.050
RIM = 0.018

#: The keys a spec is made of, in the order the sidecar prints them.
NUMBERS = ("hue", "chroma", "light", "plate_light", "plate_chroma", "detail",
           "weight", "aspect", "rotation", "margin", "bevel", "radius")


def clamp(spec: dict) -> dict:
    """Force a spec onto the legal set: known names, numbers inside their ranges.

    The same contract `gen.ui.style` makes, and for the same reason -- the numbers
    are drawn from themselves and clamped here, so a seed never lands on an icon
    that has quiet been repaired into something else.
    """
    s = motifs.canonical(spec.get("kind", "coin"))
    s.update(dict(spec))
    s["kind"] = motifs.kind(s)
    s["plate"] = str(s["plate"])
    if s["plate"] not in motifs.PLATES:
        s["plate"] = motifs.PLATES[0]
    for k in NUMBERS:
        s[k] = float(s[k])
    s["hue"] = float(s["hue"] % 1.0)
    # The plate's hue is a *second* source, not a number of its own: ``None`` means
    # the plate keeps the charge's hue (the set's original one-hue script), and a
    # number is a faction's own field, which is what ``livery`` hands in.
    s["plate_hue"] = None if s.get("plate_hue") is None else float(s["plate_hue"]) % 1.0
    s["badge"] = _badge.clamp(s.get("badge"))
    s["chroma"] = float(np.clip(s["chroma"], 0.0, 0.30))
    s["light"] = float(np.clip(s["light"], 0.30, 0.95))
    s["plate_light"] = float(np.clip(s["plate_light"], 0.05, 0.60))
    s["plate_chroma"] = float(np.clip(s["plate_chroma"], 0.0, 0.12))
    s["detail"] = float(np.clip(s["detail"], 0.0, 1.0))
    s["weight"] = float(np.clip(s["weight"], 0.0, 1.0))
    s["aspect"] = float(np.clip(s["aspect"], 0.70, 1.40))
    s["rotation"] = float(np.clip(s["rotation"], -35.0, 35.0))
    s["margin"] = float(np.clip(s["margin"], 0.40, 0.94))
    s["bevel"] = float(np.clip(s["bevel"], 0.005, 0.20))
    s["radius"] = float(np.clip(s["radius"], 0.0, 0.25))
    return s


def validate(spec: dict) -> list:
    """Return the invariants ``spec`` breaks; an empty list means legal.

    The names are checked before anything is clamped, because a name that is wrong
    is the one thing `clamp` refuses to repair: a caller that asks what is wrong
    with an icon called ``marble`` gets an answer instead of an exception.
    """
    bad = []
    safe = dict(spec)
    name = str(spec.get("kind", "coin"))
    if name not in motifs.KINDS:
        bad.append(f"kind: {name!r} is not one of {tuple(motifs.KINDS)}")
        safe["kind"] = "coin"
    if str(spec.get("plate", "round")) not in motifs.PLATES:
        bad.append(f"plate: {spec.get('plate')!r} is not one of {motifs.PLATES}")
        safe["plate"] = motifs.PLATES[0]
    try:
        _badge.clamp(safe.get("badge"))
    except (TypeError, ValueError) as exc:
        bad.append(f"badge: {exc}")
        safe["badge"] = None
    try:
        s = clamp(safe)
    except (TypeError, ValueError):
        return bad + ["spec: the numbers are not numbers"]
    if not 0.40 <= s["margin"] <= 0.94:
        bad.append("margin: the plate would leave the tile or fill it edge to edge")
    if s["bevel"] >= s["margin"] * 0.5:
        bad.append("bevel: the bevel is wider than the plate is deep")
    if s["light"] <= s["plate_light"]:
        bad.append("light: the charge is not brighter than the plate it sits on")
    return bad


def tints(spec: dict) -> dict:
    """The four colours an icon is drawn with, all from one hue.

    One hue, four lightnesses: a plate, a brighter rim, a darker recess and a
    bright charge.  That is the entire colour script of the set, and it is why two
    icons of different kinds still look like they belong to the same game.

    The plate's hue is the one exception, and it is a *second source* rather than a
    second script: ``plate_hue`` is ``None`` for the set (the plate then keeps the
    charge's hue, as it always has), and a faction's own field when a game has one,
    so a faction's plate is its colour while the charge stays the kind's.  The rim
    and the recess go with the plate, because all three are the ground: one hue for
    the ground and one for the figure is what keeps two hues from fighting.
    """
    s = clamp(spec)
    hue = s["hue"]
    plate_hue = hue if s.get("plate_hue") is None else s["plate_hue"]
    return {
        "plate": palette.color(s["plate_light"], s["plate_chroma"], plate_hue),
        "rim": palette.color(min(s["plate_light"] + 0.26, 0.78),
                             s["plate_chroma"] * 1.9, plate_hue),
        "recess": palette.color(max(s["plate_light"] - 0.13, 0.05),
                                s["plate_chroma"] * 1.2, plate_hue),
        "charge": palette.color(s["light"], s["chroma"], hue),
    }


def draw(spec: dict, size: int = 192) -> Canvas:
    """One icon at ``size`` pixels: the whole recipe, in one function."""
    s = clamp(spec)
    size = int(size)
    if size < 8:
        raise ValueError(f"an icon smaller than 8 px is a dot, not an icon: {size}")
    c = Canvas(size, size, scale=size / 2.0 * MARGIN)
    X, Y = c.grid()
    dx = c.dx

    d_plate = motifs.plate_field(s["plate"], X, Y, s)
    a_plate = render.coverage(d_plate, dx)
    d_charge = motifs.field(s, X, Y)
    t = tints(s)

    # 1. shadow -- from the plate, in world units, so it is the same shadow at
    #    every size rather than a fixed number of pixels.
    sh = render.drop_shadow(a_plate, c.scale, offset=(0.0, -0.035), sigma=0.030,
                            opacity=0.42)
    c.over_srgb(SHADOW, sh)

    # 2. plate
    c.over(render.shade(t["plate"], d_plate, dx, bevel=s["bevel"], light=LIGHT), a_plate)

    # 3. rim -- a stroke on the plate's edge, clipped to the silhouette, so the
    #    icon has a bright line to read against whatever is behind it.
    rim = np.abs(d_plate) - RIM
    c.over_srgb(t["rim"], render.coverage(rim, dx) * a_plate * 0.85)

    # 4. recess -- the ground, knocked back.  Inside the plate and outside the
    #    charge grown by GAP, which is what puts a dark halo around the symbol and
    #    is most of why a bright charge reads at 32 px.
    inner = d_plate + INSET
    wide = d_charge - GAP
    recess = sdf.smooth_intersection(inner, -wide, 0.04)
    c.over(render.shade(t["recess"], recess, dx, bevel=0.035, light=LIGHT, inset=True),
           render.coverage(recess, dx) * a_plate)

    # 5. charge -- embossed, with a bevel shallower than the plate's, because the
    #    symbol is small and a deep bevel eats it.
    c.over(render.shade(t["charge"], d_charge, dx, bevel=s["bevel"] * 0.65, light=LIGHT),
           render.coverage(d_charge, dx) * a_plate)

    # 6. badge -- a count or a tier, when the spec carries one.  Drawn last, and
    #    over the plate rather than clipped to it, because a badge is the one part
    #    of an icon that is meant to break its silhouette.  Absent, this is one
    #    branch and no pixels: an icon with no badge renders exactly as it did
    #    before there were badges at all.
    _badge.paint(c, s["badge"], s["margin"], X, Y, dx, t)
    return c


def icon(seed: int, size: int = 192, kind: str | None = None) -> Canvas:
    """A whole icon from a seed: which charge, how it leans, and what colour."""
    return draw(motifs.random_spec(seed, kind), size)


# ------------------------------------------------------------------ measuring
def silhouette(canvas: Canvas) -> float:
    """The fraction of the tile the icon actually covers.

    The number a set has to agree on.  A plate at ``margin`` covers a fixed share
    of the tile whatever the charge is, so a kind that scores far below the others
    has grown a thin, unreadable symbol and a kind that scores at the top has a
    charge that has escaped its plate.
    """
    return float((canvas.a > 0.5).mean())


def contrast(canvas: Canvas) -> float:
    """The spread of luminance *inside* the silhouette, in ``[0, 1]``.

    Taken between the 10th and 90th percentile of the covered pixels rather than
    the extremes, because a single anti-aliased edge pixel is not contrast.  An
    icon that fails this is flat: it has a plate, a charge and no difference
    between them, which is the failure that only shows up at 32 px.
    """
    a = np.asarray(canvas.a, dtype=np.float64)
    inside = a > 0.5
    if not inside.any():
        return 0.0
    lum = np.asarray(canvas.rgb, dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    v = lum[inside]
    lo, hi = np.percentile(v, 10.0), np.percentile(v, 90.0)
    return float(hi - lo)
