"""A pole is why a flag hangs from somewhere.

The cloth module makes a flag *hang*; this one gives it something to hang from.  A
pole is neither heraldry nor cloth: it bears no tincture, it is wood or iron, and it
is the one thing on a banner's canvas that the wave must not move.  So it lives in
its own module, it is drawn from its own small vocabulary, and the flag renderer
composites it *after* the wave and *over* the cloth -- over, because the plane of
the cloth passes through the shaft's axis, so the shaft is nearer the camera than
the cloth it cuts.

It decides three things, and only the first of them is about the pole itself.

* **The room.**  A pole rises above the cloth and runs below it, so the frame the
  flag is drawn in has to grow.  The room is quoted as a fraction of the cloth's
  own half-height and handed to the flag *shape*, so the cloth keeps its
  coordinates and only the frame changes.  The few hundredths of a unit by which
  the shaft itself overhangs the hoist are left to the frame's margin, which is
  several times a shaft wide.
* **The anchor.**  Cloth tied to a pole cannot move where it is tied, which is a
  statement about the wave -- ``taper`` of zero -- and it is exact rather than a
  small amplitude: :func:`anchor` sets it, and the hoist is then still at *any*
  phase.
* **The droop.**  A cloth free at the fly hangs lower there, and gravity does not
  consult the folds about which way they run, so the droop is
  :mod:`gen.heraldry.cloth`'s -- the same envelope, taken downwards.

The shading is a cylinder and a sphere, not a bevel.  The shaft's normal turns
through ``(sin t, 0, cos t)`` across its width and the finial's is the hemisphere
itself, because that is what those two objects *are*; a pole shaded like an
extruded rectangle is the one mistake that reads instantly, at any size.  The
spearhead is the exception, because a spearhead really is a plate.

**A bar is the same rod, turned a quarter turn.**  A gonfalon hangs from a crossbar
instead of a mast -- that is :mod:`gen.heraldry.cloth`'s second hanging, and the
reason this module grows a second geometry rather than a second module is that a bar
*is* a rod: the same material, the same rim, and the same cylinder, shaded by the very
same function.  That is not tidiness.  A frame carrying a mast and a crossbar lit by
two different equations is the one frame where the difference would be visible, and a
crossbar that did not match its own mast would be the single thing the eye would find.

What a bar decides is less than what a pole decides.  There is no anchor it has to
state -- :func:`anchor` serves both, because a cloth cannot move where it is held
whoever is holding it -- its ends are turned towards the shoulders rather than towards
the sky, so it carries no finial, and its room is asked for above the cloth's top edge
only, since there is nothing under a bar.  What it has that a pole does not is a
*reach*: a cloth is hung from the middle of a bar and the ends show, which is most of
what makes a gonfalon read as one.
"""
from __future__ import annotations

import numpy as np

from core import palette, render, sdf

from . import paint as _paint

LIGHT = _paint.LIGHT

__all__ = ["LIGHT", "DEFAULTS", "FINIALS", "MATERIALS", "spec", "room", "anchor",
           "field", "paint", "BAR", "bar", "bar_room", "bar_field", "bar_paint"]

DEFAULTS = {
    "width": 0.075,
    "height": 0.26,
    "over": 0.10,
    "finial": "sphere",
    "material": "wood",
}

BAR = {
    "width": 0.075,
    "reach": 0.040,
    "over": 0.030,
    "material": "wood",
}

FINIALS = ("sphere", "spear", "none")

MATERIALS = {
    "wood": (0.37, 0.055, 0.075),
    "iron": (0.50, 0.012, 0.650),
    "gilt": (0.78, 0.125, 0.270),
}

CONTRAST = 0.55

RIM = 0.28
RIM_STRENGTH = 0.42

DROOP = 0.045

_SPHERE_RADIUS = 1.60
_SPHERE_SEAT = 0.30

_SPEAR_RADIUS = 1.10
_SPEAR_HALF = 1.60
_SPEAR_SEAT = 1.30
_SPEAR_TIP = 3.60


def spec(width=None, height=None, over=None, finial=None, material=None) -> dict:
    """Build a pole spec.  A name this module does not know is refused, not fixed.

    A wrong finial or a wrong material is an identifier error -- the same line the
    icon vocabulary draws -- and quietly falling back to wood would put a plausible
    pole on a canvas under a sidecar that says something else.
    """
    kind = str(DEFAULTS["finial"] if finial is None else finial)
    stuff = str(DEFAULTS["material"] if material is None else material)
    if kind not in FINIALS:
        raise ValueError(f"a finial is one of {', '.join(FINIALS)}, not {kind!r}")
    if stuff not in MATERIALS:
        raise ValueError(f"a pole is one of {', '.join(MATERIALS)}, not {stuff!r}")
    w = float(DEFAULTS["width"] if width is None else width)
    h = float(DEFAULTS["height"] if height is None else height)
    o = float(DEFAULTS["over"] if over is None else over)
    if w <= 0.0:
        raise ValueError("a pole has a thickness")
    if h < 0.0 or o < 0.0:
        raise ValueError("a pole does not stop short of what it is asked to reach")
    return {"width": w, "height": h, "over": o, "finial": kind, "material": stuff}


def bar(width=None, reach=None, over=None, material=None) -> dict:
    """Build a crossbar spec -- the one a gonfalon hangs from.

    The same refusal line as :func:`spec`, for the same reason: a name this module
    does not know is an identifier error, and quietly hanging a banner from gilt when
    the sidecar says wood is worse than not drawing it.  A bar has no finial to name,
    because its ends are turned towards the shoulders rather than towards the sky and
    are finished by the same rim as the rest of the rod.
    """
    stuff = str(BAR["material"] if material is None else material)
    if stuff not in MATERIALS:
        raise ValueError(f"a bar is one of {', '.join(MATERIALS)}, not {stuff!r}")
    w = float(BAR["width"] if width is None else width)
    if w <= 0.0:
        raise ValueError("a bar has a thickness")
    r = float(BAR["reach"] if reach is None else reach)
    o = float(BAR["over"] if over is None else over)
    if r < 0.0 or o < 0.0:
        raise ValueError("a bar does not stop short of what it is asked to reach")
    return {"width": w, "reach": r, "over": o, "material": stuff}


def _bar_parts(spec: dict, hw: float, hh: float = 1.0):
    """A bar's geometry in the cloth's coordinates: ``(x0, x1, y0, r)``.

    The rod runs from one shoulder of the cloth to the other and over its top edge,
    so the only numbers are how far past the cloth it reaches, how high it sits above
    it and how thick it is.  It is one function for the same reason the pole's parts
    are: the field, the shading and the room all have to agree about where the rod
    is, and three places that each worked it out would eventually disagree.
    """
    r = 0.5 * abs(float(spec.get("width", BAR["width"])))
    reach = abs(float(spec.get("reach", BAR["reach"])))
    y0 = float(hh) + abs(float(spec.get("over", BAR["over"])))
    return -float(hw) - reach, float(hw) + reach, y0, r


def _parts(spec: dict, hw: float, hh: float = 1.0):
    """The pole's parts, in the cloth's coordinates.

    Returns ``(shaft, finial, geom)``: two distance-field callables (the finial's
    is ``None`` when there is not one) and the numbers behind them.  The shading
    needs those numbers -- a distance field cannot be asked where its centre is --
    and describing the ball twice, once as a disk and once as a hemisphere, is
    exactly how the two drift apart.
    """
    r = 0.5 * abs(float(spec.get("width", DEFAULTS["width"])))
    x0 = -float(hw)
    foot = -float(hh) - abs(float(spec.get("height", DEFAULTS["height"])))
    head = float(hh) + abs(float(spec.get("over", DEFAULTS["over"])))
    kind = str(spec.get("finial", DEFAULTS["finial"]))
    shaft = lambda x, y: sdf.segment(x, y, (x0, foot), (x0, head), r)
    geom = {"x0": x0, "r": r, "foot": foot, "head": head, "kind": kind,
            "up": head - float(hh), "down": -float(hh) - foot}
    fin = None
    if kind == "sphere":
        rr = _SPHERE_RADIUS * r
        cy = head + _SPHERE_SEAT * rr
        geom.update(rr=rr, cy=cy, up=cy + rr - float(hh))
        fin = lambda x, y: sdf.circle(x, y, rr, x0, cy)
    elif kind == "spear":
        rr = _SPEAR_RADIUS * r
        tip = head + _SPEAR_TIP * rr
        pts = np.array([[x0, tip], [x0 + _SPEAR_HALF * rr, head + _SPEAR_SEAT * rr],
                        [x0, head], [x0 - _SPEAR_HALF * rr, head + _SPEAR_SEAT * rr]])
        geom.update(rr=rr, tip=tip, up=tip - float(hh))
        fin = lambda x, y: sdf.polygon(x, y, pts)
    return shaft, fin, geom


def room(spec: dict) -> float:
    """How much taller the flag's frame has to be, of the cloth's half-height.

    The frame grows by this much above *and* below the cloth, so this is the larger
    of the two sides -- a margin is one number, and the honest thing is to give it
    the biggest thing that has to fit rather than to pretend it can be per-side.
    """
    _shaft, _fin, g = _parts(spec, 1.0, 1.0)
    return float(max(g["up"], g["down"])) + RIM * float(g["r"])


def bar_room(spec: dict) -> float:
    """How much taller the flag's frame has to be, of the cloth's half-height.

    Only the top, because a bar is over the cloth and there is nothing under it.  The
    reach along the cloth's width is deliberately not asked for here: this grows the
    frame *vertically*, and what it really buys is scale -- the cloth keeps its own
    coordinates and everything around it comes in a little -- so the honest question
    is only whether the rod's crown clears the frame, and the margin every flag is
    already drawn with is what covers the couple of hundredths it reaches sideways.
    """
    r = 0.5 * abs(float(spec.get("width", BAR["width"])))
    over = abs(float(spec.get("over", BAR["over"])))
    return over + r * (1.0 + RIM)


def anchor(wave, droop: float = DROOP):
    """The wave a cloth makes when something is holding it: still there, free beyond.

    Two statements out of one fact, and they hold whoever is doing the holding.  A
    cloth tied to a pole, or sleeved onto a bar, cannot move where it is held, so
    ``taper`` goes to zero -- and that is exact, not a small amplitude: the held edge
    is still at any phase.  A cloth free at its other end hangs lower there, so the
    wave gets a droop as well, and one already larger than the default is left alone.
    Given no wave it returns what it was given: a flag that is not being waved is not
    anchored to anything.
    """
    if not wave:
        return wave
    out = dict(wave)
    out["taper"] = 0.0
    out["droop"] = max(abs(float(droop)), float(wave.get("droop", 0.0)))
    return out


def field(spec: dict, hw: float, hh: float = 1.0):
    """The pole's distance field, in the cloth's coordinates."""
    shaft, fin, _g = _parts(spec, hw, hh)
    if fin is None:
        return shaft
    return lambda x, y: sdf.union(shaft(x, y), fin(x, y))


def bar_field(spec: dict, hw: float, hh: float = 1.0):
    """A bar's distance field, in the cloth's coordinates."""
    x0, x1, y0, r = _bar_parts(spec, hw, hh)
    return lambda x, y: sdf.segment(x, y, (x0, y0), (x1, y0), r)


def paint(canvas, spec: dict, hw: float, hh: float = 1.0, light=LIGHT):
    """Paint the pole, shaded as the cylinder and the sphere it is, over the cloth.

    The pole is painted *after* the wave and *in front of* the cloth, and that is
    geometry rather than taste: the plane of the cloth passes through the shaft's
    axis, so every column of the shaft's image is nearer the camera than the plane
    it cuts and the shaft wins the whole overlap.  It is also why a pole is drawn
    over the hoist rather than tucked under it -- a cloth tied to a pole is tied to
    the *near* side of it.
    """
    X, Y = canvas.grid()
    shaft, fin, g = _parts(spec, hw, hh)
    d = shaft(X, Y)
    d_fin = None
    if fin is not None:
        d_fin = fin(X, Y)
        d = sdf.union(d, d_fin)

    r = max(float(g["r"]), 1e-9)
    lat = np.clip((X - g["x0"]) / r, -1.0, 1.0)
    n = np.stack([lat, np.zeros_like(lat),
                  np.sqrt(np.clip(1.0 - lat * lat, 0.0, 1.0))], axis=-1)
    if d_fin is not None:
        inside = (d_fin <= 0.0)[..., None]
        if g["kind"] == "sphere":
            rr = max(float(g["rr"]), 1e-9)
            u = (X - g["x0"]) / rr
            v = (Y - g["cy"]) / rr
            rho = np.clip(u * u + v * v, 0.0, 1.0)
            n = np.where(inside, np.stack([u, v, np.sqrt(1.0 - rho)], axis=-1), n)
        else:

            h = np.clip(-d_fin, 0.0, 0.9 * float(g["rr"]))
            n = np.where(inside, render.height_normals(h, canvas.dx, canvas.y_up), n)
    return _rod(canvas, d, n, r, spec.get("material", DEFAULTS["material"]), light)


def bar_paint(canvas, spec: dict, hw: float, hh: float = 1.0, light=LIGHT):
    """Paint a crossbar over the cloth, shaded as the cylinder it is.

    Drawn after the cloth and in front of it, for the same reason the pole is: a
    cloth is sleeved onto a bar, so the bar is on the near side of the cloth's top
    edge, and at the default ``over`` the cloth it covers is the sleeve's own hem
    rather than any of the design.
    """
    X, Y = canvas.grid()
    x0, x1, y0, r = _bar_parts(spec, hw, hh)
    d = sdf.segment(X, Y, (x0, y0), (x1, y0), r)

    rr = max(float(r), 1e-9)
    lat = np.clip((Y - y0) / rr, -1.0, 1.0)
    n = np.stack([np.zeros_like(lat), lat,
                  np.sqrt(np.clip(1.0 - lat * lat, 0.0, 1.0))], axis=-1)
    return _rod(canvas, d, n, r, spec.get("material", BAR["material"]), light)


def _rod(canvas, d, n, r, material, light=LIGHT):
    """Shade a rod and composite it over the canvas: the one light, and the rim.

    Both a pole and a bar come through here, and that is the point of it.  They are
    the same object turned a quarter turn, so they are the same material, the same
    contrast and the same rim -- and a crossbar lit by a second copy of this code
    would be the one place a banner on a bar and a banner on a pole could disagree
    about what a piece of wood looks like under the project's one light.
    """
    a = render.coverage(d, canvas.dx)
    if not bool(np.any(a > 0.0)):
        return canvas
    L = np.asarray([light[0], light[1], 0.65], dtype=np.float64)
    L = L / max(float(np.linalg.norm(L)), 1e-12)
    lam = np.clip(n @ L, 0.0, 1.0)
    factor = 1.0 + CONTRAST * (lam - float(L[2])) / max(float(L[2]), 1e-9)
    core = palette.color(*MATERIALS[str(material)])
    rgb = palette.srgb_to_linear(core) * factor[..., None]
    rim = RIM * max(float(r), 1e-9)
    edge = np.clip(1.0 - np.clip(-d, 0.0, rim) / max(rim, 1e-9), 0.0, 1.0) * (d < 0.0)
    rgb = rgb * (1.0 - RIM_STRENGTH * edge)[..., None]
    np.clip(rgb, 0.0, 1.0, out=rgb)
    canvas.over(rgb, a)
    return canvas
