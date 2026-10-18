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
"""
from __future__ import annotations

import numpy as np

from core import palette, render, sdf

from . import paint as _paint

#: The single light shared by every generated asset.
LIGHT = _paint.LIGHT

__all__ = ["LIGHT", "DEFAULTS", "FINIALS", "MATERIALS", "spec", "room", "anchor",
           "field", "paint"]

#: What a pole gets when it is asked for and nothing more is said.  Everything is
#: quoted against the cloth's own half-height, which for a flag is 1.0.
DEFAULTS = {
    "width": 0.075,     # the shaft's thickness
    "height": 0.26,     # how far the shaft runs below the cloth
    "over": 0.10,       # how far it rises above it
    "finial": "sphere",
    "material": "wood",
}

#: A pole is not heraldic and it does not obey the rule of tincture: it has no
#: tincture at all, so its vocabulary is a material rather than a colour.
FINIALS = ("sphere", "spear", "none")

#: The three materials, as OKLCH: a dark warm wood, a cold iron, a gilt.  The hues
#: are OKLCH turns -- wood near 27 degrees, gold near 97 -- because a hue that
#: "looks about right" in degrees is the one way to get a tan and call it gold.
MATERIALS = {
    "wood": (0.37, 0.055, 0.075),
    "iron": (0.50, 0.012, 0.650),
    "gilt": (0.78, 0.125, 0.270),
}

#: How hard the shaft's curvature catches the light.  The same convention the cloth
#: uses: the Lambert term is scaled about the value a *flat* surface would have, so
#: a cylinder redistributes light instead of merely being dark at its edges.
CONTRAST = 0.55

#: The dark line just inside the silhouette, how hard it is, and how much of it
#: there is: a fraction of the shaft's own radius, so a pole several pixels wide
#: does not disappear into its own outline.
RIM = 0.28
RIM_STRENGTH = 0.42

#: How far the free end of a pole-borne flag slips down, of the folds' extent.
DROOP = 0.045

#: The ball's radius and how deeply it is seated, both as multiples of the shaft's
#: radius -- one number each, so the finial and the room it needs cannot disagree.
#: A finial fatter than about one and a half shafts stops reading as a finial and
#: starts reading as a knot, so the ball is one and a half.
_SPHERE_RADIUS = 1.60
_SPHERE_SEAT = 0.30

#: The spearhead: a kite this many shaft radii wide at its shoulders, seated this
#: far up the shaft and tipped this far above the head.
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
    shaft = lambda x, y: sdf.segment(x, y, (x0, foot), (x0, head), r)  # noqa: E731
    geom = {"x0": x0, "r": r, "foot": foot, "head": head, "kind": kind,
            "up": head - float(hh), "down": -float(hh) - foot}
    fin = None
    if kind == "sphere":
        rr = _SPHERE_RADIUS * r
        cy = head + _SPHERE_SEAT * rr
        geom.update(rr=rr, cy=cy, up=cy + rr - float(hh))
        fin = lambda x, y: sdf.circle(x, y, rr, x0, cy)  # noqa: E731
    elif kind == "spear":
        rr = _SPEAR_RADIUS * r
        tip = head + _SPEAR_TIP * rr
        pts = np.array([[x0, tip], [x0 + _SPEAR_HALF * rr, head + _SPEAR_SEAT * rr],
                        [x0, head], [x0 - _SPEAR_HALF * rr, head + _SPEAR_SEAT * rr]])
        geom.update(rr=rr, tip=tip, up=tip - float(hh))
        fin = lambda x, y: sdf.polygon(x, y, pts)  # noqa: E731
    return shaft, fin, geom


def room(spec: dict) -> float:
    """How much taller the flag's frame has to be, of the cloth's half-height.

    The frame grows by this much above *and* below the cloth, so this is the larger
    of the two sides -- a margin is one number, and the honest thing is to give it
    the biggest thing that has to fit rather than to pretend it can be per-side.
    """
    _shaft, _fin, g = _parts(spec, 1.0, 1.0)
    return float(max(g["up"], g["down"])) + RIM * float(g["r"])


def anchor(wave, droop: float = DROOP):
    """The wave a cloth tied to a pole makes: still at the hoist, trailing at the fly.

    Two statements out of one fact.  A cloth tied to a pole cannot move where it is
    tied, so ``taper`` goes to zero -- and that is exact, not a small amplitude: the
    hoist is still at any phase.  A cloth free at the fly hangs lower there, so the
    wave gets a droop as well, and one already larger than the default is left
    alone.  Given no wave it returns what it was given: a flag that is not being
    waved is not anchored to anything.
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
    return lambda x, y: sdf.union(shaft(x, y), fin(x, y))  # noqa: E731


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
    a = render.coverage(d, canvas.dx)
    if not bool(np.any(a > 0.0)):
        return canvas

    # The shaft is a cylinder, so its normal turns across the width and never
    # points sideways at the silhouette -- which is why the edges of a pole are dark
    # and its middle is not.  The finial replaces that normal where it covers it.
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
            # A spearhead is a plate, so its normal comes from its own edge bevel --
            # the one place in this module where the house bevel is the right answer.
            h = np.clip(-d_fin, 0.0, 0.9 * float(g["rr"]))
            n = np.where(inside, render.height_normals(h, canvas.dx, canvas.y_up), n)

    L = np.asarray([light[0], light[1], 0.65], dtype=np.float64)
    L = L / max(float(np.linalg.norm(L)), 1e-12)
    lam = np.clip(n @ L, 0.0, 1.0)
    factor = 1.0 + CONTRAST * (lam - float(L[2])) / max(float(L[2]), 1e-9)
    core = palette.color(*MATERIALS[str(spec.get("material", DEFAULTS["material"]))])
    rgb = palette.srgb_to_linear(core) * factor[..., None]
    rim = RIM * r
    edge = np.clip(1.0 - np.clip(-d, 0.0, rim) / max(rim, 1e-9), 0.0, 1.0) * (d < 0.0)
    rgb = rgb * (1.0 - RIM_STRENGTH * edge)[..., None]
    np.clip(rgb, 0.0, 1.0, out=rgb)
    canvas.over(rgb, a)
    return canvas
