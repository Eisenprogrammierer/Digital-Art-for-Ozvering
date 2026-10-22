"""The cut of the clothes: civilian tailoring and religious vestments.

The outfit layer in :mod:`gen.portraits.regalia` answers *what dress code* a
sitter wears -- civil, military, police, naval, ceremonial, clerical -- and paints
the marks of that code: collar tabs, boards, medals, a sash.  This module answers
the other half of the question: *how the garment is cut*.  A civilian can wear a
shirt, a blouse, a jacket, a greatcoat, a dress or a mantle; a cleric can wear a
cassock, a cope or a full habit.  The two axes are independent, so an explicit
genome may ask for a uniform greatcoat or a civilian robe, and the painter will
honour it.

The cut is a gene appended after the regalia block, so old DNA decodes to
``garment=auto``: civil means a shirt, clerical a cassock, and a service a tunic.
That is exactly the bust those genomes rendered before this layer existed.

Two cuts are about *silhouette* rather than detail, and they are the reason this
is a separate pass:

* the jacket and the greatcoat carry a shoulder pad and a lapel that runs out
  towards the deltoid, so they read as broad shoulders;
* the mantle and the habit are a cape over both shoulders, with a V opening and a
  clasp.  They deliberately hide the shoulder line and the epaulettes underneath
  them -- the one garment in the wardrobe that takes width away.

The painters only draw cloth.  They never move the head, the neck or the bust, so
a garment cannot change the sitter's anatomy; it can only decide what is visible.
"""
from __future__ import annotations

import numpy as np

from core import palette

from . import brush
from . import shape as S
from . import head

__all__ = [
    "CUTS",
    "DEFAULTS",
    "resolve",
    "base_cut",
    "neckline",
    "random_value",
    "paint_front",
    "paint_mantle",
    "paint_veil",
]

#: Every value the ``garment`` gene may hold.  ``auto`` is the compatibility
#: value: it resolves from the dress code instead of naming a cut.
CUTS = ("auto", "shirt", "blouse", "jacket", "coat", "dress", "robe", "mantle", "habit")

DEFAULTS = {"garment": "auto"}

#: A cut that is not one of the named garments but is still a valid *base*: the
#: service tunic is drawn by :mod:`detail` and :mod:`regalia`, never by this file.
_TUNIC = "tunic"


def resolve(attire: str, garment: str) -> str:
    """The visible cut, with ``auto`` resolved by the dress code."""
    cut = str(garment) if str(garment) in CUTS else "auto"
    if cut != "auto":
        return cut
    if attire == "clerical":
        return "robe"
    if attire == "civil":
        return "shirt"
    return _TUNIC


def base_cut(attire: str, garment: str) -> str:
    """The cut underneath any overgarment.

    A mantle is a cape over the dress code's own base; a habit is a robe plus a
    mantle and a veil.  Keeping that distinction here means the rest of the
    painter asks "what is underneath?" without repeating the rule.
    """
    cut = resolve(attire, garment)
    if cut == "mantle":
        return resolve(attire, "auto")
    if cut == "habit":
        return "robe"
    return cut


def neckline(attire: str, garment: str, kind: str) -> str:
    """The neckline family a cut asks for, before the culture's default."""
    cut = base_cut(attire, garment)
    if cut in ("blouse", "dress"):
        return "scoop"
    if cut == "robe":
        return "crew"
    return kind


def random_value(rng: np.random.Generator, attire: str) -> str:
    """Draw a cut that belongs to the dress code.  Never called before attire."""
    if attire == "clerical":
        return str(rng.choice(("auto", "robe", "habit", "mantle"),
                              p=(0.22, 0.36, 0.26, 0.16)))
    if attire == "civil":
        return str(rng.choice(("auto", "shirt", "blouse", "jacket", "coat",
                               "dress", "mantle"),
                              p=(0.18, 0.20, 0.15, 0.19, 0.10, 0.12, 0.06)))
    if attire == "ceremonial":
        return str(rng.choice(("auto", "coat", "mantle"),
                              p=(0.45, 0.15, 0.40)))
    return str(rng.choice(("auto", "coat", "mantle"),
                          p=(0.72, 0.16, 0.12)))


# --------------------------------------------------------------------------
# colour helpers
# --------------------------------------------------------------------------
def _lw(st) -> float:
    return float(st.ink_w) * float(st.line)


def _cloth(kit):
    return (kit.service, kit.service_hi, kit.service_light,
            kit.service_shadow, kit.service_line, kit.accent)


def _trim(kit):
    """A narrow trim colour: metal on ceremonial dress, the accent otherwise."""
    if kit.attire == "ceremonial":
        return kit.gold
    return kit.accent


def _paint_shadow(canvas, shape, colour, clip, alpha: float = 0.28,
                  drop: float = 0.010, grow: float = 0.010, soft: float = 1.4) -> None:
    shade = shape.offset(grow).place(cy=-drop) - shape
    brush.paint(canvas, shade, colour, clip=clip, alpha=alpha, soft=soft)


# --------------------------------------------------------------------------
# civilian cuts
# --------------------------------------------------------------------------
def _blouse(canvas, st, h, kit, clip) -> None:
    """A soft scoop and a knotted collar: a blouse, not a shirt."""
    top = float(st.neck_y)
    nw = float(st.neck_w)
    sw = float(st.shoulder_w)
    x0 = min(nw * 1.16, sw * 0.285)
    depth = 0.060
    lw = _lw(st)
    cloth, hi, light, shadow, line, accent = _cloth(kit)

    # The rounded collar follows the U of the scoop, not a V.
    spine = [(-x0 * 1.00, top + 0.048), (-x0 * 0.66, top - depth * 0.55),
             (0.0, top - depth - 0.004), (x0 * 0.66, top - depth * 0.55),
             (x0 * 1.00, top + 0.048)]
    band = S.lock(spine, w0=0.017, w1=0.017, bulge=0.005, per_segment=7)
    _paint_shadow(canvas, band, shadow, clip, alpha=0.22, drop=0.009,
                  grow=0.008, soft=1.2)
    brush.paint(canvas, band, palette.lighten(hi, 0.075), clip=clip,
                alpha=1.0, soft=0.35)
    brush.paint(canvas, band.outline(lw * 0.46), line, clip=clip,
                alpha=0.52, soft=0.3)
    brush.paint(canvas, S.lock([(p[0], p[1] + 0.006) for p in spine],
                               w0=0.0030, w1=0.0026, per_segment=6),
                light, clip=clip, alpha=0.36, soft=0.6)

    # A small bow at the bottom of the scoop.  Two loops and a knot; at 256 px
    # this is the single mark that separates a blouse from a plain shirt.
    cy = top - depth - 0.020
    knot = S.circle(0.011, 0.0, cy)
    left = S.poly([(0.0, cy + 0.003), (-0.050, cy + 0.025),
                   (-0.054, cy - 0.016), (-0.012, cy - 0.008)], resample=24)
    right = S.poly([(0.0, cy + 0.003), (0.050, cy + 0.025),
                    (0.054, cy - 0.016), (0.012, cy - 0.008)], resample=24)
    for loop in (left, right):
        brush.paint(canvas, loop, accent, clip=clip, alpha=0.90, soft=0.4)
        brush.paint(canvas, loop.outline(lw * 0.34), line, clip=clip,
                    alpha=0.45, soft=0.3)
    brush.paint(canvas, knot, accent, clip=clip, alpha=0.96, soft=0.3)
    brush.paint(canvas, knot.outline(lw * 0.30), line, clip=clip,
                alpha=0.48, soft=0.25)
    brush.paint(canvas, S.circle(0.004, -0.003, cy + 0.004),
                palette.lighten(accent, 0.10), clip=clip, alpha=0.8, soft=0.3)

    # Gathers at the shoulder seam: a blouse is soft where a jacket is padded.
    for side in (+1, -1):
        for k in range(2):
            x = side * (sw * (0.28 + 0.055 * k))
            brush.paint(canvas, S.lock([(x, top - 0.005), (x + side * 0.020,
                                                          top - 0.085)],
                                       w0=0.0020, w1=0.0010, per_segment=3),
                        shadow, clip=clip, alpha=0.16, soft=1.0)


# --------------------------------------------------------------------------
# tailored bodies
# --------------------------------------------------------------------------
def _body_panel(canvas, st, kit, clip, colour, v_apex: float,
                alpha: float = 0.94):
    """The jacket/coat body: cloth around a V, so the shirt reads underneath.

    Two lapels on a shirt do not make a jacket.  The body has to cover the chest
    outside the opening; what stays visible through the V is the shirt the jacket
    is worn over.  The V is subtracted from the body, not painted over it, so the
    collar and the opening underneath remain real geometry.
    """
    top = float(st.neck_y)
    nw = float(st.neck_w)
    sw = float(st.shoulder_w)
    x0 = min(nw * 1.16, sw * 0.285)
    outer = S.poly([(-sw * 0.68, top + 0.008), (sw * 0.68, top + 0.008),
                    (sw * 0.72, top - 0.44), (-sw * 0.72, top - 0.44)],
                   resample=52)
    hole = S.poly([(-x0 * 1.06, top + 0.070), (0.0, v_apex),
                   (x0 * 1.06, top + 0.070)], resample=36)
    body = outer - hole
    brush.paint(canvas, body, colour, clip=clip, alpha=alpha, soft=0.55)
    # A broad key and a broad shade: the body is a plane, not a flat fill.
    brush.paint(canvas, body & S.ellipse(sw * 0.28, 0.17,
                                         -sw * 0.22, top - 0.06),
                kit.service_light, clip=clip, alpha=0.055, soft=6.0)
    brush.paint(canvas, body & S.ellipse(sw * 0.24, 0.16,
                                         sw * 0.28, top - 0.10),
                kit.service_shadow, clip=clip, alpha=0.10, soft=5.0)
    # The seam where the sleeve meets the body.
    for side in (+1, -1):
        a = (side * sw * 0.58, top + 0.004)
        b = (side * sw * 0.63, top - 0.230)
        brush.paint(canvas, S.lock([a, b], w0=0.0024, w1=0.0030,
                                   per_segment=4),
                    kit.service_line, clip=clip, alpha=0.30, soft=0.7)
    return body


def _jacket(canvas, st, h, kit, clip) -> None:
    """A tailored jacket: a lapel, a notch and a padded shoulder."""
    top = float(st.neck_y)
    nw = float(st.neck_w)
    sw = float(st.shoulder_w)
    x0 = min(nw * 1.16, sw * 0.285)
    lw = _lw(st)
    cloth, hi, light, shadow, line, accent = _cloth(kit)
    body_col = palette.darken(cloth, 0.100)
    _body_panel(canvas, st, kit, clip, body_col, top - 0.150)

    for side in (+1, -1):
        a = (side * x0 * 1.66, top + 0.030)
        b = (side * x0 * 1.48, top - 0.062)
        c = (side * x0 * 0.72, top - 0.160)
        d = (side * 0.032, top - 0.160)
        e = (side * x0 * 0.80, top - 0.005)
        lapel = S.poly([a, b, c, d, e], resample=40)
        _paint_shadow(canvas, lapel, shadow, clip, alpha=0.26, drop=0.012,
                      grow=0.010, soft=1.4)
        lapel_col = palette.darken(body_col, 0.035)
        brush.paint(canvas, lapel, lapel_col, clip=clip,
                    alpha=1.0, soft=0.40)
        # The lapel is the jacket's own cloth, one turn darker than the body; a
        # lighter fill turns it back into a shirt panel.  A lit satin facing along
        # the free edge is what shows the turn of the cloth.
        facing = S.lock([e, d, c], w0=0.009, w1=0.011, per_segment=5)
        brush.paint(canvas, lapel & facing, palette.lighten(lapel_col, 0.080),
                    clip=clip, alpha=0.62, soft=0.7)
        # The notch: a small wedge cut out where a collar would meet the lapel.
        notch = S.poly([(side * x0 * 1.24, top - 0.022),
                        (side * x0 * 1.38, top - 0.068),
                        (side * x0 * 1.08, top - 0.060)], resample=18)
        brush.paint(canvas, notch, palette.darken(body_col, 0.060), clip=clip,
                    alpha=0.78, soft=0.5)
        brush.paint(canvas, lapel.outline(lw * 0.36), line, clip=clip,
                    alpha=0.38, soft=0.50)
        # A padded shoulder: broad, low-alpha light on the deltoid.  It is the
        # jacket's version of the epaulette -- width asserted by tailoring.
        brush.paint(canvas, S.ellipse(sw * 0.17, 0.042,
                                      side * sw * 0.42, top - 0.014),
                    light, clip=clip, alpha=0.10, soft=3.2)
        brush.paint(canvas, S.ellipse(sw * 0.10, 0.020,
                                      side * sw * 0.48, top - 0.050),
                    shadow, clip=clip, alpha=0.12, soft=2.6)

    # The jacket closes below the lapels: a centre seam and two buttons.
    brush.paint(canvas, S.lock([(0.0, top - 0.145), (0.0, top - 0.232)],
                               w0=0.0022, w1=0.0030, per_segment=4),
                line, clip=clip, alpha=0.36, soft=0.6)
    for k in range(2):
        y = top - 0.185 - 0.035 * k
        brush.paint(canvas, S.circle(0.0105, 0.0, y), kit.metal, clip=clip,
                    alpha=0.94, soft=0.3)
        brush.paint(canvas, S.circle(0.0105, 0.0, y).outline(0.0028),
                    kit.metal_shadow, clip=clip, alpha=0.68, soft=0.2)


def _coat(canvas, st, h, kit, clip) -> None:
    """A greatcoat: a broader, deeper lapel and a double-breasted front."""
    top = float(st.neck_y)
    nw = float(st.neck_w)
    sw = float(st.shoulder_w)
    x0 = min(nw * 1.16, sw * 0.285)
    lw = _lw(st)
    cloth, hi, light, shadow, line, accent = _cloth(kit)
    body_col = palette.darken(cloth, 0.155)
    _body_panel(canvas, st, kit, clip, body_col, top - 0.170)

    for side in (+1, -1):
        a = (side * x0 * 1.86, top + 0.036)
        b = (side * x0 * 1.66, top - 0.066)
        c = (side * x0 * 0.96, top - 0.172)
        d = (side * 0.058, top - 0.172)
        e = (side * x0 * 0.92, top - 0.005)
        lapel = S.poly([a, b, c, d, e], resample=44)
        _paint_shadow(canvas, lapel, shadow, clip, alpha=0.28, drop=0.014,
                      grow=0.012, soft=1.5)
        lapel_col = palette.darken(body_col, 0.045)
        brush.paint(canvas, lapel, lapel_col, clip=clip,
                    alpha=1.0, soft=0.42)
        facing = S.lock([e, d, c], w0=0.011, w1=0.013, per_segment=5)
        brush.paint(canvas, lapel & facing, palette.lighten(lapel_col, 0.090),
                    clip=clip, alpha=0.62, soft=0.7)
        notch = S.poly([(side * x0 * 1.50, top - 0.022),
                        (side * x0 * 1.66, top - 0.074),
                        (side * x0 * 1.34, top - 0.068)], resample=18)
        brush.paint(canvas, notch, palette.darken(body_col, 0.065), clip=clip,
                    alpha=0.80, soft=0.5)
        brush.paint(canvas, lapel.outline(lw * 0.38), line, clip=clip,
                    alpha=0.40, soft=0.50)
        # A heavier pad than the jacket's: a greatcoat is built, not draped.
        brush.paint(canvas, S.ellipse(sw * 0.20, 0.050,
                                      side * sw * 0.43, top - 0.012),
                    light, clip=clip, alpha=0.12, soft=3.4)
        brush.paint(canvas, S.ellipse(sw * 0.12, 0.024,
                                      side * sw * 0.49, top - 0.052),
                    shadow, clip=clip, alpha=0.14, soft=2.8)

    # Double-breasted: two columns of buttons, and a centre overlap edge.
    brush.paint(canvas, S.lock([(0.0, top - 0.150), (0.0, top - 0.235)],
                               w0=0.0024, w1=0.0032, per_segment=4),
                line, clip=clip, alpha=0.34, soft=0.6)
    for side in (+1, -1):
        x = side * 0.055
        for k in range(2):
            y = top - 0.182 - 0.033 * k
            brush.paint(canvas, S.circle(0.0105, x, y), kit.metal, clip=clip,
                        alpha=0.94, soft=0.3)
            brush.paint(canvas, S.circle(0.0105, x, y).outline(0.0028),
                        kit.metal_shadow, clip=clip, alpha=0.68, soft=0.2)


def _dress(canvas, st, h, kit, clip) -> None:
    """A fitted bodice: a scoop trim, princess seams and a waist seam."""
    top = float(st.neck_y)
    nw = float(st.neck_w)
    sw = float(st.shoulder_w)
    x0 = min(nw * 1.16, sw * 0.285)
    depth = 0.070
    lw = _lw(st)
    cloth, hi, light, shadow, line, accent = _cloth(kit)

    # The neckline trim follows the scoop and is the only hard line on a dress.
    spine = [(-x0 * 1.02, top + 0.046), (-x0 * 0.70, top - depth * 0.55),
             (0.0, top - depth - 0.005), (x0 * 0.70, top - depth * 0.55),
             (x0 * 1.02, top + 0.046)]
    trim = S.lock(spine, w0=0.010, w1=0.010, bulge=0.003, per_segment=7)
    brush.paint(canvas, trim, accent, clip=clip, alpha=0.86, soft=0.4)
    brush.paint(canvas, trim.outline(lw * 0.35), line, clip=clip,
                alpha=0.45, soft=0.3)

    # Princess seams: two long, soft curves from the neckline to the waist.  They
    # are what turn a flat cloth panel into a fitted bodice.
    for side in (+1, -1):
        a = (side * sw * 0.235, top - 0.040)
        b = (side * sw * 0.170, top - 0.212)
        seam = S.lock([a, ((a[0] + b[0]) * 0.5 + side * 0.012,
                           (a[1] + b[1]) * 0.5), b],
                      w0=0.0042, w1=0.0055, per_segment=6)
        brush.paint(canvas, seam, shadow, clip=clip, alpha=0.20, soft=2.0)
        brush.paint(canvas, S.lock([(p[0] - side * 0.006, p[1]) for p in
                                     (a, b)], w0=0.0018, w1=0.0018,
                                   per_segment=3),
                    light, clip=clip, alpha=0.16, soft=1.4)

    # The waist seam, with a narrow belt in the trim.  The bust is cropped above
    # the waist, but the seam is still the line that says "dress".
    waist = S.lock([(-sw * 0.40, top - 0.188), (0.0, top - 0.207),
                    (sw * 0.40, top - 0.188)],
                   w0=0.0060, w1=0.0060, bulge=0.004, per_segment=7)
    brush.paint(canvas, waist, line, clip=clip, alpha=0.30, soft=0.8)
    belt = S.lock([(-sw * 0.38, top - 0.182), (0.0, top - 0.201),
                   (sw * 0.38, top - 0.182)],
                  w0=0.014, w1=0.014, bulge=0.003, per_segment=7)
    brush.paint(canvas, belt, palette.darken(accent, 0.03), clip=clip,
                alpha=0.70, soft=0.6)
    brush.paint(canvas, S.circle(0.014, 0.0, top - 0.204), kit.metal,
                clip=clip, alpha=0.82, soft=0.4)


def _robe(canvas, st, h, kit, clip) -> None:
    """A cassock or scholar's robe: a closed panel, a cincture, wide sleeves."""
    top = float(st.neck_y)
    nw = float(st.neck_w)
    sw = float(st.shoulder_w)
    x0 = min(nw * 1.16, sw * 0.285)
    lw = _lw(st)
    cloth, hi, light, shadow, line, accent = _cloth(kit)

    # The closed front.  It covers the shirt's opening and placket completely:
    # a robe does not show a collarbone, and it does not button like a shirt.
    panel = S.poly([(-x0 * 1.30, top + 0.015), (x0 * 1.30, top + 0.015),
                    (sw * 0.32, top - 0.400), (-sw * 0.32, top - 0.400)],
                   resample=44)
    panel_col = palette.darken(cloth, 0.085) if kit.attire == "civil" else cloth
    brush.paint(canvas, panel, panel_col, clip=clip, alpha=0.96, soft=0.55)
    # Two long planes: the panel's left half catches the key, the right falls into
    # its own shadow.  One flat rectangle reads as cardboard.
    brush.paint(canvas, panel & S.box(sw * 0.18, 0.24, 0.0,
                                      -sw * 0.18, top - 0.12),
                light, clip=clip, alpha=0.10, soft=3.0)
    brush.paint(canvas, panel & S.box(sw * 0.18, 0.24, 0.0,
                                      sw * 0.22, top - 0.12),
                shadow, clip=clip, alpha=0.16, soft=3.0)
    for side in (+1, -1):
        seam = S.lock([(side * x0 * 1.28, top + 0.008),
                       (side * sw * 0.32, top - 0.400)],
                      w0=0.0030, w1=0.0040, per_segment=4)
        brush.paint(canvas, seam, line, clip=clip, alpha=0.36, soft=0.6)
    brush.paint(canvas, S.lock([(0.0, top - 0.030), (0.0, top - 0.400)],
                               w0=0.0022, w1=0.0030, per_segment=4),
                line, clip=clip, alpha=0.30, soft=0.6)

    # A row of small buttons on the centre line, the way a cassock closes.
    for k in range(3):
        y = top - 0.064 - 0.060 * k
        brush.paint(canvas, S.circle(0.0075, 0.0, y), light, clip=clip,
                    alpha=0.78, soft=0.3)
        brush.paint(canvas, S.circle(0.0075, 0.0, y).outline(0.0022),
                    line, clip=clip, alpha=0.52, soft=0.2)

    # The cincture: a sash at the waist, with a knot and two short ends.  For a
    # habit this is the rope belt; for a scholar's robe it is the cloth sash.
    cord = accent if kit.attire == "civil" else light
    band = S.lock([(-sw * 0.30, top - 0.192), (0.0, top - 0.202),
                   (sw * 0.30, top - 0.192)],
                  w0=0.012, w1=0.012, bulge=0.004, per_segment=7)
    brush.paint(canvas, band, cord, clip=clip, alpha=0.90, soft=0.5)
    brush.paint(canvas, band.outline(lw * 0.34), line, clip=clip,
                alpha=0.46, soft=0.3)
    brush.paint(canvas, S.circle(0.013, 0.0, top - 0.202), cord, clip=clip,
                alpha=0.95, soft=0.35)
    for side in (+1, -1):
        end = S.lock([(side * 0.006, top - 0.208),
                      (side * 0.024, top - 0.250)],
                     w0=0.008, w1=0.003, per_segment=4)
        brush.paint(canvas, end, cord, clip=clip, alpha=0.86, soft=0.4)
        brush.paint(canvas, end.outline(lw * 0.28), line, clip=clip,
                    alpha=0.40, soft=0.3)

    # Wide sleeves: a seam and a fold at the outer edge.  The arm is cropped, but
    # the break at the sleeve keeps the robe from reading as a cape.
    for side in (+1, -1):
        a = (side * sw * 0.58, top - 0.020)
        b = (side * sw * 0.64, top - 0.240)
        seam = S.lock([a, b], w0=0.0026, w1=0.0034, per_segment=4)
        brush.paint(canvas, seam, line, clip=clip, alpha=0.30, soft=0.7)
        fold = S.lock([(a[0] + side * 0.040, a[1]), (b[0] + side * 0.050, b[1])],
                      w0=0.018, w1=0.010, per_segment=4)
        brush.paint(canvas, fold, shadow, clip=clip, alpha=0.14, soft=2.6)

    # A standing collar for a civilian robe.  A clerical robe's white band is
    # regalia's job and is painted after this pass, so it stays crisp.
    if kit.attire == "civil":
        xt = min(nw * 0.72, 0.128)
        xb = min(nw * 0.98, 0.172)
        yt = top + 0.078
        yb = top - 0.006
        band = S.poly([(-xb, yb), (-xt, yt), (0.0, yt - 0.008), (xt, yt),
                       (xb, yb), (0.0, yb + 0.002)], resample=48)
        brush.paint(canvas, band, palette.lighten(hi, 0.055), clip=clip,
                    alpha=1.0, soft=0.45)
        brush.paint(canvas, band & S.box(xb * 1.2, 0.018, 0.0, 0.0, yt - 0.008),
                    shadow, clip=clip, alpha=0.30, soft=0.9)
        brush.paint(canvas, band.outline(lw * 0.48), line, clip=clip,
                    alpha=0.52, soft=0.35)
        brush.paint(canvas, S.lock([(0.0, yb + 0.006), (0.0, yt - 0.006)],
                                   w0=0.0022, w1=0.0022, per_segment=3),
                    line, clip=clip, alpha=0.36, soft=0.35)


def paint_front(canvas, st, h, kit, clip) -> None:
    """Paint the cut's own cloth, beneath the dress code's marks."""
    if clip is None:
        return
    cut = base_cut(kit.attire, kit.garment)
    if cut == "blouse":
        _blouse(canvas, st, h, kit, clip)
    elif cut == "jacket":
        _jacket(canvas, st, h, kit, clip)
    elif cut == "coat":
        _coat(canvas, st, h, kit, clip)
    elif cut == "dress":
        _dress(canvas, st, h, kit, clip)
    elif cut == "robe":
        _robe(canvas, st, h, kit, clip)
    # shirt and tunic are the base passes in detail/regalia; nothing to add.


# --------------------------------------------------------------------------
# overgarments
# --------------------------------------------------------------------------
def _white() -> np.ndarray:
    return palette.oklch_to_srgb((0.94, 0.004, 0.100))


def paint_mantle(canvas, st, h, kit, clip) -> None:
    """A shoulder cape that hides the shoulder line, over the dress code.

    The V opening is cut out of the cape, not painted as a line: the collar and
    the tunic underneath must remain real geometry, and an epaulette under a cape
    must be genuinely covered.  That is why the mantle is painted after the
    insignia and before the awards -- the boards disappear, the star stays on the
    cloth.
    """
    if clip is None:
        return
    if kit.cut not in ("mantle", "habit"):
        return
    top = float(st.neck_y)
    nw = float(st.neck_w)
    sw = float(st.shoulder_w)
    lw = _lw(st)
    cloth, hi, light, shadow, line, accent = _cloth(kit)

    outer_pts = [(-nw * 1.08, top + 0.095), (-sw * 0.36, top + 0.085),
                 (-sw * 0.62, top - 0.020), (-sw * 0.56, top - 0.190),
                 (-sw * 0.28, top - 0.202), (0.0, top - 0.212),
                 (sw * 0.28, top - 0.202), (sw * 0.56, top - 0.190),
                 (sw * 0.62, top - 0.020), (sw * 0.36, top + 0.085),
                 (nw * 1.08, top + 0.095), (0.0, top + 0.140)]
    outer = S.poly(S.smooth_loop(outer_pts, per_segment=8, resample=72))
    hole = S.poly([(-nw * 1.35, top + 0.165), (0.0, top - 0.160),
                   (nw * 1.35, top + 0.165)], resample=34)
    cape = outer - hole

    # Cast shadow on the tunic under the free hem and around the V.
    brush.paint(canvas, cape.offset(0.014).place(cy=-0.014) - cape,
                kit.service_shadow, clip=clip, alpha=0.30, soft=2.0)
    # The mantle is a darker plane than the tunic: a cape is usually a heavier
    # cloth, and at avatar size the value step is what keeps it from merging.
    fill = palette.darken(kit.service, 0.060)
    if kit.attire == "clerical":
        fill = kit.service
    brush.paint(canvas, cape, fill, clip=clip, alpha=1.0, soft=0.65)

    # Broad folds: one lit swell in the middle, two shadows under the shoulders.
    brush.paint(canvas, S.ellipse(sw * 0.22, 0.10, -sw * 0.06, top - 0.090),
                palette.lighten(fill, 0.035), clip=clip, alpha=0.10, soft=5.0)
    for side in (+1, -1):
        a = (side * sw * 0.36, top + 0.050)
        b = (side * sw * 0.42, top - 0.200)
        fold = S.lock([a, ((a[0] + b[0]) * 0.5 + side * 0.030,
                           (a[1] + b[1]) * 0.5), b],
                      w0=0.026, w1=0.010, per_segment=5)
        brush.paint(canvas, fold, shadow, clip=clip, alpha=0.20, soft=3.0)
        brush.paint(canvas, S.lock([(a[0] - side * 0.040, a[1]),
                                    (b[0] - side * 0.030, b[1])],
                                   w0=0.0030, w1=0.0015, per_segment=4),
                    light, clip=clip, alpha=0.14, soft=1.6)

    # The V edges: a lit lining strip, then the hem's braid.  The lining is what
    # says "this is a garment with an inside", not a flat silhouette.
    for side in (+1, -1):
        edge = S.lock([(side * nw * 1.06, top + 0.042),
                       (side * nw * 0.70, top - 0.060),
                       (0.0, top - 0.158)],
                      w0=0.010, w1=0.007, per_segment=6)
        brush.paint(canvas, edge, palette.lighten(fill, 0.060), clip=clip,
                    alpha=0.75, soft=0.5)
        brush.paint(canvas, edge.outline(lw * 0.32), line, clip=clip,
                    alpha=0.42, soft=0.35)
    hem = S.lock([(-sw * 0.55, top - 0.184), (0.0, top - 0.208),
                  (sw * 0.55, top - 0.184)],
                 w0=0.0070, w1=0.0070, bulge=0.004, per_segment=8)
    # A religious habit's hem is a fold of the same cloth, not a gold braid.
    hem_col = palette.lighten(fill, 0.060) if kit.attire == "clerical" else _trim(kit)
    brush.paint(canvas, hem, hem_col, clip=clip, alpha=0.72, soft=0.55)

    # The clasp at the V: a metal disc and a small stone.  It is the one piece of
    # jewellery the cape gets, and it sits exactly where the two halves meet.
    cy = top - 0.166
    disc = S.circle(0.022, 0.0, cy)
    brush.paint(canvas, disc, kit.metal, clip=clip, alpha=0.96, soft=0.35)
    brush.paint(canvas, disc.outline(0.0040), kit.metal_shadow, clip=clip,
                alpha=0.82, soft=0.3)
    brush.paint(canvas, S.circle(0.011, 0.0, cy), kit.accent_hi, clip=clip,
                alpha=0.94, soft=0.3)
    brush.paint(canvas, S.circle(0.004, -0.004, cy + 0.005), kit.metal_hi,
                clip=clip, alpha=0.9, soft=0.25)
    # A short chain from the clasp into the V, so it reads as a fastening.
    for side in (+1, -1):
        brush.paint(canvas, S.lock([(0.0, cy + 0.012),
                                    (side * 0.030, cy + 0.032),
                                    (side * 0.060, cy + 0.045)],
                                   w0=0.0034, w1=0.0022, per_segment=5),
                    kit.metal, clip=clip, alpha=0.78, soft=0.35)

    brush.paint(canvas, cape.outline(lw * 0.58), line, clip=clip,
                alpha=0.52, soft=0.5)


def paint_veil(canvas, st, h, kit, turn: float = 0.0) -> None:
    """The habit's veil and wimple, over the finished hair.

    This is the only painter that runs after the hair: a veil covers the hairstyle,
    which is the whole point of wearing one.  The face is a hole in the cloth, so
    the features are not repainted or dimmed; the wimple is a white band on the
    veil's inner edge and a white panel down the neck.

    The posed path translates the whole veil with the projected face.  That is
    enough for the opt-in three-quarter study, but it is a front-view cloth rather
    than a simulated 3D hood; the passport API always calls it at ``turn = 0``.
    """
    if kit.cut != "habit":
        return
    top = float(st.neck_y)
    top_y = float(st.top_y)
    chin = float(st.chin_y)
    hw = float(st.head_w)
    sw = float(st.shoulder_w)
    lw = _lw(st)
    cloth, hi, light, shadow, line, accent = _cloth(kit)

    # A full veil falls from the crown to the upper chest.  The outer silhouette
    # is deliberately wider than the head: it is cloth, not a swim cap.
    outer_pts = [(-hw * 1.20, chin + 0.10), (-hw * 1.18, top_y - 0.02),
                 (-hw * 0.98, top_y + 0.100), (0.0, top_y + 0.170),
                 (hw * 0.98, top_y + 0.100), (hw * 1.18, top_y - 0.02),
                 (hw * 1.20, chin + 0.10), (hw * 1.06, top + 0.010),
                 (sw * 0.40, top - 0.020), (sw * 0.32, top - 0.060),
                 (0.0, top - 0.085), (-sw * 0.32, top - 0.060),
                 (-sw * 0.40, top - 0.020), (-hw * 1.06, top + 0.010)]
    outer = S.poly(S.smooth_loop(outer_pts, per_segment=8, resample=76))

    # The face opening is face-shaped, not an ellipse: it follows the temples,
    # the cheekbones and the jaw, and it stops just above the brows.  A veil that
    # merely cuts a circle out of the middle leaves the fringe visible, which is
    # the one thing a wimple exists to cover.
    hole_pts = [(-hw * 0.92, 0.080), (-hw * 0.90, -0.020),
                (-hw * 0.82, -0.180), (-hw * 0.70, -0.320),
                (-hw * 0.40, -0.465), (0.0, -0.550),
                (hw * 0.40, -0.465), (hw * 0.70, -0.320),
                (hw * 0.82, -0.180), (hw * 0.90, -0.020),
                (hw * 0.92, 0.080), (0.0, 0.120)]
    hole = S.poly(S.smooth_loop(hole_pts, per_segment=8, resample=64))
    veil = outer - hole

    # A posed head still gets a veil, but the opening follows the projected face
    # instead of sitting dead centre.  The passport path uses turn = 0, where this
    # is exactly the geometric centre.
    dx = float(head.turn_point(0.0, -0.030, turn, st)[0])
    if abs(dx) > 1e-6:
        veil = veil.place(cx=dx)
        hole = hole.place(cx=dx)

    # The shadow the veil casts onto the face, just inside the opening.
    inner = hole - hole.offset(-0.040)
    brush.paint(canvas, inner, h.skin_shadow, alpha=0.20, soft=2.6)

    fill = palette.darken(kit.service, 0.010) if kit.attire != "civil" else h.cloth_shadow
    brush.paint(canvas, veil, fill, alpha=1.0, soft=0.75)

    # Two broad folds fall from the crown; the veil is not a flat black shape.
    for side in (+1, -1):
        a = (side * hw * 0.42 + dx, top_y + 0.04)
        b = (side * hw * 0.92 + dx, chin + 0.04)
        fold = S.lock([a, ((a[0] + b[0]) * 0.5 + side * 0.025,
                           (a[1] + b[1]) * 0.5), b],
                      w0=0.035, w1=0.020, per_segment=5)
        brush.paint(canvas, veil & fold, shadow, alpha=0.18, soft=3.0)
        brush.paint(canvas, veil & S.ellipse(hw * 0.18, 0.16,
                                             -side * hw * 0.62 + dx,
                                             top_y - 0.10),
                    light, alpha=0.055, soft=5.0)

    # The white wimple: a band on the veil around the face, a band across the
    # forehead, and a panel down the neck.  The bands are clipped to the veil (or
    # to the opening for the forehead), so they cannot paint over the features.
    white = _white()
    # A stroke around the opening is smooth; an offset band on a resampled
    # polygon scallops into a row of teeth, which is the last thing a wimple
    # should look like.  The stroke covers the face's edge by half its width,
    # which is exactly what a coif does.
    band = hole.outline(0.026)
    brush.paint(canvas, band, white, alpha=0.96, soft=0.40)
    brush.paint(canvas, band.outline(lw * 0.30), palette.darken(white, 0.14),
                alpha=0.45, soft=0.3)
    # A soft shade just under the band, on the forehead.
    brush.paint(canvas, S.ellipse(hw * 0.70, 0.010, dx, 0.062),
                h.skin_shadow, clip=hole, alpha=0.18, soft=1.8)
    # The neck panel widens towards the shoulders; a square bib reads as paper.
    y0 = chin - 0.020
    y1 = top - 0.010
    pt = hw * 0.30
    pb = hw * 0.40
    # A wimple is a fold of cloth under the chin, not a paper trapezoid.  The
    # bottom is a shallow U so the white does not end in a ruler line across the
    # chest.
    neck = S.poly([(-pt + dx, y0),
                   (-pb * 0.98 + dx, y1 + 0.004),
                   (-pb * 0.55 + dx, y1 - 0.006),
                   (0.0 + dx, y1 - 0.010),
                   (pb * 0.55 + dx, y1 - 0.006),
                   (pb * 0.98 + dx, y1 + 0.004),
                   (pt + dx, y0)], resample=48)
    brush.paint(canvas, neck, white, alpha=0.95, soft=0.9)
    brush.paint(canvas, S.ellipse(pt * 1.18, 0.022, dx, y0 + 0.020),
                palette.darken(white, 0.12), alpha=0.32, soft=1.6)
    for side in (+1, -1):
        fold = S.lock([(side * pt * 0.55 + dx, y0 + 0.010),
                       (side * pb * 0.78 + dx, y1 - 0.004)],
                      w0=0.0040, w1=0.0055, per_segment=4)
        brush.paint(canvas, neck & fold, palette.darken(white, 0.10),
                    alpha=0.42, soft=1.4)
        lit = S.lock([(side * pt * 0.30 + dx, y0 + 0.024),
                      (side * pb * 0.56 + dx, y1 - 0.002)],
                     w0=0.0028, w1=0.0038, per_segment=4)
        brush.paint(canvas, neck & lit, white, alpha=0.18, soft=1.2)
    brush.paint(canvas, S.lock([(-pb * 0.92 + dx, y1 + 0.004),
                                (0.0, y1 - 0.008), (pb * 0.92 + dx, y1 + 0.004)],
                               w0=0.0045, w1=0.0045, per_segment=6),
                palette.darken(white, 0.12), alpha=0.32, soft=0.8)
    brush.paint(canvas, neck.outline(lw * 0.30), palette.darken(white, 0.16),
                alpha=0.38, soft=0.5)

    # A lit hem separates the veil from the black mantle underneath it, and the
    # outer edge stays a soft cloth edge rather than an ink line.
    hem = S.lock([(-sw * 0.32, top - 0.058), (0.0, top - 0.083),
                  (sw * 0.32, top - 0.058)],
                 w0=0.0040, w1=0.0040, bulge=0.002, per_segment=7)
    brush.paint(canvas, hem, palette.lighten(fill, 0.060), alpha=0.28, soft=0.8)
    brush.paint(canvas, veil.outline(lw * 0.64), line, alpha=0.42, soft=1.1)
