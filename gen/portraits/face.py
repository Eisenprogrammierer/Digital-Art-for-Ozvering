"""The rest of the face: brows, ears, nose, mouth and blush.

These are the features that decide whether a face reads as *a person* rather than
as a pair of eyes on a shape, and they are all small.  Three things matter more
than their geometry:

* **The brows move as units and are coloured by the hair.**  An expression that
  moves the eyes but not the brows reads as a bug, so both read the same row of
  :mod:`gen.portraits.expression` -- and both read the *warped* style, so a mood
  that lifts the face lifts the brows with it.
* **The nose is a shadow, not a shape.**  A dot or a small wedge of shade under
  the tip is the whole anime nose; drawing its outline gives a mannequin.
* **The mouth is a line with weight in the middle and points at the corners.**
  The weight profile is the difference between a mouth and a scratch.

Every feature is placed with :func:`gen.portraits.head.turn_point`, so a turned
head slides the nose to one side and foreshortens whatever runs across the face --
the depth model in :mod:`gen.portraits.turn` is the only place a three-quarter view
is coded.
"""
from __future__ import annotations

import numpy as np

from core import palette

from . import brush, head
from . import shape as S
from .shape import Shape

__all__ = ["brows", "paint_brows", "paint_nose", "paint_mouth", "paint_blush",
           "paint_ears", "paint_ear_over_hair", "mouth_interior"]


# --------------------------------------------------------------------------
# brows
# --------------------------------------------------------------------------
def _brow_spines(st, turn: float = 0.0, expr=None, asym=None):
    """The three control points of each brow, already turned, plus the side.

    ``asym`` is the optional humanising deviation: it can lift one brow, give it
    a different arch or length, and move its inner end.  The three points are
    still turned as one spine, so an expression continues to move the whole brow
    rather than bending it around a fixed world coordinate.
    """
    if expr is None:
        from . import expression
        expr = expression.neutral()
    out = []
    for side in (+1, -1):
        b = asym.brow[side] if asym is not None else {
            "dx": 0.0, "dy": 0.0, "arch": 0.0, "sx": 1.0,
        }
        dx = st.brow_dx + b["dx"]
        hw = st.brow_w * 0.5 * b["sx"]
        y = st.brow_y + expr["brow_dy"] + b["dy"]
        inner = (side * (dx - hw * 1.02), y + expr["brow_inner_dy"])
        peak = (side * (dx + hw * 0.10),
                y + st.brow_arch + expr["brow_arch"] + b["arch"])
        outer = (side * (dx + hw), y + st.brow_arch * 0.10 + b["arch"] * 0.30)
        spine = np.array([inner, peak, outer], dtype=np.float64)
        if st.brow_tilt:
            ang = np.radians(side * float(st.brow_tilt))
            ca, sa = np.cos(ang), np.sin(ang)
            rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
            centre = spine.mean(axis=0)
            spine = (spine - centre) @ rot.T + centre
        spine[:, 0], spine[:, 1] = head.turn_point(spine[:, 0], spine[:, 1], turn, st)
        out.append((side, spine))
    return out


def brows(st, turn: float = 0.0, expr=None, asym=None) -> list:
    """Both brows as ``Shape``s, in world coordinates, already turned."""
    out = []
    for _, spine in _brow_spines(st, turn, expr, asym):
        thick = st.brow_thick
        out.append(S.lock(spine, w0=0.80 * thick, w1=0.12 * thick, bulge=0.34 * thick,
                          per_segment=7))
    return out


def paint_brows(canvas, st, h, turn: float = 0.0, expr=None, clip=None, asym=None):
    for (side, spine), brow in zip(_brow_spines(st, turn, expr, asym),
                                   brows(st, turn, expr, asym)):
        brush.paint(canvas, brow, h.brow, clip=clip)
        # Individual hairs, drawn along the brow's own spine.  The solid brow is
        # the mass; these lines are what keep it from reading as a sticker.
        for k in range(7):
            u = (k + 0.5) / 7.0
            if u < 0.5:
                p = spine[0] + (spine[1] - spine[0]) * (u / 0.5)
            else:
                p = spine[1] + (spine[2] - spine[1]) * ((u - 0.5) / 0.5)
            off = (k - 3) * 0.0018
            tip = p + np.array([side * 0.014, 0.004 + off])
            hair = S.lock([p + np.array([0.0, off]), tip], w0=st.brow_thick * 0.075,
                          w1=0.0, per_segment=3)
            colour = h.brow_hi if k % 3 == 0 else h.brow
            brush.paint(canvas, hair, colour, clip=brow, alpha=0.52)


# --------------------------------------------------------------------------
# ears: a curve inside the silhouette, which is exactly how a front view draws them
# --------------------------------------------------------------------------
def paint_ears(canvas, st, h, turn: float = 0.0, clip: Shape | None = None,
               asym=None):
    sk = head.skull(st)
    for side in (+1, -1):
        if turn != 0.0 and sk.receding(side, turn):
            continue  # the receding ear is hidden by the jaw
        ear = asym.ear[side] if asym is not None else {
            "dx": 0.0, "dy": 0.0, "scale": 1.0,
        }
        ex, ey = head.turn_point(side * st.head_w * 0.96, st.eye_y + 0.015, turn, st)
        ex = float(ex) + ear["dx"]
        ey = float(ey) + ear["dy"]
        r = st.head_w * 0.17 * float(getattr(st, "ear_size", 1.0)) * ear["scale"]
        # A wash of shade gives the ear a plane; the inner curve is the helix.
        brush.paint(canvas, S.ellipse(r * 0.70, r * 1.08, ex, ey),
                    h.skin_shadow, clip=clip, alpha=0.10, soft=2.0)
        # The concha: the bowl is what makes the front view read as an ear rather
        # than a pair of parentheses.  It is a soft, warm hollow under the helix.
        brush.paint(canvas, S.ellipse(r * 0.36, r * 0.54,
                                      ex - side * r * 0.10, ey - r * 0.10),
                    h.skin_warm, clip=clip, alpha=0.16, soft=1.3)
        inner = S.arc((ex - r * 0.25 * side, ey + r), (ex - r * 0.30 * side, ey - r),
                      -side * r * 0.85, 0.5, 14)
        brush.paint(canvas, S.lock(inner, w0=st.crease_w * 0.5, w1=st.crease_w * 0.2,
                                   bulge=st.crease_w * 1.1, per_segment=4),
                    h.line_soft, clip=clip, alpha=0.92)
        helix = S.arc((ex - r * 0.02 * side, ey + r * 0.72),
                      (ex - r * 0.10 * side, ey - r * 0.30),
                      -side * r * 0.55, 0.5, 12)
        brush.paint(canvas, S.lock(helix, w0=st.crease_w * 0.35, w1=st.crease_w * 0.12,
                                   per_segment=4),
                    h.skin_deep, clip=clip, alpha=0.50)
        # The antihelix is the second fold, inside the outer rim.  The tragus is
        # the small tab in front of the opening; at portrait size one soft comma
        # is enough, and it is the feature that stops the ear reading as a shell.
        anti = S.arc((ex - side * r * 0.08, ey + r * 0.52),
                     (ex - side * r * 0.14, ey - r * 0.28),
                     -side * r * 0.36, 0.5, 10)
        brush.paint(canvas, S.lock(anti, w0=st.crease_w * 0.22, w1=st.crease_w * 0.08,
                                   per_segment=4),
                    h.skin_shadow, clip=clip, alpha=0.34)
        trabec = S.ellipse(r * 0.11, r * 0.22,
                           ex + side * r * 0.22, ey - r * 0.18)
        brush.paint(canvas, trabec, h.skin_deep, clip=clip, alpha=0.24, soft=0.8)
        lobe = S.ellipse(r * 0.26, r * 0.30, ex + side * r * 0.10, ey - r * 0.86)
        brush.paint(canvas, lobe, h.skin_warm, clip=clip, alpha=0.20, soft=1.4)


def paint_ear_over_hair(canvas, st, h, turn: float = 0.0, clip: Shape | None = None,
                        asym=None):
    """Re-seat an ear in front of short hair.

    A crop, a spiky cut and a ponytail leave the ear exposed.  Painting the ear
    with the rest of the head put it *behind* the cap, so every short style ended
    in a continuous black shell over the temples -- one of the strongest helmet
    cues.  This small pass fills the ear's own skin disc over the finished hair,
    then redraws the helix and the concha and gives the disc a soft contour.
    Long styles never call it: their curtains are meant to cover the ear.
    """
    sk = head.skull(st)
    for side in (+1, -1):
        if turn != 0.0 and sk.receding(side, turn):
            continue
        ear = asym.ear[side] if asym is not None else {
            "dx": 0.0, "dy": 0.0, "scale": 1.0,
        }
        ex, ey = head.turn_point(side * st.head_w * 0.96, st.eye_y + 0.015, turn, st)
        ex = float(ex) + ear["dx"]
        ey = float(ey) + ear["dy"]
        r = st.head_w * 0.17 * float(getattr(st, "ear_size", 1.0)) * ear["scale"]
        disc = S.ellipse(r * 0.78, r * 1.14, ex, ey)
        # No skull clip: the exposed ear is allowed to break the head's own
        # outline.  That small bump is the difference between a haircut and a
        # black helmet, and it is still well inside the portrait frame.
        brush.paint(canvas, disc, h.skin, alpha=1.0, soft=0.30)
        paint_ears(canvas, st, h, turn, clip=disc, asym=asym)
        brush.Surface(canvas, disc).stroke(
            canvas, st.ink_w * st.line * 0.42,
            palette.mix(h.line_soft, h.skin_shadow, 0.30), inset=0.0012,
            alpha=0.70, soft=0.60)


# --------------------------------------------------------------------------
# nose
# --------------------------------------------------------------------------
def paint_nose(canvas, st, h, turn: float = 0.0, clip=None, asym=None):
    """The nose: a volume made of shade, with only a few marks on top.

    A nose is the hardest feature to draw at portrait size because it has almost
    no outline in real light.  The old version drew a dark wedge and a hard ink
    along its base, which read as a scratch or a scar.  This version builds the
    same three shapes a painter would: a bridge with a shadow side and a lit
    side, a rounded tip with a cast shadow underneath, and two alar wings with
    the nostrils tucked between them.  The named styles change how much line the
    nose is allowed to keep -- a button nose keeps none, a straight nose keeps a
    short crease -- never the anatomy.
    """
    nx, ny = head.turn_point(0.0, st.nose_y, turn, st)
    nx, ny = float(nx), float(ny)
    if asym is not None:
        nx += asym.nose_dx
    w = st.nose_w * 1.75 * (asym.nose_sx if asym is not None else 1.0)
    hh = st.nose_h * 1.65
    style = st.nose_style
    top = ny + hh * 1.90
    tip_y = ny + hh * 0.02

    # The bridge.  Two soft locks follow the dorsum: the shadow side carries the
    # plane, the lit side keeps it from becoming a dark stripe.  They fade into
    # the forehead at the top, so the nose grows out of the face instead of
    # starting on it.
    bridge_shadow = S.lock([(nx - w * 0.06, top),
                            (nx - w * 0.14, ny + hh * 0.95),
                            (nx - w * 0.17, tip_y + hh * 0.16)],
                           w0=w * 0.030, w1=w * 0.095, bulge=w * 0.018,
                           per_segment=6)
    brush.paint(canvas, bridge_shadow, h.skin_shadow, clip=clip,
                alpha=0.560, soft=0.74)
    bridge_light = S.lock([(nx + w * 0.07, top),
                           (nx + w * 0.13, ny + hh * 0.90),
                           (nx + w * 0.10, tip_y + hh * 0.14)],
                          w0=w * 0.026, w1=w * 0.070, bulge=w * 0.012,
                          per_segment=6)
    brush.paint(canvas, bridge_light, h.skin_light, clip=clip,
                alpha=0.480, soft=0.78)

    # The tip: a soft ball of light with its own cast shadow below.  A dot nose
    # is the same ball with a rounder, slightly lower shade; a straight nose puts
    # the ball a little higher and lets the bridge carry the length.
    ball_y = tip_y + (hh * 0.045 if style == "dot" else 0.0)
    brush.paint(canvas, S.ellipse(w * 0.330, hh * 0.255, nx + w * 0.035, ball_y),
                h.skin_light, clip=clip, alpha=0.520, soft=0.58)
    brush.paint(canvas, S.ellipse(w * 0.330, hh * 0.125, nx, ny - hh * 0.155),
                h.skin_deep, clip=clip, alpha=0.320, soft=0.55)
    # One small catchlight on the ball.  The bridge highlight is a plane, not a
    # feature; at 384 px the tip needs a value of its own or the whole nose
    # disappears into the cheek's modelling.
    brush.paint(canvas, S.ellipse(w * 0.105, hh * 0.062,
                                  nx + w * 0.075, tip_y + hh * 0.085),
                h.skin_light, clip=clip, alpha=0.660, soft=0.30)

    # Alar wings.  The crease where the wing meets the cheek is the one line a
    # front view really has; it is soft, short and rolls under the ball.
    for side in (+1, -1):
        wing = S.lock([(nx + side * w * 0.12, ny + hh * 0.015),
                       (nx + side * w * 0.285, ny - hh * 0.045),
                       (nx + side * w * 0.315, ny - hh * 0.185)],
                      w0=w * 0.040, w1=w * 0.016, bulge=w * 0.014,
                      per_segment=5)
        brush.paint(canvas, wing, h.skin_shadow, clip=clip,
                    alpha=0.580, soft=0.38)
        # A narrow lit edge on the outside of the wing separates it from the
        # cheek without drawing a second outline.
        rim = S.lock([(nx + side * w * 0.335, ny - hh * 0.02),
                      (nx + side * w * 0.355, ny - hh * 0.15)],
                     w0=w * 0.012, w1=0.0, per_segment=3)
        brush.paint(canvas, rim, h.skin_light, clip=clip, alpha=0.360, soft=0.34)
        # The alar crease is the one true line on a front-view nose.  It is
        # short, follows the wing, and is darker than the wing fill; without it
        # the nostril sits in a vague smudge instead of under a rim of cartilage.
        alar = S.lock([(nx + side * w * 0.205, ny + hh * 0.020),
                       (nx + side * w * 0.315, ny - hh * 0.060),
                       (nx + side * w * 0.345, ny - hh * 0.195)],
                      w0=w * 0.016, w1=w * 0.007, bulge=w * 0.007,
                      per_segment=4)
        brush.paint(canvas, alar, palette.mix(h.line_soft, h.skin_deep, 0.28),
                    clip=clip, alpha=0.700, soft=0.11)

    # A cast shadow under the tip, and a narrow lit columella.  The old shade
    # was so soft that the underside of the nose disappeared into the philtrum;
    # the contact shadow is what makes the tip project from the face.
    brush.paint(canvas, S.ellipse(w * 0.275, hh * 0.095, nx, ny - hh * 0.300),
                h.skin_deep, clip=clip, alpha=0.360, soft=0.56)

    # Nostrils: small tilted openings, not two black dots.  The columella between
    # them is a lit strip and is what makes the underside read as a nose.
    for side in (+1, -1):
        # The wing is a rounded pad of cartilage, not a streak.  A small ellipse
        # under the bridge gives it a volume, and the nostril is cut into the
        # underside rather than sitting on the cheek.
        brush.paint(canvas, S.ellipse(w * 0.255, hh * 0.205,
                                      nx + side * w * 0.255, ny - hh * 0.095),
                    palette.mix(h.skin_shadow, h.skin_deep, 0.30),
                    clip=clip, alpha=0.460, soft=0.48)
        slit = S.lock([(nx + side * w * 0.185, ny - hh * 0.100),
                       (nx + side * w * 0.300, ny - hh * 0.170)],
                      w0=w * 0.085, w1=w * 0.030, per_segment=3)
        brush.paint(canvas, slit, palette.mix(h.mouth_dark, h.skin_shadow, 0.22),
                    clip=clip, alpha=0.940, soft=0.14)
        brush.paint(canvas, S.ellipse(w * 0.042, hh * 0.030,
                                      nx + side * w * 0.255, ny - hh * 0.145),
                    h.mouth_dark, clip=clip, alpha=0.780, soft=0.24)
        # A thin lit rim under the opening separates the nostril from the
        # philtrum; without it the two dark marks become a moustache.
        rim = S.lock([(nx + side * w * 0.190, ny - hh * 0.205),
                      (nx + side * w * 0.300, ny - hh * 0.235)],
                     w0=w * 0.012, w1=w * 0.006, per_segment=3)
        brush.paint(canvas, rim, h.skin_light, clip=clip, alpha=0.300, soft=0.28)
    # The side of the tip turns away from the key.  Two small planes give the
    # ball a left and a right, which a single central shadow cannot do.
    brush.paint(canvas, S.ellipse(w * 0.155, hh * 0.150,
                                  nx + w * 0.160, ny - hh * 0.025),
                h.skin_shadow, clip=clip, alpha=0.280, soft=0.55)
    for side in (+1, -1):
        sill = S.lock([(nx + side * w * 0.115, ny - hh * 0.190),
                       (nx + side * w * 0.252, ny - hh * 0.225)],
                      w0=w * 0.009, w1=w * 0.004, per_segment=3)
        brush.paint(canvas, sill, palette.mix(h.skin_shadow, h.line_soft, 0.18),
                    clip=clip, alpha=0.420, soft=0.16)
    base_shadow = S.lock([(nx - w * 0.285, ny - hh * 0.190),
                          (nx, ny - hh * 0.255),
                          (nx + w * 0.285, ny - hh * 0.190)],
                         w0=w * 0.022, w1=w * 0.010, bulge=w * 0.014,
                         per_segment=5)
    brush.paint(canvas, base_shadow, h.skin_shadow, clip=clip,
                alpha=0.240, soft=0.80)
    brush.paint(canvas, S.lock([(nx, ny - hh * 0.085), (nx, ny - hh * 0.205)],
                               w0=w * 0.011, w1=w * 0.006, per_segment=3),
                h.skin_light, clip=clip, alpha=0.520, soft=0.24)
    # The dark contact under the columella, kept tiny: it is the shadow the nose
    # throws on the philtrum, not a mouth line.
    brush.paint(canvas, S.ellipse(w * 0.17, hh * 0.055, nx, ny - hh * 0.30),
                h.skin_shadow, clip=clip, alpha=0.170, soft=0.75)

    if style == "dot":
        # A button nose is mostly the ball; a small nostril shadow is enough.
        brush.paint(canvas, S.ellipse(w * 0.16, hh * 0.10, nx, ny - hh * 0.02),
                    h.skin_deep, clip=clip, alpha=0.15, soft=1.1)
        return

    if style == "line":
        # One short curved line on the shadow side of the bridge, ending above
        # the wing.  It is the delicate tradition: a nose you could erase.
        spine = np.array([[nx - w * 0.075, ny + hh * 0.72],
                          [nx - w * 0.15, ny + hh * 0.18],
                          [nx - w * 0.05, ny - hh * 0.12]])
        brush.paint(canvas, S.lock(spine, w0=st.lid_w * 0.34,
                                   w1=st.lid_w * 0.12, per_segment=4),
                    h.line_soft, clip=clip, alpha=0.50, soft=0.25)
        return

    # "wedge": the bridge has a plane and the tip keeps a short crease.  The old
    # version's hard triangle and long base line are gone -- this is a shade, and
    # it is clipped to the bridge so it cannot become a scar.
    plane = S.poly([[nx - w * 0.30, ny + hh * 0.42],
                    [nx + w * 0.16, ny - hh * 0.05],
                    [nx - w * 0.10, ny - hh * 0.20]],
                   resample=18)
    brush.paint(canvas, plane & bridge_shadow, h.skin_shadow, clip=clip,
                alpha=0.16, soft=1.6)
    crease = S.lock([(nx - w * 0.16, ny - hh * 0.115),
                     (nx + w * 0.18, ny - hh * 0.105)],
                    w0=st.lid_w * 0.22, w1=st.lid_w * 0.10,
                    bulge=st.lid_w * 0.16, per_segment=4)
    brush.paint(canvas, crease, h.line_soft, clip=clip, alpha=0.52, soft=0.45)


# --------------------------------------------------------------------------
# mouth
# --------------------------------------------------------------------------
def _mouth_landmarks(st, expr, asym=None):
    """The seam's centre, width and corners, including the sitter's own skew."""
    b = asym
    my = float(st.mouth_y) + float(expr["mouth_dy"])
    mw = float(st.mouth_w) * float(expr["mouth_w_s"])
    mx = 0.0
    corner_l = corner_r = 0.0
    if b is not None:
        my += b.mouth_dy
        mw *= b.mouth_w
        mx = b.mouth_dx
        corner_l = b.mouth_corner_l
        corner_r = b.mouth_corner_r
    hw = mw * 0.5
    expr_asym = float(expr["mouth_asym"])
    left = np.array([-hw + mx, my + expr_asym * 0.50 + corner_l])
    right = np.array([hw + mx, my - expr_asym * 0.50 + corner_r])
    return my, mw, mx, left, right


def mouth_interior(st, turn: float = 0.0, expr=None, asym=None):
    """The dark seam of a mouth plus its teeth band.

    A closed mouth is not a drawn line: it is a shallow seam with a dark interior,
    an upper tooth edge and a tongue in shadow.  The old function returned
    ``(None, None)`` for a neutral mouth, which left the lips painted as a shape
    with no mouth inside them -- the "lipstick without a mouth" failure.  The
    neutral opening is now a few percent of the mouth width, and an expression's
    ``mouth_open`` grows it downwards from the same lip line.
    """
    if expr is None:
        from . import expression
        expr = expression.neutral()
    open_h = float(expr["mouth_open"])
    # Even a closed mouth has a seam, but the seam is a crease, not a cavity.  It
    # used to be three percent of the mouth width, which at 384 px is almost
    # exactly the height at which the dark fill reads as parted lips.  A closed
    # mouth gets a hairline; only an expression with ``mouth_open`` gets a lens
    # deep enough to hold teeth and a tongue.
    seam = max(open_h, float(st.mouth_w) * 0.018)
    my, mw, _mx, left, right = _mouth_landmarks(st, expr, asym)
    # The cavity follows the lip line, a little above the painted lip body.
    lift = (st.mouth_curve + expr["mouth_curve"]) * 0.30
    left = left + np.array([0.0, lift])
    right = right + np.array([0.0, lift])
    # ``up`` is only half the smile's arc: the other half belongs to the lower
    # lip, and spending the whole arc on the top edge makes a mouth as tall as it
    # is wide -- a dot.
    up = max(st.mouth_curve + expr["mouth_curve"], 0.006) * 0.42
    inner = S.lens(left, right, up, up + seam, up_bias=0.48, down_bias=0.52)
    # The teeth are a strip inside the cavity, not a white filling for it: a
    # dark border all round is what reads as a mouth rather than a metal line.
    teeth = (inner - inner.place(cy=-seam * 0.62)).offset(-0.0012)
    return inner, teeth


def paint_mouth(canvas, st, h, turn: float = 0.0, expr=None, clip=None,
                asym=None):
    if expr is None:
        from . import expression
        expr = expression.neutral()
    my, mw, mx, left, right = _mouth_landmarks(st, expr, asym)
    curve = st.mouth_curve + expr["mouth_curve"]
    # The lip line is a *line*, not a band.  It used to be 18 % of the mouth's
    # own width, and then the lock builder added another 105 % of that as bulge;
    # a neutral mouth was therefore a black oval 22 % as wide as the mouth.  That
    # is why every closed mouth read as open.  A line at four percent of the
    # mouth width is the same weight as the lid line at this scale.
    line_w = max(mw * 0.040, 0.0022)
    open_mouth = float(expr["mouth_open"]) > 1e-6

    # The lips are a volume *around* the opening, not a replacement for it.
    # Fullness is a phenotype axis, so the same mouth can be thin or full without
    # a second mouth style.  The lower lip is a lit roll; the upper lip is the
    # darker, flatter plane above the seam.  Both are mixed far toward skin: a
    # passport photograph is not wearing lipstick.
    full = float(getattr(st, "lip_full", 0.0))
    body_y = -abs(curve) - mw * (0.094 + 0.022 * full)
    body = S.arc(left, right, body_y, 0.5, 16)
    body_w = (0.165 + 0.085 * full) * mw
    body_bulge = (0.125 + 0.055 * full) * mw
    lower_lip = S.lock(body, w0=body_w, w1=body_w, bulge=body_bulge, per_segment=4)
    # The shadow the lower lip drops on the chin is what gives it thickness; a
    # fill alone reads as a pink smear.
    brush.paint(canvas, S.ellipse(mw * 0.46, mw * 0.042,
                                  mx, my - mw * 0.155),
                h.skin_shadow, clip=clip, alpha=0.160, soft=1.0)
    brush.paint(canvas, lower_lip, palette.mix(h.lip, h.skin, 0.16),
                clip=clip, alpha=0.78, soft=0.14)

    upper = S.arc(left, right, curve * 0.55, 0.5, 16)
    # The upper lip is a plane, not a single pencil stroke.  Its shape is bounded
    # below by the seam and above by the cupid's bow; painting only the bow line
    # left a pale gap where the lip should be, which is why the mouth read as a
    # scratch with a pink lower lip under it.
    span = float(right[0] - left[0])
    seam_pts = S.arc(left, right, curve * 0.58, 0.5, 16)
    top_y = my + max(curve, 0.0) * 0.55 + mw * 0.120
    bow_pts = np.array([
        left,
        [left[0] + span * 0.22, top_y + mw * 0.006],
        [mx - span * 0.11, top_y - mw * 0.004],
        [mx, top_y - mw * 0.007],
        [mx + span * 0.11, top_y - mw * 0.004],
        [left[0] + span * 0.78, top_y + mw * 0.006],
        right,
    ], dtype=np.float64)
    upper_fill = S.poly(np.vstack([bow_pts, seam_pts[::-1][1:-1]]), resample=44)
    brush.paint(canvas, upper_fill,
                palette.mix(h.lip, h.skin, 0.12), clip=clip,
                alpha=0.78, soft=0.16)
    # The vermillion border: a soft edge, not an ink outline.  It separates the
    # lip from the philtrum and survives the pale skin tone better than the fill.
    brush.paint(canvas, S.lock(bow_pts, w0=mw * 0.019, w1=mw * 0.012,
                               bulge=mw * 0.004, per_segment=4),
                h.lip_shadow, clip=clip, alpha=0.72, soft=0.14)
    brush.paint(canvas, S.lock(upper, w0=(0.040 + 0.014 * full) * mw,
                               w1=(0.040 + 0.014 * full) * mw,
                               bulge=(0.032 + 0.010 * full) * mw, per_segment=4),
                palette.mix(h.lip_deep, h.skin, 0.30), clip=clip,
                alpha=0.42, soft=0.20)

    # The Cupid's bow.  It is a second, much lighter line above the lip's seam,
    # with two peaks and a central dip; without it a full upper lip is a smooth
    # arc and reads as a painted crescent.  The line follows the sitter's own
    # corner heights, so it inherits the asymmetry instead of re-imposing it.
    span = float(right[0] - left[0])
    base_y = my + max(curve, 0.0) * 0.55
    peak_y = base_y + mw * 0.112
    dip_y = base_y + mw * 0.094
    bow = np.array([
        left,
        [left[0] + span * 0.22, peak_y],
        [mx - span * 0.11, dip_y + mw * 0.003],
        [mx, dip_y],
        [mx + span * 0.11, dip_y + mw * 0.003],
        [left[0] + span * 0.78, peak_y],
        right,
    ], dtype=np.float64)
    brush.paint(canvas, S.lock(bow, w0=mw * 0.028, w1=mw * 0.015,
                               bulge=mw * 0.007, per_segment=5),
                h.lip_deep, clip=clip, alpha=0.38, soft=0.26)
    # Two short columns under the bow: the upper lip is a rolled surface, not a
    # flat crescent.  They stop above the seam so the line stays single.
    for u in (-0.26, 0.24):
        cx0 = mx + u * mw
        col_top = my + mw * 0.092
        col_bot = my + mw * 0.032
        brush.paint(canvas, S.lock([(cx0, col_top), (cx0 + u * mw * 0.02, col_bot)],
                                   w0=mw * 0.006, w1=mw * 0.003, per_segment=3),
                    h.lip_shadow, clip=clip, alpha=0.16, soft=0.22)

    inner, teeth = mouth_interior(st, turn, expr, asym=asym)
    if open_mouth:
        brush.paint(canvas, inner, h.mouth_dark, clip=clip, alpha=0.92)
        if teeth is not None:
            brush.paint(canvas, teeth, palette.mix(h.teeth, h.mouth_dark, 0.26),
                        clip=clip, alpha=0.68)
            # The shadow under the upper teeth is what gives the opening depth.
            brush.paint(canvas, S.lock(S.arc(left, right, curve, 0.5, 16),
                                       w0=0.0040, w1=0.0040, per_segment=4),
                        h.lip_shadow, clip=inner, alpha=0.50, soft=0.7)
        seam_h = max(float(inner.bbox[3] - inner.bbox[1]), 1e-4)
        tongue = S.ellipse(mw * 0.235, seam_h * 0.30, mx,
                           inner.bbox[1] + seam_h * 0.30)
        # The tongue is a shadow in the cavity, not a pink fill: too much lip
        # colour here closes the mouth back into a painted seam.
        brush.paint(canvas, tongue, palette.mix(h.mouth_dark, h.lip, 0.22),
                    clip=inner, alpha=0.66, soft=0.7)
        # The opening is inked all the way round.  A line *under* it instead
        # doubles the mouth's height and turns a grin into a smudge.
        brush.paint(canvas, inner.outline(line_w * 0.36), h.mouth_dark,
                    clip=clip, alpha=0.82, soft=0.25)
        # A thin wet line on the upper edge of the cavity.  This is the difference
        # between a mouth that has a mucous membrane and a dark decal; it is kept
        # inside the opening so it cannot brighten the lip itself.
        seam_top = S.lock(S.arc(left, right,
                                max(curve, 0.0) * 0.25 + mw * 0.012, 0.5, 14),
                          w0=mw * 0.010, w1=mw * 0.006, per_segment=4)
        brush.paint(canvas, seam_top, h.lip_light, clip=inner,
                    alpha=0.22, soft=0.35)
    else:
        # A closed mouth is a seam in a volume, not a little oval of darkness.
        # The line follows the lip curve and tapers at the corners, so the face
        # still has a mouth without showing an opening that is not there.
        seam = S.lock(S.arc(left, right, curve * 0.58, 0.5, 18),
                      w0=mw * 0.0105, w1=mw * 0.0105,
                      bulge=mw * 0.0075, per_segment=4)
        brush.paint(canvas, seam, h.mouth_dark, clip=clip,
                    alpha=0.95, soft=0.16)
        # The wet edge sits *above* the seam and is much lighter than it.
        wet = S.lock(S.arc(left, right, curve * 0.58 + mw * 0.014, 0.5, 18),
                     w0=mw * 0.0055, w1=mw * 0.0055,
                     bulge=mw * 0.0028, per_segment=3)
        brush.paint(canvas, wet, h.lip_light, clip=clip, alpha=0.25, soft=0.30)
        # The corners are the ends of the mouth, not where a lip fill happens to
        # stop.  A small darkening there closes the shape without drawing a hard
        # parenthesis on either cheek.
        for corner in (left, right):
            brush.paint(canvas, S.ellipse(mw * 0.013, mw * 0.011,
                                          float(corner[0]), float(corner[1])),
                        h.lip_shadow, clip=clip, alpha=0.34, soft=0.45)

    # The lower lip keeps a light plane outside the opening, so the mouth has a
    # body and (when open) the teeth have something to sit against.
    brush.paint(canvas, S.lock(body + np.array([0.0, mw * 0.035]),
                               w0=0.070 * mw, w1=0.070 * mw,
                               bulge=(0.050 + 0.022 * full) * mw, per_segment=4),
                h.lip_light, clip=clip, alpha=0.25, soft=0.30)
    # The lower lip's own lower edge.  Without it the lip fill is a soft pink
    # smudge with no border; with it the mouth has a drawn shape and a wet plane.
    lip_edge = S.arc(left, right,
                     body_y - mw * (0.055 + 0.012 * full), 0.5, 16)
    brush.paint(canvas, S.lock(lip_edge, w0=mw * 0.0170, w1=mw * 0.0170,
                               bulge=mw * 0.0028, per_segment=3),
                h.lip_shadow, clip=clip, alpha=0.72, soft=0.12)
    # Two or three fine vertical creases on the lower lip.  They are almost
    # invisible in a photograph, but at 384 px they are what separates a lip
    # volume from a pink smear, and they follow the mouth's own barrel curve.
    for u in (-0.18, 0.02, 0.20):
        cx0 = mx + u * mw
        cy0 = body_y + mw * 0.018
        crease = S.lock([(cx0, cy0), (cx0 + u * mw * 0.030,
                                      cy0 - mw * 0.038)],
                        w0=mw * 0.0038, w1=mw * 0.0018, per_segment=3)
        brush.paint(canvas, crease, h.lip_shadow, clip=clip,
                    alpha=0.100, soft=0.30)
    gloss_x = mx + float(asym.lean if asym is not None else 0.0) * mw * 0.055
    brush.paint(canvas, S.ellipse(mw * (0.075 + 0.018 * full), mw * 0.022,
                                  gloss_x, my - mw * 0.055),
                h.lip_gloss, clip=clip, alpha=0.22, soft=0.45)
    # The two corner dimples are what keep a fuller mouth from reading as a
    # painted oval: the line still has ends.  The sitter's own corner heights
    # mean one dimple sits a little deeper than the other.
    for corner in (left, right):
        brush.paint(canvas, S.circle(mw * 0.017, float(corner[0]), float(corner[1])),
                    h.lip_deep, clip=clip, alpha=0.30, soft=0.45)
        side = 1.0 if corner[0] > mx else -1.0
        tail = S.lock([(float(corner[0]) - side * mw * 0.006, float(corner[1])),
                       (float(corner[0]) + side * mw * 0.020,
                        float(corner[1]) + mw * 0.028)],
                      w0=mw * 0.0055, w1=mw * 0.0018, per_segment=3)
        brush.paint(canvas, tail, h.lip_shadow, clip=clip,
                    alpha=0.175, soft=0.35)
    # Philtrum: two soft ridges, a pale central plane and the two grooves that
    # separate them.  Drawing only the pale plane was invisible at 384 px; the
    # grooves are what make the upper lip sit under a nose.
    px, py = head.turn_point(0.0, my + mw * 0.15, turn, st)
    brush.paint(canvas, S.ellipse(mw * 0.055, mw * 0.10, float(px), float(py)),
                h.skin_light, clip=clip, alpha=0.10, soft=1.4)
    for side in (+1, -1):
        gx = mx + side * mw * 0.085
        gy = my + mw * 0.18
        groove = S.lock([(gx, gy), (gx + side * mw * 0.018, gy - mw * 0.11)],
                        w0=mw * 0.008, w1=mw * 0.003, per_segment=3)
        brush.paint(canvas, groove, h.skin_shadow, clip=clip,
                    alpha=0.26, soft=0.28)
    if not open_mouth:
        # The seam is the one mark that must survive every pass.  The lower lip
        # highlight used to be painted after it with a quarter opacity and a
        # wide blur, which lifted the dark line back into a pink smudge.  Paint
        # the seam last, with the lip volume already finished underneath.
        final = S.lock(S.arc(left, right, curve * 0.58, 0.5, 18),
                       w0=mw * 0.0450, w1=mw * 0.0450,
                       bulge=mw * 0.0090, per_segment=4)
        seam_ink = palette.mix(h.mouth_dark, h.lip_shadow, 0.05)
        # Two passes: a 2-3 px lock on a 384 px canvas is mostly antialiased edge,
        # so one pass leaves a grey line.  The second pass darkens the core
        # without widening the seam.
        brush.paint(canvas, final, seam_ink, alpha=0.92, soft=0.10)
        brush.paint(canvas, final, seam_ink, alpha=0.82, soft=0.10)
        brush.paint(canvas, S.lock(S.arc(left, right,
                                         curve * 0.58 + mw * 0.014, 0.5, 18),
                                   w0=mw * 0.0150, w1=mw * 0.0150,
                                   bulge=mw * 0.0026, per_segment=3),
                    h.lip_light, clip=clip, alpha=0.22, soft=0.28)
        for corner in (left, right):
            brush.paint(canvas, S.ellipse(mw * 0.014, mw * 0.012,
                                          float(corner[0]), float(corner[1])),
                        h.lip_shadow, clip=clip, alpha=0.40, soft=0.40)


# --------------------------------------------------------------------------
# blush
# --------------------------------------------------------------------------
def paint_blush(canvas, st, h, turn: float = 0.0, expr=None, clip: Shape | None = None,
                marks: bool = False):
    """Two soft pools of colour, optionally hatched the way manga does it."""
    if expr is None:
        from . import expression
        expr = expression.neutral()
    strength = float(st.blush) * float(expr["blush_mul"])
    if strength <= 0.02:
        return
    for side in (+1, -1):
        bx, by = head.turn_point(side * (st.eye_dx + st.eye_w * 0.10), st.blush_y,
                                 turn, st)
        pool = S.ellipse(st.eye_w * 0.50, st.eye_w * 0.25, float(bx), float(by))
        shape = pool if clip is None else (pool & clip)
        brush.paint(canvas, shape, h.blush, alpha=min(strength * 0.62, 0.32), soft=2.6)
        if marks and strength > 0.18:
            for k in (-1, 0, 1):
                x = float(bx) + side * k * st.eye_w * 0.20
                top = float(by) + st.eye_w * 0.24
                mark = S.lock([(x, top), (x - side * st.eye_w * 0.06, float(by) - st.eye_w * 0.20)],
                              w0=st.crease_w * 0.35, w1=st.crease_w * 0.35, per_segment=3)
                brush.paint(canvas, mark, h.blush, alpha=min(strength * 1.5, 0.75))
