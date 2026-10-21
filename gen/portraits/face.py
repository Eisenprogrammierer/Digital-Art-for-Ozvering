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

from . import brush, head
from . import shape as S
from .shape import Shape

__all__ = ["brows", "paint_brows", "paint_nose", "paint_mouth", "paint_blush",
           "paint_ears", "mouth_interior"]


# --------------------------------------------------------------------------
# brows
# --------------------------------------------------------------------------
def _brow_spines(st, turn: float = 0.0, expr=None):
    """The three control points of each brow, already turned, plus the side."""
    if expr is None:
        from . import expression
        expr = expression.neutral()
    out = []
    for side in (+1, -1):
        dx, hw = st.brow_dx, st.brow_w * 0.5
        y = st.brow_y + expr["brow_dy"]
        inner = (side * (dx - hw * 1.02), y + expr["brow_inner_dy"])
        peak = (side * (dx + hw * 0.10), y + st.brow_arch + expr["brow_arch"])
        outer = (side * (dx + hw), y + st.brow_arch * 0.10)
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


def brows(st, turn: float = 0.0, expr=None) -> list:
    """Both brows as ``Shape``s, in world coordinates, already turned."""
    out = []
    for _, spine in _brow_spines(st, turn, expr):
        thick = st.brow_thick
        out.append(S.lock(spine, w0=0.80 * thick, w1=0.12 * thick, bulge=0.34 * thick,
                          per_segment=7))
    return out


def paint_brows(canvas, st, h, turn: float = 0.0, expr=None, clip=None):
    for (side, spine), brow in zip(_brow_spines(st, turn, expr), brows(st, turn, expr)):
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
def paint_ears(canvas, st, h, turn: float = 0.0, clip: Shape | None = None):
    sk = head.skull(st)
    for side in (+1, -1):
        if turn != 0.0 and sk.receding(side, turn):
            continue  # the receding ear is hidden by the jaw
        ex, ey = head.turn_point(side * st.head_w * 0.80, st.eye_y + 0.015, turn, st)
        ex, ey = float(ex), float(ey)
        r = st.head_w * 0.17 * float(getattr(st, "ear_size", 1.0))
        # A wash of shade gives the ear a plane; the inner curve is the helix.
        brush.paint(canvas, S.ellipse(r * 0.70, r * 1.08, ex, ey),
                    h.skin_shadow, clip=clip, alpha=0.10, soft=2.0)
        inner = S.arc((ex - r * 0.25 * side, ey + r), (ex - r * 0.30 * side, ey - r),
                      -side * r * 0.85, 0.5, 14)
        brush.paint(canvas, S.lock(inner, w0=st.crease_w * 0.5, w1=st.crease_w * 0.2,
                                   bulge=st.crease_w * 1.1, per_segment=4),
                    h.line_soft, clip=clip, alpha=0.85)
        helix = S.arc((ex - r * 0.02 * side, ey + r * 0.72),
                      (ex - r * 0.10 * side, ey - r * 0.30),
                      -side * r * 0.55, 0.5, 12)
        brush.paint(canvas, S.lock(helix, w0=st.crease_w * 0.35, w1=st.crease_w * 0.12,
                                   per_segment=4),
                    h.skin_deep, clip=clip, alpha=0.42)
        lobe = S.ellipse(r * 0.26, r * 0.30, ex + side * r * 0.10, ey - r * 0.86)
        brush.paint(canvas, lobe, h.skin_warm, clip=clip, alpha=0.20, soft=1.4)


# --------------------------------------------------------------------------
# nose
# --------------------------------------------------------------------------
def paint_nose(canvas, st, h, turn: float = 0.0, clip=None):
    """A dot, a wedge or a line -- three traditions, one coordinate."""
    nx, ny = head.turn_point(0.0, st.nose_y, turn, st)
    nx, ny = float(nx), float(ny)
    w, hh = st.nose_w, st.nose_h
    style = st.nose_style
    # Every nose style gets the same bridge: a narrow shadow on the shadow side
    # and a pale plane on the lit side.  Without it a dot or a wedge floats on the
    # cheek instead of sitting on a face.
    bridge_shadow = S.lock([(nx - w * 0.10, ny + hh * 1.75),
                            (nx - w * 0.16, ny + hh * 0.55),
                            (nx - w * 0.10, ny + hh * 0.12)],
                           w0=w * 0.085, w1=w * 0.045, per_segment=5)
    brush.paint(canvas, bridge_shadow, h.skin_shadow, clip=clip, alpha=0.16, soft=1.5)
    bridge_light = S.lock([(nx + w * 0.10, ny + hh * 1.55),
                           (nx + w * 0.14, ny + hh * 0.48),
                           (nx + w * 0.06, ny + hh * 0.12)],
                          w0=w * 0.070, w1=w * 0.040, per_segment=5)
    brush.paint(canvas, bridge_light, h.skin_light, clip=clip, alpha=0.18, soft=1.4)
    # Alar creases: two short arcs where the nose meets the cheek.  They are the
    # detail that makes a dot nose read as a nose rather than a freckle.
    for side in (+1, -1):
        wing = S.lock([(nx + side * w * 0.34, ny + hh * 0.20),
                       (nx + side * w * 0.44, ny - hh * 0.02),
                       (nx + side * w * 0.24, ny - hh * 0.22)],
                      w0=w * 0.045, w1=w * 0.020, per_segment=4)
        brush.paint(canvas, wing, h.skin_shadow, clip=clip, alpha=0.20, soft=0.8)

    if style == "dot":
        brush.paint(canvas, S.circle(w * 0.36, nx, ny), h.skin_deep, clip=clip,
                    alpha=0.55, soft=2.0)
        brush.paint(canvas, S.circle(w * 0.11, nx, ny - hh * 0.10), h.line_soft, clip=clip,
                    alpha=0.75)
        for side in (+1, -1):
            brush.paint(canvas, S.circle(w * 0.075, nx + side * w * 0.26, ny - hh * 0.20),
                        h.line_soft, clip=clip, alpha=0.46, soft=0.8)
        brush.paint(canvas, S.ellipse(w * 0.16, hh * 0.10, nx, ny + hh * 0.26),
                    h.skin_light, clip=clip, alpha=0.42, soft=0.7)
        return

    if style == "line":
        spine = np.array([[nx - w * 0.35, ny + hh * 0.55], [nx + w * 0.30, ny - hh * 0.35]])
        brush.paint(canvas, S.lock(spine, w0=st.lid_w * 0.5, w1=st.lid_w * 0.9,
                                   bulge=st.lid_w * 0.5, per_segment=5),
                    h.line_soft, clip=clip, alpha=0.90)
        brush.paint(canvas, S.ellipse(w * 0.42, hh * 0.34, nx + w * 0.14, ny - hh * 0.36),
                    h.skin_shadow, clip=clip, alpha=0.34, soft=1.8)
        for side in (+1, -1):
            brush.paint(canvas, S.ellipse(w * 0.08, hh * 0.08,
                                          nx + side * w * 0.30, ny - hh * 0.15),
                        h.line_soft, clip=clip, alpha=0.42, soft=0.8)
        brush.paint(canvas, S.ellipse(w * 0.18, hh * 0.10, nx + w * 0.10, ny + hh * 0.20),
                    h.skin_light, clip=clip, alpha=0.45, soft=0.8)
        return

    # "wedge": the default.  A shaded plane, then a short ink along its base.
    wedge = S.poly([[nx - w * 0.62, ny + hh * 0.42],
                    [nx + w * 0.72, ny - hh * 0.06],
                    [nx + w * 0.05, ny - hh * 0.52]], resample=None)
    brush.paint(canvas, wedge, h.skin_shadow, clip=clip, alpha=0.70, soft=0.9)
    base = np.array([[nx - w * 0.30, ny - hh * 0.34], [nx + w * 0.62, ny - hh * 0.20]])
    brush.paint(canvas, S.lock(base, w0=st.lid_w * 0.45, w1=st.lid_w * 0.95,
                               bulge=st.lid_w * 0.5, per_segment=5),
                h.line, clip=clip, alpha=0.72)
    for side in (+1, -1):
        brush.paint(canvas, S.ellipse(w * 0.07, hh * 0.09,
                                      nx + side * w * 0.30, ny - hh * 0.18),
                    h.line_soft, clip=clip, alpha=0.40, soft=0.7)
    brush.paint(canvas, S.ellipse(w * 0.14, hh * 0.09, nx + w * 0.10, ny + hh * 0.24),
                h.skin_light, clip=clip, alpha=0.46, soft=0.8)


# --------------------------------------------------------------------------
# mouth
# --------------------------------------------------------------------------
def mouth_interior(st, turn: float = 0.0, expr=None):
    """The dark of an open mouth plus its teeth band, or ``(None, None)``.

    The opening is the same lens as the closed mouth, pushed down: ``up`` stays
    the smile's arc and only ``down`` grows by ``mouth_open``, so a grin opens
    *downwards* from the lip line instead of ballooning in both directions.  The
    teeth are the top 45 % of that opening -- a whole white interior is what makes
    a grin look like a single tooth.
    """
    if expr is None:
        from . import expression
        expr = expression.neutral()
    open_h = float(expr["mouth_open"])
    if open_h <= 0.0:
        return None, None
    my = st.mouth_y + expr["mouth_dy"] + expr["mouth_curve"] * 0.3
    hw = st.mouth_w * float(expr["mouth_w_s"]) * 0.5
    left = (-hw, my)
    right = (hw, my + expr["mouth_asym"] * 0.5)
    # ``up`` is only half the smile's arc: the other half belongs to the lower
    # lip, and spending the whole arc on the top edge makes a mouth as tall as it
    # is wide -- a dot.
    up = max(st.mouth_curve + expr["mouth_curve"], 0.010) * 0.5
    inner = S.lens(left, right, up, up + open_h, up_bias=0.48, down_bias=0.52)
    teeth = inner - inner.place(cy=-open_h * 0.45)
    return inner, teeth


def paint_mouth(canvas, st, h, turn: float = 0.0, expr=None, clip=None):
    if expr is None:
        from . import expression
        expr = expression.neutral()
    my = st.mouth_y + expr["mouth_dy"]
    mw = st.mouth_w * float(expr["mouth_w_s"])
    hw = mw * 0.5
    curve = st.mouth_curve + expr["mouth_curve"]
    asym = float(expr["mouth_asym"])
    left = np.array([-hw, my + asym * 0.35])
    right = np.array([hw, my - asym * 0.35])
    # The lip line is a *line*, not a band.  It used to be 18 % of the mouth's
    # own width, and then the lock builder added another 105 % of that as bulge;
    # a neutral mouth was therefore a black oval 22 % as wide as the mouth.  That
    # is why every closed mouth read as an open one.  A line at 4.5 % of the
    # mouth width is the same weight as the lid line at this scale.
    line_w = max(mw * 0.045, 0.0025)

    inner, teeth = mouth_interior(st, turn, expr)
    if inner is not None:
        brush.paint(canvas, inner, h.mouth_dark, clip=clip, alpha=0.94)
        if teeth is not None:
            brush.paint(canvas, teeth, h.teeth, clip=clip, alpha=0.92)
            brush.paint(canvas, S.lock(S.arc(left, right, curve, 0.5, 16),
                                       w0=0.006, w1=0.006, per_segment=4),
                        h.lip_shadow, clip=clip, alpha=0.38, soft=0.8)
        # The opening is inked all the way round.  A line *under* it instead
        # doubles the mouth's height and turns a grin into a smudge.
        brush.paint(canvas, inner.outline(line_w * 0.42), h.lip_deep, clip=clip, alpha=0.85)
        tongue = S.ellipse(mw * 0.24, mw * 0.045, 0.0, inner.bbox[1] + mw * 0.035)
        brush.paint(canvas, tongue, h.lip_deep, clip=inner, alpha=0.28, soft=1.4)
    else:
        lip = S.arc(left, right, curve, 0.5, 16)
        # The lower lip gets a volume of its own even on a line mouth: a single
        # stroke is a scratch, a stroke over a soft body is a mouth.  Fullness is
        # a phenotype axis, so the same line can be a thin mouth or a full one
        # without a second mouth style.
        full = float(getattr(st, "lip_full", 0.0))
        body_w = (0.17 + 0.105 * full) * mw
        body_bulge = (0.12 + 0.075 * full) * mw
        body = S.arc(left, right, -abs(curve) - mw * (0.085 + 0.025 * full), 0.5, 14)
        brush.paint(canvas, S.lock(body, w0=body_w, w1=body_w,
                                   bulge=body_bulge, per_segment=4),
                    h.lip, clip=clip, alpha=0.48)
        brush.paint(canvas, S.lock(lip, w0=line_w * 0.40, w1=line_w * 0.18,
                                   bulge=line_w * 0.55, per_segment=4),
                    h.line, clip=clip)
        upper = S.arc(left, right, curve * 0.55, 0.5, 14)
        brush.paint(canvas, S.lock(upper, w0=(0.065 + 0.028 * full) * mw,
                                   w1=(0.065 + 0.028 * full) * mw,
                                   bulge=(0.055 + 0.020 * full) * mw, per_segment=4),
                    h.lip_deep, clip=clip, alpha=0.30)
        brush.paint(canvas, S.lock(body + np.array([0.0, mw * 0.045]),
                                   w0=0.085 * mw, w1=0.085 * mw,
                                   bulge=(0.06 + 0.025 * full) * mw, per_segment=4),
                    h.lip_light, clip=clip, alpha=0.35)
        brush.paint(canvas, S.ellipse(mw * (0.09 + 0.020 * full), mw * 0.035,
                                      left[0] * 0.55, my - mw * 0.015),
                    h.lip_gloss, clip=clip, alpha=0.30, soft=0.8)
        # The two corner dimples are what keep a fuller mouth from reading as a
        # painted oval: the line still has ends.
        for corner in (left, right):
            brush.paint(canvas, S.circle(mw * 0.022, float(corner[0]), float(corner[1])),
                        h.lip_deep, clip=clip, alpha=0.34, soft=0.6)
    # Philtrum: two soft ridges above the middle of the upper lip.  At this size
    # one pale plane is enough; drawing the grooves would be a moustache.
    px, py = head.turn_point(0.0, my + mw * 0.15, turn, st)
    brush.paint(canvas, S.ellipse(mw * 0.055, mw * 0.10, float(px), float(py)),
                h.skin_light, clip=clip, alpha=0.10, soft=1.4)


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
