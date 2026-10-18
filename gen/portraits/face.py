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
def brows(st, turn: float = 0.0, expr=None) -> list:
    """Both brows as ``Shape``s, in world coordinates, already turned."""
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
        thick = st.brow_thick
        out.append(S.lock(spine, w0=0.80 * thick, w1=0.12 * thick, bulge=0.34 * thick,
                          per_segment=7))
    return out


def paint_brows(canvas, st, h, turn: float = 0.0, expr=None, clip=None):
    for brow in brows(st, turn, expr):
        brush.paint(canvas, brow, h.brow, clip=clip)


# --------------------------------------------------------------------------
# ears: a curve inside the silhouette, which is exactly how a front view draws them
# --------------------------------------------------------------------------
def paint_ears(canvas, st, h, turn: float = 0.0, clip: Shape | None = None):
    sk = head.skull(st)
    for side in (+1, -1):
        if turn != 0.0 and sk.receding(side, turn):
            continue  # the receding ear is hidden by the jaw
        ex, ey = head.turn_point(side * st.head_w * 0.80, st.eye_y + 0.015, turn, st)
        r = st.head_w * 0.17
        inner = S.arc((ex - r * 0.25 * side, ey + r), (ex - r * 0.30 * side, ey - r),
                      -side * r * 0.85, 0.5, 14)
        brush.paint(canvas, S.lock(inner, w0=st.crease_w * 0.5, w1=st.crease_w * 0.2,
                                   bulge=st.crease_w * 1.1, per_segment=4),
                    h.line_soft, clip=clip, alpha=0.85)


# --------------------------------------------------------------------------
# nose
# --------------------------------------------------------------------------
def paint_nose(canvas, st, h, turn: float = 0.0, clip=None):
    """A dot, a wedge or a line -- three traditions, one coordinate."""
    nx, ny = head.turn_point(0.0, st.nose_y, turn, st)
    nx, ny = float(nx), float(ny)
    w, hh = st.nose_w, st.nose_h
    style = st.nose_style

    if style == "dot":
        brush.paint(canvas, S.circle(w * 0.36, nx, ny), h.skin_deep, clip=clip,
                    alpha=0.55, soft=2.0)
        brush.paint(canvas, S.circle(w * 0.11, nx, ny - hh * 0.10), h.line_soft, clip=clip,
                    alpha=0.75)
        return

    if style == "line":
        spine = np.array([[nx - w * 0.35, ny + hh * 0.55], [nx + w * 0.30, ny - hh * 0.35]])
        brush.paint(canvas, S.lock(spine, w0=st.lid_w * 0.5, w1=st.lid_w * 0.9,
                                   bulge=st.lid_w * 0.5, per_segment=5),
                    h.line_soft, clip=clip, alpha=0.90)
        brush.paint(canvas, S.ellipse(w * 0.42, hh * 0.34, nx + w * 0.14, ny - hh * 0.36),
                    h.skin_shadow, clip=clip, alpha=0.34, soft=1.8)
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
    line_w = max(mw * 0.18, 0.010)

    inner, teeth = mouth_interior(st, turn, expr)
    if inner is not None:
        brush.paint(canvas, inner, h.mouth_dark, clip=clip, alpha=0.94)
        if teeth is not None:
            brush.paint(canvas, teeth, h.teeth, clip=clip, alpha=0.92)
        # The opening is inked all the way round.  A line *under* it instead
        # doubles the mouth's height and turns a grin into a smudge.
        brush.paint(canvas, inner.outline(line_w * 0.42), h.lip_deep, clip=clip, alpha=0.85)
    else:
        lip = S.arc(left, right, curve, 0.5, 16)
        brush.paint(canvas, S.lock(lip, w0=line_w * 0.16, w1=line_w * 0.16,
                                   bulge=line_w * 1.05, per_segment=4),
                    h.line, clip=clip)
        if st.mouth_style == "full":
            body = S.arc(left, right, -abs(curve) - mw * 0.11, 0.5, 14)
            brush.paint(canvas, S.lock(body, w0=0.20 * mw, w1=0.20 * mw,
                                       bulge=0.16 * mw, per_segment=4),
                        h.lip, clip=clip, alpha=0.55)
            brush.paint(canvas, S.lock(body + np.array([0.0, mw * 0.04]),
                                       w0=0.10 * mw, w1=0.10 * mw,
                                       bulge=0.08 * mw, per_segment=4),
                        h.lip_light, clip=clip, alpha=0.35)


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
