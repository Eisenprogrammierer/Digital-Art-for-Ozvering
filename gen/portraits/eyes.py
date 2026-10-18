"""Eyes: most of the character, built from a dozen shapes.

An anime eye is not a sphere in a socket, it is a *sigil*.  Reading it as a stack
of flat shapes is what makes it drawable, and the stack is always the same:

1. the **opening** -- two arcs meeting at two corners, the upper much higher and
   peaked past the middle;
2. the **iris**, a tall ellipse deliberately taller than the opening, so the lid
   crops it top and bottom (this crop is the single most "anime" thing about it);
3. a **rim** darker than the iris, giving the eye a wet edge;
4. the **pupil**;
5. the **lid shadow**, a band under the upper lid that falls across the iris;
6. one big and one small **highlight**, painted *over* the lid shadow;
7. the **upper lash**, a tapered lock that runs past the outer corner and flicks;
8. a thin **lower lid** line that stops short of the inner corner, and a **crease**.

Every shape is built once in a local frame -- the eye's centre at the origin, the
corners at ``±eye_w/2`` -- and then placed.  The left eye is the right eye mirrored
(``sx = -1``), so the pair is exactly symmetric without being authored twice, and a
turned head places both of an eye's corners on the skull's surface and lets the
chord it makes decide where the eye lands and how narrow it is.
"""
from __future__ import annotations

import numpy as np

from core import palette

from . import brush, head, ink
from . import shape as S

__all__ = ["shapes", "paint"]

#: A highlight is never pure white; it takes a little of the iris, or it burns.
_HL = 0.10


def shapes(st, expr: dict) -> dict:
    """Local-frame geometry for one eye.  Corners at ``(∓eye_w/2, 0)``."""
    hw = st.eye_w * 0.5
    inner = np.array([-hw, 0.0], dtype=np.float64)
    outer = np.array([hw, 0.0], dtype=np.float64)
    up = max(st.eye_up * expr["eye_up_s"] * (1.0 - 0.55 * expr["squint"]), 0.014)
    down = max(st.eye_down * expr["eye_down_s"] * (1.0 - 0.45 * expr["squint"]) - expr["lower_dy"], 0.012)

    opening = S.lens(inner, outer, up, down, up_bias=0.44, down_bias=0.48)
    icx, icy = 0.0, -0.10 * up
    iris = S.ellipse(st.iris_r, st.iris_r * st.iris_squash, icx, icy)
    iris_v = iris & opening
    rim = iris_v - iris.offset(-st.iris_r * 0.22)
    pupil = S.ellipse(st.pupil_r, st.pupil_r * 1.14, icx, icy + st.iris_r * 0.04)

    dy = float(st.lid_shadow) * (up + down) * 0.60
    lid = opening - opening.place(cy=-dy)

    hl_big = S.circle(st.hl_r, icx - st.iris_r * 0.36, icy + st.iris_r * st.iris_squash * 0.42)
    hl_small = S.circle(st.hl_r * 0.46, icx + st.iris_r * 0.34, icy - st.iris_r * st.iris_squash * 0.48)

    # The lash: the upper lid's own curve, extended past the outer corner into a wing.
    upper = S.arc(inner, outer, up, 0.44, 16)
    wing = outer + np.array([hw * 0.34, up * 0.52], dtype=np.float64)
    lash = S.lock(np.vstack([upper, wing[None]]), w0=0.10 * st.lash * expr["lash_mul"],
                  w1=0.0, bulge=0.95 * st.lash * expr["lash_mul"], per_segment=4)
    spikes = S.lock([outer + np.array([hw * 0.06, up * 0.06]),
                     outer + np.array([hw * 0.52, up * 0.34])],
                    w0=0.45 * st.lash, w1=0.0, per_segment=3)

    low = S.arc(inner, outer, -down, 0.48, 18)
    low_seg = np.ascontiguousarray(low[::-1][: max(4, int(0.62 * len(low)))])
    lid_line = S.lock(low_seg, w0=st.lid_w, w1=0.0, per_segment=4)

    crease = S.lock(S.arc(inner + np.array([hw * 0.10, up * 0.55]),
                          outer + np.array([hw * 0.16, up * 0.40]), up * 0.52, 0.5, 12),
                    w0=st.crease_w * 0.45, w1=st.crease_w * 0.10,
                    bulge=st.crease_w * 0.95, per_segment=4)

    # The socket: a soft pool of shade on the skin above the eye.
    socket = S.ellipse(hw * 1.22, up * 1.30, 0.0, up * 1.35 + st.crease_w)

    closed = S.lock(S.arc(inner, outer, up * 0.62, 0.5, 14),
                    w0=0.22 * st.lash, w1=0.0, bulge=0.88 * st.lash, per_segment=4)

    return dict(
        opening=opening, iris=iris, iris_v=iris_v, rim=rim, pupil=pupil, lid=lid,
        hl_big=hl_big, hl_small=hl_small, lash=lash, spikes=spikes, lid_line=lid_line,
        crease=crease, socket=socket, closed=closed, up=up, down=down, hw=hw,
    )


def _ramp(c_dark, c_mid, c_light, y_top: float, y_bot: float):
    """Vertical iris gradient: dark under the lid, a glow low, mid at the rim."""
    span = max(y_top - y_bot, 1e-6)

    def fn(x, y):
        t = np.clip((y_top - y) / span, 0.0, 1.0)
        return palette.ramp_map([c_dark, c_light, c_mid], t)

    return fn


def paint(canvas, st, h, side: int, turn: float = 0.0, expr=None, clip=None):
    """Draw one eye at its station on the face.  ``side = +1`` is the left eye."""
    if expr is None:
        from . import expression
        expr = expression.neutral()
    closed = side in tuple(expr.get("closed", ()))
    # An eye is a chord on a curved surface, so both its corners are placed and
    # nothing else is scaled: the lid to lid height does not change, because a
    # surface turning away compresses sideways and not downwards.  The far eye
    # narrows hard and the near eye widens a little, which is what the geometry says
    # and -- unlike the constant squeeze this replaces -- what a drawing does.
    hw = st.eye_w * 0.5
    centre = side * st.eye_dx
    x_in, _ = head.turn_point(centre - side * hw, st.eye_y, turn, st)
    x_out, _ = head.turn_point(centre + side * hw, st.eye_y, turn, st)
    cx = 0.5 * (float(x_in) + float(x_out))
    width_scale = (float(x_out) - float(x_in)) / st.eye_w
    kw = dict(cx=cx, cy=float(st.eye_y), sx=width_scale, sy=1.0, rot=side * st.eye_tilt)

    g = shapes(st, expr)
    P = lambda sh: sh.place(**kw)  # noqa: E731

    # Pool of shade on the skin, under the eye's own ink.
    if not closed and h is not None:
        brush.paint(canvas, P(g["socket"]), h.skin_shadow, clip=clip, alpha=0.30, soft=2.2)

    if closed:
        fill = h.lash if h is not None else ink.INK
        brush.paint(canvas, P(g["closed"]), fill, clip=clip)
        brush.paint(canvas, P(g["spikes"]), fill, clip=clip, alpha=0.85)
        return

    brush.paint(canvas, P(g["opening"]), h.sclera, clip=clip)
    iv = P(g["iris_v"])
    y0, y1 = iv.bbox[1], iv.bbox[3]
    brush.paint(canvas, iv, None, clip=clip,
                ramp=_ramp(h.iris_dark, h.iris, h.iris_light, y1, y0))
    brush.paint(canvas, P(g["rim"]), h.iris_rim, clip=clip, alpha=0.85)
    brush.paint(canvas, P(g["pupil"]), h.pupil, clip=clip)
    brush.paint(canvas, P(g["lid"]), h.lid_shadow, clip=clip, alpha=st.lid_shadow_alpha)
    hl = palette.mix((1.0, 1.0, 1.0), h.iris_light, _HL)
    brush.paint(canvas, P(g["hl_big"]) & P(g["opening"]), hl, clip=clip)
    brush.paint(canvas, P(g["hl_small"]) & P(g["opening"]), hl, clip=clip, alpha=0.85)
    brush.paint(canvas, P(g["lid_line"]), h.lash, clip=clip, alpha=0.80)
    brush.paint(canvas, P(g["crease"]), h.line_soft, clip=clip, alpha=0.7)
    brush.paint(canvas, P(g["spikes"]), h.lash, clip=clip, alpha=0.9)
    brush.paint(canvas, P(g["lash"]), h.lash, clip=clip)
