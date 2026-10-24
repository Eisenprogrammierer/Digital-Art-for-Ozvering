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

__all__ = ["shapes", "paint", "placement"]

#: A highlight is never pure white; it takes a little of the iris, or it burns.
_HL = 0.10


def placement(st, side: int, turn: float = 0.0, asym=None):
    """Screen centre and horizontal scale for one eye on a turned face.

    The old version projected the two *corners* independently onto the skull.
    Near the edge of a wide eye that surface is steep, so the near corner moved
    much further than the far one: measured on the seeded genomes it made the near
    eye 21 % wider and the far eye 43 % narrower at a 27-degree turn -- one huge
    eye and one slit.  An eye lies on the face plane, not over the whole curvature
    of the skull, so its width foreshortens with ``cos(turn)`` and only gains a
    few percent of perspective on the near side.  The centre still comes from the
    solid, which is what keeps both eyes and the nose on one midline.

    ``asym`` is the optional :class:`gen.portraits.humanize.Asym`; it moves one
    eye a fraction of a millimetre and scales it by a percent or two.  It is
    applied here rather than in :func:`shapes` so the same deviation is used by
    the hair's eye clearance and the painter.
    """
    t = float(turn)
    cx, _ = head.turn_point(side * st.eye_dx, st.eye_y, t, st)
    # Both eyes lie on the same face plane, so the only difference between them
    # is the few percent of perspective a real pair has.  A larger coefficient
    # made the far eye read as a different size even before the skull curvature
    # was involved, which is the asymmetry a viewer notices first.
    scale = float(np.cos(t)) * (1.0 - 0.04 * side * np.sin(t))
    if asym is not None:
        e = asym.eye[side]
        cx += e["dx"]
        scale *= e["sx"]
    return float(cx), float(np.clip(scale, 0.82, 1.08))


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
    rim = iris_v - iris.offset(-st.iris_r * 0.17)
    pupil = S.ellipse(st.pupil_r, st.pupil_r * 1.14, icx, icy + st.iris_r * 0.04)

    dy = float(st.lid_shadow) * (up + down) * 0.60
    lid = opening - opening.place(cy=-dy)

    hl_big = S.circle(st.hl_r, icx - st.iris_r * 0.36, icy + st.iris_r * st.iris_squash * 0.42)
    hl_small = S.circle(st.hl_r * 0.46, icx + st.iris_r * 0.34, icy - st.iris_r * st.iris_squash * 0.48)
    hl_tiny = S.circle(st.hl_r * 0.20, icx + st.iris_r * 0.58,
                       icy + st.iris_r * st.iris_squash * 0.18)
    iris_glow = S.ellipse(st.iris_r * 0.50, st.iris_r * 0.20, icx,
                          icy - st.iris_r * st.iris_squash * 0.58)
    pupil_ring = (S.ellipse(st.pupil_r * 1.20, st.pupil_r * 1.34, icx, icy + st.iris_r * 0.04)
                  - pupil)

    # The lash: the upper lid's own curve, extended past the outer corner into a
    # short wing.  The wing used to be half an eye wide, which turns a neutral
    # face into a mascara advert; it is now a flick the width of the outer
    # corner, and the individual hairs below carry the softness.
    upper = S.arc(inner, outer, up, 0.44, 16)
    wing = outer + np.array([hw * 0.255, up * 0.36], dtype=np.float64)
    lash = S.lock(np.vstack([upper, wing[None]]), w0=0.10 * st.lash * expr["lash_mul"],
                  w1=0.0, bulge=0.78 * st.lash * expr["lash_mul"], per_segment=4)
    spikes = S.lock([outer + np.array([hw * 0.04, up * 0.05]),
                     outer + np.array([hw * 0.40, up * 0.24])],
                    w0=0.34 * st.lash, w1=0.0, per_segment=3)

    low = S.arc(inner, outer, -down, 0.48, 18)
    low_seg = np.ascontiguousarray(low[::-1][: max(4, int(0.62 * len(low)))])
    lid_line = S.lock(low_seg, w0=st.lid_w * 0.42, w1=0.0, per_segment=4)
    lower_water = S.lock(low_seg, w0=st.lid_w * 0.55, w1=0.0, per_segment=4)

    crease = S.lock(S.arc(inner + np.array([hw * 0.10, up * 0.55]),
                          outer + np.array([hw * 0.16, up * 0.40]), up * 0.52, 0.5, 12),
                    w0=st.crease_w * 0.45, w1=st.crease_w * 0.10,
                    bulge=st.crease_w * 0.95, per_segment=4)
    crease_soft = S.lock(S.arc(inner + np.array([hw * 0.08, up * 0.66]),
                               outer + np.array([hw * 0.18, up * 0.48]), up * 0.46, 0.5, 12),
                         w0=st.crease_w * 1.5, w1=st.crease_w * 0.4,
                         bulge=st.crease_w * 1.2, per_segment=4)

    # The socket: an elongated orbital pool, not a ring around the eye.  It is
    # wider than the opening and sits above it, so its lower edge dissolves into
    # the cheek instead of drawing a dark circle under the lid.  The fine
    # infraorbital puff is a separate, much softer shape; how much of it shows is
    # a life-stage parameter, because a child has none and an elder has a bag.
    # Keep the orbital pool outside the nasal wall.  The old socket was centred
    # on the eye and reached 1.58 half-widths inward, so on an adult face it
    # crossed the bridge and the socket (painted first) swallowed the nose.  The
    # inner edge is now bounded by the alar width, which leaves the bridge as a
    # lit plane between the two eyes instead of a smudge under the inner corners.
    nasal_wall = float(st.nose_w) * 0.76 + 0.010
    socket_in = float(np.clip(float(st.eye_dx) - nasal_wall, hw * 0.82, hw * 1.22))
    socket_out = hw * 1.74
    socket = S.ellipse((socket_in + socket_out) * 0.5, up * 0.92,
                       (socket_out - socket_in) * 0.5, up * 1.02)
    socket_outer = S.ellipse(hw * 0.72, up * 0.82, hw * 0.58, up * 1.02)
    under_eye = S.ellipse(hw * 0.80, down * 0.70, 0.0, -down * 1.34)
    under_crease = S.lock(S.arc([-hw * 0.56, -down * 1.62],
                                [hw * 0.66, -down * 1.48],
                                -down * 0.34, 0.5, 14),
                          w0=st.lid_w * 0.30, w1=0.0, per_segment=4)
    inner_shade = S.ellipse(hw * 0.22, down * 0.75, -hw * 0.86, -down * 0.08)
    outer_shade = S.ellipse(hw * 0.18, down * 0.55, hw * 0.88, -down * 0.04)
    tear = S.ellipse(hw * 0.075, down * 0.20, -hw * 0.91, -down * 0.12)

    # Separate lash hairs along the same upper arc.  They share the arc's own
    # points, so they grow from the lid instead of floating above it.  Six hairs
    # of unequal length read as a lash line; four equal ones read as a comb.
    lash_fine = []
    for u, ln in ((0.16, 0.015), (0.30, 0.022), (0.44, 0.030),
                  (0.58, 0.032), (0.71, 0.027), (0.84, 0.020)):
        k = int(u * (len(upper) - 1))
        p = upper[k]
        tip = p + np.array([-0.004 + 0.014 * u, ln], dtype=np.float64)
        lash_fine.append(S.lock([p, tip], w0=st.lash * 0.15, w1=0.0,
                                per_segment=3))
    low_lash = []
    for u, ln in ((0.72, -0.010), (0.83, -0.014), (0.92, -0.010)):
        k = int(u * (len(low_seg) - 1))
        p = low_seg[k]
        low_lash.append(S.lock([p, p + np.array([0.007, ln])],
                               w0=st.lash * 0.09, w1=0.0, per_segment=3))

    # Two or three conjunctival vessels, kept inside the opening.  They are not
    # red enough to read as irritation: they are the faint directional grain that
    # separates a living sclera from a white sticker at 384 px.  The fine lines
    # run from the corners towards the iris and fade before they reach it.
    vessels = []
    for root, tip, bend, width in (
        (np.array([-hw * 0.88, -down * 0.10]),
         np.array([-hw * 0.36, -down * 0.42]), down * 0.020, 0.0010),
        (np.array([hw * 0.90, -down * 0.08]),
         np.array([hw * 0.42, -down * 0.46]), down * 0.024, 0.0011),
        (np.array([hw * 0.76, -down * 0.02]),
         np.array([hw * 0.34, -down * 0.24]), down * 0.016, 0.0007),
    ):
        mid = (root + tip) * 0.5 + np.array([bend, -bend * 0.35])
        vessels.append(S.lock([root, mid, tip], w0=width, w1=0.0,
                              per_segment=3))

    closed = S.lock(S.arc(inner, outer, up * 0.62, 0.5, 14),
                    w0=0.22 * st.lash, w1=0.0, bulge=0.88 * st.lash, per_segment=4)
    closed_crease = S.lock(S.arc(inner + np.array([hw * 0.08, up * 0.32]),
                                 outer + np.array([hw * 0.12, up * 0.24]), up * 0.34, 0.5, 12),
                           w0=st.crease_w * 0.55, w1=st.crease_w * 0.15,
                           bulge=st.crease_w * 0.8, per_segment=4)

    return dict(
        opening=opening, iris=iris, iris_v=iris_v, rim=rim, pupil=pupil,
        pupil_ring=pupil_ring, lid=lid,
        hl_big=hl_big, hl_small=hl_small, hl_tiny=hl_tiny, iris_glow=iris_glow,
        lash=lash, spikes=spikes, lid_line=lid_line, lower_water=lower_water,
        crease=crease, crease_soft=crease_soft, socket=socket,
        socket_outer=socket_outer, under_eye=under_eye, under_crease=under_crease,
        inner_shade=inner_shade, outer_shade=outer_shade, tear=tear,
        lash_fine=lash_fine, low_lash=low_lash, vessels=vessels,
        closed=closed, closed_crease=closed_crease, up=up, down=down, hw=hw,
    )


def _ramp(c_dark, c_mid, c_light, y_top: float, y_bot: float):
    """Vertical iris gradient: dark under the lid, a glow low, mid at the rim."""
    span = max(y_top - y_bot, 1e-6)

    def fn(x, y):
        t = np.clip((y_top - y) / span, 0.0, 1.0)
        return palette.ramp_map([c_dark, c_light, c_mid], t)

    return fn


def _fiber_ramp(h, bbox):
    """Radial iris fibres, as a narrow light/dark modulation over the base ramp.

    The eye is a sigil, but a flat sigil reads as a sticker.  A low-alpha ring of
    radial strokes gives the iris a direction -- fibres run from the pupil to the
    limbal ring -- without turning it into a photographic texture.
    """
    x0, y0, x1, y1 = bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    span = max(x1 - x0, y1 - y0, 1e-6)

    def fn(x, y):
        dx = (x - cx) / span
        dy = (y - cy) / span
        angle = np.arctan2(dy, dx)
        radius = np.clip(np.hypot(dx, dy) * 2.0, 0.0, 1.0)
        fibre = 0.5 + 0.5 * np.sin(angle * 37.0 + radius * 16.0)
        return palette.ramp_map([h.iris_dark, h.iris_fiber, h.iris_light], fibre)

    return fn


def paint(canvas, st, h, side: int, turn: float = 0.0, expr=None, clip=None,
          asym=None):
    """Draw one eye at its station on the face.  ``side = +1`` is the left eye.

    ``asym`` is the optional humanising deviation.  It changes the *station* and
    the *small planes* of the eye -- one lid a little lower, one iris a little
    wider, the catchlights not exactly paired.  The opening uses the same
    station, and the hair's eye clearance is built from the same opening, so a
    shifted or scaled eye is cleared by the fringe rather than merely drawn on
    top of it.
    """
    if expr is None:
        from . import expression
        expr = expression.neutral()
    closed = side in tuple(expr.get("closed", ()))
    # An eye lies on the face plane: a turn compresses it sideways by ``cos`` and
    # keeps the lid-to-lid height.  The centre is projected on the solid so it
    # stays on the nose's midline, and the local frame is mirrored for the other
    # side (``sx`` carries the sign).
    cx, width_scale = placement(st, side, turn, asym)
    if asym is None:
        e = {"dx": 0.0, "dy": 0.0, "sx": 1.0, "sy": 1.0, "rot": 0.0,
             "gaze_x": 0.0, "gaze_y": 0.0, "hl_dx": 0.0, "hl_dy": 0.0}
    else:
        e = asym.eye[side]
    kw = dict(cx=cx, cy=float(st.eye_y) + e["dy"],
              sx=side * width_scale, sy=e["sy"],
              rot=side * st.eye_tilt + e["rot"])
    # The iris is a disc on the eyeball, not a decal welded to the lid.  It gets
    # its own frame so a small gaze and an asymmetric catchlight can move it
    # while the opening stays exactly where the lids are.
    gaze_kw = dict(kw, cx=cx + e["gaze_x"], cy=kw["cy"] + e["gaze_y"])
    hl_kw = dict(gaze_kw, cx=gaze_kw["cx"] + e["hl_dx"],
                 cy=gaze_kw["cy"] + e["hl_dy"])

    g = shapes(st, expr)
    P = lambda sh: sh.place(**kw)  # noqa: E731
    G = lambda sh: sh.place(**gaze_kw)  # noqa: E731
    H = lambda sh: sh.place(**hl_kw)  # noqa: E731
    opening = P(g["opening"])

    # Pool of shade on the skin, under the eye's own ink.  The socket is broad
    # and soft; a hard, darker ellipse here is what used to read as a pair of
    # goggles.  The infraorbital puff is painted as light over shadow, so an
    # adult has a living lower lid rather than a second dark ring.
    if not closed and h is not None:
        under = float(np.clip(getattr(st, "under_eye", 0.26), 0.0, 0.9))
        brush.paint(canvas, P(g["socket"]), h.skin_shadow, clip=clip,
                    alpha=0.078, soft=3.6)
        brush.paint(canvas, P(g["socket_outer"]), h.skin_shadow, clip=clip,
                    alpha=0.032, soft=3.0)
        if under > 0.02:
            # A lower lid is a lit pillow with one crease under it, not a second
            # dark ring.  The old warm wash at full age weight read as a bruise;
            # the lift is what makes an elder's eye socket look lived-in rather
            # than injured.
            brush.paint(canvas, P(g["under_eye"]), h.skin_light, clip=clip,
                        alpha=0.060 * under, soft=2.4)
            brush.paint(canvas, P(g["under_crease"]), h.skin_shadow, clip=clip,
                        alpha=0.085 * under, soft=1.2)

    if closed:
        fill = h.lash if h is not None else ink.INK
        brush.paint(canvas, P(g["closed_crease"]), h.skin_shadow, clip=clip,
                    alpha=0.20, soft=1.8)
        brush.paint(canvas, P(g["closed"]), fill, clip=clip)
        brush.paint(canvas, P(g["spikes"]), fill, clip=clip, alpha=0.85)
        for lash in g["lash_fine"]:
            brush.paint(canvas, P(lash), fill, clip=clip, alpha=0.80)
        return

    brush.paint(canvas, opening, palette.mix(h.sclera, h.skin, 0.07), clip=clip)
    brush.paint(canvas, P(g["inner_shade"]) & opening, h.sclera_shadow,
                clip=clip, alpha=0.46, soft=1.4)
    brush.paint(canvas, P(g["outer_shade"]) & opening, h.sclera_shadow,
                clip=clip, alpha=0.38, soft=1.2)
    for vessel in g["vessels"]:
        brush.paint(canvas, P(vessel) & opening,
                    palette.mix(h.sclera_shadow, h.blush, 0.34),
                    clip=clip, alpha=0.16, soft=0.22)
    iv = G(g["iris_v"]) & opening
    y0, y1 = iv.bbox[1], iv.bbox[3]
    brush.paint(canvas, iv, None, clip=clip,
                ramp=_ramp(h.iris_dark, h.iris, h.iris_light, y1, y0))
    brush.paint(canvas, iv, None, clip=clip, alpha=0.22,
                ramp=_fiber_ramp(h, iv.bbox))
    brush.paint(canvas, G(g["rim"]) & opening, h.iris_rim, clip=clip, alpha=0.84)
    brush.paint(canvas, G(g["pupil_ring"]) & opening, h.iris_rim, clip=clip, alpha=0.58)
    brush.paint(canvas, G(g["pupil"]) & opening, h.pupil, clip=clip)
    brush.paint(canvas, P(g["lid"]), h.lid_shadow, clip=clip, alpha=st.lid_shadow_alpha)
    brush.paint(canvas, G(g["iris_glow"]) & opening, h.iris_glow, clip=clip,
                alpha=0.20, soft=1.8)
    hl = palette.mix((1.0, 1.0, 1.0), h.iris_light, _HL)
    brush.paint(canvas, H(g["hl_big"]) & opening, hl, clip=clip)
    brush.paint(canvas, H(g["hl_small"]) & opening, hl, clip=clip, alpha=0.85)
    brush.paint(canvas, H(g["hl_tiny"]) & opening, hl, clip=clip, alpha=0.68)
    brush.paint(canvas, P(g["lower_water"]), h.tear, clip=clip, alpha=0.16, soft=0.7)
    brush.paint(canvas, P(g["lid_line"]), h.lash, clip=clip, alpha=0.42)
    brush.paint(canvas, P(g["tear"]), h.tear, clip=clip, alpha=0.45, soft=0.8)
    brush.paint(canvas, P(g["crease_soft"]), h.skin_shadow, clip=clip, alpha=0.165, soft=1.8)
    brush.paint(canvas, P(g["crease"]), h.line_soft, clip=clip, alpha=0.58)
    for lash in g["lash_fine"]:
        brush.paint(canvas, P(lash), h.lash, clip=clip, alpha=0.78)
    for lash in g["low_lash"]:
        brush.paint(canvas, P(lash), h.lash, clip=clip, alpha=0.36)
    brush.paint(canvas, P(g["spikes"]), h.lash, clip=clip, alpha=0.82)
    brush.paint(canvas, P(g["lash"]), h.lash, clip=clip)
