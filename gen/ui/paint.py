"""Painting a panel: five passes, one light, one height field per pass.

The layer order is the drawing, and it is short enough to hold in the head:

    shadow   the silhouette's own coverage, blurred and dropped
    body     the gradient fill, *under* everything else
    band     the frame, shaded as a rounded moulding (or knocked in, pressed)
    rule     the inner line, a translucent ink stroke
    boss     the corner ornament, a small dome with a hairline outline

Two rules are deliberately absolute.  First, **a state is a colour, never a
layout**: ``idle``, ``hover`` and ``pressed`` share one geometry, so a button
cannot jitter as the pointer moves over it, and the silhouette alpha of the three
states is bit-identical.  Second, every pass is scaled in *design* pixels, so the
whole thing re-renders at 2x or 3x with identical proportions.
"""
from __future__ import annotations

import numpy as np

from core import palette, render
from core.canvas import Canvas

from .frame import Panel

#: One light for the whole project -- the same direction as the heraldry and
#: portrait packages use, because that is most of what makes assets read as a set.
LIGHT = (-0.45, 0.65)

#: What a panel throws on the ground.
SHADOW = (0.03, 0.03, 0.05)

STATES = ("idle", "hover", "pressed")
KINDS = ("panel", "frame")

__all__ = ["LIGHT", "SHADOW", "STATES", "KINDS", "draw", "panel", "frame", "button"]


def _tones(spec: dict, state: str) -> tuple:
    """The four colours of one state, lifted or dropped by a fixed amount.

    ``pressed`` also inverts the frame's bevel, which is the whole trick: the
    same distance field, the same moulding width, the ramp turned the other way.
    """
    if state == "hover":
        lift = 0.045
    elif state == "pressed":
        lift = -0.055
    else:
        lift = 0.0
    if lift == 0.0:
        return spec["body"], spec["body_low"], spec["edge"], spec["accent"]
    return (
        palette.lighten(spec["body"], lift),
        palette.lighten(spec["body_low"], lift * 0.6),
        palette.lighten(spec["edge"], lift * 0.7),
        palette.lighten(spec["accent"], lift * 1.1 if state == "hover" else lift * 0.4),
    )


def _window(canvas: Canvas, box, margin: float = 2.0) -> tuple:
    """Pixel bounds ``(x0, y0, x1, y1)`` of a world box on a canvas.

    The corner ornament is inside its corner box by construction, so evaluating
    its field over the whole panel would compute mostly pixels that cannot be
    seen.  This is the same "evaluate where it can be drawn" rule the portrait
    package uses, and the margin covers the anti-aliased edge and the footprint
    the normals need.
    """
    x0, y0, x1, y1 = box
    s = float(canvas.scale)
    cx, cy = canvas.center

    def col(x):
        return (x - cx) * s - 0.5 + canvas.w / 2.0

    def row(y):
        v = (cy - y) if canvas.y_up else (y - cy)
        return v * s - 0.5 + canvas.h / 2.0

    i0 = int(np.floor(min(col(x0), col(x1)) - margin))
    i1 = int(np.ceil(max(col(x0), col(x1)) + margin)) + 1
    j0 = int(np.floor(min(row(y0), row(y1)) - margin))
    j1 = int(np.ceil(max(row(y0), row(y1)) + margin)) + 1
    return (max(i0, 0), max(j0, 0), min(i1, canvas.w), min(j1, canvas.h))


def draw(spec: dict, size, kind: str = "panel", state: str = "idle", scale: float = 1.0) -> Canvas:
    """Render one panel, frame or button face."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, not {state!r}")

    p = Panel(spec, size, scale)
    s = p.spec
    body, body_low, edge, accent = _tones(s, state)
    c = Canvas(p.w, p.h, scale=scale)
    X, Y = c.grid()
    dx = c.dx

    d_outer = p.outer(X, Y)
    a_outer = render.coverage(d_outer, dx)
    d_band = p.body(X, Y)
    a_band = render.coverage(d_band, dx)

    # 1. shadow -- from the silhouette, so it costs nothing to keep consistent.
    #    A hollow frame is shadowed by its *band*: a frame is meant to lie over
    #    content, so its hole has to stay genuinely transparent.
    #    It does not react to the state: what a button throws should not flicker.
    sh = render.drop_shadow(a_band if kind == "frame" else a_outer, scale,
                            offset=(0.0, -2.0), sigma=2.6,
                            opacity=0.40 * (0.35 + 0.65 * s["alpha"]))
    c.over_srgb(SHADOW, sh)

    # 2. body.  A vertical gradient only: anything varying across the panel would
    #    smear when the middle is stretched sideways.  `ramp_map` interpolates in
    #    OKLab, so a dark-to-light body does not sag through grey in the middle.
    if kind == "panel":
        lo = palette.mix(body, body_low, s["gradient"])
        t = (p.hh - Y) / (2.0 * p.hh)  # 0 at the top edge, 1 at the bottom
        c.over(palette.srgb_to_linear(palette.ramp_map([body, lo], t)),
               a_outer * s["alpha"])

    # 3. the frame band, as a rounded moulding with a ridge down its middle.
    c.over(render.shade(edge, d_band, dx, bevel=s["bevel"], light=LIGHT,
                        inset=(state == "pressed")), a_band)

    # 4. the inner rule.
    if p.has_rule():
        a_rule = render.coverage(p.rule(X, Y), dx) * a_outer
        c.over_srgb(s["ink"], a_rule * 0.55)

    # 5. the corner ornaments, clipped to the silhouette so a hollow frame keeps
    #    its rivet on the band and nothing shows through the hole.
    if p.has_boss():
        clip = a_outer if kind == "panel" else a_band
        for i in range(4):
            x0, y0, x1, y1 = _window(c, p.corner_box(i))
            if x1 <= x0 or y1 <= y0:
                continue
            Xw, Yw = X[y0:y1, x0:x1], Y[y0:y1, x0:x1]
            sub = clip[y0:y1, x0:x1]
            d_boss = p.boss(i, Xw, Yw)
            c.over(render.shade(accent, d_boss, dx, bevel=p.boss_radius_at(i) * 0.7,
                                light=LIGHT),
                   render.coverage(d_boss, dx) * sub, x0, y0)
            w = p.boss_outline(i)
            if w > 0.05:
                a_out = render.coverage(np.abs(d_boss) - w, dx) * sub
                c.over_srgb(s["ink"], a_out * 0.45, x0, y0)
    return c


def panel(spec: dict, size, scale: float = 1.0) -> Canvas:
    """A filled panel: the ground a window is built on."""
    return draw(spec, size, kind="panel", scale=scale)


def frame(spec: dict, size, scale: float = 1.0) -> Canvas:
    """A hollow frame: only the band survives, so content shows through."""
    return draw(spec, size, kind="frame", scale=scale)


def button(spec: dict, size, state: str = "idle", scale: float = 1.0) -> Canvas:
    """A button face in one of its states."""
    return draw(spec, size, kind="panel", state=state, scale=scale)
