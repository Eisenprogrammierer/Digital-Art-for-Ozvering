"""The head: silhouette, bust, and the skull volume that gives lighting a surface.

Every number is in *head units* -- the chin at ``y = -0.5``, the crown at
``y = +0.5`` -- so a style can be re-rendered at any size without rescaling.

The silhouette is a symmetric Catmull-Rom loop through seven control points per
side.  Symmetry is not cosmetic: a face that is two pixels asymmetric reads as a
mistake, while a face that is *perfectly* symmetric and then slightly broken by
hair reads as a drawing.  So the shape is built from one half and mirrored, and
every asymmetry in a portrait comes from the hair, the brows or the expression.

``turn`` leaves the front view.  It is a genuine rotation for the *features* --
each surface point is moved by the depth of an ellipsoid at that point, which is
what slides a nose to one side and pulls the far eye towards the centre -- and an
approximation for the silhouette, which additionally narrows on the receding
side.  A true three-quarter needs its own outline template; this reads correctly
without one.
"""
from __future__ import annotations

import numpy as np

from . import cel
from . import shape as S
from .shape import Shape

__all__ = [
    "FRAME",
    "HEAD_Y",
    "TURN_MAX",
    "control_points",
    "turn_point",
    "turn_points",
    "silhouette",
    "volume",
    "neck",
    "shoulders",
    "crown_y",
    "hairline_y",
]

#: Half-extents of the world box a portrait is composed in.  The frame is a
#: *headshot*: the hair's crown clears the top by a hair's breadth, the chin sits
#: two thirds down, and the bust has the rest.  Framing it any looser makes the
#: face small, which is the one thing an avatar cannot afford.
FRAME = (0.78, 0.825)
#: Where the head's centre sits inside that box: above the middle, so the room
#: under the chin goes to the neck and the shoulders.
HEAD_Y = 0.125
#: The largest turn the silhouette approximation stays honest at, in radians.
TURN_MAX = 0.55


def _half(st) -> list:
    """Control points of the character's left half, crown to chin.

    Eight points, and the shape they trace is the whole difference between a head
    and an egg.  A face is widest at the *cheekbone*, just below the eye line, not
    at the temple; from there the profile barely changes for a third of the head,
    turns at the jaw angle, and only then tapers to the chin -- and the taper ends
    in a *rounded* chin, not a point.  Get the widest point too high and every
    style turns into an egg with a spike on the end.
    """
    return [
        (0.0, 0.500),
        (st.top_w, st.top_y),
        (st.head_w, st.wide_y),
        (st.cheek_w, st.cheek_y),
        (st.jaw_w, st.jaw_y),
        (st.chin_w * 1.60, st.chin_y + 0.085),
        (st.chin_w * 1.15, st.chin_y + 0.035),
        (0.0, st.chin_y),
    ]


def control_points(st, turn: float = 0.0) -> np.ndarray:
    """The full closed control loop, symmetric by construction before ``turn``."""
    right = _half(st)
    loop = np.vstack([right, S.mirror_x(np.asarray(right[1:-1], dtype=np.float64))[::-1]])
    return turn_points(loop, turn, st.head_w) if turn else loop


def turn_point(x, y, turn: float, rx: float, ry: float = 0.55):
    """Where a front-view surface point ends up when the head turns.

    ``z = sqrt(1 - (x/rx)^2 - (y/ry)^2)`` is the depth of an ellipsoid at that
    point, so ``x' = x cos t + z rx sin t`` rotates the surface.  Points at the
    horizontal extremes have zero depth and do not move, which is correct: an
    ellipsoid's silhouette does not change width when it rotates.
    """
    t = float(turn)
    if t == 0.0:
        return x, y
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    depth = np.sqrt(np.maximum(1.0 - (x / rx) ** 2 - (y / ry) ** 2, 0.0))
    return x * np.cos(t) + rx * depth * np.sin(t), y


def turn_points(pts, turn: float, rx: float, ry: float = 0.55) -> np.ndarray:
    """Rotate a point array, and narrow whichever side is now receding."""
    p = np.asarray(pts, dtype=np.float64).copy()
    t = float(turn)
    if t == 0.0:
        return p
    p[:, 0], p[:, 1] = turn_point(p[:, 0], p[:, 1], t, rx, ry)
    far = (np.sign(p[:, 0]) != np.sign(t)) & (np.abs(p[:, 0]) > 1e-9)
    p[far, 0] *= 1.0 - 0.30 * abs(np.sin(t))
    return p


def silhouette(st, turn: float = 0.0, detail: int = 56) -> Shape:
    """The head as a distance field: one smooth loop, mirrored and resampled."""
    loop = S.smooth_loop(control_points(st, turn), per_segment=6, resample=detail)
    return S.poly(loop)


def volume(st, turn: float = 0.0):
    """The skull as a height field, so cel shading has normals to work with."""
    return cel.dome(
        rx=st.head_w * 0.99,
        ry=0.62,
        height=0.40,
        cx=0.11 * np.sin(float(turn)) * st.head_w,
        cy=0.02,
    )


def crown_y(st) -> float:
    return 0.500


def hairline_y(st) -> float:
    """The forehead hairline: where the fringe roots and the skull cap starts."""
    return float(getattr(st, "hairline_y", float(st.eye_y) + 0.30))


def neck(st, turn: float = 0.0) -> Shape:
    """The neck: a column from inside the skull down past the shoulder line.

    It is painted *before* the bust, so the collar covers its lower end: the
    visible neck is exactly the notch between the shoulders, and no rounded
    bottom edge can appear on the chest.
    """
    top = st.chin_y + 0.17
    bottom = st.neck_y - 0.14
    return S.box(st.neck_w, (top - bottom) * 0.5, r=0.05, cy=(top + bottom) * 0.5)


def shoulders(st) -> Shape:
    """The torso: a trapezius slope that leaves the frame, not a rounded box.

    A box wide enough to leave the frame still reads as a pill, because its own
    corners are visible.  A trapezius is what makes a bust a *body*: the line
    rises from the outer edge up to where the neck emerges, and the neck is only
    as wide as the notch between the shoulders.
    """
    top = float(st.neck_y)
    bottom = float(st.shoulder_y)
    w = float(st.shoulder_w)
    nw = max(float(st.neck_w) * 1.22, 0.16)
    pts = [
        (-w, bottom),
        (-w, top - 0.22),
        (-w * 0.66, top),
        (-nw, top + 0.075),
        (nw, top + 0.075),
        (w * 0.66, top),
        (w, top - 0.22),
        (w, bottom),
    ]
    return S.poly(S.smooth_loop(pts, per_segment=8, resample=52))
