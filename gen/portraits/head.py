"""The head: silhouette, bust, and the skull volume that gives lighting a surface.

Every number is in *head units* -- the chin at ``y = -0.5``, the crown at
``y = +0.5`` -- so a style can be re-rendered at any size without rescaling.

The silhouette is a symmetric Catmull-Rom loop through seven control points per
side.  Symmetry is not cosmetic: a face that is two pixels asymmetric reads as a
mistake, while a face that is *perfectly* symmetric and then slightly broken by
hair reads as a drawing.  So the shape is built from one half and mirrored, and
every asymmetry in a portrait comes from the hair, the brows or the expression.

``turn`` leaves the front view, and it is a rotation of a *solid*: the head is a
stack of cross-sections (:mod:`gen.portraits.turn`), so the outline is the support
of a cross-section rather than a squeeze, the features are projected onto the
surface they actually sit on, and the jaw swings because it has a depth of its own.
This module is where a *style* becomes that solid.
"""
from __future__ import annotations

import numpy as np

from . import cel
from . import shape as S
from . import turn as _turn
from .shape import Shape

__all__ = [
    "FRAME",
    "HEAD_Y",
    "TURN_MAX",
    "control_points",
    "skull",
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
#: The largest turn the model is asked for, in radians.  Past it the near eye's
#: outer corner reaches the silhouette, and a turn that far is a profile view.
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


def skull(st) -> _turn.Skull:
    """The head as a solid: the same profile the silhouette is drawn from.

    ``head_d`` is the one number the style adds -- how deep the head is, as a
    multiple of its own half-width -- and the depth *profile* is the model's, so a
    round chibi and a long seinen differ in one parameter and not in eight.
    """
    pts = np.asarray(_half(st), dtype=np.float64)
    order = np.argsort(pts[:, 1], kind="stable")
    return _turn.Skull(pts[order, 1], pts[order, 0], st.head_d)


def control_points(st, turn: float = 0.0) -> np.ndarray:
    """The full closed control loop, symmetric by construction before ``turn``."""
    right = _half(st)
    loop = np.vstack([right, S.mirror_x(np.asarray(right[1:-1], dtype=np.float64))[::-1]])
    return turn_points(loop, turn, st) if turn else loop


def turn_point(x, y, turn: float, st, vol: float = 0.0):
    """Where a front-view point on the skull's surface ends up when the head turns.

    The point is taken to lie *on* the skull at its own height: the depth is the
    cross-section's there, so the middle of the face -- which has the most depth --
    slides furthest, and a point at the temple barely moves at all.  ``vol`` rides
    the shell a hair's breadth out from the surface, which is where the ears and the
    hair's roots sit.
    """
    return skull(st).project(x, y, turn, vol)


def turn_points(pts, turn: float, st) -> np.ndarray:
    """Rotate a silhouette: every point of it is on the outline at its own height."""
    return skull(st).turn_loop(pts, turn)


def silhouette(st, turn: float = 0.0, detail: int = 56) -> Shape:
    """The head as a distance field: one smooth loop, mirrored and resampled."""
    loop = S.smooth_loop(control_points(st, turn), per_segment=6, resample=detail)
    return S.poly(loop)


def volume(st, turn: float = 0.0):
    """The skull as a height field, so cel shading has normals to work with.

    The dome is a stand-in -- a style's surface is not a paraboloid -- but the one
    thing a turn moves that the light can read is where the surface comes nearest
    the camera, and that is the model's to say, so the stand-in's peak goes there
    instead of sitting on the midline.
    """
    return cel.dome(
        rx=st.head_w * 0.99,
        ry=0.62,
        height=0.40,
        cx=skull(st).crest(0.02, turn),
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


def neckline(st, kind: str = "v", w: float = 0.020) -> Shape:
    """The collar: where the cloth opens.  One stroke is the whole outfit.

    The trapezius on its own reads as a body under a coloured sheet -- the line
    where the shoulders meet the neck is anatomy, not tailoring.  A line across
    the cloth at the right depth turns the same silhouette into a garment, and the
    depth is what says *which* garment, so the three kinds are the same two
    endpoints with a different point between them.

    The endpoints are placed *on* the trapezius by interpolating it, not at a
    fixed height: a collar that stops short of the shoulder line floats on the
    chest, and one that runs past it crosses the arm.
    """
    nw = max(float(st.neck_w) * 1.22, 0.16)
    top = float(st.neck_y)
    x0, x1 = nw * 1.55, float(st.shoulder_w) * 0.66
    edge = (top + 0.075) + (x0 - nw) / max(x1 - nw, 1e-6) * (top - (top + 0.075))
    if kind == "crew":
        spine, bulge = [(-x0, top - 0.085), (0.0, top - 0.096), (x0, top - 0.085)], 0.008
    elif kind == "scoop":
        spine, bulge = [(-x0, edge), (0.0, top - 0.150), (x0, edge)], 0.044
    else:
        spine, bulge = [(-x0, edge), (0.0, top - 0.205), (x0, edge)], 0.012
    return S.lock([np.array(p) for p in spine], w0=w, w1=w,
                  bulge=bulge, per_segment=6, resample=26)
