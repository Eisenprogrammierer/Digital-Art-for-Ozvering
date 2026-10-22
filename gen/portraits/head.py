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
    "shoulder_edge_y",
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

    Nine points, and the shape they trace is the whole difference between a head
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
        (st.chin_w * 1.52, st.chin_y + 0.072),
        (st.chin_w * 1.08, st.chin_y + 0.024),
        (st.chin_w * 0.46, st.chin_y + 0.003),
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
    """The neck: a tapered column from inside the skull to the shoulder notch.

    It is painted *before* the bust, so the collar covers its lower end: the
    visible neck is exactly the notch between the shoulders, and no rounded
    bottom edge can appear on the chest.  The sides widen towards the trapezius
    instead of running as a rectangle -- a rectangle at this width reads as a
    head on a stalk, which is the one bust proportion every style has to avoid.
    """
    top = st.chin_y + 0.17
    bottom = st.neck_y - 0.14
    tw = float(st.neck_w) * 0.82
    bw = float(st.neck_w) * 1.16
    pts = [(-tw, top), (-tw * 1.02, top - 0.075), (-bw, bottom),
           (bw, bottom), (tw * 1.02, top - 0.075), (tw, top)]
    return S.poly(S.smooth_loop(pts, per_segment=6, resample=30))


def shoulders(st) -> Shape:
    """The torso: a trapezius slope and a shoulder cap, not a rounded box.

    A box wide enough to leave the frame still reads as a pill, because its own
    corners are visible.  A trapezius is what makes a bust a *body*: the line
    rises from the outer edge up to where the neck emerges, and the neck is only
    as wide as the notch between the shoulders.  The outer end is a rounded
    acromion and the upper arm drops from it; the frame crops the arm well below
    the shoulder, so both ends of the body stay inside the picture instead of
    turning the bust into a full-width band.
    """
    top = float(st.neck_y)
    bottom = float(st.shoulder_y)
    w = float(st.shoulder_w)
    nw = max(float(st.neck_w) * 1.22, 0.16)
    # ``shoulder_slope`` raises or lowers the outer end of the trapezius without
    # widening the body: a high shoulder line is a square silhouette, a low one
    # is a sloped neck into a narrow chest.  Positive drops the outer end.
    slope = float(getattr(st, "shoulder_slope", 0.0))
    outer = top - 0.155 + 0.040 * slope
    crest = top - 0.010 * slope
    # The top edge is a *neckline*, not a shelf.  A straight edge across the
    # shoulders ran over the throat above the collar and read as a shirt seam
    # crossing the sitter's neck.  The edge now meets the collar stand at the
    # sides and dips below the throat at the middle, so the cloth closes around
    # the neck and only the collar's own opening crosses it.
    pts = [
        (-w * 1.02, bottom),
        (-w * 1.03, outer - 0.135),
        (-w * 0.965, outer - 0.030),
        (-w * 0.745, crest),
        (-nw, top + 0.075),
        (0.0, top - 0.020),
        (nw, top + 0.075),
        (w * 0.745, crest),
        (w * 0.965, outer - 0.030),
        (w * 1.03, outer - 0.135),
        (w * 1.02, bottom),
    ]
    return S.poly(S.smooth_loop(pts, per_segment=8, resample=60))


def shoulder_edge_y(st, x: float) -> float:
    """The top edge of the torso at ``x``, measured from the actual shape.

    An epaulette belongs on the shoulder, and the shoulder is the *edge* of the
    bust, not a horizontal line near the chest.  The trapezius is a smooth curve,
    so interpolating its control points is not enough -- a board placed on the
    interpolation can sink into the chest on one style and float on another.
    This walks the real signed field from outside the silhouette down to the
    boundary, which is the line the board has to sit on.
    """
    sh = shoulders(st)
    top = float(st.neck_y)
    lo, hi = top - 0.30, top + 0.18
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if float(sh(float(x), mid)) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


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
    # A collar that only spans the width of the neck is a necklace; a neckline
    # has to cross the trapezius.  Keep the endpoints inside the shoulder slope
    # and let the style's own neck width set the opening.
    x0, x1 = min(nw * 2.9, float(st.shoulder_w) * 0.54), float(st.shoulder_w) * 0.70
    edge = (top + 0.075) + (x0 - nw) / max(x1 - nw, 1e-6) * (top - (top + 0.075))
    if kind == "crew":
        spine, bulge = [(-x0, top - 0.085), (0.0, top - 0.096), (x0, top - 0.085)], 0.008
    elif kind == "scoop":
        spine, bulge = [(-x0, edge), (0.0, top - 0.150), (x0, edge)], 0.044
    else:
        spine, bulge = [(-x0, edge), (0.0, top - 0.205), (x0, edge)], 0.012
    return S.lock([np.array(p) for p in spine], w0=w, w1=w,
                  bulge=bulge, per_segment=6, resample=26)
