"""A head is a solid, so a turn is a rotation rather than a squeeze.

The first version of the turn lived inside :mod:`gen.portraits.head` and was one
ellipsoid: ``z = sqrt(1 - (x/rx)^2 - (y/ry)^2)``, plus a constant squeeze on
"whichever side was receding".  It slid the nose correctly at the eye line and
lied everywhere else.  The depth came from the same ellipsoid in *y* as in *x*, so
the chin -- half a head below the equator -- had almost none, and a jaw that does
not move when the head turns is the difference between a portrait and a mask.  The
squeeze was worse than an approximation: an ellipsoid's outline does not change
width when it rotates, so the squeeze was a drawing convention dressed up as
geometry -- and it was applied to the side with a negative ``x``, which for a
positive yaw is the side the head turns *towards*.

Here the skull is a stack of horizontal cross-sections, one per height: an ellipse
of half-width ``rx(y)`` -- the same profile the silhouette is drawn from -- and
half-depth ``rz(y)``, centred a little in front of the **ear axis**, which is the
vertical line a head actually pivots about.  The face plane is further from that
axis than the back of the skull is (``FRONT_SHARE``), and the depth tapers towards
the crown and the chin (``DOME_Y``/``DOME_SPAN``).

With the camera on ``+z`` and the yaw ``t`` about the vertical axis:

* a surface point rotates to ``x' = x cos t + z sin t``;
* the outline at a height is the *support* of its cross-section in the view
  direction, ``sqrt((rx cos t)^2 + (rz sin t)^2)``: a round cross-section keeps the
  same width at any angle, and one that is deeper than it is wide gets wider, which
  is what happens to a head seen towards profile;
* the depth crest -- the point of the surface nearest the camera -- slides the
  *other* way from the features, because the face turns away from the camera on the
  side it is looking towards.

Everything else in the portrait is a caller.  The features are projected point by
point (an eye is a chord: both its corners are placed, so it narrows by exactly
what the surface foreshortens and keeps its height, since a plane turning away
compresses it sideways and not downwards).  The hair and the ears ride the surface
with an offset, which is the same ellipse grown by ``vol``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Skull", "FRONT_SHARE"]

FRONT_SHARE = 0.53

DOME_Y = 0.02
DOME_SPAN = 0.52


class Skull:
    """The head as a solid: a half-profile and a depth, and the turn they imply.

    ``ys``/``xs`` are the half-profile the silhouette is drawn from -- ``xs[i]`` is
    the half-width at ``ys[i]``, sorted by height -- and ``depth`` is the half-depth
    at the head's deepest, as a multiple of its own half-width.  One number is
    enough for the whole depth profile because the profile is the model's rather
    than the style's: what a style decides is how *deep* the head is, not that a
    jaw is shallower than a cheekbone.
    """

    __slots__ = ("ys", "xs", "depth", "peak")

    def __init__(self, ys, xs, depth: float):
        self.ys = np.asarray(ys, dtype=np.float64)
        self.xs = np.asarray(xs, dtype=np.float64)
        self.depth = float(depth)
        self.peak = float(np.max(self.xs)) if len(self.xs) else 0.0

    def width(self, y):
        """Half-width at ``y``: the profile, straight off the silhouette."""
        return np.interp(y, self.ys, self.xs, left=0.0, right=0.0)

    def semi_depth(self, y):
        """Half-depth at ``y``, from the ear axis out to the surface."""
        y = np.asarray(y, dtype=np.float64)
        return self.depth * self.peak * np.sqrt(
            np.clip(1.0 - ((y - DOME_Y) / DOME_SPAN) ** 2, 0.0, 1.0))

    def centre(self, y):
        """Where the ear axis sits in the cross-section at ``y``."""
        return (2.0 * FRONT_SHARE - 1.0) * self.semi_depth(y)

    def point(self, x, y, vol: float = 0.0):
        """The 3-D point on the front of the skull under a front-view ``(x, y)``.

        ``vol`` grows the cross-section -- it is how the hair and the ears ride a
        shell around the head -- and a point from beyond the silhouette is pulled to
        the side rather than pushed through it.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rx = self.width(y) + vol
        rz = self.semi_depth(y) + vol
        u = np.clip(np.abs(x) / np.maximum(rx, 1e-9), 0.0, 1.0)
        return x, y, self.centre(y) + rz * np.sqrt(np.maximum(1.0 - u * u, 0.0))

    def project(self, x, y, turn: float, vol: float = 0.0):
        """Rotate a surface point by ``turn`` and drop the depth."""
        t = float(turn)
        if t == 0.0:
            return x, y
        px, py, pz = self.point(x, y, vol)
        return px * np.cos(t) + pz * np.sin(t), py

    def depth_at(self, x, y, turn: float, vol: float = 0.0):
        """How near the camera that point is after the turn (bigger is nearer)."""
        t = float(turn)
        px, _, pz = self.point(x, y, vol)
        return -px * np.sin(t) + pz * np.cos(t)

    def support(self, y, turn: float, vol: float = 0.0):
        """Half the head's image width at ``y``: the cross-section's support.

        The support of an ellipse in the view direction, which is the half-width by
        ``sqrt(cos^2 t + (rz/rx)^2 sin^2 t)``: one at every angle for a round head,
        which is why a turn does not change a chibi's width, and more than one for
        a deep one.
        """
        t = float(turn)
        return np.sqrt((self.width(y) * np.cos(t)) ** 2
                       + (self.semi_depth(y) * np.sin(t)) ** 2) + vol

    def turn_loop(self, pts, turn: float) -> np.ndarray:
        """A front-view control loop as the outline it draws when the head turns.

        Each point of a silhouette *is* the widest point at its height, so a turn
        sends it to the outline at the same height: the shape scales by the support
        and slides by the ear axis's own offset.
        """
        p = np.asarray(pts, dtype=np.float64).copy()
        t = float(turn)
        if t == 0.0:
            return p
        y = p[:, 1]
        rx = self.width(y)
        grow = np.divide(self.support(y, t), rx,
                         out=np.ones_like(rx), where=rx > 1e-9)
        p[:, 0] = p[:, 0] * grow + self.centre(y) * np.sin(t)
        return p

    def receding(self, x, turn: float):
        """Is the surface at ``x`` on the side the head is turning away from?

        A head turns towards the way it looks: the features slide that way, and the
        surface goes back on the same side, so the far cheek and the far eye are the
        ones the nose is pointing at.  It is the opposite of what it looks like, and
        getting it backwards is invisible at a small angle and plain at a large one,
        so it is derived here once rather than assumed in each caller.
        """
        return (np.asarray(x) * float(turn)) > 0.0

    def crest(self, y, turn: float) -> float:
        """Where the surface at ``y`` comes nearest the camera.

        The maximiser of the depth, found by bisection: the depth along a convex
        cross-section is concave, so its derivative changes sign once.  The lighting
        volume is a stand-in for the skull, but this is the one thing a turn moves
        that the light can read, so the stand-in borrows it.
        """
        t = float(turn)
        rx = float(self.width(y))
        if t == 0.0 or rx <= 1e-9:
            return 0.0

        def slope(x):
            u = np.clip(abs(x) / rx, 0.0, 1.0)
            rz = float(self.semi_depth(y))
            dz = -rz * x / (rx * rx * np.sqrt(max(1.0 - u * u, 1e-12)))
            return -np.sin(t) + dz * np.cos(t)

        lo, hi = -rx, rx
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if slope(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
