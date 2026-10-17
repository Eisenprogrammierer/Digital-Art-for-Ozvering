"""Parametric curves: Bezier, Catmull-Rom, Chaikin corner cutting.

Curves exist here so that shapes can be *specified* by a few control points and
then materialised as dense polygons, which :func:`core.sdf.polygon` turns into an
exact signed distance field.  That keeps the whole pipeline resolution-free:
control points -> sampled outline -> exact SDF -> any output size.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "bezier",
    "bezier_quadratic",
    "bezier_cubic",
    "catmull_rom",
    "chaikin",
    "resample_closed",
    "arc_length",
]


def bezier(points, n: int = 48) -> np.ndarray:
    """Sample a Bezier curve of any degree by de Casteljau's algorithm.

    Vectorised over the sample parameter, so an arbitrary control polygon is one
    sequence of weighted averages rather than a special case per degree.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("a Bezier curve needs at least two control points")
    t = np.linspace(0.0, 1.0, n)[:, None, None]
    pts = np.repeat(points[None], n, axis=0)
    for _ in range(1, len(points)):
        pts = (1.0 - t) * pts[:, :-1] + t * pts[:, 1:]
    return pts[:, 0]


def bezier_quadratic(p0, p1, p2, n: int = 32) -> np.ndarray:
    """Quadratic Bezier: one control point, one smooth bend."""
    return bezier([p0, p1, p2], n)


def bezier_cubic(p0, p1, p2, p3, n: int = 32) -> np.ndarray:
    """Cubic Bezier: two control points, an S-curve with tangent control."""
    return bezier([p0, p1, p2, p3], n)


def catmull_rom(points, per_segment: int = 12, closed: bool = False) -> np.ndarray:
    """Interpolating spline through every control point.

    Unlike Bezier, the curve passes through the points, which is what you want
    for rivers, coastlines and any trail generated from a path.  Endpoints are
    handled by reflecting the first and last points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("Catmull-Rom needs at least two points")
    if closed:
        p = np.vstack([pts[-1], pts, pts[0], pts[1]])
    else:
        p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        a = 2.0 * p1
        b = p2 - p0
        c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
        d = -p0 + 3.0 * p1 - 3.0 * p2 + p3
        out.append(0.5 * (a + b * t + c * t * t + d * t**3))
    out.append(p[-2][None])
    return np.vstack(out)


def chaikin(points, iterations: int = 2, closed: bool = True) -> np.ndarray:
    """Chaikin corner cutting: each edge is replaced by two points at 1/4 and 3/4.

    Converges to a quadratic B-spline, so two or three iterations turn a coarse
    polygon into a smooth outline without any control-point bookkeeping.  This is
    the cheapest way to give a hand-written point list soft corners.
    """
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        if closed:
            nxt = np.roll(pts, -1, axis=0)
            q = 0.75 * pts + 0.25 * nxt
            r = 0.25 * pts + 0.75 * nxt
            pts = np.empty((len(pts) * 2, 2), dtype=np.float64)
            pts[0::2] = q
            pts[1::2] = r
        else:
            nxt = pts[1:]
            cur = pts[:-1]
            q = 0.75 * cur + 0.25 * nxt
            r = 0.25 * cur + 0.75 * nxt
            mid = np.empty((len(cur) * 2, 2), dtype=np.float64)
            mid[0::2] = q
            mid[1::2] = r
            pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def arc_length(points, closed: bool = False) -> np.ndarray:
    """Cumulative arc length along a polyline (length ``len(points)``)."""
    pts = np.asarray(points, dtype=np.float64)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if closed:
        s = np.concatenate([s, [s[-1] + np.linalg.norm(pts[0] - pts[-1])]])
    return s


def resample_closed(points, n: int = 24) -> np.ndarray:
    """Resample a closed polygon to ``n`` points spaced evenly by arc length.

    Handy right before building a distance field: even spacing keeps the
    per-edge loop in :func:`core.sdf.polygon` from wasting work on dense clusters
    of near-duplicate vertices.
    """
    pts = np.asarray(points, dtype=np.float64)
    s = arc_length(pts, closed=True)
    total = s[-1]
    if total <= 0.0:
        return pts.copy()
    targets = np.linspace(0.0, total, n, endpoint=False)
    closed_pts = np.vstack([pts, pts[:1]])
    out = np.empty((n, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(s, t, side="right") - 1)
        j = min(max(j, 0), len(pts) - 1)
        seg = s[j + 1] - s[j]
        f = 0.0 if seg <= 0.0 else (t - s[j]) / seg
        out[i] = closed_pts[j] + f * (closed_pts[j + 1] - closed_pts[j])
    return out
