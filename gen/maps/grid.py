"""A hexagonal grid in axial coordinates, and the topology that comes with it.

The map is a *rectangle* of hexes -- ``cols`` columns by ``rows`` rows, flat to the
top and bottom -- because that is the shape a strategy map has.  A rectangle of
hexes is a rectangle of *offset* coordinates, though, and offset coordinates are
useless for anything but storage: in them a neighbourhood is seven special cases.
So the constructor converts the rectangle once, into axial coordinates ``(q, r)``,
where hexagons are a lattice and everything afterwards is arithmetic.

Three facts about that lattice are used over and over, so they are stated once
here, for pointy-top hexes with y pointing up (north):

* a centre sits at ``x = size*sqrt(3)*(q + r/2)`` and ``y = size*1.5*r``;
* the neighbours are ``(+1,0) (0,+1) (-1,+1) (-1,0) (0,-1) (+1,-1)``, taken
  counter-clockwise, and neighbour ``d`` is reached across edge ``d``;
* corner ``k`` sits at ``60k - 30`` degrees, so corners ``k`` and ``k+1`` bound
  edge ``k``, and the *midpoint* of edge ``k`` is at ``60k``.

The corner ordering is what makes edge ``d`` and corner ``k`` address each other,
which is what :func:`corner_key` is for: it names a corner by the smallest of the
three hexes that meet there, so three hexes that agree about a corner produce one
name, in integers, with no tolerance and no hashing.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "NEIGHBOURS",
    "corner_key",
    "corners",
    "distance",
    "from_pixel",
    "neighbour",
    "neighbours",
    "round_axial",
    "to_pixel",
    "Cell",
    "HexGrid",
]

NEIGHBOURS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))

_SQRT3 = float(np.sqrt(3.0))

_COS = tuple(float(np.cos(np.radians(60.0 * k - 30.0))) for k in range(6))
_SIN = tuple(float(np.sin(np.radians(60.0 * k - 30.0))) for k in range(6))
_ECOS = tuple(float(np.cos(np.radians(60.0 * k))) for k in range(6))
_ESIN = tuple(float(np.sin(np.radians(60.0 * k))) for k in range(6))


def neighbour(q: int, r: int, d: int) -> tuple:
    """The hex across edge ``d``."""
    dq, dr = NEIGHBOURS[d % 6]
    return q + dq, r + dr


def neighbours(q: int, r: int) -> tuple:
    """All six neighbours, counter-clockwise from the east one."""
    return tuple(neighbour(q, r, d) for d in range(6))


def distance(a, b) -> int:
    """Hex distance in steps.

    The cube metric: with the third coordinate ``-(q + r)``, moving one hex
    changes two of the three by one, so the step count is half their total
    change.
    """
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def round_axial(fq, fr):
    """Round a fractional axial coordinate to the nearest hex centre.

    Cubing the coordinate first is what makes this correct: rounding the three
    cube components and then repairing the one that drifted furthest always
    lands in the hex a point is actually in, which simple rounding does not.
    Vectorised, because the rasteriser runs it on every pixel.
    """
    fx = np.asarray(fq, dtype=np.float64)
    fz = np.asarray(fr, dtype=np.float64)
    fy = -fx - fz
    rx, ry, rz = np.round(fx), np.round(fy), np.round(fz)
    dx = np.abs(rx - fx)
    dy = np.abs(ry - fy)
    dz = np.abs(rz - fz)
    bad_x = (dx > dy) & (dx > dz)
    bad_y = ~bad_x & (dy > dz)

    out_x = np.where(bad_x, -ry - rz, rx)
    out_y = np.where(bad_y, -out_x - rz, ry)
    out_z = np.where(~bad_x & ~bad_y, -out_x - out_y, rz)
    return out_x.astype(np.int64), out_z.astype(np.int64)


def to_pixel(q, r, size: float, origin=(0.0, 0.0)) -> tuple:
    """The centre of hex ``(q, r)`` in world coordinates."""
    x = origin[0] + _SQRT3 * size * (q + 0.5 * r)
    y = origin[1] + 1.5 * size * r
    return x, y


def from_pixel(x, y, size: float, origin=(0.0, 0.0)) -> tuple:
    """The *fractional* axial coordinate of a world point (see :func:`round_axial`)."""
    u = np.asarray(x, dtype=np.float64) - origin[0]
    v = np.asarray(y, dtype=np.float64) - origin[1]
    q = (_SQRT3 / 3.0 * u - v / 3.0) / size
    r = (2.0 / 3.0) * v / size
    return q, r


def corners(q: int, r: int, size: float, origin=(0.0, 0.0)) -> tuple:
    """The six corners of hex ``(q, r)``, counter-clockwise from the low one."""
    cx, cy = to_pixel(q, r, size, origin)
    return tuple((cx + size * _COS[k], cy + size * _SIN[k]) for k in range(6))


def corner_key(q: int, r: int, k: int) -> tuple:
    """The canonical name of a corner, as ``(q, r, k)`` of the lowest hex there.

    Three hexes meet at every corner, and each of them can name it -- as corner
    ``k`` of itself, or as one of two other indices of the two neighbours across
    the edges that end there.  Sorting the three candidates and taking the first
    gives one name, so a boundary walk can look up "the corner we are standing
    on" and get the same answer from whichever side it approached.
    """
    k %= 6
    lo = (k - 1) % 6
    dq, dr = NEIGHBOURS[lo]
    dq2, dr2 = NEIGHBOURS[k]
    return min(
        (q, r, k),
        (q + dq, r + dr, (4 + 2 * lo - k) % 6),
        (q + dq2, r + dr2, (k + 4) % 6),
    )


class Cell(NamedTuple):
    """Per-pixel hex membership: which cell, and where in it the pixel fell.

    ``sector`` is the index of the edge the pixel is nearest to by angle, which
    turns the hexagon into six triangles meeting at its centre.  That is what
    makes the relief shading possible without a loop over hexes: inside one
    sector the distance to the boundary is a linear function of the pixel, so the
    whole map can be triangulated with array arithmetic.
    """

    rc: np.ndarray
    ok: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    rho: np.ndarray
    sector: np.ndarray
    size: float

    def edge_distance(self):
        """Signed distance to the hexagon's edge, positive inside.

        The edge of the sector is the line at distance ``apothem`` along the
        sector's direction, so the signed distance is that minus the projection
        of the offset onto the direction -- no polygon test, no loop.
        """
        apothem = self.size * 0.5 * _SQRT3
        ca = np.asarray([_ECOS[k] for k in range(6)])[self.sector]
        sa = np.asarray([_ESIN[k] for k in range(6)])[self.sector]
        return apothem - (self.dx * ca + self.dy * sa)


class HexGrid:
    """A rectangle of hexes, with its neighbours precomputed.

    ``size`` is the circumradius in world units, and the map is centred on the
    origin, so a canvas that covers ``extent`` needs no offset.

    ``nbr`` is the one thing in here that is not a closed-form formula: the index
    of each hex's six neighbours, or ``-1`` where the neighbour is off the map.
    Terrain smoothing, downhill flow and province adjacency all need it, and none
    of them wants to redo the offset-to-axial conversion in a triple loop.
    """

    def __init__(self, cols: int, rows: int, size: float = 16.0):
        if cols < 1 or rows < 1:
            raise ValueError("a grid needs at least one hex")
        self.cols = int(cols)
        self.rows = int(rows)
        self.size = float(size)

        rr = np.arange(self.rows)[:, None]
        cc = np.arange(self.cols)[None, :]

        q = cc - (rr - (rr & 1)) // 2
        self.axes = np.stack([np.broadcast_to(q, (self.rows, self.cols)),
                              np.broadcast_to(rr, (self.rows, self.cols))], axis=-1).astype(np.int64)

        x = _SQRT3 * self.size * (self.axes[..., 0] + 0.5 * self.axes[..., 1])
        y = 1.5 * self.size * self.axes[..., 1]

        x0, x1 = float(x.min()) - 0.5 * _SQRT3 * self.size, float(x.max()) + 0.5 * _SQRT3 * self.size
        y0, y1 = float(y.min()) - self.size, float(y.max()) + self.size
        self.shift = (-0.5 * (x0 + x1), -0.5 * (y0 + y1))
        self.centers = np.stack([x + self.shift[0], y + self.shift[1]], axis=-1)
        self.extent = (x1 - x0, y1 - y0)

        self.nbr = np.full((self.rows, self.cols, 6, 2), -1, dtype=np.int64)
        for d in range(6):
            dq, dr = NEIGHBOURS[d]
            nq = self.axes[..., 0] + dq
            nr = self.axes[..., 1] + dr
            nc = nq + (nr - (nr & 1)) // 2
            ok = (nr >= 0) & (nr < self.rows) & (nc >= 0) & (nc < self.cols)
            self.nbr[..., d, 0] = np.where(ok, nr, -1)
            self.nbr[..., d, 1] = np.where(ok, nc, -1)

    @property
    def count(self) -> int:
        return self.rows * self.cols

    def flat(self):
        """All hexes as an ``(n, 2)`` array of ``(row, col)``."""
        return np.stack(np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing="ij"),
                        axis=-1).reshape(-1, 2)

    def rc_of(self, q: int, r: int):
        """The ``(row, col)`` of an axial hex, or ``None`` if it is off the map."""
        col = q + (r - (r & 1)) // 2
        if 0 <= r < self.rows and 0 <= col < self.cols:
            return int(r), int(col)
        return None

    def corners_of(self, row: int, col: int) -> tuple:
        """The six world corners of a cell, counter-clockwise."""
        q, r = (int(v) for v in self.axes[row, col])
        return corners(q, r, self.size, self.shift)

    @property
    def origin(self) -> tuple:
        """Where axial ``(0, 0)`` landed, in world coordinates."""
        return self.shift

    def cell(self, X, Y) -> Cell:
        """Resolve world points to hexes.

        A hexagon *is* the Voronoi cell of its centre, so the nearest centre is
        the answer; the lattice just gives it directly (cube rounding) instead of
        by search.  Points outside the map round to a hex on the rim and are
        flagged ``ok=False``: a rectangle of hexes has a ragged edge, and the
        picture should show that rather than pretend to a straight one.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        fq, fr = from_pixel(X, Y, self.size, self.shift)
        q, r = round_axial(fq, fr)
        col = q + (r - (r & 1)) // 2
        ok = (r >= 0) & (r < self.rows) & (col >= 0) & (col < self.cols)
        rr = np.clip(r, 0, self.rows - 1)
        cc = np.clip(col, 0, self.cols - 1)
        cx = self.centers[rr, cc, 0]
        cy = self.centers[rr, cc, 1]
        dx = X - cx
        dy = Y - cy
        sector = np.mod(np.round(np.arctan2(dy, dx) * 3.0 / np.pi).astype(np.int64), 6)
        return Cell(np.stack([rr, cc], axis=-1), ok, dx, dy, np.hypot(dx, dy), sector, self.size)

    def centre_of(self, rc) -> tuple:
        """The world centre of a ``(row, col)``, given as a pair or as an array."""
        rc = np.asarray(rc)
        return float(self.centers[rc[..., 0], rc[..., 1], 0]), float(self.centers[rc[..., 0], rc[..., 1], 1])

    def __repr__(self) -> str:
        w, h = self.extent
        return f"HexGrid({self.cols}x{self.rows}, size={self.size:g}, {w:g}x{h:g})"
