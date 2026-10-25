"""Coastlines: the boundary of a set of hexes, as closed loops of corners.

A coastline is not a line drawn over the map, it is the *edge of the land*, so it
has to come out of the land mask rather than out of a curve fit.  The walk here is
the standard one for the boundary of a set of cells -- follow every half-edge
whose other side is outside the set -- and it is worth spelling out why it is
unambiguous on a hexagonal grid, because it is *not* on a square one, where two
cells can meet at a corner and the walk has to choose which way to turn:

    at every corner of a hex grid exactly three hexes meet, and all three are
    pairwise edge-neighbours.  So a corner has at most two boundary half-edges,
    one arriving and one leaving, and there is never a choice to make.

That is the whole reason the walk can be a dictionary lookup instead of a rule
about which way to turn at a pinch point.  The direction is meaningful, too: the
walk is built so that the mask is on its **left**, which makes an island's loop
counter-clockwise and a lake's clockwise.  Nothing here relies on that, but a
renderer that wants to fill the land instead of stroking it will.

Corners are named by :func:`gen.maps.grid.corner_key`, which resolves the three
hexes that meet there to one integer name, so a loop closes exactly on its first
corner rather than almost.
"""
from __future__ import annotations

import numpy as np

from .grid import corner_key

__all__ = ["boundary_loops"]


def boundary_loops(grid, mask) -> list:
    """The boundary of ``mask`` (a ``(rows, cols)`` bool array) as closed loops.

    Each loop is a list of world points in order, with the first point *not*
    repeated at the end: a loop is closed by its own geometry, and a renderer
    that wants a polygon adds the closing segment itself.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (grid.rows, grid.cols):
        raise ValueError(f"mask must be {grid.rows}x{grid.cols}, got {mask.shape}")

    nxt = {}
    pos = {}
    for row, col in grid.flat():
        if not mask[row, col]:
            continue
        q, r = (int(v) for v in grid.axes[row, col])
        pts = grid.corners_of(int(row), int(col))
        for d in range(6):
            nr, nc = (int(v) for v in grid.nbr[row, col, d])
            if nr >= 0 and mask[nr, nc]:
                continue
            src = corner_key(q, r, d)
            dst = corner_key(q, r, (d + 1) % 6)
            if src in nxt:
                raise ValueError(
                    f"two boundary edges leave corner {src}: the mask is not a set "
                    "of hexes, or a neighbour table is wrong")
            nxt[src] = dst
            pos[src] = pts[d]
            pos[dst] = pts[(d + 1) % 6]

    loops = []
    seen = set()
    for start in nxt:
        if start in seen:
            continue
        path = []
        key = start
        while True:
            seen.add(key)
            key = nxt[key]
            path.append(pos[key])
            if key == start:
                break
        loops.append(path)
    return loops


def signed_area(points) -> float:
    """Twice the signed area of a closed polygon, as the shoelace sum.

    Positive for a counter-clockwise loop with y up, which is what the walk
    produces around land and what :func:`boundary_loops` promises.  Handy for a
    test, and handy for a caller that wants to sort loops by size.
    """
    p = np.asarray(points, dtype=np.float64)
    x = p[:, 0]
    y = p[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
