"""Rivers: fill the pits, drain the land, and draw what comes out.

The interesting part of a river is not its shape, it is where it *goes*, so this
module is a hydrological model first and a drawing second: what it produces is a
flow direction for every cell, from which the drainage area follows, and only then
a polyline.

Two decisions carry it.

**Pits are filled before anything drains.**  A naive "walk to the lowest
neighbour" stops at the first local minimum, and a noise field has thousands of
them: the map fills up with two-hundred one-hex lakes and every river is two hexes
long.  :func:`fill_depressions` is Barnes' priority flood -- grown outwards from
the sea -- and it leaves each cell at the lowest height at which water could
actually get from it to the ocean.  Each step also adds a hair, so the filled
surface is *strictly* descending along any flow path.  That is not cosmetic
either: a filled basin is flat by construction, and a river on a flat has no
downhill neighbour to choose.  The hair turns an arbitrary tie-break into data.

**Drainage accumulates in order of height.**  Sort the cells from high to low and
add each one's total to the cell it drains into; by the time a cell is reached,
everything above it has already been added, so one pass gives the true upstream
area.  That is the whole algorithm -- no recursion, no priority queue, no Dijkstra
-- and it is exact rather than iterative, because flow only ever goes downhill.

The roadmap said "rivers by Dijkstra + Catmull-Rom".  The Catmull-Rom is here and
it is the last thing that happens; the Dijkstra is not, because there is nothing
to search: on a grid where the height is known everywhere, the shortest path to
the sea is the path you get by following the gradient, and the only reason to
reach for a shortest-path algorithm is to work around a pit -- which filling
removes.
"""
from __future__ import annotations

import heapq

import numpy as np

from core import curve

from .grid import HexGrid

__all__ = [
    "RIVER_FRACTION",
    "courses",
    "drainage",
    "downstream",
    "fill_depressions",
    "river_mask",
    "river_threshold",
]

#: What share of the land carries a drawn river.  A *quantile* of the drainage,
#: not a drainage value, for the same reason the sea level is a quantile of
#: elevation: a fixed threshold gives a dense river network on a wet map and an
#: empty one on a dry map, so the same "--rivers" would mean two different things
#: on two different seeds.  Taking the top few per cent of the land instead makes
#: the density of the network a property of the request rather than of the luck.
RIVER_FRACTION = 0.06

#: The hair added at each step of the flood, so that a filled basin still drains
#: one way.  Below anything the eye can see, above anything the arithmetic can
#: lose: the elevation spans [0, 1] in doubles, so 1e-7 is nine digits of headroom.
_HAIR = 1e-7


def fill_depressions(grid: HexGrid, elev, water) -> np.ndarray:
    """The elevation with every pit raised to its spill point (Barnes' flood)."""
    elev = np.asarray(elev, dtype=np.float64)
    water = np.asarray(water, dtype=bool)
    filled = elev.copy()
    done = np.zeros(elev.shape, dtype=bool)
    heap = []
    # The sea is the boundary condition: water is already at sea level and is
    # where everything is trying to get to.  A map with no water at all (a
    # deliberately continental one) is flooded from its single lowest cell
    # instead, so that "downhill" still means something.
    seeds = np.argwhere(water) if water.any() else np.argwhere(elev == elev.min())[:1]
    for r, c in seeds:
        heapq.heappush(heap, (float(filled[r, c]), int(r), int(c)))
        done[r, c] = True
    while heap:
        h, r, c = heapq.heappop(heap)
        for d in range(6):
            nr, nc = (int(v) for v in grid.nbr[r, c, d])
            if nr < 0 or done[nr, nc]:
                continue
            done[nr, nc] = True
            filled[nr, nc] = max(float(filled[nr, nc]), h + _HAIR)
            heapq.heappush(heap, (float(filled[nr, nc]), nr, nc))
    return filled


def downstream(grid: HexGrid, filled, water) -> np.ndarray:
    """The flat index each cell drains into, or ``-1`` for water and for a sump.

    Ties go to the lowest-numbered direction, which is the east one, so a flat
    still drains the same way on every run and on every machine.
    """
    filled = np.asarray(filled, dtype=np.float64)
    water = np.asarray(water, dtype=bool)
    flat = np.arange(grid.count).reshape(grid.rows, grid.cols)
    out = np.full((grid.rows, grid.cols), -1, dtype=np.int64)
    for row, col in grid.flat():
        r, c = int(row), int(col)
        if water[r, c]:
            continue
        best = -1
        best_h = float(filled[r, c])
        for d in range(6):
            nr, nc = (int(v) for v in grid.nbr[r, c, d])
            if nr < 0:
                continue
            h = float(filled[nr, nc])
            if h < best_h - 1e-12:
                best_h = h
                best = int(flat[nr, nc])
        out[r, c] = best
    return out


def drainage(grid: HexGrid, filled, down) -> np.ndarray:
    """The area that drains through each cell, counted in hexes, itself included."""
    acc = np.ones(filled.shape, dtype=np.float64)
    order = np.argsort(-np.asarray(filled, dtype=np.float64).ravel(), kind="stable")
    flat_acc = acc.reshape(-1)
    flat_down = np.asarray(down, dtype=np.int64).reshape(-1)
    for i in order:
        d = int(flat_down[i])
        if d >= 0:
            flat_acc[d] += flat_acc[i]
    return acc


def river_threshold(flux, water, fraction: float = RIVER_FRACTION) -> float:
    """The drainage a cell needs to be a river: a quantile of the land's own flux."""
    land = ~np.asarray(water, dtype=bool)
    lf = np.asarray(flux, dtype=np.float64)[land]
    if lf.size == 0:
        return float("inf")
    return float(np.quantile(lf, min(max(1.0 - float(fraction), 0.0), 1.0)))


def river_mask(flux, water, fraction: float = RIVER_FRACTION):
    """Which land cells carry a river, as a bool mask."""
    flux = np.asarray(flux, dtype=np.float64)
    land = ~np.asarray(water, dtype=bool)
    thr = river_threshold(flux, water, fraction)
    return land & (flux >= thr)


def courses(grid: HexGrid, f: dict, down, flux, fraction: float = RIVER_FRACTION,
            min_cells: int = 3, per_segment: int = 6) -> list:
    """The rivers as smoothed world-space polylines, each with its drainage.

    A course starts where no river feeds it and ends at the coast -- at the last
    *land* cell, not in the middle of the sea hex it empties into, because the
    coast is where the river stops being a river and the coastline stroke is drawn
    over the join anyway.

    A course carries both answers, because they are answers to different questions:
    ``path`` is the hexes the water is in, which is what a game asks, and ``points``
    is the smoothed polyline through them, which is what a map asks.
    """
    water = np.asarray(f["water"], dtype=bool)
    river = river_mask(flux, water, fraction)
    flat = np.arange(grid.count).reshape(grid.rows, grid.cols)

    up = [[] for _ in range(grid.count)]
    for row, col in grid.flat():
        d = int(down[row, col])
        if d >= 0:
            up[d].append(int(flat[row, col]))

    river_flat = river.reshape(-1)
    out = []
    for row, col in grid.flat():
        r, c = int(row), int(col)
        if not river[r, c]:
            continue
        i = int(flat[r, c])
        if any(river_flat[u] for u in up[i]):
            continue  # something else is already flowing into here
        path = []
        walk = []
        j = i
        while True:
            rr, cc = divmod(j, grid.cols)
            if water[rr, cc]:
                break
            walk.append([int(rr), int(cc)])
            path.append((float(grid.centers[rr, cc, 0]), float(grid.centers[rr, cc, 1])))
            d = int(down[rr, cc])
            if d < 0:
                break
            j = d
        if len(path) < min_cells:
            continue
        pts = np.asarray(path, dtype=np.float64)
        if len(pts) >= 2:
            pts = curve.catmull_rom(pts, per_segment=per_segment)
        out.append({
            "points": pts,
            "path": walk,
            "flux": float(flux[r, c]),
            "cells": len(path),
            "mouth": path[-1],
        })
    return out
