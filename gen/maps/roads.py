"""Where the roads go: the cheapest way from one capital to the next.

A road is a claim about the ground before it is a line on it.  "You can march from
Aldmere to Dunfell in four days" is the biome table's costs added up along the way,
and a road drawn as a smooth curve between two dots is a picture of a road rather
than a road.  So this module is a shortest-path search with a painter attached:
Dijkstra over the lattice, one hex step at a time, water impassable, and the cost
of a cell is the cost the biome table already gives it -- the same number a game
would charge an army to move there.

That last part is worth being clear about, because it decides what the roads look
like: **the road is the cheapest walk by the map's own rules**, and not a civil
engineer's road.  A range of hills costs three a hex and a plain costs one, so a
road will cross a narrow range rather than go round a wide valley -- and that is
not a bug, it is the answer the game's own movement costs give.  A road that
disagreed with them would be a road an army could not follow.

The network is a **minimum spanning tree** over the capitals, which is a claim as
well: a spanning tree is the cheapest set of roads that connects every capital, and
every other road is a loop somebody would have to build and then never need.  Roads
that cross without meeting are not a thing on a hex map -- armies move hex to hex --
so the tree is grown over the hex graph itself, on distances that are already path
costs, and not over a straight line between the capitals.

Landmasses do not touch, so what comes out is a spanning *forest*: one tree per
landmass, grown only inside it.  A road across open water would be a bridge nobody
built, which is the same defect as a province in two pieces -- and a game that
wants crossings wants sea routes as their own kind of edge, not as a road drawn on
the sea.

The cells a route goes through are the answer, and the polyline is a rendering of
it: :func:`core.curve.catmull_rom` through the centres of those cells, for the same
reason the rivers are smoothed.  A road bends where the ground says and not at a
right angle.
"""
from __future__ import annotations

import heapq

import numpy as np

from core import curve

from . import terrain
from .grid import HexGrid

__all__ = ["cost_field", "network", "route"]


def cost_field(grid: HexGrid, f: dict) -> np.ndarray:
    """What the map charges to stand on a cell, with water impassable.

    Impassable rather than expensive, and the difference is not academic: a price
    is a price a search will pay, and a road across an ocean is exactly what a very
    expensive ocean buys.  Infinity is not a price.

    The numbers are the biome table's own -- the same ones a game charges an army --
    which is why this is a function of the table and not of a second table kept in
    step with it by hand.
    """
    costs = np.array([b.cost for b in terrain.BIOMES], dtype=np.float64)
    cost = costs[np.asarray(f["biome"])]
    cost[np.asarray(f["water"], dtype=bool)] = np.inf
    return cost


def route(grid: HexGrid, cost: np.ndarray, start, goal):
    """The cheapest walk from one hex to another: ``(cost, cells)``.

    Dijkstra with a heap over the lattice's own adjacency, so what comes out is a
    sequence of hexes something could actually march through -- which is the whole
    difference between this and a straight line.  Ties are settled by the heap's
    own order, which is by cost and then by cell, so the answer is a function of
    the map and not of the order the neighbours happened to be listed in.

    The cost is what the table charges for the cells the walk *enters*, and not for
    the one it starts in: an army is already standing in the capital, and charging
    it again would make "A to B" depend on how expensive A happens to be -- which
    would stop these numbers from being distances at all, and it is distances, of a
    kind, that the network is built from.

    Unreachable is ``(inf, [])``: across water, or off a landmass.  A caller that
    needs to know whether a road exists asks that, rather than asking whether a
    route is short.
    """
    rows, cols = cost.shape
    s = int(start[0]) * cols + int(start[1])
    g = int(goal[0]) * cols + int(goal[1])
    flat = cost.reshape(-1)
    if not np.isfinite(flat[s]):
        return np.inf, []
    best = np.full(rows * cols, np.inf, dtype=np.float64)
    came = np.full(rows * cols, -1, dtype=np.int64)
    best[s] = 0.0
    heap = [(0.0, s)]
    while heap:
        d, j = heapq.heappop(heap)
        if d > best[j]:
            continue
        if j == g:
            break
        r, c = divmod(j, cols)
        for rr, cc in grid.nbr[r, c]:
            rr, cc = int(rr), int(cc)
            if rr < 0:
                continue
            k = rr * cols + cc
            step = float(cost[rr, cc])
            if not np.isfinite(step):
                continue
            nd = d + step
            if nd < best[k]:
                best[k] = nd
                came[k] = j
                heapq.heappush(heap, (nd, k))
    if not np.isfinite(best[g]):
        return np.inf, []
    cells = []
    j = g
    while j >= 0:
        cells.append(j)
        if j == s:
            break
        j = int(came[j])
    cells.reverse()
    return float(best[g]), cells


def _spanning(weight: np.ndarray) -> list:
    """Prim's forest over a cost matrix, ties to the lowest index.

    A forest rather than a tree because ``weight`` is infinite between two capitals
    that cannot reach each other: the parts that can be connected are, and the parts
    that cannot are not quietly joined by a road over the sea.
    """
    n = weight.shape[0]
    seen = [False] * n
    edges = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        heap = [(float(weight[start, j]), start, j)
                for j in range(n) if not seen[j] and np.isfinite(weight[start, j])]
        heapq.heapify(heap)
        while heap:
            w, a, b = heapq.heappop(heap)
            if seen[b]:
                continue
            seen[b] = True
            edges.append((a, b, w))
            for j in range(n):
                if not seen[j] and np.isfinite(weight[b, j]):
                    heapq.heappush(heap, (float(weight[b, j]), b, j))
    return edges


def network(grid: HexGrid, f: dict, res: dict, per_segment: int = 6) -> list:
    """The roads between the capitals: ``(from, to, cost, cells, points)`` records.

    Every pair of capitals is routed, because a road between two of them is only
    worth building if it is cheaper than the roads that would connect them through
    a third -- and that is a comparison, not a guess.  The cheapest set that
    connects them all is then taken from those routes.

    ``points`` is the polyline to draw; ``path`` is the hexes the road walks, which
    is the answer the polyline is a rendering of; ``cells`` is how many there are;
    and ``cost`` is what the biome table charges for the walk -- the cells entered,
    so the capital it starts in is not charged twice.
    """
    if not res["count"]:
        return []
    cost = cost_field(grid, f)
    caps = [(int(p["capital"][0]), int(p["capital"][1])) for p in res["provinces"]]
    n = len(caps)
    weight = np.full((n, n), np.inf, dtype=np.float64)
    walks = {}
    for a in range(n):
        for b in range(a + 1, n):
            c, cells = route(grid, cost, caps[a], caps[b])
            if cells:
                weight[a, b] = weight[b, a] = c
                walks[(a, b)] = cells

    roads = []
    for a, b, c in _spanning(weight):
        cells = walks.get((a, b)) or walks[(b, a)]
        walk = [list(divmod(int(j), grid.cols)) for j in cells]
        pts = np.array([[float(grid.centers[r, cc, 0]), float(grid.centers[r, cc, 1])]
                        for r, cc in walk], dtype=np.float64)
        if len(pts) >= 2:
            pts = curve.catmull_rom(pts, per_segment=per_segment)
        roads.append({
            "from": int(a),
            "to": int(b),
            "cost": float(c),
            "cells": len(cells),
            "path": walk,
            "points": pts,
        })
    return roads
