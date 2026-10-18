"""Provinces: the map divided into parts that can be governed.

Two properties matter here and neither of them is visual.

**A province must be connected.**  Nearest-capital by straight-line distance is
the obvious assignment -- it is one line of arithmetic -- and it produces
provinces in two pieces, because a bay does not care about the straight line
between two points.  A province in two pieces is not a province: an army cannot
march from one half to the other without leaving it.  The assignment here is a
multi-source breadth-first search over the lattice, so it is the map's *own*
adjacency that decides, and a province is connected by construction.

**The parts should be evenly sized.**  Seeds are spread by Mitchell's best
candidate -- propose several cells, keep the one furthest from everything chosen
so far -- and then relaxed by Lloyd's algorithm: assign, move each capital to the
middle of its province, assign again.  The middle of a province is its
one-median, the cell that minimises the path distance to every other cell in it,
and it is computed exactly by breadth-first search from each of the province's
own cells rather than approximated by the centroid.  At this size the exact
answer costs well under a millisecond, and an approximation that only *sometimes*
moves the capital is exactly the kind of thing that makes a generator drift.

Seeds are shared out between the separate landmasses in proportion to their area,
because an archipelago is not one continent: without that, the first seed placed
on a small island would attract every other one, and the main continent would be
one enormous province.  An island that ends up with no seed at all becomes a
province of its own, with its capital at its own one-median -- which is not a
special case bolted on but what an island is.

**The sea is divided the same way, by the same code** (:func:`seas`).  A fleet moves
from sea province to sea province the way an army moves from province to province, so
the water needs the same two properties and gets them from the same functions; the
only things that change are which cells are in play and which endings the names
take.  Between the two calls, every hex on the map is in exactly one province, and
no hex is in both.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from core import noise, render

from . import terrain
from .grid import HexGrid

__all__ = [
    "assign",
    "components",
    "frontier_alpha",
    "name",
    "one_median",
    "provinces",
    "relax",
    "seas",
    "sites",
]

#: The sea draws its seeds from its own stream.  Sharing one with the land would
#: put a sea capital wherever a land capital happened to be, which is a
#: coincidence no chart has ever shown.
_SEA_SALT = 4099

#: Stems and endings for province names.  Deliberately few: twenty times sixteen
#: is more distinct names than a map has provinces, and a generator that invents
#: words out of nowhere produces things nobody can pronounce.
_STEM = ("ald", "bren", "cor", "dun", "el", "fen", "gar", "hal", "ith", "kel",
         "mar", "nor", "ost", "rav", "sel", "tor", "vas", "wyn", "yr", "zel")

#: Endings, in the two or three spellings the ground of a province tends to take.
#: The name is the only place a player ever reads what a province is like, so the
#: ending carries the geography: a province mostly of hills ends in ``-fell`` and
#: one mostly of marsh ends in ``-fen``, and no province has to be told apart by a
#: prefix that repeats five times on one map.
_END = {
    "": ("mere", "vale", "march", "fell", "reach", "moor", "holt", "stead",
         "wick", "gard", "heim", "bury", "thorpe", "dale", "crest", "haven"),
    "plains": ("vale", "stead", "bury", "thorpe", "dale", "march", "heim"),
    "hills": ("fell", "moor", "crest", "reach", "dale"),
    "mountain": ("fell", "crest", "crag", "march"),
    "snow": ("fell", "crest", "gard"),
    "forest": ("holt", "wick", "dale", "wood"),
    "marsh": ("fen", "moor", "mere"),
    "desert": ("reach", "waste", "dune"),
    "shore": ("haven", "mere", "wick"),
    # The water's own endings, and deliberately none of them a land ending: a land
    # province and a sea province may draw the same stem, since the stem is the
    # province's number, and the ending is the whole of what tells them apart.
    "deep water": ("deep", "gulf", "trench"),
    "ocean": ("sea", "ocean", "sound"),
    "shallows": ("shoal", "shelf", "strand", "bay"),
}


# ------------------------------------------------------------------ search
def _masked_bfs(grid: HexGrid, flat_mask, start: int) -> np.ndarray:
    """Hex distance from ``start`` to every cell of a mask, ``-1`` elsewhere.

    Breadth-first search over whole frontiers rather than one cell at a time: the
    one-median needs a search from *every* cell of a province, which is hundreds of
    searches per map, and in a Python loop that is the slowest thing in the
    generator by an order of magnitude.  Vectorised, a search costs one numpy pass
    per ring and the whole thing disappears into the noise.
    """
    flat_mask = np.asarray(flat_mask, dtype=bool)
    dist = np.full(grid.count, -1, dtype=np.int64)
    start = int(start)
    dist[start] = 0
    frontier = np.array([start], dtype=np.int64)
    step = 0
    while frontier.size:
        step += 1
        r, c = np.divmod(frontier, grid.cols)
        nbr = grid.nbr[r, c]                 # (k, 6, 2)
        rr = nbr[..., 0].reshape(-1)
        cc = nbr[..., 1].reshape(-1)
        keep = rr >= 0
        idx = (rr * grid.cols + cc)[keep]
        idx = idx[flat_mask[idx]]
        idx = idx[dist[idx] < 0]
        if idx.size == 0:
            break
        idx = np.unique(idx)
        dist[idx] = step
        frontier = idx
    return dist


def components(grid: HexGrid, mask) -> list:
    """The connected parts of a mask, largest first, as flat index arrays."""
    flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
    seen = np.zeros(grid.count, dtype=bool)
    out = []
    for start in np.flatnonzero(flat_mask):
        start = int(start)
        if seen[start]:
            continue
        seen[start] = True
        comp = [start]
        q = deque([start])
        while q:
            j = q.popleft()
            r, c = divmod(j, grid.cols)
            for d in range(6):
                nr, nc = (int(v) for v in grid.nbr[r, c, d])
                if nr < 0:
                    continue
                k = nr * grid.cols + nc
                if flat_mask[k] and not seen[k]:
                    seen[k] = True
                    comp.append(k)
                    q.append(k)
        out.append(np.array(comp, dtype=np.int64))
    # Largest first, ties by the earliest cell in reading order: the order decides
    # which landmass gets the remainder of the seeds, so it must not be incidental.
    out.sort(key=lambda a: (-int(a.size), int(a[0])))
    return out


def _allocate(count: int, sizes) -> list:
    """Share ``count`` seeds out in proportion to area (largest remainder)."""
    sizes = [int(s) for s in sizes]
    total = float(sum(sizes))
    if total <= 0 or count <= 0:
        return [0] * len(sizes)
    raw = [count * s / total for s in sizes]
    out = [int(np.floor(x)) for x in raw]
    left = count - sum(out)
    order = sorted(range(len(sizes)), key=lambda i: (-(raw[i] - out[i]), i))
    for i in order[:max(left, 0)]:
        out[i] += 1
    return out


def _pick(grid: HexGrid, flat_mask, pool, count: int, rng, tries: int = 24) -> list:
    """``count`` spread-out cells inside one connected pool, by best candidate."""
    if count <= 0 or pool.size == 0:
        return []
    chosen = []
    dist = None
    while len(chosen) < count:
        best, best_d = None, None
        for cand in rng.integers(0, pool.size, size=max(int(tries), 1)):
            c = int(pool[int(cand)])
            d = int(dist[c]) if dist is not None else 1
            if d > 0 and (best_d is None or d > best_d):
                best, best_d = c, d
        if best is None:
            break  # the pool is smaller than the seeds asked of it
        chosen.append(best)
        new = _masked_bfs(grid, flat_mask, best)
        dist = new if dist is None else np.minimum(dist, new)
    return chosen


def sites(grid: HexGrid, mask, count: int, seed: int = 0, tries: int = 24) -> list:
    """``count`` well-spread cells on the land, shared out between landmasses."""
    mask = np.asarray(mask, dtype=bool)
    flat_mask = mask.reshape(-1)
    comps = components(grid, mask)
    if not comps:
        raise ValueError("no cells to place seeds in")
    rng = np.random.default_rng(int(seed))
    chosen = []
    for comp, k in zip(comps, _allocate(int(count), [c.size for c in comps])):
        chosen.extend(_pick(grid, flat_mask, comp, k, rng, tries))
    return chosen


def assign(grid: HexGrid, mask, seeds) -> np.ndarray:
    """Grow every seed at once, so each cell joins the nearest one *by path*.

    Ties go to the lowest-numbered seed, and the queue is filled in seed order, so
    the division is the same on every run.
    """
    flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
    labels = np.full(grid.count, -1, dtype=np.int64)
    q = deque()
    for i, s in enumerate(seeds):
        s = int(s)
        if labels[s] >= 0:
            raise ValueError(f"two seeds share the cell {s}")
        labels[s] = i
        q.append(s)
    while q:
        j = q.popleft()
        r, c = divmod(j, grid.cols)
        for d in range(6):
            nr, nc = (int(v) for v in grid.nbr[r, c, d])
            if nr < 0:
                continue
            k = nr * grid.cols + nc
            if flat_mask[k] and labels[k] < 0:
                labels[k] = labels[j]
                q.append(k)
    return labels.reshape(grid.rows, grid.cols)


def one_median(grid: HexGrid, flat_mask, cells) -> int:
    """The cell of a connected set that minimises the path distance to the rest."""
    cells = [int(c) for c in cells]
    best, best_total = cells[0], None
    for s in cells:
        dist = _masked_bfs(grid, flat_mask, s)
        total = int(sum(dist[c] for c in cells))
        if best_total is None or total < best_total:
            best, best_total = s, total
    return best


def _add_islands(grid: HexGrid, land, labels, seeds) -> list:
    """Every landmass no capital could reach becomes a province of its own."""
    un = np.asarray(land, dtype=bool) & (labels < 0)
    seeds = list(seeds)
    for comp in components(grid, un):
        labels.reshape(-1)[comp] = len(seeds)
        seeds.append(one_median(grid, np.asarray(land, dtype=bool).reshape(-1), comp))
    return seeds


def relax(grid: HexGrid, mask, labels, passes: int = 2):
    """Lloyd's algorithm on the lattice: move each capital to its province's middle."""
    flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
    seeds = []
    for _ in range(max(int(passes), 0)):
        seeds = [one_median(grid, flat_mask, np.flatnonzero((labels == i).reshape(-1)))
                 for i in range(int(labels.max()) + 1)]
        if not seeds:
            break
        labels = assign(grid, mask, seeds)
    return labels, seeds


# ---------------------------------------------------------------- provinces
def _divide(grid: HexGrid, f: dict, mask, count: int, seed: int, passes: int,
            salt: int) -> dict:
    """Divide one kind of ground into parts, and describe each part.

    The land and the sea are divided by the same code because they are the same
    problem: a set of cells, connected by the lattice, wanted in pieces that are
    connected and about as big as each other.  Nothing below this line knows which
    of the two it was handed -- which is the point, because a second implementation
    for the water is how two divisions drift apart.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"labels": np.full((grid.rows, grid.cols), -1, dtype=np.int64),
                "provinces": [], "count": 0, "islands": 0}
    seeds = sites(grid, mask, count, seed=int(seed) + int(salt))
    labels = assign(grid, mask, seeds)
    islands = 0
    if seeds:
        before = len(seeds)
        seeds = _add_islands(grid, mask, labels, seeds)
        islands = len(seeds) - before
    if seeds and passes:
        labels, seeds = relax(grid, mask, labels, passes)
    label_flat = labels.reshape(-1)

    centres = grid.centers.reshape(-1, 2)
    biome_flat = np.asarray(f["biome"]).reshape(-1)
    near_sea_flat = np.asarray(f["near_sea"]).reshape(-1)
    out = []
    used = set()
    for i, s in enumerate(seeds):
        cells = np.flatnonzero(label_flat == i)
        r, c = divmod(int(s), grid.cols)
        codes = [int(biome_flat[j]) for j in cells]
        keys = sorted(set(codes))
        ground = max(keys, key=codes.count) if keys else -1
        gname = terrain.BIOMES[ground].name if ground >= 0 else ""
        neighbours = set()
        for j in cells:
            jr, jc = divmod(int(j), grid.cols)
            for d in range(6):
                nr, nc = (int(v) for v in grid.nbr[jr, jc, d])
                if nr < 0:
                    continue
                other = int(label_flat[nr * grid.cols + nc])
                if other >= 0 and other != i:
                    neighbours.add(other)
        # No two provinces on one map may share a name: a name is the handle a
        # player uses, and the generator has enough of them that a collision is
        # only ever a hashing accident, not a shortage.
        label = name(seed, i, gname)
        salt_used = 0
        while label in used and salt_used < 64:
            salt_used += 1
            label = name(seed, i, gname, salt_used)
        used.add(label)
        out.append({
            "index": i,
            "cells": cells,
            "capital": (r, c),
            "centre": (float(centres[cells, 0].mean()), float(centres[cells, 1].mean())),
            "area": int(cells.size),
            "neighbours": sorted(neighbours),
            "coastal": bool(near_sea_flat[cells].any()),
            "ground": gname,
            "name": label,
        })
    return {"labels": labels, "provinces": out, "count": len(out), "islands": islands}


def provinces(grid: HexGrid, f: dict, count: int = 8, seed: int = 0,
              passes: int = 2) -> dict:
    """Divide the land into provinces.

    Returns ``{"labels", "provinces", "count", "islands"}``.  ``labels`` is ``-1``
    on water; each province carries its cells, its capital, its neighbours, its
    area, its commonest ground and a name.
    """
    return _divide(grid, f, ~np.asarray(f["water"], dtype=bool), count, seed, passes, 0)


def seas(grid: HexGrid, f: dict, count: int = 8, seed: int = 0, passes: int = 2) -> dict:
    """Divide the water into sea provinces, and for the same reasons.

    A fleet moves province to province the way an army does, so the water needs the
    same two properties: a sea province is connected, or a ship cannot reach part of
    its own patrol, and the parts are about as big as each other, or one ocean is a
    single province with a coastline of two thousand miles.

    A body of water the seeds did not reach -- a lake, or a bay that is all coast --
    becomes a sea province of its own, exactly as an island becomes a land province.
    The shapes are the same shape; the water names end in ``-sea`` and ``-bay``
    rather than in ``-fell`` and ``-fen``.
    """
    return _divide(grid, f, np.asarray(f["water"], dtype=bool), count, seed, passes,
                   _SEA_SALT)


def frontier_alpha(grid: HexGrid, labels, cell, width: float = 0.26, dx: float = 1.0):
    """Coverage of the province frontiers, from the label across each edge.

    Drawn per pixel rather than as one stroke per frontier segment, for the same
    reason the lattice is: a frontier is a property of the two cells that meet
    there, and every pixel already knows which edge it is nearest to and who is on
    the other side of it.  Water is not part of any province, so a coast is not a
    frontier and the two are drawn by different code.

    The default width is deliberately twice the lattice's, and that is not a
    stylistic preference.  On a hex map *every* hex edge is already drawn as
    lattice, and a frontier is a set of hex edges, so a frontier line of the
    lattice's weight is invisible: it lands exactly on a line that is there
    anyway.  What tells a player where one province ends is that the line is
    heavier, not that it is a different colour.
    """
    labels = np.asarray(labels)
    r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
    c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
    own = labels[r, c]
    nr = grid.nbr[r, c, cell.sector, 0]
    nc = grid.nbr[r, c, cell.sector, 1]
    other = labels[np.clip(nr, 0, grid.rows - 1), np.clip(nc, 0, grid.cols - 1)]
    differs = (nr >= 0) & (own >= 0) & (other >= 0) & (own != other)
    band = render.coverage(cell.edge_distance() - 0.5 * float(width) * grid.size, dx)
    return band * differs * cell.ok.astype(np.float64)


def name(seed: int, index: int, ground: str = "", salt: int = 0) -> str:
    """A deterministic name: a stem, and an ending that suits the ground.

    Two separate hashes, because one hash would draw the stem and the ending from
    the same pool of bits and the two would move together.  ``salt`` lets a caller
    ask for a different name for the same province, which is how a collision
    between two provinces on one map is resolved.
    """
    ends = _END.get(ground, _END[""])
    h1 = float(noise.hash2(np.int64(index + salt), np.int64(salt), seed=int(seed) + 8191))
    h2 = float(noise.hash2(np.int64(salt), np.int64(index + salt), seed=int(seed) + 104729))
    stem = _STEM[min(int(h1 * len(_STEM)), len(_STEM) - 1)]
    end = ends[min(int(h2 * len(ends)), len(ends) - 1)]
    return stem.capitalize() + end
