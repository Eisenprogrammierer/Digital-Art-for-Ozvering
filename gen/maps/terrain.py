"""Terrain: an elevation field, a moisture field, and a table of biomes.

A map is two noise fields and a table.  The elevation decides what is sea and how
high the land is; the moisture decides what grows on it; the table says what a
cell of each combination is called, what it looks like and what it costs to walk
through.  Everything interesting here is in the first sentence, so most of the
care went into making the elevation *look like a landmass* rather than like fBm:

* it is evaluated in **normalised map coordinates**, ``x`` in units of half the
  map's width, so the same seed and the same grid shape give the same continent
  whatever ``size`` the hexes are drawn at.  A map is a function of
  ``(seed, cols, rows)``, not of its output resolution;
* it is **domain-warped first**, so every iso-line -- including the coastline --
  is bent into something that reads as geography rather than as noise;
* it is **sunk along a warped radius**, so the map is a continent in an ocean
  rather than a rectangle of mush;
* the sea level is a **quantile**, not a constant, so "42% water" means 42%
  whatever the noise happened to do;
* and it is **smoothed over the hex neighbourhood** once, because isolated
  single-hex islands and one-hex lakes are noise, and every map generator that
  skips that step produces them by the dozen.

No latitude is modelled.  A map that is a few hundred kilometres on a side has no
climate zones worth the name, and inventing a fake north would only make the
snowcaps appear in a neat band across the top, which is the tell of a generator
that pattern-matched a real map instead of generating one.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from core import noise, palette

from .grid import HexGrid

__all__ = ["Biome", "BIOMES", "INDEX", "colour_table", "fields", "classify", "smooth", "stats"]


class Biome(NamedTuple):
    """One class of ground: what it is called, what it looks like, what it costs."""

    key: str
    name: str
    hex: str
    cost: int
    water: bool

BIOMES = (
    Biome("deep", "deep water", "#15304a", 99, True),
    Biome("ocean", "ocean", "#1d4a6b", 99, True),
    Biome("shallow", "shallows", "#31799a", 99, True),
    Biome("shore", "shore", "#cbb98a", 1, False),
    Biome("desert", "desert", "#d9c88c", 2, False),
    Biome("steppe", "steppe", "#bcbd7e", 1, False),
    Biome("plains", "plains", "#9dbb72", 1, False),
    Biome("marsh", "marsh", "#6f8a5e", 3, False),
    Biome("forest", "forest", "#4b7f4e", 2, False),
    Biome("hills", "hills", "#8a9a5b", 2, False),
    Biome("mountain", "mountains", "#8f8f88", 3, False),
    Biome("snow", "snowcaps", "#e6edf2", 4, False),
)

INDEX = {b.key: i for i, b in enumerate(BIOMES)}

_WATER = np.array([b.water for b in BIOMES])


def colour_table() -> np.ndarray:
    """The biome colours as one ``(n, 3)`` sRGB array, to index per pixel."""
    return np.stack([palette.hex_to_srgb(b.hex) for b in BIOMES])


def smooth(field, grid: HexGrid, passes: int = 1, weight: float = 0.5):
    """Mean of each hex and its neighbours, repeated ``passes`` times.

    The neighbour indices come from the grid, so this is a six-step array
    accumulate rather than a loop over hexes, and the map's ragged edge is
    handled by renormalising on the count of neighbours that exist.
    """
    out = np.asarray(field, dtype=np.float64)
    for _ in range(int(passes)):
        acc = out.copy()
        n = np.ones(out.shape, dtype=np.float64)
        for d in range(6):
            r = grid.nbr[..., d, 0]
            c = grid.nbr[..., d, 1]
            ok = r >= 0
            acc = acc + np.where(ok, out[np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1)], 0.0)
            n = n + ok
        out = (1.0 - weight) * out + weight * (acc / n)
    return out


def fields(
    grid: HexGrid,
    seed: int = 0,
    sea_fraction: float = 0.42,
    features: float = 2.2,
    ridges: float = 0.30,
    edge: float = 0.95,
    smooth_passes: int = 1,
) -> dict:
    """Elevation, moisture and everything derived from them.

    Returns a dict rather than a tuple because every consumer wants a different
    subset of it and the names are the documentation.
    """
    w, h = grid.extent

    half = 0.5 * float(w)
    nx = grid.centers[..., 0] / half
    ny = grid.centers[..., 1] / half

    ux, uy = noise.warp(nx * features, ny * features, seed=seed + 3, amp=0.45, freq=1.0, octaves=3)
    elev = noise.fbm(ux, uy, seed=seed + 11, freq=1.0, octaves=6, gain=0.55)
    if ridges > 0.0:
        elev = elev + ridges * noise.ridged(ux, uy, seed=seed + 29, freq=1.7, octaves=5, gain=0.5)

    span = float(np.hypot(1.0, h / max(w, 1e-9)))
    r = np.hypot(ux / features, uy / features) / span
    t = np.clip((r - 0.50) / 0.42, 0.0, 1.0)
    elev = elev - edge * t * t

    lo, hi = float(elev.min()), float(elev.max())
    elev = (elev - lo) / max(hi - lo, 1e-9)
    elev = smooth(elev, grid, smooth_passes)

    sea = float(np.quantile(elev, sea_fraction))
    water = elev < sea
    alt = (elev - sea) / max(1.0 - sea, 1e-6)

    vx, vy = noise.warp(nx * 2.6, ny * 2.6, seed=seed + 91, amp=0.45, freq=1.0, octaves=3)
    moist = noise.fbm(vx, vy, seed=seed + 57, freq=1.0, octaves=4, gain=0.55)
    moist = (moist - moist.min()) / max(float(moist.max() - moist.min()), 1e-9)

    nbr_water = np.zeros(water.shape, dtype=bool)
    nbr_land = np.zeros(water.shape, dtype=bool)
    for d in range(6):
        rr = np.clip(grid.nbr[..., d, 0], 0, grid.rows - 1)
        cc = np.clip(grid.nbr[..., d, 1], 0, grid.cols - 1)
        ok = grid.nbr[..., d, 0] >= 0
        nbr_water = nbr_water | (ok & water[rr, cc])
        nbr_land = nbr_land | (ok & ~water[rr, cc])

    out = {
        "elev": elev,
        "alt": alt,
        "moist": moist,
        "sea": sea,
        "water": water,
        "near_sea": (~water) & nbr_water,
        "near_land": water & nbr_land,
    }

    out["height"] = np.where(water, sea, elev)
    out["biome"] = classify(out)
    return out


def classify(f: dict) -> np.ndarray:
    """Biome codes for a field dict, by altitude first and moisture second."""
    elev = f["elev"]
    alt = f["alt"]
    moist = f["moist"]
    water = f["water"]
    sea = f["sea"]

    code = np.zeros(elev.shape, dtype=np.int64)

    code[water] = INDEX["ocean"]
    code[water & f["near_land"]] = INDEX["shallow"]
    code[water & ((sea - elev) > 0.13)] = INDEX["deep"]

    land = ~water
    code[land] = INDEX["plains"]
    code[land & (moist > 0.60)] = INDEX["forest"]
    code[land & (moist > 0.74) & (alt < 0.18)] = INDEX["marsh"]
    code[land & (moist < 0.42)] = INDEX["steppe"]
    code[land & (moist < 0.30)] = INDEX["desert"]
    code[land & (alt > 0.34)] = INDEX["hills"]
    code[land & (alt > 0.60)] = INDEX["mountain"]
    code[land & (alt > 0.80)] = INDEX["snow"]
    code[f["near_sea"] & (alt < 0.06)] = INDEX["shore"]
    return code


def stats(codes) -> dict:
    """How much of the map is what, by name -- for the sidecar and the label."""
    codes = np.asarray(codes)
    total = int(codes.size)
    out = {}
    for i, b in enumerate(BIOMES):
        n = int(np.count_nonzero(codes == i))
        if n:
            out[b.name] = n
    out["hexes"] = total
    water = int(np.count_nonzero(_WATER[codes]))
    out["water"] = water / max(total, 1)
    return out
