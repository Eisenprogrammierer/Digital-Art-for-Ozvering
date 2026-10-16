"""Deterministic lattice noise: value noise, fBm, warping, Worley cells.

Everything is derived from an integer hash of the lattice cell, so a seed
reproduces the exact same field on any machine, in any numpy version, without
storing state.  That is what makes procedurally generated art reproducible:
the PNG is a pure function of ``(seed, params)``.

Value noise is used instead of Perlin/gradient noise on purpose -- for painted
textures the lattice artefacts are invisible once several octaves and a domain
warp are stacked, and the implementation stays short.
"""
from __future__ import annotations

import numpy as np

__all__ = ["hash2", "value_noise", "fbm", "ridged", "turbulence", "warp", "worley", "worley_edges"]


def _u64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64).astype(np.uint64)


def hash2(ix, iy, seed: int = 0) -> np.ndarray:
    """Hash integer lattice coordinates to floats in ``[0, 1)``.

    SplitMix64-style finaliser; unsigned overflow is intentional and expected.
    """
    with np.errstate(over="ignore"):
        h = _u64(ix) * np.uint64(0x9E3779B97F4A7C15)
        h ^= _u64(iy) * np.uint64(0xC2B2AE3D27D4EB4F)
        h ^= _u64(np.int64(seed)) * np.uint64(0x165667B19E3779F9)
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _fade(t):
    """Quintic smoothstep: C2-continuous, kills lattice kinks."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise(x, y, seed: int = 0, freq: float = 1.0) -> np.ndarray:
    """Bilinear value noise in ``[0, 1]`` on a unit lattice scaled by ``freq``."""
    x = np.asarray(x, dtype=np.float64) * freq
    y = np.asarray(y, dtype=np.float64) * freq
    ix = np.floor(x)
    iy = np.floor(y)
    fx = _fade(x - ix)
    fy = _fade(y - iy)
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)

    n00 = hash2(ix, iy, seed)
    n10 = hash2(ix + 1, iy, seed)
    n01 = hash2(ix, iy + 1, seed)
    n11 = hash2(ix + 1, iy + 1, seed)

    top = n00 + (n10 - n00) * fx
    bottom = n01 + (n11 - n01) * fx
    return top + (bottom - top) * fy


def fbm(x, y, seed: int = 0, freq: float = 1.0, octaves: int = 5, gain: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """Fractional Brownian motion: octaves summed with decreasing amplitude.

    Result is normalised to ``[0, 1]``, so it can be fed straight into a colour
    ramp.  ``gain`` controls roughness (0.5 = classic 1/f), ``lacunarity`` the
    frequency step between octaves.
    """
    total = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=np.float64)
    amp = 1.0
    norm = 0.0
    f = float(freq)
    for i in range(octaves):
        total += amp * value_noise(x, y, seed=seed + i * 131, freq=f)
        norm += amp
        amp *= gain
        f *= lacunarity
    return total / max(norm, 1e-12)


def ridged(x, y, seed: int = 0, freq: float = 1.0, octaves: int = 5, gain: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """Ridged multifractal: creased ridges, good for rock, bark and mountains."""
    total = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=np.float64)
    amp = 1.0
    norm = 0.0
    f = float(freq)
    for i in range(octaves):
        n = value_noise(x, y, seed=seed + i * 197, freq=f)
        total += amp * (1.0 - np.abs(2.0 * n - 1.0))
        norm += amp
        amp *= gain
        f *= lacunarity
    return total / max(norm, 1e-12)


def turbulence(x, y, seed: int = 0, freq: float = 1.0, octaves: int = 5, gain: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """Absolute-value fBm: ``sum |2n-1|``, a blotchy, cloud-like field."""
    total = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=np.float64)
    amp = 1.0
    norm = 0.0
    f = float(freq)
    for i in range(octaves):
        n = value_noise(x, y, seed=seed + i * 263, freq=f)
        total += amp * np.abs(2.0 * n - 1.0)
        norm += amp
        amp *= gain
        f *= lacunarity
    return total / max(norm, 1e-12)


def warp(x, y, seed: int = 0, amp: float = 0.35, freq: float = 2.0, octaves: int = 4):
    """Domain warping: offset coordinates by another noise field.

    ``(x, y) -> (x + amp*N, y + amp*N')``.  This is the cheapest way to turn a
    bland fBm into flowing, marbled, hand-painted-looking structure, because the
    warp bends every iso-line of the base field.
    """
    dx = fbm(x, y, seed=seed + 1013, freq=freq, octaves=octaves) - 0.5
    dy = fbm(x, y, seed=seed + 2027, freq=freq, octaves=octaves) - 0.5
    return x + 2.0 * amp * dx, y + 2.0 * amp * dy


def worley(x, y, seed: int = 0, freq: float = 1.0, jitter: float = 1.0) -> np.ndarray:
    """Worley/cellular noise: distance to the nearest jittered feature point.

    Roughly ``[0, 1.4]`` at unit density.  Drives cracked stone, dragon scales,
    leaf cells and province borders.
    """
    x = np.asarray(x, dtype=np.float64) * freq
    y = np.asarray(y, dtype=np.float64) * freq
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    best = np.full(np.broadcast_shapes(x.shape, y.shape), np.inf, dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx = ix + dx
            cy = iy + dy
            fx = cx + 0.5 + jitter * (hash2(cx, cy, seed) - 0.5)
            fy = cy + 0.5 + jitter * (hash2(cx, cy, seed + 7717) - 0.5)
            best = np.minimum(best, np.hypot(x - fx, y - fy))
    return best


def worley_edges(x, y, seed: int = 0, freq: float = 1.0, jitter: float = 1.0) -> np.ndarray:
    """``F2 - F1``: distance to the nearest cell border, zero *on* the border.

    Distance-to-feature (F1, see :func:`worley`) gives cell interiors; this gives
    the *walls* between them, which is what you actually want for cracks, dragon
    scales, dry lake beds and province frontiers.  Same units as ``worley``: 0 on
    a boundary, growing towards the middle of a cell.
    """
    x = np.asarray(x, dtype=np.float64) * freq
    y = np.asarray(y, dtype=np.float64) * freq
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    shape = np.broadcast_shapes(x.shape, y.shape)
    f1 = np.full(shape, np.inf, dtype=np.float64)
    f2 = np.full(shape, np.inf, dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx = ix + dx
            cy = iy + dy
            fx = cx + 0.5 + jitter * (hash2(cx, cy, seed) - 0.5)
            fy = cy + 0.5 + jitter * (hash2(cx, cy, seed + 7717) - 0.5)
            d = np.hypot(x - fx, y - fy)
            closer = d < f1
            # Update F2 from the *previous* F1 before F1 advances.
            f2 = np.where(closer, f1, np.minimum(f2, d))
            f1 = np.where(closer, d, f1)
    return f2 - f1
