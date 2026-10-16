"""Affine 2-D transforms in homogeneous coordinates.

A transform is a 3x3 matrix acting on column vectors ``(x, y, 1)``.  Because we
use column vectors, ``compose(a, b)`` builds ``a @ b`` and therefore applies
``b`` first, then ``a``.

All helpers are pure functions returning ``float64`` arrays, so they compose
freely with numpy broadcasting and never mutate their arguments.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "identity",
    "translate",
    "rotate",
    "scale",
    "shear",
    "compose",
    "invert",
    "apply",
    "apply_grid",
    "uniform_scale",
]


def identity() -> np.ndarray:
    """The 3x3 identity transform."""
    return np.eye(3, dtype=np.float64)


def translate(tx: float, ty: float) -> np.ndarray:
    """Translate by ``(tx, ty)``."""
    m = identity()
    m[0, 2] = tx
    m[1, 2] = ty
    return m


def rotate(deg: float) -> np.ndarray:
    """Rotate counter-clockwise by ``deg`` degrees (in a y-up frame)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def scale(sx: float, sy: float | None = None) -> np.ndarray:
    """Scale by ``sx`` horizontally and ``sy`` vertically (default: uniform)."""
    sy = sx if sy is None else sy
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


def shear(kx: float = 0.0, ky: float = 0.0) -> np.ndarray:
    """Shear/skew the frame: ``x' = x + kx*y``, ``y' = y + ky*x``."""
    return np.array([[1.0, kx, 0.0], [ky, 1.0, 0.0], [0.0, 0.0, 1.0]])


def compose(*mats: np.ndarray) -> np.ndarray:
    """Multiply transforms left-to-right: ``m1 @ m2 @ ...``."""
    m = identity()
    for x in mats:
        m = m @ np.asarray(x, dtype=np.float64)
    return m


def invert(m: np.ndarray) -> np.ndarray:
    """Inverse of an affine transform (general 3x3 inversion)."""
    return np.linalg.inv(np.asarray(m, dtype=np.float64))


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform an array of points shaped ``(..., 2)``."""
    p = np.asarray(pts, dtype=np.float64)
    ones = np.ones(p.shape[:-1] + (1,), dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    return (np.concatenate([p, ones], axis=-1) @ m.T)[..., :2]


def apply_grid(m: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Transform two coordinate arrays without materialising Nx2 points.

    Works on any broadcastable ``x``/``y``, including the full meshgrids that
    :class:`core.canvas.Canvas` produces.
    """
    m = np.asarray(m, dtype=np.float64)
    return (
        m[0, 0] * x + m[0, 1] * y + m[0, 2],
        m[1, 0] * x + m[1, 1] * y + m[1, 2],
    )


def uniform_scale(m: np.ndarray) -> float:
    """Best uniform scale factor of ``m``, used to keep SDF values metric.

    The determinant of the linear part equals the area scale; its square root
    is the equivalent isotropic scale, which is exact for rotations/uniform
    scaling and a sane approximation for shear and non-uniform scaling.
    """
    lin = np.asarray(m, dtype=np.float64)[:2, :2]
    det = float(np.linalg.det(lin))
    return float(np.sqrt(abs(det))) or 1.0
