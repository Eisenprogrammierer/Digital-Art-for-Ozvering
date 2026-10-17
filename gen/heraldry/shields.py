"""Silhouettes: the shields (escutcheons) and the flag fields.

Every silhouette is a signed distance field in a box spanning ``y in [-hh, hh]``
and ``x in [-hw, hw]``.  Because the field's *extent* is returned alongside the
shape, the ordinaries and charges can be positioned in the same coordinates
without any of them knowing which silhouette they are drawn on -- a fess lands
across the middle of a heater shield and across the middle of a 3:2 flag with no
special cases.
"""
from __future__ import annotations

import numpy as np

from core import curve, sdf

__all__ = ["SHAPES", "FLAG_SHAPES", "get", "flag", "names"]

Shape = "tuple[Callable, float, float]"


def _polygon_shape(points, hw: float, hh: float):
    pts = np.asarray(points, dtype=np.float64)
    return lambda x, y: sdf.polygon(x, y, pts), hw, hh


def _heater(hw: float = 0.80, hh: float = 1.0):
    """The classic escutcheon: flat chief, straight flanks, a point at the base."""
    right = curve.bezier_quadratic((hw, 0.20), (hw * 1.10, -0.52), (0.0, -hh), 18)
    pts = np.vstack(
        [
            [(-hw, hh)],
            [(hw, hh)],
            right,
            right[::-1] * np.array([-1.0, 1.0]),
        ]
    )
    return _polygon_shape(pts, hw, hh)


def _iberian(hw: float = 0.80, hh: float = 1.0):
    """Rounded base: a box plus a circle, tangent so the join needs no fillet."""
    shoulder = 0.38  # where the straight flanks end
    body = lambda x, y: np.minimum(  # noqa: E731
        sdf.box(x, y, hw, hh - shoulder, 0.06, 0.0, shoulder),
        sdf.circle(x, y, hw, 0.0, hw - hh),
    )
    return lambda x, y: sdf.intersection(body(x, y), sdf.box(x, y, hw, hh)), hw, hh


def _french(hw: float = 0.78, hh: float = 1.0):
    """Old French: a bowed chief and a shallower point."""
    right = curve.bezier_quadratic((hw, 0.35), (hw * 1.02, -0.45), (0.0, -hh * 0.92), 18)
    chief = curve.bezier_quadratic((-hw, hh), (0.0, hh + 0.10), (hw, hh), 12)
    pts = np.vstack([chief, right, right[::-1] * np.array([-1.0, 1.0])])
    return _polygon_shape(pts, hw, hh)


def _oval(hw: float = 0.78, hh: float = 1.0):
    return lambda x, y: sdf.ellipse(x, y, hw, hh), hw, hh


def _lozenge(hw: float = 0.74, hh: float = 1.0):
    return _polygon_shape([(0.0, hh), (hw, 0.0), (0.0, -hh), (-hw, 0.0)], hw, hh)


def _rounded(hw: float = 0.80, hh: float = 1.0):
    return lambda x, y: sdf.box(x, y, hw, hh, 0.12), hw, hh


def _banner(hw: float = 0.90, hh: float = 1.0):
    """A banner of arms: close to square, sharp corners."""
    return lambda x, y: sdf.box(x, y, hw, hh, 0.0), hw, hh


#: Escutcheon silhouettes for coats of arms.
SHAPES = {
    "heater": _heater,
    "iberian": _iberian,
    "french": _french,
    "oval": _oval,
    "lozenge": _lozenge,
    "rounded": _rounded,
    "banner": _banner,
}


def _flag_field(ratio=(3, 2)):
    hw = float(ratio[0]) / float(ratio[1])
    return lambda x, y: sdf.box(x, y, hw, 1.0, 0.0), hw, 1.0


def _swallowtail(ratio=(3, 2), notch: float = 0.34):
    """A flag with a swallow-tailed fly."""
    hw = float(ratio[0]) / float(ratio[1])
    cut_pts = np.array([[hw, 1.4], [hw - notch * hw, 0.0], [hw, -1.4]])

    def shape(x, y):
        return sdf.difference(sdf.box(x, y, hw, 1.0, 0.0), sdf.polygon(x, y, cut_pts))

    return shape, hw, 1.0


def _pennant(ratio=(5, 2)):
    """A triangular pennant, point to the fly."""
    hw = float(ratio[0]) / float(ratio[1])
    pts = [(-hw, 1.0), (hw, 0.0), (-hw, -1.0)]
    return _polygon_shape(pts, hw, 1.0)


def _guidon(ratio=(3, 2)):
    """A guindon: a rectangle with a triangular fly (a vertical cut)."""
    hw = float(ratio[0]) / float(ratio[1])
    return _polygon_shape([(-hw, 1.0), (hw * 0.45, 1.0), (hw, 0.0), (hw * 0.45, -1.0), (-hw, -1.0)], hw, 1.0)


#: Flag silhouettes, keyed by name; ``flag(ratio)`` gives the plain rectangle.
FLAG_SHAPES = {
    "rectangle": _flag_field,
    "swallowtail": _swallowtail,
    "pennant": _pennant,
    "guidon": _guidon,
}


def get(name) -> tuple:
    """Return ``(field_fn, hw, hh)`` for a shield silhouette name."""
    if name not in SHAPES:
        raise KeyError(f"unknown shield {name!r}; known: {', '.join(SHAPES)}")
    return SHAPES[name]()


def flag(ratio=(3, 2), kind: str = "rectangle") -> tuple:
    """Return ``(field_fn, hw, hh)`` for a flag of the given proportions."""
    if kind not in FLAG_SHAPES:
        raise KeyError(f"unknown flag shape {kind!r}; known: {', '.join(FLAG_SHAPES)}")
    return FLAG_SHAPES[kind](ratio)


def names() -> list:
    return sorted(SHAPES)
