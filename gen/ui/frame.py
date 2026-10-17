"""The geometry of a panel, as distance fields on a design-pixel grid.

A panel is five concentric rounded rectangles and four corner ornaments:

    outer   the silhouette -- the whole asset
    inner   the outer edge moved in by `border`: the frame band is between them
    rule    moved in again by `rule_gap`: the inner moulding line is a stroke
            of half-width `rule / 2` around it
    boss    a scalloped disc sitting on the band's diagonal in each of the four
            corners, whose radius is a Fourier series (see :func:`boss_field`)

No coordinates here are in output pixels.  The design lives in design pixels and
the canvas scale only decides how many pixels one of them covers, so the same
spec renders a 24 px button and a 512 px hero panel with identical proportions --
and, more importantly, the *same* corner geometry, which is what makes the
nine-slice insets meaningful.
"""
from __future__ import annotations

import numpy as np

from core import sdf

from . import style as stylelib

__all__ = ["Panel", "boss_field", "boss_radius"]

#: The four corner directions, counter-clockwise from the top left.
CORNERS = ((-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))


def boss_radius(theta, r0: float, harmonics: int = 7, amp: float = 0.12):
    """A Fourier radius function: ``r0 * (1 + amp * sum_k cos(k t + p_k) / k')``.

    This is the one place a Fourier series earns its keep in this project.  A
    border whose radius varies around the whole perimeter cannot survive a
    nine-slice stretch, because the resample stretches the perimeter -- so the
    series is used where the geometry is *fixed*: inside a corner box, which is
    never resampled at all.

    The sum is normalised by its own harmonic weight, so the result is bounded by
    ``r0 * (1 +/- amp)`` for any harmonic count, which is what :func:`style.clamp`
    relies on when it promises the ornament stays in its corner.
    """
    harmonics = int(max(harmonics, 2))
    k = np.arange(2, harmonics + 1, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    # One broadcast sum rather than a Python loop over harmonics: a contact sheet
    # evaluates this thousands of times, and that loop was the most expensive
    # arithmetic in the package.  The harmonic axis goes first so the same code
    # works for a scalar, a 1-D arc of angles and a 2-D field.
    kk = k.reshape((-1,) + (1,) * theta.ndim)
    total = np.sum(np.cos(kk * theta + 0.7 * kk) / (kk - 1.0), axis=0)
    weight = float(np.sum(1.0 / (k - 1.0)))
    return r0 * (1.0 + amp * total / max(weight, 1e-9))


def boss_field(X, Y, cx: float, cy: float, r0: float, harmonics: int = 7, amp: float = 0.12):
    """The ornament as a signed field: negative inside the scalloped disc.

    This is a polar implicit curve rather than a distance field, so its gradient
    is not a unit normal -- harmless for a shape this small, and the bevel is
    scaled to match its actual size.
    """
    dx = np.asarray(X, dtype=np.float64) - cx
    dy = np.asarray(Y, dtype=np.float64) - cy
    d = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    return d - boss_radius(theta, r0, harmonics, amp)


class Panel:
    """One panel: the fields, the corner boxes, and the content rectangle."""

    def __init__(self, spec: dict, size, scale: float = 1.0):
        self.w, self.h = _size(size)
        self.scale = float(scale)
        self.spec = stylelib.clamp(spec, (self.w / self.scale, self.h / self.scale))
        # Design pixels == world units; the canvas scale maps them to output px.
        self.hw = self.w / (2.0 * self.scale)
        self.hh = self.h / (2.0 * self.scale)

    # ---------------------------------------------------------------- basics
    @property
    def design_size(self) -> tuple:
        """The panel's size in design pixels."""
        return (self.w / self.scale, self.h / self.scale)

    @property
    def inset(self) -> float:
        """The nine-slice cut, in design pixels (= world units)."""
        return float(self.spec["inset"])

    def content(self) -> tuple:
        """The rect the frame leaves for content, as pixel bounds ``(x0, y0, x1, y1)``.

        Pixel bounds, top-left origin -- the one place in this package where the
        output pixel grid is the right coordinate system, because that is where a
        layout engine will put the text.
        """
        m = stylelib.pad(self.spec) * self.scale
        x0 = int(round(m))
        y0 = int(round(m))
        return (x0, y0, max(self.w - x0, x0 + 1), max(self.h - y0, y0 + 1))

    def corner_box(self, i: int) -> tuple:
        """The world rect ``(x0, y0, x1, y1)`` of corner ``i``'s ornament box.

        ``y1`` is the top, matching :attr:`core.canvas.Canvas.grid`.  The box
        reaches *inwards* from the panel corner by ``inset`` -- the ornament box
        and the nine-slice corner block are the same square, so this is also the
        region of the design that a resample never touches.

        Everything the design curves -- the radius, the bevel, the ornament --
        has to be inside it, which is exactly what the tests measure.
        """
        sx, sy = CORNERS[i]
        cx = sx * self.hw
        cy = sy * self.hh
        return (min(cx, cx - sx * self.inset), min(cy, cy - sy * self.inset),
                max(cx, cx - sx * self.inset), max(cy, cy - sy * self.inset))

    # ---------------------------------------------------------------- fields
    def outer(self, X, Y):
        return sdf.box(X, Y, self.hw, self.hh, self.spec["radius"])

    def inner(self, X, Y):
        b = self.spec["border"]
        return sdf.box(X, Y, self.hw - b, self.hh - b, max(self.spec["radius"] - b, 0.0))

    def body(self, X, Y):
        """The frame band: the outer silhouette minus the inner rectangle."""
        return np.maximum(self.outer(X, Y), -self.inner(X, Y))

    def rule(self, X, Y):
        """The inner moulding, as a stroke around the third rounded rectangle."""
        g = self.spec["border"] + self.spec["rule_gap"]
        d = sdf.box(X, Y, self.hw - g, self.hh - g, max(self.spec["radius"] - g, 0.0))
        return np.abs(d) - 0.5 * self.spec["rule"]

    def boss_centre(self, i: int) -> tuple:
        """Where corner ``i``'s ornament sits: on the diagonal, on the band."""
        sx, sy = CORNERS[i]
        # A fraction of the frame width out along the diagonal, so a disc of
        # about that width straddles the moulding instead of hanging over the
        # hole -- which is what lets a hollow frame wear one too.
        o = stylelib.BOSS_O * self.spec["border"] / np.sqrt(2.0)
        return (sx * (self.hw - o), sy * (self.hh - o))

    def boss_radius_at(self, i: int) -> float:
        """The ornament's base radius in design pixels; 0 when it is disabled."""
        return stylelib.BOSS_R0 * self.spec["border"] * self.spec["boss"]

    def boss_outline(self, i: int) -> float:
        """Half-width of the hairline drawn around the ornament."""
        return stylelib.BOSS_OUTLINE * self.boss_radius_at(i)

    def boss(self, i: int, X, Y):
        cx, cy = self.boss_centre(i)
        return boss_field(X, Y, cx, cy, self.boss_radius_at(i),
                          self.spec["boss_harmonics"], self.spec["boss_amp"])

    # ----------------------------------------------------------- convenience
    def has_rule(self) -> bool:
        """Whether the inner moulding exists and there is room for it."""
        g = self.spec["border"] + self.spec["rule_gap"] + 0.5 * self.spec["rule"]
        return self.spec["rule"] > 0.0 and g < min(self.hw, self.hh)

    def has_boss(self) -> bool:
        return self.boss_radius_at(0) > 0.25

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        d = self.design_size
        return f"Panel({d[0]:g}x{d[1]:g}@x{self.scale:g}, inset={self.inset:g})"


def _size(size) -> tuple:
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    return int(size), int(size)
