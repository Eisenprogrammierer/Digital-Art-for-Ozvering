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

__all__ = ["Panel", "boss_field", "boss_radius", "radius_fit"]

#: The four corner directions, counter-clockwise from the top left.
CORNERS = ((-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))


def radius_fit(radius: float, hw: float, hh: float) -> float:
    """A corner radius can never exceed the box it rounds.

    Past ``min(hw, hh)`` :func:`core.sdf.box` stops being a distance field: the
    corner arcs cross over each other and the "rounded rectangle" becomes a shape
    with no exact vector form -- and no exact bevel either, since the bevel is
    computed from the field's gradient.  A small panel with a wide title band is
    where it happens, because the band eats the inner box from both sides at once,
    so the radius gives way the way the rule gives way at a size too small for the
    frame: the *look* yields, the box is still the box.
    """
    return float(min(max(float(radius), 0.0),
                     max(float(hw), 0.0), max(float(hh), 0.0)))


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
    def insets(self) -> tuple:
        """The nine-slice cuts, in design pixels: ``(left, top, right, bottom)``.

        Per side, because a titled window's corners are not squares: the top-left
        block is as wide as the left band and as tall as the title.  That is also
        what an engine has to be handed -- a nine-slice is four numbers, and a panel
        that has a title bar is the case that makes the difference visible.
        """
        return tuple(float(v) for v in self.spec["inset"])

    @property
    def inset(self) -> float:
        """The tightest of the four sides, as a scalar.

        The curvature, the moulding and the ornament are shared by all four
        corners, so this is the number that bounds them; a caller that wants the
        cuts wants :attr:`insets`.
        """
        return min(self.insets)

    def content(self) -> tuple:
        """The rect the frame leaves for content, as pixel bounds ``(x0, y0, x1, y1)``.

        Pixel bounds, top-left origin -- the one place in this package where the
        output pixel grid is the right coordinate system, because that is where a
        layout engine will put the text.  A title bar pushes the top of it down and
        leaves the sides where they were, which is exactly what a title bar is for.
        """
        m = stylelib.pad(self.spec)
        x0 = int(round(m[0] * self.scale))
        y0 = int(round(m[1] * self.scale))
        x1 = max(self.w - int(round(m[2] * self.scale)), x0 + 1)
        y1 = max(self.h - int(round(m[3] * self.scale)), y0 + 1)
        return (x0, y0, min(x1, self.w), min(y1, self.h))

    def corner_box(self, i: int) -> tuple:
        """The world rect ``(x0, y0, x1, y1)`` of corner ``i``'s ornament box.

        ``y1`` is the top, matching :attr:`core.canvas.Canvas.grid`.  The box
        reaches *inwards* from the panel corner by that corner's two insets, which
        are the same two numbers as its nine-slice block -- so this is also the
        region of the design a resample never touches.

        Everything the design curves -- the radius, the bevel, the ornament --
        has to be inside it, which is exactly what the tests measure.
        """
        sx, sy = CORNERS[i]
        ix, iy = stylelib._corner_pair(i, self.insets)
        cx = sx * self.hw
        cy = sy * self.hh
        return (min(cx, cx - sx * ix), min(cy, cy - sy * iy),
                max(cx, cx - sx * ix), max(cy, cy - sy * iy))

    def corner_border(self, i: int) -> float:
        """The band width at corner ``i``: the thinner of its two sides.

        The ornament is a disc, and a disc on a band has one radius, so the thinner
        of the two bands is what it fits.
        """
        return min(stylelib._corner_pair(i, self.spec["border"]))

    # ---------------------------------------------------------------- fields
    def outer_radius(self) -> float:
        """The silhouette's corner radius.

        Through :func:`radius_fit` like the other two, because at a size too small
        for the style ``clamp`` cannot bound it by the box (it keeps a 4-pixel floor
        for a design's corner box) and then the field stops being a distance field.
        A radius is *derived* here, so the fit has one home rather than three.
        """
        return radius_fit(self.spec["radius"], self.hw, self.hh)

    def inner_radius(self) -> float:
        """The inner rectangle's corner radius: the outer one, worn down by the band.

        Bounded by the inner box, because a title bar can eat that box down to less
        than the band is wide -- see :func:`radius_fit`.
        """
        _, _, hw, hh = self.inner_box()
        return radius_fit(self.spec["radius"] - min(self.spec["border"]), hw, hh)

    def rule_radius(self) -> float:
        """The moulding's corner radius: the inner one again, moved in by the gap."""
        g = self.spec["rule_gap"]
        _, _, hw, hh = self.inner_box()
        return radius_fit(self.inner_radius() - g, hw - g, hh - g)

    def inner_box(self) -> tuple:
        """The inner rectangle as ``(cx, cy, hw, hh)``, given the per-side bands.

        A box with four different bands is still a box: it is the outer one shifted
        by half the difference between opposite sides, and shrunk by half their sum.
        Writing it that way means the symmetric case is the arithmetic it always
        was, down to the same floating-point result.
        """
        l, t, r, b = self.spec["border"]
        return (0.5 * (l - r), 0.5 * (b - t),
                self.hw - 0.5 * (l + r), self.hh - 0.5 * (t + b))

    def has_hole(self) -> bool:
        """Whether the inner rectangle is real.

        A panel can be smaller than its own frame -- a 3-pixel design with a 4-pixel
        band, which ``clamp``'s corner-box floor permits -- and then the inner box
        turns inside out and the band is the whole silhouette rather than a ring.
        That is a shape change and not a detail: there is no hole to cut, so nothing
        derived from the hole (the rule, the vector band) may pretend there is one.
        """
        _, _, hw, hh = self.inner_box()
        return hw > 0.0 and hh > 0.0

    def outer(self, X, Y):
        return sdf.box(X, Y, self.hw, self.hh, self.outer_radius())

    def inner(self, X, Y):
        cx, cy, hw, hh = self.inner_box()
        return sdf.box(X - cx, Y - cy, hw, hh, self.inner_radius())

    def body(self, X, Y):
        """The frame band: the outer silhouette minus the inner rectangle."""
        return np.maximum(self.outer(X, Y), -self.inner(X, Y))

    def rule(self, X, Y):
        """The inner moulding, as a stroke around the third rounded rectangle."""
        g = self.spec["rule_gap"]
        cx, cy, hw, hh = self.inner_box()
        d = sdf.box(X - cx, Y - cy, hw - g, hh - g, self.rule_radius())
        return np.abs(d) - 0.5 * self.spec["rule"]

    def boss_centre(self, i: int) -> tuple:
        """Where corner ``i``'s ornament sits: on the diagonal, on the band."""
        sx, sy = CORNERS[i]
        # A fraction of the frame width out along the diagonal, so a disc of
        # about that width straddles the moulding instead of hanging over the
        # hole -- which is what lets a hollow frame wear one too.
        o = stylelib.BOSS_O * self.corner_border(i) / np.sqrt(2.0)
        return (sx * (self.hw - o), sy * (self.hh - o))

    def boss_radius_at(self, i: int) -> float:
        """The ornament's base radius in design pixels; 0 when it is disabled."""
        return stylelib.BOSS_R0 * self.corner_border(i) * self.spec["boss"]

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
        g = min(self.spec["border"]) + self.spec["rule_gap"] + 0.5 * self.spec["rule"]
        return self.spec["rule"] > 0.0 and g < min(self.hw, self.hh)

    def has_boss(self) -> bool:
        return self.boss_radius_at(0) > 0.25

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        d = self.design_size
        return f"Panel({d[0]:g}x{d[1]:g}@x{self.scale:g}, inset={self.insets})"


def _size(size) -> tuple:
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    return int(size), int(size)
