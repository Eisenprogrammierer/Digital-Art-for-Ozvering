"""The badge: a count or a tier, worn on an icon's corner.

An icon is a plate with a charge, and a charge is what a *kind* of thing looks
like.  It cannot say *how many* or *how good* -- a coin and a purse of forty coins
are the same coin.  That is a second question, and it gets a second, smaller
answer: a **badge** pinned to the plate's lower-right corner, which is the same
construction one level down, a small plate with a small charge on it.

Two questions, two marks, and the difference is not decoration:

* a **quantity** is a count, and counts go high, so it is a *numeral* -- seven
  bars in the arrangement a calculator uses.  A digit made of seven bars is a
  symbol for a numeral the way a ring is a symbol for a coin, and it survives a
  small size where a serif would not;
* a **tier** is an ordinal rank, and ranks are few, so it is a row of **pips**.  A
  "2" in a corner is ambiguous the moment the icon also carries a quantity; two
  dots under it are not.

The badge is drawn in the icon's own two colours -- a bright disc in the *charge*
colour with its marks knocked into it in the plate's dark -- so it adds no colour
to the set and no fifth thing for a player to learn.  It is allowed to break the
plate's silhouette (a hexagon and a shield do not reach their own corners), and
that is on purpose: a badge that stays wholly inside the plate is a decoration,
and the point of a badge is that it is *noticed*.

Readability at 32 px is treated as a measurement, like everywhere else in this
package -- but the honest measurement is that a *two-digit* numeral at 32 px is a
mark rather than a number.  The badge is built to read at the sizes a count is
actually consulted at (48 px and up), and the tests hold it to what it can promise
at 32: it is there, it is separable from the plate, and it never costs the icon
contrast.
"""
from __future__ import annotations

import numpy as np

from core import render, sdf
from core.render import LIGHT

__all__ = ["LIGHT", "BADGE", "CORNER", "TIERS", "MAX_COUNT", "SHADOW", "make",
           "clamp", "field", "marks", "paint"]

#: What the badge's seat -- the dark ring that separates it from whatever is under
#: it -- is drawn in.  The icon's own shadow colour, so a badge and a plate throw
#: the same darkness.
SHADOW = (0.03, 0.03, 0.05)

#: The badge's radius, as a fraction of the plate's margin, and where its centre
#: sits (also in units of the margin).  Both are fractions rather than lengths so a
#: badge re-renders at another size as the same design, the same reason every other
#: number in this package is in world units.  The corner is past where a hexagon or
#: a shield ends, which is exactly the overlap that makes the badge read.
BADGE = 0.30
CORNER = (0.60, -0.60)

#: How wide the dark seat shows around the bright disc, as a fraction of the badge.
SEAT = 0.16

#: The marks' geometry, as fractions of the badge radius.  A seven-segment digit is
#: half-width, half-height and stroke; the segments are capsules, so a *joining*
#: corner is where two capsules overlap, which is why the verticals stop a little
#: short of the horizontals instead of being cut to meet them.
GLYPH_W, GLYPH_H, GLYPH_T = 0.27, 0.46, 0.075
GLYPH_GAP = 0.065          # between two digits of one numeral
PIP = 0.19                 # a pip's radius
PIP_GAP = 1.40             # pip spacing, in pip diameters

#: A tier is a row of pips, so it is bounded by what still reads as a row: three to
#: a line, two lines.  A count is a numeral, bounded by what a badge can hold:
#: two digits.  Either bound is a *refusal* rather than a silent round, because a
#: badge that shows "99" for a hundred coins is a badge that lies.
TIERS = 5
MAX_COUNT = 99

#: The seven segments, and which of them each digit lights.  ``a`` is the top bar
#: and the rest run clockwise from it, with ``g`` in the middle -- the arrangement
#: every seven-segment display uses, written out once.
_DIGITS = ("abcdef", "bc", "abdeg", "abcdg", "bcfg",
           "acdfg", "acdefg", "abc", "abcdefg", "abcdfg")


# ------------------------------------------------------------------- the spec
def _whole(value, what: str) -> int:
    """A whole number, or an exception -- a bool is not one, and neither is "3"."""
    if isinstance(value, bool):
        raise TypeError(f"{what} must be a whole number, not {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a whole number, not {value!r}") from None
    if number != value:
        raise ValueError(f"{what} must be a whole number, not {value!r}")
    return number


def clamp(value) -> dict | None:
    """Force a badge onto the legal set, or ``None`` for "no badge".

    The same contract ``paint.clamp`` makes for an icon and ``gen.ui.material``
    makes for a name: a badge that carries *nothing* is no badge (a count of zero is
    zero of a thing, not a badge showing nothing), while a badge that carries
    something it cannot draw is refused rather than repaired -- a count of a
    thousand shown as "99" is worse than an exception, because it is a lie a player
    would read.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"a badge is a mapping like {{'count': 3}}, not {value!r}")
    count = value.get("count")
    tier = value.get("tier")
    if count is not None and tier is not None:
        raise ValueError("a badge carries a count or a tier, not both")
    if count is not None:
        number = _whole(count, "a count")
        if number <= 0:
            return None
        if number > MAX_COUNT:
            raise ValueError(f"a badge shows at most {MAX_COUNT}; "
                             f"{number} is refused, not rounded")
        return {"count": number}
    if tier is not None:
        number = _whole(tier, "a tier")
        if number <= 0:
            return None
        if number > TIERS:
            raise ValueError(f"a tier is 1 to {TIERS} pips; for more, ask for a count, "
                             f"not {number}")
        return {"tier": number}
    return None


def make(count=None, tier=None) -> dict | None:
    """A badge showing a ``count`` or a ``tier`` -- the constructor a caller uses."""
    return clamp({"count": count, "tier": tier})


# ---------------------------------------------------------------- the geometry
def _centre(margin: float) -> tuple:
    """The badge's centre for a plate of the given margin, in world units."""
    return CORNER[0] * margin, CORNER[1] * margin


def _digit(x, y, digit: int, hw: float, hh: float, t: float):
    """One seven-segment digit, centred on the local origin."""
    lit = _DIGITS[int(digit)]
    short = 1.7 * t                        # where a vertical stops short of the middle
    parts = []
    if "a" in lit:
        parts.append(sdf.segment(x, y, (-(hw - t), hh), (hw - t, hh), t))
    if "g" in lit:
        parts.append(sdf.segment(x, y, (-(hw - t), 0.0), (hw - t, 0.0), t))
    if "d" in lit:
        parts.append(sdf.segment(x, y, (-(hw - t), -hh), (hw - t, -hh), t))
    if "f" in lit:
        parts.append(sdf.segment(x, y, (-hw, hh - t), (-hw, short), t))
    if "e" in lit:
        parts.append(sdf.segment(x, y, (-hw, -short), (-hw, -(hh - t)), t))
    if "b" in lit:
        parts.append(sdf.segment(x, y, (hw, hh - t), (hw, short), t))
    if "c" in lit:
        parts.append(sdf.segment(x, y, (hw, -short), (hw, -(hh - t)), t))
    body = parts[0]
    for part in parts[1:]:
        body = sdf.union(body, part)
    return body


def _numeral(x, y, count: int, r: float):
    """A count as a one- or two-digit numeral, centred on the badge."""
    hw, hh, t = GLYPH_W * r, GLYPH_H * r, GLYPH_T * r
    digits = str(int(count))
    if len(digits) == 1:
        return _digit(x, y, int(digits), hw, hh, t)
    step = hw + t + GLYPH_GAP * r
    return sdf.union(_digit(x - step, y, int(digits[0]), hw, hh, t),
                     _digit(x + step, y, int(digits[1]), hw, hh, t))


def _rows(n: int) -> list:
    """``n`` pips split into as few rows of at most three as they need."""
    rows = -(-n // 3)
    base, extra = divmod(n, rows)
    return [base + (1 if i < extra else 0) for i in range(rows)]


def _pips(x, y, tier: int, r: float):
    """A tier as a row (or two) of pips, centred on the badge."""
    radius = PIP * r
    step = 2.0 * PIP_GAP * radius
    sizes = _rows(int(tier))
    body = None
    for row, k in enumerate(sizes):
        ry = (row - (len(sizes) - 1) / 2.0) * -step     # the first row is the top one
        for i in range(k):
            pip = sdf.circle(x, y, radius, (i - (k - 1) / 2.0) * step, ry)
            body = pip if body is None else sdf.union(body, pip)
    return body


def field(spec, X, Y, margin):
    """The badge's own solid: a disc at the plate's corner.

    ``margin`` is the *plate's* margin, so the badge is placed and sized by the
    plate it is pinned to rather than by the tile -- which is what keeps a badge on
    a small plate from being a badge off the edge of one.  The spec decides nothing
    here: every badge is the same disc, and the spec is a parameter only so a caller
    has the same signature as :func:`marks`.
    """
    r = BADGE * float(margin)
    cx, cy = _centre(float(margin))
    return sdf.circle(X, Y, r, cx, cy)


def marks(spec, X, Y, margin):
    """The mark on the badge: a numeral for a count, pips for a tier.

    Returns an empty (all-positive) field for no badge, so a caller may ask for
    either badge there is and get nothing rather than an exception.
    """
    badge = clamp(spec)
    if badge is None:
        return np.full(np.broadcast(X, Y).shape, np.inf)
    r = BADGE * float(margin)
    cx, cy = _centre(float(margin))
    if "count" in badge:
        return _numeral(X - cx, Y - cy, badge["count"], r)
    return _pips(X - cx, Y - cy, badge["tier"], r)


# ------------------------------------------------------------------ the passes
def paint(canvas, spec, margin, X, Y, dx, tint) -> None:
    """Draw the badge on ``canvas``: a seat, a disc, and the mark knocked into it.

    The three passes are the icon's own construction at one size down -- a shadow,
    a ground, a mark -- and the colours are the icon's, so a badge is a part of the
    icon rather than a sticker on the tile.  ``tint`` is the icon's four-colour
    table, straight from :func:`gen.icons.paint.tints`.
    """
    badge = clamp(spec)
    if badge is None:
        return
    margin = float(margin)
    bevel = BADGE * margin * 0.22       # a disc-sized bevel

    d_disc = field(badge, X, Y, margin)
    a_disc = render.coverage(d_disc, dx)
    d_seat = sdf.circle(X, Y, BADGE * margin * (1.0 + SEAT), *_centre(margin))
    a_seat = render.coverage(d_seat, dx)

    # A small shadow and the dark seat, both in world units, so a badge separates
    # from the plate at every size rather than only at the size it was tuned at.
    canvas.over_srgb(SHADOW, render.drop_shadow(a_seat, canvas.scale, offset=(0.0, -0.02),
                                                sigma=0.022, opacity=0.38))
    canvas.over_srgb(SHADOW, a_seat)

    # The disc, in the icon's charge colour: a badge adds no colour to the set.
    canvas.over(render.shade(tint["charge"], d_disc, dx, bevel=bevel, light=LIGHT), a_disc)

    # The mark, knocked *into* the bright disc in the plate's dark, so it reads at
    # small sizes the way the recess does on the icon itself.
    d_marks = marks(badge, X, Y, margin)
    a_marks = render.coverage(d_marks, dx) * a_disc
    canvas.over(render.shade(tint["recess"], d_marks, dx, bevel=bevel * 0.6, light=LIGHT,
                             inset=True), a_marks)
