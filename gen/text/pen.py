"""The pen: what turns a skeleton into a letter.

Five numbers, and nothing else:

* **weight** -- the width of a stem, in em units.
* **contrast** -- how much a stroke thins as it turns towards horizontal.  A pen held
  at an angle gives a letter thick stems and thin hairlines; held vertically it gives
  a monoline.  This is the *whole* of a letter's modulation, because the width is
  read off each segment's own direction.
* **serif** -- whether the pen grows feet where a stroke *ends* on a line of the face.
* **slant** -- the shear applied to the skeleton before any of the above.
* **width** -- an overall horizontal scale, because a narrow face is a face.

None of them can change *which letter* this is: that is the skeleton's business, and
it is why one alphabet serves a monoline UI face and a modulated text face.

Two consequences of building letters this way are worth stating, because both are
things a font does and both fall out rather than being drawn in:

* **The pen is wider than the line.**  A centreline touches the x-height, so the ink
  around it overshoots by half a stem -- and a round letter overshoots the baseline
  the same way, which is why :func:`ink_box` measures the skeleton *and* the pen
  reach instead of trusting the guides.
* **A serif is a rule, not a shape.**  A serif grows at a stroke end that lands on
  one of the face's four horizontals, and it is a bar *perpendicular to the stroke*:
  a stem ending on the baseline gets a foot along it, an arm ending on the cap line
  gets a beak across it.  That is what a serif is, it is derivable from the skeleton,
  and it means the data file has no serifs in it at all.

The third thing that falls out is the *metrics*, and it is the one this module insists
on because a font is asked to lay text out and not only to draw it.  An advance is not
a number a letter carries around: it is measured here, off the letter's **body** plus a
sidebearing, because a stem at the edge of the design hangs past it and a declared
width is therefore a promise the pen can break -- exactly when the weight goes up.
The serifs are the subtlety: a serif is centred on its stem, so half of it lies outside
the body, and the metal-type answer is the one that works -- the foot hangs *into* the
sidebearing rather than widening the letter.  :func:`advance`, :func:`offset` and
:func:`overhang` read the result of the pen, and a test holds all 95 characters to it
at four weights and three serif amounts.

The joins are fillets, not corners: the pen unions its segments with the same
smooth-min the rest of the toolkit welds shapes with, so a blunted apex or a
shoulder that needs two arcs to draw comes out as one solid rather than a seam.  The
fillet is a *small* fraction of a stem on purpose -- a smooth union pulls the boundary
outward by about a quarter of its radius, so a fillet as thick as a stem does not round
a letter's corners, it swells the whole letter, and a bowl of thirty segments swells
thirty times.
"""
from __future__ import annotations

import numpy as np

from core import sdf
from core.render import LIGHT
from gen.text import skeleton

__all__ = ["DEFAULTS", "PRESETS", "LINES", "SERIF_LEN", "SERIF_THICK", "SERIF_ANGLE",
           "SIDEBEARING", "AIR", "FILLET", "LIGHT", "names", "get", "clamp", "validate",
           "thickness", "glyph_field", "ink_box", "core_box", "overhang", "advance",
           "offset", "strokes"]

#: The five faces that come for free.  The skeleton is the same for all of them.
PRESETS = {
    # a monoline: for a control, a tooltip, anything that has to read at 12 px.
    "sans": {"weight": 0.088, "contrast": 0.0, "serif": 0.0, "slant": 0.0, "width": 1.0},
    # a book face: thick stems, thin hairlines, feet.
    "text": {"weight": 0.082, "contrast": 0.32, "serif": 0.90, "slant": 0.0, "width": 0.98},
    # a headline: heavier, and the contrast opens up because it can be seen.
    "display": {"weight": 0.120, "contrast": 0.42, "serif": 0.70, "slant": 0.0, "width": 0.96},
    # the slanted book face: the same letter, sheared, with the feet left level.
    "italic": {"weight": 0.082, "contrast": 0.30, "serif": 0.75, "slant": 0.17, "width": 0.98},
    # thick, low contrast, big feet: a slab.
    "slab": {"weight": 0.108, "contrast": 0.10, "serif": 1.00, "slant": 0.0, "width": 1.02},
}

#: What an unqualified request gets.
DEFAULTS = PRESETS["text"]

#: A serif is this many stems long and this many thick, at ``serif = 1``.
SERIF_LEN = 2.6
SERIF_THICK = 0.62

#: The daylight a letter's *body* keeps on each side, in em units.  This is the whole
#: of the spacing system, and it is measured off the stems rather than off every last
#: serif -- see :func:`advance`.
SIDEBEARING = 0.05

#: The air a serif keeps from the edge of its neighbour's ground.  A serif is allowed
#: to hang into the sidebearing, and this is how far past the sidebearing it may reach
#: before the advance grows to make room.
AIR = 0.02

#: The fillet the joins get, as a fraction of a stem.  Small on purpose: a
#: smooth union pulls the boundary *out* by about a quarter of its radius, so a
#: fillet as big as a stem does not round the corners of a letter -- it swells
#: the letter, and thirty segments of a bowl swell it thirty times.
FILLET = 0.15

#: How far from vertical or horizontal a stroke may arrive and still be called a
#: stem or an arm.  Eighteen degrees excludes a slash, which is the one punctuation
#: mark whose ends are on the cap line and must not grow feet.
SERIF_ANGLE = 18.0
_COS_SERIF = float(np.cos(np.radians(SERIF_ANGLE)))

#: Which guide a serif may grow on, and how close the end has to land to it.  The
#: tolerance is in em units and is tight, because the skeletons are drawn on the
#: guides exactly: an end that is a hundredth off is an arc that happens to stop
#: nearby, not a terminal.
_GUIDE_TOL = 0.03

#: The lines a serif is allowed on, indexed.  The x-height is index 1 and is the one
#: horizontal an *arm* is not serifed on: a bowl springs from the x-height
#: tangentially, and a brush stroke landing there is a curve, not a terminal.
LINES = (("baseline", skeleton.GUIDES[0]), ("x-height", skeleton.GUIDES[1]),
         ("cap", skeleton.GUIDES[2]), ("ascender", skeleton.GUIDES[3]))


def names() -> list:
    """The preset faces, in a stable order."""
    return list(PRESETS)


def get(name: str) -> dict:
    """One preset by name.  A name nobody wrote is an error, not a default."""
    if name not in PRESETS:
        raise KeyError(f"no face called {name!r}: the presets are {tuple(PRESETS)}")
    return dict(PRESETS[name])


def clamp(spec) -> dict:
    """A pen spec with every number in range.

    The numbers are clipped rather than refused: an impossible weight is a request
    with a bad number in it, and a bad number has a nearest sensible one.  A bad
    *name* is a different thing and is refused by :func:`get`.
    """
    out = {}
    for key, value in DEFAULTS.items():
        lo, hi = _RANGE[key]
        try:
            v = float((spec or {}).get(key, value))
        except (TypeError, ValueError):
            v = float(value)
        out[key] = float(np.clip(v, lo, hi))
    return out


#: The range each number is clipped to.  A weight of zero is a letter nobody can see
#: and one of a third of an em is a letter nobody can read.
_RANGE = {
    "weight": (0.025, 0.28),
    "contrast": (0.0, 0.85),
    "serif": (0.0, 1.5),
    "slant": (-0.35, 0.35),
    "width": (0.55, 1.7),
}


def validate(spec) -> list:
    """What is wrong with ``spec``, as readable lines.  Empty means legal."""
    bad = []
    for key, value in sorted(dict(spec or {}).items()):
        if key not in DEFAULTS:
            bad.append(f"{key}: not a pen parameter (one of {tuple(DEFAULTS)})")
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            bad.append(f"{key}: {value!r} is not a number")
            continue
        lo, hi = _RANGE[key]
        if not lo <= v <= hi:
            bad.append(f"{key}: {v:g} is outside [{lo:g}, {hi:g}]")
    return bad


# --------------------------------------------------------------------------
# the pen itself
# --------------------------------------------------------------------------
def thickness(dx: float, dy: float, weight: float, contrast: float) -> float:
    """The width the pen lays down for a stroke in direction ``(dx, dy)``.

    Full width on a vertical stroke, thinnest on a horizontal one, and in between
    for a diagonal -- which is all a broad nib does.
    """
    n = float(np.hypot(dx, dy))
    across = abs(float(dx)) / n if n > 0.0 else 0.0
    return float(weight) * (1.0 - float(contrast) * across)


def _shear(pts, s: dict):
    """The skeleton points after the pen's width and slant have had their say."""
    w, sl = s["width"], s["slant"]
    if w == 1.0 and sl == 0.0:
        return pts
    return [(x * w + sl * y, y) for x, y in pts]


def _guide(y: float):
    """The index of the line ``y`` lands on, or ``None``."""
    for i, (_name, line) in enumerate(LINES):
        if abs(y - line) <= _GUIDE_TOL:
            return i
    return None


def _serifs(pts, s: dict) -> list:
    """The bars this stroke's ends have earned, as ``(cx, cy, hw, hh)`` boxes."""
    if s["serif"] <= 0.0 or len(pts) < 2:
        return []
    length = s["serif"] * SERIF_LEN * s["weight"]
    thick = SERIF_THICK * s["weight"]
    out = []
    for end, inside in ((pts[-1], pts[-2]), (pts[0], pts[1])):
        dx, dy = end[0] - inside[0], end[1] - inside[1]
        n = float(np.hypot(dx, dy))
        if n < 1e-12:
            continue
        ux, uy = abs(dx) / n, abs(dy) / n
        line = _guide(end[1])
        if line is None:
            continue
        if uy >= _COS_SERIF:
            out.append((end[0], end[1], length * 0.5, thick * 0.5))       # a foot
        elif ux >= _COS_SERIF and line != 1:
            out.append((end[0], end[1], thick * 0.5, length * 0.5))       # a beak
    return out


def strokes(ch: str, spec=None):
    """The glyph's skeleton, sheared, ready for the pen.  Handy for measuring."""
    return [_shear(pts, clamp(spec)) for pts in skeleton.strokes(ch)]


def glyph_field(ch: str, spec, X, Y):
    """The distance field of one glyph, in em units, baseline-left at ``(0, 0)``.

    The caller evaluates it on whatever grid it likes; the field is the same field at
    every size, which is the whole reason a letter can be *drawn* at 300 px and at
    12 px instead of being drawn large and resampled.
    """
    s = clamp(spec)
    weight, contrast = s["weight"], s["contrast"]
    k = FILLET * weight                    # the fillet the joins get
    reach = 0.5 * weight
    d = None
    for pts in strokes(ch, s):
        if len(pts) == 1:
            seg = sdf.circle(X, Y, reach, pts[0][0], pts[0][1])
            d = seg if d is None else sdf.smooth_union(d, seg, k)
            continue
        for a, b in zip(pts, pts[1:]):
            r = 0.5 * thickness(b[0] - a[0], b[1] - a[1], weight, contrast)
            seg = sdf.segment(X, Y, a, b, r)
            d = seg if d is None else sdf.smooth_union(d, seg, k)
        for cx, cy, hw, hh in _serifs(pts, s):
            seg = sdf.box(X, Y, hw, hh, 0.0, cx, cy)
            d = seg if d is None else sdf.smooth_union(d, seg, k)
    if d is None:
        return np.full(np.broadcast(X, Y).shape, np.inf, dtype=np.float64)
    return d


def _extent(ch: str, s: dict, serifs: bool = True) -> tuple:
    """``(x0, y0, x1, y1)`` of what the pen draws for ``ch``, in em units.

    With ``serifs`` false the bars are left out, which is the difference between the
    *body* of a letter and its finished outline -- and it is the body that spacing is
    measured from.

    Two kinds of thing reach the edge here and they are not the same kind.  A stroke
    is a *centreline*, so the ink around it overshoots by half a stem.  A serif is a
    **box**, drawn at its own size, so it overshoots by nothing -- charging a serif the
    pen's radius as well is how a foot gets measured half a stem too wide, and an
    ``i`` ends up in a slot with a hole on either side of it.
    """
    xs, ys, bx, by = [], [], [], []
    for pts in strokes(ch, s):
        if not pts:
            continue
        for x, y in pts:
            xs.append(x)
            ys.append(y)
        if serifs:
            for cx, cy, hw, hh in _serifs(pts, s):
                bx.extend((cx - hw, cx + hw))
                by.extend((cy - hh, cy + hh))
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    r = 0.5 * s["weight"]
    x0, x1 = min(xs) - r, max(xs) + r
    y0, y1 = min(ys) - r, max(ys) + r
    if bx:
        x0, x1 = min(x0, min(bx)), max(x1, max(bx))
        y0, y1 = min(y0, min(by)), max(y1, max(by))
    return (x0, y0, x1, y1)


def ink_box(ch: str, spec=None) -> tuple:
    """Where the glyph's ink reaches, ``(x0, y0, x1, y1)`` in em units.

    Measured off the skeleton and the pen rather than declared, so it follows a
    weight change, a shear and a serif without anybody remembering to update a table.
    It is the box a renderer wants for two things: where to evaluate the field (so a
    line of text costs a window per letter and not a pass over the sheet), and where
    the ink actually is (so a line can be centred on what the eye sees).
    """
    return _extent(ch, clamp(spec), True)


def core_box(ch: str, spec=None) -> tuple:
    """The box of the letter's *body*: the same, with the serifs left off."""
    return _extent(ch, clamp(spec), False)


def overhang(ch: str, spec=None) -> tuple:
    """How far the serifs reach past the body, ``(left, right)`` in em units.

    Zero for a monoline, and the number that decides how much ground a serif needs:
    a foot is centred on its stem, so half of it lands outside the body, and the
    sidebearing has to be at least that wide or two letters share a serif.
    """
    s = clamp(spec)
    if not skeleton.strokes(ch):
        return (0.0, 0.0)
    core, full = _extent(ch, s, False), _extent(ch, s, True)
    return (core[0] - full[0], full[2] - core[2])


def _sides(ch: str, s: dict) -> tuple:
    """The left and right sidebearings this glyph's pen needs, in em units.

    The body's daylight, or as much room as the serifs ask for and :data:`AIR` more --
    whichever is larger.  This is the one place the serif rule turns into a number, and
    it is why a foot can be big without a narrow letter growing a wide advance: the
    serif hangs into the sidebearing the letter already had.
    """
    over = overhang(ch, s)
    return (max(SIDEBEARING, over[0] + AIR), max(SIDEBEARING, over[1] + AIR))


def advance(ch: str, spec=None) -> float:
    """How far the pen moves after ``ch``, in em units.

    The larger of the letter's *design* width and what the body needs: the body plus
    its sidebearings.  A design width the body overruns is not a width, so the pen
    wins -- but the **serifs are not part of the body**, and that is the whole of this
    function's judgement.  A serif is centred on its stem, so half of it lies outside
    the body; measuring an advance off the serif tips charges every letter for its
    feet twice, and an ``i`` -- whose body is one stem wide and whose feet are the
    width of an ``n``'s -- ends up in a slot half again as wide as its own ink, which
    is a hole in the middle of a word rather than a letter.

    So: the body sets the advance, the serifs hang into the sidebearing, and the
    advance only grows when a serif would reach past the air it is allowed
    (:func:`_sides`).  A test holds the outcome: no glyph's finished ink leaves the
    ground its advance gives it, at four weights and three serif amounts.
    """
    s = clamp(spec)
    if not skeleton.strokes(ch):
        return skeleton.advance(ch) * s["width"]
    core = _extent(ch, s, False)
    lsb, rsb = _sides(ch, s)
    return max(skeleton.advance(ch) * s["width"], (core[2] - core[0]) + lsb + rsb)


def offset(ch: str, spec=None) -> float:
    """How far right a glyph is drawn so its left sidebearing is honoured.

    The origin the caller thinks in is the *pen's*, and a glyph whose body starts left
    of it is shifted until the sidebearing is the room the pen asked for -- so the ink
    always starts inside the ground the advance gave it.
    """
    s = clamp(spec)
    if not skeleton.strokes(ch):
        return 0.0
    return _sides(ch, s)[0] - _extent(ch, s, False)[0]
