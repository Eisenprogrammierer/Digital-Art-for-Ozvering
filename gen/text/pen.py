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
:func:`overhang` read the result of the pen, and a test holds every character in the
face to it at four weights and three serif amounts.

Marks are the same argument one step further out.  A letter with an accent on it is the
width of the letter, so the accent is fitted into the ground the advance already gives
it (:func:`_mark_of`) -- and that is possible at all only because the advance was
measured off the *body*: had it been measured off the finished ink, a mark would be
paying for its own room and the two would never agree.

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
           "SIDEBEARING", "AIR", "MARK_GAP", "FILLET", "LIGHT", "names", "get", "clamp",
           "validate", "thickness", "glyph_field", "ink_box", "core_box", "overhang",
           "advance", "offset", "strokes"]

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

#: The daylight between a letter's ink and the mark over it, in em units.  Measured
#: from ink to ink -- the letter's serifs and the pen's own width are already in the
#: numbers this is added to -- so a heavy face does not wear its accents closer than a
#: light one, and a serifed face does not wear them inside the serifs.
MARK_GAP = 0.05

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


def _box_of(pts, s: dict, serifs: bool = True):
    """``(x0, y0, x1, y1)`` of what the pen draws for these strokes, or ``None``.

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
    for line in pts:
        if not line:
            continue
        for x, y in line:
            xs.append(x)
            ys.append(y)
        if serifs:
            for cx, cy, hw, hh in _serifs(line, s):
                bx.extend((cx - hw, cx + hw))
                by.extend((cy - hh, cy + hh))
    if not xs:
        return None
    r = 0.5 * s["weight"]
    x0, x1 = min(xs) - r, max(xs) + r
    y0, y1 = min(ys) - r, max(ys) + r
    if bx:
        x0, x1 = min(x0, min(bx)), max(x1, max(bx))
        y0, y1 = min(y0, min(by)), max(y1, max(by))
    return (x0, y0, x1, y1)


def _fits(box, lo: float, hi: float) -> bool:
    """Whether an ink box lies between two edges.  No box at all -- no ink -- fits."""
    return box is None or (box[0] >= lo and box[2] <= hi)


def _mark_of(ch: str, s: dict, raw: list) -> list:
    """The mark of a composed letter, in place, in the glyph's own coordinates.

    Where a mark goes is the pen's decision and not the alphabet's, and this is the
    whole of why.  A cap's serif reaches *above* the cap line, a display weight reaches
    further than a light one, and the gap wants to be the same gap in both -- so the
    anchor is measured off the ink the base is actually carrying, serifs and pen width
    included, and the mark is placed ``MARK_GAP`` above (or below) *that*.

    And a mark is *fitted* to the letter it is over.  An acute on an ``i`` is a smaller
    acute than an acute on an ``a``, which is what a type designer draws -- and it is
    not decoration: an ``i`` is one stem wide, and an accent as wide as the letter's
    whole advance would push the next letter away and leave a hole in the middle of a
    word.  So a mark that already fits inside the ground the letter's advance gives it
    is left alone; one that does not is drawn smaller, uniformly, until it does, because
    squashing it in one direction alone would turn an acute into a different accent.
    The last resort -- a mark too tall to fit at any size, which only a very slanted
    face can produce -- slides it over a little instead, because an accent a hair off
    the middle is a smaller fault than an accent in the next letter's slot.  What comes
    out of this is a promise rather than a hope: a letter is not a wider letter for
    wearing an accent, and no accent leaves the ground its letter was given.
    """
    entry = skeleton.mark_of(ch)
    if entry is None:
        return []
    mark, _base = entry
    _skeleton, side = skeleton.MARKS[mark]
    pts = skeleton.mark_strokes(mark)
    r = 0.5 * s["weight"]
    # The anchor is measured on the letter as the alphabet drew it, because the pen's
    # shear takes the mark and the letter together.  Across, the mark sits over the
    # letter's **body** -- bodies are symmetric and serifs are not, and an accent
    # centred on a box with one beak in it slides off to the side of an ``e``.  Up, it
    # sits over the **feet**, because a cap's serif reaches above the cap line and a
    # mark a fixed distance above the skeleton would sit inside it.
    core, full = _box_of(raw, s, False), _box_of(raw, s, True)
    if full is None:
        return []
    centre = 0.5 * (core[0] + core[2])
    dy = (full[3] + MARK_GAP + r) if side == "above" else (full[1] - MARK_GAP - r)
    # The ground is the letter's, and it is the ground *without* the mark -- which is
    # what makes this callable at all, and what makes it fair: the letter is charged
    # for its body and its feet, and the mark lives in what is left.
    _core, _full, _lsb, _rsb, adv, off = _metrics(ch, s)
    lo, hi = AIR - off, adv - off - AIR

    def at(scale: float, dx: float, pen: bool = False) -> list:
        """The mark, scaled about its own anchor, in the glyph's coordinates."""
        out = [[(centre + dx + scale * x, dy + scale * y) for x, y in line]
               for line in pts]
        return [_shear(line, s) for line in out] if pen else out

    scale = 1.0
    if not _fits(_box_of(at(1.0, 0.0, True), s, False), lo, hi):
        low, high = 0.0, 1.0
        for _ in range(24):
            mid = 0.5 * (low + high)
            if _fits(_box_of(at(mid, 0.0, True), s, False), lo, hi):
                low = mid
            else:
                high = mid
        scale = low
    ink = _box_of(at(scale, 0.0, True), s, False)
    dx = 0.0
    if not _fits(ink, lo, hi):
        # A mark too tall to fit even as a hairline -- which only a very slanted face
        # can manage -- slides over instead of leaving the ground.  It is the last
        # thing tried rather than the first, because an accent off the middle of its
        # letter is a mistake the eye catches.
        dx = ((lo - ink[0]) if ink[0] < lo else (hi - ink[2])) / s["width"]
    return at(scale, dx)


def _placed(ch: str, s: dict) -> tuple:
    """``(base, marks)``: the glyph's strokes sheared and ready, and its mark's.

    The two lists stay apart because they are different things to the pen.  A serif
    grows where a *letter's* stroke ends on a line of the face; a mark is not a
    letter's stroke, it rides above one, and a foot on a breve's tip would be a foot
    nobody drew.  So the serif rule sees the base and not the mark.
    """
    raw = list(skeleton.strokes(ch))
    mark = _mark_of(ch, s, raw)
    return ([_shear(pts, s) for pts in raw],
            [_shear(pts, s) for pts in mark])


def strokes(ch: str, spec=None, marks: bool = True):
    """The glyph's skeleton, sheared, ready for the pen.  Handy for measuring.

    A letter wearing a mark comes with the mark in place, and with ``marks`` false it
    comes without one -- because the two are not the same question.  The finished ink
    is what a caller draws and what a box is measured from; the letter's own form is
    what the *spacing* is measured from, since a mark is fitted to the ground its
    letter has and is not part of the letter's body.
    """
    s = clamp(spec)
    base, mark = _placed(ch, s)
    return base + mark if marks else base


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
    base, marks = _placed(ch, s)
    d = None
    for pts, feet in ([(p, True) for p in base] + [(p, False) for p in marks]):
        if len(pts) == 1:
            seg = sdf.circle(X, Y, reach, pts[0][0], pts[0][1])
            d = seg if d is None else sdf.smooth_union(d, seg, k)
            continue
        for a, b in zip(pts, pts[1:]):
            r = 0.5 * thickness(b[0] - a[0], b[1] - a[1], weight, contrast)
            seg = sdf.segment(X, Y, a, b, r)
            d = seg if d is None else sdf.smooth_union(d, seg, k)
        if feet:
            for cx, cy, hw, hh in _serifs(pts, s):
                seg = sdf.box(X, Y, hw, hh, 0.0, cx, cy)
                d = seg if d is None else sdf.smooth_union(d, seg, k)
    if d is None:
        return np.full(np.broadcast(X, Y).shape, np.inf, dtype=np.float64)
    return d


def _metrics(ch: str, s: dict) -> tuple:
    """``(core, full, lsb, rsb, advance, offset)``: the spacing of one glyph.

    Six numbers and not one drop of ink in any of them, and that is the point.  The
    **core** is the body: the letter's own strokes, no feet and no mark, and it is what
    the sidebearings are measured against.  The **full** is the body with its feet on,
    because a foot is centred on its stem and half of it lands outside the body.  The
    two **sidebearings** are the daylight, or as much room as the feet ask for and
    :data:`AIR` more -- whichever is larger, which is why a big foot does not make a
    narrow letter wide.  The **advance** is the design width or the body and its
    sidebearings, whichever is larger.  And **offset** is how far the glyph is drawn so
    its left sidebearing is the room the pen asked for.

    A mark over the letter is in none of these.  An ``i`` with an acute on it is the
    width of an ``i``: what the ground has to hold is the *body*, and the mark is then
    fitted to the ground that leaves (:func:`_mark_of`) rather than being asked to pay
    for it.  That is also what lets :func:`_mark_of` call this without chasing itself.
    """
    base = [_shear(pts, s) for pts in skeleton.strokes(ch)]
    design = skeleton.advance(ch) * s["width"]
    if not base:
        return ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, design, 0.0)
    core = _box_of(base, s, False)
    full = _box_of(base, s, True)
    lsb = max(SIDEBEARING, core[0] - full[0] + AIR)
    rsb = max(SIDEBEARING, full[2] - core[2] + AIR)
    adv = max(design, (core[2] - core[0]) + lsb + rsb)
    return (core, full, lsb, rsb, adv, lsb - core[0])


def ink_box(ch: str, spec=None) -> tuple:
    """Where the glyph's ink reaches, ``(x0, y0, x1, y1)`` in em units.

    Measured off the skeleton and the pen rather than declared, so it follows a
    weight change, a shear and a serif without anybody remembering to update a table.
    It is the box a renderer wants for two things: where to evaluate the field (so a
    line of text costs a window per letter and not a pass over the sheet), and where
    the ink actually is (so a line can be centred on what the eye sees).

    This is the *finished* ink: the feet and the mark over the letter.  Spacing is
    measured off :func:`core_box`, which is the body and nothing else.
    """
    s = clamp(spec)
    base, marks = _placed(ch, s)
    boxes = [b for b in (_box_of(base, s, True), _box_of(marks, s, False)) if b]
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def core_box(ch: str, spec=None) -> tuple:
    """The box of the letter's *body*: no feet, no mark.  This is the spacing box."""
    return _metrics(ch, clamp(spec))[0]


def overhang(ch: str, spec=None) -> tuple:
    """How far the **feet** reach past the body, ``(left, right)`` in em units.

    Zero for a monoline, and the number that decides how much ground a serif needs:
    a foot is centred on its stem, so half of it lands outside the body, and the
    sidebearing has to be at least that wide or two letters share a serif.

    An accent over the letter is *not* counted.  A mark hangs over a letter the way a
    foot hangs off a stem -- and further, since an ``i`` is one stem wide and every
    accent there is is wider than that -- so what the pen charges a letter for is its
    feet.  What keeps a mark honest is not the sidebearing but the ground the advance
    gives it: :func:`_mark_of` fits the mark into that ground, and a test holds every
    composed letter's finished ink inside its advance.
    """
    core, full = _metrics(ch, clamp(spec))[:2]
    return (core[0] - full[0], full[2] - core[2])


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

    Nor is a **mark** part of the body.  A letter with an accent on it is the width of
    the letter: an acute over an ``i`` is fitted to the ground the ``i`` already had,
    and never the other way round.  So: the body sets the advance, the feet hang into
    the sidebearing, the mark is fitted to what is left, and the advance only grows
    when a foot would reach past the air it is allowed (:func:`_metrics`).  A test
    holds the outcome, every character in the face at four weights and three serif
    amounts: no glyph's finished ink leaves the ground its advance gives it.
    """
    return _metrics(ch, clamp(spec))[4]


def offset(ch: str, spec=None) -> float:
    """How far right a glyph is drawn so its left sidebearing is honoured.

    The origin the caller thinks in is the *pen's*, and a glyph whose body starts left
    of it is shifted until the sidebearing is the room the pen asked for -- so the ink
    always starts inside the ground the advance gave it.
    """
    return _metrics(ch, clamp(spec))[5]
