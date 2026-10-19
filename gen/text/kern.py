"""Kerning: how much white a pair of letters is allowed to keep between them.

The pen spaces a letter by its *body*: every glyph carries ``SIDEBEARING = 0.05`` em on
each side, measured off its own stems, plus whatever a foot needs beyond that.  Two
letters set at those advances therefore keep the daylight the pen promises -- and that
is the end of what an advance can know, because an advance is a property of **one**
letter.  What happens between two letters depends on the two of them: a ``T`` lets an
``o`` walk in under its bar, an ``A`` and a ``V`` leave a wedge of white wide enough to
see, and two ``n``s stand on their serifs, which are as close as anything in this face
comes.

So this module reads the *profiles* of the two glyphs out of the same field the pen
draws them with -- per row, how far left and how far right the ink reaches -- and moves
the second glyph by the smallest of three answers.  The spacing question is asked of the
**bodies** and the safety question of the finished **ink**, which is the same split the
advance makes: an accent is fitted into the ground its letter already had, and a kern
that spent that ground would walk two circumflexes into each other.

1. **the room the closest approach has left.**  The profiles are compared row by row
   with the rows that only one letter touches filled in from that letter's own box (an
   overhang is the whole point of a ``T``: its bar is *above* the ``o`` and constrains
   it all the same).  The gap may close down to :func:`floor`: the pen's own ``AIR``
   between two letters, or ``MARK_GAP`` where one of them wears an accent, because the
   daylight a mark keeps from the letter under it is not the daylight a stem keeps from
   its neighbour.  A kern is not allowed to take back a promise the advance made.
2. **the white a pair is carrying relative to the pair the face is tuned to.**  The
   reference is *measured* and not declared: it is what :func:`air` says about two
   ``n``s in this very pen.  A pair whose mean daylight over its shared heights is
   wider than that is carrying white the face does not ask for, and closing it is what
   "even colour" means.
3. **a twelfth of an em.**  A rule reading profiles will happily close a pair a designer
   would not -- everything between a stem and a period looks like excess white to it --
   so the third answer is an admission rather than a measurement: no kern here is
   bigger than :data:`CLOSE_MAX`.

The minimum of the three is the kern.  It is never positive: the advances already keep
the pen's daylight, and a rule that *opened* a pair would be second-guessing the
sidebearings rather than reading the ink.

The rule's blind spot is worth naming, because it is the reason the specimen sheet has
the notorious pairs in it.  The closest approach cannot see that a bowl leaves more
white above and below the point where it comes nearest than a stem does; two ``o``s at
the same air as two stems are still looser-looking, and this rule only takes a third of
that difference back.  A face with a foundry's eye has one more number per pair than a
face with a rule, and this is the rule's whole budget.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from gen.text import pen, skeleton

__all__ = ["PROFILE_ROWS", "PROFILE_COLS", "AIR_FLOOR", "CLOSE_MAX", "CACHE_MAX",
           "REFERENCE_PAIR", "profile", "air", "floor", "reference", "kern", "stats",
           "clear"]


#: How finely a glyph is read.  The profile is a raster of the ink, one entry per row,
#: and 64 rows over the tallest glyph in the face is about a hundredth of an em -- four
#: times finer than the smallest kern this module can produce, which is where a raster
#: of a smooth field stops being the limitation.
PROFILE_ROWS = 64
PROFILE_COLS = 128

#: The closest two letters may come, in em units.  This is the pen's own :data:`AIR`,
#: not a new number: the pen already reserves it between a foot and the next letter's
#: ground, so a kern that closed past it would be spending room the advance set aside.
AIR_FLOOR = pen.AIR

#: The closest a pair may come when one of them wears an accent.  A mark is fitted into
#: the ground its letter has (``pen._mark_of``) and keeps :data:`pen.MARK_GAP` from the
#: letter it sits on; two accents are the thinnest ink in the face, so a kern that
#: measured only the closest approach would happily close ``ÀÁ`` until the two
#: circumflexes graze.  A mark is not a stem and does not get a stem's daylight.
MARK_FLOOR = pen.MARK_GAP

#: The largest kern, in em units.  A twelfth of an em: above that the rule is guessing.
CLOSE_MAX = 1.0 / 12.0

#: Profiles are cached by (character, pen), because a line of text asks for the same
#: letter again and again and a glyph's profile is a property of the pen and not of the
#: line it happens to be in.  Bounded, because a generator that walks the alphabet with
#: a slider under it would otherwise grow a cache for every weight it was ever shown.
CACHE_MAX = 4096

_PROFILES: "OrderedDict[tuple, tuple]" = OrderedDict()
_AIR: "OrderedDict[tuple, tuple]" = OrderedDict()
_STATS = {"hits": 0, "misses": 0}

#: The pair the face is tuned to.  Two stems with nothing between them: whatever daylight
#: the pen leaves here is the daylight this face calls normal, and every other pair is
#: read against it.  It is a measurement of the pen rather than a taste, which is why a
#: monoline and a book face get different references out of the same rule.
REFERENCE_PAIR = ("n", "n")


def stats() -> dict:
    """Hits, misses and how many profiles are being held."""
    return {"hits": _STATS["hits"], "misses": _STATS["misses"],
            "profiles": len(_PROFILES), "pairs": len(_AIR)}


def clear() -> None:
    """Drop every cached profile.  Tests call this; nothing else has a reason to."""
    _PROFILES.clear()
    _AIR.clear()


def _key(spec) -> tuple:
    """A pen, as a hashable identity: the five numbers that draw a glyph."""
    s = pen.clamp(spec)
    return tuple(round(s[k], 6) for k in sorted(pen.DEFAULTS))


def profile(ch: str, spec=None, marks: bool = True):
    """``(ys, left, right, box)``: the glyph's ink as a function of height, or ``None``.

    ``left`` and ``right`` are in the pen's own coordinates -- the ones an advance is
    measured in -- so a profile can be compared with another glyph's directly.  A row
    with no ink at all is ``nan``: a letter is not a rectangle, and the rows it misses
    are exactly the rows where a neighbour may spend an overhang.

    The three rows a *hairline* occupies are the whole difficulty of a rule like this,
    which is why the last row of each glyph is sampled on the raster rather than
    interpolated: the profile is read off the field at ``PROFILE_ROWS`` heights, and the
    field is the thing the raster would have drawn anyway.

    ``marks`` chooses the *body* (false) or the finished ink (true), and the two are
    cached apart because :func:`air` needs both: the spacing is the body's question and
    the safety is the ink's.
    """
    ch = str(ch)
    s = pen.clamp(spec)
    key = (ch, marks, _key(s))
    hit = _PROFILES.get(key)
    if hit is not None:
        _STATS["hits"] += 1
        _PROFILES.move_to_end(key)
        return hit
    _STATS["misses"] += 1
    x0, y0, x1, y1 = pen.ink_box(ch, s, marks=marks)
    off = pen.offset(ch, s)
    if not (x1 > x0) or not (y1 > y0):
        profil = None                      # a space, or a letter with no ink
    else:
        ys = np.linspace(y0, y1, PROFILE_ROWS)
        xs = np.linspace(x0, x1, PROFILE_COLS)
        X = np.repeat(xs[None, :], PROFILE_ROWS, axis=0)
        Y = np.repeat(ys[:, None], PROFILE_COLS, axis=1)
        inside = pen.glyph_field(ch, s, X, Y, marks=marks) <= 0.0
        left = np.where(inside.any(axis=1), xs[np.argmax(inside, axis=1)], np.nan) + off
        right = np.where(inside.any(axis=1),
                         xs[PROFILE_COLS - 1 - np.argmax(inside[:, ::-1], axis=1)],
                         np.nan) + off
        profil = (ys, left, right, (x0 + off, y0, x1 + off, y1))
    _PROFILES[key] = profil
    if len(_PROFILES) > CACHE_MAX:
        _PROFILES.popitem(last=False)
    return profil


def _gap(pa, pb, adv) -> tuple:
    """``(close, mean)`` of two profiles at ``adv``, or ``None`` if they share no height."""
    ya, ra = pa[0], pa[2]
    yb, lb = pb[0], pb[1]
    if min(ya[-1], yb[-1]) <= max(ya[0], yb[0]):
        return None
    step = np.linspace(max(ya[0], yb[0]), min(ya[-1], yb[-1]), PROFILE_ROWS)
    right = np.interp(step, ya, ra, left=np.nan, right=np.nan)
    left = np.interp(step, yb, lb, left=np.nan, right=np.nan)
    # an overhang: the rows a glyph misses are filled in from its own far edge, because
    # a bar above a bowl constrains that bowl just as a stem beside it would
    right = np.where(np.isnan(right), np.nanmax(ra), right)
    left = np.where(np.isnan(left), np.nanmin(lb), left)
    gap = (adv + left - right)
    gap = gap[np.isfinite(gap)]
    if gap.size == 0:
        return None
    return (float(gap.min()), float(gap.mean()))


def air(a: str, b: str, spec=None):
    """``(close, mean)``: the daylight between ``a`` and ``b`` at their own advances.

    ``close`` is the narrowest the **finished ink** comes -- accents included, because
    the closest approach is a safety question and two accents that touch are a mistake
    whatever the bodies are doing.  ``mean`` is the average daylight of the two
    **bodies** over the heights they share: the spacing question, the one the pen
    answers off the body when it measures an advance.  ``None`` when either glyph has no
    ink -- a space is not a letter and takes no kern, which is a decision and not a
    fallback: a kern is a correction to the ink's spacing, and there is no ink.

    Both numbers are in em units and are properties of the *pair*: :func:`kern` reads
    them and nothing else does.
    """
    body_a, body_b = profile(a, spec, marks=False), profile(b, spec, marks=False)
    ink_a, ink_b = profile(a, spec), profile(b, spec)
    if body_a is None or body_b is None or ink_a is None or ink_b is None:
        return None
    key = (str(a), str(b), _key(spec))
    hit = _AIR.get(key)
    if hit is not None:
        _AIR.move_to_end(key)
        return hit
    adv = pen.advance(a, spec)
    ink = _gap(ink_a, ink_b, adv)
    body = _gap(body_a, body_b, adv)
    if ink is None or body is None:
        return None
    answer = (ink[0], body[1])
    _AIR[key] = answer
    if len(_AIR) > CACHE_MAX:
        _AIR.popitem(last=False)
    return answer


def floor(a: str, b: str, spec=None) -> float:
    """The daylight this pair may not go below: the pen's air, or a mark's when marked.

    Public because it is the number the safety tests compare against and the number the
    rule spends: two letters have :data:`AIR_FLOOR` to give, a pair wearing an accent has
    :data:`MARK_FLOOR`, because the daylight a mark keeps from the letter under it is not
    the daylight a stem keeps from its neighbour.
    """
    marked = skeleton.mark_of(a) is not None or skeleton.mark_of(b) is not None
    return MARK_FLOOR if marked else AIR_FLOOR


def reference(spec=None) -> float:
    """The daylight this pen calls normal: what two stems keep between them.

    Measured, and reported as the *mean* of the pair, because that is the number
    :func:`kern` compares against.  A pen whose letters are wider than its sidebearings
    says so here, and every kern in the line is measured against its own face.
    """
    got = air(*REFERENCE_PAIR, spec)
    return float("nan") if got is None else got[1]


def kern(a: str, b: str, spec=None) -> float:
    """How far ``b`` moves left behind ``a``, in em units.  Never positive.

    The smallest of the three answers in the module's docstring: the room before the
    closest approach reaches the pen's own air, the white this pair carries compared to
    the face's reference pair, and :data:`CLOSE_MAX`.
    """
    got = air(a, b, spec)
    if got is None:
        return 0.0
    close, mean = got
    ref = reference(spec)
    room = close - floor(a, b, spec)
    excess = mean - ref
    if not np.isfinite(ref):
        excess = 0.0
    return -float(max(0.0, min(room, excess, CLOSE_MAX)))
