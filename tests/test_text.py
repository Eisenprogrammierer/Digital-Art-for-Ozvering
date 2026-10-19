"""Regression tests for the generated typeface.

A font has one thing a picture does not: *metrics*.  A glyph that looks right but
reports the wrong advance is a font that breaks every layout that trusts it, and the
breakage shows up three screens away from the letter.  So the tests here are mostly
about numbers agreeing with pixels:

* the **guides** are where the file says they are -- measured on the ink, not on the
  skeleton, because a pen lays a stem around a centreline and half of it overshoots;
* the **advance** is wide enough for the ink plus its sidebearings, at every weight
  the pen allows, because that is the promise the spacing system makes;
* **measure** is the sum of the advances and nothing else, so a layout can compute a
  line's width without drawing it -- and get the same answer as the drawing does;
* a **serif** grows on a guide and nowhere else, which is the whole serif rule;
* a **missing character** is refused rather than drawn as a box, because a box in the
  middle of a menu is a bug that shipped, and a refusal is a bug that did not.

The determinism test is the same one the other suites run: the same face drawn twice,
in two processes, byte for byte.

Run either way::

    python tests/test_text.py      # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.canvas import Canvas  # noqa: E402
from gen.text import pen, skeleton, typeset  # noqa: E402


# ------------------------------------------------------------------- helpers
_EM = 200.0


def _ink(ch: str, spec=None):
    """The glyph's ink as a boolean mask on a window around its own ink box.

    Returns the mask and the window's lower-left corner in em units, so a row index
    can be turned back into the ``y`` the field was asked about -- which is what makes
    "where did the ink stop" a measurement rather than a guess.
    """
    x0, y0, x1, y1 = pen.ink_box(ch, spec)
    pad = 3
    x_lo, y_lo = x0 - pad / _EM, y0 - pad / _EM
    w = int(np.ceil((x1 - x0) * _EM)) + 2 * pad
    h = int(np.ceil((y1 - y0) * _EM)) + 2 * pad
    axis_x = x_lo + (np.arange(w) + 0.5) / _EM
    axis_y = y_lo + (np.arange(h) + 0.5) / _EM
    X, Y = np.meshgrid(axis_x, axis_y)
    return pen.glyph_field(ch, spec, X, Y) < 0.0, (x_lo, y_lo)


def _extent(ch: str, spec=None) -> tuple:
    """``(x0, y0, x1, y1)`` of the ink, measured off the raster in em units.

    Each edge is the *centre* of the outermost sample that is inside the glyph, which
    is the only honest reading of a sampled boundary: the field crosses zero somewhere
    between the last inside sample and the first outside one, and a half-texel is the
    resolution this measurement has.
    """
    mask, (x_lo, y_lo) = _ink(ch, spec)
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    assert rows.size and cols.size, ch
    return (x_lo + (cols[0] + 0.5) / _EM, y_lo + (rows[0] + 0.5) / _EM,
            x_lo + (cols[-1] + 0.5) / _EM, y_lo + (rows[-1] + 0.5) / _EM)


def _top(ch: str, spec=None) -> float:
    return _extent(ch, spec)[3]


def _bottom(ch: str, spec=None) -> float:
    return _extent(ch, spec)[1]


# --------------------------------------------------------------------- guides
def test_the_guides_are_where_the_file_says_they_are():
    """A stem's ink overshoots its centreline by exactly half a stem.

    This is the test that ties the alphabet to the pen: the skeleton is drawn *on* the
    guides, so the top of an ``H`` is the cap line plus half a stem and no more.  If
    somebody moves a guide or changes how a stem is measured, the two stop agreeing
    here instead of in a game's menu.
    """
    spec = pen.get("sans")
    half = 0.5 * spec["weight"]
    for ch, guide in (("H", skeleton.CAP_HEIGHT), ("x", skeleton.X_HEIGHT),
                      ("b", skeleton.ASCENDER)):
        got = _top(ch, spec)
        want = guide + half
        assert abs(got - want) < 0.004, (ch, got, want)
    assert abs(_bottom("p", spec) - (skeleton.DESCENDER - half)) < 0.004


def test_a_cap_is_shorter_than_an_ascender_and_both_clear_the_x_height():
    """The three horizontals, in the order the design needs them."""
    assert skeleton.X_HEIGHT < skeleton.CAP_HEIGHT < skeleton.ASCENDER
    assert skeleton.DESCENDER < 0.0 < skeleton.X_HEIGHT


def test_the_ink_box_contains_the_ink():
    """The measured box is a box the raster stays inside, at every weight."""
    for name in pen.names():
        spec = pen.get(name)
        for ch in "Hamburgefonstiv":
            x0, y0, x1, y1 = pen.ink_box(ch, spec)
            got = _extent(ch, spec)
            assert got[0] >= x0 - 0.006 and got[2] <= x1 + 0.006, (name, ch, got, (x0, x1))
            assert got[1] >= y0 - 0.006 and got[3] <= y1 + 0.006, (name, ch, got, (y0, y1))


# -------------------------------------------------------------------- metrics
def test_every_advance_holds_its_ink_plus_a_sidebearing():
    """The spacing system, measured at both ends of every number the pen allows.

    Two promises, and they are different promises.  The **body** keeps a sidebearing
    on each side, so two letters' stems are never closer than that.  The **finished
    ink** -- feet and all -- stays inside the ground the advance gives it, so no two
    letters touch, and that has to hold for the whole legal range of weight and serif
    rather than for the default face: it is exactly at a heavy weight that a foot
    reaches furthest past its stem.
    """
    for weight in (0.025, 0.082, 0.16, 0.28):
        for serif in (0.0, 1.0, 1.5):
            spec = {"weight": weight, "serif": serif}
            for ch in skeleton.names():
                if not skeleton.strokes(ch):
                    continue
                off = pen.offset(ch, spec)
                adv = pen.advance(ch, spec)
                core = pen.core_box(ch, spec)
                full = pen.ink_box(ch, spec)
                assert core[0] + off >= pen.SIDEBEARING - 1e-9, (ch, weight, serif, core, off)
                assert core[2] + off <= adv - pen.SIDEBEARING + 1e-9, (ch, weight, serif, core, adv)
                assert full[0] + off >= -1e-9, (ch, weight, serif, full, off)
                assert full[2] + off <= adv + 1e-9, (ch, weight, serif, full, adv)


def test_a_serif_hangs_into_the_sidebearing_rather_than_widening_its_letter():
    """The one place the serif rule turns into a number, and two bugs that found it.

    A serif is centred on its stem, so half of it lies outside the body.  Two ways to
    charge a letter for that and both are wrong.  Charging the advance for the serif
    *tips* gives every letter its feet twice: the first specimen's ``quick`` was set as
    ``qu ick``, because an ``i`` was being handed a slot half again as wide as its ink.
    Charging the pen's radius to a serif *box*, as if a bar were a centreline, measures
    a foot half a stem too wide, which does the same thing more quietly.

    So, for every character a serif can grow on, in every face: the body keeps its
    daylight, and the finished ink -- feet and all -- never costs more than the "ink
    plus a sidebearing" reading it replaced.
    """
    for name in pen.names():
        spec = pen.get(name)
        for ch in skeleton.names():
            if not skeleton.strokes(ch):
                continue
            core, full = pen.core_box(ch, spec), pen.ink_box(ch, spec)
            adv = pen.advance(ch, spec)
            design = skeleton.advance(ch) * spec["width"]
            assert adv >= (core[2] - core[0]) + 2 * pen.SIDEBEARING - 1e-9, (name, ch)
            assert adv <= max(design, (full[2] - full[0]) + 2 * pen.SIDEBEARING) + 1e-9, (name, ch)
    # the letter both bugs landed on: a stem, its feet, and no hole beside it
    for name in ("text", "display"):
        spec = pen.get(name)
        assert pen.overhang("i", spec)[0] > 0.0, "the foot grew nowhere"
        assert pen.advance("i", spec) <= skeleton.advance("i") * spec["width"] \
            + 2 * pen.AIR + 1e-9, pen.advance("i", spec)
        assert pen.ink_box("i", spec)[0] + pen.offset("i", spec) >= 0.0


def test_measure_is_the_sum_of_the_advances_and_the_tracking_between_them():
    """Tracking is *between* letters: n letters have n - 1 gaps, not n."""
    spec = {**pen.get("text"), "tracking": 0.04}
    text = "Handgloves"
    want = sum(typeset.advance(ch, spec) for ch in text) + 0.04 * (len(text) - 1)
    assert abs(typeset.measure(text, spec) - want) < 1e-9
    assert typeset.measure("", spec) == 0.0
    assert abs(typeset.measure("a", spec) - typeset.advance("a", spec)) < 1e-9


def test_place_never_steps_backwards_and_the_last_pen_position_is_the_measure():
    """Where the letters go, and where the pen ends up after the last one."""
    spec = pen.get("text")
    text = "The quick brown fox"
    spots = typeset.place(text, spec)
    assert [ch for ch, _ in spots] == list(text)
    xs = [x for _, x in spots]
    assert all(b > a for a, b in zip(xs, xs[1:])), xs
    # the pen after the last letter, less the trailing tracking, is the measure
    end = xs[-1] - pen.offset(text[-1], spec) + typeset.advance(text[-1], spec)
    assert abs(end - typeset.measure(text, spec)) < 1e-9


def test_the_ink_box_of_a_line_is_the_union_of_its_glyphs():
    """The box an aligner centres on, checked against the letters it is made of."""
    spec = pen.get("text")
    line = "Hxy p"
    box = typeset.ink_bounds(line, spec)
    boxes = []
    for ch, x in typeset.place(line, spec):
        if skeleton.strokes(ch):
            b = pen.ink_box(ch, spec)
            boxes.append((x + b[0], b[1], x + b[2], b[3]))
    want = (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
    assert np.allclose(box, want)


def test_a_line_with_no_ink_has_no_box():
    """Spaces measure, but they do not reach."""
    assert typeset.ink_bounds("   ") == (0.0, 0.0, 0.0, 0.0)
    assert typeset.measure("   ") > 0.0


# --------------------------------------------------------------------- serifs
def test_a_serif_grows_on_a_guide_and_nowhere_else():
    """The serif rule, as a difference between two pens.

    A pen with serifs and a pen without differ *at the guides* -- the cap, the
    baseline, the ascender -- and nowhere in the middle of a letter.  Measured as
    extra ink below the x-height, which is where feet live and where a bowl does not.
    """
    plain = {**pen.get("text"), "serif": 0.0}
    serifed = {**pen.get("text"), "serif": 1.0}
    for ch in "HILT":
        a, _ = _ink(ch, plain)
        b, _ = _ink(ch, serifed)
        assert b.sum() > a.sum(), ch
    # the x-height is the one horizontal an arm may not be serifed on, so a bowl's
    # springing point must be identical under both pens
    a, _ = _ink("o", plain)
    b, _ = _ink("o", serifed)
    assert np.array_equal(a, b), "a round letter grew a serif"


def test_a_slash_earns_no_serifs():
    """The one stroke whose ends sit on the cap line and must stay pointed."""
    plain = {**pen.get("text"), "serif": 0.0}
    serifed = {**pen.get("text"), "serif": 1.0}
    a, _ = _ink("/", plain)
    b, _ = _ink("/", serifed)
    assert np.array_equal(a, b)


def test_a_serif_is_perpendicular_to_its_stroke():
    """A foot is a bar across a vertical, a beak a bar across a horizontal."""
    s = pen.get("text")
    feet = pen._serifs(pen.strokes("I", s)[0], s)
    assert len(feet) == 2
    for _cx, _cy, hw, hh in feet:
        assert hw > hh, "a foot on a vertical stem is wider than it is thick"
    beaks = pen._serifs(pen.strokes("T", s)[0], s)
    assert len(beaks) == 2
    for _cx, _cy, hw, hh in beaks:
        assert hh > hw, "a beak on a horizontal arm is taller than it is thick"


# ------------------------------------------------------------------ refusals
def test_a_character_the_face_does_not_have_is_refused():
    """A missing glyph is an error with a name in it, not a box."""
    for ch in ("\u00e9", "\u0416", "\n"):
        try:
            skeleton.advance(ch)
        except KeyError as exc:
            assert repr(ch)[1:-1] in str(exc) or ch in str(exc), exc
        else:
            raise AssertionError(f"{ch!r} was given an advance")


def test_a_face_nobody_wrote_is_refused_and_a_number_is_clipped():
    """A bad *name* is a mistake; a bad *number* has a nearest sensible one."""
    try:
        pen.get("comic")
    except KeyError as exc:
        assert "comic" in str(exc)
    else:
        raise AssertionError("an unknown face was handed back")
    wild = pen.clamp({"weight": 9.0, "contrast": -3.0, "slant": 5.0})
    assert wild["weight"] == pen._RANGE["weight"][1]
    assert wild["contrast"] == 0.0
    assert wild["slant"] == pen._RANGE["slant"][1]
    assert pen.validate({"weight": 0.1}) == []
    assert pen.validate({"wight": 0.1}), "a misspelt parameter was accepted"


def test_the_face_covers_all_of_printable_ascii():
    """Ninety-five characters, and the count is the promise."""
    import string
    want = set(string.printable) - set("\t\n\r\x0b\x0c")
    assert set(skeleton.names()) == want
    assert len(skeleton.names()) == 95


# --------------------------------------------------------------------- drawing
def test_a_drawn_line_is_as_wide_as_its_own_ink():
    """The canvas is sized by the ink, not by the advances.

    A `draw` returns a picture of the line and nothing else: the trailing sidebearing
    of the last letter is not ink, so it is not part of the picture.  The two numbers
    that have to agree are the measured ink box and the columns that carry alpha.
    """
    spec = pen.get("text")
    text = "Measure twice"
    em = 24.0
    pad = 0.25
    box = typeset.ink_bounds(text, spec)
    c = typeset.draw(text, spec, em=em, pad=pad)
    assert abs(c.w - (box[2] - box[0] + 2 * pad) * em) <= 1, (c.w, box)
    cols = np.nonzero((c.a > 0.01).any(axis=0))[0]
    assert cols.size
    assert abs((cols[-1] - cols[0] + 1) - (box[2] - box[0]) * em) <= 2, (cols, box)
    assert np.allclose(c.a[:, :cols[0] - 1], 0.0), "ink outside the measured box"


def test_a_block_stacks_lines_exactly_one_leading_apart():
    """Leading is a baseline distance, so it does not change with the line's content.

    Measured on the pixels: each line's ink is its own band of rows, and the distance
    between the tops of two identical lines is one leading -- whether the line has a
    descender in it or not.  (A block's *total* height does depend on its last line's
    descender; the leading does not, and that is the difference.)
    """
    spec = pen.get("text")
    em = 32.0
    leading = 1.7
    for line in ("Hx", "Hp", "Hy"):
        c = typeset.block([line, line], spec, em=em, pad=0.0, leading=leading)
        covered = (c.a > 0.01).any(axis=1)
        rows = np.nonzero(covered)[0]
        runs = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
        assert len(runs) == 2, (line, len(runs))
        step = (runs[1][0] - runs[0][0]) / em
        assert abs(step - leading) < 0.02, (line, step, leading)


def test_alignment_puts_the_ink_where_it_says():
    """Left, centre and right are the *ink* edges, not the pen's origin."""
    spec = pen.get("text")
    em = 40.0
    for align, side in (("left", 0), ("right", 1)):
        c = typeset.draw("Hxy", spec, em=em, pad=0.25, align=align)
        cols = np.nonzero((c.a > 0.01).any(axis=0))[0]
        margin = cols[0] if side == 0 else c.w - 1 - cols[-1]
        assert abs(margin - 0.25 * em) <= 2.0, (align, margin)
    left = typeset.draw("Hxy", spec, em=em, pad=0.25, align="left")
    mid = typeset.draw("Hxy", spec, em=em, pad=0.25, align="centre")
    lc = np.nonzero((left.a > 0.01).any(axis=0))[0]
    mc = np.nonzero((mid.a > 0.01).any(axis=0))[0]
    assert abs((lc[0] + lc[-1]) - (mc[0] + mc[-1])) <= 2


def test_a_block_of_no_lines_is_refused():
    try:
        typeset.block([])
    except ValueError:
        pass
    else:
        raise AssertionError("an empty block was drawn")


def test_the_emboss_lights_the_letters_with_the_one_light():
    """A glyph is a shape like any other: the same light, from the same direction."""
    from core import render
    assert pen.LIGHT is render.LIGHT
    flat = typeset.draw("Handgloves", pen.get("text"), em=60, pad=0.1)
    lit = typeset.draw("Handgloves", pen.get("text"), em=60, pad=0.1, emboss=0.08)
    assert not np.array_equal(flat.rgb, lit.rgb), "the emboss changed nothing"
    assert np.allclose(flat.a, lit.a), "the emboss moved the coverage"


# --------------------------------------------------------------- determinism
_SCRIPT = """
import sys
sys.path.insert(0, %r)
from gen.text import pen, typeset
c = typeset.block(["Digital Art", "for Ozvering"], pen.get("text"), em=28, pad=0.2)
c.save(%r)
"""


def test_the_same_face_drawn_twice_is_the_same_bytes():
    """Two processes, one hash: the font is a function, not a file."""
    root = str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryDirectory() as tmp:
        digests = []
        for i in range(2):
            path = os.path.join(tmp, f"t{i}.png")
            script = _SCRIPT % (root, path)
            subprocess.run([sys.executable, "-c", script], check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(i)})
            digests.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
        assert digests[0] == digests[1], digests


def test_a_canvas_the_drawing_returns_is_a_canvas():
    c = typeset.draw("Hi", pen.get("sans"), em=16)
    assert isinstance(c, Canvas)
    assert c.w > 0 and c.h > 0
    assert np.isfinite(c.rgb).all() and np.isfinite(c.a).all()


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
