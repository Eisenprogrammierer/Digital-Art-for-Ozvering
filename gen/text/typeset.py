"""Typesetting: a string becomes a line, a line becomes ink.

Three jobs, and they are three different questions:

* **How far does the pen move?**  :func:`advance` is the skeleton's own advance with
  the pen's width applied; :func:`measure` is a line's worth of them.  Between two
  letters the pen also pays the *kern* -- the correction :mod:`gen.text.kern` reads off
  the two profiles -- and the tracking.  Both are *between* letters, so a line of ``n``
  letters has ``n - 1`` of each: that is the one thing a font's metrics have to get
  right, because every layout bug starts as an off-by-one-tracking.
* **What is a letter?**  :func:`glyphs` answers that, and the answer is not always the
  characters: with ``liga`` on, a sequence the face has a join for is set as that one
  glyph.  A ligature is not a second way of drawing the line -- the join's spacing comes
  out of the same metrics as everybody else's -- it is the alphabet admitting that two
  of its letters have one place where they want to touch.
* **Where does each letter sit?**  :func:`place` answers with the pen positions, and
  :func:`ink_bounds` answers with where the *ink* actually reached, which is not the
  same box: the em box has room in it that no letter uses, so a line centred by its
  advances is a line that looks off-centre.
* **What does it look like?**  :func:`paint` evaluates the field one glyph at a time
  on that glyph's own window and unions the results, so a line costs a pass over the
  letters rather than a pass over the sheet -- which is what makes drawing a
  300-pixel title and a 12-pixel label the same function.

What is *not* here is a rasteriser.  The field is an ordinary distance field, so a
letter takes the same coverage, the same anti-aliasing and the same lit bevel as
every other shape in the toolkit; :func:`paint` with ``emboss`` is the same call the
icons make for a charge.
"""
from __future__ import annotations

import numpy as np

from core import palette, render
from core.canvas import Canvas
from gen.text import kern, pen, skeleton

__all__ = ["INK", "JOINS", "advance", "glyphs", "measure", "place", "ink_bounds",
           "spec_of", "paint", "draw", "block", "pair"]

JOINS = tuple(sorted((("".join(parts), ch) for ch, parts in skeleton.LIGATURES.items()),
                     key=lambda pair: -len(pair[0])))

INK = (0.15, 0.17, 0.21)


def spec_of(spec=None) -> dict:
    """A full typesetting spec: the pen's five numbers, the tracking and the kerning.

    ``kern`` is a *strength* and not a distance: 1 draws the line the rule asks for, 0
    draws it on the bare advances, and anything between is a fraction of the correction.
    That is what makes "with and without" a specimen rather than an argument, and it is
    the only knob here: a per-pair table would be the second font this package refuses
    to be.
    """
    out = pen.clamp(spec)
    for key, lo, hi, fallback in (("tracking", -0.05, 0.30, 0.0), ("kern", 0.0, 1.0, 1.0)):
        try:
            v = float((spec or {}).get(key, fallback))
        except (TypeError, ValueError):
            v = fallback
        out[key] = float(np.clip(v, lo, hi))

    try:
        liga = float((spec or {}).get("liga", 1.0))
    except (TypeError, ValueError):
        liga = 1.0
    out["liga"] = 1.0 if liga >= 0.5 else 0.0
    return out


def glyphs(text: str, spec=None) -> list:
    """The glyphs a string is *set* with, which is not always its characters.

    With ``liga`` on, the longest join the face has for the text at that point replaces
    the characters it is made of -- ``ffi`` is one glyph, not ``ff`` and an ``i`` -- and
    everything else is itself.  Off, the line is the string, character for character,
    which is what a caption that must be searchable or a test that counts kerning wants.
    """
    s = spec_of(spec)
    text = str(text)
    if not s["liga"]:
        return list(text)
    out = []
    i = 0
    while i < len(text):
        for seq, one in JOINS:
            if text.startswith(seq, i):
                out.append(one)
                i += len(seq)
                break
        else:
            out.append(text[i])
            i += 1
    return out


def advance(ch: str, spec=None) -> float:
    """How far the pen moves after drawing ``ch``, in em units."""
    s = spec if spec is not None and "tracking" in spec else spec_of(spec)
    return pen.advance(ch, s)


def _pair(a: str, b: str, s: dict) -> float:
    """The pen's move from the start of ``a`` to the start of ``b``, in em units.

    One place, because three callers need the same arithmetic and the two corrections
    are both *between* letters.  The kern comes first and the tracking second, in that
    order on purpose: a kern is what the two shapes ask of each other, and tracking is
    what the typesetter asks of the line -- adding air to a pair should not cancel the
    correction its shapes asked for.
    """
    return pen.advance(a, s) + s["kern"] * kern.kern(a, b, s) + s["tracking"]


def measure(text: str, spec=None) -> float:
    """The width of ``text`` in em units: advances, kerns and tracking, not trailed.

    ``measure`` with ``kern = 0`` and no tracking is the sum of the advances and
    nothing else, which is what the sidecar's table is for; with both, it is the line
    the pen will actually lay out.
    """
    s = spec_of(spec)
    line = glyphs(text, s)
    if not line:
        return 0.0

    total = sum(_pair(a, b, s) for a, b in zip(line, line[1:]))
    return total + pen.advance(line[-1], s)


def place(text: str, spec=None) -> list:
    """Where each character sits: ``[(ch, x), ...]`` with ``x`` in em units.

    The first character sits on its own left sidebearing, so the positions are the
    pen's *drawing* positions: :func:`ink_bounds` is still the box an aligner wants,
    because the right sidebearing of the last letter is not part of the ink either.
    """
    s = spec_of(spec)
    line = glyphs(text, s)
    out = []
    x = 0.0
    for i, ch in enumerate(line):
        out.append((ch, x + pen.offset(ch, s)))
        if i + 1 < len(line):
            x += _pair(ch, line[i + 1], s)
    return out


def ink_bounds(text: str, spec=None) -> tuple:
    """Where the ink of ``text`` reaches: ``(x0, y0, x1, y1)`` in em units.

    The em box is not the ink.  A line of capitals reaches the cap line and stops --
    two tenths of the box above it are empty -- and the first letter's stem reaches
    half a weight left of its own origin.  So this is the box an aligner wants, and
    (0, 0, 0, 0) is the honest answer for a line with no ink in it at all.
    """
    s = spec_of(spec)
    box = None
    for ch, x in place(text, s):
        if not skeleton.strokes(ch):
            continue
        x0, y0, x1, y1 = pen.ink_box(ch, s)
        b = (x + x0, y0, x + x1, y1)
        if box is None:
            box = list(b)
            continue
        box[0] = min(box[0], b[0])
        box[1] = min(box[1], b[1])
        box[2] = max(box[2], b[2])
        box[3] = max(box[3], b[3])
    return tuple(box) if box is not None else (0.0, 0.0, 0.0, 0.0)


def _window(X, Y, box, pad: float = 2.0):
    """The slice of the canvas grid covering the world box ``box``, or ``None``.

    The same two-pixels-of-slack window the other packages use: a glyph is evaluated
    on its own patch and the patch's edge is never the glyph's edge.  The grid is
    passed in rather than fetched, because a line of text asks this question once per
    letter and building a meshgrid per letter is most of the cost of a line.
    """
    xs, ys = X[0], Y[:, 0]
    x0, y0, x1, y1 = box
    cols = np.nonzero((xs >= x0 - pad) & (xs <= x1 + pad))[0]
    rows = np.nonzero((ys >= y0 - pad) & (ys <= y1 + pad))[0]
    if cols.size == 0 or rows.size == 0:
        return None
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    return X[r0:r1, c0:c1], Y[r0:r1, c0:c1], r0, c0


def pair(a: str, b: str, spec=None) -> float:
    """The distance from the start of ``a`` to the start of ``b``, corrections included.

    The number :func:`measure` sums and :func:`place` walks; public because a layout
    that draws a line one letter at a time -- a menu, a caption, the credits screen --
    needs the same distance the line would have had.
    """
    return _pair(a, b, spec_of(spec))


def paint(canvas: Canvas, text: str, spec=None, x: float = 0.0, y: float = 0.0,
          colour=INK, emboss: float = 0.0):
    """Draw ``text`` on ``canvas`` with its baseline-left origin at ``(x, y)``.

    ``x`` and ``y`` are in the canvas's own world units (em units here, because a
    caller that wants a line makes a canvas whose camera says so -- see :func:`draw`).
    ``emboss`` lights the letters off their own field, tangentially to their edges,
    which is what makes a title sit on a button instead of being pasted on it.
    """
    s = spec_of(spec)
    X, Y = canvas.grid()
    d = None
    for ch, dx in place(text, s):
        if not skeleton.strokes(ch):
            continue
        bx0, by0, bx1, by1 = pen.ink_box(ch, s)
        win = _window(X, Y, (x + dx + bx0, y + by0, x + dx + bx1, y + by1))
        if win is None:
            continue
        Xw, Yw, r0, c0 = win
        sub = pen.glyph_field(ch, s, Xw - (x + dx), Yw - y)
        if d is None:
            d = np.full((canvas.h, canvas.w), np.inf, dtype=np.float64)
        rows, cols = slice(r0, r0 + Xw.shape[0]), slice(c0, c0 + Xw.shape[1])
        d[rows, cols] = np.minimum(d[rows, cols], sub)
    if d is None:
        return canvas
    a = render.coverage(d, canvas.dx)
    if emboss > 0.0:
        canvas.over(render.shade(colour, d, canvas.dx, bevel=emboss, light=pen.LIGHT), a)
    else:
        canvas.over(palette.srgb_to_linear(colour), a)
    return canvas


def block(lines, spec=None, em: float = 64.0, pad: float = 0.16, leading: float = 1.35,
          colour=INK, emboss: float = 0.0, background=None, align: str = "left") -> Canvas:
    """A block of lines, stacked on their baselines, sized by its own ink.

    ``em`` is how many pixels an em is, so the *same* block is drawn at any size:
    nothing here is scaled after the fact.  ``leading`` is the distance from one
    baseline to the next **in ems**, which is the typographer's number and not a gap:
    a block's leading is a property of the face, so it does not change when a line
    happens to be all capitals or has a descender in it.

    The sheet is as tall as the ink plus the padding, which is the difference between
    a text block and a fixed-size image with a margin -- and it is the same decision
    the activities screen makes in ``credits``.
    """
    text = [str(line) for line in lines]
    if not text:
        raise ValueError("a block of no lines has no ink to size itself by")
    if align not in ("left", "centre", "right"):
        raise ValueError(f"align must be left, centre or right, not {align!r}")
    s = spec_of(spec)
    boxes = [ink_bounds(line, s) for line in text]
    step = float(leading)

    high = boxes[0][3] + max(i * step - b[1] for i, b in enumerate(boxes))
    wide = max(b[2] - b[0] for b in boxes)
    w = max(1, int(round((wide + 2.0 * float(pad)) * em)))
    h = max(1, int(round((high + 2.0 * float(pad)) * em)))
    canvas = Canvas(w, h, scale=float(em), center=(0.0, 0.0), background=background)
    W, H = w / float(em), h / float(em)
    for i, (line, b) in enumerate(zip(text, boxes)):
        if align == "left":
            x = -W * 0.5 + float(pad) - b[0]
        elif align == "right":
            x = W * 0.5 - float(pad) - b[2]
        else:
            x = -(b[0] + b[2]) * 0.5

        baseline = H * 0.5 - float(pad) - boxes[0][3] - i * step
        paint(canvas, line, s, x, baseline, colour=colour, emboss=emboss)
    return canvas


def draw(text: str, spec=None, em: float = 64.0, pad: float = 0.16,
         colour=INK, emboss: float = 0.0, background=None, align: str = "left") -> Canvas:
    """One line on a canvas of its own.  ``block([text], ...)``, and nothing more."""
    return block([text], spec=spec, em=em, pad=pad, leading=1.0, colour=colour,
                 emboss=emboss, background=background, align=align)
