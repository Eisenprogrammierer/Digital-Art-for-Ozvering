"""Typesetting: a string becomes a line, a line becomes ink.

Three jobs, and they are three different questions:

* **How far does the pen move?**  :func:`advance` is the skeleton's own advance with
  the pen's width applied; :func:`measure` is a line's worth of them.  Tracking is
  *between* letters, so a line of ``n`` letters has ``n - 1`` of them -- the one thing
  a font's metrics have to get right, because every layout bug starts as an
  off-by-one-tracking.
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
from gen.text import pen, skeleton

__all__ = ["INK", "advance", "measure", "place", "ink_bounds", "spec_of", "paint",
           "draw", "block"]

#: The colour a line of text is drawn in when nobody says otherwise: a soft black,
#: which is what a page wants and what a bright UI wants to be told otherwise about.
INK = (0.15, 0.17, 0.21)


def spec_of(spec=None) -> dict:
    """A full typesetting spec: the pen's five numbers and the tracking."""
    out = pen.clamp(spec)
    try:
        tracking = float((spec or {}).get("tracking", 0.0))
    except (TypeError, ValueError):
        tracking = 0.0
    out["tracking"] = float(np.clip(tracking, -0.05, 0.30))
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def advance(ch: str, spec=None) -> float:
    """How far the pen moves after drawing ``ch``, in em units."""
    s = spec if spec is not None and "tracking" in spec else spec_of(spec)
    return pen.advance(ch, s)


def measure(text: str, spec=None) -> float:
    """The width of ``text`` in em units, tracking included but not trailed."""
    s = spec_of(spec)
    text = str(text)
    if not text:
        return 0.0
    return sum(advance(ch, s) for ch in text) + s["tracking"] * (len(text) - 1)


def place(text: str, spec=None) -> list:
    """Where each character sits: ``[(ch, x), ...]`` with ``x`` in em units.

    The first character sits on its own left sidebearing, so the positions are the
    pen's *drawing* positions: :func:`ink_bounds` is still the box an aligner wants,
    because the right sidebearing of the last letter is not part of the ink either.
    """
    s = spec_of(spec)
    out = []
    x = 0.0
    for ch in str(text):
        out.append((ch, x + pen.offset(ch, s)))
        x += advance(ch, s) + s["tracking"]
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


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
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
    # baselines are `step` apart; the block is as tall as the ink they carry
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
        # the first line's ink top lands a padding below the top of the sheet, and
        # every line after it is exactly one leading lower
        baseline = H * 0.5 - float(pad) - boxes[0][3] - i * step
        paint(canvas, line, s, x, baseline, colour=colour, emboss=emboss)
    return canvas


def draw(text: str, spec=None, em: float = 64.0, pad: float = 0.16,
         colour=INK, emboss: float = 0.0, background=None, align: str = "left") -> Canvas:
    """One line on a canvas of its own.  ``block([text], ...)``, and nothing more."""
    return block([text], spec=spec, em=em, pad=pad, leading=1.0, colour=colour,
                 emboss=emboss, background=background, align=align)
