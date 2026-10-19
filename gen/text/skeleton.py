"""The alphabet, as skeletons.

A letter here is not an outline.  It is a *skeleton* -- one or more polylines in em
units -- and the letter is what a pen does to it.  That split is the whole point of
the package, and it is the same split the rest of the toolkit makes: an icon is a
charge on a plate, a coat is a field and its charges, a head is a stack of
cross-sections.  Identity is the few numbers that say *which* letter this is; style
is everything the pen decides afterwards.

What that buys, concretely: **weight, contrast, serifs, slant and width are not in
this file**.  A stem and a hairline are the same skeleton read by a pen that changes
its width with the direction of the stroke; a serif is a rule about stroke *ends*; a
slant is a shear applied to the points before the pen sees them.  So one alphabet
serves a monoline UI face and a modulated text face, and the difference between
them is five numbers.

Nor are the *metrics*: the first number of each entry is the letter's **design width**,
and what the pen actually advances by is that or the ink plus a sidebearing, whichever
is wider.  A face is allowed to overrun the design, and the pen is where that is
noticed -- see :func:`gen.text.pen.advance`.

The em box, counting y upwards from the baseline::

    ASCENDER   0.80   the top of a letter with an ascender (b, d, k, l)
    CAP_HEIGHT 0.70   the top of a capital
    X_HEIGHT   0.50   the top of a lowercase x
    BASELINE   0.00
    DESCENDER -0.20   the bottom of a descender (p, g, y)

Those five are the *lines* of the face, and the skeleton is drawn to them: a
centreline touches a line, so the ink a pen lays around it overshoots by half a
stem -- which is what a real face does, and why :func:`gen.text.pen.ink_box`
measures the ink rather than trusting the line.

The first four are also the **guides**: the horizontals a serif is allowed to grow
on.  That is the whole serif rule, and it is why the alphabet can be stroke data and
still have feet.

The data grammar, one glyph per line -- an advance in tenths of an em, and its
strokes:

* strokes are separated by ``|``, and each is a polyline;
* ``x,y`` is a point in tenths (``0,0`` is the baseline at the left stem);
* ``Acx,cy,rx,ry,a0,a1`` is an elliptical arc, sampled from ``a0`` to ``a1`` degrees
  (0 = +x, 90 = +y) and appended to the polyline, so a round letter is round rather
  than a polygon with enough corners.

Tenths, because a typographer's grid is where the judgements live: the letters were
drawn on graph paper and the file is the graph paper.
"""
from __future__ import annotations

import numpy as np

__all__ = ["GLYPHS", "X_HEIGHT", "CAP_HEIGHT", "ASCENDER", "DESCENDER", "GUIDES",
           "names", "has", "advance", "strokes"]

#: The lines of the face, in em units.  See the module docstring.
X_HEIGHT = 0.50
CAP_HEIGHT = 0.70
ASCENDER = 0.80
DESCENDER = -0.20

#: The horizontals a serif may grow on, in em units.
GUIDES = (0.0, X_HEIGHT, CAP_HEIGHT, ASCENDER)

#: Degrees per arc segment.  Twelve is about a sixteenth of a circle: enough that a
#: bowl reads as round at any size a game draws text, and few enough that the pen is
#: still dealing with a polyline rather than a mesh.
ARC_STEP = 12.0


# --------------------------------------------------------------------------
# the alphabet
# --------------------------------------------------------------------------
#: One entry per printable ASCII character: ``(advance, strokes)``.  Space has no
#: strokes, and a stroke that is a single point is a dot -- which is how a period is
#: a period rather than a small box.
GLYPHS = {
    # ---- capitals.  Flat letters keep a body of four tenths; rounds take more,
    # and the wide ones (M, W) take what their strokes need.
    "A": (5.6, "0,0 2.2,7 4.4,0 | 0.55,2.6 3.85,2.6"),
    "B": (5.6, "0,0 0,7 | A0,5.4,3.0,1.6,90,-90 A0,1.9,3.4,1.9,90,-90"),
    "C": (5.6, "A2.3,3.5,2.3,3.5,40,320"),
    "D": (5.4, "0,0 0,7 | A0,3.5,3.2,3.5,90,-90"),
    "E": (5.0, "0,0 0,7 | 0,7 3.8,7 | 0,3.6 3.2,3.6 | 0,0 3.8,0"),
    "F": (4.9, "0,0 0,7 | 0,7 3.8,7 | 0,3.6 3.2,3.6"),
    "G": (5.9, "A2.3,3.5,2.3,3.5,40,300 4.15,2.3 2.9,2.3"),
    "H": (5.6, "0,0 0,7 | 4.0,0 4.0,7 | 0,3.5 4.0,3.5"),
    "I": (2.6, "1.3,0 1.3,7"),
    "J": (4.4, "2.4,7 2.4,1.8 A1.2,1.8,1.2,1.8,0,-180"),
    "K": (5.2, "0,0 0,7 | 3.8,7 0,2.8 | 1.5,4.0 4.0,0"),
    "L": (4.6, "0,0 0,7 | 0,0 3.8,0"),
    "M": (6.8, "0,0 0,7 2.8,2.0 5.6,7 5.6,0"),
    "N": (5.6, "0,0 0,7 4.0,0 4.0,7"),
    "O": (6.0, "A2.5,3.5,2.5,3.5,0,360"),
    "P": (5.2, "0,0 0,7 | A0,5.3,3.2,1.7,90,-90"),
    "Q": (6.0, "A2.5,3.5,2.5,3.5,0,360 | 2.6,1.6 4.5,-0.7"),
    "R": (5.4, "0,0 0,7 | A0,5.3,3.2,1.7,90,-90 | 1.2,3.6 4.3,0"),
    "S": (5.6, "A2.4,5.2,2.4,1.9,30,250 A2.4,1.8,2.4,1.8,110,-160"),
    "T": (4.9, "0,7 4.4,7 | 2.2,7 2.2,0"),
    "U": (5.8, "0,7 0,1.8 A2.4,1.8,2.4,1.8,180,360 4.8,0"),
    "V": (5.6, "0,7 2.2,0 4.4,7"),
    "W": (7.4, "0,7 1.5,0 3.0,5.4 4.5,0 6.0,7"),
    "X": (5.6, "0,0 4.4,7 | 0,7 4.4,0"),
    "Y": (5.6, "0,7 2.2,3.4 2.2,0 | 2.2,3.4 4.4,7"),
    "Z": (5.2, "0,7 4.2,7 0,0 4.2,0"),
    # ---- lowercase.  The x-height is 5 and the ascender 8; a descender reaches -2.
    "a": (5.1, "A2.0,2.3,2.0,2.3,0,360 | 4.0,0 4.0,5 A2.3,3.6,1.7,1.4,0,180"),
    "b": (5.0, "0,0 0,8 | A2.4,2.5,2.4,2.5,0,360"),
    "c": (5.0, "A2.1,2.5,2.1,2.5,40,320"),
    "d": (5.0, "A2.3,2.5,2.3,2.5,0,360 | 4.6,0 4.6,8"),
    "e": (5.0, "0,2.5 4.2,2.5 A2.1,2.5,2.1,2.5,0,330"),
    "f": (3.6, "A1.4,6.4,1.4,1.6,30,180 0,0 | 0,5 2.6,5"),
    "g": (5.1, "A2.1,2.6,2.1,2.4,0,360 | 4.2,4.4 4.2,-1.4 A2.6,-1.4,1.6,0.5,0,-175"),
    "h": (5.4, "0,0 0,8 | 0,3.5 A0,3.5,4.0,1.5,90,0 4.0,0"),
    "i": (2.6, "1.3,0 1.3,5 | 1.3,6.7"),
    "j": (2.6, "0.9,5 0.9,-1.5 A0.9,-1.9,0.9,0.4,90,180 | 0.9,6.7"),
    "k": (5.0, "0,0 0,8 | 3.8,5 0,1.6 | 1.6,2.9 4.0,0"),
    "l": (2.6, "1.3,0 1.3,8"),
    "m": (6.6, "0,0 0,5 | 0,3.5 A0,3.5,2.5,1.5,90,0 2.5,0 | "
                "2.5,3.5 A2.5,3.5,2.5,1.5,90,0 5.0,0"),
    "n": (5.6, "0,0 0,5 A0,3.5,4.0,1.5,90,0 4.0,0"),
    "o": (4.9, "A2.1,2.6,2.1,2.4,0,360"),
    "p": (5.0, "0,-2 0,5 | A2.4,2.5,2.4,2.5,0,360"),
    "q": (5.0, "A2.3,2.5,2.3,2.5,0,360 | 4.6,-2 4.6,5"),
    "r": (4.0, "0,0 0,5 | 0,5 A0,3.6,2.8,1.4,90,20"),
    "s": (4.6, "A2.0,3.6,2.0,1.5,30,250 A2.0,1.3,2.0,1.4,110,-160"),
    "t": (3.8, "1.0,7.2 1.0,1.4 A2.4,1.4,1.4,1.4,180,300 | -0.8,5 2.8,5"),
    "u": (5.4, "0,5 0,1.8 A2.2,1.8,2.2,1.8,180,360 4.4,0"),
    "v": (5.0, "0,5 2.1,0 4.2,5"),
    "w": (6.6, "0,5 1.4,0 2.8,3.6 4.2,0 5.6,5"),
    "x": (5.2, "0,0 4.2,5 | 0,5 4.2,0"),
    "y": (5.2, "0,5 2.2,0.4 | 4.2,5 1.5,-1.9"),
    "z": (5.0, "0,5 4.0,5 0,0 4.0,0"),
    # ---- lining figures: they sit on the baseline and rise to the cap height.
    "0": (5.6, "A2.4,3.5,2.4,3.5,0,360"),
    "1": (4.2, "1.6,5.5 2.8,7 2.8,0"),
    "2": (5.6, "0.2,5.5 A2.3,5.5,2.3,1.6,165,-20 0,0 4.9,0"),
    "3": (5.6, "A2.0,5.3,2.0,1.7,170,-45 A2.2,1.9,2.2,1.9,55,-165"),
    "4": (5.6, "3.4,0 3.4,7 0,2.0 5.0,2.0"),
    "5": (5.6, "4.4,7 0.3,7 0.42,3.75 A2.5,2.5,2.4,2.5,150,-150"),
    "6": (5.6, "A2.4,2.2,2.4,2.2,0,360 | A3.0,3.4,2.9,2.9,75,190"),
    "7": (5.4, "0.2,7 4.8,7 1.6,0"),
    "8": (5.8, "A2.4,5.0,2.2,2.0,0,360 | A2.4,1.9,2.4,1.9,0,360"),
    "9": (5.6, "A2.4,4.8,2.4,2.2,0,360 | A1.2,3.6,3.6,3.4,-105,10"),
    # ---- punctuation.  A dot is a one-point stroke: the pen draws a disc, and the
    # advance says how much of a gap it leaves.
    " ": (3.0, ""),
    ".": (2.6, "0.5,0.45"),
    ",": (2.6, "0.5,0.45 0.2,-0.8"),
    ":": (2.6, "0.5,0.45 | 0.5,3.6"),
    ";": (2.6, "0.5,0.45 0.2,-0.8 | 0.5,3.6"),
    "!": (2.6, "0.5,2.0 0.5,7 | 0.5,0.45"),
    "?": (4.0, "A1.9,5.5,1.9,1.4,170,-60 1.9,2.2 1.9,1.3 | 1.9,0.45"),
    "'": (1.8, "0.5,5.4 0.5,7"),
    '"': (3.2, "0.5,5.4 0.5,7 | 1.9,5.4 1.9,7"),
    "(": (3.4, "A2.4,3.2,1.6,5.6,120,240"),
    ")": (3.4, "A0.6,3.2,1.6,5.6,-60,60"),
    "[": (3.0, "2.3,7.7 0.8,7.7 0.8,-1.5 2.3,-1.5"),
    "]": (3.0, "0.4,7.7 1.9,7.7 1.9,-1.5 0.4,-1.5"),
    "{": (3.6, "2.4,7.7 1.5,7.7 1.5,3.7 0.5,3.2 1.5,2.7 1.5,-1.5 2.4,-1.5"),
    "}": (3.6, "0.5,7.7 1.4,7.7 1.4,3.7 2.4,3.2 1.4,2.7 1.4,-1.5 0.5,-1.5"),
    "/": (3.8, "0.2,-0.6 3.0,7"),
    "\\": (3.8, "0.2,7 3.0,-0.6"),
    "-": (4.0, "0.6,3.2 3.2,3.2"),
    "_": (5.0, "0,-1.2 4.4,-1.2"),
    "+": (4.6, "0.4,2.9 3.8,2.9 | 2.1,1.2 2.1,4.6"),
    "=": (4.6, "0.4,2.1 3.8,2.1 | 0.4,3.9 3.8,3.9"),
    "<": (4.6, "3.6,4.6 0.8,2.9 3.6,1.2"),
    ">": (4.6, "0.8,4.6 3.6,2.9 0.8,1.2"),
    "*": (4.2, "2.0,4.3 2.0,7.1 | 0.8,4.9 3.2,6.5 | 3.2,4.9 0.8,6.5"),
    "&": (6.0, "A2.0,5.9,1.8,1.1,30,210 2.5,3.0 A2.5,1.5,2.4,1.5,90,-335"),
    "@": (6.4, "A2.9,3.3,2.9,3.3,30,320 | A1.4,2.7,1.1,1.1,0,360 | "
                "2.5,3.9 2.5,1.7 A1.5,1.7,1.2,0.9,0,100"),
    "%": (5.6, "A1.1,5.0,1.1,1.1,0,360 | A1.1,1.8,1.1,1.1,0,360 | 0.3,0 4.3,7"),
    "#": (5.0, "1.0,0.7 1.0,6.3 | 3.0,0.7 3.0,6.3 | 0.2,2.3 4.0,2.3 | 0.2,4.7 4.0,4.7"),
    "$": (5.4, "A2.0,5.0,2.0,1.6,30,250 A2.0,2.0,2.0,1.7,110,-160 | 2.0,7.3 2.0,-0.4"),
    "~": (5.0, "0.2,3.2 A1.2,3.2,1.2,0.8,180,0 A3.6,3.2,1.2,0.8,180,360"),
    "^": (4.6, "0.2,4.4 2.3,7.0 4.4,4.4"),
    "`": (2.6, "0.2,6.8 1.6,5.8"),
    "|": (2.2, "0.5,-1.5 0.5,7.7"),
}


# --------------------------------------------------------------------------
# reading it
# --------------------------------------------------------------------------
def names() -> list:
    """Every character the face has a skeleton for, in a stable order."""
    return sorted(GLYPHS)


def has(ch: str) -> bool:
    """Whether ``ch`` is in the face.  A missing character is an error, not a box."""
    return str(ch) in GLYPHS


def advance(ch: str) -> float:
    """The glyph's own advance, in em units, with no tracking of any kind."""
    key = str(ch)
    if key not in GLYPHS:
        raise KeyError(f"no skeleton for {key!r}: the face covers {len(GLYPHS)} "
                       f"characters, and a missing one is a mistake rather than a box")
    return GLYPHS[key][0] / 10.0


def _arc(cx, cy, rx, ry, a0, a1, into):
    """Append an elliptical arc, sampled at :data:`ARC_STEP` degrees."""
    span = float(a1) - float(a0)
    n = max(1, int(round(abs(span) / ARC_STEP)))
    for i in range(n + 1):
        a = np.radians(float(a0) + span * i / n)
        into.append((cx + rx * np.cos(a), cy + ry * np.sin(a)))


def strokes(ch: str) -> list:
    """The glyph's skeleton: a list of polylines, in em units, y up.

    ``(0, 0)`` is the baseline at the glyph's origin, so a caller can place the glyph
    by an offset and nothing else.  A stroke of one point is a dot.
    """
    key = str(ch)
    if key not in GLYPHS:
        raise KeyError(f"no skeleton for {key!r}")
    out = []
    for text in GLYPHS[key][1].split("|"):
        text = text.strip()
        if not text:
            continue
        pts = []
        for token in text.split():
            if token[0] in "Aa":
                nums = [float(v) for v in token[1:].split(",")]
                if len(nums) != 6:
                    raise ValueError(f"{key!r}: an arc needs cx,cy,rx,ry,a0,a1: {token!r}")
                cx, cy, rx, ry, a0, a1 = nums
                _arc(cx / 10.0, cy / 10.0, rx / 10.0, ry / 10.0, a0, a1, into=pts)
            else:
                x, y = token.split(",")
                pts.append((float(x) / 10.0, float(y) / 10.0))
        if pts:
            out.append(pts)
    return out
