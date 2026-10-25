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

The alphabet is not only the Latin one.  Cyrillic is the same *design* with a second
alphabet written on it, and the file says so three ways rather than drawing it twice:

* **A letter that is another letter's shape** is an :data:`ALIASES` entry -- А is A,
  Н is H, eighteen of the sixty-six in all -- because that is a fact about the shapes
  and not a coincidence to be typed out and then allowed to drift;
* **A letter that is a letter with a mark** is a :data:`COMPOSED` entry: Ё is Е with
  two dots, Й is И with a breve, and the accented letters of ISO-8859-1 are the same
  machinery (é is e with an acute).  The mark's own strokes live in :data:`MARKS`, and
  *where* it goes is the pen's question -- the gap depends on how much ink the letter
  under it is carrying -- so this file says which mark and never how high;
* **the rest** are drawn here on the same guides and the same grid as everything else,
  which is the point: a second alphabet of one design, not a second design.

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

__all__ = ["GLYPHS", "ALIASES", "MARKS", "COMPOSED", "DOTTED", "LIGATURES", "X_HEIGHT",
           "CAP_HEIGHT", "ASCENDER", "DESCENDER", "GUIDES", "names", "has", "advance",
           "strokes", "mark_of", "mark_strokes"]

X_HEIGHT = 0.50
CAP_HEIGHT = 0.70
ASCENDER = 0.80
DESCENDER = -0.20

GUIDES = (0.0, X_HEIGHT, CAP_HEIGHT, ASCENDER)

ARC_STEP = 12.0

GLYPHS = {

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

    "Б": (5.5, "0,0 0,7 | 0,7 3.6,7 | A0,2.0,3.5,2.0,90,-90"),
    "Г": (4.9, "0,0 0,7 | 0,7 3.8,7"),
    "Д": (5.4, "0.4,7 3.6,7 | 0.4,7 -0.2,-0.7 | 3.6,7 4.2,-0.7 | -0.6,0 4.6,0"),
    "Ж": (6.2, "0,7 2.3,3.5 0,0 | 4.6,7 2.3,3.5 4.6,0 | 2.3,0 2.3,7"),
    "И": (5.6, "0,7 0,0 4.0,7 4.0,0"),
    "Л": (5.6, "0,0 1.0,7 4.0,7 4.0,0"),
    "П": (5.8, "0,0 0,7 4.4,7 4.4,0"),
    "У": (5.6, "0,7 2.2,2.4 | 4.4,7 2.2,2.4 0.4,-1.9"),
    "Ф": (6.2, "2.4,0 2.4,7 | A2.4,3.5,2.2,3.5,0,360"),
    "Ц": (6.2, "0,0 0,7 4.4,7 4.4,-1.6 5.8,-1.6"),
    "Ч": (5.6, "0,7 0,2.6 4.4,2.6 | 4.4,0 4.4,7"),
    "Ш": (6.6, "0,7 0,0 4.8,0 4.8,7 | 2.4,0 2.4,7"),
    "Щ": (7.0, "0,7 0,0 4.8,0 4.8,7 | 2.4,0 2.4,7 | 4.8,0 4.8,-1.6 6.2,-1.6"),
    "Ъ": (5.5, "0,0 0,7 | 0,7 3.6,7 | A0,2.0,3.5,2.0,90,-90"),
    "Ы": (6.4, "0,0 0,7 | A0,2.0,3.2,2.0,90,-90 | 4.6,0 4.6,7"),
    "Ь": (5.4, "0,0 0,7 | A0,2.0,3.2,2.0,90,-90"),
    "Э": (5.9, "A2.4,3.5,2.4,3.5,220,500 | 2.3,3.5 4.6,3.5"),
    "Ю": (7.4, "1.8,0 1.8,7 | A4.4,3.5,2.4,3.5,0,360"),
    "Я": (5.6, "4.4,0 4.4,7 | A4.4,5.3,3.2,1.7,90,270 | 3.4,3.6 0.1,0"),

    "б": (5.1, "A2.2,2.5,2.2,2.5,0,360 | 0,2.5 0,6.4 | A1.6,6.4,1.6,1.5,180,60"),
    "в": (4.6, "0,0 0,5 | A0,3.7,2.1,1.3,90,-90 A0,1.2,2.3,1.2,90,-90"),
    "г": (4.0, "0,0 0,5 | 0,5 3.0,5"),
    "д": (5.0, "0.4,5 3.4,5 | 0.4,5 -0.1,-0.6 | 3.4,5 3.9,-0.6 | -0.5,0 4.5,0"),
    "ж": (5.0, "0,5 1.8,2.5 0,0 | 3.6,5 1.8,2.5 3.6,0 | 1.8,0 1.8,5"),
    "з": (4.6, "A1.6,3.7,1.6,1.3,170,-45 A1.8,1.3,1.8,1.3,55,-165"),
    "и": (5.0, "0,5 0,0 3.4,5 3.4,0"),
    "к": (4.8, "0,0 0,5 | 3.0,5 0,2.0 | 1.4,2.9 3.4,0"),
    "л": (4.8, "0,0 0.8,5 3.4,5 3.4,0"),
    "м": (5.8, "0,0 0,5 2.0,1.6 4.0,5 4.0,0"),
    "н": (5.0, "0,0 0,5 | 3.4,0 3.4,5 | 0,2.5 3.4,2.5"),
    "п": (5.0, "0,0 0,5 3.4,5 3.4,0"),
    "т": (4.6, "0,5 3.4,5 | 1.7,5 1.7,0"),
    "у": (5.0, "0,5 2.0,0.4 | 3.8,5 1.2,-1.8"),
    "ф": (5.8, "A1.8,2.5,1.8,2.5,0,360 | 1.8,-2 1.8,8"),
    "ц": (5.0, "0,0 0,5 3.4,5 3.4,-1.2 4.6,-1.2"),
    "ч": (4.8, "0,5 0,2.4 3.4,2.4 | 3.4,0 3.4,5"),
    "ш": (6.2, "0,5 0,0 4.8,0 4.8,5 | 2.4,0 2.4,5"),
    "щ": (6.6, "0,5 0,0 4.8,0 4.8,5 | 2.4,0 2.4,5 | 4.8,0 4.8,-1.2 6.0,-1.2"),
    "ъ": (4.8, "0,0 0,5 | 0,5 2.6,5 | A0,1.2,2.4,1.2,90,-90"),
    "ы": (5.6, "0,0 0,5 | A0,1.2,2.4,1.2,90,-90 | 3.8,0 3.8,5"),
    "ь": (4.8, "0,0 0,5 | A0,1.2,2.4,1.2,90,-90"),
    "э": (5.0, "A1.8,2.5,1.8,2.5,220,500 | 1.7,2.5 3.5,2.5"),
    "ю": (6.2, "1.4,0 1.4,5 | A3.4,2.5,2.0,2.5,0,360"),
    "я": (4.8, "3.0,0 3.0,5 | A3.0,3.6,2.2,1.4,90,270 | 2.8,2.4 0.2,0"),
}

ALIASES = {
    "А": "A", "В": "B", "Е": "E", "З": "3", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
}

MARKS = {
    "acute": ("-1.2,0 1.4,2.8", "above"),
    "grave": ("1.2,0 -1.4,2.8", "above"),
    "circumflex": ("-1.7,0.2 0,2.6 1.7,0.2", "above"),
    "tilde": ("A-0.95,1.0,0.95,0.9,210,0 A0.95,1.0,0.95,0.9,180,390", "above"),
    "diaeresis": ("-1.2,1.0 | 1.2,1.0", "above"),
    "ring": ("A0,1.3,1.15,1.3,0,360", "above"),
    "breve": ("A0,1.6,1.8,1.6,150,390", "above"),
    "cedilla": ("0.2,0 0.2,-1.4 A-1.0,-1.4,1.2,1.2,0,-90", "below"),
}

COMPOSED = {
    "À": ("A", "grave"), "Á": ("A", "acute"), "Â": ("A", "circumflex"),
    "Ã": ("A", "tilde"), "Ä": ("A", "diaeresis"), "Å": ("A", "ring"),
    "Ç": ("C", "cedilla"),
    "È": ("E", "grave"), "É": ("E", "acute"), "Ê": ("E", "circumflex"),
    "Ë": ("E", "diaeresis"),
    "Ì": ("I", "grave"), "Í": ("I", "acute"), "Î": ("I", "circumflex"),
    "Ï": ("I", "diaeresis"),
    "Ñ": ("N", "tilde"),
    "Ò": ("O", "grave"), "Ó": ("O", "acute"), "Ô": ("O", "circumflex"),
    "Õ": ("O", "tilde"), "Ö": ("O", "diaeresis"),
    "Ù": ("U", "grave"), "Ú": ("U", "acute"), "Û": ("U", "circumflex"),
    "Ü": ("U", "diaeresis"),
    "Ý": ("Y", "acute"),
    "à": ("a", "grave"), "á": ("a", "acute"), "â": ("a", "circumflex"),
    "ã": ("a", "tilde"), "ä": ("a", "diaeresis"), "å": ("a", "ring"),
    "ç": ("c", "cedilla"),
    "è": ("e", "grave"), "é": ("e", "acute"), "ê": ("e", "circumflex"),
    "ë": ("e", "diaeresis"),
    "ì": ("i", "grave"), "í": ("i", "acute"), "î": ("i", "circumflex"),
    "ï": ("i", "diaeresis"),
    "ñ": ("n", "tilde"),
    "ò": ("o", "grave"), "ó": ("o", "acute"), "ô": ("o", "circumflex"),
    "õ": ("o", "tilde"), "ö": ("o", "diaeresis"),
    "ù": ("u", "grave"), "ú": ("u", "acute"), "û": ("u", "circumflex"),
    "ü": ("u", "diaeresis"),
    "ý": ("y", "acute"), "ÿ": ("y", "diaeresis"),

    "Ё": ("Е", "diaeresis"), "ё": ("е", "diaeresis"),
    "Й": ("И", "breve"), "й": ("и", "breve"),
}

DOTTED = ("i", "j")

LIGATURES = {
    "\ufb00": ("f", "f"),
    "\ufb01": ("f", "i"),
    "\ufb02": ("f", "l"),
    "\ufb03": ("f", "f", "i"),
    "\ufb04": ("f", "f", "l"),
}


def names() -> list:
    """Every *character* the face has a skeleton for, in a stable order.

    The joins (:data:`LIGATURES`) are deliberately not in here.  A ligature is a glyph
    the typesetter reaches for and not a character a writer types, and a name list is
    the answer to "what can a text contain", which is a different question from "what
    can this face draw" -- :func:`has` answers the second one.
    """
    return sorted(set(GLYPHS) | set(ALIASES) | set(COMPOSED))


def has(ch: str) -> bool:
    """Whether ``ch`` is in the face.  A missing character is an error, not a box."""
    key = str(ch)
    return key in GLYPHS or key in ALIASES or key in COMPOSED or key in LIGATURES


def mark_of(ch: str):
    """The ``(mark, base)`` this character is written as, or ``None``.

    The pair is what the pen needs: a mark and the letter it goes over.  Where the mark
    sits is not here, because it is not a property of the letter -- it depends on the
    ink the pen has already laid down (see :func:`gen.text.pen.strokes`).
    """
    entry = COMPOSED.get(str(ch))
    return None if entry is None else (entry[1], entry[0])


def mark_strokes(mark: str) -> list:
    """A mark's own skeleton: centred at ``x = 0``, its ink on ``y = 0``."""
    if mark not in MARKS:
        raise KeyError(f"no mark called {mark!r}: there are {len(MARKS)}, "
                       + ", ".join(sorted(MARKS)))
    return _read(MARKS[mark][0], mark)


def _base(ch: str) -> str:
    """The key a character's strokes come from, once aliases are followed.

    Three ways to have a glyph, and they are three different facts: its own entry, the
    entry of a letter it shares a form with, or -- for a letter wearing a mark -- its
    base's, because the mark is not part of the letter's form.
    """
    key = str(ch)
    if key in COMPOSED:
        key = COMPOSED[key][0]
    if key in ALIASES:
        key = ALIASES[key]
    return key


def advance(ch: str) -> float:
    """The glyph's own advance, in em units, with no tracking of any kind.

    A composed letter advances by its *base*: an accent hangs over the letter, and a
    letter with something over it is not a wider letter.  (The pen may still widen it,
    the same way a serif does, if the mark reaches past the sidebearing it was given --
    see :func:`gen.text.pen.advance`.)
    """
    key = str(ch)
    if key in LIGATURES:
        return _compose(LIGATURES[key])[1]
    key = _base(key)
    if key not in GLYPHS:
        raise KeyError(f"no skeleton for {str(ch)!r}: the face covers {len(names())} "
                       f"characters, and a missing one is a mistake rather than a box")
    return GLYPHS[key][0] / 10.0


def _arc(cx, cy, rx, ry, a0, a1, into):
    """Append an elliptical arc, sampled at :data:`ARC_STEP` degrees."""
    span = float(a1) - float(a0)
    n = max(1, int(round(abs(span) / ARC_STEP)))
    for i in range(n + 1):
        a = np.radians(float(a0) + span * i / n)
        into.append((cx + rx * np.cos(a), cy + ry * np.sin(a)))


def _read(text: str, key: str) -> list:
    """A strokes-string read into polylines in em units."""
    out = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        pts = []
        for token in part.split():
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


def _undotted(pts: list) -> list:
    """The letter without its dot: a one-point stroke above the x-height is the dot."""
    dots = [i for i, s in enumerate(pts) if len(s) == 1 and s[0][1] > X_HEIGHT]
    if not dots:
        return pts
    drop = max(dots, key=lambda i: pts[i][0][1])
    return [s for i, s in enumerate(pts) if i != drop]


def _raw(ch: str) -> str:
    """A character's own strokes-string, before the pen or a mark touches it."""
    return GLYPHS[_base(ch)][1]


def _stem_x(pts: list, part: str) -> float:
    """Where a part's stem stands.

    An ``f``'s stem is the *tail* of its hook's stroke -- the arc comes down to the
    stem's shoulder and the stroke runs on to the foot -- while an ``i``'s or an
    ``l``'s stem is the stroke itself, and starts at its foot.
    """
    stroke = pts[0]
    return float(stroke[-1][0] if part == "f" else stroke[0][0])


def _compose(parts) -> tuple:
    """A join's polylines and its design width, in em units: ``(strokes, design)``.

    Left to right: each letter stands where the one before it ended *in design*, and an
    ``i`` drops its dot.  That is the whole composition -- the strokes are the letters'
    own, shifted -- and the design width of the join is the last letter's end, so a
    ligature is the letters it replaces with the daylight between them taken out.
    """
    out = []
    end = 0.0
    bar_end = None
    for j, part in enumerate(parts):
        pts = _read(_raw(part), part)
        if part in DOTTED:
            pts = _undotted(pts)
        dx = 0.0 if j == 0 else end - _stem_x(pts, part)
        if bar_end is not None and part in DOTTED:

            stem = pts[0]
            pts = [[(stem[0][0], stem[0][1]), (stem[-1][0], stem[-1][1]),
                    (bar_end - dx, stem[-1][1])]] + pts[1:]
        out += [[(px + dx, py) for (px, py) in stroke] for stroke in pts]
        bar_end = dx + float(pts[1][-1][0]) if part == "f" else None
        end = dx + GLYPHS[_base(part)][0] / 10.0
    return out, end


def strokes(ch: str) -> list:
    """The glyph's skeleton: a list of polylines, in em units, y up.

    ``(0, 0)`` is the baseline at the glyph's origin, so a caller can place the glyph
    by an offset and nothing else.  A stroke of one point is a dot.

    Two things worth knowing at the call site.  A letter wearing a mark gives the
    **base's** strokes and not the mark's: the mark is not part of the letter's form,
    and where it goes is the pen's question, so :func:`gen.text.pen.strokes` is what
    puts it in place.  And an *i* or a *j* under a mark gives up its dot, because the
    mark lands where the dot was.
    """
    key = str(ch)
    if key in LIGATURES:
        return _compose(LIGATURES[key])[0]
    base = _base(key)
    if base not in GLYPHS:
        raise KeyError(f"no skeleton for {key!r}")
    out = _read(GLYPHS[base][1], key)
    entry = COMPOSED.get(key)
    if entry is not None and entry[0] in DOTTED and MARKS[entry[1]][1] == "above":
        out = _undotted(out)
    return out
