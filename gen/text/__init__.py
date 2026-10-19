"""Procedural text: an alphabet that is generated, not licensed.

Fonts are the one asset a game cannot generate -- or could not, until the letters
were drawn the way everything else here is.  The package has four parts and they
are the same parts the rest of the toolkit has:

``skeleton``
    The alphabet itself: one polyline per stroke, in em units, drawn on the guides
    every face agrees about.  Identity is these numbers and nothing else.
``pen``
    What turns a skeleton into a letter -- weight, contrast, serifs, slant -- and the
    ink box that comes out.  Style is these four numbers and nothing else.
``kern``
    What happens between two letters and cannot be known by either one: the pair's
    profiles read out of the pen's own field, and the daylight the pair is allowed to
    spend.  A rule with three numbers and no table of pairs.
``typeset``
    Advances, kerns, tracking, alignment, and the drawing: a line of text is a
    distance field like any other shape, so it takes the same coverage and the same
    lit bevel.

The point of the split is that a monoline UI face and a modulated book face are the
*same alphabet* read two ways, which is what makes a generated font worth having
rather than a novelty: the face is a parameter, and the parameters ship in the
sidecar next to the picture.
"""
from . import kern, pen, skeleton, typeset

__all__ = ["kern", "pen", "skeleton", "typeset"]
