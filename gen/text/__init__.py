"""Procedural text: an alphabet that is generated, not licensed.

Fonts are the one asset a game cannot generate -- or could not, until the letters
were drawn the way everything else here is.  The package has three parts and they
are the same three parts the rest of the toolkit has:

``skeleton``
    The alphabet itself: one polyline per stroke, in em units, drawn on the guides
    every face agrees about.  Identity is these numbers and nothing else.
``pen``
    What turns a skeleton into a letter -- weight, contrast, serifs, slant -- and the
    ink box that comes out.  Style is these four numbers and nothing else.
``typeset``
    Advances, tracking, alignment, and the drawing: a line of text is a distance
    field like any other shape, so it takes the same coverage and the same lit bevel.

The point of the split is that a monoline UI face and a modulated book face are the
*same alphabet* read two ways, which is what makes a generated font worth having
rather than a novelty: the face is a parameter, and the parameters ship in the
sidecar next to the picture.
"""
from . import pen, skeleton, typeset

__all__ = ["pen", "skeleton", "typeset"]
