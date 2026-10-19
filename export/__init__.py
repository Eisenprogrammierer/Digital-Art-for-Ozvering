"""Export back-ends.  PNG raster, SVG vector, atlas packing, sprite sheets, credits.

Four of these answer "what is the asset in"?  ``png`` writes the pixels, ``svg``
writes the curves where the asset has any, ``atlas`` packs several sprites into one
sheet, and ``sheet`` lays the frames of *one* sprite out over time and says when they
play.  The fifth is not an asset format at all: the pre-ship check lives in
:mod:`core.credits`, and it is mentioned here because a build directory is what it
reads.
"""
from . import atlas, png, sheet, svg

__all__ = ["atlas", "png", "sheet", "svg"]
