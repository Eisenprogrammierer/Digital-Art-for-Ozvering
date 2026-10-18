"""Export back-ends.  PNG raster, SVG vector, atlas packing, and the credits gate.

Three of these answer "what is the asset in"?  ``png`` writes the pixels, ``svg``
writes the curves where the asset has any, ``atlas`` packs several into one sheet.
The fourth is not an asset format at all: the pre-ship check lives in
:mod:`core.credits`, and it is mentioned here because a build directory is what it
reads.
"""
from . import atlas, png, svg

__all__ = ["atlas", "png", "svg"]
