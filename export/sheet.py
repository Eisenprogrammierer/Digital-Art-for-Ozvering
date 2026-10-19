"""A sprite sheet: one sprite over time, on a grid, with the timeline it plays on.

:mod:`export.atlas` answers "where did each *sprite* go"; this answers the other half
of the same question, for a set of frames that is one sprite at different moments.
Two things make the difference, and both are decisions rather than conveniences:

* **A frame is a cell, not a trim.**  An atlas trims each member to its own pixels,
  because sprites are different shapes; the frames of one animation are the same size
  by construction, and an engine that has to re-centre every frame is an engine with a
  character that rolls.  So equal cells are *required* rather than hoped for, and a set
  of frames of different sizes is refused with the name of the thing that can pack it.
* **The timeline is data.**  A frame rate and a loop flag are what an engine needs to
  play the sheet, and the value of the axis each frame was drawn along is what makes
  the animation *reproducible* rather than merely ordered.  A picture of an animation
  without its timeline is a contact sheet.

Nothing here draws.  The frames arrive drawn, and the only thing this module decides is
where they sit and when they play -- so every pixel of the sheet is a copy of a pixel
of a frame, which is what lets a test compare them byte for byte.  The drawing itself
is :func:`core.render.sheet`, and the layout is :func:`core.render.cells`, so the table
written here and the pixels drawn there are the same arithmetic.
"""
from __future__ import annotations

import numpy as np

from core import render

__all__ = ["build"]


def build(frames, cols: int | None = None, padding: int = 1, fps: float = 12.0,
          loop: bool = True, params=None, background=None):
    """Lay frames out on a grid; returns ``(canvas, info)``.

    ``info`` is the sidecar: the sheet's size, its cell and padding, the frame rate and
    whether it loops, and one entry per frame giving the rectangle it occupies plus the
    value of the axis it was drawn along (``params``) -- for a waving flag, the phase of
    its cloth in radians.  The phase is what makes the sheet *regenerable*: the picture
    says which frames there are, and the numbers say what each of them is.

    ``cols`` defaults to the squarest grid the count allows.  That is not only the
    smallest sheet: a strip of forty frames is a texture wider than some hardware will
    sample, and the square one and the strip are the same picture to an engine that has
    the table.

    A frame rate of zero and a negative padding are refused rather than clamped: both
    are silent wrong answers, one that freezes the animation and one that overlays the
    frames, and neither is what the caller meant.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("a sheet of no frames is an empty file")
    sizes = sorted({(int(c.w), int(c.h)) for c in frames})
    if len(sizes) > 1:
        raise ValueError(f"the frames of a sheet are all the same size, and these are "
                         f"{sizes} -- frames of different shapes are what "
                         f"export.atlas packs")
    cw, ch = sizes[0]
    count = len(frames)
    cols = int(np.ceil(np.sqrt(count))) if cols is None else int(cols)
    if cols < 1:
        raise ValueError("a sheet is at least one column wide")
    if int(padding) < 0:
        raise ValueError("a negative padding overlays the frames")
    if float(fps) <= 0.0:
        raise ValueError("a frame rate is a positive number of frames per second")
    if params is not None:
        params = [float(p) for p in params]
        if len(params) != count:
            raise ValueError(f"{count} frames have {count} parameters, not "
                             f"{len(params)}")

    canvas = render.sheet(frames, cols=cols, gap=int(padding), background=background)
    rects = [[x, y, cw, ch]
             for x, y in render.cells(count, cols, cw, ch, int(padding))]
    info = {
        "size": [int(canvas.w), int(canvas.h)],
        "cols": cols,
        "rows": (count + cols - 1) // cols,
        "cell": [cw, ch],
        "padding": int(padding),
        "fps": float(fps),
        "loop": bool(loop),
        "frames": [{"index": i, "rect": rect,
                    **({"param": params[i]} if params is not None else {})}
                   for i, rect in enumerate(rects)],
    }
    return canvas, info
