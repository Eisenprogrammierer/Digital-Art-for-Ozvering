"""A sheet of materials: tiles in one image, with margins that are the tiles.

A game does not bind ten floor textures, it binds one, so the last thing a texture set
needs is what a sprite set needs -- a packer and a table of rectangles.  The layout
here *is* :func:`export.atlas.pack`, and it is the same bottom-left skyline, for the
same reason.  What separates this module from a call to :func:`export.atlas.build` is
that every rule the sprite atlas is built on is the wrong rule for a tile:

* A sprite is **trimmed** to the pixels it uses, because the transparent margin around
  a sprite is not part of it.  A tile must not be trimmed at all, because a tile's edge
  is not a margin -- it is the seam.  Crop the empty-looking border off a floor tile
  and the tile no longer joins to itself: the pixels that were cropped are the ones the
  wrap was made of, and the trim would remove exactly what the texture spent its whole
  design earning.
* A sprite's gap in the sheet is **padding**, and its job is to be *nothing* -- a
  filter that steps off a sprite should find transparent pixels rather than the
  neighbour's.  A tile's margin cannot be nothing, because a tile is opaque and the
  margin is exactly what a filtered sample near the edge is going to read.  So the
  margin is filled with the tile's **own continuation**: the tile, rolled by the margin,
  so that a sample stepping off the edge sees the tile as it would be if it were laid
  out beside itself.  That is not a compromise between neighbours and a gap; for a
  seamless texture it is *exactly right*, and the tile's own wrap is the thing that
  makes it right.

The margin is a *bleed*, in the printer's sense, except that instead of ink running past
the cut it is the tile running past itself.  Two pixels is what a bilinear filter needs
-- the sample one pixel outside the tile is the tile's last pixel, and a GPU blending
four texels to place it is blending the tile with itself -- and it costs about six
percent of the sheet.

Because the margin is the tile rather than a gap, a mip chain built from this sheet
behaves too: the hardware averages each tile together with its own continuation, which
is the average the tile would have had if it were tiled, and the only place that goes
wrong is a few pixels either side of the *shared* boundary between two different tiles,
at the coarsest levels.  A per-level sheet is still the honest way to ship mips, since
halving the tiles wants a fresh packing for each level; that is a separate job.
"""
from __future__ import annotations

import numpy as np

from export import atlas as packing

__all__ = ["bleed", "build"]

#: Pixels of a tile's own continuation on every side of it in the sheet.
MARGIN = 2


def bleed(canvas, margin: int = MARGIN) -> np.ndarray:
    """The 8-bit RGBA tile with ``margin`` pixels of itself on every side.

    A wrap-pad and not a padding: the pixel above the tile's first row is the tile's
    last row, the pixel left of its first column is its last column, and the corner
    between them is the tile's far corner.  Every pixel of the result is a pixel of the
    tile -- there is nothing in the margin that is not material, which is the whole
    difference between this and a gap.
    """
    a = np.asarray(canvas.to_uint8(), dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 4:
        raise ValueError(f"a tile is 8-bit RGBA, got {a.shape}")
    m = int(margin)
    if m < 0:
        raise ValueError("a margin cannot be negative")
    if m == 0:
        return a
    return np.pad(a, ((m, m), (m, m), (0, 0)), mode="wrap")


def build(members, margin: int = MARGIN, max_size: int = 4096,
          power_of_two: bool = True):
    """Pack ``(name, canvas)`` pairs into one sheet; returns ``(canvas, info)``.

    ``info`` is the sidecar: the sheet's size, its measured density, the margin, and
    one entry per member giving the rectangle its **tile** occupies -- the rectangle a
    game samples, inside the margin and not including it -- and the size of the tile it
    came from, which is the rectangle's size, because nothing was trimmed.

    Members are packed with no padding between them, which is the reverse of the sprite
    atlas: a tile has no business being near a gap, because the only thing a filtered
    sample beside it should be able to reach is more of the same tile.
    """
    items = sorted((str(name), canvas) for name, canvas in members)
    if not items:
        raise ValueError("an atlas with no members is an empty file")
    m = int(margin)
    if m < 0:
        raise ValueError("a margin cannot be negative")

    blocks, sources = [], []
    for _name, canvas in items:
        blocks.append(bleed(canvas, m))
        sources.append((int(canvas.w), int(canvas.h)))

    rects, size = packing.pack([(int(b.shape[1]), int(b.shape[0])) for b in blocks],
                               padding=0, max_size=max_size,
                               power_of_two=power_of_two)
    W, H = size
    sheet = np.zeros((H, W, 4), dtype=np.uint8)
    tiles, entries = [], []
    for (name, _canvas), (bx, by, bw, bh), block, (w, h) in zip(items, rects, blocks,
                                                                sources):
        sheet[by:by + bh, bx:bx + bw] = block
        rect = (bx + m, by + m, w, h)
        tiles.append(rect)
        entries.append({"name": name, "rect": [int(v) for v in rect],
                        "source": [w, h]})
    info = {
        "size": [int(W), int(H)],
        "density": round(packing.density(tiles, size), 4),
        "margin": m,
        "power_of_two": bool(power_of_two),
        "members": entries,
    }
    return packing.canvas_from(sheet), info
