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
halving the tiles wants a fresh packing for each level -- a quarter-size tile with a
two-texel margin needs a different sheet, not a quarter of this one -- and that is what
:func:`mips` is for.

A level is built by :func:`reduce`-ing the tiles and then packing them, and the two
halves of that sentence are each a decision.  The reduction averages a 2x2 block **in
linear light**, because that is the average the eye is entitled to, and it averages
**periodically**, because a tile has no edge -- so the reduced tile is still seamless,
which is the one property a mip chain can destroy and the one that would be invisible
until the coarsest level of a floor.
"""
from __future__ import annotations

import numpy as np

from core.canvas import Canvas
from export import atlas as packing

__all__ = ["bleed", "build", "mips", "reduce"]

#: Pixels of a tile's own continuation on every side of it in the sheet.
MARGIN = 2


def reduce(canvas) -> Canvas:
    """One mip level down: a 2x2 box average, in linear light and on the torus.

    Two decisions, and both are the difference between a mip chain and a picture that
    merely looks like one.

    The average is taken in **linear light**.  :class:`~core.canvas.Canvas` holds linear
    RGB, so this is the arithmetic the eye is entitled to; averaging sRGB-encoded values
    instead would darken every level, and darken it by more the further down the chain
    you go, which is the kind of drift that is invisible in code and obvious in a mip.

    The average is taken **periodically**.  A tile has no edge, so every texel's
    neighbours wrap, and an odd-sized row is averaged with the row it already meets
    rather than with an invented border.  A non-periodic filter would have to anchor on
    an edge the tile does not have, and the last level of a floor would be a picture of
    a floor with a frame around it.

    Alpha is resampled the way a resample has to be: the colour is averaged
    *premultiplied* and divided back out, because a transparent texel has no colour to
    contribute and counting it as black would drag the edge of a sprite inward.  A tile
    of one texel is already as small as a level goes, so it comes back unchanged.
    """
    rgb = np.asarray(canvas.rgb, dtype=np.float64)
    a = np.asarray(canvas.a, dtype=np.float64)
    h, w = a.shape
    if h < 1 or w < 1:
        raise ValueError(f"a tile has pixels, got {h}x{w}")
    if h == 1 and w == 1:
        return canvas.clone()
    pad_y, pad_x = h % 2, w % 2
    if pad_y or pad_x:
        rgb = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)), mode="wrap")
        a = np.pad(a, ((0, pad_y), (0, pad_x)), mode="wrap")
    oh, ow = h // 2 + pad_y, w // 2 + pad_x
    out = Canvas(ow, oh, scale=canvas.scale, center=canvas.center, y_up=canvas.y_up)
    out.a = a.reshape(oh, 2, ow, 2).mean(axis=(1, 3))
    premultiplied = (rgb * a[..., None]).reshape(oh, 2, ow, 2, 3).mean(axis=(1, 3))
    opaque = out.a[..., None] > 0.0
    out.rgb = np.where(opaque, premultiplied / np.where(opaque, out.a[..., None], 1.0), 0.0)
    return out


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


def level_count(members) -> int:
    """How many levels a part of this set of tiles can halve down to.

    The chain is as long as the *largest* tile supports, and the smaller ones simply
    reach one texel early and stay there.  That is what an engine expects -- a mip chain
    is one length for the whole set -- and it is why the number is a property of the set
    rather than of a tile.
    """
    items = [(str(name), canvas) for name, canvas in members]
    if not items:
        raise ValueError("an atlas with no members is an empty file")
    top = max(max(int(c.w), int(c.h)) for _name, c in items)
    levels = 1
    while top > 1:
        top = (top + 1) // 2
        levels += 1
    return levels


def mips(members, levels: int | None = None, margin: int = MARGIN,
         max_size: int = 4096, power_of_two: bool = True) -> list:
    """A sheet per mip level, each packed fresh; returns ``[(canvas, info), ...]``.

    The chain starts at level 0, which is exactly what :func:`build` returns -- so the
    single sheet is the head of the chain and not a different picture -- and each level
    after it is the level above reduced and packed again.

    Packing again is the whole point.  A quarter-size tile with two texels of margin
    wants a sheet of its own: the margin is the same *two texels* at every level, since
    that is the reach of the filter that reads it, so it is four times as large a
    fraction of a quarter-size tile as of the original.  Keeping level 0's rectangles
    and letting the hardware average would leave each level with a margin that shrinks
    away exactly as it becomes needed.

    ``levels`` defaults to the longest chain the set supports (:func:`level_count`).
    Asking for more is refused rather than clamped, because level 8 of a 64-pixel tile
    is level 6 over again, and a chain that quietly repeats itself is a chain that
    quietly lies about its own depth.
    """
    items = sorted((str(name), canvas) for name, canvas in members)
    if not items:
        raise ValueError("an atlas with no members is an empty file")
    most = level_count(items)
    n = most if levels is None else int(levels)
    if n < 1:
        raise ValueError(f"a mip chain has at least one level, not {n}")
    if n > most:
        raise ValueError(
            f"these tiles halve down to {most} levels, so level {n} would be level "
            f"{most - 1} over again")

    names = [name for name, _canvas in items]
    canvases = [canvas for _name, canvas in items]
    chain = []
    for k in range(n):
        sheet, info = build(list(zip(names, canvases)), margin=margin,
                            max_size=max_size, power_of_two=power_of_two)
        chain.append((sheet, {**info, "level": k, "levels": n}))
        canvases = [reduce(canvas) for canvas in canvases]
    return chain
