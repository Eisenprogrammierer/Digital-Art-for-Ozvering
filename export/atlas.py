"""Packing sprites into one image, and saying where each one went.

A game does not load forty PNGs, it loads one and a table of rectangles, so this
is the step between "we can generate an asset" and "we can ship a set of them".
Two decisions live here, and both are *measurable* rather than stylistic:

* Members are **trimmed** before they are packed -- the transparent margin around a
  sprite is not part of the sprite -- and the trim is recorded, so a caller can put
  the origin back where it was.  The density of the resulting atlas is then a number
  in the sidecar instead of a claim in a comment.
* The layout is **deterministic**: a skyline packer fed members sorted by name, and
  a width chosen by trying a handful and keeping the tightest.  The same members
  always produce the same atlas, byte for byte, which is the property a content
  pipeline actually depends on.

The packer is a *bottom-left skyline* rather than a shelf: a shelf wastes the whole
band above a short sprite, and a set of icons with one banner in it is exactly the
case where that shows.  When the caller asks for power-of-two dimensions the
rounding is done after the layout and its cost shows up in the density, which is
the point of reporting the density at all.
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.canvas import Canvas

__all__ = ["trim", "pack", "build", "density", "canvas_from"]

#: Aspect ratios to try when no width is given.  A wide atlas and a tall one both
#: pack badly once power-of-two rounding is applied, so the search is worth the few
#: extra milliseconds and its answer is deterministic.
_TRIES = (0.90, 1.15, 1.45, 1.85, 2.40, 3.20)


def trim(rgba, threshold: int = 0) -> tuple:
    """The bounding box of what is actually drawn: ``(x0, y0, x1, y1)``, right-open.

    ``threshold`` is an alpha in 0..255 and defaults to zero, which counts even the
    faintest anti-aliased pixel.  That is deliberate: an 8-bit edge pixel *is*
    content, and a trim that rounds it away is a trim that clips the sprite.
    """
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] != 4:
        raise ValueError(f"an atlas works in 8-bit RGBA, got {a.shape}")
    ys, xs = np.nonzero(a[..., 3] > int(threshold))
    if ys.size == 0:
        raise ValueError("a fully transparent member has nothing to pack")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def density(rects, size) -> float:
    """The fraction of an atlas that is sprite rather than gap.

    The honest way to report how good a packing is, and the number that makes
    power-of-two rounding a cost you can see rather than a rule you obey.
    """
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0:
        return 0.0
    used = sum(int(r[2]) * int(r[3]) for r in rects)
    return float(used) / float(w * h)


def _pot(n: int) -> int:
    """The next power of two at or above ``n`` (and ``1`` for ``n <= 1``)."""
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


def _fit(size, power_of_two: bool) -> tuple:
    return (_pot(size[0]), _pot(size[1])) if power_of_two else size


def _pack_at(sizes, width: int, padding: int, max_size: int):
    """One bottom-left skyline pass at a fixed width.

    ``sizes`` are the *content* sizes; ``padding`` is added to the allocation of
    each, so that two sprites are never flush and a bilinear filter on the GPU has
    somewhere to blend into that is not the neighbour.
    """
    width = int(width)
    if width <= 0:
        raise ValueError("an atlas needs a positive width")
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i][1], -sizes[i][0], i))
    sky = [(0, 0, width)]          # (x, y, w), left to right, a partition of [0, width)
    rects = [None] * len(sizes)
    used_w = used_h = 0
    for i in order:
        w, h = sizes[i]
        need = w + padding
        if need > width or h + padding > max_size:
            raise ValueError(f"a {w}x{h} member does not fit in a {max_size} px atlas")
        best = None
        for si in range(len(sky)):
            sx = sky[si][0]
            if sx + need > width:
                continue
            y = sky[si][1]
            left, k = need, si
            ok = True
            while left > 0:
                if k >= len(sky):
                    ok = False
                    break
                y = max(y, sky[k][1])
                left -= sky[k][2]
                k += 1
            if not ok or y + h + padding > max_size:
                continue
            if best is None or (y, sx) < (best[1], best[0]):
                best = (sx, y)
        if best is None:
            raise ValueError(f"no room left at width {width}")
        x, y = best
        rects[i] = (x, y, w, h)
        used_w = max(used_w, x + w)
        used_h = max(used_h, y + h)
        # Raise the skyline over [x, x + need) to the top of what was just placed,
        # keeping whatever sticks out on either side.
        raised = []
        for sx, sy, sw in sky:
            if sx + sw <= x or sx >= x + need:
                raised.append((sx, sy, sw))
                continue
            if sx < x:
                raised.append((sx, sy, x - sx))
            if sx + sw > x + need:
                raised.append((x + need, sy, sx + sw - (x + need)))
        raised.append((x, y + h + padding, need))
        raised.sort()
        sky = []
        for sx, sy, sw in raised:                     # merge neighbours at one height
            if sky and sky[-1][0] + sky[-1][2] == sx and sky[-1][1] == sy:
                sky[-1] = (sky[-1][0], sky[-1][1], sky[-1][2] + sw)
            else:
                sky.append((sx, sy, sw))
    return rects, (used_w, used_h)


def pack(sizes, width=None, padding: int = 1, max_size: int = 4096,
         power_of_two: bool = True):
    """Place ``(w, h)`` boxes; returns ``(rects, (W, H))`` in the order given.

    The rects come back in the caller's order, not the packer's, so a caller never
    has to un-sort anything to know which rectangle is which sprite.  With no
    ``width``, a handful of candidates are tried and the one whose *final* sheet is
    the squarest wins, with the smallest sheet breaking a tie -- which is how
    power-of-two rounding is paid for rather than suffered.
    """
    sizes = [(int(w), int(h)) for w, h in sizes]
    if not sizes:
        return [], (0, 0)
    if min(min(w, h) for w, h in sizes) < 1:
        raise ValueError("a member with no pixels cannot be packed")
    if width is not None:
        rects, size = _pack_at(sizes, int(width), int(padding), int(max_size))
        final = _fit(size, power_of_two)
        if max(final) > int(max_size):
            raise ValueError(f"an atlas of {final} is larger than max_size={max_size}")
        return rects, final

    total = sum((w + padding) * (h + padding) for w, h in sizes)
    wmin = max(w for w, h in sizes) + padding
    # The obvious widths: the narrowest that works, the square one, and the powers
    # of two in between -- because a power-of-two width that packs perfectly is the
    # difference between a full atlas and a half-empty one.
    candidates = {wmin}
    candidates.update(int(np.ceil(np.sqrt(total * k))) for k in _TRIES)
    k, limit = 1, 4 * int(np.sqrt(total)) + padding
    while k <= limit:
        candidates.add(k)
        k *= 2
    best = None
    for candidate in sorted(candidates):
        if candidate < wmin:
            continue
        try:
            rects, size = _pack_at(sizes, candidate, int(padding), int(max_size))
        except ValueError:
            continue
        final = _fit(size, power_of_two)
        if max(final) > int(max_size):
            continue                       # the sheet would be over the size limit
        area = final[0] * final[1]
        # The squarest sheet wins, and among equals the smallest one.  Area alone
        # would happily return a 136x2205 strip, which is the smallest number of
        # pixels and the worst texture: tall sheets run into hardware limits and are
        # impossible to look at.  This is also what keeps the atlases of very
        # different sets comparable in shape.
        score = (max(final[0], final[1]), area, final[0])
        if best is None or score < best[0]:
            best = (score, rects, final)
    if best is None:
        raise ValueError(f"the members do not fit in a {max_size}x{max_size} atlas")
    return best[1], best[2]


def canvas_from(rgba) -> Canvas:
    """A :class:`~core.canvas.Canvas` from an 8-bit RGBA buffer.

    Atlases are assembled in 8-bit bytes, because that is what a sprite sheet is and
    because a byte is the only unit in which "the pixels were moved and not redrawn"
    is a statement precise enough to test.
    """
    a = np.asarray(rgba, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 4:
        raise ValueError(f"an atlas works in 8-bit RGBA, got {a.shape}")
    out = Canvas(int(a.shape[1]), int(a.shape[0]), scale=1.0)
    out.rgb = palette.srgb_to_linear(a[..., :3].astype(np.float64) / 255.0)
    out.a = a[..., 3].astype(np.float64) / 255.0
    return out


def build(members, padding: int = 2, max_size: int = 4096,
          power_of_two: bool = True):
    """Pack ``(name, canvas)`` pairs into one image; returns ``(canvas, info)``.

    ``info`` is the sidecar: the atlas size, its measured density, and one entry per
    member giving the rectangle it occupies, the size of the sprite it came from and
    the trim that was applied to it.  The pixels themselves are a **copy**: a sprite
    in an atlas is not re-rendered, it is moved, which is why the tests can compare
    the two byte for byte.
    """
    items = sorted((str(name), canvas) for name, canvas in members)
    if not items:
        raise ValueError("an atlas with no members is an empty file")
    blocks, trims, sizes = [], [], []
    for _name, canvas in items:
        rgba = np.asarray(canvas.to_uint8(), dtype=np.uint8)
        x0, y0, x1, y1 = trim(rgba)
        blocks.append(rgba[y0:y1, x0:x1])
        trims.append([int(x0), int(y0), int(x1), int(y1)])
        sizes.append((x1 - x0, y1 - y0))

    rects, size = pack(sizes, padding=padding, max_size=max_size,
                       power_of_two=power_of_two)
    W, H = size
    out = np.zeros((H, W, 4), dtype=np.uint8)
    entries = []
    for (name, canvas), (x, y, w, h), block, trim_box in zip(items, rects, blocks, trims):
        out[y:y + h, x:x + w] = block
        entries.append({
            "name": name,
            "rect": [int(x), int(y), int(w), int(h)],
            "source": [int(canvas.w), int(canvas.h)],
            "trim": trim_box,
        })

    atlas = canvas_from(out)
    info = {
        "size": [int(W), int(H)],
        "density": round(density(rects, size), 4),
        "padding": int(padding),
        "power_of_two": bool(power_of_two),
        "members": entries,
    }
    return atlas, info
