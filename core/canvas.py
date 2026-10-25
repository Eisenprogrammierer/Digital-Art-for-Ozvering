"""Raster canvas: a world-space camera plus a linear-light RGBA accumulator.

Two deliberate decisions:

* **Linear light internally.**  Alpha compositing and lighting are physically
  meaningful sums, so a 50 % grey over black really is half the light.  The
  sRGB transfer function is applied only on export (:meth:`to_uint8`).
* **World coordinates, not pixels.**  A :class:`Canvas` carries a scale that
  maps world units to pixels, so an SDF written once renders identically at any
  output size -- the basis for generating a 24 px icon and a 512 px hero asset
  from the same description.

Y is up by default (maths convention): row 0 of the array is the top of the
image, world ``+y`` points up.
"""
from __future__ import annotations

import numpy as np

from . import palette

__all__ = ["Canvas"]


class Canvas:
    """An RGBA image buffer in linear light with an associated world frame."""

    def __init__(
        self,
        width: int,
        height: int,
        scale: float | None = None,
        center: tuple = (0.0, 0.0),
        y_up: bool = True,
        background=None,
    ):
        if width < 1 or height < 1:
            raise ValueError("canvas must be at least 1x1")
        self.w = int(width)
        self.h = int(height)
        self.scale = float(scale) if scale is not None else min(self.w, self.h) / 2.0
        self.center = (float(center[0]), float(center[1]))
        self.y_up = bool(y_up)
        self.rgb = np.zeros((self.h, self.w, 3), dtype=np.float64)
        self.a = np.zeros((self.h, self.w), dtype=np.float64)
        self._grid = None
        if background is not None:
            self.rgb[:] = palette.srgb_to_linear(background)
            self.a[:] = 1.0

    @property
    def dx(self) -> float:
        """World units per pixel -- the natural anti-aliasing width."""
        return 1.0 / self.scale

    @property
    def size(self) -> tuple:
        return (self.w, self.h)

    def grid(self):
        """World coordinates of every pixel centre, as ``(X, Y)`` meshgrids."""
        if self._grid is None:
            xs = self.center[0] + (np.arange(self.w, dtype=np.float64) + 0.5 - self.w / 2.0) / self.scale
            ys = (np.arange(self.h, dtype=np.float64) + 0.5 - self.h / 2.0) / self.scale
            ys = self.center[1] + (ys if not self.y_up else -ys)
            self._grid = np.meshgrid(xs, ys)
        return self._grid

    def clone(self) -> "Canvas":
        """Deep copy, sharing the world frame."""
        c = Canvas(self.w, self.h, self.scale, self.center, self.y_up)
        c.rgb = self.rgb.copy()
        c.a = self.a.copy()
        return c

    def over(self, rgb, a, x0: int = 0, y0: int = 0):
        """Source-over compositing of a straight-alpha tile at pixel offset."""
        rgb = np.asarray(rgb, dtype=np.float64)
        a = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)
        h, w = a.shape
        dst_rgb = self.rgb[y0 : y0 + h, x0 : x0 + w]
        dst_a = self.a[y0 : y0 + h, x0 : x0 + w]

        out_a = a + dst_a * (1.0 - a)
        num = rgb * a[..., None] + dst_rgb * dst_a[..., None] * (1.0 - a)[..., None]
        safe = out_a[..., None]
        self.rgb[y0 : y0 + h, x0 : x0 + w] = np.divide(
            num, safe, out=np.zeros_like(num), where=safe > 1e-8
        )
        self.a[y0 : y0 + h, x0 : x0 + w] = out_a

    def over_srgb(self, srgb, a, x0: int = 0, y0: int = 0):
        """Convenience wrapper: :meth:`over` with an sRGB colour."""
        self.over(palette.srgb_to_linear(srgb), a, x0, y0)

    def blit(self, rgb, a, x: int, y: int):
        """Composite a tile at pixel offset, clipping to the canvas bounds."""
        a = np.asarray(a, dtype=np.float64)
        h, w = a.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + w), min(self.h, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sx, sy = x0 - x, y0 - y
        self.over(
            np.asarray(rgb, dtype=np.float64)[sy : sy + (y1 - y0), sx : sx + (x1 - x0)],
            a[sy : sy + (y1 - y0), sx : sx + (x1 - x0)],
            x0,
            y0,
        )

    def fill(self, srgb, a: float = 1.0):
        """Paint the whole canvas with a flat sRGB colour."""
        self.rgb[:] = palette.srgb_to_linear(srgb)
        self.a[:] = a

    def flatten(self, background=None):
        """Composite onto an opaque background, returning ``(rgb, alpha)``.

        ``background=None`` keeps the existing transparency but returns linear
        RGB un-premultiplied, ready for :meth:`to_uint8`.
        """
        if background is None:
            return self.rgb, self.a
        bg = palette.srgb_to_linear(background)
        out = self.rgb * self.a[..., None] + bg * (1.0 - self.a)[..., None]
        return out, np.ones_like(self.a)

    def to_uint8(self, background=None) -> np.ndarray:
        """8-bit RGBA array for export; applies the sRGB transfer function.

        The last step is not a quantisation but a repair, and it is worth saying
        why.  Alpha is *coverage*: a pixel whose exported alpha is zero is a pixel
        the file declares is not there.  Whatever colour such a pixel happens to
        hold is not the colour of anything in the picture -- it is the tail of a
        blur, an artefact of the order the passes ran in rather than of what the
        asset depicts.  A drop shadow written over an empty pixel leaves the
        shadow's colour at an alpha of a millionth, which rounds to zero and stays
        there; a hollow frame does it in its own hole, all the way across it.

        Leaving it is not free, because the file is not what the eye sees.  Any
        consumer that filters straight alpha -- a bilinear scale of a nine-slice
        over content, a naive box downsample, a mipmap a tool builds without
        premultiplying -- reaches into that invisible pixel and pulls its colour
        into a visible one.  It is the same halo :func:`gen.heraldry.cloth._shed`
        premultiplies to avoid.

        So the rule is total and exact: where the file says the alpha is zero, the
        colour is zero.  Nothing the eye can see changes -- every pixel the file
        calls visible is the plain quantisation of what it was -- and the colour of
        nothing stops travelling with the picture.  Where it *does* travel today is
        worth noting: a contact sheet composites its tiles onto an opaque ground, so
        the only asset in a build that still has transparent pixels is the sprite
        atlas -- which is also the asset a game samples with a bilinear kernel.
        """
        rgb, a = self.flatten(background)
        srgb = palette.linear_to_srgb(rgb)
        out = np.empty((self.h, self.w, 4), dtype=np.uint8)
        out[..., :3] = np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
        out[out[..., 3] == 0, :3] = 0
        return out

    def save(self, path, background=None):
        """Write a PNG; imports the exporter lazily to avoid a cyclic import."""
        from export import png

        return png.save(self, path, background=background)

    def __repr__(self) -> str:
        return f"Canvas({self.w}x{self.h}, scale={self.scale:g})"
