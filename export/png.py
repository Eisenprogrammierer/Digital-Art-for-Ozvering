"""PNG export: linear light -> sRGB -> 8-bit, via Pillow."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

__all__ = ["to_image", "save"]


def to_image(canvas, background=None) -> Image.Image:
    """Convert a :class:`core.canvas.Canvas` to a Pillow RGBA image."""
    return Image.fromarray(canvas.to_uint8(background=background), "RGBA")


def save(canvas, path, background=None):
    """Write ``canvas`` to ``path``, creating parent directories as needed.

    Returns the path written, so callers can log or hash it.  PNG output is
    lossless and deterministic: the same canvas always yields identical bytes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = to_image(canvas, background=background)
    img.save(path, "PNG", optimize=True)
    return path
