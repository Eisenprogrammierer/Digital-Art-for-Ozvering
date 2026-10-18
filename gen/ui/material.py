"""Material bodies: wood and stone, out of the noise the core already has.

The body of a panel is the one part of a UI asset a player looks at rather than
reads, and until now it has been a vertical gradient -- for one reason: anything
varying *across* a panel smears when a nine-slice stretches the middle sideways.
A material breaks that rule deliberately, and the price is *stated* rather than
hidden.  A panel with a material is drawn at the size it will be shown,
:func:`gen.ui.slice.stretch_error` measures what a stretch would actually cost,
and the sidecar carries that number beside the insets.

Two recipes, and they are the two things a material can be:

* **wood** is *rings*.  A sine across the grain whose phase is domain-warped,
  with a fine grain along it.  The warp is the whole trick: without it, rings
  drawn on a rectangle are stripes, and a stripe is not wood.
* **stone** is *cells*.  Worley distance gives the plates, ``worley_edges``
  gives the mortar between them -- it is zero exactly on a cell boundary -- and a
  slow fBm mottles the plates so that a large panel does not read as a
  checkerboard.

Both are evaluated in *design* pixels, so ``scale=2`` doubles the resolution and
changes no feature: the size of a grain is a number in the style, like every
other number in this package.  Both are lit by the project's one light, and the
light is **centred on the material's own mean** -- the same decision the cloth
module makes, for the same reason: a material should redistribute the light
across a surface rather than dim it, because the palette is what decides how
bright a panel is.
"""
from __future__ import annotations

import numpy as np

from core import noise, render

__all__ = ["KINDS", "DEFAULTS", "kind", "height", "body"]

#: What a body can be made of.  ``none`` is a real member and the default: a
#: plain body is the one that stretches cleanly, so which of the two a panel is
#: has to be asked for rather than implied.
KINDS = ("none", "wood", "stone")

#: The one name and four numbers a material is described by.
DEFAULTS = {
    "material": "none",
    "material_scale": 22.0,   # design pixels per ring, per cell
    "material_seed": 0,
    "material_relief": 0.55,  # how tall the material is, as a slope
    "material_tone": 0.30,    # how far the stain pulls towards the low colour
}

#: The same ambient and diffuse terms the texture package uses.  They are not the
#: moulding's numbers: a band is a rounded edge ten pixels wide, and a material is
#: a rough surface seen from farther away.
AMBIENT = 0.62
DIFFUSE = 0.55


def kind(spec: dict) -> str:
    """The material's name, checked.  Raises for a name that is not one.

    A bad name is an identifier error, not a number out of range, so this is the
    one place in the material that raises rather than clamps -- the same
    relationship :func:`gen.ui.style.validate` has with a spec it is shown.
    """
    k = str(spec.get("material", DEFAULTS["material"]))
    if k not in KINDS:
        raise ValueError(f"material must be one of {KINDS}, not {k!r}")
    return k


def height(X, Y, spec: dict):
    """The material's shape over a world grid, in ``[0, 1]``.

    Normalised on purpose: a shape is worth nothing until it is multiplied by a
    size, and the size is a number in the style.  :func:`body` does that multiply,
    which is what makes ``relief`` mean a slope rather than "some height".
    """
    k = kind(spec)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if k == "none":
        return np.zeros(np.broadcast_shapes(X.shape, Y.shape), dtype=np.float64)
    scale = max(float(spec.get("material_scale", DEFAULTS["material_scale"])), 1.0)
    seed = int(spec.get("material_seed", DEFAULTS["material_seed"]))
    u, v = X / scale, Y / scale
    return _wood(u, v, seed) if k == "wood" else _stone(u, v, seed)


def body(colour, spec: dict, X, Y, dx: float, light, stain_colour=None,
         y_up: bool = True):
    """A body colour field, shaded by a material and stained by it.

    ``colour`` is the sRGB field the gradient already produced; the return value
    is the same field with the material on it, so the caller's compositing does
    not change.  Two things happen and they are deliberately different in kind:

    * **relief** is the height field lit by the project's one light, so a material
      is a crumpled version of the same surface rather than a texture pasted onto
      it.  The height is multiplied by the feature size before it is lit, because
      the normals are ``dh/dx`` in *world* units: a material whose features are
      22 px across has to have its height in the same units to show up at all.
    * **tone** is a slow variation mixed towards the style's own low colour --
      patina and wear, not shadow -- which is what stops a big panel from reading
      as a sheet of identical planks.

    The light is centred on the material's *own* mean, so the factor is a ratio:
    a material redistributes the panel's brightness across its surface instead of
    removing some of it.
    """
    if kind(spec) == "none":
        return np.asarray(colour, dtype=np.float64)

    h = height(X, Y, spec)
    scale = max(float(spec.get("material_scale", DEFAULTS["material_scale"])), 1.0)
    relief = float(spec.get("material_relief", DEFAULTS["material_relief"]))
    n = render.height_normals(h * scale * relief, dx, y_up=y_up)

    L = np.array([light[0], light[1], 0.65], dtype=np.float64)
    L /= max(float(np.linalg.norm(L)), 1e-12)
    lam = np.clip(n @ L, 0.0, 1.0)
    flat = AMBIENT + DIFFUSE * float(lam.mean())
    factor = (AMBIENT + DIFFUSE * lam) / max(flat, 1e-9)

    out = np.asarray(colour, dtype=np.float64) * factor[..., None]
    tone = float(spec.get("material_tone", DEFAULTS["material_tone"]))
    if tone > 0.0 and stain_colour is not None:
        mix = (tone * _stain(X, Y, spec))[..., None]
        out = out * (1.0 - mix) + np.asarray(stain_colour, dtype=np.float64) * mix
    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------ the recipes
def _wood(u, v, seed: int):
    """Rings across the grain, phase-warped, with a fine grain along it.

    The rings are a sine in ``v`` -- one per feature, so the grain runs along the
    panel the way a board does -- and the warp bends every iso-line of it at once,
    which is the difference between timber and a striped wallpaper.
    """
    _, wy = noise.warp(u, v, seed=seed, amp=0.80, freq=0.70, octaves=3)
    ring = 0.5 + 0.5 * np.sin(2.0 * np.pi * wy)
    # A grain line is thin and dark: the power keeps the flats flat and narrows
    # the line between them, which is most of what makes it read as a ring.
    ring = ring ** 3.0
    fine = noise.fbm(u * 1.4, v * 6.0, seed=seed + 31, freq=1.0, octaves=3)
    return np.clip(ring * (0.66 + 0.50 * fine), 0.0, 1.0)


def _stone(u, v, seed: int):
    """Plates with mortar between them.

    ``worley`` is the distance to the nearest feature point, so it domes each
    plate; ``worley_edges`` is ``F2 - F1``, which is zero on a boundary and grows
    towards the middle of a cell, so inverting it carves the mortar.  The mortar
    is the only thing in the field with a hard edge, and a hard edge is what
    catches the light and reads as stone.
    """
    d = noise.worley(u, v, seed=seed, freq=1.0, jitter=0.85)
    wall = noise.worley_edges(u, v, seed=seed, freq=1.0, jitter=0.85)
    mottle = noise.fbm(u * 2.2, v * 2.2, seed=seed + 17, freq=1.0, octaves=4)
    plate = 0.55 + 0.45 * mottle
    dome = np.clip(1.0 - d / 0.75, 0.0, 1.0)
    groove = np.clip(1.0 - wall / 0.16, 0.0, 1.0)
    return np.clip(plate * (0.72 + 0.28 * dome) * (1.0 - 0.85 * groove), 0.0, 1.0)


def _stain(X, Y, spec: dict):
    """A slow, low-contrast variation: patina, wear, the damp corner of a cellar.

    Biased towards zero by the power, so most of a panel is clean and the wear
    collects where it pleases, and stretched 2:1 across the panel so the patches
    do not look like the material's own cells.
    """
    scale = max(float(spec.get("material_scale", DEFAULTS["material_scale"])), 1.0)
    seed = int(spec.get("material_seed", DEFAULTS["material_seed"]))
    u = np.asarray(X, dtype=np.float64) / (scale * 6.0)
    v = np.asarray(Y, dtype=np.float64) / (scale * 3.0)
    return noise.fbm(u, v, seed=seed + 53, freq=1.0, octaves=3) ** 1.5
