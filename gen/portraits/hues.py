"""The colour script: one base colour per material, everything else derived.

A generator that picks twenty colours independently produces a clown.  A
generator that picks *four* -- skin, hair, iris, cloth -- and derives the other
fifteen by rule produces something that always reads as a drawing.  The rules
are the ones an illustrator applies without thinking:

* a shadow is darker, *more* chromatic and rolled towards violet
  (:func:`gen.portraits.cel.shadow`);
* a highlight is lighter and *less* chromatic (:func:`gen.portraits.cel.highlight`);
* a line is far darker than any shadow, and keeps a trace of the hue it outlines
  (:func:`gen.portraits.ink.line_colour`);
* eye whites and teeth are never white -- they take a little of the skin, or the
  face next to them looks cut out;
* the blush is the skin's hue pushed towards red, not a separate pink.

Two derived colours are worth naming because they are what most cheap avatar
generators miss: the *lid shadow* on the eyeball (a violet grey, not a dark grey)
and the *iris rim* (darker than the iris, which is what gives the eye a wet edge).
"""
from __future__ import annotations

import numpy as np

from core import palette

from . import cel, ink

__all__ = ["Hues", "from_genes", "SCLERA_TINT"]

#: How much skin bleeds into the eye whites; 0 gives a doll's cut-out eyes.
SCLERA_TINT = 0.12


class Hues:
    """Every colour one portrait paints with, as sRGB triples."""

    __slots__ = (
        "skin", "skin_shadow", "skin_deep", "skin_warm",
        "hair", "hair_shadow", "hair_deep", "hair_hi",
        "line", "line_hair", "line_soft",
        "iris", "iris_light", "iris_dark", "iris_rim", "pupil",
        "sclera", "lid_shadow", "lash",
        "brow", "lip", "lip_light", "lip_deep", "mouth_dark", "teeth",
        "blush", "cloth", "cloth_shadow", "cloth_line",
        "bg",
    )

    def __init__(self, **kw):
        for key in self.__slots__:
            setattr(self, key, np.asarray(kw[key], dtype=np.float64))

    def as_dict(self) -> dict:
        return {k: palette.srgb_to_hex(getattr(self, k)) for k in self.__slots__}


def from_genes(genes: dict) -> Hues:
    """Derive the whole colour script from the four base colours in the genes."""
    skin = palette.hex_to_srgb(genes["skin"])
    hair = palette.hex_to_srgb(genes["hair"])
    iris = palette.hex_to_srgb(genes["iris"])
    cloth = palette.hex_to_srgb(genes["cloth"])

    iris_dark = cel.shadow(iris, depth=0.18, chroma=1.10, hue=-0.02)
    iris_light = cel.highlight(iris, lift=0.16, chroma=0.75, hue=0.012)
    iris_rim = cel.shadow(iris, depth=0.30, chroma=1.05, hue=-0.01)
    pupil = palette.oklch_to_srgb(
        (0.16, min(float(palette.srgb_to_oklch(iris)[1]) * 0.8, 0.06), float(palette.srgb_to_oklch(iris)[2]))
    )

    sclera = palette.mix((1.0, 1.0, 1.0), skin, SCLERA_TINT)
    lid_shadow = cel.shadow(skin, depth=0.22, chroma=1.60, hue=-0.055)
    lash = ink.shade_line(skin, mul=0.16, floor=0.070, chroma=1.30, hue=-0.014)
    line = ink.shade_line(skin, mul=0.30, floor=0.150, chroma=1.25, hue=-0.022)

    lip = palette.mix(skin, palette.oklch_to_srgb((0.55, 0.15, 0.985)), 0.42)
    lip_light = cel.highlight(lip, lift=0.10, chroma=0.8)

    blush_lch = palette.srgb_to_oklch(skin).copy()
    blush_lch[1] = min(float(blush_lch[1]) * 1.7 + 0.045, 0.125)
    blush_lch[2] = (float(blush_lch[2]) - 0.030) % 1.0
    blush = palette.oklch_to_srgb((float(blush_lch[0]) - 0.040, blush_lch[1], blush_lch[2]))

    bg_lch = palette.srgb_to_oklch(hair).copy()
    bg = palette.oklch_to_srgb((0.955, min(float(bg_lch[1]) * 0.16, 0.022), float(bg_lch[2])))
    teeth = palette.mix((1.0, 1.0, 1.0), skin, 0.07)
    mouth_dark = palette.oklch_to_srgb((0.235, 0.070, (float(palette.srgb_to_oklch(lip)[2]) + 0.02) % 1.0))

    # Cel shading spends lightness to buy a terminator, and a deep base has very
    # little of it to spend: the same absolute drop that reads as shade on light
    # skin reads as soot on dark skin.  So the drop is scaled by how much headroom
    # the base actually has, and the chroma takes up the difference -- a shadow on
    # dark skin should be a *richer* brown, not a blacker one.
    head_room = float(np.clip(float(palette.srgb_to_oklch(skin)[0]) / 0.70, 0.50, 1.0))
    deep_chroma = 1.22 + 0.55 * (1.0 - head_room)

    return Hues(
        skin=skin,
        skin_shadow=cel.shadow(skin, depth=0.135 * head_room, chroma=deep_chroma),
        skin_deep=cel.shadow(skin, depth=0.235 * head_room, chroma=deep_chroma + 0.12),
        skin_warm=cel.highlight(skin, lift=0.035, chroma=1.05),
        hair=hair,
        hair_shadow=cel.shadow(hair, depth=0.085, chroma=1.12),
        hair_deep=cel.shadow(hair, depth=0.165, chroma=1.22),
        hair_hi=cel.highlight(hair, lift=0.200, chroma=0.62, hue=0.010),
        line=line,
        line_hair=ink.shade_line(hair, mul=0.48, floor=0.070, chroma=1.15, hue=-0.018),
        line_soft=ink.shade_line(skin, mul=0.50, floor=0.300, chroma=1.10, hue=-0.018),
        iris=iris,
        iris_light=iris_light,
        iris_dark=iris_dark,
        iris_rim=iris_rim,
        pupil=pupil,
        sclera=sclera,
        lid_shadow=lid_shadow,
        lash=lash,
        brow=ink.shade_line(hair, mul=0.58, floor=0.115, chroma=1.10, hue=-0.010),
        lip=lip,
        lip_light=lip_light,
        lip_deep=cel.shadow(lip, depth=0.13, chroma=1.15, hue=-0.008),
        mouth_dark=mouth_dark,
        teeth=teeth,
        blush=blush,
        cloth=cloth,
        cloth_shadow=cel.shadow(cloth, depth=0.095, chroma=1.14),
        cloth_line=ink.shade_line(cloth, mul=0.42, floor=0.070, chroma=1.10),
        bg=bg,
    )
