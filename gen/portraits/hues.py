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

from . import cel, demography, ink

__all__ = ["Hues", "from_genes", "SCLERA_TINT"]

#: How much skin bleeds into the eye whites; 0 gives a doll's cut-out eyes.
SCLERA_TINT = 0.12


class Hues:
    """Every colour one portrait paints with, as sRGB triples."""

    __slots__ = (
        "skin", "skin_shadow", "skin_deep", "skin_warm", "skin_light",
        "skin_freckle", "skin_cool",
        "hair", "hair_shadow", "hair_deep", "hair_hi", "hair_root", "hair_fine",
        "line", "line_hair", "line_soft",
        "iris", "iris_light", "iris_dark", "iris_rim", "pupil",
        "iris_fiber", "iris_glow", "pupil_rim",
        "sclera", "sclera_shadow", "lid_shadow", "lash", "tear",
        "brow", "brow_hi", "lip", "lip_light", "lip_deep", "lip_gloss",
        "lip_shadow", "mouth_dark", "teeth",
        "blush", "cloth", "cloth_shadow", "cloth_hi", "cloth_line",
        "cloth_trim", "metal", "bg", "bg_hi", "bg_lo",
    )

    def __init__(self, **kw):
        for key in self.__slots__:
            setattr(self, key, np.asarray(kw[key], dtype=np.float64))

    def as_dict(self) -> dict:
        return {k: palette.srgb_to_hex(getattr(self, k)) for k in self.__slots__}


#: Dress accents are a property of a culture, not of a face.  They are broad,
#: readable colours rather than flags: a game can rename the table without having
#: to repaint the portraits.
_CULTURE_ACCENT = {
    "north": (0.56, 0.105, 0.615),
    "river": (0.57, 0.095, 0.515),
    "steppe": (0.68, 0.115, 0.115),
    "coast": (0.62, 0.125, 0.030),
    "desert": (0.74, 0.085, 0.145),
    "highland": (0.49, 0.090, 0.315),
}

#: Climate is an adaptation, so it has to survive a pinned skin or hair gene.
#: The shifts are small in OKLCH but they are the difference between a lineage
#: and a label: a cool, low-chroma north; a warm, sun-weathered dryland; a deep,
#: rich tropics; a windburnt highland.
_CLIMATE_SKIN = {
    "arctic": (0.010, 0.94, -0.006),
    "temperate": (0.0, 1.0, 0.0),
    "arid": (-0.006, 1.08, 0.006),
    "tropical": (-0.004, 1.10, 0.003),
    "highland": (0.002, 1.04, 0.005),
}
_CLIMATE_HAIR = {
    "arctic": (0.008, 0.96, -0.003),
    "temperate": (0.0, 1.0, 0.0),
    "arid": (-0.006, 1.04, 0.004),
    "tropical": (-0.004, 1.06, 0.002),
    "highland": (0.004, 1.03, 0.003),
}

#: How much grey a life stage has earned.  Hair does not *become* a different
#: colour gene; the colour script mixes the gene towards grey so the DNA remains
#: the same person at every age.
_AGE_GREY = {"child": 0.0, "youth": 0.0, "adult": 0.0, "middle": 0.16, "elder": 0.62}


def _g(genes: dict, key: str) -> float:
    return float(genes.get(key, 0.0))


def _shift(rgb, *, light: float = 0.0, chroma: float = 1.0,
           hue: float = 0.0):
    """Move a colour in OKLCH, with predictable clamps for pigment ranges."""
    lch = palette.srgb_to_oklch(rgb)
    L = float(np.clip(lch[0] + light, 0.16, 0.97))
    C = float(np.clip(lch[1] * chroma, 0.002, 0.16))
    H = float((lch[2] + hue) % 1.0)
    return palette.oklch_to_srgb((L, C, H))


def _pigment_skin(rgb, genes: dict):
    """Express the skin colour gene through its melanin, warmth and undertone."""
    melanin = _g(genes, "skin_melanin")
    warmth = _g(genes, "skin_warmth")
    rosy = _g(genes, "skin_rosiness")
    olive = _g(genes, "skin_olive")
    skin = _shift(
        rgb,
        light=-0.060 * melanin + 0.012 * rosy - 0.010 * max(olive, 0.0),
        chroma=1.0 + 0.16 * abs(melanin) + 0.12 * abs(warmth) + 0.10 * abs(olive),
        hue=0.010 * warmth - 0.009 * olive + 0.006 * rosy,
    )
    if rosy > 0.0:
        skin = palette.mix(skin, palette.oklch_to_srgb((0.54, 0.135, 0.985)), 0.10 * rosy)
    climate = str(genes.get("climate", demography.DEFAULTS["climate"]))
    light, chroma, hue = _CLIMATE_SKIN.get(climate, (0.0, 1.0, 0.0))
    return _shift(skin, light=light, chroma=chroma, hue=hue)


def _pigment_hair(rgb, genes: dict):
    melanin = _g(genes, "hair_melanin")
    warmth = _g(genes, "hair_warmth")
    red = _g(genes, "hair_redness")
    hair = _shift(
        rgb,
        light=-0.085 * melanin,
        chroma=1.0 + 0.14 * abs(melanin) + 0.10 * abs(warmth),
        hue=0.012 * warmth,
    )
    if red > 0.0:
        hair = palette.mix(hair, palette.oklch_to_srgb((0.43, 0.095, 0.042)), 0.20 * red)
    elif red < 0.0:
        hair = palette.mix(hair, palette.oklch_to_srgb((0.48, 0.018, 0.62)), -0.13 * red)
    climate = str(genes.get("climate", demography.DEFAULTS["climate"]))
    light, chroma, hue = _CLIMATE_HAIR.get(climate, (0.0, 1.0, 0.0))
    return _shift(hair, light=light, chroma=chroma, hue=hue)


def _pigment_iris(rgb, genes: dict):
    hue = _g(genes, "iris_hue")
    chroma = _g(genes, "iris_chroma")
    value = _g(genes, "iris_value")
    return _shift(rgb, light=0.055 * value, chroma=1.0 + 0.35 * chroma,
                  hue=0.018 * hue)


def from_genes(genes: dict) -> Hues:
    """Derive the whole colour script from the base colours and pigment genes."""
    skin = _pigment_skin(palette.hex_to_srgb(genes["skin"]), genes)
    raw_hair = _pigment_hair(palette.hex_to_srgb(genes["hair"]), genes)
    iris = _pigment_iris(palette.hex_to_srgb(genes["iris"]), genes)
    age = demography.stage(genes)
    culture = str(genes.get("culture", demography.DEFAULTS["culture"]))
    accent = palette.oklch_to_srgb(_CULTURE_ACCENT.get(culture, _CULTURE_ACCENT["river"]))
    # The cloth gene is a *pinned* colour.  Mixing a culture accent into the
    # body of the garment meant that choosing black produced brown, and the
    # colour widget looked broken.  The accent now lives only on the trim,
    # where it belongs; the base cloth is exactly the value the genome carries.
    cloth = palette.hex_to_srgb(genes["cloth"])
    gloss = _g(genes, "hair_gloss")
    hair = palette.mix(raw_hair, (0.78, 0.79, 0.82), _AGE_GREY.get(age, 0.0))

    iris_dark = cel.shadow(iris, depth=0.18, chroma=1.10, hue=-0.02)
    iris_light = cel.highlight(iris, lift=0.16, chroma=0.75, hue=0.012)
    iris_rim = cel.shadow(iris, depth=0.30, chroma=1.05, hue=-0.01)
    pupil = palette.oklch_to_srgb(
        (0.16, min(float(palette.srgb_to_oklch(iris)[1]) * 0.8, 0.06), float(palette.srgb_to_oklch(iris)[2]))
    )

    sclera = palette.mix((1.0, 1.0, 1.0), skin, SCLERA_TINT)
    lid_shadow = cel.shadow(skin, depth=0.20, chroma=1.35, hue=-0.045)
    lash = ink.shade_line(skin, mul=0.16, floor=0.070, chroma=1.30, hue=-0.014)
    line = ink.shade_line(skin, mul=0.30, floor=0.150, chroma=1.25, hue=-0.022)

    lip_mix = float(np.clip(0.40 + 0.18 * _g(genes, "lip_pigment"), 0.14, 0.64))
    lip = palette.mix(skin, palette.oklch_to_srgb((0.55, 0.15, 0.985)), lip_mix)
    lip_light = cel.highlight(lip, lift=0.10, chroma=0.8)
    lip_gloss = cel.highlight(lip, lift=0.22, chroma=0.62)
    lip_shadow = cel.shadow(lip, depth=0.08, chroma=1.10, hue=-0.006)
    skin_light = cel.highlight(skin, lift=0.040, chroma=0.82)
    skin_freckle = palette.mix(skin, palette.oklch_to_srgb((0.40, 0.085, 0.095)), 0.24)
    skin_cool = cel.shadow(skin, depth=0.050, chroma=1.25, hue=-0.050)

    blush_lch = palette.srgb_to_oklch(skin).copy()
    blush_lch[1] = min(float(blush_lch[1]) * 1.7 + 0.045, 0.125)
    blush_lch[2] = (float(blush_lch[2]) - 0.030) % 1.0
    blush = palette.oklch_to_srgb((float(blush_lch[0]) - 0.040, blush_lch[1], blush_lch[2]))

    bg_lch = palette.srgb_to_oklch(hair).copy()
    bg = palette.oklch_to_srgb((0.955, min(float(bg_lch[1]) * 0.16, 0.022), float(bg_lch[2])))
    bg_hi = palette.oklch_to_srgb((0.978, min(float(bg_lch[1]) * 0.10, 0.014),
                                   float(bg_lch[2]) + 0.008))
    bg_lo = palette.oklch_to_srgb((0.895, min(float(bg_lch[1]) * 0.22, 0.030),
                                   float(bg_lch[2]) - 0.015))
    teeth = palette.mix((1.0, 1.0, 1.0), skin, 0.07)
    mouth_dark = palette.oklch_to_srgb((0.235, 0.070, (float(palette.srgb_to_oklch(lip)[2]) + 0.02) % 1.0))
    cloth_hi = cel.highlight(cloth, lift=0.12, chroma=0.82)
    cloth_trim = palette.mix(cloth, accent, 0.66)
    metal = palette.oklch_to_srgb((0.72, 0.105, 0.125 if culture in ("steppe", "desert") else 0.62))

    # Cel shading spends lightness to buy a terminator, and a deep base has very
    # little of it to spend: the same absolute drop that reads as shade on light
    # skin reads as soot on dark skin.  So the drop is scaled by how much headroom
    # the base actually has, and the chroma takes up the difference -- a shadow on
    # dark skin should be a *richer* brown, not a blacker one.
    head_room = float(np.clip(float(palette.srgb_to_oklch(skin)[0]) / 0.70, 0.50, 1.0))
    # A dark skin has less lightness to spend, but the old compensation bought
    # all of it back as chroma and turned a brown shadow into a red mask.  Keep
    # it a little richer than the base, not a different pigment.
    deep_chroma = 1.15 + 0.22 * (1.0 - head_room)

    return Hues(
        skin=skin,
        skin_shadow=cel.shadow(skin, depth=0.120 * head_room, chroma=deep_chroma,
                               hue=-0.022),
        skin_deep=cel.shadow(skin, depth=0.205 * head_room, chroma=deep_chroma + 0.06,
                             hue=-0.022),
        skin_warm=cel.highlight(skin, lift=0.035, chroma=1.05),
        skin_light=skin_light,
        skin_freckle=skin_freckle,
        skin_cool=skin_cool,
        hair=hair,
        hair_shadow=cel.shadow(hair, depth=0.145, chroma=1.18),
        hair_deep=cel.shadow(hair, depth=0.270, chroma=1.32),
        hair_hi=cel.highlight(hair, lift=float(np.clip(0.250 + 0.080 * gloss, 0.14, 0.34)),
                              chroma=0.62, hue=0.010),
        hair_root=cel.shadow(hair, depth=0.30, chroma=1.18),
        hair_fine=cel.highlight(hair, lift=float(np.clip(0.120 + 0.050 * gloss, 0.06, 0.19)),
                                chroma=0.78),
        line=line,
        line_hair=ink.shade_line(hair, mul=0.48, floor=0.070, chroma=1.15, hue=-0.018),
        line_soft=ink.shade_line(skin, mul=0.50, floor=0.300, chroma=1.10, hue=-0.018),
        iris=iris,
        iris_light=iris_light,
        iris_dark=iris_dark,
        iris_rim=iris_rim,
        pupil=pupil,
        iris_fiber=palette.mix(iris_light, (1.0, 1.0, 1.0), 0.22),
        iris_glow=cel.highlight(iris, lift=0.28, chroma=0.62),
        pupil_rim=cel.shadow(iris, depth=0.38, chroma=1.05),
        sclera=sclera,
        sclera_shadow=palette.mix(sclera, lid_shadow, 0.42),
        lid_shadow=lid_shadow,
        lash=lash,
        tear=palette.mix(lip, sclera, 0.34),
        brow=ink.shade_line(hair, mul=0.58, floor=0.115, chroma=1.10, hue=-0.010),
        brow_hi=cel.highlight(hair, lift=0.06, chroma=0.82),
        lip=lip,
        lip_light=lip_light,
        lip_deep=cel.shadow(lip, depth=0.13, chroma=1.15, hue=-0.008),
        lip_gloss=lip_gloss,
        lip_shadow=lip_shadow,
        mouth_dark=mouth_dark,
        teeth=teeth,
        blush=blush,
        cloth=cloth,
        cloth_shadow=cel.shadow(cloth, depth=0.095, chroma=1.14),
        cloth_hi=cloth_hi,
        cloth_line=ink.shade_line(cloth, mul=0.42, floor=0.070, chroma=1.10),
        cloth_trim=cloth_trim,
        metal=metal,
        bg=bg,
        bg_hi=bg_hi,
        bg_lo=bg_lo,
    )
