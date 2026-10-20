"""Phenotype: the categorical and continuous genes that sculpt a face.

The style presets in :mod:`gen.portraits.style` say what a *tradition* looks
like -- chibi, shoujo, shounen, seinen.  This module is the layer above that: the
heritable variation inside a tradition.  It exists because independent sliders
without names produce a pile of random proportions, while named shapes and a
small number of measured axes produce faces a viewer can describe.

There are three kinds of gene here:

* **shape categories** (``face_shape``, ``eye_shape``, ...) are named families.
  They are additive offsets on the preset, not absolute measurements, so the
  same heart-shaped face works on a chibi and a seinen;
* **morph genes** are continuous axes in ``[-1, 1]``: forehead width, cheekbone
  position, jaw angle, lip fullness, hair volume, and so on;
* **pigment genes** are continuous axes that move the four base colours.  They
  are deliberately separate from the colour genes: the colour gene is the
  material a character owns, while the pigment genes are the way that material
  is expressed.  This is what lets a child inherit a parent's base colour but
  have their own warmth, depth and freckling.

Every function is pure.  ``modify`` only touches a style dictionary; it never
paints.  The colour script applies the pigment genes in
:func:`gen.portraits.hues.from_genes`, and the detail pass reads the freckle
category.  The category and float tables are the single source of truth for
validation as well as generation, so adding a variant is one table edit.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "FACE_SHAPES", "JAW_SHAPES", "EYE_SHAPES", "BROW_SHAPES", "NOSE_SHAPES",
    "MOUTH_SHAPES", "HAIRLINE_SHAPES", "FRECKLE_PATTERNS",
    "CATEGORIES", "MORPH", "PIGMENT", "DEFAULTS", "CATEGORY_DEFAULTS",
    "random_values", "modify", "validate",
]

#: The named silhouette families.  These are offsets on the style, not separate
#: heads: a diamond face is a cheekbone emphasis on whatever skull the style owns.
FACE_SHAPES = ("oval", "round", "square", "heart", "long", "diamond")
JAW_SHAPES = ("soft", "angular", "square", "tapered", "round", "pointed")
EYE_SHAPES = ("almond", "round", "hooded", "upturned", "downturned", "wide", "narrow")
BROW_SHAPES = ("straight", "soft", "arched", "angled", "rounded", "flat")
NOSE_SHAPES = ("button", "straight", "roman", "wide", "snub", "delicate")
MOUTH_SHAPES = ("line", "full", "bow", "wide", "small", "downturned")
HAIRLINE_SHAPES = ("straight", "rounded", "high", "low", "swept", "widow")
FRECKLE_PATTERNS = ("none", "sparse", "bridge", "cheeks", "full")

#: Every categorical phenotype gene and the values it accepts.
CATEGORIES = {
    "face_shape": FACE_SHAPES,
    "jaw_shape": JAW_SHAPES,
    "eye_shape": EYE_SHAPES,
    "brow_shape": BROW_SHAPES,
    "nose_shape": NOSE_SHAPES,
    "mouth_shape": MOUTH_SHAPES,
    "hairline_shape": HAIRLINE_SHAPES,
    "freckles": FRECKLE_PATTERNS,
}

#: Continuous geometry axes.  They are appended after the legacy shape genes so
#: older DNA strings can be read with every new gene defaulting to neutral.
MORPH = (
    "head_depth", "forehead_w", "forehead_h", "temple_w", "cheek_bone",
    "cheek_drop", "cheek_full", "jaw_angle", "jaw_width", "chin_len",
    "chin_width", "eye_spacing", "eye_height", "eye_hood", "iris_size",
    "brow_height", "brow_length", "nose_length", "nose_width", "nose_bridge",
    "mouth_height", "lip_full", "ear_size", "neck_width", "neck_length",
    "hairline_height", "hair_volume", "fringe_length", "fringe_part",
    "side_lock", "shoulder_slope",
)

#: Continuous colour-expression axes.  Zero is the base colour itself, so an old
#: genome renders exactly as it did before this layer existed.
PIGMENT = (
    "skin_melanin", "skin_warmth", "skin_rosiness", "skin_olive",
    "hair_melanin", "hair_warmth", "hair_redness", "hair_gloss",
    "iris_hue", "iris_chroma", "iris_value", "lip_pigment",
)

CATEGORY_DEFAULTS = {
    "face_shape": "oval",
    "jaw_shape": "soft",
    "eye_shape": "almond",
    "brow_shape": "soft",
    "nose_shape": "straight",
    "mouth_shape": "line",
    "hairline_shape": "straight",
    "freckles": "sparse",
}

DEFAULTS = {
    **CATEGORY_DEFAULTS,
    **{key: 0.0 for key in MORPH},
    **{key: 0.0 for key in PIGMENT},
}

#: Categorical offsets, in head units.  The numbers are half-widths and heights
#: like every other style parameter.  They stay below a fifth of the feature they
#: move: enough to read as a family, not enough to turn the preset into another
#: style.
_CATEGORY_MODS = {
    "face_shape": {
        "oval": {},
        "round": {"head_w": 0.016, "top_w": 0.013, "cheek_w": 0.024, "cheek_y": 0.010,
                  "jaw_w": 0.014, "chin_w": 0.008, "head_d": 0.035},
        "square": {"top_w": 0.015, "cheek_w": 0.009, "jaw_w": 0.027, "jaw_y": -0.010,
                   "chin_w": 0.020, "head_d": 0.055},
        "heart": {"top_w": 0.026, "top_y": 0.005, "cheek_w": 0.017, "jaw_w": -0.028,
                  "chin_w": -0.011, "chin_y": -0.004},
        "long": {"head_w": -0.018, "cheek_w": -0.017, "top_y": 0.006,
                 "jaw_w": -0.010, "chin_y": -0.013, "head_d": 0.075},
        "diamond": {"top_w": -0.022, "cheek_w": 0.030, "jaw_w": -0.019,
                    "chin_w": -0.013, "cheek_y": 0.005},
    },
    "jaw_shape": {
        "soft": {"jaw_y": 0.006, "jaw_w": -0.004},
        "angular": {"jaw_y": -0.010, "jaw_w": 0.012, "chin_w": 0.004},
        "square": {"jaw_w": 0.025, "jaw_y": -0.005, "chin_w": 0.019},
        "tapered": {"jaw_w": -0.019, "chin_w": -0.011, "jaw_y": 0.004},
        "round": {"jaw_w": 0.004, "chin_w": 0.017, "jaw_y": 0.008},
        "pointed": {"jaw_w": -0.011, "chin_w": -0.023, "chin_y": -0.005},
    },
    "eye_shape": {
        "almond": {},
        "round": {"eye_up": 0.015, "eye_down": 0.017, "eye_tilt": -1.0,
                  "iris_r": 0.004, "lash": -0.001},
        "hooded": {"eye_up": -0.011, "eye_down": -0.002, "lid_shadow": 0.070,
                   "crease_w": 0.002, "iris_r": -0.002},
        "upturned": {"eye_tilt": 5.0, "eye_up": 0.004, "eye_down": -0.003},
        "downturned": {"eye_tilt": -5.0, "eye_up": -0.003, "eye_down": 0.003,
                       "brow_tilt": -2.0},
        "wide": {"eye_w": 0.013, "eye_dx": 0.004, "iris_r": 0.004},
        "narrow": {"eye_w": -0.013, "eye_up": -0.005, "eye_down": -0.004,
                   "iris_r": -0.003},
    },
    "brow_shape": {
        "straight": {"brow_arch": -0.007, "brow_tilt": -1.0},
        "soft": {"brow_arch": 0.004},
        "arched": {"brow_arch": 0.017, "brow_gap": 0.003, "brow_thick": -0.001},
        "angled": {"brow_arch": 0.004, "brow_tilt": 6.0, "brow_thick": 0.003},
        "rounded": {"brow_arch": 0.009, "brow_thick": -0.003, "brow_tilt": 1.0},
        "flat": {"brow_arch": -0.009, "brow_tilt": -2.0, "brow_thick": 0.002},
    },
    "nose_shape": {
        "button": {"nose_style": "dot", "nose_h": -0.006, "nose_w": 0.002,
                   "nose_gap": -0.002},
        "straight": {"nose_style": "wedge", "nose_h": 0.001},
        "roman": {"nose_style": "wedge", "nose_h": 0.017, "nose_w": -0.003,
                  "nose_gap": 0.002},
        "wide": {"nose_style": "wedge", "nose_w": 0.015, "nose_h": 0.001},
        "snub": {"nose_style": "dot", "nose_h": -0.009, "nose_w": 0.005,
                 "nose_gap": -0.003},
        "delicate": {"nose_style": "line", "nose_w": -0.009, "nose_h": -0.004},
    },
    "mouth_shape": {
        "line": {},
        "full": {"lip_full": 0.28, "mouth_w": 0.004},
        "bow": {"lip_full": 0.14, "mouth_curve": 0.004, "mouth_w": -0.006},
        "wide": {"mouth_w": 0.023, "mouth_curve": 0.001},
        "small": {"mouth_w": -0.020, "lip_full": 0.040},
        "downturned": {"mouth_curve": -0.011, "mouth_w": -0.004},
    },
    "freckles": {"none": {}, "sparse": {}, "bridge": {}, "cheeks": {}, "full": {}},
    "hairline_shape": {
        "straight": {},
        "rounded": {"hairline_gap": -0.008},
        "high": {"hairline_gap": 0.013},
        "low": {"hairline_gap": -0.012},
        "swept": {"hairline_gap": 0.002},
        "widow": {"hairline_gap": -0.004},
    },
}

#: Continuous morph maps.  ``head_depth`` is a multiplier on a multiple, so its
#: scale is in the units of ``head_d`` rather than in head widths.
_MORPH_MODS = {
    "head_depth": [("head_d", 0.110)],
    "forehead_w": [("top_w", 0.023)],
    "forehead_h": [("top_y", 0.013)],
    "temple_w": [("head_w", 0.026), ("head_d", 0.025)],
    "cheek_bone": [("cheek_w", 0.024)],
    "cheek_drop": [("cheek_y", -0.019)],
    "cheek_full": [("jaw_w", 0.013), ("cheek_w", 0.009), ("chin_w", 0.006)],
    "jaw_angle": [("jaw_y", -0.015), ("jaw_w", -0.008)],
    "jaw_width": [("jaw_w", 0.025), ("chin_w", 0.011)],
    "chin_len": [("chin_y", -0.015)],
    "chin_width": [("chin_w", 0.019)],
    "eye_spacing": [("eye_dx", 0.015), ("brow_dx", 0.015)],
    "eye_height": [("eye_up", 0.015), ("eye_down", 0.013), ("iris_squash", 0.045)],
    "eye_hood": [("eye_up", -0.009), ("lid_shadow", 0.085)],
    "iris_size": [("iris_r", 0.008), ("pupil_r", 0.002)],
    "brow_height": [("brow_gap", 0.013)],
    "brow_length": [("brow_w", 0.017)],
    "nose_length": [("nose_gap", 0.009)],
    "nose_width": [("nose_w", 0.013)],
    "nose_bridge": [("nose_h", 0.017)],
    "mouth_height": [("mouth_gap", 0.013)],
    "lip_full": [("lip_full", 0.20)],
    "ear_size": [("ear_size", 0.17)],
    "neck_width": [("neck_w", 0.015)],
    "neck_length": [("neck_y", -0.017)],
    "hairline_height": [("hairline_gap", 0.017)],
    "shoulder_slope": [("shoulder_slope", 0.70)],
}

#: The part of a style that a named eye/brow/mouth shape may change, used by the
#: tests to prove that a category has a visible, bounded effect.
_STYLE_KEYS = frozenset(
    key for mods in _CATEGORY_MODS.values() for mod in mods.values() for key in mod
) | frozenset(key for mods in _MORPH_MODS.values() for key, _ in mods)


def _apply(p: dict, mods: dict) -> None:
    for key, value in mods.items():
        if isinstance(value, str):
            p[key] = value
            continue
        if key not in p:
            p[key] = 0.0
        p[key] = float(p[key]) + float(value)


def modify(p: dict, genes: dict) -> dict:
    """Apply categorical and continuous phenotype genes to a style dict.

    Hair-specific axes (``hair_volume``, ``fringe_length``, ``fringe_part``,
    ``side_lock``) are read by :mod:`gen.portraits.hair` directly; they are still
    part of :data:`MORPH` so they round-trip and inherit like every other gene.
    """
    for key, choices in CATEGORIES.items():
        value = str(genes.get(key, CATEGORY_DEFAULTS[key]))
        if value not in choices:
            value = CATEGORY_DEFAULTS[key]
        _apply(p, _CATEGORY_MODS[key].get(value, {}))
    for key, mods in _MORPH_MODS.items():
        amount = float(genes.get(key, 0.0))
        for param, scale in mods:
            p[param] = float(p.get(param, 0.0)) + amount * scale
    return p


def random_values(rng: np.random.Generator) -> dict:
    """Draw the phenotype layer for :func:`gen.portraits.dna.random_genes`.

    Categories are weighted towards the centre of their family (oval faces,
    almond eyes, straight noses, soft jaws) so a contact sheet reads as a
    population with variants rather than a costume party.  Morph genes are
    normal with a deliberately modest spread; pigment genes are a little wider
    because colour variation is where a population shows first.
    """
    g = {
        "face_shape": str(rng.choice(FACE_SHAPES, p=(0.30, 0.15, 0.13, 0.13, 0.16, 0.13))),
        "jaw_shape": str(rng.choice(JAW_SHAPES, p=(0.24, 0.18, 0.14, 0.18, 0.14, 0.12))),
        "eye_shape": str(rng.choice(EYE_SHAPES,
                                    p=(0.27, 0.16, 0.14, 0.14, 0.10, 0.10, 0.09))),
        "brow_shape": str(rng.choice(BROW_SHAPES,
                                     p=(0.20, 0.22, 0.18, 0.14, 0.14, 0.12))),
        "nose_shape": str(rng.choice(NOSE_SHAPES,
                                     p=(0.17, 0.28, 0.12, 0.14, 0.14, 0.15))),
        "mouth_shape": str(rng.choice(MOUTH_SHAPES,
                                      p=(0.28, 0.19, 0.16, 0.15, 0.12, 0.10))),
        "hairline_shape": str(rng.choice(HAIRLINE_SHAPES,
                                         p=(0.24, 0.22, 0.13, 0.13, 0.16, 0.12))),
        "freckles": str(rng.choice(FRECKLE_PATTERNS,
                                   p=(0.22, 0.30, 0.18, 0.18, 0.12))),
    }
    for key in MORPH:
        if key in ("fringe_part", "hair_volume", "fringe_length", "side_lock"):
            value = rng.normal(0.0, 0.34)
        elif key in ("head_depth", "neck_width", "neck_length", "shoulder_slope"):
            value = rng.normal(0.0, 0.30)
        else:
            value = rng.normal(0.0, 0.38)
        g[key] = float(np.clip(value, -1.0, 1.0))
    for key in PIGMENT:
        g[key] = float(np.clip(rng.normal(0.0, 0.40), -1.0, 1.0))
    return {key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in g.items()}


def validate(genes: dict) -> None:
    """Raise ``ValueError`` for a bad category or an out-of-range float."""
    for key, choices in CATEGORIES.items():
        value = str(genes.get(key, CATEGORY_DEFAULTS[key]))
        if value not in choices:
            raise ValueError(f"unknown {key} {value!r}")
    for key in MORPH + PIGMENT:
        value = float(genes.get(key, 0.0))
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"gene {key!r} must be in [-1, 1], got {value}")
