"""Demography: the parts of a portrait that are not a mood or a hairstyle.

A face is not one average person with a set of sliders.  Sex, age, the climate a
lineage adapted to and the culture it dresses in all change *which proportions
read as ordinary* before a single expression is chosen.  Keeping them in one
module does two things:

* the genome stays honest -- these are heritable parameters, not hidden branches
  in the painter;
* the geometric part is small and testable.  Climate and culture do not sculpt a
  caricature here.  Climate contributes adaptive detail (nose and brow, sun and
  wind in the skin) and culture contributes clothing, trim and ornament; the
  skull and the features keep the same construction for every character.

The labels are deliberately broad.  ``climate`` is the environment a lineage is
adapted to, not a nationality; ``culture`` is a dress tradition, not a claim that
everyone from a place has one face.  A game can rename either table without
touching the geometry.
"""
from __future__ import annotations

__all__ = [
    "SEXES", "AGES", "CLIMATES", "CULTURES", "DEFAULTS", "COLLARS",
    "age_t", "stage", "collar", "modify",
]

#: Sex presentation.  ``androgynous`` is a complete option rather than a random
#: blend, because a face that is deliberately between the two is a design and not
#: a failure to choose.
SEXES = ("female", "male", "androgynous")

#: Life stages, coarse enough to read at avatar size and fine enough to matter.
AGES = ("child", "youth", "adult", "middle", "elder")

#: Environments a lineage is adapted to.  These drive skin and feature detail,
#: not a costume.
CLIMATES = ("arctic", "temperate", "arid", "tropical", "highland")

#: Dress traditions.  These drive the collar, trim and ornament; the face stays
#: the genome's face.
CULTURES = ("north", "river", "steppe", "coast", "desert", "highland")

#: The visible neckline a dress tradition closes with.  This is clothing, not a
#: face: it is the one silhouette in the bust that can say "culture" at avatar
#: size without turning a person into a costume diagram.
COLLARS = {
    "north": "crew",
    "river": "scoop",
    "steppe": "v",
    "coast": "scoop",
    "desert": "v",
    "highland": "crew",
}

DEFAULTS = {
    "sex": "androgynous",
    "age": "adult",
    "climate": "temperate",
    "culture": "river",
}

#: A continuous coordinate for the five stages, used by skin and hair detail.
_AGE_T = {"child": 0.10, "youth": 0.30, "adult": 0.50, "middle": 0.72, "elder": 0.92}


def stage(genes: dict) -> str:
    """The age stage, with the default for an older genome or a hand-written one."""
    value = str(genes.get("age", DEFAULTS["age"]))
    return value if value in _AGE_T else DEFAULTS["age"]


def age_t(genes: dict) -> float:
    return _AGE_T[stage(genes)]


def collar(genes: dict) -> str:
    """The neckline cut this culture closes its clothes with."""
    return COLLARS.get(str(genes.get("culture", DEFAULTS["culture"])), "scoop")


def modify(p: dict, genes: dict) -> dict:
    """Apply the sex/age/climate shape rules to a style parameter dictionary.

    The scales are deliberately smaller than a caricature: a jaw is a few
    hundredths of a head, not a different head.  The painter's detail pass does
    the rest with skin, hair, brows and clothing, where age and climate can be
    specific without turning a face into a sign.
    """
    sex = str(genes.get("sex", DEFAULTS["sex"]))
    age = stage(genes)
    climate = str(genes.get("climate", DEFAULTS["climate"]))

    if sex == "male":
        p["jaw_w"] = float(p["jaw_w"]) + 0.014
        p["chin_w"] = float(p["chin_w"]) + 0.010
        p["brow_thick"] = float(p["brow_thick"]) + 0.005
        p["brow_arch"] = float(p["brow_arch"]) - 0.003
        p["neck_w"] = float(p["neck_w"]) + 0.012
        p["shoulder_w"] = float(p["shoulder_w"]) + 0.035
        p["eye_w"] = float(p["eye_w"]) - 0.003
        p["lash"] = float(p["lash"]) - 0.003
        p["line"] = float(p["line"]) + 0.04
    elif sex == "female":
        p["jaw_w"] = float(p["jaw_w"]) - 0.010
        p["chin_w"] = float(p["chin_w"]) - 0.007
        p["brow_thick"] = float(p["brow_thick"]) - 0.003
        p["neck_w"] = float(p["neck_w"]) - 0.007
        p["shoulder_w"] = float(p["shoulder_w"]) - 0.020
        p["eye_w"] = float(p["eye_w"]) + 0.004
        p["lash"] = float(p["lash"]) + 0.004
        p["blush"] = float(p["blush"]) + 0.030
    # ``androgynous`` is the baseline, not half of the binary.

    if age == "child":
        p["eye_w"] = float(p["eye_w"]) + 0.010
        p["eye_up"] = float(p["eye_up"]) + 0.006
        p["eye_down"] = float(p["eye_down"]) + 0.004
        p["eye_y"] = float(p["eye_y"]) + 0.012
        p["jaw_w"] = float(p["jaw_w"]) - 0.020
        p["chin_w"] = float(p["chin_w"]) - 0.010
        p["brow_thick"] = float(p["brow_thick"]) - 0.004
        p["nose_w"] = float(p["nose_w"]) - 0.006
        p["mouth_w"] = float(p["mouth_w"]) - 0.008
        p["neck_w"] = float(p["neck_w"]) - 0.015
        p["shoulder_w"] = float(p["shoulder_w"]) - 0.080
        p["blush"] = float(p["blush"]) + 0.100
        p["lash"] = float(p["lash"]) + 0.002
        p["line"] = float(p["line"]) - 0.05
    elif age == "youth":
        p["jaw_w"] = float(p["jaw_w"]) - 0.006
        p["chin_w"] = float(p["chin_w"]) - 0.003
        p["blush"] = float(p["blush"]) + 0.040
    elif age == "middle":
        p["jaw_w"] = float(p["jaw_w"]) + 0.004
        p["chin_w"] = float(p["chin_w"]) + 0.004
        p["eye_up"] = float(p["eye_up"]) - 0.003
        p["eye_down"] = float(p["eye_down"]) - 0.002
        p["brow_thick"] = float(p["brow_thick"]) + 0.002
        p["nose_w"] = float(p["nose_w"]) + 0.003
        p["mouth_w"] = float(p["mouth_w"]) + 0.002
        p["blush"] = float(p["blush"]) - 0.040
        p["lash"] = float(p["lash"]) - 0.002
    elif age == "elder":
        p["jaw_w"] = float(p["jaw_w"]) - 0.008
        p["chin_w"] = float(p["chin_w"]) - 0.004
        # Eyes read a little smaller with age even after the width is capped by
        # the cheek, so the child-to-elder relationship survives the harmony clamp.
        p["eye_w"] = float(p["eye_w"]) - 0.020
        p["eye_up"] = float(p["eye_up"]) - 0.007
        p["eye_down"] = float(p["eye_down"]) - 0.004
        p["brow_thick"] = float(p["brow_thick"]) + 0.003
        p["nose_w"] = float(p["nose_w"]) + 0.005
        p["nose_h"] = float(p["nose_h"]) + 0.004
        p["mouth_w"] = float(p["mouth_w"]) - 0.006
        p["mouth_curve"] = float(p["mouth_curve"]) - 0.002
        p["blush"] = float(p["blush"]) - 0.080
        p["lash"] = float(p["lash"]) - 0.004
        p["neck_w"] = float(p["neck_w"]) - 0.005
        p["line"] = float(p["line"]) + 0.05

    # Climate is an adaptation, not a costume: the nose, brow, cheek and mouth
    # carry a visible shift in this model, and the colour/detail passes add
    # weather to the skin.  The numbers are still measured in hundredths of a
    # head -- enough to read as a lineage, not enough to make a mask.
    if climate == "arctic":
        p["cheek_w"] = float(p["cheek_w"]) + 0.003
        p["nose_w"] = float(p["nose_w"]) - 0.004
        p["nose_h"] = float(p["nose_h"]) + 0.004
        p["brow_thick"] = float(p["brow_thick"]) - 0.001
        p["eye_up"] = float(p["eye_up"]) - 0.002
        p["neck_w"] = float(p["neck_w"]) + 0.004
        p["shoulder_w"] = float(p["shoulder_w"]) + 0.020
        p["blush"] = float(p["blush"]) + 0.030
    elif climate == "arid":
        p["cheek_w"] = float(p["cheek_w"]) + 0.005
        p["jaw_w"] = float(p["jaw_w"]) + 0.004
        p["nose_w"] = float(p["nose_w"]) + 0.008
        p["nose_h"] = float(p["nose_h"]) - 0.002
        p["brow_thick"] = float(p["brow_thick"]) + 0.004
        p["lid_shadow"] = float(p["lid_shadow"]) + 0.050
        p["mouth_w"] = float(p["mouth_w"]) + 0.004
        p["blush"] = float(p["blush"]) - 0.020
    elif climate == "tropical":
        p["cheek_w"] = float(p["cheek_w"]) + 0.004
        p["nose_w"] = float(p["nose_w"]) + 0.010
        p["nose_h"] = float(p["nose_h"]) - 0.002
        p["mouth_w"] = float(p["mouth_w"]) + 0.008
        p["lip_full"] = float(p["lip_full"]) + 0.100
        p["eye_up"] = float(p["eye_up"]) + 0.003
        p["blush"] = float(p["blush"]) + 0.010
    elif climate == "highland":
        p["cheek_w"] = float(p["cheek_w"]) + 0.007
        p["jaw_w"] = float(p["jaw_w"]) + 0.004
        p["nose_h"] = float(p["nose_h"]) + 0.005
        p["brow_thick"] = float(p["brow_thick"]) + 0.002
        p["head_d"] = float(p["head_d"]) + 0.040
        p["blush"] = float(p["blush"]) + 0.030
    return p
