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
    "age_t", "stage", "collar", "modify", "proportion_floors",
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
        # The skeleton is a *relationship*, not a mood.  A headshot has to read
        # at a glance, so the mandible, brow shelf, nose, neck and shoulder girdle
        # move together.  This is still a drawing in the same tradition: the
        # values are hundredths of a head, not a photographic skull.
        p["head_w"] = float(p["head_w"]) + 0.012
        p["top_w"] = float(p["top_w"]) + 0.007
        p["wide_y"] = float(p.get("wide_y", 0.100)) + 0.004
        p["cheek_w"] = float(p["cheek_w"]) + 0.016
        p["cheek_y"] = float(p["cheek_y"]) - 0.007
        p["jaw_w"] = float(p["jaw_w"]) + 0.040
        p["chin_w"] = float(p["chin_w"]) + 0.024
        p["jaw_y"] = float(p["jaw_y"]) - 0.014
        p["chin_y"] = float(p["chin_y"]) - 0.012
        p["brow_thick"] = float(p["brow_thick"]) + 0.012
        p["brow_arch"] = float(p["brow_arch"]) - 0.007
        p["nose_w"] = float(p["nose_w"]) + 0.006
        p["nose_h"] = float(p["nose_h"]) + 0.008
        p["mouth_w"] = float(p["mouth_w"]) + 0.006
        p["neck_w"] = float(p["neck_w"]) + 0.038
        p["shoulder_w"] = float(p["shoulder_w"]) + 0.110
        p["shoulder_slope"] = float(p.get("shoulder_slope", 0.0)) - 0.26
        p["under_eye"] = float(p.get("under_eye", 0.26)) + 0.030
        p["eye_w"] = float(p["eye_w"]) - 0.010
        p["iris_r"] = float(p["iris_r"]) - 0.003
        p["lash"] = float(p["lash"]) - 0.006
        p["lip_full"] = float(p["lip_full"]) - 0.040
        p["line"] = float(p["line"]) + 0.07
    elif sex == "female":
        # The other reading of the same skeleton: a narrower mandible and chin, a
        # shorter lower face, higher cheek apples, a slimmer neck, more shoulder
        # slope, larger eyes and fuller lips.  The differences are deliberately
        # redundant -- shape, detail and colour all repeat the same cue -- so the
        # pair still reads as two people when the style is a chibi.
        p["head_w"] = float(p["head_w"]) - 0.008
        p["top_w"] = float(p["top_w"]) - 0.006
        p["cheek_w"] = float(p["cheek_w"]) + 0.010
        p["cheek_y"] = float(p["cheek_y"]) + 0.008
        p["jaw_w"] = float(p["jaw_w"]) - 0.030
        p["chin_w"] = float(p["chin_w"]) - 0.020
        p["jaw_y"] = float(p["jaw_y"]) + 0.008
        p["chin_y"] = float(p["chin_y"]) + 0.010
        p["brow_thick"] = float(p["brow_thick"]) - 0.008
        p["brow_arch"] = float(p["brow_arch"]) + 0.006
        p["nose_w"] = float(p["nose_w"]) - 0.005
        p["nose_h"] = float(p["nose_h"]) - 0.005
        p["mouth_w"] = float(p["mouth_w"]) - 0.008
        p["neck_w"] = float(p["neck_w"]) - 0.022
        p["shoulder_w"] = float(p["shoulder_w"]) - 0.070
        p["shoulder_slope"] = float(p.get("shoulder_slope", 0.0)) + 0.30
        p["under_eye"] = float(p.get("under_eye", 0.26)) - 0.020
        p["eye_w"] = float(p["eye_w"]) + 0.010
        p["iris_r"] = float(p["iris_r"]) + 0.003
        p["lash"] = float(p["lash"]) + 0.009
        p["lip_full"] = float(p["lip_full"]) + 0.085
        p["blush"] = float(p["blush"]) + 0.070
    # ``androgynous`` is the baseline, not half of the binary.

    if age == "child":
        p["eye_w"] = float(p["eye_w"]) + 0.013
        p["eye_up"] = float(p["eye_up"]) + 0.007
        p["eye_down"] = float(p["eye_down"]) + 0.004
        p["eye_y"] = float(p["eye_y"]) + 0.014
        p["iris_r"] = float(p["iris_r"]) + 0.003
        p["pupil_r"] = float(p["pupil_r"]) + 0.001
        p["jaw_w"] = float(p["jaw_w"]) - 0.022
        p["chin_w"] = float(p["chin_w"]) - 0.011
        p["brow_thick"] = float(p["brow_thick"]) - 0.004
        p["nose_w"] = float(p["nose_w"]) - 0.007
        p["nose_h"] = float(p["nose_h"]) - 0.002
        p["nose_gap"] = float(p["nose_gap"]) - 0.001
        p["mouth_w"] = float(p["mouth_w"]) - 0.009
        p["mouth_gap"] = float(p["mouth_gap"]) - 0.002
        p["lip_full"] = float(p["lip_full"]) + 0.020
        p["head_w"] = float(p["head_w"]) + 0.010
        p["cheek_w"] = float(p["cheek_w"]) + 0.012
        p["chin_y"] = float(p["chin_y"]) + 0.010
        p["ear_size"] = float(p.get("ear_size", 1.0)) - 0.050
        p["neck_w"] = float(p["neck_w"]) - 0.015
        p["shoulder_w"] = float(p["shoulder_w"]) - 0.100
        p["shoulder_slope"] = float(p.get("shoulder_slope", 0.0)) + 0.080
        p["under_eye"] = 0.08
        p["blush"] = float(p["blush"]) + 0.120
        p["lash"] = float(p["lash"]) + 0.002
        p["line"] = float(p["line"]) - 0.05
    elif age == "youth":
        # A young adult has the adult's lower face with a little more eye and a
        # little less jaw; the step from child to adult is not a single jump.
        p["eye_w"] = float(p["eye_w"]) - 0.002
        p["eye_up"] = float(p["eye_up"]) - 0.0005
        p["eye_down"] = float(p["eye_down"]) - 0.0005
        p["eye_y"] = float(p["eye_y"]) + 0.003
        p["iris_r"] = float(p["iris_r"]) - 0.001
        p["jaw_w"] = float(p["jaw_w"]) - 0.006
        p["chin_w"] = float(p["chin_w"]) - 0.003
        p["nose_w"] = float(p["nose_w"]) + 0.001
        p["mouth_w"] = float(p["mouth_w"]) + 0.002
        p["shoulder_w"] = float(p["shoulder_w"]) - 0.018
        p["shoulder_slope"] = float(p.get("shoulder_slope", 0.0)) + 0.020
        p["under_eye"] = 0.17
        p["blush"] = float(p["blush"]) + 0.040
    elif age == "adult":
        # The adult pass is the single biggest anti-doll lever.  The eyes are
        # smaller and sit a little higher, the lower face is longer, the nose and
        # mouth are a touch broader, and the jaw and neck have a grown-up mass.
        # These are still anime proportions -- this is an adult in the same
        # drawing tradition, not a photograph pasted into it.
        p["eye_y"] = float(p["eye_y"]) + 0.010
        # A slightly narrower eye with a little more space at the bridge is the
        # difference between an adult and an enlarged child's eye.  The nose and
        # mouth move up by the same hair's breadth, so the lower face gains a
        # plane rather than simply getting longer.
        p["eye_w"] = float(p["eye_w"]) - 0.020
        p["eye_up"] = float(p["eye_up"]) - 0.012
        p["eye_down"] = float(p["eye_down"]) - 0.008
        p["eye_dx"] = float(p["eye_dx"]) + 0.006
        p["iris_r"] = float(p["iris_r"]) - 0.010
        p["pupil_r"] = float(p["pupil_r"]) - 0.0020
        p["nose_w"] = float(p["nose_w"]) + 0.005
        p["nose_h"] = float(p["nose_h"]) + 0.004
        p["nose_gap"] = float(p["nose_gap"]) + 0.0005
        p["mouth_w"] = float(p["mouth_w"]) + 0.010
        p["mouth_gap"] = float(p["mouth_gap"]) + 0.002
        p["lip_full"] = float(p["lip_full"]) - 0.018
        p["jaw_w"] = float(p["jaw_w"]) + 0.006
        p["chin_w"] = float(p["chin_w"]) + 0.010
        p["neck_w"] = float(p["neck_w"]) + 0.012
        if sex == "male":
            p["jaw_w"] = float(p["jaw_w"]) + 0.004
            p["chin_w"] = float(p["chin_w"]) + 0.006
            p["neck_w"] = float(p["neck_w"]) + 0.004
            p["eye_w"] = float(p["eye_w"]) - 0.006
            p["eye_up"] = float(p["eye_up"]) - 0.003
            p["eye_down"] = float(p["eye_down"]) - 0.002
            p["iris_r"] = float(p["iris_r"]) - 0.004
            p["nose_w"] = float(p["nose_w"]) + 0.003
        p["brow_thick"] = float(p["brow_thick"]) + 0.001
        p["under_eye"] = 0.28
        p["blush"] = float(p["blush"]) - 0.030
    elif age == "middle":
        p["eye_y"] = float(p["eye_y"]) + 0.006
        p["jaw_w"] = float(p["jaw_w"]) + 0.004
        p["jaw_y"] = float(p["jaw_y"]) + 0.008
        p["chin_w"] = float(p["chin_w"]) + 0.004
        p["chin_y"] = float(p["chin_y"]) + 0.002
        p["eye_w"] = float(p["eye_w"]) - 0.008
        p["eye_up"] = float(p["eye_up"]) - 0.004
        p["eye_down"] = float(p["eye_down"]) - 0.0025
        p["iris_r"] = float(p["iris_r"]) - 0.004
        p["pupil_r"] = float(p["pupil_r"]) - 0.001
        p["brow_thick"] = float(p["brow_thick"]) + 0.002
        p["brow_gap"] = float(p["brow_gap"]) + 0.003
        p["nose_w"] = float(p["nose_w"]) + 0.004
        p["nose_h"] = float(p["nose_h"]) + 0.005
        p["nose_gap"] = float(p["nose_gap"]) + 0.001
        p["mouth_w"] = float(p["mouth_w"]) + 0.003
        p["mouth_gap"] = float(p["mouth_gap"]) + 0.002
        p["lip_full"] = float(p["lip_full"]) - 0.020
        p["ear_size"] = float(p.get("ear_size", 1.0)) + 0.030
        p["hairline_gap"] = float(p.get("hairline_gap", 0.300)) + 0.012
        p["neck_w"] = float(p["neck_w"]) + 0.006
        p["shoulder_w"] = float(p["shoulder_w"]) + 0.012
        p["shoulder_slope"] = float(p.get("shoulder_slope", 0.0)) + 0.060
        p["under_eye"] = 0.48
        p["blush"] = float(p["blush"]) - 0.040
        p["lash"] = float(p["lash"]) - 0.002
        p["lid_shadow"] = float(p["lid_shadow"]) + 0.050
    elif age == "elder":
        p["jaw_w"] = float(p["jaw_w"]) - 0.008
        p["jaw_y"] = float(p["jaw_y"]) + 0.014
        p["chin_w"] = float(p["chin_w"]) - 0.004
        p["chin_y"] = float(p["chin_y"]) + 0.008
        # Eyes read a little smaller with age even after the width is capped by
        # the cheek, so the child-to-elder relationship survives the harmony clamp.
        p["eye_w"] = float(p["eye_w"]) - 0.020
        p["eye_up"] = float(p["eye_up"]) - 0.007
        p["eye_down"] = float(p["eye_down"]) - 0.004
        p["eye_y"] = float(p["eye_y"]) + 0.010
        p["brow_thick"] = float(p["brow_thick"]) + 0.003
        p["brow_gap"] = float(p["brow_gap"]) + 0.005
        p["brow_arch"] = float(p["brow_arch"]) - 0.002
        p["nose_w"] = float(p["nose_w"]) + 0.006
        p["nose_h"] = float(p["nose_h"]) + 0.012
        p["mouth_w"] = float(p["mouth_w"]) - 0.008
        p["mouth_curve"] = float(p["mouth_curve"]) - 0.002
        p["lip_full"] = float(p["lip_full"]) - 0.030
        p["ear_size"] = float(p.get("ear_size", 1.0)) + 0.060
        p["hairline_gap"] = float(p.get("hairline_gap", 0.300)) + 0.028
        p["blush"] = float(p["blush"]) - 0.080
        p["lash"] = float(p["lash"]) - 0.004
        p["neck_w"] = float(p["neck_w"]) - 0.005
        p["shoulder_w"] = float(p["shoulder_w"]) - 0.010
        p["shoulder_slope"] = float(p.get("shoulder_slope", 0.0)) + 0.100
        p["under_eye"] = 0.70
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
        p["neck_w"] = float(p["neck_w"]) + 0.012
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

    # The neck and the shoulder girdle cannot be *only* additive sex offsets:
    # on a large-headed style those offsets shrink an adult female neck to a
    # stalk and leave the bust narrower than the skull.  These floors are
    # relative to the head the style actually has, so the same genome reads as
    # the same body on a chibi, a shoujo or a seinen head.  They are deliberately
    # below the male values, so the dimorphism survives the clamp.
    proportion_floors(p, genes)
    return p


def proportion_floors(p: dict, genes: dict) -> None:
    """Keep the bust attached to the skull after every other modifier has run.

    Phenotype genes can move the neck, and the style's head can be much larger
    than the base preset expects.  Floors are applied last, in :func:`dna.apply_to`,
    so a narrow-neck gene cannot turn an adult into a lollipop.
    """
    sex = str(genes.get("sex", DEFAULTS["sex"]))
    age = stage(genes)
    male = sex == "male"
    head_w = float(p["head_w"])
    neck_floor = 0.44 if not male else 0.54
    shoulder_floor = 1.34 if not male else 1.48
    if age == "child":
        neck_floor *= 0.84
        shoulder_floor *= 0.84
    elif age == "elder":
        shoulder_floor *= 0.96
    p["neck_w"] = max(float(p["neck_w"]), head_w * neck_floor)
    p["shoulder_w"] = max(float(p["shoulder_w"]), head_w * shoulder_floor)
