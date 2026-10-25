"""DNA: a portrait as a short heritable string.

For a strategy game the interesting object is not a pretty face, it is a face
that can be *inherited*.  So a portrait is not an image with a seed; it is a
dictionary of genes -- four base colours, a style, a handful of shape modifiers in
``[-1, 1]`` -- which renders to an image, round-trips through a one-line string,
and can be crossed with another portrait's genes to produce a child.  A dynasty
is then a family tree of strings, and a save file stores a face as 300 bytes
instead of a PNG.

Two conventions make it robust:

* **Every float gene is quantised to four decimals** as it is created, so
  ``decode(encode(g)) == g`` exactly, not approximately.
* **Colours cross in OKLab and categorical genes pick a parent**, which is what
  keeps a child's skin between its parents' rather than beside them.

Shape genes are *deltas on the style preset*, not absolute measurements: a chibi
and a seinen heir of the same parents will both have their father's jaw, each in
its own proportions.
"""
from __future__ import annotations

import numpy as np

from core import palette

from . import accessories, demography, phenotype, regalia
from . import style as style_mod

__all__ = [
    "GENES", "CATEGORICAL", "LEGACY_CATEGORICAL", "COLOUR_GENES", "SHAPE_GENES",
    "DEMOGRAPHIC", "MORPH", "PIGMENT", "PHENOTYPE_CATEGORIES", "REGALIA_CATEGORIES",
    "ACCESSORY_CATEGORIES",
    "random_genes", "quantise", "encode", "decode", "breed", "mutate",
    "apply_to", "describe", "seed_of", "face_seed", "EXPRESSIONS", "HAIR_STYLES",
]

EXPRESSIONS = ("neutral", "smile", "grin", "surprised", "sad", "angry", "smug", "wink")

HAIR_STYLES = ("short", "bob", "long", "hime", "twin", "pony", "spiky", "wild")

HAIR_WEIGHTS = (0.18, 0.22, 0.24, 0.06, 0.05, 0.14, 0.05, 0.06)

LEGACY_CATEGORICAL = ("style", "hair_style", "expression")
COLOUR_GENES = ("skin", "hair", "iris", "cloth")
SHAPE_GENES = (
    "turn", "head_w", "head_len", "jaw", "eye_w", "eye_h", "eye_gap", "eye_tilt",
    "eye_y", "brow_thick", "brow_arch", "nose", "mouth_w", "mouth_smile", "blush",
    "lash", "hair_len", "bang_n",
)

DEMOGRAPHIC = ("sex", "age", "climate", "culture")
PHENOTYPE_CATEGORIES = tuple(phenotype.CATEGORIES)
REGALIA_CATEGORIES = tuple(regalia.CATEGORIES)
ACCESSORY_CATEGORIES = tuple(accessories.CATEGORIES)
MORPH = phenotype.MORPH
PIGMENT = phenotype.PIGMENT
CATEGORICAL = (LEGACY_CATEGORICAL + PHENOTYPE_CATEGORIES + REGALIA_CATEGORIES
               + ACCESSORY_CATEGORIES)

GENES = (LEGACY_CATEGORICAL + COLOUR_GENES + SHAPE_GENES + DEMOGRAPHIC
         + PHENOTYPE_CATEGORIES + MORPH + PIGMENT + REGALIA_CATEGORIES
         + ACCESSORY_CATEGORIES)

_SHAPE_MODS = {
    "head_w": [("head_w", 0.038), ("cheek_w", 0.038), ("top_w", 0.030), ("neck_w", 0.012),
               ("head_d", 0.10)],
    "head_len": [("jaw_y", -0.028), ("chin_w", 0.026), ("top_y", 0.012)],
    "jaw": [("jaw_w", 0.046), ("jaw_y", 0.022), ("chin_w", -0.020)],
    "eye_w": [("eye_w", 0.028), ("iris_r", 0.013), ("hl_r", 0.005)],
    "eye_h": [("eye_up", 0.021), ("eye_down", 0.017), ("iris_squash", 0.11), ("pupil_r", 0.005)],
    "eye_gap": [("eye_dx", 0.017), ("brow_dx", 0.019)],
    "eye_tilt": [("eye_tilt", 4.0)],
    "eye_y": [("eye_y", 0.026)],
    "brow_thick": [("brow_thick", 0.011)],
    "brow_arch": [("brow_arch", 0.016), ("brow_gap", 0.006)],
    "nose": [("nose_w", 0.017), ("nose_h", 0.020), ("nose_gap", 0.007)],
    "mouth_w": [("mouth_w", 0.028)],
    "mouth_smile": [("mouth_curve", 0.013), ("mouth_w", 0.008)],
    "blush": [("blush", 0.24)],
    "lash": [("lash", 0.010), ("line", 0.16), ("ink_w", 0.004)],
}

_SKIN_SETS = (

    (0.170, 0.050, 0.905), (0.170, 0.050, 0.885), (0.170, 0.051, 0.868),
    (0.169, 0.053, 0.850), (0.168, 0.056, 0.832), (0.166, 0.059, 0.812),
    (0.164, 0.062, 0.790), (0.162, 0.066, 0.766), (0.159, 0.070, 0.740),
    (0.156, 0.074, 0.712), (0.152, 0.078, 0.682), (0.148, 0.081, 0.648),
    (0.143, 0.083, 0.612), (0.137, 0.082, 0.572), (0.129, 0.079, 0.528),
    (0.118, 0.073, 0.474), (0.105, 0.066, 0.425),
)


def quantise(v: float) -> float:
    """Round a float gene to four decimals -- the round-trip invariant."""
    return float(round(float(v), 4))


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

_CLIMATE_SKIN = {
    "arctic": (0, 9),
    "temperate": (0, 14),
    "arid": (5, 17),
    "tropical": (8, 17),
    "highland": (2, 15),
}


def _skin(rng, climate: str = "temperate") -> str:
    lo, hi = _CLIMATE_SKIN.get(climate, _CLIMATE_SKIN["temperate"])
    hue, chroma, light = _SKIN_SETS[int(rng.integers(lo, hi))]
    lch = (
        float(np.clip(light + rng.uniform(-0.025, 0.025), 0.28, 0.95)),
        float(np.clip(chroma * rng.uniform(0.90, 1.10), 0.02, 0.11)),
        (hue + float(rng.uniform(-0.012, 0.012))) % 1.0,
    )
    return palette.oklch_to_hex(lch)


def _hair(rng) -> str:
    """A believable hair colour.

    Three natural families: near-neutral black/silver/ash, warm brown/auburn/
    blonde, and a rare cool blue-black.  The vivid branch that used to occupy one
    roll in four is gone: at contact-sheet size a single olive or violet head
    dominated the whole sheet, and "fantasy" is better carried by a muted cool
    black than by a neon hue.
    """
    roll = float(rng.random())
    if roll < 0.34:
        light = float(rng.choice([rng.uniform(0.18, 0.36), rng.uniform(0.76, 0.93)]))
        return palette.oklch_to_hex((light, float(rng.uniform(0.004, 0.030)), float(rng.random())))
    if roll < 0.94:
        hue = float(rng.uniform(0.035, 0.135))
        light = float(rng.uniform(0.32, 0.86))
        chroma = float(rng.uniform(0.030, 0.090)) * (1.15 if light > 0.7 else 1.0)
        return palette.oklch_to_hex((light, min(chroma, 0.095), hue))

    hue = float(rng.uniform(0.55, 0.88))
    light = float(rng.uniform(0.18, 0.40))
    return palette.oklch_to_hex((light, float(rng.uniform(0.018, 0.055)), hue))


def _iris(rng, hair_hex: str) -> str:
    """Iris: a natural eye palette.

    Brown, hazel, green, blue and grey are distinct *ranges*, not offsets on the
    hair hue.  The chroma stays low enough that a blue eye reads as grey-blue in
    a face rather than as a cyan jewel; the fibre pass in ``eyes`` supplies the
    saturation that a flat fill cannot.
    """
    hh = float(palette.srgb_to_oklch(palette.hex_to_srgb(hair_hex))[2])
    roll = float(rng.random())
    if roll < 0.46:
        return palette.oklch_to_hex((float(rng.uniform(0.36, 0.58)),
                                     float(rng.uniform(0.028, 0.070)),
                                     float(rng.uniform(0.060, 0.110))))
    if roll < 0.64:
        return palette.oklch_to_hex((float(rng.uniform(0.44, 0.62)),
                                     float(rng.uniform(0.030, 0.072)),
                                     float(rng.uniform(0.100, 0.155))))
    if roll < 0.78:
        return palette.oklch_to_hex((float(rng.uniform(0.46, 0.65)),
                                     float(rng.uniform(0.028, 0.068)),
                                     float(rng.uniform(0.280, 0.360))))
    if roll < 0.93:
        return palette.oklch_to_hex((float(rng.uniform(0.48, 0.70)),
                                     float(rng.uniform(0.025, 0.065)),
                                     float(rng.uniform(0.575, 0.660))))
    return palette.oklch_to_hex((float(rng.uniform(0.55, 0.73)),
                                 float(rng.uniform(0.003, 0.018)),
                                 hh))


def _cloth(rng, hair_hex: str) -> str:
    """Cloth: quieter than the hair, related to it, so it never fights the face."""
    hh = float(palette.srgb_to_oklch(palette.hex_to_srgb(hair_hex))[2])
    offset = float(rng.choice([0.03, -0.03, 0.08, -0.08, 0.5, -0.46, 0.0]))
    return palette.oklch_to_hex(
        (float(rng.uniform(0.28, 0.70)), float(rng.uniform(0.018, 0.095)), (hh + offset) % 1.0)
    )


def random_genes(seed=0) -> dict:
    """A complete, valid portrait genome -- the only source of randomness there is."""
    rng = _rng(seed)
    climate = str(rng.choice(demography.CLIMATES))
    hair = _hair(rng)
    g = {
        "style": str(rng.choice(style_mod.names())),
        "hair_style": str(rng.choice(HAIR_STYLES, p=HAIR_WEIGHTS)),
        "expression": str(rng.choice(tuple(e for e in EXPRESSIONS if e != "wink"))
                          if rng.random() < 0.85 else "wink"),
        "skin": _skin(rng, climate),
        "hair": hair,
        "iris": _iris(rng, hair),
        "cloth": _cloth(rng, hair),
    }
    for key in SHAPE_GENES:
        if key == "hair_len":
            g[key] = quantise(rng.uniform(-1.0, 1.0))
        elif key == "bang_n":
            g[key] = quantise(rng.uniform(-1.0, 1.0))
        elif key == "turn":
            g[key] = quantise(rng.uniform(-0.55, 0.55))
        else:
            g[key] = quantise(rng.normal(0.0, 0.42))

    g["sex"] = str(rng.choice(demography.SEXES, p=(0.46, 0.46, 0.08)))
    g["age"] = str(rng.choice(demography.AGES, p=(0.10, 0.26, 0.40, 0.16, 0.08)))
    g["climate"] = climate
    g["culture"] = str(rng.choice(demography.CULTURES))

    g.update(phenotype.random_values(rng))

    if g["sex"] != "male" or g["age"] == "child":
        g["facial_hair"] = "none"

    g.update(regalia.random_values(rng))
    if g["age"] == "child":
        g["awards"] = "none"
        g["insignia"] = "none"
        if g["ornament"] == "aiguillette":
            g["ornament"] = "none"
    if g["attire"] == "civil":
        g["insignia"] = "none"
        if g["ornament"] == "aiguillette":
            g["ornament"] = "chain"
    if g["attire"] == "clerical":
        g["insignia"] = "none"
        g["awards"] = "none"

    g.update(accessories.random_values(rng))
    if g["age"] == "child" and g["eyewear"] == "monocle":
        g["eyewear"] = "none"
    return g


def _wire(genes: dict, keys) -> str:
    parts = []
    for key in keys:
        value = genes[key]
        parts.append(f"{key}={quantise(value):.4f}" if isinstance(value, float) else f"{key}={value}")
    return ";".join(parts)


def encode(genes: dict) -> str:
    """``"style=shoujo;skin=#e8bf9e;..."`` -- one line, genes in :data:`GENES` order."""
    return _wire(genes, GENES)


def decode(text: str) -> dict:
    """Parse :func:`encode`; unknown keys are errors, the optional tails default.

    The legacy genes and the four base colours are mandatory, because a string
    missing one of them is a damaged genome rather than an old one.  Every gene
    appended after them -- demographic, phenotype, outfit -- may be absent, and a
    save from before that layer existed still decodes to a complete portrait.
    """
    out = {}
    for item in str(text).strip().split(";"):
        if not item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in GENES:
            raise ValueError(f"unknown gene {key!r}")
        if key in CATEGORICAL or key in DEMOGRAPHIC:
            out[key] = value.strip()
        elif key in COLOUR_GENES:
            palette.hex_to_srgb(value.strip())
            out[key] = value.strip()
        else:
            out[key] = quantise(float(value))
    missing = [k for k in LEGACY_CATEGORICAL + COLOUR_GENES + SHAPE_GENES if k not in out]
    if missing:
        raise ValueError(f"missing genes: {', '.join(missing)}")

    for key, value in demography.DEFAULTS.items():
        out.setdefault(key, value)
    for key, value in phenotype.DEFAULTS.items():
        out.setdefault(key, value)
    for key, value in regalia.DEFAULTS.items():
        out.setdefault(key, value)
    for key, value in accessories.DEFAULTS.items():
        out.setdefault(key, value)
    if out["style"] not in style_mod.names():
        raise ValueError(f"unknown style {out['style']!r}")
    if out["hair_style"] not in HAIR_STYLES:
        raise ValueError(f"unknown hair style {out['hair_style']!r}")
    if out["expression"] not in EXPRESSIONS:
        raise ValueError(f"unknown expression {out['expression']!r}")
    if out["sex"] not in demography.SEXES:
        raise ValueError(f"unknown sex {out['sex']!r}")
    if out["age"] not in demography.AGES:
        raise ValueError(f"unknown age {out['age']!r}")
    if out["climate"] not in demography.CLIMATES:
        raise ValueError(f"unknown climate {out['climate']!r}")
    if out["culture"] not in demography.CULTURES:
        raise ValueError(f"unknown culture {out['culture']!r}")
    phenotype.validate(out)
    regalia.validate(out)
    accessories.validate(out)
    return out


def _cross_colour(a: str, b: str, rng: np.random.Generator) -> str:
    ca, cb = palette.hex_to_srgb(a), palette.hex_to_srgb(b)
    if rng.random() < 0.30:
        return a if rng.random() < 0.5 else b
    t = float(rng.uniform(0.35, 0.65))
    lab = (1.0 - t) * palette.srgb_to_oklab(ca) + t * palette.srgb_to_oklab(cb)
    return palette.srgb_to_hex(palette.oklab_to_srgb(lab))


def breed(a: dict, b: dict, seed=0, mutation: float = 0.10) -> dict:
    """A child genome: colours blend, shapes average and drift, categories are picked."""
    rng = _rng(seed)
    child = {}
    for key in GENES:
        if key in COLOUR_GENES:
            child[key] = _cross_colour(a[key], b[key], rng)
        elif key in CATEGORICAL:
            child[key] = a[key] if rng.random() < 0.5 else b[key]
        elif key == "age":

            child[key] = "child" if rng.random() < 0.75 else "youth"
        elif key in ("sex", "climate", "culture"):
            child[key] = a[key] if rng.random() < 0.5 else b[key]
        elif key == "turn":
            child[key] = quantise(float(np.clip(rng.normal(0.0, 0.30), -0.55, 0.55)))
        else:
            mid = 0.5 * (float(a[key]) + float(b[key]))
            child[key] = quantise(float(np.clip(mid + rng.normal(0.0, mutation), -1.0, 1.0)))
    return child


def mutate(genes: dict, seed=0, amount: float = 0.30, rate: float = 0.35) -> dict:
    """A copy with a few genes nudged: drift for a population, not for a child."""
    rng = _rng(seed)
    out = dict(genes)
    for key in GENES:
        if rng.random() >= rate:
            continue
        if key in COLOUR_GENES:
            base = palette.srgb_to_oklch(palette.hex_to_srgb(genes[key]))
            out[key] = palette.oklch_to_hex((
                float(np.clip(base[0] + rng.normal(0.0, 0.05 * amount * 3), 0.05, 0.97)),
                max(float(base[1] + rng.normal(0.0, 0.02 * amount * 3)), 0.0),
                (float(base[2]) + float(rng.normal(0.0, 0.05 * amount * 3))) % 1.0,
            ))
        elif key in CATEGORICAL:
            if key == "style":
                pool = style_mod.names()
            elif key == "hair_style":
                pool = HAIR_STYLES
            elif key == "expression":
                pool = EXPRESSIONS
            elif key in regalia.CATEGORIES:
                pool = regalia.CATEGORIES[key]
            elif key in accessories.CATEGORIES:
                pool = accessories.CATEGORIES[key]
            else:
                pool = phenotype.CATEGORIES[key]
            out[key] = str(rng.choice(pool))
        elif key == "age":
            out[key] = str(rng.choice(demography.AGES))
        elif key == "sex":
            out[key] = str(rng.choice(demography.SEXES))
        elif key == "climate":
            out[key] = str(rng.choice(demography.CLIMATES))
        elif key == "culture":
            out[key] = str(rng.choice(demography.CULTURES))
        elif key == "turn":
            out[key] = quantise(float(np.clip(genes[key] + rng.normal(0.0, 0.2 * amount), -0.55, 0.55)))
        else:
            out[key] = quantise(float(np.clip(float(genes[key]) + rng.normal(0.0, amount), -1.0, 1.0)))
    return out


def _clamp_params(p: dict) -> dict:
    """Restore the anatomy invariants the modifiers may have broken.

    Independent sliders do not produce plausible faces -- a wide head with huge
    eyes and a tiny jaw is not a person -- so this pass enforces the handful of
    relationships that must hold: the eyes fit inside the skull, the jaw tapers,
    the face's features stay in order.  The vertical layout is not enforced here
    because :meth:`gen.portraits.style.Style` derives it from gaps; this pass only
    keeps the *gaps* in a sensible band.
    """

    p["eye_w"] = min(max(float(p["eye_w"]), 0.10), 0.36)
    p["eye_w"] = max(0.10, min(float(p["eye_w"]),
                               float(p["head_w"]) * 0.50,
                               float(p["cheek_w"]) * 0.50,
                               float(p["eye_dx"]) * 1.28))

    reach = float(p["eye_dx"]) + float(p["eye_w"]) * 0.5 + 0.045
    p["head_w"] = max(float(p["head_w"]), reach)

    p["cheek_w"] = max(min(float(p["cheek_w"]), float(p["head_w"]) + 0.026),
                       reach - 0.016)
    p["top_w"] = min(float(p["top_w"]), float(p["cheek_w"]))

    p["jaw_w"] = min(max(float(p["jaw_w"]), 0.16), float(p["cheek_w"]) - 0.045)
    p["jaw_y"] = min(max(float(p["jaw_y"]), -0.335), -0.20)
    p["chin_w"] = min(max(float(p["chin_w"]), 0.05), float(p["jaw_w"]) - 0.03)

    p["chin_y"] = float(np.clip(p["chin_y"], -0.535, -0.465))
    p["top_y"] = float(np.clip(p["top_y"], 0.405, 0.470))
    p["wide_y"] = min(max(float(p["wide_y"]), -0.05), float(p["top_y"]) - 0.08)
    p["cheek_y"] = min(float(p["cheek_y"]), float(p["wide_y"]) - 0.05)
    p["head_d"] = float(np.clip(p["head_d"], 0.85, 1.32))

    p["eye_up"] = min(max(float(p["eye_up"]), 0.035), 0.18)
    p["eye_down"] = min(max(float(p["eye_down"]), 0.028), 0.15)

    opening = float(p["eye_up"]) + float(p["eye_down"])
    low, high = float(p["eye_w"]) * 0.58, float(p["eye_w"]) * 0.96
    if opening < low:
        k = low / max(opening, 1e-9)
        p["eye_up"] = float(p["eye_up"]) * k
        p["eye_down"] = float(p["eye_down"]) * k
    elif opening > high:
        k = high / opening
        p["eye_up"] = float(p["eye_up"]) * k
        p["eye_down"] = float(p["eye_down"]) * k
    p["iris_r"] = min(float(p["iris_r"]), float(p["eye_w"]) * 0.48)
    p["pupil_r"] = min(float(p["pupil_r"]), float(p["iris_r"]) * 0.62)
    p["hl_r"] = min(float(p["hl_r"]), float(p["iris_r"]) * 0.62)
    p["brow_dx"] = max(float(p["brow_dx"]), float(p["eye_dx"]))
    p["brow_w"] = min(float(p["brow_w"]), 2.0 * (float(p["head_w"]) - float(p["brow_dx"]) + 0.06))
    p["brow_w"] = min(max(float(p["brow_w"]), 0.09), 0.30)

    p["brow_gap"] = float(np.clip(p["brow_gap"], 0.018, 0.085))
    p["nose_gap"] = float(np.clip(p["nose_gap"], 0.008, 0.052))
    nose_y = float(p["eye_y"]) - float(p["eye_down"]) - float(p["nose_gap"])
    mouth_room = nose_y - (float(p["chin_y"]) + 0.065)
    p["mouth_gap"] = float(np.clip(p["mouth_gap"], 0.030, min(0.140, max(0.030, mouth_room))))
    p["blush_gap"] = float(np.clip(p["blush_gap"], 0.028, 0.115))
    p["mouth_w"] = float(np.clip(p["mouth_w"], 0.080, float(p["jaw_w"]) * 0.78))

    p["nose_w"] = float(np.clip(p["nose_w"], 0.030, 0.118))
    p["nose_w"] = float(np.clip(p["nose_w"], 0.32 * float(p["mouth_w"]),
                                0.58 * float(p["mouth_w"])))
    p["lip_full"] = float(np.clip(p.get("lip_full", 0.0), -0.12, 0.85))
    p["under_eye"] = float(np.clip(p.get("under_eye", 0.26), 0.0, 0.90))
    p["ear_size"] = float(np.clip(p.get("ear_size", 1.0), 0.65, 1.40))

    p["hairline_gap"] = float(np.clip(p.get("hairline_gap", 0.300), 0.245, 0.360))
    p["hairline_gap"] = max(
        p["hairline_gap"],
        float(p["eye_up"]) + float(p["brow_gap"]) + 0.032,
    )

    p["neck_w"] = min(max(float(p["neck_w"]), 0.125), float(p["jaw_w"]) * 0.78)
    p["neck_y"] = min(float(p["neck_y"]), float(p["chin_y"]) - 0.14)
    p["shoulder_w"] = max(float(p["shoulder_w"]), float(p["neck_w"]) + 0.24)
    p["shoulder_y"] = min(float(p["shoulder_y"]), float(p["neck_y"]) - 0.30)
    p["shoulder_slope"] = float(np.clip(p.get("shoulder_slope", 0.0), -1.0, 1.0))
    p["blush"] = float(np.clip(p["blush"], 0.0, 0.85))
    p["ink_w"] = float(np.clip(p["ink_w"], 0.008, 0.045))
    p["line"] = float(np.clip(p["line"], 0.5, 2.0))
    p["lid_shadow"] = float(np.clip(p["lid_shadow"], 0.10, 0.75))
    p["lid_shadow_alpha"] = float(np.clip(p["lid_shadow_alpha"], 0.10, 0.75))
    return p


def apply_to(st: style_mod.Style, genes: dict) -> style_mod.Style:
    """Return the style preset with this genome's shape deltas applied.

    The result is built as a new :class:`Style`, so the face's vertical layout is
    re-derived from the (possibly modified) eye position -- a gene that moves the
    eyes moves everything below them with them.
    """
    p = dict(st.p)
    for gene, mods in _SHAPE_MODS.items():
        amount = float(genes[gene])
        for param, scale in mods:
            p[param] = float(p[param]) + amount * scale
    demography.modify(p, genes)
    phenotype.modify(p, genes)
    demography.proportion_floors(p, genes)
    return style_mod.Style(st.name, _clamp_params(p))


def describe(genes: dict) -> str:
    """One human line summarising the genome, for contact-sheet captions."""
    skin = palette.srgb_to_hex(palette.hex_to_srgb(genes["skin"]))
    face = genes.get("face_shape", phenotype.DEFAULTS["face_shape"])
    eyes = genes.get("eye_shape", phenotype.DEFAULTS["eye_shape"])
    nose = genes.get("nose_shape", phenotype.DEFAULTS["nose_shape"])
    facial = genes.get("facial_hair", phenotype.CATEGORY_DEFAULTS["facial_hair"])
    facial = "" if facial == "none" else f" \u00b7 {facial.replace('_', ' ')}"
    attire = genes.get("attire", regalia.DEFAULTS["attire"])
    awards = genes.get("awards", regalia.DEFAULTS["awards"])
    garment = genes.get("garment", regalia.DEFAULTS["garment"])
    eyewear = genes.get("eyewear", accessories.DEFAULTS["eyewear"])
    scar = genes.get("scar", accessories.DEFAULTS["scar"])
    dress = ("" if attire == "civil" and garment == "auto"
             else f" \u00b7 {attire}")
    if garment != "auto":
        dress += f" / {garment}"
    if awards != "none":
        dress += f" ({awards})"
    worn = []
    if eyewear != "none":
        worn.append(eyewear.replace("_", "-"))
    if scar != "none":
        worn.append(f"{scar} scar")
    if worn:
        dress += " · " + ", ".join(worn)
    return (
        f"{genes['style']} \u00b7 {genes['hair_style']} hair {genes['hair']}"
        f" \u00b7 iris {genes['iris']} \u00b7 skin {skin}"
        f" \u00b7 {face} face / {eyes} eyes / {nose} nose{facial}"
        f" \u00b7 {genes.get('sex', demography.DEFAULTS['sex'])}"
        f" {genes.get('age', demography.DEFAULTS['age'])}"
        f" \u00b7 {genes.get('climate', demography.DEFAULTS['climate'])}"
        f" \u00b7 {genes.get('culture', demography.DEFAULTS['culture'])}{dress}"
        f" \u00b7 {genes['expression']}"
    )


def seed_of(genes: dict) -> int:
    """A 48-bit seed derived from the genome's own string.

    This is what makes a portrait a pure function of its genes: the hair's random
    variation is driven by a generator seeded here, so there is no hidden second
    input and ``encode`` alone is enough to reproduce an image.
    """
    import hashlib

    seed_genes = tuple(key for key in GENES
                       if key != "garment" and key not in ACCESSORY_CATEGORIES)
    digest = hashlib.sha256(_wire(genes, seed_genes).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)

_FACE_SEED_EXCLUDE = frozenset(REGALIA_CATEGORIES) | frozenset(ACCESSORY_CATEGORIES) | {
    "expression", "turn", "hair_style", "hair", "cloth", "hair_len", "bang_n",
    "hair_volume", "fringe_length", "fringe_part", "side_lock", "hair_gloss",
    "hair_melanin", "hair_warmth", "hair_redness", "facial_hair",
}


def face_seed(genes: dict) -> int:
    """A stable seed for the *sitter's* small asymmetries.

    The hair generator's seed may include the dress code because a saved genome
    has to reproduce its hair; the asymmetry must not, or changing a shirt would
    change the person.  This hash therefore deliberately leaves out the entire
    regalia layer, the pose, the mood, the hairstyle and the hair/cloth colours.
    """
    import hashlib

    face_genes = tuple(key for key in GENES if key not in _FACE_SEED_EXCLUDE)
    digest = hashlib.sha256(_wire(genes, face_genes).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)
