"""The portrait's full body: a deterministic figure derived from the same genes.

A passport portrait answers "who is this?" and stops at the collarbone.  A strategy
game also needs the person standing on a map, and the worst way to get one is to
generate a second, unrelated seed: the face and the figure would then be two
people who happen to share a save file.  This module is the bridge.

Every input comes from the genome.  The body seed is ``dna.face_seed`` -- the hash
of the *sitter*, not of the outfit or the mood -- so changing a shirt or a smile
cannot change the body.  The palette comes from ``hues.from_genes``, which pins
the cloth exactly: a black uniform is black on the head and black on the legs.
The dress code becomes a silhouette (long robe/dress/habit, a mantle or habit cape) and
the hairstyle becomes long hair below the jaw.  The figure package does the drawing;
there is no second rig here.

The result is a pure function of the genome and the view, like the portrait itself.
"""
from __future__ import annotations

from core import palette

from gen.figures import body, figure

from . import dna, garments, hues, regalia

__all__ = ["FACINGS", "body_seed", "scene", "render", "frames", "describe"]

#: The four views the figure package draws, re-exported so a caller of this module
#: does not have to know where they come from.
FACINGS = figure.FACINGS


def body_seed(genes: dict) -> int:
    """The stable body seed for a genome: the sitter, not the dress code."""
    return int(dna.face_seed(genes))


def _garment(genes: dict) -> tuple:
    """The visible cut and the dress code it belongs to."""
    attire = str(genes.get("attire", regalia.DEFAULTS["attire"]))
    garment = str(genes.get("garment", regalia.DEFAULTS["garment"]))
    return garments.resolve(attire, garment), attire


def _hair_long(genes: dict) -> bool:
    style = str(genes.get("hair_style", "short"))
    length = float(genes.get("hair_len", 0.0))
    return style in ("long", "hime", "twin", "pony") or length > 0.15


def _build(genes: dict) -> float:
    """A small, honest difference of build from sex and life stage."""
    sex = str(genes.get("sex", "androgynous"))
    age = str(genes.get("age", "adult"))
    build = 1.045 if sex == "male" else (0.965 if sex == "female" else 1.0)
    if age == "child":
        build *= 0.84
    elif age == "elder":
        build *= 0.975
    return round(float(build), 4)


def _colours(genes: dict) -> dict:
    """The figure's five materials, taken from the portrait's own colour script."""
    h = hues.from_genes(genes)
    # Boots and belts are leather, not cloth.  Mixing the cloth's own line with a
    # neutral dark brown keeps the figure's feet readable on both a black uniform
    # and a pale one; the cloth itself stays exact.
    leather = palette.mix(h.cloth_line, (0.16, 0.095, 0.065), 0.45)
    return {
        "skin": tuple(float(v) for v in h.skin),
        "hair": tuple(float(v) for v in h.hair),
        "cloth": tuple(float(v) for v in h.cloth),
        "leather": tuple(float(v) for v in leather),
        "line": tuple(float(v) for v in h.line),
    }


def scene(genes: dict, *, facing: str = "down", cycle: str = "idle",
          phase: float = 0.0) -> dict:
    """The complete, JSON-safe recipe for one full-body frame.

    ``colours`` is also returned as hex here, because the sidecar is read by an
    engine that wants a swatch, not a tuple of linearised floats.  The renderer
    uses the float version through :func:`render`; the two are the same values.
    """
    cut, attire = _garment(genes)
    colours = _colours(genes)
    return {
        "portrait_seed": int(dna.seed_of(genes)),
        "body_seed": body_seed(genes),
        "dna": dna.encode(genes),
        "attire": attire,
        "garment": cut,
        "build": _build(genes),
        "hair_long": _hair_long(genes),
        "facing": str(facing),
        "cycle": str(cycle),
        "phase": round(float(phase), 5),
        "colours": {key: palette.srgb_to_hex(value) for key, value in colours.items()},
    }


def _kwargs(genes: dict) -> dict:
    cut, _attire = _garment(genes)
    return {
        "body_seed": body_seed(genes),
        "colours": _colours(genes),
        "garment": cut,
        "build": _build(genes),
        "hair_long": _hair_long(genes),
    }


def render(genes: dict, size: int = 256, *, cycle: str = "idle",
           phase: float = 0.0, facing: str = "down", backdrop=None):
    """One full-body figure derived from a portrait genome.

    The default view is ``down`` -- towards the camera -- because a character
    standing on a board is usually introduced face-on; ``right``/``left``/``up``
    are available through :data:`FACINGS`.  ``cycle`` and ``phase`` are the same
    gait clock the figure package uses, and ``idle`` is the standing pose a
    portrait is paired with.
    """
    kw = _kwargs(genes)
    return figure.render(seed=kw.pop("body_seed"), cycle=cycle, phase=phase,
                         size=size, backdrop=backdrop, facing=facing, **kw)


def frames(genes: dict, size: int = 256, *, cycle: str = "idle", count: int = 8,
           facing: str = "down", backdrop=None):
    """A whole gait cycle for a portrait-derived figure."""
    kw = _kwargs(genes)
    return figure.frames(seed=kw.pop("body_seed"), cycle=cycle, count=count,
                         size=size, backdrop=backdrop, facing=facing, **kw)


def describe(genes: dict, **scene_kwargs) -> dict:
    """The figure's table: recipe, proportions, palette and gait."""
    s = scene(genes, **scene_kwargs)
    kw = _kwargs(genes)
    table = figure.describe(seed=kw.pop("body_seed"), **kw)
    table.update({key: s[key] for key in
                  ("portrait_seed", "body_seed", "dna", "attire", "garment",
                   "build", "hair_long", "facing", "cycle", "phase")})
    table["portrait_colours"] = s["colours"]
    return table
