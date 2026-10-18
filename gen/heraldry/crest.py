"""A faction is its name.

Every other generator in this repository is asked for an asset by *seed*. That is
the right interface for a texture or a map, and the wrong one for a faction,
because a game does not have a number for the Kingdom of Alder -- it has the words.
So the name is the seed:

    from gen.heraldry import crest
    alder = crest.faction("Kingdom of Alder")

and the answer is always the same shield, the same banner and the same blazon, on
every machine and in every process, without a registry to keep them in step. Save
nothing; the name is the save file.

Two things make that work, and both of them are easy to get subtly wrong.

**The hash.** ``seed_from`` is a cryptographic hash of the name, not Python's
built-in ``hash()``. Python salts string hashing per process, so ``hash("Alder")``
is a different number every time the game starts: an asset derived from it would
change on every run and the bug would look like a rendering glitch rather than a
seed. A hash of the bytes is the same 64-bit number forever, which is the whole
point. There is a test that runs the derivation in a *fresh interpreter with a
different hash seed* and compares, because that is the only way this failure shows
up.

**The key.** ``normalise`` case-folds and collapses whitespace, so "The Grand
Duchy of Alder", "the grand duchy of alder" and "The  Grand  Duchy of Alder" are
one faction. Case is not something a player can be relied on to type consistently,
and a coat of arms that depends on it is a coat of arms that flickers.

The banner is derived from the *coat*, not from the name a second time, and that
is deliberate: a faction has one crest, and a shield and a flag that were generated
independently would sooner or later disagree about it. So the flag flies what the
shield bears -- the same tinctures, and the same ordinary where the ordinary has a
vexillological equivalent (a cross flies as a cross, a bordure as a border, a bend
as a hoist triangle) -- and the test asserts that relationship rather than either
picture.

**The livery.** The name also decides the *pigments*. Heraldry has seven tinctures
and a game has more factions than that, so two of them will have chosen Azure, and
on a map they still have to be two different armies. So the same seed draws a
shade per tincture -- a few steps of hue, chroma and lightness, small enough that
Azure is still Azure and large enough that two Azures are not one -- and the coat,
the banner and the shade all come from the name together, which is why a game
should call ``crest.render`` rather than hand a coat to the renderer itself. The
shade is stored as integers, so a sidecar can be read: two factions either wear the
same shade or wear shades a measurable step apart.
"""
from __future__ import annotations

import hashlib
import random
import unicodedata

from . import blazon, cloth as _cloth, flags, tinctures
from . import pole as _pole

__all__ = ["NAMESPACE", "arms", "faction", "flag", "normalise", "render", "seed_from",
           "shade", "shade_table"]

#: Mixed into every name before it is hashed.  It is a *namespace*, not a secret:
#: its only job is to keep "Alder" here from hashing to the same number as "Alder"
#: somewhere else.  Changing it renames every faction in the game, which is worth
#: knowing before changing it.
NAMESPACE = "ozart/faction/v1"

#: Which ordinary a banner can fly, and as what.  A shield's ordinary is a band
#: across its field; a flag's is a layout, so the ones that are a shape rather than
#: a band share one -- a bend, a chevron and a pile are all read as "a wedge from
#: the hoist", and there is no flag layout that improves on that.  Anything not in
#: this table falls through to the tinctures, which is honest: a pall has no
#: vexillological equivalent and pretending otherwise would be worse.
_FLYING = {
    "cross": ("cross", ""),
    "saltire": ("saltire", ""),
    "bordure": ("bordered", ""),
    "canton": ("canton", ""),
    "chief": ("bicolour", "horizontal"),
    "fess": ("bicolour", "horizontal"),
    "pale": ("bicolour", "vertical"),
    "bend": ("hoist_triangle", ""),
    "bend_sinister": ("hoist_triangle", ""),
    "chevron": ("hoist_triangle", ""),
    "pile": ("hoist_triangle", ""),
}


def normalise(name: str) -> str:
    """The key a name is known by: compatibility-folded, tidied, case-folded.

    Raises on a name that has nothing in it, because an empty name is not a faction
    with a plain crest -- it is a bug in the caller, and every empty name would
    otherwise share one.
    """
    if not isinstance(name, str):
        raise TypeError("a faction's name is a string")
    text = unicodedata.normalize("NFKC", " ".join(name.split()))
    if not text:
        raise ValueError("a faction with no name has no crest to derive")
    return text.casefold()


def seed_from(name: str, salt: str = "") -> int:
    """A stable 64-bit seed from a name.

    ``salt`` separates independent streams for one faction -- the coat, the banner's
    layout -- without letting them collide, which a single seed would do the moment
    two draws happened to line up.
    """
    key = f"{NAMESPACE}\x1f{salt}\x1f{normalise(name)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def arms(name: str, shield: str | None = None, charges: bool = True,
         ordinary: bool = True, style: dict | None = None) -> dict:
    """The coat of arms of the faction called ``name``."""
    return blazon.random_arms(seed=seed_from(name, "arms"), shield=shield,
                              charges=charges, ordinary=ordinary, style=style)


def _opposite(field_tincture: str, name: str) -> str:
    """A tincture that is *not* what the field already is.

    Asked for only when everything the shield carries is the field's own colour,
    which happens on a plain coat with a charge in the same tincture.  A flag of one
    flat colour is not a flag, so the tradition is asked for the other class instead
    -- a metal against a colour -- which is the rule of tincture standing in for the
    variety a flag does not get from its divisions.
    """
    pool = [t for t in tinctures.CLASSICAL
            if tinctures.is_metal(t) != tinctures.is_metal(field_tincture)]
    if not pool:
        return field_tincture
    return str(pool[seed_from(name, "second") % len(pool)])


def _second_field(field_tincture: str, candidates, name: str) -> str:
    for tincture in candidates:
        if tincture and tincture != field_tincture:
            return str(tincture)
    return _opposite(field_tincture, name)


def _banner(name: str, coat: dict | None, ratio, shape, style) -> dict:
    """The banner's *pattern*: what the coat looks like flown.  See :func:`flag`."""
    coat = coat if coat is not None else arms(name)
    field = coat["field"]
    tinctures_of_field = list(field["tinctures"])
    field_tincture = tinctures_of_field[0]
    # A divided field gives the banner its fimbriation: the second colour is already
    # in the design, and on a flag it has somewhere to be that is not another band.
    fimbriation = tinctures_of_field[1] if len(tinctures_of_field) > 1 else None

    ordinary = coat.get("ordinary")
    charges = coat.get("charges") or []
    candidates = ([ordinary["tincture"]] if ordinary else [])
    candidates += [c["tincture"] for c in charges]
    candidates += tinctures_of_field[1:]
    second = _second_field(field_tincture, candidates, name)

    if ordinary and ordinary["name"] in _FLYING:
        layout, direction = _FLYING[ordinary["name"]]
        if layout == "cross":
            return flags.cross_flag(field_tincture, second, fimbriation,
                                    ratio=ratio, shape=shape, style=style)
        if layout == "saltire":
            return flags.saltire_flag(field_tincture, second, fimbriation,
                                      ratio=ratio, shape=shape, style=style)
        if layout == "bordered":
            return flags.bordered(field_tincture, second,
                                  ratio=ratio, shape=shape, style=style)
        if layout == "canton":
            return flags.canton_flag(field_tincture, second,
                                     ratio=ratio, shape=shape, style=style)
        if layout == "hoist_triangle":
            return flags.hoist_triangle(field_tincture, second,
                                        ratio=ratio, shape=shape, style=style)
        return flags.bicolour(field_tincture, second, direction=direction,
                              ratio=ratio, shape=shape, style=style)

    if charges:
        # A coat with no ordinary to fly gets its bands: the shield carries the
        # detail, so the banner carries the colours.  Three bands when the design
        # has a third colour to spare, which is what makes a tricolour read as one.
        thirds = [t for t in candidates + tinctures_of_field if t not in (field_tincture, second)]
        if thirds and seed_from(name, "bands") % 3 == 0:
            return flags.tricolour(field_tincture, second, str(thirds[0]),
                                   direction="vertical" if seed_from(name, "fly") % 2 else "horizontal",
                                   ratio=ratio, shape=shape, style=style)
        return flags.bicolour(field_tincture, second,
                              direction="vertical" if seed_from(name, "fly") % 2 else "horizontal",
                              ratio=ratio, shape=shape, style=style)

    return flags.plain(field_tincture, ratio=ratio, shape=shape, style=style)


def _wind(name: str) -> dict:
    """The wind a flown banner gets, from the name's own seed.

    A banner on a pole is cloth, so it has folds -- and they have to be the *same*
    folds every time the faction is drawn, or the banner would ripple differently
    in every frame and read as a different banner.  Everything here comes from the
    name, and the ranges are the ones :func:`gen.heraldry.flags.random_flag` draws
    from, so a flown banner and a random flag at the same seed are the same *kind*
    of cloth.
    """
    rng = random.Random(seed_from(name, "wind"))
    return _cloth.wave(
        seed=seed_from(name, "phase"),
        amplitude=rng.uniform(0.028, 0.045),
        wavelength=rng.uniform(0.22, 0.40),
        harmonic=rng.uniform(0.15, 0.35),
    )


def flag(name: str, coat: dict | None = None, ratio=(3, 2), shape: str = "rectangle",
         style: dict | None = None, pole=False) -> dict:
    """The banner of the faction: the coat's tinctures, flown.

    Passing ``coat`` in is how :func:`faction` keeps the two from drifting apart;
    omitting it derives the coat from the name first, so the banner is a function of
    the name either way.

    ``pole`` flies the banner on one -- see :mod:`gen.heraldry.pole` -- and a
    banner on a pole is *cloth*, so it also gets the name's wind and the hoist the
    pole anchors.  Both are the pole's decisions rather than this module's, so they
    are applied here rather than left to the renderer, and both come from the name:
    drawing the same faction twice gives the same banner.
    """
    banner = _banner(name, coat, ratio, shape, style)
    if pole:
        banner["pole"] = _pole.spec(**(dict(pole) if isinstance(pole, dict) else {}))
        banner["wave"] = _pole.anchor(banner.get("wave") or _wind(name))
    return banner


def shade(name: str) -> dict:
    """The faction's livery: how far each tincture is nudged, as integer steps.

    Two houses can both choose Azure -- heraldry has seven colours and a game has
    more factions than that -- and a player still has to be able to tell their
    armies apart on a map.  So the same name that picks the coat also picks the
    *pigment*: a few steps of hue, chroma and lightness per tincture, drawn from
    the name's own seed, small enough that Azure is still Azure and large enough
    that two Azures are not one Azure.
    """
    return tinctures.random_shade(seed_from(name, "shade"))


def shade_table(name: str) -> dict:
    """The faction's livery as tinctures, ready to hand to a renderer."""
    return tinctures.shade_table(shade(name))


def render(name: str, size=512, which: str = "arms", pole=False):
    """The faction's shield or banner, painted in the faction's own shade.

    This is the entry point a game wants, because it is the one that cannot drift:
    the coat, the banner and the pigments all come from the name, and there is no
    way to render a faction's arms in the *classical* palette by forgetting an
    argument -- forgetting it takes calling :func:`gen.heraldry.blazon.render`
    yourself.  ``pole`` flies a banner from one; it is a presentation choice rather
    than something the name decides, which is why :func:`faction` does not carry it.
    """
    table = shade_table(name)
    if which == "arms":
        return blazon.render(arms(name), size=size, shade=table)
    if which == "flag":
        return flags.render(flag(name, pole=pole), size=size, shade=table)
    raise ValueError(f"a faction is rendered as 'arms' or 'flag', not {which!r}")


def faction(name: str, shield: str | None = None, ratio=(3, 2),
            shape: str = "rectangle", charges: bool = True, ordinary: bool = True,
            style: dict | None = None) -> dict:
    """Everything one name decides: the coat, the banner, the livery and the words.

    The return value is the asset.  It is JSON-serialisable, it carries both
    languages, and it carries the warnings the arms drew, because a generated coat
    is something a person has to be able to review in a sidecar rather than trust.
    """
    coat = arms(name, shield=shield, charges=charges, ordinary=ordinary, style=style)
    banner = flag(name, coat=coat, ratio=ratio, shape=shape, style=style)
    return {
        "name": " ".join(str(name).split()),
        "key": normalise(name),
        "seed": seed_from(name),
        "shade": shade(name),
        "arms": coat,
        "flag": banner,
        "blazon": {"en": blazon.blazon_text(coat, "en"),
                   "ru": blazon.blazon_text(coat, "ru")},
        "description": {"en": flags.describe(banner, "en"),
                        "ru": flags.describe(banner, "ru")},
        "warnings": blazon.validate(coat),
    }
