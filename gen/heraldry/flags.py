"""Flags: the same composition engine on a rectangular field.

A flag is not a coat of arms, and the difference is worth stating precisely
because it shapes this module:

* **The rule of tincture does not apply.**  Flags are textile, not armorial, and
  the world's flags break the metal/colour rule constantly -- blue beside red
  beside white.  So the flag patterns here pick tinctures freely and
  :func:`gen.heraldry.blazon.validate` is advisory, not law.
* **Flags are built from a much smaller repertoire.**  Nearly every flag is a
  field with bands (bicolour, tricolour), a cross (including the offset Nordic
  one), a saltire, a canton, or a triangle at the hoist.  Each of those is one
  line here because the ordinaries and divisions already exist.
* **A flag can be flown.**  ``spec["pole"]`` draws it on one, and the pole -- with
  the room it needs in the frame, the anchor it puts on the cloth's hoist and the
  droop that follows -- is :mod:`gen.heraldry.pole`.  It is composited *after* the
  wave and *over* the cloth, because a pole is the one rigid thing on the canvas.
* **Or hung.**  ``spec["bar"]`` is a crossbar, and it turns the same cloth into a
  gonfalon: the bar holds the cloth along the top of its folds rather than beside
  them, so it is :func:`gen.heraldry.cloth.drape` that hangs it and the bar that is
  drawn over the seam.  One spec key decides it, and both keys come from
  :mod:`gen.heraldry.pole`, because a bar is a shaft turned a quarter turn.

A flag pattern therefore returns an ordinary *spec* -- exactly the dict
:mod:`gen.heraldry.blazon` renders -- so flags and arms share one renderer, one
validator and one blazon writer, and a flag can carry a coat of arms on its
field without any new machinery.
"""
from __future__ import annotations

import numpy as np

from . import blazon as _blazon
from . import charges as _charges
from . import cloth as _cloth
from . import ordinaries as _ordinaries
from . import pole as _pole
from . import shields as _shields
from . import tinctures as _tinctures

__all__ = [
    "FLAG_STYLE",
    "PATTERNS",
    "plain",
    "bicolour",
    "tricolour",
    "cross_flag",
    "nordic_cross",
    "saltire_flag",
    "canton_flag",
    "hoist_triangle",
    "bordered",
    "random_flag",
    "describe",
    "render",
]

#: Flags sit in their frame and are lit like everything else, but they carry no
#: drop shadow by default: a flag is a plane, not a figurine.
FLAG_STYLE = {
    "margin": 0.035,
    "shadow": 0.0,
    "rim": 0.22,
    "rim_width": 0.028,
}


def _flag_spec(ratio, shape, field, ordinary=None, charges=None, style=None):
    spec = {
        "shield": {"kind": "flag", "ratio": tuple(ratio), "shape": shape},
        "field": field,
        "style": dict(FLAG_STYLE, **(style or {})),
    }
    if ordinary:
        spec["ordinary"] = ordinary
    if charges:
        spec["charges"] = charges
    return spec


def _thin(ratio, half_width_frac: float, constant: float) -> float:
    """Multiplier on an ordinary's canonical thickness, for a flag of ``ratio``.

    Flags are measured across their *height*, not their width, so a cross that
    should occupy a fifth of the height needs a different multiplier for every
    proportion.  This is where that correction lives.
    """
    hw = float(ratio[0]) / float(ratio[1])
    return half_width_frac / (hw * constant)


def plain(tincture: str, ratio=(3, 2), shape: str = "rectangle", style=None) -> dict:
    """A flag of one tincture."""
    return _flag_spec(ratio, shape, {"kind": "plain", "tinctures": [tincture]}, style=style)


def bicolour(a: str, b: str, direction: str = "horizontal", ratio=(3, 2), shape: str = "rectangle", style=None) -> dict:
    """Two bands: ``horizontal`` stacks them, ``vertical`` runs them hoist to fly."""
    name = "per_fess" if direction == "horizontal" else "per_pale"
    return _flag_spec(ratio, shape, {"kind": "division", "name": name, "tinctures": [a, b]}, style=style)


def tricolour(a: str, b: str, c: str, direction: str = "horizontal", ratio=(3, 2), shape: str = "rectangle", style=None) -> dict:
    """Three equal bands, the first at the chief (or at the hoist)."""
    name = "tierced_in_fess" if direction == "horizontal" else "tierced_in_pale"
    return _flag_spec(ratio, shape, {"kind": "division", "name": name, "tinctures": [a, b, c]}, style=style)


def cross_flag(field: str, cross: str, fimbriation: str | None = None, ratio=(3, 2), shape: str = "rectangle", style=None) -> dict:
    """A centred cross, optionally fimbriated."""
    ordinary = {
        "name": "cross",
        "tincture": cross,
        "thin": _thin(ratio, 0.20, _ordinaries.CROSS),
    }
    if fimbriation:
        ordinary["fimbriation"] = fimbriation
    return _flag_spec(ratio, shape, {"kind": "plain", "tinctures": [field]}, ordinary, style=style)


def nordic_cross(field: str, cross: str, fimbriation: str | None = None, ratio=(3, 2), shape: str = "rectangle", style=None) -> dict:
    """The Nordic cross: a cross whose upright is displaced towards the hoist.

    No new geometry -- it is the ``nordic_cross`` ordinary, which is a cross with
    an offset --- and the offset itself is expressed as a fraction of the flag's
    width inside :mod:`gen.heraldry.ordinaries`.
    """
    ordinary = {
        "name": "nordic_cross",
        "tincture": cross,
        "thin": _thin(ratio, 0.20, _ordinaries.CROSS),
    }
    if fimbriation:
        ordinary["fimbriation"] = fimbriation
    return _flag_spec(ratio, shape, {"kind": "plain", "tinctures": [field]}, ordinary, style=style)


def saltire_flag(field: str, saltire: str, fimbriation: str | None = None, ratio=(3, 2), shape: str = "rectangle", style=None) -> dict:
    """A saltire (St Andrew's cross) across the field."""
    ordinary = {
        "name": "saltire",
        "tincture": saltire,
        "thin": _thin(ratio, 0.13, _ordinaries.SALTIRE),
    }
    if fimbriation:
        ordinary["fimbriation"] = fimbriation
    return _flag_spec(ratio, shape, {"kind": "plain", "tinctures": [field]}, ordinary, style=style)


def canton_flag(field: str, canton: str, ratio=(3, 2), shape: str = "rectangle", style=None) -> dict:
    """A canton in the dexter chief."""
    return _flag_spec(ratio, shape, {"kind": "plain", "tinctures": [field]}, {"name": "canton", "tincture": canton}, style=style)


def hoist_triangle(field: str, triangle: str, ratio=(3, 2), shape: str = "rectangle", style=None) -> dict:
    """A triangle at the hoist, as on the flag of the Bahamas."""
    return _flag_spec(ratio, shape, {"kind": "plain", "tinctures": [field]}, {"name": "triangle_hoist", "tincture": triangle}, style=style)


def bordered(field: str, bordure: str, ratio=(3, 2), shape: str = "rectangle", style=None) -> dict:
    """A flag with a border along its own edge."""
    return _flag_spec(ratio, shape, {"kind": "plain", "tinctures": [field]}, {"name": "bordure", "tincture": bordure}, style=style)


#: Pattern names, for listings and for the Russian description.
PATTERNS = {
    "plain": ("plain", "одноцветное полотнище"),
    "bicolour": ("bicolour", "двухполосное полотнище"),
    "tricolour": ("tricolour", "трёхполосное полотнище"),
    "cross": ("cross", "полотнище с крестом"),
    "nordic_cross": ("Nordic cross", "полотнище со скандинавским крестом"),
    "saltire": ("saltire", "полотнище с косым крестом"),
    "canton": ("canton", "полотнище с вольной частью"),
    "hoist_triangle": ("hoist triangle", "полотнище с треугольником у древка"),
    "bordered": ("bordered", "полотнище с каймой"),
}

#: Which pattern name an ordinary implies, for the description.
_ORD_TO_PATTERN = {
    "cross": "cross",
    "saltire": "saltire",
    "canton": "canton",
    "triangle_hoist": "hoist_triangle",
    "bordure": "bordered",
}


def _art(word: str) -> str:
    """The indefinite article that goes before a tincture name."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def describe(spec: dict, lang: str = "en") -> str:
    """A one-line description of a flag spec.

    Flags are described, not blazoned: a vexillological description names the
    pattern and lists the tinctures in order, which is exactly what the spec
    knows about itself.  A charge, if the flag carries one, is named with the
    same grammar the blazon writer uses -- one phrase, two languages.
    """
    fspec = spec.get("field", {})
    names = fspec.get("tinctures") or [fspec.get("tincture", "argent")]
    ts = [_tinctures.get(n) for n in names]
    keys = [t.ru for t in ts] if lang == "ru" else [t.en.lower() for t in ts]
    ordinary = spec.get("ordinary") or {}
    oname = ordinary.get("name")
    fim = ordinary.get("fimbriation")
    fim_en = f" fimbriated {_tinctures.get(fim).en.lower()}" if fim else ""
    fim_ru = f" в {_tinctures.get(fim).ru_adj['instr_sg']['f']} кайме" if fim else ""
    cs = spec.get("charges") or []
    tail_en = f" Charged with {_blazon._charges_text(cs[0], 'en')}." if cs else ""
    tail_ru = f" Обременено: {_blazon._charges_text(cs[0], 'ru')}." if cs else ""

    if lang == "ru":
        if oname == "nordic_cross":
            cross = _tinctures.get(ordinary["tincture"]).ru
            prep = ts[0].ru_adj.get("prep", ts[0].ru)
            return f"Скандинавский крест ({cross}){fim_ru} на {prep} поле.{tail_ru}"
        if oname in ("cross", "saltire", "canton", "triangle_hoist", "bordure"):
            pattern = PATTERNS[_ORD_TO_PATTERN[oname]][1]
            return (f"{pattern.capitalize()} ({keys[0]}), знак — "
                    f"{_tinctures.get(ordinary['tincture']).ru}{fim_ru}.{tail_ru}")
        if fspec.get("name") in ("tierced_in_fess", "tierced_in_pale"):
            direction = "Горизонтальный" if fspec["name"] == "tierced_in_fess" else "Вертикальный"
            return f"{direction} триколор: {', '.join(keys)}.{tail_ru}"
        if fspec.get("name") in ("per_fess", "per_pale"):
            direction = "Горизонтальное" if fspec["name"] == "per_fess" else "Вертикальное"
            return f"{direction} двухполосное полотнище: {', '.join(keys)}.{tail_ru}"
        return f"Одноцветное полотнище: {keys[0]}.{tail_ru}"

    if oname == "nordic_cross":
        cross = _tinctures.get(ordinary["tincture"]).en.lower()
        return f"Nordic cross ({cross}){fim_en} on {_art(keys[0])} {keys[0]} field.{tail_en}"
    if oname in ("cross", "saltire", "canton", "triangle_hoist", "bordure"):
        subject = {
            "cross": "Cross",
            "saltire": "Saltire",
            "canton": "Canton in the dexter chief",
            "triangle_hoist": "Triangle at the hoist",
            "bordure": "Border",
        }[oname]
        mark = _tinctures.get(ordinary["tincture"]).en.lower()
        return f"{subject} ({mark}){fim_en} on {_art(keys[0])} {keys[0]} field.{tail_en}"
    if fspec.get("name") in ("tierced_in_fess", "tierced_in_pale"):
        direction = "Horizontal" if fspec["name"] == "tierced_in_fess" else "Vertical"
        return f"{direction} tricolour: {', '.join(keys)}.{tail_en}"
    if fspec.get("name") in ("per_fess", "per_pale"):
        direction = "Horizontal" if fspec["name"] == "per_fess" else "Vertical"
        return f"{direction} bicolour: {', '.join(keys)}.{tail_en}"
    return f"Plain field of {keys[0]}.{tail_en}"


def _frame_for(spec: dict) -> dict:
    """The spec with room made for what is holding the cloth, if anything is.

    The room goes into the *shape* rather than into the style margin, because the
    margin is a fraction of the whole frame while the demand is vertical: the shape's
    half-height grows and the cloth keeps its own coordinates inside it, so the same
    field function serves every case.  A holder on something that is not a flag is
    refused rather than ignored -- there is no pole that holds a shield.

    A pole and a bar can both be asked for, and that is not a contradiction but the
    classic gonfalon: a staff with a crossbar at the top and the banner hanging from
    the crossbar.  Each asks for its own headroom and the frame gets the larger of the
    two, because a frame is one rectangle and one of its two dimensions is the one
    that binds.
    """
    pole = spec.get("pole")
    bar = spec.get("bar")
    if not pole and not bar:
        return spec
    out = dict(spec)
    shield = out.get("shield")
    if isinstance(shield, str) and shield in _shields.FLAG_SHAPES:
        shield = {"kind": "flag", "ratio": (3, 2), "shape": shield}
    if not isinstance(shield, dict) or shield.get("kind") != "flag":
        raise ValueError(f"only a cloth hangs from a pole or a bar, and this is {shield!r}")
    room = max(_pole.room(pole) if pole else 0.0,
               _pole.bar_room(bar) if bar else 0.0)
    out["shield"] = dict(shield, room=room)
    return out


def render(spec: dict, size=512, background=None, shade=None):
    """Render a flag spec (the same renderer as arms), hanging if it is waved.

    A wave is part of the spec, so it survives a round trip through a sidecar and
    every caller -- the contact sheet, a faction's banner, a game's own loader --
    gets the same cloth from the same dict.  The margin is widened first: a flag
    that is going to move needs somewhere to move to.

    A ``pole`` in the spec rides along, and so does a ``bar``, and between them they
    decide how the cloth hangs: a pole holds it at the side of its folds, a bar across
    the top of them, and :mod:`gen.heraldry.cloth` has the one function for each.  Both
    are composited *after* the wave and *over* the cloth -- a holder is the one rigid
    thing on the canvas, and a bar is over the cloth's own top edge because a cloth is
    sleeved onto it.

    ``shade`` is a faction's livery; a banner that is flown for a faction is painted
    in that faction's colours, which is what keeps a shield and its flag from being
    the same *pattern* in two different pigments.
    """
    frame = _frame_for(spec)
    wave = frame.get("wave")
    bar = frame.get("bar")
    cloth_hw = 1.0
    if wave or frame.get("pole") or bar:
        cloth_hw = _blazon.resolve_shape(frame)[1]
    if wave:
        style = dict(frame.get("style", FLAG_STYLE))
        style["margin"] = float(style.get("margin", FLAG_STYLE["margin"])) + 1.6 * _cloth.reach(wave)
        canvas = _blazon.render({**frame, "style": style}, size=size, background=background,
                                shade=shade)
        # The folds are measured against the cloth, which the renderer is the one
        # that knows: the frame has been widened for the cloth to move into, and a
        # fold measured against *that* would start at the hoist only by accident.
        along_x = wave.get("direction", _cloth.DEFAULTS["direction"]) == "vertical"
        box = (-cloth_hw, cloth_hw) if along_x else (-1.0, 1.0)
        if bar:
            # A bar hangs the cloth down through its own height, and the cloth is one
            # unit tall by construction whatever the frame grew to around it.
            canvas = _cloth.drape(canvas, wave, box=box, drop=(1.0, -1.0))
        else:
            canvas = _cloth.hang(canvas, wave, box=box)
    else:
        canvas = _blazon.render(frame, size=size, background=background, shade=shade)
    if frame.get("pole"):
        # A flag's cloth is one unit tall by construction, whatever the frame grew
        # to around it, and the pole is measured against the cloth.
        _pole.paint(canvas, frame["pole"], cloth_hw, 1.0)
    if bar:
        _pole.bar_paint(canvas, bar, cloth_hw, 1.0)
    return canvas


#: Flag pattern factories, keyed by name, for the generator and the CLIs.
_FACTORIES = {
    "plain": lambda rng: plain(_pick(rng, _POOL)),
    "bicolour": lambda rng: bicolour(*_pair(rng), direction=str(rng.choice(["horizontal", "vertical"]))),
    "tricolour": lambda rng: tricolour(*_three(rng), direction=str(rng.choice(["horizontal", "vertical"]))),
    "cross": lambda rng: cross_flag(*_pair_contrast(rng), fimbriation=_maybe(rng, _POOL)),
    "nordic_cross": lambda rng: nordic_cross(*_pair_contrast(rng), fimbriation=_maybe(rng, _POOL)),
    "saltire": lambda rng: saltire_flag(*_pair_contrast(rng), fimbriation=_maybe(rng, _POOL)),
    "canton": lambda rng: canton_flag(*_pair_contrast(rng)),
    "hoist_triangle": lambda rng: hoist_triangle(*_pair_contrast(rng)),
    "bordered": lambda rng: bordered(*_pair_contrast(rng)),
}

#: Flags are textile and pull from the full classical set, unlike arms.
_POOL = [t for t in _tinctures.CLASSICAL]


def _pick(rng, pool):
    return str(rng.choice(pool))


def _pair(rng):
    a = _pick(rng, _POOL)
    b = _pick(rng, [t for t in _POOL if t != a])
    return a, b


def _pair_contrast(rng):
    """Two tinctures that at least differ in class: a flag still reads better."""
    a = _pick(rng, _POOL)
    if _tinctures.is_metal(a):
        b = _pick(rng, [t for t in _POOL if _tinctures.is_colour(t)])
    else:
        b = _pick(rng, [t for t in _POOL if _tinctures.is_metal(t)])
    # Half the time, allow a same-class pair: real flags do it constantly, but
    # a cross of one colour on a field of another reads badly, so not too often.
    if float(rng.random()) < 0.3:
        return a, _pick(rng, [t for t in _POOL if t != a])
    return a, b


def _three(rng):
    a, b = _pair(rng)
    c = _pick(rng, [t for t in _POOL if t != b])
    return a, b, c


def _maybe(rng, pool):
    return _pick(rng, pool) if float(rng.random()) < 0.45 else None


def random_flag(seed: int = 0, ratio=(3, 2), shape: str = "rectangle", style=None,
                wave=False, pole=False, bar=False) -> dict:
    """Generate a random flag spec.

    Real flags break the rule of tincture, so this does too -- but it does keep
    a metal and a colour apart more often than not, because that is what makes a
    flag legible at a distance.

    ``wave`` is off by default: the spec is a flag, and whether it is drawn
    hanging is the caller's decision.  Pass ``True`` for cloth whose folds are
    drawn from the same seed, or a dict to pin them.

    ``pole`` flies the flag from one, and it is independent of ``wave`` -- a banner
    at rest is still on a pole.  When both are asked for, the pole decides the
    cloth as well: the wave is anchored at the hoist and the fly trails, because a
    cloth tied to a pole cannot move where it is tied.

    ``bar`` hangs it from a crossbar instead, which is a gonfalon, and it decides the
    cloth by the same rule in the other direction: the bar holds the top of the folds,
    the folds deepen downwards from it, and the hem -- not the fly -- is what trails.
    The two may be asked for together, which is a staff with a crossbar on it; they may
    not be asked for with something that is not a cloth, which :func:`render` refuses.
    """
    rng = np.random.default_rng(seed)
    name = str(rng.choice(list(_FACTORIES)))
    spec = _FACTORIES[name](rng)
    spec["shield"] = {"kind": "flag", "ratio": tuple(ratio), "shape": shape}
    spec["style"] = dict(FLAG_STYLE, **(style or {}))
    # Occasional embellishment: a charge on the field, as many real flags have.
    if float(rng.random()) < 0.22:
        pool = [c for c in _charges.names() if c not in _charges.with_tag("stylised")]
        ch = str(rng.choice(pool))
        count = int(rng.choice([1, 1, 3]))
        tincture = _pick(rng, [t for t in _POOL if t != spec["field"].get("tinctures", ["argent"])[0]])
        spec["charges"] = [{"name": ch, "count": count, "arrangement": "one" if count == 1 else "three", "tincture": tincture}]
    # A pole is not cloth, so it is drawn before the wave and cannot change it.
    if pole:
        spec["pole"] = _pole.spec(**(dict(pole) if isinstance(pole, dict) else {}))
    # A bar is not cloth either, and it is drawn after the pole because it is
    # mounted on it: a gonfalon on a staff has the crossbar in front of the shaft.
    if bar:
        spec["bar"] = _pole.bar(**(dict(bar) if isinstance(bar, dict) else {}))
    # Drawn last, so that asking for cloth cannot change a flag that does not.
    if wave:
        pinned = dict(wave) if isinstance(wave, dict) else {}
        spec["wave"] = _cloth.wave(
            seed=seed + 101,
            amplitude=pinned.pop("amplitude", float(rng.uniform(0.028, 0.045))),
            wavelength=pinned.pop("wavelength", float(rng.uniform(0.22, 0.40))),
            taper=pinned.pop("taper", float(rng.uniform(0.30, 0.60))),
            harmonic=pinned.pop("harmonic", float(rng.uniform(0.15, 0.35))),
            direction=pinned.pop("direction", "vertical"),
            **pinned,
        )
        if pole or bar:
            spec["wave"] = _pole.anchor(spec["wave"])
    return spec
