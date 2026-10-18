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


def render(spec: dict, size=512, background=None, shade=None):
    """Render a flag spec (the same renderer as arms), hanging if it is waved.

    A wave is part of the spec, so it survives a round trip through a sidecar and
    every caller -- the contact sheet, a faction's banner, a game's own loader --
    gets the same cloth from the same dict.  The margin is widened first: a flag
    that is going to move needs somewhere to move to.

    ``shade`` is a faction's livery; a banner that is flown for a faction is painted
    in that faction's colours, which is what keeps a shield and its flag from being
    the same *pattern* in two different pigments.
    """
    wave = spec.get("wave")
    if not wave:
        return _blazon.render(spec, size=size, background=background, shade=shade)
    style = dict(spec.get("style", FLAG_STYLE))
    style["margin"] = float(style.get("margin", FLAG_STYLE["margin"])) + 1.6 * _cloth.reach(wave)
    canvas = _blazon.render({**spec, "style": style}, size=size, background=background,
                            shade=shade)
    return _cloth.hang(canvas, wave)


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
                wave=False) -> dict:
    """Generate a random flag spec.

    Real flags break the rule of tincture, so this does too -- but it does keep
    a metal and a colour apart more often than not, because that is what makes a
    flag legible at a distance.

    ``wave`` is off by default: the spec is a flag, and whether it is drawn
    hanging is the caller's decision.  Pass ``True`` for cloth whose folds are
    drawn from the same seed, or a dict to pin them.
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
    return spec
