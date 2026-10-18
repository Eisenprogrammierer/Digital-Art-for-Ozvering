"""Composition: a blazon specification becomes pixels, text and validation.

A spec is a plain JSON-serialisable dict::

    {
      "shield": "heater",
      "field": {"kind": "plain", "tinctures": ["azure"]},
      "ordinary": {"name": "fess", "tincture": "or"},
      "charges": [{"name": "mullet", "tincture": "argent", "count": 3,
                   "arrangement": "between_fess_three"}]
    }

Three things make this more than a serialiser:

* **The rule of tincture is checkable on real geometry.**  Because the field is
  a paint list, :func:`gen.heraldry.field.tincture_at` answers "what is under
  this point?", so validation asks the exact question heraldry asks -- what
  tincture does this charge lie on -- instead of guessing from the spec.
* **The same spec renders on any silhouette.**  Armorial bearings and flags are
  the same composition on different outlines, so a fess is a fess on a heater
  shield and on a 3:2 flag.
* **Blazon text is generated, in two languages.**  A spec already knows its own
  grammar; writing it out is what makes the system auditable by a human.  The
  English text follows the conventional word order; the Russian text uses the
  traditional terms and adjective agreement for the constructs implemented here.
"""
from __future__ import annotations

import numpy as np

from . import charges as _charges
from . import field as _field
from . import ordinaries as _ordinaries
from . import paint as _paint
from . import shields as _shields
from . import tinctures as _tinctures

__all__ = [
    "ARRANGEMENTS",
    "resolve_shape",
    "placements",
    "charge_field",
    "compose",
    "render",
    "validate",
    "blazon_text",
    "random_arms",
    "rule_ok",
]

# --------------------------------------------------------------------------
# arrangements: where the charges sit, normalised to the field's half-extents
# --------------------------------------------------------------------------
def _row(n, y, spread=0.52):
    if n == 1:
        return [(0.0, y)]
    return [(-spread + 2.0 * spread * i / (n - 1), y) for i in range(n)]


def _grid(cols, rows, w=0.50, h=0.46):
    out = []
    for r in range(rows):
        y = h * (1.0 - 2.0 * r / max(rows - 1, 1)) if rows > 1 else 0.0
        out.extend(_row(cols, y, w))
    return out


def _orle(n, radius=0.50):
    out = []
    for i in range(n):
        a = np.pi / 2.0 + 2.0 * np.pi * i / n
        out.append((radius * np.cos(a), radius * np.sin(a)))
    return out


def _bend(n, spread=0.46):
    if n == 1:
        return [(0.0, 0.0)]
    return [(-spread + 2.0 * spread * i / (n - 1), spread - 2.0 * spread * i / (n - 1)) for i in range(n)]


#: name -> (positions, default charge size as a fraction of the field's short side)
ARRANGEMENTS = {
    "one": ([(0.0, 0.0)], 0.46),
    "two": ([(-0.34, 0.04), (0.34, 0.04)], 0.36),
    "three": ([(-0.38, 0.26), (0.38, 0.26), (0.0, -0.32)], 0.28),
    "two_and_one": ([(-0.38, 0.26), (0.38, 0.26), (0.0, -0.32)], 0.28),
    "one_and_two": ([(0.0, 0.32), (-0.38, -0.26), (0.38, -0.26)], 0.28),
    "four": ([(-0.34, 0.30), (0.34, 0.30), (-0.34, -0.30), (0.34, -0.30)], 0.27),
    "five": ([(-0.36, 0.34), (0.36, 0.34), (-0.36, -0.30), (0.36, -0.30), (0.0, 0.02)], 0.24),
    "six": ([(-0.36, 0.38), (0.0, 0.38), (0.36, 0.38), (-0.36, -0.26), (0.0, -0.26), (0.36, -0.26)], 0.22),
    "seven": ([(-0.36, 0.40), (0.0, 0.40), (0.36, 0.40), (-0.36, 0.02), (0.0, 0.02), (0.36, 0.02), (0.0, -0.38)], 0.20),
    "eight": (_grid(3, 3, 0.40, 0.46)[:8], 0.19),
    "nine": (_grid(3, 3, 0.40, 0.46), 0.19),
    "in_pale": (_row(0, 0) or [], 0.26),
    "in_fess": ([], 0.26),
    "in_bend": (_bend(3), 0.26),
    "in_chief": ([(-0.36, 0.52), (0.0, 0.52), (0.36, 0.52)], 0.24),
    "in_base": ([(-0.36, -0.52), (0.0, -0.52), (0.36, -0.52)], 0.24),
    "in_orle": (_orle(6), 0.22),
    "between_fess_three": ([(-0.38, 0.52), (0.38, 0.52), (0.0, -0.50)], 0.26),
    "between_pale_three": ([(-0.55, 0.0), (0.55, 0.30), (0.55, -0.30)], 0.24),
    "around_cross": ([(-0.50, 0.50), (0.50, 0.50), (-0.50, -0.50), (0.50, -0.50)], 0.24),
    "around_chevron": ([(-0.55, -0.28), (0.0, 0.62), (0.55, -0.28)], 0.24),
    "surrounding_pale": ([(-0.50, 0.30), (-0.50, -0.30), (0.50, 0.0)], 0.24),
    "canton": ([(-0.62, 0.62)], 0.22),
}


def placements(arrangement: str, count: int, hw: float, hh: float, size: float | None = None):
    """Return ``(positions, size)`` for a charge arrangement, in world units.

    Named arrangements that encode a count (``two_and_one``) fix it themselves;
    otherwise the count selects a generic layout (1-9, then a default grid).
    """
    key = str(arrangement).strip().lower() if arrangement else ""
    if key == "in_pale":
        pts, default = [(-0.0, 0.5 - 1.0 * i / max(count - 1, 1)) for i in range(count)], 0.26
    elif key == "in_fess":
        pts, default = _row(count, 0.0, 0.52), 0.26
    elif key == "in_bend":
        pts, default = _bend(count), 0.26
    elif key == "in_orle":
        pts, default = _orle(count), 0.22
    elif key in ARRANGEMENTS:
        pts, default = ARRANGEMENTS[key]
    elif str(count) in ARRANGEMENTS:
        pts, default = ARRANGEMENTS[str(count)]
    else:
        pts, default = _grid(min(count, 3), (count + 2) // 3, 0.44, 0.48), 0.18
    world = [(px * hw * 0.94, py * hh * 0.94) for px, py in pts]
    return world, (default if size is None else float(size))


def charge_field(name: str, half_extent: float, cx: float = 0.0, cy: float = 0.0):
    """Scale a unit-box charge to ``half_extent`` world units and place it.

    Uniform scaling keeps the field metric, so the bevel shading and the
    anti-aliasing of a charge are identical at every output size.
    """
    ch = _charges.get(name)
    s = float(half_extent)
    return lambda x, y: ch((x - cx) / s, (y - cy) / s) * s


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------
def resolve_shape(spec: dict):
    """``(field_fn, hw, hh)`` for a spec's shield or flag."""
    shape = spec.get("shield", "heater")
    if isinstance(shape, dict):
        if shape.get("kind") == "flag":
            return _shields.flag(tuple(shape.get("ratio", (3, 2))), shape.get("shape", "rectangle"),
                                 shape.get("room", 0.0))
        raise ValueError(f"unknown shield spec {shape!r}")
    if shape in _shields.FLAG_SHAPES:
        return _shields.flag((3, 2), shape)
    if shape.startswith("flag"):
        return _shields.flag((3, 2), "rectangle")
    return _shields.get(shape)


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------
def compose(spec: dict, field_fn, hw: float, hh: float, shade=None):
    """The full paint list of a spec, as ``(field, tincture, emboss)`` triples.

    The ordinary is painted before the charges, and a fimbriation is just a
    wider copy of the ordinary underneath it -- which is exactly what a
    fimbriation is.

    ``shade`` is a faction's table of shaded tinctures; every name the composition
    asks for goes through it, so a shaded coat is shaded *everywhere* -- field,
    fimbriation, ordinary and charges alike -- which is the only way a livery can be
    a livery rather than a filter on part of a shield.
    """
    layers = [(region, tincture, 0.0) for region, tincture in _field.field_layers(spec.get("field", {"kind": "plain", "tinctures": ["azure"]}), hw, hh, shade=shade)]

    ord_spec = spec.get("ordinary")
    if ord_spec:
        name = ord_spec["name"]
        thin = float(ord_spec.get("thin", 1.0))
        dx, dy = float(ord_spec.get("dx", 0.0)), float(ord_spec.get("dy", 0.0))
        fim = ord_spec.get("fimbriation")
        if fim:
            wide = _ordinaries.sdf_for(name, hw, hh, thin=thin * float(ord_spec.get("fimbriation_width", 1.45)), dx=dx, dy=dy, field=field_fn)
            layers.append((wide, _tinctures.shaded(fim, shade), 0.0))
        d = _ordinaries.sdf_for(name, hw, hh, thin=thin, dx=dx, dy=dy, field=field_fn)
        layers.append((d, _tinctures.shaded(ord_spec["tincture"], shade), float(ord_spec.get("emboss", 0.0))))

    for ch in spec.get("charges", []):
        name = ch["name"]
        tincture = ch.get("tincture") or _charges.get(name).forced
        forced = _charges.get(name).forced
        if forced:
            tincture = forced
        count = int(ch.get("count", 1))
        pts, size = placements(ch.get("arrangement", "one"), count, hw, hh, ch.get("size"))
        emboss = float(ch.get("emboss", spec.get("style", {}).get("emboss", 0.0)))
        s = size * min(hw, hh)
        for i, (cx, cy) in enumerate(pts):
            if i >= count:
                break
            layers.append((charge_field(name, s, cx, cy), _tinctures.shaded(tincture, shade), emboss))
    return layers


def render(spec: dict, size=512, background=None, shade=None):
    """Render a spec to a :class:`core.canvas.Canvas`.

    With no ``shade`` this is the classical palette, which is what every caller
    that just wants a coat of arms should get.  Passing a faction's table in (see
    :func:`gen.heraldry.crest.shade_table`) paints its own livery instead: the same
    geometry, the same light, different pigments.
    """
    field_fn, hw, hh = resolve_shape(spec)
    style = spec.get("style", {})
    c = _paint.field_canvas(size, hw, hh, margin=float(style.get("margin", 0.08)), background=background)
    clip = _paint.mask_alpha(c, field_fn)
    shadow = float(style.get("shadow", 0.34))
    if shadow > 0.0:
        _paint.paint_shadow(c, field_fn, sigma=float(style.get("shadow_sigma", 0.03)), opacity=shadow)
    layers = compose(spec, field_fn, hw, hh, shade=shade)
    for region, tincture, emboss in layers:
        _paint.paint_layers(c, [(region, tincture)], clip=clip, emboss=emboss, bevel=float(style.get("bevel", 0.14)))
    rim = float(style.get("rim", 0.30))
    if rim > 0.0:
        _paint.paint_rim(c, field_fn, width=float(style.get("rim_width", 0.042)), strength=rim, light=tuple(style.get("light", _paint.LIGHT)))
    return c


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def rule_ok(below, above) -> bool:
    """Convenience wrapper over :func:`gen.heraldry.tinctures.check`."""
    return _tinctures.check(below, above)[0]


def _base_layers(spec: dict, field_fn, hw: float, hh: float):
    """The layers *underneath* the charges: field, fimbriation, ordinary."""
    return compose({**spec, "charges": []}, field_fn, hw, hh)


def _ground_layers(spec: dict, field_fn, hw: float, hh: float):
    """What a charge can stand on, with a plain fur field collapsed to the fur.

    The rule of tincture treats a fur as neutral, and a fur is painted as its two
    component tinctures -- so sampling those components where a charge stands
    would invent violations heraldry does not recognise (a charge "on Sable" when
    the field is really Ermine).  The field is therefore replaced by one layer of
    the fur itself, and everything above it -- fimbriation, ordinary -- is kept.
    """
    fspec = spec.get("field", {"kind": "plain", "tinctures": ["azure"]})
    names = fspec.get("tinctures") or [fspec.get("tincture")]
    layers = _base_layers(spec, field_fn, hw, hh)
    if fspec.get("kind", "plain") == "plain" and len(names) == 1 and _tinctures.is_fur(names[0]):
        n_field = len(_field.field_layers(fspec, hw, hh))
        above = layers[n_field:]
        return [(_field.whole_plane(), _tinctures.get(names[0]), 0.0)] + list(above)
    return layers


def _field_under(fspec: dict, field_layers, x: float, y: float):
    """What tincture the *field* has at ``(x, y)``, honouring furs.

    A plain fur field is reported as the *fur* (Ermine, Vair), not as whichever
    of its two component tinctures happens to cover the sample point -- the rule
    of tincture treats a fur as neutral, so sampling its components would invent
    violations that heraldry does not recognise.
    """
    if fspec.get("kind", "plain") == "plain":
        names = fspec.get("tinctures") or [fspec.get("tincture")]
        if len(names) == 1:
            return _tinctures.get(names[0])
    return _field.tincture_at(field_layers, x, y)


def _under_at(fspec: dict, layers, x: float, y: float):
    """What tincture lies under a charge anchor: the field *and* the ordinary."""
    return _field.tincture_at([(r, t) for r, t, _ in layers], x, y)


def validate(spec: dict, strict: bool = True) -> list:
    """Check a spec, returning a list of human-readable warnings.

    The rule of tincture is applied as practice applies it, which is not quite
    the same thing for an ordinary and for a charge:

    * A charge is judged at its **anchor points**, the places it is laid out
      from, because that is where the eye fixes it.  A charge that spans a
      partition is ordinary heraldry -- *Party per pale Gules and Azure, a lion
      Or* -- so it is not reported; what is reported is a charge that
      **disappears** into a tincture under its own footprint, which is always a
      fault.
    * An ordinary is a band running across the field, so it is judged against
      **every division it covers**.  That is why a bordure over a field that
      carries both a metal and a colour is reported: no single tincture can
      contrast with both.

    Warnings are advice, not errors -- a game faction can canonise whatever arms
    it likes, and a flag is not armorial at all.
    """
    field_fn, hw, hh = resolve_shape(spec)
    warnings = []

    fspec = spec.get("field", {})
    field_layers = _field.field_layers(fspec, hw, hh)

    ord_spec = spec.get("ordinary")
    if ord_spec:
        name = ord_spec["name"]
        ord_fn = _ordinaries.sdf_for(
            name, hw, hh,
            thin=float(ord_spec.get("thin", 1.0)),
            dx=float(ord_spec.get("dx", 0.0)),
            dy=float(ord_spec.get("dy", 0.0)),
            field=field_fn,
        )
        mask = lambda x, y: np.maximum(ord_fn(x, y), field_fn(x, y))  # noqa: E731
        for under in _footprint_tinctures(fspec, field_layers, mask, hw, hh):
            ok, why = _tinctures.check(under, ord_spec["tincture"], strict)
            if not ok:
                warnings.append(f"ordinary {name}: {why}")
                break

    under_layers = _ground_layers(spec, field_fn, hw, hh)
    for ch in spec.get("charges", []):
        name = ch["name"]
        charge = _charges.get(name)
        tincture = charge.forced or ch.get("tincture")
        if tincture is None:
            warnings.append(f"charge {name}: no tincture")
            continue
        key = _tinctures.get(tincture).key
        count = int(ch.get("count", 1))
        pts, size = placements(ch.get("arrangement", "one"), count, hw, hh, ch.get("size"))
        s = size * min(hw, hh)
        invisible = False
        for cx, cy in pts[:count]:
            under = _under_at(fspec, under_layers, cx, cy)
            if under is None:
                continue
            if under.key == key:
                warnings.append(f"charge {name}: {under.en} on {under.en} (invisible)")
                invisible = True
                continue
            ok, why = _tinctures.check(under, tincture, strict)
            if not ok:
                warnings.append(f"charge {name}: {why}")
        if not invisible:
            # Literal disappearance over part of the ground the charge covers:
            # the charge's own tincture appearing *behind* it.  Class clashes
            # across a partition are left alone; this is not a class question.
            for px, py in pts[:count]:
                for qx, qy in _field.sample_points(px, py, s * 0.9, 7):
                    u = _field.tincture_at([(r, t) for r, t, _ in under_layers], qx, qy)
                    if u is not None and u.key == key:
                        warnings.append(f"charge {name}: {u.en} ground under part of the charge (invisible there)")
                        invisible = True
                        break
                if invisible:
                    break
    return warnings


# --------------------------------------------------------------------------
# blazon text
# --------------------------------------------------------------------------
_NUM_EN = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
#: Spelled-out numbers, for "Barry of six" and "Gyronny of eight".
_NUM_WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
_NUM_RU = {
    1: {"m": "один", "f": "одна", "n": "одно"},
    2: {"m": "два", "f": "две", "n": "два"},
    3: {"m": "три", "f": "три", "n": "три"},
    4: {"m": "четыре", "f": "четыре", "n": "четыре"},
    5: {"m": "пять", "f": "пять", "n": "пять"},
    6: {"m": "шесть", "f": "шесть", "n": "шесть"},
    7: {"m": "семь", "f": "семь", "n": "семь"},
    8: {"m": "восемь", "f": "восемь", "n": "восемь"},
    9: {"m": "девять", "f": "девять", "n": "девять"},
}
_NUM_RU_INSTR = {2: "двумя", 3: "тремя", 4: "четырьмя", 5: "пятью", 6: "шестью", 7: "семью", 8: "восемью", 9: "девятью"}
_ONE_RU_INSTR = {"m": "одним", "f": "одной", "n": "одним"}

#: Russian names for divisions and variations, as used in described blazons.
#: Prepositional case, because they always appear as "В <word> поле".
_RU_FIELD = {
    "per_pale": "рассечённом",
    "per_fess": "пересечённом",
    "per_bend": "скошённом справа",
    "per_bend_sinister": "скошённом слева",
    "per_chevron": "стропильном",
    "per_chevron_inverted": "стропильном опрокинутом",
    "per_saltire": "косвенно-четверочастном",
    "quarterly": "четверочастном",
    "per_pall": "вилообразном",
    "tierced_in_pale": "рассечённом натрое",
    "tierced_in_fess": "пересечённом натрое",
    "barry": "пересечённом",
    "paly": "рассечённом",
    "bendy": "скошённом полосами",
    "bendy_sinister": "скошённом слева полосами",
    "chevronny": "стропильном",
    "checky": "шахматном",
    "gyronny": "вилообразном",
    "lozengy": "ромбовидном",
}

_EN_FIELD = {
    "per_pale": "Per pale",
    "per_fess": "Per fess",
    "per_bend": "Per bend",
    "per_bend_sinister": "Per bend sinister",
    "per_chevron": "Per chevron",
    "per_chevron_inverted": "Per chevron inverted",
    "per_saltire": "Per saltire",
    "quarterly": "Quarterly",
    "per_pall": "Per pall",
    "tierced_in_pale": "Tierced in pale",
    "tierced_in_fess": "Tierced in fess",
    "barry": "Barry",
    "paly": "Paly",
    "bendy": "Bendy",
    "bendy_sinister": "Bendy sinister",
    "chevronny": "Chevronny",
    "checky": "Checky",
    "gyronny": "Gyronny",
    "lozengy": "Lozengy",
}


def _tincture_en(t) -> str:
    return _tinctures.get(t).en


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _field_text(fspec: dict, lang: str) -> str:
    kind = fspec.get("kind", "plain")
    names = fspec.get("tinctures") or [fspec.get("tincture", "argent")]
    ts = [_tinctures.get(n) for n in names]
    if kind == "plain":
        if lang == "ru":
            return f"В {ts[0].ru_adj.get('prep', ts[0].ru)} поле"
        return ts[0].en
    name = fspec["name"]
    n = fspec.get("n")
    if lang == "ru":
        word = _RU_FIELD.get(name, name)
        if name in ("barry", "paly") and n:
            word = f"{word} {_NUM_RU.get(int(n), {'m': str(n)})['m']} раз"
        prep = [t.ru_adj.get("prep", t.ru) for t in ts]
        listed = ", ".join(prep[:-1]) + f" и {prep[-1]}" if len(prep) > 1 else prep[0]
        return f"В {word} {listed} поле"
    word = _EN_FIELD.get(name, name.replace("_", " "))
    if n:
        word = f"{word} of {_NUM_WORD.get(int(n), n)}"
    if len(ts) == 1:
        return f"{word} {ts[0].en}"
    if len(ts) == 2:
        return f"{word} {ts[0].en} and {ts[1].en}"
    return f"{word} " + ", ".join(t.en for t in ts[:-1]) + f" and {ts[-1].en}"


def _ordinary_text(ospec: dict, lang: str) -> str:
    o = _ordinaries.get(ospec["name"])
    t = _tinctures.get(ospec["tincture"])
    if lang == "ru":
        adj = t.ru_adj.get(o.gender, t.ru)
        return f"{adj} {o.ru}"
    return f"{_article(o.en)} {o.en} {t.en}"


def _gen_adj(adj: dict, gender: str, plural: bool) -> str:
    """Russian genitive adjective, derived from the nominative forms.

    The endings are regular for every tincture here (including the -ий furs),
    which is why the table only stores the nominative and instrumental forms.
    """
    if plural:
        base = adj.get("m", "")
        return base[:-2] + ("ьих" if base.endswith("ий") else "ых")
    base = adj.get(gender, adj.get("m", ""))
    if gender == "f":
        return base[:-2] + ("ьей" if base.endswith("ья") else "ой")
    return base[:-2] + ("ьего" if base.endswith("ий") else "ого")


def _charge_ru(ch: _charges.Charge, t, count: int, forced: bool):
    """``(adjective, noun)`` in Russian under numeral government.

    The rule is the ordinary one for Russian counting: nominative singular for
    one; for two to four either the whole phrase in the nominative plural
    (feminine) or the genitive singular of the noun under a genitive-plural
    adjective (masculine and neuter); genitive plural from five up.
    """
    sg, pl, adjp = _charges.gen_forms(ch.key)
    if count == 1:
        noun, adj = ch.ru, t.ru_adj.get(ch.gender, t.ru)
    elif count <= 4 and ch.gender == "f":
        noun, adj = ch.ru_pl, t.ru_adj.get("nom_pl", t.ru)
    elif count <= 4:
        # Genitive plural adjective over a genitive singular noun head, with the
        # charge's own modifier in front of the head it modifies.
        noun = f"{adjp} {sg}" if adjp else sg
        adj = _gen_adj(t.ru_adj, ch.gender, True)
    else:
        noun, adj = pl, _gen_adj(t.ru_adj, ch.gender, True)
    return (None if forced else adj), noun


def _charges_text(cspec: dict, lang: str) -> str:
    ch = _charges.get(cspec["name"])
    # A charge that names its own tincture -- a bezant is a roundel Or, a plate
    # a roundel Argent -- must not be given the adjective a second time.
    forced = ch.forced
    t = _tinctures.get(forced or cspec.get("tincture", "argent"))
    count = int(cspec.get("count", 1))
    noun_en = ch.en if count == 1 else ch.en_pl
    if lang == "ru":
        num = _NUM_RU.get(count, {"m": str(count), "f": str(count), "n": str(count)})[ch.gender]
        adj, noun = _charge_ru(ch, t, count, forced)
        return f"{num} {adj} {noun}" if adj else f"{num} {noun}"
    num = _article(ch.en) if count == 1 else _NUM_EN.get(count, str(count))
    if forced:
        return f"{num} {noun_en}"
    return f"{num} {noun_en} {t.en}"


def _charges_text_instr(cspec: dict, lang: str = "ru") -> str:
    """Russian instrumental, for use after "между" or "с"."""
    ch = _charges.get(cspec["name"])
    forced = ch.forced
    t = _tinctures.get(forced or cspec.get("tincture", "argent"))
    count = int(cspec.get("count", 1))
    if count == 1:
        noun = _charges.instr_sg(ch.key)
        if forced:
            return f"{_ONE_RU_INSTR[ch.gender]} {noun}"
        adj = t.ru_adj.get("instr_sg", {}).get(ch.gender, t.ru)
        return f"{_ONE_RU_INSTR[ch.gender]} {adj} {noun}"
    num = _NUM_RU_INSTR.get(count, str(count))
    if forced:
        return f"{num} {ch.ru_instr}"
    adj = t.ru_adj.get("instr_pl", t.ru)
    return f"{num} {adj} {ch.ru_instr}"


def blazon_text(spec: dict, lang: str = "en") -> str:
    """Write a spec out as a blazon.

    English follows the conventional order and punctuation, including the
    "charged with" construction when a charge lies on the ordinary.  Russian uses
    the traditional terms and adjective agreement for the constructs implemented
    here (plain, divided and varied fields, one ordinary, charges, and the
    "between"/"with" constructions); rarer divisions are described rather than
    idiomatically blazoned.
    """
    field_fn, hw, hh = resolve_shape(spec)
    fspec = spec.get("field", {"kind": "plain", "tinctures": ["azure"]})
    ospec = spec.get("ordinary")
    cs = list(spec.get("charges", []))

    # Does any charge actually lie on the ordinary?  Answer it from the geometry
    # rather than from the spec: if the tincture under the anchor is the
    # ordinary's, the charge is "charged upon" it.
    on_ordinary = False
    if ospec and cs:
        under_layers = _base_layers(spec, field_fn, hw, hh)
        pts, _ = placements(cs[0].get("arrangement", "one"), int(cs[0].get("count", 1)), hw, hh, cs[0].get("size"))
        if pts:
            u = _tincture_at_layers(under_layers, pts[0][0], pts[0][1])
            on_ordinary = u is not None and u.key == _tinctures.get(ospec["tincture"]).key

    arr = (cs[0].get("arrangement") if cs else "") or ""
    between = bool(cs) and arr.startswith(("between", "around"))

    if lang == "ru":
        out = [_field_text(fspec, "ru")]
        if ospec:
            out.append(_ordinary_text(ospec, "ru"))
            if between:
                out.append("между " + " и ".join(_charges_text_instr(c) for c in cs))
            elif on_ordinary:
                out.append("с " + " и ".join(_charges_text_instr(c) for c in cs))
            elif cs:
                # Attach the comma to what precedes it: appending a separate
                # ", ..." element would rejoin with a space before the comma.
                out[-1] = out[-1] + ", " + " и ".join(_charges_text(c, "ru") for c in cs)
        elif cs:
            out.append(" и ".join(_charges_text(c, "ru") for c in cs))
        text = " ".join(out)
        return text[0].upper() + text[1:] + "."

    out = [_field_text(fspec, "en")]
    tail = ""
    if ospec:
        tail = _ordinary_text(ospec, "en")
        if between:
            tail += " between " + " and ".join(_charges_text(c, "en") for c in cs)
        elif on_ordinary:
            tail += " charged with " + " and ".join(_charges_text(c, "en") for c in cs)
        elif cs:
            tail += ", " + " and ".join(_charges_text(c, "en") for c in cs)
    elif cs:
        tail = " and ".join(_charges_text(c, "en") for c in cs)
    if tail:
        return f"{out[0]}, {tail}."
    return f"{out[0]}."


def _tincture_at_layers(layers, x: float, y: float):
    return _field.tincture_at([(r, t) for r, t, _ in layers], x, y)


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------
_PLAIN_ODDS = 0.34
_DIVISION_ODDS = 0.30

_GOOD_DIVISIONS = ["per_pale", "per_fess", "per_bend", "per_bend_sinister", "per_chevron", "per_saltire", "quarterly", "per_pall", "tierced_in_pale", "tierced_in_fess"]
_GOOD_VARIATIONS = ["barry", "paly", "bendy", "bendy_sinister", "chevronny", "checky", "gyronny", "lozengy"]
_GOOD_FURS = ["ermine", "counter_ermine", "vair"]
_GOOD_ORDINARIES = ["fess", "pale", "bend", "chevron", "cross", "saltire", "chief", "bordure", "canton", "pile", "pall", "bar"]
_GOOD_CHARGES = [
    "mullet", "mullet_of_six", "estoile", "roundel", "annulet", "lozenge", "fusil", "billet",
    "escutcheon", "crescent", "fleur_de_lis", "heart", "cross_crosslet", "cross_pattee",
    "tower", "sword", "key", "tree", "escallop", "eagle", "lion", "sun",
]


def _legal(rng, unders, pool, strict: bool = True, allow_none: bool = False):
    """Pick a tincture from ``pool`` legal above *every* tincture in ``unders``.

    Returns ``None`` instead of a compromise when ``allow_none`` is set and no
    tincture satisfies all of them -- which happens on a field that carries both
    a metal and a colour, where by definition no single tincture can comply.
    Callers that cannot fail (choosing a field or ordinary tincture) leave it
    clear, and then fall back to the first beneath.
    """
    unders = [u for u in unders if u is not None]
    banned = {u.key for u in unders}

    def ok(t):
        return t not in banned and all(_tinctures.check(u, t, strict)[0] for u in unders)

    candidates = [t for t in pool if ok(t)] or [t for t in _tinctures.CLASSICAL if ok(t)]
    if not candidates:
        if allow_none:
            return None
        candidates = [t for t in pool if t not in banned] or list(pool)
    return str(rng.choice(candidates))


def _contrasting_all(rng, unders, pool, strict: bool = True):
    """``_legal`` for the cases that must return something."""
    return _legal(rng, unders, pool, strict, allow_none=False)


def _layer_paint(spec, field_fn, hw, hh):
    """The field and ordinary as ``(region, tincture)`` pairs, charges excluded."""
    return [(r, t) for r, t, _ in _ground_layers(spec, field_fn, hw, hh)]


def _charge_unders(field_fn, hw, hh, spec, name, s, pts, count):
    """``(anchor tinctures, footprint tincture keys)`` for a charge.

    The anchors are where the rule of tincture is judged; the footprint -- a
    small point cloud around each anchor, scaled to the charge -- says which
    grounds the charge actually covers, so a charge can be caught merging into
    one of them.
    """
    base = _layer_paint(spec, field_fn, hw, hh)
    anchors = [_field.tincture_at(base, cx, cy) for cx, cy in pts[:count]]
    foot = set()
    for cx, cy in pts[:count]:
        for px, py in _field.sample_points(cx, cy, s * 0.9, 7):
            u = _field.tincture_at(base, px, py)
            if u is not None:
                foot.add(u.key)
    return anchors, foot


def _legal_charge(rng, anchors, foot, pool, strict: bool = True):
    """A charge tincture that contrasts at every anchor and vanishes nowhere."""
    unders = [u for u in anchors if u is not None]
    banned = set(foot) | {u.key for u in unders}

    def ok(t):
        return t not in banned and all(_tinctures.check(u, t, strict)[0] for u in unders)

    candidates = [t for t in pool if ok(t)] or [t for t in _tinctures.CLASSICAL if ok(t)]
    return str(rng.choice(candidates)) if candidates else None


def _footprint_tinctures(fspec, field_layers, mask, hw: float, hh: float, limit: int = 16):
    """The field tinctures lying under a shape's own footprint.

    Used for ordinaries, which are bands running across the field and so must
    read against every division they cross.
    """
    xs = np.linspace(-hw, hw, 49)
    ys = np.linspace(-hh, hh, 49)
    gx, gy = np.meshgrid(xs, ys)
    inside = np.argwhere(np.asarray(mask(gx, gy)) < 0.0)
    if inside.size == 0:
        return []
    step = max(1, inside.shape[0] // limit)
    out, seen = [], set()
    for i, j in inside[::step][:limit]:
        t = _field_under(fspec, field_layers, float(gx[i, j]), float(gy[i, j]))
        if t is not None and t.key not in seen:
            seen.add(t.key)
            out.append(t)
    return out


def random_arms(seed: int = 0, shield: str | None = None, charges: bool = True, ordinary: bool = True, style: dict | None = None) -> dict:
    """Generate a random, *legal* coat of arms.

    Legality is by construction rather than by rejection: field tinctures are
    chosen so a division always contrasts, and every charge's tincture is chosen
    against the tinctures actually sampled under *all* of its anchor points --
    which is what keeps the result legal even on a gyronny or checky field.
    """
    rng = np.random.default_rng(seed)
    metals = [t for t in _tinctures.CLASSICAL if _tinctures.is_metal(t)]
    colours = [t for t in _tinctures.CLASSICAL if _tinctures.is_colour(t)]
    all_t = list(_tinctures.CLASSICAL)

    if shield is None:
        shield = str(rng.choice(["heater", "heater", "heater", "iberian", "french", "rounded", "oval", "lozenge", "banner"]))

    spec = {"shield": shield, "style": dict(style or {})}

    roll = float(rng.random())
    if roll < _PLAIN_ODDS:
        base = str(rng.choice(all_t))
        if base in _tinctures.TINCTURES and _tinctures.get(base).kind == _tinctures.FUR:
            base = str(rng.choice(all_t))
        spec["field"] = {"kind": "plain", "tinctures": [base]}
    elif roll < _PLAIN_ODDS + _DIVISION_ODDS:
        name = str(rng.choice(_GOOD_DIVISIONS))
        n_t = 3 if name in ("per_pall", "tierced_in_pale", "tierced_in_fess") else 2
        first = str(rng.choice(metals + colours))
        second = _contrasting_all(rng, [_tinctures.get(first)], [t for t in all_t if t != first])
        if not _tinctures.check(first, second)[0]:
            second = _contrasting_all(rng, [_tinctures.get(first)], all_t)
        ts = [first, second]
        if n_t == 3:
            third = _contrasting_all(rng, [_tinctures.get(second), _tinctures.get(first)], [t for t in all_t if t not in ts])
            ts.append(third)
        spec["field"] = {"kind": "division", "name": name, "tinctures": ts}
    else:
        if float(rng.random()) < 0.22:
            spec["field"] = {"kind": "plain", "tinctures": [str(rng.choice(_GOOD_FURS))]}
        else:
            name = str(rng.choice(_GOOD_VARIATIONS))
            # Gyronny must divide the circle into an even number of gyrons;
            # six, eight and twelve are the forms that actually occur.
            n = int(rng.choice([6, 8, 8, 12])) if name == "gyronny" else int(rng.integers(4, 9))
            first = str(rng.choice(metals + colours))
            second = _contrasting_all(rng, [_tinctures.get(first)], [t for t in all_t if t != first])
            if not _tinctures.check(first, second)[0]:
                second = _contrasting_all(rng, [_tinctures.get(first)], all_t)
            spec["field"] = {"kind": "variation", "name": name, "n": n, "tinctures": [first, second]}

    field_fn, hw, hh = resolve_shape(spec)

    if ordinary and float(rng.random()) < 0.62:
        field_layers = _field.field_layers(spec["field"], hw, hh)
        # Try ordinaries in random order and keep the first that can be given a
        # tincture contrasting with every division it crosses.  On a field that
        # carries both a metal and a colour many ordinaries cannot -- and then
        # the right answer is a different ordinary, not a compromise.
        for name in (str(n) for n in rng.permutation(_GOOD_ORDINARIES)):
            ord_fn = _ordinaries.sdf_for(name, hw, hh, field=field_fn)
            mask = lambda x, y, f=ord_fn: np.maximum(f(x, y), field_fn(x, y))  # noqa: E731
            unders = _footprint_tinctures(spec["field"], field_layers, mask, hw, hh)
            tin = _legal(rng, unders, all_t, allow_none=True)
            if tin is not None:
                spec["ordinary"] = {"name": name, "tincture": tin}
                break

    if charges and float(rng.random()) < (0.72 if "ordinary" in spec else 0.55):
        # Same principle: try arrangements in random order and keep the first
        # whose anchors all admit one legal tincture.  A "one" charge at the
        # centre always does, so this loop always terminates in a legal choice.
        name = str(rng.choice(_GOOD_CHARGES))
        if "ordinary" in spec:
            cands = [(a, 3) for a in ("between_fess_three", "around_cross", "around_chevron")]
            cands += [("in_chief", c) for c in (2, 3, 4)] + [("in_base", c) for c in (2, 3, 4)]
            cands += [("in_fess", 3), ("in_pale", 3), ("one", 1)]
        else:
            cands = [(str(c), c) for c in (3, 4, 5, 6)]
            cands += [("in_fess", 3), ("in_pale", 3), ("in_bend", 3), ("in_orle", 5), ("one", 1)]
        rng.shuffle(cands)
        chosen = None
        for arr, count in cands:
            pts, size = placements(arr, count, hw, hh, None)
            anchors, foot = _charge_unders(field_fn, hw, hh, spec, name, size * min(hw, hh), pts, count)
            tin = _legal_charge(rng, anchors, foot, all_t)
            if tin is not None:
                chosen = (arr, count, tin)
                break
        if chosen is None:
            pts, size = placements("one", 1, hw, hh, None)
            anchors, foot = _charge_unders(field_fn, hw, hh, spec, name, size * min(hw, hh), pts, 1)
            chosen = ("one", 1, _legal_charge(rng, anchors, foot, all_t))
        arr, count, tin = chosen
        ch = {"name": name, "count": count, "arrangement": arr, "tincture": tin}
        if float(rng.random()) < 0.35:
            ch["emboss"] = float(rng.uniform(0.05, 0.11))
        spec["charges"] = [ch]
    elif charges and "ordinary" not in spec:
        # A field alone is a thin blazon; give it something to carry.
        name = str(rng.choice(_GOOD_CHARGES))
        pts, size = placements("one", 1, hw, hh, None)
        anchors, foot = _charge_unders(field_fn, hw, hh, spec, name, size * min(hw, hh), pts, 1)
        spec["charges"] = [{
            "name": name,
            "count": 1,
            "arrangement": "one",
            "tincture": _legal_charge(rng, anchors, foot, all_t),
        }]

    return spec
