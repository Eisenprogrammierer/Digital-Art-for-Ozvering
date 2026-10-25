"""The heraldic tincture system, and the rule of tincture.

Tinctures come in three classes that behave differently when combined:

* **metals** -- Or (gold) and Argent (silver);
* **colours** -- Gules, Azure, Vert, Sable, Purpure, plus the three *stains*
  (Tenny, Sanguine, Murrey) that heraldry treats as colours for contrast;
* **furs** -- Ermine, Counter-ermine, Vair, Counter-vair, Potent, which are
  neutral: a fur may lie on anything.

That classification exists to serve one rule, the **rule of tincture**: never
place metal on metal or colour on colour.  Because it is a rule about *classes*,
it is enforceable in code -- see :func:`check` -- and the random arms generator
uses it to reject illegal combinations before they are ever rendered.

Colours are anchored to traditional emblazonment values in sRGB (so they look
like heraldry) and reasoning happens in OKLCH (so variants and contrast
measurements are perceptually sound).

The last section is the *livery*: how a faction gets its own shade of the seven
classical tinctures without leaving the system or re-deriving the contrast rules.
It is drawn as integers on a fixed budget -- see :data:`SHADE` -- and the budget is
spent only where the colour has room to spend it, because a shade that the gamut
quietly clipped is a number in a sidecar that no pixel ever showed.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from functools import lru_cache

import numpy as np

from core import palette

METAL, COLOUR, FUR = "metal", "colour", "fur"

__all__ = [
    "METAL",
    "COLOUR",
    "FUR",
    "Tincture",
    "TINCTURES",
    "CLASSICAL",
    "get",
    "names",
    "is_metal",
    "is_colour",
    "is_fur",
    "check",
    "contrast",
    "variant",
    "SHADE",
    "SHADE_AXES",
    "random_shade",
    "shade_table",
    "shaded",
]


@dataclass(frozen=True)
class Tincture:
    """One tincture: its class, its appearance, and its names in both languages."""

    key: str
    kind: str
    srgb: tuple
    en: str
    ru: str

    ru_adj: dict = _field(default_factory=dict)

    fur: tuple | None = None
    classical: bool = False

    @property
    def hex(self) -> str:
        return palette.srgb_to_hex(np.asarray(self.srgb))

    @property
    def rgb(self) -> np.ndarray:
        return np.asarray(self.srgb, dtype=np.float64)

    @property
    def lch(self) -> np.ndarray:
        return palette.srgb_to_oklch(self.rgb)

    @property
    def lightness(self) -> float:
        return float(self.lch[0])

    def __str__(self) -> str:
        return self.en


def _t(key, kind, hex_color, en, ru, ru_adj, fur=None, classical=False) -> Tincture:
    return Tincture(key, kind, tuple(palette.hex_to_srgb(hex_color)), en, ru, ru_adj, fur, classical)

TINCTURES: dict = {
    t.key: t
    for t in [
        _t("or", METAL, "#F2C200", "Or", "золото",
           {"m": "золотой", "f": "золотая", "n": "золотое", "prep": "золотом", "instr_sg": {"m": "золотым", "n": "золотым", "f": "золотой"}, "instr_pl": "золотыми", "nom_pl": "золотые"},
           classical=True),
        _t("argent", METAL, "#F4F6F7", "Argent", "серебро",
           {"m": "серебряный", "f": "серебряная", "n": "серебряное", "prep": "серебряном", "instr_sg": {"m": "серебряным", "n": "серебряным", "f": "серебряной"}, "instr_pl": "серебряными", "nom_pl": "серебряные"},
           classical=True),
        _t("gules", COLOUR, "#C8102E", "Gules", "червлень",
           {"m": "червлёный", "f": "червлёная", "n": "червлёное", "prep": "червлёном", "instr_sg": {"m": "червлёным", "n": "червлёным", "f": "червлёной"}, "instr_pl": "червлёными", "nom_pl": "червлёные"},
           classical=True),
        _t("azure", COLOUR, "#0057B7", "Azure", "лазурь",
           {"m": "лазоревый", "f": "лазоревая", "n": "лазоревое", "prep": "лазоревом", "instr_sg": {"m": "лазоревым", "n": "лазоревым", "f": "лазоревой"}, "instr_pl": "лазоревыми", "nom_pl": "лазоревые"},
           classical=True),
        _t("vert", COLOUR, "#007A33", "Vert", "зелень",
           {"m": "зелёный", "f": "зелёная", "n": "зелёное", "prep": "зелёном", "instr_sg": {"m": "зелёным", "n": "зелёным", "f": "зелёной"}, "instr_pl": "зелёными", "nom_pl": "зелёные"},
           classical=True),
        _t("sable", COLOUR, "#15161A", "Sable", "чернь",
           {"m": "чёрный", "f": "чёрная", "n": "чёрное", "prep": "чёрном", "instr_sg": {"m": "чёрным", "n": "чёрным", "f": "чёрной"}, "instr_pl": "чёрными", "nom_pl": "чёрные"},
           classical=True),
        _t("purpure", COLOUR, "#6D2077", "Purpure", "пурпур",
           {"m": "пурпурный", "f": "пурпурная", "n": "пурпурное", "prep": "пурпурном", "instr_sg": {"m": "пурпурным", "n": "пурпурным", "f": "пурпурной"}, "instr_pl": "пурпурными", "nom_pl": "пурпурные"},
           classical=True),
        _t("tenny", COLOUR, "#DD6600", "Tenny", "тенни",
           {"m": "оранжевый", "f": "оранжевая", "n": "оранжевое", "prep": "оранжевом", "instr_sg": {"m": "оранжевым", "n": "оранжевым", "f": "оранжевой"}, "instr_pl": "оранжевыми", "nom_pl": "оранжевые"}),
        _t("sanguine", COLOUR, "#7F1B2E", "Sanguine", "сангвин",
           {"m": "кровавый", "f": "кровавая", "n": "кровавое", "prep": "кровавом", "instr_sg": {"m": "кровавым", "n": "кровавым", "f": "кровавой"}, "instr_pl": "кровавыми", "nom_pl": "кровавые"}),
        _t("murrey", COLOUR, "#8A2B5C", "Murrey", "мюррей",
           {"m": "тёмно-малиновый", "f": "тёмно-малиновая", "n": "тёмно-малиновое",
            "prep": "тёмно-малиновом", "instr_sg": {"m": "тёмно-малиновым", "n": "тёмно-малиновым", "f": "тёмно-малиновой"}, "instr_pl": "тёмно-малиновыми", "nom_pl": "тёмно-малиновые"}),
        _t("ermine", FUR, "#F4F6F7", "Ermine", "горностай",
           {"m": "горностаевый", "f": "горностаевая", "n": "горностаевое",
            "prep": "горностаевом", "instr_sg": {"m": "горностаевым", "n": "горностаевым", "f": "горностаевой"}, "instr_pl": "горностаевыми", "nom_pl": "горностаевые"},
           fur=("argent", "sable"), classical=True),
        _t("counter_ermine", FUR, "#15161A", "Counter-ermine", "противогорностай",
           {"m": "противогорностаевый", "f": "противогорностаевая", "n": "противогорностаевое",
            "prep": "противогорностаевом", "instr_sg": {"m": "противогорностаевым", "n": "противогорностаевым", "f": "противогорностаевой"}, "instr_pl": "противогорностаевыми", "nom_pl": "противогорностаевые"},
           fur=("sable", "argent")),
        _t("vair", FUR, "#F4F6F7", "Vair", "беличий мех",
           {"m": "беличий", "f": "беличья", "n": "беличье",
            "prep": "беличьем", "instr_sg": {"m": "беличьим", "n": "беличьим", "f": "беличьей"}, "instr_pl": "беличьими", "nom_pl": "беличьи"},
           fur=("argent", "azure"), classical=True),
        _t("counter_vair", FUR, "#0057B7", "Counter-vair", "противобеличий",
           {"m": "противобеличий", "f": "противобеличья", "n": "противобеличье",
            "prep": "противобеличьем", "instr_sg": {"m": "противобеличьим", "n": "противобеличьим", "f": "противобеличьей"}, "instr_pl": "противобеличьими", "nom_pl": "противобеличьи"},
           fur=("azure", "argent")),
        _t("potent", FUR, "#F4F6F7", "Potent", "потент",
           {"m": "потентовый", "f": "потентовая", "n": "потентовое",
            "prep": "потентовом", "instr_sg": {"m": "потентовым", "n": "потентовым", "f": "потентовой"}, "instr_pl": "потентовыми", "nom_pl": "потентовые"},
           fur=("argent", "purpure")),
    ]
}

CLASSICAL = ("or", "argent", "gules", "azure", "vert", "sable", "purpure")

_ALIASES = {
    "gold": "or", "yellow": "or",
    "silver": "argent", "white": "argent",
    "red": "gules", "crimson": "gules",
    "blue": "azure",
    "green": "vert",
    "black": "sable",
    "purple": "purpure",
    "orange": "tenny",
    "blood": "sanguine",
    "mulberry": "murrey",
    "ermines": "counter_ermine",
    "counter-ermine": "counter_ermine",
    "counterermine": "counter_ermine",
    "counter-vair": "counter_vair",
    "countervair": "counter_vair",
}


def get(name) -> Tincture:
    """Look up a tincture by canonical name or plain-language alias."""
    if isinstance(name, Tincture):
        return name
    key = str(name).strip().lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in TINCTURES:
        raise KeyError(f"unknown tincture {name!r}; known: {', '.join(TINCTURES)}")
    return TINCTURES[key]


def names(kind: str | None = None, classical_only: bool = False) -> list:
    """List tincture keys, optionally filtered by class."""
    return [
        k
        for k, t in TINCTURES.items()
        if (kind is None or t.kind == kind) and (not classical_only or t.classical)
    ]


def is_metal(name) -> bool:
    return get(name).kind == METAL


def is_colour(name) -> bool:
    return get(name).kind == COLOUR


def is_fur(name) -> bool:
    return get(name).kind == FUR


def check(below, above, strict: bool = True) -> tuple:
    """Does placing tincture ``above`` on ``below`` obey the rule of tincture?

    Returns ``(ok, reason)``.  A fur on either side is always legal.  With
    ``strict=False`` the three stains are reported but not treated as errors,
    which matches how they are actually used.
    """
    a = get(below)
    b = get(above)
    if a.kind == FUR or b.kind == FUR:
        return True, "fur is neutral"
    if a.kind == b.kind:
        what = "metal on metal" if a.kind == METAL else "colour on colour"
        return False, f"{what}: {a.en} below {b.en}"
    return True, "metal on colour" if a.kind == METAL else "colour on metal"


def contrast(a, b) -> float:
    """Perceptual distance (OKLab) between two tinctures."""
    return palette.delta_e(get(a).rgb, get(b).rgb)


def variant(name, d_lightness: float = 0.0, d_chroma: float = 0.0, d_hue: float = 0.0, prefix: str = "~") -> Tincture:
    """A tincture of the same class, nudged in OKLCH.

    This is how a game gets faction-specific variants -- a house whose "azure"
    leans teal, a pretender whose "gules" leans orange -- without leaving the
    heraldic system or re-deriving the contrast rules.
    """
    t = get(name)
    l, c, h = t.lch
    lch = (float(np.clip(l + d_lightness, 0.05, 0.98)), max(c + d_chroma, 0.0), (h + d_hue) % 1.0)
    srgb = tuple(palette.oklch_to_srgb(lch))
    fur = None
    if t.fur is not None:
        fur = (t.fur[0], variant(t.fur[1], d_lightness, d_chroma, d_hue).key)
    return Tincture(
        key=f"{prefix}{t.key}",
        kind=t.kind,
        srgb=srgb,
        en=f"{t.en} variant",
        ru=f"{t.ru} (вариант)",
        ru_adj=dict(t.ru_adj),
        fur=fur,
        classical=False,
    )

SHADE = {
    METAL: {"l": (0.16, 8), "c": (0.05, 3), "h": (0.05, 4)},
    COLOUR: {"l": (0.16, 6), "c": (0.14, 5), "h": (0.09, 6)},
}

SHADE_AXES = ("l", "c", "h")

_L_ROOM = (0.06, 0.97)

_MIN_STEP = 0.004


def _chroma_room_up(name: str) -> float:
    """How much chroma this tincture can *gain* before the gamut gives it back.

    Azure at its own lightness already sits on the edge of sRGB, so "more
    chromatic" is not a shade of it -- it is the same pixel: ``fit_gamut``
    reproduces it by handing the chroma back, and a budget spent there would be a
    budget that lies.  So the room is *measured*, by bisecting until the round trip
    stops being lossless, rather than taken from the table.
    """
    lightness, chroma, hue = get(name).lch
    lo, hi = 0.0, 0.30
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        back = palette.srgb_to_oklch(palette.oklch_to_srgb((lightness, chroma + mid, hue)))
        if abs(float(back[1]) - (chroma + mid)) < 1e-3:
            lo = mid
        else:
            hi = mid
    return lo


@lru_cache(maxsize=None)
def _room(name: str, axis: str) -> tuple:
    """How far this tincture can move on ``axis``, in each direction, per its room.

    Two things can make a direction worthless, and both are *measured* rather than
    assumed, because neither threshold is one anybody can pick by eye:

    * the gamut -- Azure already sits on the edge of sRGB, so "more chromatic" is
      not a shade of it but the same pixel (see :func:`_chroma_room_up`), and
      Sable has no room left to go darker;
    * the colour -- a *hue* on a tincture with 0.0025 chroma is a hue on nothing,
      and a step there would be a number in a sidecar that never reached a pixel.

    So a direction is offered only if a single step along it buys at least
    :data:`_MIN_STEP` of colour.  Cached per name, because it costs a bisection and
    depends on nothing else: a game that shades a faction's whole army pays once.
    """
    t = get(name)
    span, count = SHADE[t.kind][axis]
    half = span / 2.0
    if axis == "l":
        lightness = t.lch[0]
        lo = -min(half, max(lightness - _L_ROOM[0], 0.0))
        hi = min(half, max(_L_ROOM[1] - lightness, 0.0))
    elif axis == "c":
        lo = -min(half, max(t.lch[1], 0.0))
        hi = min(half, _chroma_room_up(name))
    else:
        lo, hi = -half, half
    return _live(name, axis, lo, count), _live(name, axis, hi, count)


def _live(name: str, axis: str, room: float, count: int) -> float:
    """The room in one direction, or nothing if a single step there is not a step."""
    if room == 0.0:
        return 0.0
    return room if _moved(name, axis, room / count) >= _MIN_STEP else 0.0


def _moved(name: str, axis: str, offset: float) -> float:
    """How far an ``offset`` of one step along ``axis`` moves the tincture, in OKLab."""
    lightness, chroma, hue = get(name).lch
    if axis == "l":
        lch = (lightness + offset, chroma, hue)
    elif axis == "c":
        lch = (lightness, max(chroma + offset, 0.0), hue)
    else:
        lch = (lightness, chroma, (hue + offset) % 1.0)
    return palette.delta_e(get(name).srgb, palette.oklch_to_srgb(lch))


def _offset(step: int, count: int, room: tuple) -> float:
    """Map an integer step onto the room available on the side it moves toward."""
    if step > 0:
        return room[1] * (step / count)
    return room[0] * (-step / count)


def random_shade(seed, names=CLASSICAL) -> dict:
    """A faction's livery, drawn once from a seed, as integer steps per tincture.

    Steps are drawn on every axis and then cleared from the directions that have no
    room, rather than not drawn at all: the stream stays the same length whatever a
    tincture can take, and a step that is cleared is a step that would not have
    changed the picture -- a number in a sidecar that lies about the shield.

    Furs are skipped, and that is not a limitation: a fur is a pattern of two
    ordinary tinctures, so shading those shades the fur.  Shading a fur *itself*
    would mean shading something that is not a colour.
    """
    rng = np.random.default_rng(int(seed))
    table = {}
    for name in names:
        t = get(name)
        if t.kind not in SHADE:
            continue
        steps = {}
        for axis in SHADE_AXES:
            count = SHADE[t.kind][axis][1]
            step = int(rng.integers(-count, count + 1))
            room = _room(name, axis)
            if (step > 0 and room[1] <= 0.0) or (step < 0 and room[0] >= 0.0):
                step = 0
            steps[axis] = step
        table[str(name)] = steps
    return table


def shade_table(table) -> dict:
    """Turn a faction's integer shade into the tinctures to paint with."""
    out = {}
    for name, steps in (table or {}).items():
        t = get(name)
        if t.kind not in SHADE:
            continue
        deltas = [
            _offset(int(steps.get(axis, 0)), SHADE[t.kind][axis][1], _room(name, axis))
            for axis in SHADE_AXES
        ]
        out[str(name)] = variant(name, d_lightness=deltas[0], d_chroma=deltas[1],
                                 d_hue=deltas[2])
    return out


def shaded(name, shade=None):
    """The tincture to paint when a faction's shade is in force.

    One lookup with one rule: a name in the table is shaded, anything else resolves
    as itself -- so a table that mentions three tinctures does not have to mention
    the other four.  A fur's *components* go through the same table, which is what
    makes a faction with an ermine field distinguishable at all: the fur is the
    pattern, and the pattern is the two tinctures inside it.
    """
    if shade:
        found = shade.get(str(name))
        if found is not None:
            return found
    return get(name)
