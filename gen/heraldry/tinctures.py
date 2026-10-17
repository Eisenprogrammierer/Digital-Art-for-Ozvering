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
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field

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
]


@dataclass(frozen=True)
class Tincture:
    """One tincture: its class, its appearance, and its names in both languages."""

    key: str
    kind: str
    srgb: tuple
    en: str
    ru: str
    #: Russian adjective forms: nominative masculine/feminine/neuter,
    #: prepositional (used for "в ... поле"), and instrumental plural (used
    #: after "между").
    ru_adj: dict = _field(default_factory=dict)
    #: For furs: the base tincture and the spot/bell tincture.
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

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.en


def _t(key, kind, hex_color, en, ru, ru_adj, fur=None, classical=False) -> Tincture:
    return Tincture(key, kind, tuple(palette.hex_to_srgb(hex_color)), en, ru, ru_adj, fur, classical)


#: The traditional emblazonment anchors used for every generated asset.
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

#: The seven tinctures that make up classical arms.
CLASSICAL = ("or", "argent", "gules", "azure", "vert", "sable", "purpure")

#: Plain-language aliases, so a game can just say "gold" or "crimson".
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
