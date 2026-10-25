"""Ordinaries: the geometric bands of heraldry, in conventional proportions.

An ordinary is a shape defined *relative to the field*, which is why every
builder takes ``(hw, hh)`` and returns a distance field in world units: a fess
lands across the middle of a heater shield and across the middle of a 3:2 flag
with no special case.  Two parameters are enough to cover both the classical
coat and the flag repertoire:

``thin``
    A multiplier on the canonical thickness.  ``thin > 1`` is exactly what a
    fimbriation needs -- paint the wide band in the fimbriating tincture, then
    the normal band on top -- and small values give the thin bands of flags.
``dx, dy``
    An offset along the field.  The Nordic cross is a cross whose pale is
    displaced towards the hoist, so it is an offset, not a new shape.

Proportions are the conventional ones (Fox-Davies, *A Complete Guide to
Heraldry*): a fess occupies about a third of the field's height, a bar about a
fifth, a pale a third of the width, the chief the upper third, a bordure about a
seventh of the width.  They are named constants on purpose: a house style is a
few numbers, not a rewrite.
"""

from __future__ import annotations

import numpy as np

from core import sdf

from . import field as _field

__all__ = ["Ordinary", "ORDINARIES", "get", "names", "sdf_for"]

FESS = 1.0 / 3.0
BAR = 1.0 / 5.0
PALE = 1.0 / 3.0
BEND = 0.34
SALTIRE = 0.27
CHEVRON = 0.18
CROSS = 0.19
CHIEF = 2.0 / 3.0
BORDURE = 1.0 / 7.0
CANTON = 1.0 / 3.0


class Ordinary:
    """One ordinary: how to build it, and how to name it.

    ``gender`` is the grammatical gender of the Russian noun, needed because a
    blazon agrees its adjective with the ordinary (червлёная глава, but золотой
    пояс), and it is a property of the noun, not of the language machinery.
    """

    __slots__ = ("key", "build", "en", "ru", "gender", "tags")

    def __init__(self, key, build, en, ru, gender="m", tags=()):
        self.key = key
        self.build = build
        self.en = en
        self.ru = ru
        self.gender = gender
        self.tags = tuple(tags)

    def __call__(self, hw, hh, **kw):
        return self.build(hw, hh, **kw)

    def __repr__(self):
        return f"Ordinary({self.key})"


def _shifted(a: float, b: float, half: float, dx: float, dy: float):
    """A slab of perpendicular half-width ``half`` about the offset centre."""
    norm = float(np.hypot(a, b)) or 1.0
    u = (a * dx + b * dy) / norm
    return _field.slab(a / norm, b / norm, -half + u, half + u)


def _fess(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    return lambda x, y: sdf.box(x, y, hw * 1.4, hh * FESS * thin, 0.0, dx, dy)


def _pale(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    return lambda x, y: sdf.box(x, y, hw * PALE * thin, hh * 1.4, 0.0, dx, dy)


def _bar(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    return lambda x, y: sdf.box(x, y, hw * 1.4, hh * BAR * thin, 0.0, dx, dy)


def _bend(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    half = hw * BEND * thin
    return _shifted(hh, hw, half, dx, dy)


def _bend_sinister(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    half = hw * BEND * thin
    return _shifted(-hh, hw, half, dx, dy)


def _cross(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    th = hw * CROSS * thin
    arm_v = _shifted(1.0, 0.0, th, dx, dy)
    arm_h = _shifted(0.0, 1.0, th, dx, dy)
    return lambda x, y: np.minimum(arm_v(x, y), arm_h(x, y))


def _saltire(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    half = hw * SALTIRE * thin
    a = _shifted(hh, hw, half, dx, dy)
    b = _shifted(-hh, hw, half, dx, dy)
    return lambda x, y: np.minimum(a(x, y), b(x, y))


def _chevron(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    apex = 0.55 * hh
    m = (apex + hh) / hw
    s = float(np.hypot(1.0, m))
    half = hh * CHEVRON * thin / s
    c = apex / s

    def f(x, y):
        xx, yy = x - dx, y - dy
        u = (yy + m * np.abs(xx)) / s
        return np.abs(u - c) - half

    return f


def _chief(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    height = hh * CHIEF * thin
    cy = hh - height * 0.5
    return lambda x, y: sdf.box(x, y, hw * 1.4, height * 0.5, 0.0, dx, cy + dy)


def _base(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    height = hh * CHIEF * thin
    cy = -hh + height * 0.5
    return lambda x, y: sdf.box(x, y, hw * 1.4, height * 0.5, 0.0, dx, cy + dy)


def _bordure(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    if field is None:
        raise ValueError("a bordure needs the silhouette: pass field=<shield field>")
    t = hw * BORDURE * thin

    def f(x, y):
        d = field(x - dx, y - dy)
        return np.abs(d + t * 0.5) - t * 0.5

    return f


def _canton(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    half = hw * CANTON * thin
    return lambda x, y: sdf.box(x, y, half, half, 0.0, -hw + half + dx, hh - half + dy)


def _pile(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    half = hw * CANTON * thin
    pts = np.array([[-half + dx, hh], [half + dx, hh], [dx, -hh]])
    return lambda x, y: sdf.polygon(x, y, pts)


def _pall(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    t = hw * CROSS * thin
    stem = lambda x, y: sdf.box(x, y, t, hh, 0.0, dx, hh * 0.5 + dy)

    dexter = _shifted(hh, hw, t, dx, dy)
    sinister = _shifted(-hh, hw, t, dx, dy)

    def f(x, y):
        return np.minimum(stem(x, y), np.minimum(dexter(x, y), sinister(x, y)))

    return f

ORDINARIES = {
    o.key: o
    for o in [
        Ordinary("pale", _pale, "pale", "столб", "m"),
        Ordinary("fess", _fess, "fess", "пояс", "m"),
        Ordinary("bar", _bar, "bar", "балка", "f"),
        Ordinary("bend", _bend, "bend", "перевязь", "f"),
        Ordinary("bend_sinister", _bend_sinister, "bend sinister", "перевязь слева", "f"),
        Ordinary("chevron", _chevron, "chevron", "стропило", "n"),
        Ordinary("cross", _cross, "cross", "крест", "m"),
        Ordinary("saltire", _saltire, "saltire", "косой крест", "m"),
        Ordinary("chief", _chief, "chief", "глава", "f"),
        Ordinary("base", _base, "base", "оконечность", "f"),
        Ordinary("bordure", _bordure, "bordure", "кайма", "f", tags=("needs_field",)),
        Ordinary("canton", _canton, "canton", "вольная часть", "f"),
        Ordinary("pile", _pile, "pile", "остриё", "n"),
        Ordinary("pall", _pall, "pall", "вилообразный крест", "m"),
    ]
}

ORDINARIES["nordic_cross"] = Ordinary("nordic_cross", lambda hw, hh, thin=1.0, dx=0.0, dy=0.0, field=None: _cross(hw, hh, thin, -hw * 0.62 + dx, dy, field), "Nordic cross", "скандинавский крест", "m")


def _hoist_triangle(hw, hh, thin: float = 1.0, dx: float = 0.0, dy: float = 0.0, field=None):
    """A triangle based on the hoist edge, pointing at the fly.

    Not a pile: a pile hangs from the chief, whereas this stands on the hoist, so
    the two are different shapes that merely both happen to be triangles.
    """
    base = -hw * 1.10 + dx
    apex = hw * (0.45 + 0.35 * float(thin)) + dx
    pts = np.array([[base, hh * 1.15 + dy], [base, -hh * 1.15 + dy], [apex, dy]])
    return lambda x, y: sdf.polygon(x, y, pts)

ORDINARIES["triangle_hoist"] = Ordinary("triangle_hoist", _hoist_triangle, "hoist triangle", "треугольник у древка", "m")


def get(name) -> Ordinary:
    """Look up an ordinary by name."""
    if isinstance(name, Ordinary):
        return name
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in ORDINARIES:
        raise KeyError(f"unknown ordinary {name!r}; known: {', '.join(sorted(ORDINARIES))}")
    return ORDINARIES[key]


def names() -> list:
    return sorted(ORDINARIES)


def sdf_for(name, hw: float, hh: float, **kwargs):
    """Build the distance field for a named ordinary (``field=`` for bordure)."""
    return get(name)(hw, hh, **kwargs)
