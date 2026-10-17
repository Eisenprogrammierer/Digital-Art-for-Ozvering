"""Charges: the mobile figures of heraldry.

Every charge is built in a **unit box** (roughly ``[-1, 1]`` in both axes) as a
distance field, so the composer can place it anywhere at any size and the result
stays resolution-free.  Charges compose out of the same primitives as everything
else -- capsules, discs and polygons -- with two consequences worth naming:

* symmetry is free (:func:`core.sdf.mirrored`), which is exactly the heraldic
  aesthetic: the fleur-de-lis, the eagle displayed, the mullet;
* beasts are approximated as skeletons of capsules welded with a smooth union,
  which gives a stylised, clearly *generated* beast rather than a traced one.

A few charges are roundels of a fixed tincture -- a bezant is a roundel Or, a
torteau a roundel Gules -- so they carry ``forced`` and the composer obeys it.
"""

from __future__ import annotations

import numpy as np

from core import sdf

from . import field as _field
from . import shields as _shields

__all__ = ["Charge", "CHARGES", "get", "names", "with_tag"]


class Charge:
    """One charge: how to build it, and how to name it in both languages."""

    __slots__ = ("key", "build", "en", "en_pl", "ru", "ru_pl", "ru_instr", "gender", "forced", "tags")

    def __init__(self, key, build, en, en_pl, ru, ru_pl, ru_instr, gender="m", forced=None, tags=()):
        self.key = key
        self.build = build
        self.en = en
        self.en_pl = en_pl
        self.ru = ru
        self.ru_pl = ru_pl
        self.ru_instr = ru_instr
        self.gender = gender
        self.forced = forced
        self.tags = tuple(tags)

    def __call__(self, x, y):
        return self.build(x, y)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Charge({self.key})"


# --------------------------------------------------------------------------
# small composition helpers
# --------------------------------------------------------------------------
def _U(*fields):
    return lambda x, y: np.minimum.reduce([f(x, y) for f in fields])


def _D(a, b):
    return lambda x, y: np.maximum(a(x, y), -b(x, y))


def _DT(base, *cuts):
    """Subtract any number of shapes from a base."""
    return lambda x, y: np.maximum(base(x, y), -np.minimum.reduce([c(x, y) for c in cuts]))


def _S(k, *fields):
    def f(x, y):
        out = fields[0](x, y)
        for g in fields[1:]:
            out = sdf.smooth_union(out, g(x, y), k)
        return out

    return f


def _mirror(f):
    return lambda x, y: f(np.abs(x), y)


def _fit(f, extent: float | None = None):
    """Normalise a charge so its silhouette fits the unit box, metric intact.

    Most charges are built to sit inside ``|x|, |y| <= 1`` already, but a ring
    naturally overhangs the radius it is described by and a blade overhangs the
    length of its point.  Normalising keeps one promise instead of four special
    cases: a charge drawn at a given size *is* that size, so an annulet and a
    roundel at the same size line up.

    The extent is measured rather than written down, so editing a shape cannot
    leave a stale constant behind.
    """
    if extent is None:
        xs = np.linspace(-2.2, 2.2, 401)
        gx, gy = np.meshgrid(xs, xs)
        inside = np.asarray(f(gx, gy)) < 0.0
        if inside.any():
            rows = np.where(inside.any(axis=1))[0]
            cols = np.where(inside.any(axis=0))[0]
            extent = max(abs(xs[rows].min()), abs(xs[rows].max()),
                         abs(xs[cols].min()), abs(xs[cols].max()))
        else:
            extent = 1.0
    # Land just inside the box: the measurement itself is quantised by the grid,
    # and a hair of margin keeps the charge off the edge under any sampling.
    s = float(extent) / 0.99
    return lambda x, y: f(x * s, y * s) / s


def _polar(radius, n: int = 256):
    """Closed shape from a polar radius function, as an exact polygon."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = np.asarray(radius(t), dtype=np.float64)
    pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    return lambda x, y: sdf.polygon(x, y, pts)


def _poly(pts):
    arr = np.asarray(pts, dtype=np.float64)
    return lambda x, y: sdf.polygon(x, y, arr)


# --------------------------------------------------------------------------
# geometric charges
# --------------------------------------------------------------------------
def _star(n):
    return lambda x, y: sdf.star(x, y, n, 1.0, 0.42)


def _estoile(n=6, waviness: float = 0.58):
    """A wavy-rayed star: the radius swells into rays instead of straight spikes."""

    def radius(t):
        return 0.40 + 0.60 * np.abs(np.cos(n * (t - np.pi / 2.0) / 2.0)) ** 0.75

    return _polar(radius, 288)


def _roundel():
    return lambda x, y: sdf.circle(x, y, 1.0)


def _annulet():
    # A ring measured by its centre line overhangs the box by its own thickness.
    return _fit(lambda x, y: sdf.ring(x, y, 0.94, 0.17))


def _lozenge():
    return _poly([(0.0, 1.0), (0.70, 0.0), (0.0, -1.0), (-0.70, 0.0)])


def _fusil():
    return _poly([(0.0, 1.0), (0.34, 0.0), (0.0, -1.0), (-0.34, 0.0)])


def _billet():
    return lambda x, y: sdf.box(x, y, 0.42, 0.78, 0.05)


def _escutcheon():
    return _shields._heater(0.62, 1.0)[0]


def _crescent():
    def f(x, y):
        return sdf.difference(sdf.circle(x, y, 1.0), sdf.circle(x, y, 0.84, 0.0, 0.32))

    return f


def _heart():
    def f(x, y):
        left = sdf.circle(x, y, 0.52, -0.40, 0.34)
        right = sdf.circle(x, y, 0.52, 0.40, 0.34)
        body = sdf.polygon(x, y, [(-0.90, 0.36), (0.90, 0.36), (0.0, -0.96)])
        return sdf.smooth_union(sdf.smooth_union(left, right, 0.10), body, 0.05)

    return f


def _sun(n=12):
    """Sun in splendour: a disc with triangular rays."""
    disc = lambda x, y: sdf.circle(x, y, 0.42)  # noqa: E731
    rays = []
    for i in range(n):
        a = np.pi / 2.0 + 2.0 * np.pi * i / n
        d = np.pi / (2.0 * n)
        inner, outer = 0.32, 1.0
        rays.append(
            _poly(
                [
                    (inner * np.cos(a - d), inner * np.sin(a - d)),
                    (outer * np.cos(a), outer * np.sin(a)),
                    (inner * np.cos(a + d), inner * np.sin(a + d)),
                ]
            )
        )
    return _U(disc, *rays)


# --------------------------------------------------------------------------
# named figures
# --------------------------------------------------------------------------
_RIGHT = [
    # Right half of the fleur-de-lis outline, from the base up to the tip: base,
    # band, the side petal's outer and inner edges, the notch, then the central
    # petal.  The deep notch down to the band is what separates the three
    # petals, and it costs nothing because the polygon is concave.
    (0.30, -1.00),
    (0.26, -0.62),
    (0.20, -0.44),
    (0.62, -0.44),
    (0.64, -0.24),
    (0.80, 0.16),
    (0.84, 0.48),
    (0.70, 0.72),
    (0.54, 0.50),
    (0.44, 0.22),
    (0.34, 0.00),
    (0.24, -0.18),
    (0.28, 0.12),
    (0.30, 0.50),
    (0.15, 0.86),
    (0.00, 1.00),
]


def _fleur_de_lis():
    """The classic three-petal lily, drawn as one mirrored outline.

    Build the right half, then mirror it: the notches between the petals and the
    horizontal band fall out of a single concave polygon, which is why the SDF
    stays exact.
    """
    half = [(-_RIGHT[0][0], _RIGHT[0][1])] + list(_RIGHT)
    half += [(-x, y) for x, y in reversed(_RIGHT[1:-1])]
    return _poly(half)


def _cross_crosslet():
    arms = [
        lambda x, y: sdf.box(x, y, 0.16, 0.80),
        lambda x, y: sdf.box(x, y, 0.80, 0.16),
    ]
    caps = [
        lambda x, y: sdf.box(x, y, 0.34, 0.075, 0.0, 0.0, 0.80),
        lambda x, y: sdf.box(x, y, 0.34, 0.075, 0.0, 0.0, -0.80),
        lambda x, y: sdf.box(x, y, 0.075, 0.34, 0.0, 0.80, 0.0),
        lambda x, y: sdf.box(x, y, 0.075, 0.34, 0.0, -0.80, 0.0),
    ]
    return _U(*(arms + caps))


def _cross_pattee():
    arm = _poly([(-0.15, 0.0), (0.15, 0.0), (0.54, 0.86), (-0.54, 0.86)])
    return lambda x, y: np.minimum.reduce(
        [sdf.rotated(arm, a)(x, y) for a in (0.0, 90.0, 180.0, 270.0)]
    )


def _tower():
    body = lambda x, y: sdf.box(x, y, 0.56, 0.78, 0.04, 0.0, -0.16)  # noqa: E731
    merlons = [lambda x, y, cx=cx: sdf.box(x, y, 0.11, 0.17, 0.02, cx, 0.78) for cx in (-0.42, 0.0, 0.42)]
    shell = _U(body, *merlons)
    door = _U(
        lambda x, y: sdf.box(x, y, 0.17, 0.26, 0.0, 0.0, -0.68),
        lambda x, y: sdf.circle(x, y, 0.17, 0.0, -0.42),
    )
    window = lambda x, y: sdf.box(x, y, 0.07, 0.13, 0.03, 0.0, 0.22)  # noqa: E731
    return _D(_D(shell, door), window)


def _sword():
    blade = _poly([(-0.13, -0.50), (0.13, -0.50), (0.10, 0.40), (0.0, 0.98), (-0.10, 0.40)])
    guard = lambda x, y: sdf.box(x, y, 0.46, 0.085, 0.04, 0.0, -0.58)  # noqa: E731
    grip = lambda x, y: sdf.box(x, y, 0.095, 0.20, 0.03, 0.0, -0.75)  # noqa: E731
    pommel = lambda x, y: sdf.circle(x, y, 0.13, 0.0, -0.90)  # noqa: E731
    return _fit(_U(blade, guard, grip, pommel))


def _key():
    bow = lambda x, y: sdf.ring(x, y, 0.30, 0.095, 0.0, 0.62)  # noqa: E731
    stem = lambda x, y: sdf.box(x, y, 0.08, 0.60, 0.0, 0.0, -0.05)  # noqa: E731
    bit = _U(
        lambda x, y: sdf.box(x, y, 0.19, 0.08, 0.0, 0.18, -0.48),
        lambda x, y: sdf.box(x, y, 0.14, 0.06, 0.0, 0.13, -0.66),
    )
    return _fit(_U(bow, stem, bit))


def _tree():
    trunk = lambda x, y: sdf.box(x, y, 0.105, 0.34, 0.03, 0.0, -0.62)  # noqa: E731
    canopy = _S(
        0.26,
        lambda x, y: sdf.circle(x, y, 0.50, 0.0, 0.16),
        lambda x, y: sdf.circle(x, y, 0.36, -0.38, -0.12),
        lambda x, y: sdf.circle(x, y, 0.36, 0.38, -0.12),
        lambda x, y: sdf.circle(x, y, 0.32, 0.0, 0.58),
    )
    return _U(trunk, canopy)


def _escallop():
    """Scallop shell: a fan, a hinged nub, and a scalloped rim.

    The fan is what makes a shell a shell -- a circular sector centred on the
    hinge, widened by an ellipse so the flanks bow outward instead of running
    straight to the point.  The wavy rim comes from subtracting small discs
    along the outer edge.
    """
    hinge = (0.0, -0.98)
    lo, hi = 0.098, 0.402  # turns: a fan spanning about 110 degrees
    body = lambda x, y: np.maximum(  # noqa: E731
        _field.wedge(lo, hi, hinge[0], hinge[1])(x, y),
        sdf.ellipse(x, y, 0.98, 0.88, 0.0, -0.12),
    )
    nub = _poly([(-0.30, -0.48), (0.30, -0.48), (0.19, -1.00), (-0.19, -1.00)])
    shell = _U(body, nub)
    notches = []
    for i in range(7):
        t = np.pi * (0.22 + 0.56 * (i + 0.5) / 7.0)
        cx = 0.98 * np.cos(t)
        cy = -0.12 + 0.88 * np.sin(t)
        notches.append(lambda x, y, cx=cx, cy=cy: sdf.circle(x, y, 0.17, cx, cy))
    return _DT(shell, *notches)


def _feather(angle_deg: float, r0: float, r1: float, w0: float, w1: float, cx: float = 0.0, cy: float = 0.0):
    """A tapered quad radiating from ``(cx, cy)`` -- one heraldic feather.

    Building a wing out of feathers rather than one outline is what makes the
    displayed eagle read: the fans are generated from an angle list, so the
    plumage is a parameter rather than a drawing.
    """
    a = np.radians(angle_deg)
    u = np.array([np.cos(a), np.sin(a)])
    n = np.array([-u[1], u[0]])
    base = np.array([cx, cy])
    pts = [
        base + u * r0 + n * w0,
        base + u * r1 + n * w1,
        base + u * r1 - n * w1,
        base + u * r0 - n * w0,
    ]
    return _poly(pts)


def _eagle_displayed():
    """An eagle with spread wings, built symmetric.

    Layout is explicit in the unit box: a narrow body, a raised head, and wings
    that sweep from the shoulders towards the upper corners -- a rotated ellipse
    for the wing's mass, plus short feathers that break its outer edge.  A wing
    made of feathers alone reads as a starburst; a wing made of one blob reads
    as a mushroom.  Both together read as a bird.
    """
    body = lambda x, y: sdf.ellipse(x, y, 0.20, 0.46, 0.0, -0.18)  # noqa: E731
    neck = lambda x, y: sdf.segment(x, y, (0.0, 0.02), (0.0, 0.36), 0.115)  # noqa: E731
    head = lambda x, y: sdf.circle(x, y, 0.185, 0.0, 0.50)  # noqa: E731
    beak = _poly([(-0.12, 0.56), (-0.42, 0.46), (-0.12, 0.38)])
    shoulder = lambda x, y: sdf.circle(x, y, 0.16, 0.13, 0.02)  # noqa: E731
    wing_blob = lambda x, y: sdf.rotated(lambda a, b: sdf.ellipse(a, b, 0.44, 0.21), 44.0)(x - 0.46, y - 0.38)  # noqa: E731
    wing = _U(
        shoulder,
        wing_blob,
        _feather(26.0, 0.52, 0.95, 0.10, 0.040),
        _feather(42.0, 0.50, 0.92, 0.10, 0.040),
        _feather(58.0, 0.46, 0.84, 0.095, 0.040),
    )
    tail = _U(
        _feather(-90.0, 0.20, 0.70, 0.17, 0.13, 0.0, -0.34),
        _feather(-68.0, 0.20, 0.62, 0.16, 0.12, 0.0, -0.34),
        _feather(-112.0, 0.20, 0.62, 0.16, 0.12, 0.0, -0.34),
    )
    leg = lambda x, y: sdf.segment(x, y, (0.14, -0.40), (0.24, -0.74), 0.062)  # noqa: E731
    foot = _poly([(0.10, -0.72), (0.40, -0.82), (0.30, -0.60)])
    return _fit(_U(body, neck, head, beak, tail, wing, _mirror(wing), leg, _mirror(leg), foot, _mirror(foot)))


def _lion_rampant():
    """A rampant lion, assembled as a capsule skeleton.

    Deliberately stylised: a spine, four limbs, a head and a curved tail, welded
    with a smooth union so the joins read as muscle rather than as a joint.  It
    is the one charge here whose silhouette is not a classical outline.
    """
    spine = lambda x, y: sdf.segment(x, y, (0.06, -0.34), (-0.06, 0.30), 0.21)  # noqa: E731
    haunch = lambda x, y: sdf.circle(x, y, 0.24, 0.10, -0.34)  # noqa: E731
    chest = lambda x, y: sdf.circle(x, y, 0.20, -0.06, 0.28)  # noqa: E731
    head = lambda x, y: sdf.circle(x, y, 0.19, -0.22, 0.52)  # noqa: E731
    mane = lambda x, y: sdf.circle(x, y, 0.17, -0.34, 0.68)  # noqa: E731
    muzzle = _poly([(-0.28, 0.60), (-0.50, 0.56), (-0.46, 0.44), (-0.26, 0.44)])
    ear = _poly([(-0.20, 0.72), (-0.10, 0.80), (-0.12, 0.62)])
    front = [
        lambda x, y: sdf.segment(x, y, (-0.08, 0.24), (-0.52, 0.60), 0.075),
        lambda x, y: sdf.segment(x, y, (-0.02, 0.14), (-0.56, 0.34), 0.070),
    ]
    hind = [
        lambda x, y: sdf.segment(x, y, (0.10, -0.36), (-0.04, -0.92), 0.085),
        lambda x, y: sdf.segment(x, y, (0.14, -0.30), (0.52, -0.70), 0.080),
    ]
    tail = [
        lambda x, y: sdf.segment(x, y, (0.16, -0.44), (0.50, -0.16), 0.055),
        lambda x, y: sdf.segment(x, y, (0.50, -0.16), (0.60, 0.34), 0.050),
        lambda x, y: sdf.segment(x, y, (0.60, 0.34), (0.44, 0.66), 0.045),
    ]
    parts = [spine, haunch, chest, head, mane, muzzle, ear] + front + hind + tail
    return _S(0.055, *parts)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
_BASE = [
    Charge("mullet", _star(5), "mullet", "mullets", "звезда", "звёзды", "звёздами", "f"),
    Charge("mullet_of_six", _star(6), "mullet of six points", "mullets of six points", "шестиконечная звезда", "шестиконечные звёзды", "шестиконечными звёздами", "f"),
    Charge("estoile", _estoile(6), "estoile", "estoiles", "волнистая звезда", "волнистые звёзды", "волнистыми звёздами", "f"),
    Charge("roundel", _roundel(), "roundel", "roundels", "круг", "круги", "кругами", "m"),
    Charge("annulet", _annulet(), "annulet", "annulets", "кольцо", "кольца", "кольцами", "n"),
    Charge("lozenge", _lozenge(), "lozenge", "lozenges", "ромб", "ромбы", "ромбами", "m"),
    Charge("fusil", _fusil(), "fusil", "fusils", "веретено", "веретёна", "верётенами", "n"),
    Charge("billet", _billet(), "billet", "billets", "брусок", "бруски", "брусками", "m"),
    Charge("escutcheon", _escutcheon(), "escutcheon", "escutcheons", "щиток", "щитки", "щитками", "m"),
    Charge("crescent", _crescent(), "crescent", "crescents", "полумесяц", "полумесяцы", "полумесяцами", "m"),
    Charge("fleur_de_lis", _fleur_de_lis(), "fleur-de-lis", "fleurs-de-lis", "лилия", "лилии", "лилиями", "f"),
    Charge("heart", _heart(), "heart", "hearts", "сердце", "сердца", "сердцами", "n"),
    Charge("sun", _sun(), "sun in splendour", "suns in splendour", "солнце", "солнца", "солнцами", "n"),
    Charge("cross_crosslet", _cross_crosslet(), "cross crosslet", "crosses crosslet", "крестик", "крестики", "крестиками", "m"),
    Charge("cross_pattee", _cross_pattee(), "cross pattee", "crosses pattee", "лапчатый крест", "лапчатые кресты", "лапчатыми крестами", "m"),
    Charge("tower", _tower(), "tower", "towers", "башня", "башни", "башнями", "f"),
    Charge("sword", _sword(), "sword", "swords", "меч", "мечи", "мечами", "m"),
    Charge("key", _key(), "key", "keys", "ключ", "ключи", "ключами", "m"),
    Charge("tree", _tree(), "tree", "trees", "дерево", "деревья", "деревьями", "n"),
    Charge("escallop", _escallop(), "escallop", "escallops", "раковина", "раковины", "раковинами", "f"),
    Charge("eagle", _eagle_displayed(), "eagle displayed", "eagles displayed", "орёл", "орлы", "орлами", "m"),
    Charge("lion", _lion_rampant(), "lion rampant", "lions rampant", "лев", "львы", "львами", "m", tags=("stylised",)),
]

#: Roundels of a fixed tincture have their own names in blazon.
_ROUNDELS = [
    ("bezant", "or", "bezant", "bezants", "безант", "безанты", "безантами"),
    ("plate", "argent", "plate", "plates", "серебряный круг", "серебряные круги", "серебряными кругами"),
    ("torteau", "gules", "torteau", "torteaux", "торто", "торто", "торто"),
    ("hurt", "azure", "hurt", "hurts", "синий круг", "синие круги", "синими кругами"),
    ("pellet", "sable", "pellet", "pellets", "чёрный круг", "чёрные круги", "чёрными кругами"),
    ("pomme", "vert", "pomme", "pommes", "зелёный круг", "зелёные круги", "зелёными кругами"),
    ("golpe", "purpure", "golpe", "golpes", "пурпурный круг", "пурпурные круги", "пурпурными кругами"),
]

CHARGES = {c.key: c for c in _BASE}
for _key, _tincture, _en, _enpl, _ru, _rupl, _ruin in _ROUNDELS:
    CHARGES[_key] = Charge(_key, _roundel(), _en, _enpl, _ru, _rupl, _ruin, "m", forced=_tincture)

#: Plain-language aliases, for specs written by hand.
_ALIASES = {
    "star": "mullet",
    "star_of_six": "mullet_of_six",
    "circle": "roundel",
    "ring": "annulet",
    "diamond": "lozenge",
    "spindle": "fusil",
    "moon": "crescent",
    "lily": "fleur_de_lis",
    "fleur": "fleur_de_lis",
    "castle": "tower",
    "shell": "escallop",
    "eagle_displayed": "eagle",
    "lion_rampant": "lion",
    "cross": "cross_pattee",
}


def get(name) -> Charge:
    """Look up a charge by name or alias."""
    if isinstance(name, Charge):
        return name
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in CHARGES:
        raise KeyError(f"unknown charge {name!r}; known: {', '.join(sorted(CHARGES))}")
    return CHARGES[key]


#: Instrumental singular, for the Russian "between/with one ..." constructions.
#: Separated from :class:`Charge` because it is only needed by the blazon
#: writer, and a charge is otherwise language-agnostic.
_RU_INSTR_SG = {
    "mullet": "звездой",
    "mullet_of_six": "шестиконечной звездой",
    "estoile": "волнистой звездой",
    "roundel": "кругом",
    "annulet": "кольцом",
    "lozenge": "ромбом",
    "fusil": "веретеном",
    "billet": "бруском",
    "escutcheon": "щитком",
    "crescent": "полумесяцем",
    "fleur_de_lis": "лилией",
    "heart": "сердцем",
    "sun": "солнцем",
    "cross_crosslet": "крестиком",
    "cross_pattee": "лапчатым крестом",
    "tower": "башней",
    "sword": "мечом",
    "key": "ключом",
    "tree": "деревом",
    "escallop": "раковиной",
    "eagle": "орлом",
    "lion": "львом",
    "bezant": "безантом",
    "plate": "серебряным кругом",
    "torteau": "торто",
    "hurt": "синим кругом",
    "pellet": "чёрным кругом",
    "pomme": "зелёным кругом",
    "golpe": "пурпурным кругом",
}


def instr_sg(name) -> str:
    """Russian instrumental singular of a charge's noun.

    Nouns cannot be declined by rule, so the forms are tabulated alongside the
    nominative and instrumental plural the registry already carries.
    """
    return _RU_INSTR_SG.get(get(name).key, get(name).ru_instr)


#: Genitive singular and plural, for numeral government: "два золотых круга"
#: (genitive singular after 2-4) but "пять золотых кругов" (genitive plural
#: after 5 and above).
_RU_GEN = {
    "mullet": ("звезды", "звёзд", None),
    "mullet_of_six": ("звезды", "шестиконечных звёзд", "шестиконечных"),
    "estoile": ("звезды", "волнистых звёзд", "волнистых"),
    "roundel": ("круга", "кругов", None),
    "annulet": ("кольца", "колец", None),
    "lozenge": ("ромба", "ромбов", None),
    "fusil": ("веретена", "верётен", None),
    "billet": ("бруска", "брусков", None),
    "escutcheon": ("щитка", "щитков", None),
    "crescent": ("полумесяца", "полумесяцев", None),
    "fleur_de_lis": ("лилии", "лилий", None),
    "heart": ("сердца", "сердец", None),
    "sun": ("солнца", "солнц", None),
    "cross_crosslet": ("крестика", "крестиков", None),
    "cross_pattee": ("креста", "лапчатых крестов", "лапчатых"),
    "tower": ("башни", "башен", None),
    "sword": ("меча", "мечей", None),
    "key": ("ключа", "ключей", None),
    "tree": ("дерева", "деревьев", None),
    "escallop": ("раковины", "раковин", None),
    "eagle": ("орла", "орлов", None),
    "lion": ("льва", "львов", None),
    "bezant": ("безанта", "безантов", None),
    "plate": ("круга", "серебряных кругов", "серебряных"),
    "torteau": ("торто", "торто", None),
    "hurt": ("круга", "синих кругов", "синих"),
    "pellet": ("круга", "чёрных кругов", "чёрных"),
    "pomme": ("круга", "зелёных кругов", "зелёных"),
    "golpe": ("круга", "пурпурных кругов", "пурпурных"),
}


def gen_forms(name):
    """Russian genitive forms of a charge's noun.

    ``(genitive singular, genitive plural, genitive plural of its own modifier)``.
    The modifier is separated out because Russian numerals 2-4 put the adjective
    in the genitive plural and the noun head in the genitive singular: *три
    лазоревых лапчатых креста*, not "три лазоревых лапчатого креста".  Single-word
    charges carry ``None``.
    """
    ch = get(name)
    return _RU_GEN.get(ch.key, (ch.ru, ch.ru_pl, None))


def names(tag: str | None = None) -> list:
    """List charge keys, optionally filtered by tag."""
    if tag is None:
        return sorted(CHARGES)
    return sorted(k for k, c in CHARGES.items() if tag in c.tags)


def with_tag(tag: str) -> list:
    return names(tag)
