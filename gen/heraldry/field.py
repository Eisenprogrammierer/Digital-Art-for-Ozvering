"""Divisions and variations of the field.

The central idea of this module: **the field is a partition of the plane into
regions, and a region is a signed distance field**.  Every heraldic division --
per pale, quarterly, gyronny, barry, checky -- is then just a list of regions
paired with tincture indices, and painting is a loop.  Two consequences fall out
of that representation for free:

* anti-aliasing comes from :func:`core.render.coverage`, because a region *is*
  a distance field, and adjacent regions meet exactly where both are zero;
* "what colour is under this point?" becomes a query (:func:`tincture_at`),
  which is what lets :mod:`gen.heraldry.blazon` enforce the rule of tincture at
  the real anchor points of real charges instead of guessing.

A region's field is negative inside, positive outside, matching :mod:`core.sdf`.
Regions are ordered so that the classical tincture cycle applies: alternating
tinctures over the region list reproduces the traditional colouring (for
quarterly the two tinctures must appear as A, B, B, A, so the region list is
ordered dexter chief, sinister chief, sinister base, dexter base -- the bigram
order in which a two-cycle *is* the traditional quarterly).
"""
from __future__ import annotations

import numpy as np

from core import sdf

from . import tinctures

Region = "Callable[[ndarray, ndarray], ndarray]"
Layer = "tuple[Region, tinctures.Tincture]"

__all__ = [
    "halfplane",
    "slab",
    "wedge",
    "intersect",
    "union_of",
    "complement",
    "whole_plane",
    "DIVISIONS",
    "VARIATIONS",
    "BUILDERS",
    "layers",
    "field_layers",
    "tincture_at",
    "sample_points",
]


# --------------------------------------------------------------------------
# region primitives
# --------------------------------------------------------------------------
def halfplane(a: float, b: float, c: float = 0.0):
    """Region ``{a*x + b*y + c > 0}`` as an exact SDF (negative inside)."""
    norm = float(np.hypot(a, b)) or 1.0
    return lambda x, y: -(a * x + b * y + c) / norm


def slab(a: float, b: float, lo: float, hi: float):
    """Band between two parallel lines of the unit-normal coordinate ``(a, b)``.

    Exact: for ``a**2 + b**2 == 1`` the returned value is a true distance, so the
    two edges of every stripe anti-alias correctly.
    """
    return lambda x, y: np.maximum(lo - (a * x + b * y), (a * x + b * y) - hi)


def wedge(angle_lo: float, angle_hi: float, cx: float = 0.0, cy: float = 0.0):
    """Wedge between two rays, angles in turns, spanning less than 180 degrees."""
    a0 = 2.0 * np.pi * angle_lo
    a1 = 2.0 * np.pi * angle_hi
    n0 = (np.cos(a0 + np.pi / 2), np.sin(a0 + np.pi / 2))
    n1 = (np.cos(a1 - np.pi / 2), np.sin(a1 - np.pi / 2))
    return intersect(
        lambda x, y: -(n0[0] * (x - cx) + n0[1] * (y - cy)),
        lambda x, y: -(n1[0] * (x - cx) + n1[1] * (y - cy)),
    )


def intersect(*regions):
    """Intersection of regions (``max`` of their fields)."""
    if len(regions) == 1:
        return regions[0]
    return lambda x, y: np.maximum.reduce([r(x, y) for r in regions])


def union_of(*regions):
    """Union of regions (``min`` of their fields)."""
    if len(regions) == 1:
        return regions[0]
    return lambda x, y: np.minimum.reduce([r(x, y) for r in regions])


def complement(region):
    """Set complement: the field is negated, which flips inside/outside.

    The result is a *pseudo*-distance (the magnitude is only right near the
    boundary), which is exactly what coverage and flat painting need.
    """
    return lambda x, y: -region(x, y)


def whole_plane():
    """A region containing everything -- a base coat under an overlay."""
    return lambda x, y: -np.ones(np.broadcast_shapes(np.shape(x), np.shape(y))) - 1e3


# --------------------------------------------------------------------------
# helpers for building families of regions
# --------------------------------------------------------------------------
def _corners(hw: float, hh: float):
    return (
        np.array([-hw, hw, hw, -hw]),
        np.array([hh, hh, -hh, -hh]),
    )


def _bands(u, n: int, hw: float, hh: float):
    """``n`` equal bands of a unit-gradient coordinate ``u``, filling the field.

    The range is measured over a grid spanning the field rather than hard-coded,
    so every variation stretches to whatever box the shield or flag provides.  A
    grid, not just the four corners: a chevron coordinate peaks *inside* the box,
    and measuring only the corners would leave the last band short and punch a
    hole in the field.
    """
    xs = np.linspace(-hw, hw, 33)
    ys = np.linspace(-hh, hh, 33)
    gx, gy = np.meshgrid(xs, ys)
    vals = np.asarray(u(gx, gy), dtype=np.float64)
    edges = np.linspace(float(vals.min()), float(vals.max()), n + 1)
    out = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        out.append(lambda x, y, lo=lo, hi=hi: np.maximum(lo - u(x, y), u(x, y) - hi))
    return out


def _linear(a: float, b: float):
    """Unit-normal linear coordinate ``a*x + b*y`` (so ``|grad| == 1``)."""
    norm = float(np.hypot(a, b)) or 1.0
    return lambda x, y: (a * x + b * y) / norm


def _chevron_coord(hw: float, hh: float, k: float):
    """Chevron coordinate: level sets are ``^`` shapes rather than lines."""
    norm = float(np.hypot(1.0, k))
    return lambda x, y: (y + k * np.abs(x)) / norm


# --------------------------------------------------------------------------
# divisions: partitions of the field
# --------------------------------------------------------------------------
def _per_pale(hw, hh):
    return [(halfplane(-1.0, 0.0), 0), (halfplane(1.0, 0.0), 1)]


def _per_fess(hw, hh):
    return [(halfplane(0.0, 1.0), 0), (halfplane(0.0, -1.0), 1)]


def _per_bend(hw, hh):
    # Line through the dexter chief and sinister base corners; region 0 is the
    # half towards the chief.
    a, b = hh, hw
    return [(halfplane(a, b), 0), (halfplane(-a, -b), 1)]


def _per_bend_sinister(hw, hh):
    a, b = -hh, hw
    return [(halfplane(a, b), 0), (halfplane(-a, -b), 1)]


def _chevron_side(hw, hh, apex_y, to_base):
    """Region bounded by the two chevron arms."""
    end_y = -hh if to_base else hh
    m = (end_y - apex_y) / hw
    if to_base:
        # Above both arms.
        return union_of(halfplane(m, 1.0, -apex_y), halfplane(-m, 1.0, -apex_y))
    # Below both arms.
    return union_of(halfplane(-m, -1.0, apex_y), halfplane(m, -1.0, apex_y))


def _per_chevron(hw, hh):
    upper = _chevron_side(hw, hh, 0.10 * hh, to_base=True)
    return [(upper, 0), (complement(upper), 1)]


def _per_chevron_inverted(hw, hh):
    lower = _chevron_side(hw, hh, -0.10 * hh, to_base=False)
    return [(complement(lower), 0), (lower, 1)]


def _per_saltire(hw, hh):
    # Both diagonals cut the field into four triangles: chief, dexter, base,
    # sinister.  Ordered so that a two-tincture field alternates chief/sinister
    # with dexter/base, which is how "per saltire A and B" is read.
    chief = intersect(halfplane(-1.0, 1.0), halfplane(1.0, 1.0))
    dexter = intersect(halfplane(-1.0, -1.0), halfplane(-1.0, 1.0))
    base = intersect(halfplane(-1.0, -1.0), halfplane(1.0, -1.0))
    sinister = intersect(halfplane(1.0, 1.0), halfplane(1.0, -1.0))
    return [(chief, 0), (dexter, 1), (base, 0), (sinister, 1)]


def _quarterly(hw, hh):
    # dexter chief, sinister chief, sinister base, dexter base: A, B, A, B.
    return [
        (intersect(halfplane(-1.0, 0.0), halfplane(0.0, 1.0)), 0),
        (intersect(halfplane(1.0, 0.0), halfplane(0.0, 1.0)), 1),
        (intersect(halfplane(1.0, 0.0), halfplane(0.0, -1.0)), 0),
        (intersect(halfplane(-1.0, 0.0), halfplane(0.0, -1.0)), 1),
    ]


def _per_pall(hw, hh):
    # A Y: arms to the chief corners, stem down the centre.  Regions are the
    # dexter and sinister wedges and the chief wedge between the arms.
    apex_y = 0.10 * hh
    m = (hh - apex_y) / hw
    # Below the dexter arm on the left, below the sinister arm on the right, and
    # between the two arms above.  The slope sign must follow the side, or the
    # wedges do not meet and the field has holes.
    dexter = intersect(halfplane(-1.0, 0.0), halfplane(-m, -1.0, apex_y))
    sinister = intersect(halfplane(1.0, 0.0), halfplane(m, -1.0, apex_y))
    chief = intersect(halfplane(-m, 1.0, -apex_y), halfplane(m, 1.0, -apex_y))
    return [(dexter, 0), (sinister, 1), (chief, 2)]


def _tierced_in_pale(hw, hh):
    return [(r, i) for i, r in enumerate(_bands(_linear(1.0, 0.0), 3, hw, hh))]


def _tierced_in_fess(hw, hh):
    # Chief first, like per_fess: region 0 is the chief band.
    return [(r, i) for i, r in enumerate(_bands(_linear(0.0, -1.0), 3, hw, hh))]


DIVISIONS = {
    "per_pale": _per_pale,
    "per_fess": _per_fess,
    "per_bend": _per_bend,
    "per_bend_sinister": _per_bend_sinister,
    "per_chevron": _per_chevron,
    "per_chevron_inverted": _per_chevron_inverted,
    "per_saltire": _per_saltire,
    "quarterly": _quarterly,
    "per_pall": _per_pall,
    "tierced_in_pale": _tierced_in_pale,
    "tierced_in_fess": _tierced_in_fess,
}


# --------------------------------------------------------------------------
# variations: repeating patterns
# --------------------------------------------------------------------------
def _bars(hw, hh, n, u):
    return [(r, i % 2) for i, r in enumerate(_bands(u, n, hw, hh))]


def _barry(hw, hh, n=6):
    # u = -y so stripe 1 is at the chief, where blazon counting starts.
    return _bars(hw, hh, n, _linear(0.0, -1.0))


def _paly(hw, hh, n=6):
    return _bars(hw, hh, n, _linear(1.0, 0.0))


def _bendy(hw, hh, n=6):
    return _bars(hw, hh, n, _linear(hh, hw))


def _bendy_sinister(hw, hh, n=6):
    return _bars(hw, hh, n, _linear(-hh, hw))


def _chevronny(hw, hh, n=6):
    k = hh / hw
    return _bars(hw, hh, n, _chevron_coord(hw, hh, k))


def _checky(hw, hh, n=6, m=None):
    m = m or n
    cols = _bands(_linear(1.0, 0.0), n, hw, hh)
    rows = _bands(_linear(0.0, 1.0), m, hw, hh)
    out = []
    for j in range(m):
        for i in range(n):
            out.append((intersect(cols[i], rows[j]), (i + j) % 2))
    return out


def _gyronny(hw, hh, n=8):
    out = []
    step = 1.0 / n
    for i in range(n):
        out.append((wedge(i * step, (i + 1) * step), i % 2))
    return out


def _lozengy(hw, hh, n=6):
    """Lozenge lattice: a base coat with a second lattice on top.

    A lozenge lattice tiles the plane, so this really is a two-tincture field
    rather than an overlay that leaves gaps.
    """
    period = 2.0 * hw / n
    half_w, half_h = period * 0.5, period * 0.5 * (hh / hw)
    diamond = lambda x, y: sdf.polygon(  # noqa: E731
        x / half_w, y / half_h, [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
    ) * min(half_w, half_h)
    lattice = sdf.repeated(diamond, period, 2.0 * half_h)
    return [(whole_plane(), 0), (lambda x, y: lattice(x, y), 1)]


def _ermine_spot(x, y):
    """The ermine spot: a small arrowhead with three drops above it.

    Flores de lis-ish at any size, which is what the fur reads as in a shield.
    """
    head = sdf.polygon(x, y, [(-0.30, -0.45), (0.0, 0.30), (0.30, -0.45)])
    drops = [
        sdf.circle(x, y, 0.15, -0.36, 0.30),
        sdf.circle(x, y, 0.15, 0.0, 0.48),
        sdf.circle(x, y, 0.15, 0.36, 0.30),
    ]
    return sdf.union(head, np.minimum.reduce(drops))


def _ermine_lattice(hw, hh, period_x, period_y, scale):
    spot = lambda x, y: _ermine_spot(x / scale, y / scale) * scale  # noqa: E731
    row_a = sdf.repeated(spot, period_x, 2.0 * period_y)
    row_b = sdf.moved(row_a, period_x * 0.5, period_y)
    return lambda x, y: np.minimum(row_a(x, y), row_b(x, y))


def _fur_ermine(hw, hh):
    """Ermine: a sparse, offset lattice of spots over the fur's base tincture."""
    scale = 0.085 * min(hw, hh)
    lat = _ermine_lattice(hw, hh, 2.0 * hw / 4.5, 2.0 * hh / 5.5, scale)
    return [(whole_plane(), 0), (lat, 1)]


def _bell(x, y):
    """Vair bell: wide top, rounded bottom, fitted to a unit box."""
    top = sdf.box(x, y, 0.5, 0.34, 0.02, 0.0, 0.16)
    bottom = sdf.circle(x, y, 0.5, 0.0, -0.18)
    bell = sdf.union(top, bottom)
    return sdf.intersection(bell, sdf.box(x, y, 0.5, 0.55, 0.0))


def _potent(x, y):
    """Potent: a T, the tile of the fur of the same name."""
    return sdf.union(
        sdf.box(x, y, 0.5, 0.16, 0.0, 0.0, 0.34),
        sdf.box(x, y, 0.16, 0.5, 0.0, 0.0, -0.16),
    )


def _fur_tiles(hw, hh, shape):
    """Interlocking tiles: upright shapes, and their inversions in the gaps."""
    scale = 0.25 * min(hw, hh)
    tile = lambda x, y: shape(x / scale, y / scale) * scale  # noqa: E731
    period = 2.0 * scale
    upright = sdf.repeated(tile, period, period)
    inverted = sdf.moved(sdf.repeated(sdf.scaled(tile, 1.0, -1.0), period, period), scale, scale)
    return upright, inverted


def _fur_vair(hw, hh, shape):
    """Base coat, inverted tiles, then upright tiles -- tiles interlock."""
    upright, inverted = _fur_tiles(hw, hh, shape)
    return [
        (whole_plane(), 0),
        (lambda x, y: inverted(x, y), 1),
        (lambda x, y: upright(x, y), 0),
    ]


def _ermine(hw, hh):
    return _fur_ermine(hw, hh)


def _counter_ermine(hw, hh):
    # Same geometry; counter-ermine differs only in which tinctures the fur maps
    # base and spot to, and that lives in the tincture table.
    return _fur_ermine(hw, hh)


def _vair(hw, hh):
    return _fur_vair(hw, hh, _bell)


def _counter_vair(hw, hh):
    return _fur_vair(hw, hh, _bell)


def _potent_fur(hw, hh):
    return _fur_vair(hw, hh, _potent)


VARIATIONS = {
    "barry": _barry,
    "paly": _paly,
    "bendy": _bendy,
    "bendy_sinister": _bendy_sinister,
    "chevronny": _chevronny,
    "checky": _checky,
    "gyronny": _gyronny,
    "lozengy": _lozengy,
    "ermine": _ermine,
    "counter_ermine": _counter_ermine,
    "vair": _vair,
    "counter_vair": _counter_vair,
    "potent": _potent_fur,
}

#: Named under both keys so a fur can be used as a whole field or as a variation.
BUILDERS = {}
BUILDERS.update(DIVISIONS)
BUILDERS.update(VARIATIONS)


# --------------------------------------------------------------------------
# assembling layers
# --------------------------------------------------------------------------
def layers(name: str, palette_names, hw: float, hh: float, **kwargs) -> list:
    """Build a paint list for a named division or variation.

    ``palette_names`` is the list of tinctures from the blazon; region indices
    are taken modulo its length, so "per pale Or and Azure" and "barry of six
    Or and Azure" use the same call.
    """
    if name not in BUILDERS:
        raise KeyError(f"unknown field division or variation {name!r}; known: {', '.join(sorted(BUILDERS))}")
    pairs = BUILDERS[name](hw, hh, **kwargs)
    ts = [tinctures.get(t) for t in palette_names]
    if not ts:
        raise ValueError("a field needs at least one tincture")
    return [(region, ts[i % len(ts)]) for region, i in pairs]


def _plain_layers(tincture, hw: float, hh: float) -> list:
    """A flat field -- or, for a fur, the fur's own two-tincture pattern."""
    t = tinctures.get(tincture)
    if t.kind == tinctures.FUR and t.key in VARIATIONS:
        base_key, spot_key = t.fur
        pairs = VARIATIONS[t.key](hw, hh)
        base, spot = tinctures.get(base_key), tinctures.get(spot_key)
        return [(region, base if idx == 0 else spot) for region, idx in pairs]
    return [(whole_plane(), t)]


def field_layers(spec: dict, hw: float, hh: float) -> list:
    """Turn a field specification into a paint list.

    Accepted forms::

        {"kind": "plain",     "tinctures": ["azure"]}
        {"kind": "division",  "name": "per_pale", "tinctures": ["or", "azure"]}
        {"kind": "variation", "name": "barry", "n": 6, "tinctures": ["or", "azure"]}
    """
    kind = spec.get("kind", "plain")
    if kind == "plain":
        names = spec.get("tinctures") or [spec["tincture"]]
        if len(names) == 1:
            return _plain_layers(names[0], hw, hh)
        return [(whole_plane(), tinctures.get(names[0]))]
    # A fur used as a variation takes its tinctures from the fur itself.
    if spec["name"] in tinctures.TINCTURES and tinctures.get(spec["name"]).kind == tinctures.FUR:
        return _plain_layers(spec["name"], hw, hh)
    kwargs = {k: v for k, v in spec.items() if k not in ("kind", "name", "tinctures")}
    if "n" in kwargs:
        kwargs["n"] = int(kwargs["n"])
    return layers(spec["name"], spec["tinctures"], hw, hh, **kwargs)


def tincture_at(paint_layers, x: float, y: float):
    """Which tincture covers the point ``(x, y)``?  ``None`` if nothing does.

    This is the service that makes the rule of tincture checkable on real
    geometry: ask what lies beneath a charge's anchor point and compare classes.
    """
    found = None
    for region, tincture in paint_layers:
        if float(np.asarray(region(x, y))) < 0.0:
            found = tincture
    return found


def sample_points(x: float, y: float, radius: float, n: int = 9, turns: int = 2):
    """A small point cloud around ``(x, y)``, for footprint checks."""
    out = [(x, y)]
    for k in range(n):
        a = 2.0 * np.pi * k / n
        out.append((x + radius * np.cos(a), y + radius * np.sin(a)))
    return out[: n + 1]
