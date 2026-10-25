"""Standalone decorations: ribbon, medal, cross, star and armband.

A decoration is a small object with a strict hierarchy: the ribbon says *which*
award, the metal says *what grade*, and the motif says *what deed*.  That is also
the order in which it is read, so the painter spends its contrast in that order
and not on a border.

The generator is intentionally separate from :mod:`gen.portraits.regalia`.  The
portrait's medals are painted on a bust at an angle, clipped by the lapel and
partly hidden by the arm; a codex or an inventory needs the object alone, with a
suspension ring and a back.  Sharing the vocabulary is useful; sharing the
clipping would be wrong.

A spec is a dictionary.  Every kind is legal at 32 px, and the sheet draws each
kind before any seeded variant, because a set has to look like a set.
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.canvas import Canvas

from gen.portraits import brush
from gen.portraits import shape as S

__all__ = [
    "KINDS", "METALS", "MOTIFS", "DEFAULTS", "random_spec", "clamp",
    "describe", "render",
]

KINDS = ("medal", "cross", "star", "ribbon", "badge", "armband")
METALS = ("gold", "silver", "bronze")
MOTIFS = ("star", "cross", "laurel", "ring", "torch")

DEFAULTS = {
    "kind": "medal",
    "metal": "gold",
    "ribbon": "#8f2432",
    "ribbon2": "#e7d7ad",
    "motif": "star",
    "bars": 1,
    "rosette": False,
    "seed": 0,
}

_METAL = {

    "gold": ((.73, .125, .235), (.40, .085, .220)),
    "silver": ((.79, .009, .240), (.46, .010, .240)),
    "bronze": ((.56, .090, .165), (.31, .065, .155)),
}


def _hex(value, fallback: str) -> str:
    try:
        palette.hex_to_srgb(str(value))
        return str(value)
    except (TypeError, ValueError):
        return fallback


def clamp(spec: dict) -> dict:
    s = dict(DEFAULTS)
    s.update(dict(spec))
    s["kind"] = str(s["kind"]) if str(s["kind"]) in KINDS else DEFAULTS["kind"]
    s["metal"] = str(s["metal"]) if str(s["metal"]) in METALS else DEFAULTS["metal"]
    s["motif"] = str(s["motif"]) if str(s["motif"]) in MOTIFS else DEFAULTS["motif"]
    s["ribbon"] = _hex(s["ribbon"], DEFAULTS["ribbon"])
    s["ribbon2"] = _hex(s["ribbon2"], DEFAULTS["ribbon2"])
    s["bars"] = int(np.clip(int(s["bars"]), 1, 4))
    s["rosette"] = bool(s["rosette"])
    s["seed"] = int(s["seed"])
    return s


def random_spec(seed: int = 0, kind: str | None = None) -> dict:
    rng = np.random.default_rng(abs(int(seed)) * 3571 + 11)
    if kind not in KINDS:
        kind = str(rng.choice(KINDS, p=(.30, .16, .16, .12, .16, .10)))
    metal = str(rng.choice(METALS, p=(.52, .32, .16)))
    ribbon = palette.oklch_to_hex((float(rng.uniform(.24, .46)), float(rng.uniform(.10, .20)),
                                   float(rng.uniform(0, 1))))
    ribbon2 = palette.oklch_to_hex((float(rng.uniform(.62, .88)), float(rng.uniform(.02, .10)),
                                    float(rng.uniform(0, 1))))
    return clamp({
        "kind": kind,
        "metal": metal,
        "ribbon": ribbon,
        "ribbon2": ribbon2,
        "motif": str(rng.choice(MOTIFS, p=(.38, .25, .15, .12, .10))),
        "bars": int(rng.integers(1, 5)) if kind == "ribbon" else 1,
        "rosette": bool(rng.random() < .35),
        "seed": int(seed),
    })


def _metal(spec: dict):
    light, dark = _METAL[spec["metal"]]
    base = palette.oklch_to_srgb((light[0], light[1], light[2]))
    deep = palette.oklch_to_srgb((dark[0], dark[1], dark[2]))
    return base, palette.lighten(base, .16), deep


def _ribbon_shape(kind: str) -> S.Shape:
    if kind == "ribbon":
        return S.box(.42, .12, .018, 0.0, .58)
    return S.poly([(-.17, .94), (.17, .94), (.12, .53), (-.12, .53)], resample=8)


def _disc(cx: float, cy: float, r: float = .29) -> S.Shape:
    return S.circle(r, cx, cy)


def _motif_shape(motif: str, cx: float, cy: float, r: float) -> S.Shape:
    if motif == "star":
        pts = []
        for i in range(16):
            th = np.pi / 2 + i * np.pi / 8
            rr = r if i % 2 == 0 else r * .39
            pts.append((cx + rr * np.cos(th), cy + rr * np.sin(th)))
        return S.poly(pts, resample=16)
    if motif == "cross":
        return S.poly([(cx - r * .22, cy + r), (cx + r * .22, cy + r),
                       (cx + r * .22, cy + r * .22), (cx + r, cy + r * .22),
                       (cx + r, cy - r * .22), (cx + r * .22, cy - r * .22),
                       (cx + r * .22, cy - r), (cx - r * .22, cy - r),
                       (cx - r * .22, cy - r * .22), (cx - r, cy - r * .22),
                       (cx - r, cy + r * .22), (cx - r * .22, cy + r * .22)],
                      resample=12)
    if motif == "ring":
        return S.circle(r * .82, cx, cy) - S.circle(r * .53, cx, cy)
    if motif == "laurel":

        out = None
        for side in (+1, -1):
            for i in range(5):
                th = -0.85 + i * 0.42
                x = cx + side * (r * .62 + r * .22 * np.cos(th))
                y = cy + r * .55 * np.sin(th) - r * .05
                leaf = S.ellipse(r * .16, r * .075).place(x, y, rot=float(np.degrees(side * (th - 0.35))))
                out = leaf if out is None else out | leaf
        return out

    handle = S.poly([(cx - r * .08, cy - r * .85), (cx + r * .08, cy - r * .85),
                     (cx + r * .05, cy + r * .15), (cx - r * .05, cy + r * .15)],
                    resample=8)
    flame = S.ellipse(r * .20, r * .32, cx, cy + r * .35)
    return handle | flame


def _paint_motif(canvas, spec: dict, shape: S.Shape, metal, metal_hi, metal_lo, alpha=1.0):
    surf = brush.Surface(canvas, shape)
    surf.paint(canvas, metal, alpha=alpha)
    surf.bands(canvas, [metal_lo], (.42,), bevel=.025, soft=.45, alpha=.75 * alpha)
    surf.stroke(canvas, .0028, metal_lo, inset=.0014, alpha=.85 * alpha, soft=.15)


def _ribbon(canvas, spec: dict, shape: S.Shape, vertical: bool = True) -> None:
    a = palette.hex_to_srgb(spec["ribbon"])
    b = palette.hex_to_srgb(spec["ribbon2"])
    surf = brush.Surface(canvas, shape)
    surf.paint(canvas, a, alpha=.995)

    for k in (-2, -1, 1, 2):
        if vertical:
            line = S.box(.022, .22, 0.0, k * .070, .73)
        else:
            line = S.box(.22, .016, 0.0, 0.0, .58 + k * .060)
        brush.paint(canvas, line, b, clip=shape, alpha=.82, soft=.20)

    surf.bands(canvas, [palette.darken(a, .22)], (.44,), bevel=.025, soft=.55)
    surf.stroke(canvas, .0022, palette.darken(a, .38), inset=.0011, alpha=.70, soft=.16)

    for x in (-.12, .12):
        brush.paint(canvas, S.circle(.008, x, .60), b, clip=shape, alpha=.55, soft=.25)


def _rosette(canvas, spec: dict, cx: float, cy: float) -> None:
    a = palette.hex_to_srgb(spec["ribbon"])
    b = palette.hex_to_srgb(spec["ribbon2"])
    for i in range(8):
        th = i * np.pi / 4
        petal = S.ellipse(.075, .035).place(cx + .065 * np.cos(th),
                                            cy + .065 * np.sin(th),
                                            rot=float(np.degrees(th)))
        brush.paint(canvas, petal, a if i % 2 else b, alpha=.95, soft=.30)
    brush.paint(canvas, S.circle(.060, cx, cy), a, alpha=.98, soft=.20)
    brush.paint(canvas, S.circle(.032, cx, cy), b, alpha=.80, soft=.22)


def _medal(canvas, spec: dict):
    metal, hi, lo = _metal(spec)
    ribbon = _ribbon_shape("medal")
    _ribbon(canvas, spec, ribbon)

    ring = S.circle(.060, 0.0, .49) - S.circle(.038, 0.0, .49)
    brush.paint(canvas, ring, metal, alpha=.98, soft=.20)
    brush.paint(canvas, ring.outline(.0018), lo, alpha=.65, soft=.15)
    disc = _disc(0.0, .15, .305)
    surf = brush.Surface(canvas, disc)
    surf.paint(canvas, metal, alpha=.99)
    surf.bands(canvas, [lo], (.44,), bevel=.035, soft=.65, alpha=.72)
    surf.stroke(canvas, .0070, lo, inset=.0030, alpha=.85, soft=.25)

    field = S.circle(.235, 0.0, .15)
    brush.paint(canvas, field, palette.darken(metal, .16), alpha=.72, soft=.30)

    motif = _motif_shape(spec["motif"], 0.0, .15, .185)
    brush.paint(canvas, motif.place(.004, -.008), lo, alpha=.55, soft=.55)
    _paint_motif(canvas, spec, motif, hi, hi, lo, alpha=.99)
    if spec["rosette"]:
        _rosette(canvas, spec, 0.0, .49)

    brush.paint(canvas, S.ellipse(.030, .012).place(-.10, .36, rot=32.0),
                (1.0, 1.0, 1.0), alpha=.50, soft=.55)


def _cross(canvas, spec: dict):
    metal, hi, lo = _metal(spec)
    _ribbon(canvas, spec, _ribbon_shape("cross"))
    ring = S.circle(.045, 0.0, .51) - S.circle(.028, 0.0, .51)
    brush.paint(canvas, ring, metal, alpha=.95, soft=.20)

    arms = []
    for th in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        p = [(0.0, .06), (.115, .10), (.095, .48), (-.095, .48), (-.115, .10)]
        arm = S.poly(p, resample=10).place(0.0, 0.18, rot=float(np.degrees(th)))
        arms.append(arm)
    shape = arms[0]
    for arm in arms[1:]:
        shape = shape | arm
    surf = brush.Surface(canvas, shape)
    surf.paint(canvas, metal, alpha=.99)
    surf.bands(canvas, [lo], (.43,), bevel=.030, soft=.50, alpha=.68)
    surf.stroke(canvas, .0042, lo, inset=.0020, alpha=.85, soft=.18)
    _paint_motif(canvas, spec, _motif_shape("ring", 0.0, .20, .075), hi, hi, lo)
    brush.paint(canvas, S.circle(.036, 0.0, .20), palette.mix(metal, lo, .35), alpha=.75, soft=.20)


def _star(canvas, spec: dict):
    metal, hi, lo = _metal(spec)

    bar = S.box(.40, .055, .014, 0.0, .84)
    _ribbon(canvas, spec, bar, vertical=False)
    pts = []
    for i in range(32):
        th = np.pi / 2 + i * np.pi / 16
        rr = .58 if i % 2 == 0 else .22
        pts.append((rr * np.cos(th), .24 + rr * np.sin(th)))
    shape = S.poly(pts, resample=32)
    surf = brush.Surface(canvas, shape)
    surf.paint(canvas, metal, alpha=.99)
    surf.bands(canvas, [lo], (.45,), bevel=.030, soft=.50, alpha=.70)
    surf.stroke(canvas, .0038, lo, inset=.0018, alpha=.82, soft=.20)

    for th in (np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4):
        brush.paint(canvas, S.lock([(0.0, .24), (.48 * np.cos(th), .24 + .48 * np.sin(th))],
                                   w0=.0022, w1=.0010, per_segment=3),
                    lo, alpha=.38, soft=.18)
    disc = S.circle(.19, 0.0, .24)
    brush.paint(canvas, disc, palette.mix(metal, lo, .24), alpha=.95, soft=.25)
    _paint_motif(canvas, spec, _motif_shape(spec["motif"], 0.0, .24, .145),
                 hi, hi, lo, alpha=.98)
    brush.paint(canvas, disc.outline(.0028), lo, alpha=.60, soft=.20)


def _ribbon_rack(canvas, spec: dict):
    n = int(spec["bars"])
    a = palette.hex_to_srgb(spec["ribbon"])
    b = palette.hex_to_srgb(spec["ribbon2"])
    for i in range(n):
        y = .55 - i * .20
        bar = S.box(.44, .072, .012, 0.0, y)
        surf = brush.Surface(canvas, bar)
        surf.paint(canvas, a, alpha=.99)
        for k in (-2, -1, 1, 2):
            brush.paint(canvas, S.box(.022, .075, 0.0, k * .070, y), b,
                        clip=bar, alpha=.80, soft=.18)
        surf.stroke(canvas, .0022, palette.darken(a, .35), inset=.0011, alpha=.65, soft=.15)
        brush.paint(canvas, S.lock([(-.38, y - .066), (.38, y - .066)],
                                   w0=.0016, w1=.0016, per_segment=3),
                    palette.darken(a, .40), alpha=.55, soft=.15)

    brush.paint(canvas, S.circle(.034, -.39, .55), _metal(spec)[0], alpha=.95, soft=.20)


def _badge(canvas, spec: dict):
    metal, hi, lo = _metal(spec)
    a = palette.hex_to_srgb(spec["ribbon"])
    shape = S.poly([(-.38, .78), (.38, .78), (.44, .22), (.0, -.72),
                    (-.44, .22)], resample=12)
    surf = brush.Surface(canvas, shape)
    surf.paint(canvas, a, alpha=.99)
    surf.bands(canvas, [palette.darken(a, .24)], (.44,), bevel=.030, soft=.55)
    surf.stroke(canvas, .0060, lo, inset=.0028, alpha=.95, soft=.22)
    field = S.poly([(-.31, .68), (.31, .68), (.36, .24), (0.0, -.57),
                    (-.36, .24)], resample=10)
    brush.paint(canvas, field, palette.mix(a, metal, .16), alpha=.55, soft=.30)
    _paint_motif(canvas, spec, _motif_shape(spec["motif"], 0.0, .13, .25),
                 hi, hi, lo, alpha=.98)
    if spec["rosette"]:
        _rosette(canvas, spec, 0.0, .70)


def _armband(canvas, spec: dict):
    a = palette.hex_to_srgb(spec["ribbon"])
    b = palette.hex_to_srgb(spec["ribbon2"])
    metal, hi, lo = _metal(spec)
    band = S.poly([(-.46, .78), (.46, .78), (.40, -.70), (-.40, -.70)], resample=8)
    surf = brush.Surface(canvas, band)
    surf.paint(canvas, a, alpha=.99)
    surf.bands(canvas, [palette.darken(a, .24)], (.44,), bevel=.040, soft=.65)
    surf.stroke(canvas, .0050, palette.darken(a, .42), inset=.0022, alpha=.80, soft=.20)

    for i in range(2):
        y = .38 - i * .24
        for side in (+1, -1):
            brush.paint(canvas, S.lock([(0.0, y), (side * .24, y - .18)],
                                       w0=.030, w1=.014, per_segment=3),
                        b, alpha=.92, soft=.35)

    disc = S.circle(.13, 0.0, -.48)
    brush.paint(canvas, disc, metal, alpha=.96, soft=.25)
    brush.paint(canvas, disc.outline(.0028), lo, alpha=.65, soft=.20)
    _paint_motif(canvas, spec, _motif_shape(spec["motif"], 0.0, -.48, .095),
                 hi, hi, lo, alpha=.95)


def render(spec: dict, size: int = 256) -> Canvas:
    s = clamp(spec)
    size = int(size)
    if size < 24:
        raise ValueError("a decoration smaller than 24 px is a pixel, not an award")
    c = Canvas(size, size, scale=size / 2.0 * .94)

    if s["kind"] == "medal" or s["kind"] == "cross":
        probe = _disc(0.0, .15, .33)
    elif s["kind"] == "star":
        probe = S.circle(.58, 0.0, .24)
    elif s["kind"] == "armband":
        probe = S.box(.46, .74, 0.0, 0.0, .04)
    elif s["kind"] == "ribbon":

        probe = S.box(.46, .080 + .10 * (int(s["bars"]) - 1), .012,
                      0.0, .55 - .10 * (int(s["bars"]) - 1))
    else:
        probe = S.box(.46, .42, .0, 0.0, .10)
    brush.paint(c, probe.place(.018, -.026), (0.02, 0.02, 0.03), alpha=.16, soft=size * .012)
    if s["kind"] == "medal":
        _medal(c, s)
    elif s["kind"] == "cross":
        _cross(c, s)
    elif s["kind"] == "star":
        _star(c, s)
    elif s["kind"] == "ribbon":
        _ribbon_rack(c, s)
    elif s["kind"] == "badge":
        _badge(c, s)
    elif s["kind"] == "armband":
        _armband(c, s)
    return c


def describe(spec: dict) -> str:
    s = clamp(spec)
    return (f"{s['kind']} · {s['metal']} · {s['ribbon']} / {s['ribbon2']} · "
            f"{s['motif']}" + (" · rosette" if s["rosette"] else ""))
