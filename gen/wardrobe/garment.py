"""A flat-lay garment: cloth as an object, not as a bust.

The geometry is built in one world frame roughly one head tall and one and a
quarter heads wide.  A cut changes the *silhouette* first -- the shape of the
body, the length of the sleeve, the fall of the hem -- and only then its details.
That order matters: a coat that is a shirt with more buttons still reads as a
shirt, while a body that widens below the waist reads as a coat before a single
button is painted.

The drawing is four passes over the same cloth surface:

1. a drop shadow and a darker under-layer, so the object sits on a page;
2. the union of body and sleeves, filled and cel-shaded;
3. the tailor's marks -- a neckline, a collar or lapels, a placket, pockets,
   seams and buttons, each in the trim or the metal;
4. the cloth's own grain -- a pattern, a set of folds and a hem/cuff stitch.

The palette is derived from two input colours, ``cloth`` and ``trim``.  The metal
of the buttons is a third, and it is allowed to disagree with the trim: a naval
coat has gold buttons and a black collar.
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.canvas import Canvas

from gen.portraits import brush
from gen.portraits import shape as S

__all__ = [
    "CUTS", "COLLARS", "PATTERNS", "DEFAULTS", "random_spec", "clamp",
    "describe", "render", "body_shape",
]

CUTS = ("shirt", "blouse", "tunic", "jacket", "coat", "dress", "robe", "mantle", "habit")
COLLARS = ("none", "shirt", "notch", "stand", "shawl", "bow", "hood")
PATTERNS = ("none", "stripe", "check", "herringbone", "dot")
DEFAULTS = {
    "cut": "jacket",
    "collar": "notch",
    "pattern": "none",
    "cloth": "#4d5666",
    "trim": "#b99a5c",
    "metal": "#c9a24a",
    "buttons": 2,
    "sleeve": "long",
    "seed": 0,
}

_FITTED = {"shirt", "blouse", "tunic", "jacket"}
_LONG = {"coat", "robe", "habit"}
_NO_SLEEVE = {"mantle"}
_WIDE_SLEEVE = {"robe", "habit"}
_SHORT_SLEEVE = {"shirt", "blouse"}


def _hex(value, fallback: str) -> str:
    try:
        text = str(value)
        palette.hex_to_srgb(text)
        return text
    except (TypeError, ValueError):
        return fallback


def clamp(spec: dict) -> dict:
    """Return a legal copy of ``spec``.  Unknown names are replaced, not repaired."""
    s = dict(DEFAULTS)
    s.update(dict(spec))
    s["cut"] = str(s["cut"]) if str(s["cut"]) in CUTS else DEFAULTS["cut"]
    s["collar"] = str(s["collar"]) if str(s["collar"]) in COLLARS else None
    s["pattern"] = str(s["pattern"]) if str(s["pattern"]) in PATTERNS else DEFAULTS["pattern"]
    s["sleeve"] = str(s["sleeve"]) if str(s["sleeve"]) in ("long", "short", "none") else DEFAULTS["sleeve"]
    if s["collar"] is None:
        s["collar"] = "notch" if s["cut"] in ("jacket", "coat") else (
            "shawl" if s["cut"] == "robe" else ("hood" if s["cut"] == "habit" else "shirt"))
    s["cloth"] = _hex(s["cloth"], DEFAULTS["cloth"])
    s["trim"] = _hex(s["trim"], DEFAULTS["trim"])
    s["metal"] = _hex(s["metal"], DEFAULTS["metal"])
    s["buttons"] = int(np.clip(int(s["buttons"]), 0, 8))
    s["seed"] = int(s["seed"])
    return s


def random_spec(seed: int = 0, cut: str | None = None) -> dict:
    """A seeded garment.  The cut may be pinned to sample one row of the sheet."""
    rng = np.random.default_rng(abs(int(seed)) * 7919 + 13)
    if cut not in CUTS:
        cut = str(rng.choice(CUTS, p=(.10, .08, .14, .20, .16, .10, .10, .06, .06)))
    cloth = palette.oklch_to_hex((float(rng.uniform(.30, .62)), float(rng.uniform(.025, .085)),
                                  float(rng.uniform(0, 1))))
    trim = palette.oklch_to_hex((float(rng.uniform(.52, .78)), float(rng.uniform(.045, .13)),
                                 float(rng.uniform(0, 1))))
    metal = palette.oklch_to_hex((float(rng.uniform(.62, .82)), float(rng.uniform(.08, .14)),
                                  float(rng.uniform(.21, .27))))
    collars = {
        "shirt": ("shirt", "stand", "none"),
        "blouse": ("bow", "shirt", "none"),
        "tunic": ("stand", "shirt", "notch"),
        "jacket": ("notch", "shawl", "stand"),
        "coat": ("notch", "shawl", "stand"),
        "dress": ("none", "bow", "shirt"),
        "robe": ("shawl", "notch", "none"),
        "mantle": ("none", "shawl"),
        "habit": ("hood", "shawl", "none"),
    }
    collar = str(rng.choice(collars[cut]))
    pattern = str(rng.choice(PATTERNS, p=(.68, .10, .08, .08, .06)))
    buttons = int(rng.integers(0, 7))
    if cut in ("shirt", "blouse", "dress", "robe", "mantle", "habit"):
        buttons = int(rng.integers(0, 4))
    sleeve = "none" if cut in _NO_SLEEVE else ("short" if cut in _SHORT_SLEEVE and rng.random() < .35 else "long")
    return clamp({
        "cut": cut,
        "collar": collar,
        "pattern": pattern,
        "cloth": cloth,
        "trim": trim,
        "metal": metal,
        "buttons": buttons,
        "sleeve": sleeve,
        "seed": int(seed),
    })


def _body_points(cut: str):
    if cut == "dress":
        return [(-.30, .60), (-.17, .64), (.17, .64), (.30, .60),
                (.29, .04), (.62, -.78), (.52, -.90), (.0, -.94),
                (-.52, -.90), (-.62, -.78), (-.29, .04)]
    if cut == "mantle":
        return [(-.28, .62), (-.12, .70), (.12, .70), (.28, .62),
                (.52, .34), (.62, -.30), (.42, -.38), (.0, -.42),
                (-.42, -.38), (-.62, -.30), (-.52, .34)]
    if cut in ("coat", "robe", "habit"):
        hem = -.88 if cut in ("robe", "habit") else -.78
        return [(-.29, .60), (-.15, .66), (.15, .66), (.29, .60),
                (.38, .24), (.42, hem), (.18, hem - .02),
                (.0, hem + .02), (-.18, hem - .02), (-.42, hem),
                (-.38, .24)]
    if cut == "tunic":
        return [(-.30, .60), (-.16, .65), (.16, .65), (.30, .60),
                (.35, .18), (.34, -.60), (.20, -.64), (.0, -.62),
                (-.20, -.64), (-.34, -.60), (-.35, .18)]

    return [(-.29, .60), (-.15, .65), (.15, .65), (.29, .60),
            (.34, .22), (.32, -.52), (.18, -.57), (.0, -.55),
            (-.18, -.57), (-.32, -.52), (-.34, .22)]


def body_shape(cut: str) -> S.Shape:
    body = S.poly(_body_points(cut), resample=28)

    neck = S.ellipse(.135, .105, 0.0, .665)
    return body - neck


def _sleeves(cut: str, sleeve: str):
    if sleeve == "none" or cut in _NO_SLEEVE:
        return None
    out = []
    for side in (+1, -1):
        if cut in _WIDE_SLEEVE:
            p = [(side * .26, .58), (side * .44, .54), (side * .72, .12),
                 (side * .82, -.24), (side * .63, -.34), (side * .48, -.06),
                 (side * .38, .24)]
        elif sleeve == "short":
            p = [(side * .27, .58), (side * .43, .54), (side * .55, .28),
                 (side * .45, .20), (side * .34, .36)]
        elif cut == "coat":
            p = [(side * .27, .58), (side * .43, .55), (side * .57, .08),
                 (side * .60, -.28), (side * .51, -.34), (side * .43, -.06),
                 (side * .34, .30)]
        else:
            p = [(side * .27, .58), (side * .43, .55), (side * .57, .16),
                 (side * .55, -.22), (side * .47, -.26), (side * .40, -.02),
                 (side * .34, .30)]
        out.append(S.poly(p, resample=18))
    shape = out[0] | out[1]
    return shape


def _cuff_y(cut: str, sleeve: str) -> float:
    if sleeve == "short":
        return .25
    if cut in _WIDE_SLEEVE:
        return -.28
    if cut == "coat":
        return -.28
    return -.23


def _hood() -> S.Shape:
    cowl = S.ellipse(.40, .28, 0.0, .62)
    hole = S.ellipse(.23, .20, 0.0, .58)
    return cowl - hole


def _collar(cut: str, collar: str):
    """Return ``(shape, colour_key)`` pieces for the neckline."""
    parts = []
    if collar == "shirt":
        for side in (+1, -1):
            parts.append(S.poly([(side * .04, .66), (side * .28, .62),
                                 (side * .20, .44), (side * .04, .58)], resample=10))
    elif collar == "stand":
        parts.append(S.ellipse(.145, .118, 0.0, .67) - S.ellipse(.112, .085, 0.0, .67))
    elif collar in ("notch", "shawl"):
        for side in (+1, -1):
            if collar == "shawl":
                path = [(side * .04, .62), (side * .20, .50), (side * .29, .30),
                        (side * .18, .21), (side * .10, .42)]
                parts.append(S.poly(path, resample=14))
            else:
                parts.append(S.poly([(side * .04, .64), (side * .29, .58),
                                     (side * .21, .28), (side * .10, .43),
                                     (side * .04, .56)], resample=12))
    elif collar == "bow":
        for side in (+1, -1):
            parts.append(S.ellipse(.070, .038, side * .070, .585))
        parts.append(S.circle(.026, 0.0, .585))
    elif collar == "hood":
        parts.append(_hood())
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = out | p
    return out


def _pattern(canvas, spec: dict, clip: S.Shape) -> None:
    pattern = spec["pattern"]
    if pattern == "none":
        return

    if pattern == "stripe":
        for i in range(-11, 12):
            line = S.box(.90, .0075, 0.0, 0.0, 0.0).place(0.0, i * .085, rot=-10.0)
            brush.multiply(canvas, line, .86, clip=clip, alpha=.80)
    elif pattern == "check":
        for i in range(-10, 11):
            a = S.box(.90, .0045, 0.0, 0.0, 0.0).place(0.0, i * .095, rot=0.0)
            b = S.box(.90, .0045, 0.0, 0.0, 0.0).place(0.0, i * .095, rot=90.0)
            brush.multiply(canvas, a, .88, clip=clip, alpha=.75)
            brush.multiply(canvas, b, .88, clip=clip, alpha=.75)
    elif pattern == "herringbone":
        for row in range(-8, 9):
            for k in range(-4, 5):
                x = k * .18 + (0.09 if row % 2 else 0.0)
                y = row * .105
                seg = S.lock([(x - .055, y - .050), (x + .055, y + .050)],
                             w0=.0040, w1=.0030, per_segment=2)
                brush.multiply(canvas, seg, .84, clip=clip, alpha=.80)
    elif pattern == "dot":
        for row in range(-8, 9):
            for k in range(-6, 7):
                x = k * .12 + (0.06 if row % 2 else 0.0)
                y = row * .11
                brush.multiply(canvas, S.circle(.011, x, y), .82, clip=clip, alpha=.85)


def _folds(canvas, spec: dict, clip: S.Shape, seed: int) -> None:
    rng = np.random.default_rng(int(seed) * 131 + 7)
    cloth = palette.hex_to_srgb(spec["cloth"])
    shade = palette.darken(cloth, .16)
    lit = palette.lighten(cloth, .12)
    for side in (+1, -1):
        for k in range(2):
            y0 = .43 - k * .26
            x0 = side * (.14 + k * .07)
            a = (x0, y0)
            b = (x0 + side * (.05 + .02 * float(rng.uniform(-1, 1))), y0 - .22 - .07 * k)
            c = (x0 + side * .035, y0 - .40 - .06 * k)
            fold = S.lock([a, b, c], w0=.006, w1=.002, bulge=.008, per_segment=4)
            brush.paint(canvas, fold, shade, clip=clip, alpha=.19, soft=.65)
            edge = S.lock([(a[0] + side * .007, a[1]), (b[0] + side * .008, b[1]),
                           (c[0] + side * .006, c[1])], w0=.003, w1=.001, per_segment=4)
            brush.paint(canvas, edge, lit, clip=clip, alpha=.13, soft=.55)


def _detail(canvas, spec: dict, body: S.Shape, sleeves, cut: str) -> None:
    trim = palette.hex_to_srgb(spec["trim"])
    metal = palette.hex_to_srgb(spec["metal"])
    trim_dark = palette.darken(trim, .16)
    collar = spec["collar"]

    neck = S.ellipse(.148, .118, 0.0, .665).outline(.008)
    brush.paint(canvas, neck, trim, alpha=.94, soft=.22)
    brush.paint(canvas, S.ellipse(.138, .108, 0.0, .665).outline(.003),
                trim_dark, alpha=.45, soft=.20)

    if cut in ("shirt", "blouse", "tunic"):
        placket = S.box(.016, .50, 0.0, 0.0, .12)
        brush.paint(canvas, placket, trim, clip=body, alpha=.28, soft=.15)
        for side in (+1, -1):
            edge = S.lock([(side * .017, .56), (side * .017, -.37)],
                          w0=.0022, w1=.0020, per_segment=3)
            brush.paint(canvas, edge, trim_dark, clip=body, alpha=.52, soft=.20)
    elif cut in ("jacket", "coat"):

        edge = S.lock([(.02, .55), (.02, .10), (.12, -.55)],
                      w0=.0030, w1=.0022, per_segment=4)
        brush.paint(canvas, edge, trim_dark, clip=body, alpha=.55, soft=.30)
        for k in range(3):
            y = .34 - k * .22
            brush.paint(canvas, S.lock([(.02, y), (.16, y - .015)],
                                       w0=.0018, w1=.0014, per_segment=3),
                        trim, clip=body, alpha=.35, soft=.25)
    elif cut in ("robe", "habit"):
        wrap = S.lock([(-.24, .54), (-.02, .20), (.26, -.08), (.32, -.34)],
                      w0=.006, w1=.003, per_segment=5)
        brush.paint(canvas, wrap, trim, clip=body, alpha=.58, soft=.30)
        brush.paint(canvas, S.lock([(-.23, .55), (-.01, .22), (.27, -.06)],
                                   w0=.0018, w1=.0010, per_segment=4),
                    trim_dark, clip=body, alpha=.35, soft=.20)
    elif cut == "dress":
        waist = S.lock([(-.31, .04), (0.0, .02), (.31, .04)],
                       w0=.004, w1=.003, per_segment=4)
        brush.paint(canvas, waist, trim, clip=body, alpha=.55, soft=.25)

    if collar in ("shirt", "stand", "notch", "shawl", "bow"):
        piece = _collar(cut, collar)
        if piece is not None:
            brush.paint(canvas, piece, trim, alpha=.90, soft=.20)
            brush.paint(canvas, piece.outline(.0020), trim_dark, alpha=.55, soft=.18)
            if collar in ("notch", "shawl"):

                for side in (+1, -1):
                    inner = S.lock([(side * .10, .43), (side * .17, .22)],
                                   w0=.0012, w1=.0008, per_segment=3)
                    brush.paint(canvas, inner, palette.lighten(trim, .10),
                                alpha=.34, soft=.18)

    n = int(spec["buttons"])
    if n:
        xs = (0.13,) if cut in ("shirt", "blouse", "tunic", "dress") else (0.13, 0.24)
        ys = np.linspace(.40, -.42, n)
        for i, y in enumerate(ys):
            for x in xs:
                if not cut in ("shirt", "blouse", "tunic", "dress") and i % 2:
                    continue
                button = S.circle(.018 if cut != "coat" else .012, x, float(y))
                brush.paint(canvas, button, metal, clip=body, alpha=.98, soft=.18)
                brush.paint(canvas, S.circle(.012, x - .004, float(y) + .004),
                            palette.lighten(metal, .18), clip=body, alpha=.55, soft=.25)
                brush.paint(canvas, button.outline(.0018), palette.darken(metal, .25),
                            clip=body, alpha=.50, soft=.15)
                if cut in ("jacket", "coat"):
                    brush.paint(canvas, S.box(.007, .0015, 0, x + .035, float(y)),
                                trim_dark, clip=body, alpha=.48, soft=.15)

    if cut in ("jacket", "coat", "tunic"):
        breast = S.lock([(-.26, .31), (-.13, .32)], w0=.0020, w1=.0018, per_segment=3)
        brush.paint(canvas, breast, trim, clip=body, alpha=.55, soft=.25)
        if cut in ("coat", "jacket"):
            for side in (+1, -1):
                flap = S.lock([(side * .22, -.12), (side * .33, -.13)],
                              w0=.0030, w1=.0024, per_segment=3)
                brush.paint(canvas, flap, trim, clip=body, alpha=.48, soft=.25)
    if cut in ("coat",):
        belt = S.box(.34, .028, .0, 0.0, -.20)
        brush.paint(canvas, belt, trim, clip=body, alpha=.60, soft=.25)
        brush.paint(canvas, belt - belt.offset(-.010), trim_dark, clip=body, alpha=.32, soft=.15)
    if cut in ("robe", "habit"):
        sash = S.box(.40, .020, .0, 0.0, -.05)
        brush.paint(canvas, sash, trim, clip=body, alpha=.72, soft=.25)
        for x in (-.30, -.16, 0.0, .16, .30):
            brush.paint(canvas, S.lock([(x, .02), (x - .02, -.09)], w0=.0018,
                                       w1=.0010, per_segment=3),
                        trim_dark, clip=body, alpha=.30, soft=.15)

    for side in (+1, -1):
        if sleeves is not None and spec["sleeve"] != "none":
            yy = _cuff_y(cut, spec["sleeve"])
            brush.paint(canvas, S.lock([(side * .42, yy + .026), (side * .58, yy + .018)],
                                       w0=.0015, w1=.0012, per_segment=3),
                        trim, alpha=.60, soft=.18)
    hem_y = -.80 if cut in ("coat", "robe", "habit") else (-.48 if cut in _FITTED else -.56)
    if cut == "dress":
        hem_y = -.84
    for side in (+1, -1):
        hem = S.lock([(0.0, hem_y + .016), (side * .26, hem_y + .010)],
                     w0=.0018, w1=.0012, per_segment=3)
        brush.paint(canvas, hem, trim, clip=body, alpha=.50, soft=.20)


def render(spec: dict, size: int = 256) -> Canvas:
    """Draw one garment at ``size`` pixels on a transparent canvas."""
    s = clamp(spec)
    cut = s["cut"]
    size = int(size)
    if size < 24:
        raise ValueError("a garment smaller than 24 px is a swatch, not a garment")
    c = Canvas(size, size, scale=size / 2.0 * .92)
    body = body_shape(cut)
    sleeves = _sleeves(cut, s["sleeve"])
    cloth = palette.hex_to_srgb(s["cloth"])

    hood = _hood() if s["collar"] == "hood" else None
    if hood is not None:
        brush.paint(c, hood.place(0.0, -0.02), palette.darken(cloth, .24),
                    alpha=.96, soft=.25)
        brush.paint(c, hood.place(0.0, -0.02).outline(.003),
                    palette.darken(cloth, .34), alpha=.55, soft=.20)

    shadow = body.place(.018, -.026)
    brush.paint(c, shadow, (0.02, 0.02, 0.03), alpha=.18, soft=size * .012)

    full = body | sleeves if sleeves is not None else body
    surf = brush.Surface(c, full)
    surf.paint(c, cloth, alpha=.995)
    surf.bands(c, [palette.darken(cloth, .22), palette.darken(cloth, .38)],
               (.48, .18), bevel=.045, soft=.8)
    surf.stroke(c, .0028, palette.darken(cloth, .42), inset=.0014, alpha=.75, soft=.20)

    if sleeves is not None:
        for side in (+1, -1):
            shine = S.lock([(side * .07, .61), (side * .31, .57), (side * .45, .46)],
                           w0=.020, w1=.008, per_segment=4)
            brush.paint(c, shine, palette.lighten(cloth, .13), clip=full, alpha=.16, soft=1.1)
    _folds(c, s, full, s["seed"])

    _pattern(c, s, full)
    _detail(c, s, body, sleeves, cut)

    surf.stroke(c, .0020, palette.darken(cloth, .46), inset=.0010, alpha=.55, soft=.18)
    return c


def describe(spec: dict) -> str:
    s = clamp(spec)
    return (f"{s['cut']} · {s['collar']} collar · {s['pattern']} · "
            f"{s['buttons']} buttons · {s['cloth']} / {s['trim']}")
