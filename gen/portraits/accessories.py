"""Accessories: the things a face can *wear* without changing the face.

The portrait's genes are a person, but a passport photograph is also a record of
how that person was dressed on the day.  Glasses, a monocle and a scar are the
three marks that belong to the face itself rather than to the coat, so they live
in their own appended category layer instead of being folded into ``regalia``.

The layer is deliberately small and categorical:

* ``eyewear`` chooses a frame family -- spectacles, round glasses, a monocle, a
  pince-nez or dark glasses -- and ``eyewear_side`` chooses which eye a monocle
  occupies;
* ``scar`` chooses where an old scar sits, and ``scar_side`` chooses its side;
* both default to ``none`` / ``auto``, so a genome saved before this module
  existed still decodes to exactly the same bare face.

Two rules keep the marks from becoming stickers:

* **Geometry is read from the eyes, not authored beside them.**  A lens is built
  around the opening the eye pass already placed, so it follows a turn and the
  sitter's own asymmetry.  A scar is a lock on the skull, clipped to the skull,
  and it never moves a feature.
* **The face seed does not contain the accessory.**  Glasses are removable and a
  scar is already healed; neither is allowed to rebuild the sitter's cheekbones.
  The side is derived from the *face* seed only when the gene says ``auto``.

The frame is painted after the hair, because a temple arm lies over the hair at
the side of the head.  The lens is only a tint and a rim: the eye must still be
the strongest feature in the picture.
"""
from __future__ import annotations

import numpy as np

from core import palette

from . import brush, eyes, head
from . import shape as S

__all__ = [
    "EYEWEAR", "SIDES", "SCAR", "CATEGORIES", "DEFAULTS",
    "random_values", "validate", "paint_scars", "paint_eyewear",
]

EYEWEAR = ("none", "spectacles", "round", "monocle", "pince_nez", "sunglasses")

SIDES = ("auto", "left", "right")
SCAR = ("none", "brow", "cheek", "lip", "chin", "multiple")

CATEGORIES = {
    "eyewear": EYEWEAR,
    "eyewear_side": SIDES,
    "scar": SCAR,
    "scar_side": SIDES,
}

DEFAULTS = {
    "eyewear": "none",
    "eyewear_side": "auto",
    "scar": "none",
    "scar_side": "auto",
}


def _get(genes: dict, key: str) -> str:
    value = str(genes.get(key, DEFAULTS[key]))
    return value if value in CATEGORIES[key] else DEFAULTS[key]


def _side(genes: dict, key: str, asym) -> int:
    """Resolve ``left``/``right``/``auto`` to the package's side convention.

    ``+1`` is the character's left and therefore the viewer's right, exactly as
    in :mod:`eyes` and :mod:`humanize`.  ``auto`` follows the sitter's lean, so
    the monocle and the scar belong to the same side of the same person.
    """
    value = _get(genes, key)
    if value == "left":
        return +1
    if value == "right":
        return -1
    return +1 if getattr(asym, "lean", 0.0) >= 0.0 else -1


def random_values(rng: np.random.Generator) -> dict:
    """Draw the accessory layer.  Most people have none; the rest are rare.

    The probabilities are for *new random genomes*.  A saved genome may ask for a
    monocle and a scar on the same cheek, and the painter will honour it; the
    random draw only keeps a contact sheet from turning into a costume party.
    """
    eyewear = str(rng.choice(EYEWEAR, p=(0.76, 0.13, 0.08, 0.010, 0.010, 0.010)))
    scar = str(rng.choice(SCAR, p=(0.86, 0.035, 0.050, 0.020, 0.015, 0.020)))
    return {
        "eyewear": eyewear,
        "eyewear_side": str(rng.choice(SIDES, p=(0.80, 0.10, 0.10))),
        "scar": scar,
        "scar_side": str(rng.choice(SIDES, p=(0.80, 0.10, 0.10))),
    }


def validate(genes: dict) -> None:
    for key, choices in CATEGORIES.items():
        value = str(genes.get(key, DEFAULTS[key]))
        if value not in choices:
            raise ValueError(f"unknown {key} {value!r}")


def _eye_centre(st, side: int, turn: float, asym):
    cx, _scale = eyes.placement(st, side, turn, asym)
    e = asym.eye[side] if asym is not None else {"dy": 0.0}
    return float(cx), float(st.eye_y) + float(e["dy"])


def _lens_box(st, side: int, turn: float, asym, style: str):
    """A lens around one opening: centre, half-width and half-height."""
    cx, cy = _eye_centre(st, side, turn, asym)
    hw = float(st.eye_w) * (0.60 if style != "monocle" else 0.62)

    hh = float(st.eye_w) * (0.36 if style not in ("round", "monocle") else 0.48)

    cy += float(st.eye_up - st.eye_down) * 0.10
    return cx, cy, hw, hh


def _lens(cx: float, cy: float, hw: float, hh: float, style: str) -> S.Shape:
    if style == "round" or style == "monocle":
        return S.ellipse(hw, hh, cx, cy)
    if style == "pince_nez":
        return S.ellipse(hw * 0.88, hh * 0.82, cx, cy)

    return S.box(hw, hh, hw * 0.46, cx, cy)


def _frame_colour(h):
    """A restrained metal or dark acetate, derived from the fit's own metal."""

    return palette.mix((0.070, 0.064, 0.060), h.metal, 0.42)


def _temple_anchor(st, side: int, turn: float):
    """Where a temple arm meets the skull, just in front of the ear."""
    x, y = head.turn_point(side * float(st.head_w) * 0.94,
                           float(st.eye_y) + 0.020, turn, st)
    return float(x), float(y)


def _scar_paths(st, kind: str, side: int, turn: float):
    """The healed line(s) for one scar category, in world coordinates.

    A scar is not a random scratch: it has a direction and a story.  The brow
    scar runs through the brow into the forehead, the cheek scar follows the
    zygomatic slope, the lip scar crosses the vermilion, and the chin scar is a
    short vertical nick.  ``multiple`` is a brow and a cheek scar, not a pattern
    of unrelated marks.
    """
    paths = []
    if kind in ("brow", "multiple"):
        a = (side * float(st.brow_dx) * 0.88, float(st.brow_y) + 0.042)
        b = (side * float(st.brow_dx) * 0.78, float(st.brow_y) + 0.100)
        c = (side * float(st.brow_dx) * 0.63, float(st.brow_y) + 0.128)
        paths.append((a, b, c))
    if kind in ("cheek", "multiple"):
        a = (side * float(st.head_w) * 0.33, float(st.eye_y) - 0.070)
        b = (side * float(st.head_w) * 0.43, float(st.eye_y) - 0.145)
        c = (side * float(st.head_w) * 0.37, float(st.eye_y) - 0.230)
        paths.append((a, b, c))
    if kind == "lip":
        x = side * float(st.mouth_w) * 0.26
        paths.append(((x, float(st.mouth_y) + float(st.mouth_w) * 0.15),
                      (x + side * 0.008, float(st.mouth_y)),
                      (x + side * 0.003, float(st.mouth_y)
                       - float(st.mouth_w) * 0.16)))
    if kind == "chin":
        x = side * float(st.chin_w) * 0.32
        paths.append(((x, float(st.chin_y) + 0.130),
                      (x + side * 0.007, float(st.chin_y) + 0.082),
                      (x + side * 0.002, float(st.chin_y) + 0.046)))
    out = []
    for path in paths:
        p = np.asarray([head.turn_point(px, py, turn, st) for px, py in path],
                       dtype=np.float64)
        out.append(p)
    return out


def paint_scars(canvas, st, h, genes: dict, turn: float = 0.0, clip=None,
                asym=None) -> None:
    """Paint the sitter's old scars over the face, before the hair.

    The line has two values: a broad, slightly cool shadow where the tissue is
    thin, and a narrow pale ridge where the healed skin catches the light.  That
    pair is what makes a line read as a scar rather than a pen mark.  Everything
    is clipped to the skull; a scar cannot run onto the background or the ear.
    """
    kind = _get(genes, "scar")
    if kind == "none" or clip is None:
        return
    side = _side(genes, "scar_side", asym)
    paths = _scar_paths(st, kind, side, turn)
    if not paths:
        return
    pale = palette.lighten(palette.mix(h.skin_light, h.skin_warm, 0.35), 0.105)
    pink = palette.mix(h.skin_warm, h.lip, 0.16)
    deep = palette.mix(h.skin_deep, h.skin_cool, 0.28)
    for path in paths:

        shadow = S.lock(path + np.array([0.003 * side, -0.007]),
                        w0=0.0084, w1=0.0032, bulge=0.0022, per_segment=5)
        brush.paint(canvas, shadow, deep, clip=clip, alpha=0.36, soft=1.2)
        warm = S.lock(path + np.array([0.001 * side, -0.001]),
                      w0=0.0052, w1=0.0019, per_segment=5)
        brush.paint(canvas, warm, pink, clip=clip, alpha=0.34, soft=0.40)
        ridge = S.lock(path, w0=0.0050, w1=0.0018, bulge=0.0018, per_segment=5)
        brush.paint(canvas, ridge, pale, clip=clip, alpha=0.96, soft=0.22)

        for u in (0.22, 0.48):
            p = path[0] + (path[-1] - path[0]) * u
            q = path[0] + (path[-1] - path[0]) * (u + 0.055)
            if np.linalg.norm(q - p) > 1e-6:
                normal = np.array([-(q - p)[1], (q - p)[0]])
                normal /= max(float(np.linalg.norm(normal)), 1e-9)
                tick = S.lock([p - normal * 0.004, p + normal * 0.004],
                              w0=0.0012, w1=0.0010, per_segment=2)
                brush.paint(canvas, tick, deep, clip=clip, alpha=0.30, soft=0.25)


def paint_eyewear(canvas, st, h, genes: dict, turn: float = 0.0, asym=None) -> None:
    """Paint frames and lenses over the finished face and hair.

    The lens is a tint, a thin rim and a single long reflection.  It is kept
    translucent on purpose: the eye pass is the most expensive and the most
    characterful part of the portrait, and a pair of opaque discs would erase it.
    """
    style = _get(genes, "eyewear")
    if style == "none":
        return
    side = _side(genes, "eyewear_side", asym)
    frame = _frame_colour(h)
    frame_hi = palette.lighten(frame, 0.11)
    frame_sh = palette.darken(frame, 0.10)
    lenses = []
    for s in (+1, -1):
        if style == "monocle" and s != side:
            continue
        cx, cy, hw, hh = _lens_box(st, s, turn, asym, style)
        lens = _lens(cx, cy, hw, hh, style)
        lenses.append((s, cx, cy, hw, hh, lens))

    if style not in ("monocle", "pince_nez"):
        for s, cx, cy, hw, hh, lens in lenses:
            ax, ay = _temple_anchor(st, s, turn)
            start = (cx + s * hw * 0.92, cy + hh * 0.28)
            end = (ax, ay - 0.008)
            arm = S.lock([start, ((start[0] + end[0]) * 0.5,
                                  (start[1] + end[1]) * 0.5 + 0.004), end],
                         w0=0.0040, w1=0.0022, per_segment=4)
            brush.paint(canvas, arm, frame, alpha=0.92, soft=0.25)
            brush.paint(canvas, S.lock([start, end], w0=0.0010, w1=0.0007,
                                       per_segment=3),
                        frame_hi, alpha=0.55, soft=0.16)

    for s, cx, cy, hw, hh, lens in lenses:
        if style == "sunglasses":
            tint = palette.mix(h.lid_shadow, h.iris_dark, 0.72)
            alpha = 0.34
        else:
            tint = palette.mix(h.sclera, h.lid_shadow, 0.24)
            alpha = 0.060
        brush.paint(canvas, lens, tint, alpha=alpha, soft=0.35)

        gleam = S.lock([(cx - hw * 0.56, cy + hh * 0.62),
                        (cx + hw * 0.02, cy + hh * 0.96)],
                       w0=hw * 0.045, w1=hw * 0.012, per_segment=3)
        brush.paint(canvas, gleam & lens, palette.mix((1.0, 1.0, 1.0),
                                                      h.sclera, 0.28),
                    alpha=0.20, soft=0.45)

        brush.paint(canvas, lens - lens.offset(-0.0060), frame,
                    alpha=0.97, soft=0.18)
        brush.paint(canvas, lens.outline(0.0018), frame_hi,
                    alpha=0.60, soft=0.14)
        brush.paint(canvas, lens.outline(0.0036), frame_sh,
                    alpha=0.40, soft=0.26)

    if style == "pince_nez":
        for x, y in ((-1, 0), (1, 0)):
            cx, cy, hw, hh = _lens_box(st, x, turn, asym, style)
            if x == -1:
                left = (cx + hw * 0.96, cy + hh * 0.08)
            else:
                right = (cx - hw * 0.96, cy + hh * 0.08)
        bridge = S.lock([left, ((left[0] + right[0]) * 0.5, left[1] + 0.006),
                         right], w0=0.0026, w1=0.0026, per_segment=4)
        brush.paint(canvas, bridge, frame, alpha=0.90, soft=0.30)
        for s in (+1, -1):
            cx, cy, hw, hh = _lens_box(st, s, turn, asym, style)
            brush.paint(canvas, S.circle(0.0042, cx - s * hw * 0.76, cy - hh * 0.62),
                        frame_hi, alpha=0.70, soft=0.25)
    elif style not in ("monocle",):
        left = _lens_box(st, -1, turn, asym, style)
        right = _lens_box(st, +1, turn, asym, style)
        a = (left[0] + left[2] * 0.92, left[1] + left[3] * 0.12)
        b = (right[0] - right[2] * 0.92, right[1] + right[3] * 0.12)
        bridge = S.lock([a, ((a[0] + b[0]) * 0.5, a[1] + 0.009), b],
                        w0=0.0022, w1=0.0022, bulge=0.0014, per_segment=4)
        brush.paint(canvas, bridge, frame, alpha=0.96, soft=0.20)
        brush.paint(canvas, S.lock([a, b], w0=0.0010, w1=0.0010, per_segment=3),
                    frame_hi, alpha=0.58, soft=0.12)
    else:
        cx, cy, hw, hh = _lens_box(st, side, turn, asym, style)

        start = (cx + side * hw * 0.74, cy - hh * 0.82)
        end = (cx + side * hw * 1.10, cy - hh * 1.80)
        chain = S.lock([start, (start[0] + side * 0.012,
                                (start[1] + end[1]) * 0.5), end],
                       w0=0.0017, w1=0.0010, per_segment=4)
        brush.paint(canvas, chain, frame, alpha=0.72, soft=0.25)
        brush.paint(canvas, S.circle(0.006, start[0], start[1]),
                    frame_hi, alpha=0.60, soft=0.25)
