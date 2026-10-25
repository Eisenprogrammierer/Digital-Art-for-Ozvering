"""Humanising finish: the small asymmetries and contact shadows that stop a
procedural portrait reading as a symmetric doll.

The construction pass is deliberately regular: an eye is mirrored from an eye, a
brow from a brow, the skull loop is symmetric before a turn.  That regularity is
what makes the geometry testable, and it is also the strongest artificial tell in
the finished picture.  A real face is not a mirrored face -- one eye sits a little
higher, one brow arches more, the mouth is not level, and the light finds one
cheek before the other.

The rules here are as strict as the rest of the package:

* every offset is a pure function of :func:`gen.portraits.dna.face_seed`, which
  ignores the outfit, the mood and the hairstyle;
* every offset is small.  This is the difference between a person and a mask,
  not a caricature;
* every mark is clipped to the surface it belongs to, so a shadow can never move
  a feature or paint on the background.

The module owns no geometry of its own.  It reads the style and paints over it,
which is why it can be added to the composition as one optional pass.
"""
from __future__ import annotations

import numpy as np

from core import noise, palette

from . import brush, dna, head
from . import shape as S
from .shape import union_all

__all__ = ["Asym", "asymmetry", "paint_face_life", "paint_hair_cast",
           "contour_mask", "hair_contour_mask"]


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Asym:
    """The sitter's bilateral deviations, in head units.

    ``eye``, ``brow`` and ``ear`` are keyed by side (+1 is the character's left,
    which is the viewer's right, exactly as in :mod:`eyes`).  The scalar mouth
    offsets are in the same units and are deliberately not mirrored.
    """

    __slots__ = ("eye", "brow", "ear", "nose_dx", "nose_sx", "mouth_dy",
                 "mouth_dx", "mouth_w", "mouth_corner_l", "mouth_corner_r",
                 "lean", "seed")

    def __init__(self, eye, brow, ear, nose_dx, nose_sx, mouth_dy, mouth_dx,
                 mouth_w, mouth_corner_l, mouth_corner_r, lean, seed):
        self.eye = eye
        self.brow = brow
        self.ear = ear
        self.nose_dx = float(nose_dx)
        self.nose_sx = float(nose_sx)
        self.mouth_dy = float(mouth_dy)
        self.mouth_dx = float(mouth_dx)
        self.mouth_w = float(mouth_w)
        self.mouth_corner_l = float(mouth_corner_l)
        self.mouth_corner_r = float(mouth_corner_r)
        self.lean = float(lean)
        self.seed = int(seed)


def _clamp(value, lo, hi):
    return float(np.clip(float(value), lo, hi))


def asymmetry(genes: dict) -> Asym:
    """Build the stable bilateral offsets for a genome.

    The amplitudes were chosen by looking at 384 and 640 px renders: below about
    two thousandths of a head the eye cannot see the difference, and above about
    eight the face starts to look injured rather than individual.  The generator
    is seeded from :func:`dna.face_seed`, not from the hair generator, so a coat,
    a smile or a turned head cannot change the person.
    """
    seed = int(dna.face_seed(genes))
    rng = np.random.default_rng(seed ^ 0x9E3779B9)

    lean = _clamp(rng.normal(0.0, 0.62), -1.15, 1.15)

    eye = {}
    brow = {}
    ear = {}
    for side in (+1, -1):

        eye[side] = {
            "dx": _clamp(0.0030 * lean + rng.normal(0.0, 0.0012), -0.0060, 0.0060),
            "dy": _clamp(0.0045 * lean * side + rng.normal(0.0, 0.0018),
                         -0.0072, 0.0072),
            "sx": _clamp(1.0 + 0.022 * lean + rng.normal(0.0, 0.009), 0.940, 1.060),
            "sy": _clamp(1.0 + 0.020 * lean * side + rng.normal(0.0, 0.009),
                         0.940, 1.060),
            "rot": _clamp(1.15 * lean * side + rng.normal(0.0, 0.50), -2.20, 2.20),

            "gaze_x": _clamp(-side * (0.0012 + abs(rng.normal(0.0, 0.0008))),
                             -0.0030, -0.0004),
            "gaze_y": _clamp(rng.normal(0.0, 0.0009), -0.0022, 0.0022),
            "hl_dx": _clamp(rng.normal(0.0, 0.0014), -0.0030, 0.0030),
            "hl_dy": _clamp(rng.normal(0.0, 0.0014), -0.0030, 0.0030),
        }
        brow[side] = {
            "dx": _clamp(rng.normal(0.0, 0.0018), -0.0040, 0.0040),
            "dy": _clamp(0.0040 * lean * side + rng.normal(0.0, 0.0018),
                         -0.0068, 0.0068),
            "arch": _clamp(0.0030 * lean * side + rng.normal(0.0, 0.0016),
                           -0.0054, 0.0054),
            "sx": _clamp(1.0 + rng.normal(0.0, 0.018), 0.945, 1.055),
        }
        ear[side] = {
            "dx": _clamp(rng.normal(0.0, 0.0013), -0.0030, 0.0030),
            "dy": _clamp(0.0034 * lean * side + rng.normal(0.0, 0.0025),
                         -0.0075, 0.0075),
            "scale": _clamp(1.0 + 0.034 * lean * side + rng.normal(0.0, 0.020),
                            0.90, 1.10),
        }

    nose_dx = _clamp(0.0030 * lean + rng.normal(0.0, 0.0013), -0.0056, 0.0056)
    nose_sx = _clamp(1.0 + 0.034 * lean + rng.normal(0.0, 0.013), 0.92, 1.08)
    mouth_dy = _clamp(rng.normal(0.0, 0.0018), -0.0044, 0.0044)
    mouth_dx = _clamp(0.0024 * lean + rng.normal(0.0, 0.0012), -0.0048, 0.0048)
    mouth_w = _clamp(1.0 + 0.021 * lean + rng.normal(0.0, 0.009), 0.95, 1.05)
    mouth_corner_l = _clamp(0.0026 * lean + rng.normal(0.0, 0.0012),
                            -0.0048, 0.0048)
    mouth_corner_r = _clamp(-0.0026 * lean + rng.normal(0.0, 0.0012),
                            -0.0048, 0.0048)

    return Asym(eye, brow, ear, nose_dx, nose_sx, mouth_dy, mouth_dx, mouth_w,
                mouth_corner_l, mouth_corner_r, lean, seed)


def paint_face_life(canvas, st, h, genes: dict, turn: float, asym: Asym,
                    clip) -> None:
    """Paint the asymmetric planes and contact marks over the flat skin pass.

    The existing skin pass is broad and symmetric: it knows about a cheekbone and
    a jaw, but not about *this* cheekbone.  The marks here are intentionally
    low-alpha.  Their job is to make the eye accept two different sides, not to
    replace the cel shading.
    """
    if clip is None:
        return
    bw = float(st.head_w)
    ey = float(st.eye_y)
    heavy = 1.0 if asym.lean >= 0.0 else -1.0

    for side in (+1, -1):
        cx, cy = head.turn_point(side * bw * 0.39, ey - 0.070, turn, st)
        strength = 1.0 if side == heavy else 0.55
        brush.paint(canvas,
                    S.ellipse(bw * (0.18 + 0.02 * strength), 0.050,
                              float(cx), float(cy)),
                    h.skin_shadow, clip=clip,
                    alpha=0.042 * strength, soft=3.2)
        lx, ly = head.turn_point(-side * bw * 0.43, ey - 0.020, turn, st)
        brush.paint(canvas,
                    S.ellipse(bw * 0.16, 0.042, float(lx), float(ly)),
                    h.skin_light, clip=clip, alpha=0.044, soft=3.0)

        ja, jay = head.turn_point(side * st.jaw_w * 0.80,
                                  st.jaw_y + 0.105, turn, st)
        jb, jby = head.turn_point(side * st.chin_w * 0.58,
                                  st.chin_y + 0.095, turn, st)
        shade = S.lock([(float(ja), float(jay)),
                        ((float(ja) + float(jb)) * 0.5,
                         (float(jay) + float(jby)) * 0.5 + 0.006),
                        (float(jb), float(jby))],
                       w0=bw * 0.020, w1=bw * 0.013, bulge=bw * 0.006,
                       per_segment=5)
        brush.paint(canvas, shade, h.skin_shadow, clip=clip,
                    alpha=0.030 if side == heavy else 0.014, soft=2.0)

    nx, ny = head.turn_point(asym.nose_dx, st.nose_y, turn, st)
    brush.paint(canvas, S.ellipse(st.nose_w * 0.22, st.nose_h * 0.20,
                                  float(nx), float(ny - st.nose_h * 0.04)),
                h.skin_warm, clip=clip, alpha=0.100, soft=2.0)

    a = (float(nx) + st.nose_w * 0.10, float(ny) + st.nose_h * 0.22)
    b = (float(nx) + st.nose_w * 0.72, float(ny) - st.nose_h * 1.20)
    cast = S.lock([a, ((a[0] + b[0]) * 0.5 + st.nose_w * 0.06,
                       (a[1] + b[1]) * 0.5), b],
                  w0=st.nose_w * 0.10, w1=st.nose_w * 0.17,
                  bulge=st.nose_w * 0.055, per_segment=5)
    brush.paint(canvas, cast, h.skin_shadow, clip=clip, alpha=0.072, soft=2.2)

    brush.paint(canvas, S.ellipse(st.nose_w * 0.17, st.nose_h * 0.10,
                                  float(nx), float(ny - st.nose_h * 0.58)),
                h.skin_shadow, clip=clip, alpha=0.055, soft=1.6)

    for side in (+1, -1):
        ear = asym.ear[side]
        r = st.head_w * 0.17 * float(getattr(st, "ear_size", 1.0)) * ear["scale"]
        ex, eyy = head.turn_point(side * st.head_w * 0.80, st.eye_y + 0.015,
                                  turn, st)
        ex += ear["dx"]
        eyy += ear["dy"]
        brush.paint(canvas, S.ellipse(r * 0.48, r * 0.72, float(ex), float(eyy)),
                    h.skin_warm, clip=clip, alpha=0.075, soft=1.8)
        brush.paint(canvas, S.ellipse(r * 0.55, r * 0.82,
                                      float(ex - side * r * 0.30), float(eyy)),
                    h.skin_shadow, clip=clip, alpha=0.050, soft=2.0)

    for side in (+1, -1):
        e = asym.eye[side]
        cx, cy = head.turn_point(side * st.eye_dx, st.eye_y, turn, st)
        brush.paint(canvas,
                    S.ellipse(st.eye_w * 0.43, st.eye_up * 0.24,
                              float(cx) + e["dx"],
                              float(cy) + st.eye_up * 0.78 + e["dy"]),
                    h.skin_warm, clip=clip, alpha=0.042, soft=2.4)

    X, Y = canvas.grid()
    n = noise.fbm(X, Y, seed=(asym.seed ^ 0x51A7) & 0x7FFFFFFF,
                  freq=6.2, octaves=3)
    sheen = np.clip((n - 0.53) * 2.4, 0.0, 1.0)
    tzone = np.exp(-((X / max(bw * 0.52, 1e-4)) ** 2
                     + ((Y - (ey + 0.10)) / 0.30) ** 2))
    amount = clip * sheen * tzone * 0.070
    canvas.over(palette.srgb_to_linear(h.skin_light), amount)


def paint_hair_cast(canvas, st, h, hs, neck_cov, bust_cov, cloth_shadow=None) -> None:
    """A little contact shade where the hair meets the neck and shoulders.

    The back hair is painted before the body and the near locks after it, so
    without this pass the hair and the body are two flat layers that happen to
    overlap.  The shifted mask is taken from the hair itself, which means a short
    cut and a long fall cast different shadows without a per-style table.  The
    neck takes skin shadow and the bust takes cloth shadow; one colour for both
    would paint a brown band across the uniform.
    """
    if neck_cov is None or bust_cov is None:
        return
    if max(float(neck_cov.max()), float(bust_cov.max())) <= 0.0:
        return
    hair_cov = brush.mask(canvas, hs.back_mass)
    if hs.near_locks:
        hair_cov = np.maximum(hair_cov, brush.mask(canvas, union_all(hs.near_locks)))
    if hair_cov.max() <= 0.0:
        return
    shadow = brush.shift_cov(hair_cov, -0.030, canvas.scale)
    shadow = np.clip(shadow - hair_cov, 0.0, 1.0)
    neck_shadow = shadow * neck_cov * np.clip(1.0 - bust_cov, 0.0, 1.0)
    bust_shadow = shadow * bust_cov
    brush.paint_cov(canvas, neck_shadow, h.skin_deep, soft=2.0, alpha=0.14)
    brush.paint_cov(canvas, bust_shadow,
                    h.skin_deep if cloth_shadow is None else cloth_shadow,
                    soft=2.0, alpha=0.12)


def contour_mask(surf, st, asym: Asym):
    """A per-pixel mask for the outer skull contour.

    The current contour is already softened in ink and weight.  What it still
    lacked was permission to *stop*: a lit plane meeting the background does not
    carry the same line as the shadow under the jaw.  The mask is strongest on
    the shadow side and under the chin, and has a soft break over the lit temple.
    """
    if surf is None or surf.empty:
        return None
    x, y = surf.x, surf.y
    nx = x / max(float(st.head_w), 1e-4) + 0.06 * asym.lean
    ny = (y - 0.02) / 0.52
    shadow = _smooth((0.58 * nx - 0.48 * ny + 0.10) / 0.86)
    mask = 0.22 + 0.78 * shadow

    notch = np.exp(-(((nx + 0.58) / 0.22) ** 2 + ((ny - 0.12) / 0.28) ** 2))
    mask = mask * (1.0 - 0.52 * notch)

    mask = mask * np.clip((0.54 - y) / 0.10, 0.55, 1.0)
    return np.clip(mask, 0.10, 1.0)


def hair_contour_mask(surf, st, asym: Asym):
    """Break the hair's silhouette unevenly without redrawing its geometry."""
    if surf is None or surf.empty:
        return None
    x, y = surf.x, surf.y
    nx = x / max(float(st.head_w), 1e-4) + 0.05 * asym.lean
    ny = (y - 0.02) / 0.52
    shadow = _smooth((0.55 * nx - 0.35 * ny + 0.10) / 0.80)
    mask = 0.36 + 0.64 * shadow

    break_crown = np.exp(-(((nx + 0.42) / 0.28) ** 2 + ((ny - 0.48) / 0.24) ** 2))
    mask = mask * (1.0 - 0.58 * break_crown)
    return np.clip(mask, 0.16, 1.0)
