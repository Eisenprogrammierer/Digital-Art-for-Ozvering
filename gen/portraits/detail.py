"""The detail pass: the small, deterministic marks that stop a face reading as flat.

The construction pass gets the silhouette and the features right; this pass adds
the evidence that a person has skin, hair and clothes rather than flat fills.  It
is deliberately a separate module for one reason: every function here is optional
decoration over geometry that already has an invariant.  If a detail goes wrong it
must be clipped away, never allowed to move an eye or grow a jaw.

The marks are all keyed to the same genome:

* age thins lips, adds creases and under-eye shade, and greys the hair in
  :mod:`gen.portraits.hues`;
* sex changes the brow and jaw *before* the head is built, and adds a little
  stubble here;
* climate adds freckles, sun warmth and wind colour to the skin;
* culture changes collar, trim and clasp, not the face.

Everything is deterministic: the caller passes the same generator the hair used,
and no function reaches for a global random state.
"""
from __future__ import annotations

import numpy as np

from core import noise, palette, render

from . import brush, demography, head
from . import shape as S

__all__ = [
    "background", "paint_skin", "paint_facial_hair", "paint_neck", "paint_bust",
    "paint_hair_detail",
]


# --------------------------------------------------------------------------
# backdrop
# --------------------------------------------------------------------------
def background(canvas, h, genes: dict, rng) -> None:
    """A soft-lit backdrop with a vignette and a tooth of paper grain."""
    X, Y = canvas.grid()
    # A radial lift behind the head: the face needs a slightly brighter field than
    # the corners, or a dark-haired character merges into the backdrop.
    dx = X / 0.82
    dy = (Y + 0.08) / 0.90
    r = np.clip(np.sqrt(dx * dx + dy * dy), 0.0, 1.6)
    t = np.clip(1.0 - r * 0.72, 0.0, 1.0)
    base = palette.ramp_map([h.bg_lo, h.bg, h.bg_hi], t)
    grain = (noise.value_noise(X, Y, seed=71, freq=38.0) - 0.5) * 0.012
    base = np.clip(base + grain[..., None], 0.0, 1.0)
    canvas.rgb[:] = palette.srgb_to_linear(base)
    canvas.a[:] = 1.0

    # Two long, almost invisible diagonal washes keep the field from looking like
    # a CSS gradient.  They are painted behind everything, so a little strength is
    # enough.
    wash = S.lock([(-0.95, 0.70), (0.05, -0.20), (1.05, -0.75)],
                  w0=0.20, w1=0.34, bulge=0.10, per_segment=6, resample=28)
    brush.paint(canvas, wash, h.bg_hi, alpha=0.10, soft=6.0)
    vignette = S.ellipse(1.10, 1.05, 0.0, -0.02)
    brush.paint(canvas, vignette.outline(0.08), h.bg_lo, alpha=0.09, soft=10.0)


# --------------------------------------------------------------------------
# skin
# --------------------------------------------------------------------------
def _face_point(st, side: float, x: float, y: float, turn: float):
    return head.turn_point(side * x, y, turn, st)


def paint_skin(canvas, st, h, genes: dict, turn: float, clip, rng) -> None:
    """Mottling, weather, freckles and the planes of the face.

    The cel bands already give the big shadow.  What they cannot give is the
    difference between a cheekbone and a cheek *on the same band*, so this pass
    adds a low-frequency wash, a few soft plane changes, and the small marks that
    make skin read as skin at 384 px.
    """
    if clip is None:
        return
    X, Y = canvas.grid()
    seed = int(rng.integers(0, 2**31 - 1))

    # Low-frequency warmth/coolness.  This is not a lighting model; it is the
    # mottling real skin has, at an amplitude low enough to survive cel shading.
    n = noise.fbm(X, Y, seed=seed, freq=5.0, octaves=4)
    warm = np.clip((n - 0.48) * 2.0, 0.0, 1.0)
    cool = np.clip((0.52 - n) * 2.0, 0.0, 1.0)
    canvas.over(palette.srgb_to_linear(h.skin_warm), clip * warm * 0.032)
    canvas.over(palette.srgb_to_linear(h.skin_cool), clip * cool * 0.026)

    # Visible pores are a texture, not a noise field: a high-frequency field at a
    # few percent reads as grain at every size without becoming static.
    pore = (noise.value_noise(X, Y, seed=seed + 17, freq=58.0) - 0.5)
    canvas.over(palette.srgb_to_linear(h.skin_freckle), clip * np.abs(pore) * 0.012)

    # A continuous light term *under* the cel bands.  Flat bands are the style;
    # flat bands with nothing behind them are the reason a cheek can read as a
    # paper cut-out.  This is a low-amplitude gradient in the same light direction
    # as every other asset, and it is clipped to the skull.
    hgt = head.volume(st, turn)(X, Y)
    n = render.height_normals(hgt, canvas.dx, canvas.y_up)
    L = np.array((render.LIGHT[0], render.LIGHT[1], 0.60), dtype=np.float64)
    L = L / max(float(np.linalg.norm(L)), 1e-12)
    lam = np.clip(n @ L, 0.0, 1.0)
    shade = np.clip((0.62 - lam) * 0.62, 0.0, 0.14) * clip
    lift = np.clip((lam - 0.58) * 0.46, 0.0, 0.10) * clip
    canvas.over(palette.srgb_to_linear(h.skin_shadow), shade)
    canvas.over(palette.srgb_to_linear(h.skin_light), lift)

    # Soft planes: cheekbones, forehead, nose bridge and chin.  Soft-edged, low
    # alpha, and clipped to the skull so they cannot draw on the background.
    ex, ey = float(st.eye_y), float(st.eye_down)
    bw = float(st.head_w)
    for side in (+1, -1):
        cx, cy = _face_point(st, side, bw * 0.35, ey - 0.045, turn)
        brush.paint(canvas, S.ellipse(bw * 0.22, 0.075, float(cx), float(cy)),
                    h.skin_light, clip=clip, alpha=0.080, soft=3.2)
        under_y = ey - ex * 0.0 - st.eye_down - 0.027
        ux, uy = _face_point(st, side, st.eye_dx * 0.92, under_y, turn)
        age = demography.age_t(genes)
        brush.paint(canvas, S.ellipse(st.eye_w * 0.27, st.eye_w * 0.095,
                                      float(ux), float(uy)),
                    h.skin_cool, clip=clip, alpha=0.035 + 0.08 * age, soft=2.4)
    fx, fy = 0.0, float(st.eye_y) + 0.19
    brush.paint(canvas, S.ellipse(bw * 0.27, 0.060, fx, fy), h.skin_light,
                clip=clip, alpha=0.035, soft=3.5)
    nx, ny = head.turn_point(0.0, st.nose_y + st.nose_h * 0.40, turn, st)
    brush.paint(canvas, S.ellipse(st.nose_w * 0.28, st.nose_h * 0.45, float(nx), float(ny)),
                h.skin_light, clip=clip, alpha=0.075, soft=1.8)
    cx, cy = head.turn_point(0.0, st.chin_y + 0.065, turn, st)
    brush.paint(canvas, S.ellipse(st.chin_w * 0.55, 0.030, float(cx), float(cy)),
                h.skin_light, clip=clip, alpha=0.070, soft=2.0)
    # Chin crease: the small shadow that separates the lower lip from the chin.
    cmx, cmy = head.turn_point(0.0, st.mouth_y - st.mouth_w * 0.22, turn, st)
    brush.paint(canvas, S.ellipse(st.mouth_w * 0.27, 0.012, float(cmx), float(cmy)),
                h.skin_shadow, clip=clip, alpha=0.070, soft=1.6)

    # The jaw and temples: the face needs a lower plane and a cast shade where the
    # hair meets the skull, or the cheeks read as one flat oval.
    for side in (+1, -1):
        a = _face_point(st, side, bw * 0.53, ey - 0.16, turn)
        b = _face_point(st, side, st.chin_w * 0.70, st.chin_y + 0.055, turn)
        jaw = S.lock([a, ((a[0] + b[0]) * 0.51, (a[1] + b[1]) * 0.5), b],
                     w0=0.030, w1=0.014, bulge=0.020, per_segment=5)
        brush.paint(canvas, jaw, h.skin_shadow, clip=clip, alpha=0.080, soft=3.2)
        tx, ty = _face_point(st, side, bw * 0.72, ey + 0.11, turn)
        brush.paint(canvas, S.ellipse(bw * 0.14, 0.11, float(tx), float(ty)),
                    h.skin_shadow, clip=clip, alpha=0.100, soft=3.6)
    # Cheek warmth, always present; the blush gene adds the manga marks on top.
    for side in (+1, -1):
        bx, by = _face_point(st, side, st.eye_dx + st.eye_w * 0.12, st.blush_y, turn)
        brush.paint(canvas, S.ellipse(st.eye_w * 0.42, st.eye_w * 0.20,
                                      float(bx), float(by)),
                    h.blush, clip=clip, alpha=0.050, soft=3.0)

    _paint_freckles(canvas, st, h, genes, turn, clip, rng, seed)
    _paint_age_lines(canvas, st, h, genes, turn, clip)


def _paint_freckles(canvas, st, h, genes, turn, clip, rng, seed) -> None:
    climate = str(genes.get("climate", demography.DEFAULTS["climate"]))
    pattern = str(genes.get("freckles", "sparse"))
    chance = {"arctic": 0.45, "temperate": 1.0, "arid": 0.95,
              "tropical": 0.40, "highland": 0.75}.get(climate, 0.7)
    pattern_scale = {"none": 0.0, "sparse": 0.35, "bridge": 0.75,
                     "cheeks": 0.90, "full": 1.40}.get(pattern, 0.35)
    age = demography.age_t(genes)
    count = int(round(28 * chance * pattern_scale * (0.65 + 0.5 * age)))
    for _ in range(count):
        # Most freckles belong on the nose and the two cheek apples; a uniform
        # scatter over the whole face reads as measles.  The categorical gene
        # moves the centre of that distribution: bridge freckles cluster on the
        # nose, cheek freckles on the apples, and a full pattern crosses both.
        side = -1.0 if rng.random() < 0.5 else 1.0
        nose_p = {"bridge": 0.78, "cheeks": 0.10, "full": 0.30}.get(pattern, 0.32)
        if rng.random() < nose_p:
            x = float(rng.normal(0.0, st.nose_w * 0.55))
            y = float(st.nose_y + rng.normal(0.0, st.nose_h * 0.55))
        else:
            x = side * float(rng.uniform(st.nose_w * 0.75, st.head_w * 0.58))
            y = float(st.nose_y + rng.normal(0.0, st.head_w * 0.18) + st.nose_h * 0.18)
        if x * x / (st.head_w * 0.72) ** 2 + (y - 0.02) ** 2 / 0.30 ** 2 > 1.0:
            continue
        px, py = head.turn_point(x, y, turn, st)
        r = float(rng.uniform(0.0018, 0.0052))
        brush.paint(canvas, S.circle(r, float(px), float(py)), h.skin_freckle,
                    clip=clip, alpha=float(rng.uniform(0.18, 0.46)), soft=0.25)

    # One or two moles per face, not on the midline: asymmetry is what makes a
    # face recognisable, and a perfectly even scatter looks generated.
    for _ in range(1 + int(rng.random() < 0.45)):
        side = -1.0 if rng.random() < 0.5 else 1.0
        x = side * float(rng.uniform(st.head_w * 0.22, st.head_w * 0.55))
        y = float(rng.uniform(st.nose_y + 0.02, st.eye_y - 0.04))
        px, py = head.turn_point(x, y, turn, st)
        brush.paint(canvas, S.circle(float(rng.uniform(0.003, 0.006)), float(px), float(py)),
                    h.skin_freckle, clip=clip, alpha=0.65, soft=0.2)


def _paint_age_lines(canvas, st, h, genes, turn, clip) -> None:
    age = demography.age_t(genes)
    if age < 0.60:
        return
    strength = float(np.clip((age - 0.55) / 0.40, 0.0, 1.0))
    # Nasolabial folds: from the nose wing towards the mouth corner.  The line has
    # to stop short of the lip or it reads as a moustache.
    for side in (+1, -1):
        a = _face_point(st, side, st.nose_w * 0.82, st.nose_y - st.nose_h * 0.10, turn)
        b = _face_point(st, side, st.mouth_w * 0.40, st.mouth_y + 0.025, turn)
        fold = S.lock([a, ((a[0] + b[0]) * 0.5 + side * 0.008, (a[1] + b[1]) * 0.5), b],
                      w0=0.0035, w1=0.0015, bulge=0.004, per_segment=5)
        brush.paint(canvas, fold, h.skin_shadow, clip=clip,
                    alpha=0.10 + 0.10 * strength, soft=1.1)
    # Forehead creases: short, broken arcs rather than a single ruler line.
    for k, off in enumerate((0.075, 0.115)):
        x0, y0 = head.turn_point(-st.head_w * 0.27, st.brow_y + off, turn, st)
        x1, y1 = head.turn_point(st.head_w * 0.27, st.brow_y + off + 0.006, turn, st)
        crease = S.lock([(float(x0), float(y0)),
                         (0.0, float((y0 + y1) * 0.5) + 0.006),
                         (float(x1), float(y1))],
                        w0=0.0022, w1=0.0014, bulge=0.005, per_segment=5)
        brush.paint(canvas, crease, h.skin_shadow, clip=clip,
                    alpha=(0.055 + 0.045 * strength) * (0.8 if k else 1.0), soft=1.0)
    # Crow's feet at the outer corners, two fine strokes per side.
    for side in (+1, -1):
        ex, ey = _face_point(st, side, st.eye_dx + st.eye_w * 0.55, st.eye_y, turn)
        for j in (-1, 1):
            tip = (float(ex) + side * 0.035, float(ey) + j * 0.023)
            foot = S.lock([(float(ex), float(ey) + j * 0.006), tip],
                          w0=0.0018, w1=0.0008, per_segment=3)
            brush.paint(canvas, foot, h.skin_shadow, clip=clip,
                        alpha=0.075 * strength, soft=0.8)


def paint_facial_hair(canvas, st, h, genes: dict, turn: float, clip, rng) -> None:
    """A restrained five-o'clock shadow for adult male faces.

    This is a *plane* of cool hair colour, not a beard: the geometry for a beard
    would need its own silhouette and its own style table, and a half-built beard
    is worse than a clean jaw.  It is clipped to the skull and painted before the
    mouth, so the lip keeps its edge.
    """
    if str(genes.get("sex", demography.DEFAULTS["sex"])) != "male":
        return
    age = demography.age_t(genes)
    if age < 0.42:
        return
    strength = 0.075 + 0.055 * age
    # Jaw and chin: two soft crescents following the lower silhouette.
    for side in (+1, -1):
        a = _face_point(st, side, st.head_w * 0.48, st.eye_y - 0.15, turn)
        b = _face_point(st, side, st.chin_w * 0.70, st.chin_y + 0.045, turn)
        shadow = S.lock([a, ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5), b],
                        w0=0.030, w1=0.016, bulge=0.024, per_segment=5)
        brush.paint(canvas, shadow, h.skin_cool, clip=clip, alpha=strength, soft=2.2)
    # Upper lip and chin patch, lighter than the jaw.
    mx, my = head.turn_point(0.0, st.mouth_y + 0.030, turn, st)
    brush.paint(canvas, S.ellipse(st.mouth_w * 0.34, 0.012, float(mx), float(my)),
                h.hair_deep, clip=clip, alpha=strength * 0.55, soft=1.6)
    cx, cy = head.turn_point(0.0, st.chin_y + 0.085, turn, st)
    brush.paint(canvas, S.ellipse(st.chin_w * 0.38, 0.020, float(cx), float(cy)),
                h.hair_deep, clip=clip, alpha=strength * 0.45, soft=1.8)


# --------------------------------------------------------------------------
# neck and bust
# --------------------------------------------------------------------------
def paint_neck(canvas, st, h, genes: dict, turn: float, clip) -> None:
    """Neck planes and the soft folds age and posture put there."""
    if clip is None:
        return
    age = demography.age_t(genes)
    nw = float(st.neck_w)
    top = float(st.chin_y)
    # Central throat light, two side shadows: the cheapest way to turn a column
    # into a cylinder without abandoning cel shading.
    brush.paint(canvas, S.ellipse(nw * 0.26, 0.15, 0.0, top - 0.16),
                h.skin_light, clip=clip, alpha=0.10, soft=3.4)
    for side in (+1, -1):
        brush.paint(canvas, S.ellipse(nw * 0.20, 0.15, side * nw * 0.70, top - 0.17),
                    h.skin_shadow, clip=clip, alpha=0.12, soft=3.0)
        a = head.turn_point(side * nw * 0.62, top - 0.02, turn, st)
        b = head.turn_point(side * nw * 0.30, float(st.neck_y) + 0.02, turn, st)
        cord = S.lock([(float(a[0]), float(a[1])),
                       (side * nw * 0.46, (a[1] + b[1]) * 0.5),
                       (float(b[0]), float(b[1]))],
                      w0=0.0040, w1=0.0020, bulge=0.004, per_segment=5)
        brush.paint(canvas, cord, h.skin_shadow, clip=clip,
                    alpha=0.075 + 0.045 * age, soft=1.3)
    for k in range(1 if age < 0.75 else 2):
        y = top - 0.18 - 0.045 * k
        crease = S.lock([(-nw * 0.52, y), (0.0, y - 0.012), (nw * 0.52, y)],
                        w0=0.0020, w1=0.0012, bulge=0.005, per_segment=5)
        brush.paint(canvas, crease, h.skin_shadow, clip=clip,
                    alpha=0.055 + 0.045 * age, soft=1.0)


def paint_bust(canvas, st, h, genes: dict, kind: str, clip) -> None:
    """Collar trim, seams, fabric folds and the few fastenings a bust needs.

    The shoulders are the largest area in the frame, so a flat fill there makes
    the whole portrait look unfinished.  The clothing pass is still procedural
    geometry: broad folds, a woven grain, a collar with a trim, and a clasp or
    buttons.  Culture chooses the trim and fastening; the cut stays the same.
    """
    if clip is None:
        return
    lw = float(st.ink_w) * float(st.line)
    culture = str(genes.get("culture", demography.DEFAULTS["culture"]))
    X, Y = canvas.grid()
    nw = float(st.neck_w)
    top = float(st.neck_y)

    # Woven grain and a soft directional wash.  This is deliberately almost
    # invisible; its job is to keep the cloth from reading as a single flat
    # colour, not to become a pattern.
    weave = noise.fbm(X, Y, seed=131, freq=24.0, octaves=3) - 0.5
    canvas.over(palette.srgb_to_linear(h.cloth_hi), clip * np.clip(weave, 0.0, 1.0) * 0.045)
    canvas.over(palette.srgb_to_linear(h.cloth_shadow), clip * np.clip(-weave, 0.0, 1.0) * 0.050)
    for side in (+1, -1):
        brush.paint(canvas, S.ellipse(float(st.shoulder_w) * 0.34, 0.16,
                                      side * float(st.shoulder_w) * 0.38, top - 0.09),
                    h.cloth_hi, clip=clip, alpha=0.090, soft=8.0)
        brush.paint(canvas, S.ellipse(float(st.shoulder_w) * 0.22, 0.18,
                                      side * float(st.shoulder_w) * 0.78, top - 0.08),
                    h.cloth_shadow, clip=clip, alpha=0.13, soft=7.0)
    brush.paint(canvas, S.ellipse(nw * 1.65, 0.11, 0.0, top + 0.02),
                h.cloth_shadow, clip=clip, alpha=0.20, soft=4.0)

    # The opening.  The skin inside the neckline is what makes the line read as
    # an edge of cloth, and the depth is the only difference between the cuts.
    x0 = min(nw * 1.16, float(st.shoulder_w) * 0.285)
    # A passport shirt closes at the throat.  A deep V reads as a t-shirt
    # neckline; the opening is only wide enough to show the base of the neck.
    depth = {"v": 0.056, "scoop": 0.050, "crew": 0.042}.get(kind, 0.050)
    # The opening is only the throat, not the whole chest.  A shirt collar
    # closes below it: the skin V ends at the top button, and the collar leaves
    # lie on the cloth underneath.  Making the opening chest-deep was what made
    # every previous collar read as a V-neck t-shirt.
    opening = S.poly([(-x0, top + 0.035), (0.0, top - depth), (x0, top + 0.035)],
                     resample=38)
    brush.paint(canvas, opening, h.skin, clip=clip, alpha=0.97, soft=0.4)
    brush.paint(canvas, opening & S.ellipse(x0 * 0.82, 0.060, 0.0, top + 0.020),
                h.skin_shadow, clip=clip, alpha=0.30, soft=2.6)
    brush.paint(canvas, opening & S.ellipse(x0 * 0.48, 0.070, 0.0, top - depth * 0.50),
                h.skin_light, clip=clip, alpha=0.12, soft=2.8)

    # The collar stand is the band that actually goes round the neck.  Without
    # it the leaves below have nothing to be attached to, and the result reads as
    # a V-neck trim on a t-shirt.  It sits on the skin at the base of the neck,
    # not behind it, so it is painted here rather than with the shirt fill.
    stand_spine = [(-x0 * 1.14, top + 0.060), (0.0, top + 0.030),
                   (x0 * 1.14, top + 0.060)]
    stand = S.lock(stand_spine, w0=0.022, w1=0.022, bulge=0.016,
                   per_segment=6, resample=30)
    collar_col = palette.lighten(palette.mix(h.cloth_hi, h.cloth_trim, 0.10), 0.085)
    brush.paint(canvas, stand, collar_col, alpha=0.98, soft=0.45)
    brush.paint(canvas, S.lock([(p[0], p[1] - 0.011) for p in stand_spine],
                               w0=0.006, w1=0.006, bulge=0.004,
                               per_segment=6, resample=30),
                h.cloth_shadow, alpha=0.14, soft=0.9)
    brush.paint(canvas, stand.outline(lw * 0.50), h.cloth_line,
                alpha=0.60, soft=0.30)
    brush.paint(canvas, S.lock([(p[0], p[1] + 0.008) for p in stand_spine],
                               w0=0.003, w1=0.003, bulge=0.002,
                               per_segment=6, resample=30),
                h.cloth_hi, alpha=0.42, soft=0.6)

    # Two collar leaves.  Their inner edges follow the V; their outer edges run
    # from the shoulder to a point near the bottom of the opening.  This is the
    # silhouette of a folded collar, not a ring around a t-shirt neck.
    point_drop = {"v": 1.26, "scoop": 1.20, "crew": 1.12}.get(kind, 1.20)
    for side in (+1, -1):
        a = (side * x0 * 1.02, top + 0.065)
        b = (side * x0 * 1.58, top + 0.010)
        c = (side * x0 * 1.10, top - depth * 0.50)
        d = (side * x0 * 0.14, top - depth * point_drop)
        # The leaves close over the throat; only the narrow central notch shows
        # skin.  Subtracting the whole opening made a V-neck, not a shirt collar.
        leaf = S.poly([a, b, c, d], resample=32)
        # A soft cast shadow under the free edge lifts the collar off the shirt.
        shadow = leaf.offset(0.010).place(cy=-0.010) - leaf
        brush.paint(canvas, shadow, h.cloth_shadow, clip=clip,
                    alpha=0.30, soft=1.2)
        brush.paint(canvas, leaf, collar_col, clip=clip, alpha=1.0, soft=0.30)
        # A broad inner fold and a lit outer edge: the two planes of a collar.
        fold = S.lock([a, d], w0=x0 * 0.18, w1=x0 * 0.08,
                      bulge=x0 * 0.05, per_segment=5)
        brush.paint(canvas, leaf & fold, h.cloth_shadow, clip=clip,
                    alpha=0.22, soft=1.0)
        brush.paint(canvas, leaf, h.cloth_trim, clip=clip, alpha=0.04, soft=0.8)
        # The free edge is inked more firmly than the cloth: at avatar size the
        # collar has to survive as a silhouette, not only as a value change.
        brush.paint(canvas, leaf.outline(lw * 0.85), h.line, clip=clip,
                    alpha=0.62, soft=0.20)
        brush.paint(canvas, leaf.outline(lw * 1.25), collar_col, clip=clip,
                    alpha=0.22, soft=0.7)
    # The notch between the two collar points: a small dark wedge, not a gap
    # all the way down to the placket.
    notch_y = top - depth * point_drop + 0.012
    notch = S.poly([(-0.011, notch_y + 0.010), (0.0, notch_y - 0.004),
                    (0.011, notch_y + 0.010)], resample=18)
    brush.paint(canvas, notch, h.cloth_shadow, alpha=0.62, soft=0.5)

    # A short placket begins at the bottom of the V and runs down the chest.  It
    # is the one seam a collared shirt is allowed to have here, and it must start
    # *below* the collar rather than run through it.
    placket_top = top - depth * point_drop - 0.006
    placket_bottom = placket_top - 0.120
    placket = S.lock([(0.0, placket_top), (0.0, placket_bottom)],
                     w0=lw * 2.2, w1=lw * 2.6, bulge=lw * 0.5, per_segment=4)
    brush.paint(canvas, placket, h.cloth_hi, clip=clip, alpha=0.40, soft=0.8)
    for side in (+1, -1):
        edge = S.lock([(side * lw * 1.15, placket_top),
                       (side * lw * 1.35, placket_bottom)],
                      w0=lw * 0.26, w1=lw * 0.30, per_segment=3)
        brush.paint(canvas, edge, h.cloth_line, clip=clip, alpha=0.34, soft=0.5)

    # Two folds from the shoulder seam towards the chest.
    for side in (+1, -1):
        a = (side * float(st.shoulder_w) * 0.72, top - 0.06)
        b = (side * float(st.shoulder_w) * 0.42, top - 0.26)
        fold = S.lock([a, ((a[0] + b[0]) * 0.5 + side * 0.035, (a[1] + b[1]) * 0.5), b],
                      w0=0.022, w1=0.006, bulge=0.020, per_segment=5)
        brush.paint(canvas, fold, h.cloth_shadow, clip=clip, alpha=0.22, soft=2.2)
        brush.paint(canvas, S.lock([a, (a[0] - side * 0.035, a[1] - 0.10)],
                                   w0=0.0022, w1=0.0010, per_segment=3),
                    h.cloth_hi, clip=clip, alpha=0.26, soft=0.7)

    # Shoulder seams: two short lines at the outer slope, where a sleeve begins.
    for side in (+1, -1):
        x = side * float(st.shoulder_w) * 0.58
        seam = S.lock([(x, top - 0.02), (x + side * 0.08, top - 0.20)],
                      w0=0.0022, w1=0.0012, per_segment=3)
        brush.paint(canvas, seam, h.cloth_line, clip=clip, alpha=0.28, soft=0.7)

    # Fastenings: a clasp for the cultures that pin a collar, buttons for the
    # others.  Both are deliberately small; the bust should not become a costume
    # diagram.
    if culture in ("north", "steppe", "coast"):
        y = placket_top - 0.020
        disc = S.circle(0.014, 0.0, y)
        brush.paint(canvas, disc, h.metal, clip=clip, alpha=0.92, soft=0.2)
        brush.paint(canvas, disc.outline(0.0035), h.cloth_line, clip=clip, alpha=0.78)
        brush.paint(canvas, S.circle(0.005, -0.003, y + 0.004), h.cloth_hi,
                    clip=clip, alpha=0.78, soft=0.4)
    else:
        for k in range(2):
            y = placket_top - 0.032 - 0.060 * k
            brush.paint(canvas, S.circle(0.007, 0.0, y), h.metal, clip=clip,
                        alpha=0.82, soft=0.3)
            brush.paint(canvas, S.circle(0.007, 0.0, y).outline(0.0025),
                        h.cloth_line, clip=clip, alpha=0.55, soft=0.2)

    # A row of tiny embroidered diamonds under the collar, in the culture's
    # accent.  Three is enough to read as a border at 256 px.
    if culture in ("steppe", "desert", "highland"):
        y = top - 0.175
        for k in (-1, 0, 1):
            x = k * nw * 0.55
            diamond = S.poly([(x, y + 0.012), (x + 0.010, y),
                              (x, y - 0.012), (x - 0.010, y)], resample=None)
            brush.paint(canvas, diamond, h.cloth_trim, clip=clip, alpha=0.72, soft=0.25)


# --------------------------------------------------------------------------
# hair
# --------------------------------------------------------------------------
def paint_hair_detail(canvas, st, h, hs, turn: float, rng, skull_mask=None) -> None:
    """Clean cel light and a little texture over the finished hair.

    The hair construction already paints the cap, the fringe and the curtains.
    This pass adds one broad key light, one soft shadow, the parting, and a single
    silhouette contour.  It deliberately draws no per-lock ink: a row of outlines
    reads as straw at avatar size, and fine strand lines are not what makes hair
    look dense.
    """
    front = S.union_all([hs.cap, hs.front_mass])
    mask = brush.mask(canvas, front)
    if mask.max() <= 0.0:
        return
    X, Y = canvas.grid()

    def smooth(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    nx = X / max(float(st.head_w), 1e-3)
    ny = (Y - 0.05) / 0.52
    # Light from the upper left: the right and lower part of the hair falls into
    # shade, the left crown catches the key.  Multiplying in linear light keeps
    # the lock colours underneath instead of painting a flat grey over them.
    shadow = smooth((0.62 * nx - 0.50 * ny + 0.05) / 0.40)
    light = smooth((-0.55 * nx + 0.70 * ny - 0.05) / 0.38)
    factor = 1.0 + mask[..., None] * (
        shadow[..., None] * (0.70 - 1.0) + light[..., None] * (1.16 - 1.0)
    )
    canvas.rgb[:] = np.clip(canvas.rgb * factor, 0.0, 1.0)

    # A little vertical grain, not a fibre mesh: enough to break a perfectly
    # smooth fill, far too little to draw individual hairs.
    grain = noise.fbm(X * 2.6, Y * 0.72, seed=301, freq=15.0, octaves=3) - 0.5
    canvas.rgb[:] = np.clip(
        canvas.rgb * (1.0 + grain[..., None] * mask[..., None] * 0.05), 0.0, 1.0)

    # The parting is the one line that has to stay crisp.  It is a change of
    # plane at the scalp, so it gets a dark side and a narrow lit side.
    part = hs.p.get("part")
    if part is not None:
        px = float(part) * float(st.head_w) * 0.70
        part_line = S.lock([(px * 0.88, 0.44), (px, 0.30), (px * 1.05, 0.18)],
                           w0=0.014, w1=0.005, bulge=0.004,
                           per_segment=5, resample=24)
        brush.multiply(canvas, part_line, 0.78, clip=mask, alpha=0.28)
        lit_part = S.lock([(px * 0.88 + 0.008, 0.44),
                           (px + 0.007, 0.30), (px * 1.05 + 0.006, 0.18)],
                          w0=0.004, w1=0.002, bulge=0.002,
                          per_segment=5, resample=24)
        brush.multiply(canvas, lit_part, 1.08, clip=mask, alpha=0.12)

    # One clean silhouette for the whole front hair.  Inking every lock was what
    # made the hair look like straw; a single contour around the mass reads as a
    # drawn hairstyle and lets the fill carry the internal planes.
    lw = float(st.ink_w) * float(st.line) * 0.80
    outline = brush.Surface(canvas, front, pad=2.6 * lw)
    outline.stroke(canvas, lw, h.line_hair, inset=lw * 0.42,
                   alpha=0.64, soft=0.4)

    # A few hairs at the hairline and outside the silhouette.  They are kept
    # sparse on purpose: a rim of them is a haircut, a fringe of them is a mop.
    if skull_mask is not None:
        hairline = head.hairline_y(st)
        for i in range(5):
            u = (i + 0.5) / 5.0
            x = (u - 0.5) * float(st.head_w) * 1.18
            y = hairline + 0.012 - 0.020 * (1.0 - abs(u - 0.5) * 2.0)
            tip = (x + float(rng.uniform(-0.014, 0.014)),
                   y - float(rng.uniform(0.014, 0.030)))
            baby = S.lock([(x, y), tip], w0=0.0016, w1=0.0004, per_segment=3)
            brush.paint(canvas, baby, h.hair_fine, clip=skull_mask,
                        alpha=0.26, soft=0.35)

        # Outside the *hair*, not merely outside the skull: a flyaway drawn over
        # the cap is a light scratch across the mass, which is exactly the straw
        # artifact this pass is trying to avoid.
        all_hair = brush.mask(canvas, hs.mass)
        outside = np.clip(1.0 - skull_mask, 0.0, 1.0) * np.clip(1.0 - all_hair, 0.0, 1.0)
        for i in range(5):
            u = (i + 0.5) / 5.0
            th = np.pi * (0.22 + 0.56 * u)
            x0 = float(st.head_w) * 0.98 * np.cos(th)
            y0 = 0.30 + 0.18 * np.sin(th)
            ln = float(rng.uniform(0.040, 0.090))
            p1 = (x0 + np.cos(th) * ln * 0.6, y0 + np.sin(th) * ln * 0.6)
            p2 = (x0 + np.cos(th) * ln + float(rng.uniform(-0.020, 0.020)),
                  y0 + np.sin(th) * ln)
            fly = S.lock([(x0, y0), p1, p2], w0=0.0016, w1=0.0004,
                         bulge=0.003, per_segment=4, resample=20)
            brush.paint(canvas, fly, h.hair_fine, clip=outside,
                        alpha=0.30, soft=0.4)
