"""Regalia: the uniform, the insignia, the awards and the things a sitter wears.

A portrait is a person *in an outfit*, and an outfit is not one garment.  It is a
dress code (civilian, service, ceremonial, clerical), the marks that say which
service and which rank, the decorations earned, and the personal things worn on
top of it -- a watch chain, an aiguillette, a cross, an amulet.  Keeping those
separate is what lets the same face be a civilian in one portrait and a colonel in
another without redrawing the person.

The rules that keep this from becoming a costume shop:

* **The genome still owns the cloth.**  A service colour is a mix of a fixed dress
  colour and the sitter's own ``cloth`` gene, so a regiment has a uniform but two
  soldiers do not have identical fabric.
* **Everything is geometry, painted after the bust.**  No image is sampled, no
  symbol is traced.  A medal is a disc, a ribbon is a rectangle, a star is eight
  points, an aiguillette is a braid of tapered locks.  At avatar size that is the
  right level of abstraction -- a heraldic charge, not a photograph of one.
* **A mark is clipped to the thing it sits on.**  Epaulettes are clipped to the
  bust, earrings to the skull, a pendant to the neck and chest.  A decoration can
  therefore never float in the background or shift a feature.
* **Nothing is random except the draw.**  Ribbon colours come from a fixed,
  deliberate palette indexed by culture; rank comes from the decorations.  The
  same genome always paints the same uniform.

The genes are categorical and appended after the phenotype layer, so an older
genome reads as a civilian with no decorations rather than failing to decode.
"""
from __future__ import annotations

import numpy as np

from core import palette

from . import brush, garments, head
from . import shape as S
from .hues import Hues

__all__ = [
    "ATTIRE", "INSIGNIA", "AWARDS", "ORNAMENT", "FAITH", "CATEGORIES", "DEFAULTS",
    "build", "dress", "random_values", "validate", "paint_bust", "paint_ears",
]

#: The dress code.  ``civil`` is ordinary clothing and keeps the shirt collar the
#: rest of the portrait pipeline already draws; the rest replace it with a
#: service collar and add the marks that go with it.
ATTIRE = ("civil", "military", "police", "naval", "ceremonial", "clerical")
#: Rank marks: collar tabs, shoulder boards, both, or neither.
INSIGNIA = ("none", "collar", "shoulder", "both")
#: Decorations on the chest, and the one armband.
AWARDS = ("none", "ribbon", "medal", "order", "armband")
#: Personal ornament, worn over the dress code.
ORNAMENT = ("none", "aiguillette", "chain", "brooch", "earrings")
#: A pendant, not a religion: the shape is the visible part.
FAITH = ("none", "cross", "amulet")

CATEGORIES = {
    "attire": ATTIRE,
    "insignia": INSIGNIA,
    "awards": AWARDS,
    "ornament": ORNAMENT,
    "faith": FAITH,
    "garment": garments.CUTS,
}

DEFAULTS = {
    "attire": "civil",
    "insignia": "none",
    "awards": "none",
    "ornament": "none",
    "faith": "none",
    "garment": "auto",
}

#: Dress colours, in OKLCH.  They are the *material* of a service, mixed with the
#: sitter's cloth gene so the uniform belongs to the character and not to a flag.
_SERVICE = {
    "military": (0.32, 0.042, 0.245),    # olive drab
    "police": (0.27, 0.040, 0.635),      # service navy
    "naval": (0.23, 0.036, 0.640),       # deeper navy
    "ceremonial": (0.29, 0.078, 0.020),  # deep maroon
    "clerical": (0.19, 0.006, 0.000),    # black
}

#: Branch colours.  A service is one colour; the arm of it is another.  The
#: mapping is by culture, because a lineage's dress tradition is already a gene.
_BRANCH = {
    "north": (0.54, 0.095, 0.600),   # blue
    "river": (0.51, 0.130, 0.035),   # red
    "steppe": (0.52, 0.140, 0.030),  # red
    "coast": (0.48, 0.100, 0.285),   # green
    "desert": (0.55, 0.120, 0.075),  # ochre red
    "highland": (0.47, 0.100, 0.305),  # green
}

#: Ribbon faces.  Muted enough to sit under a face, distinct enough to read as a
#: ribbon bar rather than a coloured strip.
_RIBBONS = (
    (0.47, 0.135, 0.030),   # crimson
    (0.36, 0.080, 0.625),   # blue
    (0.44, 0.075, 0.300),   # green
    (0.72, 0.095, 0.145),   # gold
    (0.90, 0.005, 0.300),   # white
)


def _get(genes: dict, key: str) -> str:
    value = str(genes.get(key, DEFAULTS[key]))
    return value if value in CATEGORIES[key] else DEFAULTS[key]


class Kit:
    """The colours and marks one sitter's outfit needs, resolved once."""

    __slots__ = (
        "attire", "insignia", "awards", "ornament", "faith", "garment", "cut",
        "rank", "adult",
        "service", "service_hi", "service_light", "service_shadow", "service_line",
        "accent", "accent_hi", "accent_shadow", "tab", "tab_edge",
        "gold", "gold_hi", "gold_shadow", "silver", "silver_hi", "silver_shadow",
        "metal", "metal_hi", "metal_shadow", "ribbon", "rim",
    )


def build(genes: dict, h: Hues) -> Kit:
    """Resolve a genome's outfit into colours.  Pure; no RNG, no image reads."""
    k = Kit()
    k.attire = _get(genes, "attire")
    k.insignia = _get(genes, "insignia")
    k.awards = _get(genes, "awards")
    k.ornament = _get(genes, "ornament")
    k.faith = _get(genes, "faith")
    k.garment = _get(genes, "garment")
    k.cut = garments.resolve(k.attire, k.garment)
    k.adult = str(genes.get("age", "adult")) != "child"

    culture = str(genes.get("culture", "river"))
    branch = palette.oklch_to_srgb(_BRANCH.get(culture, _BRANCH["river"]))

    if k.attire == "civil":
        k.service = np.asarray(h.cloth, dtype=np.float64)
        k.service_hi = np.asarray(h.cloth_hi, dtype=np.float64)
        k.service_light = palette.lighten(h.cloth_hi, 0.040)
        k.service_shadow = np.asarray(h.cloth_shadow, dtype=np.float64)
        k.service_line = np.asarray(h.cloth_line, dtype=np.float64)
        k.accent = np.asarray(h.cloth_trim, dtype=np.float64)
        k.tab = np.asarray(h.cloth_trim, dtype=np.float64)
        k.tab_edge = np.asarray(h.metal, dtype=np.float64)
        k.gold = np.asarray(h.metal, dtype=np.float64)
    else:
        base = palette.oklch_to_srgb(_SERVICE[k.attire])
        k.service = palette.mix(base, h.cloth, 0.16)
        k.service_hi = palette.lighten(k.service, 0.045)
        k.service_light = palette.lighten(k.service, 0.085)
        k.service_shadow = palette.darken(k.service, 0.070)
        k.service_line = palette.darken(k.service, 0.180)
        if k.attire == "police":
            k.accent = palette.oklch_to_srgb((0.50, 0.095, 0.625))
            k.tab = palette.mix(k.service, k.accent, 0.55)
            k.tab_edge = palette.oklch_to_srgb((0.78, 0.012, 0.620))
            k.gold = palette.oklch_to_srgb((0.74, 0.100, 0.135))
        elif k.attire == "naval":
            k.accent = palette.oklch_to_srgb((0.76, 0.105, 0.135))
            k.tab = palette.darken(k.service, 0.030)
            k.tab_edge = k.accent
            k.gold = k.accent
        elif k.attire == "clerical":
            k.accent = palette.oklch_to_srgb((0.93, 0.004, 0.100))
            k.tab = k.accent
            k.tab_edge = palette.oklch_to_srgb((0.78, 0.012, 0.620))
            k.gold = palette.oklch_to_srgb((0.78, 0.012, 0.620))
        elif k.attire == "ceremonial":
            k.accent = palette.oklch_to_srgb((0.50, 0.130, 0.030))
            k.tab = palette.mix(k.service, branch, 0.40)
            k.tab_edge = palette.oklch_to_srgb((0.78, 0.105, 0.135))
            k.gold = k.tab_edge
        else:  # military
            k.accent = branch
            k.tab = palette.mix(k.service, branch, 0.50)
            k.tab_edge = palette.oklch_to_srgb((0.76, 0.100, 0.135))
            k.gold = k.tab_edge

    k.accent_hi = palette.lighten(k.accent, 0.060)
    k.accent_shadow = palette.darken(k.accent, 0.070)
    k.gold_hi = palette.lighten(k.gold, 0.100)
    k.gold_shadow = palette.darken(k.gold, 0.090)
    k.silver = palette.oklch_to_srgb((0.78, 0.012, 0.620))
    k.silver_hi = palette.lighten(k.silver, 0.080)
    k.silver_shadow = palette.darken(k.silver, 0.100)
    # The metal a mark is cut from: silver for police and clergy, gold otherwise.
    if k.attire in ("police", "clerical"):
        k.metal, k.metal_hi, k.metal_shadow = k.silver, k.silver_hi, k.silver_shadow
    else:
        k.metal, k.metal_hi, k.metal_shadow = k.gold, k.gold_hi, k.gold_shadow
    k.ribbon = palette.oklch_to_srgb(
        _RIBBONS[(ATTIRE.index(k.attire) + list(_BRANCH).index(culture)) % len(_RIBBONS)])
    # Rank follows the decorations rather than inventing a second axis: an
    # undecorated service dress has one bar, a decorated one two, an order three.
    k.rank = 1 + (k.awards != "none") + (k.awards in ("medal", "order"))
    k.rim = k.metal
    return k


def dress(kit: Kit, h: Hues) -> Hues:
    """The colour script with the garment replaced by the kit's service cloth.

    The bust and its detail pass are already painted from ``h``; swapping the cloth
    slots is what makes the *whole* tunic one material instead of a service-coloured
    patch pasted over a civilian shirt.  Civilian dress returns the same object.
    """
    if kit.attire == "civil":
        return h
    kw = {name: getattr(h, name) for name in Hues.__slots__}
    kw["cloth"] = kit.service
    kw["cloth_hi"] = kit.service_hi
    kw["cloth_shadow"] = kit.service_shadow
    kw["cloth_line"] = kit.service_line
    kw["cloth_trim"] = kit.accent
    kw["metal"] = kit.metal
    return Hues(**kw)


def random_values(rng: np.random.Generator) -> dict:
    """Draw the outfit layer.  Called after the phenotype layer, never before."""
    attire = str(rng.choice(ATTIRE, p=(0.46, 0.16, 0.12, 0.10, 0.08, 0.08)))
    return {
        "attire": attire,
        "insignia": str(rng.choice(INSIGNIA, p=(0.30, 0.24, 0.28, 0.18))),
        "awards": str(rng.choice(AWARDS, p=(0.46, 0.18, 0.16, 0.10, 0.10))),
        "ornament": str(rng.choice(ORNAMENT, p=(0.48, 0.12, 0.16, 0.12, 0.12))),
        "faith": str(rng.choice(FAITH, p=(0.66, 0.22, 0.12))),
        "garment": garments.random_value(rng, attire),
    }


def validate(genes: dict) -> None:
    for key, choices in CATEGORIES.items():
        value = str(genes.get(key, DEFAULTS[key]))
        if value not in choices:
            raise ValueError(f"unknown {key} {value!r}")


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------
def _star(r: float, cx: float, cy: float, points: int = 8) -> S.Shape:
    """A heraldic star: alternating long and short radii, flat shaded later."""
    n = points * 2
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rad = np.where(np.arange(n) % 2 == 0, r, r * 0.40)
    return S.poly(np.stack([cx + rad * np.cos(a), cy + rad * np.sin(a)], axis=1))


def _spine_at(spine, t: float):
    a = np.asarray(spine, dtype=np.float64)
    d = np.linalg.norm(np.diff(a, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    u /= max(u[-1], 1e-9)
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(a) - 1):
        if u[i] <= t <= u[i + 1]:
            f = (t - u[i]) / max(u[i + 1] - u[i], 1e-9)
            return a[i] + (a[i + 1] - a[i]) * f
    return a[-1]


def _normal(spine, t: float):
    p = _spine_at(spine, t)
    q = _spine_at(spine, min(1.0, t + 0.03))
    d = q - p
    n = np.array([-d[1], d[0]])
    return p, n / max(float(np.linalg.norm(n)), 1e-9)


# --------------------------------------------------------------------------
# the bust
# --------------------------------------------------------------------------
def paint_bust(canvas, st, h: Hues, genes: dict, kit: Kit, kind: str, clip) -> None:
    """The dress code, its marks and its decorations, over a finished bust."""
    if clip is None:
        return
    collar_clip = clip
    if kit.attire != "civil":
        # A collar sits around the neck, and the bust's own clip stops at the
        # neckline.  The neck coverage is added only for the collar pass, so the
        # threaded edge cannot draw over the jaw.
        collar_clip = np.clip(clip + brush.mask(canvas, head.neck(st)), 0.0, 1.0)
    # The order is the order it is put on: tunic details, then the cut of the
    # garment over them, then the sash, collar and boards, then the overgarment,
    # and last the decorations that are pinned on top of everything.
    if kit.attire != "civil":
        _tunic(canvas, st, h, kit, clip)
    garments.paint_front(canvas, st, h, kit, clip)
    if kit.awards == "order":
        _sash(canvas, st, kit, clip)
    if kit.attire != "civil":
        _collar(canvas, st, h, kit, collar_clip)
    _insignia(canvas, st, h, kit, collar_clip)
    garments.paint_mantle(canvas, st, h, kit, clip)
    _awards(canvas, st, h, kit, clip)
    _ornament(canvas, st, h, kit, collar_clip)
    _faith(canvas, st, h, kit, collar_clip)


def _collar(canvas, st, h, kit: Kit, clip) -> None:
    """A stand collar for service dress; a white band and black tab for clergy.

    The civilian collar leaves are not covered with a patch -- the tunic pass in
    :mod:`gen.portraits.detail` skips them for a uniform -- so this is the only
    collar on a service bust.  The band tapers with the neck: wider where it
    meets the tunic, narrower at the throat.  A constant-width rectangle reads as
    a neck brace; the taper is what makes it tailoring.
    """
    lw = float(st.ink_w) * float(st.line)
    top = float(st.neck_y)
    nw = float(st.neck_w)
    clerical = kit.attire == "clerical"
    if clerical:
        col = palette.oklch_to_srgb((0.93, 0.004, 0.100))
        col_hi = palette.lighten(col, 0.025)
        col_sh = palette.darken(col, 0.075)
        edge = kit.service_line
    else:
        # The band is a slightly lighter plane of the tunic's own cloth.  At
        # 256 px the same colour as the body disappears and the tabs float: the
        # collar has to be a *plane* before it can be a collar.
        col = kit.service_light
        col_hi = palette.lighten(col, 0.030)
        col_sh = kit.service_shadow
        edge = kit.service_line

    xt = min(nw * 0.72, 0.128)      # half-width at the throat
    xb = min(nw * 0.98, 0.172)      # half-width at the tunic
    yt = top + 0.078                # upper edge, under the jaw
    yb = top - 0.006                # lower edge, on the tunic
    band = S.poly([(-xb, yb), (-xt, yt), (0.0, yt - 0.008), (xt, yt),
                   (xb, yb), (0.0, yb + 0.002)], resample=48)

    # The shadow the collar throws down onto the tunic, painted first.
    brush.paint(canvas, S.box(xb * 1.02, 0.020, 0.0, 0.0, yb - 0.018),
                kit.service_shadow, clip=clip, alpha=0.20, soft=2.2)
    brush.paint(canvas, band, col, clip=clip, alpha=1.0, soft=0.45)
    # Two planes: the upper edge is in the jaw's shadow, the lower fold catches
    # the light.  A flat band is a strip; two planes make it cloth.
    brush.paint(canvas, band & S.box(xb * 1.2, 0.018, 0.0, 0.0, yt - 0.008),
                col_sh, clip=clip, alpha=0.34, soft=0.9)
    brush.paint(canvas, band & S.box(xb * 1.2, 0.014, 0.0, 0.0, yb + 0.012),
                col_hi, clip=clip, alpha=0.38, soft=0.7)
    brush.paint(canvas, band.outline(lw * 0.55), edge, clip=clip,
                alpha=0.54, soft=0.4)
    if kit.attire in ("naval", "ceremonial"):
        # A gold lace along the lower edge: the one line that says "dress"
        # rather than "fatigue" at avatar size.
        brush.paint(canvas, band & S.box(xb * 1.2, 0.0032, 0.0, 0.0, yb + 0.026),
                    kit.gold, clip=clip, alpha=0.88, soft=0.35)

    # The centre front: two halves of a stand collar meet in a seam.
    seam = S.lock([(0.0, yb + 0.006), (0.0, yt - 0.006)],
                  w0=0.0022, w1=0.0022, per_segment=3)
    brush.paint(canvas, seam, kit.service_line, clip=clip, alpha=0.42, soft=0.35)

    if clerical:
        # The Roman collar: a white band with a black tab in the notch.  It is the
        # whole garment in one shape, and it reads at 64 px.
        cy = (yt + yb) * 0.5
        tab = S.box(0.016, 0.024, 0.005, 0.0, cy)
        brush.paint(canvas, tab, kit.service, clip=clip, alpha=0.98, soft=0.40)
        brush.paint(canvas, tab.outline(0.0026), kit.service_line, clip=clip,
                    alpha=0.52, soft=0.25)
        brush.paint(canvas, S.lock([(0.0, cy + 0.024), (0.0, cy - 0.021)],
                                   w0=0.0042, w1=0.0032, per_segment=3),
                    palette.lighten(kit.service, 0.020), clip=clip,
                    alpha=0.30, soft=0.7)


def _tunic(canvas, st, h, kit: Kit, clip) -> None:
    """The details that make the torso a jacket and not a coloured field."""
    lw = float(st.ink_w) * float(st.line)
    top = float(st.neck_y)
    sw = float(st.shoulder_w)

    if kit.attire == "naval":
        # Double-breasted: two columns of gold buttons.  The centre placket drawn
        # by the detail pass is covered by them, which is the point.
        for side in (+1, -1):
            for k in range(3):
                x = side * sw * 0.095
                y = top - 0.090 - 0.056 * k
                disc = S.circle(0.011, x, y)
                brush.paint(canvas, disc, kit.gold, clip=clip, alpha=0.94, soft=0.25)
                brush.paint(canvas, disc.outline(0.0026), kit.gold_shadow,
                            clip=clip, alpha=0.80, soft=0.2)
                brush.paint(canvas, S.circle(0.0035, x - 0.003, y + 0.003),
                            kit.gold_hi, clip=clip, alpha=0.9, soft=0.3)
    elif kit.attire in ("military", "police"):
        # A breast pocket on the side the decorations do not occupy.  A flap and a
        # single button are enough; a pocket outline alone reads as a decal.
        px = -sw * 0.255
        py = top - 0.150
        pocket = S.box(0.066, 0.044, 0.006, px, py)
        brush.paint(canvas, pocket, kit.service_shadow, clip=clip,
                    alpha=0.15, soft=2.0)
        brush.paint(canvas, pocket.outline(lw * 0.40), kit.service_line,
                    clip=clip, alpha=0.34, soft=0.5)
        flap = S.box(0.069, 0.013, 0.004, px, py + 0.040)
        brush.paint(canvas, flap, kit.service_hi, clip=clip, alpha=0.88, soft=0.35)
        brush.paint(canvas, flap.outline(lw * 0.32), kit.service_line, clip=clip,
                    alpha=0.30, soft=0.4)
        button = S.circle(0.0065, px, py + 0.030)
        brush.paint(canvas, button, kit.metal, clip=clip, alpha=0.90, soft=0.25)
        if kit.attire == "police" and kit.adult:
            # A shield above the pocket: the one civic mark that is not a rank.
            # A child in a police tunic is a costume; the badge is what makes it
            # a false claim, so the age gate lives here as well as in the genome.
            bx, by = -sw * 0.235, top - 0.072
            shield = S.poly([(bx - 0.022, by + 0.024), (bx + 0.022, by + 0.024),
                             (bx + 0.019, by - 0.008), (bx, by - 0.030),
                             (bx - 0.019, by - 0.008)], resample=24)
            brush.paint(canvas, shield, kit.silver, clip=clip, alpha=0.95, soft=0.30)
            brush.paint(canvas, shield.outline(0.0028), kit.silver_shadow,
                        clip=clip, alpha=0.70, soft=0.25)
            brush.paint(canvas, S.circle(0.008, bx, by + 0.002), kit.accent,
                        clip=clip, alpha=0.85, soft=0.25)


def _insignia(canvas, st, h, kit: Kit, clip) -> None:
    # A Roman collar is a complete garment: boards or tabs on it are a contradiction
    # of the dress code, not a decoration.  This is a painter invariant, so a forced
    # genome cannot produce it either.
    if kit.attire == "clerical":
        return
    if kit.insignia in ("collar", "both"):
        _collar_tabs(canvas, st, h, kit, clip)
    if kit.insignia in ("shoulder", "both"):
        _epaulettes(canvas, st, h, kit, clip)


def _collar_tabs(canvas, st, h, kit: Kit, clip) -> None:
    lw = float(st.ink_w) * float(st.line)
    top = float(st.neck_y)
    y = top + 0.036
    for side in (+1, -1):
        cx = side * 0.060
        tab = S.poly([(cx - 0.024, y - 0.017), (cx + 0.026, y - 0.011),
                      (cx + 0.026, y + 0.016), (cx - 0.024, y + 0.012)],
                     resample=20)
        brush.paint(canvas, tab, kit.tab, clip=clip, alpha=1.0, soft=0.30)
        brush.paint(canvas, tab.outline(lw * 0.36), kit.tab_edge, clip=clip,
                    alpha=0.82, soft=0.30)
        # A tiny button at the outer end is the difference between a tab and a
        # painted rectangle.
        brush.paint(canvas, S.circle(0.0055, cx + side * 0.018, y),
                    kit.metal, clip=clip, alpha=0.92, soft=0.25)
        # A branch stripe down the middle.
        brush.paint(canvas, S.lock([(cx + side * 0.001, y + 0.015),
                                    (cx - side * 0.002, y - 0.015)],
                                   w0=0.0028, w1=0.0022, per_segment=3),
                    kit.accent_hi, clip=clip, alpha=0.65, soft=0.3)


def _epaulettes(canvas, st, h, kit: Kit, clip) -> None:
    # A shoulder cape is worn *over* the boards.  Painting them and trusting the
    # cape's anti-aliased edge to hide them leaves gold arcs on the shoulders, so
    # the correct thing is not to dress the shoulder at all under a mantle.
    if kit.cut in ("mantle", "habit"):
        return
    lw = float(st.ink_w) * float(st.line)
    top = float(st.neck_y)
    nw = float(st.neck_w)
    sw = float(st.shoulder_w)
    for side in (+1, -1):
        # A board lies on the *shoulder*, and the shoulder is the top edge of the
        # bust -- not a horizontal line near the chest.  Sample that edge at four
        # stations and keep the board's upper half inside the silhouette; the old
        # fixed anchors put the whole board on the upper chest, which is what made
        # it look like a lapel that had slipped off the shoulder.
        x0 = max(nw * 1.03, 0.185)
        x1 = sw * 0.520
        xs = [x0, x0 + (x1 - x0) * 0.34, x0 + (x1 - x0) * 0.68, x1]
        spine = [(side * x, head.shoulder_edge_y(st, side * x) - 0.034) for x in xs]
        a = spine[0]
        board = S.lock(spine, w0=0.062, w1=0.054, bulge=0.005, per_segment=7)
        # A shadow under the free end gives the board thickness.  The board sits
        # fully inside the shoulder silhouette, so the shadow has to be broad
        # enough to survive the cel shading underneath it.
        brush.paint(canvas, S.lock([(p[0], p[1] - 0.017) for p in spine],
                                   w0=0.068, w1=0.060, per_segment=6),
                    kit.service_shadow, clip=clip, alpha=0.30, soft=1.6)
        # The field is one step lighter than the tunic; without that step the
        # board disappeared into the shoulder and only its gold piping read.
        brush.paint(canvas, board, kit.service_light, clip=clip, alpha=1.0, soft=0.42)
        brush.paint(canvas, board & S.lock([(p[0], p[1] + 0.019) for p in spine],
                                          w0=0.012, w1=0.009, per_segment=5),
                    palette.lighten(kit.service_light, 0.035), clip=clip,
                    alpha=0.30, soft=0.8)
        # A dark seam outside a gold braid: the two outlines give the board an
        # edge at avatar size, where a single thin line would alias away.
        brush.paint(canvas, board.outline(lw * 0.58), kit.service_line, clip=clip,
                    alpha=0.58, soft=0.4)
        brush.paint(canvas, board.outline(lw * 0.28), kit.metal, clip=clip,
                    alpha=0.78, soft=0.32)
        # Braid and a burnished top edge.  A board is a piece of metal-cloth
        # tailoring, not a flat lozenge; the narrow cord and the lit rim are what
        # keep it from reading as a yellow stripe at 384 px.
        for k in (-1, 0, 1):
            cord = S.lock([(px, py + 0.0055 * k) for px, py in spine],
                          w0=0.0032, w1=0.0028, per_segment=6)
            brush.paint(canvas, cord, kit.gold_shadow if k else kit.gold,
                        clip=clip, alpha=0.48 if k else 0.78, soft=0.25)
        brush.paint(canvas, board & S.lock(
            [(px, py + 0.020) for px, py in spine],
            w0=0.0050, w1=0.0040, per_segment=6),
            kit.metal_hi, clip=clip, alpha=0.50, soft=0.30)
        # Piping along the outer edge, and the button at the inner end.
        brush.paint(canvas, S.lock([(p[0], p[1] - 0.029) for p in spine],
                                   w0=0.0042, w1=0.0036, per_segment=5),
                    kit.metal, clip=clip, alpha=0.90, soft=0.32)
        brush.paint(canvas, S.circle(0.012, a[0], a[1]), kit.metal, clip=clip,
                    alpha=0.95, soft=0.25)
        brush.paint(canvas, S.circle(0.012, a[0], a[1]).outline(0.0026),
                    kit.metal_shadow, clip=clip, alpha=0.75, soft=0.2)
        # Rank: stars along the outer half of the board.
        for k in range(kit.rank):
            t = 0.58 + 0.13 * k
            p, n = _normal(spine, t)
            brush.paint(canvas, _star(0.0170, float(p[0]), float(p[1]), 4),
                        kit.metal_shadow, clip=clip, alpha=0.65, soft=0.25)
            brush.paint(canvas, _star(0.0135, float(p[0]), float(p[1]) + 0.001, 4),
                        kit.metal, clip=clip, alpha=0.94, soft=0.25)
            brush.paint(canvas, _star(0.0062, float(p[0]), float(p[1]) + 0.001, 4),
                        kit.metal_hi, clip=clip, alpha=0.65, soft=0.2)


# --------------------------------------------------------------------------
# awards
# --------------------------------------------------------------------------
def _awards(canvas, st, h, kit: Kit, clip) -> None:
    if kit.awards == "none":
        return
    top = float(st.neck_y)
    sw = float(st.shoulder_w)
    ax = sw * 0.300
    ay = top - 0.150
    if kit.awards == "ribbon":
        _ribbon_bar(canvas, st, kit, ax, ay, clip)
    elif kit.awards == "medal":
        _medals(canvas, st, kit, ax, ay, clip)
    elif kit.awards == "order":
        _order_star(canvas, st, kit, ax, ay, clip)
    elif kit.awards == "armband":
        _armband(canvas, st, kit, clip)


def _ribbon_bar(canvas, st, kit: Kit, ax: float, ay: float, clip) -> None:
    for row in range(3):
        y = ay + 0.022 - row * 0.019
        bar = S.box(0.074, 0.0082, 0.0025, ax, y)
        base = kit.ribbon if row % 2 == 0 else kit.metal_shadow
        if row == 1:
            base = palette.oklch_to_srgb(_RIBBONS[2])
        brush.paint(canvas, bar, base, clip=clip, alpha=0.96, soft=0.18)
        # Three vertical stripes: the ribbon's own pattern, not an outline.
        for k, col in enumerate((palette.oklch_to_srgb(_RIBBONS[4]), kit.accent,
                                 palette.oklch_to_srgb(_RIBBONS[0]))):
            x = ax - 0.038 + k * 0.038
            stripe = bar & S.box(0.0085, 0.0082, 0.0, x, y)
            brush.paint(canvas, stripe, col, clip=clip, alpha=0.78, soft=0.15)
        brush.paint(canvas, bar.outline(0.0016), kit.metal_shadow, clip=clip,
                    alpha=0.55, soft=0.2)


def _medals(canvas, st, kit: Kit, ax: float, ay: float, clip) -> None:
    lw = float(st.ink_w) * float(st.line)
    for k in range(2):
        cx = ax - 0.032 + 0.064 * k
        ry = ay + 0.060 - 0.012 * k
        rib = S.poly([(cx - 0.020, ry), (cx + 0.020, ry),
                      (cx + 0.014, ry - 0.062), (cx - 0.014, ry - 0.062)],
                     resample=20)
        brush.paint(canvas, rib, kit.ribbon if k == 0 else kit.accent, clip=clip,
                    alpha=0.96, soft=0.25)
        brush.paint(canvas, rib & S.box(0.005, 0.034, 0.0, cx - 0.006, ry - 0.030),
                    palette.oklch_to_srgb(_RIBBONS[4]), clip=clip, alpha=0.72,
                    soft=0.2)
        brush.paint(canvas, rib & S.box(0.004, 0.050, 0.006, cx + 0.008, ry - 0.030),
                    palette.lighten(kit.ribbon if k == 0 else kit.accent, 0.07),
                    clip=clip, alpha=0.42, soft=0.30)
        brush.paint(canvas, rib.outline(lw * 0.22), kit.metal_shadow, clip=clip,
                    alpha=0.60, soft=0.2)
        disc = S.circle(0.026, cx, ry - 0.082)
        # The disc is a struck medal: a cast shadow, a struck rim, a bright
        # bevel on the lit side, and the star sitting in the field.  Flat gold
        # was the difference between a medal and a yellow dot.
        brush.paint(canvas, S.circle(0.029, cx + 0.002, ry - 0.086),
                    kit.service_shadow, clip=clip, alpha=0.24, soft=1.2)
        brush.paint(canvas, disc, kit.metal, clip=clip, alpha=0.98, soft=0.25)
        brush.paint(canvas, disc.outline(0.0038), kit.metal_shadow, clip=clip,
                    alpha=0.88, soft=0.22)
        brush.paint(canvas, disc - S.circle(0.021, cx, ry - 0.082),
                    kit.metal_hi, clip=clip, alpha=0.34, soft=0.40)
        brush.paint(canvas, _star(0.013, cx, ry - 0.082, 5), kit.metal_shadow,
                    clip=clip, alpha=0.42, soft=0.30)
        brush.paint(canvas, _star(0.0105, cx, ry - 0.0805, 5), kit.metal_hi,
                    clip=clip, alpha=0.82, soft=0.24)
        brush.paint(canvas, S.circle(0.0048, cx - 0.008, ry - 0.074),
                    kit.metal_hi, clip=clip, alpha=0.95, soft=0.22)


def _sash(canvas, st, kit: Kit, clip) -> None:
    """The broad ribbon of an order, worn shoulder to hip.

    It is the one decoration that changes the silhouette of the bust, so it is
    cloth with a fold and a gold edge, not a coloured stripe.  The top end starts
    just under the collar and the epaulette; both are painted after it, so the
    sash appears to come out from under them.
    """
    top = float(st.neck_y)
    sw = float(st.shoulder_w)
    spine = [(-sw * 0.29, top - 0.020), (-sw * 0.08, top - 0.210),
             (sw * 0.34, top - 0.480)]
    sash = S.lock(spine, w0=0.062, w1=0.072, bulge=0.008, per_segment=8)
    # A maroon tunic and a maroon sash are one object.  The ceremonial sash is
    # therefore blue against the maroon, which is both historical and readable.
    col = (palette.oklch_to_srgb((0.39, 0.095, 0.630))
           if kit.attire == "ceremonial" else kit.accent)
    brush.paint(canvas, sash, col, clip=clip, alpha=0.94, soft=0.55)
    # One broad fold, just off the centre, so the band has a length and a light.
    brush.paint(canvas, sash & S.lock([(p[0] - 0.014, p[1] + 0.004) for p in spine],
                                      w0=0.010, w1=0.013, per_segment=7),
                palette.darken(col, 0.080), clip=clip, alpha=0.24, soft=1.3)
    # The gold border is the order's own edge, not a second band.
    brush.paint(canvas, sash.outline(0.0038), kit.gold, clip=clip,
                alpha=0.88, soft=0.32)
    brush.paint(canvas, sash.outline(0.0020), kit.gold_hi, clip=clip,
                alpha=0.35, soft=0.3)


def _order_star(canvas, st, kit: Kit, ax: float, ay: float, clip) -> None:
    """The breast star: eight points, a jewelled centre, a bright top facet."""
    star = _star(0.044, ax, ay + 0.008, 8)
    brush.paint(canvas, star, kit.gold, clip=clip, alpha=0.98, soft=0.30)
    brush.paint(canvas, star.outline(0.0026), kit.gold_shadow, clip=clip,
                alpha=0.72, soft=0.25)
    brush.paint(canvas, S.circle(0.014, ax, ay + 0.008), kit.accent, clip=clip,
                alpha=0.92, soft=0.3)
    brush.paint(canvas, S.circle(0.006, ax - 0.003, ay + 0.014), kit.gold_hi,
                clip=clip, alpha=0.9, soft=0.3)


def _armband(canvas, st, kit: Kit, clip) -> None:
    lw = float(st.ink_w) * float(st.line)
    top = float(st.neck_y)
    sw = float(st.shoulder_w)
    # The upper arm is at the edge of a headshot, so the brassard is placed on the
    # visible deltoid, inside the shoulder seam, and follows the sleeve's slope.
    # A white field on a dark sleeve is the readable version; a dark band on dark
    # cloth is a smudge.
    cx = sw * 0.545
    cy = top - 0.150
    band = S.poly([(cx - 0.028, cy + 0.058), (cx + 0.026, cy + 0.050),
                   (cx + 0.030, cy - 0.050), (cx - 0.024, cy - 0.058)],
                  resample=24)
    field = palette.oklch_to_srgb((0.50, 0.160, 0.035))
    brush.paint(canvas, band, field, clip=clip, alpha=0.95, soft=0.35)
    brush.paint(canvas, band.outline(lw * 0.36), palette.darken(field, 0.20),
                clip=clip, alpha=0.48, soft=0.3)
    disc = S.circle(0.019, cx, cy)
    brush.paint(canvas, disc, palette.oklch_to_srgb((0.93, 0.005, 0.100)),
                clip=clip, alpha=0.95, soft=0.3)
    brush.paint(canvas, disc.outline(0.0028), palette.darken(field, 0.16),
                clip=clip, alpha=0.50, soft=0.25)
    brush.paint(canvas, S.circle(0.0075, cx, cy), field, clip=clip,
                alpha=0.88, soft=0.25)


# --------------------------------------------------------------------------
# worn things
# --------------------------------------------------------------------------
def _ornament(canvas, st, h, kit: Kit, clip) -> None:
    if kit.ornament == "none":
        return
    if kit.ornament == "aiguillette" and kit.cut not in ("mantle", "habit"):
        _aiguillette(canvas, st, kit, clip)
    elif kit.ornament == "chain":
        _watch_chain(canvas, st, kit, clip)
    elif kit.ornament == "brooch":
        _brooch(canvas, st, kit, clip)


def _aiguillette(canvas, st, kit: Kit, clip) -> None:
    """A braided shoulder cord: one U-shaped loop and two falling tips.

    The braid is three parallel cords with a highlight on the middle one.  Several
    separate loops crossing each other read as rope; one loop with a visible
    attachment at the collar reads as a cord.
    """
    top = float(st.neck_y)
    sw = float(st.shoulder_w)
    nw = float(st.neck_w)
    loop = [(-nw * 1.04, top + 0.052), (-sw * 0.265, top + 0.016),
            (-sw * 0.335, top - 0.040), (-sw * 0.310, top - 0.088),
            (-sw * 0.255, top - 0.098), (-sw * 0.225, top - 0.068)]
    for k in (-1, 0, 1):
        path = [(x, y + 0.0060 * k) for x, y in loop]
        cord = S.lock(path, w0=0.0046, w1=0.0042, bulge=0.0015, per_segment=9)
        brush.paint(canvas, cord, kit.gold if k == 0 else kit.gold_shadow,
                    clip=clip, alpha=0.92 if k == 0 else 0.72, soft=0.35)
    brush.paint(canvas, S.lock([(x, y + 0.0066) for x, y in loop],
                               w0=0.0018, w1=0.0016, per_segment=9),
                kit.gold_hi, clip=clip, alpha=0.85, soft=0.3)
    # Two metal tips hang from the lowest span of the loop.
    for tx, ty in ((-sw * 0.300, top - 0.096), (-sw * 0.262, top - 0.104)):
        tip = S.lock([(tx, ty), (tx - 0.004, ty - 0.042)],
                     w0=0.0072, w1=0.0048, per_segment=3)
        brush.paint(canvas, tip, kit.gold, clip=clip, alpha=0.95, soft=0.30)
        brush.paint(canvas, tip.outline(0.0016), kit.gold_shadow, clip=clip,
                    alpha=0.60, soft=0.2)
        brush.paint(canvas, S.circle(0.0052, tx - 0.004, ty - 0.044),
                    kit.gold_hi, clip=clip, alpha=0.90, soft=0.3)


def _watch_chain(canvas, st, kit: Kit, clip) -> None:
    top = float(st.neck_y)
    sw = float(st.shoulder_w)
    # A pocket-watch chain is a catenary, not a necklace: it belongs to the
    # waistcoat and the hand, and that is what distinguishes it from the pendant.
    spine = [(0.014, top - 0.115), (sw * 0.085, top - 0.155),
             (sw * 0.170, top - 0.155), (sw * 0.205, top - 0.120)]
    cord = S.lock(spine, w0=0.0040, w1=0.0040, bulge=0.0035, per_segment=8)
    brush.paint(canvas, cord, kit.gold, clip=clip, alpha=0.92, soft=0.35)
    brush.paint(canvas, S.lock([(p[0], p[1] + 0.0015) for p in spine],
                               w0=0.0014, w1=0.0014, per_segment=8),
                kit.gold_hi, clip=clip, alpha=0.8, soft=0.3)
    # The watch sits at the end of the chain, half turned away.
    cx, cy = sw * 0.205, top - 0.120
    watch = S.circle(0.020, cx, cy)
    brush.paint(canvas, watch, kit.gold, clip=clip, alpha=0.96, soft=0.3)
    brush.paint(canvas, watch.outline(0.0028), kit.gold_shadow, clip=clip,
                alpha=0.78, soft=0.25)
    brush.paint(canvas, S.circle(0.014, cx, cy), palette.oklch_to_srgb(
        (0.93, 0.004, 0.12)), clip=clip, alpha=0.92, soft=0.25)
    brush.paint(canvas, S.lock([(cx, cy + 0.007), (cx, cy + 0.001),
                                (cx + 0.005, cy - 0.003)],
                               w0=0.0016, w1=0.0010, per_segment=3),
                kit.gold_shadow, clip=clip, alpha=0.85, soft=0.2)


def _brooch(canvas, st, kit: Kit, clip) -> None:
    top = float(st.neck_y)
    nw = float(st.neck_w)
    # On the viewer's left lapel, where a brooch is actually pinned.
    cx = -min(nw * 1.02, 0.185)
    cy = top + 0.020
    ring = S.circle(0.020, cx, cy)
    brush.paint(canvas, ring, kit.gold, clip=clip, alpha=0.95, soft=0.3)
    brush.paint(canvas, ring.outline(0.0034), kit.gold_shadow, clip=clip,
                alpha=0.80, soft=0.25)
    brush.paint(canvas, S.circle(0.010, cx, cy), kit.accent_hi, clip=clip,
                alpha=0.92, soft=0.3)
    brush.paint(canvas, S.circle(0.0035, cx - 0.004, cy + 0.004), kit.gold_hi,
                clip=clip, alpha=0.95, soft=0.3)


def _faith(canvas, st, h, kit: Kit, clip) -> None:
    if kit.faith == "none":
        return
    top = float(st.neck_y)
    nw = float(st.neck_w)
    if kit.faith == "cross":
        # A fine chain, then a plain Latin cross.  The cross is metal because a
        # cross is jewellery as often as it is a sign.
        chain = S.lock([(-nw * 0.86, top - 0.012), (0.0, top - 0.095),
                        (nw * 0.86, top - 0.012)],
                       w0=0.0032, w1=0.0032, bulge=0.004, per_segment=8)
        brush.paint(canvas, chain, kit.gold, clip=clip, alpha=0.88, soft=0.35)
        cy = top - 0.152
        vert = S.box(0.0060, 0.026, 0.0015, 0.0, cy)
        horiz = S.box(0.0190, 0.0055, 0.0015, 0.0, cy + 0.011)
        brush.paint(canvas, vert | horiz, kit.gold, clip=clip, alpha=0.97,
                    soft=0.25)
        brush.paint(canvas, (vert | horiz).outline(0.0022), kit.gold_shadow,
                    clip=clip, alpha=0.72, soft=0.2)
        brush.paint(canvas, S.box(0.0025, 0.020, 0.001, -0.0022, cy + 0.001),
                    kit.gold_hi, clip=clip, alpha=0.55, soft=0.25)
    elif kit.faith == "amulet":
        # A cord and a bone disc: deliberately not metal, so the two faiths do
        # not read as the same pendant in a different colour.
        cord_col = palette.oklch_to_srgb((0.30, 0.035, 0.075))
        chain = S.lock([(-nw * 0.90, top - 0.014), (0.0, top - 0.092),
                        (nw * 0.90, top - 0.014)],
                       w0=0.0044, w1=0.0040, bulge=0.004, per_segment=8)
        brush.paint(canvas, chain, cord_col, clip=clip, alpha=0.90, soft=0.35)
        cy = top - 0.150
        disc = S.circle(0.026, 0.0, cy)
        bone = palette.oklch_to_srgb((0.88, 0.022, 0.135))
        brush.paint(canvas, disc, bone, clip=clip, alpha=0.96, soft=0.3)
        brush.paint(canvas, disc.outline(0.0032), palette.darken(bone, 0.18),
                    clip=clip, alpha=0.70, soft=0.25)
        brush.paint(canvas, S.circle(0.007, 0.0, cy + 0.005), cord_col,
                    clip=clip, alpha=0.88, soft=0.3)
        brush.paint(canvas, S.circle(0.003, -0.006, cy - 0.004),
                    palette.lighten(bone, 0.06), clip=clip, alpha=0.75, soft=0.3)


# --------------------------------------------------------------------------
# ears
# --------------------------------------------------------------------------
def paint_ears(canvas, st, h: Hues, genes: dict, turn: float, clip) -> None:
    """Earrings, painted on the ear before the hair has its say.

    The front hair is painted later, so a style that covers the ear hides the
    earring with it -- which is correct, and the reason this cannot be part of the
    bust pass.
    """
    kit = build(genes, h)
    if kit.ornament != "earrings":
        return
    sk = head.skull(st)
    for side in (+1, -1):
        if turn != 0.0 and sk.receding(side, turn):
            continue
        ex, ey = head.turn_point(side * st.head_w * 0.80, st.eye_y + 0.015, turn, st)
        r = st.head_w * 0.17 * float(getattr(st, "ear_size", 1.0))
        lx = float(ex) + side * r * 0.10
        ly = float(ey) - r * 0.86
        hoop = S.circle(0.013, lx, ly - 0.014)
        brush.paint(canvas, hoop.outline(0.0032), kit.metal, clip=clip,
                    alpha=0.92, soft=0.30)
        bead = S.circle(0.0065, lx, ly - 0.032)
        brush.paint(canvas, bead, kit.accent_hi, clip=clip, alpha=0.94, soft=0.30)
        brush.paint(canvas, S.circle(0.0022, lx - 0.002, ly - 0.030),
                    kit.metal_hi, clip=clip, alpha=0.90, soft=0.3)
