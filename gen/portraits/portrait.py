"""Composition: the layer order that turns parts into a face.

The **portrait** is a passport photograph: the sitter faces the camera, the head
is straight, and the expression is neutral.  ``turn`` and ``expression`` are still
genes in the genome, and the lower-level drawing supports both, but ``compose``
and ``render`` deliberately ignore them.  A game that wants a mood or a
three-quarter view later can call :func:`compose_posed` / :func:`render_posed`,
which is the separate API for that experiment; the passport path stays stable.

A portrait is painted in one order, and the order is most of the drawing:

1. the background;
2. the **back hair** -- behind the body, so it shows around the silhouette;
3. the **shoulders**, then the **neck**, with the head's cast shadow on it;
4. the **head**: flat skin, cel bands, the cast shade of the fringe, the ears,
   the outer contour, then the blush;
5. the **features**: nose, mouth, both eyes (socket, sclera, iris, rim, pupil,
   lid shadow, highlights, lash), brows;
6. the **front hair**: the fringe and the sidelocks, painted last, because hair
   lies over the face -- this is what stops a fringe looking like a painted-on
   headband.

Everything is a pure function of the genome: the random variation inside the hair
comes from a generator seeded by the genome's own string, so the same genes always
give the same pixels and there is no hidden second input.

The head is built once and passed down.  It is the largest silhouette in the
picture and the cheapest thing to recompute *wrongly*, so it is recomputed not at
all.

There are two styles in here and the difference is the *expression*.  The skull,
the hair, the ears and the bust are the character, so they are built from ``st``.
The face -- nose, mouth, blush, eyes, brows -- is built from ``fst``, the same
style with its layout warped by the mood (:mod:`gen.portraits.expression`).  For a
neutral expression the two are the same object and nothing has moved at all.
"""
from __future__ import annotations

import numpy as np

from core.canvas import Canvas
from core import palette

from . import (accessories, brush, cel, demography, detail, dna, expression, eyes,
              face, garments, hair, head, hues, humanize, regalia)
from . import style as style_mod
from .shape import union_all

__all__ = [
    "FRAME", "PHOTO_TURN", "PHOTO_EXPRESSION", "canvas",
    "compose", "compose_posed", "render", "render_posed", "render_seed",
    "describe", "colors",
]

#: Half-extents of the world box a portrait is composed in.
FRAME = head.FRAME

#: The passport-photo pose.  A character portrait is a straight-on photograph:
#: the sitter looks into the lens and the face is at rest.  Pose and mood remain
#: available through :func:`compose_posed`, never through the default path.
PHOTO_TURN = 0.0
PHOTO_EXPRESSION = "neutral"


def canvas(size, background=None) -> Canvas:
    """A portrait canvas: the frame from :mod:`gen.portraits.head`, head centred."""
    if isinstance(size, (tuple, list)):
        w, h = int(size[0]), int(size[1])
    else:
        w = h = int(size)
    hw, hh = head.FRAME
    scale = min(w / (2.0 * hw), h / (2.0 * hh))
    return Canvas(w, h, scale=scale, center=(0.0, -head.HEAD_Y), background=background)




def _contour_taper(y_min: float, heavy: float = 1.0, light: float = 0.3):
    """Line weight as a function of height: heavy at the jaw, light at the crown."""
    def taper(x, y):
        t = np.clip((y - y_min) / 1.0, 0.0, 1.0)
        return 1.0 + heavy * (1.0 - t) - light * t

    return taper


def compose(genes: dict, size=384, background: bool = True) -> Canvas:
    """Render a passport portrait.  Pure: same genes, same bytes.

    The pose is fixed: looking straight into the camera with a neutral face.  The
    ``turn`` and ``expression`` genes are ignored here on purpose, so an old
    genome with a three-quarter pose still produces the same straight portrait as
    a new one.  Use :func:`compose_posed` for the opt-in expression/pose study.
    """
    return compose_posed(genes, size=size, background=background,
                         expression_name=PHOTO_EXPRESSION, turn=PHOTO_TURN)


def compose_posed(genes: dict, size=384, background: bool = True,
                  expression_name: str | None = None,
                  turn: float | None = None) -> Canvas:
    """Render the opt-in pose/expression study for a genome.

    This is the separate API the passport portrait deliberately does not use.
    ``expression_name=None`` reads the genome's expression gene, and ``turn=None``
    reads its turn gene; both can be overridden explicitly.  Nothing in the saved
    DNA changes, so the mood is a viewing option rather than a different person.

    An expression warps only the face.  The *head* -- its skull, its hair, its
    ears and its bust -- is the character, so those passes keep ``st``; the face's
    landmarks move together, so the nose, mouth, blush, eyes and brows are all
    drawn from the warped copy.  For a neutral expression ``fst is st`` and
    nothing moves at all.
    """
    st = dna.apply_to(style_mod.get(genes["style"]), genes)
    h = hues.from_genes(genes)
    kit = regalia.build(genes, h)
    dh = regalia.dress(kit, h)
    expr_name = genes["expression"] if expression_name is None else expression_name
    expr = expression.get(expr_name)
    fst = expression.warp(st, expr)
    raw_turn = genes["turn"] if turn is None else turn
    turn = float(np.clip(raw_turn, -head.TURN_MAX, head.TURN_MAX))
    rng = np.random.default_rng(dna.seed_of(genes))
    # Small, stable, outfit-independent bilateral deviations.  They are built
    # from a separate face seed, never from the hair or the coat, so a person
    # does not become a different person when the sitter changes clothes or mood.
    asym = humanize.asymmetry(genes)

    c = canvas(size)
    if background:
        detail.background(c, h, genes, rng)

    skull = head.silhouette(st, turn)
    skull_surf = brush.Surface(c, skull)
    skull_cov = skull_surf.full_coverage(c)
    hs = hair.build(st, genes, rng, turn, skull=skull, eye_st=fst, eye_expr=expr,
                    asym=asym)

    # --- 2. hair behind everything ---------------------------------------  (behind the body)
    hair.paint_back(c, st, h, hs, turn)

    # --- 3. body ----------------------------------------------------------
    # The neck goes down first and the bust over it, so the collar cuts the neck
    # exactly where the trapezius starts.
    neck = brush.Surface(c, head.neck(st))
    neck.paint(c, h.skin)
    neck.bands(c, [h.skin_shadow], (0.34,),
               volume=cel.dome(st.neck_w * 1.7, 0.36, 0.30, cy=st.chin_y - 0.22), bevel=0.03,
               soft=1.2)
    neck.stroke(c, st.ink_w * st.line * 0.55, h.line_soft, inset=st.ink_w * st.line * 0.34,
                alpha=0.55, soft=0.5, taper=_contour_taper(st.chin_y))
    neck_cov = neck.full_coverage(c)
    detail.paint_neck(c, st, h, genes, turn, neck_cov)
    cast = np.minimum(neck_cov * (1.0 - skull_cov),
                      brush.shift_cov(skull_cov, -st.chin_shadow, c.scale))
    brush.paint_cov(c, cast, h.skin_deep, soft=1.2, alpha=0.80)

    bust = brush.Surface(c, head.shoulders(st))
    bust.paint(c, dh.cloth)
    bust.bands(c, [dh.cloth_shadow], (0.40,), volume=cel.dome(0.90, 0.62, 0.50, cy=-0.98),
               bevel=0.05, soft=1.8)
    bust.stroke(c, st.ink_w * st.line * 0.85, dh.cloth_line, inset=st.ink_w * st.line * 0.42)
    neck_kind = demography.collar(genes)
    bust_cov = bust.full_coverage(c)
    detail.paint_bust(c, st, dh, genes, neck_kind, bust_cov)
    regalia.paint_bust(c, st, dh, genes, kit, neck_kind, bust_cov)
    # Hair in front of the body casts onto the neck and the collar.  This goes
    # before the near locks so their own fill owns the edge; the shadow is
    # clipped to the neck and the bust separately, because they are different
    # materials.
    humanize.paint_hair_cast(c, st, h, hs, neck_cov, bust_cov,
                              cloth_shadow=dh.cloth_shadow)

    # The locks rooted at the temples hang in front of the shoulder and behind
    # the jaw, so they belong between the body and the head.
    hair.paint_near(c, st, h, hs, turn)

    # --- 4. head ----------------------------------------------------------
    skull_surf.paint(c, h.skin)
    skull_surf.bands(c, [h.skin_shadow, h.skin_deep], (0.50, 0.21), volume=head.volume(st, turn),
                     bevel=0.055, soft=1.6)
    # No ear repaint.  The ear's exposed half sits exactly at the face's edge,
    # where the skin patch and the helix arc read as a pair of tan bumps or wire
    # loops beside the eye.  A turned head still shows the loop rather than an
    # ear, so the pass stays off there as well; the cap and the hairline frame the
    # temple.  ``face.paint_ears`` remains available to a future ear that is drawn
    # as a volume instead of a line.
    detail.paint_skin(c, st, h, genes, turn, skull_cov, rng)
    humanize.paint_face_life(c, st, h, genes, turn, asym, skull_cov)
    lw = st.ink_w * st.line
    # The outer contour is the one line the eye traces first, so a hard, opaque,
    # full-weight stroke turns the whole head into a cut-out sticker.  It is drawn
    # a little lighter than the features, in a softened ink, and feathered, so the
    # silhouette reads as a surface meeting the light rather than as a border.
    skull_surf.stroke(c, lw * 0.86,
                      palette.mix(palette.mix(h.line, h.line_soft, 0.38),
                                  h.skin_shadow, 0.18),
                      inset=lw * 0.45,
                      taper=_contour_taper(st.chin_y, heavy=0.55, light=0.34),
                      mask=humanize.contour_mask(skull_surf, st, asym),
                      alpha=0.66, soft=0.85)
    detail.paint_facial_hair(c, st, h, genes, turn, skull_cov, rng)
    face.paint_blush(c, fst, h, turn, expr, clip=skull, marks=(genes["blush"] > 0.25))

    # --- 5. features ------------------------------------------------------
    front_cov = brush.mask(c, hs.front_mass)
    Xg, Yg = c.grid()
    # The cast shade is the fringe's, not the whole front hair's.  Including the
    # sidelocks in this mask threw a second brown curtain down the cheeks and
    # read as a bowl cut; including the cap did the same at the temples.  Cast
    # only from the fringe, only onto the forehead, and only just under the hem.
    fringe_shapes = list(hs.bangs)
    if hs.fringe_base is not None:
        fringe_shapes.append(hs.fringe_base)
    if fringe_shapes:
        fringe_cov = brush.mask(c, union_all(fringe_shapes))
        above_brow = np.clip((Yg - (fst.brow_y + 0.005)) / 0.030, 0.0, 1.0)
        below_hairline = np.clip((head.hairline_y(fst) - Yg) / 0.030, 0.0, 1.0)
        inside_temple = np.clip(
            (st.head_w * 0.94 - np.abs(Xg)) / (st.head_w * 0.22), 0.0, 1.0)
        fringe = np.minimum(
            brush.shift_cov(fringe_cov, -0.025, c.scale) * (1.0 - fringe_cov),
            skull_cov) * above_brow * below_hairline * inside_temple
        brush.paint_cov(c, fringe, h.skin_shadow, soft=2.0, alpha=0.070)
    face.paint_nose(c, fst, h, turn, clip=skull, asym=asym)
    face.paint_mouth(c, fst, h, turn, expr, clip=skull, asym=asym)
    for side in (+1, -1):
        eyes.paint(c, fst, h, side, turn, expr, clip=skull, asym=asym)
    face.paint_brows(c, fst, h, turn, expr, clip=skull, asym=asym)
    accessories.paint_scars(c, fst, h, genes, turn, clip=skull, asym=asym)

    # --- 6. hair in front -------------------------------------------------
    hair.paint_front(c, st, h, hs, turn, clip=front_cov)
    detail.paint_hair_detail(c, st, h, hs, turn, rng, skull_cov, asym=asym)
    # The exposed-ear repaint is deliberately gone.  A front-view ear at this
    # scale is a skin patch on the silhouette, and the old disc read as a pair of
    # tan bumps beside the eyes.  In a turned study the same pass reduces to a
    # wire helix, so it is not restored there either.  The cap and the hairline
    # frame the temple.
    garments.paint_veil(c, st, h, kit, turn)
    # Glasses sit on the face and their temples lie over the hair.  Painting them
    # last is what keeps a strand from growing through a lens.
    accessories.paint_eyewear(c, fst, h, genes, turn, asym=asym)
    return c


def render(genes: dict, size=384, background: bool = True) -> Canvas:
    """The passport portrait: straight to camera, neutral expression."""
    return compose(genes, size=size, background=background)


def render_posed(genes: dict, size=384, background: bool = True,
                 expression_name: str | None = None,
                 turn: float | None = None) -> Canvas:
    """The opt-in pose/expression study; see :func:`compose_posed`."""
    return compose_posed(genes, size=size, background=background,
                         expression_name=expression_name, turn=turn)


def render_seed(seed=0, size=384, background: bool = True, genes: dict | None = None):
    """Render a genome drawn from a seed (or pass one in)."""
    g = genes if genes is not None else dna.random_genes(seed)
    return compose(g, size=size, background=background)


def describe(genes: dict) -> str:
    """A one-line summary for contact-sheet captions and asset registries."""
    return dna.describe(genes)


def colors(genes: dict) -> dict:
    """The portrait's colour script as hex -- the part an asset registry wants."""
    return hues.from_genes(genes).as_dict()
