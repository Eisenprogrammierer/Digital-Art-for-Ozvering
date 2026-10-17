"""Composition: the layer order that turns parts into a face.

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
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.canvas import Canvas

from . import brush, cel, dna, expression, eyes, face, hair, head, hues
from . import style as style_mod

__all__ = ["FRAME", "canvas", "compose", "render", "render_seed", "describe", "colors"]

#: Half-extents of the world box a portrait is composed in.
FRAME = head.FRAME


def canvas(size, background=None) -> Canvas:
    """A portrait canvas: the frame from :mod:`gen.portraits.head`, head centred."""
    if isinstance(size, (tuple, list)):
        w, h = int(size[0]), int(size[1])
    else:
        w = h = int(size)
    hw, hh = head.FRAME
    scale = min(w / (2.0 * hw), h / (2.0 * hh))
    return Canvas(w, h, scale=scale, center=(0.0, -head.HEAD_Y), background=background)


def _gradient(canvas: Canvas, top, bottom):
    """A two-stop background in OKLab: flat enough to read as a backdrop."""
    X, Y = canvas.grid()
    y0 = Y.min()
    span = max(float(Y.max() - y0), 1e-6)
    t = np.clip((Y - y0) / span, 0.0, 1.0)
    canvas.rgb[:] = palette.srgb_to_linear(palette.ramp_map([bottom, top], t))
    canvas.a[:] = 1.0


def _contour_taper(y_min: float, heavy: float = 1.0, light: float = 0.3):
    """Line weight as a function of height: heavy at the jaw, light at the crown."""
    def taper(x, y):
        t = np.clip((y - y_min) / 1.0, 0.0, 1.0)
        return 1.0 + heavy * (1.0 - t) - light * t

    return taper


def compose(genes: dict, size=384, background: bool = True) -> Canvas:
    """Render a genome.  Pure: same genes, same bytes."""
    st = dna.apply_to(style_mod.get(genes["style"]), genes)
    h = hues.from_genes(genes)
    expr = expression.get(genes["expression"])
    turn = float(np.clip(genes["turn"], -head.TURN_MAX, head.TURN_MAX))
    rng = np.random.default_rng(dna.seed_of(genes))

    c = canvas(size)
    if background:
        _gradient(c, h.bg, palette.mix(h.bg, (0.60, 0.62, 0.70), 0.26))

    skull = head.silhouette(st, turn)
    skull_surf = brush.Surface(c, skull)
    skull_cov = skull_surf.full_coverage(c)
    hs = hair.build(st, genes, rng, turn, skull=skull)

    # --- 2. hair behind everything ---------------------------------------  (behind the body)
    hair.paint_back(c, st, h, hs, turn)

    # --- 3. body ----------------------------------------------------------
    # The neck goes down first and the bust over it, so the collar cuts the neck
    # exactly where the trapezius starts.
    neck = brush.Surface(c, head.neck(st))
    neck.paint(c, h.skin)
    neck.bands(c, [h.skin_shadow], (0.34,),
               volume=cel.dome(st.neck_w * 1.7, 0.36, 0.30, cy=st.chin_y - 0.22), bevel=0.03)
    neck.stroke(c, st.ink_w * st.line * 0.55, h.line_soft, inset=st.ink_w * st.line * 0.34,
                alpha=0.55, taper=_contour_taper(st.chin_y))
    neck_cov = neck.full_coverage(c)
    cast = np.minimum(neck_cov * (1.0 - skull_cov),
                      brush.shift_cov(skull_cov, -st.chin_shadow, c.scale))
    brush.paint_cov(c, cast, h.skin_deep, soft=1.2, alpha=0.80)

    bust = brush.Surface(c, head.shoulders(st))
    bust.paint(c, h.cloth)
    bust.bands(c, [h.cloth_shadow], (0.40,), volume=cel.dome(0.90, 0.62, 0.50, cy=-0.98),
               bevel=0.05)
    bust.stroke(c, st.ink_w * st.line * 0.85, h.cloth_line, inset=st.ink_w * st.line * 0.42)

    # --- 4. head ----------------------------------------------------------
    skull_surf.paint(c, h.skin)
    skull_surf.bands(c, [h.skin_shadow, h.skin_deep], (0.50, 0.21), volume=head.volume(st, turn),
                     bevel=0.055)
    face.paint_ears(c, st, h, turn, clip=skull)
    lw = st.ink_w * st.line
    skull_surf.stroke(c, lw, h.line, inset=lw * 0.42,
                      taper=_contour_taper(st.chin_y, heavy=0.55, light=0.34))
    face.paint_blush(c, st, h, turn, expr, clip=skull, marks=(genes["blush"] > 0.25))

    # --- 5. features ------------------------------------------------------
    front_cov = brush.mask(c, hs.front_mass)
    fringe = np.minimum(brush.shift_cov(front_cov, -0.130, c.scale) * (1.0 - front_cov),
                        skull_cov)
    brush.paint_cov(c, fringe, h.skin_shadow, soft=1.0, alpha=0.58)
    face.paint_nose(c, st, h, turn, clip=skull)
    face.paint_mouth(c, st, h, turn, expr, clip=skull)
    for side in (+1, -1):
        eyes.paint(c, st, h, side, turn, expr, clip=skull)
    face.paint_brows(c, st, h, turn, expr, clip=skull)

    # --- 6. hair in front -------------------------------------------------
    hair.paint_front(c, st, h, hs, turn, clip=front_cov)
    return c


def render(genes: dict, size=384, background: bool = True) -> Canvas:
    return compose(genes, size=size, background=background)


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
