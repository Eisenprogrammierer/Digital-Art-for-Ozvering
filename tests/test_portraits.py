"""Regression tests for the procedural portraits.

The interesting failures a portrait generator has are not crashes -- they are
faces that are still a face-shaped pile of pixels.  So the tests here come in
three kinds:

* the genome must survive a round trip through its own string form, because the
  string *is* the save file;
* the derived anatomy must stay plausible for every genome the generator can
  produce, which is what keeps independent sliders from making a monster;
* a handful of composition invariants that have broken in review -- a fringe over
  an eye, a gap between the skull cap and the fringe, a head that has drifted out
  of frame.

Run either way::

    python tests/test_portraits.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import palette  # noqa: E402
from gen.portraits import cel, demography, dna, expression, face, hair, head, hues, eyes  # noqa: E402
from gen.portraits import phenotype  # noqa: E402
from gen.portraits import portrait as portraitlib  # noqa: E402
from gen.portraits import style as style_mod  # noqa: E402
from gen.portraits.shape import Shape  # noqa: E402


# ------------------------------------------------------------------ helpers
def _genes(seed: int, **over) -> dict:
    g = dna.random_genes(seed)
    g.update(over)
    return g


def _style(genes: dict):
    return dna.apply_to(style_mod.get(genes["style"]), genes)


def _hair(genes: dict):
    st = _style(genes)
    expr = expression.get(genes["expression"])
    fst = expression.warp(st, expr)
    rng = np.random.default_rng(dna.seed_of(genes))
    return st, hair.build(st, genes, rng, float(genes["turn"]), eye_st=fst,
                          eye_expr=expr)


def _hash(canvas) -> str:
    return hashlib.sha256(canvas.to_uint8().tobytes()).hexdigest()


# ------------------------------------------------------------------ genome
def test_genome_round_trips_through_its_string():
    for seed in range(200):
        g = dna.random_genes(seed)
        assert dna.decode(dna.encode(g)) == g, f"seed {seed} did not survive encode/decode"


def test_encode_is_complete_and_in_gene_order():
    text = dna.encode(dna.random_genes(11))
    keys = [item.partition("=")[0] for item in text.split(";")]
    assert keys == list(dna.GENES), "the wire format must be stable, so the key order is fixed"
    assert "\n" not in text


def test_decode_refuses_a_genome_it_does_not_understand():
    g = dna.random_genes(3)
    good = dna.encode(g)
    items = good.split(";")
    variants = [
        dna.encode(dict(g, style="nonesuch")),
        dna.encode(dict(g, hair_style="mohawk")),
        dna.encode(dict(g, expression="beaming")),
        good + ";hat=none",
        ";".join(i for i in items if not i.startswith("eye_w=")),   # a missing gene
        ";".join(("eye_w=oops" if i.startswith("eye_w=") else i) for i in items),
        dna.encode(dict(g, skin="#nothex")),
        ";".join(i for i in items if not i.startswith("bang_n=")),  # a missing legacy gene
    ]
    for broken in variants:
        try:
            dna.decode(broken)
        except ValueError:
            continue
        raise AssertionError(f"decode accepted {broken!r}")


def test_breeding_produces_a_valid_genome():
    for seed in range(60):
        a, b = dna.random_genes(seed), dna.random_genes(seed + 4000)
        child = dna.breed(a, b, seed=seed)
        assert dna.decode(dna.encode(child)) == child
        assert child["style"] in (a["style"], b["style"])
        assert child["hair_style"] in (a["hair_style"], b["hair_style"])
        assert child["expression"] in (a["expression"], b["expression"])
        assert _style(child) is not None
        # Shape genes are averaged (and clamped), so a child is never wilder than
        # the child's own wire format allows.
        for key in dna.SHAPE_GENES + dna.MORPH + dna.PIGMENT:
            assert -1.0 <= child[key] <= 1.0
        for key in dna.PHENOTYPE_CATEGORIES:
            assert child[key] in (a[key], b[key]), f"{key} was not inherited as a category"


def test_mutation_keeps_the_genome_readable_and_coloured():
    for seed in range(40):
        g = dna.random_genes(seed)
        m = dna.mutate(g, seed=seed)
        assert dna.decode(dna.encode(m)) == m
        for key in dna.COLOUR_GENES:
            palette.hex_to_srgb(m[key])  # must still parse


def test_seed_of_is_a_function_of_the_genome():
    g = dna.random_genes(8)
    assert dna.seed_of(g) == dna.seed_of(dict(g))
    assert dna.seed_of(g) != dna.seed_of(dna.random_genes(9))
    assert 0 <= dna.seed_of(g) < 2**48


def test_quantise_is_the_wire_format_precision():
    for value in (0.0, 1.23456789, -0.00004, 3.5):
        q = dna.quantise(value)
        assert float(f"{q:.4f}") == q, "a quantised gene must print back exactly"


# ----------------------------------------------------------------- anatomy
def test_derived_anatomy_stays_plausible():
    """The invariants that keep independent genes from producing a monster.

    These are the relationships :func:`dna._clamp_params` exists to enforce: the
    eyes fit in the skull, the jaw tapers to the chin, and the features are in
    order down the face with the hairline above the brows.
    """
    for seed in range(160):
        a, b = dna.random_genes(seed), dna.random_genes(seed + 7000)
        for g in (a, dna.breed(a, b, seed=seed), dna.mutate(a, seed=seed)):
            st = _style(g)
            where = f"seed {seed} {g['style']}"
            assert st.brow_y > st.eye_y, f"{where}: brow below the eye"
            assert st.hairline_y > st.brow_y, f"{where}: hairline below the brow"
            assert st.nose_y < st.eye_y, f"{where}: nose above the eye"
            assert st.mouth_y < st.nose_y, f"{where}: mouth above the nose"
            assert st.chin_y < st.mouth_y, f"{where}: chin above the mouth"
            assert st.blush_y < st.eye_y, f"{where}: blush on the eye"
            assert st.chin_w < st.jaw_w < st.cheek_w, f"{where}: jaw does not taper"
            assert st.eye_dx + st.eye_w * 0.5 + 0.045 <= st.head_w + 1e-9, \
                f"{where}: an eye hangs off the skull"
            aspect = (st.eye_up + st.eye_down) / st.eye_w
            assert 0.58 - 1e-9 <= aspect <= 0.96 + 1e-9, \
                f"{where}: eye opening aspect {aspect:.2f} is a slit or a dinner plate"
            # Eyes are a relationship: an opening cannot swallow the inter-eye
            # gap or the temple even when both genes are at their extremes.
            assert st.eye_w <= st.head_w * 0.50 + 1e-9, \
                f"{where}: eye is too wide for the skull"
            assert st.eye_w <= st.cheek_w * 0.50 + 1e-9, \
                f"{where}: eye is too wide for the cheek"
            assert st.eye_w <= st.eye_dx * 1.28 + 1e-9, \
                f"{where}: eyes are almost touching"
            assert 2.0 * st.eye_dx - st.eye_w >= st.eye_w * 0.45 - 1e-9, \
                f"{where}: inter-eye gap collapsed"
            assert st.neck_w < st.jaw_w, f"{where}: neck thicker than the jaw"
            assert st.shoulder_w > st.neck_w + 0.24 - 1e-9, f"{where}: no shoulders"
            assert st.shoulder_y < st.neck_y < st.chin_y, f"{where}: bust above the chin"


def test_a_head_is_a_head_for_every_style():
    for name in style_mod.names():
        st = style_mod.get(name)
        shape = head.silhouette(st, 0.0)
        x0, y0, x1, y1 = shape.bbox
        assert -0.62 < x0 < x1 < 0.62, f"{name}: skull wider than the frame"
        assert -0.55 < y0 and y1 < 0.56, f"{name}: skull taller than the frame"
        assert shape.f(0.0, 0.0) < 0.0, f"{name}: the skull has a hole in it"
        assert shape.f(0.0, 0.60) > 0.0, f"{name}: the skull reaches past the crown"


# ------------------------------------------------------------------- the turn
def test_a_turn_of_zero_is_the_identity():
    """A front view is not an approximation of the solid: it *is* the solid."""
    for name in style_mod.names():
        st = style_mod.get(name)
        assert np.array_equal(head.control_points(st, 0.0), head.control_points(st)), name
        sk = head.skull(st)
        x = np.array([-0.30, 0.0, 0.205, 0.44])
        y = np.array([-0.465, -0.065, 0.10, 0.30])
        px, py = sk.project(x, y, 0.0)
        assert np.array_equal(px, x), f"{name}: a zero turn moved x"
        assert np.array_equal(py, y), f"{name}: a zero turn moved y"
        assert sk.crest(0.02, 0.0) == 0.0, f"{name}: the crest left the midline"
        assert abs(float(sk.support(-0.065, 0.0)) - float(sk.width(-0.065))) < 1e-12, name


def test_the_outline_of_a_turn_is_the_support_of_its_cross_sections():
    """The silhouette is the solid's, not a template: it ends where the body does.

    Sampled as densely as a render would and checked against the support the model
    claims, so a silhouette that has drifted away from the solid fails here instead
    of looking subtly wrong in a picture.  Both faces of a section are sampled: the
    near edge of a turned outline is the *face* of the head and the far edge is the
    back of the skull.
    """
    for name in style_mod.names():
        st = style_mod.get(name)
        sk = head.skull(st)
        for turn in (0.22, -0.40, head.TURN_MAX):
            for y in (-0.30, -0.065, 0.10):
                rx = float(sk.width(y))
                if rx <= 1e-6:
                    continue
                u = np.linspace(-1.0, 1.0, 1201)
                z = float(sk.semi_depth(y)) * np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
                x = u * rx
                front = x * np.cos(turn) + (float(sk.centre(y)) + z) * np.sin(turn)
                back = x * np.cos(turn) + (float(sk.centre(y)) - z) * np.sin(turn)
                shift = float(sk.centre(y)) * np.sin(turn)
                reach = float(sk.support(y, turn))
                where = f"{name} turn {turn:.2f} y {y}"
                edge = np.concatenate([front, back])
                assert abs(float(edge.max()) - (shift + reach)) < 2e-3, f"{where}: near outline"
                assert abs(float(edge.min()) - (shift - reach)) < 2e-3, f"{where}: far outline"


def test_the_two_ends_of_a_face_turn_less_than_the_middle_of_it():
    """The depth profile, not the width profile, is what a turn reads.

    A jaw that swung as far as a nose is a mask: the chin and the crown are the
    shallow ends of the skull, and the eye line is the deep middle.
    """
    for name in style_mod.names():
        st = style_mod.get(name)
        sk = head.skull(st)
        turn = 0.5
        slide = lambda y: float(sk.project(0.0, y, turn)[0])  # noqa: E731
        mid = slide(0.02)
        assert mid > 0.10, f"{name}: the middle of the face barely moved"
        assert slide(-0.465) < mid, f"{name}: the chin swung as far as the eye line"
        assert slide(0.42) < mid, f"{name}: the crown swung as far as the eye line"
        assert slide(-0.20) > slide(-0.465), f"{name}: the jaw turns no more than the chin"


def test_the_receding_side_is_the_one_that_goes_back():
    """``receding`` is derived rather than declared, so it must agree with depth.

    Getting this backwards is invisible at a small angle and plain at a large one --
    the eyes, the ears and the side locks all ask.
    """
    for name in style_mod.names():
        st = style_mod.get(name)
        sk = head.skull(st)
        for turn in (0.30, -0.30, head.TURN_MAX):
            for side in (-1.0, 1.0):
                x = side * 0.8 * float(sk.width(0.02))
                here = float(sk.depth_at(x, 0.02, turn))
                there = float(sk.depth_at(-x, 0.02, turn))
                where = f"{name} turn {turn:.2f} x {side:+.0f}"
                assert (here < there) == bool(sk.receding(x, turn)), \
                    f"{where}: receding disagrees with the depth"


def test_a_turned_eye_keeps_its_plane_proportions():
    """Both eyes foreshorten by almost the same amount, not one huge and one slit.

    Projecting each corner onto the skull's curvature used to stretch the near eye
    and collapse the far one; the measured ratio could pass 2:1 on a seeded genome.
    An eye is a flat opening on the face plane, so the scale is ``cos(turn)`` plus
    a few percent of perspective on the near side.
    """
    for name in style_mod.names():
        st = style_mod.get(name)
        straight = eyes.placement(st, +1, 0.0)
        assert abs(straight[0] - st.eye_dx) < 1e-12, f"{name}: the eye left its station"
        assert abs(straight[1] - 1.0) < 1e-12, f"{name}: a zero turn changed the scale"
        for turn in (0.10, 0.30, 0.45, head.TURN_MAX):
            near = eyes.placement(st, -1, turn)
            far = eyes.placement(st, +1, turn)
            where = f"{name} turn {turn:.2f}"
            assert 0.80 <= near[1] <= 1.0 and 0.80 <= far[1] <= 1.0, \
                f"{where}: eye scale {near[1]:.3f}/{far[1]:.3f} out of range"
            assert near[1] >= far[1], f"{where}: the far eye is wider than the near one"
            assert near[1] / far[1] < 1.08, \
                f"{where}: eye width ratio {near[1] / far[1]:.3f} is caricature"


def test_a_turned_head_neither_collapses_nor_explodes():
    """The outline changes width by the support of the sections and no more."""
    for name in style_mod.names():
        st = style_mod.get(name)
        straight = head.silhouette(st, 0.0).bbox
        w0 = straight[2] - straight[0]
        for turn in (-head.TURN_MAX, -0.3, 0.3, head.TURN_MAX):
            x0, y0, x1, y1 = head.silhouette(st, turn).bbox
            where = f"{name} at turn {turn:.2f}"
            assert np.isfinite([x0, y0, x1, y1]).all(), f"{where}: unbounded skull"
            assert -0.62 < x0 < x1 < 0.62 and -0.55 < y0 and y1 < 0.56, \
                f"{where}: the skull has left the frame"
            width = x1 - x0
            assert 0.90 * w0 <= width <= 1.15 * w0, f"{where}: width {width:.3f} vs {w0:.3f}"


def test_the_skull_is_finite_everywhere_it_is_asked():
    """Off the profile, past the crown, into the neck: no invented divide by zero."""
    px = np.array([-1.2, -0.6, -0.31, 0.0, 0.31, 0.6, 1.2])
    py = np.array([-0.90, -0.60, -0.465, -0.065, 0.30, 0.505, 0.60])
    X, Y = np.meshgrid(px, py)
    for name in style_mod.names():
        st = style_mod.get(name)
        sk = head.skull(st)
        assert st.head_d > 0.5, f"{name}: a head with no depth"
        for turn in (0.0, 0.30, -head.TURN_MAX):
            x, y = sk.project(X, Y, turn)
            depth, reach = sk.depth_at(X, Y, turn), sk.support(Y, turn)
            for values, what in ((x, "x"), (y, "y"), (depth, "depth"), (reach, "support")):
                assert np.isfinite(values).all(), f"{name}: a non-finite {what}"
            assert (reach >= -1e-12).all(), f"{name}: a negative half-width"
        assert np.isfinite(sk.crest(0.02, 0.30)), f"{name}: crest"
        assert np.isfinite(sk.width(0.9)) and np.isfinite(sk.semi_depth(-2.0)), name


def test_the_framing_is_a_headshot():
    """The crown near the top edge, the chin about two thirds down, bust below."""
    _, hh = head.FRAME
    top, bottom = -head.HEAD_Y + hh, -head.HEAD_Y - hh
    height = top - bottom

    def down(y: float) -> float:
        return (top - y) / height

    for seed in range(40):
        g = _genes(seed)
        st, h = _hair(g)
        crown = h.mass.bbox[3]
        # The skull itself is always in frame; the hair may run off the top, which
        # is what a headshot does, but a style that keeps it down the middle of the
        # frame has drifted out of the portrait.
        assert 0.02 <= down(st.top_y) <= 0.26, f"seed {seed}: the skull crown is cropped"
        assert -0.20 <= down(crown) <= 0.30, f"seed {seed}: crown at {down(crown):.2f}"
        assert crown > st.top_y, f"seed {seed}: the hair sits below the skull"
        assert 0.62 <= down(st.chin_y) <= 0.80, \
            f"seed {seed}: chin at {down(st.chin_y):.2f}, not two thirds down"
        assert h.mass.bbox[1] < st.chin_y, f"seed {seed}: no bust below the chin"


# -------------------------------------------------------------------- hair
def test_no_front_hair_enters_an_eye_opening():
    """A lock over an eye is the difference between a face and a curtain.

    Sampling the centre of each eye was not enough: at a three-quarter turn the
    cap and a sidelock both grazed the *outer corner* of the far opening while
    the centre stayed clear.  This samples the actual placed opening -- after the
    expression has moved it -- against both the cap and the front mass.
    """
    for seed in range(120):
        g = _genes(seed)
        st, h = _hair(g)
        turn = float(g["turn"])
        expr = expression.get(g["expression"])
        fst = expression.warp(st, expr)
        geometry = eyes.shapes(fst, expr)
        for side in (+1, -1):
            cx, scale = eyes.placement(fst, side, turn)
            opening = geometry["opening"].place(
                cx=cx, cy=fst.eye_y, sx=side * scale, sy=1.0, rot=side * fst.eye_tilt
            )
            x0, y0, x1, y1 = opening.bbox
            X, Y = np.meshgrid(np.linspace(x0, x1, 31), np.linspace(y0, y1, 31))
            inside = opening.f(X, Y) < 0.0
            where = f"seed {seed} ({g['hair_style']}) side {side:+d}"
            assert not np.any(inside & (h.cap.f(X, Y) < -0.002)), f"{where}: cap over an eye"
            assert not np.any(inside & (h.front_mass.f(X, Y) < -0.002)), \
                f"{where}: a lock over an eye"


def test_the_hairline_has_no_gap():
    """Along the hairline the cap and the fringe must overlap.

    They are two passes with two sets of roots; where they fail to meet, bare
    skull shows through as a line of teeth -- which is exactly the artefact that
    the ``fade`` on the fringe ink exists to hide.
    """
    for name in dna.HAIR_STYLES:
        for seed in range(8):
            g = _genes(seed * 13 + 1, hair_style=name)
            st, h = _hair(g)
            hairline = head.hairline_y(st) - 0.02
            xs = np.linspace(-0.34, 0.34, 25)
            above = h.cap.f(xs, np.full_like(xs, hairline + 0.035))
            below = h.front_mass.f(xs, np.full_like(xs, hairline - 0.030))
            gap = float(np.minimum(above, below).max())
            assert gap < 0.0, f"{name} seed {seed}: skull shows at the hairline by {gap:.3f}"


def test_every_hairstyle_renders_and_covers_the_crown():
    for name in dna.HAIR_STYLES:
        g = _genes(21, hair_style=name)
        st, h = _hair(g)
        assert h.cap.f(0.0, head.hairline_y(st) + 0.04) < 0.0, f"{name}: crown is bare"
        assert h.cap.f(0.0, head.hairline_y(st) - 0.12) > 0.0, f"{name}: hair over the face"
        img = portraitlib.render(g, size=64)
        assert img.to_uint8().shape[0] == 64


def test_the_crown_is_painted_hair_and_not_skin():
    """The cap covers the skull -- and it has to be *painted* there.

    Geometry was never the problem: the cap has always covered the crown above
    the hairline and the test above says so.  The cap used to be drawn with the
    hair behind the head, though, and the skin goes on afterwards, so the crown
    came out as a bare band of forehead with the fringe hanging over it and the
    partings showing scalp.  Only pixels catch an ordering mistake, so this test
    samples them.
    """
    for name in dna.HAIR_STYLES:
        g = _genes(3, hair_style=name)
        st, _ = _hair(g)
        canvas = portraitlib.render(g, size=96)
        X, Y = canvas.grid()
        img = canvas.to_uint8().reshape(-1, 4)[:, :3].astype(float)
        flat = np.stack([X.ravel(), Y.ravel()], axis=1)
        h = hues.from_genes(g)

        def pixel(x, y):
            return img[int(np.argmin(((flat - (x, y)) ** 2).sum(axis=1)))]

        def nearest(px, colours):
            # The colour script stores sRGB, not linear light; compare in the
            # space the pixels are exported in.
            ref = np.array(colours) * 255.0
            return float(np.min(np.linalg.norm(ref - px, axis=1)))

        hair_hues = [h.hair, h.hair_shadow, h.hair_deep]
        skin_hues = [h.skin, h.skin_shadow, h.skin_deep]
        crown = pixel(0.0, head.hairline_y(st) + 0.05)
        cheek = pixel(0.10, -0.20)
        assert nearest(crown, hair_hues) < nearest(crown, skin_hues) - 24.0, (
            f"{name}: the crown is painted skin, not hair")
        assert nearest(cheek, skin_hues) < nearest(cheek, hair_hues) - 24.0, (
            f"{name}: the cheek is painted hair, not skin")


def test_every_fringe_clump_carries_its_own_root():
    """The ink on a clump fades at *its* root, so the two lists have to stay aligned.

    The root is not decoration: it is the only thing in the frame that knows how
    far up a clump's strand line is allowed to run, and the painter reads it back
    through ``zip``.  Zip truncates silently, so a clump added to one list and not
    the other would not fail -- it would just lose its ink, on one style, at some
    sizes.  Checking the pairing directly is cheaper than noticing the picture.
    """
    for name in dna.HAIR_STYLES:
        g = _genes(11, hair_style=name)
        _, h = _hair(g)
        assert len(h.bang_roots) == len(h.bangs), name
        assert len(h.side_roots) == len(h.sides), name
        paired = list(zip(h.bangs, h.bang_roots)) + list(zip(h.sides, h.side_roots))
        assert paired, name
        for clump, root_y in paired:
            # A root sits above the hem it feeds, and the clump has real length:
            # a root at or below the tip would smother the whole strand line.
            assert clump.bbox[3] >= root_y - 1e-9, name
            assert root_y - clump.bbox[1] > 0.02, name


def _probe(shape, n: int = 41):
    """A grid of points that are *inside* ``shape``, for asking who else claims them."""
    x0, y0, x1, y1 = shape.bbox
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    keep = shape.f(X, Y) <= 0.0
    return X[keep], Y[keep]


def test_a_strand_never_escapes_the_clump_it_is_drawn_on():
    """The strand inside a clump is ink, and ink that leaves the silhouette draws on skin.

    It is *built* to stay inside -- the clump's own spine, less width -- and the
    construction is exact only because the two polygons share their spine points
    and their resampling; a strand resampled differently from its clump can put a
    vertex a hair outside it.  Nothing about that is visible in a diff, so the
    containment is measured instead: points known to be inside the strand are
    asked whether they are inside the clump.  It comes out at machine precision,
    which is the strength of the test -- a strand built from *its own* endpoints
    instead of the clump's escapes by a tenth of a unit and would be caught at once.
    """
    for name in dna.HAIR_STYLES:
        g = _genes(17, hair_style=name)
        _, h = _hair(g)
        assert len(h.front_strands) == len(h.bangs) + len(h.sides), name
        for (strand, top), clump in zip(h.front_strands, h.bangs + h.sides):
            X, Y = _probe(strand)
            escape = float(clump.f(X, Y).max())
            assert escape <= 1e-9, f"{name}: the strand leaves its clump by {escape:.2e}"
            # The fade has to start below the root, or the strand is invisible.
            assert top < clump.bbox[3], name

        # A tail carries a strand and a spike does not, so the crest's two lists
        # differ in length on purpose: the test is containment in the row.
        assert len(h.crest_strands) <= len(h.crest), name
        for strand, _ in h.crest_strands:
            X, Y = _probe(strand)
            escape = min(float(c.f(X, Y).max()) for c in h.crest)
            assert escape <= 1e-9, f"{name}: the strand leaves the crest by {escape:.2e}"


def test_the_spiky_styles_carry_spikes():
    """``spiky`` and ``wild`` are named for their spikes, so the count is not a knob.

    A table edit once left ``spike`` at zero on both and they rendered as the plain
    short cap with a different fringe.  The spikes are the only thing that separates
    them from ``short`` at the silhouette, so their presence is asserted directly.
    """
    for name in ("spiky", "wild"):
        g = _genes(9, hair_style=name)
        _, h = _hair(g)
        assert len(h.crest) >= 5, f"{name} has no spikes"
    _, plain = _hair(_genes(9, hair_style="short"))
    assert len(plain.crest) == 0, "the short cut grew spikes"


def test_the_crown_is_layered_over_the_cap():
    """The crown sheets are paint only: they must lie *within* the cap.

    They are never part of ``mass``, so they cannot be allowed to change the
    silhouette either -- an overhang would draw hair outside the shape whose
    contour is inked.  Every sheet has to both reach the cap and stay inside it.
    """
    for name in dna.HAIR_STYLES:
        for seed in range(4):
            g = _genes(seed * 29 + 2, hair_style=name)
            _, h = _hair(g)
            assert len(h.crown) >= 3, f"{name}: no crown sheets"
            for lock in h.crown:
                on_cap = lock & h.cap
                x0, y0, x1, y1 = on_cap.bbox
                assert x1 > x0 and y1 > y0, f"{name}: a crown sheet misses the cap"
                X, Y = _probe(on_cap)
                escape = float(h.cap.f(X, Y).max())
                assert escape <= 1e-9, (
                    f"{name} seed {seed}: a crown sheet left the cap by {escape:.2e}")


# -------------------------------------------------------------- the picture
def test_render_is_deterministic():
    g = dna.random_genes(5)
    assert _hash(portraitlib.render(g, size=96)) == _hash(portraitlib.render(g, size=96))
    a = portraitlib.render_seed(5, size=96)
    assert _hash(a) == _hash(portraitlib.render(dna.random_genes(5), size=96))


def test_export_is_byte_identical():
    g = dna.random_genes(12)
    with tempfile.TemporaryDirectory() as tmp:
        one, two = Path(tmp) / "a.png", Path(tmp) / "b.png"
        for path in (one, two):
            portraitlib.render(g, size=128).save(path)
        assert one.stat().st_size > 0
        assert one.read_bytes() == two.read_bytes(), "the same genome must give the same file"


def test_every_hairstyle_and_expression_renders():
    """Every hairstyle and every expression, through the real pipeline."""
    for i, (hairstyle, name) in enumerate(zip(dna.HAIR_STYLES, dna.EXPRESSIONS)):
        g = _genes(31 + i, hair_style=hairstyle, expression=name)
        img = portraitlib.render(g, size=64).to_uint8()
        assert img.shape[0] == 64
        assert img[:, :, :3].sum() > 0, f"{hairstyle}/{name} rendered nothing"
    for name in style_mod.names():
        assert portraitlib.render(_genes(97, style=name), size=64).to_uint8().sum() > 0


def test_every_style_and_hairstyle_builds():
    """The full cross product of presets, as geometry.

    Pixels are expensive and other tests spend them.  What a new preset pair
    breaks first is a *shape* -- an empty field, an infinite bound, a box of the
    wrong sign -- and that is cheap to check for every pair there is.
    """
    for name in style_mod.names():
        for hairstyle in dna.HAIR_STYLES:
            for expr_name in ("neutral", "grin"):
                g = _genes(31, style=name, hair_style=hairstyle, expression=expr_name)
                turn = float(np.clip(g["turn"], -head.TURN_MAX, head.TURN_MAX))
                st, h = _hair(g)
                expr = expression.get(expr_name)
                # The face is drawn from the warped style, the head from the style
                # itself -- so both halves of the pipeline are exercised here.
                fst = expression.warp(st, expr)
                shapes = {
                    "skull": head.silhouette(st, turn),
                    "neck": head.neck(st, turn),
                    "shoulders": head.shoulders(st),
                    "hair": h.mass,
                    "fringe": h.front_mass,
                    "cap": h.cap,
                    "brow": face.brows(fst, turn, expr)[0],
                }
                for key, value in eyes.shapes(fst, expr).items():
                    if isinstance(value, Shape):
                        shapes[f"eye_{key}"] = value
                inner, _ = face.mouth_interior(fst, turn, expr)
                if inner is not None:
                    shapes["mouth"] = inner
                probes = np.array([[-0.20, 0.0, 0.20, 0.0], [0.20, -0.10, -0.30, 0.05]])
                for key, shape in shapes.items():
                    x0, y0, x1, y1 = shape.bbox
                    where = f"{name}/{hairstyle}/{expr_name}:{key}"
                    assert np.isfinite([x0, y0, x1, y1]).all(), f"{where}: unbounded shape"
                    assert x1 > x0 and y1 > y0, f"{where}: empty bounding box"
                    assert -2.6 < x0 < x1 < 2.6 and -2.6 < y0 < y1 < 1.1, \
                        f"{where}: the shape has left the frame"
                    # A NaN is a divide by zero wearing a shape's clothes and it
                    # renders as a hole.  A union is allowed to report +inf outside
                    # every part's box -- a documented, safe over-estimate -- but
                    # nothing else non-finite is acceptable.
                    values = np.asarray(shape.f(probes[0], probes[1]), dtype=np.float64)
                    assert not np.isnan(values).any(), f"{where}: the field has a NaN"
                    assert (values == -np.inf).sum() == 0, f"{where}: the field has -inf"
    for name in dna.EXPRESSIONS:
        assert set(expression.get(name)) == set(expression.neutral()), \
            f"{name}: an expression key is missing"


def test_the_expression_presets_only_move_the_face():
    neutral, wink = expression.neutral(), expression.get("wink")
    assert neutral["closed"] == ()
    assert wink["closed"] == (1,), "a wink closes exactly one eye"
    for name in dna.EXPRESSIONS:
        row = expression.get(name)
        assert -0.05 < row["brow_dy"] < 0.05
        assert 0.5 <= row["eye_up_s"] <= 1.5
        assert row["mouth_open"] >= 0.0
        assert 0.8 <= row["mouth_w_s"] <= 1.2, f"{name}: the mouth is too wide or too narrow"
        for key, limit in expression.LIMITS.items():
            assert abs(row[key]) <= limit, f"{name}: {key} is past its bound"


# ------------------------------------------------------- the whole face
def test_a_neutral_expression_does_not_warp_the_face():
    """The warp is a strict addition: a neutral row returns the style *itself*.

    Identity rather than equality, because that is the claim -- a portrait that is
    not in a mood is the portrait that existed before there was a warp, and the
    cheapest proof of it is that there is nothing to compare.
    """
    for seed in range(40):
        st = _style(_genes(seed))
        assert expression.warp(st, expression.neutral()) is st


def test_an_expression_moves_the_whole_face():
    """The measurement the warp exists for, and the reason it exists at all.

    Eight per-feature offsets move the brow and the mouth and leave the eyes and
    the nose exactly where they were, which reads as a mask worn over a face.  Every
    landmark moves now, and this is the test that says so: an expression that moves
    only its own feature fails here instead of looking wrong in a picture.
    """
    landmarks = ("brow_y", "eye_y", "nose_y", "mouth_y", "blush_y")
    for name in style_mod.names():
        for seed in range(30):
            st = _style(_genes(seed, style=name))
            for expr_name in dna.EXPRESSIONS:
                if expr_name == "neutral":
                    continue
                w = expression.warp(st, expression.get(expr_name))
                moved = [k for k in landmarks if abs(getattr(w, k) - getattr(st, k)) > 1e-6]
                assert len(moved) == len(landmarks), \
                    f"{name}/{seed}/{expr_name}: only {moved} moved"


def test_a_mood_moves_a_face_and_not_a_head():
    """The skull, the chin, the hairline and the silhouette are the *character*.

    A warp moves the landmarks the expression owns and not one thing the skull owns,
    which is the property that keeps the hair on the head and the head on the
    shoulders while a face changes under both.
    """
    skull_keys = ("chin_y", "chin_w", "head_w", "head_d", "cheek_w", "cheek_y",
                  "jaw_w", "jaw_y", "top_w", "top_y", "wide_y")
    for seed in range(40):
        st = _style(_genes(seed))
        for expr_name in dna.EXPRESSIONS:
            w = expression.warp(st, expression.get(expr_name))
            assert w.hairline_y == st.hairline_y, f"{expr_name}: the hair moved with the mood"
            assert head.hairline_y(w) == head.hairline_y(st), f"{expr_name}: the fringe root moved"
            assert np.array_equal(head.control_points(w, 0.30), head.control_points(st, 0.30)), \
                f"{expr_name}: the silhouette changed with the mood"
            for key in skull_keys:
                assert getattr(w, key) == getattr(st, key), \
                    f"{expr_name}: the skull's {key} changed with the mood"
            sk, sw = head.skull(st), head.skull(w)
            assert np.array_equal(sk.project(np.array([0.1, -0.2]), np.array([0.0, -0.3]), 0.4),
                                  sw.project(np.array([0.1, -0.2]), np.array([0.0, -0.3]), 0.4))


def test_the_warp_keeps_the_face_inside_its_own_head():
    """A face is small and a skull is fixed, so the warp has to stay inside both.

    The brow is the landmark with a ceiling -- the hairline is where the fringe
    roots -- and the mouth is the one with a floor.  Bounded per style and per DNA,
    because the headroom above a brow is a fact about *that* skull: a chibi has less
    forehead than a seinen and its expression is allowed to be smaller, not wrong.
    """
    for name in style_mod.names():
        for seed in range(60):
            st = _style(_genes(seed, style=name))
            for expr_name in dna.EXPRESSIONS:
                expr = expression.get(expr_name)
                w = expression.warp(st, expr)
                brow = w.brow_y + expr["brow_dy"]
                where = f"{name}/{seed}/{expr_name}"
                assert brow + expr["brow_arch"] < head.hairline_y(w), f"{where}: the brow is in the hair"
                # The tightest case measured across four styles and every genome is
                # 0.010 head units above the lid and 0.089 above the chin, so these
                # bounds are claims with room in them rather than the current numbers.
                assert brow + expr["brow_arch"] > w.eye_y + w.eye_up + 0.004, \
                    f"{where}: the brow fell onto the eye"
                assert w.eye_y > w.nose_y > w.mouth_y + expr["mouth_dy"] > w.chin_y + 0.05, \
                    f"{where}: the landmarks are out of order"
                # The eye is the widest feature and the one a widened skull would
                # let leave the silhouette; anatomy is not warped, so it cannot.
                assert w.eye_dx + w.eye_w * 0.5 < w.head_w, f"{where}: the eye left the skull"


def test_an_expression_renders_and_is_not_another_expression():
    """The two ends of the pipeline agree: the warp reaches the pixels.

    Determinism is somebody else's test.  This one is that the *difference* is real
    -- each expression's portrait differs from the neutral one, and from its
    neighbour's -- so a warp that quietly did nothing to the picture would fail.
    """
    hashes = {}
    for name in dna.EXPRESSIONS:
        g = _genes(23, expression=name)
        hashes[name] = _hash(portraitlib.render(g, size=48))
    assert len(set(hashes.values())) == len(hashes), "two expressions drew the same picture"
    assert hashes["neutral"] != hashes["grin"]
    assert hashes["smile"] != hashes["grin"]


# ------------------------------------------------------------------ colour
def test_shadow_is_darker_more_chromatic_and_rolled_towards_violet():
    base = palette.hex_to_srgb("#b98a6a")
    for depth, chroma, hue in ((0.09, 1.18, -0.03), (0.135, 1.30, -0.03), (0.235, 1.12, -0.03)):
        sh = cel.shadow(base, depth=depth, chroma=chroma, hue=hue)
        lb, cb, hb = palette.srgb_to_oklch(base)
        ls, cs, hs = palette.srgb_to_oklch(sh)
        assert ls < lb - 0.5 * depth, "a cel shadow must lose lightness"
        assert cs > cb, "a cel shadow must gain chroma, not go grey"
        rolled = (hb - hs) % 1.0
        assert 0.015 < rolled < 0.045, "the shadow hue must roll down through magenta"


def test_highlight_lifts_lightness():
    base = palette.hex_to_srgb("#8a5a3c")
    hi = cel.highlight(base, lift=0.20, chroma=0.62)
    assert palette.srgb_to_oklch(hi)[0] > palette.srgb_to_oklch(base)[0] + 0.1


def test_the_colour_script_derives_from_the_four_base_colours():
    """The four base colours are the material; pigment genes express them.

    With every pigment gene at neutral the colour script is exactly the four
    genes, which is what makes an old genome render as it always did.  A drawn
    genome then moves skin, hair and iris through its pigment genes -- and every
    derived tone is still a valid colour.
    """
    g = dna.random_genes(4)
    # Temperate is the neutral climate.  A non-temperate lineage deliberately
    # shifts even a pinned base colour, so the exact-material invariant belongs
    # to the neutral environment.
    neutral = dict(g, climate="temperate", **{key: 0.0 for key in dna.PIGMENT})
    h0 = hues.from_genes(neutral)
    assert np.allclose(h0.skin, palette.hex_to_srgb(g["skin"]))
    assert np.allclose(h0.hair, palette.hex_to_srgb(g["hair"]))
    assert np.allclose(h0.iris, palette.hex_to_srgb(g["iris"]))

    h = hues.from_genes(g)
    assert not np.allclose(h.skin, h0.skin) or not np.allclose(h.hair, h0.hair)         or not np.allclose(h.iris, h0.iris), "pigment genes did not reach the script"
    # A derived tone must be a colour, not a channel-out-of-range accident.
    for name in h.__slots__:
        c = getattr(h, name)
        assert c.shape == (3,)
        assert np.all(np.isfinite(c))
        assert np.all(c >= -1e-9) and np.all(c <= 1.0 + 1e-9), f"{name} is outside sRGB"
    assert not np.allclose(h.skin_shadow, h.skin)
    assert not np.allclose(h.hair_shadow, h.hair)


def test_describe_names_the_genome():
    g = dna.random_genes(17)
    text = dna.describe(g)
    assert g["style"] in text and g["hair_style"] in text and g["expression"] in text
    assert g["hair"] in text


# ------------------------------------------------------------- demography
def test_old_genomes_default_the_demographic_and_phenotype_genes():
    """A saved dynasty predates sex, age, climate, culture and every new axis.

    The opposite is the important half: only the *new* genes are optional.  A
    truncated string still loses a leg, so the legacy genes stay mandatory.
    """
    g = dna.random_genes(21)
    optional = set(dna.DEMOGRAPHIC) | set(dna.PHENOTYPE_CATEGORIES) | set(dna.MORPH) | set(dna.PIGMENT)
    old = ";".join(item for item in dna.encode(g).split(";")
                   if item.partition("=")[0] not in optional)
    out = dna.decode(old)
    for key, value in {**demography.DEFAULTS, **dna.phenotype.DEFAULTS}.items():
        assert out[key] == value, f"{key} did not default"
    text = dna.encode(out)
    assert [item.partition("=")[0] for item in text.split(";")] == list(dna.GENES)
    for key in dna.LEGACY_CATEGORICAL + dna.COLOUR_GENES + dna.SHAPE_GENES:
        assert out[key] == g[key], f"{key} changed while defaulting the new tail"


def test_demography_changes_the_face_and_not_the_dna_contract():
    """Sex, age and climate sculpt the same construction; culture does not sculpt it."""
    base = _genes(7, jaw=0.0, nose=0.0, eye_w=0.0)
    female = _style(dict(base, sex="female"))
    male = _style(dict(base, sex="male"))
    child = _style(dict(base, age="child"))
    elder = _style(dict(base, age="elder"))
    arid = _style(dict(base, climate="arid"))
    temperate = _style(dict(base, climate="temperate"))

    assert male.jaw_w > female.jaw_w
    assert male.neck_w > female.neck_w
    assert child.eye_w > elder.eye_w
    assert child.jaw_w < elder.jaw_w
    assert arid.nose_w > temperate.nose_w

    # Dress is not a face: the culture must not touch a single geometric parameter.
    north = _style(dict(base, culture="north"))
    desert = _style(dict(base, culture="desert"))
    assert north.p == desert.p, "culture changed the face instead of the clothing"

    # It must still reach the picture.  The collar, the garment colour and the
    # grooming differ, while the skull and features do not.
    assert demography.collar(dict(base, culture="north")) != \
        demography.collar(dict(base, culture="desert"))
    hn = hues.from_genes(dict(base, culture="north"))
    hd = hues.from_genes(dict(base, culture="desert"))
    assert not np.allclose(hn.cloth, hd.cloth), "culture did not reach the garment"
    assert not np.allclose(hn.cloth_trim, hd.cloth_trim), "culture did not reach the trim"
    _, sn = _hair(dict(base, culture="north"))
    _, sd = _hair(dict(base, culture="desert"))
    assert sn.p["gloss"] != sd.p["gloss"], "culture did not reach grooming"


def test_climate_reaches_skin_hair_and_feature_geometry_with_pinned_genes():
    """Pinning every continuous gene must not turn climate into a label.

    The same base colours, morphs and pigments still read differently in a cold,
    dry, tropical or highland lineage: the colour script shifts, the nose/brow
    construction shifts, and the hair sits differently.
    """
    base = _genes(31, **{key: 0.0 for key in dna.PIGMENT})
    variants = {name: dict(base, climate=name) for name in demography.CLIMATES}
    skins = {name: hues.from_genes(g).skin for name, g in variants.items()}
    hairs = {name: hues.from_genes(g).hair for name, g in variants.items()}
    assert len({tuple(np.round(c, 6)) for c in skins.values()}) == len(skins), \
        "climate did not reach the skin"
    assert len({tuple(np.round(c, 6)) for c in hairs.values()}) == len(hairs), \
        "climate did not reach the hair colour"
    styles = {name: _style(g) for name, g in variants.items()}
    assert styles["arid"].nose_w > styles["arctic"].nose_w
    assert styles["tropical"].mouth_w > styles["arctic"].mouth_w
    assert styles["highland"].cheek_w > styles["temperate"].cheek_w
    hair_params = {name: _hair(g)[1].p["vol"] for name, g in variants.items()}
    assert len({round(v, 6) for v in hair_params.values()}) >= 3, \
        "climate did not reach the hair mass"


def test_age_greys_the_hair_colour_script_not_the_gene():
    g = _genes(4, age="adult", climate="temperate", **{key: 0.0 for key in dna.PIGMENT})
    adult = hues.from_genes(g)
    grey = hues.from_genes(dict(g, age="elder"))
    assert np.allclose(adult.hair, palette.hex_to_srgb(g["hair"]))
    assert not np.allclose(grey.hair, adult.hair), "elder hair stayed the raw gene colour"
    assert palette.srgb_to_oklch(grey.hair)[0] > palette.srgb_to_oklch(adult.hair)[0]


def test_demographic_variants_are_distinct_and_deterministic():
    """A gene that does not reach the picture is a label, not a portrait gene."""
    base = _genes(23)
    variants = {
        "female": dict(base, sex="female"),
        "male": dict(base, sex="male"),
        "androgynous": dict(base, sex="androgynous"),
    }
    hashes = {name: _hash(portraitlib.render(g, size=64)) for name, g in variants.items()}
    assert len(set(hashes.values())) == len(hashes), "sex did not reach the pixels"

    old = _hash(portraitlib.render(dict(base, age="child"), size=64))
    new = _hash(portraitlib.render(dict(base, age="elder"), size=64))
    assert old != new, "age did not reach the pixels"

    desert = _hash(portraitlib.render(dict(base, culture="desert"), size=64))
    north = _hash(portraitlib.render(dict(base, culture="north"), size=64))
    assert desert != north, "culture did not reach the clothing"

    # The same genes must reproduce the same picture; the demography is not a
    # hidden random input beside the genome.
    again = _hash(portraitlib.render(dict(base, age="elder"), size=64))
    assert again == new


def test_breeding_gives_a_child_a_child_stage():
    a, b = dna.random_genes(5), dna.random_genes(1005)
    for seed in range(40):
        child = dna.breed(a, b, seed=seed)
        assert child["age"] in ("child", "youth")
        assert child["sex"] in demography.SEXES
        assert child["climate"] in demography.CLIMATES
        assert child["culture"] in demography.CULTURES



# ------------------------------------------------------- phenotype and pigment
def _neutral_phenotype(genes: dict) -> dict:
    out = dict(genes)
    out.update({key: 0.0 for key in dna.MORPH})
    out.update({key: 0.0 for key in dna.PIGMENT})
    return out


def _style_signature(st) -> tuple:
    return tuple((key, round(float(value), 6)) for key, value in sorted(st.p.items())
                 if not isinstance(value, str))


def _assert_face_invariants(st, where: str) -> None:
    assert np.all(np.isfinite([float(value) for value in st.p.values()
                               if not isinstance(value, str)])), f"{where}: non-finite style"
    assert st.brow_y > st.eye_y, f"{where}: brow below the eye"
    assert st.hairline_y > st.brow_y, f"{where}: hairline below the brow"
    assert st.nose_y < st.eye_y, f"{where}: nose above the eye"
    assert st.mouth_y < st.nose_y, f"{where}: mouth above the nose"
    assert st.chin_y < st.mouth_y, f"{where}: chin above the mouth"
    assert st.chin_w < st.jaw_w < st.cheek_w, f"{where}: jaw does not taper"
    assert st.eye_dx + st.eye_w * 0.5 + 0.045 <= st.head_w + 1e-9, \
        f"{where}: an eye hangs off the skull"
    assert st.eye_w <= st.head_w * 0.50 + 1e-9, f"{where}: eye too wide for the skull"
    assert st.eye_w <= st.cheek_w * 0.50 + 1e-9, f"{where}: eye too wide for the cheek"
    assert st.eye_w <= st.eye_dx * 1.28 + 1e-9, f"{where}: eyes almost touching"
    assert st.neck_w < st.jaw_w, f"{where}: neck thicker than the jaw"


def test_every_phenotype_category_has_its_own_geometry():
    """A named family is a real axis, not a caption.

    Each value must produce a distinct style signature, and every one of those
    signatures must still satisfy the anatomy guard.  Freckles and facial hair are
    paint patterns, not geometry, so they are covered by the pixel tests below.
    """
    base = _neutral_phenotype(_genes(31))
    for key, choices in phenotype.CATEGORIES.items():
        if key in ("freckles", "facial_hair"):
            continue
        signatures = []
        for value in choices:
            st = _style(dict(base, **{key: value}))
            _assert_face_invariants(st, f"{key}={value}")
            signatures.append(_style_signature(st))
        assert len(set(signatures)) == len(choices), f"{key} has duplicate variants"


def test_every_morph_gene_reaches_geometry_or_hair():
    """Each continuous axis must move something a viewer can see.

    Four axes are hair parameters rather than style parameters; those are checked
    through the built hairstyle and through a rendered pixel hash.  The rest must
    change the style dictionary, and both extremes must stay inside the guard.
    """
    base = _neutral_phenotype(_genes(44))
    base["style"] = "shounen"
    base["hair_style"] = "wild"    # a parted fringe, so every hair axis is visible
    neutral_style = _style(base)
    neutral_sig = _style_signature(neutral_style)
    hair_keys = {"hair_volume", "fringe_length", "fringe_part", "side_lock"}
    for key in dna.MORPH:
        for value in (-1.0, 1.0):
            st = _style(dict(base, **{key: value}))
            _assert_face_invariants(st, f"{key}={value}")
        if key in hair_keys:
            st0 = _style(base)
            p0 = hair._params(base["hair_style"], st0, base)
            st1 = _style(dict(base, **{key: 0.8}))
            p1 = hair._params(base["hair_style"], st1, dict(base, **{key: 0.8}))
            assert p0 != p1, f"{key} never reached the hairstyle"
        else:
            assert _style_signature(_style(dict(base, **{key: 0.8}))) != neutral_sig, \
                f"{key} never reached the style"
    # Hair axes must reach pixels, not just a parameter dict.
    base_hash = _hash(portraitlib.render(base, size=48))
    for key in hair_keys:
        assert _hash(portraitlib.render(dict(base, **{key: 0.8}), size=48)) != base_hash, \
            f"{key} never reached the pixels"


def test_every_pigment_gene_moves_the_colour_script():
    base = _genes(52)
    neutral = dict(base, **{key: 0.0 for key in dna.PIGMENT})
    h0 = hues.from_genes(neutral)
    for key in dna.PIGMENT:
        h1 = hues.from_genes(dict(neutral, **{key: 0.8}))
        changed = any(not np.allclose(getattr(h0, slot), getattr(h1, slot))
                      for slot in h0.__slots__)
        assert changed, f"{key} never reached the colour script"
    # The strong skin axis must also survive to the picture.
    a = _hash(portraitlib.render(dict(neutral, skin_melanin=-1.0), size=64))
    b = _hash(portraitlib.render(dict(neutral, skin_melanin=1.0), size=64))
    assert a != b, "skin melanin did not reach the portrait"


def test_freckle_patterns_reach_the_pixels():
    base = _neutral_phenotype(_genes(63))
    hashes = {name: _hash(portraitlib.render(dict(base, freckles=name), size=72))
              for name in phenotype.FRECKLE_PATTERNS}
    assert len(set(hashes.values())) == len(hashes), "freckle patterns are only labels"
    again = _hash(portraitlib.render(dict(base, freckles="full"), size=72))
    assert again == hashes["full"], "the freckle pattern is not deterministic"


def test_facial_hair_styles_reach_the_pixels():
    """Facial hair is a paint pattern, so it is verified in the picture.

    Every style must change the pixels, the result must be deterministic, and the
    sitter gates must hold: a woman renders exactly as if the gene were absent,
    because the gate belongs to the face, not to the wire format.
    """
    base = _neutral_phenotype(_genes(63))
    base.update(sex="male", age="adult")
    hashes = {name: _hash(portraitlib.render(dict(base, facial_hair=name), size=96))
              for name in phenotype.FACIAL_HAIR}
    assert len(set(hashes.values())) == len(hashes), "facial-hair styles are only labels"
    again = _hash(portraitlib.render(dict(base, facial_hair="full_beard"), size=96))
    assert again == hashes["full_beard"], "the facial-hair style is not deterministic"


def test_facial_hair_is_gated_on_the_sitter():
    """The sex/age gate is a property of the face, not of the wire format.

    The genome string seeds every stroke, so two genomes that differ in
    ``facial_hair`` cannot be compared as whole pictures.  This drives the paint
    pass directly instead: for a woman and for a child the pass must be a no-op,
    and for an adult man the same call must change the canvas.
    """
    from gen.portraits import detail  # noqa: PLC0415

    base = _neutral_phenotype(_genes(63))

    def painted(genes: dict) -> str:
        c = portraitlib.canvas(64, background=(0.9, 0.9, 0.9))
        st = _style(genes)
        h = hues.from_genes(genes)
        rng = np.random.default_rng(0)
        detail.paint_facial_hair(c, st, h, genes, 0.0, np.ones((64, 64)), rng)
        return _hash(c)

    for genes in (dict(base, sex="female", age="adult"),
                  dict(base, sex="male", age="child"),
                  dict(base, sex="female", age="child")):
        assert painted(dict(genes, facial_hair="none")) == \
            painted(dict(genes, facial_hair="full_beard")), \
            f"facial hair was painted on {genes['sex']}/{genes['age']}"
    clean = painted(dict(base, sex="male", age="adult", facial_hair="none"))
    bearded = painted(dict(base, sex="male", age="adult", facial_hair="full_beard"))
    assert clean != bearded, "an adult man's beard never reached the canvas"


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
