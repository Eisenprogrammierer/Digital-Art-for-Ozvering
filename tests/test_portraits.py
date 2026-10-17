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
from gen.portraits import cel, dna, expression, face, hair, head, hues, eyes  # noqa: E402
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
    rng = np.random.default_rng(dna.seed_of(genes))
    return st, hair.build(st, genes, rng, float(genes["turn"]))


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
        ";".join(items[:-1]),
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
        for key in dna.SHAPE_GENES:
            assert -1.0 <= child[key] <= 1.0


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
def test_the_fringe_leaves_the_eyes_clear():
    """A lock over an eye is the difference between a face and a curtain."""
    for seed in range(90):
        g = _genes(seed)
        st, h = _hair(g)
        pts = np.array([[st.eye_dx, st.eye_y], [-st.eye_dx, st.eye_y], [0.0, st.eye_y]])
        d = h.front_mass.f(pts[:, 0], pts[:, 1])
        assert (d < -0.002).sum() == 0, f"seed {seed} ({g['hair_style']}): hair over an eye"


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
                shapes = {
                    "skull": head.silhouette(st, turn),
                    "neck": head.neck(st, turn),
                    "shoulders": head.shoulders(st),
                    "hair": h.mass,
                    "fringe": h.front_mass,
                    "cap": h.cap,
                    "brow": face.brows(st, turn, expr)[0],
                }
                for key, value in eyes.shapes(st, expr).items():
                    if isinstance(value, Shape):
                        shapes[f"eye_{key}"] = value
                inner, _ = face.mouth_interior(st, turn, expr)
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
    g = dna.random_genes(4)
    h = hues.from_genes(g)
    assert np.allclose(h.skin, palette.hex_to_srgb(g["skin"]))
    assert np.allclose(h.hair, palette.hex_to_srgb(g["hair"]))
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
