"""Regression tests for the heraldry toolkit.

The property that matters most here is the one the whole design rests on: a
generated coat must be *legal* -- the rule of tincture obeyed at every charge
anchor -- and the blazon text must be a faithful reading of the spec, in both
languages.  Those are tested against the generator, not against hand-written
examples, so a change to the composition rules cannot silently break the
guarantee.

Run either way::

    python tests/test_heraldry.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import palette  # noqa: E402
from export import sheet as sheetlib  # noqa: E402
from gen.heraldry import blazon, charges, cloth, crest, field, flags, ordinaries, paint, pole, shields, tinctures  # noqa: E402


# --------------------------------------------------------------- tinctures
def test_tincture_rule_by_class():
    assert tinctures.check("gules", "or")[0]          # metal on colour
    assert tinctures.check("or", "azure")[0]          # colour on metal
    assert not tinctures.check("or", "argent")[0]     # metal on metal
    assert not tinctures.check("gules", "azure")[0]   # colour on colour
    assert tinctures.is_metal("argent") and tinctures.is_colour("vert")


def test_furs_are_neutral():
    for fur in ("ermine", "counter_ermine", "vair", "counter_vair", "potent"):
        assert tinctures.is_fur(fur)
        assert tinctures.check(fur, "or")[0]
        assert tinctures.check("azure", fur)[0]
        assert tinctures.check(fur, "gules")[0]


def test_tincture_names_aliases_and_variants():
    assert tinctures.get("gold").key == "or"
    assert tinctures.get("white").key == "argent"
    assert len(tinctures.names()) >= 12
    v = tinctures.variant("azure", d_lightness=0.05)
    # A variant is a *new* tincture of the same class, not a renamed one: it is
    # how a game gives a faction its own shade without leaving the system.
    assert v.kind == tinctures.COLOUR and v.key != "azure"
    assert tinctures.variant("or", d_chroma=-0.02).kind == tinctures.METAL


def test_russian_adjectives_are_complete():
    for name in tinctures.names():
        adj = tinctures.get(name).ru_adj
        for key in ("m", "f", "n", "prep", "instr_sg", "instr_pl", "nom_pl"):
            assert key in adj, f"{name} lacks ru_adj[{key!r}]"
        for gender in ("m", "f", "n"):
            assert adj["instr_sg"].get(gender), f"{name} lacks instrumental {gender}"


# ------------------------------------------------------------------- field
def _coverage(layers, hw, hh, n=8000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-hw, hw, n)
    y = rng.uniform(-hh, hh, n)
    cov = np.zeros(n, dtype=int)
    for region, _ in layers:
        cov += np.asarray(region(x, y)) < 0
    return cov


def test_divisions_partition_the_field():
    # A division is a partition: every point belongs to exactly one region, at
    # every aspect ratio the silhouettes actually use.
    for hw, hh in ((1.0, 1.0), (0.8, 1.0), (1.5, 1.0), (0.6, 1.0)):
        for name, build in sorted(field.DIVISIONS.items()):
            cov = _coverage(build(hw, hh), hw, hh)
            assert cov.max() == 1, f"{name} overlaps at {hw}:{hh}"
            assert (cov == 1).mean() > 0.995, f"{name} has holes at {hw}:{hh}"


def test_variations_cover_the_field():
    # A variation may *overlay* its base coat (ermine, vair, a lozenge lattice),
    # so overlapping is fine, but nothing may fall through.
    for hw, hh in ((1.0, 1.0), (0.8, 1.0), (1.5, 1.0)):
        for name, build in sorted(field.VARIATIONS.items()):
            cov = _coverage(build(hw, hh), hw, hh)
            assert cov.min() >= 1, f"{name} leaves a hole at {hw}:{hh}"


def test_chevronny_reaches_the_base():
    # Regression: the chevron coordinate peaks inside the box, so measuring the
    # extent from the corners only left the lowest chevron band short.
    layers = field.VARIATIONS["chevronny"](1.0, 1.0)
    cov = _coverage(layers, 1.0, 1.0)
    assert (cov == 1).mean() > 0.995


def test_field_tincture_at_uses_the_top_layer():
    layers = [field.whole_plane(), field.halfplane(0.0, -1.0)]
    names = [tinctures.get("argent"), tinctures.get("gules")]
    pairs = list(zip(layers, names))
    assert field.tincture_at(pairs, 0.0, 0.5).key == "argent"
    assert field.tincture_at(pairs, 0.0, -0.5).key == "gules"


def test_field_layers_reports_the_declared_tinctures():
    # A fur is painted as its component tinctures laid over one another, so it
    # comes back as more than one layer; that is a rendering detail, and the
    # *rule* of tincture must still see the fur itself as neutral.
    fspec = {"kind": "plain", "tinctures": ["ermine"]}
    layers = field.field_layers(fspec, 1.0, 1.0)
    assert len(layers) >= 2
    assert {t.key for _, t in layers} <= {"argent", "sable"}

    neutral = {"shield": "heater", "field": fspec,
               "charges": [{"name": "roundel", "count": 1, "arrangement": "one",
                            "tincture": "or"}]}
    assert blazon.validate(neutral) == []

    div = field.field_layers(
        {"kind": "division", "name": "per_pale", "tinctures": ["gules", "azure"]}, 1.0, 1.0
    )
    assert len(div) == 2
    assert field.tincture_at(div, -0.5, 0.0).key == "gules"
    assert field.tincture_at(div, 0.5, 0.0).key == "azure"


# -------------------------------------------------------------- ordinaries
def test_every_ordinary_builds_and_lies_inside_the_field():
    xs = np.linspace(-1.0, 1.0, 101)
    gx, gy = np.meshgrid(xs, xs)
    for name in ordinaries.names():
        o = ordinaries.get(name)
        if "needs_field" in o.tags:
            continue
        d = o.build(1.0, 1.0)(gx, gy)
        inside = float((np.asarray(d) < 0).mean())
        assert 0.03 < inside < 0.95, f"{name} covers {inside:.3f} of the box"


def test_bordure_needs_the_silhouette_and_forms_a_ring():
    try:
        ordinaries.sdf_for("bordure", 1.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a bordure without a silhouette must be refused")

    fn, hw, hh = shields.get("heater")
    d = ordinaries.sdf_for("bordure", hw, hh, field=fn)
    assert d(0.0, 0.0) > 0.0                      # hollow in the middle
    assert d(-hw * 0.97, 0.0) < 0.0               # a band along the edge
    assert d(-hw * 0.40, 0.0) > 0.0


def test_ordinary_thickness_scales_with_thin():
    xs = np.linspace(-1.0, 1.0, 201)
    gx, gy = np.meshgrid(xs, xs)
    thin = float((ordinaries.sdf_for("fess", 1.0, 1.0)(0.0, gy) < 0).mean())
    thick = float((ordinaries.sdf_for("fess", 1.0, 1.0, thin=2.0)(0.0, gy) < 0).mean())
    assert thick > thin


# ----------------------------------------------------------------- charges
def test_every_charge_is_non_empty_and_fits_its_unit_box():
    xs = np.linspace(-1.02, 1.02, 205)
    gx, gy = np.meshgrid(xs, xs)
    for name in charges.names():
        d = np.asarray(charges.get(name)(gx, gy))
        inside = d < 0
        assert inside.any(), f"{name} renders nothing"
        rows = np.where(inside.any(axis=1))[0]
        cols = np.where(inside.any(axis=0))[0]
        extent = max(abs(xs[rows].min()), abs(xs[rows].max()),
                     abs(xs[cols].min()), abs(xs[cols].max()))
        assert extent <= 1.002, f"{name} escapes the unit box ({extent:.3f})"
        assert extent > 0.45, f"{name} is suspiciously small ({extent:.3f})"


def test_charge_russian_forms_exist():
    for name in charges.names():
        sg, pl, adjp = charges.gen_forms(name)
        assert sg and pl
        assert adjp is None or isinstance(adjp, str)
        assert charges.instr_sg(name)
        assert charges.get(name).ru and charges.get(name).ru_pl


def test_roundel_aliases_carry_their_own_tincture():
    for alias, tincture in [("bezant", "or"), ("plate", "argent"), ("hurt", "azure"),
                            ("torteau", "gules"), ("pellet", "sable"), ("pomme", "vert"),
                            ("golpe", "purpure")]:
        assert charges.get(alias).forced == tincture


# ------------------------------------------------------------------ shapes
def test_shields_contain_the_origin_and_are_bounded():
    for name in shields.names():
        fn, hw, hh = shields.get(name)
        assert float(fn(0.0, 0.0)) < 0.0, name
        assert float(fn(3.0 * hw, 3.0 * hh)) > 0.0, name
        assert hw > 0.0 and hh > 0.0


def test_flag_silhouettes_respect_the_proportion():
    for name in shields.FLAG_SHAPES:
        fn, hw, hh = shields.flag((5, 2), name)
        assert np.isclose(hw / hh, 2.5), name
        assert float(fn(0.0, 0.0)) < 0.0, name
        assert float(fn(0.0, 2.0 * hh)) > 0.0, name


# ------------------------------------------------------------------ blazon
def test_blazon_text_known_examples():
    chief = {"shield": "heater", "field": {"kind": "plain", "tinctures": ["argent"]},
             "ordinary": {"name": "chief", "tincture": "gules"}}
    assert blazon.blazon_text(chief, "en") == "Argent, a chief Gules."
    assert blazon.blazon_text(chief, "ru") == "В серебряном поле червлёная глава."

    bezants = {"shield": "heater", "field": {"kind": "plain", "tinctures": ["gules"]},
               "charges": [{"name": "bezant", "count": 5, "arrangement": "five"}]}
    assert blazon.blazon_text(bezants, "en") == "Gules, five bezants."
    assert blazon.blazon_text(bezants, "ru") == "В червлёном поле пять безантов."

    lilies = {"shield": "heater", "field": {"kind": "plain", "tinctures": ["gules"]},
              "charges": [{"name": "fleur_de_lis", "count": 4, "arrangement": "four",
                           "tincture": "or"}]}
    assert blazon.blazon_text(lilies, "ru") == "В червлёном поле четыре золотые лилии."


def test_russian_numerals_govern_the_noun_with_a_modifier():
    # Two to four take a genitive plural adjective over a genitive singular head,
    # and a charge with a modifier of its own must not break the phrase.
    spec = {"shield": "heater", "field": {"kind": "plain", "tinctures": ["gules"]},
            "charges": [{"name": "cross_pattee", "count": 3, "arrangement": "three",
                         "tincture": "or"}]}
    assert blazon.blazon_text(spec, "ru") == "В червлёном поле три золотых лапчатых креста."
    spec["charges"][0]["count"] = 5
    spec["charges"][0]["arrangement"] = "five"
    assert blazon.blazon_text(spec, "ru") == "В червлёном поле пять золотых лапчатых крестов."


def test_validate_accepts_a_legal_coat():
    spec = {"shield": "heater", "field": {"kind": "plain", "tinctures": ["gules"]},
            "ordinary": {"name": "fess", "tincture": "or"}}
    assert blazon.validate(spec) == []


def test_validate_catches_metal_on_metal_and_invisibility():
    metal = {"shield": "heater", "field": {"kind": "plain", "tinctures": ["or"]},
             "charges": [{"name": "roundel", "count": 1, "arrangement": "one",
                          "tincture": "argent"}]}
    assert any("metal on metal" in w for w in blazon.validate(metal))

    same = {"shield": "heater", "field": {"kind": "plain", "tinctures": ["gules"]},
            "charges": [{"name": "roundel", "count": 1, "arrangement": "one",
                         "tincture": "gules"}]}
    assert any("invisible" in w for w in blazon.validate(same))


def test_validate_judges_an_ordinary_over_every_division_it_crosses():
    # A bordure runs all the way round, so a field carrying both a metal and a
    # colour cannot give it a compliant tincture -- and that must be reported.
    spec = {"shield": "heater",
            "field": {"kind": "division", "name": "per_pale", "tinctures": ["gules", "argent"]},
            "ordinary": {"name": "bordure", "tincture": "azure"}}
    assert any("bordure" in w for w in blazon.validate(spec))


def test_a_charge_spanning_a_partition_is_not_reported():
    # Ordinary heraldry: a lion over a per pale field is normal, not a fault.
    spec = {"shield": "heater",
            "field": {"kind": "division", "name": "per_pale", "tinctures": ["gules", "azure"]},
            "charges": [{"name": "lion", "count": 1, "arrangement": "one", "tincture": "or"}]}
    assert blazon.validate(spec) == []


def test_generated_arms_are_legal_across_many_seeds():
    for seed in range(120):
        spec = blazon.random_arms(seed=seed)
        warnings = blazon.validate(spec)
        assert warnings == [], f"seed {seed}: {warnings} ({blazon.blazon_text(spec, 'en')})"


def test_generated_arms_are_deterministic():
    assert blazon.random_arms(seed=41) == blazon.random_arms(seed=41)
    assert blazon.random_arms(seed=41) != blazon.random_arms(seed=42)


def test_generated_arms_have_both_blazons():
    for seed in range(12):
        spec = blazon.random_arms(seed=seed)
        en, ru = blazon.blazon_text(spec, "en"), blazon.blazon_text(spec, "ru")
        assert en.endswith(".") and ru.endswith(".")
        assert ru.startswith("В ")
        assert len(en) > 4


# ------------------------------------------------------------------- flags
def test_flag_builders_place_the_first_tincture_at_the_chief_or_hoist():
    h = flags.tricolour("argent", "azure", "gules")
    fspec = h["field"]
    layers = field.field_layers(fspec, 1.5, 1.0)
    assert field.tincture_at(layers, 0.0, 0.8).key == "argent"
    assert field.tincture_at(layers, 0.0, 0.0).key == "azure"
    assert field.tincture_at(layers, 0.0, -0.8).key == "gules"

    v = flags.tricolour("sable", "or", "gules", direction="vertical")
    vlayers = field.field_layers(v["field"], 1.5, 1.0)
    assert field.tincture_at(vlayers, -1.3, 0.0).key == "sable"
    assert field.tincture_at(vlayers, 1.3, 0.0).key == "gules"


def test_flag_builders_describe_themselves():
    cases = [
        (flags.tricolour("argent", "azure", "gules"), "tricolour"),
        (flags.bicolour("or", "gules"), "bicolour"),
        (flags.nordic_cross("gules", "or", fimbriation="azure"), "Nordic cross"),
        (flags.cross_flag("azure", "argent"), "Cross"),
        (flags.saltire_flag("azure", "argent"), "Saltire"),
        (flags.canton_flag("gules", "azure"), "Canton"),
        (flags.hoist_triangle("azure", "or"), "Triangle at the hoist"),
        (flags.bordered("or", "gules"), "Border"),
        (flags.plain("vert"), "Plain field"),
    ]
    for spec, word in cases:
        en = flags.describe(spec, "en")
        ru = flags.describe(spec, "ru")
        assert word in en, (word, en)
        assert ru.endswith(".") and len(ru) > 8


def test_nordic_cross_is_offset_towards_the_hoist():
    spec = flags.nordic_cross("gules", "or")
    fn, hw, hh = blazon.resolve_shape(spec)
    d = blazon.compose(spec, fn, hw, hh)[1][0]
    # Sampled above the horizontal arm, which spans the whole width: the upright
    # sits towards the hoist, which is the whole point of the pattern.
    assert float(d(-hw * 0.62, hh * 0.6)) < 0.0   # the hoist side is on it
    assert float(d(0.0, hh * 0.6)) > 0.0          # the centre is not
    assert float(d(hw * 0.75, hh * 0.6)) > 0.0    # nor the fly side
    assert float(d(hw * 0.9, 0.0)) < 0.0          # the horizontal arm crosses

    # ... and the contrast that proves the offset is real: an ordinary cross is
    # centred, so the same sample lands on its upright.
    centred = flags.cross_flag("gules", "or")
    fn2, hw2, hh2 = blazon.resolve_shape(centred)
    dc = blazon.compose(centred, fn2, hw2, hh2)[1][0]
    assert float(dc(0.0, hh2 * 0.6)) < 0.0


def test_random_flags_are_deterministic_and_render():
    for seed in range(24):
        a = flags.random_flag(seed=seed)
        b = flags.random_flag(seed=seed)
        assert a == b
        img = flags.render(a, size=64)
        assert img.rgb.shape == (64, 64, 3)
        assert 0.0 <= img.a.min() and img.a.max() <= 1.0
    assert flags.random_flag(seed=1) != flags.random_flag(seed=2)


# ---------------------------------------------------------------------- cloth
def _lum(canvas):
    """Per-pixel luminance of the *drawn* part of a canvas, NaN elsewhere."""
    arr = np.asarray(canvas.to_uint8())
    rgb = arr[..., :3].astype(np.float64)
    return np.where(arr[..., 3] > 200, rgb.mean(axis=2), np.nan)


def _cloth_box(size, margin, hw=1.5, hh=1.0):
    """The pixel rectangle the cloth's own box maps to, ``(x0, x1, y0, y1)``.

    The frame is not the cloth: it carries the margin the cloth is going to move
    into, so the drawn extent of a hanging flag is the cloth *plus* however far the
    wave dragged it.  Anything that compares two hanging flags -- across sizes, or
    across a phase -- has to compare them on the cloth's own box, or it ends up
    measuring the margin in fractional pixels.
    """
    scale = size / max(2.0 * hw, 2.0 * hh) * (1.0 - 2.0 * margin)
    return (int(size / 2.0 - hw * scale), int(size / 2.0 + hw * scale),
            int(size / 2.0 - hh * scale), int(size / 2.0 + hh * scale))


def test_a_wave_is_a_spec_with_a_phase_from_its_seed():
    a = cloth.wave(3)
    assert a == cloth.wave(3)
    assert a["phase"] != cloth.wave(4)["phase"]
    assert 0.0 <= a["phase"] <= 2.0 * np.pi
    assert a["direction"] == "vertical"
    for wrong in (lambda: cloth.wave(0, direction="diagonal"),
                  lambda: cloth.wave(0, amplitude=-0.1),
                  lambda: cloth.wave(0, wavelength=0.0)):
        try:
            wrong()
        except ValueError:
            pass
        else:
            raise AssertionError("a wave that cannot be drawn should be refused")


def test_the_period_of_the_folds_is_a_whole_number_of_turns():
    """What a loop of a folded flag costs, in turns, and why it is not one.

    The second harmonic exists so that the pattern does *not* repeat along the flag,
    and the price of that is paid by an animator: the cloth returns only after a whole
    number of turns in which both terms do, which for this ratio is ten.  The claim is
    checked against the field rather than against the arithmetic that produced it --
    the fold at a phase and the fold a period later are the same fold -- and a cloth
    with one harmonic, which is the one an animation asks for, returns after one turn.
    """
    spec = cloth.wave(3)
    span = cloth.period(spec)
    assert np.isclose(span, 10.0 * 2.0 * np.pi)
    t = np.linspace(0.0, 1.0, 17)
    here = cloth.surface(spec, t)
    assert np.allclose(here, cloth.surface({**spec, "phase": spec["phase"] + span}, t),
                       atol=1e-12)
    assert not np.allclose(here, cloth.surface({**spec, "phase": spec["phase"] + 2.0 * np.pi}, t))
    plain = {**spec, "harmonic": 0.0}
    assert np.isclose(cloth.period(plain), 2.0 * np.pi)
    assert np.allclose(cloth.surface(plain, t),
                       cloth.surface({**plain, "phase": plain["phase"] + 2.0 * np.pi}, t),
                       atol=1e-12)


def test_a_wave_of_zero_amplitude_is_the_identity():
    """The one case where the wave must be exact rather than merely close.

    Waving resamples the canvas at fractional pixel coordinates, so getting the
    pixel-centre convention or the half-pixel offset wrong shifts the whole cloth
    by half a pixel -- invisible, and wrong.  An amplitude of zero says nothing
    moves; if anything moves anyway, the sampling is off.
    """
    spec = flags.random_flag(seed=5, ratio=(3, 2))
    flat = flags.render(spec, size=128).to_uint8()
    still = flags.render({**spec, "wave": cloth.wave(1, amplitude=0.0)}, size=128).to_uint8()
    assert np.array_equal(flat, still)


def test_the_cloth_stays_on_the_canvas():
    """What the widened margin is for.

    Displacing a flag along its folds moves pixels, and the edge they would leave
    by is the fly -- the one the eye is watching.  So every proportion and every
    silhouette shape has to keep its border cells empty, at any size.
    """
    for ratio in ((3, 2), (1, 1), (5, 3), (2, 3), (3, 5)):
        for shape in ("rectangle", "swallowtail", "pennant", "guidon"):
            for size in (64, 160):
                spec = flags.random_flag(seed=4, ratio=ratio, shape=shape, wave=True)
                a = flags.render(spec, size=size).a
                border = max(float(a[0].max()), float(a[-1].max()),
                             float(a[:, 0].max()), float(a[:, -1].max()))
                assert border < 1e-6, (ratio, shape, size, border)
                assert float(a.max()) > 0.9


def test_the_folds_redistribute_light_rather_than_taking_it_away():
    """A hanging flag is as bright as the same flag lying flat.

    Which is both true of cloth and the only way a contact sheet of flags can be
    read: a wave that simply darkened its flag would ruin the sheet the moment it
    was switched on.  The comparison is over the cloth only, because a waving flag
    is drawn a little smaller -- that is the margin doing its job, not a fold.
    """
    for seed in (0, 3, 7):
        spec = flags.random_flag(seed=seed, ratio=(3, 2), wave=True)
        flat = np.nanmean(_lum(flags.render({k: v for k, v in spec.items() if k != "wave"}, size=192)))
        hung = np.nanmean(_lum(flags.render(spec, size=192)))
        assert abs(hung / flat - 1.0) < 0.04, (seed, flat, hung)


def test_the_bands_run_across_the_folds_axis():
    """Vertical folds make vertical bands -- measured, not asserted about the code.

    The illumination of a folded surface varies *across* the fold, so a vertically
    waved flag must let its columns differ while its rows agree, and a horizontally
    waved one the other way round.  A plain field is used so that the only
    structure in the profile is the light: on a bicolour the rows would differ
    anyway, and the test would pass for the wrong reason.
    """
    def spread(canvas):
        lum = _lum(canvas)
        return float(np.nanstd(np.nanmean(lum, axis=0))), float(np.nanstd(np.nanmean(lum, axis=1)))

    plain = flags.plain("azure", ratio=(3, 2))
    shape = cloth.wave(2, amplitude=0.05, wavelength=0.25, harmonic=0.0)
    vc, vr = spread(flags.render({**plain, "wave": shape}, size=192))
    hc, hr = spread(flags.render({**plain, "wave": {**shape, "direction": "horizontal"}}, size=192))
    assert vc > 4.0 * vr, (vc, vr)
    assert hr > 4.0 * hc, (hc, hr)


def test_the_folds_are_the_same_folds_at_any_size():
    """A fold is a fraction of the flag, not a number of pixels.

    The wave is evaluated in world coordinates, so the same spec at 96 px and at
    384 px must have the same folds -- not merely the same *number* of them, which
    is what the dominant frequency of the light profile measures, but the same
    folds: the three profiles are read onto a common *world* axis and compared.  Two
    details make that comparison mean what it says.  The profiles are taken inside
    the cloth's own box, inset by what the wave is allowed to drag -- outside it is
    either margin, or the cloth that has moved into it -- and the sampling axis is
    in world units rather than in columns, because the box lands on a different
    fractional pixel at every size and reading by column index would turn that half
    pixel into a phase drift the correlation would report as a different fold.  A
    wave evaluated in pixel coordinates -- the easy mistake, since the renderer's
    field *is* a pixel grid -- would multiply the folds with the resolution, and
    neither the frequency nor the comparison would survive that.
    """
    wave = cloth.wave(3, amplitude=0.05, wavelength=0.25, harmonic=0.0)
    spec = {**flags.plain("azure", ratio=(3, 2)), "wave": wave}
    margin = flags.FLAG_STYLE["margin"] + 1.6 * cloth.reach(wave)
    profiles, folds = [], []
    for size in (96, 192, 384):
        scale = size / 3.0 * (1.0 - 2.0 * margin)
        inset = int(np.ceil(cloth.reach(wave) * size)) + 1
        x0, x1 = int(size / 2.0 - 1.5 * scale) + inset, int(size / 2.0 + 1.5 * scale) - inset
        prof = np.nanmean(_lum(flags.render(spec, size=size))[:, x0:x1], axis=0)
        spectrum = np.fft.rfft(prof - prof.mean())
        folds.append(int(np.argmax(np.abs(spectrum[1:])) + 1))
        world = (np.arange(x0, x1) + 0.5 - size / 2.0) / scale
        profiles.append(np.interp(np.linspace(-1.5, 1.5, 128), world, prof))
    assert folds[0] >= 2, folds
    assert len(set(folds)) == 1, folds
    for other in profiles[1:]:
        assert float(np.corrcoef(profiles[0], other)[0, 1]) > 0.99
        assert float(np.abs(profiles[0] - other).max()) < 6.0


def test_a_hanging_flag_survives_a_round_trip_and_a_second_render():
    import json

    spec = flags.random_flag(seed=4, ratio=(3, 2), wave=True)
    again = json.loads(json.dumps(spec))
    assert again["wave"] == spec["wave"]
    first = flags.render(again, size=96).to_uint8().tobytes()
    assert first == flags.render(spec, size=96).to_uint8().tobytes()
    assert len(first) > 0


def test_a_crest_banner_flies_flat():
    """A crest is a sign, not a flag on a pole: the wave is opt-in, so a faction
    cannot acquire folds by accident, and its banner stays comparable to its shield.
    """
    assert "wave" not in crest.faction("Kingdom of Alder")["flag"]


# ------------------------------------------------------------- the animation
def test_a_waving_flag_is_one_flag_at_a_sequence_of_phases():
    """An animation is the spec sampled, not a set of pictures of the flag.

    Every frame is the call :func:`render` already makes, so a frame is a function of
    its phase and of nothing else: the same phase is the same bytes and the next phase
    is not.  The frames also have to come back the *same size*, because a sprite sheet
    is a grid of equal cells -- if the cloth's reach moved with the phase, the margin
    would move with it and no table could cut the sheet up.
    """
    spec = flags.random_flag(seed=7, wave=True)
    frames, phases = flags.waving(spec, count=6, size=96)
    assert len(frames) == len(phases) == 6
    assert {(f.w, f.h) for f in frames} == {(96, 96)}
    start = spec["wave"]["phase"]
    assert np.allclose(phases, [start + i * 2.0 * np.pi / 6 for i in range(6)])
    same = flags.render({**spec, "wave": {**spec["wave"], "phase": phases[0]}}, size=96)
    assert frames[0].to_uint8().tobytes() == same.to_uint8().tobytes()
    for i in range(5):
        assert frames[i].to_uint8().tobytes() != frames[i + 1].to_uint8().tobytes()


def test_the_loop_closes_because_the_wave_is_periodic():
    """The claim a game cares about: the frame after the last is the first.

    Not because a list was built to wrap -- there is no list -- but because the field
    being sampled is a sinusoid, and a sinusoid one turn later is itself.  So the test
    is an equality of bytes between two independently drawn sequences: the animation
    that starts at phase ``p`` and the one that starts at ``p + 2*pi`` are the same
    animation, which is what a seamless loop *is*.

    The cloth is asked for one harmonic here, because that is the cloth whose period is
    one turn: the default second harmonic is deliberately not commensurate with the
    first (see :func:`gen.heraldry.cloth.period`), and a loop cut at one turn of *that*
    cloth is a loop with a seam.  Which is exactly the sort of thing an animator has to
    know, so it is a function and a test rather than a comment.
    """
    spec = flags.random_flag(seed=7, wave={"harmonic": 0.0})
    frames, phases = flags.waving(spec, count=4, size=96)
    turn = {**spec, "wave": {**spec["wave"], "phase": spec["wave"]["phase"] + 2.0 * np.pi}}
    later, later_phases = flags.waving(turn, count=4, size=96)
    for a, b in zip(frames, later):
        assert a.to_uint8().tobytes() == b.to_uint8().tobytes()
    assert np.allclose(later_phases, np.asarray(phases) + 2.0 * np.pi)
    # and the axis is uniform rather than a list of chosen moments: sampling every
    # other phase of a slow cycle is the same drawing as sampling every phase of a
    # fast one, because there is only one function here.
    coarse, _ = flags.waving(spec, count=2, size=96)
    for i, frame in enumerate(coarse):
        assert frame.to_uint8().tobytes() == frames[2 * i].to_uint8().tobytes()


def test_a_finer_sampling_is_the_same_motion_sampled_finer():
    """Why more frames is smoother rather than different.

    The frames are samples of a continuous field, so a quarter of the step between
    phases is a quarter of the change between pictures: an animation of twelve frames
    and one of forty-eight are the same flap, and the second is not a better drawing of
    a different one.  A test that counted frames would pass on a set of interpolated
    stills; this one measures the motion, which is the thing the player sees.
    """
    spec = flags.random_flag(seed=7, wave=True)

    def step(count):
        frames, _ = flags.waving(spec, count=count, size=64)
        a = [f.to_uint8().astype(np.int32) for f in frames]
        return float(np.mean([np.abs(a[i] - a[i + 1]).mean() for i in range(count - 1)]))

    coarse, fine = step(4), step(16)
    assert 0.0 < fine < coarse * 0.75, (coarse, fine)


def test_a_flag_that_does_not_wave_has_no_frames():
    """One frame is not an animation, and a flag at rest is not a frame of a wave.

    The contact sheet is where a flag that is not flying belongs.  Asking it for
    phases would mean inventing motion it does not have, and every frame here is drawn
    rather than interpolated -- so a spec with no wave is refused, not given a still.
    """
    spec = flags.random_flag(seed=7)
    assert "wave" not in spec
    flying = flags.random_flag(seed=7, wave=True)
    for wrong in (lambda: flags.waving(spec),
                  lambda: flags.waving(flying, count=0),
                  lambda: flags.waving(flying, turns=0.0)):
        try:
            wrong()
        except ValueError:
            pass
        else:
            raise AssertionError("a spec with no wave has no phases to be drawn at")


def test_a_sprite_sheet_is_a_grid_a_table_and_a_timeline():
    """The three things an engine needs, and the one thing it must not have to do.

    A sprite sheet is not a contact sheet: no captions, a clear background so that a
    filter cannot pull a neighbour into a frame, and a table of rectangles that says
    where each frame is -- with the value of the axis it was drawn at, so the sheet can
    be regenerated from its numbers rather than merely looked at.  The rectangles are
    checked against the pixels rather than against the formula that produced them,
    because a table that is wrong is the failure a player would see.
    """
    spec = flags.random_flag(seed=7, wave=True, pole=True)
    frames, phases = flags.waving(spec, count=5, size=48)
    canvas, info = sheetlib.build(frames, cols=3, padding=2, fps=8.0, params=phases,
                                  background=None)
    assert info["cols"] == 3 and info["rows"] == 2
    assert info["cell"] == [48, 48] and info["size"] == [canvas.w, canvas.h]
    assert info["fps"] == 8.0 and info["loop"] is True
    assert [f["index"] for f in info["frames"]] == [0, 1, 2, 3, 4]
    assert np.allclose([f["param"] for f in info["frames"]], phases)
    a = canvas.to_uint8()
    drawn = 0
    for frame, entry in zip(frames, info["frames"]):
        x, y, w, h = entry["rect"]
        assert np.array_equal(a[y:y + h, x:x + w], frame.to_uint8())
        drawn += int((frame.to_uint8()[..., 3] > 0).sum())
    assert int((a[..., 3] > 0).sum()) == drawn, "the padding is not empty"


def test_a_sheet_refuses_frames_that_are_not_the_same_size():
    """An atlas trims its members; a sheet cannot trim its frames.

    Frames of one animation are one size by construction, and an engine that has to
    re-centre every frame is an engine with a character that rolls -- so unequal frames
    are refused with the name of the thing that packs them.  The rest of the list is
    the other way a sheet fails quietly: a frame rate of zero, a padding that overlays
    the frames, a timeline that does not line up with the pictures, and no frames.
    """
    spec = flags.random_flag(seed=7, wave=True)
    small = flags.render(spec, size=32)
    big = flags.render(spec, size=64)
    for wrong in (lambda: sheetlib.build([small, big]),
                  lambda: sheetlib.build([small], fps=0.0),
                  lambda: sheetlib.build([small], padding=-1),
                  lambda: sheetlib.build([small, small], params=[1.0]),
                  lambda: sheetlib.build([])):
        try:
            wrong()
        except ValueError:
            pass
        else:
            raise AssertionError("a sheet that cannot be played should be refused")


# ---------------------------------------------------------- the pole
def _silhouette(canvas):
    """Rows and columns of the drawn part of a canvas, as index arrays."""
    a = np.asarray(canvas.a)
    ys, xs = np.where(a > 0.5)
    return ys, xs


def test_a_pole_is_a_spec_and_refuses_a_name_it_does_not_know():
    a = pole.spec()
    assert a == pole.spec()
    assert set(a) == {"width", "height", "over", "finial", "material"}
    assert a["finial"] in pole.FINIALS and a["material"] in pole.MATERIALS
    for wrong in (lambda: pole.spec(finial="onion"),
                  lambda: pole.spec(material="cheese"),
                  lambda: pole.spec(width=0.0),
                  lambda: pole.spec(height=-1.0)):
        try:
            wrong()
        except ValueError:
            pass
        else:
            raise AssertionError("a pole that cannot be drawn should be refused")


def test_an_anchored_cloth_is_exactly_still_at_the_hoist():
    """What a pole buys, and why it is a taper of zero rather than a small one.

    A cloth tied to a pole cannot move where it is tied, at any phase.  The claim
    is measured on the silhouette rather than trusted: the leftmost drawn column of
    a hanging flag is the hoist, and with the anchor on it does not move between two
    phases, while with the default taper it does -- so the test also shows that it
    is measuring something, which a comparison of two identical pictures would not.
    """
    def hoist(wave, seed):
        spec = flags.plain("azure", ratio=(3, 2))
        a = np.asarray(flags.render({**spec, "wave": wave}, size=160).a)
        ys, xs = np.where(a > 0.5)
        return int(xs.min()), ys[xs <= xs.min() + 1].mean()

    anchored = cloth.wave(1, amplitude=0.05, wavelength=0.25, taper=0.0)
    loose = dict(anchored, taper=0.45)
    for phase in (0.0, 1.1, 2.7):
        w = dict(anchored, phase=phase, droop=0.0)
        assert hoist(w, 0) == hoist(w, 1)
    moved = {hoist(dict(loose, phase=p, droop=0.0), 0)[0] for p in (0.0, 1.1, 2.7)}
    assert len(moved) > 1, "an unanchored hoist should wander, or this measures nothing"
    assert cloth.surface({"amplitude": 0.05, "wavelength": 0.25, "taper": 0.0,
                          "phase": 1.1}, np.array([0.0])).item() == 0.0


def test_the_droop_is_gravity_whichever_way_the_folds_run():
    """The fly trails, and it trails *down* even for horizontal folds.

    A droop cannot be measured the way a fold is: the cloth's own shape differs
    along the folds' axis, so comparing its two ends would compare two different
    parts of a flag.  Instead each end is asked how *low it reaches* in two renders
    -- one without the droop, one with it -- and the frame is pinned for both,
    because a droop that widens the margin would shrink the cloth and move every
    pixel for the wrong reason.  The anchored end must stay put and the free end
    must reach lower, in both fold directions, which is the whole point: gravity
    does not turn with the cloth.
    """
    size, margin = 192, 0.25
    style = {"margin": margin, "shadow": 0.0, "rim": 0.0, "rim_width": 0.0}
    x0, x1, y0, y1 = _cloth_box(size, margin)
    xspan = x1 - x0

    for direction in ("vertical", "horizontal"):
        spec = flags.plain("azure", ratio=(3, 2))
        base = cloth.wave(2, amplitude=0.04, wavelength=0.3, taper=0.0, direction=direction)
        reaches = []
        for droop in (0.0, 0.10):
            can = cloth.hang(blazon.render({**spec, "style": style}, size=size),
                             dict(base, droop=droop, phase=0.6))
            a = np.asarray(can.a) > 0.5
            if direction == "vertical":
                # The folds run across the cloth, so the hoist is its left edge and
                # the fly its right; the question is how low the cloth reaches over
                # each band of columns.  A *mean* would not do: the cloth can only
                # leave the bottom of the frame, never enter it, so the mean is
                # clipped there and reads as half the droop.
                reaches.append((int(np.where(a[:, x0:x0 + xspan // 8])[0].max()),
                                int(np.where(a[:, x1 - xspan // 8:x1])[0].max())))
            else:
                # The folds run up the cloth, so the hoist is its bottom edge and the
                # fly its top, and the question is how low each edge reaches: the
                # bottom edge is the anchored end, the top edge the free one.
                ys, _ = np.where(a)
                reaches.append((int(ys.max()), int(ys.min())))
        anchored, free = reaches[1][0] - reaches[0][0], reaches[1][1] - reaches[0][1]
        assert free > 3.5, (direction, free)
        assert abs(anchored) < 2.5, (direction, anchored)


def test_a_pole_gets_room_in_the_frame_and_stays_on_the_canvas():
    """A pole is not cloth, so it needs its own room -- and enough of it.

    The room is asked for in the *shape* rather than in the margin, and the claim
    is the same one the cloth makes: nothing reaches the border.  The pole is
    checked at both of its ends, because the finial above and the shaft below are
    the two places the frame could have been too short.
    """
    for ratio in ((3, 2), (1, 1), (5, 3), (2, 5)):
        for shape in ("rectangle", "swallowtail", "pennant"):
            for finial in pole.FINIALS:
                spec = flags.random_flag(seed=6, ratio=ratio, shape=shape, wave=True,
                                         pole={"finial": finial})
                can = flags.render(spec, size=128)
                a = np.asarray(can.a)
                border = max(float(a[0].max()), float(a[-1].max()),
                             float(a[:, 0].max()), float(a[:, -1].max()))
                assert border < 1e-6, (ratio, shape, finial, border)
                ys, xs = _silhouette(can)
                assert len(ys) > 0
                # The pole is left of the cloth's own box and reaches past it, above
                # and below: two thirds of the frame at the very least.
                assert xs.min() < 0.5 * a.shape[1], (ratio, shape, finial, xs.min())
                assert ys.min() < 0.30 * a.shape[0], (ratio, shape, finial, ys.min())
                assert ys.max() > 0.70 * a.shape[0], (ratio, shape, finial, ys.max())


def test_the_pole_is_the_one_thing_the_wave_does_not_move():
    """The pole is rigid, so a change of phase may not change a pixel of it.

    The cloth is waved *before* the pole is painted, and this is the test that
    says so where it is measurable: below the lowest pixel the cloth can reach --
    its own box, less the droop it is allowed -- the two renders must be identical,
    and that band must not be empty, or the comparison would prove nothing.
    """
    frame = flags.random_flag(seed=9, ratio=(3, 2), wave=True, pole={"height": 0.5})
    fn, hw, hh = blazon.resolve_shape(frame)
    style = dict(frame["style"])
    style["margin"] = float(style["margin"]) + 1.6 * cloth.reach(frame["wave"])
    size = 192
    scale = size / max(2.0 * hw, 2.0 * hh) * (1.0 - 2.0 * style["margin"])
    extent = float(size) / scale
    floor = -1.0 - float(frame["wave"]["droop"]) * extent
    row = int(size / 2.0 - floor * scale) + 2
    assert 0 < row < size, row

    a = np.asarray(flags.render({**frame, "wave": dict(frame["wave"], phase=0.3)},
                                size=size).to_uint8())
    b = np.asarray(flags.render({**frame, "wave": dict(frame["wave"], phase=2.4)},
                                size=size).to_uint8())
    assert not np.array_equal(a[:row, :], b[:row, :]), "the wave should move the cloth"
    band_a, band_b = a[row:, :], b[row:, :]
    assert np.array_equal(band_a, band_b)
    assert int((band_a[..., 3] > 200).sum()) > 25, "the pole should be down there to compare"


def _shaft_profile(light):
    """The shaft's luminance across its width, on a blank canvas, at mid-height.

    Taken on a bare pole rather than on a flag, because the thing being measured
    is the pole's own surface: only the pixels where the shaft actually is are
    kept, in left-to-right order.
    """
    p = pole.spec(width=0.18, finial="none")
    fn, hw, hh = shields.flag((3, 2), "rectangle", pole.room(p))
    can = paint.field_canvas(256, hw, hh, margin=0.05, background=(0.90, 0.90, 0.90))
    pole.paint(can, p, hw, 1.0, light=light)
    row = can.h // 2
    lum = np.asarray(can.to_uint8())[..., :3].astype(np.float64).mean(axis=2)[row]
    d = pole.field(p, hw, 1.0)(*can.grid())[row]
    # Only the pixels the shaft *covers*: a pixel half on the edge is half
    # background, and a profile that ended on one would be measuring the sheet.
    return lum[d < -can.dx]


def test_the_pole_is_shaded_as_a_cylinder_not_as_a_bevel():
    """A cylinder is brightest where it faces the light, not at its own rim.

    That is the whole difference between the shaft's shading and the house bevel:
    a bevel's height saturates in the middle and tilts at the rim, so it is
    brightest at the edge, while a cylinder's normal turns smoothly across the
    width and peaks somewhere inside it.  The rim light is switched off for the
    measurement, because it darkens exactly the pixels a bevel would have made
    brightest and the question is about the surface rather than the outline.  The
    light is then handed to the painter mirrored and the highlight has to follow
    it, which is what says the shading came from the normals at all.
    """
    strength = pole.RIM_STRENGTH
    pole.RIM_STRENGTH = 0.0
    try:
        left = _shaft_profile(pole.LIGHT)
        right = _shaft_profile((0.45, 0.65))
    finally:
        pole.RIM_STRENGTH = strength
    assert left.shape[0] >= 7, left.shape
    peak = int(np.argmax(left))
    assert 0 < peak < left.shape[0] - 1, (peak, left)
    assert float(left[0]) < float(left[peak]) and float(left[-1]) < float(left[peak])
    assert int(np.argmax(right)) > peak, (peak, np.argmax(right))


def test_a_pole_may_not_hold_up_a_shield():
    spec = {**flags.plain("azure", ratio=(3, 2)), "pole": pole.spec(), "shield": "heater"}
    try:
        flags.render(spec, size=64)
    except ValueError:
        pass
    else:
        raise AssertionError("a pole holds up a flag, not a shield")


def test_a_poled_flag_survives_a_round_trip():
    import json

    spec = flags.random_flag(seed=8, ratio=(3, 2), wave=True, pole=True)
    again = json.loads(json.dumps(spec))
    assert again["pole"] == spec["pole"] and again["wave"] == spec["wave"]
    assert flags.render(again, size=96).to_uint8().tobytes() == \
        flags.render(spec, size=96).to_uint8().tobytes()


def test_a_faction_banner_can_be_flown_on_a_pole():
    name = "Kingdom of Alder"
    assert "pole" not in crest.flag(name), "the pole is opt-in, so the default stands"
    flown = crest.flag(name, pole=True)
    assert flown["pole"]["material"] in pole.MATERIALS
    assert flown["field"] == crest.flag(name)["field"]
    # A banner on a pole is cloth, and the pole is what anchors it: the hoist is
    # still, and the wind comes from the name rather than from the call.
    assert flown["wave"]["taper"] == 0.0
    assert flown["wave"]["droop"] > 0.0
    assert crest.flag(name, pole=True)["wave"] == flown["wave"], "the wind is the name's"
    table = crest.shade_table(name)
    plain = np.asarray(flags.render(crest.flag(name), size=128, shade=table).a)
    poled = np.asarray(flags.render(flown, size=128, shade=table).a)
    # The pole reaches outside the cloth's own box on three sides -- the shaft to the
    # left of the hoist, the finial above, the foot below -- and the room is what the
    # frame paid for that, because the cloth keeps its coordinates either way.
    margin = flags.FLAG_STYLE["margin"] + 1.6 * cloth.reach(flown["wave"])
    x0, x1, y0, y1 = _cloth_box(128, margin)
    drawn = np.where(poled > 0.5)
    assert drawn[1].min() < x0, "the shaft is to the left of the hoist"
    assert drawn[0].min() < y0, "the finial is above the cloth"
    assert drawn[0].max() > y1, "and the foot is below it"
    assert np.where(plain > 0.5)[1].min() >= _cloth_box(128, flags.FLAG_STYLE["margin"])[0], \
        "while the flat banner is inside its own frame"
    # The drift-proof entry point flies the same banner, pole and wind included.
    one = crest.render(name, size=64, which="flag", pole=True).to_uint8().tobytes()
    two = flags.render(flown, size=64, shade=table).to_uint8().tobytes()
    assert one == two, "crest.render and crest.flag disagree about the banner"


# ---------------------------------------------------------- the bar
def _fold_contrast(canvas, fractions=(0.05, 0.5, 0.95)):
    """How much the luminance varies across the cloth at each of these heights.

    The columns are inset by a fifth at each side, because the cloth's own rim is a
    still, uniform darkening that would otherwise be most of the number being read --
    and a still darkening is exactly what must not be mistaken for a fold.
    """
    lum = _lum(canvas)
    rows = np.where(np.isfinite(lum).any(axis=1))[0]
    cols = np.where(np.isfinite(lum).any(axis=0))[0]
    pad = (cols[-1] - cols[0]) // 5
    inset = slice(cols[0] + pad, cols[-1] - pad)
    return [float(np.nanstd(lum[rows[int(f * (len(rows) - 1))], inset])) for f in fractions]


def test_a_bar_is_a_spec_and_refuses_a_name_it_does_not_know():
    a = pole.bar()
    assert a == pole.bar()
    assert set(a) == {"width", "reach", "over", "material"}
    assert a["material"] in pole.MATERIALS
    assert "finial" not in a, "a bar's ends are turned towards the shoulders"
    for wrong in (lambda: pole.bar(material="cheese"),
                  lambda: pole.bar(width=0.0),
                  lambda: pole.bar(reach=-0.1),
                  lambda: pole.bar(over=-0.1)):
        try:
            wrong()
        except ValueError:
            pass
        else:
            raise AssertionError("a bar that cannot be drawn should be refused")


def test_a_gonfalon_hangs_from_its_bar_and_the_folds_deepen_downwards():
    """The whole of the second hanging, in one number read at three heights.

    A flag on a pole has folds that are the *same* at every height: what decides how
    deep they are is how far across the cloth the point is, and a row near the top of
    the cloth and a row near the bottom are the same row.  A gonfalon's folds are not:
    the bar holds the top of them and they deepen downwards from it, so the luminance
    across a row -- the measure of a fold -- starts small and grows.  The contrast of
    the pole's flag is constant to a couple of percent and the gonfalon's nearly
    doubles; both are measured on the same wind and the same plain field, so the only
    thing between them is which edge is holding the cloth.
    """
    base = flags.render(flags.plain("azure", ratio=(3, 2)), size=192)
    wave = cloth.wave(2, amplitude=0.05, wavelength=0.25, harmonic=0.0)
    box = (-1.5, 1.5)
    hung = _fold_contrast(cloth.hang(base.clone(), wave, box=box))
    draped = _fold_contrast(cloth.drape(base.clone(), wave, box=box, drop=(1.0, -1.0)))
    assert max(hung) / min(hung) < 1.02, hung
    assert draped[0] < draped[1] < draped[-1], draped
    assert draped[-1] > 1.5 * draped[0], draped


def test_an_anchored_gonfalon_is_still_at_the_bar_and_free_at_the_hem():
    """A cloth cannot move where it is held, and a bar holds the whole top edge.

    Two phases of one wind, and the difference between them read as a function of
    height.  The measured claim is comparative rather than absolute -- the same wind
    left loose moves the top of the cloth several times as much -- and the exact claim
    is the algebra underneath it: at the bar the envelope, the height and both
    displacements are all zero, not merely small.  The bar's own pixels are not part
    of this; the bar is drawn over the cloth and never moves at all, which
    :func:`test_the_bar_is_the_one_thing_the_wave_does_not_move` is about.
    """
    base = flags.render(flags.plain("azure", ratio=(3, 2)), size=192)
    wave = cloth.wave(2, amplitude=0.05, wavelength=0.25, harmonic=0.0)
    box = (-1.5, 1.5)

    def top_and_hem(spec):
        one = cloth.drape(base.clone(), spec, box=box, drop=(1.0, -1.0))
        two = cloth.drape(base.clone(), {**spec, "phase": spec["phase"] + 1.2},
                          box=box, drop=(1.0, -1.0))
        diff = np.abs(_lum(one) - _lum(two))
        idx = np.where(np.isfinite(diff).any(axis=1))[0]
        return float(np.nanmax(diff[idx[0]])), float(np.nanmean(diff[idx[-2]]))

    loose_top, _ = top_and_hem(wave)
    held_top, held_hem = top_and_hem(pole.anchor(wave))
    assert held_top < 0.25 * loose_top, (held_top, loose_top)
    assert held_hem > 5.0, held_hem
    assert float(cloth.freedom(pole.anchor(wave), 0.0)) == 0.0
    assert float(cloth.surface(pole.anchor(wave), 0.3, 3.0, envelope=0.0)) == 0.0
    along, down = cloth.drift(pole.anchor(wave), 0.3, 3.0, envelope=0.0, drop=2.0)
    assert float(along) == 0.0 and float(down) == 0.0
    # And the bar is not asking for more than the cloth's own height: the drop it is
    # given is the cloth's, because the cloth keeps its coordinates however the frame
    # grew around it.
    assert float(cloth.surface(wave, 0.5, 3.0, envelope=0.5)) > 0.0


def test_a_bar_gets_room_in_the_frame_and_never_touches_it():
    """The bar is the widest thing on the canvas, and the margin already holds it.

    A bar reaches past both sides of the cloth and above its top edge.  A pole asks
    the *shape* for room because a pole is tall; a bar does not have to, and the
    reason is worth measuring rather than asserting: the frame is scaled by the wider
    of the cloth's two dimensions and drawn with a margin of several percent of it, so
    a bar that reaches a few hundredths of a unit past the cloth is already inside.
    The flat gonfalon is measured first, where the bar is the only thing that can be
    outside the cloth's own box, and then the border at every proportion and shape.
    """
    spec = {**flags.plain("azure", ratio=(3, 2)), "bar": pole.bar()}
    a = np.asarray(flags.render(spec, size=192).a)
    x0, x1, y0, y1 = _cloth_box(192, flags.FLAG_STYLE["margin"])
    ys, xs = np.where(a > 0.5)
    assert ys.min() < y0, "the rod is above the cloth's top edge"
    assert xs.min() < x0 and xs.max() > x1, "and past both of its sides"
    assert float(a.max()) > 0.9
    for ratio in ((3, 2), (1, 1), (5, 3), (2, 3), (3, 5)):
        for shape in ("rectangle", "swallowtail", "pennant", "guidon"):
            for size in (64, 160):
                spec = flags.random_flag(seed=4, ratio=ratio, shape=shape, wave=True, bar=True)
                a = flags.render(spec, size=size).a
                border = max(float(a[0].max()), float(a[-1].max()),
                             float(a[:, 0].max()), float(a[:, -1].max()))
                assert border < 1e-6, (ratio, shape, size, border)
                assert float(a.max()) > 0.9


def test_the_bar_is_the_one_thing_the_wave_does_not_move():
    """A bar is rigid and drawn last: the cloth moves under it and it does not.

    Measured on the band above the cloth's own box, where only the bar is drawn --
    two phases of the wind, and the pixels there have to be identical, not close.  The
    cloth's pixels below are not, which is the other half of the same statement and is
    why the band has to be taken outside the cloth rather than on it.
    """
    spec = flags.random_flag(seed=6, ratio=(3, 2), wave=True, bar=True)
    one = flags.render(spec, size=256)
    two = flags.render({**spec, "wave": {**spec["wave"], "phase": spec["wave"]["phase"] + 1.0}}, size=256)
    a1, a2 = np.asarray(one.a), np.asarray(two.a)
    margin = flags.FLAG_STYLE["margin"] + 1.6 * cloth.reach(spec["wave"])
    y0 = _cloth_box(256, margin)[2]
    band = np.arange(0, max(y0, 1))
    assert (a1[band] > 0.5).any(), "the bar is above the cloth's box"
    assert np.array_equal(a1[band], a2[band]), "the bar moved with the wind"
    assert not np.array_equal(a1[y0 + 4:], a2[y0 + 4:]), "the cloth did not"


def test_a_bar_may_not_hold_up_a_shield_but_may_share_a_frame_with_a_pole():
    spec = {**flags.plain("azure", ratio=(3, 2)), "bar": pole.bar(), "shield": "heater"}
    try:
        flags.render(spec, size=64)
    except ValueError:
        pass
    else:
        raise AssertionError("a bar hangs a cloth, not a shield")
    # The classic gonfalon is a staff with a crossbar on it: one spec with both, and
    # the frame takes the larger of the two demands rather than the sum.
    staff = {**flags.plain("azure", ratio=(3, 2)), "pole": pole.spec(), "bar": pole.bar(),
             "wave": pole.anchor(cloth.wave(1))}
    a = np.asarray(flags.render(staff, size=160).a)
    assert float(a.max()) > 0.9 and float(a[0].max()) < 1e-6
    margin = flags.FLAG_STYLE["margin"] + 1.6 * cloth.reach(staff["wave"])
    # The *cloth's* top edge, not the frame's: the frame is taller than the cloth by
    # exactly the room the staff asked for, and the staff fills it.
    y0 = _cloth_box(160, margin)[2]
    assert np.where(a > 0.5)[0].min() < y0, "the staff rises above the cloth on the bar"


def test_a_gonfalon_survives_a_round_trip():
    import json

    spec = flags.random_flag(seed=8, ratio=(3, 2), wave=True, bar=True)
    again = json.loads(json.dumps(spec))
    assert again["bar"] == spec["bar"] and again["wave"] == spec["wave"]
    assert flags.render(again, size=96).to_uint8().tobytes() == \
        flags.render(spec, size=96).to_uint8().tobytes()


def test_a_faction_banner_can_be_hung_from_a_bar():
    name = "Kingdom of Alder"
    assert "bar" not in crest.flag(name), "the bar is opt-in, so the default stands"
    hung = crest.flag(name, bar=True)
    assert hung["bar"]["material"] in pole.MATERIALS
    assert hung["field"] == crest.flag(name)["field"]
    assert hung["wave"]["taper"] == 0.0
    assert hung["wave"]["droop"] > 0.0
    assert crest.flag(name, bar=True)["wave"] == hung["wave"], "the wind is the name's"
    # A bar and a pole are two mountings of one cloth, so the faction's wind -- which
    # is the name's -- is the same either way, and only the spec key differs.
    assert crest.flag(name, pole=True)["wave"] == hung["wave"]
    table = crest.shade_table(name)
    one = crest.render(name, size=64, which="flag", bar=True).to_uint8().tobytes()
    two = flags.render(hung, size=64, shade=table).to_uint8().tobytes()
    assert one == two, "crest.render and crest.flag disagree about the banner"


# --------------------------------------------------------------- end to end
def test_rendering_is_deterministic_and_byte_identical():
    with tempfile.TemporaryDirectory() as tmp:
        spec = blazon.random_arms(seed=3)
        fspec = flags.random_flag(seed=3)
        paths = []
        for i, (kind, s) in enumerate((("arms", spec), ("flags", fspec))):
            for j in (0, 1):
                p = Path(tmp) / f"{kind}{j}.png"
                (blazon.render if kind == "arms" else flags.render)(s, size=96).save(p)
                paths.append(p)
        h = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        assert h[0] == h[1], "arms render is not byte-identical"
        assert h[2] == h[3], "flags render is not byte-identical"
        assert all(p.stat().st_size > 0 for p in paths)


# ------------------------------------------------------------------- factions
def _tinctures_used(spec) -> set:
    """Every tincture a spec actually wears."""
    used = set(spec["field"]["tinctures"])
    if spec.get("ordinary"):
        used.add(spec["ordinary"]["tincture"])
        if spec["ordinary"].get("fimbriation"):
            used.add(spec["ordinary"]["fimbriation"])
    used.update(c["tincture"] for c in spec.get("charges", []) if c.get("tincture"))
    return used


def test_the_name_is_the_whole_input():
    a = crest.faction("Kingdom of Alder")
    b = crest.faction("Kingdom of Alder")
    assert a == b
    assert a["seed"] == crest.seed_from("Kingdom of Alder")
    assert a["key"] == "kingdom of alder"
    assert a["blazon"]["en"] and a["blazon"]["ru"] and a["description"]["en"]


def test_a_fresh_process_with_a_different_hash_seed_agrees():
    """The trap this module is really about: Python salts string hashing per process.

    Seeded with ``hash(name)`` a faction would be given a different banner every
    time the game restarted, and nothing in a single process could notice -- the two
    derivations only have to disagree across a restart.  So the check runs the
    derivation somewhere else, with a different hash seed, and compares the words.
    """
    import os
    import subprocess

    root = str(Path(__file__).resolve().parents[1])
    code = (f"import sys; sys.path.insert(0, {root!r});"
            "from gen.heraldry import crest;"
            "f = crest.faction('Kingdom of Alder');"
            "print(f['seed'], '|', f['blazon']['en'])")
    seen = set()
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, timeout=120)
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1, seen
    assert next(iter(seen)).startswith(str(crest.seed_from("Kingdom of Alder")))


def test_case_and_spacing_do_not_make_a_new_faction():
    a = crest.faction("Kingdom of Alder")
    for spelling in ("  kingdom   of  alder ", "KINGDOM OF ALDER", "\tKingdom of Alder\n"):
        other = crest.faction(spelling)
        assert other["seed"] == a["seed"]
        assert other["arms"] == a["arms"] and other["flag"] == a["flag"]
    assert crest.normalise("  The   Mirefen\nLeague ") == "the mirefen league"
    # The spelling is kept for display even though the key is not.
    assert crest.faction("  Kingdom of Alder  ")["name"] == "Kingdom of Alder"


def test_different_names_are_different_factions():
    names = ["Alder", "Brythmark", "Halven", "Craymoor", "Delvane", "Ostmark"]
    assert len({crest.seed_from(n) for n in names}) == len(names)
    blazons = {crest.faction(n)["blazon"]["en"] for n in names}
    assert len(blazons) >= len(names) - 1      # a blazon collision is possible, not likely


def test_an_empty_name_is_a_bug_rather_than_a_plain_crest():
    for empty in ("", "   ", "\t\n"):
        try:
            crest.seed_from(empty)
        except ValueError:
            pass
        else:
            raise AssertionError("a nameless faction should be refused")
    try:
        crest.seed_from(7)
    except TypeError:
        pass
    else:
        raise AssertionError("a name is a string")


def test_the_salt_separates_the_streams():
    assert crest.seed_from("Alder", "arms") != crest.seed_from("Alder", "flag")
    assert crest.seed_from("Alder") != crest.seed_from("Alder", "arms")
    assert crest.seed_from("Alder") == crest.seed_from("Alder", "")


def test_the_banner_flies_what_the_shield_bears():
    """The claim that keeps a faction from having two identities.

    The banner is derived from the *coat*, not from the name a second time, so the
    two cannot drift apart -- and the tinctures they share are what makes that
    checkable.  Every faction in the demo set is asked, not just one, because the
    branches (an ordinary to fly, charges but no ordinary, neither) are all real.
    """
    flown = 0
    for name in ("Kingdom of Alder", "Brythmark", "Silverreach Compact", "House Delvane",
                 "Duchy of Halven", "The Mirefen League"):
        f = crest.faction(name)
        arms, banner = f["arms"], f["flag"]
        assert banner["field"]["tinctures"][0] == arms["field"]["tinctures"][0], name
        arms_used = _tinctures_used(arms)
        banner_used = _tinctures_used(banner)
        assert banner_used <= arms_used, (name, banner_used - arms_used)
        ordinary = arms.get("ordinary")
        if ordinary and ordinary["name"] in crest._FLYING:
            flown += 1
            assert ordinary["tincture"] in banner_used, name
    assert flown > 0, "the demo set should include at least one ordinary that flies"


def test_every_faction_renders_and_is_json():
    import json

    for name in ("Kingdom of Alder", "Brythmark", "House Delvane"):
        f = crest.faction(name)
        text = json.dumps(f, ensure_ascii=False)
        assert len(text) > 200
        assert not f["warnings"], (name, f["warnings"])
        for spec in (f["arms"], f["flag"]):
            canvas = blazon.render(spec, size=48)
            assert float(canvas.a.max()) > 0.5      # something was actually drawn


# ------------------------------------------------------------------ livery
def _shade_tables(count: int = 40):
    """The livery of ``count`` imaginary factions, as plain integer tables."""
    return [crest.shade(f"Faction {i}") for i in range(count)]


def test_a_shade_keeps_every_tincture_in_its_own_class():
    """A livery is a pigment, not a re-classification.

    The rule of tincture is a claim about *classes*, so the one thing a shade is
    not allowed to do is move a tincture across one: Or is a metal before the shade
    and a metal after it.  This is what lets the whole heraldic guarantee survive
    the faction colours -- ``validate`` only ever sees names, so if a shade could
    re-class a tincture the guarantee would be about a different picture than the
    one that gets drawn.
    """
    for name in ("Kingdom of Alder", "Brythmark", "House Delvane", "Faction 7"):
        table = crest.shade_table(name)
        assert set(table) <= set(tinctures.CLASSICAL)
        for key, t in table.items():
            assert t.kind == tinctures.get(key).kind, key
        for a, b in itertools.combinations(tinctures.CLASSICAL, 2):
            assert tinctures.check(a, b)[0] == tinctures.check(table[a], table[b])[0], (a, b)


def test_a_shade_is_a_colour_and_not_a_layout():
    """The UI package's rule, applied to a shield: a state is a colour, never a shape.

    The two renders have to agree pixel for pixel on *coverage* -- an alpha channel
    that moved would mean a shade had changed a coat's geometry, which would quietly
    make this a second composition pass rather than a pigment.
    """
    f = crest.faction("Kingdom of Alder")
    table = crest.shade_table("Kingdom of Alder")
    plain = blazon.render(f["arms"], size=128)
    shaded = blazon.render(f["arms"], size=128, shade=table)
    assert np.array_equal(plain.a, shaded.a)
    assert not np.array_equal(plain.rgb, shaded.rgb)
    assert float(np.abs(plain.rgb - shaded.rgb).max()) > 0.04


def test_a_shade_step_is_a_step():
    """Every step a sidecar records has to be a difference the picture shows.

    The table is integers precisely so a caller can trust it, which means a step
    that buys no colour is worse than no step: it makes "two shades apart" untrue.
    So a direction with no room is closed when the shade is *drawn* (see
    :func:`gen.heraldry.tinctures._room`), and this checks the other end of that
    promise -- a step that survived is worth at least what the module calls a step,
    and a table of zeros is exactly the classical tincture.
    """
    for table in _shade_tables(60):
        shaded = tinctures.shade_table(table)
        for key, steps in table.items():
            moved = palette.delta_e(shaded[key].rgb, tinctures.get(key).rgb)
            if any(steps.values()):
                assert moved >= tinctures._MIN_STEP, (key, steps, moved)
            else:
                assert moved < 1e-9, (key, steps, moved)


def test_two_factions_that_chose_the_same_tincture_are_told_apart():
    """Seven tinctures, more factions than that, and a map to read.

    Two houses can both pick Azure, and on the map they still have to be two
    colours.  The measurement is over forty names and every classical tincture: the
    shades have to be *nearly* all different, a step apart when they are, and close
    enough to the tincture they shade that Azure is still the blue a player calls
    Azure.

    The floor is per tincture, because the room is: a tincture is told apart by its
    hue and its chroma, and Argent has 0.0025 of one and 0.003 of the other.  A
    shade of a white is a lightness and a tint, and there are fewer of those than
    there are hues -- which the test says out loud rather than hiding behind a bound
    low enough for the worst case.
    """
    tables = _shade_tables(40)
    for key in tinctures.CLASSICAL:
        colours = [tinctures.shade_table(t)[key].rgb for t in tables]
        distinct = {tuple(np.round(c, 4)) for c in colours}
        floor = 36 if tinctures.get(key).lch[1] >= 0.05 else 12
        assert len(distinct) >= floor, (key, len(distinct))
        apart = [palette.delta_e(a, b) for a, b in itertools.combinations(
            sorted(distinct), 2)]
        assert min(apart) >= 0.003, (key, min(apart))
        assert max(palette.delta_e(tinctures.get(key).rgb, c) for c in colours) <= 0.25


def test_a_shade_never_collides_with_the_tincture_beside_it():
    """The cost of a livery, measured: shading can bring two tinctures closer.

    Azure and Purpure are the closest pair in the classical seven, so they are the
    pair a shade has the most power to ruin.  A floor is checked rather than
    assumed -- the two may come nearer, they may not meet.
    """
    worst, pair = 9.9, None
    for table in _shade_tables(60):
        shaded = tinctures.shade_table(table)
        for a, b in itertools.combinations(tinctures.CLASSICAL, 2):
            d = palette.delta_e(shaded[a].rgb, shaded[b].rgb)
            if d < worst:
                worst, pair = d, (a, b)
    assert worst >= 0.05, (worst, pair)
    # and the classical separation of that pair is the headroom it came out of
    assert tinctures.contrast(*pair) >= worst


def test_a_fur_is_shaded_through_its_bearings():
    """Ermine is Argent and Sable in a pattern, so shading *those* shades it.

    A fur is not a colour, and a table that tried to shade "ermine" itself would be
    shading a pattern.  Its bearings are ordinary tinctures, though, and they are in
    every livery -- so two factions whose arms are both Ermine come out two
    different greys, which is the whole point of the exercise.  Rendered, not
    asserted from the spec, because the fur is painted as its own two-tincture
    pattern and the two ways of getting that wrong look identical on paper.
    """
    coat = {"field": {"kind": "plain", "tinctures": ["ermine"]},
            "shield": "heater", "style": {"margin": 0.10}}
    for key in tinctures.CLASSICAL:
        assert tinctures.get(key).kind != tinctures.FUR
    table = tinctures.shade_table({"argent": {"l": -6, "c": 0, "h": 0},
                                   "sable": {"l": 4, "c": 3, "h": 0}})
    assert "ermine" not in table                      # the fur itself is skipped
    plain = blazon.render(coat, size=96)
    shaded = blazon.render(coat, size=96, shade=table)
    assert np.array_equal(plain.a, shaded.a)
    assert float(np.abs(plain.rgb - shaded.rgb).max()) > 0.02


def test_a_livery_is_derived_from_the_name_like_everything_else():
    """One input, one output, and no way to hold the two halves differently.

    ``faction`` carries the livery as integers so a sidecar can be *read* -- a game
    designer has to be able to see why a shield came out that colour -- and the
    derivation is the same one the coat came from, so the two cannot drift.
    """
    import json

    name = "Kingdom of Alder"
    shade = crest.shade(name)
    assert shade == tinctures.random_shade(crest.seed_from(name, "shade"))
    assert shade == crest.faction(name)["shade"]
    assert json.loads(json.dumps(shade)) == shade
    for steps in shade.values():
        assert set(steps) == set(tinctures.SHADE_AXES)
        assert all(isinstance(v, int) and not isinstance(v, bool) for v in steps.values())
    # a fresh process must draw the same livery, for the same reason the coat does
    import os
    import subprocess

    root = str(Path(__file__).resolve().parents[1])
    code = (f"import sys, json; sys.path.insert(0, {root!r});"
            "from gen.heraldry import crest;"
            "print(json.dumps(crest.shade('Kingdom of Alder'), sort_keys=True))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONHASHSEED=seed), timeout=120).stdout
            for seed in ("0", "99")}
    assert len(outs) == 1
    assert json.loads(outs.pop()) == shade


def test_rendering_without_a_shade_is_untouched():
    """The classical palette is the default, and ``{}`` is not a special case.

    Everything that renders a coat -- the contact sheets, the blazon CLI, a game
    that has no factions at all -- must get exactly what it got before there was
    such a thing as a livery, byte for byte.
    """
    for name in ("Kingdom of Alder", "Brythmark", "House Delvane"):
        f = crest.faction(name)
        for spec, render in ((f["arms"], blazon.render), (f["flag"], flags.render)):
            plain = render(spec, size=64)
            assert np.array_equal(plain.rgb, render(spec, size=64, shade=None).rgb)
            assert np.array_equal(plain.rgb, render(spec, size=64, shade={}).rgb)


def test_one_call_renders_a_faction_in_its_own_livery():
    """``crest.render`` exists so that the pigments cannot be left behind.

    A faction is a name, and the caller who forgets to pass the shade gets a
    different picture from the caller who remembers -- which is precisely the kind
    of drift every other part of this package derives away.  So there is one call
    that takes the name and returns the shield, and it refuses a word it does not
    understand rather than falling back to a default that looks plausible.
    """
    name = "The Mirefen League"
    table = crest.shade_table(name)
    assert np.array_equal(crest.render(name, size=96, which="arms").rgb,
                          blazon.render(crest.arms(name), size=96, shade=table).rgb)
    assert np.array_equal(crest.render(name, size=96, which="flag").rgb,
                          flags.render(crest.flag(name), size=96, shade=table).rgb)
    assert not np.array_equal(crest.render(name, size=96).rgb,
                              blazon.render(crest.arms(name), size=96).rgb)
    try:
        crest.render(name, which="banner")
    except ValueError:
        pass
    else:
        raise AssertionError("a faction is rendered as arms or a flag, not a banner")


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
