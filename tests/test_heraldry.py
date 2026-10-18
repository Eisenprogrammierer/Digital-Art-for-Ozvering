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
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen.heraldry import blazon, charges, crest, field, flags, ordinaries, shields, tinctures  # noqa: E402


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
