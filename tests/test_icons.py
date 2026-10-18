"""Regression tests for the icon set and the atlas a set ships in.

An icon is 32 px of meaning, and the two ways it fails are both invisible in a
source diff: a symbol that stops reading at a small size, and a set that stops
looking like a set.  Both are measured here rather than eyeballed -- every charge is
compared to its canonical self, every plate to the round one, and the atlas to the
pixels that went into it.

The atlas gets the strongest test in the file.  A sprite in an atlas is *moved*, not
re-rendered, so every rectangle is compared byte for byte with the source it came
from: a packing bug that softened or shifted a sprite could not survive that.

Run either way::

    python tests/test_icons.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export import atlas as atlaslib  # noqa: E402
from gen.icons import badge, livery, motifs, paint  # noqa: E402


# ------------------------------------------------------------------- helpers
def _fields(name, spec=None, n: int = 160):
    """The region a charge covers, as a boolean mask on a fixed design grid."""
    axis = np.linspace(-0.75, 0.75, n)
    X, Y = np.meshgrid(axis, axis)
    return motifs.field(spec if spec is not None else motifs.canonical(name), X, Y) < 0


def _iou(a, b) -> float:
    return float((a & b).sum()) / float((a | b).sum())


def _components(mask) -> int:
    """How many separate blobs a boolean grid holds (4-neighbour flood fill).

    The one measurement a *pip* earns: a tier is not "n dots' worth of area", it is
    n dots, and the only way to be sure the layout did not quietly merge two of
    them is to count the blobs.
    """
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
    return count


def _badge_marks(spec, margin: float = 0.75, n: int = 240):
    """A badge's mark field (numeral or pips) on a fixed grid covering the tile."""
    axis = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(axis, axis)
    return badge.marks(spec, X, Y, margin) < 0


def _members(count: int = 6, size: int = 64):
    names = list(motifs.KINDS)[:count]
    return [(name, paint.draw(motifs.canonical(name), size)) for name in names]


def _overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


# --------------------------------------------------------------- the charges
def test_the_plate_is_the_same_plate_for_the_whole_set():
    """One plate, many charges: the property that makes icons look like a set.

    Measured on rendered pixels rather than on the spec, because the claim is about
    what a player sees: every charge covers the same share of its tile to within a
    fraction of a percent, so a row of icons reads as a row and not as a ragged line
    of differently-sized buttons.
    """
    fractions = [paint.silhouette(paint.draw(motifs.canonical(name), 128))
                 for name in motifs.KINDS]
    assert max(fractions) - min(fractions) < 0.004, fractions


def test_the_four_plates_are_all_sensibly_sized():
    """A shield plate is not half the size of a square one.

    The plates are different shapes on purpose, so their areas cannot be equal -- a
    hexagon that filled its box would be a square.  What they must be is comparable:
    between about three fifths and a little over the round plate, so that choosing a
    plate changes the *look* of an icon and not its weight on a screen.
    """
    axis = np.linspace(-1.2, 1.2, 240)
    X, Y = np.meshgrid(axis, axis)
    areas = {}
    for plate in motifs.PLATES:
        spec = paint.clamp({**motifs.canonical("coin"), "plate": plate})
        areas[plate] = float((motifs.plate_field(plate, X, Y, spec) < 0).mean())
    for plate, value in areas.items():
        assert 0.55 <= value / areas["round"] <= 1.10, (plate, value / areas["round"])


def test_a_seed_stretches_the_symbol_and_never_replaces_it():
    """A seed moves the proportions; the symbol is the symbol.

    Measured as overlap with the canonical charge over a run of seeds.  The worst
    case is a gem whose seed switched it from the six-sided brilliant to the
    three-faced crystal, which is a category rather than a wobble -- so the floor
    here is a floor on *recognition*, not on slavishness.  Underneath it a seed
    would have started drawing a different icon, which is what a random icon
    generator does and what this one is built not to do.
    """
    for name in motifs.KINDS:
        canon = _fields(name)
        worst = min(_iou(canon, _fields(name, motifs.random_spec(seed, name)))
                    for seed in range(10))
        assert worst > 0.40, (name, worst)


def test_no_two_charges_are_the_same_shape():
    """The vocabulary is small; small is not the same as redundant.

    The closest pair is water and a flame -- a drop and a two-tongued flame agree on
    about seven tenths of their area -- and they are six hues apart, which is the
    other half of the answer.  Above the bound here, two charges would be one charge
    with two names, and the set would be smaller than it claims to be.
    """
    names = list(motifs.KINDS)
    fields = {name: _fields(name) for name in names}
    worst, pair = 0.0, None
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            value = _iou(fields[a], fields[b])
            if value > worst:
                worst, pair = value, (a, b)
    assert worst < 0.80, (worst, pair)


def test_an_unknown_charge_is_reported_and_refused():
    """A bad name is an identifier error, so it is reported and then refused.

    ``clamp`` repairs numbers and will not invent an identifier, which is the same
    line `gen.ui.material` draws: an icon that quietly became a coin because
    somebody typed ``marble`` is worse than one that stopped.
    """
    spec = {**motifs.canonical("coin"), "kind": "marble"}
    assert paint.validate(spec), "validate accepted an unknown charge"
    try:
        paint.draw(spec, 64)
    except ValueError:
        return
    raise AssertionError("an unknown charge was drawn")


# ---------------------------------------------------------------- the drawing
def test_an_icon_never_touches_its_own_edge():
    """The tile is bigger than the icon, at every size.

    A plate that reached its own edge would have its rim and its shadow cut off, and
    in an atlas it would bleed into its neighbour.  The margin is in world units, so
    it holds at 16 px as well as at 256 -- worth checking, because at 16 px one
    world unit of margin is a single pixel, and a *pixel* margin would have failed
    here.
    """
    for size in (16, 24, 32, 64, 256):
        for name in motifs.KINDS:
            a = paint.draw(motifs.canonical(name), size).a
            edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
            assert edge.max() < 0.1, (name, size, float(edge.max()))


def test_an_icon_reads_at_thirty_two_pixels():
    """The failure mode that matters: an icon that has become a blob.

    Two numbers, both taken from the 32 px render.  The plate still covers what it
    covers at full size, so the symbol has not collapsed inward; and there is still
    a spread of light inside the silhouette, so the charge has not merged into its
    own plate.  Neither of these says the icon is *good*; both say it is still
    saying something.
    """
    big = {name: paint.silhouette(paint.draw(motifs.canonical(name), 192))
           for name in motifs.KINDS}
    for name in motifs.KINDS:
        small = paint.draw(motifs.canonical(name), 32)
        assert abs(paint.silhouette(small) - big[name]) < 0.06, name
        assert paint.contrast(small) > 0.05, (name, paint.contrast(small))


def test_an_icon_is_the_same_drawing_at_any_size():
    """Designing in world units, tested from the outside.

    The 256 px render folded back down has to *be* the 128 px render: the same
    plate, the same charge, the same bevel at twice the resolution.  A recipe that
    worked in pixel coordinates would come out twice as coarse, and this is the only
    place that would show.
    """
    for name in ("coin", "blade", "wood", "shield", "ore"):
        one = paint.draw(motifs.canonical(name), 128).to_uint8()
        two = paint.draw(motifs.canonical(name), 256).to_uint8()
        folded = two.astype(np.float64).reshape(128, 2, 128, 2, 4).mean(axis=(1, 3))
        d = np.abs(folded[..., :3] - one[..., :3])
        assert d.mean() < 4.0, (name, d.mean())
        assert np.percentile(d, 99) < 45.0, (name, np.percentile(d, 99))


def test_rendering_is_deterministic_and_byte_identical():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for _ in range(2):
            for name in ("coin", "blade"):
                path = Path(tmp) / f"{name}{len(paths)}.png"
                paint.draw(motifs.canonical(name), 96).save(path)
                paths.append(path)
        h = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        assert h[0] == h[2] and h[1] == h[3], "an icon render is not byte-identical"
        assert len(set(h)) == 2, "two different charges rendered the same"
        assert all(p.stat().st_size > 0 for p in paths)


def test_rendering_is_deterministic_across_processes():
    """No salted hash and no lucky float order: two interpreters, one answer."""
    code = (
        "import sys, hashlib;"
        "sys.path.insert(0, %r);"
        "from gen.icons import paint;"
        "print(hashlib.sha256(paint.icon(21, 128).to_uint8().tobytes()).hexdigest())"
    )
    root = str(Path(__file__).resolve().parents[1])
    seen = set()
    for salt in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": salt}
        run = subprocess.run([sys.executable, "-c", code % root], capture_output=True,
                             text=True, env=env, check=True)
        seen.add(run.stdout.strip())
    assert len(seen) == 1, seen


# ------------------------------------------------------------------- the badge
def test_a_badge_is_a_count_or_a_tier_and_never_both():
    """The two questions a charge cannot answer, and one badge for either.

    A count and a tier are different questions -- "how many" and "how good" -- so a
    badge carries one of them and asking for both is a caller's bug, refused rather
    than resolved by an arbitrary precedence.
    """
    assert badge.make(3) == {"count": 3}
    assert badge.make(tier=2) == {"tier": 2}
    assert badge.make() is None
    for bad in (lambda: badge.make(3, tier=2), lambda: badge.make(count="many")):
        try:
            bad()
        except (TypeError, ValueError):
            continue
        raise AssertionError("an impossible badge was accepted")


def test_a_badge_that_cannot_be_drawn_is_refused_not_rounded():
    """A count of zero is no badge; a count of a hundred is an error, not "99".

    The first is arithmetic -- zero of a thing is not a badge showing nothing -- and
    the second is the same line ``export.atlas`` draws over an oversized sprite: a
    silent clip is how half a badge ships, and a badge that shows "99" for a hundred
    coins is a lie a player would read.
    """
    assert badge.make(count=0) is None
    assert badge.make(tier=0) is None
    for bad in (lambda: badge.make(count=badge.MAX_COUNT + 1),
                lambda: badge.make(tier=badge.TIERS + 1)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("an undrawable badge was drawn")


def test_the_pips_count_the_tier():
    """A tier is n pips, so the picture has to *have* n of them.

    Measured as connected blobs rather than as area, because area would pass for a
    layout of four pips merged into two long dashes.  Every tier from one to the
    most the badge draws is checked, so a wrapping bug in the second row shows up.
    """
    for tier in range(1, badge.TIERS + 1):
        assert _components(_badge_marks(badge.make(tier=tier))) == tier, tier


def test_every_count_is_its_own_numeral():
    """Two counts, two pictures -- including the two-digit ones, which is where a
    numeral layout can quietly repeat a digit or drop a leading one."""
    fields = {count: _badge_marks(badge.make(count=count))
              for count in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 21, 99)}
    seen = {mask.tobytes() for mask in fields.values()}
    assert len(seen) == len(fields), "two counts drew the same numeral"


def test_a_badged_icon_is_the_plain_icon_plus_a_badge():
    """The no-op and the change, both stated.

    An icon with ``badge=None`` must be the icon that existed before badges did,
    byte for byte -- the whole reason the sixth pass is a branch and not a pass that
    happens to draw nothing.  And an icon *with* a badge must differ, and only by
    growing: a badge adds coverage, it never takes the plate away.
    """
    plain_spec = motifs.canonical("coin")
    none_spec = {**plain_spec, "badge": None}
    plain = paint.draw(plain_spec, 96).to_uint8()
    same = paint.draw(none_spec, 96).to_uint8()
    assert np.array_equal(plain, same), "a None badge changed the render"

    badged = paint.draw({**plain_spec, "badge": badge.make(count=12)}, 96).to_uint8()
    assert not np.array_equal(plain, badged), "a badge drew nothing"
    assert (badged[..., 3] >= plain[..., 3]).all(), "a badge took coverage away"


def test_a_badge_does_not_recolour_the_icon():
    """The badge is drawn in the icon's own colours, and it proves it two ways:

    the colour table is *identical* with and without one (so the badge introduces no
    colour the set did not already have), and the pixels it adds are colours already
    in the plain icon rather than a new hue nobody chose.
    """
    plain = motifs.canonical("water")
    badged = {**plain, "badge": badge.make(tier=3)}
    for key, value in paint.tints(plain).items():
        assert np.allclose(paint.tints(badged)[key], value), key

    a = paint.draw(plain, 128).to_uint8()
    b = paint.draw(badged, 128).to_uint8()
    palette_plain = {tuple(c) for c in a[a[..., 3] > 200][:, :3]}
    # Every *opaque* pixel the badge adds is a shade the plain icon already used --
    # to within the quantization the transfer function does along an edge.
    added = (b[..., 3] > 200) & ~(a[..., 3] > 200)
    if added.any():
        known = np.array(sorted(palette_plain), dtype=np.float64)
        for pixel in b[added][:, :3].astype(np.float64):
            assert (np.abs(known - pixel).max(axis=1) < 24).any(), tuple(pixel)


def test_a_badge_reaches_a_size_where_it_reads_and_never_the_tile_edge():
    """Two claims about the badge's size, at the two ends of the scale.

    It reads: at 48 px a badged icon carries measurably more lightness spread than
    the same icon without one, so the badge is a mark and not a smudge.  And it
    fits: at 16 px, where the margin is a single pixel, the badge still does not
    touch the tile -- so it cannot bleed into a neighbour in an atlas.
    """
    for name in motifs.KINDS:
        plain = paint.draw(motifs.canonical(name), 48)
        badged = paint.draw({**motifs.canonical(name), "badge": badge.make(tier=3)}, 48)
        assert paint.contrast(badged) >= paint.contrast(plain) - 1e-9, name

    for size in (16, 24, 32, 64, 256):
        for name in motifs.KINDS:
            for spec in (badge.make(count=badge.MAX_COUNT), badge.make(tier=badge.TIERS)):
                a = paint.draw({**motifs.canonical(name), "badge": spec}, size).a
                edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
                assert edge.max() < 0.1, (name, size, spec, float(edge.max()))


def test_a_badge_breaks_the_silhouette_where_it_should():
    """A badge sits *on the corner*, so on a plate that does not reach its corners
    it is the one part of an icon outside the plate -- which is what makes it a
    badge.  On a round plate it is inside, and that is fine too."""
    inside = {"round"}
    for plate in motifs.PLATES:
        base = {**motifs.canonical("coin"), "plate": plate}
        plain = paint.silhouette(paint.draw(base, 128))
        badged = paint.silhouette(paint.draw({**base, "badge": badge.make(count=12)}, 128))
        if plate in inside:
            assert abs(badged - plain) < 1e-6, plate
        else:
            assert badged > plain, plate


# ------------------------------------------------------------------ the livery
#: The demo's own roster, so the test measures the names the game actually ships.
_FACTIONS = ["Kingdom of Alder", "The Mirefen League", "Brythmark",
             "Silverreach Compact", "Duchy of Halven", "The Ostmark Freeholds",
             "Vale of Craymoor", "House Delvane"]


def test_a_faction_s_hue_is_the_hue_its_provinces_are_washed_in():
    """The map and the icon set answer with one number.

    A faction's colour is derived once, in ``crest.field_tincture``, and both the
    province wash and the icon plate read it -- so this is not "the two agree", it is
    "there is only one of them".  A drift here would be a faction whose banner is
    blue and whose provinces are green.

    Only *chromatic* factions are compared, and not as a concession: a metal has no
    chroma, so the hue of its grey is numerical noise in both places and comparing
    the two noises would be measuring nothing.  That a metal has no hue is asserted
    where it matters -- in the chroma test below.
    """
    from core import palette as colours
    from gen.maps import politics
    checked = 0
    for name in _FACTIONS:
        if livery.lch(name)[1] < livery.NEUTRAL:
            continue
        washed = float(colours.srgb_to_oklch(politics.wash(name))[2])
        # Not an exact equality: the wash is an sRGB colour and the hue is read back
        # out of it, so the round trip through the transfer function costs the last
        # few digits.  A ten-thousandth of a turn is a thousand times tighter than
        # the smallest hue a plate could show.
        assert abs(livery.hue(name) - washed) < 1e-4, name
        checked += 1
    assert checked >= 2, f"only {checked} chromatic factions in the roster"


def test_a_faction_tints_the_plate_and_leaves_the_charge_alone():
    """The second hue source, stated as an equality and an inequality.

    The charge -- the kind's hue, chroma and lightness -- is *identical* before and
    after, so a faction never makes water stop being blue.  The plate does move, for
    every kind, and to the *same* colour: one faction, one ground, whatever the
    charge on it is.
    """
    for name in _FACTIONS:
        plates = []
        for kind in motifs.KINDS:
            plain = motifs.canonical(kind)
            tinted = livery.tint(plain, name)
            assert tinted["hue"] == plain["hue"]
            assert tinted["chroma"] == plain["chroma"]
            assert tinted["light"] == plain["light"]
            assert np.allclose(paint.tints(tinted)["charge"], paint.tints(plain)["charge"])
            assert abs(tinted["plate_hue"] - livery.hue(name)) < 1e-12
            assert not np.allclose(paint.tints(tinted)["plate"], paint.tints(plain)["plate"])
            plates.append(paint.tints(tinted)["plate"])
        assert all(np.allclose(p, plates[0]) for p in plates), name


def test_a_chromatic_faction_wears_a_colour_and_a_metal_wears_a_grey():
    """The rule the map uses, carried to a plate.

    A colour gives a plate with chroma on it, so a faction is a *colour* and not
    just a shade of grey; a metal or the black gives a plate with almost none, and
    is told apart by lightness -- exactly as Or and Argent are everywhere else.  The
    roster is searched for one of each rather than asserting against a typed-in hue,
    because the tincture table is the authority and a test should not outlive it.
    """
    from core import palette as colours
    chromatic = neutral = None
    for index in range(400):
        name = f"House Test {index}"
        worn_chroma = livery.lch(name)[1]
        if chromatic is None and worn_chroma > 0.05:
            chromatic = name
        if neutral is None and worn_chroma < livery.NEUTRAL:
            neutral = name
        if chromatic and neutral:
            break
    assert chromatic and neutral, "the roster search found no metal and no colour"
    assert livery.chroma(chromatic) > livery.NEUTRAL, livery.chroma(chromatic)
    assert livery.chroma(neutral) < 0.02, livery.chroma(neutral)
    grey = paint.tints(livery.tint(motifs.canonical("coin"), neutral))["plate"]
    assert float(colours.srgb_to_oklch(grey)[1]) < 0.02, "a metal wore a colour"


def test_a_livery_moves_the_plate_s_lightness_so_two_metals_differ():
    """A metal has no hue to wear, so lightness is what tells two of them apart.

    The map draws the same line -- a metal wears a grey and is read by how dark it
    is -- and a plate spends the livery's lightness step the same way.  Two factions
    with no colour at all still end up with two grounds; the bound is what stops a
    plate drifting so light that it stops being a ground for a bright charge.
    """
    from core import palette as colours
    greys = []
    for index in range(300):
        name = f"House Grey {index}"
        if livery.lch(name)[1] >= livery.NEUTRAL:
            continue
        tinted = livery.tint(motifs.canonical("coin"), name)
        assert 0.10 <= tinted["plate_light"] <= 0.56, tinted["plate_light"]
        greys.append(paint.tints(tinted)["plate"])
        if len(greys) >= 40:
            break
    assert len(greys) >= 4, len(greys)
    spread = max(colours.delta_e(a, b) for a in greys for b in greys)
    assert spread > 0.05, spread


def test_two_factions_render_one_kind_as_two_grounds():
    """What the whole thing is for: the same coin, in two factions' colours.

    The demo roster is held to *variety*, not to pairwise distinctness, because
    pairwise distinctness is not a promise the rule makes: two factions that both
    drew a neutral tincture wear the same grey -- the limit the map answers with a
    drawn roster, and worth knowing rather than asserting away.  What is promised is
    that a roster of factions is a spread of grounds, and it is checked on the
    rendered plate colour rather than on the hues, so a bug that produced eight hues
    and one grey plate would fail here.
    """
    from core import palette as colours
    plates = [paint.tints(livery.tint(motifs.canonical("coin"), name))["plate"]
              for name in _FACTIONS]
    spread = max(colours.delta_e(a, b) for a in plates for b in plates)
    distinct = {tuple(np.round(c, 3)) for c in plates}
    assert len(distinct) >= 4, distinct
    assert spread > 0.10, spread


def test_a_badged_livery_icon_is_the_same_in_another_process():
    """The whole increment through one door: a faction's icon, with a badge, derived
    from a name and a count, byte-identical in a fresh interpreter with a different
    hash seed -- because the faction's hue is a blake2b of the name, not ``hash()``.
    """
    code = (
        "import sys, hashlib;"
        "sys.path.insert(0, %r);"
        "from gen.icons import badge, livery, motifs, paint;"
        "s = livery.tint({**motifs.canonical('coin'), 'badge': badge.make(count=12)},"
        "                'Kingdom of Alder');"
        "print(hashlib.sha256(paint.draw(s, 96).to_uint8().tobytes()).hexdigest())"
    )
    root = str(Path(__file__).resolve().parents[1])
    seen = set()
    for salt in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": salt}
        run = subprocess.run([sys.executable, "-c", code % root], capture_output=True,
                             text=True, env=env, check=True)
        seen.add(run.stdout.strip())
    assert len(seen) == 1, seen


# ------------------------------------------------------------------- the atlas
def test_trim_finds_the_pixels_it_says():
    """Including the faintest edge pixel, which is the whole point of threshold 0."""
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    rgba[5:9, 7:13] = 255
    assert atlaslib.trim(rgba) == (7, 5, 13, 9)
    rgba[2, 29, 3] = 1                      # one alpha level: still content
    assert atlaslib.trim(rgba) == (7, 2, 30, 9)
    assert atlaslib.trim(rgba, threshold=1) == (7, 5, 13, 9)


def test_a_fully_transparent_member_is_refused():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    try:
        atlaslib.trim(rgba)
    except ValueError:
        return
    raise AssertionError("an empty sprite was trimmed to a rectangle")


def test_an_atlas_member_is_a_copy_and_not_a_render():
    """The strongest claim in the file: an atlas *moves* sprites, it does not redraw
    them.  Every rectangle is compared with the source pixels it came from."""
    members = _members(6, 48)
    image, info = atlaslib.build(members, padding=2)
    out = image.to_uint8()
    sources = dict(members)
    assert len(info["members"]) == len(members)
    for entry in info["members"]:
        src = sources[entry["name"]].to_uint8()
        x, y, w, h = entry["rect"]
        x0, y0, x1, y1 = entry["trim"]
        assert (w, h) == (x1 - x0, y1 - y0), entry["name"]
        assert np.array_equal(out[y:y + h, x:x + w], src[y0:y1, x0:x1]), entry["name"]


def test_nothing_in_an_atlas_overlaps_or_leaves_the_sheet():
    image, info = atlaslib.build(_members(9, 64), padding=2)
    W, H = info["size"]
    assert image.size == (W, H)
    rects = [entry["rect"] for entry in info["members"]]
    for x, y, w, h in rects:
        assert 0 <= x and 0 <= y and x + w <= W and y + h <= H
    for i, rect in enumerate(rects):
        for other in rects[i + 1:]:
            assert not _overlap(rect, other), (rect, other)


def test_padding_keeps_the_members_apart():
    """Padding is not decoration: two flush sprites bleed into each other the moment
    a bilinear filter samples across the seam.  Half the padding on each side of a
    boundary is the closest two allocations may legally come, so expanding every
    rectangle by ``padding // 2`` must still leave the set non-overlapping."""
    pad = 4
    half = pad // 2
    _image, info = atlaslib.build(_members(9, 64), padding=pad)
    grown = [(x - half, y - half, w + 2 * half, h + 2 * half)
             for x, y, w, h in (entry["rect"] for entry in info["members"])]
    for i, rect in enumerate(grown):
        for other in grown[i + 1:]:
            assert not _overlap(rect, other), (rect, other)


def test_the_same_members_always_pack_the_same_way():
    """Including from a different order, because a caller's dict order is not a
    promise: the packer sorts by name before it lays anything out."""
    members = _members(6, 48)
    first, info = atlaslib.build(members, padding=2)
    second, same = atlaslib.build(list(reversed(members)), padding=2)
    assert info == same
    assert np.array_equal(first.to_uint8(), second.to_uint8())


def test_packing_reports_its_own_density():
    """Power-of-two dimensions are a convenience, and the density is what they cost.

    The test is about the *measurement* being real rather than about a target: it is
    never above one, it drops when the rounding is switched on, and turning the
    rounding off on an even set of icons gets most of the sheet back.
    """
    members = _members(9, 64)
    _a, snug = atlaslib.build(members, padding=2, power_of_two=False)
    _b, pot = atlaslib.build(members, padding=2, power_of_two=True)
    assert 0.0 < snug["density"] <= 1.0
    assert snug["density"] >= pot["density"]
    assert pot["density"] > 0.2
    assert snug["density"] > 0.75, snug["density"]
    for value in pot["size"]:
        assert value & (value - 1) == 0, pot["size"]


def test_a_member_too_big_for_the_atlas_is_refused():
    """A silent clip is how half a sprite ships."""
    try:
        atlaslib.build(_members(3, 96), max_size=128)
    except ValueError:
        return
    raise AssertionError("an oversized set was packed anyway")


def test_an_empty_atlas_is_refused():
    try:
        atlaslib.build([])
    except ValueError:
        return
    raise AssertionError("an empty atlas was written")


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
