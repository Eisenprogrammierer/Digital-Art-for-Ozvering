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
from gen.icons import motifs, paint  # noqa: E402


# ------------------------------------------------------------------- helpers
def _fields(name, spec=None, n: int = 160):
    """The region a charge covers, as a boolean mask on a fixed design grid."""
    axis = np.linspace(-0.75, 0.75, n)
    X, Y = np.meshgrid(axis, axis)
    return motifs.field(spec if spec is not None else motifs.canonical(name), X, Y) < 0


def _iou(a, b) -> float:
    return float((a & b).sum()) / float((a | b).sum())


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
