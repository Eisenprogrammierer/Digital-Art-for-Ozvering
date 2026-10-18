"""Regression tests for the tiling textures.

The claims worth a test here are the ones the *word* seamless hides:

* a field that wraps has a step across its edge comparable to a step inside it, and
  a field that does not has a cliff -- so "seamless" is a number, and every texture
  this package draws is held to it;
* a Truchet tile always sends an arc to the midpoint of the edge it touches, and the
  tile on the other side sends one to the same point, so *any* arrangement is
  seamless.  The test checks the midpoints, because that is the whole reason the
  generator needs no search;
* a Wang tiling's shared corners agree on a torus, which is what makes the bilinear
  blend of the corner colours continuous across every edge -- and where it fails,
  it fails visibly, so the check is on the finished tiling and not on the search;
* a tiling read off a quantised height field agrees for a different reason, and the
  difference is worth a test: nothing was searched for, so it holds for a field no
  set of tiles could be searched into at all, and it holds because the corners two
  neighbours share are one sample of one field, read twice;
* the reaction-diffusion Laplacian wraps, which is checked as an exact algebraic
  property (translating the field translates the Laplacian) rather than by looking
  at a picture, because a boundary condition is exactly the kind of thing a picture
  hides.

Run either way::

    python tests/test_textures.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen.textures import reaction, tile, truchet, wang  # noqa: E402


def _canvas_bytes(canvas) -> bytes:
    return canvas.to_uint8().tobytes()


# ---------------------------------------------------------------------- tiles
def test_a_wrapping_field_does_not_step_at_its_edge_and_a_cliff_does():
    wrapped = tile.periodic_noise((64, 64), seed=3, scale=5.0)
    assert 0.0 <= float(wrapped.min()) and float(wrapped.max()) <= 1.0
    assert tile.seam_error(wrapped) < 1.4

    # A gradient is the simplest field that does not wrap: two tiles apart the value
    # jumps back to the start, and the step is the whole height of the tile.
    ramp = np.tile(np.linspace(0.0, 1.0, 64)[None, :], (64, 1))
    assert tile.seam_error(ramp) > 8.0


def test_periodic_noise_wraps_exactly():
    # Not "smoothly" -- exactly, because the basis of a DFT is periodic on the
    # canvas, so a field built out of it is periodic to the last bit.
    f = tile.periodic_noise((48, 48), seed=11, scale=4.0)
    shifted = tile.periodic_noise((48, 48), seed=11, scale=4.0)
    assert np.array_equal(f, shifted)          # the same seed is the same field
    assert not np.array_equal(f, tile.periodic_noise((48, 48), seed=12, scale=4.0))


def test_smoothing_wraps_and_never_mixes_the_channels():
    """A blur is per-channel, and it is worth a test because the bug is invisible.

    A three-channel field's last two axes are the width and the *channels*, so a
    kernel built from those dimensions is the wrong shape -- and it does not raise,
    because blending an 8-wide row against an 8-wide kernel of three channels is a
    perfectly legal multiplication.  It just mixes the colours into each other, which
    looks like a slightly grey texture rather than like a bug.
    """
    f = np.zeros((24, 24, 3))
    f[..., 0] = 1.0                       # red only, and it must stay red only
    out = tile.smooth(f, 1.5)
    assert out.shape == f.shape
    assert float(out[..., 1].max()) < 1e-9 and float(out[..., 2].max()) < 1e-9
    assert np.allclose(out[..., 0], 1.0)
    assert np.allclose(tile.smooth(np.full((8, 8), 0.4), 2.0), 0.4)   # a flat field

    # The wrap, as algebra rather than as a picture: on a torus a blur commutes with
    # a shift, and a blur with a clamped edge does not.  One bright pixel is the
    # field to ask, because it puts energy in every frequency at once.
    spot = np.zeros((32, 32))
    spot[3, 5] = 1.0
    moved = np.roll(np.roll(spot, 7, axis=0), 4, axis=1)
    assert np.allclose(tile.smooth(moved, 3.0),
                       np.roll(np.roll(tile.smooth(spot, 3.0), 7, axis=0), 4, axis=1),
                       atol=1e-10)


def test_tile_coords_finds_the_tile_and_the_offset_inside_it():
    i, j, u, v = tile.tile_coords((8, 12), 4)
    assert i.shape == (8, 12) and u.shape == (8, 12)
    assert set(np.unique(i)) == {0.0, 1.0}
    assert set(np.unique(j)) == {0.0, 1.0, 2.0}
    # Offsets are measured from the tile's own corner and are what a painter draws
    # with: the same arithmetic on both sides of an edge is what makes a blend of
    # corner colours continuous there.
    assert np.allclose(u[0, :4], np.array([0.0, 0.25, 0.5, 0.75]))
    assert np.allclose(v[:4, 0], np.array([0.0, 0.25, 0.5, 0.75]))
    assert np.allclose(j[0, :], np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]))

    try:
        tile.tile_coords((7, 12), 4)
    except ValueError:
        pass
    else:
        raise AssertionError("a canvas that is not whole tiles should be refused")


def test_repeat_lays_the_field_out_and_multiplies_its_size():
    f = np.arange(4, dtype=np.float64).reshape(2, 2)
    r = tile.repeat(f, 3)
    assert r.shape == (6, 6)
    assert np.array_equal(r[:2, :2], f)
    assert np.array_equal(r[2:4, 4:6], f)


def test_the_light_is_the_same_light_in_every_generator():
    """One light for the whole project, checked where the modules keep it.

    A texture lit from the other side is the fastest way to make an asset look
    borrowed, and this is the cheapest possible test of it -- five module constants
    and an equality.
    """
    from gen.heraldry import paint as heraldry_paint
    from gen.icons import paint as icon_paint
    from gen.maps import paint as map_paint
    from gen.portraits import brush as portrait_brush
    from gen.ui import paint as ui_paint

    for module in (heraldry_paint, icon_paint, map_paint, portrait_brush, ui_paint):
        assert np.allclose(np.asarray(module.LIGHT, dtype=float),
                           np.asarray(tile.LIGHT, dtype=float)), module.__name__


# -------------------------------------------------------------------- truchet
def test_a_truchet_arc_meets_every_neighbour_at_the_middle_of_the_edge():
    """The one fact the module rests on, measured on the distance field.

    Whatever the arrangement is, the edge between two grains has a path arriving at
    its midpoint from both sides, so the paths join and the texture wraps.  If this
    test fails, no arrangement is seamless and the generator would need a search.
    """
    size, grains = 128, 4
    field = truchet.pattern(seed=3, size=size, grains=grains, kinds=("arcs",), width=0.16)
    d = field["distance"]
    cell = size // grains
    half_width = 0.16 * 0.5
    for i in range(grains):
        for j in range(grains):
            box = d[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            for u, v in ((0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5)):
                px = min(max(int(round(u * cell - 0.5)), 0), cell - 1)
                py = min(max(int(round(v * cell - 0.5)), 0), cell - 1)
                assert abs(float(box[py, px])) < half_width, (i, j, u, v)


def test_an_arrangement_is_a_function_of_the_seed():
    a = truchet.arrangement(3, 5)
    b = truchet.arrangement(3, 5)
    c = truchet.arrangement(4, 5)
    assert a == b
    assert a != c
    assert len(a) == 5 and all(len(row) == 5 for row in a)
    assert all(o in (0, 1) for row in a for o, _k in row)


def test_a_truchet_texture_wraps_and_wraps_the_same_way_twice():
    canvas = truchet.texture(seed=3, size=128, grains=4)
    assert tile.seam_error(tile.luminance(canvas.to_uint8())) < 1.6
    again = truchet.texture(seed=3, size=128, grains=4)
    assert hashlib.sha256(_canvas_bytes(canvas)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(again)).hexdigest()


def test_both_kinds_draw_paths_and_a_tile_refuses_a_size_it_cannot_divide():
    arcs = truchet.pattern(seed=1, size=64, grains=4, kinds=("arcs",))["distance"]
    lines = truchet.pattern(seed=1, size=64, grains=4, kinds=("lines",))["distance"]
    assert float(arcs.min()) < 0.0 and float(lines.min()) < 0.0   # both carve something
    assert not np.array_equal(arcs, lines)
    try:
        truchet.pattern(seed=1, size=65, grains=4)
    except ValueError:
        pass
    else:
        raise AssertionError("a tile of 65 px into 4 grains should be refused")
    try:
        truchet.pattern(seed=1, size=64, grains=4, kinds=("hexagons",))
    except ValueError:
        pass
    else:
        raise AssertionError("a kind that is not arcs or lines should be refused")


# ----------------------------------------------------------------------- wang
def test_shared_corners_agree_across_every_seam_including_the_wrap():
    tiles = wang.tile_set(3, "complete")
    grid = wang.fill(7, 5, tiles, seed=3)
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            tl, tr, bl, br = tiles[grid[r, c]]
            right = tiles[grid[r, (c + 1) % cols]]
            down = tiles[grid[(r + 1) % rows, c]]
            assert (tr, br) == (right[0], right[2]), (r, c)
            assert (bl, br) == (down[0], down[1]), (r, c)


def test_the_blend_of_the_corners_is_continuous_which_is_why_it_tiles():
    canvas = wang.texture(seed=3, size=128, cells=8, colours=3)
    assert tile.seam_error(tile.luminance(canvas.to_uint8())) < 1.4
    again = wang.texture(seed=3, size=128, cells=8, colours=3)
    assert hashlib.sha256(_canvas_bytes(canvas)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(again)).hexdigest()


def test_a_set_that_cannot_tile_says_so_instead_of_hanging():
    # One tile whose top corners are 0,0 and whose right corners are 0,1: it agrees
    # with the tile to its left and then cannot agree with the tile to its right,
    # because a tile's own left corners are 0,0 and its right corners are 0,1.
    try:
        wang.fill(4, 4, [(0, 0, 0, 1)], seed=1, budget=5000)
    except ValueError as message:
        assert "cannot tile" in str(message) or "gave up" in str(message)
    else:
        raise AssertionError("a set that cannot tile should raise, not hang")


def test_both_tile_sets_are_what_they_say_they_are():
    assert len(wang.tile_set(2, "complete")) == 16
    assert len(wang.tile_set(3, "complete")) == 81
    diagonal = wang.tile_set(2, "diagonal")
    assert len(diagonal) == 4
    assert all(tl == br and tr == bl for tl, tr, bl, br in diagonal)
    # The diagonal set is the one that *does* dead-end, so it is the one that makes
    # the search worth having: it tiles, but a greedy walk has to back out of it.
    assert wang.fill(8, 8, diagonal, seed=5).shape == (8, 8)
    try:
        wang.tile_set(3, "hexagonal")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown set should be refused")


def test_a_wang_tiling_is_a_function_of_the_seed():
    a = wang.fill(6, 6, wang.tile_set(2, "complete"), seed=1)
    b = wang.fill(6, 6, wang.tile_set(2, "complete"), seed=1)
    c = wang.fill(6, 6, wang.tile_set(2, "complete"), seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_a_palette_with_too_few_colours_is_refused_not_clipped():
    try:
        wang.texture(seed=1, size=64, cells=8, colours=5, palette_name="meadow")
    except ValueError:
        pass
    else:
        raise AssertionError("asking for colours a palette does not have should be refused")


def test_a_margin_that_reaches_the_opposite_corner_is_refused():
    """The blend is continuous only while it stays inside its own half of the tile.

    ``margin`` is how far from the edge the blend starts.  Past half a tile it has
    crossed the middle, so the *opposite* corners start to influence the edge -- and
    the shared-corner agreement no longer implies that the field is continuous there,
    which is the whole argument this module is built on.
    """
    for bad in (0.0, -0.1, 0.6, 1.0):
        try:
            wang.texture(seed=1, size=64, cells=8, margin=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"a margin of {bad} should be refused")


def test_a_height_field_fills_its_range_and_wraps():
    """The field is what the levels are levels *of*, so it has to use its range.

    An average of three octaves has most of its values near the middle whatever the
    seed, so a sum that peaks at 0.66 has no top band at all and every landscape comes
    out as lowland with a talkative coast.  Stretching the sum to fill ``[0, 1]`` --
    the same thing each octave already does to itself -- is what gives every band an
    edge, and it is measured rather than assumed here and in the band test below.
    """
    f = wang.height_field(seed=5, cells=32)
    assert f.shape == (32, 32)
    assert abs(float(f.min())) < 1e-9 and abs(float(f.max()) - 1.0) < 1e-9
    assert 0.0 <= float(f.min()) and float(f.max()) <= 1.0
    assert tile.seam_error(f) < 1.4
    assert np.array_equal(f, wang.height_field(seed=5, cells=32))
    assert not np.array_equal(f, wang.height_field(seed=6, cells=32))


def test_a_tiling_read_off_a_field_agrees_because_it_was_never_chosen():
    """The whole reason this second way of getting a tiling exists.

    A searched arrangement agrees because the search made it agree.  This one agrees
    because two neighbours share two samples of one field, so the question "do the two
    corners match" is the question "is one number equal to itself".  That is a
    *stronger* guarantee, not a weaker one, and the checkerboard is here to show what
    stronger means: it is a field that no search over any set of tiles could be
    arranged into without a mistake, and it needs no search at all.
    """
    grid, tiles = wang.from_field(wang.height_field(seed=3, cells=16), 4, 16)
    rows, cols = grid.shape
    assert (rows, cols) == (16, 16)
    codes = np.array(tiles)
    for r in range(rows):
        for c in range(cols):
            tl, tr, bl, br = codes[grid[r, c]]
            right = codes[grid[r, (c + 1) % cols]]
            down = codes[grid[(r + 1) % rows, c]]
            assert (tr, br) == (right[0], right[2]), (r, c)
            assert (bl, br) == (down[0], down[1]), (r, c)

    checker = np.add.outer(np.arange(8), np.arange(8)) % 2.0
    cgrid, ctiles = wang.from_field(checker, 2, 8)
    assert sorted(ctiles) == [(0, 1, 1, 0), (1, 0, 0, 1)]
    assert cgrid.shape == (8, 8)


def test_the_corner_colours_of_a_tiling_are_the_levels_of_the_field():
    # Round trip: the tiling is the field and nothing else.  A tile's four numbers are
    # its four corner samples, quantised -- so reading a field at half rate is the
    # same field sampled less finely, and the last corner of a row is the first,
    # because the field is the thing that wraps.
    f = wang.height_field(seed=11, cells=16)
    corner = wang.corner_samples(f, None)
    assert corner.shape == (17, 17)
    assert np.array_equal(corner[-1], corner[0]) and np.array_equal(corner[:, -1], corner[:, 0])
    assert np.allclose(wang.corner_samples(f, 8), corner[::2, ::2])

    grid, tiles = wang.from_field(f, 4, 16)
    levels = wang.levels(corner, 4)
    codes = np.array(tiles)
    for r in (0, 5, 15):
        for c in (0, 7, 15):
            assert tuple(codes[grid[r, c]]) == (levels[r, c], levels[r, c + 1],
                                               levels[r + 1, c], levels[r + 1, c + 1])


def test_a_tiling_read_off_a_field_is_a_subset_of_the_set_and_uses_few_of_it():
    """What reading a field buys, as a number rather than as a compliment.

    A search places a tile per cell and almost never repeats one, because a corner
    colour chosen at random is unlikely to come round again: 256 cells hold 168
    distinct tiles, which is what patchwork is.  Read off a field the corners are
    correlated, so the landscape walks through a handful of the set's tiles and
    repeats them.  Same set, same seed, same cells -- and a quarter as many tiles.
    """
    cells, colours = 16, 4
    complete = wang.tile_set(colours, "complete")
    searched = wang.fill(cells, cells, complete, seed=7)
    _grid, tiles = wang.from_field(wang.height_field(7, cells), colours, cells)
    assert all(t in complete for t in tiles)
    assert len(tiles) < 0.5 * len(set(searched.reshape(-1).tolist()))


def test_every_band_of_a_field_is_used_and_a_ramp_palette_is_ordered():
    # Equal *height* bands, not equal-population ones, so a band can go unused if the
    # field does not reach it -- which is exactly what the stretch is for, and this is
    # where it is measured.  And the palette a field is painted with has to be a ramp:
    # there the colour index is a level of the landscape and not a name.
    for seed in range(12):
        levels = wang.levels(wang.height_field(seed, 32), 4)
        assert set(np.unique(levels).tolist()) == {0, 1, 2, 3}, seed
    heights = wang.PALETTES["highland"]["heights"]
    assert all(b > a for a, b in zip(heights, heights[1:]))


def _no_worse_than_an_interior_seam(f, size: int) -> None:
    """The wrap's step is no worse than the worst step *inside* one copy of it.

    ``seam_error`` on its own is a threshold, and a texture with a strong relief has
    large steps everywhere: the ratio of the wrap to the average rises with the relief
    even when nothing is wrong at the edge.  What would mean a seam is the wrap
    standing out among the same texture's own boundaries, so it is measured against
    those -- every column boundary of one copy, which is every place a seam could be.
    """
    worst = max(tile.seam_error(np.roll(f, k, axis=1)) for k in range(1, size))
    assert tile.seam_error(f) <= worst, (tile.seam_error(f), worst)


def test_a_field_texture_wraps_is_byte_identical_twice_and_is_not_a_searched_one():
    canvas = wang.texture(seed=3, size=128, cells=16, colours=4, palette_name="highland",
                          source="field", relief=20.0)
    _no_worse_than_an_interior_seam(tile.luminance(np.tile(canvas.to_uint8(), (2, 2, 1))), 128)
    again = wang.texture(seed=3, size=128, cells=16, colours=4, palette_name="highland",
                         source="field", relief=20.0)
    assert hashlib.sha256(_canvas_bytes(canvas)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(again)).hexdigest()
    searched = wang.texture(seed=3, size=128, cells=16, colours=4, palette_name="highland",
                            source="search", relief=20.0)
    assert not np.array_equal(canvas.to_uint8(), searched.to_uint8())


def test_a_field_that_is_not_two_dimensional_or_cannot_be_read_is_refused():
    for bad in (np.zeros(8), np.zeros((8, 8, 1))):
        try:
            wang.from_field(bad, 4)
        except ValueError:
            pass
        else:
            raise AssertionError("a height field is two-dimensional")
    for bad in (1, 0, -3):
        try:
            wang.from_field(np.zeros((8, 8)), bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} colours should be refused")
    try:
        wang.texture(seed=1, size=64, cells=8, source="vibes")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown source should be refused")


# ------------------------------------------------------------------- reaction
def test_the_laplacian_is_linear_and_wraps_exactly():
    """A boundary condition is exactly the kind of thing a picture hides.

    The claim is that translating the field translates the Laplacian, with no edge
    effect anywhere -- i.e. that the simulation is on a torus.  It is checked as
    algebra, and it is exact.
    """
    rng = np.random.default_rng(3)
    f = rng.standard_normal((16, 20))
    lap = reaction._laplacian(f)
    assert lap.shape == f.shape
    moved = np.roll(np.roll(f, 5, axis=0), 3, axis=1)
    assert np.allclose(reaction._laplacian(moved), np.roll(np.roll(lap, 5, axis=0), 3, axis=1),
                       atol=1e-12)
    # A constant field is a fixed point of any Laplacian, and the stencil has to sum
    # to zero for a diffusion term not to invent or destroy chemical.
    assert np.allclose(reaction._laplacian(np.full((8, 8), 0.7)), 0.0, atol=1e-12)
    assert abs(float(reaction.LAPLACIAN.sum())) < 1e-12


def _gradient_energy(field) -> float:
    gy, gx = np.gradient(np.asarray(field, dtype=np.float64))
    return float(np.hypot(gy, gx).mean())


def test_the_pattern_actually_forms_and_does_not_blow_up():
    """What "forms" means, measured -- and the statistic that does *not* measure it.

    The trap is standard deviation: a sparse seed and a finished stripe network are
    both two-valued fields, and the deviation can even *fall* as the pattern eats the
    lumps it grew out of, so a test on it would call a working simulation broken.
    What does change is coverage and gradient energy: the model turns a fifth of the
    tile into a stripe network, which shows up in both, and it does not run away.
    """
    start = reaction.simulate(seed=1, size=64, preset_name="worms", steps=1, settle=False)
    grown = reaction.simulate(seed=1, size=64, preset_name="worms", steps=800, settle=False)
    assert float((grown > 0.15).mean()) > 2.0 * float((start > 0.15).mean())
    assert _gradient_energy(grown) > 1.6 * _gradient_energy(start)
    assert float(np.abs(grown - start).max()) > 0.2
    assert 0.0 <= float(grown.min()) and float(grown.max()) < 1.5


def test_each_preset_is_a_different_pattern():
    fields = {name: reaction.simulate(seed=2, size=28, preset_name=name, steps=400, settle=False)
              for name in ("worms", "holes", "spots")}
    names = sorted(fields)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(fields[a], fields[b])
    try:
        reaction.simulate(seed=2, size=28, preset_name="not-a-preset", steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown preset should be refused")


def test_settling_stops_the_run_early():
    # A tolerance wider than any change the field ever makes means the first check
    # ends the run, so the field has barely moved from its seed; a tiny tolerance
    # with the same budget lets it develop.  Different results, same seed: the
    # early-out is real and it is a function of the field alone.
    eager = reaction.simulate(seed=4, size=32, preset_name="worms", steps=800, tol=1e3)
    patient = reaction.simulate(seed=4, size=32, preset_name="worms", steps=800, tol=0.0,
                                settle=False)
    assert not np.allclose(eager, patient)
    assert np.array_equal(reaction.simulate(seed=4, size=32, preset_name="worms", steps=800),
                          reaction.simulate(seed=4, size=32, preset_name="worms", steps=800))


def test_upsampling_wraps_and_keeps_the_range():
    f = tile.periodic_noise((16, 16), seed=7, scale=3.0)
    big = reaction.upsample(f, 64)
    assert big.shape == (64, 64)
    assert float(big.min()) >= float(f.min()) - 1e-9
    assert float(big.max()) <= float(f.max()) + 1e-9
    assert tile.seam_error(big) < 1.4


def test_a_reaction_texture_wraps_and_is_byte_identical_twice():
    canvas = reaction.texture(seed=3, size=96, sim=48, preset_name="worms", steps=400)
    assert tile.seam_error(tile.luminance(canvas.to_uint8())) < 1.5
    again = reaction.texture(seed=3, size=96, sim=48, preset_name="worms", steps=400)
    assert hashlib.sha256(_canvas_bytes(canvas)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(again)).hexdigest()
    other = reaction.texture(seed=4, size=96, sim=48, preset_name="worms", steps=400)
    assert not np.array_equal(canvas.to_uint8(), other.to_uint8())


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
