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
  hides;
* a texture atlas is a sprite atlas with both of its margin rules reversed: a tile
  cannot be trimmed, because a tile's edge is its seam and not a margin, and its
  margin cannot be a gap, because a tile is opaque and the margin is what a filtered
  sample beside it will read.  So the margin is the tile wrapped onto itself, and the
  test checks that pixel for pixel;
* a ground material is a table of fifteen fields a row, and that makes two claims
  testable: every number changes the picture (the failure a table invites is a field
  that is documented and unread), and the pair of materials closest in colour is far
  apart in relief -- because a material is what the light finds.  The cells inside one
  of them are a Voronoi diagram on a torus, checked as geometry -- where two pixels
  disagree, ``edge`` says a border is between them -- rather than against the sites it
  was built from, which is what a crop-and-tile version would pass.

Run either way::

    python tests/test_textures.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen.textures import atlas, boards, ground, reaction, tile, truchet, wang
from core import palette
from core.canvas import Canvas


def _canvas_bytes(canvas) -> bytes:
    return canvas.to_uint8().tobytes()


def test_a_wrapping_field_does_not_step_at_its_edge_and_a_cliff_does():
    wrapped = tile.periodic_noise((64, 64), seed=3, scale=5.0)
    assert 0.0 <= float(wrapped.min()) and float(wrapped.max()) <= 1.0
    assert tile.seam_error(wrapped) <= 1.5

    ramp = np.tile(np.linspace(0.0, 1.0, 64)[None, :], (64, 1))
    assert tile.seam_error(ramp) > 8.0

    assert tile.seam_error(ramp) > 10.0 * tile.seam_error(wrapped)


def test_periodic_noise_wraps_exactly():

    f = tile.periodic_noise((48, 48), seed=11, scale=4.0)
    shifted = tile.periodic_noise((48, 48), seed=11, scale=4.0)
    assert np.array_equal(f, shifted)
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
    f[..., 0] = 1.0
    out = tile.smooth(f, 1.5)
    assert out.shape == f.shape
    assert float(out[..., 1].max()) < 1e-9 and float(out[..., 2].max()) < 1e-9
    assert np.allclose(out[..., 0], 1.0)
    assert np.allclose(tile.smooth(np.full((8, 8), 0.4), 2.0), 0.4)

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
    """One light for the whole project, and *one* -- not eight that agree.

    A texture lit from the other side is the fastest way to make an asset look
    borrowed, so every package exposes a ``LIGHT``.  Those used to be eight copies
    of a literal kept in step by this test; they are now one object in
    ``core.render``, and the identity check is what says so -- an equality would
    pass just as happily on nine copies.
    """
    from core import render
    from gen.heraldry import paint as heraldry_paint
    from gen.icons import badge as icon_badge
    from gen.icons import paint as icon_paint
    from gen.maps import paint as map_paint
    from gen.portraits import brush as portrait_brush
    from gen.text import pen as text_pen
    from gen.ui import paint as ui_paint

    assert tile.LIGHT is render.LIGHT
    assert inspect.signature(render.shade).parameters["light"].default is render.LIGHT
    for module in (heraldry_paint, icon_paint, icon_badge, map_paint, portrait_brush,
                   text_pen, ui_paint):
        assert module.LIGHT is render.LIGHT, module.__name__


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
    assert tile.seam_error(tile.luminance(canvas.to_uint8())) <= 1.5
    again = truchet.texture(seed=3, size=128, grains=4)
    assert hashlib.sha256(_canvas_bytes(canvas)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(again)).hexdigest()


def test_both_kinds_draw_paths_and_a_tile_refuses_a_size_it_cannot_divide():
    arcs = truchet.pattern(seed=1, size=64, grains=4, kinds=("arcs",))["distance"]
    lines = truchet.pattern(seed=1, size=64, grains=4, kinds=("lines",))["distance"]
    assert float(arcs.min()) < 0.0 and float(lines.min()) < 0.0
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
    assert tile.seam_error(tile.luminance(canvas.to_uint8())) <= 1.5
    again = wang.texture(seed=3, size=128, cells=8, colours=3)
    assert hashlib.sha256(_canvas_bytes(canvas)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(again)).hexdigest()


def test_a_set_that_cannot_tile_says_so_instead_of_hanging():

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
    assert tile.seam_error(f) <= 1.5
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

    for seed in range(12):
        levels = wang.levels(wang.height_field(seed, 32), 4)
        assert set(np.unique(levels).tolist()) == {0, 1, 2, 3}, seed
    heights = wang.PALETTES["highland"]["heights"]
    assert all(b > a for a, b in zip(heights, heights[1:]))


def test_a_field_texture_wraps_is_byte_identical_twice_and_is_not_a_searched_one():
    canvas = wang.texture(seed=3, size=128, cells=16, colours=4, palette_name="highland",
                          source="field", relief=20.0)

    assert tile.seam_error(tile.luminance(canvas.to_uint8())) <= 1.5
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
    assert tile.seam_error(big) <= 1.5


def test_a_reaction_texture_wraps_and_is_byte_identical_twice():
    canvas = reaction.texture(seed=3, size=96, sim=48, preset_name="worms", steps=400)
    assert tile.seam_error(tile.luminance(canvas.to_uint8())) <= 1.5
    again = reaction.texture(seed=3, size=96, sim=48, preset_name="worms", steps=400)
    assert hashlib.sha256(_canvas_bytes(canvas)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(again)).hexdigest()
    other = reaction.texture(seed=4, size=96, sim=48, preset_name="worms", steps=400)
    assert not np.array_equal(canvas.to_uint8(), other.to_uint8())


def test_a_board_that_crosses_the_seam_has_one_name():
    """The one thing a floor needs that nothing else in the package does.

    The stagger moves each band's joints along, so in most bands a joint falls inside
    the tile and the board it ends runs off the left edge and comes back on the right.
    That is *one* board, and its grain and its tone have to say so -- a tile-local name
    would give its two halves different wood, and the seam would read as a mistake
    rather than as the joint it is.  The identity comes out right because it is taken
    modulo the number of boards *on the torus*, and this is where that is checked: over
    a sweep of board counts, band counts, staggers and sizes, on every row, and at the
    wrap itself.

    The coordinate is the sharper half of the check.  It has to carry on across the
    edge by exactly one pixel's worth, and where it instead runs off the end, the
    identity has to step by exactly one board -- because that is what running off the
    end *is*: a joint, and the next board along.
    """
    for across in (1, 2, 3, 4):
        for down in (1, 2, 3, 5, 6):
            for stagger in (0, 1, 2, 3):
                for size in (96, 128):
                    lay = boards.layout(size=size, across=across, down=down,
                                        stagger=stagger)
                    u, board = lay["u"], lay["board"]
                    step = across / size
                    drift = np.mod(u[:, 0] - u[:, -1] - step, 1.0)
                    assert float(np.minimum(drift, 1.0 - drift).max()) < 1e-9
                    crossed = u[:, -1] + step >= 1.0
                    ok = np.where(crossed,
                                  np.mod(board[:, 0] - board[:, -1], across) == 1 % across,
                                  board[:, 0] == board[:, -1])
                    assert ok.all(), (across, down, stagger, size)

                    v = lay["v"]
                    vstep = down / size
                    vdrift = np.mod(v[0, :] - v[-1, :] - vstep, 1.0)
                    assert float(np.minimum(vdrift, 1.0 - vdrift).max()) < 1e-9


def test_a_floor_is_a_whole_number_of_boards_across_and_down():

    for across, down in ((1, 1), (1, 6), (2, 5), (3, 3), (4, 2)):
        lay = boards.layout(size=128, across=across, down=down, stagger=2)
        assert set(np.unique(lay["band"]).tolist()) == set(range(down))
        assert set(np.unique(lay["board"]).tolist()) == set(range(across))
        assert set(np.unique(lay["id"]).tolist()) == set(range(across * down))
        assert lay["id"].shape == (128, 128) and lay["edge"].shape == (128, 128)
        assert float(lay["edge"].min()) >= 0.0


def test_stretching_the_noise_stretches_its_features():
    """``stretch`` is the primitive the whole floor is built on, so it gets its own test.

    A feature eight times longer along x than along y means the field is far more
    correlated with itself when it is shifted along x than when it is shifted along y,
    which is a measurement rather than a look at a picture.  And at 1 it is the noise
    it always was -- not "close to", the same array, because the only thing a stretch
    does is multiply a width by itself.
    """
    plain = tile.periodic_noise((96, 96), seed=4, scale=5.0)
    assert np.array_equal(plain, tile.periodic_noise((96, 96), seed=4, scale=5.0,
                                                    stretch=1.0))

    def corr(a, lag, axis):
        return float(np.corrcoef(np.roll(a, lag, axis=axis).ravel(), a.ravel())[0, 1])

    assert abs(corr(plain, 4, 1) - corr(plain, 4, 0)) < 0.05
    long = tile.periodic_noise((96, 96), seed=4, scale=5.0, stretch=8.0)
    assert corr(long, 4, 1) > 0.9
    assert corr(plain, 4, 1) < 0.9
    assert corr(long, 8, 1) > corr(long, 8, 0) + 0.4


def test_a_floor_wraps_and_is_byte_identical_twice():

    canvas = boards.texture(seed=7, size=192, across=1, down=6, stagger=2)
    assert tile.seam_error(tile.luminance(canvas.to_uint8())) <= 2.5
    again = boards.texture(seed=7, size=192, across=1, down=6, stagger=2)
    assert hashlib.sha256(_canvas_bytes(canvas)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(again)).hexdigest()
    other = boards.texture(seed=8, size=192, across=1, down=6, stagger=2)
    assert not np.array_equal(canvas.to_uint8(), other.to_uint8())


def test_a_floor_has_knots_and_they_are_marks_rather_than_planks():

    size = 192
    lay = boards.layout(size=size, across=1, down=6, stagger=2)
    inside = lay["edge"] > 5.0

    def spread(knots):
        cv = boards.texture(seed=7, size=size, across=1, down=6, stagger=2, knots=knots)
        return float(tile.luminance(cv.to_uint8())[inside].std())

    assert spread(1.0) > spread(0.0)


def test_a_floor_that_is_not_a_floor_is_refused():
    for across, down in ((0, 6), (1, 0), (-2, 3)):
        try:
            boards.layout(size=64, across=across, down=down)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{across} boards across and {down} down is not a floor")
    try:
        boards.texture(seed=1, size=0, across=1, down=6)
    except ValueError:
        pass
    else:
        raise AssertionError("a floor of no pixels should be refused")


def _ground_row(name):
    return {g.key: g for g in ground.GROUNDS}[name]


def _nearest_other_cell(fid, y, x, size):
    """Distance on the torus from a pixel to the nearest pixel of a *different* cell."""
    ys = np.abs(np.arange(size) - y)
    ys = np.minimum(ys, size - ys)
    xs = np.abs(np.arange(size) - x)
    xs = np.minimum(xs, size - xs)
    d = np.hypot(ys[:, None], xs[None, :])
    return float(d[fid != fid[y, x]].min())


def test_every_ground_tiles_at_every_size():
    """The table is the claim: six materials, and *each* of them is seamless.

    The number is measured on the finished raster rather than on the height field,
    because the seam the eye finds is in the picture: the light, the colour and the grit
    all have to wrap, and a height that wrapped on its own would earn a good number for a
    tile that still showed a line where the colour changed character.  Two is the bound
    rather than one: a material with cracks in it can have a joint land exactly on the
    seam, and a joint is not a mistake -- the boards' grooves read 1.4 on this number.
    """
    assert len(ground.names()) == 6
    assert len(set(ground.names())) == 6
    for name in ground.names():
        for size in (64, 128):
            c = ground.texture(name, seed=7, size=size)
            assert (c.w, c.h) == (size, size)
            rgba = c.to_uint8()
            assert rgba[..., 3].min() == 255, name
            seam = tile.seam_error(tile.luminance(rgba))
            assert seam <= 2.0, (name, size, seam)


def test_a_ground_is_a_height_field_with_a_colour_on_it():
    """What the module is: the material is relief first, colour second.

    Not a slogan -- a testable split.  The tone is a number in ``0..1`` over the whole
    material and the height is in texels, so the light has something to find; the
    rendered tile then has variance of its own, which is the only thing that separates a
    material from a fill.  Snow is the case that matters: it is nearly white, so all of
    its material has to come from the height, and it still has to move.
    """
    for name in ground.names():
        f = ground.fields(name, seed=7, size=128)
        g, t, h = f["ground"], f["tone"], f["height"]
        assert np.isfinite(t).all() and np.isfinite(h).all(), name
        assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0, name

        assert float(np.abs(h).max()) <= g.relief + g.grit + 1e-9, name
        assert float(h.std()) > 0.30, (name, float(h.std()))
        lum = tile.luminance(ground.texture(name, seed=7, size=64).to_uint8())
        assert float(lum.std()) > 0.015, (name, float(lum.std()))


def test_six_materials_are_six():
    """Every pair differs by more than a JPEG artefact, measured in OKLab.

    The number is the mean perceptual distance between the two finished tiles, pixel by
    pixel, so it cannot be earned by one lucky colour: two materials that differed only
    in their average score it, and so do two that differed only in their grain -- which
    is the point, because the table is allowed to use both.
    """
    seen = list(ground.names())
    tiles = {n: ground.texture(n, seed=7, size=64) for n in seen}
    lab = {n: palette.srgb_to_oklab(tiles[n].rgb) for n in seen}
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            d = float(np.linalg.norm(lab[a] - lab[b], axis=-1).mean())
            assert d > 0.05, (a, b, d)


def test_the_closest_pair_in_colour_is_far_apart_in_relief():
    """Grass and rock: a mid green and a mid grey, and nothing whatever alike.

    This is the module's one argument, made as a measurement.  Some pair of the fifteen
    has to be the closest in colour -- close enough that a generator that varied only
    colour would be entitled to call them variants -- and the claim is that the set is
    built so that this pair is far apart in the number that actually distinguishes a
    material: what the light finds.  If the six ever drifted into six tints of one
    surface, this is the test that would notice.
    """
    seen = list(ground.names())
    tiles = {n: ground.texture(n, seed=7, size=64) for n in seen}
    lab = {n: palette.srgb_to_oklab(tiles[n].rgb) for n in seen}
    pairs = [(float(np.linalg.norm(lab[a] - lab[b], axis=-1).mean()), a, b)
             for i, a in enumerate(seen) for b in seen[i + 1:]]
    _d, a, b = min(pairs)
    ha = ground.fields(a, seed=7, size=64)["height"]
    hb = ground.fields(b, seed=7, size=64)["height"]
    ratio = max(float(ha.std()), float(hb.std())) / min(float(ha.std()), float(hb.std()))
    assert ratio > 2.0, (a, b, ratio)


def test_no_number_in_the_table_is_decoration():
    """A row is fifteen fields and every value has to change the picture.

    The failure a table invites is the field that is written down, documented, put in the
    sidecar -- and not read.  So each value is perturbed on a kind that uses it and the
    finished tile has to come out different -- the base colour counts once, the thirteen
    numbers count one each, and ``key`` is the one exception, because it is the row's
    name and not a material.  Everything else has to be wired to the painter.
    """
    changes = [
        ("grass", {"base": (0.55, 0.140, 0.410)}),
        ("grass", {"drop": 0.30}),
        ("grass", {"turn": 0.15}),
        ("grass", {"lump": 0.40}),
        ("grass", {"lump_scale": 14.0}),
        ("sand", {"ribbon": 0.40}),
        ("sand", {"ribbon_scale": 4.0}),
        ("sand", {"ribbon_stretch": 1.5}),
        ("rock", {"facet": 0.50}),
        ("rock", {"cells": 6}),
        ("rock", {"bevel": 0.40}),
        ("shore", {"dome": 0.00}),
        ("grass", {"grit": 0.60}),
        ("grass", {"relief": 1.00}),
    ]
    for name, change in changes:
        row = _ground_row(name)
        before = _canvas_bytes(ground.texture(row, seed=5, size=64))
        after = _canvas_bytes(ground.texture(row._replace(**change), seed=5, size=64))
        assert before != after, (name, change)


def test_a_seventh_ground_is_a_row():
    """The extension point is data: a slate is rock's row with two numbers changed.

    Nothing is registered and nothing is special-cased -- ``texture`` takes the row
    itself -- which is the difference between a set of six pictures and a table.  The
    slate has to behave like a material: it tiles, it is darker than the stone it was
    copied from, and it is a picture, not a promise.
    """
    rock = _ground_row("rock")
    slate = rock._replace(key="slate", base=(0.36, 0.012, 0.300), drop=0.09)
    a = ground.texture(rock, seed=3, size=64)
    b = ground.texture(slate, seed=3, size=64)
    assert _canvas_bytes(a) != _canvas_bytes(b)
    assert _canvas_bytes(b) == _canvas_bytes(ground.texture(slate, seed=3, size=64))
    assert tile.seam_error(tile.luminance(b.to_uint8())) <= 2.0
    assert tile.luminance(b.to_uint8()).mean() < tile.luminance(a.to_uint8()).mean()


def test_the_row_in_the_sidecar_is_the_whole_recipe():
    """A row is not a description if a number is missing from it.

    The sidecar claims to carry the row that made the tile, so the row is put through
    the same JSON a sidecar goes through and handed back to ``texture`` without a name
    and without a registry: if the tile changes, something in the picture came from a
    default rather than from the table, and the sidecar was a summary.
    """
    for name in ground.names():
        row = {**json.loads(json.dumps(_ground_row(name)._asdict()))}

        rebuilt = ground.Ground(**{**row, "base": tuple(row["base"])})
        assert rebuilt == _ground_row(name), name
        a = _canvas_bytes(ground.texture(name, seed=3, size=64))
        b = _canvas_bytes(ground.texture(rebuilt, seed=3, size=64))
        assert a == b, name


def test_an_unknown_ground_is_refused_with_the_set_in_the_message():
    """A typo has to fail loudly, and the error is the place to list the alternatives."""
    try:
        ground.texture("moss", seed=1, size=32)
    except ValueError as e:
        assert "moss" in str(e), str(e)
        for name in ground.names():
            assert name in str(e), str(e)
    else:
        raise AssertionError("an unknown material should not be painted")


def test_the_material_set_covers_the_map_and_leaves_the_water_alone():
    """The tie to ``gen.maps``: *what a hex is* and *what it is made of* are two tables.

    The map has twelve biomes and three of them are water; a ground texture is what a
    figure stands on.  So every land biome has to name a material, no water biome may,
    and the one material with no biome at all -- dirt -- is deliberate: it is the ground
    under a road, a camp or a ploughed field, and no biome is made of it.

    The tables live in different packages on purpose, and this test is the only place
    that knows both, so either can grow without dragging the other behind it.
    """
    from gen.maps import terrain

    land = {b.key for b in terrain.BIOMES if not b.water}
    water = {b.key for b in terrain.BIOMES if b.water}
    assert set(ground.FOR_BIOME) == land, set(ground.FOR_BIOME) ^ land
    assert not (set(ground.FOR_BIOME) & water)
    assert set(ground.FOR_BIOME.values()) <= set(ground.names())
    worn = set(ground.FOR_BIOME.values())
    assert set(ground.names()) - worn == {"dirt"}, set(ground.names()) - worn


def test_a_voronoi_is_the_geometry_it_says_it_is():
    """``edge`` is checked against the picture rather than against the points.

    A nearest-site diagram has two properties that need no sites to state.  Every pixel
    belongs to one cell, so the only place a neighbour may disagree is within a pixel of
    a border, where ``edge`` is small.  And a border lies between the two sites that own
    it, so the nearest pixel of another cell is just past ``edge``.  The second is the one
    that catches a search which looked at the wrong nine cells: a wrong assignment can
    still be locally consistent, but then its ``edge`` stops predicting where the border
    is -- and on a crack, that is a kink nobody can find later.

    The prediction is one-sided on purpose.  ``edge`` is the distance to the bisector with
    the *second-nearest* site, and near a vertex that bisector can be a sliver that does
    not reach this cell at all -- so the number can be early, and it can never be late.
    Early is also what a plateau wants: the bevel starts a pixel sooner and the slab is a
    slab.
    """
    size, cells = 48, 4
    fac = ground.facets(size=size, cells=cells, seed=3)
    fid, edge = fac["id"], fac["edge"]
    assert len(np.unique(fid)) == cells * cells
    assert float(edge.min()) < 0.6
    assert float(edge.max()) > size / cells * 0.4

    for axis in (0, 1):
        seam = fid != np.roll(fid, 1, axis=axis)
        assert float(edge[seam].max()) <= 1.6, (axis, float(edge[seam].max()))

    diffs = np.array([_nearest_other_cell(fid, y, x, size) - float(edge[y, x])
                      for y in range(size) for x in range(size)])
    assert float(diffs.min()) >= -0.25, float(diffs.min())
    assert float(np.percentile(diffs, 95)) <= 1.8, float(np.percentile(diffs, 95))


def test_a_cell_that_leaves_the_edge_comes_back():
    """The torus, checked at the edge, where it is the only thing that matters.

    A cell may cross the wrap or it may not, and those are the only two answers: either
    the two sides of the seam are the same cell, or the seam *is* a border, in which case
    ``edge`` is small on both sides of it -- and the border field itself has to wrap like
    any other field.  What a Voronoi computed on a square cannot do is put two unrelated
    cells on either side of a seam with no border between them, one row in from the edge,
    which is exactly what makes the crop-and-tile version repeat.
    """
    for cells in (3, 7):
        fac = ground.facets(size=64, cells=cells, seed=5)
        fid, edge = fac["id"], fac["edge"]
        for axis in (0, 1):
            first = fid[0, :] if axis == 0 else fid[:, 0]
            last = fid[-1, :] if axis == 0 else fid[:, -1]
            e_first = edge[0, :] if axis == 0 else edge[:, 0]
            e_last = edge[-1, :] if axis == 0 else edge[:, -1]
            same = first == last
            joint = (e_first <= 1.6) & (e_last <= 1.6)
            assert np.all(same | joint), (cells, axis)
        assert tile.seam_error(edge) <= 1.5, (cells, tile.seam_error(edge))


def test_the_table_is_written_in_pixels_and_the_noise_agrees():
    """``lump_scale`` says 6 and the noise has to be 6 pixels of feature, not of blur.

    ``tile.periodic_noise`` takes a Gaussian *sigma*, and a table in sigmas is a table of
    blurs -- grass at sigma 6 reads as fog, which is what the first version of this sheet
    looked like.  So the conversion is one function, and this is the measurement it is
    written from: the field decorrelates over about half its feature size, which puts a
    lump at somewhere near the number the table asks for.
    """
    for feature in (6.0, 12.0):
        f = ground._noise(128, 5, feature)
        a = f - f.mean()
        ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(a)) ** 2).real)
        ac = ac / ac.max()
        half = int(np.argmax(ac[64, 64:] < 0.5))
        assert 0.35 * feature <= half <= 1.0 * feature, (feature, half)


def test_one_cell_has_no_crack_to_measure():
    """``cells=1`` is a flat plate, and its edge is infinite rather than a distance.

    ``fields`` clips ``edge``, so the plate comes out flat and finite; a caller that
    reads ``edge`` as a distance has to be told what the number means when there is no
    second cell.  Infinite is the answer -- there is no border inside the only cell --
    and it is documented rather than clamped to a plausible-looking half tile.
    """
    fac = ground.facets(size=32, cells=1, seed=3)
    assert np.isinf(fac["edge"]).all()
    assert np.allclose(fac["tone"], fac["tone"][0, 0])
    flat = ground.fields(_ground_row("rock")._replace(facet=1.0, cells=1),
                         seed=1, size=32)
    assert np.isfinite(flat["height"]).all()


def test_one_seed_is_one_tile_and_two_seeds_are_two():
    """Determinism, which for a material is a save file and not a nicety: a map that is
    reopened has to have the same ground under it, and two maps must not be one map.
    """
    for name in ground.names():
        a = _canvas_bytes(ground.texture(name, seed=4, size=64))
        b = _canvas_bytes(ground.texture(name, seed=4, size=64))
        c = _canvas_bytes(ground.texture(name, seed=5, size=64))
        assert a == b, name
        assert a != c, name


def test_zoom_is_a_feature_scale_and_not_a_tile_scale():
    """``zoom=2`` is the table with its feature sizes doubled, and nothing else.

    A site drawn at twice the hex size regenerates the material at twice the texels;
    the features have to double with it or a tuft grows relative to the ground it is
    on.  The test asks the definition directly: doubling the zoom is doubling the two
    feature numbers, with the ribbon switched off so its unstretched cousin cannot
    hide a difference.
    """
    row = _ground_row("rock")._replace(grit=0.0, ribbon=0.0)
    plain = _canvas_bytes(ground.texture(row, seed=3, size=64))
    assert plain == _canvas_bytes(ground.texture(row, seed=3, size=64, zoom=1.0))
    assert plain != _canvas_bytes(ground.texture(row, seed=3, size=64, zoom=2.0))
    doubled = row._replace(lump_scale=row.lump_scale * 2.0)
    assert (_canvas_bytes(ground.texture(row, seed=3, size=64, zoom=2.0))
            == _canvas_bytes(ground.texture(doubled, seed=3, size=64)))


def _atlas_members():
    return [
        ("boards", boards.texture(seed=7, size=64, across=1, down=4, stagger=2)),
        ("ground", wang.texture(seed=7, size=64, cells=8, colours=3)),
        ("stone", truchet.texture(seed=7, size=64, grains=4, kinds=("arcs",))),
    ]


def test_a_margin_is_not_a_gap_it_is_the_tile_continuing():
    """The whole difference between this atlas and the sprite one, in one array.

    A sprite's margin is padding and its job is to be *nothing*, so that a filter
    stepping off the sprite finds no neighbour.  A tile's margin cannot be nothing: the
    tile is opaque, and the margin is exactly what a filtered sample near the edge is
    going to read.  So it is the tile wrapped onto itself -- the row above the first is
    the last row, the column left of the first is the last column, and the corner
    between them is the tile's own far corner.  Nothing in the margin is a pixel the
    tile does not have, which is what makes it material rather than a border.
    """
    stone = truchet.texture(seed=7, size=64, grains=4, kinds=("arcs",))
    raw = stone.to_uint8()
    assert np.array_equal(atlas.bleed(stone, 0), raw)
    block = atlas.bleed(stone, 2)
    assert block.shape == (68, 68, 4)
    assert np.array_equal(block[2:-2, 2:-2], raw)
    n = raw.shape[0]
    for d in (1, 2):
        assert np.array_equal(block[2 - d, 2:-2], raw[n - d])
        assert np.array_equal(block[2:-2, 2 - d], raw[:, n - d])
    assert np.array_equal(block[1, 1], raw[n - 1, n - 1])

    noise = tile.periodic_noise((96, 48), seed=3, scale=6.0)
    wide = Canvas(48, 96, scale=1.0, y_up=False)
    wide.rgb = np.repeat(noise[..., None], 3, axis=2)
    wide.a = np.ones((96, 48), dtype=np.float64)
    assert atlas.bleed(wide, 3).shape == (102, 54, 4)
    assert np.array_equal(atlas.bleed(wide, 3)[3:-3, 3:-3], wide.to_uint8())


def test_a_texture_atlas_moves_tiles_and_does_not_trim_them():
    members = _atlas_members()
    sheet, info = atlas.build(members)
    rgba = sheet.to_uint8()
    assert info["margin"] == atlas.MARGIN
    by_name = {entry["name"]: entry for entry in info["members"]}
    assert set(by_name) == {name for name, _canvas in members}
    for name, canvas in members:
        x, y, w, h = by_name[name]["rect"]
        assert (w, h) == (canvas.w, canvas.h)
        assert np.array_equal(rgba[y:y + h, x:x + w], canvas.to_uint8())

        assert np.array_equal(rgba[y - 1, x - 1], canvas.to_uint8()[-1, -1])


def test_the_margins_of_two_tiles_are_flush_and_do_not_overlap():

    _sheet, info = atlas.build(_atlas_members())
    m = info["margin"]
    blocks = [(x - m, y - m, w + 2 * m, h + 2 * m)
              for x, y, w, h in (entry["rect"] for entry in info["members"])]
    for i, (x0, y0, w0, h0) in enumerate(blocks):
        for x1, y1, w1, h1 in blocks[i + 1:]:
            apart = (x0 + w0 <= x1 or x1 + w1 <= x0
                     or y0 + h0 <= y1 or y1 + h1 <= y0)
            assert apart, (blocks[i], (x1, y1, w1, h1))


def test_a_texture_atlas_is_byte_identical_twice_and_does_not_depend_on_order():
    first, info = atlas.build(_atlas_members())
    again, other_info = atlas.build(_atlas_members())
    assert hashlib.sha256(_canvas_bytes(first)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(again)).hexdigest()
    assert info == other_info
    shuffled, shuffled_info = atlas.build(list(reversed(_atlas_members())))
    assert hashlib.sha256(_canvas_bytes(shuffled)).hexdigest() == \
           hashlib.sha256(_canvas_bytes(first)).hexdigest()
    assert info == shuffled_info


def test_the_density_of_a_sheet_counts_tiles_and_not_their_margins():

    members = _atlas_members()
    content = sum(canvas.w * canvas.h for _name, canvas in members)
    tight, tight_info = atlas.build(members, margin=0, power_of_two=False)
    sheet, info = atlas.build(members)
    for got in (tight_info, info):
        assert got["density"] == round(content / (got["size"][0] * got["size"][1]), 4)
        assert 0.0 < got["density"] < 1.0
    assert info["margin"] == 2 and tight_info["margin"] == 0
    assert info["power_of_two"] and not tight_info["power_of_two"]
    assert info["density"] < tight_info["density"]


def test_a_sheet_that_is_not_a_sheet_is_refused():
    for kwargs in ({"members": []}, {"members": _atlas_members(), "margin": -1}):
        try:
            atlas.build(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs} should not build a sheet")


def _tiled(canvas, n: int = 2) -> Canvas:
    """The tile laid out ``n`` by ``n``: the picture a seamless tile makes."""
    out = Canvas(canvas.w * int(n), canvas.h * int(n), scale=canvas.scale,
                 y_up=canvas.y_up)
    for i in range(int(n)):
        for j in range(int(n)):
            out.blit(canvas.rgb, canvas.a, j * canvas.w, i * canvas.h)
    return out


def test_a_mip_level_is_the_average_of_the_light_and_not_of_the_code():
    """Halving a tile averages in linear light; averaging the bytes would darken it.

    Half the light is not half the code, and this is the one place the difference is
    not academic: a mip chain is nothing *but* averages, so an sRGB average compounds
    -- every level a shade darker than the one above it, which is the classic "the
    mips go muddy" bug and is invisible in any single picture.  A black-and-white
    checker reduced once is the grey that is half the *light*: 188 in a byte, not 128.

    A box filter also preserves the mean, which is the same statement from the other
    side: the level below has the average light of the level above, so a chain of them
    cannot drift.
    """
    checker = Canvas(2, 2, scale=1.0, y_up=False)
    black = palette.srgb_to_linear(np.array([0.0, 0.0, 0.0]))
    white = palette.srgb_to_linear(np.array([1.0, 1.0, 1.0]))
    checker.rgb[0, 0] = black
    checker.rgb[0, 1] = white
    checker.rgb[1, 0] = white
    checker.rgb[1, 1] = black
    checker.a[:] = 1.0
    half = atlas.reduce(checker)
    assert (half.w, half.h) == (1, 1)
    assert np.allclose(half.rgb[0, 0], 0.5, atol=1e-12)
    assert list(half.to_uint8()[0, 0][:3]) == [188, 188, 188]

    rng = np.random.default_rng(4)
    c = Canvas(6, 4, scale=2.0, y_up=False)
    c.rgb = rng.random((4, 6, 3))
    c.a = np.ones((4, 6))
    r = atlas.reduce(c)
    assert (r.w, r.h) == (3, 2)
    for i in range(2):
        for j in range(3):
            want = c.rgb[2 * i:2 * i + 2, 2 * j:2 * j + 2].reshape(-1, 3).mean(axis=0)
            assert np.allclose(r.rgb[i, j], want, atol=1e-12), (i, j)
    assert np.allclose(r.rgb.mean(axis=(0, 1)), c.rgb.mean(axis=(0, 1)), atol=1e-12)
    assert (r.scale, r.y_up, r.center) == (c.scale, c.y_up, c.center)


def test_a_mip_level_averages_a_tile_onto_itself():
    """The reduction commutes with tiling the tile -- which *is* what seamless means.

    A level below the first is a smaller tile, and it is worth having only if it still
    tiles.  The property that says so is exact rather than approximate: reducing the
    tile laid out twice gives the same picture, to the last bit, as laying the reduced
    tile out twice.  A filter that averaged only *inside* the tile would need an edge
    to anchor on and would fail this on the spot, which is the whole reason the
    reduction wraps an odd row rather than reaching for a border.
    """
    canvas = boards.texture(seed=7, size=32, across=1, down=6, stagger=2)
    big = atlas.reduce(_tiled(canvas, 2))
    small = _tiled(atlas.reduce(canvas), 2)
    assert (big.w, big.h) == (small.w, small.h)
    assert np.allclose(big.rgb, small.rgb, atol=1e-12)
    assert np.allclose(big.a, small.a, atol=1e-12)


def test_an_odd_tile_halves_to_the_ceiling_and_invents_nothing():
    """Three texels have no half, so the reduction rounds up and the phase is its own.

    A periodic tile of odd size cannot be halved exactly: its period is an odd number
    of texels and does not divide in two, so *any* reduction is an approximation whose
    phase is off by at most one original texel.  What it must not do is invent a step,
    and it cannot: every output texel is the average of texels that were already
    adjacent, so the reduced tile is never sharper than the tile it came from.
    """
    c = Canvas(3, 1, scale=1.0, y_up=False)
    c.rgb[:, :, 0] = [[0.0, 0.5, 1.0]]
    c.rgb[:, :, 1:] = 0.0
    c.a[:] = 1.0
    r = atlas.reduce(c)
    assert (r.w, r.h) == (2, 1)
    assert np.allclose(r.rgb[0, :, 0], [0.25, 0.5], atol=1e-12)
    steps = np.abs(np.diff(c.rgb[0, :, 0]))
    assert abs(r.rgb[0, -1, 0] - r.rgb[0, 0, 0]) <= steps.max() + 1e-12


def test_a_mip_level_resamples_alpha_premultiplied():
    """A transparent texel has no colour, and it must not contribute black.

    Averaging straight RGB would let a pixel with no colour drag the average towards
    whatever is stored in it, which on the edge of a sprite is a dark halo.  So the
    colour is averaged premultiplied and divided back out: one red texel in a 2x2 is a
    red texel covering a quarter of the area, and it stays red.
    """
    c = Canvas(2, 2, scale=1.0, y_up=False)
    c.rgb[:] = 0.0
    c.rgb[0, 0] = (1.0, 0.0, 0.0)
    c.a[:] = 0.0
    c.a[0, 0] = 1.0
    r = atlas.reduce(c)
    assert np.allclose(r.a[0, 0], 0.25, atol=1e-12)
    assert np.allclose(r.rgb[0, 0], (1.0, 0.0, 0.0), atol=1e-12)

    empty = atlas.reduce(Canvas(4, 4, scale=1.0, y_up=False))
    assert np.all(empty.a == 0.0) and np.all(np.isfinite(empty.rgb))


def test_a_mip_chain_starts_where_the_single_sheet_is():
    """The chain's head is the sheet, not a second opinion about it.

    Level 0 is what :func:`atlas.build` returns, byte for byte and field for field, so
    a single sheet is the head of a chain of one; and each level below it is the level
    above halved, tile size for tile size, which is where the two conventions of
    rounding show up.
    """
    members = _atlas_members()
    sheet, info = atlas.build(members)
    chain = atlas.mips(members, levels=4)
    assert len(chain) == 4
    head, head_info = chain[0]
    assert np.array_equal(head.to_uint8(), sheet.to_uint8())
    assert {k: v for k, v in head_info.items() if k not in ("level", "levels")} == info
    assert (head_info["level"], head_info["levels"]) == (0, 4)

    sizes = [(canvas.w, canvas.h) for canvas, _info in chain]
    assert all(b[0] <= a[0] and b[1] <= a[1] for a, b in zip(sizes, sizes[1:])), sizes
    for k, (_canvas, level) in enumerate(chain):
        assert level["level"] == k
        assert level["margin"] == atlas.MARGIN
        by_name = {entry["name"]: entry for entry in level["members"]}
        for name, member in members:
            want = (member.w, member.h)
            for _ in range(k):
                want = (want[0] // 2 + want[0] % 2, want[1] // 2 + want[1] % 2)
            assert tuple(by_name[name]["source"]) == want, (k, name, want)


def test_a_mip_chain_repacks_because_the_margin_does_not_shrink():
    """Why a chain is a job and not a loop: the margin is measured in *texels*.

    Two texels is what a bilinear filter reaches, at every level, so it is two texels
    of a tile that is now four times smaller -- four times as large a share of it.
    Keeping level 0's rectangles and letting the hardware average would leave each
    level's margin shrinking away exactly as it starts to matter.  The cost of packing
    again is real, and it is in the sidecar instead of in an argument: the coarsest
    level of a chain is very nearly all margin.
    """
    chain = atlas.mips(_atlas_members())
    assert len(chain) == atlas.level_count(_atlas_members())
    shares = []
    for canvas, level in chain:
        assert level["margin"] == atlas.MARGIN
        x, y, w, h = level["members"][0]["rect"]
        assert 0 <= x and 0 <= y and x + w <= canvas.w and y + h <= canvas.h
        shares.append(atlas.MARGIN / min(level["members"][0]["source"]))
    assert shares[0] < shares[-1]
    assert chain[-1][1]["density"] < chain[0][1]["density"]
    assert chain[-1][1]["size"][0] < chain[0][1]["size"][0]


def test_a_mip_level_that_cannot_exist_is_refused():
    """A chain that quietly repeated itself would quietly lie about its own depth.

    Level 8 of a 64-pixel tile is level 6 over again, so the number is not clamped: it
    is refused, and the message says which level the set actually reaches.  The
    default, with no number given, is the longest chain the tiles support.
    """
    members = _atlas_members()
    most = atlas.level_count(members)
    assert most == 7
    assert len(atlas.mips(members)) == most
    for levels in (0, -1):
        try:
            atlas.mips(members, levels=levels)
        except ValueError as e:
            assert "at least one" in str(e), str(e)
        else:
            raise AssertionError(f"a chain of {levels} levels should not exist")
    for levels in (most + 1, 99):
        try:
            atlas.mips(members, levels=levels)
        except ValueError as e:
            assert str(most) in str(e) and "over again" in str(e), str(e)
        else:
            raise AssertionError(f"a chain of {levels} levels should not exist")


def test_the_seam_of_a_tile_survives_the_whole_chain():
    """A mip chain that broke the wrap would break it exactly where the tile is used.

    The reduction is a local average, so it cannot introduce a step the tile did not
    have; the boundary that could still go wrong is the wrap, which is the one part of
    a tile that is not local.  So the number is measured at every level of the chain,
    including the one-texel tile at the bottom, which has no boundary inside it at all
    and reads zero.  A wrapping field reads about 1 and a cliff reads tens; this
    asserts the chain stays in the first reading all the way down.
    """
    worst = 0.0
    for name, canvas in _atlas_members():
        current = canvas
        for _level in range(atlas.level_count([(name, canvas)])):
            seam = tile.seam_error(tile.luminance(current.to_uint8()))
            worst = max(worst, seam)
            assert seam <= 4.0, (name, seam)
            current = atlas.reduce(current)
        assert (current.w, current.h) == (1, 1)
    assert worst > 0.0

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception:
            failures += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
