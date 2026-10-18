"""Regression tests for the map generator.

Three claims are worth more than the rest, and each has a test that could fail
loudly:

* the hexagon the rasteriser thinks a pixel is in is the hexagon it really is in
  -- checked against :func:`core.sdf.polygon` of the true corners, to machine
  precision, which is the only check that catches a bad lattice constant;
* a map is a function of ``(seed, cols, rows)`` and not of its output resolution,
  because the terrain is evaluated in normalised map coordinates -- so the same
  seed at two hex sizes must give the *same* elevation field;
* the coastline walk closes and puts the land on its left, including around a
  hole, where a walk that only followed the outside would produce one loop
  instead of two.

Run either way::

    python tests/test_maps.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import sdf  # noqa: E402
from core.canvas import Canvas  # noqa: E402
from gen.maps import coast, grid as gridlib, paint, terrain  # noqa: E402


# ------------------------------------------------------------------- lattice
def test_the_rectangle_of_hexes_is_a_rectangle():
    # The storage is offset coordinates, the maths is axial: the two have to agree
    # about which hex is in row 3, column 0, or every neighbour is wrong.
    g = gridlib.HexGrid(7, 5, size=10.0)
    for row in range(g.rows):
        for col in range(g.cols):
            q, r = (int(v) for v in g.axes[row, col])
            assert r == row
            assert col == q + (r - (r & 1)) // 2
            assert g.rc_of(q, r) == (row, col)


def test_a_hex_has_six_distinct_neighbours_one_step_away():
    g = gridlib.HexGrid(9, 7, size=10.0)
    for row, col in g.flat():
        q, r = (int(v) for v in g.axes[row, col])
        nb = gridlib.neighbours(q, r)
        assert len(set(nb)) == 6
        assert all(gridlib.distance((q, r), n) == 1 for n in nb)
        # Every neighbour of an interior hex is on the map, and it is the same
        # hex the neighbour table names.
        assert all(n is not None for n in nb)
        for d in range(6):
            nq, nr = gridlib.neighbour(q, r, d)
            want = g.rc_of(nq, nr)
            got = tuple(int(v) for v in g.nbr[row, col, d])
            assert got == (want if want is not None else (-1, -1))


def test_the_neighbour_table_is_symmetric():
    # If B is A's third neighbour, then A is one of B's six: a table that breaks
    # this would make rivers flow one way and provinces adjacent the other.
    g = gridlib.HexGrid(8, 6, size=10.0)
    for row, col in g.flat():
        for d in range(6):
            nr, nc = (int(v) for v in g.nbr[row, col, d])
            if nr < 0:
                continue
            back = [(int(a), int(b)) for a, b in g.nbr[nr, nc]]
            assert (int(row), int(col)) in back


def test_every_centre_lands_in_its_own_hex():
    g = gridlib.HexGrid(11, 8, size=10.0)
    cell = g.cell(g.centers[..., 0], g.centers[..., 1])
    rows, cols = np.mgrid[0:g.rows, 0:g.cols]
    assert cell.ok.all()
    assert np.array_equal(cell.rc[..., 0], rows)
    assert np.array_equal(cell.rc[..., 1], cols)
    assert float(cell.rho.max()) < 1e-9


def test_between_two_centres_the_pixel_belongs_to_one_of_them():
    g = gridlib.HexGrid(9, 6, size=12.0)
    mid = 0.5 * (g.centers[:, :-1] + g.centers[:, 1:])
    cell = g.cell(mid[..., 0], mid[..., 1])
    assert cell.ok.all()
    for col in range(g.cols - 1):
        got = cell.rc[:, col]
        assert np.array_equal(got[..., 0], np.arange(g.rows))
        assert np.logical_or(got[..., 1] == col, got[..., 1] == col + 1).all()


def test_the_hexagon_the_rasteriser_believes_in_is_the_hexagon():
    # The strongest statement in the package: the per-pixel signed distance to
    # the pixel's hexagon, computed from the sector alone, must *equal* the
    # distance to the true polygon.  Measured on the fields, not on a bounding
    # box, because a bounding box passes while the shape is wrong.
    g = gridlib.HexGrid(5, 4, size=10.0)
    c = Canvas(140, 100, scale=1.0)
    X, Y = c.grid()
    cell = g.cell(X, Y)
    d = cell.edge_distance()
    worst = 0.0
    for row in range(g.rows):
        for col in range(g.cols):
            m = cell.ok & (cell.rc[..., 0] == row) & (cell.rc[..., 1] == col)
            if not m.any():
                continue
            # sdf.polygon is negative inside; edge_distance is positive inside.
            worst = max(worst, float(np.abs(d[m] + sdf.polygon(X, Y, g.corners_of(row, col))[m]).max()))
    assert worst < 1e-9, worst


def test_a_corner_has_one_name_whichever_hex_asks():
    g = gridlib.HexGrid(7, 5, size=9.0)
    for row, col in g.flat():
        q, r = (int(v) for v in g.axes[row, col])
        pts = g.corners_of(int(row), int(col))
        for k in range(6):
            key = gridlib.corner_key(q, r, k)
            assert gridlib.corner_key(*key) == key  # idempotent
            # The two other hexes at this corner, named through the edges that
            # end here, must resolve to the same key and the same point.
            for d in ((k - 1) % 6, k):
                nq, nr = gridlib.neighbour(q, r, d)
                kk = (4 + 2 * d - k) % 6
                assert gridlib.corner_key(nq, nr, kk) == key
                nx, ny = gridlib.corners(nq, nr, g.size, g.shift)[kk]
                assert abs(nx - pts[k][0]) < 1e-9 and abs(ny - pts[k][1]) < 1e-9


def test_points_off_the_map_are_flagged_rather_than_wrapped():
    g = gridlib.HexGrid(6, 4, size=10.0)
    far = g.cell(np.array([1e4, 0.0]), np.array([1e4, -1e4]))
    assert not far.ok.any()


def test_a_degenerate_grid_is_refused():
    for cols, rows in ((0, 4), (4, 0), (-1, 2)):
        try:
            gridlib.HexGrid(cols, rows, size=10.0)
        except ValueError:
            continue
        raise AssertionError(f"HexGrid accepted {cols}x{rows}")


# ------------------------------------------------------------------- terrain
def test_the_sea_is_the_fraction_that_was_asked_for():
    # The sea level is a quantile, so "35% water" is a property of the map rather
    # than a number that happens to look right for one seed.
    g = gridlib.HexGrid(24, 16, size=8.0)
    for frac in (0.25, 0.35, 0.50, 0.65):
        f = terrain.fields(g, seed=5, sea_fraction=frac)
        got = float(f["water"].mean())
        assert abs(got - frac) <= 1.5 / g.count, (frac, got)


def test_the_geography_does_not_depend_on_the_pixel_size():
    # A map is (seed, cols, rows); `size` only decides how many pixels it is
    # drawn in.  The elevation is evaluated in normalised coordinates, so the two
    # fields agree to rounding -- and because the biomes are thresholds on that
    # field, a perturbation in the fifteenth digit cannot move one either.
    a = terrain.fields(gridlib.HexGrid(20, 13, size=6.0), seed=3)
    b = terrain.fields(gridlib.HexGrid(20, 13, size=37.0), seed=3)
    assert np.allclose(a["elev"], b["elev"], atol=1e-12)
    assert np.allclose(a["moist"], b["moist"], atol=1e-12)
    assert np.array_equal(a["biome"], b["biome"])
    # The sea level is a quantile of the same field, so it agrees to rounding
    # rather than exactly: a quantile interpolates between two cells.
    assert abs(a["sea"] - b["sea"]) < 1e-12


def test_a_map_is_the_same_map_every_time_and_a_different_one_next_seed():
    g = gridlib.HexGrid(20, 13, size=6.0)
    a = terrain.fields(g, seed=1)
    b = terrain.fields(g, seed=1)
    c = terrain.fields(g, seed=2)
    assert np.array_equal(a["elev"], b["elev"])
    assert np.array_equal(a["biome"], b["biome"])
    assert not np.allclose(a["elev"], c["elev"])
    # Not merely perturbed: a different seed should move the coastline.
    assert float((a["water"] != c["water"]).mean()) > 0.10


def test_the_elevation_is_normalised_and_smoothed_over_the_hexes():
    g = gridlib.HexGrid(18, 12, size=6.0)
    f = terrain.fields(g, seed=9, smooth_passes=0)
    assert 0.0 <= float(f["elev"].min()) and float(f["elev"].max()) <= 1.0
    assert abs(float(f["elev"].max()) - 1.0) < 1e-9
    assert abs(float(f["elev"].min())) < 1e-9
    # Smoothing is a contraction: it cannot increase the spread of the field.
    s = terrain.smooth(f["elev"], g, passes=2)
    assert float(s.std()) <= float(f["elev"].std()) + 1e-12
    assert np.array_equal(terrain.smooth(f["elev"], g, passes=0), f["elev"])


def test_every_biome_cell_agrees_with_the_water_mask():
    g = gridlib.HexGrid(22, 15, size=7.0)
    f = terrain.fields(g, seed=4)
    codes = f["biome"]
    water = np.array([b.water for b in terrain.BIOMES])[codes]
    assert np.array_equal(water, f["water"])
    # A shallow hex is water that touches land; a shore hex is land that touches
    # water.  Both are adjacency, and both must be true of the cell.
    shallow = codes == terrain.INDEX["shallow"]
    assert (shallow & ~f["near_land"]).sum() == 0
    shore = codes == terrain.INDEX["shore"]
    assert (shore & ~f["near_sea"]).sum() == 0
    assert (shore & f["water"]).sum() == 0
    # A deep cell is deep by depth, whatever it is next to.
    deep = codes == terrain.INDEX["deep"]
    assert bool((f["elev"][deep] < f["sea"] - 0.13).all()) if deep.any() else True


def test_the_stats_add_up_to_the_map():
    g = gridlib.HexGrid(16, 11, size=6.0)
    f = terrain.fields(g, seed=12)
    st = terrain.stats(f["biome"])
    assert st["hexes"] == g.count
    assert abs(st["water"] - float(f["water"].mean())) < 1e-9
    named = sum(v for k, v in st.items() if k not in ("hexes", "water"))
    assert named == g.count


def test_every_biome_in_the_table_is_actually_reachable():
    # A biome nobody can roll is a colour nobody sees, and a movement cost nobody
    # pays.  Across a handful of seeds every one of them should appear.
    seen = set()
    for seed in range(12):
        g = gridlib.HexGrid(30, 20, size=6.0)
        seen.update(np.unique(terrain.fields(g, seed=seed)["biome"]).tolist())
    missing = [b.key for i, b in enumerate(terrain.BIOMES) if i not in seen]
    assert not missing, missing


# ------------------------------------------------------------------ coastline
def test_a_single_hex_has_a_six_corner_coast():
    g = gridlib.HexGrid(5, 5, size=10.0)
    mask = np.zeros((g.rows, g.cols), dtype=bool)
    mask[2, 2] = True
    loops = coast.boundary_loops(g, mask)
    assert len(loops) == 1 and len(loops[0]) == 6
    assert coast.signed_area(loops[0]) > 0.0  # the land is on the left


def test_a_block_of_hexes_has_one_loop_and_a_hole_makes_two():
    g = gridlib.HexGrid(7, 7, size=10.0)
    mask = np.zeros((g.rows, g.cols), dtype=bool)
    mask[2:5, 2:5] = True
    outer = coast.boundary_loops(g, mask)
    assert len(outer) == 1
    assert coast.signed_area(outer[0]) > 0.0
    # Punch a hole in it: the boundary is now the rim and the rim of the hole,
    # and the hole is wound the other way -- which is the whole reason the walk
    # carries a direction.
    mask[3, 3] = False
    loops = coast.boundary_loops(g, mask)
    assert len(loops) == 2
    areas = sorted(coast.signed_area(p) for p in loops)
    assert areas[0] < 0.0 < areas[1]


def test_separate_islands_give_separate_loops():
    g = gridlib.HexGrid(9, 7, size=10.0)
    mask = np.zeros((g.rows, g.cols), dtype=bool)
    mask[1, 1] = True
    mask[5, 5] = True
    loops = coast.boundary_loops(g, mask)
    assert len(loops) == 2
    assert all(len(p) == 6 for p in loops)


def test_the_whole_block_is_the_map_frame():
    g = gridlib.HexGrid(6, 4, size=10.0)
    loops = coast.boundary_loops(g, np.ones((g.rows, g.cols), dtype=bool))
    assert len(loops) == 1
    # The frame's length is the number of edges with nothing on the other side.
    edges = 0
    for row, col in g.flat():
        for d in range(6):
            if g.nbr[row, col, d, 0] < 0:
                edges += 1
    assert len(loops[0]) == edges


def test_an_empty_mask_has_no_coast():
    g = gridlib.HexGrid(5, 5, size=10.0)
    assert coast.boundary_loops(g, np.zeros((g.rows, g.cols), dtype=bool)) == []


def test_every_boundary_edge_is_walked_exactly_once():
    # A real coastline: every corner on the boundary is visited once and every
    # loop closes, so the loop lengths must account for every half-edge.
    g = gridlib.HexGrid(24, 16, size=6.0)
    f = terrain.fields(g, seed=8)
    land = ~f["water"]
    loops = coast.boundary_loops(g, land)
    half_edges = 0
    for row, col in g.flat():
        if not land[row, col]:
            continue
        for d in range(6):
            nr, nc = (int(v) for v in g.nbr[row, col, d])
            if nr < 0 or not land[nr, nc]:
                half_edges += 1
    assert sum(len(p) for p in loops) == half_edges
    assert len(loops) >= 1
    # An island in an ocean: the rims go one way and the lakes the other.
    assert sum(1 for p in loops if coast.signed_area(p) < 0) < len(loops)


def test_a_mask_of_the_wrong_shape_is_refused():
    g = gridlib.HexGrid(5, 5, size=10.0)
    try:
        coast.boundary_loops(g, np.zeros((5, 4), dtype=bool))
    except ValueError:
        return
    raise AssertionError("boundary_loops accepted a mis-shaped mask")


# ------------------------------------------------------------------- picture
def test_a_rendered_map_is_a_pure_function_of_its_seed():
    a = paint.map_image(seed=6, cols=16, rows=11, size=9.0)
    b = paint.map_image(seed=6, cols=16, rows=11, size=9.0)
    assert hashlib.sha256(a.to_uint8().tobytes()).digest() == hashlib.sha256(b.to_uint8().tobytes()).digest()
    c = paint.map_image(seed=7, cols=16, rows=11, size=9.0)
    assert not np.array_equal(a.to_uint8(), c.to_uint8())


def test_the_block_is_opaque_and_the_corner_is_empty():
    img = paint.map_image(seed=2, cols=12, rows=8, size=10.0)
    c = Canvas(img.w, img.h, scale=1.0)
    cell = None
    from gen.maps import HexGrid  # local: the picture, not the lattice, is under test
    g = HexGrid(12, 8, 10.0)
    cell = g.cell(*c.grid())
    assert float(img.a[cell.ok].min()) > 0.99
    assert float(img.a[0, 0]) == 0.0
    assert float(img.a[-1, -1]) == 0.0


def test_land_and_sea_are_painted_in_their_own_colours():
    # Sampling means would be meaningless here -- the shelf is pale and the forest
    # is dark, so the two families overlap.  What has to hold is the thing a
    # player reads off the screen: water is blue-dominant and grass is green.
    from gen.maps import HexGrid
    g = HexGrid(20, 14, size=9.0)
    f = terrain.fields(g, seed=3)
    img = paint.map_image(seed=3, cols=20, rows=14, size=9.0)

    def pixel_at(x, y):
        col = int(round(x * img.scale + img.w / 2.0 - 0.5))
        row = int(round(img.h / 2.0 - 0.5 - y * img.scale))
        return img.rgb[row, col]

    codes = f["biome"]
    blue = (terrain.INDEX["ocean"], terrain.INDEX["deep"], terrain.INDEX["shallow"])
    green = (terrain.INDEX["forest"], terrain.INDEX["plains"], terrain.INDEX["marsh"])
    wet = [(r, c) for r, c in g.flat() if codes[r, c] in blue]
    dry = [(r, c) for r, c in g.flat() if codes[r, c] in green]
    assert wet and dry
    for r, c in wet:
        px = pixel_at(float(g.centers[r, c, 0]), float(g.centers[r, c, 1]))
        assert px[2] > px[0], (r, c, px)  # more blue than red
    for r, c in dry:
        px = pixel_at(float(g.centers[r, c, 0]), float(g.centers[r, c, 1]))
        assert px[1] > px[2], (r, c, px)  # more green than blue


def test_every_seed_paints_a_map_with_a_coast_on_it():
    for seed in (0, 4, 19):
        img = paint.map_image(seed=seed, cols=18, rows=12, size=9.0)
        px = img.to_uint8()
        colours = len(np.unique(px[..., :3].reshape(-1, 3), axis=0))
        assert colours > 20, (seed, colours)
        assert 0.05 < float((px[..., 3] > 0).mean()) < 0.98


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
