"""Regression tests for the map generator.

Five claims are worth more than the rest, and each has a test that could fail
loudly:

* the hexagon the rasteriser thinks a pixel is in is the hexagon it really is in
  -- checked against :func:`core.sdf.polygon` of the true corners, to machine
  precision, which is the only check that catches a bad lattice constant;
* a map is a function of ``(seed, cols, rows)`` and not of its output resolution,
  because the terrain is evaluated in normalised map coordinates -- so the same
  seed at two hex sizes must give the *same* elevation field;
* the coastline walk closes and puts the land on its left, including around a
  hole, where a walk that only followed the outside would produce one loop
  instead of two;
* filling a pit raises a cell to the lowest height at which water can get from it
  to the sea -- a minimax answer, checked against brute-force relaxation rather
  than against the shape of the priority queue that produced it;
* a province is connected.  That is the claim the whole of ``region`` exists for,
  so it is checked by flooding each province on its own, and also against the
  nearest-capital property the breadth-first assignment is supposed to give.

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

from core import palette, sdf  # noqa: E402
from core.canvas import Canvas  # noqa: E402
from gen.maps import coast, grid as gridlib, paint, region, rivers, terrain  # noqa: E402


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
    # Terrain only on purpose: the claim is about the ground and the water, and a
    # river is a blue line drawn on top of perfectly good grass.
    img = paint.map_image(seed=3, cols=20, rows=14, size=9.0, river=0.0, provinces=0)

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


# -------------------------------------------------------------------- rivers
def _flow(cols=30, rows=20, size=13.0, seed=7):
    """A grid and its fields, plus everything the water does on them."""
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed)
    filled = rivers.fill_depressions(g, f["elev"], f["water"])
    down = rivers.downstream(g, filled, f["water"])
    flux = rivers.drainage(g, filled, down)
    return g, f, filled, down, flux


def test_the_flood_raises_a_cell_to_its_lowest_way_out():
    """Filling a pit is a minimax problem, and this is the minimax answer.

    A cell's filled height has to be the lowest height at which water can get from
    it to the sea: the smallest possible maximum along a path.  Brute-force
    relaxation gives that number on a small grid without knowing anything about
    priority queues, so this is a check of the algorithm and not of its shape.
    """
    g = gridlib.HexGrid(7, 7, size=10.0)
    rr, cc = np.meshgrid(np.arange(7), np.arange(7), indexing="ij")
    elev = 0.55 + 0.30 * np.sin(rr * 0.9) * np.cos(cc * 0.7)
    water = np.zeros((7, 7), dtype=bool)
    water[:, 0] = True
    filled = rivers.fill_depressions(g, elev, water)

    best = np.where(water, elev, np.inf)
    for _ in range(60):
        changed = False
        for row, col in g.flat():
            row, col = int(row), int(col)
            for d in range(6):
                nr, nc = (int(v) for v in g.nbr[row, col, d])
                if nr < 0:
                    continue
                cand = max(elev[row, col], best[nr, nc])
                if cand < best[row, col] - 1e-12:
                    best[row, col] = cand
                    changed = True
        if not changed:
            break
    # Up to the hair the flood adds at each step, which is deliberately not zero.
    assert np.allclose(filled, np.maximum(elev, best), atol=1e-5)


def test_filling_never_lowers_the_ground_and_never_leaves_a_sump():
    for seed in (7, 3, 23, 19):
        g, f, filled, down, flux = _flow(seed=seed)
        land = ~f["water"]
        assert float((filled - f["elev"]).min()) >= 0.0
        # After the flood every land cell has somewhere lower to go.  Without that
        # guarantee a river would simply stop in the middle of a continent.
        assert all(int(down[r, c]) >= 0 for r, c in g.flat() if land[r, c])


def test_the_flood_leaves_almost_all_of_the_map_where_it_was():
    # A flood that raised the whole continent would be correct and useless: the
    # land would sit flat at the height of a mountain pass and the rivers would run
    # along the floor of a lake.  On real terrain only the genuine basins move.
    for seed in (7, 3, 23):
        g, f, filled, down, flux = _flow(seed=seed)
        land = ~f["water"]
        lifted = int(((filled - f["elev"])[land] > 1e-9).sum())
        assert lifted < 0.35 * int(land.sum()), (seed, lifted)


def test_the_sea_carries_the_whole_map():
    # Every cell contributes one unit of drainage and nothing is created or lost on
    # the way down; the sea does not drain anywhere itself, so its units stay where
    # they fell.  The total standing in the sea is therefore every cell on the map
    # -- and it would be less than that if any land cell were a sump, which is what
    # makes this a test of the whole flow network rather than of the arithmetic.
    g, f, filled, down, flux = _flow()
    assert int(flux[f["water"]].sum()) == g.count
    assert int(flux[f["water"]].sum()) - int(f["water"].sum()) == int((~f["water"]).sum())


def test_a_river_ends_on_land_that_drains_into_the_sea():
    g, f, filled, down, flux = _flow()
    courses = rivers.courses(g, f, down, flux)
    assert courses
    for course in courses:
        cell = g.cell(np.array([course["mouth"][0]]), np.array([course["mouth"][1]]))
        r, c = (int(v) for v in cell.rc[0])
        assert not f["water"][r, c]
        d = int(down[r, c])
        assert d >= 0
        dr, dc = divmod(d, g.cols)
        assert f["water"][dr, dc]
        assert len(course["points"]) >= 2
        assert course["cells"] >= 3


def test_the_river_density_is_what_was_asked_for_and_not_a_lucky_constant():
    for seed in (7, 3, 23, 19):
        g, f, filled, down, flux = _flow(seed=seed)
        land = int((~f["water"]).sum())
        for frac in (0.03, 0.06, 0.10):
            got = int(rivers.river_mask(flux, f["water"], frac).sum()) / land
            assert abs(got - frac) < 0.02, (seed, frac, got)


def test_asking_for_more_rivers_only_adds_cells_to_the_network():
    # The threshold is a quantile of the same field, so the smaller network has to
    # be a subset of the larger one -- if it were not, "a fraction of the land"
    # would not be a fraction of anything.
    g, f, filled, down, flux = _flow()
    small = rivers.river_mask(flux, f["water"], 0.03)
    big = rivers.river_mask(flux, f["water"], 0.10)
    assert 0 < int(small.sum()) < int(big.sum())
    assert int((small & ~big).sum()) == 0


def test_the_same_seed_drains_the_same_way():
    a = _flow(seed=11)
    b = _flow(seed=11)
    assert np.array_equal(a[3], b[3])
    assert np.array_equal(a[4], b[4])
    ca = rivers.courses(a[0], a[1], a[3], a[4])
    cb = rivers.courses(b[0], b[1], b[3], b[4])
    assert len(ca) == len(cb) > 0
    for x, y in zip(ca, cb):
        assert np.array_equal(x["points"], y["points"])
        assert x["flux"] == y["flux"]


# ----------------------------------------------------------------- provinces
def _political(cols=30, rows=20, size=13.0, seed=7, count=8, province_seed=None):
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed)
    s = seed if province_seed is None else province_seed
    return g, f, region.provinces(g, f, count=count, seed=s)


def test_every_province_is_connected():
    # The claim the whole module exists for.  A province in two pieces cannot be
    # governed, because an army cannot get from one half to the other without
    # leaving it -- and nearest-capital by straight-line distance produces exactly
    # that, which is why the assignment is a breadth-first search instead.
    for seed in (7, 3, 23, 19):
        g, f, res = _political(seed=seed)
        assert res["count"] >= 1
        for p in res["provinces"]:
            mask = np.zeros(g.count, dtype=bool)
            mask[p["cells"]] = True
            parts = region.components(g, mask.reshape(g.rows, g.cols))
            assert len(parts) == 1, (seed, p["name"], len(parts))
            assert int(parts[0].size) == p["area"]


def test_the_land_is_covered_once_and_the_sea_is_not_covered_at_all():
    g, f, res = _political()
    labels = res["labels"]
    assert np.array_equal(labels >= 0, ~f["water"])
    assert int(labels.max()) + 1 == res["count"]
    used = sorted({int(v) for v in labels.reshape(-1)})
    assert used == [-1] + list(range(res["count"]))  # and no province came out empty


def test_a_province_holds_the_cells_whose_capital_is_nearest_by_path():
    # Multi-source breadth-first search grows every capital at once, so a cell can
    # only be claimed by a capital at least as close as any other -- and that is a
    # property of the finished division, not of the way it was built.
    g, f, res = _political()
    flat = (~f["water"]).reshape(-1)
    labels = res["labels"].reshape(-1)
    dist = {p["index"]: region._masked_bfs(
        g, flat, int(p["capital"][0]) * g.cols + int(p["capital"][1]))
        for p in res["provinces"]}
    for i in range(res["count"]):
        for j in range(res["count"]):
            if i == j:
                continue
            cells = labels == i
            assert np.all(dist[i][cells] <= dist[j][cells]), (i, j)


def test_a_capital_is_within_a_hex_of_the_middle_of_its_province():
    # Lloyd's step moves a capital to the one-median of its province.  Two passes
    # do not always converge to it -- the division is re-cut between passes -- but
    # they always land within one hex, and the capital's total distance is within
    # ten per cent of the best a cell of that province could do.
    g, f, res = _political()
    flat = (~f["water"]).reshape(-1)
    for p in res["provinces"]:
        median = region.one_median(g, flat, p["cells"])
        cap = int(p["capital"][0]) * g.cols + int(p["capital"][1])
        dcap = region._masked_bfs(g, flat, cap)
        dmed = region._masked_bfs(g, flat, median)
        total_cap = int(sum(dcap[j] for j in p["cells"]))
        total_med = int(sum(dmed[j] for j in p["cells"]))
        assert total_cap <= 1.1 * total_med, (p["name"], total_cap, total_med)
        rcap, ccap = divmod(cap, g.cols)
        rmed, cmed = divmod(median, g.cols)
        a = g.axes[rcap, ccap]
        b = g.axes[rmed, cmed]
        assert gridlib.distance((int(a[0]), int(a[1])), (int(b[0]), int(b[1]))) <= 1


def test_no_two_provinces_on_a_map_share_a_name():
    for seed in (7, 3, 23):
        g, f, res = _political(seed=seed)
        names = [p["name"] for p in res["provinces"]]
        assert len(set(names)) == len(names)
        assert all(n and n[0].isupper() for n in names)


def test_a_lone_island_becomes_a_province_of_its_own():
    # A landmass no capital can reach is a province, with its capital on it.  The
    # alternative -- reaching it by distance across the sea -- would produce a
    # province in two pieces, which is the thing this module exists to prevent.
    g = gridlib.HexGrid(9, 7, size=10.0)
    water = np.ones((7, 9), dtype=bool)
    water[:, :4] = False
    water[3, 6] = False
    f = {"water": water, "biome": np.zeros((7, 9), dtype=np.int64),
         "near_sea": np.ones((7, 9), dtype=bool), "elev": np.zeros((7, 9)),
         "height": np.zeros((7, 9))}
    res = region.provinces(g, f, count=4, seed=1)
    assert res["islands"] == 1
    lonely = [p for p in res["provinces"] if p["area"] == 1]
    assert len(lonely) == 1
    assert lonely[0]["capital"] == (3, 6)
    for p in res["provinces"]:
        if p["index"] == lonely[0]["index"]:
            assert int(p["cells"][0]) == 3 * g.cols + 6
        else:
            assert all(int(j) % g.cols < 4 for j in p["cells"])


def test_no_province_ever_spans_a_channel_of_water():
    g = gridlib.HexGrid(9, 7, size=10.0)
    water = np.zeros((7, 9), dtype=bool)
    water[:, 4] = True
    f = {"water": water, "biome": np.zeros((7, 9), dtype=np.int64),
         "near_sea": water.copy(), "elev": np.zeros((7, 9)), "height": np.zeros((7, 9))}
    res = region.provinces(g, f, count=4, seed=3)
    sides = set()
    for p in res["provinces"]:
        cols = {int(j) % g.cols for j in p["cells"]}
        assert not ({3} <= cols and {5} <= cols), p["name"]
        sides.add(0 if max(cols) < 4 else 1)
    assert sides == {0, 1}  # and both sides did get provinces


def test_a_second_seed_divides_the_same_land_differently():
    g = gridlib.HexGrid(24, 16, size=11.0)
    f = terrain.fields(g, seed=7)
    a = region.provinces(g, f, count=6, seed=1)
    b = region.provinces(g, f, count=6, seed=2)
    assert np.array_equal(a["labels"] >= 0, b["labels"] >= 0)   # the same land
    assert not np.array_equal(a["labels"], b["labels"])          # cut differently
    assert {p["name"] for p in a["provinces"]} != {p["name"] for p in b["provinces"]}


def test_province_tints_stay_far_apart_however_many_are_asked_for():
    for n in (3, 6, 11):
        tints = paint.province_tints(n)
        assert tints.shape == (n, 3)
        assert float(tints.min()) >= 0.0 and float(tints.max()) <= 1.0
        for i in range(n):
            for j in range(i + 1, n):
                assert palette.delta_e(tints[i], tints[j]) > 0.04, (n, i, j)


def test_a_frontier_is_a_line_between_two_provinces_and_not_a_wash():
    g = gridlib.HexGrid(12, 9, size=10.0)
    labels = np.full((9, 12), -1, dtype=np.int64)
    labels[:, :6] = 0
    labels[:, 6:] = 1
    w, h = g.extent
    canvas = Canvas(int(np.ceil(w)) + 8, int(np.ceil(h)) + 8, scale=1.0)
    cell = g.cell(*canvas.grid())
    a = region.frontier_alpha(g, labels, cell, dx=canvas.dx)
    assert 0.0 < float(a.max()) <= 1.0
    # Thin: nothing survives more than a couple of pixels inside a hex, so the
    # frontier is a line on the shared edge and not a shaded band.
    assert float(a[cell.edge_distance() > 2.0].max()) == 0.0


def test_a_map_with_provinces_and_rivers_paints_the_same_bytes_twice():
    a = paint.map_image(seed=5, cols=20, rows=14, size=10.0, provinces=6)
    b = paint.map_image(seed=5, cols=20, rows=14, size=10.0, provinces=6)
    assert hashlib.sha256(a.to_uint8().tobytes()).hexdigest() == \
           hashlib.sha256(b.to_uint8().tobytes()).hexdigest()
    c = paint.map_image(seed=5, cols=20, rows=14, size=10.0, provinces=0, river=0.0)
    assert not np.array_equal(a.to_uint8(), c.to_uint8())


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
