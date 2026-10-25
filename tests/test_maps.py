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
  nearest-capital property the breadth-first assignment is supposed to give;
* the coast as tiles is the coast as a stroke.  A border is now six bits per hex
  and a tileset, which is a rewrite of the picture rather than a feature beside
  it, so the test that it is the *same* picture inside the map -- and that the one
  place it differs is off the edge of the map -- is the reason to trust the rest.
* a province's name is set inside its own province.  A label is the one thing on a
  map that fails by being *nearly* right -- a name that reaches over a frontier
  reads as the neighbour's -- so the anchor is checked against brute-force hex
  distance and the box against a lattice finer than the ink it stands for.

Run either way::

    python tests/test_maps.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import palette, sdf
from core.canvas import Canvas
from gen.maps import (autotile, campaign, coast, grid as gridlib,
                      iso as iso_render, labels as label_lib, paint, politics,
                      region, rivers, roads, site as site_render, terrain)


def test_the_rectangle_of_hexes_is_a_rectangle():

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

        assert all(n is not None for n in nb)
        for d in range(6):
            nq, nr = gridlib.neighbour(q, r, d)
            want = g.rc_of(nq, nr)
            got = tuple(int(v) for v in g.nbr[row, col, d])
            assert got == (want if want is not None else (-1, -1))


def test_the_neighbour_table_is_symmetric():

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

            worst = max(worst, float(np.abs(d[m] + sdf.polygon(X, Y, g.corners_of(row, col))[m]).max()))
    assert worst < 1e-9, worst


def test_a_corner_has_one_name_whichever_hex_asks():
    g = gridlib.HexGrid(7, 5, size=9.0)
    for row, col in g.flat():
        q, r = (int(v) for v in g.axes[row, col])
        pts = g.corners_of(int(row), int(col))
        for k in range(6):
            key = gridlib.corner_key(q, r, k)
            assert gridlib.corner_key(*key) == key

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


def test_the_sea_is_the_fraction_that_was_asked_for():

    g = gridlib.HexGrid(24, 16, size=8.0)
    for frac in (0.25, 0.35, 0.50, 0.65):
        f = terrain.fields(g, seed=5, sea_fraction=frac)
        got = float(f["water"].mean())
        assert abs(got - frac) <= 1.5 / g.count, (frac, got)


def test_the_geography_does_not_depend_on_the_pixel_size():

    a = terrain.fields(gridlib.HexGrid(20, 13, size=6.0), seed=3)
    b = terrain.fields(gridlib.HexGrid(20, 13, size=37.0), seed=3)
    assert np.allclose(a["elev"], b["elev"], atol=1e-12)
    assert np.allclose(a["moist"], b["moist"], atol=1e-12)
    assert np.array_equal(a["biome"], b["biome"])

    assert abs(a["sea"] - b["sea"]) < 1e-12


def test_a_map_is_the_same_map_every_time_and_a_different_one_next_seed():
    g = gridlib.HexGrid(20, 13, size=6.0)
    a = terrain.fields(g, seed=1)
    b = terrain.fields(g, seed=1)
    c = terrain.fields(g, seed=2)
    assert np.array_equal(a["elev"], b["elev"])
    assert np.array_equal(a["biome"], b["biome"])
    assert not np.allclose(a["elev"], c["elev"])

    assert float((a["water"] != c["water"]).mean()) > 0.10


def test_the_elevation_is_normalised_and_smoothed_over_the_hexes():
    g = gridlib.HexGrid(18, 12, size=6.0)
    f = terrain.fields(g, seed=9, smooth_passes=0)
    assert 0.0 <= float(f["elev"].min()) and float(f["elev"].max()) <= 1.0
    assert abs(float(f["elev"].max()) - 1.0) < 1e-9
    assert abs(float(f["elev"].min())) < 1e-9

    s = terrain.smooth(f["elev"], g, passes=2)
    assert float(s.std()) <= float(f["elev"].std()) + 1e-12
    assert np.array_equal(terrain.smooth(f["elev"], g, passes=0), f["elev"])


def test_every_biome_cell_agrees_with_the_water_mask():
    g = gridlib.HexGrid(22, 15, size=7.0)
    f = terrain.fields(g, seed=4)
    codes = f["biome"]
    water = np.array([b.water for b in terrain.BIOMES])[codes]
    assert np.array_equal(water, f["water"])

    shallow = codes == terrain.INDEX["shallow"]
    assert (shallow & ~f["near_land"]).sum() == 0
    shore = codes == terrain.INDEX["shore"]
    assert (shore & ~f["near_sea"]).sum() == 0
    assert (shore & f["water"]).sum() == 0

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

    seen = set()
    for seed in range(12):
        g = gridlib.HexGrid(30, 20, size=6.0)
        seen.update(np.unique(terrain.fields(g, seed=seed)["biome"]).tolist())
    missing = [b.key for i, b in enumerate(terrain.BIOMES) if i not in seen]
    assert not missing, missing


def test_a_single_hex_has_a_six_corner_coast():
    g = gridlib.HexGrid(5, 5, size=10.0)
    mask = np.zeros((g.rows, g.cols), dtype=bool)
    mask[2, 2] = True
    loops = coast.boundary_loops(g, mask)
    assert len(loops) == 1 and len(loops[0]) == 6
    assert coast.signed_area(loops[0]) > 0.0


def test_a_block_of_hexes_has_one_loop_and_a_hole_makes_two():
    g = gridlib.HexGrid(7, 7, size=10.0)
    mask = np.zeros((g.rows, g.cols), dtype=bool)
    mask[2:5, 2:5] = True
    outer = coast.boundary_loops(g, mask)
    assert len(outer) == 1
    assert coast.signed_area(outer[0]) > 0.0

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

    assert sum(1 for p in loops if coast.signed_area(p) < 0) < len(loops)


def test_a_mask_of_the_wrong_shape_is_refused():
    g = gridlib.HexGrid(5, 5, size=10.0)
    try:
        coast.boundary_loops(g, np.zeros((5, 4), dtype=bool))
    except ValueError:
        return
    raise AssertionError("boundary_loops accepted a mis-shaped mask")


def _alpha_at(canvas, wx, wy) -> float:
    """The canvas's alpha at a world point -- the inverse of :meth:`Canvas.grid`."""
    x = int(round((wx - canvas.center[0]) * canvas.scale + canvas.w / 2.0 - 0.5))
    y = int(round((canvas.center[1] - wy) * canvas.scale + canvas.h / 2.0 - 0.5))
    return float(canvas.a[y, x])


def _inked(key, g, width: float = 0.24, alpha: float = 0.80):
    """A canvas holding only the border layer for ``key``, and its raster lookup."""
    canvas = Canvas(int(np.ceil(g.extent[0])) + 4, int(np.ceil(g.extent[1])) + 4,
                    scale=1.0)
    cell = g.cell(*canvas.grid())
    autotile.border_layer(canvas, g, key, cell, paint.INK, width=width, alpha=alpha)
    return canvas, cell


def test_a_border_is_a_difference_and_the_two_sides_of_it_agree():

    g = gridlib.HexGrid(9, 7, size=10.0)
    labels = (np.arange(g.rows)[:, None] + np.arange(g.cols)[None, :]) % 3
    key = autotile.borders(labels, g)
    for row, col in g.flat():
        for d in range(autotile.EDGES):
            nr, nc = (int(v) for v in g.nbr[row, col, d])
            mine = int(key[row, col]) >> d & 1
            if nr < 0:
                assert mine == 1
                continue
            theirs = int(key[nr, nc]) >> ((d + 3) % autotile.EDGES) & 1
            assert mine == theirs
            assert mine == (1 if labels[nr, nc] != labels[row, col] else 0)


def test_the_rim_is_a_class_you_name_and_not_a_class_by_accident():

    g = gridlib.HexGrid(1, 1, size=10.0)
    sea = np.zeros((1, 1), dtype=np.int64)
    assert int(autotile.borders(sea, g, outside=0)[0, 0]) == 0
    assert int(autotile.borders(sea, g)[0, 0]) == autotile.MASKS - 1


def test_a_labels_grid_of_the_wrong_shape_is_refused():
    g = gridlib.HexGrid(5, 5, size=10.0)
    try:
        autotile.borders(np.zeros((5, 4), dtype=np.int64), g)
    except ValueError:
        return
    raise AssertionError("borders accepted a mis-shaped label grid")


def test_the_ink_is_on_the_edges_the_key_marks_and_nowhere_else():

    g = gridlib.HexGrid(10, 8, size=12.0)
    labels = np.zeros((g.rows, g.cols), dtype=np.int64)
    labels[3:6, 3:7] = 1
    key = autotile.borders(labels, g, outside=0)
    canvas, _cell = _inked(key, g)
    hw = 0.5 * 0.24 * g.size
    apothem = 0.5 * np.sqrt(3.0) * g.size
    for row, col in g.flat():
        cx, cy = g.centre_of((row, col))
        for d in range(autotile.EDGES):
            marked = bool(int(key[row, col]) >> d & 1)
            ang = np.radians(60.0 * d)
            wx = cx + (apothem - 0.5 * hw) * np.cos(ang)
            wy = cy + (apothem - 0.5 * hw) * np.sin(ang)
            assert (_alpha_at(canvas, wx, wy) > 0.4) == marked, (row, col, d)


def test_the_tiles_of_a_map_are_the_keys_it_wears_in_the_order_it_meets_them():
    g = gridlib.HexGrid(24, 16, size=6.0)
    f = terrain.fields(g, seed=8)
    key = autotile.borders((~f["water"]).astype(np.int64), g, outside=0)
    index, keys = autotile.tiles(key)
    assert sorted(keys) == sorted({int(v) for v in key.ravel()})
    assert keys[0] == int(key[0, 0])
    for row, col in g.flat():
        assert keys[int(index[row, col])] == int(key[row, col])

    assert 2 <= len(keys) <= 40
    assert len(autotile.tiles(autotile.borders(np.zeros((g.rows, g.cols), np.int64),
                                               g, outside=0))[1]) == 1


def test_the_ink_of_a_tile_stays_inside_its_own_hexagon():

    one = gridlib.HexGrid(1, 1, 11.0)
    for m in range(autotile.MASKS):
        tile = autotile.tile_image(m, 11.0, paint.INK)
        inside = one.cell(*tile.grid()).ok
        assert not np.any(tile.a[~inside] > 0.0), m
        assert tile.a[0, 0] == 0.0 and tile.a[-1, -1] == 0.0


def test_a_tile_with_more_edges_marked_never_has_less_ink():
    sums = [float(autotile.tile_image(m, 11.0, paint.INK).a.sum())
            for m in range(autotile.MASKS)]
    assert sums[0] == 0.0
    assert sums[autotile.MASKS - 1] > 0.0
    for m in range(autotile.MASKS):
        for d in range(autotile.EDGES):
            if not m >> d & 1:
                assert sums[m | (1 << d)] >= sums[m] - 1e-9


def test_a_tile_key_that_is_not_a_six_bit_number_is_refused():
    try:
        autotile.tile_image(autotile.MASKS, 10.0, paint.INK)
    except ValueError:
        return
    raise AssertionError("tile_image accepted a key that is not a tile")


def test_the_sheet_is_a_container_and_a_cell_is_the_tile_that_goes_in_it():
    size, gap = 12.0, 5
    sheet = autotile.sheet(paint.INK, size, gap=gap)
    tw, th = autotile.tile_size(size)
    n = autotile.SHEET_COLS
    assert (sheet.w, sheet.h) == (n * tw + (n + 1) * gap, n * th + (n + 1) * gap)
    for m in range(autotile.MASKS):
        row, col = divmod(m, autotile.SHEET_COLS)
        x, y = gap + col * (tw + gap), gap + row * (th + gap)
        tile = autotile.tile_image(m, size, paint.INK)
        assert np.array_equal(sheet.a[y:y + th, x:x + tw], tile.a), m
        assert np.array_equal(sheet.rgb[y:y + th, x:x + tw], tile.rgb), m


def test_the_tiled_coast_is_the_stroked_coast_inside_the_map():

    g = gridlib.HexGrid(12, 9, size=11.0)
    labels = np.zeros((g.rows, g.cols), dtype=np.int64)
    labels[0:5, 0:7] = 1
    labels[6, 8:10] = 2
    w, h = g.extent
    a = Canvas(int(np.ceil(w + 12)), int(np.ceil(h + 12)), scale=1.0)
    b = a.clone()
    cell = g.cell(*a.grid())
    for loop in coast.boundary_loops(g, labels.astype(bool)):
        paint.stroke_polyline(a, loop, 0.24 * g.size, paint.INK, alpha=0.80,
                              closed=True)
    key = autotile.borders(labels, g, outside=0)
    autotile.border_layer(b, g, key, cell, paint.INK, width=0.24, alpha=0.80)

    assert np.array_equal(a.to_uint8()[cell.ok], b.to_uint8()[cell.ok])
    assert np.any(a.a[~cell.ok] > 0.0)
    assert not np.any(b.a[~cell.ok] > 0.0)


def test_within_asks_the_difference_question_of_a_subset():

    g = gridlib.HexGrid(9, 7, size=10.0)
    labels = (np.arange(g.rows)[:, None] * 2 + np.arange(g.cols)[None, :]) % 5
    inside = (np.arange(g.rows)[:, None] + np.arange(g.cols)[None, :]) % 3 != 0
    key = autotile.borders(labels, g, within=inside)
    for row, col in g.flat():
        for d in range(autotile.EDGES):
            bit = int(key[row, col]) >> d & 1
            nr, nc = (int(v) for v in g.nbr[row, col, d])
            if not inside[row, col] or nr < 0 or not inside[nr, nc]:
                assert bit == 0, (row, col, d)
            else:
                assert bit == (labels[nr, nc] != labels[row, col]), (row, col, d)


def test_the_rim_drops_out_of_the_question_when_the_edge_is_not_within():
    g = gridlib.HexGrid(1, 1, size=10.0)
    labels = np.zeros((1, 1), dtype=np.int64)
    assert int(autotile.borders(labels, g, within=np.zeros((1, 1), dtype=bool))[0, 0]) == 0

    assert int(autotile.borders(labels, g, within=np.ones((1, 1), dtype=bool))[0, 0]) == 0

    assert int(autotile.borders(labels, g)[0, 0]) == autotile.MASKS - 1


def test_a_within_grid_of_the_wrong_shape_is_refused():
    g = gridlib.HexGrid(5, 5, size=10.0)
    try:
        autotile.borders(np.zeros((5, 5), dtype=np.int64), g,
                         within=np.zeros((5, 4), dtype=bool))
    except ValueError:
        return
    raise AssertionError("borders accepted a mis-shaped within grid")


def test_a_biome_border_is_an_edge_inside_the_land_and_nowhere_else():

    g = gridlib.HexGrid(20, 14, size=8.0)
    f = terrain.fields(g, seed=5, sea_fraction=0.45)
    key = autotile.borders(f["biome"], g, within=~f["water"])
    assert np.any(key)
    for row, col in g.flat():
        for d in range(autotile.EDGES):
            if not int(key[row, col]) >> d & 1:
                continue
            nr, nc = (int(v) for v in g.nbr[row, col, d])
            assert nr >= 0
            assert not f["water"][row, col] and not f["water"][nr, nc]
            assert f["biome"][nr, nc] != f["biome"][row, col]


def test_a_biome_border_wears_more_of_the_sheet_than_a_coast_does():

    g = gridlib.HexGrid(24, 16, size=6.0)
    f = terrain.fields(g, seed=8)
    coast_keys = autotile.tiles(autotile.borders((~f["water"]).astype(np.int64),
                                                 g, outside=0))[1]
    inland_keys = autotile.tiles(autotile.borders(f["biome"], g,
                                                  within=~f["water"]))[1]
    assert 2 <= len(coast_keys) <= 40
    assert len(coast_keys) < len(inland_keys) <= autotile.MASKS


def test_the_biome_borders_reach_only_land_and_are_off_until_asked_for():

    kw = dict(seed=4, cols=16, rows=11, size=11.0, provinces=0)
    off = paint.map_image(**kw)
    assert np.array_equal(off.to_uint8(), paint.map_image(**kw, inland=0.0).to_uint8())
    on = paint.map_image(**kw, inland=0.55)
    changed = np.any(off.to_uint8().astype(int) != on.to_uint8().astype(int), axis=2)
    assert changed.any()
    g = gridlib.HexGrid(16, 11, size=11.0)
    cell = g.cell(*off.grid())
    f = terrain.fields(g, seed=4)
    r = np.clip(cell.rc[..., 0], 0, g.rows - 1)
    c = np.clip(cell.rc[..., 1], 0, g.cols - 1)
    assert not np.any(changed & ~cell.ok)
    assert not np.any(changed & f["water"][r, c])


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
    from gen.maps import HexGrid
    g = HexGrid(12, 8, 10.0)
    cell = g.cell(*c.grid())
    assert float(img.a[cell.ok].min()) > 0.99
    assert float(img.a[0, 0]) == 0.0
    assert float(img.a[-1, -1]) == 0.0


def test_land_and_sea_are_painted_in_their_own_colours():

    from gen.maps import HexGrid
    g = HexGrid(20, 14, size=9.0)
    f = terrain.fields(g, seed=3)

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
        assert px[2] > px[0], (r, c, px)
    for r, c in dry:
        px = pixel_at(float(g.centers[r, c, 0]), float(g.centers[r, c, 1]))
        assert px[1] > px[2], (r, c, px)


def test_every_seed_paints_a_map_with_a_coast_on_it():
    for seed in (0, 4, 19):
        img = paint.map_image(seed=seed, cols=18, rows=12, size=9.0)
        px = img.to_uint8()
        colours = len(np.unique(px[..., :3].reshape(-1, 3), axis=0))
        assert colours > 20, (seed, colours)
        assert 0.05 < float((px[..., 3] > 0).mean()) < 0.98


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

    assert np.allclose(filled, np.maximum(elev, best), atol=1e-5)


def test_filling_never_lowers_the_ground_and_never_leaves_a_sump():
    for seed in (7, 3, 23, 19):
        g, f, filled, down, flux = _flow(seed=seed)
        land = ~f["water"]
        assert float((filled - f["elev"]).min()) >= 0.0

        assert all(int(down[r, c]) >= 0 for r, c in g.flat() if land[r, c])


def test_the_flood_leaves_almost_all_of_the_map_where_it_was():

    for seed in (7, 3, 23):
        g, f, filled, down, flux = _flow(seed=seed)
        land = ~f["water"]
        lifted = int(((filled - f["elev"])[land] > 1e-9).sum())
        assert lifted < 0.35 * int(land.sum()), (seed, lifted)


def test_the_sea_carries_the_whole_map():

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


def _political(cols=30, rows=20, size=13.0, seed=7, count=8, province_seed=None):
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed)
    s = seed if province_seed is None else province_seed
    return g, f, region.provinces(g, f, count=count, seed=s)


def test_every_province_is_connected():

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
    assert used == [-1] + list(range(res["count"]))


def test_a_province_holds_the_cells_whose_capital_is_nearest_by_path():

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
    assert sides == {0, 1}


def test_a_second_seed_divides_the_same_land_differently():
    g = gridlib.HexGrid(24, 16, size=11.0)
    f = terrain.fields(g, seed=7)
    a = region.provinces(g, f, count=6, seed=1)
    b = region.provinces(g, f, count=6, seed=2)
    assert np.array_equal(a["labels"] >= 0, b["labels"] >= 0)
    assert not np.array_equal(a["labels"], b["labels"])
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

    assert float(a[cell.edge_distance() > 2.0].max()) == 0.0


def test_a_map_with_provinces_and_rivers_paints_the_same_bytes_twice():
    a = paint.map_image(seed=5, cols=20, rows=14, size=10.0, provinces=6)
    b = paint.map_image(seed=5, cols=20, rows=14, size=10.0, provinces=6)
    assert hashlib.sha256(a.to_uint8().tobytes()).hexdigest() == \
           hashlib.sha256(b.to_uint8().tobytes()).hexdigest()
    c = paint.map_image(seed=5, cols=20, rows=14, size=10.0, provinces=0, river=0.0)
    assert not np.array_equal(a.to_uint8(), c.to_uint8())


def _named(seed=11, cols=20, rows=14, size=13.0, provinces=6):
    """A map and the names it sets on itself."""
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed)
    res = region.provinces(g, f, count=provinces, seed=seed)
    return g, res, label_lib.layout(g, res)


def _farthest(g, res, i) -> int:
    """Brute force: how far the deepest cell of a province is from any other.

    The same number the breadth-first search in ``labels`` computes, arrived at by
    asking every cell how far it is from every cell that is not its own.  Slow on
    purpose: it is the independent answer a search that walks the wrong way cannot
    fake.
    """
    lab = np.asarray(res["labels"])
    mine = [(int(j) // g.cols, int(j) % g.cols)
            for j in np.asarray(res["provinces"][i]["cells"])]
    theirs = [(r, c) for r in range(g.rows) for c in range(g.cols)
              if int(lab[r, c]) != i]
    return max(min(gridlib.distance(tuple(g.axes[r, c]), tuple(g.axes[rr, cc]))
                   for rr, cc in theirs)
               for r, c in mine)


def test_a_name_is_set_where_its_province_is_widest():
    """The anchor is the pole of inaccessibility and not the centre of mass.

    A name is set where its province is widest, because a name that reaches over a
    frontier reads as the neighbour's -- the one failure a map label cannot have.
    The anchor is therefore a cell of the province whose distance to the nearest
    cell that is *not* of the province is the province's maximum.
    """
    g, res, lay = _named()
    lab = np.asarray(res["labels"])
    for e in lay:
        r, c = e["cell"]
        assert int(lab[r, c]) == e["index"], e["name"]
        mine = min(gridlib.distance(tuple(g.axes[r, c]), tuple(g.axes[rr, cc]))
                   for rr in range(g.rows) for cc in range(g.cols)
                   if int(lab[rr, cc]) != e["index"])
        assert mine == _farthest(g, res, e["index"]), e["name"]


def test_a_name_never_reaches_past_its_own_frontier():
    """The box of a label lies in the province, sampled far finer than the ink.

    ``labels`` decides this on a nine-by-three lattice, and the test asks the same
    question on a forty-one-by-nine one: what a label must never do is cross a
    frontier, so the failure has to be impossible between the samples and not only
    at them.
    """
    g, res, lay = _named()
    lab = np.asarray(res["labels"])
    drawn = [e for e in lay if e["drawn"]]
    assert len(drawn) >= 4, [e["name"] for e in lay]
    for e in drawn:
        x0, y0, x1, y1 = e["box"]
        X, Y = np.meshgrid(np.linspace(x0, x1, 41), np.linspace(y0, y1, 9))
        cell = g.cell(X, Y)
        assert np.all(lab[cell.rc[..., 0], cell.rc[..., 1]] == e["index"]), e["name"]


def test_two_names_never_merge():
    """Two labels flank to flank read as one word, so they keep air between them."""
    _g, _res, lay = _named()
    boxes = [e["box"] for e in lay if e["drawn"]]
    for i, a in enumerate(boxes):
        for b in boxes[:i]:
            assert (a[2] <= b[0] or b[2] <= a[0]
                    or a[3] <= b[1] or b[3] <= a[1]), (a, b)


def test_a_name_that_cannot_be_set_is_skipped_and_says_why():
    """A label is never squeezed into a province that cannot hold it.

    The name here is longer than any province on the map is wide, so there is no
    size at which it fits and the layout says so.  What it must not do is set it
    anyway at three pixels, or drop it in silence: the sidecar's job is to carry
    the names a game would label with, and it can only do that if the ones this
    map left off are written down.
    """
    g, res, _lay = _named()
    fake = dict(res)
    fake["provinces"] = [dict(p) for p in res["provinces"]]
    long_name = "Widerthanitsownprovince"
    i = min(range(len(fake["provinces"])), key=lambda k: fake["provinces"][k]["area"])
    fake["provinces"][i]["name"] = long_name
    lay = label_lib.layout(g, fake)
    e = lay[i]
    assert e["name"] == long_name
    assert not e["drawn"]
    assert e["why"] in ("no room", "crowded")
    skipped = label_lib.census(lay)["skipped"]
    assert {"index": i, "name": long_name, "why": e["why"]} in skipped


def test_a_name_is_dark_ink_over_a_pale_halo():
    """What the label sits on is a painting, so it carries its own background.

    The two colours are the whole reason a name is legible over a forest and over
    a desert without either of them being redrawn: the dark ink is separated from
    the ground by a pale ring, so the map underneath can be anything.
    """
    assert float(np.mean(label_lib.INK)) < 0.30
    assert float(np.mean(label_lib.HALO)) > 0.85


def test_names_change_no_land():
    """A name is ink on its province and not a change to the map under it."""
    kw = dict(seed=11, cols=20, rows=14, size=13.0, provinces=6)
    plain = paint.map_image(names=False, **kw)
    named = paint.map_image(names=True, **kw)
    assert np.array_equal(plain.a, named.a)
    assert not np.array_equal(plain.to_uint8(), named.to_uint8())


def test_a_name_is_the_same_name_at_any_scale():
    """The label belongs to the geography, and only its pixels belong to the map.

    A name sits on the same hex at 10 px and at 20 px, is set at twice the size,
    and is drawn or skipped for the same reason, because every quantity in the
    placement is a length in hexes.
    """
    _g1, _res1, small = _named(size=10.0)
    _g2, _res2, large = _named(size=20.0)
    assert [e["name"] for e in small] == [e["name"] for e in large]
    for a, b in zip(small, large):
        assert a["cell"] == b["cell"]
        assert a["drawn"] == b["drawn"]
        if a["drawn"]:
            assert b["em"] == 2.0 * a["em"]


def test_the_names_are_the_same_names_twice():
    _g1, _res1, a = _named()
    _g2, _res2, b = _named()
    assert json.dumps(label_lib.census(a)) == json.dumps(label_lib.census(b))
    assert [e.get("box") for e in a] == [e.get("box") for e in b]


def _map(seed=11, cols=20, rows=14, size=13.0, provinces=6):
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed)
    return g, f, region.provinces(g, f, count=provinces, seed=seed)


def _painted(seed=11, cols=20, rows=14, size=13.0, provinces=6, realms=4):
    """One political map: the grid, the provinces, the roster and what it wears."""
    g, _f, res = _map(seed, cols, rows, size, provinces)
    names = politics.factions(res, realms, seed=seed)
    owners = politics.owners(res, names, seed=seed)
    table = politics.washes(names)
    return g, res, names, owners, table


def test_a_realms_provinces_are_one_block():
    """Ownership grown over the province graph is connected by construction.

    The same claim ``region`` makes about provinces is made here about realms, and
    for the same reason: a realm in two pieces is not one a player can march across.
    """
    for seed in (0, 3, 11):
        for realms in (2, 3, 5):
            _g, res, names, owners, _table = _painted(seed=seed, realms=realms)
            nbr = [p["neighbours"] for p in res["provinces"]]
            for realm in range(len(names)):
                held = {j for j, o in enumerate(owners) if o == realm}
                assert held, f"seed {seed}: realm {realm} holds nothing"
                seen = {min(held)}
                q = deque([min(held)])
                while q:
                    j = q.popleft()
                    for k in nbr[j]:
                        if owners[k] == realm and k not in seen:
                            seen.add(k)
                            q.append(k)
                assert seen == held, \
                    f"seed {seed} realm {realm}: holds {len(held)}, reaches {len(seen)}"


def test_every_province_is_held_exactly_once():
    for seed in (0, 3, 11):
        _g, res, names, owners, table = _painted(seed=seed)
        assert len(names) == 4
        assert len(owners) == res["count"]
        assert sorted(set(owners)) == list(range(len(names)))
        tints = politics.tint_table(res, owners, table)
        assert tints.shape == (res["count"], 3)
        for i, o in enumerate(owners):
            assert np.allclose(tints[i], table[o])


def test_a_landmass_is_never_left_unowned():
    """Fewer realms than islands: the largest keep theirs, and nobody is nobody's.

    A realm cannot be grown onto an island from across the water, so a landmass
    without a realm of its own is a hole in the map rather than a small realm --
    which is why the realms are shared between landmasses before they are shared
    by size.
    """
    res = {"count": 3, "provinces": [
        {"name": "A", "neighbours": [1]},
        {"name": "B", "neighbours": [0]},
        {"name": "C", "neighbours": []}]}
    two = politics.owners(res, 2, seed=1)
    assert two[2] >= 0, "the island was left to nobody"
    assert sorted(set(two)) == [0, 1]
    one = politics.owners(res, 1, seed=1)
    assert one[0] == one[1] == 0 and one[2] == -1
    assert politics.owners(res, 0, seed=1) == [-1, -1, -1]


def test_a_roster_is_a_palette_and_not_a_sample():
    """The realms on one map are a step apart, and the draw is a function of the seed.

    Seven tinctures and more factions than that means two realms will pick the same
    one, so the roster is *drawn against the colour*: a name is taken only when the
    map can tell it apart from every name already on the map.
    """
    for seed in range(6):
        _g, res, names, _owners, table = _painted(seed=seed, realms=5)
        assert len(names) == 5
        assert politics.factions(res, 5, seed=seed) == names
        assert not set(names) & {p["name"] for p in res["provinces"]}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert palette.delta_e(table[i], table[j]) >= politics.WASH_STEP, \
                    (seed, names[i], names[j])
        assert politics.factions(res, 0, seed=seed) == []


def test_a_realm_never_wears_the_colour_of_the_sea():
    """The sea is a colour on this map too, and the roster keeps out of its way.

    A wash with no chroma is never refused: a grey has no hue to be confused with
    the sea's, which is how a metal gets to keep its lightness.
    """
    hues = [float(palette.srgb_to_oklch(c)[2])
            for c in list(paint.SEA_RAMP) + [paint.RIVER]]
    middle = 0.5 * (min(hues) + max(hues))
    half = 0.5 * (max(hues) - min(hues))
    for seed in range(6):
        _g, _res, _names, _owners, table = _painted(seed=seed)
        for colour in table:
            _l, chroma, hue = palette.srgb_to_oklch(colour)
            if float(chroma) < politics.SEA_CHROMA:
                continue
            d = abs(float(hue) - middle)
            assert min(d, 1.0 - d) >= half + politics.SEA_GUARD - 1e-9, \
                (seed, palette.srgb_to_hex(colour))


def test_a_wash_stays_in_the_band_its_terrain_is_read_in():
    """The map *spends* a livery's step; it does not clip it.

    A province dark enough to swallow a mountain range is a province that has
    stopped being a map, so a wash is bounded -- and bounded *smoothly*, because a
    clip would push two different liveries onto the same colour and lose exactly the
    distinction the band was there to protect.
    """
    for seed in range(8):
        _g, _res, _names, _owners, table = _painted(seed=seed, realms=5)
        for colour in table:
            light, chroma, _h = palette.srgb_to_oklch(colour)
            assert politics.WASH_L - politics.WASH_SPAN - 1e-9 <= float(light) \
                <= politics.WASH_L + politics.WASH_SPAN + 1e-9, palette.srgb_to_hex(colour)
            assert float(chroma) <= politics.WASH_C + 1e-9, palette.srgb_to_hex(colour)


def test_a_realms_colour_survives_being_scaled_down_to_a_hex():
    """The claim the roster exists for, measured on pixels rather than on the table.

    The same map is drawn twice, with the two realms the roster put closest together
    swapped.  Everything else in the two pictures is identical -- the terrain, the
    relief, the rivers, the ink -- so every difference between them is the realm's
    colour, at a 13 px hex, through the wash and over the ground.  The *median* cell
    is checked rather than the worst one: a pixel under a frontier or a coastline is
    the same pixel in both pictures and shows nothing, which is what ink is for.
    """
    for seed in (0, 3, 11):
        g, res, names, owners, table = _painted(seed=seed, cols=20, rows=14,
                                                provinces=6, realms=4)
        _d, i, j = min((palette.delta_e(table[a], table[b]), a, b)
                       for a in range(len(names)) for b in range(a + 1, len(names)))
        order = list(range(len(names)))
        order[i], order[j] = order[j], order[i]
        kw = dict(seed=seed, cols=20, rows=14, size=13.0, provinces=6,
                  province_fill=politics.WASH_FILL, province_owners=owners)
        a = paint.map_image(province_tints=politics.tint_table(res, owners, table),
                            **kw).to_uint8()
        b = paint.map_image(province_tints=politics.tint_table(res, owners, table[order]),
                            **kw).to_uint8()
        for realm in (i, j):
            cells = []
            for k, p in enumerate(res["provinces"]):
                if owners[k] != realm:
                    continue
                for cell in p["cells"]:
                    r, c = divmod(int(cell), g.cols)
                    x, y = g.centre_of((r, c))
                    px = int(round(x + a.shape[1] / 2.0 - 0.5))
                    py = int(round(a.shape[0] / 2.0 - 0.5 - y))
                    cells.append(palette.delta_e(a[py, px, :3].astype(float) / 255.0,
                                                 b[py, px, :3].astype(float) / 255.0))
            assert cells, f"seed {seed} realm {realm} holds no cell"
            assert np.median(cells) >= 0.04, (seed, realm, float(np.median(cells)))


def test_a_border_inside_a_realm_is_not_a_frontier():
    """A realm drawn as one block is a realm a player can see.

    The same picture with everyone holding one province each, and with one realm
    holding them all: the tints are the same in both, so the difference is the ink
    on the *internal* borders -- and ink only ever darkens, which is what makes this
    a comparison and not a hash.
    """
    g, res, names, owners, table = _painted(seed=11, realms=1)
    assert len(names) == 1 and set(owners) == {0}
    tints = politics.tint_table(res, owners, table)
    kw = dict(seed=11, cols=20, rows=14, size=13.0, provinces=6)
    whole = paint.map_image(province_tints=tints, province_owners=[0] * res["count"],
                            **kw).to_uint8()
    split = paint.map_image(province_tints=tints,
                            province_owners=list(range(res["count"])), **kw).to_uint8()
    assert not np.array_equal(whole, split)
    assert np.all(whole >= split), "damping a frontier put ink on, not off"
    assert int(whole.sum()) > int(split.sum())


def _roadmap(seed=7, cols=22, rows=15, size=11.0, provinces=8, sea=0.42):
    """A map with capitals on it, and the roads between them."""
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed, sea_fraction=sea)
    res = region.provinces(g, f, count=provinces, seed=seed)
    return g, f, res, roads.network(g, f, res)


def test_a_route_is_a_walk_from_one_hex_to_another():
    """Every step is a step: the path is hexes an army could march through.

    That is the whole difference between a road and a straight line, so it is
    checked on the answer rather than trusted from the search: consecutive cells are
    neighbours, nothing is walked through twice, no cell is water, and the cost
    reported is the cost of the cells entered and nothing else.
    """
    g, f, res, _net = _roadmap()
    cost = roads.cost_field(g, f)
    flat = cost.reshape(-1)
    caps = [(int(p["capital"][0]), int(p["capital"][1])) for p in res["provinces"]]
    for a in caps:
        for b in caps:
            if a >= b:
                continue
            c, cells = roads.route(g, cost, a, b)
            assert cells, (a, b)
            assert cells[0] == a[0] * g.cols + a[1]
            assert cells[-1] == b[0] * g.cols + b[1]
            assert len(set(cells)) == len(cells), "a road walked a hex twice"
            for j, k in zip(cells, cells[1:]):
                r, cc = divmod(j, g.cols)
                rr, kk = divmod(k, g.cols)
                assert [rr, kk] in [list(n) for n in g.nbr[r, cc]], (a, b, (r, cc), (rr, kk))
            assert np.isfinite(flat[cells]).all()
            assert abs(c - float(flat[cells[1:]].sum())) < 1e-9


def test_a_route_is_the_cheapest_walk_and_not_a_walk():
    """Checked against brute force, not against the shape of the queue.

    Bellman-Ford relaxes every edge until nothing moves, which is the definition of
    a shortest path and knows nothing about heaps; a search that agrees with it on
    every pair of capitals is a search that is right for the right reason.
    """
    g, f, res, _net = _roadmap()
    cost = roads.cost_field(g, f)
    flat = cost.reshape(-1)
    n = g.rows * g.cols
    caps = [(int(p["capital"][0]), int(p["capital"][1])) for p in res["provinces"]]
    edges = [(j, k) for r in range(g.rows) for c in range(g.cols)
             for rr, cc in g.nbr[r, c] if rr >= 0
             for j, k in [(r * g.cols + c, int(rr) * g.cols + int(cc))]
             if np.isfinite(flat[k])]
    for a in caps:
        best = np.full(n, np.inf)
        best[a[0] * g.cols + a[1]] = 0.0
        for _ in range(n):
            moved = False
            for j, k in edges:
                if best[j] + flat[k] < best[k] - 1e-12:
                    best[k] = best[j] + flat[k]
                    moved = True
            if not moved:
                break
        for b in caps:
            c, _cells = roads.route(g, cost, a, b)
            assert abs(c - float(best[b[0] * g.cols + b[1]])) < 1e-9, (a, b)


def test_the_roads_join_every_capital_on_its_own_landmass():
    """A spanning forest, and no bridge.

    Capitals are grouped by flood-filling the land, which is a different algorithm
    from the one that built the roads; the roads must join the capitals of one
    landmass to each other and must never join two landmasses -- a road across open
    water is a bridge nobody built.
    """
    g, f, res, net = _roadmap()
    water = np.asarray(f["water"], dtype=bool)
    land = ~water
    label = np.full((g.rows, g.cols), -1, dtype=np.int64)
    parts = 0
    for r, c in g.flat():
        if not land[r, c] or label[r, c] >= 0:
            continue
        label[r, c] = parts
        q = deque([(int(r), int(c))])
        while q:
            rr, cc = q.popleft()
            for r2, c2 in g.nbr[rr, cc]:
                r2, c2 = int(r2), int(c2)
                if r2 >= 0 and land[r2, c2] and label[r2, c2] < 0:
                    label[r2, c2] = parts
                    q.append((r2, c2))
        parts += 1

    caps = [(int(p["capital"][0]), int(p["capital"][1])) for p in res["provinces"]]
    home = [int(label[r, c]) for r, c in caps]
    parent = list(range(len(caps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for road in net:
        a, b = find(road["from"]), find(road["to"])
        assert home[road["from"]] == home[road["to"]], "a road crossed open water"
        parent[a] = b

    for i in range(len(caps)):
        for j in range(i + 1, len(caps)):
            assert (find(i) == find(j)) == (home[i] == home[j]), (i, j)
    assert len(net) == len(caps) - parts


def test_a_road_never_crosses_the_sea_even_where_the_spline_goes():
    """The smoothing invents points, so the invented ones are checked too.

    A Catmull-Rom curve between two hexes bulges sideways; on a coast that would put
    the road in the water, which is why this is a test and not a faith.
    """
    for seed in range(4):
        g, f, _res, net = _roadmap(seed=seed, sea=0.5)
        water = np.asarray(f["water"], dtype=bool)
        for road in net:
            cell = g.cell(road["points"][:, 0], road["points"][:, 1])
            assert cell.ok.all(), (seed, road["from"], road["to"])
            assert not water[cell.rc[..., 0], cell.rc[..., 1]].any(), (seed, road["from"], road["to"])


def test_the_roads_are_drawn_only_on_the_land_they_walked():
    """The pixels, not the polyline: the stroke is a hex wide, so it counts most.

    The same map is drawn twice, with the roads off and on, and every pixel that
    changed has to be a pixel of a land cell -- which is the claim at the size it is
    actually read, where a stroke can spill over the ground its centre line is on.
    """
    g = gridlib.HexGrid(22, 15, 11.0)
    f = terrain.fields(g, seed=7, sea_fraction=0.42)
    kw = dict(seed=7, cols=22, rows=15, size=11.0, sea_fraction=0.42, provinces=8)
    off = paint.map_image(road=0.0, **kw)
    on = paint.map_image(road=0.95, **kw)
    a, b = off.to_uint8(), on.to_uint8()
    assert not np.array_equal(a, b), "the roads were not drawn"
    changed = np.any(a != b, axis=-1)
    assert 0.001 < changed.mean() < 0.12, changed.mean()
    w, h = g.extent
    canvas = Canvas(int(np.ceil(w + 2 * 7.0)), int(np.ceil(h + 2 * 7.0)), scale=1.0)
    assert (canvas.w, canvas.h) == (a.shape[1], a.shape[0])
    cell = g.cell(*canvas.grid())
    water = np.asarray(f["water"], dtype=bool)[cell.rc[..., 0], cell.rc[..., 1]] & cell.ok
    assert not (changed & water).any(), "a road was painted over water"


def test_the_same_map_lays_the_same_roads():
    g, f, res, net = _roadmap()
    again = roads.network(g, f, res)
    assert [(r["from"], r["to"], r["cost"], r["cells"]) for r in net] == \
           [(r["from"], r["to"], r["cost"], r["cells"]) for r in again]
    for one, two in zip(net, again):
        assert np.array_equal(one["points"], two["points"])


def _waters(seed=7, cols=22, rows=15, size=11.0, count=7, sea=0.42):
    """A map, its land provinces and its sea provinces."""
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed, sea_fraction=sea)
    return g, f, region.provinces(g, f, count=6, seed=seed), region.seas(g, f, count=count,
                                                                         seed=seed)


def test_the_sea_is_divided_the_way_the_land_is():
    """A sea province is connected, and the whole sea is in exactly one of them.

    A fleet moves province to province, so a sea province in two pieces is the same
    defect a land province in two pieces is; and water the seeds never reached is a
    province of its own rather than a hole, which is what an island is on land.
    """
    g, f, land, sea = _waters()
    water = np.asarray(f["water"], dtype=bool).reshape(-1)
    lab = sea["labels"].reshape(-1)
    assert sea["count"] >= 7
    seen = np.zeros(g.count, dtype=bool)
    for p in sea["provinces"]:
        assert water[p["cells"]].all(), "a sea province was given a hex of land"
        mine = np.zeros(g.count, dtype=bool)
        mine[p["cells"]] = True
        assert not (mine & seen).any()
        seen |= mine

        start = int(p["cells"][0])
        got = {start}
        q = deque([start])
        inside = set(int(j) for j in p["cells"])
        while q:
            j = q.popleft()
            r, c = divmod(j, g.cols)
            for rr, cc in g.nbr[r, c]:
                rr, cc = int(rr), int(cc)
                if rr >= 0 and rr * g.cols + cc in inside and rr * g.cols + cc not in got:
                    got.add(rr * g.cols + cc)
                    q.append(rr * g.cols + cc)
        assert got == inside, p["name"]
    assert np.array_equal(seen, water), "the sea is not covered exactly once"
    assert (lab[~water] == -1).all(), "the sea division labelled the land"


def test_the_two_divisions_tile_the_whole_map_between_them():
    """Every hex is a province or a sea province, and no hex is both."""
    g, f, land, sea = _waters()
    a = np.asarray(land["labels"]).reshape(-1)
    b = np.asarray(sea["labels"]).reshape(-1)
    assert not ((a >= 0) & (b >= 0)).any(), "a hex was in a province and a sea"
    assert ((a >= 0) | (b >= 0)).all(), "a hex was in neither"


def test_a_sea_province_reads_as_the_water_it_is():
    """The name is the only place a player reads what a province is like.

    So the water's endings and the land's are disjoint: a stem may repeat -- the
    stem is the province's number -- but nothing on the map should be called
    ``Aldfell`` and be water.
    """
    _g, _f, land, sea = _waters()
    wet = region._END["ocean"] + region._END["shallows"] + region._END["deep water"]
    dry = region._END["hills"] + region._END["plains"] + region._END["forest"]
    names = [p["name"] for p in sea["provinces"]]
    land_names = [p["name"] for p in land["provinces"]]
    assert len(set(names) | set(land_names)) == len(names) + len(land_names)
    for n in names:
        assert any(n.endswith(e) for e in wet), n
    for n in land_names:
        assert not any(n.endswith(e) for e in wet), n
    assert not (set(wet) & set(dry))


def test_charting_the_sea_does_not_touch_the_land():
    """The claim at the size it is read: a political layer over the water only."""
    g = gridlib.HexGrid(22, 15, 11.0)
    f = terrain.fields(g, seed=7, sea_fraction=0.42)
    kw = dict(seed=7, cols=22, rows=15, size=11.0, sea_fraction=0.42, provinces=8, road=0.85)
    plain = paint.map_image(seas=0, **kw).to_uint8()
    charted = paint.map_image(seas=8, **kw).to_uint8()
    assert not np.array_equal(plain, charted), "the sea provinces were not drawn"
    changed = np.any(plain != charted, axis=-1)
    assert 0.0 < changed.mean() < 0.04, changed.mean()
    w, h = g.extent
    canvas = Canvas(int(np.ceil(w + 2 * 7.0)), int(np.ceil(h + 2 * 7.0)), scale=1.0)
    cell = g.cell(*canvas.grid())
    water = np.asarray(f["water"], dtype=bool)[cell.rc[..., 0], cell.rc[..., 1]] & cell.ok
    assert (changed & ~water).sum() == 0, "a sea province was drawn on the land"


def test_a_realm_holds_the_water_off_its_own_coast():
    """Claimed, not grown -- and claimed by the realm with the longest coast.

    The rule is deliberately coarse: a sea province is one zone and goes to one
    realm, so the tally of coast is checked against a second tally taken here, and
    the ties have to fall the same way, by the lower realm, or the answer depends on
    the order the coast was walked in.
    """
    g, f, res, sea = _waters(count=9)
    names = politics.factions(res, 4, 7)
    own = politics.owners(res, names, 7)
    held = politics.waters(g, res, sea, own)
    land = np.asarray(res["labels"]).reshape(-1)
    for i, p in enumerate(sea["provinces"]):
        coast = {}
        for j in p["cells"]:
            r, c = divmod(int(j), g.cols)
            for rr, cc in g.nbr[r, c]:
                rr, cc = int(rr), int(cc)
                if rr < 0:
                    continue
                other = int(land[rr * g.cols + cc])
                if other >= 0 and int(own[other]) >= 0:
                    o = int(own[other])
                    coast[o] = coast.get(o, 0) + 1
        if coast:
            assert held[i] == min(coast, key=lambda o: (-coast[o], o)), p["name"]
        else:
            assert held[i] == -1, p["name"]
    assert (held >= 0).any(), "no realm holds any water on a map full of coasts"


def test_the_high_seas_are_nobody_s():
    """With no realm holding a coast there is nobody to claim anything."""
    g, f, res, sea = _waters(count=9)
    held = politics.waters(g, res, sea, [-1] * res["count"])
    assert (held == -1).all()


def test_a_realm_wears_its_water_without_redrawing_the_land():
    """The sea layer's own claim, now with the water *owned*.

    A territorial sea is the realm's, so it is drawn in the realm's colour rather
    than in a neutral one -- and it is still only water that changes.
    """
    g = gridlib.HexGrid(22, 15, 11.0)
    f = terrain.fields(g, seed=7, sea_fraction=0.42)
    res = region.provinces(g, f, count=8, seed=7)
    names = politics.factions(res, 4, 7)
    own = politics.owners(res, names, 7)
    kw = dict(seed=7, cols=22, rows=15, size=11.0, sea_fraction=0.42, provinces=8, road=0.85,
              province_fill=politics.WASH_FILL, province_tints=politics.washes(names),
              province_owners=own)
    plain = paint.map_image(seas=0, **kw).to_uint8()
    held = paint.map_image(seas=8, **kw).to_uint8()
    changed = np.any(plain != held, axis=-1)
    assert changed.any(), "territorial waters were not drawn"
    assert 0.0 < changed.mean() < 0.04, changed.mean()
    w, h = g.extent
    canvas = Canvas(int(np.ceil(w + 2 * 7.0)), int(np.ceil(h + 2 * 7.0)), scale=1.0)
    cell = g.cell(*canvas.grid())
    water = np.asarray(f["water"], dtype=bool)[cell.rc[..., 0], cell.rc[..., 1]] & cell.ok
    assert (changed & ~water).sum() == 0, "owning the water redrew the land"


def _campaign(seed=7, cols=26, rows=17, size=13.0, provinces=8, factions=4, sea=0.42,
              sea_provinces=8):
    """A map, its realms, its roads, its rivers -- and the file that carries them."""
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed, sea_fraction=sea)
    res = region.provinces(g, f, count=provinces, seed=seed) if provinces else None
    own, record = (), ()
    if res is not None and factions:
        names = politics.factions(res, factions, seed)
        own = politics.owners(res, names, seed)
        record = tuple({"name": n, "key": n.lower(),
                        "colour": palette.srgb_to_hex(politics.wash(n))}
                       for n in names)
    courses, _thr = paint.river_courses(g, f)
    routes = roads.network(g, f, res) if res is not None else []
    seas = region.seas(g, f, count=sea_provinces, seed=seed) if sea_provinces else None
    held = (politics.waters(g, res, seas, own)
            if (seas is not None and res is not None and own) else ())
    doc = campaign.campaign(g, f, res, seed=seed, owner=own, realms=record,
                            routes=routes, courses=courses, seas=seas, sea_owner=held)
    return g, f, res, own, routes, courses, doc


def test_the_sea_sails_into_the_campaign_file_too():
    """And it is checked the same way the land is: a partition of the water.

    A fleet needs somewhere to be, and a client that had to re-derive the sea
    division would have to re-implement the whole of :mod:`gen.maps.region` to move
    a ship.  So the sea provinces go in with their cells, exactly as the land ones
    do, and nobody owns them -- which is what ``-1`` says.
    """
    g, f, res, _own, routes, courses, doc = _campaign()
    sea = region.seas(g, f, count=8, seed=7)
    water = np.asarray(f["water"], dtype=bool).reshape(-1)
    assert len(doc["seas"]) == sea["count"] >= 8
    known = set(range(len(doc["realms"])))
    seen = np.zeros(g.count, dtype=bool)
    for p in doc["seas"]:
        mine = np.zeros(g.count, dtype=bool)
        mine[p["cells"]] = True
        assert water[p["cells"]].all(), "a sea province was given a hex of land"
        assert int(mine.sum()) == int(mine[water].sum()) == p["area"]
        assert not (mine & seen).any()
        assert p["owner"] == -1 or p["owner"] in known, p
        seen |= mine
    assert np.array_equal(seen, water), "the sea is not in the file exactly once"
    assert any(p["owner"] >= 0 for p in doc["seas"]), "nobody was given a coast"


def test_the_campaign_file_is_the_whole_map_in_numbers():
    """The land, its owners and its ways, all as cells -- checked against the map.

    A file that a client can run has to carry what the client would otherwise have
    to derive, so every province's cells are in it and every road's walk is in it.
    The check is that those cells are the ones the generator used, exactly: the
    provinces partition the land and never touch the sea, and no road is written
    down that the capitals do not lie at the ends of.
    """
    g, f, res, own, routes, courses, doc = _campaign()
    assert json.loads(json.dumps(doc)) == doc
    assert doc["format"] == campaign.FORMAT
    assert doc["cells"] == [int(v) for v in np.asarray(f["biome"]).reshape(-1)]
    assert doc["grid"]["cols"] == g.cols and doc["grid"]["rows"] == g.rows

    water = np.asarray(f["water"], dtype=bool).reshape(-1)
    seen = np.zeros(g.count, dtype=bool)
    for i, p in enumerate(doc["provinces"]):
        mine = np.zeros(g.count, dtype=bool)
        mine[p["cells"]] = True
        assert not (mine & water).any(), "a province was given a hex of sea"
        assert not (mine & seen).any(), "two provinces were given the same hex"
        seen |= mine
        assert int(mine.sum()) == p["area"] == int(res["provinces"][i]["area"])
        assert p["owner"] == (int(own[i]) if i < len(own) else -1)
        assert p["name"] == res["provinces"][i]["name"]
        assert [p["capital"][0], p["capital"][1]] == [int(v) for v in
                                                      res["provinces"][i]["capital"]]
    assert np.array_equal(seen, ~water), "the land is not covered once"

    for road, live in zip(doc["roads"], routes):
        assert road["cells"] == [int(r) * g.cols + int(c) for r, c in live["path"]]
        for i in (0, -1):
            flat = road["cells"][i]
            rr, cc = divmod(flat, g.cols)
            assert [rr, cc] in [list(res["provinces"][k]["capital"]) for k in (road["from"],
                                                                              road["to"])]
        for j, k in zip(road["cells"], road["cells"][1:]):
            r, c = divmod(j, g.cols)
            r2, c2 = divmod(k, g.cols)
            assert [r2, c2] in [list(n) for n in g.nbr[r, c]], (j, k)

    for river, live in zip(doc["rivers"], courses):
        assert river["cells"] == [int(r) * g.cols + int(c) for r, c in live["path"]]
        assert river["drainage"] == round(float(live["flux"]), 1)


def test_the_table_goes_into_the_file_with_the_map():
    """A client should not have to guess a colour, a cost, or what is water.

    The biome table is the rules; it travels in the file, in classification order,
    so that a cell's index means the same thing on both sides of the export.
    """
    _g, _f, _res, _own, _routes, _courses, doc = _campaign()
    assert len(doc["biomes"]) == len(terrain.BIOMES)
    for i, (row, b) in enumerate(zip(doc["biomes"], terrain.BIOMES)):
        assert row == {"key": b.key, "name": b.name, "cost": int(b.cost),
                       "water": bool(b.water), "colour": b.hex}, (i, row)


def test_the_campaign_is_a_function_of_the_map_and_not_of_the_pixels():
    """The hex size is a rendering choice and does not reach the file.

    Which is the reason the roads are carried as hexes and not as the polylines
    they are drawn with: a polyline at 26 px is a different polyline at 9 px, and
    the map is supposed to be the same map.
    """
    small = _campaign(size=9.0)[-1]
    large = _campaign(size=26.0)[-1]
    assert json.dumps(small) == json.dumps(large)


def test_every_owner_in_the_file_is_a_realm_in_the_file():
    """A client indexes the roster, so an owner that is not in it is a crash.

    ``-1`` is the one value the file is allowed to hand out for a province nobody
    holds; anything else has to be a realm the file also lists, or the political map
    cannot be painted at all.
    """
    _g, _f, _res, _own, _routes, _courses, doc = _campaign()
    assert doc["realms"], "the map has realms but the file does not"
    known = set(range(len(doc["realms"])))
    for p in doc["provinces"]:
        assert p["owner"] == -1 or p["owner"] in known, p
    assert all("colour" in r for r in doc["realms"])


def test_a_campaign_file_is_small_enough_to_ship_per_map():
    """Measured, because a file nobody can afford to ship is a file nobody ships.

    A 26x17 map is 442 hexes and a province each; a player downloads one of these
    per map, so the order of the number matters more than the number.
    """
    _g, _f, _res, _own, _routes, _courses, doc = _campaign()
    text = json.dumps(doc, indent=1)
    assert len(text) < 20000, len(text)
    assert len(text) < len(json.dumps(doc, indent=2))
    assert doc["water"] == round(doc["water"], 3)


def _site(seed=7, size=56.0, grounds=True, **kw):
    return site_render.site_image(seed=seed, cols=5, rows=4, size=size,
                                  grounds=grounds, **kw)


def _site_land(grid, canvas, seed=7):
    """The pixel mask of the site's land, and its cells."""
    f = terrain.fields(grid, seed=seed)
    cell = grid.cell(*canvas.grid())
    r = np.clip(cell.rc[..., 0], 0, grid.rows - 1)
    c = np.clip(cell.rc[..., 1], 0, grid.cols - 1)
    return cell, cell.ok & ~f["water"][r, c]


def test_a_site_wears_its_ground_and_the_sea_does_not_move():
    """The one claim the hook makes: materials are for land, and water has no ground.

    The sea is the same depth ramp as the flat map, *pixel for pixel*, because the
    material is applied with the water masked out rather than painted and then faded.
    The land is not the same, or the module would be a no-op with a good story.
    """
    g = gridlib.HexGrid(5, 4, 56.0)
    flat = _site(grounds=False)
    worn = _site(grounds=True)
    again = _site(grounds=True)
    cell, land = _site_land(g, flat)

    a, b = flat.to_uint8(), worn.to_uint8()
    assert np.array_equal(again.to_uint8(), b)
    water = cell.ok & ~land
    assert water.any() and land.any()
    assert np.array_equal(a[water], b[water]), "the sea must not be painted"
    assert (a[land] != b[land]).any(), "the land must be"


def test_the_material_reaches_the_light_and_not_only_the_colour():
    """A ground is a height field; if only its tone survived, this would be a tint.

    ``relief=0`` switches the height off and leaves the colour modulation, so the
    difference between the two pictures is exactly the material's contribution to the
    one Lambert pass -- and it has to be large enough to see, not merely nonzero.
    """
    g = gridlib.HexGrid(5, 4, 56.0)
    tint = _site(relief=0.0)
    _cell, land = _site_land(g, tint)
    a = tint.to_uint8()[..., :3].astype(np.float64)
    b = _site(relief=1.15).to_uint8()[..., :3].astype(np.float64)
    assert np.abs(a[land] - b[land]).mean() > 2.0


def test_the_material_belongs_to_the_map_and_not_to_the_canvas():
    """A wider margin crops the same site; it does not slide the material.

    The sampler wraps world coordinates, and the canvas is centred on the map, so the
    same pixel of ground lands at the same place in both pictures.  This is the test
    that would fail if the phase were taken from the raster's own indices -- which is
    the easy version, and is why a ground would visibly crawl when the page grew.
    """
    narrow = _site(margin=10.0).to_uint8()
    wide = _site(margin=30.0).to_uint8()
    assert wide.shape[0] - narrow.shape[0] == 40
    assert wide.shape[1] - narrow.shape[1] == 40
    assert np.array_equal(wide[20:-20, 20:-20], narrow)


def test_a_site_river_is_water_in_a_channel_and_not_on_the_sea():
    """The close-up's river is a place: a carved groove with water in it.

    The water is drawn on the land and only on the land, so the sea must not move by
    one pixel between a site with a river and one without.  That is the same claim the
    ground hook makes, asked of the second thing that is allowed to change the height
    field; the carve is why the two pictures differ at all, and the mask is why the
    difference stops at the coast.
    """
    seed, cols, rows, size = 39, 8, 6, 32.0
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed)
    courses, _threshold = paint.river_courses(g, f, 0.90)
    assert courses, "the seed chosen for this test has no river"
    dry = site_render.site_image(seed=seed, cols=cols, rows=rows, size=size,
                                 river=0.0, river_fraction=0.90).to_uint8()
    wet = site_render.site_image(seed=seed, cols=cols, rows=rows, size=size,
                                 river=0.92, river_fraction=0.90).to_uint8()
    assert dry.shape == wet.shape
    changed = np.any(dry != wet, axis=-1)
    assert changed.any(), "the river was not drawn"
    cell = g.cell(*Canvas(dry.shape[1], dry.shape[0], scale=1.0).grid())
    r = np.clip(cell.rc[..., 0], 0, rows - 1)
    c = np.clip(cell.rc[..., 1], 0, cols - 1)
    water = f["water"][r, c] | ~cell.ok
    assert not changed[water].any(), "the river changed the sea"


def test_a_site_cliff_and_wet_shore_stay_on_the_land():
    """The crease and the wet band are facts about the shore, not about the sea.

    Both are drawn as overlays on the land mask, so a site with them and a site
    without them must agree at every water pixel.  The test also asks that they
    actually do something: an overlay that is masked to nothing is a no-op with a
    sidecar.  ``grounds=False`` keeps the check about the overlays rather than about
    the material, and keeps it quick.
    """
    seed, cols, rows, size = 7, 6, 5, 32.0
    common = dict(seed=seed, cols=cols, rows=rows, size=size, grounds=False,
                  river=0.0)
    plain = site_render.site_image(cliff=0.0, shore=0.0, **common).to_uint8()
    cliff = site_render.site_image(cliff=0.55, shore=0.0, **common).to_uint8()
    shore = site_render.site_image(cliff=0.0, shore=0.052, **common).to_uint8()
    assert cliff.shape == shore.shape == plain.shape
    g = gridlib.HexGrid(cols, rows, size)
    f = terrain.fields(g, seed=seed)
    cell = g.cell(*Canvas(plain.shape[1], plain.shape[0], scale=1.0).grid())
    r = np.clip(cell.rc[..., 0], 0, rows - 1)
    c = np.clip(cell.rc[..., 1], 0, cols - 1)
    water = f["water"][r, c] | ~cell.ok
    for name, changed in (("cliff", np.any(plain != cliff, axis=-1)),
                          ("shore", np.any(plain != shore, axis=-1))):
        assert changed.any(), f"the {name} overlay drew nothing"
        assert not changed[water].any(), f"the {name} overlay changed the sea"


def _iso(seed=7, cols=6, rows=5, size=32.0, **kw):
    return iso_render.iso_image(seed=seed, cols=cols, rows=rows, size=size,
                                sea_fraction=kw.pop("sea_fraction", 0.20),
                                edge=kw.pop("edge", 0.0), **kw)


def test_a_board_is_the_same_board_at_any_margin():
    """The margin is a page, not a camera: widening it crops nothing and moves nothing.

    The projection is derived from the map's own bounds and then offset by the margin,
    so the board lands ten pixels right and ten down in a board with ten more pixels of
    margin.  If the material were indexed by canvas pixels instead of world coordinates,
    the ground would crawl under the board when the page grew -- the same failure the
    site test catches, one camera further down the road.
    """
    narrow = _iso(margin=8.0).to_uint8()
    wide = _iso(margin=18.0).to_uint8()
    assert wide.shape[0] - narrow.shape[0] == 20
    assert wide.shape[1] - narrow.shape[1] == 20
    assert np.array_equal(wide[10:-10, 10:-10], narrow)


def test_the_board_stands_up_and_the_hero_stands_on_land():
    """Height is a side: the same geography with and without a vertical scale.

    ``rise=0`` is a map seen from the south; a positive rise is a board.  The one must
    be shorter than the other, and the hero must be on a land hex -- a figure standing
    in the sea is the failure mode of "nearest cell to the middle".
    """
    flat = _iso(rise=0.0)
    tall = _iso(rise=3.0)
    assert tall.h > flat.h, (flat.h, tall.h)
    g = gridlib.HexGrid(6, 5, 32.0)
    f = terrain.fields(g, seed=7, sea_fraction=0.20, edge=0.0)
    hero = iso_render.hero_cell(g, f)
    assert hero is not None
    assert not f["water"][hero]
    all_sea = dict(f, water=np.ones_like(f["water"], dtype=bool))
    assert iso_render.hero_cell(g, all_sea) is None


def test_the_hero_can_face_four_ways_and_a_named_hex_must_be_land():
    """The board has four directions and one rule about where a figure may stand.

    Right and left are the sprite and its mirror; up and down are the back and front
    views.  All four must reach the pixel, so the test compares the finished boards
    and not the argument.  A hex named on the command line is a request, and a request
    for the sea or for the edge of the world is refused with a message rather than an
    ``IndexError`` from inside the projection.
    """
    g = gridlib.HexGrid(5, 4, 24.0)
    f = terrain.fields(g, seed=7, sea_fraction=0.20, edge=0.0)
    hero = iso_render.hero_cell(g, f)
    assert hero is not None
    boards = {}
    for facing in ("right", "left", "up", "down"):
        boards[facing] = iso_render.iso_image(
            seed=7, cols=5, rows=4, size=24.0, sea_fraction=0.20, edge=0.0,
            levels=5, hero=hero, facing=facing).to_uint8()
    assert len({b.tobytes() for b in boards.values()}) == 4
    assert not np.array_equal(boards["right"], boards["left"])
    assert not np.array_equal(boards["up"], boards["down"])
    water = tuple(np.argwhere(f["water"])[0])
    for bad in (water, (99, 99)):
        try:
            iso_render.iso_image(seed=7, cols=5, rows=4, size=24.0,
                                 sea_fraction=0.20, edge=0.0, hero=bad)
        except ValueError as exc:
            assert "water" in str(exc) or "off" in str(exc)
        else:
            raise AssertionError(f"the board accepted the hero at {bad}")
    try:
        iso_render.iso_image(seed=7, cols=5, rows=4, size=24.0,
                             sea_fraction=0.20, edge=0.0, hero=hero,
                             facing="sideways")
    except ValueError as exc:
        assert "faces" in str(exc)
    else:
        raise AssertionError("a made-up facing was accepted")


def test_the_terraces_are_a_number_of_steps():
    """A continuous field gives every edge a face; a few levels give a landscape.

    The claim is arithmetic rather than pictorial: five terraces means at most five
    heights above the sea, one terrace means none, and zero means the field the map
    itself has.  The picture is then a consequence of the number, not a second guess
    at it -- and the same call twice is the same board.
    """
    g = gridlib.HexGrid(6, 5, 32.0)
    f = terrain.fields(g, seed=7, sea_fraction=0.20, edge=0.0)
    land = ~f["water"]
    assert land.any()
    stepped = iso_render.terraces(f, 5)
    assert len(np.unique(stepped[land])) <= 5
    assert np.all(iso_render.terraces(f, 1)[land] == 0.0)
    assert np.array_equal(iso_render.terraces(f, 0), np.maximum(f["height"] - f["sea"], 0.0))
    assert not np.array_equal(stepped, iso_render.terraces(f, 0))
    assert np.array_equal(_iso(levels=5, rise=3.0).to_uint8(),
                          _iso(levels=5, rise=3.0).to_uint8())

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
