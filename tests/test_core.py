"""Regression tests for the core toolkit.

Run either way::

    python tests/test_core.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import credits, curve, mat, noise, palette, render, sdf  # noqa: E402
from core.canvas import Canvas  # noqa: E402


# --------------------------------------------------------------- geometry
def test_circle_signs():
    assert np.isclose(sdf.circle(0.0, 0.0, 1.0), -1.0)
    assert np.isclose(sdf.circle(1.0, 0.0, 1.0), 0.0, atol=1e-12)
    assert np.isclose(sdf.circle(2.0, 0.0, 1.0), 1.0)
    assert sdf.circle(0.5, 0.0, 1.0) < 0.0


def test_box_rounded_matches_extents():
    # The corner radius eats into the corners only: the flat faces keep their
    # extent, so the shortest face stays 0.5 from the centre whatever r is.
    assert np.isclose(sdf.box(0.0, 0.0, 1.0, 0.5, 0.2), -0.5)
    assert np.isclose(sdf.box(0.0, 0.0, 1.0, 0.5, 0.0), -0.5)
    # Outside the flat edge the distance is measured from the rounded corner.
    assert np.isclose(sdf.box(1.0, 0.0, 1.0, 0.5, 0.0), 0.0, atol=1e-12)
    assert np.isclose(sdf.box(2.0, 0.0, 1.0, 0.5, 0.0), 1.0, atol=1e-12)
    assert np.isclose(sdf.box(1.2, 0.8, 1.0, 0.5, 0.3), np.hypot(0.5, 0.6) - 0.3, atol=1e-12)


def test_segment_capsule():
    assert np.isclose(sdf.segment(-5.0, 0.0, (0.0, 0.0), (1.0, 0.0), 0.25), 4.75)
    assert np.isclose(sdf.segment(0.5, 0.0, (0.0, 0.0), (1.0, 0.0), 0.25), -0.25)


def test_polygon_inside_negative_for_both_windings():
    square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for pts in (square, list(reversed(square))):
        assert sdf.polygon(0.0, 0.0, pts) < 0.0
        assert sdf.polygon(2.0, 0.0, pts) > 0.0
        assert np.isclose(sdf.polygon(1.0, 0.0, pts), 0.0, atol=1e-12)
        assert np.isclose(sdf.polygon(0.0, 0.0, pts), -1.0, atol=1e-12)


def test_concave_polygon_distance():
    # An L-shape: the re-entrant corner must not push the sign the wrong way.
    pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    assert sdf.polygon(0.5, 0.5, pts) < 0.0  # inside
    assert sdf.polygon(1.6, 1.6, pts) > 0.0  # in the notch, outside
    assert np.isclose(sdf.polygon(0.0, 0.0, pts), 0.0, atol=1e-12)  # on a vertex


def test_operators_ordering():
    d1 = sdf.circle(0.0, 0.0, 1.0)
    d2 = sdf.box(0.0, 0.0, 0.5, 0.5, 0.0)
    assert np.isclose(sdf.union(d1, d2), -1.0)
    assert np.isclose(sdf.intersection(d1, d2), -0.5)
    assert sdf.difference(d1, d2) > 0.0  # the centre is carved away


def test_smooth_union_fillets_and_stays_valid():
    a = sdf.circle(0.0, 0.0, 1.0)
    # Smooth union is *sound*: it never overestimates.  Unioning a shape with
    # itself costs exactly k/4 of thickness -- the material the fillet needs.
    k = 0.3
    assert sdf.smooth_union(a, a, k) <= a
    assert np.isclose(sdf.smooth_union(a, a, k), a - k / 4.0)
    assert np.isclose(sdf.smooth_union(a, a, 0.0), a)

    # Two discs that do not touch: the fillet must add material in the gap.
    left = lambda x, y: sdf.circle(x, y, 0.5, -0.7, 0.0)  # noqa: E731
    right = lambda x, y: sdf.circle(x, y, 0.5, 0.7, 0.0)  # noqa: E731
    gap_hard = sdf.union(left(0.0, 0.0), right(0.0, 0.0))
    gap_smooth = sdf.smooth_union(left(0.0, 0.0), right(0.0, 0.0), 0.4)
    assert np.isclose(gap_hard, 0.2)
    assert gap_smooth < gap_hard
    assert np.isclose(sdf.smooth_union(left(0.0, 0.0), right(0.0, 0.0), 0.0), gap_hard)


def test_outline_thickness():
    f = sdf.outline(lambda x, y: sdf.circle(x, y, 1.0), 0.1)
    assert np.isclose(f(1.0, 0.0), -0.1)
    assert np.isclose(f(1.2, 0.0), 0.1)
    assert f(0.0, 0.0) > 0.0


def test_transforms_rotate_and_mirror():
    vertical = lambda x, y: sdf.box(x, y, 0.2, 1.0, 0.0)  # noqa: E731
    turned = sdf.rotated(vertical, 90.0)
    assert np.isclose(turned(0.9, 0.0), -0.1, atol=1e-9)  # now a wide, flat box
    assert turned(0.0, 0.9) > 0.0

    off = sdf.moved(vertical, 3.0, 0.0)
    assert np.isclose(off(3.0, 0.0), -0.2, atol=1e-9)

    asym = sdf.moved(vertical, 1.0, 0.0)
    assert np.isclose(sdf.mirrored(asym, "x")(-1.0, 0.0), -0.2, atol=1e-9)


def test_repeated_lattice_periodicity():
    dots = sdf.repeated(lambda x, y: sdf.circle(x, y, 0.12), 0.5)
    assert np.isclose(dots(0.1, 0.1), dots(0.6, 0.1))


def test_matrix_inverse_and_uniform_scale():
    m = mat.compose(mat.translate(2.0, -1.0), mat.rotate(35.0), mat.scale(2.0))
    assert np.allclose(mat.invert(m) @ m, np.eye(3), atol=1e-12)
    assert np.isclose(mat.uniform_scale(m), 2.0)
    x, y = mat.apply_grid(m, np.array([0.0]), np.array([0.0]))
    assert np.isclose(x[0], 2.0) and np.isclose(y[0], -1.0)


# ------------------------------------------------------------------- noise
def test_noise_range_and_determinism():
    g = Canvas(64, 64).grid()
    n1 = noise.fbm(*g, seed=5, freq=3.0, octaves=4)
    n2 = noise.fbm(*g, seed=5, freq=3.0, octaves=4)
    assert np.array_equal(n1, n2)
    assert n1.min() >= 0.0 and n1.max() <= 1.0
    assert not np.allclose(n1, noise.fbm(*g, seed=6, freq=3.0, octaves=4))


def test_value_noise_bounds_and_continuity():
    x = np.linspace(0.0, 4.0, 257)
    n = noise.value_noise(x, np.zeros_like(x), seed=1, freq=1.0)
    assert n.min() >= 0.0 and n.max() <= 1.0
    assert np.abs(np.diff(n)).max() < 0.5  # smooth, not white noise


def test_ridged_turbulence_worley_ranges():
    g = Canvas(48, 48).grid()
    for field in (noise.ridged, noise.turbulence):
        v = field(*g, seed=2, freq=2.0, octaves=3)
        assert v.min() >= 0.0 and v.max() <= 1.0
    w = noise.worley(*g, seed=2, freq=2.0)
    assert w.min() >= 0.0
    assert noise.worley(0.5, 0.5, seed=2, freq=1.0) < 1.5


def test_warp_actually_displaces():
    g = Canvas(32, 32).grid()
    wx, wy = noise.warp(*g, seed=1, amp=0.4, freq=2.0)
    assert not np.allclose(wx, g[0])
    assert not np.allclose(wy, g[1])
    assert np.abs(wx - g[0]).max() <= 0.4 + 1e-9


def test_hash_is_stable_and_spread():
    a = noise.hash2(np.array([0, 1, 2, 3]), np.array([0, 0, 0, 0]), seed=0)
    assert np.array_equal(a, noise.hash2(np.array([0, 1, 2, 3]), np.array([0, 0, 0, 0]), seed=0))
    assert a.min() >= 0.0 and a.max() < 1.0
    assert len(set(np.round(a, 6))) == 4
    assert noise.hash2(np.array([-1]), np.array([-1]), seed=0)[0] != noise.hash2(
        np.array([0]), np.array([0]), seed=0
    )[0]


def test_worley_edges_form_cell_walls():
    g = Canvas(96, 96, scale=96.0).grid()
    e = noise.worley_edges(*g, seed=3, freq=3.0)
    assert np.all(e >= -1e-12)  # F2 >= F1 by definition
    assert e.max() > 0.2
    assert 0.002 < (e < 0.01).mean() < 0.5  # thin walls, not solid fill
    # A cell centre is far from its own wall, a wall is exactly on the boundary.
    f1 = noise.worley(*g, seed=3, freq=3.0)
    assert e[np.unravel_index(np.argmin(f1), f1.shape)] > 0.0
    assert np.array_equal(e, noise.worley_edges(*g, seed=3, freq=3.0))
    assert not np.allclose(e, noise.worley_edges(*g, seed=4, freq=3.0))


# ------------------------------------------------------------------ colour
def test_oklab_roundtrip():
    rgb = np.array([0.2, 0.55, 0.9])
    assert np.allclose(palette.oklab_to_srgb(palette.srgb_to_oklab(rgb)), rgb, atol=1e-6)


def test_hex_roundtrip():
    for h in ("#000000", "#ffffff", "#3a7bd5", "#c0ffee"):
        assert palette.srgb_to_hex(palette.hex_to_srgb(h)) == h


def test_mix_endpoints_and_lightness_monotonic():
    a = np.array([0.05, 0.05, 0.6])
    b = np.array([0.95, 0.85, 0.1])
    assert np.allclose(palette.mix(a, b, 0.0), a, atol=1e-6)
    assert np.allclose(palette.mix(a, b, 1.0), b, atol=1e-6)
    mid = palette.mix(a, b, 0.5)
    assert not np.allclose(mid, (a + b) / 2.0, atol=1e-3)  # not a naive sRGB mix


def test_ramp_map_matches_mix_many():
    stops = [np.array([0.1, 0.2, 0.4]), np.array([0.9, 0.4, 0.2]), np.array([0.95, 0.95, 0.9])]
    ts = np.linspace(0.0, 1.0, 9)
    flat = palette.ramp_map(stops, ts)
    for i, t in enumerate(ts):
        assert np.allclose(flat[i], palette.mix_many(stops, t), atol=1e-6)
    assert np.allclose(palette.ramp(stops, 3), palette.ramp_map(stops, [0.0, 0.5, 1.0]), atol=1e-6)


def test_fit_gamut_stays_inside_srgb():
    for hue in np.linspace(0.0, 1.0, 24):
        rgb = palette.oklch_to_srgb((0.55, 0.4, hue))
        assert np.all(rgb >= -1e-9) and np.all(rgb <= 1.0 + 1e-9)


def test_harmony_and_distinct_palette():
    for scheme in ("monochromatic", "analogous", "complementary", "triadic", "tetradic"):
        cols = palette.harmony(0.3, n=5, scheme=scheme)
        assert len(cols) == 5
        assert all(c.shape == (3,) and np.all(c >= 0.0) and np.all(c <= 1.0) for c in cols)
    cols = palette.distinct_palette(6, seed=4, min_de=0.12)
    assert len(cols) == 6
    worst = min(
        palette.delta_e(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))
    )
    assert worst >= 0.12 * 0.9  # target, not a promise: hue arc + lightness spread
    assert np.array_equal(cols, palette.distinct_palette(6, seed=4, min_de=0.12))
    # Fewer colours can be spread at least as far apart.
    three = palette.distinct_palette(3, seed=4, min_de=0.12)
    assert len(three) == 3
    assert min(palette.delta_e(three[0], three[1]), palette.delta_e(three[1], three[2])) >= worst
    assert len(palette.distinct_palette(14, seed=1, min_de=0.1)) == 14
    assert palette.distinct_palette(0) == []


def test_random_palette_is_seed_reproducible():
    a = palette.random_palette(np.random.default_rng(9), n=4)
    b = palette.random_palette(np.random.default_rng(9), n=4)
    assert np.allclose(a, b)


def test_lighten_darken_shift_hue():
    c = np.array([0.4, 0.5, 0.7])
    assert palette.srgb_to_oklch(palette.lighten(c, 0.1))[0] > palette.srgb_to_oklch(c)[0]
    assert palette.srgb_to_oklch(palette.darken(c, 0.1))[0] < palette.srgb_to_oklch(c)[0]
    assert np.isclose(palette.srgb_to_oklch(palette.shift_hue(c, 0.25))[2], (palette.srgb_to_oklch(c)[2] + 0.25) % 1.0, atol=1e-6)


# ------------------------------------------------------------------ render
def test_coverage_midpoint_and_saturation():
    assert np.isclose(render.coverage(0.0, 1.0), 0.5)
    assert np.isclose(render.coverage(-1.0, 1.0), 1.0)
    assert np.isclose(render.coverage(1.0, 1.0), 0.0)


def test_height_normals_are_unit_and_oriented():
    h = np.tile(np.linspace(0.0, 1.0, 16), (16, 1))  # rises to the +x
    n = render.height_normals(h, dx=1.0, y_up=True)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)
    assert n[8, 8, 0] < 0.0  # surface tilts away from +x
    assert n[8, 8, 2] > 0.0


def test_shade_is_bounded_and_edge_aware():
    c = Canvas(64, 64, scale=32.0)
    X, Y = c.grid()
    d = sdf.circle(X, Y, 0.6)
    rgb = render.shade(np.array([0.5, 0.5, 0.5]), d, dx=c.dx, bevel=0.1)
    assert rgb.shape == (64, 64, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    # The bevel makes the two sides of the disc differ.
    assert not np.allclose(rgb[32, 15], rgb[32, 48])


def test_an_inset_bevel_is_the_same_ramp_turned_over():
    # `shade(inset=True)` is a recess, and the test is the physical signature:
    # a raised moulding is lit on its top face and dark on its bottom, and
    # knocked in the two swap.  The geometry, the light and the bevel width are
    # all identical, so the only thing the comparison can be measuring is the
    # sign of the slope.
    c = Canvas(128, 128, scale=48.0)
    X, Y = c.grid()
    d = sdf.box(X, Y, 0.6, 0.6, 0.15)
    base = np.array([0.5, 0.5, 0.5])
    raised = render.shade(base, d, dx=c.dx, bevel=0.1)
    recessed = render.shade(base, d, dx=c.dx, bevel=0.1, inset=True)
    top = slice(36, 39)      # inside the band along the upper edge
    bottom = slice(90, 93)   # ... and along the lower one
    cols = slice(54, 74)
    for img, top_is_bright in ((raised, True), (recessed, False)):
        hi = float(img[top, cols].mean())
        lo = float(img[bottom, cols].mean())
        assert (hi > lo) == top_is_bright, f"{hi:.4f} vs {lo:.4f}"
    assert not np.allclose(raised, recessed)


def test_gaussian_blur_preserves_mean_and_spreads():
    a = np.zeros((33, 33))
    a[16, 16] = 1.0
    b = render.gaussian_blur(a, 2.0)
    assert np.isclose(b.sum(), a.sum(), atol=1e-6)
    assert b[16, 18] > 0.0
    assert b[16, 16] < 1.0
    assert np.array_equal(render.gaussian_blur(a, 0.0), a)


def test_translate_array_roundtrip():
    a = np.arange(25.0).reshape(5, 5)
    up = render.translate_array(a, 1, 0)
    assert np.array_equal(render.translate_array(up, -1, 0)[:, :4], a[:, :4])
    assert np.allclose(up[:, 0], 0.0)


def test_drop_shadow_offset_direction():
    mask = np.zeros((64, 64))
    mask[30:34, 30:34] = 1.0
    sh = render.drop_shadow(mask, scale=64.0, offset=(0.0, -0.08), sigma=0.01, opacity=0.5)
    assert sh.shape == mask.shape
    assert sh[38, 32] > sh[24, 32]  # world -y means lower rows in the raster
    assert sh.max() <= 0.5 + 1e-9


def test_sheet_layout_and_background():
    tiles = [Canvas(20, 20, background=(1.0, 0.0, 0.0)) for _ in range(5)]
    s = render.sheet(tiles, cols=2, gap=4, background=(1.0, 1.0, 1.0))
    assert s.w == 2 * 20 + 3 * 4 and s.h == 3 * 20 + 4 * 4
    assert np.allclose(s.a, 1.0)
    assert np.allclose(s.rgb[0, 0], palette.srgb_to_linear((1.0, 1.0, 1.0)))


# ------------------------------------------------------------------ canvas
def test_canvas_grid_is_centred_and_y_up():
    c = Canvas(8, 4, scale=4.0)
    X, Y = c.grid()
    assert X.shape == (4, 8) and Y.shape == (4, 8)
    assert np.isclose(X.mean(), 0.0) and np.isclose(Y.mean(), 0.0)
    assert Y[0, 0] > Y[-1, 0]  # row 0 is the top of the image, world y points up
    assert np.isclose(c.dx, 0.25)


def test_canvas_compositing_over_opaque():
    c = Canvas(4, 4, background=(1.0, 1.0, 1.0))
    c.over_srgb((0.0, 0.0, 0.0), np.ones((4, 4)))
    assert np.allclose(c.rgb, 0.0, atol=1e-9)
    assert np.allclose(c.a, 1.0)


def test_canvas_composite_is_associative_ish():
    c = Canvas(4, 4)
    half = np.full((4, 4), 0.5)
    c.over_srgb((1.0, 0.0, 0.0), half)
    c.over_srgb((0.0, 0.0, 1.0), half)
    assert np.isclose(c.a[0, 0], 0.75)
    # Source-over in linear light: the second source keeps half its weight, the
    # first keeps half of the remaining half.
    expected = palette.srgb_to_linear((1.0, 0.0, 0.0)) * 0.25 + palette.srgb_to_linear((0.0, 0.0, 1.0)) * 0.5
    assert np.allclose(c.rgb[0, 0], expected / 0.75, atol=1e-9)


def test_canvas_blit_clips_out_of_bounds():
    c = Canvas(4, 4, background=(1.0, 1.0, 1.0))
    tile = Canvas(4, 4, background=(0.0, 0.0, 0.0))
    c.blit(tile.rgb, tile.a, -2, -2)
    assert np.allclose(c.rgb[0, 0], 0.0)
    assert np.allclose(c.rgb[3, 3], palette.srgb_to_linear((1.0, 1.0, 1.0)))
    c.blit(tile.rgb, tile.a, 99, 99)  # completely outside: must not raise


def test_to_uint8_roundtrip_of_flat_colour():
    col = (0.2, 0.6, 0.9)
    c = Canvas(2, 2, background=col)
    arr = c.to_uint8()
    assert arr.dtype == np.uint8 and arr.shape == (2, 2, 4)
    assert np.array_equal(arr[0, 0, :3], np.round(np.array(col) * 255).astype(np.uint8))


def test_a_pixel_the_file_calls_invisible_exports_no_colour():
    """Zero alpha and a colour cannot both survive the export.

    Alpha is coverage, so the tail of a blurred shadow is a wide field of pixels
    whose alpha is a millionth -- a pixel the file will declare is not there.  The
    colour it holds at that alpha is not the colour of anything in the picture: it
    is where the blur happened to land.  It matters anyway, because a consumer that
    filters straight alpha (a bilinear nine-slice stretch over content, a naive
    downsample) reaches into the invisible pixel and pulls its colour out into a
    visible one.  So the export rule is total: where the exported alpha is zero,
    the colour is zero too.
    """
    scale, sigma = 32.0, 0.08
    c = Canvas(96, 96, scale=scale)
    plate = np.zeros((96, 96))
    plate[30:66, 30:66] = 1.0
    sh = render.drop_shadow(plate, scale, offset=(0.0, -0.02), sigma=sigma, opacity=0.42)
    c.over_srgb((0.03, 0.03, 0.05), sh)

    # The tail is real: this many pixels round to invisible and still hold colour,
    # so a test that only checked the export would be checking nothing.
    invisible = (np.round(c.a * 255.0) == 0.0) & (c.rgb.max(axis=-1) > 0.0)
    assert invisible.sum() > 100, int(invisible.sum())

    arr = c.to_uint8()
    assert not (((arr[..., 3] == 0) & (arr[..., :3] > 0).any(axis=-1))).any()

    # And it is a *repair*: every pixel the file calls visible is exactly what a
    # plain quantisation would have written, so nothing the eye can see moved.
    visible = arr[..., 3] > 0
    plain = np.clip(np.round(palette.linear_to_srgb(c.rgb) * 255.0), 0, 255).astype(np.uint8)
    assert np.array_equal(arr[..., :3][visible], plain[visible])


# ------------------------------------------------------------------ curves
def test_bezier_hits_its_endpoints_and_degree_two_midpoint():
    p = curve.bezier_quadratic((0.0, 0.0), (1.0, 1.0), (2.0, 0.0), n=9)
    assert p.shape == (9, 2)
    assert np.allclose(p[0], (0.0, 0.0)) and np.allclose(p[-1], (2.0, 0.0))
    # At t = 1/2 a quadratic Bezier is the control polygon weighted (1, 2, 1)/4:
    # (0 + 2*1 + 2, 0 + 2*1 + 0)/4.
    assert np.allclose(p[4], (1.0, 0.5))


def test_bezier_cubic_is_flat_when_control_points_are_collinear():
    p = curve.bezier_cubic((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), n=17)
    assert np.allclose(p[:, 1], 0.0)
    assert np.all(np.diff(p[:, 0]) > 0.0)  # monotone in x


def test_bezier_rejects_a_single_point():
    try:
        curve.bezier([(0.0, 0.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("a one-point Bezier must be refused")


def test_catmull_rom_passes_through_its_control_points():
    pts = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0), (4.0, 0.0)]
    c = curve.catmull_rom(pts, per_segment=6)
    assert c.shape[0] == 3 * 6 + 1
    for ctrl in pts:
        assert np.min(np.linalg.norm(c - np.asarray(ctrl), axis=1)) < 1e-9
    assert np.allclose(c[0], pts[0]) and np.allclose(c[-1], pts[-1])


def test_chaikin_keeps_closed_shapes_inside_their_convex_hull():
    square = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    c = curve.chaikin(square, iterations=3, closed=True)
    assert c.shape == (4 * 2**3, 2)
    assert c.min() >= -1e-12 and c.max() <= 1.0 + 1e-12
    # Corner cutting cannot introduce mass: the centroid is preserved.
    assert np.allclose(c.mean(axis=0), square.mean(axis=0))


def test_chaikin_open_curve_keeps_its_ends():
    line = np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    c = curve.chaikin(line, iterations=2, closed=False)
    assert np.allclose(c[0], line[0]) and np.allclose(c[-1], line[-1])
    assert np.all(c[:, 1] == 0.0)


def test_arc_length_and_resample():
    square = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    s = curve.arc_length(square, closed=True)
    assert np.allclose(s, [0.0, 1.0, 2.0, 3.0, 4.0])
    r = curve.resample_closed(square, n=8)
    assert r.shape == (8, 2)
    # Even spacing by arc length: every step of the resampled loop is equal.
    step = np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1)
    assert np.allclose(step, step[0], atol=1e-9)
    assert np.isclose(step[0], 0.5)


# -------------------------------------------------------------- end to end
def test_icon_generation_is_deterministic():
    import cli

    a = cli.make_icon(5, 64)
    b = cli.make_icon(5, 64)
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.a, b.a)
    assert not np.array_equal(a.rgb, cli.make_icon(6, 64).rgb)
    assert 0.0 <= a.a.min() and a.a.max() <= 1.0


def test_png_write_is_byte_identical():
    import cli

    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.png"
        p2 = Path(tmp) / "b.png"
        cli.make_icon(11, 48).save(p1)
        cli.make_icon(11, 48).save(p2)
        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
        assert h1 == h2
        assert p1.stat().st_size > 0


# ---------------------------------------------------------------- credits
def test_every_source_can_actually_be_credited():
    """A source without a credit line is a licence violation, not a tidy-up.

    The list is short and hand-written, so the only way it goes wrong is by
    omission: a source added with a title and no author, or with the licence
    filed under the wrong key, is exactly the failure this exists to catch.
    """
    assert credits.missing_fields() == []
    for s in credits.sources():
        assert s["attribution"].strip()
        assert s["author"] in s["attribution"]
        if s["license"].startswith("CC "):
            # A CC licence is a *versioned* one, and the attribution has to name
            # the licence it is granted under, not just "Creative Commons".
            assert any(ch.isdigit() for ch in s["license"])
            assert s["url"].startswith("http")
            assert "Creative Commons" in s["attribution"]


def test_the_credits_text_carries_the_attribution_verbatim():
    text = credits.credits()
    for s in credits.sources():
        assert s["attribution"] in text


def test_manifest_reads_the_sidecar_envelope_and_ignores_other_json():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "a.png").write_bytes(b"")
        (d / "a.json").write_text('{"seed": 4, "source": "procedural:x", "license": "MIT",'
                                 ' "things": [1, 2, 3]}', encoding="utf-8")
        (d / "b.json").write_text('{"unrelated": true}', encoding="utf-8")
        (d / "c.json").write_text("not json at all", encoding="utf-8")
        found = credits.manifest(d)
        assert [e["image"] for e in found] == ["a.png"]
        assert found[0]["seed"] == 4 and found[0]["count"] == 3
        text = credits.credits(d)
        assert "a.png x3  seed 4  MIT" in text
        assert "b.json" not in text


def test_every_sheet_the_demos_write_has_a_sidecar():
    """No asset ships unregistered.

    This is the invariant that makes the manifest worth trusting: not that the
    sidecars are right, but that there are none missing.  The sheets are rendered
    tiny here, because what is under test is the envelope, not the pixels.
    """
    import cli

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cli.demo_icons(d / "icons.png", seed=3, count=2, size=24)
        cli.demo_shapes(d / "shapes.png", size=24)
        cli.demo_texture(d / "texture.png", seed=3, size=16)
        cli.demo_palette(d / "palette.png", seed=3, count=2)
        cli.demo_portraits(d / "portraits.png", seed=3, count=2, size=32)
        cli.demo_ui(d / "ui.png", seed=3, count=1, scale=1, style="slate")
        cli.demo_vectors(d, seed=3, count=1, scale=1, style="slate")
        cli.demo_maps(d / "maps.png", seed=3, count=1, cols=12, rows=8, size=8.0)
        cli.demo_tiles(d / "tiles.png", seed=3, size=64, sim=24, steps=60)
        cli.demo_factions(d / "factions.png", names=["Alder", "Brythmark"], size=48)
        images = {p.name for p in d.glob("*.png")}
        assert images == {"icons.png", "icons-atlas.png", "shapes.png", "texture.png",
                          "palette.png", "portraits.png", "ui.png", "maps.png",
                          "tiles.png", "tiles-atlas.png", "factions.png"}
        # A vector is an asset too, and the manifest lists it as itself rather than
        # as the PNG it is not.
        vectors = {p.name for p in d.glob("*.svg")}
        assert vectors == {"ui.slate.svg", "ui.title-bar.svg", "ui.seed-3.svg"}, vectors
        listed = {e["image"] for e in credits.manifest(d)}
        assert listed == images | vectors
        assert not any("no sidecar" in p for p in credits.check(d))

        # The UI sidecar also carries the one fact an engine cannot work out from
        # the insets: whether each kit's *middle* survives a resize.  The corners
        # are stated by the source; the body was measured, at two sizes.
        import json

        ui = json.loads((d / "ui.json").read_text(encoding="utf-8"))
        assert ui["kits"], "the ui sheet registered no kits"
        for kit in ui["kits"]:
            assert isinstance(kit["panel"]["stretchable"], bool), kit["style"]
            assert kit["panel"]["stretch_error"] >= 0.0, kit["style"]

        # The atlas sidecar is the other half of "an asset is more than a picture":
        # one image, plus the table of rectangles an engine needs to cut it up.
        packed = json.loads((d / "icons-atlas.json").read_text(encoding="utf-8"))
        assert packed["members"] and 0.0 < packed["density"] <= 1.0
        for entry in packed["members"]:
            assert len(entry["rect"]) == 4 and len(entry["trim"]) == 4, entry["name"]


def test_no_picture_the_demos_write_has_a_colour_where_it_has_no_alpha():
    """The export rule, checked on the bytes rather than on the canvas it came from.

    An icon's shadow, a hollow frame's hole and a flag's transparent margin are all
    places a colour can survive at an alpha that rounds to nothing -- and all three
    are exactly the assets a game *filters*: a frame is stretched over content, an
    atlas is sampled with a bilinear kernel.  This reads the files Pillow wrote, so
    it is the exported bytes that carry the claim.
    """
    import cli
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cli.demo_icons(d / "icons.png", seed=3, count=2, size=24)
        cli.demo_ui(d / "ui.png", seed=3, count=1, scale=1, style="slate")
        cli.demo_flags(d / "flags.png", seed=7, count=2, size=48, hang="mixed")
        sheets = sorted(d.glob("*.png"))
        assert sheets, "the fixture wrote nothing"
        for p in sheets:
            with Image.open(p) as img:
                arr = np.asarray(img.convert("RGBA"))
            bad = (arr[..., 3] == 0) & (arr[..., :3] > 0).any(axis=-1)
            assert not bad.any(), (p.name, int(bad.sum()))


# ------------------------------------------------------- the credits screen
def test_the_wrapped_credits_screen_keeps_every_word_of_the_attribution():
    """A credit line broken by the edge of a texture is a credit line that is gone.

    This is the failure mode a screen has and a text file does not: the words are
    all present in ``credits()``, and the renderer wraps them, and somewhere in
    between a line can run off the width and vanish without anything failing.  The
    wrapping is therefore its own function, and this reads it back.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "a.json").write_text(
            '{"seed": 5, "source": "procedural:x", "license": "MIT", "things": [1]}',
            encoding="utf-8")
        (d / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        for cols in (24, 48, 96):
            rows = credits.screen_lines(d, cols=cols)
            words = " ".join(text for _, text in rows).split()
            for s in credits.sources():
                for word in s["attribution"].split():
                    assert word in words, f"{cols} columns lost {word!r}"
            # The manifest has to survive the wrap too, or the screen credits the
            # music and quietly drops the assets.
            assert "a.png" in [w for text in (t for _, t in rows) for w in text.split()]
            assert all(len(text) <= cols for _, text in rows), "a row overran the width"


def test_the_credits_screen_is_sized_by_its_text_and_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "a.json").write_text(
            '{"seed": 5, "source": "procedural:x", "license": "MIT", "things": [1]}',
            encoding="utf-8")
        one = credits.screen(d, scale=2)
        two = credits.screen(d, scale=2)
        assert one.size == two.size
        assert one.tobytes() == two.tobytes(), "the same credits must draw the same screen"
        assert len(set(one.tobytes())) > 1, "the screen is not blank"

        # The geometry is the text's: every scale is the same drawing, bigger.
        small, large = credits.screen(d, scale=1), credits.screen(d, scale=4)
        assert large.size == (small.size[0] * 4, small.size[1] * 4)

        # The height follows the words: one more asset, one more line.
        (d / "b.json").write_text(
            '{"seed": 6, "source": "procedural:y", "license": "MIT", "things": [1]}',
            encoding="utf-8")
        assert credits.screen(d, scale=2).size[1] > one.size[1]


# --------------------------------------------------------- the pre-ship gate
def _build(tmp) -> Path:
    """A tiny but *real* build: sheets that write sidecars, and the credits."""
    import cli

    d = Path(tmp)
    cli.demo_icons(d / "icons.png", seed=3, count=1, size=24, atlas=False)
    cli.demo_shapes(d / "shapes.png", size=24)
    credits.write(d)
    return d


def test_a_clean_build_passes_the_gate():
    """The gate has to be silent on the thing it guards, or it is noise.

    A gate with a false positive on its own project is a gate people learn to
    ignore, which is worse than no gate: the licence check runs, prints something
    and gets typed past.  So the first thing to check is that a build the demos
    actually produced says nothing at all.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = _build(tmp)
        assert credits.manifest(d), "the fixture build has no sidecars"
        assert credits.check(d) == [], credits.check(d)
        assert credits.check(str(d)) == [], "a str directory is not a Path"


def test_the_gate_reports_an_asset_nobody_recorded():
    """A picture with no sidecar is an asset whose provenance is unknown.

    This is the leak the licensing notes warn about, and it is the ordinary way one
    happens: nobody steals anything, an exporter simply forgets to write its
    envelope and a year later no one can say where the file came from.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = _build(tmp)
        (d / "mystery.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        problems = credits.check(d)
        assert any("mystery.png" in p and "no sidecar" in p for p in problems), problems
        # The credits screen is the attribution itself, and is the one picture the
        # rule does not apply to.
        assert not any(credits.SCREEN_NAME in p for p in problems), problems


def test_the_gate_reports_a_sidecar_that_says_nothing():
    """A sidecar is only worth having if it says what the asset is and whose it is."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _build(tmp)
        (d / "vague.json").write_text('{"seed": 1}', encoding="utf-8")
        (d / "vague.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (d / "silent.json").write_text(
            '{"seed": 1, "source": "", "license": "MIT"}', encoding="utf-8")
        (d / "silent.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (d / "unowned.json").write_text(
            '{"seed": 1, "source": "procedural:x"}', encoding="utf-8")
        (d / "unowned.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        problems = credits.check(d)
        assert any("vague.json" in p and "neither a source" in p for p in problems), problems
        assert any("silent.json" in p and "no source" in p for p in problems), problems
        assert any("unowned.json" in p and "no licence" in p for p in problems), problems
        # A *data* file is not a picture with a missing envelope: it says so.
        (d / "world.json").write_text('{"format": "ozart/campaign/1", "seed": 1}',
                                      encoding="utf-8")
        problems = credits.check(d)
        assert not any("world.json" in p for p in problems), problems


def test_the_gate_reports_a_source_nobody_curated():
    """An asset that is not ours has to be one of the credited ones, or it is uncredited.

    The distinction the whole module turns on: ours is regenerable from a seed and
    ``procedural:``, and anything else needs a credit line that a human wrote.  A
    sidecar naming a source the curated list has never heard of is a third party's
    file in the build with nobody to thank.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = _build(tmp)
        (d / "track.json").write_text(
            '{"seed": 1, "source": "kevin-macleod", "license": "CC BY 4.0"}',
            encoding="utf-8")
        assert credits.check(d) == [], "a curated id was refused"
        (d / "someone-else.json").write_text(
            '{"seed": 1, "source": "stock-site:8827", "license": "royalty-free"}',
            encoding="utf-8")
        problems = credits.check(d)
        assert any("someone-else.json" in p and "not one of the curated" in p for p in problems), \
            problems


def test_the_gate_reports_a_curated_source_that_cannot_be_credited():
    """The half that can be *forgotten* is checked too, and it is checked on the code.

    A source with a title and no author cannot be credited as written whatever the
    build contains, so this is a property of the module rather than of the
    directory -- which is why ``check()`` with no directory still reports it.
    """
    original = credits.SOURCES
    try:
        credits.SOURCES = original + ({"id": "a-second-track", "title": "Overture"},)
        problems = credits.check()
        assert any("a-second-track" in p and "cannot be credited" in p for p in problems), problems
        assert credits.missing_fields(), "missing_fields missed it"
    finally:
        credits.SOURCES = original
    assert credits.check() == []


def test_the_gate_reports_a_credit_line_the_build_does_not_ship():
    """The failure mode the module exists for: the file is there and the line is not.

    Forgotten attribution is what breaks a CC licence, and it happens to honest
    people -- add a track, regenerate the assets, never regenerate the block.  The
    text that ships is the text that is checked, not the text the code could make.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = _build(tmp)
        good = (d / credits.TEXT_NAME).read_text(encoding="utf-8")
        for s in credits.sources():
            assert s["attribution"] in good
        # A stale block: written before the second source was added.
        (d / credits.TEXT_NAME).write_text("ozart -- credits\n", encoding="utf-8")
        problems = credits.check(d)
        assert any("not in it verbatim" in p for p in problems), problems
        # A build with no block at all is the same violation with nothing to read.
        (d / credits.TEXT_NAME).unlink()
        problems = credits.check(d)
        assert any("is not in the build" in p for p in problems), problems
        # ...unless the caller draws the credits itself, which is a decision and not
        # an omission, so an explicit block is used instead of the file.
        assert credits.check(d, good) == []


def test_the_gate_is_an_exit_code():
    """A licence violation must not look like success to a shell script."""
    import cli

    with tempfile.TemporaryDirectory() as tmp:
        d = _build(tmp)
        assert cli.main(["credits", "--manifest", str(d), "--check"]) == 0
        (d / "mystery.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        assert cli.main(["credits", "--manifest", str(d), "--check"]) == 1


def test_the_gate_asks_a_vector_for_a_sidecar_too():
    """The rule is about *provenance*, and a curve has provenance like anything else.

    An ``.svg`` with no sidecar is a file nobody can say the origin of, which is the
    same leak a PNG with no sidecar is -- so the gate walks the pictures it knows how
    to walk, and both kinds answer to the same rule.  The credits screen stays the
    one exception, because it is the attribution rather than an asset we made.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = _build(tmp)
        (d / "panel.svg").write_text("<svg/>", encoding="utf-8")
        problems = credits.check(d)
        assert any("panel.svg" in p and "no sidecar" in p for p in problems), problems
        credits.write(d)
        (d / "panel.json").write_text(
            '{"seed": 4, "source": "procedural:gen.ui.vector", "license": "MIT"}',
            encoding="utf-8")
        problems = credits.check(d)
        assert not any("panel.svg" in p for p in problems), problems
        assert not any(credits.SCREEN_NAME in p for p in problems), problems


def test_a_sidecar_names_the_picture_that_is_actually_there():
    """A vector's sidecar must not claim a PNG beside it.

    The manifest is what the shipped credits list is built from, so an entry that
    names a file the build does not contain is a credit line pointing at nothing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "far.svg").write_text("<svg/>", encoding="utf-8")
        (d / "far.json").write_text(
            '{"seed": 1, "source": "procedural:x", "license": "MIT"}', encoding="utf-8")
        entries = credits.manifest(d)
        assert [e["image"] for e in entries] == ["far.svg"], entries
        credits.write(d)
        assert "far.svg" in (d / credits.TEXT_NAME).read_text(encoding="utf-8")


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
