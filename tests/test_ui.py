"""Regression tests for the UI toolkit.

The claim this package makes is narrow and checkable: a panel is a nine-slice
asset, so the *corners* must survive a resize byte for byte and the design must
be flat along whatever axis gets stretched.  Those two properties are tested
against the generated pixels, not against a hand-written expectation, so a change
to a style's proportions cannot quietly break the guarantee.

A material body is the one look that gives the second property up, so it is not
tested against a promise either: the panel is drawn twice at two sizes and the
cost of the resize is measured, which is the number the sidecar carries.

Run either way::

    python tests/test_ui.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import palette  # noqa: E402
from export import svg  # noqa: E402
from gen.ui import frame as framelib  # noqa: E402
from gen.ui import material, paint, slice as slicelib, style as stylelib  # noqa: E402
from gen.ui import vector as vectorlib  # noqa: E402


# ------------------------------------------------------------------ slicing
def test_the_nine_rectangles_tile_the_image_exactly():
    # Not "roughly": a gap or an overlap of one pixel shows up on a stretched
    # control as a seam, so the arithmetic has to be exact for every size.
    for size in [(1, 1), (64, 24), (100, 100), (301, 97)]:
        for ins in [(4, 4, 4, 4), (0, 0, 0, 0), (8, 2, 8, 2)]:
            if ins[0] + ins[2] > size[0] or ins[1] + ins[3] > size[1]:
                continue
            cover = np.zeros((size[1], size[0]), dtype=int)
            for x0, y0, x1, y1 in slicelib.cuts(size, ins):
                cover[y0:y1, x0:x1] += 1
            assert cover.min() == 1 and cover.max() == 1, (size, ins)


def test_cuts_refuse_an_impossible_size():
    for size, ins in [((10, 10), (8, 8, 8, 8)), ((10, 4), (2, 2, 2, 4))]:
        try:
            slicelib.cuts(size, ins)
        except ValueError:
            continue
        raise AssertionError(f"cuts accepted {ins} in {size}")


def test_stretching_keeps_the_corners_byte_for_byte():
    spec = stylelib.get("slate")
    p = framelib.Panel(spec, (200, 200), 1.0)
    ins = slicelib.insets(p)
    src = paint.panel(spec, (200, 200)).to_uint8()
    for size in [(300, 200), (200, 320), (420, 380)]:
        out = slicelib.stretch_pixels(src, size, ins)
        assert out.shape == (size[1], size[0], 4)
        kept = 0
        for (sx0, sy0, sx1, sy1), (dx0, dy0, dx1, dy1) in zip(
                slicelib.cuts((200, 200), ins), slicelib.cuts(size, ins)):
            if (sx1 - sx0, sy1 - sy0) == (dx1 - dx0, dy1 - dy0):
                # A block that is not resampled is not merely close, it is the
                # same bytes -- corners and, for a one-axis stretch, whole edges.
                assert np.array_equal(out[dy0:dy1, dx0:dx1], src[sy0:sy1, sx0:sx1])
                kept += 1
        assert kept >= 4, "the corners were resampled"


def test_stretching_to_the_same_size_is_the_identity():
    spec = stylelib.get("moss")
    c = paint.panel(spec, (128, 128))
    ins = slicelib.insets(framelib.Panel(spec, (128, 128), 1.0))
    out = slicelib.stretch(c, (128, 128), ins)
    assert np.array_equal(out.to_uint8(), c.to_uint8())


def test_the_border_does_not_fatten_when_the_middle_stretches():
    # A flat style, so the only thing that varies along a row is the geometry.
    # The measurement is the *width of the frame* along the middle row -- where
    # the body stops being the band -- before and after the resample.
    spec = stylelib.clamp({**stylelib.get("slate"), "gradient": 0.0})
    c = paint.panel(spec, (160, 96))
    ins = slicelib.insets(framelib.Panel(spec, (160, 96), 1.0))
    body = np.asarray(spec["body"]) * 255.0

    def band_width(img, row=48):
        px = np.asarray(img.to_uint8()[row, :, :3], dtype=float)
        hit = np.abs(px - body).max(axis=1) < 6.0
        return int(np.argmax(hit)) if hit.any() else -1

    wide = slicelib.stretch(c, (400, 96), ins)
    assert band_width(c) > 0, "the frame band was not found to begin with"
    assert band_width(wide) == band_width(c), "the frame band changed width"
    assert band_width(slicelib.stretch(c, (400, 240), ins)) == band_width(c)


def test_stretch_refuses_to_shrink_past_its_corners():
    spec = stylelib.get("brass")
    c = paint.frame(spec, (120, 80))
    ins = slicelib.insets(framelib.Panel(spec, (120, 80), 1.0))
    lo = slicelib.minimum(ins)
    try:
        slicelib.stretch(c, (lo[0] - 1, 200), ins)
    except ValueError:
        pass
    else:
        raise AssertionError("stretch accepted a width smaller than the corners")


# ------------------------------------------------------------------ design
def test_the_strips_of_a_panel_are_flat_along_the_stretched_axis():
    # The real precondition, tested on pixels: outside the corner boxes the
    # design may vary *along* the stretch axis but not across it.  This catches
    # an ornament that has grown out of its corner, a rule that curves too far,
    # and a shadow that is not separable -- all at once.
    #
    # Both a plain panel and the same panel as a titled window are walked, because
    # the second is where the four sides stop being interchangeable: the strips are
    # no longer the same size, and the sample ranges have to come from the insets
    # rather than from one number.
    cases = []
    for name in stylelib.names():
        spec = stylelib.get(name)
        cases.append((name, spec))
        cases.append((f"{name} + title bar", stylelib.title_bar(spec, 24.0)))
    for name, spec in cases:
        # `material: none` on purpose: a material body is the one look in the
        # package that gives this property up, deliberately and statedly.  It is
        # not let off the hook -- `test_the_stretch_error_measures_what_the_knife_does`
        # draws it twice and prices the stretch -- but the flat-strip rule is a
        # statement about a stretchable middle, and a material has not got one.
        spec = stylelib.clamp({**spec, "gradient": 0.0, "alpha": 1.0, "material": "none"})
        img = paint.panel(spec, (240, 150))
        l, t, r, b = slicelib.insets(framelib.Panel(spec, (240, 150), 1.0))
        u = img.to_uint8()
        col = u[t + 2 : u.shape[0] - b - 2, l // 2]
        assert np.all(col == col[0]), f"{name}: left strip varies down the panel"
        row = u[t // 2, l + 2 : u.shape[1] - r - 2]
        assert np.all(row == row[0]), f"{name}: top strip varies across the panel"


def test_the_frame_band_leaves_every_other_pixel_transparent():
    for name in stylelib.names():
        spec = stylelib.get(name)
        img = paint.frame(spec, (200, 200)).to_uint8()
        x = max(2, int(round(min(spec["border"]) * 0.5)))
        assert img[100, 100, 3] == 0, f"{name}: the hollow middle is not transparent"
        assert img[100, x, 3] > 200, f"{name}: the band is not drawn"


def test_a_panel_fills_its_own_middle():
    spec = stylelib.get("parchment")
    img = paint.panel(spec, (200, 200)).to_uint8()
    assert img[100, 100, 3] == 255
    assert tuple(img[100, 100, :3]) != (0, 0, 0)


def test_a_state_changes_the_colour_and_not_the_layout():
    # A control that shifts by a pixel as the pointer crosses it looks broken,
    # and this is the test that keeps it from ever happening.
    spec = stylelib.get("brass")
    shots = [paint.button(spec, (180, 56), state=s) for s in paint.STATES]
    for a in shots:
        assert np.array_equal(a.a, shots[0].a)
    assert not np.array_equal(shots[0].to_uint8(), shots[1].to_uint8())
    means = [palette.srgb_to_oklch(np.asarray(s.to_uint8())[28, 90, :3] / 255.0)[0] for s in shots]
    assert means[1] > means[0] > means[2], "hover/pressed are not lighter/darker"


def test_a_pressed_control_is_lit_from_the_other_side():
    # `core.render.shade(inset=True)` is the whole of the effect, and the
    # signature is the sliver of the frame next to the *outer* edge: on a raised
    # moulding it faces up towards the light at the top of the control and away
    # from it at the bottom; knocked in, the two swap.  Sampling a sliver on
    # either side of the ridge keeps the moulding's own two halves apart, and
    # comparing top to bottom *within* one state keeps the colour change out.
    spec = stylelib.clamp({**stylelib.get("slate"), "border": 8.0, "bevel": 4.0})
    for state, top_is_dark in (("idle", False), ("pressed", True)):
        lum = paint.button(spec, (200, 80), state=state).to_uint8()[..., :3].mean(axis=2)
        top = float(lum[1:4, 60:140].mean())
        bottom = float(lum[76:79, 60:140].mean())
        assert (top < bottom) == top_is_dark, f"{state}: top {top:.1f} vs bottom {bottom:.1f}"


# ------------------------------------------------------------------ styles
def test_every_named_style_is_on_the_legal_set():
    for name in stylelib.names():
        spec = stylelib.get(name)
        assert not stylelib.validate(spec), f"{name}: {stylelib.validate(spec)}"
        for size in [(240, 150), (128, 40), (48, 20)]:
            fit = stylelib.clamp(spec, size)
            bad = stylelib.validate(fit, size)
            assert not bad, f"{name} at {size}: {bad}"


def test_seeded_styles_are_legal_and_varied():
    seen = set()
    for seed in range(40):
        spec = stylelib.random_style(seed)
        assert not stylelib.validate(spec), f"seed {seed}: {stylelib.validate(spec)}"
        for size in [(240, 150), (96, 32)]:
            bad = stylelib.validate(stylelib.clamp(spec, size), size)
            assert not bad, f"seed {seed} at {size}: {bad}"
        seen.add(palette.srgb_to_hex(spec["body"]))
    assert len(seen) > 30, "the seeded styles are not varied"
    # Some of those kits must be titled windows, or the per-side path would never
    # be drawn outside its own test -- and the point of drawing it from the seed is
    # that it is one code path, not a special case.
    titled = [s for s in range(40) if len(set(stylelib.random_style(s)["border"])) > 1]
    assert len(titled) >= 3, titled


def test_the_corner_ornament_stays_in_its_corner_box():
    # The geometric promise, sampled directly: outside the corner box every
    # field has to be positive, or the ornament would smear when stretched.
    for seed in range(25):
        spec = stylelib.random_style(seed)
        p = framelib.Panel(spec, (240, 150), 1.0)
        if not p.has_boss():
            continue
        xs = np.linspace(-p.hw, p.hw, 241)
        ys = np.linspace(-p.hh, p.hh, 151)
        X, Y = np.meshgrid(xs, ys)
        for i in range(4):
            x0, y0, x1, y1 = p.corner_box(i)
            outside = (X < x0) | (X > x1) | (Y < y0) | (Y > y1)
            reach = p.boss(i, X, Y) + p.boss_outline(i)
            assert reach[outside].min() > -1e-9, f"seed {seed}, corner {i}"


def test_the_boss_radius_function_is_bounded_by_its_amplitude():
    theta = np.linspace(-np.pi, np.pi, 721)
    for harmonics in (3, 5, 8, 12):
        for amp in (0.0, 0.12, 0.30):
            r = framelib.boss_radius(theta, 10.0, harmonics, amp)
            assert np.isclose(r.max(), 10.0 * (1.0 + amp), atol=0.05 * 10.0 * amp + 1e-9)
            assert np.all(np.abs(r - 10.0) <= 10.0 * amp + 1e-9)


def test_boss_reach_and_limit_are_inverse():
    for inset, border, amp in [(18.0, 5.0, 0.12), (12.0, 4.0, 0.30), (30.0, 6.5, 0.0)]:
        limit = stylelib.boss_limit(inset, border, amp)
        assert limit > 0.0
        assert stylelib.boss_reach(inset, border, limit, amp) <= inset + 1e-9
        assert stylelib.boss_reach(inset, border, limit * 1.05, amp) > inset - 1e-9


def test_the_corner_box_is_never_wider_than_the_geometry_allows():
    # Per side now, because a window with a title bar is legal exactly when the
    # *top* has room for the bar and the sides have room for what they carry.
    for size in [(240, 150), (64, 24), (400, 400)]:
        spec = stylelib.random_style(3)
        p = framelib.Panel(spec, size, 1.0)
        w, h = p.design_size
        ins, bor = p.insets, spec["border"]
        assert ins[0] + ins[2] <= w + 1e-9, (size, ins)
        assert ins[1] + ins[3] <= h + 1e-9, (size, ins)
        assert spec["radius"] + spec["bevel"] <= min(ins) + 1e-9
        assert all(b <= i + 1e-9 for b, i in zip(bor, ins)), (bor, ins)


def test_content_stays_inside_the_panel():
    for name in stylelib.names():
        for size in [(200, 120), (200, 200), (64, 40)]:
            p = framelib.Panel(stylelib.get(name), size, 1.0)
            x0, y0, x1, y1 = p.content()
            assert 0 <= x0 < x1 <= size[0]
            assert 0 <= y0 < y1 <= size[1]


# ------------------------------------------------------------------ render
def test_scaling_only_multiplies_pixels():
    # A style is resolution-independent: the 2x render has to be the same design,
    # drawn twice as wide.  It is not bit-identical -- anti-aliasing is one pixel
    # wide at every density, which is the point -- so the comparison is on the
    # coverage, where a real difference in geometry would show up immediately.
    spec = stylelib.get("moss")
    one = paint.panel(spec, (320, 200), scale=1.0).to_uint8()
    two = paint.panel(spec, (640, 400), scale=2.0).to_uint8()
    assert two.shape == (400, 640, 4)
    down = np.asarray(two[::2, ::2], dtype=float)
    assert np.abs(down[:, :, 3] - one[:, :, 3]).mean() < 6.0
    assert abs(down[:, :, 3].mean() - one[:, :, 3].mean()) < 1.0


def test_rendering_is_deterministic_and_byte_identical():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name in ("brass", "glass"):
            for j in (0, 1):
                p = Path(tmp) / f"{name}{j}.png"
                paint.button(stylelib.get(name), (160, 48), state="hover", scale=2).save(p)
                paths.append(p)
        h = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        assert h[0] == h[1], "button render is not byte-identical"
        assert h[2] == h[3], "glass render is not byte-identical"
        assert all(p.stat().st_size > 0 for p in paths)


def test_bad_kind_and_state_are_rejected():
    spec = stylelib.get("slate")
    for kw in ({"kind": "blob"}, {"state": "melted"}):
        try:
            paint.draw(spec, (64, 32), **kw)
        except ValueError:
            continue
        raise AssertionError(f"paint accepted {kw}")


def test_round_trip_through_rgba_survives():
    spec = stylelib.get("parchment")
    c = paint.panel(spec, (96, 64))
    back = slicelib.from_rgba(c.to_uint8(), c.scale)
    assert np.array_equal(back.to_uint8(), c.to_uint8())


def test_a_title_bar_is_a_band_and_not_a_second_panel():
    """A window with a title bar is the same window with a wider top band.

    Both numbers have to move -- the band has to be deep enough for the bar and the
    corner box deep enough for the band -- so the helper moves both, and the three
    sides it is not about come back untouched.
    """
    base = stylelib.get("brass")
    titled = stylelib.title_bar(base, 26.0)
    assert not stylelib.validate(titled), stylelib.validate(titled)
    assert titled["border"][1] == 26.0
    assert titled["border"][0] == base["border"][0]
    assert titled["border"][3] == base["border"][3]
    assert titled["inset"][1] >= 26.0
    assert titled["inset"][0] == base["inset"][0]
    # A floor, not a ceiling: a style that already reserves more keeps it.
    assert stylelib.title_bar(base, 2.0)["border"][1] == base["border"][1]
    for wrong in ("left", "sideways"):
        try:
            stylelib.title_bar(base, 12.0, edge=wrong)
        except ValueError:
            continue
        raise AssertionError(f"a title bar went {wrong}")


def test_the_band_is_as_thick_as_it_says_on_each_side():
    """Measured on a hollow frame, where the band is the only opaque thing.

    The claim is the whole point of the per-side numbers, and a scalar design would
    fail it by construction: the top band of a titled window is deeper than its
    sides, in pixels, at whatever size it is drawn.  The measurement is the run of
    opaque pixels from the edge, which is the band and nothing else.
    """
    spec = stylelib.title_bar(stylelib.get("slate"), 24.0)

    def run(mask):
        i = 0
        while i < len(mask) and mask[i]:
            i += 1
        return i

    for size, scale in [((240, 160), 1.0), ((480, 320), 2.0)]:
        img = paint.frame(spec, size, scale=scale).to_uint8()
        col = img[:, img.shape[1] // 2, 3] > 200
        row = img[img.shape[0] // 2, :, 3] > 200
        top, left = run(col), run(row)
        assert abs(top - spec["border"][1] * scale) <= 2, (size, top)
        assert abs(left - spec["border"][0] * scale) <= 2, (size, left)
        assert top > left + 4, "the top band is not the title bar"


def test_the_corners_of_a_titled_window_are_blocks_and_not_squares():
    """The nine-slice cuts are four numbers, and a titled panel is why.

    The top-left corner block is as wide as the left band and as tall as the bar,
    so stretching the window sideways must leave both that block *and* the strip
    below it untouched -- which a square-corner assumption would resample.
    """
    spec = stylelib.title_bar(stylelib.get("parchment"), 22.0)
    p = framelib.Panel(spec, (200, 150), 1.0)
    ins = slicelib.insets(p)
    assert ins[1] > ins[0], ins
    src = paint.panel(spec, (200, 150)).to_uint8()
    for size in [(340, 150), (200, 260), (360, 300)]:
        out = slicelib.stretch_pixels(src, size, ins)
        kept = 0
        for (sx0, sy0, sx1, sy1), (dx0, dy0, dx1, dy1) in zip(
                slicelib.cuts((200, 150), ins), slicelib.cuts(size, ins)):
            if (sx1 - sx0, sy1 - sy0) == (dx1 - dx0, dy1 - dy0):
                assert np.array_equal(out[dy0:dy1, dx0:dx1], src[sy0:sy1, sx0:sx1])
                kept += 1
        assert kept >= 4, (size, kept)


def test_a_titled_window_puts_its_corner_boxes_where_the_cuts_are():
    """The ornament box and the nine-slice block are the same rectangle.

    That identity is what lets an engine cut the image at the insets and know the
    ornament is not in the pieces it is about to stretch -- and with a title bar
    those pieces stop being squares, which is the case that would catch a
    `min()` hiding in the corner arithmetic.
    """
    for seed in range(30):
        spec = stylelib.random_style(seed)
        p = framelib.Panel(spec, (240, 160), 1.0)
        for i in range(4):
            x0, y0, x1, y1 = p.corner_box(i)
            ix, iy = stylelib._corner_pair(i, p.insets)
            assert np.isclose(min(x1 - x0, y1 - y0), min(ix, iy))
            assert np.isclose(max(x1 - x0, y1 - y0), max(ix, iy))
        if not p.has_boss():
            continue
        xs = np.linspace(-p.hw, p.hw, 241)
        ys = np.linspace(-p.hh, p.hh, 161)
        X, Y = np.meshgrid(xs, ys)
        for i in range(4):
            x0, y0, x1, y1 = p.corner_box(i)
            outside = (X < x0) | (X > x1) | (Y < y0) | (Y > y1)
            reach = p.boss(i, X, Y) + p.boss_outline(i)
            assert reach[outside].min() > -1e-9, f"seed {seed}, corner {i}"


# ---------------------------------------------------------------- material
def test_a_material_changes_the_body_and_nothing_else():
    """A material is a body, so the frame's pixels stay the frame's.

    The same rule a button state obeys, for the same reason: a window that moved
    when it was made of stone would be a layout change wearing a material's name.
    The band is opaque and painted after the body, so every fully opaque band pixel
    comes back bit-identical -- which is also the safety of drawing the material
    only where the band does not cover it.
    """
    plain = stylelib.get("slate")
    grained = stylelib.clamp({**plain, "material": "wood", "material_scale": 24.0})
    a = paint.panel(plain, (240, 150)).to_uint8()
    b = paint.panel(grained, (240, 150)).to_uint8()
    assert np.array_equal(a[..., 3], b[..., 3]), "the material moved the silhouette"
    band = paint.frame(plain, (240, 150)).to_uint8()[..., 3] > 250
    assert band.sum() > 1000, "there was no band to measure"
    # Identical but for the last bit of the compositor's rounding: the band is
    # opaque, so what is under it cannot show through except by a 1/255 tail.
    diff = np.abs(a[band].astype(np.int32) - b[band].astype(np.int32))
    assert diff.max() <= 1, (band.sum(), diff.max())
    body = np.abs(a[~band].astype(np.int32) - b[~band].astype(np.int32))
    assert body.max() > 8, "the body did not change at all"


def test_material_none_is_exactly_no_material():
    """``none`` is a real member of the set, and it means the old behaviour.

    The stretchable panel is not a material with its strength turned down: it is
    the panel this package drew before materials existed, so a kit that does not
    ask for one cannot have changed by a single byte.
    """
    spec = stylelib.get("moss")
    assert np.array_equal(paint.panel(spec, (160, 100)).to_uint8(),
                          paint.panel({**spec, "material": "none"}, (160, 100)).to_uint8())
    X, Y = np.meshgrid(np.arange(4.0), np.arange(4.0))
    assert np.array_equal(material.height(X, Y, spec), np.zeros((4, 4)))
    colour = np.full((4, 4, 3), 0.42)
    assert np.array_equal(material.body(colour, spec, X, Y, 1.0, paint.LIGHT), colour)


def test_wood_has_a_grain_direction_and_stone_has_not():
    """The two recipes are told apart by their fields, not by their names.

    Wood is a stack of rings, so the steps across the grain are several times the
    steps along it.  Stone is cells, and a plate has no direction, so the same two
    measurements agree.  Both would still *look* like a material from across the
    room if this went wrong, which is exactly why it is worth a test.
    """
    n = 200
    axis = np.linspace(0.0, n / 20.0, n)          # 20 design px per feature
    X, Y = np.meshgrid(axis, axis)
    ratios = {}
    for kind, seed in (("wood", 11), ("stone", 23)):
        spec = {"material": kind, "material_scale": 20.0, "material_seed": seed}
        h = material.height(X, Y, spec)
        assert h.min() >= 0.0 and h.max() <= 1.0, (kind, h.min(), h.max())
        dy = float(np.abs(np.diff(h, axis=0)).mean())
        dx = float(np.abs(np.diff(h, axis=1)).mean())
        ratios[kind] = dy / max(dx, 1e-9)
        assert not np.allclose(h, material.height(X, Y, {**spec, "material_seed": seed + 1})), \
            f"{kind} ignores its seed"
    assert ratios["wood"] > 2.0, ratios
    assert 0.6 < ratios["stone"] < 1.6, ratios


def test_a_material_is_the_same_material_at_any_scale():
    """The grain lives in design pixels, so 2x resolves it and does not enlarge it.

    A material's feature size is a number in the style, like the frame width, and
    from outside the only way to see it is to draw the panel twice.  Done in
    *device* pixels instead -- the mistake this test exists to catch -- the 2x
    render comes out twice as coarse and the mean difference across the panel jumps
    from an anti-aliasing fuzz to a different texture.
    """
    for name in ("timber", "granite"):
        spec = stylelib.get(name)
        one = paint.panel(spec, (200, 130), scale=1.0).to_uint8()
        two = paint.panel(spec, (400, 260), scale=2.0).to_uint8()
        folded = two.astype(np.float64).reshape(130, 2, 200, 2, 4).mean(axis=(1, 3))
        d = np.abs(folded[..., :3] - one[..., :3])
        assert d.mean() < 4.0, (name, d.mean())
        assert np.percentile(d, 99) < 14.0, (name, np.percentile(d, 99))


def test_the_stretch_error_measures_what_the_knife_does():
    """A resize is drawn twice and the disagreement is the number.

    This is the measurement the sidecar carries.  A panel whose middle is flat
    resamples into something very close to a panel drawn at that size; a panel with
    a material body does not, because the grain in the strips changes size while
    the grain in the corners beside them does not.  Nothing is declared here -- the
    two pictures are compared, and the flat panel is the control.
    """
    def cost(spec):
        dw, dh = 240, 150
        small = (144, 90)
        full = paint.panel(spec, (dw, dh)).to_uint8()
        was = paint.panel(spec, small)
        ins = slicelib.insets(framelib.Panel(spec, small, 1.0))
        return slicelib.stretch_error(
            full, slicelib.stretch_pixels(was.to_uint8(), (dw, dh), ins))

    plain = cost(stylelib.get("slate"))
    assert plain < slicelib.STRETCHABLE, plain
    for name in ("timber", "granite"):
        grained = cost(stylelib.get(name))
        assert grained > 3.0 * plain, (name, plain, grained)
        assert grained > slicelib.STRETCHABLE, (name, grained)


def test_an_unknown_material_is_reported_and_refused():
    """A bad *name* is not a number out of range, so it is refused rather than clamped.

    ``clamp`` repairs numbers; it does not invent an identifier.  So a spec that
    asks for a material nobody wrote has to stay wrong when it comes back out, and
    the field has to refuse to draw it rather than quietly fall back to a gradient.
    """
    spec = {**stylelib.get("slate"), "material": "marble"}
    assert stylelib.validate(spec), "validate accepted an unknown material"
    assert stylelib.validate(stylelib.clamp(spec)), "clamp invented a material"
    X, Y = np.meshgrid(np.arange(3.0), np.arange(3.0))
    try:
        material.height(X, Y, spec)
    except ValueError:
        return
    raise AssertionError("an unknown material was drawn")


# ---------------------------------------------------------- the material cache
def test_a_buttons_three_states_are_one_material():
    """The cache, measured as counts: a state is a colour, so the shape is shared.

    This is the claim that makes a material affordable in a UI that redraws on
    hover, and it is asserted as a count rather than as elapsed time because a
    stopwatch is a machine's opinion and a hit count is a fact about the program.
    """
    material.clear()
    spec = stylelib.get("timber")
    for state in paint.STATES:
        paint.button(spec, (160, 100), state=state)
    assert material.stats() == {"hits": 2, "misses": 1, "size": 1}, material.stats()


def test_a_cached_material_is_bit_identical_to_an_uncached_one():
    """A cache that is wrong is worse than no cache, so this compares pixels.

    The arithmetic is unchanged only if the remembered value *is* the value that
    would have been computed, and the cheapest proof of that is the picture.  Every
    style, three sizes and all three states, drawn with a cold cache and again with
    a warm one.
    """
    for name in ("timber", "granite", "slate", "moss"):
        spec = stylelib.get(name)
        for size in ((160, 100), (240, 150), (97, 61)):
            material.clear()
            hot = [paint.button(spec, size, state=s).to_uint8() for s in paint.STATES]
            material.clear()
            for state, warm in zip(paint.STATES, hot):
                material.clear()
                cold = paint.button(spec, size, state=state).to_uint8()
                assert np.array_equal(warm, cold), \
                    f"{name}/{size}/{state}: the cache changed the picture"


def test_the_material_key_includes_the_window_and_the_light():
    """The key is the whole input, and the window is part of it.

    A material is anchored to the world -- a wider panel shows *more* wood rather
    than the same wood stretched -- so two panels of different sizes sample
    different windows and a key of ``(kind, scale, seed)`` would slide the grain by
    half a panel.  The fields themselves are compared here rather than the cache's
    internals: a cache that got this wrong would draw the wrong picture, so the
    picture is the test.
    """
    spec = {"material": "wood", "material_scale": 20.0, "material_seed": 11}
    xs = np.linspace(0.0, 3.0, 40)
    ys = np.linspace(0.0, 3.0, 30)
    dx = float(xs[1] - xs[0])
    here = material.modulation("wood", spec, *np.meshgrid(xs, ys), dx, paint.LIGHT)[0]
    there = material.modulation("wood", spec, *np.meshgrid(xs + 2.0, ys), dx, paint.LIGHT)[0]
    assert not np.allclose(here, there), "a different window was answered with the same field"
    # ...and each is remembered as itself, not as the other.
    assert np.array_equal(here, material.modulation("wood", spec, *np.meshgrid(xs, ys), dx, paint.LIGHT)[0])
    assert np.array_equal(there, material.modulation("wood", spec, *np.meshgrid(xs + 2.0, ys), dx, paint.LIGHT)[0])
    # The light and the y direction are in the key for the same reason: they change
    # the shading without changing the recipe.
    down = material.modulation("wood", spec, *np.meshgrid(xs, ys), dx, paint.LIGHT, False)[0]
    assert not np.allclose(here, down), "y_up was not part of the key"


def test_the_material_cache_is_bounded_and_can_be_cleared():
    """A cache with no bound is a leak wearing a cache's name."""
    material.clear()
    assert material.stats() == {"hits": 0, "misses": 0, "size": 0}
    spec = {"material": "stone", "material_scale": 18.0, "material_seed": 5}
    for i in range(material.CACHE_MAX + 5):
        xs = np.linspace(float(i), float(i) + 2.0, 12)
        material.modulation("stone", spec, *np.meshgrid(xs, np.linspace(0.0, 2.0, 12)),
                            0.05, paint.LIGHT)
    s = material.stats()
    assert s["size"] <= material.CACHE_MAX, s
    assert s["misses"] == material.CACHE_MAX + 5, s
    material.clear()
    assert material.stats() == {"hits": 0, "misses": 0, "size": 0}


# ------------------------------------------------------------------- the vector
def _field(panel, key):
    """The field a named curve is the zero set of, as a function of world points."""
    def f(x, y):
        if key == "outer":
            return panel.outer(x, y)
        if key == "inner":
            return panel.inner(x, y)
        if key == "rule":
            # `Panel.rule` is a *stroke*: its own zero set is where the line has no
            # width, so the centreline is where it reads -rule/2.
            return panel.rule(x, y) + 0.5 * panel.spec["rule"]
        return panel.boss(int(key[-1]), x, y)
    return f


def test_every_vector_curve_lies_on_its_field():
    """The guarantee that makes a vector export safe: there is only one geometry.

    A curve is built from ``Panel``'s own numbers -- the same object the raster
    thresholds -- so every point it was built from has to sit on the zero set of the
    field the pixels come from.  If this ever fails, the SVG and the PNG have
    started to disagree about where the panel is, and no amount of shading being
    absent excuses that.

    The sizes include a panel smaller than its own frame, which is where a corner
    radius can outgrow its box and the field stops being a distance field at all.
    Measured across every style, five sizes and three scales, the worst point on any
    curve is 6.3e-14 -- four orders below the bound, which is what ``exact`` means
    for a float.
    """
    specs = [stylelib.get(name) for name in stylelib.names()]
    specs.append(stylelib.title_bar(stylelib.get("parchment"), 30.0))
    worst = 0.0
    for spec in specs:
        for size in ((240, 150), (97, 61), (512, 320), (40, 28), (9, 9)):
            for scale in (1.0, 2.0, 0.5):
                for kind in paint.KINDS:
                    panel = framelib.Panel(spec, size, scale)
                    for key, curve in vectorlib.curves(panel, kind).items():
                        pts = np.array(curve.on_curve)
                        f = _field(panel, key)
                        err = float(np.max(np.abs(f(pts[:, 0], pts[:, 1]))))
                        worst = max(worst, err)
    # Floating-point noise, not a tolerance: the curve *is* the zero set.
    assert worst < 1e-9, worst


def test_the_vector_path_is_that_curve_in_screen_coordinates():
    """SVG is y-down and the world is y-up, and that flip happens exactly once.

    Every point the path is *made of* comes back in the package's world, because
    that is the frame the fields live in, so the check is that each of those points,
    flipped, is literally a token in the ``d`` attribute.  A flip that is missed or
    applied twice is a mirrored asset, and a mirrored scallop is the kind of error
    only a picture shows -- unless it is written down as a coordinate, which this
    does.  (The arc samples are checked against the field instead: they are on the
    curve but not in the text.)
    """
    spec = stylelib.get("parchment")
    for size in ((240, 150), (97, 61)):
        panel = framelib.Panel(spec, size, 1.0)
        for kind in paint.KINDS:
            for key, curve in vectorlib.curves(panel, kind).items():
                for x, y in curve.vertices:
                    token = f"{svg.num(x)},{svg.num(-y)}"
                    assert token in curve.d, (key, token, curve.d[:120])


def test_the_ornament_really_is_unmirrored():
    """The teeth of the test above: an un-flipped scallop does not sit on the field.

    ``boss_radius`` is a Fourier series with a phase, so it is *not* symmetric under
    negation -- which is why the flip is a real piece of the geometry and not
    bookkeeping.  If this ever became symmetric, the coordinate test would quietly
    stop being able to see a missing flip, so the asymmetry is asserted rather than
    assumed.
    """
    panel = framelib.Panel(stylelib.get("slate"), (240, 150), 1.0)
    harmonics, amp = panel.spec["boss_harmonics"], panel.spec["boss_amp"]
    bx, by = panel.boss_centre(0)
    r0 = panel.boss_radius_at(0)
    wrong = svg.polar(bx, by, lambda t: framelib.boss_radius(t, r0, harmonics, amp))[1]
    pts = np.array([(x, -y) for x, y in wrong])
    err = float(np.max(np.abs(panel.boss(0, pts[:, 0], pts[:, 1]))))
    assert err > 0.05, err


def test_the_vector_band_is_the_silhouette_with_its_hole_punched_out():
    """A hollow frame is a ring, and the ring is one even-odd path.

    Two separately filled paths would leave a band you could see the body through,
    so the hole is punched with ``fill-rule="evenodd"`` instead -- and the middle of
    a frame is the place that proves it: the raster's band is out there, and the
    document's body layer is not.
    """
    spec = stylelib.get("slate")
    panel = framelib.Panel(spec, (240, 150), 1.0)
    cs = vectorlib.curves(panel, "frame")
    d = vectorlib.band(cs)
    assert cs["outer"].d in d and cs["inner"].d in d
    assert panel.body(0.0, 0.0) > 0.0, "the middle of this panel is not a hole"

    doc = vectorlib.document(spec, (240, 150), kind="frame")
    assert f'<path d="{d}" fill=' in doc
    assert 'fill-rule="evenodd"' in doc
    assert "url(#body)" not in doc, "a hollow frame grew a body"


def test_a_panel_too_small_for_its_own_frame_has_no_hole():
    """Where the band stops being a ring, and the two answers have to agree.

    ``clamp`` keeps a four-pixel floor for a design's corner box, so a three-pixel
    design can carry a four-pixel band and the inner box turns inside out.  The
    raster's band is then the whole silhouette (its ``max`` cannot carve a hole), and
    the vector has to say the same thing rather than cut a hole out of nothing.
    """
    spec = stylelib.get("brass")
    fitted = 0
    for size in ((9, 9), (17, 17), (40, 28), (240, 150)):
        for scale in (1.0, 3.0):
            panel = framelib.Panel(spec, size, scale)
            _, _, hw, hh = panel.inner_box()
            assert 0.0 <= panel.outer_radius() <= min(panel.hw, panel.hh)
            assert 0.0 <= panel.inner_radius() <= max(min(hw, hh), 0.0)
            assert 0.0 <= panel.rule_radius() <= max(min(hw, hh) - panel.spec["rule_gap"], 0.0)
            if not panel.has_hole():
                fitted += 1
                cs = vectorlib.curves(panel, "frame")
                assert "inner" not in cs and "rule" not in cs
                assert vectorlib.band(cs) == cs["outer"].d
    assert fitted, "no size in the sweep was small enough to exercise this"


def test_a_material_body_has_no_vector_form():
    """A noise field is refused rather than exported flat.

    An export that silently drops the one thing it cannot draw is worse than an
    error, because it ships: the SVG would claim to be the panel and would not be.
    So the refusal is the feature, and the message has to name the material and say
    where to go instead.
    """
    spec = stylelib.get("timber")
    assert spec["material"] != "none"
    try:
        vectorlib.document(spec, (160, 100))
    except ValueError as e:
        assert spec["material"] in str(e) and "png" in str(e).lower(), str(e)
    else:
        raise AssertionError("a material was exported as if it had curves")
    # And the same style, with the material taken off, is a panel like any other.
    plain = {**spec, "material": "none"}
    assert "<svg" in vectorlib.document(plain, (160, 100))


def test_the_vector_ramp_is_the_rasters_ramp_sampled():
    """The one approximation, measured instead of hoped for.

    SVG interpolates between stops in sRGB and the palette interpolates in OKLab, so
    a ramp is written as sampled stops.  What is claimed is a *bound* on the error,
    so the stops are compared with the ramp itself at a hundred points between them
    and the worst disagreement is asserted.  Seventeen stops is comfortably enough:
    measured across every style and state on the shelf, the worst is 1.4e-5 in OKLab
    for a step the test still allows a hundred times over -- if a style ever grew a
    ramp that sampled badly, the number here would move and this would fail.
    """
    steps = vectorlib.STOPS
    worst = 0.0
    for name in stylelib.names():
        spec = stylelib.get(name)
        for state in paint.STATES:
            body, low, _, _ = paint.tones(spec, state)
            colours = [body, palette.mix(body, low, spec["gradient"])]
            stops = np.asarray(palette.ramp(colours, steps))
            t = np.linspace(0.0, 1.0, 101)
            exact = palette.ramp_map(colours, t)
            f = np.clip(t * (steps - 1), 0.0, steps - 1 - 1e-12)
            i = f.astype(int)
            frac = (f - i)[:, None]
            linear = stops[i] * (1.0 - frac) + stops[i + 1] * frac
            worst = max(worst, max(palette.delta_e(a, b) for a, b in zip(exact, linear)))
    assert worst < 1e-3, worst


def test_a_vector_state_is_a_colour_and_not_a_layout():
    """The package's oldest invariant, stated where it is exactly true.

    A state is a colour, never a layout, and in a vector that claim is at its
    plainest: the three states share every ``d`` attribute and differ only in paint.
    The raster only *nearly* obeys this -- ``pressed`` also turns its bevel round --
    so the vector is the clean statement of the rule the raster is shaped by.
    """
    spec = stylelib.get("parchment")
    docs = [vectorlib.document(spec, (200, 120), state=state) for state in paint.STATES]
    geometry = [re.findall(r'd="([^"]*)"', doc) for doc in docs]
    assert geometry[0] == geometry[1] == geometry[2]
    assert len(set(docs)) == 3, "the three states drew the same picture"


def test_one_vector_serves_every_dpi():
    """Same design, different output size, identical paths -- that is the payoff.

    The document's user units are design pixels and its width and height are output
    pixels, so a panel drawn at one design size is *the same file* whether it is
    handed out at 100 pixels or 200.  An engine rescales it by changing two numbers;
    nothing is cut, nothing is inset, and there is no mip chain to carry.  This is
    the opposite trade from the nine-slice, and it is why the export exists.
    """
    spec = stylelib.get("moss")
    small = vectorlib.document(spec, (100, 60), scale=1.0)
    large = vectorlib.document(spec, (200, 120), scale=2.0)
    assert re.findall(r'd="([^"]*)"', small) == re.findall(r'd="([^"]*)"', large)
    assert 'width="100" height="60"' in small
    assert 'width="200" height="120"' in large
    assert re.findall(r'viewBox="([^"]*)"', small) == re.findall(r'viewBox="([^"]*)"', large)


def test_the_vector_is_deterministic_and_writes_where_it_is_told():
    spec = stylelib.get("slate")
    a = vectorlib.document(spec, (240, 150), scale=2)
    b = vectorlib.document(spec, (240, 150), scale=2)
    assert a == b, "the same panel wrote two different files"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nested" / "panel.svg"
        out = vectorlib.write(spec, (240, 150), path, scale=2)
        assert Path(out) == path
        assert path.read_text(encoding="utf-8") == a


def test_the_svg_writer_is_compact_and_never_writes_minus_zero():
    """A generated file is read by people, so the numbers are the small kind."""
    assert svg.num(0.0) == "0"
    assert svg.num(-0.0) == "0"
    assert svg.num(-0.0001) == "0"
    assert svg.num(3.0000000000000004) == "3"
    assert svg.num(-2.5) == "-2.5"
    assert svg.num(1.0 / 3.0) == "0.333"


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
