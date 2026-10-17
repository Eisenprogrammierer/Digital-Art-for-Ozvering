"""Regression tests for the UI toolkit.

The claim this package makes is narrow and checkable: a panel is a nine-slice
asset, so the *corners* must survive a resize byte for byte and the design must
be flat along whatever axis gets stretched.  Those two properties are tested
against the generated pixels, not against a hand-written expectation, so a change
to a style's proportions cannot quietly break the guarantee.

Run either way::

    python tests/test_ui.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import palette  # noqa: E402
from gen.ui import frame as framelib  # noqa: E402
from gen.ui import paint, slice as slicelib, style as stylelib  # noqa: E402


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
    for name in stylelib.names():
        spec = stylelib.clamp({**stylelib.get(name), "gradient": 0.0, "alpha": 1.0})
        img = paint.panel(spec, (240, 150))
        l, t, _, b = slicelib.insets(framelib.Panel(spec, (240, 150), 1.0))
        u = img.to_uint8()
        col = u[t + 2 : u.shape[0] - b - 2, l // 2]
        assert np.all(col == col[0]), f"{name}: left strip varies down the panel"
        row = u[t // 2, l + 2 : u.shape[1] - l - 2]
        assert np.all(row == row[0]), f"{name}: top strip varies across the panel"


def test_the_frame_band_leaves_every_other_pixel_transparent():
    for name in stylelib.names():
        spec = stylelib.get(name)
        img = paint.frame(spec, (200, 200)).to_uint8()
        x = max(2, int(round(spec["border"] * 0.5)))
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
    for size in [(240, 150), (64, 24), (400, 400)]:
        spec = stylelib.random_style(3)
        p = framelib.Panel(spec, size, 1.0)
        w, h = p.design_size
        assert 2.0 * p.inset <= min(w, h) + 1e-9
        assert p.spec["radius"] + p.spec["bevel"] <= p.inset + 1e-9
        assert p.spec["border"] <= p.inset + 1e-9


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
