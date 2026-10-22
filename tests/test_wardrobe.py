"""Regression tests for the standalone wardrobe and decoration generators.

These assets are not portraits, so the invariants are different: the object must
exist inside its tile, its silhouette must change with its category, and a pinned
colour must reach the pixels without changing the geometry.  The tests render at
a small size because that is where the generator has to stay legible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen.wardrobe import awards, garment  # noqa: E402


def _alpha(canvas) -> np.ndarray:
    return canvas.to_uint8()[..., 3]


def _rgb(canvas) -> np.ndarray:
    return canvas.to_uint8()[..., :3].astype(np.int16)


def test_every_garment_cut_renders_inside_its_tile():
    for cut in garment.CUTS:
        spec = garment.random_spec(5, cut=cut)
        c = garment.render(spec, 96)
        a = _alpha(c)
        assert a.max() > 200, cut
        ys, xs = np.where(a > 40)
        assert xs.min() > 0 and ys.min() > 0 and xs.max() < c.w - 1 and ys.max() < c.h - 1, cut
        # The garment is a tall object, not a smear across the tile.
        assert (ys.max() - ys.min()) > (xs.max() - xs.min()) * .45, cut


def test_garment_cut_changes_the_silhouette():
    base = garment.random_spec(9, cut="shirt")
    shirt = garment.render(base, 96)
    coat = garment.render(dict(base, cut="coat"), 96)
    assert not np.array_equal(_alpha(shirt), _alpha(coat))
    mantle = garment.render(dict(base, cut="mantle"), 96)
    # A cape has no sleeves; its widest row is much lower than a jacket's.
    def width(c):
        a = _alpha(c)
        return (a > 40).sum()
    assert width(mantle) < width(shirt) * 1.15


def test_garment_pins_reach_colour_and_pattern_without_reseeding_geometry():
    base = garment.random_spec(11, cut="jacket")
    a = garment.render(base, 96)
    b = garment.render(dict(base, cloth="#102030", trim="#d0b060"), 96)
    assert not np.array_equal(_rgb(a), _rgb(b))
    # The silhouette is still the jacket's; only the cloth changed.
    assert np.array_equal(_alpha(a), _alpha(b))
    c = garment.render(dict(base, pattern="check"), 96)
    assert np.array_equal(_alpha(a), _alpha(c))
    assert not np.array_equal(_rgb(a), _rgb(c))


def test_every_award_kind_renders_inside_its_tile():
    for kind in awards.KINDS:
        spec = awards.random_spec(7, kind=kind)
        c = awards.render(spec, 96)
        a = _alpha(c)
        assert a.max() > 200, kind
        ys, xs = np.where(a > 40)
        assert xs.min() > 0 and ys.min() > 0 and xs.max() < c.w - 1 and ys.max() < c.h - 1, kind


def test_award_metal_and_ribbon_are_independent_controls():
    base = awards.random_spec(13, kind="medal")
    gold = awards.render(dict(base, metal="gold", ribbon="#7a1f2a"), 96)
    silver = awards.render(dict(base, metal="silver", ribbon="#7a1f2a"), 96)
    other_ribbon = awards.render(dict(base, metal="gold", ribbon="#1f3a7a"), 96)
    assert not np.array_equal(_rgb(gold), _rgb(silver))
    assert not np.array_equal(_rgb(gold), _rgb(other_ribbon))


def test_garment_and_award_specs_are_clamped_and_deterministic():
    assert garment.clamp({"cut": "nonesuch", "buttons": 99})["cut"] in garment.CUTS
    assert garment.clamp({"cut": "nonesuch", "buttons": 99})["buttons"] == 8
    assert awards.clamp({"kind": "nonesuch", "metal": "tin"})["kind"] in awards.KINDS
    a = garment.render(garment.random_spec(3, cut="coat"), 64)
    b = garment.render(garment.random_spec(3, cut="coat"), 64)
    assert a.to_uint8().tobytes() == b.to_uint8().tobytes()
    c = awards.render(awards.random_spec(3, kind="star"), 64)
    d = awards.render(awards.random_spec(3, kind="star"), 64)
    assert c.to_uint8().tobytes() == d.to_uint8().tobytes()
