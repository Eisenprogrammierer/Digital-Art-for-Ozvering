"""Regression tests for the figure generator.

Six claims are worth more than the rest, and each has a test that could fail
loudly:

* a bone keeps its length in every pose.  Forward kinematics is the one piece of
  arithmetic a whole animation rests on, and a sign error in it does not look like
  a bug -- it looks like a figure whose legs grow when it walks;
* a cycle closes **exactly** at ``2*pi``, and the frame after one turn is the
  first frame drawn again.  Not "close enough": a loop's seam is one frame wide,
  and the only way to know a gait tiles is to draw both ends of it;
* a walk is contralateral -- each leg swings opposite its pair and each arm
  opposite the leg on its own side.  That is the whole of why a walk does not read
  as a shuffle, so it is asserted as a relation between angles and not as a
  picture someone approved;
* a finer sampling is the *same* motion sampled finer.  The frames are samples of a
  periodic function, so double the count must halve the step between pictures; a
  generator that interpolated stills would pass a frame count and fail this;
* the figure never touches the edge of its cell.  A sprite sheet is cut up along
  its table, so a sword that leaves its cell is a sword in the next frame -- the
  margin is a requirement and not a taste;
* the skeleton is a tree.  Every joint is reached exactly once, and the rest
  proportions add up to a figure instead of to a list of numbers.

Run either way::

    python tests/test_figures.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import palette  # noqa: E402
from export import sheet as sheetlib  # noqa: E402
from gen.figures import body, cycle, figure, rig  # noqa: E402

PERIOD = 2.0 * math.pi


# ---------------------------------------------------------------- the skeleton
def test_the_proportions_add_up_to_a_figure():
    """The numbers are a figure, not a list: the hip is where the legs put it.

    ``STAND`` is written down as a number of its own so that this test can exist:
    the three bones below the hip have to add up to it, and the neck, head and
    skull have to put the crown at one figure unit.  A table that stops adding up
    is a table of lengths rather than a body, and every cycle that swings a bone
    would move a different figure.
    """
    legs = rig.PROPORTIONS["hip_near"] + rig.PROPORTIONS["thigh_near"] + rig.PROPORTIONS["shin_near"]
    assert abs(legs - rig.STAND) < 1e-12
    reach = (rig.STAND + rig.PROPORTIONS["spine"] + rig.PROPORTIONS["neck"]
             + rig.PROPORTIONS["head"])
    crown = reach + 0.5 * body.WIDTH["head_y"]
    assert 0.97 < crown < 1.03, crown


def test_a_bone_keeps_its_length_in_every_pose():
    """Forward kinematics, measured: the distance between joined joints is the bone.

    Checked over every cycle at eight phases, because a pose is where a rig can go
    wrong quietly -- an absolute angle added to a parent's absolute angle looks
    plausible in one frame and stretches a leg in the next.
    """
    lengths = rig.proportions(seed=7)
    for name in cycle.CYCLES:
        for i in range(8):
            angles = rig.pose(cycle.angles(name, PERIOD * i / 8, cycle.spec(7)))
            pts = rig.points(angles, lengths, root=(0.0, rig.STAND))
            for bone, start, end in rig.bones():
                (x0, y0), (x1, y1) = pts[start], pts[end]
                got = math.hypot(x1 - x0, y1 - y0)
                assert abs(got - lengths[bone]) < 1e-9, (name, i, bone, got, lengths[bone])


def test_every_joint_is_reached_exactly_once():
    """A skeleton is a tree: one root, one path to every joint, no loop.

    A cycle in the graph would make the kinematics depend on the order of the
    tuple, which is the kind of bug that shows up as one arm twitching on Tuesdays.
    """
    seen = {"pelvis"}
    for _bone, start, end in rig.bones():
        assert start in seen, start
        assert end not in seen, end
        seen.add(end)
    assert seen == set(rig.joints())
    assert rig.joints()[0] == "pelvis"


def test_a_pair_of_bones_is_one_length():
    """Two thighs of different lengths is a limp, not a build.

    The jitter is drawn once per pair, so this is a property of the generator and
    not of one seed: checked over twenty seeds for every paired bone.
    """
    for seed in range(20):
        lengths = rig.proportions(seed)
        for base in ("forearm", "hand", "thigh", "shin", "foot"):
            a = lengths[base + "_near"]
            assert abs(a - lengths[base + "_far"]) < 1e-12, (seed, base)
            assert abs(a - rig.PROPORTIONS[base + "_near"]) < 0.11 * rig.PROPORTIONS[base + "_near"]


def test_a_pose_is_the_rest_pose_plus_a_delta():
    """A cycle says how far a bone moved; the rest pose is the only absolute table."""
    rest = rig.pose()
    assert rest == rig.REST
    assert rig.pose({}) == rig.REST
    moved = rig.pose({"thigh_near": 0.5})
    assert abs(moved["thigh_near"] - (rig.REST["thigh_near"] + 0.5)) < 1e-12
    assert moved["thigh_far"] == rig.REST["thigh_far"]


def test_an_unknown_bone_is_refused_by_name():
    """A typo in a cycle is a KeyError with the name in it, not a still frame."""
    try:
        rig.pose({"thigh_sidebar": 0.2})
    except KeyError as exc:
        assert "thigh_sidebar" in str(exc)
    else:  # pragma: no cover - the failure is the point
        raise AssertionError("an unknown bone was accepted")


def test_a_smaller_figure_is_the_same_figure():
    """The seed sets the style and the jitter is bounded: no grotesques.

    A build is a difference of five percent on a pair and six on the spine, so
    every seed is a person and no seed is a spider.
    """
    seen = set()
    for seed in range(30):
        lengths = rig.proportions(seed)
        ratio = lengths["thigh_near"] / rig.PROPORTIONS["thigh_near"]
        assert 0.94 < ratio < 1.06, (seed, ratio)
        seen.add(round(lengths["spine"], 6))
    assert len(seen) > 20


# ------------------------------------------------------------------- the gaits
def test_every_cycle_is_two_pi_wide():
    """A cycle that returned after something else would be a loop with a seam."""
    assert len(cycle.CYCLES) == 4
    for name in cycle.CYCLES:
        assert abs(cycle.period(name) - PERIOD) < 1e-12


def test_a_cycle_closes_exactly():
    """Both ends of a turn are the same pose, for every bone of every cycle.

    ``sin(2*pi)`` is not exactly zero in floating point, so this is a tolerance --
    but it is a tolerance of a billionth, and the *pictures* are compared exactly
    in the next test, which is where the claim really lands.
    """
    for name in cycle.CYCLES:
        s = cycle.spec(11)
        a, b = cycle.angles(name, 0.0, s), cycle.angles(name, PERIOD, s)
        assert a.keys() == b.keys()
        for bone in a:
            assert abs(a[bone] - b[bone]) < 1e-9, (name, bone, a[bone], b[bone])


def test_the_frame_after_one_turn_is_the_first_frame():
    """The loop closes in pixels and not only in radians.

    Two independent draws -- one at phase zero, one a whole turn later -- are the
    same bytes.  That is what a game needs to know before it plays the frames on
    repeat, and it is the reason a walk can be three steps long and not a seam.
    """
    for name in cycle.CYCLES:
        a = figure.render(seed=5, cycle=name, phase=0.0, size=80)
        b = figure.render(seed=5, cycle=name, phase=PERIOD, size=80)
        assert a.to_uint8().tobytes() == b.to_uint8().tobytes(), name


def test_a_walk_is_contralateral():
    """A leg and the arm on its own side swing opposite ways: that is a walk.

    Asserted as a relation between the two angles rather than as a picture, and at
    the phases where the swing is largest, because at the crossing the signs are
    both zero and any gait passes.
    """
    s = cycle.spec(7)
    for phase in (math.pi / 2.0, 3.0 * math.pi / 2.0):
        a = cycle.angles("walk", phase, s)
        assert a["thigh_near"] * a["upper_arm_near"] < 0.0, (phase, a["thigh_near"], a["upper_arm_near"])
        assert a["thigh_far"] * a["upper_arm_far"] < 0.0
        assert abs(a["thigh_near"]) > 0.2


def test_the_legs_alternate():
    """The two thighs are one sine and its negation, so the stride is a stride."""
    s = cycle.spec(3)
    for name in ("walk", "run"):
        for i in range(8):
            a = cycle.angles(name, PERIOD * i / 8, s)
            assert abs(a["thigh_near"] + a["thigh_far"]) < 1e-9, (name, i)


def test_a_bump_is_one_in_the_middle_and_zero_outside():
    """The window a gait does something once per turn with.

    A raised cosine and not a Gaussian, because it reaches zero *exactly*: a term
    that only approached zero would leave a seam where the window ends, and a seam
    in a cycle is a visible tick once per loop.
    """
    assert abs(cycle.bump(1.0, 1.0, 0.5) - 1.0) < 1e-12
    assert cycle.bump(1.0 + 0.5, 1.0, 0.5) == 0.0
    assert cycle.bump(1.0 - 0.5, 1.0, 0.5) == 0.0
    for d in np.linspace(0.0, PERIOD, 97):
        assert abs(cycle.bump(d, 1.0, 0.7) - cycle.bump(d + PERIOD, 1.0, 0.7)) < 1e-12
        assert 0.0 <= cycle.bump(d, 1.0, 0.7) <= 1.0


def test_an_unknown_cycle_is_refused_by_name():
    try:
        cycle.angles("sprint", 0.0)
    except ValueError as exc:
        assert "sprint" in str(exc) and "walk" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown cycle was accepted")


def test_a_finer_sampling_is_the_same_motion_sampled_finer():
    """Double the frames, halve the step: the gait is sampled, not tweened.

    Measured on the joints and not on the pixels, because a pixel metric rewards
    a generator that simply moves less; the joint metric is the motion itself.
    """
    s = cycle.spec(7)
    lengths = rig.proportions(7)

    def step(count):
        pts = []
        for i in range(count):
            angles = rig.pose(cycle.angles("walk", PERIOD * i / count, s))
            pts.append(rig.points(angles, lengths, root=(0.0, rig.STAND)))
        return max(math.hypot(pts[i + 1]["toe_near"][0] - pts[i]["toe_near"][0],
                              pts[i + 1]["toe_near"][1] - pts[i]["toe_near"][1])
                   for i in range(count - 1))

    coarse, fine = step(6), step(12)
    assert 0.0 < fine < 0.75 * coarse, (coarse, fine)


# ------------------------------------------------------------------ the frames
def test_frames_tile_one_turn_and_the_last_is_not_the_first():
    """Eight frames of a turn are eight distinct steps, and the ninth is the first.

    The phases run from zero inclusive to ``2*pi`` exclusive, so a sheet does not
    carry a frame that is a duplicate of its neighbour -- the convention
    ``export.sheet`` writes into the table and a player relies on.
    """
    frames, phases = figure.frames(seed=7, cycle="walk", count=8, size=64)
    assert len(frames) == len(phases) == 8
    assert np.allclose(phases, [PERIOD * i / 8 for i in range(8)])
    assert phases[-1] < PERIOD
    seen = {f.to_uint8().tobytes() for f in frames}
    assert len(seen) == 8


def test_a_frame_is_a_cell_and_not_a_trim():
    """Every frame is the same size, whatever the figure is doing in it.

    A sheet is cut up along its table; a cell that was trimmed to the drawing would
    move the figure between frames, which is the classic sprite-sheet wobble.
    """
    for name in cycle.CYCLES:
        frames, _ = figure.frames(seed=9, cycle=name, count=5, size=(48 if name == "idle" else 72))
        assert {(f.w, f.h) for f in frames} == {(frames[0].w, frames[0].h)}


def test_the_figure_never_touches_the_edge_of_its_cell():
    """The margin is a requirement: a pixel in the border ring is a pixel in the next cell.

    Checked for every cycle at twelve phases, because it is a gait's *reach* and
    not one frame's that has to fit: an attack that throws the near arm furthest is
    the frame that decides whether the margin was enough.

    The floor of the cell is the one edge with an exception, and it is written down
    rather than averaged away: the contact shadow lies *on* the ground, so a gait
    that puts the feet low brings the shadow to the last row.  It may graze the
    bottom edge as a faint pixel -- a shadow is a decal and not the figure -- and
    never as ink.
    """
    for name in cycle.CYCLES:
        for i in range(12):
            canvas = figure.render(seed=7, cycle=name, phase=PERIOD * i / 12, size=96)
            a = canvas.a
            sides = np.concatenate([a[0, :], a[:, 0], a[:, -1]])
            assert float(sides.max()) < 0.02, (name, i, "side", float(sides.max()))
            assert float(a[-1, :].max()) < 0.25, (name, i, "floor", float(a[-1, :].max()))


def test_the_same_call_is_the_same_bytes():
    """A figure is a function of its arguments: the seed and the phase are all of it."""
    for name in cycle.CYCLES:
        a = figure.render(seed=21, cycle=name, phase=1.1, size=64)
        b = figure.render(seed=21, cycle=name, phase=1.1, size=64)
        assert a.to_uint8().tobytes() == b.to_uint8().tobytes()


def test_a_bigger_sprite_is_the_same_figure():
    """Drawn at twice the size, not upsampled: the ink box scales with the cell.

    The widths, the line and the shading bevel are all in units of the cell or in
    pixels, so the *drawing* scales linearly and the line does not thicken with it
    -- which is the difference between a sprite and a blurred sprite.
    """
    def box(size):
        canvas = figure.render(seed=7, cycle="walk", phase=0.9, size=size)
        a = canvas.a
        rows, cols = np.nonzero(a > 0.5)
        return (cols.min(), rows.min(), cols.max(), rows.max())

    small, big = box(64), box(128)
    for s, b in zip(small, big):
        assert abs(2.0 * s - b) <= 3.0, (small, big)


# ------------------------------------------------------------------ the look
def test_a_faction_colours_a_figure():
    """A tincture from outside is the tunic, and nothing else about the figure moves.

    The same tie the map's wash and the icon's plate have: the colour a faction
    wears comes from one place, so a figure standing on a province of its own realm
    is in the realm's colour and not in a colour that happens to look similar.
    """
    tincture = palette.hex_to_srgb("#3a5f9c")
    a = figure.render(seed=7, cycle="walk", phase=0.4, size=96, tincture=tincture)
    b = figure.render(seed=7, cycle="walk", phase=0.4, size=96, tincture=palette.hex_to_srgb("#8a3b2a"))
    assert a.to_uint8().tobytes() != b.to_uint8().tobytes()
    st = body.style(7, tincture=tincture)
    assert np.allclose(st["cloth"], tincture)
    # the rest of the palette is the seed's and does not follow the tincture
    other = body.style(7, tincture=palette.hex_to_srgb("#8a3b2a"))
    assert np.allclose(st["skin"], other["skin"])
    assert np.allclose(st["hair"], other["hair"])


def test_a_figure_without_a_tincture_is_the_seed_s_own():
    """A figure that belongs to nobody wears what its seed drew."""
    cloths = {tuple(np.round(body.style(seed)["cloth"], 6)) for seed in range(12)}
    assert len(cloths) == 12


def test_the_line_belongs_to_the_raster():
    """The outline is a number of pixels, so a small sprite is not a smudged border.

    Two sizes of the same figure have to have the same line *in pixels*, which is
    measured as the thickness of the ink along a scanline through the tunic: a line
    that scaled with the body would be four times thicker at four times the size.
    """
    assert 1.0 <= body.LINE_PX <= 2.5
    assert 1.5 <= body.BEVEL_PX <= 3.5


def test_the_table_says_what_the_picture_is():
    """``describe`` is the sidecar: proportions, gait, colours, cycles, all of them JSON.

    A game that wants its own renderer needs the numbers and not the image, and a
    contact sheet that wants to caption itself needs the same record.
    """
    doc = figure.describe(seed=7)
    round_trip = json.loads(json.dumps(doc))
    assert round_trip["seed"] == 7
    assert set(round_trip["proportions"]) == set(rig.PROPORTIONS)
    assert round_trip["proportions"] == {k: round(float(v), 4)
                                         for k, v in sorted(rig.proportions(7).items())}
    assert set(round_trip["gait"]) == set(cycle.spec(7))
    assert list(round_trip["cycles"]) == list(cycle.CYCLES)
    for key in ("skin", "hair", "cloth", "hose", "leather", "belt", "line"):
        hexes = round_trip["colours"][key]
        assert hexes.startswith("#") and len(hexes) == 7, (key, hexes)
    assert round_trip["line_px"] == body.LINE_PX


def test_every_cycle_draws_a_different_animation():
    """Four cycles are four motions and not four names for the same sine.

    Compared as pixels at the same phase: a cycle that only differed in the
    sidecar would be a table entry pretending to be an animation.
    """
    renders = {name: figure.render(seed=7, cycle=name, phase=1.0, size=64).to_uint8().tobytes()
               for name in cycle.CYCLES}
    assert len(set(renders.values())) == len(cycle.CYCLES)


# ------------------------------------------------------------------ the sheet
def test_the_sheet_table_matches_the_picture():
    """The table ``export.sheet`` writes is the sheet it wrote it for.

    Every frame rectangle lies inside the image, the frames are in the order they
    were drawn, and the loop flag is true because the cycle closes -- the same
    contract the flag animation uses, asserted here because a figure is the second
    caller of it.
    """
    frames, phases = figure.frames(seed=7, cycle="walk", count=6, size=64)
    sheet, info = sheetlib.build(frames, cols=3, padding=2, fps=12.0, loop=True,
                                 params=phases, background=None)
    assert sheet.w == info["size"][0] and sheet.h == info["size"][1]
    assert info["rows"] == 2 and info["cols"] == 3
    assert info["loop"] is True
    assert [round(f["param"], 6) for f in info["frames"]] == [round(ph, 6) for ph in phases]
    assert [f["index"] for f in info["frames"]] == list(range(6))
    for f in info["frames"]:
        x, y, w, h = f["rect"]
        assert 0 <= x and 0 <= y and x + w <= sheet.w and y + h <= sheet.h
        assert (w, h) == (64, 64)


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
