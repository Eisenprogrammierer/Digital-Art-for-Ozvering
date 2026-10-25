"""A figure is a tree of bones, and a pose is one angle per bone.

Where a portrait's head is a stack of cross-sections and a letter is a skeleton a
pen thickens, a *figure* is a skeleton with a brush: seventeen joints, one bone
between a joint and the one it hangs from, and a pose is the angle of each bone.
The angles are **absolute** -- measured in the world rather than from the parent --
and that is the decision this module is built on.  A walk cycle is much easier to
write as "the thigh swings twenty degrees about straight down" than as "the thigh
swings twenty degrees relative to a pelvis that is itself leaning", and the
kinematics then accumulates *positions* down the tree instead of accumulating
rotations, which is one loop over a tuple instead of one matrix per joint.

A bone is named for **what it is**, and not for the joint at either end.  That is
worth a sentence because the first version of this file named every bone after the
joint it reached, and the result was a walk cycle that said ``hip_near`` where it
meant the thigh -- so the swing was applied to the two-hundredths-of-a-figure bone
between the pelvis and the hip joint, and the figure walked by rotating its own
hip sockets.  A skeleton with a bone called "thigh" cannot make that mistake, and a
gait is written about thighs and shins, not about the far ends of them.

The rest pose is where the proportions live.  A figure is **one unit tall** and
every bone is a fraction of that, so a figure is seven heads tall because the
numbers say so and not because a drawing was measured; a seeded jitter makes one
figure leggy and another stocky, and the jitter is a fraction of the same unit, so
a tall figure is a scaled figure and never a distorted one.

A figure is drawn in **side view** and the two limbs are therefore not left and
right: they are *near* and *far*.  Calling them ``l`` and ``r`` would be a lie now
and a bug later -- the depth cue is that the far limb is offset a little and wears
a shadow of its own material, and a name that calls the far leg "left" would make
a front view (which does have a left) look like a rename rather than a second
view.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["REST", "SEGMENTS", "PROPORTIONS", "STAND", "joints", "bones",
           "proportions", "pose", "points", "extent"]

SEGMENTS = (
    ("spine", "pelvis", "chest"),
    ("neck", "chest", "neck"),
    ("head", "neck", "head"),
    ("upper_arm_near", "chest", "shoulder_near"),
    ("forearm_near", "shoulder_near", "elbow_near"),
    ("hand_near", "elbow_near", "hand_near"),
    ("upper_arm_far", "chest", "shoulder_far"),
    ("forearm_far", "shoulder_far", "elbow_far"),
    ("hand_far", "elbow_far", "hand_far"),
    ("hip_near", "pelvis", "hip_near"),
    ("thigh_near", "hip_near", "knee_near"),
    ("shin_near", "knee_near", "foot_near"),
    ("foot_near", "foot_near", "toe_near"),
    ("hip_far", "pelvis", "hip_far"),
    ("thigh_far", "hip_far", "knee_far"),
    ("shin_far", "knee_far", "foot_far"),
    ("foot_far", "foot_far", "toe_far"),
)

REST = {
    "spine": math.pi,
    "neck": math.pi,
    "head": math.pi,

    "upper_arm_near": 0.10,
    "upper_arm_far": -0.02,
    "forearm_near": 0.0,
    "forearm_far": 0.0,

    "hand_near": -math.pi / 8.0,
    "hand_far": -math.pi / 8.0,
    "hip_near": 0.0,
    "hip_far": 0.0,
    "thigh_near": 0.0,
    "thigh_far": 0.0,
    "shin_near": 0.0,
    "shin_far": 0.0,

    "foot_near": math.pi / 2.0 - 0.12,
    "foot_far": math.pi / 2.0 - 0.12,
}

PROPORTIONS = {
    "spine": 0.270,
    "neck": 0.055,
    "head": 0.115,
    "upper_arm_near": 0.020,
    "upper_arm_far": 0.020,
    "forearm_near": 0.150,
    "forearm_far": 0.150,
    "hand_near": 0.145,
    "hand_far": 0.145,
    "hip_near": 0.020,
    "hip_far": 0.020,
    "thigh_near": 0.230,
    "thigh_far": 0.230,
    "shin_near": 0.230,
    "shin_far": 0.230,
    "foot_near": 0.075,
    "foot_far": 0.075,
}

STAND = 0.48

_JITTER = {
    "spine": 0.06,
    "neck": 0.10,
    "head": 0.05,
    "forearm": 0.05,
    "hand": 0.05,
    "thigh": 0.05,
    "shin": 0.06,
    "foot": 0.10,
}

_SIDES = ("_near", "_far")


def _pair(name: str) -> str:
    """The bone's pair: ``thigh_far`` and ``thigh_near`` are both ``thigh``."""
    for side in _SIDES:
        if name.endswith(side):
            return name[: -len(side)]
    return name


def joints() -> tuple:
    """Every joint of the figure, root first."""
    return ("pelvis",) + tuple(to for _name, _from, to in SEGMENTS)


def bones() -> tuple:
    """Every bone, as ``(name, from, to)``, in the order the kinematics walks."""
    return SEGMENTS


def proportions(seed=0, rng=None) -> dict:
    """Bone lengths for one figure, in units of its height.

    A jitter is a *style*, not a noise field: it is drawn once per *pair* of bones,
    so a figure with long legs has two long legs, and the amount is small enough
    that every figure is still seven heads tall and no figure is a grotesque.
    """
    rng = rng if rng is not None else np.random.default_rng(int(seed) + 0x51F7)
    drawn = {}
    out = {}
    for name, base in PROPORTIONS.items():
        pair = _pair(name)
        amount = _JITTER.get(pair)
        if amount is None:
            out[name] = float(base)
            continue
        if pair not in drawn:
            drawn[pair] = 1.0 + amount * (2.0 * rng.random() - 1.0)
        out[name] = float(base) * drawn[pair]
    return out


def pose(deltas: dict | None = None) -> dict:
    """The angles of a pose: the rest direction of every bone plus a delta."""
    out = dict(REST)
    if deltas:
        for name, d in deltas.items():
            if name not in out:
                raise KeyError(f"no such bone: {name}")
            out[name] = out[name] + float(d)
    return out


def points(angles: dict, lengths: dict, root=(0.0, 0.0)) -> dict:
    """The joint positions of a pose, in figure units: y is up, the sole is at 0.

    One pass down the tree.  A bone with direction ``a`` and length ``L`` reaches
    ``L*sin(a)`` forward and ``L*cos(a)`` down from its parent, which is the whole
    of the kinematics and the reason the angles are absolute.
    """
    out = {"pelvis": (float(root[0]), float(root[1]))}
    for name, start, end in SEGMENTS:
        x, y = out[start]
        a = float(angles[name])
        L = float(lengths[name])
        out[end] = (x + L * math.sin(a), y - L * math.cos(a))
    return out


def extent(angles: dict, lengths: dict, root=(0.0, 0.0)) -> tuple:
    """The box the joints of a pose occupy: ``(x0, y0, x1, y1)``."""
    pts = points(angles, lengths, root)
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    return (min(xs), min(ys), max(xs), max(ys))
