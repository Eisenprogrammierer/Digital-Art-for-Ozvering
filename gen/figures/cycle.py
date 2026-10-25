"""A gait is a periodic function of phase, and a cycle is that function sampled.

The flag animation had to fight for its loop: the cloth's second harmonic is not
commensurate with its first, so the fold pattern returns only after ten turns and
the loop had to be cut where it actually closed.  A gait has no such trouble, and
that is the point of building it this way -- every term below is a sine, a cosine
or a *window* of one, so every term is exactly ``2*pi``-periodic and the loop closes
by construction rather than by measurement.  Sampling it uniformly then gives frames
that are steps of one motion, which is the property ``export.sheet`` was built on:
a finer sampling is smaller steps and never a different walk.

The cycle speaks in **deltas**: a cycle returns how far each bone has moved from
its rest direction, and never where a bone *is*.  That keeps a cycle readable as a
sentence about walking -- "the thigh swings thirty-eight hundredths of a radian
about straight down, and the knee bends most halfway through the swing" -- and it
keeps the rest pose in one place, which is what lets a jittered figure and a plain
one share a gait.

Four cycles, and they differ in shape rather than in degree.  ``idle`` is a breath
and a sway that never quite returns to where it started; ``walk`` is a
contralateral gait -- a leg and the opposite arm swing together, which is the whole
of why a walk does not read as a shuffle; ``run`` is the same gait with a flight
phase, an airborne body and a lean; ``attack`` is not periodic at all in its nature,
so it is written as a wind-up, a strike and a recovery inside one turn, which is how
a game plays a one-shot and then falls back into the idle.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["CYCLES", "PERIOD", "spec", "angles", "rise", "period", "bump"]

PERIOD = 2.0 * math.pi

CYCLES = ("idle", "walk", "run", "attack")


def _wrap(d):
    """The shortest signed distance along the circle: ``p - centre``, folded."""
    return (d + math.pi) % PERIOD - math.pi


def bump(phase: float, centre: float, width: float):
    """A smooth window of the circle, 1 at ``centre`` and 0 outside ``width``.

    This is the building block the gaits are written with, and it is what lets a
    cycle *do something once per turn* -- a knee bending, a foot leaving the
    ground, a sword arriving -- without leaving the periodic world.  A raised
    cosine rather than a Gaussian because it reaches zero exactly, so a term built
    from it cannot leave a seam where the window ends; the width is the half-width
    of the window in radians.
    """
    t = _wrap(phase - centre) / max(float(width), 1e-6)
    if t <= -1.0 or t >= 1.0:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * t))


def spec(seed=0, rng=None) -> dict:
    """The amplitudes of one figure's gait: how it walks, not where it is.

    A stride, a swing, a knee, a bob and a lean.  Range rather than a single
    number because a figure with a long stride and a figure that shuffles are two
    characters, and the seed is what tells them apart -- the same seed that picks
    the proportions.
    """
    rng = rng if rng is not None else np.random.default_rng(int(seed) + 0x2A11)
    def u(lo, hi):
        return float(lo + (hi - lo) * rng.random())
    return {
        "stride": u(0.34, 0.50),
        "armswing": u(0.20, 0.34),
        "knee": u(0.55, 0.80),
        "ankle": u(0.12, 0.24),
        "bob": u(0.010, 0.022),
        "lean": u(0.02, 0.08),
    }


def _walk(phase: float, s: dict) -> dict:
    p = phase
    stride, arm = s["stride"], s["armswing"]
    knee, ankle, lean = s["knee"], s["ankle"], s["lean"]
    return {

        "spine": -lean + 0.018 * math.sin(2.0 * p),
        "neck": 0.4 * lean,
        "head": 0.5 * lean - 0.012 * math.sin(2.0 * p),

        "thigh_near": stride * math.sin(p),
        "thigh_far": -stride * math.sin(p),
        "shin_near": -knee * bump(p, 0.15, 1.00),
        "shin_far": -knee * bump(p, math.pi + 0.15, 1.00),

        "foot_near": -ankle * bump(p, -math.pi / 2.0, 0.95)
                     + 0.55 * ankle * bump(p, math.pi / 2.0, 0.80),
        "foot_far": -ankle * bump(p, math.pi / 2.0, 0.95)
                    + 0.55 * ankle * bump(p, -math.pi / 2.0, 0.80),
        "upper_arm_near": -arm * math.sin(p),
        "upper_arm_far": arm * math.sin(p),
        "forearm_near": 0.26 - 0.12 * math.sin(p),
        "forearm_far": 0.26 + 0.12 * math.sin(p),
    }


def _run(phase: float, s: dict) -> dict:
    p = phase
    stride, arm = s["stride"] * 1.7, s["armswing"] * 1.9
    knee, ankle, lean = s["knee"] * 1.6, s["ankle"] * 1.4, s["lean"] * 2.2
    return {

        "spine": -lean + 0.03 * math.sin(2.0 * p),
        "neck": 0.45 * lean,
        "head": 0.55 * lean - 0.02 * math.sin(2.0 * p),
        "thigh_near": stride * math.sin(p),
        "thigh_far": -stride * math.sin(p),
        "shin_near": -knee * bump(p, 0.0, 1.35) - 0.30,
        "shin_far": -knee * bump(p, math.pi, 1.35) - 0.30,
        "foot_near": -ankle * bump(p, -math.pi / 2.0, 1.0)
                     + 0.55 * ankle * bump(p, math.pi / 2.0, 0.9),
        "foot_far": -ankle * bump(p, math.pi / 2.0, 1.0)
                    + 0.55 * ankle * bump(p, -math.pi / 2.0, 0.9),
        "upper_arm_near": -arm * math.sin(p) - 0.10,
        "upper_arm_far": arm * math.sin(p) - 0.10,
        "forearm_near": 0.85 - 0.20 * math.sin(p),
        "forearm_far": 0.85 + 0.20 * math.sin(p),
    }


def _idle(phase: float, s: dict) -> dict:
    p = phase
    return {

        "spine": -0.015 + 0.008 * math.cos(p),
        "neck": 0.010 * math.cos(p),
        "head": 0.012 * math.cos(p) + 0.006 * math.sin(2.0 * p),
        "thigh_near": 0.030 * math.sin(p),
        "thigh_far": -0.020 * math.sin(p),
        "shin_near": -0.05 - 0.04 * bump(p, math.pi, 1.5),
        "shin_far": -0.05 - 0.04 * bump(p, 0.0, 1.5),
        "foot_near": 0.02 * math.sin(p),
        "foot_far": -0.02 * math.sin(p),
        "upper_arm_near": -0.030 * math.sin(p),
        "upper_arm_far": 0.030 * math.sin(p),
        "forearm_near": 0.20 - 0.03 * math.cos(p),
        "forearm_far": 0.20 + 0.03 * math.cos(p),
    }


def _attack(phase: float, s: dict) -> dict:
    p = phase

    wind = bump(p, 4.1, 1.5)
    strike = bump(p, 0.5, 1.3)
    follow = bump(p, 1.5, 1.6)
    lean = 0.10 + s["lean"]
    return {
        "spine": -lean * 0.6 - 0.16 * strike + 0.10 * wind,
        "neck": 0.04 + 0.05 * strike,
        "head": 0.05 - 0.06 * strike + 0.04 * wind,
        "thigh_near": 0.30 * strike - 0.10 * wind,
        "thigh_far": -0.22 * strike + 0.08 * wind,
        "shin_near": -0.35 * strike - 0.15 * wind,
        "shin_far": -0.20 - 0.30 * strike,

        "foot_near": -0.10 - 0.16 * wind,
        "foot_far": -0.10 + 0.22 * strike,
        "upper_arm_near": -1.30 * wind + 0.95 * strike - 0.35 * follow,
        "forearm_near": 0.55 + 0.55 * wind - 0.40 * strike,
        "upper_arm_far": 0.60 * wind - 0.45 * strike,
        "forearm_far": 0.35 - 0.30 * wind,
    }

_CYCLES = {"idle": _idle, "walk": _walk, "run": _run, "attack": _attack}

_RISE = {
    "walk": lambda p, s: -0.5 * s["bob"] * (1.0 - math.cos(2.0 * p)),
    "run": lambda p, s: -0.5 * s["bob"] * 2.4 * (1.0 - math.cos(2.0 * p)) + 0.5 * s["bob"] * math.cos(2.0 * p),
    "idle": lambda p, s: -0.35 * s["bob"] * (1.0 - math.cos(p)),
    "attack": lambda p, s: -0.012 - 0.030 * bump(p, 0.5, 1.4) + 0.012 * bump(p, 4.1, 1.5),
}


def get(cycle: str):
    """The named cycle, refused by name rather than guessed at."""
    try:
        return _CYCLES[str(cycle)]
    except KeyError:
        raise ValueError(f"no such cycle: {cycle!r} (have {', '.join(CYCLES)})") from None


def angles(cycle: str, phase: float, s: dict | None = None) -> dict:
    """The deltas of one pose: how far every bone has moved from its rest."""
    return get(cycle)(float(phase), s if s is not None else spec())


def rise(cycle: str, phase: float, s: dict | None = None) -> float:
    """The pelvis's vertical offset at a phase, in figure heights (down is -)."""
    return float(_RISE[str(cycle)](float(phase), s if s is not None else spec()))


def period(cycle: str) -> float:
    """The phase span in which a cycle returns: the same number for all of them.

    Here to be asked, and it is asked by the tests: a cycle that returned after
    something other than ``2*pi`` would be a cycle whose loop has a seam, and the
    answer being a constant is what makes an animation of any length tileable.
    """
    get(cycle)
    return PERIOD
