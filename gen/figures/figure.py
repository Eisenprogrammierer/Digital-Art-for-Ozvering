"""One sprite: a seed and a cycle and a phase in, a canvas out.

The three modules under this one decide *where* a figure is -- the rig gives it
proportions, the cycle gives it a gait, the body gives it clothes -- and this one is
the only place the seed is opened, so there is exactly one place where "the same
figure" is defined.  A frame is what ``picasso`` in this repository always is: a
deterministic function of a spec, drawn rather than interpolated, so the sheet comes
out of the same call with one number different and nothing is tweened between
frames.

The canvas is the *sprite box*: one world unit across, whatever the pixel size, so
a 48 px sprite and a 192 px sprite are the same drawing at two scales -- both
because the figure is drawn in world units and because every width, every line and
every margin in :mod:`gen.figures.body` is a fraction of the figure's height rather
than a number of pixels.  The consequence worth stating is that a figure is never
resampled: it is drawn at the size asked for.
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.canvas import Canvas
from core.render import LIGHT

from . import body, cycle as cycle_lib, rig

__all__ = ["FACINGS", "render", "frames", "describe", "cycles", "background"]

#: The four directions a board asks a figure to face.  ``right`` is the side view
#: the rig is drawn in; ``left`` is that view mirrored; ``down`` and ``up`` are the
#: front and back views, because on a board "towards the camera" is south and
#: "away" is north.  A game with a turning camera would need more than a billboard;
#: a board with a fixed one needs exactly these four.
FACINGS = ("right", "left", "up", "down")

_VIEW = {"right": "side", "left": "side", "up": "back", "down": "front"}


def background(canvas, seed=0) -> None:
    """A flat backdrop, for a contact sheet: dark enough that pale skin reads."""
    rng = np.random.default_rng(int(seed) + 0x9E11)
    base = palette.oklch_to_srgb((0.30 + 0.10 * rng.random(), 0.02 + 0.03 * rng.random(),
                                  float(rng.random())))
    canvas.rgb[:] = palette.srgb_to_linear(base)
    canvas.a[:] = 1.0


def render(seed=0, cycle: str = "walk", phase: float = 0.0, size: int = 96,
           tincture=None, backdrop=None, light=LIGHT,
           facing: str = "right", colours=None, garment=None,
           build: float = 1.0, hair_long: bool = False) -> Canvas:
    """One frame of one figure.  Pure: the same call is the same bytes.

    ``facing`` is one of :data:`FACINGS`.  A left-facing figure is the right-facing
    one mirrored *after* it is drawn, which is exact and costs no second pose; a
    front or back facing figure is drawn from the same skeleton with its limbs
    given a side, because a head-on view has a left and a right where a profile has
    a near and a far.  The side view's bytes are untouched: ``right`` is the
    default and the call every earlier version made.
    """
    if facing not in FACINGS:
        raise ValueError(f"a figure faces {', '.join(FACINGS)}; not {facing!r}")
    lengths = rig.proportions(seed)
    s = cycle_lib.spec(seed)
    angles = rig.pose(cycle_lib.angles(cycle, phase, s))
    root = (0.0, rig.STAND + cycle_lib.rise(cycle, phase, s))
    pts = body.to_world(rig.points(angles, lengths, root=root))
    st = body.style(seed, tincture=tincture, colours=colours, garment=garment,
                    build=build, hair_long=hair_long)
    st["light"] = light

    canvas = Canvas(int(size), int(size), scale=float(size))
    if backdrop is not None:
        background(canvas, seed if backdrop is True else backdrop)
    body.paint(canvas, pts, st, view=_VIEW[facing])
    if facing == "left":
        # A mirror is not a second drawing: flip the finished sprite, alpha and
        # colour together, so the light and the ink are mirrored with it.
        canvas.rgb = np.ascontiguousarray(canvas.rgb[:, ::-1])
        canvas.a = np.ascontiguousarray(canvas.a[:, ::-1])
    return canvas


def frames(seed=0, cycle: str = "walk", count: int = 8, size: int = 96,
           tincture=None, backdrop=None, light=LIGHT,
           facing: str = "right", colours=None, garment=None,
           build: float = 1.0, hair_long: bool = False) -> tuple:
    """A whole cycle: the frames and the phases they were drawn at.

    The phases tile one turn from 0 inclusive to ``2*pi`` exclusive, so the loop
    repeats frame zero and the sheet has no frame that duplicates another -- the
    same convention ``flags.waving`` uses, and the reason the last frame of a walk
    is one step short of a return rather than a return.
    """
    count = int(count)
    if count < 1:
        raise ValueError(f"a cycle needs at least one frame, not {count}")
    period = cycle_lib.period(cycle)
    phases = [period * (i / count) for i in range(count)]
    frames_out = [render(seed, cycle, p, size=size, tincture=tincture,
                         backdrop=backdrop, light=light, facing=facing,
                         colours=colours, garment=garment, build=build,
                         hair_long=hair_long)
                  for p in phases]
    return frames_out, phases


def cycles() -> tuple:
    """The cycles this package can draw, in the order they are worth looking at."""
    return cycle_lib.CYCLES


def describe(seed=0, tincture=None, colours=None, garment=None,
             build: float = 1.0, hair_long: bool = False) -> dict:
    """The figure as a table: its proportions, its gait, its colours.

    What an asset registry wants and a caption cannot say: the numbers behind the
    picture, so a game can rebuild this figure's walk with its own renderer, and so
    a sheet of them can be told apart without looking.
    """
    lengths = rig.proportions(seed)
    st = body.style(seed, tincture=tincture, colours=colours, garment=garment,
                    build=build, hair_long=hair_long)
    return {
        "seed": int(seed),
        "height": body.HEIGHT,
        "proportions": {k: round(float(v), 4) for k, v in sorted(lengths.items())},
        "gait": {k: round(float(v), 4) for k, v in sorted(cycle_lib.spec(seed).items())},
        "width": round(float(st["width"]), 4),
        "colours": {k: palette.srgb_to_hex(st[k])
                    for k in ("skin", "hair", "cloth", "hose", "leather", "belt", "line")},
        "cycles": list(cycle_lib.CYCLES),
        "line_px": body.LINE_PX,
    }
