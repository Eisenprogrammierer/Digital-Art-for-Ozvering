"""Drawing a figure: the same brush, the same ink and the same light as a portrait.

The whole point of this module is that it invents almost nothing.  A limb is a
``shape.lock`` -- the same tapered spine the portrait draws a hair clump with; the
outline is ``ink.stroke``, so the line is heavy where the limb is thick and thins
towards the hand exactly as it does down a jaw; the shadow is ``cel.shade_shape``, so
a figure and a portrait wear the same quantised Lambert band under the same light.
That is what the repository means by *consistent by construction*: it is not a
palette shared between two packages, it is one stroke and one light with a second
caller.

What a figure does not get is a face.  At 96 px an eye is two pixels and a mouth
is one, and drawing them honestly is how a sprite becomes a smudge -- the argument
:mod:`gen.icons` already makes about a sword at 32 px.  So a figure is
**silhouette, costume and colour**: the head is a skull and a hairstyle, the
costume is a tunic, a belt, hose and boots, and everything that separates one
figure from another is the proportions, the palette and the gait.

The far side is the one thing a side view adds.  Depth in a flat drawing is a
lighter line and a darker fill, so the far limb is drawn first, offset a little
down and back, and wears a shadow of its own material rather than its material.
"""
from __future__ import annotations

import numpy as np

from core import palette
from core.render import LIGHT
from gen.portraits import brush, cel, ink, shape

__all__ = ["HEIGHT", "WIDTH", "style", "parts", "paint", "to_world"]

#: The figure's height in world units, where the sprite box is one unit across.  The
#: margin that leaves -- nine percent of the box above the crown and below the soles
#: -- is what a gait swings into, and it is the number that makes the promise in
#: :func:`gen.figures.figure.render` true: an attack *dips*, and a dip that was not
#: budgeted for is a boot in the next cell of the sheet.
HEIGHT = 0.82

#: Widths, as fractions of the figure's height.  A limb is thin: the whole appeal of
#: a 96 px figure is that the silhouette is legible at four pixels per limb, which a
#: realistic thigh at eight pixels is not.
WIDTH = {
    "thigh": 0.082,
    "shin": 0.064,
    "ankle": 0.052,
    "boot": 0.074,
    "foot": 0.050,
    "upper_arm": 0.058,
    "forearm": 0.048,
    "hand": 0.044,
    "neck": 0.060,
    "waist": 0.150,
    "chest": 0.200,
    "belt": 0.040,
    "head_x": 0.126,
    "head_y": 0.142,
    "skirt": 0.172,
}

#: The ink line and the shading bevel, in **pixels** and not in fractions of the
#: figure.  A line belongs to the raster: a sprite is drawn at 32 px and at 256 px,
#: and an outline that scaled with the body would be a hairline at one and a slab at
#: the other.  Keeping these two numbers in pixels is what makes the 32 px figure a
#: small soldier rather than a smudge with a border.
LINE_PX = 1.7
BEVEL_PX = 2.2


def style(seed=0, tincture=None, rng=None) -> dict:
    """The colours and the build of one figure.

    ``tincture`` is the one input from outside: a faction's field tincture, taken
    from the same ``crest`` the map washes its provinces with and the icons dress
    their plates in, so a figure stands in a colour its own faction wears.  Without
    one the tunic is drawn from the seed, which is what a figure that belongs to
    nobody wears.
    """
    rng = rng if rng is not None else np.random.default_rng(int(seed) + 0x7C3B)
    skin = palette.oklch_to_srgb((0.70 + 0.13 * rng.random(), 0.035 + 0.045 * rng.random(),
                                  0.075 + 0.020 * rng.random()))
    hair = palette.oklch_to_srgb((0.22 + 0.30 * rng.random(), 0.02 + 0.06 * rng.random(),
                                  float(rng.random())))
    leather = palette.oklch_to_srgb((0.26 + 0.08 * rng.random(), 0.03 + 0.03 * rng.random(),
                                     0.06 + 0.03 * rng.random()))
    if tincture is not None:
        cloth = tuple(float(v) for v in tincture)
    else:
        cloth = palette.oklch_to_srgb((0.42 + 0.12 * rng.random(), 0.07 + 0.06 * rng.random(),
                                       float(rng.random())))
    # Hose and belt are the same cloth at two depths, which is what keeps a figure
    # to five colours and no more: a sprite that needs eight is a sprite whose
    # silhouette is carrying the drawing.
    return {
        "skin": skin,
        "hair": hair,
        "cloth": cloth,
        "hose": cel.shadow(cloth, depth=0.08, chroma=0.95),
        "leather": leather,
        "belt": cel.shadow(leather, depth=0.06, chroma=1.10, hue=0.03),
        "line": ink.line_colour(palette.mix(skin, cloth, 0.5), depth=0.52),
        "line_px": LINE_PX,
        "bevel_px": BEVEL_PX,
        "width": 0.90 + 0.20 * rng.random(),
        "light": LIGHT,
    }


def _mid(a, b, t: float = 0.5) -> tuple:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


#: How far a hip and a shoulder stand off the centre line in a front or back view,
#: as fractions of the figure's height.  The side view has no such distance: its
#: near and far limbs are the same limb at two depths.  A front view has a left and
#: a right, and these are the smallest numbers that read as shoulders and hips at
#: 32 px without turning the figure into a starfish.
FACE_HIP = 0.070
FACE_SHOULDER = 0.095


def _facing(pts: dict, view: str) -> dict:
    """Arrange the side rig's joints as a front or a back view.

    The rig is a side view on purpose: the near and far limbs are a depth cue, not
    a left and a right.  A board, however, asks for four directions, and the two
    the side view cannot answer are *towards* and *away* from the camera.  The
    skeleton is the same person either way, so the view is a change of the joint
    positions and not a second rig: the torso is centred, each pair of limbs is
    given a side, and the gait's forward swing becomes the small vertical step it
    would be seen as head-on.  The head is the one part that needs new geometry,
    and :func:`parts` draws it.
    """
    cx = pts["pelvis"][0]
    out = dict(pts)
    for name in ("pelvis", "chest", "neck", "head"):
        _x, y = pts[name]
        out[name] = (cx, y)
    for side, sign in (("near", 1.0), ("far", -1.0)):
        # Facing away swaps the viewer's left and right: the figure's near side is
        # on the viewer's left when the figure has turned its back.
        if view == "back":
            sign = -sign
        hx = cx + sign * FACE_HIP * HEIGHT
        _x, hy = pts["hip_" + side]
        out["hip_" + side] = (hx, hy)
        _x, ky = pts["knee_" + side]
        kx = hx + sign * 0.008 * HEIGHT
        out["knee_" + side] = (kx, ky)
        _x, ay = pts["foot_" + side]
        ax = kx + sign * 0.006 * HEIGHT
        out["foot_" + side] = (ax, ay)
        _x, ty = pts["toe_" + side]
        out["toe_" + side] = (ax + sign * 0.020 * HEIGHT, ty)
        sx = cx + sign * FACE_SHOULDER * HEIGHT
        _x, sy = pts["shoulder_" + side]
        out["shoulder_" + side] = (sx, sy)
        _x, ey = pts["elbow_" + side]
        ex = sx + sign * 0.010 * HEIGHT
        out["elbow_" + side] = (ex, ey)
        _x, hy2 = pts["hand_" + side]
        out["hand_" + side] = (ex + sign * 0.004 * HEIGHT, hy2)
    return out


def parts(pts: dict, st: dict, view: str = "side") -> dict:
    """The shapes of one pose, in world units: ``{name: Shape}``.

    Every part is a spine with a width, so a pose is drawn by handing this function
    a different set of joint positions and nothing else changes -- which is the
    whole reason the rig, the cycle and the drawing are three modules.

    ``view`` is ``side``, ``front`` or ``back``.  A side view is the rig's own; a
    front or back view first moves the joints with :func:`_facing`, widens the
    torso a little because a chest seen head-on is wider than one seen in profile,
    and draws a head with no nose.  The front and back heads differ only in what
    the hair covers -- a hairline and a face, or a whole head of hair -- because a
    figure at this size has no face to turn; the silhouette has to do the work.
    """
    if view not in ("side", "front", "back"):
        raise ValueError(f"a figure is seen from the side, front or back, not {view!r}")
    if view != "side":
        pts = _facing(pts, view)
    k = float(st["width"])
    w = {name: float(v) * HEIGHT * k for name, v in WIDTH.items()}
    if view != "side":
        w["chest"] *= 1.24
        w["waist"] *= 1.08
        w["skirt"] *= 1.12
        w["head_x"] *= 1.06
    out = {}

    for side in ("near", "far"):
        hip = pts["hip_" + side]
        knee = pts["knee_" + side]
        ankle = pts["foot_" + side]
        toe = pts["toe_" + side]
        thigh = shape.lock([hip, knee], w0=w["thigh"], w1=w["shin"]) \
            | shape.circle(0.5 * w["shin"], *knee)
        shin = shape.lock([knee, ankle], w0=w["shin"], w1=w["ankle"]) \
            | shape.circle(0.5 * w["ankle"], *ankle)
        boot = shape.lock([_mid(knee, ankle, 0.52), ankle], w0=w["boot"], w1=w["boot"] * 0.94) \
            | shape.lock([ankle, toe], w0=w["foot"], w1=w["foot"] * 0.86) \
            | shape.circle(0.5 * w["foot"], *toe)
        out["leg_" + side] = thigh | shin
        out["boot_" + side] = boot

        shoulder = pts["shoulder_" + side]
        elbow = pts["elbow_" + side]
        hand = pts["hand_" + side]
        out["sleeve_" + side] = shape.lock([shoulder, elbow], w0=w["upper_arm"],
                                           w1=w["forearm"])
        # The sleeve stops at the elbow: a bare forearm is what makes an arm read
        # against a tunic of its own colour, and it is one of the five colours.
        out["forearm_" + side] = shape.lock([elbow, hand], w0=w["forearm"],
                                            w1=w["forearm"] * 0.66)
        out["hand_" + side] = shape.circle(0.5 * w["hand"], *hand)

    pelvis, chest, neck, head = pts["pelvis"], pts["chest"], pts["neck"], pts["head"]
    # The tunic stops a little short of the chest joint, so the neck is a visible
    # column between the collar and the jaw: a head that sits straight on the
    # shoulders is the one proportion that reads as a doll at every size.
    collar = _mid(pelvis, chest, 0.90)
    out["tunic"] = shape.lock([pelvis, collar], w0=w["waist"], w1=w["chest"],
                              bulge=0.55 * (w["chest"] - w["waist"])) \
        | shape.circle(0.46 * w["chest"], *collar)
    hem = (pelvis[0], pelvis[1] - HEIGHT * 0.085)
    out["skirt"] = shape.lock([_mid(pelvis, chest, -0.04), hem],
                              w0=w["waist"] * 1.04, w1=w["skirt"])
    # The belt is laid across the torso's own axis, so it follows a lean instead of
    # staying level while the body it is tied to tips forward.
    ax, ay = chest[0] - pelvis[0], chest[1] - pelvis[1]
    n = float(np.hypot(ax, ay)) or 1.0
    nx, ny = -ay / n, ax / n
    half = 0.5 * w["waist"]
    out["belt"] = shape.lock([(pelvis[0] + nx * half, pelvis[1] + ny * half),
                              (pelvis[0] - nx * half, pelvis[1] - ny * half)],
                             w0=w["belt"] * 1.15, w1=w["belt"] * 1.15)

    out["neck"] = shape.lock([_mid(chest, neck, 1.0), _mid(head, neck, 0.35)],
                             w0=w["neck"] * 1.05, w1=w["neck"] * 0.85)
    rx, ry = 0.5 * w["head_x"], 0.5 * w["head_y"]
    if view == "side":
        skull = shape.ellipse(rx, ry, head[0], head[1]) \
            | shape.lock([head, (head[0] + rx * 0.52, head[1] - ry * 0.62)],
                         w0=ry * 0.88, w1=ry * 0.52) \
            | shape.circle(0.13 * ry, head[0] + rx * 0.92, head[1] - ry * 0.02)
        out["head"] = skull
        # The hair is one closed loop over the crown and the nape whose lower edge
        # crosses the forehead: the line it draws there *is* the hairline, so the
        # fringe costs no second shape and the face is whatever the loop does not
        # cover.
        out["hair"] = shape.poly(shape.smooth_loop([
            (head[0] + rx * 0.58, head[1] + ry * 0.50),
            (head[0] - rx * 0.02, head[1] + ry * 1.15),
            (head[0] - rx * 0.90, head[1] + ry * 0.70),
            (head[0] - rx * 1.20, head[1] - ry * 0.05),
            (head[0] - rx * 1.00, head[1] - ry * 0.80),
            (head[0] - rx * 0.30, head[1] - ry * 1.00),
            (head[0] - rx * 0.25, head[1] - ry * 0.55),
            (head[0] - rx * 0.12, head[1] - ry * 0.02),
        ], per_segment=6, resample=72))
        return out

    # A head seen head-on has no nose: the profile's nose is the one feature that
    # would turn a front view back into a side view.  The jaw is a second, lower
    # ellipse, because a single oval reads as an egg at this size.
    out["head"] = shape.ellipse(rx, ry, head[0], head[1]) \
        | shape.ellipse(rx * 0.78, ry * 0.46, head[0], head[1] - ry * 0.48)
    if view == "front":
        # A cap with a fringe: the face is the skin the loop leaves uncovered.  The
        # loop is the same idea as the side view's hairline, asked of a head that
        # has two temples instead of a profile.
        out["hair"] = shape.poly(shape.smooth_loop([
            (head[0] - rx * 1.06, head[1] + ry * 0.08),
            (head[0] - rx * 0.92, head[1] + ry * 0.78),
            (head[0], head[1] + ry * 1.10),
            (head[0] + rx * 0.92, head[1] + ry * 0.78),
            (head[0] + rx * 1.06, head[1] + ry * 0.08),
            (head[0] + rx * 0.72, head[1] + ry * 0.30),
            (head[0], head[1] + ry * 0.18),
            (head[0] - rx * 0.72, head[1] + ry * 0.30),
        ], per_segment=6, resample=72))
    else:
        # The back of a head is the whole head of hair, with a small bun at the
        # crown to say which side of the skull the hair is tied on.  No face: there
        # is nothing to see from here, and a pair of eyes on the back would be a
        # worse lie than a silhouette.
        out["hair"] = shape.ellipse(rx * 1.07, ry * 1.05, head[0], head[1]) \
            | shape.ellipse(rx * 0.30, ry * 0.30, head[0], head[1] + ry * 0.92)
    return out


def to_world(points_fig: dict, height: float = HEIGHT) -> dict:
    """Figure units -- one unit tall, the soles at 0 -- into the sprite's box.

    The box is one unit across and the figure is centred in it, which is what makes
    a sprite sheet's cells align: nothing in the drawing depends on where the feet
    landed this frame.
    """
    return {name: (x * height, (y - 0.5) * height) for name, (x, y) in points_fig.items()}


def paint(canvas, pts: dict, st: dict, ground: float | None = None,
          view: str = "side") -> None:
    """Fill, band and ink every part, far side first.

    The order is the depth order and it is fixed: a far limb is never in front of a
    near one whatever the pose, because that is what "near" means.  In a front or
    back view there is no far limb -- both are at the same depth -- so both are
    painted in their own colour and the draw order alone decides which overlaps
    which.
    """
    p = parts(pts, st, view=view)
    side = view == "side"
    light = st["light"]
    line = float(st["line_px"]) / max(float(canvas.scale), 1e-6)
    bevel = float(st["bevel_px"]) / max(float(canvas.scale), 1e-6)

    def fill(s, colour, depth: float = 0.0, far: bool = False):
        base = cel.shadow(colour, depth=depth) if far else colour
        surf = brush.Surface(canvas, s)
        surf.paint(canvas, base)
        if far:
            surf.bands(canvas, [cel.shadow(base, depth=0.10)], (0.44,), light=light,
                       bevel=bevel, soft=0.8)
        else:
            surf.bands(canvas, [cel.shadow(base)], (0.42,), light=light, bevel=bevel,
                       soft=0.8)
        surf.stroke(canvas, line, st["line"] if not far else cel.shadow(st["line"], depth=0.06),
                    inset=line * 0.45)

    if ground is None:
        ground = min(pts["toe_near"][1], pts["toe_far"][1]) - 0.02 * HEIGHT
    # The shadow is placed inside the cell and not under it: a contact shadow that
    # leaves the cell is a band in the next frame of the sheet, and the figure is
    # promised to fit its cell at every phase.
    ry = 0.042 * HEIGHT
    cy = max(ground, -0.5 + ry + 0.010)
    brush.paint(canvas, shape.ellipse(0.24 * HEIGHT, ry, pts["pelvis"][0], cy),
                (0.06, 0.06, 0.09), alpha=0.22, soft=2.0)

    fill(p["leg_far"], st["hose"], depth=0.12 if side else 0.0, far=side)
    fill(p["boot_far"], st["leather"], depth=0.12 if side else 0.0, far=side)
    fill(p["sleeve_far"] | p["forearm_far"] | p["hand_far"], st["skin"],
         depth=0.12 if side else 0.0, far=side)

    fill(p["tunic"] | p["skirt"], st["cloth"])
    fill(p["belt"], st["belt"])

    fill(p["leg_near"], st["hose"])
    fill(p["boot_near"], st["leather"])
    fill(p["neck"], st["skin"])
    fill(p["head"], st["skin"])
    fill(p["hair"], st["hair"])
    # The near side catches the light and the far side wears a shadow: that is the
    # whole of depth in a flat drawing, and it is why the near sleeve is a lighter
    # cloth than the tunic it lies on and not the same cloth.
    fill(p["sleeve_near"] | p["forearm_near"] | p["hand_near"], st["skin"])
