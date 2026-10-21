"""Style presets: the proportions that separate chibi from seinen.

A style is a bag of numbers in *head units*, where the head is exactly 1.0 tall
-- so every number below is a fraction of the head, and a style can be re-rendered
anywhere without rescaling anything.  Two conventions are fixed by the frame
rather than by the style:

* the chin sits at ``y = -0.5`` and the crown at ``y = +0.5``;
* ``y`` points up, ``x`` to the character's left (the viewer's right).

Two parameters do most of the work: the eye size and *where the eyes sit*.  A
chibi face puts huge eyes low on a round head; a seinen face puts smaller eyes on
the midline of a longer one.  The vertical layout is therefore not a free
parameter either -- the brows, nose and mouth are all *measured from the eyes*
downward, because that is the relationship the eye reads as a person rather than
as a mask.  ``brow_gap`` is the distance from the upper lid to the brow and
``nose_gap``/``mouth_gap`` are the same idea below; the concrete ``y`` values are
filled in by :func:`Style` so no preset can contradict them.

The derivation is not only a guard against contradictory presets, it is the reason
an expression can move a whole face with two numbers: move ``eye_y`` and the gaps
and every landmark moves with them (:meth:`gen.portraits.expression.warp`).  The
converse is that a number which is an *anchor* rather than a conclusion -- the
hairline, which roots on the skull -- cannot be set through :meth:`Style.with_`,
because the derivation would silently undo it; that is what :meth:`Style.pinned`
is for.

Colour is deliberately *not* here.  A style says how big a feature is; the DNA
(:mod:`gen.portraits.dna`) says what colour it is, so any hairstyle can wear any
hair colour and the two axes stay independent.
"""
from __future__ import annotations

__all__ = ["Style", "STYLES", "get", "names"]

#: Shared starting point: a mid-shounen head.  Every preset is a delta on this.
DEFAULTS = dict(
    # -- silhouette (right half, crown -> chin) --------------------------
    # ``head_w``/``wide_y`` are the temple, ``cheek_w``/``cheek_y`` the cheekbone
    # below the eye line, which is where a face is widest; ``jaw_w`` is the jaw
    # angle and ``chin_w`` the width of the chin's rounded end.
    head_w=0.390,
    top_w=0.305,
    top_y=0.455,
    wide_y=0.095,
    cheek_w=0.360,
    cheek_y=-0.190,
    jaw_w=0.286,       # half-width at the jaw angle
    jaw_y=-0.305,
    chin_w=0.118,
    chin_y=-0.500,
    # The head as a *solid*: ``head_d`` is its half-depth at the deepest, as a
    # multiple of its own half-width, and that one number is the whole of what a
    # style adds to the turn model.  One is a head as deep as it is wide -- round,
    # and their turned outlines do not change width; more is a deeper skull, which
    # widens its outline towards profile and slides its features further.
    head_d=1.07,
    # -- eyes (both, in head units) --------------------------------------
    eye_dx=0.190,      # half the centre-to-centre distance
    eye_y=-0.060,
    eye_w=0.172,       # width of the eye opening
    eye_up=0.064,      # upper lid apex above the corner chord
    eye_down=0.045,
    eye_tilt=5.0,      # degrees, outer corner raised
    iris_r=0.058,      # iris half-width
    iris_squash=1.22,  # iris height / width
    pupil_r=0.024,
    hl_r=0.017,        # big highlight radius
    lash=0.023,        # upper lash thickness at its heaviest
    lid_w=0.010,       # lower lid line
    crease_w=0.011,
    lid_shadow=0.42,   # how far the lid shadow reaches down the eye
    # -- the rest of the face, measured down from the eyes ---------------
    brow_gap=0.055,    # upper lid -> brow
    brow_dx=0.194,
    brow_w=0.184,
    brow_thick=0.033,
    brow_arch=0.025,
    brow_tilt=4.0,     # degrees, outer end raised
    nose_gap=0.042,    # lower lid -> nose
    nose_w=0.085,
    nose_h=0.064,
    nose_style="wedge",   # "dot" | "wedge" | "line"
    mouth_gap=0.132,   # nose -> mouth
    mouth_w=0.198,
    mouth_curve=0.018,
    mouth_style="line",   # "line" | "full"
    lip_full=0.10,     # extra lower-lip volume, in mouth widths
    ear_size=1.0,
    blush_gap=0.062,   # lower lid -> blush
    hairline_gap=0.300,  # eye -> hairline
    # -- finish -----------------------------------------------------------
    line=1.0,          # global line-weight multiplier
    ink_w=0.020,       # outer contour weight
    blush=0.28,
    lid_shadow_alpha=0.42,
    chin_shadow=0.055,
    # -- bust -------------------------------------------------------------
    # The frame is 1.02 half-heights tall and the head is 1.0, so the space under
    # the chin is 0.55: the shoulder line has to sit in it, and the neck has to
    # emerge from the notch with its own visible length, or the head sits on a
    # stalk.  ``shoulder_w`` is wide enough to leave the frame -- a bust whose
    # corner is visible reads as a pill no matter how the slope is shaped.
    neck_w=0.190,
    neck_y=-0.720,     # where the neck meets the shoulders
    shoulder_w=1.250,
    shoulder_y=-1.760,  # the shoulder polygon's lower edge, well below the frame
    shoulder_slope=0.0,
)

#: Presets.  Read them as a table of proportions, not as arbitrary numbers:
#: the full face is about 4.0 eye widths across in chibi, 4.5 in shoujo,
#: 4.2 in shounen and 5.0 in seinen (a real adult face is about 5).
STYLES = {
    "chibi": dict(
        head_w=0.450, top_w=0.390, top_y=0.450, wide_y=0.095,
        cheek_w=0.410, cheek_y=-0.170, jaw_w=0.348, jaw_y=-0.262, chin_w=0.198,
        head_d=0.98,
        eye_dx=0.195, eye_y=-0.105, eye_w=0.225, eye_up=0.088, eye_down=0.065,
        eye_tilt=4.0, iris_r=0.078, iris_squash=1.06, pupil_r=0.030, hl_r=0.020,
        lash=0.023, lid_w=0.012, crease_w=0.012, lid_shadow=0.50,
        brow_gap=0.040, brow_dx=0.195, brow_w=0.205, brow_thick=0.034,
        brow_arch=0.020, brow_tilt=2.0,
        nose_gap=0.050, nose_w=0.074, nose_h=0.044, nose_style="dot",
        mouth_gap=0.138, mouth_w=0.165, mouth_curve=0.014,
        blush_gap=0.065, lip_full=0.12,
        line=1.16, ink_w=0.022, blush=0.48, lid_shadow_alpha=0.34, chin_shadow=0.045,
        neck_w=0.180, neck_y=-0.710, shoulder_w=1.280, shoulder_y=-1.790,
    ),
    "shoujo": dict(
        head_w=0.368, top_w=0.294, top_y=0.452, wide_y=0.100,
        cheek_w=0.344, cheek_y=-0.192, jaw_w=0.268, jaw_y=-0.305, chin_w=0.098,
        head_d=1.04,
        eye_dx=0.182, eye_y=-0.070, eye_w=0.164, eye_up=0.067, eye_down=0.050,
        eye_tilt=7.0, iris_r=0.054, iris_squash=1.30, pupil_r=0.021, hl_r=0.014,
        lash=0.022, lid_w=0.009, crease_w=0.012, lid_shadow=0.36,
        brow_gap=0.055, brow_dx=0.190, brow_w=0.176, brow_thick=0.024,
        brow_arch=0.028, brow_tilt=6.0,
        nose_gap=0.046, nose_w=0.078, nose_h=0.050, nose_style="wedge",
        mouth_gap=0.134, mouth_w=0.178, mouth_curve=0.018,
        blush_gap=0.064, lip_full=0.10,
        line=0.92, ink_w=0.016, blush=0.42, lid_shadow_alpha=0.34, chin_shadow=0.050,
        neck_w=0.172, neck_y=-0.722, shoulder_w=1.200, shoulder_y=-1.800,
    ),
    "shounen": dict(),  # the defaults, stated for symmetry
    "seinen": dict(
        head_w=0.388, top_w=0.300, top_y=0.452, wide_y=0.100,
        cheek_w=0.372, cheek_y=-0.190, jaw_w=0.318, jaw_y=-0.305, chin_w=0.142,
        head_d=1.18,
        eye_dx=0.190, eye_y=-0.040, eye_w=0.156, eye_up=0.055, eye_down=0.039,
        eye_tilt=5.0, iris_r=0.054, iris_squash=1.18, pupil_r=0.020, hl_r=0.013,
        lash=0.020, lid_w=0.010, crease_w=0.013, lid_shadow=0.46,
        brow_gap=0.052, brow_dx=0.194, brow_w=0.178, brow_thick=0.032,
        brow_arch=0.020, brow_tilt=3.0,
        nose_gap=0.042, nose_w=0.090, nose_h=0.076, nose_style="wedge",
        mouth_gap=0.132, mouth_w=0.202, mouth_curve=0.012,
        blush_gap=0.058, lip_full=0.08,
        line=0.95, ink_w=0.019, blush=0.14, lid_shadow_alpha=0.48, chin_shadow=0.070,
        neck_w=0.200, neck_y=-0.718, shoulder_w=1.300, shoulder_y=-1.820,
    ),
}


class Style:
    """A named parameter set with attribute access, and the face's vertical layout.

    Constructing a style resolves the *dependent* parameters -- the brow, nose,
    mouth and blush heights are derived from the eyes -- so a preset only states
    the relationships and can never place a mouth above a nose.
    """

    __slots__ = ("name", "p")

    def __init__(self, name: str, params: dict | None = None):
        p = dict(DEFAULTS)
        if params is None:
            p.update(STYLES[name])
        else:
            p.update(params)
        self.name = name
        self.p = p
        self._resolve()

    def _resolve(self):
        p = self.p
        p["brow_y"] = float(p["eye_y"]) + float(p["eye_up"]) + float(p["brow_gap"])
        p["nose_y"] = float(p["eye_y"]) - float(p["eye_down"]) - float(p["nose_gap"])
        p["mouth_y"] = float(p["nose_y"]) - float(p["mouth_gap"])
        p["blush_y"] = float(p["eye_y"]) - float(p["eye_down"]) - float(p["blush_gap"])
        p["hairline_y"] = float(p["eye_y"]) + float(p["hairline_gap"])

    def __getattr__(self, key):
        try:
            return self.p[key]
        except KeyError as exc:  # pragma: no cover - typo guard
            raise AttributeError(f"style {self.name!r} has no parameter {key!r}") from exc

    def with_(self, **overrides) -> "Style":
        """A copy with parameters replaced -- how a DNA nudges a preset."""
        p = dict(self.p)
        p.update(overrides)
        return Style(self.name, p)

    def pinned(self, **values) -> "Style":
        """A copy with ``values`` written *after* the derived layout is resolved.

        The layout is derived from the eyes, and that is the whole reason a warp of
        the eyes is a warp of the face: moving ``eye_y`` and the gaps moves the
        brow, the nose, the mouth and the blush with them, because those heights are
        *conclusions* about the eyes rather than inputs.

        A few numbers are not conclusions, they are anchors on the **skull** -- the
        hairline is the one there is, and a fringe roots on a skull that does not
        change with a mood -- so they have to be written after the derivation
        instead of through it.  That is what this is for, and it is deliberately
        separate from :meth:`with_`: passing them there would be silently undone.
        """
        out = self.with_()
        out.p.update({key: float(value) for key, value in values.items()})
        return out

    def as_dict(self) -> dict:
        return dict(self.p)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Style({self.name!r})"


def get(name: str) -> Style:
    """Style by name; unknown names raise with the full list."""
    if name not in STYLES:
        raise KeyError(f"unknown style {name!r}; try one of {', '.join(names())}")
    return Style(name)


def names() -> list:
    return list(STYLES)
