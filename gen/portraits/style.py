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

EYE_LIFT = 0.0045

DEFAULTS = dict(

    head_w=0.390,
    top_w=0.305,
    top_y=0.455,
    wide_y=0.095,
    cheek_w=0.360,
    cheek_y=-0.190,
    jaw_w=0.286,
    jaw_y=-0.305,
    chin_w=0.118,
    chin_y=-0.500,

    head_d=1.07,

    eye_dx=0.190,
    eye_y=-0.060,
    eye_w=0.172,
    eye_up=0.060,
    eye_down=0.041,
    eye_tilt=5.0,
    iris_r=0.052,
    iris_squash=1.14,
    pupil_r=0.021,
    hl_r=0.014,
    lash=0.018,
    lid_w=0.008,
    crease_w=0.009,
    lid_shadow=0.40,

    brow_gap=0.055,
    brow_dx=0.194,
    brow_w=0.184,
    brow_thick=0.033,
    brow_arch=0.025,
    brow_tilt=4.0,
    nose_gap=0.042,
    nose_w=0.085,
    nose_h=0.064,
    nose_style="wedge",
    mouth_gap=0.132,
    mouth_w=0.212,
    mouth_curve=0.016,
    mouth_style="line",
    lip_full=0.10,
    ear_size=1.0,
    blush_gap=0.062,
    hairline_gap=0.300,

    line=1.0,
    ink_w=0.020,
    blush=0.28,
    lid_shadow_alpha=0.42,
    chin_shadow=0.055,

    neck_w=0.190,
    neck_y=-0.720,
    shoulder_w=0.540,
    shoulder_y=-1.760,
    shoulder_slope=0.0,
    under_eye=0.26,
)

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
        neck_w=0.180, neck_y=-0.710, shoulder_w=0.580, shoulder_y=-1.790,
    ),
    "shoujo": dict(
        head_w=0.368, top_w=0.294, top_y=0.452, wide_y=0.100,

        cheek_w=0.348, cheek_y=-0.192, jaw_w=0.282, jaw_y=-0.305, chin_w=0.118,
        head_d=1.04,
        eye_dx=0.182, eye_y=-0.070, eye_w=0.164, eye_up=0.062, eye_down=0.044,
        eye_tilt=7.0, iris_r=0.048, iris_squash=1.16, pupil_r=0.019, hl_r=0.012,
        lash=0.016, lid_w=0.007, crease_w=0.009, lid_shadow=0.34,
        brow_gap=0.055, brow_dx=0.190, brow_w=0.176, brow_thick=0.024,
        brow_arch=0.028, brow_tilt=6.0,
        nose_gap=0.046, nose_w=0.078, nose_h=0.050, nose_style="wedge",
        mouth_gap=0.134, mouth_w=0.178, mouth_curve=0.018,
        blush_gap=0.064, lip_full=0.10,
        line=0.92, ink_w=0.016, blush=0.42, lid_shadow_alpha=0.34, chin_shadow=0.050,
        neck_w=0.172, neck_y=-0.722, shoulder_w=0.515, shoulder_y=-1.800,
    ),
    "shounen": dict(),
    "seinen": dict(
        head_w=0.388, top_w=0.300, top_y=0.452, wide_y=0.100,
        cheek_w=0.372, cheek_y=-0.190, jaw_w=0.318, jaw_y=-0.305, chin_w=0.142,
        head_d=1.18,
        eye_dx=0.190, eye_y=-0.040, eye_w=0.156, eye_up=0.052, eye_down=0.037,
        eye_tilt=5.0, iris_r=0.050, iris_squash=1.12, pupil_r=0.019, hl_r=0.012,
        lash=0.016, lid_w=0.008, crease_w=0.010, lid_shadow=0.44,
        brow_gap=0.052, brow_dx=0.194, brow_w=0.178, brow_thick=0.032,
        brow_arch=0.020, brow_tilt=3.0,
        nose_gap=0.042, nose_w=0.090, nose_h=0.076, nose_style="wedge",
        mouth_gap=0.132, mouth_w=0.202, mouth_curve=0.012,
        blush_gap=0.058, lip_full=0.08,
        line=0.95, ink_w=0.019, blush=0.14, lid_shadow_alpha=0.48, chin_shadow=0.070,
        neck_w=0.200, neck_y=-0.718, shoulder_w=0.560, shoulder_y=-1.820,
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
        if "_eye_lift" not in p:
            p["eye_y"] = float(p["eye_y"]) + EYE_LIFT
            p["brow_gap"] = float(p["brow_gap"]) - EYE_LIFT
            p["nose_gap"] = float(p["nose_gap"]) + EYE_LIFT
            p["blush_gap"] = float(p["blush_gap"]) + EYE_LIFT
            p["hairline_gap"] = float(p["hairline_gap"]) - EYE_LIFT
            p["_eye_lift"] = EYE_LIFT
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
        except KeyError as exc:
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

    def __repr__(self) -> str:
        return f"Style({self.name!r})"


def get(name: str) -> Style:
    """Style by name; unknown names raise with the full list."""
    if name not in STYLES:
        raise KeyError(f"unknown style {name!r}; try one of {', '.join(names())}")
    return Style(name)


def names() -> list:
    return list(STYLES)
