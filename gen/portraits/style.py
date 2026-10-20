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
    head_w=0.415,
    top_w=0.265,
    top_y=0.445,
    wide_y=0.100,
    cheek_w=0.402,
    cheek_y=-0.195,
    jaw_w=0.338,       # half-width at the jaw angle
    jaw_y=-0.305,
    chin_w=0.140,
    chin_y=-0.500,
    # The head as a *solid*: ``head_d`` is its half-depth at the deepest, as a
    # multiple of its own half-width, and that one number is the whole of what a
    # style adds to the turn model.  One is a head as deep as it is wide -- round,
    # and their turned outlines do not change width; more is a deeper skull, which
    # widens its outline towards profile and slides its features further.
    head_d=1.05,
    # -- eyes (both, in head units) --------------------------------------
    eye_dx=0.205,      # half the centre-to-centre distance
    eye_y=-0.065,
    eye_w=0.230,       # width of the eye opening
    eye_up=0.098,      # upper lid apex above the corner chord
    eye_down=0.072,
    eye_tilt=6.0,      # degrees, outer corner raised
    iris_r=0.082,      # iris half-width
    iris_squash=1.22,  # iris height / width
    pupil_r=0.031,
    hl_r=0.028,        # big highlight radius
    lash=0.030,        # upper lash thickness at its heaviest
    lid_w=0.011,       # lower lid line
    crease_w=0.011,
    lid_shadow=0.42,   # how far the lid shadow reaches down the eye
    # -- the rest of the face, measured down from the eyes ---------------
    brow_gap=0.048,    # upper lid -> brow
    brow_dx=0.212,
    brow_w=0.195,
    brow_thick=0.036,
    brow_arch=0.030,
    brow_tilt=4.0,     # degrees, outer end raised
    nose_gap=0.030,    # lower lid -> nose
    nose_w=0.068,
    nose_h=0.060,
    nose_style="wedge",   # "dot" | "wedge" | "line"
    mouth_gap=0.125,   # nose -> mouth
    mouth_w=0.155,
    mouth_curve=0.016,
    mouth_style="line",   # "line" | "full"
    blush_gap=0.060,   # lower lid -> blush
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
    neck_w=0.155,
    neck_y=-0.790,     # where the neck meets the shoulders
    shoulder_w=1.240,
    shoulder_y=-1.760,  # the shoulder polygon's lower edge, well below the frame
)

#: Presets.  Read them as a table of proportions, not as arbitrary numbers:
#: ``eye_w / (2 * head_w)`` is the classic "eyes across the face" count --
#: chibi ~2, shoujo ~3, shounen ~3.5, seinen ~4.5 (a real face is 5).
STYLES = {
    "chibi": dict(
        head_w=0.480, top_w=0.370, top_y=0.450, wide_y=0.095,
        cheek_w=0.455, cheek_y=-0.170, jaw_w=0.402, jaw_y=-0.262, chin_w=0.238,
        head_d=0.98,
        eye_dx=0.205, eye_y=-0.105, eye_w=0.335, eye_up=0.168, eye_down=0.128,
        eye_tilt=4.0, iris_r=0.136, iris_squash=1.06, pupil_r=0.056, hl_r=0.046,
        lash=0.036, lid_w=0.014, crease_w=0.014, lid_shadow=0.50,
        brow_gap=0.035, brow_dx=0.208, brow_w=0.200, brow_thick=0.034,
        brow_arch=0.020, brow_tilt=2.0,
        nose_gap=0.030, nose_w=0.050, nose_h=0.034, nose_style="dot",
        mouth_gap=0.125, mouth_w=0.138, mouth_curve=0.014,
        blush_gap=0.068,
        line=1.22, ink_w=0.025, blush=0.48, lid_shadow_alpha=0.34, chin_shadow=0.045,
        neck_w=0.150, neck_y=-0.760, shoulder_w=1.280, shoulder_y=-1.820,
    ),
    "shoujo": dict(
        head_w=0.393, top_w=0.255, top_y=0.448, wide_y=0.105,
        cheek_w=0.388, cheek_y=-0.200, jaw_w=0.312, jaw_y=-0.310, chin_w=0.118,
        head_d=1.02,
        eye_dx=0.192, eye_y=-0.075, eye_w=0.224, eye_up=0.104, eye_down=0.086,
        eye_tilt=8.0, iris_r=0.085, iris_squash=1.30, pupil_r=0.033, hl_r=0.030,
        lash=0.029, lid_w=0.009, crease_w=0.012, lid_shadow=0.36,
        brow_gap=0.050, brow_dx=0.198, brow_w=0.190, brow_thick=0.024,
        brow_arch=0.034, brow_tilt=6.0,
        nose_gap=0.030, nose_w=0.052, nose_h=0.040, nose_style="wedge",
        mouth_gap=0.120, mouth_w=0.130, mouth_curve=0.018,
        blush_gap=0.062,
        line=0.92, ink_w=0.016, blush=0.42, lid_shadow_alpha=0.34, chin_shadow=0.050,
        neck_w=0.140, neck_y=-0.795, shoulder_w=1.200, shoulder_y=-1.840,
    ),
    "shounen": dict(),  # the defaults, stated for symmetry
    "seinen": dict(
        head_w=0.402, top_w=0.250, top_y=0.448, wide_y=0.110,
        cheek_w=0.412, cheek_y=-0.190, jaw_w=0.372, jaw_y=-0.300, chin_w=0.176,
        head_d=1.18,
        eye_dx=0.198, eye_y=-0.040, eye_w=0.196, eye_up=0.078, eye_down=0.058,
        eye_tilt=5.0, iris_r=0.070, iris_squash=1.18, pupil_r=0.028, hl_r=0.019,
        lash=0.024, lid_w=0.010, crease_w=0.013, lid_shadow=0.46,
        brow_gap=0.045, brow_dx=0.204, brow_w=0.188, brow_thick=0.032,
        brow_arch=0.020, brow_tilt=3.0,
        nose_gap=0.030, nose_w=0.072, nose_h=0.068, nose_style="wedge",
        mouth_gap=0.128, mouth_w=0.158, mouth_curve=0.012,
        blush_gap=0.055,
        line=0.95, ink_w=0.019, blush=0.14, lid_shadow_alpha=0.48, chin_shadow=0.070,
        neck_w=0.165, neck_y=-0.785, shoulder_w=1.300, shoulder_y=-1.860,
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
        p["hairline_y"] = float(p["eye_y"]) + 0.30

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
