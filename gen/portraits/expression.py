"""Expressions: the same face, a warp of its layout and a few feature deltas.

An expression is not a new drawing.  It is two things, and the first is the one
that matters:

* a **warp** of the face's *layout*.  The whole face rises or drops, and it opens or
  closes.  The layout is not a list of positions to move one at a time: it is
  *derived* (in :class:`gen.portraits.style.Style`) from ``eye_y`` and four gaps, so
  warping those moves the brows, the eyes, the nose, the mouth and the blush
  together, and by different amounts.  A smile is not a curved mouth; it is a face
  that rises;
* small **feature deltas** on top: the brows move as whole units and at their inner
  ends, the lid apexes open or narrow, the lower lid lifts, the mouth curves, opens
  and widens.

Splitting it that way is what fixes the old failure.  Eight per-feature offsets move
three features and leave the eyes and the nose exactly where they were, which reads
as a mask worn over a face; ``tests/test_portraits.py`` now measures the other thing,
that every expression moves every landmark.  The warp is the *face*; the deltas are
the features *on* it.

Horizontal distances are **not** warped, because they are not expression: the
distance between the eyes, the width of the skull, the height of the chin are
anatomy, and a mood does not change a skull.  A wide grin is the *mouth* getting
wider (a delta), not the face spreading out.

The warp is bounded (:data:`LIMITS`) because a face is a small thing: past those an
expression slides a face out of its own head, so the numbers are clipped rather than
trusted, the same way every other generator here clips its inputs.  And it is written
through ``Style.pinned`` so the hairline does not follow the eyes -- a mood moves a
face, not the hair rooted on the skull.

The signs are the convention worth stating once: ``brow_inner_dy`` positive is a
*raised* inner end, which is the brow of grief; negative is the brow of anger.
That single number is most of what separates the two faces.
"""
from __future__ import annotations

import numpy as np

__all__ = ["EXPRESSIONS", "LIMITS", "get", "neutral", "warp"]

#: How far a warp may move the layout.  ``face_dy`` is in head units and
#: ``face_spread`` is a proportion, so the bound means the same thing on a chibi head
#: and on a seinen one.
LIMITS = {"face_dy": 0.030, "face_spread": 0.22}

#: The neutral row every other expression is a delta on.
_NEUTRAL = dict(
    # -- the whole face: the layout, not a feature ------------------------
    face_dy=0.0,        # the whole face lifts (+) or drops (-), head units
    face_spread=0.0,    # the features spread apart (+) or close up (-), proportion
    # -- the features on it ------------------------------------------------
    brow_dy=0.0,        # whole brow, world units
    brow_inner_dy=0.0,  # inner end only: + grief, - anger
    brow_arch=0.0,      # extra arch height
    eye_up_s=1.0,       # upper lid apex scale
    eye_down_s=1.0,     # lower lid apex scale
    lower_dy=0.0,       # the lower lid lifts by this much (a smile squashes the eye)
    squint=0.0,         # 0..1, closes the opening from both sides
    mouth_curve=0.0,    # extra smile curvature, world units
    mouth_open=0.0,     # 0 is a line; > 0 is a mouth with a dark interior
    mouth_dy=0.0,
    mouth_asym=0.0,     # one-sided lift: the smirk
    mouth_w_s=1.0,      # the mouth's own width -- a grin is wider, a set jaw narrower
    blush_mul=1.0,
    lash_mul=1.0,
)

EXPRESSIONS = {
    "neutral": dict(_NEUTRAL),
    # A smile rises and opens a little, and the deltas curve the mouth, widen it and
    # lift the lower lid -- the parts a warp cannot do.
    "smile": dict(_NEUTRAL, face_dy=0.005, face_spread=0.07,
                  brow_dy=0.002, brow_inner_dy=0.003, eye_up_s=0.90, eye_down_s=0.94,
                  lower_dy=0.010, mouth_curve=0.022, mouth_w_s=1.06, blush_mul=1.05),
    "grin": dict(_NEUTRAL, face_dy=0.008, face_spread=0.12,
                 brow_dy=0.004, brow_inner_dy=0.002, eye_up_s=0.78, eye_down_s=0.88,
                 lower_dy=0.018, mouth_curve=0.020, mouth_open=0.018, mouth_w_s=1.14,
                 blush_mul=1.10),
    # Surprise is the largest warp there is: the whole face opens -- the brows rise,
    # the eyes widen, the mouth drops away from the nose -- and the feature deltas
    # only say how each one does its part of that.  The brow's own lift is small on
    # purpose: the warp is already carrying it, and spending it twice would push the
    # peak of a chibi's brow up to its hairline.
    "surprised": dict(_NEUTRAL, face_dy=0.012, face_spread=0.16,
                      brow_dy=0.004, brow_arch=0.014, eye_up_s=1.20, eye_down_s=1.06,
                      mouth_curve=0.004, mouth_open=0.026, mouth_w_s=0.94),
    "sad": dict(_NEUTRAL, face_dy=-0.010, face_spread=-0.06,
                brow_dy=0.002, brow_inner_dy=0.018, eye_up_s=1.02, eye_down_s=0.92,
                mouth_curve=-0.018, mouth_dy=-0.004, mouth_w_s=0.96),
    "angry": dict(_NEUTRAL, face_dy=-0.005, face_spread=-0.10,
                  brow_dy=-0.004, brow_inner_dy=-0.026, brow_arch=-0.010,
                  eye_up_s=0.92, eye_down_s=1.02, mouth_curve=-0.012, mouth_dy=-0.002,
                  mouth_w_s=0.92),
    "smug": dict(_NEUTRAL, face_dy=0.002, face_spread=0.02,
                 brow_dy=-0.004, brow_inner_dy=-0.008, eye_up_s=0.86, eye_down_s=0.98,
                 mouth_curve=0.012, mouth_asym=0.014, mouth_w_s=1.02, blush_mul=0.85),
    "wink": dict(_NEUTRAL, face_dy=0.005, face_spread=0.06,
                 brow_dy=0.004, brow_inner_dy=0.004, mouth_curve=0.020, mouth_w_s=1.05,
                 closed=(1,)),
}

#: Sides whose eye is closed, per expression.
_CLOSED = {"wink": (+1,)}

def get(name: str) -> dict:
    """Expression row by name, with the closed-eye sides resolved."""
    if name not in EXPRESSIONS:
        raise KeyError(f"unknown expression {name!r}; try {', '.join(EXPRESSIONS)}")
    row = dict(EXPRESSIONS[name])
    row["closed"] = tuple(_CLOSED.get(name, ()))
    return row


def neutral() -> dict:
    return get("neutral")


def warp(st, expr: dict):
    """The style ``st`` with its face's layout moved by the expression ``expr``.

    The returned style is a complete one -- :meth:`gen.portraits.style.Style.pinned`
    leaves the hairline where an anchor belongs -- so it can be handed to any pass
    that draws *the face*, while the head, the hair and the skull keep the style they
    were built from.  That split is deliberate and is the second half of the claim:
    a mood moves a face, not a person.

    A neutral row has nothing to do and returns ``st`` **itself**, so a portrait with
    a neutral expression is the portrait that existed before there was a warp, byte
    for byte.

    Six parameters move, and the list being short is the point: ``eye_y`` carries the
    whole layout and the four gaps are the face's vertical articulation -- the brow,
    nose, mouth and blush heights are *measured down from the eyes through them*, so
    a warp of the gaps is a warp of every landmark with one number.  Each is read by
    more than one feature, which is the test of whether it belongs here: anything a
    single feature owns is a delta, not a warp.
    """
    dy = float(np.clip(expr.get("face_dy", 0.0), -LIMITS["face_dy"], LIMITS["face_dy"]))
    spread = float(np.clip(expr.get("face_spread", 0.0),
                           -LIMITS["face_spread"], LIMITS["face_spread"]))
    if dy == 0.0 and spread == 0.0:
        return st
    # The lower face articulates a little more than the midface: a expression that
    # closes the eyes and the nose also shortens the chin-to-mouth distance.  The
    # extra 15 % is small, but it also keeps the mouth from cancelling exactly when
    # ``face_dy`` and ``face_spread`` pull it in opposite directions -- a mouth that
    # does not move at all is a mask, whichever way the arithmetic got there.
    return st.with_(
        eye_y=st.eye_y + dy,
        brow_gap=st.brow_gap * (1.0 + spread),
        nose_gap=st.nose_gap * (1.0 + spread),
        mouth_gap=st.mouth_gap * (1.0 + 1.15 * spread),
        blush_gap=st.blush_gap * (1.0 + spread),
    ).pinned(hairline_y=st.hairline_y)
