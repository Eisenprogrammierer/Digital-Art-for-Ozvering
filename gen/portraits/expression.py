"""Expressions: the same face, a different set of deltas.

An expression is not a new drawing.  It is eight small offsets -- the brows
move as whole units and at their inner ends, the lid apexes open or narrow, the
lower lid lifts, the mouth curves and opens -- and everything else stays exactly
where it was.  Keeping the table declarative means a new expression is a row, and
it means the *same* row drives the brows, the eyes and the mouth, so a surprised
face cannot end up with a smiling mouth unless someone writes that row on purpose.

The signs are the convention worth stating once: ``brow_inner_dy`` positive is a
*raised* inner end, which is the brow of grief; negative is the brow of anger.
That single number is most of what separates the two faces.
"""
from __future__ import annotations

__all__ = ["EXPRESSIONS", "get", "neutral"]

#: The neutral row every other expression is a delta on.
_NEUTRAL = dict(
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
    blush_mul=1.0,
    lash_mul=1.0,
)

EXPRESSIONS = {
    "neutral": dict(_NEUTRAL),
    "smile": dict(_NEUTRAL, brow_dy=0.004, brow_inner_dy=0.003, eye_up_s=0.90,
                  eye_down_s=0.94, lower_dy=0.010, mouth_curve=0.022, blush_mul=1.05),
    "grin": dict(_NEUTRAL, brow_dy=0.007, brow_inner_dy=0.002, eye_up_s=0.78,
                 eye_down_s=0.88, lower_dy=0.018, mouth_curve=0.020, mouth_open=0.018,
                 blush_mul=1.10),
    "surprised": dict(_NEUTRAL, brow_dy=0.024, brow_arch=0.014, eye_up_s=1.20,
                      eye_down_s=1.06, mouth_curve=0.004, mouth_open=0.026),
    "sad": dict(_NEUTRAL, brow_dy=0.006, brow_inner_dy=0.018, eye_up_s=1.02,
                eye_down_s=0.92, mouth_curve=-0.018, mouth_dy=-0.004),
    "angry": dict(_NEUTRAL, brow_dy=-0.012, brow_inner_dy=-0.026, brow_arch=-0.010,
                  eye_up_s=0.92, eye_down_s=1.02, mouth_curve=-0.012, mouth_dy=-0.002),
    "smug": dict(_NEUTRAL, brow_dy=-0.006, brow_inner_dy=-0.008, eye_up_s=0.86,
                 eye_down_s=0.98, mouth_curve=0.012, mouth_asym=0.014, blush_mul=0.85),
    "wink": dict(_NEUTRAL, brow_dy=0.008, brow_inner_dy=0.004, mouth_curve=0.020,
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
