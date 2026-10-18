"""A faction wears its livery on its icons.

An icon is read by *colour* before it is read by shape -- that is the whole reason
the set shares one hue per kind -- and it is exactly what stops an icon from being
a faction's own.  So a faction supplies a **second** hue, and the second hue has
somewhere to go that does not fight the first:

* the **charge** keeps the kind's hue, because that is what the icon *means*: water
  is still the colour of water, in every faction;
* the **plate** -- the ground, the rim and the recess -- takes the faction's hue,
  because the ground is the large surface and is where a faction belongs.

Two surfaces, two hues, one of them quiet: the plate wears the faction's hue at a
wash's chroma, so it is a ground for the charge rather than a colour arguing with
it.  A faction whose arms are a metal gets a grey plate and is told apart by
lightness, exactly as Or and Argent are everywhere else in this project -- and by
the same rule the map uses, which is not a coincidence: the hue here *is* the hue a
province of that faction is washed in, because both read it off
:func:`gen.heraldry.crest.field_tincture`.

The faction's hue is its **livery**: its field tincture, moved by the same integer
shade the heraldry draws from the name.  A faction's icons, its shield, its banner
and its provinces are therefore one colour, and none of them can drift from the
others -- the name is the only input any of them takes.
"""
from __future__ import annotations

import numpy as np

from core import palette
from gen.heraldry import crest, tinctures

from . import paint

__all__ = ["PLATE_C", "C_REF", "NEUTRAL", "L_SPAN", "L_GAIN", "field", "worn", "lch",
           "hue", "chroma", "light_step", "tint", "table"]

#: The chroma a faction's hue is worn at on a plate.  Plain icons wear their plate
#: between 0.018 and 0.048; this is a little above that top end, because the plate
#: now has to *say* something -- and still well under the charge's ~0.115, because
#: the ground has to stay the ground.
PLATE_C = 0.075

#: The chroma a tincture needs to count as fully chromatic here.  Gules is the most
#: saturated classical colour at about 0.21 and Azure and Or are near 0.17, so this
#: puts every colour at or near the full plate chroma and leaves the metals --
#: Argent at 0.0025, Sable at 0.008 -- very nearly grey.
C_REF = 0.17

#: Below this a tincture has no hue to lend a plate; it is a metal or the black.
#: The map draws the same line for the same reason: a grey has no hue to confuse
#: with anything, and a metal is told apart by lightness.
NEUTRAL = 0.03

#: How far the livery's lightness step may move the plate, and how hard it is
#: pressed against that bound.  A metal has no hue to wear, so this is the whole of
#: what tells two metals apart -- exactly as it is on the map, which spends the same
#: step the same way -- and it is bounded because a plate that drifted further would
#: stop being a dark ground for a bright charge.  The plate's lightness is the
#: *only* design number a faction moves: hue and chroma are added, lightness is
#: nudged, so an icon redrawn later is still the icon it was.
L_SPAN = 0.055
L_GAIN = 2.2


def field(name: str) -> tinctures.Tincture:
    """The faction's field tincture: the colour its coat is mostly made of."""
    return crest.field_tincture(name)


def worn(name: str) -> tinctures.Tincture:
    """The faction's field *as worn*: its own shade of that tincture."""
    return tinctures.shaded(field(name).key, crest.shade_table(name))


def lch(name: str) -> tuple:
    """The worn tincture as OKLCH ``(L, C, h)`` -- the fact both answers read from."""
    lightness, chroma, hue_ = palette.srgb_to_oklch(worn(name).rgb)
    return float(lightness), float(chroma), float(hue_)


def hue(name: str) -> float:
    """The faction's hue, in OKLCH turns.

    The same number the map washes a province of this faction in, because both take
    it from the faction's worn field tincture rather than re-deriving it.  The claim
    is about a *chromatic* faction: a metal or the black has no chroma, so it has no
    hue to lend and the number here is arbitrary -- its plate is told apart by
    lightness, which is the only thing a grey has.
    """
    return lch(name)[2]


def chroma(name: str) -> float:
    """The chroma the faction's hue is worn at on a plate, in OKLCH.

    Proportional to the tincture's own chroma, capped at :data:`PLATE_C`: a
    chromatic colour gives a coloured plate and a metal gives a grey one, without a
    branch that has to be kept in step with the tincture table.
    """
    return float(PLATE_C * min(lch(name)[1] / C_REF, 1.0))


def light_step(name: str) -> float:
    """How far the faction's livery moves its plate's lightness, in OKLab.

    The map's own arithmetic (``gen.maps.politics.wash``), on the plate's budget
    instead of the wash's: the livery's lightness step, gained and passed through a
    saturating map so it is bounded.  This is what a *metal* faction is told apart
    by, since it has no hue to lend -- and for a colour it is a second, quieter note
    on top of the hue.
    """
    spent = L_GAIN * (lch(name)[0] - float(palette.srgb_to_oklch(field(name).rgb)[0]))
    return float(np.tanh(spent / L_SPAN) * L_SPAN)


def tint(spec: dict, name: str) -> dict:
    """The icon ``spec`` in the colours of the faction called ``name``.

    The charge is left exactly as it was -- its hue, its chroma and its lightness
    are the kind's -- and only the plate is dressed, by the three things a livery
    has to spend: a second hue, the chroma that hue is worn at, and a lightness
    step.  So two factions rendering one kind differ in the ground and agree on the
    figure, which is the one arrangement where the kind and the faction can both be
    read.
    """
    s = paint.clamp(spec)
    s["plate_hue"] = hue(name)
    s["plate_chroma"] = chroma(name)
    s["plate_light"] = float(np.clip(s["plate_light"] + light_step(name), 0.10, 0.56))
    return s


def table(names) -> dict:
    """The faction hue table: every name's hue, ready for a sidecar or a shader."""
    return {str(name): hue(name) for name in names}
