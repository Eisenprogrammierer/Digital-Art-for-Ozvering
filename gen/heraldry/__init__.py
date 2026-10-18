"""Coats of arms and flags, generated from heraldic primitives.

The package mirrors the structure of the tradition itself:

``tinctures``
    The heraldic colour system (metals, colours, furs, stains) and the rule of
    tincture.
``field``
    Divisions and variations of the field, expressed as partitions of the plane
    into distance-field regions.
``ordinaries``
    The geometric bands (fess, pale, bend, chevron, cross, saltire, chief,
    bordure, ...) in their conventional proportions.
``charges``
    The mobile figures (mullets, roundels, lozenges, fleurs-de-lis, beasts, ...).
``shields``
    Silhouettes: escutcheons and flag rectangles.
``paint``
    Turning a paint list into pixels, and finishing the shield.
``blazon``
    Composition, validation and blazon text; the random arms generator.
``flags``
    Flag layouts, built from the very same primitives.
``crest``
    A faction's whole identity from its name: the name is the seed, the banner
    flies what the shield bears, and the blazon comes with it in both languages.
"""
from . import blazon, charges, crest, field, flags, ordinaries, paint, shields, tinctures

__all__ = [
    "blazon",
    "charges",
    "crest",
    "field",
    "flags",
    "ordinaries",
    "paint",
    "shields",
    "tinctures",
]
