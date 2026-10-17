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
"""
from . import blazon, charges, field, flags, ordinaries, paint, shields, tinctures

__all__ = [
    "blazon",
    "charges",
    "field",
    "flags",
    "ordinaries",
    "paint",
    "shields",
    "tinctures",
]
