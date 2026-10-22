"""Wardrobe and decorations: clothing and awards as standalone assets.

The portrait package dresses a *bust*: a coat exists only where it crosses the
shoulders.  A game also needs the coat as an object -- an inventory thumbnail, a
uniform in a codex, the decoration beside a name -- and that asset has no sitter
inside it.  This package is the separate generator for those objects.

Two families, two modules:

* :mod:`gen.wardrobe.garment` draws a flat-lay garment: a body, sleeves, a
  collar, closures, seams and folds, chosen by cut and cloth.  It shares the
  drawing language of the portrait (world-unit geometry, one light, a small set
  of cloth values) but not its clipping, because there is no bust to clip to.
* :mod:`gen.wardrobe.awards` draws a decoration: a ribbon, a medal, a cross, an
  order star or an armband.  These are deliberately *not* the portrait's medals
  scaled up: a standalone award is read as a complete object, so it gets a pin,
  a suspension ring, a full ribbon and a back.

Every spec is a plain dictionary of names, colours and integers.  A seed produces
a legal spec, a spec renders at any size, and the sheet is a row of the canonical
cuts or kinds followed by seeded variants.  Nothing is sampled, traced or
downloaded.
"""
from __future__ import annotations

from . import awards, garment

__all__ = ["awards", "garment", "draw_garment", "draw_award"]

draw_garment = garment.render
draw_award = awards.render
