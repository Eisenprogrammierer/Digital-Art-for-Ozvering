"""Icons: a plate, a charge, one light, and a set that hangs together.

A game's icon set is the asset family with the least room to be clever in.  An
icon is 32 px of meaning that a player reads in a glance, sometimes while the rest
of the screen is moving, so the rules here are the opposite of the heraldry
package's: no detail that only survives at 512 px, no charge that needs a caption,
and one construction shared by every kind so that the *set* reads as a set.

The construction is a **plate** with a **charge** on it:

* the plate is the ground -- a rounded square, a hexagon, a circle or a shield --
  shaded as a solid and given a bright rim and a knocked-in recess around the
  charge;
* the charge is the meaning: a coin is a ring, water is a drop, a blade is a
  sword seen flat.  ``motifs`` owns that vocabulary and it is deliberately small.

The colour script is one hue and four lightnesses (plate, rim, recess, charge),
so a resource icon is read by colour before it is read by shape, and two icons of
different kinds still look like they came from the same game.

Everything is in **world units** on the unit square and rendered at whatever size
is asked for, so ``--sizes 32,64,128`` generates three drawings rather than one
drawing and two resizes -- and the small ones are the ones that matter.

Typical use::

    from gen.icons import paint, motifs
    tile = paint.draw(motifs.canonical("grain"), 128)
    any_icon = paint.icon(seed=7, size=64)

``export.atlas`` is the companion: it trims a set of these, packs them into one
image and writes the table of rectangles an engine needs.
"""
from __future__ import annotations

from . import motifs, paint

__all__ = ["motifs", "paint", "draw", "icon", "canonical"]

draw = paint.draw
icon = paint.icon
canonical = motifs.canonical
