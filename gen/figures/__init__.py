"""Figures: a body, a gait, and a sprite out of the two.

``rig``
    The skeleton: seventeen joints, one bone per joint, the rest pose where the
    proportions live, and forward kinematics as one pass down a tuple.  Angles are
    absolute -- measured from straight down -- so a pose is the angle of each bone
    and a cycle can speak in sentences about walking.
``cycle``
    The gaits: ``idle``, ``walk``, ``run`` and ``attack`` as periodic functions of
    phase.  Every term is a sine, a cosine or a window of one, so every cycle closes
    exactly at ``2*pi`` and a loop has no seam to hide.  A cycle returns *deltas*
    from the rest pose, never a pose.
``body``
    The drawing: limbs as tapered locks, a tunic, a belt, hose, boots and hair, all
    from the portrait package's own brush, ink and cel pass -- one light, one
    outline, one art set.  The far side is offset and wears a shadow of its own
    material, which is the whole of depth in a flat drawing.
``figure``
    The seed and the canvas: ``render(seed, cycle, phase, size, facing)``,
    ``frames(...)`` for a whole cycle, and ``describe(seed)`` for the table that
    says what the picture is made of.  ``FACINGS`` is ``right``, ``left``, ``up``
    and ``down``.

The rig is drawn in **side view**, so its two limbs are *near* and *far* rather
than left and right.  The front and back facings keep the same rig and ask
``body`` to put the joints on two sides instead of two depths; ``left`` is the
finished right sprite mirrored.  A figure is drawn in **world units** -- one unit
across the sprite box -- so the same figure is 32 px or 256 px without being
resampled.

Typical use::

    from gen.figures import figure
    frames, phases = figure.frames(seed=7, cycle="walk", count=8, size=96)
    sheet, info = export.sheet.build(frames, fps=12.0)
"""
from . import body, cycle, figure, rig

__all__ = ["body", "cycle", "figure", "rig"]
