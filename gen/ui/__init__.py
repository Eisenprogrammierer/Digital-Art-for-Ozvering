"""UI chrome: panels, frames and buttons, generated as nine-slice assets.

The package exists because a UI asset is the first asset in this project that has
to survive being resized to a size nobody chose.  An icon is rendered at the size
it is asked for; a panel is drawn once and stretched by the layout.  So the unit
of output here is not a PNG but ``(PNG, corner insets, minimum size)`` -- the
nine-slice envelope -- and every module below is arranged around that fact:

``style``
    The numbers.  A style is a palette plus proportions in *design pixels*, which
    makes it resolution-independent: ``scale=2`` doubles the pixels and changes no
    number.  It also states the geometric invariant the whole package leans on --
    every bit of curvature, rule included, must fit inside the corner box -- and
    clamps a spec onto that legal set.  The insets and the band widths take one
    number per side, so a title bar is a wider border (``style.title_bar``) rather
    than a second kind of window.
``frame``
    The geometry: five concentric rounded rectangles and a corner ornament, all
    as distance fields.  The ornament's radius is a Fourier series, which is the
    one place a series like that is safe: inside a corner box, where nothing is
    ever resampled.
``slice``
    The contract itself: the insets, the nine rectangles, and the resample that a
    game engine will perform.  Implemented here so it can be tested here.
``paint``
    Five passes -- shadow, body, frame, rule, ornament -- over one fixed light.
    A button state is a colour change and, when pressed, the same bevel with its
    ramp inverted; the geometry never moves.

``material``
    Wood and stone: the body's surface, out of fBm and Worley noise evaluated in
    design pixels.  A material is the one look in the package that gives up the
    stretchable middle on purpose, which is why the body it fills is measured
    rather than trusted.

Typical use::

    from gen.ui import style, paint, slice
    spec = style.random_style(7)
    box  = paint.panel(spec, (220, 132), scale=2)
    ins  = slice.insets(...)          # see frame.Panel
"""
from . import frame, material, paint, slice, style

__all__ = [
    "frame", "material", "paint", "slice", "style",
    "Panel", "draw", "panel", "frame_image", "button", "title_bar",
]

Panel = frame.Panel
draw = paint.draw
panel = paint.panel
frame_image = paint.frame
button = paint.button
title_bar = style.title_bar
