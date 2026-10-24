"""Procedural anime-style character portraits, as distance fields.

The pipeline is the same one the heraldry package uses, with the two additions a
drawn face needs and a flat coat of arms does not:

``ink``
    A variable-width stroke along any shape's contour (the zero set of its
    field), which is what carries the form.  Anime is a line-art medium before it
    is a shading one.
``cel``
    A quantised Lambert term over a cheap ellipsoidal volume standing in for the
    skull, plus the two colour rules that make a shadow read as anime: chroma up
    as lightness down, and the hue rolled through magenta towards violet.

Everything else is composition.  A face is a hundred small distance fields --
hair clumps, eyelids, a nose wedge, a lip -- built from the control points in
``style`` and coloured from the genes in ``dna``.  Because a shape carries its own
bounds, each one is only ever evaluated where it can be seen; that is what makes
a portrait render in a fraction of a second instead of a minute.

``expression``
    A mood is not a feature: it is a warp of the face's *layout*.  The layout is
    derived from ``eye_y`` and four gaps, so warping those moves the brows, the
    eyes, the nose, the mouth and the blush together; the head, the hair and the
    skull are built from the style before the warp, because a mood moves a face
    and not a person.

A portrait is a *genome*: four base colours, a style, a handful of shape
modifiers.  It renders to pixels, round-trips through a one-line string, and can
be crossed with another genome to produce a child -- which, for a strategy game,
is the point.
"""
from . import (brush, cel, character, dna, expression, eyes, face, hair, head,
               hues, ink, portrait, shape, style)

__all__ = [
    "brush", "cel", "character", "dna", "expression", "eyes", "face", "hair",
    "head", "hues", "ink", "portrait", "shape", "style",
    "render", "render_seed", "compose", "canvas", "describe",
]

#: The friendliest entry point: genes in, canvas out.
render = portrait.render
render_seed = portrait.render_seed
compose = portrait.compose
canvas = portrait.canvas
describe = portrait.describe
