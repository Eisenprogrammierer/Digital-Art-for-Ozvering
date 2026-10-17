"""Core building blocks for procedural 2-D asset generation.

Modules
-------
mat
    Affine transforms in homogeneous coordinates.
sdf
    Signed distance fields: primitives, boolean operators, transforms.
noise
    Deterministic lattice noise, fBm, domain warping, Worley cells.
curve
    Bezier, Catmull-Rom and Chaikin: control points -> sampled outlines.
palette
    OKLab/OKLCH colour science, mixing, harmony, gamut fitting.
canvas
    Linear-light RGBA buffer with a world-space camera.
render
    Distance field -> pixels: coverage, bevel lighting, shadows, sheets.
"""
from . import canvas, curve, mat, noise, palette, render, sdf

__all__ = ["canvas", "curve", "mat", "noise", "palette", "render", "sdf"]
