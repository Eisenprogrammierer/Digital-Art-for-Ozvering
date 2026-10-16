"""Core building blocks for procedural 2-D asset generation.

Modules
-------
mat
    Affine transforms in homogeneous coordinates.
sdf
    Signed distance fields: primitives, boolean operators, transforms.
noise
    Deterministic lattice noise, fBm, domain warping, Worley cells.
palette
    OKLab/OKLCH colour science, mixing, harmony, gamut fitting.
canvas
    Linear-light RGBA buffer with a world-space camera.
render
    Distance field -> pixels: coverage, bevel lighting, shadows, sheets.
"""
from . import canvas, mat, noise, palette, render, sdf

__all__ = ["canvas", "mat", "noise", "palette", "render", "sdf"]
