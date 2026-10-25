"""SVG export: the curves a distance field already *is*, written as curves.

This is the one asset family in the project that has a vector form and a raster
form of the same thing.  A panel is four rounded rectangles and a polar ornament,
and a rounded rectangle is four lines and four arcs -- so the vector export is not
a second renderer of the same picture, it is the *same geometry* in a form a layout
engine can scale to any size, which is what a nine-slice exists to work around.  On
a sheet of chrome that is strictly better: no cuts, no insets, no mip chain.

The writer is therefore small and deliberately not a renderer.  It knows paths,
rounded rectangles, polar curves, gradients and groups, and it knows nothing about
panels; :mod:`gen.ui.vector` is what knows which paths a panel is.

Two things are worth stating up front, because they are where a vector back-end
usually goes wrong:

* **Coordinates here are screen coordinates** -- ``x`` right, ``y`` down, as SVG
  is.  The project's worlds are y-up, so the flip happens at the call site, once,
  rather than inside the path builder.  A ``scale(1,-1)`` in the middle of a path
  is how a rounded rectangle comes out with its corners on the wrong side and its
  arcs turning the wrong way.
* **A gradient is a list of stops, not a function.**  SVG interpolates linearly
  between stops, in sRGB, and :mod:`core.palette` interpolates in OKLab, so a
  palette ramp is written as *sampled* stops (:func:`linear_gradient`) and the error
  is a number the caller is expected to bound rather than a hope.

The paths come with the points they were built from (:func:`rounded_rect`,
:func:`polar`), so a test can prove that every curve lies on the zero set of the
very field the raster thresholds.  That is the guarantee that makes this safe: the
two cannot drift, because there is only one source of geometry.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "num",
    "path",
    "rounded_rect",
    "polar",
    "linear_gradient",
    "group",
    "Document",
]


def num(v) -> str:
    """A compact number: three decimals, no trailing zeros, no ``-0``.

    SVG files are read by people, and ``-0`` and ``3.0000000000000004`` are the two
    things that make a generated one look generated.
    """
    v = round(float(v), 3)
    if v == 0.0:
        v = 0.0
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s or "0"


def _pair(p) -> str:
    return f"{num(p[0])},{num(p[1])}"


def path(points, close: bool = True) -> str:
    """A polyline through ``points`` as an SVG ``d`` attribute."""
    pts = list(points)
    if not pts:
        raise ValueError("a path needs at least one point")
    d = "M " + " L ".join(_pair(p) for p in pts)
    return d + " Z" if close else d


def rounded_rect(cx: float, cy: float, hw: float, hh: float, radius: float):
    """A rounded rectangle, clockwise on screen, as ``(d, vertices, on_curve)``.

    ``radius`` is clamped to the box, and a radius of zero gives a plain rectangle
    rather than a degenerate arc (SVG draws a zero-radius arc as a straight line,
    but a path that says what it means is worth the four lines).

    Two lists come back, and they are not the same list.  ``vertices`` are the eight
    points the ``d`` attribute literally contains -- the places a straight run meets
    an arc -- and they are what proves the coordinate convention: each of them,
    flipped, is a token in the path.  ``on_curve`` adds samples *along* each arc,
    which is where a wrong sweep flag would show: an arc that bulged the wrong way
    still joins the same two vertices, so the vertices alone cannot see it.  Both
    are in *screen* coordinates here; the caller flips.
    """
    hw, hh = float(hw), float(hh)
    r = float(np.clip(float(radius), 0.0, max(min(hw, hh), 0.0)))
    x0, x1 = cx - hw, cx + hw
    y0, y1 = cy - hh, cy + hh

    if r <= 1e-9:
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return path(pts), pts, pts

    arcs = (
        ((x1 - r, y0 + r), -0.5 * math.pi, 0.0),
        ((x1 - r, y1 - r), 0.0, 0.5 * math.pi),
        ((x0 + r, y1 - r), 0.5 * math.pi, math.pi),
        ((x0 + r, y0 + r), math.pi, 1.5 * math.pi),
    )
    d = (
        f"M {_pair((x0 + r, y0))} "
        f"L {_pair((x1 - r, y0))} "
        f"A {num(r)},{num(r)} 0 0 1 {_pair((x1, y0 + r))} "
        f"L {_pair((x1, y1 - r))} "
        f"A {num(r)},{num(r)} 0 0 1 {_pair((x1 - r, y1))} "
        f"L {_pair((x0 + r, y1))} "
        f"A {num(r)},{num(r)} 0 0 1 {_pair((x0, y1 - r))} "
        f"L {_pair((x0, y0 + r))} "
        f"A {num(r)},{num(r)} 0 0 1 {_pair((x0 + r, y0))} Z"
    )
    vertices = [(x0 + r, y0), (x1 - r, y0), (x1, y0 + r), (x1, y1 - r),
                (x1 - r, y1), (x0 + r, y1), (x0, y1 - r), (x0, y0 + r)]
    on_curve = list(vertices)
    for (acx, acy), a0, a1 in arcs:
        for i in range(1, 4):
            a = a0 + (a1 - a0) * i / 4.0
            on_curve.append((acx + r * math.cos(a), acy + r * math.sin(a)))
    return d, vertices, on_curve


def polar(cx: float, cy: float, radius, n: int = 256):
    """A closed polar curve ``r(theta)``, as ``(d, vertices, on_curve)``.

    ``radius`` is a callable of the angle in radians -- which is exactly what
    :func:`gen.ui.frame.boss_radius` is -- and ``n`` decides how finely it is
    sampled.  A polar curve is not a Bezier, so this is a polyline and the only
    thing to be said about it is *how close* it is: the error is bounded by the
    radius's own variation over one segment, and 256 segments is well past what a
    32-pixel ornament can show.

    Every sample is a vertex of the path here, so the two lists are the same list;
    they are returned separately to keep one shape for :func:`rounded_rect`'s sake,
    where they are emphatically not the same.
    """
    theta = np.linspace(0.0, 2.0 * math.pi, int(n), endpoint=False)
    r = np.asarray(radius(theta), dtype=np.float64)
    pts = [(float(cx + ri * math.cos(t)), float(cy + ri * math.sin(t)))
           for ri, t in zip(np.broadcast_to(r, theta.shape), theta)]
    return path(pts), pts, pts


def linear_gradient(colours, gid: str, x1, y1, x2, y2, opacity=None) -> str:
    """A ``<linearGradient>`` from a list of sRGB colours, sampled as stops.

    The list is the stops, evenly spaced from 0 to 1, so a caller that wants the
    error small asks for more of them: :func:`core.palette.ramp` gives exactly this
    list, which is why a palette ramp survives the trip into a format that
    interpolates somewhere else.
    """
    from core import palette

    cols = list(colours)
    if len(cols) < 2:
        raise ValueError("a gradient needs at least two stops")
    parts = [
        f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
        f'x1="{num(x1)}" y1="{num(y1)}" x2="{num(x2)}" y2="{num(y2)}">'
    ]
    for i, c in enumerate(cols):
        off = i / (len(cols) - 1)
        a = ""
        if opacity is not None and float(opacity) < 1.0:
            a = f' stop-opacity="{num(opacity)}"'
        parts.append(f'<stop offset="{num(off)}" stop-color="{palette.srgb_to_hex(c)}"{a}/>')
    parts.append("</linearGradient>")
    return "".join(parts)


def group(children, **attrs) -> str:
    """Wrap elements in a ``<g>``, so a whole layer can carry one attribute.

    Attributes go out in sorted order, because the point of a generated file is that
    the same input gives the same bytes: a dictionary that iterated differently
    between two runs would change the file of an identical asset.
    """
    body = "".join(children)
    if not attrs:
        return f"<g>{body}</g>"
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in sorted(attrs.items()))
    return f"<g {a}>{body}</g>"


class Document:
    """A bare SVG document: a viewBox, some defs, some elements, nothing else.

    Not a scene graph and not a painter -- the two things a back-end of this size
    gets wrong are being clever about layout and being clever about state.  A panel
    is a handful of paths, so a list of strings and a list of definitions is the
    whole model.
    """

    def __init__(self, width, height, view_box=None):
        self.width = float(width)
        self.height = float(height)
        self.view_box = view_box
        self.defs = []
        self.elements = []

    def define(self, element: str) -> str:
        self.defs.append(element)
        return element

    def add(self, element: str) -> None:
        self.elements.append(element)

    def text(self) -> str:
        """The document as text, deterministic down to the newline."""
        vb = self.view_box
        view = (f'{num(vb[0])} {num(vb[1])} {num(vb[2])} {num(vb[3])}'
                if vb is not None else f'0 0 {num(self.width)} {num(self.height)}')
        head = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{num(self.width)}" height="{num(self.height)}" '
            f'viewBox="{view}">\n'
        )
        defs = ""
        if self.defs:
            defs = "<defs>\n" + "\n".join(self.defs) + "\n</defs>\n"
        body = "\n".join(self.elements)
        return f"{head}{defs}{body}\n</svg>\n"
