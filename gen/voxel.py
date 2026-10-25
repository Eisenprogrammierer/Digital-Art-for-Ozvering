"""Axonometric voxel geometry: boxes in, five orthographic views out.

The renderer is deliberately small because it is the shared contract for three
different generators.  A building, a tank and a map column are all boxes in the
same model space; a view is a camera orientation; and every surface is filled
from the analytic polygon distance field rather than from a mesh library.

Model space is right-handed with ``x`` east, ``y`` south and ``z`` up.  The four
iso views rotate that model around its vertical axis and then project it with the
classic 2:1 axonometric basis::

    screen_x = (x - y) * cos(30 degrees)
    screen_y = (x + y) * sin(30 degrees) - z * rise

The top view is a true orthographic plan, not a squashed isometric one, so a
roof reads as a roof and a footprint can be measured from the sheet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core import palette, render as renderlib, sdf
from core.canvas import Canvas

__all__ = [
    "Part",
    "VIEWS",
    "model",
    "box",
    "render",
    "views",
    "view_sheet",
    "projected_bounds",
]

VIEWS = ("north", "west", "south", "east", "top")

_COS30 = 0.8660254037844386
_SIN30 = 0.5
_RISE = 0.72
_ANGLES = {"south": 0.0, "east": 0.5 * np.pi, "north": np.pi, "west": 1.5 * np.pi}


@dataclass(frozen=True)
class Part:
    """One axis-aligned box with a flat sRGB colour."""

    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    colour: object = "#b8b8b8"
    alpha: float = 1.0
    solid: bool = True
    outline: bool = True

    def __post_init__(self):
        lo = (min(self.x0, self.x1), min(self.y0, self.y1), min(self.z0, self.z1))
        hi = (max(self.x0, self.x1), max(self.y0, self.y1), max(self.z0, self.z1))
        object.__setattr__(self, "x0", lo[0])
        object.__setattr__(self, "y0", lo[1])
        object.__setattr__(self, "z0", lo[2])
        object.__setattr__(self, "x1", hi[0])
        object.__setattr__(self, "y1", hi[1])
        object.__setattr__(self, "z1", hi[2])
        rgb = palette.hex_to_srgb(self.colour) if isinstance(self.colour, str) else self.colour
        object.__setattr__(self, "colour", np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0))

    @property
    def centre(self):
        return ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5, (self.z0 + self.z1) * 0.5)


@dataclass
class Model:
    """An ordered collection of boxes and the small construction helpers."""

    parts: list = field(default_factory=list)

    def add(self, x0, y0, z0, x1, y1, z1, colour="#b8b8b8", alpha=1.0,
            solid=True, outline=True) -> Part:
        part = Part(x0, y0, z0, x1, y1, z1, colour, alpha, solid, outline)
        self.parts.append(part)
        return part

    def clone(self) -> "Model":
        return Model(list(self.parts))

    def bounds(self):
        if not self.parts:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            min(p.x0 for p in self.parts), min(p.y0 for p in self.parts),
            min(p.z0 for p in self.parts), max(p.x1 for p in self.parts),
            max(p.y1 for p in self.parts), max(p.z1 for p in self.parts),
        )

    def centre(self):
        x0, y0, z0, x1, y1, z1 = self.bounds()
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5)

    def offset(self, dx=0.0, dy=0.0, dz=0.0) -> "Model":
        out = Model()
        for p in self.parts:
            out.add(p.x0 + dx, p.y0 + dy, p.z0 + dz,
                    p.x1 + dx, p.y1 + dy, p.z1 + dz,
                    p.colour, p.alpha, p.solid, p.outline)
        return out


def model(parts=()) -> Model:
    """Build a model from box tuples or :class:`Part` instances."""
    out = Model()
    for part in parts:
        if isinstance(part, Part):
            out.parts.append(part)
        else:
            out.add(*part)
    return out


def box(model_, x0, y0, z0, x1, y1, z1, colour="#b8b8b8", **kw) -> Part:
    return model_.add(x0, y0, z0, x1, y1, z1, colour, **kw)


def _rotate(x, y, view: str, centre) -> tuple:
    if view == "top":
        return x, y
    angle = _ANGLES[view]
    if angle == 0.0:
        return x, y
    cx, cy = centre[0], centre[1]
    dx, dy = x - cx, y - cy
    ca, sa = np.cos(angle), np.sin(angle)
    return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca


def _project_point(x, y, z, view: str, centre) -> tuple:
    x, y = _rotate(x, y, view, centre)
    if view == "top":
        return x, y
    return (x - y) * _COS30, (x + y) * _SIN30 - z * _RISE


def _corners(part: Part):
    for x in (part.x0, part.x1):
        for y in (part.y0, part.y1):
            for z in (part.z0, part.z1):
                yield x, y, z


def _projected(parts, view: str, centre):
    out = []
    for part in parts:
        pts = [_project_point(x, y, z, view, centre) for x, y, z in _corners(part)]
        out.append((part, np.asarray(pts, dtype=np.float64)))
    return out


def projected_bounds(parts, view: str, centre=None):
    """Raw projected bounds of all boxes in one view."""
    if centre is None:
        centre = Model(list(parts)).centre()
    pts = np.concatenate([p for _, p in _projected(parts, view, centre)])
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _fit(parts, view, size, margin, centre, bounds=None):
    if bounds is None:
        bounds = projected_bounds(parts, view, centre)
    x0, y0, x1, y1 = bounds
    span = max(x1 - x0, y1 - y0, 1e-6)
    scale = (float(size) - 2.0 * float(margin)) / span
    ox = (size - (x1 - x0) * scale) * 0.5 - x0 * scale
    oy = (size - (y1 - y0) * scale) * 0.5 - y0 * scale
    return scale, ox, oy


_VIEW_FACES = {
    "south": ("east", "south"),
    "east": ("east", "north"),
    "north": ("west", "north"),
    "west": ("west", "south"),
}

_FACE_NORMALS = {
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
    "south": (0.0, 1.0),
    "north": (0.0, -1.0),
}

_LIGHT = np.asarray((-0.45, 0.65), dtype=np.float64)
_LIGHT /= np.linalg.norm(_LIGHT)


def _side_factor(name: str) -> float:
    nx, ny = _FACE_NORMALS[name]
    return 0.54 + 0.24 * max(0.0, nx * _LIGHT[0] + ny * _LIGHT[1])


def _face_points(part: Part, view: str, centre, scale, ox, oy):
    def p(x, y, z):
        px, py = _project_point(x, y, z, view, centre)
        return (px * scale + ox, py * scale + oy)

    top = [p(part.x0, part.y0, part.z1), p(part.x1, part.y0, part.z1),
           p(part.x1, part.y1, part.z1), p(part.x0, part.y1, part.z1)]
    if view == "top":
        return [(top, 1.04)]
    faces = []
    for name in _VIEW_FACES[view]:
        if name == "east":
            pts = [p(part.x1, part.y0, part.z0), p(part.x1, part.y1, part.z0),
                   p(part.x1, part.y1, part.z1), p(part.x1, part.y0, part.z1)]
        elif name == "west":
            pts = [p(part.x0, part.y1, part.z0), p(part.x0, part.y0, part.z0),
                   p(part.x0, part.y0, part.z1), p(part.x0, part.y1, part.z1)]
        elif name == "south":
            pts = [p(part.x0, part.y1, part.z0), p(part.x1, part.y1, part.z0),
                   p(part.x1, part.y1, part.z1), p(part.x0, part.y1, part.z1)]
        else:
            pts = [p(part.x1, part.y0, part.z0), p(part.x0, part.y0, part.z0),
                   p(part.x0, part.y0, part.z1), p(part.x1, part.y0, part.z1)]
        faces.append((pts, _side_factor(name)))
    faces.append((top, 1.03))
    return faces


def _window(canvas: Canvas, pts, pad: float):
    arr = np.asarray(pts)
    x0 = max(0, int(np.floor(arr[:, 0].min() - pad)) - 1)
    y0 = max(0, int(np.floor(arr[:, 1].min() - pad)) - 1)
    x1 = min(canvas.w, int(np.ceil(arr[:, 0].max() + pad)) + 2)
    y1 = min(canvas.h, int(np.ceil(arr[:, 1].max() + pad)) + 2)
    return x0, y0, x1, y1


def _fill(canvas: Canvas, pts, colour, alpha=1.0, outline=True, line=0.85):
    if alpha <= 0.0:
        return
    x0, y0, x1, y1 = _window(canvas, pts, 2.0)
    if x1 <= x0 or y1 <= y0:
        return
    X, Y = canvas.grid()
    Xs = X[y0:y1, x0:x1]
    Ys = Y[y0:y1, x0:x1]
    d = sdf.polygon(Xs, Ys, pts)
    cov = renderlib.coverage(d, canvas.dx)
    if cov.any():
        canvas.over_srgb(colour, cov * float(alpha), x0, y0)
    if outline and line > 0.0:
        edge = np.clip(1.0 - np.abs(d) / float(line), 0.0, 1.0)
        if edge.any():
            dark = palette.mix(colour, (0.06, 0.07, 0.09), 0.38)
            canvas.over_srgb(dark, edge * cov * float(alpha) * 0.50, x0, y0)


def _tone(colour, factor):
    return np.clip(np.asarray(colour, dtype=np.float64) * float(factor), 0.0, 1.0)


def _base_shadow(canvas: Canvas, parts, view, centre, scale, ox, oy):
    if view == "top" or not parts:
        return
    pts = np.asarray([p for _, p in _projected(parts, view, centre)]).reshape(-1, 2)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    cx = (lo[0] + hi[0]) * 0.5 * scale + ox
    cy = (lo[1] + hi[1]) * 0.5 * scale + (hi[1] - lo[1]) * 0.08 * scale + oy
    rx = max((hi[0] - lo[0]) * scale * 0.44, 1.0)
    ry = max(rx * 0.24, 1.0)
    X, Y = canvas.grid()
    d = sdf.ellipse(X, Y, rx, ry, cx, cy)
    a = renderlib.coverage(d, canvas.dx) * 0.16
    if a.any():
        canvas.over_srgb((0.05, 0.06, 0.08), a)


def render(parts, size: int = 256, view: str = "south", margin: int = 12,
           background=None, shadow: bool = True, outline: bool = True,
           fit_bounds=None) -> Canvas:
    """Render one axonometric view of a model into a square transparent canvas."""
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    parts = list(parts)
    if not parts:
        raise ValueError("a voxel model needs at least one part")
    centre = Model(parts).centre()
    bounds = fit_bounds or projected_bounds(parts, view, centre)
    scale, ox, oy = _fit(parts, view, size, margin, centre, bounds)
    canvas = Canvas(int(size), int(size), scale=1.0,
                    center=(size * 0.5, size * 0.5), y_up=False)
    if background is not None:
        bg = palette.hex_to_srgb(background) if isinstance(background, str) else background
        canvas.fill(bg)
    if shadow:
        _base_shadow(canvas, parts, view, centre, scale, ox, oy)
    def depth(part):
        x, y = _rotate(part.centre[0], part.centre[1], view, centre)
        return (x + y, part.centre[2])

    order = sorted(parts, key=depth)
    if view == "top":
        order = sorted(parts, key=lambda p: (p.z1, p.centre[0] + p.centre[1]))
    for part in order:
        for pts, factor in _face_points(part, view, centre, scale, ox, oy):
            _fill(canvas, pts, _tone(part.colour, factor), part.alpha,
                  outline=part.outline and outline)
    return canvas


def views(parts, size: int = 256, margin: int = 12, background=None,
          shadow: bool = True, outline: bool = True) -> dict:
    """All five projections at one common scale, returned in sheet order."""
    parts = list(parts)
    if not parts:
        raise ValueError("a voxel model needs at least one part")
    centre = Model(parts).centre()
    raw = {view: projected_bounds(parts, view, centre) for view in VIEWS}
    x0 = min(b[0] for b in raw.values())
    y0 = min(b[1] for b in raw.values())
    x1 = max(b[2] for b in raw.values())
    y1 = max(b[3] for b in raw.values())
    common = (x0, y0, x1, y1)
    return {view: render(parts, size=size, view=view, margin=margin,
                         background=background, shadow=shadow, outline=outline,
                         fit_bounds=common)
            for view in VIEWS}


def view_sheet(parts, size: int = 192, margin: int = 12, gap: int = 8,
               background=(0.93, 0.94, 0.96), labels=None):
    """Lay the five projections on one row, with optional captions added by caller."""
    grid = views(parts, size=size, margin=margin, background=None)
    sheet = renderlib.sheet([grid[v] for v in VIEWS], cols=len(VIEWS), gap=gap,
                         background=background)
    return sheet
