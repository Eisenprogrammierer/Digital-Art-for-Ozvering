"""A panel as curves: the same geometry the raster thresholds, in vector form.

Everything a panel *is* is a rounded rectangle or a polar curve, and both of those
are curves, so the UI package is the one asset family here that has an exact vector
form.  What is more useful than that: because the paths are built from
:class:`gen.ui.frame.Panel` -- the same object the raster pass thresholds -- the two
cannot disagree about the *shape*.  ``tests/test_ui.py`` proves it directly, by
asserting that every point the paths were built from lies on the zero set of the
very field the pixel renderer thresholds.

So this is not a second look at a panel.  It is the panel, and it is worth being
explicit about what a vector cannot carry, because that is the whole difference:

* **the light is not in it.**  The moulding's bevel, the ornament's dome, the
  shadow and a material body are all *shading* -- they are what
  :func:`core.render.shade` does to a distance field, and a bevel is a ramp, not a
  curve.  What is here is the geometry and the palette; an engine that draws its
  own chrome does its own lighting, and that is the normal case for a UI engine
  anyway;
* **a material has no vector form at all.**  Wood and stone are noise fields, so a
  style with a material is *refused* rather than quietly exported as a flat
  rectangle -- an export that silently drops the one thing it cannot draw is worse
  than an error, because it ships.

The payoff is real and it is the opposite of the nine-slice's: a vector panel needs
no cuts, no insets and no mip chain, because the curves are exact at every size.  A
nine-slice is what you do when the asset is *pixels*; this is what you do when the
layout engine can draw.

World coordinates here are y-up and the centre of the panel is the origin, exactly
as in :mod:`gen.ui.frame`; the flip to SVG's y-down happens in :func:`curves` and
nowhere else.
"""
from __future__ import annotations

from collections import OrderedDict, namedtuple
from pathlib import Path

from core import palette
from export import svg

from . import frame, paint

__all__ = ["STOPS", "Curve", "curves", "document", "write"]

#: How many stops a palette ramp is written with.  SVG interpolates linearly
#: between stops and :mod:`core.palette` interpolates in OKLab, so this is a
#: sampled approximation of a two-stop ramp; 17 is past the point where the two
#: differ by a visible amount on any style here, and the test measures rather than
#: assumes it.
STOPS = 17

#: How finely the ornament's radius function is drawn.  A polar curve is not a
#: Bezier, so it goes out as a polyline, and the only thing to say about that is how
#: close it is: the sagitta of a chord is ``r*(1 - cos(pi/n))``, which for the
#: largest ornament on the shelf -- a radius near 8 design pixels -- is 2.4e-3 pixels
#: at this count.  A quarter of a thousandth of a pixel is not a thing a screen or
#: an engine can show, and the file stays a few kilobytes.
BOSS_SEGMENTS = 128

#: A curve and the points it was built from.  The ``d`` is written for SVG, where
#: ``y`` runs down; ``vertices`` and ``on_curve`` are in the package's world, where
#: ``y`` runs up -- because the fields are the source of truth, and the point of
#: returning the points is that a test can ask *them*.  ``vertices`` are the points
#: the path literally contains and ``on_curve`` adds samples along an arc, which is
#: the only way to see a sweep flag that turned the wrong way.
Curve = namedtuple("Curve", "d vertices on_curve")


def _world(points):
    """Screen points as the package's world: SVG is y-down, the world is y-up."""
    return [(float(x), -float(y)) for x, y in points]


# --------------------------------------------------------------------------
# the geometry
# --------------------------------------------------------------------------
def curves(panel: frame.Panel, kind: str = "panel") -> "OrderedDict[str, Curve]":
    """Every curve of a panel, named, ready to write.

    ``outer`` and ``inner`` are the two edges of the frame band (the silhouette and
    the hole), ``rule`` is the moulding's centreline -- stroked, not filled, because
    that is what a line is -- and ``boss0..3`` are the corner ornaments, whose
    radius function is the one place a Fourier series is used.  A panel too small to
    have a hole (see :meth:`gen.ui.frame.Panel.has_hole`) simply has no ``inner``,
    and then the band is the silhouette, which is what the band is.

    The ornament's angle is *negated* on the way through, and that is the entire
    content of the coordinate flip: a screen point at angle ``t`` around the centre
    is a world point at angle ``-t``, because the flip mirrors the plane.  Getting
    this wrong costs a mirrored scallop, which is the kind of error a picture shows
    and a test does not.  So the flip is done here, once, and the points that come
    back are converted *back* into the world -- a curve's ``d`` is for SVG and its
    ``on_curve`` is for the field it came from.
    """
    if kind not in paint.KINDS:
        raise ValueError(f"kind must be one of {paint.KINDS}, not {kind!r}")

    out = OrderedDict()

    def rounded(cx, cy, hw, hh, radius):
        # The radius arrives already fitted (`Panel.outer_radius` and friends are the
        # one place the rule lives), so this is a shape, not a second opinion.
        d, vertices, on_curve = svg.rounded_rect(cx, -cy, hw, hh, radius)
        return Curve(d, _world(vertices), _world(on_curve))

    out["outer"] = rounded(0.0, 0.0, panel.hw, panel.hh, panel.outer_radius())

    if panel.has_hole():
        cx, cy, hw, hh = panel.inner_box()
        out["inner"] = rounded(cx, cy, hw, hh, panel.inner_radius())

        if panel.has_rule():
            g = panel.spec["rule_gap"]
            out["rule"] = rounded(cx, cy, hw - g, hh - g, panel.rule_radius())

    if panel.has_boss():
        harmonics = panel.spec["boss_harmonics"]
        amp = panel.spec["boss_amp"]
        for i in range(4):
            bx, by = panel.boss_centre(i)
            r0 = panel.boss_radius_at(i)
            d, vertices, on_curve = svg.polar(
                bx, -by, lambda t, r0=r0: frame.boss_radius(-t, r0, harmonics, amp),
                n=BOSS_SEGMENTS)
            out[f"boss{i}"] = Curve(d, _world(vertices), _world(on_curve))
    return out


def band(curves_: "OrderedDict[str, Curve]") -> str:
    """The frame band as one even-odd path: the silhouette with its hole punched out.

    One path rather than two, because even-odd is what makes the hole genuinely
    transparent -- a hollow frame is meant to lie over content, and two separately
    filled paths would give a band you could see the body through.  A panel with no
    hole (too small for its own frame) is just the silhouette, for the same reason.
    """
    d = curves_["outer"].d
    if "inner" in curves_:
        d = f'{d} {curves_["inner"].d}'
    return d


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------
def document(spec: dict, size, kind: str = "panel", state: str = "idle",
             scale: float = 1.0, steps: int = STOPS) -> str:
    """One panel, frame or button face as an SVG document.

    The layer order is the raster's, because it is the drawing: body, band, rule,
    ornament.  What is missing between the layers is the light.
    """
    panel = frame.Panel(spec, size, scale)
    s = panel.spec
    if s["material"] != "none":
        raise ValueError(
            f"a material body is a noise field and has no curves; export {s['material']!r} "
            f"as PNG, or set material to 'none' for the vector form")
    body, body_low, edge, accent = paint.tones(s, state)
    cs = curves(panel, kind)

    # User units are design pixels, so the document is resolution-independent: the
    # width and height are output pixels and the viewBox is the design, which is
    # the whole reason a vector asset needs no nine-slice.
    doc = svg.Document(panel.w, panel.h,
                       view_box=(-panel.hw, -panel.hh, 2.0 * panel.hw, 2.0 * panel.hh))

    if kind == "panel":
        lo = palette.mix(body, body_low, s["gradient"])
        # The raster's ramp runs from the top edge (t=0) to the bottom (t=1), and
        # screen y runs down, so the start stop is at -hh.
        doc.define(svg.linear_gradient(palette.ramp([body, lo], int(steps)), "body",
                                       0.0, -panel.hh, 0.0, panel.hh, opacity=s["alpha"]))
        doc.add(f'<path d="{cs["outer"].d}" fill="url(#body)"/>')

    # The band, twice over: it is the asset for a hollow frame, and the moulding on
    # top of the body for a panel.
    doc.add(f'<path d="{band(cs)}" fill="{palette.srgb_to_hex(edge)}" '
            f'fill-rule="evenodd"/>')

    if "rule" in cs:
        doc.add(f'<path d="{cs["rule"].d}" fill="none" '
                f'stroke="{palette.srgb_to_hex(s["ink"])}" '
                f'stroke-width="{svg.num(s["rule"])}" stroke-opacity="0.55"/>')

    if "boss0" in cs:
        # The raster clips the ornament to the silhouette (a panel) or to the band
        # (a frame, so a rivet cannot show through the hole); so does this, once for
        # the whole group rather than once per mark.
        hole = (f'<path d="{band(cs)}" clip-rule="evenodd"/>' if kind == "frame"
                else f'<path d="{cs["outer"].d}"/>')
        doc.define(f'<clipPath id="band">{hole}</clipPath>')
        accent_hex = palette.srgb_to_hex(accent)
        ink_hex = palette.srgb_to_hex(s["ink"])
        marks = []
        for i in range(4):
            c = cs[f"boss{i}"]
            marks.append(f'<path d="{c.d}" fill="{accent_hex}"/>')
            w = panel.boss_outline(i)
            if w > 0.05:
                marks.append(f'<path d="{c.d}" fill="none" stroke="{ink_hex}" '
                             f'stroke-width="{svg.num(2.0 * w)}" stroke-opacity="0.45"/>')
        doc.add(svg.group(marks, clip_path="url(#band)"))
    return doc.text()


def write(spec: dict, size, path, kind: str = "panel", state: str = "idle",
          scale: float = 1.0) -> Path:
    """Write one panel as SVG.  Returns the path written, like every other exporter."""
    text = document(spec, size, kind=kind, state=state, scale=scale)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
