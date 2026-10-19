"""Province names, set on the map in the project's own face.

The map has carried the *names* of its provinces in its data since provinces
existed, and for most of that time the picture did not carry them.  The reason was
written down, which is the only way a reason can expire: this project shipped
exactly one font -- Pillow's bitmap -- and a game has its own, so text belonged to
the preview while the sidecar carried what a game would label with.
``gen/text/`` made that sentence false.  There is a font now, it is ours, it is a
function of a seed like every other asset, and so a label is an *asset* and not a
caption: it is drawn at the size its province can hold, it is regenerated with the
map, and it is the same label at every scale the map is drawn at.

Three things make a placement a decision rather than a wish.

**Where.**  A province's centre of mass is not where a name goes; the place a name
goes is where the province is widest, which is the cell farthest from its own edge
-- the pole of inaccessibility, one breadth-first search per province.  A name set
over a narrow isthmus of its own province crosses the border and reads as the
neighbour's, which is the one failure a map label cannot have.

**How big.**  A name is set as large as the province can hold and no larger, and
the test is geometric rather than a matter of taste: the label's box has to fit
inside the province's inscribed circle, which is a condition on the box's
*diagonal* -- a wide box centred in a circle pokes out of it at the corners.  A
long name therefore comes out smaller than a short one, which is what a map does,
and a name that cannot be set at a legible size is skipped and **reported** rather
than squeezed, because a squeezed label is one nobody reads.

**What it sits on.**  The ground is a painting, not a flat colour, so the ink
cannot assume a background.  Every label is a dark ink over a pale halo, and the
halo is the same glyphs drawn once and blitted at eight offsets around the centre:
a halo drawn as a stroke would be a second font, and a halo drawn as a plate would
be a rectangle on a painted map.
"""
from __future__ import annotations

import numpy as np

from core import palette
from gen.text import typeset

from .grid import HexGrid

__all__ = ["FACE", "INK", "HALO", "layout", "draw", "census"]

#: The ink of a label, and the paper it is separated from the ground by.  Text ink
#: is its own colour: a name has to read over a pastel wash and over a forest, and a
#: line's ink is chosen to sit *in* the picture rather than on top of it.
INK = palette.hex_to_srgb("#1a2430")
HALO = palette.hex_to_srgb("#f5f0e3")

#: The face a map names its provinces in.  The sans pen is the UI face, and a map
#: label is a UI job -- it is read at eight pixels or not at all -- so it is asked
#: for a little heavier than the body text of a screen, where the eye has a paragraph
#: to lean on.
FACE = {"face": "sans", "weight": 0.105, "tracking": 0.02}

#: How much of the inscribed radius the label's *diagonal* may use.  Not 1.0: the
#: inscribed circle is computed on hex centres, and the edge of a province is half a
#: cell beyond the last one, so a box that exactly fills the circle can still touch
#: the frontier line.
FILL = 0.88

#: The largest em a label may take, and the smallest that is still type, as
#: fractions of a hex's step.  A label taller than a hex is a headline in a province,
#: and one below this floor is a smudge: both are refused rather than drawn.
SMAX = 0.85
SMIN = 0.42

#: How much a label shrinks when it collides with one already placed, and how many
#: times it may try.  Eight steps take an em to about a third of where it started.
SHRINK = 0.85
TRIES = 8

#: The air kept between two labels, as a fraction of a hex's step.  Two names set
#: flank to flank read as one word, and the failure mode of a crowded map is not
#: overlap but *merger*.
AIR = 0.30

#: How far the halo is blitted from the centre, in ems, and the eight directions.
HALO_GAP = 0.10
_RING = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def _spec(face=None) -> dict:
    return typeset.spec_of({**FACE, **(face or {})})


def _rings(grid: HexGrid, inside: set) -> dict:
    """Distance, in cells, from every cell of a province to its own edge.

    A cell on the edge -- one that has a neighbour outside the province, or off the
    map -- is ring zero, and every cell after it is one more ring in.  This is a
    breadth-first search from the edge rather than from the centre because the
    answer wanted is the *width* of the province, and a province is a shape with a
    ragged edge, not a disc.
    """
    rings = {}
    queue = []
    for rc in inside:
        r, c = rc
        for d in range(6):
            nr, nc = (int(v) for v in grid.nbr[r, c, d])
            if nr < 0 or (nr, nc) not in inside:
                rings[rc] = 0
                queue.append(rc)
                break
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for d in range(6):
            nr, nc = (int(v) for v in grid.nbr[r, c, d])
            if nr < 0 or (nr, nc) not in inside or (nr, nc) in rings:
                continue
            rings[(nr, nc)] = rings[(r, c)] + 1
            queue.append((nr, nc))
    return rings


def _ink(name: str, spec: dict):
    """The ink box of a name, in ems: ``(width, height, x0, y0)``."""
    x0, y0, x1, y1 = typeset.ink_bounds(name, spec)
    return float(x1 - x0), float(y1 - y0), float(x0), float(y0)


def _inside(grid: HexGrid, lab, box, index: int) -> bool:
    """Is every point of a world box inside one province, sampled on a lattice?

    The inscribed circle is an estimate: hexagons are not discs, and a wide box that
    fits a circle of the radius the rings imply can still cross a frontier in the
    direction where the province is narrow.  So the answer is read off the *data* --
    the province index of every cell a lattice of points falls in -- rather than
    inferred from the shape.  Twenty-seven points across the box, nine by three,
    which is finer than the ink of a label at any size it is set at.
    """
    x0, y0, x1, y1 = box
    X, Y = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 3))
    cell = grid.cell(X, Y)
    r = cell.rc[..., 0]
    c = cell.rc[..., 1]
    ok = np.asarray(cell.ok) & (np.asarray(lab)[r, c] == int(index))
    return bool(ok.all())


def layout(grid: HexGrid, res: dict, face=None, fill: float = FILL) -> list:
    """Where every province's name goes, how big it is, and whether it fits.

    One entry per province, in province order, so a caller never has to un-sort
    anything to know which name is which.  A drawn entry has ``em``, ``cell``,
    ``box`` and ``spec``; a skipped one has ``drawn=False`` and a ``why`` that says
    which of the two reasons it was -- there was no room, or the room was already
    taken by a neighbour's label.

    Placement is decided largest-province-first and the list is returned in
    province order, which is the same rule the atlas uses for its rectangles: the
    order a caller reads is not the order the packer worked in.
    """
    spec = _spec(face)
    lab = np.asarray(res["labels"])
    step = float(grid.size) * float(np.sqrt(3.0))
    entries = []
    for i, p in enumerate(res["provinces"]):
        name = str(p["name"])
        inside = set()
        for j in np.asarray(p["cells"]).reshape(-1):
            inside.add((int(j) // grid.cols, int(j) % grid.cols))
        rings = _rings(grid, inside)
        cell = max(rings.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        radius = (rings[cell] + 0.5) * step
        w, h, x0, y0 = _ink(name, spec)
        em = min(SMAX * step, 2.0 * fill * radius / float(np.hypot(w, h)))
        entries.append({
            "index": i,
            "name": name,
            "cell": [int(cell[0]), int(cell[1])],
            "radius": float(radius),
            "em": float(em),
            "drawn": False,
            "spec": dict(spec),
        })

    placed = []
    air = 0.5 * AIR * step
    for e in sorted(entries, key=lambda e: (-e["radius"], e["index"])):
        if e["em"] < SMIN * step:
            e["why"] = "no room"
            continue
        w, h, _x0, _y0 = _ink(e["name"], e["spec"])
        cx, cy = grid.centre_of(e["cell"])
        em = min(e["em"], SMAX * step)
        why = "no room"
        for _ in range(TRIES):
            hw, hh = 0.5 * w * em, 0.5 * h * em
            box = (cx - hw, cy - hh, cx + hw, cy + hh)
            if _inside(grid, lab, box, e["index"]):
                why = "crowded"
                room = (box[0] - air, box[1] - air, box[2] + air, box[3] + air)
                if not any(_overlap(room, b) for b in placed):
                    e["drawn"] = True
                    e["em"] = float(em)
                    e["box"] = [float(v) for v in box]
                    placed.append(room)
                    break
            if em * SHRINK < SMIN * step:
                break
            em *= SHRINK
        if not e["drawn"]:
            e["why"] = why
    return entries


def _overlap(a, b, gap: float = 0.0) -> bool:
    """Do two world-space boxes overlap, with a little air between them?"""
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0]
                or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def _pixel(canvas, x: float, y: float) -> tuple:
    """The pixel a world point lands on, as the blit wants it."""
    sx = float(canvas.scale)
    cx, cy = canvas.center
    return ((x - cx) * sx + canvas.w / 2.0 - 0.5,
            (cy - y) * sx + canvas.h / 2.0 - 0.5)


def draw(canvas, grid: HexGrid, entries, ink=INK, halo=HALO,
         halo_gap: float = HALO_GAP) -> None:
    """Set every drawn name on the canvas: halo first, then the ink over it."""
    halo_rgb = palette.srgb_to_linear(halo)
    for e in entries:
        if not e.get("drawn"):
            continue
        em = float(e["em"])
        cx, cy = grid.centre_of(e["cell"])
        px, py = _pixel(canvas, cx, cy)
        ghost = typeset.draw(e["name"], e["spec"], em=em, pad=0.10, colour=halo)
        ox = int(round(px - ghost.w / 2.0))
        oy = int(round(py - ghost.h / 2.0))
        r = max(1, int(round(halo_gap * em)))
        for dx, dy in _RING:
            canvas.blit(np.broadcast_to(halo_rgb, ghost.rgb.shape), ghost.a,
                        ox + dx * r, oy + dy * r)
        text = typeset.draw(e["name"], e["spec"], em=em, pad=0.10, colour=ink)
        canvas.blit(text.rgb, text.a, ox, oy)


def census(entries) -> dict:
    """What was drawn and what was not, in the shape a sidecar wants.

    A skipped label is a fact about the map and not a failure of the build, so it
    is written down with its reason: a game that labels provinces itself can see
    which ones this map left without type, and why.
    """
    drawn = [e for e in entries if e.get("drawn")]
    return {
        "drawn": len(drawn),
        "skipped": [{"index": int(e["index"]), "name": e["name"], "why": e["why"]}
                    for e in entries if not e.get("drawn")],
        "em": [round(min(e["em"] for e in drawn), 3),
               round(max(e["em"] for e in drawn), 3)] if drawn else [],
        "face": dict(drawn[0]["spec"]) if drawn else None,
    }
