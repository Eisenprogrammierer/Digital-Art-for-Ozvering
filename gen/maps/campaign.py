"""The map as a file a game can load: the campaign export.

Every other asset in this project ships as a picture and a sidecar, and the sidecar
is a record *of the picture* -- it says what was drawn, for a reader who can see the
drawing.  A campaign file is the other thing.  It is the map as data, and it is
complete on its own terms: a client that has never heard of this generator can read
it and run the map.

* **which hex is what ground**, one biome index per cell, with the biome table in
  the file as well -- the name, the movement cost, whether it is water, and the
  colour.  The table travels with the map because the table *is* the rules, and
  because a client that had to guess a colour for "marsh" would put a different
  green on the same province;
* **whose hex it is**, one realm index per province and each realm's own colour, so
  the political map can be redrawn without re-deriving the livery;
* **where the roads go**, as the hexes they walk and what they cost to walk --
  because a road is a claim about the ground, and a client has to be able to check
  that claim against the same costs;
* **where the water goes**, the same thing for the rivers, with the drainage that
  decides how wide each one is drawn.

Cells are named the way the generator names them, ``r * cols + c``, row by row, so
a client indexes the same arrays this project indexes.

Nothing in here is a pixel.  The file is a function of ``(seed, cols, rows)`` and the
choices made on top of it -- and the hex size is not one of them, which is why the
roads and the rivers are carried as hexes rather than as polylines.  The same map
exports the same bytes at every scale it can be drawn at; that is the point of
calling the map data.
"""
from __future__ import annotations

import numpy as np

from . import terrain
from .grid import HexGrid

__all__ = ["FORMAT", "campaign"]

#: Bumped when the meaning of a field changes, never when one is added.
FORMAT = "ozart/campaign/1"


def _flat(cells, cols: int) -> list:
    """Cells as the file names them: ``r * cols + c``.

    A caller may hand over flat indices (what the generator uses internally) or
    ``(r, c)`` pairs (what a path is written as), and both mean the same hex.
    """
    out = []
    for cell in cells:
        if isinstance(cell, (tuple, list, np.ndarray)):
            out.append(int(cell[0]) * cols + int(cell[1]))
        else:
            out.append(int(cell))
    return out


def campaign(grid: HexGrid, f: dict, res: dict | None = None, *, seed: int = 0,
             owner=(), realms=(), routes=(), courses=()) -> dict:
    """One map, as the dict that becomes the campaign file.

    Everything is passed in rather than derived here, because every caller already
    derived it: the demo, the CLI and a game all hold the terrain field and the
    provinces before they ask for a file.  A second derivation inside this module
    would be a second map, and it would drift from the first one exactly once --
    the day a parameter is added to one of the two.
    """
    water = np.asarray(f["water"], dtype=bool)
    owner = list(owner)
    doc = {
        "format": FORMAT,
        "seed": int(seed),
        "grid": {
            "cols": int(grid.cols),
            "rows": int(grid.rows),
            "layout": "odd-r offset; centre = (sqrt(3)*(col + (row & 1)/2), 1.5*row) "
                      "in hex circumradii, y up",
            "index": "row-major: r * cols + c",
            "unit": "hex circumradius",
        },
        "biomes": [
            {"key": b.key, "name": b.name, "cost": int(b.cost),
             "water": bool(b.water), "colour": b.hex}
            for b in terrain.BIOMES
        ],
        "water": round(float(water.mean()), 3),
        "cells": [int(v) for v in np.asarray(f["biome"]).reshape(-1)],
        "provinces": [],
        "realms": [dict(r) for r in realms],
        "roads": [],
        "rivers": [],
    }
    if res is not None and res["count"]:
        for i, p in enumerate(res["provinces"]):
            doc["provinces"].append({
                "index": i,
                "name": p["name"],
                "owner": int(owner[i]) if i < len(owner) else -1,
                "ground": p["ground"],
                "capital": [int(p["capital"][0]), int(p["capital"][1])],
                "coastal": bool(p["coastal"]),
                "area": int(p["area"]),
                "neighbours": [int(n) for n in p["neighbours"]],
                "cells": _flat(p["cells"], grid.cols),
            })
    for road in routes:
        doc["roads"].append({
            "from": int(road["from"]),
            "to": int(road["to"]),
            "cost": float(road["cost"]),
            "cells": _flat(road["path"], grid.cols),
        })
    for course in courses:
        doc["rivers"].append({
            "cells": _flat(course["path"], grid.cols),
            "drainage": round(float(course["flux"]), 1),
        })
    return doc
