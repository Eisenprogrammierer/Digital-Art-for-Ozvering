"""Hex maps: a lattice, the terrain on it, its coastlines, and its picture.

``grid``
    The hexagonal lattice in axial coordinates: centres, neighbours, hex
    distance, the corner normal form, and the per-pixel raster lookup that
    resolves a world point to a cell and a position inside it.
``terrain``
    The geography: a warped, quantile-cut elevation field, a moisture field, and
    the biome table that gives every cell a name, a colour and a movement cost.
``coast``
    The boundary of a set of hexes as closed loops of corners -- the coastline,
    the lakes, and the edge of the map, all from the same walk.
``paint``
    One per-pixel raster pass for the ground and its relief, then strokes for the
    coast, the lattice and the frame.

A map is a function of ``(seed, cols, rows)`` and nothing else: ``size`` decides
only how many pixels it is drawn in, so the same seed is the same continent at
any scale.

Typical use::

    from gen.maps import map_image
    image = map_image(seed=7, cols=26, rows=17, size=14.0)
    image.save("out/map.png")
"""
from . import coast, grid, paint, terrain

__all__ = ["coast", "grid", "paint", "terrain", "HexGrid", "map_image"]

HexGrid = grid.HexGrid
map_image = paint.map_image
