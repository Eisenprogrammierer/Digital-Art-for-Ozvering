"""Textures that tile: Truchet, Wang, reaction-diffusion and boards.

``tile``
    The algebra of a wrap: what "seamless" means as a number, how a painter finds
    the tile a pixel is in without a loop over tiles, and the project's one light.
``truchet``
    Arc and maze tiles whose seamlessness is a property of the geometry -- any
    arrangement works, and there is nothing to search or to check.
``wang``
    Tiles that carry their edges, so the seamlessness *is* the problem: the choice
    at every cell is constrained by the choice before it, and the generator solves
    it with a scanline search -- or sidesteps it entirely, by reading the tiles off a
    quantised height field, where two neighbours agreeing is reading one number
    twice.
``reaction``
    Gray-Scott reaction-diffusion, run on a torus so that it wraps by construction,
    for organic patterns nothing hand-drawn gets close to.
``boards``
    A floor: the one material here with a *direction*, where the tile has to carry
    the length of a board and the way its grain runs across the seam -- including for
    the board that runs off one edge and comes back on the other.
``atlas``
    Ship the set: the tiles packed into one sheet with a table of rectangles, and with
    margins that are each tile wrapped onto itself rather than a gap -- because a tile
    has no transparent border to trim and nothing useful to blend into.

Typical use::

    from gen.textures import truchet
    truchet.texture(seed=3, size=256, grains=4).save("out/truchet.png")
"""
from . import atlas, boards, reaction, tile, truchet, wang

__all__ = ["atlas", "boards", "reaction", "tile", "truchet", "wang"]
