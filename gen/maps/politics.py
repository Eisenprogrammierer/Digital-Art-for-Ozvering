"""Who owns the provinces, and the colour that makes a realm *worn*.

A political map asks two questions the geography cannot answer, and both of them
are about what a province *is* rather than where it is.

**A realm is a block.**  Handing each province to whichever faction has the
nearest capital is one line of arithmetic and produces realms in two pieces: a
faction on both sides of a bay, with no way to march from one half to the other.
So ownership is grown over the *province* graph -- multi-source breadth-first
search, the same argument :mod:`gen.maps.region` makes for provinces within a
landmass -- and a realm comes out connected by construction.  The seeds are
chosen by graph distance for the same reason: two capitals that are near as the
crow flies and far apart around a mountain range are not two places to start from.
Landmasses that share no province are separate graphs, so the realms are shared
out between the landmasses -- one to each, before size decides -- and no landmass
is left holding provinces that nobody can reach.

**A realm is a colour.**  This is where a map meets the heraldry, and the meeting
is one function.  A faction's arms decide a *tincture* and a *livery*: the
tincture is the colour the shield is, and the livery is that colour moved by the
faction's own integer steps.  The map takes three things from it and answers each
of them itself:

* the **hue** is the tincture's, because the tincture is what the arms are, and
  Azure on a map has to still be the blue the player calls Azure;
* the **chroma** is the tincture's *class*, normalised to a wash -- so a metal
  lends no hue and wears a grey, which is the truth about Or and Argent and the
  reason the two metals are told apart by lightness, exactly as they are anywhere
  else in this project;
* the **lightness** is where a wash sits on a sheet of parchment, moved by the
  livery's own step -- and this is the interesting one.  A livery's step is one
  whole number of OKLab, because that is what a step has to be to show on a
  shield.  At map size the wash goes on at half opacity over terrain that is busy
  with relief and texture, so the same step arrives at a third of its size and
  vanishes.  The map therefore *spends* the step: stretched, and passed through a
  saturating map, so a livery one step apart is a colour apart here and no livery
  can push a province out of the band its terrain is read in.

Two realms can still pick the same tincture and draw the same livery -- there are
seven tinctures and there will be more factions than that.  So a *roster* is
drawn against the colour rather than hoping: names come off a stream, and a name
is taken only if its realm is :data:`WASH_STEP` away from every realm already on
the map.  A roster is therefore a palette -- the map's own answer to the same
question :func:`gen.maps.paint.province_tints` answers with a golden angle, but
answering it from the factions' own coats instead of from their count.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from core import palette
from gen.heraldry import crest, tinctures
from gen.maps import region

__all__ = [
    "WASH_FILL",
    "WASH_STEP",
    "factions",
    "owners",
    "tint_table",
    "wash",
    "washes",
    "waters",
]

#: Where a province's wash sits on the sheet, and how far a livery's step can take
#: it.  The band is a *span* rather than a pair of limits because the map spends
#: the step through a saturating map: two liveries a step apart stay apart however
#: far they are from the middle, and no livery can push a province out of the band
#: its terrain is still read in -- a province dark enough to swallow a mountain
#: range is a province that has stopped being a map.
WASH_L = 0.74
WASH_SPAN = 0.12

#: How much of the livery's step the map spends.  A step is one whole OKLab on a
#: shield; here it goes on at half opacity over terrain, so it is spent a few times
#: over to survive the trip.
WASH_GAIN = 2.6

#: The chroma of a wash, which is also its ceiling: a province is a wash over a
#: map, and the map is the thing the player is reading.
WASH_C = 0.13

#: Below this chroma a tincture has no hue to lend a map -- a metal, or the black.
#: Those wear a grey and are told apart by lightness, which is what a metal *is*.
NEUTRAL = 0.03

#: How far apart two realms on one map have to be, in OKLab, at the colour they
#: actually wear.  The project calls 0.04 "a different tint", and this is that
#: number *divided by the opacity the wash goes on at* -- because a province is not
#: a flat colour, it is the wash over its terrain, and a step measured before that
#: mix would be a step that is not there when the map is looked at.  The two
#: constants therefore have to move together, which is why one is written in terms
#: of the other and the renderer is handed the same number.
WASH_FILL = 0.5
WASH_STEP = 0.04 / WASH_FILL

#: The sea is a colour on this map too, and it has already been spent.  A realm
#: that wears it is a realm with no coastline, so a *chromatic* wash is kept this
#: far (in turns of hue) from every colour the water is drawn in -- and a wash with
#: no chroma at all keeps out of the way by itself, which is why a metal is never
#: refused for being grey.
SEA_GUARD = 0.05
SEA_CHROMA = 0.04

#: The name stream the realms come off.  Salted apart from the provinces' so the
#: two are not the same stream, and checked against the provinces' names as well,
#: because two different streams still run out of ideas eventually.
_NAME_SALT = 917

#: A roster is drawn, not chosen, so the draw needs a bound: one that cannot be
#: reached by a stream whose candidates are mostly acceptable, and small enough
#: that a pathological map gives up with fewer realms instead of hanging.
_DRAWS = 400


# ------------------------------------------------------------------ colour
def _tincture(name: str) -> tinctures.Tincture:
    """The tincture a realm wears: the first tincture of its coat's field.

    A plain field has one, a division has several, and the first is the one the
    shield is mostly made of.  A *fur* is a pattern of two tinctures rather than a
    colour, so it is resolved to the tincture it is made of -- and because that is
    how the heraldry shades a fur in the first place, the realm still gets its own
    lightness out of it.
    """
    field = crest.arms(name)["field"]["tinctures"]
    base = tinctures.get(str(field[0]))
    if base.fur:
        return tinctures.get(base.fur[0])
    return base


def waters(grid, res: dict, seas: dict, owners) -> np.ndarray:
    """Who holds the water: the realm with the longest coast on each sea province.

    Land is *grown*; water is *claimed*.  A sea province is not somewhere a realm
    marches into, so it is not grown into either: a zone goes to whichever realm
    holds the most of its coast, and a zone no realm's coast touches stays the high
    seas and stays ``-1``.  That is a coarser rule than the median line a chart
    draws down the middle of a bay, and it is the honest one at this size -- a
    median line needs the bay's own geometry, and what a game asks of a sea zone is
    who to ask permission from.

    The land's division is not consulted beyond its *borders*: this reads the land's
    labels and its owners and writes a separate array over the sea's, so adding
    territorial waters cannot move a capital or a border on land by one hex.
    """
    owners = [int(o) for o in owners]
    land = np.asarray(res["labels"]).reshape(-1)
    out = np.full(int(seas["count"]), -1, dtype=np.int64)
    for i, p in enumerate(seas["provinces"]):
        coast = {}
        for j in p["cells"]:
            r, c = divmod(int(j), grid.cols)
            for rr, cc in grid.nbr[r, c]:
                rr, cc = int(rr), int(cc)
                if rr < 0:
                    continue
                other = int(land[rr * grid.cols + cc])
                if other >= 0:
                    o = owners[other]
                    if o >= 0:
                        coast[o] = coast.get(o, 0) + 1
        if coast:
            # Ties go to the lower realm index, so the answer is a function of the
            # map rather than of the order the coast happened to be walked in.
            out[i] = min(coast, key=lambda o: (-coast[o], o))
    return out


def wash(name: str) -> np.ndarray:
    """The colour the realm called ``name`` wears on a map, as sRGB.

    One function, three channels, and a reason for each -- see the module
    docstring.  It is deliberately the only place a coat is turned into a colour,
    so the sidecar, the preview sheet and a game all read the same answer.
    """
    base = _tincture(name)
    worn = tinctures.shaded(base.key, tinctures.shade_table(crest.shade(name)))
    l0, c0, h0 = palette.srgb_to_oklch(base.rgb)
    l1, c1, h1 = palette.srgb_to_oklch(worn.rgb)
    if c0 < NEUTRAL:                       # a metal, or the black: grey, told apart by lightness
        chroma, hue = 0.0, h0
    else:
        # The livery's chroma step, scaled into the wash.  Two bounds, and both are
        # the map's rather than the livery's: the ratio is a *proportion* of the
        # tincture's chroma, so on a near-neutral tincture (the black, say) it is
        # enormous for a step that is a hair of colour; and a wash is drawn at a
        # chroma the terrain is read through, so a livery that *raised* its
        # tincture's chroma wears the wash's own and keeps its hue and lightness.
        chroma, hue = WASH_C * min(float(c1) / float(c0), 1.0), h1
    spent = WASH_GAIN * (float(l1) - float(l0))
    lightness = WASH_L + WASH_SPAN * float(np.tanh(spent / WASH_SPAN))
    return palette.color(lightness, chroma, hue)


def washes(names) -> np.ndarray:
    """The wash of every realm in ``names``, as one ``(n, 3)`` table."""
    out = np.empty((len(names), 3), dtype=np.float64)
    for i, name in enumerate(names):
        out[i] = wash(name)
    return out


def tint_table(res: dict, owners, table) -> np.ndarray:
    """Per *province* colours: what each province wears, given who owns it.

    Water belongs to nobody -- ``owners`` says ``-1`` there -- and is never
    painted, but the table still has to be indexable by every label, so the last
    colour stands in for the sea.
    """
    table = np.asarray(table, dtype=np.float64)
    if not len(owners):
        return np.empty((0, 3), dtype=np.float64)
    idx = np.clip(np.asarray(owners, dtype=np.int64), 0, len(table) - 1)
    return table[idx]


# ------------------------------------------------------------------ roster
def _sea_hues() -> tuple:
    """The hue band the water is drawn in, from the colours it is drawn with.

    Taken from the painter rather than typed in here, so a change to the sea ramp
    or to the river colour moves the guard with them instead of leaving a rule
    behind that the picture no longer obeys.
    """
    from gen.maps import paint                    # local: the painter is the authority
    hues = [float(palette.srgb_to_oklch(c)[2]) for c in list(paint.SEA_RAMP) + [paint.RIVER]]
    return min(hues), max(hues)


def _reaches_the_sea(colour) -> bool:
    """Would this wash be read as water?  Only a hue with a wash's chroma can be.

    A grey has no hue to confuse with the sea's, which is why a metal is never
    refused for wearing the colour the sea is nearest to.
    """
    _lightness, chroma, hue = palette.srgb_to_oklch(colour)
    if float(chroma) < SEA_CHROMA:
        return False
    lo, hi = _sea_hues()
    middle = 0.5 * (lo + hi)
    d = abs(float(hue) - middle)
    return min(d, 1.0 - d) < 0.5 * (hi - lo) + SEA_GUARD


def factions(res: dict, count: int, seed: int = 0, step: float = WASH_STEP,
             draws: int = _DRAWS) -> list:
    """``count`` realms for one map, drawn until they are colours apart.

    The names come off the same stream the provinces' names do, salted apart from
    them, and a candidate is taken only if its realm's wash is at least ``step``
    from every realm already on the map *and* is not a colour the sea is drawn in.
    That is the whole reason a roster is *drawn* rather than simply sliced: seven
    tinctures and more factions than that means two realms will pick the same one,
    and a map where two neighbours are the same colour is a map that has stopped
    being readable.

    If the stream cannot fill the roster within ``draws`` -- which a stream of
    acceptable candidates does not do -- the roster is short rather than wrong, and
    a caller that needs the count can measure it.
    """
    count = max(int(count), 0)
    if count == 0 or not res["count"]:
        return []
    taken: list = []
    colours: list = []
    seen = {str(p["name"]) for p in res["provinces"]}
    index = 0
    used = 0
    while len(taken) < count and used < int(draws):
        index += 1
        name = region.name(seed, index, "", salt=_NAME_SALT)
        if name in seen:
            continue
        used += 1
        colour = wash(name)
        if _reaches_the_sea(colour):
            continue
        if all(palette.delta_e(colour, other) >= step for other in colours):
            seen.add(name)
            taken.append(name)
            colours.append(colour)
    return taken


# ------------------------------------------------------------------ owners
def _neighbours(res: dict) -> list:
    return [[int(n) for n in p["neighbours"]] for p in res["provinces"]]


def _components(n: int, nbr: list) -> list:
    """The province graph's connected parts, each as a sorted index list."""
    seen = [False] * n
    out = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        comp = [start]
        q = deque([start])
        while q:
            j = q.popleft()
            for k in nbr[j]:
                if not seen[k]:
                    seen[k] = True
                    comp.append(k)
                    q.append(k)
        out.append(sorted(comp))
    return out


def _share(count: int, sizes) -> list:
    """Split ``count`` realms between landmasses, one each before size decides.

    A landmass is a graph of its own, so a realm cannot be grown onto it from
    outside: an island that is given no realm has provinces nobody holds, which is
    not a political map but a gap in one.  So the landmasses are served first --
    largest first, while there are realms to serve -- and only then are the
    remaining realms shared out in proportion to size.
    """
    sizes = [int(s) for s in sizes]
    n = len(sizes)
    out = [0] * n
    if count <= 0 or n == 0:
        return out
    for i in sorted(range(n), key=lambda i: (-sizes[i], i))[:count]:
        if sizes[i] > 0:
            out[i] = 1
    left = count - sum(out)
    if left <= 0:
        return out
    total = float(sum(sizes))
    raw = [left * s / total for s in sizes]
    add = [min(int(np.floor(x)), sizes[i] - out[i]) for i, x in enumerate(raw)]
    for i, a in enumerate(add):
        out[i] += max(a, 0)
    left -= sum(max(a, 0) for a in add)
    for i in sorted(range(n), key=lambda i: (-(raw[i] - np.floor(raw[i])), i)):
        if left <= 0:
            break
        if out[i] < sizes[i]:
            out[i] += 1
            left -= 1
    return out


def _graph_dist(nbr: list, start: int) -> list:
    """Graph distance from ``start`` to every province, ``-1`` unreachable."""
    dist = [-1] * len(nbr)
    dist[start] = 0
    q = deque([start])
    while q:
        j = q.popleft()
        for k in nbr[j]:
            if dist[k] < 0:
                dist[k] = dist[j] + 1
                q.append(k)
    return dist


def _spread(comp: list, nbr: list, count: int) -> list:
    """``count`` provinces of one component, as far apart as the graph allows.

    Farthest point, which is what :func:`gen.maps.region.sites` does with Mitchell's
    best candidate for the provinces themselves: start anywhere, then keep the
    province furthest from everything chosen.  Ties go to the lowest index, so the
    answer does not depend on the order a set happens to iterate in.
    """
    chosen = [comp[0]]
    if count <= 1:
        return chosen
    dist = _graph_dist(nbr, chosen[0])
    while len(chosen) < count:
        far = max(comp, key=lambda p: (dist[p], -p))
        if dist[far] <= 0:
            break                       # the component is smaller than the roster
        chosen.append(far)
        other = _graph_dist(nbr, far)
        dist = [min(a, b) if b >= 0 else a for a, b in zip(dist, other)]
    return chosen


def _grow(nbr: list, seeds: list, base: int) -> list:
    """Multi-source breadth-first search: every province joins a nearest seed.

    The queue is filled in seed order, so a tie goes to the lower realm, and a
    province is claimed once and never moved -- which is what makes each realm a
    *connected* set: a province joins the realm of a neighbour that already has it.
    """
    owner = [-1] * len(nbr)
    dist = [-1] * len(nbr)
    q = deque()
    for i, s in enumerate(seeds):
        owner[s] = base + i
        dist[s] = 0
        q.append(s)
    while q:
        j = q.popleft()
        for k in nbr[j]:
            if owner[k] < 0:
                owner[k] = owner[j]
                dist[k] = dist[j] + 1
                q.append(k)
    return owner


def owners(res: dict, realms, seed: int = 0) -> list:
    """Which realm holds each province, as a list of indices into a roster.

    ``realms`` is how many realms to grow, or the roster itself for a caller that
    already has one -- a game's own factions are not drawn here.  ``-1`` means no
    realm holds a cell: water always, and a province of a landmass there were too
    few realms to reach.  The count is shared between the landmasses first, so a
    realm is not asked to hold provinces on an island it cannot sail to, and each
    landmass then grows its share from seeds spread across its own graph.
    """
    n = int(res["count"])
    count = int(realms) if isinstance(realms, (int, np.integer)) else len(realms)
    count = max(count, 0)
    if n == 0 or count == 0:
        return [-1] * n
    nbr = _neighbours(res)
    comps = _components(n, nbr)
    share = _share(min(count, n), [len(c) for c in comps])
    out = [-1] * n
    label = 0
    for comp, k in zip(comps, share):
        if k <= 0:
            continue
        seeds = _spread(comp, nbr, k)
        grown = _grow(nbr, seeds, label)
        for p in comp:
            out[p] = grown[p]
        label += len(seeds)
    return out
