"""Hair: the biggest style lever, built from one primitive.

Every piece of hair in this package is the same object -- a tapered lock along a
spine (:func:`gen.portraits.shape.lock`) -- rooted on an ellipse that stands just
outside the skull.  The hairstyle is then only *where the roots are, which way
they hang, and how far*:

* the **cap** hugs the skull and fills the space above the hairline;
* the **fringe** is a row of locks rooted across the top of the forehead, with
  the defining detail that they are *all different lengths* -- an even fringe
  reads as a wig and is the single most common tell of a generated face;
* the **sidelocks** hang from the temples and frame the face;
* the **back mass** is a fan of wide locks that shows around the silhouette and
  hangs as far as the style's length;
* the **shine** is a row of small fat locks sitting just inside the silhouette
  along the crown, which is what stops hair looking like painted-on plastic;
* the **crown sheets** are broad paint-only locks laid over the cap, so the top
  of the head is a pile of hair rather than one flat plate.

Two construction rules matter more than the numbers.  First, every root is pulled
*inside* the skull's hull by about half the lock's own width, so the rounded root
end is hidden under the cap -- a root left on the hull pokes out of the crown and
the silhouette becomes a row of bumps.  Second, the roots sit on the hull's
ellipse, so the whole hairstyle follows the head when it turns, with nothing
authored per view.

Painting is three passes over one evaluation (:class:`gen.portraits.brush.Surface`):
fill the union, quantise its Lambert bands into the stored field, then ink the
contour as arithmetic on the same array.  A fringe of eight clumps costs one
polygon pass per clump, not thirty.
"""
from __future__ import annotations

import numpy as np

from core import noise, palette

from . import brush, cel, head
from . import shape as S
from .shape import Shape, union_all

__all__ = ["HAIR_STYLES", "Hair", "build", "paint_back", "paint_front", "mass",
           "front_mass", "describe"]

_CLIMATE_HAIR = {
    "arctic": dict(vol=-0.003, gloss=0.08, blen=0.98),
    "temperate": dict(),
    "arid": dict(vol=0.008, gloss=-0.10, blen=0.95),
    "tropical": dict(vol=0.012, gloss=0.04, blen=1.02, sdrop=1.02),
    "highland": dict(vol=0.004, gloss=0.02),
}
_CULTURE_HAIR = {
    "north": dict(vol=-0.002, gloss=0.12, blen=0.98),
    "river": dict(gloss=0.02),
    "steppe": dict(vol=0.008, gloss=0.02, sw=1.04),
    "coast": dict(vol=0.002, gloss=0.08),
    "desert": dict(vol=0.006, gloss=-0.08, blen=0.96, sdrop=0.95),
    "highland": dict(vol=0.004, gloss=0.03, sw=1.03),
}

HAIR_STYLES = {
    "short":  dict(vol=0.045, bangs=7, bw=0.245, blen=0.55, sides=0, sw=0.100, sdrop=0.180, near=False,
                   back=8, kw=0.300, kdrop=0.180, crest=0, spike=0, cap_r=3.00, flare=1.02, temple_lift=0.045, peak=0.008, sideburn=0,
                   tuft=0.0),
    "bob":    dict(vol=0.078, bangs=7, bw=0.260, blen=0.75, sides=3, sw=0.200, sdrop=0.560,
                   back=9, kw=0.345, kdrop=0.740, crest=0, spike=0, cap_r=1.70, flare=1.18),
    "long":   dict(vol=0.090, bangs=7, bw=0.275, blen=0.60, sides=3, sw=0.210, sdrop=1.080,
                   back=10, kw=0.360, kdrop=1.100, crest=0, spike=0, cap_r=1.25, flare=1.36),
    "hime":   dict(vol=0.080, bangs=9, bw=0.240, blen=0.82, sides=3, sw=0.210, sdrop=1.000,
                   back=10, kw=0.350, kdrop=1.000, crest=0, spike=0, cap_r=1.30, flare=1.30,
                   straight=True),
    "twin":   dict(vol=0.076, bangs=7, bw=0.245, blen=0.65, sides=0, sw=0.270, sdrop=0.900, near=False,
                   back=9, kw=0.320, kdrop=0.240, crest=0, spike=0, cap_r=1.60, flare=1.10,
                   twin=2),
    "pony":   dict(vol=0.068, bangs=7, bw=0.245, blen=0.55, sides=0, sw=0.260, sdrop=0.450, near=False,
                   back=8, kw=0.310, kdrop=0.200, crest=1, spike=0, cap_r=2.20, flare=1.04),
    "spiky":  dict(vol=0.054, bangs=8, bw=0.215, blen=0.30, sides=0, sw=0.220, sdrop=0.320, near=False,
                   back=8, kw=0.300, kdrop=0.170, crest=0, spike=6, cap_r=2.00, flare=1.04,
                   sl=0.100),
    "wild":   dict(vol=0.080, bangs=8, bw=0.215, blen=0.45, sides=0, sw=0.260, sdrop=0.520, near=False,
                   back=9, kw=0.315, kdrop=0.220, crest=0, spike=8, cap_r=1.45, flare=1.08,
                   sl=0.130),
}


def _params(name: str, st, genes: dict) -> dict:
    """Style parameters, stretched by the genome's hair axes.

    The legacy ``hair_len`` and ``bang_n`` still do what they always did.  The
    phenotype axes add volume, fringe length, a parting direction and sidelock
    weight; every one is clamped here rather than in the caller, because these
    numbers decide whether roots stay on the skull and the fringe stays off the
    eyes.
    """
    p = dict(HAIR_STYLES[name])
    length = float(genes.get("hair_len", 0.0))
    scale = float(np.clip(1.0 + 0.34 * length, 0.70, 1.60))
    for key in ("sdrop", "kdrop"):
        p[key] = float(p[key]) * scale
    density = float(genes.get("bang_n", 0.0))
    p["bangs"] = int(np.clip(round(float(p["bangs"]) + 2.0 * density), 4, 12))
    p["back"] = int(np.clip(round(float(p["back"]) + 2.0 * density), 4, 12))
    p["vol"] = float(np.clip(float(p["vol"]) + 0.022 * float(genes.get("hair_volume", 0.0)),
                             0.030, 0.158))
    p["blen"] = float(np.clip(float(p["blen"]) * (1.0 + 0.20 * float(genes.get("fringe_length", 0.0))),
                              0.18, 1.18))
    part = float(genes.get("fringe_part", 0.0))

    p["part"] = 0.0 if p.get("straight") else float(np.clip(part * 0.78, -0.46, 0.46))
    side = float(genes.get("side_lock", 0.0))
    p["sw"] = float(np.clip(float(p["sw"]) * (1.0 + 0.14 * side), 0.12, 0.42))
    p["sdrop"] = float(np.clip(float(p["sdrop"]) * (1.0 + 0.16 * side), 0.16, 1.70))
    p["gloss"] = float(genes.get("hair_gloss", 0.0))
    climate = str(genes.get("climate", "temperate"))
    culture = str(genes.get("culture", "river"))
    for table in (_CLIMATE_HAIR, _CULTURE_HAIR):
        adj = table.get(climate if table is _CLIMATE_HAIR else culture, {})
        p["vol"] = float(p["vol"]) + float(adj.get("vol", 0.0))
        p["gloss"] = float(p["gloss"]) + float(adj.get("gloss", 0.0))
        p["blen"] = float(p["blen"]) * float(adj.get("blen", 1.0))
        p["sw"] = float(p["sw"]) * float(adj.get("sw", 1.0))
        p["sdrop"] = float(p["sdrop"]) * float(adj.get("sdrop", 1.0))
    p["vol"] = float(np.clip(p["vol"], 0.030, 0.158))
    p["blen"] = float(np.clip(p["blen"], 0.06, 1.18))

    age = str(genes.get("age", "adult"))
    if age in ("adult", "middle"):
        factor = 0.70 if age == "adult" else 0.68
        if name == "short" and not p.get("straight"):

            factor = 0.78 if age == "adult" else 0.70
        p["blen"] = float(np.clip(p["blen"] * factor, 0.06, 1.18))
    p["sw"] = float(np.clip(p["sw"], 0.12, 0.42))
    p["sdrop"] = float(np.clip(p["sdrop"], 0.16, 1.70))
    return p


def _hull(st, vol: float):
    return st.head_w + vol, 0.505 + vol


def _root(st, vol: float, theta: float, turn: float, inset: float = 0.0) -> np.ndarray:
    """A root on the hair's own hull: an ellipse around the head, so it turns too.

    The hull is a shell *around* the skull, so the root rides the surface with the
    hair's volume as its offset -- which is what makes a lock's root follow the head
    rather than sit where the cap used to be.
    """
    rx, ry = _hull(st, vol - inset)
    x = rx * np.cos(theta)
    y = ry * np.sin(theta) + 0.015
    x, y = head.turn_point(x, y, turn, st, vol=max(vol - inset, 0.0))
    return np.array([float(x), float(y)], dtype=np.float64)


def _roughen_hull(hull: Shape, seed: int, amp: float = 0.018,
                  freq: float = 11.0) -> Shape:
    """Expand a silhouette by smooth low-frequency noise, not with stuck-on locks.

    A cap's edge is the zero contour of one field, so it is a perfect curve.  The
    first fix was a row of tiny locks; at 384 px those read as teeth, and wider
    ones as battlements.  A smooth noise term moves the contour a pixel or two at
    a 15-25 px wavelength, so the edge becomes irregular the way a clipped head
    is while remaining one continuous surface and one fill.

    The displacement is deliberately one-sided.  A side lock's painted strand is
    built from the same spine as the lock, and a noise trough somewhere on the
    contour would put that ink outside its own fill.  Expanding only keeps every
    strand inside, and outward bumps are enough to break the curve.  The noise is
    applied to the *hull* before the face is cut, so the hairline and eyes are
    untouched.
    """
    b = hull.bbox
    pad = amp * 2.5

    def f(x, y):
        n = np.clip(noise.fbm(x, y, seed=seed, freq=freq, octaves=3),
                    0.0, 1.0)
        return hull.f(x, y) - amp * (0.35 + 0.65 * n)

    return Shape(f, (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))


def _silhouette_tufts(st, p: dict, rng, turn: float) -> Shape:
    """A few short locks that stand *outside* the cap's smooth outline.

    A cap grown from a smooth ellipse is a helmet however many locks are painted
    on top of it, because its silhouette is one continuous curve.  Cutting holes
    in the edge to roughen it was worse: a hole shows the backdrop and reads as a
    moth-eaten wig.  The truthful version is the opposite -- hair that sticks out
    in a few unequal wisps, so the curve gains a silhouette rather than losing
    pieces.

    They are unioned into the cap, which means they are filled, shaded and inked
    by the same passes as the rest of the cap and cannot appear as a separate
    sticker.  All of them are above the ear line, so none can cross the face.
    """
    vol = float(p["vol"])
    out = []

    scale = float(p.get("tuft", 0.55))
    if scale <= 0.0:
        return None

    for th0 in (0.11, 0.19, 0.27, 0.73, 0.81, 0.89):
        th = np.pi * th0 + float(rng.uniform(-0.026, 0.026))
        root = _root(st, vol, th, turn, inset=0.055)
        edge = _root(st, vol, th, turn, inset=0.0)
        outward = edge - root
        outward /= max(float(np.linalg.norm(outward)), 1e-9)

        tx = 1.0 if edge[0] >= 0.0 else -1.0
        tangent = np.array([tx * abs(outward[1]), -abs(outward[0])], dtype=np.float64)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        length = float(rng.uniform(0.020, 0.042)) * scale
        sweep = float(rng.uniform(0.55, 1.15)) * length
        tip = edge + outward * length + tangent * sweep
        mid = root + (tip - root) * 0.56 - outward * 0.004
        w0 = float(rng.uniform(0.032, 0.058)) * (0.60 + 0.40 * scale)
        out.append(S.lock([root, mid, tip], w0=w0,
                          w1=w0 * float(rng.uniform(0.28, 0.48)),
                          bulge=w0 * 0.20, per_segment=6, resample=26))
    return union_all(out, pad=0.012)


def _fringe_gaps(st, p: dict, turn: float) -> Shape | None:
    """An optional tapered parting cut through the fringe and its filler.

    The fringe is built from overlapping clumps, which is right for volume and
    wrong for a hairline at avatar size: the clumps merge into a single dark sheet
    with a shaped bottom.  A real parting is not a painted line on top of that
    sheet; it is a place where the hair separates and the forehead shows through
    at the hem.  The slit is narrow at the top and wider at the brow, and stops
    above the eye.  It is off unless a style explicitly asks for it: the part gene
    and the clump lengths already separate a combed fringe, and a subtracted slit
    was too easy to read as a lightning bolt at 384 px.

    Blunt hime cuts deliberately have no gap; keeping one straight edge is the
    definition of that cut.
    """
    if p.get("straight"):
        return None

    if not p.get("cut_part"):
        return None
    bw = float(st.head_w)
    part = float(p.get("part") or 0.0)
    top = float(st.brow_y) + 0.190
    bottom = float(st.brow_y) + 0.032
    out = []
    x0 = part * bw * 0.30 - 0.004
    slant = 0.009
    half_top = 0.0030
    half_bot = 0.0068

    spine = [(x0, top), (x0 + slant * 0.45, (top + bottom) * 0.5),
             (x0 + slant, bottom)]
    out.append(S.lock(spine, w0=half_top * 2.0, w1=half_bot * 2.0,
                      bulge=0.0, per_segment=5, resample=18))
    return union_all(out, pad=0.010)


def _back_locks(st, p: dict, rng, turn: float) -> tuple:
    """A fan of wide locks behind the head: the silhouette and the length.

    The roots stay above the hairline (so the cap hides them) and the tips are
    cut rather than pointed, which is what gives long hair a level hem instead of
    a row of needles.

    The fan is returned in two halves, because the locks rooted near the temples
    and the locks rooted over the crown do not hang in the same place.  A lock
    from the crown hangs down the *back* of the head and its tip never leaves the
    silhouette; a lock from the temple hangs beside the jaw, in front of the neck
    and then in front of the shoulder.  Painting the second kind with the first
    leaves long hair sliced off level with the collar, which is the one hem a
    hairstyle never has.
    """
    n = int(p["back"])
    kw, drop, vol = float(p["kw"]), float(p["kdrop"]), float(p["vol"])
    far, near, far_s, near_s = [], [], [], []
    for i in range(n):
        u = (i + 0.5) / n
        th = np.pi * (0.19 + 0.62 * u)
        root = _root(st, vol, th, turn, inset=kw * 0.58)
        side = 1.0 if root[0] >= 0.0 else -1.0
        length = drop * float(rng.uniform(0.82, 1.16))
        flare = float(p.get("flare", 1.25)) * float(rng.uniform(0.97, 1.04))
        tip = np.array([root[0] * flare + side * 0.075 * float(rng.uniform(0.35, 1.55)),
                        root[1] - length])
        mid = np.array([root[0] * (1.0 + (flare - 1.0) * 0.46) + side * 0.018,
                        root[1] - length * 0.46])
        w0 = kw * float(rng.uniform(0.90, 1.08))

        w1 = kw * float(rng.uniform(0.16, 0.32))
        lock = S.lock([root, mid, tip], w0=w0, w1=w1, bulge=kw * 0.10,
                      per_segment=7, resample=34)
        strand = _strand([root, mid, tip], w1, mid[1] + 0.05,
                         per_segment=7, resample=34)
        if abs(th - np.pi * 0.5) > np.pi * 0.16 and p.get("near", True):
            near.append(lock)
            near_s.append(strand)
        else:
            far.append(lock)
            far_s.append(strand)
    return far, near, far_s, near_s


def _strand(spine, tip_w: float, top: float, per_segment: int, resample: int) -> tuple:
    """A line drawn *inside* a clump, and the height it has to fade out at.

    A clump's boundary is inked, and the clump in front covers most of it, so what
    survives of the ink is a few millimetres of edge near the hem: the fringe reads
    as a smooth sheet with a sawtooth bottom.  What is missing is the strand -- the
    line an artist draws *down* a lock, not around it.

    It is built from the clump's *own* spine on purpose.  A line laid beside the
    spine from its own endpoints can bulge out past the hem and draw on the skin,
    and that is the one thing ink must never do; sharing the spine and asking for
    less width makes the strand a subset of its clump, by construction.

    The sampling has to be handed in and has to match the clump's, because that is
    the only thing making the subset exact: the two polygons share their spine
    points, so each of the strand's is strictly nearer the spine than the clump's,
    and the resampler then cuts the same corners on both.  Let the two disagree and
    a vertex of the thin polygon can land a hair outside the fat one -- which is
    why the test for this measures the escape instead of trusting the argument.

    It is thin at the top and wider at the hem, because the edge of a lock gathers
    ink where the lock ends, and it fades out at ``top`` -- the middle of the clump,
    which is roughly where the clump behind has crept far enough forward to be on
    top of it, so the line disappears into the mass instead of crossing it.
    """
    w = max(float(tip_w), 1e-3)

    return (S.lock(spine, w0=w * 0.48, w1=w * 0.80, bulge=0.0,
                   per_segment=per_segment, resample=resample), float(top))


def _inner_lock(spine, w0: float, w1: float, bulge: float, per_segment: int,
                resample: int, shrink: float = 0.16) -> Shape:
    """A narrow lock along the same spine, strictly inside its parent clump.

    The builder is the only place that knows the clump's width profile.  Deriving
    the strand from that profile -- instead of eroding the finished shape by a
    fixed distance -- keeps it a line of *visible* width at the hem, and keeps it
    inside the clump all the way to the tip (where the parent has width zero).
    """
    return S.lock(spine, w0=max(float(w0) * shrink, 1e-6),
                  w1=max(float(w1) * shrink, 0.0),
                  bulge=float(bulge) * shrink,
                  per_segment=per_segment, resample=resample)


def _bangs(st, p: dict, rng, turn: float) -> tuple:
    """The fringe: many clumps of *different* lengths, with cut ends.

    The length is not a style parameter, it is a measurement.  A fringe that hangs
    past the brow hides the face's expression, and one that stops short leaves a
    bare forehead the style's proportions never allowed for, so each clump is cut
    to land above the brow and the *style's* own landmarks decide where that is.
    The style only biases the result (``blen``), says how many clumps there are,
    and how far they sway.

    The roots come back beside the clumps because the ink needs them: a clump's
    strand line has to fade out as it approaches *its own* root, and only the
    builder knows where that is.  The strands come back with them.
    """

    n = int(p["bangs"]) + 5
    bw, vol = float(p["bw"]), float(p["vol"])
    fill = float(p.get("blen", 1.0))
    straight = bool(p.get("straight", False))
    brow = float(getattr(st, "brow_y", 0.10))

    part_gene = p.get("part")
    part = float(rng.uniform(-0.45, 0.45)) if part_gene is None else float(part_gene)
    if straight:
        part = 0.0
    part_u = float(np.clip(0.5 + 0.5 * part, 0.16, 0.84))

    out, roots, strands = [], [], []
    for i in range(n):
        u = (i + 0.5) / n
        uu = u
        th = np.pi * (0.17 + 0.66 * uu)
        root = _root(st, vol, th, turn, inset=bw * 0.66)
        side = 1.0 if root[0] >= 0.0 else -1.0

        if straight:
            edge, jitter, tip_w = 1.0, float(rng.uniform(0.94, 1.06)), bw * 0.52
        else:
            edge = 1.0 - 0.18 * abs(uu - 0.5) * 2.0

            jitter, tip_w = float(rng.uniform(0.35, 1.55)), None
        sweep = float(np.clip((uu - part_u) * 2.20, -0.85, 0.85))

        if straight:
            short_y = brow + 0.032 + 0.050 * abs(sweep) * 2.0
            long_y = brow + 0.004
        else:

            short_y = brow + 0.170 - 0.095 * fill - 0.045 * sweep
            long_y = brow + 0.110 - 0.085 * fill
        short = max(root[1] - short_y, 0.05)
        long = max(root[1] - long_y, short + 0.018)

        longness = sweep if part_u < 0.5 else -sweep
        local_fill = fill if straight else float(np.clip(fill + 0.60 * longness, 0.08, 0.98))
        if not straight:

            tier = 1.0 if i % 2 == 0 else 0.22
            local_fill = float(np.clip(local_fill * 0.55 + tier * 0.45, 0.08, 0.98))
        length = (short + (long - short) * local_fill) * jitter * edge

        ceiling = long_y + float(rng.uniform(
            0.0, 0.024 if straight else 0.060))
        length = min(length, max(root[1] - ceiling, 0.04))
        sway = (0.022 + 0.040 * abs(u - 0.5)) * float(rng.uniform(-1.0, 1.0))
        tip = np.array([root[0] * 1.12 + sway + sweep * 0.380 + side * 0.018,
                        root[1] - length])
        mid = np.array([root[0] * 1.05 + sway * 0.40 + sweep * 0.050,
                        root[1] - length * 0.50])
        spine = [root, mid, tip]
        cw = bw * float(rng.uniform(0.15, 0.25))
        if tip_w is None:

            tip_w = cw * float(rng.uniform(0.10, 0.22))
        bulge = bw * 0.045
        out.append(S.lock(spine, w0=cw, w1=tip_w, bulge=bulge,
                          per_segment=7, resample=32))
        roots.append(float(root[1]))

        strands.append((_inner_lock(spine, cw, tip_w, bulge, 7, 32, shrink=0.44),
                        float(root[1]) - 0.025))
    return out, roots, strands


def _sidelocks(st, p: dict, rng, turn: float) -> tuple:
    """The curtains: wide locks that hang in front of the temple and the cheek.

    These are the locks that cover the *sides* of the head, and they have to be
    wide: the cap stops at the hairline, so anything narrower leaves a band of
    bare skin between the fringe and the jaw and the whole hairstyle reads as a
    headband.  Their roots are above the hairline (the cap and the crown repaint
    hide the root), and their tips are cut, not pointed, because a curtain ends in
    a hem.
    """
    n = int(p["sides"])
    sw, drop, vol = float(p["sw"]), float(p["sdrop"]), float(p["vol"])
    straight = bool(p.get("straight", False))
    sk = head.skull(st)
    out, roots, strands = [], [], []
    for sign in (+1, -1):
        if turn != 0.0 and bool(sk.receding(sign, turn)):
            continue
        for k in range(n):
            th = np.pi * ((0.19 + 0.055 * k) if sign > 0 else (1.0 - 0.19 - 0.055 * k))
            inset = sw * 0.55 + 0.030 * k
            root = _root(st, vol, th, turn, inset=inset)
            length = drop * float(rng.uniform(0.78, 1.22)) * (1.0 - 0.10 * k)
            bow = 0.014 if straight else 0.048

            out_x = max(float(st.head_w) * 0.92 + vol * 0.25 - sw * 0.24 + 0.025 * k
                        + 0.030 + sign * float(rng.uniform(-0.026, 0.026)),
                        float(st.head_w) * 1.02)
            tip = np.array([sign * (out_x - float(rng.uniform(0.022, 0.044))),
                            root[1] - length])
            mid = np.array([sign * (out_x + bow * float(rng.uniform(0.74, 1.28))),
                            root[1] - length * float(rng.uniform(0.38, 0.50))])

            base_w = sw * 0.52 * (1.0 - 0.28 * k) * float(rng.uniform(0.84, 1.10))

            spine = [root, mid, tip]
            cw0, cw1 = base_w * 0.26, base_w * float(rng.uniform(0.18, 0.32))
            bulge = base_w * 0.13
            out.append(S.lock(spine, w0=cw0, w1=cw1, bulge=bulge,
                              per_segment=7, resample=34))
            roots.append(float(root[1]))
            strands.append((_inner_lock(spine, cw0, cw1, bulge, 7, 34),
                            mid[1] + 0.03))

        if not n and p.get("sideburn"):
            th = np.pi * (0.135 if sign > 0 else 0.865)
            root = _root(st, vol, th, turn, inset=0.020)
            tip_y = float(st.eye_y) + 0.030
            tip_x = sign * float(st.head_w) * 0.905
            spine = [root, (tip_x + sign * 0.004, (root[1] + tip_y) * 0.5), (tip_x, tip_y)]
            w0 = float(st.head_w) * 0.038
            out.append(S.lock(spine, w0=w0, w1=w0 * 0.34,
                              bulge=w0 * 0.10, per_segment=6, resample=26))
            roots.append(float(root[1]))
            strands.append((_inner_lock(spine, w0, w0 * 0.34, w0 * 0.10, 6, 26),
                            root[1] - 0.05))

        if not n:
            for j in range(2):
                th = np.pi * ((0.168 + 0.022 * j) if sign > 0
                              else (0.832 - 0.022 * j))
                root = _root(st, vol, th, turn, inset=0.020 + 0.016 * j)
                length = (max(drop * 0.45, 0.085)
                          * float(rng.uniform(0.82, 1.18)) * (1.0 - 0.24 * j))
                out_x = (float(st.head_w) * (1.040 + 0.018 * j) + vol * 0.24
                         + float(rng.uniform(-0.012, 0.022)))
                tip = np.array([sign * (out_x + float(rng.uniform(-0.024, 0.012))),
                                root[1] - length])
                mid = np.array([sign * (out_x + float(rng.uniform(0.014, 0.036))),
                                root[1] - length * float(rng.uniform(0.36, 0.56))])
                ww = float(st.head_w) * float(rng.uniform(0.018, 0.030)) * (1.0 - 0.20 * j)
                spine = [root, mid, tip]
                out.append(S.lock(spine, w0=ww, w1=ww * float(rng.uniform(0.26, 0.44)),
                                  bulge=ww * 0.18, per_segment=6, resample=26))
                roots.append(float(root[1]))
                strands.append((_inner_lock(spine, ww, ww * 0.30, ww * 0.18,
                                            6, 26), mid[1] + 0.02))

        if n:
            th = np.pi * (0.165 if sign > 0 else 0.835)
            root = _root(st, vol, th, turn, inset=sw * 0.70)
            length = drop * float(rng.uniform(0.55, 0.86))
            out_x = max(float(st.head_w) * 0.94 + vol * 0.18 - sw * 0.16,
                        float(st.head_w) * 0.98)
            tip = np.array([sign * (out_x - 0.040), root[1] - length])
            mid = np.array([sign * (out_x + 0.026), root[1] - length * 0.46])
            front_w = sw * float(rng.uniform(0.36, 0.46))
            spine = [root, mid, tip]
            out.append(S.lock(spine, w0=front_w, w1=front_w * 0.56,
                              bulge=front_w * 0.20, per_segment=7, resample=32))
            roots.append(float(root[1]))
            strands.append((_inner_lock(spine, front_w, front_w * 0.56,
                                        front_w * 0.20, 7, 32),
                            mid[1] + 0.03))

        if n:
            for j in range(3):
                th = np.pi * ((0.174 + 0.012 * j) if sign > 0
                              else (0.826 - 0.012 * j))
                root = _root(st, vol, th, turn, inset=sw * (0.62 - 0.020 * j))
                length = drop * (0.38 + 0.22 * j) * float(rng.uniform(0.78, 1.24))
                out_x = max(float(st.head_w) * (1.025 + 0.010 * j)
                            + vol * 0.28 - sw * (0.105 - 0.010 * j),
                            float(st.head_w) * (1.060 + 0.010 * j))
                out_x += float(rng.uniform(-0.020, 0.026)) + 0.006 * j
                tip = np.array([sign * (out_x - float(rng.uniform(0.022, 0.046))
                                       - 0.004 * j),
                                root[1] - length])
                mid = np.array([sign * (out_x + float(rng.uniform(0.026, 0.056))
                                       - 0.004 * j),
                                root[1] - length * float(rng.uniform(0.40, 0.54))])
                ow = sw * float(rng.uniform(0.24, 0.34)) * (1.0 - 0.06 * j)
                spine = [root, mid, tip]
                out.append(S.lock(spine, w0=ow, w1=ow * 0.20,
                                  bulge=ow * 0.22, per_segment=7, resample=34))
                roots.append(float(root[1]))
                strands.append((_inner_lock(spine, ow, ow * 0.16, ow * 0.22,
                                            7, 34), mid[1] + 0.03))
            for j in range(2):
                th = np.pi * ((0.158 + 0.020 * j) if sign > 0
                              else (0.842 - 0.020 * j))
                root = _root(st, vol, th, turn, inset=sw * (0.48 + 0.08 * j))
                length = drop * float(rng.uniform(0.34, 0.66)) * (1.0 - 0.07 * j)
                out_x = (float(st.head_w) * (1.045 + 0.024 * j) + vol * 0.18
                         + float(rng.uniform(-0.022, 0.034)))
                tip_x = out_x + float(rng.uniform(-0.034, 0.020))
                tip = np.array([sign * tip_x, root[1] - length])
                mid = np.array([sign * (out_x + float(rng.uniform(0.020, 0.052))),
                                root[1] - length * float(rng.uniform(0.38, 0.56))])
                ww = sw * float(rng.uniform(0.085, 0.145))
                spine = [root, mid, tip]
                out.append(S.lock(spine, w0=ww, w1=ww * float(rng.uniform(0.22, 0.42)),
                                  bulge=ww * 0.16, per_segment=6, resample=28))
                roots.append(float(root[1]))
                strands.append((_inner_lock(spine, ww, ww * 0.30, ww * 0.16,
                                            6, 28), mid[1] + 0.025))
    return out, roots, strands


def _crest(st, p: dict, rng, turn: float) -> tuple:
    """Ponytails, twin tails and spikes: bundles that live outside the skull.

    The tails come back with a strand each, because a tail is a bundle of hair and
    its own contour says nothing about that.  The spikes do not: a spike is one
    short lock, and a line drawn down it is ink on top of ink.  So the two lists
    are deliberately not the same length -- it is the *tails* that carry strands,
    and the test for them is containment in the row, not pairing.
    """
    vol = float(p["vol"])
    out, strands = [], []
    for _ in range(int(p.get("crest", 0))):
        side = 1.0 if rng.random() < 0.5 else -1.0

        root = _root(st, vol, np.pi * 0.5, turn, inset=0.01)

        root[1] -= 0.180
        rise = float(rng.uniform(-0.015, 0.005))
        fall = float(rng.uniform(0.78, 1.06))

        spine = [root,
                 np.array([root[0] + side * 0.19, root[1] + rise]),
                 np.array([root[0] + side * 0.27, root[1] - fall * 0.42]),
                 np.array([root[0] + side * 0.23, root[1] - fall])]
        out.append(S.lock(spine, w0=0.082, w1=0.024, bulge=0.030,
                          per_segment=8, resample=44))

        strands.append(_strand(spine, 0.030, float(root[1]) - 0.04,
                               per_segment=8, resample=44))
    for _ in range(int(p.get("twin", 0)) // 2):
        for side in (+1, -1):
            w = float(p["sw"]) * 0.40

            root = _root(st, vol, np.pi * (0.20 if side > 0 else 0.80), turn, inset=0.02)
            drop = float(p["sdrop"]) * 0.92
            sway = float(rng.uniform(0.88, 1.16))
            spine = [root,
                     root + np.array([side * 0.170 * sway, -drop * 0.26]),
                     root + np.array([side * 0.195 * sway, -drop * 0.64]),
                     root + np.array([side * 0.145 * sway, -drop])]

            out.append(S.lock(spine, w0=w * 0.90, w1=w * 0.30, bulge=w * 0.15,
                              per_segment=8, resample=44))

            strands.append(_strand(spine, w * 0.24, float(root[1]) - 0.05,
                                   per_segment=8, resample=44))
    for _ in range(int(p.get("spike", 0))):

        d = float(rng.uniform(0.0, 1.0)) ** 0.60 * 0.22 + 0.20
        th = np.pi * (0.5 + (d if rng.random() < 0.5 else -d))

        root = _root(st, vol, th, turn, inset=0.01)
        rx, ry = _hull(st, vol)
        radial = np.array([np.cos(th) / max(rx, 1e-6), np.sin(th) / max(ry, 1e-6)])
        radial = radial / max(float(np.linalg.norm(radial)), 1e-9)
        ln = float(p.get("sl", 0.16)) * float(rng.uniform(0.70, 0.95))

        ln = min(ln, max(0.030, (0.664 - float(root[1])) / max(float(radial[1]), 0.18)))
        lean = np.array([radial[1], -radial[0]]) * float(rng.uniform(-0.28, 0.28))
        out.append(S.lock([root, root + radial * ln * 0.55 + lean,
                           root + radial * ln + lean * 1.7],
                          w0=0.028 + 0.26 * ln, w1=0.010, bulge=0.024,
                          per_segment=6, resample=30))
    return out, strands


def _shine(st, p: dict, rng, turn: float) -> list:
    """The gloss: shallow arcs just inside the silhouette along the crown.

    A shine is a *reflection*, so it is thin and it follows the form -- a fat
    blob of highlight reads as a bald patch, not as hair.  Six narrow arcs read
    as a row of lit strands; three broad ones read as a painted stripe.
    """
    vol = float(p["vol"])
    out = []
    for i in range(5):
        u = (i + 0.5) / 5.0
        th = np.pi * (0.22 + 0.56 * u)
        centre = _root(st, vol, th, turn, inset=0.065)
        above = _root(st, vol, th + 0.230, turn, inset=0.065)
        below = _root(st, vol, th - 0.230, turn, inset=0.065)
        out.append(S.lock([below, centre, above], w0=0.010, w1=0.005,
                          bulge=float(rng.uniform(0.018, 0.030)), per_segment=7,
                          resample=30))
    return out


def _crown_locks(st, p: dict, rng, turn: float) -> list:
    """Broad sheets laid along the crown, over the cap.

    The cap is one filled surface, and one filled surface is a helmet: a head of
    hair is a pile of sheets, and each catches the light a little differently.
    These locks are *paint only* -- they are never part of ``mass`` or
    ``front_mass``, so they cannot change the silhouette or reach the face, and
    each is clipped to the cap before it is painted.  They are a second read of
    the volume the cap already has, not a second hairstyle.

    They radiate from the parting, which is why the parting is a gene: a crown
    combed both ways from a centre part and a crown combed to one side are not
    the same head.
    """
    vol = float(p["vol"])
    part_x = float(p.get("part") or 0.0) * float(st.head_w) * 0.30
    _, ry = _hull(st, vol)
    root = np.array(head.turn_point(part_x, ry - 0.050, turn, st, vol=vol))
    out = []
    for i in range(5):
        side = 1.0 if i % 2 == 0 else -1.0
        u = (i // 2 + 0.5) / 2.0
        th = np.pi * (0.5 + side * (0.12 + 0.32 * u))
        tip = _root(st, vol, th, turn, inset=0.030 + 0.022 * float(rng.random()))
        mid = root + (tip - root) * 0.52 + np.array([side * 0.012, 0.026 * u])
        w0 = float(st.head_w) * float(rng.uniform(0.105, 0.145))
        w1 = w0 * float(rng.uniform(0.34, 0.54))
        out.append(S.lock([root, mid, tip], w0=w0, w1=w1,
                          bulge=w0 * 0.30, per_segment=6, resample=30))
    return out


def _cap_flow(st, p: dict, rng, turn: float) -> tuple:
    """Broad locks combed from the parting to the hairline, for the cap.

    The cap is one filled surface, and one filled surface is a helmet no matter
    what colour it is.  These locks are paint-only and clipped to the cap: they
    never alter the silhouette and cannot reach the face, but they put the
    direction back into the crown.  Each side gets seven sheets of unequal width
    -- four combed down to the hairline and three over the upper temple -- so the
    top of the head reads as hair combed away from a part rather than as a
    pressed-felt dome.  The lines return separately because they are painted
    after every sheet, exactly like the strands inside a fringe clump.
    """
    hairline = head.hairline_y(st)
    part = float(p.get("part") or 0.0)
    part_x = part * float(st.head_w) * 0.26
    locks, lines = [], []
    for side in (+1.0, -1.0):
        for k in range(4):
            u = (k + 0.5) / 4.0

            tip_x = side * float(st.head_w) * (0.16 + 1.02 * u)
            tip_y = hairline + 0.010 - 0.040 * u
            root_x = part_x + side * float(st.head_w) * (0.015 + 0.115 * u)
            root_y = hairline + 0.195 + 0.020 * (1.0 - u)
            mid = np.array([(root_x + tip_x) * 0.5 + side * 0.012,
                            (root_y + tip_y) * 0.5 + 0.010], dtype=np.float64)
            spine = [np.array([root_x, root_y]), mid,
                     np.array([tip_x, tip_y])]
            w0 = float(st.head_w) * (0.050 + 0.024 * (1.0 - u)) \
                * float(rng.uniform(0.88, 1.12))
            w1 = w0 * (0.44 + 0.18 * u)
            bulge = w0 * 0.10
            locks.append((S.lock(spine, w0=w0, w1=w1, bulge=bulge,
                                 per_segment=6, resample=28), side))
            lines.append((_inner_lock(spine, w0, w1, bulge, 6, 28, shrink=0.110),
                          side))

        for k in range(3):
            u = (k + 0.5) / 3.0

            tip_x = side * float(st.head_w) * (0.82 + 0.30 * u)
            tip_y = hairline + 0.170 - 0.090 * u
            root_x = part_x + side * float(st.head_w) * (0.030 + 0.050 * u)
            root_y = hairline + 0.215
            mid = np.array([(root_x + tip_x) * 0.5 + side * 0.018,
                            (root_y + tip_y) * 0.5 + 0.028], dtype=np.float64)
            spine = [np.array([root_x, root_y]), mid,
                     np.array([tip_x, tip_y])]
            w0 = float(st.head_w) * (0.072 + 0.020 * (1.0 - u)) \
                * float(rng.uniform(0.90, 1.12))
            w1 = w0 * 0.52
            bulge = w0 * 0.16
            locks.append((S.lock(spine, w0=w0, w1=w1, bulge=bulge,
                                 per_segment=6, resample=28), side))
            lines.append((_inner_lock(spine, w0, w1, bulge, 6, 28, shrink=0.120),
                          side))
    return locks, lines


class Hair:
    """Everything a hairstyle is made of, ready to paint in two passes.

    ``back_locks`` are the locks that hang behind the head and ``near_locks`` the
    ones rooted at the temples, which hang beside the jaw and over the shoulder
    and so have to be painted *after* the body.  See :func:`_back_locks`.

    ``bang_roots`` and ``side_roots`` are the root heights of the fringe's clumps,
    aligned with ``bangs`` and ``sides``: the ink on a clump has to fade out at
    its own root, and nothing else in the frame knows where that is.

    ``front_strands`` are the lines drawn inside the fringe's clumps, each paired
    with the height it fades out at.  They are a second pass over the same spines
    rather than part of the clumps because the clumps ink each other out: a strand
    has to be painted after the last clump has had its say.  ``crest_strands`` is
    the same idea for the tails, and is not paired with ``crest`` -- the spikes in
    that list are shorter than a strand is thick.
    """

    __slots__ = ("name", "p", "cap", "fringe_base", "back_locks",
                 "near_locks", "bangs", "bang_roots", "sides", "side_roots",
                 "front_strands", "crest", "crest_strands", "shine", "crown",
                 "back_strands", "near_strands", "back_mass", "front_mass", "mass",
                 "flow", "flow_lines")

    def __init__(self, name, p, cap, back_locks, near_locks, bangs, bang_roots, sides,
                 side_roots, front_strands, crest, crest_strands, shine,
                 back_strands=(), near_strands=(), fringe_base=None, crown=(),
                 flow=(), flow_lines=()):
        self.name = name
        self.p = p
        self.cap = cap
        self.fringe_base = fringe_base
        self.back_locks = back_locks
        self.near_locks = near_locks
        self.back_strands = list(back_strands)
        self.near_strands = list(near_strands)
        self.bangs = bangs
        self.bang_roots = bang_roots
        self.sides = sides
        self.side_roots = side_roots
        self.front_strands = front_strands
        self.crest = crest
        self.crest_strands = crest_strands
        self.shine = shine
        self.crown = list(crown)
        self.flow = list(flow)
        self.flow_lines = list(flow_lines)
        self.back_mass = union_all([cap] + back_locks + crest)
        front = list(bangs) + list(sides)
        if fringe_base is not None:
            front.append(fringe_base)
        self.front_mass = union_all(front)
        self.mass = union_all([self.back_mass, self.front_mass] + near_locks)


def _eye_clearance(st, turn: float, expr=None, asym=None):
    """The actual eye openings, grown by a hair's breadth, for front hair to avoid.

    The projected hairline is enough for the cap almost everywhere, but at a
    three-quarter turn a sidelock and the cap independently graze the far eye.
    This is the geometric version of the composition invariant the tests already
    assert: a lock may hide the temple, never the opening.

    The first version cleared a fat ellipse around each eye.  It passed the
    overlap test but lifted the hairline in a round notch well above the lid.  A
    cut that follows the opening itself is both smaller and less legible as a
    cut: it reads as the hair tucked behind the eye instead of a bite out of the
    fringe.
    """
    from . import expression as _expression
    from . import eyes as _eyes
    if expr is None:
        expr = _expression.neutral()
    geometry = _eyes.shapes(st, expr)
    out = []
    for side in (+1, -1):
        cx, ws = _eyes.placement(st, side, turn, asym)
        if asym is None:
            dy, sy, rot = 0.0, 1.0, side * float(st.eye_tilt)
        else:
            e = asym.eye[side]
            dy, sy = e["dy"], e["sy"]
            rot = side * float(st.eye_tilt) + e["rot"]
        opening = geometry["opening"].place(
            cx=cx, cy=float(st.eye_y) + dy, sx=side * ws, sy=sy, rot=rot,
        )
        out.append(opening.offset(0.012))
    return out


def _hairline_tufts(st, p: dict, rng, turn: float) -> Shape | None:
    """A row of short cut ends along the forehead hairline.

    The cap is a smooth shell.  On a short cut that shell is also the only thing
    at the hairline, so its arc is read as a bowl rim.  Real cropped hair ends in
    a slightly ragged line: a few locks sit a little lower than their neighbours,
    and that is enough to break the arc without adding side lobes (the earlier
    outward tufts made ears when they were applied to a short cut).
    """
    hairline = head.hairline_y(st)
    vol = float(p["vol"])
    out = []
    count = 13
    for i in range(count):
        u = (i + 0.5) / count
        x = (u - 0.5) * 2.0 * float(st.head_w) * 0.82
        jitter = float(rng.uniform(-0.014, 0.014))
        root = head.turn_point(x, hairline + 0.014, turn, st, vol=vol)
        tip_x = x + jitter + float(rng.uniform(-0.012, 0.012))
        length = float(rng.uniform(0.024, 0.054))
        tip = head.turn_point(tip_x, hairline - length, turn, st, vol=vol * 0.55)
        mid = ((root[0] + tip[0]) * 0.5,
               (root[1] + tip[1]) * 0.5 + 0.005)
        width = float(rng.uniform(0.0080, 0.0130))
        out.append(S.lock([root, mid, tip], w0=width, w1=0.0010,
                          bulge=width * 0.32, per_segment=4, resample=18))
    return union_all(out)


def _cap(st, p: dict, turn: float, hairline: float, skull: Shape | None = None,
         clearance: Shape | None = None, rng=None) -> Shape:
    """The skull cap: the head's own outline grown by ``vol``, minus the face.

    Reusing the head silhouette is what makes a cap fit *this* skull -- a wide
    head gets a wide cap, and there is no second set of proportions to keep in
    sync.  Every lock in the style is rooted *above* the hairline so this cap
    covers its root; a root left below it shows its own straight cut and the
    crown becomes a row of teeth.

    The face is cut out with a large circle rather than a straight line.  A
    straight cut gives the hair a level bottom edge, which reads as a headband --
    the sides of the skull have to be covered down to the temples or the hairstyle
    floats.  An arc that is nearly flat at the forehead and dips at the temples is
    also what a hairline actually looks like.  ``cap_r`` sets how sharply it dips:
    a big radius keeps the ears bare (short cuts), a small one buries them (long
    ones), and it is the single number that separates a bob from a crew cut at the
    silhouette.
    """
    hull = (skull if skull is not None else head.silhouette(st, turn)).offset(float(p["vol"]))
    seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else 90125
    hull = _roughen_hull(hull, seed, amp=0.024, freq=13.5)
    r = float(st.head_w) * float(p.get("cap_r", 1.8))

    sk = head.skull(st)
    xs = np.linspace(-r, r, 73)
    lift = float(p.get("temple_lift", 0.0))
    peak = float(p.get("peak", 0.0))
    if lift or peak:

        u = np.clip(xs / max(float(st.head_w), 1e-6), -1.25, 1.25)
        ys = hairline + lift * u * u - peak * (1.0 - u * u)
    else:
        ys = hairline - r + np.sqrt(np.maximum(r * r - xs * xs, 0.0))
    cut = [sk.project(float(x), float(y), turn) for x, y in zip(xs, ys)]
    xmin = min(float(x) for x, _ in cut) - 0.05
    xmax = max(float(x) for x, _ in cut) + 0.05
    cut.extend([(xmax, -1.8), (xmin, -1.8)])
    face = S.poly(np.asarray(cut, dtype=np.float64))

    clear = clearance if clearance is not None else union_all(
        _eye_clearance(st, turn))
    return hull - face - clear


def _trim_cap_for_locks(cap: Shape, st, sides) -> Shape:
    """Let the painted curtains, not the smooth hull, own the lower silhouette.

    The cap is an offset of the skull.  Even when three broad curtains are painted
    over it, its own contour remains outside them, so the eye sees a smooth helmet
    with hair drawn inside.  The upper cap is still the right shape for the crown
    and the hairline; below the temple, the cap is cut away and the lock union is
    the edge.  A small overlap is kept above the cut so the two pieces cannot
    separate when the head turns.
    """
    if not sides:
        return cap

    trim_y = float(st.eye_y) + 0.045
    below = S.box(float(st.head_w) * 3.0, 1.0, 0.0, 0.0, trim_y - 1.0)
    lower = cap & below

    cover = union_all(sides)
    if cover is None:
        return cap - below
    backing = lower & cover.offset(0.004)

    hw = float(st.head_w)
    temple = None
    for side in (+1, -1):
        wedge = S.box(hw * 0.22, 0.115, 0.0,
                      side * hw * 0.70, trim_y - 0.095)
        temple = wedge if temple is None else (temple | wedge)
    if temple is not None:
        backing = backing | (lower & temple)
    return (cap - below) | backing


def build(st, genes: dict, rng=None, turn: float = 0.0, skull: Shape | None = None,
          eye_st=None, eye_expr=None, asym=None) -> Hair:
    """Construct a hairstyle for a genome.  Deterministic given ``rng``.

    ``eye_st`` is the expression-warped style the eyes are drawn with, and
    ``eye_expr`` the expression row.  Hair keeps the unwarped skull and hairline,
    but its clearance has to follow the eyes wherever the expression put them.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    name = genes.get("hair_style", "short")
    if name not in HAIR_STYLES:
        raise KeyError(f"unknown hair style {name!r}")
    p = _params(name, st, genes)
    hairline = head.hairline_y(st) - 0.02
    back, near, back_strands, near_strands = _back_locks(st, p, rng, turn)
    bangs, bang_roots, bang_strands = _bangs(st, p, rng, turn)
    sides, side_roots, side_strands = _sidelocks(st, p, rng, turn)
    back_seed = int(rng.integers(0, 2**31 - 1))
    back = [_roughen_hull(lock, back_seed + 53 * i, amp=0.016, freq=12.5)
            for i, lock in enumerate(back)]
    near = [_roughen_hull(lock, back_seed + 101 + 59 * i, amp=0.016, freq=12.5)
            for i, lock in enumerate(near)]

    clear = union_all(_eye_clearance(eye_st or st, turn, eye_expr, asym))
    bangs = [clump - clear for clump in bangs]
    side_seed = int(rng.integers(0, 2**31 - 1))
    sides = [_roughen_hull(lock, side_seed + 37 * i, amp=0.020, freq=14.5) - clear
             for i, lock in enumerate(sides)]

    bang_strands = [(strand - clear, top) for strand, top in bang_strands]
    side_strands = [(strand - clear, top) for strand, top in side_strands]

    gaps = _fringe_gaps(st, p, turn)
    if gaps is not None:
        bangs = [clump - gaps for clump in bangs]
        bang_strands = [(strand - gaps, top) for strand, top in bang_strands]
    crest, crest_strands = _crest(st, p, rng, turn)

    fringe_base = None
    if bangs and p.get("base", 1):
        if p.get("straight"):

            tips = sorted(float(clump.bbox[1]) for clump in bangs)
            base_bottom = min(tips[0] + 0.015, float(st.brow_y) + 0.075)
            base_top = hairline + 0.060
            w_top = float(st.head_w) * 1.06
            w_bottom = float(st.head_w) * 0.92
            band = S.poly([(-w_top, base_top), (w_top, base_top),
                           (w_bottom, base_bottom + 0.004),
                           (0.0, base_bottom - 0.006),
                           (-w_bottom, base_bottom + 0.004)],
                          resample=28)
            fringe_base = band & head.silhouette(st, turn)
        else:

            base = (union_all(bangs).offset(0.014)
                    & head.silhouette(st, turn)) - clear

            if bangs:

                shelf = min(float(clump.bbox[1]) for clump in bangs)
                base = base & S.box(float(st.head_w) * 2.4, 2.0, 0.0, 0.0,
                                    shelf - 0.010 + 1.0)
            fringe_base = base
        if gaps is not None:
            fringe_base = fringe_base - gaps

    shine = _shine(st, p, rng, turn)
    crown = _crown_locks(st, p, rng, turn)
    flow, flow_lines = _cap_flow(st, p, rng, turn)
    tufts = _silhouette_tufts(st, p, rng, turn)
    cap = _cap(st, p, turn, hairline, skull, clearance=clear, rng=rng)
    if tufts is not None:
        cap = cap | tufts
    if name == "short" and float(p.get("tuft", 0.0)) > 0.0:
        fringe = _hairline_tufts(st, p, rng, turn)
        if fringe is not None:
            cap = (cap | fringe) - clear

    cap = _trim_cap_for_locks(cap, st, sides)
    return Hair(
        name=name,
        p=p,
        cap=cap,
        back_locks=back,
        near_locks=near,
        bangs=bangs,
        bang_roots=bang_roots,
        sides=sides,
        side_roots=side_roots,
        front_strands=bang_strands + side_strands,
        crest=crest,
        crest_strands=crest_strands,
        shine=shine,
        back_strands=back_strands,
        near_strands=near_strands,
        fringe_base=fringe_base,
        crown=crown,
        flow=flow,
        flow_lines=flow_lines,
    )


def mass(h: Hair) -> Shape:
    return h.back_mass


def front_mass(h: Hair) -> Shape:
    return h.front_mass


def _hair_dome(st, h: Hair, turn: float = 0.0):
    """A volume over the skull, so the terminator lands where a head's would."""
    return cel.dome(rx=st.head_w * 1.18, ry=0.70, height=0.44, cy=0.03)


def _paint_locks(canvas, st, hues, h: Hair, locks, strands=()) -> None:
    """Fill, shade and ink a row of locks, each one over the outline of the last.

    This is what turns a mass of hair into strands.  A union of ten locks painted
    in one pass and inked once along its contour is a smooth sheet with a single
    terminator -- which is the one thing hair is not.  Painted in order, each
    lock fills over the outline of the one before it and shades with its own small
    bevel, so what survives is a mass of strands: the strand lines are the
    boundaries where a lock stops and the next begins.  The terminator still runs
    across the whole head because every layer shares one volume.
    """
    lw = float(st.ink_w) * float(st.line) * 0.85
    dome = _hair_dome(st, h)
    for i, lock in enumerate(locks):
        s = brush.Surface(canvas, lock, pad=2.2 * lw)
        s.paint(canvas, hues.hair)

        if i % 3 == 1:
            s.paint(canvas, hues.hair_shadow, alpha=0.135)
        elif i % 3 == 2:
            s.paint(canvas, hues.hair_fine, alpha=0.085)
        s.bands(canvas, [hues.hair_shadow, hues.hair_deep], (0.44, 0.15),
                volume=dome, bevel=0.030, soft=0.8)


def _paint_strands(canvas, hues, strands, lw: float, clip=None, alpha: float = 0.52,
                   colour=None) -> None:
    """The lines drawn *inside* locks, in one pass after every fill is down.

    Shared by the fringe and the crest, which differ only in where the fade
    starts: a clump of fringe fades halfway down, because by then the clump behind
    has crept over the top of it, and a tail fades at its root, because nothing is
    painted over a tail at all.
    """
    if not strands:
        return
    _, Yg = canvas.grid()
    for strand, top in strands:
        s = brush.Surface(canvas, strand, pad=2.6 * lw)
        if s.empty:
            continue
        rows = slice(s.rows, s.rows + s.d.shape[0])
        cols = slice(s.cols, s.cols + s.d.shape[1])
        amt = np.clip((top - Yg[rows, cols]) / 0.10, 0.0, 1.0)
        if clip is not None:
            amt = amt * clip[rows, cols]
        s.paint(canvas, hues.line_hair if colour is None else colour,
                mask=amt, alpha=float(alpha), soft=1.0)


def paint_back(canvas, st, hues, h: Hair, turn: float = 0.0, clip=None):
    """The hair that hangs *behind* the head: the back locks and the crest.

    Nothing here is faded or clipped to hide a lock's root, because the *cap* is
    painted later and covers every root: every lock in a style is rooted above the
    hairline on purpose, and the cap is exactly the hair between the crown and the
    hairline.  That is also why the cap is not painted here -- it sits on the
    skull, in front of the skin, not behind the head with these locks.

    The tails' strands go on last, after every lock in the row has been filled and
    inked, for the same reason the fringe's do: the lock in front has to have had
    its say before the line is drawn on top of it.
    """
    _paint_locks(canvas, st, hues, h, h.back_locks + h.crest,
                 strands=h.back_strands + h.crest_strands)


def paint_near(canvas, st, hues, h: Hair, turn: float = 0.0, clip=None):
    """The temple-rooted locks: over the body, under the face.

    They hang beside the jaw and then over the shoulder, so they cannot travel
    with the locks behind the head -- painted there, the bust covers their tips
    and long hair ends in a level hem at the collar.  Nor can they travel with
    the fringe: their roots are inside the skull, and painted after the face they
    would draw a strand of hair across a cheek.  So they go between the two, and
    the head -- painted afterwards -- hides exactly the part of them that belongs
    behind it.
    """
    _paint_locks(canvas, st, hues, h, h.near_locks, strands=h.near_strands)


def paint_front(canvas, st, hues, h: Hair, turn: float = 0.0, clip=None):
    """Everything on the near side of the skin: the cap, the fringe, the shine.

    The cap is painted first, over the skin, because a cap painted behind the head
    leaves bare scalp between the hairline and the fringe's hem.  It is a darker
    underlayer.  A soft base then fills the gaps behind the clumps without drawing
    a second hem; the bangs and sidelocks are painted over it as clean, uninked
    shapes.  Their roots are construction detail, so the cap is repainted above
    the hairline to hide them and the fringe gets one smooth vertical gradient
    instead of per-clump patches.

    Internal lock boundaries are deliberately not inked.  A row of clump outlines
    is what made the hair read as straw; the silhouette line is drawn once, in
    :func:`gen.portraits.detail.paint_hair_detail`, and the hem carries the shape.
    """
    lw = float(st.ink_w) * float(st.line) * 0.78
    dome = _hair_dome(st, h, turn)
    cap = brush.Surface(canvas, h.cap, pad=2.6 * lw)

    cap.paint(canvas, palette.mix(hues.hair, hues.hair_shadow, 0.14))

    if h.fringe_base is not None:
        base = brush.Surface(canvas, h.fringe_base, pad=2.6 * lw)
        base.paint(canvas, palette.mix(hues.hair, hues.hair_shadow, 0.30))
        base.bands(canvas, [hues.hair_shadow, hues.hair_deep], (0.50, 0.16),
                   volume=dome, bevel=0.0, soft=0.5)

    _, Yg = canvas.grid()
    for i, (clump, root_y) in enumerate(zip(h.bangs + h.sides,
                                            h.bang_roots + h.side_roots)):
        s = brush.Surface(canvas, clump, pad=2.6 * lw)
        s.paint(canvas, hues.hair)
        if i < len(h.bangs):

            if i % 3 == 1:
                s.paint(canvas, hues.hair_fine, alpha=0.230)
            elif i % 3 == 2:
                s.paint(canvas, hues.hair_shadow, alpha=0.205)

            s.stroke(canvas, lw * 0.48,
                     palette.mix(hues.hair_shadow, hues.hair, 0.24),
                     inset=lw * 0.12, alpha=0.06, soft=1.2)

            fbx0, fby0, fbx1, fby1 = clump.bbox
            fw = max(float(fbx1) - float(fbx0), 1e-4)
            fcx = 0.5 * (float(fbx0) + float(fbx1))
            fh = min(0.010, max(0.004, (float(fby1) - float(fby0)) * 0.045))
            brush.paint(canvas,
                        clump & S.box(fw * 0.40, fh * 0.52, fh * 0.25,
                                      fcx, float(fby0) + fh * 0.56),
                        hues.hair_deep, clip=clump, alpha=0.18, soft=0.45)
            brush.paint(canvas,
                        clump & S.box(fw * 0.34, fh * 0.18, fh * 0.10,
                                      fcx, float(fby0) + fh * 1.45),
                        hues.hair_fine, clip=clump, alpha=0.075, soft=0.60)
        else:

            k = i - len(h.bangs)
            cx = 0.5 * (float(clump.bbox[0]) + float(clump.bbox[2]))
            sgn = 1.0 if cx >= 0.0 else -1.0
            if k % 3 == 1:
                s.paint(canvas, hues.hair_fine, alpha=0.135)
            elif k % 3 == 2:
                s.paint(canvas, hues.hair_shadow, alpha=0.120)

            width = max(float(clump.bbox[2]) - float(clump.bbox[0]), 1e-4)
            inner = clump - clump.place(cx=sgn * min(0.040, width * 0.28))
            outer = clump - clump.place(cx=-sgn * min(0.046, width * 0.32))
            brush.multiply(canvas, inner, 0.61, clip=clump, alpha=0.46)

            brush.paint(canvas, outer, hues.hair_fine, clip=clump,
                        alpha=0.085, soft=1.2)

            if width > 0.035:
                s.stroke(canvas, lw * 0.58,
                         palette.mix(hues.hair_shadow, hues.hair, 0.22),
                         inset=lw * 0.10, alpha=0.31, soft=0.9)

                height = max(float(clump.bbox[3]) - float(clump.bbox[1]), 1e-4)
                bottom = float(clump.bbox[1])
                hem = min(0.014, max(0.006, height * 0.055))
                brush.paint(canvas,
                            clump & S.box(width * 0.46, hem * 0.55,
                                          hem * 0.30, cx, bottom + hem * 0.52),
                            hues.hair_deep, clip=clump, alpha=0.26, soft=0.42)
                brush.paint(canvas,
                            clump & S.box(width * 0.38, hem * 0.22,
                                          hem * 0.12, cx, bottom + hem * 1.42),
                            hues.hair_fine, clip=clump, alpha=0.11, soft=0.70)

    clumps = h.bangs + h.sides
    for i, clump in enumerate(clumps):
        front = union_all(clumps[i + 1:]) if i + 1 < len(clumps) else None
        visible = clump if front is None else (clump - front)
        bx0, by0, bx1, by1 = clump.bbox
        hh = max(float(by1) - float(by0), 1e-4)

        if i % 2:
            brush.paint(canvas, visible, hues.hair_shadow, alpha=0.27, soft=1.8)
        else:
            brush.paint(canvas, visible, hues.hair_fine, alpha=0.13, soft=1.8)
        lift = min(0.014, max(0.006, hh * (0.050 if i < len(h.bangs) else 0.070)))
        hem = visible - visible.place(cy=lift)
        brush.paint(canvas, hem, hues.hair_deep, alpha=0.34, soft=0.42)
        lip = (visible - visible.place(cy=lift * 1.75)) - hem
        brush.paint(canvas, lip, hues.hair_fine, alpha=0.075, soft=0.55)

    cap_cov = brush.mask(canvas, h.cap)
    _, Ycrown = canvas.grid()

    crown_cov = cap_cov * np.clip(
        (Ycrown - (head.hairline_y(st) - 0.030)) / 0.030, 0.0, 1.0)
    if cap_cov.max() > 0.0:
        _, Yg = canvas.grid()
        above = np.clip((Yg - (head.hairline_y(st) - 0.015)) / 0.030, 0.0, 1.0)
        brush.paint_cov(canvas, cap_cov * above,
                        palette.mix(hues.hair, hues.hair_shadow, 0.14), alpha=1.0)

    if cap_cov.max() > 0.0:
        _, Yc = canvas.grid()
        roll = np.clip((Yc - (head.hairline_y(st) - 0.04)) / 0.34, 0.0, 1.0)
        brush.paint_cov(canvas, cap_cov * (1.0 - roll), hues.hair_shadow,
                        alpha=0.225, soft=2.0)

        _, Xc = canvas.grid()
        key = np.clip((Xc / max(float(st.head_w), 1e-6) + 1.0) * 0.5, 0.0, 1.0)
        brush.paint_cov(canvas, crown_cov * key, hues.hair_shadow,
                        alpha=0.085, soft=2.6)
        brush.paint_cov(canvas, crown_cov * (1.0 - key),
                        hues.hair_fine, alpha=0.065, soft=2.6)

        part = float(h.p.get("part") or 0.0)
        px = part * float(st.head_w) * 0.30
        part_line = S.lock([(px, head.hairline_y(st) + 0.105),
                            (px + part * 0.010, 0.415)], w0=0.010, w1=0.004,
                           bulge=0.004, per_segment=4)
        brush.paint(canvas, part_line & h.cap, hues.hair_shadow,
                    alpha=0.13, soft=1.6)
        for i, lock in enumerate(h.crown):

            body = lock & h.cap
            if i % 2:
                brush.paint(canvas, body, hues.hair_shadow,
                            clip=crown_cov, alpha=0.340, soft=0.65)
            else:
                brush.paint(canvas, body, hues.hair_fine,
                            clip=crown_cov, alpha=0.230, soft=0.75)

            brush.paint(canvas, lock.outline(lw * 0.42), hues.hair_shadow,
                        clip=crown_cov, alpha=0.255, soft=0.65)

        for i, lock in enumerate(h.crown):
            cx = 0.5 * (float(lock.bbox[0]) + float(lock.bbox[2]))
            sgn = 1.0 if cx >= 0.0 else -1.0
            ridge = (lock - lock.place(cx=sgn * 0.012)) & h.cap
            brush.paint(canvas, ridge, hues.hair_fine, clip=crown_cov,
                        alpha=0.085 if i % 2 else 0.055, soft=0.70)
            crease = (lock - lock.place(cx=-sgn * 0.014)) & h.cap
            brush.multiply(canvas, crease, 0.82, clip=crown_cov, alpha=0.30)

        for i, (lock, side) in enumerate(h.flow):

            if i % 3 == 1:
                brush.paint(canvas, lock, hues.hair_fine, clip=crown_cov,
                            alpha=0.205, soft=1.2)
            elif i % 3 == 2:
                brush.paint(canvas, lock, hues.hair_shadow, clip=crown_cov,
                            alpha=0.190, soft=1.2)
            else:
                brush.paint(canvas, lock, hues.hair_fine, clip=crown_cov,
                            alpha=0.155, soft=1.2)
            ridge = lock - lock.place(cx=float(side) * 0.018)
            brush.paint(canvas, ridge, hues.hair_hi, clip=crown_cov,
                        alpha=0.125, soft=0.75)

            crease = lock - lock.place(cx=-float(side) * 0.016, cy=-0.004)
            brush.paint(canvas, crease, hues.hair_shadow, clip=crown_cov,
                        alpha=0.235, soft=0.95)
        for i, (line, _side) in enumerate(h.flow_lines):

            if i % 3 == 1:
                brush.paint(canvas, line, hues.hair_shadow, clip=crown_cov,
                            alpha=0.230, soft=0.75)
            else:
                brush.paint(canvas, line, hues.hair_fine, clip=crown_cov,
                            alpha=0.310, soft=0.75)

    if h.bangs:
        fringe_cov = brush.mask(canvas, S.union_all(h.bangs))
        _, Yf = canvas.grid()

        below = np.clip((head.hairline_y(st) - Yf) / 0.020, 0.0, 1.0)
        fringe_cov = fringe_cov * below
        if fringe_cov.max() > 0.0:
            top = head.hairline_y(st) - 0.020
            g = np.clip((Yf - top) / 0.100, 0.0, 1.0)
            brush.paint_cov(canvas, fringe_cov * (1.0 - g), hues.hair_shadow,
                            alpha=0.14, soft=1.0)
            brush.paint_cov(canvas, fringe_cov * g, hues.hair_fine,
                            alpha=0.08, soft=1.0)

    if h.front_strands:
        front_cov = brush.mask(canvas, S.union_all(h.bangs + h.sides))
        front_cov = front_cov * np.clip((head.hairline_y(st) - Yg) / 0.020, 0.0, 1.0)
        n_b = len(h.bangs)
        if n_b:

            bangs = h.front_strands[:n_b]
            _paint_strands(canvas, hues, bangs[0::2], lw * 0.82,
                           clip=front_cov, alpha=0.60, colour=hues.hair_deep)
            _paint_strands(canvas, hues, bangs[1::2], lw * 0.72,
                           clip=front_cov, alpha=0.50,
                           colour=palette.mix(hues.hair_fine, hues.hair_hi, 0.42))

            lit = [(strand.place(cx=0.010 if j % 2 == 0 else -0.010), top)
                   for j, (strand, top) in enumerate(bangs)]
            _paint_strands(canvas, hues, lit, lw * 0.58,
                           clip=front_cov, alpha=0.36,
                           colour=palette.mix(hues.hair_fine, hues.hair_hi, 0.55))

        if len(h.sides):
            _paint_strands(canvas, hues, h.front_strands[n_b:], lw * 0.62,
                           clip=front_cov, alpha=0.29)

    if cap_cov.max() > 0.0:
        for lock in h.shine:
            brush.paint(canvas, lock, hues.hair_hi, clip=crown_cov,
                        alpha=0.095, soft=1.4)
            brush.multiply(canvas, lock, 1.09, clip=crown_cov, alpha=0.14)
    gloss = float(h.p.get("gloss", 0.0))
    if gloss > 0.12:

        for lock in h.shine:

            brush.multiply(canvas, lock, 1.24 + 0.070 * gloss, clip=crown_cov, alpha=0.55)


def describe(h: Hair) -> str:
    return (f"{h.name}: {len(h.bangs)} bangs, {len(h.sides)} sidelocks, "
            f"{len(h.back_locks)} back, {len(h.near_locks)} near, {len(h.shine)} shine, "
            f"{len(h.crown)} crown")
