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
  along the crown, which is what stops hair looking like painted-on plastic.

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

from . import brush, cel, head
from . import shape as S
from .shape import Shape, union_all

__all__ = ["HAIR_STYLES", "Hair", "build", "paint_back", "paint_front", "mass",
           "front_mass", "describe"]

#: Per-hairstyle parameters.  ``drop`` values are in head units, measured down
#: from the root; a drop of 1.0 hangs a full head height below the crown.
HAIR_STYLES = {
    "short":  dict(vol=0.100, bangs=7, bw=0.230, blen=0.55, sides=1, sw=0.220, sdrop=0.300,
                   back=8, kw=0.330, kdrop=0.520, crest=0, spike=0, cap_r=2.40, flare=1.02),
    "bob":    dict(vol=0.115, bangs=7, bw=0.235, blen=0.75, sides=2, sw=0.300, sdrop=0.620,
                   back=9, kw=0.345, kdrop=0.740, crest=0, spike=0, cap_r=1.70, flare=1.18),
    "long":   dict(vol=0.130, bangs=7, bw=0.250, blen=0.60, sides=2, sw=0.290, sdrop=1.300,
                   back=10, kw=0.360, kdrop=1.100, crest=0, spike=0, cap_r=1.25, flare=1.36),
    "hime":   dict(vol=0.115, bangs=7, bw=0.240, blen=0.95, sides=2, sw=0.280, sdrop=1.150,
                   back=10, kw=0.350, kdrop=1.000, crest=0, spike=0, cap_r=1.30, flare=1.30,
                   straight=True),
    "twin":   dict(vol=0.115, bangs=7, bw=0.230, blen=0.65, sides=1, sw=0.270, sdrop=0.900,
                   back=9, kw=0.340, kdrop=0.580, crest=0, spike=0, cap_r=1.60, flare=1.16,
                   twin=2),
    "pony":   dict(vol=0.105, bangs=7, bw=0.230, blen=0.55, sides=1, sw=0.260, sdrop=0.450,
                   back=8, kw=0.330, kdrop=0.500, crest=1, spike=0, cap_r=2.20, flare=1.06),
    "spiky":  dict(vol=0.095, bangs=8, bw=0.200, blen=0.30, sides=1, sw=0.220, sdrop=0.320,
                   back=9, kw=0.320, kdrop=0.450, crest=0, spike=7, cap_r=2.00, flare=1.10,
                   sl=0.170),
    "wild":   dict(vol=0.135, bangs=9, bw=0.200, blen=0.45, sides=2, sw=0.260, sdrop=0.520,
                   back=10, kw=0.350, kdrop=0.640, crest=0, spike=6, cap_r=1.45, flare=1.22,
                   sl=0.160),
}


def _params(name: str, st, genes: dict) -> dict:
    """Style parameters, stretched by the genome's ``hair_len`` and ``bang_n``."""
    p = dict(HAIR_STYLES[name])
    length = float(genes.get("hair_len", 0.0))
    scale = float(np.clip(1.0 + 0.34 * length, 0.70, 1.60))
    for key in ("sdrop", "kdrop"):
        p[key] = float(p[key]) * scale
    density = float(genes.get("bang_n", 0.0))
    p["bangs"] = int(np.clip(round(float(p["bangs"]) + 2.0 * density), 3, 10))
    p["back"] = int(np.clip(round(float(p["back"]) + 2.0 * density), 4, 11))
    return p


def _hull(st, vol: float):
    return st.head_w + vol, 0.505 + vol


def _root(st, vol: float, theta: float, turn: float, inset: float = 0.0) -> np.ndarray:
    rx, ry = _hull(st, vol - inset)
    x = rx * np.cos(theta)
    y = ry * np.sin(theta) + 0.015
    x, y = head.turn_point(x, y, turn, st.head_w)
    return np.array([float(x), float(y)], dtype=np.float64)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
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
    far, near = [], []
    for i in range(n):
        u = (i + 0.5) / n
        th = np.pi * (0.19 + 0.62 * u)
        root = _root(st, vol, th, turn, inset=kw * 0.58)
        side = 1.0 if root[0] >= 0.0 else -1.0
        length = drop * float(rng.uniform(0.88, 1.10))
        flare = float(p.get("flare", 1.25))
        tip = np.array([root[0] * flare + side * 0.050 * float(rng.uniform(0.5, 1.4)),
                        root[1] - length])
        mid = np.array([root[0] * (1.0 + (flare - 1.0) * 0.46) + side * 0.018,
                        root[1] - length * 0.46])
        lock = S.lock([root, mid, tip], w0=kw * float(rng.uniform(0.88, 1.10)),
                      w1=kw * float(rng.uniform(0.10, 0.28)), bulge=kw * 0.14,
                      per_segment=7, resample=34)
        (near if abs(th - np.pi * 0.5) > np.pi * 0.16 else far).append(lock)
    return far, near


def _bangs(st, p: dict, rng, turn: float) -> list:
    """The fringe: many clumps of *different* lengths, with cut ends.

    The length is not a style parameter, it is a measurement.  A fringe that hangs
    past the brow hides the face's expression, and one that stops short leaves a
    bare forehead the style's proportions never allowed for, so each clump is cut
    to land between the brow and the upper lid and the *style's* own landmarks
    decide where that is.  The style only biases the result (``blen``), says how
    many clumps there are, and how far they sway.
    """
    n = int(p["bangs"])
    bw, vol = float(p["bw"]), float(p["vol"])
    fill = float(p.get("blen", 1.0))
    straight = bool(p.get("straight", False))
    brow = float(getattr(st, "brow_y", 0.10))
    upper = float(st.eye_y) + float(st.eye_up) * 0.55
    # Which side the fringe sweeps away from.  A fringe with a parting is worth
    # more than any amount of extra clumps: an even curtain of hair across the
    # forehead is the single strongest "generated" tell a portrait can have.
    part = 0.0 if straight else float(rng.uniform(-0.45, 0.45))
    out = []
    for i in range(n):
        u = (i + 0.5) / n
        th = np.pi * (0.17 + 0.66 * u)
        root = _root(st, vol, th, turn, inset=bw * 0.66)
        side = 1.0 if root[0] >= 0.0 else -1.0
        # A blunt cut is blunt on purpose: one length across the brow, clumps that
        # stay wide to the tip so they meet, and no taper to make a saw.  Pointed
        # clumps of uneven length are a different hairstyle, not a longer version
        # of this one.
        if straight:
            edge, jitter, tip_w = 1.0, float(rng.uniform(0.94, 1.06)), bw * 0.52
        else:
            edge = 1.0 - 0.42 * abs(u - 0.5) * 2.0
            jitter, tip_w = float(rng.uniform(0.80, 1.20)), bw * 0.13
        sweep = u - (0.5 + 0.5 * part)
        short = max(root[1] - (brow + 0.030 + 0.075 * abs(sweep) * 2.0), 0.05)
        long = max(root[1] - upper, short + 0.02)
        length = (short + (long - short) * fill) * jitter * edge
        sway = (0.022 + 0.040 * abs(u - 0.5)) * float(rng.uniform(-1.0, 1.0))
        tip = np.array([root[0] * 1.12 + sway + sweep * 0.170 + side * 0.018,
                        root[1] - length])
        mid = np.array([root[0] * 1.05 + sway * 0.40 + sweep * 0.075,
                        root[1] - length * 0.50])
        out.append(S.lock([root, mid, tip], w0=bw * float(rng.uniform(0.70, 1.30)),
                          w1=tip_w, bulge=bw * 0.16, per_segment=7, resample=32))
    return out


def _sidelocks(st, p: dict, rng, turn: float) -> list:
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
    out = []
    for sign in (+1, -1):
        if turn != 0.0 and (sign * turn) < 0.0:
            continue  # the receding side's locks are behind the jaw
        for k in range(n):
            th = np.pi * ((0.19 + 0.055 * k) if sign > 0 else (1.0 - 0.19 - 0.055 * k))
            inset = sw * 0.55 + 0.030 * k
            root = _root(st, vol, th, turn, inset=inset)
            length = drop * float(rng.uniform(0.88, 1.08)) * (1.0 - 0.12 * k)
            bow = 0.014 if straight else 0.048
            out_x = float(st.head_w) * 0.92 + vol * 0.25 - sw * 0.24 + 0.025 * k
            tip = np.array([sign * out_x, root[1] - length])
            mid = np.array([sign * (out_x + bow), root[1] - length * 0.42])
            out.append(S.lock([root, mid, tip],
                              w0=sw * (1.0 - 0.32 * k) * float(rng.uniform(0.90, 1.06)),
                              w1=sw * (0.30 - 0.06 * k), bulge=sw * 0.13,
                              per_segment=7, resample=34))
    return out


def _crest(st, p: dict, rng, turn: float) -> list:
    """Ponytails, twin tails and spikes: bundles that live outside the skull."""
    vol = float(p["vol"])
    out = []
    for _ in range(int(p.get("crest", 0))):
        side = 1.0 if rng.random() < 0.5 else -1.0
        # Rooted on the silhouette itself, so the tail's own root arc lands on the
        # cap's contour instead of drawing a line across the crown.
        root = _root(st, vol, np.pi * 0.5, turn, inset=0.01)
        rise = float(rng.uniform(0.03, 0.08))
        fall = float(rng.uniform(0.78, 1.06))
        # Gathered at the crown and hung *down* the back, so the visible part is
        # the length that clears the head's silhouette beside the jaw.  A tail
        # arched up and over the crown reads as a handle on a jug.
        out.append(S.lock(
            [root,
             np.array([root[0] + side * 0.19, root[1] + rise]),
             np.array([root[0] + side * 0.27, root[1] - fall * 0.42]),
             np.array([root[0] + side * 0.23, root[1] - fall])],
            w0=0.115, w1=0.032, bulge=0.038, per_segment=8, resample=44))
    for _ in range(int(p.get("twin", 0)) // 2):
        for side in (+1, -1):
            w = float(p["sw"]) * 0.40
            # Tied just above the hairline at the side of the skull and swung
            # *outward* from there.  A tail that hangs straight down from a root on
            # the crown crosses the cheek and reads as a flap glued to the face;
            # twin tails belong beside the head, out where the silhouette is.
            root = _root(st, vol, np.pi * (0.20 if side > 0 else 0.80), turn, inset=0.02)
            drop = float(p["sdrop"]) * 0.92
            sway = float(rng.uniform(0.88, 1.16))
            out.append(S.lock(
                [root,
                 root + np.array([side * 0.170 * sway, -drop * 0.26]),
                 root + np.array([side * 0.195 * sway, -drop * 0.64]),
                 root + np.array([side * 0.145 * sway, -drop])],
                w0=w * 1.15, w1=w * 0.45, bulge=w * 0.50, per_segment=8, resample=44))
    for _ in range(int(p.get("spike", 0))):
        # Drawn toward the temples.  The top of the crown is the one place the
        # frame cannot afford a spike -- there are about ten hundredths of a unit
        # of headroom above the cap -- and the sides are where a spiky cut shows
        # anyway, because that is where it leaves the silhouette.
        d = float(rng.uniform(0.0, 1.0)) ** 0.60 * 0.24 + 0.12
        th = np.pi * (0.5 + (d if rng.random() < 0.5 else -d))
        # On the hull, not inside it: a spike's root is a cut, and a cut hidden
        # inside the mass is a line drawn across the crown.
        root = _root(st, vol, th, turn, inset=0.01)
        rx, ry = _hull(st, vol)
        radial = np.array([np.cos(th) / max(rx, 1e-6), np.sin(th) / max(ry, 1e-6)])
        radial = radial / max(float(np.linalg.norm(radial)), 1e-9)
        ln = float(p.get("sl", 0.16)) * float(rng.uniform(0.85, 1.25))
        # The frame's top edge is at y = 0.70.  A spike that crosses it is a shape
        # cut in half and reads as a horn, so the length gives way to the frame
        # rather than the other way round.
        ln = min(ln, max(0.030, (0.664 - float(root[1])) / max(float(radial[1]), 0.18)))
        lean = np.array([radial[1], -radial[0]]) * float(rng.uniform(-0.12, 0.12))
        out.append(S.lock([root, root + radial * ln * 0.6 + lean,
                           root + radial * ln + lean * 2.0],
                          w0=0.045 + 0.42 * ln, w1=0.022, bulge=0.042,
                          per_segment=6, resample=30))
    return out


def _shine(st, p: dict, rng, turn: float) -> list:
    """The gloss: shallow arcs just inside the silhouette along the crown.

    A shine is a *reflection*, so it is thin and it follows the form -- a fat
    blob of highlight reads as a bald patch, not as hair.
    """
    vol = float(p["vol"])
    out = []
    for i in range(3):
        u = (i + 0.5) / 3.0
        th = np.pi * (0.24 + 0.52 * u)
        centre = _root(st, vol, th, turn, inset=0.075)
        above = _root(st, vol, th + 0.13, turn, inset=0.075)
        below = _root(st, vol, th - 0.13, turn, inset=0.075)
        out.append(S.lock([below, centre, above], w0=0.030, w1=0.016,
                          bulge=float(rng.uniform(0.036, 0.058)), per_segment=6, resample=26))
    return out


class Hair:
    """Everything a hairstyle is made of, ready to paint in two passes.

    ``back_locks`` are the locks that hang behind the head and ``near_locks`` the
    ones rooted at the temples, which hang beside the jaw and over the shoulder
    and so have to be painted *after* the body.  See :func:`_back_locks`.
    """

    __slots__ = ("name", "p", "cap", "back_locks", "near_locks", "bangs", "sides",
                 "crest", "shine", "back_mass", "front_mass", "mass")

    def __init__(self, name, p, cap, back_locks, near_locks, bangs, sides, crest, shine):
        self.name = name
        self.p = p
        self.cap = cap
        self.back_locks = back_locks
        self.near_locks = near_locks
        self.bangs = bangs
        self.sides = sides
        self.crest = crest
        self.shine = shine
        self.back_mass = union_all([cap] + back_locks + crest)
        self.front_mass = union_all(bangs + sides)
        self.mass = union_all([self.back_mass, self.front_mass] + near_locks)


def _cap(st, p: dict, turn: float, hairline: float, skull: Shape | None = None) -> Shape:
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
    r = float(st.head_w) * float(p.get("cap_r", 1.8))
    face = S.ellipse(r, r, cy=hairline - r)
    return hull - face


def build(st, genes: dict, rng=None, turn: float = 0.0, skull: Shape | None = None) -> Hair:
    """Construct a hairstyle for a genome.  Deterministic given ``rng``."""
    if rng is None:
        rng = np.random.default_rng(0)
    name = genes.get("hair_style", "short")
    if name not in HAIR_STYLES:
        raise KeyError(f"unknown hair style {name!r}")
    p = _params(name, st, genes)
    hairline = head.hairline_y(st) - 0.02
    back, near = _back_locks(st, p, rng, turn)
    return Hair(
        name=name,
        p=p,
        cap=_cap(st, p, turn, hairline, skull),
        back_locks=back,
        near_locks=near,
        bangs=_bangs(st, p, rng, turn),
        sides=_sidelocks(st, p, rng, turn),
        crest=_crest(st, p, rng, turn),
        shine=_shine(st, p, rng, turn),
    )


def mass(h: Hair) -> Shape:
    return h.back_mass


def front_mass(h: Hair) -> Shape:
    return h.front_mass


def _hair_dome(st, h: Hair, turn: float = 0.0):
    """A volume over the skull, so the terminator lands where a head's would."""
    return cel.dome(rx=st.head_w * 1.18, ry=0.70, height=0.44, cy=0.03)


# --------------------------------------------------------------------------
# painting
# --------------------------------------------------------------------------
def _paint_locks(canvas, st, hues, h: Hair, locks) -> None:
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
    for lock in locks:
        s = brush.Surface(canvas, lock, pad=2.2 * lw)
        s.paint(canvas, hues.hair)
        s.bands(canvas, [hues.hair_shadow, hues.hair_deep], (0.44, 0.15),
                volume=dome, bevel=0.018, soft=0.5)
        s.stroke(canvas, lw * 0.55, hues.line_hair, inset=lw * 0.24, soft=0.35)


def paint_back(canvas, st, hues, h: Hair, turn: float = 0.0, clip=None):
    """The hair that hangs *behind* the head: the back locks and the crest.

    Nothing here is faded or clipped to hide a lock's root, because the *cap* is
    painted later and covers every root: every lock in a style is rooted above the
    hairline on purpose, and the cap is exactly the hair between the crown and the
    hairline.  That is also why the cap is not painted here -- it sits on the
    skull, in front of the skin, not behind the head with these locks.
    """
    _paint_locks(canvas, st, hues, h, h.back_locks + h.crest)


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
    _paint_locks(canvas, st, hues, h, h.near_locks)


def paint_front(canvas, st, hues, h: Hair, turn: float = 0.0, clip=None):
    """Everything on the near side of the skin: the cap, the fringe, the shine.

    The cap comes first and it comes *here*, not with the back mass, because it is
    the hair lying on the skull: painted behind the head it is covered by the skin
    except for a rim, and then the forehead between the hairline and the fringe's
    hem is bare skin with the fringe hanging over it -- a band of scalp showing
    through the partings, which is the one thing a fringe never does.  With the
    cap on top of the skin, the partings show hair, and the fringe's clumps read
    as strands *lying on* the mass instead of gaps cut out of it.

    The clumps are then painted one over the next, so a clump in front hides the
    strand line of the one behind it: only visible boundaries get inked, which is
    the whole difference between strands and a wire mesh.

    The ink is faded out above the hairline.  A clump's outline runs right around
    it, so its *root* -- a straight cut lying inside the crown -- is inked along
    with its sides, and a row of those reads as teeth.  The clumps do get painted
    over the cap's own shading, but that is invisible: the two passes differ only
    by a constant offset to the *height*, and a constant offset does not change a
    gradient, so both quantise to the same band to the last bit.  Give a clump a
    bevel of its own and its whole interior lands a band away from the crown it
    lies on, which shows up as pale patches and steps right across the head.
    """
    lw = float(st.ink_w) * float(st.line) * 0.78
    dome = _hair_dome(st, h, turn)
    cap = brush.Surface(canvas, h.cap, pad=2.6 * lw)
    cap.paint(canvas, hues.hair)
    cap.bands(canvas, [hues.hair_shadow, hues.hair_deep], (0.44, 0.15),
              volume=dome, bevel=0.045)
    cap.stroke(canvas, lw, hues.line_hair, inset=lw * 0.42)

    _, Yg = canvas.grid()
    fade = np.clip((float(head.hairline_y(st)) + 0.045 - Yg) / 0.07, 0.0, 1.0)
    if clip is not None:
        fade = fade * clip
    for clump in h.bangs + h.sides:
        s = brush.Surface(canvas, clump, pad=2.6 * lw)
        s.paint(canvas, hues.hair)
        s.bands(canvas, [hues.hair_shadow, hues.hair_deep], (0.44, 0.15),
                volume=dome, bevel=0.0, soft=0.5)
        if not s.empty:
            rows = slice(s.rows, s.rows + s.d.shape[0])
            cols = slice(s.cols, s.cols + s.d.shape[1])
            s.stroke(canvas, lw * 0.50, hues.line_hair, inset=lw * 0.20,
                     mask=fade[rows, cols], alpha=0.92, soft=0.4)
    for lock in h.shine:
        brush.paint(canvas, lock, hues.hair_hi, clip=clip, alpha=0.86, soft=0.8)


def describe(h: Hair) -> str:
    return (f"{h.name}: {len(h.bangs)} bangs, {len(h.sides)} sidelocks, "
            f"{len(h.back_locks)} back, {len(h.near_locks)} near, {len(h.shine)} shine")
