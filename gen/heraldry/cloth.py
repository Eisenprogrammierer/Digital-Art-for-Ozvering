"""Cloth: a flag hangs, it does not sit.

Every other renderer in this package paints flat, because heraldry is flat by
tradition -- a shield is a *sign*, not a surface.  A flag is not a sign: it is a
rectangle of cloth under tension, and the reason a flat flag reads as a picture
*of* a flag rather than as a flag is precisely that hanging cloth never stays in
a plane.

Two things make it hang, and they come from one height field:

* **The light.**  A sinusoid ``z(t)`` across the fly has normals proportional to
  ``(-dz/dt, 0, 1)``, so the Lambert term of *the same light the rest of the
  project uses* comes out as alternating bands down the flag.  Nothing is drawn:
  the folds are the lighting.
* **The projection.**  Displacing the surface along its own normal moves it out
  of the screen, which under even a small oblique view shifts the *pattern*
  across the flag -- the cloth bunches where it comes forward and stretches where
  it dips.  The displacement is therefore applied to the finished RGBA rather
  than to a mask, so the silhouette and the design cannot disagree about where
  the cloth is.

One envelope decides everything that moves.  ``taper`` is the cloth's *freedom* at
the hoist as a fraction of its freedom at the fly, and it is applied to the folds
and to the droop alike, because both of them are the same physical statement: the
cloth is held there and free here.  Set it to zero and the hoist is *exactly* still
-- for any phase -- which is what a flag on a pole looks like, and the one thing
that makes a flag hang *from* something rather than sag in mid-air.  A second
harmonic at a different frequency is added because one sinusoid is a corrugation,
and cloth is not corrugated.

**The droop is gravity, not a fold.**  It is an in-plane slip downwards that grows
with the cloth's freedom, so the fly ends up lower than the hoist and a pennant
*trails* instead of flying level.  It takes no part in the lighting, because the
slope that bends light is the fold's and the droop is a displacement of whole
columns of cloth -- and it runs downwards whichever way the folds run, which is why
it is computed from the same envelope instead of being folded into the height.

**There are two ways to hold a cloth, and they are a quarter turn apart.**  A pole
holds the flag at an edge that runs *beside* its folds -- the hoist, parallel to every
fold line -- so the fold's own coordinate is the one that says how far the cloth is
from what is holding it, and :func:`hang` reads the envelope off exactly that.  A bar
holds it at an edge that runs *across* them, over the top, and then the folds keep
their direction while the envelope moves to the other axis: a gonfalon's folds run down
the cloth just as a flag's do, and what decides how deep they are is now the distance
*down* from the bar.  :func:`drape` is that second mounting, and that is the whole of
the difference -- one envelope, the same gravity, the same one light, and the same
height field doing the lighting.

The one thing that changes is what the droop *is*.  Held at the side of its folds, a
cloth trails: the free edge slips downwards and sideways, and a pennant hangs point
down.  Held along the top of them, the cloth is stretched *along* its own folds -- the
bar does not move, the hem goes lower than the cloth was cut, and what a gonfalon
gains from gravity is length rather than a trail.

None of it is special-cased in the renderer: like every other asset here, a
hanging flag is a *spec* -- ``spec["wave"]`` -- so it round-trips through the
sidecar, and :mod:`gen.heraldry.blazon` still knows nothing about any of it.
"""
from __future__ import annotations

import numpy as np

from core import render

from . import paint as _paint

#: The single light shared by every generated asset.  A flag lit from the other
#: side is the fastest way to make it look borrowed.
LIGHT = _paint.LIGHT

__all__ = ["LIGHT", "DEFAULTS", "wave", "reach", "freedom", "surface", "drift",
           "hang", "drape"]

#: What a flag gets when it is asked to hang and nothing more is said.
#:
#: The wavelength is the parameter that matters most, and it is shorter than a
#: cloth-hanger's instinct suggests: what the light reads is the *slope* of the
#: fold, ``amplitude * 2*pi / wavelength``, and a long shallow wave is a gradient
#: rather than a fold.  Three or four folds across a flag, each a few percent
#: deep, is what a flag on a pole actually looks like.
DEFAULTS = {
    "amplitude": 0.035,     # of the flag's extent along the folds' axis
    "wavelength": 0.30,     # ditto -- a third of the extent is one fold
    "taper": 0.45,          # the cloth's freedom where it is held, of the free end's
    "harmonic": 0.3,        # strength of the second, higher frequency
    "shear": 0.5,           # in-plane displacement per unit of height
    "droop": 0.0,           # how far the free end slips down, of the folds' extent
    "direction": "vertical",
}

#: The second harmonic's frequency, as a multiple of the first.  Not a rational
#: multiple, so the pattern does not repeat exactly along the flag.
_HARMONIC_RATIO = 1.7

#: How hard the folds catch the light: the Lambert term is scaled about the value
#: a *flat* flag would have, so the mean brightness is preserved and a fold is a
#: redistribution of light rather than a darkening.
CONTRAST = 0.55


def wave(seed: int = 0, amplitude=None, wavelength=None, direction=None,
         taper=None, harmonic=None, shear=None, droop=None, phase=None) -> dict:
    """Build a wave spec.  ``seed`` sets the phase; everything else is explicit.

    Four things are refused rather than clamped, because each of them is a silent
    wrong answer otherwise: a direction that is neither vertical nor horizontal
    (which would quietly pick one), a negative amplitude (which would mean
    nothing), a wavelength of zero (which would divide) and a negative droop
    (which is not a flag, it is a flag being blown upwards).

    ``taper`` of zero is worth building on purpose: it is the anchored cloth, the
    one a pole is holding, and it is exactly still at the hoist for any phase.
    """
    if direction is None:
        direction = DEFAULTS["direction"]
    if direction not in ("vertical", "horizontal"):
        raise ValueError(f"a wave runs vertical or horizontal, not {direction!r}")
    out = {
        "amplitude": float(DEFAULTS["amplitude"] if amplitude is None else amplitude),
        "wavelength": float(DEFAULTS["wavelength"] if wavelength is None else wavelength),
        "direction": direction,
        "taper": float(DEFAULTS["taper"] if taper is None else taper),
        "harmonic": float(DEFAULTS["harmonic"] if harmonic is None else harmonic),
        "shear": float(DEFAULTS["shear"] if shear is None else shear),
        "droop": float(DEFAULTS["droop"] if droop is None else droop),
        "phase": float(phase if phase is not None
                       else np.random.default_rng(int(seed)).uniform(0.0, 2.0 * np.pi)),
    }
    if out["amplitude"] < 0.0:
        raise ValueError("a fold has a height, not a negative one")
    if out["wavelength"] <= 0.0:
        raise ValueError("a wavelength of zero is not a wavelength")
    if out["droop"] < 0.0:
        raise ValueError("a droop is gravity; it does not pull upwards")
    return out


def reach(spec: dict) -> float:
    """How far the cloth wanders in-plane, as a fraction of its own extent.

    A flag that is asked to hang needs room for the fly to move, so the renderer
    shrinks the design by this much before waving it -- otherwise the edge of the
    cloth would wave off the edge of the canvas, and the flag would be cut off in
    the one place the eye is looking.  The folds and the droop travel in different
    directions, so adding them is conservative in both; a margin is one number, and
    the honest thing is to give it the largest one rather than to pretend.
    """
    amp = abs(float(spec.get("amplitude", DEFAULTS["amplitude"])))
    shear = abs(float(spec.get("shear", DEFAULTS["shear"])))
    harm = abs(float(spec.get("harmonic", DEFAULTS["harmonic"])))
    droop = abs(float(spec.get("droop", DEFAULTS["droop"])))
    return shear * amp * (1.0 + harm) + droop


def freedom(spec: dict, t):
    """How free the cloth is at ``t``, 0 where it is held and 1 at its free end.

    One envelope for everything that moves, because the folds and the droop are
    the same statement about the same cloth.  ``taper`` is the value where the cloth
    is held, so a taper of zero is a cloth that is *held*: nothing there moves, at
    any phase, and that is what a pole -- or a bar -- buys.
    """
    t = np.asarray(t, dtype=np.float64)
    taper = float(spec.get("taper", DEFAULTS["taper"]))
    return taper + (1.0 - taper) * np.clip(t, 0.0, 1.0)


def _shape(spec: dict, t):
    """The fold's own profile at ``t``, normalised so ``amplitude`` means it."""
    k = 2.0 * np.pi / max(float(spec["wavelength"]), 1e-9)
    ph = float(spec.get("phase", 0.0))
    harm = float(spec.get("harmonic", DEFAULTS["harmonic"]))
    wave = np.sin(k * t + ph) + harm * np.sin(_HARMONIC_RATIO * (k * t + ph))
    return wave / (1.0 + harm)


def surface(spec: dict, t, extent: float = 1.0, envelope=None):
    """The cloth's height at ``t``: 0 where it is held, 1 at its free end, in world units.

    Normalised by ``1 + harmonic`` so that ``amplitude`` means what it says -- the
    largest displacement the cloth actually makes, not the largest of one of its
    terms, which is what :func:`reach` needs to be able to trust.

    ``envelope`` is the coordinate the *freedom* is read at, and it defaults to ``t``.
    The two are the same coordinate whenever the cloth is held at the side of its
    folds, which is a flag on a pole and is why one argument used to do; they come
    apart when it is held *across* them -- a gonfalon on a bar -- where the fold's own
    profile runs across the cloth and the distance from what is holding it runs down.
    """
    t = np.asarray(t, dtype=np.float64)
    v = t if envelope is None else np.asarray(envelope, dtype=np.float64)
    amp = abs(float(spec["amplitude"]))
    return amp * float(extent) * freedom(spec, v) * _shape(spec, t)


def drift(spec: dict, t, extent: float = 1.0, envelope=None, drop=None):
    """The cloth's in-plane displacement at ``t``, as ``(along, down)`` world units.

    ``along`` is the fold pushing the cloth sideways, signed by the fold's own
    height; ``down`` is the droop, and it is *always* downwards -- gravity does not
    consult the folds about which way they run.  The droop rides the envelope alone
    rather than the fold's oscillation, because a droop that oscillated would lift
    the fly above the hoist on the next half-turn of the sinusoid.

    ``envelope`` is the freedom's coordinate, as in :func:`surface`.  ``drop`` is the
    span the droop is a fraction *of*, and it defaults to ``extent``: for a cloth held
    at the side of its folds the two spans are the same one, and for one held across
    them they are perpendicular and neither is a stand-in for the other.
    """
    t = np.asarray(t, dtype=np.float64)
    v = t if envelope is None else np.asarray(envelope, dtype=np.float64)
    shear = abs(float(spec.get("shear", DEFAULTS["shear"])))
    droop = abs(float(spec.get("droop", DEFAULTS["droop"])))
    length = float(extent if drop is None else drop)
    return (shear * surface(spec, t, extent, envelope=v),
            -droop * length * freedom(spec, v))


def _sample(arr, x, y):
    """Bilinear sample of ``arr`` at float pixel coordinates, zero outside.

    Outside means *transparent*, not clamped: the fold carries the silhouette
    with it, so a column that has moved off the cloth must arrive empty rather
    than smeared from the border.
    """
    h, w = arr.shape[:2]
    inside = (x >= 0.0) & (x <= w - 1) & (y >= 0.0) & (y <= h - 1)
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    xi0 = x0.astype(np.int64)
    yi0 = y0.astype(np.int64)
    xi1 = np.minimum(xi0 + 1, w - 1)
    yi1 = np.minimum(yi0 + 1, h - 1)
    if arr.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
        inside = inside[..., None]
    top = arr[yi0, xi0] * (1.0 - fx) + arr[yi0, xi1] * fx
    bot = arr[yi1, xi0] * (1.0 - fx) + arr[yi1, xi1] * fx
    return (top * (1.0 - fy) + bot * fy) * inside


def _span(canvas, along_x: bool, box=None):
    """The span the cloth occupies on one axis, ``(lo, hi)`` in world units.

    Given ``box``, that; otherwise measured from the drawn pixels, which is the
    frame when the caller has nothing better, and is the answer to within an edge
    pixel.  See :func:`hang` for why a caller that knows the cloth's own box should
    say so.
    """
    X, Y = canvas.grid()
    u = X if along_x else Y
    if box is not None:
        return float(box[0]), float(box[1])
    drawn = np.asarray(canvas.a) > 0.5
    cells = np.where(drawn.any(axis=0) if along_x else drawn.any(axis=1))[0]
    if cells.size:
        line = u[0, cells] if along_x else u[cells, 0]
        return float(line.min()), float(line.max())
    return float(u.min()), float(u.max())


def _shed(canvas, z, move_x, move_y, light=LIGHT, contrast: float = CONTRAST):
    """Warp the cloth out of its plane by a displacement and shade it by ``z``.

    ``move_x`` and ``move_y`` are in *world* units and world-signed -- ``+y`` is up
    the world, not up the screen -- and the canvas's own ``y_up`` is what turns them
    into rows, in one place, so that neither caller has to have an opinion about it.

    The pixels are warped first and shaded second, so that the shading belongs to the
    geometry the eye ends up seeing rather than to the geometry the renderer started
    with.  RGB is warped *premultiplied*: sampling straight alpha would pull the
    transparent black around a flag into its own edge, and the edge of a flag is the
    whole silhouette.
    """
    ys = np.arange(canvas.h, dtype=np.float64)[:, None] + np.zeros((1, canvas.w))
    xs = np.arange(canvas.w, dtype=np.float64)[None, :] + np.zeros((canvas.h, 1))
    i_src = xs - move_x * canvas.scale
    # Raster rows run downwards while world y runs up, so the same physical offset
    # is the opposite sign in pixel space.  (This is the mistake the whole project
    # keeps one eye on: see core.render.height_normals.)
    j_src = ys + (move_y * canvas.scale if canvas.y_up else -move_y * canvas.scale)

    premultiplied = canvas.rgb * canvas.a[..., None]
    a = _sample(canvas.a, i_src, j_src)
    rgb = _sample(premultiplied, i_src, j_src)
    canvas.a = a
    canvas.rgb = np.divide(rgb, a[..., None], out=np.zeros_like(rgb), where=a[..., None] > 1e-12)

    # The folds are the lighting: the normals of the height field against the one
    # light.  The modulation is centred on the flag's own mean illumination, so a
    # fold *redistributes* light rather than taking it away -- a hanging flag is
    # exactly as bright as a flat one, which is both true of cloth and the only
    # way a sheet of them can be looked at side by side.
    n = render.height_normals(z, canvas.dx, canvas.y_up)
    L = np.asarray([light[0], light[1], 0.65], dtype=np.float64)
    L = L / max(float(np.linalg.norm(L)), 1e-12)
    lam = np.clip(n @ L, 0.0, 1.0)
    flat = float(L[2])
    inside = canvas.a > 0.5
    if inside.any():
        flat = float(lam[inside].mean())
    factor = 1.0 + float(contrast) * (lam - flat) / max(float(L[2]), 1e-9)
    weight = np.clip(canvas.a, 0.0, 1.0)
    canvas.rgb *= (1.0 + (factor - 1.0) * weight)[..., None]
    np.clip(canvas.rgb, 0.0, 1.0, out=canvas.rgb)
    return canvas


def hang(canvas, spec: dict, light=LIGHT, contrast: float = CONTRAST, box=None):
    """Wave and light a finished flag, in place, and return it.

    The cloth is held at the side of its folds -- a pole, or nothing -- and that is
    what ``t`` measures: :func:`drape` is the other mounting, where what holds the
    cloth runs across them instead.

    ``box`` is the span the cloth occupies along its folds, ``(lo, hi)`` in world
    units, when the caller knows it -- the renderer drew it, so it does.  It matters
    because the frame usually carries margin the cloth is going to move into, and
    measuring the folds against the frame instead would put the hoist partway up the
    fold: "taper at the hoist" would not mean the hoist, and the anchor a pole puts
    there would not be an anchor.  Without it the drawn extent is used, which is the
    same answer to within an edge pixel.
    """
    X, Y = canvas.grid()
    along_x = spec.get("direction", DEFAULTS["direction"]) == "vertical"
    u = X if along_x else Y
    lo, hi = _span(canvas, along_x, box)
    extent = hi - lo
    t = (u - lo) / max(extent, 1e-9)
    z = surface(spec, t, extent)

    # The cloth that lands in each output pixel.  Along the folds' axis the offset
    # is the shear and across it there is nothing; the droop is the other way round
    # -- it moves every column of cloth down the screen whichever way the folds run,
    # so for folds that already run down the screen the two simply add.
    along, down = drift(spec, t, extent)
    move_x = along if along_x else np.zeros_like(along)
    move_y = down + (0.0 if along_x else along)
    return _shed(canvas, z, move_x, move_y, light, contrast)


def drape(canvas, spec: dict, light=LIGHT, contrast: float = CONTRAST, box=None,
          drop=None):
    """Hang a cloth from a bar, in place, and return it: the quarter-turn mounting.

    A gonfalon is the same cloth as a flag, and this function is exactly as long as
    it takes to say what a bar changes.  The folds still run *down* the cloth and
    still vary across it, so ``box`` is still the span along the folds' axis and the
    fold's own profile is still read from it.  What has moved is the thing that says
    how deep a fold is.  On a pole the cloth travels away from the hoist *along* the
    wave's axis, so one coordinate serves both the profile and the envelope; on a bar
    it travels away from the bar *down*, and the two come apart.  ``drop`` is that
    span, ``(top, bottom)`` in world units with the bar at ``top`` -- which, on the
    canvases these renders build, counting y upwards, is the larger number -- and the
    freedom is read from it rather than from ``t``.

    The droop is the same gravity, and it is worth a sentence because it is where the
    two mountings differ most.  A cloth held at the side of its folds trails: the free
    end slips sideways-down.  A cloth held along the top of its folds cannot trail --
    the bar does not move -- so the same slip becomes a *stretch* down the cloth, and
    the droop is measured against ``drop`` rather than against the folds' span: a
    tenth of a wide flag and a tenth of a long one are not the same number of
    pixels, and only the span the cloth actually hangs through can say which is which.

    A wave with horizontal folds is allowed here and means what it says: folds that
    run across the cloth, held along a bar at the top of them, deepening downwards.
    That is a gathered and pinned cloth rather than a hanging banner, and it is a
    different object from the same wave on a pole -- which is the point.
    """
    along_x = spec.get("direction", DEFAULTS["direction"]) == "vertical"
    X, Y = canvas.grid()
    u = X if along_x else Y
    lo, hi = _span(canvas, along_x, box)
    extent = hi - lo
    t = (u - lo) / max(extent, 1e-9)

    if drop is None:
        lo_y, hi_y = _span(canvas, False)
        top, bottom = hi_y, lo_y        # the bar is over the top of the cloth
    else:
        top, bottom = float(drop[0]), float(drop[1])
    depth = abs(top - bottom)
    # Distance down from the bar, 0 at the bar and 1 at the hem.
    v = (top - Y) / max(depth, 1e-9)

    z = surface(spec, t, extent, envelope=v)
    along, down = drift(spec, t, extent, envelope=v, drop=depth)
    # Gravity, taken downwards in the world whichever way the folds run -- that is
    # what makes a cloth a hanging one, and it is the same statement :func:`hang`
    # makes.  What differs is where the shear goes: the envelope is always down the
    # screen, while the folds' axis is x for vertical folds and y for horizontal
    # ones, so the shear and the droop come out perpendicular in the first case and
    # along the same axis in the second.
    move_x = along if along_x else np.zeros_like(along)
    move_y = down + (0.0 if along_x else along)
    return _shed(canvas, z, move_x, move_y, light, contrast)
