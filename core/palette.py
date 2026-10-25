"""Perceptual colour: sRGB <-> OKLab/OKLCH, mixing, harmony, gamut fitting.

Mixing colours in sRGB is physically wrong (it produces muddy midpoints and
uneven lightness ramps).  All interpolation here happens in OKLab/OKLCH, which
is near-uniform perceptually, so generated palettes and gradients keep constant
apparent lightness and step evenly by hue.

The pipeline keeps one convention: **artists specify sRGB** (hex or 0..1
triples), the renderer works in **linear light**, and OKLab is used for
*decisions* (mixing, ramps, distinctness).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "srgb_to_oklab",
    "oklab_to_srgb",
    "srgb_to_oklch",
    "oklch_to_srgb",
    "oklch_to_linear",
    "hex_to_srgb",
    "srgb_to_hex",
    "hex_to_oklch",
    "oklch_to_hex",
    "mix",
    "mix_many",
    "ramp",
    "ramp_map",
    "lighten",
    "darken",
    "shift_hue",
    "saturate",
    "delta_e",
    "color",
    "harmony",
    "distinct_palette",
    "random_palette",
]

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_linear(c) -> np.ndarray:
    """Decode an sRGB-encoded value in ``[0, 1]`` to linear light."""
    c = np.asarray(c, dtype=np.float64)
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c) -> np.ndarray:
    """Encode linear light back to sRGB in ``[0, 1]``."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)
    return np.clip(out, 0.0, 1.0)


def srgb_to_oklab(rgb) -> np.ndarray:
    """sRGB (0..1, gamma-encoded) -> OKLab ``(L, a, b)``."""
    lin = srgb_to_linear(rgb)
    lms = lin @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_srgb(lab) -> np.ndarray:
    """OKLab ``(L, a, b)`` -> sRGB (0..1, gamma-encoded), clipped to gamut."""
    lab = np.asarray(lab, dtype=np.float64)
    lms_ = lab @ _M2_INV.T
    lin = (lms_**3) @ _M1_INV.T
    return linear_to_srgb(lin)


def srgb_to_oklch(rgb) -> np.ndarray:
    """sRGB -> OKLCH.  Hue is in ``[0, 1)`` turns, chroma is unbounded above 0."""
    lab = srgb_to_oklab(rgb)
    return _lab_to_lch(lab)


def oklch_to_srgb(lch) -> np.ndarray:
    """OKLCH -> sRGB, clipping out-of-gamut chroma rather than clipping RGB."""
    return oklab_to_srgb(_lch_to_lab(fit_gamut(lch)))


def oklch_to_linear(lch) -> np.ndarray:
    """OKLCH -> linear-light RGB, *without* gamut fitting."""
    lab = _lch_to_lab(np.asarray(lch, dtype=np.float64))
    lms_ = lab @ _M2_INV.T
    return (lms_**3) @ _M1_INV.T


def _lab_to_lch(lab) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    C = np.hypot(a, b)
    h = np.mod(np.arctan2(b, a) / (2.0 * np.pi), 1.0)
    return np.stack([L, C, h], axis=-1)


def _lch_to_lab(lch) -> np.ndarray:
    lch = np.asarray(lch, dtype=np.float64)
    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    ang = h * 2.0 * np.pi
    return np.stack([L, C * np.cos(ang), C * np.sin(ang)], axis=-1)


def fit_gamut(lch, tol: float = 1e-4) -> np.ndarray:
    """Reduce chroma by bisection until the colour fits inside sRGB.

    Naive clipping of the RGB channels shifts hue and flattens saturated
    colours; desaturating in OKLCH preserves both hue and lightness.
    """
    lch = np.asarray(lch, dtype=np.float64)
    single = lch.ndim == 1
    lch = np.atleast_2d(lch)
    out = lch.copy()
    for i, (L, C, h) in enumerate(lch):
        L = float(np.clip(L, 0.0, 1.0))
        if C <= tol:
            out[i] = (L, 0.0, h)
            continue
        lin = _lch_to_linear_arr(L, C, h)
        if np.all(lin >= -tol) and np.all(lin <= 1.0 + tol):
            out[i] = (L, C, h)
            continue
        lo, hi = 0.0, C
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            lin = _lch_to_linear_arr(L, mid, h)
            if np.all(lin >= -tol) and np.all(lin <= 1.0 + tol):
                lo = mid
            else:
                hi = mid
        out[i] = (L, lo, h)
    return out[0] if single else out


def _lch_to_linear_arr(L, C, h) -> np.ndarray:
    ang = h * 2.0 * np.pi
    lab = np.array([L, C * np.cos(ang), C * np.sin(ang)])
    lms_ = lab @ _M2_INV.T
    return (lms_**3) @ _M1_INV.T


def hex_to_srgb(s: str) -> np.ndarray:
    """``"#rrggbb"`` (or ``"rrggbb"``) -> sRGB triple in ``[0, 1]``."""
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"expected a 3 or 6 digit hex colour, got {s!r}")
    return np.array([int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


def srgb_to_hex(rgb) -> str:
    """sRGB triple in ``[0, 1]`` -> ``"#rrggbb"``."""
    v = np.clip(np.round(np.asarray(rgb, dtype=np.float64) * 255.0), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*v)


def hex_to_oklch(s: str) -> np.ndarray:
    return srgb_to_oklch(hex_to_srgb(s))


def oklch_to_hex(lch) -> str:
    return srgb_to_hex(oklch_to_srgb(lch))


def mix(c1, c2, t: float) -> np.ndarray:
    """Linear blend of two sRGB colours in OKLab; ``t=0`` gives ``c1``."""
    lab = (1.0 - t) * srgb_to_oklab(c1) + t * srgb_to_oklab(c2)
    return oklab_to_srgb(lab)


def mix_many(colors, t: float) -> np.ndarray:
    """Sample a multi-stop gradient at ``t`` in ``[0, 1]`` (OKLab, C0)."""
    colors = np.asarray(colors, dtype=np.float64)
    if len(colors) == 1:
        return colors[0]
    t = float(np.clip(t, 0.0, 1.0)) * (len(colors) - 1)
    i = min(int(t), len(colors) - 2)
    return mix(colors[i], colors[i + 1], t - i)


def ramp(colors, n: int) -> np.ndarray:
    """``n`` evenly spaced colours along a multi-stop gradient, shape ``(n, 3)``."""
    return np.stack([mix_many(colors, i / max(n - 1, 1)) for i in range(n)])


def ramp_map(colors, t) -> np.ndarray:
    """Vectorised multi-stop gradient evaluated at every element of ``t``.

    The per-pixel counterpart of :func:`mix_many`: a whole texture can be
    colourised with one pass, and because the interpolation happens in OKLab the
    ramp never dips in brightness between stops the way sRGB would.
    """
    labs = srgb_to_oklab(np.asarray(colors, dtype=np.float64))
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * (len(labs) - 1)
    i = np.clip(np.floor(t).astype(np.int64), 0, len(labs) - 2)
    f = (t - i)[..., None]
    return oklab_to_srgb(labs[i] * (1.0 - f) + labs[i + 1] * f)


def lighten(c, amount: float = 0.1) -> np.ndarray:
    """Add ``amount`` to OKLab lightness, refitting chroma to stay in gamut."""
    lch = srgb_to_oklch(c).copy()
    lch[0] = np.clip(lch[0] + amount, 0.0, 1.0)
    return oklch_to_srgb(lch)


def darken(c, amount: float = 0.1) -> np.ndarray:
    return lighten(c, -amount)


def shift_hue(c, turns: float) -> np.ndarray:
    """Rotate hue by ``turns`` (0.5 = complementary)."""
    lch = srgb_to_oklch(c).copy()
    lch[2] = np.mod(lch[2] + turns, 1.0)
    return oklch_to_srgb(lch)


def saturate(c, factor: float = 1.25) -> np.ndarray:
    """Multiply OKLCH chroma, refitting to gamut."""
    lch = srgb_to_oklch(c).copy()
    lch[1] = max(lch[1] * factor, 0.0)
    return oklch_to_srgb(lch)


def delta_e(c1, c2) -> float:
    """Euclidean distance in OKLab: a cheap, perceptually sane colour distance."""
    return float(np.linalg.norm(srgb_to_oklab(c1) - srgb_to_oklab(c2)))


def color(L: float, C: float, hue: float) -> np.ndarray:
    """Build an sRGB colour from OKLCH; hue in turns, chroma auto-fitted."""
    return oklch_to_srgb((float(np.clip(L, 0.0, 1.0)), float(C), float(hue)))


def harmony(hue: float, n: int = 5, scheme: str = "analogous", L: float = 0.65, C: float = 0.15) -> list:
    """Family of hues around ``hue`` following a classical colour scheme.

    ``scheme`` is one of ``monochromatic``, ``analogous``, ``complementary``,
    ``triadic``, ``tetradic``.  Each returned colour is gamut-fitted, so a
    saturated scheme degrades gracefully instead of clipping to mud.
    """
    if scheme == "monochromatic":
        return [color(l, C, hue) for l in np.linspace(max(L - 0.28, 0.12), min(L + 0.28, 0.95), n)]
    if scheme == "analogous":
        offs = np.linspace(-0.09, 0.09, n)
    elif scheme == "complementary":
        offs = np.linspace(0.0, 0.5, n)
    elif scheme == "triadic":
        offs = np.array([0.0, 1 / 3, 2 / 3])[:n] if n <= 3 else np.linspace(0.0, 2 / 3, n)
    elif scheme == "tetradic":
        offs = np.linspace(0.0, 0.75, n)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return [color(L, C, hue + o) for o in offs]


def distinct_palette(n: int, seed: int = 0, min_de: float = 0.16, L: float = 0.66, C: float = 0.16, resolution: int = 96) -> list:
    """Exactly ``n`` colours, greedily maximising the minimum OKLab separation.

    Candidates sit on a grid over hue (offset by a seeded jitter so different
    seeds pick different representatives) and, when ``n * min_de`` says the hue
    circle alone would be crowded, over a few lightness levels as well.  Each
    next colour is the candidate furthest from everything chosen so far -- the
    greedy 2-approximation to max-min dispersion.

    ``min_de`` is therefore a target rather than a hard promise: the sampler
    widens the lightness spread to reach it, and still returns ``n`` distinct
    colours when hue and lightness are both exhausted.
    """
    if n < 1:
        return []
    rng = np.random.default_rng(seed)
    jitter = float(rng.random())
    hues = (jitter + np.arange(resolution) / resolution) % 1.0

    levels = 1 if n * min_de <= 0.9 else 5
    offsets = np.linspace(-0.22, 0.22, levels) if levels > 1 else np.array([0.0])
    cands = np.concatenate([np.stack([color(L + dl, C, h) for h in hues]) for dl in offsets])
    labs = srgb_to_oklab(cands)

    first = int(rng.integers(0, len(cands)))
    chosen = [first]
    dists = np.linalg.norm(labs - labs[first], axis=-1)
    while len(chosen) < n:
        pick = int(np.argmax(dists))
        chosen.append(pick)
        dists = np.minimum(dists, np.linalg.norm(labs - labs[pick], axis=-1))
    return [cands[i] for i in chosen]


def random_palette(rng: np.random.Generator, n: int = 5, scheme: str | None = None, L: float = 0.65, C: float = 0.16) -> list:
    """Draw a harmonious palette using a seeded generator (choices, not noise)."""
    hue = float(rng.random())
    if scheme is None:
        scheme = str(rng.choice(["analogous", "complementary", "triadic", "tetradic", "monochromatic"]))
    return harmony(hue, n=n, scheme=scheme, L=float(np.clip(L + rng.uniform(-0.08, 0.08), 0.2, 0.9)), C=C)
