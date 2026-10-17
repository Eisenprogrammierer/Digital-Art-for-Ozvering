# Digital-Art-for-Ozvering

A set of tools for making music and assets for my strategy game project.

This repository currently holds **ozart**: a procedural 2-D asset generator. Assets
are not drawn -- they are *computed* from geometry (signed distance fields), affine
linear algebra, deterministic noise, and perceptual (OKLab) colour.

## Why procedural

* **Copyright is unambiguous.** An asset is `f(seed, params) -> PNG`. No training
  data, no scraped art, no licence to trace. The generator is MIT; the output is
  yours to ship.
* **Reproducible.** The same seed always yields byte-identical PNGs, so generated
  art can be versioned, diffed and rebuilt alongside the code that made it.
* **Consistent by construction.** One light direction, one palette generator and
  one bevel width across every asset means the whole set looks like one art set,
  even though nothing was drawn by hand.
* **Scales.** Because geometry lives in world units, the same description renders
  a 24 px icon and a 512 px illustration.

## Requirements

Python 3.10+, `numpy`, `Pillow` (see `requirements.txt`).

```bash
pip install -r requirements.txt
python cli.py demo --out out --seed 7
```

The demo writes `out/icons.png`, `out/shapes.png`, `out/texture.png`,
`out/palette.png`, `out/arms.png` and `out/flags.png`, each with a `.json`
sidecar recording the seed, the spec and the blazon of every tile.

## Usage

```bash
python cli.py demo    --out out --seed 7        # everything
python cli.py icons   --out out/icons.png --seed 3 --count 8 --size 192 --cols 4
python cli.py shapes  --out out/shapes.png      # SDF reference sheet
python cli.py texture --out out/texture.png --seed 3 --size 256
python cli.py palette --out out/palette.png --seed 3 --count 6
python cli.py arms    --out out/arms.png --seed 7 --count 12
python cli.py flags   --out out/flags.png --seed 7 --count 12 --ratio 3:2
python cli.py blazon  --seed 5 --out out/one.png # text + render, both languages
python tests/test_core.py                       # 46 tests, no pytest needed
python tests/test_heraldry.py                   # 31 tests
```

## Core toolkit (`core/`)

| Module | What it provides |
| --- | --- |
| `mat` | Affine transforms in homogeneous coordinates: translate/rotate/scale/shear, composition, inversion, `uniform_scale` for metric correctness. |
| `sdf` | Distance-field primitives (disc, rounded box, capsule, ring, arbitrary polygon, regular polygon, star) and operators: hard/smooth union, intersection, difference, stroke (`outline`), offset, plus `transformed`/`moved`/`rotated`/`scaled`/`mirrored`/`repeated`. |
| `noise` | Deterministic hash-based value noise, fBm, ridged/turbulence variants, domain warping, Worley cells (F1) and Worley cell walls (F2−F1). |
| `palette` | sRGB ↔ OKLab/OKLCH, gamut fitting by chroma bisection, perceptually even mixing and ramps, harmony schemes, and farthest-point `distinct_palette`. |
| `canvas` | Linear-light RGBA buffer with a world-space camera (`scale`, `dx`), correct source-over compositing, sRGB conversion only on export. |
| `render` | Distance field → pixels: anti-aliased `coverage`, bevel lighting from the field's gradient, separable blur, drop shadows, contact sheets. |
| `export` | PNG writer (SVG/atlases/credits to come). |

### The three ideas worth remembering

1. **A distance field is a height field waiting to happen.** Its gradient gives
   surface normals, so bevels, rim light and specular highlights are free:

   ```python
   d = sdf.smooth_difference(box(X, Y), circle(X, Y, 0.4), k=0.15)
   canvas.over(render.shade("#3a7bd5", d, dx=canvas.dx, bevel=0.1), render.coverage(d, canvas.dx))
   ```

2. **Smooth-min buys fillets.** `smooth_union(a, b, k)` welds seams with a fillet
   of radius `k` in one polynomial; `smooth_difference` does the same for carved
   recesses. It is *sound* (never overestimates distance) at the cost of at most
   `k/4` of thickness.

3. **Mix colour in OKLab, decision-make in OKLCH.** Ramps stay perceptually even
   and palettes get guaranteed separation, which is what stops a generated set
   from turning into a pile of near-identical blues.

## Heraldry and flags (`gen/heraldry/`)

Heraldry is the best possible first customer for a procedural renderer, because it
is *already* a formal system: a coat of arms is a composition of a field, some
ordinaries and some charges, painted in tinctures that obey a rule. So the module
is a small compiler, not a drawing routine.

| Module | What it provides |
| --- | --- |
| `tinctures` | The tincture table (metals, colours, stains, furs) with the rule of tincture as `check(below, above)`, aliases ("gold" → `or`), and `variant` for faction shades. |
| `field` | Divisions (per pale/fess/bend/saltire/pall/chevron, quarterly, tierced) and variations (barry, paly, bendy, chevronny, checky, lozengy, gyronny, furs) as *partitions of the plane* into SDF regions, plus `tincture_at`/`field_layers`. |
| `ordinaries` | Chief, fess, pale, bend, chevron, cross, saltire, pall, pile, canton, bordure: builders, real proportions, `thin` scaling, fimbriation. |
| `shields` | Shield silhouettes (heater, iberian, french, rounded, oval, lozenge, banner) and flag shapes (rectangle, swallowtail, pennant, guidon) with their proportions. |
| `charges` | 29 charges built from SDF primitives in a normalised unit box — a lion is a set of capsules and discs, so it reads as a *generated* beast rather than a traced one. |
| `paint` | Flat and embossed tincture painting, rim light, drop shadow, all in linear light. |
| `blazon` | Composition, `validate`, EN/RU blazon text, `random_arms`. |
| `flags` | Flag patterns (bicolour, tricolour, cross, Nordic cross, saltire, canton, hoist triangle, bordered), `describe`, `random_flag`. |

An asset is a **spec** — a plain dict — and everything else is a view of it:

```python
from gen.heraldry import blazon, flags

spec = blazon.random_arms(seed=5)
blazon.blazon_text(spec, "en")   # 'Gyronny of eight Gules and Argent, three lozenges Or.'
blazon.blazon_text(spec, "ru")   # 'В вилообразном червлёном и серебряном поле три золотых ромба.'
blazon.validate(spec)            # [] -- legal by construction, not by luck
blazon.render(spec, size=384)    # a Canvas

flag = flags.random_flag(seed=6, ratio=(3, 2))
flags.describe(flag, "ru")       # 'Скандинавский крест (серебро) в червлёной кайме на лазоревом поле.'
```

Note that the *text* is derivable, which is the point: a coat can be reviewed in a
pull request, stored in a save file, or shown to a player without a picture.

### How the rule of tincture is enforced

Practice does not apply the rule identically to an ordinary and to a charge, so
the validator does not either:

* A **charge** is judged at its **anchor points**, where it is laid out from,
  because that is where the eye fixes it. A charge spanning a partition is normal
  heraldry (*Party per pale Gules and Azure, a lion Or*) and is not reported. What
  *is* reported is a charge that **disappears** into a tincture beneath its own
  footprint, which is always a fault.
* An **ordinary** is a band running across the field, so it is judged against
  **every division it crosses**. That is why a bordure over a field carrying both
  a metal and a colour is reported: no single tincture can contrast with both, and
  heraldry's answer is to counterchange or not to do it.
* A **fur** is neutral on either side, and is sampled as the fur rather than as
  its component tinctures, so an ermine field does not manufacture violations.

`random_arms` is legal by construction rather than by rejection: it tries
arrangements in random order and keeps the first whose anchors all admit a legal
tincture, falling back on a single charge at the centre, which always does. Over
300 seeds, `validate` returns no warnings.

Flags are **textile, not armorial**: any two tinctures may touch (real flags do it
constantly), so the rule is not enforced there and `validate` is advisory.

## Roadmap

Generators to sit on top of the core, in the order they become useful:

- **icons** -- resource/unit icons, atlas packing, `--size` sweep for mip levels. *(prototype in `cli.py`)**
- **heraldry** -- ~~shields, field divisions, charges, tincture rules~~ *(done:
  `gen/heraldry/`)*; next, a faction's crest derived from its name as the seed.
- **flags** -- ~~bands by projection onto directions, overlaps by half-plane
  clipping~~ *(done: `gen/heraldry/flags.py`)*; next, cloth folds as sinusoidal
  normal displacement, so a flag hangs rather than sits.
- **ui** -- 9-slice frames and panels, ornamental borders from a Fourier series
  of a radius function, wood/stone textures via fBm + domain warping.
- **maps** -- hex/axial grids, Voronoi provinces with Lloyd relaxation, adjacency
  from Delaunay, rivers by Dijkstra + Catmull-Rom, coastlines by marching squares.
- **textures** -- Truchet tiles, Wang tiles for seamless tiling, Gray-Scott
  reaction-diffusion for organic patterns.
- **export extras** -- SVG for vector UI, atlas packing, and an
  `assets.yaml` registry carrying `source/license/attribution/seed` for every
  asset. That registry also generates the credits screen, which matters because
  the Kevin MacLeod tracks are CC-BY and *require* attribution.

## Licensing notes

* Code: MIT (see `LICENSE`).
* Generated PNGs: no third-party rights are involved -- keep the seed in the
  registry and the asset can always be rebuilt.
* Watch these three, they are the usual ways third-party rights leak in:
  **fonts** (use OFL/Apache, or draw glyphs procedurally -- Pillow's built-in
  bitmap font is used here for dev previews only), **external textures/brushes**
  (don't), and **CC-BY music** (attribution is mandatory).
