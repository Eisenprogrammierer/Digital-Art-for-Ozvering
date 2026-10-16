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

The demo writes `out/icons.png`, `out/shapes.png`, `out/texture.png` and
`out/palette.png` + `out/palette.json`.

## Usage

```bash
python cli.py demo    --out out --seed 7        # everything
python cli.py icons   --out out/icons.png --seed 3 --count 8 --size 192 --cols 4
python cli.py shapes  --out out/shapes.png      # SDF reference sheet
python cli.py texture --out out/texture.png --seed 3 --size 256
python cli.py palette --out out/palette.png --seed 3 --count 6
python tests/test_core.py                       # 39 tests, no pytest needed
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

## Roadmap

Generators to sit on top of the core, in the order they become useful:

- **icons** -- resource/unit icons, atlas packing, `--size` sweep for mip levels. *(prototype in `cli.py`)**
- **heraldry** -- shields, field divisions, charges, tincture rules as a graph
  constraint; a faction's crest derived from its name as the seed.
- **flags** -- bands by projection onto directions, overlaps by half-plane
  clipping, cloth folds as sinusoidal normal displacement.
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
