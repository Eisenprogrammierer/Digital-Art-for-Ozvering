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
`out/palette.png`, `out/arms.png`, `out/flags.png`, `out/portraits.png` and
`out/lineage.png`, each with a `.json` sidecar recording the seed, the spec, the
blazon and -- for portraits -- the genome of every tile.

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
python cli.py portrait   --seed 5 --size 384 --out out/one.png
python cli.py portraits  --seed 7 --count 8 --size 256
python cli.py lineage    --seed 7 --children 6  # two parents and their children
python tests/test_core.py                       # 46 tests, no pytest needed
python tests/test_heraldry.py                   # 31 tests
python tests/test_portraits.py                  # 23 tests
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

## Portraits (`gen/portraits/`)

A face is the hardest thing here to get away with, because the eye is unforgiving
in one direction only: a blank icon can be wrong and still useful, a face that is
wrong is *uncanny*. The way through is to lean into the medium instead of away from
it -- draw an anime face, where form is carried by line art, shading is a handful
of flat tones, and "simple" is a style rather than a compromise.

| Module | What it provides |
| --- | --- |
| `style` | Four presets (chibi, shoujo, shounen, seinen) in one parameter table, with the face's vertical layout *derived* from the eye position rather than typed in beside it. |
| `head` | Skull silhouette, neck, trapezius, collar and hairline: control points → closed Catmull-Rom loop → distance field. |
| `shape` | The vocabulary a face is written in -- `lock` (a variable-width tapered stroke), `lens`, `arc`, `poly`, `smooth_loop` -- over `core.sdf`. |
| `brush` | Painting a *field* rather than a rectangle: `Surface` binds a shape to its window and its coverage, so a fill, a band or a stroke is one pass over the pixels that can see it. |
| `ink` | Contour strokes that live on the field: `abs(d + inset) - w/2` is a band along the zero set, so an outline costs no new geometry. |
| `cel` | The quantised Lambert term, the two colour rules, and the ellipsoid that stands in for a skull. |
| `hues` | The whole colour script -- about thirty tones -- derived from four base colours. |
| `eyes`, `face`, `hair`, `expression` | The features themselves. |
| `dna` | The genome: `random_genes`, `encode`/`decode`, `breed`, `mutate`, `apply_to`, `describe`. |
| `portrait` | Framing, layer order, `compose`/`render`/`render_seed`. |

### Five decisions, in the order they matter

1. **Line art carries the form.** Every silhouette is inked along its own contour,
   with a taper that thins the stroke where the form turns away from the viewer.
   Shading is secondary: if the ink is right, the face reads at 64 px.
2. **Cel shading is a hybrid height.** A cheap ellipsoid gives the normals a real
   head would, and a couple of pixels of bevel off the silhouette give the rim. The
   Lambert term is then *quantised*, because a smooth gradient reads as CG and a
   hard terminator reads as animation.
3. **A shadow is a colour, not a darkness.** As lightness falls the chroma *rises*
   and the hue rolls down through magenta towards violet. Rolling it the other way
   -- towards yellow -- is the fastest way to make a face look ill; that is a
   measurement, not a preference. The lightness drop is also scaled by how much
   headroom the base skin has, so a deep tone gets a *richer* shadow rather than a
   sootier one.
4. **Features are symbols.** An anime eye is a lens, an arc, a ring and two
   highlights; a nose is often one dot. Drawing the symbol instead of the organ is
   what keeps the result on the right side of the uncanny line.
5. **The layer order is the drawing.** Hair behind the head, then the body, then the
   locks that hang past the jaw, then the head, then the face, then the hair that
   lies on it. Nearly every pass in that list is somewhere a working portrait
   broke: the skull cap painted with the hair behind the head leaves the crown a
   band of bare forehead, and the locks rooted at the temples painted there leave
   long hair sliced off level with the collar. Order is also the one thing a test
   can only hold down by sampling pixels.

Hair is where the order shows most. Ten locks painted as one union and inked once
along its contour are a sheet with a single terminator, so they are painted one
over the next: a lock fills over the outline of the lock behind it, and what
survives is strands. The fringe is the same pass, and both waves of it shade with
the *same* volume and the same clamp on the height, so a lock lying on the crown
quantises to the same cel band to the last bit -- shade a lock with a bevel of its
own and its interior lands a band away from the hair it lies on, which shows up as
pale steps and triangles right across the head.

The parameterisation is deliberately *not* a wall of independent sliders -- those
produce monsters. Presets hold the anatomy, gaps encode the vertical layout, and
the genome may only push a preset around inside limits a `_clamp_params` pass then
re-checks.

### A portrait is a genome

The strategic asset is not the pixels, it is the string:

```python
from gen.portraits import dna, portrait

genes = dna.random_genes(seed=5)
text  = dna.encode(genes)          # 'style=shounen;hair_style=twin;...' -- the save file
portrait.render(genes, size=384)   # a Canvas; same genes, same bytes

child = dna.breed(dna.random_genes(5), dna.random_genes(1005), seed=12)
dna.describe(child)                # 'chibi · long hair #cba18f · iris … · smile'
```

Colours blend in OKLab, shapes average and drift, categories are drawn from one
parent or the other, and the result is still a genome. `cli.py lineage` renders two
parents and six children for exactly that reason: a strategy game can populate a
dynasty by crossing genomes instead of by inventing a face per character, and the
roster it stores is a few kilobytes of text.

### What the tests hold down

`tests/test_portraits.py` checks that the genome round-trips through its own string,
that bred and mutated children are still readable, that the derived anatomy stays
plausible across hundreds of genomes (the eyes fit in the skull, the jaw tapers,
the features stay in order down the face with the hairline above the brows), and
five composition invariants -- each of which broke once in review, which is why
each has a test:

* the fringe never covers an eye;
* the skull cap and the fringe overlap along the hairline, so no band of bare skull
  shows between them (the artefact is a "crown of teeth");
* the crown is *painted* hair and not skin, sampled from the pixels, because
  geometry alone cannot catch a layer in the wrong place;
* the skull stays in frame with the chin about two thirds down;
* every style × hairstyle × expression builds, and the ones that are rendered
  render something.

## Roadmap

Generators to sit on top of the core, in the order they become useful:

- **icons** -- resource/unit icons, atlas packing, `--size` sweep for mip levels. *(prototype in `cli.py`)**
- **heraldry** -- ~~shields, field divisions, charges, tincture rules~~ *(done:
  `gen/heraldry/`)*; next, a faction's crest derived from its name as the seed.
- **flags** -- ~~bands by projection onto directions, overlaps by half-plane
  clipping~~ *(done: `gen/heraldry/flags.py`)*; next, cloth folds as sinusoidal
  normal displacement, so a flag hangs rather than sits.
- **portraits** -- ~~SDF head, ink contours, cel shading, traceable genome,
  heredity~~ *(done: `gen/portraits/`)*; ~~outfits built from the same shoulder
  line~~ *(done: a collar stroke, three kinds, one per seed)*; next, turn the head
  in depth (a proper 3-D skull projected to 2-D, not a 2-D face that fakes a turn),
  and expressions that move the whole face rather than three features.
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
