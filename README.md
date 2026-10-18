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
`out/palette.png`, `out/arms.png`, `out/flags.png`, `out/portraits.png`,
`out/lineage.png`, `out/ui.png`, `out/maps.png`, `out/tiles.png` and
`out/factions.png`, each with a `.json` sidecar recording the seed, the spec, the
blazon, the genome of every portrait tile, the corner insets of every UI part, for
every map its biome census, its rivers and its provinces, for every tiling texture
its seam error, and for the factions the *names* they were derived from -- which,
for that sheet, are the input rather than the seed -- plus
`out/credits.txt` and `out/credits.png`, which are built from those sidecars. The
credits are not a seeded asset and carry no sidecar of their own: they are what
the sidecars are *for*.

## Usage

```bash
python cli.py demo    --out out --seed 7        # everything
python cli.py icons   --out out/icons.png --seed 3 --count 8 --size 192 --cols 4
python cli.py shapes  --out out/shapes.png      # SDF reference sheet
python cli.py texture --out out/texture.png --seed 3 --size 256
python cli.py palette --out out/palette.png --seed 3 --count 6
python cli.py arms    --out out/arms.png --seed 7 --count 12
python cli.py flags   --out out/flags.png --seed 7 --count 12 --ratio 3:2
python cli.py factions --name "Kingdom of Alder"       # a name, and all it decides
python cli.py blazon  --seed 5 --out out/one.png # text + render, both languages
python cli.py portrait   --seed 5 --size 384 --out out/one.png
python cli.py portraits  --seed 7 --count 8 --size 256
python cli.py lineage    --seed 7 --children 6  # two parents and their children
python cli.py ui         --seed 7 --count 3 --scale 2    # nine-slice UI kit sheet
python cli.py maps       --seed 7 --count 2 --cols 26 --rows 17 --sea 0.42 --provinces 8
python cli.py tiles      --seed 7 --size 256              # Truchet, Wang, reaction-diffusion
python cli.py credits    --manifest out         # what we owe, and what we made
python cli.py credits    --manifest out --png out/credits.png   # and the screen
python tests/test_core.py                       # 53 tests, no pytest needed
python tests/test_heraldry.py                   # 39 tests
python tests/test_portraits.py                  # 25 tests
python tests/test_ui.py                         # 22 tests
python tests/test_maps.py                       # 46 tests
python tests/test_textures.py                   # 23 tests
```

## Core toolkit (`core/`)

| Module | What it provides |
| --- | --- |
| `mat` | Affine transforms in homogeneous coordinates: translate/rotate/scale/shear, composition, inversion, `uniform_scale` for metric correctness. |
| `sdf` | Distance-field primitives (disc, rounded box, capsule, ring, arbitrary polygon, regular polygon, star) and operators: hard/smooth union, intersection, difference, stroke (`outline`), offset, plus `transformed`/`moved`/`rotated`/`scaled`/`mirrored`/`repeated`. |
| `noise` | Deterministic hash-based value noise, fBm, ridged/turbulence variants, domain warping, Worley cells (F1) and Worley cell walls (F2−F1). |
| `palette` | sRGB ↔ OKLab/OKLCH, gamut fitting by chroma bisection, perceptually even mixing and ramps, harmony schemes, and farthest-point `distinct_palette`. |
| `canvas` | Linear-light RGBA buffer with a world-space camera (`scale`, `dx`), correct source-over compositing, sRGB conversion only on export. |
| `render` | Distance field → pixels: anti-aliased `coverage`, bevel lighting from the field's gradient (raised or, with `inset=True`, the same ramp turned over into a recess), separable blur, drop shadows, contact sheets. |
| `credits` | The two halves of an attribution: the hand-written list of sources we do not own (with the credit line each licence demands) and the manifest read out of the sidecars every exporter writes -- and the credits screen drawn from both. |
| `export` | PNG writer (SVG/atlases to come). |

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
| `crest` | A faction's whole identity from its *name*: the coat, the banner it flies, and the words for both. |

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

### A faction is its name

Every other generator here is asked for an asset by *seed*. That is right for a
texture and wrong for a faction, because a game does not have a number for the
Kingdom of Alder -- it has the words. So the name is the seed, and there is no
registry to keep in step:

```python
from gen.heraldry import crest

alder = crest.faction("Kingdom of Alder")
alder["blazon"]["en"]        # 'Per bend Argent and Azure, a fess Or.'
alder["description"]["en"]   # 'Horizontal bicolour: argent, or.'
alder["seed"]                 # 4779193178708914195, forever
```

Save nothing; the name is the save file. Two things make that work, and both are
easy to get subtly wrong.

**The hash.** `seed_from` is a cryptographic hash of the name, not Python's
built-in `hash()`. Python salts string hashing per process, so `hash("Alder")` is a
different number every time the game starts: an asset derived from it would change
on every run, and the bug would present as a rendering glitch rather than as a seed.
It is also the kind of failure a single process cannot see, so the test starts a
*fresh interpreter with a different `PYTHONHASHSEED`* and compares the blazon.

**The key.** `normalise` case-folds and collapses whitespace, so "The Grand Duchy of
Alder", "the grand duchy of alder" and "The  Grand  Duchy of Alder" are one faction.

The banner is derived from the **coat**, not from the name a second time, and that
is deliberate: a faction has one crest, and a shield and a flag generated
independently would sooner or later disagree about it. So the banner flies what the
shield bears — the same tinctures, and the same ordinary wherever the ordinary has a
vexillological equivalent, since a cross flies as a cross, a bordure as a border,
and a bend, a chevron or a pile as a triangle at the hoist. Where there is no
equivalent the tinctures carry it, which is honest: a pall has no flag layout and
pretending otherwise would be worse than a bicolour. The test asserts that
relationship rather than either picture.

`faction()` returns the whole thing as one JSON-serialisable dict — coat, banner,
both languages, and the validator's warnings, because a generated coat is something
a person has to be able to review in a sidecar rather than trust.

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

The strand line is the surviving half of that outline, so it fades out before it
reaches the clamp's *root* -- a straight cut lying inside the crown, which inked
along with the sides reads as a row of teeth. Fading it against the *hairline*
instead is the obvious move and it is wrong: the hairline is a landmark of the
face, the roots belong to the hair, and on a style whose hem falls above the
hairline the entire fringe ends up with no ink in it at all, a bare coloured sheet
on the front of the head. Faded against its own root, every clump carries a strand
from its hem up to just short of where it is attached, whatever the style's
proportions are.

An outline is still only an outline, and a clump in front covers most of the one
behind it, so what survives of the ink is a few millimetres of edge at the hem --
a smooth sheet with a sawtooth bottom. The line an artist actually draws runs
*down* a lock, not around it, so each clump gets one: the same spine, less width.
Sharing the spine is the point. A line laid beside it from its own endpoints can
bulge past the hem and draw on the skin, and that is the one thing ink must never
do; asking for less width on the same spine makes the strand a sub-set of its
clump, and -- because that argument goes through two resamplers -- a test measures
the escape at machine precision rather than trusting it. Ponytails and twin tails
get the same line for the same reason, and their fade starts at the root instead
of halfway down, because nothing is ever painted over a tail.

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

One more is structural rather than compositional: every fringe clump carries its
own root height, because that is what the strand ink fades against, and a clump
missing from that list would not fail loudly -- it would just lose its ink on one
style at some sizes. `zip` truncates in silence, so the pairing is asserted.

The strands that lie inside the clumps get a test of their own, and it is a
containment test: a strand is built from its clump's spine with less width, and
"built to be inside" survives two resamplers and a bulge term only if someone
checks. Ink that leaves the silhouette draws on skin.

## UI chrome (`gen/ui/`)

A panel is the first asset here that has to survive being resized to a size
nobody chose. An icon is rendered at the size it is asked for; a panel is drawn
once at a comfortable size and then stretched by the layout, which is what the
nine-slice convention is for. That inverts the usual question about what a
generator produces:

> The asset is not the PNG. The asset is `(PNG, corner insets, minimum size)`.

Everything in the package is arranged around that sentence.

| Module | What it provides |
| --- | --- |
| `style` | The numbers: a palette plus proportions in **design pixels**, the named kits (`parchment`, `slate`, `brass`, `moss`, `glass`), a seeded `random_style`, and the geometric invariant the rest of the package leans on. |
| `frame` | The geometry as distance fields: concentric rounded rectangles, the band between them, the inner rule, and a corner ornament whose radius is a Fourier series. |
| `slice` | The contract itself — the insets, the nine rectangles, and the resample a game engine will perform, implemented here so that it can be tested here. |
| `paint` | Five passes over one fixed light: shadow, body, band, rule, ornament. |

### The one hard geometric rule

```
every bit of curvature in the design must fit inside the corner box,
and the design must be constant across whatever axis gets stretched.
```

A nine-slice resample keeps the corners and stretches the four edges and the
middle. A straight edge survives that; a curve does not — it would smear. So the
corner radius, the bevel and the ornament all have to live within `inset` pixels
of the corner, and the style may vary *along* the stretch axis but never across
it (which is why the body gradient is a function of `y` and of nothing else).

`style.validate` states that, `style.clamp` enforces it, and the test checks the
*pixels* rather than the algebra: outside the corner boxes, every column of the
left strip has to be identical down the panel and every row of the top strip
identical across it. That one assertion catches an ornament that has grown out of
its corner, a rule that curves too far, and a drop shadow that is not separable,
all at once.

### `inset` is derived, not requested

The style states a floor, and `clamp` widens the corner box to whatever the look
actually needs — `radius + bevel`, or the ornament's reach. The nine-slice cut can
therefore never be too small to hold the curvature, by construction rather than by
check. `size` runs the argument the other way: at a size with no room for the
corner box the *look* gives way instead, keeping the radius (which is the shape)
and flattening the bevel (which is only the shading) first.

### The Fourier series goes in the corner

A border whose radius varied around the whole perimeter is exactly the thing a
nine-slice stretch destroys, because the resample stretches that perimeter. So the
radius function is used where the geometry is *fixed*: inside a corner box, which
is never resampled at all. `frame.boss_radius` normalises by its own harmonic
weight, so `r(θ)` stays within `r0 · [1 ± amp]` for any harmonic count, and
`style.boss_limit` inverts the reach formula to cap the ornament at precisely the
size the box can hold — the ornament yields to the frame, not the other way round.

### A state is a colour, never a layout

`idle`, `hover` and `pressed` share one geometry, so a control cannot jitter as
the pointer crosses it, and the silhouette alpha of the three is bit-identical
(tested). `pressed` is the same bevel with the ramp turned over —
`render.shade(..., inset=True)`: a recess is not a second effect, it is one height
field with the sign of its slope flipped, which is why the two share every other
parameter.

Because a style is written in design pixels, `--scale` doubles the pixels and
changes no number: the same spec renders a 48 px icon-button and a 512 px hero
panel, with the same corner geometry in both — which is what makes the insets
mean something.

## Maps (`gen/maps/`)

A map is the first asset here that is *data* before it is a picture: a grid
someone plays on, with a cell size, a coastline, movement costs, rivers and
provinces. So the generator works in cells and draws in pixels, and the two are
kept strictly apart:

> The map is `(seed, cols, rows)`. The hex size decides how many pixels it is
> drawn in, and nothing else.

| Module | What it provides |
| --- | --- |
| `grid` | The lattice in axial coordinates: centres, neighbours, hex distance, the corner normal form, and the per-pixel lookup that resolves a world point to a cell and a position inside it. |
| `terrain` | The geography: a domain-warped elevation field, a moisture field, and the biome table that gives every class of ground a name, a colour and a movement cost. |
| `coast` | The boundary of a set of hexes as closed loops of corners -- the coastline, the lakes and the edge of the map, all from one walk. |
| `rivers` | Where the water goes: pits filled to their spill points, flow accumulated in order of height, and the courses that carry enough of it turned into splines. |
| `region` | The political map: provinces grown by breadth-first search so that they are connected, capitals at their one-medians, and names. |
| `paint` | One raster pass for the ground and its relief, a wash and a frontier line for the provinces, strokes for the rivers, the coast and the frame. |

### Why axial coordinates, and what they buy

The map is stored as a rectangle of *offset* hexes, because that is the shape a
strategy map has, and converted once to *axial* coordinates `(q, r)`, because in
those a hexagon is a lattice and everything afterwards is arithmetic: the six
neighbours are six constant offsets, distance is half the sum of three coordinate
differences, and the centre of a cell is a linear function of its indices.

The raster pass then rests on one fact -- **a hexagon is the Voronoi cell of its
centre** -- so "which cell was this pixel in" is cube rounding and not a search.
Every property of the ground is an array indexed by that answer, which is why the
map has no seams, no anti-aliased cracks between tiles and no per-hex canvas, and
costs one pass over the pixels whatever the hex size.

It also means the hexagon can be measured analytically. Inside the sector towards
one edge, the signed distance to the cell's own border is the apothem minus the
projection of the pixel's offset onto that edge's direction -- so a test can
compare the rasteriser's idea of the hexagon against `sdf.polygon` of the true
corners, to machine precision, which is the check that catches a wrong lattice
constant.

### The relief, and why the creases are drawn

The height under a pixel is not stored: it is interpolated *within* the pixel from
its own cell's height out to the height of the neighbour across the sector's edge.
One hex is six triangles from its centre, the two cells agree exactly on the edge
they share, and the surface therefore creases there and only there -- which is
what a hex map looks like, hills lit on one flank and the joins landing on the
grid instead of fighting it.

That is honest, and nearly invisible, because most of the map really is flat. So
the creases get drawn: where the ground drops steeply across an edge, each of the
two cells draws its own half of a dark band along it, and the threshold comes from
the map's own distribution of drops rather than from a constant, so a smooth
continent and a shattered one both get the right amount of shading.

### The coastline is the edge of the land

A coastline is not a curve fitted to the terrain, it is the set of hex edges with
land on one side and water on the other, walked into closed loops. On a square
grid that walk needs a rule for corners, because two cells can meet at one without
sharing an edge. On a hex grid it does not:

> at every corner exactly three hexes meet, and all three are pairwise
> edge-neighbours, so a corner has at most one boundary edge arriving and one
> leaving and there is never a choice to make.

Which is why the whole walk is a dictionary from a corner to the next corner.
Corners are named by the *lowest* of the three hexes that meet there -- an integer
normal form, so a loop closes exactly on its first corner -- and the walk is built
to keep the land on its left, which makes an island's loop counter-clockwise and a
lake's clockwise. The tests use that: punch a hole in a block of hexes and the
boundary becomes two loops, wound opposite ways.

### Rivers need the pits filled first

A river is a hydrological statement before it is a curve, so the module is a model
with a painter attached. The naive version of it -- "walk to the lowest neighbour"
-- is a disaster on a noise field: a local minimum is a place where the water
stops, and there are thousands of them, so the map fills up with one-hex lakes and
every river is two hexes long. So the pits are filled first, with Barnes' priority
flood grown outwards from the sea: each cell ends up at the lowest height at which
water could actually get from it to the ocean. On real terrain that lifts only 3 to
10% of the land, and the test that keeps it honest checks the answer against a
brute-force minimax relaxation instead of against the shape of the queue.

Filled basins are flat, and a river on a flat has no downhill neighbour to choose,
so the flood adds a hair at each step. That is not cosmetic: it turns an arbitrary
tie-break into data, and the flow direction of a filled basin becomes part of the
map rather than a property of the order the cells happened to be visited in.

Drainage then accumulates in one pass: sort the cells from high to low and add each
one's total to the cell below it. By the time a cell is reached everything above it
has been added, so a single pass gives the true upstream area -- no recursion, no
priority queue, and exact rather than iterative, because flow only goes downhill.
A river's width follows the square root of that, which is the standard hydraulic
relation and the reason a river fattens towards its mouth.

The road map said "rivers by Dijkstra + Catmull-Rom". The Catmull-Rom is there and
it is the last thing that happens. The Dijkstra is not, because on a grid where the
height is known everywhere there is nothing to search: the way to the sea is the way
the gradient points, and a shortest-path search would only be needed to route around
a pit -- which filling has already removed.

And which cells get to be a river is a **quantile** of the land's own drainage, for
the same reason the sea level is a quantile of elevation: a fixed threshold draws a
dense network on a wet map and nothing on a dry one, so `--river` would mean two
different things on two different seeds. Taking the top few per cent instead makes
the density of the network a property of the request and not of the luck.

### A province is connected, or it is not a province

The obvious way to divide a map into provinces -- give every cell to its nearest
capital -- is one line of arithmetic and it produces provinces in two pieces,
because a bay does not care about the straight line between two points. A province
in two pieces cannot be governed: an army cannot get from one half to the other
without leaving it. So the assignment here is a multi-source breadth-first search
over the lattice, which is the map's own adjacency, and a province comes out
connected by construction.

Seeds are spread by Mitchell's best candidate and then relaxed by Lloyd's
algorithm: assign, move each capital to the middle of its province, assign again.
The middle of a province is its one-median -- the cell that minimises the path
distance to every other cell in it -- and it is computed exactly, by a breadth-first
search from each of the province's own cells, rather than approximated by the
centroid. Two passes are enough for a capital to land within one hex of the median
and within ten per cent of the best possible total distance, which is what the test
asserts; more passes would buy the last hex and cost another search per cell per
province.

A landmass no capital can reach becomes a province of its own, with its capital at
its own one-median. That is not a special case bolted on -- it is what an island is,
and reaching it by distance across the sea would produce the two-piece province this
whole module exists to prevent. The count is therefore a floor: an archipelago gets
more provinces than it asked for, and the sidecar says how many of them are islands.

Names are the only text anywhere in these assets, and they are drawn on the preview
sheet, never in the PNG: this project ships one font (Pillow's) and a game has its
own, so the sidecar carries the names and the sheet shows them. A name is a stem
plus an ending that suits the ground -- `-fell` for a province mostly of hills,
`-fen` for one mostly of marsh -- which is where the geography of a province is
written down, and it keeps eleven provinces from all being called "High-"
something.

### One thing the pixels taught the code

Provinces are drawn as a wash plus a frontier line, and the first version of the
frontier was invisible. Not because it was faint, but because on a hex map *every*
hex edge is already drawn as lattice, and a frontier is a set of hex edges: a line
of the lattice's weight lands exactly on a line that is there anyway. What tells a
player where one province ends is that the line is *heavier*, not that it is another
colour. So the frontier is drawn twice as wide as the lattice, the lattice went
slightly lighter, and the map reads terrain, then provinces, then hexes -- in that
order, which is the order the three of them matter in.

Two of the numbers are worth stating because they are the difference between a map
and a picture of one. The sea level is a **quantile**, so `--sea 0.42` means 42% of
the hexes are water for any seed; and the biome table keeps the *rules* -- what a
cell costs to walk through -- in the same row as its colour, so the art and the
gameplay cannot drift apart.

## Tiling textures (`gen/textures/`)

Every other asset here is a picture of one thing at one size: an icon, a portrait,
a panel that will be stretched, a map of a particular grid. A tiling texture is the
first one that is asked a question about its *edges*:

> Lay it down next to itself, four times, and there must be no line where they meet.

The modules in this package exist because that question has three different
answers, and each one is a different *kind* of guarantee -- one that is geometric,
one that is searched for, and one that is never created in the first place.

| Module | What it provides |
| --- | --- |
| `tile` | What "seamless" means as a number (`seam_error`), the per-pixel tile lookup, the periodic noise and periodic blur that keep a wrap, and the project's one light applied to a height field. |
| `truchet` | Truchet tiles: two arcs on a square, an arrangement with nothing to search, and the carved-stone material that comes out of it. |
| `wang` | Wang tiles: four corner colours per tile, a tiling searched for on a torus, and the bilinear blend of those corners into ground. |
| `reaction` | Gray-Scott reaction-diffusion, simulated on a torus, for patterns that are grown rather than drawn. |

### Seamless is a number, not a compliment

`tile.seam_error` divides the step across the wrap by the mean step inside the tile.
A field built to wrap comes out at about 1 -- the last row really is one pixel away
from the first -- and a field that does not, at about 10. That turns the word into
something a test can hold a generator to, and the demo sheet records it per tile, so
a texture that stopped tiling would be visible in the sidecar as well as on the page.

Getting to about 1 is where the interesting failures live. The first Truchet texture
had a *pattern* that wrapped perfectly and visible lines all the same, because the
grain on it was `core.noise` evaluated in pixel coordinates: two tiles apart the
grain hashes a different lattice cell, so the material changed character at the seam
even though the shapes met exactly. It cannot be fixed by choosing a round cell
size -- the value at `x` and the value at `x + width` are different hashes by
construction. What works is to stop asking for noise at a *point* and ask for noise
at a *frequency* instead: damp the spectrum of seeded white noise with a Gaussian.
That is smooth, and it is exactly periodic, because every basis function of a DFT
already is. The same trick does the periodic blur, and for the same reason.

### Truchet has nothing to check; Wang has everything to check

A Truchet tile is two arcs, and **every arc ends at the midpoint of an edge**. The
tile on the other side of that edge sends one of its own arcs to the same point, so
the paths join whatever the neighbour decided, and *any* arrangement is seamless.
There is no constraint to satisfy and nothing to search, which is why the generator
is three lines and a seeded random choice. The package offers two kinds of path, the
arcs and the maze-like lines meeting at the centre; they join the same midpoints, so
they can be mixed in one arrangement without breaking the argument -- and the test
measures the distance field exactly there, at the middle of every edge, because that
is the fact the module rests on.

A Wang tile carries its edges instead, so the choice *is* the problem: put two tiles
next to each other and the shared edge has to agree. The form used here carries the
four **corners**, which reads as terrain rather than as plumbing, and it makes the
checking collapse: two neighbours agree when the two corners they share are equal,
and then a colour field that is the *bilinear blend* of the four corner colours is
continuous at every edge -- the blend at a tile's right-hand edge is the average of
its two right corners, the blend at the neighbour's left-hand edge is the average of
its two left corners, and those are the same two numbers, because that is what
agreeing means. Nothing else needs to be checked, and that is the whole reason the
module can afford a plain depth-first search.

The wrap is part of the constraint, because a texture has to tile on a torus: the
last column must agree with the first. A scanline search gets that almost for free --
the tile it must agree with was laid down a whole row ago -- so the search is a
depth-first walk with a stack, and it can say honestly that a set either tiles or
does not. The `complete` set (all `colours**4` tiles) never backtracks; the
`diagonal` set does, which is why the stack is there; and a set that cannot tile
raises rather than hanging. One number is a rule rather than a knob: `margin`, the
fraction of a tile spent blending, has to stay at or below a half, because past that
the blend reaches the opposite corner and agreeing about the shared corners stops
being enough.

Where Wang is *not* clever is the look, and it is worth saying so: a bilinear blend
is continuous but its *slope* jumps where a flat cell meets a graded one, and a
normal computed from an unsmoothed height turns that kink into a visible edge. So
the relief is blurred before it is lit, the grain on it is gentle, and the colour
gets a pixel of blur too. It still reads as patchwork -- that is what a Wang tiling
of random corner colours is -- so it belongs on ground, and the Truchet and the
reaction are the ones to reach for when the material is the point.

### The reaction runs on a torus, and that is the whole design

Gray-Scott is two chemicals, one reaction, and a diffusion term each:

```
du/dt = Du * lap(u) - u*v**2 + F * (1 - u)
dv/dt = Dv * lap(v) + u*v**2 - (F + k) * v
```

and the only decision that matters here is that the Laplacian **wraps**. The field is
periodic by construction, so the texture tiles and there is no seam to fix,
apologise for or test for -- the same move the map makes when it evaluates terrain in
normalised coordinates, and the same reason: the boundary condition is the asset.
The test asserts it as algebra rather than as a picture, because a boundary condition
is exactly the kind of thing a picture hides: translating the field translates its
Laplacian, exactly, to the last bit.

The rest is engineering around two facts. A pattern has no detail finer than a few
cells, so it is simulated at 96 and sampled up on the torus -- sixteen times cheaper
for a picture nobody could tell apart; and the *upsample* has to wrap as well, since
a clamped edge would put back exactly the border the simulation had removed. And the
initial condition is not a detail: the seeding density decides how much of the tile
the pattern has covered by the time the run ends, because the pattern grows outwards
from wherever it started. A sparse start is a picture of dots however long it runs; a
generous one is a material. The lumps are round for much the same reason -- a square
lump is a square shape at the scale the instability selects, and the first version of
this module grew squares out of it and looked like a low-resolution voxel game.

The five presets are the standard corners of the `(F, k)` plane, and the tests treat
them as what they are: different patterns from the same seed. "The pattern formed" is
measured as coverage and gradient energy rather than as standard deviation, which is
worth stating because the obvious statistic is a trap -- a sparse seed and a finished
stripe network are both two-valued fields, and the deviation can even *fall* as the
pattern eats the lumps it grew from.

### The bug that did not raise

`smooth` had to blur a three-channel field, and the first version reshaped the field
so that the channels led, built a kernel from the last two axes, and multiplied. That
looks like the same operation and is not: `reshape` reinterprets the memory order, so
the planes it produced were interleaved slices of all three channels at once, and the
kernel came out `(height, 3)` instead of `(height, width)`. Nothing raised -- a
`(256, 256, 3)` array times a `(256, 3)` kernel broadcasts perfectly happily -- and
nothing ran slower. The only symptom was a texture that had gone slightly grey, and
the fix is to make the two-dimensional plane the primitive and map over channels,
which cannot do that.

The Truchet module had the same shape of bug: `kinds` was validated, and the drawing
keyed off the *index* (`kind == 0` meant arcs), so every grain drew arcs and asking
for `("lines",)` returned a picture identical to `("arcs",)`. Both were found by
tests, not by looking at output -- one asserting that a red-only field stays red-only,
one asserting that the two kinds differ -- and that is the argument for the tests in
this package being about *properties* rather than about pixels.

## Roadmap

Generators to sit on top of the core, in the order they become useful:

- **icons** -- resource/unit icons, atlas packing, `--size` sweep for mip levels. *(prototype in `cli.py`)**
- **heraldry** -- ~~shields, field divisions, charges, tincture rules~~ and a
  ~~faction's crest derived from its name as the seed~~ *(done: `gen/heraldry/`, with
  `crest.py` deriving the seed from a hash of the name rather than Python's salted
  `hash()`, and deriving the banner from the coat so the two cannot disagree)*;
  next, faction shades of the tinctures in a sidecar, so two factions can both be
  azure and still be told apart.
- **flags** -- ~~bands by projection onto directions, overlaps by half-plane
  clipping~~ *(done: `gen/heraldry/flags.py`)*; next, cloth folds as sinusoidal
  normal displacement, so a flag hangs rather than sits.
- **portraits** -- ~~SDF head, ink contours, cel shading, traceable genome,
  heredity~~ *(done: `gen/portraits/`)*; ~~outfits built from the same shoulder
  line~~ *(done: a collar stroke, three kinds, one per seed)*; next, turn the head
  in depth (a proper 3-D skull projected to 2-D, not a 2-D face that fakes a turn),
  and expressions that move the whole face rather than three features.
- **ui** -- ~~9-slice frames and panels, ornamental borders from a Fourier series
  of a radius function~~ *(done: `gen/ui/`, with the series moved into the corner
  box, which is the only part of a nine-slice asset that is never resampled)*;
  next, asymmetric insets for title bars, and wood/stone textures for the body via
  fBm + domain warping.
- **maps** -- ~~hex/axial grids, Voronoi provinces with Lloyd relaxation, adjacency
  from Delaunay, rivers by Dijkstra + Catmull-Rom, coastlines by marching
  squares~~ *(done: `gen/maps/` -- the lattice, the terrain and its coastline drawn
  as one raster pass, pits filled by a priority flood, rivers accumulated in order
  of height, provinces grown by breadth-first search and relaxed by Lloyd, and
  names).  Three of the plan's ingredients are deliberately not there: adjacency is
  one hex step rather than a Delaunay edge, because that is what an army actually
  walks; a coastline is the edge of the land rather than a marching-squares contour
  of a field, because the land already knows where it ends; and rivers are followed
  down the gradient rather than searched for with Dijkstra, because once the pits
  are filled the gradient is the answer and nothing is left to search*; next, roads
  between capitals, sea provinces, and a campaign-friendly export of the sidecar.
- **textures** -- ~~Truchet tiles, Wang tiles for seamless tiling, Gray-Scott
  reaction-diffusion for organic patterns~~ *(done: `gen/textures/` -- Truchet
  arrangements, which are seamless for any choice; Wang tilings searched for on a
  torus, seamless because the four corner colours of a tile agree with its
  neighbours' and the blend between them is continuous; and a reaction-diffusion
  simulated on a torus, which is seamless because it never had an edge. All three
  are held to a measured seam error, and the same number goes into the sidecar)*;
  next, a Wang set whose corner colours come from a quantised height field rather
  than at random, which is what would make it read as terrain instead of as
  patchwork; anisotropic tiles for a floor that is stretched along one axis; and a
  texture atlas.
- **export extras** -- SVG for vector UI, atlas packing, and an ~~`assets.yaml`
  registry carrying `source/license/attribution/seed` for every asset~~
  *(done, as `core/credits.py` + the sidecars: JSON rather than YAML, because a
  YAML parser is a dependency this project does not take)*. The registry also
  generates the credits block ~~and the screen~~ *(done: `credits.screen`, a
  scroll drawn with the built-in font)*, which matters because the Kevin MacLeod
  tracks are CC-BY and *require* attribution; next, use the same manifest to check
  a build before it ships.

## Licensing notes

* Code: MIT (see `LICENSE`).
* Generated PNGs: no third-party rights are involved -- keep the seed in the
  registry and the asset can always be rebuilt.
* Watch these three, they are the usual ways third-party rights leak in:
  **fonts** (use OFL/Apache, or draw glyphs procedurally -- Pillow's built-in
  bitmap font is the only one used here, and only for the credits screen and dev
  previews), **external textures/brushes** (don't), and **CC-BY music**
  (attribution is mandatory).

### Attribution

`core/credits.py` keeps the two halves apart on purpose. The half that can be
*forgotten* is hand-written and short -- a list of sources with the credit line
each licence asks for, stored as the string that has to appear on screen rather
than assembled at print time. The half that can be got *wrong* is generated: the
manifest, read out of the sidecars the exporters already write, so a new
generator is covered the day it writes one.

```bash
python cli.py credits --manifest out            # print the credits block
python cli.py credits --manifest out --out out  # ...or write out/credits.txt
python cli.py credits --manifest out --png out/credits.png   # ...or draw the screen
```

A text file is not what a game ships. `screen` draws the same block with Pillow's
built-in bitmap font and scales it up with nearest-neighbour -- the one font this
project is allowed to ship, and at least the right kind of ugly -- and the size
follows the text in both directions, because a credits list is a *scroll*: page it
and the one line the licence actually demands ends up on page four. What is fixed
instead is the line *length*, in characters rather than pixels, so the wrap cannot
move when the output size does. A credit line that fits at 960 and is clipped at
480 is a credit line that is not there, and the wrapping is therefore a function
that returns rows rather than pixels, so a test can read the words back out.

The manifest is worth trusting for the same kind of reason: a test asserts that
every sheet the demos write has a sidecar. Not that the entries are right -- that
none of them are missing.
