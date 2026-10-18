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

The demo writes `out/icons.png`, `out/icons-atlas.png`, `out/shapes.png`,
`out/texture.png`, `out/palette.png`, `out/arms.png`, `out/flags.png`,
`out/portraits.png`, `out/lineage.png`, `out/ui.png`, `out/maps.png`,
`out/tiles.png` and `out/factions.png`, each with a `.json` sidecar recording the
seed, the spec, the blazon, the genome of every portrait tile, the corner insets of
every UI part, for every map its biome census, its rivers, its roads, its realms and
its provinces, for every tiling texture its seam error, for the icon atlas the
rectangle each sprite occupies and how densely the sheet packed, and for the
factions the *names* they were derived from -- which, for that sheet, are the input
rather than the seed -- the coat each name decided and the livery it is painted in.
A map is written out twice, and deliberately: the sheet with its sidecar is the map
as a *picture*, and `out/maps.campaign.<seed>.json` is the map as **data** -- the
ground, the owners, the roads and the rivers as hexes, with the biome table in the
file, which is what a game without this generator loads. Then
`out/credits.txt` and `out/credits.png`, which are built from those sidecars. The
credits are not a seeded asset and carry no sidecar of their own: they are what
the sidecars are *for*.

## Usage

```bash
python cli.py demo    --out out --seed 7        # everything
python cli.py icons   --out out/icons.png --seed 3 --count 8 --size 192 \
                      --sizes 32,64,128 --cols 9   # set, seeds, mip sweep, atlas
python cli.py shapes  --out out/shapes.png      # SDF reference sheet
python cli.py texture --out out/texture.png --seed 3 --size 256
python cli.py palette --out out/palette.png --seed 3 --count 6
python cli.py arms    --out out/arms.png --seed 7 --count 12
python cli.py flags   --out out/flags.png --seed 7 --count 12 --ratio 3:2 [--flat]  # on their poles
python cli.py factions --name "Kingdom of Alder"       # a name, its coat and its livery
python cli.py blazon  --seed 5 --out out/one.png # text + render, both languages
python cli.py portrait   --seed 5 --size 384 --out out/one.png
python cli.py portraits  --seed 7 --count 8 --size 256
python cli.py lineage    --seed 7 --children 6  # two parents and their children
python cli.py ui         --seed 7 --count 3 --scale 2    # nine-slice UI kit sheet
python cli.py maps       --seed 7 --count 2 --cols 26 --rows 17 --sea 0.42 --provinces 8 --factions 4
python cli.py tiles      --seed 7 --size 256              # Truchet, Wang, reaction-diffusion
python cli.py credits    --manifest out         # what we owe, and what we made
python cli.py credits    --manifest out --png out/credits.png   # and the screen
python tests/test_core.py                       # 53 tests, no pytest needed
python tests/test_heraldry.py                   # 65 tests
python tests/test_portraits.py                  # 32 tests
python tests/test_ui.py                         # 32 tests
python tests/test_maps.py                       # 65 tests
python tests/test_textures.py                   # 23 tests
python tests/test_icons.py                      # 19 tests
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
| `export` | PNG writer, and `export.atlas`: trim a sprite to the pixels it actually uses, pack a set with a bottom-left skyline, and emit the rectangle table an engine cuts the sheet up with. |

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

## Icons (`gen/icons/`, `export/atlas.py`)

An icon is the asset family with the least room to be clever in. It is thirty-two
pixels of meaning that a player reads in a glance, sometimes while the rest of the
screen is moving, so the rules here are the opposite of the heraldry package's: no
detail that only survives at 512 px, no charge that needs a caption, and one
construction shared by every icon in the game, because a *set* is the thing that
gets looked at, not a single icon.

| Module | What it provides |
| --- | --- |
| `motifs` | The charge vocabulary -- nine symbols, each a distance field in a normalised box -- the four plate shapes, and `random_spec(seed, kind)`, which turns a seed into a spec without ever changing *which* charge was asked for. |
| `paint` | `draw(spec, size)`: the five passes. Plus `clamp`/`validate` on the spec, and the two measurements the tests hold an icon to, `silhouette` and `contrast`. |
| `export.atlas` | `trim`, `pack`, `build`: a set of canvases in, one sheet plus the rectangle table out, with the density of the packing measured rather than asserted. |

```python
from gen.icons import motifs, paint
from export import atlas

tile = paint.draw(motifs.canonical("grain"), 128)   # one icon, drawn at 128
sheet, info = atlas.build([("grain", tile), ("coin", paint.icon(7, 128))])
info["density"]                                      # 0.9.. how much of it is sprite
info["members"][0]["rect"]                           # [x, y, w, h] for the engine
```

### An icon is a plate with a charge on it

The plate is the ground -- a rounded square, a hexagon, a circle or a shield,
shaded as a solid -- and the charge is the meaning: a coin is a ring, water is a
drop, a blade is a sword seen flat. The plate is deliberately *not* part of the
seed's decisions about the symbol: a set shares one plate, and the seed may vary
the charge's proportions, its colour, and which of the four plates it sits on, but
never what the symbol *is*. That gives the package its central claim, and the test
that holds it: **a seed stretches the symbol and never replaces it**. The overlap
between a seeded charge and its canonical self is about 0.8 for most kinds, and the
worst case over ten seeds is 0.5 -- a gem whose seed happened to switch it from the
six-sided brilliant to the three-faced crystal, which is a change of category and
not of style, and the floor the test uses (0.4) is a floor on *recognition*.

The same measurement runs the other way, over the nine charges against each other.
The closest pair is water and a flame -- a drop, and a flame drawn with two tongues
-- which agree on about seven tenths of their area. That is fine, because they are
six hues apart, and hue is the other half of the answer. It is also exactly the
kind of collision a small vocabulary has to be *checked* for: the first version of
`ore` was a five-sided gem, and a five-sided gem sitting alone on a shield plate
*is* a shield. The vocabulary lost a symbol that day, not a pixel.

Below that, the plates themselves have to be comparable rather than equal -- a
hexagon that filled its box would be a square -- so the test asks that every plate
covers between three fifths and a little over what the round one covers. Measured
against the round plate: the hexagon 0.67, the circle 0.83, the shield 0.64. Every
charge on the same plate covers the same share of its tile to within a third of a
percent, which is what makes a row of these read as a *row* -- and that check is
made on rendered pixels, not on the spec, because it is a claim about what a player
sees.

### The vocabulary is symbols, not depictions

A sword, drawn honestly, is a thin line; at 32 px a thin line is a smudge. So the
charges are drawn as *symbols of* their objects rather than as small pictures of
them: a blade is a wide tapering bar with a heavy crossguard, not a sword with a
leather grip and a wrapped wire pommel. The rule of thumb here is the one the
earlier versions of `grain` and `wood` broke and the tests could not have caught --
a wheat ear whose grains are too small and whose stalk is too long is an *arrow*,
and two planks crossed at a right angle are a plus sign. Both were found by looking
at a contact sheet, which is the argument for generating one: a generator that
quietly draws the wrong symbol is exactly as deterministic as one that draws the
right one, and no test can tell you that a shape reads the way you hoped.

Hue carries the meaning that shape cannot. Each kind has a base hue -- ore is
violet, water is blue, flame is red, blade is a cold steel blue -- and the seed
jitters it by two hundredths of a turn, which is enough to make a set look grown
and not enough to make a player wonder what the icon meant.

### Five passes and one light

The construction is short enough to state completely, and every icon in the game
is drawn by it:

1. **shadow** -- the plate's own coverage, blurred and dropped, in world units so
   the shadow is the same shadow at 16 px and at 512;
2. **plate** -- the ground, shaded as a rounded solid, with a single bevel width;
3. **rim** -- a bright stroke inside the plate's edge, so the silhouette has a line
   to read against whatever the icon is placed on;
4. **recess** -- the ground knocked *back*, inside the plate and outside the charge
   grown by a clearance, which is what puts a dark halo around the symbol and is
   most of why a bright charge reads at 32 px;
5. **charge** -- the symbol itself, embossed with a shallower bevel than the
   plate's, because the symbol is small and a deep bevel eats it.

Every one of those passes is lit by the same `LIGHT = (-0.45, 0.65)` as the
heraldry, the portraits, the UI and the maps, and `tests/test_textures.py` asserts
it -- one light across the project is most of what makes individually generated
assets look like one art set rather than one folder.

Nothing in the recipe is in pixels. The margin, the bevel, the clearance, the rim
width and the shadow offset are all world units on the unit square, which is why
`--sizes 32,64,128` is worth generating rather than resizing: the small icon is
*drawn* small, with the same bevel and the same clearance, and the tests check that
the 256 px render folded back down *is* the 128 px render. And because the margin is
in world units, an icon never touches the edge of its own tile at any size -- which
matters more than it sounds, because at 16 px one world unit of margin is a single
pixel, and an icon that reached its own edge would bleed into its neighbour in an
atlas.

### An atlas moves sprites; it does not redraw them

The reason the atlas lives in `export/` and not in `gen/icons/` is that it knows
nothing about icons. It takes a list of `(name, canvas)` pairs, and it is the same
code for a texture set or a font sheet.

Two decisions in it are worth stating, because both are measurable rather than
stylistic:

* Members are **trimmed** before they are packed -- the transparent margin around a
  sprite is not part of a sprite -- and the trim is recorded, so the caller can put
  the origin back. Trimming counts even the faintest anti-aliased pixel, because an
  8-bit edge *is* content and a trim that rounds it away is a trim that clips.
* The layout is **deterministic**: members sorted by name, one bottom-left skyline
  pass, and a width chosen by trying a handful of candidates and keeping the one
  whose final sheet is squarest (with the smallest breaking a tie, because the
  squarest-and-smallest is what fits a texture limit and is worth looking at). The
  same members always produce the same atlas, byte for byte, from a different
  dictionary order if need be -- which is the property a content pipeline actually
  depends on, and the one thing a hand-arranged sprite sheet can never offer.

A skyline rather than a shelf, because a shelf wastes the whole band above a short
sprite, and a set of icons with one banner in it is exactly the case where that
shows. The result is that the strongest test in the package is one line long: a
sprite in an atlas is *moved*, not re-rendered, so every rectangle is compared
**byte for byte** with the source pixels it came from. A packing bug that softened
or shifted a sprite cannot survive that.

### Density is what power-of-two rounding costs

The atlas reports its own density -- the fraction of the sheet that is sprite
rather than gap -- and that number exists for one reason: it is how you find out
what a convention costs before you agree to it.

Measured on the icon set this package generates -- the nine charges and eight
seeded icons at 192 px -- the constraint-free packing fills 0.81 of its sheet, while
the same set rounded up to power-of-two dimensions fills about half of one (0.53).
Nothing was lost: the pixels are all there, but the texture is twice the size, and
the reason is not subtle. Each icon's trimmed sprite is 128 px across with two
pixels of padding, so four of them need 520 px, and a 512 px sheet is eight pixels
too narrow to hold them. Power-of-two dimensions are still the default, because that
is what an engine wants, but the density goes into the sidecar so the price is a
number in the build log instead of a rule nobody questions.

## Heraldry and flags (`gen/heraldry/`)

Heraldry is the best possible first customer for a procedural renderer, because it
is *already* a formal system: a coat of arms is a composition of a field, some
ordinaries and some charges, painted in tinctures that obey a rule. So the module
is a small compiler, not a drawing routine.

| Module | What it provides |
| --- | --- |
| `tinctures` | The tincture table (metals, colours, stains, furs) with the rule of tincture as `check(below, above)`, aliases ("gold" → `or`), and the livery: `random_shade` draws a faction's integer shade, `shade_table` resolves it to tinctures, `shaded` looks one name up through it. |
| `field` | Divisions (per pale/fess/bend/saltire/pall/chevron, quarterly, tierced) and variations (barry, paly, bendy, chevronny, checky, lozengy, gyronny, furs) as *partitions of the plane* into SDF regions, plus `tincture_at`/`field_layers`. |
| `ordinaries` | Chief, fess, pale, bend, chevron, cross, saltire, pall, pile, canton, bordure: builders, real proportions, `thin` scaling, fimbriation. |
| `shields` | Shield silhouettes (heater, iberian, french, rounded, oval, lozenge, banner) and flag shapes (rectangle, swallowtail, pennant, guidon) with their proportions. |
| `charges` | 29 charges built from SDF primitives in a normalised unit box — a lion is a set of capsules and discs, so it reads as a *generated* beast rather than a traced one. |
| `paint` | Flat and embossed tincture painting, rim light, drop shadow, all in linear light. |
| `blazon` | Composition, `validate`, EN/RU blazon text, `random_arms`, and `render(spec, size, shade=...)` — one extra argument, and the classical palette is what you get without it. |
| `flags` | Flag patterns (bicolour, tricolour, cross, Nordic cross, saltire, canton, hoist triangle, bordered), `describe`, `random_flag`, and the same `shade` argument as arms. A `pole` in the spec flies it. |
| `cloth` | Why a flag hangs instead of sitting: one height field, lit and projected. |
| `pole` | The shaft, its finial and the material it is made of; the room the flag's frame has to make for it, the anchor it puts on the cloth's hoist, and the droop that follows. |
| `crest` | A faction's whole identity from its *name*: the coat, the banner it flies, the livery it wears, the words for both, and `render`, which cannot be handed a coat without its colours. |

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
livery, both languages, and the validator's warnings, because a generated coat is
something a person has to be able to review in a sidecar rather than trust.

### The livery

Heraldry has seven classical tinctures and a game has more factions than that. Two
of them will have chosen Azure, and on a map they still have to read as two
different armies — so the same name that picks the coat also picks the *pigment*:

```python
from gen.heraldry import crest

crest.shade("Kingdom of Alder")      # {'or': {'l': -5, 'c': 0, 'h': 3}, ...}
crest.render("Kingdom of Alder", size=512)            # its shield, in its colours
crest.render("Kingdom of Alder", size=512, which="flag")
```

A shade is **three integers per tincture** — one each for lightness, chroma and hue
— and the integers matter. A continuous nudge would make "two shades apart" a
promise nobody could check; integers mean two factions either wear the same shade or
wear shades a step apart, and a test can hold that. The table is in the sidecar, so
a designer can *read* why a shield came out that colour, and it is drawn from the
name's own seed, so the livery cannot drift from the coat.

The budget is per class, because the classes have different room. Argent is 0.0025
chroma and Sable is 0.008: a hue on either is a hue on nothing, so a shade of a
metal is a *lightness* and a tint, while a colour's budget goes into a hue it can
actually lean with. And the budget is spent only where the colour has room to spend
it, which is the part that is measured rather than assumed. Azure at its own
lightness already sits on the edge of sRGB, so "more chromatic" is not a shade of it
— `fit_gamut` hands the chroma straight back and the pixel is identical. Left alone
that would put a number in the sidecar that never reached the screen, so each
direction is offered only if a single step along it buys at least 0.004 of OKLab
colour — a tenth of what the rest of this project calls a different tint — and a
direction below that is *closed* instead. The result is a step that is either
visible or absent, never a rounding artefact: over three hundred drawn liveries
there are zero recorded steps that change nothing.

The numbers, measured over forty imaginary factions: a shade moves its tincture by
0.08–0.12 in OKLab at most (about twice what the map package calls a different
tint, and well short of the 0.19 that separates Azure from Purpure), thirty-eight
or thirty-nine of the forty get their own Azure, and the closest two shades *inside
one coat* come to 0.095. The exception is the extreme: only sixteen of the forty
get their own Argent, because a white has nowhere to go, and the test says so out
loud rather than picking a bound low enough to cover the worst case.

Furs are shaded through their *bearings*. Ermine is Argent and Sable in a pattern,
so the table never mentions the fur — it shades the two ordinary tinctures inside
it, and two factions whose arms are both Ermine come out two different greys. What
a shade cannot do is change a tincture's *class*: Or is a metal before the shade and
a metal after it, so the rule of tincture is exactly as true of a livery as of the
classical palette, and the test checks the verdicts rather than trusting the
argument.

Nothing is shaded unless it is asked for. `blazon.render(spec)` and
`flags.render(spec)` still paint the seven classical tinctures, byte for byte as
before; a shade is an extra argument, or one call to `crest.render` for the caller
who would rather not be able to forget it.

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

### A flag is cloth, not paper

Everything else in `gen/heraldry/` paints flat, because heraldry is flat by
tradition -- a shield is a *sign*, not a surface. A flag is not a sign: it is a
rectangle of cloth under tension, and a flat flag reads as a picture *of* a flag
precisely because hanging cloth never stays in a plane.

One height field across the fly, `z(t)`, gives both halves of the effect:

* **The light.** A sinusoid has normals proportional to `(-dz/dt, 0, 1)`, so the
  Lambert term of *the same light the rest of the project uses* comes out as bands
  down the flag. Nothing is drawn: the folds are the lighting.
* **The projection.** Displacing the surface along its own normal moves it out of
  the screen, which under even a small oblique view shifts the *pattern* across the
  flag -- the cloth bunches where it comes forward and stretches where it dips. So
  the displacement is applied to the finished RGBA rather than to a mask, and the
  silhouette and the design cannot disagree about where the cloth is.

The amplitude grows towards the fly (`taper`), because the hoist is the edge tied
to the pole: a fold that starts at zero and grows is most of what makes a flag hang
*from* something rather than sag in mid-air. A second harmonic at an incommensurate
frequency is added because one sinusoid is a corrugation, and cloth is not
corrugated.

`taper` is a fraction of the **cloth's** extent, which sounds like a detail and is
not. The frame a flag is drawn in carries margin for the cloth to move into, so a
fold measured against the *frame* would start a tenth of the way up the flag: "the
amplitude at the hoist" would mean the amplitude at the hoist plus a margin, and the
anchor a pole puts there would not be an anchor. So the folds are measured against
the cloth, and the renderer -- which drew it, and therefore knows -- says where the
cloth's box is; a caller that does not say gets the drawn extent, which is the same
answer to within an edge pixel.

The parameter that decides whether it works is the **wavelength**, and it is
shorter than instinct suggests. What the light reads is the *slope* of the fold,
`amplitude * 2*pi / wavelength`; with a wavelength of half the flag the slope is
under 10 degrees and the result is a gradient, not a fold.

Two properties are design decisions rather than details. The modulation is centred
on the flag's own mean illumination, so a fold **redistributes** light instead of
taking it away and a hanging flag is exactly as bright as a flat one -- which is
both true of cloth and the only way a contact sheet of them can be read side by
side. And a flag that is going to move is drawn a little smaller (`cloth.reach`),
because the edge it would otherwise wave off is the fly, which is the one the eye
is watching.

Like everything else here it is just a **spec**: `spec["wave"]` is a dict of
numbers, so it round-trips through a sidecar, and `blazon` still knows nothing
about any of it. It is also opt-in, so a faction's banner stays flat and a crest
cannot pick up folds by accident; `cli.py flags` hangs its flags by default, and
`--flat` is the special case.

The tests are the ways to get this wrong. A wave of **zero amplitude must be the
identity** bit for bit -- the sampler resamples at fractional pixel coordinates, so
a wrong pixel-centre convention shifts the whole cloth by half a pixel, invisibly.
The bands must run **across** the folds' axis, measured on a plain field so that
the only structure in the profile is the light. The same spec must have the **same
folds at 96 px and at 384 px**, which is what fails if the wave is evaluated in
pixel coordinates instead of world ones -- the renderer's field is a pixel grid, so
this is the easy mistake -- and the comparison is read onto a *world* axis, because
a box that lands on a different fractional pixel at every size would otherwise
report a phase drift as a different fold. An **anchored** cloth must be still at
the hoist at any phase, and the hoist it is measured against is the cloth's. A
**droop** must move the free end down in both fold directions, which is the one
thing a fold may not do -- and it must leave the anchored end alone. And the cloth
must **stay on the canvas** at every proportion and silhouette.

### A pole is why a flag hangs from somewhere

`cloth` makes a flag hang; a pole is what it hangs *from*, and it decides three
things, only the first of which is about the pole.

* **The room.** A pole rises above the cloth and runs below it, so the frame the
  flag is drawn in has to grow. The room is asked for in the flag's *shape*
  (`shields.flag(..., room=...)`), which grows the frame's half-height and leaves
  the cloth's own coordinates alone, so one field function serves a flag on a pole
  and a flag without one. The few hundredths of a unit by which the shaft overhangs
  the hoist are left to the frame's margin, which is several shafts wide.
* **The anchor.** Cloth tied to a pole cannot move where it is tied, so
  `pole.anchor` sets the wave's `taper` to **zero** -- exact, not small: the hoist
  is still at any phase. That claim only means anything because the folds are
  measured against the cloth. A faction's banner takes its wind from the *name*
  (the banner's own seed stream), so flying the same faction twice flies the same
  folds.
* **The droop.** A cloth free at the fly hangs lower there, and gravity does not
  consult the folds about which way they run: the droop is the same envelope taken
  downwards, one term in `cloth.drift`, always down and taking no part in the
  lighting. A `taper` of zero means the hoist does not droop either, so an anchored
  flag is still at the pole and trailing at the fly.

The pole is drawn from its own vocabulary, and it is **not heraldic**: it bears no
tincture and obeys no rule, so it is wood, iron or gilt with a ball, a spearhead or
nothing on top. A finial or material this module does not know is refused rather
than repaired -- the same line the icon vocabulary draws.

The shading is a cylinder and a sphere, not a bevel, because that is what those two
objects *are*. The shaft's normal turns through `(sin t, 0, cos t)` across its
width, so it is brightest where it faces the light and dark at both edges; a bevel,
which saturates in the middle and tilts at the rim, would be brightest *at* the rim
instead. The test switches the rim light off for the measurement -- it darkens
exactly the pixels a bevel would have made brightest -- checks that the highlight is
strictly inside the shaft, then hands the painter a mirrored light and checks that
the highlight follows it, which is what says the shading came from the normals at
all.

The pole is the one rigid thing on a banner's canvas, so the wave must not move it.
It is therefore composited **after** the wave, and the test renders two phases and
requires the band below the lowest pixel the cloth can reach to be identical -- and
non-empty, or the comparison would prove nothing. It is also drawn *over* the cloth
rather than tucked under the hoist, and that is geometry rather than taste: the
plane of the cloth passes through the shaft's axis, so every column of the shaft's
image is nearer the camera than the plane it cuts, and the shaft wins the whole
overlap. A pole is opt-in -- `random_flag(..., pole=...)`, `crest.flag(...,
pole=...)`, `crest.render(..., which="flag", pole=True)` -- so a crest used as a
sign stays flat, and `cli.py flags --flat` is the special case on the other side.

## Portraits (`gen/portraits/`)

A face is the hardest thing here to get away with, because the eye is unforgiving
in one direction only: a blank icon can be wrong and still useful, a face that is
wrong is *uncanny*. The way through is to lean into the medium instead of away from
it -- draw an anime face, where form is carried by line art, shading is a handful
of flat tones, and "simple" is a style rather than a compromise.

| Module | What it provides |
| --- | --- |
| `style` | Four presets (chibi, shoujo, shounen, seinen) in one parameter table, with the face's vertical layout *derived* from the eye position rather than typed in beside it. |
| `turn` | The head as a solid: a half-profile and a depth, the cross-sections they imply, and the rotation that turns a front view into a three-quarter. |
| `head` | Skull silhouette, neck, trapezius, collar and hairline: control points → closed Catmull-Rom loop → distance field. |
| `shape` | The vocabulary a face is written in -- `lock` (a variable-width tapered stroke), `lens`, `arc`, `poly`, `smooth_loop` -- over `core.sdf`. |
| `brush` | Painting a *field* rather than a rectangle: `Surface` binds a shape to its window and its coverage, so a fill, a band or a stroke is one pass over the pixels that can see it. |
| `ink` | Contour strokes that live on the field: `abs(d + inset) - w/2` is a band along the zero set, so an outline costs no new geometry. |
| `cel` | The quantised Lambert term, the two colour rules, and the dome that stands in for a skull's height field. |
| `hues` | The whole colour script -- about thirty tones -- derived from four base colours. |
| `eyes`, `face`, `hair`, `expression` | The features themselves. |
| `dna` | The genome: `random_genes`, `encode`/`decode`, `breed`, `mutate`, `apply_to`, `describe`. |
| `portrait` | Framing, layer order, `compose`/`render`/`render_seed`. |

### Five decisions, in the order they matter

1. **Line art carries the form.** Every silhouette is inked along its own contour,
   with a taper that thins the stroke where the form turns away from the viewer.
   Shading is secondary: if the ink is right, the face reads at 64 px.
2. **Cel shading is a hybrid height.** A cheap dome gives the normals a real head
   would, and a couple of pixels of bevel off the silhouette give the rim. The
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

### A turn is a rotation, not a squeeze

For a long time `turn` was an ellipsoid inside `head`: one surface, one depth, plus
a constant squeeze on "whichever side was receding". It slid the nose correctly at
the eye line and lied everywhere else. The depth came out of the same ellipsoid in
*y* as in *x*, so the chin -- half a head below the equator -- had almost none, and
a jaw that does not move when the head turns is the difference between a portrait
and a mask. The squeeze was worse than an approximation: an ellipsoid's outline does
not change width when it rotates, so the squeeze was a drawing convention dressed up
as geometry. And it was applied to the side with a negative `x`, which for a
positive yaw is the side the head turns *towards*.

`turn` now says what a head is: a stack of horizontal cross-sections, one per
height, each an ellipse of half-width `rx(y)` -- the same profile the silhouette is
drawn from -- and half-depth `rz(y)`, centred a little in front of the **ear axis**,
the vertical line a head actually pivots about. Two conventions fix the rest. The
face plane is further from that axis than the back of the skull is (`FRONT_SHARE`),
which is what tilts a three-quarter outline; and the depth tapers towards the crown
and the chin (`DOME_Y`, `DOME_SPAN`), so the two ends of a face turn *less* than
the middle of it. A style adds exactly one number, `head_d`, its half-depth as a
multiple of its own half-width -- and the width gene nudges it, because a wider head
is a deeper one.

Everything else is a caller, and each caller asks for the thing it is:

* **a feature** is a point on the surface, so its depth is the section's at its own
  height and width: the middle of the face has the most and slides furthest, a
  temple point barely moves at all, and the jaw swings because it has a depth of its
  own;
* **the outline** is the *support* of a section in the view direction --
  `sqrt((rx cos t)^2 + (rz sin t)^2)`, edged by the ear axis's own offset. A round
  section keeps the same width at every angle, which is why a chibi's turned
  silhouette is its front-on one to a per cent, and a deep one widens towards
  profile. A test samples both faces of each section and holds the silhouette the
  renderer draws against the projection the features use;
* **an eye** is a chord, not a point: both its corners are placed and nothing else
  is scaled. It narrows by exactly what the surface foreshortens and keeps its
  height, because a plane turning away compresses it sideways and not downwards.
  The far eye narrows hard and the near one widens a little -- which is what the
  geometry says, and, unlike a constant squeeze, what a drawing does;
* **the hair and the ears** ride a shell a hair's breadth out from the surface, so a
  lock's root follows the head instead of sitting where the cap used to be.

The side a head turns away from is *derived* rather than declared. It is the side
the nose is pointing at, which is the opposite of what it looks like, and getting it
backwards is invisible at a small angle and plain at a large one -- so it lives in
one method (`receding`) that the eyes, the ears and the side locks all ask, and a
test ties it to the depth it claims to describe. The lighting dome is still a
stand-in for a skull's surface, but a turn moves one thing the light can read --
where the surface comes nearest the camera -- so the stand-in's peak goes to the
model's answer for it, and the terminator swings with the turn instead of staying
where the front view put it.

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

The turn has seven tests of its own, and they are all the same shape: an invariant
of a solid that a 2-D approximation cannot have. A turn of zero is the identity *bit
for bit*, so the front view is not an approximation of the model. The outline a
renderer draws is the projection of the body's cross-sections, sampled densely. The
chin and the crown swing less than the eye line. `receding` agrees with the depth it
claims to describe. An eye's corners move and its height does not, and the far eye
is narrower than the front view while the near one is wider. A turned head neither
collapses nor explodes -- its width stays within fifteen per cent of its front-on
one -- and nothing anywhere in the model is infinite, including at heights where the
profile has no width left.

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
| `style` | The numbers: a palette plus proportions in **design pixels**, one per side where a title bar needs it, the named kits (`parchment`, `slate`, `brass`, `moss`, `glass`, `timber`, `granite`), `title_bar`, a seeded `random_style`, and the geometric invariant the rest of the package leans on. |
| `frame` | The geometry as distance fields: concentric rounded rectangles, the band between them, the inner rule, and a corner ornament whose radius is a Fourier series. |
| `slice` | The contract itself — the insets, the nine rectangles, the resample a game engine will perform, and the measurement of what that resample costs. |
| `paint` | Five passes over one fixed light: shadow, body, band, rule, ornament. |
| `material` | Wood and stone, out of fBm and Worley noise in design pixels — the body's surface, and the one look here that gives up a stretchable middle on purpose. |

### The one hard geometric rule

```
every bit of curvature in the design must fit inside the corner box,
and the design must be constant across whatever axis gets stretched.
```

A nine-slice resample keeps the corners and stretches the four edges and the
middle. A straight edge survives that; a curve does not — it would smear. So the
corner radius, the bevel, the frame band, the rule and the ornament all have to
live within the corner box, and the style may vary *along* the stretch axis but
never across it (which is why the body gradient is a function of `y` and of
nothing else).

The rule is in that list because it is design, not body: a rule that crossed out
of the corner box would lie in the strip that gets stretched sideways and break in
two at the first resize, which is a flaw the eye finds immediately and the algebra
does not mention.

`style.validate` states that, `style.clamp` enforces it, and the test checks the
*pixels* rather than the algebra: outside the corner boxes, every column of the
left strip has to be identical down the panel and every row of the top strip
identical across it. That one assertion catches an ornament that has grown out of
its corner, a rule that curves too far, and a drop shadow that is not separable,
all at once.

A **material body** is the one deliberate exception, and it is not let off: the
resize it costs is drawn twice and measured, and the machine-readable answer goes
in the sidecar (`stretchable`). See *Nine-slice and procedural texture are two
answers to the same problem*, below.

### The corner box is a rectangle, not a square

The insets and the band widths take **one number per side** —
`(left, top, right, bottom)` — because a window with a title bar is not a second
kind of window: it is the same window whose top band is wide enough to write in.
`style.title_bar(spec, height)` is the whole API for it: it raises the top band and
the top corner box together, since the two numbers have to move as a pair.

Everything that is *not* per side stays shared. There is one radius, one bevel and
one ornament size for the kit, so the number that bounds them is the smallest of
the four sides — a bar that is thin on one axis limits the ornament on all four
corners — and the ornament keeps its single radius by measuring itself against the
thinner of the two bands meeting at each corner.

What this changes downstream is exactly one thing, and it is the thing an engine
cares about: the nine-slice is still four numbers, but the top-left block is no
longer a square. Stretching a titled window sideways leaves that block *and* the
strip below it untouched, which is what the tests measure, block by block, against
a source image.

### `inset` is derived, not requested

The style states a floor, and `clamp` widens the corner box to whatever the look
actually needs — `radius + bevel`, the frame plus its rule, or the ornament's
reach. The nine-slice cut can therefore never be too small to hold the design, by
construction rather than by check. `size` runs the argument the other way: at a
size with no room for the corner box the *look* gives way instead, keeping the
radius (which is the shape) and flattening the bevel (which is only the shading)
first, and thinning the rule rather than letting it spill.

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

### Nine-slice and procedural texture are two answers to the same problem

A texture generator's job is to make a surface that never repeats visibly; a
nine-slice's job is to make one image survive every size. They are in tension,
and it is worth being explicit about where the line is: a nine-slice can only
stretch a middle that is **constant along the stretch axis**, so a body that
varies in both directions — which is what a real material does — cannot be
stretched, only drawn at the size it is shown.

`material` takes the second road and pays for it honestly. Wood is a stack of
rings whose phase is domain-warped (without the warp, rings on a rectangle are
stripes, and a stripe is not wood); stone is Worley plates with `worley_edges`
carved into the mortar between them. Both are evaluated in design pixels, so
`scale=2` resolves the grain rather than enlarging it, and both are lit by the
same one light, centred on the material's own mean — a material redistributes
the light across a surface rather than dimming the panel, exactly as the cloth
module does for a flag.

The cost is then **measured rather than declared**, which is the part worth
keeping:

> `slice.stretch_error` draws the asset twice — once at the size it is shown,
> once small and then stretched to that size by the nine-slice rule — and
> reports the difference as a fraction of the panel's own range.

A gradient body scores 0.002–0.005 (the resampling fuzz of a ramp and nothing
else); `timber` scores 0.049 and `granite` 0.037, because the grain in the strips
changes size while the grain in the corners beside them does not. `paint` only
evaluates a material on the inner rectangle, since the band over it is opaque and
a material never touches a frame pixel (asserted), and the sidecar carries both
numbers per kit:

```json
"panel": {"design_size": [240, 150], "stretch_error": 0.049, "stretchable": false}
```

So an engine can ask what it needs to know — *may I resize this?* — and get an
answer that was computed from the pixels rather than a flag the style set about
itself. A subtle material can genuinely come out stretchable; a bold one cannot.
Both are correct, and both are the same code path.

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
| `politics` | Who holds the provinces: realms drawn against their own coats so that two of them are a colour apart, grown over the province graph so that each is one block, and the colour each province wears -- a faction's tincture taken to a map's wash, with the livery's own step spent rather than clipped. |
| `roads` | The roads between the capitals: Dijkstra over the lattice on the biome table's own movement costs, water impassable, and the cheapest set of roads that joins every capital on a landmass. |
| `campaign` | The map as a file rather than as a picture: the biome table, one biome index per hex, the owner of each province, and the hexes the roads and rivers walk -- everything a client that does not have this generator needs to run the map. |
| `paint` | One raster pass for the ground and its relief, a wash and a frontier line for the provinces, strokes for the rivers, the roads, the coast and the frame. |

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

### A province is not just owned, it is worn

A province answers "where does this end". A realm answers a different question --
"who" -- and a map that answers it in flat colours is no longer a map of the
ground. So the realms arrive as a second sheet of meaning over the same geography,
and the provinces do not move by a pixel when they do.

Ownership is grown exactly the way provinces are, and for the same reason: give
each province to the realm with the nearest capital and you get realms in two
pieces, because a bay does not care how close two capitals are. So the realms are
seeds on the *province* graph and grow by breadth-first search, which makes each
realm connected by construction -- and the seeds are spread by graph distance
rather than by straight line, because two capitals that look close across a
mountain range are not two places to start from. Two landmasses never touch, so
the realms are shared out between them *before* they are shared out by size: a
realm cannot be grown onto an island from across the water, and an island given no
realm would be a hole in the map rather than a small realm.

The colour is where a map meets the heraldry, and it is one function. A faction's
arms decide a tincture and a livery; the map takes the tincture's hue -- Azure on
a map has to still be the blue the player calls Azure -- normalises its chroma to
a wash, so a metal lends no hue and wears a grey, which is the truth about Or and
Argent and the reason the two metals are told apart by lightness everywhere else
in this project; and spends the livery's lightness step. That last one is the
interesting one, and it is the whole reason the module exists:

> A livery's step is one whole number of OKLab, because that is what a step has to
> be to show on a shield. At map size it goes on at half opacity over terrain that
> is busy with relief and ink, so the same step arrives at a third of its size and
> vanishes.

So the map does not pass the step through, it **spends** it -- stretched, and
passed through a saturating map rather than a clip, because a clip would push two
different liveries onto the same colour and lose exactly the distinction the band
was there to protect. The constant follows from the claim rather than being picked:

```python
WASH_FILL = 0.5
WASH_STEP = 0.04 / WASH_FILL     # "a different tint", at the opacity it goes on at
```

Two realms can still draw the same tincture *and* the same livery -- seven
tinctures and more factions than that -- so a **roster** is drawn against the
colour and not merely sliced off the name stream: a name is taken only if its realm
is `WASH_STEP` away from every realm already on the map. The test does not check
the table, it renders the same map twice with the two closest realms swapped and
measures the pixels at a 13 px hex; everything else in the two pictures is
identical, so every difference between them is the colour of the realm.

One colour is already spent: the sea's. A realm wearing the shallows has no
coastline, so a chromatic wash is kept away from every colour the water is drawn
in -- and a wash with no chroma keeps out of the way by itself, which is why a
metal is never refused for being near the colour the sea is nearest to. Borders
inside a realm are damped for the same reason a realm exists: a realm drawn as one
block is a realm a player can see, and the provinces are still there underneath,
faintly, which is what a province is on a map of realms.

### A road is a claim about the ground

"You can march from Aldmere to Dunfell in four days" is a statement about the
*map*, and the picture of it is a line. So the roads are routed before they are
drawn: Dijkstra over the lattice, one hex step at a time, and the cost of a cell is
the cost the biome table already gives it -- the same number a game charges an army
to move there. Which decides what the roads look like, and it is worth saying out
loud:

> The road is the cheapest walk by the map's own rules, not a civil engineer's
> road. Hills cost three a hex and plains cost one, so a road will cross a narrow
> range rather than go round a wide valley -- and that is the answer the movement
> costs give, not a bug in the routing. A road that disagreed with them would be a
> road an army could not follow.

Water is *impassable* rather than expensive, and that difference is not pedantry: a
price is a price a search will pay, and a road across an ocean is exactly what a
very expensive ocean buys. Infinity is not a price. The network is then a
**minimum spanning tree** over the capitals, because a spanning tree is the cheapest
set of roads that joins them all and every other road is a loop somebody would
build and never need; crossing without meeting is not a thing on a hex map, so the
tree is grown over the hex graph on distances that are already path costs. Since
landmasses do not touch, what comes out is a spanning *forest* -- one tree per
landmass, never a bridge.

Two of those decisions are measured rather than asserted. The route is checked
against Bellman-Ford, which is the definition of a shortest path and knows nothing
about heaps -- a search that agrees with it is right for the right reason. And
because a hex centre to a hex centre is a *spline* once it is drawn, the road is
sampled along the curve it actually is: a Catmull-Rom bend bulges sideways, and on
a coast that would put the road in the water. The pixels are checked too, since a
stroke is a hex wide -- the same map is drawn with the roads off and on, and every
pixel that changed has to be a pixel of a land cell.

### A map is data, and the picture is one of its outputs

Everything above this line is a *picture*: a raster pass, washes, strokes, and a
sidecar that says what was drawn for a reader who can see the drawing. A game cannot
load a sidecar accurately enough to run a border dispute off it. So each map is also
written as a **campaign file** -- `maps.campaign.<seed>.json` -- and it is the map,
not a photograph of one:

* **which hex is what ground**, one biome index per cell, with the biome table in
  the file as well: the name, the movement cost, whether it is water, and the
  colour. The table travels because the table *is* the rules -- and because a client
  that had to guess a colour for "marsh" would put a different green on the same
  province;
* **whose hex it is**, one realm per province and each realm's own colour, so the
  political map can be redrawn without re-deriving anybody's livery;
* **where the roads go**, as the hexes they walk, and what they cost to walk --
  because the road is a claim about the ground, and a client has to be able to check
  that claim against the same numbers the generator checked it against.

Cells are named the way the generator names them, `r * cols + c`, row by row, so a
client indexes the same arrays this project indexes. And nothing in the file is a
pixel:

> A polyline at 26 px is a different polyline at 9 px, and the map is supposed to be
> the same map. So the roads and the rivers go into the file as **hexes**, and the
> hex size does not go in at all.

That last decision has a test of its own -- the same map exported at 9 px and at
26 px has to produce the same bytes -- and it is the same rule the terrain has kept
from the beginning, one layer up: the geography is a function of `(seed, cols,
rows)`, and everything derived from it inherits that. A 26x17 map is 442 hexes,
eight provinces and a file of about 9 kB, which is a size a player can download one
of per map without noticing.

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

- **icons** -- ~~resource/unit icons, atlas packing, `--size` sweep for mip levels~~
  *(done: `gen/icons/`, where an icon is a plate with a charge on it and a set
  shares one plate; the charges are symbols rather than depictions because a sword
  drawn honestly is a smudge at 32 px; the five passes, the margin and the bevel are
  world units, so the 32 px icon is *drawn* small rather than resized down; and
  `export/atlas.py` trims, packs and reports the density of the sheet, which is how
  the cost of power-of-two dimensions became a number in the sidecar instead of a
  convention nobody questioned)*; next, quantities and tiers drawn as a badge on the
  plate, and a faction hue table so a faction's icons are its own colours -- the
  charge already owns a hue, so what is missing is a second source of hue that does
  not fight the first.
- **heraldry** -- ~~shields, field divisions, charges, tincture rules~~, a
  ~~faction's crest derived from its name as the seed~~, and ~~faction shades of the
  tinctures in a sidecar~~ *(done: `gen/heraldry/`, with `crest.py` deriving the seed
  from a hash of the name rather than Python's salted `hash()`, deriving the banner
  from the coat so the two cannot disagree, and deriving a **livery** from the same
  name -- three integers per tincture, spent only where the colour has room, so two
  factions that both chose Azure are two Azures and a shade is either a visible step
  or no step at all)*; next, meeting heraldry on the map: faction colours in the map
  sidecar, so a province is not just owned but *worn*, which is the first place the
  livery has to survive being scaled down to a hex.
- **flags** -- ~~bands by projection onto directions, overlaps by half-plane
  clipping~~, ~~cloth folds as sinusoidal normal displacement~~ and ~~a pole and a fly
  that trails~~ *(done: `gen/heraldry/flags.py`, `cloth.py` and `pole.py`, where one
  height field supplies both the lighting and the projection, in world coordinates
  so that the folds are the same folds at any resolution; a taper of zero is an
  exact anchor at the hoist and a droop is gravity taken downwards, so a pennant has
  a direction to hang from and now something to hang from; and the pole is a
  cylinder and a sphere shaded by the one light rather than a bevel)*; next, a
  gonfalon -- a banner hung from a crossbar rather than a mast, which is a second
  *hanging* rather than a second pole.
- **portraits** -- ~~SDF head, ink contours, cel shading, traceable genome,
  heredity~~ *(done: `gen/portraits/`)*; ~~outfits built from the same shoulder
  line~~ *(done: a collar stroke, three kinds, one per seed)*; ~~turn the head in
  depth~~ *(done: `gen/portraits/turn.py` -- the head is a stack of cross-sections,
  so the outline is a support rather than a squeeze, an eye is a chord on the
  surface, the jaw swings because it has a depth of its own, and the side a head
  turns away from is derived rather than declared)*; next, expressions that move the
  whole face rather than three features.
- **ui** -- ~~9-slice frames and panels, ornamental borders from a Fourier series
  of a radius function~~, ~~asymmetric insets for title bars~~ and ~~wood/stone
  material bodies via fBm + domain warping~~ *(done: `gen/ui/`, with the series
  moved into the corner box, which is the only part of a nine-slice asset that is
  never resampled; the insets and bands taking one number per side so that a title
  bar is a wider border rather than a second kind of window; and the one look that
  gives up a stretchable middle, priced by `slice.stretch_error` and carried in the
  sidecar as a measured `stretchable` rather than a promise)*; next, a material
  that is cheap enough to draw live -- the recipes are two noise fields per panel
  at the moment, which is fine for a shipped sheet and too slow for a UI that
  redraws on hover, and the obvious fix is a small cache keyed by
  `(kind, scale, seed)` that the paint pass asks first.
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
  are filled the gradient is the answer and nothing is left to search)*; ~~faction
  colours in the sidecar so that a province is not just owned but worn~~ *(done:
  `gen/maps/politics.py` -- realms drawn against their own coats until they are
  `WASH_STEP` apart and grown over the province graph so each is one block, and a
  wash per province that keeps the livery's hue, class and lightness step while
  spending it to survive being laid over terrain at half opacity; `WASH_STEP` is
  `0.04 / WASH_FILL` rather than a constant, and the claim is measured on the
  pixels of a 13 px hex rather than on the table)*; ~~roads between the
  capitals~~ *(done: `gen/maps/roads.py` -- Dijkstra over the lattice on the biome
  table's own movement costs, water impassable rather than expensive, and the
  cheapest set that joins every capital on a landmass as a spanning forest; the
  route is checked against Bellman-Ford and the drawn spline, pixel by pixel,
  against the water)*; ~~a campaign-friendly export of the sidecar~~ *(done:
  `gen/maps/campaign.py` -- the map as data, one file per map, with the biome table
  in the file, one biome index per hex, the owner of every province and the hexes
  the roads and rivers walk; nothing in it is a pixel, so the same map exports the
  same bytes at every scale it can be drawn at)*; next, sea provinces -- the water is
  drawn and kept out of the political map, but nothing owns it yet -- and a naval
  layer for the campaign file to carry them in.
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
- **export extras** -- SVG for vector UI, ~~atlas packing~~ *(done, as
  `export/atlas.py` and shared with the icons sheet)*, and an ~~`assets.yaml`
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
