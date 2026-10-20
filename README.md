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
`out/anim.png`, `out/figures.png`, `out/portraits.png`, `out/lineage.png`, `out/ui.png`, `out/maps.png`,
`out/coast-tiles.png`, `out/borders.png`, `out/tiles.png`, `out/factions.png`, `out/text.png`, a sheet per mip level as
`out/tiles-mip0.png` through `out/tiles-mip7.png`, and an SVG per UI kit as
`out/ui.<style>.svg` -- each with a `.json`
sidecar recording the
seed, the spec, the blazon, the genome of every portrait tile, the corner insets of
every UI part, for the typeface every character's advance and ink box per face, for
every map its biome census, its rivers, its roads, its realms, its
provinces and the sea provinces with the realms that hold them plus the province
names it set into the picture and the size of each, for the coast and
biome-border tiles the grid they sit in and how many of the sixty-four each map
wears of each, for every tiling
texture its seam error and how many tiles its tiling used, for a ground material the
row of the table that made it, the biomes that wear it and its seam, for the atlases the
rectangle each sprite or tile occupies and how densely the sheet packed -- per level,
for a mip chain -- for the
icons the tiers and counts it badged and the factions it dressed a set in, for an
animation the rectangle of every frame with the phase it was drawn at and the rate it
plays at, for a figure's sheet the cycle, the table and the figure itself -- its
proportions, its gait and the five colours it wears -- and for the
factions the *names* they were derived from -- which, for that sheet, are the input
rather than the seed -- the coat each name decided and the livery it is painted in.
A map is written out twice, and deliberately: the sheet with its sidecar is the map
as a *picture*, and `out/maps.campaign.<seed>.json` is the map as **data** -- the
ground, the owners, the roads, the rivers and the sea as hexes, with the biome table
in the file, which is what a game without this generator loads. Then
`out/credits.txt` and `out/credits.png`, which are built from those sidecars, and
last `out/index.html`, the build as one page: every sheet with its sidecar's numbers,
its digest and the command that rebuilds it. The
credits are not a seeded asset and carry no sidecar of their own: they are what
the sidecars are *for*.

## Usage

```bash
python cli.py demo    --out out --seed 7        # everything
python cli.py icons   --out out/icons.png --seed 3 --count 8 --size 192 \
                      --sizes 32,64,128 --cols 9   # set, seed, badges, faction plate, atlas
python cli.py shapes  --out out/shapes.png      # SDF reference sheet
python cli.py texture --out out/texture.png --seed 3 --size 256
python cli.py palette --out out/palette.png --seed 3 --count 6
python cli.py arms    --out out/arms.png --seed 7 --count 12
python cli.py flags   --out out/flags.png --seed 7 --count 12 --ratio 3:2 [--hang bar]  # & staffs
python cli.py anim    --out out/anim.png  --seed 7 --count 12 --fps 12   # one flag waving: frames + timeline
python cli.py figures --out out/figures.png --seed 7 --cycle walk --count 8  # one figure walking: frames + timeline
python cli.py factions --name "Kingdom of Alder"       # a name, its coat and its livery
python cli.py blazon  --seed 5 --out out/one.png # text + render, both languages
python cli.py portrait   --seed 5 --size 384 --out out/one.png
python cli.py portraits  --seed 7 --count 8 --size 256
python cli.py lineage    --seed 7 --children 6  # two parents and their children
python cli.py ui         --seed 7 --count 3 --scale 2    # nine-slice UI kit sheet
python cli.py ui         --seed 7 --count 3 --scale 2 --svg   # ...and each kit as SVG
python cli.py maps       --seed 7 --count 2 --cols 26 --rows 17 --sea 0.42 --provinces 8 --factions 4
python cli.py maps       --seed 7 --no-names              # ...without the province names
python cli.py tiles      --seed 7 --size 256              # Truchet, Wang, reaction, boards
python cli.py tiles      --seed 7 --size 256 --mips 4     # ...and the mip chain, packed per level
python cli.py grounds    --seed 7 --size 256              # six ground materials for map tiles, + atlas
python cli.py text       --size 20 --out out/text.png  # the typeface: a specimen, its metrics
python cli.py credits    --manifest out         # what we owe, and what we made
python cli.py credits    --manifest out --png out/credits.png   # and the screen
python cli.py credits    --manifest out --check # is this build shippable?  (exit 0/1)
python cli.py gallery    --manifest out         # the build as one page (add --embed to send it)
python tests/test_core.py                       # 68 tests, no pytest needed
python tests/test_heraldry.py                   # 80 tests
python tests/test_portraits.py                  # 37 tests
python tests/test_ui.py                         # 47 tests
python tests/test_maps.py                       # 97 tests
python tests/test_textures.py                   # 62 tests
python tests/test_icons.py                      # 33 tests
python tests/test_text.py                       # 53 tests
python tests/test_figures.py                    # 26 tests
python tests/test_gallery.py                    # 20 tests
```

Every one of those runs on every push as well -- the ten suites, the demo built
twice and compared byte for byte, and `credits --check` -- on Python 3.10 and 3.13,
in `.github/workflows/tests.yml`.

## Core toolkit (`core/`)

| Module | What it provides |
| --- | --- |
| `mat` | Affine transforms in homogeneous coordinates: translate/rotate/scale/shear, composition, inversion, `uniform_scale` for metric correctness. |
| `sdf` | Distance-field primitives (disc, rounded box, capsule, ring, arbitrary polygon, regular polygon, star) and operators: hard/smooth union, intersection, difference, stroke (`outline`), offset, plus `transformed`/`moved`/`rotated`/`scaled`/`mirrored`/`repeated`. |
| `noise` | Deterministic hash-based value noise, fBm, ridged/turbulence variants, domain warping, Worley cells (F1) and Worley cell walls (F2−F1). |
| `palette` | sRGB ↔ OKLab/OKLCH, gamut fitting by chroma bisection, perceptually even mixing and ramps, harmony schemes, and farthest-point `distinct_palette`. |
| `canvas` | Linear-light RGBA buffer with a world-space camera (`scale`, `dx`), correct source-over compositing, sRGB conversion only on export, and one export rule: a pixel whose alpha rounds to zero carries no colour. |
| `render` | Distance field → pixels: anti-aliased `coverage`, bevel lighting from the field's gradient (raised or, with `inset=True`, the same ramp turned over into a recess), separable blur, drop shadows, contact sheets. |
| `credits` | The two halves of an attribution: the hand-written list of sources we do not own (with the credit line each licence demands) and the manifest read out of the sidecars every exporter writes -- and the credits screen, drawn from both in `gen/text`'s own face. `check` reads them back as a pre-ship gate. |
| `export` | PNG writer, `export.atlas`: trim a sprite to the pixels it actually uses, pack a set with a bottom-left skyline, and emit the rectangle table an engine cuts the sheet up with; and `export.sheet`: the frames of *one* sprite laid out over time, with the timeline they play on. |

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

4. **A pixel the file calls invisible carries no colour.** Alpha is *coverage*, so
   the export rule is total: where the byte is `0`, the colour is `0` too. A
   blurred shadow leaves its own colour at an alpha of a millionth over a wide
   field of pixels; the eye cannot see any of it, and a *filter* can -- a bilinear
   nine-slice stretch of a frame over content, a naive downsample, a tool that
   builds a mipmap without premultiplying -- which is the straight-alpha halo this
   project premultiplies to avoid inside the cloth. `Canvas.to_uint8` zeroes those
   pixels, and leaves every pixel the file calls visible exactly the plain
   quantisation of what it was.

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
| `paint` | `draw(spec, size)`: the five passes, plus a badge when the spec carries one. Plus `clamp`/`validate` on the spec, and the two measurements the tests hold an icon to, `silhouette` and `contrast`. |
| `badge` | A **count** or a **tier** on the plate's corner: `make(count=12)` draws a numeral, `make(tier=3)` a row of pips, and `clamp` refuses a badge it cannot draw rather than rounding one. |
| `livery` | A faction's **own colours**, worn on the plate: `hue`/`chroma`/`light_step` read from the coat, and `tint(spec, name)` dresses a spec without ever touching the charge. |
| `export.atlas` | `trim`, `pack`, `build`: a set of canvases in, one sheet plus the rectangle table out, with the density of the packing measured rather than asserted. |

```python
from gen.icons import badge, livery, motifs, paint
from export import atlas

tile = paint.draw(motifs.canonical("grain"), 128)   # one icon, drawn at 128
stack = paint.draw({**motifs.canonical("coin"), "badge": badge.make(count=12)}, 64)
alder = paint.draw(livery.tint(motifs.canonical("coin"), "Kingdom of Alder"), 64)
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

There is a sixth pass, the **badge**, and it is not in this list because it is not
always there: it is drawn only when a spec carries one, on top of the other five,
and it is the one part of an icon allowed outside the plate's silhouette. It has its
own section below, because the reason for it is a question the construction so far
cannot answer.

Every one of those passes is lit by the same `LIGHT = (-0.45, 0.65)` as the
heraldry, the portraits, the UI, the maps and the type, and `tests/test_textures.py`
asserts it -- one light across the project is most of what makes individually
generated assets look like one art set rather than one folder.

Nothing in the recipe is in pixels. The margin, the bevel, the clearance, the rim
width and the shadow offset are all world units on the unit square, which is why
`--sizes 32,64,128` is worth generating rather than resizing: the small icon is
*drawn* small, with the same bevel and the same clearance, and the tests check that
the 256 px render folded back down *is* the 128 px render. And because the margin is
in world units, an icon never touches the edge of its own tile at any size -- which
matters more than it sounds, because at 16 px one world unit of margin is a single
pixel, and an icon that reached its own edge would bleed into its neighbour in an
atlas.

### A charge cannot say how many

An icon's charge says *what* a thing is, and there is nothing in a coin that can say
*twelve*. That is a second question, and it gets a second, smaller answer: a
**badge** pinned to the plate's lower-right corner, which is the plate-and-charge
construction one level down -- a small disc with a small mark on it, drawn in the
icon's own colours, so it adds no hue to the set and no fifth thing for a player to
learn.

The two questions a badge answers are not the same question, and they are not drawn
the same way. A **quantity** is a count and a count can be ninety-nine, so it is a
*numeral* -- seven bars per digit, in the arrangement a calculator uses, because a
digit made of seven bars is a symbol for a numeral the way a ring is a symbol for a
coin, and it survives a small size where a serif does not. A **tier** is an ordinal
rank and ranks are few, so it is a row of **pips**: a "2" in a corner is ambiguous
the moment the icon also carries a quantity of two, and two dots under it are not.
The tests hold a pip to what a pip is -- the number of separate blobs in the mark
*is* the tier, counted from one to five, so a second row that quietly merged two of
them would fail -- and hold every numeral to being its own picture, two digits
included.

The badge is *data*, not design: a charge is chosen by the seed, a count is chosen by
the game, so `random_spec` never fills a badge in and a spec with `badge=None`
renders byte-for-byte as it did before badges existed. What a badge may not do is
lie: a count larger than the badge can hold is refused rather than shown as "99" --
the same line `export.atlas` draws over an oversized sprite -- and a count of zero is
not a badge showing nothing, it is no badge. The honest limit is stated rather than
hidden: at 32 px a two-digit numeral is a *mark* and not a number, and the badge is
built to read at the sizes a count is actually consulted at.

### A faction's icons are its own colours

An icon is read by colour before it is read by shape, and until there is a faction
that colour belongs to the *kind*: every coin the same amber, in every kingdom. A
faction needs a second colour, and what makes that work is giving it somewhere to go
that does not fight the first. The charge keeps the kind's hue -- water is the colour
of water in every faction -- and the **plate** takes the faction's: the ground, the
rim and the recess are the large surface, and a large surface is where a faction
belongs. Two hues, then, one the ground and one the figure, which is exactly how a
coat of arms is arranged and why it does not read as two colours arguing.

The faction's colour is not a new kind of number. It is the first tincture of the
faction's field, worn at the livery's own shade -- the same
`gen.heraldry.crest.field_tincture` the map washes its provinces with, so a faction's
icons and its provinces cannot drift apart, and the test asserts that equality rather
than trusting it. The plate's chroma is proportional to the tincture's, so a colour
gives a coloured ground and a **metal gives a grey one**: Or is a gold plate, Argent a
grey, and two metals are told apart by lightness -- the livery's light step, gained
and squashed through the same saturating map the map uses -- which is exactly how Or
and Argent are told apart everywhere else in this project. The charge's hue, its
chroma and its lightness are left *exactly* as they were, so a faction never makes
water stop being blue, and one faction wears one plate colour across the whole set
whatever charge is on it.

### An atlas moves sprites; it does not redraw them

The reason the atlas lives in `export/` and not in `gen/icons/` is that it knows
nothing about icons. It takes a list of `(name, canvas)` pairs, and it is the same
*layout* for a texture set or a font sheet. What a texture set needs on top of the
layout is a different question, and it has its own module and its own section below
-- the honest summary being that a tile breaks both of this atlas's rules.

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
| `flags` | Flag patterns (bicolour, tricolour, cross, Nordic cross, saltire, canton, hoist triangle, bordered), `describe`, `random_flag`, and the same `shade` argument as arms. A `pole` in the spec flies it and a `bar` hangs it. `waving` returns the frames and the phases of one flag over a cycle. |
| `cloth` | Why a flag hangs instead of sitting: one height field, lit and projected -- and the two ways to hold it, at the side of its folds (`hang`) or across the top of them (`drape`). `period` is the phase span in which the fold pattern returns, which is what an animation has to cut its loop on. |
| `pole` | The shaft, its finial and the material it is made of; the room the flag's frame has to make for it, the anchor it puts on the cloth's hoist, and the droop that follows. A crossbar (`bar`) is the same rod turned a quarter turn. |
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

### A gonfalon is the same cloth, hung the other way

A pole holds the cloth at an edge that runs **beside** its folds: the hoist is
parallel to every fold line, so the fold's own coordinate is also the coordinate
that says how far the cloth is from what is holding it, and one `t` serves both.
Hang the same cloth from a **bar** and the holding edge runs **across** the folds --
and that quarter turn is the whole of the difference, which is why `cloth.drape` is
a sibling of `cloth.hang` rather than a second cloth module.

What changes is not the physics but which coordinate the physics is read from. The
folds still run down the cloth and still vary across it; what decides how deep they
are is now the distance *down* from the bar, so `surface` and `drift` take an
`envelope` to read the freedom at when it is not the same coordinate as the fold's
profile. The droop needs the same treatment for a subtler reason: `drop` is the span
the cloth hangs through, because a droop of a tenth is a tenth *of what the cloth
has to fall*. A tenth of a wide flag and a tenth of a long one are not the same
number of pixels, and only the span the cloth actually hangs through can say which
is which.

The other consequence is what the droop **is**. A flag trails: the free edge slips
sideways-down and a pennant hangs point down. A gonfalon cannot trail, because the
bar does not move, so the same slip becomes a **stretch** down the cloth -- the hem
hangs lower than the cloth was cut -- and gravity's contribution is length rather
than a trail. This is the one claim in the increment that a number can settle, and
it is settled on a plain field, where the only structure in a row is the light: the
contrast across a row of a pole-held flag is **constant** to within a couple of
percent from the hoist to the fly, and across a row of a gonfalon it nearly doubles
from the bar to the hem. The same wind, the same field, the same `t` -- and the
only thing between them is which edge is holding the cloth.

A bar is a **shaft turned a quarter turn**, and the proof is that it is shaded by
the same function: `pole.bar_paint` and `pole.paint` both end in one `_rod` that
applies the one light and the one rim, because a frame carrying a mast and a
crossbar lit by two pieces of code is the one frame where the difference would be
visible. What a bar does *not* have is a finial -- its ends are turned towards the
shoulders rather than the sky, and the same rim finishes them -- and it has no
anchor or droop of its own, because those are the *cloth's* statements about where
it is held, and `pole.anchor` serves whichever thing is doing the holding.

Its room is the part that looks like it should be complicated and is not. A pole is
tall, so it asks the flag's *shape* for room; a bar is wide, and asks for none,
because the frame is scaled by the wider of the cloth's two dimensions and drawn
with a margin of several percent of it -- so a bar that reaches a few hundredths of
a unit past the cloth's sides is already inside the margin the flag is drawn with.
That is measured rather than assumed: five proportions, four silhouettes and two
sizes, and the border stays empty. Its `over` is the sleeve: a banner has a hand's
width of cloth above its field, the rod passes through *that*, and the default --
just under half the rod's own thickness -- is the rod's underside meeting the cloth's
top edge and biting a hair into it, because a rod that only touches leaves a
background hairline at some sizes and a rod whose axis sits on the edge crops a rod's
width off the top of the artwork. An `over` of a few hundredths is the same bar hung
on rings above the cloth, which is a different object rather than a wrong one.

A pole and a bar may be asked for together, and that is not a contradiction but the
classic processional gonfalon: a staff with a crossbar across the top of it and the
banner hanging from the crossbar. Each asks for its own headroom, the frame takes
the larger of the two, and the bar is composited after the staff because it is
mounted in front of it. A bar on something that is not a cloth is refused, on the
same line the pole and the icon vocabulary draw.

`random_flag(..., bar=...)`, `crest.flag(..., bar=...)` and
`crest.render(..., which="flag", bar=True)` are the entry points, and `cli.py flags
--hang bar|staff|mixed` is the sheet. `mixed` walks the three hangings in turn,
which is what `cli.py demo` does with its flags: the three are one cloth, and
putting them beside each other is the only way to see that the hanging is all that
differs. The **description** does not change with the mounting -- "Azure, a cross
Or" is what the cloth says, and whether it is on a pole or a bar is not something a
blazon has a word for -- which is the same reason `crest.faction` carries neither.

### An animation is a flag sampled along the axis it already had

The last increment is the one that needed no new physics, and that is the point of it.
Everything else here is a function of a seed; a waving flag is a function of a seed
**and a phase**, because cloth that moves had to be a field in `t` before it could be a
picture at all. So an animation is not a set of drawings of a flag -- it is *one* call
sampled: `flags.waving(spec, count=12)` returns twelve frames and the twelve phases they
were drawn at, and frame *i* is `flags.render` with one number different. Nothing is
interpolated, nothing is tweened, and no frame is a resample of another: an engine that
plays the sheet plays twelve real drawings, and a reviewer who doubts the fourth one can
regenerate that exact frame from the sidecar.

That changes what the delivery has to be. A `flags.png` sheet is for a person and has
captions; `export.sheet.build` writes the other thing -- a grid of equal cells with no
captions, clear padding between frames, and a table saying where each frame is with the
phase it was drawn at, plus the frame rate and whether it loops. Two decisions are
behind the format rather than inside it. **A frame is a cell, not a trim**: an atlas
trims each member to its own pixels because sprites are different shapes, and the frames
of one animation are one size by construction -- so unequal frames are *refused*, with
the refusal naming the atlas that can pack them, because an engine that has to re-centre
every frame is an engine with a character that rolls. And **the rectangle is the
layout**: the table comes from `render.cells`, the same function `render.sheet` draws
on, so the picture and the table cannot disagree about a pixel -- and the test reads the
pixels back rather than re-deriving the formula. The convention is `export.atlas`'s, top
left with y down, so one cutter reads both kinds of sheet.

The loop is where the cloth had something to say, and it said it in a number. A flag's
fold is a fundamental plus a second harmonic, and `_HARMONIC_RATIO` is deliberately not
a whole number -- the pattern must not repeat along the flag -- so the two terms return
together only after **ten turns**, which is what `cloth.period` reports: 20π radians.
Animating one turn of *that* cloth gives a loop with a seam, because the harmonic comes
back 0.7 of a cycle out. So the demo asks for a cloth with one harmonic, whose period is
one turn, and the loop then closes *exactly*: the test renders the animation at phase
`p` and again at `p + 2π`, in two independent calls, and requires the bytes to match.
What is true of both cloths is that the sampling is uniform -- one turn divided by the
frame count, at every step including the wrap -- which is why a seam is a step like any
other step and never a special frame. And because the frames are samples of a continuous
field, a finer sampling is *smaller steps* rather than a different motion: the test
measures the mean change per step at four frames and at sixteen, and the finer one is
about a quarter of the coarse one -- four times the frames, a quarter of the step.

The cost is the cheapest thing in the repository: a dozen frames of a 200-pixel flag, no
simulation and no cached state. `python cli.py anim --count 12 --fps 12` writes
`out/anim.png` and its sidecar, and the `demo` does too -- `out/anim.png`, 12 frames in
a 4×3 grid, transparent between cells, with the pole identical in every frame and only
the cloth moving. The sidecar is what `credits` reads, so a sheet an engine plays is a
sheet the licence gate can account for.

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
| `eyes`, `face`, `hair`, `expression` | The features themselves, and the warp that moves the whole face for a mood. |
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

### An expression is a warp, not three features

For a long time an expression was eight small offsets: the brows move as units and at
their inner ends, the lid apexes open or narrow, the lower lid lifts, the mouth curves
and opens. Every one of those is a *feature*, and the sum of them is not a face. Eight
per-feature offsets move three features: the eyes and the nose stay exactly where they
were, and the result reads as a mask worn over a face rather than a face wearing an
expression. A smile is not a curved mouth; it is a face that *rises*.

The way out was already in the code. The vertical layout is not a list of positions to
move one at a time -- `style` exists precisely so that the brow, nose, mouth and blush
heights are *derived* from `eye_y` and four gaps, because that relationship is what
makes the eye read as a person rather than a mask. So the expression gets a **warp** of
those five numbers and lets the derivation do the rest:

```python
fst = expression.warp(st, expr)   # eye_y and four gaps moved; everything else follows
```

Two numbers spend the warp. `face_dy` lifts or drops the whole face -- surprise and a
grin rise, sadness and anger sink -- and `face_spread` opens or closes it, because a
surprised face puts its brows *further* from its mouth and an angry one pulls them
together. One number moves every landmark, which is the test of whether a number
belongs in the warp: anything a single feature owns is a delta, not a warp. The eight
deltas survive on top, doing what a warp cannot -- though several of them shrank, and
surprise's `brow_dy` shrank a lot, because the warp was already carrying the lift and
spending it twice is how the peak of a chibi's brow ends up in its hairline.

What is **not** warped is any *horizontal* distance. The distance between the eyes, the
width of the skull, the height of the chin are anatomy, and a mood does not change a
skull. A wide grin is the *mouth* getting wider (`mouth_w_s`, a delta), not the face
spreading out. An earlier attempt had a `face_wide` number that pushed the eyes apart,
and it was wrong in a way a picture would not have shown: `dna._clamp_params` sizes the
skull from the eye positions, so widening the eyes by a proportion quietly walked them
out through the silhouette that had just been fitted to them.

That same argument splits the portrait in two. `portrait.compose` now builds the head,
the hair, the ears and the bust from the style it always did, and builds *the face* --
nose, mouth, blush, eyes, brows -- from the warped copy. A mood moves a face, not a
person: nothing anatomical is allowed to move. The hairline is the one thing the
derivation would otherwise drag along, so the warp writes it back through
`Style.pinned`, which exists for exactly this -- a value that is an *anchor on the
skull* rather than a conclusion about the face, and that `with_` would silently undo.

A neutral expression is the identity **object**: `expression.warp(st, neutral())` returns
`st` itself, so a portrait that is not in a mood is the portrait that existed before
there was a warp, to the byte. That is checked against a `git worktree` of the previous
commit, not just against itself.

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

The warp has five tests of its own, and one of them is the whole reason it exists:
that **every** expression moves **every** landmark -- the eyes and the nose included,
which no per-feature offset ever moved -- measured across four styles and thirty
genomes, so an expression that moves only its own feature fails there instead of
looking subtly wrong in a picture. The others hold down the two halves of the claim
that a mood moves a face and not a person: a neutral row returns the style *itself*
(`is`, not `==`), and the skull, the chin, the hairline, the silhouette and the turned
solid are bit-identical under every expression. A bound test walks the cross product
of styles, genomes and expressions and asserts the face stays inside its own head --
the brow's peak below the hairline where the fringe roots, the brow above the lid,
the landmarks in order, the mouth clear of the chin, the eye inside the skull -- with
the margin *recorded in the test* rather than left implicit, so a later change that
eats it says so.

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

## Figures (`gen/figures/`)

A figure is a body, a gait and a brush, and those are three modules because they
answer three different questions in that order. `rig` says where the joints are for a
given *pose*; `cycle` says what the pose is for a given *phase*; `body` says what the
body is made of; and `figure` is the only place the seed is opened, because "the same
figure" has to be defined exactly once.

| Module | What it provides |
| --- | --- |
| `rig` | Seventeen joints in a tree, a rest pose that holds the proportions, forward kinematics as one pass down a tuple, and a seeded jitter that makes one figure leggy and another stocky. |
| `cycle` | `idle`, `walk`, `run` and `attack` as periodic functions of phase, each a dict of *deltas* from the rest pose, plus the pelvis's rise -- and every one of them closes exactly at `2*pi`. |
| `body` | The drawing: limbs as tapered locks, a tunic, a belt, hose, boots and hair, from the portrait package's own brush, ink and cel pass, with the far side drawn first and shaded. |
| `figure` | `render(seed, cycle, phase, size)`, `frames(...)` for a whole cycle and `describe(seed)` for the table a sidecar carries. |

### A bone is named after what it is

The first version of `rig.py` named every bone after the joint it *reached* -- the
bone up from the pelvis was called `chest`, and the bone down to the knee was called
`knee`. That is a tidy convention, and it produced a walk cycle that said
`hip_near` where it meant the thigh. There are two hip bones in that scheme: the
two-hundredths-of-a-figure bone from the pelvis to the hip socket, and the thigh.
`hip_near` named the first one, so a stride of thirty-eight hundredths of a radian
was applied to a bone 0.02 long, and the figure walked by rotating its own hip
sockets by a quarter of a degree. Nothing crashed. The picture was a person standing
still with a slight lean.

A bone is now named for what it *is* -- `thigh_near`, `shin_far`, `upper_arm_near` --
and the convention is the sentence: a gait is written about thighs and shins, so a
skeleton that has them cannot make that mistake. The joints are still named after
themselves, and the two dicts never meet: `pose()` speaks bone names, `points()`
answers joint names.

The rest pose is where the proportions live. A figure is **one unit tall**, every
bone is a fraction of that, and `STAND` is written down as a number of its own so a
test can check that the three bones below the hip add up to it -- a hip is where its
legs say it is, and not where a drawing happened to put it. The angles are
**absolute**: a pose says where a bone points in the world, and the kinematics
accumulates positions down the tree rather than rotations. A cycle then speaks in
**deltas** from the rest pose, which is the unit a gait is actually written in --
"the thigh swings twenty degrees about straight down" is a sentence about walking,
while "the thigh's world angle is 0.35 rad" is a sentence about trigonometry.

### A gait is a periodic function of phase

The flag had to fight for its loop. Its cloth carries a second harmonic that is not
commensurate with the first, so the fold pattern returns after ten turns rather than
one, and a one-turn loop of that cloth has a seam (`cloth.period` is `20π`). A gait
has no such trouble, and that is a decision rather than luck: every term in
`cycle.py` is a sine, a cosine, or a **window of one**, so every term is exactly
`2*pi`-periodic and the loop closes by construction. The test does not measure
whether it closes; it draws phase `0` and phase `2*pi` independently and compares
the bytes.

The window is the piece that makes a periodic cycle able to *do* something once per
turn -- bend a knee, lift a foot, land a sword -- without leaving the periodic world:

```
bump(p, centre, width) = 1/2 * (1 + cos(pi * clamp((p - centre) / width, -1, 1)))
```

A raised cosine and not a Gaussian, because it reaches zero **exactly** at the edge
of its window. A term that only approached zero would leave a step where the window
ended, which is a visible tick once per loop. A walk is then four lines of it: each
thigh is one sine and its negation, each shin bends most halfway through its swing,
each arm swings with the leg on the far side of the body, and the pelvis bobs twice
per cycle because the legs carry it. That contralateral relation is the whole of why
a walk does not read as a shuffle, so it is asserted as a relation between two angles
and not as a picture someone approved.

The four cycles differ in *shape* and not in degree. `run` is not `walk` with bigger
numbers: it has a lean, bent elbows carried high, a flight phase, and a bigger bob.
`attack` is not periodic in its nature -- a blow is a one-shot -- so it is written as
a wind-up, a strike and a recovery **inside one turn**, which is how a game plays it
once and then falls back into the idle. The windows overlap on purpose: a swing that
teleports to its end is a swing nobody believes.

### A figure is drawn in side view, so it has a near side and a far side

There is no left and right in a profile, and calling the two limbs `l` and `r` would
be a lie now and a bug later, when a front view -- which does have a left -- arrives
and looks like a rename rather than a second view. So the bones are `near` and
`far`. Depth in a flat drawing is two things and neither of them is a shadow map: the
far limb is drawn *first*, so the near one crosses it, and it wears a shadow of its
own material rather than its material. The draw order is the depth order and it is
fixed: a far limb is never in front of a near one *whatever the pose*, because that
is what "far" means.

Nothing else about the drawing is new, and that is the point. A limb is a
`shape.lock` -- the same tapered spine a portrait draws a hair clump with; the outline
is `ink.stroke`, so a figure's contour is heavy where the limb is thick and thins
towards the hand exactly as it does down a jaw; the shadow is `cel.shade_shape`, so a
figure and a portrait wear the same quantised Lambert band under the **one** light,
`core.render.LIGHT`. Consistency here is not a shared palette; it is one stroke and
one light with a second caller.

What a figure does not get is a face. At 96 px an eye is two pixels and a mouth is
one, and drawing them honestly is how a sprite becomes a smudge -- the argument
`gen/icons` already makes about a sword at 32 px. So the whole of a figure is
**silhouette, costume and colour**: a skull, a hairstyle, a tunic, a belt, hose,
boots, and five colours. The tunic takes its colour from outside when a faction hands
one over -- `crest.field_tincture`, the same place the map's wash and the icon's plate
read -- so a figure stands in the colour of the realm it belongs to, and a figure that
belongs to nobody wears what its seed drew.

### The line belongs to the raster, the body belongs to the figure

Everything about the *body* is a fraction of the figure's height: the widths, the
head, the skirt, the margin. Those are properties of the drawing and they scale with
it, which is why a 192 px figure is the same figure as a 48 px one and not a resample
of it.

The **line and the shading bevel are in pixels**. A line belongs to the raster: at
32 px an outline that scaled with the body would be a hairline, and at 256 px the same
rule would give a slab. Keeping `LINE_PX = 1.7` is what makes the small sprite a small
soldier rather than a smudge with a border, and the test says so as a range rather
than as a pixel count, because 1.7 px is drawn by the anti-aliasing and not by the
edge.

The cell is sized by the gait and not by the rest pose. A figure occupies 82% of its
cell, which leaves nine percent above and below -- and that margin is not taste: an
attack *dips*, and a dip that was not budgeted for is a boot in the next cell of the
sheet. The test walks all four cycles at twelve phases and checks that the sides and
the top are clear and that the floor is clear of ink, with one written-down exception:
the contact shadow may graze the bottom row as a faint pixel, because a shadow is a
decal on the ground and not part of the figure.

A frame is a **cell and not a trim**: every frame is the same size and the figure is
centred in it, so the frame that the figure barely enters has the same rectangle as
the frame it fills. `export/sheet.py` is the second caller it was written for -- it
had one, the waving flag -- and it gets the same contract: equal cells, clear padding,
a table taken from the same `render.cells` the picture is drawn on, the frame rate, and
a loop flag that is true because the cycle closes.

### What is not here

One view, facing `+x`. No weapons or props (a sword is a second skeleton's worth of
decisions about where a hand is), no cloth or ragdoll physics, and no blending between
cycles: the cycles are separate functions, so a game cuts from `walk` to `attack`
rather than cross-fading, and the wind-up inside `attack` is what makes the cut read.
A front view, a back view and a left-facing mirror are the next thing this package
would grow -- the mirror is free, the other two are not, because the near and far
limbs stop being a depth cue and become an anatomy.

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
| `material` | Wood and stone, out of fBm and Worley noise in design pixels — the body's surface, and the one look here that gives up a stretchable middle on purpose. The lit field is remembered per recipe and window, so a state change costs a multiply. |
| `vector` | The same panel as curves, in SVG: the silhouette, the hole, the rule and the four ornaments, built from the same `Panel` the raster thresholds. The light is not in it and a material is refused — the two differences that are stated instead of faked. |

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

### A material is remembered, and the key is not `(kind, scale, seed)`

That measurement is what makes the material *honest*; what makes it affordable is
that it turns out not to depend on the colour it is mixed into. `body` is really
two halves, and only the last one touches the palette:

```python
factor, mix = material.modulation(kind, spec, X, Y, dx, LIGHT)   # the noise
body = clip(colour * factor + stain * mix)                       # the multiply
```

The first half — the lit relief as a ratio of its own mean, and the stain's blend
weight — is a pure function of the recipe, the four numbers, the light and the
*pixel grid*. So it is computed once and remembered, and since **a button state is
a colour and never a layout**, the three states of a button are one material:
`material.stats()` reads `{'hits': 2, 'misses': 1}` after `paint.button` is called
three times, and that is a test rather than a hope.

The measured difference, on one machine: a 400×250 window costs 528 ms (wood) or
675 ms (stone) to compute and 11–14 ms to re-apply — **38× to 69× cheaper** — and a
512×320 button drawn in all three of its states drops from 4.4 s to 2.4 s. That
last number is worth reading properly: the material was about half of the draw and
the shadow, the moulding band, the rule and the four bosses are the other half, so
this buys the *redraw* — hover, press, a resize — and not the first draw. Nothing
in this package made the first draw faster, and nothing pretends to.

The obvious key is `(kind, scale, seed)` and it is wrong. A material is anchored to
the **world**, not to the panel — that is the whole reason `stretch_error` exists —
so a wider panel samples a wider *window* of the same wood. Answering a 400-px panel
with a 200-px panel's field would slide the grain by half a panel, and a cache that
is wrong is worse than no cache. So the key is the complete input: the recipe, its
four numbers, `dx`, `y_up`, the light, and the grid (`shape` and the two axes'
origins and far ends — a canvas grid is a separable meshgrid, so that is exact).

The cache is transparent and bounded. Transparent means a hit is bit-identical to a
miss, which is checked by drawing every style at three sizes in three states with a
cold cache and with a warm one and comparing the pictures, not the arrays. Bounded
means `CACHE_MAX = 16`, an ordered map evicted from the front: a cache with no bound
is a leak wearing a cache's name. A kit of mock windows peaks at 12 live entries, so
16 is a bound that the thing it bounds never reaches.

### The same panel as curves, and what a vector cannot carry

Every other asset in this project is a picture. A panel is the one that is also a
*shape*: four rounded rectangles and a polar ornament, and a rounded rectangle is
four lines and four arcs. So a panel has an exact vector form, and it is worth
being precise about why it is safe to ship one — a second renderer of the same
picture is exactly the thing that drifts.

> The paths are built from the same `frame.Panel` the raster pass thresholds.

That is the whole guarantee, and it is checkable: the raster's silhouette *is*
`coverage(panel.outer)`, so the outer curve has to be the zero set of
`panel.outer`, and likewise for the hole, the rule's centreline and each ornament.
`tests/test_ui.py` walks every style, five sizes and three scales, and asks the
fields about every point the curves were built from: **225,536 points, worst
6.3e-14**. Not "close" — the same number, in a different notation.

A curve comes back with two point lists, because they answer different questions.
`vertices` are the eight points the `d` attribute literally contains, which is how
the flip to SVG's y-down convention is checked: each one, flipped, must be a token
in the path. `on_curve` adds samples *along* each arc, because an arc that bulged
the wrong way still joins the same two vertices — a sweep flag is invisible to the
vertices and obvious to the samples. And the flip is a real piece of geometry, not
bookkeeping: the ornament's radius is a Fourier series with a phase, so it is *not*
symmetric under negation, and the test asserts that too (an un-mirrored scallop
misses its field by 146 pixels, i.e. off the panel).

Two things a vector cannot carry, and both are said out loud rather than faked:

* **the light is not in it.** The moulding's bevel, the ornament's dome and the
  drop shadow are what `render.shade` does to a distance field, and a bevel is a
  ramp, not a curve. What is in the file is the geometry and the palette's colours,
  which is what an engine that draws its own chrome wants anyway — and it is why a
  `pressed` button in SVG differs from an idle one in nothing but its paint. The
  raster only *nearly* obeys "a state is a colour, never a layout" (pressing also
  turns the bevel round); the vector is that rule stated exactly.
* **a material has no vector form at all.** Wood and stone are noise fields, so a
  style with one is refused — `ValueError`, naming the material and pointing at the
  PNG — rather than exported as a flat rectangle. An export that silently drops the
  one thing it cannot draw is worse than an error, because it ships.

One approximation survives, and it is measured rather than asserted away. SVG
interpolates between gradient stops in sRGB and `core.palette` interpolates in
OKLab, so the body ramp is written as **sampled stops**: `STOPS = 17`, which across
every style and state on the shelf disagrees with the true ramp by at most
**1.4e-5 in OKLab** at the worst point between two stops. The test allows a
hundred times that.

What the export *buys* is the opposite of what the nine-slice buys. A nine-slice is
what you do when the asset is pixels; a vector needs no cuts and no insets at all,
because the curves are exact at every size. The document's user units are design
pixels and its width and height are output pixels, so the same design drawn at 100
pixels and at 200 is *the same paths* with two numbers changed — a test compares
the two files and finds every `d` attribute identical. One file serves every DPI,
and there is no mip chain to carry.

Building this turned up a real wart worth recording. `sdf.box` does not clamp its
corner radius, and `clamp` keeps a four-pixel floor for a design's corner box, so a
panel small enough to be narrower than its own frame used to round *past* its box:
the corner arcs crossed over each other, the field stopped being a distance field,
and the "rounded rectangle" had no exact curve and no exact bevel either. The three
radii are now fitted to the box they round (`frame.radius_fit`, one definition that
the fields and the vector both read), and a panel with an inside-out inner box
simply has no hole — which is what the raster's `max` was already drawing, and what
`Panel.has_hole` now says out loud.

The vector has eleven tests of its own, and they are mostly the one above wearing
different hats: every curve on its field, the flip in the coordinates, the ornament
un-mirrored, the band an even-odd ring with the middle empty (and a frame with no
body layer at all), a material refused by name, the ramp bounded, the three states
sharing every `d`, one file across two DPIs, a byte-identical rewrite, and the
writer's numbers compact and never `-0`.

## Text (`gen/text/`)

Every other package here is a painter of *pictures*. This one is a painter of
something that has obligations a picture does not: a **font**.

The licensing notes called fonts the one asset family that leaks. You cannot draw
your way out of that leak with a better picture, because a font is not a picture of
letters -- it is a *measurement* of them. An engine asks a font where a letter ends
so it knows where the next one begins, how tall a line is so it knows where the next
one goes, and which characters exist so it knows what to refuse. Ship a glyph and a
guess and the guess is what breaks.

So this package generates the alphabet the same way the rest of the toolkit generates
everything: a few numbers, deterministically, in world units, with the shape and the
style kept apart. It is not a font *file*. It is a function from `(text, face, size)`
to pixels, and a sidecar of metrics.

| Module | What it provides |
| --- | --- |
| `skeleton` | The alphabet as data: one polyline per stroke in **tenths of an em**, drawn on the four guides, with arcs given as ellipses rather than as polygons. 95 characters of printable ASCII, 66 of Cyrillic and the 53 accented letters of ISO-8859-1 — 214 in all, in three tables: drawn, **aliased** (А is A) and **composed** (é is e and an acute). Plus the five **joins** (ﬀ ﬁ ﬂ ﬃ ﬄ), which are glyphs and not characters: composed from the letters they replace, never drawn. Also `advance` and `advance`'s refusal. |
| `pen` | What turns a skeleton into a letter: **weight**, **contrast** (a broad nib is thin on a horizontal), **serif**, **slant** and **width**, the serif rule, the marks and where they go (`MARK_GAP`, a mark fitted into its letter's own advance), and — measured off the result rather than declared beside it — `ink_box`, `advance` and `offset`. Five presets: `sans`, `text`, `display`, `italic`, `slab`. |
| `kern` | Optical spacing from the pen's own profiles: how far a pair may close before it spends the daylight it is promised, how much white it carries over the face's **reference pair** (two `n`s, measured in the same pen), and the twelfth-of-an-em clamp that stops a rule from guessing. Cached by `(character, pen)`. |
| `typeset` | `measure`, `place`, `ink_bounds`, `pair`, `paint`, `draw`, `block`: advances, **kerns** and tracking between letters, the **join substitution** (`glyphs`, with `spec["liga"]`), alignment on the *ink*, and the drawing, which is one distance field per glyph unioned over the line. |

### A letter is a skeleton, and the letter is what a pen does to it

The split is the whole package, and it is the same split the icons make between a
charge and a plate:

* the **skeleton** says *which* letter this is — one or more polylines, in em units,
  drawn on the guides every face agrees about;
* the **pen** says how thick, how contrasted, how serifed, how slanted and how wide.

Nothing about style is in the alphabet file. A stem and a hairline are the *same*
skeleton read by a pen that changes width with the direction of the stroke; a serif
is a rule about stroke *ends*; a slant is a shear applied to the points before the
pen sees them. So one alphabet is a monoline UI face and a modulated book face, and
the difference between them is four numbers — which is why there is no new gene and
no new file, and why the five presets in the demo are one alphabet printed five
times.

The alphabet is written in **tenths of an em** and the grammar is three tokens:

```
"a": (5.1, "A2.0,2.3,2.0,2.3,0,360 | 4.0,0 4.0,5 A2.3,3.6,1.7,1.4,0,180"),
```

— a body width in tenths, then strokes split by `|`; `x,y` is a point; `Acx,cy,rx,ry,a0,a1`
is an elliptical arc sampled from `a0` to `a1` degrees. Tenths, because a typographer's
grid is where the judgements live: the letters were drawn on graph paper and the file
*is* the graph paper. Arc steps are 12°, which is about a sixteenth of a circle —
enough that a bowl reads as round at any size a game draws text, and few enough that
the pen still has a polyline rather than a mesh.

### The second alphabet is not a second font

Cyrillic is the same design with a second alphabet written on it, and the file says so
three ways rather than drawing it twice:

* **a letter that is another letter's shape** is an `ALIASES` entry — `А` is `A`, `Н` is
  `H`, eighteen of the sixty-six — because that is a fact about the shapes and not a
  coincidence to be typed out and then allowed to drift;
* **a letter that is a letter with a mark** is a `COMPOSED` entry: `é` is `e` and an
  acute, `Й` is `И` and a breve. The mark's own strokes live in `MARKS`, and *where* it
  goes is the pen's question, never this file's, because the gap depends on how much ink
  the letter under it is carrying;
* **the rest** are drawn here, on the same guides and the same grid as everything else —
  which is the point: a second alphabet of one design, not a second design.

The marks are the same eight, reused: acute, grave, circumflex, tilde, diaeresis, ring,
breve and cedilla. That is every accented letter of ISO-8859-1 — the closed Western
European set a game actually types — plus the two Cyrillic ones, so `ё` and `й` are not
a special case but the same machinery in the other alphabet.

### An accent is fitted into the letter, not added to it

An `i` is one stem wide, and every accent in the world is wider than that. Charging the
advance for the accent gives the letter a slot with a hole in it, and `í` in the middle
of a word is exactly where the hole shows. So the advance is measured off the letter
**without** its mark, and the mark is then fitted into the ground that leaves: drawn at
full size when it fits, scaled down uniformly when it does not — uniformly, because
squashing an acute in one direction turns it into a different accent — and only slid
sideways in the one case that cannot be fitted at all.

Two smaller judgements live in the same place. A mark over a dotted letter **takes the
dot's place**: `í` is a dotless `i` with an acute where the dot was, because the dot is
not a mark and an acute sitting on one reads as a colon. And the mark is anchored to the
**ink**, not to the skeleton, so the paper between a cap's serif and the accent over it
is the same `MARK_GAP = 0.05` em at every weight; across, it is centred over the letter's
*body*, because bodies are symmetric and serifs are not — an accent centred on a box
with one beak in it slides off the side of an `e`.

### The guides are the alphabet's interface

The em box, counting up from the baseline:

```
ASCENDER   0.80   the top of a letter with an ascender (b, d, k, l)
CAP_HEIGHT 0.70   the top of a capital
X_HEIGHT   0.50   the top of a lowercase x
BASELINE   0.00
DESCENDER -0.20   the bottom of a descender (p, g, y)
```

The first four are the **guides**, and they are load-bearing twice. They are the lines
the skeletons are drawn *to* — a centreline touches a guide, so the ink the pen lays
around it overshoots by half a stem, which is exactly what a real face does and
exactly why `ink_box` measures rather than trusts. And they are the only horizontals a
serif is allowed to grow on, which is how an alphabet can be stroke data and still
have feet.

A test measures it: the top of an `H` in the `sans` pen is the cap line plus half a
stem, to a half-texel. If somebody moves a guide or changes how a stem is measured,
the two stop agreeing *there* instead of in a game's menu.

### A serif is a rule about ends

A serif is not a shape. It is:

* a bar **perpendicular to the stroke**;
* only at a stroke's **end**;
* and only if that end lands on a guide **and arrives within 18° of vertical or
  horizontal**.

A vertical arrival gets a **foot** (a bar across the stem); a horizontal arrival gets
a **beak** (a bar standing on the arm). Two refinements earn their keep:

* the **x-height is the one horizontal an arm may not be serifed on**. A bowl springs
  from the x-height *tangentially*, so an `n`'s shoulder and an `o`'s flank arrive
  there looking exactly like a terminal. The rule reads the guide, and the x-height is
  excluded for arms — so a round letter is identical under a serifed pen and a
  monoline one, which is a test.
* an end within 18° of vertical is a stem; a **slash** is 58° off and is the one
  punctuation mark whose ends sit on the cap line and must stay pointed. Also a test.

Everything else about a serif — its length, its thickness, the fillet where it meets
the stem — is a number, and it scales with the weight, so the same rule reads as a
book face's feet and a slab's blocks.

### The advance is measured, not declared

This is where writing the first specimen paid for itself. The sketch version had each
glyph declare a width and the renderer trust it; the first three-line specimen showed
`quick` set as `ąuick` and `brown` with its `b` touching its `o`.

The cause was not a bad number. It was that **a width is not a property of a letter,
it is a property of the letter and the pen together**: the serif on a stem's foot
hangs past the stem by half its length, so a face whose advance was right at weight
0.08 collides at weight 0.12 and collides *worse* with serifs on. Any declared
advance is therefore a promise the pen can break.

So the promise moved to the pen — but where exactly it moved is the part that took
three specimens to get right, and the wrong answer is instructive:

* **`advance` is the larger of the design width and the letter's *body* plus a
  sidebearing on each side.** A design width the body overruns is not a width — the
  pen wins.
* **`offset`** shifts a glyph right so that its body starts on its own left
  sidebearing, because a stem at `x = 0` with a foot on it has ink to the left of the
  origin, and a glyph reaching into its neighbour's space is the same bug from the
  other side.

The trap is the word "body". The first fix measured the advance off the **finished
ink** — feet and all — which is the obvious reading and is wrong, because a serif is
*centred on its stem* and half of it therefore lies outside it. Charging the advance
for the serif tips charges every letter for its feet twice, and it lands hardest on the
narrowest letters: an `i` is one stem wide with the feet of an `n`, so it was being
handed a slot half again as wide as its own ink, and the specimen showed the second
bug as plainly as it had shown the first — `quick` set as `qu ick`, a hole in the
middle of a word.

The metal-type answer is the right one. A serif hangs **into the sidebearing**: the
letter's ground is set by its stems, the feet are allowed to use the room that was
already there, and the advance only grows when a foot would reach past the air it is
allowed (`AIR = 0.02` em beyond the sidebearing). It is the sort, the little block of
metal a letter was cast on, and it is why a foot can be substantial without a narrow
letter growing wide: the serif is not something the letter *adds*, it is something the
letter does inside the room it already had.

The third bug was one line of arithmetic in the same place. Measuring the ink, the
code added the pen's half-width to **everything** — but a stroke is a *centreline*,
which the pen haloes, and a serif is a **box**, drawn at its own size. Charging the pen
radius to a box measures a foot half a stem too wide, and it does the same damage more
quietly, with no collision to point at it.

So the tests are about the two promises separately, which is the only way to keep them
from eating each other:

* the **body** keeps at least `SIDEBEARING = 0.05` em on each side, so two stems are
  never closer than that;
* the **finished ink** stays inside the ground the advance gives it, so two letters
  never touch at all — and because that has to hold for the whole legal range of weight
  and serif, the test walks four weights and three serif amounts across all 214
  characters. It is exactly at a heavy weight that a foot reaches furthest past its
  stem. A mark is in that walk too, which is the point of the next section: an accented
  letter is a letter, and its accent has to keep out of its neighbour's way as well.

Kerning is not a table here and hinting is not here at all (see *A kern is a rule about
white*, below); neither is what stops two letters touching. What it takes instead is
being honest about what a serif is.

### Leading is a baseline distance, not a gap

The first `block` spaced lines by leaving a gap between their ink, which is the same
mistake one line down: a gap between ink makes a line of all-capitals sit closer to
its neighbour than a line with a descender in it, and the paragraph gets a rhythm of
its own that has nothing to do with the text.

`leading` is therefore **baseline to baseline, in ems** — the typographer's number,
a property of the face — and a block's baselines are exactly that far apart whatever
is on them. The sheet's *height* still follows the ink, because a block is sized by
what is in it; but the distance between two baselines does not move, which is the
distinction the test measures on the pixels: the same line printed twice, with and
without a descender, is one leading apart both times.

Tracking is the same discipline one step in: it goes *between* letters, so a line of
`n` letters has `n - 1` of them. Every layout bug in the world starts as an
off-by-one-tracking.

### A kern is a rule about white, not a table of pairs

An advance is a property of **one** letter, which is why it cannot know what happens
between two. Two stems stand at the daylight the pen gives them and that is right; an
`A` and a `V` stand at the same daylight and leave a wedge you can see through, and a
`T`'s bar ends a hundredth of an em short of an `o` that could have walked in further.
A foundry fixes this with a table of pairs, one number at a time, because every pair is
a judgement.

There is no table here. A table is a second font that can disagree with the first, and
214 characters would need forty-five thousand judgements -- numbers with nothing behind
them. `gen/text/kern.py` reads instead the two letters' **profiles** out of the field the
pen draws them with (for each height, how far left and right the ink reaches) and closes
the pair by the **smallest** of three answers:

1. **the room the closest approach has left** before it reaches the daylight the pair is
   promised: `AIR = 0.02` em between stems, `MARK_GAP = 0.05` em if either letter wears
   an accent, because two accents are the thinnest ink in the face;
2. **the white the pair is carrying over the reference**, which is measured rather than
   declared: what two `n`s keep between them in this very pen -- 0.17 em in `text` and
   0.13 in `sans`, because a serifed face keeps wider daylight than a monoline;
3. **a twelfth of an em**, which is an admission rather than a measurement: a rule that
   reads profiles will happily close a pair a designer would not.

Two decisions in that list are places where the obvious version is wrong. The spacing
question is asked of the **bodies**, because an accent is fitted into the ground its
letter already had -- the same reason the advance is measured off the body and not off
the ink. The safety question is asked of the **finished ink**, so a kern cannot walk a
mark into its neighbour. And the profiles count the rows a letter *misses*, filled in
from that letter's own far edge: a `T`'s bar is *above* the `o` and constrains it all the
same, and without that the closest approach between them is the stem and the rule never
asks the bar anything.

The numbers that come out are small and checkable: `T`+`o` closes by 0.08 em, `A`+`V` by
0.02 (its serifs already nearly touch), two `o`s by 0.03, two `n`s by nothing at all, as
the pair everything else is measured against. The spread of the air across a specimen
falls by a fifth in the book face -- which is what "even colour" means when it is
measured rather than claimed -- and the tests hold the promise that no pair, with the
correction applied, comes nearer than the daylight it was promised. The specimen sheet
prints the notorious pairs twice, with the rule and without it: a claim about white is
checked by looking, which is also how the third constant was chosen.

What the rule cannot see is that a bowl is not a stem. A round side leaves more white
above and below the point where it comes nearest, so two `o`s at a stem's daylight still
read a little loose, and the rule takes back only a third of that. That is the honest
budget of a rule where a foundry has a table. What the rule cannot do at all is the
thing the next section is about, and it is not a distance: an overhang has to be priced
against every letter that follows, because the rule cannot see which one it is.

### A ligature is what a kern would do if the hook knew what was under it

The `f` is the one lower-case letter whose ink reaches right at the top: its hook ends in
mid-air, over the letter after it. Set `fi` with the rule above and the pair closes by
0.06 em and no further — and not because the ink is in the way. The rule prices an
**overhang** at every height, because a bar over a bowl constrains that bowl just as a
stem beside it would, so the `f`'s hook is charged against the whole column of the `i`
though the hook is a shoulder above it. That is the right answer for a kern, which cannot
know what comes next. What a join adds is that knowledge: the hook is *above* the `i`, not
beside it, and with it the stem stands 0.16 em further in than the kern dares.

So a join here is the *same pair*, set by someone who can see the shapes. Five entries in
`skeleton.LIGATURES` (ﬀ ﬁ ﬂ ﬃ ﬄ), each the sequence it replaces, so the composition cannot
drift away from the alphabet the way a second copy of a letter would. Two moves make it:

* the letter after an `f` stands at the **`f`'s own design width** — the edge of the slot
  the `f` asks for and never fills. That is 0.36 em where the `f`'s advance is 0.46, and
  it leaves the `f`'s crossbar a hair short of the stem — the same daylight the kern floor
  spends;
* an `i` gives up its **dot**, and its stem **turns into the crossbar** instead of ending
  beside it. Both are forced, and the second one is the one that took three attempts. A
  stroke that ends on the x-height earns a *head* serif there — the serif rule gives the
  x-height a foot and no beak — and a head reaches back across the bar, so the stem has to
  arrive as a corner of the bar, not as an end under it. And a stem standing directly
  under the hook closes the arch into an **A**: the hook comes down onto the end of the
  `f`'s own crossbar, so the letter after it must stand clear of that, and the two designs
  that did not were rendered and thrown away. The specimen sheet is what decided it; the
  tests now measure the two things that keep it from coming back — no ink to the right of
  a joined stem at the x-height, and the hook a shoulder's height above it. The dot goes
  for a reason of its own: brought in to the join's distance it lands beside the hook's
  terminal, at the same height, and the two round marks a pen's width apart read as a slip
  of the pen rather than as a letter. It is the only part of the `i` up where the arm is.

The `f` itself is untouched, the last one included: in `ff` its crossbar ends a hair
short of the next stem, which is the `f`'s own daylight and stays. What the join buys is
visible from the outside. An advance is 0.58 em where the two letters' are 0.72, `office`
is 0.11 em narrower with joins than without, and the pair's ink is **one piece** where
`fi` set apart is three — the `f`, the stem, and the dot.

`liga` is a *switch*, not a strength: a join is one glyph or it is not, and there is no
half of one. `spec["liga"] = 0` sets a line character for character, which is what a
caption that has to stay countable wants; the default is on, and the demo sheet prints
the same line both ways so the difference can be looked at.
The joins are glyphs and not characters — `skeleton.names()` still lists 214, because
nobody types a ligature — and the sidecar's `ligatures` table says which sequences the
face joins and to what, which is all an engine needs to make the same substitution. The
join's own advance and ink box are in the metrics table like every other glyph's.

### A missing character is refused, not drawn

`ß`, `ї` and `ł` are not in the alphabet, the alphabet knows it, and asking for one
raises an error naming the character and saying how many the face covers. It does not
fall back to a box, and it does not fall back to a question mark.

The boundary is a decision rather than an accident: printable ASCII, the whole of
Cyrillic, and every accented letter of ISO-8859-1 — 214 characters. Those three sets are
*closed*, which is the only reason a face can promise them. A letter outside them (a
caron, an ogonek, a stroke) is one line in `GLYPHS` or `COMPOSED` away, and until
somebody writes that line the honest answer is a refusal.

The reason is that a box or a question mark is *visible in a shipped menu*. A face that
draws a box has decided, silently, that a missing glyph is a rendering problem; it is
not, it is a **data** problem, and the only useful moment to say so is the moment
somebody asks for a character the face does not have. The refusal is also asymmetric on
purpose: a bad *name* (`--face comic`) is refused, because there is no nearest sensible
face; a bad *number* (`weight = 9`) is clipped into range, because there is a nearest
sensible weight and refusing it would only make a live parameter unusable.

### What is not here

No hinting, no marks stacked on marks, no `.ttf` -- and no pair table and no ligature
table, which are decisions and not gaps: the kerns are the rule above and the joins are
five sequences in the alphabet, both derived from the pen every time a line is laid out.
The letters are drawn at the size they are asked for, and the honest version of "optically
correct at 12 px" is to pick the `sans` pen, which is what a UI face is for. The demo
sheet's last block is the same sentence at 11, 12, 14, 18, 24, 32 and 48 px, and every one
of them is *generated* at that size rather than scaled down from a big one — which is the
one claim this package shares with every other one here, and the reason it can make it.

The sidecar is the other half of the deliverable: `ozart/font/1`, with the guides, the
leading, every character's advance and ink box in em units, and the five joins (sequence
to glyph) under `ligatures`. A game that lays out text from it never has to re-derive the
pen, and a game that does not can still draw the letters.

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
| `autotile` | The border as a tileset: which of a hex's six edges is a border, the sixty-four tiles those six bits address, the sheet a game draws a coastline from instead of walking it -- and, because a border is only where two hexes disagree, the same sixty-four tiles for a biome border, asked *within* the land so the shore stays the coast's. |
| `rivers` | Where the water goes: pits filled to their spill points, flow accumulated in order of height, and the courses that carry enough of it turned into splines. |
| `region` | The political map: provinces grown by breadth-first search so that they are connected, capitals at their one-medians, and names -- and, by the same code, the *sea* provinces a fleet moves between. |
| `labels` | The names set into the picture: each province's pole of inaccessibility by a breadth-first search from its own edge, the largest size whose box fits inside the province, a pale halo blitted under the ink, and a census of what was drawn, what was skipped and why. |
| `politics` | Who holds the provinces: realms drawn against their own coats so that two of them are a colour apart, grown over the province graph so that each is one block, and the colour each province wears -- a faction's tincture taken to a map's wash, with the livery's own step spent rather than clipped. Then who holds the *water*: each sea province goes to the realm with the longest coast on it, and a zone no realm's coast touches stays the high seas. |
| `roads` | The roads between the capitals: Dijkstra over the lattice on the biome table's own movement costs, water impassable, and the cheapest set of roads that joins every capital on a landmass. |
| `campaign` | The map as a file rather than as a picture: the biome table, one biome index per hex, the owner of each province, the sea provinces, and the hexes the roads and rivers walk -- everything a client that does not have this generator needs to run the map. |
| `paint` | One raster pass for the ground and its relief, a wash and a frontier line for the provinces, strokes for the rivers, the roads and the frame, and the border layers -- the coast and the biome border both, drawn from the same tileset rather than walked. |

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

### A coast is a tileset, and the ink stays inside its own hexagon

A walked coastline produces a picture, and a game cannot load a picture. What a
game can load is a tileset: a small number of sprites and the rule that says which
hex wears which. So the coast is reduced to its decision, and the decision is
local:

> a hex wears the six-bit number of which of its six edges is a border.

Both hexes across an edge set the bit -- "different from my neighbour" is
symmetric -- so no walk round anything is needed, and a border belongs to the
*edge* rather than to either hex that meets there. A tile is then one of the
sixty-four six-bit numbers, and `gen/maps/autotile.py` draws all sixty-four into
an eight-by-eight sheet, because a tileset is an *asset*: it has to be complete to
be usable on somebody else's map, not only on the map that happened to produce it.
That sheet is the one file in the demo that no seed can change, and what the seed
is for is the other numbers in its sidecar -- how few of the sixty-four a map
actually wears. There are two of them, because the tileset answers two questions,
and together they are the measured answer to "do sixty-four tiles matter": a
coastline wears 26 and 27 of the sixty-four on the demo's two maps, while a *biome*
border wears 57 and 59. A coast is one line; ground mixes hex by hex, and almost
every key turns up somewhere.

The rule everything else follows from is that **a tile's ink lies inside its own
ground**. A stroke centred on the shared edge would belong half to each neighbour,
and then the picture could not be cut into tiles at all. Ink inside the hexagon
keeps the tiles disjoint -- hexagons already tile the plane, so tiles that stay
inside their hexagon compose with no rule and no seam -- and it is also what the
picture wants: a border is the rim of each of the two things, not a line hanging
in the space between them. The two sides of a border meet exactly *at* the edge,
so a coast is still a single continuous line; the ink reaching the edge at full
opacity from both sides is what makes that true without either side knowing what
the other is.

One thing is deliberately not in the tileset: the ground. Relief comes from the
height of a hex *and of its neighbours*, so two hexes wearing the same edge key can
be a plain and a mountain, and a sheet of ground tiles would be a sheet of lies. The
border is the part that is genuinely local, so the border is the part that becomes
tiles -- and the map is drawn through the same function, the ink of the edges the
key marks, measured inside the pixel's own hexagon, so the sheet and the map cannot
disagree about what a tile looks like.

That is also the test the refactor is held to. The tiled coast is compared, pixel
by pixel, against the boundary walk it replaced, and inside the map the two are the
same picture to the exported byte. The one place they differ is off the edge of the
map, where a centred stroke leaves ink on the paper and a tile does not -- which is
the place the tiles earn their keep, and the reason a coastline stops hanging over
the edge of the sheet.

A border is where two hexes disagree, and none of the module is about water. The
biome table is a labelling like any other, so the same sixty-four tiles draw an
inland border as well: `out/borders.png` is one map twice, once with the coastline
and once with both borders, and the only thing that differs between the two panels
is the second layer. The one thing a biome border needs that a coast does not is a
place to stop. Which ground lies on the other side of the water is the coastline's
business, and a border drawn there would only make the coast thicker, so the
question is asked `within` the land -- the same question, restricted to edges whose
*both* hexes are in a set -- and because a hex off the map is never in the set, a
biome border cannot hang over the rim of the map either, which is a decision that
falls out of the restriction instead of being a second rule.

The coast is special only in being the border the map is actually drawn with, and
it is asked the same way: `borders(land, grid, outside=sea)`, because what lies
beyond this map's edge is water. On the map the biome border is off unless the
caller asks for it (`paint.map_image(..., inland=0.55)`), and that is deliberate:
the map already draws a lattice to count steps on, and two sets of hex-shaped lines
is one too many. In `borders.png` the lattice is off for the same reason, so the
only hex-shaped thing in the picture is the border that is being shown.

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

### The water is provinces too

A navy moves from sea province to sea province the way an army moves from province
to province, so the sea is divided by the *same code* that divides the land: seeds
spread by best candidate, assignment by breadth-first search, capitals at the
one-medians, the same relaxation. Nothing in the division knows which of the two it
was handed, which is deliberate -- a second implementation for the water is how two
divisions drift apart. The two calls tile the map between them: every hex comes out
in exactly one province, and none in both.

Two things are the sea's own. The seeds come off **their own stream** -- the land's
stream would put a sea capital wherever a land capital happened to be, which is a
coincidence no chart has ever shown -- and the **names** take water endings
(``-sound``, ``-shoal``, ``-deep``, ``-bay``) where the land takes ``-fell`` and
``-fen``. The stem may repeat between a land province and a sea province, because
the stem is the province's number; the ending is the whole of what tells them
apart, so the two sets of endings are disjoint and a test insists on it.

The sea is drawn as a **maritime boundary and nothing else**: no wash, because the
water is a depth ramp and a wash over it would be mud, and a thinner line than a
land frontier, because the sea is not divided by walls. Its ink is where the
ownership shows: a zone that belongs to a realm is drawn in that realm's own wash --
the same colour its provinces wear on land -- and a zone nobody holds is drawn in the
neutral pale blue of a chart. So the claim is exact and the tests measure it on the
pixels: turning the sea provinces on changes water and not one pixel of land, with
the water owned as much as without it.

### Land is grown; water is claimed

A realm's land is *grown* over the province graph by breadth-first search, because
the land is somewhere an army walks and a realm in two pieces is a defect. A sea
province is not somewhere a realm marches into, so it is not grown into either: it
goes to whichever realm holds the most of its coast, and a zone no realm's coast
touches stays the high seas and stays ``-1``. That is coarser than the median line a
chart draws down the middle of a bay -- a median line needs the bay's own geometry --
and it is the honest rule at this size, because what a game asks of a sea zone is who
to ask permission from.

It is also deliberately not a second growth over the land's. ``waters`` reads the
land's labels and owners and writes a separate array over the sea's, and it counts
coast in *edges* rather than in provinces, so the realm with the longer coast takes
the bigger water and no answer depends on the order the coast was walked in: ties go
to the lower realm index. So adding territorial waters cannot move a capital or a
border on land by one hex, and both the campaign file and the sidecar carry the
answer -- the sea provinces go in with an owner, and the owner is a realm in the same
file, or ``-1`` for the water that is nobody's.

### A name goes where the province is widest

Provinces have carried names in the data since they existed, and for most of that
time the *picture* did not carry them. The reason was written down, which is the
only way a reason can expire:

> Those names are drawn here and not in the asset: this project ships exactly one
> font, Pillow's, and a game has its own, so text belongs to the preview while the
> sidecar carries the data a game would label with.

`gen/text/` is the end of that sentence. There is a font now, it is ours, it is a
function of a seed like every other asset -- and so a label is an **asset** and not
a caption: it is drawn at the size its province can hold, it is regenerated with the
map, and it is the same label at every scale the map is drawn at. The contact sheet
stopped drawing names at the capitals in Pillow's bitmap font in the same commit
that the map started drawing them itself.

Three things then make a placement a decision rather than a wish.

**Where.** A province's centre of mass is not where a name goes. The place a name
goes is where the province is widest, which is the cell farthest from its own edge
-- the pole of inaccessibility, one breadth-first search per province, seeded from
the cells that touch the frontier. A name set over a narrow isthmus of its own
province crosses the border and reads as the neighbour's, which is the one failure a
map label cannot have, so the anchor is not a taste: a test checks it against
brute-force hex distance.

**How big.** A name is set as large as its province can hold and no larger, and the
test is geometric. The box has to fit inside the province's inscribed circle, which
is a condition on the box's *diagonal* -- a wide box centred in a circle pokes out
of it at the corners -- so a long name comes out smaller than a short one, which is
what a map does. The circle is only an estimate, though, because hexagons are not
discs: the answer is read off the **data**, the province index of every cell under a
nine-by-three lattice of points across the box, and the name shrinks by 15% and
tries again, up to eight times. Below four tenths of a hex of em it is not type any
more, and the name is **skipped and reported** rather than squeezed:

```json
"labels": {"drawn": 7, "em": [11.978, 19.139], "skipped": [
  {"index": 3, "name": "Widerthanitsownprovince", "why": "no room"}]}
```

The two reasons are `no room` and `crowded`, and they are different facts about the
map: one says the province is too small for its own name, the other that a neighbour
got there first. Placement goes largest province first and the entries come back in
province order, which is the rule the atlas's packer already followed: the order a
caller reads is not the order the work was done in.

**What it sits on.** The ground is a painting, not a flat colour, so the ink cannot
assume a background. Every label is a dark ink over a pale halo, and the halo is the
*same glyphs drawn once and blitted at eight offsets around the centre*: a halo
drawn as a stroke would be a second font, and one drawn as a plate would be a
rectangle lying on a painted map. Two labels keep a third of a hex of air between
them, because the failure mode of a crowded map is not overlap but **merger** --
"Yrwick Tormoor" flank to flank reads as one word.

Everything in the placement is a length in hexes, so the label belongs to the
geography and only its pixels belong to the picture: the same seed at 10 px and at
20 px puts a name on the same cell, draws or skips it for the same reason, and sets
it at exactly twice the size. The test says so in that form -- `b["em"] ==
2.0 * a["em"]`.

Names go on after the roads and before the capitals, for the reason the roads
follow: a road is drawn *to* a capital, so a name must never cut one and the boss
must never be buried under type. The sea's names are in the data and are not drawn:
a maritime label wants a different ink and a different rule, and inventing one here
would be a second label system rather than this one.

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
  that claim against the same numbers the generator checked it against;
* **where the sea is divided and who holds it**, because a fleet needs somewhere to
  be: the sea provinces go in with their cells exactly as the land ones do, each with
  the realm that holds its coast or ``-1`` for the high seas, so the client does not
  have to re-implement the whole of `region` to move a ship or the whole of
  `politics` to know whose water it is in.

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
eight provinces, the sea divided as finely again, and a file of about 13 kB -- a
size a player can download one of per map without noticing.

## Tiling textures (`gen/textures/`)

Every other asset here is a picture of one thing at one size: an icon, a portrait,
a panel that will be stretched, a map of a particular grid. A tiling texture is the
first one that is asked a question about its *edges*:

> Lay it down next to itself, four times, and there must be no line where they meet.

The modules in this package exist because that question has four different answers,
and each one is a different *kind* of guarantee -- one that is geometric, one that is
searched for, one that is never chosen at all, and one that is never created in the
first place. Then there is a fifth module that asks a different question about the same
edge: not *does the pattern line up*, but *does it know which way it is pointing*. And
a sixth that asks what the tile is *made of* -- which turns out to be a question about
relief, and whose answer is a table of numbers rather than a function.

| Module | What it provides |
| --- | --- |
| `tile` | What "seamless" means as a number (`seam_error`), the per-pixel tile lookup, the periodic noise and periodic blur that keep a wrap, and the project's one light applied to a height field. |
| `truchet` | Truchet tiles: two arcs on a square, an arrangement with nothing to search, and the carved-stone material that comes out of it. |
| `wang` | Wang tiles: four corner colours per tile, a tiling searched for on a torus or read off a quantised height field, and the bilinear blend of those corners into ground. |
| `reaction` | Gray-Scott reaction-diffusion, simulated on a torus, for patterns that are grown rather than drawn. |
| `boards` | A plank floor: the one material here with a direction, where a board that runs off one edge and comes back on the other has to be the same board, and where the grain is long along it. |
| `ground` | Six materials for map tiles — grass, dirt, sand, rock, snow, shore — as one painter over a table of numbers, with the layers it mixes (mottle, ripple, facets, grit) and the periodic Voronoi the facets are cut from. |
| `atlas` | Ship the set: tiles packed into one sheet with a rectangle table, and with margins that are each tile wrapped onto itself rather than a gap — plus `mips`, a chain of sheets packed one per level, because a margin is two *texels* at every level and a level is a repack rather than a resample. |

### Seamless is a number, not a compliment

`tile.seam_error` divides the step across the wrap by the sharpest boundary inside the
tile. A field built to wrap comes out at about 1 -- the last row really is one pixel
away from the first, so the wrap is an ordinary edge of it -- and a field that does
not, in the tens. That turns the word into something a test can hold a generator to,
and the demo sheet records it per tile, so a texture that stopped tiling would be
visible in the sidecar as well as on the page.

The denominator is the sharpest boundary rather than the average one, and that is
worth a paragraph because the obvious version of this number is a liar. A floor is
flat between its boards and has a dark groove where two of them meet, so its *mean*
step is a hundredth of a groove; divided by that, a floor that plainly tiles reports
a cliff -- the number a broken field gets. A metric that cannot tell a floor from a
mistake is measuring smoothness. Against the sharpest boundary inside, which for a
floor is a groove too, the same seam reads 1.4: *the seam is a joint, and joints are
what the material is made of.* The lesson is not about floors. It is that "how big is
the step at the edge" is only a seam question if you have already said what the
texture's own steps are worth -- and for a material with sparse, sharp features, the
mean is not that.

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
of *random* corner colours is -- so this one belongs on ground, and the Truchet and
the reaction are the ones to reach for when the material is the point. What changes
that is not a better search but a different answer to where the corners come from.

### The tiling that was never chosen

A search arranges tiles until the corners agree. The second way to get an arrangement
removes the search instead of solving it: sample a **periodic height field** at the
corners of the grid, round each sample into one of `colours` levels, and let a tile be
the four levels at its four corners. Two neighbours share two corners, which are the
same two samples of the same field -- so they agree, and they agree because reading
one number twice gives one number. The wrap is not a special case either: the last
sample of a row is the first, because the field is the thing that wraps.

That an agreement cannot fail is the smaller half of the point. A search has a budget
and can honestly fail on a set that does not tile; reading a field cannot fail on any
field at all, which the test shows with a checkerboard -- a field whose every corner
disagrees with the corner beside it, and which comes out as two tiles with nothing
searched for. The larger half is that a search answers a different question: it finds
*a* tiling that agrees, where a field says *which* tiling. The four numbers on a tile
are four heights of one landscape, so the tiles stop being a set and become bands of
ground, with a ragged edge where two bands meet. The demo draws one of each from the
same seed and the same sixteen-by-sixteen grid, and the sidecar records what
separates them: the search uses 168 distinct tiles for its 256 cells and the field
uses 40. Same set, same cells, four times as much repetition -- which is the
difference between a picture of tiles and a picture of ground.

Three details are what make it read as ground rather than as a contour map. The first
is that the relief is the **field** and not the band it was rounded to. A level is a
staircase, and lighting a staircase shades only its steps; the field the levels are
levels *of* is smooth, so the colour posterises and the light does not -- which is
exactly what a painted texture does, and what turns a ring of nested bands into the
shaded flank of a hill. It also means the palette wants more relief than a staircase
does, because a staircase packs its whole height change into the steps while the field
spreads it across the hill: `highland` asks for twenty where the searched palettes are
happy with seven.

The second is that the field is stretched to fill `[0, 1]` before anything is read off
it. An average of octaves has most of its values near the middle whatever the seed, so
a sum that peaks at 0.66 has no top band at all -- and then every seed would come out
as lowland with a talkative coast. The stretch, which is the same thing each octave
already does to itself, is what gives every band an edge, and the test measures that
all four bands are used rather than trusting that they are.

The third is the palette, which has to be a **ramp**: there a colour index is a level
of the landscape and not a name, so `highland` runs from low wet green through grass
to bare ground and is the only palette here whose heights climb. And the grid wants to
be finer than a searched tiling's -- eight-pixel cells rather than sixteen -- because
these tiles are carrying samples and are not the picture; the features are a fixed
fraction of the tile rather than a fixed number of pixels, so a finer grid samples the
same kind of landscape rather than a smaller one.

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

### A material with a direction

Everything above is isotropic. Turn a Truchet tile, a Wang tiling or a reaction pattern
a quarter turn and you have a different picture, but you have not drawn a different
*kind* of thing. A floor is not like that. A floorboard is long and thin, its grain
runs the length of it, and both of those are directions the tile has to carry across
the seam -- which makes this the first texture here whose seamlessness is not only
about lines matching up.

Most of what it needs is arithmetic on a lattice. A board is a rectangle on a grid, so
the grid has to divide the tile: a whole number of boards across and a whole number of
bands down, or the last partial board would break the wrap. The stagger is a fraction
of a board *per band* chosen so that `down` bands advance the joints by exactly the
number of boards asked for -- which is what makes the joints come back to a position
they have already been in by the time the stack reaches the top. Get that wrong and the
floor does not tear; it *jogs*, one band's joints slipping sideways relative to the
band below, and a jog is worse than a tear because it reads as a mistake rather than as
a joint.

The interesting part is the board that crosses the edge. The stagger puts a joint
somewhere inside most bands, so the board it ends runs off the left side of the tile
and comes back on the right -- and it is *one* board, so its grain and its tone have to
carry across. The obvious implementation fails here in a way that looks like an art
problem rather than a bug. A board's identity is naturally computed from the tile's own
coordinates: which board am I in, counting from the left edge? That gives the two ends
of one board two different names, so the tile is handed two different pieces of wood
and the seam shows up not as a line but as a board that changes colour halfway along
itself. The fix is to compute the identity **on the torus** -- take the coordinate
modulo the number of boards, on a grid that closes -- and then the wrap is not a case
at all, it is the same arithmetic. The test sweeps board counts, band counts, staggers
and sizes, and checks both halves of that on every row: the board's own coordinate
carries on across the edge by exactly one pixel, and where it instead runs off the
end, the identity advances by exactly one board, because running off the end is what a
joint *is*.

The surface is a bevel and nothing else. A sanded board is flat, its grain is *in* the
wood rather than on top of it, so the light has only one thing to find: the groove
where two boards meet, which comes up over a couple of pixels and by a couple of
pixels. Grain given its own relief -- even a little -- turns the floor into corrugated
iron, which is a different and worse material, so the grain is a colour modulation and
the bevel is the whole of the geometry. What makes it grain rather than noise is
`stretch`: `periodic_noise` takes a factor by which features are longer along x than
along y, which is one number in a Gaussian kernel and the difference between a floor
and a top-down view of gravel. At `stretch = 1` it is the old noise to the last bit,
which is the test, because the only thing the parameter does is multiply a width.

Two smaller things follow from a board being a board. Every pixel of it has to agree
about its wood, so the per-board values -- tone, hue turn, which slice of grain it was
cut from, whether it has a knot -- come out of a table indexed by the board and not
from a stream of draws, because a stream would give one board two answers as the
painter walked across it. And the grain is rolled along per board out of one periodic
field: a *roll* of a field that wraps is continuous, which is the same property that
made the noise worth having in the first place, so no two boards are cut from the same
piece while every board's grain still crosses the seam without a step in it.

### Six materials, one painter, a table of numbers

A map draws ground as a flat biome colour, and at a hex or two that is right: the
question a map answers is *what is this hex*, not *what does it feel like*. The moment
the camera comes down -- a battle map, a site screen, a tactical view -- the colour stops
being enough, because what the eye reads as a material is not its hue but the way the
light breaks across it. Sand and dry grass are both a pale olive in a satellite
photograph; what separates them is that sand has ripples and grass has tufts, and both of
those are *relief*. So `gen/textures/ground.py` builds a ground as a height field with a
colour on it, and its six materials are one painter over one table:

```python
Ground("rock", base=(0.56, 0.008, 0.290), drop=0.19, turn=0.010,
       lump=0.30, lump_scale=20.0, ribbon=0.00, ribbon_scale=3.0, ribbon_stretch=1.0,
       facet=1.00, cells=3, bevel=0.06, dome=0.0, grit=0.10, relief=7.0)
Ground("sand", base=(0.78, 0.062, 0.260), drop=0.07, turn=0.012,
       lump=0.40, lump_scale=16.0, ribbon=1.00, ribbon_scale=10.0, ribbon_stretch=6.0,
       facet=0.00, cells=1, bevel=0.00, dome=0.0, grit=0.10, relief=3.2)
```

Fifteen numbers a row, and the weights *are* the material. `lump` is isotropic mottling
at a chosen feature size -- tufts, clods, drifts. `ribbon` is the same noise stretched
along one axis, which is the parameter that was grown for the boards' grain and turns out
to be the whole of sand: ripples are features that are long in one direction and short in
the other. `facet` is a periodic Voronoi, for plates of stone and pebbles on a shore.
`grit` is one random value per pixel, which is what a close-up material has and a
painter's flat fill does not. Rock is almost all facets and almost no colour variation;
snow is a big soft lump with the relief turned down so the light barely finds it; shore is
the same facets as rock with more of them, a wider bevel and a dome.

The point of the table is that a seventh material is a **row**, and not a function.
`texture` takes a row as well as a name, so

```python
slate = rock._replace(key="slate", base=(0.36, 0.012, 0.300), drop=0.09)
ground.texture(slate, seed=3, size=64)      # painted, lit, and held to the seam
```

is a material, with nothing registered and nothing special-cased. That is a claim a test
has to keep honest, because the failure a table invites is the field that is written
down, documented, put in the sidecar -- and not read. So one test perturbs each of the
fifteen numbers on a material that uses it and asserts the finished tile changes; `key`
is the only exception, because it is the row's name rather than a material.

One number is a conversion rather than a measurement. `tile.periodic_noise` takes a
Gaussian *sigma*, and a table written in sigmas is a table of blurs: the first sheet of
grass had a six-pixel sigma and read as fog, because a field smoothed that hard is a
mottle about eighteen pixels across. What a material is written in is the size the eye
sees, so `_noise` divides by three, once, and the tests hold the conversion to its
measurement -- the field decorrelates over between a third and three quarters of the
feature size it was asked for. The rest of the sheet followed from that change alone.

The facets are the one genuinely new piece of machinery here, and they are the reason
the module is not just four noises. A Voronoi diagram drawn on a square and cropped in
the middle of a cell **repeats**: the same pattern comes back tile after tile, and the eye
finds that faster than it finds a line, which is a worse seam than the one it set out to
avoid. So the diagram is computed on the torus -- a point is also its own image a whole
tile away, and the displacement is the shortest of the three on each axis -- and a slab
that leaves the right edge is the same slab coming back on the left, with the crack
between them where the crack would be. The test does not check that against the points it
was built from. It checks the geometry: either the two sides of the seam are the same
cell, or the seam *is* a border and `edge` is small on both sides of it, and where a pixel
has a neighbour in another cell, the distance to the nearest foreign pixel has to agree
with `edge`. That is the property a crop-and-tile version fails and a correct one cannot.

How the facets are found is worth a paragraph because the obvious version is a lie.
Every cell is measured against every pixel -- `cells ** 2` passes, nine for rock and a
hundred and forty-four for shore -- where the fast version is to look at the nine cells
around the one a pixel falls in. That search is wrong for jittered points: a point
jittered towards a corner can be nearer, across a border, than the pixel's own point is,
so being one cell out is a kink in a crack that nobody can find later. At the sizes a
tile is made in, the honest loop is also the fast one.

What the facets are worth is three numbers, and they are the difference between
flagstones and gravel. `edge` is half the gap to the second-nearest site, so it is zero
on a crack and grows inward; clipped at a `bevel` it is a plate with a flat top -- rock.
`dome` then decides what happens to the top: at 0 it stays flat until the bevel, and at 1
it is `sin` of the same number, round all the way to the crack, which is a pebble. Same
cells, same painter, three materials -- which is the same argument the whole module
makes, one layer down.

Two tables, and one test between them. `FOR_BIOME` says which material a map biome
wears: plains, steppe, forest and marsh are all grass; hills and mountains are both rock;
`dirt` is the one material **no biome wears**, because it is the ground under a road, a
camp or a ploughed field, and a map that wants it knows why. The water biomes are absent
because a ground texture is what a figure stands on. The map's table says what a hex *is*
and this one says what it is made of, and they live in different packages on purpose --
`gen.textures` does not import `gen.maps`, and the only place that knows both is the test
that asserts every land biome has a material, no water biome does, and dirt is left over.
The colours are tuned to the same palette without importing it: the bases sit where the
map's biomes sit in OKLCH -- plains at 0.353 turns, desert 0.262, mountain 0.296, snow
0.663 -- a little darker, because a hex a figure stands on is read from a metre away.

And the module's thesis, made as a measurement. Of the fifteen pairs of materials, the
closest in colour is **grass and rock**: a mid green and a mid grey, 0.08 apart in OKLab
per pixel, close enough that a generator varying only colour would be entitled to call
them variants. They are the farthest apart in what actually varies, which is relief --
their height fields differ by more than three times in spread. The test finds the
closest pair in colour on its own and asserts that pair is far apart in relief, so it
would notice if the six ever drifted into six tints of one surface.

The seams are 1.2 to 1.8 on the sheet and the test's bound is 2.0 rather than the 1.5
the smooth tiles are held to, for the same reason a floor reads 1.4 on this number: a
material with cracks in it can have a joint land exactly on the seam, and a joint is what
the material is made of. The demo draws each tile four times, captions it with the biomes
that wear it and the tile size, and records every row of the table, the seam error, the
mapping the other way round, and the packed atlas of all six in the sidecar.

### An atlas of materials is not an atlas of sprites

A game does not load ten floor textures, it loads one, so a texture set needs what a
sprite set needs: a packer and a table of rectangles. The layout is the same
bottom-left skyline as `export.atlas`, and the reason this is a separate module rather
than a call to `export.atlas.build` is that a tile **breaks both of that atlas's
rules**, and it breaks them for the same reason twice over -- a tile's edge is not a
border, it is the seam.

The first rule is that a sprite is **trimmed** to the pixels it uses, because the
transparent margin around a sprite is not part of it. A tile cannot be trimmed at all.
Crop the empty-looking border off a floor tile and the tile no longer joins to itself,
because the pixels that got cropped are the ones the wrap was made of; a trim would
remove exactly what the texture spent its whole design earning.

The second is that a sprite's gap in the sheet is **padding**, and its job is to be
*nothing* -- a filter that steps off the sprite should find transparent pixels rather
than the neighbour's. A tile's margin cannot be nothing. A tile is opaque, and the
margin is precisely what a filtered sample near the edge is going to read. So the
margin is filled with the tile's **own continuation**: the tile wrapped onto itself,
so that the row above a tile's first row is the tile's last row, the column left of its
first column is its last column, and the corner between the two is the tile's own far
corner. It is not a compromise between neighbouring tiles and a hole; for a seamless
texture it is *exactly right*. Nothing in the margin is a pixel the tile does not have,
and the property is not taken on trust -- the test compares the margin with the tile,
in bytes, and does it for a wide tile as well as a square one.

The word for it is a **bleed**, and it is the printer's word: the part of the plate
that runs past the cut. What runs past the cut here is the tile running past itself.
Two pixels of it is what a bilinear filter needs, and it costs about six percent of
the sheet. There is no gap *between* tiles either, which is the same reversal one more
time: a padding pixel in an opaque sheet is a hole, and the only thing a sample beside
a tile should be able to reach is more of that tile, so the blocks are laid flush.

That margin also means a mip chain built from this sheet behaves. The hardware
averages each tile together with its own continuation, which is the average the tile
would have had if it were tiled out, and the only place that goes wrong is a few pixels
either side of the boundary *between two different tiles*, at the coarsest levels. But
the hardware is guessing, and the honest thing is still to hand it sheets that are
right rather than sheets that nearly are: `atlas.mips` packs one per level, and it
repacks rather than resampling a sheet, because a margin is measured in **texels**.

Two texels is what a bilinear filter reaches, at *every* level, so it is two texels of
a tile that is four times smaller -- four times as large a share of it. Keep level 0's
rectangles and let the GPU average, and each level's margin shrinks away exactly as it
starts to matter. Level 0 of the chain is `atlas.build` byte for byte, so a single
sheet is the head of a chain of one, and the price of the convention is a number per
level in the sidecar: on the demo's ten tiles the top level packs at **0.31** and the
bottom, a sheet of one-texel tiles, at **0.02**, and the whole eight-level chain costs
1.34 times the pixels of level 0 alone. The extra third is entirely margin -- spent at
the levels where a tile has become smaller than its own bleed. That is the same
argument as the density rule above, arriving at the level where it is hardest to argue
with.

Two decisions make a level a level rather than a blur. The average is taken in **linear
light**, because half the light is not half the byte -- a black-and-white checker
reduces to 188 and not 128, and a chain of sRGB averages would darken every level and
compound; and it is taken **periodically**, because a tile has no edge, so the reduced
tile still tiles. That second one is exact and is tested as an equality: reducing a tile
laid out twice gives the same picture as laying the reduced tile out twice. A periodic
tile of odd size cannot be halved exactly -- an odd period does not divide in two, so
the reduction rounds up and carries a phase error of at most one texel -- but it still
cannot invent a step, because every output texel is an average of neighbours the tile
actually had.

And the density rule from the icon sheet applies here in its sharpest form. Ten
128-pixel tiles with two pixels of margin are 132, and three of those fit across a 512
sheet with 116 pixels to spare -- so the fourth row goes over the edge, the sheet
doubles to 512x1024, and the density falls from 0.63 to **0.31**. Two pixels of margin
did that, and nothing is lost or wrong: the sheet is simply twice the pixels it needs.
Power-of-two is still the default, because that is what an engine wants, and the price
is in the sidecar rather than in a comment -- 0.31 as the sheet ships, against 0.78 for
the same ten tiles with the rounding turned off.

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

## The build as a page (`export/gallery.py`)

Thirty-three sheets, a sidecar beside each, and no way to see what a build contains
without opening files one at a time. That was the oldest item on the list. This is
the interface to it: `out/index.html`, written by `demo` as its last step, or on its
own with

```bash
python cli.py gallery --manifest out            # linked: 29 KB, opens beside the pictures
python cli.py gallery --manifest out --embed    # inlined: 6.5 MB, travels as one file
```

### A page, not a window

The obvious answer to "a GUI over the generators" is a Tk window, and tkinter is in
the standard library, so this is a decision rather than a shortage. The argument
against it is that a generator set's interface *is the build*: every picture already
has a sidecar that names the function which drew it, the seed it came from, and the
licence it is under. Anything that displays those pictures has to read those
sidecars, or it will drift from them -- and the moment it reads them it is no longer
a program with a state machine and an event loop. It is a rendering of files that
exist. A page is that rendering with nothing left over: no install, no dependency, no
event loop to keep alive, and one artefact that can be opened from disk today and in
ten years.

The page is deliberately dumb. One stylesheet, inline. No JavaScript, no web font, no
build step of its own. It reads `credits.manifest()` -- the same walk of sidecars the
licensing gate uses -- so a generator appears on the page the day it starts writing a
sidecar, and nothing has to be remembered. A registry of descriptions would be a
second list to keep true; the sidecar is the first one.

### What a card says

Four things a sheet raises and does not answer:

- **what drew it** -- the sidecar's `source`, verbatim: `procedural:gen.maps.autotile`
  is a module you can open, and it is the answer to "how was this made";
- **from what** -- its seed, and its count, because `x8` is the difference between a
  sheet and a sprite;
- **the numbers** -- the sidecar as a table, sorted by key;
- **that the file is the file** -- the first 16 hex of its sha256, which is the digest
  `cli.py demo` prints and what makes "byte for byte" checkable by a person and not
  only by CI.

The title is the file stem with its dashes opened out (`icons-atlas` is "icons
atlas"), and that is the whole of the naming: a caption written by hand would be one
more thing that can disagree with the build, and this page exists to be checked
against the build.

### A list is a length

A sidecar is not a small document. An animation's sidecar holds every frame's
rectangle and the phase it was drawn at; a map's holds its provinces, its rivers, its
roads. Putting that in a card would make the page a JSON viewer with pictures in it,
so a list becomes `12 items` and a dict becomes `6 keys`, and only the scalars are
printed. The frame rectangles are one `export.sheet` call away for anyone who wants
them; the card's job is to tell you whether to want them.

### The gate is part of the view

The page says, in the gate's own words, whether the gate would ship the build it is
showing. A clean build gets one green line. A picture with no sidecar, or a sidecar
that records no licence or names a source nobody can account for, gets a red block at
the top listing the files -- the same list `cli.py credits --check` exits nonzero on.

This is the one place where the gallery is not neutral. CC BY means attribution is a
condition of shipping, so a view that quietly displayed a build the gate would reject
would be a view that moved the licence violation into the place people look. The
credits block itself travels at the bottom of the page, verbatim from
`credits.credits()`, including the Kevin MacLeod line. A gallery that is missing it is
not a gallery of *this* build.

A campaign file (`out/maps.campaign.<seed>.json`) is not an asset and has no picture,
so it gets a line -- name, `format`, size -- under `data` instead of a card. The page
is a catalogue of the build, not only of its illustrations.

### Two files or one

By default the page links the pictures beside it: 29 KB of HTML over 4.9 MB of pictures,
which is the version to regenerate while working, because the browser is reading the
files that were just written. `--embed` base64s each picture into the page, which is
6.5 MB of HTML and one file with no folder to lose -- the version to send to someone
who does not have the build. Both are pure functions of the directory: build twice,
get the same bytes, which is what the 20 tests in `tests/test_gallery.py` assert along
with the claims a view can get wrong -- that every sidecar becomes exactly one card,
that an embedded picture decodes back to the file's bytes, that a 400-entry list is
one line, and that `<script>` in a sidecar is text.

### What is not here

No editing, no regeneration, no live preview. The page describes a build; the build
is still `cli.py demo` and the seed. A viewer that could also *make* things would need
a second copy of every default, and the first copy is already in the modules. If the
loop wants closing, the honest next step is a build watcher that re-runs `demo` and
reloads the tab, not a second interface with its own idea of what a parameter is.

## Roadmap

Generators to sit on top of the core, in the order they become useful:

- **icons** -- ~~resource/unit icons, atlas packing, `--size` sweep for mip levels~~,
  ~~quantities and tiers drawn as a badge on the plate~~ and ~~a faction hue table so
  a faction's icons are its own colours~~ *(done: `gen/icons/`, where an icon is a
  plate with a charge on it and a set shares one plate; the charges are symbols
  rather than depictions because a sword drawn honestly is a smudge at 32 px; the
  passes, the margin and the bevel are world units, so the 32 px icon is *drawn*
  small rather than resized down; `export/atlas.py` trims, packs and reports the
  density of the sheet, which is how the cost of power-of-two dimensions became a
  number in the sidecar instead of a convention nobody questioned; `badge.py` gives a
  count a seven-segment numeral and a tier a row of pips, in the icon's own colours
  and refused rather than rounded when it cannot be drawn; and `livery.py` lets a
  faction's own field dress the **plate** while the charge keeps the kind's hue, so
  the second source of hue lands on the ground and does not fight the first -- read
  from `crest.field_tincture`, the same place the map reads its wash, so the two
  cannot disagree)*.
- **heraldry** -- ~~shields, field divisions, charges, tincture rules~~, a
  ~~faction's crest derived from its name as the seed~~, and ~~faction shades of the
  tinctures in a sidecar~~ *(done: `gen/heraldry/`, with `crest.py` deriving the seed
  from a hash of the name rather than Python's salted `hash()`, deriving the banner
  from the coat so the two cannot disagree, and deriving a **livery** from the same
  name -- three integers per tincture, spent only where the colour has room, so two
  factions that both chose Azure are two Azures and a shade is either a visible step
  or no step at all)*; ~~meeting heraldry on the map~~ *(done, in the maps:
  `gen/maps/politics.py` washes a realm's provinces in its own field tincture down
  to a 13 px hex, which is the first place the livery had to survive being spent)*
  -- the heraldry is whole, and it dresses four things now: a shield, a banner, a
  province and (in the icons) a plate.
- **flags** -- ~~bands by projection onto directions, overlaps by half-plane
  clipping~~, ~~cloth folds as sinusoidal normal displacement~~, ~~a pole and a fly
  that trails~~ and ~~a gonfalon hung from a crossbar~~ *(done: `gen/heraldry/`,
  where one height field supplies both the lighting and the projection, in world
  coordinates so that the folds are the same folds at any resolution; a taper of
  zero is an exact anchor at the hoist and a droop is gravity taken downwards, so a
  pennant has a direction to hang from and now something to hang from; the pole is a
  cylinder and a sphere shaded by the one light rather than a bevel; and a bar is
  the same rod turned a quarter turn and the same cloth held the other way, so the
  folds deepen downwards from it and what gravity does to the cloth is stretch it).*
- **animation** -- ~~sprite sheets, one frame a cell, the table an engine cuts the sheet
  up with, and the timeline it plays on~~ *(done: there was no physics to add, because
  a waving flag was already a field in `t`. `flags.waving` returns the frames and the
  phases they were drawn at, so frame *i* is the same call the contact sheet makes with
  one number different, and nothing is interpolated. `export/sheet.py` is the delivery
  -- equal cells, clear padding, a table taken from the same `render.cells` the picture
  is drawn on, and the frame rate and loop flag beside it -- and unequal frames are
  refused with the atlas named, because a sheet's cells are what an engine aligns and
  an atlas is what trims. The loop is the part the cloth had an answer to: the second
  harmonic is not commensurate with the first, so a folded cloth returns after ten
  turns (`cloth.period`, 20π) and a one-turn loop of *that* cloth has a seam -- which
  is why the demo asks for a cloth with one harmonic, whose loop closes byte for byte
  at `p + 2π`. What holds either way is that the sampling is uniform, so the seam is a
  step like any other step, and that a finer sampling is smaller steps rather than a
  different motion; both are measured in the tests rather than asserted)*.
- **portraits** -- ~~SDF head, ink contours, cel shading, traceable genome,
  heredity~~ *(done: `gen/portraits/`)*; ~~outfits built from the same shoulder
  line~~ *(done: a collar stroke, three kinds, one per seed)*; ~~turn the head in
  depth~~ *(done: `gen/portraits/turn.py` -- the head is a stack of cross-sections,
  so the outline is a support rather than a squeeze, an eye is a chord on the
  surface, the jaw swings because it has a depth of its own, and the side a head
  turns away from is derived rather than declared)*; ~~expressions that move the
  whole face rather than three features~~ *(done: `gen/portraits/expression.py` warps
  the *layout* -- `eye_y` and the four gaps the face is derived from, so one number
  moves the brow, the eyes, the nose, the mouth and the blush together and by
  different amounts -- and the eight feature deltas ride on top of it; the head, the
  hair and the skull are built from the unwarped style and the hairline is pinned
  through `Style.pinned`, because a mood moves a face and not a person; the
  horizontal is deliberately not warped, since the distance between the eyes is
  anatomy, so a wide grin is the mouth getting wider rather than the face spreading
  out)*.
- **figures** -- ~~a body that walks, a gait that closes, and a sprite sheet out of the
  two~~ *(done: `gen/figures/`, and the first thing it did was rename every bone.  A
  bone named after the joint it *reaches* gave a thigh the length of a hip socket, so
  a stride of thirty-eight hundredths of a radian moved the figure by a quarter of a
  degree and the walk stood still -- nothing crashed and the picture was plausible.
  The angles are now absolute, a cycle speaks in deltas from a rest pose, and every
  bone is named for what it is.  Every term of a gait is a sine or a raised-cosine
  window, so a loop closes at `2*pi` by construction and the test compares the bytes
  of phase `0` and phase `2*pi` rather than trusting the algebra; the flag's
  incommensurate harmonic is the contrast that makes that worth stating.  A walk is
  contralateral, a run has a lean and a flight phase, and an attack is a one-shot
  inside one turn, which is how a game plays it.  The drawing is the portrait
  package's brush, ink and cel pass under the **one** light, so a figure and a face
  are one art set -- while the line and the shading bevel are in **pixels**, because
  a line belongs to the raster; and a figure is silhouette, costume and five colours
  rather than a face, for the reason the icon gives about a sword at 32 px)*;
  ~~the other views~~ *(open: a mirror is free, a front and a back view are not -- the
  near and far limbs stop being a depth cue and become an anatomy)*.
- **ui** -- ~~9-slice frames and panels, ornamental borders from a Fourier series
  of a radius function~~, ~~asymmetric insets for title bars~~ and ~~wood/stone
  material bodies via fBm + domain warping~~ *(done: `gen/ui/`, with the series
  moved into the corner box, which is the only part of a nine-slice asset that is
  never resampled; the insets and bands taking one number per side so that a title
  bar is a wider border rather than a second kind of window; and the one look that
  gives up a stretchable middle, priced by `slice.stretch_error` and carried in the
  sidecar as a measured `stretchable` rather than a promise)*; ~~a material that is
  cheap enough to draw live~~ *(done: the plan said "a small cache keyed by
  `(kind, scale, seed)`", and the interesting part is why that key would have been
  wrong.  A material does not depend on the colour it is mixed into, so what is
  cached is the *modulation* -- the lit relief and the stain -- and a button's three
  states cost one material between them; but a material is anchored to the world, so
  a wider panel samples a wider window of it and the window is part of the key.  38x
  to 69x cheaper to re-apply, a 1.86x faster three-state button, transparent (a hit
  is bit-identical to a miss, checked on the pictures) and bounded at
  `material.CACHE_MAX = 16`).*
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
  same bytes at every scale it can be drawn at)*; ~~sea provinces~~ *(done: the sea
  is divided by the same code as the land -- `region.seas`, one stream of seeds for
  the water, water endings on the names, drawn as a maritime boundary and nothing
  else so that charting the sea changes water and not one pixel of land, and carried
  in the campaign file with an owner of `-1`)*; ~~territorial waters~~ *(done:
  `politics.waters` -- land is grown and water is *claimed*, so a sea province goes
  to the realm with the longest coast on it and a zone no coast touches stays the
  high seas; the count is in coast *edges* so a longer coast takes a bigger water,
  ties fall to the lower realm so the answer is a function of the map, and the land's
  own division is not touched, which is measured on the pixels -- the water is owned
  and not one pixel of land moves)*; ~~the coastline as a tileset rather than a
  stroke~~ *(done: `gen/maps/autotile.py` -- a border is six bits per hex, which of
  its six edges has something else across it, and the sixty-four tiles those bits
  address are an asset a game draws a coast from instead of a picture; the ink lies
  inside its own hexagon so the tiles compose with no rule and no seam, and the map
  is drawn through the same function the sheet is, so the two cannot disagree -- the
  tiled coast is the walked one to the exported byte inside the map, and differs only
  off the edge of the map, where a stroke leaves ink on the paper and a tile does
  not)*; ~~the same tiles for a *biome* border~~ *(done: nothing in the tileset was
  ever about water -- a border is where two hexes disagree -- so `autotile.borders`
  grew one parameter, `within`, and the biome border is the coast's own question
  asked of the land alone: an edge is a border only where both hexes are in the set,
  which keeps the shore the coastline's business and, because a hex off the map is
  never in the set, keeps a biome border off the rim as well. The count the sheet
  prints is now two counts, and they are the argument: a coast wears 26 of the
  sixty-four keys, a biome border 57, so the sheet has to be complete either way.
  `out/borders.png` is one map twice, coast only and both, with the lattice off so
  that the only hex-shaped thing in the picture is the border being shown, and the
  map draws it only when asked -- `paint.map_image(..., inland=...)` -- because two
  sets of hex-shaped lines on one map is one too many)*; ~~a province's name set into
  the map rather than into its sidecar~~ *(done: `gen/maps/labels.py`, and it is the
  paragraph above the code that matters -- the reason names were preview-only was
  written down next to them, "this project ships exactly one font, Pillow's", and
  `gen/text/` expired it.  A label is an asset now: the anchor is the pole of
  inaccessibility rather than the centre of mass, the size is the largest whose box
  keeps every lattice-sampled point inside the province, a name that cannot be set is
  skipped and *reported* with its reason, two names keep air between them because the
  failure of a crowded map is merger rather than overlap, and the halo is the same
  glyphs blitted at eight offsets so the ink never assumes a background.  Every
  quantity in the placement is a length in hexes, so the same seed puts the same name
  on the same hex at twice the size when the map is drawn twice as large -- which is
  the claim the test makes, `b["em"] == 2.0 * a["em"]`)*.
- **textures** -- ~~Truchet tiles, Wang tiles for seamless tiling, Gray-Scott
  reaction-diffusion for organic patterns~~ *(done: `gen/textures/` -- Truchet
  arrangements, which are seamless for any choice; Wang tilings searched for on a
  torus, seamless because the four corner colours of a tile agree with its
  neighbours' and the blend between them is continuous; and a reaction-diffusion
  simulated on a torus, which is seamless because it never had an edge. All three
  are held to a measured seam error, and the same number goes into the sidecar)*;
  ~~a Wang set whose corner colours come from a quantised height field rather than at
  random~~ *(done: `wang.from_field` reads the tiling off a field -- the corners are
  samples of one periodic landscape rounded into bands, so two neighbours agreeing is
  one number read twice and no search exists to fail; the relief is the field rather
  than the band it was rounded to, because a staircase shades only its steps, and the
  demo draws the two ways side by side with the tiles each one used in the sidecar)*;
  ~~anisotropic tiles for a floor that is stretched along one axis~~ *(done:
  `gen/textures/boards.py` -- boards on a lattice that divides the tile, staggered so
  the joints return to a grid they have been on, with a board's identity taken on the
  *torus* so that the board running off the left edge and back on the right is one
  board and its grain carries across; the surface is the bevel into the groove and
  nothing more, because grain with relief of its own is corrugated iron, and
  `periodic_noise` grew a `stretch` for the grain to be long along the board)*;
  ~~a texture atlas~~ *(done: `gen/textures/atlas.py` -- the same skyline as the sprite
  atlas and the opposite margins, because a tile's edge is its seam: nothing is
  trimmed, since the border a trim would remove is the wrap, and the margin is the
  tile wrapped onto itself rather than a gap, so a filtered sample beside a tile
  reads the material and not a neighbour; the block costs about six percent of the
  sheet and the power-of-two rounding cost is in the sidecar, which is where the demo
  shows the margin pushing a set over an edge and doubling the sheet)*; ~~a sheet per
  mip level~~ *(done: `gen/textures/atlas.mips` -- the chain's head is `build` byte for
  byte, so a single sheet is a chain of one, and each level below is the level above
  reduced and *repacked*, because a margin is two texels at every level and a texel of
  a quarter-size tile is a quarter of the picture; a level averages in linear light (a
  black-and-white checker reduces to 188, not 128, and an sRGB chain darkens at every
  step) and averages periodically, which is tested as an equality -- reducing a tile
  laid out twice is laying the reduced tile out twice -- so the level below still
  tiles; an odd-sized tile cannot be halved exactly and the reduction says so instead
  of pretending, and the sidecar carries each level's own density, where the coarsest
  sheet of a ten-tile chain reads 0.02 against 0.31 at the top)*;
  ~~a ground material set for map tiles~~ *(done: `gen/textures/ground.py` -- grass,
  dirt, sand, rock, snow and shore as one painter over a table of fifteen numbers a
  row, mixing four layers: a mottle at a feature size, the same noise stretched into
  ripples, periodic Voronoi facets for plates and pebbles, and one random value per
  pixel of grit.  A seventh material is a row, `rock._replace(...)`, and a test
  perturbs every number to prove none of them is decoration.  The facets are the new
  machinery and they are cut on a *torus* -- every cell against every pixel, because
  the fast nine-cell search is wrong for jittered points and being one cell out is a
  kink in a crack -- so a slab that leaves the right edge is the same slab coming back
  on the left.  `FOR_BIOME` ties the six to the map's nine land biomes without either
  package importing the other, and dirt is deliberately worn by none: it is the ground
  under a road.  The measurement the module is built on is that the closest pair in
  colour, grass and rock, is the farthest apart in relief)*.
- **text** -- ~~a face that is code rather than a file: an alphabet of skeletons, a
  pen that gives it weight, contrast, serifs and slant, and the metrics an engine
  needs to lay a line out~~ *(done: `gen/text/` -- 214 characters: printable ASCII,
  the whole Cyrillic alphabet and the accented letters of ISO-8859-1, as polylines in
  tenths of an em, drawn on four guides. The second alphabet is not a second font:
  where a Cyrillic letter has a Latin letter's shape it is that letter, aliased, and
  where it is a letter with a mark it is that letter and a mark, so 214 characters are
  95 characters' worth of drawing and two tables of the rest. The split between
  what a letter *is* and what a pen *does to it* is what makes five faces one alphabet,
  which is why the genome, the sidecar formats and the credits gate all learned nothing
  new. Two of the decisions are places where the obvious version was wrong: an advance
  is *measured* off the ink rather than declared beside it, because a width that a
  serif overruns is not a width and the collision appears exactly when the weight goes
  up; and a missing character raises rather than drawing a box, because a box in a
  shipped menu is a question that was answered silently. A third is the mark: it is
  fitted into the ground the letter already has, so an accent never widens a word nor
  lands in its neighbour's slot)*; ~~kerning, without a table of pairs~~ *(done:
  `gen/text/kern.py` -- the two letters' profiles read out of the pen's own field, the
  close bounded by the daylight the pair is promised, the white it carries over the
  measured reference pair, and a twelfth of an em; the spacing question asked of the
  bodies and the safety question of the finished ink, which is the same split the
  advance makes)*; ~~ligatures~~ *(done: `gen/text/skeleton.py` -- five joins, ﬀ ﬁ ﬂ ﬃ
  ﬄ, composed from the letters they replace. The letter after an `f` stands at the `f`'s
  own design width, an `i` gives up its dot, and its stem turns into the crossbar so it
  grows no head; `spec["liga"]` switches them and the sidecar lists them. Two designs
  were rendered and thrown away first -- a stem under the hook, and a join whose stem
  kept the head serif it earns on the x-height; both close the arch into an A -- and the
  tests now measure the two things that keep it from coming back)*. This is also the
  licensing note's "fonts leak" closed:
  the shipped face is a function of a seed like everything else, and Pillow's bitmap
  font is back to being what it always was here -- a dev tool).*
- **export extras** -- ~~SVG for vector UI~~ *(done, as `export/svg.py` + the
  `vector` module in `gen/ui/`: a panel is four rounded rectangles and a polar
  ornament, so it is the one asset here with an exact vector form. The paths are
  built from the same `frame.Panel` the raster thresholds, which is what makes the
  two impossible to drift apart -- a test asserts that every point any curve was
  built from lies on the zero set of *its* field, 2.3e5 points, worst 6.3e-14. The
  light is not in it (a bevel is a ramp, not a curve) and a material is refused
  outright, because an export that silently drops what it cannot draw ships. The
  payoff is the opposite of the nine-slice's: no cuts, no insets, no mip chain, one
  file for every DPI)*, ~~atlas packing~~ *(done, as
  `export/atlas.py` and shared with the icons sheet)*, and an ~~`assets.yaml`
  registry carrying `source/license/attribution/seed` for every asset~~
  *(done, as `core/credits.py` + the sidecars: JSON rather than YAML, because a
  YAML parser is a dependency this project does not take)*. The registry also
  generates the credits block ~~and the screen~~ *(done: `credits.screen`, a
  scroll set in this project's own face)*, which matters because the Kevin MacLeod
  tracks are CC-BY and *require* attribution; ~~use the same manifest to check a
  build before it ships~~ *(done: `credits.check`, and `cli.py credits --check`.
  The manifest is read *back* rather than trusted, and the three ways a build can
  owe somebody something it has not recorded are kept apart because they are fixed
  in different places: a curated source missing a field (a property of the code,
  reported with no directory at all), an asset nobody recorded (a picture with no
  sidecar -- raster or vector, `credits.PICTURES` is both -- or a sidecar naming a
  source that is neither ours nor curated), and an attribution that is not in the
  credits *on disk* -- which is the failure mode of forgetting, and happens to
  honest people.  The gate reports rather than repairs and returns a nonzero exit
  code, and the credits screen is the one picture exempt from the sidecar rule
  because it is the attribution itself)*.
- **a GUI over the generators** -- ~~a view of the build: every sheet, the numbers
  behind it, and the command that rebuilds it~~ *(done: `export/gallery.py` writes
  `out/index.html` -- a page rather than a window, because the interface of a
  generator set is the build and the build already describes itself in sidecars; the
  cards are the manifest, the gate's verdict is on the page in the gate's own words,
  and the credits travel with it)*; ~~live regeneration from the view~~ *(open, and
  it is a build watcher rather than a second interface: a page that could also make
  things would need a second copy of every default)*.

## Licensing notes

* Code: MIT (see `LICENSE`).
* Generated PNGs: no third-party rights are involved -- keep the seed in the
  registry and the asset can always be rebuilt.
* Watch these three, they are the usual ways third-party rights leak in:
  **fonts** (closed here: `gen/text/` draws its own -- 214 characters from an alphabet
  of skeletons and a pen, with the Cyrillic as aliases and marks on the same letters
  rather than a second font, so the shipped face is code and a seed like everything
  else. Pillow's built-in bitmap font survives only in the dev preview captions,
  which is the right place for it and the reason this package exists at all),
  **external textures/brushes** (don't), and **CC-BY music**
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

A text file is not what a game ships. `screen` draws the same block in this project's
own face -- the last place Pillow's bitmap font was doing a real job, and the one that
mattered, because a credit is a thing a licence depends on being legible. Each row is
*drawn* at the size it is asked for (`--size`, the body em in pixels) in the face its
kind calls for: display for the title, the book face for the credit line, the sans for
the asset list and the italic for the notes. Nothing is enlarged after the fact, so a
bigger screen is a bigger drawing and not a blurrier one -- and that is testable rather
than aspirational, which is why a test asserts that the 48-pixel sheet is *not* the
12-pixel one scaled up.

The size follows the text in both directions, because a credits list is a *scroll*:
page it and the one line the licence actually demands ends up on page four. What is
fixed instead is the line *length*, in characters rather than pixels, so the wrap
cannot move when the output size does. A credit line that fits at 960 and is clipped
at 480 is a credit line that is not there, and the wrapping is therefore a function
that returns rows rather than pixels, so a test can read the words back out. The width
is then the widest *measured* row plus the margin, which is what "sized by its text"
means for a proportional face -- and a test draws the sheet and checks that no ink
reaches the edge of it, which is the same claim looked at on the pixels.

Setting the licence in our own font is a real decision and worth arguing rather than
asserting. The worry is that the one file whose job is to be read ends up set in a face
that can change under it, and that is exactly why the artefact the *gate* reads is
still `credits.txt`: `check` compares the attribution against the text on disk,
character for character, so nothing about the licence depends on the renderer. The
screen is drawn from the same rows, and a test holds every character of those rows
against the alphabet -- a face one character short would raise rather than quietly draw
a box where the credit line should be.

The manifest is worth trusting for the same kind of reason: a test asserts that
every sheet the demos write has a sidecar. Not that the entries are right -- that
none of them are missing.

### The gate: a build is checked, not trusted

A credits screen is a thing a game ships; a *check* is a thing a build runs, and
`credits.check` is the same manifest read back:

```bash
python cli.py credits --manifest out --check
```

Three ways a build can owe somebody something it has not recorded, and they are
worth keeping apart because they are fixed in different places:

* **a source that cannot be credited as written.** A curated entry is missing the
  author, the licence or the URL, so the credit line cannot be assembled whatever
  the build contains. This is a property of the *code*, which is why `check()` with
  no directory at all still reports it;
* **an asset nobody recorded.** A picture with no sidecar is an asset whose
  provenance is unknown — the leak the licensing notes warn about, and the ordinary
  way one happens: nobody steals anything, an exporter forgets its envelope, and a
  year later no one can say where the file came from. A sidecar with no source, no
  licence, or a source that is neither `procedural:…` nor one of the curated ids
  fails the same way; a JSON that is neither a sidecar nor a `format`-tagged data
  file has no envelope at all;
* **an attribution that is not in the credits the build ships.** The file is there
  and the line is not in it. This is the failure mode this module exists for, and
  it happens to honest people: add a second track, regenerate the assets, never
  regenerate the block. The text that is checked is the text *on disk*, not the
  text the code could make — and a build with a curated source and no credits text
  at all is the same violation with nothing to read.

The one exemption is the credits screen itself. It is not an asset we made, it is
the attribution, so it is the one picture the gate does not ask for a sidecar from.
And the gate **reports** rather than repairs, the way `style.validate` does, and
returns a nonzero exit code — because a licence violation should not look like
success to a shell script, and a gate that fixes what it finds is a gate nobody can
trust to be looking.
