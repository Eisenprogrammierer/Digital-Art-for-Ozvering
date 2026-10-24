#!/usr/bin/env python3
"""ozart -- procedural asset generator for a 2-D strategy game.

Every command is a pure function of ``--seed`` plus parameters: run it twice and
you get byte-identical PNGs, which is what makes generated art versionable
alongside the code that produced it.

Usage
-----
    python cli.py demo    --out out --seed 7
    python cli.py icons   --out out/icons.png --seed 3 --count 8
    python cli.py texture --out out/texture.png --seed 3 --size 512
    python cli.py palette --out out/palette.png --seed 3 --count 6
    python cli.py shapes  --out out/shapes.png
    python cli.py arms    --out out/arms.png --seed 7 --count 12
    python cli.py flags   --out out/flags.png --seed 7 --count 12 [--hang bar]  # & poles, staffs
    python cli.py anim    --out out/anim.png  --seed 7 --count 12  # one flag, waving: frames + timeline
    python cli.py figures --out out/figures.png --seed 7 --cycle walk --count 8  # one figure, one gait
    python cli.py blazon  --seed 5 [--spec spec.json] [--out out/arms_one.png]
    python cli.py portrait   --seed 5 --size 384 --out out/portrait.png
    python cli.py portraits  --seed 7 --count 8 --size 256
    python cli.py lineage    --seed 7 --children 6
    python cli.py ui         --seed 7 --count 3 --scale 2
    python cli.py text       --size 20 --out out/text.png
    python cli.py maps       --seed 7 --count 2 --cols 26 --rows 17 --provinces 8 --factions 4
    python cli.py site       --out out/site.png --seed 7 --size 56     # one site, two views
    python cli.py iso        --out out/iso.png  --seed 7 --cols 11 --rows 8  # a board + a hero
    python cli.py studio     --port 8000                  # the same generators, in a browser
    python cli.py coast      --out out/coast-tiles.png --size 20  # & borders.png, in use
    python cli.py credits    --out out --manifest out
    python cli.py credits    --manifest out --png out/credits.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import credits as creditslib  # noqa: E402
from core import noise, palette, render, sdf  # noqa: E402
from core.canvas import Canvas  # noqa: E402
from export import atlas as atlaslib  # noqa: E402
from export import gallery as gallerylib  # noqa: E402
from export import png, sheet as sheetlib  # noqa: E402
from gen.figures import figure as figurelib  # noqa: E402
from gen.heraldry import blazon as heraldry  # noqa: E402
from gen.heraldry import crest as crestlib  # noqa: E402
from gen.heraldry import flags as flaglib  # noqa: E402
from gen.heraldry import tinctures as tinctureslib  # noqa: E402
from gen.icons import badge as icon_badge  # noqa: E402
from gen.icons import livery as icon_livery  # noqa: E402
from gen.icons import motifs as icon_motifs  # noqa: E402
from gen.icons import paint as icon_paint  # noqa: E402
from gen.icons import wear as icon_wear  # noqa: E402
from gen.wardrobe import awards as wardrobe_awards  # noqa: E402
from gen.wardrobe import garment as wardrobe_garment  # noqa: E402
from gen.maps import autotile as map_autotile  # noqa: E402
from gen.maps import campaign as map_campaign  # noqa: E402
from gen.maps import grid as map_grid  # noqa: E402
from gen.maps import iso as map_iso  # noqa: E402
from gen.maps import labels as map_labels  # noqa: E402
from gen.maps import paint as map_paint  # noqa: E402
from gen.maps import politics as map_politics  # noqa: E402
from gen.maps import region as map_region  # noqa: E402
from gen.maps import roads as map_roads  # noqa: E402
from gen.maps import site as map_site  # noqa: E402
from gen.maps import terrain as map_terrain  # noqa: E402
from gen.text import pen as text_pen  # noqa: E402
from gen.text import skeleton as text_skeleton  # noqa: E402
from gen.text import kern as kernlib  # noqa: E402
from gen.text import typeset  # noqa: E402
from gen.textures import atlas as tex_atlas  # noqa: E402
from gen.textures import boards as tex_boards  # noqa: E402
from gen.textures import ground as tex_ground  # noqa: E402
from gen.textures import reaction as tex_reaction  # noqa: E402
from gen.textures import tile as tilelib  # noqa: E402
from gen.textures import truchet as tex_truchet  # noqa: E402
from gen.textures import wang as tex_wang  # noqa: E402
from gen.portraits import accessories as portrait_accessories  # noqa: E402
from gen.portraits import character as portrait_character  # noqa: E402
from gen.portraits import demography as portrait_demography  # noqa: E402
from gen.portraits import dna as portrait_dna  # noqa: E402
from gen.portraits import phenotype as portrait_phenotype  # noqa: E402
from gen.portraits import regalia as portrait_regalia  # noqa: E402
from gen.portraits import portrait as portraitlib  # noqa: E402
from gen.portraits import style as portrait_style  # noqa: E402
from gen.ui import frame as ui_frame  # noqa: E402
from gen.ui import paint as ui_paint  # noqa: E402
from gen.ui import slice as ui_slice  # noqa: E402
from gen.ui import style as ui_style  # noqa: E402
from gen.ui import vector as ui_vector  # noqa: E402

#: A roster of faction names, shared by the heraldry sheet and the icon set -- the
#: same houses on both, which is the point of a faction owning a coat *and* an icon
#: set.  A faction's identity is a name and nothing else, so a demo has to be handed
#: names; pass your own with ``--name`` and the sheets are your game's factions.
FACTIONS = ("Kingdom of Alder", "The Mirefen League", "Brythmark",
            "Silverreach Compact", "Duchy of Halven", "The Ostmark Freeholds",
            "Vale of Craymoor", "House Delvane")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ratio(text) -> tuple:
    """``"3:2"`` (or ``3x2``) into a proportion, for the flag-shaped commands."""
    parts = str(text).replace(":", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"a proportion looks like 3:2, not {text!r}")
    w, h = (int(v) for v in parts)
    if w < 1 or h < 1:
        raise ValueError("a proportion needs two positive numbers")
    return w, h


def _ints(text) -> tuple:
    """``"32,64,128"`` into ``(32, 64, 128)``, for the size sweeps."""
    values = tuple(int(v) for v in str(text).replace(",", " ").split())
    if not values:
        raise ValueError(f"expected one or more sizes, got {text!r}")
    return values


def _rc(text) -> tuple:
    """``"3,4"`` into a ``(row, col)``, for a hex named on the command line."""
    parts = tuple(int(v) for v in str(text).replace(",", " ").split())
    if len(parts) != 2:
        raise ValueError(f"a hex looks like row,col, not {text!r}")
    return parts


def _font(size: int = 13):
    """Pillow's bundled bitmap font, for development previews only.

    Shipping UI text must use an OFL/Apache font or drawn glyphs; the built-in
    font is fine here because it never leaves these contact sheets.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _label(img, xy, text, font=None, dark=(24, 24, 30, 235), plate=(255, 255, 255, 205)):
    """Draw a legible caption on a preview sheet: a plate, then the text."""
    draw = ImageDraw.Draw(img)
    font = font or _font(14)
    x, y = xy
    if plate is not None:
        l, t, r, b = draw.textbbox((x, y), text, font=font)
        draw.rounded_rectangle([l - 6, t - 4, r + 6, b + 4], radius=5, fill=plate)
    draw.text((x, y), text, fill=dark, font=font)


def _grid_pos(canvases, cols: int, gap: int, i: int) -> tuple:
    """Where tile ``i`` lands on a sheet, and how big a cell is.

    The same arithmetic `render.sheet` uses, repeated because the captions have to
    agree with the layout pixel for pixel -- and a sheet is the one place where a
    label a few pixels out is worse than no label.
    """
    cw = max(c.w for c in canvases)
    ch = max(c.h for c in canvases)
    tile = canvases[i]
    r, col = divmod(i, cols)
    return (gap + col * (cw + gap) + (cw - tile.w) // 2,
            gap + r * (ch + gap) + (ch - tile.h) // 2, cw, ch)


def _centre_label(img, box, text, font=None, dark=(24, 24, 30, 235), plate=None):
    """Caption centred inside a pixel box, for labelling generated controls."""
    draw = ImageDraw.Draw(img)
    font = font or _font(12)
    x0, y0, x1, y1 = box
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    xy = (int((x0 + x1 - (r - l)) / 2) - l, int((y0 + y1 - (b - t)) / 2) - t)
    _label(img, xy, text, font=font, dark=dark, plate=plate)


def _dashed(draw, a, b, fill=(226, 62, 122, 215), dash=5, gap=4):
    """A dashed line, used to draw the nine-slice cuts over a real asset."""
    ax, ay = a
    bx, by = b
    length = float(np.hypot(bx - ax, by - ay))
    if length < 1.0:
        return
    ux, uy = (bx - ax) / length, (by - ay) / length
    s = 0.0
    while s < length:
        e = min(s + dash, length)
        draw.line([(ax + ux * s, ay + uy * s), (ax + ux * e, ay + uy * e)], fill=fill, width=1)
        s = e + gap


def _wrap_label(img, xy, text, width, font=None, lines=2, dark=(40, 40, 50, 255)):
    """Draw wrapped caption text on a preview sheet, at most ``lines`` lines."""
    draw = ImageDraw.Draw(img)
    font = font or _font(11)
    words = text.split()
    rows, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if not cur or draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            rows.append(cur)
            cur = w
            if len(rows) == lines:
                break
    if cur and len(rows) < lines:
        rows.append(cur)
    if rows and " ".join(rows) != text:
        rows[-1] = rows[-1].rstrip(",.") + "\u2026"
    x, y = xy
    for k, row in enumerate(rows):
        draw.text((x, y + k * 13), row, fill=dark, font=font)


def _sha256(path: Path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def _report(path: Path) -> Path:
    print(f"  {path}  {path.stat().st_size:>7} B  sha256={_sha256(path)}")
    return path


def _credits_check(directory: Path) -> int:
    """The pre-ship gate, as an exit code: 0 to ship, 1 to read the reasons.

    A build that cannot be credited is a licence violation, and a licence violation
    should not look like success to a shell script.  So the gate says what is wrong
    and returns nonzero, and it repairs nothing -- a gate that fixes what it finds
    is a gate nobody can trust to be looking.
    """
    problems = creditslib.check(directory)
    if not problems:
        print(f"  {directory}: clean -- {len(creditslib.sources())} curated source(s), "
              f"{len(creditslib.manifest(directory))} generated asset(s)")
        return 0
    print(f"  {directory}: not shippable, {len(problems)} problem(s)")
    for why in problems:
        print(f"    - {why}")
    return 1


def _sidecar(out: Path, seed: int, source: str, **extra) -> Path:
    """The two lines that make an asset reproducible, and that credits reads.

    A sheet with a seed and a named entry point *is* the asset: regenerate it and
    the bytes match.  ``core.credits`` walks a build directory for exactly this
    envelope, so writing the sidecar is what puts an asset on the manifest.
    """
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({"seed": seed, "source": source, "license": "MIT", **extra},
                   indent=2) + "\n",
        encoding="utf-8",
    )
    return _report(meta)


def _unit3(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


# --------------------------------------------------------------------------
# icons: a plate with a charge on it, and an atlas to ship the set in
# --------------------------------------------------------------------------
def make_icon(seed: int, size: int = 192, wear: int | None = None) -> Canvas:
    """One complete icon from a seed -- the CLI's smallest door into `gen.icons`.

    This used to be the icon generator.  It is one line now, deliberately: the
    charge vocabulary, the plate and the passes live in the package, and a
    second implementation left behind in the CLI is how the two drift apart.
    ``wear`` renders one of the ten cumulative damage stages; ``None`` keeps the
    seed's own stage, which is zero for a fresh random icon.
    """
    spec = icon_motifs.random_spec(seed)
    if wear is not None:
        spec = icon_wear.stage(spec, wear)
    return icon_paint.draw(spec, size)


def demo_icons(out: Path, seed: int = 3, count: int = 8, size: int = 192, cols: int = 9,
               sweep=(32, 64, 128), atlas: bool = True, factions=None,
               family: str | None = None, wear: bool = False,
               wear_size: int = 48) -> Path:
    """The icon set, seeds drawn from it, one icon at three sizes, and the atlas.

    The first tiles are the *set*: every charge, one plate, one hue family, which is
    what makes a folder of icons look like one game's icons.  Then seeds, which is
    what a generator adds to a set.  Then the same icon at 32, 64 and 128 px,
    *generated* rather than resized -- generating a sweep is only worth doing if the
    small drawing is drawn small, and that is exactly what designing in world units
    buys.

    The last two blocks are the two things a charge cannot say for itself.  A badge
    carries a *count* or a *tier* on the plate's corner -- the shield's tiers as
    pips, a pile of coins as a numeral -- and a faction dresses a whole set in its own
    field: the same coin and the same ear of wheat, eight times, in eight houses'
    colours, with the charge still saying what it is.  A house whose coat is a
    *metal* wears a grey plate and is told apart by its lightness, which is the map's
    rule too; that is what a metal is, not a gap in the table.
    """
    factions = list(factions or FACTIONS)
    if family is not None and family not in icon_motifs.FAMILIES:
        raise ValueError(f"family must be one of {tuple(icon_motifs.FAMILIES)}, not {family!r}")
    kind_names = (list(icon_motifs.FAMILIES[family]) if family
                  else list(icon_motifs.KINDS))
    set_tiles = [(name, icon_paint.draw(icon_motifs.canonical(name), size))
                 for name in kind_names]
    seed_tiles = [(f"seed {seed + i}", make_icon(seed + i, size)) for i in range(count)]
    sweep_tiles = [(f"blade at {s}px", icon_paint.draw(icon_motifs.canonical("blade"), s))
                   for s in sweep]
    badge_tiles = [(f"tier {t}",
                    icon_paint.draw({**icon_motifs.canonical("shield"),
                                     "badge": icon_badge.make(tier=t)}, size))
                   for t in range(1, icon_badge.TIERS + 1)]
    badge_tiles += [(f"count {c}",
                     icon_paint.draw({**icon_motifs.canonical("coin"),
                                      "badge": icon_badge.make(count=c)}, size))
                    for c in (3, 12, 99)]
    faction_tiles = [(name, icon_paint.draw(icon_livery.tint(icon_motifs.canonical(kind), name), size))
                     for name in factions for kind in ("coin", "grain")]
    tiles = set_tiles + seed_tiles + sweep_tiles + badge_tiles + faction_tiles
    canvases = [c for _, c in tiles]

    gap = 12
    sheet = render.sheet(canvases, cols=cols, gap=gap, background=(0.94, 0.94, 0.96))
    sheet.save(out)
    img = png.to_image(sheet)
    for i, (label, tile) in enumerate(tiles):
        tx, ty, _, _ = _grid_pos(canvases, cols, gap, i)
        _label(img, (tx + 4, ty + tile.h + 3), label)
    img.save(out, "PNG", optimize=True)

    _sidecar(out, seed, "procedural:gen.icons.paint",
             set=kind_names,
             family=family,
             families={name: list(members)
                       for name, members in icon_motifs.FAMILIES.items()},
             seeded=[seed + i for i in range(count)],
             sizes=list(sweep),
             tiers=list(range(1, icon_badge.TIERS + 1)),
             counts=[3, 12, 99],
             factions=factions,
             wear=list(icon_wear.LEVELS),
             wear_policy="cumulative")

    # The atlas is what a game actually loads: one image and a table of rectangles.
    # Its density is in the sidecar, because how much of the sheet is sprite rather
    # than gap is the number that decides whether the padding was worth paying for.
    if atlas:
        image, info = atlaslib.build(set_tiles + seed_tiles, padding=2)
        path = out.with_name(f"{out.stem}-atlas.png")
        image.save(path)
        _sidecar(path, seed, "procedural:export.atlas", **info)
        _report(path)

    # The wear sheet: every charge in the family at all ten stages.  It is a
    # separate file because a stage is a variant of an icon, not a second icon on
    # the design sheet -- and because ten rows of damage would bury the pristine
    # set the sheet exists to show.  The atlas is the shippable form: its member
    # names carry both the kind and the stage, so an engine can pick
    # ``rifle:w07`` without a second table.
    if wear:
        wear_tiles = [(f"{kind}:w{level:02d}",
                       icon_paint.draw(icon_motifs.canonical(kind), wear_size,
                                       wear=level))
                      for kind in kind_names for level in icon_wear.LEVELS]
        wear_path = out.with_name(f"{out.stem}-wear.png")
        wear_canvases = [c for _, c in wear_tiles]
        wear_gap = 14
        wear_sheet = render.sheet(wear_canvases, cols=len(icon_wear.LEVELS),
                                  gap=wear_gap, background=(0.94, 0.94, 0.96))
        wear_img = png.to_image(wear_sheet)
        for i, (label, tile) in enumerate(wear_tiles):
            tx, ty, _, _ = _grid_pos(wear_canvases, len(icon_wear.LEVELS),
                                     wear_gap, i)
            # The atlas keeps ``kind:w07`` for engines; the review sheet shortens
            # the caption so it fits under a 48 px tile without touching the row
            # below.
            kind, level = label.split(":")
            _label(wear_img, (tx + 3, ty + tile.h + 1),
                   f"{kind} {level.lstrip('w')}", font=_font(8))
        wear_img.save(wear_path, "PNG", optimize=True)
        _sidecar(wear_path, seed, "procedural:gen.icons.wear",
                 family=family, kinds=kind_names, stages=list(icon_wear.LEVELS),
                 policy="cumulative", size=int(wear_size))
        if atlas:
            wear_atlas, wear_info = atlaslib.build(wear_tiles, padding=1)
            wear_atlas_path = out.with_name(f"{out.stem}-wear-atlas.png")
            wear_atlas.save(wear_atlas_path)
            _sidecar(wear_atlas_path, seed, "procedural:export.atlas",
                     stages=list(icon_wear.LEVELS), **wear_info)
            _report(wear_atlas_path)
        _report(wear_path)
    return _report(out)


def demo_text(out: Path, seed: int = 0, size: int = 20, faces=None,
              width: int = 1400) -> Path:
    """The specimen: two alphabets, five faces, and the metrics in the sidecar.

    Every label on this sheet is set in the face it labels, which is the only honest
    way to show a font: a caption in somebody else's type proves nothing about
    legibility.  The bottom block is the same sentence at six sizes, each *generated*
    at that size rather than scaled -- a 12-pixel line here is a 12-pixel line drawn
    by a 12-pixel pen, which is the claim the package makes.

    The Cyrillic rows are not a second font: they are the Latin letters where the shape
    is the same and the same skeleton under a mark where it is not, which is what makes
    the two alphabets one face.  The accented rows are the same machinery doing the
    same job in the other alphabet.

    The sidecar is the deliverable an engine loads: the guides, the leading, and every
    glyph's advance and ink box, in em units, so layout does not have to re-derive the
    pen to know where a letter ends.
    """
    faces = list(faces or text_pen.names())
    paper = (0.965, 0.957, 0.933)
    ink = (0.13, 0.14, 0.18)
    quiet = (0.46, 0.46, 0.50)
    margin = 44
    gap = 10

    rows = []

    def add(canvas, after=gap):
        rows.append((canvas, after))

    # the masthead
    add(typeset.draw("Digital Art for Ozvering", text_pen.get("display"),
                     em=size * 2.15, pad=0.08, colour=ink), after=6)
    add(typeset.draw("an alphabet you can regenerate -- 214 characters, two "
                     "alphabets, five faces, no font file; the advances are in the "
                     "sidecar",
                     text_pen.get("italic"),
                     em=size * 0.92, pad=0.08, colour=quiet), after=26)

    alphabet = [
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789 .,:;!?()[]{}&@%$#",
        "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ Ё",
        "абвгдежзийклмнопрстуфхцчшщъыьэюя ё",
        "ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ",
        "àáâãäåçèéêëìíîïñòóôõöùúûüýÿ",
        "The quick brown fox jumps over the lazy dog",
        "Съешь же ещё этих мягких французских булок, да выпей чаю",
        "Le vaisseau était à l'ancre; noël approche",
    ]
    for name in faces:
        spec = text_pen.get(name)
        add(typeset.draw(
            f"{name}   weight {spec['weight']:.3f}   contrast {spec['contrast']:.2f}"
            f"   serif {spec['serif']:.2f}   slant {spec['slant']:.2f}"
            f"   width {spec['width']:.2f}",
            {**spec, "weight": 0.052, "contrast": 0.0, "serif": 0.0},
            em=size * 0.62, pad=0.06, colour=quiet), after=4)
        for row in alphabet:
            add(typeset.draw(row, spec, em=size, pad=0.06, colour=ink), after=1)
        add(Canvas(1, 1), after=18)

    # the kerning specimen: one line with the rule on and the same line with it off.
    # A kern is a claim about white, and the only way to check a claim about white is
    # to put the two lines beside each other -- which is also why the pairs here are
    # the notorious ones and not a sentence that happens to contain them.
    add(Canvas(1, 1), after=14)
    add(typeset.draw("Kerning -- the same line with the rule, and without it",
                     {**text_pen.get("sans"), "weight": 0.052},
                     em=size * 0.62, pad=0.06, colour=quiet), after=6)
    kerned = "AVATAR WAVE To Type LT fiy oo nn r. va yo"
    add(typeset.draw(kerned, {**text_pen.get("text"), "kern": 1.0},
                     em=size * 1.9, pad=0.06, colour=ink), after=7)
    add(typeset.draw(kerned, {**text_pen.get("text"), "kern": 0.0},
                     em=size * 1.9, pad=0.06, colour=quiet), after=14)

    # the ligature specimen, on the same principle: the words are the ones a text
    # actually has, and the line is set both ways so the join can be looked at.
    add(Canvas(1, 1), after=10)
    add(typeset.draw("Ligatures -- the same words, joined and apart",
                     {**text_pen.get("sans"), "weight": 0.052},
                     em=size * 0.62, pad=0.06, colour=quiet), after=6)
    joined = "fi fl ff ffi ffl -- office, file, affix, fluffy, waffle"
    add(typeset.draw(joined, {**text_pen.get("text"), "liga": 1.0},
                     em=size * 1.9, pad=0.06, colour=ink), after=7)
    add(typeset.draw(joined, {**text_pen.get("text"), "liga": 0.0},
                     em=size * 1.9, pad=0.06, colour=quiet), after=20)

    # the size sweep: the same sentence, drawn at the size it says it is.  The
    # monoline, because a size ladder is a question about a UI face -- and because the
    # honest answer to "does it read at 11 px" is a pen that was *chosen* for 11 px.
    add(typeset.draw("Sizes, each drawn at its own size",
                     {**text_pen.get("sans"), "weight": 0.052},
                     em=size * 0.62, pad=0.06, colour=quiet), after=4)
    for em in (11, 12, 14, 18, 24, 32, 48):
        add(typeset.draw(f"{em} px   Handgloves, WAVE 1942, quick brown fox",
                         text_pen.get("sans"), em=float(em), pad=0.05, colour=ink),
            after=max(4, int(em * 0.22)))

    # the paragraph: the case the face is actually for
    add(Canvas(1, 1), after=14)
    add(typeset.block([
        "A letter is a skeleton and a pen: the skeleton says which letter it is,",
        "the pen says how thick, how contrasted, how serifed and how slanted.",
        "Change the pen and the same alphabet becomes a UI face or a book face;",
        "change nothing and the same seed is the same picture, byte for byte.",
        "",
        "The second alphabet is not a second font. Where a Cyrillic letter has a",
        "Latin letter's shape it is that letter, aliased; where it is a letter with",
        "a mark it is that letter and a mark, fitted into the letter's own width.",
    ], text_pen.get("text"), em=size * 0.85, pad=0.06, leading=1.40,
        colour=ink), after=0)

    height = sum(c.h + after for c, after in rows) + 2 * margin
    sheet = Canvas(width, height, scale=1.0, center=(0.0, 0.0), background=paper)
    y = margin
    for canvas, after in rows:
        if canvas.w > 1:
            sheet.blit(canvas.rgb, canvas.a, margin, y)
        y += canvas.h + after
    sheet.save(out)

    # The metrics: what an engine needs, and what a reader of the sidecar can verify.
    metric = {}
    for name in faces:
        spec = text_pen.get(name)
        metric[name] = {
            "spec": {k: round(v, 4) for k, v in spec.items()},
            "leading": 1.35,
            "advances": {ch: round(text_pen.advance(ch, spec), 4)
                         for ch in text_skeleton.names()},
            "ink": {ch: [round(v, 4) for v in text_pen.ink_box(ch, spec)]
                    for ch in text_skeleton.names()},
        }
    _sidecar(out, seed, "procedural:gen.text.pen",
             format="ozart/font/1",
             em=1.0, units="em",
             guides={"baseline": 0.0, "x_height": text_skeleton.X_HEIGHT,
                     "cap_height": text_skeleton.CAP_HEIGHT,
                     "ascender": text_skeleton.ASCENDER,
                     "descender": text_skeleton.DESCENDER},
             # Kerning is a rule and not a table, so what a reader of the sidecar gets
             # is the rule's three numbers and the air each face calls normal -- the
             # number every pair in that face is measured against.  A foundry would
             # ship the pairs; a rule that can name its own reference does not need to.
             kerning={"air_floor": kernlib.AIR_FLOOR,
                      "close_max": round(kernlib.CLOSE_MAX, 4),
                      "profile_rows": kernlib.PROFILE_ROWS,
                      "reference": {name: round(kernlib.reference(text_pen.get(name)), 4)
                                    for name in faces}},
             # A ligature is a glyph and not a character, so the sidecar lists the
             # *sequences* a text is set with -- the join's own advance is in the table
             # above like every other glyph's, and an engine that lays a line out needs
             # the substitution and nothing else.
             ligatures={"".join(parts): ch
                        for ch, parts in text_skeleton.LIGATURES.items()},
             faces=metric)
    return _report(out)


# --------------------------------------------------------------------------
# wardrobe and decorations: the outfit as an object, not as a bust
# --------------------------------------------------------------------------
def _garment_specs(seed: int, count: int, cut=None, pattern=None,
                   cloth=None, trim=None, metal=None) -> list:
    pins = {"pattern": pattern, "cloth": cloth, "trim": trim, "metal": metal}
    pins = {k: v for k, v in pins.items() if v}
    specs = []
    if cut is None:
        base = wardrobe_garment.random_spec(seed)
        for i, name in enumerate(wardrobe_garment.CUTS):
            spec = wardrobe_garment.random_spec(seed + i * 17, cut=name)
            spec.update({k: base[k] for k in ("cloth", "trim", "metal")})
            spec.update(pins)
            specs.append(wardrobe_garment.clamp(spec))
    for i in range(count):
        spec = wardrobe_garment.random_spec(seed + 101 + i * 23, cut=cut)
        spec.update(pins)
        specs.append(wardrobe_garment.clamp(spec))
    return specs


def demo_wardrobe(out: Path, seed: int = 3, count: int = 6, size: int = 256,
                  cols: int = 5, cut=None, pattern=None, cloth=None,
                  trim=None, metal=None) -> Path:
    """A sheet of garment cuts: the canonical set, then seeded variants.

    The canonical row is one wardrobe: the same cloth, trim and metal across every
    cut, because a set should look like it came from one tailor.  The seeded tiles
    are the generator: each one draws its own palette, collar and pattern.  Pinning
    a colour applies it to both blocks, which is how a game gets a faction's coat
    without abandoning the cut vocabulary.
    """
    count = max(1, int(count))
    specs = _garment_specs(seed, count, cut=cut, pattern=pattern,
                           cloth=cloth, trim=trim, metal=metal)
    tiles = []
    canon = len(wardrobe_garment.CUTS) if cut is None else 0
    for i, spec in enumerate(specs):
        label = spec["cut"] if i < canon else f"{spec['cut']} {seed + 101 + (i - canon) * 23}"
        tiles.append((label, wardrobe_garment.render(spec, size)))
    canvases = [c for _, c in tiles]
    # A caption sits under its tile; the gap has to be tall enough for the
    # caption or the last row's labels are cut off by the sheet's own margin.
    gap = 26
    sheet = render.sheet(canvases, cols=cols, gap=gap, background=(0.94, 0.94, 0.96))
    sheet.save(out)
    img = png.to_image(sheet)
    for i, (label, tile) in enumerate(tiles):
        tx, ty, _, _ = _grid_pos(canvases, cols, gap, i)
        _label(img, (tx + 4, ty + tile.h + 3), label)
    img.save(out, "PNG", optimize=True)
    _sidecar(out, seed, "procedural:gen.wardrobe.garment",
             cuts=list(wardrobe_garment.CUTS),
             seeded=[seed + 101 + i * 23 for i in range(count)],
             specs=[wardrobe_garment.clamp(s) for s in specs])
    return _report(out)


def demo_decorations(out: Path, seed: int = 3, count: int = 6, size: int = 256,
                     cols: int = 5, kind=None, metal=None, motif=None,
                     bars=None, rosette: bool = False) -> Path:
    """A sheet of awards: the kind families, then seeded decorations.

    An award is read ribbon first, metal second, motif third.  The canonical set
    therefore keeps the ribbon and metal constant while the *kind* changes; the
    seeded block is where the generator varies all three.
    """
    count = max(1, int(count))
    pins = {"metal": metal, "motif": motif}
    if bars is not None:
        pins["bars"] = int(bars)
    if rosette:
        pins["rosette"] = True
    pins = {k: v for k, v in pins.items() if v is not None and v != ""}
    specs = []
    if kind is None:
        base = wardrobe_awards.random_spec(seed)
        for i, name in enumerate(wardrobe_awards.KINDS):
            spec = wardrobe_awards.random_spec(seed + i * 19, kind=name)
            spec.update({"ribbon": base["ribbon"], "ribbon2": base["ribbon2"],
                         "metal": base["metal"]})
            spec.update(pins)
            specs.append(wardrobe_awards.clamp(spec))
    for i in range(count):
        spec = wardrobe_awards.random_spec(seed + 211 + i * 29, kind=kind)
        spec.update(pins)
        specs.append(wardrobe_awards.clamp(spec))
    tiles = []
    canon = len(wardrobe_awards.KINDS) if kind is None else 0
    for i, spec in enumerate(specs):
        label = spec["kind"] if i < canon else f"{spec['kind']} {seed + 211 + (i - canon) * 29}"
        tiles.append((label, wardrobe_awards.render(spec, size)))
    canvases = [c for _, c in tiles]
    # A caption sits under its tile; the gap has to be tall enough for the
    # caption or the last row's labels are cut off by the sheet's own margin.
    gap = 26
    sheet = render.sheet(canvases, cols=cols, gap=gap, background=(0.94, 0.94, 0.96))
    sheet.save(out)
    img = png.to_image(sheet)
    for i, (label, tile) in enumerate(tiles):
        tx, ty, _, _ = _grid_pos(canvases, cols, gap, i)
        _label(img, (tx + 4, ty + tile.h + 3), label)
    img.save(out, "PNG", optimize=True)
    _sidecar(out, seed, "procedural:gen.wardrobe.awards",
             kinds=list(wardrobe_awards.KINDS),
             seeded=[seed + 211 + i * 29 for i in range(count)],
             specs=[wardrobe_awards.clamp(s) for s in specs])
    return _report(out)


def demo_shapes(out: Path, size: int = 160, cols: int = 3) -> Path:
    """Reference sheet of every SDF primitive and operator, shaded identically."""
    star = lambda x, y: sdf.star(x, y, 5, 0.72, 0.32)  # noqa: E731
    fields = [
        ("circle", lambda x, y: sdf.circle(x, y, 0.76)),
        ("box r=0.10", lambda x, y: sdf.box(x, y, 0.68, 0.52, 0.10)),
        ("hexagon", lambda x, y: sdf.regular_polygon(x, y, 6, 0.80)),
        ("star", star),
        ("capsule", lambda x, y: sdf.segment(x, y, (-0.66, -0.48), (0.66, 0.48), 0.26)),
        ("ring", lambda x, y: sdf.ring(x, y, 0.62, 0.15)),
        ("smooth_union", lambda x, y: sdf.smooth_union(
            sdf.circle(x, y, 0.48, -0.26, 0.0), sdf.box(x, y, 0.36, 0.36, 0.10, 0.24, 0.16), 0.28
        )),
        ("smooth_difference", lambda x, y: sdf.smooth_difference(
            sdf.box(x, y, 0.72, 0.72, 0.14), sdf.circle(x, y, 0.42, 0.0, 0.06), 0.16
        )),
        ("outline", sdf.outline(star, 0.055)),
        ("repeated", sdf.repeated(lambda x, y: sdf.circle(x, y, 0.15), 0.42)),
        ("mirrored", sdf.mirrored(lambda x, y: sdf.box(x - 0.36, y, 0.28, 0.11, 0.05), axis="x")),
        ("transformed", sdf.transformed(
            lambda x, y: sdf.box(x, y, 0.60, 0.26, 0.08),
            np.array([[0.906, -0.423, 0.10], [0.423, 0.906, -0.05], [0.0, 0.0, 1.0]]),
        )),
    ]

    colors = palette.distinct_palette(len(fields), seed=11, min_de=0.13, L=0.66, C=0.15)
    cells = []
    for i, (name, field) in enumerate(fields):
        c = Canvas(size, size, scale=size / 2.0 * 0.80)
        X, Y = c.grid()
        d = field(X, Y)
        c.over((0.0, 0.0, 0.0), render.drop_shadow(
            render.coverage(d, c.dx), c.scale, offset=(0.0, -0.045), sigma=0.025, opacity=0.3
        ))
        c.over(render.shade(colors[i], d, dx=c.dx, bevel=0.09), render.coverage(d, c.dx))
        cells.append(c)

    sheet = render.sheet(cells, cols=cols, gap=10, background=(0.94, 0.94, 0.96))
    sheet.save(out)

    img = png.to_image(sheet)
    cell = size
    gap = 10
    for i, (name, _) in enumerate(fields):
        r, col = divmod(i, cols)
        _label(img, (gap + col * (cell + gap) + 7, gap + r * (cell + gap) + cell - 23), name)
    img.save(out, "PNG", optimize=True)
    _sidecar(out, 0, "procedural:core.sdf reference sheet",
             fields=[name for name, _ in fields])
    return _report(out)


def demo_texture(out: Path, seed: int = 3, size: int = 256) -> Path:
    """Four materials from one toolkit: cloud, marble, wood grain, cracked stone.

    Each variant returns a normalised field plus a height field; the ramp gives
    it colour and the height field gives it relief.  The lighting uses the same
    direction as the icons, which is what makes a texture and a sprite read as
    parts of the same world.
    """
    s = size / 512.0

    def cloud():
        c = Canvas(size, size, scale=1.0, y_up=False)
        X, Y = c.grid()
        wx, wy = noise.warp(X, Y, seed=seed, amp=30.0 * s, freq=0.010 / s, octaves=4)
        v = noise.fbm(wx, wy, seed=seed, freq=0.022 / s, octaves=6)
        cells = noise.worley(X, Y, seed=seed + 31, freq=0.011 / s)
        return np.clip(v * 0.85 + (1.0 - cells) * 0.15, 0.0, 1.0), (v - 0.5) * size * 0.014

    def marble():
        c = Canvas(size, size, scale=1.0, y_up=False)
        X, Y = c.grid()
        wx, wy = noise.warp(X, Y, seed=seed + 5, amp=44.0 * s, freq=0.006 / s, octaves=4)
        broad = noise.fbm(X, Y, seed=seed + 9, freq=0.003 / s, octaves=4)
        t = 0.5 + 0.5 * np.sin((wy * 0.028 / s + 7.0 * broad) * np.pi)
        return t, (t - 0.5) * size * 0.010

    def wood():
        c = Canvas(size, size, scale=1.0, y_up=False)
        X, Y = c.grid()
        r = np.hypot(X, Y * 2.2 * s)  # elliptical rings = end grain
        swirl = noise.fbm(X, Y, seed=seed + 17, freq=0.012 / s, octaves=3) - 0.5
        t = 0.5 + 0.5 * np.sin((r * 0.020 / s + 4.0 * swirl) * np.pi)
        return t, (t - 0.5) * size * 0.012

    def stone():
        c = Canvas(size, size, scale=1.0, y_up=False)
        X, Y = c.grid()
        freq = 0.026 / s
        edge = noise.worley_edges(X, Y, seed=seed + 23, freq=freq)  # 0 on borders
        grit = noise.fbm(X, Y, seed=seed + 29, freq=0.09 / s, octaves=4)
        crack = np.clip(1.0 - edge * 9.0, 0.0, 1.0)  # thin walls, wide cell faces
        t = np.clip(0.80 - 0.66 * crack + 0.10 * (grit - 0.5), 0.0, 1.0)
        return t, (np.minimum(edge, 0.075) - 0.075) * size * 0.30

    rng = np.random.default_rng(seed)
    hue = float(rng.random())
    variants = [
        ("fbm + domain warp", cloud, [palette.color(0.24, 0.06, hue),
                                      palette.color(0.52, 0.14, (hue + 0.03) % 1.0),
                                      palette.color(0.86, 0.10, (hue + 0.07) % 1.0)]),
        ("marble veining", marble, [palette.color(0.26, 0.03, (hue + 0.5) % 1.0),
                                    palette.color(0.94, 0.02, (hue + 0.5) % 1.0)]),
        ("wood end-grain", wood, [palette.color(0.38, 0.09, (hue + 0.08) % 1.0),
                                  palette.color(0.74, 0.08, (hue + 0.08) % 1.0)]),
        ("cracked stone", stone, [palette.color(0.32, 0.02, (hue + 0.25) % 1.0),
                                  palette.color(0.88, 0.03, (hue + 0.25) % 1.0)]),
    ]

    tiles = []
    for _, make, ramp in variants:
        t, h = make()
        n = render.height_normals(h, dx=1.0, y_up=False)
        lam = np.clip(n @ _unit3((-0.5, 0.55, 0.75)), 0.0, 1.0)
        c = Canvas(size, size, scale=1.0, y_up=False)
        c.rgb[:] = np.clip(palette.ramp_map(ramp, t) * (0.74 + 0.44 * lam)[..., None], 0.0, 1.0)
        c.a[:] = 1.0
        tiles.append(c)

    gap = 10
    sheet = render.sheet(tiles, cols=2, gap=gap, background=(0.94, 0.94, 0.96))
    sheet.save(out)

    img = png.to_image(sheet)
    for i, (name, _, _) in enumerate(variants):
        r, col = divmod(i, 2)
        _label(img, (gap + col * (size + gap) + 9, gap + r * (size + gap) + 8), name)
    img.save(out, "PNG", optimize=True)
    _sidecar(out, seed, "procedural:core.noise materials",
             materials=[name for name, _, _ in variants])
    return _report(out)


def demo_palette(out: Path, seed: int = 3, count: int = 6, cell=(150, 96), cols: int = 3, gap: int = 10) -> Path:
    """Swatch sheet plus a JSON dump -- the start of the asset/licence registry."""
    rng = np.random.default_rng(seed)
    colors = palette.random_palette(rng, n=count)
    cw, ch = cell
    rows = (count + cols - 1) // cols
    sheet = Canvas(cols * cw + (cols + 1) * gap, rows * ch + (rows + 1) * gap, background=(0.94, 0.94, 0.96))

    placed = []
    for i, col in enumerate(colors):
        r, cc = divmod(i, cols)
        x = gap + cc * (cw + gap)
        y = gap + r * (ch + gap)
        tile = Canvas(cw, ch, background=col)
        sheet.blit(tile.rgb, tile.a, x, y)
        placed.append((x, y, col))

    img = png.to_image(sheet)
    for x, y, col in placed:
        _label(img, (x + 9, y + ch - 24), palette.srgb_to_hex(col))
    img.save(out, "PNG", optimize=True)

    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "seed": seed,
                "source": "procedural:core.palette.random_palette",
                "license": "MIT",
                "colors": [
                    {
                        "hex": palette.srgb_to_hex(c),
                        "oklch": [round(float(v), 4) for v in palette.srgb_to_oklch(c)],
                    }
                    for c in colors
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _report(meta)
    return _report(out)


def demo_arms(out: Path, seed: int = 7, count: int = 12, size: int = 190, cols: int = 4,
              shield: str | None = None) -> Path:
    """Contact sheet of generated coats of arms, captioned with their blazons.

    The sidecar ``.json`` records the seed, the spec and both blazon languages for
    every tile: an arm is a data structure first and a picture second, and the
    text is what makes a generated coat reviewable.
    """
    specs = [heraldry.random_arms(seed=seed + i, shield=shield) for i in range(count)]
    tiles = [heraldry.render(spec, size=size) for spec in specs]

    gap = 36
    sheet = render.sheet(tiles, cols=cols, gap=gap, background=(0.94, 0.94, 0.96))
    sheet.save(out)
    img = png.to_image(sheet)
    for i, spec in enumerate(specs):
        r, col = divmod(i, cols)
        x = gap + col * (size + gap)
        y = gap + r * (size + gap) + size + 4
        _wrap_label(img, (x, y), heraldry.blazon_text(spec, "en"), width=size)
    img.save(out, "PNG", optimize=True)

    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "seed": seed,
                "source": "procedural:gen.heraldry.blazon.random_arms",
                "license": "MIT",
                "arms": [
                    {
                        "seed": seed + i,
                        "blazon_en": heraldry.blazon_text(spec, "en"),
                        "blazon_ru": heraldry.blazon_text(spec, "ru"),
                        "warnings": heraldry.validate(spec),
                        "spec": spec,
                    }
                    for i, spec in enumerate(specs)
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _report(meta)
    return _report(out)


#: What a cloth in the flags sheet can hang from, in the order ``mixed`` walks them.
HANGINGS = ("pole", "bar", "staff")

#: How far the demo's texture mip chain goes.  ``None`` is every level the tiles
#: support -- the whole chain, down to a sheet of one-texel tiles -- because that is
#: where the price of the margin rule actually shows, and a chain that stopped before
#: it would be a chain that stopped before the interesting number.
TILE_MIPS = None


def demo_flags(out: Path, seed: int = 7, count: int = 12, size: int = 200, cols: int = 4,
               ratio=(3, 2), shape: str = "rectangle", wave: bool = True,
               hang: str = "pole") -> Path:
    """Contact sheet of generated flags, described vexillologically.

    The flags hang by default -- from a pole, because a flag that hangs hangs from
    something: the pole stands the cloth up, anchors its hoist and lets the fly
    trail.  One sheet is asked for cloth with ``--flat``, because a flag on a pole is
    what the asset is and a flat flag is the special case.

    ``hang`` chooses what the cloth hangs from, and a bar is not a pole: a gonfalon
    hung from a crossbar has its folds deepening *downwards* from the bar rather than
    growing along the wave, and a staff with a crossbar across it is the classic form
    that carries both.  ``mixed`` walks the three in turn, which is what the ``demo``
    sheet does -- the three hangings are one cloth, and putting them beside each other
    is the only way to see that the hanging is all that differs.
    """
    if hang not in HANGINGS + ("mixed", "none"):
        raise ValueError(f"a cloth hangs from one of {', '.join(HANGINGS)}, "
                         f"one of them in turn, or nothing -- not {hang!r}")

    def holder(i: int) -> dict:
        if hang == "none":
            return {}
        kind = HANGINGS[i % len(HANGINGS)] if hang == "mixed" else hang
        return {"pole": {"pole": True},
                "bar": {"bar": True},
                "staff": {"pole": True, "bar": True}}[kind]

    specs = [flaglib.random_flag(seed=seed + i, ratio=ratio, shape=shape, wave=wave,
                                 **holder(i))
             for i in range(count)]
    tiles = [flaglib.render(spec, size=size) for spec in specs]

    gap = 36
    # Flags are flat and argent is near-white, so the sheet back is a mid grey
    # rather than the pale one the shields sit on.
    sheet = render.sheet(tiles, cols=cols, gap=gap, background=(0.76, 0.78, 0.83))
    sheet.save(out)
    img = png.to_image(sheet)
    for i, spec in enumerate(specs):
        r, col = divmod(i, cols)
        x = gap + col * (size + gap)
        y = gap + r * (size + gap) + size + 4
        _wrap_label(img, (x, y), flaglib.describe(spec, "en"), width=size)
    img.save(out, "PNG", optimize=True)

    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "seed": seed,
                "source": "procedural:gen.heraldry.flags.random_flag",
                "license": "MIT",
                "flags": [
                    {
                        "seed": seed + i,
                        "description_en": flaglib.describe(spec, "en"),
                        "description_ru": flaglib.describe(spec, "ru"),
                        "spec": spec,
                    }
                    for i, spec in enumerate(specs)
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _report(meta)
    return _report(out)


def demo_anim(out: Path, seed: int = 7, count: int = 12, size: int = 200,
              cols: int = 4, fps: float = 12.0, padding: int = 2) -> Path:
    """Sprite sheet of one flag, waving: the frames, the table and the timeline.

    An animation is not a set of pictures of a flag.  It is *one* flag at a sequence
    of phases, and the phases are all that differ between the frames -- which is why
    the artwork is `flags.waving` and the only thing this function decides is how the
    frames are delivered.  That is a sheet without captions and with a timeline: an
    engine cuts it up with the table and plays it at `fps`.

    The cloth is asked for *one* harmonic, and that is a decision this command makes
    rather than a default it inherits: the default second harmonic is not commensurate
    with the first, so a folded flag returns after ten turns rather than one
    (`cloth.period`), and a loop cut at one turn would stutter once per loop.  One
    harmonic makes the loop close exactly -- the last frame is followed by the first
    because the field is a sinusoid -- and ten flaps per loop is not what a flag does.

    The frames are drawn one at a time at their own phase, exactly as every other
    asset here is drawn at its own size, so nothing about the sheet is a resample of
    anything.  The background is transparent rather than a paper colour, and the
    padding between frames is clear: a sheet is cut up along its table, and a pixel
    the file says is not there is a pixel no filter can pull into a frame.
    """
    spec = flaglib.random_flag(seed=seed, wave={"harmonic": 0.0}, pole=True)
    frames, phases = flaglib.waving(spec, count=count, size=size)
    sheet, info = sheetlib.build(frames, cols=cols, padding=padding, fps=fps,
                                 loop=True, params=phases, background=None)
    sheet.save(out)
    _sidecar(out, seed, "procedural:gen.heraldry.flags.waving", spec=spec, **info)
    return _report(out)


def demo_figures(out: Path, seed: int = 7, count: int = 8, size: int = 96,
                 cols: int = 8, fps: float = 12.0, cycle: str = "walk",
                 facing: str = "right") -> Path:
    """Sprite sheet of one figure's gait: the frames, the table and the timeline.

    A figure is not a set of drawings of a person.  It is *one* body -- proportions,
    palette, a gait -- at a sequence of phases, and the phases are all that differ
    between the frames.  So the artwork is ``figure.frames`` and this function decides
    only how the frames are delivered, exactly as ``anim`` does for the flag: a sheet
    without captions, the rectangles in the sidecar, and a timeline an engine plays.

    The cycle is asked for by name and the loop closes by construction -- every term
    of a gait is a sine or a window of one, so ``2*pi`` is a period and not a measured
    guess -- which is why the sheet can say ``loop: true`` and mean it.  The frame is a
    cell and not a trim: every cell is the same size and the figure is centred in it, so
    a frame the figure barely enters still has the same rectangle as one it fills.
    """
    if cycle not in figurelib.cycles():
        raise SystemExit(f"no such cycle: {cycle!r} (have {', '.join(figurelib.cycles())})")
    if facing not in figurelib.FACINGS:
        raise SystemExit(f"a figure faces {', '.join(figurelib.FACINGS)}; not {facing!r}")
    frames, phases = figurelib.frames(seed=seed, cycle=cycle, count=count, size=size,
                                      facing=facing)
    sheet, info = sheetlib.build(frames, cols=cols, padding=2, fps=fps, loop=True,
                                 params=phases, background=None)
    sheet.save(out)
    _sidecar(out, seed, "procedural:gen.figures.figure.frames", cycle=cycle,
             facing=facing, tiles=int(count), figure=figurelib.describe(seed), **info)
    return _report(out)


def demo_blazon(seed: int = 5, spec_path: Path | None = None, out: Path | None = None,
                shield: str | None = None) -> dict:
    """Print a coat's blazon in both languages, and optionally render and dump it.

    This is the command that proves the round trip: the spec is the asset, the
    blazon is a *view* of it, and either can be handed to a person.
    """
    if spec_path is not None:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        seed = int(spec.get("seed", seed))
    else:
        spec = heraldry.random_arms(seed=seed, shield=shield)

    print(f"seed {seed}")
    print(f"  EN  {heraldry.blazon_text(spec, 'en')}")
    print(f"  RU  {heraldry.blazon_text(spec, 'ru')}")
    warnings = heraldry.validate(spec)
    print(f"  rule of tincture: {'ok' if not warnings else '; '.join(warnings)}")

    if out is not None:
        img = png.to_image(heraldry.render(spec, size=384))
        img.save(out, "PNG", optimize=True)
        _report(out)
        meta = out.with_suffix(".json")
        meta.write_text(
            json.dumps({
                "seed": seed,
                "source": "procedural:gen.heraldry.blazon.random_arms",
                "license": "MIT",
                "spec": spec,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _report(meta)
    return spec


def _apply_demography(genes: dict, sex: str | None = None, age: str | None = None,
                      climate: str | None = None, culture: str | None = None) -> dict:
    """Pin any of the demographic genes on a genome, without making the others fixed."""
    for key, value, allowed in (("sex", sex, portrait_demography.SEXES),
                                ("age", age, portrait_demography.AGES),
                                ("climate", climate, portrait_demography.CLIMATES),
                                ("culture", culture, portrait_demography.CULTURES)):
        if value is None:
            continue
        if value not in allowed:
            raise ValueError(f"unknown {key} {value!r}")
        genes[key] = value
    return genes


def _parse_gene_pairs(pairs) -> dict:
    """``--gene face_shape=heart --gene skin_melanin=-0.4`` into a dict.

    This is the escape hatch that keeps the command line usable as the genome
    grows: the curated flags cover the categories a designer reaches for, and
    ``--gene`` reaches every numeric or categorical axis without another parser
    entry.  Validation still goes through :func:`gen.portraits.dna.decode`, so a
    typo is an error rather than a silently ignored override.
    """
    out = {}
    for item in pairs or ():
        key, sep, value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"a gene override looks like key=value, not {item!r}")
        if key not in portrait_dna.GENES:
            raise ValueError(f"unknown portrait gene {key!r}")
        if key in portrait_dna.CATEGORICAL or key in portrait_dna.DEMOGRAPHIC:
            out[key] = value.strip()
        elif key in portrait_dna.COLOUR_GENES:
            palette.hex_to_srgb(value.strip())
            out[key] = value.strip()
        else:
            out[key] = portrait_dna.quantise(float(value))
    return out


def _apply_overrides(genes: dict, overrides: dict | None) -> dict:
    """Pin any gene on a genome without replacing the rest of it."""
    if overrides:
        unknown = [key for key in overrides if key not in portrait_dna.GENES]
        if unknown:
            raise ValueError(f"unknown portrait genes: {', '.join(unknown)}")
        genes.update(overrides)
    return genes


def _phenotype_overrides(args) -> dict:
    """The curated phenotype flags plus every ``--gene`` pair, as one dict."""
    keys = (*portrait_phenotype.CATEGORIES, *portrait_regalia.CATEGORIES,
            *portrait_accessories.CATEGORIES)
    out = {key: getattr(args, key) for key in keys
           if getattr(args, key, None) is not None}
    for gene, attr in (("skin", "skin_color"), ("hair", "hair_color"),
                       ("iris", "iris_color"), ("cloth", "cloth_color")):
        value = getattr(args, attr, None)
        if value:
            palette.hex_to_srgb(value)
            out[gene] = value
    out.update(_parse_gene_pairs(getattr(args, "gene", None)))
    return out


def _add_phenotype_args(parser) -> None:
    for key, choices in (*portrait_phenotype.CATEGORIES.items(),
                         *portrait_regalia.CATEGORIES.items(),
                         *portrait_accessories.CATEGORIES.items()):
        parser.add_argument("--" + key.replace("_", "-"), default=None,
                            choices=list(choices), help=f"{key.replace('_', ' ')} family")
    for gene, attr, label in (
            ("skin", "skin_color", "base skin colour"),
            ("hair", "hair_color", "base hair colour"),
            ("iris", "iris_color", "base iris colour"),
            ("cloth", "cloth_color", "base cloth colour")):
        parser.add_argument("--" + attr.replace("_", "-"), "--" + attr.replace("_", "-").replace("color", "colour"),
                            dest=attr, default=None, metavar="#RRGGBB",
                            help=f"pin the {label} directly")
    parser.add_argument("--gene", action="append", default=[], metavar="KEY=VALUE",
                        help="pin any genome axis, e.g. skin_melanin=-0.4 or face_shape=heart; "
                             "repeat for more than one")


def demo_portraits(out: Path, seed: int = 7, count: int = 8, size: int = 256, cols: int = 4,
                   sex: str | None = None, age: str | None = None,
                   climate: str | None = None, culture: str | None = None,
                   overrides: dict | None = None) -> Path:
    """Contact sheet of generated portraits, captioned with their genome.

    The sidecar ``.json`` is the point: every tile carries the one-line DNA
    string that would reproduce it exactly, so a sheet of avatars is also a
    browsable catalogue of the genomes that produced them.  The demographic
    flags pin one gene for the whole sheet; leave them out to sample the range.
    ``overrides`` pins any phenotype or legacy gene on every tile.
    """
    genomes = [_apply_overrides(
        _apply_demography(portrait_dna.random_genes(seed + i), sex=sex, age=age,
                          climate=climate, culture=culture), overrides)
               for i in range(count)]
    genomes = [portrait_dna.decode(portrait_dna.encode(g)) for g in genomes]
    tiles = [portraitlib.render(g, size=size) for g in genomes]

    gap = 24
    sheet = render.sheet(tiles, cols=cols, gap=gap, background=(0.90, 0.90, 0.93))
    sheet.save(out)
    img = png.to_image(sheet)
    for i, g in enumerate(genomes):
        r, col = divmod(i, cols)
        x = gap + col * (size + gap)
        y = gap + r * (size + gap) + size + 3
        _wrap_label(img, (x, y), f"{seed + i}: {portrait_dna.describe(g)}", width=size, lines=2)
    img.save(out, "PNG", optimize=True)

    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "seed": seed,
                "source": "procedural:gen.portraits.dna.random_genes",
                "license": "MIT",
                "demography": {"sex": sex, "age": age, "climate": climate, "culture": culture},
                "portraits": [
                    {
                        "seed": seed + i,
                        "dna": portrait_dna.encode(g),
                        "description": portrait_dna.describe(g),
                        "figure": portrait_character.scene(g),
                        "genes": g,
                    }
                    for i, g in enumerate(genomes)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _report(meta)
    return _report(out)


def demo_lineage(out: Path, seed: int = 7, size: int = 208, children: int = 6) -> Path:
    """Two parents and their children: what a heritable genome buys.

    Colours blend, shapes average and drift, and the categories are drawn -- so a
    strategy game can populate a dynasty by crossing genomes instead of by
    inventing a face per character.
    """
    a = portrait_dna.random_genes(seed)
    b = portrait_dna.random_genes(seed + 1000)
    kids = [portrait_dna.breed(a, b, seed=seed + 10 * k) for k in range(children)]

    tiles = [portraitlib.render(g, size=size) for g in [a, b] + kids]
    gap = 22
    cols = 4
    sheet = render.sheet(tiles, cols=cols, gap=gap, background=(0.90, 0.90, 0.93))
    sheet.save(out)
    img = png.to_image(sheet)
    captions = ["parent A", "parent B"] + [f"child {k + 1}" for k in range(children)]
    for i, (g, cap) in enumerate(zip([a, b] + kids, captions)):
        r, col = divmod(i, cols)
        x = gap + col * (size + gap)
        y = gap + r * (size + gap) + size + 3
        _wrap_label(img, (x, y), f"{cap}: {portrait_dna.describe(g)}", width=size, lines=2)
    img.save(out, "PNG", optimize=True)

    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "seed": seed,
                "source": "procedural:gen.portraits.dna.breed",
                "license": "MIT",
                "parents": [portrait_dna.encode(a), portrait_dna.encode(b)],
                "children": [portrait_dna.encode(g) for g in kids],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _report(meta)
    return _report(out)


def portrait_one(out: Path, seed: int | None = None, dna: str | None = None, size: int = 384,
                 style: str | None = None, hair: str | None = None,
                 expression: str | None = None, sex: str | None = None,
                 age: str | None = None, climate: str | None = None,
                 culture: str | None = None, overrides: dict | None = None) -> Path:
    """Render one portrait and print its genome.

    The command is a passport photo by default: the head is straight and the face
    neutral, whatever the genome's ``turn`` and ``expression`` genes say.  Passing
    ``--expression`` opts into the separate posed study, so the flag is not a
    silent no-op.
    """
    if dna is not None:
        genes = portrait_dna.decode(dna)
        if seed is None:
            seed = portrait_dna.seed_of(genes)
    else:
        genes = portrait_dna.random_genes(0 if seed is None else seed)
    # Explicit flags win over the genome, which is how a designer pins a style
    # while keeping everything else random.
    if style:
        genes["style"] = style
    if hair:
        genes["hair_style"] = hair
    if expression:
        genes["expression"] = expression
    _apply_demography(genes, sex=sex, age=age, climate=climate, culture=culture)
    _apply_overrides(genes, overrides)
    genes = portrait_dna.decode(portrait_dna.encode(genes))  # validate and re-clamp

    text = portrait_dna.encode(genes)
    print(f"seed {seed}")
    print(f"  {portrait_dna.describe(genes)}")
    print(f"  {text}")
    if expression is not None:
        print(f"  posed study: expression={expression!r}, turn from the genome")
        img = png.to_image(portraitlib.render_posed(
            genes, size=size, expression_name=expression))
    else:
        img = png.to_image(portraitlib.render(genes, size=size))
    img.save(out, "PNG", optimize=True)
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({"seed": seed, "dna": text,
                    "figure": portrait_character.scene(genes),
                    "genes": genes}, indent=2) + "\n",
        encoding="utf-8",
    )
    _report(meta)
    return _report(out)


def character_one(out: Path, seed: int | None = None, dna: str | None = None,
                  size: int = 256, cycle: str = "idle", facing: str = "down",
                  sex: str | None = None, age: str | None = None,
                  climate: str | None = None, culture: str | None = None,
                  overrides: dict | None = None) -> Path:
    """Render the full-body figure of one portrait genome.

    The DNA is the input, so the portrait and the figure are the same person: the
    body seed is the sitter's own hash, and the palette and visible garment come
    from the same genes.  ``--dna`` makes the pairing exact; a seed makes a new
    genome, which is then printed and stored in the sidecar.
    """
    if dna is not None:
        genes = portrait_dna.decode(dna)
        if seed is None:
            seed = portrait_dna.seed_of(genes)
    else:
        seed = 0 if seed is None else int(seed)
        genes = portrait_dna.random_genes(seed)
    _apply_demography(genes, sex=sex, age=age, climate=climate, culture=culture)
    _apply_overrides(genes, overrides)
    genes = portrait_dna.decode(portrait_dna.encode(genes))  # validate and re-clamp

    text = portrait_dna.encode(genes)
    print(f"seed {seed}")
    print(f"  {portrait_dna.describe(genes)}")
    print(f"  {text}")
    img = png.to_image(portrait_character.render(
        genes, size=size, cycle=cycle, facing=facing))
    img.save(out, "PNG", optimize=True)
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({"seed": seed, "dna": text,
                    "figure": portrait_character.describe(
                        genes, cycle=cycle, facing=facing),
                    "genes": genes}, indent=2) + "\n",
        encoding="utf-8",
    )
    _report(meta)
    return _report(out)


# --------------------------------------------------------------------------
# UI kit: panels, frames and buttons, generated as nine-slice assets
# --------------------------------------------------------------------------
def _ui_part(spec: dict, size, scale: int = 2, **kw) -> Canvas:
    """Render one part from its *design* size; ``scale`` is the pixel density."""
    return ui_paint.draw(spec, (int(size[0] * scale), int(size[1] * scale)), scale=float(scale), **kw)


def _ui_kit(spec: dict, scale: int = 2) -> tuple:
    """One style as a small mock window, plus the notes to annotate it with.

    A palette is not a kit until something is drawn with it, so the tile is a
    miniature screen: a header, a control in each of its states, one control that
    was *drawn narrow and stretched wide* by the nine-slice rule, and a hollow
    frame marked up with the cuts.  The notes come back as plain coordinates in
    the tile, so the sheet code can draw them on top of the finished render.
    """
    dw, dh = 240, 150
    win = _ui_part(spec, (dw, dh), scale)

    header = _ui_part({**spec, "body": spec["edge"], "body_low": spec["edge"],
                       "gradient": 0.35, "radius": 5.0, "boss": 0.0, "rule": 0.0,
                       "alpha": 1.0}, (dw - 16, 22), scale)
    win.blit(header.rgb, header.a, 8 * scale, 8 * scale)

    btn = {**spec, "radius": 5.0, "boss": 1.2}
    bars = []
    for i, state in enumerate(ui_paint.STATES):
        part = _ui_part(btn, (100, 22), scale, state=state)
        bars.append(part)
        win.blit(part.rgb, part.a, 8 * scale, (36 + i * 26) * scale)

    # The nine-slice claim, made once, in one line of code.
    ins = ui_slice.insets(ui_frame.Panel(btn, (100 * scale, 22 * scale), scale))
    wide = ui_slice.stretch(bars[0], ((dw - 16) * scale, 22 * scale), ins)
    win.blit(wide.rgb, wide.a, 8 * scale, 114 * scale)

    fr = {**spec, "boss": 1.2}
    box = (116, 36, 116, 74)
    part = _ui_part(fr, (box[2], box[3]), scale, kind="frame")
    win.blit(part.rgb, part.a, box[0] * scale, box[1] * scale)

    cut = ui_slice.insets(ui_frame.Panel(fr, (box[2] * scale, box[3] * scale), scale))
    x0, y0 = box[0] * scale, box[1] * scale
    x1, y1 = x0 + box[2] * scale, y0 + box[3] * scale
    cuts = [(x0 + cut[0], y0, x0 + cut[0], y1), (x1 - cut[2], y0, x1 - cut[2], y1),
            (x0, y0 + cut[1], x1, y0 + cut[1]), (x0, y1 - cut[3], x1, y1 - cut[3])]

    # Notes are in design pixels, relative to the tile.  Text over a dark body
    # has to be light, so the colour is decided here, where the style is known.
    body_l = float(palette.srgb_to_oklch(spec["body"])[0])
    ink = (246, 242, 234, 255) if body_l < 0.55 else (26, 26, 32, 255)
    plate = (0, 0, 0, 120) if body_l < 0.55 else (255, 255, 255, 200)
    notes = [((8, 12), "title bar", (246, 242, 234, 235), None)]
    for i, label in enumerate(list(ui_paint.STATES) + ["stretched"]):
        notes.append((("centre", 8, 36 + i * 26, 108, 36 + i * 26 + 22), label, ink, plate))
    hole = (box[0] + cut[0], box[1] + cut[1], box[2] - 2 * cut[0], box[3] - 2 * cut[1])
    notes.append((("centre", hole[0], hole[1], hole[0] + hole[2], hole[1] + 18),
                  "corners kept", ink, plate))
    notes.append((("centre", hole[0], hole[1] + 20, hole[0] + hole[2], hole[1] + 38),
                  "middle stretches", ink, plate))
    info = {"insets": list(cut), "design": [box[2], box[3]], "cuts": cuts, "notes": notes}
    return win, info


def _stretch_check(spec: dict, scale: int = 2, design=(240, 150), shrink: float = 0.6,
                   kind: str = "panel"):
    """Measure what a nine-slice resize would cost this style's body.

    The same asset is drawn twice: once at the size it is meant to be shown, and
    once at a smaller design size and then stretched to the first by the nine-slice
    rule.  For a design whose middle is constant along the stretch axis the two are
    the same picture; for one with a material body they are not, and the size of the
    disagreement is the number.  This is why the sidecar can say whether a panel is
    stretchable without anybody promising it: it was drawn twice and compared.
    """
    dw, dh = int(design[0]), int(design[1])
    small = (max(int(dw * shrink), 8), max(int(dh * shrink), 8))
    full = _ui_part(spec, (dw, dh), scale, kind=kind).to_uint8()
    was = _ui_part(spec, small, scale, kind=kind)
    ins = ui_slice.insets(ui_frame.Panel(spec, (small[0] * scale, small[1] * scale), scale))
    stretched = ui_slice.stretch_pixels(was.to_uint8(), (dw * scale, dh * scale), ins)
    return ui_slice.stretch_error(full, stretched)


def demo_ui(out: Path, seed: int = 7, count: int = 3, scale: int = 2, style: str | None = None) -> Path:
    """The UI kit sheet: every named style, plus seeded ones, as mock windows.

    The sidecar is the deliverable, not the PNG: a kit is ``(image, insets,
    minimum size)``, and an engine needs all three.
    """
    kits = []
    for name in ([style] if style else ui_style.names()):
        spec = ui_style.get(name)
        kits.append((name, spec, _ui_kit(spec, scale)))
    # One window with a title bar, because that is the case the per-side insets
    # exist for: the corners are blocks instead of squares, and the header has a
    # carved band to sit in rather than a blitted strip over the body.
    titled = ui_style.title_bar(ui_style.get(ui_style.names()[0]), 30.0)
    kits.append(("title bar", titled, _ui_kit(titled, scale)))
    for i in range(count):
        spec = ui_style.random_style(seed + i)
        kits.append((f"seed {seed + i}", spec, _ui_kit(spec, scale)))

    cols = 2
    gap = 22
    sheet = render.sheet([k[2][0] for k in kits], cols=cols, gap=gap, background=(0.90, 0.90, 0.93))
    sheet.save(out)
    img = png.to_image(sheet)
    draw = ImageDraw.Draw(img)

    # Measured once per kit, before anything is drawn with the number.
    errors = [_stretch_check(spec, scale) for _, spec, _ in kits]

    canvases = [k[2][0] for k in kits]
    for i, (name, spec, (tile, info)) in enumerate(kits):
        tx, ty, _, _ = _grid_pos(canvases, cols, gap, i)
        for x0, y0, x1, y1 in info["cuts"]:
            _dashed(draw, (tx + x0, ty + y0), (tx + x1, ty + y1))
        for where, text, ink, plate in info["notes"]:
            if len(where) == 5:
                _, nx0, ny0, nx1, ny1 = where
                _centre_label(img, (tx + nx0 * scale, ty + ny0 * scale,
                                    tx + nx1 * scale, ty + ny1 * scale),
                              text, dark=ink, plate=plate)
            else:
                _label(img, (tx + where[0] * scale, ty + where[1] * scale), text,
                       dark=ink, plate=plate)
        ins = info["insets"]
        shown = f"{ins[0]}" if len(set(ins)) == 1 else "/".join(str(v) for v in ins)
        lo = ui_slice.minimum(ins)
        note = "" if errors[i] < ui_slice.STRETCHABLE else f"   stretch {errors[i]:.2f}"
        _label(img, (tx + 4, ty + tile.h + 4),
               f"{name}   inset {shown} px @ {scale}x   min {lo[0]}x{lo[1]} px{note}")
    img.save(out, "PNG", optimize=True)

    # The sidecar carries both numbers an engine needs and both facts it cannot
    # guess: the corners, which only the source can state, and whether the middle
    # survives a resize, which is measured here because it is a property of the
    # pixels rather than of the spec.
    _sidecar(out, seed, "procedural:gen.ui.style", kits=[
        {"style": name, "insets": info["insets"],
         "design_size": info["design"], "spec": _json_style(spec),
         "panel": {"design_size": [240, 150],
                   "stretch_error": round(errors[i], 4),
                   "stretchable": errors[i] < ui_slice.STRETCHABLE}}
        for i, (name, spec, (_, info)) in enumerate(kits)
    ])
    return _report(out)


def _slug(name: str) -> str:
    """A kit's name as a file name: ``"seed 7"`` into ``"seed-7"``.

    The sheet calls its tiles things like *title bar*, which is a fine label and not
    a file name, and the two have to agree about which file is which tile.
    """
    out = "".join(c if c.isalnum() else "-" for c in name.strip().lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "kit"


def demo_vectors(directory: Path, seed: int = 7, count: int = 3, scale: int = 2,
                 style: str | None = None, prefix: str = "ui") -> Path:
    """The same kits as SVG: the geometry, for an engine that draws its own chrome.

    A panel is rounded rectangles and a polar ornament, so it is the one asset here
    with an exact vector form -- and what that buys is the opposite of what the
    nine-slice buys.  A nine-slice is what you do when the asset is *pixels*: cut it,
    state the insets, hope the middle stretches.  A vector needs none of that, because
    the curves are exact at every size, so one file serves every DPI.

    A kit is one file per style, named ``<prefix>.<style>.svg`` and written beside the
    sheet's own sidecar so the gate finds it: an ``.svg`` is an asset and needs a
    sidecar for the same reason a PNG does.

    What a vector cannot carry is the light -- the bevel, the dome, the shadow -- and
    a *material* cannot be carried at all, since wood grain is a noise field.  Those
    kits are named and left out rather than exported flat: an export that silently
    drops what it cannot draw is worse than an error, because it ships.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written, refused = [], []
    names = [style] if style else ui_style.names()
    kits = [(name, ui_style.get(name)) for name in names]
    kits.append(("title bar", ui_style.title_bar(ui_style.get(names[0]), 30.0)))
    kits += [(f"seed {seed + i}", ui_style.random_style(seed + i)) for i in range(count)]

    for name, spec in kits:
        path = directory / f"{prefix}.{_slug(name)}.svg"
        try:
            ui_vector.write(spec, (240 * scale, 150 * scale), path, scale=scale)
        except ValueError as why:
            refused.append((name, str(why)))
            continue
        written.append(path)
        _sidecar(path, seed, "procedural:gen.ui.vector", style=name,
                 scale=scale, spec=_json_style(spec),
                 design_size=[240, 150], units="design pixels")

    for path in written:
        _report(path)
    for name, why in refused:
        print(f"  {name}: no vector form -- {why}")
    if not written:
        raise ValueError("no kit on this sheet has a vector form")
    return directory


def _realm_record(name: str, colour) -> dict:
    """One realm in the map's sidecar: what it is called, and what it wears.

    The key is the arms' own identity, so a game can find the coat this colour came
    off -- and the one shade entry is the livery that made it, because the colour is
    the *field tincture's* step and not the other six.  A sidecar that carries the
    field, the step and the colour is a sidecar the picture can be reproduced from.
    """
    field = str(crestlib.arms(name)["field"]["tinctures"][0])
    return {
        "name": name,
        "key": crestlib.normalise(name),
        "tincture": field,
        "shade": crestlib.shade(name).get(field, {}),
        "colour": palette.srgb_to_hex(colour),
    }


def _realms(res, count: int, seed: int) -> dict | None:
    """The realms on one map's provinces, or ``None`` if there is nowhere to put them.

    The roster is drawn before the ownership is grown, because a set of realms that
    cannot be told apart is not worth assigning: the colour is the roster's, and the
    borders only decide where it stops.
    """
    if res is None or count <= 0 or not res["count"]:
        return None
    names = map_politics.factions(res, count, seed=seed)
    if not names:
        return None
    owners = map_politics.owners(res, len(names), seed=seed)
    table = map_politics.washes(names)
    return {
        "names": names,
        "owners": owners,
        "washes": table,
        "tints": map_politics.tint_table(res, owners, table),
        "record": [_realm_record(name, table[i]) for i, name in enumerate(names)],
    }


#: One tile of the maps sheet.  A tuple would do and did, until the file for a
#: campaign needed the terrain as well -- a name is how a caller says which field
#: it means instead of counting commas.
_MapTile = namedtuple("_MapTile",
                      "seed image stats courses res grid realms roads field seas waters")


def demo_maps(out: Path, seed: int = 7, count: int = 2, cols: int = 26, rows: int = 17,
              size: float = 13.0, sea: float = 0.42, provinces: int = 8,
              rivers: float = 0.92, names: bool = True, factions: int = 4,
              road: float = 0.85, sea_provinces: int = 8,
              faction_fill: float = map_politics.WASH_FILL) -> Path:
    """Contact sheet of maps: the geography is the seed, and nothing else.

    Each tile is labelled with the seed that makes it, the grid it is played on, the
    water fraction and the commonest ground.  Its *provinces* are named in the
    picture itself, in the project's own face: the paragraph that used to stand here
    said the names belonged to the preview because the only font in this project was
    Pillow's, and `gen/text/` is the end of that sentence.  A name is set as large as
    its province can hold, so a small island may be left without one -- the sidecar
    lists every name that was drawn and every one that was skipped, with its reason,
    because a label that is missing silently is the one thing a map cannot do.

    ``factions`` puts realms on the provinces, which is a second sheet of meaning
    over the same geography: the provinces are grown as they always were, the realms
    are drawn and grown over them, and the map fills with what each realm *wears* --
    its arms' tincture taken to a wash, at its own livery's lightness.  The fill is
    heavier than a province wash because it has to survive the same trip, and the
    sidecar carries the roster with the ownership, so a game can put the same
    colours on the same provinces.

    ``road`` draws the roads the capitals are joined by, which are routed by the
    biome table's own movement costs: what the picture shows is what an army would
    actually pay to march, so the map can be argued with.

    ``sea_provinces`` divides the water the same way the land is divided, drawn as a
    maritime boundary and nothing else -- no wash, because the sea's colour is the
    depth of the sea.  The land is identical either way, which is the claim the
    test measures.

    Each tile is also written out as a *campaign file* -- the map as data, with the
    biome table, the owner of every province, and the hexes the roads and rivers
    walk -- which is what a client that does not have this generator loads.  The
    picture is for the eye and the campaign file is for the game; both come from the
    same derivation, so they cannot disagree.
    """
    tiles = []
    for i in range(count):
        s = seed + i
        g = map_grid.HexGrid(cols, rows, size)
        f = map_terrain.fields(g, seed=s, sea_fraction=sea)
        st = map_terrain.stats(f["biome"])
        courses, _threshold = map_paint.river_courses(g, f) if rivers > 0.0 else ([], 0.0)
        res = map_region.provinces(g, f, count=provinces, seed=s) if provinces > 0 else None
        seas = map_region.seas(g, f, count=sea_provinces, seed=s) if sea_provinces > 0 else None
        realms = _realms(res, factions, s)
        waters = (map_politics.waters(g, res, seas, realms["owners"])
                  if (seas is not None and realms is not None) else None)
        political = {} if realms is None else {
            "province_fill": faction_fill,
            "province_tints": realms["tints"],
            "province_owners": realms["owners"],
        }
        routes = map_roads.network(g, f, res) if (res is not None and road > 0.0) else []
        img = map_paint.map_image(seed=s, cols=cols, rows=rows, size=size, sea_fraction=sea,
                                  provinces=provinces, river=rivers, road=road,
                                  seas=sea_provinces, names=names, **political)
        tiles.append(_MapTile(s, img, st, courses, res, g, realms, routes, f, seas, waters))

    gap = 24
    sheet = render.sheet([t.image for t in tiles], cols=min(2, len(tiles)), gap=gap,
                         background=(0.90, 0.90, 0.93))
    sheet.save(out)
    img = png.to_image(sheet)
    cw = max(t.image.w for t in tiles)
    ch = max(t.image.h for t in tiles)
    per_row = min(2, len(tiles))
    for i, t in enumerate(tiles):
        s, tile, st, courses, res, g, realms, routes = (t.seed, t.image, t.stats, t.courses,
                                                        t.res, t.grid, t.realms, t.roads)
        r, col = divmod(i, per_row)
        tx = gap + col * (cw + gap) + (cw - tile.w) // 2
        ty = gap + r * (ch + gap) + (ch - tile.h) // 2
        land = 100.0 * (1.0 - st["water"])
        counts = [f"land {land:.0f}%"]
        if courses:
            counts.append(f"{len(courses)} rivers")
        if res is not None:
            counts.append(f"{res['count']} provinces"
                          + (f" of {res['count'] - res['islands']}"
                             if res["islands"] else ""))
        if t.seas:
            counts.append(f"{t.seas['count']} sea provinces")
        if realms:
            counts.append(f"{len(realms['names'])} realms")
        if routes:
            counts.append(f"{len(routes)} roads")
        _label(img, (tx + 3, ty + tile.h + 4),
               f"seed {s}   {cols}x{rows} hexes at {size:g} px   " + "   ".join(counts))
    img.save(out, "PNG", optimize=True)

    _sidecar(out, seed, "procedural:gen.maps", maps=[
        {
            "seed": t.seed, "cols": cols, "rows": rows, "hex_size": size,
            "sea_target": sea, "water": t.stats["water"], "biomes": t.stats,
            "rivers": [{"cells": int(c["cells"]), "drainage": round(float(c["flux"]), 1)}
                       for c in t.courses],
            "roads": [{"from": int(r["from"]), "to": int(r["to"]),
                       "cells": int(r["cells"]), "cost": round(float(r["cost"]), 1)}
                      for r in t.roads],
            "factions": (t.realms["record"] if t.realms else []),
            "labels": (map_labels.census(map_labels.layout(t.grid, t.res))
                       if (names and t.res is not None) else None),
            "seas": ([{"name": p["name"], "area": int(p["area"]), "ground": p["ground"],
                       "capital": [int(p["capital"][0]), int(p["capital"][1])],
                       "owner": int(t.waters[i])}
                      for i, p in enumerate(t.seas["provinces"])] if t.seas else []),
            "provinces": ([
                {"name": p["name"], "area": int(p["area"]), "capital": [int(p["capital"][0]),
                                                                       int(p["capital"][1])],
                 "neighbours": [int(n) for n in p["neighbours"]], "ground": p["ground"],
                 "coastal": bool(p["coastal"]),
                 "owner": (int(t.realms["owners"][i]) if t.realms else -1)}
                for i, p in enumerate(t.res["provinces"])] if t.res is not None else []),
        }
        for t in tiles
    ])

    # And then the same map as a *campaign*: no pixels, no hex size, nothing that a
    # client would have to invert.  One file per map, because a campaign is one map.
    for t in tiles:
        doc = map_campaign.campaign(
            t.grid, t.field, t.res, seed=t.seed,
            owner=(t.realms["owners"] if t.realms else ()),
            realms=(t.realms["record"] if t.realms else ()),
            routes=t.roads, courses=t.courses, seas=t.seas,
            sea_owner=(t.waters if t.waters is not None else ()))
        path = out.with_name(f"{out.stem}.campaign.{t.seed}.json")
        path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        _report(path)
    return _report(out)


def demo_coast(out: Path, seed: int = 7, count: int = 2, cols: int = 26, rows: int = 17,
               hex_size: float = 13.0, sea: float = 0.42, size: float = 20.0,
               gap: int = 8, inland: float = 0.55) -> Path:
    """The coastline as a tileset: all sixty-four edge tiles, and what a map wears.

    A coast used to be a polyline walked round the land and stroked, which is a
    picture and not an asset.  It is an *edge* -- which of a hex's six sides has
    something else on the other side -- so it is six bits, and the tileset that
    draws it is sixty-four sprites.  This is that sheet: tile ``m`` in the
    ``(m // 8, m % 8)`` cell, the ink inside its own hexagon, and nothing on the
    image outside the grid but a caption.  A game lifts a tile out and blits it
    where the key says, which is what :func:`gen.maps.paint.map_image` now does
    with them as well.

    The sheet is the one file in this demo that no seed can change, because it is
    every tile there could be and not the tiles one map happened to need.  What
    the seed is used for is the other column of numbers: how few of the sixty-four
    each of the demo's maps actually wears, which is the measured answer to "do
    sixty-four tiles matter".

    And the count is asked twice, because the tileset answers two questions.  A
    border is where two hexes disagree, so the same sprites draw the *biome*
    border as well -- that is the coast's question asked ``within`` the land --
    and the two numbers say something worth knowing about each: a coastline wears
    a couple of dozen of the sixty-four, because a coast is a line, while a biome
    border wears nearly all of them, because ground mixes hex by hex.  Both facts
    are why the sheet has to be complete: a game cannot be handed the keys one map
    happened to use.  ``borders.png`` draws both on the same map to show it; the
    sheet here is what they are cut from.

    The caption is drawn below the grid rather than on it, so the grid stays a
    container: the tile rects are recorded in the sidecar and a game crops them
    from the file, exactly as it would from any other sheet.
    """
    sheet = map_autotile.sheet(map_paint.INK, size, gap=gap)
    tw, th = map_autotile.tile_size(size)
    worn = []
    for i in range(count):
        s = seed + i
        g = map_grid.HexGrid(cols, rows, hex_size)
        f = map_terrain.fields(g, seed=s, sea_fraction=sea)
        edge = map_autotile.borders((~f["water"]).astype(np.int64), g, outside=0)
        _coast, coast_keys = map_autotile.tiles(edge)
        # The same arithmetic, a different field: a biome border is an edge between
        # two land hexes of different ground, so it is asked *within* the land and
        # the coast keeps the shore to itself.
        biome = map_autotile.borders(f["biome"], g, within=~f["water"])
        _inland, inland_keys = map_autotile.tiles(biome)
        worn.append((s, len(coast_keys), len(inland_keys)))
    caption = (f"{map_autotile.MASKS} edge keys, {map_autotile.EDGES} edges each   "
               f"hex {size:g} px"
               + "".join(f"   seed {s}: {n} coast, {m} inland" for s, n, m in worn))
    font = _font(12)
    need = int(ImageDraw.Draw(png.to_image(sheet)).textlength(caption, font=font)) + 14
    strip = 26
    page = Canvas(max(sheet.w, need), sheet.h + strip, scale=1.0)
    page.over(sheet.rgb, sheet.a)
    band = np.zeros((page.h, page.w), dtype=np.float64)
    band[sheet.h:] = 1.0
    page.over_srgb((1.0, 1.0, 1.0), band)
    preview = png.to_image(page)
    _label(preview, (6, sheet.h + 5), caption, font=font, plate=None)
    out.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out, "PNG", optimize=True)

    _sidecar(out, seed, "procedural:gen.maps.autotile", coast_tiles={
        "edges": map_autotile.EDGES,
        "keys": map_autotile.MASKS,
        "columns": map_autotile.SHEET_COLS,
        "gap": gap,
        "tile": [tw, th],
        "grid": [gap, gap],
        "hex_size": size,
        "pitch": [round(np.sqrt(3.0) * size, 4), round(1.5 * size, 4)],
        "ink": {"colour": palette.srgb_to_hex(map_paint.INK),
                "width": 0.24, "alpha": 0.80},
        "worn": [{"seed": s, "coast": n, "inland": m} for s, n, m in worn],
    })

    # The second use, drawn rather than asserted.  The same map twice: once with the
    # coastline alone and once with the biome borders as well, and nothing else
    # differs between the two -- the lattice is off on both, so the only hex-shaped
    # thing in the picture is the border itself, which is the thing being shown.
    s0, n0, m0 = worn[0]
    variants = [(0.0, "coastline", n0)]
    if inland > 0.0:
        variants.append((inland, "+ biome borders", m0))
    panels, captions = [], []
    for alpha, what, keys in variants:
        panels.append(map_paint.map_image(
            seed=s0, cols=cols, rows=rows, size=hex_size, sea_fraction=sea,
            lattice=0.0, provinces=0, seas=0, road=0.0, inland=alpha))
        captions.append(f"seed {s0}   {what}   {keys} of {map_autotile.MASKS} keys")
    borders = out.with_name("borders.png")
    sheet_gap = 26
    uses = png.to_image(render.sheet(panels, cols=len(variants), gap=sheet_gap,
                                     background=(0.90, 0.90, 0.93)))
    font = _font(11)
    for i, text in enumerate(captions):
        x, y, _cw, _ch = _grid_pos(panels, len(variants), sheet_gap, i)
        _label(uses, (x + 3, y + panels[i].h + 4), text, font=font)
    uses.save(borders, "PNG", optimize=True)
    _sidecar(borders, seed, "procedural:gen.maps.autotile", borders={
        "grid": [cols, rows],
        "hex_size": hex_size,
        "sea_target": sea,
        "ink": {"colour": palette.srgb_to_hex(map_paint.INK)},
        "coast": {"width": 0.24, "alpha": 0.80},
        "inland": {"width": 0.11, "alpha": inland, "asked": "within the land"},
        "lattice": 0.0,
        "maps": [{"seed": s, "coast_keys": n, "inland_keys": m} for s, n, m in worn],
    })
    _report(borders)
    return _report(out)


def _json_style(spec: dict) -> dict:
    """A spec with the colours as hex, so the sidecar is readable and stable."""
    out = {}
    for k, v in spec.items():
        if k in ui_style.COLOURS:
            out[k] = palette.srgb_to_hex(v)
        elif k in ui_style.SIDED:
            out[k] = [round(float(x), 4) for x in v]
        elif isinstance(v, str):
            out[k] = v          # the material's name is an identifier, not a number
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = round(float(v), 4)
    return out


def _wang_tiles(cells: int, colours: int, seed: int, source: str,
                kind: str = "complete") -> int:
    """How many distinct tiles a tiling uses -- measured, not promised.

    The interesting number about the two ways of getting a tiling out of ``wang`` is
    this one.  A search has to find an arrangement in which every shared corner
    agrees, and there are not many ways to do that with corner colours that were
    chosen at random, so the arrangement it finds is close to *free*: nearly every
    cell ends up with a different tile, 168 of 256 on the demo's grid.  A tiling read
    off a field repeats -- 40 of the same 256 -- because a landscape keeps coming back
    through the same bands, which is why it looks like ground and not like a set of
    tiles.
    """
    if source == "field":
        grid, _tiles = tex_wang.from_field(tex_wang.height_field(seed, cells), colours, cells)
    else:
        grid = tex_wang.fill(cells, cells, tex_wang.tile_set(colours, kind), seed=seed)
    return len(set(int(v) for v in grid.reshape(-1)))


def demo_tiles(out: Path, seed: int = 7, size: int = 256, sim: int = 80,
               steps: int = 5000, atlas: bool = True, mips: int | None = 1) -> Path:
    """Contact sheet of tiling textures: Truchet, Wang, reaction-diffusion, boards.

    Every cell is the tile laid down **four times** rather than shown once.  That is
    the only way to see whether a texture tiles: one copy of it is seamless by
    definition, and a seam only exists between two of them.  The sidecar records the
    seam error of each tile, which is the number this sheet is here to justify.

    The two Wang rows are the two answers to the same question.  ``meadow`` and
    ``stone`` are tilings *searched* for over a set of corner colours; ``highland`` is
    read off a quantised height field, so its corners are heights of one landscape and
    its seamlessness is not a property of the search at all.  Its grid is twice as
    fine, because there the tiles carry samples rather than being the picture.

    The two boards are the same floor at two lengths, which is the parameter that
    decides whether it reads as a wooden floor or as brickwork: a board six times as
    long as it is wide is a floorboard, and one twice as long is a tile.

    ``mips`` decides how the sheet a *game* loads is shaped.  At 1 -- the default --
    it is one packed sheet.  Above that it is a chain of sheets, one per mip level,
    each packed fresh, because halving a tile wants a new layout and not a quarter of
    the old one: the margin is the same two *texels* at every level, which is what a
    bilinear filter reaches, so it is a larger fraction of a smaller tile.  ``None``
    is the whole chain, down to one-texel tiles, and that is where the price is
    legible -- the sidecar of each level carries its own density, and the coarsest is
    a sheet that is very nearly all margin.  The demo prints the numbers rather than
    arguing about them.
    """
    half = int(size) // 2
    wang_cells = half // 16
    panels = [
        ("truchet, arcs", tex_truchet.texture(seed=seed, size=half, grains=4,
                                             kinds=("arcs",)), {}),
        ("truchet, mixed", tex_truchet.texture(seed=seed + 1, size=half, grains=4,
                                              kinds=("arcs", "lines"), mix=0.5), {}),
        ("truchet, lines", tex_truchet.texture(seed=seed + 2, size=half, grains=8,
                                              kinds=("lines",)), {}),
        ("wang, meadow", tex_wang.texture(seed=seed, size=half, cells=wang_cells, colours=3),
         {"source": "search", "palette": "meadow", "colours": 3,
          "set": len(tex_wang.tile_set(3)), "used": _wang_tiles(wang_cells, 3, seed, "search")}),
        ("wang, stone", tex_wang.texture(seed=seed + 3, size=half, cells=wang_cells, colours=4,
                                        palette_name="stone"),
         {"source": "search", "palette": "stone", "colours": 4,
          "set": len(tex_wang.tile_set(4)),
          "used": _wang_tiles(wang_cells, 4, seed + 3, "search")}),
        ("wang, highland", tex_wang.texture(seed=seed, size=half, cells=half // 8, colours=4,
                                           palette_name="highland", source="field", relief=20.0),
         {"source": "field", "palette": "highland", "colours": 4,
          "set": len(tex_wang.tile_set(4)),
          "used": _wang_tiles(half // 8, 4, seed, "field")}),
        ("reaction, worms", tex_reaction.texture(seed=seed, size=half, sim=sim,
                                               preset_name="worms", steps=steps), {}),
        ("reaction, holes", tex_reaction.texture(seed=seed, size=half, sim=sim,
                                               preset_name="holes", steps=steps), {}),
        ("boards, floor", tex_boards.texture(seed=seed, size=half, across=1, down=6,
                                            stagger=2),
         {"across": 1, "down": 6, "stagger": 2}),
        ("boards, short", tex_boards.texture(seed=seed + 1, size=half, across=3, down=6,
                                            stagger=1),
         {"across": 3, "down": 6, "stagger": 1}),
    ]

    gap = 22
    per_row = 5
    sheet = render.sheet([_repeat_canvas(panel, 2) for _name, panel, _facts in panels],
                         cols=per_row, gap=gap, background=(0.90, 0.90, 0.93))
    sheet.save(out)
    img = png.to_image(sheet)
    cw = int(size)
    ch = int(size)
    for i, (name, _cv, _facts) in enumerate(panels):
        r, col = divmod(i, per_row)
        tx = gap + col * (cw + gap)
        ty = gap + r * (ch + gap)
        _label(img, (tx + 3, ty + ch + 4), f"{name}   seed {seed}   {cw}x{ch}")
    img.save(out, "PNG", optimize=True)

    _sidecar(out, seed, "procedural:gen.textures", tiles=[
        {
            "name": name,
            "size": cw,
            "seed": seed,
            "seam_error": round(tilelib.seam_error(
                tilelib.luminance(panel.to_uint8())), 3),
            **facts,
        }
        for name, panel, facts in panels
    ])

    # The contact sheet is for a person; this is the sheet a *game* loads.  Same ten
    # tiles, and the difference is entirely in the margins: a sprite atlas trims its
    # members and leaves a transparent gap, and for a tile both of those are wrong.  A
    # tile's edge is its *seam*, so there is nothing to trim, and its margin has to be
    # more of the tile -- the tile wrapped onto itself -- so that a filtered sample
    # beside it reads the material rather than a neighbour or a gap.  Above one level,
    # the same thing is done at every scale: ``mips`` repacks rather than resampling a
    # sheet, because a margin is two texels at every level and not two pixels at level
    # zero divided by eight.
    if atlas:
        members = [(name, panel) for name, panel, _f in panels]
        if mips is None or int(mips) > 1:
            # A chain: level 0 is the single sheet, so the chain does not have a
            # different head from the sheet it grew out of.  ``None`` is every level
            # the tiles support, which is the honest version of "how bad does it get":
            # the answer is at the bottom, where a tile is smaller than its own bleed.
            levels = None if mips is None else int(mips)
            for image, info in tex_atlas.mips(members, levels=levels):
                path = out.with_name(f"{out.stem}-mip{info['level']}.png")
                image.save(path)
                _sidecar(path, seed, "procedural:gen.textures.atlas", **info)
                _report(path)
        else:
            image, info = tex_atlas.build(members)
            path = out.with_name(f"{out.stem}-atlas.png")
            image.save(path)
            _sidecar(path, seed, "procedural:gen.textures.atlas", **info)
            _report(path)
    return _report(out)


def _repeat_canvas(canvas, n: int = 2):
    """A canvas laid out ``n`` by ``n``: the picture that shows a tile wraps."""
    out = Canvas(canvas.w * int(n), canvas.h * int(n), scale=canvas.scale,
                 y_up=canvas.y_up)
    for i in range(int(n)):
        for j in range(int(n)):
            out.blit(canvas.rgb, canvas.a, j * canvas.w, i * canvas.h)
    return out


def demo_grounds(out: Path, seed: int = 7, size: int = 256, atlas: bool = True,
                 mips: int | None = 1) -> Path:
    """Contact sheet of the six ground materials, each shown as a tiling of four.

    The materials are a *set* rather than six pictures, so the sheet is the place to
    check that they are six: laid side by side at the same size and the same seed, with
    the seams on show, the differences are the ones the table says they are -- grass is
    a soft mottle, sand is a ripple, rock is a plate, snow is a drift, and the two that
    carry pebbles differ in how many and how big.

    The captions carry the *map* as well as the material, because that is what this set
    is for: ``FOR_BIOME`` says which biome wears which ground, and a caption that read
    only "grass" would leave a reader to guess that plains, steppe, forest and marsh are
    all the same material.  The sidecar carries the same mapping the other way round and
    every number in the row that made the tile, so a seventh material can be argued about
    in numbers rather than in adjectives.

    ``mips`` packs the atlas the way the tiles sheet is packed: 1 is the one sheet, and
    above that a chain whose levels are reduced and *repacked*, because the margin is two
    texels at every level.  The demo ships the head by default -- a ground is read from a
    metre away, and six materials are a set rather than a level of detail -- but a game
    that keeps a ground under a camera that moves has the same chain the tiles do.
    """
    half = max(int(size) // 2, 8)
    panels = [(name, tex_ground.texture(name, seed=seed, size=half))
              for name in tex_ground.names()]
    gap, per_row, head, margin = 42, 3, 92, 40

    sheet = render.sheet([_repeat_canvas(panel, 2) for _name, panel in panels],
                         cols=per_row, gap=gap, background=(0.90, 0.90, 0.93))
    page = Canvas(sheet.w + 2 * margin, sheet.h + head + margin, scale=1.0,
                  y_up=False, background=(0.90, 0.90, 0.93))
    page.blit(sheet.rgb, sheet.a, margin, head)
    page.save(out)

    img = png.to_image(page)
    _label(img, (margin, 18), "Grounds -- six materials for map tiles",
           font=_font(21), plate=None, dark=(28, 28, 34, 255))
    _label(img, (margin, 50),
           "each tile laid down twice each way; the seam error of every one is in the "
           "sidecar", font=_font(13), plate=None, dark=(78, 78, 88, 255))
    for i, (name, panel) in enumerate(panels):
        r, col = divmod(i, per_row)
        tx = margin + gap + col * (half * 2 + gap)
        ty = head + gap + r * (half * 2 + gap) + half * 2 + 5
        wears = sorted(k for k, v in tex_ground.FOR_BIOME.items() if v == name)
        tail = ", ".join(wears) if wears else "no biome -- a road or a camp"
        _label(img, (tx + 3, ty), f"{name}   tile {panel.w}   seed {seed}",
               font=_font(13))
        _label(img, (tx + 3, ty + 17), tail, font=_font(12), plate=None,
               dark=(60, 60, 70, 255))
    img.save(out, "PNG", optimize=True)

    g = {row.key: row for row in tex_ground.GROUNDS}
    _sidecar(out, seed, "procedural:gen.textures.ground",
             tile=half,
             for_biome=dict(sorted(tex_ground.FOR_BIOME.items())),
             grounds=[
                 {
                     "name": name,
                     "seed": seed,
                     "size": panel.w,
                     # The whole row, not a selection of it: the docstring above
                     # promises "every number in the row", and a sidecar that
                     # dropped the scales could not be fed back to ``texture``.
                     # ``base`` stays in OKLCH, because that is the parameter the
                     # painter reads and a hex would round it to 8 bits; ``base_hex``
                     # is the swatch, for a reader rather than for the code.
                     "row": {**g[name]._asdict(), "base": list(g[name].base)},
                     "base_hex": palette.srgb_to_hex(
                         palette.oklch_to_srgb(g[name].base)),
                     "biomes": sorted(k for k, v in tex_ground.FOR_BIOME.items()
                                      if v == name),
                     "seam_error": round(tilelib.seam_error(
                         tilelib.luminance(panel.to_uint8())), 3),
                 }
                 for name, panel in panels
             ])

    # The same six as one packed sheet, with the bleed a wrapped tile needs: the
    # margins are more of the tile rather than a transparent gap, because the pixel
    # beside a tile's edge is its own opposite edge.
    if atlas:
        members = [(name, panel) for name, panel in panels]
        if mips is None or int(mips) > 1:
            levels = None if mips is None else int(mips)
            for image, info in tex_atlas.mips(members, levels=levels):
                path = out.with_name(f"{out.stem}-mip{info['level']}.png")
                image.save(path)
                _sidecar(path, seed, "procedural:gen.textures.atlas", **info)
                _report(path)
        else:
            image, info = tex_atlas.build(members)
            path = out.with_name(f"{out.stem}-atlas.png")
            image.save(path)
            _sidecar(path, seed, "procedural:gen.textures.atlas", **info)
            _report(path)
    return _report(out)


def demo_site(out: Path, seed: int = 7, size: int = 56, cols: int = 8,
              rows: int = 6, flat: bool = False,
              river_fraction: float | None = None) -> Path:
    """One site twice: the biome's colour, and the ground the hex wears.

    The two panels are the same geography and the same seed, so the difference is the
    one the module is about.  At the strategic map's fourteen pixels the left picture is
    the honest one; at fifty-six the right one is, because a hex is now big enough for
    the eye to ask what the ground is *made of* rather than what it is.

    ``flat`` asks for the left panel alone, which is what a caller comparing a site with
    ``out/maps.png`` wants; the demo ships both because the pair is the argument.
    """
    # The map's own river quantile leaves a 48-cell site with no river at all; a site
    # is a *place*, and the demo opens the fraction a little so the picture has the
    # stream the module can draw.  The sidecar records it, so the number is not hidden.
    # A caller can ask for another fraction without abandoning the demo's framing.
    river = dict(river_fraction=0.90 if river_fraction is None
                 else float(river_fraction))
    flat_panel = map_site.site_image(seed=seed, cols=cols, rows=rows, size=size,
                                     grounds=False, **river)
    panels = [("flat", flat_panel,
               "the biome's colour, as the strategic map draws it")]
    if not flat:
        panels.append(("worn", map_site.site_image(seed=seed, cols=cols, rows=rows,
                                                   size=size, **river),
                       "the same hexes, each wearing its biome's ground"))

    gap, head, foot, margin = 26, 96, 50, 30
    pw, ph = panels[0][1].w, panels[0][1].h
    page = Canvas(len(panels) * pw + (len(panels) - 1) * gap + 2 * margin,
                  head + ph + foot + margin, scale=1.0, y_up=False,
                  background=(0.90, 0.90, 0.93))
    for i, (_tag, panel, _note) in enumerate(panels):
        page.blit(panel.rgb, panel.a, margin + i * (pw + gap), head)
    page.save(out)

    img = png.to_image(page)
    _label(img, (margin, 18), "A site at close range", font=_font(21), plate=None,
           dark=(28, 28, 34, 255))
    _label(img, (margin, 50),
           f"the same {cols} x {rows} hexes at {size} px, seed {seed}; the material is "
           "sampled in world coordinates, the river is a groove, the coast is wet",
           font=_font(13), plate=None, dark=(78, 78, 88, 255))
    for i, (tag, _panel, note) in enumerate(panels):
        x = margin + i * (pw + gap) + 3
        _label(img, (x, head + ph + 6), f"{tag} -- {note}", font=_font(13))
    img.save(out, "PNG", optimize=True)

    _sidecar(out, seed, "procedural:gen.maps.site",
             cols=cols, rows=rows, size=size,
             span=map_site.SPAN, mix=map_site.MIX, relief=map_site.RELIEF,
             terrain_relief=map_site.TERRAIN_RELIEF,
             cliff=0.55, shore=0.052, river=0.92,
             river_fraction=river["river_fraction"],
             river_width=0.17, river_depth=0.06,
             for_biome=dict(sorted(tex_ground.FOR_BIOME.items())),
             panels=[tag for tag, _panel, _note in panels])
    return _report(out)


def demo_iso(out: Path, seed: int = 7, cols: int = 11, rows: int = 8,
              size: int = 46, sea: float = 0.16, rise: float = 2.5,
              levels: int = 7, hero=None, hero_seed: int | None = None,
              cycle: str = "idle", facing: str = "right") -> Path:
    """One board from the south: the same hexes as the site, standing up with a hero.

    The geography is :mod:`gen.maps.terrain`'s, asked for a *land* board: the strategic
    map's default sink puts a ring of sea round every rectangle, which is right for a
    map and wrong for the middle of a campaign.  ``sea=0.16, edge=0`` gives a continent
    with a lake in it, which is what a board is for.

    The hero is a figure sprite, not a new kind of drawing: ``gen.figures`` already
    knows how a person stands, and the board only has to say where the ground is.  He
    gets his own seed so a sheet of boards can keep one hero while the maps change,
    and a facing -- right or left across the board, down towards the camera or up away
    from it -- so the same body can walk in all four directions.
    """
    grid = map_grid.HexGrid(cols, rows, size)
    f = map_terrain.fields(grid, seed=seed, sea_fraction=sea, edge=0.0)
    if hero is None:
        hero = map_iso.hero_cell(grid, f)
    hero_seed = int(seed) if hero_seed is None else int(hero_seed)

    board = map_iso.iso_image(seed=seed, cols=cols, rows=rows, size=size,
                              sea_fraction=sea, edge=0.0, rise=rise,
                              levels=levels, hero=hero, hero_seed=hero_seed,
                              cycle=cycle, facing=facing)
    head, foot, margin = 96, 62, 34
    page = Canvas(board.w + 2 * margin, board.h + head + foot + margin,
                  scale=1.0, y_up=False, background=(0.90, 0.90, 0.93))
    # The board is an object on a table, and the shadow is what says so; it belongs
    # to the page and not to the board, because a transparent margin should stay
    # transparent in a game.
    shadow = render.drop_shadow(board.a, scale=1.0, offset=(3.0, -7.0),
                                sigma=5.0, opacity=0.30)
    page.over_srgb((0.05, 0.05, 0.07), shadow, margin, head)
    page.blit(board.rgb, board.a, margin, head)
    page.save(out)

    img = png.to_image(page)
    _label(img, (margin, 18), "An isometric board -- the same hexes, seen from the south",
           font=_font(21), plate=None, dark=(28, 28, 34, 255))
    _label(img, (margin, 50),
           f"{cols} x {rows} hexes at {size} px, seed {seed}; orthographic axonometry, "
           f"{levels} terraces, the tops wear the site's ground",
           font=_font(13), plate=None, dark=(78, 78, 88, 255))
    if hero is None:
        ground = "no land to stand on -- the board is all sea"
    else:
        ground = (f"the hero stands on row {hero[0]}, column {hero[1]} "
                  f"(figure seed {hero_seed}, {cycle}, facing {facing})")
    _label(img, (margin, head + board.h + 8), ground, font=_font(13))
    _label(img, (margin, head + board.h + 30),
           "four cardinal facings are what a board with no yaw can promise; the "
           "faces are the earth a cliff exposes", font=_font(12), plate=None,
           dark=(78, 78, 88, 255))
    img.save(out, "PNG", optimize=True)

    _sidecar(out, seed, "procedural:gen.maps.iso",
             cols=cols, rows=rows, size=size,
             squash=map_iso.SQUASH, rise=rise, levels=levels,
             sea_fraction=sea, edge=0.0,
             span=map_site.SPAN, mix=map_site.MIX,
             hero_size=map_iso.HERO_SIZE, lattice=map_iso.LATTICE,
             foam=map_iso.FOAM, facing=facing,
             hero=None if hero is None else {
                 "row": int(hero[0]), "col": int(hero[1]),
                 "seed": hero_seed, "cycle": cycle, "facing": facing},
             for_biome=dict(sorted(tex_ground.FOR_BIOME.items())))
    return _report(out)


def demo_factions(out: Path, names=None, size: int = 168, seed: int = 7,
                  ratio=(3, 2)) -> Path:
    """Contact sheet of factions, each with its shield and the banner it flies.

    The names are hand-written and everything else is derived from them, which is
    the point of the exercise: a faction's identity is a *name*, and the demo has to
    be handed names because that is the only input the generator takes.  Pass your
    own with ``--name`` and the sheet is your game's factions.

    Both tiles are painted in the faction's own shade of the tinctures, not the
    classical ones.  Seven colours and twelve factions means two of them will have
    chosen Azure; a player looking at a map has to be able to tell those two apart,
    and the shade is derived from the same name as the coat, so it cannot drift.
    """
    if not names:
        names = list(FACTIONS)
    factions = [crestlib.faction(name, ratio=ratio) for name in names]

    tiles = []
    for f in factions:
        table = tinctureslib.shade_table(f["shade"])
        tiles.append(heraldry.render(f["arms"], size=size, shade=table))
        tiles.append(flaglib.render(f["flag"], size=size, shade=table))

    gap = 46
    sheet = render.sheet(tiles, cols=4, gap=gap, background=(0.90, 0.91, 0.94))
    sheet.save(out)
    img = png.to_image(sheet)
    for i, f in enumerate(factions):
        # Two tiles per faction, four to a row, so a faction's caption belongs under
        # the *pair*, and the gap between rows is what it is drawn in.
        x = gap + (2 * i) % 4 * (size + gap)
        y = gap + (i // 2) * (size + gap) + size + 5
        _wrap_label(img, (x, y), f["name"], width=2 * size + gap - 6, font=_font(13))
        _wrap_label(img, (x, y + 16), f["blazon"]["en"], width=2 * size + gap - 6)
    img.save(out, "PNG", optimize=True)

    _sidecar(out, seed, "procedural:gen.heraldry.crest.faction",
             names=[f["name"] for f in factions],
             factions=[
        {
            "name": f["name"],
            "key": f["key"],
            "seed": f["seed"],
            "shade": f["shade"],
            "blazon_en": f["blazon"]["en"],
            "blazon_ru": f["blazon"]["ru"],
            "banner_en": f["description"]["en"],
            "banner_ru": f["description"]["ru"],
            "warnings": f["warnings"],
            "arms": f["arms"],
            "flag": f["flag"],
        }
        for f in factions
    ])
    return _report(out)


def demo(out: Path, seed: int = 7) -> None:
    """Render everything into ``out/`` -- the smoke test for the whole core."""
    print(f"ozart demo  (seed={seed}, out={out})")
    _report(demo_icons(out / "icons.png", seed=seed, count=8))
    demo_shapes(out / "shapes.png")
    demo_texture(out / "texture.png", seed=seed, size=256)
    demo_palette(out / "palette.png", seed=seed, count=6)
    demo_arms(out / "arms.png", seed=seed, count=12)
    demo_flags(out / "flags.png", seed=seed, count=12, hang="mixed")
    demo_anim(out / "anim.png", seed=seed)
    demo_figures(out / "figures.png", seed=seed)
    demo_portraits(out / "portraits.png", seed=seed, count=8, size=256)
    demo_lineage(out / "lineage.png", seed=seed)
    demo_ui(out / "ui.png", seed=seed, count=2, scale=2)
    demo_vectors(out, seed=seed, count=2, scale=2)
    demo_maps(out / "maps.png", seed=seed, count=2)
    demo_coast(out / "coast-tiles.png", seed=seed, count=2)
    demo_tiles(out / "tiles.png", seed=seed, mips=TILE_MIPS)
    demo_grounds(out / "grounds.png", seed=seed)
    demo_site(out / "site.png", seed=seed)
    demo_iso(out / "iso.png", seed=seed)
    demo_text(out / "text.png")
    demo_factions(out / "factions.png", seed=seed)
    _report(creditslib.write(out))
    _report(creditslib.write_screen(out))
    _report(gallerylib.write(out))
    print("done.")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ozart", description="Procedural 2-D asset generator (SDF + OKLab).")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="render all core demos into a directory")
    d.add_argument("--out", default="out", type=Path)
    d.add_argument("--seed", type=int, default=7)

    su = sub.add_parser("studio", help="a local browser workbench for making assets")
    su.add_argument("--host", default="127.0.0.1",
                    help="interface to bind; the default is this machine only")
    su.add_argument("--port", type=int, default=8000, help="0 asks the OS for a free port")
    su.add_argument("--root", default=Path("out/studio"), type=Path,
                    help="where renders are written")
    su.add_argument("--no-browser", action="store_true",
                    help="do not open a browser window")

    w = sub.add_parser("gallery", help="the build as one page: every sheet, its numbers, its command")
    w.add_argument("--manifest", default=Path("out"), type=Path)
    w.add_argument("--out", default=None, type=Path, help="default: <manifest>/index.html")
    w.add_argument("--embed", action="store_true",
                   help="inline the pictures, so the page travels as one file")
    w.add_argument("--stats", action="store_true",
                   help="print the page and picture sizes instead of writing the page")
    w.add_argument("--title", default="ozart build")

    f = sub.add_parser("figures", help="sprite sheet of one figure's cycle")
    f.add_argument("--out", default="out/figures.png", type=Path)
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--cycle", default="walk", choices=figurelib.cycles())
    f.add_argument("--count", type=int, default=8)
    f.add_argument("--size", type=int, default=96)
    f.add_argument("--cols", type=int, default=8)
    f.add_argument("--fps", type=float, default=12.0)
    f.add_argument("--facing", default="right", choices=figurelib.FACINGS,
                   help="right/left across the screen, down towards the camera, up away")

    i = sub.add_parser("icons", help="icon set, seeds, size sweep and a packed atlas")
    i.add_argument("--out", default="out/icons.png", type=Path)
    i.add_argument("--seed", type=int, default=3)
    i.add_argument("--count", type=int, default=8)
    i.add_argument("--size", type=int, default=192, help="size of the set and seed tiles")
    i.add_argument("--cols", type=int, default=9, help="tiles per row")
    i.add_argument("--sizes", default="32,64,128",
                   help="comma-separated sizes to generate for the sweep")
    i.add_argument("--no-atlas", action="store_true", help="skip the packed atlas")
    i.add_argument("--name", action="append", default=None,
                   help="a faction name to dress the set in (repeatable)")
    i.add_argument("--family", default=None, choices=tuple(icon_motifs.FAMILIES),
                   help="restrict the set to a NATO-like or HOI4-like role family")
    i.add_argument("--no-wear", action="store_true",
                   help="skip the ten-stage cumulative wear sheet and atlas")
    i.add_argument("--wear-size", type=int, default=48,
                   help="tile size for the wear sheet")

    wd = sub.add_parser("wardrobe", help="garment cuts and seeded wardrobe variants")
    wd.add_argument("--out", default="out/wardrobe.png", type=Path)
    wd.add_argument("--seed", type=int, default=3)
    wd.add_argument("--count", type=int, default=6, help="seeded variants after the cut set")
    wd.add_argument("--size", type=int, default=256)
    wd.add_argument("--cols", type=int, default=5)
    wd.add_argument("--cut", default=None, choices=wardrobe_garment.CUTS)
    wd.add_argument("--pattern", default=None, choices=wardrobe_garment.PATTERNS)
    wd.add_argument("--cloth", default=None, metavar="#RRGGBB")
    wd.add_argument("--trim", default=None, metavar="#RRGGBB")
    wd.add_argument("--metal", default=None, metavar="#RRGGBB")

    dc = sub.add_parser("decorations", help="award families and seeded decorations")
    dc.add_argument("--out", default="out/decorations.png", type=Path)
    dc.add_argument("--seed", type=int, default=3)
    dc.add_argument("--count", type=int, default=6)
    dc.add_argument("--size", type=int, default=256)
    dc.add_argument("--cols", type=int, default=5)
    dc.add_argument("--kind", default=None, choices=wardrobe_awards.KINDS)
    dc.add_argument("--metal", default=None, choices=wardrobe_awards.METALS)
    dc.add_argument("--motif", default=None, choices=wardrobe_awards.MOTIFS)
    dc.add_argument("--bars", type=int, default=None, help="ribbon rack rows (1-4)")
    dc.add_argument("--rosette", action="store_true")

    s = sub.add_parser("shapes", help="reference sheet of SDF primitives")
    s.add_argument("--out", default="out/shapes.png", type=Path)
    s.add_argument("--size", type=int, default=160)
    s.add_argument("--cols", type=int, default=3)

    t = sub.add_parser("texture", help="materials sheet: fBm, marble, wood, stone")
    t.add_argument("--out", default="out/texture.png", type=Path)
    t.add_argument("--seed", type=int, default=3)
    t.add_argument("--size", type=int, default=512)

    a = sub.add_parser("palette", help="swatch sheet + JSON dump")
    a.add_argument("--out", default="out/palette.png", type=Path)
    a.add_argument("--seed", type=int, default=3)
    a.add_argument("--count", type=int, default=6)

    r = sub.add_parser("arms", help="contact sheet of generated coats of arms")
    r.add_argument("--out", default="out/arms.png", type=Path)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--count", type=int, default=12)
    r.add_argument("--size", type=int, default=190)
    r.add_argument("--cols", type=int, default=4)
    r.add_argument("--shield", default=None, help="force one shield shape")

    f = sub.add_parser("flags", help="contact sheet of generated flags")
    f.add_argument("--out", default="out/flags.png", type=Path)
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--count", type=int, default=12)
    f.add_argument("--size", type=int, default=200)
    f.add_argument("--cols", type=int, default=4)
    f.add_argument("--ratio", default="3:2", help="flag proportion, e.g. 3:2 or 5:8")
    f.add_argument("--shape", default="rectangle",
                   choices=["rectangle", "swallowtail", "pennant", "guidon"])
    f.add_argument("--hang", default="pole",
                   choices=list(HANGINGS) + ["mixed", "none"],
                   help="what the cloth hangs from: a pole, a crossbar (a gonfalon), "
                        "a staff with a crossbar on it, the three in turn, or nothing")
    f.add_argument("--flat", action="store_true",
                   help="draw the flags in a plane, holding nothing, instead of hanging them")

    a = sub.add_parser("anim", help="sprite sheet of one flag, waving: frames and a timeline")
    a.add_argument("--out", default="out/anim.png", type=Path)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--count", type=int, default=12, help="frames through one cycle")
    a.add_argument("--size", type=int, default=200, help="pixels per frame")
    a.add_argument("--cols", type=int, default=4)
    a.add_argument("--fps", type=float, default=12.0)
    a.add_argument("--padding", type=int, default=2, help="clear pixels between frames")

    b = sub.add_parser("blazon", help="print the blazon of a coat (and render it)")
    b.add_argument("--seed", type=int, default=5)
    b.add_argument("--shield", default=None)
    b.add_argument("--spec", default=None, type=Path, help="read a spec JSON instead of a seed")
    b.add_argument("--out", default=None, type=Path, help="also render to this PNG")

    pr = sub.add_parser("portrait", help="render one portrait, from a seed or a DNA string")
    pr.add_argument("--out", default="out/portrait.png", type=Path)
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--dna", default=None, help="a genome string, as printed by this command")
    pr.add_argument("--size", type=int, default=384)
    pr.add_argument("--style", default=None, choices=portrait_style.names())
    pr.add_argument("--hair", default=None, choices=list(portrait_dna.HAIR_STYLES))
    pr.add_argument("--expression", default=None, choices=list(portrait_dna.EXPRESSIONS))
    pr.add_argument("--sex", default=None, choices=list(portrait_demography.SEXES))
    pr.add_argument("--age", default=None, choices=list(portrait_demography.AGES))
    pr.add_argument("--climate", default=None, choices=list(portrait_demography.CLIMATES))
    pr.add_argument("--culture", default=None, choices=list(portrait_demography.CULTURES))
    _add_phenotype_args(pr)

    ch = sub.add_parser("character", help="full-body figure derived from a portrait genome")
    ch.add_argument("--out", default="out/character.png", type=Path)
    ch.add_argument("--seed", type=int, default=None)
    ch.add_argument("--dna", default=None, help="a genome string, as printed by portrait")
    ch.add_argument("--size", type=int, default=256)
    ch.add_argument("--cycle", default="idle", choices=figurelib.cycles())
    ch.add_argument("--facing", default="down", choices=figurelib.FACINGS)
    ch.add_argument("--sex", default=None, choices=list(portrait_demography.SEXES))
    ch.add_argument("--age", default=None, choices=list(portrait_demography.AGES))
    ch.add_argument("--climate", default=None, choices=list(portrait_demography.CLIMATES))
    ch.add_argument("--culture", default=None, choices=list(portrait_demography.CULTURES))
    _add_phenotype_args(ch)

    ps = sub.add_parser("portraits", help="contact sheet of generated portraits")
    ps.add_argument("--out", default="out/portraits.png", type=Path)
    ps.add_argument("--seed", type=int, default=7)
    ps.add_argument("--count", type=int, default=8)
    ps.add_argument("--size", type=int, default=256)
    ps.add_argument("--cols", type=int, default=4)
    ps.add_argument("--sex", default=None, choices=list(portrait_demography.SEXES))
    ps.add_argument("--age", default=None, choices=list(portrait_demography.AGES))
    ps.add_argument("--climate", default=None, choices=list(portrait_demography.CLIMATES))
    ps.add_argument("--culture", default=None, choices=list(portrait_demography.CULTURES))
    _add_phenotype_args(ps)

    ln = sub.add_parser("lineage", help="two parents and their bred children")
    ln.add_argument("--out", default="out/lineage.png", type=Path)
    ln.add_argument("--seed", type=int, default=7)
    ln.add_argument("--size", type=int, default=208)
    ln.add_argument("--children", type=int, default=6)

    t = sub.add_parser("tiles", help="tiling textures: Truchet, Wang, reaction-diffusion, boards")
    t.add_argument("--out", default="out/tiles.png", type=Path)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--no-atlas", action="store_true", help="skip the packed atlas")
    t.add_argument("--size", type=int, default=256, help="pixels per cell of the sheet")
    t.add_argument("--sim", type=int, default=80, help="cells the reaction-diffusion runs on")
    t.add_argument("--steps", type=int, default=5000, help="steps of reaction-diffusion, at most")
    t.add_argument("--mips", type=int, default=1,
                   help="levels of packed mip sheets (1 = the one atlas sheet)")

    gr = sub.add_parser("grounds", help="ground materials for map tiles: grass, dirt, "
                                       "sand, rock, snow, shore")
    gr.add_argument("--out", default="out/grounds.png", type=Path)
    gr.add_argument("--seed", type=int, default=7)
    gr.add_argument("--size", type=int, default=256, help="pixels per cell of the sheet")
    gr.add_argument("--no-atlas", action="store_true", help="skip the packed atlas")
    gr.add_argument("--mips", type=int, default=1,
                    help="levels of packed mip sheets (1 = the one atlas sheet)")

    st = sub.add_parser("site", help="a map at close range: every hex wears its "
                                     "biome's ground")
    st.add_argument("--out", default="out/site.png", type=Path)
    st.add_argument("--seed", type=int, default=7)
    st.add_argument("--size", type=int, default=56, help="pixels per hex")
    st.add_argument("--cols", type=int, default=8)
    st.add_argument("--rows", type=int, default=6)
    st.add_argument("--flat", action="store_true",
                    help="the biome colour only, without the ground")
    st.add_argument("--river-fraction", type=float, default=None,
                    help="the drainage quantile a river needs (default: 0.90 for the demo)")

    iso = sub.add_parser("iso", help="an isometric board: terraced hexes and a hero")
    iso.add_argument("--out", default="out/iso.png", type=Path)
    iso.add_argument("--seed", type=int, default=7)
    iso.add_argument("--size", type=int, default=46, help="pixels per hex")
    iso.add_argument("--cols", type=int, default=11)
    iso.add_argument("--rows", type=int, default=8)
    iso.add_argument("--sea", type=float, default=0.16,
                     help="fraction of the board that is water")
    iso.add_argument("--rise", type=float, default=2.5,
                     help="height scale, in hex sizes per unit of elevation")
    iso.add_argument("--levels", type=int, default=7,
                     help="terraces the land is quantised into")
    iso.add_argument("--hero", default=None, metavar="ROW,COL",
                     help="stand the figure on this hex (default: the middle plain)")
    iso.add_argument("--hero-seed", type=int, default=None,
                     help="the figure's own seed (default: the map's)")
    iso.add_argument("--cycle", default="idle", choices=figurelib.cycles())
    iso.add_argument("--facing", default="right", choices=figurelib.FACINGS,
                     help="right/left across the screen, down towards the camera, up away")

    tx = sub.add_parser("text", help="the generated typeface: specimen sheet and font metrics")
    tx.add_argument("--out", default="out/text.png", type=Path)
    tx.add_argument("--size", type=int, default=20, help="pixels per em for the specimen rows")
    tx.add_argument("--face", action="append", default=None, choices=text_pen.names(),
                    help="a face to show; repeat it for several (default: all five)")
    tx.add_argument("--width", type=int, default=1400, help="pixels wide")

    fa = sub.add_parser("factions", help="a faction from its name: shield, banner, blazon")
    fa.add_argument("--out", default="out/factions.png", type=Path)
    fa.add_argument("--seed", type=int, default=7, help="only labels the sheet; the names decide")
    fa.add_argument("--size", type=int, default=168, help="pixels per tile")
    fa.add_argument("--ratio", default="3:2", help="flag proportion, e.g. 3:2 or 5:8")
    fa.add_argument("--name", action="append", default=None,
                    help="a faction name; repeat it for several (default: a built-in demo set)")

    u = sub.add_parser("ui", help="UI kit sheet: panels, frames, buttons, nine-slice cuts")
    u.add_argument("--out", default="out/ui.png", type=Path)
    u.add_argument("--seed", type=int, default=7)
    u.add_argument("--count", type=int, default=3, help="how many seeded kits to add")
    u.add_argument("--scale", type=int, default=2, help="pixel density of the kit")
    u.add_argument("--style", default=None, choices=ui_style.names(),
                   help="render one named style instead of all of them")
    u.add_argument("--svg", action="store_true",
                   help="also write each kit's vector (SVG) form beside the sheet")

    m = sub.add_parser("maps", help="hex maps: terrain, relief and coastline")
    m.add_argument("--out", default="out/maps.png", type=Path)
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--count", type=int, default=2, help="how many seeded maps")
    m.add_argument("--cols", type=int, default=26, help="hexes across")
    m.add_argument("--rows", type=int, default=17, help="hexes down")
    m.add_argument("--size", type=float, default=13.0, help="hex circumradius in pixels")
    m.add_argument("--sea", type=float, default=0.42, help="fraction of the map that is water")
    m.add_argument("--provinces", type=int, default=8, help="provinces per map, 0 for none")
    m.add_argument("--river", type=float, default=0.92, help="river strength, 0 for none")
    m.add_argument("--road", type=float, default=0.85, help="road opacity, 0 for none")
    m.add_argument("--seas", type=int, default=8, help="sea provinces, 0 for none")
    m.add_argument("--factions", type=int, default=4, help="realms holding the provinces, 0 for none")
    m.add_argument("--no-names", action="store_true",
                   help="leave the province names out of the map")

    co = sub.add_parser("coast", help="the coastline as a tileset: 64 edge tiles in a sheet")
    co.add_argument("--out", default="out/coast-tiles.png", type=Path)
    co.add_argument("--seed", type=int, default=7, help="only counts what the maps wear")
    co.add_argument("--count", type=int, default=2, help="how many seeded maps to count")
    co.add_argument("--size", type=float, default=20.0, help="hex circumradius in the sheet")
    co.add_argument("--hex", type=float, default=13.0, dest="hex_size",
                    help="hex circumradius the counted maps are drawn at")
    co.add_argument("--cols", type=int, default=26, help="hexes across, for the counting")
    co.add_argument("--rows", type=int, default=17, help="hexes down, for the counting")
    co.add_argument("--sea", type=float, default=0.42, help="water fraction of the counted maps")
    co.add_argument("--inland", type=float, default=0.55,
                    help="opacity of the biome-border panel, 0 to leave it out")

    cr = sub.add_parser("credits", help="what we owe, and the manifest of what we made")
    cr.add_argument("--out", default=None, type=Path,
                    help="write credits.txt here (default: print it)")
    cr.add_argument("--png", default=None, type=Path,
                    help="also draw the credits screen to this PNG")
    cr.add_argument("--size", type=float, default=24.0,
                    help="the screen's em, in pixels (drawn, not scaled)")
    cr.add_argument("--cols", type=int, default=48,
                    help="characters per line on the screen")
    cr.add_argument("--manifest", default="out", type=Path,
                    help="directory whose sidecars list the generated assets")
    cr.add_argument("--check", action="store_true",
                    help="check the build before it ships instead of printing the block")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "demo":
        demo(args.out, seed=args.seed)
    elif args.cmd == "icons":
        demo_icons(args.out, seed=args.seed, count=args.count, size=args.size,
                   cols=args.cols, sweep=_ints(args.sizes), atlas=not args.no_atlas,
                   factions=args.name, family=args.family, wear=not args.no_wear,
                   wear_size=args.wear_size)
    elif args.cmd == "wardrobe":
        demo_wardrobe(args.out, seed=args.seed, count=args.count, size=args.size,
                      cols=args.cols, cut=args.cut, pattern=args.pattern,
                      cloth=args.cloth, trim=args.trim, metal=args.metal)
    elif args.cmd == "decorations":
        demo_decorations(args.out, seed=args.seed, count=args.count, size=args.size,
                         cols=args.cols, kind=args.kind, metal=args.metal,
                         motif=args.motif, bars=args.bars, rosette=args.rosette)
    elif args.cmd == "shapes":
        demo_shapes(args.out, size=args.size, cols=args.cols)
    elif args.cmd == "texture":
        demo_texture(args.out, seed=args.seed, size=args.size)
    elif args.cmd == "palette":
        demo_palette(args.out, seed=args.seed, count=args.count)
    elif args.cmd == "arms":
        demo_arms(args.out, seed=args.seed, count=args.count, size=args.size, cols=args.cols,
                  shield=args.shield)
    elif args.cmd == "flags":
        demo_flags(args.out, seed=args.seed, count=args.count, size=args.size, cols=args.cols,
                   ratio=_ratio(args.ratio), shape=args.shape, wave=not args.flat,
                   hang="none" if args.flat else args.hang)
    elif args.cmd == "anim":
        demo_anim(args.out, seed=args.seed, count=args.count, size=args.size,
                  cols=args.cols, fps=args.fps, padding=args.padding)
    elif args.cmd == "studio":
        from studio.server import serve
        serve(host=args.host, port=args.port, root=args.root,
              open_browser=not args.no_browser)
    elif args.cmd == "gallery":
        if args.stats:
            print(json.dumps(gallerylib.stats(args.manifest), indent=2, sort_keys=True))
            return 0
        out = args.out if args.out is not None else args.manifest / gallerylib.PAGE_NAME
        _report(gallerylib.write(args.manifest, out, title=args.title, embed=args.embed))
    elif args.cmd == "figures":
        demo_figures(args.out, seed=args.seed, cycle=args.cycle, count=args.count,
                     size=args.size, cols=args.cols, fps=args.fps,
                     facing=args.facing)
    elif args.cmd == "blazon":
        demo_blazon(seed=args.seed, spec_path=args.spec, out=args.out, shield=args.shield)
    elif args.cmd == "portrait":
        portrait_one(args.out, seed=args.seed, dna=args.dna, size=args.size, style=args.style,
                     hair=args.hair, expression=args.expression, sex=args.sex, age=args.age,
                     climate=args.climate, culture=args.culture,
                     overrides=_phenotype_overrides(args))
    elif args.cmd == "character":
        character_one(args.out, seed=args.seed, dna=args.dna, size=args.size,
                      cycle=args.cycle, facing=args.facing, sex=args.sex,
                      age=args.age, climate=args.climate, culture=args.culture,
                      overrides=_phenotype_overrides(args))
    elif args.cmd == "portraits":
        demo_portraits(args.out, seed=args.seed, count=args.count, size=args.size, cols=args.cols,
                       sex=args.sex, age=args.age, climate=args.climate, culture=args.culture,
                       overrides=_phenotype_overrides(args))
    elif args.cmd == "lineage":
        demo_lineage(args.out, seed=args.seed, size=args.size, children=args.children)
    elif args.cmd == "ui":
        if args.svg:
            demo_vectors(args.out.parent, seed=args.seed, count=args.count,
                         scale=args.scale, style=args.style, prefix=args.out.stem)
        demo_ui(args.out, seed=args.seed, count=args.count, scale=args.scale,
                style=args.style)
    elif args.cmd == "maps":
        demo_maps(args.out, seed=args.seed, count=args.count, cols=args.cols,
                  rows=args.rows, size=args.size, sea=args.sea,
                  provinces=args.provinces, rivers=args.river,
                  factions=args.factions, road=args.road,
                  names=not args.no_names, sea_provinces=args.seas)
    elif args.cmd == "coast":
        demo_coast(args.out, seed=args.seed, count=args.count, cols=args.cols,
                   rows=args.rows, hex_size=args.hex_size, sea=args.sea,
                   size=args.size, inland=args.inland)
    elif args.cmd == "text":
        # a face has no seed: the alphabet and the pen are the whole of it, and the
        # sidecar says so with a seed of zero rather than inventing one
        demo_text(args.out, size=args.size, faces=args.face, width=args.width)
    elif args.cmd == "factions":
        demo_factions(args.out, names=args.name, size=args.size, seed=args.seed,
                      ratio=_ratio(args.ratio))
    elif args.cmd == "tiles":
        demo_tiles(args.out, seed=args.seed, size=args.size, sim=args.sim,
                   steps=args.steps, atlas=not args.no_atlas, mips=args.mips)
    elif args.cmd == "grounds":
        demo_grounds(args.out, seed=args.seed, size=args.size,
                     atlas=not args.no_atlas, mips=args.mips)
    elif args.cmd == "site":
        demo_site(args.out, seed=args.seed, size=args.size, cols=args.cols,
                  rows=args.rows, flat=args.flat,
                  river_fraction=args.river_fraction)
    elif args.cmd == "iso":
        try:
            demo_iso(args.out, seed=args.seed, cols=args.cols, rows=args.rows,
                     size=args.size, sea=args.sea, rise=args.rise, levels=args.levels,
                     hero=None if args.hero is None else _rc(args.hero),
                     hero_seed=args.hero_seed, cycle=args.cycle,
                     facing=args.facing)
        except ValueError as exc:
            # A named hex can be off the board or in the water.  That is a user
            # mistake with a one-line answer, not a traceback.
            raise SystemExit(f"iso: {exc}") from None
    elif args.cmd == "credits":
        if args.out is not None:
            _report(creditslib.write(args.manifest, args.out))
        if args.png is not None:
            _report(creditslib.write_screen(args.manifest, args.png,
                                            size=args.size, cols=args.cols))
        if args.check:
            return _credits_check(args.manifest)
        if args.out is None:
            print(creditslib.credits(args.manifest), end="")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
