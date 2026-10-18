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
    python cli.py flags   --out out/flags.png --seed 7 --count 12 [--flat]  # on their poles
    python cli.py blazon  --seed 5 [--spec spec.json] [--out out/arms_one.png]
    python cli.py portrait   --seed 5 --size 384 --out out/portrait.png
    python cli.py portraits  --seed 7 --count 8 --size 256
    python cli.py lineage    --seed 7 --children 6
    python cli.py ui         --seed 7 --count 3 --scale 2
    python cli.py maps       --seed 7 --count 2 --cols 26 --rows 17 --provinces 8
    python cli.py credits    --out out --manifest out
    python cli.py credits    --manifest out --png out/credits.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import credits as creditslib  # noqa: E402
from core import noise, palette, render, sdf  # noqa: E402
from core.canvas import Canvas  # noqa: E402
from export import atlas as atlaslib  # noqa: E402
from export import png  # noqa: E402
from gen.heraldry import blazon as heraldry  # noqa: E402
from gen.heraldry import crest as crestlib  # noqa: E402
from gen.heraldry import flags as flaglib  # noqa: E402
from gen.heraldry import tinctures as tinctureslib  # noqa: E402
from gen.icons import motifs as icon_motifs  # noqa: E402
from gen.icons import paint as icon_paint  # noqa: E402
from gen.maps import grid as map_grid  # noqa: E402
from gen.maps import paint as map_paint  # noqa: E402
from gen.maps import region as map_region  # noqa: E402
from gen.maps import terrain as map_terrain  # noqa: E402
from gen.textures import reaction as tex_reaction  # noqa: E402
from gen.textures import tile as tilelib  # noqa: E402
from gen.textures import truchet as tex_truchet  # noqa: E402
from gen.textures import wang as tex_wang  # noqa: E402
from gen.portraits import dna as portrait_dna  # noqa: E402
from gen.portraits import portrait as portraitlib  # noqa: E402
from gen.portraits import style as portrait_style  # noqa: E402
from gen.ui import frame as ui_frame  # noqa: E402
from gen.ui import paint as ui_paint  # noqa: E402
from gen.ui import slice as ui_slice  # noqa: E402
from gen.ui import style as ui_style  # noqa: E402

# A single light direction for the whole project.  Consistency of lighting is
# most of what makes individually generated assets read as one art set.
LIGHT = (-0.45, 0.65)


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
def make_icon(seed: int, size: int = 192) -> Canvas:
    """One complete icon from a seed -- the CLI's smallest door into `gen.icons`.

    This used to be the icon generator.  It is one line now, deliberately: the
    charge vocabulary, the plate and the five passes live in the package, and a
    second implementation left behind in the CLI is how the two drift apart.
    """
    return icon_paint.icon(seed, size)


def demo_icons(out: Path, seed: int = 3, count: int = 8, size: int = 192, cols: int = 9,
               sweep=(32, 64, 128), atlas: bool = True) -> Path:
    """The icon set, seeds drawn from it, one icon at three sizes, and the atlas.

    The first tiles are the *set*: every charge, one plate, one hue family, which is
    what makes a folder of icons look like one game's icons.  Then seeds, which is
    what a generator adds to a set.  Then the same icon at 32, 64 and 128 px,
    *generated* rather than resized -- generating a sweep is only worth doing if the
    small drawing is drawn small, and that is exactly what designing in world units
    buys.
    """
    set_tiles = [(name, icon_paint.draw(icon_motifs.canonical(name), size))
                 for name in icon_motifs.KINDS]
    seed_tiles = [(f"seed {seed + i}", make_icon(seed + i, size)) for i in range(count)]
    sweep_tiles = [(f"blade at {s}px", icon_paint.draw(icon_motifs.canonical("blade"), s))
                   for s in sweep]
    tiles = set_tiles + seed_tiles + sweep_tiles
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
             set=list(icon_motifs.KINDS),
             seeded=[seed + i for i in range(count)],
             sizes=list(sweep))

    # The atlas is what a game actually loads: one image and a table of rectangles.
    # Its density is in the sidecar, because how much of the sheet is sprite rather
    # than gap is the number that decides whether the padding was worth paying for.
    if atlas:
        image, info = atlaslib.build(set_tiles + seed_tiles, padding=2)
        path = out.with_name(f"{out.stem}-atlas.png")
        image.save(path)
        _sidecar(path, seed, "procedural:export.atlas", **info)
        _report(path)
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
        c.over(render.shade(colors[i], d, dx=c.dx, bevel=0.09, light=LIGHT), render.coverage(d, c.dx))
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


def demo_flags(out: Path, seed: int = 7, count: int = 12, size: int = 200, cols: int = 4,
               ratio=(3, 2), shape: str = "rectangle", wave: bool = True) -> Path:
    """Contact sheet of generated flags, described vexillologically.

    The flags hang by default -- from a pole, because a flag that hangs hangs from
    something: the pole stands the cloth up, anchors its hoist and lets the fly
    trail.  One is asked for cloth with ``--flat``, because a flag on a pole is what
    the asset is and a flat flag is the special case.
    """
    specs = [flaglib.random_flag(seed=seed + i, ratio=ratio, shape=shape, wave=wave,
                                 pole=wave)
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
            json.dumps({"seed": seed, "spec": spec}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _report(meta)
    return spec


def demo_portraits(out: Path, seed: int = 7, count: int = 8, size: int = 256, cols: int = 4) -> Path:
    """Contact sheet of generated portraits, captioned with their genome.

    The sidecar ``.json`` is the point: every tile carries the one-line DNA
    string that would reproduce it exactly, so a sheet of avatars is also a
    browsable catalogue of the genomes that produced them.
    """
    genomes = [portrait_dna.random_genes(seed + i) for i in range(count)]
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
                "portraits": [
                    {
                        "seed": seed + i,
                        "dna": portrait_dna.encode(g),
                        "description": portrait_dna.describe(g),
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
                 expression: str | None = None) -> Path:
    """Render one portrait, from a seed or from a DNA string, and print its genome."""
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
    genes = portrait_dna.decode(portrait_dna.encode(genes))  # re-clamp through the wire format

    text = portrait_dna.encode(genes)
    print(f"seed {seed}")
    print(f"  {portrait_dna.describe(genes)}")
    print(f"  {text}")
    img = png.to_image(portraitlib.render(genes, size=size))
    img.save(out, "PNG", optimize=True)
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({"seed": seed, "dna": text, "genes": genes}, indent=2) + "\n",
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


def demo_maps(out: Path, seed: int = 7, count: int = 2, cols: int = 26, rows: int = 17,
              size: float = 13.0, sea: float = 0.42, provinces: int = 8,
              rivers: float = 0.92, names: bool = True) -> Path:
    """Contact sheet of maps: the geography is the seed, and nothing else.

    Each tile is labelled with the seed that makes it, the grid it is played on, the
    water fraction and the commonest ground -- and then with its *provinces*, whose
    names are drawn at their capitals.  Those names are drawn here and not in the
    asset: this project ships exactly one font, Pillow's, and a game has its own,
    so text belongs to the preview while the sidecar carries the data a game would
    label with.
    """
    tiles = []
    for i in range(count):
        s = seed + i
        g = map_grid.HexGrid(cols, rows, size)
        f = map_terrain.fields(g, seed=s, sea_fraction=sea)
        st = map_terrain.stats(f["biome"])
        courses, _threshold = map_paint.river_courses(g, f) if rivers > 0.0 else ([], 0.0)
        res = map_region.provinces(g, f, count=provinces, seed=s) if provinces > 0 else None
        img = map_paint.map_image(seed=s, cols=cols, rows=rows, size=size, sea_fraction=sea,
                                  provinces=provinces, river=rivers)
        tiles.append((s, img, st, courses, res, g))

    gap = 24
    sheet = render.sheet([t[1] for t in tiles], cols=min(2, len(tiles)), gap=gap,
                         background=(0.90, 0.90, 0.93))
    sheet.save(out)
    img = png.to_image(sheet)
    cw = max(t[1].w for t in tiles)
    ch = max(t[1].h for t in tiles)
    per_row = min(2, len(tiles))
    font = _font(11)
    for i, (s, tile, st, courses, res, g) in enumerate(tiles):
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
        _label(img, (tx + 3, ty + tile.h + 4),
               f"seed {s}   {cols}x{rows} hexes at {size:g} px   " + "   ".join(counts))
        if names and res is not None:
            # The tile is a canvas in world coordinates centred on its own origin,
            # y up: the same mapping the rasteriser inverts.
            for p in res["provinces"]:
                if p["area"] < 3:
                    continue  # a one-hex island cannot hold a word
                x, y = g.centre_of(p["capital"])
                px = tx + int(round(x + tile.w / 2.0 - 0.5))
                py = ty + int(round(tile.h / 2.0 - 0.5 - y))
                _centre_label(img, (px - 60, py - 8, px + 60, py + 8), p["name"],
                              dark=(20, 20, 26, 255), font=font)
    img.save(out, "PNG", optimize=True)

    _sidecar(out, seed, "procedural:gen.maps", maps=[
        {
            "seed": s, "cols": cols, "rows": rows, "hex_size": size,
            "sea_target": sea, "water": st["water"], "biomes": st,
            "rivers": [{"cells": int(c["cells"]), "drainage": round(float(c["flux"]), 1)}
                       for c in courses],
            "provinces": ([
                {"name": p["name"], "area": int(p["area"]), "capital": [int(p["capital"][0]),
                                                                       int(p["capital"][1])],
                 "neighbours": [int(n) for n in p["neighbours"]], "ground": p["ground"],
                 "coastal": bool(p["coastal"])}
                for p in res["provinces"]] if res is not None else []),
        }
        for s, _, st, courses, res, _g in tiles
    ])
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


def demo_tiles(out: Path, seed: int = 7, size: int = 256, sim: int = 80,
               steps: int = 5000) -> Path:
    """Contact sheet of tiling textures: Truchet, Wang and reaction-diffusion.

    Every cell is the tile laid down **four times** rather than shown once.  That is
    the only way to see whether a texture tiles: one copy of it is seamless by
    definition, and a seam only exists between two of them.  The sidecar records the
    seam error of each tile, which is the number this sheet is here to justify.
    """
    half = int(size) // 2
    panels = [
        ("truchet, arcs", tex_truchet.texture(seed=seed, size=half, grains=4, kinds=("arcs",))),
        ("truchet, mixed", tex_truchet.texture(seed=seed + 1, size=half, grains=4, kinds=("arcs", "lines"),
                                           mix=0.5)),
        ("wang, meadow", tex_wang.texture(seed=seed, size=half, cells=half // 16, colours=3)),
        ("wang, stone", tex_wang.texture(seed=seed + 2, size=half, cells=half // 16, colours=4,
                                     palette_name="stone")),
        ("reaction, worms", tex_reaction.texture(seed=seed, size=half, sim=sim,
                                             preset_name="worms", steps=steps)),
        ("reaction, holes", tex_reaction.texture(seed=seed, size=half, sim=sim,
                                             preset_name="holes", steps=steps)),
    ]

    gap = 22
    per_row = 3
    sheet = render.sheet([_repeat_canvas(panel, 2) for _name, panel in panels], cols=per_row,
                         gap=gap, background=(0.90, 0.90, 0.93))
    sheet.save(out)
    img = png.to_image(sheet)
    cw = int(size)
    ch = int(size)
    for i, (name, _cv) in enumerate(panels):
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
                tilelib.luminance(np.tile(panel.to_uint8(), (2, 2, 1)))), 3),
        }
        for name, panel in panels
    ])
    return _report(out)


def _repeat_canvas(canvas, n: int = 2):
    """A canvas laid out ``n`` by ``n``: the picture that shows a tile wraps."""
    out = Canvas(canvas.w * int(n), canvas.h * int(n), scale=canvas.scale,
                 y_up=canvas.y_up)
    for i in range(int(n)):
        for j in range(int(n)):
            out.blit(canvas.rgb, canvas.a, j * canvas.w, i * canvas.h)
    return out


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
        names = ["Kingdom of Alder", "The Mirefen League", "Brythmark",
                 "Silverreach Compact", "Duchy of Halven", "The Ostmark Freeholds",
                 "Vale of Craymoor", "House Delvane"]
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
    demo_flags(out / "flags.png", seed=seed, count=12)
    demo_portraits(out / "portraits.png", seed=seed, count=8, size=256)
    demo_lineage(out / "lineage.png", seed=seed)
    demo_ui(out / "ui.png", seed=seed, count=2, scale=2)
    demo_maps(out / "maps.png", seed=seed, count=2)
    demo_tiles(out / "tiles.png", seed=seed)
    demo_factions(out / "factions.png", seed=seed)
    _report(creditslib.write(out))
    _report(creditslib.write_screen(out))
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

    i = sub.add_parser("icons", help="icon set, seeds, size sweep and a packed atlas")
    i.add_argument("--out", default="out/icons.png", type=Path)
    i.add_argument("--seed", type=int, default=3)
    i.add_argument("--count", type=int, default=8)
    i.add_argument("--size", type=int, default=192, help="size of the set and seed tiles")
    i.add_argument("--cols", type=int, default=9, help="tiles per row")
    i.add_argument("--sizes", default="32,64,128",
                   help="comma-separated sizes to generate for the sweep")
    i.add_argument("--no-atlas", action="store_true", help="skip the packed atlas")

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
    f.add_argument("--flat", action="store_true",
                   help="draw the flags in a plane, without a pole, instead of hanging them")

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

    ps = sub.add_parser("portraits", help="contact sheet of generated portraits")
    ps.add_argument("--out", default="out/portraits.png", type=Path)
    ps.add_argument("--seed", type=int, default=7)
    ps.add_argument("--count", type=int, default=8)
    ps.add_argument("--size", type=int, default=256)
    ps.add_argument("--cols", type=int, default=4)

    ln = sub.add_parser("lineage", help="two parents and their bred children")
    ln.add_argument("--out", default="out/lineage.png", type=Path)
    ln.add_argument("--seed", type=int, default=7)
    ln.add_argument("--size", type=int, default=208)
    ln.add_argument("--children", type=int, default=6)

    t = sub.add_parser("tiles", help="tiling textures: Truchet, Wang tiles, reaction-diffusion")
    t.add_argument("--out", default="out/tiles.png", type=Path)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--size", type=int, default=256, help="pixels per cell of the sheet")
    t.add_argument("--sim", type=int, default=80, help="cells the reaction-diffusion runs on")
    t.add_argument("--steps", type=int, default=5000, help="steps of reaction-diffusion, at most")

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

    cr = sub.add_parser("credits", help="what we owe, and the manifest of what we made")
    cr.add_argument("--out", default=None, type=Path,
                    help="write credits.txt here (default: print it)")
    cr.add_argument("--png", default=None, type=Path,
                    help="also draw the credits screen to this PNG")
    cr.add_argument("--scale", type=int, default=3,
                    help="screen pixel size relative to the built-in font")
    cr.add_argument("--cols", type=int, default=48,
                    help="characters per line on the screen")
    cr.add_argument("--manifest", default="out", type=Path,
                    help="directory whose sidecars list the generated assets")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "demo":
        demo(args.out, seed=args.seed)
    elif args.cmd == "icons":
        demo_icons(args.out, seed=args.seed, count=args.count, size=args.size,
                   cols=args.cols, sweep=_ints(args.sizes), atlas=not args.no_atlas)
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
                   ratio=_ratio(args.ratio), shape=args.shape, wave=not args.flat)
    elif args.cmd == "blazon":
        demo_blazon(seed=args.seed, spec_path=args.spec, out=args.out, shield=args.shield)
    elif args.cmd == "portrait":
        portrait_one(args.out, seed=args.seed, dna=args.dna, size=args.size, style=args.style,
                     hair=args.hair, expression=args.expression)
    elif args.cmd == "portraits":
        demo_portraits(args.out, seed=args.seed, count=args.count, size=args.size, cols=args.cols)
    elif args.cmd == "lineage":
        demo_lineage(args.out, seed=args.seed, size=args.size, children=args.children)
    elif args.cmd == "ui":
        demo_ui(args.out, seed=args.seed, count=args.count, scale=args.scale,
                style=args.style)
    elif args.cmd == "maps":
        demo_maps(args.out, seed=args.seed, count=args.count, cols=args.cols,
                  rows=args.rows, size=args.size, sea=args.sea,
                  provinces=args.provinces, rivers=args.river)
    elif args.cmd == "factions":
        demo_factions(args.out, names=args.name, size=args.size, seed=args.seed,
                      ratio=_ratio(args.ratio))
    elif args.cmd == "tiles":
        demo_tiles(args.out, seed=args.seed, size=args.size, sim=args.sim, steps=args.steps)
    elif args.cmd == "credits":
        if args.out is None:
            print(creditslib.credits(args.manifest), end="")
        else:
            _report(creditslib.write(args.manifest, args.out))
        if args.png is not None:
            _report(creditslib.write_screen(args.manifest, args.png,
                                            scale=args.scale, cols=args.cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
