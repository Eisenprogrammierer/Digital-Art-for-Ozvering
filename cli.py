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
    python cli.py flags   --out out/flags.png --seed 7 --count 12
    python cli.py blazon  --seed 5 [--spec spec.json] [--out out/arms_one.png]
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

from core import noise, palette, render, sdf  # noqa: E402
from core.canvas import Canvas  # noqa: E402
from export import png  # noqa: E402
from gen.heraldry import blazon as heraldry  # noqa: E402
from gen.heraldry import flags as flaglib  # noqa: E402

# A single light direction for the whole project.  Consistency of lighting is
# most of what makes individually generated assets read as one art set.
LIGHT = (-0.45, 0.65)

# Heraldic shield outline, used as one of the icon silhouettes.
SHIELD = np.array(
    [[-0.62, 0.80], [0.62, 0.80], [0.62, 0.10], [0.00, -0.84], [-0.62, 0.10]],
    dtype=np.float64,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
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
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle([l - 6, t - 4, r + 6, b + 4], radius=5, fill=plate)
    draw.text((x, y), text, fill=dark, font=font)


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


def _unit3(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


# --------------------------------------------------------------------------
# icon generator: the reference "asset from nothing" demo
# --------------------------------------------------------------------------
def make_icon(seed: int, size: int = 192) -> Canvas:
    """One complete icon from a seed: silhouette, carved motif, bevel, shadow."""
    rng = np.random.default_rng(seed)
    colors = palette.random_palette(rng, n=3)
    base, accent, recess_color = colors[0], colors[1], colors[2]

    c = Canvas(size, size, scale=size / 2.0 * 0.80)
    X, Y = c.grid()

    silhouette = int(rng.integers(0, 4))
    if silhouette == 0:
        outer = lambda x, y: sdf.circle(x, y, 0.78)  # noqa: E731
    elif silhouette == 1:
        outer = lambda x, y: sdf.box(x, y, 0.70, 0.70, 0.18)  # noqa: E731
    elif silhouette == 2:
        rot = float(rng.uniform(0.0, 60.0))
        outer = lambda x, y: sdf.regular_polygon(x, y, 6, 0.82, rotation=rot)  # noqa: E731
    else:
        outer = lambda x, y: sdf.polygon(x, y, SHIELD)  # noqa: E731

    motif_kind = int(rng.integers(0, 4))
    if motif_kind == 0:
        motif = lambda x, y: sdf.circle(x, y, 0.33, 0.0, 0.04)  # noqa: E731
    elif motif_kind == 1:
        rot = float(rng.uniform(0.0, 72.0))
        motif = lambda x, y: sdf.star(x, y, 5, 0.40, 0.17, rotation=rot)  # noqa: E731
    elif motif_kind == 2:
        motif = lambda x, y: sdf.box(x, y, 0.30, 0.30, 0.07, 0.0, 0.04)  # noqa: E731
    else:
        motif = lambda x, y: sdf.segment(x, y, (-0.34, -0.30), (0.34, 0.32), 0.10)  # noqa: E731

    d_outer = outer(X, Y)
    d_motif = motif(X, Y)
    shape = sdf.smooth_difference(d_outer, d_motif, 0.07)  # emblem carved out
    rim = sdf.outline(outer, 0.028)(X, Y)  # thin bright border

    out = Canvas(size, size, scale=c.scale)
    shadow = render.drop_shadow(
        render.coverage(d_outer, c.dx), c.scale, offset=(0.0, -0.055), sigma=0.03, opacity=0.35
    )
    out.over((0.0, 0.0, 0.0), shadow)
    out.over(render.shade(base, shape, dx=c.dx, bevel=0.10, light=LIGHT), render.coverage(shape, c.dx))

    inside = render.coverage(d_outer, c.dx)  # keep the overlays inside the icon
    recess = np.clip(-d_motif / 0.05, 0.0, 1.0) * inside
    out.over_srgb(recess_color, recess * 0.55)
    out.over_srgb(accent, render.coverage(rim, c.dx))
    return out


# --------------------------------------------------------------------------
# demos
# --------------------------------------------------------------------------
def demo_icons(out: Path, seed: int = 3, count: int = 8, size: int = 192, cols: int = 4) -> Path:
    """Contact sheet of seed-driven icons -- proof the generator has range."""
    icons = [make_icon(seed + i, size) for i in range(count)]
    sheet = render.sheet(icons, cols=cols, gap=10, background=(0.94, 0.94, 0.96))
    return sheet.save(out)


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
               ratio=(3, 2), shape: str = "rectangle") -> Path:
    """Contact sheet of generated flags, described vexillologically."""
    specs = [flaglib.random_flag(seed=seed + i, ratio=ratio, shape=shape) for i in range(count)]
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


def demo(out: Path, seed: int = 7) -> None:
    """Render everything into ``out/`` -- the smoke test for the whole core."""
    print(f"ozart demo  (seed={seed}, out={out})")
    _report(demo_icons(out / "icons.png", seed=seed, count=8))
    demo_shapes(out / "shapes.png")
    demo_texture(out / "texture.png", seed=seed, size=256)
    demo_palette(out / "palette.png", seed=seed, count=6)
    demo_arms(out / "arms.png", seed=seed, count=12)
    demo_flags(out / "flags.png", seed=seed, count=12)
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

    i = sub.add_parser("icons", help="contact sheet of procedural icons")
    i.add_argument("--out", default="out/icons.png", type=Path)
    i.add_argument("--seed", type=int, default=3)
    i.add_argument("--count", type=int, default=8)
    i.add_argument("--size", type=int, default=192)
    i.add_argument("--cols", type=int, default=4)

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

    b = sub.add_parser("blazon", help="print the blazon of a coat (and render it)")
    b.add_argument("--seed", type=int, default=5)
    b.add_argument("--shield", default=None)
    b.add_argument("--spec", default=None, type=Path, help="read a spec JSON instead of a seed")
    b.add_argument("--out", default=None, type=Path, help="also render to this PNG")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "demo":
        demo(args.out, seed=args.seed)
    elif args.cmd == "icons":
        _report(demo_icons(args.out, seed=args.seed, count=args.count, size=args.size, cols=args.cols))
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
        ratio = tuple(int(v) for v in str(args.ratio).replace(":", "x").split("x"))
        demo_flags(args.out, seed=args.seed, count=args.count, size=args.size, cols=args.cols,
                   ratio=ratio, shape=args.shape)
    elif args.cmd == "blazon":
        demo_blazon(seed=args.seed, spec_path=args.spec, out=args.out, shield=args.shield)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
