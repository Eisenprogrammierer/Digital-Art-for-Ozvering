"""Who to thank: the curated sources, and the manifest of everything we made.

Almost every asset in this repository is ours in the strong sense -- it is a
function of a seed, and the sidecar written next to it records the seed, so the
pixels can be regenerated from a few kilobytes of text and nothing is owed to
anyone.  Music is not like that.  It is a real recording by a real person, and
its licence is the one thing in the tree that can be broken by *forgetting*: an
attribution that is not in the build is a licence violation even if the file is
present and the credit is in a README nobody ships.

So there are two halves.  :data:`SOURCES` is the short, hand-written list of
things we did not make, each with the exact credit line its licence demands.
:func:`manifest` is the generated half: it walks an output directory and reads
the ``source`` and ``license`` fields the exporters already write into every
sidecar.  :func:`credits` puts the two together into the block a credits screen
is made of, :func:`write` drops it in the build as text, and :func:`screen` draws
it -- because the artefact a game has to ship is a screen, and a credit line that
was generated correctly but never drawn is the same violation as no credit at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Everything in the tree that is not generated from a seed.
#:
#: A source is a dict with ``id``, ``title``, ``author``, ``license``, ``url`` and
#: ``attribution``.  The ``attribution`` string is the one that has to survive into
#: the shipped credits *verbatim*: it is what the licence asks for, so it is stored
#: in the form it has to appear rather than assembled at print time.
SOURCES = (
    {
        "id": "kevin-macleod",
        "title": "Music",
        "author": "Kevin MacLeod",
        "license": "CC BY 4.0",
        "url": "https://incompetech.com/",
        "attribution": (
            'Music by Kevin MacLeod (incompetech.com), licensed under '
            'Creative Commons Attribution 4.0.'
        ),
    },
)


def sources() -> list:
    """The curated sources, as a list of copies."""
    return [dict(s) for s in SOURCES]


def missing_fields() -> list:
    """Sources whose licence cannot be honoured as written.

    A credit line needs a correct spelling of what it names, which is why this is
    a validation function rather than a comment: CC BY needs the author, the title
    and the licence, and a URL is what makes it checkable.
    """
    required = ("id", "title", "author", "license", "url", "attribution")
    return [dict(s) for s in SOURCES if any(not s.get(k) for k in required)]


def manifest(directory) -> list:
    """The generated assets under a directory, read from their sidecars.

    Generators differ in what they call their list (``"portraits"``, ``"arms"``,
    ``"flags"``), so this reads the *envelope* instead: a JSON file with a
    ``source`` is a sidecar, and its seed and licence describe everything in it.
    Keeping the reading generic means a new generator is covered the day it starts
    exporting a sidecar, with no registry to update and no chance of forgetting.
    """
    out = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or "source" not in meta:
            continue
        counts = [v for k, v in meta.items() if isinstance(v, list)]
        out.append(
            {
                "sidecar": path.name,
                "image": path.with_suffix(".png").name,
                "source": meta.get("source", ""),
                "license": meta.get("license", ""),
                "seed": meta.get("seed"),
                "count": max((len(v) for v in counts), default=0),
            }
        )
    return out


def credits(directory=None) -> str:
    """The credits block: what we owe, then what we made."""
    lines = ["ozart -- credits", ""]
    lines.append("Music and other non-generated assets")
    for s in SOURCES:
        lines.append(f"  {s['attribution']}")
        lines.append(f"    ({s['title']}, {s['license']}, {s['url']})")
    lines.append("")
    lines.append("Generated assets")
    lines.append(
        "  Everything else here is generated: a function of a seed, reproduced "
        "from the\n  sidecar beside it.  No third party holds rights in it."
    )
    if directory is not None:
        entries = manifest(directory)
        for e in entries:
            seed = "-" if e["seed"] is None else str(e["seed"])
            count = f" x{e['count']}" if e["count"] else ""
            lines.append(f"    {e['image']}{count}  seed {seed}  {e['license']}")
        if not entries:
            lines.append(f"    (no sidecars found in {directory})")
    lines.append("")
    return "\n".join(lines)


def write(directory, path=None) -> Path:
    """Write the credits into a build directory.  Returns the file written."""
    directory = Path(directory)
    path = Path(path) if path is not None else directory / "credits.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credits(directory), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------
def screen_lines(directory=None, cols: int = 48) -> list:
    """The credits as rows of ``(kind, text)``, wrapped to ``cols`` characters.

    Built from the *data* rather than by parsing :func:`credits`, because a screen
    has to be able to style what it draws -- the credit line a licence asks for is
    a different thing from a note about seeds and should not look like one.

    The wrapping is where a credits screen actually goes wrong.  A licence line
    that runs off the edge of a texture is a licence line that is not there, and
    nothing about a rendered image says so, so the wrapping is its own function and
    a test reads it back.
    """
    import textwrap

    rows = []

    def wrap(kind, text):
        rows.extend((kind, part) for part in textwrap.wrap(text, cols) or [""])

    wrap("title", "ozart")
    wrap("head", "Music and other non-generated assets")
    for s in SOURCES:
        wrap("credit", s["attribution"])
        wrap("note", f"({s['title']}, {s['license']}, {s['url']})")

    wrap("head", "Generated assets")
    wrap("note", "Everything else here is generated: a function of a seed, "
                 "reproduced from the sidecar beside it.  No third party holds "
                 "rights in it.")
    rows.append(("note", ""))  # the list needs air above it on a screen
    if directory is not None:
        entries = manifest(directory)
        for e in entries:
            seed = "-" if e["seed"] is None else str(e["seed"])
            count = f" x{e['count']}" if e["count"] else ""
            wrap("body", f"{e['image']}{count}  seed {seed}  {e['license']}")
        if not entries:
            wrap("note", f"(no sidecars found in {directory})")
    return rows


#: Ink per row kind, in sRGB, for the screen.
_SCREEN_INK = {
    "title": (0.97, 0.95, 0.90),
    "head": (0.86, 0.74, 0.44),
    "credit": (0.93, 0.91, 0.86),
    "body": (0.74, 0.77, 0.81),
    "note": (0.52, 0.56, 0.63),
}


def screen(directory=None, cols: int = 48, scale: int = 3):
    """The credits as an image whose size is whatever the words need.

    A credits list is a *scroll*, not a page.  Paging it means the one line the
    licence actually demands can be on page four, and the thing a game wants to
    draw is one texture it can move; so the size follows the text, and the caller
    crops or scrolls.

    The line *length* is what has to be fixed instead, and it is a constant number
    of characters rather than a number of pixels: this is drawn at ``1/scale`` with
    Pillow's built-in bitmap font and scaled up with nearest-neighbour, and a
    bitmap font has one size per integer scale.  Ask for a pixel width instead and
    the scale stops being a whole number, the wrap moves with the output size, and
    a credit line that fitted at one resolution is clipped at another.

    A game with its own font should draw :func:`screen_lines` itself, which is why
    the rows come out of a function that does not touch pixels.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    cw = max(float(font.getlength("M")), 1.0)
    lh = float(font.getbbox("Ag")[3]) + 6.0
    margin = 8.0
    rows = screen_lines(directory, cols=cols)

    w = int(2.0 * margin + float(cols) * cw)
    h = int(2.0 * margin + lh * len(rows))
    # A soft vertical gradient, so it is a screen and not a stark block: the top
    # is a little lighter, which is what makes a flat panel read as lit.
    ramp = np.linspace(0.100, 0.055, h)[:, None]
    bg = ramp * np.array([0.92, 0.98, 1.25])[None, :]      # (h, 3)
    bg = np.repeat(bg[:, None, :], w, axis=1)              # (h, w, 3)
    img = Image.fromarray((np.clip(bg, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    y = margin
    for kind, text in rows:
        if kind == "head":
            y += lh * 0.40  # a heading needs air above it or it is just a line
        draw.text((margin, y), text,
                  fill=tuple(int(round(255.0 * v)) for v in _SCREEN_INK[kind]),
                  font=font)
        y += lh
        if kind == "title":
            y += lh * 0.15
            draw.line([(margin, y), (w - margin, y)], fill=(70, 78, 92))
            y += lh * 0.85
        elif kind == "head":
            y += lh * 0.25
    return img.resize((w * scale, h * scale), Image.NEAREST)


def write_screen(directory, path=None, **kw) -> Path:
    """Write the credits screen next to the text.  Returns the file written."""
    directory = Path(directory)
    path = Path(path) if path is not None else directory / "credits.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    screen(directory, **kw).save(path, "PNG", optimize=True)
    return path
