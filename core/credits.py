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

:func:`check` is the other end of the same idea: it reads the manifest back and
says whether a build is *shippable*, so that "we remembered" is a command rather
than a habit.
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

#: What a credit line needs to be one.  A licence asks for a *correct* spelling of
#: what it names, so this is a validation list rather than a comment: CC BY needs
#: the author, the title and the licence, and a URL is what makes it checkable.
REQUIRED = ("id", "title", "author", "license", "url", "attribution")

#: The two files a build carries the attribution *in*.  The screen is not an asset
#: we made -- it is the credit itself -- so it is the one picture the gate below
#: does not ask for a sidecar from.
TEXT_NAME = "credits.txt"
SCREEN_NAME = "credits.png"

#: The kinds of picture a build can hold.  A vector the same as a raster: an ``.svg``
#: with no sidecar is a file whose provenance nobody wrote down, which is the same
#: leak a PNG with no sidecar is -- so the gate walks both, and a new back-end is
#: covered the day it starts writing.
PICTURES = (".png", ".svg")

#: Ours, and the prefix every generated sidecar's ``source`` begins with.
OURS = "procedural:"


def _picture(sidecar: Path) -> str:
    """The picture a sidecar describes, by whatever extension is actually there.

    The extension is *read* rather than assumed: a vector's sidecar must not claim a
    PNG beside it, because the manifest becomes the shipped credits, and a credit
    line naming a file the build does not contain points at nothing.  A sidecar with
    no picture at all -- a broken build rather than a normal one -- falls back to its
    own stem, which claims no extension either.
    """
    for ext in PICTURES:
        if sidecar.with_suffix(ext).exists():
            return sidecar.with_suffix(ext).name
    return sidecar.stem


def sources() -> list:
    """The curated sources, as a list of copies."""
    return [dict(s) for s in SOURCES]


def missing_fields() -> list:
    """Sources whose licence cannot be honoured as written."""
    return [dict(s) for s in SOURCES if any(not s.get(k) for k in REQUIRED)]


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
                "image": _picture(path),
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
    path = Path(path) if path is not None else directory / TEXT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credits(directory), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def unrecorded(directory) -> list:
    """Assets in a build whose rights nobody wrote down.

    A picture is ours only if the sidecar beside it says so: a seed to rebuild it
    and a ``source`` to rebuild it *from*.  A picture with no sidecar -- raster or
    vector, it makes no difference -- is an asset whose provenance is unknown, which
    the licensing notes call out as the usual way a third party's rights leak into a
    build: not by being stolen, but by nobody being able to say any more what it is.
    The other thing a build holds is *data*, which is not a picture and says so with
    a ``format`` field instead of a ``source``.
    """
    directory = Path(directory)
    problems = []
    known = {s["id"] for s in SOURCES}
    sidecars = set()
    for path in sorted(directory.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path.name}: not readable as JSON ({exc})")
            continue
        if not isinstance(meta, dict):
            problems.append(f"{path.name}: a sidecar is an object, not a {type(meta).__name__}")
            continue
        if "source" not in meta:
            if "format" not in meta:
                problems.append(
                    f"{path.name}: records neither a source (a sidecar) nor a "
                    f"format (a data file), so nothing ties it to anything")
            continue
        source = str(meta.get("source") or "")
        if not source.strip():
            problems.append(f"{path.name}: records no source")
        elif not source.startswith(OURS) and source not in known:
            problems.append(
                f"{path.name}: names the source {source!r}, which is not ours "
                f"({OURS}...) and is not one of the curated ones {sorted(known)}")
        if not str(meta.get("license") or "").strip():
            problems.append(f"{path.name}: records no licence")
        sidecars.add(path.with_suffix("").name)
    for ext in PICTURES:
        for picture in sorted(directory.glob(f"*{ext}")):
            if picture.name != SCREEN_NAME and picture.stem not in sidecars:
                problems.append(f"{picture.name}: no sidecar -- no seed, no source, no licence")
    return problems


_UNSET = object()


def check(directory=None, credits_text=_UNSET) -> list:
    """What is wrong with a build, as sentences.  An empty list means: ship it.

    A build can owe somebody something it has not recorded in three different ways,
    and they are worth keeping apart because they are fixed in different places:

    * **a source that cannot be credited as written** -- a curated entry is missing
      a field, so the credit line its licence demands cannot be assembled whatever
      the build contains.  This is a property of the code, not of the build;
    * **an asset nobody recorded** (:func:`unrecorded`) -- a picture with no
      sidecar, a sidecar with no source, or a source that is neither ours nor one of
      the curated ids.  An asset whose provenance is unknown is the leak;
    * **an attribution that is not in the credits the build ships** -- the file is
      there and the credit line is not in it.  This is the failure mode of
      *forgetting*, the one this module exists for, and it happens to honest people:
      add a second track and never regenerate the shipped block.

    ``credits_text`` is the block as *shipped*.  Left alone, it is read from
    :data:`TEXT_NAME` in ``directory`` -- and a build with a curated source and no
    credits text at all is a problem, because then the attribution is nowhere in
    it.  Pass the text (including ``""``) when the caller draws the credits itself,
    which is what a game with its own font does.

    Nothing here repairs anything: the gate reports, the way
    :func:`gen.ui.style.validate` reports.  A gate that fixes what it finds is a
    gate nobody can trust to be looking.
    """
    problems = []
    for s in missing_fields():
        missing = [k for k in REQUIRED if not s.get(k)]
        problems.append(f"source {s.get('id', '?')!r}: cannot be credited as written "
                        f"-- no {' or '.join(missing)}")
    if directory is not None:
        problems.extend(unrecorded(directory))
    if directory is not None or credits_text is not _UNSET:
        text = credits_text
        if text is _UNSET:
            path = Path(directory) / TEXT_NAME
            text = path.read_text(encoding="utf-8") if path.exists() else None
        if text is None:
            if SOURCES:
                problems.append(
                    f"{TEXT_NAME} is not in the build, so nothing shipped carries "
                    f"the attribution")
        else:
            for s in SOURCES:
                line = str(s.get("attribution") or "")
                if line and line not in text:
                    problems.append(f"{TEXT_NAME}: the credit for {s.get('id', '?')!r} "
                                    f"is not in it verbatim")
    return problems


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
    path = Path(path) if path is not None else directory / SCREEN_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    screen(directory, **kw).save(path, "PNG", optimize=True)
    return path
