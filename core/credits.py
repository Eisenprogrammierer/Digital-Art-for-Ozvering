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

from .canvas import Canvas

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
    if path.is_dir():
        path = path / TEXT_NAME
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

#: How a row kind is set: the face, an optional weight of its own, the row's em as a
#: multiple of the body em, and the air above it.  Every row is drawn at its own size,
#: so there is no scale anywhere in this file -- a title is a bigger drawing, not a
#: bigger picture of a small one.
_SCREEN_SET = {
    "title": ("display", None, 2.05, 0.00),
    "head": ("text", 0.112, 1.06, 0.62),
    "credit": ("text", None, 1.00, 0.00),
    "body": ("sans", 0.086, 0.96, 0.00),
    "note": ("italic", None, 0.93, 0.00),
}

#: How much of a row's em its line box claims above and below the baseline.  A line
#: box, not a measurement: a row with no descender in it still holds the row below it
#: down, which is what keeps baselines on a rhythm instead of on a slope.
_SCREEN_ASCENT = 0.80
_SCREEN_DESCENT = 0.34

#: The paper around the text and the air of a blank row, in body ems.
_SCREEN_MARGIN = 1.15
_SCREEN_AIR = 0.62

#: The rule under the title: its ink in sRGB, how far below the title's baseline it
#: sits (in ems of the title's own em) and how thick it is (in body ems).
_SCREEN_RULE = (0.34, 0.38, 0.45)
_RULE_DROP = 0.30
_RULE_WIDTH = 0.07

#: The padding a row is drawn with, in ems of its own em: the gap the pen leaves
#: between a line's ink and the edge of the little canvas the line is set on.  It is a
#: rounding guard for the blit that follows, not a design number -- the margin is the
#: design number.
_SCREEN_PAD = 0.02


def _screen_spec(kind: str) -> dict:
    """The pen spec a row of ``kind`` is set in."""
    from gen.text import pen

    face, weight, _em, _air = _SCREEN_SET[kind]
    spec = dict(pen.get(face))
    if weight is not None:
        spec["weight"] = weight
    return spec


def screen(directory=None, cols: int = 48, size: float = 24.0):
    """The credits as an image whose size is whatever the words need.

    A credits list is a *scroll*, not a page.  Paging it means the one line the
    licence actually demands can be on page four, and the thing a game wants to
    draw is one texture it can move; so the size follows the text, and the caller
    crops or scrolls.

    The drawing is this project's own face, at ``size`` pixels to the em: every row
    is set by the same pen the specimen uses, in the face its kind calls for --
    display for the title, the book face for the credit line, the sans for the asset
    list and the italic for the notes -- and each row is *drawn* at its own size
    rather than one drawing being enlarged.  A bigger screen is a bigger drawing, not
    a blurrier one, which is the same promise the rest of the toolkit makes.

    The line *length* is still a number of characters rather than pixels, and the
    wrapping lives in :func:`screen_lines`, so the wrap cannot move when ``size``
    does.  The width is then the widest **measured** row plus the margin, which is
    what "sized by its text" has to mean for a proportional face -- a character
    count times a cell would be either a lie or a clipping.

    A game that would rather lay the credits out itself should draw
    :func:`screen_lines`, which is why the rows come out of a function that does not
    touch pixels.
    """
    from PIL import Image

    from gen.text import typeset
    from . import palette

    rows = screen_lines(directory, cols=cols)
    em = float(size)
    specs = [_screen_spec(kind) for kind, _text in rows]

    # the lines, measured: the sheet is as wide as its widest row and no wider
    widest = max((typeset.measure(text, spec) * _SCREEN_SET[kind][2]
                  for (kind, text), spec in zip(rows, specs)), default=0.0)
    width_em = 2.0 * _SCREEN_MARGIN + widest

    # the baselines, from the top of the sheet, in body ems: a line box each, with
    # the air its kind asks for above it
    baselines, room = [], _SCREEN_MARGIN
    for kind, text in rows:
        _face, _weight, k, air = _SCREEN_SET[kind]
        room += air
        baseline = room + _SCREEN_ASCENT * k
        baselines.append(baseline)
        room = baseline + (_SCREEN_DESCENT * k if text else _SCREEN_AIR)
    height_em = room + _SCREEN_MARGIN
    title_k = _SCREEN_SET["title"][2]

    w = max(1, int(round(width_em * em)))
    h = max(1, int(round(height_em * em)))
    sheet = Canvas(w, h, scale=em, center=(0.0, 0.0))

    for (kind, text), spec, top in zip(rows, specs, baselines):
        if not text:
            continue
        k = _SCREEN_SET[kind][2]
        line = typeset.draw(text, spec, em=em * k, pad=_SCREEN_PAD,
                            colour=_SCREEN_INK[kind])
        # where the line's own canvas put its ink and its baseline, in its own pixels
        ink_top = typeset.ink_bounds(text, spec)[3]
        x = _SCREEN_MARGIN * em - _SCREEN_PAD * em * k
        y = top * em - (_SCREEN_PAD + ink_top) * em * k
        sheet.blit(line.rgb, line.a, int(round(x)), int(round(y)))
        if kind == "title":
            # the rule is part of the layout, so it is drawn where the layout put it
            row = int(round((top + _RULE_DROP * title_k) * em))
            row = max(0, min(h - 1, row))
            thick = max(1, min(int(round(_RULE_WIDTH * em)), h - row))
            x0 = int(round(_SCREEN_MARGIN * em))
            a = np.zeros((thick, w), dtype=np.float64)
            a[:, x0:w - x0] = 1.0
            sheet.over(palette.srgb_to_linear(_SCREEN_RULE), a, 0, row)

    # A soft vertical gradient, so it is a screen and not a stark block: the top is
    # a little lighter, which is what makes a flat panel read as lit.  It is built in
    # linear light like everything else and composited under the ink in one pass.
    ramp = np.linspace(0.108, 0.052, h)[:, None]
    bg = palette.srgb_to_linear(np.clip(ramp * np.array([0.92, 0.98, 1.25])[None, :],
                                        0.0, 1.0))
    out = sheet.rgb * sheet.a[..., None] + bg[:, None, :] * (1.0 - sheet.a[..., None])
    srgb = palette.linear_to_srgb(out)
    return Image.fromarray(np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8), "RGB")


def write_screen(directory, path=None, **kw) -> Path:
    """Write the credits screen next to the text.  Returns the file written."""
    directory = Path(directory)
    path = Path(path) if path is not None else directory / SCREEN_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    screen(directory, **kw).save(path, "PNG", optimize=True)
    return path
