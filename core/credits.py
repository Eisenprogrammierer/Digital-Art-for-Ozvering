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
is made of, and :func:`write` drops it in the build.
"""
from __future__ import annotations

import json
from pathlib import Path

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
