"""A build is a directory of pictures and sidecars; the gallery is that directory as a page.

The argument for a page over a window is not that a window is hard.  It is that a
generator set's interface is *the build*: forty sheets, each with a sidecar that
already says which function drew it, from which seed, under which licence.  Anything
that displays them has to read those sidecars or it will drift from them, and the
moment it reads them there is no reason for it to be a program with a state machine
and an event loop -- it is a rendering of files that exist.

So the page is deliberately dumb.  No JavaScript, no web font, one stylesheet inline,
one card per sidecar, and the pictures linked beside it (or inlined with ``embed`` so
the page travels alone).  What it *adds* to the pictures is the thing a person
browsing a build actually lacks:

* which module drew this, and from what seed -- so a picture can be argued with;
* the sidecar's numbers as a table, because "how many frames" and "how wide is this
  atlas" are the questions a sheet raises and does not answer;
* that the file is the file: its sha256, which is what ``cli.py demo`` prints and what
  makes "byte for byte" checkable by a person and not only by CI;
* whether the licensing gate would ship this build, in the same red as the gate's
  refusal -- because a gallery that shows a build the gate rejects and says nothing is
  a gallery that lies about the one thing this project cannot get wrong.

The manifest is :func:`core.credits.manifest`, which walks sidecars instead of
consulting a registry -- so a new generator appears on the page the day it starts
writing one, and nothing here has to be remembered.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
from pathlib import Path

from core import credits as creditslib

__all__ = ["PAGE_NAME", "CSS", "read", "build", "write", "stats"]

PAGE_NAME = "index.html"

MIME = {".png": "image/png", ".svg": "image/svg+xml"}

CSS = """
:root { --ink: #e7e3db; --dim: #9a958c; --bg: #16181c; --card: #1e2126;
        --rule: #2c3036; --bad: #e2725b; --good: #7fae7a; }
* { box-sizing: border-box; }
body { margin: 0; padding: 2.4rem 2rem 4rem; background: var(--bg); color: var(--ink);
       font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
h1 { margin: 0 0 .3rem; font-size: 1.5rem; font-weight: 600; letter-spacing: .01em; }
h2 { margin: 2.4rem 0 .8rem; font-size: .82rem; font-weight: 600; letter-spacing: .14em;
     text-transform: uppercase; color: var(--dim); }
code, .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
              font-size: .84em; }
.sub { margin: 0 0 .2rem; color: var(--dim); }
.cmd { margin: .9rem 0 0; padding: .55rem .7rem; background: #101216; border-radius: 6px;
       border: 1px solid var(--rule); display: inline-block; }
.grid { display: grid; gap: 1.1rem; align-items: start;
        grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); }
.card { background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
        overflow: hidden; display: flex; flex-direction: column; }
.card .shot { background: #e8e5df; display: flex; align-items: center; justify-content: center;
              height: 210px; padding: .5rem; border-bottom: 1px solid var(--rule); }
.card img { width: 100%; height: 100%; object-fit: contain; display: block; }
.card .body { padding: .75rem .85rem .9rem; border-top: 1px solid var(--rule); }
.card h3 { margin: 0 0 .15rem; font-size: .98rem; font-weight: 600; }
.card .src { margin: 0 0 .55rem; color: var(--dim); font-size: .78rem; word-break: break-all; }
table { border-collapse: collapse; width: 100%; font-size: .78rem; }
td { padding: .1rem 0; vertical-align: top; }
td.k { color: var(--dim); padding-right: .7rem; white-space: nowrap; }
td.v { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; word-break: break-all; }
.meta { margin-top: .5rem; color: var(--dim); font-size: .74rem; display: flex;
        justify-content: space-between; gap: .6rem; }
.bad { border: 1px solid var(--bad); border-radius: 10px; padding: .8rem 1rem;
       background: #24191a; }
.bad h3 { margin: 0 0 .35rem; color: var(--bad); font-size: .95rem; }
.bad ul { margin: .2rem 0 0; padding-left: 1.1rem; }
.good { color: var(--good); }
pre { background: #101216; border: 1px solid var(--rule); border-radius: 8px; padding: .9rem 1rem;
      overflow-x: auto; font-size: .78rem; line-height: 1.5; }
.empty { color: var(--dim); font-style: italic; }
.toc { margin: 1.1rem 0 0; font-size: .82rem; line-height: 2; }
.toc a { color: var(--ink); text-decoration: none; border-bottom: 1px dotted var(--rule); }
.toc a:hover { border-bottom-color: var(--ink); }
"""


def _e(value) -> str:
    """Everything from a sidecar is text until it is escaped: a caption is not markup."""
    return html.escape(str(value), quote=True)


def _title(stem: str) -> str:
    """"icons-atlas" is an "icons atlas": a filename is not a title, but it is close."""
    return stem.replace("-", " ").replace("_", " ").strip()


def _short(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str):
        return _short(value)
    return str(value)


def _numbers(meta: dict) -> list:
    """The sidecar as a table of *sizes*, not contents.

    A list becomes its length and a dict its key count, because a gallery is not a
    viewer for JSON: a sidecar with four hundred frame rectangles would otherwise
    put four hundred lines into a card that exists to be skimmed.  The three fields
    the caption already carries -- source, seed, licence -- are left out of the table
    so that a number cannot disagree with the line above it.
    """
    rows = []
    for key in sorted(meta):
        if key in ("source", "seed", "license"):
            continue
        value = meta[key]
        if isinstance(value, list):
            rows.append((key, f"{len(value)} item{'s' if len(value) != 1 else ''}"))
        elif isinstance(value, dict):
            rows.append((key, f"{len(value)} key{'s' if len(value) != 1 else ''}"))
        elif isinstance(value, (bool, int, float, str)):
            rows.append((key, _scalar(value)))
    return rows


def read(directory) -> list:
    """The build as records: one per asset, with its picture, sidecar and digest.

    Sorted by image name, because the page is compared byte for byte in the tests and
    a directory listing is not a promise about order.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a build directory: {directory}")
    out = []
    for entry in creditslib.manifest(directory):
        picture = directory / entry["image"]
        sidecar = directory / entry["sidecar"]
        if not picture.is_file():
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append(
            {
                "image": entry["image"],
                "stem": picture.stem,
                "path": picture,
                "sidecar": entry["sidecar"],
                "meta": meta,
                "seed": entry["seed"],
                "license": entry["license"],
                "count": entry["count"],
                "sha": hashlib.sha256(picture.read_bytes()).hexdigest()[:16],
            }
        )
    out.sort(key=lambda r: r["image"])
    return out


def _data(directory: Path) -> list:
    """The build's *data* files: the ones that record a format instead of a source.

    A campaign file is not an asset with a picture, and leaving it off the page would
    make the gallery a catalogue of illustrations rather than of the build -- so it
    gets a line with its format and its size instead of a card.
    """
    rows = []
    for path in sorted(directory.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict) and "format" in meta and "source" not in meta:
            rows.append((path.name, str(meta["format"]), path.stat().st_size))
    return rows


def _command(seeds) -> str:
    """The one command that rebuilds the build -- with the seed most of it shares.

    The directory is the *canonical* one (``out``) and not the one the page happens to
    sit in, because a page is a function of the assets: building the same assets into
    two directories has to produce the same page, or the reproducibility claim stops
    at the HTML.  Where a build lives is the caller's business; what rebuilt it is not.

    A sidecar names a *function*, not a command line, so a per-card command would be
    invented.  What is true is that the demo rebuilds the whole set from a seed, and
    that seed is the one to type; a build with several seeds says so under the line
    instead of pretending to one.
    """
    seeds = [s for s in seeds if s is not None]
    seed = max(set(seeds), key=seeds.count) if seeds else 0
    return f"python cli.py demo --out out --seed {seed}"


def _card(record: dict, embed: bool) -> str:
    path: Path = record["path"]
    meta: dict = record["meta"]
    if embed:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = MIME.get(path.suffix.lower(), "application/octet-stream")
        src = f"data:{mime};base64,{payload}"
    else:
        src = _e(path.name)
    rows = "\n".join(
        f'      <tr><td class="k">{_e(k)}</td><td class="v">{_e(v)}</td></tr>'
        for k, v in _numbers(meta)
    )
    seed = "-" if record["seed"] is None else record["seed"]
    count = f" &times; {record['count']}" if record["count"] else ""
    return f"""  <figure class="card" id="asset-{_e(record['stem'])}">
    <div class="shot"><img src="{src}" alt="{_e(_title(record['stem']))}"></div>
    <div class="body">
      <h3>{_e(_title(record['stem']))}</h3>
      <p class="src mono">{_e(meta.get('source', ''))}</p>
      <table>
{rows}
      </table>
      <div class="meta"><span>seed {_e(seed)}{count}</span><span>{_e(record['license'])}</span>
        <span class="mono">{record['sha']}</span></div>
    </div>
  </figure>"""


def build(directory, *, title: str = "ozart build", embed: bool = False) -> str:
    """The whole page as one string.  Pure: the same build is the same bytes.

    The gate's verdict is part of the page and not a log line: an unrecorded picture
    and a sidecar with no licence are what a build looks like just before a licence
    violation ships, so they are shown in the place a person is looking.

    Nothing in the output names the directory the page was built from -- see
    :func:`_command` -- so the same assets in two directories are the same page.
    """
    directory = Path(directory)
    records = read(directory)
    problems = creditslib.unrecorded(directory)
    data = _data(directory)
    seeds = [r["seed"] for r in records]
    command = _command(seeds)
    distinct = sorted({s for s in seeds if s is not None})
    credit_text = creditslib.credits(directory)

    intro = (
        f'<p class="sub">{len(records)} asset{"s" if len(records) != 1 else ""} '
        'in this build. Every one is a function of a seed: the sidecar beside it is '
        'what rebuilds it.</p>'
    )
    if distinct:
        intro += f'<p class="cmd mono">{_e(command)}</p>'
        if len(distinct) > 1:
            intro += ('<p class="sub">'
                      + _e("this build holds several seeds (" + ", ".join(map(str, distinct))
                           + "); each card names its own")
                      + '</p>')

    if problems:
        gate = ('<div class="bad"><h3>the licensing gate would refuse this build</h3>'
                '<p class="sub">An asset whose rights nobody wrote down is how a '
                'third party\'s work leaks into a build.</p><ul>'
                + "".join(f"<li>{_e(p)}</li>" for p in problems)
                + "</ul></div>")
    else:
        gate = ('<p class="sub"><span class="good">&#10003;</span> '
                'the licensing gate would ship this build: every picture has a sidecar, '
                'and every sidecar names a source and a licence.</p>')

    toc = ""
    if records:

        toc = '<p class="toc">' + " \u00b7 ".join(
            f'<a href="#asset-{_e(r["stem"])}">{_e(_title(r["stem"]))}</a>' for r in records
        ) + "</p>"

    cards = "\n".join(_card(r, embed) for r in records)
    assets = (
        f'<section>\n<h2>assets</h2>\n{toc}\n<div class="grid">\n{cards}\n</div>\n</section>'
        if records else
        '<section><h2>assets</h2><p class="empty">nothing with a sidecar yet -- '
        'run <span class="mono">python cli.py demo --out out</span>'
        '</p></section>'
    )

    data_block = ""
    if data:
        lines = "\n".join(
            f'      <tr><td class="k mono">{_e(name)}</td><td class="v">{_e(fmt)}</td>'
            f'<td class="v">{size} B</td></tr>'
            for name, fmt, size in data
        )
        data_block = (f'<section>\n<h2>data</h2>\n<table>\n{lines}\n</table>\n</section>')

    credits_block = (
        f'<section>\n<h2>credits</h2>\n<pre>{_e(credit_text)}</pre>\n</section>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{_e(title)}</h1>
{intro}
{gate}
{assets}
{data_block}
{credits_block}
</body>
</html>
"""


def write(directory, path=None, **kw) -> Path:
    """Write the page into the build.  Returns the file written."""
    directory = Path(directory)
    path = Path(path) if path is not None else directory / PAGE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(directory, **kw), encoding="utf-8")
    return path


def stats(directory) -> dict:
    """The page's size and the build's pictures, measured rather than remembered.

    The README used to carry three byte counts, and every increment with a new
    picture made them a small lie.  A number that a command can print does not
    belong in prose: this returns the linked page, the embedded page, the pictures
    both of them carry and the cards they become, so a release note can ask for it
    and a test can hold it to the same build.
    """
    directory = Path(directory)
    records = read(directory)
    pictures = sum(p.stat().st_size for p in directory.iterdir()
                   if p.is_file() and p.suffix in MIME)
    return {
        "cards": len(records),
        "pictures_bytes": int(pictures),
        "linked_bytes": len(build(directory, embed=False).encode("utf-8")),
        "embed_bytes": len(build(directory, embed=True).encode("utf-8")),
    }
