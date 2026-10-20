# The studio

A local browser workbench over `cli.py`. It is the place to make one asset at a time:
choose a generator, set a seed and its options, press Generate, and get the real PNG,
its sidecar, a zip and the command that reproduces them.

```bash
python cli.py studio --port 8000
# http://127.0.0.1:8000/
```

The user-facing walkthrough is in [`../docs/GUIDE.md`](../docs/GUIDE.md)
([по-русски](../docs/GUIDE.ru.md)). This file is for someone changing the studio.

## Design

The studio does **not** import the generator functions. It runs the command line:

```
python cli.py <generator> <flags...> --out out/studio/<token>/<name>
```

That is deliberate. The CLI is the interface CI and scripts use, and a second call
path would be a second definition of what `--facing` or `--river-fraction` means. A
field in the form is a flag in argv; the command is printed beside the result so the
browser can hand the work back to a terminal.

The server is stdlib-only (`http.server`, `subprocess`, `zipfile`) on top of the
project's existing numpy/Pillow dependency. It binds to `127.0.0.1` by default, keeps
one global render lock so two boards do not fight for cores, and serves:

| Route | What it does |
| --- | --- |
| `GET /` | the page (`static/index.html`) |
| `GET /static/*` | `index.html`, `app.js`, `style.css` — a three-name whitelist |
| `GET /api/generators` | the public catalog (no argv, timeouts or filenames) |
| `GET /api/health` | `{ok, root}`; the page uses `root` as a hint |
| `POST /api/render` | `{generator, values}` → one render and its file list |
| `GET /studio/<token>/<file>` | one file from that render's directory |
| `GET /studio/<token>/bundle.zip` | all of that render's files |

A token is `YYYYMMDD-HHMMSS-<6 hex>`, made fresh per render. Old render directories are
pruned at startup: older than seven days, or beyond the newest two hundred. The output
root defaults to `out/studio`; `--root` changes it.

## Files

| File | Responsibility |
| --- | --- |
| `catalog.py` | every generator, its group, command, output name, timeout and fields |
| `server.py` | validation, argv, subprocess, file serving, zip, cleanup |
| `static/index.html` | the page |
| `static/style.css` | the one stylesheet, no web font |
| `static/app.js` | catalog → form → `fetch` → preview/downloads |
| `../tests/test_studio.py` | catalog, render wire, refusals, traversal, static page |

## Adding a generator

1. Add or extend a `cli.py` subcommand. Give every option a long flag and a `--out`
   that accepts a file path.
2. Add an entry to `GENERATORS` in `catalog.py`:

```python
{
    "id": "myasset",              # also the CLI subcommand
    "group": "World",             # the <optgroup> on the page
    "title": "My asset",
    "blurb": "One sentence about what it is, not how it is coded.",
    "base": ["myasset"],          # fixed argv after cli.py
    "out": "myasset.png",         # primary file, relative to --out
    "slow": True,                 # optional; the page warns
    "timeout": 240,               # optional; seconds, default 120
    "fields": [
        _int("seed", "Seed", 7, 0, 2**31 - 1),
        _choice("mode", "Mode", "a", ["a", "b"]),
        _bool("no_atlas", "No atlas", "--no-atlas"),
        _text("name", "Name", "", optional=True, repeat=True),
    ],
}
```

3. Keep the field name equal to the flag, or provide a `flag` explicitly through a
   helper. `_flag` turns `river_fraction` into `--river-fraction`; booleans get their
   flag by hand because `--no-atlas` is not `--no_atlas`.
4. Use `optional=True` for a field whose empty value means "do not pass the flag".
   Use `repeat=True` for a flag that may be given more than once (names, faces); the
   form splits on commas and newlines.
5. Run `python tests/test_studio.py` and, if you changed an option, its own suite.

The public catalog deliberately drops `base`, `out`, `timeout` and each field's
`flag`; the browser has no business naming an argv. `catalog.public()` is the only
shape the page sees.

## Security notes

* Local by default. `--host 0.0.0.0` is possible and makes the workbench reachable on
  the network; there is no authentication, so do not do that on an untrusted network.
* The only subprocess is `sys.executable cli.py ...` with an argv built from the
  catalog and validated field values. There is no shell and no user-controlled
  executable.
* A request body is capped at 64 kB and must be JSON. Fields are typed, range-checked
  and choice-checked before a process starts.
* A file request is a token matching `^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$` plus a basename;
  the resolved path must stay inside that token's directory. `/static/` is a
  three-name whitelist, not a directory join.
* The page has no CDN, no web font, no cookie and no outbound request.

## Testing

```bash
python tests/test_studio.py       # 6 tests
python -m pyflakes studio
```

The render test is intentionally end-to-end: it starts the server on port 0, posts a
figure form, then reads the PNG and sidecar back over HTTP. A mocked runner would not
notice a flag that stopped existing.
