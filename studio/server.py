"""The local studio: a small HTTP server over the command line.

Why a server and not a second application?  Because the assets are already a
function of a subcommand and a seed.  The studio's job is to make the *choice* of
subcommand and seed a form, show the picture, and hand back the command that would
reproduce it.  It runs ``python cli.py ... --out <fresh directory>`` in a subprocess,
so a form field cannot mean something different from the flag beside it, and a
picture made in the browser is the same file CI makes.

The server binds to ``127.0.0.1`` by default and serves only:

* the static page in ``studio/static``;
* the catalog;
* files under the studio's own output directory, addressed by an unguessable token;
* a zip of one render's files.

There is no upload, no shell, no path from a request to an arbitrary file, and a
render is serialized by a lock so a browser cannot start two heavy boards at once.
It is a development workbench for one person on one machine, not a public service.
"""
from __future__ import annotations

import json
import mimetypes
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import catalog

#: The repository root, from this file's location.  The subprocess runs here so
#: ``cli.py`` and the packages import exactly as they do in a terminal.
ROOT = Path(__file__).resolve().parents[1]

#: Where a render writes.  A token per render keeps two pictures with the same name
#: from overwriting each other and makes a download link stable after the request.
DEFAULT_ROOT = ROOT / "out" / "studio"

#: The page, its stylesheet and its script.  A whitelist rather than ``..`` checks:
#: a static path is one of three names.
STATIC = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"index.html", "app.js", "style.css"}

#: One render at a time: the generators are CPU-bound and a browser can press the
#: button twice.  A queue is friendlier than two numpy processes fighting for cores.
RENDER_LOCK = threading.Lock()

#: Tokens are timestamped so the output directory sorts by age.  The random tail is
#: what makes the URL unguessable; the timestamp is for a human looking at ``out``.
TOKEN_RE = r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$"


def _token() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


def _clean(root: Path, keep_days: int = 7, keep: int = 200) -> None:
    """Forget old workbenches, so ``out/studio`` does not grow without a reason."""
    try:
        dirs = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return
    cutoff = datetime.now() - timedelta(days=keep_days)
    for p in dirs:
        try:
            if datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass
    dirs = [p for p in dirs if p.exists()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in dirs[keep:]:
        shutil.rmtree(p, ignore_errors=True)


def _fields(gen: dict) -> dict:
    return {f["name"]: f for f in gen["fields"]}


def _value(field: dict, raw):
    """One form value, checked against the field it came from.

    The checks are the same kinds the CLI makes: an integer is an integer, a choice
    is one of the choices, and a number is inside the range the form promised.  The
    CLI is still the last word -- this is a form, not a second parser -- but a
    request that cannot mean anything is refused before a subprocess is started.
    """
    kind = field["type"]
    if kind == "bool":
        return bool(raw)
    text = "" if raw is None else str(raw).strip()
    if field.get("optional") and text == "":
        return ""
    if kind == "int":
        try:
            value = int(text)
        except (TypeError, ValueError):
            raise ValueError(f"{field['label']} is a whole number") from None
        if not (field["min"] <= value <= field["max"]):
            raise ValueError(f"{field['label']} is between {field['min']} and {field['max']}")
        return value
    if kind == "float":
        try:
            value = float(text)
        except (TypeError, ValueError):
            raise ValueError(f"{field['label']} is a number") from None
        if not (field["min"] <= value <= field["max"]):
            raise ValueError(f"{field['label']} is between {field['min']} and {field['max']}")
        return value
    if kind == "choice":
        if text not in field["choices"]:
            raise ValueError(f"{field['label']} is one of {', '.join(field['choices'])}")
        return text
    if kind == "text":
        if len(text) > 200:
            raise ValueError(f"{field['label']} is at most 200 characters")
        if any(ord(ch) < 32 and ch not in "\n\t" for ch in text):
            raise ValueError(f"{field['label']} has a control character")
        return text
    if kind == "color":
        if text == "" and field.get("optional"):
            return ""
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", text):
            raise ValueError(f"{field['label']} is a colour like #aabbcc")
        return text
    raise ValueError(f"unknown field type: {kind}")


def _command(gen: dict, values: dict, out_path: Path) -> list:
    """The argv for one render, from the catalog and the form."""
    cmd = [sys.executable, str(ROOT / "cli.py")] + list(gen["base"])
    for field in gen["fields"]:
        value = values[field["name"]]
        if field["type"] == "bool":
            if value:
                cmd.append(field["flag"])
            continue
        if field.get("optional") and (value is None or str(value).strip() == ""):
            continue
        if field.get("repeat"):
            parts = [p.strip() for p in str(value).replace("\n", ",").split(",")
                     if p.strip()]
            for part in parts:
                cmd.extend([field["flag"], part])
        else:
            cmd.extend([field["flag"], str(value)])
    cmd.extend(["--out", str(out_path)])
    return cmd


def _shown(cmd: list, root: Path) -> str:
    """A copyable command, with the repository-relative output path."""
    try:
        cli = str(Path(cmd[1]).resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        cli = cmd[1]
    shown = ["python", cli]
    for arg in cmd[2:]:
        try:
            shown.append(str(Path(arg).resolve().relative_to(root.resolve())))
        except (ValueError, OSError):
            shown.append(arg)
    return shlex.join(shown)


def _primary(outdir: Path, wanted: str) -> Path | None:
    path = outdir / wanted
    if path.is_file():
        return path
    images = sorted((p for p in outdir.iterdir()
                     if p.is_file() and p.suffix.lower() in (".png", ".svg")),
                    key=lambda p: p.stat().st_size, reverse=True)
    return images[0] if images else None


def _sidecar(primary: Path | None, outdir: Path):
    if primary is None:
        return None
    path = primary.with_suffix(".json")
    if not path.is_file():
        found = sorted(outdir.glob("*.json"))
        path = found[0] if found else None
    if path is None or not path.is_file() or path.stat().st_size > 2_000_000:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render(gen: dict, values: dict, root: Path) -> dict:
    """Run one command into a fresh directory and describe what it wrote."""
    outdir = root / _token()
    outdir.mkdir(parents=True, exist_ok=False)
    out_path = outdir / gen["out"]
    cmd = _command(gen, values, out_path)
    started = time.time()
    try:
        with RENDER_LOCK:
            done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  timeout=int(gen.get("timeout", 120)))
    except subprocess.TimeoutExpired:
        shutil.rmtree(outdir, ignore_errors=True)
        raise ValueError(f"the {gen['title'].lower()} took too long and was stopped")
    elapsed = time.time() - started
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        shutil.rmtree(outdir, ignore_errors=True)
        raise ValueError("the generator refused: " + (" / ".join(detail[-3:])
                                                     if detail else "no message"))
    primary = _primary(outdir, gen["out"])
    if primary is None:
        shutil.rmtree(outdir, ignore_errors=True)
        raise ValueError("the generator wrote no picture")
    files = sorted((p for p in outdir.iterdir() if p.is_file()),
                   key=lambda p: (p != primary, p.name))
    token = outdir.name
    base = f"/studio/{token}"
    return {
        "token": token,
        "folder": str(outdir.relative_to(ROOT)) if outdir.is_relative_to(ROOT) else str(outdir),
        "primary": primary.name,
        "primary_url": f"{base}/{primary.name}",
        "bundle_url": f"{base}/bundle.zip",
        "files": [{"name": p.name,
                   "bytes": p.stat().st_size,
                   "kind": p.suffix.lstrip(".").lower(),
                   "url": f"{base}/{p.name}"} for p in files],
        "sidecar": _sidecar(primary, outdir),
        "command": _shown(cmd, ROOT),
        "elapsed": round(elapsed, 2),
        "title": gen["title"],
    }


class Handler(BaseHTTPRequestHandler):
    """The page, the catalog, one render endpoint and the files a render wrote."""

    server_version = "ozart-studio/1"

    # ------------------------------------------------------------- plumbing
    def log_message(self, fmt, *args):  # noqa: D401 - BaseHTTPRequestHandler API
        if not getattr(self.server, "quiet", True):
            sys.stderr.write("studio: " + (fmt % args) + "\n")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, obj) -> None:
        self._send(status, _json_bytes(obj), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _static(self, name: str) -> None:
        if name not in STATIC_FILES:
            self._error(404, "no such page")
            return
        path = STATIC / name
        ctype = {"html": "text/html; charset=utf-8",
                 "js": "text/javascript; charset=utf-8",
                 "css": "text/css; charset=utf-8"}[path.suffix.lstrip(".")]
        try:
            body = path.read_bytes()
        except OSError:
            self._error(404, "the studio page is missing")
            return
        self._send(200, body, ctype)

    def _asset(self, token: str, name: str) -> None:
        root = self.server.root
        if not re.fullmatch(TOKEN_RE, token):
            self._error(404, "no such render")
            return
        folder = (root / token).resolve()
        try:
            folder.relative_to(root.resolve())
        except ValueError:
            self._error(404, "no such render")
            return
        if name == "bundle.zip":
            self._bundle(folder)
            return
        if Path(name).name != name or name in ("", ".", ".."):
            self._error(404, "no such file")
            return
        path = (folder / name).resolve()
        try:
            path.relative_to(folder)
        except ValueError:
            self._error(404, "no such file")
            return
        if not path.is_file():
            self._error(404, "no such file")
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    def _bundle(self, folder: Path) -> None:
        if not folder.is_dir():
            self._error(404, "no such render")
            return
        path = folder / "bundle.zip"
        if not path.is_file():
            try:
                with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in sorted(folder.iterdir()):
                        if f.is_file() and f.name != "bundle.zip":
                            z.write(f, f.name)
            except OSError as exc:
                self._error(500, f"could not make the bundle: {exc}")
                return
        self._send(200, path.read_bytes(), "application/zip")

    # --------------------------------------------------------------- routes
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._static("index.html")
        elif path.startswith("/static/"):
            self._static(path[len("/static/"):])
        elif path == "/api/generators":
            self._json(200, catalog.public())
        elif path == "/api/health":
            self._json(200, {"ok": True, "root": str(self.server.root)})
        elif path.startswith("/studio/"):
            rest = path[len("/studio/"):]
            token, _, name = rest.partition("/")
            self._asset(token, name)
        else:
            self._error(404, "no such page")

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/api/render":
            self._error(404, "no such endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "bad request length")
            return
        if length <= 0 or length > 64_000:
            self._error(413, "the form is too large")
            return
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._error(400, "the request is not JSON")
            return
        name = request.get("generator")
        gen = catalog.BY_ID.get(name) if isinstance(name, str) else None
        if gen is None:
            self._error(400, "no such generator")
            return
        raw = request.get("values") or {}
        if not isinstance(raw, dict):
            self._error(400, "the values are not an object")
            return
        try:
            values = {f["name"]: _value(f, raw.get(f["name"], f["default"]))
                      for f in gen["fields"]}
        except ValueError as exc:
            self._error(400, str(exc))
            return
        try:
            result = render(gen, values, self.server.root)
        except ValueError as exc:
            self._error(500, str(exc))
            return
        self._json(200, result)


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, root: Path, quiet: bool = True):
        super().__init__(address, Handler)
        self.root = Path(root).expanduser().resolve()
        self.quiet = quiet


def make_server(host: str = "127.0.0.1", port: int = 8000,
                root=None, quiet: bool = True) -> StudioServer:
    """A server ready to ``serve_forever``; ``port=0`` asks the OS for one.

    Tests use this directly: it does not open a browser, print a banner or block.
    """
    root = Path(root) if root is not None else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    _clean(root)
    return StudioServer((host, int(port)), root=root, quiet=quiet)


def serve(host: str = "127.0.0.1", port: int = 8000, root=None,
          open_browser: bool = True) -> None:
    """Run the studio until interrupted, and say where it is."""
    httpd = make_server(host, port, root=root, quiet=False)
    actual_host, actual_port = httpd.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"
    print(f"ozart studio: {url}")
    print(f"  assets go to {httpd.root}")
    print("  Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstudio: stopped")
    finally:
        httpd.server_close()
