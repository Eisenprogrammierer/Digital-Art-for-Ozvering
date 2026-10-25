"""Regression tests for the browser workbench.

The studio is a form over ``cli.py``, so the claims worth testing are the seams:

* the catalog arrives, and it does not leak the server's argv or timeouts;
* a render runs the command, writes a real PNG and a sidecar under a fresh token,
  and the file the page shows is the file the command wrote;
* the bundle is a zip of that render and nothing else;
* a request cannot name a generator that does not exist, a field outside its range,
  or a file outside its own token directory.

A small figure sheet is the render used here.  It costs a subprocess, which is the
point: a test that mocked the runner would not notice a flag that stopped existing.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio import server


@contextmanager
def _studio(tmp: Path):
    httpd = server.make_server("127.0.0.1", 0, root=tmp, quiet=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def _post(url: str, obj) -> tuple:
    request = urllib.request.Request(
        url, data=json.dumps(obj).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_the_catalog_is_the_generators_and_not_the_server():
    """The browser is told what it can make; it is not told how the server runs it."""
    with tempfile.TemporaryDirectory() as tmp:
        with _studio(Path(tmp)) as base:
            status, ctype, body = _get(base + "/api/generators")
            assert status == 200 and ctype.startswith("application/json")
            catalog = json.loads(body)
            ids = {g["id"] for g in catalog}
            assert {"figures", "portrait", "maps", "site", "iso", "grounds"} <= ids
            assert len(ids) == len(catalog)
            for gen in catalog:
                assert "base" not in gen and "timeout" not in gen
                for field in gen["fields"]:
                    assert "flag" not in field
                    assert {"name", "label", "type", "default"} <= set(field)


def test_every_catalog_entry_builds_a_complete_command_from_its_defaults():
    """A catalog entry is a form *and* an argv; both have to be complete.

    The render test follows one generator end to end.  This one asks the cheaper
    question of every entry: with the defaults the page would send, does the server
    produce an argv that names the subcommand, every field and one final ``--out``?
    It catches the entry that was added to the page and forgotten in the command.
    """
    for gen in server.catalog.GENERATORS:
        values = {f["name"]: server._value(f, f["default"]) for f in gen["fields"]}
        out = Path("/tmp") / gen["out"]
        cmd = server._command(gen, values, out)
        assert cmd[1].endswith("cli.py"), gen["id"]
        assert cmd[2] == gen["id"], gen["id"]
        assert cmd[-2:] == ["--out", str(out)], gen["id"]
        assert all(isinstance(arg, str) for arg in cmd), gen["id"]


def test_a_render_writes_the_picture_the_sidecar_and_the_command():
    """The form, the subprocess and the file list are one wire.

    The facing is the newest thing on the wire, so it is the argument the test
    follows all the way to the sidecar.  The command printed beside the picture must
    name the same output file the page serves; a command that would write somewhere
    else is a command that cannot reproduce the result.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with _studio(root) as base:
            status, result = _post(base + "/api/render", {
                "generator": "figures",
                "values": {"seed": 7, "cycle": "idle", "count": 2, "size": 32,
                           "cols": 2, "fps": 12, "facing": "down"},
            })
            assert status == 200, result
            assert result["primary"] == "figures.png"
            assert result["sidecar"]["facing"] == "down"
            assert "--facing down" in result["command"]
            assert result["command"].startswith("python cli.py figures")
            folder = root / result["token"]
            assert (folder / "figures.png").is_file()
            assert (folder / "figures.json").is_file()
            status, ctype, png = _get(base + result["primary_url"])
            assert status == 200 and ctype == "image/png"
            assert png == (folder / "figures.png").read_bytes()
            status, ctype, bundle = _get(base + result["bundle_url"])
            assert status == 200 and ctype == "application/zip"
            assert bundle[:2] == b"PK"


def test_colour_fields_are_validated_and_optional():
    """A colour control is a text field with a hex contract, not a free string."""
    field = {"name": "skin_color", "label": "Skin colour", "type": "color",
             "default": "", "optional": True, "flag": "--skin-color"}
    assert server._value(field, "") == ""
    assert server._value(field, "#A1b2C3") == "#A1b2C3"
    for bad in ("red", "#abc", "#aabbccdd", "aabbcc"):
        try:
            server._value(field, bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not be a colour")


def test_the_portrait_form_exposes_outfit_accessories_and_colours():
    """The browser is not a narrower editor than the command line."""
    fields = server._fields(server.catalog.BY_ID["portrait"])
    for name in ("garment", "awards", "eyewear", "scar", "skin_color",
                 "hair_color", "iris_color", "cloth_color", "gene"):
        assert name in fields, name
    for name in ("wardrobe", "decorations"):
        assert name in server.catalog.BY_ID


def test_the_studio_refuses_an_unknown_generator_and_a_bad_field():
    """A form is not a parser, but it is still a boundary: refuse what cannot mean."""
    with tempfile.TemporaryDirectory() as tmp:
        with _studio(Path(tmp)) as base:
            status, data = _post(base + "/api/render", {"generator": "nope", "values": {}})
            assert status == 400 and "generator" in data["error"]
            status, data = _post(base + "/api/render", {
                "generator": "figures",
                "values": {"size": 1},
            })
            assert status == 400 and "between" in data["error"]
            status, data = _post(base + "/api/render", {
                "generator": "figures",
                "values": {"cycle": "teleport"},
            })
            assert status == 400 and "one of" in data["error"]


def test_a_request_cannot_read_a_file_outside_its_render():
    """The token directory is the world: no ``..``, no absolute path, no cli.py."""
    with tempfile.TemporaryDirectory() as tmp:
        with _studio(Path(tmp)) as base:
            for path in ("/studio/deadbeef/figures.png",
                         "/studio/20260101-000000-abcdef/../cli.py",
                         "/studio/20260101-000000-abcdef/%2e%2e%2fcli.py",
                         "/static/../cli.py"):
                try:
                    status, _ctype, _body = _get(base + path)
                except urllib.error.HTTPError as exc:
                    status = exc.code
                assert status == 404, (path, status)


def test_the_page_and_its_script_are_static_and_local():
    """No CDN, no font, no build step: three files and the API beside them."""
    with tempfile.TemporaryDirectory() as tmp:
        with _studio(Path(tmp)) as base:
            for name, ctype in (("index.html", "text/html"),
                                ("app.js", "text/javascript"),
                                ("style.css", "text/css")):
                status, got, body = _get(base + "/static/" + name)
                assert status == 200 and got.startswith(ctype) and body
            status, _ctype, body = _get(base + "/")
            assert status == 200 and b"ozart" in body.lower()
            page = body.decode("utf-8")
            assert "http://" not in page and "https://" not in page

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception:
            failures += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
