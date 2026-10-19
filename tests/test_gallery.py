"""Regression tests for the build gallery.

The gallery is a *view*, so the claims worth testing are about what a view can get
wrong:

* every asset that has a sidecar gets a card, and no asset that does not;
* the pictures are the pictures -- embedded, they decode back to the same bytes; and
  linked, the ``src`` is the file beside the page, because a page that points at a
  picture it does not have is a broken link rather than an asset;
* the numbers on a card are the sidecar's, and a list is its *length*: half a
  megabyte of frame rectangles in a card is not a gallery, it is a JSON viewer;
* anything that comes out of a file is escaped, because a caption is not markup;
* the credits travel with the page.  This is the one thing the project cannot get
  wrong, and a gallery that shows a build without its attribution is a gallery that
  has quietly moved the licence violation somewhere a person looks at it;
* a build the gate would refuse says so, in the gate's own words, at the top of the
  page it refuses -- a view that agrees with everything is not a check;
* the page is a function of the build: build it twice and the bytes match.

Run either way::

    python tests/test_gallery.py     # no pytest required
    python -m pytest tests -q
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import credits as creditslib  # noqa: E402
from core.canvas import Canvas  # noqa: E402
from export import gallery, png  # noqa: E402


# ------------------------------------------------------------------ fixtures
def _png(path: Path, r=0.4, g=0.6, b=0.3) -> Path:
    canvas = Canvas(8, 6)
    canvas.rgb[:] = (r, g, b)
    canvas.a[:] = 1.0
    png.save(canvas, path)
    return path


def _build(root: Path, n: int = 2, **over) -> Path:
    """A tiny build: ``n`` pictures, each with a sidecar, and one data file."""
    for i in range(n):
        name = f"sheet{i}"
        _png(root / f"{name}.png", 0.2 + 0.2 * i, 0.5, 0.4)
        meta = {
            "seed": 7,
            "source": f"procedural:gen.test.thing{i}",
            "license": "MIT",
            "cols": 4,
            "fps": 12.0,
            "loop": True,
            "frames": [{"index": k, "rect": [k, 0, 4, 4]} for k in range(12)],
        }
        meta.update(over)
        (root / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / "map.campaign.7.json").write_text(
        json.dumps({"format": "ozart-campaign-1", "cells": [1, 2, 3]}), encoding="utf-8")
    return root


def _page(root: Path, **kw) -> str:
    return gallery.build(root, **kw)


# ---------------------------------------------------------------- the cards
def test_every_asset_with_a_sidecar_gets_a_card():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=3)
        page = _page(root)
        assert page.count('<figure class="card"') == 3
        for i in range(3):
            assert f'id="asset-sheet{i}"' in page
            assert f"<h3>sheet{i}</h3>" in page
            assert f"procedural:gen.test.thing{i}" in page


def test_the_page_is_the_manifest_and_not_more():
    """Cards come from sidecars, so a picture without one is a *refusal*, not a card."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1)
        _png(root / "stray.png")
        page = _page(root)
        assert page.count('<figure class="card"') == 1
        assert "stray.png" in page
        assert "no sidecar" in page
        assert "would refuse this build" in page


def test_a_sidecar_with_no_licence_is_a_refusal():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1, license="")
        page = _page(root)
        assert "would refuse this build" in page
        assert "records no licence" in page


def test_a_clean_build_says_the_gate_would_ship_it():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1)
        page = _page(root)
        assert "would refuse this build" not in page
        assert "would ship this build" in page


# --------------------------------------------------------------- the pictures
def test_linked_pictures_point_at_the_files_beside_the_page():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=2)
        page = _page(root, embed=False)
        assert '<img src="sheet0.png"' in page
        assert '<img src="sheet1.png"' in page
        assert "data:image/png;base64" not in page


def test_an_embedded_picture_is_the_picture():
    """Inline it and the bytes still decode to the file: a gallery to send someone."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1)
        page = _page(root, embed=True)
        assert 'src="sheet0.png"' not in page
        start = page.index("data:image/png;base64,") + len("data:image/png;base64,")
        payload = page[start:page.index('"', start)]
        assert base64.b64decode(payload) == (root / "sheet0.png").read_bytes()
        assert "data:image/png;base64" in page


# ---------------------------------------------------------------- the numbers
def test_the_numbers_on_a_card_are_the_sidecar_s():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1)
        page = _page(root)
        assert "seed 7" in page
        assert "MIT" in page
        assert "<td class=\"v\">12</td>" in page      # fps
        assert "<td class=\"v\">true</td>" in page    # loop
        assert "<td class=\"v\">4</td>" in page       # cols


def test_a_list_is_its_length_and_not_its_contents():
    """A sidecar with four hundred frames stays one line long on the page."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1)
        meta = json.loads((root / "sheet0.json").read_text(encoding="utf-8"))
        meta["frames"] = [{"index": k} for k in range(400)]
        (root / "sheet0.json").write_text(json.dumps(meta), encoding="utf-8")
        page = _page(root)
        assert "400 items" in page
        assert '"index"' not in page
        assert page.count("<tr>") < 40


def test_the_digest_on_the_card_is_the_digest_of_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1)
        import hashlib
        sha = hashlib.sha256((root / "sheet0.png").read_bytes()).hexdigest()[:16]
        assert sha in _page(root)


def test_the_command_names_the_build_and_the_seed_most_of_it_shares():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=3)
        page = _page(root)
        assert "python cli.py demo --out" in page
        assert "--seed 7" in page
        # one asset with another seed: the page says so instead of pretending
        meta = json.loads((root / "sheet2.json").read_text(encoding="utf-8"))
        meta["seed"] = 3
        (root / "sheet2.json").write_text(json.dumps(meta), encoding="utf-8")
        page = _page(root)
        assert "several seeds" in page and "3, 7" in page


# ---------------------------------------------------------------- the safety
def test_markup_from_a_file_is_escaped():
    """A caption is not markup: a sidecar string is text even when it looks like HTML."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1, note='<script>alert("x")</script>')
        _png(root / "a&b.png")
        (root / "a&b.json").write_text(json.dumps({
            "seed": 1, "source": "procedural:gen.test", "license": "MIT",
            "title": "<b>bold</b>",
        }), encoding="utf-8")
        page = _page(root)
        assert "<script>alert" not in page
        assert "&lt;script&gt;" in page
        assert "&lt;b&gt;bold&lt;/b&gt;" in page
        assert 'src="a&amp;b.png"' in page


def test_a_missing_directory_is_refused_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "nope"
        try:
            gallery.build(missing)
        except FileNotFoundError as exc:
            assert "nope" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("a missing build directory was accepted")


# --------------------------------------------------------------- the credits
def test_the_credits_travel_with_the_page():
    """The attribution the licences demand is on the page, verbatim."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1)
        page = _page(root)
        for source in creditslib.sources():
            assert source["attribution"] in page.replace("&quot;", '"')
        assert "Kevin MacLeod" in page
        assert "CC BY 4.0" in page


# ---------------------------------------------------------------- the build
def test_an_embedded_vector_gets_the_svg_media_type():
    """``image/svg`` looks like the pattern and is not a media type.

    The seven vector panels in a build are ``.svg`` files, so this is not a corner
    case: get the type wrong and an inlined panel decodes to an empty box.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>'
        (root / "panel.svg").write_text(svg, encoding="utf-8")
        (root / "panel.json").write_text(json.dumps(
            {"seed": 1, "source": "procedural:gen.ui.vector", "license": "MIT"}),
            encoding="utf-8")
        page = _page(root, embed=True)
        assert "data:image/svg+xml;base64," in page
        assert "data:image/svg;base64," not in page
        payload = page.split("data:image/svg+xml;base64,")[1].split('"')[0]
        assert base64.b64decode(payload).decode("utf-8") == svg


def test_the_page_fetches_nothing_but_its_pictures():
    """No script, no stylesheet link, no remote anything: it opens offline, forever."""
    with tempfile.TemporaryDirectory() as tmp:
        page = _page(_build(Path(tmp), n=1))
        assert "<script" not in page
        assert "<link " not in page
        assert 'src="http' not in page


def test_the_page_is_a_function_of_the_build():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=2)
        assert _page(root) == _page(root)
        assert _page(root, embed=True) == _page(root, embed=True)
        assert _page(root) != _page(root, embed=True)


def test_the_page_does_not_name_where_the_build_lives():
    """A page is a function of the assets: two directories of the same assets agree.

    The first version put ``--out <the directory it was built in>`` into the header,
    which made every demo run differ in one line and quietly turned the ritual's
    ``diff -r`` into something with a known exception in it.
    """
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        one, two = _build(Path(a), n=2), _build(Path(b), n=2)
        page_one, page_two = _page(one), _page(two)
        assert page_one == page_two
        assert str(one) not in page_one
        assert "demo --out out" in page_one


def test_an_empty_build_is_a_page_that_says_what_to_run():
    with tempfile.TemporaryDirectory() as tmp:
        page = _page(Path(tmp))
        assert "<title>" in page
        assert "nothing with a sidecar yet" in page
        assert "python cli.py demo" in page


def test_data_files_are_a_line_and_not_a_card():
    """A campaign file is the build's data, not an illustration of it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1)
        page = _page(root)
        assert "map.campaign.7.json" in page
        assert "ozart-campaign-1" in page
        assert page.count('<figure class="card"') == 1


def test_write_reports_where_the_page_went():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build(Path(tmp), n=1)
        out = gallery.write(root)
        assert out == root / gallery.PAGE_NAME
        assert out.read_text(encoding="utf-8") == gallery.build(root)
        other = gallery.write(root, root / "deeper" / "page.html", embed=True)
        assert other.is_file() and other.parent.name == "deeper"
        assert "data:image/png;base64" in other.read_text(encoding="utf-8")


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
