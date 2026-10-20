"""Regression tests for the command line: the wiring, not the pictures.

The packages have their own suites; this one exists because a subcommand is the one
place where a correct function can still be unreachable.  The ``--mips`` flag once
reached ``demo_grounds`` with the default value because the dispatch forgot to pass it,
and no picture test could see that: the function was right, the parser was right, and
the line between them was missing.  So these tests call ``main`` with the argv a user
would type, then read the sidecar back -- the two ends of the wire.

They are deliberately small assets.  A wiring test that costs a full demo is a test
that gets skipped.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli  # noqa: E402


def _run(*argv) -> int:
    """Call the CLI with its chatter captured, so a failure is the failure."""
    with contextlib.redirect_stdout(io.StringIO()):
        return cli.main([str(v) for v in argv])


def test_the_site_command_writes_a_picture_and_its_sidecar():
    """``site`` is the door to the closest camera; ``--flat`` is its one panel."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "site.png"
        assert _run("site", "--out", out, "--seed", 7, "--size", 24,
                    "--cols", 4, "--rows", 3, "--flat",
                    "--river-fraction", "0.88") == 0
        assert out.is_file() and out.stat().st_size > 0
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["source"] == "procedural:gen.maps.site"
        assert meta["seed"] == 7
        assert meta["cols"] == 4 and meta["rows"] == 3
        assert meta["panels"] == ["flat"]
        assert meta["river_fraction"] == 0.88


def test_the_iso_command_puts_the_hero_on_the_hex_it_was_given():
    """The board is new, so the wiring is new: a hex on argv has to reach the figure."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "iso.png"
        assert _run("iso", "--out", out, "--seed", 7, "--size", 28,
                    "--cols", 5, "--rows", 4, "--hero", "1,2",
                    "--levels", "5", "--facing", "left") == 0
        assert out.is_file() and out.stat().st_size > 0
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["source"] == "procedural:gen.maps.iso"
        assert meta["levels"] == 5
        assert meta["facing"] == "left"
        assert meta["hero"]["row"] == 1 and meta["hero"]["col"] == 2
        assert meta["hero"]["seed"] == 7
        assert meta["hero"]["facing"] == "left"
        for key in ("hero_size", "lattice", "foam", "span", "mix"):
            assert key in meta, key


def test_the_iso_command_refuses_a_hero_in_the_water():
    """A hex named on argv is checked against the map; the sea is not a typo.

    The error is a one-line ``SystemExit`` with the reason, not an ``IndexError``
    from the projection and not a picture of a figure standing on the lake.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "iso.png"
        try:
            _run("iso", "--out", out, "--seed", 7, "--size", 28,
                 "--cols", 5, "--rows", 4, "--hero", "1,0")
        except SystemExit as exc:
            assert "water" in str(exc)
        else:
            raise AssertionError("the board accepted a hero in the water")
        assert not out.exists()


def test_the_blazon_command_writes_a_sidecar_the_gate_can_read():
    """A coat is an asset, so its sidecar needs the same envelope as every other.

    ``blazon`` prints the text and draws the shield; without a ``source`` and a
    ``license`` the file would make the licensing gate reject a build that only
    contains a coat the generator made.  This is the one command whose sidecar was
    hand-written rather than passed through ``_sidecar``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "blazon.png"
        assert _run("blazon", "--seed", 5, "--out", out) == 0
        assert out.is_file()
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["source"] == "procedural:gen.heraldry.blazon.random_arms"
        assert meta["license"] == "MIT"
        assert meta["seed"] == 5


def test_the_studio_command_passes_its_options_through():
    """The studio is the one command that blocks; its wiring is still a wire.

    The server is tested on its own (``tests/test_studio.py``); this test replaces
    ``serve`` for the duration of the call, so it can check that the parser's four
    options reach it without opening a socket or a browser.
    """
    import studio.server as studio_server

    calls = {}
    original = studio_server.serve

    def fake_serve(**kwargs):
        calls.update(kwargs)

    studio_server.serve = fake_serve
    try:
        with tempfile.TemporaryDirectory() as tmp:
            assert _run("studio", "--host", "127.0.0.1", "--port", "8123",
                        "--root", tmp, "--no-browser") == 0
            assert calls["host"] == "127.0.0.1"
            assert calls["port"] == 8123
            assert Path(calls["root"]) == Path(tmp)
            assert calls["open_browser"] is False
    finally:
        studio_server.serve = original


def test_the_figures_command_passes_the_facing_through():
    """Facing is a new axis and the parser is the only place it can get lost."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "figures.png"
        assert _run("figures", "--out", out, "--seed", 7, "--size", 32,
                    "--count", 2, "--cols", 2, "--cycle", "idle",
                    "--facing", "down") == 0
        assert out.is_file() and out.stat().st_size > 0
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["facing"] == "down"
        assert meta["tiles"] == 2


def test_the_grounds_command_passes_the_mip_count_through():
    """The one bug this suite was written for: ``--mips`` reached the parser and stopped.

    ``demo_grounds`` has always been able to build a chain; the dispatch called it with
    the default, so the flag silently did nothing.  The sidecar's ``level`` is the
    receipt, and the head sheet must still be there -- a chain without its head is a
    different asset.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "grounds.png"
        assert _run("grounds", "--out", out, "--seed", 7, "--size", 64,
                    "--mips", 2) == 0
        assert out.is_file()
        mips = sorted(Path(tmp).glob("grounds-mip*.png"))
        assert len(mips) == 2, [p.name for p in mips]
        levels = []
        for path in mips:
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            assert meta["source"] == "procedural:gen.textures.atlas"
            levels.append(meta["level"])
        assert levels == [0, 1]


def test_the_portrait_command_passes_the_demographic_flags_through():
    """Demography is four more flags on an existing command; wiring is the risk.

    The portrait suite proves the genes reach the picture; this proves the
    *command* reaches the genes, and that the sidecar records the result a save
    file needs.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "portrait.png"
        assert _run("portrait", "--out", out, "--seed", 7, "--size", 64,
                    "--sex", "male", "--age", "elder",
                    "--climate", "arid", "--culture", "desert") == 0
        assert out.is_file() and out.stat().st_size > 0
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        for key, value in (("sex", "male"), ("age", "elder"),
                           ("climate", "arid"), ("culture", "desert")):
            assert meta["genes"][key] == value, key
            assert f"{key}={value}" in meta["dna"], key


def test_the_portrait_phenotype_flags_and_gene_overrides_reach_the_genome():
    """The phenotype adds a family flag per category and one escape hatch.

    Categories are the choices a designer makes by name, so each has a first-class
    flag; the continuous axes are too many for flags and go through ``--gene``.  Both
    paths must land in the sidecar, and the sheet command must pin them on every
    tile rather than only the first.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "phenotype.png"
        assert _run(
            "portrait", "--out", out, "--seed", 7, "--size", 64,
            "--face-shape", "heart", "--jaw-shape", "tapered",
            "--eye-shape", "hooded", "--brow-shape", "angled",
            "--nose-shape", "roman", "--mouth-shape", "full",
            "--hairline-shape", "widow", "--freckles", "bridge",
            "--gene", "skin_melanin=-0.4", "--gene", "lip_full=0.7",
        ) == 0
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        expected = {
            "face_shape": "heart", "jaw_shape": "tapered", "eye_shape": "hooded",
            "brow_shape": "angled", "nose_shape": "roman", "mouth_shape": "full",
            "hairline_shape": "widow", "freckles": "bridge",
            "skin_melanin": -0.4, "lip_full": 0.7,
        }
        for key, value in expected.items():
            assert meta["genes"][key] == value, key
            assert f"{key}={value}" in meta["dna"], key

        sheet = Path(tmp) / "sheet.png"
        assert _run("portraits", "--out", sheet, "--seed", 7, "--count", 2,
                    "--size", 64, "--cols", 2, "--face-shape", "diamond",
                    "--gene", "skin_melanin=0.6") == 0
        meta = json.loads(sheet.with_suffix(".json").read_text(encoding="utf-8"))
        for cell in meta["portraits"]:
            assert cell["genes"]["face_shape"] == "diamond"
            assert cell["genes"]["skin_melanin"] == 0.6


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
