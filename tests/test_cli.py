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
                    "--cols", 4, "--rows", 3, "--flat") == 0
        assert out.is_file() and out.stat().st_size > 0
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["source"] == "procedural:gen.maps.site"
        assert meta["seed"] == 7
        assert meta["cols"] == 4 and meta["rows"] == 3
        assert meta["panels"] == ["flat"]


def test_the_iso_command_puts_the_hero_on_the_hex_it_was_given():
    """The board is new, so the wiring is new: a hex on argv has to reach the figure."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "iso.png"
        assert _run("iso", "--out", out, "--seed", 7, "--size", 28,
                    "--cols", 5, "--rows", 4, "--hero", "1,2",
                    "--levels", "5") == 0
        assert out.is_file() and out.stat().st_size > 0
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["source"] == "procedural:gen.maps.iso"
        assert meta["levels"] == 5
        assert meta["hero"]["row"] == 1 and meta["hero"]["col"] == 2
        assert meta["hero"]["seed"] == 7


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
