"""A local browser workbench for the generators.

``python cli.py studio`` starts :mod:`studio.server`; the page is a form over
``cli.py``, the files it writes are real assets under ``out/studio``, and every render
comes with the command that reproduces it.  The package is deliberately small: the
catalog knows what can be made, the server runs it, and the static page knows how to
ask.  There is no second generator here.
"""
from . import catalog
from .server import DEFAULT_ROOT, make_server, render, serve

__all__ = ["catalog", "DEFAULT_ROOT", "make_server", "render", "serve"]
