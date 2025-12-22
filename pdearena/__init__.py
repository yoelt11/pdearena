"""PDEArena package.

This file makes `pdearena` a regular Python package (not a namespace package),
which enables `pip install -e .` to work reliably and allows running scripts
like `python scripts/train.py` without manually setting `PYTHONPATH`.
"""

from .version import __version__

__all__ = ["__version__"]


