# -*- coding: utf-8 -*-
from __future__ import division
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .Matcher import Matcher
from .datasets import load_demo
from . import utils
from . import modeling
from . import matching
from importlib import import_module

try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    try:
        __version__ = _pkg_version("pysmatch")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "Matcher",
    "load_demo",
    "matching",
    "modeling",
    "utils",
    "visualization",
    "__version__",
]


def __getattr__(name: str):
    """Load optional plotting helpers only when they are requested."""

    if name == "visualization":
        module = import_module(".visualization", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
