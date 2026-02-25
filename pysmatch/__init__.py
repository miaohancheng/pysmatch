# -*- coding: utf-8 -*-
from __future__ import division
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .Matcher import Matcher
from . import utils
from . import modeling
from . import matching
from . import visualization

try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    try:
        __version__ = _pkg_version("pysmatch")
    except PackageNotFoundError:
        __version__ = "0.0.0"
