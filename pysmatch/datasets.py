"""Small datasets used by the documentation and quickstart examples."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd


def load_demo() -> pd.DataFrame:
    """Load the compact loan demo dataset shipped with :mod:`pysmatch`.

    The returned frame contains 500 rows with ``loan_status`` as the grouping
    variable.  It is intended for tutorials and smoke tests, not for
    production inference or benchmark claims.
    """

    resource = files("pysmatch.data").joinpath("loan.csv")
    with resource.open("r", encoding="utf-8") as handle:
        return pd.read_csv(handle)


__all__ = ["load_demo"]
