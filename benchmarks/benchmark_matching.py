"""Small, dependency-light matching benchmark.

Run from the repository root with ``python benchmarks/benchmark_matching.py``.
The benchmark is intentionally separate from the test suite so timing noise does
not make CI flaky.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from pysmatch.matching import perform_exhaustive_match, perform_match


def make_data(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treated = rows // 5
    control = rows - treated
    return pd.DataFrame(
        {
            "record_id": np.arange(rows),
            "treated": np.r_[np.ones(treated, dtype=int), np.zeros(control, dtype=int)],
            "scores": rng.random(rows),
        }
    )


def time_call(label: str, function, data: pd.DataFrame) -> None:
    start = time.perf_counter()
    result = function(data)
    elapsed = time.perf_counter() - start
    print(f"{label:16s} {elapsed:8.4f}s  rows={len(result):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 2:
        parser.error("--rows must be at least 2")

    data = make_data(args.rows, args.seed)
    time_call(
        "standard",
        lambda frame: perform_match(frame, "treated", threshold=0.01, nmatches=1),
        data,
    )
    time_call(
        "exhaustive",
        lambda frame: perform_exhaustive_match(frame, "treated", threshold=0.01, nmatches=1),
        data,
    )


if __name__ == "__main__":
    main()
