"""Generate the synthetic extracts the test suite runs against.

The real data is 257 MB of collision records and is not something CI should
read. The tests instead run against a handful of synthetic years that carry the
real schema, the real value vocabulary, and roughly the real class imbalance —
enough that a schema or dtype regression is caught, without shipping or parsing
the actual extracts.

The vocabulary is drawn from the codebook, so the fixture cannot drift away from
the documentation: add a value there and it becomes available here.

Records are drawn independently, which means the fixture has no signal between
features and severity beyond what chance provides. That is deliberate. Fixtures
exist to exercise plumbing, and a fixture with invented structure invites tests
that assert on that structure and then fail against real data.

Run from the repository root::

    uv run python tools/make_fixture.py --years 2011 2012 --rows 400
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vision_zero.data.codebook import load_codebook
from vision_zero.data.schema import columns, dtypes

# Approximate class shares in the real extract. Reproduced so that fixture-based
# tests meet the same degenerate-minority-class conditions the real data has.
SEVERITY_WEIGHTS: dict[str, float] = {
    "Material damage below the reporting threshold": 0.15,
    "Material damage only": 0.65,
    "Minor": 0.19,
    "Fatal or serious": 0.01,
}

# Share of values left blank, for variables the documentation says may be blank.
MISSING_RATE = 0.04


def _draw(rng: np.random.Generator, values: list[str], nullable: bool) -> Any:
    """Draw one value, occasionally blank where the documentation allows it."""
    if nullable and rng.random() < MISSING_RATE:
        return None
    return str(rng.choice(values))


def build_year(year: int, rows: int, seed: int) -> pd.DataFrame:
    """Build one synthetic year.

    Parameters
    ----------
    year
        Calendar year to stamp on the records.
    rows
        Number of records.
    seed
        Seed for this year. Distinct years get distinct seeds so that the
        fixture is not twelve copies of one table.

    Returns
    -------
    pandas.DataFrame
        Synthetic records with the documented columns, in schema order.
    """
    rng = np.random.default_rng(seed)
    book = load_codebook()

    data: dict[str, list[Any]] = {}
    for name in columns():
        variable = book[name]
        if name == "YEAR":
            data[name] = [year] * rows
            continue
        if name == "ID":
            data[name] = [f"{year} _ {index + 1}" for index in range(rows)]
            continue
        if name == "SEVERITY":
            data[name] = list(
                rng.choice(
                    list(SEVERITY_WEIGHTS),
                    size=rows,
                    p=list(SEVERITY_WEIGHTS.values()),
                )
            )
            continue

        options = list(variable.values)
        if not options:
            data[name] = [None] * rows
            continue
        data[name] = [_draw(rng, options, variable.is_nullable) for _ in range(rows)]

    frame = pd.DataFrame(data, columns=list(columns()))

    # NUM_VICTIMS must agree with SEVERITY, or the derived targets contradict
    # each other and every leakage test becomes meaningless.
    has_victims = frame["SEVERITY"].isin(["Minor", "Fatal or serious"])
    frame.loc[~has_victims, "NUM_VICTIMS"] = "0"
    frame.loc[has_victims, "NUM_VICTIMS"] = [
        str(rng.choice(["1", "2", "9"], p=[0.75, 0.2, 0.05])) for _ in range(int(has_victims.sum()))
    ]
    return frame


def main() -> int:
    """Write the synthetic extracts. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=[2011, 2012, 2013])
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--out", type=Path, default=Path("tests/fixtures/raw"), help="destination directory"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for offset, year in enumerate(args.years):
        frame = build_year(year, args.rows, args.seed + offset)
        destination = args.out / f"saaq_{year}.csv"
        frame.to_csv(destination, index=False)
        total += len(frame)

    expected = set(dtypes())
    written = set(frame.columns)
    if expected != written:
        print(f"schema mismatch: {expected ^ written}")
        return 1

    print(f"wrote {len(args.years)} year(s), {total} records, to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
