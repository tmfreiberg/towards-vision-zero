"""Report on values the data contains but the documentation does not.

Two anomalies surfaced when the real dataset was first checked against the
codebook: a ``LIGHT`` value of ``0``, which the published documentation does not
list, and a single record with no ``WKDY_WKND``. Neither can be handled sensibly
without knowing how many records are involved and how they are distributed
across years — a code confined to particular years means the variable's coding
changed at some point, which is a different problem from a code scattered
throughout.

Run from the repository root after ``tvz data prepare``::

    uv run python tools/diagnose_anomalies.py
"""

from __future__ import annotations

import pandas as pd

from vision_zero.config import get_settings
from vision_zero.data.codebook import load_codebook
from vision_zero.data.prepare import PARQUET_NAME
from vision_zero.data.schema import columns

#: A variable can be null on well over a million records. Printing them all is
#: not a diagnostic.
SAMPLE_LIMIT = 10


def _rule(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> int:
    """Print the anomaly report. Returns a process exit code."""
    path = get_settings().processed_data_dir / PARQUET_NAME
    if not path.is_file():
        print(f"No compiled dataset at {path}. Run `tvz data prepare` first.")
        return 1

    frame = pd.read_parquet(path)
    book = load_codebook()
    print(f"{len(frame):,} records, {len(frame.columns)} columns")

    _rule("Undocumented values, by variable and year")
    for name in columns():
        documented = set(book[name].values)
        if not documented:
            continue
        observed = frame[name].dropna()
        surplus = {str(v) for v in observed.unique()} - documented
        if not surplus:
            continue

        for value in sorted(surplus):
            mask = observed.astype("string") == value
            rows = frame.loc[observed.index[mask]]
            total = len(rows)
            share = 100 * total / len(frame)
            print(f"\n{name} = {value!r}: {total:,} records ({share:.4f}%)")

            by_year = rows.groupby("YEAR", observed=True).size()
            all_years = frame.groupby("YEAR", observed=True).size()
            print("  year    count      share of that year")
            for year in sorted(all_years.index):
                count = int(by_year.get(year, 0))
                pct = 100 * count / int(all_years[year])
                flag = "  <--" if count else ""
                print(f"  {year}  {count:>9,}  {pct:>10.3f}%{flag}")

            _rule(f"Severity distribution where {name} = {value!r}")
            print((100 * rows["TNRY_SEV"].value_counts(normalize=True)).round(2))
            print("\n  ... compared with the whole dataset:")
            print((100 * frame["TNRY_SEV"].value_counts(normalize=True)).round(2))

    _rule("Nulls in variables the documentation says are never unspecified")
    for name in columns():
        if book[name].is_nullable:
            continue
        nulls = int(frame[name].isna().sum())
        if not nulls:
            continue
        share = 100 * nulls / len(frame)
        print(f"\n{name}: {nulls:,} null record(s) ({share:.3f}%)")
        offending = frame[frame[name].isna()]
        if nulls > SAMPLE_LIMIT:
            print(f"  showing the first {SAMPLE_LIMIT} of {nulls:,}:")
            offending = offending.head(SAMPLE_LIMIT)
        print(offending[["YEAR", "ID", "MONTH", "SEVERITY"]].to_string(index=False))

    _rule("Null rate by variable, for reference")
    rates = (100 * frame[list(columns())].isna().mean()).round(3).sort_values(ascending=False)
    print(rates[rates > 0].to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
