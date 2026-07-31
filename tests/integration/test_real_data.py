"""Checks that run against the full dataset when it is present.

Everything else in the suite runs against a synthetic fixture, so that CI needs
neither the data nor the time to parse it. These tests are the counterweight: a
fixture faithful to the schema can still diverge from the real extract, and
nothing else would notice.

They are marked ``requires_data`` and skip when the compiled dataset is absent,
which is the normal state on a fresh clone and always the state in CI. Run them
after ``tvz data prepare``::

    uv run pytest -m requires_data

The expected figures are the ones the analysis reports. Fixing them here means a
change to loading, typing, or the codebook that shifts a headline number fails a
test instead of quietly altering a conclusion.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vision_zero.config import get_settings
from vision_zero.data.codebook import load_codebook
from vision_zero.data.prepare import PARQUET_NAME, read_manifest
from vision_zero.data.schema import UNDOCUMENTED_VALUES, columns, dtypes

pytestmark = pytest.mark.requires_data

# Records across all of Quebec, 2011-2022.
EXPECTED_RECORDS = 1_717_407

# Records after restricting to the Montréal administrative region.
EXPECTED_MONTREAL_RECORDS = 330_446

MONTREAL = "Montréal (06)"
FIRST_YEAR, LAST_YEAR = 2011, 2022

# Share of Montréal records in each ternary severity class, to one decimal place.
# The minority class is under one percent, which is the fact the whole analysis
# turns on; a change here is a change to the problem, not to the code.
EXPECTED_MONTREAL_SHARES = {0: 79.8, 1: 19.3, 2: 0.9}
SHARE_TOLERANCE = 0.2

# Variables that carry nulls despite no documented unspecified state, with the
# largest share tolerated. Enumerated rather than exempted, so that a new one —
# or an existing one growing — fails instead of passing quietly.
#
# RDWX is the documented gap: `CD_ZON_TRAVX_ROUTR` defines only the positive
# value, so every record not in a work zone is blank. Whether a blank means
# "not in a work zone" or "not recorded" cannot be settled from the
# documentation, which is why the variable carries a note in the codebook.
#
# WKDY_WKND has exactly one blank record in twelve years. That is data entry.
TOLERATED_NULL_SHARES = {
    "RDWX": 99.0,
    "WKDY_WKND": 0.001,
}

# Null rates worth holding steady, as a share of all records. These are
# properties of the published data, and a shift means the extract changed or
# the loader did. SPD_LIM is the one that matters analytically: close to a
# quarter of records have no speed limit recorded.
EXPECTED_NULL_SHARES = {
    "RDWX": 97.8,
    "SPD_LIM": 22.8,
    "RD_CONFG": 8.7,
    "ACCDN_TYPE": 7.6,
    "LONG_LOC": 7.0,
}
NULL_SHARE_TOLERANCE = 0.5


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """The compiled dataset, or a skip if it has not been built."""
    path = get_settings().processed_data_dir / PARQUET_NAME
    if not path.is_file():
        pytest.skip(f"no compiled dataset at {path}; run `tvz data prepare`")
    return pd.read_parquet(path)


def test_record_count(frame: pd.DataFrame) -> None:
    """The published extract has a known size. Losing records must be visible."""
    assert len(frame) == EXPECTED_RECORDS


def test_all_years_are_present(frame: pd.DataFrame) -> None:
    years = sorted(frame["YEAR"].dropna().unique())
    assert years == list(range(FIRST_YEAR, LAST_YEAR + 1))


def test_no_year_is_disproportionately_small(frame: pd.DataFrame) -> None:
    """A truncated input would show up as a year well below its neighbours."""
    counts = frame.groupby("YEAR", observed=True).size()
    assert counts.min() > 0.4 * counts.median()


def test_the_extract_carries_every_expected_column(frame: pd.DataFrame) -> None:
    """Catches a compiled extract left behind by a schema change."""
    from vision_zero.data.prepare import expected_columns

    assert expected_columns() <= set(frame.columns)


def test_types_survived_compilation(frame: pd.DataFrame) -> None:
    schema = dtypes()
    for name in columns():
        assert str(frame[name].dtype) == schema[name], name


def test_every_observed_value_is_documented(frame: pd.DataFrame) -> None:
    """The codebook must describe the data, not an idealisation of it."""
    book = load_codebook()
    undocumented: dict[str, list[str]] = {}

    for name in columns():
        documented = set(book[name].values)
        if not documented:
            continue
        observed = {str(v) for v in frame[name].dropna().unique()}
        surplus = sorted(observed - documented)
        if surplus:
            undocumented[name] = surplus

    assert not undocumented, f"values present in the data but not in the codebook: {undocumented}"


def test_nulls_in_non_nullable_variables_are_all_accounted_for(frame: pd.DataFrame) -> None:
    """A blank where the documentation admits none is a finding, not a default."""
    book = load_codebook()
    unexplained: dict[str, float] = {}

    for name in columns():
        if book[name].is_nullable or not frame[name].isna().any():
            continue
        share = 100 * float(frame[name].isna().mean())
        tolerated = TOLERATED_NULL_SHARES.get(name)
        if tolerated is None or share > tolerated:
            unexplained[name] = round(share, 4)

    assert not unexplained, (
        f"unaccounted nulls in variables documented as never unspecified: {unexplained}"
    )


def test_null_rates_are_stable(frame: pd.DataFrame) -> None:
    """These rates characterise the published data. Drift means something changed."""
    for name, expected in EXPECTED_NULL_SHARES.items():
        share = 100 * float(frame[name].isna().mean())
        assert abs(share - expected) < NULL_SHARE_TOLERANCE, (
            f"{name}: {share:.2f}% null, expected about {expected}%"
        )


def test_undocumented_values_were_coerced_away(frame: pd.DataFrame) -> None:
    """Loading nulls the handful of values the documentation does not admit."""
    for name, values in UNDOCUMENTED_VALUES.items():
        observed = set(frame[name].dropna().astype("string").unique())
        assert not (observed & values), f"{name} still carries {sorted(observed & values)}"


def test_montreal_subset_size(frame: pd.DataFrame) -> None:
    assert int((frame["REGION"] == MONTREAL).sum()) == EXPECTED_MONTREAL_RECORDS


def test_montreal_class_imbalance(frame: pd.DataFrame) -> None:
    """The severe class is under one percent. This is the premise of the analysis."""
    montreal = frame.loc[frame["REGION"] == MONTREAL, "TNRY_SEV"]
    shares = 100 * montreal.value_counts(normalize=True)

    for label, expected in EXPECTED_MONTREAL_SHARES.items():
        assert abs(float(shares[label]) - expected) < SHARE_TOLERANCE, (
            f"class {label}: {shares[label]:.2f}%, expected about {expected}%"
        )


def test_derived_severity_agrees_with_victim_count(frame: pd.DataFrame) -> None:
    """Nobody hurt means class 0, and someone hurt means class 1 or 2."""
    determined = frame["TNRY_SEV"].notna() & frame["NUM_VICTIMS"].notna()
    subset = frame.loc[determined]
    assert (subset.loc[subset["NUM_VICTIMS"] == 0, "TNRY_SEV"] == 0).all()
    assert (subset.loc[subset["NUM_VICTIMS"] > 0, "TNRY_SEV"] > 0).all()


def test_manifest_matches_the_compiled_file(frame: pd.DataFrame) -> None:
    manifest = read_manifest()
    assert manifest is not None
    assert manifest.rows == len(frame)
    assert sum(manifest.rows_by_year.values()) == len(frame)
    assert len(manifest.inputs) == LAST_YEAR - FIRST_YEAR + 1
