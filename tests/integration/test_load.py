"""Loading tests, run against the synthetic fixture rather than the real data."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from vision_zero.data.load import (
    MissingYearsError,
    SchemaMismatchError,
    add_derived_columns,
    load_year,
    load_years,
    read_raw_csv,
)
from vision_zero.data.schema import columns

FIXTURE_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
FIXTURE_YEARS = (2011, 2012, 2013)


@pytest.fixture
def raw_dir() -> Path:
    return FIXTURE_RAW


def test_fixture_exists(raw_dir: Path) -> None:
    assert sorted(p.name for p in raw_dir.glob("*.csv")) == [
        f"saaq_{year}.csv" for year in FIXTURE_YEARS
    ]


def test_load_year_applies_documented_types(raw_dir: Path) -> None:
    frame = load_year(2011, raw_dir)
    assert str(frame["WEATHER"].dtype) == "Int64"
    assert str(frame["SEVERITY"].dtype) == "category"
    assert str(frame["ID"].dtype) == "string"


def test_ordinal_categories_are_ordered_after_loading(raw_dir: Path) -> None:
    frame = load_year(2011, raw_dir)
    assert frame["SPD_LIM"].cat.ordered
    assert frame["SPD_LIM"].cat.categories[0] == "<50"


def test_load_years_concatenates_in_year_order(raw_dir: Path) -> None:
    frame = load_years(FIXTURE_YEARS, raw_dir)
    assert list(frame["YEAR"].unique()) == list(FIXTURE_YEARS)
    assert len(frame) == 1200


def test_a_missing_year_is_an_error_by_default(raw_dir: Path) -> None:
    """Silent loss of a year would be invisible in every downstream number."""
    with pytest.raises(MissingYearsError, match="2099"):
        load_years([2011, 2099], raw_dir)


def test_a_missing_year_can_be_tolerated_explicitly(raw_dir: Path) -> None:
    frame = load_years([2011, 2099], raw_dir, strict=False)
    assert list(frame["YEAR"].unique()) == [2011]


def test_no_readable_year_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could be read"):
        load_years([2011], tmp_path, strict=False)


def test_absent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_raw_csv(tmp_path / "nope.csv")


def test_a_dropped_column_is_fatal(raw_dir: Path, tmp_path: Path) -> None:
    """A vanished column would otherwise read as null everywhere it is used."""
    source = raw_dir / "saaq_2011.csv"
    frame = pd.read_csv(source).drop(columns=["WEATHER"])
    frame.to_csv(tmp_path / "saaq_2011.csv", index=False)

    with pytest.raises(SchemaMismatchError, match="WEATHER"):
        load_year(2011, tmp_path)


def test_an_extra_column_is_tolerated(raw_dir: Path, tmp_path: Path) -> None:
    shutil.copy(raw_dir / "saaq_2011.csv", tmp_path / "saaq_2011.csv")
    frame = pd.read_csv(tmp_path / "saaq_2011.csv")
    frame["FUTURE_FIELD"] = 1
    frame.to_csv(tmp_path / "saaq_2011.csv", index=False)

    loaded = load_year(2011, tmp_path)
    assert set(columns()) <= set(loaded.columns)


def test_derived_columns_sit_beside_their_source(raw_dir: Path) -> None:
    """A source may have several derivations; all of them sit together after it."""
    frame = add_derived_columns(load_year(2011, raw_dir))
    order = list(frame.columns)

    for source, derived in (("SEVERITY", {"TNRY_SEV", "SEV4"}), ("NUM_VICTIMS", {"VICTIMS"})):
        start = order.index(source)
        following = set(order[start + 1 : start + 1 + len(derived)])
        assert following == derived, f"{source} derivations are not adjacent to it"


def test_ternary_severity_is_consistent_with_victims(raw_dir: Path) -> None:
    """A collision with victims cannot be class 0, and one without cannot exceed it."""
    frame = add_derived_columns(load_years(FIXTURE_YEARS, raw_dir))
    with_victims = frame["NUM_VICTIMS"] > 0
    assert (frame.loc[with_victims, "TNRY_SEV"] > 0).all()
    assert (frame.loc[~with_victims, "TNRY_SEV"] == 0).all()


def test_loading_is_deterministic(raw_dir: Path) -> None:
    first = load_years(FIXTURE_YEARS, raw_dir)
    second = load_years(FIXTURE_YEARS, raw_dir)
    pd.testing.assert_frame_equal(first, second)


def test_undocumented_values_become_null(raw_dir: Path, tmp_path: Path) -> None:
    """An undocumented code carries no meaning, so it is recorded as unrecorded."""
    frame = pd.read_csv(raw_dir / "saaq_2011.csv")
    before = int(frame["LIGHT"].isna().sum())
    frame.loc[0:3, "LIGHT"] = 0
    frame.to_csv(tmp_path / "saaq_2011.csv", index=False)

    loaded = load_year(2011, tmp_path)
    assert 0 not in set(loaded["LIGHT"].dropna().unique())
    assert int(loaded["LIGHT"].isna().sum()) == before + 4


def test_coercion_survives_the_float_round_trip(raw_dir: Path, tmp_path: Path) -> None:
    """A column with blanks reads as float, so '0' arrives spelled '0.0'."""
    frame = pd.read_csv(raw_dir / "saaq_2011.csv")
    assert frame["LIGHT"].isna().any(), "fixture must contain blanks for this to bite"
    frame.loc[0, "LIGHT"] = 0
    frame.to_csv(tmp_path / "saaq_2011.csv", index=False)

    assert str(pd.read_csv(tmp_path / "saaq_2011.csv")["LIGHT"].dtype) == "float64"
    assert 0 not in set(load_year(2011, tmp_path)["LIGHT"].dropna().unique())


def test_documented_values_are_untouched(raw_dir: Path) -> None:
    loaded = load_year(2011, raw_dir)
    assert set(loaded["LIGHT"].dropna().unique()) <= {1, 2, 3, 4}
