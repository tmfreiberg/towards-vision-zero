"""Tests for the descriptive tables."""

from __future__ import annotations

import pandas as pd
import pytest

from vision_zero.descriptive import class_shares, crosstab_shares, frequency, trend


@pytest.fixture
def records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MONTH": [1, 1, 7, 7, 7, 12],
            "WEATHER": [11, 11, 12, 12, 11, None],
            "TNRY_SEV": [0, 1, 0, 0, 2, 0],
        }
    )


def test_frequency_counts_and_shares(records: pd.DataFrame) -> None:
    table = frequency(records, "MONTH")
    assert table["n"].sum() == len(records)
    assert table["share"].sum() == pytest.approx(1.0)


def test_frequency_preserves_the_natural_order(records: pd.DataFrame) -> None:
    """Sorting an ordered variable by count would scramble the axis."""
    table = frequency(records, "MONTH", sort_by_value=True)
    assert list(table["value"]) == [1, 7, 12]


def test_frequency_can_sort_by_count(records: pd.DataFrame) -> None:
    table = frequency(records, "MONTH", sort_by_value=False)
    assert table["n"].is_monotonic_decreasing


def test_labels_come_from_the_codebook(records: pd.DataFrame) -> None:
    table = frequency(records, "MONTH")
    assert "Jul" in set(table["label"])


def test_unrecorded_values_are_labelled(records: pd.DataFrame) -> None:
    table = frequency(records, "WEATHER")
    assert "not recorded" in set(table["label"])


def test_labels_are_available_in_french(records: pd.DataFrame) -> None:
    """The bilingual codebook has to reach the output, not just exist."""
    english = frequency(records, "MONTH", lang="en")
    french = frequency(records, "MONTH", lang="fr")
    assert "Jul" in set(english["label"])
    assert "juil" in set(french["label"])


def test_class_shares_sum_to_one_per_value(records: pd.DataFrame) -> None:
    table = class_shares(records, "MONTH", "TNRY_SEV")
    shares = table[[c for c in table.columns if c.startswith("p_class_")]]
    assert shares.sum(axis=1).round(6).eq(1.0).all()


def test_class_shares_report_support(records: pd.DataFrame) -> None:
    table = class_shares(records, "MONTH", "TNRY_SEV")
    assert table["n"].sum() == len(records.dropna(subset=["MONTH"]))


def test_class_shares_can_suppress_thin_values(records: pd.DataFrame) -> None:
    table = class_shares(records, "MONTH", "TNRY_SEV", min_records=3)
    assert list(table["MONTH"]) == [7]


def test_crosstab_labels_both_axes(records: pd.DataFrame) -> None:
    table = crosstab_shares(records, "MONTH", "TNRY_SEV", normalise="none")
    assert "Jul" in list(table.index)
    assert int(table.to_numpy().sum()) == len(records)


def test_crosstab_row_shares_sum_to_one(records: pd.DataFrame) -> None:
    table = crosstab_shares(records, "MONTH", "TNRY_SEV", normalise="index")
    assert table.sum(axis=1).round(6).eq(1.0).all()


def test_trend_without_a_target_is_a_frequency(records: pd.DataFrame) -> None:
    assert list(trend(records, "MONTH")["n"]) == list(frequency(records, "MONTH")["n"])


def test_trend_with_a_target_stays_ordered(records: pd.DataFrame) -> None:
    table = trend(records, "MONTH", target="TNRY_SEV")
    assert list(table["MONTH"]) == sorted(table["MONTH"])
