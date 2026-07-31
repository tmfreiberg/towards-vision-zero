"""Tests for the missingness diagnostic.

Blank fields are not blank at random on this data: a thorough report accompanies
a serious collision. These tests check that the effect is detected when present
and not invented when absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_zero.ceiling.missingness import (
    missing_count_profile,
    missingness_indicators,
    missingness_report,
)


def test_indicators_are_one_boolean_column_per_feature() -> None:
    X = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
    indicators = missingness_indicators(X)
    assert list(indicators.columns) == ["a_missing", "b_missing"]
    assert indicators["a_missing"].tolist() == [False, True, False]


def test_informative_blankness_shows_as_lift() -> None:
    """Blank fields concentrated on one outcome must register."""
    X = pd.DataFrame({"f": [None] * 500 + ["a"] * 500})
    y = pd.Series([0] * 500 + [0] * 250 + [1] * 250)

    report = missingness_report(X, y)
    row = report.iloc[0]
    assert row["p_class_0_missing"] == pytest.approx(1.0)
    assert row["p_class_0_present"] == pytest.approx(0.5)
    assert float(row["lift_class_0"]) == pytest.approx(2.0)


def test_uninformative_blankness_shows_no_lift() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.choice([0, 1], size=4000))
    X = pd.DataFrame({"f": pd.Series(["a"] * 4000).mask(rng.random(4000) < 0.3)})

    report = missingness_report(X, y)
    assert float(report.iloc[0]["lift_class_0"]) == pytest.approx(1.0, abs=0.1)


def test_features_without_blanks_are_omitted() -> None:
    X = pd.DataFrame({"complete": ["a"] * 100, "sparse": [None] * 60 + ["b"] * 40})
    y = pd.Series([0] * 50 + [1] * 50)
    assert set(missingness_report(X, y)["feature"]) == {"sparse"}


def test_thin_blankness_is_ignored() -> None:
    X = pd.DataFrame({"f": [None] * 3 + ["a"] * 200})
    y = pd.Series([0] * 203)
    assert missingness_report(X, y, min_missing=30).empty


def test_the_count_profile_covers_every_record() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {f"c{i}": pd.Series(["a"] * 500).mask(rng.random(500) < 0.2) for i in range(4)}
    )
    y = pd.Series(rng.choice([0, 1], size=500))

    profile = missing_count_profile(X, y)
    assert profile["n_records"].sum() == len(X)
    assert profile["share_of_records"].sum() == pytest.approx(1.0)


def test_the_count_profile_detects_a_monotone_trend() -> None:
    """More blanks accompanying milder outcomes is the pattern to catch."""
    rng = np.random.default_rng(2)
    n = 6000
    severity = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    # Reports on injury collisions are filled in more completely.
    blank_rate = np.where(severity == 0, 0.4, 0.05)
    X = pd.DataFrame(
        {f"c{i}": pd.Series(["a"] * n).mask(rng.random(n) < blank_rate) for i in range(5)}
    )
    profile = missing_count_profile(X, pd.Series(severity))
    busy = profile[profile["n_records"] >= 50]
    assert busy["p_class_0"].is_monotonic_increasing
