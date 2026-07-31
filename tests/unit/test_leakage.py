"""Tests for the leakage audit.

The audit exists because a variable that restates the target does not announce
itself in its name. It has to separate three cases: a value that entails an
outcome, a value that merely predicts it very well, and a value that says
little. Only the first is leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_zero.ceiling.leakage import (
    conditional_class_distribution,
    deterministic_values,
    excluded_outcomes,
    leakage_report,
)


@pytest.fixture
def planted() -> tuple[pd.DataFrame, pd.Series]:
    """Three features: one entailing the target, one strong, one useless."""
    rng = np.random.default_rng(0)
    n = 5000
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.8, 0.19, 0.01]))
    X = pd.DataFrame(
        {
            "noise": rng.choice(list("abcde"), size=n),
            "strong": np.where(rng.random(n) < 0.97, y.map({0: "lo", 1: "hi", 2: "hi"}), "lo"),
            "entails": np.where(y > 0, "Y", "N"),
        }
    )
    return X, y


def test_a_variable_that_entails_the_target_is_flagged(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    assert "entails" in set(deterministic_values(X, y)["feature"])


def test_a_merely_strong_predictor_is_not_flagged(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    """97% is evidence. 100% is a definition. The audit must not confuse them."""
    X, y = planted
    assert "strong" not in set(deterministic_values(X, y)["feature"])


def test_noise_is_not_flagged(planted: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = planted
    assert "noise" not in set(deterministic_values(X, y)["feature"])


def test_independent_features_produce_an_empty_result() -> None:
    """The reassuring outcome must be representable."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.choice(list("xyz"), size=2000)})
    y = pd.Series(rng.choice([0, 1], size=2000))
    assert deterministic_values(X, y).empty
    assert excluded_outcomes(X, y).empty


def test_exclusion_catches_what_certainty_misses() -> None:
    """A value can rule a class out without fixing which of the rest applies.

    The negative value must stay uninformative, as it is in the real data: most
    injury collisions involve no pedestrian at all, so `PED = N` says nothing.
    """
    rng = np.random.default_rng(7)
    n = 6000
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.8, 0.19, 0.01]))
    involved = rng.random(n) < 0.10
    X = pd.DataFrame({"victim": np.where((y.to_numpy() > 0) & involved, "Y", "N")})

    assert deterministic_values(X, y).empty, "neither value fixes a single class"
    excluded = excluded_outcomes(X, y)
    assert excluded["excluded_class"].tolist() == [0]
    assert excluded["value"].tolist() == ["Y"]


def test_a_rare_class_absence_is_not_treated_as_evidence() -> None:
    """Never seeing a one-percent class among fifty records proves nothing."""
    rng = np.random.default_rng(3)
    y = pd.Series(rng.choice([0, 1], size=3000, p=[0.995, 0.005]))
    X = pd.DataFrame({"f": rng.choice(list("abcdefghij"), size=3000)})
    assert excluded_outcomes(X, y).empty


def test_thinly_supported_values_are_ignored() -> None:
    """A value seen twice and followed twice by one outcome is unremarkable."""
    X = pd.DataFrame({"f": ["rare", "rare", *["common"] * 200]})
    y = pd.Series([1, 1, *list(np.resize([0, 1], 200))])
    assert deterministic_values(X, y, min_support=30).empty
    assert not deterministic_values(X, y, min_support=2).empty


def test_the_report_ranks_entailment_above_association(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    report = leakage_report(X, y).set_index("feature")
    peaks = report["max_share"].astype(float).to_dict()
    assert peaks["entails"] == pytest.approx(1.0)
    assert 0.9 < peaks["strong"] < 1.0
    assert peaks["noise"] < 0.9


def test_the_report_is_sorted_by_severity(planted: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = planted
    assert leakage_report(X, y)["max_share"].is_monotonic_decreasing


def test_conditional_distribution_sums_to_one_per_value(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    table = conditional_class_distribution(X, y, "strong")
    shares = table[[column for column in table.columns if column.startswith("p_class_")]]
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)


def test_conditional_distribution_reports_support(
    planted: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = planted
    table = conditional_class_distribution(X, y, "noise")
    assert table["n"].sum() == len(X)


def test_the_audit_would_have_caught_a_victim_indicator() -> None:
    """A regression test for the specific mistake this module was written after.

    `PED` records a pedestrian victim rather than a pedestrian involved, so it
    entails a non-zero severity. Reconstructed here in miniature.
    """
    rng = np.random.default_rng(2)
    n = 4000
    severity = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.8, 0.19, 0.01]))
    involved = rng.random(n) < 0.05
    X = pd.DataFrame(
        {
            "PED_victim": np.where((severity > 0) & involved, "Y", "N"),
            "BICYC_involved": np.where(involved, "Y", "N"),
        }
    )
    # It does not fix the class — a pedestrian victim may be minor or serious —
    # so the certainty test misses it. What it does is rule out class 0.
    assert "PED_victim" not in set(deterministic_values(X, severity)["feature"])

    excluded = excluded_outcomes(X, severity)
    victim_rows = excluded[excluded["feature"] == "PED_victim"]
    assert not victim_rows.empty
    assert victim_rows["excluded_class"].tolist() == [0]
    assert "BICYC_involved" not in set(excluded["feature"])
