"""Golden-value tests for the ceiling estimators.

These are the load-bearing tests of the project. The central claim — that there
is a limit on achievable accuracy which belongs to the data rather than to any
model — rests on the numbers these functions return. If they are wrong, the
claim is wrong, and no amount of downstream care would reveal it.

So they are checked against constructed datasets whose ceilings are known by
hand, and against analytic values of Fano's inequality.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from vision_zero.ceiling.bayes_error import (
    held_out_ceiling,
    in_sample_ceiling,
    multiplicity_profile,
)
from vision_zero.ceiling.information import (
    binary_entropy,
    conditional_entropy,
    entropy,
    fano_error_bound,
)

TOLERANCE = 1e-9


def _cells(pattern: dict[str, list[int]], repeats: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Build a dataset from a mapping of feature value to its label sequence."""
    features: list[str] = []
    labels: list[int] = []
    for key, sequence in pattern.items():
        for _ in range(repeats):
            features.extend([key] * len(sequence))
            labels.extend(sequence)
    return pd.DataFrame({"f": features}), pd.Series(labels)


# --------------------------------------------------------------------------
# In-sample ceiling, against hand-computed values
# --------------------------------------------------------------------------


def test_a_deterministic_relationship_has_a_ceiling_of_one() -> None:
    """Each feature value always yields the same outcome: nothing is unpredictable."""
    X, y = _cells({"a": [0, 0, 0], "b": [1, 1, 1], "c": [2, 2]})
    assert in_sample_ceiling(X, y).ceiling == pytest.approx(1.0, abs=TOLERANCE)


def test_an_even_split_has_a_ceiling_of_one_half() -> None:
    """Every feature value yields both outcomes equally: half must be wrong."""
    X, y = _cells({"a": [0, 1], "b": [0, 1], "c": [0, 1]})
    assert in_sample_ceiling(X, y).ceiling == pytest.approx(0.5, abs=TOLERANCE)


def test_a_hand_computed_mixture() -> None:
    """4 of 4 from 'a', 3 of 4 from 'b', 2 of 4 from 'c': 9 of 12."""
    X, y = _cells({"a": [0, 0, 0, 0], "b": [0, 0, 0, 1], "c": [0, 0, 1, 1]})
    assert in_sample_ceiling(X, y).ceiling == pytest.approx(9 / 12, abs=TOLERANCE)


def test_three_way_split_across_three_classes() -> None:
    """One third correct at best, and the estimator must not round it up."""
    X, y = _cells({"a": [0, 1, 2], "b": [0, 1, 2]})
    assert in_sample_ceiling(X, y).ceiling == pytest.approx(1 / 3, abs=TOLERANCE)


def test_singleton_cells_inflate_the_estimate_to_one() -> None:
    """Every combination unique: the bound is vacuous, and must be seen to be."""
    X = pd.DataFrame({"f": [f"v{i}" for i in range(50)]})
    y = pd.Series(np.random.default_rng(0).integers(0, 3, size=50))
    assert in_sample_ceiling(X, y).ceiling == pytest.approx(1.0, abs=TOLERANCE)


def test_the_multiplicity_cut_off_excludes_small_cells() -> None:
    X, y = _cells({"a": [0, 1], "b": [0, 1], "lonely": [0]})
    everything = in_sample_ceiling(X, y, min_multiplicity=1)
    filtered = in_sample_ceiling(X, y, min_multiplicity=2)

    assert everything.covered == 5
    assert everything.ceiling == pytest.approx(3 / 5, abs=TOLERANCE)
    assert filtered.covered == 4
    assert filtered.ceiling == pytest.approx(0.5, abs=TOLERANCE)
    assert filtered.coverage == pytest.approx(0.8, abs=TOLERANCE)


def test_ambiguity_counts_identify_the_troublesome_cells() -> None:
    X, y = _cells({"clean": [1, 1, 1], "mixed": [0, 1]})
    estimate = in_sample_ceiling(X, y)
    assert estimate.cells == 2
    assert estimate.ambiguous_cells == 1
    assert estimate.ambiguous_records == 2


def test_records_missing_the_same_field_share_a_cell() -> None:
    """Treating each null as unique would fragment the data and inflate the bound."""
    X = pd.DataFrame({"f": [None, None, None, None]})
    y = pd.Series([0, 0, 1, 1])
    assert in_sample_ceiling(X, y).ceiling == pytest.approx(0.5, abs=TOLERANCE)


def test_mismatched_lengths_are_rejected() -> None:
    X, y = _cells({"a": [0, 1]})
    with pytest.raises(ValueError, match="rows"):
        in_sample_ceiling(X, y.iloc[:1])


def test_the_multiplicity_profile_tightens_as_the_cut_off_rises() -> None:
    """The bound and its coverage move together; a reader must see both."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f": rng.integers(0, 20, size=1000).astype(str)})
    y = pd.Series(rng.integers(0, 3, size=1000))

    profile = multiplicity_profile(X, y, thresholds=(1, 5, 20))
    assert list(profile["min_multiplicity"]) == [1, 5, 20]
    assert (profile["coverage"].diff().dropna() <= 1e-9).all()


# --------------------------------------------------------------------------
# Held-out ceiling
# --------------------------------------------------------------------------


def test_held_out_ceiling_recovers_a_deterministic_rule() -> None:
    X_train = pd.DataFrame({"f": ["a", "a", "b", "b"]})
    y_train = pd.Series([0, 0, 1, 1])
    X_val = pd.DataFrame({"f": ["a", "b"]})
    y_val = pd.Series([0, 1])

    result = held_out_ceiling(X_train, y_train, X_val, y_val)
    assert result.accuracy == pytest.approx(1.0, abs=TOLERANCE)
    assert result.coverage == pytest.approx(1.0, abs=TOLERANCE)


def test_held_out_ceiling_is_not_inflated_by_singletons() -> None:
    """The correction that makes the estimate honest: no cell fits itself."""
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"f": [f"v{i}" for i in range(200)]})
    y_train = pd.Series(rng.integers(0, 3, size=200))
    X_val = pd.DataFrame({"f": [f"v{i}" for i in range(200)]})
    y_val = pd.Series(rng.integers(0, 3, size=200))

    assert in_sample_ceiling(X_train, y_train).ceiling == pytest.approx(1.0, abs=TOLERANCE)
    assert held_out_ceiling(X_train, y_train, X_val, y_val).accuracy < 0.6


def test_unseen_combinations_are_not_scored() -> None:
    """Coverage below one is itself a finding about how sparse the space is."""
    X_train = pd.DataFrame({"f": ["a", "a"]})
    y_train = pd.Series([0, 0])
    X_val = pd.DataFrame({"f": ["a", "z"]})
    y_val = pd.Series([0, 1])

    result = held_out_ceiling(X_train, y_train, X_val, y_val)
    assert result.covered == 1
    assert result.coverage == pytest.approx(0.5, abs=TOLERANCE)
    assert result.accuracy == pytest.approx(1.0, abs=TOLERANCE)


def test_per_class_recall_exposes_an_abandoned_class() -> None:
    """The rule predicts the cell majority, so a class that is never a majority is lost."""
    X_train = pd.DataFrame({"f": ["a"] * 9 + ["a"]})
    y_train = pd.Series([0] * 9 + [1])
    X_val = pd.DataFrame({"f": ["a", "a"]})
    y_val = pd.Series([0, 1])

    result = held_out_ceiling(X_train, y_train, X_val, y_val)
    assert result.per_class_recall[0] == pytest.approx(1.0, abs=TOLERANCE)
    assert result.per_class_recall[1] == pytest.approx(0.0, abs=TOLERANCE)


# --------------------------------------------------------------------------
# Entropy and Fano, against analytic values
# --------------------------------------------------------------------------


def test_entropy_of_a_fair_coin_is_one_bit() -> None:
    assert entropy(np.array([0, 1])) == pytest.approx(1.0, abs=TOLERANCE)


def test_entropy_of_a_certainty_is_zero() -> None:
    assert entropy(np.array([3, 3, 3])) == pytest.approx(0.0, abs=TOLERANCE)


def test_entropy_of_a_uniform_distribution_is_log_of_its_support() -> None:
    for classes in (2, 3, 4, 8):
        labels = np.arange(classes)
        assert entropy(labels) == pytest.approx(np.log2(classes), abs=1e-9)


def test_binary_entropy_peaks_at_one_half() -> None:
    assert binary_entropy(0.5) == pytest.approx(1.0, abs=TOLERANCE)
    assert binary_entropy(0.0) == pytest.approx(0.0, abs=TOLERANCE)
    assert binary_entropy(1.0) == pytest.approx(0.0, abs=TOLERANCE)


@pytest.mark.parametrize("classes", [2, 3, 4, 5, 10])
def test_fano_is_tight_when_the_features_say_nothing(classes: int) -> None:
    """H(Y|X) = log2(M) forces error at least (M-1)/M, and Fano attains it exactly."""
    bound = fano_error_bound(float(np.log2(classes)), n_classes=classes)
    assert bound == pytest.approx((classes - 1) / classes, abs=1e-6)


@pytest.mark.parametrize("classes", [2, 3, 5])
def test_fano_gives_no_floor_when_the_features_determine_the_outcome(classes: int) -> None:
    assert fano_error_bound(0.0, n_classes=classes) == pytest.approx(0.0, abs=TOLERANCE)


def test_fano_is_monotone_in_the_remaining_entropy() -> None:
    """More uncertainty left over cannot mean a lower floor under the error."""
    bounds = [fano_error_bound(h, n_classes=3) for h in np.linspace(0, np.log2(3), 25)]
    assert all(later >= earlier - 1e-12 for earlier, later in pairwise(bounds))


def test_fano_inverts_its_own_inequality() -> None:
    """The returned p must satisfy the inequality with equality."""
    for target in (0.25, 0.5, 1.0, 1.4):
        p = fano_error_bound(target, n_classes=3)
        assert binary_entropy(p) + p * np.log2(2) == pytest.approx(target, abs=1e-8)


def test_fano_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="at least two classes"):
        fano_error_bound(0.5, n_classes=1)
    with pytest.raises(ValueError, match="cannot be negative"):
        fano_error_bound(-0.1, n_classes=3)
    with pytest.raises(ValueError, match="exceeds"):
        fano_error_bound(5.0, n_classes=3)


def test_conditional_entropy_is_zero_under_a_deterministic_rule() -> None:
    X, y = _cells({"a": [0, 0, 0], "b": [1, 1, 1]})
    assert conditional_entropy(X, y) == pytest.approx(0.0, abs=TOLERANCE)


def test_conditional_entropy_equals_label_entropy_when_features_are_useless() -> None:
    """One cell holding everything: knowing the features changes nothing."""
    X = pd.DataFrame({"f": ["same"] * 300})
    y = pd.Series([0, 1, 2] * 100)
    assert conditional_entropy(X, y) == pytest.approx(entropy(y), abs=1e-9)


def test_the_two_bounds_agree_on_a_constructed_case() -> None:
    """Ceiling 0.5 above and a Fano floor of 0.5 below: they meet, as they must."""
    X, y = _cells({"a": [0, 1], "b": [0, 1]}, repeats=50)
    ceiling = in_sample_ceiling(X, y).ceiling
    floor = fano_error_bound(conditional_entropy(X, y), n_classes=2)

    assert ceiling == pytest.approx(0.5, abs=TOLERANCE)
    assert floor == pytest.approx(0.5, abs=1e-6)
    assert ceiling == pytest.approx(1 - floor, abs=1e-6)


# --------------------------------------------------------------------------
# Absence of evidence is reported as such, not as zero
# --------------------------------------------------------------------------


def test_a_ceiling_over_no_records_is_undefined_not_zero() -> None:
    """Zero would read as 'nothing is predictable'; the truth is 'no data to say'."""
    X, y = _cells({"a": [0], "b": [1], "c": [0]})
    estimate = in_sample_ceiling(X, y, min_multiplicity=2)
    assert estimate.covered == 0
    assert np.isnan(estimate.ceiling)
    assert estimate.coverage == 0.0


def test_a_held_out_ceiling_with_no_overlap_is_undefined() -> None:
    X_train = pd.DataFrame({"f": ["a", "b"]})
    y_train = pd.Series([0, 1])
    X_val = pd.DataFrame({"f": ["y", "z"]})
    y_val = pd.Series([0, 1])

    result = held_out_ceiling(X_train, y_train, X_val, y_val)
    assert result.covered == 0
    assert np.isnan(result.accuracy)
    assert result.coverage == 0.0


def test_recall_for_a_class_absent_from_covered_records_is_undefined() -> None:
    X_train = pd.DataFrame({"f": ["a", "a"]})
    y_train = pd.Series([0, 0])
    X_val = pd.DataFrame({"f": ["a", "unseen"]})
    y_val = pd.Series([0, 1])

    result = held_out_ceiling(X_train, y_train, X_val, y_val)
    assert result.per_class_recall[0] == pytest.approx(1.0)
    assert np.isnan(result.per_class_recall[1])


def test_a_fully_sparse_dataset_reports_vacuous_coverage_at_every_cut_off() -> None:
    """Every combination unique: the estimator must say so rather than invent a number."""
    X = pd.DataFrame({"f": [f"v{i}" for i in range(40)]})
    y = pd.Series(np.random.default_rng(2).integers(0, 3, size=40))
    profile = multiplicity_profile(X, y, thresholds=(2, 5))
    assert (profile["coverage"] == 0.0).all()
    assert profile["ceiling"].isna().all()


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def test_the_profile_groups_the_records_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grouping is the expensive step and does not depend on the cut-off.

    Written as a structural check rather than a timing one, which would be
    flaky. On a few hundred thousand records the difference between grouping
    once and grouping per threshold is the difference between a second and
    several minutes.
    """
    from vision_zero.ceiling import bayes_error

    calls = 0
    original = bayes_error.cell_labels

    def counted(*args: object, **kwargs: object) -> pd.Series:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(bayes_error, "cell_labels", counted)

    X, y = _cells({"a": [0, 1], "b": [0, 0], "c": [1, 1]})
    bayes_error.multiplicity_profile(X, y, thresholds=(1, 2, 5, 10))
    assert calls == 1


def test_precomputed_cells_give_the_same_answer() -> None:
    """The optimisation must not change the number."""
    from vision_zero.ceiling.bayes_error import cell_labels

    X, y = _cells({"a": [0, 0, 1], "b": [1, 1, 1], "c": [0, 1]})
    assert in_sample_ceiling(X, y).ceiling == pytest.approx(
        in_sample_ceiling(X, y, cells=cell_labels(X)).ceiling, abs=TOLERANCE
    )


def test_null_values_group_together_across_columns() -> None:
    """Two records unrecorded in the same fields are the same combination."""
    from vision_zero.ceiling.bayes_error import cell_labels

    X = pd.DataFrame({"a": [None, None, "x"], "b": ["p", "p", "p"]})
    labels = cell_labels(X)
    assert labels.iloc[0] == labels.iloc[1]
    assert labels.iloc[0] != labels.iloc[2]


# --------------------------------------------------------------------------
# The entropy bracket
# --------------------------------------------------------------------------


def test_the_bracket_straddles_the_truth_when_features_are_useless() -> None:
    """H(Y|X) equals H(Y) here, and the two biased estimates must sit either side."""
    from vision_zero.ceiling.information import conditional_entropy_bracket

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f": rng.integers(0, 50, size=4000).astype(str)})
    y = pd.Series(rng.integers(0, 3, size=4000))
    bracket = conditional_entropy_bracket(
        X.iloc[:3000], y.iloc[:3000], X.iloc[3000:], y.iloc[3000:]
    )

    assert bracket.plug_in < bracket.label_entropy < bracket.held_out


def test_the_bracket_collapses_under_a_deterministic_rule() -> None:
    from vision_zero.ceiling.information import conditional_entropy_bracket

    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f": rng.integers(0, 50, size=4000).astype(str)})
    y = pd.Series(X["f"].astype(int) % 3)
    bracket = conditional_entropy_bracket(
        X.iloc[:3000], y.iloc[:3000], X.iloc[3000:], y.iloc[3000:]
    )

    assert bracket.plug_in == pytest.approx(0.0, abs=1e-9)
    assert bracket.held_out < 0.1
    assert bracket.conservative_floor == pytest.approx(0.0, abs=1e-9)


def test_the_conservative_floor_never_exceeds_the_optimistic_one() -> None:
    """The quoted bound must be the weaker of the two, or it is not conservative."""
    from vision_zero.ceiling.information import conditional_entropy_bracket

    rng = np.random.default_rng(2)
    X = pd.DataFrame({"f": rng.integers(0, 30, size=3000).astype(str)})
    y = pd.Series(rng.integers(0, 3, size=3000))
    bracket = conditional_entropy_bracket(
        X.iloc[:2000], y.iloc[:2000], X.iloc[2000:], y.iloc[2000:]
    )

    assert bracket.conservative_floor <= bracket.optimistic_floor + 1e-12


def test_held_out_entropy_reports_coverage_and_declines_without_it() -> None:
    from vision_zero.ceiling.information import held_out_conditional_entropy

    X_train = pd.DataFrame({"f": ["a", "a", "b", "b"]})
    y_train = pd.Series([0, 1, 0, 1])
    X_val = pd.DataFrame({"f": ["y", "z"]})
    y_val = pd.Series([0, 1])

    value, coverage = held_out_conditional_entropy(X_train, y_train, X_val, y_val)
    assert np.isnan(value)
    assert coverage == 0.0


def test_smoothing_keeps_an_unseen_outcome_finite() -> None:
    """An outcome absent from its training cell must not contribute infinite surprise."""
    from vision_zero.ceiling.information import held_out_conditional_entropy

    X_train = pd.DataFrame({"f": ["a"] * 20})
    y_train = pd.Series([0] * 20)
    X_val = pd.DataFrame({"f": ["a"]})
    y_val = pd.Series([1])

    value, coverage = held_out_conditional_entropy(X_train, y_train, X_val, y_val)
    assert np.isfinite(value)
    assert coverage == 1.0
