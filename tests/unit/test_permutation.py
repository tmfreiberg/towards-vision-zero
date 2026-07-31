"""Tests for the permutation null."""

from __future__ import annotations

import numpy as np
import pytest

from vision_zero.ceiling.permutation import permutation_null


def test_a_score_reading_the_labels_beats_its_null() -> None:
    y = np.array([0] * 50 + [1] * 50)
    signal = np.arange(100)

    def score(labels: np.ndarray) -> float:
        return float(abs(np.corrcoef(signal, labels)[0, 1]))

    # A step function against a linear ramp correlates at sqrt(3)/2, not 1.
    result = permutation_null(score, y, n_permutations=200, seed=0)
    assert result.observed == pytest.approx(np.sqrt(3) / 2, abs=0.01)
    assert result.null_mean < 0.2
    assert result.p_value < 0.01
    assert result.effect_size > 3


def test_a_score_ignoring_the_labels_does_not_beat_its_null() -> None:
    y = np.array([0] * 50 + [1] * 50)
    result = permutation_null(lambda _: 0.5, y, n_permutations=50, seed=0)
    assert result.p_value == pytest.approx(1.0)
    assert result.effect_size == 0.0


def test_the_p_value_can_never_be_zero() -> None:
    """Finitely many shuffles cannot establish an exactly zero probability."""
    y = np.arange(20)
    result = permutation_null(lambda labels: float(labels[0]), y, n_permutations=99, seed=0)
    assert result.p_value >= 1 / 100


def test_the_smallest_attainable_p_value_is_set_by_the_count() -> None:
    y = np.arange(50)
    result = permutation_null(
        lambda labels: 1e9 if labels[0] == 0 else 0.0, y, n_permutations=9, seed=0
    )
    assert result.p_value >= 1 / 10


def test_permutation_preserves_the_class_distribution() -> None:
    """Shuffling severs the relationship and changes nothing else."""
    y = np.array([0] * 80 + [1] * 19 + [2])
    seen: list[np.ndarray] = []

    def score(labels: np.ndarray) -> float:
        seen.append(np.bincount(labels, minlength=3))
        return 0.0

    permutation_null(score, y, n_permutations=20, seed=0)
    for counts in seen:
        np.testing.assert_array_equal(counts, np.array([80, 19, 1]))


def test_results_are_reproducible() -> None:
    y = np.arange(40) % 3
    score = np.mean
    first = permutation_null(lambda a: float(score(a[:10])), y, n_permutations=30, seed=5)
    second = permutation_null(lambda a: float(score(a[:10])), y, n_permutations=30, seed=5)
    np.testing.assert_allclose(first.null_scores, second.null_scores)


def test_a_precomputed_observation_is_used() -> None:
    y = np.arange(10)
    result = permutation_null(lambda _: 0.0, y, n_permutations=5, seed=0, observed=0.9)
    assert result.observed == 0.9


def test_at_least_one_permutation_is_required() -> None:
    with pytest.raises(ValueError, match="at least one"):
        permutation_null(lambda _: 0.0, np.arange(5), n_permutations=0)


def test_summary_is_a_flat_mapping() -> None:
    result = permutation_null(lambda _: 0.5, np.arange(10), n_permutations=10, seed=0)
    summary = result.summary()
    assert set(summary) == {
        "observed",
        "null_mean",
        "null_std",
        "p_value",
        "effect_size",
        "n_permutations",
    }
