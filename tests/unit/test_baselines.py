"""Tests for the trivial predictors."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vision_zero.modeling.baselines import Baseline, all_baselines, baseline_predictions

IMBALANCED = np.array([0] * 800 + [1] * 190 + [2] * 10)


def test_majority_predicts_the_modal_class() -> None:
    prediction = baseline_predictions(Baseline.MAJORITY, IMBALANCED, n=50)
    assert set(prediction.labels.tolist()) == {0}


def test_minority_predicts_the_rarest_present_class() -> None:
    prediction = baseline_predictions(Baseline.MINORITY, IMBALANCED, n=50)
    assert set(prediction.labels.tolist()) == {2}


def test_minority_ignores_classes_absent_from_training() -> None:
    """Predicting a never-observed class is a different failure from a rare one."""
    y = np.array([0] * 10 + [2] * 2)
    prediction = baseline_predictions(Baseline.MINORITY, y, n=5, classes=np.array([0, 1, 2]))
    assert set(prediction.labels.tolist()) == {2}


def test_prior_reproduces_the_training_distribution() -> None:
    prediction = baseline_predictions(Baseline.PRIOR, IMBALANCED, n=20_000, seed=0)
    share = (prediction.labels == 0).mean()
    assert abs(share - 0.8) < 0.02


def test_uniform_ignores_the_training_distribution() -> None:
    prediction = baseline_predictions(Baseline.UNIFORM, IMBALANCED, n=20_000, seed=0)
    share = (prediction.labels == 2).mean()
    assert abs(share - 1 / 3) < 0.02


@pytest.mark.parametrize("kind", list(Baseline))
def test_probabilities_are_a_distribution(kind: Baseline) -> None:
    prediction = baseline_predictions(kind, IMBALANCED, n=100, seed=0)
    assert prediction.probabilities.shape == (100, 3)
    np.testing.assert_allclose(prediction.probabilities.sum(axis=1), 1.0)
    assert (prediction.probabilities >= 0).all()


@pytest.mark.parametrize("kind", [Baseline.MAJORITY, Baseline.MINORITY])
def test_constant_baselines_are_certain(kind: Baseline) -> None:
    prediction = baseline_predictions(kind, IMBALANCED, n=10)
    assert set(np.unique(prediction.probabilities).tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("kind", list(Baseline))
def test_requested_length_is_honoured(kind: Baseline) -> None:
    assert len(baseline_predictions(kind, IMBALANCED, n=37, seed=0).labels) == 37


def test_stochastic_baselines_are_reproducible() -> None:
    first = baseline_predictions(Baseline.PRIOR, IMBALANCED, n=500, seed=3)
    second = baseline_predictions(Baseline.PRIOR, IMBALANCED, n=500, seed=3)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_all_baselines_returns_every_kind() -> None:
    produced = all_baselines(IMBALANCED, n=10, seed=0)
    assert set(produced) == {kind.value for kind in Baseline}


def test_unknown_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a valid Baseline"):
        baseline_predictions("wishful", IMBALANCED, n=5)


@given(
    counts=st.lists(st.integers(min_value=1, max_value=200), min_size=2, max_size=5),
    n=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=40, deadline=None)
def test_predictions_are_always_valid_classes(counts: list[int], n: int) -> None:
    y = np.concatenate([np.full(count, label) for label, count in enumerate(counts)])
    for kind in Baseline:
        prediction = baseline_predictions(kind, y, n=n, seed=0)
        assert set(prediction.labels.tolist()) <= set(np.unique(y).tolist())
        assert len(prediction.labels) == n
