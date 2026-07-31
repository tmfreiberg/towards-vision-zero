"""Tests for the scoring rules."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vision_zero.evaluation.metrics import classification_scores, f_beta, f_star


def test_f1_of_equal_precision_and_recall_is_that_value() -> None:
    assert f_beta(np.array([0.5]), np.array([0.5]), np.array([1.0])) == pytest.approx([0.5])


def test_a_class_with_neither_precision_nor_recall_scores_zero() -> None:
    """Not undefined. A model that never finds a class has failed on it."""
    assert f_beta(np.array([0.0]), np.array([0.0]), np.array([2.0])) == pytest.approx([0.0])


def test_one_zero_class_does_not_void_the_aggregate() -> None:
    """The whole score must not evaporate because one class was abandoned."""
    scores = f_beta(np.array([0.9, 0.8, 0.0]), np.array([0.9, 0.7, 0.0]), np.array([0.5, 1.0, 2.0]))
    aggregate = f_star(scores, np.array([1.0, 4.0, 16.0]))
    assert np.isfinite(aggregate)
    assert 0.0 < aggregate < 0.2


def test_beta_above_one_weights_recall() -> None:
    high_recall = f_beta(np.array([0.2]), np.array([0.9]), np.array([2.0]))[0]
    high_precision = f_beta(np.array([0.9]), np.array([0.2]), np.array([2.0]))[0]
    assert high_recall > high_precision


def test_beta_below_one_weights_precision() -> None:
    high_recall = f_beta(np.array([0.2]), np.array([0.9]), np.array([0.5]))[0]
    high_precision = f_beta(np.array([0.9]), np.array([0.2]), np.array([0.5]))[0]
    assert high_precision > high_recall


def test_perfect_prediction_scores_one_throughout() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    scores = classification_scores(y, y, beta=[0.5, 1.0, 2.0], weights=[1.0, 4.0, 16.0])
    assert scores.f_star == pytest.approx(1.0)
    assert scores.accuracy == pytest.approx(1.0)
    assert scores.mcc == pytest.approx(1.0)


def test_the_aggregate_is_a_weighted_mean() -> None:
    scores = np.array([1.0, 0.0, 0.5])
    assert f_star(scores, np.array([1.0, 1.0, 1.0])) == pytest.approx(0.5)
    assert f_star(scores, np.array([0.0, 0.0, 1.0])) == pytest.approx(0.5)
    assert f_star(scores, np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_weights_summing_to_zero_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to zero"):
        f_star(np.array([1.0, 0.0]), np.array([0.0, 0.0]))


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="weights"):
        f_star(np.array([1.0, 0.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="predicted"):
        classification_scores(np.array([0, 1]), np.array([0]))
    with pytest.raises(ValueError, match="entries"):
        classification_scores(np.array([0, 1]), np.array([0, 1]), beta=[1.0])


def test_abandoned_classes_are_reported() -> None:
    """The most important thing a score can tell you about this problem."""
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 0, 0])
    scores = classification_scores(y_true, y_pred)
    assert scores.abandoned_classes == (1, 2)
    assert scores.recall[1] == pytest.approx(0.0)


def test_a_majority_only_model_scores_well_on_accuracy_and_badly_on_f_star() -> None:
    """The reason accuracy is the wrong headline for this problem."""
    y_true = np.array([0] * 80 + [1] * 19 + [2])
    y_pred = np.zeros_like(y_true)
    scores = classification_scores(y_true, y_pred, beta=[0.5, 1.0, 2.0], weights=[1.0, 4.0, 16.0])
    assert scores.accuracy == pytest.approx(0.8)
    assert scores.f_star < 0.05
    assert scores.mcc == pytest.approx(0.0)


def test_absent_classes_still_get_a_row_when_declared() -> None:
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 1])
    scores = classification_scores(y_true, y_pred, classes=[0, 1, 2])
    assert len(scores.classes) == 3
    assert scores.support[2] == 0
    assert scores.fbeta[2] == pytest.approx(0.0)


@given(
    precision=st.floats(min_value=0.0, max_value=1.0),
    recall=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=100, deadline=None)
def test_f_beta_is_always_a_proportion(precision: float, recall: float, beta: float) -> None:
    value = f_beta(np.array([precision]), np.array([recall]), np.array([beta]))[0]
    assert 0.0 <= value <= 1.0 + 1e-12
    assert np.isfinite(value)


@given(
    recall=st.floats(min_value=0.01, max_value=0.99),
    delta=st.floats(min_value=0.001, max_value=0.5),
)
@settings(max_examples=50, deadline=None)
def test_f_beta_increases_with_recall(recall: float, delta: float) -> None:
    lower = f_beta(np.array([0.5]), np.array([recall]), np.array([2.0]))[0]
    upper = f_beta(np.array([0.5]), np.array([min(recall + delta, 1.0)]), np.array([2.0]))[0]
    assert upper >= lower - 1e-12
