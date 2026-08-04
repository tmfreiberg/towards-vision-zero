"""Tests for the scoring rules."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vision_zero.evaluation.metrics import classification_scores


def test_perfect_prediction_scores_one_throughout() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    scores = classification_scores(y, y)
    assert scores.accuracy == pytest.approx(1.0)
    assert scores.balanced_accuracy == pytest.approx(1.0)
    assert scores.macro_f1 == pytest.approx(1.0)
    assert scores.mcc == pytest.approx(1.0)


def test_a_class_with_neither_precision_nor_recall_scores_zero() -> None:
    """Not undefined. A model that never finds a class has failed on it."""
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 0])
    scores = classification_scores(y_true, y_pred)
    assert scores.f1[1] == pytest.approx(0.0)
    assert np.isfinite(scores.macro_f1)


def test_the_metric_this_project_ranks_by_is_zero_for_a_trivial_predictor() -> None:
    """The reason MCC replaced a weighted average of per-class scores."""
    y_true = np.array([0] * 80 + [1] * 19 + [2])
    y_pred = np.zeros_like(y_true)
    scores = classification_scores(y_true, y_pred)

    assert scores.accuracy == pytest.approx(0.8)
    assert scores.mcc == pytest.approx(0.0)
    assert scores.balanced_accuracy == pytest.approx(1 / 3)


def test_abandoned_classes_are_reported() -> None:
    """The most important thing a score can tell you about this problem."""
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 0, 0])
    scores = classification_scores(y_true, y_pred)
    assert scores.abandoned_classes == (1, 2)
    assert scores.recall[1] == pytest.approx(0.0)


def test_classes_can_be_addressed_by_label() -> None:
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 2, 2, 0])
    scores = classification_scores(y_true, y_pred)
    assert scores.recall_for(2) == pytest.approx(0.5)
    assert scores.precision_for(0) == pytest.approx(0.5)


def test_the_per_class_table_hides_nothing() -> None:
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 0])
    table = classification_scores(y_true, y_pred).per_class()

    assert list(table["class"]) == [0, 1, 2]
    assert table["support"].sum() == len(y_true)
    assert table["predicted"].sum() == len(y_pred)
    assert set(table.columns) == {"class", "support", "predicted", "precision", "recall", "f1"}


def test_absent_classes_still_get_a_row_when_declared() -> None:
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 1])
    scores = classification_scores(y_true, y_pred, classes=[0, 1, 2])
    assert len(scores.classes) == 3
    assert scores.support[2] == 0
    assert scores.f1[2] == pytest.approx(0.0)


def test_an_absent_class_does_not_drag_the_aggregates_down() -> None:
    """A class with no records is not a class the model failed on."""
    y = np.array([0, 0, 1, 1])
    scores = classification_scores(y, y, classes=[0, 1, 2])
    assert scores.balanced_accuracy == pytest.approx(1.0)
    assert scores.macro_f1 == pytest.approx(1.0)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="predicted"):
        classification_scores(np.array([0, 1]), np.array([0]))


def test_the_confusion_matrix_accounts_for_every_record() -> None:
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 0, 2, 1, 1])
    scores = classification_scores(y_true, y_pred)
    assert scores.confusion.sum() == len(y_true)
    np.testing.assert_array_equal(scores.confusion.sum(axis=1), scores.support)
    np.testing.assert_array_equal(scores.confusion.sum(axis=0), scores.predicted)


@given(
    n=st.integers(min_value=5, max_value=200),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50, deadline=None)
def test_every_reported_quantity_is_finite_and_bounded(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 3, size=n)
    y_pred = rng.integers(0, 3, size=n)
    scores = classification_scores(y_true, y_pred, classes=[0, 1, 2])

    for array in (scores.precision, scores.recall, scores.f1):
        assert np.all(np.isfinite(array))
        assert np.all((array >= 0) & (array <= 1 + 1e-12))
    for value in (scores.accuracy, scores.balanced_accuracy, scores.macro_f1):
        assert 0.0 <= value <= 1.0 + 1e-12
    assert -1.0 - 1e-12 <= scores.mcc <= 1.0 + 1e-12
