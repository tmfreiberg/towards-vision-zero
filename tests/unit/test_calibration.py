"""Tests for calibration, as distinct from discrimination."""

from __future__ import annotations

import numpy as np
import pytest

from vision_zero.evaluation.calibration import (
    brier_score,
    calibration_report,
    expected_calibration_error,
    reliability_table,
)


def test_a_confident_correct_prediction_scores_zero() -> None:
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert brier_score(np.array([0, 1]), probabilities, classes=[0, 1]) == pytest.approx(0.0)


def test_a_confident_wrong_prediction_scores_the_maximum() -> None:
    probabilities = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert brier_score(np.array([1, 1]), probabilities, classes=[0, 1]) == pytest.approx(2.0)


def test_brier_matches_a_hand_computation() -> None:
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    assert brier_score(np.array([0, 0, 1]), probabilities, classes=[0, 1]) == pytest.approx(
        (0.02 + 0.08 + 0.18) / 3
    )


def test_perfect_calibration_has_no_error() -> None:
    """Stated probability equals observed frequency at every level."""
    rng = np.random.default_rng(0)
    stated = rng.choice([0.1, 0.3, 0.5, 0.9], size=40_000)
    outcomes = (rng.random(40_000) < stated).astype(int)
    probabilities = np.column_stack([1 - stated, stated])

    error = expected_calibration_error(outcomes, probabilities, 1, classes=[0, 1], n_bins=10)
    assert error < 0.01


def test_systematic_overconfidence_is_detected() -> None:
    rng = np.random.default_rng(0)
    stated = np.full(20_000, 0.9)
    outcomes = (rng.random(20_000) < 0.3).astype(int)
    probabilities = np.column_stack([1 - stated, stated])

    error = expected_calibration_error(outcomes, probabilities, 1, classes=[0, 1], n_bins=10)
    assert error == pytest.approx(0.6, abs=0.02)


def test_a_model_can_be_well_calibrated_and_never_predict_the_class() -> None:
    """The distinction the whole analysis turns on, made concrete."""
    rng = np.random.default_rng(0)
    n = 50_000
    stated = np.full(n, 0.04)
    outcomes = (rng.random(n) < 0.04).astype(int)
    probabilities = np.column_stack([1 - stated, stated])

    predicted_labels = np.argmax(probabilities, axis=1)
    assert set(predicted_labels.tolist()) == {0}, "argmax never names the rare class"

    error = expected_calibration_error(outcomes, probabilities, 1, classes=[0, 1], n_bins=10)
    assert error < 0.01, "yet the stated odds are right"


def test_reliability_table_reports_only_populated_bins() -> None:
    probabilities = np.array([[0.95, 0.05]] * 100)
    table = reliability_table(np.zeros(100, dtype=int), probabilities, 1, classes=[0, 1])
    assert len(table) == 1
    assert table["n"].iloc[0] == 100


def test_reliability_gap_is_signed() -> None:
    probabilities = np.array([[0.9, 0.1]] * 100)
    outcomes = np.array([1] * 50 + [0] * 50)
    table = reliability_table(outcomes, probabilities, 1, classes=[0, 1])
    assert table["gap"].iloc[0] == pytest.approx(0.4, abs=1e-9)


def test_report_covers_every_class() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet([1, 1, 1], size=500)
    outcomes = rng.integers(0, 3, size=500)
    report = calibration_report(outcomes, probabilities, classes=[0, 1, 2])

    assert report.classes == (0, 1, 2)
    assert set(report.calibration_error) == {0, 1, 2}
    assert len(report.summary()) == 3


def test_calibration_error_is_undefined_without_data() -> None:
    empty = np.zeros((0, 2))
    assert np.isnan(expected_calibration_error(np.array([]), empty, 1, classes=[0, 1]))
