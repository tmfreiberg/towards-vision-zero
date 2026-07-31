"""How much to trust the probabilities, as distinct from the labels.

A classifier that cannot pick the right label may still be right about the odds.
These are different abilities, and this project turns on the distinction.

Discrimination asks whether the highest-probability class is the correct one.
On severely imbalanced data with weak features it can be hopeless: if no
combination of recorded conditions ever makes a serious injury the most likely
outcome, then argmax never names it, and recall on that class is near zero
however well the model understands the risk.

Calibration asks something weaker and more useful. When the model says four
percent, do four percent of such collisions turn out serious? A model can be
well calibrated and almost useless as a classifier at the same time, and when
that happens the honest output was never a label. It was the distribution.

That is the conclusion the analysis reaches, and this module is where it is
checked rather than asserted.

Examples
--------
>>> import numpy as np
>>> probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
>>> y_true = np.array([0, 0, 1])
>>> round(brier_score(y_true, probabilities, classes=[0, 1]), 4)
0.0933
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CalibrationReport",
    "brier_score",
    "calibration_report",
    "expected_calibration_error",
    "reliability_table",
]


def brier_score(y_true: np.ndarray, probabilities: np.ndarray, classes: Sequence[int]) -> float:
    """Return the multiclass Brier score: mean squared error of the probabilities.

    Lower is better, and zero is a perfect confident prediction. Unlike accuracy
    it rewards being right about the odds even when the argmax is wrong, which
    is exactly the distinction that matters here.

    Parameters
    ----------
    y_true
        Actual labels.
    probabilities
        Predicted probability per class, columns in the order of ``classes``.
    classes
        Class labels, matching the probability columns.

    Returns
    -------
    float
        Mean squared difference between the predicted distribution and the
        one-hot truth.
    """
    labels = np.asarray(classes)
    truth = np.zeros_like(probabilities, dtype=float)
    for index, label in enumerate(labels):
        truth[np.asarray(y_true) == label, index] = 1.0
    return float(((probabilities - truth) ** 2).sum(axis=1).mean())


def reliability_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_index: int,
    *,
    classes: Sequence[int],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compare predicted probability against observed frequency, in bins.

    The row to read for this project is the one covering low predicted
    probabilities of the severe class. If the model says two percent and two
    percent of those collisions were serious, the model knows something real
    even though it will never predict that class outright.

    Parameters
    ----------
    y_true
        Actual labels.
    probabilities
        Predicted probability per class.
    class_index
        Which column to assess.
    classes
        Class labels, matching the columns.
    n_bins
        Number of equal-width bins over :math:`[0, 1]`.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty bin: predicted mean, observed frequency, count,
        and the signed gap between them.
    """
    label = np.asarray(classes)[class_index]
    predicted = np.asarray(probabilities)[:, class_index]
    observed = (np.asarray(y_true) == label).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assigned = np.clip(np.digitize(predicted, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for index in range(n_bins):
        mask = assigned == index
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": edges[index],
                "bin_upper": edges[index + 1],
                "n": int(mask.sum()),
                "predicted": float(predicted[mask].mean()),
                "observed": float(observed[mask].mean()),
                "gap": float(observed[mask].mean() - predicted[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_index: int,
    *,
    classes: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Return the count-weighted mean absolute gap between predicted and observed.

    Zero means the stated probabilities match the observed frequencies at every
    level. It says nothing about whether the model can separate the classes, and
    that silence is the point.
    """
    table = reliability_table(y_true, probabilities, class_index, classes=classes, n_bins=n_bins)
    if table.empty:
        return float("nan")
    return float((table["gap"].abs() * table["n"]).sum() / table["n"].sum())


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Calibration alongside the discrimination it is distinct from.

    Attributes
    ----------
    brier
        Multiclass Brier score.
    calibration_error
        Expected calibration error per class.
    reliability
        Reliability table per class.
    classes
        Class labels.
    """

    brier: float
    calibration_error: dict[int, float]
    reliability: dict[int, pd.DataFrame]
    classes: tuple[int, ...]

    def summary(self) -> pd.DataFrame:
        """One row per class: calibration error and the range of stated probabilities."""
        rows = []
        for label in self.classes:
            table = self.reliability[label]
            rows.append(
                {
                    "class": label,
                    "calibration_error": self.calibration_error[label],
                    "max_predicted": float(table["predicted"].max()) if not table.empty else 0.0,
                    "observed_rate": (
                        float((table["observed"] * table["n"]).sum() / table["n"].sum())
                        if not table.empty
                        else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)


def calibration_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: Sequence[int],
    n_bins: int = 10,
) -> CalibrationReport:
    """Assess calibration for every class at once."""
    labels = tuple(int(label) for label in classes)
    return CalibrationReport(
        brier=brier_score(y_true, probabilities, classes),
        calibration_error={
            label: expected_calibration_error(
                y_true, probabilities, index, classes=classes, n_bins=n_bins
            )
            for index, label in enumerate(labels)
        },
        reliability={
            label: reliability_table(y_true, probabilities, index, classes=classes, n_bins=n_bins)
            for index, label in enumerate(labels)
        },
        classes=labels,
    )
