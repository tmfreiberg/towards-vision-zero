"""Trivial predictors, for comparison.

A classification score means nothing on its own. Eighty percent accuracy sounds
respectable until one notices that always answering "material damage" scores
eighty percent on this data without looking at anything. These baselines make
that comparison explicit and unavoidable.

Four are provided, each answering a different question:

``majority``
    Always the most common class. The number any useful model must beat.
``minority``
    Always the rarest class. Terrible on accuracy, but its recall on the class
    that matters is perfect, which is a useful reminder that a single metric can
    be gamed by an obviously worthless predictor.
``prior``
    Draws in proportion to the training distribution. What guessing looks like
    when you know the base rates and nothing else.
``uniform``
    Draws uniformly across classes. Knowing nothing at all.

Examples
--------
>>> import numpy as np
>>> from vision_zero.modeling.baselines import Baseline, baseline_predictions
>>> y = np.array([0, 0, 0, 0, 1, 2])
>>> baseline_predictions(Baseline.MAJORITY, y, n=3).labels.tolist()
[0, 0, 0]
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

__all__ = ["Baseline", "BaselinePrediction", "all_baselines", "baseline_predictions"]


class Baseline(StrEnum):
    """The trivial predictors worth comparing against."""

    MAJORITY = "majority"
    MINORITY = "minority"
    PRIOR = "prior"
    UNIFORM = "uniform"


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    """Labels and class probabilities from a trivial predictor.

    Attributes
    ----------
    labels
        Predicted class for each record.
    probabilities
        Predicted probability per class, columns in ascending class order.
    classes
        Class labels, ascending.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray


def baseline_predictions(
    kind: Baseline | str,
    y_train: np.ndarray,
    n: int,
    *,
    seed: int = 42,
    classes: np.ndarray | None = None,
) -> BaselinePrediction:
    """Produce predictions from a trivial rule fitted to the training labels.

    Parameters
    ----------
    kind
        Which baseline.
    y_train
        Training labels, from which class frequencies are taken.
    n
        Number of predictions to produce.
    seed
        Seed for the stochastic baselines.
    classes
        Class labels to predict over. Inferred from ``y_train`` when omitted.
        Worth passing explicitly when a fold might not contain every class.

    Returns
    -------
    BaselinePrediction
        Labels and probabilities.

    Raises
    ------
    ValueError
        If the baseline is not recognised.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0] * 90 + [1] * 9 + [2])
    >>> prediction = baseline_predictions("prior", y, n=1000, seed=0)
    >>> round(float((prediction.labels == 0).mean()), 1)
    0.9
    """
    kind = Baseline(kind)
    labels = np.asarray(classes if classes is not None else np.unique(y_train))
    counts = np.array([(y_train == label).sum() for label in labels], dtype=float)
    rng = np.random.default_rng(seed)

    if kind is Baseline.MAJORITY:
        chosen = int(np.argmax(counts))
        return _constant(chosen, labels, n)

    if kind is Baseline.MINORITY:
        # Ignore classes absent from the training fold: predicting a class that
        # was never observed is a different failure from predicting a rare one.
        present = np.where(counts > 0, counts, np.inf)
        return _constant(int(np.argmin(present)), labels, n)

    if kind is Baseline.PRIOR:
        weights = counts / counts.sum()
    else:
        weights = np.full(len(labels), 1 / len(labels))

    drawn = rng.choice(len(labels), size=n, p=weights)
    probabilities = np.tile(weights, (n, 1))
    return BaselinePrediction(labels=labels[drawn], probabilities=probabilities, classes=labels)


def _constant(index: int, labels: np.ndarray, n: int) -> BaselinePrediction:
    """Predictions that are always one class, with certainty."""
    probabilities = np.zeros((n, len(labels)))
    probabilities[:, index] = 1.0
    return BaselinePrediction(
        labels=np.full(n, labels[index]), probabilities=probabilities, classes=labels
    )


def all_baselines(
    y_train: np.ndarray, n: int, *, seed: int = 42, classes: np.ndarray | None = None
) -> dict[str, BaselinePrediction]:
    """Produce every baseline at once, keyed by name."""
    return {
        kind.value: baseline_predictions(kind, y_train, n, seed=seed, classes=classes)
        for kind in Baseline
    }
