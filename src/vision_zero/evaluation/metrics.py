r"""Scoring rules.

Classification scores on this problem are easy to misread. The majority class
holds four fifths of the records, so accuracy near eighty percent means a model
has learned nothing, and a model that never predicts the severe class at all can
still look respectable on any metric that averages over classes by frequency.

What is wanted is a score that weights classes by what it costs to get them
wrong, and that reports per-class figures alongside any aggregate so the
aggregate cannot hide a class it has abandoned.

The aggregate used here is

.. math::

    F^{*} = \frac{\sum_k w_k F^{(k)}_{\beta_k}}{\sum_k w_k},

with :math:`\beta_k > 1` weighting recall for classes where a miss is costly,
:math:`\beta_k < 1` weighting precision where a false alarm is costly, and
:math:`w_k` counteracting the class imbalance.

Examples
--------
>>> import numpy as np
>>> y_true = np.array([0, 0, 0, 1, 1, 2])
>>> y_pred = np.array([0, 0, 1, 1, 0, 0])
>>> scores = classification_scores(y_true, y_pred, beta=[0.5, 1.0, 2.0], weights=[1.0, 4.0, 16.0])
>>> round(scores.f_star, 4)
0.1203
>>> scores.recall[2]
0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["ClassificationScores", "classification_scores", "f_beta", "f_star"]


def f_beta(precision: np.ndarray, recall: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute the F-beta score class by class.

    Parameters
    ----------
    precision, recall
        Per-class values in :math:`[0, 1]`.
    beta
        Per-class beta. Above one weights recall; below one weights precision.

    Returns
    -------
    numpy.ndarray
        Per-class F-beta.

    Notes
    -----
    A class with neither precision nor recall scores zero, not undefined. This
    is the usual convention and the only one that composes: a model that never
    predicts the severe class and never gets it right has failed on that class,
    and an aggregate should say so rather than evaporate into a null.

    Examples
    --------
    >>> import numpy as np
    >>> f_beta(np.array([0.5]), np.array([0.5]), np.array([1.0]))
    array([0.5])
    >>> f_beta(np.array([0.0]), np.array([0.0]), np.array([2.0]))
    array([0.])
    """
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    beta = np.asarray(beta, dtype=float)

    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    numerator = (1 + beta_squared) * precision * recall

    result: np.ndarray = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    return result


def f_star(scores: np.ndarray, weights: np.ndarray) -> float:
    """Combine per-class scores into a weighted mean.

    Parameters
    ----------
    scores
        Per-class F-beta.
    weights
        Per-class weight. Must not sum to zero.

    Returns
    -------
    float
        The weighted mean.

    Raises
    ------
    ValueError
        If the lengths disagree, or the weights sum to zero.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if scores.shape != weights.shape:
        raise ValueError(f"{scores.shape} scores but {weights.shape} weights")
    total = weights.sum()
    if total == 0:
        raise ValueError("Weights sum to zero")

    return float((scores * weights).sum() / total)


@dataclass(frozen=True, slots=True)
class ClassificationScores:
    """Per-class and aggregate scores for one set of predictions.

    Attributes
    ----------
    classes
        Class labels, ascending.
    support
        Actual records per class.
    predicted
        Predicted records per class. A zero here means the model never chose
        that class, which no aggregate should be allowed to obscure.
    precision, recall, fbeta
        Per-class values.
    f_star
        The weighted aggregate.
    accuracy
        Fraction correct. Reported because readers expect it, and shown beside
        the majority-class share so it can be read properly.
    balanced_accuracy
        Mean per-class recall.
    mcc
        Matthews correlation coefficient. Zero means performance
        indistinguishable from guessing at the observed base rates, which makes
        it the most direct answer to "is this better than chance".
    confusion
        Rows actual, columns predicted.
    """

    classes: np.ndarray
    support: np.ndarray
    predicted: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    fbeta: np.ndarray
    f_star: float
    accuracy: float
    balanced_accuracy: float
    mcc: float
    confusion: np.ndarray

    @property
    def abandoned_classes(self) -> tuple[int, ...]:
        """Classes the model never predicted, ascending.

        Worth surfacing on its own: a model that abandons the class the analysis
        exists to study has told you something no aggregate score will.
        """
        return tuple(
            int(label)
            for label, count in zip(self.classes, self.predicted, strict=True)
            if count == 0
        )


def classification_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    beta: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    classes: Sequence[int] | None = None,
) -> ClassificationScores:
    """Score predictions class by class and in aggregate.

    Parameters
    ----------
    y_true, y_pred
        Actual and predicted labels.
    beta
        Per-class beta, in class order. Defaults to one throughout.
    weights
        Per-class weight, in class order. Defaults to one throughout.
    classes
        Class labels to report on. Inferred from the data when omitted; pass it
        explicitly when a class might be absent from both actual and predicted
        and still needs a row.

    Returns
    -------
    ClassificationScores
        Per-class and aggregate figures.

    Raises
    ------
    ValueError
        If the inputs differ in length, or beta or weights disagree with the
        number of classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"{y_true.shape} actual labels but {y_pred.shape} predicted")

    labels = np.asarray(
        classes if classes is not None else np.union1d(np.unique(y_true), np.unique(y_pred))
    )
    n_classes = len(labels)

    beta_array = np.ones(n_classes) if beta is None else np.asarray(beta, dtype=float)
    weight_array = np.ones(n_classes) if weights is None else np.asarray(weights, dtype=float)
    for name, array in (("beta", beta_array), ("weights", weight_array)):
        if array.shape != (n_classes,):
            raise ValueError(f"{n_classes} classes but {array.shape[0]} {name} entries")

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)

    precision = np.divide(
        correct, predicted, out=np.zeros(n_classes, dtype=float), where=predicted > 0
    )
    recall = np.divide(correct, support, out=np.zeros(n_classes, dtype=float), where=support > 0)
    per_class = f_beta(precision, recall, beta_array)

    with np.errstate(invalid="ignore"):
        mcc = float(matthews_corrcoef(y_true, y_pred))

    return ClassificationScores(
        classes=labels,
        support=support,
        predicted=predicted,
        precision=precision,
        recall=recall,
        fbeta=per_class,
        f_star=f_star(per_class, weight_array),
        accuracy=float(correct.sum() / matrix.sum()) if matrix.sum() else 0.0,
        balanced_accuracy=float(recall[support > 0].mean()) if (support > 0).any() else 0.0,
        mcc=0.0 if np.isnan(mcc) else mcc,
        confusion=matrix,
    )
