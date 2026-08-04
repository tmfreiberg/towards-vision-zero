"""Scoring rules.

Classification scores on this problem are easy to misread. The majority class
holds four fifths of the records, so accuracy near eighty percent means a model
has learned nothing, and a model that never predicts the severe class at all can
still look respectable on any metric that averages over classes by frequency.

The response taken here is to prefer reporting over aggregating. Precision,
recall and F1 are given for every class alongside its support, which is what
`sklearn`'s classification report provides and what most published work shows,
and no weighting is applied that a reader would have to accept on trust.

Where a single number is needed, for ranking estimator families or tracing a
learning curve, it is the Matthews correlation coefficient. It has no tunable
parameters, it is near zero for any degenerate predictor including the one that
always answers with the majority class, and on imbalanced data it is the
conventional choice for exactly that reason [@chicco2020]. Balanced accuracy and
macro-averaged F1 are reported beside it; both are standard and neither has any
free parameters either.

Where the costs of different mistakes genuinely differ, the conventional way to
say so is an explicit cost matrix rather than a weighted average of per-class
scores: stating that missing a serious collision is worth some number of false
alarms is a claim a reader can dispute, whereas a set of per-class weights is
not. This module does not impose one. The trade is shown directly instead, by
:func:`~vision_zero.ceiling.bayes_error.achievable_frontier`.

Examples
--------
>>> import numpy as np
>>> y_true = np.array([0, 0, 0, 1, 1, 2])
>>> y_pred = np.array([0, 0, 1, 1, 0, 0])
>>> scores = classification_scores(y_true, y_pred)
>>> round(scores.mcc, 4)
0.1066
>>> scores.recall[2]
0.0
>>> scores.abandoned_classes
(2,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

__all__ = ["ClassificationScores", "classification_scores"]


def _f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Harmonic mean of precision and recall, class by class.

    A class with neither precision nor recall scores zero rather than
    undefined. This is the usual convention and the only one that composes: a
    model that never predicts the severe class and never gets it right has
    failed on that class, and a summary should say so rather than evaporate
    into a null.
    """
    denominator = precision + recall
    result: np.ndarray = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    return result


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
    precision, recall, f1
        Per-class values, in class order.
    accuracy
        Fraction correct. Reported because readers expect it, and never without
        the majority-class share beside it.
    balanced_accuracy
        Mean per-class recall. Equivalent to accuracy on a dataset where every
        class were equally common.
    macro_f1
        Unweighted mean of the per-class F1 scores. Treats every class as
        equally important, which is a choice, but a conventional one with no
        free parameters.
    mcc
        Matthews correlation coefficient. Zero means performance
        indistinguishable from guessing at the observed base rates, which makes
        it the most direct answer to "is this better than chance", and the
        aggregate this project ranks by.
    confusion
        Rows actual, columns predicted.
    """

    classes: np.ndarray
    support: np.ndarray
    predicted: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
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

    def recall_for(self, label: int) -> float:
        """Recall for one class, by its label rather than its position."""
        position = int(np.flatnonzero(self.classes == label)[0])
        return float(self.recall[position])

    def precision_for(self, label: int) -> float:
        """Precision for one class, by its label rather than its position."""
        position = int(np.flatnonzero(self.classes == label)[0])
        return float(self.precision[position])

    def per_class(self) -> pd.DataFrame:
        """Return the per-class figures as a table, in class order.

        The form most of this project's reporting takes, because it hides
        nothing and requires the reader to accept no weighting.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "class": self.classes,
                "support": self.support,
                "predicted": self.predicted,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
            }
        )


def classification_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    classes: Sequence[int] | None = None,
) -> ClassificationScores:
    """Score predictions class by class, and in aggregate without weights.

    Parameters
    ----------
    y_true, y_pred
        Actual and predicted labels.
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
        If the inputs differ in length.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"{y_true.shape} actual labels but {y_pred.shape} predicted")

    labels = np.asarray(
        classes if classes is not None else np.union1d(np.unique(y_true), np.unique(y_pred))
    )
    n_classes = len(labels)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)

    precision = np.divide(
        correct, predicted, out=np.zeros(n_classes, dtype=float), where=predicted > 0
    )
    recall = np.divide(correct, support, out=np.zeros(n_classes, dtype=float), where=support > 0)
    f1 = _f1(precision, recall)

    with np.errstate(invalid="ignore"):
        mcc = float(matthews_corrcoef(y_true, y_pred))

    observed = support > 0
    return ClassificationScores(
        classes=labels,
        support=support,
        predicted=predicted,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=float(correct.sum() / matrix.sum()) if matrix.sum() else 0.0,
        balanced_accuracy=float(recall[observed].mean()) if observed.any() else 0.0,
        macro_f1=float(f1[observed].mean()) if observed.any() else 0.0,
        mcc=0.0 if np.isnan(mcc) else mcc,
        confusion=matrix,
    )
