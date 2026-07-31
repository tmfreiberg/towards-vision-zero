"""How well any classifier could possibly do with these features.

Suppose two collisions are recorded identically — same weather, same road
surface, same lighting, same everything the report captures — and one produced a
fatality while the other produced a dented bumper. No classifier can be right
about both. It sees one input and must give one answer.

Count how often that happens and you have an upper bound on achievable
accuracy that depends on the data alone. No amount of tuning, no larger model,
and no extra training records can exceed it, because the limit is not in the
estimator.

Two estimators are provided, and the difference between them matters.

:func:`in_sample_ceiling` groups the records by their feature vector and asks,
within each group, how many could be got right by answering with that group's
most common outcome. It is optimistic, badly so when groups are small: a feature
vector occurring exactly once contributes a perfect score, having been fitted to
itself. Reported alone it would be close to meaningless on sparse data.

:func:`held_out_ceiling` fixes that. It learns each group's most common outcome
from training records only, then applies it to validation records, scoring only
those whose feature vector was seen during training. That is the accuracy of the
best possible classifier of this form, measured honestly on records it did not
learn from, and it is the number the argument rests on.

Examples
--------
>>> import pandas as pd
>>> X = pd.DataFrame({"a": ["x", "x", "y", "y"]})
>>> y = pd.Series([0, 0, 0, 1])
>>> round(in_sample_ceiling(X, y).ceiling, 3)
0.75
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CeilingEstimate",
    "HeldOutCeiling",
    "cell_labels",
    "held_out_ceiling",
    "in_sample_ceiling",
    "multiplicity_profile",
]

log = get_logger(__name__)


def cell_labels(X: pd.DataFrame, features: Sequence[str] | None = None) -> pd.Series:
    """Assign each record an identifier for its exact feature combination.

    Records with unrecorded values in the same fields share a cell. This is the
    generous choice: treating each null as unique would fragment the data into
    singletons and drive the ceiling toward one for reasons that have nothing to
    do with predictability.

    Parameters
    ----------
    X
        Feature frame.
    features
        Columns forming the combination. All columns by default.

    Returns
    -------
    pandas.Series
        Integer cell identifier per record, aligned with ``X``.

    Notes
    -----
    Grouping is done with ``groupby(...).ngroup()`` rather than by building a
    string key per row. The two agree, but on a few hundred thousand records
    across twenty columns the row-wise version takes roughly thirty seconds and
    this takes under one — which matters, because several estimators here need
    the same grouping and would otherwise each pay for it.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({"a": [1, 1, 2], "b": ["p", "p", "q"]})
    >>> sorted(set(cell_labels(X).tolist()))
    [0, 1]
    >>> cell_labels(X).iloc[0] == cell_labels(X).iloc[1]
    True
    """
    columns = list(features) if features is not None else list(X.columns)
    grouped = X.groupby(columns, observed=True, dropna=False, sort=False).ngroup()
    return pd.Series(grouped.to_numpy(), index=X.index, name="cell")


@dataclass(frozen=True, slots=True)
class CeilingEstimate:
    """An upper bound on achievable accuracy, with the caveats it needs.

    Attributes
    ----------
    ceiling
        Largest fraction of records any classifier of these features could get
        right, over the records this estimate covers. ``nan`` when the estimate
        covers nothing, which is a different statement from zero: one says the
        data cannot answer, the other says nothing is predictable.
    covered
        Records the estimate covers.
    total
        Records considered.
    cells
        Distinct feature combinations covered.
    min_multiplicity
        Combinations rarer than this were excluded.
    ambiguous_cells
        Covered combinations carrying more than one outcome. These are the whole
        phenomenon: a combination with one outcome poses no difficulty.
    ambiguous_records
        Records living in those combinations.

    Notes
    -----
    ``ceiling`` is an upper bound on accuracy, not an estimate of it, and it is
    optimistic in proportion to how many small cells it covers.
    """

    ceiling: float
    covered: int
    total: int
    cells: int
    min_multiplicity: int
    ambiguous_cells: int
    ambiguous_records: int

    @property
    def coverage(self) -> float:
        """Fraction of records the estimate speaks about."""
        return self.covered / self.total if self.total else 0.0

    @property
    def ambiguous_share(self) -> float:
        """Fraction of covered records in combinations with more than one outcome."""
        return self.ambiguous_records / self.covered if self.covered else 0.0


def in_sample_ceiling(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    min_multiplicity: int = 1,
    cells: pd.Series | None = None,
) -> CeilingEstimate:
    """Bound achievable accuracy from within-cell label agreement.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Columns forming the combination. All by default.
    min_multiplicity
        Ignore combinations occurring fewer than this many times. Raising it
        removes the cells that flatter the estimate, at the cost of covering
        fewer records; the trade-off is what :func:`multiplicity_profile`
        exists to show.

    Returns
    -------
    CeilingEstimate
        The bound, with its coverage and ambiguity counts.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` differ in length.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

    identifiers = cell_labels(X, features) if cells is None else cells
    frame = pd.DataFrame({"cell": identifiers.to_numpy(), "y": np.asarray(y)})

    counts = frame.groupby(["cell", "y"], observed=True).size()
    per_cell = counts.groupby(level="cell").agg(["sum", "max", "size"])
    per_cell.columns = ["records", "largest", "outcomes"]

    eligible = per_cell[per_cell["records"] >= min_multiplicity]
    covered = int(eligible["records"].sum())
    ambiguous = eligible[eligible["outcomes"] > 1]

    return CeilingEstimate(
        ceiling=float(eligible["largest"].sum() / covered) if covered else float("nan"),
        covered=covered,
        total=len(frame),
        cells=len(eligible),
        min_multiplicity=min_multiplicity,
        ambiguous_cells=len(ambiguous),
        ambiguous_records=int(ambiguous["records"].sum()),
    )


@dataclass(frozen=True, slots=True)
class HeldOutCeiling:
    """Accuracy of the best possible classifier of these features, measured honestly.

    The classifier in question answers, for each feature combination, whichever
    outcome was most common among training records sharing it. No classifier
    restricted to these features can do better in expectation, so its held-out
    accuracy estimates the ceiling without the self-fitting that inflates the
    in-sample figure.

    Attributes
    ----------
    accuracy
        Fraction correct among covered validation records. ``nan`` when no
        validation record's combination appeared in training, which says the
        feature space is too sparse for this estimate rather than that accuracy
        is zero.
    covered
        Validation records whose combination appeared in training.
    total
        Validation records.
    per_class_recall
        Recall by class, over covered records. The severe class is the one to
        read: if even this classifier cannot find it, no model will.
    predictions
        The classifier's prediction per covered record.
    truth
        Actual label per covered record.
    majority_baseline
        Accuracy, over the same covered records, of always answering with the
        commonest training class.
    """

    accuracy: float
    covered: int
    total: int
    per_class_recall: dict[int, float]
    predictions: np.ndarray
    truth: np.ndarray
    majority_baseline: float = float("nan")

    @property
    def beats_baseline(self) -> bool:
        """Whether the estimate exceeds always answering with the commonest class.

        When it does not, the estimate is not a ceiling. The cell-majority rule
        is Bayes-optimal only in the limit of many records per cell; where cells
        hold barely more than one record each, a cell's majority is close to a
        single random draw and the rule is fitting noise. It then scores below a
        constant predictor, which no genuine upper bound can do.

        A false result means the data is too sparse for this estimator, not that
        the problem is unusually hard.
        """
        if np.isnan(self.accuracy) or np.isnan(self.majority_baseline):
            return False
        return bool(self.accuracy > self.majority_baseline)

    @property
    def coverage(self) -> float:
        """Fraction of validation records whose combination was seen in training.

        Low coverage is itself a finding. It says the feature space is sparse
        enough that most new records are unlike anything already recorded, which
        limits what any method can generalise from.
        """
        return self.covered / self.total if self.total else 0.0


def held_out_ceiling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    features: Sequence[str] | None = None,
    min_multiplicity: int = 1,
) -> HeldOutCeiling:
    """Estimate the ceiling by fitting cell majorities on train and scoring on validation.

    Parameters
    ----------
    X_train, y_train
        Records from which each combination's most common outcome is learned.
    X_val, y_val
        Records on which the resulting rule is scored.
    features
        Columns forming the combination.
    min_multiplicity
        Require a combination to appear at least this often in training before
        its majority is trusted. One is the natural default and already honest,
        since the validation record is not among the training records.

    Returns
    -------
    HeldOutCeiling
        Accuracy, coverage, and per-class recall.
    """
    columns = list(features) if features is not None else list(X_train.columns)
    combined = pd.concat([X_train[columns], X_val[columns]], ignore_index=True)
    all_cells = cell_labels(combined, columns)

    train_cells = all_cells.iloc[: len(X_train)].to_numpy()
    val_cells = all_cells.iloc[len(X_train) :].to_numpy()

    frame = pd.DataFrame({"cell": train_cells, "y": np.asarray(y_train)})
    counts = frame.groupby(["cell", "y"], observed=True).size()

    totals = counts.groupby(level="cell").sum()
    eligible = set(totals[totals >= min_multiplicity].index)
    majority = counts.groupby(level="cell").idxmax()
    lookup = {cell: int(pair[1]) for cell, pair in majority.items() if cell in eligible}

    seen = np.array([cell in lookup for cell in val_cells])
    truth = np.asarray(y_val)[seen]
    predictions = np.array([lookup[cell] for cell in val_cells[seen]], dtype=int)

    recall: dict[int, float] = {}
    for label in np.unique(np.asarray(y_val)):
        mask = truth == label
        recall[int(label)] = (
            float((predictions[mask] == label).mean()) if mask.any() else float("nan")
        )

    accuracy = float((predictions == truth).mean()) if len(truth) else float("nan")

    train_labels, train_counts = np.unique(np.asarray(y_train), return_counts=True)
    commonest = train_labels[int(np.argmax(train_counts))]
    baseline = float((truth == commonest).mean()) if len(truth) else float("nan")

    if len(truth) and accuracy <= baseline:
        log.warning(
            "Held-out ceiling %.4f does not beat the majority baseline %.4f. "
            "Cells are too sparse for this estimator; treat it as uninformative.",
            accuracy,
            baseline,
        )

    log.info(
        "Held-out ceiling: %.4f accuracy over %d of %d record(s) (%.1f%% covered)",
        accuracy,
        len(truth),
        len(y_val),
        100 * len(truth) / len(y_val) if len(y_val) else 0.0,
    )
    return HeldOutCeiling(
        accuracy=accuracy,
        covered=len(truth),
        total=len(y_val),
        per_class_recall=recall,
        predictions=predictions,
        truth=truth,
        majority_baseline=baseline,
    )


def multiplicity_profile(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    thresholds: Sequence[int] = (1, 2, 5, 10, 20, 50),
) -> pd.DataFrame:
    """Tabulate the ceiling against the multiplicity cut-off.

    The in-sample bound is optimistic where cells are small, so the honest way
    to report it is as a curve rather than a number: raising the cut-off tightens
    the bound and narrows its coverage at the same time, and a reader should see
    both moving together.

    Returns
    -------
    pandas.DataFrame
        One row per threshold, with the ceiling, coverage, cell count, and the
        share of covered records in cells carrying more than one outcome.
    """
    identifiers = cell_labels(X, features)
    rows = []
    for threshold in thresholds:
        estimate = in_sample_ceiling(X, y, min_multiplicity=threshold, cells=identifiers)
        rows.append(
            {
                "min_multiplicity": threshold,
                "ceiling": estimate.ceiling,
                "coverage": estimate.coverage,
                "cells": estimate.cells,
                "covered_records": estimate.covered,
                "ambiguous_cells": estimate.ambiguous_cells,
                "ambiguous_share": estimate.ambiguous_share,
            }
        )
    return pd.DataFrame(rows)
