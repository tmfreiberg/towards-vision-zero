"""What each group of variables contributes, over and above the others.

A single score for a whole feature set answers the wrong question. Road-safety
work does not ask whether severity can be predicted; it asks whether anything
under anyone's control affects it. Those come apart here, because the variables
that predict — who was involved — are precisely the ones nobody chooses, and the
variables an engineer could change are the ones that predict least.

Separating the blocks makes the distinction measurable. Three views, each
answering something different:

``alone``
    What can this block do by itself? Establishes each block's ceiling in
    isolation, and shows how much of the total one block already reaches.
``added``
    What does this block add to everything else? The direct answer to "would
    knowing the road layout help, given that we already know a pedestrian was
    struck". Computed by fitting on everything and again on everything but this
    block; the difference is the contribution.
``cumulative``
    What happens as blocks are added in a stated order? Useful for a narrative
    that starts from the unavoidable and works toward the actionable.

Every comparison holds the estimator, the folds, the preprocessing, and the
scoring rule fixed. Only the columns change.

Examples
--------
>>> from vision_zero.ceiling.ablation import block_ablation  # doctest: +SKIP
>>> block_ablation(experiment, dataset)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from vision_zero.ceiling.sweep import _score_fold
from vision_zero.data.schema import blocks_for
from vision_zero.evaluation.metrics import classification_scores
from vision_zero.logging import get_logger
from vision_zero.modeling.baselines import Baseline, baseline_predictions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vision_zero.modeling.config import Experiment
    from vision_zero.modeling.dataset import Dataset

__all__ = ["block_ablation", "cumulative_ablation"]

log = get_logger(__name__)

SCORE_COLUMNS = ("f_star", "mcc", "accuracy", "balanced_accuracy", "severe_recall")


def _prepare(
    experiment: Experiment, dataset: Dataset, subsample: int | None, folds: int
) -> tuple[Dataset, list[tuple[np.ndarray, np.ndarray]]]:
    """Subsample if asked, and fix the folds every variant will share."""
    X, y = dataset.X_train, dataset.y_train
    if subsample is not None and len(X) > subsample:
        X, _, y, _ = train_test_split(
            X, y, train_size=subsample, random_state=experiment.split.seed, stratify=y
        )
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        log.info("Ablation on a stratified subsample of %d record(s)", len(X))

    reduced = type(dataset)(
        X_train=X,
        y_train=y,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        features=dataset.features,
        target=dataset.target,
        classes=dataset.classes,
        dropped_constant=dataset.dropped_constant,
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=experiment.split.seed)
    return reduced, list(splitter.split(X, y))


def _score_features(
    experiment: Experiment,
    dataset: Dataset,
    indices: list[tuple[np.ndarray, np.ndarray]],
    features: tuple[str, ...],
) -> dict[str, float]:
    """Cross-validate one feature subset and average across folds."""
    per_fold = [
        _score_fold(
            experiment,
            dataset,
            experiment.model.kind,
            dict(experiment.model.params),
            train_index,
            validation_index,
            features=features,
        )
        for train_index, validation_index in indices
    ]
    frame = pd.DataFrame(per_fold)
    return {column: float(frame[column].mean()) for column in SCORE_COLUMNS}


def _baseline(
    experiment: Experiment,
    dataset: Dataset,
    indices: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Score the majority-class rule on the same folds, as the floor."""
    per_fold = []
    for train_index, validation_index in indices:
        truth = dataset.y_train.iloc[validation_index].to_numpy()
        prediction = baseline_predictions(
            Baseline.MAJORITY,
            dataset.y_train.iloc[train_index].to_numpy(),
            n=len(truth),
            classes=np.asarray(dataset.classes),
        )
        scores = classification_scores(
            truth,
            prediction.labels,
            beta=experiment.evaluation.beta,
            weights=experiment.evaluation.weights,
            classes=list(dataset.classes),
        )
        per_fold.append(
            {
                "f_star": scores.f_star,
                "mcc": scores.mcc,
                "accuracy": scores.accuracy,
                "balanced_accuracy": scores.balanced_accuracy,
                "severe_recall": float(scores.recall[-1]),
            }
        )
    frame = pd.DataFrame(per_fold)
    return {column: float(frame[column].mean()) for column in SCORE_COLUMNS}


def block_ablation(
    experiment: Experiment,
    dataset: Dataset,
    *,
    folds: int = 3,
    subsample: int | None = 60_000,
) -> pd.DataFrame:
    """Score each block alone, and measure what each adds to the rest.

    Parameters
    ----------
    experiment
        Supplies the estimator, the split, and the scoring rule.
    dataset
        Assembled features and targets.
    folds
        Cross-validation folds. Every variant uses the same ones.
    subsample
        Cap on training records, applied once before the folds are drawn so that
        all variants see identical data.

    Returns
    -------
    pandas.DataFrame
        One row per block plus rows for the majority baseline and the full
        feature set. ``f_star_alone`` is the block by itself; ``f_star_added``
        is the full set minus the full set without this block, so a value near
        zero means the block contributes nothing the others do not already
        carry.
    """
    reduced, indices = _prepare(experiment, dataset, subsample, folds)
    blocks = blocks_for(dataset.features)

    baseline = _baseline(experiment, reduced, indices)
    everything = _score_features(experiment, reduced, indices, dataset.features)

    rows: list[dict[str, object]] = [
        {
            "block": "majority baseline",
            "n_features": 0,
            **{f"{k}_alone": v for k, v in baseline.items()},
        },
        {
            "block": "all features",
            "n_features": len(dataset.features),
            **{f"{k}_alone": v for k, v in everything.items()},
        },
    ]

    for name, members in blocks.items():
        alone = _score_features(experiment, reduced, indices, members)
        without = tuple(f for f in dataset.features if f not in members)
        rest = _score_features(experiment, reduced, indices, without) if without else baseline
        row: dict[str, object] = {"block": name, "n_features": len(members)}
        for column in SCORE_COLUMNS:
            row[f"{column}_alone"] = alone[column]
            row[f"{column}_added"] = everything[column] - rest[column]
        rows.append(row)
        log.info(
            "%s: F* alone %.4f, added %.4f",
            name,
            row["f_star_alone"],
            row["f_star_added"],
        )

    return pd.DataFrame(rows)


def cumulative_ablation(
    experiment: Experiment,
    dataset: Dataset,
    *,
    order: Sequence[str] = ("geography", "participants", "temporal", "environment"),
    folds: int = 3,
    subsample: int | None = 60_000,
) -> pd.DataFrame:
    """Score blocks added one at a time, in a stated order.

    The default order runs from least actionable to most: where it happened, who
    was involved, when, and finally the road itself. The last step is therefore
    the one the project exists to measure — what the built environment adds once
    everything outside anyone's control is already known.

    Returns
    -------
    pandas.DataFrame
        One row per step, with the running feature count, the scores, and the
        change in each score from the previous step.
    """
    reduced, indices = _prepare(experiment, dataset, subsample, folds)
    blocks = blocks_for(dataset.features)

    rows = [
        {"step": "majority baseline", "n_features": 0, **_baseline(experiment, reduced, indices)}
    ]

    accumulated: tuple[str, ...] = ()
    for name in order:
        if name not in blocks:
            continue
        accumulated = (*accumulated, *blocks[name])
        scores = _score_features(experiment, reduced, indices, accumulated)
        rows.append({"step": f"+ {name}", "n_features": len(accumulated), **scores})
        log.info("+ %s (%d features): F* %.4f", name, len(accumulated), scores["f_star"])

    frame = pd.DataFrame(rows)
    for column in SCORE_COLUMNS:
        frame[f"{column}_delta"] = frame[column].diff()
    return frame
