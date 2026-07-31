"""What each kind of feature adds, once the others are known.

A model on this data reaches an MCC near 0.44, and almost all of it comes from
one fact: collisions involving pedestrians, cyclists, and motorcyclists tend to
injure someone. That is true, and it is not news. Presented as a headline it
would be close to dishonest.

The question with content is what remains. Once it is known who was involved,
does knowing the road configuration, the lighting, the surface condition, or the
posted speed limit improve the answer? Those are the variables a municipality
can act on. If they add nothing measurable, that is a finding about what this
data can support — and a sharper one than any statement about a model's
accuracy.

Two views are offered, and they answer different questions.

:func:`cumulative_value` adds blocks one at a time in a stated order and reports
what each addition buys. It reads as a narrative: here is what participants
alone achieve, here is what geometry adds to that, and so on.

:func:`block_contributions` removes one block at a time from the full set. This
measures each block's *unique* contribution — what is lost that no other block
supplies. Where two blocks carry the same information, the cumulative view
credits whichever comes first and the leave-one-out view credits neither, and
the gap between the two readings is itself informative.

Examples
--------
>>> from vision_zero.ceiling.increments import cumulative_value  # doctest: +SKIP
>>> cumulative_value(experiment, dataset)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vision_zero.data.schema import FEATURE_BLOCKS
from vision_zero.evaluation.metrics import classification_scores
from vision_zero.logging import get_logger
from vision_zero.modeling.baselines import Baseline, baseline_predictions
from vision_zero.modeling.pipeline import build_pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vision_zero.modeling.config import Experiment
    from vision_zero.modeling.dataset import Dataset

__all__ = ["DEFAULT_ORDER", "block_contributions", "cumulative_value"]

log = get_logger(__name__)

#: Blocks are added least-interesting first, so that the levers a municipality
#: actually holds are credited only with what they add beyond the obvious.
#: Ordering matters wherever blocks overlap, which is why
#: :func:`block_contributions` exists alongside this.
DEFAULT_ORDER: tuple[str, ...] = ("participants", "geometry", "conditions", "timing")


def _score_subset(
    experiment: Experiment,
    dataset: Dataset,
    features: tuple[str, ...],
    folds: int,
) -> dict[str, float]:
    """Cross-validate the experiment's estimator on a subset of the features."""
    from sklearn.model_selection import StratifiedKFold

    if not features:
        return _baseline_scores(experiment, dataset, folds)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=experiment.split.seed)
    classes = np.asarray(dataset.classes)
    collected = []

    for train_index, validation_index in splitter.split(dataset.X_train, dataset.y_train):
        pipeline = build_pipeline(experiment, features)
        pipeline.fit(
            dataset.X_train.iloc[train_index][list(features)],
            dataset.y_train.iloc[train_index],
        )
        probabilities = pipeline.predict_proba(
            dataset.X_train.iloc[validation_index][list(features)]
        )
        predictions = classes[np.argmax(probabilities, axis=1)]
        scores = classification_scores(
            dataset.y_train.iloc[validation_index].to_numpy(),
            predictions,
            beta=experiment.evaluation.beta,
            weights=experiment.evaluation.weights,
            classes=list(dataset.classes),
        )
        collected.append(
            {
                "f_star": scores.f_star,
                "mcc": scores.mcc,
                "accuracy": scores.accuracy,
                "balanced_accuracy": scores.balanced_accuracy,
                "severe_recall": float(scores.recall[-1]),
            }
        )

    frame = pd.DataFrame(collected)
    return {column: float(frame[column].mean()) for column in frame.columns}


def _baseline_scores(experiment: Experiment, dataset: Dataset, folds: int) -> dict[str, float]:
    """Score the majority-class rule, which is what no features at all buys."""
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=experiment.split.seed)
    collected = []
    for train_index, validation_index in splitter.split(dataset.X_train, dataset.y_train):
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
        collected.append(
            {
                "f_star": scores.f_star,
                "mcc": scores.mcc,
                "accuracy": scores.accuracy,
                "balanced_accuracy": scores.balanced_accuracy,
                "severe_recall": float(scores.recall[-1]),
            }
        )
    frame = pd.DataFrame(collected)
    return {column: float(frame[column].mean()) for column in frame.columns}


def _available(dataset: Dataset, block: str) -> tuple[str, ...]:
    """Members of a block that survived dataset assembly."""
    return tuple(name for name in dataset.features if name in FEATURE_BLOCKS[block])


def cumulative_value(
    experiment: Experiment,
    dataset: Dataset,
    *,
    order: Sequence[str] = DEFAULT_ORDER,
    folds: int = 3,
    subsample: int | None = 60_000,
) -> pd.DataFrame:
    """Add feature blocks in order and report what each one buys.

    Parameters
    ----------
    experiment
        Supplies the estimator and the scoring rule.
    dataset
        Assembled features and targets.
    order
        Blocks, in the order they are added. The default puts the obvious
        first, so that everything else is credited only with what it adds.
    folds
        Cross-validation folds per step.
    subsample
        Cap on training records, stratified.

    Returns
    -------
    pandas.DataFrame
        One row per step, with the scores and the change from the step before.
    """
    from sklearn.model_selection import train_test_split

    working = dataset
    if subsample is not None and len(dataset.X_train) > subsample:
        X, _, y, _ = train_test_split(
            dataset.X_train,
            dataset.y_train,
            train_size=subsample,
            random_state=experiment.split.seed,
            stratify=dataset.y_train,
        )
        working = type(dataset)(
            X_train=X.reset_index(drop=True),
            y_train=y.reset_index(drop=True),
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            features=dataset.features,
            target=dataset.target,
            classes=dataset.classes,
            dropped_constant=dataset.dropped_constant,
        )

    rows = []
    accumulated: tuple[str, ...] = ()
    previous: dict[str, float] | None = None

    for step, block in enumerate(["none", *order]):
        if block != "none":
            accumulated = accumulated + _available(working, block)
            if not accumulated:
                continue

        scores = _score_subset(experiment, working, accumulated, folds)
        row: dict[str, object] = {
            "step": step,
            "added": block,
            "n_features": len(accumulated),
            **scores,
        }
        if previous is not None:
            for metric in ("f_star", "mcc", "accuracy", "severe_recall"):
                row[f"delta_{metric}"] = scores[metric] - previous[metric]
        rows.append(row)
        previous = scores
        log.info(
            "+%s (%d features): F* %.4f, MCC %.4f",
            block,
            len(accumulated),
            scores["f_star"],
            scores["mcc"],
        )

    return pd.DataFrame(rows)


def block_contributions(
    experiment: Experiment,
    dataset: Dataset,
    *,
    blocks: Sequence[str] = DEFAULT_ORDER,
    folds: int = 3,
    subsample: int | None = 60_000,
) -> pd.DataFrame:
    """Remove one block at a time and report what is lost.

    Measures each block's unique contribution: what no other block can supply in
    its absence. A block that reads as valuable cumulatively but worthless here
    is duplicating information another block already carries.

    Returns
    -------
    pandas.DataFrame
        One row per block, with the full-model score, the score without it, and
        the loss. Negative loss means the block was doing nothing and its
        removal happened to help.
    """
    from sklearn.model_selection import train_test_split

    working = dataset
    if subsample is not None and len(dataset.X_train) > subsample:
        X, _, y, _ = train_test_split(
            dataset.X_train,
            dataset.y_train,
            train_size=subsample,
            random_state=experiment.split.seed,
            stratify=dataset.y_train,
        )
        working = type(dataset)(
            X_train=X.reset_index(drop=True),
            y_train=y.reset_index(drop=True),
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            features=dataset.features,
            target=dataset.target,
            classes=dataset.classes,
            dropped_constant=dataset.dropped_constant,
        )

    full = _score_subset(experiment, working, working.features, folds)
    rows = []
    for block in blocks:
        members = set(_available(working, block))
        if not members:
            continue
        remaining = tuple(name for name in working.features if name not in members)
        without = _score_subset(experiment, working, remaining, folds)
        rows.append(
            {
                "block": block,
                "n_removed": len(members),
                "f_star_full": full["f_star"],
                "f_star_without": without["f_star"],
                "f_star_loss": full["f_star"] - without["f_star"],
                "mcc_full": full["mcc"],
                "mcc_without": without["mcc"],
                "mcc_loss": full["mcc"] - without["mcc"],
            }
        )
        log.info(
            "Without %s: F* %.4f (loss %.4f)", block, without["f_star"], rows[-1]["f_star_loss"]
        )

    return pd.DataFrame(rows).sort_values("mcc_loss", ascending=False).reset_index(drop=True)
