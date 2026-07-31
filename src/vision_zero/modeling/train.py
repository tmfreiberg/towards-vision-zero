"""Fitting an experiment.

Cross-validates the specified pipeline over the training records, keeping the
out-of-fold predictions, and fits one final pipeline on everything for use
against the held-out test set.

Training produces predictions; it does not score them. Keeping the two apart
means a scoring rule can be reconsidered — and the analysis argues about
scoring rules at some length — without refitting anything, and it keeps a
module that should be deterministic free of judgements about what counts as
good.

Baseline predictions are computed on the same folds, from the same training
labels, at the same time. A fold score without its baseline invites the reader
to supply an intuition about what is good, and intuitions about accuracy on
imbalanced data are reliably wrong.

Examples
--------
>>> from vision_zero.modeling.train import train  # doctest: +SKIP
>>> run = train(experiment, dataset)  # doctest: +SKIP
>>> len(run.folds)  # doctest: +SKIP
5
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vision_zero.logging import get_logger
from vision_zero.modeling.baselines import all_baselines
from vision_zero.modeling.pipeline import build_pipeline
from vision_zero.modeling.resampling import class_counts, upsample

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from vision_zero.modeling.baselines import BaselinePrediction
    from vision_zero.modeling.config import Experiment
    from vision_zero.modeling.dataset import Dataset

__all__ = ["FoldResult", "TrainingRun", "fit_final", "train"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Out-of-fold predictions for one cross-validation fold.

    Attributes
    ----------
    fold
        Zero-based fold index.
    validation_index
        Positions within the training set that formed this validation fold.
    y_true, y_pred
        Actual and predicted labels for the validation records.
    probabilities
        Predicted class probabilities, columns in ascending class order.
    classes
        Class labels, ascending.
    baselines
        Trivial predictions over the same validation records.
    importances
        Feature importance by name, where the estimator exposes it.
    fit_seconds
        Wall-clock time to fit.
    train_counts
        Records per class in the training fold, after any resampling.
    """

    fold: int
    validation_index: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray
    baselines: dict[str, BaselinePrediction]
    importances: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    train_counts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Everything one fitting of an experiment produced.

    Attributes
    ----------
    experiment
        The specification that produced this run.
    folds
        Per-fold out-of-fold predictions.
    pipeline
        Pipeline refitted on all training records, for the held-out test set.
    features
        Feature columns used, in order.
    classes
        Class labels, ascending.
    """

    experiment: Experiment
    folds: tuple[FoldResult, ...]
    pipeline: Pipeline
    features: tuple[str, ...]
    classes: tuple[int, ...]

    def out_of_fold(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate the folds into one out-of-fold prediction set.

        Every training record appears exactly once, predicted by a pipeline that
        did not see it. This is the right basis for an overall figure: averaging
        per-fold scores weights small folds equally with large ones, and hides
        variance that matters when a class has only a few hundred members.

        Returns
        -------
        tuple
            Actual labels, predicted labels, and probabilities, ordered by the
            record's position in the training set.
        """
        order = np.concatenate([fold.validation_index for fold in self.folds])
        y_true = np.concatenate([fold.y_true for fold in self.folds])
        y_pred = np.concatenate([fold.y_pred for fold in self.folds])
        probabilities = np.vstack([fold.probabilities for fold in self.folds])

        restore = np.argsort(order)
        return y_true[restore], y_pred[restore], probabilities[restore]

    def mean_importances(self) -> pd.Series:
        """Average feature importance across folds, descending.

        Empty when the estimator exposes no importances.
        """
        frames = [pd.Series(fold.importances) for fold in self.folds if fold.importances]
        if not frames:
            return pd.Series(dtype=float)
        return pd.concat(frames, axis=1).mean(axis=1).sort_values(ascending=False)


def _importances(pipeline: Pipeline, features: tuple[str, ...]) -> dict[str, float]:
    """Extract per-feature importance, aggregating one-hot columns.

    A one-hot encoded variable becomes many columns, so its importance is the
    sum over them. Without that, a variable with twenty categories looks
    unimportant purely by being spread thin.
    """
    estimator = pipeline.named_steps["estimate"]
    raw = getattr(estimator, "feature_importances_", None)
    if raw is None:
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return {}
        raw = np.abs(np.asarray(coefficients)).mean(axis=0)

    try:
        encoded = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    except (AttributeError, ValueError):
        return {}

    totals = dict.fromkeys(features, 0.0)
    for name, value in zip(encoded, np.asarray(raw, dtype=float), strict=False):
        for feature in features:
            if name == feature or name.startswith(f"{feature}_"):
                totals[feature] += float(value)
                break
    return totals


def train(experiment: Experiment, dataset: Dataset) -> TrainingRun:
    """Cross-validate the experiment and refit on all training records.

    Parameters
    ----------
    experiment
        The validated specification.
    dataset
        Assembled features and targets.

    Returns
    -------
    TrainingRun
        Per-fold predictions, baselines, and the final pipeline.
    """
    classes = np.asarray(dataset.classes)
    results: list[FoldResult] = []

    for index, (train_positions, validation_positions) in enumerate(dataset.folds(experiment)):
        X_fold = dataset.X_train.iloc[train_positions]
        y_fold = dataset.y_train.iloc[train_positions]
        X_validation = dataset.X_train.iloc[validation_positions]
        y_validation = dataset.y_train.iloc[validation_positions]

        # Resampling applies to the training fold only. Applied earlier, copies
        # of one record would land on both sides of the split.
        X_fitted, y_fitted = upsample(
            X_fold,
            y_fold,
            max_repeats=experiment.resampling.max_repeats,
            seed=experiment.split.seed + index,
        )

        pipeline = build_pipeline(experiment, dataset.features)
        started = time.perf_counter()
        pipeline.fit(X_fitted, y_fitted)
        elapsed = time.perf_counter() - started

        probabilities = pipeline.predict_proba(X_validation)
        predictions = classes[np.argmax(probabilities, axis=1)]

        results.append(
            FoldResult(
                fold=index,
                validation_index=np.asarray(validation_positions),
                y_true=np.asarray(y_validation),
                y_pred=predictions,
                probabilities=probabilities,
                classes=classes,
                baselines=all_baselines(
                    np.asarray(y_fitted),
                    len(y_validation),
                    seed=experiment.split.seed + index,
                    classes=classes,
                ),
                importances=_importances(pipeline, dataset.features),
                fit_seconds=elapsed,
                train_counts=class_counts(y_fitted),
            )
        )
        log.info(
            "Fold %d: fitted %d record(s) in %.1fs, validated on %d",
            index,
            len(y_fitted),
            elapsed,
            len(y_validation),
        )

    final = fit_final(experiment, dataset)
    return TrainingRun(
        experiment=experiment,
        folds=tuple(results),
        pipeline=final,
        features=dataset.features,
        classes=dataset.classes,
    )


def fit_final(experiment: Experiment, dataset: Dataset) -> Pipeline:
    """Fit one pipeline on all training records, for the held-out test set."""
    X, y = upsample(
        dataset.X_train,
        dataset.y_train,
        max_repeats=experiment.resampling.max_repeats,
        seed=experiment.split.seed,
    )
    pipeline = build_pipeline(experiment, dataset.features)
    started = time.perf_counter()
    pipeline.fit(X, y)
    log.info("Final fit on %d record(s) in %.1fs", len(y), time.perf_counter() - started)
    return pipeline


def predict(pipeline: Pipeline, X: pd.DataFrame, classes: tuple[int, ...]) -> dict[str, Any]:
    """Predict labels and probabilities with an already-fitted pipeline."""
    probabilities = pipeline.predict_proba(X)
    labels = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    return {"y_pred": labels, "probabilities": probabilities}
