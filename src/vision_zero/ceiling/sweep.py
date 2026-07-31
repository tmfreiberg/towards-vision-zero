"""Showing that the limit belongs to the data, not to the estimator.

A weak result invites two explanations: the wrong model was used, or there is
nothing to find. Distinguishing them is the difference between a project that
failed and a project with a finding, and neither the cell-counting bound nor
Fano's inequality settles it on its own — both concern the data, so a sceptic
may reasonably ask whether they have been computed correctly.

The direct evidence is comparative.

:func:`model_sweep` fits several estimator families under identical
preprocessing, splits, and scoring. Families with very different inductive
biases — a linear model, a single tree, a boosted ensemble, a neural network —
landing within a rounding error of each other is difficult to explain by
mis-specification. Capacity is not the binding constraint when adding capacity
changes nothing.

:func:`learning_curve` refits on growing fractions of the training records. A
curve still climbing at full size means more data would help; a curve flat from
a quarter onward means it would not, and the shortage is of information rather
than of records.

Both are expensive — every point is a full cross-validation — so both take a
fold count and a subsample size, and both default to something smaller than the
headline experiment uses.

Examples
--------
>>> from vision_zero.ceiling.sweep import model_sweep  # doctest: +SKIP
>>> model_sweep(experiment, dataset, folds=3)  # doctest: +SKIP
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vision_zero.evaluation.metrics import classification_scores
from vision_zero.logging import get_logger
from vision_zero.modeling.baselines import Baseline, baseline_predictions
from vision_zero.modeling.pipeline import build_pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vision_zero.modeling.config import Experiment
    from vision_zero.modeling.dataset import Dataset

__all__ = ["DEFAULT_FAMILIES", "learning_curve", "model_sweep"]

log = get_logger(__name__)

#: Families spanning a wide range of inductive bias. Parameters are modest
#: throughout: the question is whether capacity is binding, and answering it
#: does not require any of them to be tuned.
DEFAULT_FAMILIES: dict[str, dict[str, object]] = {
    "logistic": {"max_iter": 200, "solver": "saga"},
    "decision_tree": {"max_depth": 8},
    "random_forest": {"n_estimators": 50, "max_depth": 12, "n_jobs": -1},
    "xgboost": {"n_estimators": 100, "max_depth": 6, "tree_method": "hist"},
    "mlp": {"hidden_layer_sizes": (32,), "max_iter": 30, "early_stopping": True},
}


def _score_fold(
    experiment: Experiment,
    dataset: Dataset,
    kind: str,
    params: dict[str, object],
    train_index: np.ndarray,
    validation_index: np.ndarray,
    features: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Fit one family on one fold and return its scores.

    ``features`` restricts the fit to a subset, which is what makes the block
    ablation possible: everything else about the comparison is held identical.
    """
    columns = features if features is not None else dataset.features
    spec = experiment.model_copy(
        update={"model": experiment.model.model_copy(update={"kind": kind, "params": params})}
    )
    pipeline = build_pipeline(spec, columns)

    started = time.perf_counter()
    pipeline.fit(
        dataset.X_train.iloc[train_index][list(columns)], dataset.y_train.iloc[train_index]
    )
    elapsed = time.perf_counter() - started

    probabilities = pipeline.predict_proba(dataset.X_train.iloc[validation_index][list(columns)])
    classes = np.asarray(dataset.classes)
    predictions = classes[np.argmax(probabilities, axis=1)]
    truth = dataset.y_train.iloc[validation_index].to_numpy()

    scores = classification_scores(
        truth,
        predictions,
        beta=experiment.evaluation.beta,
        weights=experiment.evaluation.weights,
        classes=list(dataset.classes),
    )
    return {
        "accuracy": scores.accuracy,
        "balanced_accuracy": scores.balanced_accuracy,
        "f_star": scores.f_star,
        "mcc": scores.mcc,
        "severe_recall": float(scores.recall[-1]),
        "fit_seconds": elapsed,
    }


def model_sweep(
    experiment: Experiment,
    dataset: Dataset,
    *,
    families: dict[str, dict[str, object]] | None = None,
    folds: int = 3,
    subsample: int | None = 60_000,
) -> pd.DataFrame:
    """Score several estimator families under identical conditions.

    Parameters
    ----------
    experiment
        Supplies the split, the scoring rule, and everything else held fixed.
    dataset
        Assembled features and targets.
    families
        Estimator kind to parameters. Defaults to :data:`DEFAULT_FAMILIES`.
    folds
        Cross-validation folds. Fewer than the headline experiment, because the
        comparison needs a common footing rather than a precise figure.
    subsample
        Cap on training records, for tractability. ``None`` uses all of them.
        Sampling is stratified, so the class balance the problem turns on is
        preserved.

    Returns
    -------
    pandas.DataFrame
        One row per family, with the mean and standard deviation of each score
        across folds, plus the majority-class baseline for comparison.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    families = families or DEFAULT_FAMILIES
    X, y = dataset.X_train, dataset.y_train

    if subsample is not None and len(X) > subsample:
        X, _, y, _ = train_test_split(
            X, y, train_size=subsample, random_state=experiment.split.seed, stratify=y
        )
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        log.info("Sweep on a stratified subsample of %d record(s)", len(X))

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=experiment.split.seed)
    indices = list(splitter.split(X, y))
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

    rows = []
    for kind, params in families.items():
        per_fold = []
        for train_index, validation_index in indices:
            per_fold.append(
                _score_fold(experiment, reduced, kind, params, train_index, validation_index)
            )
        frame = pd.DataFrame(per_fold)
        row: dict[str, object] = {"family": kind}
        for column in frame.columns:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_std"] = float(frame[column].std(ddof=1)) if folds > 1 else 0.0
        rows.append(row)
        log.info(
            "%s: F* %.4f, MCC %.4f, severe recall %.4f",
            kind,
            row["f_star_mean"],
            row["mcc_mean"],
            row["severe_recall_mean"],
        )

    rows.append(_baseline_row(experiment, reduced, indices))
    return pd.DataFrame(rows).sort_values("f_star_mean", ascending=False).reset_index(drop=True)


def _baseline_row(
    experiment: Experiment,
    dataset: Dataset,
    indices: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    """Score the majority-class baseline on the same folds."""
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
                "accuracy": scores.accuracy,
                "balanced_accuracy": scores.balanced_accuracy,
                "f_star": scores.f_star,
                "mcc": scores.mcc,
                "severe_recall": float(scores.recall[-1]),
                "fit_seconds": 0.0,
            }
        )
    frame = pd.DataFrame(per_fold)
    row: dict[str, object] = {"family": "majority baseline"}
    for column in frame.columns:
        row[f"{column}_mean"] = float(frame[column].mean())
        row[f"{column}_std"] = float(frame[column].std(ddof=1)) if len(frame) > 1 else 0.0
    return row


def learning_curve(
    experiment: Experiment,
    dataset: Dataset,
    *,
    fractions: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
    folds: int = 3,
    subsample: int | None = 60_000,
) -> pd.DataFrame:
    """Score the experiment's estimator on growing fractions of the training data.

    Parameters
    ----------
    experiment
        Supplies the estimator, split, and scoring rule.
    dataset
        Assembled features and targets.
    fractions
        Training-set fractions to evaluate. Sampling is stratified at each size.
    folds
        Cross-validation folds per point.
    subsample
        Cap on training records before the fractions are taken.

    Returns
    -------
    pandas.DataFrame
        One row per fraction, with training size and mean scores.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    X, y = dataset.X_train, dataset.y_train
    if subsample is not None and len(X) > subsample:
        X, _, y, _ = train_test_split(
            X, y, train_size=subsample, random_state=experiment.split.seed, stratify=y
        )
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=experiment.split.seed)
    rows = []
    for fraction in fractions:
        per_fold = []
        for train_index, validation_index in splitter.split(X, y):
            if fraction < 1.0:
                subset, _ = train_test_split(
                    train_index,
                    train_size=fraction,
                    random_state=experiment.split.seed,
                    stratify=y.iloc[train_index],
                )
            else:
                subset = train_index

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
            per_fold.append(
                _score_fold(
                    experiment,
                    reduced,
                    experiment.model.kind,
                    dict(experiment.model.params),
                    np.asarray(subset),
                    validation_index,
                )
            )
        frame = pd.DataFrame(per_fold)
        rows.append(
            {
                "fraction": fraction,
                "n_train": int(fraction * len(X) * (folds - 1) / folds),
                "f_star": float(frame["f_star"].mean()),
                "accuracy": float(frame["accuracy"].mean()),
                "mcc": float(frame["mcc"].mean()),
                "severe_recall": float(frame["severe_recall"].mean()),
            }
        )
        log.info("Fraction %.2f: F* %.4f", fraction, rows[-1]["f_star"])

    return pd.DataFrame(rows)
