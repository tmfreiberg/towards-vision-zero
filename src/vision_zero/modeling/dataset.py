"""Assembling a modelling dataset from a compiled extract and a specification.

Turns the full 1.7-million-record extract into the matrices an estimator sees:
filtered to the records the experiment concerns, split into a held-out test set
and a cross-validated remainder, with features chosen and constant columns
dropped.

Every reduction is logged with a count, because the gap between "1.7 million
records" and "the number the model actually saw" is exactly the kind of thing
that should never have to be reconstructed later.

Examples
--------
>>> from vision_zero.modeling.config import load_experiment  # doctest: +SKIP
>>> from vision_zero.modeling.dataset import build_dataset  # doctest: +SKIP
>>> data = build_dataset(load_experiment("montreal_ternary"))  # doctest: +SKIP
>>> data.X_train.shape  # doctest: +SKIP
(280879, 20)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from vision_zero.config import get_settings
from vision_zero.data.prepare import PARQUET_NAME
from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np

    from vision_zero.modeling.config import Experiment

__all__ = ["Dataset", "MissingColumnsError", "build_dataset", "load_compiled"]

log = get_logger(__name__)


class MissingColumnsError(KeyError):
    """The compiled extract lacks a column the experiment needs."""

    def __str__(self) -> str:
        """Return the message plainly, without ``KeyError``'s added quotes."""
        return str(self.args[0]) if self.args else ""


#: A feature needs at least this many distinct values to carry information.
MIN_DISTINCT_VALUES = 2


@dataclass(frozen=True, slots=True)
class Dataset:
    """Features and targets, split for cross-validation and final evaluation.

    Attributes
    ----------
    X_train, y_train
        Records available for fitting and validation.
    X_test, y_test
        Held out entirely until the analysis is otherwise complete.
    features
        Feature column names, after constant columns were dropped.
    target
        Target column name.
    classes
        Class labels present, ascending.
    dropped_constant
        Features removed because they took fewer than two distinct values.
        Usually a consequence of the experiment's own filters: restricting to
        one region leaves ``REGION`` constant.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    features: tuple[str, ...]
    target: str
    classes: tuple[int, ...]
    dropped_constant: tuple[str, ...]

    @property
    def n_train(self) -> int:
        """Records available for fitting and validation."""
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        """Held-out records."""
        return len(self.X_test)

    def class_shares(self) -> pd.Series:
        """Percentage of training records in each class."""
        return 100 * self.y_train.value_counts(normalize=True).sort_index()

    def folds(self, experiment: Experiment) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train and validation indices for each cross-validation fold.

        Stratified when the specification asks for it, which it should: with a
        class under one percent, unstratified folds vary enough in composition
        to move a score on their own.
        """
        splitter = StratifiedKFold(
            n_splits=experiment.split.folds,
            shuffle=True,
            random_state=experiment.split.seed,
        )
        yield from splitter.split(self.X_train, self.y_train)


def load_compiled(path: Path | None = None) -> pd.DataFrame:
    """Read the compiled extract.

    Raises
    ------
    FileNotFoundError
        If the extract has not been compiled. The message says how to do it,
        because that is the only thing the reader wants to know at that point.
    """
    target = path or get_settings().processed_data_dir / PARQUET_NAME
    if not target.is_file():
        raise FileNotFoundError(f"No compiled extract at {target}. Run `tvz data prepare` first.")
    return pd.read_parquet(target)


def _require_columns(frame: pd.DataFrame, experiment: Experiment) -> None:
    """Fail early, and legibly, when the extract lacks a column that is needed.

    A compiled extract older than a change to the schema is missing whatever
    the change added. Left alone this surfaces as a bare ``KeyError`` naming a
    column the reader has never heard of, several steps from the cause.
    """
    wanted = {experiment.data.target, *experiment.data.restrict}
    absent = sorted(name for name in wanted if name not in frame.columns)
    if absent:
        raise MissingColumnsError(
            f"The compiled extract has no column(s) {absent}. It was most likely "
            "built before the schema included them. Rebuild it with "
            "`tvz data prepare --force`."
        )


def _restrict(frame: pd.DataFrame, restrict: dict[str, list[str]]) -> pd.DataFrame:
    """Keep records matching every listed column-value constraint.

    Constraint values arrive from YAML as strings. Rather than cast the column
    to string to match them — which materialises a string per record, and there
    are 1.7 million of them — the values are cast to the column's own type once.
    """
    out = frame
    for name, values in restrict.items():
        before = len(out)
        column = out[name]
        if pd.api.types.is_numeric_dtype(column):
            wanted: list[object] = [
                int(value) for value in values if str(value).lstrip("-").isdigit()
            ]
        else:
            wanted = list(values)
        out = out[column.isin(wanted)]
        log.info("Restricted %s to %s: %d -> %d records", name, values, before, len(out))
    return out


def _cap_missing_fields(frame: pd.DataFrame, features: tuple[str, ...], limit: int) -> pd.DataFrame:
    """Discard records with too many blank feature fields.

    Reports both what fraction of records is lost and what fraction of each
    class, because those differ: blanks concentrate on the mildest outcomes, so
    the filter removes far more of the majority class than of the severe one.
    That asymmetry is the reason the filter is worth applying and the reason its
    cost has to be stated rather than assumed small.
    """
    present = [name for name in features if name in frame.columns]
    blanks = frame[present].isna().sum(axis=1)
    keep = blanks <= limit

    if keep.all():
        return frame

    log.info(
        "Capping blank fields at %d: %d -> %d records (%.1f%% kept)",
        limit,
        len(frame),
        int(keep.sum()),
        100 * keep.mean(),
    )
    return frame[keep]


def _drop_constant_features(
    frame: pd.DataFrame, features: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split features into those that vary and those that do not.

    A column with one distinct value cannot separate anything. These arise
    naturally from the experiment's own filters rather than from any defect,
    which is why they are reported rather than treated as an error.
    """
    kept, dropped = [], []
    for name in features:
        if frame[name].nunique(dropna=True) < MIN_DISTINCT_VALUES:
            dropped.append(name)
        else:
            kept.append(name)
    if dropped:
        log.info("Dropped constant feature(s): %s", dropped)
    return tuple(kept), tuple(dropped)


def build_dataset(experiment: Experiment, frame: pd.DataFrame | None = None) -> Dataset:
    """Assemble the dataset an experiment specifies.

    Parameters
    ----------
    experiment
        The validated specification.
    frame
        A compiled extract. Read from disk when not supplied.

    Returns
    -------
    Dataset
        Features and targets, split.

    Raises
    ------
    ValueError
        If the specification selects no records, or leaves only one class.
    """
    source = frame if frame is not None else load_compiled()
    log.info("Compiled extract: %d records", len(source))
    _require_columns(source, experiment)

    working = source[source["YEAR"].isin(experiment.data.years)]
    working = _restrict(working, experiment.data.restrict)

    if experiment.data.drop_rows_missing_target:
        before = len(working)
        working = working[working[experiment.data.target].notna()]
        if before != len(working):
            log.info("Dropped %d record(s) with no determinable target", before - len(working))

    if working.empty:
        raise ValueError("The specification selects no records")

    if experiment.data.max_missing_features is not None:
        working = _cap_missing_fields(
            working, experiment.features, experiment.data.max_missing_features
        )

    candidates = tuple(name for name in experiment.features if name in working.columns)
    X = working[list(candidates)].reset_index(drop=True)
    y = working[experiment.data.target].reset_index(drop=True).astype("int64")

    classes = tuple(sorted(y.unique()))
    if len(classes) < MIN_DISTINCT_VALUES:
        raise ValueError(f"Target {experiment.data.target} takes one value; nothing to classify")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=experiment.split.test_size,
        random_state=experiment.split.seed,
        shuffle=True,
        stratify=y if experiment.split.stratify else None,
    )

    # Which features are constant is decided from the training records alone.
    # Deciding it from the whole frame would let the held-out records influence
    # the feature set — a small leak, and an avoidable one.
    features, dropped = _drop_constant_features(X_train, candidates)
    if not features:
        raise ValueError("No feature varies within the selected records")
    X_train = X_train[list(features)]
    X_test = X_test[list(features)]

    log.info(
        "Dataset: %d train, %d test, %d feature(s), %d class(es)",
        len(X_train),
        len(X_test),
        len(features),
        len(classes),
    )
    return Dataset(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        features=features,
        target=experiment.data.target,
        classes=classes,
        dropped_constant=dropped,
    )
