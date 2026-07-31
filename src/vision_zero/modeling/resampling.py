"""Upsampling minority classes.

With the severe class under one percent, an estimator minimising overall loss
can do well by never predicting it. Repeating minority records raises their
weight in the loss without touching the estimator's objective.

This is applied to a training fold only, never before splitting. Resampling
first would place copies of the same record on both sides of a split, so a model
would be evaluated partly on records it had memorised — inflating every score
while explaining nothing.

Upsampling does not create information. It changes the loss surface, which can
change what an estimator predicts, but the number of distinct minority records
is unchanged. Where a ceiling comes from the features rather than from the
optimiser, resampling moves the operating point along that ceiling and does not
raise it.

Examples
--------
>>> import numpy as np, pandas as pd
>>> X = pd.DataFrame({"a": range(10)})
>>> y = pd.Series([0] * 9 + [1])
>>> Xr, yr = upsample(X, y, max_repeats=5, seed=0)
>>> int((yr == 1).sum())
5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vision_zero.logging import get_logger

__all__ = ["class_counts", "upsample"]

log = get_logger(__name__)


def class_counts(y: pd.Series) -> dict[int, int]:
    """Return records per class, ascending by label."""
    counts = y.value_counts().sort_index()
    return {int(label): int(counts.loc[label]) for label in counts.index}


def upsample(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_repeats: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Repeat minority records, up to a bound.

    Each class is raised toward the size of the largest, but no class is
    repeated more than ``max_repeats`` times. The bound matters: without it, a
    class at one percent would be repeated a hundredfold, and the fitted model
    would be dominated by a few thousand distinct records seen over and over.

    Parameters
    ----------
    X
        Training features.
    y
        Training labels, aligned with ``X``.
    max_repeats
        Largest multiple of its original size any class may reach. One returns
        the inputs untouched.
    seed
        Seed for the shuffle and for the partial-repeat sample.

    Returns
    -------
    tuple
        Resampled features and labels, shuffled, with a fresh index.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` differ in length.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    if max_repeats <= 1:
        return X, y

    rng = np.random.default_rng(seed)
    counts = class_counts(y)
    largest = max(counts.values())

    positions: list[np.ndarray] = []
    for label, count in counts.items():
        target = min(largest, count * max_repeats)
        index = np.flatnonzero(np.asarray(y) == label)

        whole, remainder = divmod(target, count)
        positions.append(np.repeat(index, whole))
        if remainder:
            # Sample the shortfall without replacement, so that within a class
            # every record is used the same number of times give or take one.
            positions.append(rng.choice(index, size=remainder, replace=False))

    combined = np.concatenate(positions)
    rng.shuffle(combined)

    resampled_y = y.iloc[combined].reset_index(drop=True)
    log.debug("Upsampled %s -> %s", counts, class_counts(resampled_y))
    return X.iloc[combined].reset_index(drop=True), resampled_y
