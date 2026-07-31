"""Whether a field was filled in, and what that reveals.

Police reports vary in how completely they are filled out, and the variation is
not random. A collision that hurt somebody gets a thorough report; one that bent
a bumper gets a cursory one. So the blankness of a field carries information
about severity — and the information is about the reporting process, not about
the road.

This matters for three reasons.

It is a form of leakage, though a subtler one than a mislabelled variable. A
model that learns "sparse report implies minor collision" has learned something
true and useless: nobody designing an intersection knows how thoroughly a future
report will be completed.

It inflates every estimate in this package that treats a blank as a value. The
ceiling estimators group records by exact feature combination, and two records
both missing the same fields are counted as identical — which is the right
choice for measuring predictability from the data as recorded, and the wrong one
for measuring predictability from the circumstances themselves.

And it is worth reporting for its own sake. That report completeness tracks
severity is a finding about how administrative data is produced, and it applies
well beyond this dataset.

Examples
--------
>>> import numpy as np, pandas as pd
>>> X = pd.DataFrame({"f": [None] * 50 + ["a"] * 50})
>>> y = pd.Series([0] * 50 + [0] * 25 + [1] * 25)
>>> report = missingness_report(X, y)
>>> float(report.loc[0, "lift_class_0"]) > 1
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "missing_count_profile",
    "missingness_indicators",
    "missingness_report",
]

log = get_logger(__name__)


def missingness_indicators(X: pd.DataFrame, features: Sequence[str] | None = None) -> pd.DataFrame:
    """Return one boolean column per feature, recording whether it was left blank.

    Useful as a feature set in its own right. A model given nothing but these
    indicators cannot know anything about the collision, only about the report —
    so whatever it achieves is a measure of how much the paperwork gives away.
    """
    columns = list(features) if features is not None else list(X.columns)
    return pd.DataFrame({f"{name}_missing": X[name].isna() for name in columns}, index=X.index)


def missingness_report(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    min_missing: int = 30,
) -> pd.DataFrame:
    """Compare the outcome distribution for blank against filled fields.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Columns to examine. All by default.
    min_missing
        Blank records a feature needs before it is reported.

    Returns
    -------
    pandas.DataFrame
        One row per feature with any blanks: how many, and for each class the
        share among blank records, the share among filled ones, and the ratio
        between them. A lift far from one means blankness is informative.
    """
    columns = list(features) if features is not None else list(X.columns)
    labels = np.unique(np.asarray(y))
    values = np.asarray(y)

    rows = []
    for name in columns:
        blank = X[name].isna().to_numpy()
        if blank.sum() < min_missing or blank.all():
            continue

        row: dict[str, object] = {
            "feature": name,
            "n_missing": int(blank.sum()),
            "share_missing": float(blank.mean()),
        }
        for label in labels:
            among_blank = float((values[blank] == label).mean())
            among_filled = float((values[~blank] == label).mean())
            row[f"p_class_{label}_missing"] = among_blank
            row[f"p_class_{label}_present"] = among_filled
            row[f"lift_class_{label}"] = (
                among_blank / among_filled if among_filled > 0 else float("nan")
            )
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["feature", "n_missing", "share_missing"])

    frame = pd.DataFrame(rows)
    severe = f"lift_class_{labels[-1]}"
    return (
        frame.sort_values("n_missing", ascending=False)
        .reset_index(drop=True)
        .assign(**{severe: frame.sort_values("n_missing", ascending=False)[severe].to_numpy()})
    )


def missing_count_profile(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Tabulate the outcome distribution against how many fields were left blank.

    The single clearest view of the effect. If severity falls monotonically as
    the number of blanks rises, then report completeness is tracking severity,
    and any model with access to the blanks is partly reading the thoroughness
    of the paperwork.

    Returns
    -------
    pandas.DataFrame
        One row per count of blank fields, with the record count and the share
        of each class.
    """
    columns = list(features) if features is not None else list(X.columns)
    blanks = X[columns].isna().sum(axis=1)
    frame = pd.DataFrame({"n_missing": blanks.to_numpy(), "y": np.asarray(y)})

    counts = pd.crosstab(frame["n_missing"], frame["y"])
    support = counts.sum(axis=1)
    shares = counts.div(support, axis=0)
    shares.columns = [f"p_class_{label}" for label in counts.columns]

    out = shares.copy()
    out.insert(0, "n_records", support)
    out.insert(1, "share_of_records", support / len(frame))
    return out.reset_index()
