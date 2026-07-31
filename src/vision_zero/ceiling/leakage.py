"""Finding variables that answer the question instead of informing it.

A feature leaks when knowing it settles the outcome by definition rather than by
evidence. The distinction is not always visible in a variable's name, and on
this dataset one variable crossed it without announcing itself: ``PED`` records
a pedestrian *victim*, where the other road-user indicators record a vehicle
*involved*. Nothing in the column name says so.

The reliable way to find such variables is not to read names but to measure,
and the measurement has to be the right one. The obvious test — does some value
always produce the same outcome — is too weak. ``PED = Y`` does not produce a
single outcome: it splits between minor and serious injury. What it does is
make "material damage only" *impossible*, and an impossibility is just as much
a definition as a certainty.

So two things are checked. A value **determines** an outcome when it is
followed by one class and no other. A value **excludes** an outcome when some
class that should have appeared hundreds of times never appears at all. The
second catches strictly more, and it is the one that catches victim indicators.

What this cannot settle is the difference between *entailment* and *very strong
association*. A collision with a pedestrian is not by definition an injury, but
it very often is, and a value that predicts an outcome 97% of the time is
legitimate signal while one that predicts it 100% of the time is a definition
wearing a disguise. This module reports the number and leaves the reading to
whoever knows what the variable means.

One further caution it cannot measure at all. A variable can behave like an
outcome because of how records come to exist rather than because of what it
means: if collisions with unharmed pedestrians are seldom reported, then
``ACCDN_TYPE = pedestrian`` will look near-deterministic through selection, not
definition. That is a question about the collection process, and no
within-dataset statistic can answer it.

Examples
--------
>>> import pandas as pd
>>> X = pd.DataFrame({"leaky": ["y", "y", "n", "n"], "ordinary": ["a", "b", "a", "b"]})
>>> y = pd.Series([1, 1, 0, 1])
>>> flagged = deterministic_values(X, y, min_support=2)
>>> flagged["feature"].tolist()
['leaky']
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DETERMINISTIC_THRESHOLD",
    "MIN_EXPECTED_COUNT",
    "conditional_class_distribution",
    "deterministic_values",
    "excluded_outcomes",
    "leakage_report",
]

log = get_logger(__name__)

#: A value whose outcome is this predictable is treated as suspect. Set at one
#: rather than slightly below deliberately: the object is to separate entailment
#: from strong association, and anything short of certainty is association.
DETERMINISTIC_THRESHOLD = 1.0

#: Values seen fewer times than this are ignored. A value occurring twice and
#: followed twice by the same outcome is unremarkable.
MIN_SUPPORT = 30

#: How many records of a class must have been expected, under independence,
#: before its complete absence counts as evidence rather than as chance. Ten
#: expected and none observed has probability well under one in ten thousand.
MIN_EXPECTED_COUNT = 10.0


def conditional_class_distribution(X: pd.DataFrame, y: pd.Series, feature: str) -> pd.DataFrame:
    """Return the outcome distribution for each value of one feature.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    feature
        Column to break down.

    Returns
    -------
    pandas.DataFrame
        One row per observed value, with the record count, the share of each
        class, and the largest of those shares.
    """
    frame = pd.DataFrame({"value": X[feature].astype("string"), "y": np.asarray(y)})
    counts = pd.crosstab(frame["value"], frame["y"], dropna=False)
    support = counts.sum(axis=1)
    shares = counts.div(support, axis=0)

    out = shares.copy()
    out.columns = [f"p_class_{label}" for label in shares.columns]
    out.insert(0, "n", support)
    out["max_share"] = shares.max(axis=1)
    out["modal_class"] = shares.idxmax(axis=1)
    return out.reset_index().rename(columns={"value": "value"})


def deterministic_values(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    threshold: float = DETERMINISTIC_THRESHOLD,
    min_support: int = MIN_SUPPORT,
) -> pd.DataFrame:
    """Find feature values that are followed by a single outcome.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Columns to check. All by default.
    threshold
        Share of a single class at or above which a value is flagged.
    min_support
        Records a value needs before it is considered.

    Returns
    -------
    pandas.DataFrame
        Flagged values, most-supported first. Empty is the reassuring result.
    """
    columns = list(features) if features is not None else list(X.columns)
    rows = []
    for name in columns:
        table = conditional_class_distribution(X, y, name)
        table = table[table["n"] >= min_support]
        flagged = table[table["max_share"] >= threshold - 1e-12]
        for _, row in flagged.iterrows():
            rows.append(
                {
                    "feature": name,
                    "value": row["value"],
                    "n": int(row["n"]),
                    "share_of_records": float(row["n"] / len(X)),
                    "modal_class": row["modal_class"],
                    "max_share": float(row["max_share"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["feature", "value", "n", "share_of_records", "modal_class", "max_share"]
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def excluded_outcomes(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    min_support: int = MIN_SUPPORT,
    min_expected: float = MIN_EXPECTED_COUNT,
) -> pd.DataFrame:
    """Find feature values that make some outcome impossible.

    The more sensitive of the two checks, and the one that catches a variable
    recording an outcome under a participant's name. A value need not fix the
    class to give the game away; ruling a class out is enough.

    A class is reported as excluded when it never occurs alongside the value,
    although the overall class frequency says it should have occurred at least
    ``min_expected`` times. Seeing none of an expected ten has probability under
    one in ten thousand, so the absence is structural rather than incidental.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Columns to check. All by default.
    min_support
        Records a value needs before it is considered.
    min_expected
        Expected count of the missing class, below which absence proves nothing.

    Returns
    -------
    pandas.DataFrame
        One row per excluded (value, class) pair, most surprising first.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> y = pd.Series([0] * 800 + [1] * 190 + [2] * 10)
    >>> X = pd.DataFrame({"victim": np.where(y.to_numpy() > 0, "Y", "N")})
    >>> excluded_outcomes(X, y)["excluded_class"].tolist()
    [0]
    """
    columns = list(features) if features is not None else list(X.columns)
    labels = np.unique(np.asarray(y))
    priors = {int(label): float((np.asarray(y) == label).mean()) for label in labels}

    rows = []
    for name in columns:
        table = conditional_class_distribution(X, y, name)
        table = table[table["n"] >= min_support]
        for _, row in table.iterrows():
            support = int(row["n"])
            for label in labels:
                share = float(row[f"p_class_{label}"])
                expected = support * priors[int(label)]
                if share == 0.0 and expected >= min_expected:
                    rows.append(
                        {
                            "feature": name,
                            "value": row["value"],
                            "excluded_class": int(label),
                            "n": support,
                            "expected": expected,
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["feature", "value", "excluded_class", "n", "expected"])
    return pd.DataFrame(rows).sort_values("expected", ascending=False).reset_index(drop=True)


def leakage_report(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    min_support: int = MIN_SUPPORT,
) -> pd.DataFrame:
    """Rank features by how nearly their values settle the outcome.

    Where :func:`deterministic_values` asks whether any value is certain, this
    asks how close each feature comes. It is the diagnostic to read when the
    strict test finds nothing but a feature still looks too good: a variable
    whose worst value still predicts one class 97% of the time deserves an
    explanation, even though it entails nothing.

    Returns
    -------
    pandas.DataFrame
        One row per feature: the number of values, the highest single-class
        share reached by any adequately supported value, and the records
        involved.
    """
    columns = list(features) if features is not None else list(X.columns)
    rows = []
    for name in columns:
        table = conditional_class_distribution(X, y, name)
        eligible = table[table["n"] >= min_support]
        if eligible.empty:
            continue
        position = int(np.argmax(eligible["max_share"].to_numpy()))
        peak = eligible.iloc[position]
        rows.append(
            {
                "feature": name,
                "values": len(eligible),
                "max_share": float(peak["max_share"].item()),
                "at_value": peak["value"],
                "n_at_value": int(peak["n"].item()),
                "modal_class": peak["modal_class"],
            }
        )
    return pd.DataFrame(rows).sort_values("max_share", ascending=False).reset_index(drop=True)
