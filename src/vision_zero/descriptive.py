"""Plain description of what is in the data.

Before any question about prediction, there are questions about counts: when do
collisions happen, where, to whom. Most of the answers are unsurprising —
collisions peak in summer, cluster at rush hour, and concentrate on weekdays —
and being unsurprising is the point. Showing that the data reproduces facts
everyone already knows is how one earns the right to be believed about the parts
that are less obvious.

Every function here returns a table with readable labels drawn from the
codebook, so the output is legible without a lookup.

Examples
--------
>>> from vision_zero.descriptive import frequency  # doctest: +SKIP
>>> frequency(frame, "MONTH")  # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from vision_zero.data.codebook import load_codebook

if TYPE_CHECKING:
    from vision_zero.data.codebook import Language

__all__ = ["class_shares", "crosstab_shares", "frequency", "trend"]


def _label(feature: str, value: object, lang: Language) -> str:
    """Render one value using the codebook, falling back to its raw form."""
    book = load_codebook()
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return "not recorded"
    if feature in book:
        return book[feature].short_for(value, lang) or str(value)
    return str(value)


def frequency(
    frame: pd.DataFrame,
    feature: str,
    *,
    lang: Language = "en",
    sort_by_value: bool = True,
) -> pd.DataFrame:
    """Count records by the value of one variable.

    Parameters
    ----------
    frame
        Records.
    feature
        Column to count over.
    lang
        Language for the labels.
    sort_by_value
        Order by the variable's own values rather than by frequency. Correct for
        anything ordered — months, hours, speed limits — where sorting by count
        would scramble the axis.

    Returns
    -------
    pandas.DataFrame
        Value, label, count, and share.
    """
    counts = frame[feature].value_counts(dropna=False)
    if sort_by_value:
        counts = counts.sort_index()

    out = pd.DataFrame(
        {
            "value": counts.index,
            "label": [_label(feature, value, lang) for value in counts.index],
            "n": counts.to_numpy(),
        }
    )
    out["share"] = out["n"] / out["n"].sum()
    return out.reset_index(drop=True)


def class_shares(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    lang: Language = "en",
    min_records: int = 0,
) -> pd.DataFrame:
    """Break the outcome distribution down by the value of one variable.

    The workhorse of the descriptive section: it shows both how common each
    value is and how outcomes differ across values, which is what a reader needs
    in order to judge whether a difference matters.

    Parameters
    ----------
    frame
        Records, including the target column.
    feature
        Column to break down by.
    target
        Outcome column.
    lang
        Language for the labels.
    min_records
        Suppress values with fewer records than this.

    Returns
    -------
    pandas.DataFrame
        One row per value, with counts and the share of each outcome class.
    """
    counts = pd.crosstab(frame[feature], frame[target], dropna=False)
    support = counts.sum(axis=1)
    counts = counts[support >= min_records]
    support = support[support >= min_records]

    shares = counts.div(support, axis=0)
    shares.columns = [f"p_class_{label}" for label in counts.columns]

    out = shares.reset_index()
    out.insert(1, "label", [_label(feature, value, lang) for value in out[feature]])
    out.insert(2, "n", support.to_numpy())
    out.insert(3, "share_of_records", support.to_numpy() / support.sum())
    return out


def crosstab_shares(
    frame: pd.DataFrame,
    row: str,
    column: str,
    *,
    normalise: str = "index",
    lang: Language = "en",
) -> pd.DataFrame:
    """Cross-tabulate two variables, with readable labels on both axes.

    Parameters
    ----------
    frame
        Records.
    row, column
        Variables to cross.
    normalise
        ``"index"`` for row shares, ``"columns"`` for column shares, ``"all"``
        for shares of the whole, ``"none"`` for raw counts.
    lang
        Language for the labels.

    Returns
    -------
    pandas.DataFrame
        Rows and columns labelled from the codebook.
    """
    if normalise == "none":
        table = pd.crosstab(frame[row], frame[column])
    elif normalise == "index":
        table = pd.crosstab(frame[row], frame[column], normalize="index")
    elif normalise == "columns":
        table = pd.crosstab(frame[row], frame[column], normalize="columns")
    else:
        table = pd.crosstab(frame[row], frame[column], normalize="all")
    table.index = pd.Index([_label(row, value, lang) for value in table.index], name=row)
    table.columns = pd.Index([_label(column, value, lang) for value in table.columns], name=column)
    return table


def trend(
    frame: pd.DataFrame,
    period: str,
    *,
    target: str | None = None,
    lang: Language = "en",
) -> pd.DataFrame:
    """Count records over an ordered period, optionally split by outcome.

    Intended for ``MONTH``, ``HOUR``, ``WKDY_WKND``, and ``YEAR``, where the
    values have a natural order that must survive into the output.

    Returns
    -------
    pandas.DataFrame
        One row per period, ordered, with counts and — when ``target`` is
        given — the share of each outcome class.
    """
    if target is None:
        return frequency(frame, period, lang=lang, sort_by_value=True)

    out = class_shares(frame, period, target, lang=lang)
    return out.sort_values(period).reset_index(drop=True)
