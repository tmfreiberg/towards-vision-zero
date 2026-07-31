"""The argument without the mathematics.

Fano's inequality and the cell-counting bound are two ways of saying one thing,
and the thing is simple enough to see directly: find a group of collisions that
are identical on every recorded variable, and look at what happened to them. If
one killed somebody and another dented a bumper, no procedure that reads only
those variables can tell them apart. It is given the same input twice and must
answer the same way twice.

Multiply that across a dataset and the limit follows. Nothing about model
capacity, training time, or sample size enters into it. The features are not
fine enough to distinguish outcomes that genuinely differ, and no amount of
effort makes a variable record something it does not record.

This module produces those groups so they can be shown rather than described. A
reader who does not want to think about conditional entropy can look at a table
of forty-seven collisions with the same weather, the same road surface, the same
lighting, the same everything, and read off that thirty-eight caused property
damage, eight caused injuries, and one was serious. That is the whole argument.

Examples
--------
>>> import pandas as pd
>>> X = pd.DataFrame({"a": ["x", "x", "x", "y"]})
>>> y = pd.Series([0, 0, 1, 0])
>>> cells = ambiguous_cells(X, y, min_multiplicity=2)
>>> int(cells.iloc[0]["n"])
3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vision_zero.ceiling.bayes_error import cell_labels
from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vision_zero.data.codebook import Codebook, Language

__all__ = ["ambiguous_cells", "describe_cell", "worst_case_pairs"]

log = get_logger(__name__)


def _entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    """Return the entropy in bits of each row of a count matrix."""
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        proportions = np.where(totals > 0, counts / totals, 0.0)
        terms = np.where(proportions > 0, -proportions * np.log2(proportions), 0.0)
    return terms.sum(axis=1)


def ambiguous_cells(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    min_multiplicity: int = 10,
    top_n: int = 20,
    cells: pd.Series | None = None,
) -> pd.DataFrame:
    """Find groups of identical records whose outcomes disagree.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Columns forming the combination. All by default.
    min_multiplicity
        Records a group needs before it is reported. The default is high on
        purpose: a group of two that disagrees is unremarkable, and the exhibit
        is only convincing when the groups are large enough that disagreement
        cannot be dismissed as coincidence.
    top_n
        How many groups to return.
    cells
        Precomputed cell identifiers, to avoid regrouping a large frame.

    Returns
    -------
    pandas.DataFrame
        One row per group, most striking first, with the record count, the count
        and share of each class, the entropy in bits, and the largest class
        share — which is the fraction a perfect classifier would get right on
        that group, and therefore the local ceiling.

    Notes
    -----
    Ranked by ``n * entropy``: a group is interesting when it is both large and
    genuinely mixed. Ranking by entropy alone surfaces tiny even splits; ranking
    by size alone surfaces large unanimous groups. Neither makes the point.
    """
    identifiers = cell_labels(X, features) if cells is None else cells
    labels = np.unique(np.asarray(y))

    frame = pd.DataFrame({"cell": identifiers.to_numpy(), "y": np.asarray(y)})
    tallied = frame.groupby(["cell", "y"], observed=True).size()
    counts = tallied.unstack(fill_value=0).reindex(columns=labels, fill_value=0)  # noqa: PD010

    totals = counts.sum(axis=1)
    eligible = counts[totals >= min_multiplicity]
    if eligible.empty:
        return pd.DataFrame()

    matrix = eligible.to_numpy(dtype=float)
    sizes = matrix.sum(axis=1)
    entropy = _entropy_from_counts(matrix)

    summary = pd.DataFrame({"cell": eligible.index, "n": sizes.astype(int)})
    for position, label in enumerate(labels):
        summary[f"n_class_{label}"] = matrix[:, position].astype(int)
        summary[f"p_class_{label}"] = matrix[:, position] / sizes
    summary["entropy_bits"] = entropy
    summary["max_share"] = matrix.max(axis=1) / sizes
    summary["classes_present"] = (matrix > 0).sum(axis=1)
    summary["interest"] = sizes * entropy

    ranked = summary.sort_values("interest", ascending=False).head(top_n)
    return ranked.reset_index(drop=True)


def describe_cell(
    X: pd.DataFrame,
    cell_id: int,
    *,
    features: Sequence[str] | None = None,
    cells: pd.Series | None = None,
    codebook: Codebook | None = None,
    lang: Language = "en",
) -> dict[str, str]:
    """Return the feature values shared by every record in a group.

    Parameters
    ----------
    X
        Feature frame.
    cell_id
        Identifier from :func:`~vision_zero.ceiling.bayes_error.cell_labels`.
    features
        Columns forming the combination.
    cells
        Precomputed cell identifiers.
    codebook
        Used to render values as readable labels. Raw values when omitted.
    lang
        Language for the labels.

    Returns
    -------
    dict
        Feature name to its value in this group, as a string. Unrecorded values
        render as "not recorded", which is itself part of what the records share.
    """
    identifiers = cell_labels(X, features) if cells is None else cells
    members = X[identifiers.to_numpy() == cell_id]
    if members.empty:
        return {}

    columns = list(features) if features is not None else list(X.columns)
    row = members.iloc[0]

    described: dict[str, str] = {}
    for name in columns:
        value = row[name]
        if pd.isna(value):
            described[name] = "not recorded"
        elif codebook is not None and name in codebook:
            described[name] = codebook[name].short_for(value, lang) or str(value)
        else:
            described[name] = str(value)
    return described


def worst_case_pairs(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    severe_class: int | None = None,
    min_multiplicity: int = 10,
    top_n: int = 10,
    cells: pd.Series | None = None,
) -> pd.DataFrame:
    """Find groups containing both the mildest and the severest outcome.

    The sharpest form of the exhibit. A group holding a collision that killed or
    seriously injured somebody alongside collisions that hurt nobody, with every
    recorded variable identical between them, is a case where the recorded
    circumstances demonstrably do not determine the outcome — and where a
    classifier must be wrong about one of them.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Columns forming the combination.
    severe_class
        Which label counts as severe. The largest by default.
    min_multiplicity
        Records a group needs.
    top_n
        How many groups to return.
    cells
        Precomputed cell identifiers.

    Returns
    -------
    pandas.DataFrame
        Groups holding both extremes, largest first, with the counts of each.
    """
    labels = np.unique(np.asarray(y))
    severe = int(labels.max()) if severe_class is None else severe_class
    mild = int(labels.min())

    summary = ambiguous_cells(
        X,
        y,
        features=features,
        min_multiplicity=min_multiplicity,
        top_n=len(X),
        cells=cells,
    )
    if summary.empty:
        return summary

    both = summary[(summary[f"n_class_{severe}"] > 0) & (summary[f"n_class_{mild}"] > 0)]
    return both.sort_values("n", ascending=False).head(top_n).reset_index(drop=True)
