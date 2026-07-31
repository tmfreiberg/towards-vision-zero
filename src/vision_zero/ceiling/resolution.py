"""How sharply the features resolve the outcome, as the match is relaxed.

Requiring every recorded variable to agree produces small groups, and small
groups are both the strongest form of the argument and its weakest: strongest
because the records really are indistinguishable, weakest because most
collisions have no such group to belong to. Only 44% of held-out records share
their exact combination with anything in training.

Relaxing the match fixes the second problem and costs the first. Match on
seventeen of eighteen variables and the groups grow; match on ten and they grow
much more; match on none and there is one group holding everything, whose
outcome distribution is the overall base rate by definition.

So there is a trade, and it can be measured. For each number of variables
matched on, this module reports how large the groups become, how much their
outcome distributions still differ from the base rate, and how much of the
label entropy is thereby explained. The curve those quantities trace is the
honest summary of what the features are worth: not a single ceiling, but a
statement about resolution against coverage.

What the curve cannot do is surprise anyone about its endpoints. Matching on
everything gives maximum resolution and minimum coverage; matching on nothing
gives the reverse. The question is the shape between them — whether the
information falls away gently, which would mean many variables each contribute
a little, or collapses once a few particular variables are dropped, which would
mean the rest were never doing much.

Examples
--------
>>> import pandas as pd
>>> X = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"]})
>>> y = pd.Series([0, 0, 1, 1])
>>> float(slice_dispersion(X, y, ("a",))["information_bits"]) > 0.9
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vision_zero.ceiling.bayes_error import cell_labels
from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "drop_one_profile",
    "resolution_profile",
    "single_feature_profile",
    "slice_dispersion",
]

log = get_logger(__name__)


def _cell_counts(X: pd.DataFrame, y: pd.Series, features: Sequence[str]) -> np.ndarray:
    """Return a matrix of class counts, one row per distinct feature combination."""
    if not features:
        labels = np.unique(np.asarray(y))
        return np.array([[(np.asarray(y) == label).sum() for label in labels]], dtype=float)

    identifiers = cell_labels(X, features)
    frame = pd.DataFrame({"cell": identifiers.to_numpy(), "y": np.asarray(y)})
    tallied = frame.groupby(["cell", "y"], observed=True).size()
    matrix = tallied.unstack(fill_value=0)  # noqa: PD010
    return matrix.to_numpy(dtype=float)


def _held_out_information(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    *,
    seed: int,
    alpha: float,
) -> float:
    """Estimate the information the features carry, without self-fitting.

    Splits the records in half, learns each group's outcome distribution from
    one half, and scores the other half against it. A group absent from the
    first half falls back to the base rate, which is the honest thing to do:
    having never seen the combination, one knows nothing beyond the overall
    frequencies.

    Returned as label entropy less the resulting cross entropy, so a set of
    variables carrying nothing scores about zero, and one carrying only noise
    can score *below* zero — estimated distributions worse than the base rate is
    exactly what overfitting looks like from outside.

    This replaces the plug-in estimate wherever variable sets are compared. The
    plug-in figure rises as variables are added whether or not they mean
    anything, because more variables make smaller groups and a group of one
    shows no uncertainty. On a test with two informative variables among ten,
    the plug-in ranking placed all eight noise variables above both signals.
    """
    labels = np.unique(np.asarray(y))
    n_classes = len(labels)
    position = {label: index for index, label in enumerate(labels)}
    values = np.asarray(y)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    first, second = order[: len(order) // 2], order[len(order) // 2 :]
    if len(first) == 0 or len(second) == 0:
        return 0.0

    identifiers = (
        cell_labels(X, features).to_numpy() if features else np.zeros(len(y), dtype=np.int64)
    )

    fitted = pd.DataFrame({"cell": identifiers[first], "y": values[first]})
    tallied = fitted.groupby(["cell", "y"], observed=True).size()
    counts = tallied.unstack(fill_value=0).reindex(columns=labels, fill_value=0)  # noqa: PD010
    smoothed = (counts + alpha).div(counts.sum(axis=1) + alpha * n_classes, axis=0)

    base = np.bincount([position[label] for label in values[first]], minlength=n_classes).astype(
        float
    )
    base = (base + alpha) / (base.sum() + alpha * n_classes)

    observed = np.array([position[label] for label in values[second]])
    seen = np.isin(identifiers[second], smoothed.index)

    probabilities = np.empty(len(second), dtype=float)
    if seen.any():
        rows = smoothed.loc[identifiers[second][seen]].to_numpy()
        probabilities[seen] = rows[np.arange(int(seen.sum())), observed[seen]]
    probabilities[~seen] = base[observed[~seen]]

    cross_entropy = float(-np.log2(probabilities).mean())
    base_entropy = float(-(base[base > 0] * np.log2(base[base > 0])).sum())
    return base_entropy - cross_entropy


def slice_dispersion(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    *,
    seed: int = 42,
    alpha: float = 0.5,
) -> pd.Series:
    """Summarise what matching on one set of variables buys.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Variables the match is required to agree on. Empty means no match at
        all, which yields one group and the base rate.
    seed
        Seed for the half-split used by the honest information estimate.
    alpha
        Additive smoothing on the fitted group distributions.

    Returns
    -------
    pandas.Series
        ``n_features``, ``n_cells``, ``mean_cell_size``, ``median_cell_size``,
        ``singleton_share`` (records alone in their group), ``local_ceiling``
        (weighted mean of each group's largest class share),
        ``mean_abs_deviation`` (weighted mean absolute distance between a
        group's outcome shares and the base rate), ``information_bits`` (the
        honest estimate, from a half-split), and ``information_bits_plugin``
        (the naive one, retained for comparison).

    Notes
    -----
    Read ``information_bits``. ``information_bits_plugin`` rises as variables
    are added whether or not they mean anything, because more variables make
    smaller groups and a group of one shows no uncertainty; ``local_ceiling``
    and ``mean_abs_deviation`` inherit the same bias and should likewise be read
    against ``singleton_share`` rather than on their own.
    """
    counts = _cell_counts(X, y, features)
    sizes = counts.sum(axis=1)
    total = sizes.sum()

    base_rate = counts.sum(axis=0) / total
    shares = counts / sizes[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shares > 0, -shares * np.log2(shares), 0.0)
    group_entropy = terms.sum(axis=1)

    base_entropy = float(-(base_rate[base_rate > 0] * np.log2(base_rate[base_rate > 0])).sum())
    weighted_entropy = float((group_entropy * sizes).sum() / total)

    deviation = np.abs(shares - base_rate).sum(axis=1) / 2

    return pd.Series(
        {
            "n_features": len(features),
            "n_cells": len(counts),
            "mean_cell_size": float(sizes.mean()),
            "median_cell_size": float(np.median(sizes)),
            "singleton_share": float(sizes[sizes == 1].sum() / total),
            "local_ceiling": float((counts.max(axis=1)).sum() / total),
            "mean_abs_deviation": float((deviation * sizes).sum() / total),
            "information_bits": _held_out_information(X, y, features, seed=seed, alpha=alpha),
            "information_bits_plugin": base_entropy - weighted_entropy,
            "base_entropy_bits": base_entropy,
        }
    )


def resolution_profile(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    sizes: Sequence[int] | None = None,
    n_samples: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Trace resolution against coverage as fewer variables are matched on.

    For each requested number of variables, several random subsets of that size
    are evaluated and their summaries averaged. Averaging over subsets matters:
    a single subset of five variables might happen to contain the two that carry
    everything, or might miss both, and either would misrepresent what five
    variables are generally worth.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Variables to draw subsets from. All columns by default.
    sizes
        Subset sizes to evaluate. Defaults to a spread from one variable up to
        all of them.
    n_samples
        Random subsets per size. The full set and the empty set have only one
        subset each and are evaluated once regardless.
    seed
        Seed for the subset draws.

    Returns
    -------
    pandas.DataFrame
        One row per size, with the mean of each summary across subsets and the
        spread of the information estimate.
    """
    columns = list(features) if features is not None else list(X.columns)
    if sizes is None:
        span = sorted({0, 1, 2, 3, 5, 8, 12, len(columns) - 1, len(columns)})
        sizes = [size for size in span if 0 <= size <= len(columns)]

    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        if size in (0, len(columns)):
            draws = [tuple(columns[:size])]
        else:
            draws = [tuple(rng.choice(columns, size=size, replace=False)) for _ in range(n_samples)]

        summaries = pd.DataFrame([slice_dispersion(X, y, subset) for subset in draws])
        row = summaries.mean().to_dict()
        row["n_features"] = size
        row["n_subsets"] = len(draws)
        row["information_bits_std"] = (
            float(summaries["information_bits"].std(ddof=1)) if len(draws) > 1 else 0.0
        )
        rows.append(row)
        log.info(
            "%2d feature(s): %.3f bits, mean group %.1f records, %.1f%% singletons",
            size,
            row["information_bits"],
            row["mean_cell_size"],
            100 * row["singleton_share"],
        )

    return pd.DataFrame(rows).sort_values("n_features").reset_index(drop=True)


def single_feature_profile(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure what each variable is worth on its own.

    The robust counterpart to :func:`drop_one_profile`, and the one to reach for
    first. Matching on a single variable produces groups large enough that the
    estimate is stable, so the ranking means what it appears to mean.

    It answers a narrower question — what a variable carries by itself, ignoring
    overlap with the others — but it answers it reliably, which
    :func:`drop_one_profile` cannot promise once the full variable set is large
    enough to overfit.

    Returns
    -------
    pandas.DataFrame
        One row per variable, most informative first.
    """
    columns = list(features) if features is not None else list(X.columns)
    rows = []
    for name in columns:
        summary = slice_dispersion(X, y, (name,), seed=seed)
        rows.append(
            {
                "feature": name,
                "information_bits": summary["information_bits"],
                "n_values": summary["n_cells"],
                "mean_group_size": summary["mean_cell_size"],
                "mean_abs_deviation": summary["mean_abs_deviation"],
            }
        )
    return (
        pd.DataFrame(rows).sort_values("information_bits", ascending=False).reset_index(drop=True)
    )


def drop_one_profile(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Measure what is lost by no longer requiring each variable to match.

    A model-free importance measure. Dropping a variable from the match merges
    groups that differed only in it; how much information that merge destroys is
    how much the variable was contributing, given all the others.

    Unlike a model's feature importance this depends on no estimator, no
    hyperparameter, and no fit. It answers the question the ablation asks, one
    variable at a time instead of one block at a time.

    .. warning::

       Reliable only while the full variable set still carries positive
       information. Once matching on everything overfits — which it does as soon
       as most groups hold one or two records — every reduced set also overfits,
       and this function is differencing two negative numbers whose ordering is
       noise. A warning is logged when that happens, and
       :func:`single_feature_profile` should be used instead.

    Returns
    -------
    pandas.DataFrame
        One row per variable, ordered by information lost, with the group sizes
        that result from dropping it.
    """
    columns = list(features) if features is not None else list(X.columns)
    full = slice_dispersion(X, y, columns)

    if full["information_bits"] <= 0:
        log.warning(
            "Matching on all %d variables carries %.4f bits, which is not positive: "
            "the full set already overfits, so drop-one differences are unreliable. "
            "Use single_feature_profile instead.",
            len(columns),
            full["information_bits"],
        )

    rows = []
    for name in columns:
        without = [column for column in columns if column != name]
        reduced = slice_dispersion(X, y, without)
        rows.append(
            {
                "dropped": name,
                "information_bits": reduced["information_bits"],
                "information_lost": full["information_bits"] - reduced["information_bits"],
                "n_cells": reduced["n_cells"],
                "mean_cell_size": reduced["mean_cell_size"],
                "singleton_share": reduced["singleton_share"],
            }
        )

    return (
        pd.DataFrame(rows).sort_values("information_lost", ascending=False).reset_index(drop=True)
    )
