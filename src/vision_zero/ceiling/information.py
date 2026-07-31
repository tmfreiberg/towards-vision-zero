r"""Bounding the error rate from below, by how much the features say.

The cell-counting argument bounds accuracy from above. This one comes at the
same conclusion from the opposite direction, and depends on nothing about how a
classifier is built.

Fano's inequality relates the error probability of *any* estimator of :math:`Y`
from :math:`X` to the entropy remaining in :math:`Y` once :math:`X` is known:

.. math::

    H_b(P_e) + P_e \log_2(M - 1) \;\ge\; H(Y \mid X),

for :math:`M` classes, with :math:`H_b` the binary entropy. The left side
increases in :math:`P_e` over the range of interest, so inverting it gives a
floor under the error rate of every possible method — Bayes-optimal ones
included.

The catch is estimating :math:`H(Y \mid X)`, and it is worth being plain about
it. The plug-in estimate formed by averaging within-cell entropies is biased
*downward* when cells are small: a feature combination seen once has zero
observed conditional entropy regardless of the truth. Downward bias in
:math:`H(Y \mid X)` produces a bound that is too weak, so the resulting floor
errs toward being too generous. That direction is the safe one for this
argument — an understated floor cannot manufacture a conclusion — but it does
mean the number should be read as conservative rather than sharp, and
:func:`conditional_entropy` takes a multiplicity cut-off for the same reason
the cell-counting estimator does.

Examples
--------
>>> import numpy as np
>>> # Three classes, features telling you nothing: the floor is 2/3.
>>> round(fano_error_bound(np.log2(3), n_classes=3), 4)
0.6667
>>> # Features determining the outcome: no floor at all.
>>> fano_error_bound(0.0, n_classes=3)
0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.feature_selection import mutual_info_classif

from vision_zero.ceiling.bayes_error import cell_labels

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "EntropyBracket",
    "binary_entropy",
    "conditional_entropy",
    "conditional_entropy_bracket",
    "entropy",
    "fano_error_bound",
    "held_out_conditional_entropy",
    "mutual_information_per_feature",
]

#: Additive smoothing used when scoring a held-out record against the class
#: distribution of its training cell. Without it, a record whose outcome never
#: occurred in its cell contributes an infinite surprise and the estimate
#: diverges on a single observation.
LAPLACE_ALPHA = 0.5


def entropy(y: np.ndarray | pd.Series, base: float = 2.0) -> float:
    """Return the entropy of a label distribution.

    Examples
    --------
    >>> import numpy as np
    >>> round(entropy(np.array([0, 1])), 4)
    1.0
    >>> entropy(np.array([0, 0, 0]))
    0.0
    """
    values = np.asarray(y)
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log(nonzero) / np.log(base)).sum())


def binary_entropy(p: float) -> float:
    """Return the entropy of a Bernoulli variable, in bits.

    Examples
    --------
    >>> binary_entropy(0.5)
    1.0
    >>> binary_entropy(0.0)
    0.0
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def conditional_entropy(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str] | None = None,
    min_multiplicity: int = 1,
    cells: pd.Series | None = None,
) -> float:
    """Estimate the entropy remaining in the labels once the features are known.

    Computed as the multiplicity-weighted mean of within-cell label entropy.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    features
        Columns forming the combination.
    min_multiplicity
        Ignore combinations rarer than this. Small cells drag the estimate
        toward zero, so raising the cut-off counteracts the downward bias
        described in the module docstring.

    Returns
    -------
    float
        Estimated conditional entropy in bits. Zero when the features determine
        the labels within every cell considered.
    """
    identifiers = cell_labels(X, features) if cells is None else cells
    frame = pd.DataFrame({"cell": identifiers.to_numpy(), "y": np.asarray(y)})

    # Computed from class counts per cell rather than by calling `entropy` on
    # each cell in turn. The two agree, but there are tens of thousands of cells
    # and a Python call for each is the difference between three seconds and a
    # fraction of one.
    counts: pd.Series = frame.groupby(["cell", "y"], observed=True).size()
    totals: pd.Series = counts.groupby(level="cell").sum()
    eligible: pd.Series = totals[totals >= min_multiplicity]
    if len(eligible) == 0:
        return 0.0

    kept = counts[counts.index.get_level_values("cell").isin(eligible.index)]
    denominators = eligible.reindex(kept.index.get_level_values("cell")).to_numpy()
    proportions = kept.to_numpy() / denominators
    terms = pd.Series(-proportions * np.log2(proportions), index=kept.index)
    per_cell = terms.groupby(level="cell").sum()

    return float((per_cell * eligible).sum() / eligible.sum())


def held_out_conditional_entropy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    features: Sequence[str] | None = None,
    alpha: float = LAPLACE_ALPHA,
) -> tuple[float, float]:
    r"""Estimate the remaining entropy by scoring held-out records.

        Learns each cell's class distribution from training records, then measures
        the average surprise of validation records under it. This is a cross
        entropy, and cross entropy exceeds the true conditional entropy by the
        divergence between the estimated and actual distributions:

        .. math::

            \mathrm{CE} = H(Y \mid X) + \mathbb{E}\left[
                D_{\mathrm{KL}}(p \,\|\, \hat{p})
    ight] \;\ge\; H(Y \mid X).

        So it errs *upward* where the plug-in estimate errs downward, and the two
        together bracket the quantity rather than either one pinning it. That
        direction matters for how the result may be used: an overstated
        :math:`H(Y \mid X)` yields an overstated floor under the error rate, which
        would flatter the argument this project is making. It is reported as the
        optimistic end of a range and never on its own.

    Parameters
    ----------
        X_train, y_train
            Records from which each cell's class distribution is learned.
        X_val, y_val
            Records scored against it.
        features
            Columns forming the combination.
        alpha
            Additive smoothing on the training counts.

    Returns
    -------
        tuple of float
            Cross entropy in bits, and the fraction of validation records covered.
            Coverage below one means most validation records fell in combinations
            absent from training, and the figure speaks only for the rest.
    """
    columns = list(features) if features is not None else list(X_train.columns)
    combined = pd.concat([X_train[columns], X_val[columns]], ignore_index=True)
    identifiers = cell_labels(combined, columns)

    train_cells = identifiers.iloc[: len(X_train)].to_numpy()
    val_cells = identifiers.iloc[len(X_train) :].to_numpy()

    classes = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_val)]))
    n_classes = len(classes)
    position = {label: index for index, label in enumerate(classes)}

    # `groupby().size().unstack()` rather than a pivot table: the aggregation is
    # already a count, and pivot_table would recompute it more slowly.
    pairs = pd.DataFrame({"cell": train_cells, "y": np.asarray(y_train)})
    tallied = pairs.groupby(["cell", "y"], observed=True).size()
    counts = tallied.unstack(fill_value=0).reindex(columns=classes, fill_value=0)  # noqa: PD010
    smoothed = (counts + alpha).div(counts.sum(axis=1) + alpha * n_classes, axis=0)

    known = smoothed.index
    seen = np.isin(val_cells, known)
    if not seen.any():
        return float("nan"), 0.0

    rows = smoothed.loc[val_cells[seen]].to_numpy()
    observed = np.array([position[label] for label in np.asarray(y_val)[seen]])
    probabilities = rows[np.arange(len(observed)), observed]

    return float(-np.log2(probabilities).mean()), float(seen.mean())


@dataclass(frozen=True, slots=True)
class EntropyBracket:
    """A range for the entropy the features leave behind, and what it implies.

    Neither available estimator is unbiased, and they err in opposite
    directions, so the honest report is an interval.

    Attributes
    ----------
    plug_in
        Within-cell entropy averaged over cells, biased downward because a cell
        seen once shows no uncertainty whatever the truth.
    held_out
        Cross entropy of validation records under training-cell distributions,
        biased upward.
    label_entropy
        :math:`H(Y)`, for scale.
    n_classes
        Number of classes.
    coverage
        Fraction of validation records the held-out figure speaks for.

    Notes
    -----
    ``conservative_floor`` is the one to quote. It follows from the downward
    biased estimate and therefore understates the difficulty; a bound that
    errs toward being too easy cannot manufacture the conclusion this project
    argues for.
    """

    plug_in: float
    held_out: float
    label_entropy: float
    n_classes: int
    coverage: float

    @property
    def conservative_floor(self) -> float:
        """Fano floor from the downward-biased estimate. The one to quote."""
        return fano_error_bound(self.plug_in, self.n_classes)

    @property
    def optimistic_floor(self) -> float:
        """Fano floor from the upward-biased estimate. An upper end, not a claim."""
        capped = min(self.held_out, float(np.log2(self.n_classes)))
        return fano_error_bound(capped, self.n_classes)

    @property
    def information_share(self) -> float:
        """Fraction of label entropy the features account for, conservatively."""
        if self.label_entropy <= 0:
            return 0.0
        return max(0.0, 1 - self.plug_in / self.label_entropy)


def conditional_entropy_bracket(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    features: Sequence[str] | None = None,
    min_multiplicity: int = 2,
) -> EntropyBracket:
    """Bracket the remaining entropy between its two biased estimates."""
    plug_in = conditional_entropy(
        X_train, y_train, features=features, min_multiplicity=min_multiplicity
    )
    held_out, coverage = held_out_conditional_entropy(
        X_train, y_train, X_val, y_val, features=features
    )
    labels = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
    return EntropyBracket(
        plug_in=plug_in,
        held_out=held_out,
        label_entropy=entropy(labels),
        n_classes=len(np.unique(labels)),
        coverage=coverage,
    )


def fano_error_bound(conditional_entropy_bits: float, n_classes: int) -> float:
    r"""Invert Fano's inequality to bound the error rate from below.

    Solves :math:`H_b(p) + p \log_2(M - 1) = H(Y \mid X)` for the smallest
    :math:`p`. The left side is strictly increasing on :math:`[0, (M-1)/M]`, so
    the root is unique there and is the tightest floor the inequality supports.

    Parameters
    ----------
    conditional_entropy_bits
        :math:`H(Y \mid X)` in bits.
    n_classes
        Number of classes, at least two.

    Returns
    -------
    float
        Lower bound on the error probability of any estimator of ``y`` from
        ``X``. Zero when the features leave no uncertainty.

    Raises
    ------
    ValueError
        If ``n_classes`` is below two, or the entropy is negative or exceeds
        what that many classes admit.

    Examples
    --------
    >>> import numpy as np
    >>> round(fano_error_bound(np.log2(3), n_classes=3), 4)
    0.6667
    >>> round(fano_error_bound(1.0, n_classes=2), 4)
    0.5
    """
    if n_classes < 2:
        raise ValueError(f"Need at least two classes, got {n_classes}")
    if conditional_entropy_bits < 0:
        raise ValueError("Conditional entropy cannot be negative")

    maximum = float(np.log2(n_classes))
    if conditional_entropy_bits > maximum + 1e-9:
        raise ValueError(
            f"Conditional entropy {conditional_entropy_bits:.4f} exceeds "
            f"log2({n_classes}) = {maximum:.4f}"
        )
    if conditional_entropy_bits <= 0:
        return 0.0

    ceiling_p = (n_classes - 1) / n_classes

    def gap(p: float) -> float:
        excess = binary_entropy(p) + p * float(np.log2(n_classes - 1))
        return excess - conditional_entropy_bits

    # At p = (M-1)/M the left side equals log2(M) exactly, so a root exists
    # whenever the conditional entropy is within range.
    if gap(ceiling_p) < 0:
        return ceiling_p
    root: float = brentq(gap, 0.0, ceiling_p, xtol=1e-12)
    return float(root)


def mutual_information_per_feature(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = 42,
    missing_fill: int = -1,
) -> pd.Series:
    """Estimate each feature's mutual information with the labels, separately.

    Per-feature figures do not add up to the joint information — features
    overlap — so this does not bound anything on its own. What it does show is
    which individual variables carry any signal at all, and how little. A
    feature whose mutual information with the outcome is a hundredth of a bit is
    not going to rescue a model.

    Parameters
    ----------
    X, y
        Features and labels, aligned.
    seed
        Seed for the estimator, which is stochastic.
    missing_fill
        Value standing in for nulls, which the estimator cannot accept.

    Returns
    -------
    pandas.Series
        Mutual information in nats, descending.
    """
    encoded = X.copy()
    for name in encoded.columns:
        if isinstance(encoded[name].dtype, pd.CategoricalDtype):
            encoded[name] = encoded[name].cat.codes.replace(-1, missing_fill)
        else:
            encoded[name] = pd.to_numeric(encoded[name], errors="coerce").fillna(missing_fill)

    scores = mutual_info_classif(
        encoded.to_numpy(dtype=float),
        np.asarray(y),
        discrete_features=True,
        random_state=seed,
    )
    return pd.Series(scores, index=list(X.columns)).sort_values(ascending=False)
