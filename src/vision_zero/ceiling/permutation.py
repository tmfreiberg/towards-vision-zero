"""Testing a score against the null of no relationship.

"Better than chance" is a claim about a distribution, and the way to check it is
to destroy the relationship and see what scores survive. Shuffling the labels
leaves every marginal untouched — the same class imbalance, the same feature
distribution, the same number of records — while severing any connection between
features and outcome. Whatever a procedure scores on shuffled data is what it
scores on nothing.

This matters twice over here. It licenses the claim that a fitted model beats
chance, and, less obviously, it exposes how much of the cell-counting ceiling is
an artifact of sparsity: run :func:`permutation_null` over
:func:`~vision_zero.ceiling.bayes_error.in_sample_ceiling` and the ceiling stays
high even though, by construction, there is nothing left to predict.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> y = np.array([0] * 80 + [1] * 20)
>>> result = permutation_null(
...     lambda labels: float((labels[:50] == 0).mean()), y, n_permutations=50, seed=0
... )
>>> 0.6 < result.null_mean < 1.0
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vision_zero.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["PermutationResult", "permutation_null"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PermutationResult:
    """A score set against the distribution of scores under shuffled labels.

    Attributes
    ----------
    observed
        The score on the real labels.
    null_scores
        Scores under each shuffle.
    p_value
        Fraction of shuffles scoring at least as high as the observation, with
        the observation itself counted. Including it is the standard
        conservative correction and keeps the p-value from ever being zero,
        which it cannot honestly be from finitely many shuffles.
    n_permutations
        Number of shuffles.
    """

    observed: float
    null_scores: np.ndarray
    p_value: float
    n_permutations: int

    @property
    def null_mean(self) -> float:
        """Mean score under the null."""
        return float(self.null_scores.mean())

    @property
    def null_std(self) -> float:
        """Standard deviation of scores under the null."""
        return float(self.null_scores.std(ddof=1)) if len(self.null_scores) > 1 else 0.0

    @property
    def effect_size(self) -> float:
        """How many null standard deviations separate the observation from the null mean.

        Zero when the null has no spread, which happens when a procedure returns
        the same score however the labels fall.
        """
        spread = self.null_std
        return (self.observed - self.null_mean) / spread if spread > 0 else 0.0

    def summary(self) -> dict[str, float]:
        """Return the result as a flat mapping, for tabulation."""
        return {
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "n_permutations": float(self.n_permutations),
        }


def permutation_null(
    score: Callable[[np.ndarray], float],
    y: np.ndarray,
    *,
    n_permutations: int = 200,
    seed: int = 42,
    observed: float | None = None,
) -> PermutationResult:
    """Score the real labels against scores from shuffled ones.

    Parameters
    ----------
    score
        Takes a label array and returns a score, higher being better. Everything
        else the score depends on — features, splits, the estimator — must be
        held fixed inside the callable, or the shuffle is not the only thing
        varying between runs.
    y
        The real labels.
    n_permutations
        Number of shuffles. The smallest attainable p-value is
        ``1 / (n_permutations + 1)``, so 200 shuffles cannot report below 0.005
        however strong the effect.
    seed
        Seed for the shuffles.
    observed
        Score on the real labels, if already computed. Saves one evaluation.

    Returns
    -------
    PermutationResult
        Observation, null distribution, and p-value.

    Raises
    ------
    ValueError
        If ``n_permutations`` is below one.
    """
    if n_permutations < 1:
        raise ValueError(f"Need at least one permutation, got {n_permutations}")

    labels = np.asarray(y)
    real = score(labels) if observed is None else observed

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        null[index] = score(rng.permutation(labels))

    at_least_as_extreme = int((null >= real).sum())
    p_value = (at_least_as_extreme + 1) / (n_permutations + 1)

    log.info(
        "Permutation test: observed %.4f, null %.4f +/- %.4f, p = %.4f",
        real,
        null.mean(),
        null.std(ddof=1) if n_permutations > 1 else 0.0,
        p_value,
    )
    return PermutationResult(
        observed=real, null_scores=null, p_value=p_value, n_permutations=n_permutations
    )
