"""Predictability of traffic accident severity from SAAQ police-report features.

This package accompanies an investigation into how much the features recorded on
Quebec accident reports can tell us about the severity of a collision, and how
one would establish that a limit had been reached.

The short answer, developed in the accompanying analysis, is that these features
carry real but small information about severity; that there is a ceiling on
achievable classification performance which is a property of the data rather
than of the model class or the quantity of training data; and that the honest
output is therefore a conditional distribution over outcomes rather than a
predicted label.

Subpackages
-----------
data
    Loading, the bilingual codebook, and the typed schema.
features
    Encoding of categorical and ordinal variables.
modeling
    Pipeline construction, baselines, resampling, and training.
evaluation
    Metrics, confusion matrices, thresholds, and calibration.
ceiling
    Estimation of the predictability ceiling: empirical Bayes error, Fano
    bounds, permutation tests, and neighbourhood structure.
retrospective
    Matching a feature combination against historical records and reporting the
    empirical outcome distribution among the matches.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("towards-vision-zero")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
