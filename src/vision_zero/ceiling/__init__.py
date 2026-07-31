"""Estimating the limit of what these features can support.

Three independent lines of evidence, arriving at the same place:

- :mod:`~vision_zero.ceiling.bayes_error` bounds achievable accuracy from above,
  by counting how often identical feature vectors carry different outcomes.
- :mod:`~vision_zero.ceiling.information` bounds the error rate from below, via
  Fano's inequality applied to the entropy the features leave behind.
- :mod:`~vision_zero.ceiling.permutation` establishes what any of it is worth
  against the null of no relationship at all.
"""

from __future__ import annotations

from vision_zero.ceiling.ablation import block_ablation, cumulative_ablation
from vision_zero.ceiling.bayes_error import (
    CeilingEstimate,
    HeldOutCeiling,
    cell_labels,
    held_out_ceiling,
    in_sample_ceiling,
    multiplicity_profile,
)
from vision_zero.ceiling.examples import ambiguous_cells, describe_cell, worst_case_pairs
from vision_zero.ceiling.information import (
    EntropyBracket,
    conditional_entropy,
    conditional_entropy_bracket,
    entropy,
    fano_error_bound,
    held_out_conditional_entropy,
    mutual_information_per_feature,
)
from vision_zero.ceiling.leakage import (
    conditional_class_distribution,
    deterministic_values,
    leakage_report,
)
from vision_zero.ceiling.missingness import (
    missing_count_profile,
    missingness_indicators,
    missingness_report,
)
from vision_zero.ceiling.permutation import PermutationResult, permutation_null
from vision_zero.ceiling.resolution import (
    drop_one_profile,
    resolution_profile,
    single_feature_profile,
    slice_dispersion,
)
from vision_zero.ceiling.sweep import learning_curve, model_sweep

__all__ = [
    "CeilingEstimate",
    "EntropyBracket",
    "HeldOutCeiling",
    "PermutationResult",
    "ambiguous_cells",
    "block_ablation",
    "cell_labels",
    "conditional_class_distribution",
    "conditional_entropy",
    "conditional_entropy_bracket",
    "cumulative_ablation",
    "describe_cell",
    "deterministic_values",
    "drop_one_profile",
    "entropy",
    "fano_error_bound",
    "held_out_ceiling",
    "held_out_conditional_entropy",
    "in_sample_ceiling",
    "leakage_report",
    "learning_curve",
    "missing_count_profile",
    "missingness_indicators",
    "missingness_report",
    "model_sweep",
    "multiplicity_profile",
    "mutual_information_per_feature",
    "permutation_null",
    "resolution_profile",
    "single_feature_profile",
    "slice_dispersion",
    "worst_case_pairs",
]
