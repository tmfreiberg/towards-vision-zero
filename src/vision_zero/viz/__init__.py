"""Figures for the analysis documents.

Every function returns a :class:`~matplotlib.figure.Figure` and does nothing
else. None of them writes a file, calls ``show``, or reads from disk. Where a
figure is saved, and at what size, is the document's business, and a plotting
function that decides it cannot be reused or tested.
"""

from __future__ import annotations

from vision_zero.viz.plots import (
    plot_ablation,
    plot_ambiguous_group,
    plot_class_shares,
    plot_learning_curve,
    plot_missingness_profile,
    plot_multiplicity_profile,
    plot_reliability,
    plot_sweep,
    plot_trend,
)
from vision_zero.viz.theme import SEVERITY_COLOURS, SEVERITY_LABELS, apply_theme, figure

__all__ = [
    "SEVERITY_COLOURS",
    "SEVERITY_LABELS",
    "apply_theme",
    "figure",
    "plot_ablation",
    "plot_ambiguous_group",
    "plot_class_shares",
    "plot_learning_curve",
    "plot_missingness_profile",
    "plot_multiplicity_profile",
    "plot_reliability",
    "plot_sweep",
    "plot_trend",
]
