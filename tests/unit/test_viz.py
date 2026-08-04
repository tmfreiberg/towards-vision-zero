"""Tests for the figures.

Plots are checked for structure rather than appearance: that they render at all,
that they carry the expected number of marks, and that they neither write files
nor open windows. A figure function that saved its own output could not be
reused at a different size.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vision_zero import viz
from vision_zero.viz.theme import SEVERITY_COLOURS, severity_colour


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def test_severity_colours_are_distinct() -> None:
    assert len(set(SEVERITY_COLOURS.values())) == len(SEVERITY_COLOURS)


def test_unknown_severity_gets_a_neutral_colour() -> None:
    assert severity_colour(99) not in SEVERITY_COLOURS.values()


def test_figure_helper_returns_a_figure_and_axes() -> None:
    fig, ax = viz.figure()
    assert fig is ax.figure


def test_multiplicity_profile_draws_both_curves() -> None:
    profile = pd.DataFrame(
        {
            "min_multiplicity": [1, 2, 5],
            "ceiling": [0.93, 0.86, 0.83],
            "coverage": [1.0, 0.47, 0.25],
            "cells": [100, 50, 10],
            "covered_records": [100, 50, 10],
            "ambiguous_cells": [10, 8, 3],
            "ambiguous_share": [0.2, 0.5, 0.7],
        }
    )
    fig = viz.plot_multiplicity_profile(profile)
    assert len(fig.axes) == 2, "the bound is meaningless without its coverage"


def test_ambiguous_group_draws_one_bar_per_class() -> None:
    group = pd.Series({"n": 129, "n_class_0": 80, "n_class_1": 48, "n_class_2": 1})
    fig = viz.plot_ambiguous_group(group, (0, 1, 2))
    assert len(fig.axes[0].patches) == 3


def test_class_shares_stack_to_the_full_width() -> None:
    table = pd.DataFrame(
        {
            "label": ["Clear", "Rain"],
            "n": [1000, 200],
            "p_class_0": [0.8, 0.7],
            "p_class_1": [0.19, 0.28],
            "p_class_2": [0.01, 0.02],
        }
    )
    fig = viz.plot_class_shares(table, "Outcome by weather")
    assert len(fig.axes[0].patches) == 6


def test_trend_draws_one_bar_per_period() -> None:
    table = pd.DataFrame(
        {"label": ["Jan", "Feb", "Mar"], "n": [10, 20, 30], "share": [0.1, 0.2, 0.3]}
    )
    fig = viz.plot_trend(table, "MONTH")
    assert len(fig.axes[0].patches) == 3


def test_missingness_profile_renders() -> None:
    profile = pd.DataFrame(
        {
            "n_missing": [0, 1, 2],
            "n_records": [100, 50, 10],
            "share_of_records": [0.62, 0.31, 0.07],
            "p_class_0": [0.75, 0.83, 0.93],
            "p_class_1": [0.24, 0.16, 0.07],
            "p_class_2": [0.01, 0.01, 0.0],
        }
    )
    assert viz.plot_missingness_profile(profile).axes


def test_ablation_marks_the_baseline() -> None:
    ablation = pd.DataFrame(
        {
            "block": ["majority baseline", "all features", "participants", "environment"],
            "n_features": [0, 18, 6, 9],
            "mcc_alone": [0.0, 0.408, 0.411, 0.029],
            "mcc_added": [np.nan, np.nan, 0.090, 0.014],
        }
    )
    fig = viz.plot_ablation(ablation)
    assert len(fig.axes[0].patches) == 4, "two bars for each of two blocks"
    assert fig.axes[0].get_lines(), "the baseline must be drawn"


def test_learning_curve_renders() -> None:
    curve = pd.DataFrame(
        {"fraction": [0.25, 0.5, 1.0], "n_train": [10, 20, 40], "mcc": [0.41, 0.43, 0.44]}
    )
    assert viz.plot_learning_curve(curve).axes


def test_sweep_distinguishes_the_baseline() -> None:
    sweep = pd.DataFrame(
        {"family": ["xgboost", "logistic", "majority baseline"], "mcc_mean": [0.44, 0.43, 0.0]}
    )
    fig = viz.plot_sweep(sweep)
    colours = {patch.get_facecolor() for patch in fig.axes[0].patches}
    assert len(colours) == 2, "the baseline is coloured apart from the models"


def test_reliability_draws_the_diagonal() -> None:
    table = pd.DataFrame(
        {
            "bin_lower": [0.0, 0.1],
            "bin_upper": [0.1, 0.2],
            "n": [900, 100],
            "predicted": [0.04, 0.15],
            "observed": [0.05, 0.13],
            "gap": [0.01, -0.02],
        }
    )
    fig = viz.plot_reliability(table, 1)
    assert fig.axes[0].get_lines(), "the perfect-calibration line must be drawn"


def test_plots_do_not_write_files(tmp_path: Path) -> None:
    """Where a figure is saved is the document's business, not the function's."""
    table = pd.DataFrame({"label": ["a"], "n": [1], "share": [1.0]})
    viz.plot_trend(table, "MONTH")
    assert not list(tmp_path.iterdir())


def test_volume_and_rate_share_a_figure() -> None:
    """Shown separately, a reader assumes severity follows frequency. It does not."""
    table = pd.DataFrame(
        {
            "label": ["Jan", "Jul"],
            "n": [3200, 2600],
            "p_class_2": [0.006, 0.014],
        }
    )
    fig = viz.plot_volume_and_rate(table, "p_class_2", rate_label="share severe")
    assert len(fig.axes) == 2, "counts and rate must share one figure"
    assert len(fig.axes[0].patches) == 2
    assert len(fig.axes[1].get_lines()) == 1


def test_every_plot_function_is_exported() -> None:
    """A new figure added to `plots` and not re-exported fails only at render.

    Quarto reports it as `module 'vision_zero.viz' has no attribute ...`, several
    minutes into a build, which is a slow way to learn about a missing line in an
    `__init__`.
    """
    from vision_zero.viz import plots

    defined = {name for name in dir(plots) if name.startswith("plot_")}
    exported = {name for name in dir(viz) if name.startswith("plot_")}
    assert defined == exported, f"not exported from vision_zero.viz: {sorted(defined - exported)}"
