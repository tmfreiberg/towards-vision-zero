"""Figures for the analysis documents.

Each function takes a table produced elsewhere in the package and returns a
figure. Nothing here computes anything of substance: if a chart needs a number,
the number comes from the module that owns it, so that the figure and the text
around it cannot disagree.

Examples
--------
>>> from vision_zero.ceiling import multiplicity_profile  # doctest: +SKIP
>>> from vision_zero.viz import plot_multiplicity_profile  # doctest: +SKIP
>>> fig = plot_multiplicity_profile(multiplicity_profile(X, y))  # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vision_zero.viz.theme import GREY, MUTED, SEVERITY_LABELS, figure, severity_colour

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

__all__ = [
    "plot_ablation",
    "plot_ambiguous_group",
    "plot_class_shares",
    "plot_learning_curve",
    "plot_missingness_profile",
    "plot_multiplicity_profile",
    "plot_reliability",
    "plot_sweep",
    "plot_trend",
    "plot_volume_and_rate",
]


def _class_columns(frame: pd.DataFrame, prefix: str = "p_class_") -> list[str]:
    """Return the outcome-share columns of a table, in class order."""
    return sorted(
        (column for column in frame.columns if column.startswith(prefix)),
        key=lambda column: int(column.removeprefix(prefix)),
    )


def plot_multiplicity_profile(profile: pd.DataFrame) -> Figure:
    """Plot the accuracy bound and its coverage against the multiplicity cut-off.

    Both curves belong on one figure because neither means anything alone: the
    bound tightens as the cut-off rises only because it is speaking for fewer
    and fewer records, and a reader shown the first without the second would
    take a number computed on two percent of the data for a fact about all of
    it.
    """
    fig, ax = figure()
    cutoffs = profile["min_multiplicity"].to_numpy()

    ax.plot(cutoffs, profile["ceiling"], marker="o", color=severity_colour(2), label="ceiling")
    ax.set_xscale("log")
    ax.set_xticks(cutoffs)
    ax.set_xticklabels([str(int(value)) for value in cutoffs])
    ax.set_xlabel("minimum records per feature combination")
    ax.set_ylabel("upper bound on accuracy")
    ax.set_ylim(0, 1)

    coverage = ax.twinx()
    coverage.plot(
        cutoffs, profile["coverage"], marker="s", linestyle="--", color=MUTED, label="coverage"
    )
    coverage.set_ylabel("share of records covered")
    coverage.set_ylim(0, 1)
    coverage.grid(visible=False)

    handles = list(ax.get_lines()) + list(coverage.get_lines())
    ax.legend(handles=handles, loc="center right")
    ax.set_title("The bound tightens only by covering fewer records")
    return fig


def plot_ambiguous_group(group: pd.Series, classes: tuple[int, ...]) -> Figure:
    """Plot the outcomes of one group of identically recorded collisions.

    The single most direct statement of the limit: one bar per outcome, over
    records that no procedure reading these variables could tell apart.
    """
    fig, ax = figure(height=2.4)
    counts = [int(group[f"n_class_{label}"]) for label in classes]
    positions = np.arange(len(classes))

    bars = ax.barh(positions, counts, color=[severity_colour(label) for label in classes])
    ax.set_yticks(positions)
    ax.set_yticklabels([SEVERITY_LABELS.get(label, str(label)) for label in classes])
    ax.invert_yaxis()
    ax.set_xlabel("collisions")
    ax.grid(visible=False, axis="y")

    total = sum(counts)
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_width() + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({count / total:.0%})",
            va="center",
            fontsize=8,
            color=GREY,
        )
    ax.set_xlim(0, total * 1.18)
    ax.set_title(f"{total:,} collisions recorded identically")
    return fig


def plot_class_shares(table: pd.DataFrame, title: str = "") -> Figure:
    """Plot outcome shares across the values of one variable, as stacked bars.

    Bar widths are not varied by support, so the record counts are printed
    beside the labels instead: a value covering two hundred records and one
    covering two hundred thousand otherwise look equally authoritative.
    """
    fig, ax = figure(height=max(2.4, 0.34 * len(table) + 1.2))
    share_columns = _class_columns(table)
    positions = np.arange(len(table))

    left = np.zeros(len(table))
    for column in share_columns:
        label = int(column.removeprefix("p_class_"))
        widths = table[column].to_numpy()
        ax.barh(
            positions,
            widths,
            left=left,
            color=severity_colour(label),
            label=SEVERITY_LABELS.get(label, str(label)),
        )
        left += widths

    labels = [f"{row['label']}  ({int(row['n']):,})" for _, row in table.iterrows()]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of collisions")
    ax.grid(visible=False, axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    if title:
        ax.set_title(title, pad=28)
    return fig


def plot_trend(table: pd.DataFrame, period: str, title: str = "") -> Figure:
    """Plot record counts over an ordered period.

    For the descriptive section, where the point of a chart is usually that it
    shows what everyone expected.
    """
    fig, ax = figure()
    positions = np.arange(len(table))
    ax.bar(positions, table["n"], color=MUTED)
    ax.set_xticks(positions)
    ax.set_xticklabels(table["label"], rotation=45, ha="right")
    ax.set_ylabel("collisions")
    ax.set_xlabel(period)
    ax.grid(visible=False, axis="x")
    if title:
        ax.set_title(title)
    return fig


def plot_volume_and_rate(
    table: pd.DataFrame,
    rate_column: str,
    *,
    rate_label: str = "",
    title: str = "",
) -> Figure:
    """Plot how often something happens against how badly it turns out.

    The two need to share a figure because the interesting cases are the ones
    where they disagree. Collisions are most frequent in winter and most severe
    in summer; showing either alone invites the reader to assume the other
    follows it.

    Parameters
    ----------
    table
        Rows in the order they should appear, with ``label``, ``n``, and the
        rate column.
    rate_column
        Column holding the rate to draw as a line, typically the share of one
        outcome class.
    rate_label
        Axis label for the rate.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Counts as bars against the left axis, the rate as a line against the
        right.
    """
    fig, ax = figure()
    positions = np.arange(len(table))

    ax.bar(positions, table["n"], color="#d5d8d1", label="collisions")
    ax.set_xticks(positions)
    ax.set_xticklabels(table["label"], rotation=45, ha="right")
    ax.set_ylabel("collisions")
    ax.grid(visible=False, axis="x")

    rate = ax.twinx()
    rate.plot(
        positions,
        table[rate_column],
        marker="o",
        color=severity_colour(2),
        linewidth=1.8,
        label=rate_label or rate_column,
    )
    rate.set_ylabel(rate_label or rate_column)
    rate.set_ylim(bottom=0)
    rate.grid(visible=False)

    handles = [*ax.containers, *rate.get_lines()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    if title:
        ax.set_title(title, pad=30)
    return fig


def plot_missingness_profile(profile: pd.DataFrame) -> Figure:
    """Plot outcome shares against the number of blank fields.

    A monotone trend here is a statement about report completeness rather than
    about roads, which is why it is worth a figure of its own.
    """
    fig, ax = figure()
    share_columns = _class_columns(profile)
    blanks = profile["n_missing"].to_numpy()

    bottom = np.zeros(len(profile))
    for column in share_columns:
        label = int(column.removeprefix("p_class_"))
        heights = profile[column].to_numpy()
        ax.bar(
            blanks,
            heights,
            bottom=bottom,
            color=severity_colour(label),
            label=SEVERITY_LABELS.get(label, str(label)),
        )
        bottom += heights

    ax.set_xticks(blanks)
    ax.set_xlabel("blank fields in the report")
    ax.set_ylabel("share of collisions")
    ax.set_ylim(0, 1)
    ax.grid(visible=False, axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    ax.set_title("Outcome by how completely the report was filled in", pad=28)
    return fig


def plot_ablation(ablation: pd.DataFrame, metric: str = "mcc") -> Figure:
    """Plot each block's score alone against what it adds to the others.

    The two are different questions and the figure keeps them side by side. A
    block can score well alone and add nothing, which happens whenever two
    blocks carry the same information.
    """
    fig, ax = figure(height=3.0)
    blocks = ablation[~ablation["block"].isin(["majority baseline", "all features"])]
    positions = np.arange(len(blocks))
    height = 0.38

    ax.barh(
        positions - height / 2,
        blocks[f"{metric}_alone"],
        height=height,
        color=MUTED,
        label="alone",
    )
    ax.barh(
        positions + height / 2,
        blocks[f"{metric}_added"],
        height=height,
        color=severity_colour(2),
        label="added to the others",
    )

    baseline = ablation.loc[ablation["block"] == "majority baseline", f"{metric}_alone"]
    if not baseline.empty:
        ax.axvline(
            float(baseline.iloc[0]),
            color=GREY,
            linestyle=":",
            linewidth=1.2,
            label="majority baseline",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(blocks["block"])
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace("_", " "))
    ax.grid(visible=False, axis="y")
    ax.legend(loc="lower right")
    ax.set_title("What each group of variables is worth")
    return fig


def plot_learning_curve(curve: pd.DataFrame, metric: str = "mcc") -> Figure:
    """Plot performance against training-set size.

    A flat curve says the shortage is of information rather than of records, and
    that collecting more of the same would not help.
    """
    fig, ax = figure()
    ax.plot(curve["n_train"], curve[metric], marker="o", color=severity_colour(2))
    ax.set_xscale("log")
    ax.set_xlabel("training records")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(bottom=0)
    ax.set_title("More records do not help")
    return fig


def plot_sweep(sweep: pd.DataFrame, metric: str = "mcc_mean") -> Figure:
    """Plot estimator families against one another and against the baseline.

    Families with very different inductive biases arriving at the same score is
    the evidence that the limit is in the data.
    """
    fig, ax = figure(height=2.8)
    ordered = sweep.sort_values(metric)
    positions = np.arange(len(ordered))
    colours = [
        GREY if name == "majority baseline" else severity_colour(2) for name in ordered["family"]
    ]

    ax.barh(positions, ordered[metric], color=colours)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["family"])
    ax.set_xlabel(metric.replace("_", " ").replace(" mean", ""))
    ax.grid(visible=False, axis="y")
    ax.set_title("Every estimator family lands in the same place")
    return fig


def plot_reliability(table: pd.DataFrame, class_label: int) -> Figure:
    """Plot stated probability against observed frequency for one class.

    Points on the diagonal mean the model's odds are right. That can hold while
    the class is never predicted outright, which is the distinction between
    being calibrated and being able to discriminate.
    """
    fig, ax = figure(width=4.2, height=4.0)
    ax.plot([0, 1], [0, 1], linestyle=":", color=GREY, linewidth=1.2, label="perfect")

    sizes = 20 + 180 * table["n"] / table["n"].max()
    ax.scatter(
        table["predicted"],
        table["observed"],
        s=sizes,
        color=severity_colour(class_label),
        alpha=0.85,
        label="observed",
    )
    ax.set_xlabel("stated probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, max(0.05, float(table["predicted"].max()) * 1.15))
    ax.set_ylim(0, max(0.05, float(table["observed"].max()) * 1.15))
    ax.legend(loc="upper left")
    ax.set_title(f"Calibration, {SEVERITY_LABELS.get(class_label, class_label)}")
    return fig
