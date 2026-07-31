"""Command-line interface.

Every long-running operation is reachable from here. That keeps the analysis
documents thin — a document calls a function or a command, it does not contain
the analysis — and it gives the end-to-end tests something meaningful to invoke.

Commands are grouped by concern::

    tvz data prepare      compile raw CSVs into the Parquet intermediate
    tvz data describe     print a codebook entry, in English or French
    tvz train             fit a model from an experiment config
    tvz evaluate          evaluate a fitted model
    tvz ceiling           estimate the predictability ceiling
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from vision_zero import __version__
from vision_zero.config import get_settings
from vision_zero.logging import configure_logging

app = typer.Typer(
    name="tvz",
    help="Predictability of traffic accident severity from SAAQ report features.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(name="data", help="Data preparation and inspection.", no_args_is_help=True)
app.add_typer(data_app)

console = Console()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Set log level to DEBUG."),
) -> None:
    """Configure logging before any subcommand runs."""
    configure_logging(level="DEBUG" if verbose else None)


@app.command()
def version() -> None:
    """Print the installed package version."""
    console.print(__version__)


@app.command()
def info() -> None:
    """Show resolved paths and whether the expected inputs are present.

    The first thing to run after cloning: it answers "where does this think the
    data is, and is it there?" without requiring anyone to read the source.
    """
    settings = get_settings()

    table = Table(title="tvz configuration", show_lines=False)
    table.add_column("setting", style="bold")
    table.add_column("value")
    table.add_column("exists", justify="center")

    for label, path in (
        ("project root", settings.project_root),
        ("raw data", settings.raw_data_dir),
        ("processed data", settings.processed_data_dir),
        ("model artifacts", settings.model_artifacts_dir),
        ("reports", settings.report_dir),
        ("experiment config", settings.conf_dir),
    ):
        table.add_row(label, str(path), "yes" if path.exists() else "no")

    table.add_row("random seed", str(settings.random_seed), "")
    table.add_row("log level", settings.log_level, "")
    table.add_row("strict data", str(settings.strict_data), "")

    console.print(table)


@data_app.command("prepare")
def data_prepare(
    start: int = typer.Option(2011, "--start", help="First year to compile."),
    end: int = typer.Option(2022, "--end", help="Last year to compile, inclusive."),
    force: bool = typer.Option(False, "--force", help="Recompile even if up to date."),
) -> None:
    """Compile the raw SAAQ CSVs into a single Parquet file."""
    from vision_zero.data.prepare import is_stale, prepare

    years = range(start, end + 1)
    if not force and not is_stale(years):
        console.print("Compiled dataset is up to date. Pass --force to rebuild.")
        return

    result = prepare(years)
    size_mb = result.parquet_path.stat().st_size / (1 << 20)
    console.print(f"Wrote {result.rows:,} records to {result.parquet_path} ({size_mb:.1f} MB)")


@app.command()
def train(
    experiment: str = typer.Argument(..., help="Experiment name or path to a YAML file."),
) -> None:
    """Fit an experiment and report fold sizes and feature importances."""
    from vision_zero.modeling.config import load_experiment
    from vision_zero.modeling.dataset import build_dataset
    from vision_zero.modeling.train import train as run_training

    specification = load_experiment(experiment)
    dataset = build_dataset(specification)
    run = run_training(specification, dataset)

    table = Table(title=f"{specification.name}: {len(run.folds)} folds")
    table.add_column("fold", justify="right")
    table.add_column("fitted", justify="right")
    table.add_column("validated", justify="right")
    table.add_column("seconds", justify="right")
    for fold in run.folds:
        table.add_row(
            str(fold.fold),
            f"{sum(fold.train_counts.values()):,}",
            f"{len(fold.y_true):,}",
            f"{fold.fit_seconds:.1f}",
        )
    console.print(table)

    if dataset.dropped_constant:
        console.print(f"Dropped constant feature(s): {list(dataset.dropped_constant)}")

    importances = run.mean_importances()
    if not importances.empty:
        top = Table(title="Mean feature importance, highest first")
        top.add_column("feature")
        top.add_column("importance", justify="right")
        for name, value in importances.head(10).items():
            top.add_row(str(name), f"{value:.4f}")
        console.print(top)

    console.print("Scoring is deliberately separate from fitting; see the evaluation commands.")


@app.command()
def ceiling(
    experiment: str = typer.Argument(..., help="Experiment name or path to a YAML file."),
    permutations: int = typer.Option(0, "--permutations", help="Shuffles for the null test."),
) -> None:
    """Estimate how well any classifier could do with these features."""
    import numpy as np

    from vision_zero.ceiling import (
        conditional_entropy_bracket,
        held_out_ceiling,
        multiplicity_profile,
        mutual_information_per_feature,
        permutation_null,
    )
    from vision_zero.ceiling.bayes_error import in_sample_ceiling
    from vision_zero.modeling.config import load_experiment
    from vision_zero.modeling.dataset import build_dataset

    specification = load_experiment(experiment)
    dataset = build_dataset(specification)

    profile = Table(title="Upper bound on accuracy, by multiplicity cut-off")
    for column in ("cut-off", "ceiling", "coverage", "cells", "ambiguous"):
        profile.add_column(column, justify="right")
    for _, row in multiplicity_profile(dataset.X_train, dataset.y_train).iterrows():
        ceiling_value = row["ceiling"]
        profile.add_row(
            f"{int(row['min_multiplicity'])}",
            "n/a" if np.isnan(ceiling_value) else f"{ceiling_value:.4f}",
            f"{row['coverage']:.4f}",
            f"{int(row['cells']):,}",
            f"{row['ambiguous_share']:.4f}",
        )
    console.print(profile)

    train_index, validation_index = next(dataset.folds(specification))
    held_out = held_out_ceiling(
        dataset.X_train.iloc[train_index],
        dataset.y_train.iloc[train_index],
        dataset.X_train.iloc[validation_index],
        dataset.y_train.iloc[validation_index],
    )
    console.print(
        f"Held-out ceiling: {held_out.accuracy:.4f} accuracy "
        f"over {held_out.coverage:.1%} of validation records "
        f"(majority baseline {held_out.majority_baseline:.4f})"
    )
    console.print(f"  per-class recall: {held_out.per_class_recall}")
    if not held_out.beats_baseline:
        console.print(
            "  [!] This does not beat a constant predictor, so it is not a ceiling. "
            "Cells are too sparse here for the estimator to say anything."
        )

    bracket = conditional_entropy_bracket(
        dataset.X_train.iloc[train_index],
        dataset.y_train.iloc[train_index],
        dataset.X_train.iloc[validation_index],
        dataset.y_train.iloc[validation_index],
    )
    console.print(
        f"H(Y) = {bracket.label_entropy:.4f} bits; "
        f"H(Y|X) between {bracket.plug_in:.4f} and {bracket.held_out:.4f} bits "
        f"({bracket.coverage:.1%} covered)"
    )
    console.print(
        f"Fano floor on error: {bracket.conservative_floor:.4f} conservatively, "
        f"{bracket.optimistic_floor:.4f} at the optimistic end. "
        "Quote the conservative one."
    )

    information = Table(title="Mutual information per feature, in nats")
    information.add_column("feature")
    information.add_column("information", justify="right")
    for name, value in (
        mutual_information_per_feature(
            dataset.X_train, dataset.y_train, seed=specification.split.seed
        )
        .head(10)
        .items()
    ):
        information.add_row(str(name), f"{value:.5f}")
    console.print(information)

    if permutations:
        result = permutation_null(
            lambda labels: in_sample_ceiling(dataset.X_train, pd.Series(labels)).ceiling,
            dataset.y_train.to_numpy(),
            n_permutations=permutations,
            seed=specification.split.seed,
        )
        console.print(
            f"Permutation null on the in-sample ceiling: observed {result.observed:.4f}, "
            f"null {result.null_mean:.4f}, p = {result.p_value:.4f}"
        )


@app.command()
def sweep(
    experiment: str = typer.Argument(..., help="Experiment name or path to a YAML file."),
    folds: int = typer.Option(3, "--folds", help="Cross-validation folds per family."),
    subsample: int = typer.Option(60_000, "--subsample", help="Cap on training records."),
    curve: bool = typer.Option(False, "--curve", help="Also compute the learning curve."),
) -> None:
    """Compare estimator families, to locate the limit in the data or the model."""
    from vision_zero.ceiling.sweep import learning_curve, model_sweep
    from vision_zero.modeling.config import load_experiment
    from vision_zero.modeling.dataset import build_dataset

    specification = load_experiment(experiment)
    dataset = build_dataset(specification)

    result = model_sweep(specification, dataset, folds=folds, subsample=subsample or None)
    table = Table(title="Estimator families under identical conditions")
    for column in ("family", "F*", "MCC", "accuracy", "severe recall"):
        table.add_column(column, justify="right" if column != "family" else "left")
    for _, row in result.iterrows():
        table.add_row(
            str(row["family"]),
            f"{row['f_star_mean']:.4f}",
            f"{row['mcc_mean']:+.4f}",
            f"{row['accuracy_mean']:.4f}",
            f"{row['severe_recall_mean']:.4f}",
        )
    console.print(table)

    if curve:
        points = learning_curve(specification, dataset, folds=folds, subsample=subsample or None)
        curve_table = Table(title="Learning curve")
        for column in ("fraction", "n train", "F*", "MCC", "severe recall"):
            curve_table.add_column(column, justify="right")
        for _, row in points.iterrows():
            curve_table.add_row(
                f"{row['fraction']:.2f}",
                f"{int(row['n_train']):,}",
                f"{row['f_star']:.4f}",
                f"{row['mcc']:+.4f}",
                f"{row['severe_recall']:.4f}",
            )
        console.print(curve_table)


@app.command()
def audit(
    experiment: str = typer.Argument(..., help="Experiment name or path to a YAML file."),
    min_support: int = typer.Option(30, "--min-support", help="Records a value needs."),
    feature: str = typer.Option("", "--feature", help="Show one feature's full breakdown."),
) -> None:
    """Check whether any feature settles the outcome rather than informing it."""
    from vision_zero.ceiling.leakage import (
        conditional_class_distribution,
        deterministic_values,
        excluded_outcomes,
        leakage_report,
    )
    from vision_zero.modeling.config import load_experiment
    from vision_zero.modeling.dataset import build_dataset

    specification = load_experiment(experiment)
    dataset = build_dataset(specification)

    if feature:
        breakdown = conditional_class_distribution(dataset.X_train, dataset.y_train, feature)
        detail = Table(title=f"{feature}: outcome distribution by value")
        detail.add_column("value")
        detail.add_column("records", justify="right")
        for label in dataset.classes:
            detail.add_column(f"P(class {label})", justify="right")
        for _, row in breakdown.sort_values("n", ascending=False).iterrows():
            detail.add_row(
                str(row["value"]),
                f"{int(row['n']):,}",
                *[f"{float(row[f'p_class_{label}']):.4f}" for label in dataset.classes],
            )
        console.print(detail)
        return

    flagged = deterministic_values(dataset.X_train, dataset.y_train, min_support=min_support)
    if flagged.empty:
        console.print(
            "No feature value is followed by a single outcome. Nothing entails the target."
        )
    else:
        table = Table(title="Values followed by exactly one outcome")
        for column in ("feature", "value", "records", "share", "outcome"):
            table.add_column(column, justify="right" if column != "feature" else "left")
        for _, row in flagged.iterrows():
            table.add_row(
                str(row["feature"]),
                str(row["value"]),
                f"{int(row['n']):,}",
                f"{row['share_of_records']:.4f}",
                str(row["modal_class"]),
            )
        console.print(table)

    ruled_out = excluded_outcomes(dataset.X_train, dataset.y_train, min_support=min_support)
    if ruled_out.empty:
        console.print("No feature value rules an outcome out. Nothing excludes the target.")
    else:
        excluded_table = Table(title="Values that make an outcome impossible")
        for column in ("feature", "value", "excluded", "records", "expected"):
            excluded_table.add_column(column, justify="right" if column != "feature" else "left")
        for _, row in ruled_out.iterrows():
            excluded_table.add_row(
                str(row["feature"]),
                str(row["value"]),
                str(row["excluded_class"]),
                f"{int(row['n']):,}",
                f"{row['expected']:.1f}",
            )
        console.print(excluded_table)

    report = leakage_report(dataset.X_train, dataset.y_train, min_support=min_support)
    ranked = Table(title="How nearly each feature settles the outcome")
    for column in ("feature", "peak share", "at value", "records", "outcome"):
        ranked.add_column(column, justify="right" if column != "feature" else "left")
    for _, row in report.head(12).iterrows():
        ranked.add_row(
            str(row["feature"]),
            f"{row['max_share']:.4f}",
            str(row["at_value"]),
            f"{int(row['n_at_value']):,}",
            str(row["modal_class"]),
        )
    console.print(ranked)
    console.print(
        "A peak share of 1.0 is a definition in disguise. Anything below it is evidence, "
        "and how much depends on what the variable means."
    )


@app.command()
def export(
    experiment: str = typer.Argument(..., help="Experiment name or path to a YAML file."),
    destination: str = typer.Option("app", "--to", help="Directory to write the payload into."),
) -> None:
    """Build the static payload the explorer page reads."""
    from pathlib import Path as _Path

    from vision_zero.export import export_for_browser
    from vision_zero.modeling.config import load_experiment
    from vision_zero.modeling.dataset import build_dataset

    specification = load_experiment(experiment)
    dataset = build_dataset(specification)
    result = export_for_browser(specification, dataset, _Path(destination))

    console.print(
        f"Wrote {result.n_records:,} records over {result.n_features} features "
        f"to {result.data_path.parent} ({result.megabytes:.2f} MB compressed)"
    )
    console.print(f"Serve it with:  python -m http.server --directory {result.data_path.parent}")


@app.command()
def examples(
    experiment: str = typer.Argument(..., help="Experiment name or path to a YAML file."),
    count: int = typer.Option(5, "--count", help="How many groups to show."),
    min_records: int = typer.Option(20, "--min-records", help="Smallest group to consider."),
    lang: str = typer.Option("en", "--lang", "-l", help="'en' or 'fr' for value labels."),
    worst_case: bool = typer.Option(
        False, "--worst-case", help="Only groups holding both the mildest and severest outcome."
    ),
) -> None:
    """Show groups of identical records whose outcomes disagree.

    The limit on predictability, demonstrated rather than derived. Every record
    within a group carries the same value for every recorded variable, so any
    procedure reading only those variables answers identically for all of them.
    Where the outcomes differ, it must be wrong about some.
    """
    from vision_zero.ceiling.bayes_error import cell_labels
    from vision_zero.ceiling.examples import ambiguous_cells, describe_cell, worst_case_pairs
    from vision_zero.data.codebook import load_codebook
    from vision_zero.modeling.config import load_experiment
    from vision_zero.modeling.dataset import build_dataset

    specification = load_experiment(experiment)
    dataset = build_dataset(specification)
    identifiers = cell_labels(dataset.X_train)
    book = load_codebook()
    language: Any = lang

    finder = worst_case_pairs if worst_case else ambiguous_cells
    groups = finder(
        dataset.X_train,
        dataset.y_train,
        min_multiplicity=min_records,
        top_n=count,
        cells=identifiers,
    )
    if groups.empty:
        console.print(f"No group of at least {min_records} records qualifies.")
        return

    for _, group in groups.iterrows():
        shared = describe_cell(
            dataset.X_train,
            int(group["cell"]),
            cells=identifiers,
            codebook=book,
            lang=language,
        )
        total = int(group["n"])
        console.print()
        console.print(
            f"[bold]{total} collisions recorded identically[/bold] "
            f"(local ceiling {group['max_share']:.1%}, "
            f"{group['entropy_bits']:.2f} bits of remaining uncertainty)"
        )

        outcome = Table(show_header=True, box=None, padding=(0, 2))
        outcome.add_column("outcome", justify="left")
        outcome.add_column("records", justify="right")
        outcome.add_column("share", justify="right")
        for label in dataset.classes:
            outcome.add_row(
                str(label),
                f"{int(group[f'n_class_{label}']):,}",
                f"{group[f'p_class_{label}']:.1%}",
            )
        console.print(outcome)

        shared_table = Table(show_header=False, box=None, padding=(0, 2))
        shared_table.add_column("field", style="dim")
        shared_table.add_column("value")
        for name, value in shared.items():
            shared_table.add_row(name, value)
        console.print("  shared by all of them:")
        console.print(shared_table)

    console.print()
    console.print(
        "Every record above is indistinguishable from its neighbours on the evidence "
        "available. No model reading these variables can separate them."
    )


@app.command()
def evaluate(
    experiment: str = typer.Argument(..., help="Experiment name or path to a YAML file."),
    confirm: bool = typer.Option(
        False, "--i-am-finished-exploring", help="Required. Opens the held-out set."
    ),
) -> None:
    """Score a fitted model on the held-out set. Run this once, at the end.

    Every other command works within the training records. This one opens the
    test set, and opening it more than once quietly turns it into a validation
    set: each look lets a choice be made in light of it, and the final figure
    stops being an estimate of performance on unseen data.

    The flag is deliberately awkward to type.
    """
    if not confirm:
        console.print(
            "The held-out set is untouched. Opening it ends the exploratory phase: "
            "any decision made after looking at it makes the final number optimistic.\n"
            "Pass --i-am-finished-exploring when that is genuinely true."
        )
        raise typer.Exit(code=1)

    import numpy as np

    from vision_zero.evaluation.calibration import calibration_report
    from vision_zero.evaluation.metrics import classification_scores
    from vision_zero.modeling.baselines import Baseline, baseline_predictions
    from vision_zero.modeling.config import load_experiment
    from vision_zero.modeling.dataset import build_dataset
    from vision_zero.modeling.train import fit_final

    specification = load_experiment(experiment)
    dataset = build_dataset(specification)
    pipeline = fit_final(specification, dataset)

    probabilities = pipeline.predict_proba(dataset.X_test)
    classes = list(dataset.classes)
    predictions = np.asarray(classes)[probabilities.argmax(axis=1)]

    scores = classification_scores(
        dataset.y_test.to_numpy(),
        predictions,
        beta=specification.evaluation.beta,
        weights=specification.evaluation.weights,
        classes=classes,
    )
    baseline = baseline_predictions(
        Baseline.MAJORITY,
        dataset.y_train.to_numpy(),
        n=dataset.n_test,
        classes=np.asarray(classes),
    )
    baseline_scores = classification_scores(
        dataset.y_test.to_numpy(),
        baseline.labels,
        beta=specification.evaluation.beta,
        weights=specification.evaluation.weights,
        classes=classes,
    )

    table = Table(title=f"{specification.name}: held-out set, {dataset.n_test:,} records")
    for column in ("metric", "model", "majority baseline"):
        table.add_column(column, justify="right" if column != "metric" else "left")
    for label, value, floor in (
        ("F*", scores.f_star, baseline_scores.f_star),
        ("MCC", scores.mcc, baseline_scores.mcc),
        ("accuracy", scores.accuracy, baseline_scores.accuracy),
        ("balanced accuracy", scores.balanced_accuracy, baseline_scores.balanced_accuracy),
    ):
        table.add_row(label, f"{value:.4f}", f"{floor:.4f}")
    console.print(table)

    per_class = Table(title="Per class")
    for column in ("class", "support", "predicted", "precision", "recall", "F-beta"):
        per_class.add_column(column, justify="right")
    for index, label in enumerate(scores.classes):
        per_class.add_row(
            str(label),
            f"{int(scores.support[index]):,}",
            f"{int(scores.predicted[index]):,}",
            f"{scores.precision[index]:.4f}",
            f"{scores.recall[index]:.4f}",
            f"{scores.fbeta[index]:.4f}",
        )
    console.print(per_class)

    if scores.abandoned_classes:
        console.print(f"Classes never predicted: {list(scores.abandoned_classes)}")

    report = calibration_report(dataset.y_test.to_numpy(), probabilities, classes=classes)
    console.print(f"Brier score: {report.brier:.4f}")
    calibration = Table(title="Calibration by class")
    for column in ("class", "calibration error", "highest stated probability", "observed rate"):
        calibration.add_column(column, justify="right" if column != "class" else "left")
    for _, row in report.summary().iterrows():
        calibration.add_row(
            str(int(row["class"])),
            f"{row['calibration_error']:.4f}",
            f"{row['max_predicted']:.4f}",
            f"{row['observed_rate']:.4f}",
        )
    console.print(calibration)
    console.print(
        "A low calibration error beside a recall near zero is the finding, not a "
        "contradiction: the stated odds are right even though the labels are useless."
    )


@app.command()
def ablate(
    experiment: str = typer.Argument(..., help="Experiment name or path to a YAML file."),
    folds: int = typer.Option(3, "--folds", help="Cross-validation folds per variant."),
    subsample: int = typer.Option(60_000, "--subsample", help="Cap on training records."),
) -> None:
    """Measure what each group of variables contributes over and above the rest."""
    from vision_zero.ceiling.ablation import block_ablation, cumulative_ablation
    from vision_zero.modeling.config import load_experiment
    from vision_zero.modeling.dataset import build_dataset

    specification = load_experiment(experiment)
    dataset = build_dataset(specification)
    cap = subsample or None

    blocks = block_ablation(specification, dataset, folds=folds, subsample=cap)
    table = Table(title="Each block alone, and what it adds to the others")
    for column in ("block", "features", "F* alone", "F* added", "MCC alone", "severe recall alone"):
        table.add_column(column, justify="right" if column != "block" else "left")
    for _, row in blocks.iterrows():
        added = row.get("f_star_added")
        table.add_row(
            str(row["block"]),
            f"{int(row['n_features'])}",
            f"{row['f_star_alone']:.4f}",
            "-" if pd.isna(added) else f"{added:+.4f}",
            f"{row['mcc_alone']:+.4f}",
            f"{row['severe_recall_alone']:.4f}",
        )
    console.print(table)

    steps = cumulative_ablation(specification, dataset, folds=folds, subsample=cap)
    running = Table(title="Blocks added in order, from the unavoidable to the actionable")
    for column in ("step", "features", "F*", "change", "MCC", "severe recall"):
        running.add_column(column, justify="right" if column != "step" else "left")
    for _, row in steps.iterrows():
        delta = row["f_star_delta"]
        running.add_row(
            str(row["step"]),
            f"{int(row['n_features'])}",
            f"{row['f_star']:.4f}",
            "-" if pd.isna(delta) else f"{delta:+.4f}",
            f"{row['mcc']:+.4f}",
            f"{row['severe_recall']:.4f}",
        )
    console.print(running)
    console.print(
        "The number to read is what `environment` adds: it is the block anyone "
        "could actually change."
    )


@app.command()
def missingness(
    experiment: str = typer.Argument(..., help="Experiment name or path to a YAML file."),
) -> None:
    """Report what a blank field reveals about the outcome."""
    from vision_zero.ceiling.missingness import missing_count_profile, missingness_report
    from vision_zero.modeling.config import load_experiment
    from vision_zero.modeling.dataset import build_dataset

    specification = load_experiment(experiment)
    dataset = build_dataset(specification)
    severe = dataset.classes[-1]

    report = missingness_report(dataset.X_train, dataset.y_train)
    table = Table(title="Outcome shares when a field is blank, against when it is filled")
    for column in (
        "feature",
        "blank",
        "share",
        "P(0) blank",
        "P(0) filled",
        f"lift class {severe}",
    ):
        table.add_column(column, justify="right" if column != "feature" else "left")
    for _, row in report.iterrows():
        table.add_row(
            str(row["feature"]),
            f"{int(row['n_missing']):,}",
            f"{row['share_missing']:.3f}",
            f"{row['p_class_0_missing']:.4f}",
            f"{row['p_class_0_present']:.4f}",
            f"{row[f'lift_class_{severe}']:.3f}",
        )
    console.print(table)

    profile = missing_count_profile(dataset.X_train, dataset.y_train)
    counts = Table(title="Outcome shares by how many fields were left blank")
    counts.add_column("blanks", justify="right")
    counts.add_column("records", justify="right")
    for label in dataset.classes:
        counts.add_column(f"P(class {label})", justify="right")
    for _, row in profile.iterrows():
        counts.add_row(
            str(int(row["n_missing"])),
            f"{int(row['n_records']):,}",
            *[f"{float(row[f'p_class_{label}']):.4f}" for label in dataset.classes],
        )
    console.print(counts)
    console.print(
        "A trend here is a fact about report completeness, not about roads. "
        "It is available to a model and useless for prevention."
    )


@data_app.command("describe")
def data_describe(
    term: str = typer.Argument(..., help="Variable name, e.g. SEVERITY."),
    lang: str = typer.Option("en", "--lang", "-l", help="'en' or 'fr'."),
) -> None:
    """Print the codebook entry for a variable."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    app()
