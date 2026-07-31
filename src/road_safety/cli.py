"""Command-line interface.

Every long-running operation is reachable from here. That keeps the analysis
documents thin — a document calls a function or a command, it does not contain
the analysis — and it gives the end-to-end tests something meaningful to invoke.

Commands are grouped by concern::

    road-safety data prepare      compile raw CSVs into the Parquet intermediate
    road-safety data describe     print a codebook entry, in English or French
    road-safety train             fit a model from an experiment config
    road-safety evaluate          evaluate a fitted model
    road-safety ceiling           estimate the predictability ceiling
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from road_safety import __version__
from road_safety.config import get_settings
from road_safety.logging import configure_logging

app = typer.Typer(
    name="road-safety",
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

    table = Table(title="road-safety configuration", show_lines=False)
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
def data_prepare() -> None:
    """Compile the raw SAAQ CSVs into a single Parquet file."""
    raise NotImplementedError


@data_app.command("describe")
def data_describe(
    term: str = typer.Argument(..., help="Variable name, e.g. SEVERITY."),
    lang: str = typer.Option("en", "--lang", "-l", help="'en' or 'fr'."),
) -> None:
    """Print the codebook entry for a variable."""
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    app()
