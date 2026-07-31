"""End-to-end tests for the CLI.

These assert that every command is reachable and that stubs fail rather than
silently appearing to succeed. A command that cannot even be imported is caught
here and nowhere else.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vision_zero import __version__
from vision_zero.cli import app
from vision_zero.config import get_settings

runner = CliRunner()


def test_help_exits_cleanly() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "tvz" in result.stdout


def test_version_matches_package_metadata() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_info_reports_resolved_paths() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "project root" in result.stdout


def test_unimplemented_commands_fail_loudly() -> None:
    """A stub must not silently succeed and appear to have done work."""
    result = runner.invoke(app, ["data", "describe", "SEVERITY"])
    assert result.exit_code != 0


def test_data_prepare_compiles_the_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: a real invocation writing real artifacts."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    shutil.copytree(fixture, root / "data" / "raw")

    monkeypatch.setenv("TVZ_PROJECT_ROOT", str(root))
    get_settings.cache_clear()

    result = runner.invoke(app, ["data", "prepare", "--start", "2011", "--end", "2013"])
    assert result.exit_code == 0, result.output
    assert (root / "data" / "processed" / "saaq.parquet").is_file()
    assert (root / "data" / "processed" / "saaq.manifest.json").is_file()


def test_data_prepare_skips_when_up_to_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    shutil.copytree(fixture, root / "data" / "raw")

    monkeypatch.setenv("TVZ_PROJECT_ROOT", str(root))
    get_settings.cache_clear()

    runner.invoke(app, ["data", "prepare", "--start", "2011", "--end", "2013"])
    second = runner.invoke(app, ["data", "prepare", "--start", "2011", "--end", "2013"])
    assert "up to date" in second.stdout


def test_evaluate_refuses_without_the_confirmation_flag() -> None:
    """The held-out set must not be openable by accident."""
    result = runner.invoke(app, ["evaluate", "montreal_ternary"])
    assert result.exit_code != 0
    assert "untouched" in result.stdout
