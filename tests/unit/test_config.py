"""Tests for runtime settings resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vision_zero.config import Settings, get_settings


def test_derived_paths_hang_off_project_root(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    assert settings.raw_data_dir == tmp_path / "data" / "raw"
    assert settings.processed_data_dir == tmp_path / "data" / "processed"
    assert settings.conf_dir == tmp_path / "conf"


def test_explicit_data_dir_overrides_derivation(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    settings = Settings(project_root=tmp_path, data_dir=elsewhere)
    assert settings.raw_data_dir == elsewhere / "raw"


def test_relative_paths_are_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative path in the environment must not depend on the caller's cwd later."""
    (tmp_path / "nested").mkdir()
    monkeypatch.chdir(tmp_path)
    settings = Settings(project_root=Path("nested"))
    assert settings.project_root == (tmp_path / "nested").resolve()


def test_environment_is_read_with_the_expected_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TVZ_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("TVZ_RANDOM_SEED", "1234")
    settings = Settings()
    assert settings.project_root == tmp_path.resolve()
    assert settings.random_seed == 1234


def test_unknown_settings_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(not_a_setting=1)  # type: ignore[call-arg]


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(log_level="CHATTY")  # type: ignore[arg-type]


def test_settings_are_immutable(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    with pytest.raises(ValidationError):
        settings.random_seed = 7  # type: ignore[misc]


def test_strict_data_defaults_to_true() -> None:
    """Failing loudly on bad input must be the default, not an opt-in."""
    assert Settings().strict_data is True


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_ensure_writable_dirs_creates_outputs_but_not_inputs(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.ensure_writable_dirs()
    assert settings.processed_data_dir.is_dir()
    assert settings.model_artifacts_dir.is_dir()
    assert settings.report_dir.is_dir()
    assert not settings.raw_data_dir.exists(), "input dirs must not be conjured into existence"
