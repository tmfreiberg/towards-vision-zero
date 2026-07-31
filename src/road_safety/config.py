"""Runtime configuration for the project.

This module holds *runtime* configuration only: where the data lives, which seed
to use, how verbose to be. Experiment configuration — features, targets,
hyperparameters, evaluation weights — lives in YAML under ``conf/`` and is
loaded separately, so that changing an experiment never means changing an
environment, and so that the configuration which produced a given result is a
reviewable file rather than an ambient property of the machine it ran on.

Settings are read, in decreasing order of precedence, from:

1. arguments passed to :class:`Settings` directly,
2. environment variables prefixed ``ROAD_SAFETY_``,
3. a ``.env`` file in the working directory,
4. the defaults below.

Nothing here mutates :data:`sys.path` or performs any import-time side effect
beyond reading the environment. The package is installed; module resolution is
the installer's problem, not this module's.

Examples
--------
>>> from road_safety.config import get_settings
>>> settings = get_settings()
>>> settings.raw_data_dir.name
'raw'

Point the project somewhere else for a single process::

    ROAD_SAFETY_PROJECT_ROOT=/tmp/scratch road-safety data prepare
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def _default_project_root() -> Path:
    """Locate the project root by walking up from this file.

    Used only when ``ROAD_SAFETY_PROJECT_ROOT`` is unset. The package lives at
    ``src/road_safety``, so the root is two directories above the package.

    A marker-file search would tolerate an editable install relocated after
    installation, which is not a case worth the extra failure mode.

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Validated runtime settings.

    Prefer :func:`get_settings` over constructing this directly; it caches, so
    settings resolve once per process and every caller observes one object.

    Attributes
    ----------
    project_root
        Repository root. All other paths derive from it unless overridden.
    random_seed
        Global seed. Every stochastic operation takes its seed from here or
        from an explicit argument; nothing draws from unseeded global state.
        Reproducibility is a tested property, not a convention.
    log_level
        Threshold for the package logger.
    strict_data
        When true, a missing or malformed input file raises. When false, it
        warns and continues.

        The default is true. Partial ingestion has to fail loudly: a year of
        records dropped during loading is invisible in every downstream number,
        and an analysis whose conclusions concern the data cannot afford a
        silent completeness failure. Set this false only for deliberate
        exploration of an incomplete extract.
    """

    model_config = SettingsConfigDict(
        env_prefix="ROAD_SAFETY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    project_root: Path = Field(default_factory=_default_project_root)
    random_seed: int = 42
    log_level: LogLevel = "INFO"
    strict_data: bool = True

    data_dir: Path | None = None
    models_dir: Path | None = None
    reports_dir: Path | None = None

    @field_validator("project_root", "data_dir", "models_dir", "reports_dir")
    @classmethod
    def _resolve(cls, value: Path | None) -> Path | None:
        """Expand ``~`` and resolve to an absolute path.

        A relative path read from the environment would otherwise resolve
        against whichever directory a command happened to be run from.
        """
        if value is None:
            return None
        return value.expanduser().resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_root(self) -> Path:
        """Root of the data tree."""
        return self.data_dir or self.project_root / "data"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_data_dir(self) -> Path:
        """Source CSVs. Read-only: nothing in the package writes here."""
        return self.data_root / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_data_dir(self) -> Path:
        """Compiled Parquet and its provenance manifest. Regenerable."""
        return self.data_root / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_artifacts_dir(self) -> Path:
        """Serialised pipelines and metric records. Regenerable."""
        return self.models_dir or self.project_root / "artifacts" / "models"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def report_dir(self) -> Path:
        """Generated figures and tables consumed by the analysis documents."""
        return self.reports_dir or self.project_root / "artifacts" / "reports"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def conf_dir(self) -> Path:
        """Experiment configuration in YAML."""
        return self.project_root / "conf"

    def ensure_writable_dirs(self) -> None:
        """Create the output directories if they do not exist.

        Output directories only. Input directories are never created: a missing
        ``data/raw`` means the data is absent, and conjuring an empty one turns
        a clear failure here into a confusing one further downstream.
        """
        for path in (self.processed_data_dir, self.model_artifacts_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, resolving them on first call.

    Cached so every caller sees one object and the environment is read once.
    Call ``get_settings.cache_clear()`` in a test that manipulates the
    environment.

    Returns
    -------
    Settings
        The validated settings for this process.

    Examples
    --------
    >>> from road_safety.config import get_settings
    >>> get_settings() is get_settings()
    True
    """
    return Settings()
