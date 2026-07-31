"""Logging setup for the package.

Library code logs and returns values; it does not print. Printing from a library
puts output beyond the caller's reach — it cannot be suppressed, redirected,
levelled, or asserted on — and turns functions that ought to be pure into ones
with I/O side effects.

:mod:`road_safety.cli` is the single exception: it is a user interface and is
allowed to write to stdout. Ruff rule ``T20`` enforces the rest, with a per-file
exemption for the CLI.

Examples
--------
>>> from road_safety.logging import configure_logging, get_logger
>>> configure_logging(level="WARNING")
>>> log = get_logger(__name__)
>>> log.debug("not emitted at WARNING")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.logging import RichHandler

if TYPE_CHECKING:
    from road_safety.config import LogLevel

__all__ = ["configure_logging", "get_logger"]

_ROOT_LOGGER_NAME = "road_safety"
_configured = False


def configure_logging(level: LogLevel | str | None = None, *, force: bool = False) -> None:
    """Attach a Rich handler to the package logger.

    Idempotent: repeated calls are no-ops unless ``force`` is set. Notebooks and
    rendered documents re-import freely, and stacking handlers duplicates every
    subsequent line.

    Parameters
    ----------
    level
        Threshold. Defaults to :attr:`~road_safety.config.Settings.log_level`.
    force
        Reconfigure even if already configured, replacing existing handlers.

    Notes
    -----
    Only the ``road_safety`` logger is touched. The root logger is left alone,
    so importing this package does not alter logging behaviour for anything
    else in the process.
    """
    global _configured
    if _configured and not force:
        return

    from road_safety.config import get_settings

    resolved = str(level) if level is not None else get_settings().log_level

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if force:
        logger.handlers.clear()

    handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(handler)
    logger.setLevel(resolved)
    logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the package namespace.

    Parameters
    ----------
    name
        Usually ``__name__``. A name outside the package namespace is placed
        beneath it, so the package's level and handlers always apply.

    Returns
    -------
    logging.Logger
        A logger beneath ``road_safety``.

    Examples
    --------
    >>> get_logger("road_safety.data.load").name
    'road_safety.data.load'
    >>> get_logger("scratch").name
    'road_safety.scratch'
    """
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
