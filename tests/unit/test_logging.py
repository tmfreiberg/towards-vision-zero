"""Tests for logging configuration."""

from __future__ import annotations

import logging

from vision_zero.logging import configure_logging, get_logger


def test_names_outside_the_package_are_placed_under_it() -> None:
    assert get_logger("scratch").name == "vision_zero.scratch"
    assert get_logger("vision_zero.data.load").name == "vision_zero.data.load"


def test_configuration_is_idempotent() -> None:
    configure_logging(level="INFO", force=True)
    before = len(logging.getLogger("vision_zero").handlers)
    configure_logging(level="INFO")
    configure_logging(level="INFO")
    assert len(logging.getLogger("vision_zero").handlers) == before


def test_package_logger_does_not_propagate_to_root() -> None:
    configure_logging(level="INFO", force=True)
    assert logging.getLogger("vision_zero").propagate is False
