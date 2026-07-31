"""Shared fixtures.

Nothing here reads the real SAAQ data. Tests that need it are marked
``requires_data`` and skip when it is absent, so the suite runs anywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from vision_zero.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Reset the settings cache around every test.

    ``get_settings`` is ``lru_cache``d, so a test that sets an environment
    variable would otherwise leak its configuration into whatever ran next.
    Autouse removes the whole class of order-dependent failure that
    ``pytest-randomly`` exists to surface.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings rooted at an empty temporary directory."""
    return Settings(project_root=tmp_path)
