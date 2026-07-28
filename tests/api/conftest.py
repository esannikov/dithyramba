"""Shared persisted API fixture."""

from pathlib import Path

import pytest

from .support import ApiWorld, build_api_world


@pytest.fixture
def api_world(tmp_path: Path) -> ApiWorld:
    return build_api_world(tmp_path)
