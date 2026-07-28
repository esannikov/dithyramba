"""Secure loopback server launcher CLI smoke tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dithyramba.api import bearer_token_for
from dithyramba.cli import app
from dithyramba.library import LibraryConfig
from dithyramba.persistence import initialize_library

runner = CliRunner()


def test_serve_pins_library_loopback_origin_and_fresh_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_home = tmp_path / "data"
    with initialize_library(
        LibraryConfig(name="Serve CLI"),
        data_root=data_home,
    ) as repository:
        library_id = repository.library_id

    captured: dict[str, object] = {}

    def fake_run(application: object, **kwargs: object) -> None:
        captured["application"] = application
        captured.update(kwargs)

    monkeypatch.setattr("dithyramba.cli.uvicorn.run", fake_run)
    result = runner.invoke(
        app,
        [
            "serve",
            "--library",
            library_id,
            "--data-home",
            str(data_home),
            "--port",
            "8765",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "viewer: http://127.0.0.1:8765\n"
    match = re.fullmatch(r"mutation_bearer_token: ([A-Za-z0-9_-]{43})\n", result.stderr)
    assert match is not None
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["access_log"] is False
    assert captured["server_header"] is False
    assert captured["date_header"] is False
    assert bearer_token_for(captured["application"]) == match.group(1)  # type: ignore[arg-type]


def test_serve_requires_explicit_library_and_data_home() -> None:
    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 2
    assert "Missing option" in result.output
