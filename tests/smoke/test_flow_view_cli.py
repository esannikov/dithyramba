"""CLI smoke test for the lightweight flow server."""

from __future__ import annotations

from typing import cast

import pytest
from fastapi import FastAPI
from typer.testing import CliRunner

from dithyramba.cli import app


def test_flow_view_cli_pins_loopback_and_read_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(application: FastAPI, **kwargs: object) -> None:
        captured["application"] = application
        captured.update(kwargs)

    monkeypatch.setattr("dithyramba.cli.uvicorn.run", fake_run)
    result = CliRunner().invoke(app, ["lens", "flow", "--port", "8767"])

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "lens: http://127.0.0.1:8767\nlens_mode: flow\nmode: read-only static process map\n"
    )
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8767
    assert captured["access_log"] is False
    assert captured["server_header"] is False
    assert captured["date_header"] is False
    application = cast(FastAPI, captured["application"])
    assert application.state.flow_view_config.version == "1.0"
