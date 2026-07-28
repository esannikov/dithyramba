"""Reading Room loopback launcher CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from typer.testing import CliRunner

from dithyramba.cli import app

runner = CliRunner()


def _json(arguments: list[str]) -> dict[str, object]:
    result = runner.invoke(app, [*arguments, "--json"])
    assert result.exit_code == 0, result.output
    return cast(dict[str, object], json.loads(result.stdout))


def test_reading_room_pins_explicit_scope_and_runs_on_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_home = tmp_path / "data"
    source_root = tmp_path / "corpus"
    source_root.mkdir()
    (source_root / "memory.md").write_text(
        "# Memory\n\nA source-grounded statement.\n",
        encoding="utf-8",
    )
    library = _json(
        [
            "library",
            "init",
            "Reading Room CLI",
            "--data-home",
            str(data_home),
            "--source-root",
            str(source_root),
        ]
    )
    library_id = cast(str, library["library_id"])
    collection = _json(
        [
            "collection",
            "add",
            "Research corpus",
            "--library",
            library_id,
            "--root",
            str(source_root),
            "--kind",
            "corpus",
            "--data-home",
            str(data_home),
        ]
    )
    collection_id = cast(str, collection["collection_id"])
    policy = _json(
        [
            "access-policy",
            "create",
            "Research",
            "--library",
            library_id,
            "--purpose",
            "research",
            "--allow-collection",
            collection_id,
            "--data-home",
            str(data_home),
        ]
    )
    _json(
        [
            "index",
            "--library",
            library_id,
            "--collection",
            collection_id,
            "--data-home",
            str(data_home),
        ]
    )
    snapshot = _json(
        [
            "collection",
            "freeze",
            "--library",
            library_id,
            "--collection",
            collection_id,
            "--data-home",
            str(data_home),
        ]
    )

    captured: dict[str, object] = {}

    def fake_run(application: FastAPI, **kwargs: object) -> None:
        captured["application"] = application
        captured.update(kwargs)

    monkeypatch.setattr("dithyramba.cli.uvicorn.run", fake_run)
    result = runner.invoke(
        app,
        [
            "reading-room",
            "--library",
            library_id,
            "--snapshot",
            cast(str, snapshot["corpus_snapshot_id"]),
            "--access-policy",
            cast(str, policy["access_policy_id"]),
            "--collection",
            collection_id,
            "--purpose",
            "research",
            "--data-home",
            str(data_home),
            "--port",
            "8766",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "reading_room: http://127.0.0.1:8766\n"
        "projection_json: http://127.0.0.1:8766/projection.json\n"
        "mode: read-only\n"
    )
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8766
    assert captured["access_log"] is False
    assert captured["server_header"] is False
    assert captured["date_header"] is False
    application = cast(FastAPI, captured["application"])
    config = application.state.reading_room_config
    assert config.loopback.library_id == library_id
    assert config.scope.collection_ids == (collection_id,)
    assert config.scope.purpose == "research"


def test_reading_room_requires_complete_explicit_scope() -> None:
    result = runner.invoke(app, ["reading-room"])

    assert result.exit_code == 2
    assert "Missing option" in result.output
