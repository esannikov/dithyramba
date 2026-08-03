"""CLI smoke checks for the scoped concept Lens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from dithyramba.cli import app
from dithyramba.contracts import canonical_json_bytes

from ..ontology.test_candidate_ontology import build_manifest


def test_concept_lens_cli_starts_loopback_with_valid_projection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    path = tmp_path / "ontology.json"
    path.write_bytes(canonical_json_bytes(build_manifest().model_dump(mode="json")))
    calls: list[dict[str, object]] = []

    def fake_run(application: object, **kwargs: object) -> None:
        calls.append({"application": application, **kwargs})

    monkeypatch.setattr("dithyramba.cli.uvicorn.run", fake_run)
    result = CliRunner().invoke(
        app,
        ["lens", "concepts", "--projection", str(path), "--port", "8362"],
    )

    assert result.exit_code == 0
    assert "lens: http://127.0.0.1:8362" in result.stdout
    assert "lens_mode: concepts" in result.stdout
    assert "candidate-only read-only projection" in result.stdout
    assert calls[0]["host"] == "127.0.0.1"


def test_concept_lens_cli_rejects_relative_projection() -> None:
    result = CliRunner().invoke(
        app,
        ["lens", "concepts", "--projection", "ontology.json"],
    )

    assert result.exit_code == 2
    assert "projection must be an absolute path" in result.output
