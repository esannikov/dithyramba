"""Reasoning-check CLI smoke coverage."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dithyramba.cli import app

from ..unit.test_reasoning_closure import _accepted_trace

runner = CliRunner()


def test_reasoning_check_emits_replayable_closure(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    trace, case_set, entailment = _accepted_trace(fixture_root)
    trace_path = tmp_path / "trace.json"
    case_set_path = tmp_path / "case-set.json"
    entailment_path = tmp_path / "entailment.json"
    trace_path.write_text(trace.model_dump_json(), encoding="utf-8")
    case_set_path.write_text(case_set.model_dump_json(), encoding="utf-8")
    entailment_path.write_text(entailment.model_dump_json(), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "reasoning-check",
            "--trace",
            str(trace_path),
            "--case-set",
            str(case_set_path),
            "--entailment",
            str(entailment_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["decision"] == "passed"
    assert payload["review_eligible"] is True
    assert payload["trace_id"] == trace.trace_id


def test_reasoning_check_requires_absolute_paths() -> None:
    result = runner.invoke(
        app,
        [
            "reasoning-check",
            "--trace",
            "trace.json",
            "--case-set",
            "case-set.json",
            "--entailment",
            "entailment.json",
        ],
    )

    assert result.exit_code != 0
    assert "must be absolute" in result.output
