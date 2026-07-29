"""Operator smoke tests for explicit local model provisioning."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import dithyramba.recall.provisioning as provisioning_module
from dithyramba.cli import app

runner = CliRunner()


def test_model_help_exposes_explicit_provisioning() -> None:
    result = runner.invoke(app, ["model", "--help"])

    assert result.exit_code == 0
    assert "pinned optional local models" in result.stdout
    assert "provision" in result.stdout


def test_unknown_profile_and_implicit_network_fail_cleanly(tmp_path: Path) -> None:
    unknown = runner.invoke(
        app,
        ["model", "provision", "unknown", "--data-home", str(tmp_path)],
    )
    assert unknown.exit_code == 1
    assert "expected e5-small or harrier-270m" in unknown.stderr
    assert "Traceback" not in unknown.stderr

    missing = runner.invoke(
        app,
        ["model", "provision", "e5-small", "--data-home", str(tmp_path)],
    )
    assert missing.exit_code == 1
    assert "explicit allow_network=True" in missing.stderr
    assert not (tmp_path / "models").exists()


def test_provision_then_offline_verify_is_machine_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    fake_hf = tmp_path / "isolated-hf"

    def fake_run(
        arguments: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(arguments)
        target = Path(arguments[arguments.index("--local-dir") + 1])
        for relative_path in arguments[3 : arguments.index("--revision")]:
            path = target / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{relative_path}\n", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(provisioning_module, "_resolve_hf_executable", lambda: str(fake_hf))
    monkeypatch.setattr(subprocess, "run", fake_run)
    first = runner.invoke(
        app,
        [
            "model",
            "provision",
            "e5-small",
            "--data-home",
            str(tmp_path),
            "--allow-network",
            "--json",
        ],
    )
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.stdout)
    assert first_payload["model_id"] == "intfloat/multilingual-e5-small"
    assert first_payload["revision"] == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert first_payload["file_count"] == 9
    assert first_payload["network_authorized"] is True
    assert calls and calls[0][:2] == (str(fake_hf), "download")

    def forbidden_run(
        _arguments: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("offline verification must not invoke hf")

    monkeypatch.setattr(subprocess, "run", forbidden_run)
    second = runner.invoke(
        app,
        [
            "model",
            "provision",
            "e5-small",
            "--data-home",
            str(tmp_path),
            "--json",
        ],
    )
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.stdout)
    assert second_payload["receipt_hash"] == first_payload["receipt_hash"]
    assert second_payload["network_authorized"] is False


@pytest.mark.parametrize(
    ("alias", "model_id", "revision"),
    [
        (
            "harrier-270m",
            "microsoft/harrier-oss-v1-270m",
            "31de22b673913c7d658c0f03f792d77c2dcf8ebd",
        ),
    ],
)
def test_embedding_profiles_are_explicitly_pinned_and_machine_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
    model_id: str,
    revision: str,
) -> None:
    fake_hf = tmp_path / "isolated-hf"

    def fake_run(
        arguments: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        target = Path(arguments[arguments.index("--local-dir") + 1])
        for relative_path in arguments[3 : arguments.index("--revision")]:
            path = target / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{relative_path}\n", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(provisioning_module, "_resolve_hf_executable", lambda: str(fake_hf))
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = runner.invoke(
        app,
        [
            "model",
            "provision",
            alias,
            "--data-home",
            str(tmp_path),
            "--allow-network",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["model_id"] == model_id
    assert payload["revision"] == revision
    assert payload["file_count"] == 9
    assert payload["profile_id"].startswith("embedding_profile_")
