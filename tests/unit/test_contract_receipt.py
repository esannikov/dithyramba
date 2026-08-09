"""Release contract receipt tests."""

from pathlib import Path

import pytest
from scripts.contract_receipt import build_contract_receipt

ROOT = Path(__file__).resolve().parents[2]
COMMIT_SHA = "a" * 40


def test_contract_receipt_is_deterministic_and_covers_release_surfaces() -> None:
    first = build_contract_receipt(source_root=ROOT, commit_sha=COMMIT_SHA)
    second = build_contract_receipt(source_root=ROOT, commit_sha=COMMIT_SHA)

    assert first == second
    assert len(str(first["receipt_hash"])) == 64
    cli_commands = first["cli_commands"]
    assert isinstance(cli_commands, list)
    assert "dithyramba mcp" in cli_commands
    assert first["sqlite"]["status"] == "ok"  # type: ignore[index]
    assert first["documentation"]["broken_local_links"] == []  # type: ignore[index]
    assert len(first["mcp"]["tools"]) == 7  # type: ignore[index]


@pytest.mark.parametrize("commit_sha", ("abc", "G" * 40))
def test_contract_receipt_rejects_invalid_commit_identity(commit_sha: str) -> None:
    with pytest.raises(ValueError, match="commit SHA"):
        build_contract_receipt(source_root=ROOT, commit_sha=commit_sha)
