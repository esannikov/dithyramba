"""Unit tests for the canonical deprecated-identifier gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dithyramba.tooling.terminology import (
    EXTERNAL_QUOTE_END,
    EXTERNAL_QUOTE_START,
    exit_code_for_findings,
    findings_as_jsonable,
    iter_text_files,
    main,
    scan_file,
    scan_paths,
    scan_text,
)


def test_exact_identifier_is_reported_with_source_address_and_replacement() -> None:
    findings = scan_text("prefix\nDecision, then Claim.\n", path="docs/design.md")

    assert findings_as_jsonable(reversed(findings)) == [
        {
            "rule": "deprecated-identifier",
            "path": "docs/design.md",
            "line": 2,
            "column": 1,
            "identifier": "Decision",
            "replacement": "ReviewDecision",
            "text": "Decision, then Claim.",
        },
        {
            "rule": "deprecated-identifier",
            "path": "docs/design.md",
            "line": 2,
            "column": 16,
            "identifier": "Claim",
            "replacement": "Statement",
            "text": "Decision, then Claim.",
        },
    ]


def test_canonical_compounds_are_not_substring_matches() -> None:
    text = "ReviewDecision EvidenceGap Claimant PositionMap decision review_decision"

    assert scan_text(text) == ()


def test_replacement_table_allows_only_the_replacement_column() -> None:
    text = """\
| Канонічна назва | Замінює | Значення |
|---|---|---|
| `ReviewDecision` | `Decision` | Correct historical alias. |
| `Statement` | `Claim` | Decision must not appear in this explanation. |
"""

    findings = scan_text(text, path="TERMINOLOGY.md")

    assert [(finding.identifier, finding.line) for finding in findings] == [("Decision", 4)]


def test_canonical_deprecated_registry_block_is_allowed() -> None:
    text = """\
## Deprecated-term gate

```text
Decision Claim MemorySpace
```
"""

    assert scan_text(text, path="TERMINOLOGY.md") == ()
    assert [item.identifier for item in scan_text(text, path="other.md")] == [
        "Decision",
        "Claim",
        "MemorySpace",
    ]


def test_closed_external_quote_is_allowed_without_hiding_surrounding_text() -> None:
    text = (
        f"Decision before. {EXTERNAL_QUOTE_START} Claim in quote. "
        f"{EXTERNAL_QUOTE_END} Position after."
    )

    findings = scan_text(text, path="docs/analysis.md")

    assert [finding.identifier for finding in findings] == ["Decision", "Position"]


def test_unclosed_external_quote_does_not_silence_the_file() -> None:
    text = f"{EXTERNAL_QUOTE_START}\nDecision\n"

    assert [finding.identifier for finding in scan_text(text)] == ["Decision"]


def test_migration_history_fixture_is_allowed_but_live_migration_is_not() -> None:
    historical = scan_text(
        "CREATE TABLE MemoryProject (...);",
        path="tests/fixtures/migration_history/v0.sql",
    )
    live = scan_text("CREATE TABLE MemoryProject (...);", path="migrations/0001.sql")

    assert historical == ()
    assert [finding.identifier for finding in live] == ["MemoryProject"]


def test_scan_file_and_exit_code_helper(tmp_path: Path) -> None:
    clean_path = tmp_path / "clean.md"
    dirty_path = tmp_path / "dirty.md"
    clean_path.write_text("ReviewDecision\n", encoding="utf-8")
    dirty_path.write_text("Decision\n", encoding="utf-8")

    clean = scan_file(clean_path)
    dirty = scan_file(dirty_path)

    assert clean == ()
    assert exit_code_for_findings(clean) == 0
    assert exit_code_for_findings(dirty) == 1


def test_repository_scan_is_stable_and_ignores_generated_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "src" / "clean.py").write_text("ReviewDecision\n", encoding="utf-8")
    (tmp_path / "src" / "dirty.py").write_text("Decision\n", encoding="utf-8")
    (tmp_path / ".venv" / "ignored.py").write_text("Claim\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    files = iter_text_files([tmp_path])
    findings = scan_paths([tmp_path])

    assert [path.name for path in files] == ["clean.py", "dirty.py"]
    assert [(item.identifier, Path(item.path).name) for item in findings] == [
        ("Decision", "dirty.py")
    ]


def test_main_emits_json_and_failure_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "dirty.md"
    path.write_text("Decision\n", encoding="utf-8")

    assert main([str(path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["findings"][0]["replacement"] == "ReviewDecision"
