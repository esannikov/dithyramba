"""Release-facing checks for the thin Codex orchestration skill."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = ROOT / "skills" / "dithyramba"


def test_skill_is_small_complete_and_source_repository_safe() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    openai = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    references = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((SKILL_ROOT / "references").glob("*.md"))
    }

    assert skill.startswith("---\nname: dithyramba\ndescription:")
    assert len(skill.splitlines()) < 500
    assert "TODO" not in skill
    assert set(references) == {
        "evidence-contract.md",
        "lifecycle.md",
        "mcp-tools.md",
    }
    assert 'default_prompt: "Use $dithyramba' in openai
    assert "/Users/" not in "\n".join((skill, openai, *references.values()))


def test_skill_preserves_the_seven_tool_and_human_review_boundary() -> None:
    mcp = (SKILL_ROOT / "references" / "mcp-tools.md").read_text(encoding="utf-8")
    evidence = (SKILL_ROOT / "references" / "evidence-contract.md").read_text(encoding="utf-8")

    for tool in (
        "open_session",
        "recall",
        "session_context",
        "prepare_answer",
        "record_draft",
        "record_gap",
        "reject_path",
    ):
        assert f"`{tool}`" in mcp
    assert "does not expose source deletion" in mcp
    assert "Human-reviewed decision" in evidence
    assert "does not prove that a source is true" in " ".join(evidence.split())
