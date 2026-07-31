from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dithyramba.atlas import migrate_legacy_research_atlas
from dithyramba.atlas.migration import (
    _IdentifierMap,
    _mapped_state,
    _migrate_source_binding,
    _safe_identifier,
)
from dithyramba.cli import app
from dithyramba.contracts import sha256_hex


def _legacy_payload() -> dict[str, object]:
    return {
        "schema_id": "dithyramba.research_atlas/1.0",
        "atlas_id": "mars:legacy",
        "case_id": "mars_case",
        "title": "Mars Lens",
        "subject": "Mars",
        "subtitle": "A migration fixture",
        "generated_at": "2026-07-31T12:00:00Z",
        "language": "uk",
        "read_only": True,
        "corpus": {
            "label": "Mars",
            "source_count": 1,
            "source_family_count": 1,
            "scope_note": "One source.",
            "cutoff_note": "Frozen fixture.",
        },
        "sources": [
            {
                "source_id": "ntrs:one",
                "title": "Source",
                "creator": "NASA",
                "source_kind": "institutional_record",
                "voice_kind": "institutional",
                "date_label": "2026-07-31T12:30:00+00:00",
                "independence_group": "ntrs:one",
                "verification_state": "full_text_ready",
                "artifact_path": "artifacts/source.md",
                "rights_note": "Public.",
            }
        ],
        "evidence": [
            {
                "evidence_id": "e:one",
                "source_id": "ntrs:one",
                "locator": "legacy:chars",
                "excerpt": "direct evidence",
                "role": "supports",
                "asserting_voice": "NASA",
                "source_fragment_id": "fragment:one",
                "fragment_text_sha256": "a" * 64,
                "source_address": {
                    "kind": "legacy",
                    "char_start": 13,
                    "char_end": 28,
                },
            }
        ],
        "questions": [
            {
                "question_id": "q:one",
                "prompt": "What is known?",
                "short_answer": "The direct evidence is limited.",
                "state": "unknown",
                "evidence_ids": ["e:one"],
                "tags": [],
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "h:one",
                "title": "A working model",
                "synthesis": "Direct evidence supports a cautious model.",
                "state": "model",
                "evidence_ids": ["e:one"],
                "counterevidence_ids": [],
                "question_ids": ["q:one"],
            }
        ],
        "relations": [],
        "timeline": [],
        "gaps": [],
        "methodology_note": "Historical fixture.",
    }


def test_migration_normalizes_states_ids_bindings_and_exact_traces(tmp_path: Path) -> None:
    artifact_root = tmp_path / "data"
    artifact = artifact_root / "artifacts" / "source.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        "Heading line\nThe direct evidence is limited.\nTail line\n", encoding="utf-8"
    )
    payload = _legacy_payload()
    original = deepcopy(payload)

    result = migrate_legacy_research_atlas(
        payload,
        artifact_root=artifact_root,
        trace_overrides={
            "q:one": [
                {
                    "text": "direct evidence",
                    "kind": "fact",
                    "evidence_ids": ["e:one"],
                }
            ],
            "h:one": [
                {
                    "text": "Direct evidence",
                    "kind": "synthesis",
                    "evidence_ids": ["e:one"],
                }
            ],
        },
    )

    assert payload == original
    assert result.manifest.atlas_id == "mars_legacy"
    assert result.manifest.sources[0].source_id == "ntrs_one"
    assert result.manifest.sources[0].verification_state.value == "verified"
    assert result.manifest.sources[0].date_label == "2026-07-31"
    assert result.manifest.questions[0].state.value == "open"
    assert result.manifest.hypotheses[0].state.value == "working"
    assert result.manifest.questions[0].trace_spans[0].text == "direct evidence"
    evidence = result.manifest.evidence[0]
    assert evidence.source_address is not None
    assert evidence.source_address.line_start == 2
    assert evidence.source_address.line_end == 2
    assert evidence.fragment_text_sha256 == sha256_hex(b"The direct evidence is limited.")
    assert result.receipt.exact_binding_count == 1
    assert result.receipt.trace_span_count == 2
    assert result.receipt.input_sha256 != result.receipt.output_sha256


def test_migration_fails_on_ambiguous_trace_override() -> None:
    payload = _legacy_payload()
    payload["questions"][0]["short_answer"] = "same same"  # type: ignore[index]

    with pytest.raises(ValueError, match="must occur exactly once"):
        migrate_legacy_research_atlas(
            payload,
            trace_overrides={
                "q:one": [
                    {
                        "text": "same",
                        "kind": "fact",
                        "evidence_ids": ["e:one"],
                    }
                ]
            },
        )


def test_migration_omits_unverifiable_exact_binding() -> None:
    result = migrate_legacy_research_atlas(_legacy_payload())

    evidence = result.manifest.evidence[0]
    assert evidence.source_address is None
    assert evidence.source_fragment_id is None
    assert evidence.fragment_text_sha256 is None
    assert result.receipt.exact_binding_count == 0
    assert result.receipt.omitted_binding_count == 1


def test_atlas_migrate_cli_writes_manifest_and_receipt(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    output = tmp_path / "atlas.json"

    result = CliRunner().invoke(
        app,
        [
            "atlas-migrate",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert (tmp_path / "MIGRATION_RECEIPT.json").is_file()
    assert "sources: 1" in result.stdout


def test_atlas_migrate_cli_accepts_exact_binding_and_trace_inputs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "data"
    artifact = artifact_root / "artifacts" / "source.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        "Heading line\nThe direct evidence is limited.\nTail line\n",
        encoding="utf-8",
    )
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    overrides = tmp_path / "traces.json"
    overrides.write_text(
        json.dumps(
            {
                "q:one": [
                    {
                        "text": "direct evidence",
                        "kind": "fact",
                        "evidence_ids": ["e:one"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "atlas.json"
    receipt = tmp_path / "receipt.json"

    result = CliRunner().invoke(
        app,
        [
            "atlas-migrate",
            str(source),
            "--output",
            str(output),
            "--artifact-root",
            str(artifact_root),
            "--trace-overrides",
            str(overrides),
            "--receipt-output",
            str(receipt),
        ],
    )

    assert result.exit_code == 0, result.output
    assert receipt.is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["exact_binding_count"] == 1


def test_atlas_migrate_cli_rejects_invalid_boundaries(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "atlas.json"
    missing = runner.invoke(
        app,
        ["atlas-migrate", str(tmp_path / "missing.json"), "--output", str(output)],
    )
    assert missing.exit_code == 1

    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    relative_output = runner.invoke(
        app,
        ["atlas-migrate", str(source), "--output", "atlas.json"],
    )
    assert relative_output.exit_code == 1
    bad_root = runner.invoke(
        app,
        [
            "atlas-migrate",
            str(source),
            "--output",
            str(output),
            "--artifact-root",
            str(tmp_path / "missing-root"),
        ],
    )
    assert bad_root.exit_code == 1

    source.write_text("{", encoding="utf-8")
    bad_json = runner.invoke(
        app,
        ["atlas-migrate", str(source), "--output", str(output)],
    )
    assert bad_json.exit_code == 1

    source.write_text("[]", encoding="utf-8")
    bad_shape = runner.invoke(
        app,
        ["atlas-migrate", str(source), "--output", str(output)],
    )
    assert bad_shape.exit_code == 1

    source.write_text(json.dumps(_legacy_payload()), encoding="utf-8")
    overrides = tmp_path / "traces.json"
    overrides.write_text("[]", encoding="utf-8")
    bad_overrides = runner.invoke(
        app,
        [
            "atlas-migrate",
            str(source),
            "--output",
            str(output),
            "--trace-overrides",
            str(overrides),
        ],
    )
    assert bad_overrides.exit_code == 1

    bad_receipt = runner.invoke(
        app,
        [
            "atlas-migrate",
            str(source),
            "--output",
            str(output),
            "--receipt-output",
            str(tmp_path / "missing-parent" / "receipt.json"),
        ],
    )
    assert bad_receipt.exit_code == 1


def test_migration_covers_relations_timeline_gaps_and_optional_fragment() -> None:
    payload = _legacy_payload()
    evidence = payload["evidence"][0]  # type: ignore[index]
    evidence.pop("source_fragment_id")
    second_hypothesis = deepcopy(payload["hypotheses"][0])  # type: ignore[index]
    second_hypothesis["hypothesis_id"] = "h:two"
    second_hypothesis["title"] = "A second working model"
    payload["hypotheses"].append(second_hypothesis)  # type: ignore[attr-defined]
    payload["relations"] = [
        {
            "relation_id": "relation:one",
            "source_hypothesis_id": "h:one",
            "target_hypothesis_id": "h:two",
            "kind": "parallels",
            "evidence_ids": ["e:one"],
            "explanation": "Historical self-reference used only as a migration fixture.",
        }
    ]
    payload["timeline"] = [
        {
            "event_id": "event:one",
            "date_start": "2026",
            "date_label": "2026",
            "title": "Event",
            "summary": "Summary",
            "state": "model",
            "evidence_ids": ["e:one"],
            "question_ids": ["q:one"],
            "hypothesis_ids": ["h:one"],
        }
    ]
    payload["gaps"] = [
        {
            "gap_id": "gap:one",
            "label": "Unresolved evidence",
            "why_it_matters": "It matters.",
            "next_evidence": "Find a record.",
            "related_question_ids": ["q:one"],
            "related_hypothesis_ids": ["h:one"],
        }
    ]

    result = migrate_legacy_research_atlas(payload)

    assert result.manifest.relations[0].relation_id == "relation_one"
    assert result.manifest.timeline[0].event_id == "event_one"
    assert result.manifest.timeline[0].state.value == "qualified"
    assert result.manifest.gaps[0].gap_id == "gap_one"
    assert result.manifest.evidence[0].source_fragment_id is None


def test_migration_rejects_non_array_collections_and_unknown_states() -> None:
    payload = _legacy_payload()
    payload["gaps"] = {}
    with pytest.raises(ValueError, match="must be JSON arrays"):
        migrate_legacy_research_atlas(payload)

    for collection, field in (
        ("sources", "verification_state"),
        ("questions", "state"),
        ("hypotheses", "state"),
    ):
        payload = _legacy_payload()
        payload[collection][0][field] = "unsupported"  # type: ignore[index]
        with pytest.raises(ValueError, match="unsupported legacy"):
            migrate_legacy_research_atlas(payload)

    with pytest.raises(ValueError, match="unsupported legacy direct state"):
        _mapped_state("missing", {"known": "known"}, label="direct")


def test_identifier_normalization_is_bounded_cached_and_collision_safe() -> None:
    identifiers = _IdentifierMap()
    first = identifiers.get("a:b", domain="source", prefix="source")
    cached = identifiers.get("a:b", domain="source", prefix="source")
    collision = identifiers.get("a b", domain="source", prefix="source")

    assert first == cached == "a_b"
    assert collision.startswith("a_b_")
    assert len(collision) <= 96
    assert _safe_identifier("1", prefix="source") == "source_1"
    long_value = _safe_identifier("A" * 140, prefix="source")
    assert len(long_value) == 96
    assert long_value.startswith("a" * 20)


def test_source_binding_failures_remain_omitted(tmp_path: Path) -> None:
    evidence = {
        "source_address": {
            "char_start": 0,
            "char_end": 2,
        }
    }
    root = tmp_path / "root"
    root.mkdir()

    assert (
        _migrate_source_binding(
            evidence,
            artifact_root=root,
            artifact_path="../outside.md",
        )
        is None
    )
    artifact = root / "source.md"
    artifact.write_text("abc", encoding="utf-8")
    evidence["source_address"] = {"char_start": -1, "char_end": 2}
    assert (
        _migrate_source_binding(
            evidence,
            artifact_root=root,
            artifact_path="source.md",
        )
        is None
    )
    evidence["source_address"] = {"char_start": 1, "char_end": 8}
    assert (
        _migrate_source_binding(
            evidence,
            artifact_root=root,
            artifact_path="source.md",
        )
        is None
    )


def test_trace_override_can_use_normalized_id_and_reject_missing_text() -> None:
    payload = _legacy_payload()
    result = migrate_legacy_research_atlas(
        payload,
        trace_overrides={
            "q_one": [
                {
                    "text": "direct evidence",
                    "kind": "fact",
                    "evidence_ids": ["e:one"],
                }
            ]
        },
    )
    assert result.manifest.questions[0].trace_spans[0].span_id == "trace_q_one_1"

    with pytest.raises(ValueError, match="must occur exactly once"):
        migrate_legacy_research_atlas(
            payload,
            trace_overrides={
                "q:one": [
                    {
                        "text": "absent text",
                        "kind": "fact",
                        "evidence_ids": ["e:one"],
                    }
                ]
            },
        )
