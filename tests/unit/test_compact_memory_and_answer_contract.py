"""Compact task routing and exact answer-contract gate."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from dithyramba.answers import (
    AnswerCitation,
    AnswerContractGate,
    AnswerCoverageDecision,
    AnswerCoverageGate,
    AnswerCoverageSpec,
    AnswerDecision,
    AnswerStatus,
    GateCheckStatus,
    RequiredFacet,
    ResearchAnswer,
)
from dithyramba.atlas import (
    CompactMemoryPacket,
    ResearchAtlasManifest,
    RouteLimits,
    RouteStatus,
    TaskRouter,
)
from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)


def _manifest() -> ResearchAtlasManifest:
    crisis = "Gauguin arrived in Arles. Vincent suffered a crisis in December 1888."
    work = "In Auvers, Vincent continued working with close attention to his canvases."
    payload = {
        "schema_id": ResearchAtlasManifest.SCHEMA,
        "atlas_id": "atlas_van_gogh",
        "case_id": "case_van_gogh",
        "title": "Van Gogh research memory",
        "subject": "Vincent van Gogh",
        "subtitle": "A tiny hermetic test atlas.",
        "generated_at": datetime(2026, 7, 25, tzinfo=UTC).isoformat(),
        "language": "uk",
        "read_only": True,
        "corpus": {
            "label": "Test corpus",
            "source_count": 2,
            "source_family_count": 2,
            "scope_note": "Two exact local sources.",
            "cutoff_note": "Frozen for tests.",
        },
        "sources": [
            {
                "source_id": "source_chronology",
                "title": "Scholarly chronology",
                "creator": "Editors",
                "source_kind": "scholarly_article",
                "voice_kind": "scholarly",
                "date_label": "2009",
                "publisher_or_archive": "Test archive",
                "independence_group": "family_chronology",
                "verification_state": "verified",
                "original_url": None,
                "artifact_path": "artifacts/chronology.md",
                "rights_note": "Test excerpt.",
            },
            {
                "source_id": "source_letter",
                "title": "Letter from Auvers",
                "creator": "Vincent van Gogh",
                "source_kind": "letter",
                "voice_kind": "first_person",
                "date_label": "1890",
                "publisher_or_archive": "Test archive",
                "independence_group": "family_letter",
                "verification_state": "verified",
                "original_url": None,
                "artifact_path": "artifacts/letter.md",
                "rights_note": "Test excerpt.",
            },
        ],
        "evidence": [
            {
                "evidence_id": "evidence_crisis",
                "source_id": "source_chronology",
                "locator": "line 1",
                "excerpt": crisis,
                "role": "supports",
                "asserting_voice": "Scholarly editors",
                "limitation": "A chronology is a synthesis.",
                "source_fragment_id": "fragment_crisis",
                "fragment_text_sha256": sha256(crisis.encode()).hexdigest(),
                "source_address": {
                    "kind": "markdown",
                    "heading_path": ["Crisis"],
                    "line_start": 1,
                    "line_end": 1,
                    "char_start": 0,
                    "char_end": len(crisis),
                },
            },
            {
                "evidence_id": "evidence_work",
                "source_id": "source_letter",
                "locator": "line 1",
                "excerpt": work,
                "role": "supports",
                "asserting_voice": "Vincent van Gogh",
                "limitation": "First-person account.",
                "source_fragment_id": "fragment_work",
                "fragment_text_sha256": sha256(work.encode()).hexdigest(),
                "source_address": {
                    "kind": "markdown",
                    "heading_path": ["Auvers"],
                    "line_start": 1,
                    "line_end": 1,
                    "char_start": 0,
                    "char_end": len(work),
                },
            },
        ],
        "questions": [
            {
                "question_id": "question_crisis",
                "prompt": "Що відомо про кризу грудня 1888 року і Gauguin?",
                "short_answer": "Корпус фіксує кризу, але не остаточний мотив.",
                "state": "qualified",
                "evidence_ids": ["evidence_crisis"],
                "gap": "Потрібен незалежний клінічний запис.",
                "tags": ["arles", "gauguin"],
            },
            {
                "question_id": "question_auvers",
                "prompt": "Чи продовжував він працювати в Auvers?",
                "short_answer": "Так, лист прямо описує роботу над полотнами.",
                "state": "confirmed",
                "evidence_ids": ["evidence_work"],
                "gap": None,
                "tags": ["auvers"],
            },
        ],
        "hypotheses": [
            {
                "hypothesis_id": "hypothesis_crisis",
                "title": "Криза не має однієї доведеної причини",
                "synthesis": "Подія підтверджена, причинна реконструкція відкрита.",
                "state": "working",
                "evidence_ids": ["evidence_crisis"],
                "counterevidence_ids": [],
                "gap": "Знайти незалежний запис.",
                "question_ids": ["question_crisis"],
            }
        ],
        "relations": [],
        "timeline": [
            {
                "event_id": "event_crisis",
                "date_start": "1888-12",
                "date_end": None,
                "date_label": "December 1888",
                "title": "Arles crisis",
                "summary": "A crisis and hospitalisation are recorded.",
                "state": "qualified",
                "evidence_ids": ["evidence_crisis"],
                "question_ids": ["question_crisis"],
                "hypothesis_ids": ["hypothesis_crisis"],
            },
            {
                "event_id": "event_auvers",
                "date_start": "1890",
                "date_end": None,
                "date_label": "1890",
                "title": "Work in Auvers",
                "summary": "The final letter records continued work.",
                "state": "confirmed",
                "evidence_ids": ["evidence_work"],
                "question_ids": ["question_auvers"],
                "hypothesis_ids": [],
            },
        ],
        "gaps": [
            {
                "gap_id": "gap_crisis_record",
                "label": "Independent crisis record",
                "why_it_matters": "It would separate the event from later interpretation.",
                "next_evidence": "Hospital or police record.",
                "related_question_ids": ["question_crisis"],
                "related_hypothesis_ids": ["hypothesis_crisis"],
            }
        ],
        "methodology_note": "Every claim returns to an exact local excerpt.",
    }
    return ResearchAtlasManifest.model_validate_json(json.dumps(payload))


def _packet(manifest: ResearchAtlasManifest) -> CompactMemoryPacket:
    manifest_hash = sha256_hex(canonical_json_bytes(manifest.model_dump(mode="json")))
    return TaskRouter(manifest, manifest_hash).route(
        "Що сталося під час кризи з Gauguin у грудні 1888 року?",
        limits=RouteLimits(
            max_questions=1,
            max_hypotheses=1,
            max_timeline_events=1,
            max_gaps=1,
        ),
    )


def _wide_packet(manifest: ResearchAtlasManifest) -> CompactMemoryPacket:
    manifest_hash = sha256_hex(canonical_json_bytes(manifest.model_dump(mode="json")))
    return TaskRouter(manifest, manifest_hash).route(
        "Що відомо про кризу Gauguin та роботу в Auvers?",
        limits=RouteLimits(
            max_questions=2,
            max_hypotheses=1,
            max_timeline_events=2,
            max_gaps=1,
        ),
    )


def _answer_json(packet: CompactMemoryPacket) -> str:
    return json.dumps(
        {
            "schema_id": "dithyramba.research_answer/1.0",
            "task_id": "task_crisis",
            "route_receipt": {
                "packet_id": packet.packet_id,
                "packet_hash": packet.packet_hash,
            },
            "status": "qualified",
            "short_answer": "Криза підтверджена, але її мотив лишається відкритим.",
            "claims": [
                {
                    "claim_id": "claim_crisis",
                    "text": "Криза сталася у грудні 1888 року.",
                    "status": "qualified",
                    "evidence_ids": ["evidence_crisis"],
                }
            ],
            "citations": [
                {
                    "evidence_id": "evidence_crisis",
                    "source_id": "source_chronology",
                    "exact_quote": "Vincent suffered a crisis in December 1888.",
                }
            ],
            "limitations": ["Хронологія не встановлює мотив."],
            "open_gap": {
                "question": "Що стало безпосереднім мотивом?",
                "why_it_matters": "Це відділяє факт події від інтерпретації.",
                "next_evidence": "Незалежний лікарняний або поліцейський запис.",
            },
        }
    )


def _write_artifacts(root: Path) -> None:
    artifacts = root / "artifacts"
    artifacts.mkdir()
    artifacts.joinpath("chronology.md").write_text(
        "Gauguin arrived in Arles. Vincent suffered a crisis in December 1888.",
        encoding="utf-8",
    )
    artifacts.joinpath("letter.md").write_text(
        "In Auvers, Vincent continued working with close attention to his canvases.",
        encoding="utf-8",
    )


def _readdress(packet: CompactMemoryPacket) -> CompactMemoryPacket:
    payload = packet.semantic_payload()
    return packet.model_copy(
        update={
            "packet_id": canonical_content_id("memory_packet", payload),
            "packet_hash": canonical_sha256_hex(payload),
        }
    )


def test_router_is_deterministic_closed_and_compact() -> None:
    manifest = _manifest()
    first = _packet(manifest)
    second = _packet(manifest)

    assert first == second
    assert first.route_status is RouteStatus.ROUTED
    assert first.packet_id == second.packet_id
    assert [item.question_id for item in first.questions] == ["question_crisis"]
    assert [item.evidence_id for item in first.evidence] == ["evidence_crisis"]
    assert [item.source_id for item in first.sources] == ["source_chronology"]
    assert first.hypotheses[0].question_ids == ("question_crisis",)


def test_router_preserves_no_match_and_rejects_invalid_inputs() -> None:
    manifest = _manifest()
    manifest_hash = sha256_hex(canonical_json_bytes(manifest.model_dump(mode="json")))
    router = TaskRouter(manifest, manifest_hash)

    packet = router.route("квантова телепортація нейтрино")
    assert packet.route_status is RouteStatus.NO_MATCH
    assert packet.evidence == ()
    with pytest.raises(ValueError, match="blank"):
        router.route("  ")
    with pytest.raises(ValueError, match="manifest_hash"):
        TaskRouter(manifest, "bad")


def test_packet_rejects_tampered_content_address() -> None:
    packet = _packet(_manifest())
    payload = packet.model_dump(mode="json")
    payload["query"] = "tampered"
    with pytest.raises(ValidationError, match="packet_id"):
        CompactMemoryPacket.model_validate_json(json.dumps(payload))


def test_answer_gate_accepts_exact_packet_bound_answer(tmp_path: Path) -> None:
    packet = _packet(_manifest())
    _write_artifacts(tmp_path)

    result = AnswerContractGate().evaluate_payload(
        _answer_json(packet),
        task_id="task_crisis",
        packet=packet,
        artifact_root=tmp_path,
    )

    assert result.decision is AnswerDecision.ACCEPTED
    assert result.answer_id is not None
    assert result.answer_hash is not None
    assert result.violations == ()
    assert all(item.status is GateCheckStatus.PASSED for item in result.checks)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("task", "task_identity"),
        ("receipt", "route_receipt"),
        ("membership", "packet_membership"),
        ("quote", "exact_local_evidence"),
    ],
)
def test_answer_gate_rejects_unbound_claims(
    tmp_path: Path, mutation: str, failed_check: str
) -> None:
    packet = _packet(_manifest())
    _write_artifacts(tmp_path)
    payload = json.loads(_answer_json(packet))
    if mutation == "task":
        payload["task_id"] = "task_other"
    elif mutation == "receipt":
        payload["route_receipt"]["packet_hash"] = "0" * 64
    elif mutation == "membership":
        payload["citations"][0]["source_id"] = "source_letter"
    else:
        payload["citations"][0]["exact_quote"] = "invented quotation"

    result = AnswerContractGate().evaluate_payload(
        json.dumps(payload),
        task_id="task_crisis",
        packet=packet,
        artifact_root=tmp_path,
    )

    assert result.decision is AnswerDecision.REJECTED
    assert failed_check in {
        item.name for item in result.checks if item.status is GateCheckStatus.FAILED
    }


def test_answer_gate_reports_schema_and_missing_artifact(tmp_path: Path) -> None:
    packet = _packet(_manifest())
    malformed = json.loads(_answer_json(packet))
    malformed["status"] = "certain"

    schema_result = AnswerContractGate().evaluate_payload(
        json.dumps(malformed),
        task_id="task_crisis",
        packet=packet,
        artifact_root=tmp_path,
    )
    missing_result = AnswerContractGate().evaluate_payload(
        _answer_json(packet),
        task_id="task_crisis",
        packet=packet,
        artifact_root=tmp_path,
    )

    assert schema_result.checks[0].name == "schema"
    assert schema_result.checks[0].status is GateCheckStatus.FAILED
    assert missing_result.decision is AnswerDecision.REJECTED
    assert "missing" in missing_result.violations[0]


def test_research_answer_rejects_each_ambiguous_shape() -> None:
    packet = _packet(_manifest())
    base = json.loads(_answer_json(packet))
    cases: list[tuple[dict[str, object], str]] = []

    wrong_schema = deepcopy(base)
    wrong_schema["schema_id"] = "wrong"
    cases.append((wrong_schema, "schema_id"))

    duplicate_claim = deepcopy(base)
    duplicate_claim["claims"].append(deepcopy(duplicate_claim["claims"][0]))
    cases.append((duplicate_claim, "claim_id values"))

    duplicate_citation = deepcopy(base)
    duplicate_citation["citations"].append(deepcopy(duplicate_citation["citations"][0]))
    cases.append((duplicate_citation, "one evidence_id"))

    duplicate_claim_evidence = deepcopy(base)
    duplicate_claim_evidence["claims"][0]["evidence_ids"].append("evidence_crisis")
    cases.append((duplicate_claim_evidence, "claim evidence_ids"))

    uncited = deepcopy(base)
    uncited["claims"][0]["evidence_ids"] = ["evidence_missing"]
    cases.append((uncited, "uncited evidence"))

    unused = deepcopy(base)
    unused["citations"].append(
        {
            "evidence_id": "evidence_work",
            "source_id": "source_letter",
            "exact_quote": "continued working",
        }
    )
    cases.append((unused, "citations must support"))

    duplicate_limit = deepcopy(base)
    duplicate_limit["limitations"].append(duplicate_limit["limitations"][0])
    cases.append((duplicate_limit, "limitations must be unique"))

    duplicate_gap = deepcopy(base)
    duplicate_gap["open_gap"]["question"] = duplicate_gap["limitations"][0]
    cases.append((duplicate_gap, "must not duplicate"))

    insufficient_without_gap = deepcopy(base)
    insufficient_without_gap["status"] = "insufficient"
    insufficient_without_gap["open_gap"] = None
    cases.append((insufficient_without_gap, "require an explicit open_gap"))

    qualified_without_proof = deepcopy(base)
    qualified_without_proof["claims"] = []
    qualified_without_proof["citations"] = []
    cases.append((qualified_without_proof, "require claims and citations"))

    supported_without_supported_claim = deepcopy(base)
    supported_without_supported_claim["status"] = "supported"
    cases.append((supported_without_supported_claim, "supported claim"))

    for payload, message in cases:
        with pytest.raises(ValidationError, match=message):
            ResearchAnswer.model_validate_json(json.dumps(payload))


def test_insufficient_answer_is_valid_with_an_actionable_gap() -> None:
    packet = _packet(_manifest())
    payload = json.loads(_answer_json(packet))
    payload.update(
        {
            "status": "insufficient",
            "claims": [],
            "citations": [],
            "short_answer": "Наявної памʼяті недостатньо.",
        }
    )
    answer = ResearchAnswer.model_validate_json(json.dumps(payload))
    result = AnswerContractGate().evaluate_payload(
        answer,
        task_id="task_crisis",
        packet=packet,
        artifact_root="/does/not/matter",
    )

    assert result.decision is AnswerDecision.ACCEPTED
    assert answer.semantic_payload()["status"] == "insufficient"


def test_answer_gate_rejects_unknown_evidence(tmp_path: Path) -> None:
    packet = _packet(_manifest())
    _write_artifacts(tmp_path)
    payload = json.loads(_answer_json(packet))
    payload["claims"][0]["evidence_ids"] = ["evidence_unknown"]
    payload["citations"][0]["evidence_id"] = "evidence_unknown"

    result = AnswerContractGate().evaluate_payload(
        json.dumps(payload),
        task_id="task_crisis",
        packet=packet,
        artifact_root=tmp_path,
    )

    assert result.decision is AnswerDecision.REJECTED
    assert "unknown evidence" in result.violations[0]


def test_artifact_gate_reports_binding_path_range_drift_and_excerpt_edges(
    tmp_path: Path,
) -> None:
    packet = _packet(_manifest())
    _write_artifacts(tmp_path)
    citation = AnswerCitation(
        evidence_id="evidence_crisis",
        source_id="source_chronology",
        exact_quote="Vincent suffered a crisis",
    )
    evidence = packet.evidence[0]
    source = packet.sources[0]
    assert evidence.source_address is not None
    assert evidence.excerpt is not None
    gate = AnswerContractGate()

    no_binding = evidence.model_copy(
        update={
            "source_fragment_id": None,
            "fragment_text_sha256": None,
            "source_address": None,
        }
    )
    no_path = source.model_copy(update={"artifact_path": None})
    escaped = source.model_copy(update={"artifact_path": "../../outside.md"})
    missing = source.model_copy(update={"artifact_path": "artifacts/missing.md"})
    unavailable_address = evidence.source_address.model_copy(update={"line_end": 2})
    unavailable = evidence.model_copy(update={"source_address": unavailable_address})

    assert (
        "binding is absent"
        in gate._artifact_errors(
            (citation,), {evidence.evidence_id: no_binding}, {source.source_id: source}, tmp_path
        )[0]
    )
    assert (
        "path is absent"
        in gate._artifact_errors(
            (citation,), {evidence.evidence_id: evidence}, {source.source_id: no_path}, tmp_path
        )[0]
    )
    assert (
        "escapes"
        in gate._artifact_errors(
            (citation,), {evidence.evidence_id: evidence}, {source.source_id: escaped}, tmp_path
        )[0]
    )
    assert (
        "missing"
        in gate._artifact_errors(
            (citation,), {evidence.evidence_id: evidence}, {source.source_id: missing}, tmp_path
        )[0]
    )
    assert (
        "range is unavailable"
        in gate._artifact_errors(
            (citation,), {evidence.evidence_id: unavailable}, {source.source_id: source}, tmp_path
        )[0]
    )

    chronology = tmp_path / "artifacts/chronology.md"
    chronology.write_text(evidence.excerpt + " changed", encoding="utf-8")
    assert (
        "has drifted"
        in gate._artifact_errors(
            (citation,), {evidence.evidence_id: evidence}, {source.source_id: source}, tmp_path
        )[0]
    )

    changed_excerpt = "Different accepted excerpt with Vincent suffered a crisis"
    changed = evidence.model_copy(update={"excerpt": changed_excerpt})
    chronology.write_text(evidence.excerpt or "", encoding="utf-8")
    assert (
        "outside its bound fragment"
        in gate._artifact_errors(
            (citation,), {evidence.evidence_id: changed}, {source.source_id: source}, tmp_path
        )[0]
    )

    assert gate._artifact_errors((citation,), {}, {}, tmp_path) == []


def test_packet_validator_rejects_tampered_graph_and_status() -> None:
    packet = _packet(_manifest())
    cases = [
        (packet.model_copy(update={"schema_id": "wrong"}), "schema_id"),
        (packet.model_copy(update={"packet_hash": "0" * 64}), "packet_hash"),
        (
            _readdress(
                packet.model_copy(update={"sources": (packet.sources[0], packet.sources[0])})
            ),
            "identifiers must be unique",
        ),
        (
            _readdress(packet.model_copy(update={"sources": ()})),
            "evidence source references",
        ),
        (
            _readdress(packet.model_copy(update={"evidence": ()})),
            "question evidence references",
        ),
        (
            _readdress(packet.model_copy(update={"route_status": RouteStatus.NO_MATCH})),
            "no_match packets",
        ),
        (
            _readdress(
                packet.model_copy(
                    update={
                        "sources": (),
                        "evidence": (),
                        "questions": (),
                        "hypotheses": (),
                        "relations": (),
                        "timeline": (),
                        "gaps": (),
                        "route_matches": (),
                    }
                )
            ),
            "routed packets",
        ),
    ]
    for candidate, message in cases:
        with pytest.raises(ValidationError, match=message):
            CompactMemoryPacket.model_validate_json(candidate.model_dump_json())


def test_router_handles_stop_words_long_queries_and_zero_link_budgets() -> None:
    manifest = _manifest()
    manifest_hash = sha256_hex(canonical_json_bytes(manifest.model_dump(mode="json")))
    router = TaskRouter(manifest, manifest_hash)

    assert router.route("що як про але").route_status is RouteStatus.NO_MATCH
    with pytest.raises(ValueError, match="at most 4000"):
        router.route("x" * 4_001)

    packet = router.route(
        "Що відомо про кризу Gauguin 1888?",
        limits=RouteLimits(
            max_questions=1,
            max_hypotheses=0,
            max_timeline_events=0,
            max_gaps=0,
        ),
    )
    assert packet.hypotheses == ()
    assert packet.timeline == ()
    assert packet.gaps == ()


def _coverage_spec(packet: CompactMemoryPacket) -> AnswerCoverageSpec:
    return AnswerCoverageSpec(
        schema_id=AnswerCoverageSpec.SCHEMA,
        task_id="task_crisis",
        packet_id=packet.packet_id,
        packet_hash=packet.packet_hash,
        allowed_statuses=(AnswerStatus.QUALIFIED,),
        required_facets=(
            RequiredFacet(
                facet_id="crisis_event",
                label="The December crisis",
                evidence_ids_any=("evidence_crisis",),
            ),
            RequiredFacet(
                facet_id="continued_work",
                label="Continued work in Auvers",
                evidence_ids_any=("evidence_work",),
            ),
        ),
        require_open_gap=True,
        min_cited_sources=2,
    )


def _wide_answer(packet: CompactMemoryPacket, *, include_work: bool) -> ResearchAnswer:
    payload = json.loads(_answer_json(packet))
    if include_work:
        payload["claims"].append(
            {
                "claim_id": "claim_work",
                "text": "В Auvers він продовжував працювати над полотнами.",
                "status": "supported",
                "evidence_ids": ["evidence_work"],
            }
        )
        payload["citations"].append(
            {
                "evidence_id": "evidence_work",
                "source_id": "source_letter",
                "exact_quote": "Vincent continued working",
            }
        )
    return ResearchAnswer.model_validate_json(json.dumps(payload))


def test_answer_coverage_gate_accepts_every_required_facet() -> None:
    packet = _wide_packet(_manifest())
    answer = _wide_answer(packet, include_work=True)
    spec = _coverage_spec(packet)

    result = AnswerCoverageGate().evaluate(answer, spec=spec, packet=packet)

    assert result.decision is AnswerCoverageDecision.COMPLETE
    assert result.covered_facets == result.required_facets == 2
    assert result.coverage_basis_points == 10_000
    assert result.unique_cited_sources == 2
    assert result.violations == ()
    assert result.spec_id == spec.spec_id
    assert result.spec_hash == spec.spec_hash


def test_answer_coverage_gate_rejects_exact_but_incomplete_answer() -> None:
    packet = _wide_packet(_manifest())
    answer = _wide_answer(packet, include_work=False)

    result = AnswerCoverageGate().evaluate(
        answer,
        spec=_coverage_spec(packet),
        packet=packet,
    )

    assert result.decision is AnswerCoverageDecision.INCOMPLETE
    assert result.covered_facets == 1
    assert result.required_facets == 2
    assert result.coverage_basis_points == 5_000
    assert "required facet continued_work is not covered" in result.violations
    assert "answer cites 1 unique sources; 2 required" in result.violations


def test_answer_coverage_gate_fails_closed_on_wrong_binding_status_and_gap() -> None:
    packet = _wide_packet(_manifest())
    answer_payload = json.loads(_answer_json(packet))
    answer_payload["open_gap"] = None
    answer = ResearchAnswer.model_validate_json(json.dumps(answer_payload))
    spec = _coverage_spec(packet).model_copy(
        update={
            "task_id": "task_other",
            "packet_hash": "0" * 64,
            "allowed_statuses": (AnswerStatus.SUPPORTED,),
            "required_facets": (
                RequiredFacet(
                    facet_id="unknown_record",
                    label="Evidence outside this packet",
                    evidence_ids_any=("evidence_unknown",),
                ),
            ),
        }
    )

    result = AnswerCoverageGate().evaluate(answer, spec=spec, packet=packet)

    assert result.decision is AnswerCoverageDecision.INCOMPLETE
    assert any("task_id does not match" in item for item in result.violations)
    assert any("are not bound" in item for item in result.violations)
    assert any("outside the packet" in item for item in result.violations)
    assert any("status qualified is not allowed" in item for item in result.violations)
    assert "the task requires an explicit open_gap" in result.violations


def test_answer_coverage_spec_rejects_ambiguous_requirements() -> None:
    packet = _wide_packet(_manifest())
    facet = RequiredFacet(
        facet_id="crisis_event",
        label="Crisis",
        evidence_ids_any=("evidence_crisis",),
    )
    base = {
        "schema_id": AnswerCoverageSpec.SCHEMA,
        "task_id": "task_crisis",
        "packet_id": packet.packet_id,
        "packet_hash": packet.packet_hash,
        "allowed_statuses": (AnswerStatus.QUALIFIED,),
        "required_facets": (facet,),
    }

    with pytest.raises(ValidationError, match="schema_id"):
        AnswerCoverageSpec.model_validate({**base, "schema_id": "wrong"})
    with pytest.raises(ValidationError, match="facet_id values"):
        AnswerCoverageSpec.model_validate({**base, "required_facets": (facet, facet)})
    with pytest.raises(ValidationError, match="allowed_statuses must be unique"):
        AnswerCoverageSpec.model_validate(
            {
                **base,
                "allowed_statuses": (AnswerStatus.QUALIFIED, AnswerStatus.QUALIFIED),
            }
        )
    with pytest.raises(ValidationError, match="facet evidence IDs must be unique"):
        RequiredFacet(
            facet_id="duplicate_evidence",
            label="Duplicate",
            evidence_ids_any=("evidence_crisis", "evidence_crisis"),
        )
    with pytest.raises(ValidationError, match="must not be blank"):
        RequiredFacet(
            facet_id="blank_evidence",
            label="Blank",
            evidence_ids_any=("",),
        )
    with pytest.raises(ValidationError, match="min_evidence"):
        RequiredFacet(
            facet_id="impossible_minimum",
            label="Impossible",
            evidence_ids_any=("evidence_crisis",),
            min_evidence=2,
        )
