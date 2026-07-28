"""Semantic receipt and bounded CompactMemoryPacket/1.1 routing contracts."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.atlas import (
    CompactMemoryPacket,
    CompactMemoryPacketV1_1,
    LexicalQuestionCandidate,
    ResearchAtlasManifest,
    RouteBudget,
    RouteCandidateChannel,
    RouteCandidateDecision,
    RouteCandidateReceipt,
    RouteLimits,
    RouteObjectKind,
    RouteOmissionReason,
    RouteStatus,
    RouteTraceEntry,
    RouteTraceStage,
    SemanticQuestionCandidate,
    TaskRouter,
    build_route_candidate_receipt,
)
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

_MODEL_HASH = "1" * 64
_RUNTIME_HASH = "2" * 64
_INDEX_HASH = "3" * 64


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
    return TaskRouter(manifest, _manifest_hash(manifest)).route(
        "Що сталося під час кризи з Gauguin у грудні 1888 року?",
        limits=RouteLimits(
            max_questions=1,
            max_hypotheses=1,
            max_timeline_events=1,
            max_gaps=1,
        ),
    )


def _manifest_hash(manifest: ResearchAtlasManifest) -> str:
    return canonical_sha256_hex(manifest.model_dump(mode="json"))


def _receipt(
    manifest: ResearchAtlasManifest,
    *,
    query: str = "Що відомо про Gauguin і Auvers?",
    semantic_scores: dict[str, float] | None = None,
    lexical_scores: dict[str, int] | None = None,
    threshold: float = 0.5,
    decision: RouteCandidateDecision | None = None,
) -> RouteCandidateReceipt:
    return build_route_candidate_receipt(
        manifest,
        manifest_hash=_manifest_hash(manifest),
        query=query,
        model_profile_hash=_MODEL_HASH,
        runtime_profile_hash=_RUNTIME_HASH,
        index_hash=_INDEX_HASH,
        semantic_threshold=threshold,
        semantic_scores=semantic_scores or {},
        lexical_scores=lexical_scores or {},
        decision=decision,
    )


def _readdress_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    semantic_payload = {
        key: value for key, value in payload.items() if key not in {"receipt_id", "receipt_hash"}
    }
    payload["receipt_id"] = canonical_content_id(
        "route_candidate_receipt",
        semantic_payload,
    )
    payload["receipt_hash"] = canonical_sha256_hex(semantic_payload)
    return payload


def _readdress_packet(payload: dict[str, Any]) -> dict[str, Any]:
    semantic_payload = {
        key: value for key, value in payload.items() if key not in {"packet_id", "packet_hash"}
    }
    payload["packet_id"] = canonical_content_id("memory_packet", semantic_payload)
    payload["packet_hash"] = canonical_sha256_hex(semantic_payload)
    return payload


def _load_receipt(payload: dict[str, Any]) -> RouteCandidateReceipt:
    return RouteCandidateReceipt.model_validate_json(json.dumps(payload))


def _load_v1_1_packet(payload: dict[str, Any]) -> CompactMemoryPacketV1_1:
    return CompactMemoryPacketV1_1.model_validate_json(json.dumps(payload))


def _manifest_with_relation() -> ResearchAtlasManifest:
    payload = _manifest().model_dump(mode="json")
    payload["hypotheses"].append(
        {
            "hypothesis_id": "hypothesis_context",
            "title": "The crisis also has a social context",
            "synthesis": "The shared chronology supports a bounded contextual reading.",
            "state": "working",
            "evidence_ids": ["evidence_crisis"],
            "counterevidence_ids": [],
            "gap": "Independent testimony remains absent.",
            "question_ids": ["question_crisis"],
        }
    )
    payload["relations"].append(
        {
            "relation_id": "relation_context",
            "source_hypothesis_id": "hypothesis_crisis",
            "target_hypothesis_id": "hypothesis_context",
            "kind": "qualifies",
            "evidence_ids": ["evidence_crisis"],
            "explanation": "The contextual reading qualifies a single-cause account.",
        }
    )
    return ResearchAtlasManifest.model_validate(payload)


def _no_expansion_budget(**updates: int) -> RouteBudget:
    values = {
        "max_questions": 4,
        "max_linked_hypotheses": 0,
        "max_relations": 0,
        "max_timeline_events": 0,
        "max_gaps": 0,
        "max_evidence": 10,
        "max_sources": 10,
    }
    values.update(updates)
    return RouteBudget(**values)


def test_v1_packet_replays_with_the_frozen_schema_and_content_address() -> None:
    manifest = _manifest()
    packet = _packet(manifest)
    serialized = packet.model_dump_json()
    loaded = CompactMemoryPacket.model_validate_json(serialized)

    assert loaded == packet
    assert packet.schema_id == "dithyramba.compact_memory_packet/1.0"
    assert packet.packet_id == "memory_packet_eae9af3c3362e8de5f77839ef1b3245a"
    assert packet.packet_hash == (
        "eae9af3c3362e8de5f77839ef1b3245aa6e1a18684fbd64c2c5aa5da8acccf94"
    )
    assert "route_receipt_id" not in packet.model_dump(mode="json")
    assert loaded.semantic_payload() == packet.semantic_payload()


def test_v1_schema_refuses_v2_only_status_and_route_object_kind() -> None:
    packet = _packet(_manifest())
    status_payload = packet.model_dump(mode="json")
    status_payload["route_status"] = "budget_exceeded"
    status_semantic_payload = {
        key: value
        for key, value in status_payload.items()
        if key not in {"packet_id", "packet_hash"}
    }
    status_payload["packet_id"] = canonical_content_id(
        "memory_packet",
        status_semantic_payload,
    )
    status_payload["packet_hash"] = canonical_sha256_hex(status_semantic_payload)
    with pytest.raises(ValidationError, match="not valid"):
        CompactMemoryPacket.model_validate_json(json.dumps(status_payload))

    relation_payload = packet.model_dump(mode="json")
    relation_payload["route_matches"][0]["object_kind"] = "relation"
    relation_semantic_payload = {
        key: value
        for key, value in relation_payload.items()
        if key not in {"packet_id", "packet_hash"}
    }
    relation_payload["packet_id"] = canonical_content_id(
        "memory_packet",
        relation_semantic_payload,
    )
    relation_payload["packet_hash"] = canonical_sha256_hex(relation_semantic_payload)
    with pytest.raises(ValidationError, match="relation route matches"):
        CompactMemoryPacket.model_validate_json(json.dumps(relation_payload))


def test_receipt_is_provider_agnostic_content_addressed_and_exactly_ranked() -> None:
    manifest = _manifest()
    receipt = _receipt(
        manifest,
        semantic_scores={"question_auvers": 0.75, "question_crisis": 0.875},
    )

    assert [item.question_id for item in receipt.semantic_candidates] == [
        "question_crisis",
        "question_auvers",
    ]
    assert [item.rank for item in receipt.semantic_candidates] == [1, 2]
    assert receipt.semantic_candidates[0].score_hex == (0.875).hex()
    assert receipt.semantic_threshold_hex == (0.5).hex()
    assert receipt.receipt_id == f"route_candidate_receipt_{receipt.receipt_hash[:32]}"
    assert "sentence_transformers" not in receipt.canonical_bytes.decode("utf-8")


def test_receipt_rejects_tampering_order_threshold_duplicates_and_nonfinite_scores() -> None:
    manifest = _manifest()
    receipt = _receipt(
        manifest,
        semantic_scores={"question_crisis": 0.9},
        lexical_scores={"question_auvers": 9},
    )

    tampered = receipt.model_dump(mode="json")
    tampered["query"] = "tampered"
    with pytest.raises(ValidationError, match=r"query_hash|receipt_id"):
        RouteCandidateReceipt.model_validate_json(json.dumps(tampered))

    reordered_receipt = _receipt(
        manifest,
        semantic_scores={"question_crisis": 0.9, "question_auvers": 0.8},
    )
    reordered = reordered_receipt.model_dump(mode="json")
    reordered["semantic_candidates"] = list(reversed(reordered["semantic_candidates"]))
    with pytest.raises(ValidationError, match=r"ordered|ranks"):
        RouteCandidateReceipt.model_validate_json(json.dumps(_readdress_receipt(reordered)))

    below_threshold = receipt.model_dump(mode="json")
    below_threshold["semantic_candidates"][0]["score_hex"] = (0.25).hex()
    with pytest.raises(ValidationError, match="below the receipt threshold"):
        RouteCandidateReceipt.model_validate_json(json.dumps(_readdress_receipt(below_threshold)))

    duplicate = receipt.model_dump(mode="json")
    duplicate["lexical_candidates"][0]["question_id"] = "question_crisis"
    with pytest.raises(ValidationError, match="unique across route channels"):
        RouteCandidateReceipt.model_validate_json(json.dumps(_readdress_receipt(duplicate)))

    with pytest.raises(ValueError, match="finite"):
        _receipt(manifest, semantic_scores={"question_crisis": math.nan})


def test_receipt_rejects_unknown_questions_and_stale_query_or_manifest_at_consumption() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest, semantic_scores={"question_crisis": 0.9})
    router = TaskRouter(manifest, _manifest_hash(manifest))

    with pytest.raises(ValueError, match="stale for the normalized query"):
        router.route_v2("Інший запит", receipt, RouteBudget())

    changed_manifest = ResearchAtlasManifest.model_validate(
        {**manifest.model_dump(mode="json"), "title": "Changed Atlas"}
    )
    changed_router = TaskRouter(changed_manifest, _manifest_hash(changed_manifest))
    with pytest.raises(ValueError, match="stale for the Atlas manifest"):
        changed_router.route_v2(receipt.query, receipt, RouteBudget())

    unknown_payload = receipt.model_dump(mode="json")
    unknown_payload["semantic_candidates"][0]["question_id"] = "question_unknown"
    unknown_receipt = RouteCandidateReceipt.model_validate_json(
        json.dumps(_readdress_receipt(unknown_payload))
    )
    with pytest.raises(ValueError, match="unknown question IDs"):
        router.route_v2(receipt.query, unknown_receipt, RouteBudget())
    with pytest.raises(ValueError, match="unknown question IDs"):
        _receipt(manifest, semantic_scores={"question_unknown": 0.9})


def test_route_v2_is_deterministic_semantic_only_and_keeps_channel_scores_separate() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest, semantic_scores={"question_auvers": 0.9})
    router = TaskRouter(manifest, _manifest_hash(manifest))
    budget = RouteBudget()

    first = router.route_v2(receipt.query, receipt, budget)
    second = router.route_v2(receipt.query, receipt, budget)

    assert first == second
    assert first.packet_hash == second.packet_hash
    assert first.schema_id == CompactMemoryPacketV1_1.SCHEMA
    assert first.route_receipt_id == receipt.receipt_id
    assert first.route_receipt_hash == receipt.receipt_hash
    assert first.route_matches == ()
    assert [item.question_id for item in first.questions] == ["question_auvers"]
    seed = first.route_trace[0]
    assert seed.stage is RouteTraceStage.SEED
    assert seed.object_kind is RouteObjectKind.QUESTION
    assert seed.semantic_score_hex == (0.9).hex()
    assert seed.lexical_score is None
    assert all(item.stage is RouteTraceStage.GRAPH_EXPANSION for item in first.route_trace[1:])


def test_route_v2_preserves_channel_precedence_without_comparing_score_scales() -> None:
    manifest = _manifest()
    receipt = _receipt(
        manifest,
        semantic_scores={"question_auvers": 0.51},
        lexical_scores={"question_crisis": 1_000_000},
    )
    packet = TaskRouter(manifest, _manifest_hash(manifest)).route_v2(
        receipt.query,
        receipt,
        _no_expansion_budget(max_questions=2),
    )

    assert [item.question_id for item in packet.questions] == [
        "question_auvers",
        "question_crisis",
    ]
    seeds = [item for item in packet.route_trace if item.stage is RouteTraceStage.SEED]
    assert seeds[0].semantic_score_hex == (0.51).hex()
    assert seeds[0].lexical_score is None
    assert seeds[1].semantic_score_hex is None
    assert seeds[1].lexical_score == 1_000_000


def test_route_v2_normalizes_nfc_and_nfd_to_one_receipt_and_packet_identity() -> None:
    manifest = _manifest()
    query_nfc = "Café Gauguin"
    query_nfd = unicodedata.normalize("NFD", query_nfc)
    receipt = _receipt(
        manifest,
        query=query_nfd,
        semantic_scores={"question_crisis": 0.9},
    )
    router = TaskRouter(manifest, _manifest_hash(manifest))

    from_nfc = router.route_v2(query_nfc, receipt, RouteBudget())
    from_nfd = router.route_v2(query_nfd, receipt, RouteBudget())

    assert receipt.query == query_nfc
    assert from_nfc.query == query_nfc
    assert from_nfc.packet_hash == from_nfd.packet_hash
    assert from_nfc.packet_id == from_nfd.packet_id


def test_route_v2_abstention_is_no_match_even_when_candidates_were_disclosed() -> None:
    manifest = _manifest()
    receipt = _receipt(
        manifest,
        semantic_scores={"question_crisis": 0.9},
        decision=RouteCandidateDecision.ABSTAINED,
    )
    packet = TaskRouter(manifest, _manifest_hash(manifest)).route_v2(
        receipt.query,
        receipt,
        RouteBudget(),
    )

    assert packet.route_status is RouteStatus.NO_MATCH
    assert packet.questions == ()
    assert packet.evidence == ()
    assert packet.sources == ()
    assert packet.route_trace == ()
    assert packet.omissions == ()


def test_atomic_graph_closure_prunes_objects_whose_complete_references_do_not_fit() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest, semantic_scores={"question_crisis": 0.9})
    budget = RouteBudget(
        max_questions=1,
        max_linked_hypotheses=0,
        max_relations=0,
        max_timeline_events=2,
        max_gaps=2,
        max_evidence=2,
        max_sources=2,
    )
    packet = TaskRouter(manifest, _manifest_hash(manifest)).route_v2(
        receipt.query,
        receipt,
        budget,
    )

    assert [item.question_id for item in packet.questions] == ["question_crisis"]
    assert packet.hypotheses == ()
    assert packet.timeline == ()
    assert packet.gaps == ()
    omissions = {(item.object_kind, item.object_id): item.reason for item in packet.omissions}
    assert omissions[(RouteObjectKind.HYPOTHESIS, "hypothesis_crisis")] is (
        RouteOmissionReason.HYPOTHESIS_BUDGET
    )
    assert omissions[(RouteObjectKind.TIMELINE, "event_crisis")] is (
        RouteOmissionReason.HYPOTHESIS_BUDGET
    )
    assert omissions[(RouteObjectKind.GAP, "gap_crisis_record")] is (
        RouteOmissionReason.HYPOTHESIS_BUDGET
    )
    assert {item.evidence_id for item in packet.evidence} == {"evidence_crisis"}
    assert {item.source_id for item in packet.sources} == {"source_chronology"}


@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        (
            _no_expansion_budget(max_evidence=1, max_sources=2),
            RouteOmissionReason.EVIDENCE_BUDGET,
        ),
        (
            _no_expansion_budget(max_evidence=2, max_sources=1),
            RouteOmissionReason.SOURCE_BUDGET,
        ),
    ],
)
def test_question_seed_admission_is_atomic_across_evidence_and_source_budgets(
    budget: RouteBudget,
    reason: RouteOmissionReason,
) -> None:
    manifest = _manifest()
    receipt = _receipt(
        manifest,
        semantic_scores={"question_crisis": 0.9, "question_auvers": 0.8},
    )
    packet = TaskRouter(manifest, _manifest_hash(manifest)).route_v2(
        receipt.query,
        receipt,
        budget,
    )

    assert [item.question_id for item in packet.questions] == ["question_crisis"]
    assert [item.evidence_id for item in packet.evidence] == ["evidence_crisis"]
    assert [item.source_id for item in packet.sources] == ["source_chronology"]
    omitted = next(item for item in packet.omissions if item.object_id == "question_auvers")
    assert omitted.object_kind is RouteObjectKind.QUESTION
    assert omitted.reason is reason


def test_budget_exceeded_is_explicit_when_candidates_exist_but_no_seed_bundle_fits() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest, semantic_scores={"question_crisis": 0.9})
    budget = _no_expansion_budget(max_evidence=0, max_sources=0)
    packet = TaskRouter(manifest, _manifest_hash(manifest)).route_v2(
        receipt.query,
        receipt,
        budget,
    )

    assert packet.route_status is RouteStatus.BUDGET_EXCEEDED
    assert packet.questions == ()
    assert packet.evidence == ()
    assert packet.sources == ()
    assert packet.route_trace == ()
    assert len(packet.omissions) == 1
    assert packet.omissions[0].object_id == "question_crisis"
    assert packet.omissions[0].reason is RouteOmissionReason.EVIDENCE_BUDGET


def test_v1_1_fails_closed_when_receipt_bindings_are_absent() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest, semantic_scores={"question_auvers": 0.9})
    packet = TaskRouter(manifest, _manifest_hash(manifest)).route_v2(
        receipt.query,
        receipt,
        RouteBudget(),
    )
    payload = packet.model_dump(mode="json")
    del payload["route_receipt_id"]
    del payload["route_receipt_hash"]

    with pytest.raises(
        ValidationError,
        match=r"route_receipt_id|route_receipt_hash",
    ):
        CompactMemoryPacketV1_1.model_validate_json(json.dumps(payload))


def _malformed_receipt(
    manifest: ResearchAtlasManifest,
    *,
    manifest_hash: str | None = None,
    model_profile_hash: str = _MODEL_HASH,
    semantic_threshold: float = 0.5,
    decision: RouteCandidateDecision | None = None,
    semantic_scores: Mapping[str, float] | Iterable[tuple[str, float]] = (),
    lexical_scores: Mapping[str, int] | Iterable[tuple[str, int]] = (),
) -> RouteCandidateReceipt:
    return build_route_candidate_receipt(
        manifest,
        manifest_hash=_manifest_hash(manifest) if manifest_hash is None else manifest_hash,
        query="Gauguin crisis",
        model_profile_hash=model_profile_hash,
        runtime_profile_hash=_RUNTIME_HASH,
        index_hash=_INDEX_HASH,
        semantic_threshold=semantic_threshold,
        semantic_scores=semantic_scores,
        lexical_scores=lexical_scores,
        decision=decision,
    )


def test_receipt_builder_rejects_malformed_lineage_inputs_before_canonicalizing() -> None:
    manifest = _manifest()
    cases: list[tuple[Callable[[], RouteCandidateReceipt], str]] = [
        (
            lambda: _malformed_receipt(manifest, manifest_hash="0" * 64),
            "manifest_hash is stale",
        ),
        (
            lambda: _malformed_receipt(manifest, model_profile_hash="A" * 64),
            "64 lowercase hexadecimal",
        ),
        (lambda: _malformed_receipt(manifest, semantic_threshold=cast(float, 1)), "finite float"),
        (
            lambda: _malformed_receipt(
                manifest,
                decision=cast(RouteCandidateDecision, "routed"),
            ),
            "RouteCandidateDecision",
        ),
        (
            lambda: _malformed_receipt(manifest, semantic_scores=[("Bad ID", 0.9)]),
            "canonical Atlas IDs",
        ),
        (
            lambda: _malformed_receipt(
                manifest,
                semantic_scores=[("question_crisis", 0.9), ("question_crisis", 0.8)],
            ),
            "must be unique",
        ),
        (
            lambda: _malformed_receipt(manifest, lexical_scores=[("Bad ID", 2)]),
            "canonical Atlas IDs",
        ),
        (
            lambda: _malformed_receipt(
                manifest,
                lexical_scores=[("question_crisis", True)],
            ),
            "positive bounded integers",
        ),
        (
            lambda: _malformed_receipt(
                manifest,
                lexical_scores=[("question_crisis", 2), ("question_crisis", 1)],
            ),
            "must be unique",
        ),
        (
            lambda: _malformed_receipt(
                manifest,
                semantic_scores={"question_crisis": 0.9},
                lexical_scores={"question_crisis": 2},
            ),
            "unique across route channels",
        ),
    ]

    for build, message in cases:
        with pytest.raises(ValueError, match=message):
            build()


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (None, "must be a string"),
        (" \t\n", "must not be blank"),
        ("q" * 4_001, "at most 4000 characters"),
    ],
)
def test_receipt_builder_rejects_queries_outside_the_canonical_boundary(
    query: Any,
    message: str,
) -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match=message):
        build_route_candidate_receipt(
            manifest,
            manifest_hash=_manifest_hash(manifest),
            query=query,
            model_profile_hash=_MODEL_HASH,
            runtime_profile_hash=_RUNTIME_HASH,
            index_hash=_INDEX_HASH,
            semantic_threshold=0.5,
        )


def test_receipt_models_reject_wrong_channels_and_noncanonical_float_encodings() -> None:
    with pytest.raises(ValidationError, match="semantic channel lineage"):
        SemanticQuestionCandidate(
            channel=RouteCandidateChannel.LEXICAL,
            question_id="question_crisis",
            score_hex=(0.9).hex(),
            rank=1,
        )
    with pytest.raises(ValidationError, match="lexical channel lineage"):
        LexicalQuestionCandidate(
            channel=RouteCandidateChannel.SEMANTIC,
            question_id="question_crisis",
            score=1,
            rank=1,
        )
    for score_hex, message in (
        ("not-a-float", "canonical float.hex syntax"),
        ("inf", "must be finite"),
        ("0X1.0000000000000P+0", "exact canonical"),
    ):
        with pytest.raises(ValidationError, match=message):
            SemanticQuestionCandidate(
                question_id="question_crisis",
                score_hex=score_hex,
                rank=1,
            )


def test_receipt_rejects_invalid_schema_decision_identity_and_rank_contracts() -> None:
    manifest = _manifest()
    receipt = _receipt(
        manifest,
        semantic_scores={"question_crisis": 0.9, "question_auvers": 0.8},
    )

    non_normalized = receipt.model_dump(mode="json")
    non_normalized["query"] = f" {receipt.query} "
    with pytest.raises(ValidationError, match="stripped and NFC-normalized"):
        _load_receipt(_readdress_receipt(non_normalized))

    wrong_schema = receipt.model_dump(mode="json")
    wrong_schema["schema_id"] = "dithyramba.route_candidate_receipt/9.9"
    with pytest.raises(ValidationError, match="schema_id must be"):
        _load_receipt(_readdress_receipt(wrong_schema))

    empty_routed = receipt.model_dump(mode="json")
    empty_routed["semantic_candidates"] = []
    with pytest.raises(ValidationError, match="require at least one"):
        _load_receipt(_readdress_receipt(empty_routed))

    wrong_id = receipt.model_dump(mode="json")
    wrong_id["receipt_id"] = f"route_candidate_receipt_{'0' * 32}"
    with pytest.raises(ValidationError, match="receipt_id does not match"):
        _load_receipt(wrong_id)

    wrong_hash = receipt.model_dump(mode="json")
    wrong_hash["receipt_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="receipt_hash does not match"):
        _load_receipt(wrong_hash)

    bad_ranks = receipt.model_dump(mode="json")
    bad_ranks["semantic_candidates"][0]["rank"] = 2
    bad_ranks["semantic_candidates"][1]["rank"] = 1
    with pytest.raises(ValidationError, match="ranks must be contiguous"):
        _load_receipt(_readdress_receipt(bad_ranks))

    lexical = _receipt(
        manifest,
        lexical_scores={"question_crisis": 3, "question_auvers": 2},
    ).model_dump(mode="json")
    lexical["lexical_candidates"] = list(reversed(lexical["lexical_candidates"]))
    with pytest.raises(ValidationError, match="lexical candidates must be ordered"):
        _load_receipt(_readdress_receipt(lexical))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "stage": RouteTraceStage.SEED,
                "object_kind": RouteObjectKind.HYPOTHESIS,
                "object_id": "hypothesis_crisis",
                "channel": RouteCandidateChannel.SEMANTIC,
                "channel_rank": 1,
                "semantic_score_hex": (0.9).hex(),
            },
            "seeds must be questions",
        ),
        (
            {
                "stage": RouteTraceStage.SEED,
                "object_kind": RouteObjectKind.QUESTION,
                "object_id": "question_crisis",
            },
            "explicit channel lineage",
        ),
        (
            {
                "stage": RouteTraceStage.SEED,
                "object_kind": RouteObjectKind.QUESTION,
                "object_id": "question_crisis",
                "channel": RouteCandidateChannel.SEMANTIC,
                "channel_rank": 1,
                "semantic_score_hex": (0.9).hex(),
                "parent_object_kind": RouteObjectKind.QUESTION,
                "parent_object_id": "question_crisis",
            },
            "cannot declare a graph-expansion parent",
        ),
        (
            {
                "stage": RouteTraceStage.SEED,
                "object_kind": RouteObjectKind.QUESTION,
                "object_id": "question_crisis",
                "channel": RouteCandidateChannel.SEMANTIC,
                "channel_rank": 1,
                "semantic_score_hex": (0.9).hex(),
                "lexical_score": 1,
            },
            "semantic seeds require only",
        ),
        (
            {
                "stage": RouteTraceStage.SEED,
                "object_kind": RouteObjectKind.QUESTION,
                "object_id": "question_crisis",
                "channel": RouteCandidateChannel.LEXICAL,
                "channel_rank": 1,
                "semantic_score_hex": (0.9).hex(),
            },
            "lexical seeds require only",
        ),
        (
            {
                "stage": RouteTraceStage.GRAPH_EXPANSION,
                "object_kind": RouteObjectKind.GAP,
                "object_id": "gap_crisis_record",
                "channel": RouteCandidateChannel.LEXICAL,
                "channel_rank": 1,
                "lexical_score": 1,
                "parent_object_kind": RouteObjectKind.QUESTION,
                "parent_object_id": "question_crisis",
            },
            "graph expansion cannot carry",
        ),
        (
            {
                "stage": RouteTraceStage.GRAPH_EXPANSION,
                "object_kind": RouteObjectKind.GAP,
                "object_id": "gap_crisis_record",
            },
            "requires an explicit parent",
        ),
    ],
)
def test_route_trace_entries_fail_closed_on_ambiguous_lineage(
    payload: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        RouteTraceEntry.model_validate(payload)


def test_route_v2_rejects_invalid_query_and_stale_router_manifest_binding() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest, semantic_scores={"question_crisis": 0.9})
    router = TaskRouter(manifest, _manifest_hash(manifest))

    for query, message in ((" \n", "must not be blank"), ("q" * 4_001, "at most 4000")):
        with pytest.raises(ValueError, match=message):
            router.route_v2(query, receipt, RouteBudget())

    stale_router = TaskRouter(manifest, "0" * 64)
    with pytest.raises(ValueError, match="router manifest_hash is stale"):
        stale_router.route_v2(receipt.query, receipt, RouteBudget())


def test_route_v2_reports_question_and_gap_budget_boundaries_precisely() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest, semantic_scores={"question_crisis": 0.9})
    router = TaskRouter(manifest, _manifest_hash(manifest))

    no_questions = router.route_v2(
        receipt.query,
        receipt,
        _no_expansion_budget(max_questions=0),
    )
    assert no_questions.route_status is RouteStatus.BUDGET_EXCEEDED
    assert no_questions.omissions[0].reason is RouteOmissionReason.QUESTION_BUDGET

    no_gaps = router.route_v2(
        receipt.query,
        receipt,
        RouteBudget(
            max_questions=1,
            max_linked_hypotheses=1,
            max_relations=0,
            max_timeline_events=1,
            max_gaps=0,
            max_evidence=2,
            max_sources=2,
        ),
    )
    gap_omission = next(
        item for item in no_gaps.omissions if item.object_kind is RouteObjectKind.GAP
    )
    assert gap_omission.reason is RouteOmissionReason.GAP_BUDGET


def test_route_v2_relations_are_admitted_or_omitted_with_explicit_budget_reason() -> None:
    manifest = _manifest_with_relation()
    receipt = _receipt(manifest, semantic_scores={"question_crisis": 0.9})
    router = TaskRouter(manifest, _manifest_hash(manifest))
    base_budget = {
        "max_questions": 1,
        "max_linked_hypotheses": 2,
        "max_relations": 1,
        "max_timeline_events": 0,
        "max_gaps": 0,
        "max_evidence": 2,
        "max_sources": 2,
    }

    admitted = router.route_v2(receipt.query, receipt, RouteBudget(**base_budget))
    assert [item.relation_id for item in admitted.relations] == ["relation_context"]
    relation_trace = next(
        item for item in admitted.route_trace if item.object_kind is RouteObjectKind.RELATION
    )
    assert relation_trace.parent_object_id == "hypothesis_crisis"

    relation_limited = router.route_v2(
        receipt.query,
        receipt,
        RouteBudget(**(base_budget | {"max_relations": 0})),
    )
    relation_omission = next(
        item for item in relation_limited.omissions if item.object_kind is RouteObjectKind.RELATION
    )
    assert relation_omission.reason is RouteOmissionReason.RELATION_BUDGET

    endpoint_limited = router.route_v2(
        receipt.query,
        receipt,
        RouteBudget(**(base_budget | {"max_linked_hypotheses": 1})),
    )
    endpoint_omission = next(
        item for item in endpoint_limited.omissions if item.object_kind is RouteObjectKind.RELATION
    )
    assert endpoint_omission.reason is RouteOmissionReason.REFERENCE_BUDGET


def test_v1_1_packet_rejects_identity_trace_budget_and_status_inconsistencies() -> None:
    manifest = _manifest()
    receipt = _receipt(manifest, semantic_scores={"question_crisis": 0.9})
    router = TaskRouter(manifest, _manifest_hash(manifest))
    routed = router.route_v2(receipt.query, receipt, RouteBudget())

    wrong_id = routed.model_dump(mode="json")
    wrong_id["packet_id"] = f"memory_packet_{'0' * 32}"
    with pytest.raises(ValidationError, match="packet_id does not match"):
        _load_v1_1_packet(wrong_id)

    wrong_hash = routed.model_dump(mode="json")
    wrong_hash["packet_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="packet_hash does not match"):
        _load_v1_1_packet(wrong_hash)

    wrong_receipt_id = routed.model_dump(mode="json")
    wrong_receipt_id["route_receipt_id"] = f"route_candidate_receipt_{'0' * 32}"
    with pytest.raises(ValidationError, match="ID/hash binding is inconsistent"):
        _load_v1_1_packet(_readdress_packet(wrong_receipt_id))

    legacy_match = routed.model_dump(mode="json")
    legacy_match["route_matches"] = [
        {
            "object_kind": "question",
            "object_id": "question_crisis",
            "score": 1,
            "matched_terms": ["crisis"],
        }
    ]
    with pytest.raises(ValidationError, match="uses route_trace"):
        _load_v1_1_packet(_readdress_packet(legacy_match))

    duplicate_trace = routed.model_dump(mode="json")
    duplicate_trace["route_trace"].append(duplicate_trace["route_trace"][0])
    with pytest.raises(ValidationError, match="route_trace object entries must be unique"):
        _load_v1_1_packet(_readdress_packet(duplicate_trace))

    overlap = routed.model_dump(mode="json")
    overlap["omissions"].append(
        {
            "object_kind": "question",
            "object_id": "question_crisis",
            "reason": "question_budget",
        }
    )
    with pytest.raises(ValidationError, match="both admitted and omitted"):
        _load_v1_1_packet(_readdress_packet(overlap))

    incomplete_trace = routed.model_dump(mode="json")
    incomplete_trace["route_trace"].pop()
    with pytest.raises(ValidationError, match="exactly cover every admitted"):
        _load_v1_1_packet(_readdress_packet(incomplete_trace))

    child_before_parent = routed.model_dump(mode="json")
    child_before_parent["route_trace"] = [
        *child_before_parent["route_trace"][1:],
        child_before_parent["route_trace"][0],
    ]
    with pytest.raises(ValidationError, match="parents must precede"):
        _load_v1_1_packet(_readdress_packet(child_before_parent))

    over_budget = routed.model_dump(mode="json")
    over_budget["route_budget"]["max_questions"] = 0
    with pytest.raises(ValidationError, match="question closure exceeds"):
        _load_v1_1_packet(_readdress_packet(over_budget))

    no_match_receipt = _receipt(manifest)
    empty = router.route_v2(no_match_receipt.query, no_match_receipt, RouteBudget())

    routed_without_seed = empty.model_dump(mode="json")
    routed_without_seed["route_status"] = "routed"
    with pytest.raises(ValidationError, match="require an admitted seed"):
        _load_v1_1_packet(_readdress_packet(routed_without_seed))

    no_match_with_memory = routed.model_dump(mode="json")
    no_match_with_memory["route_status"] = "no_match"
    with pytest.raises(ValidationError, match=r"no_match schema 1\.1 packets must be empty"):
        _load_v1_1_packet(_readdress_packet(no_match_with_memory))

    budget_exceeded_without_omission = empty.model_dump(mode="json")
    budget_exceeded_without_omission["route_status"] = "budget_exceeded"
    with pytest.raises(ValidationError, match="empty with explicit omissions"):
        _load_v1_1_packet(_readdress_packet(budget_exceeded_without_omission))
