"""Identity and tamper tests for gate-bound answer preparations."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dithyramba.evidence import (
    EvidenceAnswerability,
    EvidenceCandidate,
    EvidenceCoverageGate,
    EvidenceGateSpec,
    EvidenceRequirement,
)
from dithyramba.interactive import AgentAnswerPreparation, AgentSourceDrilldown
from dithyramba.interactive.service import _compact_ready_candidates
from dithyramba.recall import CandidateNoiseReason, CandidateQualityAssessment

HASH_A = "a" * 64
HASH_B = "b" * 64
SESSION_ID = "research_session_" + "1" * 32
EVENT_ID = "session_event_" + "2" * 32


def _candidate(
    fragment_id: str,
    *,
    rank: int,
    text: str,
    source_id: str,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        rank=rank,
        source_fragment_id=fragment_id,
        source_id=source_id,
        text=text,
        source_address={
            "schema": "dithyramba.source_address/1.0",
            "kind": "markdown",
            "heading_path": ["Evidence"],
            "line_start": rank,
            "line_end": rank,
            "char_start": 0,
            "char_end": len(text),
        },
        source_kind="book",
        source_family=f"family_{source_id}",
        authority="unclassified",
        independence_group=source_id,
    )


def _drilldown() -> AgentSourceDrilldown:
    return AgentSourceDrilldown.create(
        question="eyeline control",
        source_ids=("source_book",),
        retrieval_result_hash=HASH_A,
        candidate_count=2,
        selected_fragment_ids=("fragment_passage",),
        filtered_out_count=1,
    )


def _ready_preparation() -> AgentAnswerPreparation:
    text = "The passage names eyeline control as the decisive technique."
    candidate = EvidenceCandidate(
        rank=1,
        source_fragment_id="fragment_passage",
        source_id="source_book",
        text=text,
        source_address={
            "schema": "dithyramba.source_address/1.0",
            "kind": "markdown",
            "heading_path": ["Technique"],
            "line_start": 2,
            "line_end": 2,
            "char_start": 10,
            "char_end": 10 + len(text),
        },
        source_kind="book",
        source_family="family_book",
        authority="unclassified",
        independence_group="source_book",
    )
    spec = EvidenceGateSpec(
        query_key="eyeline_control",
        question="How is eyeline control used?",
        expected_answerability=EvidenceAnswerability.ANSWERABLE,
        requirements=(
            EvidenceRequirement(
                key="exact_technique",
                label="Exact named technique",
                allowed_source_ids=("source_book",),
                anchor_groups=(("eyeline control",),),
            ),
        ),
    )
    candidates = (candidate,)
    return AgentAnswerPreparation.create(
        session_id=SESSION_ID,
        evidence_event_id=EVENT_ID,
        evidence_packet_id="packet_answer",
        evidence_packet_hash=HASH_A,
        gate_spec=spec,
        gate_result=EvidenceCoverageGate(spec).evaluate(candidates),
        candidates=candidates,
        quality_assessments=(
            CandidateQualityAssessment(
                source_fragment_id="fragment_passage",
                admitted=True,
            ),
        ),
        drilldown=_drilldown(),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_id", "wrong", "schema_id"),
        ("source_ids", (), "unique Source IDs"),
        ("source_ids", ("source_book", "source_book"), "unique Source IDs"),
        (
            "selected_fragment_ids",
            ("fragment_passage", "fragment_passage"),
            "fragment IDs must be unique",
        ),
        ("drilldown_hash", HASH_B, "drilldown_hash"),
    ],
)
def test_source_drilldown_rejects_identity_tampering(
    field: str,
    value: object,
    message: str,
) -> None:
    payload: dict[str, Any] = _drilldown().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        AgentSourceDrilldown.model_validate(payload)


def test_gap_preparation_has_an_explicit_non_answer_mode() -> None:
    spec = EvidenceGateSpec(
        query_key="known_gap",
        question="Is the exact event present?",
        expected_answerability=EvidenceAnswerability.NOT_IN_CORPUS,
        requirements=(
            EvidenceRequirement(
                key="overturn_gap",
                label="Evidence that would overturn the gap",
                anchor_groups=(("exact event",),),
            ),
        ),
    )
    result = EvidenceCoverageGate(spec).evaluate(())

    preparation = AgentAnswerPreparation.create(
        session_id=SESSION_ID,
        evidence_event_id=EVENT_ID,
        evidence_packet_id="packet_gap",
        evidence_packet_hash=HASH_A,
        gate_spec=spec,
        gate_result=result,
        candidates=(),
        quality_assessments=(),
        drilldown=None,
    )

    assert preparation.response_mode == "gap"


def test_ready_candidates_are_compacted_to_the_minimal_covering_set() -> None:
    spec = EvidenceGateSpec(
        query_key="compact_ready",
        question="Where is the exact technique?",
        expected_answerability=EvidenceAnswerability.ANSWERABLE,
        requirements=(
            EvidenceRequirement(
                key="exact_technique",
                label="Exact technique",
                anchor_groups=(("triangulated eyeline",),),
            ),
        ),
    )
    candidates = (
        _candidate(
            "fragment_first",
            rank=1,
            text="The method uses triangulated eyeline control.",
            source_id="source_first",
        ),
        _candidate(
            "fragment_redundant",
            rank=2,
            text="A second passage repeats triangulated eyeline control.",
            source_id="source_second",
        ),
        _candidate(
            "fragment_related",
            rank=3,
            text="The chapter discusses dialogue staging.",
            source_id="source_third",
        ),
    )
    assessments = (
        *(
            CandidateQualityAssessment(source_fragment_id=item.source_fragment_id, admitted=True)
            for item in candidates
        ),
        CandidateQualityAssessment(
            source_fragment_id="fragment_noise",
            admitted=False,
            reasons=(CandidateNoiseReason.TOPIC_DRIFT,),
        ),
    )

    compact, compact_assessments, result = _compact_ready_candidates(
        gate_spec=spec,
        candidates=candidates,
        assessments=assessments,
    )

    assert tuple(item.source_fragment_id for item in compact) == ("fragment_first",)
    assert tuple(item.rank for item in compact) == (1,)
    assert result.decision.value == "ready"
    assert tuple(item.source_fragment_id for item in compact_assessments) == (
        "fragment_first",
        "fragment_noise",
    )


def test_ready_compaction_keeps_required_independent_sources_and_missing_input() -> None:
    spec = EvidenceGateSpec(
        query_key="independent_support",
        question="Which independent sources support the event?",
        expected_answerability=EvidenceAnswerability.ANSWERABLE,
        requirements=(
            EvidenceRequirement(
                key="two_sources",
                label="Two independent sources",
                anchor_groups=(("dated event",),),
                min_independent_groups=2,
            ),
        ),
        min_total_independent_groups=2,
    )
    candidates = (
        _candidate(
            "fragment_source_a",
            rank=4,
            text="The archive records the dated event.",
            source_id="source_a",
        ),
        _candidate(
            "fragment_source_b",
            rank=9,
            text="The catalogue independently records the dated event.",
            source_id="source_b",
        ),
    )
    assessments = tuple(
        CandidateQualityAssessment(source_fragment_id=item.source_fragment_id, admitted=True)
        for item in candidates
    )

    compact, _compact_assessments, result = _compact_ready_candidates(
        gate_spec=spec,
        candidates=candidates,
        assessments=assessments,
    )
    missing, missing_assessments, missing_result = _compact_ready_candidates(
        gate_spec=spec,
        candidates=candidates[:1],
        assessments=assessments[:1],
    )

    assert len(compact) == 2
    assert tuple(item.rank for item in compact) == (1, 2)
    assert result.decision.value == "ready"
    assert missing == candidates[:1]
    assert missing_assessments == assessments[:1]
    assert missing_result.decision.value == "insufficient"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema_id"),
        ("duplicate_candidates", "unique fragment IDs"),
        ("duplicate_assessments", "unique fragment IDs"),
        ("assessment_mismatch", "quality-eligible fragments"),
        ("gate_mismatch", "gate_result"),
        ("mode", "response_mode"),
        ("hash", "preparation_hash"),
    ],
)
def test_answer_preparation_rejects_identity_tampering(
    mutation: str,
    message: str,
) -> None:
    ready = _ready_preparation()
    payload: dict[str, Any] = ready.model_dump(mode="python")
    if mutation == "schema":
        payload["schema_id"] = "wrong"
    elif mutation == "duplicate_candidates":
        payload["candidates"] = (payload["candidates"][0], payload["candidates"][0])
    elif mutation == "duplicate_assessments":
        payload["quality_assessments"] = (
            payload["quality_assessments"][0],
            payload["quality_assessments"][0],
        )
    elif mutation == "assessment_mismatch":
        payload["quality_assessments"] = ()
    elif mutation == "gate_mismatch":
        payload["gate_result"] = EvidenceCoverageGate(ready.gate_spec).evaluate(())
    elif mutation == "mode":
        payload["response_mode"] = "blocked"
    else:
        payload["preparation_hash"] = HASH_B

    with pytest.raises(ValidationError, match=message):
        AgentAnswerPreparation.model_validate(payload)
