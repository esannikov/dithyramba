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
from dithyramba.recall import CandidateQualityAssessment

HASH_A = "a" * 64
HASH_B = "b" * 64
SESSION_ID = "research_session_" + "1" * 32
EVENT_ID = "session_event_" + "2" * 32


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema_id"),
        ("duplicate_candidates", "unique fragment IDs"),
        ("duplicate_assessments", "unique fragment IDs"),
        ("assessment_mismatch", "quality-admitted fragments"),
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
