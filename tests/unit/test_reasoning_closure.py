"""IdeaTrace contracts and deterministic evidence-closure tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dithyramba.answers import (
    AnswerClaimState,
    ClaimEvidenceCaseSet,
    ClaimEvidenceEntailmentDecision,
    ClaimEvidenceEntailmentResult,
    ClaimEvidenceVerdict,
)
from dithyramba.contracts import canonical_content_id, sha256_hex
from dithyramba.reasoning import (
    IdeaTrace,
    IdeaTraceStep,
    ReasoningClosureDecision,
    ReasoningClosureGate,
    ReasoningClosureResult,
    ReasoningGap,
    ReasoningOperation,
    ReasoningQualifier,
    TraceAuthorKind,
)

from .test_claim_evidence_entailment import (
    _answer_with_claim_states,
    _bundle,
    _gate,
    _judgment,
    _receipt,
)
from .test_compact_memory_and_answer_contract import _manifest, _wide_packet

_PROFILE_HASH = sha256_hex(b"idea trace profile v1")
_INSTRUCTION_HASH = sha256_hex(b"public evidence-grounded trace instruction")
_RAW_HASH = sha256_hex(b"bounded public trace response")


def _accepted_trace(
    tmp_path: Path,
) -> tuple[IdeaTrace, ClaimEvidenceCaseSet, ClaimEvidenceEntailmentResult]:
    packet = _wide_packet(_manifest())
    answer = _answer_with_claim_states(
        packet,
        (AnswerClaimState.SUPPORTED, AnswerClaimState.INFERENCE),
    )
    bundle = _bundle(tmp_path, answer=answer)
    receipt = _receipt(
        bundle.case_set,
        (
            _judgment(bundle.case_set, 0, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED),
            _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.SUPPORTED_INFERENCE),
        ),
    )
    entailment = _gate(bundle, receipt)
    first_case, second_case = bundle.case_set.cases
    extracted = IdeaTraceStep.create(
        claim_id=first_case.claim_id,
        case_id=first_case.case_id,
        statement=first_case.claim_text,
        operation=ReasoningOperation.EXTRACT,
        qualifier=ReasoningQualifier.DIRECT,
    )
    inferred = IdeaTraceStep.create(
        claim_id=second_case.claim_id,
        case_id=second_case.case_id,
        statement=second_case.claim_text,
        operation=ReasoningOperation.INFER,
        premise_step_ids=(extracted.step_id,),
        warrant="The second packet claim is an explicitly qualified inference from the first.",
        qualifier=ReasoningQualifier.BOUNDED,
    )
    trace = IdeaTrace.create(
        task_id=bundle.case_set.task_id,
        answer_id=bundle.case_set.answer_id,
        answer_hash=bundle.case_set.answer_hash,
        packet_id=bundle.case_set.packet_id,
        packet_hash=bundle.case_set.packet_hash,
        case_set_id=bundle.case_set.case_set_id,
        case_set_hash=bundle.case_set.case_set_hash,
        receipt_id=receipt.receipt_id,
        receipt_hash=receipt.receipt_hash,
        question="What bounded interpretation follows from these exact claims?",
        author_kind=TraceAuthorKind.MODEL,
        author_profile_id="trace_profile_v1",
        author_profile_hash=_PROFILE_HASH,
        instruction_hash=_INSTRUCTION_HASH,
        raw_response_hash=_RAW_HASH,
        steps=(extracted, inferred),
        final_step_id=inferred.step_id,
        gaps=(
            ReasoningGap(
                question="Does an independent source support the inference?",
                why_it_matters="The chain currently closes over one bounded answer route.",
                next_evidence="Retrieve an independent primary or scholarly source.",
            ),
        ),
    )
    return trace, bundle.case_set, entailment


def test_accepted_trace_is_content_addressed_and_closes(tmp_path: Path) -> None:
    trace, case_set, entailment = _accepted_trace(tmp_path)

    result = ReasoningClosureGate().evaluate(
        trace,
        case_set=case_set,
        entailment=entailment,
    )
    replay = ReasoningClosureGate().evaluate(
        IdeaTrace.model_validate_json(trace.model_dump_json()),
        case_set=case_set,
        entailment=entailment,
    )

    assert entailment.decision is ClaimEvidenceEntailmentDecision.PASSED
    assert result.decision is ReasoningClosureDecision.PASSED
    assert result.review_eligible is True
    assert all(item.closed for item in result.steps)
    assert result == replay
    assert result.closure_hash == replay.closure_hash
    assert "private" not in trace.model_dump_json().lower()


def test_trace_rejects_tampering_or_hidden_topology(tmp_path: Path) -> None:
    trace, _, _ = _accepted_trace(tmp_path)
    payload = trace.model_dump(mode="json")
    payload["steps"][0]["statement"] += " Tampered."
    with pytest.raises(ValidationError, match="step_id"):
        IdeaTrace.model_validate_json(json.dumps(payload))

    extracted = trace.steps[0]
    orphan = IdeaTraceStep.create(
        claim_id="claim_orphan",
        case_id=f"claim_evidence_case_{'1' * 32}",
        statement="An unrelated statement.",
        operation=ReasoningOperation.EXTRACT,
        qualifier=ReasoningQualifier.DIRECT,
    )
    with pytest.raises(ValidationError, match="must contribute"):
        IdeaTrace.create(
            task_id=trace.task_id,
            answer_id=trace.answer_id,
            answer_hash=trace.answer_hash,
            packet_id=trace.packet_id,
            packet_hash=trace.packet_hash,
            case_set_id=trace.case_set_id,
            case_set_hash=trace.case_set_hash,
            receipt_id=trace.receipt_id,
            receipt_hash=trace.receipt_hash,
            question=trace.question,
            author_kind=trace.author_kind,
            author_profile_id=trace.author_profile_id,
            author_profile_hash=trace.author_profile_hash,
            instruction_hash=trace.instruction_hash,
            raw_response_hash=trace.raw_response_hash,
            steps=(extracted, orphan, trace.steps[-1]),
            final_step_id=trace.final_step_id,
        )


def test_stale_receipt_requires_review_without_promotion(tmp_path: Path) -> None:
    trace, case_set, entailment = _accepted_trace(tmp_path)
    stale = trace.model_copy(update={"receipt_hash": "0" * 64})

    result = ReasoningClosureGate().evaluate(
        stale,
        case_set=case_set,
        entailment=entailment,
    )

    assert result.decision is ReasoningClosureDecision.REVIEW_REQUIRED
    assert result.review_eligible is False
    assert all(not item.closed for item in result.steps)
    assert any("content identity" in item for item in result.review_reasons)


def test_failed_semantic_case_blocks_reasoning_closure(tmp_path: Path) -> None:
    trace, case_set, _ = _accepted_trace(tmp_path)
    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    packet = _wide_packet(_manifest())
    answer = _answer_with_claim_states(
        packet,
        (AnswerClaimState.SUPPORTED, AnswerClaimState.INFERENCE),
    )
    bundle = _bundle(failed_root, answer=answer)
    receipt = _receipt(
        bundle.case_set,
        (
            _judgment(bundle.case_set, 0, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED),
            _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.UNSUPPORTED),
        ),
    )
    failed = _gate(bundle, receipt)
    rebound = IdeaTrace.create(
        task_id=trace.task_id,
        answer_id=trace.answer_id,
        answer_hash=trace.answer_hash,
        packet_id=trace.packet_id,
        packet_hash=trace.packet_hash,
        case_set_id=trace.case_set_id,
        case_set_hash=trace.case_set_hash,
        receipt_id=receipt.receipt_id,
        receipt_hash=receipt.receipt_hash,
        question=trace.question,
        author_kind=trace.author_kind,
        author_profile_id=trace.author_profile_id,
        author_profile_hash=trace.author_profile_hash,
        instruction_hash=trace.instruction_hash,
        raw_response_hash=trace.raw_response_hash,
        steps=trace.steps,
        final_step_id=trace.final_step_id,
        gaps=trace.gaps,
    )

    result = ReasoningClosureGate().evaluate(
        rebound,
        case_set=case_set,
        entailment=failed,
    )

    assert failed.decision is ClaimEvidenceEntailmentDecision.FAILED
    assert result.decision is ReasoningClosureDecision.FAILED
    assert result.review_eligible is False
    assert any("semantic evidence gate failed" in item for item in result.violations)


@pytest.mark.parametrize(
    ("operation", "premises", "warrant", "qualifier", "message"),
    [
        (
            ReasoningOperation.EXTRACT,
            ("idea_trace_step_" + "0" * 32,),
            None,
            ReasoningQualifier.DIRECT,
            "cannot have premise",
        ),
        (
            ReasoningOperation.COMPARE,
            (),
            "A comparison.",
            ReasoningQualifier.BOUNDED,
            "at least 2",
        ),
        (
            ReasoningOperation.INFER,
            ("idea_trace_step_" + "0" * 32,),
            None,
            ReasoningQualifier.BOUNDED,
            "require a concise",
        ),
    ],
)
def test_step_shape_fails_closed(
    operation: ReasoningOperation,
    premises: tuple[str, ...],
    warrant: str | None,
    qualifier: ReasoningQualifier,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        IdeaTraceStep.create(
            claim_id="claim_shape",
            case_id=f"claim_evidence_case_{'2' * 32}",
            statement="A bounded statement.",
            operation=operation,
            premise_step_ids=premises,
            warrant=warrant,
            qualifier=qualifier,
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (
            {"premise_step_ids": ["idea_trace_step_" + "1" * 32] * 2},
            "premise_step_ids must be unique",
        ),
        (
            {"counterevidence_ids": ["evidence_one", "evidence_one"]},
            "counterevidence_ids must be unique",
        ),
        (
            {"warrant": "Extracted text needs no derivation warrant."},
            "do not carry a derivation warrant",
        ),
        (
            {"qualifier": ReasoningQualifier.BOUNDED.value},
            "must be direct or contested",
        ),
        (
            {"qualifier": ReasoningQualifier.CONTESTED.value},
            "require explicit counterevidence_ids",
        ),
        (
            {"counterevidence_ids": ["evidence_one"]},
            "reserved for contested steps",
        ),
    ],
)
def test_step_rejects_ambiguous_public_derivation_shapes(
    update: dict[str, object],
    message: str,
) -> None:
    base = IdeaTraceStep.create(
        claim_id="claim_shape",
        case_id=f"claim_evidence_case_{'2' * 32}",
        statement="A bounded statement.",
        operation=ReasoningOperation.EXTRACT,
        qualifier=ReasoningQualifier.DIRECT,
    ).model_dump(mode="json")
    base.update(update)
    semantic = {key: value for key, value in base.items() if key != "step_id"}
    base["step_id"] = canonical_content_id("idea_trace_step", semantic)

    with pytest.raises(ValidationError, match=message):
        IdeaTraceStep.model_validate_json(json.dumps(base))


def test_derived_step_rejects_direct_label_and_self_reference() -> None:
    premise = "idea_trace_step_" + "1" * 32
    with pytest.raises(ValidationError, match="cannot be labelled direct"):
        IdeaTraceStep.create(
            claim_id="claim_derived",
            case_id=f"claim_evidence_case_{'3' * 32}",
            statement="A derived statement.",
            operation=ReasoningOperation.INFER,
            premise_step_ids=(premise,),
            warrant="The premise supports a bounded inference.",
            qualifier=ReasoningQualifier.DIRECT,
        )

    step = IdeaTraceStep.create(
        claim_id="claim_derived",
        case_id=f"claim_evidence_case_{'3' * 32}",
        statement="A derived statement.",
        operation=ReasoningOperation.INFER,
        premise_step_ids=(premise,),
        warrant="The premise supports a bounded inference.",
        qualifier=ReasoningQualifier.BOUNDED,
    )
    with pytest.raises(ValidationError, match="cannot cite itself"):
        IdeaTraceStep.model_validate_json(
            json.dumps(step.model_dump(mode="json") | {"premise_step_ids": [step.step_id]})
        )


def test_trace_rejects_wrong_schema_order_duplicate_gaps_and_hash(tmp_path: Path) -> None:
    trace, _, _ = _accepted_trace(tmp_path)
    payload = trace.model_dump(mode="json")

    with pytest.raises(ValidationError, match="schema_id"):
        IdeaTrace.model_validate_json(
            json.dumps(payload | {"schema_id": "dithyramba.idea_trace/0.0"})
        )
    with pytest.raises(ValidationError, match="final IdeaTrace step"):
        IdeaTrace.model_validate_json(
            json.dumps(payload | {"final_step_id": trace.steps[0].step_id})
        )

    reversed_steps = list(reversed(payload["steps"]))
    with pytest.raises(ValidationError, match="premise steps must exist earlier"):
        IdeaTrace.model_validate_json(
            json.dumps(
                payload
                | {
                    "steps": reversed_steps,
                    "final_step_id": reversed_steps[-1]["step_id"],
                }
            )
        )

    gap = trace.gaps[0]
    with pytest.raises(ValidationError, match="gaps must be unique"):
        IdeaTrace.create(
            task_id=trace.task_id,
            answer_id=trace.answer_id,
            answer_hash=trace.answer_hash,
            packet_id=trace.packet_id,
            packet_hash=trace.packet_hash,
            case_set_id=trace.case_set_id,
            case_set_hash=trace.case_set_hash,
            receipt_id=trace.receipt_id,
            receipt_hash=trace.receipt_hash,
            question=trace.question,
            author_kind=trace.author_kind,
            author_profile_id=trace.author_profile_id,
            author_profile_hash=trace.author_profile_hash,
            instruction_hash=trace.instruction_hash,
            raw_response_hash=trace.raw_response_hash,
            steps=trace.steps,
            final_step_id=trace.final_step_id,
            gaps=(gap, gap),
        )

    with pytest.raises(ValidationError, match="trace_hash"):
        IdeaTrace.model_validate_json(json.dumps(payload | {"trace_hash": "0" * 64}))


def test_trace_rejects_duplicate_claim_and_case_bindings(tmp_path: Path) -> None:
    trace, _, _ = _accepted_trace(tmp_path)
    first = trace.steps[0]
    duplicate_claim = IdeaTraceStep.create(
        claim_id=first.claim_id,
        case_id=f"claim_evidence_case_{'4' * 32}",
        statement="A second statement for the same claim.",
        operation=ReasoningOperation.INFER,
        premise_step_ids=(first.step_id,),
        warrant="This intentionally duplicates the claim binding.",
        qualifier=ReasoningQualifier.BOUNDED,
    )
    duplicate_case = IdeaTraceStep.create(
        claim_id="claim_other",
        case_id=first.case_id,
        statement="A second statement for the same evidence case.",
        operation=ReasoningOperation.INFER,
        premise_step_ids=(first.step_id,),
        warrant="This intentionally duplicates the case binding.",
        qualifier=ReasoningQualifier.BOUNDED,
    )
    with pytest.raises(ValidationError, match="claim may appear only once"):
        IdeaTrace.create(
            task_id=trace.task_id,
            answer_id=trace.answer_id,
            answer_hash=trace.answer_hash,
            packet_id=trace.packet_id,
            packet_hash=trace.packet_hash,
            case_set_id=trace.case_set_id,
            case_set_hash=trace.case_set_hash,
            receipt_id=trace.receipt_id,
            receipt_hash=trace.receipt_hash,
            question=trace.question,
            author_kind=trace.author_kind,
            author_profile_id=trace.author_profile_id,
            author_profile_hash=trace.author_profile_hash,
            instruction_hash=trace.instruction_hash,
            raw_response_hash=trace.raw_response_hash,
            steps=(first, duplicate_claim),
            final_step_id=duplicate_claim.step_id,
        )
    with pytest.raises(ValidationError, match="case may appear only once"):
        IdeaTrace.create(
            task_id=trace.task_id,
            answer_id=trace.answer_id,
            answer_hash=trace.answer_hash,
            packet_id=trace.packet_id,
            packet_hash=trace.packet_hash,
            case_set_id=trace.case_set_id,
            case_set_hash=trace.case_set_hash,
            receipt_id=trace.receipt_id,
            receipt_hash=trace.receipt_hash,
            question=trace.question,
            author_kind=trace.author_kind,
            author_profile_id=trace.author_profile_id,
            author_profile_hash=trace.author_profile_hash,
            instruction_hash=trace.instruction_hash,
            raw_response_hash=trace.raw_response_hash,
            steps=(first, duplicate_case),
            final_step_id=duplicate_case.step_id,
        )


def test_gate_reports_missing_or_stale_case_outcomes(tmp_path: Path) -> None:
    trace, case_set, entailment = _accepted_trace(tmp_path)
    scenarios = (
        (
            case_set.model_copy(update={"cases": case_set.cases[1:]}),
            entailment,
            "claim-evidence case is absent",
        ),
        (
            case_set,
            entailment.model_copy(update={"cases": entailment.cases[1:]}),
            "semantic case result is absent",
        ),
        (
            case_set,
            entailment.model_copy(
                update={
                    "cases": (
                        entailment.cases[0].model_copy(update={"claim_id": "claim_other"}),
                        entailment.cases[1],
                    )
                }
            ),
            "semantic case result names a different claim",
        ),
        (
            case_set,
            entailment.model_copy(
                update={
                    "cases": (
                        entailment.cases[0].model_copy(
                            update={"decision": ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED}
                        ),
                        entailment.cases[1],
                    )
                }
            ),
            "semantic evidence gate is review_required",
        ),
    )

    for selected_cases, selected_entailment, expected in scenarios:
        result = ReasoningClosureGate().evaluate(
            trace,
            case_set=selected_cases,
            entailment=selected_entailment,
        )
        assert result.decision is ReasoningClosureDecision.REVIEW_REQUIRED
        assert any(expected in item for item in result.review_reasons)


def test_gate_rejects_stale_top_level_dependencies(tmp_path: Path) -> None:
    trace, case_set, entailment = _accepted_trace(tmp_path)
    mismatches = (
        (
            case_set.model_copy(update={"answer_hash": "0" * 64}),
            entailment,
            "different answer/case set",
        ),
        (
            case_set,
            entailment.model_copy(update={"case_set_hash": "0" * 64}),
            "different case set",
        ),
        (
            case_set,
            entailment.model_copy(update={"receipt_hash": "0" * 64}),
            "different semantic judgment receipt",
        ),
    )
    for selected_cases, selected_entailment, expected in mismatches:
        result = ReasoningClosureGate().evaluate(
            trace,
            case_set=selected_cases,
            entailment=selected_entailment,
        )
        assert result.decision is ReasoningClosureDecision.REVIEW_REQUIRED
        assert any(expected in item for item in result.review_reasons)


def test_gate_blocks_direct_extraction_from_inference_claim(tmp_path: Path) -> None:
    trace, case_set, entailment = _accepted_trace(tmp_path)
    inference_case = case_set.cases[1]
    step = IdeaTraceStep.create(
        claim_id=inference_case.claim_id,
        case_id=inference_case.case_id,
        statement=inference_case.claim_text,
        operation=ReasoningOperation.EXTRACT,
        qualifier=ReasoningQualifier.DIRECT,
    )
    rebound = IdeaTrace.create(
        task_id=trace.task_id,
        answer_id=trace.answer_id,
        answer_hash=trace.answer_hash,
        packet_id=trace.packet_id,
        packet_hash=trace.packet_hash,
        case_set_id=trace.case_set_id,
        case_set_hash=trace.case_set_hash,
        receipt_id=trace.receipt_id,
        receipt_hash=trace.receipt_hash,
        question=trace.question,
        author_kind=trace.author_kind,
        author_profile_id=trace.author_profile_id,
        author_profile_hash=trace.author_profile_hash,
        instruction_hash=trace.instruction_hash,
        raw_response_hash=trace.raw_response_hash,
        steps=(step,),
        final_step_id=step.step_id,
    )

    result = ReasoningClosureGate().evaluate(
        rebound,
        case_set=case_set,
        entailment=entailment,
    )
    assert result.decision is ReasoningClosureDecision.FAILED
    assert any("direct extraction requires" in item for item in result.violations)


def test_closure_contract_rejects_false_pass_and_identity_tampering(tmp_path: Path) -> None:
    trace, case_set, entailment = _accepted_trace(tmp_path)
    result = ReasoningClosureGate().evaluate(
        trace,
        case_set=case_set,
        entailment=entailment,
    )
    payload = result.model_dump(mode="json")
    with pytest.raises(ValidationError, match="schema_id"):
        ReasoningClosureResult.model_validate_json(
            json.dumps(payload | {"schema_id": "dithyramba.reasoning_closure/0.0"})
        )
    with pytest.raises(ValidationError, match="review-eligible"):
        ReasoningClosureResult.model_validate_json(json.dumps(payload | {"review_eligible": False}))
    with pytest.raises(ValidationError, match="without findings"):
        ReasoningClosureResult.model_validate_json(
            json.dumps(payload | {"violations": ["fabricated"]})
        )
    with pytest.raises(ValidationError, match="closure_id"):
        ReasoningClosureResult.model_validate_json(
            json.dumps(payload | {"closure_id": "reasoning_closure_" + "0" * 32})
        )
    with pytest.raises(ValidationError, match="closure_hash"):
        ReasoningClosureResult.model_validate_json(json.dumps(payload | {"closure_hash": "0" * 64}))
