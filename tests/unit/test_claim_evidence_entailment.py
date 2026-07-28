"""Semantic claim/evidence receipt and unified-validator tests."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import ValidationError

from dithyramba.answers import (
    AnswerClaimState,
    AnswerContractGate,
    AnswerContractResult,
    AnswerCoverageDecision,
    AnswerCoverageGate,
    AnswerCoverageResult,
    AnswerCoverageSpec,
    AnswerDecision,
    ClaimEvidenceCaseSet,
    ClaimEvidenceCaseSetBuilder,
    ClaimEvidenceContractError,
    ClaimEvidenceEntailmentDecision,
    ClaimEvidenceEntailmentGate,
    ClaimEvidenceEntailmentResult,
    ClaimEvidenceJudgeKind,
    ClaimEvidenceJudgment,
    ClaimEvidenceJudgmentReceipt,
    ClaimEvidenceVerdict,
    ResearchAnswer,
    UnsupportedClaimSpan,
)
from dithyramba.atlas import CompactMemoryPacket
from dithyramba.contracts import sha256_hex
from dithyramba.research import (
    ResearchAnswerValidationDecision,
    ResearchAnswerValidationResult,
    ResearchAnswerValidator,
)

from .test_compact_memory_and_answer_contract import (
    _coverage_spec,
    _manifest,
    _wide_answer,
    _wide_packet,
    _write_artifacts,
)

_PROFILE_ID = "judge_multilingual_v1"
_PROFILE_HASH = sha256_hex(b"judge multilingual v1")
_PROMPT_HASH = sha256_hex(b"claim evidence prompt v1")
_RAW_HASH = sha256_hex(b"external judgment response")


@dataclass(frozen=True)
class _Bundle:
    packet: CompactMemoryPacket
    answer: ResearchAnswer
    spec: AnswerCoverageSpec
    case_set: ClaimEvidenceCaseSet
    exact: AnswerContractResult
    coverage: AnswerCoverageResult


class _ValidationArgs(TypedDict):
    task_id: str
    packet: CompactMemoryPacket
    coverage_spec: AnswerCoverageSpec
    artifact_root: Path
    expected_judge_kind: ClaimEvidenceJudgeKind
    expected_judge_profile_id: str
    expected_judge_profile_hash: str
    expected_prompt_hash: str
    expected_raw_response_hash: str


def _bundle(tmp_path: Path, *, answer: ResearchAnswer | None = None) -> _Bundle:
    packet = _wide_packet(_manifest())
    _write_artifacts(tmp_path)
    selected_answer = _wide_answer(packet, include_work=True) if answer is None else answer
    spec = _coverage_spec(packet)
    exact = AnswerContractGate().evaluate(
        selected_answer,
        task_id="task_crisis",
        packet=packet,
        artifact_root=tmp_path,
    )
    coverage = AnswerCoverageGate().evaluate(selected_answer, spec=spec, packet=packet)
    case_set = ClaimEvidenceCaseSetBuilder().build(
        selected_answer,
        task_id="task_crisis",
        packet=packet,
        coverage_spec=spec,
    )
    return _Bundle(packet, selected_answer, spec, case_set, exact, coverage)


def _judgment(
    case_set: ClaimEvidenceCaseSet,
    index: int,
    verdict: ClaimEvidenceVerdict,
    *,
    spans: tuple[UnsupportedClaimSpan, ...] = (),
) -> ClaimEvidenceJudgment:
    case = case_set.cases[index]
    return ClaimEvidenceJudgment(
        case_id=case.case_id,
        claim_id=case.claim_id,
        verdict=verdict,
        unsupported_claim_spans=spans,
        reason=f"External semantic review: {verdict.value}.",
    )


def _receipt(
    case_set: ClaimEvidenceCaseSet,
    judgments: tuple[ClaimEvidenceJudgment, ...] | None = None,
    *,
    profile_id: str = _PROFILE_ID,
) -> ClaimEvidenceJudgmentReceipt:
    selected = (
        tuple(
            _judgment(case_set, index, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED)
            for index in range(len(case_set.cases))
        )
        if judgments is None
        else judgments
    )
    return ClaimEvidenceJudgmentReceipt.create(
        case_set=case_set,
        judge_kind=ClaimEvidenceJudgeKind.MODEL,
        judge_profile_id=profile_id,
        judge_profile_hash=_PROFILE_HASH,
        prompt_hash=_PROMPT_HASH,
        raw_response_hash=_RAW_HASH,
        judgments=selected,
    )


def _gate(
    bundle: _Bundle,
    receipt: ClaimEvidenceJudgmentReceipt | None,
) -> ClaimEvidenceEntailmentResult:
    return ClaimEvidenceEntailmentGate().evaluate(
        bundle.case_set,
        answer_contract_result=bundle.exact,
        answer_coverage_result=bundle.coverage,
        receipt=receipt,
        expected_judge_kind=ClaimEvidenceJudgeKind.MODEL,
        expected_judge_profile_id=_PROFILE_ID,
        expected_judge_profile_hash=_PROFILE_HASH,
        expected_prompt_hash=_PROMPT_HASH,
        expected_raw_response_hash=_RAW_HASH,
    )


def _answer_with_claim_states(
    packet: CompactMemoryPacket,
    states: tuple[AnswerClaimState, AnswerClaimState],
) -> ResearchAnswer:
    payload = _wide_answer(packet, include_work=True).model_dump(mode="json")
    for claim, state in zip(payload["claims"], states, strict=True):
        claim["status"] = state.value
    return ResearchAnswer.model_validate_json(json.dumps(payload))


def test_case_set_hash_is_stable_and_hidden_context_is_excluded(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    second = ClaimEvidenceCaseSetBuilder().build(
        bundle.answer,
        task_id="task_crisis",
        packet=bundle.packet,
        coverage_spec=bundle.spec,
    )

    assert second == bundle.case_set
    assert second.case_set_id == bundle.case_set.case_set_id
    assert second.case_set_hash == bundle.case_set.case_set_hash
    serialized = second.model_dump_json()
    assert "Gauguin arrived in Arles" not in serialized  # wider excerpt
    assert '"locator"' not in serialized
    assert '"source_address"' not in serialized
    assert '"heading_path"' not in serialized
    assert "Подія підтверджена" not in serialized  # Atlas synthesis
    assert "exact_quote" in serialized
    assert "source_kind" in serialized
    assert "voice_kind" in serialized
    assert "evidence_role" in serialized
    assert "limitation" in serialized


def test_case_set_models_reject_tampered_or_ambiguous_content(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    case = bundle.case_set.cases[0]
    citation = case.citations[0]
    with pytest.raises(ValidationError, match="unique evidence IDs"):
        type(case)(
            case_id=case.case_id,
            claim_id=case.claim_id,
            claim_text=case.claim_text,
            claim_state=case.claim_state,
            citations=(citation, citation),
            unquoted_numeric_literals=case.unquoted_numeric_literals,
        )
    with pytest.raises(ValidationError, match="numeric literals must be unique"):
        type(case)(
            case_id=case.case_id,
            claim_id=case.claim_id,
            claim_text=case.claim_text,
            claim_state=case.claim_state,
            citations=case.citations,
            unquoted_numeric_literals=("1888", "1888"),
        )
    with pytest.raises(ValidationError, match="case_id"):
        type(case)(
            case_id=f"claim_evidence_case_{'0' * 32}",
            claim_id=case.claim_id,
            claim_text=case.claim_text,
            claim_state=case.claim_state,
            citations=case.citations,
            unquoted_numeric_literals=case.unquoted_numeric_literals,
        )

    schema_tamper = bundle.case_set.model_dump(mode="json")
    schema_tamper["schema_id"] = "wrong"
    with pytest.raises(ValidationError, match="schema_id"):
        ClaimEvidenceCaseSet.model_validate_json(json.dumps(schema_tamper))
    duplicate_case = bundle.case_set.model_dump(mode="json")
    duplicate_case["cases"].append(duplicate_case["cases"][0])
    with pytest.raises(ValidationError, match="case_id values must be unique"):
        ClaimEvidenceCaseSet.model_validate_json(json.dumps(duplicate_case))
    id_tamper = bundle.case_set.model_dump(mode="json")
    id_tamper["case_set_id"] = f"claim_evidence_case_set_{'0' * 32}"
    with pytest.raises(ValidationError, match="case_set_id"):
        ClaimEvidenceCaseSet.model_validate_json(json.dumps(id_tamper))


def test_builder_rejects_broken_bindings_and_joins(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    builder = ClaimEvidenceCaseSetBuilder()
    with pytest.raises(ClaimEvidenceContractError, match="task_id"):
        builder.build(
            bundle.answer,
            task_id="task_other",
            packet=bundle.packet,
            coverage_spec=bundle.spec,
        )
    with pytest.raises(ClaimEvidenceContractError, match="exactly bound"):
        builder.build(
            bundle.answer,
            packet=bundle.packet,
            coverage_spec=bundle.spec.model_copy(update={"packet_hash": "0" * 64}),
        )

    broken_claim = bundle.answer.claims[0].model_copy(
        update={"evidence_ids": ("evidence_missing",)}
    )
    missing_evidence_answer = bundle.answer.model_copy(
        update={"claims": (broken_claim, bundle.answer.claims[1])}
    )
    with pytest.raises(ClaimEvidenceContractError, match="not joined"):
        builder.build(
            missing_evidence_answer,
            packet=bundle.packet,
            coverage_spec=bundle.spec,
        )

    wrong_citation = bundle.answer.citations[0].model_copy(update={"source_id": "source_letter"})
    wrong_source_answer = bundle.answer.model_copy(
        update={"citations": (wrong_citation, bundle.answer.citations[1])}
    )
    with pytest.raises(ClaimEvidenceContractError, match="packet source"):
        builder.build(
            wrong_source_answer,
            packet=bundle.packet,
            coverage_spec=bundle.spec,
        )

    packet_without_source = bundle.packet.model_copy(
        update={
            "sources": tuple(
                item for item in bundle.packet.sources if item.source_id != "source_chronology"
            )
        }
    )
    with pytest.raises(ClaimEvidenceContractError, match="source outside"):
        builder.build(
            bundle.answer,
            packet=packet_without_source,
            coverage_spec=bundle.spec,
        )


def test_multilingual_paraphrase_passes_only_via_bound_receipt(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    result = _gate(bundle, _receipt(bundle.case_set))

    assert bundle.case_set.cases[0].claim_text.startswith("Криза")
    assert bundle.case_set.cases[0].citations[0].exact_quote.startswith("Vincent")
    assert result.decision is ClaimEvidenceEntailmentDecision.PASSED
    assert result.promotion_eligible is True


def test_unquoted_date_is_advisory_but_partial_receipt_blocks(tmp_path: Path) -> None:
    packet = _wide_packet(_manifest())
    answer = _wide_answer(packet, include_work=True)
    payload = answer.model_dump(mode="json")
    payload["claims"][1]["text"] = "У 1890 році в Auvers він продовжував працювати."
    dated = ResearchAnswer.model_validate_json(json.dumps(payload))
    bundle = _bundle(tmp_path, answer=dated)
    case = bundle.case_set.cases[1]
    date_start = case.claim_text.index("1890")
    partial = _judgment(
        bundle.case_set,
        1,
        ClaimEvidenceVerdict.PARTIAL,
        spans=(
            UnsupportedClaimSpan(
                start=date_start,
                end=date_start + 4,
                text="1890",
            ),
        ),
    )
    receipt = _receipt(
        bundle.case_set,
        (_judgment(bundle.case_set, 0, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED), partial),
    )

    assert case.unquoted_numeric_literals == ("1890",)
    result = _gate(bundle, receipt)
    assert result.decision is ClaimEvidenceEntailmentDecision.FAILED
    assert result.promotion_eligible is False


def test_one_claim_can_expose_multiple_exact_quotes(tmp_path: Path) -> None:
    packet = _wide_packet(_manifest())
    payload = _wide_answer(packet, include_work=True).model_dump(mode="json")
    payload["claims"] = [
        {
            "claim_id": "claim_combined",
            "text": "Криза сталася, а пізніше Vincent продовжував працювати.",
            "status": "qualified",
            "evidence_ids": ["evidence_crisis", "evidence_work"],
        }
    ]
    answer = ResearchAnswer.model_validate_json(json.dumps(payload))
    bundle = _bundle(tmp_path, answer=answer)

    assert [item.evidence_id for item in bundle.case_set.cases[0].citations] == [
        "evidence_crisis",
        "evidence_work",
    ]
    assert _gate(bundle, _receipt(bundle.case_set)).decision is (
        ClaimEvidenceEntailmentDecision.PASSED
    )


@pytest.mark.parametrize("mode", ["missing", "duplicate", "extra"])
def test_judgment_closure_errors_require_review(tmp_path: Path, mode: str) -> None:
    bundle = _bundle(tmp_path)
    normal = tuple(
        _judgment(bundle.case_set, index, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED)
        for index in range(len(bundle.case_set.cases))
    )
    if mode == "missing":
        judgments = normal[:-1]
    elif mode == "duplicate":
        judgments = (*normal, normal[0])
    else:
        judgments = (
            *normal,
            ClaimEvidenceJudgment(
                case_id=f"claim_evidence_case_{'0' * 32}",
                claim_id="claim_extra",
                verdict=ClaimEvidenceVerdict.DIRECTLY_SUPPORTED,
                unsupported_claim_spans=(),
                reason="This case was not requested.",
            ),
        )

    result = _gate(bundle, _receipt(bundle.case_set, judgments))
    assert result.decision is ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    assert any(mode in item for item in result.review_reasons)


def test_stale_wrong_profile_uncertain_and_hash_mismatch_fail_closed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    receipt = _receipt(bundle.case_set)
    changed_payload = bundle.answer.model_dump(mode="json")
    changed_payload["claims"][0]["text"] += " Уточнення."
    changed = ResearchAnswer.model_validate_json(json.dumps(changed_payload))
    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    changed_bundle = _bundle(changed_root, answer=changed)

    stale = _gate(changed_bundle, receipt)
    wrong_profile = _gate(bundle, _receipt(bundle.case_set, profile_id="judge_other_v1"))
    uncertain_receipt = _receipt(
        bundle.case_set,
        (
            _judgment(bundle.case_set, 0, ClaimEvidenceVerdict.UNCERTAIN),
            _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED),
        ),
    )
    uncertain = _gate(bundle, uncertain_receipt)

    assert stale.decision is ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    assert wrong_profile.decision is ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    assert uncertain.decision is ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    tampered = receipt.model_dump(mode="json")
    tampered["receipt_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="receipt_hash"):
        ClaimEvidenceJudgmentReceipt.model_validate_json(json.dumps(tampered))
    tampered_case_set = bundle.case_set.model_dump(mode="json")
    tampered_case_set["case_set_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="case_set_hash"):
        ClaimEvidenceCaseSet.model_validate_json(json.dumps(tampered_case_set))


def test_malformed_spans_require_review_or_fail_schema(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    case = bundle.case_set.cases[0]
    out_of_bounds = ClaimEvidenceJudgment(
        case_id=case.case_id,
        claim_id=case.claim_id,
        verdict=ClaimEvidenceVerdict.PARTIAL,
        unsupported_claim_spans=(
            UnsupportedClaimSpan(start=0, end=len(case.claim_text) + 1, text="wrong"),
        ),
        reason="The marked text is not supported.",
    )
    receipt = _receipt(
        bundle.case_set,
        (out_of_bounds, _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED)),
    )

    assert _gate(bundle, receipt).decision is ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    with pytest.raises(ValidationError, match="start < end"):
        UnsupportedClaimSpan(start=5, end=5, text="x")
    with pytest.raises(ValidationError, match="claim order"):
        ClaimEvidenceJudgment(
            case_id=case.case_id,
            claim_id=case.claim_id,
            verdict=ClaimEvidenceVerdict.PARTIAL,
            unsupported_claim_spans=(
                UnsupportedClaimSpan(start=6, end=7, text=case.claim_text[6:7]),
                UnsupportedClaimSpan(start=0, end=1, text=case.claim_text[0:1]),
            ),
            reason="Out-of-order spans.",
        )
    with pytest.raises(ValidationError, match="must not overlap"):
        ClaimEvidenceJudgment(
            case_id=case.case_id,
            claim_id=case.claim_id,
            verdict=ClaimEvidenceVerdict.PARTIAL,
            unsupported_claim_spans=(
                UnsupportedClaimSpan(start=0, end=3, text=case.claim_text[0:3]),
                UnsupportedClaimSpan(start=2, end=4, text=case.claim_text[2:4]),
            ),
            reason="Overlapping spans.",
        )


def test_receipt_and_gate_integrity_guards(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    receipt = _receipt(bundle.case_set)
    receipt_json = receipt.model_dump(mode="json")
    receipt_json["schema_id"] = "wrong"
    with pytest.raises(ValidationError, match="schema_id"):
        ClaimEvidenceJudgmentReceipt.model_validate_json(json.dumps(receipt_json))
    receipt_json = receipt.model_dump(mode="json")
    receipt_json["receipt_id"] = f"claim_evidence_judgment_receipt_{'0' * 32}"
    with pytest.raises(ValidationError, match="receipt_id"):
        ClaimEvidenceJudgmentReceipt.model_validate_json(json.dumps(receipt_json))

    tampered_receipt = receipt.model_copy(
        update={
            "schema_id": "wrong",
            "receipt_id": f"claim_evidence_judgment_receipt_{'0' * 32}",
            "receipt_hash": "0" * 64,
        }
    )
    tampered_result = _gate(bundle, tampered_receipt)
    assert tampered_result.decision is ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    assert any("schema" in item for item in tampered_result.review_reasons)
    assert any("receipt ID" in item for item in tampered_result.review_reasons)
    assert any("receipt hash" in item for item in tampered_result.review_reasons)

    wrong_policy = ClaimEvidenceEntailmentGate().evaluate(
        bundle.case_set,
        answer_contract_result=bundle.exact,
        answer_coverage_result=bundle.coverage,
        receipt=receipt,
        expected_judge_kind=ClaimEvidenceJudgeKind.HUMAN,
        expected_judge_profile_hash="0" * 64,
        expected_prompt_hash="0" * 64,
        expected_raw_response_hash="0" * 64,
    )
    assert wrong_policy.decision is ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    assert any("judge kind" in item for item in wrong_policy.review_reasons)
    assert any("profile hash" in item for item in wrong_policy.review_reasons)
    assert any("prompt hash" in item for item in wrong_policy.review_reasons)
    assert any("raw response hash" in item for item in wrong_policy.review_reasons)

    case = bundle.case_set.cases[0]
    supporting_with_span = ClaimEvidenceJudgment(
        case_id=case.case_id,
        claim_id=case.claim_id,
        verdict=ClaimEvidenceVerdict.DIRECTLY_SUPPORTED,
        unsupported_claim_spans=(UnsupportedClaimSpan(start=0, end=1, text=case.claim_text[0:1]),),
        reason="Internally inconsistent judgment.",
    )
    wrong_claim = _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED).model_copy(
        update={"claim_id": "claim_crisis"}
    )
    malformed = _receipt(bundle.case_set, (supporting_with_span, wrong_claim))
    malformed_result = _gate(bundle, malformed)
    assert malformed_result.decision is ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    assert supporting_with_span.unsupported_spans == supporting_with_span.unsupported_claim_spans
    assert any("cannot contain" in item for item in malformed_result.review_reasons)
    assert any("judgment names claim" in item for item in malformed_result.review_reasons)

    mismatched_text = ClaimEvidenceJudgment(
        case_id=case.case_id,
        claim_id=case.claim_id,
        verdict=ClaimEvidenceVerdict.PARTIAL,
        unsupported_claim_spans=(UnsupportedClaimSpan(start=0, end=1, text="x"),),
        reason="Bad exact span text.",
    )
    mismatch_receipt = _receipt(
        bundle.case_set,
        (mismatched_text, _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED)),
    )
    assert _gate(bundle, mismatch_receipt).decision is (
        ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    )


@pytest.mark.parametrize(
    ("state", "verdict", "expected"),
    [
        (AnswerClaimState.SUPPORTED, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED, "passed"),
        (AnswerClaimState.SUPPORTED, ClaimEvidenceVerdict.SUPPORTED_INFERENCE, "failed"),
        (AnswerClaimState.QUALIFIED, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED, "passed"),
        (AnswerClaimState.QUALIFIED, ClaimEvidenceVerdict.SUPPORTED_INFERENCE, "failed"),
        (AnswerClaimState.INFERENCE, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED, "passed"),
        (AnswerClaimState.INFERENCE, ClaimEvidenceVerdict.SUPPORTED_INFERENCE, "passed"),
        (AnswerClaimState.INFERENCE, ClaimEvidenceVerdict.PARTIAL, "failed"),
        (AnswerClaimState.INFERENCE, ClaimEvidenceVerdict.UNSUPPORTED, "failed"),
        (AnswerClaimState.INFERENCE, ClaimEvidenceVerdict.CONTRADICTED, "failed"),
    ],
)
def test_claim_state_verdict_mapping(
    tmp_path: Path,
    state: AnswerClaimState,
    verdict: ClaimEvidenceVerdict,
    expected: str,
) -> None:
    packet = _wide_packet(_manifest())
    other_state = (
        AnswerClaimState.SUPPORTED
        if state is not AnswerClaimState.SUPPORTED
        else AnswerClaimState.QUALIFIED
    )
    answer = _answer_with_claim_states(packet, (state, other_state))
    bundle = _bundle(tmp_path, answer=answer)
    judgments = (
        _judgment(bundle.case_set, 0, verdict),
        _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED),
    )

    result = _gate(bundle, _receipt(bundle.case_set, judgments))
    assert result.cases[0].decision.value == expected


def test_failed_prerequisites_skip_semantic_stage(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    rejected_exact = bundle.exact.model_copy(update={"decision": AnswerDecision.REJECTED})
    result = ClaimEvidenceEntailmentGate().evaluate(
        bundle.case_set,
        answer_contract_result=rejected_exact,
        answer_coverage_result=bundle.coverage,
        receipt=_receipt(bundle.case_set),
    )

    assert result.decision is ClaimEvidenceEntailmentDecision.UNCHECKED
    assert result.skipped_reason is not None
    assert result.promotion_eligible is False

    incomplete_coverage = bundle.coverage.model_copy(
        update={"decision": AnswerCoverageDecision.INCOMPLETE}
    )
    assert (
        ClaimEvidenceEntailmentGate()
        .evaluate(
            bundle.case_set,
            answer_contract_result=bundle.exact,
            answer_coverage_result=incomplete_coverage,
        )
        .decision
        is ClaimEvidenceEntailmentDecision.UNCHECKED
    )
    stale_exact = bundle.exact.model_copy(update={"answer_hash": "0" * 64})
    assert (
        ClaimEvidenceEntailmentGate()
        .evaluate(
            bundle.case_set,
            answer_contract_result=stale_exact,
            answer_coverage_result=bundle.coverage,
        )
        .decision
        is ClaimEvidenceEntailmentDecision.UNCHECKED
    )
    stale_coverage = bundle.coverage.model_copy(update={"spec_hash": "0" * 64})
    assert (
        ClaimEvidenceEntailmentGate()
        .evaluate(
            bundle.case_set,
            answer_contract_result=bundle.exact,
            answer_coverage_result=stale_coverage,
        )
        .decision
        is ClaimEvidenceEntailmentDecision.UNCHECKED
    )


def test_unified_facade_final_decisions_and_fail_fast(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    validator = ResearchAnswerValidator()
    common: _ValidationArgs = {
        "task_id": "task_crisis",
        "packet": bundle.packet,
        "coverage_spec": bundle.spec,
        "artifact_root": tmp_path,
        "expected_judge_kind": ClaimEvidenceJudgeKind.MODEL,
        "expected_judge_profile_id": _PROFILE_ID,
        "expected_judge_profile_hash": _PROFILE_HASH,
        "expected_prompt_hash": _PROMPT_HASH,
        "expected_raw_response_hash": _RAW_HASH,
    }
    review = validator.validate(bundle.answer, **common)
    accepted = validator.validate(
        bundle.answer,
        receipt=_receipt(bundle.case_set),
        **common,
    )
    partial_receipt = _receipt(
        bundle.case_set,
        (
            _judgment(bundle.case_set, 0, ClaimEvidenceVerdict.PARTIAL),
            _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED),
        ),
    )
    revision_required = validator.validate(
        bundle.answer,
        receipt=partial_receipt,
        **common,
    )
    unsupported_receipt = _receipt(
        bundle.case_set,
        (
            _judgment(bundle.case_set, 0, ClaimEvidenceVerdict.UNSUPPORTED),
            _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED),
        ),
    )
    rejected_semantic = validator.validate(
        bundle.answer,
        receipt=unsupported_receipt,
        **common,
    )

    bad_quote = bundle.answer.model_dump(mode="json")
    bad_quote["citations"][0]["exact_quote"] = "invented quotation"
    rejected_exact = validator.validate(
        ResearchAnswer.model_validate_json(json.dumps(bad_quote)),
        **common,
    )

    incomplete_answer = _wide_answer(bundle.packet, include_work=False)
    rejected_coverage = validator.validate(incomplete_answer, **common)

    assert review.decision is ResearchAnswerValidationDecision.REVIEW_REQUIRED
    assert accepted.decision is ResearchAnswerValidationDecision.ACCEPTED
    assert accepted.promotion_eligible is True
    assert revision_required.decision is ResearchAnswerValidationDecision.REVISION_REQUIRED
    assert revision_required.promotion_eligible is False
    assert rejected_semantic.decision is ResearchAnswerValidationDecision.REJECTED
    assert rejected_exact.decision is ResearchAnswerValidationDecision.REJECTED
    assert rejected_exact.coverage is None
    assert rejected_exact.case_set is None
    assert rejected_exact.semantic.decision is ClaimEvidenceEntailmentDecision.UNCHECKED
    assert rejected_coverage.decision is ResearchAnswerValidationDecision.REJECTED
    assert rejected_coverage.coverage is not None
    assert rejected_coverage.coverage.decision is AnswerCoverageDecision.INCOMPLETE
    assert rejected_coverage.case_set is None

    payload_accepted = validator.validate_payload(
        bundle.answer.model_dump_json(),
        receipt=_receipt(bundle.case_set),
        **common,
    )
    object_review = validator.validate_payload(bundle.answer, **common)
    invalid_payload = validator.validate_payload("{}", **common)
    alias_review = validator.evaluate(bundle.answer, **common)
    assert payload_accepted.decision is ResearchAnswerValidationDecision.ACCEPTED
    assert object_review.decision is ResearchAnswerValidationDecision.REVIEW_REQUIRED
    assert invalid_payload.decision is ResearchAnswerValidationDecision.REJECTED
    assert invalid_payload.coverage is None
    assert alias_review.decision is ResearchAnswerValidationDecision.REVIEW_REQUIRED
    assert accepted.final_decision is accepted.decision
    assert accepted.exact_result is accepted.exact
    assert accepted.coverage_result is accepted.coverage
    assert accepted.semantic_result is accepted.semantic
    assert accepted.answer_contract_result is accepted.exact
    assert accepted.answer_coverage_result is accepted.coverage
    assert accepted.entailment_result is accepted.semantic

    with pytest.raises(ValidationError, match="promotion-eligible"):
        ResearchAnswerValidationResult(
            decision=ResearchAnswerValidationDecision.ACCEPTED,
            promotion_eligible=False,
            exact=accepted.exact,
            coverage=accepted.coverage,
            case_set=accepted.case_set,
            semantic=accepted.semantic,
        )
    with pytest.raises(ValidationError, match="promotion-eligible"):
        ResearchAnswerValidationResult(
            decision=ResearchAnswerValidationDecision.REVISION_REQUIRED,
            promotion_eligible=True,
            exact=revision_required.exact,
            coverage=revision_required.coverage,
            case_set=revision_required.case_set,
            semantic=revision_required.semantic,
        )
    with pytest.raises(ValidationError, match="three stages"):
        ResearchAnswerValidationResult(
            decision=ResearchAnswerValidationDecision.ACCEPTED,
            promotion_eligible=True,
            exact=rejected_exact.exact,
            coverage=None,
            case_set=None,
            semantic=rejected_exact.semantic,
        )


def test_facade_handles_internal_case_builder_refusal(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    validator = ResearchAnswerValidator()

    class _RefusingBuilder:
        def build(self, *_args: object, **_kwargs: object) -> ClaimEvidenceCaseSet:
            raise ClaimEvidenceContractError("deliberate test refusal")

    validator._case_builder = _RefusingBuilder()  # type: ignore[assignment]
    result = validator.validate(
        bundle.answer,
        task_id="task_crisis",
        packet=bundle.packet,
        coverage_spec=bundle.spec,
        artifact_root=tmp_path,
    )
    assert result.decision is ResearchAnswerValidationDecision.REVIEW_REQUIRED
    assert result.case_set is None
    assert result.semantic.decision is ClaimEvidenceEntailmentDecision.UNCHECKED
