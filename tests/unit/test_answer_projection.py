"""Clause-level answer freedom without losing exact evidence routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dithyramba.answers import (
    AnswerClaimState,
    AnswerProjection,
    AnswerProjectionJudgmentReceipt,
    AnswerProposition,
    ClaimEvidenceCaseSet,
    ClaimEvidenceEntailmentDecision,
    ClaimEvidenceEntailmentResult,
    ClaimEvidenceJudgeKind,
    ClaimEvidenceVerdict,
    HypothesisProbe,
    PropositionCoverage,
    PropositionCoverageDecision,
    PropositionCoverageGate,
    PropositionCoverageResult,
    PropositionJudgment,
    PropositionKind,
    PropositionVerdict,
    ResearchAnswer,
)
from dithyramba.contracts import sha256_hex

from .test_claim_evidence_entailment import (
    _answer_with_claim_states,
    _bundle,
    _gate,
    _judgment,
    _receipt,
)
from .test_compact_memory_and_answer_contract import _manifest, _wide_packet

_AUTHOR_HASH = sha256_hex(b"answer projection author v1")
_AUTHOR_INSTRUCTION_HASH = sha256_hex(b"classify every exact displayed span")
_AUTHOR_RAW_HASH = sha256_hex(b"projection response")
_JUDGE_HASH = sha256_hex(b"answer projection judge v1")
_JUDGE_PROMPT_HASH = sha256_hex(b"check declared proposition roles")
_JUDGE_RAW_HASH = sha256_hex(b"projection judgment response")


def _projection_bundle(
    tmp_path: Path,
) -> tuple[
    ResearchAnswer,
    ClaimEvidenceCaseSet,
    ClaimEvidenceEntailmentResult,
    AnswerProjection,
    AnswerProjectionJudgmentReceipt,
]:
    packet = _wide_packet(_manifest())
    base = _answer_with_claim_states(
        packet,
        (AnswerClaimState.SUPPORTED, AnswerClaimState.INFERENCE),
    )
    fact = "Криза підтверджена."
    synthesis = " Разом із пізнішою роботою це утворює картину відновленої практики."
    hypothesis = " Можливо, повернення до роботи допомагало відновлювати внутрішню структуру."
    question = " Яке незалежне джерело може перевірити цей зв'язок?"
    framing = " Це напрям дослідження, а не завершена біографічна формула."  # noqa: RUF001
    answer_payload = base.model_dump(mode="json")
    answer_payload["short_answer"] = fact + synthesis + hypothesis + question + framing
    answer = ResearchAnswer.model_validate_json(json.dumps(answer_payload))
    bundle = _bundle(tmp_path, answer=answer)
    receipt = _receipt(
        bundle.case_set,
        (
            _judgment(bundle.case_set, 0, ClaimEvidenceVerdict.DIRECTLY_SUPPORTED),
            _judgment(bundle.case_set, 1, ClaimEvidenceVerdict.SUPPORTED_INFERENCE),
        ),
    )
    entailment = _gate(bundle, receipt)
    claim_ids = tuple(item.claim_id for item in answer.claims)
    segments: list[AnswerProposition] = []
    cursor = 0
    for text, kind, premises, probe in (
        (fact, PropositionKind.FACT, (claim_ids[0],), None),
        (synthesis, PropositionKind.SYNTHESIS, claim_ids, None),
        (
            hypothesis,
            PropositionKind.HYPOTHESIS,
            claim_ids,
            HypothesisProbe(
                test_question="Чи корелювало повернення до роботи зі стабілізацією стану?",
                falsifier="Незалежні записи показують стабілізацію до повернення до роботи.",
                next_evidence="Датовані листи, медичні записи й робоча хронологія.",
            ),
        ),
        (question, PropositionKind.QUESTION, claim_ids, None),
        (framing, PropositionKind.FRAMING, (), None),
    ):
        segment = AnswerProposition.create(
            start=cursor,
            text=text,
            kind=kind,
            claim_ids=premises,
            hypothesis_probe=probe,
        )
        segments.append(segment)
        cursor = segment.end
    projection = AnswerProjection.create(
        answer=answer,
        case_set=bundle.case_set,
        entailment=entailment,
        author_kind=ClaimEvidenceJudgeKind.MODEL,
        author_profile_id="projection_author_v1",
        author_profile_hash=_AUTHOR_HASH,
        instruction_hash=_AUTHOR_INSTRUCTION_HASH,
        raw_response_hash=_AUTHOR_RAW_HASH,
        propositions=tuple(segments),
    )
    verdicts = (
        PropositionVerdict.ENTAILED,
        PropositionVerdict.BOUNDED_SYNTHESIS,
        PropositionVerdict.DISCLOSED_HYPOTHESIS,
        PropositionVerdict.OPEN_QUESTION,
        PropositionVerdict.NON_PROPOSITIONAL,
    )
    projection_receipt = AnswerProjectionJudgmentReceipt.create(
        projection=projection,
        judge_kind=ClaimEvidenceJudgeKind.MODEL,
        judge_profile_id="projection_judge_v1",
        judge_profile_hash=_JUDGE_HASH,
        prompt_hash=_JUDGE_PROMPT_HASH,
        raw_response_hash=_JUDGE_RAW_HASH,
        judgments=tuple(
            PropositionJudgment(
                proposition_id=segment.proposition_id,
                verdict=verdict,
                reason=f"The exact span is a valid {verdict.value} segment.",
            )
            for segment, verdict in zip(segments, verdicts, strict=True)
        ),
    )
    return answer, bundle.case_set, entailment, projection, projection_receipt


def _evaluate(
    tmp_path: Path,
    *,
    include_receipt: bool = True,
) -> PropositionCoverageResult:
    answer, case_set, entailment, projection, projection_receipt = _projection_bundle(tmp_path)
    selected_receipt = projection_receipt if include_receipt else None
    return PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=projection,
        receipt=selected_receipt,
        expected_judge_kind=ClaimEvidenceJudgeKind.MODEL,
        expected_judge_profile_id="projection_judge_v1",
        expected_judge_profile_hash=_JUDGE_HASH,
        expected_prompt_hash=_JUDGE_PROMPT_HASH,
        expected_raw_response_hash=_JUDGE_RAW_HASH,
    )


def test_rich_answer_passes_without_promoting_hypothesis_to_fact(tmp_path: Path) -> None:
    answer, case_set, entailment, projection, receipt = _projection_bundle(tmp_path)
    result = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=projection,
        receipt=receipt,
    )

    assert entailment.decision is ClaimEvidenceEntailmentDecision.PASSED
    assert projection.propositions[-3].kind is PropositionKind.HYPOTHESIS
    assert projection.propositions[-3].hypothesis_probe is not None
    assert result.decision is PropositionCoverageDecision.PASSED
    assert result.display_eligible is True
    assert result.factual_propositions == 1
    assert result.exploratory_propositions == 2
    assert "".join(item.text for item in projection.propositions) == answer.short_answer


def test_unlabeled_overclaim_fails_only_its_exact_span(tmp_path: Path) -> None:
    answer, case_set, entailment, projection, receipt = _projection_bundle(tmp_path)
    judgments = list(receipt.judgments)
    target = projection.propositions[1]
    judgments[1] = PropositionJudgment(
        proposition_id=target.proposition_id,
        verdict=PropositionVerdict.UNSUPPORTED,
        reason="The synthesis says more than its two premise claims establish.",
    )
    failed_receipt = AnswerProjectionJudgmentReceipt.create(
        projection=projection,
        judge_kind=receipt.judge_kind,
        judge_profile_id=receipt.judge_profile_id,
        judge_profile_hash=receipt.judge_profile_hash,
        prompt_hash=receipt.prompt_hash,
        raw_response_hash=receipt.raw_response_hash,
        judgments=tuple(judgments),
    )

    result = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=projection,
        receipt=failed_receipt,
    )

    assert result.decision is PropositionCoverageDecision.FAILED
    assert result.display_eligible is False
    assert result.propositions[0].closed is True
    assert result.propositions[1].closed is False
    assert result.propositions[2].closed is True
    assert len(result.violations) == 1
    assert "overclaims" in result.violations[0]


def test_missing_judgment_requires_review_instead_of_flat_rejection(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, include_receipt=False)

    assert result.decision is PropositionCoverageDecision.REVIEW_REQUIRED
    assert result.display_eligible is False
    assert result.violations == ()
    assert result.review_reasons == ("answer projection judgment receipt is missing",)


def test_hypothesis_requires_falsifier_and_fact_requires_direct_route(
    tmp_path: Path,
) -> None:
    answer, case_set, entailment, projection, receipt = _projection_bundle(tmp_path)
    hypothesis = projection.propositions[2]
    payload = hypothesis.model_dump(mode="json")
    payload["hypothesis_probe"] = None
    with pytest.raises(ValidationError, match="falsifiable probe"):
        AnswerProposition.model_validate_json(json.dumps(payload))

    second_claim = answer.claims[1]
    fact_like = AnswerProposition.create(
        start=projection.propositions[0].start,
        text=projection.propositions[0].text,
        kind=PropositionKind.FACT,
        claim_ids=(second_claim.claim_id,),
    )
    altered = AnswerProjection.create(
        answer=answer,
        case_set=case_set,
        entailment=entailment,
        author_kind=projection.author_kind,
        author_profile_id=projection.author_profile_id,
        author_profile_hash=projection.author_profile_hash,
        instruction_hash=projection.instruction_hash,
        raw_response_hash=projection.raw_response_hash,
        propositions=(fact_like, *projection.propositions[1:]),
    )
    altered_receipt = AnswerProjectionJudgmentReceipt.create(
        projection=altered,
        judge_kind=receipt.judge_kind,
        judge_profile_id=receipt.judge_profile_id,
        judge_profile_hash=receipt.judge_profile_hash,
        prompt_hash=receipt.prompt_hash,
        raw_response_hash=receipt.raw_response_hash,
        judgments=tuple(
            PropositionJudgment(
                proposition_id=item.proposition_id,
                verdict=(
                    PropositionVerdict.ENTAILED if index == 0 else receipt.judgments[index].verdict
                ),
                reason="Rebound semantic role.",
            )
            for index, item in enumerate(altered.propositions)
        ),
    )

    result = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=altered,
        receipt=altered_receipt,
    )

    assert result.decision is PropositionCoverageDecision.FAILED
    assert any("lacks direct support" in item for item in result.violations)


def test_projection_rejects_gaps_tampering_and_stale_answer(tmp_path: Path) -> None:
    answer, case_set, entailment, projection, receipt = _projection_bundle(tmp_path)
    segments = list(projection.propositions)
    shifted = segments[1].model_dump(mode="json")
    shifted["start"] += 1
    shifted["end"] += 1
    shifted["proposition_id"] = "answer_proposition_" + "0" * 32
    with pytest.raises(ValidationError, match="proposition_id"):
        AnswerProposition.model_validate_json(json.dumps(shifted))

    stale_answer = answer.model_copy(update={"short_answer": answer.short_answer + " Нове."})
    result = PropositionCoverageGate().evaluate(
        stale_answer,
        case_set=case_set,
        entailment=entailment,
        projection=projection,
        receipt=receipt,
    )

    assert result.decision is PropositionCoverageDecision.REVIEW_REQUIRED
    assert any("different answer" in item for item in result.review_reasons)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"start": 2, "end": 1}, "start < end"),
        ({"end": 3}, "offsets"),
        ({"claim_ids": ["claim_supported", "claim_supported"]}, "unique"),
        ({"claim_ids": []}, "require premise"),
        ({"proposition_id": "answer_proposition_" + "0" * 32}, "canonical"),
    ),
)
def test_proposition_shape_rejects_ambiguous_or_tampered_spans(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    _, _, _, projection, _ = _projection_bundle(tmp_path)
    payload = projection.propositions[0].model_dump(mode="json")
    payload.update(change)

    with pytest.raises(ValidationError, match=message):
        AnswerProposition.model_validate_json(json.dumps(payload))


def test_declared_roles_reject_wrong_binding_shapes(tmp_path: Path) -> None:
    _, _, _, projection, _ = _projection_bundle(tmp_path)
    fact = projection.propositions[0]
    hypothesis = projection.propositions[2]
    framing = projection.propositions[-1]

    with pytest.raises(ValidationError, match="framing propositions"):
        AnswerProposition.create(
            start=framing.start,
            text=framing.text,
            kind=PropositionKind.FRAMING,
            claim_ids=fact.claim_ids,
        )
    with pytest.raises(ValidationError, match="reserved for hypothesis"):
        AnswerProposition.create(
            start=fact.start,
            text=fact.text,
            kind=PropositionKind.FACT,
            claim_ids=fact.claim_ids,
            hypothesis_probe=hypothesis.hypothesis_probe,
        )


def test_projection_requires_exact_contiguous_content_and_receipt(tmp_path: Path) -> None:
    answer, case_set, entailment, projection, _ = _projection_bundle(tmp_path)
    first = projection.propositions[0]
    with pytest.raises(ValidationError, match="character 0"):
        AnswerProjection.create(
            answer=answer,
            case_set=case_set,
            entailment=entailment,
            author_kind=projection.author_kind,
            author_profile_id=projection.author_profile_id,
            author_profile_hash=projection.author_profile_hash,
            instruction_hash=projection.instruction_hash,
            raw_response_hash=projection.raw_response_hash,
            propositions=(
                AnswerProposition.create(
                    start=1,
                    text=first.text,
                    kind=first.kind,
                    claim_ids=first.claim_ids,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="unique"):
        AnswerProjection.create(
            answer=answer,
            case_set=case_set,
            entailment=entailment,
            author_kind=projection.author_kind,
            author_profile_id=projection.author_profile_id,
            author_profile_hash=projection.author_profile_hash,
            instruction_hash=projection.instruction_hash,
            raw_response_hash=projection.raw_response_hash,
            propositions=(first, first),
        )
    unchecked = entailment.model_copy(update={"receipt_id": None, "receipt_hash": None})
    with pytest.raises(ValueError, match="checked entailment receipt"):
        AnswerProjection.create(
            answer=answer,
            case_set=case_set,
            entailment=unchecked,
            author_kind=projection.author_kind,
            author_profile_id=projection.author_profile_id,
            author_profile_hash=projection.author_profile_hash,
            instruction_hash=projection.instruction_hash,
            raw_response_hash=projection.raw_response_hash,
            propositions=projection.propositions,
        )


def test_receipt_policy_drift_and_uncertainty_require_review(tmp_path: Path) -> None:
    answer, case_set, entailment, projection, receipt = _projection_bundle(tmp_path)
    result = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=projection,
        receipt=receipt,
        expected_judge_kind=ClaimEvidenceJudgeKind.HUMAN,
        expected_judge_profile_id="different_profile",
        expected_judge_profile_hash="0" * 64,
        expected_prompt_hash="1" * 64,
        expected_raw_response_hash="2" * 64,
    )
    assert result.decision is PropositionCoverageDecision.REVIEW_REQUIRED
    assert len(result.review_reasons) == 5

    judgments = list(receipt.judgments)
    judgments[0] = PropositionJudgment(
        proposition_id=judgments[0].proposition_id,
        verdict=PropositionVerdict.UNCERTAIN,
        reason="The role needs a human decision.",
    )
    uncertain_receipt = AnswerProjectionJudgmentReceipt.create(
        projection=projection,
        judge_kind=receipt.judge_kind,
        judge_profile_id=receipt.judge_profile_id,
        judge_profile_hash=receipt.judge_profile_hash,
        prompt_hash=receipt.prompt_hash,
        raw_response_hash=receipt.raw_response_hash,
        judgments=tuple(judgments),
    )
    uncertain = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=projection,
        receipt=uncertain_receipt,
    )
    assert uncertain.decision is PropositionCoverageDecision.REVIEW_REQUIRED
    assert any("uncertain" in item for item in uncertain.review_reasons)


def test_coverage_result_cannot_claim_a_false_pass(tmp_path: Path) -> None:
    result = _evaluate(tmp_path)
    payload = result.model_dump(mode="json")
    payload["display_eligible"] = False
    with pytest.raises(ValidationError, match="display-eligible"):
        PropositionCoverageResult.model_validate_json(json.dumps(payload))

    open_span = PropositionCoverage(
        proposition_id=result.propositions[0].proposition_id,
        kind=result.propositions[0].kind,
        verdict=result.propositions[0].verdict,
        claim_ids=result.propositions[0].claim_ids,
        closed=False,
        reason="The span is still open.",
    )
    payload = result.model_dump(mode="json")
    payload["propositions"][0] = open_span.model_dump(mode="json")
    with pytest.raises(ValidationError, match="every span"):
        PropositionCoverageResult.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_id", "dithyramba.answer_projection/9.9", "schema_id"),
        ("projection_id", "answer_projection_" + "0" * 32, "projection_id"),
        ("projection_hash", "0" * 64, "projection_hash"),
    ),
)
def test_projection_identity_validation_fails_closed(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    _, _, _, projection, _ = _projection_bundle(tmp_path)
    payload = projection.model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        AnswerProjection.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_id", "dithyramba.answer_projection_receipt/9.9", "schema_id"),
        ("receipt_id", "answer_projection_receipt_" + "0" * 32, "receipt_id"),
        ("receipt_hash", "0" * 64, "receipt_hash"),
    ),
)
def test_projection_receipt_identity_validation_fails_closed(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    _, _, _, _, receipt = _projection_bundle(tmp_path)
    payload = receipt.model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        AnswerProjectionJudgmentReceipt.model_validate_json(json.dumps(payload))


def test_gate_rejects_incomplete_or_inexact_answer_partition(tmp_path: Path) -> None:
    answer, case_set, entailment, projection, receipt = _projection_bundle(tmp_path)
    first_only = AnswerProjection.create(
        answer=answer,
        case_set=case_set,
        entailment=entailment,
        author_kind=projection.author_kind,
        author_profile_id=projection.author_profile_id,
        author_profile_hash=projection.author_profile_hash,
        instruction_hash=projection.instruction_hash,
        raw_response_hash=projection.raw_response_hash,
        propositions=(projection.propositions[0],),
    )
    first_receipt = AnswerProjectionJudgmentReceipt.create(
        projection=first_only,
        judge_kind=receipt.judge_kind,
        judge_profile_id=receipt.judge_profile_id,
        judge_profile_hash=receipt.judge_profile_hash,
        prompt_hash=receipt.prompt_hash,
        raw_response_hash=receipt.raw_response_hash,
        judgments=(receipt.judgments[0],),
    )
    incomplete = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=first_only,
        receipt=first_receipt,
    )
    assert incomplete.decision is PropositionCoverageDecision.FAILED
    assert any("complete short_answer" in item for item in incomplete.violations)

    original = projection.propositions[0]
    changed_text = "X" + original.text[1:]
    inexact_first = AnswerProposition.create(
        start=original.start,
        text=changed_text,
        kind=original.kind,
        claim_ids=original.claim_ids,
    )
    inexact_projection = AnswerProjection.create(
        answer=answer,
        case_set=case_set,
        entailment=entailment,
        author_kind=projection.author_kind,
        author_profile_id=projection.author_profile_id,
        author_profile_hash=projection.author_profile_hash,
        instruction_hash=projection.instruction_hash,
        raw_response_hash=projection.raw_response_hash,
        propositions=(inexact_first, *projection.propositions[1:]),
    )
    inexact = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=inexact_projection,
        receipt=None,
    )
    assert inexact.decision is PropositionCoverageDecision.FAILED
    assert any("span is not exact" in item for item in inexact.violations)


def test_gate_detects_stale_projection_and_entailment_bindings(tmp_path: Path) -> None:
    answer, case_set, entailment, projection, receipt = _projection_bundle(tmp_path)
    corrupt_projection = projection.model_copy(update={"projection_hash": "0" * 64})
    corrupt = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=corrupt_projection,
        receipt=receipt,
    )
    assert any("content identity" in item for item in corrupt.review_reasons)

    failed_entailment = entailment.model_copy(
        update={"decision": ClaimEvidenceEntailmentDecision.FAILED}
    )
    stale = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=failed_entailment,
        projection=projection,
        receipt=receipt,
    )
    assert any("exact passed" in item for item in stale.review_reasons)


def test_gate_rejects_missing_premise_and_wrong_semantic_role(tmp_path: Path) -> None:
    answer, case_set, entailment, projection, receipt = _projection_bundle(tmp_path)
    synthesis = projection.propositions[1]
    missing = AnswerProposition.create(
        start=synthesis.start,
        text=synthesis.text,
        kind=synthesis.kind,
        claim_ids=("missing_claim",),
    )
    missing_projection = AnswerProjection.create(
        answer=answer,
        case_set=case_set,
        entailment=entailment,
        author_kind=projection.author_kind,
        author_profile_id=projection.author_profile_id,
        author_profile_hash=projection.author_profile_hash,
        instruction_hash=projection.instruction_hash,
        raw_response_hash=projection.raw_response_hash,
        propositions=(projection.propositions[0], missing, *projection.propositions[2:]),
    )
    missing_receipt = AnswerProjectionJudgmentReceipt.create(
        projection=missing_projection,
        judge_kind=receipt.judge_kind,
        judge_profile_id=receipt.judge_profile_id,
        judge_profile_hash=receipt.judge_profile_hash,
        prompt_hash=receipt.prompt_hash,
        raw_response_hash=receipt.raw_response_hash,
        judgments=tuple(
            PropositionJudgment(
                proposition_id=item.proposition_id,
                verdict=receipt.judgments[index].verdict,
                reason="Role reviewed.",
            )
            for index, item in enumerate(missing_projection.propositions)
        ),
    )
    result = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=missing_projection,
        receipt=missing_receipt,
    )
    assert any("absent from the answer" in item for item in result.violations)

    judgments = list(receipt.judgments)
    judgments[1] = PropositionJudgment(
        proposition_id=synthesis.proposition_id,
        verdict=PropositionVerdict.ENTAILED,
        reason="Incorrectly declared as a direct fact.",
    )
    wrong_role_receipt = AnswerProjectionJudgmentReceipt.create(
        projection=projection,
        judge_kind=receipt.judge_kind,
        judge_profile_id=receipt.judge_profile_id,
        judge_profile_hash=receipt.judge_profile_hash,
        prompt_hash=receipt.prompt_hash,
        raw_response_hash=receipt.raw_response_hash,
        judgments=tuple(judgments),
    )
    wrong_role = PropositionCoverageGate().evaluate(
        answer,
        case_set=case_set,
        entailment=entailment,
        projection=projection,
        receipt=wrong_role_receipt,
    )
    assert any("role conflicts" in item for item in wrong_role.violations)
