"""Unified, provider-free validation for packet-grounded research answers."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from dithyramba.answers.contracts import (
    AnswerContractGate,
    AnswerContractResult,
    AnswerDecision,
    ResearchAnswer,
)
from dithyramba.answers.coverage import (
    AnswerCoverageDecision,
    AnswerCoverageGate,
    AnswerCoverageResult,
    AnswerCoverageSpec,
)
from dithyramba.answers.entailment import (
    ClaimEvidenceCaseSet,
    ClaimEvidenceCaseSetBuilder,
    ClaimEvidenceContractError,
    ClaimEvidenceEntailmentDecision,
    ClaimEvidenceEntailmentGate,
    ClaimEvidenceEntailmentResult,
    ClaimEvidenceJudgeKind,
    ClaimEvidenceJudgmentReceipt,
    ClaimEvidenceVerdict,
)
from dithyramba.atlas import CompactMemoryPacket


class ResearchAnswerValidationDecision(StrEnum):
    ACCEPTED = "accepted"
    REVISION_REQUIRED = "revision_required"
    REJECTED = "rejected"
    REVIEW_REQUIRED = "review_required"


class ResearchAnswerValidationResult(BaseModel):
    """One strict result containing the exact, coverage, and semantic stages."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    decision: ResearchAnswerValidationDecision
    promotion_eligible: bool
    exact: AnswerContractResult
    coverage: AnswerCoverageResult | None
    case_set: ClaimEvidenceCaseSet | None
    semantic: ClaimEvidenceEntailmentResult

    @model_validator(mode="after")
    def validate_final_decision(self) -> Self:
        if self.promotion_eligible != (self.decision is ResearchAnswerValidationDecision.ACCEPTED):
            raise ValueError("only an accepted unified result is promotion-eligible")
        if self.decision is ResearchAnswerValidationDecision.ACCEPTED and (
            self.exact.decision is not AnswerDecision.ACCEPTED
            or self.coverage is None
            or self.coverage.decision is not AnswerCoverageDecision.COMPLETE
            or self.semantic.decision is not ClaimEvidenceEntailmentDecision.PASSED
        ):
            raise ValueError("accepted final decision requires all three stages to pass")
        return self

    @property
    def final_decision(self) -> ResearchAnswerValidationDecision:
        return self.decision

    @property
    def exact_result(self) -> AnswerContractResult:
        return self.exact

    @property
    def coverage_result(self) -> AnswerCoverageResult | None:
        return self.coverage

    @property
    def semantic_result(self) -> ClaimEvidenceEntailmentResult:
        return self.semantic

    @property
    def answer_contract_result(self) -> AnswerContractResult:
        return self.exact

    @property
    def answer_coverage_result(self) -> AnswerCoverageResult | None:
        return self.coverage

    @property
    def entailment_result(self) -> ClaimEvidenceEntailmentResult:
        return self.semantic


class ResearchAnswerValidator:
    """Run all answer gates without making a provider or model call.

    The first two stages remain deterministic mechanical checks.  The semantic
    stage only validates an externally supplied judgment receipt and applies
    the frozen verdict policy; a model verdict is not treated as historical
    truth.
    """

    def __init__(self) -> None:
        self._exact_gate = AnswerContractGate()
        self._coverage_gate = AnswerCoverageGate()
        self._case_builder = ClaimEvidenceCaseSetBuilder()
        self._semantic_gate = ClaimEvidenceEntailmentGate()

    def validate(
        self,
        answer: ResearchAnswer,
        *,
        task_id: str,
        packet: CompactMemoryPacket,
        coverage_spec: AnswerCoverageSpec,
        artifact_root: str | Path,
        receipt: ClaimEvidenceJudgmentReceipt | None = None,
        expected_judge_kind: ClaimEvidenceJudgeKind | None = None,
        expected_judge_profile_id: str | None = None,
        expected_judge_profile_hash: str | None = None,
        expected_prompt_hash: str | None = None,
        expected_raw_response_hash: str | None = None,
    ) -> ResearchAnswerValidationResult:
        exact = self._exact_gate.evaluate(
            answer,
            task_id=task_id,
            packet=packet,
            artifact_root=artifact_root,
        )
        if exact.decision is not AnswerDecision.ACCEPTED:
            semantic = self._semantic_gate.unchecked(
                case_set=None,
                reason="exact answer contract was rejected; later validation stages were skipped",
            )
            return _unified_result(
                exact=exact,
                coverage=None,
                case_set=None,
                semantic=semantic,
            )
        coverage = self._coverage_gate.evaluate(answer, spec=coverage_spec, packet=packet)
        if coverage.decision is not AnswerCoverageDecision.COMPLETE:
            semantic = self._semantic_gate.unchecked(
                case_set=None,
                reason="required-facet coverage was incomplete; semantic review was skipped",
            )
            return _unified_result(
                exact=exact,
                coverage=coverage,
                case_set=None,
                semantic=semantic,
            )
        case_set: ClaimEvidenceCaseSet | None
        try:
            case_set = self._case_builder.build(
                answer,
                task_id=task_id,
                packet=packet,
                coverage_spec=coverage_spec,
            )
        except ClaimEvidenceContractError as exc:
            case_set = None
            semantic = self._semantic_gate.unchecked(
                case_set=None,
                reason=f"claim-evidence case set could not be built: {exc}",
            )
        else:
            semantic = self._semantic_gate.evaluate(
                case_set,
                answer_contract_result=exact,
                answer_coverage_result=coverage,
                receipt=receipt,
                expected_judge_kind=expected_judge_kind,
                expected_judge_profile_id=expected_judge_profile_id,
                expected_judge_profile_hash=expected_judge_profile_hash,
                expected_prompt_hash=expected_prompt_hash,
                expected_raw_response_hash=expected_raw_response_hash,
            )
        return _unified_result(
            exact=exact,
            coverage=coverage,
            case_set=case_set,
            semantic=semantic,
        )

    def validate_payload(
        self,
        payload: object,
        *,
        task_id: str,
        packet: CompactMemoryPacket,
        coverage_spec: AnswerCoverageSpec,
        artifact_root: str | Path,
        receipt: ClaimEvidenceJudgmentReceipt | None = None,
        expected_judge_kind: ClaimEvidenceJudgeKind | None = None,
        expected_judge_profile_id: str | None = None,
        expected_judge_profile_hash: str | None = None,
        expected_prompt_hash: str | None = None,
        expected_raw_response_hash: str | None = None,
    ) -> ResearchAnswerValidationResult:
        """Validate JSON-like input while preserving the exact gate's schema result."""

        try:
            if isinstance(payload, (str, bytes, bytearray)):
                answer = ResearchAnswer.model_validate_json(payload)
            else:
                answer = ResearchAnswer.model_validate(payload)
        except ValidationError:
            exact = self._exact_gate.evaluate_payload(
                payload,
                task_id=task_id,
                packet=packet,
                artifact_root=artifact_root,
            )
            semantic = self._semantic_gate.unchecked(
                case_set=None,
                reason="answer schema was invalid; later validation stages were skipped",
            )
            return _unified_result(
                exact=exact,
                coverage=None,
                case_set=None,
                semantic=semantic,
            )
        return self.validate(
            answer,
            task_id=task_id,
            packet=packet,
            coverage_spec=coverage_spec,
            artifact_root=artifact_root,
            receipt=receipt,
            expected_judge_kind=expected_judge_kind,
            expected_judge_profile_id=expected_judge_profile_id,
            expected_judge_profile_hash=expected_judge_profile_hash,
            expected_prompt_hash=expected_prompt_hash,
            expected_raw_response_hash=expected_raw_response_hash,
        )

    evaluate = validate
    evaluate_payload = validate_payload


def _unified_result(
    *,
    exact: AnswerContractResult,
    coverage: AnswerCoverageResult | None,
    case_set: ClaimEvidenceCaseSet | None,
    semantic: ClaimEvidenceEntailmentResult,
) -> ResearchAnswerValidationResult:
    if (
        exact.decision is not AnswerDecision.ACCEPTED
        or coverage is None
        or coverage.decision is not AnswerCoverageDecision.COMPLETE
    ):
        decision = ResearchAnswerValidationDecision.REJECTED
    elif semantic.decision is ClaimEvidenceEntailmentDecision.PASSED:
        decision = ResearchAnswerValidationDecision.ACCEPTED
    elif semantic.decision is ClaimEvidenceEntailmentDecision.FAILED:
        failed_cases = tuple(
            item
            for item in semantic.cases
            if item.decision is ClaimEvidenceEntailmentDecision.FAILED
        )
        if failed_cases and all(
            item.verdict is ClaimEvidenceVerdict.PARTIAL for item in failed_cases
        ):
            decision = ResearchAnswerValidationDecision.REVISION_REQUIRED
        else:
            decision = ResearchAnswerValidationDecision.REJECTED
    else:
        decision = ResearchAnswerValidationDecision.REVIEW_REQUIRED
    return ResearchAnswerValidationResult(
        decision=decision,
        promotion_eligible=decision is ResearchAnswerValidationDecision.ACCEPTED,
        exact=exact,
        coverage=coverage,
        case_set=case_set,
        semantic=semantic,
    )
