"""Deterministic closure gate for public IdeaTrace artifacts."""

from __future__ import annotations

from dithyramba.answers import (
    AnswerClaimState,
    ClaimEvidenceCaseSet,
    ClaimEvidenceEntailmentDecision,
    ClaimEvidenceEntailmentResult,
    ClaimEvidenceVerdict,
)
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

from .models import (
    IdeaTrace,
    IdeaTraceStep,
    ReasoningClosureDecision,
    ReasoningClosureResult,
    ReasoningOperation,
    ReasoningQualifier,
    ReasoningStepClosure,
)


class ReasoningClosureGate:
    """Check trace topology and exact semantic-receipt bindings without a model call."""

    def evaluate(
        self,
        trace: IdeaTrace,
        *,
        case_set: ClaimEvidenceCaseSet,
        entailment: ClaimEvidenceEntailmentResult,
    ) -> ReasoningClosureResult:
        review_reasons = _binding_findings(trace, case_set, entailment)
        if review_reasons:
            steps = tuple(
                _unclosed_step(item, "trace dependencies are not exactly bound")
                for item in trace.steps
            )
            return ReasoningClosureResult.create(
                trace=trace,
                decision=ReasoningClosureDecision.REVIEW_REQUIRED,
                steps=steps,
                review_reasons=tuple(review_reasons),
            )

        cases = {item.case_id: item for item in case_set.cases}
        outcomes = {item.case_id: item for item in entailment.cases}
        known_evidence = {
            citation.evidence_id for case in case_set.cases for citation in case.citations
        }
        violations: list[str] = []
        pending_review: list[str] = []
        step_results: list[ReasoningStepClosure] = []

        for step in trace.steps:
            case = cases.get(step.case_id)
            outcome = outcomes.get(step.case_id)
            local_violations: list[str] = []
            local_review: list[str] = []
            if case is None:
                local_review.append("claim-evidence case is absent")
            elif case.claim_id != step.claim_id or case.claim_text != step.statement:
                local_review.append("step statement is stale or bound to a different claim")
            if outcome is None:
                local_review.append("semantic case result is absent")
            elif outcome.claim_id != step.claim_id:
                local_review.append("semantic case result names a different claim")
            else:
                if outcome.decision is ClaimEvidenceEntailmentDecision.FAILED:
                    local_violations.append(f"semantic evidence gate failed: {outcome.reason}")
                elif outcome.decision is not ClaimEvidenceEntailmentDecision.PASSED:
                    local_review.append(f"semantic evidence gate is {outcome.decision.value}")

            missing_counterevidence = set(step.counterevidence_ids).difference(known_evidence)
            if missing_counterevidence:
                local_review.append(
                    "counterevidence is outside the exact case set: "
                    + ", ".join(sorted(missing_counterevidence))
                )
            if case is not None and outcome is not None:
                _operation_findings(
                    step.operation,
                    step.qualifier,
                    case.claim_state,
                    outcome.verdict,
                    local_violations,
                )

            violations.extend(f"{step.step_id}: {item}" for item in local_violations)
            pending_review.extend(f"{step.step_id}: {item}" for item in local_review)
            closed = not local_violations and not local_review
            reason = (
                "exact claim and semantic receipt close this public step"
                if closed
                else "; ".join((*local_violations, *local_review))
            )
            step_results.append(
                ReasoningStepClosure(
                    step_id=step.step_id,
                    claim_id=step.claim_id,
                    case_id=step.case_id,
                    verdict=None if outcome is None else outcome.verdict,
                    semantic_decision=(
                        ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
                        if outcome is None
                        else outcome.decision
                    ),
                    closed=closed,
                    reason=reason,
                )
            )

        if entailment.decision is ClaimEvidenceEntailmentDecision.FAILED:
            violations.append("the bound claim-evidence result failed")
        elif entailment.decision is not ClaimEvidenceEntailmentDecision.PASSED:
            pending_review.append(f"the bound claim-evidence result is {entailment.decision.value}")

        if violations:
            decision = ReasoningClosureDecision.FAILED
        elif pending_review:
            decision = ReasoningClosureDecision.REVIEW_REQUIRED
        else:
            decision = ReasoningClosureDecision.PASSED
        return ReasoningClosureResult.create(
            trace=trace,
            decision=decision,
            steps=tuple(step_results),
            violations=tuple(violations),
            review_reasons=tuple(pending_review),
        )


def _binding_findings(
    trace: IdeaTrace,
    case_set: ClaimEvidenceCaseSet,
    entailment: ClaimEvidenceEntailmentResult,
) -> list[str]:
    findings: list[str] = []
    trace_payload = trace.semantic_payload()
    if (
        trace.trace_id != canonical_content_id("idea_trace", trace_payload)
        or trace.trace_hash != canonical_sha256_hex(trace_payload)
        or any(
            step.step_id != canonical_content_id("idea_trace_step", step.semantic_payload())
            for step in trace.steps
        )
    ):
        findings.append("IdeaTrace content identity is invalid or was modified after validation")
    if (
        trace.task_id != case_set.task_id
        or trace.answer_id != case_set.answer_id
        or trace.answer_hash != case_set.answer_hash
        or trace.packet_id != case_set.packet_id
        or trace.packet_hash != case_set.packet_hash
        or trace.case_set_id != case_set.case_set_id
        or trace.case_set_hash != case_set.case_set_hash
    ):
        findings.append("IdeaTrace is stale or bound to a different answer/case set")
    if (
        entailment.case_set_id != case_set.case_set_id
        or entailment.case_set_hash != case_set.case_set_hash
    ):
        findings.append("claim-evidence result is bound to a different case set")
    if entailment.receipt_id != trace.receipt_id or entailment.receipt_hash != trace.receipt_hash:
        findings.append("IdeaTrace is bound to a different semantic judgment receipt")
    return findings


def _operation_findings(
    operation: ReasoningOperation,
    qualifier: ReasoningQualifier,
    claim_state: AnswerClaimState,
    verdict: ClaimEvidenceVerdict | None,
    violations: list[str],
) -> None:
    if operation is ReasoningOperation.EXTRACT:
        if qualifier is ReasoningQualifier.DIRECT and (
            claim_state is not AnswerClaimState.SUPPORTED
            or verdict is not ClaimEvidenceVerdict.DIRECTLY_SUPPORTED
        ):
            violations.append("direct extraction requires a directly supported claim")
        if qualifier is ReasoningQualifier.CONTESTED and (
            claim_state is not AnswerClaimState.QUALIFIED
            or verdict is not ClaimEvidenceVerdict.DIRECTLY_SUPPORTED
        ):
            violations.append("contested extraction requires a qualified supported claim")
        return

    if qualifier is ReasoningQualifier.BOUNDED and (
        claim_state is not AnswerClaimState.INFERENCE
        or verdict
        not in {
            ClaimEvidenceVerdict.DIRECTLY_SUPPORTED,
            ClaimEvidenceVerdict.SUPPORTED_INFERENCE,
        }
    ):
        violations.append("bounded derivation requires an inference claim with support")
    if qualifier is ReasoningQualifier.CONTESTED and (
        claim_state not in {AnswerClaimState.QUALIFIED, AnswerClaimState.INFERENCE}
        or verdict
        not in {
            ClaimEvidenceVerdict.DIRECTLY_SUPPORTED,
            ClaimEvidenceVerdict.SUPPORTED_INFERENCE,
        }
    ):
        violations.append("contested derivation requires a qualified or inference claim")


def _unclosed_step(step: IdeaTraceStep, reason: str) -> ReasoningStepClosure:
    return ReasoningStepClosure(
        step_id=step.step_id,
        claim_id=step.claim_id,
        case_id=step.case_id,
        verdict=None,
        semantic_decision=ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED,
        closed=False,
        reason=reason,
    )
