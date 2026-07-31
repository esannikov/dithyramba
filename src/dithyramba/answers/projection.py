"""Clause-level projection for expressive, evidence-governed research answers.

The strict answer contract proves exact citations and accepted claims.  This
module keeps that evidence spine while allowing the final prose to contain
bounded synthesis, disclosed hypotheses, research questions, and purely
explanatory framing.  It never promotes those exploratory segments to facts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

from .contracts import AnswerClaim, AnswerClaimState, ResearchAnswer
from .entailment import (
    ClaimEvidenceCaseResult,
    ClaimEvidenceCaseSet,
    ClaimEvidenceEntailmentDecision,
    ClaimEvidenceEntailmentResult,
    ClaimEvidenceJudgeKind,
    ClaimEvidenceVerdict,
)

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_PROPOSITION_ID_PATTERN = r"^answer_proposition_[0-9a-f]{32}$"
_PROJECTION_ID_PATTERN = r"^answer_projection_[0-9a-f]{32}$"
_PROJECTION_RECEIPT_ID_PATTERN = r"^answer_projection_receipt_[0-9a-f]{32}$"


class _ProjectionModel(BaseModel):
    # Proposition text is an exact character span and may intentionally include
    # leading whitespace between sentences.  Stripping it would invalidate the
    # source offsets and weaken replay.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PropositionKind(StrEnum):
    """The epistemic role of one exact span in the displayed answer."""

    FACT = "fact"
    SYNTHESIS = "synthesis"
    HYPOTHESIS = "hypothesis"
    QUESTION = "question"
    FRAMING = "framing"


class PropositionVerdict(StrEnum):
    """External semantic assessment of whether a span matches its declared role."""

    ENTAILED = "entailed"
    BOUNDED_SYNTHESIS = "bounded_synthesis"
    DISCLOSED_HYPOTHESIS = "disclosed_hypothesis"
    OPEN_QUESTION = "open_question"
    NON_PROPOSITIONAL = "non_propositional"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class PropositionCoverageDecision(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class HypothesisProbe(_ProjectionModel):
    """A falsifiable research route attached to an exploratory hypothesis."""

    test_question: str = Field(min_length=1, max_length=2_000)
    falsifier: str = Field(min_length=1, max_length=2_000)
    next_evidence: str = Field(min_length=1, max_length=2_000)


class AnswerProposition(_ProjectionModel):
    """One exact, half-open span of the displayed ``short_answer``."""

    proposition_id: str = Field(pattern=_PROPOSITION_ID_PATTERN)
    start: int = Field(ge=0, le=8_000)
    end: int = Field(ge=1, le=8_000)
    text: str = Field(min_length=1, max_length=8_000)
    kind: PropositionKind
    claim_ids: tuple[str, ...] = Field(max_length=16)
    hypothesis_probe: HypothesisProbe | None = None

    @model_validator(mode="after")
    def validate_proposition(self) -> Self:
        if self.end <= self.start:
            raise ValueError("answer proposition must have start < end")
        if self.end - self.start != len(self.text):
            raise ValueError("answer proposition offsets must match its exact text length")
        if len(self.claim_ids) != len(set(self.claim_ids)):
            raise ValueError("answer proposition claim_ids must be unique")

        evidence_bound = {
            PropositionKind.FACT,
            PropositionKind.SYNTHESIS,
            PropositionKind.HYPOTHESIS,
            PropositionKind.QUESTION,
        }
        if self.kind in evidence_bound and not self.claim_ids:
            raise ValueError(f"{self.kind.value} propositions require premise claim_ids")
        if self.kind is PropositionKind.FRAMING and self.claim_ids:
            raise ValueError("framing propositions cannot carry claim_ids")
        if self.kind is PropositionKind.HYPOTHESIS:
            if self.hypothesis_probe is None:
                raise ValueError("hypothesis propositions require a falsifiable probe")
        elif self.hypothesis_probe is not None:
            raise ValueError("hypothesis_probe is reserved for hypothesis propositions")

        if self.proposition_id != canonical_content_id(
            "answer_proposition", self.semantic_payload()
        ):
            raise ValueError("proposition_id does not match the canonical proposition content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "kind": self.kind.value,
            "claim_ids": list(self.claim_ids),
            "hypothesis_probe": (
                None
                if self.hypothesis_probe is None
                else self.hypothesis_probe.model_dump(mode="json")
            ),
        }

    @classmethod
    def create(
        cls,
        *,
        start: int,
        text: str,
        kind: PropositionKind,
        claim_ids: tuple[str, ...] = (),
        hypothesis_probe: HypothesisProbe | None = None,
    ) -> AnswerProposition:
        payload: dict[str, object] = {
            "start": start,
            "end": start + len(text),
            "text": text,
            "kind": kind.value,
            "claim_ids": list(claim_ids),
            "hypothesis_probe": (
                None if hypothesis_probe is None else hypothesis_probe.model_dump(mode="json")
            ),
        }
        return cls(
            proposition_id=canonical_content_id("answer_proposition", payload),
            start=start,
            end=start + len(text),
            text=text,
            kind=kind,
            claim_ids=claim_ids,
            hypothesis_probe=hypothesis_probe,
        )


class AnswerProjection(_ProjectionModel):
    """Content-addressed epistemic overlay for one exact displayed answer."""

    SCHEMA: ClassVar[str] = "dithyramba.answer_projection/1.0"

    schema_id: str
    projection_id: str = Field(pattern=_PROJECTION_ID_PATTERN)
    projection_hash: str = Field(pattern=_HASH_PATTERN)
    answer_id: str = Field(pattern=r"^research_answer_[0-9a-f]{32}$")
    answer_hash: str = Field(pattern=_HASH_PATTERN)
    case_set_id: str = Field(pattern=r"^claim_evidence_case_set_[0-9a-f]{32}$")
    case_set_hash: str = Field(pattern=_HASH_PATTERN)
    entailment_receipt_id: str = Field(pattern=r"^claim_evidence_judgment_receipt_[0-9a-f]{32}$")
    entailment_receipt_hash: str = Field(pattern=_HASH_PATTERN)
    author_kind: ClaimEvidenceJudgeKind
    author_profile_id: str = Field(pattern=_ID_PATTERN)
    author_profile_hash: str = Field(pattern=_HASH_PATTERN)
    instruction_hash: str = Field(pattern=_HASH_PATTERN)
    raw_response_hash: str = Field(pattern=_HASH_PATTERN)
    propositions: tuple[AnswerProposition, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        proposition_ids = tuple(item.proposition_id for item in self.propositions)
        if len(proposition_ids) != len(set(proposition_ids)):
            raise ValueError("proposition_id values must be unique")
        if self.propositions[0].start != 0:
            raise ValueError("the first answer proposition must start at character 0")
        for previous, current in zip(
            self.propositions,
            self.propositions[1:],
            strict=False,
        ):
            if current.start != previous.end:
                raise ValueError("answer propositions must be ordered and contiguous")
        payload = self.semantic_payload()
        if self.projection_id != canonical_content_id("answer_projection", payload):
            raise ValueError("projection_id does not match the canonical projection content")
        if self.projection_hash != canonical_sha256_hex(payload):
            raise ValueError("projection_hash does not match the canonical projection content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "answer_id": self.answer_id,
            "answer_hash": self.answer_hash,
            "case_set_id": self.case_set_id,
            "case_set_hash": self.case_set_hash,
            "entailment_receipt_id": self.entailment_receipt_id,
            "entailment_receipt_hash": self.entailment_receipt_hash,
            "author_kind": self.author_kind.value,
            "author_profile_id": self.author_profile_id,
            "author_profile_hash": self.author_profile_hash,
            "instruction_hash": self.instruction_hash,
            "raw_response_hash": self.raw_response_hash,
            "propositions": [
                {"proposition_id": item.proposition_id, **item.semantic_payload()}
                for item in self.propositions
            ],
        }

    @classmethod
    def create(
        cls,
        *,
        answer: ResearchAnswer,
        case_set: ClaimEvidenceCaseSet,
        entailment: ClaimEvidenceEntailmentResult,
        author_kind: ClaimEvidenceJudgeKind,
        author_profile_id: str,
        author_profile_hash: str,
        instruction_hash: str,
        raw_response_hash: str,
        propositions: tuple[AnswerProposition, ...],
    ) -> AnswerProjection:
        if entailment.receipt_id is None or entailment.receipt_hash is None:
            raise ValueError("answer projection requires a checked entailment receipt")
        payload: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "answer_id": answer.answer_id,
            "answer_hash": answer.answer_hash,
            "case_set_id": case_set.case_set_id,
            "case_set_hash": case_set.case_set_hash,
            "entailment_receipt_id": entailment.receipt_id,
            "entailment_receipt_hash": entailment.receipt_hash,
            "author_kind": author_kind.value,
            "author_profile_id": author_profile_id,
            "author_profile_hash": author_profile_hash,
            "instruction_hash": instruction_hash,
            "raw_response_hash": raw_response_hash,
            "propositions": [
                {"proposition_id": item.proposition_id, **item.semantic_payload()}
                for item in propositions
            ],
        }
        return cls(
            schema_id=cls.SCHEMA,
            projection_id=canonical_content_id("answer_projection", payload),
            projection_hash=canonical_sha256_hex(payload),
            answer_id=answer.answer_id,
            answer_hash=answer.answer_hash,
            case_set_id=case_set.case_set_id,
            case_set_hash=case_set.case_set_hash,
            entailment_receipt_id=entailment.receipt_id,
            entailment_receipt_hash=entailment.receipt_hash,
            author_kind=author_kind,
            author_profile_id=author_profile_id,
            author_profile_hash=author_profile_hash,
            instruction_hash=instruction_hash,
            raw_response_hash=raw_response_hash,
            propositions=propositions,
        )


class PropositionJudgment(_ProjectionModel):
    proposition_id: str = Field(pattern=_PROPOSITION_ID_PATTERN)
    verdict: PropositionVerdict
    reason: str = Field(min_length=1, max_length=4_000)


class AnswerProjectionJudgmentReceipt(_ProjectionModel):
    """External semantic review of the declared role of every answer span."""

    SCHEMA: ClassVar[str] = "dithyramba.answer_projection_receipt/1.0"

    schema_id: str
    receipt_id: str = Field(pattern=_PROJECTION_RECEIPT_ID_PATTERN)
    receipt_hash: str = Field(pattern=_HASH_PATTERN)
    projection_id: str = Field(pattern=_PROJECTION_ID_PATTERN)
    projection_hash: str = Field(pattern=_HASH_PATTERN)
    judge_kind: ClaimEvidenceJudgeKind
    judge_profile_id: str = Field(pattern=_ID_PATTERN)
    judge_profile_hash: str = Field(pattern=_HASH_PATTERN)
    prompt_hash: str = Field(pattern=_HASH_PATTERN)
    raw_response_hash: str = Field(pattern=_HASH_PATTERN)
    judgments: tuple[PropositionJudgment, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        payload = self.semantic_payload()
        if self.receipt_id != canonical_content_id("answer_projection_receipt", payload):
            raise ValueError("receipt_id does not match the canonical projection judgment")
        if self.receipt_hash != canonical_sha256_hex(payload):
            raise ValueError("receipt_hash does not match the canonical projection judgment")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "projection_id": self.projection_id,
            "projection_hash": self.projection_hash,
            "judge_kind": self.judge_kind.value,
            "judge_profile_id": self.judge_profile_id,
            "judge_profile_hash": self.judge_profile_hash,
            "prompt_hash": self.prompt_hash,
            "raw_response_hash": self.raw_response_hash,
            "judgments": [item.model_dump(mode="json") for item in self.judgments],
        }

    @classmethod
    def create(
        cls,
        *,
        projection: AnswerProjection,
        judge_kind: ClaimEvidenceJudgeKind,
        judge_profile_id: str,
        judge_profile_hash: str,
        prompt_hash: str,
        raw_response_hash: str,
        judgments: tuple[PropositionJudgment, ...],
    ) -> AnswerProjectionJudgmentReceipt:
        payload: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "projection_id": projection.projection_id,
            "projection_hash": projection.projection_hash,
            "judge_kind": judge_kind.value,
            "judge_profile_id": judge_profile_id,
            "judge_profile_hash": judge_profile_hash,
            "prompt_hash": prompt_hash,
            "raw_response_hash": raw_response_hash,
            "judgments": [item.model_dump(mode="json") for item in judgments],
        }
        return cls(
            schema_id=cls.SCHEMA,
            receipt_id=canonical_content_id("answer_projection_receipt", payload),
            receipt_hash=canonical_sha256_hex(payload),
            projection_id=projection.projection_id,
            projection_hash=projection.projection_hash,
            judge_kind=judge_kind,
            judge_profile_id=judge_profile_id,
            judge_profile_hash=judge_profile_hash,
            prompt_hash=prompt_hash,
            raw_response_hash=raw_response_hash,
            judgments=judgments,
        )


class PropositionCoverage(_ProjectionModel):
    proposition_id: str = Field(pattern=_PROPOSITION_ID_PATTERN)
    kind: PropositionKind
    verdict: PropositionVerdict | None
    claim_ids: tuple[str, ...] = Field(max_length=16)
    closed: bool
    reason: str = Field(min_length=1, max_length=4_000)


class PropositionCoverageResult(_ProjectionModel):
    """Display-safety result; a pass never promotes exploratory text to memory."""

    decision: PropositionCoverageDecision
    projection_id: str = Field(pattern=_PROJECTION_ID_PATTERN)
    projection_hash: str = Field(pattern=_HASH_PATTERN)
    receipt_id: str | None = Field(default=None, pattern=_PROJECTION_RECEIPT_ID_PATTERN)
    receipt_hash: str | None = Field(default=None, pattern=_HASH_PATTERN)
    propositions: tuple[PropositionCoverage, ...] = Field(min_length=1, max_length=128)
    factual_propositions: int = Field(ge=0, le=128)
    exploratory_propositions: int = Field(ge=0, le=128)
    violations: tuple[str, ...]
    review_reasons: tuple[str, ...]
    display_eligible: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.display_eligible != (self.decision is PropositionCoverageDecision.PASSED):
            raise ValueError("only a passed proposition projection is display-eligible")
        if self.decision is PropositionCoverageDecision.PASSED and (
            self.violations
            or self.review_reasons
            or not all(item.closed for item in self.propositions)
        ):
            raise ValueError("passed proposition coverage requires every span to close")
        return self


class PropositionCoverageGate:
    """Validate a rich answer span-by-span while preserving exact claim routes."""

    def evaluate(
        self,
        answer: ResearchAnswer,
        *,
        case_set: ClaimEvidenceCaseSet,
        entailment: ClaimEvidenceEntailmentResult,
        projection: AnswerProjection,
        receipt: AnswerProjectionJudgmentReceipt | None = None,
        expected_judge_kind: ClaimEvidenceJudgeKind | None = None,
        expected_judge_profile_id: str | None = None,
        expected_judge_profile_hash: str | None = None,
        expected_prompt_hash: str | None = None,
        expected_raw_response_hash: str | None = None,
    ) -> PropositionCoverageResult:
        binding_findings = _projection_binding_findings(
            answer,
            case_set=case_set,
            entailment=entailment,
            projection=projection,
        )
        if binding_findings:
            return _projection_result(
                projection,
                receipt=receipt,
                decision=PropositionCoverageDecision.REVIEW_REQUIRED,
                propositions=tuple(
                    _unclosed_proposition(item, "projection bindings are stale or incomplete")
                    for item in projection.propositions
                ),
                review_reasons=tuple(binding_findings),
            )

        structural_violations = _projection_text_findings(answer, projection)
        receipt_findings = _projection_receipt_findings(
            projection,
            receipt,
            expected_judge_kind=expected_judge_kind,
            expected_judge_profile_id=expected_judge_profile_id,
            expected_judge_profile_hash=expected_judge_profile_hash,
            expected_prompt_hash=expected_prompt_hash,
            expected_raw_response_hash=expected_raw_response_hash,
        )
        if structural_violations:
            return _projection_result(
                projection,
                receipt=receipt,
                decision=PropositionCoverageDecision.FAILED,
                propositions=tuple(
                    _unclosed_proposition(item, "projection does not match the displayed answer")
                    for item in projection.propositions
                ),
                violations=tuple(structural_violations),
            )
        if receipt_findings:
            return _projection_result(
                projection,
                receipt=receipt,
                decision=PropositionCoverageDecision.REVIEW_REQUIRED,
                propositions=tuple(
                    _unclosed_proposition(item, "semantic role receipt is missing or stale")
                    for item in projection.propositions
                ),
                review_reasons=tuple(receipt_findings),
            )

        assert receipt is not None
        judgments = {item.proposition_id: item for item in receipt.judgments}
        claims = {item.claim_id: item for item in answer.claims}
        outcomes = {item.claim_id: item for item in entailment.cases}
        violations: list[str] = []
        review_reasons: list[str] = []
        results: list[PropositionCoverage] = []

        for proposition in projection.propositions:
            judgment = judgments[proposition.proposition_id]
            local_violations: list[str] = []
            local_review: list[str] = []
            _claim_route_findings(
                proposition,
                claims=claims,
                outcomes=outcomes,
                violations=local_violations,
                review_reasons=local_review,
            )
            _verdict_findings(
                proposition,
                judgment.verdict,
                violations=local_violations,
                review_reasons=local_review,
            )
            violations.extend(f"{proposition.proposition_id}: {item}" for item in local_violations)
            review_reasons.extend(f"{proposition.proposition_id}: {item}" for item in local_review)
            closed = not local_violations and not local_review
            results.append(
                PropositionCoverage(
                    proposition_id=proposition.proposition_id,
                    kind=proposition.kind,
                    verdict=judgment.verdict,
                    claim_ids=proposition.claim_ids,
                    closed=closed,
                    reason=(
                        "declared role and exact premise routes are closed"
                        if closed
                        else "; ".join((*local_violations, *local_review))
                    ),
                )
            )

        if violations:
            decision = PropositionCoverageDecision.FAILED
        elif review_reasons:
            decision = PropositionCoverageDecision.REVIEW_REQUIRED
        else:
            decision = PropositionCoverageDecision.PASSED
        return _projection_result(
            projection,
            receipt=receipt,
            decision=decision,
            propositions=tuple(results),
            violations=tuple(violations),
            review_reasons=tuple(review_reasons),
        )


def _projection_binding_findings(
    answer: ResearchAnswer,
    *,
    case_set: ClaimEvidenceCaseSet,
    entailment: ClaimEvidenceEntailmentResult,
    projection: AnswerProjection,
) -> list[str]:
    findings: list[str] = []
    projection_payload = projection.semantic_payload()
    if (
        projection.projection_id != canonical_content_id("answer_projection", projection_payload)
        or projection.projection_hash != canonical_sha256_hex(projection_payload)
        or any(
            item.proposition_id
            != canonical_content_id("answer_proposition", item.semantic_payload())
            for item in projection.propositions
        )
    ):
        findings.append("answer projection content identity is invalid")
    if (
        projection.answer_id != answer.answer_id
        or projection.answer_hash != answer.answer_hash
        or case_set.answer_id != answer.answer_id
        or case_set.answer_hash != answer.answer_hash
        or projection.case_set_id != case_set.case_set_id
        or projection.case_set_hash != case_set.case_set_hash
    ):
        findings.append("answer projection is bound to a different answer or case set")
    if (
        entailment.decision is not ClaimEvidenceEntailmentDecision.PASSED
        or entailment.case_set_id != case_set.case_set_id
        or entailment.case_set_hash != case_set.case_set_hash
        or entailment.receipt_id != projection.entailment_receipt_id
        or entailment.receipt_hash != projection.entailment_receipt_hash
    ):
        findings.append("answer projection requires the exact passed claim-evidence receipt")
    return findings


def _projection_text_findings(
    answer: ResearchAnswer,
    projection: AnswerProjection,
) -> list[str]:
    findings: list[str] = []
    if projection.propositions[-1].end != len(answer.short_answer):
        findings.append("answer propositions do not cover the complete short_answer")
    for proposition in projection.propositions:
        if answer.short_answer[proposition.start : proposition.end] != proposition.text:
            findings.append(f"{proposition.proposition_id}: span is not exact in short_answer")
    return findings


def _projection_receipt_findings(
    projection: AnswerProjection,
    receipt: AnswerProjectionJudgmentReceipt | None,
    *,
    expected_judge_kind: ClaimEvidenceJudgeKind | None,
    expected_judge_profile_id: str | None,
    expected_judge_profile_hash: str | None,
    expected_prompt_hash: str | None,
    expected_raw_response_hash: str | None,
) -> list[str]:
    if receipt is None:
        return ["answer projection judgment receipt is missing"]
    findings: list[str] = []
    receipt_payload = receipt.semantic_payload()
    if receipt.schema_id != receipt.SCHEMA:
        findings.append("answer projection receipt schema is invalid")
    if receipt.receipt_id != canonical_content_id("answer_projection_receipt", receipt_payload):
        findings.append("answer projection receipt ID is invalid")
    if receipt.receipt_hash != canonical_sha256_hex(receipt_payload):
        findings.append("answer projection receipt hash is invalid")
    if (
        receipt.projection_id != projection.projection_id
        or receipt.projection_hash != projection.projection_hash
    ):
        findings.append("answer projection receipt is bound to another projection")
    expected_ids = tuple(item.proposition_id for item in projection.propositions)
    actual_ids = tuple(item.proposition_id for item in receipt.judgments)
    if len(actual_ids) != len(set(actual_ids)):
        findings.append("answer projection receipt has duplicate judgments")
    if set(actual_ids) != set(expected_ids):
        findings.append("answer projection receipt does not close every proposition")
    if expected_judge_kind is not None and receipt.judge_kind is not expected_judge_kind:
        findings.append("answer projection judge kind does not match policy")
    if (
        expected_judge_profile_id is not None
        and receipt.judge_profile_id != expected_judge_profile_id
    ):
        findings.append("answer projection judge profile ID does not match policy")
    if (
        expected_judge_profile_hash is not None
        and receipt.judge_profile_hash != expected_judge_profile_hash
    ):
        findings.append("answer projection judge profile hash does not match policy")
    if expected_prompt_hash is not None and receipt.prompt_hash != expected_prompt_hash:
        findings.append("answer projection prompt hash does not match policy")
    if (
        expected_raw_response_hash is not None
        and receipt.raw_response_hash != expected_raw_response_hash
    ):
        findings.append("answer projection raw response hash does not match policy")
    return findings


def _claim_route_findings(
    proposition: AnswerProposition,
    *,
    claims: dict[str, AnswerClaim],
    outcomes: dict[str, ClaimEvidenceCaseResult],
    violations: list[str],
    review_reasons: list[str],
) -> None:
    if proposition.kind is PropositionKind.FRAMING:
        return
    for claim_id in proposition.claim_ids:
        claim = claims.get(claim_id)
        outcome = outcomes.get(claim_id)
        if claim is None:
            violations.append(f"premise claim {claim_id} is absent from the answer")
            continue
        if outcome is None:
            review_reasons.append(f"premise claim {claim_id} has no semantic result")
            continue
        decision = outcome.decision
        verdict = outcome.verdict
        if decision is not ClaimEvidenceEntailmentDecision.PASSED:
            review_reasons.append(f"premise claim {claim_id} is not semantically closed")
            continue
        accepted = {
            ClaimEvidenceVerdict.DIRECTLY_SUPPORTED,
            ClaimEvidenceVerdict.SUPPORTED_INFERENCE,
        }
        if verdict not in accepted:
            violations.append(f"premise claim {claim_id} is not supported")
        if proposition.kind is PropositionKind.FACT:
            state = claim.status
            if state not in {AnswerClaimState.SUPPORTED, AnswerClaimState.QUALIFIED}:
                violations.append(f"fact premise {claim_id} is not a factual claim")
            if verdict is not ClaimEvidenceVerdict.DIRECTLY_SUPPORTED:
                violations.append(f"fact premise {claim_id} lacks direct support")


def _verdict_findings(
    proposition: AnswerProposition,
    verdict: PropositionVerdict,
    *,
    violations: list[str],
    review_reasons: list[str],
) -> None:
    if verdict is PropositionVerdict.UNCERTAIN:
        review_reasons.append("semantic role judgment is uncertain")
        return
    if verdict is PropositionVerdict.UNSUPPORTED:
        violations.append("the displayed span overclaims its routed premises")
        return
    expected = {
        PropositionKind.FACT: PropositionVerdict.ENTAILED,
        PropositionKind.SYNTHESIS: PropositionVerdict.BOUNDED_SYNTHESIS,
        PropositionKind.HYPOTHESIS: PropositionVerdict.DISCLOSED_HYPOTHESIS,
        PropositionKind.QUESTION: PropositionVerdict.OPEN_QUESTION,
        PropositionKind.FRAMING: PropositionVerdict.NON_PROPOSITIONAL,
    }[proposition.kind]
    if verdict is not expected:
        violations.append(
            f"declared {proposition.kind.value} role conflicts with {verdict.value} judgment"
        )


def _unclosed_proposition(
    proposition: AnswerProposition,
    reason: str,
) -> PropositionCoverage:
    return PropositionCoverage(
        proposition_id=proposition.proposition_id,
        kind=proposition.kind,
        verdict=None,
        claim_ids=proposition.claim_ids,
        closed=False,
        reason=reason,
    )


def _projection_result(
    projection: AnswerProjection,
    *,
    receipt: AnswerProjectionJudgmentReceipt | None,
    decision: PropositionCoverageDecision,
    propositions: tuple[PropositionCoverage, ...],
    violations: tuple[str, ...] = (),
    review_reasons: tuple[str, ...] = (),
) -> PropositionCoverageResult:
    factual = sum(item.kind is PropositionKind.FACT for item in propositions)
    exploratory = sum(
        item.kind in {PropositionKind.SYNTHESIS, PropositionKind.HYPOTHESIS}
        for item in propositions
    )
    return PropositionCoverageResult(
        decision=decision,
        projection_id=projection.projection_id,
        projection_hash=projection.projection_hash,
        receipt_id=None if receipt is None else receipt.receipt_id,
        receipt_hash=None if receipt is None else receipt.receipt_hash,
        propositions=propositions,
        factual_propositions=factual,
        exploratory_propositions=exploratory,
        violations=violations,
        review_reasons=review_reasons,
        display_eligible=decision is PropositionCoverageDecision.PASSED,
    )
