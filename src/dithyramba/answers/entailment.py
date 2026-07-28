"""Receipt-bound semantic review for packet-grounded answer claims.

The case-set builder is deterministic and provider-free: it exposes only the
claim and the exact quotations already admitted by the mechanical answer
contract.  A judgment receipt is an external semantic-review artifact, not a
statement of historical truth.  The gate only validates artifact bindings and
applies the frozen claim-state/verdict policy; it never calls a model and never
uses lexical diagnostics to decide acceptance.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dithyramba.atlas import CompactMemoryPacket
from dithyramba.atlas.models import EvidenceRole, SourceKind, VoiceKind
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

from .contracts import (
    AnswerClaimState,
    AnswerContractResult,
    AnswerDecision,
    ResearchAnswer,
)
from .coverage import (
    AnswerCoverageDecision,
    AnswerCoverageResult,
    AnswerCoverageSpec,
)

_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_TASK_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{1,95}$"
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_CASE_ID_PATTERN = r"^claim_evidence_case_[0-9a-f]{32}$"
_CASE_SET_ID_PATTERN = r"^claim_evidence_case_set_[0-9a-f]{32}$"
_RECEIPT_ID_PATTERN = r"^claim_evidence_judgment_receipt_[0-9a-f]{32}$"
_NUMERIC_LITERAL_PATTERN = re.compile(
    r"(?<!\w)[+-]?[0-9]+(?:[.,:/-][0-9]+)*(?:[%‰])?(?!\w)",
    re.UNICODE,
)


class ClaimEvidenceContractError(ValueError):
    """A claim/evidence artifact cannot be built from the supplied bindings."""


class _EntailmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class ClaimEvidenceJudgeKind(StrEnum):
    HUMAN = "human"
    MODEL = "model"


class ClaimEvidenceVerdict(StrEnum):
    DIRECTLY_SUPPORTED = "directly_supported"
    SUPPORTED_INFERENCE = "supported_inference"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class ClaimEvidenceEntailmentDecision(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"
    UNCHECKED = "unchecked"


class ClaimEvidenceCitation(_EntailmentModel):
    """The deliberately narrow evidence display for one cited exact quotation."""

    evidence_id: str = Field(pattern=_ID_PATTERN)
    source_id: str = Field(pattern=_ID_PATTERN)
    source_title: str = Field(min_length=1, max_length=500)
    source_creator: str = Field(min_length=1, max_length=300)
    source_kind: SourceKind
    voice_kind: VoiceKind
    asserting_voice: str = Field(min_length=1, max_length=300)
    evidence_role: EvidenceRole
    limitation: str | None = Field(default=None, max_length=2_000)
    exact_quote: str = Field(min_length=1, max_length=1_000)


class ClaimEvidenceCase(_EntailmentModel):
    """One claim joined only to the citations that the answer assigned to it."""

    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    claim_id: str = Field(pattern=_ID_PATTERN)
    claim_text: str = Field(min_length=1, max_length=4_000)
    claim_state: AnswerClaimState
    citations: tuple[ClaimEvidenceCitation, ...] = Field(min_length=1, max_length=24)
    unquoted_numeric_literals: tuple[str, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def validate_case_identity(self) -> Self:
        evidence_ids = tuple(item.evidence_id for item in self.citations)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("claim-evidence citations must have unique evidence IDs")
        if len(self.unquoted_numeric_literals) != len(set(self.unquoted_numeric_literals)):
            raise ValueError("unquoted numeric literals must be unique")
        if self.case_id != canonical_content_id("claim_evidence_case", self.semantic_payload()):
            raise ValueError("case_id does not match the canonical case content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "claim_state": self.claim_state.value,
            "citations": [item.model_dump(mode="json") for item in self.citations],
            "unquoted_numeric_literals": list(self.unquoted_numeric_literals),
        }

    @classmethod
    def create(
        cls,
        *,
        claim_id: str,
        claim_text: str,
        claim_state: AnswerClaimState,
        citations: tuple[ClaimEvidenceCitation, ...],
        unquoted_numeric_literals: tuple[str, ...],
    ) -> ClaimEvidenceCase:
        payload: dict[str, object] = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "claim_state": claim_state.value,
            "citations": [item.model_dump(mode="json") for item in citations],
            "unquoted_numeric_literals": list(unquoted_numeric_literals),
        }
        return cls(
            case_id=canonical_content_id("claim_evidence_case", payload),
            claim_id=claim_id,
            claim_text=claim_text,
            claim_state=claim_state,
            citations=citations,
            unquoted_numeric_literals=unquoted_numeric_literals,
        )


class ClaimEvidenceCaseSet(_EntailmentModel):
    """Content-addressed, least-context semantic-review input for one answer."""

    SCHEMA: ClassVar[str] = "dithyramba.claim_evidence_case_set/1.0"

    schema_id: str
    case_set_id: str = Field(pattern=_CASE_SET_ID_PATTERN)
    case_set_hash: str = Field(pattern=_HASH_PATTERN)
    task_id: str = Field(pattern=_TASK_ID_PATTERN)
    answer_id: str = Field(pattern=r"^research_answer_[0-9a-f]{32}$")
    answer_hash: str = Field(pattern=_HASH_PATTERN)
    packet_id: str = Field(pattern=r"^memory_packet_[0-9a-f]{32}$")
    packet_hash: str = Field(pattern=_HASH_PATTERN)
    coverage_spec_id: str = Field(pattern=r"^answer_coverage_spec_[0-9a-f]{32}$")
    coverage_spec_hash: str = Field(pattern=_HASH_PATTERN)
    cases: tuple[ClaimEvidenceCase, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_case_set(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        case_ids = tuple(item.case_id for item in self.cases)
        claim_ids = tuple(item.claim_id for item in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be unique")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("one semantic-review case is required per unique claim")
        payload = self.semantic_payload()
        if self.case_set_id != canonical_content_id("claim_evidence_case_set", payload):
            raise ValueError("case_set_id does not match the canonical case-set content")
        if self.case_set_hash != canonical_sha256_hex(payload):
            raise ValueError("case_set_hash does not match the canonical case-set content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "task_id": self.task_id,
            "answer_id": self.answer_id,
            "answer_hash": self.answer_hash,
            "packet_id": self.packet_id,
            "packet_hash": self.packet_hash,
            "coverage_spec_id": self.coverage_spec_id,
            "coverage_spec_hash": self.coverage_spec_hash,
            "cases": [{"case_id": item.case_id, **item.semantic_payload()} for item in self.cases],
        }


class ClaimEvidenceCaseSetBuilder:
    """Build deterministic review cases without a provider or wider Atlas context."""

    def build(
        self,
        answer: ResearchAnswer,
        *,
        packet: CompactMemoryPacket,
        coverage_spec: AnswerCoverageSpec,
        task_id: str | None = None,
    ) -> ClaimEvidenceCaseSet:
        bound_task_id = answer.task_id if task_id is None else task_id
        self._validate_bindings(
            answer,
            task_id=bound_task_id,
            packet=packet,
            coverage_spec=coverage_spec,
        )

        citation_by_evidence = {item.evidence_id: item for item in answer.citations}
        evidence_by_id = {item.evidence_id: item for item in packet.evidence}
        source_by_id = {item.source_id: item for item in packet.sources}
        cases: list[ClaimEvidenceCase] = []
        for claim in answer.claims:
            displays: list[ClaimEvidenceCitation] = []
            for evidence_id in claim.evidence_ids:
                citation = citation_by_evidence.get(evidence_id)
                evidence = evidence_by_id.get(evidence_id)
                if citation is None or evidence is None:
                    raise ClaimEvidenceContractError(
                        f"claim {claim.claim_id} is not joined to packet evidence {evidence_id}"
                    )
                if citation.source_id != evidence.source_id:
                    raise ClaimEvidenceContractError(
                        f"citation {evidence_id} does not match its packet source"
                    )
                source = source_by_id.get(citation.source_id)
                if source is None:
                    raise ClaimEvidenceContractError(
                        f"citation {evidence_id} references a source outside the packet"
                    )
                displays.append(
                    ClaimEvidenceCitation(
                        evidence_id=evidence_id,
                        source_id=source.source_id,
                        source_title=source.title,
                        source_creator=source.creator,
                        source_kind=source.source_kind,
                        voice_kind=source.voice_kind,
                        asserting_voice=evidence.asserting_voice,
                        evidence_role=evidence.role,
                        limitation=evidence.limitation,
                        exact_quote=citation.exact_quote,
                    )
                )
            cases.append(
                ClaimEvidenceCase.create(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    claim_state=claim.status,
                    citations=tuple(displays),
                    unquoted_numeric_literals=_unquoted_numeric_literals(
                        claim.text,
                        tuple(item.exact_quote for item in displays),
                    ),
                )
            )

        semantic_payload: dict[str, object] = {
            "schema_id": ClaimEvidenceCaseSet.SCHEMA,
            "task_id": bound_task_id,
            "answer_id": answer.answer_id,
            "answer_hash": answer.answer_hash,
            "packet_id": packet.packet_id,
            "packet_hash": packet.packet_hash,
            "coverage_spec_id": coverage_spec.spec_id,
            "coverage_spec_hash": coverage_spec.spec_hash,
            "cases": [{"case_id": item.case_id, **item.semantic_payload()} for item in cases],
        }
        return ClaimEvidenceCaseSet(
            schema_id=ClaimEvidenceCaseSet.SCHEMA,
            case_set_id=canonical_content_id("claim_evidence_case_set", semantic_payload),
            case_set_hash=canonical_sha256_hex(semantic_payload),
            task_id=bound_task_id,
            answer_id=answer.answer_id,
            answer_hash=answer.answer_hash,
            packet_id=packet.packet_id,
            packet_hash=packet.packet_hash,
            coverage_spec_id=coverage_spec.spec_id,
            coverage_spec_hash=coverage_spec.spec_hash,
            cases=tuple(cases),
        )

    @staticmethod
    def _validate_bindings(
        answer: ResearchAnswer,
        *,
        task_id: str,
        packet: CompactMemoryPacket,
        coverage_spec: AnswerCoverageSpec,
    ) -> None:
        if answer.task_id != task_id or coverage_spec.task_id != task_id:
            raise ClaimEvidenceContractError("answer and coverage specification must match task_id")
        expected_packet = (packet.packet_id, packet.packet_hash)
        if (
            answer.route_receipt.packet_id,
            answer.route_receipt.packet_hash,
        ) != expected_packet or (
            coverage_spec.packet_id,
            coverage_spec.packet_hash,
        ) != expected_packet:
            raise ClaimEvidenceContractError(
                "answer, packet, and coverage specification must be exactly bound"
            )


class UnsupportedClaimSpan(_EntailmentModel):
    """A half-open span in the exact claim text that the judgment does not support."""

    start: int = Field(ge=0, le=4_000)
    end: int = Field(ge=1, le=4_000)
    text: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError("unsupported claim span must have start < end")
        return self


class ClaimEvidenceJudgment(_EntailmentModel):
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    claim_id: str = Field(pattern=_ID_PATTERN)
    verdict: ClaimEvidenceVerdict
    unsupported_claim_spans: tuple[UnsupportedClaimSpan, ...] = Field(max_length=64)
    reason: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def validate_span_order(self) -> Self:
        positions = tuple((item.start, item.end) for item in self.unsupported_claim_spans)
        if positions != tuple(sorted(positions)):
            raise ValueError("unsupported claim spans must be in claim order")
        for previous, current in zip(
            self.unsupported_claim_spans,
            self.unsupported_claim_spans[1:],
            strict=False,
        ):
            if current.start < previous.end:
                raise ValueError("unsupported claim spans must not overlap")
        return self

    @property
    def unsupported_spans(self) -> tuple[UnsupportedClaimSpan, ...]:
        """Compatibility shorthand for callers that already use ``unsupported_spans``."""

        return self.unsupported_claim_spans


class ClaimEvidenceJudgmentReceipt(_EntailmentModel):
    """Content-addressed external review output bound to one exact case set.

    A model-authored receipt records one judge's assessment; it does not turn
    that assessment into historical truth.  Admission happens only after the
    deterministic gate validates the binding, judgment closure, and policy.
    """

    SCHEMA: ClassVar[str] = "dithyramba.claim_evidence_judgment_receipt/1.0"

    schema_id: str
    receipt_id: str = Field(pattern=_RECEIPT_ID_PATTERN)
    receipt_hash: str = Field(pattern=_HASH_PATTERN)
    case_set_id: str = Field(pattern=_CASE_SET_ID_PATTERN)
    case_set_hash: str = Field(pattern=_HASH_PATTERN)
    judge_kind: ClaimEvidenceJudgeKind
    judge_profile_id: str = Field(pattern=_ID_PATTERN)
    judge_profile_hash: str = Field(pattern=_HASH_PATTERN)
    prompt_hash: str = Field(pattern=_HASH_PATTERN)
    raw_response_hash: str = Field(pattern=_HASH_PATTERN)
    judgments: tuple[ClaimEvidenceJudgment, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_receipt_identity(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        payload = self.semantic_payload()
        if self.receipt_id != canonical_content_id("claim_evidence_judgment_receipt", payload):
            raise ValueError("receipt_id does not match the canonical judgment content")
        if self.receipt_hash != canonical_sha256_hex(payload):
            raise ValueError("receipt_hash does not match the canonical judgment content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "case_set_id": self.case_set_id,
            "case_set_hash": self.case_set_hash,
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
        case_set: ClaimEvidenceCaseSet,
        judge_kind: ClaimEvidenceJudgeKind,
        judge_profile_id: str,
        judge_profile_hash: str,
        prompt_hash: str,
        raw_response_hash: str,
        judgments: tuple[ClaimEvidenceJudgment, ...],
    ) -> ClaimEvidenceJudgmentReceipt:
        payload: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "case_set_id": case_set.case_set_id,
            "case_set_hash": case_set.case_set_hash,
            "judge_kind": judge_kind.value,
            "judge_profile_id": judge_profile_id,
            "judge_profile_hash": judge_profile_hash,
            "prompt_hash": prompt_hash,
            "raw_response_hash": raw_response_hash,
            "judgments": [item.model_dump(mode="json") for item in judgments],
        }
        return cls(
            schema_id=cls.SCHEMA,
            receipt_id=canonical_content_id("claim_evidence_judgment_receipt", payload),
            receipt_hash=canonical_sha256_hex(payload),
            case_set_id=case_set.case_set_id,
            case_set_hash=case_set.case_set_hash,
            judge_kind=judge_kind,
            judge_profile_id=judge_profile_id,
            judge_profile_hash=judge_profile_hash,
            prompt_hash=prompt_hash,
            raw_response_hash=raw_response_hash,
            judgments=judgments,
        )

    build = create


class ClaimEvidenceCaseResult(_EntailmentModel):
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    claim_id: str = Field(pattern=_ID_PATTERN)
    claim_state: AnswerClaimState
    verdict: ClaimEvidenceVerdict | None
    decision: ClaimEvidenceEntailmentDecision
    unsupported_claim_spans: tuple[UnsupportedClaimSpan, ...] = Field(max_length=64)
    reason: str = Field(min_length=1, max_length=4_000)


class ClaimEvidenceEntailmentResult(_EntailmentModel):
    decision: ClaimEvidenceEntailmentDecision
    case_set_id: str | None = Field(default=None, pattern=_CASE_SET_ID_PATTERN)
    case_set_hash: str | None = Field(default=None, pattern=_HASH_PATTERN)
    receipt_id: str | None = Field(default=None, pattern=_RECEIPT_ID_PATTERN)
    receipt_hash: str | None = Field(default=None, pattern=_HASH_PATTERN)
    cases: tuple[ClaimEvidenceCaseResult, ...] = Field(max_length=64)
    violations: tuple[str, ...]
    review_reasons: tuple[str, ...]
    skipped_reason: str | None = Field(default=None, min_length=1, max_length=2_000)
    promotion_eligible: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.promotion_eligible != (self.decision is ClaimEvidenceEntailmentDecision.PASSED):
            raise ValueError("only a passed semantic gate is promotion-eligible")
        if self.decision is ClaimEvidenceEntailmentDecision.UNCHECKED:
            if self.skipped_reason is None:
                raise ValueError("unchecked semantic results require a skipped_reason")
        elif self.skipped_reason is not None:
            raise ValueError("checked semantic results cannot have a skipped_reason")
        return self


class ClaimEvidenceEntailmentGate:
    """Validate a receipt binding and apply verdict policy without judging prose."""

    def evaluate(
        self,
        case_set: ClaimEvidenceCaseSet,
        *,
        answer_contract_result: AnswerContractResult,
        answer_coverage_result: AnswerCoverageResult,
        receipt: ClaimEvidenceJudgmentReceipt | None = None,
        expected_judge_kind: ClaimEvidenceJudgeKind | None = None,
        expected_judge_profile_id: str | None = None,
        expected_judge_profile_hash: str | None = None,
        expected_prompt_hash: str | None = None,
        expected_raw_response_hash: str | None = None,
    ) -> ClaimEvidenceEntailmentResult:
        prerequisite_error = _prerequisite_error(
            case_set,
            answer_contract_result=answer_contract_result,
            answer_coverage_result=answer_coverage_result,
        )
        if prerequisite_error is not None:
            return self.unchecked(case_set=case_set, reason=prerequisite_error)

        if receipt is None:
            return _semantic_result(
                decision=ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED,
                case_set=case_set,
                receipt=None,
                cases=(),
                review_reasons=("semantic judgment receipt is missing",),
            )

        integrity_errors = _receipt_integrity_errors(
            case_set,
            receipt,
            expected_judge_kind=expected_judge_kind,
            expected_judge_profile_id=expected_judge_profile_id,
            expected_judge_profile_hash=expected_judge_profile_hash,
            expected_prompt_hash=expected_prompt_hash,
            expected_raw_response_hash=expected_raw_response_hash,
        )
        case_results, closure_errors = _evaluate_judgments(case_set, receipt)
        integrity_errors.extend(closure_errors)
        if integrity_errors:
            return _semantic_result(
                decision=ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED,
                case_set=case_set,
                receipt=receipt,
                cases=case_results,
                review_reasons=tuple(integrity_errors),
            )

        violations = tuple(
            f"{item.claim_id}: {item.reason}"
            for item in case_results
            if item.decision is ClaimEvidenceEntailmentDecision.FAILED
        )
        uncertain = tuple(
            f"{item.claim_id}: {item.reason}"
            for item in case_results
            if item.decision is ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
        )
        if violations:
            decision = ClaimEvidenceEntailmentDecision.FAILED
        elif uncertain:
            decision = ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
        else:
            decision = ClaimEvidenceEntailmentDecision.PASSED
        return _semantic_result(
            decision=decision,
            case_set=case_set,
            receipt=receipt,
            cases=case_results,
            violations=violations,
            review_reasons=uncertain,
        )

    @staticmethod
    def unchecked(
        *,
        case_set: ClaimEvidenceCaseSet | None,
        reason: str,
    ) -> ClaimEvidenceEntailmentResult:
        return ClaimEvidenceEntailmentResult(
            decision=ClaimEvidenceEntailmentDecision.UNCHECKED,
            case_set_id=None if case_set is None else case_set.case_set_id,
            case_set_hash=None if case_set is None else case_set.case_set_hash,
            receipt_id=None,
            receipt_hash=None,
            cases=(),
            violations=(),
            review_reasons=(),
            skipped_reason=reason,
            promotion_eligible=False,
        )


def _unquoted_numeric_literals(claim_text: str, quotes: tuple[str, ...]) -> tuple[str, ...]:
    quoted_literals = {
        match.group(0) for quote in quotes for match in _NUMERIC_LITERAL_PATTERN.finditer(quote)
    }
    seen: set[str] = set()
    result: list[str] = []
    for match in _NUMERIC_LITERAL_PATTERN.finditer(claim_text):
        literal = match.group(0)
        if literal not in quoted_literals and literal not in seen:
            seen.add(literal)
            result.append(literal)
    return tuple(result)


def _prerequisite_error(
    case_set: ClaimEvidenceCaseSet,
    *,
    answer_contract_result: AnswerContractResult,
    answer_coverage_result: AnswerCoverageResult,
) -> str | None:
    if answer_contract_result.decision is not AnswerDecision.ACCEPTED:
        return "exact answer contract was not accepted; semantic review was skipped"
    if answer_coverage_result.decision is not AnswerCoverageDecision.COMPLETE:
        return "required-facet coverage was not complete; semantic review was skipped"
    if (
        answer_contract_result.answer_id != case_set.answer_id
        or answer_contract_result.answer_hash != case_set.answer_hash
    ):
        return "exact answer-contract result is not bound to the case-set answer"
    if (
        answer_coverage_result.answer_id != case_set.answer_id
        or answer_coverage_result.answer_hash != case_set.answer_hash
        or answer_coverage_result.spec_id != case_set.coverage_spec_id
        or answer_coverage_result.spec_hash != case_set.coverage_spec_hash
    ):
        return "coverage result is not bound to the case set"
    return None


def _receipt_integrity_errors(
    case_set: ClaimEvidenceCaseSet,
    receipt: ClaimEvidenceJudgmentReceipt,
    *,
    expected_judge_kind: ClaimEvidenceJudgeKind | None,
    expected_judge_profile_id: str | None,
    expected_judge_profile_hash: str | None,
    expected_prompt_hash: str | None,
    expected_raw_response_hash: str | None,
) -> list[str]:
    errors: list[str] = []
    if receipt.schema_id != ClaimEvidenceJudgmentReceipt.SCHEMA:
        errors.append("judgment receipt schema is unsupported")
    payload = receipt.semantic_payload()
    if receipt.receipt_id != canonical_content_id("claim_evidence_judgment_receipt", payload):
        errors.append("judgment receipt ID does not match its content")
    if receipt.receipt_hash != canonical_sha256_hex(payload):
        errors.append("judgment receipt hash does not match its content")
    if (
        receipt.case_set_id != case_set.case_set_id
        or receipt.case_set_hash != case_set.case_set_hash
    ):
        errors.append("judgment receipt is stale or bound to a different case set")
    if expected_judge_kind is not None and receipt.judge_kind is not expected_judge_kind:
        errors.append("judgment receipt uses the wrong judge kind")
    if (
        expected_judge_profile_id is not None
        and receipt.judge_profile_id != expected_judge_profile_id
    ):
        errors.append("judgment receipt uses the wrong judge profile ID")
    if (
        expected_judge_profile_hash is not None
        and receipt.judge_profile_hash != expected_judge_profile_hash
    ):
        errors.append("judgment receipt uses the wrong judge profile hash")
    if expected_prompt_hash is not None and receipt.prompt_hash != expected_prompt_hash:
        errors.append("judgment receipt uses the wrong prompt hash")
    if (
        expected_raw_response_hash is not None
        and receipt.raw_response_hash != expected_raw_response_hash
    ):
        errors.append("judgment receipt uses the wrong raw response hash")
    return errors


def _evaluate_judgments(
    case_set: ClaimEvidenceCaseSet,
    receipt: ClaimEvidenceJudgmentReceipt,
) -> tuple[tuple[ClaimEvidenceCaseResult, ...], list[str]]:
    errors: list[str] = []
    expected_case_ids = {item.case_id for item in case_set.cases}
    expected_claim_ids = {item.claim_id for item in case_set.cases}
    judgment_case_ids = tuple(item.case_id for item in receipt.judgments)
    judgment_claim_ids = tuple(item.claim_id for item in receipt.judgments)
    duplicate_cases = _duplicates(judgment_case_ids)
    duplicate_claims = _duplicates(judgment_claim_ids)
    if duplicate_cases:
        errors.append(f"duplicate case judgments: {', '.join(duplicate_cases)}")
    if duplicate_claims:
        errors.append(f"duplicate claim judgments: {', '.join(duplicate_claims)}")
    missing_cases = expected_case_ids.difference(judgment_case_ids)
    extra_cases = set(judgment_case_ids).difference(expected_case_ids)
    extra_claims = set(judgment_claim_ids).difference(expected_claim_ids)
    if missing_cases:
        errors.append(f"missing case judgments: {', '.join(sorted(missing_cases))}")
    if extra_cases:
        errors.append(f"extra case judgments: {', '.join(sorted(extra_cases))}")
    if extra_claims:
        errors.append(f"extra claim judgments: {', '.join(sorted(extra_claims))}")

    judgment_by_case: dict[str, ClaimEvidenceJudgment] = {}
    for judgment in receipt.judgments:
        judgment_by_case.setdefault(judgment.case_id, judgment)

    results: list[ClaimEvidenceCaseResult] = []
    for case in case_set.cases:
        case_judgment = judgment_by_case.get(case.case_id)
        if case_judgment is None:
            results.append(
                ClaimEvidenceCaseResult(
                    case_id=case.case_id,
                    claim_id=case.claim_id,
                    claim_state=case.claim_state,
                    verdict=None,
                    decision=ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED,
                    unsupported_claim_spans=(),
                    reason="semantic judgment is missing",
                )
            )
            continue
        if case_judgment.claim_id != case.claim_id:
            errors.append(
                f"case {case.case_id} judgment names claim {case_judgment.claim_id}, "
                f"not {case.claim_id}"
            )
        span_error = _span_error(case.claim_text, case_judgment)
        if span_error is not None:
            errors.append(f"case {case.case_id}: {span_error}")
        decision = _claim_decision(case.claim_state, case_judgment.verdict)
        results.append(
            ClaimEvidenceCaseResult(
                case_id=case.case_id,
                claim_id=case.claim_id,
                claim_state=case.claim_state,
                verdict=case_judgment.verdict,
                decision=decision,
                unsupported_claim_spans=case_judgment.unsupported_claim_spans,
                reason=case_judgment.reason,
            )
        )
    return tuple(results), errors


def _span_error(claim_text: str, judgment: ClaimEvidenceJudgment) -> str | None:
    if (
        judgment.verdict
        in {
            ClaimEvidenceVerdict.DIRECTLY_SUPPORTED,
            ClaimEvidenceVerdict.SUPPORTED_INFERENCE,
        }
        and judgment.unsupported_claim_spans
    ):
        return "a supporting verdict cannot contain unsupported claim spans"
    for span in judgment.unsupported_claim_spans:
        if span.end > len(claim_text):
            return "unsupported claim span is outside the exact claim text"
        if claim_text[span.start : span.end] != span.text:
            return "unsupported claim span text does not match the exact claim text"
    return None


def _claim_decision(
    claim_state: AnswerClaimState,
    verdict: ClaimEvidenceVerdict,
) -> ClaimEvidenceEntailmentDecision:
    if verdict is ClaimEvidenceVerdict.UNCERTAIN:
        return ClaimEvidenceEntailmentDecision.REVIEW_REQUIRED
    if verdict in {
        ClaimEvidenceVerdict.PARTIAL,
        ClaimEvidenceVerdict.UNSUPPORTED,
        ClaimEvidenceVerdict.CONTRADICTED,
    }:
        return ClaimEvidenceEntailmentDecision.FAILED
    if claim_state in {AnswerClaimState.SUPPORTED, AnswerClaimState.QUALIFIED}:
        return (
            ClaimEvidenceEntailmentDecision.PASSED
            if verdict is ClaimEvidenceVerdict.DIRECTLY_SUPPORTED
            else ClaimEvidenceEntailmentDecision.FAILED
        )
    return (
        ClaimEvidenceEntailmentDecision.PASSED
        if verdict
        in {
            ClaimEvidenceVerdict.DIRECTLY_SUPPORTED,
            ClaimEvidenceVerdict.SUPPORTED_INFERENCE,
        }
        else ClaimEvidenceEntailmentDecision.FAILED
    )


def _semantic_result(
    *,
    decision: ClaimEvidenceEntailmentDecision,
    case_set: ClaimEvidenceCaseSet,
    receipt: ClaimEvidenceJudgmentReceipt | None,
    cases: tuple[ClaimEvidenceCaseResult, ...],
    violations: tuple[str, ...] = (),
    review_reasons: tuple[str, ...] = (),
) -> ClaimEvidenceEntailmentResult:
    return ClaimEvidenceEntailmentResult(
        decision=decision,
        case_set_id=case_set.case_set_id,
        case_set_hash=case_set.case_set_hash,
        receipt_id=None if receipt is None else receipt.receipt_id,
        receipt_hash=None if receipt is None else receipt.receipt_hash,
        cases=cases,
        violations=violations,
        review_reasons=review_reasons,
        skipped_reason=None,
        promotion_eligible=decision is ClaimEvidenceEntailmentDecision.PASSED,
    )


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return tuple(sorted(duplicates))


# Compatibility-friendly concise aliases; the prefixed names remain canonical.
ClaimEvidenceJudgmentVerdict = ClaimEvidenceVerdict
ClaimEvidenceGateDecision = ClaimEvidenceEntailmentDecision
JudgeKind = ClaimEvidenceJudgeKind
