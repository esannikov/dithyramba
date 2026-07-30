"""Public contracts for short, inspectable, evidence-grounded idea traces.

An IdeaTrace is not a hidden chain of thought.  It is a bounded public artifact:
short statements, named operations, explicit premises, concise warrants, and
exact bindings to claims that have already passed the claim/evidence route.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dithyramba.answers import (
    ClaimEvidenceEntailmentDecision,
    ClaimEvidenceVerdict,
)
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_TASK_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{1,95}$"
_STEP_ID_PATTERN = r"^idea_trace_step_[0-9a-f]{32}$"
_TRACE_ID_PATTERN = r"^idea_trace_[0-9a-f]{32}$"
_CLOSURE_ID_PATTERN = r"^reasoning_closure_[0-9a-f]{32}$"
_CASE_ID_PATTERN = r"^claim_evidence_case_[0-9a-f]{32}$"
_CASE_SET_ID_PATTERN = r"^claim_evidence_case_set_[0-9a-f]{32}$"
_RECEIPT_ID_PATTERN = r"^claim_evidence_judgment_receipt_[0-9a-f]{32}$"


class _ReasoningModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class TraceAuthorKind(StrEnum):
    HUMAN = "human"
    MODEL = "model"
    DETERMINISTIC = "deterministic"


class ReasoningOperation(StrEnum):
    EXTRACT = "extract"
    COMPARE = "compare"
    CONNECT = "connect"
    INFER = "infer"


class ReasoningQualifier(StrEnum):
    DIRECT = "direct"
    BOUNDED = "bounded"
    CONTESTED = "contested"


class ReasoningClosureDecision(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class ReasoningGap(_ReasoningModel):
    question: str = Field(min_length=1, max_length=2_000)
    why_it_matters: str = Field(min_length=1, max_length=2_000)
    next_evidence: str = Field(min_length=1, max_length=2_000)


class IdeaTraceStep(_ReasoningModel):
    """One public statement and its explicit derivation shape."""

    step_id: str = Field(pattern=_STEP_ID_PATTERN)
    claim_id: str = Field(pattern=_ID_PATTERN)
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    statement: str = Field(min_length=1, max_length=4_000)
    operation: ReasoningOperation
    premise_step_ids: tuple[str, ...] = Field(max_length=8)
    warrant: str | None = Field(default=None, min_length=1, max_length=2_000)
    qualifier: ReasoningQualifier
    counterevidence_ids: tuple[str, ...] = Field(max_length=24)

    @model_validator(mode="after")
    def validate_step(self) -> Self:
        if len(self.premise_step_ids) != len(set(self.premise_step_ids)):
            raise ValueError("premise_step_ids must be unique")
        if self.step_id in self.premise_step_ids:
            raise ValueError("an IdeaTrace step cannot cite itself as a premise")
        if len(self.counterevidence_ids) != len(set(self.counterevidence_ids)):
            raise ValueError("counterevidence_ids must be unique")

        if self.operation is ReasoningOperation.EXTRACT:
            if self.premise_step_ids:
                raise ValueError("extract steps cannot have premise steps")
            if self.warrant is not None:
                raise ValueError("extract steps do not carry a derivation warrant")
            if self.qualifier not in {
                ReasoningQualifier.DIRECT,
                ReasoningQualifier.CONTESTED,
            }:
                raise ValueError("extract steps must be direct or contested")
        else:
            minimum = (
                2
                if self.operation
                in {
                    ReasoningOperation.COMPARE,
                    ReasoningOperation.CONNECT,
                }
                else 1
            )
            if len(self.premise_step_ids) < minimum:
                raise ValueError(
                    f"{self.operation.value} steps require at least {minimum} premises"
                )
            if self.warrant is None:
                raise ValueError("derived steps require a concise public warrant")
            if self.qualifier is ReasoningQualifier.DIRECT:
                raise ValueError("derived steps cannot be labelled direct")

        if self.qualifier is ReasoningQualifier.CONTESTED and not self.counterevidence_ids:
            raise ValueError("contested steps require explicit counterevidence_ids")
        if self.qualifier is not ReasoningQualifier.CONTESTED and self.counterevidence_ids:
            raise ValueError("counterevidence_ids are reserved for contested steps")
        if self.step_id != canonical_content_id("idea_trace_step", self.semantic_payload()):
            raise ValueError("step_id does not match the canonical IdeaTrace step content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "case_id": self.case_id,
            "statement": self.statement,
            "operation": self.operation.value,
            "premise_step_ids": list(self.premise_step_ids),
            "warrant": self.warrant,
            "qualifier": self.qualifier.value,
            "counterevidence_ids": list(self.counterevidence_ids),
        }

    @classmethod
    def create(
        cls,
        *,
        claim_id: str,
        case_id: str,
        statement: str,
        operation: ReasoningOperation,
        premise_step_ids: tuple[str, ...] = (),
        warrant: str | None = None,
        qualifier: ReasoningQualifier,
        counterevidence_ids: tuple[str, ...] = (),
    ) -> IdeaTraceStep:
        payload: dict[str, object] = {
            "claim_id": claim_id,
            "case_id": case_id,
            "statement": statement,
            "operation": operation.value,
            "premise_step_ids": list(premise_step_ids),
            "warrant": warrant,
            "qualifier": qualifier.value,
            "counterevidence_ids": list(counterevidence_ids),
        }
        return cls(
            step_id=canonical_content_id("idea_trace_step", payload),
            claim_id=claim_id,
            case_id=case_id,
            statement=statement,
            operation=operation,
            premise_step_ids=premise_step_ids,
            warrant=warrant,
            qualifier=qualifier,
            counterevidence_ids=counterevidence_ids,
        )


class IdeaTrace(_ReasoningModel):
    """Content-addressed, bounded reasoning candidate over one accepted answer."""

    SCHEMA: ClassVar[str] = "dithyramba.idea_trace/1.0"

    schema_id: str
    trace_id: str = Field(pattern=_TRACE_ID_PATTERN)
    trace_hash: str = Field(pattern=_HASH_PATTERN)
    task_id: str = Field(pattern=_TASK_ID_PATTERN)
    answer_id: str = Field(pattern=r"^research_answer_[0-9a-f]{32}$")
    answer_hash: str = Field(pattern=_HASH_PATTERN)
    packet_id: str = Field(pattern=r"^memory_packet_[0-9a-f]{32}$")
    packet_hash: str = Field(pattern=_HASH_PATTERN)
    case_set_id: str = Field(pattern=_CASE_SET_ID_PATTERN)
    case_set_hash: str = Field(pattern=_HASH_PATTERN)
    receipt_id: str = Field(pattern=_RECEIPT_ID_PATTERN)
    receipt_hash: str = Field(pattern=_HASH_PATTERN)
    question: str = Field(min_length=1, max_length=4_000)
    author_kind: TraceAuthorKind
    author_profile_id: str = Field(pattern=_ID_PATTERN)
    author_profile_hash: str = Field(pattern=_HASH_PATTERN)
    instruction_hash: str = Field(pattern=_HASH_PATTERN)
    raw_response_hash: str = Field(pattern=_HASH_PATTERN)
    steps: tuple[IdeaTraceStep, ...] = Field(min_length=1, max_length=32)
    final_step_id: str = Field(pattern=_STEP_ID_PATTERN)
    gaps: tuple[ReasoningGap, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def validate_trace(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        step_ids = tuple(item.step_id for item in self.steps)
        claim_ids = tuple(item.claim_id for item in self.steps)
        case_ids = tuple(item.case_id for item in self.steps)
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("IdeaTrace step_id values must be unique")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("one answer claim may appear only once in an IdeaTrace")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("one claim-evidence case may appear only once in an IdeaTrace")
        if self.final_step_id != step_ids[-1]:
            raise ValueError("the final IdeaTrace step must be the last step")

        prior: set[str] = set()
        for step in self.steps:
            missing = set(step.premise_step_ids).difference(prior)
            if missing:
                raise ValueError(
                    "premise steps must exist earlier in the trace: " + ", ".join(sorted(missing))
                )
            prior.add(step.step_id)

        ancestors = _ancestor_ids(self.steps, self.final_step_id)
        orphaned = set(step_ids).difference(ancestors | {self.final_step_id})
        if orphaned:
            raise ValueError(
                "every IdeaTrace step must contribute to the final step: "
                + ", ".join(sorted(orphaned))
            )
        gap_keys = tuple(
            (item.question, item.why_it_matters, item.next_evidence) for item in self.gaps
        )
        if len(gap_keys) != len(set(gap_keys)):
            raise ValueError("IdeaTrace gaps must be unique")

        payload = self.semantic_payload()
        if self.trace_id != canonical_content_id("idea_trace", payload):
            raise ValueError("trace_id does not match the canonical IdeaTrace content")
        if self.trace_hash != canonical_sha256_hex(payload):
            raise ValueError("trace_hash does not match the canonical IdeaTrace content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "task_id": self.task_id,
            "answer_id": self.answer_id,
            "answer_hash": self.answer_hash,
            "packet_id": self.packet_id,
            "packet_hash": self.packet_hash,
            "case_set_id": self.case_set_id,
            "case_set_hash": self.case_set_hash,
            "receipt_id": self.receipt_id,
            "receipt_hash": self.receipt_hash,
            "question": self.question,
            "author_kind": self.author_kind.value,
            "author_profile_id": self.author_profile_id,
            "author_profile_hash": self.author_profile_hash,
            "instruction_hash": self.instruction_hash,
            "raw_response_hash": self.raw_response_hash,
            "steps": [{"step_id": item.step_id, **item.semantic_payload()} for item in self.steps],
            "final_step_id": self.final_step_id,
            "gaps": [item.model_dump(mode="json") for item in self.gaps],
        }

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        answer_id: str,
        answer_hash: str,
        packet_id: str,
        packet_hash: str,
        case_set_id: str,
        case_set_hash: str,
        receipt_id: str,
        receipt_hash: str,
        question: str,
        author_kind: TraceAuthorKind,
        author_profile_id: str,
        author_profile_hash: str,
        instruction_hash: str,
        raw_response_hash: str,
        steps: tuple[IdeaTraceStep, ...],
        final_step_id: str,
        gaps: tuple[ReasoningGap, ...] = (),
    ) -> IdeaTrace:
        payload: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "task_id": task_id,
            "answer_id": answer_id,
            "answer_hash": answer_hash,
            "packet_id": packet_id,
            "packet_hash": packet_hash,
            "case_set_id": case_set_id,
            "case_set_hash": case_set_hash,
            "receipt_id": receipt_id,
            "receipt_hash": receipt_hash,
            "question": question,
            "author_kind": author_kind.value,
            "author_profile_id": author_profile_id,
            "author_profile_hash": author_profile_hash,
            "instruction_hash": instruction_hash,
            "raw_response_hash": raw_response_hash,
            "steps": [{"step_id": item.step_id, **item.semantic_payload()} for item in steps],
            "final_step_id": final_step_id,
            "gaps": [item.model_dump(mode="json") for item in gaps],
        }
        return cls(
            schema_id=cls.SCHEMA,
            trace_id=canonical_content_id("idea_trace", payload),
            trace_hash=canonical_sha256_hex(payload),
            task_id=task_id,
            answer_id=answer_id,
            answer_hash=answer_hash,
            packet_id=packet_id,
            packet_hash=packet_hash,
            case_set_id=case_set_id,
            case_set_hash=case_set_hash,
            receipt_id=receipt_id,
            receipt_hash=receipt_hash,
            question=question,
            author_kind=author_kind,
            author_profile_id=author_profile_id,
            author_profile_hash=author_profile_hash,
            instruction_hash=instruction_hash,
            raw_response_hash=raw_response_hash,
            steps=steps,
            final_step_id=final_step_id,
            gaps=gaps,
        )


class ReasoningStepClosure(_ReasoningModel):
    step_id: str = Field(pattern=_STEP_ID_PATTERN)
    claim_id: str = Field(pattern=_ID_PATTERN)
    case_id: str = Field(pattern=_CASE_ID_PATTERN)
    verdict: ClaimEvidenceVerdict | None
    semantic_decision: ClaimEvidenceEntailmentDecision
    closed: bool
    reason: str = Field(min_length=1, max_length=4_000)


class ReasoningClosureResult(_ReasoningModel):
    """Content-addressed structural verdict; never a truth or review decision."""

    SCHEMA: ClassVar[str] = "dithyramba.reasoning_closure/1.0"

    schema_id: str
    closure_id: str = Field(pattern=_CLOSURE_ID_PATTERN)
    closure_hash: str = Field(pattern=_HASH_PATTERN)
    trace_id: str = Field(pattern=_TRACE_ID_PATTERN)
    trace_hash: str = Field(pattern=_HASH_PATTERN)
    case_set_id: str = Field(pattern=_CASE_SET_ID_PATTERN)
    case_set_hash: str = Field(pattern=_HASH_PATTERN)
    receipt_id: str = Field(pattern=_RECEIPT_ID_PATTERN)
    receipt_hash: str = Field(pattern=_HASH_PATTERN)
    decision: ReasoningClosureDecision
    steps: tuple[ReasoningStepClosure, ...] = Field(min_length=1, max_length=32)
    violations: tuple[str, ...]
    review_reasons: tuple[str, ...]
    review_eligible: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if self.review_eligible != (self.decision is ReasoningClosureDecision.PASSED):
            raise ValueError("only a passed closure result is review-eligible")
        if self.decision is ReasoningClosureDecision.PASSED and (
            self.violations or self.review_reasons or not all(item.closed for item in self.steps)
        ):
            raise ValueError("passed closure requires every step to close without findings")
        payload = self.semantic_payload()
        if self.closure_id != canonical_content_id("reasoning_closure", payload):
            raise ValueError("closure_id does not match the canonical closure content")
        if self.closure_hash != canonical_sha256_hex(payload):
            raise ValueError("closure_hash does not match the canonical closure content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "trace_id": self.trace_id,
            "trace_hash": self.trace_hash,
            "case_set_id": self.case_set_id,
            "case_set_hash": self.case_set_hash,
            "receipt_id": self.receipt_id,
            "receipt_hash": self.receipt_hash,
            "decision": self.decision.value,
            "steps": [item.model_dump(mode="json") for item in self.steps],
            "violations": list(self.violations),
            "review_reasons": list(self.review_reasons),
            "review_eligible": self.review_eligible,
        }

    @classmethod
    def create(
        cls,
        *,
        trace: IdeaTrace,
        decision: ReasoningClosureDecision,
        steps: tuple[ReasoningStepClosure, ...],
        violations: tuple[str, ...] = (),
        review_reasons: tuple[str, ...] = (),
    ) -> ReasoningClosureResult:
        review_eligible = decision is ReasoningClosureDecision.PASSED
        payload: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "trace_id": trace.trace_id,
            "trace_hash": trace.trace_hash,
            "case_set_id": trace.case_set_id,
            "case_set_hash": trace.case_set_hash,
            "receipt_id": trace.receipt_id,
            "receipt_hash": trace.receipt_hash,
            "decision": decision.value,
            "steps": [item.model_dump(mode="json") for item in steps],
            "violations": list(violations),
            "review_reasons": list(review_reasons),
            "review_eligible": review_eligible,
        }
        return cls(
            schema_id=cls.SCHEMA,
            closure_id=canonical_content_id("reasoning_closure", payload),
            closure_hash=canonical_sha256_hex(payload),
            trace_id=trace.trace_id,
            trace_hash=trace.trace_hash,
            case_set_id=trace.case_set_id,
            case_set_hash=trace.case_set_hash,
            receipt_id=trace.receipt_id,
            receipt_hash=trace.receipt_hash,
            decision=decision,
            steps=steps,
            violations=violations,
            review_reasons=review_reasons,
            review_eligible=review_eligible,
        )


def _ancestor_ids(steps: tuple[IdeaTraceStep, ...], final_step_id: str) -> set[str]:
    by_id = {item.step_id: item for item in steps}
    pending = list(by_id[final_step_id].premise_step_ids)
    ancestors: set[str] = set()
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        pending.extend(by_id[current].premise_step_ids)
    return ancestors
