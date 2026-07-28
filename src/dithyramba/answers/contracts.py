"""Strict answer schema and deterministic proof-binding gate.

The language model may compose prose, but it cannot grant itself evidence.
This gate accepts an answer only when every cited fragment belongs to the
frozen compact packet and is still present verbatim in its local artifact.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from dithyramba.atlas.memory_packet import CompactMemoryPacket
from dithyramba.atlas.models import AtlasEvidence, AtlasSource
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex, sha256_hex

_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_TASK_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{1,95}$"
_HASH_PATTERN = r"^[0-9a-f]{64}$"


class _AnswerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class AnswerStatus(StrEnum):
    SUPPORTED = "supported"
    QUALIFIED = "qualified"
    INSUFFICIENT = "insufficient"


class AnswerClaimState(StrEnum):
    SUPPORTED = "supported"
    QUALIFIED = "qualified"
    INFERENCE = "inference"


class AnswerRouteReceipt(_AnswerModel):
    packet_id: str = Field(pattern=r"^memory_packet_[0-9a-f]{32}$")
    packet_hash: str = Field(pattern=_HASH_PATTERN)


class AnswerCitation(_AnswerModel):
    evidence_id: str = Field(pattern=_ID_PATTERN)
    source_id: str = Field(pattern=_ID_PATTERN)
    exact_quote: str = Field(min_length=1, max_length=1_000)


class AnswerClaim(_AnswerModel):
    claim_id: str = Field(pattern=_ID_PATTERN)
    text: str = Field(min_length=1, max_length=4_000)
    status: AnswerClaimState
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=24)


class AnswerGap(_AnswerModel):
    question: str = Field(min_length=1, max_length=2_000)
    why_it_matters: str = Field(min_length=1, max_length=2_000)
    next_evidence: str = Field(min_length=1, max_length=2_000)


class ResearchAnswer(_AnswerModel):
    """One agent answer with machine-checkable proof references."""

    SCHEMA: ClassVar[str] = "dithyramba.research_answer/1.0"

    schema_id: str
    task_id: str = Field(pattern=_TASK_ID_PATTERN)
    route_receipt: AnswerRouteReceipt
    status: AnswerStatus
    short_answer: str = Field(min_length=1, max_length=8_000)
    claims: tuple[AnswerClaim, ...] = Field(max_length=64)
    citations: tuple[AnswerCitation, ...] = Field(max_length=64)
    limitations: tuple[str, ...] = Field(max_length=24)
    open_gap: AnswerGap | None

    @model_validator(mode="after")
    def validate_answer_shape(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        claim_ids = tuple(item.claim_id for item in self.claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique")
        citation_ids = tuple(item.evidence_id for item in self.citations)
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("one evidence_id may appear only once in citations")
        cited = set(citation_ids)
        used: set[str] = set()
        for claim in self.claims:
            if len(claim.evidence_ids) != len(set(claim.evidence_ids)):
                raise ValueError("claim evidence_ids must be unique")
            missing = set(claim.evidence_ids).difference(cited)
            if missing:
                raise ValueError(f"claim references uncited evidence: {', '.join(sorted(missing))}")
            used.update(claim.evidence_ids)
        unused = cited.difference(used)
        if unused:
            raise ValueError(f"citations must support a claim: {', '.join(sorted(unused))}")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("limitations must be unique")
        if self.open_gap is not None and self.open_gap.question in self.limitations:
            raise ValueError("open_gap.question must not duplicate a limitation")
        if self.status is AnswerStatus.INSUFFICIENT:
            if self.open_gap is None:
                raise ValueError("insufficient answers require an explicit open_gap")
        elif not self.claims or not self.citations:
            raise ValueError("supported and qualified answers require claims and citations")
        if self.status is AnswerStatus.SUPPORTED and not any(
            item.status is AnswerClaimState.SUPPORTED for item in self.claims
        ):
            raise ValueError("supported answer requires at least one supported claim")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @property
    def answer_id(self) -> str:
        return canonical_content_id("research_answer", self.semantic_payload())

    @property
    def answer_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())


class GateCheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateCheck(_AnswerModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,95}$")
    status: GateCheckStatus
    detail: str = Field(min_length=1, max_length=2_000)


class AnswerDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class AnswerContractResult(_AnswerModel):
    decision: AnswerDecision
    answer_id: str | None = Field(default=None, pattern=r"^research_answer_[0-9a-f]{32}$")
    answer_hash: str | None = Field(default=None, pattern=_HASH_PATTERN)
    checks: tuple[GateCheck, ...] = Field(min_length=1)
    violations: tuple[str, ...]


class AnswerContractGate:
    """Validate schema, route receipt, accepted IDs, and exact local evidence."""

    def evaluate_payload(
        self,
        payload: object,
        *,
        task_id: str,
        packet: CompactMemoryPacket,
        artifact_root: str | Path,
    ) -> AnswerContractResult:
        try:
            if isinstance(payload, (str, bytes, bytearray)):
                answer = ResearchAnswer.model_validate_json(payload)
            else:
                answer = ResearchAnswer.model_validate(payload)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            return _result(
                (GateCheck(name="schema", status=GateCheckStatus.FAILED, detail=detail),)
            )
        return self.evaluate(
            answer,
            task_id=task_id,
            packet=packet,
            artifact_root=artifact_root,
        )

    def evaluate(
        self,
        answer: ResearchAnswer,
        *,
        task_id: str,
        packet: CompactMemoryPacket,
        artifact_root: str | Path,
    ) -> AnswerContractResult:
        checks: list[GateCheck] = [
            GateCheck(
                name="schema",
                status=GateCheckStatus.PASSED,
                detail="Strict answer schema is valid.",
            ),
            _boolean_check(
                "task_identity",
                answer.task_id == task_id,
                "Answer task_id matches the frozen task.",
                "Answer task_id does not match the frozen task.",
            ),
            _boolean_check(
                "route_receipt",
                answer.route_receipt.packet_id == packet.packet_id
                and answer.route_receipt.packet_hash == packet.packet_hash,
                "Answer points to the exact compact packet.",
                "Answer route receipt does not match the compact packet.",
            ),
        ]
        evidence_by_id = {item.evidence_id: item for item in packet.evidence}
        source_by_id = {item.source_id: item for item in packet.sources}
        membership_errors: list[str] = []
        for citation in answer.citations:
            evidence = evidence_by_id.get(citation.evidence_id)
            if evidence is None:
                membership_errors.append(f"unknown evidence {citation.evidence_id}")
            elif evidence.source_id != citation.source_id:
                membership_errors.append(
                    f"{citation.evidence_id} belongs to {evidence.source_id}, "
                    f"not {citation.source_id}"
                )
        checks.append(
            _errors_check(
                "packet_membership",
                membership_errors,
                "Every citation belongs to accepted evidence in the compact packet.",
            )
        )

        artifact_errors = self._artifact_errors(
            answer.citations,
            evidence_by_id,
            source_by_id,
            Path(artifact_root).expanduser(),
        )
        checks.append(
            _errors_check(
                "exact_local_evidence",
                artifact_errors,
                "Every quotation and bound fragment is exact in its local source artifact.",
            )
        )
        return _result(tuple(checks), answer=answer)

    @staticmethod
    def _artifact_errors(
        citations: tuple[AnswerCitation, ...],
        evidence_by_id: dict[str, AtlasEvidence],
        source_by_id: dict[str, AtlasSource],
        artifact_root: Path,
    ) -> list[str]:
        errors: list[str] = []
        root = artifact_root.resolve(strict=False)
        for citation in citations:
            evidence_value = evidence_by_id.get(citation.evidence_id)
            source_value = source_by_id.get(citation.source_id)
            if evidence_value is None or source_value is None:
                continue
            evidence = evidence_value
            source = source_value
            if evidence.excerpt is None or citation.exact_quote not in evidence.excerpt:
                errors.append(
                    f"{citation.evidence_id}: quotation is not inside the accepted excerpt"
                )
                continue
            if evidence.fragment_text_sha256 is None or evidence.source_address is None:
                errors.append(f"{citation.evidence_id}: exact fragment binding is absent")
                continue
            if source.artifact_path is None:
                errors.append(f"{citation.source_id}: local artifact path is absent")
                continue
            artifact = (root / source.artifact_path).resolve(strict=False)
            try:
                artifact.relative_to(root)
            except ValueError:
                errors.append(f"{citation.source_id}: artifact escapes the configured root")
                continue
            if not artifact.is_file():
                errors.append(f"{citation.source_id}: local artifact is missing")
                continue
            artifact_text = artifact.read_text(encoding="utf-8")
            lines = artifact_text.splitlines()
            line_start = evidence.source_address.line_start
            line_end = evidence.source_address.line_end
            if line_end > len(lines):
                errors.append(f"{citation.evidence_id}: bound source range is unavailable")
                continue
            bound_fragment = "\n".join(lines[line_start - 1 : line_end])
            if sha256_hex(bound_fragment.encode("utf-8")) != evidence.fragment_text_sha256:
                errors.append(f"{citation.evidence_id}: bound source fragment has drifted")
                continue
            if evidence.excerpt not in bound_fragment:
                errors.append(
                    f"{citation.evidence_id}: accepted excerpt is outside its bound fragment"
                )
        return errors


def _boolean_check(name: str, condition: bool, passed: str, failed: str) -> GateCheck:
    return GateCheck(
        name=name,
        status=GateCheckStatus.PASSED if condition else GateCheckStatus.FAILED,
        detail=passed if condition else failed,
    )


def _errors_check(name: str, errors: list[str], passed: str) -> GateCheck:
    return GateCheck(
        name=name,
        status=GateCheckStatus.FAILED if errors else GateCheckStatus.PASSED,
        detail="; ".join(errors) if errors else passed,
    )


def _result(
    checks: tuple[GateCheck, ...],
    *,
    answer: ResearchAnswer | None = None,
) -> AnswerContractResult:
    violations = tuple(item.detail for item in checks if item.status is GateCheckStatus.FAILED)
    accepted = not violations
    return AnswerContractResult(
        decision=AnswerDecision.ACCEPTED if accepted else AnswerDecision.REJECTED,
        answer_id=answer.answer_id if accepted and answer is not None else None,
        answer_hash=answer.answer_hash if accepted and answer is not None else None,
        checks=checks,
        violations=violations,
    )
