"""Deterministic required-facet coverage for packet-bound research answers.

The exact-answer gate proves that every citation is real and authorized.  This
module checks the complementary question: did the answer use every evidence
facet that the frozen task declared mandatory?  It performs no semantic
inference and must run after :class:`AnswerContractGate`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.atlas import CompactMemoryPacket
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

from .contracts import AnswerStatus, ResearchAnswer

_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_TASK_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{1,95}$"
_HASH_PATTERN = r"^[0-9a-f]{64}$"


class _CoverageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class FacetCoverageStatus(StrEnum):
    COVERED = "covered"
    MISSING = "missing"


class AnswerCoverageDecision(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class RequiredFacet(_CoverageModel):
    """One operator-authored facet represented by accepted packet evidence."""

    facet_id: str = Field(pattern=_ID_PATTERN)
    label: str = Field(min_length=1, max_length=500)
    evidence_ids_any: tuple[str, ...] = Field(min_length=1, max_length=24)
    min_evidence: int = Field(default=1, ge=1, le=24)

    @field_validator("evidence_ids_any")
    @classmethod
    def normalize_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("facet evidence IDs must not be blank")
        normalized = tuple(sorted(value))
        if len(normalized) != len(set(normalized)):
            raise ValueError("facet evidence IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_minimum(self) -> Self:
        if self.min_evidence > len(self.evidence_ids_any):
            raise ValueError("min_evidence cannot exceed the facet evidence set")
        return self


class AnswerCoverageSpec(_CoverageModel):
    """Frozen task-level completeness contract for one compact packet."""

    SCHEMA: ClassVar[str] = "dithyramba.answer_coverage_spec/1.0"

    schema_id: str
    task_id: str = Field(pattern=_TASK_ID_PATTERN)
    packet_id: str = Field(pattern=r"^memory_packet_[0-9a-f]{32}$")
    packet_hash: str = Field(pattern=_HASH_PATTERN)
    allowed_statuses: tuple[AnswerStatus, ...] = Field(min_length=1, max_length=3)
    required_facets: tuple[RequiredFacet, ...] = Field(min_length=1, max_length=64)
    require_open_gap: bool = False
    min_cited_sources: int = Field(default=1, ge=1, le=64)

    @field_validator("allowed_statuses")
    @classmethod
    def normalize_statuses(cls, value: tuple[AnswerStatus, ...]) -> tuple[AnswerStatus, ...]:
        normalized = tuple(sorted(value, key=lambda item: item.value))
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed_statuses must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_spec(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        facet_ids = tuple(item.facet_id for item in self.required_facets)
        if len(facet_ids) != len(set(facet_ids)):
            raise ValueError("facet_id values must be unique")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @property
    def spec_id(self) -> str:
        return canonical_content_id("answer_coverage_spec", self.semantic_payload())

    @property
    def spec_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())


class FacetCoverage(_CoverageModel):
    facet_id: str = Field(pattern=_ID_PATTERN)
    label: str = Field(min_length=1, max_length=500)
    status: FacetCoverageStatus
    min_evidence: int = Field(ge=1, le=24)
    matched_evidence_ids: tuple[str, ...] = Field(max_length=24)
    accepted_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=24)


class AnswerCoverageResult(_CoverageModel):
    decision: AnswerCoverageDecision
    spec_id: str = Field(pattern=r"^answer_coverage_spec_[0-9a-f]{32}$")
    spec_hash: str = Field(pattern=_HASH_PATTERN)
    answer_id: str = Field(pattern=r"^research_answer_[0-9a-f]{32}$")
    answer_hash: str = Field(pattern=_HASH_PATTERN)
    covered_facets: int = Field(ge=0, le=64)
    required_facets: int = Field(ge=1, le=64)
    coverage_basis_points: int = Field(ge=0, le=10_000)
    status_allowed: bool
    open_gap_present: bool
    unique_cited_sources: int = Field(ge=0, le=64)
    facets: tuple[FacetCoverage, ...] = Field(min_length=1, max_length=64)
    violations: tuple[str, ...]


class AnswerCoverageGate:
    """Check actual claim use against a frozen required-facet specification."""

    def evaluate(
        self,
        answer: ResearchAnswer,
        *,
        spec: AnswerCoverageSpec,
        packet: CompactMemoryPacket,
    ) -> AnswerCoverageResult:
        violations: list[str] = []
        if answer.task_id != spec.task_id:
            violations.append("answer task_id does not match the coverage specification")
        if (
            packet.packet_id != spec.packet_id
            or packet.packet_hash != spec.packet_hash
            or answer.route_receipt.packet_id != spec.packet_id
            or answer.route_receipt.packet_hash != spec.packet_hash
        ):
            violations.append("answer, packet, and coverage specification are not bound")

        packet_evidence_ids = {item.evidence_id for item in packet.evidence}
        used_evidence_ids = {
            evidence_id for claim in answer.claims for evidence_id in claim.evidence_ids
        }
        facet_results: list[FacetCoverage] = []
        for facet in spec.required_facets:
            unknown = set(facet.evidence_ids_any).difference(packet_evidence_ids)
            if unknown:
                violations.append(
                    f"facet {facet.facet_id} references evidence outside the packet: "
                    f"{', '.join(sorted(unknown))}"
                )
            matched = tuple(sorted(set(facet.evidence_ids_any).intersection(used_evidence_ids)))
            status = (
                FacetCoverageStatus.COVERED
                if len(matched) >= facet.min_evidence and not unknown
                else FacetCoverageStatus.MISSING
            )
            if status is FacetCoverageStatus.MISSING:
                violations.append(f"required facet {facet.facet_id} is not covered")
            facet_results.append(
                FacetCoverage(
                    facet_id=facet.facet_id,
                    label=facet.label,
                    status=status,
                    min_evidence=facet.min_evidence,
                    matched_evidence_ids=matched,
                    accepted_evidence_ids=facet.evidence_ids_any,
                )
            )

        status_allowed = answer.status in spec.allowed_statuses
        if not status_allowed:
            violations.append(f"answer status {answer.status.value} is not allowed for this task")
        open_gap_present = answer.open_gap is not None
        if spec.require_open_gap and not open_gap_present:
            violations.append("the task requires an explicit open_gap")
        unique_sources = len({item.source_id for item in answer.citations})
        if unique_sources < spec.min_cited_sources:
            violations.append(
                f"answer cites {unique_sources} unique sources; {spec.min_cited_sources} required"
            )

        covered = sum(item.status is FacetCoverageStatus.COVERED for item in facet_results)
        required = len(facet_results)
        return AnswerCoverageResult(
            decision=(
                AnswerCoverageDecision.COMPLETE
                if not violations
                else AnswerCoverageDecision.INCOMPLETE
            ),
            spec_id=spec.spec_id,
            spec_hash=spec.spec_hash,
            answer_id=answer.answer_id,
            answer_hash=answer.answer_hash,
            covered_facets=covered,
            required_facets=required,
            coverage_basis_points=(covered * 10_000) // required,
            status_allowed=status_allowed,
            open_gap_present=open_gap_present,
            unique_cited_sources=unique_sources,
            facets=tuple(facet_results),
            violations=tuple(violations),
        )
