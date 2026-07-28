"""Exact-fragment coverage gate between retrieval and an evidence-backed answer.

Retrieval ranks plausible fragments.  This module performs the narrower,
deterministic check that every declared evidence requirement is actually
covered by an exact fragment with the demanded source role and literal anchors.
It does not infer historical truth and it does not call a model.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
)
from dithyramba.ingest import (
    SOURCE_ADDRESS_SCHEMA,
    MarkdownSourceAddress,
    PdfSourceAddress,
)

_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_FRAGMENT_ID_PATTERN = re.compile(r"^fragment_[a-z0-9]+(?:_[a-z0-9]+)*$")
_REFERENCE_PATTERN = re.compile(r"^[a-z][a-z0-9]+(?:_[a-z0-9]+)*$")
_SPACE_PATTERN = re.compile(r"\s+")


class EvidenceGateContractError(ValueError):
    """An evidence coverage input violates the deterministic gate contract."""


class EvidenceAnswerability(StrEnum):
    """Whether the frozen question expects evidence or an explicit corpus gap."""

    ANSWERABLE = "answerable"
    NOT_IN_CORPUS = "not_in_corpus"


class RequirementCoverageStatus(StrEnum):
    COVERED = "covered"
    MISSING = "missing"


class EvidenceGateDecision(StrEnum):
    """Human-facing terminal states of one gate evaluation."""

    READY = "ready"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    GAP_PRESERVED = "gap_preserved"
    GAP_CHALLENGED = "gap_challenged"


class _FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class EvidenceCandidate(_FrozenContract):
    """One exact retrieved fragment plus source-role metadata used by the gate."""

    rank: int = Field(ge=1, le=10_000)
    source_fragment_id: str
    source_id: str
    text: str
    source_address: dict[str, object]
    source_kind: str
    source_family: str
    authority: str
    independence_group: str
    evidence_tags: tuple[str, ...] = ()

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        if type(value) is not str or _FRAGMENT_ID_PATTERN.fullmatch(value) is None:
            raise EvidenceGateContractError("source_fragment_id must use fragment_<key>")
        return value

    @field_validator("source_id")
    @classmethod
    def _source_id(cls, value: str) -> str:
        return _require_reference(value, "source_id")

    @field_validator(
        "source_kind",
        "source_family",
        "authority",
        "independence_group",
    )
    @classmethod
    def _metadata(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFC", value)
        if not normalized.strip() or "\x00" in normalized:
            raise EvidenceGateContractError("candidate text must be nonblank NFC text")
        return normalized

    @field_validator("source_address", mode="before")
    @classmethod
    def _source_address(cls, value: object) -> dict[str, object]:
        return _validated_source_address(value)

    @field_validator("evidence_tags")
    @classmethod
    def _tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _normalized_selector_tuple(value, "evidence_tags")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "source_fragment_id": self.source_fragment_id,
            "source_id": self.source_id,
            "text": self.text,
            "source_address": self.source_address,
            "source_kind": self.source_kind,
            "source_family": self.source_family,
            "authority": self.authority,
            "independence_group": self.independence_group,
            "evidence_tags": list(self.evidence_tags),
        }


class EvidenceRequirement(_FrozenContract):
    """One explicit condition that a sufficient answer must satisfy."""

    SCHEMA: ClassVar[str] = "dithyramba.evidence_requirement/1.0"

    key: str
    label: str
    allowed_source_ids: tuple[str, ...] = ()
    source_kinds_any: tuple[str, ...] = ()
    source_families_any: tuple[str, ...] = ()
    authorities_any: tuple[str, ...] = ()
    anchor_groups: tuple[tuple[str, ...], ...] = ()
    required_tags: tuple[str, ...] = ()
    forbidden_anchors: tuple[str, ...] = ()
    min_independent_groups: int = Field(default=1, ge=1, le=20)

    @field_validator("key")
    @classmethod
    def _key(cls, value: str) -> str:
        if type(value) is not str or _KEY_PATTERN.fullmatch(value) is None:
            raise EvidenceGateContractError("requirement key must be stable snake_case")
        return value

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        return _require_text(value, "label")

    @field_validator("allowed_source_ids")
    @classmethod
    def _source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for source_id in value:
            _require_reference(source_id, "allowed_source_ids")
        return _unique_sorted(value, "allowed_source_ids")

    @field_validator(
        "source_kinds_any",
        "source_families_any",
        "authorities_any",
        "required_tags",
        "forbidden_anchors",
    )
    @classmethod
    def _selectors(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        return _normalized_selector_tuple(value, info.field_name)

    @field_validator("anchor_groups")
    @classmethod
    def _anchor_groups(cls, value: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
        normalized_groups: list[tuple[str, ...]] = []
        for group in value:
            normalized = _normalized_selector_tuple(group, "anchor_groups")
            if not normalized:
                raise EvidenceGateContractError("anchor groups cannot be empty")
            normalized_groups.append(normalized)
        if len(set(normalized_groups)) != len(normalized_groups):
            raise EvidenceGateContractError("anchor groups must be unique")
        return tuple(sorted(normalized_groups, key=_canonical_sort_key))

    @model_validator(mode="after")
    def _has_a_real_condition(self) -> EvidenceRequirement:
        selectors = (
            self.allowed_source_ids,
            self.source_kinds_any,
            self.source_families_any,
            self.authorities_any,
            self.anchor_groups,
            self.required_tags,
            self.forbidden_anchors,
        )
        if not any(selectors):
            raise EvidenceGateContractError(
                "EvidenceRequirement must declare at least one exact condition"
            )
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "key": self.key,
            "label": self.label,
            "allowed_source_ids": list(self.allowed_source_ids),
            "source_kinds_any": list(self.source_kinds_any),
            "source_families_any": list(self.source_families_any),
            "authorities_any": list(self.authorities_any),
            "anchor_groups": [list(group) for group in self.anchor_groups],
            "required_tags": list(self.required_tags),
            "forbidden_anchors": list(self.forbidden_anchors),
            "min_independent_groups": self.min_independent_groups,
        }

    @property
    def evidence_requirement_id(self) -> str:
        return canonical_content_id("evidence_requirement", self.semantic_payload())

    @property
    def requirement_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())


class EvidenceGateSpec(_FrozenContract):
    """Frozen question-level sufficiency contract evaluated after retrieval."""

    SCHEMA: ClassVar[str] = "dithyramba.evidence_gate_spec/1.0"

    query_key: str
    question: str
    expected_answerability: EvidenceAnswerability
    requirements: tuple[EvidenceRequirement, ...]
    min_total_independent_groups: int = Field(default=1, ge=1, le=50)
    matching_profile: Literal["exact_fragment_unicode_v1"] = "exact_fragment_unicode_v1"

    @field_validator("query_key")
    @classmethod
    def _query_key(cls, value: str) -> str:
        if type(value) is not str or _KEY_PATTERN.fullmatch(value) is None:
            raise EvidenceGateContractError("query_key must be stable snake_case")
        return value

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        return _require_text(value, "question")

    @field_validator("expected_answerability")
    @classmethod
    def _answerability(cls, value: EvidenceAnswerability) -> EvidenceAnswerability:
        if not isinstance(value, EvidenceAnswerability):
            raise EvidenceGateContractError("expected_answerability must be EvidenceAnswerability")
        return value

    @field_validator("requirements")
    @classmethod
    def _requirements(
        cls, value: tuple[EvidenceRequirement, ...]
    ) -> tuple[EvidenceRequirement, ...]:
        if not value:
            raise EvidenceGateContractError("gate spec requires at least one requirement")
        if any(type(item) is not EvidenceRequirement for item in value):
            raise EvidenceGateContractError("requirements must be exact EvidenceRequirement values")
        ordered = tuple(sorted(value, key=lambda item: item.key.encode("ascii")))
        if len({item.key for item in ordered}) != len(ordered):
            raise EvidenceGateContractError("requirement keys must be unique")
        return ordered

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "query_key": self.query_key,
            "question": self.question,
            "expected_answerability": self.expected_answerability.value,
            "requirements": [item.semantic_payload() for item in self.requirements],
            "min_total_independent_groups": self.min_total_independent_groups,
            "matching_profile": self.matching_profile,
        }

    @property
    def spec_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def spec_id(self) -> str:
        return canonical_content_id("evidence_gate_spec", self.semantic_payload())


class CandidateAssessment(_FrozenContract):
    """Why one near candidate did not satisfy a requirement."""

    source_fragment_id: str
    source_id: str
    rank: int = Field(ge=1, le=10_000)
    rejection_codes: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_id": self.source_id,
            "rank": self.rank,
            "rejection_codes": list(self.rejection_codes),
        }


class RequirementCoverage(_FrozenContract):
    """Deterministic coverage result for exactly one EvidenceRequirement."""

    evidence_requirement_id: str
    requirement_key: str
    label: str
    status: RequirementCoverageStatus
    matched_fragment_ids: tuple[str, ...]
    matched_source_ids: tuple[str, ...]
    matched_independence_groups: tuple[str, ...]
    missing_conditions: tuple[str, ...]
    closest_candidates: tuple[CandidateAssessment, ...]

    def payload(self) -> dict[str, object]:
        return {
            "evidence_requirement_id": self.evidence_requirement_id,
            "requirement_key": self.requirement_key,
            "label": self.label,
            "status": self.status.value,
            "matched_fragment_ids": list(self.matched_fragment_ids),
            "matched_source_ids": list(self.matched_source_ids),
            "matched_independence_groups": list(self.matched_independence_groups),
            "missing_conditions": list(self.missing_conditions),
            "closest_candidates": [item.payload() for item in self.closest_candidates],
        }


class EvidenceCoverageResult(_FrozenContract):
    """Content-addressed gate decision with requirement-level explanations."""

    SCHEMA: ClassVar[str] = "dithyramba.evidence_coverage_result/1.0"

    query_key: str
    spec_id: str
    spec_hash: str
    candidate_set_hash: str
    expected_answerability: EvidenceAnswerability
    decision: EvidenceGateDecision
    covered_requirement_ids: tuple[str, ...]
    missing_requirement_ids: tuple[str, ...]
    matched_independence_groups: tuple[str, ...]
    gate_conditions: tuple[str, ...]
    requirements: tuple[RequirementCoverage, ...]

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "query_key": self.query_key,
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "candidate_set_hash": self.candidate_set_hash,
            "expected_answerability": self.expected_answerability.value,
            "decision": self.decision.value,
            "covered_requirement_ids": list(self.covered_requirement_ids),
            "missing_requirement_ids": list(self.missing_requirement_ids),
            "matched_independence_groups": list(self.matched_independence_groups),
            "gate_conditions": list(self.gate_conditions),
            "requirements": [item.payload() for item in self.requirements],
        }

    @property
    def result_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def result_id(self) -> str:
        return canonical_content_id("evidence_coverage", self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


class EvidenceCoverageGate:
    """Pure, provider-free evaluator for one frozen EvidenceGateSpec."""

    _CLOSEST_LIMIT = 3

    def __init__(self, spec: EvidenceGateSpec) -> None:
        if type(spec) is not EvidenceGateSpec:
            raise EvidenceGateContractError("spec must be an exact EvidenceGateSpec")
        self._spec = spec

    @property
    def spec(self) -> EvidenceGateSpec:
        return self._spec

    def evaluate(self, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceCoverageResult:
        """Evaluate exact candidates without inference, network, or model calls."""

        if any(type(candidate) is not EvidenceCandidate for candidate in candidates):
            raise EvidenceGateContractError("candidates must be exact EvidenceCandidate values")
        fragment_ids = [candidate.source_fragment_id for candidate in candidates]
        if len(set(fragment_ids)) != len(fragment_ids):
            raise EvidenceGateContractError("candidate fragment IDs must be unique")
        ordered = tuple(
            sorted(
                candidates,
                key=lambda item: (item.rank, item.source_fragment_id.encode("ascii")),
            )
        )
        coverages = tuple(
            self._evaluate_requirement(requirement, ordered)
            for requirement in self._spec.requirements
        )
        covered_ids = tuple(
            item.evidence_requirement_id
            for item in coverages
            if item.status is RequirementCoverageStatus.COVERED
        )
        missing_ids = tuple(
            item.evidence_requirement_id
            for item in coverages
            if item.status is RequirementCoverageStatus.MISSING
        )
        matched_groups = tuple(
            sorted(
                {
                    group
                    for item in coverages
                    if item.status is RequirementCoverageStatus.COVERED
                    for group in item.matched_independence_groups
                },
                key=_unicode_sort_key,
            )
        )
        gate_conditions: tuple[str, ...] = ()
        all_requirements_covered = not missing_ids
        if len(matched_groups) < self._spec.min_total_independent_groups:
            gate_conditions = ("insufficient_total_independent_groups",)
            all_requirements_covered = False

        decision = _decision(
            expected=self._spec.expected_answerability,
            covered_count=len(covered_ids),
            all_requirements_covered=all_requirements_covered,
        )
        candidate_set_payload = {
            "schema": "dithyramba.evidence_candidate_set/1.0",
            "items": [item.semantic_payload() for item in ordered],
        }
        return EvidenceCoverageResult(
            query_key=self._spec.query_key,
            spec_id=self._spec.spec_id,
            spec_hash=self._spec.spec_hash,
            candidate_set_hash=canonical_sha256_hex(candidate_set_payload),
            expected_answerability=self._spec.expected_answerability,
            decision=decision,
            covered_requirement_ids=covered_ids,
            missing_requirement_ids=missing_ids,
            matched_independence_groups=matched_groups,
            gate_conditions=gate_conditions,
            requirements=coverages,
        )

    def _evaluate_requirement(
        self,
        requirement: EvidenceRequirement,
        candidates: tuple[EvidenceCandidate, ...],
    ) -> RequirementCoverage:
        assessments = tuple(_assess_candidate(requirement, candidate) for candidate in candidates)
        matched_candidates = tuple(
            candidate
            for candidate, assessment in zip(candidates, assessments, strict=True)
            if not assessment.rejection_codes
        )
        matched_groups = tuple(
            sorted(
                {candidate.independence_group for candidate in matched_candidates},
                key=_unicode_sort_key,
            )
        )
        covered = len(matched_groups) >= requirement.min_independent_groups
        missing_conditions: tuple[str, ...] = ()
        if not matched_candidates:
            missing_conditions = ("no_candidate_satisfies_all_conditions",)
        elif not covered:
            missing_conditions = ("insufficient_independent_groups",)

        closest = tuple(
            sorted(
                (item for item in assessments if item.rejection_codes),
                key=lambda item: (
                    len(item.rejection_codes),
                    item.rank,
                    item.source_fragment_id.encode("ascii"),
                ),
            )[: self._CLOSEST_LIMIT]
        )
        return RequirementCoverage(
            evidence_requirement_id=requirement.evidence_requirement_id,
            requirement_key=requirement.key,
            label=requirement.label,
            status=(
                RequirementCoverageStatus.COVERED if covered else RequirementCoverageStatus.MISSING
            ),
            matched_fragment_ids=tuple(
                candidate.source_fragment_id for candidate in matched_candidates
            ),
            matched_source_ids=tuple(
                sorted(
                    {candidate.source_id for candidate in matched_candidates},
                    key=lambda item: item.encode("ascii"),
                )
            ),
            matched_independence_groups=matched_groups,
            missing_conditions=missing_conditions,
            closest_candidates=closest,
        )


def _assess_candidate(
    requirement: EvidenceRequirement,
    candidate: EvidenceCandidate,
) -> CandidateAssessment:
    rejection_codes: list[str] = []
    if requirement.allowed_source_ids and candidate.source_id not in requirement.allowed_source_ids:
        rejection_codes.append("wrong_source")
    if requirement.source_kinds_any and not _matches_selector(
        candidate.source_kind, requirement.source_kinds_any
    ):
        rejection_codes.append("wrong_source_kind")
    if requirement.source_families_any and not _matches_selector(
        candidate.source_family, requirement.source_families_any
    ):
        rejection_codes.append("wrong_source_family")
    if requirement.authorities_any and not _matches_selector(
        candidate.authority, requirement.authorities_any
    ):
        rejection_codes.append("wrong_authority")

    normalized_text = _match_text(candidate.text)
    for index, group in enumerate(requirement.anchor_groups, start=1):
        if not any(_match_text(anchor) in normalized_text for anchor in group):
            rejection_codes.append(f"missing_anchor_group_{index}")
    candidate_tags = {_match_text(tag) for tag in candidate.evidence_tags}
    for tag in requirement.required_tags:
        if _match_text(tag) not in candidate_tags:
            rejection_codes.append(f"missing_tag_{tag}")
    if any(_match_text(anchor) in normalized_text for anchor in requirement.forbidden_anchors):
        rejection_codes.append("forbidden_anchor_present")
    return CandidateAssessment(
        source_fragment_id=candidate.source_fragment_id,
        source_id=candidate.source_id,
        rank=candidate.rank,
        rejection_codes=tuple(rejection_codes),
    )


def _decision(
    *,
    expected: EvidenceAnswerability,
    covered_count: int,
    all_requirements_covered: bool,
) -> EvidenceGateDecision:
    if expected is EvidenceAnswerability.NOT_IN_CORPUS:
        return (
            EvidenceGateDecision.GAP_CHALLENGED
            if all_requirements_covered
            else EvidenceGateDecision.GAP_PRESERVED
        )
    if all_requirements_covered:
        return EvidenceGateDecision.READY
    if covered_count:
        return EvidenceGateDecision.PARTIAL
    return EvidenceGateDecision.INSUFFICIENT


def _matches_selector(value: str, choices: tuple[str, ...]) -> bool:
    normalized = _match_text(value)
    return any(normalized == _match_text(choice) for choice in choices)


def _match_text(value: str) -> str:
    """Casefold and collapse layout noise without inventing semantic equivalence."""

    normalized = unicodedata.normalize("NFKC", value).casefold().replace("\u00ad", "")
    return _SPACE_PATTERN.sub(" ", normalized).strip()


def _require_reference(value: str, label: str) -> str:
    if type(value) is not str or _REFERENCE_PATTERN.fullmatch(value) is None:
        raise EvidenceGateContractError(f"{label} must be a stable lowercase reference")
    return value


def _require_text(value: str, label: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    if (
        type(value) is not str
        or not normalized.strip()
        or normalized != normalized.strip()
        or "\x00" in normalized
    ):
        raise EvidenceGateContractError(f"{label} must be nonblank, unpadded NFC text")
    return normalized


def _normalized_selector_tuple(value: tuple[str, ...], label: str) -> tuple[str, ...]:
    normalized = tuple(_require_text(item, label) for item in value)
    return _unique_sorted(normalized, label)


def _unique_sorted(value: tuple[str, ...], label: str) -> tuple[str, ...]:
    if len({_match_text(item) for item in value}) != len(value):
        raise EvidenceGateContractError(f"{label} values must be unique")
    return tuple(sorted(value, key=_unicode_sort_key))


def _unicode_sort_key(value: str) -> bytes:
    return unicodedata.normalize("NFC", value).encode("utf-8")


def _canonical_sort_key(value: tuple[str, ...]) -> bytes:
    return canonical_json_bytes(list(value))


def _validated_source_address(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise EvidenceGateContractError("source_address must be a canonical object")
    payload = dict(value)
    if payload.get("schema") != SOURCE_ADDRESS_SCHEMA:
        raise EvidenceGateContractError("source_address schema is invalid")
    kind = payload.get("kind")
    try:
        address: MarkdownSourceAddress | PdfSourceAddress
        if kind == "markdown":
            address = MarkdownSourceAddress(
                heading_path=tuple(_string_list(payload.get("heading_path"), "heading_path")),
                line_start=_integer(payload.get("line_start"), "line_start"),
                line_end=_integer(payload.get("line_end"), "line_end"),
                char_start=_integer(payload.get("char_start"), "char_start"),
                char_end=_integer(payload.get("char_end"), "char_end"),
            )
        elif kind == "pdf":
            address = PdfSourceAddress(
                page=_integer(payload.get("page"), "page"),
                bbox=tuple(_string_list(payload.get("bbox"), "bbox")),  # type: ignore[arg-type]
                char_start=_integer(payload.get("char_start"), "char_start"),
                char_end=_integer(payload.get("char_end"), "char_end"),
            )
        else:
            raise EvidenceGateContractError("source_address kind is invalid")
    except (TypeError, ValueError) as error:
        raise EvidenceGateContractError(f"source_address is invalid: {error}") from error
    canonical = address.payload()
    if set(payload) != set(canonical):
        raise EvidenceGateContractError("source_address keys must match its schema exactly")
    return canonical


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise EvidenceGateContractError(f"{label} must be an integer")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise EvidenceGateContractError(f"{label} must be a string array")
    return value
