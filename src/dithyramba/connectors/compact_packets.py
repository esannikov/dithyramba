"""Provider-free projection of Atlas memory into bounded consumer packets.

The caller supplies every semantic role.  This module only validates those
assignments against an already content-addressed ``CompactMemoryPacket`` and
copies the selected, source-closed provenance into a smaller contract.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import ClassVar, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from dithyramba.atlas.memory_packet import CompactMemoryPacket, CompactMemoryPacketV1_1
from dithyramba.atlas.models import (
    AtlasEvidence,
    AtlasGap,
    AtlasSource,
    EvidenceRole,
    SourceKind,
    VerificationState,
    VoiceKind,
)
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

CONNECTOR_PACKET_SCHEMA = "dithyramba.compact_connector_packet/1.0"
CONNECTOR_PACKET_PROFILE = "caller_assigned_source_closed_projection_v1"

_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_TASK_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{1,95}$"
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_SOURCE_PACKET_ID_PATTERN = r"^memory_packet_[0-9a-f]{32}$"
_CONNECTOR_PACKET_ID_PATTERN = r"^connector_packet_[0-9a-f]{32}$"


class ConnectorPacketError(ValueError):
    """A projection input failed a closed connector boundary."""


class _ConnectorModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ConnectorRole(StrEnum):
    """Caller-assigned use of one exact source-packet item."""

    RECIPE = "recipe"
    PRECONDITION = "precondition"
    VALIDATION = "validation"
    CONTRAINDICATION = "contraindication"
    GAP = "gap"


_ROLE_ORDER = tuple(ConnectorRole)
_ROLE_RANK = {role: rank for rank, role in enumerate(_ROLE_ORDER)}


class ConnectorPacketStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class ConnectorPacketBudget(_ConnectorModel):
    """Hard per-packet maxima; projection never truncates to fit them."""

    max_evidence: int = Field(default=16, ge=0, le=64)
    max_gaps: int = Field(default=8, ge=0, le=32)
    max_sources: int = Field(default=16, ge=0, le=64)
    max_excerpt_chars: int = Field(default=16_000, ge=0, le=64_000)


class ConnectorRoleAssignment(_ConnectorModel):
    """One explicit caller decision for one connector role."""

    role: ConnectorRole
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=64)
    gap_ids: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_assignment(self) -> Self:
        _require_unique(self.evidence_ids, "assignment evidence_ids")
        _require_unique(self.gap_ids, "assignment gap_ids")
        if self.role is ConnectorRole.GAP:
            if self.evidence_ids or not self.gap_ids:
                raise ValueError("gap assignments require only one or more gap_ids")
        elif self.gap_ids or not self.evidence_ids:
            raise ValueError("non-gap assignments require only one or more evidence_ids")
        return self


class ConnectorPacketRequest(_ConnectorModel):
    """Consumer identity, role requirements, caller assignments, and budgets."""

    consumer_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_TASK_ID_PATTERN)
    required_roles: tuple[ConnectorRole, ...] = Field(
        default=_ROLE_ORDER,
        min_length=1,
        max_length=len(_ROLE_ORDER),
    )
    assignments: tuple[ConnectorRoleAssignment, ...] = Field(
        default=(),
        max_length=len(_ROLE_ORDER),
    )
    budget: ConnectorPacketBudget = Field(default_factory=ConnectorPacketBudget)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_unique(self.required_roles, "required_roles")
        assignment_roles = tuple(item.role for item in self.assignments)
        _require_unique(assignment_roles, "assignment roles")

        evidence_owner: dict[str, ConnectorRole] = {}
        for assignment in self.assignments:
            for evidence_id in assignment.evidence_ids:
                owner = evidence_owner.setdefault(evidence_id, assignment.role)
                if owner is not assignment.role:
                    raise ValueError(
                        f"conflicting evidence assignment for {evidence_id}: "
                        f"{owner.value} and {assignment.role.value}"
                    )
        return self


class ConnectorSourceAddress(_ConnectorModel):
    kind: str = Field(min_length=1, max_length=80)
    heading_path: tuple[str, ...] = Field(default=(), max_length=32)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered_ranges(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("source address line range is reversed")
        if self.char_end < self.char_start:
            raise ValueError("source address character range is reversed")
        return self


class ConnectorSource(_ConnectorModel):
    """Exact source descriptor copied from the source packet."""

    source_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    creator: str = Field(min_length=1, max_length=300)
    source_kind: SourceKind
    voice_kind: VoiceKind
    date_label: str | None = Field(default=None, max_length=120)
    publisher_or_archive: str | None = Field(default=None, max_length=300)
    independence_group: str = Field(pattern=_ID_PATTERN)
    verification_state: VerificationState
    original_url: str | None = Field(default=None, max_length=2_000)
    artifact_path: str | None = Field(default=None, max_length=1_000)
    rights_note: str = Field(min_length=1, max_length=1_000)

    @field_validator("original_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("original_url must be an absolute HTTP(S) URL")
        return value

    @field_validator("artifact_path")
    @classmethod
    def validate_artifact_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact_path must be a case-relative path")
        return value


class ConnectorEvidence(_ConnectorModel):
    """Exact evidence and provenance selected for one caller role."""

    connector_role: ConnectorRole
    evidence_id: str = Field(pattern=_ID_PATTERN)
    source_id: str = Field(pattern=_ID_PATTERN)
    locator: str = Field(min_length=1, max_length=500)
    excerpt: str | None = Field(default=None, max_length=1_000)
    evidence_role: EvidenceRole
    asserting_voice: str = Field(min_length=1, max_length=300)
    limitation: str | None = Field(default=None, max_length=2_000)
    source_fragment_id: str | None = Field(
        default=None,
        pattern=r"^fragment_[a-z0-9_]+$",
    )
    fragment_text_sha256: str | None = Field(default=None, pattern=_HASH_PATTERN)
    source_address: ConnectorSourceAddress | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.connector_role is ConnectorRole.GAP:
            raise ValueError("gap is not an evidence connector role")
        exact_binding = (
            self.source_fragment_id,
            self.fragment_text_sha256,
            self.source_address,
        )
        if any(item is not None for item in exact_binding) and any(
            item is None for item in exact_binding
        ):
            raise ValueError("exact evidence binding must be complete")
        return self


class ConnectorGap(_ConnectorModel):
    """Selected source-packet gap; its role is always ``gap``."""

    connector_role: ConnectorRole
    gap_id: str = Field(pattern=_ID_PATTERN)
    label: str = Field(min_length=1, max_length=500)
    why_it_matters: str = Field(min_length=1, max_length=2_000)
    next_evidence: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_gap_role(self) -> Self:
        if self.connector_role is not ConnectorRole.GAP:
            raise ValueError("selected gaps must use the gap connector role")
        return self


class CompactConnectorPacket(_ConnectorModel):
    """Versioned, content-addressed, source-closed consumer projection."""

    SCHEMA: ClassVar[str] = CONNECTOR_PACKET_SCHEMA

    schema_id: str
    connector_packet_id: str = Field(pattern=_CONNECTOR_PACKET_ID_PATTERN)
    connector_packet_hash: str = Field(pattern=_HASH_PATTERN)
    projection_profile: str
    source_packet_schema_id: str
    source_packet_id: str = Field(pattern=_SOURCE_PACKET_ID_PATTERN)
    source_packet_hash: str = Field(pattern=_HASH_PATTERN)
    consumer_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_TASK_ID_PATTERN)
    status: ConnectorPacketStatus
    required_roles: tuple[ConnectorRole, ...] = Field(min_length=1, max_length=5)
    missing_roles: tuple[ConnectorRole, ...] = Field(max_length=5)
    budget: ConnectorPacketBudget
    sources: tuple[ConnectorSource, ...] = Field(max_length=64)
    evidence: tuple[ConnectorEvidence, ...] = Field(max_length=64)
    gaps: tuple[ConnectorGap, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def validate_packet(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if self.projection_profile != CONNECTOR_PACKET_PROFILE:
            raise ValueError(f"projection_profile must be {CONNECTOR_PACKET_PROFILE}")
        if self.source_packet_schema_id not in {
            CompactMemoryPacket.SCHEMA,
            CompactMemoryPacketV1_1.SCHEMA,
        }:
            raise ValueError("unsupported source compact-memory-packet schema")

        _require_unique(self.required_roles, "required_roles")
        _require_unique(self.missing_roles, "missing_roles")
        if self.required_roles != _canonical_roles(self.required_roles):
            raise ValueError("required_roles must use canonical role order")
        if self.missing_roles != _canonical_roles(self.missing_roles):
            raise ValueError("missing_roles must use canonical role order")

        source_ids = tuple(item.source_id for item in self.sources)
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        gap_ids = tuple(item.gap_id for item in self.gaps)
        _require_unique(source_ids, "source IDs")
        _require_unique(evidence_ids, "evidence IDs")
        _require_unique(gap_ids, "gap IDs")
        if source_ids != tuple(sorted(source_ids)):
            raise ValueError("sources must use canonical source_id order")
        evidence_order = tuple(
            (_ROLE_RANK[item.connector_role], item.evidence_id) for item in self.evidence
        )
        if evidence_order != tuple(sorted(evidence_order)):
            raise ValueError("evidence must use canonical role and evidence_id order")
        if gap_ids != tuple(sorted(gap_ids)):
            raise ValueError("gaps must use canonical gap_id order")

        selected_roles = {item.connector_role for item in self.evidence}
        if self.gaps:
            selected_roles.add(ConnectorRole.GAP)
        expected_missing = tuple(role for role in self.required_roles if role not in selected_roles)
        if self.missing_roles != expected_missing:
            raise ValueError("missing_roles must exactly cover unassigned required roles")
        expected_status = (
            ConnectorPacketStatus.PARTIAL if expected_missing else ConnectorPacketStatus.COMPLETE
        )
        if self.status is not expected_status:
            raise ValueError("status must reflect required-role completeness")

        required_source_ids = {item.source_id for item in self.evidence}
        if set(source_ids) != required_source_ids:
            raise ValueError("sources must exactly close selected evidence provenance")
        self._validate_budget()

        payload = self.semantic_payload()
        if self.connector_packet_id != canonical_content_id("connector_packet", payload):
            raise ValueError("connector_packet_id does not match canonical packet content")
        if self.connector_packet_hash != canonical_sha256_hex(payload):
            raise ValueError("connector_packet_hash does not match canonical packet content")
        return self

    def _validate_budget(self) -> None:
        counts = (
            (len(self.evidence), self.budget.max_evidence, "evidence"),
            (len(self.gaps), self.budget.max_gaps, "gap"),
            (len(self.sources), self.budget.max_sources, "source"),
        )
        for actual, maximum, label in counts:
            if actual > maximum:
                raise ValueError(f"selected {label} count exceeds connector budget")
        excerpt_chars = sum(len(item.excerpt or "") for item in self.evidence)
        if excerpt_chars > self.budget.max_excerpt_chars:
            raise ValueError("selected evidence excerpts exceed connector character budget")

    def semantic_payload(self) -> dict[str, object]:
        """Return every immutable field covered by the packet ID and hash."""

        return {
            "schema_id": self.schema_id,
            "projection_profile": self.projection_profile,
            "source_packet_schema_id": self.source_packet_schema_id,
            "source_packet_id": self.source_packet_id,
            "source_packet_hash": self.source_packet_hash,
            "consumer_id": self.consumer_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "required_roles": [role.value for role in self.required_roles],
            "missing_roles": [role.value for role in self.missing_roles],
            "budget": self.budget.model_dump(mode="json"),
            "sources": [item.model_dump(mode="json") for item in self.sources],
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
            "gaps": [item.model_dump(mode="json") for item in self.gaps],
        }


def project_compact_memory_packet(
    source_packet: CompactMemoryPacket,
    request: ConnectorPacketRequest,
) -> CompactConnectorPacket:
    """Project caller-assigned IDs without inference, I/O, providers, or truncation."""

    packet = _validated_source_packet(source_packet)
    validated_request = _validated_request(request)
    evidence_by_id = {item.evidence_id: item for item in packet.evidence}
    gap_by_id = {item.gap_id: item for item in packet.gaps}
    source_by_id = {item.source_id: item for item in packet.sources}

    missing_evidence_ids = sorted(
        {
            evidence_id
            for assignment in validated_request.assignments
            for evidence_id in assignment.evidence_ids
            if evidence_id not in evidence_by_id
        }
    )
    missing_gap_ids = sorted(
        {
            gap_id
            for assignment in validated_request.assignments
            for gap_id in assignment.gap_ids
            if gap_id not in gap_by_id
        }
    )
    if missing_evidence_ids:
        raise ConnectorPacketError(
            "assigned evidence IDs are absent from source packet: "
            + ", ".join(missing_evidence_ids)
        )
    if missing_gap_ids:
        raise ConnectorPacketError(
            "assigned gap IDs are absent from source packet: " + ", ".join(missing_gap_ids)
        )

    selected_evidence = tuple(
        _project_evidence(assignment.role, evidence_by_id[evidence_id])
        for assignment in sorted(
            validated_request.assignments,
            key=lambda item: _ROLE_RANK[item.role],
        )
        for evidence_id in sorted(assignment.evidence_ids)
    )
    selected_gaps = tuple(
        _project_gap(gap_by_id[gap_id])
        for assignment in validated_request.assignments
        if assignment.role is ConnectorRole.GAP
        for gap_id in sorted(assignment.gap_ids)
    )
    selected_source_ids = sorted({item.source_id for item in selected_evidence})
    selected_sources = tuple(
        _project_source(source_by_id[source_id]) for source_id in selected_source_ids
    )
    _enforce_budget(selected_evidence, selected_gaps, selected_sources, validated_request.budget)

    required_roles = _canonical_roles(validated_request.required_roles)
    selected_roles = {item.connector_role for item in selected_evidence}
    if selected_gaps:
        selected_roles.add(ConnectorRole.GAP)
    missing_roles = tuple(role for role in required_roles if role not in selected_roles)
    status = ConnectorPacketStatus.PARTIAL if missing_roles else ConnectorPacketStatus.COMPLETE
    payload = _semantic_payload(
        source_packet=packet,
        request=validated_request,
        status=status,
        required_roles=required_roles,
        missing_roles=missing_roles,
        sources=selected_sources,
        evidence=selected_evidence,
        gaps=selected_gaps,
    )
    return CompactConnectorPacket(
        schema_id=CONNECTOR_PACKET_SCHEMA,
        connector_packet_id=canonical_content_id("connector_packet", payload),
        connector_packet_hash=canonical_sha256_hex(payload),
        projection_profile=CONNECTOR_PACKET_PROFILE,
        source_packet_schema_id=packet.schema_id,
        source_packet_id=packet.packet_id,
        source_packet_hash=packet.packet_hash,
        consumer_id=validated_request.consumer_id,
        task_id=validated_request.task_id,
        status=status,
        required_roles=required_roles,
        missing_roles=missing_roles,
        budget=validated_request.budget,
        sources=selected_sources,
        evidence=selected_evidence,
        gaps=selected_gaps,
    )


def _validated_source_packet(source_packet: CompactMemoryPacket) -> CompactMemoryPacket:
    if not isinstance(source_packet, CompactMemoryPacket):
        raise ConnectorPacketError("source_packet must be a CompactMemoryPacket 1.0 or 1.1")
    model: type[CompactMemoryPacket] | type[CompactMemoryPacketV1_1]
    if source_packet.schema_id == CompactMemoryPacket.SCHEMA:
        model = CompactMemoryPacket
    elif source_packet.schema_id == CompactMemoryPacketV1_1.SCHEMA:
        model = CompactMemoryPacketV1_1
    else:
        raise ConnectorPacketError("source_packet uses an unsupported schema")
    try:
        return model.model_validate(source_packet.model_dump(mode="python"), strict=True)
    except ValidationError as exc:
        raise ConnectorPacketError(
            "source_packet failed strict content-address and closure validation"
        ) from exc


def _validated_request(request: ConnectorPacketRequest) -> ConnectorPacketRequest:
    if not isinstance(request, ConnectorPacketRequest):
        raise ConnectorPacketError("request must be a ConnectorPacketRequest")
    try:
        return ConnectorPacketRequest.model_validate(
            request.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise ConnectorPacketError("request failed strict assignment validation") from exc


def _project_source(source: AtlasSource) -> ConnectorSource:
    try:
        payload = source.model_dump(mode="python")
        return ConnectorSource.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise ConnectorPacketError("selected source could not be projected exactly") from exc


def _project_evidence(role: ConnectorRole, evidence: AtlasEvidence) -> ConnectorEvidence:
    try:
        address = evidence.source_address
        projected_address = (
            None
            if address is None
            else ConnectorSourceAddress.model_validate(
                address.model_dump(mode="python"),
                strict=True,
            )
        )
        return ConnectorEvidence(
            connector_role=role,
            evidence_id=evidence.evidence_id,
            source_id=evidence.source_id,
            locator=evidence.locator,
            excerpt=evidence.excerpt,
            evidence_role=evidence.role,
            asserting_voice=evidence.asserting_voice,
            limitation=evidence.limitation,
            source_fragment_id=evidence.source_fragment_id,
            fragment_text_sha256=evidence.fragment_text_sha256,
            source_address=projected_address,
        )
    except ValidationError as exc:
        raise ConnectorPacketError("selected evidence could not be projected exactly") from exc


def _project_gap(gap: AtlasGap) -> ConnectorGap:
    try:
        return ConnectorGap(
            connector_role=ConnectorRole.GAP,
            gap_id=gap.gap_id,
            label=gap.label,
            why_it_matters=gap.why_it_matters,
            next_evidence=gap.next_evidence,
        )
    except ValidationError as exc:
        raise ConnectorPacketError("selected gap could not be projected exactly") from exc


def _enforce_budget(
    evidence: tuple[ConnectorEvidence, ...],
    gaps: tuple[ConnectorGap, ...],
    sources: tuple[ConnectorSource, ...],
    budget: ConnectorPacketBudget,
) -> None:
    counts = (
        (len(evidence), budget.max_evidence, "evidence"),
        (len(gaps), budget.max_gaps, "gap"),
        (len(sources), budget.max_sources, "source"),
    )
    for actual, maximum, label in counts:
        if actual > maximum:
            raise ConnectorPacketError(f"selected {label} count exceeds connector budget")
    excerpt_chars = sum(len(item.excerpt or "") for item in evidence)
    if excerpt_chars > budget.max_excerpt_chars:
        raise ConnectorPacketError("selected evidence excerpts exceed connector character budget")


def _semantic_payload(
    *,
    source_packet: CompactMemoryPacket,
    request: ConnectorPacketRequest,
    status: ConnectorPacketStatus,
    required_roles: tuple[ConnectorRole, ...],
    missing_roles: tuple[ConnectorRole, ...],
    sources: tuple[ConnectorSource, ...],
    evidence: tuple[ConnectorEvidence, ...],
    gaps: tuple[ConnectorGap, ...],
) -> dict[str, object]:
    return {
        "schema_id": CONNECTOR_PACKET_SCHEMA,
        "projection_profile": CONNECTOR_PACKET_PROFILE,
        "source_packet_schema_id": source_packet.schema_id,
        "source_packet_id": source_packet.packet_id,
        "source_packet_hash": source_packet.packet_hash,
        "consumer_id": request.consumer_id,
        "task_id": request.task_id,
        "status": status.value,
        "required_roles": [role.value for role in required_roles],
        "missing_roles": [role.value for role in missing_roles],
        "budget": request.budget.model_dump(mode="json"),
        "sources": [item.model_dump(mode="json") for item in sources],
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "gaps": [item.model_dump(mode="json") for item in gaps],
    }


def _canonical_roles(roles: tuple[ConnectorRole, ...]) -> tuple[ConnectorRole, ...]:
    selected = set(roles)
    return tuple(role for role in _ROLE_ORDER if role in selected)


def _require_unique(values: tuple[object, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


__all__ = [
    "CONNECTOR_PACKET_PROFILE",
    "CONNECTOR_PACKET_SCHEMA",
    "CompactConnectorPacket",
    "ConnectorEvidence",
    "ConnectorGap",
    "ConnectorPacketBudget",
    "ConnectorPacketError",
    "ConnectorPacketRequest",
    "ConnectorPacketStatus",
    "ConnectorRole",
    "ConnectorRoleAssignment",
    "ConnectorSource",
    "ConnectorSourceAddress",
    "project_compact_memory_packet",
]
