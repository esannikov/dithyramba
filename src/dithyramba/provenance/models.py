"""Immutable provenance, coverage, and ingest result records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OUTCOME_PAYLOAD_KEYS = frozenset(
    {
        "collection_root_id",
        "relative_path",
        "terminal_outcome",
        "disposition",
        "failure_code",
        "retryable",
        "source_id",
        "source_version_id",
        "source_family_id",
        "root_source_id",
        "fragment_count",
        "infrastructure_failure",
    }
)


class SourceFamilyRole(StrEnum):
    """Frozen VS0 SourceFamily membership roles."""

    ROOT = "root"
    DERIVATIVE = "derivative"
    DUPLICATE = "duplicate"


class IngestDisposition(StrEnum):
    """Observable content-history result for one successfully parsed input."""

    ADDED = "added"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    REVERTED = "reverted"


class TerminalInputOutcome(StrEnum):
    """Coverage classification for exactly one declared ingest input."""

    PROCESSED = "processed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ProcessingRunStatus(StrEnum):
    """ProcessingRun states returned by the public service boundary."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PromotedBlob:
    """One physically verified content-addressed blob ready for a DB reference."""

    content_sha256: str
    byte_size: int
    relative_path: str
    path: Path
    created: bool

    def __post_init__(self) -> None:
        _require_hash(self.content_sha256, "content_sha256")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("byte_size must be a non-negative integer")
        expected = f"{self.content_sha256[:2]}/{self.content_sha256[2:]}"
        if self.relative_path != expected:
            raise ValueError("blob relative_path must be derived from content_sha256")
        if not self.path.is_absolute() or self.path.as_posix().endswith("/../"):
            raise ValueError("blob path must be absolute")
        if type(self.created) is not bool:
            raise TypeError("created must be bool")


@dataclass(frozen=True, slots=True)
class BlobRecord:
    """One content-addressed Blob row and its verified local path."""

    content_sha256: str
    byte_size: int
    relative_path: str
    created_at: str
    path: Path


@dataclass(frozen=True, slots=True)
class CollectionMembershipRecord:
    """A Source's membership inside one Collection."""

    collection_id: str
    state: str
    created_at: str


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One immutable Source identity with its root-source lineage."""

    source_id: str
    library_id: str
    canonical_uri: str
    media_type: str
    title: str | None
    source_family_id: str
    family_role: SourceFamilyRole
    root_source_id: str
    current_source_version_id: str
    created_at: str
    memberships: tuple[CollectionMembershipRecord, ...]


@dataclass(frozen=True, slots=True)
class SourceVersionRecord:
    """One immutable, byte-addressed observation of a Source."""

    source_version_id: str
    source_id: str
    version_number: int
    content_sha256: str
    byte_size: int
    observed_at: str
    source_modified_at: str | None
    parser_profile: str
    parse_status: str
    failure_code: str | None
    is_current: bool


@dataclass(frozen=True, slots=True)
class SourceFragmentRecord:
    """Metadata for one exact SourceFragment; text remains behind the access gate."""

    source_fragment_id: str
    source_version_id: str
    ordinal: int
    fragment_kind: str
    text_sha256: str
    source_address_json: str
    address_hash: str


@dataclass(frozen=True, slots=True)
class PersistedSourceOutcome:
    """Result of one serialized processed-source transaction."""

    disposition: IngestDisposition
    source_id: str
    source_version_id: str
    source_family_id: str
    root_source_id: str
    family_role: SourceFamilyRole
    fragment_count: int


@dataclass(frozen=True, slots=True)
class IngestInputOutcome:
    """One terminal, canonical coverage classification for an input."""

    collection_root_id: str
    relative_path: str | None
    terminal_outcome: TerminalInputOutcome
    disposition: IngestDisposition | None = None
    failure_code: str | None = None
    retryable: bool = False
    source_id: str | None = None
    source_version_id: str | None = None
    source_family_id: str | None = None
    root_source_id: str | None = None
    fragment_count: int = 0
    infrastructure_failure: bool = False

    def __post_init__(self) -> None:
        if type(self.collection_root_id) is not str or not self.collection_root_id:
            raise ValueError("collection_root_id must be non-empty")
        if self.relative_path is not None and (
            type(self.relative_path) is not str or not self.relative_path
        ):
            raise ValueError("relative_path must be non-empty when present")
        if not isinstance(self.terminal_outcome, TerminalInputOutcome):
            raise TypeError("terminal_outcome must be TerminalInputOutcome")
        if self.disposition is not None and not isinstance(self.disposition, IngestDisposition):
            raise TypeError("disposition must be IngestDisposition when present")
        for label, value in (
            ("failure_code", self.failure_code),
            ("source_id", self.source_id),
            ("source_version_id", self.source_version_id),
            ("source_family_id", self.source_family_id),
            ("root_source_id", self.root_source_id),
        ):
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"{label} must be non-empty text when present")
        if type(self.retryable) is not bool or type(self.infrastructure_failure) is not bool:
            raise TypeError("retryable and infrastructure_failure must be bool")
        if type(self.fragment_count) is not int or self.fragment_count < 0:
            raise ValueError("fragment_count must be a non-negative integer")
        if self.terminal_outcome is TerminalInputOutcome.PROCESSED:
            if self.disposition is None or self.failure_code is not None:
                raise ValueError("processed outcome requires a disposition and no failure code")
            if self.source_id is None or self.source_version_id is None:
                raise ValueError("processed outcome requires Source and SourceVersion IDs")
            if self.source_family_id is None or self.root_source_id is None:
                raise ValueError("processed outcome requires root-source lineage")
            if self.infrastructure_failure:
                raise ValueError("processed outcome cannot be an infrastructure failure")
        else:
            if self.disposition is not None or self.failure_code is None:
                raise ValueError("skipped/failed outcome requires only a failure code")
            if (
                any(
                    value is not None
                    for value in (
                        self.source_id,
                        self.source_version_id,
                        self.source_family_id,
                        self.root_source_id,
                    )
                )
                or self.fragment_count
            ):
                raise ValueError("skipped/failed outcome cannot claim persisted provenance")
            if (
                self.infrastructure_failure
                and self.terminal_outcome is not TerminalInputOutcome.FAILED
            ):
                raise ValueError("infrastructure failure must be a failed outcome")

    def payload(self) -> dict[str, object]:
        """Return the canonical JSON value persisted inside CoverageReport."""

        return {
            "collection_root_id": self.collection_root_id,
            "relative_path": self.relative_path,
            "terminal_outcome": self.terminal_outcome.value,
            "disposition": None if self.disposition is None else self.disposition.value,
            "failure_code": self.failure_code,
            "retryable": self.retryable,
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "source_family_id": self.source_family_id,
            "root_source_id": self.root_source_id,
            "fragment_count": self.fragment_count,
            "infrastructure_failure": self.infrastructure_failure,
        }

    @classmethod
    def from_payload(cls, payload: object) -> IngestInputOutcome:
        """Reconstruct one outcome only from its exact canonical JSON shape."""

        if type(payload) is not dict:
            raise TypeError("ingest outcome payload must be an object")
        values = cast(dict[object, object], payload)
        if set(values) != _OUTCOME_PAYLOAD_KEYS:
            raise ValueError("ingest outcome payload keys must match the frozen schema exactly")
        terminal_raw = values["terminal_outcome"]
        if type(terminal_raw) is not str:
            raise TypeError("terminal_outcome payload value must be text")
        try:
            terminal = TerminalInputOutcome(terminal_raw)
        except ValueError as exc:
            raise ValueError("terminal_outcome payload value is invalid") from exc
        disposition_raw = values["disposition"]
        if disposition_raw is None:
            disposition = None
        elif type(disposition_raw) is str:
            try:
                disposition = IngestDisposition(disposition_raw)
            except ValueError as exc:
                raise ValueError("disposition payload value is invalid") from exc
        else:
            raise TypeError("disposition payload value must be text or null")
        return cls(
            collection_root_id=cast(str, values["collection_root_id"]),
            relative_path=cast(str | None, values["relative_path"]),
            terminal_outcome=terminal,
            disposition=disposition,
            failure_code=cast(str | None, values["failure_code"]),
            retryable=cast(bool, values["retryable"]),
            source_id=cast(str | None, values["source_id"]),
            source_version_id=cast(str | None, values["source_version_id"]),
            source_family_id=cast(str | None, values["source_family_id"]),
            root_source_id=cast(str | None, values["root_source_id"]),
            fragment_count=cast(int, values["fragment_count"]),
            infrastructure_failure=cast(bool, values["infrastructure_failure"]),
        )


@dataclass(frozen=True, slots=True)
class OmissionRecord:
    """One counted or redacted omission attached to CoverageReport."""

    omission_id: str
    coverage_report_id: str
    category: str
    disclosure: str
    count: int | None
    reason_code: str


@dataclass(frozen=True, slots=True)
class CoverageReportRecord:
    """One canonical ingest or recall coverage report."""

    coverage_report_id: str
    processing_run_id: str
    stage: str
    processed_count: int
    skipped_count: int
    failed_count: int
    policy_omission_present: bool
    report_json: str
    report_hash: str
    omissions: tuple[OmissionRecord, ...]


@dataclass(frozen=True, slots=True)
class ProcessingRunRecord:
    """One auditable ProcessingRun lifecycle snapshot."""

    processing_run_id: str
    kind: str
    status: ProcessingRunStatus
    code_version: str
    profile_version: str
    started_at: str
    finished_at: str | None
    error_code: str | None
    output_hash: str | None


@dataclass(frozen=True, slots=True)
class IngestBatchResult:
    """Complete immutable result suitable for CLI or API serialization."""

    run: ProcessingRunRecord
    coverage: CoverageReportRecord
    outcomes: tuple[IngestInputOutcome, ...]

    @property
    def succeeded(self) -> bool:
        return self.run.status is ProcessingRunStatus.SUCCEEDED


def _require_hash(value: str, label: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
