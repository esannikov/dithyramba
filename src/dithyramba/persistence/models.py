"""Immutable records returned by the Library-scoped repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dithyramba.access import AccessPolicySnapshot, CompiledAccess, RequestScope
from dithyramba.collections import CollectionConfig
from dithyramba.library import LibraryConfig, LibraryPaths
from dithyramba.provenance import ProcessingRunRecord

# P2 provenance records live in ``dithyramba.provenance``. They are re-exported
# by the persistence package, while this module remains the P1 record home.


@dataclass(frozen=True, slots=True)
class LibraryRecord:
    """One verified Library identity and its physical installation."""

    config: LibraryConfig
    logical_identity_hash: str
    created_at: str
    paths: LibraryPaths


@dataclass(frozen=True, slots=True)
class CollectionRecord:
    """One logical Collection with stable persisted root identifiers."""

    config: CollectionConfig
    collection_root_ids: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class AccessPolicyRecord:
    """One named, immutable AccessPolicy snapshot."""

    name: str
    snapshot: AccessPolicySnapshot
    created_at: str


@dataclass(frozen=True, slots=True)
class LibraryCounts:
    """Compact persisted-object counts for one physically isolated Library."""

    collections: int
    access_policies: int
    sources: int
    source_versions: int
    source_fragments: int
    collection_memberships: int
    corpus_snapshots: int
    snapshot_members: int
    evidence_packets: int
    research_sessions: int
    processing_runs: int


@dataclass(frozen=True, slots=True)
class CorpusSnapshotSummary:
    """Read-only snapshot identity without expanding its complete manifest."""

    corpus_snapshot_id: str
    manifest_hash: str
    collection_ids: tuple[str, ...]
    member_count: int
    created_at: str


@dataclass(frozen=True, slots=True)
class LibraryDescription:
    """One reconstructable operator overview with no source text."""

    library: LibraryRecord
    counts: LibraryCounts
    collections: tuple[CollectionRecord, ...]
    access_policies: tuple[AccessPolicyRecord, ...]
    snapshots: tuple[CorpusSnapshotSummary, ...]
    recent_processing_runs: tuple[ProcessingRunRecord, ...]


@dataclass(frozen=True, slots=True)
class OutboxEvent:
    """One committed domain event awaiting or having completed export."""

    event_id: str
    event_type: str
    aggregate_type: str
    aggregate_id: str
    payload: dict[str, object]
    payload_hash: str
    occurred_at: str
    delivered_at: str | None

    def envelope(self) -> dict[str, object]:
        """Return the canonical JSONL envelope written to the audit surface."""

        return {
            "schema": "dithyramba.event/1.0",
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_type": self.aggregate_type,
            "aggregate_id": self.aggregate_id,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "occurred_at": self.occurred_at,
        }


@dataclass(frozen=True, slots=True)
class OutboxExportResult:
    """Summary of one idempotent outbox delivery pass."""

    appended_count: int
    deduplicated_count: int
    delivered_count: int
    last_event_id: str | None


@dataclass(frozen=True, slots=True)
class BackupRecord:
    """One verified SQLite backup promoted into a Library's backup directory."""

    path: Path
    created_at: str
    sha256: str
    schema_version: int


@dataclass(frozen=True, slots=True)
class AuthorizedRead:
    """Ephemeral repository-issued capability for one DB-derived permitted set.

    Public token material alone cannot create a usable authorization: the
    repository retains both the exact object identity and an independent
    issuance fingerprint, then revalidates policy and snapshot metadata
    immediately before reading fragment text.
    """

    authorization_id: str
    repository_instance_id: str
    access_policy_id: str
    scope: RequestScope
    compiled: CompiledAccess


@dataclass(frozen=True, slots=True)
class SourceFragmentText:
    """A text-bearing SourceFragment returned only after authorization."""

    source_fragment_id: str
    source_version_id: str
    source_id: str
    ordinal: int
    fragment_kind: str
    text: str
    text_sha256: str
    source_address_json: str
