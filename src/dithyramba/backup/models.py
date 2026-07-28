"""Canonical complete-backup manifest contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_[a-z0-9]+(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True, order=True)
class BackupFileEntry:
    """One private regular file covered by a BackupBundle manifest."""

    path: str
    byte_size: int
    sha256: str
    mode: int = 0o600

    def __post_init__(self) -> None:
        value = PurePosixPath(self.path)
        if value.is_absolute() or ".." in value.parts or value.as_posix() != self.path:
            raise ValueError("backup file path must be canonical and relative")
        if not value.parts or any(part in ("", ".") for part in value.parts):
            raise ValueError("backup file path must contain canonical components")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("backup file byte_size must be non-negative")
        if type(self.sha256) is not str or _HASH_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("backup file sha256 must be lowercase hex")
        if type(self.mode) is not int or self.mode != 0o600:
            raise ValueError("backup file mode must be 0600")

    def payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True, order=True)
class BackupMigrationEntry:
    version: int
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 1:
            raise ValueError("migration version must be positive")
        if type(self.name) is not str or not re.fullmatch(r"[0-9]{4}_[a-z0-9_]+\.sql", self.name):
            raise ValueError("migration name is invalid")
        if type(self.sha256) is not str or _HASH_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("migration sha256 must be lowercase hex")

    def payload(self) -> dict[str, object]:
        return {"version": self.version, "name": self.name, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class BackupBundleManifest:
    """Exact manifest for a single-Library, portable BackupBundle."""

    SCHEMA = "dithyramba.backup_bundle/1.0"

    backup_bundle_id: str
    library_id: str
    library_name: str
    logical_identity_hash: str
    created_at: str
    schema_version: int
    schema_fingerprint: str
    migrations: tuple[BackupMigrationEntry, ...]
    row_counts: tuple[tuple[str, int], ...]
    last_event_id: str | None
    event_count: int
    files: tuple[BackupFileEntry, ...]

    def __post_init__(self) -> None:
        if not self.backup_bundle_id.startswith("backup_bundle_") or not _ID_PATTERN.fullmatch(
            self.backup_bundle_id
        ):
            raise ValueError("backup_bundle_id is invalid")
        if not self.library_id.startswith("library_") or not _ID_PATTERN.fullmatch(self.library_id):
            raise ValueError("library_id is invalid")
        if type(self.library_name) is not str or not self.library_name.strip():
            raise ValueError("library_name must be nonblank")
        if _HASH_PATTERN.fullmatch(self.logical_identity_hash) is None:
            raise ValueError("logical identity hash is invalid")
        if (
            type(self.created_at) is not str
            or _TIMESTAMP_PATTERN.fullmatch(self.created_at) is None
        ):
            raise ValueError("created_at must be canonical UTC text")
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if _HASH_PATTERN.fullmatch(self.schema_fingerprint) is None:
            raise ValueError("schema fingerprint is invalid")
        if tuple(item.version for item in self.migrations) != tuple(
            range(1, self.schema_version + 1)
        ):
            raise ValueError("migration manifest must be contiguous through schema_version")
        table_names = tuple(name for name, _count in self.row_counts)
        if table_names != tuple(sorted(set(table_names))) or any(
            type(name) is not str
            or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
            or type(count) is not int
            or count < 0
            for name, count in self.row_counts
        ):
            raise ValueError("row_counts must be a sorted unique non-negative mapping")
        if self.last_event_id is not None and (
            not self.last_event_id.startswith("event_")
            or not _ID_PATTERN.fullmatch(self.last_event_id)
        ):
            raise ValueError("last_event_id is invalid")
        if type(self.event_count) is not int or self.event_count < 0:
            raise ValueError("event_count must be non-negative")
        if (self.event_count == 0) != (self.last_event_id is None):
            raise ValueError("event_count and last_event_id disagree")
        file_paths = tuple(item.path for item in self.files)
        if file_paths != tuple(sorted(set(file_paths))):
            raise ValueError("backup files must be uniquely sorted by path")
        if "memory.sqlite3" not in file_paths or "events.jsonl" not in file_paths:
            raise ValueError("backup bundle requires database and event files")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "backup_bundle_id": self.backup_bundle_id,
            "library_id": self.library_id,
            "library_name": self.library_name,
            "logical_identity_hash": self.logical_identity_hash,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "schema_fingerprint": self.schema_fingerprint,
            "migrations": [item.payload() for item in self.migrations],
            "row_counts": {name: count for name, count in self.row_counts},
            "last_event_id": self.last_event_id,
            "event_count": self.event_count,
            "files": [item.payload() for item in self.files],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256_hex(self.payload())


@dataclass(frozen=True, slots=True)
class BackupBundleRecord:
    path: Path
    manifest: BackupBundleManifest

    @property
    def manifest_hash(self) -> str:
        return self.manifest.manifest_hash

    @property
    def is_current_schema(self) -> bool:
        """Return whether this verified bundle is at the installed code's head.

        Backup verification is intentionally distinct from current-schema
        verification: a checksummed historical migration prefix can be a valid
        recovery artifact without being safe to open as a current Library.
        """

        from dithyramba.store.migrations import discover_migrations

        return self.manifest.schema_version == discover_migrations()[-1].version


@dataclass(frozen=True, slots=True)
class LibraryMigrationReceipt:
    """Canonical proof that one historical Library reached the current head."""

    SCHEMA = "dithyramba.library_migration_receipt/1.0"

    library_id: str
    logical_identity_hash: str
    old_schema_version: int
    new_schema_version: int
    old_schema_fingerprint: str
    new_schema_fingerprint: str
    backup_bundle_id: str
    backup_manifest_hash: str
    backup_database_sha256: str
    applied_migrations: tuple[BackupMigrationEntry, ...]

    def __post_init__(self) -> None:
        if not self.library_id.startswith("library_") or not _ID_PATTERN.fullmatch(self.library_id):
            raise ValueError("library_id is invalid")
        if _HASH_PATTERN.fullmatch(self.logical_identity_hash) is None:
            raise ValueError("logical identity hash is invalid")
        if (
            type(self.old_schema_version) is not int
            or type(self.new_schema_version) is not int
            or self.old_schema_version < 1
            or self.new_schema_version <= self.old_schema_version
        ):
            raise ValueError("migration receipt versions are invalid")
        for value in (
            self.old_schema_fingerprint,
            self.new_schema_fingerprint,
            self.backup_manifest_hash,
            self.backup_database_sha256,
        ):
            if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError("migration receipt hash is invalid")
        if not self.backup_bundle_id.startswith("backup_bundle_") or not _ID_PATTERN.fullmatch(
            self.backup_bundle_id
        ):
            raise ValueError("backup_bundle_id is invalid")
        if tuple(item.version for item in self.applied_migrations) != tuple(
            range(self.old_schema_version + 1, self.new_schema_version + 1)
        ):
            raise ValueError("applied migrations do not span old to new schema versions")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "logical_identity_hash": self.logical_identity_hash,
            "old_schema_version": self.old_schema_version,
            "new_schema_version": self.new_schema_version,
            "old_schema_fingerprint": self.old_schema_fingerprint,
            "new_schema_fingerprint": self.new_schema_fingerprint,
            "backup_bundle_id": self.backup_bundle_id,
            "backup_manifest_hash": self.backup_manifest_hash,
            "backup_database_sha256": self.backup_database_sha256,
            "applied_migrations": [item.payload() for item in self.applied_migrations],
            "status": "verified",
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.payload())
