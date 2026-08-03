"""Complete, verified, single-Library BackupBundle creation and restore."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, new_id
from dithyramba.library import (
    LibraryConfig,
    LibraryPaths,
    create_library_layout,
    validate_library_id,
    validate_library_layout,
)
from dithyramba.library.errors import LibraryAlreadyExistsError
from dithyramba.library.paths import PathInput
from dithyramba.persistence.repository import (
    LibraryRepository,
    library_logical_identity_hash,
    open_library,
)
from dithyramba.store import MigrationRunner, schema_fingerprint, verify_integrity
from dithyramba.store.database import Store

from .errors import (
    BackupBundleConflictError,
    BackupBundleError,
    BackupBundleIntegrityError,
)
from .models import (
    BackupBundleManifest,
    BackupBundleRecord,
    BackupFileEntry,
    BackupMigrationEntry,
)

_FILE_MODE = 0o600
_DIRECTORY_MODE = 0o700
_CHUNK_SIZE = 1024 * 1024
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BLOB_PATH_PATTERN = re.compile(r"^blobs/[0-9a-f]{2}/[0-9a-f]{62}$")
_MANIFEST_KEYS = {
    "schema",
    "backup_bundle_id",
    "library_id",
    "library_name",
    "logical_identity_hash",
    "created_at",
    "schema_version",
    "schema_fingerprint",
    "migrations",
    "row_counts",
    "last_event_id",
    "event_count",
    "files",
}


def create_backup_bundle(
    repository: LibraryRepository,
    *,
    output_directory: str | os.PathLike[str] | None = None,
) -> BackupBundleRecord:
    """Create one atomic DB/events/blob bundle from a consistent SQLite cut."""

    if not isinstance(repository, LibraryRepository):
        raise TypeError("create_backup_bundle requires a LibraryRepository")
    repository.verify()
    if repository._store.connection.in_transaction:
        raise BackupBundleError("cannot create a BackupBundle inside a transaction")
    return _create_backup_bundle(
        paths=repository.paths,
        source_connection=repository._store.connection,
        output_directory=output_directory,
        created_at=_repository_timestamp(repository),
        require_current_schema=True,
    )


def create_pre_migration_backup_bundle(
    store: Store,
    paths: LibraryPaths,
    *,
    output_directory: str | os.PathLike[str] | None = None,
) -> BackupBundleRecord:
    """Create a complete bundle from a verified historical migration prefix.

    This is the only bundle-creation path that accepts a non-head schema. It
    requires an explicit migration-only ``Store`` bound to the exact Library
    database and refuses an empty or already-current schema.
    """

    if not isinstance(store, Store) or not isinstance(paths, LibraryPaths):
        raise TypeError("pre-migration backup requires Store and LibraryPaths")
    validate_library_layout(paths)
    if store.path.resolve(strict=True) != paths.database.resolve(strict=True):
        raise BackupBundleIntegrityError("migration Store is not bound to the Library database")
    if store.connection.in_transaction:
        raise BackupBundleError("cannot create a BackupBundle inside a transaction")
    runner = MigrationRunner(store.connection)
    migrations = runner.verify()
    if not migrations:
        raise BackupBundleIntegrityError("pre-migration database has no managed schema")
    if len(migrations) == runner.latest_version:
        raise BackupBundleError("pre-migration backup requires a Library with pending migrations")
    verify_integrity(store.connection)
    return _create_backup_bundle(
        paths=paths,
        source_connection=store.connection,
        output_directory=output_directory,
        created_at=_utc_timestamp(),
        require_current_schema=False,
    )


def _create_backup_bundle(
    *,
    paths: LibraryPaths,
    source_connection: sqlite3.Connection,
    output_directory: str | os.PathLike[str] | None,
    created_at: str,
    require_current_schema: bool,
) -> BackupBundleRecord:
    """Create one atomic DB/events/blob bundle from a verified SQLite cut."""

    source_runner = MigrationRunner(source_connection)
    source_migrations = (
        source_runner.verify_at_head() if require_current_schema else source_runner.verify()
    )
    if not source_migrations:
        raise BackupBundleIntegrityError("backup database has no managed schema")
    verify_integrity(source_connection)
    output = (
        paths.backups
        if output_directory is None
        else _absolute_existing_private_directory(Path(output_directory))
    )
    if output_directory is None:
        _require_private_directory(output, "backup output directory")
    backup_bundle_id = new_id("backup_bundle")
    destination = output / f"{backup_bundle_id}.dithyramba"
    if os.path.lexists(destination):
        raise BackupBundleConflictError("BackupBundle destination already exists")
    staging = Path(tempfile.mkdtemp(prefix=".bundle-", suffix=".tmp", dir=output))
    os.chmod(staging, _DIRECTORY_MODE)
    try:
        database_path = staging / "memory.sqlite3"
        destination_connection = sqlite3.connect(database_path, isolation_level=None)
        try:
            source_connection.backup(destination_connection)
            runner = MigrationRunner(destination_connection)
            migrations = runner.verify_at_head() if require_current_schema else runner.verify()
            if migrations != source_migrations:
                raise BackupBundleIntegrityError(
                    "backup migration prefix differs from its source database"
                )
            verify_integrity(destination_connection)
            snapshot = _database_snapshot(destination_connection)
            event_bytes, event_count, last_event_id = _event_log_bytes(destination_connection)
        finally:
            destination_connection.close()
        _secure_and_fsync_file(database_path)

        events_path = staging / "events.jsonl"
        _write_private_file(events_path, event_bytes)

        file_paths: list[Path] = [database_path, events_path]
        for content_sha256, byte_size, relative_path in snapshot.blobs:
            expected_relative = f"{content_sha256[:2]}/{content_sha256[2:]}"
            if relative_path != expected_relative:
                raise BackupBundleIntegrityError("database Blob path is not content-addressed")
            source = paths.blobs / PurePosixPath(relative_path)
            target = staging / "blobs" / PurePosixPath(relative_path)
            target.parent.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
            _secure_directory_chain(target.parent, staging)
            _copy_exact_private_file(
                source,
                target,
                expected_sha256=content_sha256,
                expected_size=byte_size,
            )
            file_paths.append(target)

        entries = tuple(
            sorted(
                (_file_entry(path, relative_to=staging) for path in file_paths),
                key=lambda item: item.path,
            )
        )
        manifest = BackupBundleManifest(
            backup_bundle_id=backup_bundle_id,
            library_id=snapshot.library_id,
            library_name=snapshot.library_name,
            logical_identity_hash=snapshot.logical_identity_hash,
            created_at=created_at,
            schema_version=len(migrations),
            schema_fingerprint=snapshot.schema_fingerprint,
            migrations=tuple(
                BackupMigrationEntry(item.version, item.name, item.sha256) for item in migrations
            ),
            row_counts=snapshot.row_counts,
            last_event_id=last_event_id,
            event_count=event_count,
            files=entries,
        )
        _write_private_file(staging / "manifest.json", manifest.canonical_bytes)
        _fsync_tree(staging)
        os.replace(staging, destination)
        _fsync_directory(output)
        record = verify_backup_bundle(destination)
        if record.manifest != manifest:
            raise BackupBundleIntegrityError("promoted BackupBundle differs from its manifest")
        if require_current_schema and not record.is_current_schema:
            raise BackupBundleIntegrityError("current-schema backup was not verified at head")
        return record
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)
        raise


def verify_backup_bundle(path: str | os.PathLike[str]) -> BackupBundleRecord:
    """Verify paths, permissions, hashes, schema, rows, events, and Blob closure."""

    root = Path(path).expanduser()
    if not root.is_absolute() or ".." in root.parts:
        raise BackupBundleIntegrityError("BackupBundle path must be absolute without '..'")
    _require_private_directory(root, "BackupBundle root")
    manifest_path = root / "manifest.json"
    manifest_bytes = _read_private_file(manifest_path, maximum=8 * 1024 * 1024)
    manifest = _parse_manifest(manifest_bytes)
    expected_files = {entry.path: entry for entry in manifest.files}
    actual_files = _walk_bundle_files(root)
    if set(actual_files) != set(expected_files) | {"manifest.json"}:
        raise BackupBundleIntegrityError("BackupBundle file set differs from manifest")
    for relative_path, entry in expected_files.items():
        candidate = actual_files[relative_path]
        _verify_file_entry(candidate, entry)

    database_path = actual_files["memory.sqlite3"]
    connection = _open_immutable_database(database_path)
    try:
        runner = MigrationRunner(connection)
        migrations = runner.verify()
        if not migrations:
            raise BackupBundleIntegrityError("backup database has no managed schema")
        verify_integrity(connection)
        snapshot = _database_snapshot(connection)
        event_bytes, event_count, last_event_id = _event_log_bytes(connection)
    finally:
        connection.close()
    if manifest.schema_version != len(migrations):
        raise BackupBundleIntegrityError("manifest schema version differs from database")
    expected_migrations = tuple(
        BackupMigrationEntry(item.version, item.name, item.sha256) for item in migrations
    )
    if manifest.migrations != expected_migrations:
        raise BackupBundleIntegrityError("manifest migrations differ from database")
    if (
        manifest.library_id != snapshot.library_id
        or manifest.library_name != snapshot.library_name
        or manifest.logical_identity_hash != snapshot.logical_identity_hash
        or manifest.schema_fingerprint != snapshot.schema_fingerprint
        or manifest.row_counts != snapshot.row_counts
    ):
        raise BackupBundleIntegrityError("manifest identity/schema/rows differ from database")
    if manifest.event_count != event_count or manifest.last_event_id != last_event_id:
        raise BackupBundleIntegrityError("manifest event cursor differs from database")
    if _read_private_file(actual_files["events.jsonl"]) != event_bytes:
        raise BackupBundleIntegrityError("events.jsonl differs from the database snapshot")
    expected_blob_paths = {
        f"blobs/{relative_path}" for _digest, _size, relative_path in snapshot.blobs
    }
    manifested_blob_paths = {name for name in expected_files if name.startswith("blobs/")}
    if expected_blob_paths != manifested_blob_paths:
        raise BackupBundleIntegrityError("manifest Blob set differs from database references")
    return BackupBundleRecord(path=root, manifest=manifest)


def restore_backup_bundle(
    path: str | os.PathLike[str],
    *,
    data_root: PathInput | None = None,
    declared_source_roots: tuple[PathInput, ...] = (),
    declared_sync_roots: tuple[PathInput, ...] = (),
) -> LibraryRepository:
    """Restore a verified bundle under a new physical path, preserving logical IDs."""

    record = verify_backup_bundle(path)
    manifest = record.manifest
    config = LibraryConfig(
        library_id=validate_library_id(manifest.library_id), name=manifest.library_name
    )
    if library_logical_identity_hash(config) != manifest.logical_identity_hash:
        raise BackupBundleIntegrityError("manifest Library logical identity is inconsistent")
    historical_schema = not record.is_current_schema
    if historical_schema:
        raise BackupBundleIntegrityError(
            "pre-v1 BackupBundle detected; Dithyramba v1 does not rewrite historical "
            "Libraries in place. Restore it with the matching pre-v1 release or rebuild "
            "a new v1 Library from the original read-only sources."
        )
    paths = None
    try:
        paths = create_library_layout(
            config,
            data_root=data_root,
            declared_source_roots=declared_source_roots,
            declared_sync_roots=declared_sync_roots,
        )
        entry_by_name = {entry.path: entry for entry in manifest.files}
        source_by_name = {name: record.path / PurePosixPath(name) for name in entry_by_name}
        _replace_from_verified_file(
            source_by_name["memory.sqlite3"],
            paths.database,
            expected=entry_by_name["memory.sqlite3"],
        )
        _replace_from_verified_file(
            source_by_name["events.jsonl"],
            paths.events,
            expected=entry_by_name["events.jsonl"],
        )
        for entry in manifest.files:
            if not entry.path.startswith("blobs/"):
                continue
            relative_blob = PurePosixPath(entry.path).relative_to("blobs")
            destination = paths.blobs / relative_blob
            destination.parent.mkdir(mode=_DIRECTORY_MODE, parents=True, exist_ok=True)
            _secure_directory_chain(destination.parent, paths.root)
            _copy_exact_private_file(
                source_by_name[entry.path],
                destination,
                expected_sha256=entry.sha256,
                expected_size=entry.byte_size,
            )
        _fsync_tree(paths.root)
        repository = open_library(config.library_id, data_root=paths.application_data_root)
        try:
            if repository.schema_fingerprint != manifest.schema_fingerprint:
                raise BackupBundleIntegrityError("restored schema fingerprint differs")
            restored_counts = _row_counts(repository._store.connection)
            if restored_counts != manifest.row_counts:
                raise BackupBundleIntegrityError("restored table counts differ")
            restored_events, event_count, last_event_id = _event_log_bytes(
                repository._store.connection
            )
            if event_count != manifest.event_count or last_event_id != manifest.last_event_id:
                raise BackupBundleIntegrityError("restored event cursor differs")
            if _read_private_file(paths.events) != restored_events:
                raise BackupBundleIntegrityError("restored events.jsonl differs from database")
            for digest, _size, _relative in _blob_rows(repository._store.connection):
                repository.get_blob(digest)
        except Exception:
            repository.close()
            raise
        return repository
    except (FileExistsError, LibraryAlreadyExistsError) as exc:
        raise BackupBundleConflictError("restore destination already exists") from exc
    except Exception:
        if paths is not None:
            shutil.rmtree(paths.root, ignore_errors=True)
        raise


class _DatabaseSnapshot:
    __slots__ = (
        "blobs",
        "library_id",
        "library_name",
        "logical_identity_hash",
        "row_counts",
        "schema_fingerprint",
    )

    def __init__(
        self,
        *,
        library_id: str,
        library_name: str,
        logical_identity_hash: str,
        schema_fingerprint_value: str,
        row_counts: tuple[tuple[str, int], ...],
        blobs: tuple[tuple[str, int, str], ...],
    ) -> None:
        self.library_id = library_id
        self.library_name = library_name
        self.logical_identity_hash = logical_identity_hash
        self.schema_fingerprint = schema_fingerprint_value
        self.row_counts = row_counts
        self.blobs = blobs


def _database_snapshot(connection: sqlite3.Connection) -> _DatabaseSnapshot:
    rows = connection.execute(
        "SELECT library_id, name, logical_identity_hash FROM libraries ORDER BY library_id"
    ).fetchall()
    if len(rows) != 1:
        raise BackupBundleIntegrityError("backup database must contain exactly one Library")
    row = rows[0]
    library_id = validate_library_id(str(row[0]))
    config = LibraryConfig(library_id=library_id, name=str(row[1]))
    identity_hash = str(row[2])
    if identity_hash != library_logical_identity_hash(config):
        raise BackupBundleIntegrityError("backup database Library identity hash is invalid")
    return _DatabaseSnapshot(
        library_id=library_id,
        library_name=config.name,
        logical_identity_hash=identity_hash,
        schema_fingerprint_value=schema_fingerprint(connection),
        row_counts=_row_counts(connection),
        blobs=_blob_rows(connection),
    )


def _blob_rows(connection: sqlite3.Connection) -> tuple[tuple[str, int, str], ...]:
    rows = connection.execute(
        "SELECT content_sha256, byte_size, relative_path FROM blobs ORDER BY content_sha256"
    ).fetchall()
    result: list[tuple[str, int, str]] = []
    for row in rows:
        digest = str(row[0])
        size = row[1]
        relative_path = str(row[2])
        if _HASH_PATTERN.fullmatch(digest) is None or type(size) is not int or size < 0:
            raise BackupBundleIntegrityError("database Blob metadata is invalid")
        if relative_path != f"{digest[:2]}/{digest[2:]}":
            raise BackupBundleIntegrityError("database Blob relative path is invalid")
        result.append((digest, size, relative_path))
    return tuple(result)


def _event_log_bytes(connection: sqlite3.Connection) -> tuple[bytes, int, str | None]:
    rows = connection.execute(
        """
        SELECT event_id, event_type, aggregate_type, aggregate_id,
               payload_json, payload_hash, occurred_at
        FROM event_outbox ORDER BY occurred_at, event_id
        """
    ).fetchall()
    lines: list[bytes] = []
    for row in rows:
        payload_text = str(row[4])
        payload = _canonical_object(payload_text, "event payload")
        if canonical_sha256_hex(payload) != str(row[5]):
            raise BackupBundleIntegrityError("event payload hash is invalid")
        envelope = {
            "schema": "dithyramba.event/1.0",
            "event_id": str(row[0]),
            "event_type": str(row[1]),
            "aggregate_type": str(row[2]),
            "aggregate_id": str(row[3]),
            "payload": payload,
            "payload_hash": str(row[5]),
            "occurred_at": str(row[6]),
        }
        lines.append(canonical_json_bytes(envelope) + b"\n")
    return b"".join(lines), len(rows), (None if not rows else str(rows[-1][0]))


def _row_counts(connection: sqlite3.Connection) -> tuple[tuple[str, int], ...]:
    table_rows = connection.execute(
        """
        SELECT name FROM sqlite_schema
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    result: list[tuple[str, int]] = []
    for row in table_rows:
        name = str(row[0])
        if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise BackupBundleIntegrityError("database table name is not canonical")
        count = int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        result.append((name, count))
    return tuple(result)


def _parse_manifest(raw: bytes) -> BackupBundleManifest:
    try:
        value: object = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BackupBundleIntegrityError("manifest.json is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise BackupBundleIntegrityError("manifest.json must contain one object")
    payload = cast(dict[str, object], value)
    if canonical_json_bytes(payload) != raw or set(payload) != _MANIFEST_KEYS:
        raise BackupBundleIntegrityError("manifest.json is not canonical or has unknown fields")
    try:
        raw_migrations = payload["migrations"]
        raw_counts = payload["row_counts"]
        raw_files = payload["files"]
        if (
            type(raw_migrations) is not list
            or type(raw_counts) is not dict
            or type(raw_files) is not list
        ):
            raise ValueError("manifest collections have invalid types")
        migrations = tuple(
            BackupMigrationEntry(
                version=_exact_int(item, "version"),
                name=_exact_str(item, "name"),
                sha256=_exact_str(item, "sha256"),
            )
            for item in _exact_object_items(
                raw_migrations, {"version", "name", "sha256"}, "migration"
            )
        )
        counts: list[tuple[str, int]] = []
        for name, count in raw_counts.items():
            if type(name) is not str or type(count) is not int:
                raise ValueError("row count has invalid type")
            counts.append((name, count))
        files = tuple(
            BackupFileEntry(
                path=_exact_str(item, "path"),
                byte_size=_exact_int(item, "byte_size"),
                sha256=_exact_str(item, "sha256"),
                mode=_exact_int(item, "mode"),
            )
            for item in _exact_object_items(
                raw_files, {"path", "byte_size", "sha256", "mode"}, "file"
            )
        )
        manifest = BackupBundleManifest(
            backup_bundle_id=_exact_str(payload, "backup_bundle_id"),
            library_id=_exact_str(payload, "library_id"),
            library_name=_exact_str(payload, "library_name"),
            logical_identity_hash=_exact_str(payload, "logical_identity_hash"),
            created_at=_exact_str(payload, "created_at"),
            schema_version=_exact_int(payload, "schema_version"),
            schema_fingerprint=_exact_str(payload, "schema_fingerprint"),
            migrations=migrations,
            row_counts=tuple(sorted(counts)),
            last_event_id=_optional_str(payload, "last_event_id"),
            event_count=_exact_int(payload, "event_count"),
            files=files,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BackupBundleIntegrityError("manifest.json violates BackupBundle contract") from exc
    if manifest.canonical_bytes != raw:
        raise BackupBundleIntegrityError("manifest differs from typed reconstruction")
    return manifest


def _exact_object_items(
    values: object,
    keys: set[str],
    label: str,
) -> tuple[dict[str, object], ...]:
    if type(values) is not list:
        raise ValueError(f"{label} values must be an array")
    result: list[dict[str, object]] = []
    for value in values:
        if type(value) is not dict or set(value) != keys:
            raise ValueError(f"{label} entry has invalid fields")
        result.append(cast(dict[str, object], value))
    return tuple(result)


def _exact_str(value: dict[str, object], key: str) -> str:
    item = value[key]
    if type(item) is not str:
        raise ValueError(f"{key} must be text")
    return item


def _optional_str(value: dict[str, object], key: str) -> str | None:
    item = value[key]
    if item is not None and type(item) is not str:
        raise ValueError(f"{key} must be text or null")
    return item


def _exact_int(value: dict[str, object], key: str) -> int:
    item = value[key]
    if type(item) is not int:
        raise ValueError(f"{key} must be an integer")
    return item


def _walk_bundle_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        _require_private_directory(current, "BackupBundle directory")
        for name in tuple(names):
            candidate = current / name
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise BackupBundleIntegrityError("BackupBundle contains a non-directory path")
        for filename in filenames:
            candidate = current / filename
            relative = candidate.relative_to(root).as_posix()
            _require_private_file(candidate, "BackupBundle file")
            if relative in result:
                raise BackupBundleIntegrityError("BackupBundle contains duplicate paths")
            if relative not in {"manifest.json", "memory.sqlite3", "events.jsonl"} and (
                _BLOB_PATH_PATTERN.fullmatch(relative) is None
            ):
                raise BackupBundleIntegrityError("BackupBundle contains an unsupported path")
            result[relative] = candidate
    return result


def _verify_file_entry(path: Path, entry: BackupFileEntry) -> None:
    metadata = path.stat()
    if metadata.st_size != entry.byte_size or stat.S_IMODE(metadata.st_mode) != entry.mode:
        raise BackupBundleIntegrityError(f"BackupBundle file metadata differs: {entry.path}")
    if _file_sha256(path) != entry.sha256:
        raise BackupBundleIntegrityError(f"BackupBundle file hash differs: {entry.path}")


def _file_entry(path: Path, *, relative_to: Path) -> BackupFileEntry:
    _require_private_file(path, "BackupBundle file")
    metadata = path.stat()
    return BackupFileEntry(
        path=path.relative_to(relative_to).as_posix(),
        byte_size=metadata.st_size,
        sha256=_file_sha256(path),
        mode=stat.S_IMODE(metadata.st_mode),
    )


def _replace_from_verified_file(
    source: Path,
    destination: Path,
    *,
    expected: BackupFileEntry,
) -> None:
    """Replace one core file only with bytes bound to the verified manifest."""

    temporary = destination.with_name(f".{destination.name}.restore.tmp")
    temporary.unlink(missing_ok=True)
    try:
        _copy_exact_private_file(
            source,
            temporary,
            expected_sha256=expected.sha256,
            expected_size=expected.byte_size,
        )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_exact_private_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    _require_private_file(source, "copy source")
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor: int | None = None
    digest = hashlib.sha256()
    total = 0
    try:
        _require_private_descriptor(source_descriptor, "copy source")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            _FILE_MODE,
        )
        while True:
            chunk = os.read(source_descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            _write_all(destination_descriptor, chunk)
        os.fchmod(destination_descriptor, _FILE_MODE)
        os.fsync(destination_descriptor)
    except Exception:
        # O_EXCL may fail because a destination already exists. Only remove a
        # file created by this invocation; preserving pre-existing bytes is a
        # defense-in-depth guarantee even though current callers use new paths.
        if destination_descriptor is not None:
            os.close(destination_descriptor)
            destination_descriptor = None
            destination.unlink(missing_ok=True)
        raise
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    if total != expected_size or digest.hexdigest() != expected_sha256:
        destination.unlink(missing_ok=True)
        raise BackupBundleIntegrityError("copied file differs from expected bytes")


def _write_private_file(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        _FILE_MODE,
    )
    try:
        _write_all(descriptor, data)
        os.fchmod(descriptor, _FILE_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        amount = os.write(descriptor, view[written:])
        if amount <= 0:
            raise BackupBundleError("file write made no progress")
        written += amount


def _read_private_file(path: Path, maximum: int | None = None) -> bytes:
    _require_private_file(path, "BackupBundle file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = _require_private_descriptor(descriptor, "BackupBundle file")
        if maximum is not None and metadata.st_size > maximum:
            raise BackupBundleIntegrityError("BackupBundle file exceeds its safe limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                raise BackupBundleIntegrityError("BackupBundle file exceeds its safe limit")
            chunks.append(chunk)
        if total != metadata.st_size:
            raise BackupBundleIntegrityError("BackupBundle file changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        before = _require_private_descriptor(descriptor, "hash input")
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise BackupBundleIntegrityError("file changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _secure_and_fsync_file(path: Path) -> None:
    os.chmod(path, _FILE_MODE)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        _require_private_descriptor(descriptor, "SQLite backup")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _absolute_existing_private_directory(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute() or ".." in expanded.parts:
        raise BackupBundleError("backup output directory must be absolute without '..'")
    resolved = expanded.resolve(strict=True)
    _require_private_directory(resolved, "backup output directory")
    return resolved


def _require_private_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BackupBundleIntegrityError(f"{label} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise BackupBundleIntegrityError(f"{label} must be a real directory")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
    ):
        raise BackupBundleIntegrityError(f"{label} must be owner-only 0700")


def _require_private_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BackupBundleIntegrityError(f"{label} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BackupBundleIntegrityError(f"{label} must be a regular file")
    if metadata.st_nlink != 1:
        raise BackupBundleIntegrityError(f"{label} must not be hardlinked")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
    ):
        raise BackupBundleIntegrityError(f"{label} must be owner-only 0600")


def _require_private_descriptor(descriptor: int, label: str) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise BackupBundleIntegrityError(f"{label} descriptor is not one regular file")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
    ):
        raise BackupBundleIntegrityError(f"{label} descriptor is not private")
    return metadata


def _secure_directory_chain(path: Path, boundary: Path) -> None:
    current = path
    while True:
        os.chmod(current, _DIRECTORY_MODE)
        _require_private_directory(current, "bundle directory")
        if current == boundary:
            return
        if boundary not in current.parents:
            raise BackupBundleIntegrityError("directory escaped its expected boundary")
        current = current.parent


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for directory, _names, _files in os.walk(root):
        directories.append(Path(directory))
    for sync_directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(sync_directory)


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_immutable_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _canonical_object(value: str, label: str) -> dict[str, object]:
    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise BackupBundleIntegrityError(f"{label} is not valid JSON") from exc
    if type(decoded) is not dict:
        raise BackupBundleIntegrityError(f"{label} must be an object")
    result = cast(dict[str, object], decoded)
    if canonical_json_bytes(result).decode("utf-8") != value:
        raise BackupBundleIntegrityError(f"{label} is not canonical JSON")
    return result


def _repository_timestamp(repository: LibraryRepository) -> str:
    from dithyramba.persistence.repository import _timestamp

    return _timestamp(repository._clock)


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
