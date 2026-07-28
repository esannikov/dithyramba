"""Backup-first orchestration for explicitly migrating one historical Library."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path

from dithyramba.library import (
    LibraryConfig,
    LibraryPaths,
    library_paths,
    validate_library_id,
    validate_library_layout,
)
from dithyramba.library.paths import PathInput
from dithyramba.persistence.repository import library_logical_identity_hash
from dithyramba.store.database import Store
from dithyramba.store.migrations import Migration, MigrationRunner, schema_fingerprint

from .bundle import (
    create_pre_migration_backup_bundle,
    verify_backup_bundle,
)
from .errors import BackupBundleIntegrityError, LibraryMigrationError
from .models import (
    BackupBundleRecord,
    BackupMigrationEntry,
    LibraryMigrationReceipt,
)

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class LibraryMigrationResult:
    """A verified migration receipt and the exact backup that authorized it."""

    receipt: LibraryMigrationReceipt
    backup: BackupBundleRecord


@dataclass(frozen=True, slots=True)
class _LibraryIdentity:
    library_id: str
    name: str
    logical_identity_hash: str
    created_at: str


def migrate_library(
    library_id: str,
    *,
    data_root: PathInput | None = None,
    backup_output_directory: str | os.PathLike[str] | None = None,
) -> LibraryMigrationResult:
    """Create, verify, and use a complete pre-migration BackupBundle."""

    paths = library_paths(validate_library_id(library_id), data_root=data_root)
    return _migrate_library_paths(
        paths,
        expected_library_id=library_id,
        backup_output_directory=backup_output_directory,
        existing_backup_path=None,
    )


def migrate_library_from_backup(
    library_id: str,
    *,
    data_root: PathInput,
    backup_bundle: str | os.PathLike[str],
) -> LibraryMigrationResult:
    """Migrate restored historical bytes using their source bundle as proof.

    This recovery seam is intentionally separate from the normal operator
    command: the destination must already contain the exact historical Library
    represented by ``backup_bundle``.
    """

    paths = library_paths(validate_library_id(library_id), data_root=data_root)
    return _migrate_library_paths(
        paths,
        expected_library_id=library_id,
        backup_output_directory=None,
        existing_backup_path=Path(backup_bundle),
    )


def _migrate_library_paths(
    paths: LibraryPaths,
    *,
    expected_library_id: str,
    backup_output_directory: str | os.PathLike[str] | None,
    existing_backup_path: Path | None,
) -> LibraryMigrationResult:
    validate_library_layout(paths)
    with Store.open_library_for_migration(paths) as store:
        runner = MigrationRunner(store.connection)
        applied_prefix = runner.verify()
        if not applied_prefix:
            raise LibraryMigrationError("Library has no managed migration history")
        old_version = applied_prefix[-1].version
        if old_version == runner.latest_version:
            raise LibraryMigrationError(
                f"Library is already at schema head version {runner.latest_version}"
            )

        identity_before = _library_identity(store.connection, expected_library_id)
        fingerprint_before = schema_fingerprint(store.connection)
        database_hash_before = _logical_database_hash(store.connection)
        source_surfaces_before = _source_surface_hashes(paths)
        database_stat_before = paths.database.stat()

        if existing_backup_path is None:
            backup = create_pre_migration_backup_bundle(
                store,
                paths,
                output_directory=backup_output_directory,
            )
        else:
            backup = verify_backup_bundle(existing_backup_path)

        backup, backup_database_hash = _verified_backup_proof(
            backup.path,
            expected_manifest_hash=backup.manifest_hash,
            old_version=old_version,
            fingerprint_before=fingerprint_before,
            identity_before=identity_before,
        )
        if backup_database_hash != database_hash_before:
            raise LibraryMigrationError(
                "pre-migration BackupBundle does not represent the live Library state"
            )
        _acquire_exclusive_migration_lock(store.connection)

        prepared_database_hashes: dict[int, str] = {}

        def verify_migration_state(
            connection: sqlite3.Connection,
            migration: Migration,
        ) -> str:
            if connection is not store.connection:
                raise LibraryMigrationError("migration runner changed its SQLite connection")
            reverified, current_backup_hash = _verified_backup_proof(
                backup.path,
                expected_manifest_hash=backup.manifest_hash,
                old_version=old_version,
                fingerprint_before=fingerprint_before,
                identity_before=identity_before,
            )
            if (
                reverified.manifest != backup.manifest
                or current_backup_hash != backup_database_hash
            ):
                raise LibraryMigrationError("pre-migration BackupBundle changed after verification")
            current_database_hash = _logical_database_hash(connection)
            if migration.version == old_version + 1:
                if current_database_hash != backup_database_hash:
                    raise LibraryMigrationError("Library changed after its pre-migration backup")
            elif MigrationRunner(connection).current_version() != migration.version - 1:
                raise LibraryMigrationError("pending migration sequence changed unexpectedly")
            if _library_identity(connection, expected_library_id) != identity_before:
                raise LibraryMigrationError("Library identity changed before migration")
            if _source_surface_hashes(paths) != source_surfaces_before:
                raise LibraryMigrationError("Library source surfaces changed before migration")
            return current_database_hash

        def verify_backup_hook(connection: sqlite3.Connection, migration: Migration) -> None:
            prepared_database_hashes[migration.version] = verify_migration_state(
                connection,
                migration,
            )

        def verify_locked_guard(connection: sqlite3.Connection, migration: Migration) -> None:
            if not connection.in_transaction:
                raise LibraryMigrationError("locked migration guard has no write transaction")
            locked_database_hash = verify_migration_state(connection, migration)
            prepared_database_hash = prepared_database_hashes.get(migration.version)
            if prepared_database_hash is None or locked_database_hash != prepared_database_hash:
                raise LibraryMigrationError(
                    "Library changed between backup proof and locked migration"
                )

        completed = runner.apply_all(
            backup_hook=verify_backup_hook,
            locked_guard=verify_locked_guard,
        )
        runner.verify_at_head()
        store.verify()

        identity_after = _library_identity(store.connection, expected_library_id)
        if identity_after != identity_before:
            raise LibraryMigrationError("Library identity changed during migration")
        if _source_surface_hashes(paths) != source_surfaces_before:
            raise LibraryMigrationError("Library source surfaces changed during migration")
        database_stat_after = paths.database.stat()
        if (database_stat_before.st_dev, database_stat_before.st_ino) != (
            database_stat_after.st_dev,
            database_stat_after.st_ino,
        ):
            raise LibraryMigrationError("Library database was replaced during migration")

        final_backup, final_backup_hash = _verified_backup_proof(
            backup.path,
            expected_manifest_hash=backup.manifest_hash,
            old_version=old_version,
            fingerprint_before=fingerprint_before,
            identity_before=identity_before,
        )
        if final_backup.manifest != backup.manifest or final_backup_hash != backup_database_hash:
            raise LibraryMigrationError("pre-migration BackupBundle changed during migration")

        database_entry = next(
            entry for entry in backup.manifest.files if entry.path == "memory.sqlite3"
        )
        receipt = LibraryMigrationReceipt(
            library_id=identity_before.library_id,
            logical_identity_hash=identity_before.logical_identity_hash,
            old_schema_version=old_version,
            new_schema_version=runner.latest_version,
            old_schema_fingerprint=fingerprint_before,
            new_schema_fingerprint=schema_fingerprint(store.connection),
            backup_bundle_id=backup.manifest.backup_bundle_id,
            backup_manifest_hash=backup.manifest_hash,
            backup_database_sha256=database_entry.sha256,
            applied_migrations=tuple(
                BackupMigrationEntry(item.version, item.name, item.sha256) for item in completed
            ),
        )
        return LibraryMigrationResult(receipt=receipt, backup=backup)


def _acquire_exclusive_migration_lock(connection: sqlite3.Connection) -> None:
    """Keep SQLite's file lock across every committed migration in one command."""

    try:
        row = connection.execute("PRAGMA locking_mode = EXCLUSIVE").fetchone()
        if row is None or str(row[0]).lower() != "exclusive":
            raise LibraryMigrationError("SQLite refused exclusive migration locking")
        connection.execute("BEGIN EXCLUSIVE")
        connection.commit()
    except sqlite3.DatabaseError as error:
        if connection.in_transaction:
            connection.rollback()
        raise LibraryMigrationError("could not acquire exclusive migration lock") from error


def _verified_backup_proof(
    path: Path,
    *,
    expected_manifest_hash: str,
    old_version: int,
    fingerprint_before: str,
    identity_before: _LibraryIdentity,
) -> tuple[BackupBundleRecord, str]:
    record = verify_backup_bundle(path)
    manifest = record.manifest
    if record.manifest_hash != expected_manifest_hash:
        raise BackupBundleIntegrityError("pre-migration backup manifest hash changed")
    if manifest.schema_version != old_version:
        raise BackupBundleIntegrityError("pre-migration backup schema version differs")
    if manifest.schema_fingerprint != fingerprint_before:
        raise BackupBundleIntegrityError("pre-migration backup schema fingerprint differs")
    if (
        manifest.library_id != identity_before.library_id
        or manifest.library_name != identity_before.name
        or manifest.logical_identity_hash != identity_before.logical_identity_hash
    ):
        raise BackupBundleIntegrityError("pre-migration backup Library identity differs")
    if record.is_current_schema:
        raise BackupBundleIntegrityError("pre-migration backup incorrectly claims current head")

    database = record.path / "memory.sqlite3"
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        backup_hash = _logical_database_hash(connection)
    finally:
        connection.close()
    return record, backup_hash


def _library_identity(
    connection: sqlite3.Connection,
    expected_library_id: str,
) -> _LibraryIdentity:
    rows = connection.execute(
        """
        SELECT library_id, name, logical_identity_hash, created_at
        FROM libraries ORDER BY library_id
        """
    ).fetchall()
    if len(rows) != 1:
        raise LibraryMigrationError("migration database must contain exactly one Library")
    row = rows[0]
    library_id = validate_library_id(str(row[0]))
    name = str(row[1])
    config = LibraryConfig(library_id=library_id, name=name)
    identity_hash = str(row[2])
    if library_id != expected_library_id:
        raise LibraryMigrationError("Library path and database identity differ")
    if identity_hash != library_logical_identity_hash(config):
        raise LibraryMigrationError("Library logical identity hash is invalid")
    return _LibraryIdentity(
        library_id=library_id,
        name=name,
        logical_identity_hash=identity_hash,
        created_at=str(row[3]),
    )


def _logical_database_hash(connection: sqlite3.Connection) -> str:
    """Hash a consistent logical dump so online backup bytes can be compared."""

    owns_transaction = not connection.in_transaction
    digest = hashlib.sha256(b"dithyramba.logical_database/1.0\x00")
    if owns_transaction:
        connection.execute("BEGIN")
    try:
        for statement in connection.iterdump():
            payload = statement.encode("utf-8")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    finally:
        if owns_transaction:
            connection.rollback()
    return digest.hexdigest()


def _source_surface_hashes(paths: LibraryPaths) -> tuple[tuple[str, int, str], ...]:
    """Hash immutable source-bearing Library files outside the SQLite database."""

    candidates = [("events.jsonl", paths.events)]
    for directory, names, filenames in os.walk(paths.blobs, topdown=True, followlinks=False):
        current = Path(directory)
        for name in names:
            candidate = current / name
            if candidate.is_symlink():
                raise LibraryMigrationError("Library Blob tree contains a symlink directory")
        for filename in filenames:
            candidate = current / filename
            relative = candidate.relative_to(paths.root).as_posix()
            candidates.append((relative, candidate))

    result: list[tuple[str, int, str]] = []
    for relative, candidate in sorted(candidates):
        metadata = candidate.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise LibraryMigrationError("Library source surface is not one regular file")
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
        after = candidate.stat()
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise LibraryMigrationError("Library source surface changed while hashing")
        result.append((relative, metadata.st_size, digest.hexdigest()))
    return tuple(result)
