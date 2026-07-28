"""Backup-first v3-to-P6-head Library migration acceptance tests."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import dithyramba.backup.migration as migration_module
from dithyramba.backup import (
    BackupBundleConflictError,
    BackupBundleIntegrityError,
    LibraryMigrationError,
    create_backup_bundle,
    migrate_library,
    restore_backup_bundle,
    verify_backup_bundle,
)
from dithyramba.backup.bundle import create_pre_migration_backup_bundle
from dithyramba.backup.models import BackupBundleRecord
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.library import LibraryConfig, LibraryPaths, create_library_layout
from dithyramba.persistence import initialize_library, open_library
from dithyramba.persistence.repository import library_logical_identity_hash
from dithyramba.store import MigrationStateError
from dithyramba.store.database import Store
from dithyramba.store.errors import MigrationApplyError
from dithyramba.store.migrations import (
    BackupHook,
    LockedMigrationGuard,
    Migration,
    MigrationRunner,
    discover_migrations,
)

_NOW = "2026-07-21T12:00:00.000000Z"
_V3_FINAL_GUARD_CALL = 2 * (discover_migrations()[-1].version - 3) + 2


def _v3_library(tmp_path: Path) -> tuple[LibraryConfig, Path, LibraryPaths, Path]:
    data_home = tmp_path / "data"
    config = LibraryConfig(name="P6 migration fixture")
    paths = create_library_layout(config, data_root=data_home)
    prefix = tmp_path / "v3-migrations"
    prefix.mkdir()
    repository_root = Path(__file__).resolve().parents[2]
    for name in ("0001_core.sql", "0002_source_heads.sql", "0003_recall_run_artifacts.sql"):
        shutil.copyfile(repository_root / "migrations" / name, prefix / name)

    source = tmp_path / "source" / "document.md"
    source.parent.mkdir()
    source_bytes = b"# Historical source\n\nMigration must never rewrite these bytes.\n"
    source.write_bytes(source_bytes)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    relative_blob = f"{source_digest[:2]}/{source_digest[2:]}"
    blob = paths.blobs / relative_blob
    blob.parent.mkdir(mode=0o700)
    blob.parent.chmod(0o700)
    blob.write_bytes(source_bytes)
    blob.chmod(0o600)

    with Store.open_library(paths, migration_source=prefix) as store:
        identity_hash = library_logical_identity_hash(config)
        store.connection.execute(
            "INSERT INTO libraries VALUES (?, ?, ?, ?)",
            (config.library_id, config.name, identity_hash, _NOW),
        )
        store.connection.execute(
            "INSERT INTO blobs VALUES (?, ?, ?, ?)",
            (source_digest, len(source_bytes), relative_blob, _NOW),
        )
        store.connection.execute(
            "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
            (
                "source_migration_fixture",
                config.library_id,
                source.as_uri(),
                "text/markdown",
                "Historical source",
                _NOW,
            ),
        )
        store.connection.execute(
            """
            INSERT INTO source_versions VALUES (
                'source_version_migration_fixture', 'source_migration_fixture', 1,
                ?, ?, ?, ?, 'index/1.0', 'processed', NULL
            )
            """,
            (source_digest, len(source_bytes), _NOW, _NOW),
        )
        store.connection.execute(
            """
            INSERT INTO source_heads VALUES (
                'source_migration_fixture', 'source_version_migration_fixture', ?, ?
            )
            """,
            (_NOW, _NOW),
        )
        event_payload = {
            "schema": "dithyramba.library_created/1.0",
            "library_id": config.library_id,
            "name": config.name,
            "logical_identity_hash": identity_hash,
        }
        store.connection.execute(
            "INSERT INTO event_outbox VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                "event_migration_fixture",
                "library.created",
                "library",
                config.library_id,
                canonical_json_bytes(event_payload).decode("utf-8"),
                canonical_sha256_hex(event_payload),
                _NOW,
            ),
        )
        assert store.schema_version == 3
    return config, data_home, paths, source


def _logical_dump(database: Path) -> tuple[str, ...]:
    connection = sqlite3.connect(database)
    try:
        return tuple(connection.iterdump())
    finally:
        connection.close()


def _schema_version(database: Path) -> int:
    connection = sqlite3.connect(database)
    try:
        return int(connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0])
    finally:
        connection.close()


def test_v3_library_requires_explicit_backup_first_migration_and_emits_receipt(
    tmp_path: Path,
) -> None:
    config, data_home, paths, source = _v3_library(tmp_path)
    source_hash_before = hashlib.sha256(source.read_bytes()).hexdigest()

    with Store.open_library_for_migration(paths) as historical:
        assert historical.schema_version == 3
    with pytest.raises(MigrationStateError, match="behind code version 9"):
        open_library(config.library_id, data_root=data_home)

    result = migrate_library(config.library_id, data_root=data_home)

    assert result.receipt.old_schema_version == 3
    assert result.receipt.new_schema_version == 9
    assert [item.version for item in result.receipt.applied_migrations] == [4, 5, 6, 7, 8, 9]
    assert result.receipt.backup_manifest_hash == result.backup.manifest_hash
    assert result.receipt.payload()["status"] == "verified"
    assert result.receipt.canonical_bytes == canonical_json_bytes(result.receipt.payload())
    assert len(result.receipt.receipt_hash) == 64
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash_before

    historical_bundle = verify_backup_bundle(result.backup.path)
    assert historical_bundle.manifest.schema_version == 3
    assert historical_bundle.is_current_schema is False
    assert historical_bundle.manifest.logical_identity_hash == result.receipt.logical_identity_hash
    assert any(entry.path.startswith("blobs/") for entry in historical_bundle.manifest.files)

    with open_library(config.library_id, data_root=data_home) as repository:
        assert repository.schema_version == 9
        assert repository.library.logical_identity_hash == result.receipt.logical_identity_hash
        source_row = repository._store.connection.execute(
            "SELECT canonical_uri FROM sources WHERE source_id = 'source_migration_fixture'"
        ).fetchone()
        assert source_row is not None and str(source_row[0]) == source.as_uri()
        assert (
            repository._store.connection.execute("SELECT COUNT(*) FROM statements").fetchone()[0]
            == 0
        )


def test_backup_tamper_refuses_before_any_schema_or_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_home, paths, source = _v3_library(tmp_path)
    database_before = _logical_dump(paths.database)
    source_before = source.read_bytes()
    original_create = create_pre_migration_backup_bundle

    def create_then_tamper(
        store: Store,
        library_paths: LibraryPaths,
        *,
        output_directory: str | os.PathLike[str] | None = None,
    ) -> BackupBundleRecord:
        record = original_create(
            store,
            library_paths,
            output_directory=output_directory,
        )
        events = record.path / "events.jsonl"
        events.write_bytes(events.read_bytes() + b"tamper\n")
        events.chmod(0o600)
        return record

    monkeypatch.setattr(
        migration_module,
        "create_pre_migration_backup_bundle",
        create_then_tamper,
    )

    with pytest.raises(BackupBundleIntegrityError, match=r"metadata differs|hash differs"):
        migrate_library(config.library_id, data_root=data_home)

    assert _schema_version(paths.database) == 3
    assert _logical_dump(paths.database) == database_before
    assert source.read_bytes() == source_before


def test_backup_creation_failure_refuses_without_database_or_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_home, paths, source = _v3_library(tmp_path)
    database_before = _logical_dump(paths.database)
    source_before = source.read_bytes()

    def fail_backup(*_args: object, **_kwargs: object) -> None:
        raise BackupBundleIntegrityError("injected complete-backup failure")

    monkeypatch.setattr(migration_module, "create_pre_migration_backup_bundle", fail_backup)

    with pytest.raises(BackupBundleIntegrityError, match="injected complete-backup failure"):
        migrate_library(config.library_id, data_root=data_home)

    assert _schema_version(paths.database) == 3
    assert _logical_dump(paths.database) == database_before
    assert source.read_bytes() == source_before
    assert tuple(paths.backups.iterdir()) == ()


def test_historical_migration_bundle_restores_then_migrates_without_overwrite(
    tmp_path: Path,
) -> None:
    config, data_home, _paths, source = _v3_library(tmp_path)
    migrated = migrate_library(config.library_id, data_root=data_home)
    restored_home = tmp_path / "restored"

    with restore_backup_bundle(migrated.backup.path, data_root=restored_home) as restored:
        assert restored.schema_version == 9
        assert restored.library_id == config.library_id
        assert restored.library.logical_identity_hash == migrated.receipt.logical_identity_hash
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert restored.get_blob(digest).path.read_bytes() == source.read_bytes()
        assert (
            restored._store.connection.execute(
                "SELECT COUNT(*) FROM sources WHERE source_id = 'source_migration_fixture'"
            ).fetchone()[0]
            == 1
        )

    restored_database = restored_home / "libraries" / config.library_id / "memory.sqlite3"
    assert _schema_version(restored_database) == 9
    with pytest.raises(BackupBundleConflictError, match="already exists"):
        restore_backup_bundle(migrated.backup.path, data_root=restored_home)


def test_migration_rejects_changed_live_library_after_verified_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_home, paths, _source = _v3_library(tmp_path)
    original_proof = migration_module._verified_backup_proof
    calls = 0

    def verify_then_change(
        path: Path,
        *,
        expected_manifest_hash: str,
        old_version: int,
        fingerprint_before: str,
        identity_before: migration_module._LibraryIdentity,
    ) -> tuple[BackupBundleRecord, str]:
        nonlocal calls
        result = original_proof(
            path,
            expected_manifest_hash=expected_manifest_hash,
            old_version=old_version,
            fingerprint_before=fingerprint_before,
            identity_before=identity_before,
        )
        calls += 1
        if calls == 1:
            connection = sqlite3.connect(paths.database)
            try:
                connection.execute(
                    "INSERT INTO collections VALUES (?, ?, ?, ?, ?)",
                    (
                        "collection_concurrent_change",
                        config.library_id,
                        "Concurrent",
                        "corpus",
                        _NOW,
                    ),
                )
                connection.commit()
            finally:
                connection.close()
        return result

    monkeypatch.setattr(migration_module, "_verified_backup_proof", verify_then_change)

    with pytest.raises(MigrationApplyError, match="backup verification failed") as captured:
        migrate_library(config.library_id, data_root=data_home)
    assert isinstance(captured.value.__cause__, LibraryMigrationError)
    assert "changed after its pre-migration backup" in str(captured.value.__cause__)
    assert _schema_version(paths.database) == 3


def test_concurrent_writer_cannot_land_between_locked_proof_and_migration_sql(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_home, paths, _source = _v3_library(tmp_path)
    locked_proof_started = threading.Event()
    writer_attempting = threading.Event()
    writer_acquired = threading.Event()
    writer_versions: list[int] = []
    writer_errors: list[BaseException] = []

    def concurrent_writer() -> None:
        try:
            if not locked_proof_started.wait(timeout=5):
                raise AssertionError("locked proof never started")
            connection = sqlite3.connect(paths.database, isolation_level=None, timeout=5)
            try:
                writer_attempting.set()
                connection.execute("BEGIN IMMEDIATE")
                writer_acquired.set()
                writer_versions.append(
                    int(
                        connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[
                            0
                        ]
                    )
                )
                connection.execute(
                    "INSERT INTO collections VALUES (?, ?, ?, ?, ?)",
                    (
                        "collection_after_locked_migration",
                        config.library_id,
                        "After migration",
                        "corpus",
                        _NOW,
                    ),
                )
                connection.commit()
            finally:
                connection.close()
        except BaseException as exc:
            writer_errors.append(exc)

    writer = threading.Thread(target=concurrent_writer, daemon=True)
    writer.start()
    original_hash = migration_module._logical_database_hash
    observed_locked_proof = False

    def observe_locked_hash(connection: sqlite3.Connection) -> str:
        nonlocal observed_locked_proof
        if connection.in_transaction and not observed_locked_proof:
            observed_locked_proof = True
            locked_proof_started.set()
            assert writer_attempting.wait(timeout=5)
            assert writer_acquired.is_set() is False
        return original_hash(connection)

    monkeypatch.setattr(migration_module, "_logical_database_hash", observe_locked_hash)
    migrated = migrate_library(config.library_id, data_root=data_home)
    writer.join(timeout=5)

    assert migrated.receipt.new_schema_version == 9
    assert observed_locked_proof is True
    assert writer.is_alive() is False
    assert writer_errors == []
    assert writer_acquired.is_set() is True
    assert writer_versions == [9]
    with open_library(config.library_id, data_root=data_home) as repository:
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM collections "
                "WHERE collection_id = 'collection_after_locked_migration'"
            ).fetchone()[0]
            == 1
        )


def test_migration_rejects_empty_history_and_mismatching_backup_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unmanaged = LibraryConfig(name="Empty migration history")
    create_library_layout(unmanaged, data_root=tmp_path / "empty")
    with pytest.raises(LibraryMigrationError, match="no managed migration history"):
        migrate_library(unmanaged.library_id, data_root=tmp_path / "empty")

    config, data_home, paths, _source = _v3_library(tmp_path / "mismatch")
    original_proof = migration_module._verified_backup_proof

    def mismatching_proof(
        path: Path,
        *,
        expected_manifest_hash: str,
        old_version: int,
        fingerprint_before: str,
        identity_before: migration_module._LibraryIdentity,
    ) -> tuple[BackupBundleRecord, str]:
        record, _database_hash = original_proof(
            path,
            expected_manifest_hash=expected_manifest_hash,
            old_version=old_version,
            fingerprint_before=fingerprint_before,
            identity_before=identity_before,
        )
        return record, "f" * 64

    monkeypatch.setattr(migration_module, "_verified_backup_proof", mismatching_proof)
    with pytest.raises(LibraryMigrationError, match="does not represent the live Library"):
        migrate_library(config.library_id, data_root=data_home)
    assert _schema_version(paths.database) == 3


def test_migration_hook_rejects_connection_sequence_and_reverified_backup_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_home, paths, _source = _v3_library(tmp_path / "connection")

    def wrong_connection_apply(
        self: MigrationRunner,
        *,
        backup_hook: BackupHook | None = None,
        locked_guard: LockedMigrationGuard | None = None,
    ) -> tuple[Migration, ...]:
        assert backup_hook is not None
        assert locked_guard is not None
        other = sqlite3.connect(":memory:")
        try:
            backup_hook(other, self.migrations[-1])
        finally:
            other.close()
        return ()

    monkeypatch.setattr(MigrationRunner, "apply_all", wrong_connection_apply)
    with pytest.raises(LibraryMigrationError, match="changed its SQLite connection"):
        migrate_library(config.library_id, data_root=data_home)
    assert _schema_version(paths.database) == 3

    monkeypatch.undo()
    config, data_home, paths, _source = _v3_library(tmp_path / "sequence")

    def wrong_sequence_apply(
        self: MigrationRunner,
        *,
        backup_hook: BackupHook | None = None,
        locked_guard: LockedMigrationGuard | None = None,
    ) -> tuple[Migration, ...]:
        assert backup_hook is not None
        assert locked_guard is not None
        backup_hook(
            self._connection,
            Migration(5, "0005_probe.sql", "f" * 64, "SELECT 1;"),
        )
        return ()

    monkeypatch.setattr(MigrationRunner, "apply_all", wrong_sequence_apply)
    with pytest.raises(LibraryMigrationError, match="sequence changed unexpectedly"):
        migrate_library(config.library_id, data_root=data_home)
    assert _schema_version(paths.database) == 3

    monkeypatch.undo()
    config, data_home, paths, _source = _v3_library(tmp_path / "bundle-drift")
    original_proof = migration_module._verified_backup_proof
    calls = 0

    def drifting_proof(
        path: Path,
        *,
        expected_manifest_hash: str,
        old_version: int,
        fingerprint_before: str,
        identity_before: migration_module._LibraryIdentity,
    ) -> tuple[BackupBundleRecord, str]:
        nonlocal calls
        record, database_hash = original_proof(
            path,
            expected_manifest_hash=expected_manifest_hash,
            old_version=old_version,
            fingerprint_before=fingerprint_before,
            identity_before=identity_before,
        )
        calls += 1
        return record, ("f" * 64 if calls == 2 else database_hash)

    monkeypatch.setattr(migration_module, "_verified_backup_proof", drifting_proof)
    with pytest.raises(MigrationApplyError, match="backup verification failed") as captured:
        migrate_library(config.library_id, data_root=data_home)
    assert isinstance(captured.value.__cause__, LibraryMigrationError)
    assert "changed after verification" in str(captured.value.__cause__)
    assert _schema_version(paths.database) == 3


@pytest.mark.parametrize(
    "drift",
    ["identity_before", "source_before", "identity_after", "source_after"],
)
def test_migration_rejects_identity_or_source_surface_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    config, data_home, paths, _source = _v3_library(tmp_path)
    if drift.startswith("identity"):
        original_identity = migration_module._library_identity
        calls = 0

        def drifting_identity(
            connection: sqlite3.Connection,
            expected_library_id: str,
        ) -> migration_module._LibraryIdentity:
            nonlocal calls
            identity = original_identity(connection, expected_library_id)
            calls += 1
            target_call = 2 if drift == "identity_before" else _V3_FINAL_GUARD_CALL
            return replace(identity, created_at="changed") if calls == target_call else identity

        monkeypatch.setattr(migration_module, "_library_identity", drifting_identity)
    else:
        original_surfaces = migration_module._source_surface_hashes
        calls = 0

        def drifting_surfaces(
            library_paths: LibraryPaths,
        ) -> tuple[tuple[str, int, str], ...]:
            nonlocal calls
            surfaces = original_surfaces(library_paths)
            calls += 1
            target_call = 2 if drift == "source_before" else _V3_FINAL_GUARD_CALL
            return (*surfaces, ("drift", 0, "f" * 64)) if calls == target_call else surfaces

        monkeypatch.setattr(migration_module, "_source_surface_hashes", drifting_surfaces)

    if drift.endswith("before"):
        with pytest.raises(MigrationApplyError, match="backup verification failed"):
            migrate_library(config.library_id, data_root=data_home)
        assert _schema_version(paths.database) == 3
    else:
        with pytest.raises(LibraryMigrationError, match="changed during migration"):
            migrate_library(config.library_id, data_root=data_home)
        assert _schema_version(paths.database) == 9


def test_migration_rejects_database_replacement_and_final_backup_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, data_home, paths, _source = _v3_library(tmp_path / "replacement")
    original_apply = MigrationRunner.apply_all

    def replace_after_apply(
        self: MigrationRunner,
        *,
        backup_hook: BackupHook | None = None,
        locked_guard: LockedMigrationGuard | None = None,
    ) -> tuple[Migration, ...]:
        completed = original_apply(
            self,
            backup_hook=backup_hook,
            locked_guard=locked_guard,
        )
        self._connection.execute("PRAGMA wal_checkpoint(FULL)")
        replacement = paths.database.with_name("replacement.sqlite3")
        shutil.copyfile(paths.database, replacement)
        replacement.chmod(0o600)
        os.replace(replacement, paths.database)
        return completed

    monkeypatch.setattr(MigrationRunner, "apply_all", replace_after_apply)
    with pytest.raises(LibraryMigrationError, match="database was replaced"):
        migrate_library(config.library_id, data_root=data_home)
    assert _schema_version(paths.database) == 9

    monkeypatch.undo()
    config, data_home, paths, _source = _v3_library(tmp_path / "final-backup")
    original_proof = migration_module._verified_backup_proof
    calls = 0

    def final_drift(
        path: Path,
        *,
        expected_manifest_hash: str,
        old_version: int,
        fingerprint_before: str,
        identity_before: migration_module._LibraryIdentity,
    ) -> tuple[BackupBundleRecord, str]:
        nonlocal calls
        record, database_hash = original_proof(
            path,
            expected_manifest_hash=expected_manifest_hash,
            old_version=old_version,
            fingerprint_before=fingerprint_before,
            identity_before=identity_before,
        )
        calls += 1
        return record, ("f" * 64 if calls == _V3_FINAL_GUARD_CALL else database_hash)

    monkeypatch.setattr(migration_module, "_verified_backup_proof", final_drift)
    with pytest.raises(LibraryMigrationError, match="changed during migration"):
        migrate_library(config.library_id, data_root=data_home)
    assert _schema_version(paths.database) == 9


def test_backup_proof_identity_and_current_head_guards(tmp_path: Path) -> None:
    config, data_home, _paths, _source = _v3_library(tmp_path / "historical")
    migrated = migrate_library(config.library_id, data_root=data_home)
    manifest = migrated.backup.manifest
    identity = migration_module._LibraryIdentity(
        manifest.library_id,
        manifest.library_name,
        manifest.logical_identity_hash,
        _NOW,
    )
    with pytest.raises(BackupBundleIntegrityError, match="manifest hash changed"):
        migration_module._verified_backup_proof(
            migrated.backup.path,
            expected_manifest_hash="f" * 64,
            old_version=manifest.schema_version,
            fingerprint_before=manifest.schema_fingerprint,
            identity_before=identity,
        )
    with pytest.raises(BackupBundleIntegrityError, match="schema version differs"):
        migration_module._verified_backup_proof(
            migrated.backup.path,
            expected_manifest_hash=migrated.backup.manifest_hash,
            old_version=2,
            fingerprint_before=manifest.schema_fingerprint,
            identity_before=identity,
        )
    with pytest.raises(BackupBundleIntegrityError, match="schema fingerprint differs"):
        migration_module._verified_backup_proof(
            migrated.backup.path,
            expected_manifest_hash=migrated.backup.manifest_hash,
            old_version=manifest.schema_version,
            fingerprint_before="f" * 64,
            identity_before=identity,
        )
    with pytest.raises(BackupBundleIntegrityError, match="Library identity differs"):
        migration_module._verified_backup_proof(
            migrated.backup.path,
            expected_manifest_hash=migrated.backup.manifest_hash,
            old_version=manifest.schema_version,
            fingerprint_before=manifest.schema_fingerprint,
            identity_before=replace(identity, name="Changed"),
        )

    current = LibraryConfig(name="Current backup")
    with initialize_library(current, data_root=tmp_path / "current") as repository:
        current_backup = create_backup_bundle(repository)
    current_manifest = current_backup.manifest
    current_identity = migration_module._LibraryIdentity(
        current_manifest.library_id,
        current_manifest.library_name,
        current_manifest.logical_identity_hash,
        repository.library.created_at,
    )
    with pytest.raises(BackupBundleIntegrityError, match="incorrectly claims current head"):
        migration_module._verified_backup_proof(
            current_backup.path,
            expected_manifest_hash=current_backup.manifest_hash,
            old_version=current_manifest.schema_version,
            fingerprint_before=current_manifest.schema_fingerprint,
            identity_before=current_identity,
        )


def test_migration_identity_hash_transaction_and_source_surface_guards(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE libraries(library_id TEXT, name TEXT, logical_identity_hash TEXT, "
        "created_at TEXT)"
    )
    with pytest.raises(LibraryMigrationError, match="exactly one Library"):
        migration_module._library_identity(connection, "library_missing")

    first = LibraryConfig(name="First")
    second = LibraryConfig(name="Second")
    connection.execute(
        "INSERT INTO libraries VALUES (?, ?, ?, ?)",
        (first.library_id, first.name, library_logical_identity_hash(first), _NOW),
    )
    with pytest.raises(LibraryMigrationError, match="path and database identity"):
        migration_module._library_identity(connection, second.library_id)
    connection.execute("UPDATE libraries SET logical_identity_hash = ?", ("f" * 64,))
    with pytest.raises(LibraryMigrationError, match="logical identity hash is invalid"):
        migration_module._library_identity(connection, first.library_id)
    connection.execute(
        "INSERT INTO libraries VALUES (?, ?, ?, ?)",
        (second.library_id, second.name, library_logical_identity_hash(second), _NOW),
    )
    with pytest.raises(LibraryMigrationError, match="exactly one Library"):
        migration_module._library_identity(connection, first.library_id)

    connection.commit()
    outside_hash = migration_module._logical_database_hash(connection)
    connection.execute("BEGIN")
    assert migration_module._logical_database_hash(connection) == outside_hash
    assert connection.in_transaction is True
    connection.rollback()
    connection.close()

    _config, _data_home, paths, _source = _v3_library(tmp_path / "surfaces")
    linked = paths.blobs / "linked-directory"
    linked.symlink_to(paths.blobs, target_is_directory=True)
    with pytest.raises(LibraryMigrationError, match="symlink directory"):
        migration_module._source_surface_hashes(paths)
    linked.unlink()

    second_event_link = tmp_path / "events-hardlink"
    os.link(paths.events, second_event_link)
    with pytest.raises(LibraryMigrationError, match="not one regular file"):
        migration_module._source_surface_hashes(paths)
