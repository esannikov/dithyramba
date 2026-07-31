"""Adversarial branch coverage for complete BackupBundle invariants."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import dithyramba.backup.bundle as bundle_module
from dithyramba.backup import (
    BackupBundleConflictError,
    BackupBundleError,
    BackupBundleIntegrityError,
    BackupBundleManifest,
    BackupBundleRecord,
    BackupFileEntry,
    BackupMigrationEntry,
    create_backup_bundle,
    restore_backup_bundle,
    verify_backup_bundle,
)
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import canonical_json_bytes
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import LibraryRepository, initialize_library
from dithyramba.persistence.repository import library_logical_identity_hash, open_library

_ZERO_HASH = "0" * 64
_ONE_HASH = "1" * 64


def _manifest() -> BackupBundleManifest:
    return BackupBundleManifest(
        backup_bundle_id="backup_bundle_abcd",
        library_id="library_abcd",
        library_name="Research memory",
        logical_identity_hash=_ZERO_HASH,
        created_at="2026-07-21T10:11:12.123456Z",
        schema_version=1,
        schema_fingerprint=_ONE_HASH,
        migrations=(BackupMigrationEntry(1, "0001_initial.sql", _ZERO_HASH),),
        row_counts=(("libraries", 1),),
        last_event_id="event_abcd",
        event_count=1,
        files=(
            BackupFileEntry("events.jsonl", 0, _ZERO_HASH),
            BackupFileEntry("memory.sqlite3", 0, _ZERO_HASH),
        ),
    )


def _make_bundle(tmp_path: Path, *, populated: bool = False) -> BackupBundleRecord:
    repository = initialize_library(
        LibraryConfig(name="Backup edge test"), data_root=tmp_path / "source-data"
    )
    with repository:
        if populated:
            source_root = tmp_path / "source"
            source_root.mkdir()
            (source_root / "evidence.md").write_text(
                "# Evidence\n\nPortable provenance bytes.\n", encoding="utf-8"
            )
            collection = repository.create_collection(
                CollectionConfig(
                    library_id=repository.library_id,
                    name="Corpus",
                    kind=CollectionKind.CORPUS,
                    roots=(
                        build_collection_root(
                            source_root,
                            data_root=repository.paths.application_data_root,
                        ),
                    ),
                )
            )
            IngestService(repository).ingest_path(collection.config.collection_id, "evidence.md")
        return create_backup_bundle(repository)


def _read_manifest_payload(root: Path) -> dict[str, object]:
    value: object = json.loads((root / "manifest.json").read_bytes())
    assert type(value) is dict
    return cast(dict[str, object], value)


def _write_manifest_payload(root: Path, payload: dict[str, object]) -> None:
    path = root / "manifest.json"
    path.write_bytes(canonical_json_bytes(payload))
    path.chmod(0o600)


def _manifest_files(payload: dict[str, object]) -> list[dict[str, object]]:
    files = payload["files"]
    assert type(files) is list
    assert all(type(item) is dict for item in files)
    return cast(list[dict[str, object]], files)


def _refresh_manifested_file(root: Path, relative_path: str) -> None:
    payload = _read_manifest_payload(root)
    target = root / relative_path
    for entry in _manifest_files(payload):
        if entry["path"] == relative_path:
            data = target.read_bytes()
            entry["byte_size"] = len(data)
            entry["sha256"] = hashlib.sha256(data).hexdigest()
            entry["mode"] = 0o600
            _write_manifest_payload(root, payload)
            return
    raise AssertionError(f"missing manifest entry: {relative_path}")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BackupFileEntry("/absolute", 0, _ZERO_HASH),
        lambda: BackupFileEntry("../escape", 0, _ZERO_HASH),
        lambda: BackupFileEntry("nested//file", 0, _ZERO_HASH),
        lambda: BackupFileEntry(".", 0, _ZERO_HASH),
        lambda: BackupFileEntry("memory.sqlite3", -1, _ZERO_HASH),
        lambda: BackupFileEntry("memory.sqlite3", True, _ZERO_HASH),
        lambda: BackupFileEntry("memory.sqlite3", 0, "A" * 64),
        lambda: BackupFileEntry("memory.sqlite3", 0, cast(str, 12)),
        lambda: BackupFileEntry("memory.sqlite3", 0, _ZERO_HASH, 0o644),
        lambda: BackupFileEntry("memory.sqlite3", 0, _ZERO_HASH, cast(int, True)),
    ],
)
def test_backup_file_entry_rejects_noncanonical_or_unsafe_metadata(
    factory: Callable[[], object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BackupMigrationEntry(0, "0001_initial.sql", _ZERO_HASH),
        lambda: BackupMigrationEntry(cast(int, True), "0001_initial.sql", _ZERO_HASH),
        lambda: BackupMigrationEntry(1, "Initial.sql", _ZERO_HASH),
        lambda: BackupMigrationEntry(1, cast(str, 12), _ZERO_HASH),
        lambda: BackupMigrationEntry(1, "0001_initial.sql", "A" * 64),
        lambda: BackupMigrationEntry(1, "0001_initial.sql", cast(str, 12)),
    ],
)
def test_backup_migration_entry_rejects_invalid_contract_fields(
    factory: Callable[[], object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(_manifest(), backup_bundle_id="other_abcd"),
        lambda: replace(_manifest(), backup_bundle_id="backup_bundle_BAD"),
        lambda: replace(_manifest(), library_id="other_abcd"),
        lambda: replace(_manifest(), library_id="library_BAD"),
        lambda: replace(_manifest(), library_name="  "),
        lambda: replace(_manifest(), library_name=cast(str, 12)),
        lambda: replace(_manifest(), logical_identity_hash="bad"),
        lambda: replace(_manifest(), created_at="2026-07-21"),
        lambda: replace(_manifest(), created_at=cast(str, 12)),
        lambda: replace(_manifest(), schema_version=0),
        lambda: replace(_manifest(), schema_version=cast(int, True)),
        lambda: replace(_manifest(), schema_fingerprint="bad"),
        lambda: replace(_manifest(), schema_version=2),
        lambda: replace(_manifest(), row_counts=(("z", 0), ("a", 0))),
        lambda: replace(_manifest(), row_counts=(("a", 0), ("a", 1))),
        lambda: replace(_manifest(), row_counts=(("Bad-name", 0),)),
        lambda: replace(_manifest(), row_counts=(("a", cast(int, True)),)),
        lambda: replace(_manifest(), row_counts=(("a", -1),)),
        lambda: replace(_manifest(), last_event_id="other_abcd"),
        lambda: replace(_manifest(), last_event_id="event_BAD"),
        lambda: replace(_manifest(), event_count=-1),
        lambda: replace(_manifest(), event_count=cast(int, True)),
        lambda: replace(_manifest(), event_count=0),
        lambda: replace(_manifest(), last_event_id=None),
        lambda: replace(_manifest(), files=tuple(reversed(_manifest().files))),
        lambda: replace(_manifest(), files=(_manifest().files[0],) * 2),
        lambda: replace(_manifest(), files=(_manifest().files[0],)),
    ],
)
def test_backup_manifest_rejects_invalid_identity_cursor_rows_and_files(
    factory: Callable[[], object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_manifest_and_record_hashes_are_deterministic() -> None:
    manifest = _manifest()
    record = BackupBundleRecord(path=Path("/tmp/fixture"), manifest=manifest)

    assert manifest.canonical_bytes == canonical_json_bytes(manifest.payload())
    assert len(manifest.manifest_hash) == 64
    assert record.manifest_hash == manifest.manifest_hash


def test_create_rejects_wrong_type_open_transaction_and_unsafe_output(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="LibraryRepository"):
        create_backup_bundle(cast(LibraryRepository, object()))

    repository = initialize_library(LibraryConfig(name="Transactions"), data_root=tmp_path / "db")
    with repository:
        repository._store.connection.execute("BEGIN")
        with pytest.raises(BackupBundleError, match="inside a transaction"):
            create_backup_bundle(repository)
        repository._store.connection.rollback()

        relative = Path("relative-backups")
        with pytest.raises(BackupBundleError, match="absolute"):
            create_backup_bundle(repository, output_directory=relative)
        with pytest.raises(FileNotFoundError):
            create_backup_bundle(repository, output_directory=tmp_path / "missing")

        output = tmp_path / "public-output"
        output.mkdir(mode=0o700)
        output.chmod(0o755)
        with pytest.raises(BackupBundleIntegrityError, match="0700"):
            create_backup_bundle(repository, output_directory=output)


def test_create_uses_explicit_private_output_and_refuses_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = initialize_library(LibraryConfig(name="Explicit"), data_root=tmp_path / "db")
    output = tmp_path / "private-output"
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    with repository:
        record = create_backup_bundle(repository, output_directory=output)
        assert record.path.parent == output

        monkeypatch.setattr(bundle_module, "new_id", lambda _prefix: "backup_bundle_collision")
        collision = output / "backup_bundle_collision.dithyramba"
        collision.mkdir(mode=0o700)
        with pytest.raises(BackupBundleConflictError, match="already exists"):
            create_backup_bundle(repository, output_directory=output)


def test_create_failure_removes_staging_and_promoted_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = initialize_library(LibraryConfig(name="Atomic cleanup"), data_root=tmp_path / "db")
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    original_write = bundle_module._write_private_file

    def fail_events(path: Path, data: bytes) -> None:
        if path.name == "events.jsonl":
            raise OSError("injected write failure")
        original_write(path, data)

    with repository:
        monkeypatch.setattr(bundle_module, "_write_private_file", fail_events)
        with pytest.raises(OSError, match="injected"):
            create_backup_bundle(repository, output_directory=output)
    assert list(output.iterdir()) == []


def test_create_rejects_non_content_addressed_snapshot_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = initialize_library(LibraryConfig(name="Blob path"), data_root=tmp_path / "db")
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    original_snapshot = bundle_module._database_snapshot

    def corrupt_snapshot(connection: sqlite3.Connection) -> bundle_module._DatabaseSnapshot:
        clean = original_snapshot(connection)
        return bundle_module._DatabaseSnapshot(
            library_id=clean.library_id,
            library_name=clean.library_name,
            logical_identity_hash=clean.logical_identity_hash,
            schema_fingerprint_value=clean.schema_fingerprint,
            row_counts=clean.row_counts,
            blobs=((_ZERO_HASH, 0, "wrong/path"),),
        )

    with repository:
        monkeypatch.setattr(bundle_module, "_database_snapshot", corrupt_snapshot)
        with pytest.raises(BackupBundleIntegrityError, match="content-addressed"):
            create_backup_bundle(repository, output_directory=output)
    assert list(output.iterdir()) == []


def test_create_detects_post_promotion_manifest_mismatch_and_cleans_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = initialize_library(LibraryConfig(name="Promotion"), data_root=tmp_path / "db")
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    original_verify = bundle_module.verify_backup_bundle

    def mismatching_verify(path: str | os.PathLike[str]) -> BackupBundleRecord:
        record = original_verify(path)
        return BackupBundleRecord(
            path=record.path,
            manifest=replace(record.manifest, library_name="Different valid name"),
        )

    with repository:
        monkeypatch.setattr(bundle_module, "verify_backup_bundle", mismatching_verify)
        with pytest.raises(BackupBundleIntegrityError, match="differs from its manifest"):
            create_backup_bundle(repository, output_directory=output)
    assert list(output.iterdir()) == []


def test_verify_rejects_path_root_and_manifest_container_failures(tmp_path: Path) -> None:
    with pytest.raises(BackupBundleIntegrityError, match="absolute"):
        verify_backup_bundle("relative")
    with pytest.raises(BackupBundleIntegrityError, match="absolute"):
        verify_backup_bundle(tmp_path / "child" / ".." / "bundle")
    with pytest.raises(BackupBundleIntegrityError, match="missing"):
        verify_backup_bundle(tmp_path / "missing")

    plain_file = tmp_path / "plain"
    plain_file.write_text("x", encoding="utf-8")
    plain_file.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match="real directory"):
        verify_backup_bundle(plain_file)

    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    with pytest.raises(BackupBundleIntegrityError, match="missing"):
        verify_backup_bundle(root)

    manifest = root / "manifest.json"
    manifest.write_bytes(b"[]")
    manifest.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match="one object"):
        verify_backup_bundle(root)
    manifest.write_bytes(b"{not-json")
    with pytest.raises(BackupBundleIntegrityError, match="valid UTF-8 JSON"):
        verify_backup_bundle(root)
    manifest.write_bytes(b"\xff")
    with pytest.raises(BackupBundleIntegrityError, match="valid UTF-8 JSON"):
        verify_backup_bundle(root)


def test_verify_rejects_noncanonical_and_oversized_manifest(tmp_path: Path) -> None:
    record = _make_bundle(tmp_path)
    payload = _read_manifest_payload(record.path)
    manifest_path = record.path / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest_path.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match="not canonical"):
        verify_backup_bundle(record.path)

    manifest_path.write_bytes(b" " * (8 * 1024 * 1024 + 1))
    manifest_path.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match="safe limit"):
        verify_backup_bundle(record.path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(migrations={}), "contract"),
        (lambda value: value.update(row_counts=[]), "contract"),
        (lambda value: value.update(files={}), "contract"),
        (lambda value: value.update(migrations=[{"version": 1}]), "contract"),
        (lambda value: value.update(migrations=["bad"]), "contract"),
        (
            lambda value: value.update(
                migrations=[{"version": "1", "name": "0001_initial.sql", "sha256": _ZERO_HASH}]
            ),
            "contract",
        ),
        (
            lambda value: value.update(
                migrations=[{"version": 1, "name": 1, "sha256": _ZERO_HASH}]
            ),
            "contract",
        ),
        (
            lambda value: value.update(
                migrations=[{"version": 1, "name": "0001_initial.sql", "sha256": 1}]
            ),
            "contract",
        ),
        (lambda value: value.update(row_counts={"libraries": "1"}), "contract"),
        (lambda value: value.update(files=[{"path": "events.jsonl"}]), "contract"),
        (lambda value: value.update(backup_bundle_id=1), "contract"),
        (lambda value: value.update(schema_version="1"), "contract"),
        (lambda value: value.update(last_event_id=1), "contract"),
        (lambda value: value.update(schema="wrong-schema"), "typed reconstruction"),
    ],
)
def test_manifest_parser_rejects_exact_shape_and_type_violations(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    record = _make_bundle(tmp_path)
    payload = _read_manifest_payload(record.path)
    mutation(payload)
    _write_manifest_payload(record.path, payload)

    with pytest.raises(BackupBundleIntegrityError, match=message):
        verify_backup_bundle(record.path)


def test_private_manifest_helpers_reject_nonarrays_and_bad_scalar_types() -> None:
    with pytest.raises(ValueError, match="array"):
        bundle_module._exact_object_items({}, {"field"}, "values")
    with pytest.raises(ValueError, match="invalid fields"):
        bundle_module._exact_object_items([{"other": 1}], {"field"}, "values")
    with pytest.raises(ValueError, match="must be text"):
        bundle_module._exact_str({"field": 1}, "field")
    with pytest.raises(ValueError, match="text or null"):
        bundle_module._optional_str({"field": 1}, "field")
    with pytest.raises(ValueError, match="integer"):
        bundle_module._exact_int({"field": True}, "field")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema_version", 13, "schema version"),
        ("migrations", "different", "migrations differ"),
        ("library_name", "Changed name", "identity/schema/rows"),
        ("event_count", "increment", "event cursor"),
    ],
)
def test_verify_compares_manifest_to_database_semantics(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    record = _make_bundle(tmp_path)
    payload = _read_manifest_payload(record.path)
    migrations = payload["migrations"]
    assert type(migrations) is list
    if field == "schema_version":
        payload["schema_version"] = 13
        migrations.append({"version": 13, "name": "0013_future.sql", "sha256": _ZERO_HASH})
    elif field == "migrations":
        first = cast(dict[str, object], migrations[0])
        first["sha256"] = _ONE_HASH if first["sha256"] != _ONE_HASH else _ZERO_HASH
    elif field == "event_count":
        count = payload["event_count"]
        assert type(count) is int
        payload["event_count"] = count + 1
    else:
        payload[field] = replacement
    _write_manifest_payload(record.path, payload)

    with pytest.raises(BackupBundleIntegrityError, match=message):
        verify_backup_bundle(record.path)


def test_verify_detects_same_size_hash_change_and_event_projection_drift(tmp_path: Path) -> None:
    record = _make_bundle(tmp_path)
    events = record.path / "events.jsonl"
    original = events.read_bytes()
    events.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    events.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match="hash differs"):
        verify_backup_bundle(record.path)

    _refresh_manifested_file(record.path, "events.jsonl")
    with pytest.raises(BackupBundleIntegrityError, match="differs from the database"):
        verify_backup_bundle(record.path)


def test_verify_detects_manifested_blob_not_referenced_by_database(tmp_path: Path) -> None:
    record = _make_bundle(tmp_path)
    fake_digest = "f" * 64
    relative = f"blobs/{fake_digest[:2]}/{fake_digest[2:]}"
    fake = record.path / relative
    fake.parent.mkdir(parents=True, mode=0o700)
    bundle_module._secure_directory_chain(fake.parent, record.path)
    fake.write_bytes(b"")
    fake.chmod(0o600)
    payload = _read_manifest_payload(record.path)
    files = _manifest_files(payload)
    files.append(
        {"path": relative, "byte_size": 0, "sha256": hashlib.sha256(b"").hexdigest(), "mode": 0o600}
    )
    files.sort(key=lambda item: cast(str, item["path"]))
    _write_manifest_payload(record.path, payload)

    with pytest.raises(BackupBundleIntegrityError, match="Blob set differs"):
        verify_backup_bundle(record.path)


def test_verify_detects_manifest_file_set_gap_and_non_directory_path(tmp_path: Path) -> None:
    record = _make_bundle(tmp_path)
    payload = _read_manifest_payload(record.path)
    files = _manifest_files(payload)
    files[:] = [entry for entry in files if entry["path"] != "events.jsonl"]
    _write_manifest_payload(record.path, payload)
    with pytest.raises(BackupBundleIntegrityError, match="contract"):
        verify_backup_bundle(record.path)

    record = _make_bundle(tmp_path / "second")
    bad_directory = record.path / "blobs"
    bad_directory.write_bytes(b"not a directory")
    bad_directory.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match="unsupported path"):
        verify_backup_bundle(record.path)


def test_database_snapshot_rejects_cardinality_identity_blob_and_table_corruption() -> None:
    empty = sqlite3.connect(":memory:")
    empty.execute("CREATE TABLE libraries(library_id TEXT, name TEXT, logical_identity_hash TEXT)")
    empty.execute("CREATE TABLE blobs(content_sha256 TEXT, byte_size INTEGER, relative_path TEXT)")
    with pytest.raises(BackupBundleIntegrityError, match="exactly one"):
        bundle_module._database_snapshot(empty)
    empty.close()

    bad_identity = sqlite3.connect(":memory:")
    bad_identity.execute(
        "CREATE TABLE libraries(library_id TEXT, name TEXT, logical_identity_hash TEXT)"
    )
    bad_identity.execute(
        "CREATE TABLE blobs(content_sha256 TEXT, byte_size INTEGER, relative_path TEXT)"
    )
    valid_id = LibraryConfig(name="Identity").library_id
    bad_identity.execute(
        "INSERT INTO libraries VALUES (?, ?, ?)", (valid_id, "Identity", _ZERO_HASH)
    )
    with pytest.raises(BackupBundleIntegrityError, match="identity hash"):
        bundle_module._database_snapshot(bad_identity)
    bad_identity.close()

    blobs = sqlite3.connect(":memory:")
    blobs.execute("CREATE TABLE blobs(content_sha256 TEXT, byte_size INTEGER, relative_path TEXT)")
    blobs.execute("INSERT INTO blobs VALUES (?, ?, ?)", ("bad", 1, "ba/d"))
    with pytest.raises(BackupBundleIntegrityError, match="metadata"):
        bundle_module._blob_rows(blobs)
    blobs.execute("DELETE FROM blobs")
    blobs.execute("INSERT INTO blobs VALUES (?, ?, ?)", (_ZERO_HASH, 1, "wrong/path"))
    with pytest.raises(BackupBundleIntegrityError, match="relative path"):
        bundle_module._blob_rows(blobs)
    blobs.close()

    tables = sqlite3.connect(":memory:")
    tables.execute('CREATE TABLE "Bad-Name"(value INTEGER)')
    with pytest.raises(BackupBundleIntegrityError, match="table name"):
        bundle_module._row_counts(tables)
    tables.close()


@pytest.mark.parametrize(
    ("payload", "payload_hash", "message"),
    [
        ("not-json", _ZERO_HASH, "not valid JSON"),
        ("[]", _ZERO_HASH, "must be an object"),
        ('{"b":1, "a":2}', _ZERO_HASH, "not canonical JSON"),
        ("{}", _ZERO_HASH, "payload hash"),
    ],
)
def test_event_projection_rejects_noncanonical_or_hash_mismatched_payload(
    payload: str,
    payload_hash: str,
    message: str,
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE event_outbox(
            event_id TEXT, event_type TEXT, aggregate_type TEXT, aggregate_id TEXT,
            payload_json TEXT, payload_hash TEXT, occurred_at TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO event_outbox VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("event_abcd", "test", "library", "library_abcd", payload, payload_hash, "now"),
    )
    with pytest.raises(BackupBundleIntegrityError, match=message):
        bundle_module._event_log_bytes(connection)
    connection.close()


def test_filesystem_guards_reject_links_permissions_descriptors_and_escape(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(BackupBundleIntegrityError, match="missing"):
        bundle_module._require_private_file(missing, "fixture")

    directory = tmp_path / "directory"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    with pytest.raises(BackupBundleIntegrityError, match="regular file"):
        bundle_module._require_private_file(directory, "fixture")

    private = tmp_path / "private"
    private.write_bytes(b"x")
    private.chmod(0o600)
    public = tmp_path / "public"
    public.write_bytes(b"x")
    public.chmod(0o644)
    with pytest.raises(BackupBundleIntegrityError, match="0600"):
        bundle_module._require_private_file(public, "fixture")

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        with pytest.raises(BackupBundleIntegrityError, match="one regular file"):
            bundle_module._require_private_descriptor(descriptor, "fixture")
    finally:
        os.close(descriptor)
    descriptor = os.open(public, os.O_RDONLY)
    try:
        with pytest.raises(BackupBundleIntegrityError, match="not private"):
            bundle_module._require_private_descriptor(descriptor, "fixture")
    finally:
        os.close(descriptor)

    boundary = tmp_path / "boundary"
    boundary.mkdir(mode=0o700)
    boundary.chmod(0o700)
    outsider = tmp_path / "outsider"
    outsider.mkdir(mode=0o700)
    outsider.chmod(0o700)
    with pytest.raises(BackupBundleIntegrityError, match="escaped"):
        bundle_module._secure_directory_chain(outsider, boundary)


def test_copy_hash_read_and_write_fail_closed_and_remove_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write_all = bundle_module._write_all
    source = tmp_path / "source"
    source.write_bytes(b"source bytes")
    source.chmod(0o600)
    destination = tmp_path / "destination"
    with pytest.raises(BackupBundleIntegrityError, match="differs"):
        bundle_module._copy_exact_private_file(
            source,
            destination,
            expected_sha256=_ZERO_HASH,
            expected_size=source.stat().st_size,
        )
    assert not destination.exists()

    def fail_write(_descriptor: int, _data: bytes) -> None:
        raise OSError("injected copy failure")

    monkeypatch.setattr(bundle_module, "_write_all", fail_write)
    with pytest.raises(OSError, match="injected"):
        bundle_module._copy_exact_private_file(
            source,
            destination,
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_size=source.stat().st_size,
        )
    assert not destination.exists()

    monkeypatch.setattr(bundle_module, "_write_all", original_write_all)
    monkeypatch.setattr(os, "write", lambda _fd, _data: 0)
    with pytest.raises(BackupBundleError, match="no progress"):
        bundle_module._write_all(1, b"x")


def test_read_and_hash_detect_size_limit_and_concurrent_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "private"
    path.write_bytes(b"abcdef")
    path.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match="safe limit"):
        bundle_module._read_private_file(path, maximum=5)

    original_fstat = os.fstat
    calls = 0

    def changed_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        result = original_fstat(descriptor)
        if calls == 2:
            values = list(result)
            values[8] = result.st_mtime + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(os, "fstat", changed_fstat)
    with pytest.raises(BackupBundleIntegrityError, match="changed while hashing"):
        bundle_module._file_sha256(path)


def test_bounded_read_detects_growth_after_initial_metadata_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "private"
    path.write_bytes(b"x")
    path.chmod(0o600)
    calls = 0

    def growing_read(_descriptor: int, _count: int) -> bytes:
        nonlocal calls
        calls += 1
        return b"xx" if calls == 1 else b""

    monkeypatch.setattr(os, "read", growing_read)
    with pytest.raises(BackupBundleIntegrityError, match="safe limit"):
        bundle_module._read_private_file(path, maximum=1)


def test_copy_destination_open_failure_has_no_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"source")
    source.chmod(0o600)
    destination = tmp_path / "destination"
    original_open = os.open

    def fail_destination_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes], flags: int, mode: int = 0o777
    ) -> int:
        if path == destination:
            raise PermissionError("injected destination-open failure")
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", fail_destination_open)
    with pytest.raises(PermissionError, match="injected"):
        bundle_module._copy_exact_private_file(
            source,
            destination,
            expected_sha256=hashlib.sha256(b"source").hexdigest(),
            expected_size=6,
        )
    assert not destination.exists()


def test_walk_rejects_symlink_directory_and_duplicate_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    target.chmod(0o700)
    (root / "linked").symlink_to(target)
    with pytest.raises(BackupBundleIntegrityError, match="non-directory"):
        bundle_module._walk_bundle_files(root)

    (root / "linked").unlink()
    file = root / "events.jsonl"
    file.write_bytes(b"")
    file.chmod(0o600)

    def duplicate_walk(
        _root: Path, *, topdown: bool, followlinks: bool
    ) -> list[tuple[str, list[str], list[str]]]:
        assert topdown is True
        assert followlinks is False
        return [(str(root), [], [file.name, file.name])]

    monkeypatch.setattr(os, "walk", duplicate_walk)
    with pytest.raises(BackupBundleIntegrityError, match="duplicate"):
        bundle_module._walk_bundle_files(root)


def test_fsync_directory_is_portably_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(os, "O_DIRECTORY")
    bundle_module._fsync_directory(tmp_path)


def test_restore_rejects_inconsistent_verified_identity_before_creating_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _make_bundle(tmp_path)
    inconsistent = BackupBundleRecord(
        path=record.path,
        manifest=replace(record.manifest, logical_identity_hash=_ZERO_HASH),
    )
    monkeypatch.setattr(bundle_module, "verify_backup_bundle", lambda _path: inconsistent)
    destination = tmp_path / "restore"
    with pytest.raises(BackupBundleIntegrityError, match="logical identity"):
        restore_backup_bundle(record.path, data_root=destination)
    assert not destination.exists()


def test_restore_failure_after_layout_closes_repository_and_removes_new_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _make_bundle(tmp_path)
    destination = tmp_path / "restore"
    closed = False

    class BadRestoredRepository:
        schema_fingerprint = _ZERO_HASH

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(
        bundle_module,
        "open_library",
        lambda _library_id, *, data_root: cast(LibraryRepository, BadRestoredRepository()),
    )
    with pytest.raises(BackupBundleIntegrityError, match="schema fingerprint"):
        restore_backup_bundle(record.path, data_root=destination)
    assert closed
    assert not (destination / "libraries" / record.manifest.library_id).exists()


def test_restore_copy_failure_removes_only_new_library_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _make_bundle(tmp_path)
    destination = tmp_path / "restore"

    def fail_copy(
        _source: Path,
        _destination: Path,
        *,
        expected: BackupFileEntry,
    ) -> None:
        assert isinstance(expected, BackupFileEntry)
        raise OSError("injected restore copy failure")

    monkeypatch.setattr(bundle_module, "_replace_from_verified_file", fail_copy)
    with pytest.raises(OSError, match="injected"):
        restore_backup_bundle(record.path, data_root=destination)
    assert not (destination / "libraries" / record.manifest.library_id).exists()


@pytest.mark.parametrize("relative_path", ["memory.sqlite3", "events.jsonl"])
def test_restore_rejects_core_file_changed_after_bundle_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    record = _make_bundle(tmp_path)
    destination = tmp_path / "restore"
    original_verify = bundle_module.verify_backup_bundle

    def verify_then_mutate(path: str | os.PathLike[str]) -> BackupBundleRecord:
        verified = original_verify(path)
        target = verified.path / relative_path
        with target.open("ab") as stream:
            stream.write(b"post-verification mutation\n")
        target.chmod(0o600)
        return verified

    monkeypatch.setattr(bundle_module, "verify_backup_bundle", verify_then_mutate)
    with pytest.raises(BackupBundleIntegrityError, match="copied file differs"):
        restore_backup_bundle(record.path, data_root=destination)
    assert not (destination / "libraries" / record.manifest.library_id).exists()


def test_restore_revalidates_installed_event_projection_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _make_bundle(tmp_path)
    destination = tmp_path / "restore"
    original_open = open_library
    opened: LibraryRepository | None = None

    def mutate_after_install(library_id: str, *, data_root: Path) -> LibraryRepository:
        nonlocal opened
        opened = original_open(library_id, data_root=data_root)
        with opened.paths.events.open("ab") as stream:
            stream.write(b'{"post_install":"mutation"}\n')
        return opened

    monkeypatch.setattr(bundle_module, "open_library", mutate_after_install)
    with pytest.raises(BackupBundleIntegrityError, match="differs from database"):
        restore_backup_bundle(record.path, data_root=destination)
    assert opened is not None
    assert opened._store._connection is None
    assert not (destination / "libraries" / record.manifest.library_id).exists()


def test_restore_revalidates_manifest_event_cursor_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _make_bundle(tmp_path)
    inconsistent = BackupBundleRecord(
        path=record.path,
        manifest=replace(record.manifest, event_count=record.manifest.event_count + 1),
    )
    monkeypatch.setattr(bundle_module, "verify_backup_bundle", lambda _path: inconsistent)
    destination = tmp_path / "restore"

    with pytest.raises(BackupBundleIntegrityError, match="event cursor"):
        restore_backup_bundle(record.path, data_root=destination)
    assert not (destination / "libraries" / record.manifest.library_id).exists()


def test_restore_layout_failure_before_path_assignment_propagates_without_cleanup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _make_bundle(tmp_path)

    def fail_layout(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected layout failure")

    monkeypatch.setattr(bundle_module, "create_library_layout", fail_layout)
    with pytest.raises(OSError, match="injected layout"):
        restore_backup_bundle(record.path, data_root=tmp_path / "restore")


def test_restore_forwards_declared_roots_and_verifies_blob_closure(tmp_path: Path) -> None:
    record = _make_bundle(tmp_path, populated=True)
    source_root = tmp_path / "declared-source"
    source_root.mkdir()
    sync_root = tmp_path / "declared-sync"
    sync_root.mkdir()
    destination = tmp_path / "restore"

    with restore_backup_bundle(
        record.path,
        data_root=destination,
        declared_source_roots=(source_root,),
        declared_sync_roots=(sync_root,),
    ) as restored:
        blob = restored._store.connection.execute(
            "SELECT content_sha256 FROM blobs ORDER BY content_sha256 LIMIT 1"
        ).fetchone()
        assert blob is not None
        assert restored.get_blob(str(blob[0])).path.is_file()


def test_absolute_output_rejects_symlink_and_missing_directory(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    target.chmod(0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    assert bundle_module._absolute_existing_private_directory(link) == target
    with pytest.raises(FileNotFoundError):
        bundle_module._absolute_existing_private_directory(tmp_path / "missing")


def test_directory_and_file_guards_reject_symlinks_and_public_root(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    public_root.mkdir(mode=0o700)
    public_root.chmod(0o755)
    with pytest.raises(BackupBundleIntegrityError, match="0700"):
        bundle_module._require_private_directory(public_root, "fixture")

    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o700)
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(private_root)
    with pytest.raises(BackupBundleIntegrityError, match="real directory"):
        bundle_module._require_private_directory(directory_link, "fixture")

    target = tmp_path / "target-file"
    target.write_bytes(b"x")
    target.chmod(0o600)
    file_link = tmp_path / "file-link"
    file_link.symlink_to(target)
    with pytest.raises(BackupBundleIntegrityError, match="regular file"):
        bundle_module._require_private_file(file_link, "fixture")


def test_canonical_object_accepts_only_canonical_json_objects() -> None:
    assert bundle_module._canonical_object("{}", "fixture") == {}
    with pytest.raises(BackupBundleIntegrityError, match="valid JSON"):
        bundle_module._canonical_object("{", "fixture")
    with pytest.raises(BackupBundleIntegrityError, match="object"):
        bundle_module._canonical_object("[]", "fixture")
    with pytest.raises(BackupBundleIntegrityError, match="canonical"):
        bundle_module._canonical_object('{"b":1, "a":2}', "fixture")


def test_database_snapshot_exposes_exact_values_for_valid_minimal_schema() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE libraries(library_id TEXT, name TEXT, logical_identity_hash TEXT)"
    )
    connection.execute(
        "CREATE TABLE blobs(content_sha256 TEXT, byte_size INTEGER, relative_path TEXT)"
    )
    config = LibraryConfig(name="Minimal")
    identity = library_logical_identity_hash(config)
    connection.execute(
        "INSERT INTO libraries VALUES (?, ?, ?)", (config.library_id, config.name, identity)
    )
    snapshot = bundle_module._database_snapshot(connection)
    assert snapshot.library_id == config.library_id
    assert snapshot.library_name == config.name
    assert snapshot.logical_identity_hash == identity
    assert snapshot.blobs == ()
    assert dict(snapshot.row_counts) == {"blobs": 0, "libraries": 1}
    connection.close()


def test_read_private_file_detects_short_read_after_metadata_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "private"
    path.write_bytes(b"abcdef")
    path.chmod(0o600)
    original_read = os.read
    first = True

    def truncating_read(descriptor: int, count: int) -> bytes:
        nonlocal first
        if first:
            first = False
            path.write_bytes(b"")
            path.chmod(0o600)
        return original_read(descriptor, count)

    monkeypatch.setattr(os, "read", truncating_read)
    with pytest.raises(BackupBundleIntegrityError, match="changed while reading"):
        bundle_module._read_private_file(path)


def test_restore_row_count_mismatch_closes_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _make_bundle(tmp_path)
    destination = tmp_path / "restore"
    original_open = open_library
    original_counts = bundle_module._row_counts
    opened: LibraryRepository | None = None

    def track_open(library_id: str, *, data_root: Path) -> LibraryRepository:
        nonlocal opened
        opened = original_open(library_id, data_root=data_root)
        return opened

    def drift_counts(connection: sqlite3.Connection) -> tuple[tuple[str, int], ...]:
        counts = original_counts(connection)
        if opened is not None and connection is opened._store.connection:
            return (*counts, ("synthetic_drift", 1))
        return counts

    monkeypatch.setattr(bundle_module, "open_library", track_open)
    monkeypatch.setattr(bundle_module, "_row_counts", drift_counts)
    with pytest.raises(BackupBundleIntegrityError, match="table counts"):
        restore_backup_bundle(record.path, data_root=destination)
    assert opened is not None
    assert opened._store._connection is None
    assert not (destination / "libraries" / record.manifest.library_id).exists()


def test_restore_missing_blob_is_detected_after_open_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _make_bundle(tmp_path, populated=True)
    destination = tmp_path / "restore"
    original_open = open_library

    def delete_restored_blob(library_id: str, *, data_root: Path) -> LibraryRepository:
        repository = original_open(library_id, data_root=data_root)
        row = repository._store.connection.execute(
            "SELECT relative_path FROM blobs ORDER BY content_sha256 LIMIT 1"
        ).fetchone()
        assert row is not None
        (repository.paths.blobs / str(row[0])).unlink()
        return repository

    monkeypatch.setattr(bundle_module, "open_library", delete_restored_blob)
    with pytest.raises(Exception, match=r"missing|verified|exist"):
        restore_backup_bundle(record.path, data_root=destination)
    assert not (destination / "libraries" / record.manifest.library_id).exists()


def test_verify_rejects_database_with_corrupt_event_hash_after_manifest_rehash(
    tmp_path: Path,
) -> None:
    record = _make_bundle(tmp_path)
    database = record.path / "memory.sqlite3"
    connection = sqlite3.connect(database)
    triggers = connection.execute(
        "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger' ORDER BY name"
    ).fetchall()
    for name, _sql in triggers:
        connection.execute(f'DROP TRIGGER "{name!s}"')
    connection.execute("UPDATE event_outbox SET payload_hash = ?", (_ZERO_HASH,))
    for _name, sql in triggers:
        assert type(sql) is str
        connection.execute(sql)
    connection.commit()
    connection.close()
    database.chmod(0o600)
    _refresh_manifested_file(record.path, "memory.sqlite3")
    with pytest.raises(BackupBundleIntegrityError, match="payload hash"):
        verify_backup_bundle(record.path)


def test_verify_rejects_database_identity_corruption_after_manifest_rehash(
    tmp_path: Path,
) -> None:
    record = _make_bundle(tmp_path)
    database = record.path / "memory.sqlite3"
    connection = sqlite3.connect(database)
    triggers = connection.execute(
        "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger' ORDER BY name"
    ).fetchall()
    for name, _sql in triggers:
        connection.execute(f'DROP TRIGGER "{name!s}"')
    connection.execute("UPDATE libraries SET logical_identity_hash = ?", (_ZERO_HASH,))
    for _name, sql in triggers:
        assert type(sql) is str
        connection.execute(sql)
    connection.commit()
    connection.close()
    database.chmod(0o600)
    _refresh_manifested_file(record.path, "memory.sqlite3")
    with pytest.raises(BackupBundleIntegrityError, match="identity hash"):
        verify_backup_bundle(record.path)


def test_verify_rejects_missing_manifested_file(tmp_path: Path) -> None:
    record = _make_bundle(tmp_path)
    (record.path / "events.jsonl").unlink()
    with pytest.raises(BackupBundleIntegrityError, match="file set"):
        verify_backup_bundle(record.path)


def test_copy_rejects_hardlinked_source_descriptor(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"bytes")
    source.chmod(0o600)
    os.link(source, tmp_path / "second-link")
    with pytest.raises(BackupBundleIntegrityError, match="hardlinked"):
        bundle_module._copy_exact_private_file(
            source,
            tmp_path / "destination",
            expected_sha256=hashlib.sha256(b"bytes").hexdigest(),
            expected_size=5,
        )


def test_copy_exclusive_create_never_removes_a_preexisting_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"new bytes")
    source.chmod(0o600)
    destination = tmp_path / "destination"
    destination.write_bytes(b"preserve me")
    destination.chmod(0o600)

    with pytest.raises(FileExistsError):
        bundle_module._copy_exact_private_file(
            source,
            destination,
            expected_sha256=hashlib.sha256(b"new bytes").hexdigest(),
            expected_size=len(b"new bytes"),
        )

    assert destination.read_bytes() == b"preserve me"


def test_restore_existing_plain_path_is_reported_as_conflict(tmp_path: Path) -> None:
    record = _make_bundle(tmp_path)
    data_root = tmp_path / "restore"
    library_root = data_root / "libraries" / record.manifest.library_id
    library_root.mkdir(parents=True)
    with pytest.raises(BackupBundleConflictError, match="already exists"):
        restore_backup_bundle(record.path, data_root=data_root)


def test_internal_snapshot_container_assigns_all_slots() -> None:
    snapshot = bundle_module._DatabaseSnapshot(
        library_id="library_abcd",
        library_name="Name",
        logical_identity_hash=_ZERO_HASH,
        schema_fingerprint_value=_ONE_HASH,
        row_counts=(("libraries", 1),),
        blobs=((_ZERO_HASH, 0, f"{_ZERO_HASH[:2]}/{_ZERO_HASH[2:]}"),),
    )
    assert (
        SimpleNamespace(
            library_id=snapshot.library_id,
            library_name=snapshot.library_name,
            logical_identity_hash=snapshot.logical_identity_hash,
            schema_fingerprint=snapshot.schema_fingerprint,
            row_counts=snapshot.row_counts,
            blobs=snapshot.blobs,
        ).library_id
        == "library_abcd"
    )
