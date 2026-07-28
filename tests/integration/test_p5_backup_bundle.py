"""Complete BackupBundle creation, verification, and restore acceptance tests."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from dithyramba.backup import (
    BackupBundleConflictError,
    BackupBundleIntegrityError,
    create_backup_bundle,
    restore_backup_bundle,
    verify_backup_bundle,
)
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import canonical_json_bytes
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import LibraryRepository, initialize_library


def _collection(repository: LibraryRepository, root: Path, name: str = "Corpus") -> str:
    return repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name=name,
            kind=CollectionKind.CORPUS,
            roots=(
                build_collection_root(
                    root,
                    data_root=repository.paths.application_data_root,
                ),
            ),
        )
    ).config.collection_id


def _populated_library(tmp_path: Path) -> tuple[LibraryConfig, Path, LibraryRepository]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "paper.md").write_text(
        "# Provenance\n\nExact source bytes survive a portable restore.\n",
        encoding="utf-8",
    )
    library = LibraryConfig(name="Portable research memory")
    repository = initialize_library(library, data_root=tmp_path / "source-data")
    collection_id = _collection(repository, source_root)
    IngestService(repository).ingest_path(collection_id, "paper.md")
    return library, source_root, repository


def _identity_rows(repository: LibraryRepository) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in repository._store.connection.execute(
            """
            SELECT sv.source_version_id, sv.content_sha256,
                   sf.source_fragment_id, sf.text_sha256, sf.address_hash
            FROM source_versions AS sv
            JOIN source_fragments AS sf ON sf.source_version_id = sv.source_version_id
            ORDER BY sv.source_version_id, sf.ordinal
            """
        ).fetchall()
    )


def test_complete_bundle_round_trip_preserves_identity_rows_events_and_blobs(
    tmp_path: Path,
) -> None:
    library, _source_root, repository = _populated_library(tmp_path)
    with repository:
        before_rows = _identity_rows(repository)
        before_counts = {
            str(row[0]): int(
                repository._store.connection.execute(
                    f'SELECT COUNT(*) FROM "{row[0]!s}"'
                ).fetchone()[0]
            )
            for row in repository._store.connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
        }
        bundle = create_backup_bundle(repository)
        verified = verify_backup_bundle(bundle.path)

        assert verified.manifest.library_id == library.library_id
        assert verified.manifest.row_counts == tuple(sorted(before_counts.items()))
        assert {entry.path for entry in verified.manifest.files} >= {
            "memory.sqlite3",
            "events.jsonl",
        }
        assert any(entry.path.startswith("blobs/") for entry in verified.manifest.files)
        assert verified.manifest.event_count > 0
        assert verified.manifest.last_event_id is not None

    with restore_backup_bundle(bundle.path, data_root=tmp_path / "restored-data") as restored:
        assert restored.library_id == library.library_id
        assert _identity_rows(restored) == before_rows
        assert {
            str(row[0]): int(
                restored._store.connection.execute(f'SELECT COUNT(*) FROM "{row[0]!s}"').fetchone()[
                    0
                ]
            )
            for row in restored._store.connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
        } == before_counts
        for row in restored._store.connection.execute(
            "SELECT content_sha256 FROM blobs ORDER BY content_sha256"
        ).fetchall():
            restored.get_blob(str(row[0]))
        restored_events = restored.paths.events.read_bytes()
        bundled_events = (bundle.path / "events.jsonl").read_bytes()
        assert restored_events == bundled_events


def test_restore_collision_fails_without_changing_existing_library(tmp_path: Path) -> None:
    library, _source_root, repository = _populated_library(tmp_path)
    with repository:
        bundle = create_backup_bundle(repository)
        rows_before = _identity_rows(repository)
        with pytest.raises(BackupBundleConflictError, match="already exists"):
            restore_backup_bundle(bundle.path, data_root=repository.paths.application_data_root)
        assert _identity_rows(repository) == rows_before
        assert repository.library_id == library.library_id


@pytest.mark.parametrize("relative_path", ["memory.sqlite3", "events.jsonl"])
def test_truncated_or_changed_core_file_fails_before_restore(
    tmp_path: Path,
    relative_path: str,
) -> None:
    _library, _source_root, repository = _populated_library(tmp_path)
    with repository:
        bundle = create_backup_bundle(repository)
    damaged = tmp_path / f"damaged-{relative_path.replace('.', '-')}"
    shutil.copytree(bundle.path, damaged)
    target = damaged / relative_path
    target.write_bytes(target.read_bytes()[: max(0, target.stat().st_size // 2)])
    target.chmod(0o600)

    with pytest.raises(BackupBundleIntegrityError, match=r"metadata differs|hash differs"):
        restore_backup_bundle(damaged, data_root=tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_blob_tamper_extra_path_and_permission_change_fail_closed(tmp_path: Path) -> None:
    _library, _source_root, repository = _populated_library(tmp_path)
    with repository:
        bundle = create_backup_bundle(repository)
    blob_entry = next(item for item in bundle.manifest.files if item.path.startswith("blobs/"))

    tampered = tmp_path / "tampered"
    shutil.copytree(bundle.path, tampered)
    blob = tampered / blob_entry.path
    blob.write_bytes(blob.read_bytes() + b"tamper")
    blob.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match=r"metadata differs|hash differs"):
        verify_backup_bundle(tampered)

    extra = tmp_path / "extra"
    shutil.copytree(bundle.path, extra)
    (extra / "unexpected.txt").write_text("not manifested", encoding="utf-8")
    (extra / "unexpected.txt").chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match=r"unsupported path|file set"):
        verify_backup_bundle(extra)

    permissive = tmp_path / "permissive"
    shutil.copytree(bundle.path, permissive)
    (permissive / "manifest.json").chmod(0o644)
    with pytest.raises(BackupBundleIntegrityError, match="0600"):
        verify_backup_bundle(permissive)


def test_manifest_path_escape_and_unknown_field_are_rejected(tmp_path: Path) -> None:
    _library, _source_root, repository = _populated_library(tmp_path)
    with repository:
        bundle = create_backup_bundle(repository)
    escaped = tmp_path / "escaped"
    shutil.copytree(bundle.path, escaped)
    manifest_path = escaped / "manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    payload["files"][0]["path"] = "../memory.sqlite3"
    manifest_path.write_bytes(canonical_json_bytes(payload))
    manifest_path.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match="contract"):
        verify_backup_bundle(escaped)

    unknown = tmp_path / "unknown"
    shutil.copytree(bundle.path, unknown)
    manifest_path = unknown / "manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    payload["automatic_promotion"] = True
    manifest_path.write_bytes(canonical_json_bytes(payload))
    manifest_path.chmod(0o600)
    with pytest.raises(BackupBundleIntegrityError, match="unknown fields"):
        verify_backup_bundle(unknown)


def test_symlink_and_hardlink_bundle_files_are_rejected(tmp_path: Path) -> None:
    _library, _source_root, repository = _populated_library(tmp_path)
    with repository:
        bundle = create_backup_bundle(repository)

    symlinked = tmp_path / "symlinked"
    shutil.copytree(bundle.path, symlinked)
    database = symlinked / "memory.sqlite3"
    real_database = symlinked / "real.sqlite3"
    database.rename(real_database)
    database.symlink_to(real_database)
    with pytest.raises(BackupBundleIntegrityError, match=r"regular file|unsupported path"):
        verify_backup_bundle(symlinked)

    hardlinked = tmp_path / "hardlinked"
    shutil.copytree(bundle.path, hardlinked)
    database = hardlinked / "memory.sqlite3"
    os.link(database, tmp_path / "second-link.sqlite3")
    with pytest.raises(BackupBundleIntegrityError, match="hardlinked"):
        verify_backup_bundle(hardlinked)
