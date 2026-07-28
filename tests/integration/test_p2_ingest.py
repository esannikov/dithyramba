from __future__ import annotations

import json
import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import sha256_hex
from dithyramba.ingest.models import ParseResult, ParserLimits, SourceBytes
from dithyramba.ingest.parsers import parse_source
from dithyramba.ingest.reader import read_source_bytes
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    CollectionRecord,
    LibraryRepository,
    PersistenceConflictError,
    ProcessingRunStatus,
    SourceFamilyRole,
    initialize_library,
    open_library,
)
from dithyramba.provenance import (
    BlobStore,
    IngestBatchResult,
    IngestDisposition,
    TerminalInputOutcome,
)

NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)


def fixed_clock() -> datetime:
    return NOW


def _collection(
    repository: LibraryRepository,
    source_root: Path,
    *,
    name: str = "Corpus",
    kind: CollectionKind = CollectionKind.CORPUS,
) -> CollectionRecord:
    return repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name=name,
            kind=kind,
            roots=(
                build_collection_root(
                    source_root,
                    data_root=repository.paths.application_data_root,
                ),
            ),
        )
    )


def _counts(repository: LibraryRepository) -> dict[str, int]:
    connection = repository._store.connection
    return {
        table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        for table in (
            "blobs",
            "sources",
            "source_families",
            "source_family_members",
            "source_versions",
            "source_heads",
            "source_fragments",
            "collection_memberships",
        )
    }


def test_add_unchanged_change_and_revert_preserve_immutable_history(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    path = source_root / "paper.md"
    original = "# Thesis\n\nFirst formulation.\n"
    changed = "# Thesis\n\nRevised formulation.\n"
    path.write_text(original, encoding="utf-8")
    library = LibraryConfig(name="Version history")

    with initialize_library(
        library,
        data_root=tmp_path / "data",
        clock=fixed_clock,
    ) as repository:
        collection = _collection(repository, source_root)
        service = IngestService(repository)

        added = service.ingest_collection(collection.config.collection_id)
        first = added.outcomes[0]
        assert first.disposition is IngestDisposition.ADDED
        after_added = _counts(repository)
        first_version_id = first.source_version_id

        unchanged = service.ingest_collection(collection.config.collection_id)
        assert unchanged.outcomes[0].disposition is IngestDisposition.UNCHANGED
        assert unchanged.outcomes[0].source_version_id == first_version_id
        assert _counts(repository) == after_added

        path.write_text(changed, encoding="utf-8")
        changed_result = service.ingest_collection(collection.config.collection_id)
        second = changed_result.outcomes[0]
        assert second.disposition is IngestDisposition.CHANGED
        assert second.source_version_id != first_version_id
        assert _counts(repository)["source_versions"] == 2

        path.write_text(original, encoding="utf-8")
        reverted = service.ingest_collection(collection.config.collection_id)
        assert reverted.outcomes[0].disposition is IngestDisposition.REVERTED
        assert reverted.outcomes[0].source_version_id == first_version_id
        assert _counts(repository)["source_versions"] == 2

        source = repository.get_source(first.source_id or "")
        versions = repository.list_source_versions(source.source_id)
        assert source.current_source_version_id == first_version_id
        assert tuple(version.is_current for version in versions) == (True, False)
        assert tuple(version.version_number for version in versions) == (1, 2)
        assert any(
            event.event_type == "source_head.reverted"
            for event in repository.pending_outbox_events()
        )


def test_duplicate_bytes_at_new_uri_share_blob_and_root_family(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    payload = "# Shared\n\nSame exact bytes.\n"
    (source_root / "a.md").write_text(payload, encoding="utf-8")
    (source_root / "b.md").write_text(payload, encoding="utf-8")
    library = LibraryConfig(name="Duplicates")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection = _collection(repository, source_root)
        result = IngestService(repository).ingest_collection(collection.config.collection_id)

        assert tuple(outcome.disposition for outcome in result.outcomes) == (
            IngestDisposition.ADDED,
            IngestDisposition.ADDED,
        )
        sources = repository.list_sources(collection_id=collection.config.collection_id)
        by_name = {Path(source.canonical_uri).name: source for source in sources}
        root = by_name["a.md"]
        duplicate = by_name["b.md"]
        assert root.family_role is SourceFamilyRole.ROOT
        assert duplicate.family_role is SourceFamilyRole.DUPLICATE
        assert duplicate.source_family_id == root.source_family_id
        assert duplicate.root_source_id == root.source_id
        assert _counts(repository) == {
            "blobs": 1,
            "sources": 2,
            "source_families": 1,
            "source_family_members": 2,
            "source_versions": 2,
            "source_heads": 2,
            "source_fragments": 4,
            "collection_memberships": 2,
        }


def test_batch_succeeds_with_explicit_skip_and_failure_and_no_provenance(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "empty.md").write_text("\n\n", encoding="utf-8")
    (source_root / "invalid.txt").write_bytes(b"\xff")
    library = LibraryConfig(name="Outcomes")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection = _collection(repository, source_root)
        result = IngestService(repository).ingest_collection(collection.config.collection_id)

        assert result.run.status is ProcessingRunStatus.SUCCEEDED
        assert result.coverage.processed_count == 0
        assert result.coverage.skipped_count == 1
        assert result.coverage.failed_count == 1
        assert {outcome.terminal_outcome for outcome in result.outcomes} == {
            TerminalInputOutcome.SKIPPED,
            TerminalInputOutcome.FAILED,
        }
        assert {outcome.failure_code for outcome in result.outcomes} == {
            "invalid_utf8",
            "no_extractable_text",
        }
        assert {
            (item.category, item.reason_code, item.count) for item in result.coverage.omissions
        } == {
            ("parser_failure", "invalid_utf8", 1),
            ("unsupported", "no_extractable_text", 1),
        }
        assert repository.list_sources() == ()
        assert _counts(repository)["blobs"] == 0
        payload = json.loads(result.coverage.report_json)
        assert payload["policy_omission"] == {"count": None, "present": False}


def test_source_transaction_rolls_back_every_row_when_outbox_insert_fails(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "paper.md").write_text("# Exact\n\nGrounded text.\n", encoding="utf-8")
    library = LibraryConfig(name="Atomic")

    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection = _collection(repository, source_root)
        existing_event_id = repository.pending_outbox_events()[0].event_id
        source = read_source_bytes(collection.config.roots[0], "paper.md", ParserLimits())
        parsed = parse_source(source, ParserLimits(), pdf_temp_root=repository.paths.cache)
        blob = BlobStore(repository.paths).promote(source.data, source.content_sha256)
        repository._event_id_factory = lambda: existing_event_id

        with pytest.raises(PersistenceConflictError, match="processed Source"):
            repository.record_processed_source(
                collection_id=collection.config.collection_id,
                collection_root_id=collection.collection_root_ids[0],
                source=source,
                parsed=parsed,
                blob=blob,
            )

        assert _counts(repository) == {
            "blobs": 0,
            "sources": 0,
            "source_families": 0,
            "source_family_members": 0,
            "source_versions": 0,
            "source_heads": 0,
            "source_fragments": 0,
            "collection_memberships": 0,
        }
        assert blob.path.is_file()


def test_concurrent_same_input_serializes_to_added_plus_unchanged(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "paper.md").write_text("# Concurrent\n\nOne source.\n", encoding="utf-8")
    library = LibraryConfig(name="Concurrent")
    with initialize_library(library, data_root=data_root) as repository:
        collection = _collection(repository, source_root)
        collection_id = collection.config.collection_id

    barrier = threading.Barrier(2)

    def ingest_once() -> IngestDisposition:
        with open_library(library.library_id, data_root=data_root) as repository:
            barrier.wait()
            result = IngestService(repository).ingest_path(collection_id, "paper.md")
            disposition = result.outcomes[0].disposition
            assert disposition is not None
            return disposition

    with ThreadPoolExecutor(max_workers=2) as executor:
        dispositions = tuple(executor.map(lambda _index: ingest_once(), range(2)))

    assert set(dispositions) == {IngestDisposition.ADDED, IngestDisposition.UNCHANGED}
    with open_library(library.library_id, data_root=data_root) as repository:
        counts = _counts(repository)
        assert counts["sources"] == counts["source_versions"] == counts["source_heads"] == 1
        assert counts["source_families"] == counts["source_family_members"] == 1
        assert counts["source_fragments"] == 2
        assert repository._store.connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_stale_capture_is_rejected_before_blob_and_newer_ingest_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    source_root.mkdir()
    path = source_root / "paper.txt"
    older_text = "Older captured formulation"
    newer_text = "Newer canonical formulation"
    path.write_text(older_text, encoding="utf-8")
    library = LibraryConfig(name="Stale capture")
    with initialize_library(library, data_root=data_root) as repository:
        collection_id = _collection(repository, source_root).config.collection_id

    first_parsed = threading.Event()
    release_first = threading.Event()
    calls_lock = threading.Lock()
    parse_calls = 0
    original_parse = parse_source

    def delayed_first_parse(
        source: SourceBytes,
        limits: ParserLimits,
        *,
        pdf_temp_root: Path | None = None,
    ) -> ParseResult:
        nonlocal parse_calls
        result = original_parse(source, limits, pdf_temp_root=pdf_temp_root)
        with calls_lock:
            parse_calls += 1
            is_first = parse_calls == 1
        if is_first:
            first_parsed.set()
            assert release_first.wait(timeout=5)
        return result

    def ingest_once() -> IngestBatchResult:
        with open_library(library.library_id, data_root=data_root) as repository:
            return IngestService(repository).ingest_path(collection_id, "paper.txt")

    monkeypatch.setattr("dithyramba.ingest.service.parse_source", delayed_first_parse)
    with ThreadPoolExecutor(max_workers=2) as executor:
        older_future = executor.submit(ingest_once)
        assert first_parsed.wait(timeout=5)
        path.write_text(newer_text, encoding="utf-8")
        newer_future = executor.submit(ingest_once)
        assert not newer_future.done()
        release_first.set()
        older = older_future.result(timeout=5)
        newer = newer_future.result(timeout=5)

    assert older.run.status is ProcessingRunStatus.SUCCEEDED
    assert older.outcomes[0].failure_code == "source_changed_during_ingest"
    assert older.outcomes[0].retryable is True
    assert newer.outcomes[0].disposition is IngestDisposition.ADDED
    with open_library(library.library_id, data_root=data_root) as repository:
        assert _counts(repository)["blobs"] == 1
        assert _counts(repository)["source_versions"] == 1
        version = repository.get_source_version(newer.outcomes[0].source_version_id or "")
        assert version.content_sha256 == sha256_hex(newer_text.encode("utf-8"))
        old_digest = sha256_hex(older_text.encode("utf-8"))
        assert not (repository.paths.blobs / old_digest[:2] / old_digest[2:]).exists()
        source_locks = repository.paths.cache / "locks" / "sources"
        assert stat.S_IMODE(source_locks.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(item.stat().st_mode) == 0o600 for item in source_locks.iterdir())


def test_holdout_collection_creates_holdout_membership(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "canary.txt").write_text("Private canary", encoding="utf-8")
    library = LibraryConfig(name="Holdout")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection = _collection(
            repository,
            source_root,
            name="Canaries",
            kind=CollectionKind.HOLDOUT,
        )
        result = IngestService(repository).ingest_collection(collection.config.collection_id)
        source = repository.get_source(result.outcomes[0].source_id or "")
        assert source.memberships[0].state == "holdout"


def test_fragment_listing_exposes_addresses_and_hashes_but_not_text(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "note.md").write_text("# Heading\n\nParagraph.\n", encoding="utf-8")
    library = LibraryConfig(name="Metadata")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection = _collection(repository, source_root)
        result = IngestService(repository).ingest_collection(collection.config.collection_id)
        version_id = result.outcomes[0].source_version_id or ""
        fragments = repository.list_source_fragments(version_id)

        assert tuple(fragment.ordinal for fragment in fragments) == (0, 1)
        assert all(not hasattr(fragment, "text") for fragment in fragments)
        assert all(
            json.loads(fragment.source_address_json)["schema"].endswith("/1.0")
            for fragment in fragments
        )


def test_online_backup_contains_blob_rows_heads_and_coverage(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "note.txt").write_text("Backed up provenance", encoding="utf-8")
    library = LibraryConfig(name="Backup P2")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection = _collection(repository, source_root)
        IngestService(repository).ingest_collection(collection.config.collection_id)
        backup = repository.create_online_backup()

    connection = sqlite3.connect(backup.path)
    try:
        assert connection.execute("SELECT count(*) FROM sources").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM source_heads").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM coverage_reports").fetchone()[0] == 1
    finally:
        connection.close()
