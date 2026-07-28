from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from dithyramba.collections import (
    CollectionConfig,
    CollectionKind,
    CollectionRoot,
    build_collection_root,
)
from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)
from dithyramba.ingest.errors import (
    IngestPersistenceError,
    SourceChangedDuringIngestError,
    SourceUnavailableError,
)
from dithyramba.ingest.models import (
    FileIdentity,
    FragmentKind,
    MarkdownSourceAddress,
    MediaType,
    ParsedFragment,
    ParseResult,
    SourceBytes,
)
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    CollectionNotFoundError,
    CoverageReportNotFoundError,
    LibraryRepository,
    PersistenceConflictError,
    PersistenceIntegrityError,
    ProcessingRunNotFoundError,
    ProcessingRunStatus,
    SourceNotFoundError,
    SourceVersionNotFoundError,
    initialize_library,
)
from dithyramba.provenance import (
    BlobIntegrityError,
    BlobStore,
    BlobStoreError,
    CoverageDraft,
    IngestDisposition,
    IngestInputOutcome,
    ProcessingRunRecord,
    PromotedBlob,
    SourceFragmentRecord,
    SourceLineageIntegrityError,
    TerminalInputOutcome,
    build_ingest_coverage,
)
from dithyramba.provenance.blobs import PrivateLockStore
from dithyramba.provenance.models import SourceFamilyRole


def _processed_outcome(
    *, disposition: IngestDisposition = IngestDisposition.ADDED
) -> IngestInputOutcome:
    return IngestInputOutcome(
        collection_root_id="root_test",
        relative_path="paper.md",
        terminal_outcome=TerminalInputOutcome.PROCESSED,
        disposition=disposition,
        source_id="source_test",
        source_version_id="source_version_test",
        source_family_id="family_test",
        root_source_id="source_test",
        fragment_count=2,
    )


def _repository_with_collection(
    tmp_path: Path,
    *,
    roots: int = 1,
) -> tuple[LibraryRepository, str, tuple[Path, ...]]:
    source_roots: list[Path] = []
    collection_roots: list[CollectionRoot] = []
    data_root = tmp_path / "data"
    library = LibraryConfig(name="P2 unit")
    repository = initialize_library(library, data_root=data_root)
    for index in range(roots):
        source_root = tmp_path / f"source-{index}"
        source_root.mkdir()
        source_roots.append(source_root)
        collection_roots.append(build_collection_root(source_root, data_root=data_root))
    collection = repository.create_collection(
        CollectionConfig(
            library_id=library.library_id,
            name="Corpus",
            kind=CollectionKind.CORPUS,
            roots=tuple(collection_roots),
        )
    )
    return repository, collection.config.collection_id, tuple(source_roots)


def _source_bytes(data: bytes = b"Grounded text") -> SourceBytes:
    digest = sha256_hex(data)
    return SourceBytes(
        relative_path="paper.txt",
        canonical_uri="file:///tmp/paper.txt",
        media_type=MediaType.PLAIN_TEXT,
        data=data,
        content_sha256=digest,
        source_modified_at="2026-07-20T12:00:00.000000Z",
        identity=FileIdentity(1, 2, len(data), 3),
    )


def _parsed(text: str = "Grounded text") -> ParseResult:
    return ParseResult.processed(
        (
            ParsedFragment(
                ordinal=0,
                kind=FragmentKind.PARAGRAPH,
                text=text,
                address=MarkdownSourceAddress((), 1, 1, 0, len(text)),
            ),
        ),
        parser_revision="plain/1.0",
        normalized_codepoints=len(text),
    )


def _draft_with_payload(
    draft: CoverageDraft,
    payload: dict[str, object],
) -> CoverageDraft:
    return replace(
        draft,
        coverage_report_id=canonical_content_id("coverage", payload),
        report_json=canonical_json_bytes(payload).decode("utf-8"),
        report_hash=canonical_sha256_hex(payload),
    )


def test_coverage_payload_is_canonical_counted_and_content_addressed() -> None:
    outcomes = (
        _processed_outcome(),
        _processed_outcome(disposition=IngestDisposition.REVERTED),
        IngestInputOutcome(
            "root_test",
            "empty.md",
            TerminalInputOutcome.SKIPPED,
            failure_code="no_extractable_text",
        ),
        IngestInputOutcome(
            "root_test",
            "bad-1.md",
            TerminalInputOutcome.FAILED,
            failure_code="invalid_utf8",
        ),
        IngestInputOutcome(
            "root_test",
            "bad-2.md",
            TerminalInputOutcome.FAILED,
            failure_code="invalid_utf8",
        ),
    )

    first = build_ingest_coverage(
        processing_run_id="run_test",
        collection_id="collection_test",
        outcomes=outcomes,
    )
    second = build_ingest_coverage(
        processing_run_id="run_test",
        collection_id="collection_test",
        outcomes=outcomes,
    )

    assert first == second
    assert first.coverage_report_id.startswith("coverage_")
    assert first.processed_count == 2
    assert first.skipped_count == 1
    assert first.failed_count == 2
    assert {(item.category, item.reason_code, item.count) for item in first.omissions} == {
        ("unsupported", "no_extractable_text", 1),
        ("parser_failure", "invalid_utf8", 2),
    }
    assert (
        json.dumps(
            json.loads(first.report_json), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        == first.report_json
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "collection_root_id": "root",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.PROCESSED,
            },
            "disposition",
        ),
        (
            {
                "collection_root_id": "root",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.FAILED,
                "disposition": IngestDisposition.CHANGED,
                "failure_code": "failed",
            },
            "only a failure code",
        ),
        (
            {
                "collection_root_id": "root",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.SKIPPED,
                "failure_code": "skipped",
                "infrastructure_failure": True,
            },
            "infrastructure failure",
        ),
        (
            {
                "collection_root_id": "",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.FAILED,
                "failure_code": "failed",
            },
            "collection_root_id",
        ),
        (
            {
                "collection_root_id": "root",
                "relative_path": "",
                "terminal_outcome": TerminalInputOutcome.FAILED,
                "failure_code": "failed",
            },
            "relative_path",
        ),
        (
            {
                "collection_root_id": "root",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.FAILED,
                "failure_code": "failed",
                "retryable": "yes",
            },
            "must be bool",
        ),
        (
            {
                "collection_root_id": "root",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.FAILED,
                "failure_code": "failed",
                "fragment_count": -1,
            },
            "fragment_count",
        ),
        (
            {
                "collection_root_id": "root",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.PROCESSED,
                "disposition": IngestDisposition.ADDED,
            },
            "Source and SourceVersion",
        ),
        (
            {
                "collection_root_id": "root",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.PROCESSED,
                "disposition": IngestDisposition.ADDED,
                "source_id": "source",
                "source_version_id": "version",
            },
            "root-source lineage",
        ),
        (
            {
                "collection_root_id": "root",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.PROCESSED,
                "disposition": IngestDisposition.ADDED,
                "source_id": "source",
                "source_version_id": "version",
                "source_family_id": "family",
                "root_source_id": "source",
                "infrastructure_failure": True,
            },
            "cannot be an infrastructure",
        ),
        (
            {
                "collection_root_id": "root",
                "relative_path": "a.md",
                "terminal_outcome": TerminalInputOutcome.FAILED,
                "failure_code": "failed",
                "source_id": "source",
            },
            "cannot claim persisted provenance",
        ),
    ],
)
def test_input_outcome_rejects_inconsistent_states(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        IngestInputOutcome(**kwargs)  # type: ignore[arg-type]


def test_blob_store_promotes_once_and_verifies_existing_bytes(tmp_path: Path) -> None:
    library = LibraryConfig(name="Blobs")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        store = BlobStore(repository.paths)
        data = b"same immutable bytes"
        digest = sha256_hex(data)

        first = store.promote(data, digest)
        second = store.promote(data, digest)

        assert first.created is True
        assert second.created is False
        assert first.path == second.path
        assert first.path.read_bytes() == data
        assert stat.S_IMODE(first.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(first.path.parent.stat().st_mode) == 0o700
        store.verify(second)


def test_blob_store_rejects_mismatch_corruption_and_forged_path(tmp_path: Path) -> None:
    library = LibraryConfig(name="Blob integrity")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        store = BlobStore(repository.paths)
        data = b"verified"
        digest = sha256_hex(data)
        with pytest.raises(BlobIntegrityError, match="do not match"):
            store.promote(data, "a" * 64)

        blob = store.promote(data, digest)
        forged = PromotedBlob(
            blob.content_sha256,
            blob.byte_size,
            blob.relative_path,
            tmp_path / "elsewhere",
            False,
        )
        with pytest.raises(BlobIntegrityError, match="escaped"):
            store.verify(forged)

        blob.path.write_bytes(b"corrupt!")
        with pytest.raises(BlobIntegrityError, match=r"byte size|content address"):
            store.verify(blob)


def test_blob_store_rejects_hardlinked_or_nonprivate_blob(tmp_path: Path) -> None:
    library = LibraryConfig(name="Blob metadata")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        store = BlobStore(repository.paths)
        blob = store.promote(b"private", sha256_hex(b"private"))
        hardlink = tmp_path / "linked"
        os.link(blob.path, hardlink)
        try:
            with pytest.raises(BlobIntegrityError, match="one private regular file"):
                store.verify(blob)
        finally:
            hardlink.unlink()
        blob.path.chmod(0o644)
        with pytest.raises(BlobIntegrityError, match="not private"):
            store.verify(blob)


def test_blob_store_serializes_identical_publication_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = LibraryConfig(name="Blob publication lock")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        first_store = BlobStore(repository.paths)
        second_store = BlobStore(repository.paths)
        data = b"one concurrently published immutable payload"
        digest = sha256_hex(data)
        linked = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        call_lock = threading.Lock()
        delayed = False
        original_link = os.link

        def delayed_link(*args: Any, **kwargs: Any) -> None:
            nonlocal delayed
            original_link(*args, **kwargs)
            with call_lock:
                should_delay = not delayed
                delayed = True
            if should_delay:
                linked.set()
                assert release.wait(timeout=5)

        def promote_second() -> PromotedBlob:
            second_started.set()
            return second_store.promote(data, digest)

        monkeypatch.setattr("dithyramba.provenance.blobs.os.link", delayed_link)
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(first_store.promote, data, digest)
            assert linked.wait(timeout=5)
            second_future = executor.submit(promote_second)
            assert second_started.wait(timeout=5)
            assert not second_future.done()
            release.set()
            first = first_future.result(timeout=5)
            second = second_future.result(timeout=5)

        assert {first.created, second.created} == {True, False}
        assert first.path == second.path
        assert first.path.stat().st_nlink == 1
        assert not tuple(first.path.parent.glob(".blob-*.tmp"))


def test_blob_store_serializes_identical_publication_across_processes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    library = LibraryConfig(name="Cross-process blob publication")
    with initialize_library(library, data_root=data_root):
        pass
    data = b"same bytes from separate processes"
    digest = sha256_hex(data)
    script = """
import json
import sys
from pathlib import Path
from dithyramba.persistence import open_library
from dithyramba.provenance import BlobStore

with open_library(sys.argv[1], data_root=Path(sys.argv[2])) as repository:
    print("ready", flush=True)
    sys.stdin.readline()
    payload = bytes.fromhex(sys.argv[3])
    blob = BlobStore(repository.paths).promote(payload, sys.argv[4])
    print(json.dumps({"created": blob.created, "path": str(blob.path)}), flush=True)
"""
    processes = tuple(
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                library.library_id,
                str(data_root),
                data.hex(),
                digest,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    )
    try:
        for process in processes:
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "ready"
        for process in processes:
            assert process.stdin is not None
            process.stdin.write("go\n")
            process.stdin.flush()
        results: list[dict[str, object]] = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            results.append(cast(dict[str, object], json.loads(stdout)))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    assert {item["created"] for item in results} == {True, False}
    assert len({item["path"] for item in results}) == 1
    final_path = Path(cast(str, results[0]["path"]))
    assert final_path.read_bytes() == data
    assert final_path.stat().st_nlink == 1


def test_blob_store_recovers_only_controlled_stale_temp_hardlink(tmp_path: Path) -> None:
    library = LibraryConfig(name="Blob stale publication")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        store = BlobStore(repository.paths)
        data = b"recoverable crash window"
        digest = sha256_hex(data)
        blob = store.promote(data, digest)
        controlled = blob.path.parent / f".blob-{'a' * 32}.tmp"
        os.link(blob.path, controlled)
        assert blob.path.stat().st_nlink == 2

        recovered = store.promote(data, digest)

        assert recovered.created is False
        assert recovered.path.stat().st_nlink == 1
        assert not controlled.exists()

        uncontrolled = blob.path.parent / "external-alias"
        os.link(blob.path, uncontrolled)
        try:
            with pytest.raises(BlobIntegrityError, match="one private regular file"):
                store.promote(data, digest)
            assert uncontrolled.exists()
        finally:
            uncontrolled.unlink()


def test_private_lock_surfaces_are_nofollow_and_private(tmp_path: Path) -> None:
    library = LibraryConfig(name="Private locks")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        store = BlobStore(repository.paths)
        data = b"locked"
        store.promote(data, sha256_hex(data))
        locks = repository.paths.cache / "locks"
        blob_locks = locks / "blobs"
        assert stat.S_IMODE(locks.stat().st_mode) == 0o700
        assert stat.S_IMODE(blob_locks.stat().st_mode) == 0o700
        lock_files = tuple(blob_locks.glob("*.lock"))
        assert len(lock_files) == 1
        assert stat.S_IMODE(lock_files[0].stat().st_mode) == 0o600
        assert lock_files[0].stat().st_nlink == 1

    other_library = LibraryConfig(name="Lock symlink rejection")
    with initialize_library(other_library, data_root=tmp_path / "other-data") as repository:
        outside = tmp_path / "outside-locks"
        outside.mkdir()
        (repository.paths.cache / "locks").symlink_to(outside, target_is_directory=True)
        with pytest.raises(BlobStoreError, match="lock operation"):
            BlobStore(repository.paths).promote(b"x", sha256_hex(b"x"))
        assert not tuple(repository.paths.blobs.iterdir())


def test_blob_store_rejects_symlink_shard(tmp_path: Path) -> None:
    library = LibraryConfig(name="Blob symlink")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        data = b"symlink shard"
        digest = sha256_hex(data)
        outside = tmp_path / "outside"
        outside.mkdir()
        (repository.paths.blobs / digest[:2]).symlink_to(outside, target_is_directory=True)

        with pytest.raises(BlobStoreError):
            BlobStore(repository.paths).promote(data, digest)


def test_blob_store_type_missing_and_race_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="LibraryPaths"):
        BlobStore(cast(Any, object()))
    library = LibraryConfig(name="Blob branches")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        store = BlobStore(repository.paths)
        with pytest.raises(TypeError, match="immutable bytes"):
            store.promote(cast(Any, bytearray(b"bad")), sha256_hex(b"bad"))
        with pytest.raises(TypeError, match="PromotedBlob"):
            store.verify(cast(Any, object()))

        data = b"race-safe bytes"
        digest = sha256_hex(data)
        original_link = os.link

        def winning_racer(*args: Any, **kwargs: Any) -> None:
            original_link(*args, **kwargs)
            raise FileExistsError("simulated concurrent winner")

        monkeypatch.setattr("dithyramba.provenance.blobs.os.link", winning_racer)
        raced = store.promote(data, digest)
        assert raced.created is False
        store.verify(raced)

        monkeypatch.undo()
        raced.path.unlink()
        with pytest.raises(BlobIntegrityError, match="missing"):
            store.verify(raced)


def test_blob_store_cleans_temporary_file_on_io_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = LibraryConfig(name="Blob cleanup")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        store = BlobStore(repository.paths)
        data = b"cleanup"
        digest = sha256_hex(data)
        with PrivateLockStore(repository.paths).hold("blobs", digest):
            pass
        original_fsync = os.fsync
        calls = 0

        def fail_second_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated fsync failure")
            original_fsync(descriptor)

        monkeypatch.setattr("dithyramba.provenance.blobs.os.fsync", fail_second_fsync)
        with pytest.raises(BlobStoreError, match="promotion failed"):
            store.promote(data, digest)
        shard = repository.paths.blobs / digest[:2]
        assert not tuple(shard.glob("*.tmp"))


def test_blob_descriptor_helpers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dithyramba.provenance import blobs as blobs_module

    directory = tmp_path / "directory"
    directory.mkdir(mode=0o700)
    directory.chmod(0o755)
    with pytest.raises(BlobIntegrityError, match="directory is not private"):
        blobs_module._open_private_directory(directory)

    plain_file = tmp_path / "plain"
    plain_file.write_bytes(b"x")
    plain_file.chmod(0o600)
    descriptor = os.open(plain_file, os.O_RDONLY)
    try:
        with pytest.raises(BlobIntegrityError, match="not a real directory"):
            blobs_module._require_private_directory_descriptor(descriptor)
    finally:
        os.close(descriptor)

    writable = os.open(plain_file, os.O_WRONLY)
    try:
        monkeypatch.setattr("dithyramba.provenance.blobs.os.write", lambda *_args: 0)
        with pytest.raises(BlobStoreError, match="short write"):
            blobs_module._write_all(writable, b"payload")
    finally:
        os.close(writable)

    plain_file.chmod(0o644)
    descriptor = os.open(plain_file, os.O_RDONLY)
    try:
        with pytest.raises(BlobIntegrityError, match="not private"):
            blobs_module._verify_blob_descriptor(
                descriptor,
                expected_sha256=sha256_hex(b"x"),
                expected_size=1,
            )
    finally:
        os.close(descriptor)


def test_service_maps_retryable_reader_failure_to_successful_explicit_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.md").write_text("# Paper", encoding="utf-8")

        def changed(*_args: object, **_kwargs: object) -> None:
            raise SourceChangedDuringIngestError("changed")

        monkeypatch.setattr("dithyramba.ingest.service.read_source_bytes", changed)
        result = IngestService(repository).ingest_path(collection_id, "paper.md")

        assert result.run.status is ProcessingRunStatus.SUCCEEDED
        assert result.coverage.failed_count == 1
        assert result.outcomes[0].failure_code == "source_changed_during_ingest"
        assert result.outcomes[0].retryable is True
        assert repository.list_sources() == ()
    finally:
        repository.close()


def test_service_public_boundary_and_collection_discovery_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="LibraryRepository"):
        IngestService(cast(Any, object()))
    repository, collection_id, _roots = _repository_with_collection(tmp_path)
    try:
        service = IngestService(repository)
        assert service.repository is repository

        def unavailable(*_args: object, **_kwargs: object) -> None:
            raise SourceUnavailableError("root disappeared")

        monkeypatch.setattr("dithyramba.ingest.service.discover_source_paths", unavailable)
        result = service.ingest_collection(collection_id)
        assert result.run.status is ProcessingRunStatus.SUCCEEDED
        assert result.outcomes[0].relative_path is None
        assert result.outcomes[0].failure_code == "source_unavailable"
        assert result.outcomes[0].retryable
    finally:
        repository.close()


def test_collection_ingest_known_and_unexpected_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.md").write_text("# Paper", encoding="utf-8")

        def disk_failure(*_args: object, **_kwargs: object) -> None:
            raise BlobStoreError("disk unavailable")

        monkeypatch.setattr(BlobStore, "promote", disk_failure)
        failed = IngestService(repository).ingest_collection(collection_id)
        assert failed.run.status is ProcessingRunStatus.FAILED
        assert failed.outcomes[0].infrastructure_failure

        monkeypatch.undo()

        def programmer_failure(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("bad parser integration")

        monkeypatch.setattr("dithyramba.ingest.service.parse_source", programmer_failure)
        with pytest.raises(RuntimeError, match="bad parser integration"):
            IngestService(repository).ingest_collection(collection_id)
        assert (
            repository._store.connection.execute(
                "SELECT count(*) FROM processing_runs WHERE status = 'failed'"
            ).fetchone()[0]
            == 2
        )
    finally:
        repository.close()


def test_service_wraps_start_and_failure_receipt_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.md").write_text("# Paper", encoding="utf-8")
        service = IngestService(repository)

        def no_start(*_args: object, **_kwargs: object) -> None:
            raise PersistenceIntegrityError("no start")

        monkeypatch.setattr(repository, "begin_ingest_run", no_start)
        with pytest.raises(IngestPersistenceError, match="could not be started"):
            service.ingest_path(collection_id, "paper.md")

        monkeypatch.undo()
        service = IngestService(repository)

        def programmer_failure(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("unexpected")

        def no_receipt(*_args: object, **_kwargs: object) -> None:
            raise PersistenceIntegrityError("receipt unavailable")

        monkeypatch.setattr("dithyramba.ingest.service.parse_source", programmer_failure)
        monkeypatch.setattr(repository, "complete_ingest_run", no_receipt)
        with pytest.raises(IngestPersistenceError, match="failure receipt"):
            service.ingest_path(collection_id, "paper.md")
    finally:
        repository.close()


def test_collection_failure_receipt_failure_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.md").write_text("# Paper", encoding="utf-8")

        def programmer_failure(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("unexpected")

        def no_receipt(*_args: object, **_kwargs: object) -> None:
            raise PersistenceIntegrityError("receipt unavailable")

        monkeypatch.setattr("dithyramba.ingest.service.parse_source", programmer_failure)
        monkeypatch.setattr(repository, "complete_ingest_run", no_receipt)
        with pytest.raises(IngestPersistenceError, match="failure receipt"):
            IngestService(repository).ingest_collection(collection_id)
    finally:
        repository.close()


def test_service_marks_known_infrastructure_failure_and_returns_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.md").write_text("# Paper", encoding="utf-8")

        def fail_promote(*_args: object, **_kwargs: object) -> None:
            raise BlobStoreError("disk unavailable")

        monkeypatch.setattr(BlobStore, "promote", fail_promote)
        result = IngestService(repository).ingest_path(collection_id, "paper.md")

        assert result.run.status is ProcessingRunStatus.FAILED
        assert result.run.error_code == "ingest_persistence_failed"
        assert result.outcomes[0].infrastructure_failure is True
        assert result.coverage.failed_count == 1
    finally:
        repository.close()


def test_service_marks_unexpected_failure_then_reraises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.md").write_text("# Paper", encoding="utf-8")

        def explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("programmer bug")

        monkeypatch.setattr("dithyramba.ingest.service.parse_source", explode)
        with pytest.raises(RuntimeError, match="programmer bug"):
            IngestService(repository).ingest_path(collection_id, "paper.md")

        row = repository._store.connection.execute(
            "SELECT status, error_code FROM processing_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        assert tuple(row) == ("failed", "internal_ingest_error")
        assert (
            repository._store.connection.execute(
                "SELECT count(*) FROM coverage_reports"
            ).fetchone()[0]
            == 1
        )
    finally:
        repository.close()


def test_service_requires_root_id_for_multi_root_collection(tmp_path: Path) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path, roots=2)
    try:
        (roots[1] / "paper.txt").write_text("Second root", encoding="utf-8")
        service = IngestService(repository)
        with pytest.raises(ValueError, match="required"):
            service.ingest_path(collection_id, "paper.txt")
        with pytest.raises(ValueError, match="absent"):
            service.ingest_path(collection_id, "paper.txt", collection_root_id="root_missing")

        root_id = repository.get_collection(collection_id).collection_root_ids[1]
        result = service.ingest_path(
            collection_id,
            "paper.txt",
            collection_root_id=root_id,
        )
        assert result.outcomes[0].disposition is IngestDisposition.ADDED
    finally:
        repository.close()


def test_empty_collection_has_zero_count_success_and_path_is_rejected(tmp_path: Path) -> None:
    library = LibraryConfig(name="No roots")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        collection = repository.create_collection(
            CollectionConfig(
                library_id=library.library_id,
                name="Empty",
                kind=CollectionKind.CORPUS,
            )
        )
        service = IngestService(repository)
        result = service.ingest_collection(collection.config.collection_id)
        assert result.succeeded
        assert result.outcomes == ()
        with pytest.raises(ValueError, match="no roots"):
            service.ingest_path(collection.config.collection_id, "paper.md")


def test_missing_records_raise_typed_errors(tmp_path: Path) -> None:
    library = LibraryConfig(name="Missing records")
    with initialize_library(library, data_root=tmp_path / "data") as repository:
        with pytest.raises(SourceNotFoundError):
            repository.get_source("source_missing")
        with pytest.raises(SourceVersionNotFoundError):
            repository.get_source_version("source_version_missing")
        with pytest.raises(ProcessingRunNotFoundError):
            repository.get_processing_run("run_missing")
        with pytest.raises(CoverageReportNotFoundError):
            repository.get_coverage_report("coverage_missing")


def test_repository_rejects_invalid_processed_source_boundaries(tmp_path: Path) -> None:
    repository, _collection_id, _roots = _repository_with_collection(tmp_path)
    try:
        source = _source_bytes()
        parsed = _parsed()
        store = BlobStore(repository.paths)
        blob = store.promote(source.data, source.content_sha256)
        with pytest.raises(TypeError, match="SourceBytes"):
            repository.record_processed_source(
                collection_id="collection_missing",
                collection_root_id="root_missing",
                source=cast(Any, object()),
                parsed=parsed,
                blob=blob,
            )
        with pytest.raises(ValueError, match="processed ParseResult"):
            repository.record_processed_source(
                collection_id="collection_missing",
                collection_root_id="root_missing",
                source=source,
                parsed=ParseResult.skipped("unsupported_format", parser_revision="test/1.0"),
                blob=blob,
            )
        with pytest.raises(TypeError, match="PromotedBlob"):
            repository.record_processed_source(
                collection_id="collection_missing",
                collection_root_id="root_missing",
                source=source,
                parsed=parsed,
                blob=cast(Any, object()),
            )
        other = store.promote(b"other bytes", sha256_hex(b"other bytes"))
        with pytest.raises(PersistenceIntegrityError, match="differs"):
            repository.record_processed_source(
                collection_id="collection_missing",
                collection_root_id="root_missing",
                source=source,
                parsed=parsed,
                blob=other,
            )
        with pytest.raises(CollectionNotFoundError):
            repository.record_processed_source(
                collection_id="collection_missing",
                collection_root_id="root_missing",
                source=source,
                parsed=parsed,
                blob=blob,
            )
        assert repository.list_sources() == ()
    finally:
        repository.close()


def test_repository_binds_source_to_exact_collection_root_and_uri(tmp_path: Path) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path, roots=2)
    try:
        collection = repository.get_collection(collection_id)
        source = replace(
            _source_bytes(),
            canonical_uri=(roots[0] / "paper.txt").as_uri(),
        )
        parsed = _parsed()
        blob = BlobStore(repository.paths).promote(source.data, source.content_sha256)

        with pytest.raises(PersistenceIntegrityError, match="does not match"):
            repository.record_processed_source(
                collection_id=collection_id,
                collection_root_id=collection.collection_root_ids[1],
                source=source,
                parsed=parsed,
                blob=blob,
            )
        forged_uri = replace(source, canonical_uri=(roots[0] / "forged.txt").as_uri())
        with pytest.raises(PersistenceIntegrityError, match="does not match"):
            repository.record_processed_source(
                collection_id=collection_id,
                collection_root_id=collection.collection_root_ids[0],
                source=forged_uri,
                parsed=parsed,
                blob=blob,
            )
        with pytest.raises(PersistenceIntegrityError, match="does not belong"):
            repository.record_processed_source(
                collection_id=collection_id,
                collection_root_id="root_not_in_collection",
                source=source,
                parsed=parsed,
                blob=blob,
            )

        persisted = repository.record_processed_source(
            collection_id=collection_id,
            collection_root_id=collection.collection_root_ids[0],
            source=source,
            parsed=parsed,
            blob=blob,
        )
        assert persisted.disposition is IngestDisposition.ADDED
        assert len(repository.list_sources(collection_id=collection_id)) == 1
    finally:
        repository.close()


def test_processing_run_loader_fails_closed_on_invalid_lifecycle(tmp_path: Path) -> None:
    repository, _collection_id, _roots = _repository_with_collection(tmp_path)
    try:
        run = repository.begin_ingest_run(code_version="test", profile_version="index/1.0")
        connection = repository._store.connection
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE processing_runs SET status = 'unknown' WHERE processing_run_id = ?",
            (run.processing_run_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="status is invalid"):
            repository.get_processing_run(run.processing_run_id)

        terminal_at = "2026-07-20T12:00:00.000000Z"
        output_hash = "a" * 64
        connection.execute(
            """
            UPDATE processing_runs
            SET status = 'running', finished_at = ?, error_code = NULL, output_hash = ?
            WHERE processing_run_id = ?
            """,
            (terminal_at, output_hash, run.processing_run_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="running ProcessingRun"):
            repository.get_processing_run(run.processing_run_id)

        connection.execute(
            """
            UPDATE processing_runs
            SET status = 'succeeded', finished_at = NULL, error_code = NULL, output_hash = NULL
            WHERE processing_run_id = ?
            """,
            (run.processing_run_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="lacks completion"):
            repository.get_processing_run(run.processing_run_id)

        connection.execute(
            """
            UPDATE processing_runs
            SET status = 'failed', finished_at = ?, error_code = NULL, output_hash = ?
            WHERE processing_run_id = ?
            """,
            (terminal_at, output_hash, run.processing_run_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="lacks an error"):
            repository.get_processing_run(run.processing_run_id)

        connection.execute(
            """
            UPDATE processing_runs
            SET status = 'succeeded', finished_at = ?, error_code = 'wrong', output_hash = ?
            WHERE processing_run_id = ?
            """,
            (terminal_at, output_hash, run.processing_run_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="successful ProcessingRun"):
            repository.get_processing_run(run.processing_run_id)
    finally:
        repository.close()


def test_source_loaders_detect_corrupt_projections(tmp_path: Path) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.txt").write_text("Relational integrity", encoding="utf-8")
        result = IngestService(repository).ingest_path(collection_id, "paper.txt")
        outcome = result.outcomes[0]
        source_id = outcome.source_id or ""
        source_version_id = outcome.source_version_id or ""
        fragment_id = repository.list_source_fragments(source_version_id)[0].source_fragment_id
        connection = repository._store.connection
        connection.execute("PRAGMA ignore_check_constraints = ON")

        connection.execute("DROP TRIGGER source_versions_no_update")
        connection.execute(
            "UPDATE source_versions SET parse_status = 'unknown' WHERE source_version_id = ?",
            (source_version_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="parse status"):
            repository.get_source_version(source_version_id)
        connection.execute(
            "UPDATE source_versions SET parse_status = 'processed', failure_code = 'wrong' "
            "WHERE source_version_id = ?",
            (source_version_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="failure projection"):
            repository.get_source_version(source_version_id)
        connection.execute(
            "UPDATE source_versions SET failure_code = NULL WHERE source_version_id = ?",
            (source_version_id,),
        )
        connection.execute(
            "UPDATE source_versions SET source_modified_at = 'not-a-timestamp' "
            "WHERE source_version_id = ?",
            (source_version_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="operational timestamp"):
            repository.get_source_version(source_version_id)
        connection.execute(
            "UPDATE source_versions SET source_modified_at = ? WHERE source_version_id = ?",
            ("2026-07-20T12:00:00.000000Z", source_version_id),
        )

        fragment = repository.list_source_fragments(source_version_id)[0]
        connection.execute("DROP TRIGGER source_fragments_no_update")
        connection.execute(
            "UPDATE source_fragments SET address_hash = ? WHERE source_fragment_id = ?",
            ("a" * 64, fragment_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="SourceAddress hash"):
            repository.list_source_fragments(source_version_id)
        connection.execute(
            "UPDATE source_fragments SET address_hash = ?, ordinal = 2 "
            "WHERE source_fragment_id = ?",
            (fragment.address_hash, fragment_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="ordinals"):
            repository.list_source_fragments(source_version_id)
        connection.execute(
            "UPDATE source_fragments SET ordinal = 0 WHERE source_fragment_id = ?",
            (fragment_id,),
        )

        source = repository.get_source(source_id)
        connection.execute("DROP TRIGGER sources_no_update")
        connection.execute(
            "UPDATE sources SET canonical_uri = 'invalid' WHERE source_id = ?", (source_id,)
        )
        with pytest.raises(PersistenceIntegrityError, match="file URI"):
            repository.get_source(source_id)
        connection.execute(
            "UPDATE sources SET canonical_uri = ?, media_type = 'invalid' WHERE source_id = ?",
            (source.canonical_uri, source_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="media type"):
            repository.get_source(source_id)
        connection.execute(
            "UPDATE sources SET media_type = ? WHERE source_id = ?", (source.media_type, source_id)
        )
        connection.execute("DROP TRIGGER source_heads_no_delete")
        connection.execute("DELETE FROM source_heads WHERE source_id = ?", (source_id,))
        with pytest.raises(PersistenceIntegrityError, match="current SourceVersion"):
            repository.get_source(source_id)
    finally:
        repository.close()


def test_source_lineage_loader_rejects_invalid_family_shapes(tmp_path: Path) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.txt").write_text("Lineage", encoding="utf-8")
        outcome = IngestService(repository).ingest_path(collection_id, "paper.txt").outcomes[0]
        source_id = outcome.source_id or ""
        connection = repository._store.connection
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("DROP TRIGGER source_family_members_no_update")

        connection.execute(
            "UPDATE source_family_members SET role = 'unknown' WHERE source_id = ?", (source_id,)
        )
        with pytest.raises(SourceLineageIntegrityError, match="role is invalid"):
            repository.get_source(source_id)
        connection.execute(
            "UPDATE source_family_members SET role = 'root', root_source_id = ? "
            "WHERE source_id = ?",
            (source_id, source_id),
        )
        with pytest.raises(SourceLineageIntegrityError, match="root Source"):
            repository.get_source(source_id)
        connection.execute(
            "UPDATE source_family_members SET role = 'duplicate', root_source_id = NULL "
            "WHERE source_id = ?",
            (source_id,),
        )
        with pytest.raises(SourceLineageIntegrityError, match="lacks root_source_id"):
            repository.get_source(source_id)
        connection.execute(
            "UPDATE source_family_members SET root_source_id = ? WHERE source_id = ?",
            (source_id, source_id),
        )
        with pytest.raises(SourceLineageIntegrityError, match="does not resolve"):
            repository.get_source(source_id)

        connection.execute("DROP TRIGGER source_family_members_no_delete")
        connection.execute("DELETE FROM source_family_members WHERE source_id = ?", (source_id,))
        with pytest.raises(SourceLineageIntegrityError, match="exactly one"):
            repository.get_source(source_id)
    finally:
        repository.close()


def test_conflicting_content_families_and_membership_fail_closed(tmp_path: Path) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        first_path = roots[0] / "first.txt"
        second_path = roots[0] / "second.txt"
        first_path.write_text("First family", encoding="utf-8")
        second_path.write_text("Second family", encoding="utf-8")
        service = IngestService(repository)
        first = service.ingest_path(collection_id, "first.txt")
        second = service.ingest_path(collection_id, "second.txt")

        first_path.write_text("Second family", encoding="utf-8")
        collision = service.ingest_path(collection_id, "first.txt")
        assert collision.run.status is ProcessingRunStatus.FAILED
        assert collision.outcomes[0].failure_code == "ingest_persistence_failed"
        assert len(repository.list_source_versions(first.outcomes[0].source_id or "")) == 1

        second_source_id = second.outcomes[0].source_id or ""
        repository._store.connection.execute(
            "UPDATE collection_memberships SET state = 'holdout' "
            "WHERE collection_id = ? AND source_id = ?",
            (collection_id, second_source_id),
        )
        membership_conflict = service.ingest_path(collection_id, "second.txt")
        assert membership_conflict.run.status is ProcessingRunStatus.FAILED
        assert membership_conflict.outcomes[0].failure_code == "ingest_persistence_failed"
    finally:
        repository.close()


def test_complete_ingest_run_rejects_invalid_contract_combinations(tmp_path: Path) -> None:
    repository, collection_id, _roots = _repository_with_collection(tmp_path)
    try:
        run = repository.begin_ingest_run(code_version="test", profile_version="index/1.0")
        draft = build_ingest_coverage(
            processing_run_id=run.processing_run_id,
            collection_id=collection_id,
            outcomes=(_processed_outcome(),),
        )
        with pytest.raises(TypeError, match="CoverageDraft"):
            repository.complete_ingest_run(
                coverage=cast(Any, object()),
                status=ProcessingRunStatus.SUCCEEDED,
            )
        with pytest.raises(ValueError, match="succeeded or failed"):
            repository.complete_ingest_run(
                coverage=draft,
                status=ProcessingRunStatus.CANCELLED,
            )
        with pytest.raises(ValueError, match="requires error_code"):
            repository.complete_ingest_run(
                coverage=draft,
                status=ProcessingRunStatus.FAILED,
            )
        with pytest.raises(ValueError, match="cannot have error_code"):
            repository.complete_ingest_run(
                coverage=draft,
                status=ProcessingRunStatus.SUCCEEDED,
                error_code="wrong",
            )

        invalid_drafts = (
            replace(draft, report_hash="a" * 64),
            replace(draft, coverage_report_id="coverage_wrong"),
            replace(draft, processing_run_id="run_other"),
            replace(draft, collection_id="collection_other"),
            replace(draft, processed_count=2),
        )
        for invalid in invalid_drafts:
            with pytest.raises(PersistenceIntegrityError):
                repository.complete_ingest_run(
                    coverage=invalid,
                    status=ProcessingRunStatus.SUCCEEDED,
                )

        payload = json.loads(draft.report_json)
        malformed_outcomes = dict(payload)
        malformed_outcomes["outcomes"] = "not-an-array"
        wrong_classification = dict(payload)
        wrong_classification["outcomes"] = []
        wrong_schema = dict(payload)
        wrong_schema["schema"] = "wrong"
        wrong_policy = dict(payload)
        wrong_policy["policy_omission"] = {"count": 0, "present": False}
        wrong_omissions = dict(payload)
        wrong_omissions["omissions"] = [{"reason_code": "wrong"}]
        extra_top_level = dict(payload)
        extra_top_level["unexpected"] = True
        outcome_with_extra_key = dict(payload)
        extra_outcome = dict(payload["outcomes"][0])
        extra_outcome["unexpected"] = True
        outcome_with_extra_key["outcomes"] = [extra_outcome]
        outcome_with_invalid_bool = dict(payload)
        invalid_bool_outcome = dict(payload["outcomes"][0])
        invalid_bool_outcome["retryable"] = 1
        outcome_with_invalid_bool["outcomes"] = [invalid_bool_outcome]
        duplicate_outcomes = dict(payload)
        duplicate_outcomes["outcomes"] = [payload["outcomes"][0], payload["outcomes"][0]]
        for invalid_payload in (
            malformed_outcomes,
            wrong_classification,
            wrong_schema,
            wrong_policy,
            wrong_omissions,
            extra_top_level,
            outcome_with_extra_key,
            outcome_with_invalid_bool,
            duplicate_outcomes,
        ):
            with pytest.raises(PersistenceIntegrityError):
                repository.complete_ingest_run(
                    coverage=_draft_with_payload(draft, invalid_payload),
                    status=ProcessingRunStatus.SUCCEEDED,
                )

        omission_draft = build_ingest_coverage(
            processing_run_id=run.processing_run_id,
            collection_id=collection_id,
            outcomes=(
                IngestInputOutcome(
                    "root_test",
                    "bad.md",
                    TerminalInputOutcome.FAILED,
                    failure_code="invalid_utf8",
                ),
            ),
        )
        forged_omission = replace(
            omission_draft,
            omissions=(replace(omission_draft.omissions[0], omission_id="omission_wrong"),),
        )
        with pytest.raises(PersistenceIntegrityError, match="omission ID"):
            repository.complete_ingest_run(
                coverage=forged_omission,
                status=ProcessingRunStatus.SUCCEEDED,
            )
    finally:
        repository.close()


def test_ingest_run_status_requires_matching_infrastructure_outcome(tmp_path: Path) -> None:
    repository, collection_id, _roots = _repository_with_collection(tmp_path)
    try:
        run = repository.begin_ingest_run(code_version="test", profile_version="index/1.0")
        infrastructure = IngestInputOutcome(
            "root_test",
            "paper.md",
            TerminalInputOutcome.FAILED,
            failure_code="ingest_persistence_failed",
            retryable=True,
            infrastructure_failure=True,
        )
        infrastructure_draft = build_ingest_coverage(
            processing_run_id=run.processing_run_id,
            collection_id=collection_id,
            outcomes=(infrastructure,),
        )
        with pytest.raises(PersistenceIntegrityError, match="successful ingest"):
            repository.complete_ingest_run(
                coverage=infrastructure_draft,
                status=ProcessingRunStatus.SUCCEEDED,
            )
        with pytest.raises(PersistenceIntegrityError, match="error_code must match"):
            repository.complete_ingest_run(
                coverage=infrastructure_draft,
                status=ProcessingRunStatus.FAILED,
                error_code="other_failure",
            )

        parser_failure_draft = build_ingest_coverage(
            processing_run_id=run.processing_run_id,
            collection_id=collection_id,
            outcomes=(
                IngestInputOutcome(
                    "root_test",
                    "paper.md",
                    TerminalInputOutcome.FAILED,
                    failure_code="invalid_utf8",
                ),
            ),
        )
        with pytest.raises(PersistenceIntegrityError, match="requires an infrastructure"):
            repository.complete_ingest_run(
                coverage=parser_failure_draft,
                status=ProcessingRunStatus.FAILED,
                error_code="invalid_utf8",
            )

        completed, coverage = repository.complete_ingest_run(
            coverage=infrastructure_draft,
            status=ProcessingRunStatus.FAILED,
            error_code="ingest_persistence_failed",
        )
        assert completed.status is ProcessingRunStatus.FAILED
        assert coverage.failed_count == 1
    finally:
        repository.close()


def test_coverage_loader_rejects_typed_shape_and_run_consistency_corruption(
    tmp_path: Path,
) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.txt").write_text("Coverage integrity", encoding="utf-8")
        result = IngestService(repository).ingest_path(collection_id, "paper.txt")
        coverage_id = result.coverage.coverage_report_id
        run_id = result.run.processing_run_id
        original_json = result.coverage.report_json
        original_hash = result.coverage.report_hash
        payload = json.loads(original_json)
        forged_outcome = dict(payload["outcomes"][0])
        forged_outcome["unexpected"] = True
        payload["outcomes"] = [forged_outcome]
        forged_json = canonical_json_bytes(payload).decode("utf-8")
        forged_hash = canonical_sha256_hex(payload)
        connection = repository._store.connection
        connection.execute("DROP TRIGGER coverage_reports_no_update")
        connection.execute(
            "UPDATE coverage_reports SET report_json = ?, report_hash = ? "
            "WHERE coverage_report_id = ?",
            (forged_json, forged_hash, coverage_id),
        )
        connection.execute(
            "UPDATE processing_runs SET output_hash = ? WHERE processing_run_id = ?",
            (forged_hash, run_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="outcome payload"):
            repository.get_coverage_report(coverage_id)

        connection.execute(
            "UPDATE coverage_reports SET report_json = ?, report_hash = ? "
            "WHERE coverage_report_id = ?",
            (original_json, original_hash, coverage_id),
        )
        connection.execute(
            "UPDATE processing_runs SET status = 'failed', error_code = ?, output_hash = ? "
            "WHERE processing_run_id = ?",
            ("ingest_persistence_failed", original_hash, run_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="requires an infrastructure"):
            repository.get_coverage_report(coverage_id)
    finally:
        repository.close()


def test_complete_ingest_run_missing_terminal_and_conflict_paths(tmp_path: Path) -> None:
    repository, collection_id, _roots = _repository_with_collection(tmp_path)
    try:
        missing = build_ingest_coverage(
            processing_run_id="run_missing",
            collection_id=collection_id,
            outcomes=(),
        )
        with pytest.raises(ProcessingRunNotFoundError):
            repository.complete_ingest_run(
                coverage=missing,
                status=ProcessingRunStatus.SUCCEEDED,
            )

        run = repository.begin_ingest_run(code_version="test", profile_version="index/1.0")
        draft = build_ingest_coverage(
            processing_run_id=run.processing_run_id,
            collection_id=collection_id,
            outcomes=(),
        )
        repository.complete_ingest_run(
            coverage=draft,
            status=ProcessingRunStatus.SUCCEEDED,
        )
        with pytest.raises(PersistenceConflictError, match="already terminal"):
            repository.complete_ingest_run(
                coverage=draft,
                status=ProcessingRunStatus.SUCCEEDED,
            )

        other_run = repository.begin_ingest_run(code_version="test", profile_version="index/1.0")
        other = build_ingest_coverage(
            processing_run_id=other_run.processing_run_id,
            collection_id=collection_id,
            outcomes=(),
        )
        conflict_event_id = repository.pending_outbox_events()[0].event_id
        repository._event_id_factory = lambda: conflict_event_id
        with pytest.raises(PersistenceConflictError, match="completion conflicted"):
            repository.complete_ingest_run(
                coverage=other,
                status=ProcessingRunStatus.SUCCEEDED,
            )
        assert repository.get_processing_run(other_run.processing_run_id).status is (
            ProcessingRunStatus.RUNNING
        )
    finally:
        repository.close()


def test_blob_record_round_trip_missing_and_corrupt_file(tmp_path: Path) -> None:
    repository, collection_id, roots = _repository_with_collection(tmp_path)
    try:
        (roots[0] / "paper.txt").write_text("Blob record", encoding="utf-8")
        result = IngestService(repository).ingest_path(collection_id, "paper.txt")
        source = repository.get_source(result.outcomes[0].source_id or "")
        version = repository.get_source_version(source.current_source_version_id)
        blob = repository.get_blob(version.content_sha256)
        assert blob.path.read_text(encoding="utf-8") == "Blob record"
        with pytest.raises(SourceNotFoundError):
            repository.get_blob("a" * 64)
        with pytest.raises(PersistenceIntegrityError, match="SHA-256"):
            repository.get_blob("invalid")
        blob.path.unlink()
        with pytest.raises(BlobIntegrityError, match="missing"):
            repository.get_blob(version.content_sha256)
    finally:
        repository.close()


def test_models_reject_invalid_blob_and_run_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        PromotedBlob("bad", 1, "ba/d", tmp_path.resolve(), False)
    with pytest.raises(ValueError, match="derived"):
        PromotedBlob("a" * 64, 1, "wrong/path", tmp_path.resolve(), False)
    with pytest.raises(ValueError, match="byte_size"):
        PromotedBlob("a" * 64, -1, f"aa/{'a' * 62}", tmp_path.resolve(), False)
    with pytest.raises(ValueError, match="absolute"):
        PromotedBlob("a" * 64, 1, f"aa/{'a' * 62}", Path("relative"), False)
    with pytest.raises(TypeError, match="created"):
        PromotedBlob(
            "a" * 64,
            1,
            f"aa/{'a' * 62}",
            tmp_path.resolve(),
            cast(Any, 1),
        )

    record = ProcessingRunRecord(
        processing_run_id="run_test",
        kind="ingest",
        status=ProcessingRunStatus.RUNNING,
        code_version="test",
        profile_version="index/1.0",
        started_at="now",
        finished_at=None,
        error_code=None,
        output_hash=None,
    )
    assert record.status is ProcessingRunStatus.RUNNING
    fragment = SourceFragmentRecord(
        "fragment_x", "source_version_x", 0, "paragraph", "a" * 64, "{}", "b" * 64
    )
    assert not hasattr(fragment, "text")
    assert SourceFamilyRole.ROOT.value == "root"
