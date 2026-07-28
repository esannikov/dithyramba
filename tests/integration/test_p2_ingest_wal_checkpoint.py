from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import NoReturn

import pytest

import dithyramba.store.database as store_database
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.ingest.models import ParserLimits, SourceBytes
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import CollectionRecord, LibraryRepository, initialize_library
from dithyramba.provenance import ProcessingRunStatus
from dithyramba.store import Store, TransactionStateError


def _collection(repository: LibraryRepository, source_root: Path) -> CollectionRecord:
    return repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="Checkpoint corpus",
            kind=CollectionKind.CORPUS,
            roots=(
                build_collection_root(
                    source_root,
                    data_root=repository.paths.application_data_root,
                ),
            ),
        )
    )


def _assert_separate_transactions(statements: tuple[str, ...], expected: int) -> None:
    transaction_open = False
    begin_count = 0
    commit_count = 0
    for statement in statements:
        if statement.startswith("BEGIN"):
            assert transaction_open is False
            transaction_open = True
            begin_count += 1
        elif statement == "COMMIT":
            assert transaction_open is True
            transaction_open = False
            commit_count += 1
        elif statement == "ROLLBACK":
            assert transaction_open is True
            transaction_open = False
    assert transaction_open is False
    assert begin_count == commit_count == expected


def test_collection_ingest_defers_then_checkpoints_once_after_receipt(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "a.md").write_text("# A\n\nFirst source.\n", encoding="utf-8")
    (source_root / "b.md").write_text("# B\n\nSecond source.\n", encoding="utf-8")

    with initialize_library(
        LibraryConfig(name="Deferred checkpoints"),
        data_root=tmp_path / "data",
    ) as repository:
        collection = _collection(repository, source_root)
        connection = repository._store.connection
        connection.execute("PRAGMA wal_autocheckpoint = 17")
        traced: list[str] = []
        connection.set_trace_callback(traced.append)
        try:
            result = IngestService(repository).ingest_collection(collection.config.collection_id)
        finally:
            connection.set_trace_callback(None)

        statements = tuple(statement.strip() for statement in traced)
        assert result.run.status is ProcessingRunStatus.SUCCEEDED
        assert result.coverage.processed_count == 2
        assert result.coverage.skipped_count == result.coverage.failed_count == 0
        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 17
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert statements.count("PRAGMA wal_autocheckpoint = 0") == 1
        assert statements.count("PRAGMA wal_autocheckpoint = 17") == 1
        assert statements.count("PRAGMA wal_checkpoint(PASSIVE)") == 1
        assert statements.index("PRAGMA wal_autocheckpoint = 17") > max(
            index for index, statement in enumerate(statements) if statement == "COMMIT"
        )
        assert statements.index("PRAGMA wal_checkpoint(PASSIVE)") > statements.index(
            "PRAGMA wal_autocheckpoint = 17"
        )
        # One run start, one transaction per Source, then the run/coverage receipt.
        _assert_separate_transactions(statements, expected=4)


def test_collection_ingest_restores_threshold_and_does_not_mask_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "paper.md").write_text("# Failure\n\nOriginal error wins.\n", encoding="utf-8")

    class OriginalIngestError(RuntimeError):
        pass

    def fail_parse(
        source: SourceBytes,
        limits: ParserLimits,
        *,
        pdf_temp_root: Path | None = None,
    ) -> NoReturn:
        del source, limits, pdf_temp_root
        raise OriginalIngestError("parser exploded")

    checkpoint_calls: list[sqlite3.Connection] = []

    def fail_checkpoint(connection: sqlite3.Connection) -> None:
        checkpoint_calls.append(connection)
        raise sqlite3.OperationalError("checkpoint unavailable")

    monkeypatch.setattr("dithyramba.ingest.service.parse_source", fail_parse)
    monkeypatch.setattr(store_database, "_passive_wal_checkpoint", fail_checkpoint)

    with initialize_library(
        LibraryConfig(name="Checkpoint error precedence"),
        data_root=tmp_path / "data",
    ) as repository:
        collection = _collection(repository, source_root)
        connection = repository._store.connection
        connection.execute("PRAGMA wal_autocheckpoint = 29")

        with pytest.raises(OriginalIngestError, match="parser exploded") as raised:
            IngestService(repository).ingest_collection(collection.config.collection_id)

        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 29
        assert checkpoint_calls == [connection]
        assert any("checkpoint unavailable" in note for note in raised.value.__notes__)
        assert connection.in_transaction is False
        run_row = connection.execute("SELECT status, error_code FROM processing_runs").fetchone()
        assert tuple(run_row) == ("failed", "internal_ingest_error")
        coverage_row = connection.execute(
            "SELECT processed_count, skipped_count, failed_count FROM coverage_reports"
        ).fetchone()
        assert tuple(coverage_row) == (0, 0, 1)


def test_checkpoint_deferral_is_not_a_transaction_and_rejects_nesting(
    tmp_path: Path,
) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        with store.defer_wal_autocheckpoints():
            assert not bool(connection.in_transaction)
            with store.transaction(immediate=True):
                assert bool(connection.in_transaction)
            assert not bool(connection.in_transaction)

        with (
            store.transaction(),
            pytest.raises(TransactionStateError, match="inside an active transaction"),
            store.defer_wal_autocheckpoints(),
        ):
            pass


def test_single_path_ingest_keeps_connection_checkpoint_policy_unchanged(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "paper.md").write_text("# API\n\nSingle path.\n", encoding="utf-8")

    with initialize_library(
        LibraryConfig(name="Single-path checkpoints"),
        data_root=tmp_path / "data",
    ) as repository:
        collection = _collection(repository, source_root)
        connection = repository._store.connection
        connection.execute("PRAGMA wal_autocheckpoint = 31")
        traced: list[str] = []
        connection.set_trace_callback(traced.append)
        try:
            result = IngestService(repository).ingest_path(
                collection.config.collection_id,
                "paper.md",
            )
        finally:
            connection.set_trace_callback(None)

        assert result.run.status is ProcessingRunStatus.SUCCEEDED
        assert connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 31
        assert not any("wal_autocheckpoint" in statement for statement in traced)
        assert not any("wal_checkpoint" in statement for statement in traced)
