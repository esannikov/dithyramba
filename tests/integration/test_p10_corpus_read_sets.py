"""Schema-v10 CorpusReadSet normalization and atomic batch completion."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from dithyramba.backup import create_backup_bundle, restore_backup_bundle
from dithyramba.persistence import PersistenceIntegrityError, SQLiteRecallBackend
from dithyramba.recall import QueryRequest, RecallPersistenceError, RecallService
from dithyramba.store import MigrationRunner, Store, discover_migrations
from tests.unit.test_p3_recall_persistence import _context

_ROOT = Path(__file__).resolve().parents[2]


def _request(base: QueryRequest, question: str) -> QueryRequest:
    return QueryRequest(
        question=question,
        library_id=base.library_id,
        collection_ids=base.collection_ids,
        corpus_snapshot_id=base.corpus_snapshot_id,
        access_policy_id=base.access_policy_id,
        purpose=base.purpose,
        exclusions=base.exclusions,
        retrieval=base.retrieval,
        result_contract=base.result_contract,
    )


def _service(context: Any) -> RecallService:
    return RecallService(
        context.backend,
        profile_version=context.profile_version,
        code_version="0.1.0.dev3",
    )


def _table_count(connection: Any, table: str) -> int:
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _convert_receipt_to_legacy(connection: Any, read_receipt_id: str) -> None:
    connection.execute("DROP TRIGGER read_receipt_corpus_sets_no_delete")
    read_set_id = str(
        connection.execute(
            "SELECT corpus_read_set_id FROM read_receipt_corpus_sets WHERE read_receipt_id = ?",
            (read_receipt_id,),
        ).fetchone()[0]
    )
    connection.execute(
        "DELETE FROM read_receipt_corpus_sets WHERE read_receipt_id = ?",
        (read_receipt_id,),
    )
    connection.execute(
        """
        INSERT INTO read_receipt_items(
            read_receipt_id, source_fragment_id, read_order, text_sha256
        )
        SELECT ?, source_fragment_id, read_order, text_sha256
        FROM corpus_read_set_items
        WHERE corpus_read_set_id = ?
        ORDER BY read_order
        """,
        (read_receipt_id, read_set_id),
    )


def test_v10_migration_copies_are_identical_and_fresh_schema_is_head(tmp_path: Path) -> None:
    root = _ROOT / "migrations/0010_corpus_read_sets.sql"
    packaged = _ROOT / "src/dithyramba/store/sql/0010_corpus_read_sets.sql"
    assert root.read_bytes() == packaged.read_bytes()
    assert discover_migrations(_ROOT / "migrations")[-1].version == 11

    with Store.open(tmp_path / "fresh.sqlite3") as store:
        assert store.schema_version == 11
        tables = {
            str(row[0])
            for row in store.connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "corpus_read_sets",
            "corpus_read_set_items",
            "read_receipt_corpus_sets",
        } <= tables


def test_batch_completion_rejects_inexact_inputs_before_touching_storage(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        result = _service(context).recall(context.request)
        run_id = result.run.processing_run_id
        packet = result.packet

        for invalid in ((), cast(Any, [])):
            with pytest.raises(TypeError, match="non-empty exact tuple"):
                context.backend.complete_recall_batch(completions=invalid)
        with pytest.raises(TypeError, match="items must be"):
            context.backend.complete_recall_batch(
                completions=cast(Any, ((run_id,),)),
            )
        with pytest.raises(TypeError, match="exact EvidencePackets"):
            context.backend.complete_recall_batch(
                completions=((run_id, cast(Any, object())),),
            )
        with pytest.raises(PersistenceIntegrityError, match="must be unique"):
            context.backend.complete_recall_batch(
                completions=((run_id, packet), (run_id, packet)),
            )

        connection = context.repository._store.connection
        assert _table_count(connection, "evidence_packets") == 1
        assert _table_count(connection, "recall_run_artifacts") == 1
    finally:
        context.repository.close()


def test_two_queries_share_one_read_set_and_replay_adds_no_set_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, paragraph_count=12)
    try:
        validation_counts = {"closure": 0, "projections": 0, "set": 0}
        original_closure = context.backend._validate_receipt_closure
        original_projections = context.backend._load_fragment_projections
        original_set = context.backend._load_corpus_read_set_items

        def tracked_closure(*args: Any, **kwargs: Any) -> Any:
            validation_counts["closure"] += 1
            return original_closure(*args, **kwargs)

        def tracked_projections(*args: Any, **kwargs: Any) -> Any:
            validation_counts["projections"] += 1
            return original_projections(*args, **kwargs)

        def tracked_set(*args: Any, **kwargs: Any) -> Any:
            validation_counts["set"] += 1
            return original_set(*args, **kwargs)

        monkeypatch.setattr(context.backend, "_validate_receipt_closure", tracked_closure)
        monkeypatch.setattr(context.backend, "_load_fragment_projections", tracked_projections)
        monkeypatch.setattr(context.backend, "_load_corpus_read_set_items", tracked_set)
        service = _service(context)
        results = service.recall_batch(
            (
                _request(context.request, "Munch expressive style"),
                _request(context.request, "Mars verified system"),
            )
        )
        assert validation_counts == {"closure": 1, "projections": 1, "set": 1}
        connection = context.repository._store.connection
        fragment_count = len(results[0].packet.read_receipt.items)
        assert _table_count(connection, "read_receipts") == 2
        assert _table_count(connection, "read_receipt_corpus_sets") == 2
        assert _table_count(connection, "corpus_read_sets") == 1
        assert _table_count(connection, "corpus_read_set_items") == fragment_count
        assert _table_count(connection, "read_receipt_items") == 0
        for result in results:
            loaded = context.backend.load_evidence_packet(result.packet.evidence_packet_id)
            assert loaded.read_receipt.canonical_bytes == result.packet.read_receipt.canonical_bytes
            assert loaded.canonical_bytes == result.packet.canonical_bytes

        normalized_before = tuple(
            _table_count(connection, table)
            for table in (
                "corpus_read_sets",
                "corpus_read_set_items",
                "read_receipts",
                "read_receipt_corpus_sets",
            )
        )
        replay = service.replay(results[0].packet.evidence_packet_id)
        assert replay.packet.canonical_bytes == results[0].packet.canonical_bytes
        assert normalized_before == tuple(
            _table_count(connection, table)
            for table in (
                "corpus_read_sets",
                "corpus_read_set_items",
                "read_receipts",
                "read_receipt_corpus_sets",
            )
        )
    finally:
        context.repository.close()


def test_same_items_in_another_library_have_a_distinct_read_set(tmp_path: Path) -> None:
    first = _context(tmp_path / "first", paragraph_count=3)
    second = _context(tmp_path / "second", paragraph_count=3)
    try:
        _service(first).recall(first.request)
        _service(second).recall(second.request)
        first_id = str(
            first.repository._store.connection.execute(
                "SELECT corpus_read_set_id FROM corpus_read_sets"
            ).fetchone()[0]
        )
        second_id = str(
            second.repository._store.connection.execute(
                "SELECT corpus_read_set_id FROM corpus_read_sets"
            ).fetchone()[0]
        )
        assert first.repository.library_id != second.repository.library_id
        assert first_id != second_id
    finally:
        first.repository.close()
        second.repository.close()


def test_legacy_and_shared_storage_reconstruct_identical_public_bytes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        connection = context.repository._store.connection
        expected = packet.canonical_bytes
        _convert_receipt_to_legacy(connection, packet.read_receipt.read_receipt_id)

        legacy = context.backend.load_evidence_packet(packet.evidence_packet_id)
        assert legacy.canonical_bytes == expected
        assert legacy.read_receipt.canonical_bytes == packet.read_receipt.canonical_bytes

        connection.execute("DROP TRIGGER read_receipt_corpus_sets_validate_insert")
        read_set_id = str(
            connection.execute("SELECT corpus_read_set_id FROM corpus_read_sets").fetchone()[0]
        )
        connection.execute(
            "INSERT INTO read_receipt_corpus_sets VALUES (?, ?)",
            (packet.read_receipt.read_receipt_id, read_set_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="both legacy and shared"):
            context.backend.load_evidence_packet(packet.evidence_packet_id)
    finally:
        context.repository.close()


@pytest.mark.parametrize("target", ["order", "hash", "library"])
def test_shared_read_set_corruption_fails_closed(tmp_path: Path, target: str) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        connection = context.repository._store.connection
        read_set_id = str(
            connection.execute("SELECT corpus_read_set_id FROM corpus_read_sets").fetchone()[0]
        )
        if target == "order":
            connection.execute("DROP TRIGGER corpus_read_set_items_no_update")
            connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute(
                "UPDATE corpus_read_set_items SET read_order = -1 "
                "WHERE corpus_read_set_id = ? AND read_order = 0",
                (read_set_id,),
            )
        elif target == "hash":
            connection.execute("DROP TRIGGER corpus_read_set_items_no_update")
            connection.execute(
                "UPDATE corpus_read_set_items SET text_sha256 = ? "
                "WHERE corpus_read_set_id = ? AND read_order = 0",
                ("a" * 64, read_set_id),
            )
        else:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("DROP TRIGGER corpus_read_sets_no_update")
            connection.execute(
                "UPDATE corpus_read_sets SET library_id = 'library_other' "
                "WHERE corpus_read_set_id = ?",
                (read_set_id,),
            )
        with pytest.raises(PersistenceIntegrityError):
            context.backend.load_evidence_packet(packet.evidence_packet_id)
    finally:
        context.repository.close()


def test_atomic_batch_failure_rolls_back_every_success_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, paragraph_count=4)
    try:
        service = _service(context)
        original = context.backend._ensure_evidence_packet
        calls = 0

        def fail_second(*args: Any, **kwargs: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected batch persistence failure")
            original(*args, **kwargs)

        monkeypatch.setattr(context.backend, "_ensure_evidence_packet", fail_second)
        with pytest.raises(RecallPersistenceError, match="atomically persisted"):
            service.recall_batch(
                (
                    _request(context.request, "Munch"),
                    _request(context.request, "Mars"),
                )
            )

        connection = context.repository._store.connection
        for table in (
            "corpus_read_sets",
            "corpus_read_set_items",
            "read_receipts",
            "read_receipt_corpus_sets",
            "access_receipts",
            "retrieval_receipts",
            "evidence_packets",
            "recall_run_artifacts",
        ):
            assert _table_count(connection, table) == 0
        assert (
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM coverage_reports WHERE stage = 'recall'"
                ).fetchone()[0]
            )
            == 0
        )
        assert (
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM processing_runs "
                    "WHERE kind = 'recall' AND status = 'failed'"
                ).fetchone()[0]
            )
            == 2
        )
    finally:
        context.repository.close()


@pytest.mark.parametrize("target", ["access", "read", "retrieval", "coverage"])
def test_batch_receipt_drift_fails_closed_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    context = _context(tmp_path, paragraph_count=4)
    try:
        service = _service(context)
        original = context.backend.complete_recall_batch

        def persist_with_drift(
            *,
            completions: tuple[tuple[str, Any], ...],
        ) -> Any:
            first, second = completions
            run_id, packet = second
            if target == "access":
                replacement = (
                    "0" * 64 if packet.access_receipt.policy_hash != "0" * 64 else "1" * 64
                )
                packet = packet.model_copy(
                    update={
                        "access_receipt": packet.access_receipt.model_copy(
                            update={"policy_hash": replacement}
                        )
                    }
                )
            elif target == "read":
                packet = packet.model_copy(
                    update={
                        "read_receipt": packet.read_receipt.model_copy(
                            update={"items": tuple(reversed(packet.read_receipt.items))}
                        )
                    }
                )
            elif target == "retrieval":
                packet = packet.model_copy(
                    update={
                        "retrieval_receipt": packet.retrieval_receipt.model_copy(
                            update={"profile_version": "tampered-profile/1.0"}
                        )
                    }
                )
            else:
                packet = packet.model_copy(
                    update={
                        "coverage_report": packet.coverage_report.model_copy(
                            update={"processed_count": packet.coverage_report.processed_count + 1}
                        )
                    }
                )
            return original(completions=(first, (run_id, packet)))

        monkeypatch.setattr(context.backend, "complete_recall_batch", persist_with_drift)
        with pytest.raises(RecallPersistenceError, match="atomically persisted"):
            service.recall_batch(
                (
                    _request(context.request, "Munch"),
                    _request(context.request, "Mars"),
                )
            )

        connection = context.repository._store.connection
        for table in (
            "corpus_read_sets",
            "corpus_read_set_items",
            "read_receipts",
            "read_receipt_corpus_sets",
            "access_receipts",
            "retrieval_receipts",
            "evidence_packets",
            "recall_run_artifacts",
        ):
            assert _table_count(connection, table) == 0
        assert (
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM processing_runs "
                    "WHERE kind = 'recall' AND status = 'failed'"
                ).fetchone()[0]
            )
            == 2
        )
    finally:
        context.repository.close()


def test_v9_legacy_packet_migrates_and_replays_without_backfill(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        service = _service(context)
        packet = service.recall(context.request).packet
        connection = context.repository._store.connection
        _convert_receipt_to_legacy(connection, packet.read_receipt.read_receipt_id)

        connection.execute("DROP TRIGGER read_receipt_items_reject_shared_insert")
        connection.execute("DROP TRIGGER reasoning_closure_results_no_delete")
        connection.execute("DROP TRIGGER reasoning_closure_results_no_update")
        connection.execute("DROP TABLE reasoning_closure_results")
        connection.execute("DROP TRIGGER idea_traces_no_delete")
        connection.execute("DROP TRIGGER idea_traces_no_update")
        connection.execute("DROP TABLE idea_traces")
        connection.execute("DROP TABLE read_receipt_corpus_sets")
        connection.execute("DROP TABLE corpus_read_set_items")
        connection.execute("DROP TABLE corpus_read_sets")
        connection.execute("DROP TRIGGER schema_migrations_no_delete")
        connection.execute("DELETE FROM schema_migrations WHERE version IN (10, 11)")
        connection.execute(
            "CREATE TRIGGER schema_migrations_no_delete BEFORE DELETE ON "
            "schema_migrations BEGIN SELECT RAISE(ABORT, "
            "'schema_migrations is append-only'); END"
        )

        v9 = tmp_path / "v9-migrations"
        v9.mkdir()
        for migration in discover_migrations(_ROOT / "migrations")[:9]:
            shutil.copyfile(_ROOT / "migrations" / migration.name, v9 / migration.name)
        MigrationRunner(connection, v9).verify_at_head()
        backed_up: list[int] = []
        applied = MigrationRunner(connection, _ROOT / "migrations").apply_all(
            backup_hook=lambda _connection, migration: backed_up.append(migration.version)
        )
        assert [item.version for item in applied] == [10, 11]
        assert backed_up == [10, 11]
        assert _table_count(connection, "corpus_read_sets") == 0

        loaded = context.backend.load_evidence_packet(packet.evidence_packet_id)
        assert loaded.canonical_bytes == packet.canonical_bytes
        replay = service.replay(packet.evidence_packet_id)
        assert replay.packet.canonical_bytes == packet.canonical_bytes
        assert _table_count(connection, "corpus_read_sets") == 0
    finally:
        context.repository.close()


def test_backup_restore_preserves_shared_read_set_and_packet(tmp_path: Path) -> None:
    context = _context(tmp_path)
    packet_id: str
    packet_bytes: bytes
    library_id = context.repository.library_id
    try:
        packet = _service(context).recall(context.request).packet
        packet_id = packet.evidence_packet_id
        packet_bytes = packet.canonical_bytes
        bundle = create_backup_bundle(context.repository)
    finally:
        context.repository.close()

    with restore_backup_bundle(bundle.path, data_root=tmp_path / "restored") as restored:
        assert restored.library_id == library_id
        connection = restored._store.connection
        assert _table_count(connection, "corpus_read_sets") == 1
        assert _table_count(connection, "read_receipt_corpus_sets") == 1
        loaded = SQLiteRecallBackend(restored).load_evidence_packet(packet_id)
        assert loaded.canonical_bytes == packet_bytes


def test_reduced_scaling_invariant_is_f_plus_q_not_f_times_q(tmp_path: Path) -> None:
    """A reduced structural witness for the 40k-fragment x 50-query case."""

    context = _context(tmp_path, paragraph_count=80)
    try:
        questions = tuple(
            _request(context.request, f"verified system paragraph {index:02d}")
            for index in range(50)
        )
        results = _service(context).recall_batch(questions)
        connection = context.repository._store.connection
        fragment_count = len(results[0].packet.read_receipt.items)
        assert _table_count(connection, "corpus_read_set_items") == fragment_count
        assert _table_count(connection, "read_receipt_corpus_sets") == len(questions)
        assert _table_count(connection, "read_receipt_items") == 0
        assert fragment_count + len(questions) < fragment_count * len(questions)
    finally:
        context.repository.close()
