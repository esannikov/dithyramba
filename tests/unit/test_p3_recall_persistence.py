from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

import dithyramba.persistence.recall as recall_persistence
import dithyramba.recall.artifacts as recall_artifacts
from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect, RequestScope
from dithyramba.collections import (
    CollectionConfig,
    CollectionKind,
    CollectionRoot,
    build_collection_root,
)
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    PersistenceConflictError,
    PersistenceIntegrityError,
    ProcessingRunStatus,
    SQLiteRecallBackend,
    initialize_library,
)
from dithyramba.persistence.repository import LibraryRepository
from dithyramba.recall import (
    EvidencePacket,
    OmissionCategory,
    OmissionDisclosure,
    PacketResultStatus,
    QueryRequest,
    ReadReceipt,
    ReadReceiptItem,
    RecallBackend,
    RecallCoverageReport,
    RecallOmission,
    RecallPersistenceError,
    RecallService,
    search_ephemeral_fts,
)


@dataclass(frozen=True, slots=True)
class _RecallContext:
    repository: LibraryRepository
    backend: SQLiteRecallBackend
    request: QueryRequest
    profile_version: str


@dataclass(slots=True)
class _InsertCursor:
    rowcount: int

    def fetchall(self) -> tuple[tuple[object, ...], ...]:
        return ()


@dataclass(slots=True)
class _RecordingConnection:
    rows: tuple[tuple[object, ...], ...] = ()

    def execute(self, _sql: str, _parameters: object) -> _InsertCursor:
        return _InsertCursor(rowcount=1)

    def executemany(self, _sql: str, parameters: object) -> None:
        self.rows = tuple(cast(Any, parameters))


def _profile_version() -> str:
    return search_ephemeral_fts(
        question="profile",
        fragments=(),
        max_candidates=1,
    ).profile_version


def _context(
    tmp_path: Path,
    *,
    with_source: bool = True,
    paragraph_count: int = 1,
) -> _RecallContext:
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True)
    roots: tuple[CollectionRoot, ...] = ()
    if with_source:
        paragraphs = [
            "Edvard Munch painted The Scream in an expressive style."
            if index == 0
            else f"Mars evidence paragraph {index:04d} describes a verified system."
            for index in range(paragraph_count)
        ]
        (source_root / "munch.md").write_text(
            "# Edvard Munch\n\n" + "\n\n".join(paragraphs),
            encoding="utf-8",
        )
        roots = (build_collection_root(source_root, data_root=data_root),)
    repository = initialize_library(
        LibraryConfig(name="P3 recall persistence"),
        data_root=data_root,
    )
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="Corpus",
            kind=CollectionKind.CORPUS,
            roots=roots,
        )
    )
    if with_source:
        result = IngestService(repository).ingest_path(
            collection.config.collection_id,
            "munch.md",
        )
        assert result.run.status is ProcessingRunStatus.SUCCEEDED
    snapshot = repository.freeze_snapshot((collection.config.collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id="policy_research",
        library_id=repository.library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),),
    )
    repository.persist_access_policy(name="Research", snapshot=policy)
    request = QueryRequest(
        question="What is known about Munch and The Scream?",
        library_id=repository.library_id,
        collection_ids=(collection.config.collection_id,),
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        access_policy_id=policy.access_policy_id,
        purpose="research",
    )
    return _RecallContext(
        repository=repository,
        backend=SQLiteRecallBackend(repository),
        request=request,
        profile_version=_profile_version(),
    )


def _service(context: _RecallContext) -> RecallService:
    return RecallService(
        context.backend,
        profile_version=context.profile_version,
        code_version="0.1.0.dev3",
    )


def _as_protocol(value: SQLiteRecallBackend) -> RecallBackend:
    return value


def test_adapter_structurally_matches_service_protocol_and_query_is_idempotent(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        assert _as_protocol(context.backend) is context.backend
        context.backend.persist_query_request(context.request)
        context.backend.persist_query_request(context.request)

        loaded = context.backend.load_query_request(context.request.query_request_id)
        assert loaded == context.request
        assert loaded.canonical_bytes == context.request.canonical_bytes
        connection = context.repository._store.connection
        assert connection.execute("SELECT COUNT(*) FROM query_requests").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM event_outbox WHERE event_type = 'query_request.persisted'"
            ).fetchone()[0]
            == 1
        )
    finally:
        context.repository.close()


def test_read_receipt_id_is_computed_once_before_shared_head_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A content-derived receipt ID must not be rehashed once per item."""

    context = _context(tmp_path)
    try:
        receipt = ReadReceipt(
            query_request_hash="a" * 64,
            retrieval_corpus_hash="b" * 64,
            items=tuple(
                ReadReceiptItem(
                    source_fragment_id=f"fragment_{index}",
                    source_version_id=f"source_version_{index}",
                    read_order=index,
                    text_sha256=f"{index:064x}",
                )
                for index in range(64)
            ),
        )
        id_computations = 0
        original_content_id = cast(
            Callable[[str, bytes], str],
            vars(recall_artifacts)["content_id"],
        )

        def tracked_content_id(prefix: str, canonical_bytes: bytes) -> str:
            nonlocal id_computations
            id_computations += 1
            return original_content_id(prefix, canonical_bytes)

        monkeypatch.setattr(
            recall_artifacts,
            "content_id",
            tracked_content_id,
        )
        monkeypatch.setattr(
            recall_persistence,
            "_require_unique_artifact_identity",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            context.backend,
            "_load_read_receipt",
            lambda *_args, **_kwargs: receipt,
        )
        monkeypatch.setattr(
            context.backend,
            "_ensure_corpus_read_set",
            lambda *_args, **_kwargs: recall_persistence._CorpusReadSetIdentity(
                corpus_read_set_id="corpus_read_set_test",
                set_hash="c" * 64,
            ),
        )
        connection = _RecordingConnection()

        context.backend._ensure_read_receipt(
            cast(Any, connection),
            processing_run_id="run_recall_test",
            query_request_id=context.request.query_request_id,
            receipt=receipt,
        )

        assert id_computations == 1
        assert receipt.read_receipt_id.startswith("read_")
    finally:
        context.repository.close()


def test_recall_and_replay_reuse_immutable_artifacts_but_link_random_runs(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        service = _service(context)
        first = service.recall(context.request)
        replay = service.replay(first.packet.evidence_packet_id)

        assert first.run.status is ProcessingRunStatus.SUCCEEDED
        assert replay.run.status is ProcessingRunStatus.SUCCEEDED
        assert replay.run.processing_run_id != first.run.processing_run_id
        assert replay.packet.canonical_bytes == first.packet.canonical_bytes
        assert replay.packet.packet_hash == first.packet.packet_hash
        assert (
            context.backend.load_evidence_packet(first.packet.evidence_packet_id).canonical_bytes
            == first.packet.canonical_bytes
        )

        connection = context.repository._store.connection
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM coverage_reports WHERE stage = 'recall'"
            ).fetchone()[0]
            == 1
        )
        for table in (
            "read_receipts",
            "access_receipts",
            "retrieval_receipts",
            "evidence_packets",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM recall_run_artifacts").fetchone()[0] == 2
        output_hashes = {
            str(row[0])
            for row in connection.execute(
                "SELECT output_hash FROM processing_runs WHERE kind IN ('recall', 'replay')"
            ).fetchall()
        }
        assert output_hashes == {first.packet.packet_hash}
    finally:
        context.repository.close()


def test_large_manifest_projection_bounds_lineage_queries_by_unique_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, paragraph_count=450)
    try:
        snapshot = context.backend.get_corpus_snapshot(context.request.corpus_snapshot_id)
        scope = RequestScope(
            library_id=context.request.library_id,
            snapshot_hash=snapshot.manifest_hash,
            purpose=context.request.purpose,
            collection_ids=context.request.collection_ids,
            exclusions=context.request.exclusions,
        )
        authorization = context.backend.authorize_read(
            access_policy_id=context.request.access_policy_id,
            scope=scope,
        )
        fragment_ids = tuple(
            item.source_fragment_id for item in authorization.compiled.manifest.items
        )
        assert len(fragment_ids) > 400
        original = context.repository._lineage_for_source
        calls = 0

        def counted_lineage(connection: Any, source_id: str) -> Any:
            nonlocal calls
            calls += 1
            return original(connection, source_id)

        monkeypatch.setattr(context.repository, "_lineage_for_source", counted_lineage)
        with context.repository._store.transaction() as connection:
            projections = context.backend._load_fragment_projections(connection, fragment_ids)

        assert set(projections) == set(fragment_ids)
        assert len({item.source_id for item in projections.values()}) == 1
        assert calls == 1
    finally:
        context.repository.close()


@pytest.mark.parametrize("corruption", ["missing", "text"])
def test_large_manifest_projection_failure_cleans_temporary_read_gate(
    tmp_path: Path,
    corruption: str,
) -> None:
    context = _context(tmp_path, paragraph_count=450)
    try:
        snapshot = context.backend.get_corpus_snapshot(context.request.corpus_snapshot_id)
        authorization = context.backend.authorize_read(
            access_policy_id=context.request.access_policy_id,
            scope=RequestScope(
                library_id=context.request.library_id,
                snapshot_hash=snapshot.manifest_hash,
                purpose=context.request.purpose,
                collection_ids=context.request.collection_ids,
                exclusions=context.request.exclusions,
            ),
        )
        fragment_ids = tuple(
            item.source_fragment_id for item in authorization.compiled.manifest.items
        )
        connection = context.repository._store.connection
        if corruption == "missing":
            selected_ids = (*fragment_ids, "fragment_missing")
            message = "exact persisted row set"
        else:
            selected_ids = fragment_ids
            connection.execute("DROP TRIGGER source_fragments_no_update")
            connection.execute(
                "UPDATE source_fragments SET text = 'tampered' WHERE source_fragment_id = ?",
                (fragment_ids[0],),
            )
            message = "text hash"

        with (
            pytest.raises(PersistenceIntegrityError, match=message),
            context.repository._store.transaction() as transaction,
        ):
            context.backend._load_fragment_projections(transaction, selected_ids)

        assert context.repository._fragment_text_read_depth == 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_temp_master "
                "WHERE type = 'table' AND name = 'dithyramba_projection_read_permit'"
            ).fetchone()[0]
            == 0
        )
    finally:
        context.repository.close()


def test_large_manifest_operational_replay_preserves_packet_bytes(tmp_path: Path) -> None:
    context = _context(tmp_path, paragraph_count=450)
    try:
        service = _service(context)
        first = service.recall(context.request)
        replay = service.replay(first.packet.evidence_packet_id)

        assert first.run.status is ProcessingRunStatus.SUCCEEDED
        assert replay.run.status is ProcessingRunStatus.SUCCEEDED
        assert replay.packet.canonical_bytes == first.packet.canonical_bytes
        assert replay.packet.packet_hash == first.packet.packet_hash
    finally:
        context.repository.close()


def test_verify_replay_is_store_read_only(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        service = _service(context)
        first = service.recall(context.request)
        connection = context.repository._store.connection
        before = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "processing_runs",
                "recall_run_artifacts",
                "coverage_reports",
                "read_receipts",
                "access_receipts",
                "retrieval_receipts",
                "evidence_packets",
            )
        }

        verified = service.verify_replay(first.packet.evidence_packet_id)

        assert verified.canonical_bytes == first.packet.canonical_bytes
        after = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before
        }
        assert after == before
    finally:
        context.repository.close()


@pytest.mark.parametrize("fault", ["runtime", "interrupt"])
def test_sqlite_begin_post_commit_fault_is_reconciled_without_running_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    context = _context(tmp_path)
    try:
        original = context.backend.begin_recall_run
        committed_id = ""

        def begin_then_fault(**kwargs: Any) -> NoReturn:
            nonlocal committed_id
            committed = original(**kwargs)
            committed_id = committed.processing_run_id
            if fault == "runtime":
                raise RuntimeError("simulated post-commit begin failure")
            raise KeyboardInterrupt

        monkeypatch.setattr(context.backend, "begin_recall_run", begin_then_fault)

        expected = RecallPersistenceError if fault == "runtime" else KeyboardInterrupt
        with pytest.raises(expected):
            _service(context).recall(context.request)

        run = context.repository.get_processing_run(committed_id)
        assert run.status is ProcessingRunStatus.FAILED
        assert run.error_code == (
            RecallPersistenceError.code if fault == "runtime" else "recall_interrupted"
        )
        assert (
            context.repository._store.connection.execute(
                "SELECT COUNT(*) FROM processing_runs WHERE kind = 'recall' AND status = 'running'"
            ).fetchone()[0]
            == 0
        )
    finally:
        context.repository.close()


@pytest.mark.parametrize("fault", ["runtime", "interrupt"])
def test_sqlite_completion_post_commit_fault_preserves_exact_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    context = _context(tmp_path)
    try:
        original = context.backend.complete_recall_run
        committed_id = ""

        def complete_then_fault(
            *,
            processing_run_id: str,
            packet: EvidencePacket,
        ) -> NoReturn:
            nonlocal committed_id
            original(processing_run_id=processing_run_id, packet=packet)
            committed_id = processing_run_id
            if fault == "runtime":
                raise RuntimeError("simulated post-commit completion failure")
            raise KeyboardInterrupt

        monkeypatch.setattr(context.backend, "complete_recall_run", complete_then_fault)

        if fault == "runtime":
            result = _service(context).recall(context.request)
            assert result.run.status is ProcessingRunStatus.SUCCEEDED
            assert result.run.output_hash == result.packet.packet_hash
        else:
            with pytest.raises(KeyboardInterrupt):
                _service(context).recall(context.request)

        run = context.repository.get_processing_run(committed_id)
        assert run.status is ProcessingRunStatus.SUCCEEDED
        assert run.error_code is None
        assert (
            context.repository._store.connection.execute(
                "SELECT COUNT(*) FROM processing_runs WHERE kind = 'recall' AND status = 'failed'"
            ).fetchone()[0]
            == 0
        )
    finally:
        context.repository.close()


def test_completion_returns_validated_packet_without_rehydrating_full_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, paragraph_count=12)
    try:
        load_calls = 0

        def forbidden_full_load(_packet_id: str) -> NoReturn:
            nonlocal load_calls
            load_calls += 1
            raise AssertionError("completion must not rehydrate its full persisted receipt")

        monkeypatch.setattr(context.backend, "load_evidence_packet", forbidden_full_load)
        result = _service(context).recall(context.request)

        assert load_calls == 0
        assert result.run.output_hash == result.packet.packet_hash
        assert result.packet.read_receipt.items
        independently_loaded = SQLiteRecallBackend(context.repository).load_evidence_packet(
            result.packet.evidence_packet_id
        )
        assert independently_loaded.canonical_bytes == result.packet.canonical_bytes
    finally:
        context.repository.close()


def test_sqlite_failure_recording_interrupt_before_commit_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    try:

        def broken_fts(**kwargs: Any) -> NoReturn:
            del kwargs
            raise RuntimeError("force an ordinary recall failure")

        original = context.backend.fail_recall_run
        calls = 0

        def interrupt_once(**kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return original(**kwargs)

        monkeypatch.setattr(context.backend, "fail_recall_run", interrupt_once)
        service = RecallService(
            context.backend,
            profile_version=context.profile_version,
            code_version="0.1.0.dev3",
            fts_search=broken_fts,
        )

        with pytest.raises(KeyboardInterrupt):
            service.recall(context.request)

        rows = context.repository._store.connection.execute(
            "SELECT status, error_code FROM processing_runs WHERE kind = 'recall'"
        ).fetchall()
        assert calls == 2
        assert [(str(row[0]), str(row[1])) for row in rows] == [
            ("failed", "recall_execution_failed")
        ]
    finally:
        context.repository.close()


def test_empty_permitted_snapshot_persists_real_no_evidence_packet(tmp_path: Path) -> None:
    context = _context(tmp_path, with_source=False)
    try:
        result = _service(context).recall(context.request)
        packet = result.packet
        assert packet.result_status is PacketResultStatus.NO_EVIDENCE
        assert packet.source_fragments == ()
        assert packet.read_receipt.items == ()
        assert packet.retrieval_receipt.candidate_count == 0
        assert packet.coverage_report.processed_count == 0
        assert (
            context.backend.load_evidence_packet(packet.evidence_packet_id).canonical_bytes
            == packet.canonical_bytes
        )
    finally:
        context.repository.close()


def test_failed_run_is_terminal_audited_and_cannot_be_rewritten(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        context.backend.persist_query_request(context.request)
        run = context.backend.begin_recall_run(
            processing_run_id="run_failed_terminal",
            kind="recall",
            query_request_id=context.request.query_request_id,
            code_version="0.1.0.dev3",
            profile_version=context.profile_version,
        )
        error_hash = canonical_sha256_hex(
            {"schema": "test.failure/1.0", "error_code": "forced_failure"}
        )
        failed = context.backend.fail_recall_run(
            processing_run_id=run.processing_run_id,
            error_code="forced_failure",
            error_hash=error_hash,
        )
        assert failed.status is ProcessingRunStatus.FAILED
        assert failed.error_code == "forced_failure"
        assert failed.output_hash == error_hash
        with pytest.raises(PersistenceConflictError, match="already terminal"):
            context.backend.fail_recall_run(
                processing_run_id=run.processing_run_id,
                error_code="second_failure",
                error_hash=error_hash,
            )
        assert (
            context.repository._store.connection.execute(
                "SELECT COUNT(*) FROM recall_run_artifacts WHERE processing_run_id = ?",
                (run.processing_run_id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        context.repository.close()


@pytest.mark.parametrize("target", ["access_json", "source_text"])
def test_packet_loader_fails_closed_on_persisted_corruption(
    tmp_path: Path,
    target: str,
) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        connection = context.repository._store.connection
        if target == "access_json":
            connection.execute("DROP TRIGGER access_receipts_no_update")
            connection.execute(
                "UPDATE access_receipts SET receipt_json = '{}' WHERE access_receipt_id = ?",
                (packet.access_receipt.access_receipt_id,),
            )
        else:
            connection.execute("DROP TRIGGER source_fragments_no_update")
            fragment_id = packet.read_receipt.items[0].source_fragment_id
            connection.execute(
                "UPDATE source_fragments SET text = 'tampered' WHERE source_fragment_id = ?",
                (fragment_id,),
            )
        with pytest.raises(PersistenceIntegrityError):
            context.backend.load_evidence_packet(packet.evidence_packet_id)
    finally:
        context.repository.close()


def test_query_request_cannot_cross_library_or_snapshot_scope(tmp_path: Path) -> None:
    context = _context(tmp_path / "first")
    other = _context(tmp_path / "second")
    try:
        cross_library = context.request.model_copy(
            update={"library_id": other.repository.library_id}
        )
        with pytest.raises(PersistenceIntegrityError, match="different Library"):
            context.backend.persist_query_request(cross_library)

        wrong_scope = context.request.model_copy(
            update={"collection_ids": (other.request.collection_ids[0],)}
        )
        with pytest.raises(PersistenceIntegrityError):
            context.backend.persist_query_request(wrong_scope)
        assert (
            context.repository._store.connection.execute(
                "SELECT COUNT(*) FROM query_requests"
            ).fetchone()[0]
            == 0
        )
    finally:
        context.repository.close()
        other.repository.close()


def test_public_adapter_validates_types_kinds_and_missing_identities(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="LibraryRepository"):
        SQLiteRecallBackend(cast(Any, object()))
    context = _context(tmp_path)
    try:
        assert context.backend.repository is context.repository
        assert (
            context.backend.get_corpus_snapshot(
                context.request.corpus_snapshot_id
            ).corpus_snapshot_id
            == context.request.corpus_snapshot_id
        )
        with pytest.raises(TypeError, match="exact QueryRequest"):
            context.backend.persist_query_request(cast(Any, object()))
        with pytest.raises(Exception, match="QueryRequest does not exist"):
            context.backend.load_query_request("query_absent")
        with pytest.raises(ValueError, match="recall or replay"):
            context.backend.begin_recall_run(
                processing_run_id="run_invalid_kind",
                kind=cast(Any, "ingest"),
                query_request_id=context.request.query_request_id,
                code_version="0.1.0.dev3",
                profile_version=context.profile_version,
            )
        with pytest.raises(TypeError, match="exact EvidencePacket"):
            context.backend.complete_recall_run(
                processing_run_id="run_absent",
                packet=cast(Any, object()),
            )
        with pytest.raises(Exception, match="EvidencePacket does not exist"):
            context.backend.load_evidence_packet("packet_absent")
        with pytest.raises(Exception, match="ProcessingRun does not exist"):
            context.backend.fail_recall_run(
                processing_run_id="run_absent",
                error_code="forced_failure",
                error_hash="a" * 64,
            )
    finally:
        context.repository.close()


def _canonical_text(payload: dict[str, object]) -> str:
    return canonical_json_bytes(payload).decode("utf-8")


def test_strict_json_reconstructors_reject_malformed_and_reordered_payloads(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet

        query_payload = cast(dict[str, object], json.loads(context.request.canonical_bytes))
        assert (
            recall_persistence._query_request_from_json(
                context.request.canonical_bytes.decode("utf-8")
            )
            == context.request
        )
        query_variants: list[dict[str, object]] = []
        wrong_schema = dict(query_payload)
        wrong_schema["schema"] = "wrong"
        query_variants.append(wrong_schema)
        wrong_nested = dict(query_payload)
        wrong_nested["collection_ids"] = "collection_wrong"
        query_variants.append(wrong_nested)
        wrong_exclusions = dict(query_payload)
        wrong_exclusions["exclusions"] = {
            "source_ids": "source_wrong",
            "source_family_ids": [],
            "source_fragment_ids": [],
        }
        query_variants.append(wrong_exclusions)
        invalid_question = dict(query_payload)
        invalid_question["question"] = 7
        query_variants.append(invalid_question)
        reordered = dict(query_payload)
        reordered["exclusions"] = {
            "source_ids": ["source_z", "source_a"],
            "source_family_ids": [],
            "source_fragment_ids": [],
        }
        query_variants.append(reordered)
        for payload in query_variants:
            with pytest.raises(PersistenceIntegrityError):
                recall_persistence._query_request_from_json(_canonical_text(payload))

        coverage_payload = packet.coverage_report.semantic_payload()
        assert recall_persistence._coverage_from_payload(coverage_payload) == (
            packet.coverage_report
        )
        coverage_variants: list[dict[str, object]] = []
        invalid_coverage_shape = dict(coverage_payload)
        invalid_coverage_shape["stage"] = "ingest"
        coverage_variants.append(invalid_coverage_shape)
        invalid_policy = dict(coverage_payload)
        invalid_policy["policy_omission"] = {"present": 1, "count": None}
        coverage_variants.append(invalid_policy)
        invalid_omission_shape = dict(coverage_payload)
        invalid_omission_shape["omissions"] = [1]
        coverage_variants.append(invalid_omission_shape)
        invalid_omission = dict(coverage_payload)
        invalid_omission["omissions"] = [
            {
                "category": "budget",
                "disclosure": "counted",
                "count": 1,
                "reason_code": "INVALID",
            }
        ]
        coverage_variants.append(invalid_omission)
        reordered_omissions = dict(coverage_payload)
        reordered_omissions["omissions"] = [
            {
                "category": "explicit_exclusion",
                "disclosure": "counted",
                "count": 1,
                "reason_code": "explicit_query_exclusion",
            },
            {
                "category": "budget",
                "disclosure": "counted",
                "count": 1,
                "reason_code": "selection_budget",
            },
        ]
        coverage_variants.append(reordered_omissions)
        for payload in coverage_variants:
            with pytest.raises(PersistenceIntegrityError):
                recall_persistence._coverage_from_payload(payload)

        access_payload = packet.access_receipt.semantic_payload()
        assert recall_persistence._access_from_payload(access_payload) == packet.access_receipt
        wrong_access_schema = dict(access_payload)
        wrong_access_schema["schema"] = "wrong"
        invalid_access = dict(access_payload)
        invalid_access["policy_omission_present"] = 1
        for payload in (wrong_access_schema, invalid_access):
            with pytest.raises(PersistenceIntegrityError):
                recall_persistence._access_from_payload(payload)

        retrieval_payload = packet.retrieval_receipt.semantic_payload()
        assert recall_persistence._retrieval_from_payload(retrieval_payload) == (
            packet.retrieval_receipt
        )
        wrong_retrieval_shape = dict(retrieval_payload)
        wrong_retrieval_shape["trace"] = "invalid"
        invalid_trace_item = dict(retrieval_payload)
        invalid_trace_item["trace"] = [1]
        invalid_trace_rank = dict(retrieval_payload)
        invalid_trace_rank["trace"] = [
            {"source_fragment_id": "fragment_a", "rank": True, "score": "0.000000"}
        ]
        reordered_trace = dict(retrieval_payload)
        reordered_trace["trace"] = list(reversed(cast(list[object], retrieval_payload["trace"])))
        for payload in (
            wrong_retrieval_shape,
            invalid_trace_item,
            invalid_trace_rank,
            reordered_trace,
        ):
            with pytest.raises(PersistenceIntegrityError):
                recall_persistence._retrieval_from_payload(payload)

        for value in ("not-json", "[]", '{"a": 1}', '{"a":1.5}'):
            with pytest.raises(PersistenceIntegrityError):
                recall_persistence._load_canonical_object(value, "test")
        with pytest.raises(PersistenceIntegrityError):
            recall_persistence._strict_bool(2, "test")
    finally:
        context.repository.close()


@pytest.mark.parametrize(
    "target",
    [
        "terminal_status",
        "packet_json",
        "coverage_scalar",
        "read_hash",
        "access_scalar",
        "retrieval_scalar",
        "packet_item_score",
        "packet_item_delete",
        "no_link",
        "query_hash",
        "source_family",
    ],
)
def test_packet_loader_rejects_relational_projection_corruption(
    tmp_path: Path,
    target: str,
) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        connection = context.repository._store.connection
        run_id = connection.execute(
            "SELECT processing_run_id FROM recall_run_artifacts WHERE evidence_packet_id = ?",
            (packet.evidence_packet_id,),
        ).fetchone()[0]
        if target == "terminal_status":
            connection.execute(
                "UPDATE processing_runs SET status = 'failed' WHERE processing_run_id = ?",
                (run_id,),
            )
        elif target == "packet_json":
            connection.execute("DROP TRIGGER evidence_packets_no_update")
            connection.execute(
                "UPDATE evidence_packets SET packet_json = '{}' WHERE evidence_packet_id = ?",
                (packet.evidence_packet_id,),
            )
        elif target == "coverage_scalar":
            connection.execute("DROP TRIGGER coverage_reports_no_update")
            connection.execute(
                "UPDATE coverage_reports SET processed_count = processed_count + 1 "
                "WHERE coverage_report_id = ?",
                (packet.coverage_report.coverage_report_id,),
            )
        elif target == "read_hash":
            connection.execute("DROP TRIGGER corpus_read_set_items_no_update")
            connection.execute(
                "UPDATE corpus_read_set_items SET text_sha256 = ? "
                "WHERE corpus_read_set_id = ("
                "SELECT corpus_read_set_id FROM read_receipt_corpus_sets "
                "WHERE read_receipt_id = ?)",
                ("a" * 64, packet.read_receipt.read_receipt_id),
            )
        elif target == "access_scalar":
            connection.execute("DROP TRIGGER access_receipts_no_update")
            connection.execute(
                "UPDATE access_receipts SET policy_hash = ? WHERE access_receipt_id = ?",
                ("a" * 64, packet.access_receipt.access_receipt_id),
            )
        elif target == "retrieval_scalar":
            connection.execute("DROP TRIGGER retrieval_receipts_no_update")
            connection.execute(
                "UPDATE retrieval_receipts SET candidate_count = candidate_count + 1 "
                "WHERE retrieval_receipt_id = ?",
                (packet.retrieval_receipt.retrieval_receipt_id,),
            )
        elif target == "packet_item_score":
            connection.execute("DROP TRIGGER packet_items_no_update")
            connection.execute(
                "UPDATE packet_items SET score_text = 'invalid' WHERE evidence_packet_id = ?",
                (packet.evidence_packet_id,),
            )
        elif target == "packet_item_delete":
            connection.execute("DROP TRIGGER packet_items_no_delete")
            connection.execute(
                "DELETE FROM packet_items WHERE evidence_packet_id = ? AND rank = 1",
                (packet.evidence_packet_id,),
            )
        elif target == "no_link":
            connection.execute("DROP TRIGGER recall_run_artifacts_no_delete")
            connection.execute(
                "DELETE FROM recall_run_artifacts WHERE evidence_packet_id = ?",
                (packet.evidence_packet_id,),
            )
        elif target == "query_hash":
            connection.execute("DROP TRIGGER query_requests_no_update")
            connection.execute(
                "UPDATE query_requests SET request_hash = ? WHERE query_request_id = ?",
                ("a" * 64, packet.query_request_id),
            )
        else:
            connection.execute("DROP TRIGGER source_family_members_no_delete")
            connection.execute(
                "DELETE FROM source_family_members WHERE source_id IN ("
                "SELECT sv.source_id FROM source_versions AS sv "
                "JOIN corpus_read_set_items AS rri "
                "JOIN read_receipt_corpus_sets AS rrcs "
                "ON rrcs.corpus_read_set_id = rri.corpus_read_set_id "
                "JOIN source_fragments AS sf ON sf.source_fragment_id = rri.source_fragment_id "
                "AND sf.source_version_id = sv.source_version_id "
                "WHERE rrcs.read_receipt_id = ?)",
                (packet.read_receipt.read_receipt_id,),
            )
        with pytest.raises(PersistenceIntegrityError):
            context.backend.load_evidence_packet(packet.evidence_packet_id)
    finally:
        context.repository.close()


@pytest.mark.parametrize(
    "target",
    ["terminal_status", "packet_json", "packet_item_score", "packet_item_delete", "no_link"],
)
def test_compact_packet_binding_rejects_its_relational_closure_corruption(
    tmp_path: Path,
    target: str,
) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        connection = context.repository._store.connection
        run_id = connection.execute(
            "SELECT processing_run_id FROM recall_run_artifacts WHERE evidence_packet_id = ?",
            (packet.evidence_packet_id,),
        ).fetchone()[0]
        if target == "terminal_status":
            connection.execute(
                "UPDATE processing_runs SET status = 'failed' WHERE processing_run_id = ?",
                (run_id,),
            )
        elif target == "packet_json":
            connection.execute("DROP TRIGGER evidence_packets_no_update")
            connection.execute(
                "UPDATE evidence_packets SET packet_json = '{}' WHERE evidence_packet_id = ?",
                (packet.evidence_packet_id,),
            )
        elif target == "packet_item_score":
            connection.execute("DROP TRIGGER packet_items_no_update")
            connection.execute(
                "UPDATE packet_items SET score_text = 'invalid' WHERE evidence_packet_id = ?",
                (packet.evidence_packet_id,),
            )
        elif target == "packet_item_delete":
            connection.execute("DROP TRIGGER packet_items_no_delete")
            connection.execute(
                "DELETE FROM packet_items WHERE evidence_packet_id = ? AND rank = 1",
                (packet.evidence_packet_id,),
            )
        else:
            connection.execute("DROP TRIGGER recall_run_artifacts_no_delete")
            connection.execute(
                "DELETE FROM recall_run_artifacts WHERE evidence_packet_id = ?",
                (packet.evidence_packet_id,),
            )

        with pytest.raises(PersistenceIntegrityError):
            context.backend.load_evidence_packet_binding(packet.evidence_packet_id)
    finally:
        context.repository.close()


def test_private_artifact_loaders_fail_closed_for_missing_rows_and_bad_identity_query(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        context.backend.persist_query_request(context.request)
        connection = context.repository._store.connection
        with pytest.raises(Exception, match="RecallCoverageReport does not exist"):
            context.backend._load_coverage(
                connection,
                "coverage_absent",
                query_request_id=context.request.query_request_id,
                query_request_hash=context.request.request_hash,
            )
        with pytest.raises(Exception, match="ReadReceipt does not exist"):
            context.backend._load_read_receipt(
                connection,
                "read_absent",
                query_request_id=context.request.query_request_id,
                query_request_hash=context.request.request_hash,
                retrieval_corpus_hash="a" * 64,
            )
        with pytest.raises(Exception, match="AccessReceipt does not exist"):
            context.backend._load_access_receipt(
                connection,
                "access_absent",
                query_request_id=context.request.query_request_id,
            )
        with pytest.raises(Exception, match="RetrievalReceipt does not exist"):
            context.backend._load_retrieval_receipt(
                connection,
                "retrieval_absent",
                query_request_id=context.request.query_request_id,
            )
        with pytest.raises(AssertionError, match="unsupported"):
            recall_persistence._require_unique_artifact_identity(
                connection,
                table="wrong",
                id_column="wrong",
                identifier="wrong",
                hash_column="wrong",
                digest="a" * 64,
            )
        connection.execute("CREATE TABLE temp_identity(identifier TEXT, digest TEXT)")
        with pytest.raises(AssertionError):
            recall_persistence._require_unique_artifact_identity(
                connection,
                table="temp_identity",
                id_column="identifier",
                identifier="id",
                hash_column="digest",
                digest="a" * 64,
            )
    finally:
        context.repository.close()


def test_receipt_and_packet_closure_reject_every_dependency_drift(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        snapshot = context.backend.get_corpus_snapshot(context.request.corpus_snapshot_id)
        connection = context.repository._store.connection

        def validate(
            *,
            request: QueryRequest = context.request,
            selected_snapshot: Any = snapshot,
            coverage: Any = packet.coverage_report,
            read: Any = packet.read_receipt,
            access: Any = packet.access_receipt,
            retrieval: Any = packet.retrieval_receipt,
        ) -> dict[str, Any]:
            return context.backend._validate_receipt_closure(
                connection,
                request=request,
                snapshot=selected_snapshot,
                coverage=coverage,
                read=read,
                access=access,
                retrieval=retrieval,
                code_version="0.1.0.dev3",
                profile_version=context.profile_version,
            )

        projections = validate()
        assert projections

        class DerivedCoverage(RecallCoverageReport):
            pass

        derived = DerivedCoverage.model_validate(packet.coverage_report.model_dump())
        invalid_calls = (
            {"coverage": derived},
            {
                "selected_snapshot": snapshot.model_copy(
                    update={"collection_ids": ("collection_wrong",)}
                )
            },
            {
                "coverage": packet.coverage_report.model_copy(
                    update={"query_request_hash": "a" * 64}
                )
            },
            {"retrieval": packet.retrieval_receipt.model_copy(update={"max_candidates": 1})},
            {"access": packet.access_receipt.model_copy(update={"policy_hash": "a" * 64})},
            {"read": packet.read_receipt.model_copy(update={"items": ()})},
            {
                "read": packet.read_receipt.model_copy(
                    update={
                        "items": (
                            packet.read_receipt.items[0].model_copy(
                                update={"source_version_id": "source_version_wrong"}
                            ),
                            *packet.read_receipt.items[1:],
                        )
                    }
                )
            },
            {
                "access": packet.access_receipt.model_copy(
                    update={"retrieval_corpus_hash": "a" * 64}
                )
            },
            {"coverage": packet.coverage_report.model_copy(update={"processed_count": 99})},
            {
                "retrieval": packet.retrieval_receipt.model_copy(
                    update={
                        "trace": (
                            packet.retrieval_receipt.trace[0].model_copy(
                                update={"source_fragment_id": "fragment_unknown"}
                            ),
                        )
                    }
                )
            },
        )
        for values in invalid_calls:
            with pytest.raises(PersistenceIntegrityError):
                validate(**values)

        with pytest.raises(PersistenceIntegrityError, match="identity closure"):
            context.backend._validate_packet_closure(
                packet.model_copy(update={"query_request_id": "query_wrong"}),
                request=context.request,
                snapshot=snapshot,
                projections=projections,
            )
        with pytest.raises(PersistenceIntegrityError, match="outside"):
            context.backend._validate_packet_closure(
                packet,
                request=context.request,
                snapshot=snapshot,
                projections={},
            )
        first_id = packet.source_fragments[0].source_fragment_id
        wrong_projection = replace(projections[first_id], text="different")
        with pytest.raises(PersistenceIntegrityError, match="stored source"):
            context.backend._validate_packet_closure(
                packet,
                request=context.request,
                snapshot=snapshot,
                projections={**projections, first_id: wrong_projection},
            )
    finally:
        context.repository.close()


def test_running_run_and_missing_policy_integrity_checks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    try:
        connection = context.repository._store.connection
        connection.execute("DROP TRIGGER access_policy_collection_rules_no_delete")
        connection.execute("DROP TRIGGER access_policies_no_delete")
        connection.execute(
            "DELETE FROM access_policy_collection_rules WHERE access_policy_id = ?",
            (context.request.access_policy_id,),
        )
        connection.execute(
            "DELETE FROM access_policies WHERE access_policy_id = ?",
            (context.request.access_policy_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="absent"):
            context.backend._validate_request_dependencies(connection, context.request)
    finally:
        context.repository.close()

    second = _context(tmp_path / "run")
    try:
        second.backend.persist_query_request(second.request)
        run = second.backend.begin_recall_run(
            processing_run_id="run_invalid_running_fields",
            kind="recall",
            query_request_id=second.request.query_request_id,
            code_version="0.1.0.dev3",
            profile_version=second.profile_version,
        )
        connection = second.repository._store.connection
        connection.execute(
            "UPDATE processing_runs SET finished_at = ? WHERE processing_run_id = ?",
            ("2026-07-20T12:00:00.000000Z", run.processing_run_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="invalid fields"):
            second.backend._load_running_recall_run(connection, run.processing_run_id)
    finally:
        second.repository.close()


def test_write_transactions_roll_back_on_integrity_and_lost_update(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        connection = context.repository._store.connection
        existing_event_id = str(
            connection.execute("SELECT event_id FROM event_outbox LIMIT 1").fetchone()[0]
        )
        original_factory = context.repository._event_id_factory
        context.repository._event_id_factory = lambda: existing_event_id
        with pytest.raises(PersistenceConflictError, match="QueryRequest"):
            context.backend.persist_query_request(context.request)
        assert connection.execute("SELECT COUNT(*) FROM query_requests").fetchone()[0] == 0

        context.repository._event_id_factory = original_factory
        context.backend.persist_query_request(context.request)
        context.repository._event_id_factory = lambda: existing_event_id
        with pytest.raises(PersistenceConflictError, match="insert conflicted"):
            context.backend.begin_recall_run(
                processing_run_id="run_conflicting_start_event",
                kind="recall",
                query_request_id=context.request.query_request_id,
                code_version="0.1.0.dev3",
                profile_version=context.profile_version,
            )

        context.repository._event_id_factory = original_factory
        packet = _service(context).recall(context.request).packet
        replay_run = context.backend.begin_recall_run(
            processing_run_id="run_conflicting_completion_event",
            kind="replay",
            query_request_id=context.request.query_request_id,
            code_version="0.1.0.dev3",
            profile_version=context.profile_version,
        )
        context.repository._event_id_factory = lambda: existing_event_id
        with pytest.raises(PersistenceConflictError, match="completion conflicted"):
            context.backend.complete_recall_run(
                processing_run_id=replay_run.processing_run_id,
                packet=packet,
            )
        assert (
            context.repository.get_processing_run(replay_run.processing_run_id).status
            is ProcessingRunStatus.RUNNING
        )

        context.repository._event_id_factory = original_factory
        failed_run = context.backend.begin_recall_run(
            processing_run_id="run_conflicting_failure_event",
            kind="recall",
            query_request_id=context.request.query_request_id,
            code_version="0.1.0.dev3",
            profile_version=context.profile_version,
        )
        context.repository._event_id_factory = lambda: existing_event_id
        with pytest.raises(PersistenceConflictError, match="failure transition"):
            context.backend.fail_recall_run(
                processing_run_id=failed_run.processing_run_id,
                error_code="forced_failure",
                error_hash="a" * 64,
            )
        assert (
            context.repository.get_processing_run(failed_run.processing_run_id).status
            is ProcessingRunStatus.RUNNING
        )
    finally:
        context.repository.close()


@pytest.mark.parametrize("terminal", ["succeeded", "failed"])
def test_lost_update_guards_fail_closed(tmp_path: Path, terminal: str) -> None:
    context = _context(tmp_path)
    try:
        service = _service(context)
        packet = service.recall(context.request).packet
        run = context.backend.begin_recall_run(
            processing_run_id=f"run_lost_update_{terminal}",
            kind="replay" if terminal == "succeeded" else "recall",
            query_request_id=context.request.query_request_id,
            code_version="0.1.0.dev3",
            profile_version=context.profile_version,
        )
        connection = context.repository._store.connection
        connection.execute(
            f"""
            CREATE TRIGGER ignore_test_terminal_update
            BEFORE UPDATE ON processing_runs
            WHEN NEW.status = '{terminal}'
            BEGIN
                SELECT RAISE(IGNORE);
            END
            """
        )
        if terminal == "succeeded":
            with pytest.raises(PersistenceConflictError, match="changed during completion"):
                context.backend.complete_recall_run(
                    processing_run_id=run.processing_run_id,
                    packet=packet,
                )
        else:
            with pytest.raises(PersistenceConflictError, match="changed during failure"):
                context.backend.fail_recall_run(
                    processing_run_id=run.processing_run_id,
                    error_code="forced_failure",
                    error_hash="a" * 64,
                )
    finally:
        context.repository.close()


def test_artifact_projection_helpers_detect_mismatch_and_cover_omission_identity(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        connection = context.repository._store.connection
        with pytest.raises(PersistenceIntegrityError, match="ID/hash"):
            recall_persistence._require_unique_artifact_identity(
                connection,
                table="read_receipts",
                id_column="read_receipt_id",
                identifier="read_wrong",
                hash_column="receipt_hash",
                digest=packet.read_receipt.receipt_hash,
            )
        omission = RecallOmission(
            category=OmissionCategory.BUDGET,
            disclosure=OmissionDisclosure.COUNTED,
            count=1,
            reason_code="selection_budget",
        )
        assert recall_persistence._recall_omission_id(
            packet.coverage_report.coverage_report_id,
            omission,
        ).startswith("omission_")

        connection.execute(
            """
            INSERT INTO omissions(
                omission_id, coverage_report_id, category,
                disclosure, count, reason_code
            ) VALUES (?, ?, 'budget', 'counted', 1, 'unexpected_budget')
            """,
            (
                "omission_extra",
                packet.coverage_report.coverage_report_id,
            ),
        )
        with pytest.raises(PersistenceIntegrityError, match="omissions differ"):
            context.backend._load_coverage(
                connection,
                packet.coverage_report.coverage_report_id,
                query_request_id=context.request.query_request_id,
                query_request_hash=context.request.request_hash,
            )
    finally:
        context.repository.close()


def test_loader_rejects_conflicting_run_links_and_private_cross_library_request(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path / "first")
    other = _context(tmp_path / "other")
    try:
        first = _service(context).recall(context.request).packet
        second_request = context.request.model_copy(
            update={"question": "A different canonical question about expressionism"}
        )
        second = _service(context).recall(second_request).packet
        connection = context.repository._store.connection
        connection.execute("DROP TRIGGER recall_run_artifacts_no_update")
        connection.execute(
            "UPDATE recall_run_artifacts SET evidence_packet_id = ? WHERE evidence_packet_id = ?",
            (first.evidence_packet_id, second.evidence_packet_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="conflicting"):
            context.backend.load_evidence_packet(first.evidence_packet_id)

        with pytest.raises(PersistenceIntegrityError, match="different Library"):
            context.backend._validate_request_dependencies(connection, other.request)
    finally:
        context.repository.close()
        other.repository.close()


def test_fragment_projection_and_packet_item_internal_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        snapshot = context.backend.get_corpus_snapshot(context.request.corpus_snapshot_id)
        connection = context.repository._store.connection
        fragment_ids = tuple(item.source_fragment_id for item in packet.read_receipt.items)
        projections = context.backend._load_fragment_projections(connection, fragment_ids)

        original_projection_loader = context.backend._load_fragment_projections
        monkeypatch.setattr(
            context.backend,
            "_load_fragment_projections",
            lambda _connection, _ids: {},
        )
        with pytest.raises(PersistenceIntegrityError, match="absent"):
            context.backend._validate_receipt_closure(
                connection,
                request=context.request,
                snapshot=snapshot,
                coverage=packet.coverage_report,
                read=packet.read_receipt,
                access=packet.access_receipt,
                retrieval=packet.retrieval_receipt,
                code_version="0.1.0.dev3",
                profile_version=context.profile_version,
            )
        monkeypatch.setattr(
            context.backend,
            "_load_fragment_projections",
            original_projection_loader,
        )

        original_lineage = context.repository._lineage_for_source
        monkeypatch.setattr(
            context.repository,
            "_lineage_for_source",
            lambda _connection, source_id: ("family_wrong", source_id, "root"),
        )
        with pytest.raises(PersistenceIntegrityError, match="SourceFamily"):
            context.backend._load_fragment_projections(connection, fragment_ids)
        monkeypatch.setattr(context.repository, "_lineage_for_source", original_lineage)

        connection.execute("DROP TRIGGER packet_items_no_update")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE packet_items SET role = 'wrong' WHERE evidence_packet_id = ?",
            (packet.evidence_packet_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="role"):
            context.backend._load_packet_items(
                connection,
                evidence_packet_id=packet.evidence_packet_id,
                projections=projections,
            )
        connection.execute(
            "UPDATE packet_items SET role = 'evidence' WHERE evidence_packet_id = ?",
            (packet.evidence_packet_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="outside"):
            context.backend._load_packet_items(
                connection,
                evidence_packet_id=packet.evidence_packet_id,
                projections={},
            )

        connection.execute("DROP TRIGGER source_family_members_no_delete")
        source_id = next(iter(projections.values())).source_id
        connection.execute(
            "DELETE FROM source_family_members WHERE source_id = ?",
            (source_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="one-family"):
            context.backend._load_fragment_projections(connection, fragment_ids)
    finally:
        context.repository.close()


def test_insert_or_verify_helpers_reject_mocked_reload_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        connection = context.repository._store.connection
        run_id = str(
            connection.execute(
                "SELECT processing_run_id FROM recall_run_artifacts WHERE evidence_packet_id = ?",
                (packet.evidence_packet_id,),
            ).fetchone()[0]
        )
        original_coverage = context.backend._load_coverage
        monkeypatch.setattr(
            context.backend,
            "_load_coverage",
            lambda *_args, **_kwargs: packet.coverage_report.model_copy(
                update={"processed_count": 999}
            ),
        )
        with (
            pytest.raises(PersistenceIntegrityError, match="CoverageReport conflicts"),
            context.repository._store.transaction(immediate=True) as transaction,
        ):
            context.backend._ensure_coverage(
                transaction,
                processing_run_id=run_id,
                query_request_id=context.request.query_request_id,
                coverage=packet.coverage_report,
            )
        monkeypatch.setattr(context.backend, "_load_coverage", original_coverage)

        original_read = context.backend._load_read_receipt
        monkeypatch.setattr(
            context.backend,
            "_load_read_receipt",
            lambda *_args, **_kwargs: packet.read_receipt.model_copy(update={"items": ()}),
        )
        with (
            pytest.raises(PersistenceIntegrityError, match="ReadReceipt conflicts"),
            context.repository._store.transaction(immediate=True) as transaction,
        ):
            context.backend._ensure_read_receipt(
                transaction,
                processing_run_id=run_id,
                query_request_id=context.request.query_request_id,
                receipt=packet.read_receipt,
            )
        monkeypatch.setattr(context.backend, "_load_read_receipt", original_read)

        original_access = context.backend._load_access_receipt
        monkeypatch.setattr(
            context.backend,
            "_load_access_receipt",
            lambda *_args, **_kwargs: packet.access_receipt.model_copy(
                update={"policy_hash": "a" * 64}
            ),
        )
        with (
            pytest.raises(PersistenceIntegrityError, match="AccessReceipt conflicts"),
            context.repository._store.transaction(immediate=True) as transaction,
        ):
            context.backend._ensure_access_receipt(
                transaction,
                processing_run_id=run_id,
                query_request_id=context.request.query_request_id,
                receipt=packet.access_receipt,
            )
        monkeypatch.setattr(context.backend, "_load_access_receipt", original_access)

        original_retrieval = context.backend._load_retrieval_receipt
        monkeypatch.setattr(
            context.backend,
            "_load_retrieval_receipt",
            lambda *_args, **_kwargs: packet.retrieval_receipt.model_copy(
                update={"candidate_count": 999}
            ),
        )
        with (
            pytest.raises(PersistenceIntegrityError, match="RetrievalReceipt conflicts"),
            context.repository._store.transaction(immediate=True) as transaction,
        ):
            context.backend._ensure_retrieval_receipt(
                transaction,
                processing_run_id=run_id,
                query_request_id=context.request.query_request_id,
                receipt=packet.retrieval_receipt,
            )
        monkeypatch.setattr(context.backend, "_load_retrieval_receipt", original_retrieval)
    finally:
        context.repository.close()


def test_normalized_artifact_loaders_reject_wrong_run_and_corrupt_item_identity(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    try:
        packet = _service(context).recall(context.request).packet
        connection = context.repository._store.connection
        with pytest.raises(PersistenceIntegrityError, match="run closure"):
            context.backend._load_coverage(
                connection,
                packet.coverage_report.coverage_report_id,
                query_request_id="query_wrong",
                query_request_hash=context.request.request_hash,
            )
        with pytest.raises(PersistenceIntegrityError, match="first-run closure"):
            context.backend._load_read_receipt(
                connection,
                packet.read_receipt.read_receipt_id,
                query_request_id="query_wrong",
                query_request_hash=context.request.request_hash,
                retrieval_corpus_hash=packet.read_receipt.retrieval_corpus_hash,
            )
        with pytest.raises(PersistenceIntegrityError, match="first-run closure"):
            context.backend._load_access_receipt(
                connection,
                packet.access_receipt.access_receipt_id,
                query_request_id="query_wrong",
            )
        with pytest.raises(PersistenceIntegrityError, match="first-run closure"):
            context.backend._load_retrieval_receipt(
                connection,
                packet.retrieval_receipt.retrieval_receipt_id,
                query_request_id="query_wrong",
            )

        connection.execute("DROP TRIGGER corpus_read_set_items_no_update")
        connection.execute("DROP TRIGGER read_receipts_no_update")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE corpus_read_set_items SET read_order = -1 "
            "WHERE corpus_read_set_id = ("
            "SELECT corpus_read_set_id FROM read_receipt_corpus_sets "
            "WHERE read_receipt_id = ?) AND read_order = 0",
            (packet.read_receipt.read_receipt_id,),
        )
        with pytest.raises(PersistenceIntegrityError, match="CorpusReadSet items are invalid"):
            context.backend._load_read_receipt(
                connection,
                packet.read_receipt.read_receipt_id,
                query_request_id=context.request.query_request_id,
                query_request_hash=context.request.request_hash,
                retrieval_corpus_hash=packet.read_receipt.retrieval_corpus_hash,
            )
        connection.execute(
            "UPDATE corpus_read_set_items SET read_order = 0 "
            "WHERE corpus_read_set_id = ("
            "SELECT corpus_read_set_id FROM read_receipt_corpus_sets "
            "WHERE read_receipt_id = ?) AND read_order = -1",
            (packet.read_receipt.read_receipt_id,),
        )
        connection.execute(
            "UPDATE read_receipts SET receipt_hash = ? WHERE read_receipt_id = ?",
            ("a" * 64, packet.read_receipt.read_receipt_id),
        )
        with pytest.raises(PersistenceIntegrityError, match="identity mismatch"):
            context.backend._load_read_receipt(
                connection,
                packet.read_receipt.read_receipt_id,
                query_request_id=context.request.query_request_id,
                query_request_hash=context.request.request_hash,
                retrieval_corpus_hash=packet.read_receipt.retrieval_corpus_hash,
            )
    finally:
        context.repository.close()
