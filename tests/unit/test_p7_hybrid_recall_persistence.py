"""Atomic, text-free persistence tests for complete P7 hybrid recalls."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from dithyramba.persistence.errors import PersistenceIntegrityError
from dithyramba.persistence.hybrid_recall import (
    HybridSuccessfulRunRecord,
    SQLiteHybridRecallBackend,
)
from dithyramba.recall.hybrid_engine import HybridExecutionArtifacts, execute_hybrid_retrieval
from dithyramba.recall.hybrid_models import HybridQueryRequest, HybridRetrievalBudget
from dithyramba.recall.semantic_provider import SemanticProvider
from dithyramba.recall.vector import PackedVector, pack_normalized_vector
from tests.unit.test_p7_hybrid_persistence import HybridContext, _close, _context


class StaticSemanticProvider:
    def __init__(self, context: HybridContext) -> None:
        self.model_profile = context.model
        self.runtime_profile = context.runtime

    def embed_query(self, question: str) -> PackedVector:
        assert question
        return pack_normalized_vector((1.0, 0.0))

    def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
        raise AssertionError(f"recall may not re-embed persisted passages: {texts!r}")


@dataclass(frozen=True, slots=True)
class RecallCase:
    context: HybridContext
    backend: SQLiteHybridRecallBackend
    request: HybridQueryRequest
    run: HybridSuccessfulRunRecord
    artifacts: HybridExecutionArtifacts


def _recall_case(tmp_path: Path, *, suffix: str = "main") -> RecallCase:
    context = _context(tmp_path, suffix=suffix)
    request = HybridQueryRequest(
        question="What alpha evidence is present?",
        library_id=context.repository.library_id,
        collection_ids=(context.collection_id,),
        corpus_snapshot_id=context.snapshot.corpus_snapshot_id,
        access_policy_id=context.policy.access_policy_id,
        purpose="research",
        model_profile_hash=context.model.profile_hash,
        runtime_profile_hash=context.runtime.profile_hash,
        layer_profile_hash=context.layers.profile_hash,
        fusion_profile_hash=context.fusion.profile_hash,
        retrieval=HybridRetrievalBudget(
            max_fts_candidates=2,
            max_dense_candidates=2,
            max_fused_candidates=2,
            max_source_fragments=2,
        ),
    )
    backend = SQLiteHybridRecallBackend(context.repository)
    backend.persist_query_request(request)
    authorization = context.repository.authorize_read(
        access_policy_id=context.policy.access_policy_id,
        scope=context.scope,
    )
    fragments = context.repository.read_permitted_fragments(authorization)
    _generation, ordered_vectors = context.store.load_corpus_vector_generation(
        context.generation.generation_hash
    )
    artifacts = execute_hybrid_retrieval(
        request=request,
        snapshot=context.snapshot,
        compiled_access=authorization.compiled,
        fragments=fragments,
        layer_profile=context.layers,
        fusion_profile=context.fusion,
        vector_generation=context.generation,
        vectors=ordered_vectors,
        semantic_provider=cast(SemanticProvider, StaticSemanticProvider(context)),
    )
    run = HybridSuccessfulRunRecord(
        processing_run_id=f"run_{suffix}",
        request_id=request.request_id,
        vector_generation_hash=context.generation.generation_hash,
        kind="recall",
        code_version="0.1.0rc0",
        profile_version="p7-hybrid-v1",
        started_at="2026-07-21T12:00:00.000000Z",
        finished_at="2026-07-21T12:00:01.000000Z",
        output_hash=artifacts.packet.packet_hash,
    )
    return RecallCase(context, backend, request, run, artifacts)


def test_complete_run_round_trips_idempotently_without_raw_text(tmp_path: Path) -> None:
    case = _recall_case(tmp_path)
    try:
        assert case.backend.persist_query_request(case.request) == case.request
        assert case.backend.persist_successful_run(case.run, case.artifacts) == (
            case.run,
            case.artifacts,
        )
        assert case.backend.persist_successful_run(case.run, case.artifacts) == (
            case.run,
            case.artifacts,
        )
        assert case.backend.load_successful_run(case.run.processing_run_id) == (
            case.run,
            case.artifacts,
        )
        connection = case.context.repository._store.connection
        assert connection.execute("SELECT COUNT(*) FROM hybrid_processing_runs").fetchone()[0] == 1
        persisted = connection.execute(
            """
            SELECT fts_profile_version, fts_result_hash, query_vector_hash
            FROM hybrid_retrieval_receipts
            """
        ).fetchone()
        assert tuple(persisted) == (
            case.artifacts.retrieval_receipt.fts_profile_version,
            case.artifacts.retrieval_receipt.fts_result_hash,
            case.artifacts.retrieval_receipt.query_vector_hash,
        )
        p7_tables = tuple(
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE type = 'table' AND (
                    name LIKE 'hybrid_%' OR name LIKE 'corpus_vector_%'
                )
                ORDER BY name
                """
            ).fetchall()
        )
        for table in p7_tables:
            columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
            assert "text" not in columns
            assert "question" not in columns or table == "hybrid_query_requests"
    finally:
        _close(case.context)


def test_run_load_fails_closed_when_receipt_hash_input_is_corrupted(tmp_path: Path) -> None:
    case = _recall_case(tmp_path)
    try:
        case.backend.persist_successful_run(case.run, case.artifacts)
        connection = case.context.repository._store.connection
        connection.execute("DROP TRIGGER hybrid_retrieval_receipts_no_update")
        connection.execute(
            "UPDATE hybrid_retrieval_receipts SET fts_result_hash = ?",
            ("f" * 64,),
        )
        with pytest.raises(PersistenceIntegrityError, match="hybrid artifacts"):
            case.backend.load_successful_run(case.run.processing_run_id)
    finally:
        _close(case.context)


def test_replay_run_reuses_identical_content_addressed_artifacts(tmp_path: Path) -> None:
    case = _recall_case(tmp_path)
    replay = replace(
        case.run,
        processing_run_id="run_replay",
        kind="replay",
        started_at="2026-07-21T12:01:00.000000Z",
        finished_at="2026-07-21T12:01:01.000000Z",
    )
    try:
        case.backend.persist_successful_run(case.run, case.artifacts)
        assert case.backend.persist_successful_run(replay, case.artifacts) == (
            replay,
            case.artifacts,
        )
        assert case.backend.load_successful_run(replay.processing_run_id) == (
            replay,
            case.artifacts,
        )
        connection = case.context.repository._store.connection
        assert connection.execute("SELECT COUNT(*) FROM hybrid_processing_runs").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM hybrid_run_artifacts").fetchone()[0] == 2
        for table in (
            "hybrid_coverage_reports",
            "hybrid_read_receipts",
            "hybrid_access_receipts",
            "hybrid_retrieval_receipts",
            "hybrid_evidence_packets",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
    finally:
        _close(case.context)


def test_artifact_failure_rolls_back_the_entire_terminal_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _recall_case(tmp_path)
    original = SQLiteHybridRecallBackend._insert_artifacts

    def fail_after_insert(
        backend: SQLiteHybridRecallBackend,
        connection: sqlite3.Connection,
        run: HybridSuccessfulRunRecord,
        artifacts: HybridExecutionArtifacts,
    ) -> None:
        original(backend, connection, run, artifacts)
        raise RuntimeError("simulated terminal write failure")

    monkeypatch.setattr(SQLiteHybridRecallBackend, "_insert_artifacts", fail_after_insert)
    try:
        with pytest.raises(RuntimeError, match="simulated terminal write failure"):
            case.backend.persist_successful_run(case.run, case.artifacts)
        connection = case.context.repository._store.connection
        for table in (
            "hybrid_processing_runs",
            "hybrid_coverage_reports",
            "hybrid_read_receipts",
            "hybrid_access_receipts",
            "hybrid_retrieval_receipts",
            "hybrid_evidence_packets",
            "hybrid_run_artifacts",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        _close(case.context)


def test_query_and_run_tables_are_append_only(tmp_path: Path) -> None:
    case = _recall_case(tmp_path)
    try:
        case.backend.persist_successful_run(case.run, case.artifacts)
        connection = case.context.repository._store.connection
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE hybrid_query_requests SET question = question")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM hybrid_processing_runs")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE hybrid_packet_items SET rank = rank")
    finally:
        _close(case.context)


def test_a_request_from_another_library_is_rejected_before_insert(tmp_path: Path) -> None:
    first = _recall_case(tmp_path, suffix="first")
    second = _context(tmp_path, suffix="second")
    try:
        backend = SQLiteHybridRecallBackend(second.repository)
        with pytest.raises(PersistenceIntegrityError, match="different Library"):
            backend.persist_query_request(first.request)
        assert (
            second.repository._store.connection.execute(
                "SELECT COUNT(*) FROM hybrid_query_requests"
            ).fetchone()[0]
            == 0
        )
    finally:
        _close(first.context, second)
