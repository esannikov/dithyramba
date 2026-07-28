"""Append-only SQLite persistence for complete successful P7 hybrid recalls.

The adapter normalizes every artifact into the frozen 0005 tables. It never
stores arbitrary JSON and never selects raw SourceFragment text. A successful
run is inserted terminally in one transaction; partial run repair and failed
run persistence are deliberately outside this boundary.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from pydantic import ValidationError

from dithyramba.access import QueryExclusions, RequestScope
from dithyramba.persistence.errors import (
    PersistenceConflictError,
    PersistenceIntegrityError,
)
from dithyramba.recall.fusion import (
    FusionContractError,
    select_layer_aware_fragment_ids,
)
from dithyramba.recall.hybrid_artifacts import (
    HybridAccessReceipt,
    HybridArtifactError,
    HybridCoverageReport,
    HybridReadReceipt,
    HybridReadReceiptItem,
)
from dithyramba.recall.hybrid_engine import (
    HybridExecutionArtifacts,
    HybridExecutionError,
)
from dithyramba.recall.hybrid_models import (
    CorpusLayer,
    HybridContractError,
    HybridEvidencePacket,
    HybridPacketItem,
    HybridQueryRequest,
    HybridRankTraceItem,
    HybridRetrievalBudget,
    HybridRetrievalReceipt,
)
from dithyramba.recall.vector import CorpusVectorGeneration

from .hybrid import SQLiteHybridStore
from .repository import (
    LibraryRepository,
    _timestamp,
    _validated_sha256,
    _validated_timestamp,
)

_RUN_ID_PATTERN = re.compile(r"^run_[a-z0-9]+(?:_[a-z0-9]+)*$")
_MAX_VERSION_LENGTH = 128
_CHANNELS = ("fts", "dense", "fused", "reranked")


@dataclass(frozen=True, slots=True)
class HybridSuccessfulRunRecord:
    """Explicit terminal identity and operational metadata for one P7 recall."""

    processing_run_id: str
    request_id: str
    vector_generation_hash: str
    kind: Literal["recall", "replay"]
    code_version: str
    profile_version: str
    started_at: str
    finished_at: str
    output_hash: str

    def __post_init__(self) -> None:
        if (
            type(self.processing_run_id) is not str
            or _RUN_ID_PATTERN.fullmatch(self.processing_run_id) is None
        ):
            raise PersistenceIntegrityError("hybrid run ID must use a canonical run_ prefix")
        if type(self.request_id) is not str or not self.request_id.startswith("hybrid_query_"):
            raise PersistenceIntegrityError("hybrid run requires a HybridQueryRequest ID")
        if self.kind not in ("recall", "replay"):
            raise PersistenceIntegrityError(
                "only successful hybrid recall/replay runs are supported"
            )
        _bounded_version(self.code_version, "code version")
        _bounded_version(self.profile_version, "profile version")
        _validated_sha256(self.vector_generation_hash, "vector generation hash")
        _validated_sha256(self.output_hash, "hybrid run output hash")
        _validated_timestamp(self.started_at)
        _validated_timestamp(self.finished_at)
        if _parse_timestamp(self.finished_at) < _parse_timestamp(self.started_at):
            raise PersistenceIntegrityError("hybrid run finished before it started")


class SQLiteHybridRecallBackend:
    """Persist exact hybrid requests and complete successful run closures."""

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteHybridRecallBackend requires a LibraryRepository")
        self._repository = repository
        self._hybrid_store = SQLiteHybridStore(repository)

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def persist_query_request(self, request: HybridQueryRequest) -> HybridQueryRequest:
        """Insert one normalized content-addressed request or verify it exactly."""

        if type(request) is not HybridQueryRequest:
            raise TypeError("persist_query_request requires a HybridQueryRequest")
        self._validate_query_dependencies(request)
        created_at = _timestamp(self._repository._clock)
        exclusion_rows = _exclusion_rows(request.exclusions)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO hybrid_query_requests(
                        hybrid_query_request_id, library_id, corpus_snapshot_id,
                        access_policy_id, question, purpose, model_profile_hash,
                        runtime_profile_hash, layer_profile_hash, fusion_profile_hash,
                        reranker_profile_hash, exclusion_hash, collection_count,
                        exclusion_count, max_fts_candidates, max_dense_candidates,
                        max_fused_candidates, max_source_fragments, result_format,
                        request_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.request_id,
                        request.library_id,
                        request.corpus_snapshot_id,
                        request.access_policy_id,
                        request.question,
                        request.purpose,
                        request.model_profile_hash,
                        request.runtime_profile_hash,
                        request.layer_profile_hash,
                        request.fusion_profile_hash,
                        request.reranker_profile_hash,
                        self._scope_for(request).exclusion_hash,
                        len(request.collection_ids),
                        len(exclusion_rows),
                        request.retrieval.max_fts_candidates,
                        request.retrieval.max_dense_candidates,
                        request.retrieval.max_fused_candidates,
                        request.retrieval.max_source_fragments,
                        request.result_format,
                        request.request_hash,
                        created_at,
                    ),
                )
                if cursor.rowcount == 1:
                    connection.executemany(
                        """
                        INSERT INTO hybrid_query_collections(
                            hybrid_query_request_id, collection_id
                        ) VALUES (?, ?)
                        """,
                        ((request.request_id, item) for item in request.collection_ids),
                    )
                    connection.executemany(
                        """
                        INSERT INTO hybrid_query_exclusions(
                            hybrid_query_request_id, target_type, target_id
                        ) VALUES (?, ?, ?)
                        """,
                        (
                            (request.request_id, target_type, target_id)
                            for target_type, target_id in exclusion_rows
                        ),
                    )
                persisted = self._load_query_request(connection, request.request_id)
                if persisted != request:
                    raise PersistenceIntegrityError(
                        "persisted HybridQueryRequest conflicts with its content address"
                    )
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("HybridQueryRequest insert conflicted") from error
        return request

    def load_query_request(self, request_id: str) -> HybridQueryRequest:
        """Load one normalized request and independently recompute its scope/hash."""

        with self._repository._store.transaction() as connection:
            request = self._load_query_request(connection, request_id)
        self._validate_query_dependencies(request)
        return request

    def persist_successful_run(
        self,
        run: HybridSuccessfulRunRecord,
        artifacts: HybridExecutionArtifacts,
    ) -> tuple[HybridSuccessfulRunRecord, HybridExecutionArtifacts]:
        """Atomically append one complete terminal successful recall."""

        if type(run) is not HybridSuccessfulRunRecord:
            raise TypeError("persist_successful_run requires a HybridSuccessfulRunRecord")
        if type(artifacts) is not HybridExecutionArtifacts:
            raise TypeError("persist_successful_run requires HybridExecutionArtifacts")
        request = self.load_query_request(run.request_id)
        generation, _vectors = self._hybrid_store.load_corpus_vector_generation(
            run.vector_generation_hash
        )
        self._validate_execution_closure(run, request, generation, artifacts)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                persisted_generation = self._load_generation(connection, run.vector_generation_hash)
                if persisted_generation != generation:
                    raise PersistenceIntegrityError(
                        "CorpusVectorGeneration changed before hybrid run persistence"
                    )
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO hybrid_processing_runs(
                        hybrid_processing_run_id, hybrid_query_request_id,
                        corpus_vector_generation_id, kind, status, code_version,
                        profile_version, started_at, finished_at, error_code, output_hash
                    ) VALUES (?, ?, ?, ?, 'succeeded', ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        run.processing_run_id,
                        run.request_id,
                        generation.generation_id,
                        run.kind,
                        run.code_version,
                        run.profile_version,
                        run.started_at,
                        run.finished_at,
                        run.output_hash,
                    ),
                )
                if cursor.rowcount == 1:
                    self._insert_artifacts(connection, run, artifacts)
                persisted = self._load_successful_run(connection, run.processing_run_id)
                if persisted != (run, artifacts):
                    raise PersistenceIntegrityError(
                        "persisted hybrid run conflicts with its immutable closure"
                    )
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("successful hybrid run insert conflicted") from error
        return run, artifacts

    def load_successful_run(
        self,
        processing_run_id: str,
    ) -> tuple[HybridSuccessfulRunRecord, HybridExecutionArtifacts]:
        """Load and independently re-hash one complete successful run closure."""

        run_key = _text(processing_run_id, "hybrid processing run ID")
        with self._repository._store.transaction() as connection:
            scope_row = connection.execute(
                """
                SELECT run.hybrid_query_request_id, generation.generation_hash
                FROM hybrid_processing_runs AS run
                JOIN corpus_vector_generations AS generation
                  ON generation.corpus_vector_generation_id = run.corpus_vector_generation_id
                WHERE run.hybrid_processing_run_id = ?
                  AND generation.library_id = ?
                """,
                (run_key, self.library_id),
            ).fetchone()
            if scope_row is None:
                raise PersistenceIntegrityError(
                    "successful hybrid run does not exist in this Library"
                )
            request_id = _text(scope_row[0], "hybrid run request ID")
            generation_hash = _hash(scope_row[1], "hybrid run generation hash")

        # Recompile the exact policy/snapshot scope before the vector capability
        # gate is opened by SQLiteHybridStore.
        request = self.load_query_request(request_id)
        generation, _vectors = self._hybrid_store.load_corpus_vector_generation(generation_hash)
        with self._repository._store.transaction() as connection:
            run, artifacts = self._load_successful_run(connection, run_key)
            exact_request = self._load_query_request(connection, run.request_id)
        if exact_request != request:
            raise PersistenceIntegrityError("hybrid run request changed during load")
        if run.vector_generation_hash != generation.generation_hash:
            raise PersistenceIntegrityError("hybrid run generation changed during load")
        self._validate_execution_closure(run, request, generation, artifacts)
        return run, artifacts

    def _load_query_request(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> HybridQueryRequest:
        request_key = _text(request_id, "HybridQueryRequest ID")
        row = connection.execute(
            """
            SELECT hybrid_query_request_id, library_id, corpus_snapshot_id,
                   access_policy_id, question, purpose, model_profile_hash,
                   runtime_profile_hash, layer_profile_hash, fusion_profile_hash,
                   reranker_profile_hash, exclusion_hash, collection_count,
                   exclusion_count, max_fts_candidates, max_dense_candidates,
                   max_fused_candidates, max_source_fragments, result_format,
                   request_hash, created_at
            FROM hybrid_query_requests
            WHERE hybrid_query_request_id = ? AND library_id = ?
            """,
            (request_key, self.library_id),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError("HybridQueryRequest does not exist in this Library")
        collection_count = _positive_integer(row[12], "request Collection count")
        exclusion_count = _nonnegative_integer(row[13], "request exclusion count")
        collection_rows = connection.execute(
            """
            SELECT collection_id FROM hybrid_query_collections
            WHERE hybrid_query_request_id = ?
            ORDER BY collection_id COLLATE BINARY
            """,
            (request_key,),
        ).fetchall()
        exclusion_rows = connection.execute(
            """
            SELECT target_type, target_id FROM hybrid_query_exclusions
            WHERE hybrid_query_request_id = ?
            ORDER BY target_type COLLATE BINARY, target_id COLLATE BINARY
            """,
            (request_key,),
        ).fetchall()
        if len(collection_rows) != collection_count or len(exclusion_rows) != exclusion_count:
            raise PersistenceIntegrityError("HybridQueryRequest child count mismatch")
        exclusions = _exclusions_from_rows(exclusion_rows)
        snapshot = self._repository.get_corpus_snapshot(
            _text(row[2], "HybridQueryRequest snapshot ID")
        )
        try:
            request = HybridQueryRequest(
                question=_text(row[4], "HybridQueryRequest question"),
                library_id=_text(row[1], "HybridQueryRequest Library ID"),
                collection_ids=tuple(
                    _text(item[0], "HybridQueryRequest Collection ID") for item in collection_rows
                ),
                corpus_snapshot_id=snapshot.corpus_snapshot_id,
                access_policy_id=_text(row[3], "HybridQueryRequest policy ID"),
                purpose=_text(row[5], "HybridQueryRequest purpose"),
                model_profile_hash=_hash(row[6], "HybridQueryRequest model hash"),
                runtime_profile_hash=_hash(row[7], "HybridQueryRequest runtime hash"),
                layer_profile_hash=_hash(row[8], "HybridQueryRequest layer hash"),
                fusion_profile_hash=_hash(row[9], "HybridQueryRequest fusion hash"),
                reranker_profile_hash=(
                    None if row[10] is None else _hash(row[10], "HybridQueryRequest reranker hash")
                ),
                exclusions=exclusions,
                retrieval=HybridRetrievalBudget(
                    max_fts_candidates=_positive_integer(row[14], "max FTS candidates"),
                    max_dense_candidates=_positive_integer(row[15], "max dense candidates"),
                    max_fused_candidates=_positive_integer(row[16], "max fused candidates"),
                    max_source_fragments=_positive_integer(row[17], "max source fragments"),
                ),
                result_format=cast(
                    Literal["hybrid_evidence_packet"],
                    _text(row[18], "HybridQueryRequest result format"),
                ),
            )
        except (HybridContractError, ValidationError, TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted HybridQueryRequest is invalid") from error
        scope = RequestScope(
            library_id=request.library_id,
            snapshot_hash=snapshot.manifest_hash,
            purpose=request.purpose,
            collection_ids=request.collection_ids,
            exclusions=request.exclusions,
        )
        if (
            request.request_id != _text(row[0], "HybridQueryRequest ID")
            or request.request_id != request_key
            or request.request_hash != _hash(row[19], "HybridQueryRequest hash")
            or scope.exclusion_hash != _hash(row[11], "HybridQueryRequest exclusion hash")
        ):
            raise PersistenceIntegrityError("persisted HybridQueryRequest identity mismatch")
        _validated_timestamp(_text(row[20], "HybridQueryRequest timestamp"))
        return request

    def _scope_for(self, request: HybridQueryRequest) -> RequestScope:
        snapshot = self._repository.get_corpus_snapshot(request.corpus_snapshot_id)
        return RequestScope(
            library_id=request.library_id,
            snapshot_hash=snapshot.manifest_hash,
            purpose=request.purpose,
            collection_ids=request.collection_ids,
            exclusions=request.exclusions,
        )

    def _validate_query_dependencies(self, request: HybridQueryRequest) -> None:
        if request.library_id != self.library_id:
            raise PersistenceIntegrityError("HybridQueryRequest belongs to a different Library")
        scope = self._scope_for(request)
        snapshot = self._repository.get_corpus_snapshot(request.corpus_snapshot_id)
        if not set(request.collection_ids).issubset(snapshot.collection_ids):
            raise PersistenceIntegrityError(
                "HybridQueryRequest Collections are outside its snapshot"
            )
        policy = self._repository.get_access_policy(request.access_policy_id).snapshot
        if policy.library_id != self.library_id:
            raise PersistenceIntegrityError("HybridQueryRequest policy crosses the Library")
        with self._repository._store.transaction() as _connection:
            compiled = self._repository._compile_db_access(
                access_policy_id=request.access_policy_id,
                scope=scope,
            )
        self._hybrid_store.load_embedding_model_profile(request.model_profile_hash)
        self._hybrid_store.load_model_runtime_profile(request.runtime_profile_hash)
        layer = self._hybrid_store.load_corpus_layer_profile(request.layer_profile_hash)
        self._hybrid_store.load_fusion_profile(request.fusion_profile_hash)
        if (
            layer.library_id != self.library_id
            or layer.corpus_snapshot_id != request.corpus_snapshot_id
            or layer.exclusion_hash != scope.exclusion_hash
            or layer.permitted_set_hash != compiled.manifest.permitted_set_hash
        ):
            raise PersistenceIntegrityError("HybridQueryRequest layer/scope closure mismatch")
        layer.require_exact_sources(
            tuple(
                sorted(
                    {item.source_id for item in compiled.manifest.items},
                    key=lambda value: value.encode("ascii"),
                )
            )
        )
        if request.reranker_profile_hash is not None:
            self._hybrid_store.load_reranker_profile(request.reranker_profile_hash)

    def _load_generation(
        self,
        connection: sqlite3.Connection,
        generation_hash: str,
    ) -> CorpusVectorGeneration:
        generation, _vectors = self._hybrid_store._load_corpus_vector_generation(
            connection,
            generation_hash,
        )
        return generation

    def _validate_execution_closure(
        self,
        run: HybridSuccessfulRunRecord,
        request: HybridQueryRequest,
        generation: CorpusVectorGeneration,
        artifacts: HybridExecutionArtifacts,
    ) -> None:
        if run.request_id != request.request_id:
            raise PersistenceIntegrityError("hybrid run/request identity mismatch")
        if run.vector_generation_hash != generation.generation_hash:
            raise PersistenceIntegrityError("hybrid run/vector generation mismatch")
        if run.output_hash != artifacts.packet.packet_hash:
            raise PersistenceIntegrityError("hybrid run output is not its exact packet")

        read = artifacts.read_receipt
        access = artifacts.access_receipt
        coverage = artifacts.coverage_report
        retrieval = artifacts.retrieval_receipt
        packet = artifacts.packet
        request_ids = {
            read.request_id,
            access.request_id,
            coverage.request_id,
            retrieval.request_id,
            packet.request_id,
        }
        request_hashes = {
            read.request_hash,
            access.request_hash,
            coverage.request_hash,
            retrieval.request_hash,
            packet.request_hash,
        }
        if request_ids != {request.request_id} or request_hashes != {request.request_hash}:
            raise PersistenceIntegrityError("hybrid artifacts do not close over one request")

        scope = self._scope_for(request)
        snapshot = self._repository.get_corpus_snapshot(request.corpus_snapshot_id)
        policy = self._repository.get_access_policy(request.access_policy_id).snapshot
        if (
            generation.library_id != self.library_id
            or generation.corpus_snapshot_id != request.corpus_snapshot_id
            or generation.snapshot_hash != snapshot.manifest_hash
            or generation.access_policy_id != request.access_policy_id
            or generation.policy_hash != policy.policy_hash
            or generation.exclusion_hash != scope.exclusion_hash
            or generation.model_profile_hash != request.model_profile_hash
            or generation.runtime_profile_hash != request.runtime_profile_hash
        ):
            raise PersistenceIntegrityError("hybrid generation differs from its request scope")
        if (
            access.library_id != self.library_id
            or access.access_policy_id != request.access_policy_id
            or access.policy_hash != generation.policy_hash
            or access.corpus_snapshot_id != request.corpus_snapshot_id
            or access.snapshot_hash != generation.snapshot_hash
            or access.exclusion_hash != generation.exclusion_hash
            or access.permitted_set_hash != generation.permitted_set_hash
            or read.retrieval_corpus_hash != access.retrieval_corpus_hash
            or coverage.policy_omission_present != access.policy_omission_present
        ):
            raise PersistenceIntegrityError("hybrid access/read/coverage closure mismatch")

        generation_manifest = {
            item.source_fragment_id: item.text_sha256 for item in generation.items
        }
        read_manifest = {item.source_fragment_id: item.text_sha256 for item in read.items}
        if read_manifest != generation_manifest:
            raise PersistenceIntegrityError(
                "hybrid read receipt is not the exact vector generation manifest"
            )
        if (
            coverage.processed_count != len(generation.items)
            or coverage.processed_count != len(read.items)
            or coverage.skipped_count != 0
            or coverage.failed_count != 0
        ):
            raise PersistenceIntegrityError(
                "hybrid coverage does not cover every permitted Fragment"
            )

        if (
            retrieval.exclusion_hash != generation.exclusion_hash
            or retrieval.permitted_set_hash != generation.permitted_set_hash
            or retrieval.model_profile_hash != request.model_profile_hash
            or retrieval.runtime_profile_hash != request.runtime_profile_hash
            or retrieval.layer_profile_hash != request.layer_profile_hash
            or retrieval.fusion_profile_hash != request.fusion_profile_hash
            or retrieval.vector_generation_hash != generation.generation_hash
            or retrieval.reranker_profile_hash != request.reranker_profile_hash
        ):
            raise PersistenceIntegrityError("hybrid retrieval receipt profile closure mismatch")

        permitted_ids = set(generation_manifest)
        for channel, limit in (
            (retrieval.fts, request.retrieval.max_fts_candidates),
            (retrieval.dense, request.retrieval.max_dense_candidates),
            (retrieval.fused, request.retrieval.max_fused_candidates),
        ):
            if len(channel) > limit or any(
                item.source_fragment_id not in permitted_ids for item in channel
            ):
                raise PersistenceIntegrityError(
                    "hybrid retrieval trace exceeds its budget or permitted set"
                )
        if len(retrieval.reranked) > request.retrieval.max_fused_candidates:
            raise PersistenceIntegrityError("hybrid reranker trace exceeds the fused budget")

        selected_trace = retrieval.reranked if request.reranker_profile_hash else retrieval.fused
        packet_ids = tuple(item.source_fragment_id for item in packet.items)
        layer_profile = self._hybrid_store.load_corpus_layer_profile(request.layer_profile_hash)
        fusion_profile = self._hybrid_store.load_fusion_profile(request.fusion_profile_hash)
        layer_by_source = {item.source_id: item.layer for item in layer_profile.items}
        try:
            expected_packet_ids = select_layer_aware_fragment_ids(
                tuple(
                    (
                        item.source_fragment_id,
                        item.source_id,
                        layer_by_source[item.source_id],
                    )
                    for item in selected_trace
                ),
                limit=request.retrieval.max_source_fragments,
                profile=fusion_profile,
            )
        except (FusionContractError, KeyError) as error:
            raise PersistenceIntegrityError(
                "hybrid selected trace is outside its layer profile"
            ) from error
        if packet_ids != expected_packet_ids:
            raise PersistenceIntegrityError("hybrid packet is outside its selected trace")
        source_by_fragment: dict[str, str] = {}
        for item in (*retrieval.fts, *retrieval.dense, *retrieval.fused, *retrieval.reranked):
            existing = source_by_fragment.setdefault(item.source_fragment_id, item.source_id)
            if existing != item.source_id:
                raise PersistenceIntegrityError(
                    "hybrid trace assigns one Fragment to multiple Sources"
                )
        if any(
            source_by_fragment.get(item.source_fragment_id) != item.source_id
            for item in packet.items
        ):
            raise PersistenceIntegrityError("hybrid packet Source lineage differs from retrieval")
        if (
            packet.corpus_snapshot_id != request.corpus_snapshot_id
            or packet.access_policy_id != request.access_policy_id
        ):
            raise PersistenceIntegrityError("hybrid packet scope differs from its request")

    def _insert_artifacts(
        self,
        connection: sqlite3.Connection,
        run: HybridSuccessfulRunRecord,
        artifacts: HybridExecutionArtifacts,
    ) -> None:
        coverage = artifacts.coverage_report
        read = artifacts.read_receipt
        access = artifacts.access_receipt
        retrieval = artifacts.retrieval_receipt
        packet = artifacts.packet
        connection.execute(
            """
            INSERT OR IGNORE INTO hybrid_coverage_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                coverage.report_id,
                run.processing_run_id,
                run.request_id,
                coverage.processed_count,
                coverage.skipped_count,
                coverage.failed_count,
                int(coverage.policy_omission_present),
                coverage.report_hash,
            ),
        )
        read_cursor = connection.execute(
            """
            INSERT OR IGNORE INTO hybrid_read_receipts VALUES (?, ?, ?, ?, ?)
            """,
            (
                read.receipt_id,
                run.processing_run_id,
                run.request_id,
                len(read.items),
                read.receipt_hash,
            ),
        )
        if read_cursor.rowcount == 1:
            connection.executemany(
                """
                INSERT INTO hybrid_read_receipt_items VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        read.receipt_id,
                        item.source_fragment_id,
                        item.read_order,
                        item.text_sha256,
                    )
                    for item in read.items
                ),
            )
        connection.execute(
            """
            INSERT OR IGNORE INTO hybrid_access_receipts
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                access.receipt_id,
                run.processing_run_id,
                run.request_id,
                access.access_policy_id,
                access.policy_hash,
                access.snapshot_hash,
                access.exclusion_hash,
                access.permitted_set_hash,
                access.retrieval_corpus_hash,
                int(access.policy_omission_present),
                access.receipt_hash,
            ),
        )
        retrieval_cursor = connection.execute(
            """
            INSERT OR IGNORE INTO hybrid_retrieval_receipts(
                hybrid_retrieval_receipt_id, hybrid_processing_run_id,
                hybrid_query_request_id, request_hash, exclusion_hash,
                permitted_set_hash, model_profile_hash, runtime_profile_hash,
                layer_profile_hash, fusion_profile_hash, vector_generation_hash,
                fts_profile_version, fts_result_hash, query_vector_hash,
                reranker_profile_hash, receipt_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                retrieval.receipt_id,
                run.processing_run_id,
                run.request_id,
                retrieval.request_hash,
                retrieval.exclusion_hash,
                retrieval.permitted_set_hash,
                retrieval.model_profile_hash,
                retrieval.runtime_profile_hash,
                retrieval.layer_profile_hash,
                retrieval.fusion_profile_hash,
                retrieval.vector_generation_hash,
                retrieval.fts_profile_version,
                retrieval.fts_result_hash,
                retrieval.query_vector_hash,
                retrieval.reranker_profile_hash,
                retrieval.receipt_hash,
            ),
        )
        if retrieval_cursor.rowcount == 1:
            for channel in _CHANNELS:
                trace = cast(tuple[HybridRankTraceItem, ...], getattr(retrieval, channel))
                connection.executemany(
                    """
                    INSERT INTO hybrid_retrieval_trace_items VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            retrieval.receipt_id,
                            channel,
                            item.source_fragment_id,
                            item.source_id,
                            item.rank,
                            item.score_repr,
                        )
                        for item in trace
                    ),
                )
        packet_cursor = connection.execute(
            """
            INSERT OR IGNORE INTO hybrid_evidence_packets
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                packet.packet_id,
                run.request_id,
                packet.corpus_snapshot_id,
                packet.access_policy_id,
                packet.read_receipt_hash,
                packet.access_receipt_hash,
                packet.coverage_report_hash,
                packet.retrieval_receipt_hash,
                packet.result_status,
                len(packet.items),
                packet.packet_hash,
                run.finished_at,
            ),
        )
        if packet_cursor.rowcount == 1:
            connection.executemany(
                """
                INSERT INTO hybrid_packet_items VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        packet.packet_id,
                        item.rank,
                        item.source_fragment_id,
                        item.source_id,
                        item.source_family_id,
                        item.layer.value,
                    )
                    for item in packet.items
                ),
            )
        connection.execute(
            """
            INSERT INTO hybrid_run_artifacts VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run.processing_run_id,
                coverage.report_id,
                read.receipt_id,
                access.receipt_id,
                retrieval.receipt_id,
                packet.packet_id,
            ),
        )

    def _load_successful_run(
        self,
        connection: sqlite3.Connection,
        processing_run_id: str,
    ) -> tuple[HybridSuccessfulRunRecord, HybridExecutionArtifacts]:
        run_key = _text(processing_run_id, "hybrid processing run ID")
        row = connection.execute(
            """
            SELECT run.hybrid_processing_run_id, run.hybrid_query_request_id,
                   generation.generation_hash, run.kind, run.status,
                   run.code_version, run.profile_version, run.started_at,
                   run.finished_at, run.error_code, run.output_hash,
                   artifacts.hybrid_coverage_report_id,
                   artifacts.hybrid_read_receipt_id,
                   artifacts.hybrid_access_receipt_id,
                   artifacts.hybrid_retrieval_receipt_id,
                   artifacts.hybrid_evidence_packet_id
            FROM hybrid_processing_runs AS run
            JOIN corpus_vector_generations AS generation
              ON generation.corpus_vector_generation_id = run.corpus_vector_generation_id
            JOIN hybrid_run_artifacts AS artifacts
              ON artifacts.hybrid_processing_run_id = run.hybrid_processing_run_id
            WHERE run.hybrid_processing_run_id = ? AND generation.library_id = ?
            """,
            (run_key, self.library_id),
        ).fetchone()
        if row is None:
            raise PersistenceIntegrityError(
                "complete successful hybrid run does not exist in this Library"
            )
        if row[4] != "succeeded" or row[9] is not None:
            raise PersistenceIntegrityError("hybrid run is not a terminal success")
        try:
            run = HybridSuccessfulRunRecord(
                processing_run_id=_text(row[0], "hybrid processing run ID"),
                request_id=_text(row[1], "hybrid run request ID"),
                vector_generation_hash=_hash(row[2], "hybrid run generation hash"),
                kind=cast(
                    Literal["recall", "replay"],
                    _text(row[3], "hybrid run kind"),
                ),
                code_version=_text(row[5], "hybrid run code version"),
                profile_version=_text(row[6], "hybrid run profile version"),
                started_at=_text(row[7], "hybrid run start timestamp"),
                finished_at=_text(row[8], "hybrid run finish timestamp"),
                output_hash=_hash(row[10], "hybrid run output hash"),
            )
        except (TypeError, ValueError) as error:
            raise PersistenceIntegrityError("persisted hybrid run is invalid") from error
        if run.processing_run_id != run_key:
            raise PersistenceIntegrityError("persisted hybrid run identity mismatch")
        request = self._load_query_request(connection, run.request_id)

        coverage_id = _text(row[11], "hybrid coverage report ID")
        read_id = _text(row[12], "hybrid read receipt ID")
        access_id = _text(row[13], "hybrid access receipt ID")
        retrieval_id = _text(row[14], "hybrid retrieval receipt ID")
        packet_id = _text(row[15], "hybrid evidence packet ID")
        coverage_row = connection.execute(
            """
            SELECT hybrid_query_request_id, processed_count, skipped_count,
                   failed_count, policy_omission_present, report_hash
            FROM hybrid_coverage_reports
            WHERE hybrid_coverage_report_id = ?
            """,
            (coverage_id,),
        ).fetchone()
        access_row = connection.execute(
            """
            SELECT hybrid_query_request_id, access_policy_id, policy_hash,
                   snapshot_hash, exclusion_hash, permitted_set_hash,
                   retrieval_corpus_hash, policy_omission_present, receipt_hash
            FROM hybrid_access_receipts
            WHERE hybrid_access_receipt_id = ?
            """,
            (access_id,),
        ).fetchone()
        read_row = connection.execute(
            """
            SELECT hybrid_query_request_id, item_count, receipt_hash
            FROM hybrid_read_receipts
            WHERE hybrid_read_receipt_id = ?
            """,
            (read_id,),
        ).fetchone()
        retrieval_row = connection.execute(
            """
            SELECT hybrid_query_request_id, request_hash, exclusion_hash,
                   permitted_set_hash, model_profile_hash, runtime_profile_hash,
                   layer_profile_hash, fusion_profile_hash, vector_generation_hash,
                   fts_profile_version, fts_result_hash, query_vector_hash,
                   reranker_profile_hash, receipt_hash
            FROM hybrid_retrieval_receipts
            WHERE hybrid_retrieval_receipt_id = ?
            """,
            (retrieval_id,),
        ).fetchone()
        packet_row = connection.execute(
            """
            SELECT hybrid_query_request_id, corpus_snapshot_id, access_policy_id,
                   read_receipt_hash, access_receipt_hash, coverage_report_hash,
                   retrieval_receipt_hash, result_status, item_count, packet_hash,
                   created_at
            FROM hybrid_evidence_packets WHERE hybrid_evidence_packet_id = ?
            """,
            (packet_id,),
        ).fetchone()
        if any(
            item is None for item in (coverage_row, access_row, read_row, retrieval_row, packet_row)
        ):
            raise PersistenceIntegrityError("hybrid run artifact closure is incomplete")
        coverage_row = cast(sqlite3.Row, coverage_row)
        access_row = cast(sqlite3.Row, access_row)
        read_row = cast(sqlite3.Row, read_row)
        retrieval_row = cast(sqlite3.Row, retrieval_row)
        packet_row = cast(sqlite3.Row, packet_row)

        read_count = _nonnegative_integer(read_row[1], "hybrid read item count")
        read_rows = connection.execute(
            """
            SELECT item.source_fragment_id, fragment.source_version_id,
                   item.read_order, item.text_sha256
            FROM hybrid_read_receipt_items AS item
            JOIN source_fragments AS fragment
              ON fragment.source_fragment_id = item.source_fragment_id
            JOIN source_versions AS version
              ON version.source_version_id = fragment.source_version_id
            JOIN sources AS source ON source.source_id = version.source_id
            WHERE item.hybrid_read_receipt_id = ? AND source.library_id = ?
            ORDER BY item.read_order
            """,
            (read_id, self.library_id),
        ).fetchall()
        if len(read_rows) != read_count:
            raise PersistenceIntegrityError("hybrid read receipt item count mismatch")
        trace_rows = connection.execute(
            """
            SELECT channel, source_fragment_id, source_id, rank, score_repr
            FROM hybrid_retrieval_trace_items
            WHERE hybrid_retrieval_receipt_id = ?
            ORDER BY CASE channel
                       WHEN 'fts' THEN 1 WHEN 'dense' THEN 2
                       WHEN 'fused' THEN 3 WHEN 'reranked' THEN 4
                     END, rank
            """,
            (retrieval_id,),
        ).fetchall()
        trace_values: dict[str, list[HybridRankTraceItem]] = {channel: [] for channel in _CHANNELS}
        for trace_row in trace_rows:
            channel = _text(trace_row[0], "hybrid trace channel")
            if channel not in trace_values:
                raise PersistenceIntegrityError("persisted hybrid trace channel is invalid")
            trace_values[channel].append(
                HybridRankTraceItem(
                    source_fragment_id=_text(trace_row[1], "hybrid trace Fragment ID"),
                    source_id=_text(trace_row[2], "hybrid trace Source ID"),
                    rank=_positive_integer(trace_row[3], "hybrid trace rank"),
                    score_repr=_text(trace_row[4], "hybrid trace score"),
                )
            )
        packet_count = _nonnegative_integer(packet_row[8], "hybrid packet item count")
        packet_rows = connection.execute(
            """
            SELECT rank, source_fragment_id, source_id, source_family_id, layer
            FROM hybrid_packet_items
            WHERE hybrid_evidence_packet_id = ? ORDER BY rank
            """,
            (packet_id,),
        ).fetchall()
        if len(packet_rows) != packet_count:
            raise PersistenceIntegrityError("hybrid packet item count mismatch")

        try:
            coverage = HybridCoverageReport(
                request_id=_text(coverage_row[0], "coverage request ID"),
                request_hash=request.request_hash,
                processed_count=_nonnegative_integer(coverage_row[1], "processed count"),
                skipped_count=_nonnegative_integer(coverage_row[2], "skipped count"),
                failed_count=_nonnegative_integer(coverage_row[3], "failed count"),
                policy_omission_present=_boolean(coverage_row[4], "coverage omission"),
            )
            access = HybridAccessReceipt(
                library_id=self.library_id,
                request_id=_text(access_row[0], "access request ID"),
                request_hash=request.request_hash,
                access_policy_id=_text(access_row[1], "access policy ID"),
                policy_hash=_hash(access_row[2], "access policy hash"),
                corpus_snapshot_id=request.corpus_snapshot_id,
                snapshot_hash=_hash(access_row[3], "access snapshot hash"),
                exclusion_hash=_hash(access_row[4], "access exclusion hash"),
                permitted_set_hash=_hash(access_row[5], "access permitted-set hash"),
                retrieval_corpus_hash=_hash(access_row[6], "retrieval corpus hash"),
                policy_omission_present=_boolean(access_row[7], "access omission"),
            )
            read = HybridReadReceipt(
                request_id=_text(read_row[0], "read request ID"),
                request_hash=request.request_hash,
                retrieval_corpus_hash=access.retrieval_corpus_hash,
                items=tuple(
                    HybridReadReceiptItem(
                        source_fragment_id=_text(item[0], "read Fragment ID"),
                        source_version_id=_text(item[1], "read SourceVersion ID"),
                        read_order=_nonnegative_integer(item[2], "read order"),
                        text_sha256=_hash(item[3], "read text hash"),
                    )
                    for item in read_rows
                ),
            )
            retrieval = HybridRetrievalReceipt(
                request_id=_text(retrieval_row[0], "retrieval request ID"),
                request_hash=_hash(retrieval_row[1], "retrieval request hash"),
                exclusion_hash=_hash(retrieval_row[2], "retrieval exclusion hash"),
                permitted_set_hash=_hash(retrieval_row[3], "retrieval permitted-set hash"),
                model_profile_hash=_hash(retrieval_row[4], "retrieval model hash"),
                runtime_profile_hash=_hash(retrieval_row[5], "retrieval runtime hash"),
                layer_profile_hash=_hash(retrieval_row[6], "retrieval layer hash"),
                fusion_profile_hash=_hash(retrieval_row[7], "retrieval fusion hash"),
                vector_generation_hash=_hash(retrieval_row[8], "retrieval generation hash"),
                fts_profile_version=_text(retrieval_row[9], "FTS profile version"),
                fts_result_hash=_hash(retrieval_row[10], "FTS result hash"),
                query_vector_hash=_hash(retrieval_row[11], "query vector hash"),
                reranker_profile_hash=(
                    None
                    if retrieval_row[12] is None
                    else _hash(retrieval_row[12], "retrieval reranker hash")
                ),
                fts=tuple(trace_values["fts"]),
                dense=tuple(trace_values["dense"]),
                fused=tuple(trace_values["fused"]),
                reranked=tuple(trace_values["reranked"]),
            )
            packet = HybridEvidencePacket(
                request_id=_text(packet_row[0], "packet request ID"),
                request_hash=request.request_hash,
                corpus_snapshot_id=_text(packet_row[1], "packet snapshot ID"),
                access_policy_id=_text(packet_row[2], "packet policy ID"),
                read_receipt_hash=_hash(packet_row[3], "packet read hash"),
                access_receipt_hash=_hash(packet_row[4], "packet access hash"),
                coverage_report_hash=_hash(packet_row[5], "packet coverage hash"),
                retrieval_receipt_hash=_hash(packet_row[6], "packet retrieval hash"),
                result_status=cast(
                    Literal["evidence_found", "no_evidence"],
                    _text(packet_row[7], "packet result status"),
                ),
                items=tuple(
                    HybridPacketItem(
                        rank=_positive_integer(item[0], "packet rank"),
                        source_fragment_id=_text(item[1], "packet Fragment ID"),
                        source_id=_text(item[2], "packet Source ID"),
                        source_family_id=_text(item[3], "packet SourceFamily ID"),
                        layer=CorpusLayer(_text(item[4], "packet layer")),
                    )
                    for item in packet_rows
                ),
            )
            artifacts = HybridExecutionArtifacts(
                read_receipt=read,
                access_receipt=access,
                coverage_report=coverage,
                retrieval_receipt=retrieval,
                packet=packet,
            )
        except (
            HybridArtifactError,
            HybridContractError,
            HybridExecutionError,
            ValidationError,
            TypeError,
            ValueError,
        ) as error:
            raise PersistenceIntegrityError("persisted hybrid artifacts are invalid") from error

        identity_pairs = (
            (coverage.report_id, coverage_id, coverage.report_hash, coverage_row[5]),
            (read.receipt_id, read_id, read.receipt_hash, read_row[2]),
            (access.receipt_id, access_id, access.receipt_hash, access_row[8]),
            (retrieval.receipt_id, retrieval_id, retrieval.receipt_hash, retrieval_row[13]),
            (packet.packet_id, packet_id, packet.packet_hash, packet_row[9]),
        )
        if any(
            computed_id != persisted_id
            or computed_hash != _hash(stored_hash, "hybrid artifact hash")
            for computed_id, persisted_id, computed_hash, stored_hash in identity_pairs
        ):
            raise PersistenceIntegrityError("persisted hybrid artifact identity mismatch")
        _validated_timestamp(_text(packet_row[10], "hybrid packet timestamp"))
        return run, artifacts


def _bounded_version(value: str, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > _MAX_VERSION_LENGTH
        or "\x00" in value
    ):
        raise PersistenceIntegrityError(f"hybrid {label} must be bounded unpadded text")
    return value


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(_validated_timestamp(value), "%Y-%m-%dT%H:%M:%S.%fZ")


def _exclusion_rows(exclusions: QueryExclusions) -> tuple[tuple[str, str], ...]:
    if type(exclusions) is not QueryExclusions:
        raise PersistenceIntegrityError("hybrid exclusions require QueryExclusions")
    rows = (
        *(("source", value) for value in exclusions.source_ids),
        *(("source_family", value) for value in exclusions.source_family_ids),
        *(("source_fragment", value) for value in exclusions.source_fragment_ids),
    )
    return tuple(sorted(rows, key=lambda item: (item[0].encode(), item[1].encode())))


def _exclusions_from_rows(rows: list[sqlite3.Row]) -> QueryExclusions:
    values: dict[str, list[str]] = {
        "source": [],
        "source_family": [],
        "source_fragment": [],
    }
    for row in rows:
        target_type = _text(row[0], "hybrid exclusion target type")
        if target_type not in values:
            raise PersistenceIntegrityError("persisted hybrid exclusion type is invalid")
        values[target_type].append(_text(row[1], "hybrid exclusion target ID"))
    try:
        return QueryExclusions(
            source_ids=tuple(values["source"]),
            source_family_ids=tuple(values["source_family"]),
            source_fragment_ids=tuple(values["source_fragment"]),
        )
    except (TypeError, ValueError) as error:
        raise PersistenceIntegrityError("persisted hybrid exclusions are invalid") from error


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        raise PersistenceIntegrityError(f"persisted {label} must be text")
    return value


def _hash(value: object, label: str) -> str:
    return _validated_sha256(_text(value, label), label)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise PersistenceIntegrityError(f"persisted {label} must be an integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result < 1:
        raise PersistenceIntegrityError(f"persisted {label} must be positive")
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise PersistenceIntegrityError(f"persisted {label} must be nonnegative")
    return result


def _boolean(value: object, label: str) -> bool:
    result = _integer(value, label)
    if result not in (0, 1):
        raise PersistenceIntegrityError(f"persisted {label} must be a SQLite boolean")
    return bool(result)
