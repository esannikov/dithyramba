"""SQLite adapter for the provider-free VS0 recall service.

The adapter is deliberately separate from :mod:`dithyramba.persistence.repository`:
the repository remains the sole protected text-read boundary, while this module
owns QueryRequest/run lifecycle persistence and the atomic materialization of
content-addressed recall artifacts.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import ValidationError

from dithyramba.access import QueryExclusions, RequestScope
from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)
from dithyramba.provenance import ProcessingRunRecord, ProcessingRunStatus, SourceFamilyRole
from dithyramba.recall.artifacts import (
    AccessReceipt,
    EvidenceFragment,
    EvidencePacket,
    OmissionCategory,
    OmissionDisclosure,
    PacketResultStatus,
    ReadReceipt,
    ReadReceiptItem,
    RecallCoverageReport,
    RecallOmission,
    RetrievalReceipt,
    RetrievalTraceItem,
)
from dithyramba.recall.models import (
    EvidencePacketResultContract,
    QueryRequest,
    RetrievalBudget,
)
from dithyramba.snapshots import CorpusSnapshot

from .errors import (
    EvidencePacketNotFoundError,
    PersistenceConflictError,
    PersistenceIntegrityError,
    ProcessingRunNotFoundError,
    QueryRequestNotFoundError,
    RecallArtifactNotFoundError,
)
from .models import AuthorizedRead, SourceFragmentText
from .repository import (
    LibraryRepository,
    _insert_outbox_event,
    _timestamp,
    _validated_label,
    _validated_sha256,
    _validated_timestamp,
)

RecallRunKind = Literal["recall", "replay"]
_RUN_ID_PATTERN = re.compile(r"^run_[a-z0-9]+(?:_[a-z0-9]+)*$")
_LARGE_PROJECTION_READ_THRESHOLD = 300
_PROJECTION_READ_PERMIT_TABLE = "dithyramba_projection_read_permit"


@dataclass(frozen=True, slots=True)
class _FragmentProjection:
    source_fragment_id: str
    source_version_id: str
    source_id: str
    source_family_id: str
    text: str
    text_sha256: str
    source_address: dict[str, object]


@dataclass(frozen=True, slots=True)
class _CorpusReadSetIdentity:
    """Internal content address for one exact ordered protected-read corpus."""

    corpus_read_set_id: str
    set_hash: str
    payload: dict[str, object]


class SQLiteRecallBackend:
    """Library-scoped durable backend implementing ``RecallBackend``.

    Every multi-row write uses ``BEGIN IMMEDIATE``. Content-addressed rows are
    inserted once and then reused only after an exact relational and canonical
    verification; a random run is linked to those immutable rows through
    ``recall_run_artifacts``.
    """

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteRecallBackend requires a LibraryRepository")
        self._repository = repository

    @property
    def repository(self) -> LibraryRepository:
        """Return the single Library repository owned by this adapter."""

        return self._repository

    def get_corpus_snapshot(self, corpus_snapshot_id: str) -> CorpusSnapshot:
        """Load one fully revalidated immutable CorpusSnapshot."""

        return self._repository.get_corpus_snapshot(corpus_snapshot_id)

    def persist_query_request(self, request: QueryRequest) -> None:
        """Insert or exactly verify one content-addressed QueryRequest."""

        if type(request) is not QueryRequest:
            raise TypeError("persist_query_request requires an exact QueryRequest")
        if request.library_id != self._repository.library_id:
            raise PersistenceIntegrityError("QueryRequest belongs to a different Library")
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                self._validate_request_dependencies(connection, request)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO query_requests(
                        query_request_id, library_id, corpus_snapshot_id,
                        access_policy_id, request_json, request_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.query_request_id,
                        request.library_id,
                        request.corpus_snapshot_id,
                        request.access_policy_id,
                        request.canonical_bytes.decode("utf-8"),
                        request.request_hash,
                        created_at,
                    ),
                )
                identity_rows = connection.execute(
                    """
                    SELECT query_request_id FROM query_requests
                    WHERE query_request_id = ? OR request_hash = ?
                    ORDER BY query_request_id
                    """,
                    (request.query_request_id, request.request_hash),
                ).fetchall()
                if len(identity_rows) != 1 or str(identity_rows[0][0]) != request.query_request_id:
                    raise PersistenceIntegrityError(
                        "QueryRequest ID/hash resolve to conflicting persisted rows"
                    )
                persisted = self._load_query_request(connection, request.query_request_id)
                if (
                    persisted.canonical_bytes != request.canonical_bytes
                    or persisted.request_hash != request.request_hash
                    or persisted.query_request_id != request.query_request_id
                ):
                    raise PersistenceIntegrityError(
                        "persisted QueryRequest conflicts with its content address"
                    )
                if cursor.rowcount == 1:
                    _insert_outbox_event(
                        connection,
                        event_id=self._repository._event_id_factory(),
                        event_type="query_request.persisted",
                        aggregate_type="query_request",
                        aggregate_id=request.query_request_id,
                        payload={
                            "schema": "dithyramba.query_request_persisted/1.0",
                            "library_id": request.library_id,
                            "query_request_id": request.query_request_id,
                            "request_hash": request.request_hash,
                            "corpus_snapshot_id": request.corpus_snapshot_id,
                            "access_policy_id": request.access_policy_id,
                        },
                        occurred_at=created_at,
                    )
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("QueryRequest insert conflicted") from error

    def load_query_request(self, query_request_id: str) -> QueryRequest:
        """Load and independently re-hash one QueryRequest."""

        with self._repository._store.transaction() as connection:
            return self._load_query_request(connection, query_request_id)

    def begin_recall_run(
        self,
        *,
        processing_run_id: str,
        kind: RecallRunKind,
        query_request_id: str,
        code_version: str,
        profile_version: str,
    ) -> ProcessingRunRecord:
        """Start one random, auditable recall or replay run."""

        validated_kind = _validated_recall_kind(kind)
        code = _validated_label(code_version, "code version")
        profile = _validated_label(profile_version, "profile version")
        run_id = _validated_run_id(processing_run_id)
        started_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                request = self._load_query_request(connection, query_request_id)
                connection.execute(
                    """
                    INSERT INTO processing_runs(
                        processing_run_id, kind, query_request_id, status,
                        code_version, profile_version, started_at, finished_at,
                        error_code, output_hash
                    ) VALUES (?, ?, ?, 'running', ?, ?, ?, NULL, NULL, NULL)
                    """,
                    (
                        run_id,
                        validated_kind,
                        request.query_request_id,
                        code,
                        profile,
                        started_at,
                    ),
                )
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="recall_run.started",
                    aggregate_type="processing_run",
                    aggregate_id=run_id,
                    payload={
                        "schema": "dithyramba.recall_run_started/1.0",
                        "processing_run_id": run_id,
                        "kind": validated_kind,
                        "query_request_id": request.query_request_id,
                        "code_version": code,
                        "profile_version": profile,
                    },
                    occurred_at=started_at,
                )
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("recall ProcessingRun insert conflicted") from error
        return self._repository.get_processing_run(run_id)

    def authorize_read(
        self,
        *,
        access_policy_id: str,
        scope: RequestScope,
    ) -> AuthorizedRead:
        """Delegate capability issuance to the protected repository boundary."""

        return self._repository.authorize_read(
            access_policy_id=access_policy_id,
            scope=scope,
        )

    def get_processing_run(self, processing_run_id: str) -> ProcessingRunRecord:
        """Expose the repository-owned lifecycle read for interruption recovery."""

        return self._repository.get_processing_run(processing_run_id)

    def read_permitted_fragments(
        self,
        authorization: AuthorizedRead,
    ) -> tuple[SourceFragmentText, ...]:
        """Delegate policy-revalidated text materialization to the repository."""

        return self._repository.read_permitted_fragments(authorization)

    def complete_recall_run(
        self,
        *,
        processing_run_id: str,
        packet: EvidencePacket,
    ) -> tuple[ProcessingRunRecord, EvidencePacket]:
        """Atomically persist/reuse every artifact, link them, and finish a run."""

        if type(packet) is not EvidencePacket:
            raise TypeError("complete_recall_run requires an exact EvidencePacket")
        finished_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                run_row = self._load_running_recall_run(connection, processing_run_id)
                query_request_id = str(run_row[2])
                request = self._load_query_request(connection, query_request_id)
                snapshot = self._repository._reconstruct_corpus_snapshot(
                    connection,
                    request.corpus_snapshot_id,
                )
                projections = self._validate_receipt_closure(
                    connection,
                    request=request,
                    snapshot=snapshot,
                    coverage=packet.coverage_report,
                    read=packet.read_receipt,
                    access=packet.access_receipt,
                    retrieval=packet.retrieval_receipt,
                    code_version=str(run_row[4]),
                    profile_version=str(run_row[5]),
                )
                self._validate_packet_closure(
                    packet,
                    request=request,
                    snapshot=snapshot,
                    projections=projections,
                )

                self._ensure_coverage(
                    connection,
                    processing_run_id=processing_run_id,
                    query_request_id=query_request_id,
                    coverage=packet.coverage_report,
                )
                self._ensure_read_receipt(
                    connection,
                    processing_run_id=processing_run_id,
                    query_request_id=query_request_id,
                    receipt=packet.read_receipt,
                )
                self._ensure_access_receipt(
                    connection,
                    processing_run_id=processing_run_id,
                    query_request_id=query_request_id,
                    receipt=packet.access_receipt,
                )
                self._ensure_retrieval_receipt(
                    connection,
                    processing_run_id=processing_run_id,
                    query_request_id=query_request_id,
                    receipt=packet.retrieval_receipt,
                )
                self._ensure_evidence_packet(
                    connection,
                    packet=packet,
                    created_at=finished_at,
                )

                connection.execute(
                    """
                    INSERT INTO recall_run_artifacts(
                        processing_run_id, coverage_report_id, read_receipt_id,
                        access_receipt_id, retrieval_receipt_id, evidence_packet_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        processing_run_id,
                        packet.coverage_report.coverage_report_id,
                        packet.read_receipt.read_receipt_id,
                        packet.access_receipt.access_receipt_id,
                        packet.retrieval_receipt.retrieval_receipt_id,
                        packet.evidence_packet_id,
                    ),
                )
                cursor = connection.execute(
                    """
                    UPDATE processing_runs
                    SET status = 'succeeded', finished_at = ?, error_code = NULL,
                        output_hash = ?
                    WHERE processing_run_id = ? AND status = 'running'
                    """,
                    (finished_at, packet.packet_hash, processing_run_id),
                )
                if cursor.rowcount != 1:
                    raise PersistenceConflictError("recall ProcessingRun changed during completion")
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="recall_run.completed",
                    aggregate_type="processing_run",
                    aggregate_id=processing_run_id,
                    payload={
                        "schema": "dithyramba.recall_run_completed/1.0",
                        "processing_run_id": processing_run_id,
                        "kind": str(run_row[1]),
                        "query_request_id": query_request_id,
                        "evidence_packet_id": packet.evidence_packet_id,
                        "packet_hash": packet.packet_hash,
                        "result_status": packet.result_status.value,
                    },
                    occurred_at=finished_at,
                )
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("recall completion conflicted") from error
        return (
            self._repository.get_processing_run(processing_run_id),
            self.load_evidence_packet(packet.evidence_packet_id),
        )

    def complete_recall_batch(
        self,
        *,
        completions: tuple[tuple[str, EvidencePacket], ...],
    ) -> tuple[tuple[ProcessingRunRecord, EvidencePacket], ...]:
        """Commit one exact-scope batch as a single all-or-nothing transaction."""

        if type(completions) is not tuple or not completions:
            raise TypeError("complete_recall_batch requires a non-empty exact tuple")
        run_ids: list[str] = []
        packets: list[EvidencePacket] = []
        for item in completions:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("batch completion items must be (run ID, EvidencePacket) tuples")
            run_id, packet = item
            run_ids.append(_validated_run_id(run_id))
            if type(packet) is not EvidencePacket:
                raise TypeError("batch completion requires exact EvidencePackets")
            packets.append(packet)
        if len(set(run_ids)) != len(run_ids):
            raise PersistenceIntegrityError("batch completion ProcessingRun IDs must be unique")

        finished_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                prepared: list[
                    tuple[sqlite3.Row, QueryRequest, CorpusSnapshot, EvidencePacket]
                ] = []
                first_request: QueryRequest | None = None
                first_snapshot: CorpusSnapshot | None = None
                first_packet: EvidencePacket | None = None
                shared_projections: dict[str, _FragmentProjection] | None = None

                for run_id, packet in zip(run_ids, packets, strict=True):
                    run_row = self._load_running_recall_run(connection, run_id)
                    request = self._load_query_request(connection, str(run_row[2]))
                    snapshot = self._repository._reconstruct_corpus_snapshot(
                        connection,
                        request.corpus_snapshot_id,
                    )
                    if first_request is None:
                        shared_projections = self._validate_receipt_closure(
                            connection,
                            request=request,
                            snapshot=snapshot,
                            coverage=packet.coverage_report,
                            read=packet.read_receipt,
                            access=packet.access_receipt,
                            retrieval=packet.retrieval_receipt,
                            code_version=str(run_row[4]),
                            profile_version=str(run_row[5]),
                        )
                        first_request = request
                        first_snapshot = snapshot
                        first_packet = packet
                    else:
                        if (
                            first_snapshot is None
                            or first_packet is None
                            or shared_projections is None
                        ):
                            raise AssertionError("batch shared validation state is incomplete")
                        self._validate_batch_receipt_closure(
                            request=request,
                            snapshot=snapshot,
                            packet=packet,
                            code_version=str(run_row[4]),
                            profile_version=str(run_row[5]),
                            first_request=first_request,
                            first_snapshot=first_snapshot,
                            first_packet=first_packet,
                            projections=shared_projections,
                        )
                    if shared_projections is None:
                        raise AssertionError("batch projections were not prepared")
                    self._validate_packet_closure(
                        packet,
                        request=request,
                        snapshot=snapshot,
                        projections=shared_projections,
                    )
                    prepared.append((run_row, request, snapshot, packet))

                if first_packet is None:
                    raise AssertionError("batch packet state is empty")
                read_set = self._ensure_corpus_read_set(
                    connection,
                    first_packet.read_receipt,
                )
                for run_id, (run_row, request, _snapshot, packet) in zip(
                    run_ids,
                    prepared,
                    strict=True,
                ):
                    self._ensure_coverage(
                        connection,
                        processing_run_id=run_id,
                        query_request_id=request.query_request_id,
                        coverage=packet.coverage_report,
                    )
                    self._ensure_batch_read_receipt_head(
                        connection,
                        processing_run_id=run_id,
                        query_request_id=request.query_request_id,
                        receipt=packet.read_receipt,
                        read_set=read_set,
                    )
                    self._ensure_access_receipt(
                        connection,
                        processing_run_id=run_id,
                        query_request_id=request.query_request_id,
                        receipt=packet.access_receipt,
                    )
                    self._ensure_retrieval_receipt(
                        connection,
                        processing_run_id=run_id,
                        query_request_id=request.query_request_id,
                        receipt=packet.retrieval_receipt,
                    )
                    self._ensure_evidence_packet(
                        connection,
                        packet=packet,
                        created_at=finished_at,
                    )
                    connection.execute(
                        """
                        INSERT INTO recall_run_artifacts(
                            processing_run_id, coverage_report_id, read_receipt_id,
                            access_receipt_id, retrieval_receipt_id, evidence_packet_id
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            packet.coverage_report.coverage_report_id,
                            packet.read_receipt.read_receipt_id,
                            packet.access_receipt.access_receipt_id,
                            packet.retrieval_receipt.retrieval_receipt_id,
                            packet.evidence_packet_id,
                        ),
                    )
                    cursor = connection.execute(
                        """
                        UPDATE processing_runs
                        SET status = 'succeeded', finished_at = ?, error_code = NULL,
                            output_hash = ?
                        WHERE processing_run_id = ? AND status = 'running'
                        """,
                        (finished_at, packet.packet_hash, run_id),
                    )
                    if cursor.rowcount != 1:
                        raise PersistenceConflictError(
                            "recall ProcessingRun changed during batch completion"
                        )
                    _insert_outbox_event(
                        connection,
                        event_id=self._repository._event_id_factory(),
                        event_type="recall_run.completed",
                        aggregate_type="processing_run",
                        aggregate_id=run_id,
                        payload={
                            "schema": "dithyramba.recall_run_completed/1.0",
                            "processing_run_id": run_id,
                            "kind": str(run_row[1]),
                            "query_request_id": request.query_request_id,
                            "evidence_packet_id": packet.evidence_packet_id,
                            "packet_hash": packet.packet_hash,
                            "result_status": packet.result_status.value,
                        },
                        occurred_at=finished_at,
                    )
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("recall batch completion conflicted") from error

        return tuple(
            (self._repository.get_processing_run(run_id), packet)
            for run_id, packet in zip(run_ids, packets, strict=True)
        )

    def fail_recall_run(
        self,
        *,
        processing_run_id: str,
        error_code: str,
        error_hash: str,
    ) -> ProcessingRunRecord:
        """Atomically record a terminal fail-closed recall outcome."""

        code = _validated_label(error_code, "error code")
        digest = _validated_sha256(error_hash, "error hash")
        finished_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                run_row = self._load_running_recall_run(connection, processing_run_id)
                cursor = connection.execute(
                    """
                    UPDATE processing_runs
                    SET status = 'failed', finished_at = ?, error_code = ?, output_hash = ?
                    WHERE processing_run_id = ? AND status = 'running'
                    """,
                    (finished_at, code, digest, processing_run_id),
                )
                if cursor.rowcount != 1:
                    raise PersistenceConflictError("recall ProcessingRun changed during failure")
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="recall_run.failed",
                    aggregate_type="processing_run",
                    aggregate_id=processing_run_id,
                    payload={
                        "schema": "dithyramba.recall_run_failed/1.0",
                        "processing_run_id": processing_run_id,
                        "kind": str(run_row[1]),
                        "query_request_id": str(run_row[2]),
                        "error_code": code,
                        "error_hash": digest,
                    },
                    occurred_at=finished_at,
                )
        except sqlite3.IntegrityError as error:
            raise PersistenceConflictError("recall failure transition conflicted") from error
        return self._repository.get_processing_run(processing_run_id)

    def load_evidence_packet(self, evidence_packet_id: str) -> EvidencePacket:
        """Strictly reconstruct a packet from normalized rows and stored source text."""

        with self._repository._store.transaction() as connection:
            packet_row = connection.execute(
                """
                SELECT ep.evidence_packet_id, ep.query_request_id,
                       ep.corpus_snapshot_id, ep.result_status, ep.packet_json,
                       ep.packet_hash, ep.created_at
                FROM evidence_packets AS ep
                JOIN query_requests AS qr
                  ON qr.query_request_id = ep.query_request_id
                WHERE ep.evidence_packet_id = ? AND qr.library_id = ?
                """,
                (evidence_packet_id, self._repository.library_id),
            ).fetchone()
            if packet_row is None:
                raise EvidencePacketNotFoundError(
                    f"EvidencePacket does not exist: {evidence_packet_id}"
                )
            _validated_timestamp(str(packet_row[6]))
            request = self._load_query_request(connection, str(packet_row[1]))
            snapshot = self._repository._reconstruct_corpus_snapshot(
                connection,
                str(packet_row[2]),
            )
            link_rows = connection.execute(
                """
                SELECT rra.processing_run_id, rra.coverage_report_id,
                       rra.read_receipt_id, rra.access_receipt_id,
                       rra.retrieval_receipt_id, rra.evidence_packet_id,
                       pr.kind, pr.query_request_id, pr.status,
                       pr.code_version, pr.profile_version, pr.output_hash
                FROM recall_run_artifacts AS rra
                JOIN processing_runs AS pr
                  ON pr.processing_run_id = rra.processing_run_id
                WHERE rra.evidence_packet_id = ?
                ORDER BY rra.processing_run_id
                """,
                (evidence_packet_id,),
            ).fetchall()
            if not link_rows:
                raise PersistenceIntegrityError("EvidencePacket has no recall-run artifact link")
            artifact_tuples = {tuple(str(row[index]) for index in range(1, 6)) for row in link_rows}
            if len(artifact_tuples) != 1:
                raise PersistenceIntegrityError(
                    "EvidencePacket resolves to conflicting recall artifact tuples"
                )
            first = link_rows[0]
            for row in link_rows:
                if (
                    str(row[5]) != evidence_packet_id
                    or str(row[6]) not in ("recall", "replay")
                    or str(row[7]) != request.query_request_id
                    or str(row[8]) != ProcessingRunStatus.SUCCEEDED.value
                    or str(row[9]) != str(first[9])
                    or str(row[10]) != str(first[10])
                    or str(row[11]) != str(packet_row[5])
                ):
                    raise PersistenceIntegrityError(
                        "EvidencePacket recall-run link is not terminally closed"
                    )

            coverage = self._load_coverage(
                connection,
                str(first[1]),
                query_request_id=request.query_request_id,
                query_request_hash=request.request_hash,
            )
            access = self._load_access_receipt(
                connection,
                str(first[3]),
                query_request_id=request.query_request_id,
            )
            read = self._load_read_receipt(
                connection,
                str(first[2]),
                query_request_id=request.query_request_id,
                query_request_hash=request.request_hash,
                retrieval_corpus_hash=access.retrieval_corpus_hash,
            )
            retrieval = self._load_retrieval_receipt(
                connection,
                str(first[4]),
                query_request_id=request.query_request_id,
            )
            projections = self._validate_receipt_closure(
                connection,
                request=request,
                snapshot=snapshot,
                coverage=coverage,
                read=read,
                access=access,
                retrieval=retrieval,
                code_version=str(first[9]),
                profile_version=str(first[10]),
            )
            evidence = self._load_packet_items(
                connection,
                evidence_packet_id=evidence_packet_id,
                projections=projections,
            )
            try:
                packet = EvidencePacket(
                    query_request_id=request.query_request_id,
                    query_request_hash=request.request_hash,
                    corpus_snapshot_id=snapshot.corpus_snapshot_id,
                    corpus_snapshot_hash=snapshot.manifest_hash,
                    result_status=PacketResultStatus(str(packet_row[3])),
                    source_fragments=evidence,
                    coverage_report=coverage,
                    read_receipt=read,
                    access_receipt=access,
                    retrieval_receipt=retrieval,
                )
            except (TypeError, ValueError, ValidationError) as error:
                raise PersistenceIntegrityError(
                    "persisted EvidencePacket dependencies are invalid"
                ) from error
            self._validate_packet_closure(
                packet,
                request=request,
                snapshot=snapshot,
                projections=projections,
            )
            packet_json = str(packet_row[4])
            _load_canonical_object(packet_json, "EvidencePacket payload")
            if (
                packet.evidence_packet_id != str(packet_row[0])
                or packet.packet_hash != str(packet_row[5])
                or packet.canonical_bytes.decode("utf-8") != packet_json
            ):
                raise PersistenceIntegrityError(
                    "persisted EvidencePacket differs from its canonical reconstruction"
                )
            return packet

    def _load_query_request(
        self,
        connection: sqlite3.Connection,
        query_request_id: str,
    ) -> QueryRequest:
        rows = connection.execute(
            """
            SELECT query_request_id, library_id, corpus_snapshot_id,
                   access_policy_id, request_json, request_hash, created_at
            FROM query_requests
            WHERE query_request_id = ? AND library_id = ?
            ORDER BY query_request_id
            """,
            (query_request_id, self._repository.library_id),
        ).fetchall()
        if not rows:
            raise QueryRequestNotFoundError(f"QueryRequest does not exist: {query_request_id}")
        if len(rows) != 1:
            raise PersistenceIntegrityError("QueryRequest ID/hash resolve to conflicting rows")
        row = rows[0]
        request = _query_request_from_json(str(row[4]))
        _validated_timestamp(str(row[6]))
        if (
            request.query_request_id != str(row[0])
            or request.query_request_id != query_request_id
            or request.library_id != str(row[1])
            or request.corpus_snapshot_id != str(row[2])
            or request.access_policy_id != str(row[3])
            or request.request_hash != str(row[5])
        ):
            raise PersistenceIntegrityError(
                "persisted QueryRequest differs from relational identity fields"
            )
        self._validate_request_dependencies(connection, request)
        return request

    def _validate_request_dependencies(
        self,
        connection: sqlite3.Connection,
        request: QueryRequest,
    ) -> None:
        if request.library_id != self._repository.library_id:
            raise PersistenceIntegrityError("QueryRequest belongs to a different Library")
        snapshot = self._repository._reconstruct_corpus_snapshot(
            connection,
            request.corpus_snapshot_id,
        )
        if (
            snapshot.library_id != request.library_id
            or snapshot.collection_ids != request.collection_ids
        ):
            raise PersistenceIntegrityError(
                "QueryRequest scope differs from its persisted CorpusSnapshot"
            )
        policy_row = connection.execute(
            """
            SELECT policy_hash FROM access_policies
            WHERE access_policy_id = ? AND library_id = ?
            """,
            (request.access_policy_id, self._repository.library_id),
        ).fetchone()
        if policy_row is None:
            raise PersistenceIntegrityError("QueryRequest AccessPolicy is absent from its Library")
        _validated_sha256(str(policy_row[0]), "AccessPolicy hash")

    def _load_running_recall_run(
        self,
        connection: sqlite3.Connection,
        processing_run_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT pr.processing_run_id, pr.kind, pr.query_request_id,
                   pr.status, pr.code_version, pr.profile_version,
                   pr.started_at, pr.finished_at, pr.error_code, pr.output_hash
            FROM processing_runs AS pr
            JOIN query_requests AS qr ON qr.query_request_id = pr.query_request_id
            WHERE pr.processing_run_id = ? AND qr.library_id = ?
            """,
            (processing_run_id, self._repository.library_id),
        ).fetchone()
        if row is None or str(row[1]) not in ("recall", "replay"):
            raise ProcessingRunNotFoundError(
                f"recall ProcessingRun does not exist: {processing_run_id}"
            )
        if str(row[3]) != ProcessingRunStatus.RUNNING.value:
            raise PersistenceConflictError("recall ProcessingRun is already terminal")
        if row[2] is None or row[7] is not None or row[8] is not None or row[9] is not None:
            raise PersistenceIntegrityError("running recall ProcessingRun has invalid fields")
        _validated_label(str(row[4]), "code version")
        _validated_label(str(row[5]), "profile version")
        _validated_timestamp(str(row[6]))
        return cast(sqlite3.Row, row)

    def _validate_receipt_closure(
        self,
        connection: sqlite3.Connection,
        *,
        request: QueryRequest,
        snapshot: CorpusSnapshot,
        coverage: RecallCoverageReport,
        read: ReadReceipt,
        access: AccessReceipt,
        retrieval: RetrievalReceipt,
        code_version: str,
        profile_version: str,
    ) -> dict[str, _FragmentProjection]:
        for value, expected_type, label in (
            (coverage, RecallCoverageReport, "RecallCoverageReport"),
            (read, ReadReceipt, "ReadReceipt"),
            (access, AccessReceipt, "AccessReceipt"),
            (retrieval, RetrievalReceipt, "RetrievalReceipt"),
        ):
            if type(value) is not expected_type:
                raise PersistenceIntegrityError(f"{label} must use its exact contract type")
        if (
            request.query_request_id != canonical_content_id("query", request.semantic_payload())
            or request.request_hash != canonical_sha256_hex(request.semantic_payload())
            or snapshot.corpus_snapshot_id != request.corpus_snapshot_id
            or snapshot.manifest_hash != canonical_sha256_hex(snapshot.semantic_payload())
            or snapshot.collection_ids != request.collection_ids
        ):
            raise PersistenceIntegrityError("QueryRequest/CorpusSnapshot identity closure failed")
        if any(
            artifact.query_request_hash != request.request_hash
            for artifact in (coverage, read, access, retrieval)
        ):
            raise PersistenceIntegrityError("recall receipt references a different QueryRequest")
        if (
            access.library_id != self._repository.library_id
            or access.access_policy_id != request.access_policy_id
            or access.corpus_snapshot_id != snapshot.corpus_snapshot_id
            or access.snapshot_hash != snapshot.manifest_hash
            or retrieval.code_version != code_version
            or retrieval.profile_version != profile_version
            or retrieval.profile_version != _validated_label(profile_version, "profile version")
            or retrieval.code_version != _validated_label(code_version, "code version")
            or retrieval.profile != request.retrieval.profile
            or retrieval.max_candidates != request.retrieval.max_candidates
            or retrieval.max_source_fragments != request.retrieval.max_source_fragments
        ):
            raise PersistenceIntegrityError("recall receipt dependency projection mismatch")

        scope = RequestScope(
            library_id=request.library_id,
            snapshot_hash=snapshot.manifest_hash,
            purpose=request.purpose,
            collection_ids=request.collection_ids,
            exclusions=request.exclusions,
        )
        compiled = self._repository._compile_db_access(
            access_policy_id=request.access_policy_id,
            scope=scope,
        )
        token = compiled.token
        policy = self._repository.get_access_policy(request.access_policy_id).snapshot
        if (
            access.policy_hash != policy.policy_hash
            or access.policy_hash != token.policy_hash
            or access.exclusion_hash != scope.exclusion_hash
            or access.exclusion_hash != token.exclusion_hash
            or access.permitted_set_hash != compiled.manifest.permitted_set_hash
            or access.permitted_set_hash != token.permitted_set_hash
            or access.policy_omission_present != compiled.public_result.policy_omission_present
            or coverage.policy_omission_present != compiled.public_result.policy_omission_present
        ):
            raise PersistenceIntegrityError("AccessReceipt differs from freshly compiled access")

        manifest_items = compiled.manifest.items
        read_ids = tuple(item.source_fragment_id for item in read.items)
        manifest_ids = tuple(item.source_fragment_id for item in manifest_items)
        if read_ids != manifest_ids:
            raise PersistenceIntegrityError(
                "ReadReceipt is not the exact ordered permitted manifest"
            )
        projections = self._load_fragment_projections(connection, read_ids)
        manifest_by_id = {item.source_fragment_id: item for item in manifest_items}
        for read_item in read.items:
            projection = projections.get(read_item.source_fragment_id)
            manifest_item = manifest_by_id.get(read_item.source_fragment_id)
            if projection is None or manifest_item is None:
                raise PersistenceIntegrityError("ReadReceipt fragment is absent from the Library")
            if (
                read_item.source_version_id != projection.source_version_id
                or read_item.text_sha256 != projection.text_sha256
                or manifest_item.source_version_id != projection.source_version_id
                or manifest_item.source_id != projection.source_id
                or manifest_item.source_family_id != projection.source_family_id
            ):
                raise PersistenceIntegrityError(
                    "ReadReceipt or permitted manifest fragment lineage mismatch"
                )
        corpus_hash = canonical_sha256_hex(
            {
                "schema": "dithyramba.fts_corpus/1.0",
                "fragments": [
                    {
                        "source_fragment_id": fragment_id,
                        "text_sha256": projections[fragment_id].text_sha256,
                    }
                    for fragment_id in sorted(projections)
                ],
            }
        )
        if any(
            value != corpus_hash
            for value in (
                read.retrieval_corpus_hash,
                access.retrieval_corpus_hash,
                retrieval.retrieval_corpus_hash,
            )
        ):
            raise PersistenceIntegrityError("retrieval corpus hash is not reproducible")
        if coverage.processed_count != len(
            {projection.source_version_id for projection in projections.values()}
        ):
            raise PersistenceIntegrityError(
                "RecallCoverageReport processed count differs from read SourceVersions"
            )
        trace_ids = tuple(item.source_fragment_id for item in retrieval.trace)
        if not set(trace_ids).issubset(projections):
            raise PersistenceIntegrityError(
                "RetrievalReceipt contains a fragment outside the permitted read set"
            )
        return projections

    def _validate_batch_receipt_closure(
        self,
        *,
        request: QueryRequest,
        snapshot: CorpusSnapshot,
        packet: EvidencePacket,
        code_version: str,
        profile_version: str,
        first_request: QueryRequest,
        first_snapshot: CorpusSnapshot,
        first_packet: EvidencePacket,
        projections: dict[str, _FragmentProjection],
    ) -> None:
        coverage = packet.coverage_report
        read = packet.read_receipt
        access = packet.access_receipt
        retrieval = packet.retrieval_receipt
        for value, expected_type, label in (
            (coverage, RecallCoverageReport, "RecallCoverageReport"),
            (read, ReadReceipt, "ReadReceipt"),
            (access, AccessReceipt, "AccessReceipt"),
            (retrieval, RetrievalReceipt, "RetrievalReceipt"),
        ):
            if type(value) is not expected_type:
                raise PersistenceIntegrityError(f"{label} must use its exact contract type")
        if (
            _recall_batch_scope_payload(request) != _recall_batch_scope_payload(first_request)
            or snapshot.corpus_snapshot_id != first_snapshot.corpus_snapshot_id
            or snapshot.manifest_hash != first_snapshot.manifest_hash
            or snapshot.library_id != first_snapshot.library_id
            or request.query_request_id != canonical_content_id("query", request.semantic_payload())
            or request.request_hash != canonical_sha256_hex(request.semantic_payload())
        ):
            raise PersistenceIntegrityError("batch completion mixes recall scopes")
        if any(
            artifact.query_request_hash != request.request_hash
            for artifact in (coverage, read, access, retrieval)
        ):
            raise PersistenceIntegrityError("batch receipt references a different QueryRequest")
        first_access = first_packet.access_receipt
        if (
            access.library_id != self._repository.library_id
            or access.access_policy_id != request.access_policy_id
            or access.corpus_snapshot_id != snapshot.corpus_snapshot_id
            or access.snapshot_hash != snapshot.manifest_hash
            or access.policy_hash != first_access.policy_hash
            or access.exclusion_hash != first_access.exclusion_hash
            or access.permitted_set_hash != first_access.permitted_set_hash
            or access.retrieval_corpus_hash != first_access.retrieval_corpus_hash
            or access.policy_omission_present != first_access.policy_omission_present
            or coverage.policy_omission_present != first_access.policy_omission_present
        ):
            raise PersistenceIntegrityError("batch AccessReceipt differs from shared access")
        if (
            read.items != first_packet.read_receipt.items
            or read.retrieval_corpus_hash != first_packet.read_receipt.retrieval_corpus_hash
            or read.retrieval_corpus_hash != access.retrieval_corpus_hash
        ):
            raise PersistenceIntegrityError("batch ReadReceipt differs from shared read set")
        if (
            retrieval.code_version != _validated_label(code_version, "code version")
            or retrieval.profile_version != _validated_label(profile_version, "profile version")
            or retrieval.profile != request.retrieval.profile
            or retrieval.max_candidates != request.retrieval.max_candidates
            or retrieval.max_source_fragments != request.retrieval.max_source_fragments
            or retrieval.retrieval_corpus_hash != read.retrieval_corpus_hash
        ):
            raise PersistenceIntegrityError("batch RetrievalReceipt dependency mismatch")
        expected_versions = {projection.source_version_id for projection in projections.values()}
        if coverage.processed_count != len(expected_versions):
            raise PersistenceIntegrityError(
                "batch CoverageReport processed count differs from shared read set"
            )
        if not {item.source_fragment_id for item in retrieval.trace}.issubset(projections):
            raise PersistenceIntegrityError(
                "batch RetrievalReceipt contains a fragment outside the shared read set"
            )

    def _load_fragment_projections(
        self,
        connection: sqlite3.Connection,
        fragment_ids: tuple[str, ...],
    ) -> dict[str, _FragmentProjection]:
        if not fragment_ids:
            return {}
        if len(fragment_ids) > _LARGE_PROJECTION_READ_THRESHOLD:
            rows = self._load_large_fragment_projection_rows(connection, fragment_ids)
        else:
            rows = []
            with self._repository._permit_fragment_text_read():
                for identifiers in _chunks(fragment_ids, 400):
                    values = ", ".join("(?)" for _ in identifiers)
                    rows.extend(
                        connection.execute(
                            f"""
                            WITH requested(source_fragment_id) AS (VALUES {values})
                            SELECT sf.source_fragment_id, sf.source_version_id,
                                   sv.source_id, sf.text, sf.text_sha256,
                                   sf.source_address_json, sfm.source_family_id
                            FROM requested
                            CROSS JOIN source_fragments AS sf
                                 INDEXED BY sqlite_autoindex_source_fragments_1
                            JOIN source_versions AS sv
                              ON sv.source_version_id = sf.source_version_id
                            JOIN sources AS s ON s.source_id = sv.source_id
                            JOIN source_family_members AS sfm
                              ON sfm.source_id = s.source_id
                            JOIN source_families AS family
                              ON family.source_family_id = sfm.source_family_id
                            WHERE s.library_id = ? AND family.library_id = ?
                              AND sf.source_fragment_id = requested.source_fragment_id
                            """,
                            (
                                *identifiers,
                                self._repository.library_id,
                                self._repository.library_id,
                            ),
                        ).fetchall()
                    )
        if len(rows) != len(fragment_ids) or {str(row[0]) for row in rows} != set(fragment_ids):
            raise PersistenceIntegrityError(
                "fragment projection does not resolve to an exact one-family Library set"
            )
        result: dict[str, _FragmentProjection] = {}
        lineage_by_source: dict[str, tuple[str, str, SourceFamilyRole]] = {}
        for row in rows:
            fragment_id = str(row[0])
            text = str(row[3])
            text_hash = _validated_sha256(str(row[4]), "fragment text hash")
            if sha256_hex(text.encode("utf-8")) != text_hash:
                raise PersistenceIntegrityError("persisted fragment text hash mismatch")
            address = _load_canonical_object(str(row[5]), "SourceAddress")
            source_id = str(row[2])
            lineage = lineage_by_source.get(source_id)
            if lineage is None:
                lineage = self._repository._lineage_for_source(connection, source_id)
                lineage_by_source[source_id] = lineage
            family_id, _root_id, _role = lineage
            if family_id != str(row[6]):
                raise PersistenceIntegrityError("fragment SourceFamily projection mismatch")
            result[fragment_id] = _FragmentProjection(
                source_fragment_id=fragment_id,
                source_version_id=str(row[1]),
                source_id=source_id,
                source_family_id=family_id,
                text=text,
                text_sha256=text_hash,
                source_address=address,
            )
        return result

    def _load_large_fragment_projection_rows(
        self,
        connection: sqlite3.Connection,
        fragment_ids: tuple[str, ...],
    ) -> list[sqlite3.Row]:
        """Materialize a large exact projection set in monotonic rowid order."""

        table = _PROJECTION_READ_PERMIT_TABLE
        connection.execute(f"DROP TABLE IF EXISTS temp.{table}")
        connection.execute(
            f"""
            CREATE TEMP TABLE {table}(
                source_fragment_id TEXT PRIMARY KEY,
                read_order INTEGER NOT NULL UNIQUE,
                source_rowid INTEGER UNIQUE
            ) WITHOUT ROWID
            """
        )
        try:
            connection.executemany(
                f"""
                INSERT INTO {table}(source_fragment_id, read_order, source_rowid)
                VALUES (?, ?, NULL)
                """,
                ((fragment_id, read_order) for read_order, fragment_id in enumerate(fragment_ids)),
            )
            connection.execute(
                f"""
                UPDATE {table} AS permit
                SET source_rowid = (
                    SELECT sf.rowid
                    FROM source_fragments AS sf
                         INDEXED BY sqlite_autoindex_source_fragments_1
                    WHERE sf.source_fragment_id = permit.source_fragment_id
                )
                """
            )
            rowid_rows = connection.execute(
                f"SELECT source_fragment_id, source_rowid FROM {table}"
            ).fetchall()
            if (
                len(rowid_rows) != len(fragment_ids)
                or {str(row[0]) for row in rowid_rows} != set(fragment_ids)
                or any(row[1] is None for row in rowid_rows)
            ):
                raise PersistenceIntegrityError(
                    "fragment projection does not resolve to an exact persisted row set"
                )
            rowids = [int(row[1]) for row in rowid_rows]
            if len(set(rowids)) != len(fragment_ids):
                raise PersistenceIntegrityError(
                    "fragment projection does not resolve to unique persisted rows"
                )

            rows: list[sqlite3.Row] = []
            with self._repository._permit_fragment_text_read():
                for rowid_chunk in _chunks(tuple(str(value) for value in sorted(rowids)), 500):
                    placeholders = ", ".join("?" for _ in rowid_chunk)
                    rows.extend(
                        connection.execute(
                            f"""
                            SELECT sf.source_fragment_id, sf.source_version_id,
                                   sv.source_id, sf.text, sf.text_sha256,
                                   sf.source_address_json, sfm.source_family_id
                            FROM source_fragments AS sf
                            JOIN source_versions AS sv
                              ON sv.source_version_id = sf.source_version_id
                            JOIN sources AS s ON s.source_id = sv.source_id
                            JOIN source_family_members AS sfm
                              ON sfm.source_id = s.source_id
                            JOIN source_families AS family
                              ON family.source_family_id = sfm.source_family_id
                            WHERE s.library_id = ? AND family.library_id = ?
                              AND sf.rowid IN ({placeholders})
                            ORDER BY sf.rowid
                            """,
                            (
                                self._repository.library_id,
                                self._repository.library_id,
                                *rowid_chunk,
                            ),
                        ).fetchall()
                    )
            return rows
        finally:
            connection.execute(f"DROP TABLE IF EXISTS temp.{table}")

    def _validate_packet_closure(
        self,
        packet: EvidencePacket,
        *,
        request: QueryRequest,
        snapshot: CorpusSnapshot,
        projections: dict[str, _FragmentProjection],
    ) -> None:
        if (
            packet.query_request_id != request.query_request_id
            or packet.query_request_hash != request.request_hash
            or packet.corpus_snapshot_id != snapshot.corpus_snapshot_id
            or packet.corpus_snapshot_hash != snapshot.manifest_hash
            or packet.evidence_packet_id
            != canonical_content_id("packet", packet.semantic_payload())
            or packet.packet_hash != canonical_sha256_hex(packet.semantic_payload())
        ):
            raise PersistenceIntegrityError("EvidencePacket identity closure failed")
        for evidence in packet.source_fragments:
            projection = projections.get(evidence.source_fragment_id)
            if projection is None:
                raise PersistenceIntegrityError(
                    "EvidencePacket fragment is outside the permitted read corpus"
                )
            if (
                evidence.source_version_id != projection.source_version_id
                or evidence.source_family_id != projection.source_family_id
                or evidence.text != projection.text
                or evidence.text_sha256 != projection.text_sha256
                or evidence.source_address.payload() != projection.source_address
            ):
                raise PersistenceIntegrityError(
                    "EvidencePacket fragment differs from stored source provenance"
                )

    def _ensure_coverage(
        self,
        connection: sqlite3.Connection,
        *,
        processing_run_id: str,
        query_request_id: str,
        coverage: RecallCoverageReport,
    ) -> None:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO coverage_reports(
                coverage_report_id, processing_run_id, stage,
                processed_count, skipped_count, failed_count,
                policy_omission_present, report_json, report_hash
            ) VALUES (?, ?, 'recall', ?, ?, ?, ?, ?, ?)
            """,
            (
                coverage.coverage_report_id,
                processing_run_id,
                coverage.processed_count,
                coverage.skipped_count,
                coverage.failed_count,
                int(coverage.policy_omission_present),
                coverage.canonical_bytes.decode("utf-8"),
                coverage.report_hash,
            ),
        )
        if cursor.rowcount == 1:
            connection.executemany(
                """
                INSERT INTO omissions(
                    omission_id, coverage_report_id, category,
                    disclosure, count, reason_code
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        _recall_omission_id(coverage.coverage_report_id, omission),
                        coverage.coverage_report_id,
                        omission.category.value,
                        omission.disclosure.value,
                        omission.count,
                        omission.reason_code,
                    )
                    for omission in coverage.omissions
                ),
            )
        _require_unique_artifact_identity(
            connection,
            table="coverage_reports",
            id_column="coverage_report_id",
            identifier=coverage.coverage_report_id,
            hash_column="report_hash",
            digest=coverage.report_hash,
        )
        loaded = self._load_coverage(
            connection,
            coverage.coverage_report_id,
            query_request_id=query_request_id,
            query_request_hash=coverage.query_request_hash,
        )
        if loaded != coverage or loaded.canonical_bytes != coverage.canonical_bytes:
            raise PersistenceIntegrityError(
                "persisted RecallCoverageReport conflicts with content address"
            )

    def _ensure_read_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        processing_run_id: str,
        query_request_id: str,
        receipt: ReadReceipt,
    ) -> None:
        read_receipt_id = receipt.read_receipt_id
        receipt_hash = receipt.receipt_hash
        existing = connection.execute(
            """
            SELECT read_receipt_id, receipt_hash
            FROM read_receipts
            WHERE read_receipt_id = ? OR receipt_hash = ?
            ORDER BY read_receipt_id
            """,
            (read_receipt_id, receipt_hash),
        ).fetchall()
        if not existing:
            read_set = self._ensure_corpus_read_set(connection, receipt)
            connection.execute(
                """
                INSERT INTO read_receipts(
                    read_receipt_id, processing_run_id, receipt_hash
                ) VALUES (?, ?, ?)
                """,
                (read_receipt_id, processing_run_id, receipt_hash),
            )
            connection.execute(
                """
                INSERT INTO read_receipt_corpus_sets(
                    read_receipt_id, corpus_read_set_id
                ) VALUES (?, ?)
                """,
                (read_receipt_id, read_set.corpus_read_set_id),
            )
        elif len(existing) != 1 or (str(existing[0][0]), str(existing[0][1])) != (
            read_receipt_id,
            receipt_hash,
        ):
            raise PersistenceIntegrityError(
                "ReadReceipt ID/hash resolve to conflicting persisted rows"
            )
        _require_unique_artifact_identity(
            connection,
            table="read_receipts",
            id_column="read_receipt_id",
            identifier=read_receipt_id,
            hash_column="receipt_hash",
            digest=receipt_hash,
        )
        loaded = self._load_read_receipt(
            connection,
            read_receipt_id,
            query_request_id=query_request_id,
            query_request_hash=receipt.query_request_hash,
            retrieval_corpus_hash=receipt.retrieval_corpus_hash,
        )
        if loaded != receipt or loaded.canonical_bytes != receipt.canonical_bytes:
            raise PersistenceIntegrityError("persisted ReadReceipt conflicts with content address")

    def _ensure_batch_read_receipt_head(
        self,
        connection: sqlite3.Connection,
        *,
        processing_run_id: str,
        query_request_id: str,
        receipt: ReadReceipt,
        read_set: _CorpusReadSetIdentity,
    ) -> None:
        expected_set = _corpus_read_set_identity(
            library_id=self._repository.library_id,
            retrieval_corpus_hash=receipt.retrieval_corpus_hash,
            items=receipt.items,
        )
        if expected_set != read_set:
            raise PersistenceIntegrityError(
                "batch ReadReceipt does not reference the validated CorpusReadSet"
            )
        read_receipt_id = receipt.read_receipt_id
        receipt_hash = receipt.receipt_hash
        rows = connection.execute(
            """
            SELECT read_receipt_id, receipt_hash
            FROM read_receipts
            WHERE read_receipt_id = ? OR receipt_hash = ?
            ORDER BY read_receipt_id
            """,
            (read_receipt_id, receipt_hash),
        ).fetchall()
        if not rows:
            connection.execute(
                """
                INSERT INTO read_receipts(
                    read_receipt_id, processing_run_id, receipt_hash
                ) VALUES (?, ?, ?)
                """,
                (read_receipt_id, processing_run_id, receipt_hash),
            )
            connection.execute(
                """
                INSERT INTO read_receipt_corpus_sets(
                    read_receipt_id, corpus_read_set_id
                ) VALUES (?, ?)
                """,
                (read_receipt_id, read_set.corpus_read_set_id),
            )
        elif len(rows) == 1 and (str(rows[0][0]), str(rows[0][1])) == (
            read_receipt_id,
            receipt_hash,
        ):
            loaded = self._load_read_receipt(
                connection,
                read_receipt_id,
                query_request_id=query_request_id,
                query_request_hash=receipt.query_request_hash,
                retrieval_corpus_hash=receipt.retrieval_corpus_hash,
            )
            if loaded != receipt or loaded.canonical_bytes != receipt.canonical_bytes:
                raise PersistenceIntegrityError(
                    "persisted batch ReadReceipt conflicts with content address"
                )
        else:
            raise PersistenceIntegrityError(
                "ReadReceipt ID/hash resolve to conflicting persisted rows"
            )
        _require_unique_artifact_identity(
            connection,
            table="read_receipts",
            id_column="read_receipt_id",
            identifier=read_receipt_id,
            hash_column="receipt_hash",
            digest=receipt_hash,
        )

    def _ensure_corpus_read_set(
        self,
        connection: sqlite3.Connection,
        receipt: ReadReceipt,
    ) -> _CorpusReadSetIdentity:
        identity = _corpus_read_set_identity(
            library_id=self._repository.library_id,
            retrieval_corpus_hash=receipt.retrieval_corpus_hash,
            items=receipt.items,
        )
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO corpus_read_sets(
                corpus_read_set_id, library_id, retrieval_corpus_hash,
                item_count, set_hash
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                identity.corpus_read_set_id,
                self._repository.library_id,
                receipt.retrieval_corpus_hash,
                len(receipt.items),
                identity.set_hash,
            ),
        )
        if cursor.rowcount == 1:
            connection.executemany(
                """
                INSERT INTO corpus_read_set_items(
                    corpus_read_set_id, source_fragment_id, source_version_id,
                    read_order, text_sha256
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        identity.corpus_read_set_id,
                        item.source_fragment_id,
                        item.source_version_id,
                        item.read_order,
                        item.text_sha256,
                    )
                    for item in receipt.items
                ),
            )
        rows = connection.execute(
            """
            SELECT corpus_read_set_id, library_id, retrieval_corpus_hash,
                   item_count, set_hash
            FROM corpus_read_sets
            WHERE corpus_read_set_id = ? OR set_hash = ?
            ORDER BY corpus_read_set_id
            """,
            (identity.corpus_read_set_id, identity.set_hash),
        ).fetchall()
        if len(rows) != 1:
            raise PersistenceIntegrityError(
                "CorpusReadSet ID/hash resolve to conflicting persisted rows"
            )
        row = rows[0]
        if (
            str(row[0]) != identity.corpus_read_set_id
            or str(row[1]) != self._repository.library_id
            or str(row[2]) != receipt.retrieval_corpus_hash
            or int(row[3]) != len(receipt.items)
            or str(row[4]) != identity.set_hash
        ):
            raise PersistenceIntegrityError(
                "persisted CorpusReadSet conflicts with its content address"
            )
        loaded_items = self._load_corpus_read_set_items(
            connection,
            corpus_read_set_id=identity.corpus_read_set_id,
            expected_library_id=self._repository.library_id,
        )
        loaded_identity = _corpus_read_set_identity(
            library_id=self._repository.library_id,
            retrieval_corpus_hash=receipt.retrieval_corpus_hash,
            items=loaded_items,
        )
        if loaded_identity != identity or loaded_items != receipt.items:
            raise PersistenceIntegrityError(
                "persisted CorpusReadSet items conflict with its content address"
            )
        return identity

    def _load_corpus_read_set_items(
        self,
        connection: sqlite3.Connection,
        *,
        corpus_read_set_id: str,
        expected_library_id: str,
    ) -> tuple[ReadReceiptItem, ...]:
        set_rows = connection.execute(
            """
            SELECT corpus_read_set_id, library_id, retrieval_corpus_hash,
                   item_count, set_hash
            FROM corpus_read_sets
            WHERE corpus_read_set_id = ?
            """,
            (corpus_read_set_id,),
        ).fetchall()
        if len(set_rows) != 1:
            raise PersistenceIntegrityError("CorpusReadSet identity is absent or ambiguous")
        set_row = set_rows[0]
        if str(set_row[1]) != expected_library_id:
            raise PersistenceIntegrityError("CorpusReadSet crosses the Library boundary")
        raw_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM corpus_read_set_items
                WHERE corpus_read_set_id = ?
                """,
                (corpus_read_set_id,),
            ).fetchone()[0]
        )
        rows = connection.execute(
            """
            SELECT item.source_fragment_id, item.source_version_id,
                   item.read_order, item.text_sha256,
                   fragment.source_version_id, fragment.text_sha256,
                   source.library_id
            FROM corpus_read_set_items AS item
            JOIN source_fragments AS fragment
              ON fragment.source_fragment_id = item.source_fragment_id
            JOIN source_versions AS version
              ON version.source_version_id = fragment.source_version_id
            JOIN sources AS source ON source.source_id = version.source_id
            WHERE item.corpus_read_set_id = ?
            ORDER BY item.read_order
            """,
            (corpus_read_set_id,),
        ).fetchall()
        if len(rows) != raw_count or raw_count != int(set_row[3]):
            raise PersistenceIntegrityError("CorpusReadSet item count is inconsistent")
        try:
            items = tuple(
                ReadReceiptItem(
                    source_fragment_id=str(row[0]),
                    source_version_id=str(row[1]),
                    read_order=int(row[2]),
                    text_sha256=str(row[3]),
                )
                for row in rows
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise PersistenceIntegrityError("persisted CorpusReadSet items are invalid") from error
        if any(
            str(row[1]) != str(row[4])
            or str(row[3]) != str(row[5])
            or str(row[6]) != expected_library_id
            for row in rows
        ):
            raise PersistenceIntegrityError(
                "CorpusReadSet item lineage, hash, or Library is inconsistent"
            )
        identity = _corpus_read_set_identity(
            library_id=expected_library_id,
            retrieval_corpus_hash=str(set_row[2]),
            items=items,
        )
        if identity.corpus_read_set_id != str(set_row[0]) or identity.set_hash != str(set_row[4]):
            raise PersistenceIntegrityError("persisted CorpusReadSet identity mismatch")
        return items

    def _ensure_access_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        processing_run_id: str,
        query_request_id: str,
        receipt: AccessReceipt,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO access_receipts(
                access_receipt_id, processing_run_id, access_policy_id,
                policy_hash, snapshot_hash, exclusion_hash,
                permitted_set_hash, retrieval_corpus_hash,
                policy_omission_present, receipt_json, receipt_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.access_receipt_id,
                processing_run_id,
                receipt.access_policy_id,
                receipt.policy_hash,
                receipt.snapshot_hash,
                receipt.exclusion_hash,
                receipt.permitted_set_hash,
                receipt.retrieval_corpus_hash,
                int(receipt.policy_omission_present),
                receipt.canonical_bytes.decode("utf-8"),
                receipt.receipt_hash,
            ),
        )
        _require_unique_artifact_identity(
            connection,
            table="access_receipts",
            id_column="access_receipt_id",
            identifier=receipt.access_receipt_id,
            hash_column="receipt_hash",
            digest=receipt.receipt_hash,
        )
        loaded = self._load_access_receipt(
            connection,
            receipt.access_receipt_id,
            query_request_id=query_request_id,
        )
        if loaded != receipt or loaded.canonical_bytes != receipt.canonical_bytes:
            raise PersistenceIntegrityError(
                "persisted AccessReceipt conflicts with content address"
            )

    def _ensure_retrieval_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        processing_run_id: str,
        query_request_id: str,
        receipt: RetrievalReceipt,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO retrieval_receipts(
                retrieval_receipt_id, processing_run_id, profile,
                profile_version, candidate_count, selected_count,
                trace_json, receipt_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.retrieval_receipt_id,
                processing_run_id,
                receipt.profile,
                receipt.profile_version,
                receipt.candidate_count,
                receipt.selected_count,
                receipt.canonical_bytes.decode("utf-8"),
                receipt.receipt_hash,
            ),
        )
        _require_unique_artifact_identity(
            connection,
            table="retrieval_receipts",
            id_column="retrieval_receipt_id",
            identifier=receipt.retrieval_receipt_id,
            hash_column="receipt_hash",
            digest=receipt.receipt_hash,
        )
        loaded = self._load_retrieval_receipt(
            connection,
            receipt.retrieval_receipt_id,
            query_request_id=query_request_id,
        )
        if loaded != receipt or loaded.canonical_bytes != receipt.canonical_bytes:
            raise PersistenceIntegrityError(
                "persisted RetrievalReceipt conflicts with content address"
            )

    def _ensure_evidence_packet(
        self,
        connection: sqlite3.Connection,
        *,
        packet: EvidencePacket,
        created_at: str,
    ) -> None:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO evidence_packets(
                evidence_packet_id, query_request_id, corpus_snapshot_id,
                result_status, packet_json, packet_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                packet.evidence_packet_id,
                packet.query_request_id,
                packet.corpus_snapshot_id,
                packet.result_status.value,
                packet.canonical_bytes.decode("utf-8"),
                packet.packet_hash,
                created_at,
            ),
        )
        if cursor.rowcount == 1:
            connection.executemany(
                """
                INSERT INTO packet_items(
                    evidence_packet_id, source_fragment_id, role, rank, score_text
                ) VALUES (?, ?, 'evidence', ?, ?)
                """,
                (
                    (
                        packet.evidence_packet_id,
                        item.source_fragment_id,
                        item.rank,
                        item.score,
                    )
                    for item in packet.source_fragments
                ),
            )
        rows = connection.execute(
            """
            SELECT evidence_packet_id, query_request_id, corpus_snapshot_id,
                   result_status, packet_json, packet_hash, created_at
            FROM evidence_packets
            WHERE evidence_packet_id = ? OR packet_hash = ?
            ORDER BY evidence_packet_id
            """,
            (packet.evidence_packet_id, packet.packet_hash),
        ).fetchall()
        if len(rows) != 1:
            raise PersistenceIntegrityError("EvidencePacket ID/hash resolve to conflicting rows")
        row = rows[0]
        _validated_timestamp(str(row[6]))
        if (
            str(row[0]) != packet.evidence_packet_id
            or str(row[1]) != packet.query_request_id
            or str(row[2]) != packet.corpus_snapshot_id
            or str(row[3]) != packet.result_status.value
            or str(row[4]) != packet.canonical_bytes.decode("utf-8")
            or str(row[5]) != packet.packet_hash
        ):
            raise PersistenceIntegrityError(
                "persisted EvidencePacket conflicts with content address"
            )
        item_rows = connection.execute(
            """
            SELECT source_fragment_id, role, rank, score_text
            FROM packet_items WHERE evidence_packet_id = ?
            ORDER BY rank
            """,
            (packet.evidence_packet_id,),
        ).fetchall()
        expected = tuple(
            (item.source_fragment_id, "evidence", item.rank, item.score)
            for item in packet.source_fragments
        )
        actual = tuple((str(row[0]), str(row[1]), int(row[2]), str(row[3])) for row in item_rows)
        if actual != expected:
            raise PersistenceIntegrityError("persisted packet_items differ from EvidencePacket")

    def _load_coverage(
        self,
        connection: sqlite3.Connection,
        coverage_report_id: str,
        *,
        query_request_id: str,
        query_request_hash: str,
    ) -> RecallCoverageReport:
        rows = connection.execute(
            """
            SELECT cr.coverage_report_id, cr.processing_run_id, cr.stage,
                   cr.processed_count, cr.skipped_count, cr.failed_count,
                   cr.policy_omission_present, cr.report_json, cr.report_hash,
                   pr.kind, pr.query_request_id
            FROM coverage_reports AS cr
            JOIN processing_runs AS pr
              ON pr.processing_run_id = cr.processing_run_id
            WHERE cr.coverage_report_id = ?
            ORDER BY cr.coverage_report_id
            """,
            (coverage_report_id,),
        ).fetchall()
        if not rows:
            raise RecallArtifactNotFoundError(
                f"RecallCoverageReport does not exist: {coverage_report_id}"
            )
        if len(rows) != 1:
            raise PersistenceIntegrityError("CoverageReport ID/hash resolve to conflicting rows")
        row = rows[0]
        if (
            str(row[0]) != coverage_report_id
            or str(row[2]) != "recall"
            or str(row[9]) not in ("recall", "replay")
            or str(row[10]) != query_request_id
        ):
            raise PersistenceIntegrityError("RecallCoverageReport run closure failed")
        payload = _load_canonical_object(str(row[7]), "RecallCoverageReport payload")
        coverage = _coverage_from_payload(payload)
        if (
            coverage.coverage_report_id != str(row[0])
            or coverage.processed_count != int(row[3])
            or coverage.skipped_count != int(row[4])
            or coverage.failed_count != int(row[5])
            or coverage.policy_omission_present != _strict_bool(row[6], "coverage policy omission")
            or coverage.report_hash != str(row[8])
            or coverage.query_request_hash != query_request_hash
        ):
            raise PersistenceIntegrityError(
                "RecallCoverageReport JSON differs from relational fields"
            )
        omission_rows = connection.execute(
            """
            SELECT omission_id, category, disclosure, count, reason_code
            FROM omissions WHERE coverage_report_id = ?
            ORDER BY category, reason_code, omission_id
            """,
            (coverage_report_id,),
        ).fetchall()
        expected = tuple(
            (
                _recall_omission_id(coverage_report_id, omission),
                omission.category.value,
                omission.disclosure.value,
                omission.count,
                omission.reason_code,
            )
            for omission in coverage.omissions
        )
        actual = tuple(
            (
                str(item[0]),
                str(item[1]),
                str(item[2]),
                None if item[3] is None else int(item[3]),
                str(item[4]),
            )
            for item in omission_rows
        )
        if actual != expected:
            raise PersistenceIntegrityError("persisted recall omissions differ from CoverageReport")
        return coverage

    def _load_read_receipt(
        self,
        connection: sqlite3.Connection,
        read_receipt_id: str,
        *,
        query_request_id: str,
        query_request_hash: str,
        retrieval_corpus_hash: str,
    ) -> ReadReceipt:
        row = connection.execute(
            """
            SELECT rr.read_receipt_id, rr.processing_run_id, rr.receipt_hash,
                   pr.kind, pr.query_request_id
            FROM read_receipts AS rr
            JOIN processing_runs AS pr
              ON pr.processing_run_id = rr.processing_run_id
            WHERE rr.read_receipt_id = ?
            """,
            (read_receipt_id,),
        ).fetchone()
        if row is None:
            raise RecallArtifactNotFoundError(f"ReadReceipt does not exist: {read_receipt_id}")
        if str(row[3]) not in ("recall", "replay") or str(row[4]) != query_request_id:
            raise PersistenceIntegrityError("ReadReceipt first-run closure failed")
        legacy_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM read_receipt_items WHERE read_receipt_id = ?",
                (read_receipt_id,),
            ).fetchone()[0]
        )
        shared_rows = connection.execute(
            """
            SELECT link.corpus_read_set_id, read_set.library_id,
                   read_set.retrieval_corpus_hash
            FROM read_receipt_corpus_sets AS link
            JOIN corpus_read_sets AS read_set
              ON read_set.corpus_read_set_id = link.corpus_read_set_id
            WHERE link.read_receipt_id = ?
            """,
            (read_receipt_id,),
        ).fetchall()
        if legacy_count and shared_rows:
            raise PersistenceIntegrityError(
                "ReadReceipt has both legacy and shared item representations"
            )
        if len(shared_rows) > 1:
            raise PersistenceIntegrityError("ReadReceipt has multiple shared item sets")
        if shared_rows:
            shared = shared_rows[0]
            if (
                str(shared[1]) != self._repository.library_id
                or str(shared[2]) != retrieval_corpus_hash
            ):
                raise PersistenceIntegrityError(
                    "ReadReceipt shared set crosses its Library or corpus boundary"
                )
            items = self._load_corpus_read_set_items(
                connection,
                corpus_read_set_id=str(shared[0]),
                expected_library_id=self._repository.library_id,
            )
        else:
            item_rows = connection.execute(
                """
                SELECT rri.source_fragment_id, sf.source_version_id,
                       rri.read_order, rri.text_sha256, sf.text_sha256
                FROM read_receipt_items AS rri
                JOIN source_fragments AS sf
                  ON sf.source_fragment_id = rri.source_fragment_id
                JOIN source_versions AS sv
                  ON sv.source_version_id = sf.source_version_id
                JOIN sources AS s ON s.source_id = sv.source_id
                WHERE rri.read_receipt_id = ? AND s.library_id = ?
                ORDER BY rri.read_order
                """,
                (read_receipt_id, self._repository.library_id),
            ).fetchall()
            if len(item_rows) != legacy_count:
                raise PersistenceIntegrityError("ReadReceipt items cross the Library boundary")
            try:
                items = tuple(
                    ReadReceiptItem(
                        source_fragment_id=str(item[0]),
                        source_version_id=str(item[1]),
                        read_order=int(item[2]),
                        text_sha256=str(item[3]),
                    )
                    for item in item_rows
                )
            except (TypeError, ValueError, ValidationError) as error:
                raise PersistenceIntegrityError("persisted ReadReceipt is invalid") from error
            if any(str(item[3]) != str(item[4]) for item in item_rows):
                raise PersistenceIntegrityError("ReadReceipt text hash differs from SourceFragment")
        try:
            receipt = ReadReceipt(
                query_request_hash=query_request_hash,
                retrieval_corpus_hash=retrieval_corpus_hash,
                items=items,
            )
        except PersistenceIntegrityError:
            raise
        except (TypeError, ValueError, ValidationError) as error:
            raise PersistenceIntegrityError("persisted ReadReceipt is invalid") from error
        if receipt.read_receipt_id != str(row[0]) or receipt.receipt_hash != str(row[2]):
            raise PersistenceIntegrityError("persisted ReadReceipt identity mismatch")
        return receipt

    def _load_access_receipt(
        self,
        connection: sqlite3.Connection,
        access_receipt_id: str,
        *,
        query_request_id: str,
    ) -> AccessReceipt:
        row = connection.execute(
            """
            SELECT ar.access_receipt_id, ar.processing_run_id,
                   ar.access_policy_id, ar.policy_hash, ar.snapshot_hash,
                   ar.exclusion_hash, ar.permitted_set_hash,
                   ar.retrieval_corpus_hash, ar.policy_omission_present,
                   ar.receipt_json, ar.receipt_hash,
                   pr.kind, pr.query_request_id
            FROM access_receipts AS ar
            JOIN processing_runs AS pr
              ON pr.processing_run_id = ar.processing_run_id
            WHERE ar.access_receipt_id = ?
            """,
            (access_receipt_id,),
        ).fetchone()
        if row is None:
            raise RecallArtifactNotFoundError(f"AccessReceipt does not exist: {access_receipt_id}")
        if str(row[11]) not in ("recall", "replay") or str(row[12]) != query_request_id:
            raise PersistenceIntegrityError("AccessReceipt first-run closure failed")
        receipt = _access_from_payload(_load_canonical_object(str(row[9]), "AccessReceipt payload"))
        relational = (
            str(row[0]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[5]),
            str(row[6]),
            str(row[7]),
            _strict_bool(row[8], "access policy omission"),
            str(row[10]),
        )
        expected = (
            receipt.access_receipt_id,
            receipt.access_policy_id,
            receipt.policy_hash,
            receipt.snapshot_hash,
            receipt.exclusion_hash,
            receipt.permitted_set_hash,
            receipt.retrieval_corpus_hash,
            receipt.policy_omission_present,
            receipt.receipt_hash,
        )
        if relational != expected:
            raise PersistenceIntegrityError("AccessReceipt JSON differs from relational fields")
        return receipt

    def _load_retrieval_receipt(
        self,
        connection: sqlite3.Connection,
        retrieval_receipt_id: str,
        *,
        query_request_id: str,
    ) -> RetrievalReceipt:
        row = connection.execute(
            """
            SELECT rr.retrieval_receipt_id, rr.processing_run_id,
                   rr.profile, rr.profile_version, rr.candidate_count,
                   rr.selected_count, rr.trace_json, rr.receipt_hash,
                   pr.kind, pr.query_request_id, pr.code_version,
                   pr.profile_version
            FROM retrieval_receipts AS rr
            JOIN processing_runs AS pr
              ON pr.processing_run_id = rr.processing_run_id
            WHERE rr.retrieval_receipt_id = ?
            """,
            (retrieval_receipt_id,),
        ).fetchone()
        if row is None:
            raise RecallArtifactNotFoundError(
                f"RetrievalReceipt does not exist: {retrieval_receipt_id}"
            )
        if str(row[8]) not in ("recall", "replay") or str(row[9]) != query_request_id:
            raise PersistenceIntegrityError("RetrievalReceipt first-run closure failed")
        receipt = _retrieval_from_payload(
            _load_canonical_object(str(row[6]), "RetrievalReceipt payload")
        )
        relational = (
            str(row[0]),
            str(row[2]),
            str(row[3]),
            int(row[4]),
            int(row[5]),
            str(row[7]),
            str(row[10]),
            str(row[11]),
        )
        expected = (
            receipt.retrieval_receipt_id,
            receipt.profile,
            receipt.profile_version,
            receipt.candidate_count,
            receipt.selected_count,
            receipt.receipt_hash,
            receipt.code_version,
            receipt.profile_version,
        )
        if relational != expected:
            raise PersistenceIntegrityError(
                "RetrievalReceipt JSON differs from relational/run fields"
            )
        return receipt

    def _load_packet_items(
        self,
        connection: sqlite3.Connection,
        *,
        evidence_packet_id: str,
        projections: dict[str, _FragmentProjection],
    ) -> tuple[EvidenceFragment, ...]:
        rows = connection.execute(
            """
            SELECT source_fragment_id, role, rank, score_text
            FROM packet_items
            WHERE evidence_packet_id = ?
            ORDER BY rank
            """,
            (evidence_packet_id,),
        ).fetchall()
        evidence: list[EvidenceFragment] = []
        try:
            for row in rows:
                fragment_id = str(row[0])
                if str(row[1]) != "evidence":
                    raise PersistenceIntegrityError("packet_item role is invalid")
                projection = projections.get(fragment_id)
                if projection is None:
                    raise PersistenceIntegrityError("packet_item is outside the permitted read set")
                evidence.append(
                    EvidenceFragment.model_validate(
                        {
                            "source_fragment_id": fragment_id,
                            "source_version_id": projection.source_version_id,
                            "source_family_id": projection.source_family_id,
                            "rank": int(row[2]),
                            "score": str(row[3]),
                            "text": projection.text,
                            "text_sha256": projection.text_sha256,
                            "source_address": projection.source_address,
                        },
                        strict=True,
                    )
                )
        except PersistenceIntegrityError:
            raise
        except (TypeError, ValueError, ValidationError) as error:
            raise PersistenceIntegrityError("persisted packet_items are invalid") from error
        return tuple(evidence)


def _query_request_from_json(value: str) -> QueryRequest:
    payload = _load_canonical_object(value, "QueryRequest payload")
    _expect_exact_keys(
        payload,
        {
            "schema",
            "question",
            "library_id",
            "collection_ids",
            "corpus_snapshot_id",
            "access_policy_id",
            "purpose",
            "exclusions",
            "retrieval",
            "result_contract",
        },
        "QueryRequest",
    )
    if payload["schema"] != QueryRequest.SCHEMA:
        raise PersistenceIntegrityError("persisted QueryRequest schema is invalid")
    collection_ids = payload["collection_ids"]
    exclusions = payload["exclusions"]
    retrieval = payload["retrieval"]
    result_contract = payload["result_contract"]
    if (
        type(collection_ids) is not list
        or type(exclusions) is not dict
        or type(retrieval) is not dict
        or type(result_contract) is not dict
    ):
        raise PersistenceIntegrityError("persisted QueryRequest nested shape is invalid")
    _expect_exact_keys(
        cast(dict[str, object], exclusions),
        {"source_ids", "source_family_ids", "source_fragment_ids"},
        "QueryRequest exclusions",
    )
    _expect_exact_keys(
        cast(dict[str, object], retrieval),
        {"profile", "max_candidates", "max_source_fragments"},
        "QueryRequest retrieval",
    )
    _expect_exact_keys(
        cast(dict[str, object], result_contract),
        {"format", "require_source_addresses"},
        "QueryRequest result_contract",
    )
    raw_exclusions = cast(dict[str, object], exclusions)
    source_ids = raw_exclusions["source_ids"]
    family_ids = raw_exclusions["source_family_ids"]
    fragment_ids = raw_exclusions["source_fragment_ids"]
    if any(type(item) is not list for item in (source_ids, family_ids, fragment_ids)):
        raise PersistenceIntegrityError("persisted QueryRequest exclusions are invalid")
    try:
        request = QueryRequest(
            question=cast(str, payload["question"]),
            library_id=cast(str, payload["library_id"]),
            collection_ids=tuple(cast(list[str], collection_ids)),
            corpus_snapshot_id=cast(str, payload["corpus_snapshot_id"]),
            access_policy_id=cast(str, payload["access_policy_id"]),
            purpose=cast(str, payload["purpose"]),
            exclusions=QueryExclusions(
                source_ids=tuple(cast(list[str], source_ids)),
                source_family_ids=tuple(cast(list[str], family_ids)),
                source_fragment_ids=tuple(cast(list[str], fragment_ids)),
            ),
            retrieval=RetrievalBudget.model_validate(retrieval, strict=True),
            result_contract=EvidencePacketResultContract.model_validate(
                result_contract,
                strict=True,
            ),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise PersistenceIntegrityError("persisted QueryRequest is invalid") from error
    if request.canonical_bytes.decode("utf-8") != value:
        raise PersistenceIntegrityError("persisted QueryRequest is not canonical")
    return request


def _coverage_from_payload(payload: dict[str, object]) -> RecallCoverageReport:
    _expect_exact_keys(
        payload,
        {
            "schema",
            "stage",
            "query_request_hash",
            "processed_count",
            "skipped_count",
            "failed_count",
            "policy_omission",
            "omissions",
        },
        "RecallCoverageReport",
    )
    policy = payload["policy_omission"]
    omissions = payload["omissions"]
    if (
        payload["schema"] != RecallCoverageReport.SCHEMA
        or payload["stage"] != "recall"
        or type(policy) is not dict
        or type(omissions) is not list
    ):
        raise PersistenceIntegrityError("persisted RecallCoverageReport shape is invalid")
    _expect_exact_keys(cast(dict[str, object], policy), {"present", "count"}, "policy omission")
    policy_values = cast(dict[str, object], policy)
    if type(policy_values["present"]) is not bool or policy_values["count"] is not None:
        raise PersistenceIntegrityError("persisted policy omission projection is invalid")
    typed_omissions: list[RecallOmission] = []
    try:
        for raw in cast(list[object], omissions):
            if type(raw) is not dict:
                raise PersistenceIntegrityError("persisted omission must be an object")
            item = cast(dict[str, object], raw)
            _expect_exact_keys(
                item,
                {"category", "disclosure", "count", "reason_code"},
                "RecallOmission",
            )
            typed_omissions.append(
                RecallOmission(
                    category=OmissionCategory(cast(str, item["category"])),
                    disclosure=OmissionDisclosure(cast(str, item["disclosure"])),
                    count=cast(int | None, item["count"]),
                    reason_code=cast(str, item["reason_code"]),
                )
            )
        coverage = RecallCoverageReport(
            query_request_hash=cast(str, payload["query_request_hash"]),
            processed_count=cast(int, payload["processed_count"]),
            skipped_count=cast(int, payload["skipped_count"]),
            failed_count=cast(int, payload["failed_count"]),
            policy_omission_present=policy_values["present"],
            omissions=tuple(typed_omissions),
        )
    except PersistenceIntegrityError:
        raise
    except (TypeError, ValueError, ValidationError) as error:
        raise PersistenceIntegrityError("persisted RecallCoverageReport is invalid") from error
    if coverage.semantic_payload() != payload:
        raise PersistenceIntegrityError(
            "persisted RecallCoverageReport is not its typed canonical projection"
        )
    return coverage


def _access_from_payload(payload: dict[str, object]) -> AccessReceipt:
    _expect_exact_keys(
        payload,
        {
            "schema",
            "library_id",
            "query_request_hash",
            "access_policy_id",
            "policy_hash",
            "corpus_snapshot_id",
            "snapshot_hash",
            "exclusion_hash",
            "permitted_set_hash",
            "retrieval_corpus_hash",
            "policy_omission_present",
        },
        "AccessReceipt",
    )
    if payload["schema"] != AccessReceipt.SCHEMA:
        raise PersistenceIntegrityError("persisted AccessReceipt schema is invalid")
    try:
        receipt = AccessReceipt(
            library_id=cast(str, payload["library_id"]),
            query_request_hash=cast(str, payload["query_request_hash"]),
            access_policy_id=cast(str, payload["access_policy_id"]),
            policy_hash=cast(str, payload["policy_hash"]),
            corpus_snapshot_id=cast(str, payload["corpus_snapshot_id"]),
            snapshot_hash=cast(str, payload["snapshot_hash"]),
            exclusion_hash=cast(str, payload["exclusion_hash"]),
            permitted_set_hash=cast(str, payload["permitted_set_hash"]),
            retrieval_corpus_hash=cast(str, payload["retrieval_corpus_hash"]),
            policy_omission_present=cast(bool, payload["policy_omission_present"]),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise PersistenceIntegrityError("persisted AccessReceipt is invalid") from error
    if receipt.semantic_payload() != payload:
        raise PersistenceIntegrityError("persisted AccessReceipt is not canonical")
    return receipt


def _retrieval_from_payload(payload: dict[str, object]) -> RetrievalReceipt:
    _expect_exact_keys(
        payload,
        {
            "schema",
            "profile",
            "query_request_hash",
            "profile_version",
            "code_version",
            "retrieval_corpus_hash",
            "max_candidates",
            "max_source_fragments",
            "candidate_count",
            "selected_count",
            "trace",
        },
        "RetrievalReceipt",
    )
    trace = payload["trace"]
    if payload["schema"] != RetrievalReceipt.SCHEMA or type(trace) is not list:
        raise PersistenceIntegrityError("persisted RetrievalReceipt shape is invalid")
    typed_trace: list[RetrievalTraceItem] = []
    try:
        for raw in cast(list[object], trace):
            if type(raw) is not dict:
                raise PersistenceIntegrityError("persisted retrieval trace item is invalid")
            item = cast(dict[str, object], raw)
            _expect_exact_keys(
                item,
                {"source_fragment_id", "rank", "score"},
                "RetrievalTraceItem",
            )
            typed_trace.append(
                RetrievalTraceItem(
                    source_fragment_id=cast(str, item["source_fragment_id"]),
                    rank=cast(int, item["rank"]),
                    score=cast(str, item["score"]),
                )
            )
        receipt = RetrievalReceipt(
            profile=cast(Literal["fts_v1"], payload["profile"]),
            query_request_hash=cast(str, payload["query_request_hash"]),
            profile_version=cast(str, payload["profile_version"]),
            code_version=cast(str, payload["code_version"]),
            retrieval_corpus_hash=cast(str, payload["retrieval_corpus_hash"]),
            max_candidates=cast(int, payload["max_candidates"]),
            max_source_fragments=cast(int, payload["max_source_fragments"]),
            candidate_count=cast(int, payload["candidate_count"]),
            selected_count=cast(int, payload["selected_count"]),
            trace=tuple(typed_trace),
        )
    except PersistenceIntegrityError:
        raise
    except (TypeError, ValueError, ValidationError) as error:
        raise PersistenceIntegrityError("persisted RetrievalReceipt is invalid") from error
    if receipt.semantic_payload() != payload:
        raise PersistenceIntegrityError("persisted RetrievalReceipt is not canonical")
    return receipt


def _corpus_read_set_identity(
    *,
    library_id: str,
    retrieval_corpus_hash: str,
    items: tuple[ReadReceiptItem, ...],
) -> _CorpusReadSetIdentity:
    payload: dict[str, object] = {
        "schema": "dithyramba.corpus_read_set/1.0",
        "library_id": library_id,
        "retrieval_corpus_hash": retrieval_corpus_hash,
        "items": [item.payload() for item in items],
    }
    return _CorpusReadSetIdentity(
        corpus_read_set_id=canonical_content_id("corpus_read_set", payload),
        set_hash=canonical_sha256_hex(payload),
        payload=payload,
    )


def _recall_batch_scope_payload(request: QueryRequest) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "dithyramba.recall_batch_scope/1.0",
            "library_id": request.library_id,
            "collection_ids": list(request.collection_ids),
            "corpus_snapshot_id": request.corpus_snapshot_id,
            "access_policy_id": request.access_policy_id,
            "purpose": request.purpose,
            "exclusions": request.exclusions.payload(),
            "retrieval": request.retrieval.payload(),
            "result_contract": request.result_contract.payload(),
        }
    )


def _recall_omission_id(
    coverage_report_id: str,
    omission: RecallOmission,
) -> str:
    return canonical_content_id(
        "omission",
        {
            "schema": "dithyramba.omission/1.0",
            "coverage_report_id": coverage_report_id,
            **omission.payload(),
        },
    )


def _load_canonical_object(value: str, label: str) -> dict[str, object]:
    try:
        decoded: object = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        raise PersistenceIntegrityError(f"{label} is not valid JSON") from error
    if type(decoded) is not dict:
        raise PersistenceIntegrityError(f"{label} must be a JSON object")
    payload = cast(dict[str, object], decoded)
    try:
        canonical = canonical_json_bytes(payload).decode("utf-8")
    except Exception as error:
        raise PersistenceIntegrityError(f"{label} is not canonicalizable") from error
    if canonical != value:
        raise PersistenceIntegrityError(f"{label} is not canonical JSON")
    return payload


def _expect_exact_keys(
    payload: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(payload) != expected:
        raise PersistenceIntegrityError(f"{label} keys differ from the frozen schema")


def _strict_bool(value: object, label: str) -> bool:
    if type(value) is not int or value not in (0, 1):
        raise PersistenceIntegrityError(f"{label} must be stored as 0 or 1")
    return bool(value)


def _validated_recall_kind(value: object) -> RecallRunKind:
    if type(value) is not str or value not in ("recall", "replay"):
        raise ValueError("recall run kind must be recall or replay")
    return cast(RecallRunKind, value)


def _require_unique_artifact_identity(
    connection: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    identifier: str,
    hash_column: str,
    digest: str,
) -> None:
    allowed = {
        ("coverage_reports", "coverage_report_id", "report_hash"),
        ("read_receipts", "read_receipt_id", "receipt_hash"),
        ("access_receipts", "access_receipt_id", "receipt_hash"),
        ("retrieval_receipts", "retrieval_receipt_id", "receipt_hash"),
    }
    if (table, id_column, hash_column) not in allowed:
        raise AssertionError("unsupported recall artifact identity query")
    rows = connection.execute(
        f"SELECT {id_column} FROM {table} "
        f"WHERE {id_column} = ? OR {hash_column} = ? ORDER BY {id_column}",
        (identifier, digest),
    ).fetchall()
    if len(rows) != 1 or str(rows[0][0]) != identifier:
        raise PersistenceIntegrityError(
            "recall artifact ID/hash resolve to conflicting persisted rows"
        )


def _chunks(values: tuple[str, ...], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(values[index : index + size] for index in range(0, len(values), size))


def _validated_run_id(value: str) -> str:
    if type(value) is not str or _RUN_ID_PATTERN.fullmatch(value) is None:
        raise PersistenceIntegrityError("recall run ID must use a canonical run_ prefix")
    return value


__all__ = ["SQLiteRecallBackend"]
