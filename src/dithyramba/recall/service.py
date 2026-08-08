"""Synchronous, provider-free VS0 recall and deterministic replay orchestration.

The service depends on a narrow structural backend instead of a concrete
repository.  This keeps policy-gated text reads and artifact persistence behind
one injectable boundary while allowing the orchestration itself to be tested
without SQLite, filesystem, provider, or network access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, cast, runtime_checkable

from dithyramba._version import __version__
from dithyramba.access import (
    CompiledAccess,
    MembershipState,
    PermittedManifest,
    PermittedManifestItem,
    PermittedToken,
    PublicAccessResult,
    RequestScope,
)
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, new_id, sha256_hex
from dithyramba.persistence.errors import ProcessingRunNotFoundError
from dithyramba.persistence.models import AuthorizedRead, SourceFragmentText
from dithyramba.provenance import ProcessingRunRecord, ProcessingRunStatus
from dithyramba.snapshots import CorpusSnapshot, SnapshotMember

from .artifacts import (
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
from .fts import (
    FtsCandidate,
    FtsFragment,
    FtsRuntimeProfile,
    FtsSearchResult,
    PermittedFtsSession,
    search_ephemeral_fts,
)
from .models import QueryRequest

RecallRunKind = Literal["recall", "replay"]

_VERSION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,127}$")
_RUN_ID_PATTERN = re.compile(r"^run_[a-z0-9]+(?:_[a-z0-9]+)*$")
_PACKET_ID_PATTERN = re.compile(r"^packet_[a-z0-9]+(?:_[a-z0-9]+)*$")
_AUTHORIZATION_ID_PATTERN = re.compile(r"^authorization_[a-z0-9]+(?:_[a-z0-9]+)*$")
_REPOSITORY_ID_PATTERN = re.compile(r"^repository_[a-z0-9]+(?:_[a-z0-9]+)*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RecallError(RuntimeError):
    """Base class for stable, public RecallService failures."""

    code: ClassVar[str] = "recall_failed"


class RecallRequestError(RecallError):
    """The caller supplied an invalid or non-canonical recall identity."""

    code = "recall_request_invalid"


class RecallSnapshotError(RecallError):
    """The loaded CorpusSnapshot does not close over the request scope."""

    code = "recall_snapshot_mismatch"


class RecallDependencyMismatchError(RecallError):
    """Replay dependencies differ from the dependencies recorded originally."""

    code = "replay_dependency_mismatch"


class RecallExecutionError(RecallError):
    """Authorization, protected reading, FTS, or artifact assembly failed."""

    code = "recall_execution_failed"


class RecallInterruptedError(RecallError):
    """An operator or runtime interruption stopped an already-started recall."""

    code = "recall_interrupted"


class RecallPersistenceError(RecallError):
    """A required durable recall transition could not be completed."""

    code = "recall_persistence_failed"


class RecallReplayMismatchError(RecallError):
    """A replay completed locally but did not reproduce the original packet."""

    code = "replay_result_mismatch"


@dataclass(frozen=True, slots=True)
class RecallResult:
    """One terminal successful recall/replay run and its canonical packet."""

    run: ProcessingRunRecord
    packet: EvidencePacket

    @property
    def succeeded(self) -> bool:
        return self.run.status is ProcessingRunStatus.SUCCEEDED


@dataclass(frozen=True, slots=True)
class RecallScopeSessionStats:
    """Non-canonical runtime observations for one ephemeral recall session."""

    permitted_fragment_count: int
    index_build_count: int
    search_count: int


@dataclass(frozen=True, slots=True)
class SourceLocalDrilldownResult:
    """One bounded FTS result restricted to already authorized Sources."""

    question: str
    source_ids: tuple[str, ...]
    result: FtsSearchResult
    fragments: tuple[SourceFragmentText, ...]


class RecallScopeSession:
    """Ephemeral authorized read-set and FTS index bound to one exact scope.

    The capability is deliberately process-local and rebuildable.  It does not
    persist source text or ranking state, and every recall performed through it
    still writes the ordinary QueryRequest, ProcessingRun, receipts and
    EvidencePacket.  Closing the capability destroys its in-memory SQLite
    connection.
    """

    def __init__(
        self,
        *,
        owner: RecallService,
        scope_bytes: bytes,
        snapshot: CorpusSnapshot,
        prepared: _PreparedRecallCorpus,
        fts_session: PermittedFtsSession,
    ) -> None:
        self._owner = owner
        self._scope_bytes = scope_bytes
        self._snapshot = snapshot
        self._prepared = prepared
        self._fts_session = fts_session
        self._search_count = 0
        self._completion_capability = object()

    @property
    def closed(self) -> bool:
        return self._fts_session.closed

    @property
    def scope_hash(self) -> str:
        return sha256_hex(self._scope_bytes)

    @property
    def stats(self) -> RecallScopeSessionStats:
        return RecallScopeSessionStats(
            permitted_fragment_count=len(self._prepared.fts_fragments),
            index_build_count=1,
            search_count=self._search_count,
        )

    def __enter__(self) -> RecallScopeSession:
        if self.closed:
            raise RecallRequestError("closed RecallScopeSession cannot be reopened")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._fts_session.close()

    def _require_request(self, owner: RecallService, request: QueryRequest) -> None:
        if owner is not self._owner:
            raise RecallRequestError("RecallScopeSession belongs to a different RecallService")
        if self.closed:
            raise RecallRequestError("RecallScopeSession is closed")
        if _session_scope_bytes(request) != self._scope_bytes:
            raise RecallRequestError("QueryRequest differs from the RecallScopeSession scope")

    def _search(self, request: QueryRequest) -> FtsSearchResult:
        result = self._fts_session.search(
            question=request.question,
            max_candidates=request.retrieval.max_candidates,
        )
        self._search_count += 1
        return result


@dataclass(slots=True)
class _RecallLifecycle:
    """Mutable orchestration state used only to reconcile ambiguous commits."""

    processing_run_id: str
    kind: RecallRunKind
    started: ProcessingRunRecord | None = None
    packet: EvidencePacket | None = None


@dataclass(frozen=True, slots=True)
class _PreparedRecallCorpus:
    """One validated protected read reusable only inside an exact batch scope."""

    authorization: AuthorizedRead
    manifest: PermittedManifest
    read_fragments: tuple[SourceFragmentText, ...]
    read_receipt_items: tuple[ReadReceiptItem, ...]
    fragments_by_id: dict[str, SourceFragmentText]
    addresses: dict[str, dict[str, object]]
    fts_fragments: tuple[FtsFragment, ...]


class FtsSearch(Protocol):
    """Injected fresh-FTS seam; implementations must remain provider-free."""

    def __call__(
        self,
        *,
        question: str,
        fragments: tuple[FtsFragment, ...],
        max_candidates: int,
    ) -> FtsSearchResult: ...


class RecallBackend(Protocol):
    """Minimal protected-read and atomic-artifact backend required by recall."""

    def get_corpus_snapshot(self, corpus_snapshot_id: str) -> CorpusSnapshot: ...

    def persist_query_request(self, request: QueryRequest) -> None: ...

    def load_query_request(self, query_request_id: str) -> QueryRequest: ...

    def begin_recall_run(
        self,
        *,
        processing_run_id: str,
        kind: RecallRunKind,
        query_request_id: str,
        code_version: str,
        profile_version: str,
    ) -> ProcessingRunRecord: ...

    def get_processing_run(self, processing_run_id: str) -> ProcessingRunRecord: ...

    def authorize_read(
        self,
        *,
        access_policy_id: str,
        scope: RequestScope,
    ) -> AuthorizedRead: ...

    def read_permitted_fragments(
        self,
        authorization: AuthorizedRead,
    ) -> tuple[SourceFragmentText, ...]: ...

    def complete_recall_run(
        self,
        *,
        processing_run_id: str,
        packet: EvidencePacket,
    ) -> tuple[ProcessingRunRecord, EvidencePacket]: ...

    def fail_recall_run(
        self,
        *,
        processing_run_id: str,
        error_code: str,
        error_hash: str,
    ) -> ProcessingRunRecord: ...

    def load_evidence_packet(self, evidence_packet_id: str) -> EvidencePacket: ...


@runtime_checkable
class _AtomicRecallBatchBackend(Protocol):
    """Optional backend extension for one-transaction batch completion."""

    def complete_recall_batch(
        self,
        *,
        completions: tuple[tuple[str, EvidencePacket], ...],
    ) -> tuple[tuple[ProcessingRunRecord, EvidencePacket], ...]: ...


@runtime_checkable
class _ScopedRecallCompletionBackend(Protocol):
    """Optional backend seam for reusing one fully validated scope in-process."""

    def complete_scoped_recall_run(
        self,
        *,
        processing_run_id: str,
        packet: EvidencePacket,
        scope_capability: object,
    ) -> tuple[ProcessingRunRecord, EvidencePacket]: ...


class RecallService:
    """Build EvidencePackets from one exact snapshot through protected local FTS."""

    def __init__(
        self,
        backend: RecallBackend,
        *,
        profile_version: str,
        code_version: str = __version__,
        fts_search: FtsSearch = search_ephemeral_fts,
    ) -> None:
        self._backend = backend
        self._profile_version = _validated_version(profile_version, "profile_version")
        self._code_version = _validated_version(code_version, "code_version")
        if not callable(fts_search):
            raise TypeError("fts_search must be callable")
        self._fts_search = fts_search

    @property
    def backend(self) -> RecallBackend:
        """Return the injected Library-scoped backend."""

        return self._backend

    def recall(self, request: QueryRequest) -> RecallResult:
        """Persist and execute one exact QueryRequest against its frozen snapshot."""

        canonical_request = _validated_request(request)
        snapshot = self._load_and_validate_snapshot(canonical_request)
        try:
            self._backend.persist_query_request(canonical_request)
            persisted = self._backend.load_query_request(canonical_request.query_request_id)
        except Exception as error:
            raise RecallPersistenceError(
                "QueryRequest could not be persisted and reloaded"
            ) from error
        try:
            _require_same_request(persisted, canonical_request)
        except RecallPersistenceError:
            raise
        except Exception as error:
            raise RecallPersistenceError("persisted QueryRequest is not canonical") from error
        return self._execute(
            kind="recall",
            request=canonical_request,
            snapshot=snapshot,
            expected_packet=None,
        )

    def open_scope_session(self, request: QueryRequest) -> RecallScopeSession:
        """Build one reusable, process-local index for an exact authorized scope."""

        if self._fts_search is not search_ephemeral_fts:
            raise RecallDependencyMismatchError(
                "scope sessions require the canonical reusable FTS dependency"
            )
        canonical_request = _validated_request(request)
        snapshot = self._load_and_validate_snapshot(canonical_request)
        try:
            prepared = self._prepare_recall_corpus(
                request=canonical_request,
                snapshot=snapshot,
            )
            fts_session = PermittedFtsSession(fragments=prepared.fts_fragments)
        except RecallError:
            raise
        except Exception as error:
            raise RecallExecutionError("RecallScopeSession could not be built") from error
        if fts_session.profile.profile_version != self._profile_version:
            fts_session.close()
            raise RecallDependencyMismatchError(
                "scope-session FTS runtime differs from the service profile"
            )
        return RecallScopeSession(
            owner=self,
            scope_bytes=_session_scope_bytes(canonical_request),
            snapshot=snapshot,
            prepared=prepared,
            fts_session=fts_session,
        )

    def recall_in_session(
        self,
        request: QueryRequest,
        scope_session: RecallScopeSession,
    ) -> RecallResult:
        """Run one ordinary durable recall through a reusable scope session."""

        canonical_request = _validated_request(request)
        if type(scope_session) is not RecallScopeSession:
            raise RecallRequestError("recall_in_session requires an exact RecallScopeSession")
        scope_session._require_request(self, canonical_request)
        snapshot = self._load_and_validate_snapshot(canonical_request)
        if snapshot != scope_session._snapshot:
            raise RecallSnapshotError("RecallScopeSession snapshot differs from QueryRequest")
        try:
            self._backend.persist_query_request(canonical_request)
            persisted = self._backend.load_query_request(canonical_request.query_request_id)
            _require_same_request(persisted, canonical_request)
        except RecallPersistenceError:
            raise
        except Exception as error:
            raise RecallPersistenceError(
                "QueryRequest could not be persisted and reloaded"
            ) from error
        return self._execute(
            kind="recall",
            request=canonical_request,
            snapshot=snapshot,
            expected_packet=None,
            scope_session=scope_session,
        )

    def source_local_drilldown(
        self,
        request: QueryRequest,
        scope_session: RecallScopeSession,
        *,
        question: str,
        source_ids: tuple[str, ...],
        max_candidates: int = 100,
    ) -> SourceLocalDrilldownResult:
        """Search only inside named Sources from one existing authorized scope.

        This is an answer-route drilldown, not a replacement retrieval receipt. The
        original broad FTS packet remains immutable; callers must bind any local
        candidates to a subsequent evidence gate before using them in an answer.
        """

        canonical_request = _validated_request(request)
        if type(scope_session) is not RecallScopeSession:
            raise RecallRequestError("source drilldown requires an exact RecallScopeSession")
        scope_session._require_request(self, canonical_request)
        if (
            type(source_ids) is not tuple
            or not source_ids
            or any(type(source_id) is not str or not source_id for source_id in source_ids)
            or len(set(source_ids)) != len(source_ids)
        ):
            raise RecallRequestError("source drilldown requires unique nonblank Source IDs")
        selected_source_ids = tuple(sorted(source_ids, key=lambda item: item.encode("ascii")))
        available_source_ids = {
            fragment.source_id for fragment in scope_session._prepared.read_fragments
        }
        if not set(selected_source_ids).issubset(available_source_ids):
            raise RecallRequestError("source drilldown exceeds the authorized read scope")
        selected_fragments = tuple(
            fragment
            for fragment in scope_session._prepared.read_fragments
            if fragment.source_id in selected_source_ids
        )
        fts_fragments = tuple(
            FtsFragment(
                source_fragment_id=fragment.source_fragment_id,
                text=fragment.text,
                text_sha256=fragment.text_sha256,
            )
            for fragment in selected_fragments
        )
        result = search_ephemeral_fts(
            question=question,
            fragments=fts_fragments,
            max_candidates=max_candidates,
        )
        result = _validated_fts_result(
            result,
            fragments=fts_fragments,
            profile_version=self._profile_version,
            max_candidates=max_candidates,
        )
        fragments_by_id = {fragment.source_fragment_id: fragment for fragment in selected_fragments}
        scope_session._search_count += 1
        return SourceLocalDrilldownResult(
            question=question,
            source_ids=selected_source_ids,
            result=result,
            fragments=tuple(fragments_by_id[item.source_fragment_id] for item in result.trace),
        )

    def recall_batch(self, requests: tuple[QueryRequest, ...]) -> tuple[RecallResult, ...]:
        """Recall one exact scope through one protected read and one FTS session.

        Every request keeps its ordinary durable QueryRequest, ProcessingRun,
        EvidencePacket, and receipt identities.  The optimization is restricted
        to the immutable authorized corpus: mixed scopes and substituted
        one-shot FTS dependencies are rejected before any run starts.
        """

        canonical_requests = _validated_batch_requests(requests)
        if self._fts_search is not search_ephemeral_fts:
            raise RecallDependencyMismatchError(
                "batch recall requires the canonical reusable FTS dependency"
            )
        snapshot = self._load_and_validate_snapshot(canonical_requests[0])
        self._persist_batch_requests(canonical_requests)
        lifecycles = tuple(
            _RecallLifecycle(processing_run_id=new_id("run"), kind="recall")
            for _request in canonical_requests
        )
        try:
            try:
                for request, lifecycle in zip(canonical_requests, lifecycles, strict=True):
                    lifecycle.started = self._begin_run(
                        processing_run_id=lifecycle.processing_run_id,
                        kind="recall",
                        request=request,
                    )
                prepared = self._prepare_recall_corpus(
                    request=canonical_requests[0],
                    snapshot=snapshot,
                )
                with PermittedFtsSession(fragments=prepared.fts_fragments) as session:
                    if session.profile.profile_version != self._profile_version:
                        raise RecallDependencyMismatchError(
                            "batch FTS runtime differs from the service profile"
                        )
                    for request, lifecycle in zip(canonical_requests, lifecycles, strict=True):
                        result = session.search(
                            question=request.question,
                            max_candidates=request.retrieval.max_candidates,
                        )
                        validated_result = _validated_fts_result(
                            result,
                            fragments=prepared.fts_fragments,
                            profile_version=self._profile_version,
                            max_candidates=request.retrieval.max_candidates,
                        )
                        lifecycle.packet = self._assemble_packet(
                            request=request,
                            snapshot=snapshot,
                            prepared=prepared,
                            result=validated_result,
                        )
                prepared_completions: list[tuple[str, EvidencePacket]] = []
                for lifecycle in lifecycles:
                    if lifecycle.started is None or lifecycle.packet is None:
                        raise RecallExecutionError("batch recall did not prepare every request")
                    prepared_completions.append((lifecycle.processing_run_id, lifecycle.packet))
                if isinstance(self._backend, _AtomicRecallBatchBackend):
                    try:
                        completions = self._backend.complete_recall_batch(
                            completions=tuple(prepared_completions),
                        )
                    except Exception as error:
                        raise RecallPersistenceError(
                            "batch recall artifacts could not be atomically persisted"
                        ) from error
                    if type(completions) is not tuple or len(completions) != len(lifecycles):
                        raise RecallPersistenceError(
                            "atomic batch backend returned an invalid completion tuple"
                        )
                else:
                    sequential: list[tuple[ProcessingRunRecord, EvidencePacket]] = []
                    for processing_run_id, packet in prepared_completions:
                        try:
                            sequential.append(
                                self._backend.complete_recall_run(
                                    processing_run_id=processing_run_id,
                                    packet=packet,
                                )
                            )
                        except Exception as error:
                            raise RecallPersistenceError(
                                "batch recall artifacts could not be atomically persisted"
                            ) from error
                    completions = tuple(sequential)

                results: list[RecallResult] = []
                for completion, lifecycle in zip(completions, lifecycles, strict=True):
                    started = lifecycle.started
                    final_packet = lifecycle.packet
                    if (
                        started is None or final_packet is None
                    ):  # pragma: no cover - invariant guard
                        raise RecallExecutionError("batch recall did not prepare every request")
                    completed_run, persisted_packet = _validated_completion(
                        completion,
                        started=started,
                        packet=final_packet,
                    )
                    results.append(RecallResult(run=completed_run, packet=persisted_packet))
                return tuple(results)
            except Exception as error:
                typed = (
                    error
                    if isinstance(error, RecallError)
                    else RecallExecutionError("batch recall execution failed closed")
                )
                self._reconcile_batch_failure(lifecycles, typed)
                if typed is error:
                    raise
                raise typed from error
        except BaseException as interruption:
            if isinstance(interruption, Exception):
                raise
            self._reconcile_batch_interruption(lifecycles)
            raise

    def replay(self, evidence_packet_id: str) -> RecallResult:
        """Re-run an original request and byte-compare its complete packet."""

        original, request, snapshot = self._prepare_replay(evidence_packet_id)
        return self._execute(
            kind="replay",
            request=request,
            snapshot=snapshot,
            expected_packet=original,
        )

    def verify_replay(self, evidence_packet_id: str) -> EvidencePacket:
        """Recompute and verify a packet without creating any durable run rows.

        This is the read-only verification surface for evaluators and audits.
        Operational ``replay`` remains intentionally audited as a new
        ProcessingRun; callers that merely need to prove reproducibility must
        use this method so verification cannot contend with a live writer or
        alter the store it is measuring.
        """

        original, request, snapshot = self._prepare_replay(evidence_packet_id)
        try:
            packet = self._build_packet(request=request, snapshot=snapshot)
        except Exception as error:
            raise RecallExecutionError("replay verification failed closed") from error
        if (
            packet.packet_hash != original.packet_hash
            or packet.evidence_packet_id != original.evidence_packet_id
            or packet.canonical_bytes != original.canonical_bytes
        ):
            raise RecallReplayMismatchError(
                "replay verification did not reproduce the original EvidencePacket"
            )
        return packet

    def _prepare_replay(
        self,
        evidence_packet_id: str,
    ) -> tuple[EvidencePacket, QueryRequest, CorpusSnapshot]:
        """Load and validate immutable inputs shared by audited and read-only replay."""

        if (
            type(evidence_packet_id) is not str
            or _PACKET_ID_PATTERN.fullmatch(evidence_packet_id) is None
        ):
            raise RecallRequestError("replay requires a canonical EvidencePacket ID")
        try:
            original = self._backend.load_evidence_packet(evidence_packet_id)
        except Exception as error:
            raise RecallPersistenceError("original EvidencePacket could not be loaded") from error
        original = _validated_loaded_packet(original, evidence_packet_id)
        try:
            request = self._backend.load_query_request(original.query_request_id)
        except Exception as error:
            raise RecallPersistenceError("original QueryRequest could not be loaded") from error
        try:
            request = _validated_request(request)
        except RecallRequestError as error:
            raise RecallPersistenceError("original QueryRequest is not canonical") from error
        if (
            request.query_request_id != original.query_request_id
            or request.request_hash != original.query_request_hash
        ):
            raise RecallRequestError("original QueryRequest does not match EvidencePacket")
        snapshot = self._load_and_validate_snapshot(request)
        _validate_packet_request_closure(original, request, snapshot)
        receipt = original.retrieval_receipt
        if (
            receipt.code_version != self._code_version
            or receipt.profile_version != self._profile_version
        ):
            raise RecallDependencyMismatchError(
                "replay requires the original code and FTS runtime profile"
            )
        return original, request, snapshot

    def _load_and_validate_snapshot(self, request: QueryRequest) -> CorpusSnapshot:
        try:
            loaded = self._backend.get_corpus_snapshot(request.corpus_snapshot_id)
        except Exception as error:
            raise RecallSnapshotError("CorpusSnapshot could not be loaded") from error
        try:
            snapshot = _validated_snapshot(loaded)
        except RecallSnapshotError:
            raise
        except Exception as error:
            raise RecallSnapshotError("CorpusSnapshot is not canonical") from error
        if snapshot.corpus_snapshot_id != request.corpus_snapshot_id:
            raise RecallSnapshotError("CorpusSnapshot identity differs from QueryRequest")
        if snapshot.library_id != request.library_id:
            raise RecallSnapshotError("CorpusSnapshot belongs to a different Library")
        if snapshot.collection_ids != request.collection_ids:
            raise RecallSnapshotError("CorpusSnapshot Collection scope differs from QueryRequest")
        return snapshot

    def _persist_batch_requests(self, requests: tuple[QueryRequest, ...]) -> None:
        try:
            for request in requests:
                self._backend.persist_query_request(request)
            for request in requests:
                persisted = self._backend.load_query_request(request.query_request_id)
                _require_same_request(persisted, request)
        except RecallPersistenceError:
            raise
        except Exception as error:
            raise RecallPersistenceError(
                "batch QueryRequests could not be persisted and reloaded"
            ) from error

    def _execute(
        self,
        *,
        kind: RecallRunKind,
        request: QueryRequest,
        snapshot: CorpusSnapshot,
        expected_packet: EvidencePacket | None,
        scope_session: RecallScopeSession | None = None,
    ) -> RecallResult:
        lifecycle = _RecallLifecycle(
            processing_run_id=new_id("run"),
            kind=kind,
        )
        try:
            try:
                started = self._begin_run(
                    processing_run_id=lifecycle.processing_run_id,
                    kind=kind,
                    request=request,
                )
                lifecycle.started = started
                lifecycle.packet = self._build_packet(
                    request=request,
                    snapshot=snapshot,
                    scope_session=scope_session,
                )
                packet = lifecycle.packet
                if expected_packet is not None and (
                    packet.packet_hash != expected_packet.packet_hash
                    or packet.evidence_packet_id != expected_packet.evidence_packet_id
                    or packet.canonical_bytes != expected_packet.canonical_bytes
                ):
                    raise RecallReplayMismatchError(
                        "replay did not reproduce the original EvidencePacket"
                    )
                try:
                    if scope_session is not None and isinstance(
                        self._backend, _ScopedRecallCompletionBackend
                    ):
                        completion = self._backend.complete_scoped_recall_run(
                            processing_run_id=lifecycle.processing_run_id,
                            packet=packet,
                            scope_capability=scope_session._completion_capability,
                        )
                    else:
                        completion = self._backend.complete_recall_run(
                            processing_run_id=lifecycle.processing_run_id,
                            packet=packet,
                        )
                except Exception as error:
                    raise RecallPersistenceError(
                        "recall artifacts could not be atomically persisted"
                    ) from error
                completed_run, persisted_packet = _validated_completion(
                    completion,
                    started=started,
                    packet=packet,
                )
                return RecallResult(run=completed_run, packet=persisted_packet)
            except Exception as error:
                typed = (
                    error
                    if isinstance(error, RecallError)
                    else RecallExecutionError("recall execution failed closed")
                )
                recovered = self._reconcile_ordinary_failure(lifecycle, typed)
                if recovered is not None:
                    return recovered
                if typed is error:
                    raise
                raise typed from error
        except BaseException as interruption:
            # This outer guard also catches an interruption raised *inside* the
            # ordinary-error handler while it is recording the failed run.
            # Ordinary exceptions deliberately pass through unchanged.
            if isinstance(interruption, Exception):
                raise
            self._reconcile_interruption(lifecycle)
            raise

    def _begin_run(
        self,
        *,
        processing_run_id: str,
        kind: RecallRunKind,
        request: QueryRequest,
    ) -> ProcessingRunRecord:
        try:
            run = self._backend.begin_recall_run(
                processing_run_id=processing_run_id,
                kind=kind,
                query_request_id=request.query_request_id,
                code_version=self._code_version,
                profile_version=self._profile_version,
            )
        except Exception as error:
            raise RecallPersistenceError("recall ProcessingRun could not be started") from error
        try:
            _validate_started_run(
                run,
                processing_run_id=processing_run_id,
                kind=kind,
                code_version=self._code_version,
                profile_version=self._profile_version,
            )
        except Exception as error:
            raise RecallPersistenceError(
                "backend returned an invalid running ProcessingRun"
            ) from error
        return run

    def _build_packet(
        self,
        *,
        request: QueryRequest,
        snapshot: CorpusSnapshot,
        scope_session: RecallScopeSession | None = None,
    ) -> EvidencePacket:
        if scope_session is None:
            prepared = self._prepare_recall_corpus(request=request, snapshot=snapshot)
            result = self._fts_search(
                question=request.question,
                fragments=prepared.fts_fragments,
                max_candidates=request.retrieval.max_candidates,
            )
        else:
            scope_session._require_request(self, request)
            if snapshot != scope_session._snapshot:
                raise RecallSnapshotError("RecallScopeSession snapshot differs from QueryRequest")
            prepared = scope_session._prepared
            result = scope_session._search(request)
        result = _validated_fts_result(
            result,
            fragments=prepared.fts_fragments,
            profile_version=self._profile_version,
            max_candidates=request.retrieval.max_candidates,
        )
        return self._assemble_packet(
            request=request,
            snapshot=snapshot,
            prepared=prepared,
            result=result,
        )

    def _prepare_recall_corpus(
        self,
        *,
        request: QueryRequest,
        snapshot: CorpusSnapshot,
    ) -> _PreparedRecallCorpus:
        scope = RequestScope(
            library_id=request.library_id,
            snapshot_hash=snapshot.manifest_hash,
            purpose=request.purpose,
            collection_ids=request.collection_ids,
            exclusions=request.exclusions,
        )
        authorization = self._backend.authorize_read(
            access_policy_id=request.access_policy_id,
            scope=scope,
        )
        manifest = _validated_authorization(
            authorization,
            request=request,
            snapshot=snapshot,
            expected_scope=scope,
        )
        read_fragments = self._backend.read_permitted_fragments(authorization)
        fragments_by_id, addresses, fts_fragments = _validated_read_fragments(
            read_fragments,
            manifest=manifest,
        )
        return _PreparedRecallCorpus(
            authorization=authorization,
            manifest=manifest,
            read_fragments=read_fragments,
            read_receipt_items=tuple(
                ReadReceiptItem(
                    source_fragment_id=fragment.source_fragment_id,
                    source_version_id=fragment.source_version_id,
                    read_order=order,
                    text_sha256=fragment.text_sha256,
                )
                for order, fragment in enumerate(read_fragments)
            ),
            fragments_by_id=fragments_by_id,
            addresses=addresses,
            fts_fragments=fts_fragments,
        )

    def _assemble_packet(
        self,
        *,
        request: QueryRequest,
        snapshot: CorpusSnapshot,
        prepared: _PreparedRecallCorpus,
        result: FtsSearchResult,
    ) -> EvidencePacket:
        selected_count = min(len(result.trace), request.retrieval.max_source_fragments)
        selected_trace = result.trace[:selected_count]

        read_receipt = ReadReceipt(
            query_request_hash=request.request_hash,
            retrieval_corpus_hash=result.retrieval_corpus_hash,
            items=prepared.read_receipt_items,
        )
        token = prepared.authorization.compiled.token
        public_result = prepared.authorization.compiled.public_result
        access_receipt = AccessReceipt(
            library_id=request.library_id,
            query_request_hash=request.request_hash,
            access_policy_id=request.access_policy_id,
            policy_hash=token.policy_hash,
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            snapshot_hash=snapshot.manifest_hash,
            exclusion_hash=token.exclusion_hash,
            permitted_set_hash=token.permitted_set_hash,
            retrieval_corpus_hash=result.retrieval_corpus_hash,
            policy_omission_present=public_result.policy_omission_present,
        )
        retrieval_receipt = RetrievalReceipt(
            query_request_hash=request.request_hash,
            profile="fts_v1",
            profile_version=result.profile_version,
            code_version=self._code_version,
            retrieval_corpus_hash=result.retrieval_corpus_hash,
            max_candidates=request.retrieval.max_candidates,
            max_source_fragments=request.retrieval.max_source_fragments,
            candidate_count=result.candidate_count,
            selected_count=selected_count,
            trace=tuple(
                RetrievalTraceItem(
                    source_fragment_id=item.source_fragment_id,
                    rank=item.rank,
                    score=item.score,
                )
                for item in result.trace
            ),
        )
        coverage = RecallCoverageReport(
            query_request_hash=request.request_hash,
            processed_count=len({item.source_version_id for item in prepared.read_fragments}),
            skipped_count=0,
            failed_count=0,
            policy_omission_present=public_result.policy_omission_present,
            omissions=_build_omissions(
                request=request,
                candidate_count=result.candidate_count,
                selected_count=selected_count,
                policy_omission_present=public_result.policy_omission_present,
            ),
        )
        manifest_by_id = {item.source_fragment_id: item for item in prepared.manifest.items}
        evidence = tuple(
            _evidence_fragment(
                candidate,
                source=prepared.fragments_by_id[candidate.source_fragment_id],
                source_family_id=manifest_by_id[candidate.source_fragment_id].source_family_id,
                source_address=prepared.addresses[candidate.source_fragment_id],
            )
            for candidate in selected_trace
        )
        return EvidencePacket(
            query_request_id=request.query_request_id,
            query_request_hash=request.request_hash,
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            corpus_snapshot_hash=snapshot.manifest_hash,
            result_status=(
                PacketResultStatus.EVIDENCE_FOUND if evidence else PacketResultStatus.NO_EVIDENCE
            ),
            source_fragments=evidence,
            coverage_report=coverage,
            read_receipt=read_receipt,
            access_receipt=access_receipt,
            retrieval_receipt=retrieval_receipt,
        )

    def _reconcile_ordinary_failure(
        self,
        lifecycle: _RecallLifecycle,
        error: RecallError,
    ) -> RecallResult | None:
        """Settle an ordinary failure against the durable lifecycle truth."""

        current = self._reload_lifecycle(
            lifecycle,
            allow_missing=lifecycle.started is None,
        )
        if current is None:
            return None
        return self._settle_failed_lifecycle(
            lifecycle,
            current=current,
            error=error,
            re_raise_transition_interruption=True,
        )

    def _reconcile_batch_failure(
        self,
        lifecycles: tuple[_RecallLifecycle, ...],
        error: RecallError,
    ) -> None:
        """Terminalize every started batch run and never return partial results."""

        first_failure: BaseException | None = None
        for lifecycle in lifecycles:
            try:
                self._reconcile_ordinary_failure(lifecycle, error)
            except BaseException as reconcile_error:
                if first_failure is None:
                    first_failure = reconcile_error
        if first_failure is not None:
            if not isinstance(first_failure, Exception):
                raise first_failure
            raise RecallPersistenceError(
                "batch recall failure could not terminalize every started run"
            ) from first_failure

    def _reconcile_batch_interruption(
        self,
        lifecycles: tuple[_RecallLifecycle, ...],
    ) -> None:
        """Best-effort terminalization for every batch run after interruption."""

        first_failure: BaseException | None = None
        for lifecycle in lifecycles:
            try:
                self._reconcile_interruption(lifecycle)
            except BaseException as reconcile_error:
                if first_failure is None:
                    first_failure = reconcile_error
        if first_failure is not None:
            raise first_failure

    def _reconcile_interruption(self, lifecycle: _RecallLifecycle) -> None:
        """Terminalize a known running row, preserving any committed terminal state."""

        current = self._reload_lifecycle(
            lifecycle,
            allow_missing=lifecycle.started is None,
        )
        if current is None:
            return
        self._settle_failed_lifecycle(
            lifecycle,
            current=current,
            error=RecallInterruptedError("recall execution was interrupted"),
            re_raise_transition_interruption=False,
        )

    def _settle_failed_lifecycle(
        self,
        lifecycle: _RecallLifecycle,
        *,
        current: ProcessingRunRecord,
        error: RecallError,
        re_raise_transition_interruption: bool,
    ) -> RecallResult | None:
        if current.status is ProcessingRunStatus.SUCCEEDED:
            return self._recover_committed_success(lifecycle, current)
        if current.status is ProcessingRunStatus.FAILED:
            return None
        return self._record_failed_run(
            lifecycle,
            error,
            re_raise_transition_interruption=re_raise_transition_interruption,
        )

    def _record_failed_run(
        self,
        lifecycle: _RecallLifecycle,
        error: RecallError,
        *,
        re_raise_transition_interruption: bool,
    ) -> RecallResult | None:
        """Record failure, reconciling commit ambiguity and terminal races."""

        started = lifecycle.started
        if started is None:
            raise RecallPersistenceError("recall ProcessingRun start could not be recovered")
        error_hash = canonical_sha256_hex(
            {
                "schema": "dithyramba.recall_failure/1.0",
                "error_code": error.code,
                "error_type": type(error).__name__,
            }
        )
        pending_interruption: BaseException | None = None
        last_error: BaseException | None = None
        for _attempt in range(3):
            try:
                failed = self._backend.fail_recall_run(
                    processing_run_id=started.processing_run_id,
                    error_code=error.code,
                    error_hash=error_hash,
                )
                _validate_failed_run(
                    failed,
                    started=started,
                    error_code=error.code,
                    error_hash=error_hash,
                )
            except BaseException as failure_error:
                last_error = failure_error
                if not isinstance(failure_error, Exception) and pending_interruption is None:
                    pending_interruption = failure_error
                current = self._reload_lifecycle(lifecycle, allow_missing=False)
                if current is None:  # pragma: no cover - forbidden by allow_missing=False
                    raise AssertionError(
                        "known ProcessingRun unexpectedly disappeared"
                    ) from failure_error
                if current.status is ProcessingRunStatus.SUCCEEDED:
                    recovered = self._recover_committed_success(lifecycle, current)
                    if pending_interruption is not None and re_raise_transition_interruption:
                        raise pending_interruption from failure_error
                    return recovered
                if current.status is ProcessingRunStatus.FAILED:
                    if pending_interruption is not None and re_raise_transition_interruption:
                        raise pending_interruption from failure_error
                    return None
                continue
            if pending_interruption is not None and re_raise_transition_interruption:
                raise pending_interruption
            return None
        raise RecallPersistenceError(
            "recall failed and its failure ProcessingRun could not be recorded"
        ) from last_error

    def _reload_lifecycle(
        self,
        lifecycle: _RecallLifecycle,
        *,
        allow_missing: bool,
    ) -> ProcessingRunRecord | None:
        """Reload and validate one service-owned run after an ambiguous operation."""

        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                current = self._backend.get_processing_run(lifecycle.processing_run_id)
            except ProcessingRunNotFoundError:
                if allow_missing:
                    return None
                raise RecallPersistenceError("known recall ProcessingRun disappeared") from None
            except Exception as error:
                last_error = error
                continue
            break
        else:
            raise RecallPersistenceError(
                "recall ProcessingRun could not be reloaded"
            ) from last_error
        started = _validated_lifecycle_run(
            current,
            processing_run_id=lifecycle.processing_run_id,
            kind=lifecycle.kind,
            code_version=self._code_version,
            profile_version=self._profile_version,
            started=lifecycle.started,
        )
        if lifecycle.started is None:
            lifecycle.started = started
        return current

    def _recover_committed_success(
        self,
        lifecycle: _RecallLifecycle,
        current: ProcessingRunRecord,
    ) -> RecallResult:
        """Validate and return the exact packet of an ambiguously committed success."""

        started = lifecycle.started
        packet = lifecycle.packet
        if started is None or packet is None:
            raise RecallPersistenceError(
                "terminal recall success cannot be reconciled without its expected packet"
            )
        if current.output_hash != packet.packet_hash:
            raise RecallPersistenceError(
                "terminal recall success output differs from the expected packet"
            )
        try:
            persisted = self._backend.load_evidence_packet(packet.evidence_packet_id)
        except Exception as error:
            raise RecallPersistenceError(
                "committed EvidencePacket could not be reloaded"
            ) from error
        completed_run, persisted_packet = _validated_completion(
            (current, persisted),
            started=started,
            packet=packet,
        )
        return RecallResult(run=completed_run, packet=persisted_packet)


def _validated_request(value: QueryRequest) -> QueryRequest:
    if type(value) is not QueryRequest:
        raise RecallRequestError("recall requires an exact QueryRequest")
    try:
        canonical = QueryRequest(
            question=value.question,
            library_id=value.library_id,
            collection_ids=value.collection_ids,
            corpus_snapshot_id=value.corpus_snapshot_id,
            access_policy_id=value.access_policy_id,
            purpose=value.purpose,
            exclusions=value.exclusions,
            retrieval=value.retrieval,
            result_contract=value.result_contract,
        )
        if (
            canonical.canonical_bytes != value.canonical_bytes
            or canonical.request_hash != value.request_hash
            or canonical.query_request_id != value.query_request_id
        ):
            raise RecallRequestError("QueryRequest identity is not canonical")
    except RecallRequestError:
        raise
    except Exception as error:
        raise RecallRequestError("QueryRequest is not canonical") from error
    return canonical


def _validated_batch_requests(
    values: tuple[QueryRequest, ...],
) -> tuple[QueryRequest, ...]:
    if type(values) is not tuple or not values:
        raise RecallRequestError("batch recall requires a non-empty QueryRequest tuple")
    canonical = tuple(_validated_request(value) for value in values)
    identifiers = tuple(request.query_request_id for request in canonical)
    if len(set(identifiers)) != len(identifiers):
        raise RecallRequestError("batch recall requires unique QueryRequests")
    expected_scope = _batch_scope_bytes(canonical[0])
    if any(_batch_scope_bytes(request) != expected_scope for request in canonical[1:]):
        raise RecallRequestError("batch recall requires one exact shared request scope")
    return canonical


def _batch_scope_bytes(request: QueryRequest) -> bytes:
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


def _session_scope_bytes(request: QueryRequest) -> bytes:
    """Hash domain for reusable authorized text; retrieval budgets stay per query."""

    return canonical_json_bytes(
        {
            "schema": "dithyramba.recall_scope_session/1.0",
            "library_id": request.library_id,
            "collection_ids": list(request.collection_ids),
            "corpus_snapshot_id": request.corpus_snapshot_id,
            "access_policy_id": request.access_policy_id,
            "purpose": request.purpose,
            "exclusions": request.exclusions.payload(),
            "result_contract": request.result_contract.payload(),
        }
    )


def _require_same_request(value: QueryRequest, expected: QueryRequest) -> None:
    actual = _validated_request(value)
    if (
        actual.query_request_id != expected.query_request_id
        or actual.request_hash != expected.request_hash
        or actual.canonical_bytes != expected.canonical_bytes
    ):
        raise RecallPersistenceError("persisted QueryRequest differs from the submitted request")


def _validated_snapshot(value: CorpusSnapshot) -> CorpusSnapshot:
    if type(value) is not CorpusSnapshot:
        raise RecallSnapshotError("backend must return an exact CorpusSnapshot")
    canonical = CorpusSnapshot(
        library_id=value.library_id,
        collection_ids=value.collection_ids,
        members=value.members,
    )
    if (
        canonical.canonical_bytes != value.canonical_bytes
        or canonical.manifest_hash != value.manifest_hash
        or canonical.corpus_snapshot_id != value.corpus_snapshot_id
    ):
        raise RecallSnapshotError("CorpusSnapshot identity is not canonical")
    return canonical


def _validated_authorization(
    authorization: AuthorizedRead,
    *,
    request: QueryRequest,
    snapshot: CorpusSnapshot,
    expected_scope: RequestScope,
) -> PermittedManifest:
    if type(authorization) is not AuthorizedRead:
        raise RecallExecutionError("backend must return an exact AuthorizedRead")
    if (
        type(authorization.authorization_id) is not str
        or _AUTHORIZATION_ID_PATTERN.fullmatch(authorization.authorization_id) is None
        or type(authorization.repository_instance_id) is not str
        or _REPOSITORY_ID_PATTERN.fullmatch(authorization.repository_instance_id) is None
    ):
        raise RecallExecutionError("AuthorizedRead identity is invalid")
    if authorization.access_policy_id != request.access_policy_id:
        raise RecallExecutionError("AuthorizedRead uses a different AccessPolicy")
    if type(authorization.scope) is not RequestScope or authorization.scope != expected_scope:
        raise RecallExecutionError("AuthorizedRead scope differs from QueryRequest")
    compiled = authorization.compiled
    if type(compiled) is not CompiledAccess:
        raise RecallExecutionError("AuthorizedRead must contain an exact CompiledAccess")
    manifest = compiled.manifest
    token = compiled.token
    public_result = compiled.public_result
    if (
        type(manifest) is not PermittedManifest
        or type(token) is not PermittedToken
        or type(public_result) is not PublicAccessResult
        or any(type(item) is not PermittedManifestItem for item in manifest.items)
    ):
        raise RecallExecutionError("CompiledAccess contains a non-canonical access value")
    if (
        manifest.library_id != request.library_id
        or token.library_id != request.library_id
        or manifest.snapshot_hash != snapshot.manifest_hash
        or token.snapshot_hash != snapshot.manifest_hash
        or token.exclusion_hash != expected_scope.exclusion_hash
        or token.permitted_set_hash != manifest.permitted_set_hash
    ):
        raise RecallExecutionError("CompiledAccess tuple differs from the exact request scope")
    _validate_manifest_snapshot_lineage(manifest, request=request, snapshot=snapshot)
    return manifest


def _validate_manifest_snapshot_lineage(
    manifest: PermittedManifest,
    *,
    request: QueryRequest,
    snapshot: CorpusSnapshot,
) -> None:
    requested = set(request.collection_ids)
    by_version: dict[str, list[SnapshotMember]] = {}
    for member in snapshot.members:
        if member.collection_id in requested:
            by_version.setdefault(member.source_version_id, []).append(member)
    for item in manifest.items:
        members = by_version.get(item.source_version_id, [])
        if not members:
            raise RecallExecutionError("permitted fragment SourceVersion is absent from snapshot")
        if any(
            member.source_id != item.source_id or member.source_family_id != item.source_family_id
            for member in members
        ):
            raise RecallExecutionError("permitted fragment lineage differs from snapshot")
        expected_collections = tuple(sorted(member.collection_id for member in members))
        if item.collection_ids != expected_collections:
            raise RecallExecutionError("permitted fragment Collections differ from snapshot scope")
        if any(member.membership_state is not MembershipState.ACTIVE for member in members):
            raise RecallExecutionError("holdout or excluded membership crossed the access boundary")


def _validated_read_fragments(
    fragments: tuple[SourceFragmentText, ...],
    *,
    manifest: PermittedManifest,
) -> tuple[
    dict[str, SourceFragmentText],
    dict[str, dict[str, object]],
    tuple[FtsFragment, ...],
]:
    if type(fragments) is not tuple or any(
        type(item) is not SourceFragmentText for item in fragments
    ):
        raise RecallExecutionError("protected read must return exact SourceFragmentText values")
    expected_ids = tuple(item.source_fragment_id for item in manifest.items)
    actual_ids = tuple(item.source_fragment_id for item in fragments)
    if actual_ids != expected_ids:
        raise RecallExecutionError("protected read differs from the exact permitted manifest")
    manifest_by_id = {item.source_fragment_id: item for item in manifest.items}
    by_id: dict[str, SourceFragmentText] = {}
    addresses: dict[str, dict[str, object]] = {}
    fts: list[FtsFragment] = []
    for fragment in fragments:
        item = manifest_by_id[fragment.source_fragment_id]
        if (
            fragment.source_version_id != item.source_version_id
            or fragment.source_id != item.source_id
        ):
            raise RecallExecutionError("protected fragment lineage differs from manifest")
        if (
            type(fragment.ordinal) is not int
            or fragment.ordinal < 0
            or type(fragment.fragment_kind) is not str
            or not fragment.fragment_kind
        ):
            raise RecallExecutionError("protected fragment metadata is invalid")
        address = _load_canonical_address(fragment.source_address_json)
        try:
            fts_fragment = FtsFragment(
                source_fragment_id=fragment.source_fragment_id,
                text=fragment.text,
                text_sha256=fragment.text_sha256,
            )
        except Exception as error:
            raise RecallExecutionError("protected fragment text is not canonical") from error
        by_id[fragment.source_fragment_id] = fragment
        addresses[fragment.source_fragment_id] = address
        fts.append(fts_fragment)
    return by_id, addresses, tuple(fts)


def _load_canonical_address(value: str) -> dict[str, object]:
    if type(value) is not str:
        raise RecallExecutionError("SourceAddress must be canonical JSON text")
    try:
        decoded: object = json.loads(value, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise RecallExecutionError("SourceAddress is not valid unique-key JSON") from error
    if type(decoded) is not dict:
        raise RecallExecutionError("SourceAddress must be a JSON object")
    result = cast(dict[str, object], decoded)
    try:
        if canonical_json_bytes(result).decode("utf-8") != value:
            raise RecallExecutionError("SourceAddress JSON is not canonical")
    except RecallExecutionError:
        raise
    except Exception as error:
        raise RecallExecutionError("SourceAddress JSON is not canonical") from error
    return result


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _validated_fts_result(
    result: FtsSearchResult,
    *,
    fragments: tuple[FtsFragment, ...],
    profile_version: str,
    max_candidates: int,
) -> FtsSearchResult:
    if (
        type(result) is not FtsSearchResult
        or type(result.profile) is not FtsRuntimeProfile
        or any(type(item) is not FtsCandidate for item in result.trace)
    ):
        raise RecallExecutionError("fts_search returned a non-canonical result type")
    expected_corpus_hash = canonical_sha256_hex(
        {
            "schema": "dithyramba.fts_corpus/1.0",
            "fragments": [
                fragment.identity_payload()
                for fragment in sorted(fragments, key=lambda item: item.source_fragment_id)
            ],
        }
    )
    read_ids = {item.source_fragment_id for item in fragments}
    if (
        result.profile_version != profile_version
        or result.max_candidates != max_candidates
        or result.retrieval_corpus_hash != expected_corpus_hash
        or result.candidate_count > len(fragments)
        or len(result.trace) != min(result.candidate_count, max_candidates)
        or any(item.source_fragment_id not in read_ids for item in result.trace)
    ):
        raise RecallExecutionError("fts_search result differs from its injected dependencies")
    return result


def _build_omissions(
    *,
    request: QueryRequest,
    candidate_count: int,
    selected_count: int,
    policy_omission_present: bool,
) -> tuple[RecallOmission, ...]:
    omissions: list[RecallOmission] = []
    explicit_count = (
        len(request.exclusions.source_ids)
        + len(request.exclusions.source_family_ids)
        + len(request.exclusions.source_fragment_ids)
    )
    if explicit_count:
        omissions.append(
            RecallOmission(
                category=OmissionCategory.EXPLICIT_EXCLUSION,
                disclosure=OmissionDisclosure.COUNTED,
                count=explicit_count,
                reason_code="caller_exclusions",
            )
        )
    budget_count = candidate_count - selected_count
    if budget_count:
        omissions.append(
            RecallOmission(
                category=OmissionCategory.BUDGET,
                disclosure=OmissionDisclosure.COUNTED,
                count=budget_count,
                reason_code="candidate_budget",
            )
        )
    if policy_omission_present:
        omissions.append(
            RecallOmission(
                category=OmissionCategory.POLICY,
                disclosure=OmissionDisclosure.REDACTED,
                count=None,
                reason_code="policy_omission",
            )
        )
    return tuple(omissions)


def _evidence_fragment(
    candidate: FtsCandidate,
    *,
    source: SourceFragmentText,
    source_family_id: str | None,
    source_address: dict[str, object],
) -> EvidenceFragment:
    try:
        return EvidenceFragment.model_validate(
            {
                "source_fragment_id": source.source_fragment_id,
                "source_version_id": source.source_version_id,
                "source_family_id": source_family_id,
                "rank": candidate.rank,
                "score": candidate.score,
                "text": source.text,
                "text_sha256": source.text_sha256,
                "source_address": source_address,
            }
        )
    except Exception as error:
        raise RecallExecutionError("selected EvidenceFragment is not canonical") from error


def _validate_started_run(
    run: ProcessingRunRecord,
    *,
    processing_run_id: str,
    kind: RecallRunKind,
    code_version: str,
    profile_version: str,
) -> None:
    if type(run) is not ProcessingRunRecord:
        raise TypeError("run must be an exact ProcessingRunRecord")
    if (
        type(run.processing_run_id) is not str
        or _RUN_ID_PATTERN.fullmatch(run.processing_run_id) is None
        or run.processing_run_id != processing_run_id
        or run.kind != kind
        or run.status is not ProcessingRunStatus.RUNNING
        or run.code_version != code_version
        or run.profile_version != profile_version
        or type(run.started_at) is not str
        or not run.started_at
        or run.finished_at is not None
        or run.error_code is not None
        or run.output_hash is not None
    ):
        raise ValueError("running ProcessingRun fields are inconsistent")


def _validated_lifecycle_run(
    run: ProcessingRunRecord,
    *,
    processing_run_id: str,
    kind: RecallRunKind,
    code_version: str,
    profile_version: str,
    started: ProcessingRunRecord | None,
) -> ProcessingRunRecord:
    """Validate a reloaded state and return its immutable running projection."""

    if type(run) is not ProcessingRunRecord or type(run.status) is not ProcessingRunStatus:
        raise RecallPersistenceError("reloaded ProcessingRun is not canonical")
    projection = ProcessingRunRecord(
        processing_run_id=run.processing_run_id,
        kind=run.kind,
        status=ProcessingRunStatus.RUNNING,
        code_version=run.code_version,
        profile_version=run.profile_version,
        started_at=run.started_at,
        finished_at=None,
        error_code=None,
        output_hash=None,
    )
    try:
        _validate_started_run(
            projection,
            processing_run_id=processing_run_id,
            kind=kind,
            code_version=code_version,
            profile_version=profile_version,
        )
    except Exception as error:
        raise RecallPersistenceError("reloaded ProcessingRun identity changed") from error
    if started is not None and projection != started:
        raise RecallPersistenceError("reloaded ProcessingRun start fields changed")
    if run.status is ProcessingRunStatus.RUNNING:
        if run != projection:
            raise RecallPersistenceError("running ProcessingRun has terminal fields")
    elif run.status is ProcessingRunStatus.SUCCEEDED:
        if (
            type(run.finished_at) is not str
            or not run.finished_at
            or run.error_code is not None
            or type(run.output_hash) is not str
            or _SHA256_PATTERN.fullmatch(run.output_hash) is None
        ):
            raise RecallPersistenceError("successful ProcessingRun fields are inconsistent")
    elif run.status is ProcessingRunStatus.FAILED:
        if (
            type(run.finished_at) is not str
            or not run.finished_at
            or type(run.error_code) is not str
            or not run.error_code
            or type(run.output_hash) is not str
            or _SHA256_PATTERN.fullmatch(run.output_hash) is None
        ):
            raise RecallPersistenceError("failed ProcessingRun fields are inconsistent")
    else:  # pragma: no cover - exact enum check makes this defensive
        raise RecallPersistenceError("ProcessingRun status is invalid")
    return projection


def _validated_completion(
    completion: tuple[ProcessingRunRecord, EvidencePacket],
    *,
    started: ProcessingRunRecord,
    packet: EvidencePacket,
) -> tuple[ProcessingRunRecord, EvidencePacket]:
    if type(completion) is not tuple or len(completion) != 2:
        raise RecallPersistenceError("completion must return a run/packet tuple")
    run, persisted = completion
    if type(run) is not ProcessingRunRecord or type(persisted) is not EvidencePacket:
        raise RecallPersistenceError("completion returned non-canonical artifact types")
    if (
        run.processing_run_id != started.processing_run_id
        or run.kind != started.kind
        or run.status is not ProcessingRunStatus.SUCCEEDED
        or run.code_version != started.code_version
        or run.profile_version != started.profile_version
        or run.started_at != started.started_at
        or type(run.finished_at) is not str
        or not run.finished_at
        or run.error_code is not None
        or run.output_hash != packet.packet_hash
    ):
        raise RecallPersistenceError("completed ProcessingRun is inconsistent")
    _validated_loaded_packet(persisted, packet.evidence_packet_id)
    if (
        persisted.packet_hash != packet.packet_hash
        or persisted.canonical_bytes != packet.canonical_bytes
    ):
        raise RecallPersistenceError("persisted EvidencePacket differs from service output")
    return run, persisted


def _validate_failed_run(
    failed: ProcessingRunRecord,
    *,
    started: ProcessingRunRecord,
    error_code: str,
    error_hash: str,
) -> None:
    if type(failed) is not ProcessingRunRecord:
        raise TypeError("failed run must be an exact ProcessingRunRecord")
    if (
        failed.processing_run_id != started.processing_run_id
        or failed.kind != started.kind
        or failed.status is not ProcessingRunStatus.FAILED
        or failed.code_version != started.code_version
        or failed.profile_version != started.profile_version
        or failed.started_at != started.started_at
        or type(failed.finished_at) is not str
        or not failed.finished_at
        or failed.error_code != error_code
        or failed.output_hash != error_hash
    ):
        raise ValueError("failed ProcessingRun is inconsistent")


def _validated_loaded_packet(value: EvidencePacket, expected_id: str) -> EvidencePacket:
    if type(value) is not EvidencePacket:
        raise RecallPersistenceError("backend must return an exact EvidencePacket")
    try:
        reconstructed = EvidencePacket(
            query_request_id=value.query_request_id,
            query_request_hash=value.query_request_hash,
            corpus_snapshot_id=value.corpus_snapshot_id,
            corpus_snapshot_hash=value.corpus_snapshot_hash,
            result_status=value.result_status,
            source_fragments=value.source_fragments,
            coverage_report=value.coverage_report,
            read_receipt=value.read_receipt,
            access_receipt=value.access_receipt,
            retrieval_receipt=value.retrieval_receipt,
        )
        if (
            reconstructed.evidence_packet_id != expected_id
            or value.evidence_packet_id != expected_id
            or reconstructed.packet_hash != value.packet_hash
            or reconstructed.canonical_bytes != value.canonical_bytes
        ):
            raise RecallPersistenceError("EvidencePacket identity is not canonical")
    except RecallPersistenceError:
        raise
    except Exception as error:
        raise RecallPersistenceError("EvidencePacket is not canonical") from error
    return reconstructed


def _validate_packet_request_closure(
    packet: EvidencePacket,
    request: QueryRequest,
    snapshot: CorpusSnapshot,
) -> None:
    if (
        packet.query_request_id != request.query_request_id
        or packet.query_request_hash != request.request_hash
        or packet.corpus_snapshot_id != snapshot.corpus_snapshot_id
        or packet.corpus_snapshot_hash != snapshot.manifest_hash
        or packet.access_receipt.library_id != request.library_id
        or packet.access_receipt.access_policy_id != request.access_policy_id
        or packet.access_receipt.exclusion_hash
        != RequestScope(
            library_id=request.library_id,
            snapshot_hash=snapshot.manifest_hash,
            purpose=request.purpose,
            collection_ids=request.collection_ids,
            exclusions=request.exclusions,
        ).exclusion_hash
        or packet.retrieval_receipt.max_candidates != request.retrieval.max_candidates
        or packet.retrieval_receipt.max_source_fragments != request.retrieval.max_source_fragments
    ):
        raise RecallRequestError("EvidencePacket does not close over its QueryRequest")


def _validated_version(value: str, label: str) -> str:
    if type(value) is not str or _VERSION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical reproducibility version")
    return value


__all__ = [
    "FtsSearch",
    "RecallBackend",
    "RecallDependencyMismatchError",
    "RecallError",
    "RecallExecutionError",
    "RecallPersistenceError",
    "RecallReplayMismatchError",
    "RecallRequestError",
    "RecallResult",
    "RecallRunKind",
    "RecallService",
    "RecallSnapshotError",
]
