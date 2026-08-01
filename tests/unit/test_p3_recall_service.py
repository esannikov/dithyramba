from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, NoReturn, cast

import pytest

import dithyramba.recall.service as recall_service_module
from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    CompiledAccess,
    MembershipState,
    PermittedManifest,
    PermittedManifestItem,
    PermittedToken,
    PolicyEffect,
    QueryExclusions,
    RequestScope,
    SnapshotCandidate,
    SnapshotMembership,
    SourceRule,
    compile_access,
)
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.ingest import MarkdownSourceAddress
from dithyramba.persistence.errors import ProcessingRunNotFoundError
from dithyramba.persistence.models import AuthorizedRead, SourceFragmentText
from dithyramba.provenance import ProcessingRunRecord, ProcessingRunStatus, SourceFamilyRole
from dithyramba.recall import (
    EvidencePacket,
    FtsCandidate,
    FtsFragment,
    FtsRuntimeProfile,
    FtsSearchResult,
    OmissionCategory,
    PacketResultStatus,
    PermittedFtsSession,
    QueryRequest,
    RetrievalBudget,
    current_fts_runtime_profile,
)
from dithyramba.recall.service import (
    RecallDependencyMismatchError,
    RecallExecutionError,
    RecallInterruptedError,
    RecallPersistenceError,
    RecallReplayMismatchError,
    RecallRequestError,
    RecallResult,
    RecallRunKind,
    RecallService,
    RecallSnapshotError,
)
from dithyramba.snapshots import CorpusSnapshot, SnapshotMember

LIBRARY_ID = "library_research"
COLLECTION_ID = "collection_phd"
POLICY_ID = "policy_research"
CODE_VERSION = "0.1.0.dev2"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    text: str
    state: MembershipState = MembershipState.ACTIVE

    @property
    def source_id(self) -> str:
        return f"source_{self.name}"

    @property
    def version_id(self) -> str:
        return f"source_version_{self.name}"

    @property
    def family_id(self) -> str:
        return f"family_{self.name}"

    @property
    def fragment_id(self) -> str:
        return f"fragment_{self.name}"


class FakeFts:
    def __init__(
        self,
        *,
        trace_ids: tuple[str, ...] | None = None,
        candidate_count: int | None = None,
        profile: FtsRuntimeProfile | None = None,
    ) -> None:
        self.trace_ids = trace_ids
        self.candidate_count = candidate_count
        self.profile = profile or FtsRuntimeProfile(
            sqlite_version="3.45.1",
            compile_options_hash="a" * 64,
        )
        self.calls: list[tuple[str, tuple[FtsFragment, ...], int]] = []

    def __call__(
        self,
        *,
        question: str,
        fragments: tuple[FtsFragment, ...],
        max_candidates: int,
    ) -> FtsSearchResult:
        self.calls.append((question, fragments, max_candidates))
        available = {fragment.source_fragment_id for fragment in fragments}
        trace_ids = (
            tuple(fragment.source_fragment_id for fragment in fragments)
            if self.trace_ids is None
            else self.trace_ids
        )
        if any(fragment_id not in available for fragment_id in trace_ids):
            raise AssertionError("test FTS trace references a fragment it did not receive")
        trace = tuple(
            FtsCandidate(
                source_fragment_id=fragment_id,
                rank=rank,
                score=f"{-len(trace_ids) + rank - 1}.000000",
            )
            for rank, fragment_id in enumerate(trace_ids, start=1)
        )
        candidate_count = len(trace) if self.candidate_count is None else self.candidate_count
        return FtsSearchResult(
            retrieval_corpus_hash=_fts_corpus_hash(fragments),
            query_tokens=("alpha",) if trace else (),
            max_candidates=max_candidates,
            candidate_count=candidate_count,
            trace=trace,
            profile=self.profile,
        )


class FakeBackend:
    def __init__(
        self,
        *,
        snapshot: CorpusSnapshot,
        policy: AccessPolicySnapshot,
        candidates: tuple[SnapshotCandidate, ...],
        fragments: tuple[SourceFragmentText, ...],
    ) -> None:
        self.snapshot = snapshot
        self.policy = policy
        self.candidates = candidates
        self.fragments = {fragment.source_fragment_id: fragment for fragment in fragments}
        self.queries: dict[str, QueryRequest] = {}
        self.packets: dict[str, EvidencePacket] = {}
        self.started_runs: list[ProcessingRunRecord] = []
        self.failed_runs: list[ProcessingRunRecord] = []
        self.completed_runs: list[ProcessingRunRecord] = []
        self.authorization_scopes: list[RequestScope] = []
        self.read_ids: list[tuple[str, ...]] = []
        self.complete_error = False
        self.persist_error = False

    def get_corpus_snapshot(self, corpus_snapshot_id: str) -> CorpusSnapshot:
        return self.snapshot

    def persist_query_request(self, request: QueryRequest) -> None:
        if self.persist_error:
            raise RuntimeError("simulated query persistence failure")
        previous = self.queries.setdefault(request.query_request_id, request)
        if previous.canonical_bytes != request.canonical_bytes:
            raise RuntimeError("content identity collision")

    def load_query_request(self, query_request_id: str) -> QueryRequest:
        return self.queries[query_request_id]

    def begin_recall_run(
        self,
        *,
        processing_run_id: str,
        kind: RecallRunKind,
        query_request_id: str,
        code_version: str,
        profile_version: str,
    ) -> ProcessingRunRecord:
        if query_request_id not in self.queries:
            raise RuntimeError("QueryRequest must exist before run start")
        if any(run.processing_run_id == processing_run_id for run in self.started_runs):
            raise RuntimeError("ProcessingRun identity already exists")
        sequence = len(self.started_runs) + 1
        run = ProcessingRunRecord(
            processing_run_id=processing_run_id,
            kind=kind,
            status=ProcessingRunStatus.RUNNING,
            code_version=code_version,
            profile_version=profile_version,
            started_at=f"2026-07-20T00:00:0{sequence}.000000Z",
            finished_at=None,
            error_code=None,
            output_hash=None,
        )
        self.started_runs.append(run)
        return run

    def authorize_read(
        self,
        *,
        access_policy_id: str,
        scope: RequestScope,
    ) -> AuthorizedRead:
        if access_policy_id != self.policy.access_policy_id:
            raise RuntimeError("unknown policy")
        self.authorization_scopes.append(scope)
        compiled = compile_access(policy=self.policy, scope=scope, candidates=self.candidates)
        return AuthorizedRead(
            authorization_id=f"authorization_{len(self.started_runs)}",
            repository_instance_id="repository_test",
            access_policy_id=access_policy_id,
            scope=scope,
            compiled=compiled,
        )

    def read_permitted_fragments(
        self,
        authorization: AuthorizedRead,
    ) -> tuple[SourceFragmentText, ...]:
        identifiers = tuple(
            item.source_fragment_id for item in authorization.compiled.manifest.items
        )
        self.read_ids.append(identifiers)
        return tuple(self.fragments[fragment_id] for fragment_id in identifiers)

    def complete_recall_run(
        self,
        *,
        processing_run_id: str,
        packet: EvidencePacket,
    ) -> tuple[ProcessingRunRecord, EvidencePacket]:
        if self.complete_error:
            raise RuntimeError("simulated atomic persistence failure")
        started = self._started(processing_run_id)
        persisted = self.packets.setdefault(packet.evidence_packet_id, packet)
        run = ProcessingRunRecord(
            processing_run_id=processing_run_id,
            kind=started.kind,
            status=ProcessingRunStatus.SUCCEEDED,
            code_version=started.code_version,
            profile_version=started.profile_version,
            started_at=started.started_at,
            finished_at="2026-07-20T00:01:00.000000Z",
            error_code=None,
            output_hash=persisted.packet_hash,
        )
        self.completed_runs.append(run)
        return run, persisted

    def fail_recall_run(
        self,
        *,
        processing_run_id: str,
        error_code: str,
        error_hash: str,
    ) -> ProcessingRunRecord:
        started = self._started(processing_run_id)
        failed = ProcessingRunRecord(
            processing_run_id=processing_run_id,
            kind=started.kind,
            status=ProcessingRunStatus.FAILED,
            code_version=started.code_version,
            profile_version=started.profile_version,
            started_at=started.started_at,
            finished_at="2026-07-20T00:01:00.000000Z",
            error_code=error_code,
            output_hash=error_hash,
        )
        self.failed_runs.append(failed)
        return failed

    def get_processing_run(self, processing_run_id: str) -> ProcessingRunRecord:
        for run in reversed((*self.completed_runs, *self.failed_runs)):
            if run.processing_run_id == processing_run_id:
                return run
        try:
            return self._started(processing_run_id)
        except StopIteration as error:
            raise ProcessingRunNotFoundError(
                f"ProcessingRun does not exist: {processing_run_id}"
            ) from error

    def load_evidence_packet(self, evidence_packet_id: str) -> EvidencePacket:
        return self.packets[evidence_packet_id]

    def _started(self, processing_run_id: str) -> ProcessingRunRecord:
        return next(run for run in self.started_runs if run.processing_run_id == processing_run_id)


@dataclass(frozen=True, slots=True)
class Scenario:
    backend: FakeBackend
    request: QueryRequest
    fts: FakeFts
    service: RecallService


def _scenario(
    specs: tuple[SourceSpec, ...],
    *,
    denied_sources: tuple[str, ...] = (),
    exclusions: QueryExclusions | None = None,
    retrieval: RetrievalBudget | None = None,
    fts: FakeFts | None = None,
) -> Scenario:
    snapshot = CorpusSnapshot(
        library_id=LIBRARY_ID,
        collection_ids=(COLLECTION_ID,),
        members=tuple(_snapshot_member(spec) for spec in specs),
    )
    policy = AccessPolicySnapshot(
        access_policy_id=POLICY_ID,
        library_id=LIBRARY_ID,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(COLLECTION_ID, PolicyEffect.ALLOW),),
        source_rules=tuple(
            SourceRule(source_id=source_id, effect=PolicyEffect.DENY)
            for source_id in denied_sources
        ),
    )
    backend = FakeBackend(
        snapshot=snapshot,
        policy=policy,
        candidates=tuple(_candidate(spec) for spec in specs),
        fragments=tuple(_stored_fragment(spec, ordinal=index) for index, spec in enumerate(specs)),
    )
    request = QueryRequest(
        question="alpha evidence",
        library_id=LIBRARY_ID,
        collection_ids=(COLLECTION_ID,),
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        access_policy_id=POLICY_ID,
        purpose="research",
        exclusions=exclusions or QueryExclusions(),
        retrieval=retrieval or RetrievalBudget(max_candidates=10, max_source_fragments=10),
    )
    injected = fts or FakeFts()
    service = RecallService(
        backend,
        profile_version=injected.profile.profile_version,
        code_version=CODE_VERSION,
        fts_search=injected,
    )
    return Scenario(backend=backend, request=request, fts=injected, service=service)


def _snapshot_member(spec: SourceSpec) -> SnapshotMember:
    return SnapshotMember(
        collection_id=COLLECTION_ID,
        source_id=spec.source_id,
        source_version_id=spec.version_id,
        content_sha256=sha256_hex(spec.text.encode("utf-8")),
        membership_state=spec.state,
        source_family_id=spec.family_id,
        root_source_id=spec.source_id,
        family_role=SourceFamilyRole.ROOT,
    )


def _candidate(spec: SourceSpec) -> SnapshotCandidate:
    return SnapshotCandidate(
        source_fragment_id=spec.fragment_id,
        source_version_id=spec.version_id,
        source_id=spec.source_id,
        source_family_id=spec.family_id,
        memberships=(SnapshotMembership(COLLECTION_ID, spec.state),),
    )


def _stored_fragment(spec: SourceSpec, *, ordinal: int) -> SourceFragmentText:
    address = MarkdownSourceAddress(
        heading_path=("Evidence",),
        line_start=ordinal + 1,
        line_end=ordinal + 1,
        char_start=0,
        char_end=len(spec.text),
    )
    return SourceFragmentText(
        source_fragment_id=spec.fragment_id,
        source_version_id=spec.version_id,
        source_id=spec.source_id,
        ordinal=ordinal,
        fragment_kind="paragraph",
        text=spec.text,
        text_sha256=sha256_hex(spec.text.encode("utf-8")),
        source_address_json=canonical_json_bytes(address.payload()).decode("utf-8"),
    )


def _fts_corpus_hash(fragments: tuple[FtsFragment, ...]) -> str:
    return canonical_sha256_hex(
        {
            "schema": "dithyramba.fts_corpus/1.0",
            "fragments": [
                fragment.identity_payload()
                for fragment in sorted(fragments, key=lambda item: item.source_fragment_id)
            ],
        }
    )


def _omissions(packet: EvidencePacket) -> dict[OmissionCategory, int | None]:
    return {item.category: item.count for item in packet.coverage_report.omissions}


def _raise_runtime(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("simulated backend fault")


def _forged_authorization(
    authorization: AuthorizedRead,
    item: PermittedManifestItem,
) -> AuthorizedRead:
    original = authorization.compiled
    manifest = PermittedManifest(
        library_id=original.manifest.library_id,
        snapshot_hash=original.manifest.snapshot_hash,
        items=(item,),
    )
    token = PermittedToken(
        library_id=original.token.library_id,
        snapshot_hash=original.token.snapshot_hash,
        policy_hash=original.token.policy_hash,
        exclusion_hash=original.token.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
    )
    return replace(
        authorization,
        compiled=CompiledAccess(
            manifest=manifest,
            token=token,
            public_result=original.public_result,
        ),
    )


def test_empty_permitted_set_returns_complete_no_evidence_packet() -> None:
    scenario = _scenario(
        (SourceSpec("holdout", "alpha hidden", MembershipState.HOLDOUT),),
    )

    result = scenario.service.recall(scenario.request)

    assert result.succeeded
    assert result.packet.result_status is PacketResultStatus.NO_EVIDENCE
    assert result.packet.source_fragments == ()
    assert result.packet.read_receipt.items == ()
    assert result.packet.retrieval_receipt.trace == ()
    assert result.packet.coverage_report.processed_count == 0
    assert result.packet.coverage_report.omissions == ()
    assert scenario.backend.read_ids == [()]
    assert scenario.fts.calls[0][1] == ()


def test_policy_denial_and_holdout_never_cross_the_protected_read_or_fts_boundary() -> None:
    allowed = SourceSpec("allowed", "alpha public")
    denied = SourceSpec("denied", "alpha policy secret")
    holdout = SourceSpec("holdout", "alpha holdout secret", MembershipState.HOLDOUT)
    scenario = _scenario(
        (allowed, denied, holdout),
        denied_sources=(denied.source_id,),
    )

    packet = scenario.service.recall(scenario.request).packet

    assert scenario.backend.read_ids == [(allowed.fragment_id,)]
    assert tuple(item.source_fragment_id for item in scenario.fts.calls[0][1]) == (
        allowed.fragment_id,
    )
    assert packet.access_receipt.policy_omission_present is True
    assert _omissions(packet) == {OmissionCategory.POLICY: None}
    assert denied.text not in packet.canonical_bytes.decode("utf-8")
    assert holdout.text not in packet.canonical_bytes.decode("utf-8")


def test_read_receipt_covers_every_fragment_populating_fts_not_only_selected_prefix() -> None:
    first = SourceSpec("first", "alpha first")
    second = SourceSpec("second", "alpha second")
    fts = FakeFts(trace_ids=(first.fragment_id, second.fragment_id))
    scenario = _scenario(
        (first, second),
        retrieval=RetrievalBudget(max_candidates=5, max_source_fragments=1),
        fts=fts,
    )

    packet = scenario.service.recall(scenario.request).packet

    assert len(packet.source_fragments) == 1
    assert tuple(item.source_fragment_id for item in packet.read_receipt.items) == (
        first.fragment_id,
        second.fragment_id,
    )
    assert tuple(item.source_fragment_id for item in fts.calls[0][1]) == (
        first.fragment_id,
        second.fragment_id,
    )
    assert packet.coverage_report.processed_count == 2
    assert _omissions(packet) == {OmissionCategory.BUDGET: 1}


def test_explicit_exclusions_count_only_caller_ids_and_budget_counts_unselected_matches() -> None:
    included = (
        SourceSpec("included_a", "alpha included a"),
        SourceSpec("included_b", "alpha included b"),
        SourceSpec("included_c", "alpha included c"),
        SourceSpec("included_d", "alpha included d"),
    )
    excluded = SourceSpec("excluded", "alpha excluded")
    exclusions = QueryExclusions(
        source_ids=(excluded.source_id,),
        source_family_ids=("family_declared_only",),
        source_fragment_ids=("fragment_declared_only",),
    )
    fts = FakeFts(
        trace_ids=(included[0].fragment_id, included[1].fragment_id),
        candidate_count=4,
    )
    scenario = _scenario(
        (*included, excluded),
        exclusions=exclusions,
        retrieval=RetrievalBudget(max_candidates=2, max_source_fragments=1),
        fts=fts,
    )

    packet = scenario.service.recall(scenario.request).packet

    assert scenario.backend.read_ids == [
        tuple(spec.fragment_id for spec in included),
    ]
    assert _omissions(packet) == {
        OmissionCategory.BUDGET: 3,
        OmissionCategory.EXPLICIT_EXCLUSION: 3,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"library_id": "library_other"},
        {"collection_ids": ("collection_other",)},
        {"corpus_snapshot_id": "snapshot_other"},
    ],
)
def test_query_must_close_over_the_exact_loaded_snapshot_scope(
    change: dict[str, object],
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    request = scenario.request.model_copy(update=change)

    with pytest.raises((RecallSnapshotError, RecallRequestError)):
        scenario.service.recall(request)

    assert scenario.backend.started_runs == []


def test_query_persistence_failure_occurs_before_any_processing_run() -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    scenario.backend.persist_error = True

    with pytest.raises(RecallPersistenceError, match="persisted and reloaded"):
        scenario.service.recall(scenario.request)

    assert scenario.backend.started_runs == []
    assert scenario.backend.failed_runs == []


def test_replay_reuses_original_request_and_reproduces_packet_byte_for_byte() -> None:
    first = SourceSpec("first", "alpha first")
    second = SourceSpec("second", "alpha second")
    scenario = _scenario((first, second))
    original = scenario.service.recall(scenario.request)

    replayed = scenario.service.replay(original.packet.evidence_packet_id)

    assert isinstance(replayed, RecallResult)
    assert replayed.run.kind == "replay"
    assert replayed.packet.evidence_packet_id == original.packet.evidence_packet_id
    assert replayed.packet.packet_hash == original.packet.packet_hash
    assert replayed.packet.canonical_bytes == original.packet.canonical_bytes
    assert [run.kind for run in scenario.backend.started_runs] == ["recall", "replay"]


def test_verify_replay_reproduces_packet_without_persisting_a_run() -> None:
    first = SourceSpec("first", "alpha first")
    second = SourceSpec("second", "alpha second")
    scenario = _scenario((first, second))
    original = scenario.service.recall(scenario.request)
    before = (
        tuple(scenario.backend.started_runs),
        tuple(scenario.backend.completed_runs),
        tuple(scenario.backend.failed_runs),
    )

    verified = scenario.service.verify_replay(original.packet.evidence_packet_id)

    assert verified.evidence_packet_id == original.packet.evidence_packet_id
    assert verified.packet_hash == original.packet.packet_hash
    assert verified.canonical_bytes == original.packet.canonical_bytes
    assert tuple(scenario.backend.started_runs) == before[0]
    assert tuple(scenario.backend.completed_runs) == before[1]
    assert tuple(scenario.backend.failed_runs) == before[2]


def test_verify_replay_mismatch_is_typed_without_persisting_a_failed_run() -> None:
    first = SourceSpec("first", "alpha first")
    second = SourceSpec("second", "alpha second")
    scenario = _scenario((first, second))
    original = scenario.service.recall(scenario.request)
    verifier = RecallService(
        scenario.backend,
        profile_version=scenario.fts.profile.profile_version,
        code_version=CODE_VERSION,
        fts_search=FakeFts(
            trace_ids=(second.fragment_id, first.fragment_id),
            profile=scenario.fts.profile,
        ),
    )
    before = (
        tuple(scenario.backend.started_runs),
        tuple(scenario.backend.completed_runs),
        tuple(scenario.backend.failed_runs),
    )

    with pytest.raises(RecallReplayMismatchError, match="verification did not reproduce"):
        verifier.verify_replay(original.packet.evidence_packet_id)

    assert tuple(scenario.backend.started_runs) == before[0]
    assert tuple(scenario.backend.completed_runs) == before[1]
    assert tuple(scenario.backend.failed_runs) == before[2]


def test_verify_replay_execution_failure_is_typed_without_a_run() -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.service.recall(scenario.request)
    verifier = RecallService(
        scenario.backend,
        profile_version=scenario.fts.profile.profile_version,
        code_version=CODE_VERSION,
        fts_search=_raise_runtime,
    )
    started_count = len(scenario.backend.started_runs)

    with pytest.raises(RecallExecutionError, match="verification failed closed"):
        verifier.verify_replay(original.packet.evidence_packet_id)

    assert len(scenario.backend.started_runs) == started_count
    assert scenario.backend.failed_runs == []


@pytest.mark.parametrize(
    ("code_version", "profile_version"),
    [
        ("0.1.0.dev3", None),
        (None, "fts_v1.different"),
    ],
)
def test_replay_rejects_dependency_drift_before_starting_a_run(
    code_version: str | None,
    profile_version: str | None,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.service.recall(scenario.request)
    start_count = len(scenario.backend.started_runs)
    changed = RecallService(
        scenario.backend,
        code_version=code_version or CODE_VERSION,
        profile_version=profile_version or scenario.fts.profile.profile_version,
        fts_search=scenario.fts,
    )

    with pytest.raises(RecallDependencyMismatchError):
        changed.replay(original.packet.evidence_packet_id)

    assert len(scenario.backend.started_runs) == start_count


def test_replay_mismatch_is_typed_and_records_a_failed_replay_run() -> None:
    first = SourceSpec("first", "alpha first")
    second = SourceSpec("second", "alpha second")
    scenario = _scenario((first, second))
    original = scenario.service.recall(scenario.request)
    changed_fts = FakeFts(
        trace_ids=(second.fragment_id, first.fragment_id),
        profile=scenario.fts.profile,
    )
    replay_service = RecallService(
        scenario.backend,
        profile_version=scenario.fts.profile.profile_version,
        code_version=CODE_VERSION,
        fts_search=changed_fts,
    )

    with pytest.raises(RecallReplayMismatchError):
        replay_service.replay(original.packet.evidence_packet_id)

    failed = scenario.backend.failed_runs[-1]
    assert failed.kind == "replay"
    assert failed.error_code == RecallReplayMismatchError.code
    assert failed.output_hash == canonical_sha256_hex(
        {
            "schema": "dithyramba.recall_failure/1.0",
            "error_code": RecallReplayMismatchError.code,
            "error_type": "RecallReplayMismatchError",
        }
    )


def test_atomic_persistence_failure_is_typed_and_records_stable_failed_run() -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    scenario.backend.complete_error = True

    with pytest.raises(RecallPersistenceError, match="atomically persisted"):
        scenario.service.recall(scenario.request)

    assert scenario.backend.completed_runs == []
    failed = scenario.backend.failed_runs[-1]
    assert failed.status is ProcessingRunStatus.FAILED
    assert failed.error_code == RecallPersistenceError.code
    assert failed.output_hash == canonical_sha256_hex(
        {
            "schema": "dithyramba.recall_failure/1.0",
            "error_code": RecallPersistenceError.code,
            "error_type": "RecallPersistenceError",
        }
    )


def test_fts_exception_fails_closed_and_never_completes_artifacts() -> None:
    class BrokenFts(FakeFts):
        def __call__(
            self,
            *,
            question: str,
            fragments: tuple[FtsFragment, ...],
            max_candidates: int,
        ) -> NoReturn:
            raise RuntimeError("simulated injected FTS failure")

    broken = BrokenFts()
    scenario = _scenario((SourceSpec("allowed", "alpha public"),), fts=broken)

    with pytest.raises(RecallExecutionError, match="recall execution failed closed"):
        scenario.service.recall(scenario.request)

    assert scenario.backend.completed_runs == []
    assert scenario.backend.failed_runs[-1].error_code == "recall_execution_failed"


def test_constructor_validates_dependencies_and_exposes_injected_backend() -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))

    assert scenario.service.backend is scenario.backend
    with pytest.raises(ValueError, match="profile_version"):
        RecallService(scenario.backend, profile_version="")
    with pytest.raises(ValueError, match="code_version"):
        RecallService(
            scenario.backend,
            profile_version=scenario.fts.profile.profile_version,
            code_version="INVALID",
        )
    with pytest.raises(TypeError, match="callable"):
        RecallService(
            scenario.backend,
            profile_version=scenario.fts.profile.profile_version,
            fts_search=cast(Any, None),
        )


def test_recall_rejects_noncanonical_request_and_persisted_request_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    with pytest.raises(RecallRequestError, match="exact QueryRequest"):
        scenario.service.recall(cast(Any, object()))
    malformed = scenario.request.model_copy(update={"retrieval": object()})
    with pytest.raises(RecallRequestError, match="not canonical"):
        scenario.service.recall(malformed)

    different = scenario.request.model_copy(update={"question": "different valid question"})
    monkeypatch.setattr(
        scenario.backend,
        "load_query_request",
        lambda query_request_id: different,
    )
    with pytest.raises(RecallPersistenceError, match="differs"):
        scenario.service.recall(scenario.request)

    monkeypatch.undo()
    invalid_persisted = _scenario((SourceSpec("allowed", "alpha public"),))
    monkeypatch.setattr(
        invalid_persisted.backend,
        "load_query_request",
        lambda query_request_id: cast(QueryRequest, object()),
    )
    with pytest.raises(RecallPersistenceError, match="not canonical"):
        invalid_persisted.service.recall(invalid_persisted.request)


@pytest.mark.parametrize(
    "fault", ["snapshot_load", "snapshot_type", "run_start", "run_type", "run_state"]
)
def test_snapshot_and_run_backend_faults_fail_before_execution(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    if fault == "snapshot_load":
        monkeypatch.setattr(scenario.backend, "get_corpus_snapshot", _raise_runtime)
        expected: type[Exception] = RecallSnapshotError
    elif fault == "snapshot_type":
        monkeypatch.setattr(
            scenario.backend,
            "get_corpus_snapshot",
            lambda corpus_snapshot_id: cast(Any, object()),
        )
        expected = RecallSnapshotError
    elif fault == "run_start":
        monkeypatch.setattr(scenario.backend, "begin_recall_run", _raise_runtime)
        expected = RecallPersistenceError
    elif fault == "run_type":
        monkeypatch.setattr(
            scenario.backend,
            "begin_recall_run",
            lambda **kwargs: cast(Any, object()),
        )
        expected = RecallPersistenceError
    else:
        original = scenario.backend.begin_recall_run

        def invalid_run(**kwargs: Any) -> ProcessingRunRecord:
            return replace(original(**kwargs), status=ProcessingRunStatus.SUCCEEDED)

        monkeypatch.setattr(scenario.backend, "begin_recall_run", invalid_run)
        expected = RecallPersistenceError

    with pytest.raises(expected):
        scenario.service.recall(scenario.request)

    assert scenario.backend.completed_runs == []


@pytest.mark.parametrize(
    "fault",
    ["type", "identity", "policy", "scope", "compiled", "token_scope"],
)
def test_authorization_must_be_the_exact_request_bound_capability(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.backend.authorize_read

    def invalid_authorization(**kwargs: Any) -> AuthorizedRead:
        authorization = original(**kwargs)
        if fault == "type":
            return cast(AuthorizedRead, object())
        if fault == "identity":
            return replace(authorization, authorization_id="bad")
        if fault == "policy":
            return replace(authorization, access_policy_id="policy_other")
        if fault == "scope":
            return replace(
                authorization,
                scope=replace(authorization.scope, purpose="other"),
            )
        if fault == "compiled":
            return replace(authorization, compiled=cast(Any, object()))
        token = authorization.compiled.token
        mismatched = PermittedToken(
            library_id=token.library_id,
            snapshot_hash=token.snapshot_hash,
            policy_hash=token.policy_hash,
            exclusion_hash="f" * 64,
            permitted_set_hash=token.permitted_set_hash,
        )
        return replace(
            authorization,
            compiled=CompiledAccess(
                manifest=authorization.compiled.manifest,
                token=mismatched,
                public_result=authorization.compiled.public_result,
            ),
        )

    monkeypatch.setattr(scenario.backend, "authorize_read", invalid_authorization)

    with pytest.raises(RecallExecutionError):
        scenario.service.recall(scenario.request)

    assert scenario.backend.read_ids == []
    assert scenario.backend.failed_runs[-1].error_code == RecallExecutionError.code


@pytest.mark.parametrize("fault", ["missing_version", "lineage", "collections", "holdout"])
def test_permitted_manifest_must_close_over_active_snapshot_lineage(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = MembershipState.HOLDOUT if fault == "holdout" else MembershipState.ACTIVE
    spec = SourceSpec("member", "alpha member", state)
    scenario = _scenario((spec,))
    original = scenario.backend.authorize_read

    def invalid_authorization(**kwargs: Any) -> AuthorizedRead:
        authorization = original(**kwargs)
        item = PermittedManifestItem(
            source_fragment_id=spec.fragment_id,
            source_version_id=(
                "source_version_missing" if fault == "missing_version" else spec.version_id
            ),
            source_id="source_other" if fault == "lineage" else spec.source_id,
            source_family_id=spec.family_id,
            collection_ids=("collection_other" if fault == "collections" else COLLECTION_ID,),
        )
        return _forged_authorization(authorization, item)

    monkeypatch.setattr(scenario.backend, "authorize_read", invalid_authorization)

    with pytest.raises(RecallExecutionError):
        scenario.service.recall(scenario.request)

    assert scenario.backend.read_ids == []


@pytest.mark.parametrize(
    "fault",
    [
        "type",
        "manifest_set",
        "lineage",
        "metadata",
        "address_type",
        "address_json",
        "address_duplicate",
        "address_shape",
        "address_noncanonical",
        "address_float",
        "address_contract",
        "text_hash",
    ],
)
def test_protected_read_rejects_noncanonical_or_nonmanifest_text(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.backend.read_permitted_fragments

    def invalid_read(authorization: AuthorizedRead) -> tuple[SourceFragmentText, ...]:
        fragments = original(authorization)
        fragment = fragments[0]
        if fault == "type":
            return cast(tuple[SourceFragmentText, ...], list(fragments))
        if fault == "manifest_set":
            return ()
        if fault == "lineage":
            fragment = replace(fragment, source_id="source_other")
        elif fault == "metadata":
            fragment = replace(fragment, ordinal=-1)
        elif fault == "address_type":
            fragment = replace(fragment, source_address_json=cast(Any, 1))
        elif fault == "address_json":
            fragment = replace(fragment, source_address_json="{")
        elif fault == "address_duplicate":
            fragment = replace(fragment, source_address_json='{"kind":"x","kind":"y"}')
        elif fault == "address_shape":
            fragment = replace(fragment, source_address_json="[]")
        elif fault == "address_noncanonical":
            fragment = replace(fragment, source_address_json=fragment.source_address_json + "\n")
        elif fault == "address_float":
            fragment = replace(fragment, source_address_json='{"value":NaN}')
        elif fault == "address_contract":
            fragment = replace(
                fragment,
                source_address_json=canonical_json_bytes(
                    {
                        "schema": "dithyramba.source_address/1.0",
                        "kind": "unknown",
                    }
                ).decode("utf-8"),
            )
        elif fault == "text_hash":
            fragment = replace(fragment, text_sha256="0" * 64)
        return (fragment,)

    monkeypatch.setattr(scenario.backend, "read_permitted_fragments", invalid_read)

    with pytest.raises(RecallExecutionError):
        scenario.service.recall(scenario.request)

    assert scenario.backend.completed_runs == []


@pytest.mark.parametrize(
    "fault",
    ["type", "profile", "max_candidates", "corpus_hash", "inflated_count", "foreign_trace"],
)
def test_injected_fts_result_must_close_over_exact_read_corpus_and_profile(
    fault: str,
) -> None:
    spec = SourceSpec("allowed", "alpha public")
    base = FakeFts()

    def invalid_fts(
        *,
        question: str,
        fragments: tuple[FtsFragment, ...],
        max_candidates: int,
    ) -> FtsSearchResult:
        result = base(
            question=question,
            fragments=fragments,
            max_candidates=max_candidates,
        )
        if fault == "type":
            return cast(FtsSearchResult, object())
        if fault == "profile":
            return replace(
                result,
                profile=FtsRuntimeProfile("3.45.2", "b" * 64),
            )
        if fault == "max_candidates":
            return replace(result, max_candidates=max_candidates - 1)
        if fault == "corpus_hash":
            return replace(result, retrieval_corpus_hash="b" * 64)
        if fault == "inflated_count":
            return replace(result, candidate_count=2)
        return replace(
            result,
            trace=(FtsCandidate("fragment_foreign", 1, "-1.000000"),),
        )

    scenario = _scenario((spec,), fts=base)
    service = RecallService(
        scenario.backend,
        profile_version=base.profile.profile_version,
        code_version=CODE_VERSION,
        fts_search=invalid_fts,
    )

    with pytest.raises(RecallExecutionError):
        service.recall(scenario.request)

    assert scenario.backend.failed_runs[-1].error_code == RecallExecutionError.code


def test_operator_interrupt_is_reraised_after_run_is_terminalized() -> None:
    class InterruptedFts(FakeFts):
        def __call__(
            self,
            *,
            question: str,
            fragments: tuple[FtsFragment, ...],
            max_candidates: int,
        ) -> NoReturn:
            raise KeyboardInterrupt

    scenario = _scenario(
        (SourceSpec("allowed", "alpha public"),),
        fts=InterruptedFts(),
    )

    with pytest.raises(KeyboardInterrupt):
        scenario.service.recall(scenario.request)

    assert len(scenario.backend.started_runs) == 1
    assert len(scenario.backend.failed_runs) == 1
    assert scenario.backend.failed_runs[0].error_code == RecallInterruptedError.code


def test_begin_commit_then_runtime_error_is_reconciled_without_a_running_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.backend.begin_recall_run
    committed_id = ""

    def begin_then_error(**kwargs: Any) -> NoReturn:
        nonlocal committed_id
        committed = original(**kwargs)
        committed_id = committed.processing_run_id
        raise RuntimeError("simulated post-commit begin failure")

    monkeypatch.setattr(scenario.backend, "begin_recall_run", begin_then_error)

    with pytest.raises(RecallPersistenceError, match="could not be started"):
        scenario.service.recall(scenario.request)

    current = scenario.backend.get_processing_run(committed_id)
    assert current.status is ProcessingRunStatus.FAILED
    assert current.error_code == RecallPersistenceError.code
    assert all(
        run.status is not ProcessingRunStatus.RUNNING for run in scenario.backend.failed_runs
    )


def test_begin_commit_then_keyboard_interrupt_is_terminalized_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.backend.begin_recall_run
    committed_id = ""

    def begin_then_interrupt(**kwargs: Any) -> NoReturn:
        nonlocal committed_id
        committed = original(**kwargs)
        committed_id = committed.processing_run_id
        raise KeyboardInterrupt

    monkeypatch.setattr(scenario.backend, "begin_recall_run", begin_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        scenario.service.recall(scenario.request)

    current = scenario.backend.get_processing_run(committed_id)
    assert current.status is ProcessingRunStatus.FAILED
    assert current.error_code == RecallInterruptedError.code


def test_interrupt_before_failure_commit_is_retried_terminalized_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenFts(FakeFts):
        def __call__(
            self,
            *,
            question: str,
            fragments: tuple[FtsFragment, ...],
            max_candidates: int,
        ) -> NoReturn:
            raise RuntimeError("force ordinary execution failure")

    scenario = _scenario(
        (SourceSpec("allowed", "alpha public"),),
        fts=BrokenFts(),
    )
    original = scenario.backend.fail_recall_run
    calls = 0

    def interrupt_once_before_failure_commit(**kwargs: Any) -> ProcessingRunRecord:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original(**kwargs)

    monkeypatch.setattr(
        scenario.backend,
        "fail_recall_run",
        interrupt_once_before_failure_commit,
    )

    with pytest.raises(KeyboardInterrupt):
        scenario.service.recall(scenario.request)

    assert calls == 2
    assert len(scenario.backend.failed_runs) == 1
    assert scenario.backend.failed_runs[0].status is ProcessingRunStatus.FAILED
    assert scenario.backend.failed_runs[0].error_code == RecallExecutionError.code


def test_interrupt_after_atomic_completion_preserves_terminal_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.backend.complete_recall_run

    def complete_then_interrupt(
        *,
        processing_run_id: str,
        packet: EvidencePacket,
    ) -> NoReturn:
        original(processing_run_id=processing_run_id, packet=packet)
        raise KeyboardInterrupt

    monkeypatch.setattr(scenario.backend, "complete_recall_run", complete_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        scenario.service.recall(scenario.request)

    assert len(scenario.backend.completed_runs) == 1
    assert scenario.backend.completed_runs[0].status is ProcessingRunStatus.SUCCEEDED
    assert scenario.backend.failed_runs == []


def test_runtime_error_after_atomic_completion_returns_reloaded_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.backend.complete_recall_run

    def complete_then_error(
        *,
        processing_run_id: str,
        packet: EvidencePacket,
    ) -> NoReturn:
        original(processing_run_id=processing_run_id, packet=packet)
        raise RuntimeError("simulated post-commit completion failure")

    monkeypatch.setattr(scenario.backend, "complete_recall_run", complete_then_error)

    result = scenario.service.recall(scenario.request)

    assert result.run.status is ProcessingRunStatus.SUCCEEDED
    assert result.run.output_hash == result.packet.packet_hash
    assert scenario.backend.failed_runs == []


def test_concurrent_success_between_failure_reload_and_transition_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    atomic_complete = scenario.backend.complete_recall_run
    packet_to_commit: EvidencePacket | None = None

    def fail_completion_before_commit(
        *,
        processing_run_id: str,
        packet: EvidencePacket,
    ) -> NoReturn:
        nonlocal packet_to_commit
        packet_to_commit = packet
        raise RuntimeError("simulated completion transport failure")

    def concurrent_success_then_conflict(
        *,
        processing_run_id: str,
        error_code: str,
        error_hash: str,
    ) -> NoReturn:
        del error_code, error_hash
        assert packet_to_commit is not None
        atomic_complete(processing_run_id=processing_run_id, packet=packet_to_commit)
        raise RuntimeError("failure transition lost to concurrent success")

    monkeypatch.setattr(
        scenario.backend,
        "complete_recall_run",
        fail_completion_before_commit,
    )
    monkeypatch.setattr(
        scenario.backend,
        "fail_recall_run",
        concurrent_success_then_conflict,
    )

    result = scenario.service.recall(scenario.request)

    assert result.run.status is ProcessingRunStatus.SUCCEEDED
    assert result.packet is packet_to_commit
    assert scenario.backend.failed_runs == []


def test_already_failed_state_is_preserved_without_a_second_failure_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    atomic_fail = scenario.backend.fail_recall_run
    preexisting_hash = "d" * 64

    def complete_after_concurrent_failure(
        *,
        processing_run_id: str,
        packet: EvidencePacket,
    ) -> NoReturn:
        del packet
        atomic_fail(
            processing_run_id=processing_run_id,
            error_code="preexisting_failure",
            error_hash=preexisting_hash,
        )
        raise RuntimeError("completion lost to concurrent failure")

    def unexpected_second_failure(**kwargs: Any) -> NoReturn:
        del kwargs
        raise AssertionError("terminal failed state must not be rewritten")

    monkeypatch.setattr(
        scenario.backend,
        "complete_recall_run",
        complete_after_concurrent_failure,
    )
    monkeypatch.setattr(
        scenario.backend,
        "fail_recall_run",
        unexpected_second_failure,
    )

    with pytest.raises(RecallPersistenceError, match="atomically persisted"):
        scenario.service.recall(scenario.request)

    assert len(scenario.backend.failed_runs) == 1
    assert scenario.backend.failed_runs[0].error_code == "preexisting_failure"
    assert scenario.backend.failed_runs[0].output_hash == preexisting_hash


@pytest.mark.parametrize("fault", ["shape", "types", "run", "packet"])
def test_atomic_completion_must_return_the_exact_terminal_run_and_packet(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))

    def invalid_completion(
        *,
        processing_run_id: str,
        packet: EvidencePacket,
    ) -> tuple[ProcessingRunRecord, EvidencePacket]:
        if fault == "shape":
            return cast(tuple[ProcessingRunRecord, EvidencePacket], ())
        if fault == "types":
            return cast(tuple[ProcessingRunRecord, EvidencePacket], (object(), packet))
        started = scenario.backend._started(processing_run_id)
        run = ProcessingRunRecord(
            processing_run_id=processing_run_id,
            kind=started.kind,
            status=ProcessingRunStatus.SUCCEEDED,
            code_version=started.code_version,
            profile_version=started.profile_version,
            started_at=started.started_at,
            finished_at="2026-07-20T00:01:00.000000Z",
            error_code=None,
            output_hash=("b" * 64 if fault == "run" else packet.packet_hash),
        )
        if fault == "packet":
            packet = packet.model_copy(update={"result_status": PacketResultStatus.NO_EVIDENCE})
        return run, packet

    monkeypatch.setattr(scenario.backend, "complete_recall_run", invalid_completion)

    with pytest.raises(RecallPersistenceError):
        scenario.service.recall(scenario.request)

    assert scenario.backend.failed_runs[-1].error_code == RecallPersistenceError.code


@pytest.mark.parametrize("fault", ["raise", "type", "state"])
def test_failed_run_must_itself_be_durable_and_canonical(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenFts(FakeFts):
        def __call__(
            self,
            *,
            question: str,
            fragments: tuple[FtsFragment, ...],
            max_candidates: int,
        ) -> NoReturn:
            raise RuntimeError("force failed lifecycle")

    scenario = _scenario((SourceSpec("allowed", "alpha public"),), fts=BrokenFts())
    original = scenario.backend.fail_recall_run
    if fault == "raise":
        monkeypatch.setattr(scenario.backend, "fail_recall_run", _raise_runtime)
    elif fault == "type":
        monkeypatch.setattr(
            scenario.backend,
            "fail_recall_run",
            lambda **kwargs: cast(Any, object()),
        )
    else:

        def invalid_failed(**kwargs: Any) -> ProcessingRunRecord:
            return replace(original(**kwargs), status=ProcessingRunStatus.RUNNING)

        monkeypatch.setattr(scenario.backend, "fail_recall_run", invalid_failed)

    if fault == "state":
        with pytest.raises(RecallExecutionError, match="execution failed closed"):
            scenario.service.recall(scenario.request)
        assert scenario.backend.failed_runs[-1].status is ProcessingRunStatus.FAILED
    else:
        with pytest.raises(RecallPersistenceError, match="failure ProcessingRun"):
            scenario.service.recall(scenario.request)


def test_replay_load_and_identity_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.service.recall(scenario.request)

    with pytest.raises(RecallRequestError, match="canonical EvidencePacket ID"):
        scenario.service.replay("bad")

    monkeypatch.setattr(scenario.backend, "load_evidence_packet", _raise_runtime)
    with pytest.raises(RecallPersistenceError, match="could not be loaded"):
        scenario.service.replay(original.packet.evidence_packet_id)

    monkeypatch.undo()
    monkeypatch.setattr(
        scenario.backend,
        "load_evidence_packet",
        lambda evidence_packet_id: cast(EvidencePacket, object()),
    )
    with pytest.raises(RecallPersistenceError, match="exact EvidencePacket"):
        scenario.service.replay(original.packet.evidence_packet_id)

    monkeypatch.undo()
    monkeypatch.setattr(
        scenario.backend,
        "load_evidence_packet",
        lambda evidence_packet_id: original.packet,
    )
    with pytest.raises(RecallPersistenceError, match="identity"):
        scenario.service.replay("packet_other")


def test_replay_query_load_and_canonicalization_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.service.recall(scenario.request)
    monkeypatch.setattr(scenario.backend, "load_query_request", _raise_runtime)
    with pytest.raises(RecallPersistenceError, match="QueryRequest could not be loaded"):
        scenario.service.replay(original.packet.evidence_packet_id)

    monkeypatch.undo()
    monkeypatch.setattr(
        scenario.backend,
        "load_query_request",
        lambda query_request_id: cast(QueryRequest, object()),
    )
    with pytest.raises(RecallPersistenceError, match="QueryRequest is not canonical"):
        scenario.service.replay(original.packet.evidence_packet_id)


def test_replay_rejects_packet_query_and_access_closure_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    original = scenario.service.recall(scenario.request)
    other_request = scenario.request.model_copy(update={"question": "other question"})
    monkeypatch.setattr(
        scenario.backend,
        "load_query_request",
        lambda query_request_id: other_request,
    )
    with pytest.raises(RecallRequestError, match="does not match"):
        scenario.service.replay(original.packet.evidence_packet_id)

    monkeypatch.undo()
    different_access = original.packet.access_receipt.model_copy(
        update={"library_id": "library_other"}
    )
    drifted = EvidencePacket(
        query_request_id=original.packet.query_request_id,
        query_request_hash=original.packet.query_request_hash,
        corpus_snapshot_id=original.packet.corpus_snapshot_id,
        corpus_snapshot_hash=original.packet.corpus_snapshot_hash,
        result_status=original.packet.result_status,
        source_fragments=original.packet.source_fragments,
        coverage_report=original.packet.coverage_report,
        read_receipt=original.packet.read_receipt,
        access_receipt=different_access,
        retrieval_receipt=original.packet.retrieval_receipt,
    )
    scenario.backend.packets[drifted.evidence_packet_id] = drifted

    with pytest.raises(RecallRequestError, match="close over"):
        scenario.service.replay(drifted.evidence_packet_id)


def _request_with_question(
    request: QueryRequest,
    question: str,
    *,
    purpose: str | None = None,
    retrieval: RetrievalBudget | None = None,
) -> QueryRequest:
    return QueryRequest(
        question=question,
        library_id=request.library_id,
        collection_ids=request.collection_ids,
        corpus_snapshot_id=request.corpus_snapshot_id,
        access_policy_id=request.access_policy_id,
        purpose=purpose or request.purpose,
        exclusions=request.exclusions,
        retrieval=retrieval or request.retrieval,
        result_contract=request.result_contract,
    )


def _canonical_fts_service(scenario: Scenario) -> RecallService:
    return RecallService(
        scenario.backend,
        profile_version=current_fts_runtime_profile().profile_version,
        code_version=CODE_VERSION,
    )


def test_recall_batch_matches_sequential_packets_with_one_read_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (
        SourceSpec("alpha", "alpha evidence and common craft"),
        SourceSpec("beta", "beta evidence and common craft"),
        SourceSpec("gamma", "gamma evidence and common craft"),
    )
    sequential_scenario = _scenario(specs)
    batch_scenario = _scenario(specs)
    sequential_service = _canonical_fts_service(sequential_scenario)
    batch_service = _canonical_fts_service(batch_scenario)
    sequential_requests = tuple(
        _request_with_question(sequential_scenario.request, question)
        for question in ("alpha craft", "beta evidence", "gamma common")
    )
    batch_requests = tuple(
        _request_with_question(batch_scenario.request, question)
        for question in ("alpha craft", "beta evidence", "gamma common")
    )
    sequential = tuple(sequential_service.recall(request) for request in sequential_requests)

    class CountingSession(PermittedFtsSession):
        constructions = 0

        def __init__(self, *, fragments: tuple[FtsFragment, ...]) -> None:
            type(self).constructions += 1
            super().__init__(fragments=fragments)

    monkeypatch.setattr(recall_service_module, "PermittedFtsSession", CountingSession)
    batched = batch_service.recall_batch(batch_requests)

    assert CountingSession.constructions == 1
    assert len(batch_scenario.backend.authorization_scopes) == 1
    assert len(batch_scenario.backend.read_ids) == 1
    assert len(batch_scenario.backend.started_runs) == len(batch_requests)
    assert len(batch_scenario.backend.completed_runs) == len(batch_requests)
    assert tuple(item.packet.canonical_bytes for item in batched) == tuple(
        item.packet.canonical_bytes for item in sequential
    )

    for result in batched:
        replayed = batch_service.replay(result.packet.evidence_packet_id)
        assert replayed.packet.canonical_bytes == result.packet.canonical_bytes
        assert replayed.run.kind == "replay"


def test_recall_scope_session_matches_one_shot_and_reuses_one_authorized_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (
        SourceSpec("alpha", "alpha evidence and common craft"),
        SourceSpec("beta", "beta evidence and common craft"),
    )
    sequential_scenario = _scenario(specs)
    cached_scenario = _scenario(specs)
    sequential_service = _canonical_fts_service(sequential_scenario)
    cached_service = _canonical_fts_service(cached_scenario)
    sequential_requests = tuple(
        _request_with_question(sequential_scenario.request, question)
        for question in ("alpha craft", "beta evidence")
    )
    cached_requests = tuple(
        _request_with_question(cached_scenario.request, question)
        for question in ("alpha craft", "beta evidence")
    )
    expected = tuple(sequential_service.recall(request) for request in sequential_requests)

    class CountingSession(PermittedFtsSession):
        constructions = 0

        def __init__(self, *, fragments: tuple[FtsFragment, ...]) -> None:
            type(self).constructions += 1
            super().__init__(fragments=fragments)

    monkeypatch.setattr(recall_service_module, "PermittedFtsSession", CountingSession)
    with cached_service.open_scope_session(cached_requests[0]) as scope_session:
        actual = tuple(
            cached_service.recall_in_session(request, scope_session) for request in cached_requests
        )
        assert scope_session.stats.index_build_count == 1
        assert scope_session.stats.search_count == 2
        assert scope_session.stats.permitted_fragment_count == len(specs)

    assert scope_session.closed
    assert CountingSession.constructions == 1
    assert len(cached_scenario.backend.authorization_scopes) == 1
    assert len(cached_scenario.backend.read_ids) == 1
    assert len(cached_scenario.backend.started_runs) == 2
    assert tuple(result.packet.canonical_bytes for result in actual) == tuple(
        result.packet.canonical_bytes for result in expected
    )
    with pytest.raises(RecallRequestError, match="closed"):
        cached_service.recall_in_session(cached_requests[0], scope_session)


def test_recall_scope_session_rejects_scope_or_service_drift_before_write() -> None:
    first_scenario = _scenario((SourceSpec("alpha", "alpha public"),))
    second_scenario = _scenario((SourceSpec("alpha", "alpha public"),))
    first_service = _canonical_fts_service(first_scenario)
    second_service = _canonical_fts_service(second_scenario)
    scope_session = first_service.open_scope_session(first_scenario.request)
    wrong_scope = _request_with_question(
        first_scenario.request,
        "alpha",
        purpose="analysis",
    )

    with pytest.raises(RecallRequestError, match="differs"):
        first_service.recall_in_session(wrong_scope, scope_session)
    with pytest.raises(RecallRequestError, match="different RecallService"):
        second_service.recall_in_session(second_scenario.request, scope_session)

    assert first_scenario.backend.queries == {}
    assert second_scenario.backend.queries == {}
    scope_session.close()


def test_recall_batch_rejects_mixed_scope_before_any_write() -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    service = _canonical_fts_service(scenario)
    first = _request_with_question(scenario.request, "alpha")
    second = _request_with_question(
        scenario.request,
        "public",
        retrieval=RetrievalBudget(max_candidates=20, max_source_fragments=10),
    )

    with pytest.raises(RecallRequestError, match="one exact shared request scope"):
        service.recall_batch((first, second))

    assert scenario.backend.queries == {}
    assert scenario.backend.started_runs == []
    assert scenario.backend.authorization_scopes == []
    assert scenario.backend.read_ids == []


@pytest.mark.parametrize("requests", [(), cast(Any, [])])
def test_recall_batch_rejects_empty_or_inexact_containers_before_any_write(
    requests: tuple[QueryRequest, ...],
) -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    service = _canonical_fts_service(scenario)

    with pytest.raises(RecallRequestError, match="non-empty QueryRequest tuple"):
        service.recall_batch(requests)

    assert scenario.backend.queries == {}
    assert scenario.backend.started_runs == []


def test_recall_batch_rejects_duplicate_requests_before_any_write() -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    service = _canonical_fts_service(scenario)

    with pytest.raises(RecallRequestError, match="unique QueryRequests"):
        service.recall_batch((scenario.request, scenario.request))

    assert scenario.backend.queries == {}
    assert scenario.backend.started_runs == []


def test_recall_batch_rejects_substituted_fts_before_any_write() -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))

    with pytest.raises(RecallDependencyMismatchError, match="canonical reusable FTS"):
        scenario.service.recall_batch((scenario.request,))

    assert scenario.backend.queries == {}
    assert scenario.backend.started_runs == []


def test_recall_batch_query_persistence_failure_starts_no_runs() -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    service = _canonical_fts_service(scenario)
    scenario.backend.persist_error = True

    with pytest.raises(RecallPersistenceError, match="could not be persisted"):
        service.recall_batch((scenario.request,))

    assert scenario.backend.started_runs == []
    assert scenario.backend.authorization_scopes == []


def test_recall_batch_fails_all_runs_on_runtime_profile_drift() -> None:
    scenario = _scenario((SourceSpec("allowed", "alpha public"),))
    service = RecallService(
        scenario.backend,
        profile_version=FtsRuntimeProfile(
            sqlite_version="3.45.1",
            compile_options_hash="b" * 64,
        ).profile_version,
        code_version=CODE_VERSION,
    )
    requests = (
        _request_with_question(scenario.request, "alpha"),
        _request_with_question(scenario.request, "public"),
    )

    with pytest.raises(RecallDependencyMismatchError, match="runtime differs"):
        service.recall_batch(requests)

    assert len(scenario.backend.authorization_scopes) == 1
    assert len(scenario.backend.read_ids) == 1
    assert len(scenario.backend.failed_runs) == len(requests)
    assert scenario.backend.completed_runs == []
    assert scenario.backend.packets == {}


def test_recall_batch_completion_failure_terminalizes_every_run() -> None:
    scenario = _scenario(
        (
            SourceSpec("alpha", "alpha public"),
            SourceSpec("beta", "beta public"),
        )
    )
    service = _canonical_fts_service(scenario)
    requests = (
        _request_with_question(scenario.request, "alpha"),
        _request_with_question(scenario.request, "beta"),
    )
    scenario.backend.complete_error = True

    with pytest.raises(RecallPersistenceError, match="atomically persisted"):
        service.recall_batch(requests)

    assert len(scenario.backend.failed_runs) == len(requests)
    assert scenario.backend.completed_runs == []
    assert scenario.backend.packets == {}


def test_recall_batch_rejects_malformed_atomic_completion_without_returning_partial_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(
        (
            SourceSpec("alpha", "alpha public"),
            SourceSpec("beta", "beta public"),
        )
    )
    service = _canonical_fts_service(scenario)
    requests = (
        _request_with_question(scenario.request, "alpha"),
        _request_with_question(scenario.request, "beta"),
    )

    def incomplete_atomic_completion(
        _backend: FakeBackend,
        *,
        completions: tuple[tuple[str, EvidencePacket], ...],
    ) -> tuple[tuple[ProcessingRunRecord, EvidencePacket], ...]:
        processing_run_id, packet = completions[0]
        return (
            _backend.complete_recall_run(
                processing_run_id=processing_run_id,
                packet=packet,
            ),
        )

    monkeypatch.setattr(
        FakeBackend,
        "complete_recall_batch",
        incomplete_atomic_completion,
        raising=False,
    )

    with pytest.raises(RecallPersistenceError, match="invalid completion tuple"):
        service.recall_batch(requests)

    assert len(scenario.backend.failed_runs) == len(requests) - 1
    assert len(scenario.backend.completed_runs) == 1
    assert len(scenario.backend.packets) == 1


def test_recall_batch_returns_no_partial_results_when_one_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(
        (
            SourceSpec("alpha", "alpha public"),
            SourceSpec("beta", "beta public"),
        )
    )
    service = _canonical_fts_service(scenario)
    requests = (
        _request_with_question(scenario.request, "alpha"),
        _request_with_question(scenario.request, "beta"),
    )

    class FailingSecondSearch(PermittedFtsSession):
        constructions = 0

        def __init__(self, *, fragments: tuple[FtsFragment, ...]) -> None:
            type(self).constructions += 1
            self.search_count = 0
            super().__init__(fragments=fragments)

        def search(self, *, question: str, max_candidates: int) -> FtsSearchResult:
            self.search_count += 1
            if self.search_count == 2:
                raise RuntimeError("simulated second-query failure")
            return super().search(question=question, max_candidates=max_candidates)

    monkeypatch.setattr(
        recall_service_module,
        "PermittedFtsSession",
        FailingSecondSearch,
    )

    with pytest.raises(RecallExecutionError, match="batch recall execution"):
        service.recall_batch(requests)

    assert FailingSecondSearch.constructions == 1
    assert len(scenario.backend.authorization_scopes) == 1
    assert len(scenario.backend.read_ids) == 1
    assert len(scenario.backend.failed_runs) == len(requests)
    assert scenario.backend.completed_runs == []
    assert scenario.backend.packets == {}


def test_recall_batch_interruption_terminalizes_every_started_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(
        (
            SourceSpec("alpha", "alpha public"),
            SourceSpec("beta", "beta public"),
        )
    )
    service = _canonical_fts_service(scenario)
    requests = (
        _request_with_question(scenario.request, "alpha"),
        _request_with_question(scenario.request, "beta"),
    )

    class InterruptSecondSearch(PermittedFtsSession):
        def __init__(self, *, fragments: tuple[FtsFragment, ...]) -> None:
            self.search_count = 0
            super().__init__(fragments=fragments)

        def search(self, *, question: str, max_candidates: int) -> FtsSearchResult:
            self.search_count += 1
            if self.search_count == 2:
                raise KeyboardInterrupt
            return super().search(question=question, max_candidates=max_candidates)

    monkeypatch.setattr(
        recall_service_module,
        "PermittedFtsSession",
        InterruptSecondSearch,
    )

    with pytest.raises(KeyboardInterrupt):
        service.recall_batch(requests)

    assert len(scenario.backend.failed_runs) == len(requests)
    assert all(
        run.error_code == RecallInterruptedError.code for run in scenario.backend.failed_runs
    )
    assert scenario.backend.completed_runs == []
    assert scenario.backend.packets == {}
