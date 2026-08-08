"""Agent-safe orchestration over recall and the durable session journal."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime

from dithyramba.access import QueryExclusions
from dithyramba.contracts import canonical_json_bytes, sha256_hex
from dithyramba.evidence import (
    EvidenceCandidate,
    EvidenceCoverageGate,
    EvidenceCoverageResult,
    EvidenceGateDecision,
    EvidenceGateSpec,
)
from dithyramba.persistence import (
    LibraryRepository,
    SQLiteResearchSessionRepository,
)
from dithyramba.persistence.models import SourceFragmentText
from dithyramba.persistence.recall import SQLiteRecallBackend
from dithyramba.persistence.sessions import RecallCommandCompletion
from dithyramba.recall import (
    CandidateQualityAssessment,
    EvidenceFragment,
    QueryRequest,
    RecallScopeSession,
    RecallScopeSessionStats,
    RecallService,
    RetrievalBudget,
    assess_candidate_quality,
    meaningful_query_tokens,
)
from dithyramba.sessions import (
    ResearchSession,
    ResearchSessionBrief,
    ResearchSessionCoordinator,
    SessionActorKind,
    SessionArtifactKind,
    SessionArtifactReference,
    SessionEvent,
    SessionEventKind,
)

from .errors import AgentResearchError
from .models import (
    AgentAnswerPreparation,
    AgentEvidencePacket,
    AgentResearchTurn,
    AgentSessionContext,
    AgentSourceDrilldown,
    AgentSourceReference,
    SessionContextBudget,
)

_Clock = Callable[[], datetime]
_DRILLDOWN_SOURCE_LIMIT = 3
_DRILLDOWN_CANDIDATE_LIMIT = 100


class AgentResearchFacade:
    """Small Python API for an agent; human review and promotion are absent."""

    def __init__(
        self,
        repository: LibraryRepository,
        *,
        agent_id: str,
        profile_version: str,
        clock: _Clock | None = None,
        context_budget: SessionContextBudget | None = None,
        scope_session_capacity: int = 4,
    ) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("AgentResearchFacade requires a LibraryRepository")
        if not agent_id:
            raise AgentResearchError("agent_id must be nonblank")
        if type(scope_session_capacity) is not int or not 1 <= scope_session_capacity <= 32:
            raise AgentResearchError("scope_session_capacity must be an integer from 1 to 32")
        self._repository = repository
        self._agent_id = agent_id
        self._clock = clock or (lambda: datetime.now(UTC))
        self._context_budget = context_budget or SessionContextBudget()
        self._sessions = SQLiteResearchSessionRepository(repository)
        self._coordinator = ResearchSessionCoordinator()
        self._recall_backend = SQLiteRecallBackend(repository)
        self._recall = RecallService(
            self._recall_backend,
            profile_version=profile_version,
        )
        self._scope_session_capacity = scope_session_capacity
        self._scope_sessions: OrderedDict[str, RecallScopeSession] = OrderedDict()

    def __enter__(self) -> AgentResearchFacade:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close_scope_sessions()

    def close_scope_sessions(self) -> None:
        """Destroy every process-local authorized FTS capability."""

        while self._scope_sessions:
            _session_id, scope_session = self._scope_sessions.popitem(last=False)
            scope_session.close()

    def cache_stats(self, session_id: str) -> RecallScopeSessionStats | None:
        """Return non-canonical runtime stats without opening or rebuilding a cache."""

        scope_session = self._scope_sessions.get(session_id)
        return None if scope_session is None else scope_session.stats

    def open(
        self,
        *,
        brief: ResearchSessionBrief,
        corpus_snapshot_id: str,
        access_policy_id: str,
        purpose: str,
        collection_ids: tuple[str, ...],
        exclusions: QueryExclusions | None = None,
    ) -> AgentSessionContext:
        """Open one model-authored session bound to repository-derived snapshot hash."""

        snapshot = self._repository.get_corpus_snapshot(corpus_snapshot_id)
        session = ResearchSession.create(
            brief=brief,
            library_id=self._repository.library_id,
            corpus_snapshot_id=corpus_snapshot_id,
            snapshot_hash=snapshot.manifest_hash,
            access_policy_id=access_policy_id,
            purpose=purpose,
            collection_ids=collection_ids,
            exclusions=exclusions,
            created_by_kind=SessionActorKind.MODEL,
            created_by_id=self._agent_id,
            created_at=self._timestamp(),
        )
        self._sessions.persist_session(session)
        return self.context(session.session_id)

    def recall(
        self,
        session_id: str,
        question: str,
        *,
        command_id: str,
        retrieval: RetrievalBudget | None = None,
    ) -> AgentResearchTurn:
        """Run or exactly replay one idempotent, source-grounded recall command."""

        session = self._sessions.get_session(session_id)
        request = QueryRequest(
            question=question,
            library_id=session.library_id,
            collection_ids=session.collection_ids,
            corpus_snapshot_id=session.corpus_snapshot_id,
            access_policy_id=session.access_policy_id,
            purpose=session.purpose,
            exclusions=session.scope.exclusions,
            retrieval=retrieval or RetrievalBudget(),
        )
        started_at = self._timestamp()
        start = self._sessions.begin_recall_command(
            session_id=session_id,
            command_id=command_id,
            request=request,
            actor_id=self._agent_id,
            occurred_at=started_at,
        )
        completed = self._sessions.load_recall_command_completion(command_id)
        if completed is not None:
            if (
                completed.command_hash != start.command_hash
                or completed.question_event_id != start.question_event.event_id
            ):
                raise AgentResearchError("completed recall command disagrees with its start")
            return self._turn_from_completion(completed)

        scope_session = self._scope_session(session.session_id, request)
        result = self._recall.recall_in_session(request, scope_session)
        packet = result.packet
        references = (
            SessionArtifactReference(
                artifact_kind=SessionArtifactKind.EVIDENCE_PACKET,
                artifact_id=packet.evidence_packet_id,
                artifact_hash=packet.packet_hash,
                role="recall",
            ),
            *tuple(
                SessionArtifactReference(
                    artifact_kind=SessionArtifactKind.SOURCE_FRAGMENT,
                    artifact_id=fragment.source_fragment_id,
                    artifact_hash=fragment.text_sha256,
                    role="supports",
                )
                for fragment in packet.source_fragments
            ),
        )
        evidence_event = self._build_event(
            session,
            kind=SessionEventKind.EVIDENCE_ATTACHED,
            summary=(
                f"Attached recall packet with {len(packet.source_fragments)} exact "
                "source fragments."
            ),
            artifact_refs=references,
        )
        events = self._sessions.list_events(session_id)
        state = self._coordinator.replay(session, (*events, evidence_event))
        projected_context = AgentSessionContext.create(
            session=session,
            state=state,
            budget=self._context_budget,
        )
        agent_evidence = AgentEvidencePacket.create(
            packet,
            source_references=self._source_references(packet.source_fragments),
        )
        turn = AgentResearchTurn.create(
            command_id=command_id,
            question_event_id=start.question_event.event_id,
            evidence_event_id=evidence_event.event_id,
            evidence_packet=agent_evidence,
            context=projected_context,
        )
        completion = self._sessions.complete_recall_command(
            start=start,
            evidence_event=evidence_event,
            evidence_packet_id=packet.evidence_packet_id,
            evidence_packet_hash=packet.packet_hash,
            agent_evidence_json=canonical_json_bytes(agent_evidence.model_dump(mode="json")).decode(
                "utf-8"
            ),
            context_json=canonical_json_bytes(projected_context.model_dump(mode="json")).decode(
                "utf-8"
            ),
            turn_hash=turn.turn_hash,
            occurred_at=evidence_event.occurred_at,
        )
        return self._turn_from_completion(completion)

    def prepare_answer(
        self,
        session_id: str,
        *,
        evidence_event_id: str,
        gate_spec: EvidenceGateSpec,
    ) -> AgentAnswerPreparation:
        """Filter, drill down and gate one recall turn before answer generation."""

        if type(gate_spec) is not EvidenceGateSpec:
            raise TypeError("prepare_answer requires an exact EvidenceGateSpec")
        session = self._sessions.get_session(session_id)
        evidence_event = next(
            (
                event
                for event in self._sessions.list_events(session_id)
                if event.event_id == evidence_event_id
            ),
            None,
        )
        if evidence_event is None or evidence_event.kind is not SessionEventKind.EVIDENCE_ATTACHED:
            raise AgentResearchError("answer preparation requires a session evidence event")
        completion = self._sessions.load_recall_completion_for_evidence_event(evidence_event_id)
        if completion is None:
            raise AgentResearchError("answer preparation cannot resolve the recall completion")
        binding = self._recall_backend.load_evidence_packet_binding(completion.evidence_packet_id)
        request = binding.query_request
        self._require_session_request(session, request)
        if gate_spec.question != request.question:
            raise AgentResearchError("EvidenceGateSpec question differs from the recall question")
        if completion.agent_evidence_json is None:
            packet = self._recall_backend.load_evidence_packet(completion.evidence_packet_id)
            agent_evidence = AgentEvidencePacket.create(
                packet,
                source_references=self._source_references(packet.source_fragments),
            )
        else:
            agent_evidence = AgentEvidencePacket.model_validate_json(completion.agent_evidence_json)
        if (
            agent_evidence.evidence_packet_id != completion.evidence_packet_id
            or agent_evidence.evidence_packet_hash != completion.evidence_packet_hash
            or binding.packet_hash != completion.evidence_packet_hash
        ):
            raise AgentResearchError("answer preparation packet binding drifted")

        broad_items = tuple(
            (fragment, reference)
            for fragment, reference in zip(
                agent_evidence.source_fragments,
                agent_evidence.source_references,
                strict=True,
            )
        )
        candidates, assessments = self._quality_eligible_candidates(
            broad_items,
            question=request.question,
        )
        gate_result = EvidenceCoverageGate(gate_spec).evaluate(candidates)
        drilldown_projection: AgentSourceDrilldown | None = None

        if gate_result.decision in {
            EvidenceGateDecision.PARTIAL,
            EvidenceGateDecision.INSUFFICIENT,
        }:
            source_ids = _first_source_ids(
                tuple(reference.source_id for _fragment, reference in broad_items),
                limit=_DRILLDOWN_SOURCE_LIMIT,
            )
            drilldown_query = _drilldown_query(gate_spec, gate_result)
            if source_ids and drilldown_query:
                scope_session = self._scope_session(session_id, request)
                local = self._recall.source_local_drilldown(
                    request,
                    scope_session,
                    question=drilldown_query,
                    source_ids=source_ids,
                    max_candidates=_DRILLDOWN_CANDIDATE_LIMIT,
                )
                local_items = tuple(
                    (fragment, self._source_reference_from_text(fragment))
                    for fragment in local.fragments
                )
                local_candidates, local_assessments = self._quality_eligible_candidates(
                    local_items,
                    question=drilldown_query,
                    starting_rank=len(candidates) + 1,
                    excluded_fragment_ids={item.source_fragment_id for item in assessments},
                )
                candidates = (*candidates, *local_candidates)
                assessments = (*assessments, *local_assessments)
                candidates = tuple(
                    candidate.model_copy(update={"rank": rank})
                    for rank, candidate in enumerate(candidates, start=1)
                )
                gate_result = EvidenceCoverageGate(gate_spec).evaluate(candidates)
                drilldown_projection = AgentSourceDrilldown.create(
                    question=drilldown_query,
                    source_ids=local.source_ids,
                    retrieval_result_hash=local.result.result_hash,
                    candidate_count=local.result.candidate_count,
                    selected_fragment_ids=tuple(
                        item.source_fragment_id for item in local_candidates
                    ),
                    filtered_out_count=sum(1 for item in local_assessments if not item.admitted),
                )

        if gate_result.decision is EvidenceGateDecision.READY:
            candidates, assessments, gate_result = _compact_ready_candidates(
                gate_spec=gate_spec,
                candidates=candidates,
                assessments=assessments,
            )
            if drilldown_projection is not None:
                selected_ids = {item.source_fragment_id for item in candidates}
                drilldown_projection = AgentSourceDrilldown.create(
                    question=drilldown_projection.question,
                    source_ids=drilldown_projection.source_ids,
                    retrieval_result_hash=drilldown_projection.retrieval_result_hash,
                    candidate_count=drilldown_projection.candidate_count,
                    selected_fragment_ids=tuple(
                        fragment_id
                        for fragment_id in drilldown_projection.selected_fragment_ids
                        if fragment_id in selected_ids
                    ),
                    filtered_out_count=drilldown_projection.filtered_out_count,
                )

        return AgentAnswerPreparation.create(
            session_id=session_id,
            evidence_event_id=evidence_event_id,
            evidence_packet_id=completion.evidence_packet_id,
            evidence_packet_hash=completion.evidence_packet_hash,
            gate_spec=gate_spec,
            gate_result=gate_result,
            candidates=candidates,
            quality_assessments=assessments,
            drilldown=drilldown_projection,
        )

    def record_draft(
        self,
        session_id: str,
        text: str,
        *,
        preparation: AgentAnswerPreparation,
    ) -> AgentSessionContext:
        """Record an answer draft only after exact gate preparation is replayed."""

        if type(preparation) is not AgentAnswerPreparation:
            raise TypeError("record_draft requires an exact AgentAnswerPreparation")
        if preparation.session_id != session_id:
            raise AgentResearchError("answer preparation belongs to another session")
        verified = self.prepare_answer(
            session_id,
            evidence_event_id=preparation.evidence_event_id,
            gate_spec=preparation.gate_spec,
        )
        if verified != preparation:
            raise AgentResearchError("answer preparation no longer replays exactly")
        if verified.response_mode != "answer":
            raise AgentResearchError("EvidenceCoverageGate did not admit a source-backed answer")
        session = self._sessions.get_session(session_id)
        matched_ids = _matched_fragment_ids(verified)
        candidates_by_id = {
            candidate.source_fragment_id: candidate for candidate in verified.candidates
        }
        references = (
            SessionArtifactReference(
                artifact_kind=SessionArtifactKind.EVIDENCE_PACKET,
                artifact_id=verified.evidence_packet_id,
                artifact_hash=verified.evidence_packet_hash,
                role="answer_gate",
            ),
            *tuple(
                SessionArtifactReference(
                    artifact_kind=SessionArtifactKind.SOURCE_FRAGMENT,
                    artifact_id=fragment_id,
                    artifact_hash=sha256_hex(candidates_by_id[fragment_id].text.encode("utf-8")),
                    role="admitted_support",
                )
                for fragment_id in matched_ids[:31]
            ),
        )
        self._append(
            session,
            kind=SessionEventKind.ANSWER_DRAFTED,
            summary=text,
            artifact_refs=references,
        )
        return self.context(session_id)

    def record_gap(self, session_id: str, text: str) -> AgentSessionContext:
        return self._record_text(session_id, SessionEventKind.GAP_RECORDED, text)

    def reject_path(self, session_id: str, text: str) -> AgentSessionContext:
        return self._record_text(session_id, SessionEventKind.PATH_REJECTED, text)

    def attach_evidence(
        self,
        session_id: str,
        *,
        summary: str,
        references: tuple[SessionArtifactReference, ...],
    ) -> AgentSessionContext:
        session = self._sessions.get_session(session_id)
        self._append(
            session,
            kind=SessionEventKind.EVIDENCE_ATTACHED,
            summary=summary,
            artifact_refs=references,
        )
        return self.context(session_id)

    def link_candidates(
        self,
        session_id: str,
        *,
        summary: str,
        references: tuple[SessionArtifactReference, ...],
    ) -> AgentSessionContext:
        session = self._sessions.get_session(session_id)
        self._append(
            session,
            kind=SessionEventKind.CANDIDATE_LINKED,
            summary=summary,
            artifact_refs=references,
        )
        return self.context(session_id)

    def context(self, session_id: str) -> AgentSessionContext:
        session = self._sessions.get_session(session_id)
        state = self._sessions.get_state(session_id)
        return AgentSessionContext.create(
            session=session,
            state=state,
            budget=self._context_budget,
        )

    def _record_text(
        self,
        session_id: str,
        kind: SessionEventKind,
        text: str,
    ) -> AgentSessionContext:
        session = self._sessions.get_session(session_id)
        self._append(session, kind=kind, summary=text)
        return self.context(session_id)

    def _scope_session(
        self,
        session_id: str,
        request: QueryRequest,
    ) -> RecallScopeSession:
        existing = self._scope_sessions.pop(session_id, None)
        if existing is not None:
            self._scope_sessions[session_id] = existing
            return existing
        created = self._recall.open_scope_session(request)
        self._scope_sessions[session_id] = created
        while len(self._scope_sessions) > self._scope_session_capacity:
            _evicted_id, evicted = self._scope_sessions.popitem(last=False)
            evicted.close()
        return created

    def _append(
        self,
        session: ResearchSession,
        *,
        kind: SessionEventKind,
        summary: str,
        artifact_refs: tuple[SessionArtifactReference, ...] = (),
    ) -> SessionEvent:
        events = self._sessions.list_events(session.session_id)
        updated = self._coordinator.append(
            session,
            events,
            kind=kind,
            actor_kind=SessionActorKind.MODEL,
            actor_id=self._agent_id,
            summary=summary,
            artifact_refs=artifact_refs,
            occurred_at=self._timestamp(),
        )
        return self._sessions.append_event(updated[-1])

    def _build_event(
        self,
        session: ResearchSession,
        *,
        kind: SessionEventKind,
        summary: str,
        artifact_refs: tuple[SessionArtifactReference, ...] = (),
    ) -> SessionEvent:
        events = self._sessions.list_events(session.session_id)
        return self._coordinator.append(
            session,
            events,
            kind=kind,
            actor_kind=SessionActorKind.MODEL,
            actor_id=self._agent_id,
            summary=summary,
            artifact_refs=artifact_refs,
            occurred_at=self._timestamp(),
        )[-1]

    def _turn_from_completion(
        self,
        completion: RecallCommandCompletion,
    ) -> AgentResearchTurn:
        turn_schema = AgentResearchTurn.SCHEMA
        if completion.agent_evidence_json is None:
            packet = self._recall_backend.load_evidence_packet(completion.evidence_packet_id)
            if packet.packet_hash != completion.evidence_packet_hash:
                raise AgentResearchError("completed recall packet hash drifted")
            agent_evidence = AgentEvidencePacket.create(packet)
            turn_schema = AgentResearchTurn.LEGACY_SCHEMA
        else:
            agent_evidence = AgentEvidencePacket.model_validate_json(completion.agent_evidence_json)
            if (
                agent_evidence.evidence_packet_id != completion.evidence_packet_id
                or agent_evidence.evidence_packet_hash != completion.evidence_packet_hash
            ):
                raise AgentResearchError("completed agent evidence projection drifted")
        context = AgentSessionContext.model_validate_json(completion.context_json)
        turn = AgentResearchTurn.create(
            command_id=completion.command_id,
            question_event_id=completion.question_event_id,
            evidence_event_id=completion.evidence_event_id,
            evidence_packet=agent_evidence,
            context=context,
            schema_id=turn_schema,
        )
        if turn.turn_hash != completion.turn_hash:
            raise AgentResearchError("completed recall turn hash drifted")
        return turn

    def _source_references(
        self,
        fragments: tuple[EvidenceFragment, ...],
    ) -> tuple[AgentSourceReference, ...]:
        references: list[AgentSourceReference] = []
        for fragment in fragments:
            if type(fragment) is not EvidenceFragment:
                raise AgentResearchError("agent evidence contains an invalid fragment")
            source_version = self._repository.get_source_version(fragment.source_version_id)
            source = self._repository.get_source(source_version.source_id)
            if (
                fragment.source_family_id is not None
                and fragment.source_family_id != source.source_family_id
            ):
                raise AgentResearchError("agent evidence source lineage drifted")
            references.append(
                AgentSourceReference(
                    source_fragment_id=fragment.source_fragment_id,
                    source_version_id=fragment.source_version_id,
                    source_id=source.source_id,
                    source_family_id=source.source_family_id,
                    root_source_id=source.root_source_id,
                    family_role=source.family_role.value,
                    title=source.title,
                    canonical_uri=source.canonical_uri,
                )
            )
        return tuple(references)

    def _source_reference_from_text(
        self,
        fragment: SourceFragmentText,
    ) -> AgentSourceReference:
        source_version = self._repository.get_source_version(fragment.source_version_id)
        if source_version.source_id != fragment.source_id:
            raise AgentResearchError("source-local fragment version lineage drifted")
        source = self._repository.get_source(fragment.source_id)
        return AgentSourceReference(
            source_fragment_id=fragment.source_fragment_id,
            source_version_id=fragment.source_version_id,
            source_id=source.source_id,
            source_family_id=source.source_family_id,
            root_source_id=source.root_source_id,
            family_role=source.family_role.value,
            title=source.title,
            canonical_uri=source.canonical_uri,
        )

    def _quality_eligible_candidates(
        self,
        items: tuple[
            tuple[EvidenceFragment | SourceFragmentText, AgentSourceReference],
            ...,
        ],
        *,
        question: str,
        starting_rank: int = 1,
        excluded_fragment_ids: set[str] | None = None,
    ) -> tuple[tuple[EvidenceCandidate, ...], tuple[CandidateQualityAssessment, ...]]:
        excluded = excluded_fragment_ids or set()
        candidates: list[EvidenceCandidate] = []
        assessments: list[CandidateQualityAssessment] = []
        for fragment, reference in items:
            if fragment.source_fragment_id in excluded:
                continue
            source_fragment = self._repository.get_source_fragment(fragment.source_fragment_id)
            address = (
                fragment.source_address.payload()
                if isinstance(fragment, EvidenceFragment)
                else json.loads(fragment.source_address_json)
            )
            assessment = assess_candidate_quality(
                source_fragment_id=fragment.source_fragment_id,
                question=question,
                text=fragment.text,
                fragment_kind=source_fragment.fragment_kind,
                source_address=address,
            )
            assessments.append(assessment)
            if not assessment.admitted:
                continue
            source = self._repository.get_source(reference.source_id)
            candidates.append(
                EvidenceCandidate(
                    rank=starting_rank + len(candidates),
                    source_fragment_id=fragment.source_fragment_id,
                    source_id=reference.source_id,
                    text=fragment.text,
                    source_address=address,
                    source_kind=source.media_type,
                    source_family=reference.source_family_id,
                    authority="unclassified",
                    independence_group=reference.root_source_id,
                    evidence_tags=tuple(
                        sorted(
                            {
                                source_fragment.fragment_kind,
                                source.media_type,
                                reference.family_role,
                            }
                        )
                    ),
                )
            )
        return tuple(candidates), tuple(assessments)

    @staticmethod
    def _require_session_request(session: ResearchSession, request: QueryRequest) -> None:
        if (
            request.library_id != session.library_id
            or request.collection_ids != session.collection_ids
            or request.corpus_snapshot_id != session.corpus_snapshot_id
            or request.access_policy_id != session.access_policy_id
            or request.purpose != session.purpose
            or request.exclusions != session.scope.exclusions
        ):
            raise AgentResearchError("answer preparation request differs from its session")

    def _timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise AgentResearchError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _first_source_ids(values: tuple[str, ...], *, limit: int) -> tuple[str, ...]:
    selected: list[str] = []
    for value in values:
        if value not in selected:
            selected.append(value)
        if len(selected) == limit:
            break
    return tuple(selected)


def _drilldown_query(gate_spec: EvidenceGateSpec, gate_result: object) -> str | None:
    missing_ids = set(getattr(gate_result, "missing_requirement_ids", ()))
    raw = " ".join(
        anchor
        for requirement in gate_spec.requirements
        if requirement.evidence_requirement_id in missing_ids
        for group in requirement.anchor_groups
        for anchor in group
    )
    tokens = (*meaningful_query_tokens(raw), *meaningful_query_tokens(gate_spec.question))
    selected: list[str] = []
    for token in tokens:
        if token not in selected:
            selected.append(token)
        if len(selected) == 48:
            break
    return " ".join(selected) or None


def _compact_ready_candidates(
    *,
    gate_spec: EvidenceGateSpec,
    candidates: tuple[EvidenceCandidate, ...],
    assessments: tuple[CandidateQualityAssessment, ...],
) -> tuple[
    tuple[EvidenceCandidate, ...],
    tuple[CandidateQualityAssessment, ...],
    EvidenceCoverageResult,
]:
    """Return a deterministic inclusion-minimal candidate set that stays ready."""

    gate = EvidenceCoverageGate(gate_spec)
    initial = gate.evaluate(candidates)
    if initial.decision is not EvidenceGateDecision.READY:
        return candidates, assessments, initial
    matched_ids = {
        fragment_id
        for requirement in initial.requirements
        for fragment_id in requirement.matched_fragment_ids
    }
    selected = [
        candidate for candidate in candidates if candidate.source_fragment_id in matched_ids
    ]
    for candidate in tuple(reversed(selected)):
        trial = [
            item for item in selected if item.source_fragment_id != candidate.source_fragment_id
        ]
        if gate.evaluate(tuple(trial)).decision is EvidenceGateDecision.READY:
            selected = trial
    compact = tuple(
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(selected, start=1)
    )
    selected_ids = {item.source_fragment_id for item in compact}
    compact_assessments = tuple(
        assessment
        for assessment in assessments
        if not assessment.admitted or assessment.source_fragment_id in selected_ids
    )
    result = gate.evaluate(compact)
    if result.decision is not EvidenceGateDecision.READY:
        raise AgentResearchError("ready candidate compaction changed the gate decision")
    return compact, compact_assessments, result


def _matched_fragment_ids(preparation: AgentAnswerPreparation) -> tuple[str, ...]:
    selected: list[str] = []
    for requirement in preparation.gate_result.requirements:
        for fragment_id in requirement.matched_fragment_ids:
            if fragment_id not in selected:
                selected.append(fragment_id)
    return tuple(selected)
