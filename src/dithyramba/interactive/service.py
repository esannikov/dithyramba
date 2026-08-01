"""Agent-safe orchestration over recall and the durable session journal."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from dithyramba.access import QueryExclusions
from dithyramba.contracts import canonical_json_bytes
from dithyramba.persistence import (
    LibraryRepository,
    SQLiteResearchSessionRepository,
)
from dithyramba.persistence.recall import SQLiteRecallBackend
from dithyramba.persistence.sessions import RecallCommandCompletion
from dithyramba.recall import QueryRequest, RecallService, RetrievalBudget
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
    AgentEvidencePacket,
    AgentResearchTurn,
    AgentSessionContext,
    SessionContextBudget,
)

_Clock = Callable[[], datetime]


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
    ) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("AgentResearchFacade requires a LibraryRepository")
        if not agent_id:
            raise AgentResearchError("agent_id must be nonblank")
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

        result = self._recall.recall(request)
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
        agent_evidence = AgentEvidencePacket.create(packet)
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

    def record_draft(self, session_id: str, text: str) -> AgentSessionContext:
        return self._record_text(session_id, SessionEventKind.ANSWER_DRAFTED, text)

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

    def _timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise AgentResearchError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
