"""Agent-safe orchestration over recall and the durable session journal."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from dithyramba.access import QueryExclusions
from dithyramba.persistence import (
    LibraryRepository,
    SQLiteResearchSessionRepository,
)
from dithyramba.persistence.recall import SQLiteRecallBackend
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
from .models import AgentResearchTurn, AgentSessionContext, SessionContextBudget

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
        self._recall = RecallService(
            SQLiteRecallBackend(repository),
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
        retrieval: RetrievalBudget | None = None,
    ) -> AgentResearchTurn:
        """Record a question, run exact scoped recall, and attach its durable packet."""

        session = self._sessions.get_session(session_id)
        question_event = self._append(
            session,
            kind=SessionEventKind.QUESTION_ASKED,
            summary=question,
        )
        result = self._recall.recall(
            QueryRequest(
                question=question,
                library_id=session.library_id,
                collection_ids=session.collection_ids,
                corpus_snapshot_id=session.corpus_snapshot_id,
                access_policy_id=session.access_policy_id,
                purpose=session.purpose,
                exclusions=session.scope.exclusions,
                retrieval=retrieval or RetrievalBudget(),
            )
        )
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
        evidence_event = self._append(
            session,
            kind=SessionEventKind.EVIDENCE_ATTACHED,
            summary=(
                f"Attached recall packet with {len(packet.source_fragments)} exact "
                "source fragments."
            ),
            artifact_refs=references,
        )
        return AgentResearchTurn.create(
            question_event_id=question_event.event_id,
            evidence_event_id=evidence_event.event_id,
            evidence_packet=packet,
            context=self.context(session_id),
        )

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

    def _timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise AgentResearchError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
