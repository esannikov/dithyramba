"""Deterministic append and replay for research-session event streams."""

from __future__ import annotations

from dithyramba.sessions.errors import ResearchSessionReplayError
from dithyramba.sessions.models import (
    GENESIS_EVENT_HASH,
    ResearchSession,
    ResearchSessionState,
    SessionActorKind,
    SessionArtifactReference,
    SessionEvent,
    SessionEventKind,
    SessionStatus,
    SessionTextEntry,
)


class ResearchSessionCoordinator:
    """Build and replay one immutable event stream without persistence or providers."""

    def append(
        self,
        session: ResearchSession,
        events: tuple[SessionEvent, ...],
        *,
        kind: SessionEventKind,
        actor_kind: SessionActorKind,
        actor_id: str,
        summary: str,
        artifact_refs: tuple[SessionArtifactReference, ...] = (),
        occurred_at: str,
    ) -> tuple[SessionEvent, ...]:
        state = self.replay(session, events)
        if state.status is SessionStatus.CLOSED:
            raise ResearchSessionReplayError("a closed research session cannot accept events")
        previous_hash = events[-1].event_hash if events else GENESIS_EVENT_HASH
        event = SessionEvent.create(
            session=session,
            sequence=len(events) + 1,
            previous_event_hash=previous_hash,
            kind=kind,
            actor_kind=actor_kind,
            actor_id=actor_id,
            summary=summary,
            artifact_refs=artifact_refs,
            occurred_at=occurred_at,
        )
        updated = (*events, event)
        self.replay(session, updated)
        return updated

    def replay(
        self,
        session: ResearchSession,
        events: tuple[SessionEvent, ...],
    ) -> ResearchSessionState:
        questions: list[SessionTextEntry] = []
        answer_drafts: list[SessionTextEntry] = []
        gaps: list[SessionTextEntry] = []
        rejected_paths: list[SessionTextEntry] = []
        decisions: list[SessionTextEntry] = []
        evidence_refs: list[SessionArtifactReference] = []
        candidate_refs: list[SessionArtifactReference] = []
        previous_hash = GENESIS_EVENT_HASH
        closed = False

        for expected_sequence, event in enumerate(events, start=1):
            if event.session_id != session.session_id or event.session_hash != session.session_hash:
                raise ResearchSessionReplayError("session event is bound to another session")
            if event.sequence != expected_sequence:
                raise ResearchSessionReplayError("session event sequence is not contiguous")
            if event.previous_event_hash != previous_hash:
                raise ResearchSessionReplayError("session event previous hash does not close")
            if closed:
                raise ResearchSessionReplayError("session event appears after closure")

            entry = SessionTextEntry(event_id=event.event_id, text=event.summary)
            if event.kind is SessionEventKind.QUESTION_ASKED:
                questions.append(entry)
            elif event.kind is SessionEventKind.EVIDENCE_ATTACHED:
                _extend_unique(evidence_refs, event.artifact_refs)
            elif event.kind is SessionEventKind.ANSWER_DRAFTED:
                answer_drafts.append(entry)
            elif event.kind is SessionEventKind.CANDIDATE_LINKED:
                _extend_unique(candidate_refs, event.artifact_refs)
            elif event.kind is SessionEventKind.GAP_RECORDED:
                gaps.append(entry)
            elif event.kind is SessionEventKind.PATH_REJECTED:
                rejected_paths.append(entry)
            elif event.kind is SessionEventKind.DECISION_RECORDED:
                decisions.append(entry)
            elif event.kind is SessionEventKind.SESSION_CLOSED:
                closed = True
            previous_hash = event.event_hash

        return ResearchSessionState.create(
            session=session,
            status=SessionStatus.CLOSED if closed else SessionStatus.OPEN,
            event_count=len(events),
            latest_event_hash=previous_hash,
            questions=tuple(questions),
            answer_drafts=tuple(answer_drafts),
            gaps=tuple(gaps),
            rejected_paths=tuple(rejected_paths),
            decisions=tuple(decisions),
            evidence_refs=tuple(evidence_refs),
            candidate_refs=tuple(candidate_refs),
        )


def _extend_unique(
    target: list[SessionArtifactReference],
    values: tuple[SessionArtifactReference, ...],
) -> None:
    known = {
        (item.artifact_kind, item.artifact_id, item.artifact_hash, item.role) for item in target
    }
    for item in values:
        key = (item.artifact_kind, item.artifact_id, item.artifact_hash, item.role)
        if key not in known:
            target.append(item)
            known.add(key)
