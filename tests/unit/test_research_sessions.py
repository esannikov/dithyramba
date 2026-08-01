from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dithyramba.sessions import (
    GENESIS_EVENT_HASH,
    ResearchSession,
    ResearchSessionBrief,
    ResearchSessionCoordinator,
    ResearchSessionReplayError,
    ResearchSessionState,
    SessionActorKind,
    SessionArtifactKind,
    SessionArtifactReference,
    SessionEvent,
    SessionEventKind,
    SessionStatus,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
NOW = "2026-08-01T10:00:00.000000Z"


def _brief() -> ResearchSessionBrief:
    return ResearchSessionBrief.create(
        question="How should the project stage a disputed historical episode?",
        intended_use="directorial research",
        success_criteria=("locate primary evidence", "separate fact from hypothesis"),
        boundaries=("do not treat chat as evidence",),
    )


def _session(*, created_at: str = NOW) -> ResearchSession:
    return ResearchSession.create(
        brief=_brief(),
        library_id="library_main",
        corpus_snapshot_id="snapshot_august",
        snapshot_hash=HASH_A,
        access_policy_id="policy_research",
        purpose="research",
        collection_ids=("collection_secondary", "collection_primary"),
        created_by_kind=SessionActorKind.HUMAN,
        created_by_id="operator:eugene",
        created_at=created_at,
    )


def _reference(
    kind: SessionArtifactKind = SessionArtifactKind.EVIDENCE_PACKET,
    *,
    role: str = "supports",
) -> SessionArtifactReference:
    artifact_id = {
        SessionArtifactKind.SOURCE_FRAGMENT: "fragment_alpha",
        SessionArtifactKind.EVIDENCE_PACKET: "packet_alpha",
        SessionArtifactKind.MEMORY_PACKET: "memory_packet_" + "1" * 32,
        SessionArtifactKind.EVIDENCE_LINK: "evidence_alpha",
        SessionArtifactKind.STATEMENT: "statement_alpha",
        SessionArtifactKind.CONCEPT: "concept_alpha",
        SessionArtifactKind.ENTITY: "entity_alpha",
        SessionArtifactKind.IDEA_TRACE: "idea_trace_" + "2" * 32,
        SessionArtifactKind.RESEARCH_ANSWER: "research_answer_" + "3" * 32,
        SessionArtifactKind.ANSWER_PROJECTION: "answer_projection_" + "4" * 32,
    }[kind]
    return SessionArtifactReference(
        artifact_kind=kind,
        artifact_id=artifact_id,
        artifact_hash=HASH_B,
        role=role,
    )


def test_brief_and_session_are_content_addressed_and_canonical() -> None:
    brief = _brief()
    session = _session()

    assert brief.brief_id.startswith("session_brief_")
    assert session.session_id.startswith("research_session_")
    assert session.collection_ids == ("collection_primary", "collection_secondary")
    assert ResearchSessionBrief.model_validate_json(brief.model_dump_json()) == brief
    assert ResearchSession.model_validate_json(session.model_dump_json()) == session


def test_brief_rejects_duplicate_lists_unknown_fields_and_tampering() -> None:
    with pytest.raises(ValidationError, match="success_criteria must be unique"):
        ResearchSessionBrief.create(
            question="Question",
            intended_use="Use",
            success_criteria=("same", "same"),
        )
    with pytest.raises(ValidationError, match="boundaries must be unique"):
        ResearchSessionBrief.create(
            question="Question",
            intended_use="Use",
            success_criteria=("criterion",),
            boundaries=("same", "same"),
        )

    payload = _brief().model_dump(mode="json")
    payload["unknown"] = "value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResearchSessionBrief.model_validate(payload)

    payload.pop("unknown")
    payload["question"] = "Changed after hashing"
    with pytest.raises(ValidationError, match="brief_id does not match"):
        ResearchSessionBrief.model_validate_json(json.dumps(payload))

    for field, value, message in (
        ("schema_id", "wrong", "schema_id must be"),
        ("brief_hash", HASH_B, "brief_hash does not match"),
    ):
        payload = _brief().model_dump(mode="json")
        payload[field] = value
        with pytest.raises(ValidationError, match=message):
            ResearchSessionBrief.model_validate_json(json.dumps(payload))


def test_session_rejects_invalid_actor_collection_and_identity() -> None:
    with pytest.raises(ValidationError, match="opened by a human or model"):
        ResearchSession.create(
            brief=_brief(),
            library_id="library_main",
            corpus_snapshot_id="snapshot_august",
            snapshot_hash=HASH_A,
            access_policy_id="policy_research",
            purpose="research",
            collection_ids=("collection_primary",),
            created_by_kind=SessionActorKind.DETERMINISTIC,
            created_by_id="system",
            created_at=NOW,
        )

    payload = _session().model_dump(mode="json")
    payload["collection_ids"] = ["wrong"]
    with pytest.raises(ValidationError, match="collection_ prefix"):
        ResearchSession.model_validate_json(json.dumps(payload))

    payload = _session().model_dump(mode="json")
    payload["collection_ids"] = ["collection_primary", "collection_primary"]
    with pytest.raises(ValidationError, match="sorted and unique"):
        ResearchSession.model_validate_json(json.dumps(payload))

    for field, value, message in (
        ("schema_id", "wrong", "schema_id must be"),
        ("session_id", "research_session_" + "f" * 32, "session_id does not match"),
    ):
        payload = _session().model_dump(mode="json")
        payload[field] = value
        with pytest.raises(ValidationError, match=message):
            ResearchSession.model_validate_json(json.dumps(payload))

    payload = _session().model_dump(mode="json")
    payload["session_hash"] = HASH_B
    with pytest.raises(ValidationError, match="session_hash does not match"):
        ResearchSession.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("kind", tuple(SessionArtifactKind))
def test_all_artifact_reference_kinds_validate_their_prefix(kind: SessionArtifactKind) -> None:
    assert _reference(kind).artifact_kind is kind


def test_artifact_reference_rejects_wrong_prefix() -> None:
    with pytest.raises(ValidationError, match="must start with statement_"):
        SessionArtifactReference(
            artifact_kind=SessionArtifactKind.STATEMENT,
            artifact_id="concept_wrong",
            artifact_hash=HASH_A,
            role="candidate",
        )


def test_event_contract_rejects_role_and_artifact_boundary_violations() -> None:
    session = _session()
    with pytest.raises(ValidationError, match="requires a human actor"):
        SessionEvent.create(
            session=session,
            sequence=1,
            previous_event_hash=GENESIS_EVENT_HASH,
            kind=SessionEventKind.DECISION_RECORDED,
            actor_kind=SessionActorKind.MODEL,
            actor_id="model:test",
            summary="Accept the candidate",
            occurred_at=NOW,
        )
    with pytest.raises(ValidationError, match="requires at least one"):
        SessionEvent.create(
            session=session,
            sequence=1,
            previous_event_hash=GENESIS_EVENT_HASH,
            kind=SessionEventKind.EVIDENCE_ATTACHED,
            actor_kind=SessionActorKind.MODEL,
            actor_id="model:test",
            summary="Found evidence",
            occurred_at=NOW,
        )
    with pytest.raises(ValidationError, match="disallowed artifact kinds: statement"):
        SessionEvent.create(
            session=session,
            sequence=1,
            previous_event_hash=GENESIS_EVENT_HASH,
            kind=SessionEventKind.EVIDENCE_ATTACHED,
            actor_kind=SessionActorKind.MODEL,
            actor_id="model:test",
            summary="Wrongly labelled evidence",
            artifact_refs=(_reference(SessionArtifactKind.STATEMENT),),
            occurred_at=NOW,
        )
    with pytest.raises(ValidationError, match="disallowed artifact kinds: evidence_packet"):
        SessionEvent.create(
            session=session,
            sequence=1,
            previous_event_hash=GENESIS_EVENT_HASH,
            kind=SessionEventKind.CANDIDATE_LINKED,
            actor_kind=SessionActorKind.MODEL,
            actor_id="model:test",
            summary="Wrongly labelled candidate",
            artifact_refs=(_reference(),),
            occurred_at=NOW,
        )


def test_event_contract_rejects_duplicate_refs_closure_refs_and_tampering() -> None:
    session = _session()
    reference = _reference()
    with pytest.raises(ValidationError, match="artifact_refs must be unique"):
        SessionEvent.create(
            session=session,
            sequence=1,
            previous_event_hash=GENESIS_EVENT_HASH,
            kind=SessionEventKind.EVIDENCE_ATTACHED,
            actor_kind=SessionActorKind.MODEL,
            actor_id="model:test",
            summary="Duplicate",
            artifact_refs=(reference, reference),
            occurred_at=NOW,
        )
    with pytest.raises(ValidationError, match="session_closed cannot attach artifacts"):
        SessionEvent.create(
            session=session,
            sequence=1,
            previous_event_hash=GENESIS_EVENT_HASH,
            kind=SessionEventKind.SESSION_CLOSED,
            actor_kind=SessionActorKind.HUMAN,
            actor_id="operator:eugene",
            summary="Closed",
            artifact_refs=(reference,),
            occurred_at=NOW,
        )

    valid = SessionEvent.create(
        session=session,
        sequence=1,
        previous_event_hash=GENESIS_EVENT_HASH,
        kind=SessionEventKind.QUESTION_ASKED,
        actor_kind=SessionActorKind.HUMAN,
        actor_id="operator:eugene",
        summary="Original question",
        occurred_at=NOW,
    )
    payload = valid.model_dump(mode="json")
    payload["summary"] = "Changed"
    with pytest.raises(ValidationError, match="event_id does not match"):
        SessionEvent.model_validate_json(json.dumps(payload))

    for field, value, message in (
        ("schema_id", "wrong", "schema_id must be"),
        ("event_hash", HASH_A, "event_hash does not match"),
    ):
        payload = valid.model_dump(mode="json")
        payload[field] = value
        with pytest.raises(ValidationError, match=message):
            SessionEvent.model_validate_json(json.dumps(payload))


def test_replay_builds_compact_state_and_deduplicates_references() -> None:
    session = _session()
    coordinator = ResearchSessionCoordinator()
    events: tuple[SessionEvent, ...] = ()
    evidence = _reference()
    candidate = _reference(SessionArtifactKind.STATEMENT, role="candidate")
    plan = (
        (SessionEventKind.QUESTION_ASKED, "What happened?", (), SessionActorKind.HUMAN),
        (SessionEventKind.EVIDENCE_ATTACHED, "Attach packet", (evidence,), SessionActorKind.MODEL),
        (SessionEventKind.EVIDENCE_ATTACHED, "Reuse packet", (evidence,), SessionActorKind.MODEL),
        (SessionEventKind.ANSWER_DRAFTED, "A bounded draft", (), SessionActorKind.MODEL),
        (
            SessionEventKind.CANDIDATE_LINKED,
            "Candidate claim",
            (candidate,),
            SessionActorKind.MODEL,
        ),
        (SessionEventKind.GAP_RECORDED, "Missing date", (), SessionActorKind.MODEL),
        (SessionEventKind.PATH_REJECTED, "Blog was unsourced", (), SessionActorKind.MODEL),
        (SessionEventKind.DECISION_RECORDED, "Keep the claim open", (), SessionActorKind.HUMAN),
        (
            SessionEventKind.SESSION_CLOSED,
            "Operator closed the session",
            (),
            SessionActorKind.HUMAN,
        ),
    )
    for index, (kind, summary, refs, actor) in enumerate(plan):
        events = coordinator.append(
            session,
            events,
            kind=kind,
            actor_kind=actor,
            actor_id="operator:eugene" if actor is SessionActorKind.HUMAN else "model:test",
            summary=summary,
            artifact_refs=refs,
            occurred_at=f"2026-08-01T10:00:{index:02d}.000000Z",
        )

    state = coordinator.replay(session, events)
    assert state.status is SessionStatus.CLOSED
    assert state.event_count == 9
    assert tuple(item.text for item in state.questions) == ("What happened?",)
    assert tuple(item.text for item in state.answer_drafts) == ("A bounded draft",)
    assert tuple(item.text for item in state.gaps) == ("Missing date",)
    assert tuple(item.text for item in state.rejected_paths) == ("Blog was unsourced",)
    assert tuple(item.text for item in state.decisions) == ("Keep the claim open",)
    assert state.evidence_refs == (evidence,)
    assert state.candidate_refs == (candidate,)
    assert ResearchSessionState.model_validate_json(state.model_dump_json()) == state


def test_replay_rejects_foreign_sequence_hash_and_post_closure_events() -> None:
    session = _session()
    foreign = _session(created_at="2026-08-01T11:00:00.000000Z")
    coordinator = ResearchSessionCoordinator()
    foreign_event = SessionEvent.create(
        session=foreign,
        sequence=1,
        previous_event_hash=GENESIS_EVENT_HASH,
        kind=SessionEventKind.QUESTION_ASKED,
        actor_kind=SessionActorKind.HUMAN,
        actor_id="operator:eugene",
        summary="Foreign",
        occurred_at=NOW,
    )
    with pytest.raises(ResearchSessionReplayError, match="another session"):
        coordinator.replay(session, (foreign_event,))

    wrong_sequence = SessionEvent.create(
        session=session,
        sequence=2,
        previous_event_hash=GENESIS_EVENT_HASH,
        kind=SessionEventKind.QUESTION_ASKED,
        actor_kind=SessionActorKind.HUMAN,
        actor_id="operator:eugene",
        summary="Skipped sequence",
        occurred_at=NOW,
    )
    with pytest.raises(ResearchSessionReplayError, match="not contiguous"):
        coordinator.replay(session, (wrong_sequence,))

    wrong_hash = SessionEvent.create(
        session=session,
        sequence=1,
        previous_event_hash=HASH_A,
        kind=SessionEventKind.QUESTION_ASKED,
        actor_kind=SessionActorKind.HUMAN,
        actor_id="operator:eugene",
        summary="Wrong predecessor",
        occurred_at=NOW,
    )
    with pytest.raises(ResearchSessionReplayError, match="previous hash"):
        coordinator.replay(session, (wrong_hash,))

    closed = SessionEvent.create(
        session=session,
        sequence=1,
        previous_event_hash=GENESIS_EVENT_HASH,
        kind=SessionEventKind.SESSION_CLOSED,
        actor_kind=SessionActorKind.HUMAN,
        actor_id="operator:eugene",
        summary="Closed",
        occurred_at=NOW,
    )
    after = SessionEvent.create(
        session=session,
        sequence=2,
        previous_event_hash=closed.event_hash,
        kind=SessionEventKind.QUESTION_ASKED,
        actor_kind=SessionActorKind.HUMAN,
        actor_id="operator:eugene",
        summary="Too late",
        occurred_at="2026-08-01T10:00:01.000000Z",
    )
    with pytest.raises(ResearchSessionReplayError, match="after closure"):
        coordinator.replay(session, (closed, after))
    with pytest.raises(ResearchSessionReplayError, match="closed research session"):
        coordinator.append(
            session,
            (closed,),
            kind=SessionEventKind.QUESTION_ASKED,
            actor_kind=SessionActorKind.HUMAN,
            actor_id="operator:eugene",
            summary="Too late",
            occurred_at="2026-08-01T10:00:01.000000Z",
        )


def test_empty_state_and_state_identity_fail_closed() -> None:
    session = _session()
    state = ResearchSessionCoordinator().replay(session, ())
    assert state.status is SessionStatus.OPEN
    assert state.event_count == 0
    assert state.latest_event_hash == GENESIS_EVENT_HASH

    with pytest.raises(ValidationError, match="empty session state"):
        ResearchSessionState.create(
            session=session,
            status=SessionStatus.OPEN,
            event_count=0,
            latest_event_hash=HASH_A,
        )

    with pytest.raises(ValidationError, match="non-empty session state"):
        ResearchSessionState.create(
            session=session,
            status=SessionStatus.OPEN,
            event_count=1,
            latest_event_hash=GENESIS_EVENT_HASH,
        )

    payload = state.model_dump(mode="json")
    payload["state_hash"] = HASH_A
    with pytest.raises(ValidationError, match="state_hash does not match"):
        ResearchSessionState.model_validate_json(json.dumps(payload))

    for field, value, message in (
        ("schema_id", "wrong", "schema_id must be"),
        ("state_id", "session_state_" + "f" * 32, "state_id does not match"),
    ):
        payload = state.model_dump(mode="json")
        payload[field] = value
        with pytest.raises(ValidationError, match=message):
            ResearchSessionState.model_validate_json(json.dumps(payload))
