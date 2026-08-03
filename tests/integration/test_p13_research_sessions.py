"""P13 research-session migration, persistence, replay, and corruption tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    ResearchSessionPersistenceError,
    SQLiteResearchSessionRepository,
    initialize_library,
    open_library,
)
from dithyramba.persistence.recall import SQLiteRecallBackend
from dithyramba.recall import (
    QueryRequest,
    RecallService,
    RetrievalBudget,
    current_fts_runtime_profile,
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
    SessionStatus,
)

HASH_A = "a" * 64
NOW = "2026-08-01T10:00:00.000000Z"


def _prepared_session(
    repository: object,
    root: Path,
) -> tuple[ResearchSession, SessionArtifactReference, SessionArtifactReference]:
    from dithyramba.persistence import LibraryRepository

    assert isinstance(repository, LibraryRepository)
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="Session corpus",
            kind=CollectionKind.CORPUS,
            roots=(
                build_collection_root(
                    root,
                    data_root=repository.paths.application_data_root,
                ),
            ),
        )
    )
    collection_id = collection.config.collection_id
    IngestService(repository).ingest_path(collection_id, "paper.md")
    snapshot = repository.freeze_snapshot((collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id="policy_session_research",
        library_id=repository.library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
    )
    repository.persist_access_policy(name="Session research", snapshot=policy)
    recalled = RecallService(
        SQLiteRecallBackend(repository),
        profile_version=current_fts_runtime_profile().profile_version,
    ).recall(
        QueryRequest(
            question="Evidence",
            library_id=repository.library_id,
            collection_ids=(collection_id,),
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            purpose="research",
            retrieval=RetrievalBudget(max_candidates=10, max_source_fragments=5),
        )
    )
    selected = recalled.packet.source_fragments[0]
    session = ResearchSession.create(
        brief=ResearchSessionBrief.create(
            question="How can this investigation continue after restart?",
            intended_use="research session persistence",
            success_criteria=("replay exact events", "retain source closure"),
        ),
        library_id=repository.library_id,
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        snapshot_hash=snapshot.manifest_hash,
        access_policy_id=policy.access_policy_id,
        purpose="research",
        collection_ids=(collection_id,),
        created_by_kind=SessionActorKind.HUMAN,
        created_by_id="operator:eugene",
        created_at=NOW,
    )
    packet_ref = SessionArtifactReference(
        artifact_kind=SessionArtifactKind.EVIDENCE_PACKET,
        artifact_id=recalled.packet.evidence_packet_id,
        artifact_hash=recalled.packet.packet_hash,
        role="recall",
    )
    fragment_ref = SessionArtifactReference(
        artifact_kind=SessionArtifactKind.SOURCE_FRAGMENT,
        artifact_id=selected.source_fragment_id,
        artifact_hash=selected.text_sha256,
        role="supports",
    )
    return session, packet_ref, fragment_ref


def _event_stream(
    session: ResearchSession,
    packet_ref: SessionArtifactReference,
    fragment_ref: SessionArtifactReference,
) -> tuple[SessionEvent, ...]:
    coordinator = ResearchSessionCoordinator()
    events: tuple[SessionEvent, ...] = ()
    plan = (
        (
            SessionEventKind.QUESTION_ASKED,
            SessionActorKind.HUMAN,
            "What preserves the exact research path?",
            (),
        ),
        (
            SessionEventKind.EVIDENCE_ATTACHED,
            SessionActorKind.MODEL,
            "Attach the exact packet and selected fragment.",
            (packet_ref, fragment_ref),
        ),
        (
            SessionEventKind.ANSWER_DRAFTED,
            SessionActorKind.MODEL,
            "The event chain preserves the investigated path; this sentence is a draft.",
            (),
        ),
        (
            SessionEventKind.SESSION_CLOSED,
            SessionActorKind.HUMAN,
            "The operator closed this bounded session.",
            (),
        ),
    )
    for index, (kind, actor_kind, summary, references) in enumerate(plan):
        events = coordinator.append(
            session,
            events,
            kind=kind,
            actor_kind=actor_kind,
            actor_id="operator:eugene" if actor_kind is SessionActorKind.HUMAN else "model:test",
            summary=summary,
            artifact_refs=references,
            occurred_at=f"2026-08-01T10:00:{index:02d}.000000Z",
        )
    return events


def test_session_events_are_append_only_and_reopen_exactly(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "paper.md").write_text(
        "# Evidence\n\nAn exact append-only path preserves the research state.\n",
        encoding="utf-8",
    )
    data_root = tmp_path / "data"
    with initialize_library(
        LibraryConfig(name="Research session persistence"),
        data_root=data_root,
    ) as repository:
        library_id = repository.library_id
        session, packet_ref, fragment_ref = _prepared_session(repository, root)
        events = _event_stream(session, packet_ref, fragment_ref)
        store = SQLiteResearchSessionRepository(repository)

        assert store.persist_session(session) == session
        assert store.persist_session(session) == session
        for event in events:
            assert store.append_event(event) == event
            assert store.append_event(event) == event
        assert store.list_events(session.session_id) == events
        state = store.get_state(session.session_id)
        assert state.status is SessionStatus.CLOSED
        assert state.event_count == 4
        assert state.evidence_refs == (packet_ref, fragment_ref)
        assert repository.schema_version == 1
        assert (
            repository._store.connection.execute(
                """
                SELECT COUNT(*) FROM event_outbox
                WHERE event_type IN (
                    'research_session.created',
                    'research_session.event_appended'
                )
                """
            ).fetchone()[0]
            == 5
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repository._store.connection.execute(
                "UPDATE research_sessions SET created_by_kind = 'model' WHERE session_id = ?",
                (session.session_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repository._store.connection.execute(
                "DELETE FROM research_session_events WHERE event_id = ?",
                (events[0].event_id,),
            )

    with open_library(library_id, data_root=data_root) as repository:
        reopened = SQLiteResearchSessionRepository(repository)
        assert reopened.get_session(session.session_id) == session
        assert reopened.list_events(session.session_id) == events
        assert reopened.get_state(session.session_id).state_hash == state.state_hash


def test_session_scope_and_artifact_hash_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "paper.md").write_text("# Evidence\n\nScoped evidence.\n", encoding="utf-8")
    data_root = tmp_path / "data"
    with initialize_library(
        LibraryConfig(name="First session Library"), data_root=data_root
    ) as repository:
        session, packet_ref, fragment_ref = _prepared_session(repository, root)
        store = SQLiteResearchSessionRepository(repository)
        stale_snapshot_session = ResearchSession.create(
            brief=session.brief,
            library_id=session.library_id,
            corpus_snapshot_id=session.corpus_snapshot_id,
            snapshot_hash=HASH_A,
            access_policy_id=session.access_policy_id,
            purpose=session.purpose,
            collection_ids=session.collection_ids,
            exclusions=session.scope.exclusions,
            created_by_kind=session.created_by_kind,
            created_by_id=session.created_by_id,
            created_at=session.created_at,
        )
        with pytest.raises(ResearchSessionPersistenceError, match="snapshot ID/hash"):
            store.persist_session(stale_snapshot_session)
        store.persist_session(session)
        event = ResearchSessionCoordinator().append(
            session,
            (),
            kind=SessionEventKind.EVIDENCE_ATTACHED,
            actor_kind=SessionActorKind.MODEL,
            actor_id="model:test",
            summary="Stale fragment reference",
            artifact_refs=(fragment_ref.model_copy(update={"artifact_hash": HASH_A}),),
            occurred_at=NOW,
        )[0]
        with pytest.raises(ResearchSessionPersistenceError, match="absent, stale, or outside"):
            store.append_event(event)

        packet_event = ResearchSessionCoordinator().append(
            session,
            (),
            kind=SessionEventKind.EVIDENCE_ATTACHED,
            actor_kind=SessionActorKind.MODEL,
            actor_id="model:test",
            summary="Stale packet reference",
            artifact_refs=(packet_ref.model_copy(update={"artifact_hash": HASH_A}),),
            occurred_at=NOW,
        )[0]
        with pytest.raises(ResearchSessionPersistenceError, match="absent, stale, or outside"):
            store.append_event(packet_event)

    with (
        initialize_library(
            LibraryConfig(name="Second session Library"), data_root=data_root
        ) as other,
        pytest.raises(ResearchSessionPersistenceError, match="different Library"),
    ):
        SQLiteResearchSessionRepository(other).persist_session(session)


def test_corrupt_session_and_event_rows_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "paper.md").write_text("# Evidence\n\nIntegrity.\n", encoding="utf-8")
    with initialize_library(
        LibraryConfig(name="Session corruption"), data_root=tmp_path / "data"
    ) as repository:
        session, packet_ref, fragment_ref = _prepared_session(repository, root)
        events = _event_stream(session, packet_ref, fragment_ref)
        store = SQLiteResearchSessionRepository(repository)
        with pytest.raises(ResearchSessionPersistenceError, match="content identity"):
            store.persist_session(session.model_copy(update={"session_hash": HASH_A}))
        store.persist_session(session)
        with pytest.raises(ResearchSessionPersistenceError, match="content identity"):
            store.append_event(events[0].model_copy(update={"event_hash": HASH_A}))
        store.append_event(events[0])

        connection = repository._store.connection
        connection.execute("DROP TRIGGER research_sessions_no_update")
        payload = json.loads(
            str(
                connection.execute(
                    "SELECT session_json FROM research_sessions WHERE session_id = ?",
                    (session.session_id,),
                ).fetchone()[0]
            )
        )
        payload["created_by_id"] = "operator:changed"
        connection.execute(
            "UPDATE research_sessions SET session_json = ? WHERE session_id = ?",
            (json.dumps(payload), session.session_id),
        )
        with pytest.raises(ResearchSessionPersistenceError, match="persisted ResearchSession"):
            store.get_session(session.session_id)
