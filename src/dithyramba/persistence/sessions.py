"""Durable, append-only persistence for interactive research sessions."""

from __future__ import annotations

import sqlite3

from pydantic import ValidationError

from dithyramba.access import AccessContractError
from dithyramba.contracts import canonical_json_bytes
from dithyramba.meaning import MeaningError, MeaningReviewTargetType
from dithyramba.sessions import (
    ResearchSession,
    ResearchSessionCoordinator,
    ResearchSessionState,
    SessionArtifactKind,
    SessionArtifactReference,
    SessionEvent,
)

from .errors import (
    PersistenceError,
    ResearchSessionNotFoundError,
    ResearchSessionPersistenceError,
    SessionArtifactNotFoundError,
)
from .meaning import SQLiteMeaningRepository
from .recall import SQLiteRecallBackend
from .repository import LibraryRepository, _insert_outbox_event


class SQLiteResearchSessionRepository:
    """Persist and replay one Library's exact research-session event streams."""

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteResearchSessionRepository requires a LibraryRepository")
        self._repository = repository
        self._coordinator = ResearchSessionCoordinator()

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def persist_session(self, session: ResearchSession) -> ResearchSession:
        if not isinstance(session, ResearchSession):
            raise TypeError("persist_session requires a ResearchSession")
        session = _validated_session(session)
        self._validate_session_scope(session)
        existing = self._optional_session(session.session_id)
        if existing is not None:
            if existing != session:
                raise ResearchSessionPersistenceError(
                    "ResearchSession ID conflicts with persisted content"
                )
            return existing

        session_json = _canonical_model_text(session)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO research_sessions(
                        session_id, library_id, schema_id, session_hash,
                        brief_id, brief_hash, corpus_snapshot_id, snapshot_hash,
                        access_policy_id, scope_hash, created_by_kind,
                        session_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.session_id,
                        self.library_id,
                        session.schema_id,
                        session.session_hash,
                        session.brief.brief_id,
                        session.brief.brief_hash,
                        session.corpus_snapshot_id,
                        session.snapshot_hash,
                        session.access_policy_id,
                        session.scope_hash,
                        session.created_by_kind.value,
                        session_json,
                        session.created_at,
                    ),
                )
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="research_session.created",
                    aggregate_type="research_session",
                    aggregate_id=session.session_id,
                    payload={
                        "schema": "dithyramba.research_session_created/1.0",
                        "library_id": self.library_id,
                        "session_id": session.session_id,
                        "session_hash": session.session_hash,
                        "brief_id": session.brief.brief_id,
                        "corpus_snapshot_id": session.corpus_snapshot_id,
                        "access_policy_id": session.access_policy_id,
                        "scope_hash": session.scope_hash,
                    },
                    occurred_at=session.created_at,
                )
        except sqlite3.IntegrityError as exc:
            raise ResearchSessionPersistenceError(
                "ResearchSession append-only write conflicted"
            ) from exc
        return self.get_session(session.session_id)

    def get_session(self, session_id: str) -> ResearchSession:
        session = self._optional_session(session_id)
        if session is None:
            raise ResearchSessionNotFoundError("ResearchSession does not exist in this Library")
        return session

    def append_event(self, event: SessionEvent) -> SessionEvent:
        if not isinstance(event, SessionEvent):
            raise TypeError("append_event requires a SessionEvent")
        event = _validated_event(event)
        session = self.get_session(event.session_id)
        if event.session_hash != session.session_hash:
            raise ResearchSessionPersistenceError(
                "SessionEvent is bound to a stale ResearchSession"
            )
        existing = self._optional_event(event.event_id)
        if existing is not None:
            if existing != event:
                raise ResearchSessionPersistenceError(
                    "SessionEvent ID conflicts with persisted content"
                )
            return existing

        events = self.list_events(session.session_id)
        self._coordinator.replay(session, (*events, event))
        self._validate_artifacts(session, event.artifact_refs)
        event_json = _canonical_model_text(event)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO research_session_events(
                        event_id, library_id, session_id, schema_id, event_hash,
                        sequence, previous_event_hash, kind, actor_kind,
                        event_json, occurred_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        self.library_id,
                        event.session_id,
                        event.schema_id,
                        event.event_hash,
                        event.sequence,
                        event.previous_event_hash,
                        event.kind.value,
                        event.actor_kind.value,
                        event_json,
                        event.occurred_at,
                    ),
                )
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="research_session.event_appended",
                    aggregate_type="research_session",
                    aggregate_id=session.session_id,
                    payload={
                        "schema": "dithyramba.research_session_event_appended/1.0",
                        "library_id": self.library_id,
                        "session_id": session.session_id,
                        "session_hash": session.session_hash,
                        "event_id": event.event_id,
                        "event_hash": event.event_hash,
                        "sequence": event.sequence,
                        "kind": event.kind.value,
                    },
                    occurred_at=event.occurred_at,
                )
        except sqlite3.IntegrityError as exc:
            raise ResearchSessionPersistenceError(
                "SessionEvent append-only write conflicted"
            ) from exc
        persisted = self._optional_event(event.event_id)
        if persisted is None:
            raise ResearchSessionPersistenceError("SessionEvent insert was not durable")
        return persisted

    def list_events(self, session_id: str) -> tuple[SessionEvent, ...]:
        session = self.get_session(session_id)
        rows = self._repository._store.connection.execute(
            """
            SELECT event_id, session_id, schema_id, event_hash, sequence,
                   previous_event_hash, kind, actor_kind, event_json, occurred_at
            FROM research_session_events
            WHERE session_id = ? AND library_id = ?
            ORDER BY sequence
            """,
            (session_id, self.library_id),
        ).fetchall()
        events = tuple(_event_from_row(row) for row in rows)
        self._coordinator.replay(session, events)
        return events

    def get_state(self, session_id: str) -> ResearchSessionState:
        session = self.get_session(session_id)
        return self._coordinator.replay(session, self.list_events(session_id))

    def _validate_session_scope(self, session: ResearchSession) -> None:
        if session.library_id != self.library_id:
            raise ResearchSessionPersistenceError("ResearchSession belongs to a different Library")
        snapshot = self._repository.get_corpus_snapshot(session.corpus_snapshot_id)
        if snapshot.manifest_hash != session.snapshot_hash:
            raise ResearchSessionPersistenceError(
                "ResearchSession snapshot ID/hash binding is stale"
            )
        if not set(session.collection_ids).issubset(snapshot.collection_ids):
            raise ResearchSessionPersistenceError(
                "ResearchSession Collections exceed its CorpusSnapshot"
            )
        try:
            self._repository.authorize_read(
                access_policy_id=session.access_policy_id,
                scope=session.scope,
            )
        except (AccessContractError, PersistenceError) as exc:
            raise ResearchSessionPersistenceError(
                "ResearchSession scope is not authorized by its AccessPolicy"
            ) from exc

    def _validate_artifacts(
        self,
        session: ResearchSession,
        references: tuple[SessionArtifactReference, ...],
    ) -> None:
        if not references:
            return
        authorization = self._repository.authorize_read(
            access_policy_id=session.access_policy_id,
            scope=session.scope,
        )
        permitted_fragments = {
            item.source_fragment_id for item in authorization.compiled.manifest.items
        }
        meaning = SQLiteMeaningRepository(self._repository)
        recall = SQLiteRecallBackend(self._repository)
        for reference in references:
            if reference.artifact_kind is SessionArtifactKind.SOURCE_FRAGMENT:
                row = self._repository._store.connection.execute(
                    """
                    SELECT sf.text_sha256
                    FROM source_fragments AS sf
                    JOIN source_versions AS sv
                      ON sv.source_version_id = sf.source_version_id
                    JOIN sources AS s ON s.source_id = sv.source_id
                    WHERE sf.source_fragment_id = ? AND s.library_id = ?
                    """,
                    (reference.artifact_id, self.library_id),
                ).fetchone()
                if (
                    row is None
                    or reference.artifact_id not in permitted_fragments
                    or str(row[0]) != reference.artifact_hash
                ):
                    raise _artifact_error(reference)
                continue
            if reference.artifact_kind is SessionArtifactKind.EVIDENCE_PACKET:
                try:
                    packet = recall.load_evidence_packet(reference.artifact_id)
                    request = recall.load_query_request(packet.query_request_id)
                except (PersistenceError, ValueError, sqlite3.DatabaseError) as exc:
                    raise _artifact_error(reference) from exc
                if (
                    packet.packet_hash != reference.artifact_hash
                    or request.library_id != session.library_id
                    or request.corpus_snapshot_id != session.corpus_snapshot_id
                    or request.access_policy_id != session.access_policy_id
                    or request.purpose != session.purpose
                    or request.collection_ids != session.collection_ids
                    or request.exclusions != session.scope.exclusions
                ):
                    raise _artifact_error(reference)
                continue
            target_type = {
                SessionArtifactKind.EVIDENCE_LINK: MeaningReviewTargetType.EVIDENCE_LINK,
                SessionArtifactKind.STATEMENT: MeaningReviewTargetType.STATEMENT,
                SessionArtifactKind.CONCEPT: MeaningReviewTargetType.CONCEPT,
                SessionArtifactKind.ENTITY: MeaningReviewTargetType.ENTITY,
            }.get(reference.artifact_kind)
            if target_type is None:
                raise _artifact_error(reference)
            try:
                target = meaning.get_review_target(target_type, reference.artifact_id)
            except (MeaningError, PersistenceError, sqlite3.DatabaseError) as exc:
                raise _artifact_error(reference) from exc
            if target.target_hash != reference.artifact_hash or not set(
                target.collection_ids
            ).issubset(session.collection_ids):
                raise _artifact_error(reference)

    def _optional_session(self, session_id: str) -> ResearchSession | None:
        row = self._repository._store.connection.execute(
            """
            SELECT session_id, library_id, schema_id, session_hash, brief_id,
                   brief_hash, corpus_snapshot_id, snapshot_hash,
                   access_policy_id, scope_hash, created_by_kind,
                   session_json, created_at
            FROM research_sessions
            WHERE session_id = ? AND library_id = ?
            """,
            (session_id, self.library_id),
        ).fetchone()
        if row is None:
            return None
        return _session_from_row(row)

    def _optional_event(self, event_id: str) -> SessionEvent | None:
        row = self._repository._store.connection.execute(
            """
            SELECT event_id, session_id, schema_id, event_hash, sequence,
                   previous_event_hash, kind, actor_kind, event_json, occurred_at
            FROM research_session_events
            WHERE event_id = ? AND library_id = ?
            """,
            (event_id, self.library_id),
        ).fetchone()
        if row is None:
            return None
        return _event_from_row(row)


def _canonical_model_text(value: ResearchSession | SessionEvent) -> str:
    return canonical_json_bytes(value.model_dump(mode="json")).decode("utf-8")


def _validated_session(value: ResearchSession) -> ResearchSession:
    try:
        return ResearchSession.model_validate_json(_canonical_model_text(value))
    except ValidationError as exc:
        raise ResearchSessionPersistenceError(
            "ResearchSession content identity is invalid"
        ) from exc


def _validated_event(value: SessionEvent) -> SessionEvent:
    try:
        return SessionEvent.model_validate_json(_canonical_model_text(value))
    except ValidationError as exc:
        raise ResearchSessionPersistenceError("SessionEvent content identity is invalid") from exc


def _session_from_row(row: sqlite3.Row) -> ResearchSession:
    session = _load_session(str(row[11]))
    expected = (
        session.session_id,
        session.library_id,
        session.schema_id,
        session.session_hash,
        session.brief.brief_id,
        session.brief.brief_hash,
        session.corpus_snapshot_id,
        session.snapshot_hash,
        session.access_policy_id,
        session.scope_hash,
        session.created_by_kind.value,
        session.created_at,
    )
    actual = tuple(str(row[index]) for index in (*range(11), 12))
    if actual != expected:
        raise ResearchSessionPersistenceError(
            "persisted ResearchSession columns disagree with canonical JSON"
        )
    return session


def _event_from_row(row: sqlite3.Row) -> SessionEvent:
    event = _load_event(str(row[8]))
    expected: tuple[object, ...] = (
        event.event_id,
        event.session_id,
        event.schema_id,
        event.event_hash,
        event.sequence,
        event.previous_event_hash,
        event.kind.value,
        event.actor_kind.value,
        event.occurred_at,
    )
    actual: tuple[object, ...] = (*tuple(row[index] for index in range(8)), row[9])
    if actual != expected:
        raise ResearchSessionPersistenceError(
            "persisted SessionEvent columns disagree with canonical JSON"
        )
    return event


def _load_session(value: str) -> ResearchSession:
    try:
        session = ResearchSession.model_validate_json(value)
    except ValidationError as exc:
        raise ResearchSessionPersistenceError("persisted ResearchSession is invalid") from exc
    if _canonical_model_text(session) != value:
        raise ResearchSessionPersistenceError("persisted ResearchSession is not canonical JSON")
    return session


def _load_event(value: str) -> SessionEvent:
    try:
        event = SessionEvent.model_validate_json(value)
    except ValidationError as exc:
        raise ResearchSessionPersistenceError("persisted SessionEvent is invalid") from exc
    if _canonical_model_text(event) != value:
        raise ResearchSessionPersistenceError("persisted SessionEvent is not canonical JSON")
    return event


def _artifact_error(reference: SessionArtifactReference) -> SessionArtifactNotFoundError:
    return SessionArtifactNotFoundError(
        f"session artifact is absent, stale, or outside scope: {reference.artifact_id}"
    )
