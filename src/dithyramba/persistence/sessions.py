"""Durable, append-only persistence for interactive research sessions."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from pydantic import ValidationError

from dithyramba.access import AccessContractError
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.meaning import MeaningError, MeaningReviewTargetType
from dithyramba.recall.models import QueryRequest
from dithyramba.sessions import (
    ResearchSession,
    ResearchSessionCoordinator,
    ResearchSessionState,
    SessionActorKind,
    SessionArtifactKind,
    SessionArtifactReference,
    SessionEvent,
    SessionEventKind,
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

_COMMAND_STARTED = "research_session.recall_command_started"
_COMMAND_COMPLETED = "research_session.recall_command_completed"
_COMMAND_ID_PATTERN = re.compile(r"^command_[a-z0-9]+(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class RecallCommandStart:
    """One immutable idempotency receipt bound to its question event."""

    command_id: str
    command_hash: str
    query_request_id: str
    question_event: SessionEvent


@dataclass(frozen=True, slots=True)
class RecallCommandCompletion:
    """Persisted identifiers and exact response projection for one completed command."""

    command_id: str
    command_hash: str
    question_event_id: str
    evidence_event_id: str
    evidence_packet_id: str
    evidence_packet_hash: str
    agent_evidence_json: str | None
    context_json: str
    turn_hash: str


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

    def begin_recall_command(
        self,
        *,
        session_id: str,
        command_id: str,
        request: QueryRequest,
        actor_id: str,
        occurred_at: str,
    ) -> RecallCommandStart:
        """Insert a question and its idempotency receipt in one transaction."""

        command = _validated_command_id(command_id)
        if not isinstance(request, QueryRequest):
            raise TypeError("begin_recall_command requires a QueryRequest")
        session = self.get_session(session_id)
        if (
            request.library_id != session.library_id
            or request.collection_ids != session.collection_ids
            or request.corpus_snapshot_id != session.corpus_snapshot_id
            or request.access_policy_id != session.access_policy_id
            or request.purpose != session.purpose
            or request.exclusions != session.scope.exclusions
        ):
            raise ResearchSessionPersistenceError(
                "QueryRequest scope differs from its ResearchSession"
            )
        semantic = {
            "schema": "dithyramba.recall_command/1.0",
            "command_id": command,
            "session_id": session.session_id,
            "query_request_id": request.query_request_id,
            "request_hash": request.request_hash,
        }
        command_hash = canonical_sha256_hex(semantic)
        existing = self._load_command_start(command)
        if existing is not None:
            _require_same_command(existing, command_hash, request.query_request_id)
            return existing

        try:
            with self._repository._store.transaction(immediate=True) as connection:
                existing = self._load_command_start(command, connection=connection)
                if existing is not None:
                    _require_same_command(existing, command_hash, request.query_request_id)
                    return existing
                events = self.list_events(session.session_id)
                question_event = self._coordinator.append(
                    session,
                    events,
                    kind=SessionEventKind.QUESTION_ASKED,
                    actor_kind=SessionActorKind.MODEL,
                    actor_id=actor_id,
                    summary=request.question,
                    occurred_at=occurred_at,
                )[-1]
                self._insert_event(connection, session, question_event)
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type=_COMMAND_STARTED,
                    aggregate_type="recall_command",
                    aggregate_id=command,
                    payload={
                        "schema": "dithyramba.recall_command_started/1.0",
                        **semantic,
                        "command_hash": command_hash,
                        "question_event_id": question_event.event_id,
                        "question_event_hash": question_event.event_hash,
                    },
                    occurred_at=occurred_at,
                )
        except sqlite3.IntegrityError as exc:
            raise ResearchSessionPersistenceError("recall command start conflicted") from exc
        persisted = self._load_command_start(command)
        if persisted is None:
            raise ResearchSessionPersistenceError("recall command start was not durable")
        _require_same_command(persisted, command_hash, request.query_request_id)
        return persisted

    def complete_recall_command(
        self,
        *,
        start: RecallCommandStart,
        evidence_event: SessionEvent,
        evidence_packet_id: str,
        evidence_packet_hash: str,
        agent_evidence_json: str,
        context_json: str,
        turn_hash: str,
        occurred_at: str,
    ) -> RecallCommandCompletion:
        """Atomically append evidence and the exact command-response receipt."""

        if not isinstance(start, RecallCommandStart):
            raise TypeError("complete_recall_command requires a RecallCommandStart")
        existing = self.load_recall_command_completion(start.command_id)
        if existing is not None:
            _require_completion_matches_start(existing, start)
            return existing
        session = self.get_session(evidence_event.session_id)
        if start.question_event.session_id != session.session_id:
            raise ResearchSessionPersistenceError("recall command belongs to another session")
        events = self.list_events(session.session_id)
        self._coordinator.replay(session, (*events, evidence_event))
        self._validate_artifacts(session, evidence_event.artifact_refs)
        agent_evidence = _canonical_json_text(agent_evidence_json, "agent_evidence_json")
        context = _canonical_json_text(context_json, "context_json")
        completion_payload: dict[str, object] = {
            "schema": "dithyramba.recall_command_completed/1.1",
            "command_id": start.command_id,
            "command_hash": start.command_hash,
            "question_event_id": start.question_event.event_id,
            "evidence_event_id": evidence_event.event_id,
            "evidence_packet_id": evidence_packet_id,
            "evidence_packet_hash": evidence_packet_hash,
            "agent_evidence_json": agent_evidence,
            "context_json": context,
            "turn_hash": turn_hash,
        }
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                existing = self.load_recall_command_completion(
                    start.command_id,
                    connection=connection,
                )
                if existing is not None:
                    _require_completion_matches_start(existing, start)
                    return existing
                events = self.list_events(session.session_id)
                self._coordinator.replay(session, (*events, evidence_event))
                self._insert_event(connection, session, evidence_event)
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type=_COMMAND_COMPLETED,
                    aggregate_type="recall_command",
                    aggregate_id=start.command_id,
                    payload=completion_payload,
                    occurred_at=occurred_at,
                )
        except sqlite3.IntegrityError as exc:
            raise ResearchSessionPersistenceError("recall command completion conflicted") from exc
        persisted = self.load_recall_command_completion(start.command_id)
        if persisted is None:
            raise ResearchSessionPersistenceError("recall command completion was not durable")
        _require_completion_matches_start(persisted, start)
        return persisted

    def load_recall_command_completion(
        self,
        command_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> RecallCommandCompletion | None:
        command = _validated_command_id(command_id)
        payload = self._load_command_payload(
            _COMMAND_COMPLETED,
            command,
            connection=connection,
        )
        if payload is None:
            return None
        try:
            return RecallCommandCompletion(
                command_id=_payload_text(payload, "command_id"),
                command_hash=_payload_text(payload, "command_hash"),
                question_event_id=_payload_text(payload, "question_event_id"),
                evidence_event_id=_payload_text(payload, "evidence_event_id"),
                evidence_packet_id=_payload_text(payload, "evidence_packet_id"),
                evidence_packet_hash=_payload_text(payload, "evidence_packet_hash"),
                agent_evidence_json=_payload_optional_text(payload, "agent_evidence_json"),
                context_json=_payload_text(payload, "context_json"),
                turn_hash=_payload_text(payload, "turn_hash"),
            )
        except (TypeError, ValueError) as exc:
            raise ResearchSessionPersistenceError(
                "persisted recall command completion is invalid"
            ) from exc

    def _load_command_start(
        self,
        command_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> RecallCommandStart | None:
        payload = self._load_command_payload(
            _COMMAND_STARTED,
            _validated_command_id(command_id),
            connection=connection,
        )
        if payload is None:
            return None
        try:
            question_event_id = _payload_text(payload, "question_event_id")
            question_event = self._optional_event(question_event_id)
            if question_event is None or question_event.event_hash != _payload_text(
                payload,
                "question_event_hash",
            ):
                raise ResearchSessionPersistenceError(
                    "recall command question event is absent or stale"
                )
            return RecallCommandStart(
                command_id=_payload_text(payload, "command_id"),
                command_hash=_payload_text(payload, "command_hash"),
                query_request_id=_payload_text(payload, "query_request_id"),
                question_event=question_event,
            )
        except (TypeError, ValueError) as exc:
            raise ResearchSessionPersistenceError(
                "persisted recall command start is invalid"
            ) from exc

    def _load_command_payload(
        self,
        event_type: str,
        command_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, object] | None:
        active = connection or self._repository._store.connection
        rows = active.execute(
            """
            SELECT payload_json, payload_hash FROM event_outbox
            WHERE event_type = ? AND aggregate_type = 'recall_command'
              AND aggregate_id = ?
            ORDER BY occurred_at, event_id
            """,
            (event_type, command_id),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ResearchSessionPersistenceError(
                "recall command has more than one immutable receipt"
            )
        raw_json = str(rows[0][0])
        try:
            value = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ResearchSessionPersistenceError(
                "recall command receipt is not valid JSON"
            ) from exc
        if (
            not isinstance(value, dict)
            or canonical_json_bytes(value).decode("utf-8") != raw_json
            or canonical_sha256_hex(value) != str(rows[0][1])
        ):
            raise ResearchSessionPersistenceError(
                "recall command receipt is not canonical or its hash drifted"
            )
        return value

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        session: ResearchSession,
        event: SessionEvent,
    ) -> None:
        event = _validated_event(event)
        if event.session_hash != session.session_hash:
            raise ResearchSessionPersistenceError(
                "SessionEvent is bound to a stale ResearchSession"
            )
        event_json = _canonical_model_text(event)
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


def _validated_command_id(value: str) -> str:
    if type(value) is not str or len(value) > 96 or _COMMAND_ID_PATTERN.fullmatch(value) is None:
        raise ResearchSessionPersistenceError(
            "command_id must match command_[a-z0-9]+(?:_[a-z0-9]+)*"
        )
    return value


def _payload_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if type(value) is not str or not value:
        raise ValueError(f"recall command payload requires text field {key}")
    return value


def _payload_optional_text(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError(f"recall command payload requires optional text field {key}")
    return value


def _canonical_json_text(value: str, label: str) -> str:
    if type(value) is not str or not value:
        raise ResearchSessionPersistenceError(f"{label} must be nonblank canonical JSON")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ResearchSessionPersistenceError(f"{label} must be valid JSON") from exc
    if canonical_json_bytes(decoded).decode("utf-8") != value:
        raise ResearchSessionPersistenceError(f"{label} must be canonical JSON")
    return value


def _require_same_command(
    persisted: RecallCommandStart,
    command_hash: str,
    query_request_id: str,
) -> None:
    if persisted.command_hash != command_hash or persisted.query_request_id != query_request_id:
        raise ResearchSessionPersistenceError(
            "command_id was already used for different recall input"
        )


def _require_completion_matches_start(
    completion: RecallCommandCompletion,
    start: RecallCommandStart,
) -> None:
    if (
        completion.command_id != start.command_id
        or completion.command_hash != start.command_hash
        or completion.question_event_id != start.question_event.event_id
    ):
        raise ResearchSessionPersistenceError(
            "recall command completion disagrees with its immutable start"
        )
