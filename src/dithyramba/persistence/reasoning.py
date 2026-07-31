"""Append-only SQLite persistence for public IdeaTrace artifacts."""

from __future__ import annotations

import sqlite3

from pydantic import ValidationError

from dithyramba.contracts import canonical_json_bytes
from dithyramba.reasoning import (
    IdeaTrace,
    ReasoningClosureResult,
    ReasoningNotFoundError,
    ReasoningPersistenceError,
)

from .repository import LibraryRepository, _insert_outbox_event, _timestamp


class SQLiteReasoningRepository:
    """Store and reopen immutable traces inside exactly one Library."""

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteReasoningRepository requires a LibraryRepository")
        self._repository = repository

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def persist_trace(self, trace: IdeaTrace) -> IdeaTrace:
        if not isinstance(trace, IdeaTrace):
            raise TypeError("persist_trace requires an IdeaTrace")
        trace = _validated_trace_instance(trace)
        existing = self._optional_trace(trace.trace_id)
        if existing is not None:
            if existing != trace:
                raise ReasoningPersistenceError("IdeaTrace ID conflicts with persisted content")
            return existing

        created_at = _timestamp(self._repository._clock)
        trace_json = _canonical_model_text(trace)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO idea_traces(
                        trace_id, library_id, schema_id, trace_hash, task_id,
                        answer_id, answer_hash, packet_id, packet_hash,
                        case_set_id, case_set_hash, receipt_id, receipt_hash,
                        author_kind, trace_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace.trace_id,
                        self.library_id,
                        trace.schema_id,
                        trace.trace_hash,
                        trace.task_id,
                        trace.answer_id,
                        trace.answer_hash,
                        trace.packet_id,
                        trace.packet_hash,
                        trace.case_set_id,
                        trace.case_set_hash,
                        trace.receipt_id,
                        trace.receipt_hash,
                        trace.author_kind.value,
                        trace_json,
                        created_at,
                    ),
                )
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="idea_trace.created",
                    aggregate_type="idea_trace",
                    aggregate_id=trace.trace_id,
                    payload={
                        "schema": "dithyramba.idea_trace_created/1.0",
                        "library_id": self.library_id,
                        "trace_id": trace.trace_id,
                        "trace_hash": trace.trace_hash,
                        "task_id": trace.task_id,
                        "case_set_id": trace.case_set_id,
                        "receipt_id": trace.receipt_id,
                    },
                    occurred_at=created_at,
                )
        except sqlite3.IntegrityError as exc:
            raise ReasoningPersistenceError("IdeaTrace append-only write conflicted") from exc
        return self.get_trace(trace.trace_id)

    def get_trace(self, trace_id: str) -> IdeaTrace:
        trace = self._optional_trace(trace_id)
        if trace is None:
            raise ReasoningNotFoundError("IdeaTrace does not exist in this Library")
        return trace

    def persist_closure(self, result: ReasoningClosureResult) -> ReasoningClosureResult:
        if not isinstance(result, ReasoningClosureResult):
            raise TypeError("persist_closure requires a ReasoningClosureResult")
        result = _validated_closure_instance(result)
        trace = self.get_trace(result.trace_id)
        if (
            trace.trace_hash != result.trace_hash
            or trace.case_set_id != result.case_set_id
            or trace.case_set_hash != result.case_set_hash
            or trace.receipt_id != result.receipt_id
            or trace.receipt_hash != result.receipt_hash
        ):
            raise ReasoningPersistenceError("closure result is bound to a different IdeaTrace")
        existing = self._optional_closure(result.closure_id)
        if existing is not None:
            if existing != result:
                raise ReasoningPersistenceError(
                    "reasoning closure ID conflicts with persisted content"
                )
            return existing

        created_at = _timestamp(self._repository._clock)
        closure_json = _canonical_model_text(result)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO reasoning_closure_results(
                        closure_id, library_id, trace_id, trace_hash,
                        closure_hash, decision, closure_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.closure_id,
                        self.library_id,
                        result.trace_id,
                        result.trace_hash,
                        result.closure_hash,
                        result.decision.value,
                        closure_json,
                        created_at,
                    ),
                )
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="reasoning_closure.completed",
                    aggregate_type="reasoning_closure",
                    aggregate_id=result.closure_id,
                    payload={
                        "schema": "dithyramba.reasoning_closure_completed/1.0",
                        "library_id": self.library_id,
                        "closure_id": result.closure_id,
                        "closure_hash": result.closure_hash,
                        "trace_id": result.trace_id,
                        "decision": result.decision.value,
                        "review_eligible": result.review_eligible,
                    },
                    occurred_at=created_at,
                )
        except sqlite3.IntegrityError as exc:
            raise ReasoningPersistenceError(
                "reasoning closure append-only write conflicted"
            ) from exc
        return self.get_closure(result.closure_id)

    def get_closure(self, closure_id: str) -> ReasoningClosureResult:
        result = self._optional_closure(closure_id)
        if result is None:
            raise ReasoningNotFoundError("reasoning closure result does not exist in this Library")
        trace = self.get_trace(result.trace_id)
        if (
            trace.trace_hash != result.trace_hash
            or trace.case_set_id != result.case_set_id
            or trace.case_set_hash != result.case_set_hash
            or trace.receipt_id != result.receipt_id
            or trace.receipt_hash != result.receipt_hash
        ):
            raise ReasoningPersistenceError("persisted closure has a stale IdeaTrace binding")
        return result

    def _optional_trace(self, trace_id: str) -> IdeaTrace | None:
        row = self._repository._store.connection.execute(
            """
            SELECT trace_json FROM idea_traces
            WHERE trace_id = ? AND library_id = ?
            """,
            (trace_id, self.library_id),
        ).fetchone()
        if row is None:
            return None
        return _load_trace(str(row[0]))

    def _optional_closure(self, closure_id: str) -> ReasoningClosureResult | None:
        row = self._repository._store.connection.execute(
            """
            SELECT closure_json FROM reasoning_closure_results
            WHERE closure_id = ? AND library_id = ?
            """,
            (closure_id, self.library_id),
        ).fetchone()
        if row is None:
            return None
        return _load_closure(str(row[0]))


def _canonical_model_text(value: IdeaTrace | ReasoningClosureResult) -> str:
    return canonical_json_bytes(value.model_dump(mode="json")).decode("utf-8")


def _validated_trace_instance(value: IdeaTrace) -> IdeaTrace:
    try:
        return IdeaTrace.model_validate_json(_canonical_model_text(value))
    except ValidationError as exc:
        raise ReasoningPersistenceError("IdeaTrace content identity is invalid") from exc


def _validated_closure_instance(value: ReasoningClosureResult) -> ReasoningClosureResult:
    try:
        return ReasoningClosureResult.model_validate_json(_canonical_model_text(value))
    except ValidationError as exc:
        raise ReasoningPersistenceError("reasoning closure content identity is invalid") from exc


def _load_trace(value: str) -> IdeaTrace:
    try:
        trace = IdeaTrace.model_validate_json(value)
    except ValidationError as exc:
        raise ReasoningPersistenceError("persisted IdeaTrace is invalid") from exc
    if _canonical_model_text(trace) != value:
        raise ReasoningPersistenceError("persisted IdeaTrace is not canonical JSON")
    return trace


def _load_closure(value: str) -> ReasoningClosureResult:
    try:
        result = ReasoningClosureResult.model_validate_json(value)
    except ValidationError as exc:
        raise ReasoningPersistenceError("persisted reasoning closure is invalid") from exc
    if _canonical_model_text(result) != value:
        raise ReasoningPersistenceError("persisted reasoning closure is not canonical JSON")
    return result
