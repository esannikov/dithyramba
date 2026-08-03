"""Append-only SQLite persistence for proposition-level answer projections."""

from __future__ import annotations

import sqlite3

from pydantic import ValidationError

from dithyramba.answers import AnswerProjection, AnswerProjectionJudgmentReceipt
from dithyramba.contracts import canonical_json_bytes

from .errors import AnswerProjectionNotFoundError, AnswerProjectionPersistenceError
from .repository import LibraryRepository, _insert_outbox_event, _timestamp


class SQLiteAnswerProjectionRepository:
    """Store exact display-governance overlays inside one Library."""

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteAnswerProjectionRepository requires a LibraryRepository")
        self._repository = repository

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def persist_projection(self, projection: AnswerProjection) -> AnswerProjection:
        if not isinstance(projection, AnswerProjection):
            raise TypeError("persist_projection requires an AnswerProjection")
        projection = _validated_projection(projection)
        existing = self._optional_projection(projection.projection_id)
        if existing is not None:
            if existing != projection:
                raise AnswerProjectionPersistenceError(
                    "AnswerProjection ID conflicts with persisted content"
                )
            return existing

        created_at = _timestamp(self._repository._clock)
        projection_json = _canonical_model_text(projection)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO answer_projections(
                        projection_id, library_id, schema_id, projection_hash,
                        answer_id, answer_hash, case_set_id, case_set_hash,
                        entailment_receipt_id, entailment_receipt_hash,
                        author_kind, projection_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        projection.projection_id,
                        self.library_id,
                        projection.schema_id,
                        projection.projection_hash,
                        projection.answer_id,
                        projection.answer_hash,
                        projection.case_set_id,
                        projection.case_set_hash,
                        projection.entailment_receipt_id,
                        projection.entailment_receipt_hash,
                        projection.author_kind.value,
                        projection_json,
                        created_at,
                    ),
                )
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="answer_projection.created",
                    aggregate_type="answer_projection",
                    aggregate_id=projection.projection_id,
                    payload={
                        "schema": "dithyramba.answer_projection_created/1.0",
                        "library_id": self.library_id,
                        "projection_id": projection.projection_id,
                        "projection_hash": projection.projection_hash,
                        "answer_id": projection.answer_id,
                        "case_set_id": projection.case_set_id,
                        "entailment_receipt_id": projection.entailment_receipt_id,
                    },
                    occurred_at=created_at,
                )
        except sqlite3.IntegrityError as exc:
            raise AnswerProjectionPersistenceError(
                "AnswerProjection append-only write conflicted"
            ) from exc
        return self.get_projection(projection.projection_id)

    def get_projection(self, projection_id: str) -> AnswerProjection:
        projection = self._optional_projection(projection_id)
        if projection is None:
            raise AnswerProjectionNotFoundError("AnswerProjection does not exist in this Library")
        return projection

    def persist_receipt(
        self,
        receipt: AnswerProjectionJudgmentReceipt,
    ) -> AnswerProjectionJudgmentReceipt:
        if not isinstance(receipt, AnswerProjectionJudgmentReceipt):
            raise TypeError("persist_receipt requires an AnswerProjectionJudgmentReceipt")
        receipt = _validated_receipt(receipt)
        projection = self.get_projection(receipt.projection_id)
        if projection.projection_hash != receipt.projection_hash:
            raise AnswerProjectionPersistenceError(
                "projection judgment receipt is bound to a different AnswerProjection"
            )
        existing = self._optional_receipt(receipt.receipt_id)
        if existing is not None:
            if existing != receipt:
                raise AnswerProjectionPersistenceError(
                    "projection judgment receipt ID conflicts with persisted content"
                )
            return existing

        created_at = _timestamp(self._repository._clock)
        receipt_json = _canonical_model_text(receipt)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO answer_projection_receipts(
                        receipt_id, library_id, projection_id, projection_hash,
                        receipt_hash, judge_kind, receipt_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.receipt_id,
                        self.library_id,
                        receipt.projection_id,
                        receipt.projection_hash,
                        receipt.receipt_hash,
                        receipt.judge_kind.value,
                        receipt_json,
                        created_at,
                    ),
                )
                _insert_outbox_event(
                    connection,
                    event_id=self._repository._event_id_factory(),
                    event_type="answer_projection_judgment.completed",
                    aggregate_type="answer_projection_judgment",
                    aggregate_id=receipt.receipt_id,
                    payload={
                        "schema": "dithyramba.answer_projection_judgment_completed/1.0",
                        "library_id": self.library_id,
                        "receipt_id": receipt.receipt_id,
                        "receipt_hash": receipt.receipt_hash,
                        "projection_id": receipt.projection_id,
                        "judge_kind": receipt.judge_kind.value,
                    },
                    occurred_at=created_at,
                )
        except sqlite3.IntegrityError as exc:
            raise AnswerProjectionPersistenceError(
                "projection judgment receipt append-only write conflicted"
            ) from exc
        return self.get_receipt(receipt.receipt_id)

    def get_receipt(self, receipt_id: str) -> AnswerProjectionJudgmentReceipt:
        receipt = self._optional_receipt(receipt_id)
        if receipt is None:
            raise AnswerProjectionNotFoundError(
                "projection judgment receipt does not exist in this Library"
            )
        projection = self.get_projection(receipt.projection_id)
        if projection.projection_hash != receipt.projection_hash:
            raise AnswerProjectionPersistenceError(
                "persisted projection judgment has a stale AnswerProjection binding"
            )
        return receipt

    def _optional_projection(self, projection_id: str) -> AnswerProjection | None:
        row = self._repository._store.connection.execute(
            """
            SELECT projection_json FROM answer_projections
            WHERE projection_id = ? AND library_id = ?
            """,
            (projection_id, self.library_id),
        ).fetchone()
        if row is None:
            return None
        return _load_projection(str(row[0]))

    def _optional_receipt(
        self,
        receipt_id: str,
    ) -> AnswerProjectionJudgmentReceipt | None:
        row = self._repository._store.connection.execute(
            """
            SELECT receipt_json FROM answer_projection_receipts
            WHERE receipt_id = ? AND library_id = ?
            """,
            (receipt_id, self.library_id),
        ).fetchone()
        if row is None:
            return None
        return _load_receipt(str(row[0]))


def _canonical_model_text(
    value: AnswerProjection | AnswerProjectionJudgmentReceipt,
) -> str:
    return canonical_json_bytes(value.model_dump(mode="json")).decode("utf-8")


def _validated_projection(value: AnswerProjection) -> AnswerProjection:
    try:
        return AnswerProjection.model_validate_json(_canonical_model_text(value))
    except ValidationError as exc:
        raise AnswerProjectionPersistenceError(
            "AnswerProjection content identity is invalid"
        ) from exc


def _validated_receipt(
    value: AnswerProjectionJudgmentReceipt,
) -> AnswerProjectionJudgmentReceipt:
    try:
        return AnswerProjectionJudgmentReceipt.model_validate_json(_canonical_model_text(value))
    except ValidationError as exc:
        raise AnswerProjectionPersistenceError(
            "projection judgment receipt content identity is invalid"
        ) from exc


def _load_projection(value: str) -> AnswerProjection:
    try:
        projection = AnswerProjection.model_validate_json(value)
    except ValidationError as exc:
        raise AnswerProjectionPersistenceError("persisted AnswerProjection is invalid") from exc
    if _canonical_model_text(projection) != value:
        raise AnswerProjectionPersistenceError("persisted AnswerProjection is not canonical JSON")
    return projection


def _load_receipt(value: str) -> AnswerProjectionJudgmentReceipt:
    try:
        receipt = AnswerProjectionJudgmentReceipt.model_validate_json(value)
    except ValidationError as exc:
        raise AnswerProjectionPersistenceError(
            "persisted projection judgment receipt is invalid"
        ) from exc
    if _canonical_model_text(receipt) != value:
        raise AnswerProjectionPersistenceError(
            "persisted projection judgment receipt is not canonical JSON"
        )
    return receipt
