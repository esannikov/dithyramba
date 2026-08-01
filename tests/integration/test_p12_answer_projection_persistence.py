"""P12 proposition projection migration, persistence, and corruption tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    AnswerProjectionNotFoundError,
    AnswerProjectionPersistenceError,
    SQLiteAnswerProjectionRepository,
    initialize_library,
    open_library,
)

from ..unit.test_answer_projection import _projection_bundle


def test_projection_and_receipt_are_append_only_and_reopen_exactly(
    tmp_path: Path,
) -> None:
    (tmp_path / "fixture").mkdir()
    _answer, _case_set, _entailment, projection, receipt = _projection_bundle(tmp_path / "fixture")
    data_root = tmp_path / "data"
    with initialize_library(
        LibraryConfig(name="Answer projection persistence"),
        data_root=data_root,
    ) as repository:
        library_id = repository.library_id
        store = SQLiteAnswerProjectionRepository(repository)

        assert store.persist_projection(projection) == projection
        assert store.persist_projection(projection) == projection
        assert store.persist_receipt(receipt) == receipt
        assert store.persist_receipt(receipt) == receipt
        assert repository.schema_version == 13
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM answer_projections"
            ).fetchone()[0]
            == 1
        )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM answer_projection_receipts"
            ).fetchone()[0]
            == 1
        )
        assert (
            repository._store.connection.execute(
                """
                SELECT COUNT(*) FROM event_outbox
                WHERE event_type IN (
                    'answer_projection.created',
                    'answer_projection_judgment.completed'
                )
                """
            ).fetchone()[0]
            == 2
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repository._store.connection.execute(
                "UPDATE answer_projections SET author_kind = 'human' WHERE projection_id = ?",
                (projection.projection_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repository._store.connection.execute(
                "DELETE FROM answer_projection_receipts WHERE receipt_id = ?",
                (receipt.receipt_id,),
            )

    with open_library(library_id, data_root=data_root) as repository:
        reopened = SQLiteAnswerProjectionRepository(repository)
        assert reopened.get_projection(projection.projection_id) == projection
        assert reopened.get_receipt(receipt.receipt_id) == receipt


def test_receipt_requires_its_projection_in_the_same_library(tmp_path: Path) -> None:
    (tmp_path / "fixture").mkdir()
    _answer, _case_set, _entailment, _projection, receipt = _projection_bundle(tmp_path / "fixture")
    with initialize_library(
        LibraryConfig(name="Missing projection"),
        data_root=tmp_path / "data",
    ) as repository:
        store = SQLiteAnswerProjectionRepository(repository)
        with pytest.raises(AnswerProjectionNotFoundError, match="does not exist"):
            store.persist_receipt(receipt)


def test_projection_content_identity_fails_before_storage(tmp_path: Path) -> None:
    (tmp_path / "fixture").mkdir()
    _answer, _case_set, _entailment, projection, receipt = _projection_bundle(tmp_path / "fixture")
    with initialize_library(
        LibraryConfig(name="Answer projection identity"),
        data_root=tmp_path / "data",
    ) as repository:
        store = SQLiteAnswerProjectionRepository(repository)
        with pytest.raises(AnswerProjectionPersistenceError, match="content identity"):
            store.persist_projection(projection.model_copy(update={"projection_hash": "0" * 64}))
        store.persist_projection(projection)
        with pytest.raises(AnswerProjectionPersistenceError, match="content identity"):
            store.persist_receipt(receipt.model_copy(update={"receipt_hash": "0" * 64}))


def test_corrupt_projection_json_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "fixture").mkdir()
    _answer, _case_set, _entailment, projection, _receipt = _projection_bundle(tmp_path / "fixture")
    with initialize_library(
        LibraryConfig(name="Answer projection corruption"),
        data_root=tmp_path / "data",
    ) as repository:
        store = SQLiteAnswerProjectionRepository(repository)
        store.persist_projection(projection)
        connection = repository._store.connection
        row = connection.execute(
            "SELECT projection_json FROM answer_projections WHERE projection_id = ?",
            (projection.projection_id,),
        ).fetchone()
        assert row is not None
        payload = json.loads(str(row[0]))
        noncanonical = json.dumps(payload, ensure_ascii=False, indent=2)
        connection.execute("DROP TRIGGER answer_projections_no_update")
        connection.execute(
            "UPDATE answer_projections SET projection_json = ? WHERE projection_id = ?",
            (noncanonical, projection.projection_id),
        )

        with pytest.raises(AnswerProjectionPersistenceError, match="not canonical"):
            store.get_projection(projection.projection_id)


def test_projection_is_library_scoped(tmp_path: Path) -> None:
    (tmp_path / "fixture").mkdir()
    _answer, _case_set, _entailment, projection, _receipt = _projection_bundle(tmp_path / "fixture")
    with initialize_library(
        LibraryConfig(name="Answer projection source"),
        data_root=tmp_path / "source",
    ) as first:
        SQLiteAnswerProjectionRepository(first).persist_projection(projection)
    with (
        initialize_library(
            LibraryConfig(name="Answer projection other"),
            data_root=tmp_path / "other",
        ) as second,
        pytest.raises(AnswerProjectionNotFoundError, match="does not exist"),
    ):
        SQLiteAnswerProjectionRepository(second).get_projection(projection.projection_id)


def test_repository_rejects_wrong_types_and_missing_receipt(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="LibraryRepository"):
        SQLiteAnswerProjectionRepository(object())  # type: ignore[arg-type]

    with initialize_library(
        LibraryConfig(name="Answer projection type checks"),
        data_root=tmp_path / "data",
    ) as repository:
        store = SQLiteAnswerProjectionRepository(repository)
        with pytest.raises(TypeError, match="requires an AnswerProjection"):
            store.persist_projection(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="requires an AnswerProjectionJudgmentReceipt"):
            store.persist_receipt(object())  # type: ignore[arg-type]
        with pytest.raises(AnswerProjectionNotFoundError, match="does not exist"):
            store.get_receipt("answer_projection_receipt_missing")


@pytest.mark.parametrize("corruption", ["invalid", "noncanonical"])
def test_corrupt_projection_receipt_json_fails_closed(
    tmp_path: Path,
    corruption: str,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _answer, _case_set, _entailment, projection, receipt = _projection_bundle(fixture)
    with initialize_library(
        LibraryConfig(name=f"Answer receipt {corruption}"),
        data_root=tmp_path / "data",
    ) as repository:
        store = SQLiteAnswerProjectionRepository(repository)
        store.persist_projection(projection)
        store.persist_receipt(receipt)
        connection = repository._store.connection
        row = connection.execute(
            "SELECT receipt_json FROM answer_projection_receipts WHERE receipt_id = ?",
            (receipt.receipt_id,),
        ).fetchone()
        assert row is not None
        replacement = (
            "{not-json"
            if corruption == "invalid"
            else json.dumps(json.loads(str(row[0])), ensure_ascii=False, indent=2)
        )
        connection.execute("DROP TRIGGER answer_projection_receipts_no_update")
        connection.execute(
            "UPDATE answer_projection_receipts SET receipt_json = ? WHERE receipt_id = ?",
            (replacement, receipt.receipt_id),
        )

        with pytest.raises(
            AnswerProjectionPersistenceError,
            match="invalid" if corruption == "invalid" else "not canonical",
        ):
            store.get_receipt(receipt.receipt_id)
