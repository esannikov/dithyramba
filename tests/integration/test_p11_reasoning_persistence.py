"""P11 IdeaTrace migration, persistence, replay, and corruption tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    LibraryRepository,
    SQLiteReasoningRepository,
    initialize_library,
    open_library,
)
from dithyramba.reasoning import (
    IdeaTrace,
    ReasoningClosureGate,
    ReasoningClosureResult,
    ReasoningNotFoundError,
    ReasoningPersistenceError,
)

from ..unit.test_reasoning_closure import _accepted_trace


def test_trace_and_closure_are_append_only_and_reopen_exactly(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    trace, case_set, entailment = _accepted_trace(fixture_root)
    data_root = tmp_path / "data"
    with initialize_library(
        LibraryConfig(name="Reasoning persistence"),
        data_root=data_root,
    ) as repository:
        library_id = repository.library_id
        store = SQLiteReasoningRepository(repository)
        closure = ReasoningClosureGate().evaluate(
            trace,
            case_set=case_set,
            entailment=entailment,
        )

        assert store.persist_trace(trace) == trace
        assert store.persist_trace(trace) == trace
        assert store.persist_closure(closure) == closure
        assert store.persist_closure(closure) == closure
        with pytest.raises(ReasoningPersistenceError, match="content identity"):
            store.persist_trace(trace.model_copy(update={"trace_hash": "0" * 64}))
        with pytest.raises(ReasoningPersistenceError, match="content identity"):
            store.persist_closure(closure.model_copy(update={"case_set_hash": "0" * 64}))
        assert repository.schema_version == 13
        assert (
            repository._store.connection.execute("SELECT COUNT(*) FROM idea_traces").fetchone()[0]
            == 1
        )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM reasoning_closure_results"
            ).fetchone()[0]
            == 1
        )
        assert (
            repository._store.connection.execute(
                """
                SELECT COUNT(*) FROM event_outbox
                WHERE event_type IN ('idea_trace.created', 'reasoning_closure.completed')
                """
            ).fetchone()[0]
            == 2
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repository._store.connection.execute(
                "UPDATE idea_traces SET task_id = 'changed' WHERE trace_id = ?",
                (trace.trace_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            repository._store.connection.execute(
                "DELETE FROM reasoning_closure_results WHERE closure_id = ?",
                (closure.closure_id,),
            )

    with open_library(library_id, data_root=data_root) as reopened:
        store = SQLiteReasoningRepository(reopened)
        assert store.get_trace(trace.trace_id) == trace
        assert store.get_closure(closure.closure_id) == closure
        replay = ReasoningClosureGate().evaluate(
            store.get_trace(trace.trace_id),
            case_set=case_set,
            entailment=entailment,
        )
        assert replay == closure
        assert replay.closure_hash == closure.closure_hash


def test_missing_cross_library_and_corrupt_rows_fail_closed(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    trace, case_set, entailment = _accepted_trace(fixture_root)
    data_root = tmp_path / "data"
    with initialize_library(
        LibraryConfig(name="First reasoning library"),
        data_root=data_root,
    ) as first:
        closure = ReasoningClosureGate().evaluate(
            trace,
            case_set=case_set,
            entailment=entailment,
        )
        store = SQLiteReasoningRepository(first)
        store.persist_trace(trace)
        store.persist_closure(closure)
        first._store.connection.execute("DROP TRIGGER idea_traces_no_update")
        first._store.connection.execute(
            "UPDATE idea_traces SET trace_json = ? WHERE trace_id = ?",
            ('{"not":"canonical trace"}', trace.trace_id),
        )
        with pytest.raises(ReasoningPersistenceError, match="invalid"):
            store.get_trace(trace.trace_id)

    with initialize_library(
        LibraryConfig(name="Second reasoning library"),
        data_root=data_root,
    ) as second:
        store = SQLiteReasoningRepository(second)
        with pytest.raises(ReasoningNotFoundError, match="does not exist"):
            store.get_trace(trace.trace_id)
        with pytest.raises(ReasoningNotFoundError, match="does not exist"):
            store.get_closure(closure.closure_id)


def test_repository_type_guards_and_conflicting_content_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="LibraryRepository"):
        SQLiteReasoningRepository(cast(LibraryRepository, object()))

    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    trace, case_set, entailment = _accepted_trace(fixture_root)
    closure = ReasoningClosureGate().evaluate(
        trace,
        case_set=case_set,
        entailment=entailment,
    )
    with initialize_library(
        LibraryConfig(name="Reasoning type guards"),
        data_root=tmp_path / "data",
    ) as repository:
        store = SQLiteReasoningRepository(repository)
        with pytest.raises(TypeError, match="requires an IdeaTrace"):
            store.persist_trace(cast(IdeaTrace, object()))
        with pytest.raises(TypeError, match="requires a ReasoningClosureResult"):
            store.persist_closure(cast(ReasoningClosureResult, object()))

        different_trace = trace.model_copy(update={"question": "Different public question."})
        monkeypatch.setattr(store, "_optional_trace", lambda _trace_id: different_trace)
        with pytest.raises(ReasoningPersistenceError, match="conflicts"):
            store.persist_trace(trace)

        monkeypatch.undo()
        store.persist_trace(trace)
        different_closure = closure.model_copy(
            update={"review_reasons": ("Different persisted finding.",)}
        )
        monkeypatch.setattr(store, "_optional_closure", lambda _closure_id: different_closure)
        with pytest.raises(ReasoningPersistenceError, match="conflicts"):
            store.persist_closure(closure)


def test_duplicate_sql_writes_and_stale_closure_binding_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    trace, case_set, entailment = _accepted_trace(fixture_root)
    closure = ReasoningClosureGate().evaluate(
        trace,
        case_set=case_set,
        entailment=entailment,
    )
    with initialize_library(
        LibraryConfig(name="Reasoning duplicate writes"),
        data_root=tmp_path / "data",
    ) as repository:
        store = SQLiteReasoningRepository(repository)
        store.persist_trace(trace)
        monkeypatch.setattr(store, "_optional_trace", lambda _trace_id: None)
        with pytest.raises(ReasoningPersistenceError, match="append-only write conflicted"):
            store.persist_trace(trace)

        monkeypatch.undo()
        store.persist_closure(closure)
        monkeypatch.setattr(store, "_optional_closure", lambda _closure_id: None)
        with pytest.raises(ReasoningPersistenceError, match="append-only write conflicted"):
            store.persist_closure(closure)

        stale = closure.model_copy(update={"trace_hash": "0" * 64})
        monkeypatch.setattr(store, "_optional_closure", lambda _closure_id: stale)
        with pytest.raises(ReasoningPersistenceError, match="stale IdeaTrace binding"):
            store.get_closure(closure.closure_id)


@pytest.mark.parametrize("artifact", ["trace", "closure"])
def test_noncanonical_reasoning_json_is_rejected(tmp_path: Path, artifact: str) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    trace, case_set, entailment = _accepted_trace(fixture_root)
    closure = ReasoningClosureGate().evaluate(
        trace,
        case_set=case_set,
        entailment=entailment,
    )
    with initialize_library(
        LibraryConfig(name=f"Reasoning noncanonical {artifact}"),
        data_root=tmp_path / "data",
    ) as repository:
        store = SQLiteReasoningRepository(repository)
        store.persist_trace(trace)
        store.persist_closure(closure)
        if artifact == "trace":
            repository._store.connection.execute("DROP TRIGGER idea_traces_no_update")
            repository._store.connection.execute(
                "UPDATE idea_traces SET trace_json = ? WHERE trace_id = ?",
                ("\n" + trace.model_dump_json(), trace.trace_id),
            )
            with pytest.raises(ReasoningPersistenceError, match="not canonical JSON"):
                store.get_trace(trace.trace_id)
        else:
            repository._store.connection.execute("DROP TRIGGER reasoning_closure_results_no_update")
            repository._store.connection.execute(
                "UPDATE reasoning_closure_results SET closure_json = ? WHERE closure_id = ?",
                ("\n" + closure.model_dump_json(), closure.closure_id),
            )
            with pytest.raises(ReasoningPersistenceError, match="not canonical JSON"):
                store.get_closure(closure.closure_id)
