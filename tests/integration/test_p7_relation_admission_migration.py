"""Schema-v8 Relation import membership and review-admission integration tests."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from dithyramba.relations import RelationImportRequest, RelationIntegrityError
from dithyramba.store import MigrationRunner, discover_migrations
from dithyramba.store.errors import MigrationApplyError
from tests.integration.test_migrations import _insert_complete_chain
from tests.integration.test_p7_relations import (
    NOW_TEXT,
    _context,
    _import_request,
    _proposals,
)

_ROOT = Path(__file__).resolve().parents[2]
_HASHES = tuple(f"{number:064x}" for number in range(100, 132))


def _v7_source(tmp_path: Path) -> Path:
    source = tmp_path / "v7-migrations"
    source.mkdir()
    for migration in discover_migrations(_ROOT / "migrations")[:7]:
        shutil.copyfile(_ROOT / "migrations" / migration.name, source / migration.name)
    return source


def _open_v7(tmp_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(tmp_path / "v7.sqlite3", isolation_level=None)
    connection.execute("PRAGMA foreign_keys = ON")
    MigrationRunner(connection, _v7_source(tmp_path)).apply_all()
    return connection


def _seed_v7_relations(connection: sqlite3.Connection) -> None:
    _insert_complete_chain(connection)
    connection.execute(
        """
        INSERT INTO meaning_processing_runs VALUES (
            'run_meaning_relation_backfill', ?, 'meaning_import', ?, ?, ?, ?, 'deep',
            'p7', '{}', ?, '{}', ?, ?, ?, ?, 'succeeded', 3, 0, 0, NULL,
            '{}', ?, ?, ?, ?
        )
        """,
        (
            "library_test",
            "snapshot_test",
            "policy_test",
            _HASHES[0],
            _HASHES[1],
            _HASHES[2],
            _HASHES[3],
            _HASHES[4],
            _HASHES[5],
            _HASHES[6],
            _HASHES[7],
            _HASHES[8],
            NOW_TEXT,
            NOW_TEXT,
        ),
    )
    connection.execute(
        """
        INSERT INTO voices VALUES (
            'voice_relation_backfill', 'library_test', 'author', 'Backfill author',
            'candidate', ?, 'run_meaning_relation_backfill', ?
        )
        """,
        (_HASHES[9], NOW_TEXT),
    )
    for number in range(2):
        connection.execute(
            """
            INSERT INTO statements VALUES (
                ?, 'library_test', 'collection_test', 'source_claim', ?,
                'voice_relation_backfill', NULL, 'candidate', 'active', ?,
                'run_meaning_relation_backfill', ?
            )
            """,
            (
                f"statement_relation_backfill_{number}",
                f"Backfill statement {number}",
                _HASHES[10 + number],
                NOW_TEXT,
            ),
        )

    runs = (
        ("relation_run_backfill_a", 2, _HASHES[12], _HASHES[14], _HASHES[16]),
        ("relation_run_backfill_b", 1, _HASHES[13], _HASHES[15], _HASHES[17]),
    )
    for run_id, relation_count, input_hash, output_hash, receipt_hash in runs:
        connection.execute(
            """
            INSERT INTO relation_import_runs VALUES (
                ?, 'library_test', 'snapshot_test', 'policy_test', ?, ?, ?,
                'p7', 'relations/1.0', ?, ?, ?, ?, ?
            )
            """,
            (
                run_id,
                _HASHES[18],
                _HASHES[19],
                _HASHES[20],
                input_hash,
                relation_count,
                output_hash,
                receipt_hash,
                NOW_TEXT,
            ),
        )

    relation_rows = (
        ("relation_backfill_z", "relation_run_backfill_a", _HASHES[21]),
        ("relation_backfill_a", "relation_run_backfill_a", _HASHES[22]),
        ("relation_backfill_m", "relation_run_backfill_b", _HASHES[23]),
    )
    for relation_id, run_id, content_hash in relation_rows:
        connection.execute(
            """
            INSERT INTO relations VALUES (
                ?, ?, 'library_test', 'supports',
                'statement', 'statement_relation_backfill_0',
                'statement', 'statement_relation_backfill_1',
                'Synthetic migration row', 'test/1.0', NULL, NULL,
                1, 'candidate', ?, ?
            )
            """,
            (relation_id, run_id, content_hash, NOW_TEXT),
        )


def _different_request(request: RelationImportRequest, version: str) -> RelationImportRequest:
    return replace(request, code_version=version)


def _insert_review_session(
    connection: sqlite3.Connection,
    *,
    review_session_id: str = "review_session_relations",
    library_id: str,
    decision_limit: int = 10,
) -> None:
    connection.execute(
        """
        INSERT INTO meaning_review_sessions(
            review_session_id, library_id, review_budget_json,
            review_budget_hash, decision_limit, created_at
        ) VALUES (?, ?, '{}', ?, ?, ?)
        """,
        (review_session_id, library_id, _HASHES[24], decision_limit, NOW_TEXT),
    )


def _insert_relation_review(
    connection: sqlite3.Connection,
    *,
    review_decision_id: str,
    review_session_id: str,
    library_id: str,
    relation_id: str,
    target_hash: str,
    action: str = "accept",
    scope_json: str = "{}",
    replacement_target_id: str | None = None,
    supersedes_review_decision_id: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO relation_review_decisions(
            review_decision_id, review_session_id, library_id,
            target_type, target_id, target_hash, action, reason,
            authority, scope_json, replacement_target_id,
            supersedes_review_decision_id, created_at
        ) VALUES (?, ?, ?, 'relation', ?, ?, ?, 'Reviewed', 'tester', ?, ?, ?, ?)
        """,
        (
            review_decision_id,
            review_session_id,
            library_id,
            relation_id,
            target_hash,
            action,
            scope_json,
            replacement_target_id,
            supersedes_review_decision_id,
            NOW_TEXT,
        ),
    )


def test_schema_v8_fresh_apply_and_migration_copies_are_identical() -> None:
    root_copy = _ROOT / "migrations" / "0008_relation_admission.sql"
    package_copy = _ROOT / "src/dithyramba/store/sql/0008_relation_admission.sql"
    assert root_copy.read_bytes() == package_copy.read_bytes()

    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        runner = MigrationRunner(connection, _ROOT / "migrations")
        assert [migration.version for migration in runner.apply_all()] == list(range(1, 12))
        assert runner.current_version() == 11
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            ).fetchall()
        }
        assert {"relation_import_run_items", "relation_review_decisions"} <= tables
    finally:
        connection.close()


def test_v7_to_v8_zero_relation_rows_is_backup_first(tmp_path: Path) -> None:
    connection = _open_v7(tmp_path)
    try:
        backed_up: list[int] = []
        runner = MigrationRunner(connection, _ROOT / "migrations")
        applied = runner.apply_all(
            backup_hook=lambda _connection, migration: backed_up.append(migration.version)
        )
        assert [migration.version for migration in applied] == [8, 9, 10, 11]
        assert backed_up == [8, 9, 10, 11]
        assert runner.current_version() == 11
        assert (
            connection.execute("SELECT COUNT(*) FROM relation_import_run_items").fetchone()[0] == 0
        )
    finally:
        connection.close()


def test_v7_to_v8_backfills_exact_prior_run_outputs_in_relation_id_order(
    tmp_path: Path,
) -> None:
    connection = _open_v7(tmp_path)
    try:
        _seed_v7_relations(connection)
        prior_outputs = {
            str(run[0]): tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT relation_id FROM relations "
                    "WHERE relation_import_run_id = ? ORDER BY relation_id",
                    (str(run[0]),),
                ).fetchall()
            )
            for run in connection.execute(
                "SELECT relation_import_run_id FROM relation_import_runs"
            ).fetchall()
        }
        lineage_before = connection.execute(
            "SELECT relation_id, relation_import_run_id FROM relations ORDER BY relation_id"
        ).fetchall()

        runner = MigrationRunner(connection, _ROOT / "migrations")
        runner.apply_all(backup_hook=lambda _connection, _migration: None)

        backfilled_outputs = {
            str(run[0]): tuple(
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT relation_id
                    FROM relation_import_run_items
                    WHERE relation_import_run_id = ?
                    ORDER BY output_order
                    """,
                    (str(run[0]),),
                ).fetchall()
            )
            for run in connection.execute(
                "SELECT relation_import_run_id FROM relation_import_runs"
            ).fetchall()
        }
        assert backfilled_outputs == prior_outputs
        assert connection.execute(
            """
            SELECT relation_import_run_id, relation_id, output_order
            FROM relation_import_run_items
            ORDER BY relation_import_run_id, output_order
            """
        ).fetchall() == [
            ("relation_run_backfill_a", "relation_backfill_a", 0),
            ("relation_run_backfill_a", "relation_backfill_z", 1),
            ("relation_run_backfill_b", "relation_backfill_m", 0),
        ]
        assert (
            connection.execute(
                "SELECT relation_id, relation_import_run_id FROM relations ORDER BY relation_id"
            ).fetchall()
            == lineage_before
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def test_v7_to_v8_refuses_incomplete_declared_relation_run_backfill(
    tmp_path: Path,
) -> None:
    connection = _open_v7(tmp_path)
    try:
        _seed_v7_relations(connection)
        connection.execute("DROP TRIGGER relation_import_runs_no_update")
        connection.execute(
            "UPDATE relation_import_runs SET relation_count = 3 "
            "WHERE relation_import_run_id = 'relation_run_backfill_a'"
        )
        connection.execute(
            "CREATE TRIGGER relation_import_runs_no_update BEFORE UPDATE ON "
            "relation_import_runs BEGIN SELECT RAISE(ABORT, "
            "'relation_import_runs is append-only'); END"
        )

        runner = MigrationRunner(connection, _ROOT / "migrations")
        with pytest.raises(MigrationApplyError, match="failed to apply"):
            runner.apply_all(backup_hook=lambda _connection, _migration: None)

        assert runner.current_version() == 7
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_schema "
                "WHERE type = 'table' AND name = 'relation_import_run_items'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_v7_to_v8_refuses_legacy_review_session_over_budget(tmp_path: Path) -> None:
    connection = _open_v7(tmp_path)
    try:
        _seed_v7_relations(connection)
        statement_hash = str(
            connection.execute(
                "SELECT content_hash FROM statements "
                "WHERE statement_id = 'statement_relation_backfill_0'"
            ).fetchone()[0]
        )
        connection.execute(
            "INSERT INTO meaning_review_sessions VALUES "
            "('review_session_overbudget', 'library_test', '{}', ?, 1, ?)",
            (_HASHES[24], NOW_TEXT),
        )
        for suffix, use in (("one", "research"), ("two", "publication")):
            scope_json = (
                '{"collection_ids":["collection_test"],'
                '"schema":"dithyramba.meaning_review_scope/1.0",'
                f'"use":"{use}"}}'
            )
            connection.execute(
                """
                INSERT INTO meaning_review_decisions VALUES (
                    ?, 'review_session_overbudget', 'library_test', 'statement',
                    'statement_relation_backfill_0', ?, 'accept', 'Reviewed',
                    'tester', ?, NULL, NULL, ?
                )
                """,
                (f"review_overbudget_{suffix}", statement_hash, scope_json, NOW_TEXT),
            )

        runner = MigrationRunner(connection, _ROOT / "migrations")
        with pytest.raises(MigrationApplyError, match="failed to apply"):
            runner.apply_all(backup_hook=lambda _connection, _migration: None)
        assert runner.current_version() == 7
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_schema "
                "WHERE type = 'table' AND name = 'relation_review_decisions'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_relation_import_membership_constraints_and_append_only_triggers(
    tmp_path: Path,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        result = relations.import_relations(
            _import_request(snapshot_id, policy, scope), (_proposals(statements, evidence)[0],)
        )
        connection = repository._store.connection
        run_id = result.receipt.import_run_id
        relation_id = result.relations[0].relation_id

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            connection.execute(
                "INSERT INTO relation_import_run_items VALUES (?, ?, 0)",
                (run_id, relation_id),
            )

        connection.execute(
            """
            INSERT INTO relation_import_runs
            SELECT 'relation_run_output_bounds', library_id, corpus_snapshot_id,
                   access_policy_id, policy_hash, scope_hash, permitted_set_hash,
                   code_version, profile_version, ?, 1, output_hash, ?, created_at
            FROM relation_import_runs WHERE relation_import_run_id = ?
            """,
            (_HASHES[25], _HASHES[26], run_id),
        )
        with pytest.raises(sqlite3.IntegrityError, match="output bounds"):
            connection.execute(
                "INSERT INTO relation_import_run_items VALUES ('relation_run_output_bounds', ?, 1)",
                (relation_id,),
            )

        connection.execute("DROP TRIGGER libraries_one_row_insert")
        connection.execute(
            "INSERT INTO libraries VALUES ('library_other', 'Other', ?, ?)",
            (_HASHES[27], NOW_TEXT),
        )
        connection.execute(
            """
            INSERT INTO relation_import_runs
            SELECT 'relation_run_other_library', 'library_other', corpus_snapshot_id,
                   access_policy_id, policy_hash, scope_hash, permitted_set_hash,
                   code_version, profile_version, input_hash, 1, output_hash, ?, created_at
            FROM relation_import_runs WHERE relation_import_run_id = ?
            """,
            (_HASHES[28], run_id),
        )
        with pytest.raises(sqlite3.IntegrityError, match="same-Library"):
            connection.execute(
                "INSERT INTO relation_import_run_items VALUES ('relation_run_other_library', ?, 0)",
                (relation_id,),
            )

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE relation_import_run_items SET output_order = 0 "
                "WHERE relation_import_run_id = ? AND relation_id = ?",
                (run_id, relation_id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM relation_import_run_items "
                "WHERE relation_import_run_id = ? AND relation_id = ?",
                (run_id, relation_id),
            )
    finally:
        repository.close()


def test_same_run_is_idempotent_and_distinct_run_reuses_canonical_relation(
    tmp_path: Path,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        proposal = (_proposals(statements, evidence)[0],)
        request = _import_request(snapshot_id, policy, scope)
        first = relations.import_relations(request, proposal)
        assert relations.import_relations(request, proposal) == first

        second = relations.import_relations(_different_request(request, "p7-rerun"), proposal)
        assert second.receipt.import_run_id != first.receipt.import_run_id
        assert second.relations[0].relation_id == first.relations[0].relation_id
        assert second.relations[0].creation_import_run_id == first.receipt.import_run_id
        assert second.relations[0].payload()["import_run_id"] == first.receipt.import_run_id
        assert relations.load_import_result(second.receipt.import_run_id) == second

        connection = repository._store.connection
        assert connection.execute("SELECT COUNT(*) FROM relation_import_runs").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM relation_import_run_items").fetchone()[0] == 2
        )
        assert (
            connection.execute(
                "SELECT relation_import_run_id FROM relations WHERE relation_id = ?",
                (first.relations[0].relation_id,),
            ).fetchone()[0]
            == first.receipt.import_run_id
        )
    finally:
        repository.close()


def test_mixed_existing_and_new_relations_preserve_creation_lineage(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        proposals = _proposals(statements, evidence)
        request = _import_request(snapshot_id, policy, scope)
        first = relations.import_relations(request, (proposals[0],))
        mixed = relations.import_relations(
            _different_request(request, "p7-mixed"),
            proposals,
        )
        by_id = {relation.relation_id: relation for relation in mixed.relations}
        reused_id = first.relations[0].relation_id
        new_relation = next(
            relation for relation in mixed.relations if relation.relation_id != reused_id
        )
        assert by_id[reused_id].creation_import_run_id == first.receipt.import_run_id
        assert new_relation.creation_import_run_id == mixed.receipt.import_run_id
        assert relations.load_import_result(mixed.receipt.import_run_id) == mixed

        connection = repository._store.connection
        assert connection.execute("SELECT COUNT(*) FROM relation_import_runs").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0] == 2
        assert (
            connection.execute("SELECT COUNT(*) FROM relation_import_run_items").fetchone()[0] == 3
        )
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("relation_count", 2, "count mismatch"),
        ("output_hash", "f" * 64, "output hash mismatch"),
        ("receipt_hash", "e" * 64, "receipt hash mismatch"),
    ),
)
def test_import_loader_fails_closed_on_run_integrity_corruption(
    tmp_path: Path,
    column: str,
    value: object,
    message: str,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        result = relations.import_relations(
            _import_request(snapshot_id, policy, scope), (_proposals(statements, evidence)[0],)
        )
        connection = repository._store.connection
        connection.execute("DROP TRIGGER relation_import_runs_no_update")
        connection.execute(
            f'UPDATE relation_import_runs SET "{column}" = ? WHERE relation_import_run_id = ?',
            (value, result.receipt.import_run_id),
        )
        with pytest.raises(RelationIntegrityError, match=message):
            relations.load_import_result(result.receipt.import_run_id)
    finally:
        repository.close()


def test_import_loader_fails_closed_on_missing_run_item(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        request = _import_request(snapshot_id, policy, scope)
        proposals = (_proposals(statements, evidence)[0],)
        result = relations.import_relations(request, proposals)
        connection = repository._store.connection
        connection.execute("DROP TRIGGER relation_import_run_items_no_delete")
        connection.execute(
            "DELETE FROM relation_import_run_items WHERE relation_import_run_id = ?",
            (result.receipt.import_run_id,),
        )
        with pytest.raises(RelationIntegrityError, match="count mismatch"):
            relations.load_import_result(result.receipt.import_run_id)
        with pytest.raises(RelationIntegrityError, match="count mismatch"):
            relations.import_relations(request, proposals)
        assert connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM relation_import_runs").fetchone()[0] == 1
    finally:
        repository.close()


def test_import_loader_fails_closed_on_noncanonical_output_order(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        result = relations.import_relations(
            _import_request(snapshot_id, policy, scope), _proposals(statements, evidence)
        )
        connection = repository._store.connection
        connection.execute("DROP TRIGGER relation_import_run_items_no_delete")
        connection.execute(
            "DELETE FROM relation_import_run_items WHERE relation_import_run_id = ?",
            (result.receipt.import_run_id,),
        )
        connection.executemany(
            "INSERT INTO relation_import_run_items VALUES (?, ?, ?)",
            (
                (result.receipt.import_run_id, relation_id, output_order)
                for output_order, relation_id in enumerate(reversed(result.receipt.relation_ids))
            ),
        )
        with pytest.raises(RelationIntegrityError, match="output order is not canonical"):
            relations.load_import_result(result.receipt.import_run_id)
    finally:
        repository.close()


def test_relation_import_rolls_back_run_relation_evidence_and_membership(
    tmp_path: Path,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        request = _import_request(snapshot_id, policy, scope)
        proposals = _proposals(statements, evidence)
        relations.import_relations(request, (proposals[0],))
        connection = repository._store.connection
        before = tuple(
            connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in (
                "relation_import_runs",
                "relations",
                "relation_evidence_links",
                "relation_import_run_items",
            )
        )
        connection.execute(
            """
            CREATE TRIGGER relation_import_run_items_injected_failure
            BEFORE INSERT ON relation_import_run_items
            BEGIN
                SELECT RAISE(ABORT, 'injected membership failure');
            END
            """
        )

        with pytest.raises(RelationIntegrityError, match="insert conflicted"):
            relations.import_relations(
                _different_request(request, "p7-rollback"),
                (proposals[1],),
            )
        after = tuple(
            connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in (
                "relation_import_runs",
                "relations",
                "relation_evidence_links",
                "relation_import_run_items",
            )
        )
        assert after == before
    finally:
        repository.close()


def test_relation_review_extension_enforces_hash_library_supersede_and_append_only(
    tmp_path: Path,
) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        result = relations.import_relations(
            _import_request(snapshot_id, policy, scope), (_proposals(statements, evidence)[0],)
        )
        relation = result.relations[0]
        connection = repository._store.connection
        _insert_review_session(connection, library_id=repository.library_id)

        with pytest.raises(sqlite3.IntegrityError, match="target hash"):
            _insert_relation_review(
                connection,
                review_decision_id="review_relation_wrong_hash",
                review_session_id="review_session_relations",
                library_id=repository.library_id,
                relation_id=relation.relation_id,
                target_hash="f" * 64,
            )

        connection.execute("DROP TRIGGER libraries_one_row_insert")
        connection.execute(
            "INSERT INTO libraries VALUES ('library_other', 'Other', ?, ?)",
            (_HASHES[27], NOW_TEXT),
        )
        _insert_review_session(
            connection,
            review_session_id="review_session_other",
            library_id="library_other",
        )
        with pytest.raises(sqlite3.IntegrityError, match="session must belong"):
            _insert_relation_review(
                connection,
                review_decision_id="review_relation_cross_library",
                review_session_id="review_session_other",
                library_id=repository.library_id,
                relation_id=relation.relation_id,
                target_hash=relation.content_hash,
            )

        _insert_relation_review(
            connection,
            review_decision_id="review_relation_prior",
            review_session_id="review_session_relations",
            library_id=repository.library_id,
            relation_id=relation.relation_id,
            target_hash=relation.content_hash,
        )
        with pytest.raises(sqlite3.IntegrityError, match="retain Library, target, hash"):
            _insert_relation_review(
                connection,
                review_decision_id="review_relation_wrong_scope",
                review_session_id="review_session_relations",
                library_id=repository.library_id,
                relation_id=relation.relation_id,
                target_hash=relation.content_hash,
                action="supersede",
                scope_json='{"use":"other"}',
                supersedes_review_decision_id="review_relation_prior",
            )
        _insert_relation_review(
            connection,
            review_decision_id="review_relation_supersede",
            review_session_id="review_session_relations",
            library_id=repository.library_id,
            relation_id=relation.relation_id,
            target_hash=relation.content_hash,
            action="supersede",
            supersedes_review_decision_id="review_relation_prior",
        )
        with pytest.raises(sqlite3.IntegrityError, match="already superseded"):
            _insert_relation_review(
                connection,
                review_decision_id="review_relation_supersede_twice",
                review_session_id="review_session_relations",
                library_id=repository.library_id,
                relation_id=relation.relation_id,
                target_hash=relation.content_hash,
                action="supersede",
                supersedes_review_decision_id="review_relation_prior",
            )
        with pytest.raises(sqlite3.IntegrityError, match="retain Library, target, hash"):
            _insert_relation_review(
                connection,
                review_decision_id="review_relation_supersede_disposition",
                review_session_id="review_session_relations",
                library_id=repository.library_id,
                relation_id=relation.relation_id,
                target_hash=relation.content_hash,
                action="supersede",
                supersedes_review_decision_id="review_relation_supersede",
            )

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE relation_review_decisions SET reason = 'changed' "
                "WHERE review_decision_id = 'review_relation_prior'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM relation_review_decisions "
                "WHERE review_decision_id = 'review_relation_prior'"
            )
    finally:
        repository.close()


def test_relation_review_budget_counts_meaning_and_relation_decisions(tmp_path: Path) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = _context(tmp_path)
    try:
        result = relations.import_relations(
            _import_request(snapshot_id, policy, scope), (_proposals(statements, evidence)[0],)
        )
        connection = repository._store.connection
        _insert_review_session(
            connection,
            review_session_id="review_session_shared_budget",
            library_id=repository.library_id,
            decision_limit=1,
        )
        statement_hash = str(
            connection.execute(
                "SELECT content_hash FROM statements WHERE statement_id = ?",
                (statements[0],),
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO meaning_review_decisions VALUES (
                'review_meaning_budget', 'review_session_shared_budget', ?,
                'statement', ?, ?, 'accept', 'Reviewed', 'tester', '{}',
                NULL, NULL, ?
            )
            """,
            (repository.library_id, statements[0], statement_hash, NOW_TEXT),
        )
        with pytest.raises(sqlite3.IntegrityError, match="budget is exhausted"):
            _insert_relation_review(
                connection,
                review_decision_id="review_relation_over_budget",
                review_session_id="review_session_shared_budget",
                library_id=repository.library_id,
                relation_id=result.relations[0].relation_id,
                target_hash=result.relations[0].content_hash,
            )

        _insert_review_session(
            connection,
            review_session_id="review_session_reverse_shared_budget",
            library_id=repository.library_id,
            decision_limit=1,
        )
        _insert_relation_review(
            connection,
            review_decision_id="review_relation_budget_first",
            review_session_id="review_session_reverse_shared_budget",
            library_id=repository.library_id,
            relation_id=result.relations[0].relation_id,
            target_hash=result.relations[0].content_hash,
        )
        second_statement_hash = str(
            connection.execute(
                "SELECT content_hash FROM statements WHERE statement_id = ?",
                (statements[1],),
            ).fetchone()[0]
        )
        with pytest.raises(sqlite3.IntegrityError, match="budget is exhausted"):
            connection.execute(
                """
                INSERT INTO meaning_review_decisions VALUES (
                    'review_meaning_over_budget',
                    'review_session_reverse_shared_budget', ?,
                    'statement', ?, ?, 'accept', 'Reviewed', 'tester', '{}',
                    NULL, NULL, ?
                )
                """,
                (repository.library_id, statements[1], second_statement_hash, NOW_TEXT),
            )
    finally:
        repository.close()
