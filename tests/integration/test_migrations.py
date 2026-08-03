"""Acceptance tests for the compact v1 SQLite schema and migration boundary."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from dithyramba.store import (
    IntegrityCheckError,
    LegacySchemaError,
    MigrationBackupRequiredError,
    MigrationChecksumError,
    MigrationDiscoveryError,
    MigrationRunner,
    MigrationStateError,
    NewerSchemaError,
    Store,
    discover_migrations,
    packaged_migration_source,
    schema_fingerprint,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
NOW = "2026-07-20T12:00:00.000000Z"

EXPECTED_TABLES = {
    "access_policies",
    "access_policy_collection_rules",
    "access_policy_source_rules",
    "access_receipts",
    "answer_projection_receipts",
    "answer_projections",
    "blobs",
    "collection_memberships",
    "collection_root_identity_manifests",
    "collection_roots",
    "collections",
    "concept_aliases",
    "concept_meaning_statements",
    "concept_meanings",
    "concept_mentions",
    "concepts",
    "corpus_read_set_items",
    "corpus_read_sets",
    "corpus_snapshots",
    "coverage_reports",
    "entities",
    "entity_aliases",
    "entity_mentions",
    "event_outbox",
    "evidence_links",
    "evidence_packets",
    "idea_traces",
    "libraries",
    "meaning_coverage_reports",
    "meaning_processing_runs",
    "meaning_read_receipt_items",
    "meaning_read_receipts",
    "meaning_review_decisions",
    "meaning_review_sessions",
    "omissions",
    "packet_items",
    "processing_runs",
    "query_requests",
    "read_receipt_corpus_sets",
    "read_receipt_items",
    "read_receipts",
    "reasoning_closure_results",
    "recall_run_artifacts",
    "relation_evidence_links",
    "relation_import_run_items",
    "relation_import_runs",
    "relation_path_collections",
    "relation_path_exclusions",
    "relation_path_receipts",
    "relation_path_relation_types",
    "relation_path_seeds",
    "relation_path_steps",
    "relation_paths",
    "relation_review_decisions",
    "relations",
    "research_session_events",
    "research_sessions",
    "retrieval_receipts",
    "review_decisions",
    "schema_migrations",
    "snapshot_members",
    "source_families",
    "source_family_identities",
    "source_family_members",
    "source_fragments",
    "source_heads",
    "source_identity_bindings",
    "source_versions",
    "sources",
    "statements",
    "structure_unit_generations",
    "structure_unit_members",
    "structure_unit_read_receipt_items",
    "structure_unit_read_receipts",
    "structure_units",
    "time_contexts",
    "voices",
}

RETIRED_TABLE_PREFIXES = (
    "corpus_layer",
    "corpus_vector",
    "embedding_model",
    "fusion_profile",
    "hybrid_",
    "model_runtime",
    "reranker",
    "semantic_span",
    "source_fragment_text_metrics",
)

FROZEN_TABLES = {
    "access_policies",
    "access_policy_collection_rules",
    "access_policy_source_rules",
    "access_receipts",
    "answer_projection_receipts",
    "answer_projections",
    "blobs",
    "collection_root_identity_manifests",
    "concept_aliases",
    "concept_meaning_statements",
    "concept_meanings",
    "concept_mentions",
    "concepts",
    "corpus_read_set_items",
    "corpus_read_sets",
    "corpus_snapshots",
    "coverage_reports",
    "entities",
    "entity_aliases",
    "entity_mentions",
    "evidence_links",
    "evidence_packets",
    "idea_traces",
    "libraries",
    "meaning_coverage_reports",
    "meaning_processing_runs",
    "meaning_read_receipt_items",
    "meaning_read_receipts",
    "meaning_review_decisions",
    "meaning_review_sessions",
    "omissions",
    "packet_items",
    "query_requests",
    "read_receipt_corpus_sets",
    "read_receipt_items",
    "read_receipts",
    "reasoning_closure_results",
    "recall_run_artifacts",
    "relation_evidence_links",
    "relation_import_run_items",
    "relation_import_runs",
    "relation_path_collections",
    "relation_path_exclusions",
    "relation_path_receipts",
    "relation_path_relation_types",
    "relation_path_seeds",
    "relation_path_steps",
    "relation_paths",
    "relation_review_decisions",
    "relations",
    "research_session_events",
    "research_sessions",
    "retrieval_receipts",
    "review_decisions",
    "schema_migrations",
    "snapshot_members",
    "source_families",
    "source_family_identities",
    "source_family_members",
    "source_fragments",
    "source_identity_bindings",
    "source_versions",
    "sources",
    "statements",
    "structure_unit_generations",
    "structure_unit_members",
    "structure_unit_read_receipt_items",
    "structure_unit_read_receipts",
    "structure_units",
    "time_contexts",
    "voices",
}


def test_v1_baseline_is_exact_and_second_apply_is_noop(tmp_path: Path) -> None:
    migrations = discover_migrations()
    assert len(migrations) == 1
    assert migrations[0].name == "0001_v1.sql"

    with Store.open(tmp_path / "memory.sqlite3") as store:
        runner = MigrationRunner(store.connection)
        tables = {
            str(row[0])
            for row in store.connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }

        assert tables == EXPECTED_TABLES
        assert not any(name.startswith(RETIRED_TABLE_PREFIXES) for name in tables)
        assert runner.current_version() == 1
        assert runner.apply_all() == ()
        assert store.schema_version == 1


def test_packaged_baseline_is_the_only_runtime_schema_source() -> None:
    entries = tuple(item.name for item in packaged_migration_source().iterdir() if item.is_file())
    assert "0001_v1.sql" in entries
    assert not any(name.endswith(".sql") and name != "0001_v1.sql" for name in entries)


def test_store_applies_required_sqlite_profile(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
        assert connection.execute("PRAGMA trusted_schema").fetchone()[0] == 0


def test_source_heads_enforce_same_source_and_allow_audited_reversion(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        connection.execute(
            "INSERT INTO libraries VALUES ('library_one', 'One', ?, ?)", (HASH_A, NOW)
        )
        for suffix, digest in (("one", HASH_A), ("two", HASH_B)):
            connection.execute(
                "INSERT OR IGNORE INTO blobs VALUES (?, 1, ?, ?)",
                (digest, f"aa/{digest}", NOW),
            )
            connection.execute(
                "INSERT INTO sources VALUES (?, 'library_one', ?, 'text/plain', ?, ?)",
                (f"source_{suffix}", f"file:///{suffix}.txt", suffix, NOW),
            )
            connection.execute(
                "INSERT INTO source_versions VALUES (?, ?, 1, ?, 1, ?, ?, 'index/1.0', "
                "'processed', NULL)",
                (f"source_version_{suffix}", f"source_{suffix}", digest, NOW, NOW),
            )

        with pytest.raises(sqlite3.IntegrityError, match="another Source"):
            connection.execute(
                "INSERT INTO source_heads VALUES ('source_one', 'source_version_two', ?, ?)",
                (NOW, NOW),
            )
        connection.execute(
            "INSERT INTO source_heads VALUES ('source_one', 'source_version_one', ?, ?)",
            (NOW, NOW),
        )
        with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
            connection.execute("DELETE FROM source_heads WHERE source_id = 'source_one'")


def test_generic_future_migration_is_backup_first(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_initial.sql").write_text(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT, sha256 TEXT, "
        "applied_at TEXT); CREATE TABLE one(value TEXT);",
        encoding="utf-8",
    )
    connection = sqlite3.connect(tmp_path / "future.sqlite3", isolation_level=None)
    try:
        MigrationRunner(connection, migrations).apply_all()
        (migrations / "0002_more.sql").write_text("CREATE TABLE two(value TEXT);", encoding="utf-8")
        runner = MigrationRunner(connection, migrations)
        with pytest.raises(MigrationBackupRequiredError, match="verified backup"):
            runner.apply_all()

        backed_up: list[int] = []
        applied = runner.apply_all(
            backup_hook=lambda _connection, migration: backed_up.append(migration.version)
        )
        assert [item.version for item in applied] == [2]
        assert backed_up == [2]
    finally:
        connection.close()


def test_mutated_applied_migration_fails_closed(tmp_path: Path) -> None:
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    migration_file = migration_dir / "0001_initial.sql"
    migration_file.write_text(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT, sha256 TEXT, "
        "applied_at TEXT); CREATE TABLE item(value TEXT);",
        encoding="utf-8",
    )
    connection = sqlite3.connect(tmp_path / "memory.sqlite3", isolation_level=None)
    try:
        MigrationRunner(connection, migration_dir).apply_all()
        migration_file.write_text(
            migration_file.read_text(encoding="utf-8") + "\n-- forbidden mutation\n",
            encoding="utf-8",
        )
        with pytest.raises(MigrationChecksumError, match="checksum"):
            MigrationRunner(connection, migration_dir).verify()
    finally:
        connection.close()


def test_pre_v1_library_is_rejected_before_any_file_write(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_file = legacy_dir / "0001_legacy.sql"
    legacy_file.write_text(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT, sha256 TEXT, "
        "applied_at TEXT); CREATE TABLE libraries(value TEXT);",
        encoding="utf-8",
    )
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        MigrationRunner(connection, legacy_dir).apply_all()
        connection.execute("UPDATE schema_migrations SET name = '0001_core.sql' WHERE version = 1")
    finally:
        connection.close()
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    with pytest.raises(LegacySchemaError, match="pre-v1 Library"):
        Store.open(database)

    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert not database.with_name(database.name + "-wal").exists()


def test_newer_database_schema_fails_closed(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        store.connection.execute(
            "INSERT INTO schema_migrations(version, name, sha256, applied_at) "
            "VALUES (2, '0002_future.sql', ?, ?)",
            (HASH_A, NOW),
        )
        with pytest.raises(NewerSchemaError, match="newer than code"):
            MigrationRunner(store.connection).verify()


def test_unmanaged_application_schema_fails_closed(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "memory.sqlite3", isolation_level=None)
    try:
        connection.execute("CREATE TABLE foreign_table(value TEXT)")
        with pytest.raises(MigrationStateError, match="no schema_migrations"):
            MigrationRunner(connection).verify()
    finally:
        connection.close()


def test_discovery_rejects_non_contiguous_invalid_or_incomplete_sql(tmp_path: Path) -> None:
    gap_dir = tmp_path / "gap"
    gap_dir.mkdir()
    (gap_dir / "0002_gap.sql").write_text("CREATE TABLE gap(value TEXT);", encoding="utf-8")
    with pytest.raises(MigrationDiscoveryError, match="contiguous"):
        discover_migrations(gap_dir)

    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "bad.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(MigrationDiscoveryError, match="filename"):
        discover_migrations(invalid_dir)

    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    (incomplete_dir / "0001_bad.sql").write_text("CREATE TABLE bad(", encoding="utf-8")
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        with pytest.raises(MigrationDiscoveryError, match="incomplete"):
            MigrationRunner(connection, incomplete_dir).apply_all()
    finally:
        connection.close()


def test_schema_fingerprint_detects_unrecorded_schema_drift(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        before = schema_fingerprint(store.connection)
        store.connection.execute("CREATE TABLE unrecorded(value TEXT)")
        assert schema_fingerprint(store.connection) != before
        with pytest.raises(MigrationStateError, match="differs"):
            store.verify()


def test_foreign_keys_and_integrity_verification_fail_closed(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            store.connection.execute(
                "INSERT INTO collections(collection_id, library_id, name, kind, created_at) "
                "VALUES ('collection_orphan', 'library_missing', 'orphan', 'corpus', ?)",
                (NOW,),
            )
        store.connection.execute("PRAGMA foreign_keys = OFF")
        store.connection.execute(
            "INSERT INTO collections(collection_id, library_id, name, kind, created_at) "
            "VALUES ('collection_orphan', 'library_missing', 'orphan', 'corpus', ?)",
            (NOW,),
        )
        store.connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(IntegrityCheckError, match="foreign_key_check"):
            store.verify()


def test_append_only_triggers_cover_every_frozen_v1_table(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        trigger_names = {
            str(row[0])
            for row in store.connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'trigger'"
            )
        }
        for table in FROZEN_TABLES:
            assert f"{table}_no_update" in trigger_names
            assert f"{table}_no_delete" in trigger_names


def test_recall_run_artifact_link_rejects_cross_query_artifacts_and_has_indexes(
    tmp_path: Path,
) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        _insert_complete_chain(connection)
        index_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'index' "
                "AND tbl_name = 'recall_run_artifacts'"
            )
        }
        assert {
            "recall_run_artifacts_coverage_idx",
            "recall_run_artifacts_read_idx",
            "recall_run_artifacts_access_idx",
            "recall_run_artifacts_retrieval_idx",
            "recall_run_artifacts_packet_idx",
        }.issubset(index_names)
        connection.execute(
            "INSERT INTO query_requests VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("query_other", "library_test", "snapshot_test", "policy_test", "{}", HASH_B, NOW),
        )
        connection.execute(
            "INSERT INTO processing_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "run_other",
                "recall",
                "query_other",
                "succeeded",
                "test",
                "fts_v1",
                NOW,
                NOW,
                None,
                HASH_B,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="inconsistent"):
            connection.execute(
                "INSERT INTO recall_run_artifacts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "run_other",
                    "coverage_test",
                    "read_test",
                    "access_test",
                    "retrieval_test",
                    "packet_test",
                ),
            )


def test_store_transaction_commits_and_rolls_back(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        with store.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO libraries VALUES ('library_committed', 'Committed', ?, ?)",
                (HASH_A, NOW),
            )
        with pytest.raises(RuntimeError, match="abort"), store.transaction() as connection:
            connection.execute(
                "INSERT INTO collections VALUES "
                "('collection_rollback', 'library_committed', 'Rollback', 'corpus', ?)",
                (NOW,),
            )
            raise RuntimeError("abort")
        assert store.connection.execute("SELECT count(*) FROM libraries").fetchone()[0] == 1
        assert store.connection.execute("SELECT count(*) FROM collections").fetchone()[0] == 0


def _insert_complete_chain(connection: sqlite3.Connection) -> None:
    statements: tuple[tuple[str, tuple[object, ...]], ...] = (
        ("INSERT INTO libraries VALUES (?, ?, ?, ?)", ("library_test", "Test", HASH_A, NOW)),
        (
            "INSERT INTO collections VALUES (?, ?, ?, ?, ?)",
            ("collection_test", "library_test", "Corpus", "corpus", NOW),
        ),
        (
            "INSERT INTO access_policies VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("policy_test", "library_test", "Policy", "[]", 0, 0, HASH_A, NOW),
        ),
        (
            "INSERT INTO blobs VALUES (?, ?, ?, ?)",
            (HASH_A, 4, "aa/blob", NOW),
        ),
        (
            "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
            ("source_test", "library_test", "file:///test.md", "text/markdown", "Test", NOW),
        ),
        (
            "INSERT INTO source_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "source_version_test",
                "source_test",
                1,
                HASH_A,
                4,
                NOW,
                None,
                "markdown_v1",
                "processed",
                None,
            ),
        ),
        (
            "INSERT INTO source_fragments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("fragment_test", "source_version_test", 0, "paragraph", "Text", HASH_A, "{}", HASH_A),
        ),
        (
            "INSERT INTO collection_memberships VALUES (?, ?, ?, ?)",
            ("collection_test", "source_test", "active", NOW),
        ),
        (
            "INSERT INTO corpus_snapshots VALUES (?, ?, ?, ?, ?)",
            ("snapshot_test", "library_test", "{}", HASH_A, NOW),
        ),
        (
            "INSERT INTO snapshot_members VALUES (?, ?, ?, ?)",
            ("snapshot_test", "collection_test", "source_version_test", "active"),
        ),
        (
            "INSERT INTO query_requests VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("query_test", "library_test", "snapshot_test", "policy_test", "{}", HASH_A, NOW),
        ),
        (
            "INSERT INTO processing_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "run_test",
                "recall",
                "query_test",
                "succeeded",
                "test",
                "fts_v1",
                NOW,
                NOW,
                None,
                HASH_A,
            ),
        ),
        (
            "INSERT INTO coverage_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("coverage_test", "run_test", "recall", 1, 0, 0, 1, "{}", HASH_A),
        ),
        (
            "INSERT INTO read_receipts VALUES (?, ?, ?)",
            ("read_test", "run_test", HASH_A),
        ),
        (
            "INSERT INTO read_receipt_items VALUES (?, ?, ?, ?)",
            ("read_test", "fragment_test", 0, HASH_A),
        ),
        (
            "INSERT INTO access_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "access_test",
                "run_test",
                "policy_test",
                HASH_A,
                HASH_A,
                HASH_A,
                HASH_A,
                HASH_A,
                1,
                "{}",
                HASH_A,
            ),
        ),
        (
            "INSERT INTO retrieval_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("retrieval_test", "run_test", "fts_v1", "1", 1, 1, "{}", HASH_A),
        ),
        (
            "INSERT INTO evidence_packets VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("packet_test", "query_test", "snapshot_test", "evidence_found", "{}", HASH_A, NOW),
        ),
        (
            "INSERT INTO packet_items VALUES (?, ?, ?, ?, ?)",
            ("packet_test", "fragment_test", "evidence", 1, "-1.0"),
        ),
        (
            "INSERT INTO recall_run_artifacts VALUES (?, ?, ?, ?, ?, ?)",
            (
                "run_test",
                "coverage_test",
                "read_test",
                "access_test",
                "retrieval_test",
                "packet_test",
            ),
        ),
    )
    for sql, parameters in statements:
        connection.execute(sql, parameters)
