from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from dithyramba.store import (
    IntegrityCheckError,
    Migration,
    MigrationBackupRequiredError,
    MigrationChecksumError,
    MigrationDiscoveryError,
    MigrationRunner,
    MigrationStateError,
    NewerSchemaError,
    Store,
    discover_migrations,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
MODEL_HASH = "c" * 64
RUNTIME_HASH = "d" * 64
PROVISIONING_HASH = "e" * 64
SPAN_PROFILE_HASH = "f" * 64
SPAN_PLAN_HASH = "1" * 64
PARENT_MANIFEST_HASH = "2" * 64
SPAN_CLOSURE_HASH = "3" * 64
SPAN_HASH = "4" * 64
AUDIT_HASH = "5" * 64
AUDIT_BINDING_HASH = "6" * 64
AUDIT_MANIFEST_HASH = "7" * 64
VECTOR_GENERATION_HASH = "8" * 64
VECTOR_HASH = "9" * 64
SPAN_MANIFEST_HASH = "ab" * 32
NOW = "2026-07-20T12:00:00.000000Z"

EXPECTED_TABLES = {
    "access_policies",
    "access_policy_collection_rules",
    "access_policy_source_rules",
    "access_receipts",
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
    "corpus_layer_items",
    "corpus_layer_profiles",
    "corpus_read_set_items",
    "corpus_read_sets",
    "corpus_snapshots",
    "corpus_vector_generations",
    "corpus_vector_items",
    "coverage_reports",
    "entities",
    "entity_aliases",
    "entity_mentions",
    "event_outbox",
    "evidence_links",
    "evidence_packets",
    "embedding_model_profiles",
    "fusion_profiles",
    "hybrid_access_receipts",
    "hybrid_coverage_reports",
    "hybrid_evidence_packets",
    "hybrid_packet_items",
    "hybrid_processing_runs",
    "hybrid_query_collections",
    "hybrid_query_exclusions",
    "hybrid_query_requests",
    "hybrid_read_receipt_items",
    "hybrid_read_receipts",
    "hybrid_retrieval_receipts",
    "hybrid_retrieval_trace_items",
    "hybrid_run_artifacts",
    "hybrid_run_observations",
    "libraries",
    "meaning_coverage_reports",
    "meaning_processing_runs",
    "meaning_read_receipt_items",
    "meaning_read_receipts",
    "meaning_review_decisions",
    "meaning_review_sessions",
    "model_runtime_profiles",
    "omissions",
    "packet_items",
    "processing_runs",
    "query_requests",
    "recall_run_artifacts",
    "read_receipt_items",
    "read_receipt_corpus_sets",
    "read_receipts",
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
    "retrieval_receipts",
    "review_decisions",
    "reranker_profiles",
    "schema_migrations",
    "semantic_span_parents",
    "semantic_span_plans",
    "semantic_span_profiles",
    "semantic_span_vector_generations",
    "semantic_span_vector_items",
    "semantic_spans",
    "snapshot_members",
    "source_families",
    "source_family_identities",
    "source_family_members",
    "source_identity_bindings",
    "source_fragment_text_metrics",
    "source_fragments",
    "source_heads",
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

P9_APPEND_ONLY_TABLES = {
    "collection_root_identity_manifests",
    "source_family_identities",
    "source_identity_bindings",
}

P10_APPEND_ONLY_TABLES = {
    "corpus_read_set_items",
    "corpus_read_sets",
    "read_receipt_corpus_sets",
}

APPEND_ONLY_TABLES = {
    "access_policies",
    "access_policy_collection_rules",
    "access_policy_source_rules",
    "schema_migrations",
    "libraries",
    "blobs",
    "sources",
    "source_families",
    "source_family_members",
    "source_versions",
    "source_fragments",
    "corpus_snapshots",
    "snapshot_members",
    "query_requests",
    "recall_run_artifacts",
    "coverage_reports",
    "omissions",
    "read_receipts",
    "read_receipt_items",
    "access_receipts",
    "retrieval_receipts",
    "evidence_packets",
    "packet_items",
    "review_decisions",
}

P6_APPEND_ONLY_TABLES = {
    "concept_aliases",
    "concept_meaning_statements",
    "concept_meanings",
    "concept_mentions",
    "concepts",
    "entities",
    "entity_aliases",
    "entity_mentions",
    "evidence_links",
    "meaning_coverage_reports",
    "meaning_processing_runs",
    "meaning_read_receipt_items",
    "meaning_read_receipts",
    "meaning_review_decisions",
    "meaning_review_sessions",
    "statements",
    "time_contexts",
    "voices",
}

P7_STRUCTURE_RELATION_APPEND_ONLY_TABLES = {
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
    "structure_unit_generations",
    "structure_unit_members",
    "structure_unit_read_receipt_items",
    "structure_unit_read_receipts",
    "structure_units",
}

P7_SEMANTIC_SPAN_APPEND_ONLY_TABLES = {
    "semantic_span_parents",
    "semantic_span_plans",
    "semantic_span_profiles",
    "semantic_span_vector_generations",
    "semantic_span_vector_items",
    "semantic_spans",
    "source_fragment_text_metrics",
}

P7_SEMANTIC_SPAN_INSERT_TRIGGERS = {
    "semantic_span_parents_validate_insert",
    "semantic_span_profiles_validate_insert",
    "semantic_span_vector_generations_validate_insert",
    "semantic_span_vector_items_validate_insert",
    "semantic_spans_validate_insert",
    "source_fragments_capture_text_metrics",
}

P7_SEMANTIC_SPAN_INDEXES = {
    "semantic_span_parents_fragment_idx",
    "semantic_span_parents_source_ordinal_idx",
    "semantic_span_plans_profile_idx",
    "semantic_span_profiles_model_runtime_idx",
    "semantic_span_vector_generations_model_runtime_idx",
    "semantic_span_vector_generations_plan_idx",
    "semantic_span_vector_generations_scope_idx",
    "semantic_span_vector_items_parent_group_idx",
    "semantic_span_vector_items_span_reverse_idx",
    "semantic_span_vector_items_vector_hash_idx",
    "semantic_spans_parent_group_idx",
    "semantic_spans_profile_model_idx",
    "source_fragment_text_metrics_hash_idx",
}


def test_initial_migration_is_exact_and_second_apply_is_noop(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    with Store.open(database) as store:
        runner = MigrationRunner(store.connection)
        tables = {
            str(row[0])
            for row in store.connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }

        assert tables == EXPECTED_TABLES
        assert runner.current_version() == 10
        assert runner.apply_all() == ()
        assert store.schema_version == 10


def test_store_applies_required_sqlite_profile(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection

        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
        assert connection.execute("PRAGMA trusted_schema").fetchone()[0] == 0


def test_root_and_packaged_migration_are_identical() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    for name in (
        "0001_core.sql",
        "0002_source_heads.sql",
        "0003_recall_run_artifacts.sql",
        "0004_meaning_core.sql",
        "0005_hybrid_recall.sql",
        "0006_structure_relations.sql",
        "0007_semantic_span_vectors.sql",
        "0008_relation_admission.sql",
        "0009_source_identity_manifest.sql",
        "0010_corpus_read_sets.sql",
    ):
        root_sql = (repository_root / "migrations" / name).read_bytes()
        packaged_sql = (
            repository_root / "src" / "dithyramba" / "store" / "sql" / name
        ).read_bytes()
        assert root_sql == packaged_sql


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
        connection.execute(
            "UPDATE source_heads SET source_version_id = 'source_version_one' "
            "WHERE source_id = 'source_one'"
        )
        with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
            connection.execute("DELETE FROM source_heads WHERE source_id = 'source_one'")


def test_migration_two_backfills_latest_source_head(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    first_only = tmp_path / "first-only"
    first_only.mkdir()
    (first_only / "0001_core.sql").write_bytes(
        (repository_root / "migrations" / "0001_core.sql").read_bytes()
    )
    connection = sqlite3.connect(tmp_path / "upgrade.sqlite3", isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        MigrationRunner(connection, first_only).apply_all()
        connection.execute(
            "INSERT INTO libraries VALUES ('library_one', 'One', ?, ?)", (HASH_A, NOW)
        )
        connection.execute("INSERT INTO blobs VALUES (?, 1, 'aa/one', ?)", (HASH_A, NOW))
        connection.execute("INSERT INTO blobs VALUES (?, 1, 'bb/two', ?)", (HASH_B, NOW))
        connection.execute(
            "INSERT INTO sources VALUES "
            "('source_one', 'library_one', 'file:///one.txt', 'text/plain', 'One', ?)",
            (NOW,),
        )
        for number, digest in ((1, HASH_A), (2, HASH_B)):
            connection.execute(
                "INSERT INTO source_versions VALUES (?, 'source_one', ?, ?, 1, ?, ?, "
                "'index/1.0', 'processed', NULL)",
                (f"source_version_{number}", number, digest, NOW, NOW),
            )
        backed_up: list[int] = []
        runner = MigrationRunner(connection, repository_root / "migrations")
        applied = runner.apply_all(
            backup_hook=lambda _connection, migration: backed_up.append(migration.version)
        )

        assert [migration.version for migration in applied] == [2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert backed_up == [2, 3, 4, 5, 6, 7, 8, 9, 10]
        row = connection.execute("SELECT source_id, source_version_id FROM source_heads").fetchone()
        assert tuple(row) == ("source_one", "source_version_2")
    finally:
        connection.close()


def test_mutated_applied_migration_fails_closed(tmp_path: Path) -> None:
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    canonical_sql = Path(__file__).resolve().parents[2] / "migrations" / "0001_core.sql"
    migration_file = migration_dir / "0001_core.sql"
    migration_file.write_bytes(canonical_sql.read_bytes())

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


def test_audited_pre_release_migration_three_checksum_remains_compatible(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    pre_release = tmp_path / "pre-release-migrations"
    pre_release.mkdir()
    for name in (
        "0001_core.sql",
        "0002_source_heads.sql",
        "0003_recall_run_artifacts.sql",
    ):
        payload = (repository_root / "migrations" / name).read_bytes()
        if name == "0003_recall_run_artifacts.sql":
            payload = payload.replace(
                b"END;\nCREATE TRIGGER recall_run_artifacts_no_delete",
                b"END;\n\nCREATE TRIGGER recall_run_artifacts_no_delete",
            )
            assert payload.endswith(b"\n")
            payload += b"\n"
            assert hashlib.sha256(payload).hexdigest() == (
                "840e47d732e849990b6c02bda71bf63ea103bd06a45c8db34427766d5d4226bb"
            )
        (pre_release / name).write_bytes(payload)

    connection = sqlite3.connect(tmp_path / "pre-release.sqlite3", isolation_level=None)
    try:
        MigrationRunner(connection, pre_release).apply_all()
        applied = MigrationRunner(connection, repository_root / "migrations").verify()

        assert [migration.version for migration in applied] == [1, 2, 3]
        assert applied[-1].sha256 == (
            "d9162150f3aeb1c1a8650f77e23f57d4529235f701fc3f65306cdd1b53ee1998"
        )
    finally:
        connection.close()


def test_v2_to_head_requires_and_accepts_verified_backups(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    first_two = tmp_path / "first-two"
    first_two.mkdir()
    for name in ("0001_core.sql", "0002_source_heads.sql"):
        (first_two / name).write_bytes((repository_root / "migrations" / name).read_bytes())
    database = tmp_path / "upgrade-v2.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        MigrationRunner(connection, first_two).apply_all()
        runner = MigrationRunner(connection, repository_root / "migrations")
        with pytest.raises(MigrationBackupRequiredError, match="verified backup"):
            runner.apply_all()

        backed_up: list[int] = []

        def verified_backup(source: sqlite3.Connection, migration: Migration) -> None:
            backup_path = tmp_path / f"verified-v{migration.version - 1}.sqlite3"
            destination = sqlite3.connect(backup_path, isolation_level=None)
            try:
                source.backup(destination)
                assert destination.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert (
                    MigrationRunner(destination, repository_root / "migrations").current_version()
                    == migration.version - 1
                )
            finally:
                destination.close()
            backed_up.append(migration.version)

        applied = runner.apply_all(backup_hook=verified_backup)

        assert [migration.version for migration in applied] == [3, 4, 5, 6, 7, 8, 9, 10]
        assert backed_up == [3, 4, 5, 6, 7, 8, 9, 10]
        assert runner.current_version() == 10
        assert (tmp_path / "verified-v2.sqlite3").is_file()
        assert (tmp_path / "verified-v3.sqlite3").is_file()
        assert (tmp_path / "verified-v4.sqlite3").is_file()
        assert (tmp_path / "verified-v5.sqlite3").is_file()
        assert (tmp_path / "verified-v6.sqlite3").is_file()
        assert (tmp_path / "verified-v7.sqlite3").is_file()
        assert (tmp_path / "verified-v8.sqlite3").is_file()
        assert (tmp_path / "verified-v9.sqlite3").is_file()
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_schema "
                "WHERE type = 'table' AND name = 'recall_run_artifacts'"
            ).fetchone()[0]
            == 1
        )
    finally:
        connection.close()


def test_newer_database_schema_fails_closed(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        store.connection.execute(
            """
            INSERT INTO schema_migrations(version, name, sha256, applied_at)
            VALUES (11, '0011_future.sql', ?, ?)
            """,
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


def test_discovery_rejects_non_contiguous_or_invalid_names(tmp_path: Path) -> None:
    gap_dir = tmp_path / "gap"
    gap_dir.mkdir()
    (gap_dir / "0002_gap.sql").write_text("CREATE TABLE gap(value TEXT);", encoding="utf-8")

    with pytest.raises(MigrationDiscoveryError, match="contiguous"):
        MigrationRunner(sqlite3.connect(":memory:"), gap_dir)

    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "bad.sql").write_text("SELECT 1;", encoding="utf-8")

    with pytest.raises(MigrationDiscoveryError, match="filename"):
        MigrationRunner(sqlite3.connect(":memory:"), invalid_dir)


def test_foreign_keys_and_integrity_verification_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    with Store.open(database) as store:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            store.connection.execute(
                """
                INSERT INTO collections(collection_id, library_id, name, kind, created_at)
                VALUES ('collection_orphan', 'library_missing', 'orphan', 'corpus', ?)
                """,
                (NOW,),
            )

        store.connection.execute("PRAGMA foreign_keys = OFF")
        store.connection.execute(
            """
            INSERT INTO collections(collection_id, library_id, name, kind, created_at)
            VALUES ('collection_orphan', 'library_missing', 'orphan', 'corpus', ?)
            """,
            (NOW,),
        )
        store.connection.execute("PRAGMA foreign_keys = ON")

        with pytest.raises(IntegrityCheckError, match="foreign_key_check"):
            store.verify()


def test_append_only_triggers_cover_every_frozen_immutable_table(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        _insert_complete_chain(connection)
        trigger_rows = connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'trigger'"
        ).fetchall()
        trigger_names = {str(row[0]) for row in trigger_rows}

        for table in (
            APPEND_ONLY_TABLES
            | P6_APPEND_ONLY_TABLES
            | P7_STRUCTURE_RELATION_APPEND_ONLY_TABLES
            | P7_SEMANTIC_SPAN_APPEND_ONLY_TABLES
            | P9_APPEND_ONLY_TABLES
            | P10_APPEND_ONLY_TABLES
        ):
            assert f"{table}_no_update" in trigger_names
            assert f"{table}_no_delete" in trigger_names
        for table in APPEND_ONLY_TABLES:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f'UPDATE "{table}" SET rowid = rowid')
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f'DELETE FROM "{table}"')


def test_semantic_span_v7_schema_surface_and_append_only_rows(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        _insert_complete_chain(connection)
        _insert_semantic_span_v7_chain(connection)

        trigger_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'trigger'"
            ).fetchall()
        }
        index_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'index'"
            ).fetchall()
        }
        expected_append_only_triggers = {
            f"{table}_{operation}"
            for table in P7_SEMANTIC_SPAN_APPEND_ONLY_TABLES
            for operation in ("no_update", "no_delete")
        }
        assert trigger_names >= P7_SEMANTIC_SPAN_INSERT_TRIGGERS
        assert trigger_names >= expected_append_only_triggers
        assert index_names >= P7_SEMANTIC_SPAN_INDEXES

        for table in P7_SEMANTIC_SPAN_APPEND_ONLY_TABLES:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f'UPDATE "{table}" SET rowid = rowid')
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f'DELETE FROM "{table}"')


def test_semantic_span_v7_structural_closure_fails_closed(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        _insert_complete_chain(connection)
        _insert_semantic_span_v7_inputs(connection)

        assert tuple(
            connection.execute(
                "SELECT text_sha256, text_char_count "
                "FROM source_fragment_text_metrics "
                "WHERE source_fragment_id = 'fragment_test'"
            ).fetchone()
        ) == (HASH_A, 4)

        with pytest.raises(sqlite3.IntegrityError, match="prefix, or token budget"):
            _insert_semantic_span_profile(
                connection,
                profile_hash="ac" * 32,
                passage_prefix="wrong: ",
            )
        _insert_semantic_span_profile(connection)
        _insert_semantic_span_plan(connection)

        connection.execute(
            "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
            (
                "source_wrong",
                "library_test",
                "file:///wrong.md",
                "text/markdown",
                "Wrong",
                NOW,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="source lineage"):
            _insert_semantic_span_parent(connection, source_id="source_wrong")
        with pytest.raises(sqlite3.IntegrityError, match="source lineage"):
            _insert_semantic_span_parent(connection, text_char_count=5)
        _insert_semantic_span_parent(connection)

        _insert_alternate_embedding_profile(connection)
        with pytest.raises(sqlite3.IntegrityError, match="profile, model, token budget"):
            _insert_semantic_span(connection, model_profile_hash="0" * 64)
        with pytest.raises(sqlite3.IntegrityError, match="profile, model, token budget"):
            _insert_semantic_span(connection, prepared_token_count=513)
        _insert_semantic_span(connection)

        with pytest.raises(sqlite3.IntegrityError, match="plan, or model closure"):
            _insert_semantic_span_vector_generation(
                connection,
                parent_manifest_hash="ba" * 32,
            )
        _insert_semantic_span_vector_generation(connection)

        with pytest.raises(sqlite3.IntegrityError, match="span hashes, parent"):
            _insert_semantic_span_vector_item(
                connection,
                semantic_span_hash="0" * 64,
            )
        with pytest.raises(sqlite3.IntegrityError, match="dimensions"):
            _insert_semantic_span_vector_item(connection, vector_blob=b"\x00" * 8)
        _insert_semantic_span_vector_item(connection)

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            _insert_semantic_span_vector_generation(
                connection,
                generation_hash="0" * 64,
                span_manifest_hash="ba" * 32,
            )

        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_text_metrics_trigger_supports_narrow_authorizer_without_opening_text_reads(
    tmp_path: Path,
) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        protected_read_triggers: list[str | None] = []

        def authorizer(
            action_code: int,
            table_name: str | None,
            column_name: str | None,
            database_name: str | None,
            trigger_name: str | None,
        ) -> int:
            del database_name
            if (
                action_code == sqlite3.SQLITE_READ
                and table_name == "source_fragments"
                and column_name == "text"
            ):
                protected_read_triggers.append(trigger_name)
                if trigger_name != "source_fragments_capture_text_metrics":
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        _insert_complete_chain(connection)

        assert tuple(
            connection.execute(
                "SELECT source_fragment_id, text_sha256, text_char_count "
                "FROM source_fragment_text_metrics"
            ).fetchone()
        ) == ("fragment_test", HASH_A, 4)
        with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
            connection.execute("SELECT text FROM source_fragments").fetchall()
        assert protected_read_triggers == [
            "source_fragments_capture_text_metrics",
            None,
        ]


def test_v6_to_v7_is_backup_first_and_leaves_existing_rows_untouched(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    v6_source = tmp_path / "v6-migrations"
    v6_source.mkdir()
    for migration in discover_migrations(repository_root / "migrations")[:6]:
        (v6_source / migration.name).write_bytes(
            (repository_root / "migrations" / migration.name).read_bytes()
        )

    connection = sqlite3.connect(tmp_path / "v6.sqlite3", isolation_level=None)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        v6_runner = MigrationRunner(connection, v6_source)
        v6_runner.apply_all()
        _insert_complete_chain(connection)
        protected_tables = (
            "libraries",
            "sources",
            "source_versions",
            "source_fragments",
            "corpus_snapshots",
            "query_requests",
        )
        before = {
            table: connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            for table in protected_tables
        }

        head_runner = MigrationRunner(connection, repository_root / "migrations")
        with pytest.raises(MigrationBackupRequiredError, match="verified backup"):
            head_runner.apply_all()
        backed_up: list[int] = []
        applied = head_runner.apply_all(
            backup_hook=lambda _connection, migration: backed_up.append(migration.version)
        )

        assert [migration.version for migration in applied] == [7, 8, 9, 10]
        assert backed_up == [7, 8, 9, 10]
        assert head_runner.current_version() == 10
        assert {
            table: connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            for table in protected_tables
        } == before
        for table in P7_SEMANTIC_SPAN_APPEND_ONLY_TABLES:
            expected_count = 1 if table == "source_fragment_text_metrics" else 0
            assert (
                connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                == expected_count
            )
        assert connection.execute(
            "SELECT source_fragment_id, text_sha256, text_char_count "
            "FROM source_fragment_text_metrics"
        ).fetchone() == ("fragment_test", HASH_A, 4)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def test_recall_run_artifact_link_rejects_cross_query_artifacts_and_has_indexes(
    tmp_path: Path,
) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        _insert_complete_chain(connection)
        index_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'index' AND tbl_name = 'recall_run_artifacts'"
            ).fetchall()
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
            (
                "query_other",
                "library_test",
                "snapshot_test",
                "policy_test",
                "{}",
                HASH_B,
                NOW,
            ),
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
                """
                INSERT INTO libraries(library_id, name, logical_identity_hash, created_at)
                VALUES ('library_committed', 'Committed', ?, ?)
                """,
                (HASH_A, NOW),
            )

        with pytest.raises(RuntimeError, match="abort"), store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO collections(collection_id, library_id, name, kind, created_at)
                VALUES ('collection_rollback', 'library_committed', 'Rollback', 'corpus', ?)
                """,
                (NOW,),
            )
            raise RuntimeError("abort")

        assert store.connection.execute("SELECT count(*) FROM libraries").fetchone()[0] == 1
        assert store.connection.execute("SELECT count(*) FROM collections").fetchone()[0] == 0


def test_store_transaction_rolls_back_when_commit_itself_fails(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        with pytest.raises(sqlite3.IntegrityError), store.transaction(immediate=True) as connection:
            connection.execute("PRAGMA defer_foreign_keys = ON")
            connection.execute(
                """
                INSERT INTO collections(collection_id, library_id, name, kind, created_at)
                VALUES ('collection_deferred', 'library_missing', 'Deferred', 'corpus', ?)
                """,
                (NOW,),
            )

        assert not store.connection.in_transaction
        assert store.connection.execute("SELECT count(*) FROM collections").fetchone()[0] == 0

        with store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO libraries(library_id, name, logical_identity_hash, created_at)
                VALUES ('library_after_failure', 'Recovered', ?, ?)
                """,
                (HASH_A, NOW),
            )
        assert store.connection.execute("SELECT count(*) FROM libraries").fetchone()[0] == 1


def _insert_complete_chain(connection: sqlite3.Connection) -> None:
    statements: tuple[tuple[str, tuple[object, ...]], ...] = (
        (
            "INSERT INTO libraries VALUES (?, ?, ?, ?)",
            ("library_test", "Test", HASH_A, NOW),
        ),
        (
            "INSERT INTO collections VALUES (?, ?, ?, ?, ?)",
            ("collection_test", "library_test", "Corpus", "corpus", NOW),
        ),
        (
            "INSERT INTO access_policies VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("policy_test", "library_test", "Policy", "[]", 0, 0, HASH_A, NOW),
        ),
        (
            "INSERT INTO access_policy_collection_rules VALUES (?, ?, ?)",
            ("policy_test", "collection_test", "allow"),
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
            "INSERT INTO access_policy_source_rules VALUES (?, ?, ?)",
            ("policy_test", "source_test", "deny"),
        ),
        (
            "INSERT INTO source_families VALUES (?, ?, ?, ?)",
            ("family_test", "library_test", "Family", NOW),
        ),
        (
            "INSERT INTO source_family_members VALUES (?, ?, ?, ?, ?)",
            ("family_test", "source_test", "root", None, NOW),
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
            "INSERT INTO omissions VALUES (?, ?, ?, ?, ?, ?)",
            ("omission_test", "coverage_test", "policy", "redacted", None, "policy_denied"),
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
        (
            "INSERT INTO review_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "review_test",
                "library_test",
                "evidence_packet",
                "packet_test",
                HASH_B,
                "accept",
                "valid",
                "tester",
                "{}",
                None,
                NOW,
            ),
        ),
    )
    for sql, parameters in statements:
        connection.execute(sql, parameters)


def _insert_semantic_span_v7_inputs(connection: sqlite3.Connection) -> None:
    snapshot_id = f"snapshot_{HASH_B[:32]}"
    connection.execute(
        "INSERT INTO corpus_snapshots VALUES (?, ?, ?, ?, ?)",
        (snapshot_id, "library_test", "{}", HASH_B, NOW),
    )
    connection.execute(
        "INSERT INTO snapshot_members VALUES (?, ?, ?, ?)",
        (snapshot_id, "collection_test", "source_version_test", "active"),
    )
    connection.execute(
        """
        INSERT INTO embedding_model_profiles(
            embedding_model_profile_id, provider, model_id, revision,
            model_license, dimensions, max_tokens, query_prefix,
            passage_prefix, normalize_embeddings, trust_remote_code,
            profile_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"embedding_profile_{MODEL_HASH[:32]}",
            "sentence_transformers",
            "test/semantic-e5",
            "c" * 40,
            "MIT",
            3,
            512,
            "query: ",
            "passage: ",
            1,
            0,
            MODEL_HASH,
            NOW,
        ),
    )
    connection.execute(
        """
        INSERT INTO model_runtime_profiles(
            model_runtime_profile_id, sentence_transformers_version,
            transformers_version, torch_version, device, dtype, batch_size,
            local_files_only, trust_remote_code, profile_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"runtime_profile_{RUNTIME_HASH[:32]}",
            "5.0.0",
            "5.0.0",
            "2.0.0",
            "cpu",
            "float32",
            8,
            1,
            0,
            RUNTIME_HASH,
            NOW,
        ),
    )


def _insert_alternate_embedding_profile(connection: sqlite3.Connection) -> None:
    alternate_hash = "0" * 64
    connection.execute(
        """
        INSERT INTO embedding_model_profiles(
            embedding_model_profile_id, provider, model_id, revision,
            model_license, dimensions, max_tokens, query_prefix,
            passage_prefix, normalize_embeddings, trust_remote_code,
            profile_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"embedding_profile_{alternate_hash[:32]}",
            "sentence_transformers",
            "test/alternate-e5",
            "0" * 40,
            "MIT",
            3,
            512,
            "query: ",
            "passage: ",
            1,
            0,
            alternate_hash,
            NOW,
        ),
    )


def _insert_semantic_span_profile(
    connection: sqlite3.Connection,
    *,
    profile_hash: str = SPAN_PROFILE_HASH,
    passage_prefix: str = "passage: ",
) -> None:
    connection.execute(
        """
        INSERT INTO semantic_span_profiles(
            semantic_span_profile_id, profile_version, model_profile_hash,
            runtime_profile_hash, provisioning_receipt_hash,
            max_prepared_tokens, overlap_content_tokens, passage_prefix,
            profile_hash, created_at
        ) VALUES (?, '1.0', ?, ?, ?, 512, 64, ?, ?, ?)
        """,
        (
            f"semantic_span_profile_{profile_hash[:32]}",
            MODEL_HASH,
            RUNTIME_HASH,
            PROVISIONING_HASH,
            passage_prefix,
            profile_hash,
            NOW,
        ),
    )


def _insert_semantic_span_plan(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        INSERT INTO semantic_span_plans(
            semantic_span_plan_id, plan_hash, semantic_span_profile_hash,
            parent_manifest_hash, span_closure_hash, parent_count,
            span_count, created_at
        ) VALUES (?, ?, ?, ?, ?, 1, 1, ?)
        """,
        (
            f"semantic_span_plan_{SPAN_PLAN_HASH[:32]}",
            SPAN_PLAN_HASH,
            SPAN_PROFILE_HASH,
            PARENT_MANIFEST_HASH,
            SPAN_CLOSURE_HASH,
            NOW,
        ),
    )


def _insert_semantic_span_parent(
    connection: sqlite3.Connection,
    *,
    source_id: str = "source_test",
    text_char_count: int = 4,
) -> None:
    connection.execute(
        """
        INSERT INTO semantic_span_parents(
            semantic_span_plan_id, source_fragment_id, source_version_id,
            source_id, ordinal, text_sha256, text_char_count,
            content_token_count, full_prepared_token_count,
            content_offsets_hash, span_count
        ) VALUES (?, 'fragment_test', 'source_version_test', ?, 0, ?, ?, 1, 3, ?, 1)
        """,
        (
            f"semantic_span_plan_{SPAN_PLAN_HASH[:32]}",
            source_id,
            HASH_A,
            text_char_count,
            "cd" * 32,
        ),
    )


def _insert_semantic_span(
    connection: sqlite3.Connection,
    *,
    model_profile_hash: str = MODEL_HASH,
    prepared_token_count: int = 3,
) -> None:
    connection.execute(
        """
        INSERT INTO semantic_spans(
            semantic_span_plan_id, semantic_span_id, semantic_span_hash,
            parent_fragment_id, parent_text_sha256, parent_ordinal,
            span_ordinal, char_start, char_end, content_token_start,
            content_token_end, span_text_sha256, prepared_token_count,
            profile_hash, model_profile_hash
        ) VALUES (?, ?, ?, 'fragment_test', ?, 0, 0, 0, 4, 0, 1, ?, ?, ?, ?)
        """,
        (
            f"semantic_span_plan_{SPAN_PLAN_HASH[:32]}",
            f"semantic_span_{SPAN_HASH[:32]}",
            SPAN_HASH,
            HASH_A,
            HASH_A,
            prepared_token_count,
            SPAN_PROFILE_HASH,
            model_profile_hash,
        ),
    )


def _insert_semantic_span_vector_generation(
    connection: sqlite3.Connection,
    *,
    generation_hash: str = VECTOR_GENERATION_HASH,
    parent_manifest_hash: str = PARENT_MANIFEST_HASH,
    span_manifest_hash: str = SPAN_MANIFEST_HASH,
) -> None:
    connection.execute(
        """
        INSERT INTO semantic_span_vector_generations(
            semantic_span_vector_generation_id, library_id,
            corpus_snapshot_id, snapshot_hash, access_policy_id, policy_hash,
            exclusion_hash, permitted_set_hash, semantic_audit_id,
            semantic_audit_hash, semantic_audit_binding_hash,
            semantic_audit_fragment_manifest_hash, semantic_span_plan_id,
            semantic_span_plan_hash, semantic_span_profile_hash,
            semantic_span_parent_manifest_hash, semantic_span_closure_hash,
            model_profile_hash, runtime_profile_hash,
            provisioning_receipt_hash, dimensions, normalization,
            parent_count, span_count, span_manifest_hash, generation_hash,
            created_at
        ) VALUES (
            ?, 'library_test', ?, ?, 'policy_test', ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, 3, 'l2_unit_float32_v1', 1, 1, ?, ?, ?
        )
        """,
        (
            f"semantic_span_vector_generation_{generation_hash[:32]}",
            f"snapshot_{HASH_B[:32]}",
            HASH_B,
            HASH_A,
            "de" * 32,
            "ef" * 32,
            f"semantic_audit_{AUDIT_HASH[:32]}",
            AUDIT_HASH,
            AUDIT_BINDING_HASH,
            AUDIT_MANIFEST_HASH,
            f"semantic_span_plan_{SPAN_PLAN_HASH[:32]}",
            SPAN_PLAN_HASH,
            SPAN_PROFILE_HASH,
            parent_manifest_hash,
            SPAN_CLOSURE_HASH,
            MODEL_HASH,
            RUNTIME_HASH,
            PROVISIONING_HASH,
            span_manifest_hash,
            generation_hash,
            NOW,
        ),
    )


def _insert_semantic_span_vector_item(
    connection: sqlite3.Connection,
    *,
    semantic_span_hash: str = SPAN_HASH,
    vector_blob: bytes = b"\x00" * 12,
) -> None:
    connection.execute(
        """
        INSERT INTO semantic_span_vector_items(
            semantic_span_vector_generation_id, semantic_span_plan_id,
            semantic_span_id, parent_fragment_id, span_ordinal,
            semantic_span_hash, span_text_sha256, vector_sha256, vector_blob
        ) VALUES (?, ?, ?, 'fragment_test', 0, ?, ?, ?, ?)
        """,
        (
            f"semantic_span_vector_generation_{VECTOR_GENERATION_HASH[:32]}",
            f"semantic_span_plan_{SPAN_PLAN_HASH[:32]}",
            f"semantic_span_{SPAN_HASH[:32]}",
            semantic_span_hash,
            HASH_A,
            VECTOR_HASH,
            vector_blob,
        ),
    )


def _insert_semantic_span_v7_chain(connection: sqlite3.Connection) -> None:
    _insert_semantic_span_v7_inputs(connection)
    _insert_semantic_span_profile(connection)
    _insert_semantic_span_plan(connection)
    _insert_semantic_span_parent(connection)
    _insert_semantic_span(connection)
    _insert_semantic_span_vector_generation(connection)
    _insert_semantic_span_vector_item(connection)
