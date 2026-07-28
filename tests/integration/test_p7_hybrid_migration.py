"""P7 hybrid-recall migration integrity and lineage acceptance tests."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from dithyramba.store.errors import MigrationApplyError
from dithyramba.store.migrations import MigrationRunner

NOW = "2026-07-21T12:00:00.000000Z"
P7_TABLES = {
    "corpus_layer_items",
    "corpus_layer_profiles",
    "corpus_vector_generations",
    "corpus_vector_items",
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
    "model_runtime_profiles",
    "reranker_profiles",
}


def _hash(number: int) -> str:
    return f"{number:064x}"


def _open_head(tmp_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(tmp_path / "memory.sqlite3", isolation_level=None)
    connection.execute("PRAGMA foreign_keys = ON")
    repository_root = Path(__file__).resolve().parents[2]
    MigrationRunner(connection, repository_root / "migrations").apply_all()
    return connection


def _seed_vs0(connection: sqlite3.Connection) -> None:
    connection.execute("INSERT INTO libraries VALUES ('library_one', 'One', ?, ?)", (_hash(1), NOW))
    connection.execute(
        "INSERT INTO collections VALUES ('collection_one', 'library_one', 'Corpus', 'corpus', ?)",
        (NOW,),
    )
    for suffix, policy_hash in (("one", _hash(2)), ("two", _hash(3))):
        connection.execute(
            "INSERT INTO access_policies VALUES (?, 'library_one', ?, '[\"research\"]', "
            "0, 0, ?, ?)",
            (f"policy_{suffix}", f"Policy {suffix}", policy_hash, NOW),
        )
    connection.execute("INSERT INTO blobs VALUES (?, 5, '00/source', ?)", (_hash(4), NOW))
    connection.execute(
        "INSERT INTO sources VALUES "
        "('source_one', 'library_one', 'file:///source.txt', 'text/plain', 'Source', ?)",
        (NOW,),
    )
    connection.execute(
        "INSERT INTO source_families VALUES ('family_one', 'library_one', 'Source family', ?)",
        (NOW,),
    )
    connection.execute(
        "INSERT INTO source_family_members VALUES ('family_one', 'source_one', 'root', NULL, ?)",
        (NOW,),
    )
    connection.execute(
        "INSERT INTO source_versions VALUES "
        "('source_version_one', 'source_one', 1, ?, 5, ?, ?, 'index/1.0', "
        "'processed', NULL)",
        (_hash(4), NOW, NOW),
    )
    connection.execute(
        "INSERT INTO source_fragments VALUES "
        "('fragment_one', 'source_version_one', 0, 'paragraph', 'hello', ?, '{}', ?)",
        (_hash(5), _hash(6)),
    )
    connection.execute(
        "INSERT INTO collection_memberships VALUES ('collection_one', 'source_one', 'active', ?)",
        (NOW,),
    )
    connection.execute(
        "INSERT INTO corpus_snapshots VALUES ('snapshot_one', 'library_one', '{}', ?, ?)",
        (_hash(7), NOW),
    )
    connection.execute(
        "INSERT INTO snapshot_members VALUES "
        "('snapshot_one', 'collection_one', 'source_version_one', 'active')"
    )


def _seed_profiles(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        INSERT INTO embedding_model_profiles VALUES (
            'embedding_profile_one', 'sentence_transformers', 'owner/model',
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'MIT', 1, 512,
            'query: ', 'passage: ', 1, 0, ?, ?
        )
        """,
        (_hash(10), NOW),
    )
    connection.execute(
        """
        INSERT INTO model_runtime_profiles VALUES (
            'runtime_profile_one', '5.0.0', '5.0.0', '2.7.0', 'cpu',
            'float32', 16, 1, 0, ?, ?
        )
        """,
        (_hash(11), NOW),
    )
    connection.execute(
        "INSERT INTO fusion_profiles VALUES (?, 'rrf_v1', 60, 1, 1, 2, ?)",
        (_hash(12), NOW),
    )
    connection.execute(
        """
        INSERT INTO reranker_profiles VALUES (
            'reranker_profile_one', 'sentence_transformers_cross_encoder',
            'owner/reranker', 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            'MIT', 512, 1, 0, ?, ?
        )
        """,
        (_hash(13), NOW),
    )
    connection.execute(
        """
        INSERT INTO corpus_layer_profiles VALUES (
            'layer_profile_one', 'library_one', 'snapshot_one', ?, ?, 1, ?, ?
        )
        """,
        (_hash(14), _hash(15), _hash(16), NOW),
    )
    connection.execute(
        "INSERT INTO corpus_layer_items VALUES ('layer_profile_one', 'source_one', 'primary')"
    )


def _insert_generation(
    connection: sqlite3.Connection,
    *,
    generation_id: str = "vector_generation_one",
    policy_id: str = "policy_one",
    policy_hash: str | None = None,
    exclusion_hash: str | None = None,
    permitted_set_hash: str | None = None,
    generation_hash: str | None = None,
    item_count: int = 1,
) -> None:
    policy_hash = _hash(2) if policy_hash is None else policy_hash
    exclusion_hash = _hash(14) if exclusion_hash is None else exclusion_hash
    permitted_set_hash = _hash(15) if permitted_set_hash is None else permitted_set_hash
    generation_hash = _hash(17) if generation_hash is None else generation_hash
    connection.execute(
        """
        INSERT INTO corpus_vector_generations VALUES (
            ?, 'library_one', 'snapshot_one', ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?
        )
        """,
        (
            generation_id,
            _hash(7),
            policy_id,
            policy_hash,
            exclusion_hash,
            permitted_set_hash,
            _hash(10),
            _hash(11),
            item_count,
            generation_hash,
            NOW,
        ),
    )


def _seed_complete_hybrid_run(connection: sqlite3.Connection) -> None:
    _seed_vs0(connection)
    _seed_profiles(connection)
    _insert_generation(connection)
    connection.execute(
        "INSERT INTO corpus_vector_items VALUES ('vector_generation_one', 'fragment_one', ?, ?, ?)",
        (_hash(5), _hash(18), sqlite3.Binary(b"\x00\x00\x80\x3f")),
    )
    connection.execute(
        """
        INSERT INTO hybrid_query_requests VALUES (
            'hybrid_query_one', 'library_one', 'snapshot_one', 'policy_one',
            'What does the source establish?', 'research', ?, ?, ?, ?, NULL,
            ?, 1, 0, 100, 100, 100, 30, 'hybrid_evidence_packet', ?, ?
        )
        """,
        (_hash(10), _hash(11), _hash(16), _hash(12), _hash(14), _hash(19), NOW),
    )
    connection.execute(
        "INSERT INTO hybrid_query_collections VALUES ('hybrid_query_one', 'collection_one')"
    )
    connection.execute(
        """
        INSERT INTO hybrid_processing_runs VALUES (
            'run_hybrid_one', 'hybrid_query_one', 'vector_generation_one',
            'recall', 'succeeded', 'p7', 'hybrid/1.0', ?, ?, NULL, ?
        )
        """,
        (NOW, NOW, _hash(30)),
    )
    connection.execute(
        "INSERT INTO hybrid_coverage_reports VALUES "
        "('hybrid_coverage_one', 'run_hybrid_one', 'hybrid_query_one', 1, 0, 0, 0, ?)",
        (_hash(20),),
    )
    connection.execute(
        "INSERT INTO hybrid_read_receipts VALUES "
        "('hybrid_read_one', 'run_hybrid_one', 'hybrid_query_one', 1, ?)",
        (_hash(21),),
    )
    connection.execute(
        "INSERT INTO hybrid_read_receipt_items VALUES ('hybrid_read_one', 'fragment_one', 0, ?)",
        (_hash(5),),
    )
    connection.execute(
        """
        INSERT INTO hybrid_access_receipts VALUES (
            'hybrid_access_one', 'run_hybrid_one', 'hybrid_query_one', 'policy_one',
            ?, ?, ?, ?, ?, 0, ?
        )
        """,
        (_hash(2), _hash(7), _hash(14), _hash(15), _hash(22), _hash(23)),
    )
    connection.execute(
        """
        INSERT INTO hybrid_retrieval_receipts VALUES (
            'hybrid_retrieval_one', 'run_hybrid_one', 'hybrid_query_one',
            ?, ?, ?, ?, ?, ?, ?, ?, 'fts_v1.test', ?, ?, NULL, ?
        )
        """,
        (
            _hash(19),
            _hash(14),
            _hash(15),
            _hash(10),
            _hash(11),
            _hash(16),
            _hash(12),
            _hash(17),
            _hash(25),
            _hash(26),
            _hash(24),
        ),
    )
    connection.execute(
        "INSERT INTO hybrid_retrieval_trace_items VALUES "
        "('hybrid_retrieval_one', 'fts', 'fragment_one', 'source_one', 1, '1.0')"
    )
    connection.execute(
        "INSERT INTO hybrid_retrieval_trace_items VALUES "
        "('hybrid_retrieval_one', 'fused', 'fragment_one', 'source_one', 1, '1/61')"
    )
    connection.execute(
        """
        INSERT INTO hybrid_evidence_packets VALUES (
            'hybrid_packet_one', 'hybrid_query_one', 'snapshot_one', 'policy_one',
            ?, ?, ?, ?, 'evidence_found', 1, ?, ?
        )
        """,
        (_hash(21), _hash(23), _hash(20), _hash(24), _hash(30), NOW),
    )
    connection.execute(
        "INSERT INTO hybrid_packet_items VALUES "
        "('hybrid_packet_one', 1, 'fragment_one', 'source_one', 'family_one', 'primary')"
    )
    connection.execute(
        """
        INSERT INTO hybrid_run_observations VALUES (
            'hybrid_observation_one', 'run_hybrid_one', 'hybrid_query_one',
            1, 2, 3, 4, 0, 1024, 2048, 0, ?
        )
        """,
        (NOW,),
    )
    connection.execute(
        """
        INSERT INTO hybrid_run_artifacts VALUES (
            'run_hybrid_one', 'hybrid_coverage_one', 'hybrid_read_one',
            'hybrid_access_one', 'hybrid_retrieval_one', 'hybrid_packet_one'
        )
        """
    )


def test_root_and_packaged_p7_migrations_are_byte_identical() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    root = repository_root / "migrations" / "0005_hybrid_recall.sql"
    packaged = repository_root / "src" / "dithyramba" / "store" / "sql" / root.name

    assert root.read_bytes() == packaged.read_bytes()


def test_p7_migration_is_atomic_and_does_not_alter_vs0_tables(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    v4 = tmp_path / "v4"
    v4.mkdir()
    for name in (
        "0001_core.sql",
        "0002_source_heads.sql",
        "0003_recall_run_artifacts.sql",
        "0004_meaning_core.sql",
    ):
        shutil.copyfile(repository_root / "migrations" / name, v4 / name)
    database = tmp_path / "migration.sqlite3"
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        MigrationRunner(connection, v4).apply_all()
        frozen_before = {
            table: tuple(connection.execute(f"PRAGMA table_info({table})"))
            for table in ("processing_runs", "query_requests")
        }

        broken = tmp_path / "broken"
        shutil.copytree(v4, broken)
        payload = (repository_root / "migrations" / "0005_hybrid_recall.sql").read_text(
            encoding="utf-8"
        )
        (broken / "0005_hybrid_recall.sql").write_text(
            payload + "\nCREATE TABLE p7_atomic_probe (id INTEGER);\nINVALID SQL;\n",
            encoding="utf-8",
        )
        with pytest.raises(MigrationApplyError, match=r"0005_hybrid_recall\.sql"):
            MigrationRunner(connection, broken).apply_all(
                backup_hook=lambda _connection, _migration: None
            )

        assert connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 4
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name = 'p7_atomic_probe'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name = 'embedding_model_profiles'"
            ).fetchone()[0]
            == 0
        )

        head = tmp_path / "head"
        shutil.copytree(v4, head)
        shutil.copyfile(
            repository_root / "migrations" / "0005_hybrid_recall.sql",
            head / "0005_hybrid_recall.sql",
        )
        MigrationRunner(connection, head).apply_all(
            backup_hook=lambda _connection, _migration: None
        )
        assert {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND "
                "(name LIKE 'hybrid_%' OR name IN ('embedding_model_profiles', "
                "'model_runtime_profiles', 'fusion_profiles', 'reranker_profiles', "
                "'corpus_layer_profiles', 'corpus_layer_items', "
                "'corpus_vector_generations', 'corpus_vector_items'))"
            )
        } == P7_TABLES
        for table, columns in frozen_before.items():
            assert tuple(connection.execute(f"PRAGMA table_info({table})")) == columns
    finally:
        connection.close()


def test_p7_exact_lineage_closes_and_every_table_is_append_only(tmp_path: Path) -> None:
    connection = _open_head(tmp_path)
    try:
        _seed_complete_hybrid_run(connection)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT COUNT(*) FROM hybrid_run_artifacts").fetchone()[0] == 1

        append_only = {
            str(row[0])
            for row in connection.execute(
                "SELECT tbl_name FROM sqlite_schema WHERE type = 'trigger' "
                "AND name LIKE '%_no_update' AND tbl_name IN "
                f"({','.join('?' for _ in P7_TABLES)})",
                tuple(sorted(P7_TABLES)),
            )
        }
        delete_protected = {
            str(row[0])
            for row in connection.execute(
                "SELECT tbl_name FROM sqlite_schema WHERE type = 'trigger' "
                "AND name LIKE '%_no_delete' AND tbl_name IN "
                f"({','.join('?' for _ in P7_TABLES)})",
                tuple(sorted(P7_TABLES)),
            )
        }
        assert append_only == P7_TABLES
        assert delete_protected == P7_TABLES
        for table in (
            "embedding_model_profiles",
            "corpus_vector_items",
            "hybrid_retrieval_trace_items",
            "hybrid_packet_items",
            "hybrid_run_artifacts",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f"UPDATE {table} SET rowid = rowid")
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f"DELETE FROM {table}")
    finally:
        connection.close()


def test_vectors_and_packets_are_text_free_and_scope_tuples_coexist(tmp_path: Path) -> None:
    connection = _open_head(tmp_path)
    try:
        _seed_vs0(connection)
        _seed_profiles(connection)
        _insert_generation(connection)
        connection.execute(
            "INSERT INTO corpus_vector_items VALUES "
            "('vector_generation_one', 'fragment_one', ?, ?, ?)",
            (_hash(5), _hash(18), sqlite3.Binary(b"\x00\x00\x80\x3f")),
        )
        _insert_generation(
            connection,
            generation_id="vector_generation_empty",
            policy_id="policy_two",
            policy_hash=_hash(3),
            exclusion_hash=_hash(31),
            permitted_set_hash=_hash(32),
            generation_hash=_hash(33),
            item_count=0,
        )

        assert (
            connection.execute("SELECT COUNT(*) FROM corpus_vector_generations").fetchone()[0] == 2
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM corpus_vector_items "
                "WHERE corpus_vector_generation_id = 'vector_generation_empty'"
            ).fetchone()[0]
            == 0
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            _insert_generation(
                connection,
                generation_id="vector_generation_duplicate_scope",
                generation_hash=_hash(34),
            )
        with pytest.raises(sqlite3.IntegrityError, match="dimensions"):
            connection.execute(
                "INSERT INTO corpus_vector_items VALUES "
                "('vector_generation_empty', 'fragment_one', ?, ?, ?)",
                (_hash(5), _hash(35), sqlite3.Binary(b"not-a-vector")),
            )

        vector_columns = {
            str(row[1]): str(row[2]).upper()
            for row in connection.execute("PRAGMA table_info(corpus_vector_items)")
        }
        packet_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(hybrid_evidence_packets)")
        }
        packet_item_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(hybrid_packet_items)")
        }
        assert vector_columns["vector_blob"] == "BLOB"
        assert "text" not in vector_columns
        assert set(vector_columns) == {
            "corpus_vector_generation_id",
            "source_fragment_id",
            "text_sha256",
            "vector_sha256",
            "vector_blob",
        }
        forbidden = {"text", "content", "quote", "packet_json", "score_text"}
        assert packet_columns.isdisjoint(forbidden)
        assert packet_item_columns.isdisjoint(forbidden)
    finally:
        connection.close()


def test_p7_rejects_cross_scope_and_incomplete_lineage(tmp_path: Path) -> None:
    connection = _open_head(tmp_path)
    try:
        _seed_vs0(connection)
        _seed_profiles(connection)
        with pytest.raises(sqlite3.IntegrityError, match="scope or profile"):
            _insert_generation(connection, policy_hash=_hash(999))

        _insert_generation(connection)
        connection.execute(
            "INSERT INTO corpus_vector_items VALUES "
            "('vector_generation_one', 'fragment_one', ?, ?, ?)",
            (_hash(5), _hash(18), sqlite3.Binary(b"\x00\x00\x80\x3f")),
        )
        connection.execute(
            """
            INSERT INTO hybrid_query_requests VALUES (
                'hybrid_query_one', 'library_one', 'snapshot_one', 'policy_one',
                'Question?', 'research', ?, ?, ?, ?, NULL, ?, 1, 0,
                100, 100, 100, 30, 'hybrid_evidence_packet', ?, ?
            )
            """,
            (_hash(10), _hash(11), _hash(16), _hash(12), _hash(14), _hash(19), NOW),
        )
        with pytest.raises(sqlite3.IntegrityError, match="child counts"):
            connection.execute(
                """
                INSERT INTO hybrid_processing_runs VALUES (
                    'run_incomplete', 'hybrid_query_one', 'vector_generation_one',
                    'recall', 'succeeded', 'p7', 'hybrid/1.0', ?, ?, NULL, ?
                )
                """,
                (NOW, NOW, _hash(1000)),
            )
    finally:
        connection.close()
