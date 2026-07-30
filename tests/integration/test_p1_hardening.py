from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest

from dithyramba.library import (
    LibraryConfig,
    LibraryLayoutError,
    create_library_layout,
    validate_library_layout,
)
from dithyramba.store import (
    MigrationDiscoveryError,
    MigrationRunner,
    MigrationStateError,
    Store,
    StoreOpenError,
    verify_migrations_smoke,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
NOW = "2026-07-20T12:00:00.000000Z"


def test_packaged_migration_smoke_returns_schema_evidence() -> None:
    evidence = verify_migrations_smoke()

    assert evidence["status"] == "ok"
    assert evidence["schema_version"] == 11
    assert len(str(evidence["schema_fingerprint"])) == 64
    checksums = evidence["migration_checksums"]
    assert isinstance(checksums, dict)
    assert set(checksums) == {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
    }


def test_actual_schema_tampering_fails_even_with_valid_migration_history(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    with Store.open(database):
        pass

    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute("DROP TABLE sources")
    connection.close()

    with pytest.raises(MigrationStateError, match="actual SQLite schema differs"):
        Store.open(database, apply_migrations=False)


def test_incomplete_migration_rolls_back_every_statement_and_transaction(tmp_path: Path) -> None:
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "0001_bad.sql").write_text(
        "CREATE TABLE partial(value TEXT);\nCREATE TABLE incomplete(",
        encoding="utf-8",
    )
    connection = sqlite3.connect(":memory:", isolation_level=None)
    runner = MigrationRunner(connection, migration_dir)

    with pytest.raises(MigrationDiscoveryError, match="incomplete"):
        runner.apply_all()

    assert connection.in_transaction is False
    assert (
        connection.execute("SELECT count(*) FROM sqlite_schema WHERE name = 'partial'").fetchone()[
            0
        ]
        == 0
    )
    connection.close()


def test_empty_history_with_or_without_rogue_schema_fails_closed(tmp_path: Path) -> None:
    migration_sql = """
    CREATE TABLE schema_migrations(
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        applied_at TEXT NOT NULL
    );
    CREATE TABLE rogue(value TEXT);
    """
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.executescript(migration_sql)

    with pytest.raises(MigrationStateError, match="history is empty"):
        MigrationRunner(connection).verify()
    connection.close()

    empty_database = tmp_path / "empty.sqlite3"
    with pytest.raises(MigrationStateError, match="behind code version"):
        Store.open(empty_database, apply_migrations=False)
    assert not empty_database.exists()


def test_store_rejects_ambiguous_symlink_and_hardlink_paths(tmp_path: Path) -> None:
    with pytest.raises(StoreOpenError, match="absolute"):
        Store.open("relative.sqlite3")

    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    symlink = tmp_path / "symlink.sqlite3"
    symlink.symlink_to(target)
    with pytest.raises(StoreOpenError, match="regular file"):
        Store.open(symlink)

    hardlink = tmp_path / "hardlink.sqlite3"
    os.link(target, hardlink)
    with pytest.raises(StoreOpenError, match="hardlinked"):
        Store.open(target)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission invariant")
def test_store_and_layout_are_private_and_detect_substitution(tmp_path: Path) -> None:
    first = create_library_layout(LibraryConfig(name="First"), data_root=tmp_path / "data")
    second = create_library_layout(LibraryConfig(name="Second"), data_root=tmp_path / "data")
    assert stat.S_IMODE(first.database.stat().st_mode) == 0o600
    assert stat.S_IMODE(first.root.stat().st_mode) == 0o700

    first.events.chmod(0o644)
    with pytest.raises(LibraryLayoutError, match="unsafe mode"):
        validate_library_layout(first)
    first.events.chmod(0o600)

    displaced = first.root.with_name(first.root.name + "-displaced")
    first.root.rename(displaced)
    first.root.symlink_to(second.root, target_is_directory=True)
    with pytest.raises(LibraryLayoutError, match="real directory"):
        validate_library_layout(first)


def test_identity_policy_and_outbox_payload_are_immutable(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        connection.execute(
            "INSERT INTO libraries VALUES (?, ?, ?, ?)",
            ("library_test", "Test", HASH_A, NOW),
        )
        connection.execute(
            "INSERT INTO collections VALUES (?, ?, ?, ?, ?)",
            ("collection_test", "library_test", "Corpus", "corpus", NOW),
        )
        connection.execute(
            "INSERT INTO access_policies VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "policy_test",
                "library_test",
                "Research",
                '["research"]',
                0,
                0,
                HASH_B,
                NOW,
            ),
        )
        connection.execute(
            "INSERT INTO access_policy_collection_rules VALUES (?, ?, ?)",
            ("policy_test", "collection_test", "allow"),
        )
        connection.execute(
            "INSERT INTO event_outbox VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            ("event_test", "created", "library", "library_test", "{}", HASH_A, NOW),
        )

        for statement in (
            "UPDATE libraries SET name = 'Changed'",
            "DELETE FROM libraries",
            "UPDATE access_policies SET name = 'Changed'",
            "DELETE FROM access_policy_collection_rules",
            "UPDATE event_outbox SET payload_json = '{\"tampered\":true}'",
            "DELETE FROM event_outbox",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement)

        delivered = "2026-07-20T12:01:00.000000Z"
        connection.execute("UPDATE event_outbox SET delivered_at = ?", (delivered,))
        with pytest.raises(sqlite3.IntegrityError, match="first delivery"):
            connection.execute("UPDATE event_outbox SET delivered_at = ?", (delivered,))


def test_one_source_cannot_belong_to_multiple_lineage_families(tmp_path: Path) -> None:
    with Store.open(tmp_path / "memory.sqlite3") as store:
        connection = store.connection
        connection.execute(
            "INSERT INTO libraries VALUES (?, ?, ?, ?)",
            ("library_test", "Test", HASH_A, NOW),
        )
        connection.execute(
            "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
            ("source_test", "library_test", "file:///source.md", "text/markdown", None, NOW),
        )
        connection.execute(
            "INSERT INTO source_families VALUES (?, ?, ?, ?)",
            ("family_first", "library_test", "First", NOW),
        )
        connection.execute(
            "INSERT INTO source_families VALUES (?, ?, ?, ?)",
            ("family_second", "library_test", "Second", NOW),
        )
        connection.execute(
            "INSERT INTO source_family_members VALUES (?, ?, 'root', NULL, ?)",
            ("family_first", "source_test", NOW),
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            connection.execute(
                "INSERT INTO source_family_members VALUES (?, ?, 'root', NULL, ?)",
                ("family_second", "source_test", NOW),
            )
