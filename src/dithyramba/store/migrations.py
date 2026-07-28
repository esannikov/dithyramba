"""Discover, verify, and atomically apply immutable SQLite migrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TypeAlias

from dithyramba.contracts import canonical_sha256_hex
from dithyramba.store.errors import (
    MigrationApplyError,
    MigrationBackupRequiredError,
    MigrationChecksumError,
    MigrationDiscoveryError,
    MigrationStateError,
    NewerSchemaError,
    TransactionStateError,
)

MigrationSource: TypeAlias = str | Path | Traversable
BackupHook: TypeAlias = Callable[[sqlite3.Connection, "Migration"], None]
LockedMigrationGuard: TypeAlias = Callable[[sqlite3.Connection, "Migration"], None]

_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]{4})_[a-z0-9][a-z0-9_]*\.sql$")


@dataclass(frozen=True, slots=True)
class Migration:
    """One immutable, checksummed migration artifact."""

    version: int
    name: str
    sha256: str
    sql: str


def packaged_migration_source() -> Traversable:
    """Return the wheel-safe package directory containing SQL migrations."""

    return files("dithyramba.store").joinpath("sql")


def discover_migrations(source: MigrationSource | None = None) -> tuple[Migration, ...]:
    """Load a contiguous set of numbered UTF-8 SQL migrations.

    SQL files with an invalid name, duplicate versions, gaps, or invalid UTF-8
    fail closed. Non-SQL package files such as ``__init__.py`` are ignored.
    """

    migration_source = _coerce_source(source)
    try:
        entries = tuple(migration_source.iterdir())
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise MigrationDiscoveryError(
            f"migration source cannot be read: {migration_source}"
        ) from exc

    migrations: list[Migration] = []
    seen_versions: set[int] = set()
    for entry in sorted(entries, key=lambda item: item.name):
        if not entry.is_file() or not entry.name.endswith(".sql"):
            continue

        match = _MIGRATION_NAME.fullmatch(entry.name)
        if match is None:
            raise MigrationDiscoveryError(f"invalid migration filename: {entry.name}")

        version = int(match.group("version"))
        if version <= 0 or version in seen_versions:
            raise MigrationDiscoveryError(f"duplicate or invalid migration version: {version}")

        try:
            payload = entry.read_bytes()
            sql = payload.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise MigrationDiscoveryError(f"migration cannot be decoded: {entry.name}") from exc

        if not sql.strip():
            raise MigrationDiscoveryError(f"migration is empty: {entry.name}")

        seen_versions.add(version)
        migrations.append(
            Migration(
                version=version,
                name=entry.name,
                sha256=hashlib.sha256(payload).hexdigest(),
                sql=sql,
            )
        )

    if not migrations:
        raise MigrationDiscoveryError("no numbered SQL migrations were found")

    migrations.sort(key=lambda migration: migration.version)
    actual_versions = [migration.version for migration in migrations]
    expected_versions = list(range(1, len(migrations) + 1))
    if actual_versions != expected_versions:
        raise MigrationDiscoveryError(
            f"migration versions must be contiguous from 1: got {actual_versions}"
        )
    return tuple(migrations)


class MigrationRunner:
    """Verify an immutable history and apply every pending migration once."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        source: MigrationSource | None = None,
    ) -> None:
        self._connection = connection
        self._migrations = discover_migrations(source)

    @property
    def migrations(self) -> tuple[Migration, ...]:
        """Return the discovered immutable migration set."""

        return self._migrations

    @property
    def latest_version(self) -> int:
        """Return the newest version understood by this runner."""

        return self._migrations[-1].version

    def current_version(self) -> int:
        """Return the highest applied version after validating history."""

        applied = self.verify()
        return applied[-1].version if applied else 0

    def verify(self) -> tuple[Migration, ...]:
        """Fail closed when migration history or actual SQLite schema differs."""

        if not self._migration_table_exists():
            unmanaged = self._application_tables()
            if unmanaged:
                raise MigrationStateError(
                    "database has application tables but no schema_migrations history: "
                    + ", ".join(unmanaged)
                )
            return ()

        try:
            rows = self._connection.execute(
                "SELECT version, name, sha256 FROM schema_migrations ORDER BY version"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise MigrationStateError("schema_migrations cannot be read") from exc

        expected_by_version = {migration.version: migration for migration in self._migrations}
        verified: list[Migration] = []
        previous_version = 0
        for raw_version, raw_name, raw_sha256 in rows:
            if not isinstance(raw_version, int):
                raise MigrationStateError("migration version is not an integer")
            if raw_version != previous_version + 1:
                raise MigrationStateError(
                    f"applied migration history is non-contiguous at version {raw_version}"
                )
            if raw_version > self.latest_version:
                raise NewerSchemaError(
                    f"database schema version {raw_version} is newer than code version "
                    f"{self.latest_version}"
                )

            expected = expected_by_version.get(raw_version)
            if expected is None:
                raise MigrationStateError(f"code has no migration for version {raw_version}")
            if raw_name != expected.name:
                raise MigrationStateError(
                    f"migration {raw_version} name changed: database={raw_name!r}, "
                    f"code={expected.name!r}"
                )
            if raw_sha256 != expected.sha256:
                raise MigrationChecksumError(
                    f"migration {expected.name} checksum differs from applied history"
                )

            verified.append(expected)
            previous_version = raw_version
        if not verified:
            unmanaged = self._application_tables()
            raise MigrationStateError(
                "schema_migrations exists but history is empty; application tables: "
                + ", ".join(unmanaged)
            )
        self._verify_schema_fingerprint(len(verified))
        return tuple(verified)

    def verify_at_head(self) -> tuple[Migration, ...]:
        """Verify history/schema and require the database to be fully migrated."""

        applied = self.verify()
        if len(applied) != len(self._migrations):
            raise MigrationStateError(
                f"database schema version {len(applied)} is behind code version "
                f"{self.latest_version}"
            )
        return applied

    def apply_all(
        self,
        *,
        backup_hook: BackupHook | None = None,
        locked_guard: LockedMigrationGuard | None = None,
    ) -> tuple[Migration, ...]:
        """Apply pending migrations and return the migrations applied in this call.

        Every migration gets its own ``BEGIN IMMEDIATE`` transaction. When the
        database already had an applied schema at entry, a caller must supply a
        backup hook before each pending version 2 or later. A brand-new empty
        database has no prior state to preserve and may reach head directly.
        An optional locked guard runs after that transaction acquires SQLite's
        write lock and before the first migration SQL statement.
        """

        if self._connection.in_transaction:
            raise TransactionStateError("cannot migrate inside an active transaction")

        applied = self.verify()
        applied_count = len(applied)
        database_had_schema = applied_count > 0
        pending = self._migrations[applied_count:]
        completed: list[Migration] = []
        for migration in pending:
            if database_had_schema and migration.version > 1:
                if backup_hook is None:
                    raise MigrationBackupRequiredError(
                        f"verified backup hook required before {migration.name}"
                    )
                try:
                    backup_hook(self._connection, migration)
                except Exception as exc:
                    raise MigrationApplyError(
                        f"backup verification failed before {migration.name}"
                    ) from exc
                if _connection_in_transaction(self._connection):
                    self._connection.rollback()
                    raise TransactionStateError("backup hook left an active transaction")

            self._apply_one(migration, locked_guard=locked_guard)
            completed.append(migration)

        self.verify_at_head()
        return tuple(completed)

    def _apply_one(
        self,
        migration: Migration,
        *,
        locked_guard: LockedMigrationGuard | None = None,
    ) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            if locked_guard is not None:
                try:
                    locked_guard(self._connection, migration)
                except Exception as exc:
                    raise MigrationApplyError(
                        f"locked migration guard failed before {migration.name}"
                    ) from exc
                if not _connection_in_transaction(self._connection):
                    raise TransactionStateError("locked migration guard ended the transaction")
            for statement in _iter_sql_statements(migration.sql):
                self._connection.execute(statement)
            self._connection.execute(
                """
                INSERT INTO schema_migrations(version, name, sha256, applied_at)
                VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%f000Z', 'now'))
                """,
                (migration.version, migration.name, migration.sha256),
            )
            self._connection.commit()
        except sqlite3.DatabaseError as exc:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise MigrationApplyError(f"failed to apply {migration.name}") from exc
        except BaseException:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise

    def _verify_schema_fingerprint(self, applied_count: int) -> None:
        actual = schema_fingerprint(self._connection)
        expected = _expected_schema_fingerprint(self._migrations, applied_count)
        if actual != expected:
            raise MigrationStateError(
                "actual SQLite schema differs from the checksummed migration history"
            )

    def _migration_table_exists(self) -> bool:
        row = self._connection.execute(
            """
            SELECT 1
            FROM sqlite_schema
            WHERE type = 'table' AND name = 'schema_migrations'
            """
        ).fetchone()
        return row is not None

    def _application_tables(self) -> tuple[str, ...]:
        rows = self._connection.execute(
            """
            SELECT name
            FROM sqlite_schema
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        return tuple(str(row[0]) for row in rows)


def _coerce_source(source: MigrationSource | None) -> Traversable:
    if source is None:
        return packaged_migration_source()
    if isinstance(source, (str, Path)):
        return Path(source)
    return source


def _iter_sql_statements(sql: str) -> Iterator[str]:
    """Yield complete SQLite statements without using ``executescript``.

    ``sqlite3.Connection.executescript`` commits an open transaction before it
    runs. Accumulating through ``sqlite3.complete_statement`` preserves trigger
    bodies while keeping the runner's explicit transaction atomic.
    """

    buffer: list[str] = []
    for character in sql:
        buffer.append(character)
        if character != ";":
            continue
        candidate = "".join(buffer)
        if sqlite3.complete_statement(candidate):
            if _contains_sql(candidate):
                yield candidate
            buffer.clear()

    remainder = "".join(buffer)
    if _contains_sql(remainder):
        raise MigrationDiscoveryError("migration ends with an incomplete SQL statement")


def _contains_sql(value: str) -> bool:
    without_block_comments = re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL)
    without_line_comments = re.sub(r"--[^\n]*(?:\n|$)", "", without_block_comments)
    return bool(without_line_comments.strip())


def _connection_in_transaction(connection: sqlite3.Connection) -> bool:
    """Return transaction state without relying on typeshed's literal narrowing."""

    return bool(connection.in_transaction)


def migration_checksums(
    migrations: Iterable[Migration],
) -> dict[int, str]:
    """Return a version-to-checksum map for manifests and diagnostics."""

    return {migration.version: migration.sha256 for migration in migrations}


def schema_fingerprint(connection: sqlite3.Connection) -> str:
    """Hash the exact managed table/index/trigger/view definitions."""

    rows = connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name, tbl_name
        """
    ).fetchall()
    manifest = [
        {
            "type": str(row[0]),
            "name": str(row[1]),
            "table": str(row[2]),
            "sql": None if row[3] is None else str(row[3]),
        }
        for row in rows
    ]
    return canonical_sha256_hex(
        {"schema": "dithyramba.sqlite_schema_manifest/1.0", "objects": manifest}
    )


@lru_cache(maxsize=8)
def _expected_schema_fingerprint(
    migrations: tuple[Migration, ...],
    applied_count: int,
) -> str:
    """Build the trusted expected schema from immutable migration resources."""

    if applied_count <= 0 or applied_count > len(migrations):
        raise MigrationStateError(f"invalid applied migration count: {applied_count}")
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for migration in migrations[:applied_count]:
            connection.execute("BEGIN")
            try:
                for statement in _iter_sql_statements(migration.sql):
                    connection.execute(statement)
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()
        return schema_fingerprint(connection)
    except sqlite3.DatabaseError as exc:
        raise MigrationStateError("trusted migration set cannot reconstruct its schema") from exc
    finally:
        connection.close()


def verify_migrations_smoke() -> dict[str, object]:
    """Apply packaged migrations from empty to head and return stable evidence."""

    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        runner = MigrationRunner(connection)
        applied = runner.apply_all()
        runner.verify_at_head()
        return {
            "status": "ok",
            "schema_version": runner.latest_version,
            "schema_fingerprint": schema_fingerprint(connection),
            "migration_checksums": {
                str(version): checksum for version, checksum in migration_checksums(applied).items()
            },
        }
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    """Run the standalone deterministic migration smoke command."""

    parser = argparse.ArgumentParser(prog="python -m dithyramba.store.migrations")
    parser.add_argument("command", choices=("verify",))
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        print(json.dumps(verify_migrations_smoke(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess smoke
    raise SystemExit(main())
