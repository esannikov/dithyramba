"""SQLite connection profile and transaction boundary for one Library."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType

from dithyramba.library import LibraryPaths, validate_library_layout
from dithyramba.store.errors import (
    IntegrityCheckError,
    StoreClosedError,
    StoreOpenError,
    TransactionStateError,
)
from dithyramba.store.migrations import MigrationRunner, MigrationSource


class Store:
    """One profiled SQLite connection for a Dithyramba Library database."""

    def __init__(self, path: str | Path, connection: sqlite3.Connection) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = connection

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        migration_source: MigrationSource | None = None,
        apply_migrations: bool = True,
    ) -> Store:
        """Open a database, enforce the VS0 profile, and verify its schema."""

        return cls._open_verified(
            path,
            migration_source=migration_source,
            apply_migrations=apply_migrations,
            require_head=True,
        )

    @classmethod
    def open_for_migration(
        cls,
        path: str | Path,
        *,
        migration_source: MigrationSource | None = None,
    ) -> Store:
        """Open a checksummed historical schema only for explicit migration work.

        This seam validates the applied migration prefix, its reconstructed
        schema fingerprint, and SQLite integrity without applying anything and
        without claiming that the database is at the code's current head.
        Normal callers must keep using :meth:`open`.
        """

        return cls._open_verified(
            path,
            migration_source=migration_source,
            apply_migrations=False,
            require_head=False,
        )

    @classmethod
    def _open_verified(
        cls,
        path: str | Path,
        *,
        migration_source: MigrationSource | None,
        apply_migrations: bool,
        require_head: bool,
    ) -> Store:
        """Open one database under an explicit migration verification mode."""

        database_path, created_file = _prepare_database_file(path)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                database_path,
                isolation_level=None,
                timeout=5.0,
                # Repository authorization is stateful and SQLite invokes an
                # authorizer while preparing a statement. Reusing a statement
                # prepared inside a protected read could otherwise bypass the
                # authorizer after that read scope closes.
                cached_statements=0,
            )
            connection.row_factory = sqlite3.Row
            _apply_profile(connection)
            runner = MigrationRunner(connection, migration_source)
            if apply_migrations:
                runner.apply_all()
            elif require_head:
                runner.verify_at_head()
            else:
                runner.verify()
            verify_integrity(connection)
        except (OSError, sqlite3.DatabaseError) as exc:
            if connection is not None:
                connection.close()
            if created_file:
                database_path.unlink(missing_ok=True)
            raise StoreOpenError(f"could not open SQLite store at {database_path}") from exc
        except Exception:
            if connection is not None:
                connection.close()
            if created_file:
                database_path.unlink(missing_ok=True)
            raise
        if connection is None:  # pragma: no cover - defensive type narrowing
            raise StoreOpenError(f"could not open SQLite store at {database_path}")
        return cls(database_path, connection)

    @classmethod
    def open_library(
        cls,
        paths: LibraryPaths,
        *,
        migration_source: MigrationSource | None = None,
        apply_migrations: bool = True,
    ) -> Store:
        """Open only after validating the complete physical Library boundary."""

        if not isinstance(paths, LibraryPaths):
            raise StoreOpenError("production Library open requires LibraryPaths")
        validate_library_layout(paths)
        return cls.open(
            paths.database,
            migration_source=migration_source,
            apply_migrations=apply_migrations,
        )

    @classmethod
    def open_library_for_migration(
        cls,
        paths: LibraryPaths,
        *,
        migration_source: MigrationSource | None = None,
    ) -> Store:
        """Open a complete Library through the historical-schema migration seam."""

        if not isinstance(paths, LibraryPaths):
            raise StoreOpenError("production Library open requires LibraryPaths")
        validate_library_layout(paths)
        return cls.open_for_migration(paths.database, migration_source=migration_source)

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the live profiled connection."""

        if self._connection is None:
            raise StoreClosedError("SQLite store is closed")
        return self._connection

    @property
    def schema_version(self) -> int:
        """Return the verified migration version."""

        return MigrationRunner(self.connection).current_version()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Run a caller-owned atomic transaction and roll back on any failure."""

        connection = self.connection
        if connection.in_transaction:
            raise TransactionStateError("nested store transactions are not supported")

        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield connection
            connection.commit()
        except BaseException:
            # sqlite3 treats rollback outside an active transaction as a
            # no-op.  Keeping it unconditional also covers commit-time
            # failures without relying on the driver's intermediate state.
            connection.rollback()
            raise

    @contextmanager
    def defer_wal_autocheckpoints(self) -> Iterator[None]:
        """Defer commit-triggered WAL checkpoints without widening transactions.

        The auto-checkpoint threshold is local to this connection. Setting it
        to zero leaves every caller-owned transaction and ``synchronous=FULL``
        commit unchanged while allowing a bounded batch to checkpoint once at
        its outer boundary. A passive checkpoint never discards WAL frames
        that an active reader still needs, so an incomplete checkpoint remains
        crash-recoverable.
        """

        connection = self.connection
        if connection.in_transaction:
            raise TransactionStateError(
                "cannot defer WAL auto-checkpoints inside an active transaction"
            )
        previous_threshold = int(connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0])
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        body_error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            try:
                connection.execute(f"PRAGMA wal_autocheckpoint = {previous_threshold:d}")
            except BaseException as exc:
                cleanup_error = exc
            try:
                _passive_wal_checkpoint(connection)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            if cleanup_error is not None:
                if body_error is None:
                    raise cleanup_error
                body_error.add_note(f"WAL checkpoint cleanup also failed: {cleanup_error}")

    def verify(self) -> None:
        """Verify migration history, SQLite integrity, and foreign keys."""

        MigrationRunner(self.connection).verify_at_head()
        verify_integrity(self.connection)

    def close(self) -> None:
        """Close the connection; repeated closes are safe."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def verify_integrity(connection: sqlite3.Connection) -> None:
    """Fail if SQLite reports corruption or unresolved foreign keys."""

    integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
    integrity_messages = tuple(str(row[0]) for row in integrity_rows)
    if integrity_messages != ("ok",):
        raise IntegrityCheckError("SQLite integrity_check failed: " + "; ".join(integrity_messages))

    foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_rows:
        descriptions = "; ".join("/".join(str(value) for value in row) for row in foreign_key_rows)
        raise IntegrityCheckError(f"SQLite foreign_key_check failed: {descriptions}")


def _passive_wal_checkpoint(connection: sqlite3.Connection) -> None:
    """Checkpoint all currently safe WAL frames without blocking readers."""

    connection.execute("PRAGMA wal_checkpoint(PASSIVE)")


def _apply_profile(connection: sqlite3.Connection) -> None:
    pragmas = (
        "PRAGMA foreign_keys = ON",
        "PRAGMA journal_mode = WAL",
        "PRAGMA synchronous = FULL",
        "PRAGMA busy_timeout = 5000",
        "PRAGMA trusted_schema = OFF",
    )
    for pragma in pragmas:
        connection.execute(pragma)

    foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
    journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    synchronous = int(connection.execute("PRAGMA synchronous").fetchone()[0])
    trusted_schema = int(connection.execute("PRAGMA trusted_schema").fetchone()[0])
    busy_timeout = int(connection.execute("PRAGMA busy_timeout").fetchone()[0])
    if (
        foreign_keys != 1
        or journal_mode != "wal"
        or synchronous != 2
        or trusted_schema != 0
        or busy_timeout < 5000
    ):
        raise StoreOpenError("SQLite connection did not accept the required VS0 profile")


def _prepare_database_file(path: str | Path) -> tuple[Path, bool]:
    """Validate an explicit private path and securely create the DB file if absent.

    High-level production code must additionally validate a complete
    ``LibraryPaths`` boundary and declared source/sync roots. This low-level
    seam deliberately refuses relative paths, implicit parent creation,
    symlink components, non-regular files, hardlinks, and group/other access.
    """

    database_path = Path(path).expanduser()
    if not database_path.is_absolute() or ".." in database_path.parts:
        raise StoreOpenError("SQLite store path must be absolute and contain no '..'")
    if not database_path.parent.exists() or not database_path.parent.is_dir():
        raise StoreOpenError(f"SQLite store parent must already exist: {database_path.parent}")
    _reject_symlink_components(database_path.parent)

    created = False
    if not os.path.lexists(database_path):
        try:
            descriptor = os.open(database_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(descriptor)
        except OSError as exc:
            raise StoreOpenError(
                f"could not securely create SQLite store: {database_path}"
            ) from exc
        created = True
    try:
        metadata = database_path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise StoreOpenError(f"SQLite store must be a regular file: {database_path}")
        if metadata.st_nlink != 1:
            raise StoreOpenError(f"SQLite store must not be hardlinked: {database_path}")
        if os.name == "posix":
            if metadata.st_uid != os.geteuid():
                raise StoreOpenError(f"SQLite store has a different owner: {database_path}")
            database_path.chmod(0o600)
            if stat.S_IMODE(database_path.lstat().st_mode) != 0o600:
                raise StoreOpenError(f"SQLite store permissions are not private: {database_path}")
    except OSError as exc:
        if created:
            database_path.unlink(missing_ok=True)
        raise StoreOpenError(f"could not secure SQLite store: {database_path}") from exc
    except Exception:
        if created:
            database_path.unlink(missing_ok=True)
        raise
    return database_path, created


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component == component.parent:
            continue
        if os.path.lexists(component) and component.is_symlink():
            raise StoreOpenError(f"SQLite store path contains a symlink component: {component}")
