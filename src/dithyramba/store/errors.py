"""Typed persistence failures for Dithyramba's SQLite boundary."""

from __future__ import annotations


class StoreError(RuntimeError):
    """Base class for failures at the SQLite persistence boundary."""


class StoreOpenError(StoreError):
    """A SQLite store could not be opened with the required profile."""


class StoreClosedError(StoreError):
    """An operation was attempted after the store had been closed."""


class TransactionStateError(StoreError):
    """A transaction cannot start in the connection's current state."""


class IntegrityCheckError(StoreError):
    """SQLite integrity or foreign-key verification failed."""


class MigrationError(StoreError):
    """Base class for migration discovery, verification, or application failures."""


class MigrationDiscoveryError(MigrationError):
    """The immutable migration set is missing, malformed, or non-monotonic."""


class MigrationStateError(MigrationError):
    """The database migration history is malformed or unmanaged."""


class LegacySchemaError(MigrationStateError):
    """A pre-v1 Library was detected and must not be changed in place."""


class MigrationChecksumError(MigrationStateError):
    """An applied migration no longer has its recorded checksum."""


class NewerSchemaError(MigrationStateError):
    """The database schema is newer than the migration set understood by this code."""


class MigrationApplyError(MigrationError):
    """A migration failed and its transaction was rolled back."""


class MigrationBackupRequiredError(MigrationError):
    """A non-initial migration was attempted without a verified backup hook."""
