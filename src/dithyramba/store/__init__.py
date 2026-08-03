"""Dithyramba's typed SQLite persistence boundary."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dithyramba.store.database import Store, verify_integrity
    from dithyramba.store.errors import (
        IntegrityCheckError,
        LegacySchemaError,
        MigrationApplyError,
        MigrationBackupRequiredError,
        MigrationChecksumError,
        MigrationDiscoveryError,
        MigrationError,
        MigrationStateError,
        NewerSchemaError,
        StoreClosedError,
        StoreError,
        StoreOpenError,
        TransactionStateError,
    )
    from dithyramba.store.migrations import (
        Migration,
        MigrationRunner,
        discover_migrations,
        migration_checksums,
        packaged_migration_source,
        schema_fingerprint,
        verify_migrations_smoke,
    )

_EXPORT_MODULES = {
    "IntegrityCheckError": "dithyramba.store.errors",
    "LegacySchemaError": "dithyramba.store.errors",
    "Migration": "dithyramba.store.migrations",
    "MigrationApplyError": "dithyramba.store.errors",
    "MigrationBackupRequiredError": "dithyramba.store.errors",
    "MigrationChecksumError": "dithyramba.store.errors",
    "MigrationDiscoveryError": "dithyramba.store.errors",
    "MigrationError": "dithyramba.store.errors",
    "MigrationRunner": "dithyramba.store.migrations",
    "MigrationStateError": "dithyramba.store.errors",
    "NewerSchemaError": "dithyramba.store.errors",
    "Store": "dithyramba.store.database",
    "StoreClosedError": "dithyramba.store.errors",
    "StoreError": "dithyramba.store.errors",
    "StoreOpenError": "dithyramba.store.errors",
    "TransactionStateError": "dithyramba.store.errors",
    "discover_migrations": "dithyramba.store.migrations",
    "migration_checksums": "dithyramba.store.migrations",
    "packaged_migration_source": "dithyramba.store.migrations",
    "schema_fingerprint": "dithyramba.store.migrations",
    "verify_integrity": "dithyramba.store.database",
    "verify_migrations_smoke": "dithyramba.store.migrations",
}

__all__ = [
    "IntegrityCheckError",
    "LegacySchemaError",
    "Migration",
    "MigrationApplyError",
    "MigrationBackupRequiredError",
    "MigrationChecksumError",
    "MigrationDiscoveryError",
    "MigrationError",
    "MigrationRunner",
    "MigrationStateError",
    "NewerSchemaError",
    "Store",
    "StoreClosedError",
    "StoreError",
    "StoreOpenError",
    "TransactionStateError",
    "discover_migrations",
    "migration_checksums",
    "packaged_migration_source",
    "schema_fingerprint",
    "verify_integrity",
    "verify_migrations_smoke",
]


def __getattr__(name: str) -> Any:
    """Load public store symbols without pre-importing the migrations CLI module."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
