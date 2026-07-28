"""Complete portable BackupBundle creation, verification, restore, and migration."""

from .bundle import (
    create_backup_bundle,
    create_pre_migration_backup_bundle,
    restore_backup_bundle,
    verify_backup_bundle,
)
from .errors import (
    BackupBundleConflictError,
    BackupBundleError,
    BackupBundleIntegrityError,
    LibraryMigrationError,
)
from .migration import LibraryMigrationResult, migrate_library, migrate_library_from_backup
from .models import (
    BackupBundleManifest,
    BackupBundleRecord,
    BackupFileEntry,
    BackupMigrationEntry,
    LibraryMigrationReceipt,
)

__all__ = [
    "BackupBundleConflictError",
    "BackupBundleError",
    "BackupBundleIntegrityError",
    "BackupBundleManifest",
    "BackupBundleRecord",
    "BackupFileEntry",
    "BackupMigrationEntry",
    "LibraryMigrationError",
    "LibraryMigrationReceipt",
    "LibraryMigrationResult",
    "create_backup_bundle",
    "create_pre_migration_backup_bundle",
    "migrate_library",
    "migrate_library_from_backup",
    "restore_backup_bundle",
    "verify_backup_bundle",
]
