"""Typed BackupBundle and restore failures."""

from __future__ import annotations


class BackupBundleError(RuntimeError):
    """Base class for complete portable backup operations."""


class BackupBundleIntegrityError(BackupBundleError):
    """A bundle fails its canonical, cryptographic, or relational checks."""


class BackupBundleConflictError(BackupBundleError):
    """Creation or restore would overwrite an existing artifact or Library."""


class LibraryMigrationError(BackupBundleError):
    """A backup-first Library migration could not be proven safe."""
