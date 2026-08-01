"""Typed failures for Dithyramba's Library-scoped repository layer."""

from __future__ import annotations


class PersistenceError(RuntimeError):
    """Base class for repository, outbox, and backup failures."""


class PersistenceIntegrityError(PersistenceError):
    """Persisted state does not match its canonical or relational contract."""


class LibraryNotFoundError(PersistenceError):
    """A requested physical Library installation does not exist."""


class CollectionNotFoundError(PersistenceError):
    """A Collection does not exist inside the current Library."""


class AccessPolicyNotFoundError(PersistenceError):
    """An AccessPolicy does not exist inside the current Library."""


class SourceNotFoundError(PersistenceError):
    """A Source does not exist inside the current Library."""


class SourceVersionNotFoundError(PersistenceError):
    """A SourceVersion does not exist inside the current Library."""


class ProcessingRunNotFoundError(PersistenceError):
    """A ProcessingRun does not exist in the current Library database."""


class CoverageReportNotFoundError(PersistenceError):
    """A CoverageReport does not exist in the current Library database."""


class CorpusSnapshotNotFoundError(PersistenceError):
    """A CorpusSnapshot does not exist in the current Library database."""


class QueryRequestNotFoundError(PersistenceError):
    """A QueryRequest does not exist in the current Library database."""


class EvidencePacketNotFoundError(PersistenceError):
    """An EvidencePacket does not exist in the current Library database."""


class RecallArtifactNotFoundError(PersistenceError):
    """A required immutable recall artifact is absent from the database."""


class AnswerProjectionNotFoundError(PersistenceError):
    """A proposition projection or its judgment receipt does not exist."""


class AnswerProjectionPersistenceError(PersistenceError):
    """A proposition projection cannot be stored or reopened exactly."""


class ResearchSessionNotFoundError(PersistenceError):
    """A research session does not exist in the current Library."""


class ResearchSessionPersistenceError(PersistenceError):
    """A research session or event cannot be stored or reopened exactly."""


class SessionArtifactNotFoundError(ResearchSessionPersistenceError):
    """A session event references an absent, stale, or out-of-scope artifact."""


class PersistenceConflictError(PersistenceError):
    """An insert-only repository write conflicts with existing state."""


class OutboxExportError(PersistenceError):
    """The durable outbox cannot be exported without losing audit integrity."""


class BackupError(PersistenceError):
    """An online SQLite backup could not be verified and promoted."""


class AuthorizationError(PersistenceError):
    """A fragment read is not backed by a current repository authorization."""
