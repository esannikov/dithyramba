"""Typed failures for provenance storage and immutable source history."""

from __future__ import annotations


class ProvenanceError(RuntimeError):
    """Base failure for provenance and content-addressed blob operations."""


class BlobStoreError(ProvenanceError):
    """A blob could not be promoted or verified without weakening integrity."""


class BlobIntegrityError(BlobStoreError):
    """Stored blob bytes or filesystem metadata disagree with their digest."""


class SourceHistoryConflictError(ProvenanceError):
    """The frozen schema cannot represent the requested immutable history."""


class SourceLineageIntegrityError(ProvenanceError):
    """SourceFamily state is missing, ambiguous, or internally inconsistent."""
