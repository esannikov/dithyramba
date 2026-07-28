"""Typed failures for scoped human review operations."""

from __future__ import annotations


class ReviewError(RuntimeError):
    """Base class for ReviewDecision operations."""


class ReviewTargetNotFoundError(ReviewError):
    """A target is absent from the explicitly selected Library."""


class ReviewDecisionNotFoundError(ReviewError):
    """A ReviewDecision is absent from the explicitly selected Library."""


class ReviewIntegrityError(ReviewError):
    """Persisted or requested review state violates its closure contract."""


class ReviewConflictError(ReviewError):
    """An append-only decision conflicts with already persisted state."""
