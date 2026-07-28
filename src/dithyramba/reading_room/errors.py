"""Stable, non-disclosing errors for Reading Room projections."""

from __future__ import annotations


class ReadingRoomError(Exception):
    """Base class for read-only projection failures."""


class ReadingRoomAuthorizationError(ReadingRoomError):
    """The requested projection scope could not be authorized."""


class ReadingRoomIntegrityError(ReadingRoomError):
    """Persisted metadata could not be projected without weakening closure."""


class ReadingRoomLimitError(ReadingRoomError):
    """A requested projection bound is outside the supported safe range."""
