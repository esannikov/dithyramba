"""Typed failures for Library identity and filesystem boundaries."""

from __future__ import annotations

from dithyramba.contracts import ContractError


class LibraryError(ContractError):
    """Base class for invalid Library configuration or placement."""


class InvalidLibraryIdError(LibraryError):
    """A Library identifier does not follow the UUID4 contract."""


class InvalidLibraryNameError(LibraryError):
    """A Library name is empty, padded, or too long."""


class ApplicationDataPathError(LibraryError):
    """The application-data location is ambiguous or unsafe."""


class PathOverlapError(ApplicationDataPathError):
    """Live application data overlaps a source or synchronization root."""


class LibraryAlreadyExistsError(LibraryError):
    """The requested Library directory already exists."""


class LibraryLayoutError(LibraryError):
    """A Library layout could not be created or does not match its contract."""
