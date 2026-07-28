"""Typed failures for logical Collection configuration and roots."""

from __future__ import annotations

from dithyramba.contracts import ContractError


class CollectionError(ContractError):
    """Base class for an invalid Collection contract."""


class InvalidCollectionIdError(CollectionError):
    """A Collection identifier does not follow the UUID4 contract."""


class InvalidCollectionNameError(CollectionError):
    """A Collection name is empty, padded, or too long."""


class InvalidCollectionGlobError(CollectionError):
    """A Collection include/exclude glob is ambiguous or unsafe."""


class CollectionRootError(CollectionError):
    """Base class for an invalid physical Collection root."""


class CollectionRootPathError(CollectionRootError):
    """A root path is relative or contains parent traversal."""


class CollectionRootNotDirectoryError(CollectionRootError):
    """A root path does not resolve to an existing directory."""


class CollectionRootOverlapError(CollectionRootError):
    """A Collection root overlaps live application data."""


class SourcePathEscapeError(CollectionRootError):
    """A source path resolves outside its declared Collection root."""


class SourceIdentityManifestError(CollectionRootError):
    """A pinned administrative source-identity manifest is unsafe or invalid."""
