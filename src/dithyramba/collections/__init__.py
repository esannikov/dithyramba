"""Logical Collection models and filesystem boundary validation."""

from .boundaries import build_collection_root, resolve_source_path
from .errors import (
    CollectionError,
    CollectionRootError,
    CollectionRootNotDirectoryError,
    CollectionRootOverlapError,
    CollectionRootPathError,
    InvalidCollectionGlobError,
    InvalidCollectionIdError,
    InvalidCollectionNameError,
    SourceIdentityManifestError,
    SourcePathEscapeError,
)
from .identity_manifest import (
    SOURCE_IDENTITY_MANIFEST_SCHEMA,
    SourceIdentityDeclaration,
    SourceIdentityManifest,
    load_source_identity_manifest,
)
from .models import (
    CollectionConfig,
    CollectionKind,
    CollectionRoot,
    new_collection_id,
    validate_collection_id,
    validate_collection_name,
    validate_globs,
)

__all__ = [
    "SOURCE_IDENTITY_MANIFEST_SCHEMA",
    "CollectionConfig",
    "CollectionError",
    "CollectionKind",
    "CollectionRoot",
    "CollectionRootError",
    "CollectionRootNotDirectoryError",
    "CollectionRootOverlapError",
    "CollectionRootPathError",
    "InvalidCollectionGlobError",
    "InvalidCollectionIdError",
    "InvalidCollectionNameError",
    "SourceIdentityDeclaration",
    "SourceIdentityManifest",
    "SourceIdentityManifestError",
    "SourcePathEscapeError",
    "build_collection_root",
    "load_source_identity_manifest",
    "new_collection_id",
    "resolve_source_path",
    "validate_collection_id",
    "validate_collection_name",
    "validate_globs",
]
