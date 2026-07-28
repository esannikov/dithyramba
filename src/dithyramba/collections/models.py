"""Immutable logical Collection configuration models."""

from __future__ import annotations

import re
import uuid
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dithyramba.contracts import new_id
from dithyramba.library.models import validate_library_id

from .errors import (
    InvalidCollectionGlobError,
    InvalidCollectionIdError,
    InvalidCollectionNameError,
)

_COLLECTION_ID_PATTERN = re.compile(r"^collection_([0-9a-f]{32})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_NAME_LENGTH = 200


class CollectionKind(StrEnum):
    """Frozen VS0 Collection roles."""

    CORPUS = "corpus"
    CASE = "case"
    BRANCH = "branch"
    HOLDOUT = "holdout"


def new_collection_id() -> str:
    """Return a new opaque Collection ID using the shared UUID4 contract."""

    return new_id("collection")


def validate_collection_id(value: str) -> str:
    """Validate and return an opaque ``collection_<uuid4 hex>`` ID."""

    if type(value) is not str:
        raise InvalidCollectionIdError("collection_id must be a string")
    match = _COLLECTION_ID_PATTERN.fullmatch(value)
    if match is None:
        raise InvalidCollectionIdError("collection_id must match collection_<uuid4 hex>")
    parsed = uuid.UUID(hex=match.group(1))
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise InvalidCollectionIdError("collection_id must contain an RFC 4122 UUID4")
    return value


def validate_collection_name(value: str) -> str:
    """Validate a human-facing Collection label without rewriting it."""

    if type(value) is not str:
        raise InvalidCollectionNameError("Collection name must be a string")
    if not value or value != value.strip():
        raise InvalidCollectionNameError("Collection name must be non-empty and unpadded")
    if len(value) > _MAX_NAME_LENGTH:
        raise InvalidCollectionNameError("Collection name must contain at most 200 code points")
    return value


def validate_globs(values: tuple[str, ...]) -> tuple[str, ...]:
    """Reject absolute, duplicate, empty, or parent-traversing glob patterns."""

    if len(set(values)) != len(values):
        raise InvalidCollectionGlobError("Collection globs must be unique")
    for value in values:
        if not value or "\x00" in value:
            raise InvalidCollectionGlobError("Collection globs must be non-empty text")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise InvalidCollectionGlobError("Collection globs must stay relative to their root")
    return values


class CollectionRoot(BaseModel):
    """A canonical root and deterministic ingest include/exclude rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    identity_manifest_path: Path | None = None
    identity_manifest_sha256: str | None = None
    include_globs: tuple[str, ...] = (
        "**/*.md",
        "**/*.markdown",
        "**/*.txt",
        "**/*.pdf",
    )
    exclude_globs: tuple[str, ...] = ()

    @field_validator("path")
    @classmethod
    def _path_is_unambiguous(cls, value: Path) -> Path:
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("CollectionRoot.path must be absolute and contain no '..'")
        return value

    @field_validator("identity_manifest_path")
    @classmethod
    def _manifest_path_is_unambiguous(cls, value: Path | None) -> Path | None:
        if value is not None and (not value.is_absolute() or ".." in value.parts):
            raise ValueError(
                "CollectionRoot.identity_manifest_path must be absolute and contain no '..'"
            )
        return value

    @field_validator("identity_manifest_sha256")
    @classmethod
    def _manifest_hash_is_valid(cls, value: str | None) -> str | None:
        if value is not None and (
            type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None
        ):
            raise ValueError(
                "CollectionRoot.identity_manifest_sha256 must be lowercase SHA-256 hex"
            )
        return value

    @field_validator("include_globs", "exclude_globs")
    @classmethod
    def _globs_are_valid(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return validate_globs(value)


class CollectionConfig(BaseModel):
    """Immutable logical namespace inside exactly one Library."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_id: str = Field(default_factory=new_collection_id)
    library_id: str
    name: str
    kind: CollectionKind
    roots: tuple[CollectionRoot, ...] = ()

    @field_validator("collection_id")
    @classmethod
    def _collection_id_is_valid(cls, value: str) -> str:
        return validate_collection_id(value)

    @field_validator("library_id")
    @classmethod
    def _library_id_is_valid(cls, value: str) -> str:
        return validate_library_id(value)

    @field_validator("name")
    @classmethod
    def _name_is_valid(cls, value: str) -> str:
        return validate_collection_name(value)

    @property
    def id(self) -> str:
        """Expose the opaque ID for generic integration code."""

        return self.collection_id
