"""Immutable configuration values for a physical Library boundary."""

from __future__ import annotations

import re
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dithyramba.contracts import new_id

from .errors import InvalidLibraryIdError, InvalidLibraryNameError

_LIBRARY_ID_PATTERN = re.compile(r"^library_([0-9a-f]{32})$")
_MAX_NAME_LENGTH = 200


def new_library_id() -> str:
    """Return a new opaque Library ID using the shared UUID4 contract."""

    return new_id("library")


def validate_library_id(value: str) -> str:
    """Validate and return an opaque ``library_<uuid4 hex>`` identifier."""

    if type(value) is not str:
        raise InvalidLibraryIdError("library_id must be a string")
    match = _LIBRARY_ID_PATTERN.fullmatch(value)
    if match is None:
        raise InvalidLibraryIdError("library_id must match library_<uuid4 hex>")
    parsed = uuid.UUID(hex=match.group(1))
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise InvalidLibraryIdError("library_id must contain an RFC 4122 UUID4")
    return value


def validate_library_name(value: str) -> str:
    """Validate a stable human-facing Library label without rewriting it."""

    if type(value) is not str:
        raise InvalidLibraryNameError("Library name must be a string")
    if not value or value != value.strip():
        raise InvalidLibraryNameError("Library name must be non-empty and unpadded")
    if len(value) > _MAX_NAME_LENGTH:
        raise InvalidLibraryNameError("Library name must contain at most 200 code points")
    return value


class LibraryConfig(BaseModel):
    """Immutable portable identity for one physical privacy/rights boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    library_id: str = Field(default_factory=new_library_id)
    name: str

    @field_validator("library_id")
    @classmethod
    def _library_id_is_valid(cls, value: str) -> str:
        return validate_library_id(value)

    @field_validator("name")
    @classmethod
    def _name_is_valid(cls, value: str) -> str:
        return validate_library_name(value)

    @property
    def id(self) -> str:
        """Expose the opaque ID for generic integration code."""

        return self.library_id
