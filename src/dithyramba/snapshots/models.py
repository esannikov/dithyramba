"""Strict immutable contracts for content-addressed CorpusSnapshots."""

from __future__ import annotations

import re
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from dithyramba.access import MembershipState
from dithyramba.contracts import canonical_content_id, canonical_json_bytes, canonical_sha256_hex
from dithyramba.provenance import SourceFamilyRole

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class SnapshotContractError(ValueError):
    """A CorpusSnapshot value violates the frozen VS0 contract."""


class SnapshotMember(BaseModel):
    """One Collection/SourceVersion membership with explicit root lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    collection_id: str
    source_id: str
    source_version_id: str
    content_sha256: str
    membership_state: MembershipState
    source_family_id: str
    root_source_id: str
    family_role: SourceFamilyRole

    @field_validator("collection_id")
    @classmethod
    def _collection_id(cls, value: str) -> str:
        return _require_id(value, "collection")

    @field_validator("source_id", "root_source_id")
    @classmethod
    def _source_id(cls, value: str) -> str:
        return _require_id(value, "source")

    @field_validator("source_version_id")
    @classmethod
    def _source_version_id(cls, value: str) -> str:
        return _require_id(value, "source_version")

    @field_validator("source_family_id")
    @classmethod
    def _source_family_id(cls, value: str) -> str:
        return _require_id(value, "family")

    @field_validator("content_sha256")
    @classmethod
    def _content_hash(cls, value: str) -> str:
        return _require_hash(value, "content_sha256")

    @model_validator(mode="after")
    def _lineage_is_consistent(self) -> Self:
        if self.family_role is SourceFamilyRole.ROOT and self.root_source_id != self.source_id:
            raise SnapshotContractError("root lineage must point to the member Source")
        if self.family_role is not SourceFamilyRole.ROOT and self.root_source_id == self.source_id:
            raise SnapshotContractError("non-root lineage must point to another root Source")
        return self

    def payload(self) -> dict[str, object]:
        """Return the stable member manifest payload."""

        return {
            "collection_id": self.collection_id,
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "content_sha256": self.content_sha256,
            "membership_state": self.membership_state.value,
            "source_family_id": self.source_family_id,
            "root_source_id": self.root_source_id,
            "family_role": self.family_role.value,
        }


class CorpusSnapshot(BaseModel):
    """Canonical snapshot scope and immutable SourceVersion manifest."""

    SCHEMA: ClassVar[str] = "dithyramba.corpus_snapshot_manifest/1.0"
    SCOPE_SCHEMA: ClassVar[str] = "dithyramba.corpus_snapshot_scope/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    library_id: str
    collection_ids: tuple[str, ...]
    members: tuple[SnapshotMember, ...] = ()

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("collection_ids")
    @classmethod
    def _collection_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise SnapshotContractError("snapshot scope requires at least one Collection")
        if len(value) > 16:
            raise SnapshotContractError("snapshot scope permits at most 16 Collections")
        for collection_id in value:
            _require_id(collection_id, "collection")
        if len(set(value)) != len(value):
            raise SnapshotContractError("snapshot Collection IDs must be unique")
        return tuple(sorted(value))

    @field_validator("members")
    @classmethod
    def _members(cls, value: tuple[SnapshotMember, ...]) -> tuple[SnapshotMember, ...]:
        if any(type(member) is not SnapshotMember for member in value):
            raise SnapshotContractError("snapshot members must be SnapshotMember values")
        keys = [(member.collection_id, member.source_version_id) for member in value]
        if len(set(keys)) != len(keys):
            raise SnapshotContractError("snapshot member keys must be unique")
        return tuple(sorted(value, key=lambda item: (item.collection_id, item.source_version_id)))

    @model_validator(mode="after")
    def _manifest_is_in_scope(self) -> Self:
        scope = set(self.collection_ids)
        if any(member.collection_id not in scope for member in self.members):
            raise SnapshotContractError("snapshot member belongs to a Collection outside scope")
        lineage_by_version: dict[str, tuple[object, ...]] = {}
        for member in self.members:
            lineage = (
                member.source_id,
                member.content_sha256,
                member.source_family_id,
                member.root_source_id,
                member.family_role,
            )
            previous = lineage_by_version.setdefault(member.source_version_id, lineage)
            if previous != lineage:
                raise SnapshotContractError(
                    "one SourceVersion has inconsistent content or lineage across Collections"
                )
        return self

    def semantic_payload(self) -> dict[str, object]:
        """Return the complete content-addressed manifest hash domain."""

        return {
            "schema": self.SCHEMA,
            "scope": {
                "schema": self.SCOPE_SCHEMA,
                "library_id": self.library_id,
                "collection_ids": list(self.collection_ids),
            },
            "members": [member.payload() for member in self.members],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def corpus_snapshot_id(self) -> str:
        return canonical_content_id("snapshot", self.semantic_payload())


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise SnapshotContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise SnapshotContractError(
            f"identifier must use the {expected} prefix with a canonical non-empty suffix"
        )
    return value


def _require_hash(value: str, label: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise SnapshotContractError(f"{label} must be lowercase SHA-256 hex")
    return value
