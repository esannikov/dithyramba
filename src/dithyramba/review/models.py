"""Strict, scoped, append-only human review contracts for VS0."""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_USE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)


class ReviewContractError(ValueError):
    """A ReviewDecision value violates its frozen public contract."""


class ReviewTargetType(StrEnum):
    EVIDENCE_PACKET = "evidence_packet"
    PACKET_ITEM = "packet_item"
    SOURCE_FRAGMENT = "source_fragment"


class ReviewAction(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    REVISE = "revise"
    DEFER = "defer"
    SUPERSEDE = "supersede"


class ReviewScope(BaseModel):
    """The exact Collection and use-context in which a decision applies."""

    SCHEMA: ClassVar[str] = "dithyramba.review_scope/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    collection_ids: tuple[str, ...]
    use: str

    @field_validator("collection_ids")
    @classmethod
    def _collections(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 16:
            raise ReviewContractError("ReviewScope requires 1-16 Collections")
        for collection_id in value:
            _require_id(collection_id, "collection")
        if len(set(value)) != len(value):
            raise ReviewContractError("ReviewScope Collection IDs must be unique")
        return tuple(sorted(value))

    @field_validator("use")
    @classmethod
    def _use(cls, value: str) -> str:
        if type(value) is not str or _USE_PATTERN.fullmatch(value) is None:
            raise ReviewContractError("ReviewScope use must match [a-z][a-z0-9_-]{0,63}")
        return value

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "collection_ids": list(self.collection_ids),
            "use": self.use,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def scope_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())


class ReviewDecisionRequest(BaseModel):
    """Optimistic, hash-pinned request for one append-only review decision."""

    SCHEMA: ClassVar[str] = "dithyramba.review_decision_request/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    target_type: ReviewTargetType
    target_id: str
    target_hash: str
    action: ReviewAction
    reason: str
    authority: str
    scope: ReviewScope
    supersedes_review_decision_id: str | None = None

    @field_validator("target_id")
    @classmethod
    def _target_id(cls, value: str) -> str:
        # Type-dependent validation is repeated after all fields are available.
        if type(value) is not str or not value:
            raise ReviewContractError("target_id must be nonblank")
        return value

    @field_validator("target_hash")
    @classmethod
    def _target_hash(cls, value: str) -> str:
        if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
            raise ReviewContractError("target_hash must be a lowercase SHA-256 value")
        return value

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str) -> str:
        return _normalized_text(value, "reason", maximum=4_000)

    @field_validator("authority")
    @classmethod
    def _authority(cls, value: str) -> str:
        return _normalized_text(value, "authority", maximum=200)

    @field_validator("scope")
    @classmethod
    def _scope(cls, value: ReviewScope) -> ReviewScope:
        if type(value) is not ReviewScope:
            raise ReviewContractError("scope must be an exact ReviewScope value")
        return value

    @field_validator("supersedes_review_decision_id")
    @classmethod
    def _supersedes_id(cls, value: str | None) -> str | None:
        return None if value is None else _require_id(value, "review")

    @model_validator(mode="after")
    def _target_and_supersede_contract(self) -> Self:
        prefix = {
            ReviewTargetType.EVIDENCE_PACKET: "packet",
            ReviewTargetType.PACKET_ITEM: "packet_item",
            ReviewTargetType.SOURCE_FRAGMENT: "fragment",
        }[self.target_type]
        _require_id(self.target_id, prefix)
        if self.action is ReviewAction.SUPERSEDE:
            if self.supersedes_review_decision_id is None:
                raise ReviewContractError("supersede requires a prior ReviewDecision ID")
        elif self.supersedes_review_decision_id is not None:
            raise ReviewContractError("only supersede may reference a prior ReviewDecision")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "target_type": self.target_type.value,
            "target_id": self.target_id,
            "target_hash": self.target_hash,
            "action": self.action.value,
            "reason": self.reason,
            "authority": self.authority,
            "scope": self.scope.semantic_payload(),
            "supersedes_review_decision_id": self.supersedes_review_decision_id,
        }


class ReviewDecision(BaseModel):
    """Persisted human decision; identity/time are explicit audit fields."""

    SCHEMA: ClassVar[str] = "dithyramba.review_decision/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    review_decision_id: str
    library_id: str
    target_type: ReviewTargetType
    target_id: str
    target_hash: str
    action: ReviewAction
    reason: str
    authority: str
    scope: ReviewScope
    supersedes_review_decision_id: str | None = None
    created_at: str

    @model_validator(mode="before")
    @classmethod
    def _validate_request_fields(cls, value: object) -> object:
        if type(value) is not dict:
            return value
        request_fields = {
            key: value.get(key)
            for key in (
                "target_type",
                "target_id",
                "target_hash",
                "action",
                "reason",
                "authority",
                "scope",
                "supersedes_review_decision_id",
            )
        }
        ReviewDecisionRequest.model_validate(request_fields)
        return value

    @field_validator("review_decision_id")
    @classmethod
    def _decision_id(cls, value: str) -> str:
        return _require_id(value, "review")

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("created_at")
    @classmethod
    def _created_at(cls, value: str) -> str:
        if type(value) is not str or _TIMESTAMP_PATTERN.fullmatch(value) is None:
            raise ReviewContractError("created_at must be canonical UTC text")
        return value

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "review_decision_id": self.review_decision_id,
            "library_id": self.library_id,
            "target_type": self.target_type.value,
            "target_id": self.target_id,
            "target_hash": self.target_hash,
            "action": self.action.value,
            "reason": self.reason,
            "authority": self.authority,
            "scope": self.scope.semantic_payload(),
            "supersedes_review_decision_id": self.supersedes_review_decision_id,
            "created_at": self.created_at,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())


def _normalized_text(value: str, label: str, *, maximum: int) -> str:
    if type(value) is not str:
        raise ReviewContractError(f"{label} must be text")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized.strip() or normalized != normalized.strip() or "\x00" in normalized:
        raise ReviewContractError(f"{label} must be nonblank, unpadded NFC text without NUL")
    if len(normalized) > maximum:
        raise ReviewContractError(f"{label} must contain at most {maximum} code points")
    return normalized


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise ReviewContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise ReviewContractError(
            f"identifier must use the {expected} prefix with a canonical non-empty suffix"
        )
    return value
