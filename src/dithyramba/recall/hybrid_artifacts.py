"""Text-free, content-addressed support receipts for P7 hybrid recall."""

from __future__ import annotations

import re
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.contracts import canonical_content_id, canonical_json_bytes, canonical_sha256_hex

from .hybrid_models import HybridContractError

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class HybridArtifactError(HybridContractError):
    """A supporting P7 receipt violates its immutable public contract."""


class _FrozenArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class HybridReadReceiptItem(_FrozenArtifact):
    """One exact permitted fragment whose text was materialized for the run."""

    source_fragment_id: str
    source_version_id: str
    read_order: int = Field(ge=0)
    text_sha256: str

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        return _require_id(value, "fragment")

    @field_validator("source_version_id")
    @classmethod
    def _version_id(cls, value: str) -> str:
        return _require_id(value, "source_version")

    @field_validator("text_sha256")
    @classmethod
    def _text_hash(cls, value: str) -> str:
        return _require_hash(value)

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_version_id": self.source_version_id,
            "read_order": self.read_order,
            "text_sha256": self.text_sha256,
        }


class HybridReadReceipt(_FrozenArtifact):
    """Canonical order and identity of every fragment text read by hybrid recall."""

    SCHEMA: ClassVar[str] = "dithyramba.hybrid_read_receipt/1.0"

    request_id: str
    request_hash: str
    retrieval_corpus_hash: str
    items: tuple[HybridReadReceiptItem, ...] = ()

    @field_validator("request_id")
    @classmethod
    def _request_id(cls, value: str) -> str:
        return _require_id(value, "hybrid_query")

    @field_validator("request_hash", "retrieval_corpus_hash")
    @classmethod
    def _hashes(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("items")
    @classmethod
    def _items(
        cls,
        value: tuple[HybridReadReceiptItem, ...],
    ) -> tuple[HybridReadReceiptItem, ...]:
        if any(type(item) is not HybridReadReceiptItem for item in value):
            raise HybridArtifactError("hybrid read items must use the exact contract")
        ordered = tuple(sorted(value, key=lambda item: item.read_order))
        if tuple(item.read_order for item in ordered) != tuple(range(len(ordered))):
            raise HybridArtifactError("hybrid read order must be contiguous from zero")
        identifiers = tuple(item.source_fragment_id for item in ordered)
        if len(set(identifiers)) != len(identifiers):
            raise HybridArtifactError("hybrid read fragment IDs must be unique")
        return ordered

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "retrieval_corpus_hash": self.retrieval_corpus_hash,
            "items": [item.payload() for item in self.items],
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def receipt_id(self) -> str:
        return canonical_content_id("hybrid_read", self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


class HybridAccessReceipt(_FrozenArtifact):
    """Exact policy tuple compiled before any text, vector, or model access."""

    SCHEMA: ClassVar[str] = "dithyramba.hybrid_access_receipt/1.0"

    library_id: str
    request_id: str
    request_hash: str
    access_policy_id: str
    policy_hash: str
    corpus_snapshot_id: str
    snapshot_hash: str
    exclusion_hash: str
    permitted_set_hash: str
    retrieval_corpus_hash: str
    policy_omission_present: bool = False

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("request_id")
    @classmethod
    def _request_id(cls, value: str) -> str:
        return _require_id(value, "hybrid_query")

    @field_validator("access_policy_id")
    @classmethod
    def _policy_id(cls, value: str) -> str:
        return _require_id(value, "policy")

    @field_validator("corpus_snapshot_id")
    @classmethod
    def _snapshot_id(cls, value: str) -> str:
        return _require_id(value, "snapshot")

    @field_validator(
        "request_hash",
        "policy_hash",
        "snapshot_hash",
        "exclusion_hash",
        "permitted_set_hash",
        "retrieval_corpus_hash",
    )
    @classmethod
    def _hashes(cls, value: str) -> str:
        return _require_hash(value)

    @model_validator(mode="after")
    def _snapshot_binding(self) -> Self:
        if self.corpus_snapshot_id != f"snapshot_{self.snapshot_hash[:32]}":
            raise HybridArtifactError("hybrid access snapshot ID/hash tuple is inconsistent")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "access_policy_id": self.access_policy_id,
            "policy_hash": self.policy_hash,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "exclusion_hash": self.exclusion_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "retrieval_corpus_hash": self.retrieval_corpus_hash,
            "policy_omission_present": self.policy_omission_present,
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def receipt_id(self) -> str:
        return canonical_content_id("hybrid_access", self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


class HybridCoverageReport(_FrozenArtifact):
    """Bounded public coverage; policy-denied identities and counts never enter it."""

    SCHEMA: ClassVar[str] = "dithyramba.hybrid_coverage_report/1.0"

    request_id: str
    request_hash: str
    processed_count: int = Field(ge=0)
    skipped_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    policy_omission_present: bool = False

    @field_validator("request_id")
    @classmethod
    def _request_id(cls, value: str) -> str:
        return _require_id(value, "hybrid_query")

    @field_validator("request_hash")
    @classmethod
    def _request_hash(cls, value: str) -> str:
        return _require_hash(value)

    @model_validator(mode="after")
    def _recall_has_no_hidden_failures(self) -> Self:
        if self.skipped_count != 0 or self.failed_count != 0:
            raise HybridArtifactError(
                "successful hybrid recall cannot hide skipped or failed permitted fragments"
            )
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "stage": "hybrid_recall",
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "processed_count": self.processed_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "policy_omission": {"present": self.policy_omission_present, "count": None},
        }

    @property
    def report_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def report_id(self) -> str:
        return canonical_content_id("hybrid_coverage", self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


def _require_hash(value: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise HybridArtifactError("artifact hash must be lowercase SHA-256 hex")
    return value


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise HybridArtifactError(f"identifier must use the {expected} prefix")
    if _ID_SUFFIX_PATTERN.fullmatch(value[len(expected) :]) is None:
        raise HybridArtifactError(f"identifier must use a canonical {expected} suffix")
    return value
