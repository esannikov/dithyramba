"""Strict QueryRequest contracts for the frozen VS0 lexical recall profile."""

from __future__ import annotations

import re
import unicodedata
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.access import QueryExclusions
from dithyramba.contracts import canonical_content_id, canonical_json_bytes, canonical_sha256_hex

_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_PURPOSE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class QueryContractError(ValueError):
    """A QueryRequest value violates the frozen VS0 public contract."""


class RetrievalBudget(BaseModel):
    """Bounded FTS-only candidate and packet selection budget."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    profile: Literal["fts_v1"] = "fts_v1"
    max_candidates: int = Field(default=100, ge=1, le=500)
    max_source_fragments: int = Field(default=30, ge=1, le=100)

    @model_validator(mode="after")
    def _selection_fits_candidates(self) -> RetrievalBudget:
        if self.max_source_fragments > self.max_candidates:
            raise QueryContractError("max_source_fragments must not exceed max_candidates")
        return self

    def payload(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "max_candidates": self.max_candidates,
            "max_source_fragments": self.max_source_fragments,
        }


class EvidencePacketResultContract(BaseModel):
    """The sole VS0 result shape; extension requires a new schema version."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    format: Literal["evidence_packet"] = "evidence_packet"
    require_source_addresses: Literal[True] = True

    def payload(self) -> dict[str, object]:
        return {
            "format": self.format,
            "require_source_addresses": self.require_source_addresses,
        }


class QueryRequest(BaseModel):
    """One-library, explicit-scope, replayable VS0 recall request."""

    SCHEMA: ClassVar[str] = "dithyramba.query_request/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    question: str
    library_id: str
    collection_ids: tuple[str, ...]
    corpus_snapshot_id: str
    access_policy_id: str
    purpose: str
    exclusions: QueryExclusions = Field(default_factory=QueryExclusions)
    retrieval: RetrievalBudget = Field(default_factory=RetrievalBudget)
    result_contract: EvidencePacketResultContract = Field(
        default_factory=EvidencePacketResultContract
    )

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFC", value)
        if not normalized.strip() or normalized != normalized.strip() or "\x00" in normalized:
            raise QueryContractError("question must be nonblank, unpadded NFC text without NUL")
        if len(normalized) > 2_000:
            raise QueryContractError("question must contain at most 2,000 code points")
        return normalized

    @field_validator("library_id")
    @classmethod
    def _library_id(cls, value: str) -> str:
        return _require_id(value, "library")

    @field_validator("corpus_snapshot_id")
    @classmethod
    def _snapshot_id(cls, value: str) -> str:
        return _require_id(value, "snapshot")

    @field_validator("access_policy_id")
    @classmethod
    def _policy_id(cls, value: str) -> str:
        return _require_id(value, "policy")

    @field_validator("purpose")
    @classmethod
    def _purpose(cls, value: str) -> str:
        if type(value) is not str or _PURPOSE_PATTERN.fullmatch(value) is None:
            raise QueryContractError("purpose must match [a-z][a-z0-9_-]{0,63}")
        return value

    @field_validator("collection_ids")
    @classmethod
    def _collection_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 16:
            raise QueryContractError("QueryRequest requires 1-16 Collections")
        for collection_id in value:
            _require_id(collection_id, "collection")
        if len(set(value)) != len(value):
            raise QueryContractError("QueryRequest Collection IDs must be unique")
        return tuple(sorted(value))

    @field_validator("exclusions")
    @classmethod
    def _exclusions(cls, value: QueryExclusions) -> QueryExclusions:
        if type(value) is not QueryExclusions:
            raise QueryContractError("exclusions must be an explicit QueryExclusions value")
        for source_id in value.source_ids:
            _require_id(source_id, "source")
        for family_id in value.source_family_ids:
            _require_id(family_id, "family")
        for fragment_id in value.source_fragment_ids:
            _require_id(fragment_id, "fragment")
        return value

    @field_validator("retrieval")
    @classmethod
    def _retrieval(cls, value: RetrievalBudget) -> RetrievalBudget:
        if type(value) is not RetrievalBudget:
            raise QueryContractError("retrieval must be an exact RetrievalBudget value")
        return value

    @field_validator("result_contract")
    @classmethod
    def _result_contract(cls, value: EvidencePacketResultContract) -> EvidencePacketResultContract:
        if type(value) is not EvidencePacketResultContract:
            raise QueryContractError(
                "result_contract must be an exact EvidencePacketResultContract value"
            )
        return value

    def semantic_payload(self) -> dict[str, object]:
        """Return the exact public hash domain, with no run/time/provider fields."""

        return {
            "schema": self.SCHEMA,
            "question": self.question,
            "library_id": self.library_id,
            "collection_ids": list(self.collection_ids),
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "purpose": self.purpose,
            "exclusions": self.exclusions.payload(),
            "retrieval": self.retrieval.payload(),
            "result_contract": self.result_contract.payload(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def request_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def query_request_id(self) -> str:
        return canonical_content_id("query", self.semantic_payload())


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise QueryContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise QueryContractError(
            f"identifier must use the {expected} prefix with a canonical non-empty suffix"
        )
    return value
