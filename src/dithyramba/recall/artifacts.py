"""Pure content-addressed receipts, coverage, and EvidencePacket artifacts."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)
from dithyramba.ingest import (
    SOURCE_ADDRESS_SCHEMA,
    MarkdownSourceAddress,
    PdfSourceAddress,
    SourceAddress,
)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_VERSION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,127}$")
_SCORE_PATTERN = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{6}$")


class ArtifactContractError(ValueError):
    """A recall artifact violates its frozen canonical contract."""


class _FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ReadReceiptItem(_FrozenContract):
    """One exact SourceFragment materialized through the protected read boundary."""

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
        return _require_hash(value, "text_sha256")

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_version_id": self.source_version_id,
            "read_order": self.read_order,
            "text_sha256": self.text_sha256,
        }


class ReadReceipt(_FrozenContract):
    """Canonical list of fragment texts actually read, including FTS population."""

    SCHEMA: ClassVar[str] = "dithyramba.read_receipt/1.0"
    query_request_hash: str
    retrieval_corpus_hash: str
    items: tuple[ReadReceiptItem, ...] = ()

    @field_validator("query_request_hash", "retrieval_corpus_hash")
    @classmethod
    def _hashes(cls, value: str, info: Any) -> str:
        return _require_hash(value, info.field_name)

    @field_validator("items")
    @classmethod
    def _items(cls, value: tuple[ReadReceiptItem, ...]) -> tuple[ReadReceiptItem, ...]:
        if any(type(item) is not ReadReceiptItem for item in value):
            raise ArtifactContractError("ReadReceipt items must be ReadReceiptItem values")
        fragment_ids = [item.source_fragment_id for item in value]
        orders = [item.read_order for item in value]
        if len(set(fragment_ids)) != len(fragment_ids) or len(set(orders)) != len(orders):
            raise ArtifactContractError("ReadReceipt fragment IDs and orders must be unique")
        ordered = tuple(sorted(value, key=lambda item: item.read_order))
        if tuple(item.read_order for item in ordered) != tuple(range(len(ordered))):
            raise ArtifactContractError("ReadReceipt orders must be contiguous from zero")
        return ordered

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "query_request_hash": self.query_request_hash,
            "retrieval_corpus_hash": self.retrieval_corpus_hash,
            "items": [item.payload() for item in self.items],
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def read_receipt_id(self) -> str:
        return canonical_content_id("read", self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


class AccessReceipt(_FrozenContract):
    """Exact policy/snapshot/exclusion tuple that authorized the retrieval corpus."""

    SCHEMA: ClassVar[str] = "dithyramba.access_receipt/1.0"

    library_id: str
    query_request_hash: str
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

    @field_validator("access_policy_id")
    @classmethod
    def _policy_id(cls, value: str) -> str:
        return _require_id(value, "policy")

    @field_validator("corpus_snapshot_id")
    @classmethod
    def _snapshot_id(cls, value: str) -> str:
        return _require_id(value, "snapshot")

    @field_validator(
        "policy_hash",
        "query_request_hash",
        "snapshot_hash",
        "exclusion_hash",
        "permitted_set_hash",
        "retrieval_corpus_hash",
    )
    @classmethod
    def _hashes(cls, value: str, info: Any) -> str:
        return _require_hash(value, info.field_name)

    @model_validator(mode="after")
    def _snapshot_id_matches_hash(self) -> Self:
        if self.corpus_snapshot_id != _content_id_from_hash("snapshot", self.snapshot_hash):
            raise ArtifactContractError("AccessReceipt snapshot ID/hash tuple is inconsistent")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "query_request_hash": self.query_request_hash,
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
    def access_receipt_id(self) -> str:
        return canonical_content_id("access", self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


class RetrievalTraceItem(_FrozenContract):
    """One ranked FTS candidate with a monotonic quantized score."""

    source_fragment_id: str
    rank: int = Field(ge=1)
    score: str

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        return _require_id(value, "fragment")

    @field_validator("score")
    @classmethod
    def _score(cls, value: str) -> str:
        return _require_score(value)

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "rank": self.rank,
            "score": self.score,
        }


class RetrievalReceipt(_FrozenContract):
    """Deterministic permitted-only lexical retrieval trace."""

    SCHEMA: ClassVar[str] = "dithyramba.retrieval_receipt/1.0"

    profile: Literal["fts_v1"] = "fts_v1"
    query_request_hash: str
    profile_version: str
    code_version: str
    retrieval_corpus_hash: str
    max_candidates: int = Field(ge=1, le=500)
    max_source_fragments: int = Field(ge=1, le=100)
    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    trace: tuple[RetrievalTraceItem, ...] = ()

    @field_validator("profile_version", "code_version")
    @classmethod
    def _versions(cls, value: str) -> str:
        if type(value) is not str or _VERSION_PATTERN.fullmatch(value) is None:
            raise ArtifactContractError("profile/code version is not canonical")
        return value

    @field_validator("query_request_hash", "retrieval_corpus_hash")
    @classmethod
    def _hashes(cls, value: str, info: Any) -> str:
        return _require_hash(value, info.field_name)

    @field_validator("trace")
    @classmethod
    def _trace(cls, value: tuple[RetrievalTraceItem, ...]) -> tuple[RetrievalTraceItem, ...]:
        if any(type(item) is not RetrievalTraceItem for item in value):
            raise ArtifactContractError("retrieval trace must contain RetrievalTraceItem values")
        fragment_ids = [item.source_fragment_id for item in value]
        if len(set(fragment_ids)) != len(fragment_ids):
            raise ArtifactContractError("retrieval trace fragment IDs must be unique")
        ordered = tuple(sorted(value, key=lambda item: item.rank))
        if tuple(item.rank for item in ordered) != tuple(range(1, len(ordered) + 1)):
            raise ArtifactContractError("retrieval ranks must be contiguous from one")
        # The persisted score is a six-decimal presentation of SQLite's raw bm25
        # value. Distinct raw scores can quantize to the same string, so an ID
        # comparison after serialization would invent ordering information that
        # the receipt no longer carries. The FTS boundary owns the raw-score/ID
        # tie-break; this contract can only prove monotonic serialized scores.
        scores = tuple(Decimal(item.score) for item in ordered)
        if scores != tuple(sorted(scores)):
            raise ArtifactContractError("retrieval order must be bm25 ASC")
        return ordered

    @model_validator(mode="after")
    def _counts_match_trace(self) -> Self:
        if self.max_source_fragments > self.max_candidates:
            raise ArtifactContractError("max_source_fragments must not exceed max_candidates")
        if len(self.trace) > self.max_candidates:
            raise ArtifactContractError("retrieval trace exceeds max_candidates")
        if not (self.selected_count <= len(self.trace) <= self.candidate_count):
            raise ArtifactContractError(
                "retrieval counts must satisfy selected_count <= trace <= candidate_count"
            )
        if self.selected_count > self.max_source_fragments:
            raise ArtifactContractError("selected_count exceeds max_source_fragments")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "profile": self.profile,
            "query_request_hash": self.query_request_hash,
            "profile_version": self.profile_version,
            "code_version": self.code_version,
            "retrieval_corpus_hash": self.retrieval_corpus_hash,
            "max_candidates": self.max_candidates,
            "max_source_fragments": self.max_source_fragments,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "trace": [item.payload() for item in self.trace],
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def retrieval_receipt_id(self) -> str:
        return canonical_content_id("retrieval", self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())


class OmissionCategory(StrEnum):
    EXPLICIT_EXCLUSION = "explicit_exclusion"
    BUDGET = "budget"
    UNSUPPORTED = "unsupported"
    PARSER_FAILURE = "parser_failure"
    POLICY = "policy"


class OmissionDisclosure(StrEnum):
    COUNTED = "counted"
    REDACTED = "redacted"


class RecallOmission(_FrozenContract):
    """One bounded omission; policy omissions never expose a count."""

    category: OmissionCategory
    disclosure: OmissionDisclosure
    count: int | None = Field(default=None, ge=0)
    reason_code: str

    @field_validator("reason_code")
    @classmethod
    def _reason(cls, value: str) -> str:
        if type(value) is not str or _CODE_PATTERN.fullmatch(value) is None:
            raise ArtifactContractError("omission reason_code must use stable snake_case")
        return value

    @model_validator(mode="after")
    def _disclosure_matches_category(self) -> Self:
        if self.category is OmissionCategory.POLICY:
            if self.disclosure is not OmissionDisclosure.REDACTED or self.count is not None:
                raise ArtifactContractError("policy omission must be redacted without a count")
        elif self.disclosure is not OmissionDisclosure.COUNTED or self.count is None:
            raise ArtifactContractError("non-policy omission must have a counted value")
        return self

    def payload(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "disclosure": self.disclosure.value,
            "count": self.count,
            "reason_code": self.reason_code,
        }


class RecallCoverageReport(_FrozenContract):
    """Semantic recall coverage with no ProcessingRun identity or timestamps."""

    SCHEMA: ClassVar[str] = "dithyramba.coverage_report/1.0"

    query_request_hash: str
    processed_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    policy_omission_present: bool = False
    omissions: tuple[RecallOmission, ...] = ()

    @field_validator("query_request_hash")
    @classmethod
    def _query_hash(cls, value: str) -> str:
        return _require_hash(value, "query_request_hash")

    @field_validator("omissions")
    @classmethod
    def _omissions(cls, value: tuple[RecallOmission, ...]) -> tuple[RecallOmission, ...]:
        if any(type(item) is not RecallOmission for item in value):
            raise ArtifactContractError("coverage omissions must be RecallOmission values")
        keys = [(item.category.value, item.reason_code) for item in value]
        if len(set(keys)) != len(keys):
            raise ArtifactContractError("coverage omission category/reason pairs must be unique")
        return tuple(sorted(value, key=lambda item: (item.category.value, item.reason_code)))

    @model_validator(mode="after")
    def _policy_presence_matches_omissions(self) -> Self:
        if self.skipped_count != 0 or self.failed_count != 0:
            raise ArtifactContractError(
                "VS0 recall CoverageReport requires skipped_count and failed_count to be zero"
            )
        has_policy = any(item.category is OmissionCategory.POLICY for item in self.omissions)
        if self.policy_omission_present != has_policy:
            raise ArtifactContractError("policy omission presence must match redacted omission")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "stage": "recall",
            "query_request_hash": self.query_request_hash,
            "processed_count": self.processed_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "policy_omission": {"present": self.policy_omission_present, "count": None},
            "omissions": [item.payload() for item in self.omissions],
        }

    @property
    def report_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def coverage_report_id(self) -> str:
        return canonical_content_id("coverage", self.semantic_payload())

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    def packet_payload(self) -> dict[str, object]:
        return {
            "coverage_report_id": self.coverage_report_id,
            "coverage_report_hash": self.report_hash,
            **self.semantic_payload(),
        }


class EvidenceFragment(_FrozenContract):
    """One exact stored fragment selected into an EvidencePacket."""

    source_fragment_id: str
    source_version_id: str
    source_family_id: str | None
    rank: int = Field(ge=1)
    score: str
    text: str
    text_sha256: str
    source_address: SourceAddress

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        return _require_id(value, "fragment")

    @field_validator("source_version_id")
    @classmethod
    def _version_id(cls, value: str) -> str:
        return _require_id(value, "source_version")

    @field_validator("source_family_id")
    @classmethod
    def _family_id(cls, value: str | None) -> str | None:
        return None if value is None else _require_id(value, "family")

    @field_validator("score")
    @classmethod
    def _score(cls, value: str) -> str:
        return _require_score(value)

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFC", value)
        if not normalized.strip() or "\x00" in normalized:
            raise ArtifactContractError("EvidenceFragment text must be nonblank NFC text")
        return normalized

    @field_validator("text_sha256")
    @classmethod
    def _text_hash(cls, value: str) -> str:
        return _require_hash(value, "text_sha256")

    @field_validator("source_address", mode="before")
    @classmethod
    def _address(cls, value: object) -> SourceAddress:
        return _validated_source_address(value)

    @model_validator(mode="after")
    def _text_matches_hash_and_address(self) -> Self:
        if sha256_hex(self.text.encode("utf-8")) != self.text_sha256:
            raise ArtifactContractError("EvidenceFragment text_sha256 does not match text")
        if self.source_address.char_end - self.source_address.char_start != len(self.text):
            raise ArtifactContractError("SourceAddress span length does not match fragment text")
        return self

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_version_id": self.source_version_id,
            "source_family_id": self.source_family_id,
            "rank": self.rank,
            "score": self.score,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "source_address": self.source_address.payload(),
        }


class PacketResultStatus(StrEnum):
    EVIDENCE_FOUND = "evidence_found"
    NO_EVIDENCE = "no_evidence"


class EvidencePacket(_FrozenContract):
    """Bounded reproducible VS0 packet with only receipt hashes in its hash domain."""

    SCHEMA: ClassVar[str] = "dithyramba.evidence_packet/1.0"

    query_request_id: str
    query_request_hash: str
    corpus_snapshot_id: str
    corpus_snapshot_hash: str
    result_status: PacketResultStatus
    source_fragments: tuple[EvidenceFragment, ...]
    coverage_report: RecallCoverageReport
    read_receipt: ReadReceipt
    access_receipt: AccessReceipt
    retrieval_receipt: RetrievalReceipt

    @field_validator("query_request_id")
    @classmethod
    def _query_id(cls, value: str) -> str:
        return _require_id(value, "query")

    @field_validator("corpus_snapshot_id")
    @classmethod
    def _snapshot_id(cls, value: str) -> str:
        return _require_id(value, "snapshot")

    @field_validator("query_request_hash", "corpus_snapshot_hash")
    @classmethod
    def _hashes(cls, value: str, info: Any) -> str:
        return _require_hash(value, info.field_name)

    @field_validator("source_fragments")
    @classmethod
    def _fragments(cls, value: tuple[EvidenceFragment, ...]) -> tuple[EvidenceFragment, ...]:
        if any(type(item) is not EvidenceFragment for item in value):
            raise ArtifactContractError("packet source_fragments must be EvidenceFragment values")
        fragment_ids = [item.source_fragment_id for item in value]
        if len(set(fragment_ids)) != len(fragment_ids):
            raise ArtifactContractError("packet fragment IDs must be unique")
        ordered = tuple(sorted(value, key=lambda item: item.rank))
        if tuple(item.rank for item in ordered) != tuple(range(1, len(ordered) + 1)):
            raise ArtifactContractError("packet fragment ranks must be contiguous from one")
        return ordered

    @field_validator("coverage_report")
    @classmethod
    def _coverage_report(cls, value: RecallCoverageReport) -> RecallCoverageReport:
        if type(value) is not RecallCoverageReport:
            raise ArtifactContractError("coverage_report must be an exact RecallCoverageReport")
        return value

    @field_validator("read_receipt")
    @classmethod
    def _read_receipt(cls, value: ReadReceipt) -> ReadReceipt:
        if type(value) is not ReadReceipt:
            raise ArtifactContractError("read_receipt must be an exact ReadReceipt")
        return value

    @field_validator("access_receipt")
    @classmethod
    def _access_receipt(cls, value: AccessReceipt) -> AccessReceipt:
        if type(value) is not AccessReceipt:
            raise ArtifactContractError("access_receipt must be an exact AccessReceipt")
        return value

    @field_validator("retrieval_receipt")
    @classmethod
    def _retrieval_receipt(cls, value: RetrievalReceipt) -> RetrievalReceipt:
        if type(value) is not RetrievalReceipt:
            raise ArtifactContractError("retrieval_receipt must be an exact RetrievalReceipt")
        return value

    @model_validator(mode="after")
    def _dependencies_are_consistent(self) -> Self:
        has_evidence = bool(self.source_fragments)
        if has_evidence != (self.result_status is PacketResultStatus.EVIDENCE_FOUND):
            raise ArtifactContractError("packet result_status must match selected evidence")
        if len(self.source_fragments) != self.retrieval_receipt.selected_count:
            raise ArtifactContractError("packet items must equal retrieval selected_count")
        if self.query_request_id != _content_id_from_hash("query", self.query_request_hash):
            raise ArtifactContractError("packet QueryRequest ID/hash tuple is inconsistent")
        if self.corpus_snapshot_id != _content_id_from_hash("snapshot", self.corpus_snapshot_hash):
            raise ArtifactContractError("packet CorpusSnapshot ID/hash tuple is inconsistent")
        if self.read_receipt.query_request_hash != self.query_request_hash:
            raise ArtifactContractError("ReadReceipt references a different QueryRequest")
        if self.access_receipt.query_request_hash != self.query_request_hash:
            raise ArtifactContractError("AccessReceipt references a different QueryRequest")
        if self.retrieval_receipt.query_request_hash != self.query_request_hash:
            raise ArtifactContractError("RetrievalReceipt references a different QueryRequest")
        if self.coverage_report.query_request_hash != self.query_request_hash:
            raise ArtifactContractError("CoverageReport references a different QueryRequest")
        if self.access_receipt.corpus_snapshot_id != self.corpus_snapshot_id:
            raise ArtifactContractError("AccessReceipt references a different CorpusSnapshot")
        if self.access_receipt.snapshot_hash != self.corpus_snapshot_hash:
            raise ArtifactContractError("AccessReceipt snapshot hash differs from packet")
        if (
            self.access_receipt.retrieval_corpus_hash
            != self.retrieval_receipt.retrieval_corpus_hash
        ):
            raise ArtifactContractError("receipt retrieval corpus hashes differ")
        if self.read_receipt.retrieval_corpus_hash != self.retrieval_receipt.retrieval_corpus_hash:
            raise ArtifactContractError("ReadReceipt retrieval corpus hash differs")
        if (
            self.access_receipt.policy_omission_present
            != self.coverage_report.policy_omission_present
        ):
            raise ArtifactContractError(
                "AccessReceipt and CoverageReport policy omission disclosures differ"
            )
        read_versions = {item.source_version_id for item in self.read_receipt.items}
        if self.coverage_report.processed_count != len(read_versions):
            raise ArtifactContractError(
                "CoverageReport processed_count must equal unique ReadReceipt SourceVersions"
            )
        selected_trace = self.retrieval_receipt.trace[: self.retrieval_receipt.selected_count]
        selected = tuple(
            (item.source_fragment_id, item.rank, item.score) for item in self.source_fragments
        )
        traced = tuple((item.source_fragment_id, item.rank, item.score) for item in selected_trace)
        if selected != traced:
            raise ArtifactContractError(
                "packet evidence must equal selected retrieval trace prefix"
            )
        read_items = {item.source_fragment_id: item for item in self.read_receipt.items}
        for fragment in self.source_fragments:
            read_item = read_items.get(fragment.source_fragment_id)
            if read_item is None:
                raise ArtifactContractError(
                    "packet evidence contains a fragment absent from ReadReceipt"
                )
            if (
                fragment.source_version_id != read_item.source_version_id
                or fragment.text_sha256 != read_item.text_sha256
            ):
                raise ArtifactContractError(
                    "packet evidence version/text hash differs from ReadReceipt"
                )
        read_ids = set(read_items)
        trace_ids = {item.source_fragment_id for item in self.retrieval_receipt.trace}
        if not trace_ids.issubset(read_ids):
            raise ArtifactContractError(
                "retrieval trace contains a fragment absent from ReadReceipt"
            )
        return self

    def semantic_payload(self) -> dict[str, object]:
        """Return the packet hash domain, excluding only packet identity/presentation."""

        return {
            "schema": self.SCHEMA,
            "query_request_id": self.query_request_id,
            "query_request_hash": self.query_request_hash,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "corpus_snapshot_hash": self.corpus_snapshot_hash,
            "result_status": self.result_status.value,
            "source_fragments": [item.payload() for item in self.source_fragments],
            "counterevidence": {"status": "not_evaluated", "items": []},
            "evidence_gaps": {"status": "not_evaluated", "items": []},
            "coverage_report": self.coverage_report.packet_payload(),
            "receipts": {
                "read_receipt_id": self.read_receipt.read_receipt_id,
                "read_receipt_hash": self.read_receipt.receipt_hash,
                "access_receipt_id": self.access_receipt.access_receipt_id,
                "access_receipt_hash": self.access_receipt.receipt_hash,
                "retrieval_receipt_id": self.retrieval_receipt.retrieval_receipt_id,
                "retrieval_receipt_hash": self.retrieval_receipt.receipt_hash,
            },
        }

    @property
    def packet_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def evidence_packet_id(self) -> str:
        return canonical_content_id("packet", self.semantic_payload())

    def payload(self) -> dict[str, object]:
        return {
            **self.semantic_payload(),
            "evidence_packet_id": self.evidence_packet_id,
            "packet_hash": self.packet_hash,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())


def _validated_source_address(value: object) -> SourceAddress:
    if isinstance(value, (MarkdownSourceAddress, PdfSourceAddress)):
        if type(value) not in (MarkdownSourceAddress, PdfSourceAddress):
            raise ArtifactContractError("source_address must use an exact SourceAddress type")
        return value
    if type(value) is not dict:
        raise ArtifactContractError("source_address must be a SourceAddress payload")
    if value.get("schema") != SOURCE_ADDRESS_SCHEMA:
        raise ArtifactContractError("source_address schema is not supported")
    kind = value.get("kind")
    try:
        if kind == "markdown":
            expected = {
                "schema",
                "kind",
                "heading_path",
                "line_start",
                "line_end",
                "char_start",
                "char_end",
            }
            if set(value) != expected or type(value.get("heading_path")) is not list:
                raise ArtifactContractError("Markdown SourceAddress shape is invalid")
            address: SourceAddress = MarkdownSourceAddress(
                heading_path=tuple(value["heading_path"]),
                line_start=value["line_start"],
                line_end=value["line_end"],
                char_start=value["char_start"],
                char_end=value["char_end"],
            )
        elif kind == "pdf":
            expected = {"schema", "kind", "page", "bbox", "char_start", "char_end"}
            if set(value) != expected or type(value.get("bbox")) is not list:
                raise ArtifactContractError("PDF SourceAddress shape is invalid")
            bbox = value["bbox"]
            if len(bbox) != 4:
                raise ArtifactContractError("PDF SourceAddress bbox must have four points")
            address = PdfSourceAddress(
                page=value["page"],
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                char_start=value["char_start"],
                char_end=value["char_end"],
            )
        else:
            raise ArtifactContractError("source_address kind is not supported")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ArtifactContractError):
            raise
        raise ArtifactContractError("source_address payload is invalid") from error
    if canonical_json_bytes(address.payload()) != canonical_json_bytes(value):
        raise ArtifactContractError("source_address payload is not canonical")
    return address


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise ArtifactContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise ArtifactContractError(
            f"identifier must use the {expected} prefix with a canonical non-empty suffix"
        )
    return value


def _require_hash(value: str, label: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise ArtifactContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _content_id_from_hash(prefix: str, digest: str) -> str:
    return f"{prefix}_{digest[:32]}"


def _require_score(value: str) -> str:
    if type(value) is not str or _SCORE_PATTERN.fullmatch(value) is None:
        raise ArtifactContractError("score must be a fixed six-decimal string")
    try:
        decimal = Decimal(value)
    except InvalidOperation as error:
        raise ArtifactContractError("score must be a finite decimal string") from error
    if not decimal.is_finite() or (decimal == 0 and value != "0.000000"):
        raise ArtifactContractError("score must use canonical finite decimal form")
    return value
