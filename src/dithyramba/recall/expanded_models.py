"""Pure, versioned quality-first expansion contracts over frozen P7 recall.

These models make no persistence or execution claims.  They only describe the
exact request, candidate closure, omissions, provenance and selected manifest
that an expansion engine must later verify and persist.
"""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.contracts import canonical_content_id, canonical_json_bytes, canonical_sha256_hex
from dithyramba.relations import RelationNodeType, RelationType

from .hybrid_models import CorpusLayer, HybridContractError

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_MAX_CANDIDATES = 500
_MAX_FINAL = 100
_MAX_GROUNDING_FRAGMENTS = 2_000
_MAX_GROUNDING_SOURCES = 500
_MAX_RECEIPT_HASHES = 2_000


class ExpandedContractError(HybridContractError):
    """A quality-first expansion artifact violates its public contract."""


class ExpandedCandidateKind(StrEnum):
    """Identity families admitted to expansion and final selection."""

    SOURCE_FRAGMENT = "source_fragment"
    STRUCTURE_UNIT = "structure_unit"
    RELATION_NODE = "relation_node"


class ExpansionOmissionReason(StrEnum):
    """Closed omission vocabulary; free-form rationales are deliberately absent."""

    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    UNGROUNDED = "ungrounded"
    LIMIT_EXCEEDED = "limit_exceeded"
    UNSUPPORTED = "unsupported"


class _ExpandedModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class GraphExpansionConfig(_ExpandedModel):
    """Presence means enabled; absence from the request means disabled."""

    SCHEMA: ClassVar[str] = "dithyramba.graph_expansion_config/1.0"

    enabled: Literal[True] = True
    relation_types: tuple[RelationType, ...]
    max_hops: int = Field(ge=1, le=3)

    @field_validator("relation_types")
    @classmethod
    def _relation_types(cls, value: tuple[RelationType, ...]) -> tuple[RelationType, ...]:
        if type(value) is not tuple or not value:
            raise ExpandedContractError(
                "enabled graph expansion requires at least one RelationType"
            )
        if any(not isinstance(item, RelationType) for item in value):
            raise ExpandedContractError("graph relation_types must contain RelationType values")
        ordered = tuple(sorted(set(value), key=lambda item: item.value.encode("ascii")))
        if len(ordered) != len(value):
            raise ExpandedContractError("graph relation_types must be unique")
        return ordered

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "enabled": self.enabled,
            "relation_types": [item.value for item in self.relation_types],
            "max_hops": self.max_hops,
        }


class ExpandedQueryRequest(_ExpandedModel):
    """A bounded expansion request bound to one exact HybridQueryRequest/1.0."""

    SCHEMA: ClassVar[str] = "dithyramba.expanded_query_request/1.0"

    base_request_id: str
    base_request_hash: str
    structure_generation_id: str | None = None
    structure_generation_hash: str | None = None
    reranker_profile_hash: str
    graph: GraphExpansionConfig | None = None
    max_final: int = Field(default=20, ge=1, le=_MAX_FINAL)
    result_format: Literal["expanded_evidence_packet"] = "expanded_evidence_packet"

    @field_validator("base_request_id")
    @classmethod
    def _base_request_id(cls, value: str) -> str:
        return _require_id(value, "hybrid_query")

    @field_validator("structure_generation_id")
    @classmethod
    def _structure_generation_id(cls, value: str | None) -> str | None:
        return None if value is None else _require_id(value, "structure_generation")

    @field_validator(
        "base_request_hash",
        "structure_generation_hash",
        "reranker_profile_hash",
    )
    @classmethod
    def _hash(cls, value: str | None) -> str | None:
        return None if value is None else _require_hash(value)

    @field_validator("graph")
    @classmethod
    def _graph(cls, value: GraphExpansionConfig | None) -> GraphExpansionConfig | None:
        if value is not None and type(value) is not GraphExpansionConfig:
            raise ExpandedContractError("graph must be an exact GraphExpansionConfig")
        return value

    @model_validator(mode="after")
    def _structure_binding_is_complete(self) -> ExpandedQueryRequest:
        if (self.structure_generation_id is None) != (self.structure_generation_hash is None):
            raise ExpandedContractError(
                "structure generation ID and hash must be supplied together"
            )
        return self

    @property
    def structure_enabled(self) -> bool:
        return self.structure_generation_id is not None

    @property
    def graph_enabled(self) -> bool:
        return self.graph is not None

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "base_request_id": self.base_request_id,
            "base_request_hash": self.base_request_hash,
            "structure_generation_id": self.structure_generation_id,
            "structure_generation_hash": self.structure_generation_hash,
            "reranker_profile_hash": self.reranker_profile_hash,
            "graph": None if self.graph is None else self.graph.payload(),
            "max_final": self.max_final,
            "result_format": self.result_format,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def request_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def request_id(self) -> str:
        return canonical_content_id("expanded_query", self.semantic_payload())


class _CandidateIdentity(_ExpandedModel):
    candidate_kind: ExpandedCandidateKind
    candidate_id: str
    relation_node_type: RelationNodeType | None = None

    @model_validator(mode="after")
    def _candidate_prefix_matches_kind(self) -> _CandidateIdentity:
        if not isinstance(self.candidate_kind, ExpandedCandidateKind):
            raise ExpandedContractError("candidate_kind must be an ExpandedCandidateKind")
        if self.candidate_kind is ExpandedCandidateKind.SOURCE_FRAGMENT:
            _require_id(self.candidate_id, "fragment")
            if self.relation_node_type is not None:
                raise ExpandedContractError("source_fragment candidate cannot declare a node type")
        elif self.candidate_kind is ExpandedCandidateKind.STRUCTURE_UNIT:
            _require_id(self.candidate_id, "structure_unit")
            if self.relation_node_type is not None:
                raise ExpandedContractError("structure_unit candidate cannot declare a node type")
        else:
            if not isinstance(self.relation_node_type, RelationNodeType):
                raise ExpandedContractError("relation_node candidate requires its exact node type")
            _require_id(self.candidate_id, self.relation_node_type.value)
        return self

    @property
    def candidate_key(self) -> tuple[str, str, str]:
        return (
            self.candidate_kind.value,
            "" if self.relation_node_type is None else self.relation_node_type.value,
            self.candidate_id,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "candidate_kind": self.candidate_kind.value,
            "candidate_id": self.candidate_id,
            "relation_node_type": (
                None if self.relation_node_type is None else self.relation_node_type.value
            ),
        }


class ExpansionSeedRank(_ExpandedModel):
    """One exact atomic seed rank from the frozen base packet."""

    source_fragment_id: str
    atomic_rank: int = Field(ge=1, le=_MAX_CANDIDATES)

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        return _require_id(value, "fragment")

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "atomic_rank": self.atomic_rank,
        }


class ExpandedCandidateTrace(_CandidateIdentity):
    """Text-free exact provenance and score for one eligible or reranked candidate."""

    rank: int = Field(ge=1, le=_MAX_CANDIDATES)
    source_fragment_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    source_family_ids: tuple[str, ...]
    corpus_layers: tuple[CorpusLayer, ...]
    anchor_atomic_ranks: tuple[int, ...]
    text_sha256: str
    pair_token_count: int = Field(ge=1, le=32_768)
    score_hex: str

    @field_validator("source_fragment_ids")
    @classmethod
    def _fragment_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_ids(
            value,
            prefix="fragment",
            field="source_fragment_ids",
            maximum=_MAX_GROUNDING_FRAGMENTS,
            non_empty=True,
        )

    @field_validator("source_ids")
    @classmethod
    def _source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_ids(
            value,
            prefix="source",
            field="source_ids",
            maximum=_MAX_GROUNDING_SOURCES,
            non_empty=True,
        )

    @field_validator("source_family_ids")
    @classmethod
    def _family_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_ids(
            value,
            prefix="family",
            field="source_family_ids",
            maximum=_MAX_GROUNDING_SOURCES,
            non_empty=True,
        )

    @field_validator("corpus_layers")
    @classmethod
    def _layers(cls, value: tuple[CorpusLayer, ...]) -> tuple[CorpusLayer, ...]:
        if type(value) is not tuple or not value:
            raise ExpandedContractError("corpus_layers must be a non-empty tuple")
        if any(not isinstance(item, CorpusLayer) for item in value):
            raise ExpandedContractError("corpus_layers must contain CorpusLayer values")
        ordered = tuple(sorted(set(value), key=lambda item: item.value.encode("ascii")))
        if len(ordered) != len(value):
            raise ExpandedContractError("corpus_layers must be unique")
        return ordered

    @field_validator("anchor_atomic_ranks")
    @classmethod
    def _anchor_ranks(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if (
            type(value) is not tuple
            or not value
            or len(value) > _MAX_CANDIDATES
            or any(type(item) is not int or not 1 <= item <= _MAX_CANDIDATES for item in value)
        ):
            raise ExpandedContractError("anchor_atomic_ranks must contain 1-500 exact ranks")
        ordered = tuple(sorted(value))
        if len(set(ordered)) != len(ordered):
            raise ExpandedContractError("anchor_atomic_ranks must be unique")
        return ordered

    @field_validator("text_sha256")
    @classmethod
    def _text_hash(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("score_hex")
    @classmethod
    def _score(cls, value: str) -> str:
        if type(value) is not str:
            raise ExpandedContractError("candidate score must use canonical float.hex text")
        try:
            score = float.fromhex(value)
        except ValueError as error:
            raise ExpandedContractError(
                "candidate score must use canonical float.hex text"
            ) from error
        if not math.isfinite(score) or score.hex() != value:
            raise ExpandedContractError("candidate score must use canonical finite float.hex text")
        return value

    @model_validator(mode="after")
    def _grounding_matches_kind(self) -> ExpandedCandidateTrace:
        if self.candidate_kind is ExpandedCandidateKind.SOURCE_FRAGMENT:
            if self.source_fragment_ids != (self.candidate_id,):
                raise ExpandedContractError(
                    "source_fragment candidate must ground itself and only itself"
                )
            if not (
                len(self.source_ids) == len(self.source_family_ids) == len(self.corpus_layers) == 1
            ):
                raise ExpandedContractError(
                    "source_fragment candidate requires one Source, family and layer"
                )
        elif self.candidate_kind is ExpandedCandidateKind.STRUCTURE_UNIT:
            if len(self.source_fragment_ids) > 128:
                raise ExpandedContractError(
                    "structure_unit candidate may contain at most 128 SourceFragments"
                )
            if not (
                len(self.source_ids) == len(self.source_family_ids) == len(self.corpus_layers) == 1
            ):
                raise ExpandedContractError(
                    "structure_unit candidate requires one Source, family and layer"
                )
        return self

    def grounding_payload(self) -> dict[str, object]:
        """Candidate identity, exact grounding and atomic anchors without text."""

        return self.identity_payload() | {
            "source_fragment_ids": list(self.source_fragment_ids),
            "source_ids": list(self.source_ids),
            "source_family_ids": list(self.source_family_ids),
            "corpus_layers": [item.value for item in self.corpus_layers],
            "anchor_atomic_ranks": list(self.anchor_atomic_ranks),
            "text_sha256": self.text_sha256,
        }

    def provenance_payload(self) -> dict[str, object]:
        """Candidate grounding plus token count, excluding ordering and score."""

        return self.grounding_payload() | {
            "pair_token_count": self.pair_token_count,
        }

    def payload(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            **self.provenance_payload(),
            "score_hex": self.score_hex,
        }


class RelationPathReceiptRef(_ExpandedModel):
    """Exact content-addressed RelationPathReceipt reference."""

    receipt_id: str
    receipt_hash: str

    @field_validator("receipt_id")
    @classmethod
    def _receipt_id(cls, value: str) -> str:
        return _require_id(value, "relation_path")

    @field_validator("receipt_hash")
    @classmethod
    def _receipt_hash(cls, value: str) -> str:
        return _require_hash(value)

    @model_validator(mode="after")
    def _id_is_derived_from_hash(self) -> RelationPathReceiptRef:
        if self.receipt_id != f"relation_path_{self.receipt_hash[:32]}":
            raise ExpandedContractError(
                "RelationPathReceipt ID must be derived from its receipt hash"
            )
        return self

    def payload(self) -> dict[str, str]:
        return {"receipt_id": self.receipt_id, "receipt_hash": self.receipt_hash}


class ExpansionPath(_CandidateIdentity):
    """Text-free lineage from atomic anchors to one exact expanded candidate."""

    SCHEMA: ClassVar[str] = "dithyramba.expansion_path/1.0"

    source_fragment_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    source_family_ids: tuple[str, ...]
    corpus_layers: tuple[CorpusLayer, ...]
    anchor_atomic_ranks: tuple[int, ...]
    text_sha256: str
    structure_read_receipt_id: str | None = None
    structure_read_receipt_hash: str | None = None
    relation_path_receipts: tuple[RelationPathReceiptRef, ...] = ()

    @field_validator("source_fragment_ids")
    @classmethod
    def _fragment_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_ids(
            value,
            prefix="fragment",
            field="source_fragment_ids",
            maximum=_MAX_GROUNDING_FRAGMENTS,
            non_empty=True,
        )

    @field_validator("source_ids")
    @classmethod
    def _source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_ids(
            value,
            prefix="source",
            field="source_ids",
            maximum=_MAX_GROUNDING_SOURCES,
            non_empty=True,
        )

    @field_validator("source_family_ids")
    @classmethod
    def _family_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_ids(
            value,
            prefix="family",
            field="source_family_ids",
            maximum=_MAX_GROUNDING_SOURCES,
            non_empty=True,
        )

    @field_validator("corpus_layers")
    @classmethod
    def _layers(cls, value: tuple[CorpusLayer, ...]) -> tuple[CorpusLayer, ...]:
        if type(value) is not tuple or not value:
            raise ExpandedContractError("corpus_layers must be a non-empty tuple")
        if any(not isinstance(item, CorpusLayer) for item in value):
            raise ExpandedContractError("corpus_layers must contain CorpusLayer values")
        ordered = tuple(sorted(set(value), key=lambda item: item.value.encode("ascii")))
        if len(ordered) != len(value):
            raise ExpandedContractError("corpus_layers must be unique")
        return ordered

    @field_validator("anchor_atomic_ranks")
    @classmethod
    def _anchor_ranks(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if (
            type(value) is not tuple
            or not value
            or len(value) > _MAX_CANDIDATES
            or any(type(item) is not int or not 1 <= item <= _MAX_CANDIDATES for item in value)
        ):
            raise ExpandedContractError("anchor_atomic_ranks must contain 1-500 exact ranks")
        ordered = tuple(sorted(value))
        if len(set(ordered)) != len(ordered):
            raise ExpandedContractError("anchor_atomic_ranks must be unique")
        return ordered

    @field_validator("text_sha256", "structure_read_receipt_hash")
    @classmethod
    def _hash(cls, value: str | None) -> str | None:
        return None if value is None else _require_hash(value)

    @field_validator("structure_read_receipt_id")
    @classmethod
    def _structure_receipt_id(cls, value: str | None) -> str | None:
        return None if value is None else _require_id(value, "structure_read")

    @field_validator("relation_path_receipts")
    @classmethod
    def _relation_receipts(
        cls, value: tuple[RelationPathReceiptRef, ...]
    ) -> tuple[RelationPathReceiptRef, ...]:
        if type(value) is not tuple or len(value) > _MAX_RECEIPT_HASHES:
            raise ExpandedContractError("relation_path_receipts must contain at most 2,000 values")
        if any(type(item) is not RelationPathReceiptRef for item in value):
            raise ExpandedContractError(
                "relation_path_receipts must contain exact RelationPathReceiptRef values"
            )
        ordered = tuple(
            sorted(
                value,
                key=lambda item: (
                    item.receipt_id.encode("ascii"),
                    item.receipt_hash.encode("ascii"),
                ),
            )
        )
        if len({item.receipt_id for item in ordered}) != len(ordered):
            raise ExpandedContractError("RelationPathReceipt IDs must be unique")
        if len({item.receipt_hash for item in ordered}) != len(ordered):
            raise ExpandedContractError("RelationPathReceipt hashes must be unique")
        return ordered

    @model_validator(mode="after")
    def _evidence_matches_candidate_kind(self) -> ExpansionPath:
        structure_pair = (
            self.structure_read_receipt_id is not None,
            self.structure_read_receipt_hash is not None,
        )
        if structure_pair[0] != structure_pair[1]:
            raise ExpandedContractError(
                "StructureReadReceipt ID and hash must be supplied together"
            )
        if self.structure_read_receipt_id is not None:
            structure_hash = self.structure_read_receipt_hash
            if structure_hash is None or self.structure_read_receipt_id != (
                f"structure_read_{structure_hash[:32]}"
            ):
                raise ExpandedContractError(
                    "StructureReadReceipt ID must be derived from its receipt hash"
                )

        if self.candidate_kind is ExpandedCandidateKind.SOURCE_FRAGMENT:
            if self.source_fragment_ids != (self.candidate_id,):
                raise ExpandedContractError(
                    "source_fragment ExpansionPath must ground itself and only itself"
                )
            if structure_pair[0] or self.relation_path_receipts:
                raise ExpandedContractError(
                    "source_fragment ExpansionPath cannot claim structure or relation receipts"
                )
            if not (
                len(self.source_ids) == len(self.source_family_ids) == len(self.corpus_layers) == 1
            ):
                raise ExpandedContractError(
                    "source_fragment ExpansionPath requires one Source, family and layer"
                )
        elif self.candidate_kind is ExpandedCandidateKind.STRUCTURE_UNIT:
            if not structure_pair[0]:
                raise ExpandedContractError(
                    "structure_unit ExpansionPath requires an exact StructureReadReceipt"
                )
            if self.relation_path_receipts:
                raise ExpandedContractError(
                    "structure_unit ExpansionPath cannot claim RelationPathReceipts"
                )
            if len(self.source_fragment_ids) > 128 or not (
                len(self.source_ids) == len(self.source_family_ids) == len(self.corpus_layers) == 1
            ):
                raise ExpandedContractError(
                    "structure_unit ExpansionPath requires one Source, family, layer "
                    "and at most 128 fragments"
                )
        else:
            if structure_pair[0]:
                raise ExpandedContractError(
                    "relation_node ExpansionPath cannot claim a StructureReadReceipt"
                )
            if not self.relation_path_receipts:
                raise ExpandedContractError(
                    "relation_node ExpansionPath requires exact RelationPathReceipts"
                )
        return self

    def grounding_payload(self) -> dict[str, object]:
        return self.identity_payload() | {
            "source_fragment_ids": list(self.source_fragment_ids),
            "source_ids": list(self.source_ids),
            "source_family_ids": list(self.source_family_ids),
            "corpus_layers": [item.value for item in self.corpus_layers],
            "anchor_atomic_ranks": list(self.anchor_atomic_ranks),
            "text_sha256": self.text_sha256,
        }

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            **self.grounding_payload(),
            "structure_read_receipt_id": self.structure_read_receipt_id,
            "structure_read_receipt_hash": self.structure_read_receipt_hash,
            "relation_path_receipts": [item.payload() for item in self.relation_path_receipts],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def path_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def path_id(self) -> str:
        return canonical_content_id("expansion_path", self.semantic_payload())


class ExpansionOmission(_CandidateIdentity):
    """One candidate excluded by a closed, machine-readable reason."""

    reason: ExpansionOmissionReason
    observed_token_count: int | None = Field(default=None, ge=1, le=1_000_000)
    expansion_path_hash: str | None = None

    @field_validator("expansion_path_hash")
    @classmethod
    def _path_hash(cls, value: str | None) -> str | None:
        return None if value is None else _require_hash(value)

    @model_validator(mode="after")
    def _token_count_matches_reason(self) -> ExpansionOmission:
        if self.reason is ExpansionOmissionReason.TOKEN_BUDGET_EXCEEDED:
            if self.observed_token_count is None:
                raise ExpandedContractError(
                    "token_budget_exceeded omission requires its observed token count"
                )
        elif self.observed_token_count is not None:
            raise ExpandedContractError(
                "observed token count is reserved for token_budget_exceeded omissions"
            )
        if self.reason in {
            ExpansionOmissionReason.TOKEN_BUDGET_EXCEEDED,
            ExpansionOmissionReason.LIMIT_EXCEEDED,
        }:
            if self.expansion_path_hash is None:
                raise ExpandedContractError(
                    "grounded expansion omission requires its ExpansionPath hash"
                )
        elif self.expansion_path_hash is not None:
            raise ExpandedContractError(
                "ungrounded or unsupported omission cannot claim an ExpansionPath"
            )
        return self

    def payload(self) -> dict[str, object]:
        return self.identity_payload() | {
            "reason": self.reason.value,
            "observed_token_count": self.observed_token_count,
            "expansion_path_hash": self.expansion_path_hash,
        }


class ExpansionReceipt(_ExpandedModel):
    """Complete expansion closure, independent of storage or execution timing."""

    SCHEMA: ClassVar[str] = "dithyramba.expansion_receipt/1.0"

    expanded_request_id: str
    expanded_request_hash: str
    base_request_id: str
    base_request_hash: str
    base_retrieval_receipt_hash: str
    reranker_profile_hash: str
    seed_ranks: tuple[ExpansionSeedRank, ...] = ()
    eligible: tuple[ExpandedCandidateTrace, ...] = ()
    reranked: tuple[ExpandedCandidateTrace, ...] = ()
    omissions: tuple[ExpansionOmission, ...] = ()
    expansion_path_hashes: tuple[str, ...] = ()
    expansion_paths: tuple[ExpansionPath, ...] = ()
    structure_read_receipt_hashes: tuple[str, ...] = ()
    relation_path_receipt_hashes: tuple[str, ...] = ()

    @field_validator("expanded_request_id")
    @classmethod
    def _expanded_request_id(cls, value: str) -> str:
        return _require_id(value, "expanded_query")

    @field_validator("base_request_id")
    @classmethod
    def _base_request_id(cls, value: str) -> str:
        return _require_id(value, "hybrid_query")

    @field_validator(
        "expanded_request_hash",
        "base_request_hash",
        "base_retrieval_receipt_hash",
        "reranker_profile_hash",
    )
    @classmethod
    def _artifact_hash(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("seed_ranks")
    @classmethod
    def _seeds(cls, value: tuple[ExpansionSeedRank, ...]) -> tuple[ExpansionSeedRank, ...]:
        if type(value) is not tuple or len(value) > _MAX_CANDIDATES:
            raise ExpandedContractError("seed_ranks must contain at most 500 values")
        if any(type(item) is not ExpansionSeedRank for item in value):
            raise ExpandedContractError("seed_ranks must contain exact ExpansionSeedRank values")
        ordered = tuple(
            sorted(
                value,
                key=lambda item: (
                    item.atomic_rank,
                    item.source_fragment_id.encode("ascii"),
                ),
            )
        )
        if len({item.atomic_rank for item in ordered}) != len(ordered):
            raise ExpandedContractError("seed atomic ranks must be unique")
        if len({item.source_fragment_id for item in ordered}) != len(ordered):
            raise ExpandedContractError("seed SourceFragment IDs must be unique")
        return ordered

    @field_validator("eligible", "reranked")
    @classmethod
    def _candidate_trace(
        cls, value: tuple[ExpandedCandidateTrace, ...]
    ) -> tuple[ExpandedCandidateTrace, ...]:
        if type(value) is not tuple or len(value) > _MAX_CANDIDATES:
            raise ExpandedContractError("candidate trace must contain at most 500 values")
        if any(type(item) is not ExpandedCandidateTrace for item in value):
            raise ExpandedContractError(
                "candidate trace must contain exact ExpandedCandidateTrace values"
            )
        _validate_ranked_trace(value)
        return value

    @field_validator("omissions")
    @classmethod
    def _omissions(cls, value: tuple[ExpansionOmission, ...]) -> tuple[ExpansionOmission, ...]:
        if type(value) is not tuple or len(value) > _MAX_CANDIDATES:
            raise ExpandedContractError("omissions must contain at most 500 values")
        if any(type(item) is not ExpansionOmission for item in value):
            raise ExpandedContractError("omissions must contain exact ExpansionOmission values")
        ordered = tuple(sorted(value, key=lambda item: item.candidate_key))
        if len({item.candidate_key for item in ordered}) != len(ordered):
            raise ExpandedContractError("omissions must identify unique candidates")
        return ordered

    @field_validator(
        "expansion_path_hashes",
        "structure_read_receipt_hashes",
        "relation_path_receipt_hashes",
    )
    @classmethod
    def _receipt_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_hashes(value, maximum=_MAX_RECEIPT_HASHES)

    @field_validator("expansion_paths")
    @classmethod
    def _expansion_paths(cls, value: tuple[ExpansionPath, ...]) -> tuple[ExpansionPath, ...]:
        if type(value) is not tuple or len(value) > _MAX_CANDIDATES:
            raise ExpandedContractError("expansion_paths must contain at most 500 values")
        if any(type(item) is not ExpansionPath for item in value):
            raise ExpandedContractError("expansion_paths must contain exact ExpansionPath values")
        ordered = tuple(sorted(value, key=lambda item: item.path_hash.encode("ascii")))
        if len({item.path_hash for item in ordered}) != len(ordered):
            raise ExpandedContractError("ExpansionPath hashes must be unique")
        if len({item.candidate_key for item in ordered}) != len(ordered):
            raise ExpandedContractError("ExpansionPaths must identify unique candidates")
        return ordered

    @model_validator(mode="after")
    def _candidate_closure(self) -> ExpansionReceipt:
        eligible_by_key = {item.candidate_key: item for item in self.eligible}
        reranked_by_key = {item.candidate_key: item for item in self.reranked}
        if set(eligible_by_key) != set(reranked_by_key):
            raise ExpandedContractError(
                "reranked trace must preserve the exact eligible candidate set"
            )
        for key, eligible in eligible_by_key.items():
            reranked = reranked_by_key[key]
            if eligible.provenance_payload() != reranked.provenance_payload():
                raise ExpandedContractError(
                    "reranked trace cannot change eligible candidate provenance"
                )

        seed_rank_values = {item.atomic_rank for item in self.seed_ranks}
        if self.eligible and not seed_rank_values:
            raise ExpandedContractError("eligible candidates require exact atomic seed ranks")
        if any(
            not set(item.anchor_atomic_ranks).issubset(seed_rank_values)
            for item in (*self.eligible, *self.reranked)
        ):
            raise ExpandedContractError("candidate anchor ranks must reference exact seed ranks")

        eligible_keys = set(eligible_by_key)
        if any(item.candidate_key in eligible_keys for item in self.omissions):
            raise ExpandedContractError("eligible candidates cannot also be omissions")

        paths_by_key = {item.candidate_key: item for item in self.expansion_paths}
        grounded_omissions = {
            item.candidate_key: item
            for item in self.omissions
            if item.expansion_path_hash is not None
        }
        if set(paths_by_key) != eligible_keys | set(grounded_omissions):
            raise ExpandedContractError(
                "ExpansionPaths must cover exact eligible and grounded-omission candidates"
            )
        for key, eligible in eligible_by_key.items():
            if paths_by_key[key].grounding_payload() != eligible.grounding_payload():
                raise ExpandedContractError(
                    "ExpansionPath grounding must match eligible candidate provenance"
                )
        for key, omission in grounded_omissions.items():
            if paths_by_key[key].path_hash != omission.expansion_path_hash:
                raise ExpandedContractError(
                    "grounded omission must reference its exact ExpansionPath"
                )

        actual_path_hashes = tuple(item.path_hash for item in self.expansion_paths)
        if actual_path_hashes != self.expansion_path_hashes:
            raise ExpandedContractError(
                "expansion_path_hashes must close over the exact ExpansionPath artifacts"
            )
        actual_structure_hashes = tuple(
            sorted(
                {
                    item.structure_read_receipt_hash
                    for item in self.expansion_paths
                    if item.structure_read_receipt_hash is not None
                },
                key=str.encode,
            )
        )
        if actual_structure_hashes != self.structure_read_receipt_hashes:
            raise ExpandedContractError(
                "structure receipt hashes must close over exact ExpansionPaths"
            )
        actual_relation_hashes = tuple(
            sorted(
                {
                    receipt.receipt_hash
                    for item in self.expansion_paths
                    for receipt in item.relation_path_receipts
                },
                key=str.encode,
            )
        )
        if actual_relation_hashes != self.relation_path_receipt_hashes:
            raise ExpandedContractError(
                "relation receipt hashes must close over exact ExpansionPaths"
            )
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "expanded_request_id": self.expanded_request_id,
            "expanded_request_hash": self.expanded_request_hash,
            "base_request_id": self.base_request_id,
            "base_request_hash": self.base_request_hash,
            "base_retrieval_receipt_hash": self.base_retrieval_receipt_hash,
            "reranker_profile_hash": self.reranker_profile_hash,
            "seed_ranks": [item.payload() for item in self.seed_ranks],
            "eligible": [item.payload() for item in self.eligible],
            "reranked": [item.payload() for item in self.reranked],
            "omissions": [item.payload() for item in self.omissions],
            "expansion_path_hashes": list(self.expansion_path_hashes),
            "structure_read_receipt_hashes": list(self.structure_read_receipt_hashes),
            "relation_path_receipt_hashes": list(self.relation_path_receipt_hashes),
        }

    @property
    def eligible_set_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": "dithyramba.expansion_eligible_set/1.0",
                "candidates": [item.provenance_payload() for item in self.eligible],
            }
        )

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def receipt_id(self) -> str:
        return canonical_content_id("expansion_receipt", self.semantic_payload())


class ExpandedPacketItem(ExpandedCandidateTrace):
    """One final text-free candidate with the exact reranked provenance."""

    rank: int = Field(ge=1, le=_MAX_FINAL)


class ExpandedEvidencePacket(_ExpandedModel):
    """Text-free selected subset layered over one frozen HybridEvidencePacket/1.0."""

    SCHEMA: ClassVar[str] = "dithyramba.expanded_evidence_packet/2.0"

    expanded_request_id: str
    expanded_request_hash: str
    base_packet_id: str
    base_packet_hash: str
    expansion_receipt_hash: str
    result_status: Literal["evidence_found", "no_evidence"]
    items: tuple[ExpandedPacketItem, ...] = ()

    @field_validator("expanded_request_id")
    @classmethod
    def _expanded_request_id(cls, value: str) -> str:
        return _require_id(value, "expanded_query")

    @field_validator("base_packet_id")
    @classmethod
    def _base_packet_id(cls, value: str) -> str:
        return _require_id(value, "hybrid_packet")

    @field_validator(
        "expanded_request_hash",
        "base_packet_hash",
        "expansion_receipt_hash",
    )
    @classmethod
    def _hash(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("items")
    @classmethod
    def _items(cls, value: tuple[ExpandedPacketItem, ...]) -> tuple[ExpandedPacketItem, ...]:
        if type(value) is not tuple or len(value) > _MAX_FINAL:
            raise ExpandedContractError("expanded packet may contain at most 100 items")
        if any(type(item) is not ExpandedPacketItem for item in value):
            raise ExpandedContractError("packet items must be exact ExpandedPacketItem values")
        _validate_ranked_trace(value)
        return value

    @model_validator(mode="after")
    def _status_matches_items(self) -> ExpandedEvidencePacket:
        if (self.result_status == "evidence_found") != bool(self.items):
            raise ExpandedContractError(
                "expanded packet result_status must match whether evidence exists"
            )
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "expanded_request_id": self.expanded_request_id,
            "expanded_request_hash": self.expanded_request_hash,
            "base_packet_id": self.base_packet_id,
            "base_packet_hash": self.base_packet_hash,
            "expansion_receipt_hash": self.expansion_receipt_hash,
            "result_status": self.result_status,
            "items": [item.payload() for item in self.items],
        }

    @property
    def packet_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def packet_id(self) -> str:
        return canonical_content_id("expanded_packet", self.semantic_payload())


def _validate_ranked_trace(
    values: tuple[ExpandedCandidateTrace, ...] | tuple[ExpandedPacketItem, ...],
) -> None:
    if tuple(item.rank for item in values) != tuple(range(1, len(values) + 1)):
        raise ExpandedContractError("candidate trace ranks must be contiguous from one")
    keys = tuple(item.candidate_key for item in values)
    if len(set(keys)) != len(keys):
        raise ExpandedContractError("candidate trace contains duplicate candidate identities")


def _canonical_ids(
    values: tuple[str, ...],
    *,
    prefix: str,
    field: str,
    maximum: int,
    non_empty: bool,
) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) > maximum or (non_empty and not values):
        qualifier = "a non-empty" if non_empty else "an"
        raise ExpandedContractError(f"{field} must be {qualifier} bounded tuple")
    ordered = tuple(sorted((_require_id(item, prefix) for item in values), key=str.encode))
    if len(set(ordered)) != len(ordered):
        raise ExpandedContractError(f"{field} must contain unique IDs")
    return ordered


def _canonical_hashes(values: tuple[str, ...], *, maximum: int) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) > maximum:
        raise ExpandedContractError(f"hash collection must contain at most {maximum} values")
    ordered = tuple(sorted((_require_hash(item) for item in values), key=str.encode))
    if len(set(ordered)) != len(ordered):
        raise ExpandedContractError("hash collection must contain unique values")
    return ordered


def _require_hash(value: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise ExpandedContractError("artifact hash must be 64 lowercase hexadecimal characters")
    return value


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise ExpandedContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise ExpandedContractError(f"identifier must use a canonical {expected} suffix")
    return value
