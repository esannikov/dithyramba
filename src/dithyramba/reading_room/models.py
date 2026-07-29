"""Immutable, canonical contracts for the read-only Reading Room projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, TypeAlias

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex

from .errors import ReadingRoomIntegrityError, ReadingRoomLimitError

MetadataValue: TypeAlias = str | int | bool


class ReadingRoomNodeType(StrEnum):
    """Node types that may appear in the explicit semantic projection."""

    SOURCE_FRAGMENT = "source_fragment"
    STATEMENT = "statement"
    EVIDENCE_LINK = "evidence_link"
    VOICE = "voice"
    ENTITY = "entity"
    ENTITY_MENTION = "entity_mention"
    CONCEPT = "concept"
    CONCEPT_MENTION = "concept_mention"
    CONCEPT_MEANING = "concept_meaning"
    TIME_CONTEXT = "time_context"


class ReadingRoomEdgeType(StrEnum):
    """Only persisted typed relations; co-occurrence is deliberately absent."""

    STATEMENT_VOICE = "statement_voice"
    STATEMENT_EVIDENCE_LINK = "statement_evidence_link"
    EVIDENCE_LINK_FRAGMENT = "evidence_link_fragment"
    ENTITY_MENTION_ENTITY = "entity_mention_entity"
    ENTITY_MENTION_FRAGMENT = "entity_mention_fragment"
    CONCEPT_MENTION_CONCEPT = "concept_mention_concept"
    CONCEPT_MENTION_FRAGMENT = "concept_mention_fragment"
    CONCEPT_MEANING_CONCEPT = "concept_meaning_concept"
    CONCEPT_MEANING_VOICE = "concept_meaning_voice"
    CONCEPT_MEANING_STATEMENT = "concept_meaning_statement"
    STATEMENT_TIME_CONTEXT = "statement_time_context"
    CONCEPT_MEANING_TIME_CONTEXT = "concept_meaning_time_context"


@dataclass(frozen=True, slots=True, order=True)
class ReadingRoomMetadata:
    """One deterministic scalar metadata field on a projected node."""

    key: str
    value: MetadataValue

    def __post_init__(self) -> None:
        if type(self.key) is not str or not self.key:
            raise ReadingRoomIntegrityError("projection metadata key is invalid")
        if type(self.value) not in (str, int, bool):
            raise ReadingRoomIntegrityError("projection metadata value is invalid")

    def payload(self) -> dict[str, object]:
        return {"key": self.key, "value": self.value}


@dataclass(frozen=True, slots=True)
class ReadingRoomLimits:
    """Hard, caller-visible bounds for one projection request."""

    MAX_NODES: ClassVar[int] = 2_000
    MAX_EDGES: ClassVar[int] = 5_000
    MAX_RUNS: ClassVar[int] = 100
    MAX_REVIEW_ITEMS: ClassVar[int] = 500
    MAX_PROVENANCE_RUNS: ClassVar[int] = 250
    MAX_CORPUS_FRAGMENTS: ClassVar[int] = 2_000

    node_limit: int = 500
    edge_limit: int = 1_000
    run_limit: int = 20
    review_limit: int = 100
    provenance_run_limit: int = 100
    corpus_fragment_limit: int = 200

    def __post_init__(self) -> None:
        for value, maximum, label in (
            (self.node_limit, self.MAX_NODES, "node_limit"),
            (self.edge_limit, self.MAX_EDGES, "edge_limit"),
            (self.run_limit, self.MAX_RUNS, "run_limit"),
            (self.review_limit, self.MAX_REVIEW_ITEMS, "review_limit"),
            (
                self.provenance_run_limit,
                self.MAX_PROVENANCE_RUNS,
                "provenance_run_limit",
            ),
            (
                self.corpus_fragment_limit,
                self.MAX_CORPUS_FRAGMENTS,
                "corpus_fragment_limit",
            ),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ReadingRoomLimitError(f"{label} must be between 1 and {maximum}")

    def payload(self) -> dict[str, int]:
        return {
            "node_limit": self.node_limit,
            "edge_limit": self.edge_limit,
            "run_limit": self.run_limit,
            "review_limit": self.review_limit,
            "provenance_run_limit": self.provenance_run_limit,
            "corpus_fragment_limit": self.corpus_fragment_limit,
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomLibrary:
    library_id: str
    name: str
    logical_identity_hash: str
    created_at: str
    storage_schema_version: int
    storage_schema_fingerprint: str

    def payload(self) -> dict[str, object]:
        return {
            "library_id": self.library_id,
            "name": self.name,
            "logical_identity_hash": self.logical_identity_hash,
            "created_at": self.created_at,
            "storage_schema_version": self.storage_schema_version,
            "storage_schema_fingerprint": self.storage_schema_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomAccess:
    corpus_snapshot_id: str
    snapshot_hash: str
    snapshot_created_at: str
    access_policy_id: str
    access_policy_name: str
    policy_hash: str
    policy_created_at: str
    purpose: str
    permitted_collection_ids: tuple[str, ...]
    exclusion_hash: str
    permitted_set_hash: str
    permitted_token_hash: str
    policy_omission_present: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "permitted_collection_ids",
            tuple(sorted(set(self.permitted_collection_ids))),
        )

    def payload(self) -> dict[str, object]:
        return {
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "snapshot_created_at": self.snapshot_created_at,
            "access_policy_id": self.access_policy_id,
            "access_policy_name": self.access_policy_name,
            "policy_hash": self.policy_hash,
            "policy_created_at": self.policy_created_at,
            "purpose": self.purpose,
            "permitted_collection_ids": list(self.permitted_collection_ids),
            "exclusion_hash": self.exclusion_hash,
            "permitted_set_hash": self.permitted_set_hash,
            "permitted_token_hash": self.permitted_token_hash,
            "policy_omission_present": self.policy_omission_present,
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomFragmentRef:
    """One permitted, text-free fragment lineage row in the corpus inventory."""

    source_fragment_id: str
    source_version_id: str
    source_id: str
    source_family_id: str | None
    collection_ids: tuple[str, ...]
    text_sha256: str
    address_hash: str
    ordinal: int
    fragment_kind: str
    version_number: int
    content_sha256: str
    observed_at: str
    source_modified_at: str | None
    parser_profile: str
    parse_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "collection_ids", tuple(sorted(set(self.collection_ids))))

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_version_id": self.source_version_id,
            "source_id": self.source_id,
            "source_family_id": self.source_family_id,
            "collection_ids": list(self.collection_ids),
            "text_sha256": self.text_sha256,
            "address_hash": self.address_hash,
            "ordinal": self.ordinal,
            "fragment_kind": self.fragment_kind,
            "version_number": self.version_number,
            "content_sha256": self.content_sha256,
            "observed_at": self.observed_at,
            "source_modified_at": self.source_modified_at,
            "parser_profile": self.parser_profile,
            "parse_status": self.parse_status,
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomCorpus:
    collection_count: int
    source_count: int
    source_version_count: int
    source_fragment_count: int
    oldest_source_observed_at: str | None
    newest_source_observed_at: str | None
    newest_source_modified_at: str | None
    parser_profiles: tuple[str, ...]
    fragments: tuple[ReadingRoomFragmentRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parser_profiles", tuple(sorted(set(self.parser_profiles))))
        object.__setattr__(
            self,
            "fragments",
            tuple(sorted(self.fragments, key=lambda item: item.source_fragment_id)),
        )

    def payload(self) -> dict[str, object]:
        return {
            "collection_count": self.collection_count,
            "source_count": self.source_count,
            "source_version_count": self.source_version_count,
            "source_fragment_count": self.source_fragment_count,
            "oldest_source_observed_at": self.oldest_source_observed_at,
            "newest_source_observed_at": self.newest_source_observed_at,
            "newest_source_modified_at": self.newest_source_modified_at,
            "parser_profiles": list(self.parser_profiles),
            "fragments": [item.payload() for item in self.fragments],
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomRun:
    processing_run_id: str
    kind: str
    status: str
    extraction_profile: str
    code_version: str
    generator_hash: str
    review_budget_hash: str
    input_hash: str
    proposal_hash: str | None
    output_hash: str
    run_receipt_hash: str
    started_at: str
    finished_at: str
    coverage_report_id: str
    coverage_state: str
    coverage_report_hash: str
    read_receipt_id: str
    read_receipt_hash: str
    read_fragment_count: int
    candidate_count: int
    omission_count: int
    backlog_warning: bool
    error_code: str | None

    def payload(self) -> dict[str, object]:
        return {
            "processing_run_id": self.processing_run_id,
            "kind": self.kind,
            "status": self.status,
            "extraction_profile": self.extraction_profile,
            "code_version": self.code_version,
            "generator_hash": self.generator_hash,
            "review_budget_hash": self.review_budget_hash,
            "input_hash": self.input_hash,
            "proposal_hash": self.proposal_hash,
            "output_hash": self.output_hash,
            "run_receipt_hash": self.run_receipt_hash,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "coverage_report_id": self.coverage_report_id,
            "coverage_state": self.coverage_state,
            "coverage_report_hash": self.coverage_report_hash,
            "read_receipt_id": self.read_receipt_id,
            "read_receipt_hash": self.read_receipt_hash,
            "read_fragment_count": self.read_fragment_count,
            "candidate_count": self.candidate_count,
            "omission_count": self.omission_count,
            "backlog_warning": self.backlog_warning,
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomNode:
    node_id: str
    node_type: ReadingRoomNodeType
    kind: str | None
    label: str | None
    candidate_state: str
    review_state: str
    latest_review_decision_id: str | None
    content_hash: str
    provenance_processing_run_id: str | None
    provenance_fragment_ids: tuple[str, ...]
    collection_ids: tuple[str, ...]
    created_at: str | None
    metadata: tuple[ReadingRoomMetadata, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.node_type, ReadingRoomNodeType):
            raise ReadingRoomIntegrityError("projection node type is invalid")
        fragments = tuple(sorted(set(self.provenance_fragment_ids)))
        collections = tuple(sorted(set(self.collection_ids)))
        metadata = tuple(sorted(self.metadata, key=lambda item: item.key))
        if len({item.key for item in metadata}) != len(metadata):
            raise ReadingRoomIntegrityError("projection node metadata keys must be unique")
        object.__setattr__(self, "provenance_fragment_ids", fragments)
        object.__setattr__(self, "collection_ids", collections)
        object.__setattr__(self, "metadata", metadata)

    def payload(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "kind": self.kind,
            "label": self.label,
            "candidate_state": self.candidate_state,
            "review_state": self.review_state,
            "latest_review_decision_id": self.latest_review_decision_id,
            "content_hash": self.content_hash,
            "provenance_processing_run_id": self.provenance_processing_run_id,
            "provenance_fragment_ids": list(self.provenance_fragment_ids),
            "collection_ids": list(self.collection_ids),
            "created_at": self.created_at,
            "metadata": [item.payload() for item in self.metadata],
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomEdge:
    edge_id: str
    edge_type: ReadingRoomEdgeType
    source_node_id: str
    target_node_id: str
    relation_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.edge_type, ReadingRoomEdgeType):
            raise ReadingRoomIntegrityError("projection edge type is invalid")

    def payload(self) -> dict[str, str]:
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type.value,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relation_hash": self.relation_hash,
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomGraph:
    nodes: tuple[ReadingRoomNode, ...]
    edges: tuple[ReadingRoomEdge, ...]
    visible_node_count: int
    visible_edge_count: int

    def __post_init__(self) -> None:
        nodes = tuple(sorted(self.nodes, key=lambda item: (item.node_type.value, item.node_id)))
        edges = tuple(
            sorted(
                self.edges,
                key=lambda item: (
                    item.edge_type.value,
                    item.source_node_id,
                    item.target_node_id,
                    item.edge_id,
                ),
            )
        )
        node_ids = {item.node_id for item in nodes}
        if len(node_ids) != len(nodes):
            raise ReadingRoomIntegrityError("projection graph contains duplicate nodes")
        if len({item.edge_id for item in edges}) != len(edges):
            raise ReadingRoomIntegrityError("projection graph contains duplicate edges")
        if any(
            edge.source_node_id not in node_ids or edge.target_node_id not in node_ids
            for edge in edges
        ):
            raise ReadingRoomIntegrityError("projection graph contains a dangling edge")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    def payload(self) -> dict[str, object]:
        return {
            "nodes": [item.payload() for item in self.nodes],
            "edges": [item.payload() for item in self.edges],
            "visible_node_count": self.visible_node_count,
            "visible_edge_count": self.visible_edge_count,
        }


@dataclass(frozen=True, slots=True, order=True)
class ReadingRoomReviewStatusCount:
    review_state: str
    count: int

    def payload(self) -> dict[str, object]:
        return {"review_state": self.review_state, "count": self.count}


@dataclass(frozen=True, slots=True)
class ReadingRoomReviewItem:
    target_type: str
    target_id: str
    target_hash: str
    collection_ids: tuple[str, ...]
    created_at: str
    review_state: str
    latest_review_decision_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "collection_ids", tuple(sorted(set(self.collection_ids))))

    def payload(self) -> dict[str, object]:
        return {
            "target_type": self.target_type,
            "target_id": self.target_id,
            "target_hash": self.target_hash,
            "collection_ids": list(self.collection_ids),
            "created_at": self.created_at,
            "review_state": self.review_state,
            "latest_review_decision_id": self.latest_review_decision_id,
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomReview:
    visible_candidate_count: int
    unresolved_count: int
    status_counts: tuple[ReadingRoomReviewStatusCount, ...]
    queue: tuple[ReadingRoomReviewItem, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status_counts", tuple(sorted(self.status_counts)))
        object.__setattr__(
            self,
            "queue",
            tuple(
                sorted(
                    self.queue,
                    key=lambda item: (item.created_at, item.target_type, item.target_id),
                )
            ),
        )

    def payload(self) -> dict[str, object]:
        return {
            "visible_candidate_count": self.visible_candidate_count,
            "unresolved_count": self.unresolved_count,
            "status_counts": [item.payload() for item in self.status_counts],
            "queue": [item.payload() for item in self.queue],
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomOmissions:
    limits: ReadingRoomLimits
    provenance_run_scan_truncated: bool
    graph_nodes_truncated: bool
    graph_edges_truncated: bool
    recent_runs_truncated: bool
    review_queue_truncated: bool
    corpus_fragments_truncated: bool
    closure_omission_present: bool

    def payload(self) -> dict[str, object]:
        return {
            "limits": self.limits.payload(),
            "provenance_run_scan_truncated": self.provenance_run_scan_truncated,
            "graph_nodes_truncated": self.graph_nodes_truncated,
            "graph_edges_truncated": self.graph_edges_truncated,
            "recent_runs_truncated": self.recent_runs_truncated,
            "review_queue_truncated": self.review_queue_truncated,
            "corpus_fragments_truncated": self.corpus_fragments_truncated,
            "closure_omission_present": self.closure_omission_present,
        }


@dataclass(frozen=True, slots=True)
class ReadingRoomProjection:
    """Canonical access-safe dashboard state; no raw source or evidence text."""

    SCHEMA: ClassVar[str] = "dithyramba.reading_room_projection/1.0"
    STORAGE_SCHEMA_VERSION: ClassVar[int] = 10

    library: ReadingRoomLibrary
    access: ReadingRoomAccess
    corpus: ReadingRoomCorpus
    recent_runs: tuple[ReadingRoomRun, ...]
    review: ReadingRoomReview
    graph: ReadingRoomGraph
    omissions: ReadingRoomOmissions
    projection_hash: str = field(init=False)

    def __post_init__(self) -> None:
        runs = tuple(
            sorted(
                self.recent_runs,
                key=lambda item: (item.finished_at, item.processing_run_id),
                reverse=True,
            )
        )
        object.__setattr__(self, "recent_runs", runs)
        object.__setattr__(self, "projection_hash", canonical_sha256_hex(self.semantic_payload()))

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library": self.library.payload(),
            "access": self.access.payload(),
            "corpus": self.corpus.payload(),
            "recent_runs": [item.payload() for item in self.recent_runs],
            "review": self.review.payload(),
            "graph": self.graph.payload(),
            "omissions": self.omissions.payload(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    def payload(self) -> dict[str, object]:
        return {**self.semantic_payload(), "projection_hash": self.projection_hash}
