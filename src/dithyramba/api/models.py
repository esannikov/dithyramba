"""Strict public JSON contracts for the one-Library loopback API."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field

from dithyramba.access import QueryExclusions
from dithyramba.collections import CollectionConfig, CollectionKind, CollectionRoot
from dithyramba.contracts import canonical_json_bytes
from dithyramba.ingest import MarkdownSourceAddress, PdfSourceAddress, SourceAddress
from dithyramba.persistence import (
    CollectionRecord,
    IngestBatchResult,
    IngestDisposition,
    LibraryRecord,
    ProcessingRunRecord,
    ProcessingRunStatus,
    SourceRecord,
    SourceVersionRecord,
    TerminalInputOutcome,
)
from dithyramba.recall import (
    EvidenceFragment,
    EvidencePacket,
    EvidencePacketResultContract,
    OmissionCategory,
    OmissionDisclosure,
    PacketResultStatus,
    QueryRequest,
    ReadReceipt,
    RecallResult,
    RetrievalBudget,
)
from dithyramba.review import (
    ReviewAction,
    ReviewDecision,
    ReviewDecisionRequest,
    ReviewScope,
    ReviewTargetType,
)
from dithyramba.snapshots import CorpusSnapshot


class _ApiModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ErrorResponse(_ApiModel):
    detail: str


class HealthResponse(_ApiModel):
    status: Literal["ok"] = "ok"
    library_id: str
    schema_version: int


class LibraryResponse(_ApiModel):
    created_at: str
    library_id: str
    logical_identity_hash: str
    name: str
    path: str

    @classmethod
    def from_record(cls, record: LibraryRecord) -> Self:
        return cls(
            created_at=record.created_at,
            library_id=record.config.library_id,
            logical_identity_hash=record.logical_identity_hash,
            name=record.config.name,
            path=str(record.paths.root),
        )


class LibraryListResponse(_ApiModel):
    libraries: tuple[LibraryResponse, ...]


class CollectionRootCreateRequest(_ApiModel):
    path: str = Field(min_length=1, max_length=4_096)
    include_globs: list[str] = Field(
        default_factory=lambda: ["**/*.md", "**/*.markdown", "**/*.txt", "**/*.pdf"],
        max_length=64,
    )
    exclude_globs: list[str] = Field(default_factory=list, max_length=64)


class CollectionCreateRequest(_ApiModel):
    collection_id: str | None = Field(
        default=None,
        pattern=r"^collection_[0-9a-f]{32}$",
    )
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["corpus", "case", "branch", "holdout"]
    roots: list[CollectionRootCreateRequest] = Field(min_length=1, max_length=16)

    def to_domain(self, *, library_id: str) -> CollectionConfig:
        roots = tuple(
            CollectionRoot(
                path=Path(root.path),
                include_globs=tuple(root.include_globs),
                exclude_globs=tuple(root.exclude_globs),
            )
            for root in self.roots
        )
        if self.collection_id is None:
            return CollectionConfig(
                library_id=library_id,
                name=self.name,
                kind=CollectionKind(self.kind),
                roots=roots,
            )
        return CollectionConfig(
            collection_id=self.collection_id,
            library_id=library_id,
            name=self.name,
            kind=CollectionKind(self.kind),
            roots=roots,
        )


class CollectionRootResponse(_ApiModel):
    collection_root_id: str
    exclude_globs: tuple[str, ...]
    include_globs: tuple[str, ...]
    path: str


class CollectionResponse(_ApiModel):
    collection_id: str
    created_at: str
    kind: CollectionKind
    library_id: str
    name: str
    roots: tuple[CollectionRootResponse, ...]

    @classmethod
    def from_record(cls, record: CollectionRecord) -> Self:
        return cls(
            collection_id=record.config.collection_id,
            created_at=record.created_at,
            kind=record.config.kind,
            library_id=record.config.library_id,
            name=record.config.name,
            roots=tuple(
                CollectionRootResponse(
                    collection_root_id=root_id,
                    exclude_globs=root.exclude_globs,
                    include_globs=root.include_globs,
                    path=str(root.path),
                )
                for root_id, root in zip(
                    record.collection_root_ids,
                    record.config.roots,
                    strict=True,
                )
            ),
        )


class CollectionListResponse(_ApiModel):
    collections: tuple[CollectionResponse, ...]
    library_id: str


class SnapshotMemberResponse(_ApiModel):
    collection_id: str
    source_id: str
    source_version_id: str
    content_sha256: str
    membership_state: Literal["active", "excluded", "holdout"]
    source_family_id: str
    root_source_id: str
    family_role: Literal["root", "derivative", "duplicate"]


class SnapshotScopeResponse(_ApiModel):
    schema_id: Literal["dithyramba.corpus_snapshot_scope/1.0"] = Field(alias="schema")
    library_id: str
    collection_ids: tuple[str, ...]


class CorpusSnapshotResponse(_ApiModel):
    schema_id: Literal["dithyramba.corpus_snapshot_manifest/1.0"] = Field(alias="schema")
    corpus_snapshot_id: str
    manifest_hash: str
    scope: SnapshotScopeResponse
    members: tuple[SnapshotMemberResponse, ...]

    @classmethod
    def from_snapshot(cls, snapshot: CorpusSnapshot) -> Self:
        payload = {
            **snapshot.semantic_payload(),
            "corpus_snapshot_id": snapshot.corpus_snapshot_id,
            "manifest_hash": snapshot.manifest_hash,
        }
        return cls.model_validate_json(canonical_json_bytes(payload))


class SourceIngestRequest(_ApiModel):
    library_id: str = Field(pattern=r"^library_[0-9a-f]{32}$")
    collection_id: str = Field(pattern=r"^collection_[0-9a-f]{32}$")
    relative_path: str = Field(min_length=1, max_length=4_096)
    collection_root_id: str | None = Field(
        default=None,
        pattern=r"^root_[a-z0-9_]+$",
    )


class SourceMembershipResponse(_ApiModel):
    collection_id: str
    created_at: str
    state: Literal["active", "excluded", "holdout"]


class SourceResponse(_ApiModel):
    created_at: str
    current_source_version_id: str
    family_role: Literal["root", "derivative", "duplicate"]
    library_id: str
    media_type: Literal["text/markdown", "text/plain", "application/pdf"]
    memberships: tuple[SourceMembershipResponse, ...]
    root_source_id: str
    source_family_id: str
    source_id: str
    title: str | None

    @classmethod
    def from_record(cls, record: SourceRecord) -> Self:
        return cls(
            created_at=record.created_at,
            current_source_version_id=record.current_source_version_id,
            family_role=record.family_role.value,
            library_id=record.library_id,
            media_type=cast(
                Literal["text/markdown", "text/plain", "application/pdf"],
                record.media_type,
            ),
            memberships=tuple(
                SourceMembershipResponse(
                    collection_id=membership.collection_id,
                    created_at=membership.created_at,
                    state=cast(
                        Literal["active", "excluded", "holdout"],
                        membership.state,
                    ),
                )
                for membership in record.memberships
            ),
            root_source_id=record.root_source_id,
            source_family_id=record.source_family_id,
            source_id=record.source_id,
            title=record.title,
        )


class SourceVersionResponse(_ApiModel):
    byte_size: int
    content_sha256: str
    failure_code: str | None
    is_current: bool
    observed_at: str
    parse_status: Literal["processed", "skipped", "failed"]
    parser_profile: str
    source_id: str
    source_modified_at: str | None
    source_version_id: str
    version_number: int

    @classmethod
    def from_record(cls, record: SourceVersionRecord) -> Self:
        return cls(
            byte_size=record.byte_size,
            content_sha256=record.content_sha256,
            failure_code=record.failure_code,
            is_current=record.is_current,
            observed_at=record.observed_at,
            parse_status=cast(
                Literal["processed", "skipped", "failed"],
                record.parse_status,
            ),
            parser_profile=record.parser_profile,
            source_id=record.source_id,
            source_modified_at=record.source_modified_at,
            source_version_id=record.source_version_id,
            version_number=record.version_number,
        )


class SourceVersionListResponse(_ApiModel):
    library_id: str
    source: SourceResponse
    versions: tuple[SourceVersionResponse, ...]


class ProcessingRunResponse(_ApiModel):
    processing_run_id: str
    kind: str
    status: ProcessingRunStatus
    code_version: str
    profile_version: str
    started_at: str
    finished_at: str | None
    error_code: str | None
    output_hash: str | None

    @classmethod
    def from_record(cls, record: ProcessingRunRecord) -> Self:
        return cls(
            processing_run_id=record.processing_run_id,
            kind=record.kind,
            status=record.status,
            code_version=record.code_version,
            profile_version=record.profile_version,
            started_at=record.started_at,
            finished_at=record.finished_at,
            error_code=record.error_code,
            output_hash=record.output_hash,
        )


class IngestOmissionResponse(_ApiModel):
    category: str
    count: int | None
    disclosure: str
    omission_id: str
    reason_code: str


class IngestCoverageResponse(_ApiModel):
    coverage_report_id: str
    failed_count: int
    omissions: tuple[IngestOmissionResponse, ...]
    policy_omission_present: bool
    processed_count: int
    processing_run_id: str
    report_hash: str
    skipped_count: int
    stage: str


class IngestOutcomeResponse(_ApiModel):
    collection_root_id: str
    relative_path: str | None
    terminal_outcome: TerminalInputOutcome
    disposition: IngestDisposition | None
    failure_code: str | None
    retryable: bool
    source_id: str | None
    source_version_id: str | None
    source_family_id: str | None
    root_source_id: str | None
    fragment_count: int
    infrastructure_failure: bool


class IngestResultResponse(_ApiModel):
    schema_id: Literal["dithyramba.ingest_result/1.0"] = Field(
        default="dithyramba.ingest_result/1.0",
        alias="schema",
    )
    collection_id: str
    coverage: IngestCoverageResponse
    library_id: str
    outcomes: tuple[IngestOutcomeResponse, ...]
    run: ProcessingRunResponse

    @classmethod
    def from_result(
        cls,
        result: IngestBatchResult,
        *,
        library_id: str,
        collection_id: str,
    ) -> Self:
        coverage = result.coverage
        return cls(
            collection_id=collection_id,
            coverage=IngestCoverageResponse(
                coverage_report_id=coverage.coverage_report_id,
                failed_count=coverage.failed_count,
                omissions=tuple(
                    IngestOmissionResponse(
                        category=omission.category,
                        count=omission.count,
                        disclosure=omission.disclosure,
                        omission_id=omission.omission_id,
                        reason_code=omission.reason_code,
                    )
                    for omission in coverage.omissions
                ),
                policy_omission_present=coverage.policy_omission_present,
                processed_count=coverage.processed_count,
                processing_run_id=coverage.processing_run_id,
                report_hash=coverage.report_hash,
                skipped_count=coverage.skipped_count,
                stage=coverage.stage,
            ),
            library_id=library_id,
            outcomes=tuple(
                IngestOutcomeResponse(
                    collection_root_id=outcome.collection_root_id,
                    relative_path=outcome.relative_path,
                    terminal_outcome=outcome.terminal_outcome,
                    disposition=outcome.disposition,
                    failure_code=outcome.failure_code,
                    retryable=outcome.retryable,
                    source_id=outcome.source_id,
                    source_version_id=outcome.source_version_id,
                    source_family_id=outcome.source_family_id,
                    root_source_id=outcome.root_source_id,
                    fragment_count=outcome.fragment_count,
                    infrastructure_failure=outcome.infrastructure_failure,
                )
                for outcome in result.outcomes
            ),
            run=ProcessingRunResponse.from_record(result.run),
        )


class MarkdownAddressResponse(_ApiModel):
    schema_id: Literal["dithyramba.source_address/1.0"] = Field(alias="schema")
    kind: Literal["markdown"]
    heading_path: tuple[str, ...]
    line_start: int
    line_end: int
    char_start: int
    char_end: int


class PdfAddressResponse(_ApiModel):
    schema_id: Literal["dithyramba.source_address/1.0"] = Field(alias="schema")
    kind: Literal["pdf"]
    page: int
    bbox: tuple[str, str, str, str]
    char_start: int
    char_end: int


SourceAddressResponse = MarkdownAddressResponse | PdfAddressResponse


class EvidenceFragmentResponse(_ApiModel):
    source_fragment_id: str
    source_version_id: str
    source_family_id: str | None
    rank: int
    score: str
    text: str
    text_sha256: str
    source_address: SourceAddressResponse

    @classmethod
    def from_fragment(cls, fragment: EvidenceFragment) -> Self:
        return cls(
            source_fragment_id=fragment.source_fragment_id,
            source_version_id=fragment.source_version_id,
            source_family_id=fragment.source_family_id,
            rank=fragment.rank,
            score=fragment.score,
            text=fragment.text,
            text_sha256=fragment.text_sha256,
            source_address=address_response(fragment.source_address),
        )


class SourceChipResponse(_ApiModel):
    SCHEMA: ClassVar[Literal["dithyramba.source_chip/1.0"]] = "dithyramba.source_chip/1.0"

    schema_id: Literal["dithyramba.source_chip/1.0"] = Field(
        default=SCHEMA,
        alias="schema",
    )
    evidence_packet_id: str
    packet_hash: str
    source_id: str
    source_family_id: str
    root_source_id: str
    family_role: Literal["root", "derivative", "duplicate"]
    source_version_id: str
    source_fragment_id: str
    text_sha256: str
    source_address: SourceAddressResponse
    viewer_url: str


class NotEvaluatedResponse(_ApiModel):
    status: Literal["not_evaluated"]
    items: tuple[object, ...]


class PolicyOmissionResponse(_ApiModel):
    present: bool
    count: None


class RecallOmissionResponse(_ApiModel):
    category: OmissionCategory
    disclosure: OmissionDisclosure
    count: int | None
    reason_code: str


class RecallCoverageResponse(_ApiModel):
    coverage_report_id: str
    coverage_report_hash: str
    schema_id: Literal["dithyramba.coverage_report/1.0"] = Field(alias="schema")
    stage: Literal["recall"]
    query_request_hash: str
    processed_count: int
    skipped_count: int
    failed_count: int
    policy_omission: PolicyOmissionResponse
    omissions: tuple[RecallOmissionResponse, ...]


class ReceiptReferencesResponse(_ApiModel):
    read_receipt_id: str
    read_receipt_hash: str
    access_receipt_id: str
    access_receipt_hash: str
    retrieval_receipt_id: str
    retrieval_receipt_hash: str


class EvidencePacketResponse(_ApiModel):
    """Exact ``EvidencePacket.payload()`` shape shared by API and CLI."""

    schema_id: Literal["dithyramba.evidence_packet/1.0"] = Field(alias="schema")
    evidence_packet_id: str
    query_request_id: str
    query_request_hash: str
    corpus_snapshot_id: str
    corpus_snapshot_hash: str
    result_status: PacketResultStatus
    source_fragments: tuple[EvidenceFragmentResponse, ...]
    counterevidence: NotEvaluatedResponse
    evidence_gaps: NotEvaluatedResponse
    coverage_report: RecallCoverageResponse
    receipts: ReceiptReferencesResponse
    packet_hash: str

    @classmethod
    def from_packet(cls, packet: EvidencePacket) -> Self:
        return cls.model_validate_json(packet.canonical_bytes)


class SourceChipListResponse(_ApiModel):
    schema_id: Literal["dithyramba.source_chip_list/1.0"] = Field(
        default="dithyramba.source_chip_list/1.0",
        alias="schema",
    )
    evidence_packet_id: str
    packet_hash: str
    items: tuple[SourceChipResponse, ...]


class PacketBackedSourceFragmentResponse(_ApiModel):
    schema_id: Literal["dithyramba.packet_backed_source_fragment/1.0"] = Field(
        default="dithyramba.packet_backed_source_fragment/1.0",
        alias="schema",
    )
    evidence_packet_id: str
    packet_hash: str
    source_id: str
    fragment: EvidenceFragmentResponse
    source_chip: SourceChipResponse


class ReadReceiptItemResponse(_ApiModel):
    source_fragment_id: str
    source_version_id: str
    read_order: int
    text_sha256: str


class ReadReceiptResponse(_ApiModel):
    schema_id: Literal["dithyramba.read_receipt/1.0"] = Field(alias="schema")
    query_request_hash: str
    retrieval_corpus_hash: str
    items: tuple[ReadReceiptItemResponse, ...]
    read_receipt_id: str
    receipt_hash: str

    @classmethod
    def from_receipt(cls, receipt: ReadReceipt) -> Self:
        payload = {
            **receipt.semantic_payload(),
            "read_receipt_id": receipt.read_receipt_id,
            "receipt_hash": receipt.receipt_hash,
        }
        return cls.model_validate_json(canonical_json_bytes(payload))


class QueryExclusionsRequest(_ApiModel):
    source_ids: list[str] = Field(max_length=500)
    source_family_ids: list[str] = Field(max_length=500)
    source_fragment_ids: list[str] = Field(max_length=500)


class RetrievalBudgetRequest(_ApiModel):
    profile: Literal["fts_v1"]
    max_candidates: int = Field(ge=1, le=500)
    max_source_fragments: int = Field(ge=1, le=100)


class EvidencePacketResultContractRequest(_ApiModel):
    format: Literal["evidence_packet"]
    require_source_addresses: Literal[True]


class RecallRequest(_ApiModel):
    schema_id: Literal["dithyramba.query_request/1.0"] = Field(
        alias="schema",
    )
    question: str = Field(min_length=1, max_length=2_000)
    library_id: str = Field(pattern=r"^library_[0-9a-f]{32}$")
    collection_ids: list[str] = Field(min_length=1, max_length=16)
    corpus_snapshot_id: str = Field(pattern=r"^snapshot_[a-z0-9_]+$")
    access_policy_id: str = Field(pattern=r"^policy_[a-z0-9_]+$")
    purpose: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    exclusions: QueryExclusionsRequest
    retrieval: RetrievalBudgetRequest
    result_contract: EvidencePacketResultContractRequest

    def to_domain(self) -> QueryRequest:
        return QueryRequest(
            question=self.question,
            library_id=self.library_id,
            collection_ids=tuple(self.collection_ids),
            corpus_snapshot_id=self.corpus_snapshot_id,
            access_policy_id=self.access_policy_id,
            purpose=self.purpose,
            exclusions=QueryExclusions(
                source_ids=tuple(self.exclusions.source_ids),
                source_family_ids=tuple(self.exclusions.source_family_ids),
                source_fragment_ids=tuple(self.exclusions.source_fragment_ids),
            ),
            retrieval=RetrievalBudget(
                profile=self.retrieval.profile,
                max_candidates=self.retrieval.max_candidates,
                max_source_fragments=self.retrieval.max_source_fragments,
            ),
            result_contract=EvidencePacketResultContract(
                format=self.result_contract.format,
                require_source_addresses=self.result_contract.require_source_addresses,
            ),
        )


class RecallResultResponse(_ApiModel):
    schema_id: Literal["dithyramba.recall_result/1.0"] = Field(
        default="dithyramba.recall_result/1.0",
        alias="schema",
    )
    run: ProcessingRunResponse
    packet: EvidencePacketResponse

    @classmethod
    def from_result(cls, result: RecallResult) -> Self:
        return cls(
            run=ProcessingRunResponse.from_record(result.run),
            packet=EvidencePacketResponse.from_packet(result.packet),
        )


class ReviewScopeResponse(_ApiModel):
    schema_id: Literal["dithyramba.review_scope/1.0"] = Field(
        default="dithyramba.review_scope/1.0",
        alias="schema",
    )
    collection_ids: tuple[str, ...]
    use: str

    @classmethod
    def from_scope(cls, scope: ReviewScope) -> Self:
        return cls(collection_ids=scope.collection_ids, use=scope.use)


class ReviewScopeRequest(_ApiModel):
    schema_id: Literal["dithyramba.review_scope/1.0"] = Field(
        alias="schema",
    )
    collection_ids: list[str]
    use: str


class ReviewDecisionCreateRequest(_ApiModel):
    schema_id: Literal["dithyramba.review_decision_request/1.0"] = Field(alias="schema")
    target_type: Literal["evidence_packet", "packet_item", "source_fragment"]
    target_id: str
    target_hash: str
    action: Literal["accept", "reject", "revise", "defer", "supersede"]
    reason: str
    authority: str
    scope: ReviewScopeRequest
    supersedes_review_decision_id: str | None = None

    def to_domain(self) -> ReviewDecisionRequest:
        return ReviewDecisionRequest(
            target_type=ReviewTargetType(self.target_type),
            target_id=self.target_id,
            target_hash=self.target_hash,
            action=ReviewAction(self.action),
            reason=self.reason,
            authority=self.authority,
            scope=ReviewScope(
                collection_ids=tuple(self.scope.collection_ids),
                use=self.scope.use,
            ),
            supersedes_review_decision_id=self.supersedes_review_decision_id,
        )


class ReviewDecisionResponse(_ApiModel):
    schema_id: Literal["dithyramba.review_decision/1.0"] = Field(
        default="dithyramba.review_decision/1.0",
        alias="schema",
    )
    review_decision_id: str
    library_id: str
    target_type: ReviewTargetType
    target_id: str
    target_hash: str
    action: ReviewAction
    reason: str
    authority: str
    scope: ReviewScopeResponse
    supersedes_review_decision_id: str | None
    created_at: str

    @classmethod
    def from_decision(cls, decision: ReviewDecision) -> Self:
        return cls(
            review_decision_id=decision.review_decision_id,
            library_id=decision.library_id,
            target_type=decision.target_type,
            target_id=decision.target_id,
            target_hash=decision.target_hash,
            action=decision.action,
            reason=decision.reason,
            authority=decision.authority,
            scope=ReviewScopeResponse.from_scope(decision.scope),
            supersedes_review_decision_id=decision.supersedes_review_decision_id,
            created_at=decision.created_at,
        )


class ReviewDecisionListResponse(_ApiModel):
    schema_id: Literal["dithyramba.review_decision_list/1.0"] = Field(
        default="dithyramba.review_decision_list/1.0",
        alias="schema",
    )
    library_id: str
    items: tuple[ReviewDecisionResponse, ...]


class PacketPresentationStatusResponse(_ApiModel):
    candidate_status: Literal["candidate"] = "candidate"
    review_status: Literal["unreviewed", "has_decisions"]
    review_decision_ids: tuple[str, ...]


class PacketExportRequest(_ApiModel):
    schema_id: Literal["dithyramba.packet_export_request/1.0"] = Field(alias="schema")
    format: Literal["json", "markdown"]


def address_response(address: SourceAddress) -> SourceAddressResponse:
    if type(address) is MarkdownSourceAddress:
        return MarkdownAddressResponse(
            schema="dithyramba.source_address/1.0",
            kind="markdown",
            heading_path=address.heading_path,
            line_start=address.line_start,
            line_end=address.line_end,
            char_start=address.char_start,
            char_end=address.char_end,
        )
    if type(address) is PdfSourceAddress:
        return PdfAddressResponse(
            schema="dithyramba.source_address/1.0",
            kind="pdf",
            page=address.page,
            bbox=address.bbox,
            char_start=address.char_start,
            char_end=address.char_end,
        )
    raise TypeError("unsupported SourceAddress type")
