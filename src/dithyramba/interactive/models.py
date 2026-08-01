"""Compact, content-addressed projections for an agent research turn."""

from __future__ import annotations

from typing import ClassVar, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from dithyramba.contracts import canonical_sha256_hex
from dithyramba.recall import (
    EvidenceFragment,
    EvidencePacket,
    PacketResultStatus,
    RecallOmission,
)
from dithyramba.sessions import (
    ResearchSession,
    ResearchSessionBrief,
    ResearchSessionState,
    SessionArtifactReference,
    SessionStatus,
    SessionTextEntry,
)

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_SESSION_ID_PATTERN = r"^research_session_[0-9a-f]{32}$"
_EVENT_ID_PATTERN = r"^session_event_[0-9a-f]{32}$"
_COMMAND_ID_PATTERN = r"^command_[a-z0-9]+(?:_[a-z0-9]+)*$"
_T = TypeVar("_T")


class _InteractiveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SessionContextBudget(_InteractiveModel):
    """Maximum whole journal items exposed in one disposable context packet."""

    recent_questions: int = Field(default=3, ge=1, le=20)
    recent_drafts: int = Field(default=1, ge=0, le=10)
    recent_gaps: int = Field(default=5, ge=0, le=20)
    recent_rejected_paths: int = Field(default=3, ge=0, le=20)
    recent_decisions: int = Field(default=5, ge=0, le=20)
    evidence_references: int = Field(default=20, ge=0, le=100)
    candidate_references: int = Field(default=20, ge=0, le=100)

    def payload(self) -> dict[str, int]:
        return self.model_dump(mode="json")


class SessionContextOmissions(_InteractiveModel):
    """Explicit counts for session material left outside one context packet."""

    questions: int = Field(ge=0)
    drafts: int = Field(ge=0)
    gaps: int = Field(ge=0)
    rejected_paths: int = Field(ge=0)
    decisions: int = Field(ge=0)
    evidence_references: int = Field(ge=0)
    candidate_references: int = Field(ge=0)


class AgentSessionContext(_InteractiveModel):
    """Small derived session state; exact evidence text travels in a packet."""

    SCHEMA: ClassVar[str] = "dithyramba.agent_session_context/1.0"

    schema_id: str
    context_hash: str = Field(pattern=_HASH_PATTERN)
    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    session_hash: str = Field(pattern=_HASH_PATTERN)
    state_hash: str = Field(pattern=_HASH_PATTERN)
    library_id: str
    corpus_snapshot_id: str
    access_policy_id: str
    scope_hash: str = Field(pattern=_HASH_PATTERN)
    collection_ids: tuple[str, ...]
    status: SessionStatus
    brief: ResearchSessionBrief
    questions: tuple[SessionTextEntry, ...]
    drafts: tuple[SessionTextEntry, ...]
    gaps: tuple[SessionTextEntry, ...]
    rejected_paths: tuple[SessionTextEntry, ...]
    decisions: tuple[SessionTextEntry, ...]
    evidence_references: tuple[SessionArtifactReference, ...]
    candidate_references: tuple[SessionArtifactReference, ...]
    omissions: SessionContextOmissions
    budget: SessionContextBudget

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if self.context_hash != canonical_sha256_hex(self.semantic_payload()):
            raise ValueError("context_hash does not match the compact session context")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "session_id": self.session_id,
            "session_hash": self.session_hash,
            "state_hash": self.state_hash,
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "scope_hash": self.scope_hash,
            "collection_ids": list(self.collection_ids),
            "status": self.status.value,
            "brief": self.brief.model_dump(mode="json"),
            "questions": [item.model_dump(mode="json") for item in self.questions],
            "drafts": [item.model_dump(mode="json") for item in self.drafts],
            "gaps": [item.model_dump(mode="json") for item in self.gaps],
            "rejected_paths": [item.model_dump(mode="json") for item in self.rejected_paths],
            "decisions": [item.model_dump(mode="json") for item in self.decisions],
            "evidence_references": [
                item.model_dump(mode="json") for item in self.evidence_references
            ],
            "candidate_references": [
                item.model_dump(mode="json") for item in self.candidate_references
            ],
            "omissions": self.omissions.model_dump(mode="json"),
            "budget": self.budget.payload(),
        }

    @classmethod
    def create(
        cls,
        *,
        session: ResearchSession,
        state: ResearchSessionState,
        budget: SessionContextBudget | None = None,
    ) -> AgentSessionContext:
        selected_budget = budget or SessionContextBudget()
        if state.session_id != session.session_id or state.session_hash != session.session_hash:
            raise ValueError("session state is bound to another ResearchSession")
        questions, omitted_questions = _tail(state.questions, selected_budget.recent_questions)
        drafts, omitted_drafts = _tail(state.answer_drafts, selected_budget.recent_drafts)
        gaps, omitted_gaps = _tail(state.gaps, selected_budget.recent_gaps)
        rejected, omitted_rejected = _tail(
            state.rejected_paths,
            selected_budget.recent_rejected_paths,
        )
        decisions, omitted_decisions = _tail(
            state.decisions,
            selected_budget.recent_decisions,
        )
        evidence, omitted_evidence = _tail(
            state.evidence_refs,
            selected_budget.evidence_references,
        )
        candidates, omitted_candidates = _tail(
            state.candidate_refs,
            selected_budget.candidate_references,
        )
        omissions = SessionContextOmissions(
            questions=omitted_questions,
            drafts=omitted_drafts,
            gaps=omitted_gaps,
            rejected_paths=omitted_rejected,
            decisions=omitted_decisions,
            evidence_references=omitted_evidence,
            candidate_references=omitted_candidates,
        )
        semantic: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "session_id": session.session_id,
            "session_hash": session.session_hash,
            "state_hash": state.state_hash,
            "library_id": session.library_id,
            "corpus_snapshot_id": session.corpus_snapshot_id,
            "access_policy_id": session.access_policy_id,
            "scope_hash": session.scope_hash,
            "collection_ids": list(session.collection_ids),
            "status": state.status.value,
            "brief": session.brief.model_dump(mode="json"),
            "questions": [item.model_dump(mode="json") for item in questions],
            "drafts": [item.model_dump(mode="json") for item in drafts],
            "gaps": [item.model_dump(mode="json") for item in gaps],
            "rejected_paths": [item.model_dump(mode="json") for item in rejected],
            "decisions": [item.model_dump(mode="json") for item in decisions],
            "evidence_references": [item.model_dump(mode="json") for item in evidence],
            "candidate_references": [item.model_dump(mode="json") for item in candidates],
            "omissions": omissions.model_dump(mode="json"),
            "budget": selected_budget.payload(),
        }
        return cls(
            schema_id=cls.SCHEMA,
            context_hash=canonical_sha256_hex(semantic),
            session_id=session.session_id,
            session_hash=session.session_hash,
            state_hash=state.state_hash,
            library_id=session.library_id,
            corpus_snapshot_id=session.corpus_snapshot_id,
            access_policy_id=session.access_policy_id,
            scope_hash=session.scope_hash,
            collection_ids=session.collection_ids,
            status=state.status,
            brief=session.brief,
            questions=questions,
            drafts=drafts,
            gaps=gaps,
            rejected_paths=rejected,
            decisions=decisions,
            evidence_references=evidence,
            candidate_references=candidates,
            omissions=omissions,
            budget=selected_budget,
        )


class AgentEvidencePacket(_InteractiveModel):
    """Least-context projection of a full, locally retained EvidencePacket."""

    SCHEMA: ClassVar[str] = "dithyramba.agent_evidence_packet/1.0"

    schema_id: str
    projection_hash: str = Field(pattern=_HASH_PATTERN)
    evidence_packet_id: str
    evidence_packet_hash: str = Field(pattern=_HASH_PATTERN)
    query_request_id: str
    query_request_hash: str = Field(pattern=_HASH_PATTERN)
    corpus_snapshot_id: str
    corpus_snapshot_hash: str = Field(pattern=_HASH_PATTERN)
    result_status: PacketResultStatus
    source_fragments: tuple[EvidenceFragment, ...]
    coverage_report_id: str
    coverage_report_hash: str = Field(pattern=_HASH_PATTERN)
    processed_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    policy_omission_present: bool
    omissions: tuple[RecallOmission, ...]
    retrieval_candidate_count: int = Field(ge=0)
    retrieval_selected_count: int = Field(ge=0)
    read_receipt_id: str
    read_receipt_hash: str = Field(pattern=_HASH_PATTERN)
    access_receipt_id: str
    access_receipt_hash: str = Field(pattern=_HASH_PATTERN)
    retrieval_receipt_id: str
    retrieval_receipt_hash: str = Field(pattern=_HASH_PATTERN)

    @field_serializer("source_fragments")
    def serialize_source_fragments(
        self,
        value: tuple[EvidenceFragment, ...],
    ) -> list[dict[str, object]]:
        return [item.payload() for item in value]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if self.projection_hash != canonical_sha256_hex(self.semantic_payload()):
            raise ValueError("projection_hash does not match the agent evidence packet")
        has_evidence = bool(self.source_fragments)
        if has_evidence != (self.result_status is PacketResultStatus.EVIDENCE_FOUND):
            raise ValueError("result_status must match projected source fragments")
        if self.retrieval_selected_count != len(self.source_fragments):
            raise ValueError("retrieval_selected_count must match projected fragments")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "evidence_packet_id": self.evidence_packet_id,
            "evidence_packet_hash": self.evidence_packet_hash,
            "query_request_id": self.query_request_id,
            "query_request_hash": self.query_request_hash,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "corpus_snapshot_hash": self.corpus_snapshot_hash,
            "result_status": self.result_status.value,
            "source_fragments": [item.payload() for item in self.source_fragments],
            "coverage": {
                "coverage_report_id": self.coverage_report_id,
                "coverage_report_hash": self.coverage_report_hash,
                "processed_count": self.processed_count,
                "skipped_count": self.skipped_count,
                "failed_count": self.failed_count,
                "policy_omission_present": self.policy_omission_present,
                "omissions": [item.payload() for item in self.omissions],
            },
            "retrieval": {
                "candidate_count": self.retrieval_candidate_count,
                "selected_count": self.retrieval_selected_count,
            },
            "receipts": {
                "read_receipt_id": self.read_receipt_id,
                "read_receipt_hash": self.read_receipt_hash,
                "access_receipt_id": self.access_receipt_id,
                "access_receipt_hash": self.access_receipt_hash,
                "retrieval_receipt_id": self.retrieval_receipt_id,
                "retrieval_receipt_hash": self.retrieval_receipt_hash,
            },
        }

    @classmethod
    def create(cls, packet: EvidencePacket) -> AgentEvidencePacket:
        if type(packet) is not EvidencePacket:
            raise TypeError("AgentEvidencePacket requires an exact EvidencePacket")
        semantic: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "evidence_packet_id": packet.evidence_packet_id,
            "evidence_packet_hash": packet.packet_hash,
            "query_request_id": packet.query_request_id,
            "query_request_hash": packet.query_request_hash,
            "corpus_snapshot_id": packet.corpus_snapshot_id,
            "corpus_snapshot_hash": packet.corpus_snapshot_hash,
            "result_status": packet.result_status.value,
            "source_fragments": [item.payload() for item in packet.source_fragments],
            "coverage": {
                "coverage_report_id": packet.coverage_report.coverage_report_id,
                "coverage_report_hash": packet.coverage_report.report_hash,
                "processed_count": packet.coverage_report.processed_count,
                "skipped_count": packet.coverage_report.skipped_count,
                "failed_count": packet.coverage_report.failed_count,
                "policy_omission_present": packet.coverage_report.policy_omission_present,
                "omissions": [item.payload() for item in packet.coverage_report.omissions],
            },
            "retrieval": {
                "candidate_count": packet.retrieval_receipt.candidate_count,
                "selected_count": packet.retrieval_receipt.selected_count,
            },
            "receipts": {
                "read_receipt_id": packet.read_receipt.read_receipt_id,
                "read_receipt_hash": packet.read_receipt.receipt_hash,
                "access_receipt_id": packet.access_receipt.access_receipt_id,
                "access_receipt_hash": packet.access_receipt.receipt_hash,
                "retrieval_receipt_id": packet.retrieval_receipt.retrieval_receipt_id,
                "retrieval_receipt_hash": packet.retrieval_receipt.receipt_hash,
            },
        }
        return cls(
            schema_id=cls.SCHEMA,
            projection_hash=canonical_sha256_hex(semantic),
            evidence_packet_id=packet.evidence_packet_id,
            evidence_packet_hash=packet.packet_hash,
            query_request_id=packet.query_request_id,
            query_request_hash=packet.query_request_hash,
            corpus_snapshot_id=packet.corpus_snapshot_id,
            corpus_snapshot_hash=packet.corpus_snapshot_hash,
            result_status=packet.result_status,
            source_fragments=packet.source_fragments,
            coverage_report_id=packet.coverage_report.coverage_report_id,
            coverage_report_hash=packet.coverage_report.report_hash,
            processed_count=packet.coverage_report.processed_count,
            skipped_count=packet.coverage_report.skipped_count,
            failed_count=packet.coverage_report.failed_count,
            policy_omission_present=packet.coverage_report.policy_omission_present,
            omissions=packet.coverage_report.omissions,
            retrieval_candidate_count=packet.retrieval_receipt.candidate_count,
            retrieval_selected_count=packet.retrieval_receipt.selected_count,
            read_receipt_id=packet.read_receipt.read_receipt_id,
            read_receipt_hash=packet.read_receipt.receipt_hash,
            access_receipt_id=packet.access_receipt.access_receipt_id,
            access_receipt_hash=packet.access_receipt.receipt_hash,
            retrieval_receipt_id=packet.retrieval_receipt.retrieval_receipt_id,
            retrieval_receipt_hash=packet.retrieval_receipt.receipt_hash,
        )


class AgentResearchTurn(_InteractiveModel):
    """One compact evidence projection plus the compact state that follows it."""

    SCHEMA: ClassVar[str] = "dithyramba.agent_research_turn/1.2"
    LEGACY_SCHEMA: ClassVar[str] = "dithyramba.agent_research_turn/1.1"

    schema_id: str
    turn_hash: str = Field(pattern=_HASH_PATTERN)
    command_id: str = Field(pattern=_COMMAND_ID_PATTERN, max_length=96)
    question_event_id: str = Field(pattern=_EVENT_ID_PATTERN)
    evidence_event_id: str = Field(pattern=_EVENT_ID_PATTERN)
    evidence_packet: AgentEvidencePacket
    context: AgentSessionContext

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.schema_id not in (self.SCHEMA, self.LEGACY_SCHEMA):
            raise ValueError(f"schema_id must be {self.SCHEMA} or {self.LEGACY_SCHEMA}")
        if self.turn_hash != canonical_sha256_hex(self.semantic_payload()):
            raise ValueError("turn_hash does not match the agent research turn")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "command_id": self.command_id,
            "question_event_id": self.question_event_id,
            "evidence_event_id": self.evidence_event_id,
            "evidence_packet_id": self.evidence_packet.evidence_packet_id,
            "evidence_packet_hash": self.evidence_packet.evidence_packet_hash,
            "context_hash": self.context.context_hash,
        }

    @classmethod
    def create(
        cls,
        *,
        command_id: str,
        question_event_id: str,
        evidence_event_id: str,
        evidence_packet: AgentEvidencePacket,
        context: AgentSessionContext,
        schema_id: str | None = None,
    ) -> AgentResearchTurn:
        selected_schema = schema_id or cls.SCHEMA
        semantic: dict[str, object] = {
            "schema_id": selected_schema,
            "command_id": command_id,
            "question_event_id": question_event_id,
            "evidence_event_id": evidence_event_id,
            "evidence_packet_id": evidence_packet.evidence_packet_id,
            "evidence_packet_hash": evidence_packet.evidence_packet_hash,
            "context_hash": context.context_hash,
        }
        return cls(
            schema_id=selected_schema,
            turn_hash=canonical_sha256_hex(semantic),
            command_id=command_id,
            question_event_id=question_event_id,
            evidence_event_id=evidence_event_id,
            evidence_packet=evidence_packet,
            context=context,
        )


def _tail(values: tuple[_T, ...], limit: int) -> tuple[tuple[_T, ...], int]:
    selected = values[-limit:] if limit else ()
    return selected, len(values) - len(selected)
