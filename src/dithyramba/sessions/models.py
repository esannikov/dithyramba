"""Immutable contracts for continuing, source-grounded research sessions."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_ACTOR_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_.:@/-]{0,127}$"
_TIMESTAMP_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
_SESSION_ID_PATTERN = r"^research_session_[0-9a-f]{32}$"
_BRIEF_ID_PATTERN = r"^session_brief_[0-9a-f]{32}$"
_EVENT_ID_PATTERN = r"^session_event_[0-9a-f]{32}$"
_STATE_ID_PATTERN = r"^session_state_[0-9a-f]{32}$"
GENESIS_EVENT_HASH = "0" * 64


class _SessionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class SessionActorKind(StrEnum):
    HUMAN = "human"
    MODEL = "model"
    DETERMINISTIC = "deterministic"


class SessionStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class SessionEventKind(StrEnum):
    QUESTION_ASKED = "question_asked"
    EVIDENCE_ATTACHED = "evidence_attached"
    ANSWER_DRAFTED = "answer_drafted"
    CANDIDATE_LINKED = "candidate_linked"
    GAP_RECORDED = "gap_recorded"
    PATH_REJECTED = "path_rejected"
    DECISION_RECORDED = "decision_recorded"
    SESSION_CLOSED = "session_closed"


class SessionArtifactKind(StrEnum):
    SOURCE_FRAGMENT = "source_fragment"
    EVIDENCE_PACKET = "evidence_packet"
    MEMORY_PACKET = "memory_packet"
    EVIDENCE_LINK = "evidence_link"
    STATEMENT = "statement"
    CONCEPT = "concept"
    ENTITY = "entity"
    IDEA_TRACE = "idea_trace"
    RESEARCH_ANSWER = "research_answer"
    ANSWER_PROJECTION = "answer_projection"


class ResearchSessionBrief(_SessionModel):
    """The operator's bounded purpose for one continuing investigation."""

    SCHEMA: ClassVar[str] = "dithyramba.research_session_brief/1.0"

    schema_id: str
    brief_id: str = Field(pattern=_BRIEF_ID_PATTERN)
    brief_hash: str = Field(pattern=_HASH_PATTERN)
    question: str = Field(min_length=1, max_length=4_000)
    intended_use: str = Field(min_length=1, max_length=1_000)
    success_criteria: tuple[str, ...] = Field(min_length=1, max_length=16)
    boundaries: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if len(self.success_criteria) != len(set(self.success_criteria)):
            raise ValueError("success_criteria must be unique")
        if len(self.boundaries) != len(set(self.boundaries)):
            raise ValueError("boundaries must be unique")
        payload = self.semantic_payload()
        if self.brief_id != canonical_content_id("session_brief", payload):
            raise ValueError("brief_id does not match the canonical brief content")
        if self.brief_hash != canonical_sha256_hex(payload):
            raise ValueError("brief_hash does not match the canonical brief content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "question": self.question,
            "intended_use": self.intended_use,
            "success_criteria": list(self.success_criteria),
            "boundaries": list(self.boundaries),
        }

    @classmethod
    def create(
        cls,
        *,
        question: str,
        intended_use: str,
        success_criteria: tuple[str, ...],
        boundaries: tuple[str, ...] = (),
    ) -> ResearchSessionBrief:
        payload: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "question": question,
            "intended_use": intended_use,
            "success_criteria": list(success_criteria),
            "boundaries": list(boundaries),
        }
        return cls(
            schema_id=cls.SCHEMA,
            brief_id=canonical_content_id("session_brief", payload),
            brief_hash=canonical_sha256_hex(payload),
            question=question,
            intended_use=intended_use,
            success_criteria=success_criteria,
            boundaries=boundaries,
        )


class ResearchSession(_SessionModel):
    """One immutable brief bound to an exact Library access scope."""

    SCHEMA: ClassVar[str] = "dithyramba.research_session/1.0"

    schema_id: str
    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    session_hash: str = Field(pattern=_HASH_PATTERN)
    brief: ResearchSessionBrief
    library_id: str = Field(pattern=r"^library_[a-z0-9_]+$")
    corpus_snapshot_id: str = Field(pattern=r"^snapshot_[a-z0-9_]+$")
    snapshot_hash: str = Field(pattern=_HASH_PATTERN)
    access_policy_id: str = Field(pattern=r"^policy_[a-z0-9_]+$")
    purpose: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    collection_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    created_by_kind: SessionActorKind
    created_by_id: str = Field(pattern=_ACTOR_ID_PATTERN)
    created_at: str = Field(pattern=_TIMESTAMP_PATTERN)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if self.created_by_kind is SessionActorKind.DETERMINISTIC:
            raise ValueError("a research session must be opened by a human or model actor")
        if tuple(sorted(set(self.collection_ids))) != self.collection_ids:
            raise ValueError("collection_ids must be sorted and unique")
        if any(not value.startswith("collection_") for value in self.collection_ids):
            raise ValueError("collection_ids must use the collection_ prefix")
        payload = self.semantic_payload()
        if self.session_id != canonical_content_id("research_session", payload):
            raise ValueError("session_id does not match the canonical session content")
        if self.session_hash != canonical_sha256_hex(payload):
            raise ValueError("session_hash does not match the canonical session content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "brief": self.brief.model_dump(mode="json"),
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "access_policy_id": self.access_policy_id,
            "purpose": self.purpose,
            "collection_ids": list(self.collection_ids),
            "created_by_kind": self.created_by_kind.value,
            "created_by_id": self.created_by_id,
            "created_at": self.created_at,
        }

    @classmethod
    def create(
        cls,
        *,
        brief: ResearchSessionBrief,
        library_id: str,
        corpus_snapshot_id: str,
        snapshot_hash: str,
        access_policy_id: str,
        purpose: str,
        collection_ids: tuple[str, ...],
        created_by_kind: SessionActorKind,
        created_by_id: str,
        created_at: str,
    ) -> ResearchSession:
        canonical_collections = tuple(sorted(set(collection_ids)))
        semantic: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "brief": brief.model_dump(mode="json"),
            "library_id": library_id,
            "corpus_snapshot_id": corpus_snapshot_id,
            "snapshot_hash": snapshot_hash,
            "access_policy_id": access_policy_id,
            "purpose": purpose,
            "collection_ids": list(canonical_collections),
            "created_by_kind": created_by_kind.value,
            "created_by_id": created_by_id,
            "created_at": created_at,
        }
        return cls(
            schema_id=cls.SCHEMA,
            session_id=canonical_content_id("research_session", semantic),
            session_hash=canonical_sha256_hex(semantic),
            brief=brief,
            library_id=library_id,
            corpus_snapshot_id=corpus_snapshot_id,
            snapshot_hash=snapshot_hash,
            access_policy_id=access_policy_id,
            purpose=purpose,
            collection_ids=canonical_collections,
            created_by_kind=created_by_kind,
            created_by_id=created_by_id,
            created_at=created_at,
        )


class SessionArtifactReference(_SessionModel):
    """An exact pointer to an existing artifact; never an implicit promotion."""

    artifact_kind: SessionArtifactKind
    artifact_id: str = Field(pattern=_ID_PATTERN)
    artifact_hash: str = Field(pattern=_HASH_PATTERN)
    role: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")

    @model_validator(mode="after")
    def validate_prefix(self) -> Self:
        prefix = {
            SessionArtifactKind.SOURCE_FRAGMENT: "fragment_",
            SessionArtifactKind.EVIDENCE_PACKET: "packet_",
            SessionArtifactKind.MEMORY_PACKET: "memory_packet_",
            SessionArtifactKind.EVIDENCE_LINK: "evidence_",
            SessionArtifactKind.STATEMENT: "statement_",
            SessionArtifactKind.CONCEPT: "concept_",
            SessionArtifactKind.ENTITY: "entity_",
            SessionArtifactKind.IDEA_TRACE: "idea_trace_",
            SessionArtifactKind.RESEARCH_ANSWER: "research_answer_",
            SessionArtifactKind.ANSWER_PROJECTION: "answer_projection_",
        }[self.artifact_kind]
        if not self.artifact_id.startswith(prefix):
            raise ValueError(f"{self.artifact_kind.value} artifact_id must start with {prefix}")
        return self

    def semantic_payload(self) -> dict[str, str]:
        return {
            "artifact_kind": self.artifact_kind.value,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "role": self.role,
        }


class SessionEvent(_SessionModel):
    """One content-addressed event in an append-only research journal."""

    SCHEMA: ClassVar[str] = "dithyramba.session_event/1.0"

    schema_id: str
    event_id: str = Field(pattern=_EVENT_ID_PATTERN)
    event_hash: str = Field(pattern=_HASH_PATTERN)
    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    session_hash: str = Field(pattern=_HASH_PATTERN)
    sequence: int = Field(ge=1, le=1_000_000)
    previous_event_hash: str = Field(pattern=_HASH_PATTERN)
    kind: SessionEventKind
    actor_kind: SessionActorKind
    actor_id: str = Field(pattern=_ACTOR_ID_PATTERN)
    summary: str = Field(min_length=1, max_length=8_000)
    artifact_refs: tuple[SessionArtifactReference, ...] = Field(default=(), max_length=32)
    occurred_at: str = Field(pattern=_TIMESTAMP_PATTERN)

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        reference_keys = tuple(
            (item.artifact_kind, item.artifact_id, item.artifact_hash, item.role)
            for item in self.artifact_refs
        )
        if len(reference_keys) != len(set(reference_keys)):
            raise ValueError("artifact_refs must be unique")
        if (
            self.kind
            in {
                SessionEventKind.DECISION_RECORDED,
                SessionEventKind.SESSION_CLOSED,
            }
            and self.actor_kind is not SessionActorKind.HUMAN
        ):
            raise ValueError(f"{self.kind.value} requires a human actor")
        if self.kind is SessionEventKind.EVIDENCE_ATTACHED:
            _require_artifacts(
                self.artifact_refs,
                {
                    SessionArtifactKind.SOURCE_FRAGMENT,
                    SessionArtifactKind.EVIDENCE_PACKET,
                    SessionArtifactKind.MEMORY_PACKET,
                    SessionArtifactKind.EVIDENCE_LINK,
                },
                "evidence_attached",
            )
        if self.kind is SessionEventKind.CANDIDATE_LINKED:
            _require_artifacts(
                self.artifact_refs,
                {
                    SessionArtifactKind.STATEMENT,
                    SessionArtifactKind.CONCEPT,
                    SessionArtifactKind.ENTITY,
                    SessionArtifactKind.IDEA_TRACE,
                    SessionArtifactKind.RESEARCH_ANSWER,
                    SessionArtifactKind.ANSWER_PROJECTION,
                },
                "candidate_linked",
            )
        if self.kind is SessionEventKind.SESSION_CLOSED and self.artifact_refs:
            raise ValueError("session_closed cannot attach artifacts")
        payload = self.semantic_payload()
        if self.event_id != canonical_content_id("session_event", payload):
            raise ValueError("event_id does not match the canonical event content")
        if self.event_hash != canonical_sha256_hex(payload):
            raise ValueError("event_hash does not match the canonical event content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "session_id": self.session_id,
            "session_hash": self.session_hash,
            "sequence": self.sequence,
            "previous_event_hash": self.previous_event_hash,
            "kind": self.kind.value,
            "actor_kind": self.actor_kind.value,
            "actor_id": self.actor_id,
            "summary": self.summary,
            "artifact_refs": [item.semantic_payload() for item in self.artifact_refs],
            "occurred_at": self.occurred_at,
        }

    @classmethod
    def create(
        cls,
        *,
        session: ResearchSession,
        sequence: int,
        previous_event_hash: str,
        kind: SessionEventKind,
        actor_kind: SessionActorKind,
        actor_id: str,
        summary: str,
        artifact_refs: tuple[SessionArtifactReference, ...] = (),
        occurred_at: str,
    ) -> SessionEvent:
        semantic: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "session_id": session.session_id,
            "session_hash": session.session_hash,
            "sequence": sequence,
            "previous_event_hash": previous_event_hash,
            "kind": kind.value,
            "actor_kind": actor_kind.value,
            "actor_id": actor_id,
            "summary": summary,
            "artifact_refs": [item.semantic_payload() for item in artifact_refs],
            "occurred_at": occurred_at,
        }
        return cls(
            schema_id=cls.SCHEMA,
            event_id=canonical_content_id("session_event", semantic),
            event_hash=canonical_sha256_hex(semantic),
            session_id=session.session_id,
            session_hash=session.session_hash,
            sequence=sequence,
            previous_event_hash=previous_event_hash,
            kind=kind,
            actor_kind=actor_kind,
            actor_id=actor_id,
            summary=summary,
            artifact_refs=artifact_refs,
            occurred_at=occurred_at,
        )


class SessionTextEntry(_SessionModel):
    event_id: str = Field(pattern=_EVENT_ID_PATTERN)
    text: str = Field(min_length=1, max_length=8_000)


class ResearchSessionState(_SessionModel):
    """Disposable compact state derived only from a verified event stream."""

    SCHEMA: ClassVar[str] = "dithyramba.research_session_state/1.0"

    schema_id: str
    state_id: str = Field(pattern=_STATE_ID_PATTERN)
    state_hash: str = Field(pattern=_HASH_PATTERN)
    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    session_hash: str = Field(pattern=_HASH_PATTERN)
    status: SessionStatus
    event_count: int = Field(ge=0, le=1_000_000)
    latest_event_hash: str = Field(pattern=_HASH_PATTERN)
    questions: tuple[SessionTextEntry, ...]
    answer_drafts: tuple[SessionTextEntry, ...]
    gaps: tuple[SessionTextEntry, ...]
    rejected_paths: tuple[SessionTextEntry, ...]
    decisions: tuple[SessionTextEntry, ...]
    evidence_refs: tuple[SessionArtifactReference, ...]
    candidate_refs: tuple[SessionArtifactReference, ...]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if self.event_count == 0 and self.latest_event_hash != GENESIS_EVENT_HASH:
            raise ValueError("an empty session state must retain the genesis event hash")
        if self.event_count > 0 and self.latest_event_hash == GENESIS_EVENT_HASH:
            raise ValueError("a non-empty session state requires a real latest event hash")
        payload = self.semantic_payload()
        if self.state_id != canonical_content_id("session_state", payload):
            raise ValueError("state_id does not match the canonical state content")
        if self.state_hash != canonical_sha256_hex(payload):
            raise ValueError("state_hash does not match the canonical state content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "session_id": self.session_id,
            "session_hash": self.session_hash,
            "status": self.status.value,
            "event_count": self.event_count,
            "latest_event_hash": self.latest_event_hash,
            "questions": [item.model_dump(mode="json") for item in self.questions],
            "answer_drafts": [item.model_dump(mode="json") for item in self.answer_drafts],
            "gaps": [item.model_dump(mode="json") for item in self.gaps],
            "rejected_paths": [item.model_dump(mode="json") for item in self.rejected_paths],
            "decisions": [item.model_dump(mode="json") for item in self.decisions],
            "evidence_refs": [item.semantic_payload() for item in self.evidence_refs],
            "candidate_refs": [item.semantic_payload() for item in self.candidate_refs],
        }

    @classmethod
    def create(
        cls,
        *,
        session: ResearchSession,
        status: SessionStatus,
        event_count: int,
        latest_event_hash: str,
        questions: tuple[SessionTextEntry, ...] = (),
        answer_drafts: tuple[SessionTextEntry, ...] = (),
        gaps: tuple[SessionTextEntry, ...] = (),
        rejected_paths: tuple[SessionTextEntry, ...] = (),
        decisions: tuple[SessionTextEntry, ...] = (),
        evidence_refs: tuple[SessionArtifactReference, ...] = (),
        candidate_refs: tuple[SessionArtifactReference, ...] = (),
    ) -> ResearchSessionState:
        semantic: dict[str, object] = {
            "schema_id": cls.SCHEMA,
            "session_id": session.session_id,
            "session_hash": session.session_hash,
            "status": status.value,
            "event_count": event_count,
            "latest_event_hash": latest_event_hash,
            "questions": [item.model_dump(mode="json") for item in questions],
            "answer_drafts": [item.model_dump(mode="json") for item in answer_drafts],
            "gaps": [item.model_dump(mode="json") for item in gaps],
            "rejected_paths": [item.model_dump(mode="json") for item in rejected_paths],
            "decisions": [item.model_dump(mode="json") for item in decisions],
            "evidence_refs": [item.semantic_payload() for item in evidence_refs],
            "candidate_refs": [item.semantic_payload() for item in candidate_refs],
        }
        return cls(
            schema_id=cls.SCHEMA,
            state_id=canonical_content_id("session_state", semantic),
            state_hash=canonical_sha256_hex(semantic),
            session_id=session.session_id,
            session_hash=session.session_hash,
            status=status,
            event_count=event_count,
            latest_event_hash=latest_event_hash,
            questions=questions,
            answer_drafts=answer_drafts,
            gaps=gaps,
            rejected_paths=rejected_paths,
            decisions=decisions,
            evidence_refs=evidence_refs,
            candidate_refs=candidate_refs,
        )


def _require_artifacts(
    references: tuple[SessionArtifactReference, ...],
    allowed: set[SessionArtifactKind],
    label: str,
) -> None:
    if not references:
        raise ValueError(f"{label} requires at least one artifact reference")
    disallowed = {item.artifact_kind for item in references}.difference(allowed)
    if disallowed:
        names = ", ".join(sorted(item.value for item in disallowed))
        raise ValueError(f"{label} contains disallowed artifact kinds: {names}")
