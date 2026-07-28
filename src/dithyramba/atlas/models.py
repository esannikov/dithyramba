"""Strict interchange contract for a source-grounded Research Atlas."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import ClassVar, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_DATE_PATTERN = r"^[0-9?]{4}(?:-[0-9?]{2})?(?:-[0-9?]{2})?$"


class _AtlasModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SourceKind(StrEnum):
    LETTER = "letter"
    DIARY = "diary"
    CONTRACT = "contract"
    PATENT = "patent"
    ARCHIVAL_RECORD = "archival_record"
    ARTWORK = "artwork"
    PHOTOGRAPH = "photograph"
    NEWSPAPER = "newspaper"
    SCHOLARLY_ARTICLE = "scholarly_article"
    BOOK = "book"
    DATASET = "dataset"
    INSTITUTIONAL_RECORD = "institutional_record"
    OTHER = "other"


class VoiceKind(StrEnum):
    FIRST_PERSON = "first_person"
    WITNESS = "witness"
    INSTITUTIONAL = "institutional"
    SCHOLARLY = "scholarly"
    JOURNALISTIC = "journalistic"
    CURATORIAL = "curatorial"
    UNKNOWN = "unknown"


class VerificationState(StrEnum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"


class EvidenceRole(StrEnum):
    SUPPORTS = "supports"
    QUALIFIES = "qualifies"
    REFUTES = "refutes"
    CONTEXT = "context"


class FindingState(StrEnum):
    CONFIRMED = "confirmed"
    QUALIFIED = "qualified"
    CONTESTED = "contested"
    REFUTED = "refuted"
    OPEN = "open"


class HypothesisState(StrEnum):
    ESTABLISHED = "established"
    WORKING = "working"
    CONTESTED = "contested"
    REFUTED = "refuted"
    OPEN = "open"


class RelationKind(StrEnum):
    SUPPORTS = "supports"
    QUALIFIES = "qualifies"
    REFUTES = "refutes"
    PRECEDES = "precedes"
    INFLUENCES = "influences"
    PARALLELS = "parallels"


class AtlasSource(_AtlasModel):
    source_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    creator: str = Field(min_length=1, max_length=300)
    source_kind: SourceKind
    voice_kind: VoiceKind
    date_label: str | None = Field(default=None, max_length=120)
    publisher_or_archive: str | None = Field(default=None, max_length=300)
    independence_group: str = Field(pattern=_ID_PATTERN)
    verification_state: VerificationState
    original_url: str | None = Field(default=None, max_length=2_000)
    artifact_path: str | None = Field(default=None, max_length=1_000)
    rights_note: str = Field(min_length=1, max_length=1_000)

    @field_validator("original_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("original_url must be an absolute HTTP(S) URL")
        return value

    @field_validator("artifact_path")
    @classmethod
    def validate_artifact_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact_path must be a case-relative path")
        return value


class AtlasSourceAddress(_AtlasModel):
    kind: str = Field(min_length=1, max_length=80)
    heading_path: tuple[str, ...] = Field(default=(), max_length=32)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered_ranges(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("source address line range is reversed")
        if self.char_end < self.char_start:
            raise ValueError("source address character range is reversed")
        return self


class AtlasEvidence(_AtlasModel):
    evidence_id: str = Field(pattern=_ID_PATTERN)
    source_id: str = Field(pattern=_ID_PATTERN)
    locator: str = Field(min_length=1, max_length=500)
    excerpt: str | None = Field(default=None, max_length=1_000)
    role: EvidenceRole
    asserting_voice: str = Field(min_length=1, max_length=300)
    limitation: str | None = Field(default=None, max_length=2_000)
    source_fragment_id: str | None = Field(
        default=None,
        pattern=r"^fragment_[a-z0-9_]+$",
    )
    fragment_text_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_address: AtlasSourceAddress | None = None

    @model_validator(mode="after")
    def exact_binding_is_complete(self) -> Self:
        binding = (
            self.source_fragment_id,
            self.fragment_text_sha256,
            self.source_address,
        )
        if any(item is not None for item in binding) and any(item is None for item in binding):
            raise ValueError("exact evidence binding must be complete")
        return self


class AtlasQuestion(_AtlasModel):
    question_id: str = Field(pattern=_ID_PATTERN)
    prompt: str = Field(min_length=1, max_length=2_000)
    short_answer: str = Field(min_length=1, max_length=8_000)
    state: FindingState
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=32)
    gap: str | None = Field(default=None, max_length=2_000)
    tags: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def evidence_boundary(self) -> Self:
        if self.state is not FindingState.OPEN and not self.evidence_ids:
            raise ValueError("non-open question requires evidence")
        return self


class AtlasHypothesis(_AtlasModel):
    hypothesis_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    synthesis: str = Field(min_length=1, max_length=8_000)
    state: HypothesisState
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=48)
    counterevidence_ids: tuple[str, ...] = Field(default=(), max_length=24)
    gap: str | None = Field(default=None, max_length=2_000)
    question_ids: tuple[str, ...] = Field(default=(), max_length=24)

    @model_validator(mode="after")
    def evidence_boundary(self) -> Self:
        if self.state is HypothesisState.ESTABLISHED and not self.evidence_ids:
            raise ValueError("established hypothesis requires evidence")
        if set(self.evidence_ids).intersection(self.counterevidence_ids):
            raise ValueError("one evidence item cannot support and counter the same hypothesis")
        return self


class AtlasRelation(_AtlasModel):
    relation_id: str = Field(pattern=_ID_PATTERN)
    source_hypothesis_id: str = Field(pattern=_ID_PATTERN)
    target_hypothesis_id: str = Field(pattern=_ID_PATTERN)
    kind: RelationKind
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=24)
    explanation: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def no_self_relation(self) -> Self:
        if self.source_hypothesis_id == self.target_hypothesis_id:
            raise ValueError("hypothesis relation cannot target itself")
        return self


class AtlasTimelineEvent(_AtlasModel):
    event_id: str = Field(pattern=_ID_PATTERN)
    date_start: str = Field(pattern=_DATE_PATTERN)
    date_end: str | None = Field(default=None, pattern=_DATE_PATTERN)
    date_label: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=4_000)
    state: FindingState
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=32)
    question_ids: tuple[str, ...] = Field(default=(), max_length=16)
    hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def evidence_boundary(self) -> Self:
        if self.state is not FindingState.OPEN and not self.evidence_ids:
            raise ValueError("non-open timeline event requires evidence")
        return self


class AtlasGap(_AtlasModel):
    gap_id: str = Field(pattern=_ID_PATTERN)
    label: str = Field(min_length=1, max_length=500)
    why_it_matters: str = Field(min_length=1, max_length=2_000)
    next_evidence: str = Field(min_length=1, max_length=2_000)
    related_question_ids: tuple[str, ...] = Field(default=(), max_length=24)
    related_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=24)


class AtlasCorpus(_AtlasModel):
    label: str = Field(min_length=1, max_length=300)
    source_count: int = Field(ge=0)
    source_family_count: int = Field(ge=0)
    scope_note: str = Field(min_length=1, max_length=2_000)
    cutoff_note: str = Field(min_length=1, max_length=2_000)


class ResearchAtlasManifest(_AtlasModel):
    SCHEMA: ClassVar[str] = "dithyramba.research_atlas/1.0"

    schema_id: str
    atlas_id: str = Field(pattern=_ID_PATTERN)
    case_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    subject: str = Field(min_length=1, max_length=300)
    subtitle: str = Field(min_length=1, max_length=1_000)
    generated_at: datetime
    language: str = Field(pattern=r"^[a-z]{2}(?:-[A-Z]{2})?$")
    read_only: bool
    corpus: AtlasCorpus
    sources: tuple[AtlasSource, ...] = Field(max_length=2_000)
    evidence: tuple[AtlasEvidence, ...] = Field(max_length=10_000)
    questions: tuple[AtlasQuestion, ...] = Field(max_length=1_000)
    hypotheses: tuple[AtlasHypothesis, ...] = Field(max_length=1_000)
    relations: tuple[AtlasRelation, ...] = Field(default=(), max_length=4_000)
    timeline: tuple[AtlasTimelineEvent, ...] = Field(max_length=4_000)
    gaps: tuple[AtlasGap, ...] = Field(default=(), max_length=1_000)
    methodology_note: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if not self.read_only:
            raise ValueError("Research Atlas manifests are read-only")

        source_ids = _unique_ids(item.source_id for item in self.sources)
        evidence_ids = _unique_ids(item.evidence_id for item in self.evidence)
        question_ids = _unique_ids(item.question_id for item in self.questions)
        hypothesis_ids = _unique_ids(item.hypothesis_id for item in self.hypotheses)
        _unique_ids(item.relation_id for item in self.relations)
        _unique_ids(item.event_id for item in self.timeline)
        _unique_ids(item.gap_id for item in self.gaps)

        for evidence in self.evidence:
            _require_subset((evidence.source_id,), source_ids, "evidence source")
        for question in self.questions:
            _require_subset(question.evidence_ids, evidence_ids, "question evidence")
        for hypothesis in self.hypotheses:
            _require_subset(hypothesis.evidence_ids, evidence_ids, "hypothesis evidence")
            _require_subset(
                hypothesis.counterevidence_ids,
                evidence_ids,
                "hypothesis counterevidence",
            )
            _require_subset(hypothesis.question_ids, question_ids, "hypothesis question")
        for relation in self.relations:
            _require_subset(
                (relation.source_hypothesis_id, relation.target_hypothesis_id),
                hypothesis_ids,
                "relation hypothesis",
            )
            _require_subset(relation.evidence_ids, evidence_ids, "relation evidence")
        for event in self.timeline:
            _require_subset(event.evidence_ids, evidence_ids, "timeline evidence")
            _require_subset(event.question_ids, question_ids, "timeline question")
            _require_subset(event.hypothesis_ids, hypothesis_ids, "timeline hypothesis")
        for gap in self.gaps:
            _require_subset(gap.related_question_ids, question_ids, "gap question")
            _require_subset(gap.related_hypothesis_ids, hypothesis_ids, "gap hypothesis")

        if self.corpus.source_count != len(self.sources):
            raise ValueError("corpus.source_count must equal the manifest source count")
        family_count = len({item.independence_group for item in self.sources})
        if self.corpus.source_family_count != family_count:
            raise ValueError(
                "corpus.source_family_count must equal the distinct independence groups"
            )
        return self


def _unique_ids(values: Iterable[str]) -> frozenset[str]:
    identifiers = tuple(values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Atlas identifiers must be unique within their object type")
    return frozenset(identifiers)


def _require_subset(values: tuple[str, ...], valid: frozenset[str], label: str) -> None:
    missing = sorted(set(values).difference(valid))
    if missing:
        raise ValueError(f"{label} references missing IDs: {', '.join(missing)}")
