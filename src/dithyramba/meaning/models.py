"""Immutable typed contracts for Dithyramba's P6 meaning slice."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from dithyramba.access import RequestScope
from dithyramba.contracts import canonical_sha256_hex, sha256_hex

from .errors import MeaningContractError

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]+$")
_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)


class MeaningRunKind(StrEnum):
    """The two P6 ProcessingRun kinds."""

    EXTRACT = "meaning_extract"
    IMPORT = "meaning_import"


class MeaningRunStatus(StrEnum):
    """Terminal, immutable semantic ProcessingRun states."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    REFUSED = "refused"


class ExtractionProfile(StrEnum):
    """Frozen P6 extraction depth profiles."""

    INDEX = "index"
    LIGHT = "light"
    DEEP = "deep"

    @property
    def source_version_candidate_cap(self) -> int:
        """Return the frozen per-SourceVersion candidate cap."""

        return {self.INDEX: 12, self.LIGHT: 24, self.DEEP: 60}[self]


class CoverageState(StrEnum):
    """P6 CoverageReport terminal states."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    REFUSED = "refused"


class StatementKind(StrEnum):
    """Frozen Statement kinds."""

    SOURCE_CLAIM = "source_claim"
    OBSERVATION = "observation"
    INTERPRETATION = "interpretation"
    HYPOTHESIS = "hypothesis"
    PROPOSAL = "proposal"
    RULE = "rule"


class VoiceKind(StrEnum):
    """Frozen Voice kinds."""

    AUTHOR = "author"
    NARRATOR = "narrator"
    CHARACTER = "character"
    INSTITUTION = "institution"
    SCHOOL = "school"
    COLLECTIVE = "collective"


class EntityKind(StrEnum):
    """Frozen Entity kinds."""

    PERSON = "person"
    ORGANIZATION = "organization"
    WORK = "work"
    PLACE = "place"
    EVENT = "event"
    OBJECT = "object"


class EvidencePolarity(StrEnum):
    """How exact evidence bears on a Statement."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"


class SemanticAlignment(StrEnum):
    """Strength of semantic alignment without a confidence float."""

    EXACT = "exact"
    PARTIAL = "partial"
    CONTEXTUAL = "contextual"


class OmissionCategory(StrEnum):
    """Coverage omissions that are explicitly not EvidenceGaps."""

    BUDGET = "budget"
    PROFILE = "profile"
    FAILURE = "failure"
    UNSUPPORTED = "unsupported"


class MeaningReviewTargetType(StrEnum):
    """Typed semantic ReviewDecision targets."""

    STATEMENT = "statement"
    EVIDENCE_LINK = "evidence_link"
    VOICE = "voice"
    ENTITY = "entity"
    CONCEPT = "concept"
    CONCEPT_MEANING = "concept_meaning"
    TIME_CONTEXT = "time_context"
    RELATION = "relation"


class MeaningReviewAction(StrEnum):
    """Append-only semantic ReviewDecision actions."""

    ACCEPT = "accept"
    REJECT = "reject"
    REVISE = "revise"
    DEFER = "defer"
    SUPERSEDE = "supersede"


@dataclass(frozen=True, slots=True)
class ReviewBudget:
    """Explicit extraction, backlog, read, and human-review limits."""

    max_candidates: int = 400
    max_read_fragments: int = 2_000
    max_unresolved_candidates: int = 1_000
    review_decision_limit: int = 30
    budget_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_int_range(self.max_candidates, 1, 400, "max_candidates")
        _require_int_range(self.max_read_fragments, 1, 2_000, "max_read_fragments")
        _require_int_range(
            self.max_unresolved_candidates,
            1,
            1_000_000,
            "max_unresolved_candidates",
        )
        _require_int_range(self.review_decision_limit, 1, 10_000, "review_decision_limit")
        object.__setattr__(self, "budget_hash", canonical_sha256_hex(self.payload()))

    @property
    def has_explicit_backlog_override(self) -> bool:
        """Return whether the caller raised the frozen hard backlog ceiling."""

        return self.max_unresolved_candidates > 1_000

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.review_budget/1.0",
            "max_candidates": self.max_candidates,
            "max_read_fragments": self.max_read_fragments,
            "max_unresolved_candidates": self.max_unresolved_candidates,
            "review_decision_limit": self.review_decision_limit,
        }


@dataclass(frozen=True, slots=True)
class GeneratorProfile:
    """Explicit generator/model receipt fields; never ambient configuration."""

    generator_id: str
    revision: str
    license: str
    prompt_version: str
    dimensions: int | None = None
    external_provider: bool = False
    generator_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for value, label in (
            (self.generator_id, "generator_id"),
            (self.revision, "revision"),
            (self.license, "license"),
            (self.prompt_version, "prompt_version"),
        ):
            _require_text(value, label)
        if self.dimensions is not None:
            _require_int_range(self.dimensions, 1, 1_000_000, "dimensions")
        if type(self.external_provider) is not bool:
            raise MeaningContractError("external_provider must be bool")
        object.__setattr__(self, "generator_hash", canonical_sha256_hex(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.meaning_generator/1.0",
            "generator_id": self.generator_id,
            "revision": self.revision,
            "license": self.license,
            "prompt_version": self.prompt_version,
            "dimensions": self.dimensions,
            "external_provider": self.external_provider,
        }


@dataclass(frozen=True, slots=True)
class MeaningRunRequest:
    """One policy-, snapshot-, profile-, generator-, and budget-bound P6 request."""

    corpus_snapshot_id: str
    access_policy_id: str
    scope: RequestScope
    profile: ExtractionProfile
    generator: GeneratorProfile
    review_budget: ReviewBudget
    code_version: str

    def __post_init__(self) -> None:
        _require_prefixed_id(self.corpus_snapshot_id, "snapshot_", "corpus_snapshot_id")
        _require_prefixed_id(self.access_policy_id, "policy_", "access_policy_id")
        if not isinstance(self.scope, RequestScope):
            raise MeaningContractError("scope must be a RequestScope")
        if not isinstance(self.profile, ExtractionProfile):
            raise MeaningContractError("profile must be an ExtractionProfile")
        if not isinstance(self.generator, GeneratorProfile):
            raise MeaningContractError("generator must be a GeneratorProfile")
        if not isinstance(self.review_budget, ReviewBudget):
            raise MeaningContractError("review_budget must be a ReviewBudget")
        _require_text(self.code_version, "code_version")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.meaning_run_request/2.0",
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "scope": {
                "library_id": self.scope.library_id,
                "snapshot_hash": self.scope.snapshot_hash,
                "purpose": self.scope.purpose,
                "collection_ids": list(self.scope.collection_ids),
                "exclusions": self.scope.exclusions.payload(),
            },
            "profile": self.profile.value,
            "generator": self.generator.payload(),
            "review_budget": self.review_budget.payload(),
            "code_version": self.code_version,
        }


@dataclass(frozen=True, slots=True)
class MeaningInputFragment:
    """Permitted text supplied to a semantic generator after policy compilation."""

    source_fragment_id: str
    source_version_id: str
    source_id: str
    collection_ids: tuple[str, ...]
    text: str
    text_sha256: str

    def __post_init__(self) -> None:
        _require_prefixed_id(self.source_fragment_id, "fragment_", "source_fragment_id")
        _require_prefixed_id(self.source_version_id, "source_version_", "source_version_id")
        _require_prefixed_id(self.source_id, "source_", "source_id")
        collections = _canonical_ids(self.collection_ids, "collection_", "collection_ids")
        if not collections:
            raise MeaningContractError("MeaningInputFragment requires a Collection")
        _require_text(self.text, "text")
        _require_hash(self.text_sha256, "text_sha256")
        if sha256_hex(self.text.encode("utf-8")) != self.text_sha256:
            raise MeaningContractError("MeaningInputFragment text hash does not match")
        object.__setattr__(self, "collection_ids", collections)

    def receipt_payload(self) -> dict[str, object]:
        """Return text-free actual-input receipt material."""

        return {
            "source_fragment_id": self.source_fragment_id,
            "source_version_id": self.source_version_id,
            "source_id": self.source_id,
            "collection_ids": list(self.collection_ids),
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True, slots=True)
class VoiceProposal:
    key: str
    kind: VoiceKind
    label: str

    def __post_init__(self) -> None:
        _require_key(self.key)
        if not isinstance(self.kind, VoiceKind):
            raise MeaningContractError("Voice kind must be VoiceKind")
        _require_text(self.label, "Voice label")

    def payload(self) -> dict[str, object]:
        return {"key": self.key, "kind": self.kind.value, "label": self.label}


@dataclass(frozen=True, slots=True)
class EntityProposal:
    key: str
    kind: EntityKind
    label: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_key(self.key)
        if not isinstance(self.kind, EntityKind):
            raise MeaningContractError("Entity kind must be EntityKind")
        _require_text(self.label, "Entity label")
        object.__setattr__(self, "aliases", _canonical_texts(self.aliases, "Entity aliases"))

    def payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kind": self.kind.value,
            "label": self.label,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class ConceptProposal:
    key: str
    label: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_key(self.key)
        _require_text(self.label, "Concept label")
        object.__setattr__(self, "aliases", _canonical_texts(self.aliases, "Concept aliases"))

    def payload(self) -> dict[str, object]:
        return {"key": self.key, "label": self.label, "aliases": list(self.aliases)}


@dataclass(frozen=True, slots=True)
class TimeContextProposal:
    key: str
    event_valid_start: str | None = None
    event_valid_end: str | None = None
    recorded_at: str | None = None
    available_at: str | None = None
    revealed_at: str | None = None
    reviewed_at: str | None = None

    def __post_init__(self) -> None:
        _require_key(self.key)
        values = (
            self.event_valid_start,
            self.event_valid_end,
            self.recorded_at,
            self.available_at,
            self.revealed_at,
            self.reviewed_at,
        )
        if all(value is None for value in values):
            raise MeaningContractError("TimeContext must contain at least one clock")
        for value in values:
            if value is not None:
                _require_timestamp(value, "TimeContext clock")
        if (
            self.event_valid_start is not None
            and self.event_valid_end is not None
            and self.event_valid_start > self.event_valid_end
        ):
            raise MeaningContractError("TimeContext event interval is reversed")

    def payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "event_valid_start": self.event_valid_start,
            "event_valid_end": self.event_valid_end,
            "recorded_at": self.recorded_at,
            "available_at": self.available_at,
            "revealed_at": self.revealed_at,
            "reviewed_at": self.reviewed_at,
        }


@dataclass(frozen=True, slots=True)
class ExactMentionProposal:
    """One exact, separately stored Entity or Concept mention."""

    target_key: str
    collection_id: str
    source_fragment_id: str
    quote_start: int
    quote_end: int
    quote_text: str
    quote_sha256: str

    def __post_init__(self) -> None:
        _require_key(self.target_key)
        _require_prefixed_id(self.collection_id, "collection_", "collection_id")
        _require_prefixed_id(self.source_fragment_id, "fragment_", "source_fragment_id")
        _validate_quote_fields(
            self.quote_start,
            self.quote_end,
            self.quote_text,
            self.quote_sha256,
        )

    def payload(self) -> dict[str, object]:
        return {
            "target_key": self.target_key,
            "collection_id": self.collection_id,
            "source_fragment_id": self.source_fragment_id,
            "quote_start": self.quote_start,
            "quote_end": self.quote_end,
            "quote_text": self.quote_text,
            "quote_sha256": self.quote_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvidenceLinkProposal:
    source_fragment_id: str
    quote_start: int
    quote_end: int
    quote_text: str
    quote_sha256: str
    attributed_voice_key: str
    polarity: EvidencePolarity = EvidencePolarity.SUPPORTS
    alignment: SemanticAlignment = SemanticAlignment.EXACT
    limits_text: str | None = None
    extraction_method: str = "meaning_import/1.0"

    def __post_init__(self) -> None:
        _require_prefixed_id(self.source_fragment_id, "fragment_", "source_fragment_id")
        _validate_quote_fields(
            self.quote_start,
            self.quote_end,
            self.quote_text,
            self.quote_sha256,
        )
        _require_key(self.attributed_voice_key)
        if not isinstance(self.polarity, EvidencePolarity):
            raise MeaningContractError("EvidenceLink polarity must be EvidencePolarity")
        if not isinstance(self.alignment, SemanticAlignment):
            raise MeaningContractError("EvidenceLink alignment must be SemanticAlignment")
        if self.limits_text is not None:
            _require_text(self.limits_text, "EvidenceLink limits")
        _require_text(self.extraction_method, "EvidenceLink extraction_method")

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "quote_start": self.quote_start,
            "quote_end": self.quote_end,
            "quote_text": self.quote_text,
            "quote_sha256": self.quote_sha256,
            "attributed_voice_key": self.attributed_voice_key,
            "polarity": self.polarity.value,
            "alignment": self.alignment.value,
            "limits_text": self.limits_text,
            "extraction_method": self.extraction_method,
        }


@dataclass(frozen=True, slots=True)
class StatementProposal:
    key: str
    kind: StatementKind
    statement_text: str
    voice_key: str
    collection_id: str
    evidence: tuple[EvidenceLinkProposal, ...]
    time_context_key: str | None = None

    def __post_init__(self) -> None:
        _require_key(self.key)
        if not isinstance(self.kind, StatementKind):
            raise MeaningContractError("Statement kind must be StatementKind")
        _require_text(self.statement_text, "Statement text")
        _require_key(self.voice_key)
        _require_prefixed_id(self.collection_id, "collection_", "collection_id")
        if self.time_context_key is not None:
            _require_key(self.time_context_key)
        if not self.evidence or any(
            not isinstance(item, EvidenceLinkProposal) for item in self.evidence
        ):
            raise MeaningContractError("every Statement requires typed EvidenceLinks")
        evidence = tuple(
            sorted(
                self.evidence,
                key=lambda item: (
                    item.source_fragment_id,
                    item.quote_start,
                    item.quote_end,
                    item.attributed_voice_key,
                ),
            )
        )
        if len({canonical_sha256_hex(item.payload()) for item in evidence}) != len(evidence):
            raise MeaningContractError("Statement EvidenceLinks must be unique")
        object.__setattr__(self, "evidence", evidence)

    def payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kind": self.kind.value,
            "statement_text": self.statement_text,
            "voice_key": self.voice_key,
            "collection_id": self.collection_id,
            "time_context_key": self.time_context_key,
            "evidence": [item.payload() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class ConceptMeaningProposal:
    key: str
    concept_key: str
    voice_key: str
    meaning_text: str
    statement_keys: tuple[str, ...]
    time_context_key: str | None = None

    def __post_init__(self) -> None:
        _require_key(self.key)
        _require_key(self.concept_key)
        _require_key(self.voice_key)
        _require_text(self.meaning_text, "ConceptMeaning text")
        statements = _canonical_keys(self.statement_keys, "ConceptMeaning statement keys")
        if not statements:
            raise MeaningContractError("ConceptMeaning requires a source-grounded Statement")
        if self.time_context_key is not None:
            _require_key(self.time_context_key)
        object.__setattr__(self, "statement_keys", statements)

    def payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "concept_key": self.concept_key,
            "voice_key": self.voice_key,
            "meaning_text": self.meaning_text,
            "statement_keys": list(self.statement_keys),
            "time_context_key": self.time_context_key,
        }


@dataclass(frozen=True, slots=True)
class MeaningOmission:
    category: OmissionCategory
    reason_code: str
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.category, OmissionCategory):
            raise MeaningContractError("omission category must be OmissionCategory")
        _require_key(self.reason_code)
        _require_int_range(self.count, 1, 1_000_000, "omission count")

    def payload(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "reason_code": self.reason_code,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class MeaningProposal:
    """One candidate-only generator/import proposal before exact validation."""

    state: CoverageState = CoverageState.COMPLETE
    voices: tuple[VoiceProposal, ...] = ()
    entities: tuple[EntityProposal, ...] = ()
    entity_mentions: tuple[ExactMentionProposal, ...] = ()
    concepts: tuple[ConceptProposal, ...] = ()
    concept_mentions: tuple[ExactMentionProposal, ...] = ()
    concept_meanings: tuple[ConceptMeaningProposal, ...] = ()
    time_contexts: tuple[TimeContextProposal, ...] = ()
    statements: tuple[StatementProposal, ...] = ()
    omissions: tuple[MeaningOmission, ...] = ()
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, CoverageState):
            raise MeaningContractError("proposal state must be CoverageState")
        typed_groups: tuple[tuple[object, ...], ...] = (
            self.voices,
            self.entities,
            self.entity_mentions,
            self.concepts,
            self.concept_mentions,
            self.concept_meanings,
            self.time_contexts,
            self.statements,
            self.omissions,
        )
        expected_types: tuple[type[object], ...] = (
            VoiceProposal,
            EntityProposal,
            ExactMentionProposal,
            ConceptProposal,
            ExactMentionProposal,
            ConceptMeaningProposal,
            TimeContextProposal,
            StatementProposal,
            MeaningOmission,
        )
        for values, expected in zip(typed_groups, expected_types, strict=True):
            if type(values) is not tuple or any(not isinstance(item, expected) for item in values):
                raise MeaningContractError("MeaningProposal collections must be typed tuples")

        outputs_present = any(typed_groups[:-1])
        if self.state is CoverageState.COMPLETE:
            if self.failure_code is not None:
                raise MeaningContractError("complete proposal cannot have failure_code")
        else:
            if outputs_present:
                raise MeaningContractError(
                    "partial/failed/refused extraction cannot create successful candidates"
                )
            if self.failure_code is None or not self.omissions:
                raise MeaningContractError(
                    "non-complete proposal requires failure_code and CoverageReport omissions"
                )
            _require_key(self.failure_code)

        object.__setattr__(
            self,
            "voices",
            tuple(sorted(self.voices, key=lambda item: canonical_sha256_hex(item.payload()))),
        )
        object.__setattr__(
            self,
            "entities",
            tuple(sorted(self.entities, key=lambda item: canonical_sha256_hex(item.payload()))),
        )
        object.__setattr__(
            self,
            "entity_mentions",
            tuple(
                sorted(
                    self.entity_mentions,
                    key=lambda item: canonical_sha256_hex(item.payload()),
                )
            ),
        )
        object.__setattr__(
            self,
            "concepts",
            tuple(sorted(self.concepts, key=lambda item: canonical_sha256_hex(item.payload()))),
        )
        object.__setattr__(
            self,
            "concept_mentions",
            tuple(
                sorted(
                    self.concept_mentions,
                    key=lambda item: canonical_sha256_hex(item.payload()),
                )
            ),
        )
        object.__setattr__(
            self,
            "concept_meanings",
            tuple(
                sorted(
                    self.concept_meanings,
                    key=lambda item: canonical_sha256_hex(item.payload()),
                )
            ),
        )
        object.__setattr__(
            self,
            "time_contexts",
            tuple(
                sorted(
                    self.time_contexts,
                    key=lambda item: canonical_sha256_hex(item.payload()),
                )
            ),
        )
        object.__setattr__(
            self,
            "statements",
            tuple(
                sorted(
                    self.statements,
                    key=lambda item: canonical_sha256_hex(item.payload()),
                )
            ),
        )
        object.__setattr__(
            self,
            "omissions",
            tuple(
                sorted(
                    self.omissions,
                    key=lambda item: canonical_sha256_hex(item.payload()),
                )
            ),
        )
        for values, label in (
            (self.voices, "Voice"),
            (self.entities, "Entity"),
            (self.concepts, "Concept"),
            (self.concept_meanings, "ConceptMeaning"),
            (self.time_contexts, "TimeContext"),
            (self.statements, "Statement"),
        ):
            keys = tuple(item.key for item in values)
            if len(set(keys)) != len(keys):
                raise MeaningContractError(f"{label} proposal keys must be unique")

    @property
    def proposal_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.meaning_proposal/2.0",
            "state": self.state.value,
            "voices": [item.payload() for item in self.voices],
            "entities": [item.payload() for item in self.entities],
            "entity_mentions": [item.payload() for item in self.entity_mentions],
            "concepts": [item.payload() for item in self.concepts],
            "concept_mentions": [item.payload() for item in self.concept_mentions],
            "concept_meanings": [item.payload() for item in self.concept_meanings],
            "time_contexts": [item.payload() for item in self.time_contexts],
            "statements": [item.payload() for item in self.statements],
            "omissions": [item.payload() for item in self.omissions],
            "failure_code": self.failure_code,
        }


@dataclass(frozen=True, slots=True)
class MeaningOutputManifest:
    """Typed candidate IDs emitted by one ProcessingRun."""

    voice_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    entity_mention_ids: tuple[str, ...] = ()
    concept_ids: tuple[str, ...] = ()
    concept_mention_ids: tuple[str, ...] = ()
    concept_meaning_ids: tuple[str, ...] = ()
    time_context_ids: tuple[str, ...] = ()
    statement_ids: tuple[str, ...] = ()
    evidence_link_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, prefix in (
            ("voice_ids", "voice_"),
            ("entity_ids", "entity_"),
            ("entity_mention_ids", "entity_mention_"),
            ("concept_ids", "concept_"),
            ("concept_mention_ids", "concept_mention_"),
            ("concept_meaning_ids", "concept_meaning_"),
            ("time_context_ids", "time_context_"),
            ("statement_ids", "statement_"),
            ("evidence_link_ids", "evidence_"),
        ):
            object.__setattr__(self, name, _canonical_ids(getattr(self, name), prefix, name))

    @property
    def candidate_count(self) -> int:
        return sum(
            len(values)
            for values in (
                self.voice_ids,
                self.entity_ids,
                self.entity_mention_ids,
                self.concept_ids,
                self.concept_mention_ids,
                self.concept_meaning_ids,
                self.time_context_ids,
                self.statement_ids,
                self.evidence_link_ids,
            )
        )

    @property
    def output_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.meaning_output_manifest/2.0",
            "voice_ids": list(self.voice_ids),
            "entity_ids": list(self.entity_ids),
            "entity_mention_ids": list(self.entity_mention_ids),
            "concept_ids": list(self.concept_ids),
            "concept_mention_ids": list(self.concept_mention_ids),
            "concept_meaning_ids": list(self.concept_meaning_ids),
            "time_context_ids": list(self.time_context_ids),
            "statement_ids": list(self.statement_ids),
            "evidence_link_ids": list(self.evidence_link_ids),
        }


@dataclass(frozen=True, slots=True)
class MeaningProcessingRun:
    processing_run_id: str
    kind: MeaningRunKind
    corpus_snapshot_id: str
    access_policy_id: str
    scope_hash: str
    permitted_set_hash: str
    profile: ExtractionProfile
    code_version: str
    generator_hash: str
    review_budget_hash: str
    input_hash: str
    proposal_hash: str | None
    status: MeaningRunStatus
    unresolved_before: int
    backlog_warning: bool
    error_code: str | None
    output: MeaningOutputManifest
    receipt_hash: str
    started_at: str
    finished_at: str


@dataclass(frozen=True, slots=True)
class MeaningCoverageReport:
    coverage_report_id: str
    processing_run_id: str
    profile: ExtractionProfile
    state: CoverageState
    read_fragment_count: int
    candidate_count: int
    omissions: tuple[MeaningOmission, ...]
    report_hash: str


@dataclass(frozen=True, slots=True)
class MeaningReadReceiptItem:
    source_fragment_id: str
    source_version_id: str
    source_id: str
    collection_ids: tuple[str, ...]
    read_order: int
    text_sha256: str

    def __post_init__(self) -> None:
        _require_prefixed_id(self.source_fragment_id, "fragment_", "source_fragment_id")
        _require_prefixed_id(self.source_version_id, "source_version_", "source_version_id")
        _require_prefixed_id(self.source_id, "source_", "source_id")
        collections = _canonical_ids(self.collection_ids, "collection_", "collection_ids")
        if not collections:
            raise MeaningContractError("ReadReceipt item requires a Collection")
        _require_int_range(self.read_order, 0, 1_999, "read_order")
        _require_hash(self.text_sha256, "text_sha256")
        object.__setattr__(self, "collection_ids", collections)

    def input_payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_version_id": self.source_version_id,
            "source_id": self.source_id,
            "collection_ids": list(self.collection_ids),
            "text_sha256": self.text_sha256,
        }

    def receipt_payload(self) -> dict[str, object]:
        return {**self.input_payload(), "read_order": self.read_order}


@dataclass(frozen=True, slots=True)
class MeaningReadReceipt:
    read_receipt_id: str
    processing_run_id: str
    input_hash: str
    items: tuple[MeaningReadReceiptItem, ...]
    receipt_hash: str


@dataclass(frozen=True, slots=True)
class MeaningRunResult:
    run: MeaningProcessingRun
    coverage_report: MeaningCoverageReport
    read_receipt: MeaningReadReceipt
    deduplicated: bool = False


@dataclass(frozen=True, slots=True)
class MeaningReviewScope:
    collection_ids: tuple[str, ...]
    use: str

    def __post_init__(self) -> None:
        collections = _canonical_ids(self.collection_ids, "collection_", "collection_ids")
        if not collections:
            raise MeaningContractError("ReviewScope requires at least one Collection")
        _require_key(self.use)
        object.__setattr__(self, "collection_ids", collections)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.meaning_review_scope/1.0",
            "collection_ids": list(self.collection_ids),
            "use": self.use,
        }


@dataclass(frozen=True, slots=True)
class MeaningReviewRequest:
    review_session_id: str
    target_type: MeaningReviewTargetType
    target_id: str
    target_hash: str
    action: MeaningReviewAction
    reason: str
    authority: str
    scope: MeaningReviewScope
    replacement_target_id: str | None = None
    supersedes_review_decision_id: str | None = None

    def __post_init__(self) -> None:
        _require_prefixed_id(self.review_session_id, "review_session_", "review_session_id")
        if not isinstance(self.target_type, MeaningReviewTargetType):
            raise MeaningContractError("target_type must be MeaningReviewTargetType")
        _require_prefixed_id(self.target_id, _target_prefix(self.target_type), "target_id")
        _require_hash(self.target_hash, "target_hash")
        if not isinstance(self.action, MeaningReviewAction):
            raise MeaningContractError("action must be MeaningReviewAction")
        _require_text(self.reason, "ReviewDecision reason")
        _require_text(self.authority, "ReviewDecision authority")
        if not isinstance(self.scope, MeaningReviewScope):
            raise MeaningContractError("scope must be MeaningReviewScope")
        if self.action is MeaningReviewAction.REVISE:
            if self.replacement_target_id is None:
                raise MeaningContractError("revise requires a replacement candidate")
            _require_prefixed_id(
                self.replacement_target_id,
                _target_prefix(self.target_type),
                "replacement_target_id",
            )
        elif self.replacement_target_id is not None:
            raise MeaningContractError("only revise may name a replacement candidate")
        if self.action is MeaningReviewAction.SUPERSEDE:
            if self.supersedes_review_decision_id is None:
                raise MeaningContractError("supersede requires a prior ReviewDecision")
            _require_prefixed_id(
                self.supersedes_review_decision_id,
                "review_",
                "supersedes_review_decision_id",
            )
        elif self.supersedes_review_decision_id is not None:
            raise MeaningContractError("only supersede may name a prior ReviewDecision")


@dataclass(frozen=True, slots=True)
class MeaningReviewSession:
    review_session_id: str
    library_id: str
    review_budget: ReviewBudget
    created_at: str


@dataclass(frozen=True, slots=True)
class MeaningReviewDecision:
    review_decision_id: str
    review_session_id: str
    target_type: MeaningReviewTargetType
    target_id: str
    target_hash: str
    action: MeaningReviewAction
    reason: str
    authority: str
    scope: MeaningReviewScope
    replacement_target_id: str | None
    supersedes_review_decision_id: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class RelationAdmission:
    """Exact reviewed permission for one Relation in one query use-context."""

    relation_id: str
    target_hash: str
    review_decision_id: str
    scope: MeaningReviewScope

    def __post_init__(self) -> None:
        _require_prefixed_id(self.relation_id, "relation_", "relation_id")
        _require_hash(self.target_hash, "target_hash")
        _require_prefixed_id(
            self.review_decision_id,
            "review_relation_",
            "review_decision_id",
        )
        if not isinstance(self.scope, MeaningReviewScope):
            raise MeaningContractError("RelationAdmission scope must be MeaningReviewScope")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "dithyramba.relation_admission/1.0",
            "relation_id": self.relation_id,
            "target_hash": self.target_hash,
            "review_decision_id": self.review_decision_id,
            "scope": self.scope.payload(),
        }

    @property
    def admission_hash(self) -> str:
        return canonical_sha256_hex(self.payload())


@dataclass(frozen=True, slots=True)
class MeaningReviewQueueItem:
    target_type: MeaningReviewTargetType
    target_id: str
    target_hash: str
    collection_ids: tuple[str, ...]
    created_at: str


MeaningGenerator: TypeAlias = Callable[
    [tuple[MeaningInputFragment, ...], MeaningRunRequest], MeaningProposal
]


def _target_prefix(target_type: MeaningReviewTargetType) -> str:
    return {
        MeaningReviewTargetType.STATEMENT: "statement_",
        MeaningReviewTargetType.EVIDENCE_LINK: "evidence_",
        MeaningReviewTargetType.VOICE: "voice_",
        MeaningReviewTargetType.ENTITY: "entity_",
        MeaningReviewTargetType.CONCEPT: "concept_",
        MeaningReviewTargetType.CONCEPT_MEANING: "concept_meaning_",
        MeaningReviewTargetType.TIME_CONTEXT: "time_context_",
        MeaningReviewTargetType.RELATION: "relation_",
    }[target_type]


def _require_key(value: str) -> None:
    if type(value) is not str or _KEY_PATTERN.fullmatch(value) is None:
        raise MeaningContractError("semantic key must match [a-z][a-z0-9_-]{0,63}")


def _require_text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip() or len(value) > 100_000:
        raise MeaningContractError(f"{label} must be non-empty bounded text")


def _require_hash(value: str, label: str) -> None:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise MeaningContractError(f"{label} must be lowercase SHA-256")


def _require_prefixed_id(value: str, prefix: str, label: str) -> None:
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or len(value) <= len(prefix)
        or _ID_PATTERN.fullmatch(value) is None
    ):
        raise MeaningContractError(f"{label} must be a {prefix} identifier")


def _require_int_range(value: int, minimum: int, maximum: int, label: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise MeaningContractError(f"{label} must be between {minimum} and {maximum}")


def _canonical_texts(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise MeaningContractError(f"{label} must be a tuple")
    for value in values:
        _require_text(value, label)
    canonical = tuple(sorted(set(values)))
    if len(canonical) != len(values):
        raise MeaningContractError(f"{label} must be unique")
    if len(canonical) > 32:
        raise MeaningContractError(f"{label} may contain at most 32 lexical aliases")
    return canonical


def _canonical_keys(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise MeaningContractError(f"{label} must be a tuple")
    for value in values:
        _require_key(value)
    canonical = tuple(sorted(set(values)))
    if len(canonical) != len(values):
        raise MeaningContractError(f"{label} must be unique")
    return canonical


def _canonical_ids(values: tuple[str, ...], prefix: str, label: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise MeaningContractError(f"{label} must be a tuple")
    for value in values:
        _require_prefixed_id(value, prefix, label)
    canonical = tuple(sorted(set(values)))
    if len(canonical) != len(values):
        raise MeaningContractError(f"{label} must be unique")
    return canonical


def _validate_quote_fields(start: int, end: int, text: str, digest: str) -> None:
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise MeaningContractError("exact quote offsets must define a non-empty range")
    _require_text(text, "exact quote")
    _require_hash(digest, "quote_sha256")
    if sha256_hex(text.encode("utf-8")) != digest:
        raise MeaningContractError("exact quote hash does not match quote text")


def _require_timestamp(value: str, label: str) -> None:
    if type(value) is not str or _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise MeaningContractError(f"{label} must be a canonical UTC timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise MeaningContractError(f"{label} is not a real UTC timestamp") from exc
