"""Bounded evidence-seeking recall with q0-anchored Harrier relevance.

This module is intentionally an in-memory library seam.  It operates only on
already-authorized exact fragments, reuses one ephemeral FTS session, lets
Harrier order the bounded candidate union by the original question, and lets
the deterministic EvidenceCoverageGate decide sufficiency.  Generated repair
queries can discover candidates; they can never become evidence themselves.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar, Literal, Protocol

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from dithyramba.contracts import canonical_sha256_hex, sha256_hex
from dithyramba.evidence import (
    EvidenceCandidate,
    EvidenceCoverageGate,
    EvidenceCoverageResult,
    EvidenceGateDecision,
    EvidenceGateSpec,
)

from .fts import FtsFragment, FtsSearchResult, PermittedFtsSession
from .hybrid_models import EmbeddingModelProfile, ModelRuntimeProfile
from .provisioning import HARRIER_OSS_V1_270M_PROFILE
from .vector import DenseVectorItem, PackedVector, exact_cosine_scan

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FRAGMENT_ID_PATTERN = re.compile(r"^fragment_[a-z0-9]+(?:_[a-z0-9]+)*$")
_REFERENCE_PATTERN = re.compile(r"^[a-z][a-z0-9]+(?:_[a-z0-9]+)*$")
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_QUOTED_PHRASE_PATTERN = re.compile(r"[\"“”]([^\"“”]{2,160})[\"“”]")
_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[^\W\d_])(?=[0-9])|(?<=[0-9])(?=[^\W\d_])")
_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z\u0430-\u044f\u0456\u0457\u0454\u0491])"
    r"(?=[A-Z\u0410-\u042f\u0406\u0407\u0404\u0490])"
)
_MAX_SCORING_CANDIDATES = 500
_MAX_DISCOVERED_CANDIDATES = 5_000


class AdaptiveRecallError(RuntimeError):
    """The bounded adaptive route could not satisfy its frozen contract."""


class QueryRepairKind(StrEnum):
    """Auditable reason for one conditional derived query."""

    LEXICAL_BRIDGE = "lexical_bridge"
    EVIDENCE_ROLE_SPLIT = "evidence_role_split"
    RELATION_STATE_GUARD = "relation_state_guard"


class CandidateOriginKind(StrEnum):
    ORIGINAL_FTS = "original_fts"
    BOUNDARY_FTS = "boundary_fts"
    ALIAS_FTS = "alias_fts"
    EXACT_PHRASE = "exact_phrase"
    NEIGHBOR = "neighbor"
    QUERY_CLOUD = "query_cloud"


class CandidateSelectionScope(StrEnum):
    HARRIER_POOL = "harrier_pool"
    FINAL_UNION = "final_union"


class CandidateSelectionReason(StrEnum):
    PRIOR_GATE_MATCH = "prior_gate_match"
    QUERY_CLOUD_RESERVE = "query_cloud_reserve"


class CandidateSelectionDisposition(StrEnum):
    ALREADY_VISIBLE = "already_visible"
    FORCED = "forced"
    SKIPPED_CAPACITY = "skipped_capacity"


class _FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class AdaptiveFragment(_FrozenContract):
    """One exact authorized fragment plus proof metadata supplied by the caller."""

    source_fragment_id: str
    source_version_id: str
    source_id: str
    ordinal: int = Field(ge=0)
    fragment_kind: str
    text: str
    text_sha256: str
    source_address: dict[str, object]
    source_kind: str
    source_family: str
    authority: str
    independence_group: str
    evidence_tags: tuple[str, ...] = ()
    body_proof_eligible: bool = True

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        if type(value) is not str or _FRAGMENT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("source_fragment_id must use fragment_<key>")
        return value

    @field_validator("source_version_id", "source_id")
    @classmethod
    def _reference(cls, value: str) -> str:
        if type(value) is not str or _REFERENCE_PATTERN.fullmatch(value) is None:
            raise ValueError("source references must be stable lowercase identifiers")
        return value

    @field_validator(
        "fragment_kind",
        "source_kind",
        "source_family",
        "authority",
        "independence_group",
    )
    @classmethod
    def _metadata_text(cls, value: str) -> str:
        return _bounded_text(value, maximum=256)

    @field_validator("text")
    @classmethod
    def _body(cls, value: str) -> str:
        return _bounded_text(value, maximum=2_000_000)

    @field_validator("text_sha256")
    @classmethod
    def _text_hash(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("evidence_tags")
    @classmethod
    def _tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_bounded_text(item, maximum=128) for item in value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("evidence_tags must be unique")
        return tuple(sorted(normalized, key=lambda item: item.encode("utf-8")))

    @model_validator(mode="after")
    def _hash_matches(self) -> AdaptiveFragment:
        if sha256_hex(self.text.encode("utf-8")) != self.text_sha256:
            raise ValueError("text_sha256 does not match fragment text")
        return self

    def fts_fragment(self) -> FtsFragment:
        return FtsFragment(
            source_fragment_id=self.source_fragment_id,
            text=self.text,
            text_sha256=self.text_sha256,
        )

    def evidence_candidate(self, *, rank: int) -> EvidenceCandidate:
        return EvidenceCandidate(
            rank=rank,
            source_fragment_id=self.source_fragment_id,
            source_id=self.source_id,
            text=self.text,
            source_address=self.source_address,
            source_kind=self.source_kind,
            source_family=self.source_family,
            authority=self.authority,
            independence_group=self.independence_group,
            evidence_tags=self.evidence_tags,
        )


class AliasEntry(_FrozenContract):
    """One controlled, non-destructive lexical equivalence set."""

    canonical: str
    alternatives: tuple[str, ...]

    @field_validator("canonical")
    @classmethod
    def _canonical(cls, value: str) -> str:
        return _bounded_text(value, maximum=128)

    @field_validator("alternatives")
    @classmethod
    def _alternatives(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 32:
            raise ValueError("alias entry requires 1-32 alternatives")
        normalized = tuple(_bounded_text(item, maximum=128) for item in value)
        if len({_match_text(item) for item in normalized}) != len(normalized):
            raise ValueError("alias alternatives must be unique")
        return tuple(sorted(normalized, key=lambda item: item.encode("utf-8")))


class AliasRegistry(_FrozenContract):
    """Versioned caller-owned aliases, names and acronym expansions."""

    SCHEMA: ClassVar[str] = "dithyramba.alias_registry/1.0"

    version: str = "alias_v1"
    entries: tuple[AliasEntry, ...] = ()

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        if type(value) is not str or _KEY_PATTERN.fullmatch(value) is None:
            raise ValueError("alias registry version must be stable snake_case")
        return value

    @field_validator("entries")
    @classmethod
    def _entries(cls, value: tuple[AliasEntry, ...]) -> tuple[AliasEntry, ...]:
        ordered = tuple(sorted(value, key=lambda item: _match_text(item.canonical)))
        if len({_match_text(item.canonical) for item in ordered}) != len(ordered):
            raise ValueError("alias canonicals must be unique")
        return ordered

    @property
    def registry_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": self.SCHEMA,
                "version": self.version,
                "entries": [entry.model_dump(mode="json") for entry in self.entries],
            }
        )

    def expanded_question(self, question: str) -> str | None:
        """Append only relevant controlled variants; never replace q0 text."""

        normalized_question = _match_text(question)
        additions: list[str] = []
        for entry in self.entries:
            names = (entry.canonical, *entry.alternatives)
            if not any(_contains_name(normalized_question, name) for name in names):
                continue
            for name in names:
                if not _contains_name(normalized_question, name) and name not in additions:
                    projected = f"{question} {' '.join((*additions, name))}"
                    if len(projected) > 2_000:
                        break
                    additions.append(name)
        if not additions:
            return None
        return f"{question} {' '.join(additions)}"


class DerivedQuery(_FrozenContract):
    query_key: str
    text: str
    kind: QueryRepairKind
    target_requirement_keys: tuple[str, ...]

    @field_validator("query_key")
    @classmethod
    def _query_key(cls, value: str) -> str:
        if value not in {"q1", "q2"}:
            raise ValueError("derived query key must be q1 or q2")
        return value

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        return _bounded_text(value, maximum=2_000)

    @field_validator("target_requirement_keys")
    @classmethod
    def _target_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("derived query must target at least one missing requirement")
        for item in value:
            if type(item) is not str or _KEY_PATTERN.fullmatch(item) is None:
                raise ValueError("target requirement keys must be stable snake_case")
        if len(set(value)) != len(value):
            raise ValueError("target requirement keys must be unique")
        return tuple(sorted(value, key=lambda item: item.encode("ascii")))


class QueryRepairPlan(_FrozenContract):
    """One bounded, persisted-in-the-result QueryCloud proposal."""

    SCHEMA: ClassVar[str] = "dithyramba.query_repair_plan/1.0"

    original_question_sha256: str
    planner_id: str
    queries: tuple[DerivedQuery, ...]

    @field_validator("original_question_sha256")
    @classmethod
    def _hash(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("planner_id")
    @classmethod
    def _planner_id(cls, value: str) -> str:
        if type(value) is not str or _KEY_PATTERN.fullmatch(value) is None:
            raise ValueError("planner_id must be stable snake_case")
        return value

    @field_validator("queries")
    @classmethod
    def _queries(cls, value: tuple[DerivedQuery, ...]) -> tuple[DerivedQuery, ...]:
        if not 1 <= len(value) <= 2:
            raise ValueError("QueryCloud requires one or two derived queries")
        if len({item.query_key for item in value}) != len(value):
            raise ValueError("derived query keys must be unique")
        if len({_match_text(item.text) for item in value}) != len(value):
            raise ValueError("derived query texts must be unique")
        return tuple(sorted(value, key=lambda item: item.query_key))

    @property
    def plan_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": self.SCHEMA,
                "original_question_sha256": self.original_question_sha256,
                "planner_id": self.planner_id,
                "queries": [item.model_dump(mode="json") for item in self.queries],
            }
        )


class QueryRepairProvider(Protocol):
    """One-call provider seam; implementations do not receive corpus text."""

    def plan(
        self,
        *,
        question: str,
        missing_requirement_keys: tuple[str, ...],
        alias_registry: AliasRegistry,
    ) -> QueryRepairPlan: ...


class CandidateText(_FrozenContract):
    source_fragment_id: str
    source_id: str
    text: str
    text_sha256: str


class HarrierScoreItem(_FrozenContract):
    source_fragment_id: str
    text_sha256: str
    rank: int = Field(ge=1, le=500)
    score_hex: str
    vector_sha256: str

    @field_validator("text_sha256", "vector_sha256")
    @classmethod
    def _hashes(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("score_hex")
    @classmethod
    def _score(cls, value: str) -> str:
        try:
            score = float.fromhex(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Harrier score must be exact float.hex") from error
        if not -1.000001 <= score <= 1.000001:
            raise ValueError("Harrier score is outside cosine range")
        return value

    @property
    def score(self) -> float:
        return float.fromhex(self.score_hex)


class HarrierScoreBatch(_FrozenContract):
    """Relevance receipt for one bounded candidate union scored against q0."""

    SCHEMA: ClassVar[str] = "dithyramba.harrier_score_batch/1.0"

    original_question_sha256: str
    model_profile_hash: str
    runtime_profile_hash: str
    provisioning_receipt_hash: str
    items: tuple[HarrierScoreItem, ...]

    @field_validator(
        "original_question_sha256",
        "model_profile_hash",
        "runtime_profile_hash",
        "provisioning_receipt_hash",
    )
    @classmethod
    def _hashes(cls, value: str) -> str:
        return _require_hash(value)

    @field_validator("items")
    @classmethod
    def _items(cls, value: tuple[HarrierScoreItem, ...]) -> tuple[HarrierScoreItem, ...]:
        if not value or len(value) > 500:
            raise ValueError("Harrier batch requires 1-500 scores")
        if tuple(item.rank for item in value) != tuple(range(1, len(value) + 1)):
            raise ValueError("Harrier ranks must be contiguous from one")
        if len({item.source_fragment_id for item in value}) != len(value):
            raise ValueError("Harrier score fragment IDs must be unique")
        scores = tuple(item.score for item in value)
        if scores != tuple(sorted(scores, reverse=True)):
            raise ValueError("Harrier scores must be monotonic descending")
        return value

    @property
    def batch_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": self.SCHEMA,
                **self.model_dump(mode="json", exclude={"items"}),
                "items": [item.model_dump(mode="json") for item in self.items],
            }
        )


class HarrierCandidateScorer(Protocol):
    """Candidate-only semantic relevance seam; it is not an evidence judge."""

    def score(
        self,
        *,
        q0: str,
        candidates: tuple[CandidateText, ...],
    ) -> HarrierScoreBatch: ...


class HarrierSemanticProvider(Protocol):
    """Small provider view needed by the scorer, avoiding an eager import cycle."""

    @property
    def model_profile(self) -> EmbeddingModelProfile: ...

    @property
    def runtime_profile(self) -> ModelRuntimeProfile: ...

    def embed_query(self, question: str) -> PackedVector: ...

    def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]: ...


class SemanticHarrierCandidateScorer:
    """Exact cosine adapter over the pinned local Harrier embedding provider."""

    def __init__(
        self,
        provider: HarrierSemanticProvider,
        *,
        provisioning_receipt_hash: str,
    ) -> None:
        if provider.model_profile != HARRIER_OSS_V1_270M_PROFILE:
            raise AdaptiveRecallError("Harrier scorer requires the pinned Harrier profile")
        self._provider = provider
        self._provisioning_receipt_hash = _require_hash(provisioning_receipt_hash)
        self._query_cache: dict[str, PackedVector] = {}
        self._passage_cache: dict[tuple[str, str, str, str], PackedVector] = {}

    def score(
        self,
        *,
        q0: str,
        candidates: tuple[CandidateText, ...],
    ) -> HarrierScoreBatch:
        q0 = _bounded_text(q0, maximum=2_000)
        if not candidates or len(candidates) > 500:
            raise AdaptiveRecallError("Harrier scorer requires 1-500 candidate texts")
        identifiers = tuple(item.source_fragment_id for item in candidates)
        if len(set(identifiers)) != len(identifiers):
            raise AdaptiveRecallError("Harrier candidate IDs must be unique")
        model_hash = self._provider.model_profile.profile_hash
        runtime_hash = self._provider.runtime_profile.profile_hash
        q0_hash = sha256_hex(q0.encode("utf-8"))
        query_key = canonical_sha256_hex(
            {"q0_sha256": q0_hash, "model": model_hash, "runtime": runtime_hash}
        )
        query_vector = self._query_cache.get(query_key)
        if query_vector is None:
            query_vector = self._provider.embed_query(q0)
            self._query_cache[query_key] = query_vector

        vectors: dict[str, PackedVector] = {}
        missing: list[CandidateText] = []
        for candidate in candidates:
            key = (
                candidate.text_sha256,
                model_hash,
                runtime_hash,
                self._provisioning_receipt_hash,
            )
            cached = self._passage_cache.get(key)
            if cached is None:
                missing.append(candidate)
            else:
                vectors[candidate.source_fragment_id] = cached
        if missing:
            embedded = self._provider.embed_passages(tuple(item.text for item in missing))
            if len(embedded) != len(missing):
                raise AdaptiveRecallError("Harrier provider returned the wrong vector count")
            for candidate, vector in zip(missing, embedded, strict=True):
                key = (
                    candidate.text_sha256,
                    model_hash,
                    runtime_hash,
                    self._provisioning_receipt_hash,
                )
                self._passage_cache[key] = vector
                vectors[candidate.source_fragment_id] = vector

        by_id = {item.source_fragment_id: item for item in candidates}
        hits = exact_cosine_scan(
            query_vector,
            (
                DenseVectorItem(
                    source_fragment_id=item.source_fragment_id,
                    source_id=item.source_id,
                    vector=vectors[item.source_fragment_id],
                )
                for item in candidates
            ),
            limit=len(candidates),
        )
        return HarrierScoreBatch(
            original_question_sha256=q0_hash,
            model_profile_hash=model_hash,
            runtime_profile_hash=runtime_hash,
            provisioning_receipt_hash=self._provisioning_receipt_hash,
            items=tuple(
                HarrierScoreItem(
                    source_fragment_id=hit.source_fragment_id,
                    text_sha256=by_id[hit.source_fragment_id].text_sha256,
                    rank=hit.rank,
                    score_hex=hit.score_hex,
                    vector_sha256=vectors[hit.source_fragment_id].vector_hash,
                )
                for hit in hits
            ),
        )


class AdaptiveRetrievalConfig(_FrozenContract):
    SCHEMA: ClassVar[str] = "dithyramba.adaptive_retrieval_config/2.1"

    base_fts_candidates: int = Field(default=50, ge=1, le=500)
    rescue_fts_candidates: int = Field(default=100, ge=1, le=500)
    max_union_candidates: int = Field(default=160, ge=1, le=500)
    family_diversity_window_candidates: int = Field(
        default=30,
        ge=1,
        le=500,
        validation_alias=AliasChoices(
            "family_diversity_window_candidates",
            "max_proof_candidates",
        ),
    )
    max_gate_scan_characters: int = Field(default=5_000_000, ge=1, le=100_000_000)
    query_cloud_reserve_per_query: int = Field(default=10, ge=0, le=50)
    max_per_family_before_diversity: int = Field(default=2, ge=1, le=20)
    max_exact_phrase_candidates: int = Field(default=20, ge=0, le=100)
    max_neighbor_candidates: int = Field(default=50, ge=0, le=200)

    @model_validator(mode="after")
    def _limits(self) -> AdaptiveRetrievalConfig:
        if self.rescue_fts_candidates < self.base_fts_candidates:
            raise ValueError("rescue FTS budget must not be below base budget")
        if self.family_diversity_window_candidates > self.max_union_candidates:
            raise ValueError("family diversity window must fit candidate union")
        if self.query_cloud_reserve_per_query * 2 > self.max_union_candidates:
            raise ValueError("QueryCloud reserve for two queries must fit candidate union")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "base_fts_candidates": self.base_fts_candidates,
            "rescue_fts_candidates": self.rescue_fts_candidates,
            "max_union_candidates": self.max_union_candidates,
            "family_diversity_window_candidates": self.family_diversity_window_candidates,
            "max_gate_scan_characters": self.max_gate_scan_characters,
            "query_cloud_reserve_per_query": self.query_cloud_reserve_per_query,
            "max_per_family_before_diversity": self.max_per_family_before_diversity,
            "max_exact_phrase_candidates": self.max_exact_phrase_candidates,
            "max_neighbor_candidates": self.max_neighbor_candidates,
        }

    @property
    def config_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())


class CandidateOrigin(_FrozenContract):
    kind: CandidateOriginKind
    query_key: str
    rank: int | None = Field(default=None, ge=1, le=500)


class AdaptiveCandidateSelectionDecision(_FrozenContract):
    """Why one mandatory candidate was visible, forced, or safely skipped."""

    source_fragment_id: str
    reasons: tuple[CandidateSelectionReason, ...]
    query_origins: tuple[CandidateOrigin, ...] = ()
    disposition: CandidateSelectionDisposition
    evicted_fragment_id: str | None = None

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        if type(value) is not str or _FRAGMENT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("source_fragment_id must use fragment_<key>")
        return value

    @field_validator("evicted_fragment_id")
    @classmethod
    def _evicted_id(cls, value: str | None) -> str | None:
        if value is not None and _FRAGMENT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("evicted_fragment_id must use fragment_<key>")
        return value

    @model_validator(mode="after")
    def _closed_reason(self) -> AdaptiveCandidateSelectionDecision:
        if not self.reasons or len(set(self.reasons)) != len(self.reasons):
            raise ValueError("selection reasons must be non-empty and unique")
        has_query_reason = CandidateSelectionReason.QUERY_CLOUD_RESERVE in self.reasons
        if has_query_reason != bool(self.query_origins):
            raise ValueError("QueryCloud selection reason must close over query origins")
        if any(
            origin.kind is not CandidateOriginKind.QUERY_CLOUD or origin.rank is None
            for origin in self.query_origins
        ):
            raise ValueError("selection query origins must be ranked QueryCloud origins")
        if len(set(self.query_origins)) != len(self.query_origins):
            raise ValueError("selection query origins must be unique")
        if self.disposition is CandidateSelectionDisposition.FORCED:
            if self.evicted_fragment_id is None:
                raise ValueError("forced selection requires one evicted fragment")
        elif self.evicted_fragment_id is not None:
            raise ValueError("only forced selection may name an evicted fragment")
        return self


class AdaptiveCandidateSelectionReceipt(_FrozenContract):
    """Replayable bounded selection between discovery, Harrier, and the Gate."""

    SCHEMA: ClassVar[str] = "dithyramba.adaptive_candidate_selection/1.0"

    scope: CandidateSelectionScope
    capacity: int = Field(ge=1, le=500)
    query_cloud_reserve_per_query: int = Field(ge=0, le=50)
    input_fragment_ids: tuple[str, ...]
    initial_fragment_ids: tuple[str, ...]
    protected_fragment_ids: tuple[str, ...]
    decisions: tuple[AdaptiveCandidateSelectionDecision, ...]
    already_visible_fragment_ids: tuple[str, ...]
    forced_fragment_ids: tuple[str, ...]
    skipped_capacity_fragment_ids: tuple[str, ...]
    evicted_fragment_ids: tuple[str, ...]
    protected_output_fragment_ids: tuple[str, ...]
    output_fragment_ids: tuple[str, ...]
    input_hash: str
    initial_hash: str
    protected_output_hash: str
    output_hash: str

    @field_validator("input_hash", "initial_hash", "protected_output_hash", "output_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        return _require_hash(value)

    @model_validator(mode="after")
    def _closed_selection(self) -> AdaptiveCandidateSelectionReceipt:
        inputs = self.input_fragment_ids
        initial = self.initial_fragment_ids
        protected_output = self.protected_output_fragment_ids
        outputs = self.output_fragment_ids
        protected = self.protected_fragment_ids
        for label, values in (
            ("selection input", inputs),
            ("selection initial output", initial),
            ("selection protected output", protected_output),
            ("selection output", outputs),
            ("protected fragments", protected),
            ("already-visible fragments", self.already_visible_fragment_ids),
            ("forced fragments", self.forced_fragment_ids),
            ("capacity-skipped fragments", self.skipped_capacity_fragment_ids),
            ("evicted fragments", self.evicted_fragment_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique")
            if any(_FRAGMENT_ID_PATTERN.fullmatch(value) is None for value in values):
                raise ValueError(f"{label} must use fragment_<key>")
        if len(outputs) != min(len(inputs), self.capacity):
            raise ValueError("selection output must fill the bounded capacity")
        if initial != inputs[: self.capacity]:
            raise ValueError("selection initial output must be the bounded input prefix")
        if len(protected_output) != len(initial):
            raise ValueError("protected selection output must preserve bounded size")
        if not set(outputs).issubset(set(inputs)):
            raise ValueError("selection output must be a subset of its input")
        if not set(protected).issubset(set(outputs)):
            raise ValueError("prior Gate matches must remain selected")
        if not set(protected).issubset(set(protected_output)):
            raise ValueError("prior Gate matches must survive the protected phase")
        if _candidate_sequence_hash(self.scope, inputs) != self.input_hash:
            raise ValueError("selection input hash does not match input IDs")
        if _candidate_sequence_hash(self.scope, initial) != self.initial_hash:
            raise ValueError("selection initial hash does not match initial IDs")
        if _candidate_sequence_hash(self.scope, protected_output) != self.protected_output_hash:
            raise ValueError("selection protected-output hash does not match its IDs")
        if _candidate_sequence_hash(self.scope, outputs) != self.output_hash:
            raise ValueError("selection output hash does not match output IDs")
        decision_ids = tuple(item.source_fragment_id for item in self.decisions)
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("selection decisions must name unique candidates")
        if not set(decision_ids).issubset(set(inputs)):
            raise ValueError("selection decisions must refer to input candidates")
        initial_set = set(initial)
        prior_decision_ids = tuple(
            item.source_fragment_id
            for item in self.decisions
            if CandidateSelectionReason.PRIOR_GATE_MATCH in item.reasons
        )
        if prior_decision_ids != protected:
            raise ValueError("prior-match decisions must exactly match protected fragments")
        forced = tuple(
            item
            for item in self.decisions
            if item.disposition is CandidateSelectionDisposition.FORCED
        )
        evicted_ids = tuple(
            item.evicted_fragment_id for item in forced if item.evicted_fragment_id is not None
        )
        if len(evicted_ids) != len(forced):
            raise ValueError("forced selections must name an evicted candidate")
        if len(set(evicted_ids)) != len(evicted_ids):
            raise ValueError("forced selections must evict unique candidates")
        if any(value not in initial_set for value in evicted_ids):
            raise ValueError("forced selections may evict only initial candidates")
        expected_already_visible = tuple(
            item.source_fragment_id
            for item in self.decisions
            if item.disposition is CandidateSelectionDisposition.ALREADY_VISIBLE
        )
        expected_forced = tuple(item.source_fragment_id for item in forced)
        expected_skipped = tuple(
            item.source_fragment_id
            for item in self.decisions
            if item.disposition is CandidateSelectionDisposition.SKIPPED_CAPACITY
        )
        if self.already_visible_fragment_ids != expected_already_visible:
            raise ValueError("already-visible list must match selection decisions")
        if self.forced_fragment_ids != expected_forced:
            raise ValueError("forced list must match selection decisions")
        if self.skipped_capacity_fragment_ids != expected_skipped:
            raise ValueError("capacity-skipped list must match selection decisions")
        if self.evicted_fragment_ids != evicted_ids:
            raise ValueError("evicted list must match forced selection decisions")
        for item in self.decisions:
            is_initial = item.source_fragment_id in initial_set
            is_output = item.source_fragment_id in set(outputs)
            if item.disposition is CandidateSelectionDisposition.ALREADY_VISIBLE:
                if not is_initial or not is_output:
                    raise ValueError("already-visible selection must remain in output")
            elif item.disposition is CandidateSelectionDisposition.FORCED:
                if is_initial or not is_output:
                    raise ValueError("forced selection must enter from outside initial output")
            elif is_output:
                raise ValueError("capacity-skipped selection must not enter output")
        query_counts: dict[str, int] = {}
        for item in self.decisions:
            for origin in item.query_origins:
                query_counts[origin.query_key] = query_counts.get(origin.query_key, 0) + 1
        if any(count > self.query_cloud_reserve_per_query for count in query_counts.values()):
            raise ValueError("selection exceeds the per-query QueryCloud reserve")
        if self.query_cloud_reserve_per_query == 0 and query_counts:
            raise ValueError("selection cannot reserve QueryCloud candidates when disabled")

        replay = list(initial)
        protected_snapshot: tuple[str, ...] | None = None
        query_only_seen = False
        for item in self.decisions:
            has_prior_reason = CandidateSelectionReason.PRIOR_GATE_MATCH in item.reasons
            if query_only_seen and has_prior_reason:
                raise ValueError("prior-match decisions must precede query-only reserve decisions")
            if not has_prior_reason:
                if protected_snapshot is None:
                    replay_ids = set(replay)
                    protected_snapshot = tuple(
                        fragment_id for fragment_id in inputs if fragment_id in replay_ids
                    )
                query_only_seen = True
            if item.disposition is CandidateSelectionDisposition.FORCED:
                evicted_fragment_id = item.evicted_fragment_id
                if evicted_fragment_id is None:
                    raise ValueError("forced selection requires an evicted candidate")
                victim_index = replay.index(evicted_fragment_id)
                replay.pop(victim_index)
                replay.append(item.source_fragment_id)
        if protected_snapshot is None:
            replay_ids = set(replay)
            protected_snapshot = tuple(
                fragment_id for fragment_id in inputs if fragment_id in replay_ids
            )
        if protected_output != protected_snapshot:
            raise ValueError("protected selection output does not replay from decisions")
        replay_ids = set(replay)
        replay_output = tuple(fragment_id for fragment_id in inputs if fragment_id in replay_ids)
        if outputs != replay_output:
            raise ValueError("selection output does not replay from its decisions")
        return self

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": self.SCHEMA,
                **self.model_dump(mode="json"),
            }
        )


class AdaptiveCandidateTrace(_FrozenContract):
    source_fragment_id: str
    final_rank: int = Field(ge=1, le=500)
    harrier_score_hex: str
    text_sha256: str
    source_family: str
    body_proof_eligible: bool
    origins: tuple[CandidateOrigin, ...]
    query_cloud_reserved: bool = False

    @field_validator("text_sha256")
    @classmethod
    def _text_hash(cls, value: str) -> str:
        return _require_hash(value)

    @model_validator(mode="after")
    def _canonical_origins(self) -> AdaptiveCandidateTrace:
        if not self.origins or len(set(self.origins)) != len(self.origins):
            raise ValueError("candidate origins must be non-empty and unique")
        if self.origins != tuple(sorted(self.origins, key=_candidate_origin_sort_key)):
            raise ValueError("candidate origins must use canonical order")
        return self


class AdaptiveGateScanItem(_FrozenContract):
    source_fragment_id: str
    text_sha256: str
    rank: int = Field(ge=1, le=500)
    character_count: int = Field(ge=1, le=2_000_000)

    @field_validator("source_fragment_id")
    @classmethod
    def _fragment_id(cls, value: str) -> str:
        if type(value) is not str or _FRAGMENT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("source_fragment_id must use fragment_<key>")
        return value

    @field_validator("text_sha256")
    @classmethod
    def _text_hash(cls, value: str) -> str:
        return _require_hash(value)

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class AdaptiveGateScanReceipt(_FrozenContract):
    """Auditable boundary between local Gate scan and compact proof projection."""

    SCHEMA: ClassVar[str] = "dithyramba.adaptive_gate_scan_receipt/1.0"

    projection_policy: Literal["all_gate_matches_v1"] = "all_gate_matches_v1"
    items: tuple[AdaptiveGateScanItem, ...]
    candidate_count: int = Field(ge=0, le=500)
    character_count: int = Field(ge=0, le=100_000_000)
    candidate_set_hash: str
    matched_fragment_ids: tuple[str, ...]
    matched_character_count: int = Field(ge=0, le=100_000_000)
    matched_proof_hash: str

    @field_validator("candidate_set_hash", "matched_proof_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        return _require_hash(value)

    @model_validator(mode="after")
    def _closed_counts(self) -> AdaptiveGateScanReceipt:
        item_ids = tuple(item.source_fragment_id for item in self.items)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("Gate scan fragment IDs must be unique")
        if self.candidate_count != len(self.items):
            raise ValueError("Gate scan candidate_count must equal item count")
        if self.character_count != sum(item.character_count for item in self.items):
            raise ValueError("Gate scan character_count must equal item characters")
        if len(set(self.matched_fragment_ids)) != len(self.matched_fragment_ids):
            raise ValueError("matched proof fragment IDs must be unique")
        if not set(self.matched_fragment_ids).issubset(set(item_ids)):
            raise ValueError("matched proof must be a subset of the Gate scan")
        matched_characters = sum(
            item.character_count
            for item in self.items
            if item.source_fragment_id in set(self.matched_fragment_ids)
        )
        if self.matched_character_count != matched_characters:
            raise ValueError("matched proof characters must equal matched scan-item characters")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "projection_policy": self.projection_policy,
            "items": [item.semantic_payload() for item in self.items],
            "candidate_count": self.candidate_count,
            "character_count": self.character_count,
            "candidate_set_hash": self.candidate_set_hash,
            "matched_fragment_ids": list(self.matched_fragment_ids),
            "matched_character_count": self.matched_character_count,
            "matched_proof_hash": self.matched_proof_hash,
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())


class AdaptiveStageReceipt(_FrozenContract):
    SCHEMA: ClassVar[str] = "dithyramba.adaptive_stage_receipt/2.0"

    stage: str
    fts_result_hashes: tuple[str, ...]
    discovered_candidate_count: int = Field(ge=0, le=_MAX_DISCOVERED_CANDIDATES)
    candidate_count: int = Field(ge=0, le=500)
    pool_selection: AdaptiveCandidateSelectionReceipt
    union_selection: AdaptiveCandidateSelectionReceipt
    gate_scan: AdaptiveGateScanReceipt
    coverage: EvidenceCoverageResult
    coverage_result_hash: str
    decision: EvidenceGateDecision

    @model_validator(mode="after")
    def _closed_stage(self) -> AdaptiveStageReceipt:
        if self.stage not in {"base", "rescue", "query_cloud"}:
            raise ValueError("adaptive stage must be base, rescue, or query_cloud")
        if self.pool_selection.scope is not CandidateSelectionScope.HARRIER_POOL:
            raise ValueError("stage pool selection must use Harrier-pool scope")
        if self.union_selection.scope is not CandidateSelectionScope.FINAL_UNION:
            raise ValueError("stage union selection must use final-union scope")
        if self.discovered_candidate_count != len(self.pool_selection.input_fragment_ids):
            raise ValueError("discovered candidate count must match pool input")
        if self.candidate_count != len(self.pool_selection.output_fragment_ids):
            raise ValueError("candidate count must match scored Harrier pool")
        if set(self.union_selection.input_fragment_ids) != set(
            self.pool_selection.output_fragment_ids
        ):
            raise ValueError("final-union input must close over the Harrier pool")
        union_ids = self.union_selection.output_fragment_ids
        for item in self.gate_scan.items:
            if item.source_fragment_id not in set(union_ids):
                raise ValueError("Gate scan must be a subset of the final union")
            if union_ids[item.rank - 1] != item.source_fragment_id:
                raise ValueError("Gate scan rank must match final-union rank")
        if not set(self.pool_selection.protected_fragment_ids).issubset(
            set(self.gate_scan.matched_fragment_ids)
        ):
            raise ValueError("prior Gate matches must remain matched after this stage")
        if (
            self.pool_selection.protected_fragment_ids
            != self.union_selection.protected_fragment_ids
        ):
            raise ValueError("pool and union must protect the same prior Gate matches")
        if self.coverage_result_hash != self.coverage.result_hash:
            raise ValueError("stage coverage hash must match embedded coverage")
        if self.decision is not self.coverage.decision:
            raise ValueError("stage decision must match embedded coverage")
        if self.gate_scan.candidate_set_hash != self.coverage.candidate_set_hash:
            raise ValueError("stage Gate scan must match embedded coverage")
        coverage_matched_ids = {
            fragment_id
            for requirement in self.coverage.requirements
            for fragment_id in requirement.matched_fragment_ids
        }
        expected_matched_ids = tuple(
            item.source_fragment_id
            for item in self.gate_scan.items
            if item.source_fragment_id in coverage_matched_ids
        )
        if self.gate_scan.matched_fragment_ids != expected_matched_ids:
            raise ValueError("stage matched proof must equal all Gate-matched scan items")
        return self


class AdaptiveRecallResult(_FrozenContract):
    """Complete in-memory result; persistence is a later explicit adapter."""

    SCHEMA: ClassVar[str] = "dithyramba.adaptive_recall_result/2.2"

    original_question_sha256: str
    retrieval_corpus_hash: str
    alias_registry_hash: str
    gate_spec_hash: str
    config: AdaptiveRetrievalConfig
    harrier_batch_hash: str | None
    query_repair_plan: QueryRepairPlan | None
    stages: tuple[AdaptiveStageReceipt, ...]
    candidates: tuple[AdaptiveCandidateTrace, ...]
    matched_proof: tuple[EvidenceCandidate, ...]
    coverage: EvidenceCoverageResult

    @model_validator(mode="after")
    def _final_projection_closes(self) -> AdaptiveRecallResult:
        if not self.stages:
            raise ValueError("adaptive result requires at least one stage")
        stage_names = tuple(item.stage for item in self.stages)
        if stage_names not in {
            ("base",),
            ("base", "rescue"),
            ("base", "rescue", "query_cloud"),
        }:
            raise ValueError("adaptive stages must follow base, rescue, QueryCloud order")
        if any(stage.coverage.spec_hash != self.gate_spec_hash for stage in self.stages):
            raise ValueError("every stage must use the frozen Gate specification")
        has_query_cloud_stage = stage_names[-1] == "query_cloud"
        if has_query_cloud_stage != (self.query_repair_plan is not None):
            raise ValueError("QueryCloud stage and query-repair plan must exist together")
        if self.stages[0].pool_selection.protected_fragment_ids:
            raise ValueError("base stage cannot protect prior Gate matches")
        for previous, current in zip(self.stages, self.stages[1:], strict=False):
            expected_protected = previous.gate_scan.matched_fragment_ids
            if current.pool_selection.protected_fragment_ids != expected_protected:
                raise ValueError("stage pool must protect prior Gate matches")
            if current.union_selection.protected_fragment_ids != expected_protected:
                raise ValueError("stage union must protect prior Gate matches")
            if not set(previous.coverage.covered_requirement_ids).issubset(
                set(current.coverage.covered_requirement_ids)
            ):
                raise ValueError("covered evidence requirements must be monotonic")
            if not set(previous.gate_scan.matched_fragment_ids).issubset(
                set(current.gate_scan.matched_fragment_ids)
            ):
                raise ValueError("matched proof fragments must be monotonic")
        final_stage = self.stages[-1]
        final_scan = final_stage.gate_scan
        if final_stage.coverage_result_hash != self.coverage.result_hash:
            raise ValueError("final stage coverage hash must match final coverage")
        if final_stage.decision is not self.coverage.decision:
            raise ValueError("final stage decision must match final coverage")
        if final_stage.coverage.result_hash != self.coverage.result_hash:
            raise ValueError("embedded final stage coverage must match final coverage")
        if self.coverage.spec_hash != self.gate_spec_hash:
            raise ValueError("final coverage must use the frozen Gate specification")
        if final_scan.candidate_set_hash != self.coverage.candidate_set_hash:
            raise ValueError("final Gate scan must match coverage candidate set")
        candidate_ids = tuple(item.source_fragment_id for item in self.candidates)
        if candidate_ids != final_stage.union_selection.output_fragment_ids:
            raise ValueError("final candidate traces must match final-union output")
        if tuple(item.final_rank for item in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("final candidate ranks must be contiguous")
        scan_by_id = {item.source_fragment_id: item for item in final_scan.items}
        expected_scan_ids = tuple(
            item.source_fragment_id for item in self.candidates if item.body_proof_eligible
        )
        if expected_scan_ids != tuple(item.source_fragment_id for item in final_scan.items):
            raise ValueError("Gate scan must contain every proof-eligible final candidate")
        for trace in self.candidates:
            scan_item = scan_by_id.get(trace.source_fragment_id)
            if trace.body_proof_eligible:
                if (
                    scan_item is None
                    or scan_item.rank != trace.final_rank
                    or scan_item.text_sha256 != trace.text_sha256
                ):
                    raise ValueError("Gate scan must match candidate rank and text hash")
            elif scan_item is not None:
                raise ValueError("proof-ineligible candidate cannot enter the Gate scan")
        reserved_ids = {
            item.source_fragment_id
            for item in final_stage.union_selection.decisions
            if CandidateSelectionReason.QUERY_CLOUD_RESERVE in item.reasons
            and item.disposition is not CandidateSelectionDisposition.SKIPPED_CAPACITY
        }
        if any(
            item.query_cloud_reserved != (item.source_fragment_id in reserved_ids)
            for item in self.candidates
        ):
            raise ValueError("candidate reserve flags must match final-union receipt")
        matched_ids = tuple(item.source_fragment_id for item in self.matched_proof)
        if matched_ids != final_scan.matched_fragment_ids:
            raise ValueError("final matched proof must match the Gate scan receipt")
        if _matched_proof_hash(self.matched_proof) != final_scan.matched_proof_hash:
            raise ValueError("final matched proof hash does not match its receipt")
        if sum(len(item.text) for item in self.matched_proof) != final_scan.matched_character_count:
            raise ValueError("final matched proof character count does not match its receipt")
        for item in self.matched_proof:
            if item.rank > len(self.candidates):
                raise ValueError("matched proof rank must fit final candidate traces")
            trace = self.candidates[item.rank - 1]
            if (
                trace.source_fragment_id != item.source_fragment_id
                or trace.text_sha256 != sha256_hex(item.text.encode("utf-8"))
            ):
                raise ValueError("matched proof must close over final candidate traces")
        return self

    @property
    def config_hash(self) -> str:
        return self.config.config_hash

    @property
    def result_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": self.SCHEMA,
                "original_question_sha256": self.original_question_sha256,
                "retrieval_corpus_hash": self.retrieval_corpus_hash,
                "alias_registry_hash": self.alias_registry_hash,
                "gate_spec_hash": self.gate_spec_hash,
                "config_hash": self.config_hash,
                "harrier_batch_hash": self.harrier_batch_hash,
                "query_repair_plan_hash": (
                    None if self.query_repair_plan is None else self.query_repair_plan.plan_hash
                ),
                "stages": [item.model_dump(mode="json") for item in self.stages],
                "candidates": [item.model_dump(mode="json") for item in self.candidates],
                "matched_proof": [item.semantic_payload() for item in self.matched_proof],
                "coverage_result_hash": self.coverage.result_hash,
            }
        )


class _CandidatePool:
    def __init__(self, fragments: Mapping[str, AdaptiveFragment]) -> None:
        self.fragments = fragments
        self.origins: dict[str, list[CandidateOrigin]] = {}

    def add_result(
        self,
        result: FtsSearchResult,
        *,
        kind: CandidateOriginKind,
        query_key: str,
    ) -> None:
        for item in result.trace:
            self.add(
                item.source_fragment_id,
                CandidateOrigin(kind=kind, query_key=query_key, rank=item.rank),
            )

    def add(self, fragment_id: str, origin: CandidateOrigin) -> None:
        if fragment_id not in self.fragments:
            raise AdaptiveRecallError("retrieval returned a fragment outside the authorized set")
        existing = self.origins.get(fragment_id)
        if existing is None:
            if len(self.origins) >= _MAX_DISCOVERED_CANDIDATES:
                raise AdaptiveRecallError("bounded discovery candidate budget exceeded")
            existing = []
            self.origins[fragment_id] = existing
        if origin not in existing:
            existing.append(origin)

    def discovery_order(self) -> tuple[AdaptiveFragment, ...]:
        return tuple(
            self.fragments[fragment_id]
            for fragment_id in sorted(
                self.origins,
                key=lambda fragment_id: (
                    min(_candidate_origin_sort_key(origin) for origin in self.origins[fragment_id]),
                    fragment_id.encode("ascii"),
                ),
            )
        )

    def canonical_origins(self, fragment_id: str) -> tuple[CandidateOrigin, ...]:
        return tuple(sorted(self.origins[fragment_id], key=_candidate_origin_sort_key))

    def add_exact_phrases(self, phrases: tuple[str, ...], *, max_new: int) -> None:
        if max_new == 0:
            return
        matching_ids: set[str] = set()
        for phrase in phrases:
            needle = _match_text(phrase)
            matching_ids.update(
                fragment_id
                for fragment_id, fragment in self.fragments.items()
                if needle in _match_text(fragment.text)
            )
        novel_added = 0
        for fragment_id in sorted(matching_ids, key=lambda item: item.encode("ascii")):
            already_visible = fragment_id in self.origins
            if not already_visible and novel_added >= max_new:
                continue
            self.add(
                fragment_id,
                CandidateOrigin(
                    kind=CandidateOriginKind.EXACT_PHRASE,
                    query_key="q0",
                ),
            )
            if not already_visible:
                novel_added += 1

    def add_neighbors(self, *, max_new: int) -> None:
        if max_new == 0:
            return
        by_position = {
            (fragment.source_version_id, fragment.ordinal): fragment.source_fragment_id
            for fragment in self.fragments.values()
        }
        seeds = tuple(sorted(self.origins, key=lambda item: item.encode("ascii")))
        neighbor_ids: set[str] = set()
        for fragment_id in seeds:
            fragment = self.fragments[fragment_id]
            for ordinal in (fragment.ordinal - 1, fragment.ordinal + 1):
                neighbor_id = by_position.get((fragment.source_version_id, ordinal))
                if neighbor_id is not None:
                    neighbor_ids.add(neighbor_id)
        novel_added = 0
        for neighbor_id in sorted(neighbor_ids, key=lambda item: item.encode("ascii")):
            already_visible = neighbor_id in self.origins
            if not already_visible and novel_added >= max_new:
                continue
            self.add(
                neighbor_id,
                CandidateOrigin(
                    kind=CandidateOriginKind.NEIGHBOR,
                    query_key="q0",
                ),
            )
            if not already_visible:
                novel_added += 1


def execute_adaptive_retrieval(
    *,
    question: str,
    fragments: tuple[AdaptiveFragment, ...],
    gate_spec: EvidenceGateSpec,
    harrier: HarrierCandidateScorer,
    alias_registry: AliasRegistry | None = None,
    query_repair: QueryRepairProvider | None = None,
    config: AdaptiveRetrievalConfig | None = None,
) -> AdaptiveRecallResult:
    """Run FTS → q0 Harrier → proof admission → Gate with bounded repairs."""

    q0 = _bounded_text(question, maximum=2_000)
    if type(gate_spec) is not EvidenceGateSpec or gate_spec.question != q0:
        raise AdaptiveRecallError("gate spec must be frozen for the exact original question")
    if not fragments or len(fragments) > 100_000:
        raise AdaptiveRecallError("adaptive recall requires 1-100000 authorized fragments")
    ordered_fragments = tuple(
        sorted(fragments, key=lambda item: item.source_fragment_id.encode("ascii"))
    )
    fragment_map = {fragment.source_fragment_id: fragment for fragment in ordered_fragments}
    if len(fragment_map) != len(fragments):
        raise AdaptiveRecallError("adaptive fragment IDs must be unique")
    fragment_positions = {
        (fragment.source_version_id, fragment.ordinal) for fragment in ordered_fragments
    }
    if len(fragment_positions) != len(ordered_fragments):
        raise AdaptiveRecallError("adaptive source-version ordinal positions must be unique")
    registry = AliasRegistry() if alias_registry is None else alias_registry
    settings = AdaptiveRetrievalConfig() if config is None else config
    if type(registry) is not AliasRegistry or type(settings) is not AdaptiveRetrievalConfig:
        raise AdaptiveRecallError("adaptive registry and config must use exact contracts")
    # Score a bounded discovery union (including possible q1/q2) before the
    # proof-eligible Gate scan is selected.  Only exact Gate matches are later
    # projected as proof; discovery candidates never become agent context by
    # virtue of rank alone.
    pool = _CandidatePool(fragment_map)
    fts_fragments = tuple(fragment.fts_fragment() for fragment in ordered_fragments)
    stages: list[AdaptiveStageReceipt] = []
    plan: QueryRepairPlan | None = None
    final_batch: HarrierScoreBatch | None = None
    final_coverage: EvidenceCoverageResult | None = None
    final_scan: AdaptiveGateScanReceipt | None = None
    final_pool_selection: AdaptiveCandidateSelectionReceipt | None = None
    final_union_selection: AdaptiveCandidateSelectionReceipt | None = None
    final_matched_proof: tuple[EvidenceCandidate, ...] = ()
    final_order: tuple[AdaptiveFragment, ...] = ()
    fts_hashes: list[str] = []

    with PermittedFtsSession(fragments=fts_fragments) as session:
        for stage, limit in (
            ("base", settings.base_fts_candidates),
            ("rescue", settings.rescue_fts_candidates),
        ):
            if stage == "rescue" and final_coverage is not None and _terminal(final_coverage):
                break
            q0_result = session.search(question=q0, max_candidates=limit)
            fts_hashes.append(q0_result.result_hash)
            pool.add_result(
                q0_result,
                kind=CandidateOriginKind.ORIGINAL_FTS,
                query_key="q0",
            )
            boundary_question = _boundary_projection(q0)
            if boundary_question != q0:
                boundary_result = session.search(
                    question=boundary_question,
                    max_candidates=limit,
                )
                fts_hashes.append(boundary_result.result_hash)
                pool.add_result(
                    boundary_result,
                    kind=CandidateOriginKind.BOUNDARY_FTS,
                    query_key="q0_boundary",
                )
            alias_question = registry.expanded_question(q0)
            if alias_question is not None:
                alias_result = session.search(question=alias_question, max_candidates=limit)
                fts_hashes.append(alias_result.result_hash)
                pool.add_result(
                    alias_result,
                    kind=CandidateOriginKind.ALIAS_FTS,
                    query_key="q0_alias",
                )
            pool.add_exact_phrases(
                _quoted_phrases(q0),
                max_new=settings.max_exact_phrase_candidates,
            )
            pool.add_neighbors(max_new=settings.max_neighbor_candidates)
            (
                final_batch,
                final_order,
                final_coverage,
                final_scan,
                final_matched_proof,
                final_pool_selection,
                final_union_selection,
            ) = _score_and_gate(
                q0=q0,
                pool=pool,
                harrier=harrier,
                gate_spec=gate_spec,
                config=settings,
                protected_fragment_ids=(
                    () if final_scan is None else final_scan.matched_fragment_ids
                ),
            )
            stages.append(
                AdaptiveStageReceipt(
                    stage=stage,
                    fts_result_hashes=tuple(fts_hashes),
                    discovered_candidate_count=len(pool.origins),
                    candidate_count=len(final_pool_selection.output_fragment_ids),
                    pool_selection=final_pool_selection,
                    union_selection=final_union_selection,
                    gate_scan=final_scan,
                    coverage=final_coverage,
                    coverage_result_hash=final_coverage.result_hash,
                    decision=final_coverage.decision,
                )
            )

        if (
            final_coverage is not None
            and final_scan is not None
            and not _terminal(final_coverage)
            and query_repair is not None
        ):
            missing_keys = tuple(
                item.requirement_key
                for item in final_coverage.requirements
                if item.evidence_requirement_id in final_coverage.missing_requirement_ids
            )
            plan = query_repair.plan(
                question=q0,
                missing_requirement_keys=missing_keys,
                alias_registry=registry,
            )
            if type(plan) is not QueryRepairPlan:
                raise AdaptiveRecallError("query repair provider returned an invalid plan")
            q0_hash = sha256_hex(q0.encode("utf-8"))
            if plan.original_question_sha256 != q0_hash:
                raise AdaptiveRecallError("query repair plan is not bound to q0")
            if any(
                not set(query.target_requirement_keys).issubset(set(missing_keys))
                for query in plan.queries
            ):
                raise AdaptiveRecallError("query repair plan targets non-missing requirements")
            for query in plan.queries:
                result = session.search(
                    question=query.text,
                    max_candidates=settings.rescue_fts_candidates,
                )
                fts_hashes.append(result.result_hash)
                pool.add_result(
                    result,
                    kind=CandidateOriginKind.QUERY_CLOUD,
                    query_key=query.query_key,
                )
            pool.add_neighbors(max_new=settings.max_neighbor_candidates)
            (
                final_batch,
                final_order,
                final_coverage,
                final_scan,
                final_matched_proof,
                final_pool_selection,
                final_union_selection,
            ) = _score_and_gate(
                q0=q0,
                pool=pool,
                harrier=harrier,
                gate_spec=gate_spec,
                config=settings,
                protected_fragment_ids=final_scan.matched_fragment_ids,
            )
            stages.append(
                AdaptiveStageReceipt(
                    stage="query_cloud",
                    fts_result_hashes=tuple(fts_hashes),
                    discovered_candidate_count=len(pool.origins),
                    candidate_count=len(final_pool_selection.output_fragment_ids),
                    pool_selection=final_pool_selection,
                    union_selection=final_union_selection,
                    gate_scan=final_scan,
                    coverage=final_coverage,
                    coverage_result_hash=final_coverage.result_hash,
                    decision=final_coverage.decision,
                )
            )

        retrieval_corpus_hash = session.retrieval_corpus_hash

    if (
        final_coverage is None
        or final_scan is None
        or final_pool_selection is None
        or final_union_selection is None
    ):
        raise AdaptiveRecallError("adaptive retrieval produced no coverage decision")
    score_by_id = (
        {} if final_batch is None else {item.source_fragment_id: item for item in final_batch.items}
    )
    query_cloud_reserved_ids = {
        item.source_fragment_id
        for item in final_union_selection.decisions
        if CandidateSelectionReason.QUERY_CLOUD_RESERVE in item.reasons
        and item.disposition is not CandidateSelectionDisposition.SKIPPED_CAPACITY
    }
    return AdaptiveRecallResult(
        original_question_sha256=sha256_hex(q0.encode("utf-8")),
        retrieval_corpus_hash=retrieval_corpus_hash,
        alias_registry_hash=registry.registry_hash,
        gate_spec_hash=gate_spec.spec_hash,
        config=settings,
        harrier_batch_hash=None if final_batch is None else final_batch.batch_hash,
        query_repair_plan=plan,
        stages=tuple(stages),
        candidates=tuple(
            AdaptiveCandidateTrace(
                source_fragment_id=fragment.source_fragment_id,
                final_rank=rank,
                harrier_score_hex=score_by_id[fragment.source_fragment_id].score_hex,
                text_sha256=fragment.text_sha256,
                source_family=fragment.source_family,
                body_proof_eligible=fragment.body_proof_eligible,
                origins=pool.canonical_origins(fragment.source_fragment_id),
                query_cloud_reserved=(fragment.source_fragment_id in query_cloud_reserved_ids),
            )
            for rank, fragment in enumerate(final_order, start=1)
        ),
        matched_proof=final_matched_proof,
        coverage=final_coverage,
    )


def _score_and_gate(
    *,
    q0: str,
    pool: _CandidatePool,
    harrier: HarrierCandidateScorer,
    gate_spec: EvidenceGateSpec,
    config: AdaptiveRetrievalConfig,
    protected_fragment_ids: tuple[str, ...],
) -> tuple[
    HarrierScoreBatch | None,
    tuple[AdaptiveFragment, ...],
    EvidenceCoverageResult,
    AdaptiveGateScanReceipt,
    tuple[EvidenceCandidate, ...],
    AdaptiveCandidateSelectionReceipt,
    AdaptiveCandidateSelectionReceipt,
]:
    scoring_pool, pool_selection = _bounded_candidate_selection(
        pool.discovery_order(),
        origins=pool.origins,
        protected_fragment_ids=protected_fragment_ids,
        query_cloud_reserve_per_query=config.query_cloud_reserve_per_query,
        capacity=_MAX_SCORING_CANDIDATES,
        scope=CandidateSelectionScope.HARRIER_POOL,
    )
    candidates = tuple(
        CandidateText(
            source_fragment_id=fragment.source_fragment_id,
            source_id=fragment.source_id,
            text=fragment.text,
            text_sha256=fragment.text_sha256,
        )
        for fragment in sorted(
            scoring_pool,
            key=lambda item: item.source_fragment_id.encode("ascii"),
        )
    )
    if not candidates:
        coverage = EvidenceCoverageGate(gate_spec).evaluate(())
        matched_proof: tuple[EvidenceCandidate, ...] = ()
        final_union, union_selection = _bounded_candidate_selection(
            (),
            origins=pool.origins,
            protected_fragment_ids=protected_fragment_ids,
            query_cloud_reserve_per_query=config.query_cloud_reserve_per_query,
            capacity=config.max_union_candidates,
            scope=CandidateSelectionScope.FINAL_UNION,
        )
        return (
            None,
            final_union,
            coverage,
            _gate_scan_receipt((), coverage, matched_proof),
            matched_proof,
            pool_selection,
            union_selection,
        )
    batch = harrier.score(q0=q0, candidates=candidates)
    if type(batch) is not HarrierScoreBatch:
        raise AdaptiveRecallError("Harrier scorer returned an invalid score batch")
    if batch.original_question_sha256 != sha256_hex(q0.encode("utf-8")):
        raise AdaptiveRecallError("Harrier score batch is not anchored to q0")
    expected = {item.source_fragment_id: item.text_sha256 for item in candidates}
    actual = {item.source_fragment_id: item.text_sha256 for item in batch.items}
    if actual != expected:
        raise AdaptiveRecallError("Harrier score batch does not close over the candidate union")
    ranked = tuple(pool.fragments[item.source_fragment_id] for item in batch.items)
    diverse = _family_diverse_order(
        ranked,
        max_per_family=config.max_per_family_before_diversity,
        window=config.family_diversity_window_candidates,
    )
    selected, union_selection = _bounded_candidate_selection(
        diverse,
        origins=pool.origins,
        protected_fragment_ids=protected_fragment_ids,
        query_cloud_reserve_per_query=config.query_cloud_reserve_per_query,
        capacity=config.max_union_candidates,
        scope=CandidateSelectionScope.FINAL_UNION,
    )
    eligible = tuple(
        (rank, fragment)
        for rank, fragment in enumerate(selected, start=1)
        if fragment.body_proof_eligible
    )
    scanned_characters = sum(len(fragment.text) for _, fragment in eligible)
    if scanned_characters > config.max_gate_scan_characters:
        raise AdaptiveRecallError(
            "wide Gate scan character budget exceeded; split oversized fragments "
            "or raise the explicit local-only budget"
        )
    proof = tuple(
        fragment.evidence_candidate(rank=global_rank) for global_rank, fragment in eligible
    )
    coverage = EvidenceCoverageGate(gate_spec).evaluate(proof)
    matched_ids = {
        fragment_id
        for requirement in coverage.requirements
        for fragment_id in requirement.matched_fragment_ids
    }
    matched_proof = tuple(
        candidate for candidate in proof if candidate.source_fragment_id in matched_ids
    )
    return (
        batch,
        selected,
        coverage,
        _gate_scan_receipt(proof, coverage, matched_proof),
        matched_proof,
        pool_selection,
        union_selection,
    )


def _gate_scan_receipt(
    proof: tuple[EvidenceCandidate, ...],
    coverage: EvidenceCoverageResult,
    matched_proof: tuple[EvidenceCandidate, ...],
) -> AdaptiveGateScanReceipt:
    return AdaptiveGateScanReceipt(
        items=tuple(
            AdaptiveGateScanItem(
                source_fragment_id=item.source_fragment_id,
                text_sha256=sha256_hex(item.text.encode("utf-8")),
                rank=item.rank,
                character_count=len(item.text),
            )
            for item in proof
        ),
        candidate_count=len(proof),
        character_count=sum(len(item.text) for item in proof),
        candidate_set_hash=coverage.candidate_set_hash,
        matched_fragment_ids=tuple(item.source_fragment_id for item in matched_proof),
        matched_character_count=sum(len(item.text) for item in matched_proof),
        matched_proof_hash=_matched_proof_hash(matched_proof),
    )


def _matched_proof_hash(items: tuple[EvidenceCandidate, ...]) -> str:
    return canonical_sha256_hex(
        {
            "schema": "dithyramba.matched_proof_projection/1.0",
            "projection_policy": "all_gate_matches_v1",
            "items": [item.semantic_payload() for item in items],
        }
    )


def _family_diverse_order(
    ranked: tuple[AdaptiveFragment, ...],
    *,
    max_per_family: int,
    window: int,
) -> tuple[AdaptiveFragment, ...]:
    selected: list[AdaptiveFragment] = []
    deferred: list[AdaptiveFragment] = []
    counts: dict[str, int] = {}
    for fragment in ranked:
        count = counts.get(fragment.source_family, 0)
        if len(selected) < window and count < max_per_family:
            selected.append(fragment)
            counts[fragment.source_family] = count + 1
        else:
            deferred.append(fragment)
    # Diversity is a soft display-order policy, not a proof visibility limit.
    # Fill unused positions with the best deferred items, then preserve original
    # Harrier order for everything outside the window.
    selected.extend(deferred[: max(0, window - len(selected))])
    selected_ids = {item.source_fragment_id for item in selected}
    tail = tuple(item for item in ranked if item.source_fragment_id not in selected_ids)
    return tuple((*selected, *tail))


def _candidate_origin_sort_key(origin: CandidateOrigin) -> tuple[int, bytes, int]:
    priority = {
        CandidateOriginKind.EXACT_PHRASE: 0,
        CandidateOriginKind.ORIGINAL_FTS: 1,
        CandidateOriginKind.BOUNDARY_FTS: 2,
        CandidateOriginKind.ALIAS_FTS: 3,
        CandidateOriginKind.NEIGHBOR: 4,
        CandidateOriginKind.QUERY_CLOUD: 5,
    }
    return (
        priority[origin.kind],
        origin.query_key.encode("utf-8"),
        501 if origin.rank is None else origin.rank,
    )


def _candidate_sequence_hash(
    scope: CandidateSelectionScope,
    fragment_ids: tuple[str, ...],
) -> str:
    return canonical_sha256_hex(
        {
            "schema": "dithyramba.adaptive_candidate_sequence/1.0",
            "scope": scope.value,
            "fragment_ids": list(fragment_ids),
        }
    )


def _query_cloud_reserved_candidates(
    origins: Mapping[str, list[CandidateOrigin]],
    *,
    per_query: int,
) -> tuple[tuple[str, tuple[CandidateOrigin, ...]], ...]:
    """Select the best lexical discoveries from each bounded repair query.

    Harrier remains anchored to q0, so a cross-language or vocabulary-bridge
    query can discover an exact fragment that q0 semantics ranks below the
    final union.  The reserve keeps only the best FTS-ranked discoveries from
    q1/q2 visible to the deterministic Gate; it never promotes them to proof.
    """

    if per_query == 0:
        return ()
    by_query: dict[str, list[tuple[int, str, CandidateOrigin]]] = {}
    for fragment_id, fragment_origins in origins.items():
        for origin in fragment_origins:
            if origin.kind is CandidateOriginKind.QUERY_CLOUD and origin.rank is not None:
                by_query.setdefault(origin.query_key, []).append((origin.rank, fragment_id, origin))
    selected_origins: dict[str, list[CandidateOrigin]] = {}
    for query_key in sorted(by_query, key=lambda item: item.encode("utf-8")):
        ordered = sorted(
            by_query[query_key],
            key=lambda item: (item[0], item[1].encode("ascii")),
        )
        for _, fragment_id, origin in ordered[:per_query]:
            if fragment_id not in selected_origins:
                selected_origins[fragment_id] = []
            selected_origins[fragment_id].append(origin)
    return tuple(
        (
            fragment_id,
            tuple(sorted(selected_origins[fragment_id], key=_candidate_origin_sort_key)),
        )
        for fragment_id in sorted(selected_origins, key=lambda item: item.encode("ascii"))
    )


def _bounded_candidate_selection(
    ranked: tuple[AdaptiveFragment, ...],
    *,
    origins: Mapping[str, list[CandidateOrigin]],
    protected_fragment_ids: tuple[str, ...],
    query_cloud_reserve_per_query: int,
    capacity: int,
    scope: CandidateSelectionScope,
) -> tuple[tuple[AdaptiveFragment, ...], AdaptiveCandidateSelectionReceipt]:
    """Project a bounded set while preserving prior proof and q1/q2 discoveries."""

    input_ids = tuple(item.source_fragment_id for item in ranked)
    if len(set(input_ids)) != len(input_ids):
        raise AdaptiveRecallError("bounded selection input fragment IDs must be unique")
    by_id = {item.source_fragment_id: item for item in ranked}
    if not set(protected_fragment_ids).issubset(set(input_ids)):
        raise AdaptiveRecallError("prior Gate matches disappeared before bounded selection")
    if len(set(protected_fragment_ids)) != len(protected_fragment_ids):
        raise AdaptiveRecallError("prior Gate matches must be unique")
    reserve_origins = {
        fragment_id: selected_origins
        for fragment_id, selected_origins in _query_cloud_reserved_candidates(
            origins,
            per_query=query_cloud_reserve_per_query,
        )
        if fragment_id in by_id
    }
    protected_set = set(protected_fragment_ids)
    query_only_order = tuple(
        fragment_id
        for fragment_id in input_ids
        if fragment_id in reserve_origins and fragment_id not in protected_set
    )
    mandatory_ids = protected_set | set(query_only_order)
    if len(mandatory_ids) > capacity:
        raise AdaptiveRecallError(
            "protected proof plus QueryCloud reserve exceed bounded selection capacity"
        )

    initial_ids = input_ids[:capacity]
    selected_ids = set(initial_ids)
    decisions: list[AdaptiveCandidateSelectionDecision] = []
    evicted_ids: list[str] = []

    def apply_mandatory(fragment_id: str, *, protected: bool) -> None:
        item_reasons = (CandidateSelectionReason.PRIOR_GATE_MATCH,) if protected else ()
        item_query_origins = reserve_origins.get(fragment_id, ())
        if item_query_origins:
            item_reasons = (*item_reasons, CandidateSelectionReason.QUERY_CLOUD_RESERVE)
        if fragment_id in selected_ids:
            decisions.append(
                AdaptiveCandidateSelectionDecision(
                    source_fragment_id=fragment_id,
                    reasons=item_reasons,
                    query_origins=item_query_origins,
                    disposition=CandidateSelectionDisposition.ALREADY_VISIBLE,
                )
            )
            return
        victim_id = next(
            (
                candidate_id
                for candidate_id in reversed(initial_ids)
                if candidate_id in selected_ids and candidate_id not in mandatory_ids
            ),
            None,
        )
        if victim_id is None:
            raise AdaptiveRecallError(
                "mandatory candidate cannot enter bounded selection without proof loss"
            )
        selected_ids.remove(victim_id)
        selected_ids.add(fragment_id)
        evicted_ids.append(victim_id)
        decisions.append(
            AdaptiveCandidateSelectionDecision(
                source_fragment_id=fragment_id,
                reasons=item_reasons,
                query_origins=item_query_origins,
                disposition=CandidateSelectionDisposition.FORCED,
                evicted_fragment_id=victim_id,
            )
        )

    for fragment_id in protected_fragment_ids:
        apply_mandatory(fragment_id, protected=True)
    protected_output_ids = tuple(
        fragment_id for fragment_id in input_ids if fragment_id in selected_ids
    )
    for fragment_id in query_only_order:
        apply_mandatory(fragment_id, protected=False)

    output_ids = tuple(fragment_id for fragment_id in input_ids if fragment_id in selected_ids)
    output = tuple(by_id[fragment_id] for fragment_id in output_ids)
    already_visible_ids = tuple(
        item.source_fragment_id
        for item in decisions
        if item.disposition is CandidateSelectionDisposition.ALREADY_VISIBLE
    )
    forced_ids = tuple(
        item.source_fragment_id
        for item in decisions
        if item.disposition is CandidateSelectionDisposition.FORCED
    )
    skipped_ids = tuple(
        item.source_fragment_id
        for item in decisions
        if item.disposition is CandidateSelectionDisposition.SKIPPED_CAPACITY
    )
    receipt = AdaptiveCandidateSelectionReceipt(
        scope=scope,
        capacity=capacity,
        query_cloud_reserve_per_query=query_cloud_reserve_per_query,
        input_fragment_ids=input_ids,
        initial_fragment_ids=initial_ids,
        protected_fragment_ids=protected_fragment_ids,
        decisions=tuple(decisions),
        already_visible_fragment_ids=already_visible_ids,
        forced_fragment_ids=forced_ids,
        skipped_capacity_fragment_ids=skipped_ids,
        evicted_fragment_ids=tuple(evicted_ids),
        protected_output_fragment_ids=protected_output_ids,
        output_fragment_ids=output_ids,
        input_hash=_candidate_sequence_hash(scope, input_ids),
        initial_hash=_candidate_sequence_hash(scope, initial_ids),
        protected_output_hash=_candidate_sequence_hash(scope, protected_output_ids),
        output_hash=_candidate_sequence_hash(scope, output_ids),
    )
    return output, receipt


def _terminal(result: EvidenceCoverageResult) -> bool:
    return result.decision in {
        EvidenceGateDecision.READY,
        EvidenceGateDecision.GAP_PRESERVED,
        EvidenceGateDecision.GAP_CHALLENGED,
    }


def _quoted_phrases(question: str) -> tuple[str, ...]:
    values = tuple(
        _bounded_text(match.group(1), maximum=160)
        for match in _QUOTED_PHRASE_PATTERN.finditer(question)
    )
    return tuple(dict.fromkeys(values))


def _boundary_projection(question: str) -> str:
    projected = _CAMEL_BOUNDARY.sub(" ", question)
    projected = _LETTER_DIGIT_BOUNDARY.sub(" ", projected)
    return " ".join(projected.split())


def _bounded_text(value: str, *, maximum: int) -> str:
    if type(value) is not str:
        raise ValueError("text must be str")
    normalized = unicodedata.normalize("NFC", value)
    if (
        normalized != value
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise ValueError("text must be bounded, unpadded NFC text without NUL")
    return value


def _match_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _contains_name(normalized_haystack: str, name: str) -> bool:
    needle = _match_text(name)
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", normalized_haystack) is not None


def _require_hash(value: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError("hash must be lowercase SHA-256 hex")
    return value
