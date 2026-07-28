"""Provider-agnostic semantic and lexical question-route receipts.

This module records routing results; it does not compute embeddings.  A caller
may use any provider to produce question scores and then pass the finite Python
floats to :func:`build_route_candidate_receipt`.  Scores enter the canonical
receipt only through their exact ``float.hex`` representation.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    sha256_hex,
)

from .models import ResearchAtlasManifest

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_RECEIPT_ID_PATTERN = r"^route_candidate_receipt_[0-9a-f]{32}$"


class _RouteReceiptModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RouteCandidateChannel(StrEnum):
    """Independent score channels retained without cross-score comparison."""

    SEMANTIC = "semantic"
    LEXICAL = "lexical"


class RouteCandidateDecision(StrEnum):
    """Explicit upstream decision consumed by the bounded packet router."""

    ROUTED = "routed"
    ABSTAINED = "abstained"


class SemanticQuestionCandidate(_RouteReceiptModel):
    """One threshold-qualified dense candidate with its exact binary64 score."""

    channel: RouteCandidateChannel = RouteCandidateChannel.SEMANTIC
    question_id: str = Field(pattern=_ID_PATTERN)
    score_hex: str = Field(min_length=1, max_length=32)
    rank: int = Field(ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_semantic_candidate(self) -> Self:
        if self.channel is not RouteCandidateChannel.SEMANTIC:
            raise ValueError("semantic candidates must declare semantic channel lineage")
        _parse_canonical_float_hex(self.score_hex, label="semantic score")
        return self

    @property
    def score(self) -> float:
        return _parse_canonical_float_hex(self.score_hex, label="semantic score")


class LexicalQuestionCandidate(_RouteReceiptModel):
    """One caller-filtered strong lexical candidate."""

    channel: RouteCandidateChannel = RouteCandidateChannel.LEXICAL
    question_id: str = Field(pattern=_ID_PATTERN)
    score: int = Field(ge=1, le=1_000_000)
    rank: int = Field(ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_lexical_candidate(self) -> Self:
        if self.channel is not RouteCandidateChannel.LEXICAL:
            raise ValueError("lexical candidates must declare lexical channel lineage")
        return self


class RouteCandidateReceipt(_RouteReceiptModel):
    """Content-addressed ordered question candidates for one exact Atlas query."""

    SCHEMA: ClassVar[str] = "dithyramba.route_candidate_receipt/1.0"

    schema_id: str
    receipt_id: str = Field(pattern=_RECEIPT_ID_PATTERN)
    receipt_hash: str = Field(pattern=_HASH_PATTERN)
    query: str = Field(min_length=1, max_length=4_000)
    query_hash: str = Field(pattern=_HASH_PATTERN)
    atlas_id: str = Field(pattern=_ID_PATTERN)
    case_id: str = Field(pattern=_ID_PATTERN)
    manifest_hash: str = Field(pattern=_HASH_PATTERN)
    model_profile_hash: str = Field(pattern=_HASH_PATTERN)
    runtime_profile_hash: str = Field(pattern=_HASH_PATTERN)
    index_hash: str = Field(pattern=_HASH_PATTERN)
    semantic_threshold_hex: str = Field(min_length=1, max_length=32)
    semantic_candidates: tuple[SemanticQuestionCandidate, ...] = Field(max_length=1_000)
    lexical_candidates: tuple[LexicalQuestionCandidate, ...] = Field(max_length=1_000)
    decision: RouteCandidateDecision

    @field_validator("query")
    @classmethod
    def validate_normalized_query(cls, value: str) -> str:
        if value != unicodedata.normalize("NFC", value).strip():
            raise ValueError("receipt query must be stripped and NFC-normalized")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")

        threshold = _parse_canonical_float_hex(
            self.semantic_threshold_hex,
            label="semantic threshold",
        )
        _validate_semantic_order(self.semantic_candidates, threshold=threshold)
        _validate_lexical_order(self.lexical_candidates)

        candidate_ids = (
            *(item.question_id for item in self.semantic_candidates),
            *(item.question_id for item in self.lexical_candidates),
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("question candidates must be unique across route channels")
        if self.decision is RouteCandidateDecision.ROUTED and not candidate_ids:
            raise ValueError("routed receipts require at least one question candidate")

        if self.query_hash != sha256_hex(self.query.encode("utf-8")):
            raise ValueError("query_hash does not match the normalized query")
        payload = self.semantic_payload()
        if self.receipt_id != canonical_content_id("route_candidate_receipt", payload):
            raise ValueError("receipt_id does not match the canonical receipt content")
        if self.receipt_hash != canonical_sha256_hex(payload):
            raise ValueError("receipt_hash does not match the canonical receipt content")
        return self

    def semantic_payload(self) -> dict[str, object]:
        """Return the immutable payload covered by the receipt ID and hash."""

        return {
            "schema_id": self.schema_id,
            "query": self.query,
            "query_hash": self.query_hash,
            "atlas_id": self.atlas_id,
            "case_id": self.case_id,
            "manifest_hash": self.manifest_hash,
            "model_profile_hash": self.model_profile_hash,
            "runtime_profile_hash": self.runtime_profile_hash,
            "index_hash": self.index_hash,
            "semantic_threshold_hex": self.semantic_threshold_hex,
            "semantic_candidates": [
                item.model_dump(mode="json") for item in self.semantic_candidates
            ],
            "lexical_candidates": [
                item.model_dump(mode="json") for item in self.lexical_candidates
            ],
            "decision": self.decision.value,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.semantic_payload())

    @property
    def ordered_candidates(
        self,
    ) -> tuple[SemanticQuestionCandidate | LexicalQuestionCandidate, ...]:
        """Return semantic seeds first, then caller-filtered lexical-only seeds."""

        return (*self.semantic_candidates, *self.lexical_candidates)

    def validate_for(
        self,
        manifest: ResearchAtlasManifest,
        *,
        manifest_hash: str,
        query: str,
    ) -> None:
        """Fail closed when this receipt is stale or names an unknown question."""

        normalized_query = normalize_route_query(query)
        current_hash = canonical_sha256_hex(manifest.model_dump(mode="json"))
        if manifest_hash != current_hash:
            raise ValueError("router manifest_hash is stale for the current manifest")
        if self.query != normalized_query or self.query_hash != sha256_hex(
            normalized_query.encode("utf-8")
        ):
            raise ValueError("route receipt is stale for the normalized query")
        if (
            self.atlas_id != manifest.atlas_id
            or self.case_id != manifest.case_id
            or self.manifest_hash != manifest_hash
        ):
            raise ValueError("route receipt is stale for the Atlas manifest")
        known_question_ids = {item.question_id for item in manifest.questions}
        unknown = sorted(
            {
                item.question_id
                for item in self.ordered_candidates
                if item.question_id not in known_question_ids
            }
        )
        if unknown:
            raise ValueError(f"route receipt references unknown question IDs: {', '.join(unknown)}")


ScorePairs = Mapping[str, float] | Iterable[tuple[str, float]]
LexicalScorePairs = Mapping[str, int] | Iterable[tuple[str, int]]


def build_route_candidate_receipt(
    manifest: ResearchAtlasManifest,
    *,
    manifest_hash: str,
    query: str,
    model_profile_hash: str,
    runtime_profile_hash: str,
    index_hash: str,
    semantic_threshold: float,
    semantic_scores: ScorePairs = (),
    lexical_scores: LexicalScorePairs = (),
    decision: RouteCandidateDecision | None = None,
) -> RouteCandidateReceipt:
    """Build a canonical receipt from already-computed, provider-free scores.

    Semantic scores below ``semantic_threshold`` are intentionally absent from
    the receipt.  Lexical scores are assumed to have already passed the
    caller's explicit "strong lexical" policy; non-positive values are refused
    rather than silently promoted.
    """

    normalized_query = normalize_route_query(query)
    if manifest_hash != canonical_sha256_hex(manifest.model_dump(mode="json")):
        raise ValueError("manifest_hash is stale for the supplied manifest")
    for label, value in (
        ("model_profile_hash", model_profile_hash),
        ("runtime_profile_hash", runtime_profile_hash),
        ("index_hash", index_hash),
    ):
        if type(value) is not str or re.fullmatch(_HASH_PATTERN, value) is None:
            raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    if type(semantic_threshold) is not float or not math.isfinite(semantic_threshold):
        raise ValueError("semantic_threshold must be a finite float")
    if decision is not None and type(decision) is not RouteCandidateDecision:
        raise ValueError("decision must use RouteCandidateDecision")

    known_question_ids = {item.question_id for item in manifest.questions}
    semantic_pairs = _semantic_pairs(semantic_scores)
    lexical_pairs = _lexical_pairs(lexical_scores)
    supplied_ids = tuple(question_id for question_id, _ in (*semantic_pairs, *lexical_pairs))
    if len(supplied_ids) != len(set(supplied_ids)):
        raise ValueError("question candidates must be unique across route channels")
    unknown = sorted(set(supplied_ids).difference(known_question_ids))
    if unknown:
        raise ValueError(f"unknown question IDs: {', '.join(unknown)}")

    threshold_hex = semantic_threshold.hex()
    qualified_semantic = sorted(
        (
            (question_id, score)
            for question_id, score in semantic_pairs
            if score >= semantic_threshold
        ),
        key=lambda item: (-item[1], item[0]),
    )
    ordered_lexical = sorted(lexical_pairs, key=lambda item: (-item[1], item[0]))
    semantic_candidates = tuple(
        SemanticQuestionCandidate(
            question_id=question_id,
            score_hex=score.hex(),
            rank=rank,
        )
        for rank, (question_id, score) in enumerate(qualified_semantic, start=1)
    )
    lexical_candidates = tuple(
        LexicalQuestionCandidate(question_id=question_id, score=score, rank=rank)
        for rank, (question_id, score) in enumerate(ordered_lexical, start=1)
    )
    selected_decision = decision or (
        RouteCandidateDecision.ROUTED
        if semantic_candidates or lexical_candidates
        else RouteCandidateDecision.ABSTAINED
    )
    payload: dict[str, object] = {
        "schema_id": RouteCandidateReceipt.SCHEMA,
        "query": normalized_query,
        "query_hash": sha256_hex(normalized_query.encode("utf-8")),
        "atlas_id": manifest.atlas_id,
        "case_id": manifest.case_id,
        "manifest_hash": manifest_hash,
        "model_profile_hash": model_profile_hash,
        "runtime_profile_hash": runtime_profile_hash,
        "index_hash": index_hash,
        "semantic_threshold_hex": threshold_hex,
        "semantic_candidates": [item.model_dump(mode="json") for item in semantic_candidates],
        "lexical_candidates": [item.model_dump(mode="json") for item in lexical_candidates],
        "decision": selected_decision.value,
    }
    return RouteCandidateReceipt(
        schema_id=RouteCandidateReceipt.SCHEMA,
        receipt_id=canonical_content_id("route_candidate_receipt", payload),
        receipt_hash=canonical_sha256_hex(payload),
        query=normalized_query,
        query_hash=sha256_hex(normalized_query.encode("utf-8")),
        atlas_id=manifest.atlas_id,
        case_id=manifest.case_id,
        manifest_hash=manifest_hash,
        model_profile_hash=model_profile_hash,
        runtime_profile_hash=runtime_profile_hash,
        index_hash=index_hash,
        semantic_threshold_hex=threshold_hex,
        semantic_candidates=semantic_candidates,
        lexical_candidates=lexical_candidates,
        decision=selected_decision,
    )


def normalize_route_query(query: str) -> str:
    """Apply the one normalization rule shared by receipt production/consumption."""

    if type(query) is not str:
        raise ValueError("query must be a string")
    normalized = unicodedata.normalize("NFC", query).strip()
    if not normalized:
        raise ValueError("query must not be blank")
    if len(normalized) > 4_000:
        raise ValueError("query must be at most 4000 characters")
    return normalized


def _semantic_pairs(scores: ScorePairs) -> tuple[tuple[str, float], ...]:
    pairs = tuple(scores.items()) if isinstance(scores, Mapping) else tuple(scores)
    identifiers: list[str] = []
    validated: list[tuple[str, float]] = []
    for question_id, score in pairs:
        if type(question_id) is not str or re.fullmatch(_ID_PATTERN, question_id) is None:
            raise ValueError("semantic score question IDs must be canonical Atlas IDs")
        if type(score) is not float or not math.isfinite(score):
            raise ValueError("semantic scores must be finite floats")
        identifiers.append(question_id)
        validated.append((question_id, score))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("semantic score question IDs must be unique")
    return tuple(validated)


def _lexical_pairs(scores: LexicalScorePairs) -> tuple[tuple[str, int], ...]:
    pairs = tuple(scores.items()) if isinstance(scores, Mapping) else tuple(scores)
    identifiers: list[str] = []
    validated: list[tuple[str, int]] = []
    for question_id, score in pairs:
        if type(question_id) is not str or re.fullmatch(_ID_PATTERN, question_id) is None:
            raise ValueError("lexical score question IDs must be canonical Atlas IDs")
        if type(score) is not int or score < 1 or score > 1_000_000:
            raise ValueError("lexical scores must be positive bounded integers")
        identifiers.append(question_id)
        validated.append((question_id, score))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("lexical score question IDs must be unique")
    return tuple(validated)


def _parse_canonical_float_hex(value: str, *, label: str) -> float:
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} must use canonical float.hex syntax") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    if parsed.hex() != value:
        raise ValueError(f"{label} must use the exact canonical float.hex representation")
    return parsed


def _validate_semantic_order(
    candidates: tuple[SemanticQuestionCandidate, ...],
    *,
    threshold: float,
) -> None:
    expected = tuple(sorted(candidates, key=lambda item: (-item.score, item.question_id)))
    if candidates != expected:
        raise ValueError("semantic candidates must be ordered by score then question_id")
    if tuple(item.rank for item in candidates) != tuple(range(1, len(candidates) + 1)):
        raise ValueError("semantic candidate ranks must be contiguous from one")
    if any(item.score < threshold for item in candidates):
        raise ValueError("semantic candidate score is below the receipt threshold")


def _validate_lexical_order(candidates: tuple[LexicalQuestionCandidate, ...]) -> None:
    expected = tuple(sorted(candidates, key=lambda item: (-item.score, item.question_id)))
    if candidates != expected:
        raise ValueError("lexical candidates must be ordered by score then question_id")
    if tuple(item.rank for item in candidates) != tuple(range(1, len(candidates) + 1)):
        raise ValueError("lexical candidate ranks must be contiguous from one")
