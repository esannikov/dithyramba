"""Deterministic task routing into a compact, accepted Research Atlas packet.

The router is deliberately provider-free.  It ranks manifest records by
transparent lexical overlap, follows only declared Atlas links, and emits the
smallest closed subgraph needed by a downstream research agent.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

from .models import (
    AtlasEvidence,
    AtlasGap,
    AtlasHypothesis,
    AtlasQuestion,
    AtlasRelation,
    AtlasSource,
    AtlasTimelineEvent,
    ResearchAtlasManifest,
)
from .semantic_route import (
    LexicalQuestionCandidate,
    RouteCandidateChannel,
    RouteCandidateDecision,
    RouteCandidateReceipt,
    SemanticQuestionCandidate,
)

_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_STOP_WORDS = frozenset(
    {
        "але",
        "без",
        "був",
        "була",
        "були",
        "було",
        "від",
        "він",
        "вона",
        "вони",
        "для",
        "його",
        "коли",
        "лише",
        "можна",
        "про",
        "після",
        "робота",
        "роботи",
        "роботу",
        "року",
        "роки",
        "сказати",
        "достовірно",
        "відомо",
        "так",
        "такий",
        "таке",
        "також",
        "того",
        "цей",
        "через",
        "чому",
        "щодо",
        "який",
        "яка",
        "яке",
        "як",
        "the",
        "and",
        "for",
        "from",
        "how",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


class _PacketModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class RouteStatus(StrEnum):
    ROUTED = "routed"
    NO_MATCH = "no_match"
    BUDGET_EXCEEDED = "budget_exceeded"


class RouteObjectKind(StrEnum):
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    RELATION = "relation"
    TIMELINE = "timeline"
    GAP = "gap"


class RouteLimits(_PacketModel):
    """Maximum lexical seeds; declared graph closure may add linked records."""

    max_questions: int = Field(default=4, ge=1, le=20)
    max_hypotheses: int = Field(default=3, ge=0, le=12)
    max_timeline_events: int = Field(default=4, ge=0, le=20)
    max_gaps: int = Field(default=3, ge=0, le=12)


class RouteMatch(_PacketModel):
    object_kind: RouteObjectKind
    object_id: str = Field(pattern=_ID_PATTERN)
    score: int = Field(ge=1)
    matched_terms: tuple[str, ...] = Field(min_length=1, max_length=64)


class RouteBudget(_PacketModel):
    """Hard maxima for v2 seed admission and explicit graph expansion."""

    max_questions: int = Field(default=4, ge=0, le=100)
    max_linked_hypotheses: int = Field(default=3, ge=0, le=100)
    max_relations: int = Field(default=20, ge=0, le=200)
    max_timeline_events: int = Field(default=4, ge=0, le=200)
    max_gaps: int = Field(default=3, ge=0, le=100)
    max_evidence: int = Field(default=100, ge=0, le=1_000)
    max_sources: int = Field(default=50, ge=0, le=200)


class RouteTraceStage(StrEnum):
    SEED = "seed"
    GRAPH_EXPANSION = "graph_expansion"


class RouteTraceEntry(_PacketModel):
    """One admitted seed or one explicitly linked graph-expansion object."""

    stage: RouteTraceStage
    object_kind: RouteObjectKind
    object_id: str = Field(pattern=_ID_PATTERN)
    channel: RouteCandidateChannel | None = None
    channel_rank: int | None = Field(default=None, ge=1, le=1_000)
    semantic_score_hex: str | None = Field(default=None, min_length=1, max_length=32)
    lexical_score: int | None = Field(default=None, ge=1, le=1_000_000)
    parent_object_kind: RouteObjectKind | None = None
    parent_object_id: str | None = Field(default=None, pattern=_ID_PATTERN)

    @model_validator(mode="after")
    def validate_trace_entry(self) -> Self:
        if self.stage is RouteTraceStage.SEED:
            if self.object_kind is not RouteObjectKind.QUESTION:
                raise ValueError("route seeds must be questions")
            if self.channel is None or self.channel_rank is None:
                raise ValueError("route seeds require explicit channel lineage and rank")
            if self.parent_object_kind is not None or self.parent_object_id is not None:
                raise ValueError("route seeds cannot declare a graph-expansion parent")
            if self.channel is RouteCandidateChannel.SEMANTIC:
                if self.semantic_score_hex is None or self.lexical_score is not None:
                    raise ValueError("semantic seeds require only an exact semantic score")
                _require_float_hex(self.semantic_score_hex)
            elif self.lexical_score is None or self.semantic_score_hex is not None:
                raise ValueError("lexical seeds require only an integer lexical score")
            return self

        if any(
            item is not None
            for item in (
                self.channel,
                self.channel_rank,
                self.semantic_score_hex,
                self.lexical_score,
            )
        ):
            raise ValueError("graph expansion cannot carry a seed channel score")
        if self.parent_object_kind is None or self.parent_object_id is None:
            raise ValueError("graph expansion requires an explicit parent object")
        return self


class RouteOmissionReason(StrEnum):
    QUESTION_BUDGET = "question_budget"
    HYPOTHESIS_BUDGET = "hypothesis_budget"
    RELATION_BUDGET = "relation_budget"
    TIMELINE_BUDGET = "timeline_budget"
    GAP_BUDGET = "gap_budget"
    EVIDENCE_BUDGET = "evidence_budget"
    SOURCE_BUDGET = "source_budget"
    REFERENCE_BUDGET = "reference_budget"


class RouteOmission(_PacketModel):
    """One deterministic refusal to admit an otherwise eligible object."""

    object_kind: RouteObjectKind
    object_id: str = Field(pattern=_ID_PATTERN)
    reason: RouteOmissionReason


class CompactMemoryPacket(_PacketModel):
    """Content-addressed accepted-memory subgraph for exactly one task."""

    SCHEMA: ClassVar[str] = "dithyramba.compact_memory_packet/1.0"

    schema_id: str
    packet_id: str = Field(pattern=r"^memory_packet_[0-9a-f]{32}$")
    packet_hash: str = Field(pattern=_HASH_PATTERN)
    router_profile: str
    route_status: RouteStatus
    query: str = Field(min_length=1, max_length=4_000)
    query_terms: tuple[str, ...] = Field(max_length=128)
    atlas_id: str = Field(pattern=_ID_PATTERN)
    case_id: str = Field(pattern=_ID_PATTERN)
    manifest_hash: str = Field(pattern=_HASH_PATTERN)
    route_matches: tuple[RouteMatch, ...] = Field(max_length=100)
    sources: tuple[AtlasSource, ...] = Field(max_length=200)
    evidence: tuple[AtlasEvidence, ...] = Field(max_length=1_000)
    questions: tuple[AtlasQuestion, ...] = Field(max_length=100)
    hypotheses: tuple[AtlasHypothesis, ...] = Field(max_length=100)
    relations: tuple[AtlasRelation, ...] = Field(max_length=200)
    timeline: tuple[AtlasTimelineEvent, ...] = Field(max_length=200)
    gaps: tuple[AtlasGap, ...] = Field(max_length=100)

    @model_validator(mode="after")
    def validate_packet(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        payload = self.semantic_payload()
        if self.packet_id != canonical_content_id("memory_packet", payload):
            raise ValueError("packet_id does not match the canonical packet content")
        if self.packet_hash != canonical_sha256_hex(payload):
            raise ValueError("packet_hash does not match the canonical packet content")

        source_ids = _ids((item.source_id for item in self.sources), "source")
        evidence_ids = _ids((item.evidence_id for item in self.evidence), "evidence")
        question_ids = _ids((item.question_id for item in self.questions), "question")
        hypothesis_ids = _ids((item.hypothesis_id for item in self.hypotheses), "hypothesis")
        for evidence in self.evidence:
            _require_refs((evidence.source_id,), source_ids, "evidence source")
        for question in self.questions:
            _require_refs(question.evidence_ids, evidence_ids, "question evidence")
        for hypothesis in self.hypotheses:
            _require_refs(hypothesis.evidence_ids, evidence_ids, "hypothesis evidence")
            _require_refs(
                hypothesis.counterevidence_ids,
                evidence_ids,
                "hypothesis counterevidence",
            )
            _require_refs(hypothesis.question_ids, question_ids, "hypothesis question")
        for relation in self.relations:
            _require_refs(
                (relation.source_hypothesis_id, relation.target_hypothesis_id),
                hypothesis_ids,
                "relation hypothesis",
            )
            _require_refs(relation.evidence_ids, evidence_ids, "relation evidence")
        for event in self.timeline:
            _require_refs(event.evidence_ids, evidence_ids, "timeline evidence")
            _require_refs(event.question_ids, question_ids, "timeline question")
            _require_refs(event.hypothesis_ids, hypothesis_ids, "timeline hypothesis")
        for gap in self.gaps:
            _require_refs(gap.related_question_ids, question_ids, "gap question")
            _require_refs(gap.related_hypothesis_ids, hypothesis_ids, "gap hypothesis")

        is_empty = not any(
            (self.sources, self.evidence, self.questions, self.hypotheses, self.timeline, self.gaps)
        )
        if self.schema_id == CompactMemoryPacket.SCHEMA:
            if self.route_status is RouteStatus.BUDGET_EXCEEDED:
                raise ValueError("budget_exceeded is not valid in compact packet schema 1.0")
            if any(item.object_kind is RouteObjectKind.RELATION for item in self.route_matches):
                raise ValueError(
                    "relation route matches are not valid in compact packet schema 1.0"
                )
            if self.route_status is RouteStatus.NO_MATCH and not is_empty:
                raise ValueError("no_match packets must not contain routed memory")
            if self.route_status is RouteStatus.ROUTED and (is_empty or not self.route_matches):
                raise ValueError("routed packets require memory and route matches")
        return self

    def semantic_payload(self) -> dict[str, object]:
        """Return the immutable content covered by packet_id and packet_hash."""

        return {
            "schema_id": self.schema_id,
            "router_profile": self.router_profile,
            "route_status": self.route_status.value,
            "query": self.query,
            "query_terms": list(self.query_terms),
            "atlas_id": self.atlas_id,
            "case_id": self.case_id,
            "manifest_hash": self.manifest_hash,
            "route_matches": [item.model_dump(mode="json") for item in self.route_matches],
            "sources": [item.model_dump(mode="json") for item in self.sources],
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
            "questions": [item.model_dump(mode="json") for item in self.questions],
            "hypotheses": [item.model_dump(mode="json") for item in self.hypotheses],
            "relations": [item.model_dump(mode="json") for item in self.relations],
            "timeline": [item.model_dump(mode="json") for item in self.timeline],
            "gaps": [item.model_dump(mode="json") for item in self.gaps],
        }


class CompactMemoryPacketV1_1(CompactMemoryPacket):
    """Receipt-bound compact packet with bounded, omission-aware graph closure."""

    SCHEMA: ClassVar[str] = "dithyramba.compact_memory_packet/1.1"

    route_receipt_id: str = Field(pattern=r"^route_candidate_receipt_[0-9a-f]{32}$")
    route_receipt_hash: str = Field(pattern=_HASH_PATTERN)
    route_budget: RouteBudget
    route_trace: tuple[RouteTraceEntry, ...] = Field(max_length=700)
    omissions: tuple[RouteOmission, ...] = Field(max_length=11_000)

    @model_validator(mode="after")
    def validate_v1_1_packet(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if self.route_receipt_id != (f"route_candidate_receipt_{self.route_receipt_hash[:32]}"):
            raise ValueError("route receipt ID/hash binding is inconsistent")
        if self.route_matches:
            raise ValueError("schema 1.1 uses route_trace instead of legacy route_matches")

        trace_keys = tuple((item.object_kind, item.object_id) for item in self.route_trace)
        if len(trace_keys) != len(set(trace_keys)):
            raise ValueError("route_trace object entries must be unique")
        omission_keys = tuple((item.object_kind, item.object_id) for item in self.omissions)
        if len(omission_keys) != len(set(omission_keys)):
            raise ValueError("route omissions must be unique")
        overlap = set(trace_keys).intersection(omission_keys)
        if overlap:
            raise ValueError("one route object cannot be both admitted and omitted")

        expected_trace_keys = {
            *((RouteObjectKind.QUESTION, item.question_id) for item in self.questions),
            *((RouteObjectKind.HYPOTHESIS, item.hypothesis_id) for item in self.hypotheses),
            *((RouteObjectKind.RELATION, item.relation_id) for item in self.relations),
            *((RouteObjectKind.TIMELINE, item.event_id) for item in self.timeline),
            *((RouteObjectKind.GAP, item.gap_id) for item in self.gaps),
        }
        if set(trace_keys) != expected_trace_keys:
            raise ValueError("route_trace must exactly cover every admitted graph object")
        admitted_trace_keys: set[tuple[RouteObjectKind, str]] = set()
        for item in self.route_trace:
            if item.stage is RouteTraceStage.GRAPH_EXPANSION:
                parent = (item.parent_object_kind, item.parent_object_id)
                if parent not in admitted_trace_keys:
                    raise ValueError(
                        "graph-expansion trace parents must precede their admitted children"
                    )
            admitted_trace_keys.add((item.object_kind, item.object_id))

        budget_counts = (
            (len(self.questions), self.route_budget.max_questions, "question"),
            (
                len(self.hypotheses),
                self.route_budget.max_linked_hypotheses,
                "hypothesis",
            ),
            (len(self.relations), self.route_budget.max_relations, "relation"),
            (
                len(self.timeline),
                self.route_budget.max_timeline_events,
                "timeline",
            ),
            (len(self.gaps), self.route_budget.max_gaps, "gap"),
            (len(self.evidence), self.route_budget.max_evidence, "evidence"),
            (len(self.sources), self.route_budget.max_sources, "source"),
        )
        for actual, maximum, label in budget_counts:
            if actual > maximum:
                raise ValueError(f"schema 1.1 {label} closure exceeds its route budget")

        is_empty = not any(
            (
                self.sources,
                self.evidence,
                self.questions,
                self.hypotheses,
                self.relations,
                self.timeline,
                self.gaps,
            )
        )
        seed_count = sum(item.stage is RouteTraceStage.SEED for item in self.route_trace)
        if self.route_status is RouteStatus.ROUTED:
            if is_empty or seed_count < 1:
                raise ValueError("routed schema 1.1 packets require an admitted seed")
        elif self.route_status is RouteStatus.NO_MATCH:
            if not is_empty or self.route_trace or self.omissions:
                raise ValueError("no_match schema 1.1 packets must be empty without omissions")
        elif not is_empty or self.route_trace or not self.omissions:
            raise ValueError(
                "budget_exceeded schema 1.1 packets must be empty with explicit omissions"
            )
        return self

    def semantic_payload(self) -> dict[str, object]:
        """Return the exact 1.1 identity payload, including routing provenance."""

        payload = super().semantic_payload()
        payload.update(
            {
                "route_receipt_id": self.route_receipt_id,
                "route_receipt_hash": self.route_receipt_hash,
                "route_budget": self.route_budget.model_dump(mode="json"),
                "route_trace": [item.model_dump(mode="json") for item in self.route_trace],
                "omissions": [item.model_dump(mode="json") for item in self.omissions],
            }
        )
        return payload


class TaskRouter:
    """Transparent lexical router followed by deterministic graph closure."""

    PROFILE = "lexical_graph_closure_v1"
    PROFILE_V2 = "semantic_receipt_graph_closure_v2"

    def __init__(self, manifest: ResearchAtlasManifest, manifest_hash: str) -> None:
        if not re.fullmatch(_HASH_PATTERN, manifest_hash):
            raise ValueError("manifest_hash must be 64 lowercase hexadecimal characters")
        self._manifest = manifest
        self._manifest_hash = manifest_hash

    def route(self, query: str, *, limits: RouteLimits | None = None) -> CompactMemoryPacket:
        normalized_query = unicodedata.normalize("NFC", query).strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if len(normalized_query) > 4_000:
            raise ValueError("query must be at most 4000 characters")
        selected_limits = limits or RouteLimits()
        query_terms = _terms(normalized_query)
        if not query_terms:
            return self._build_packet(normalized_query, (), (), (), (), (), ())

        question_scores = self._rank_questions(query_terms)
        hypothesis_scores = self._rank_hypotheses(query_terms)
        timeline_scores = self._rank_timeline(query_terms)
        gap_scores = self._rank_gaps(query_terms)
        if not any((question_scores, hypothesis_scores, timeline_scores, gap_scores)):
            return self._build_packet(normalized_query, query_terms, (), (), (), ())

        question_ids = {
            item_id for _, item_id, _ in question_scores[: selected_limits.max_questions]
        }
        linked_hypotheses = [
            item
            for item in self._manifest.hypotheses
            if question_ids.intersection(item.question_ids)
        ]
        linked_hypothesis_ids = {item.hypothesis_id for item in linked_hypotheses}
        scored_hypothesis_ids = [
            item_id
            for _, item_id, _ in hypothesis_scores
            if not question_ids or item_id in linked_hypothesis_ids
        ]
        hypothesis_ids = _capped_ids(
            scored_hypothesis_ids,
            (item.hypothesis_id for item in linked_hypotheses),
            selected_limits.max_hypotheses,
        )

        linked_events = [
            item
            for item in self._manifest.timeline
            if question_ids.intersection(item.question_ids)
            or hypothesis_ids.intersection(item.hypothesis_ids)
        ]
        linked_event_ids = {item.event_id for item in linked_events}
        scored_event_ids = [
            item_id
            for _, item_id, _ in timeline_scores
            if (not question_ids and not hypothesis_ids) or item_id in linked_event_ids
        ]
        event_ids = _capped_ids(
            scored_event_ids,
            (item.event_id for item in linked_events),
            selected_limits.max_timeline_events,
        )

        linked_gaps = [
            item
            for item in self._manifest.gaps
            if question_ids.intersection(item.related_question_ids)
            or hypothesis_ids.intersection(item.related_hypothesis_ids)
        ]
        linked_gap_ids = {item.gap_id for item in linked_gaps}
        scored_gap_ids = [
            item_id
            for _, item_id, _ in gap_scores
            if (not question_ids and not hypothesis_ids) or item_id in linked_gap_ids
        ]
        gap_ids = _capped_ids(
            scored_gap_ids,
            (item.gap_id for item in linked_gaps),
            selected_limits.max_gaps,
        )

        question_ids, hypothesis_ids = self._close_object_references(
            question_ids,
            hypothesis_ids,
            event_ids,
            gap_ids,
        )

        questions = tuple(
            item for item in self._manifest.questions if item.question_id in question_ids
        )
        hypotheses = tuple(
            item for item in self._manifest.hypotheses if item.hypothesis_id in hypothesis_ids
        )
        timeline = tuple(item for item in self._manifest.timeline if item.event_id in event_ids)
        gaps = tuple(item for item in self._manifest.gaps if item.gap_id in gap_ids)
        relations = tuple(
            item
            for item in self._manifest.relations
            if item.source_hypothesis_id in hypothesis_ids
            and item.target_hypothesis_id in hypothesis_ids
        )
        matches = _route_matches(
            _selected_scores(question_scores, question_ids),
            _selected_scores(hypothesis_scores, hypothesis_ids),
            _selected_scores(timeline_scores, event_ids),
            _selected_scores(gap_scores, gap_ids),
        )
        return self._build_packet(
            normalized_query,
            query_terms,
            matches,
            questions,
            hypotheses,
            timeline,
            gaps,
            relations,
        )

    def route_v2(
        self,
        query: str,
        receipt: RouteCandidateReceipt,
        budget: RouteBudget,
    ) -> CompactMemoryPacketV1_1:
        """Route one fresh candidate receipt through an explicit bounded closure."""

        normalized_query = unicodedata.normalize("NFC", query).strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if len(normalized_query) > 4_000:
            raise ValueError("query must be at most 4000 characters")
        validated_receipt = RouteCandidateReceipt.model_validate(receipt.model_dump(mode="python"))
        validated_budget = RouteBudget.model_validate(budget.model_dump(mode="python"))
        validated_receipt.validate_for(
            self._manifest,
            manifest_hash=self._manifest_hash,
            query=normalized_query,
        )
        query_terms = _terms(normalized_query)
        if (
            validated_receipt.decision is RouteCandidateDecision.ABSTAINED
            or not validated_receipt.ordered_candidates
        ):
            return self._build_packet_v2(
                normalized_query,
                query_terms,
                validated_receipt,
                validated_budget,
                RouteStatus.NO_MATCH,
                _BudgetedRouteSelection(self._manifest, validated_budget),
            )

        selection = _BudgetedRouteSelection(self._manifest, validated_budget)
        for candidate in validated_receipt.ordered_candidates:
            selection.admit_seed(candidate)
        if not selection.seed_question_ids:
            return self._build_packet_v2(
                normalized_query,
                query_terms,
                validated_receipt,
                validated_budget,
                RouteStatus.BUDGET_EXCEEDED,
                selection,
            )

        selection.expand_graph()
        return self._build_packet_v2(
            normalized_query,
            query_terms,
            validated_receipt,
            validated_budget,
            RouteStatus.ROUTED,
            selection,
        )

    def _close_object_references(
        self,
        question_ids: set[str],
        hypothesis_ids: set[str],
        event_ids: set[str],
        gap_ids: set[str],
    ) -> tuple[set[str], set[str]]:
        """Include records directly named by a selected graph object."""

        closed_questions = set(question_ids)
        closed_hypotheses = set(hypothesis_ids)
        changed = True
        while changed:
            before = (len(closed_questions), len(closed_hypotheses))
            for hypothesis in self._manifest.hypotheses:
                if hypothesis.hypothesis_id in closed_hypotheses:
                    closed_questions.update(hypothesis.question_ids)
            for event in self._manifest.timeline:
                if event.event_id in event_ids:
                    closed_questions.update(event.question_ids)
                    closed_hypotheses.update(event.hypothesis_ids)
            for gap in self._manifest.gaps:
                if gap.gap_id in gap_ids:
                    closed_questions.update(gap.related_question_ids)
                    closed_hypotheses.update(gap.related_hypothesis_ids)
            changed = before != (len(closed_questions), len(closed_hypotheses))
        return closed_questions, closed_hypotheses

    def _rank_questions(self, terms: tuple[str, ...]) -> list[tuple[int, str, tuple[str, ...]]]:
        return _rank(
            (
                (
                    item.question_id,
                    (
                        (item.prompt, 6),
                        (" ".join(item.tags), 5),
                        (item.short_answer, 2),
                        (item.gap or "", 1),
                    ),
                )
                for item in self._manifest.questions
            ),
            terms,
        )

    def _rank_hypotheses(self, terms: tuple[str, ...]) -> list[tuple[int, str, tuple[str, ...]]]:
        return _rank(
            (
                (
                    item.hypothesis_id,
                    ((item.title, 5), (item.synthesis, 2), (item.gap or "", 1)),
                )
                for item in self._manifest.hypotheses
            ),
            terms,
        )

    def _rank_timeline(self, terms: tuple[str, ...]) -> list[tuple[int, str, tuple[str, ...]]]:
        return _rank(
            (
                (
                    item.event_id,
                    ((item.title, 5), (item.summary, 2), (item.date_label, 4)),
                )
                for item in self._manifest.timeline
            ),
            terms,
        )

    def _rank_gaps(self, terms: tuple[str, ...]) -> list[tuple[int, str, tuple[str, ...]]]:
        return _rank(
            (
                (
                    item.gap_id,
                    ((item.label, 5), (item.why_it_matters, 2), (item.next_evidence, 2)),
                )
                for item in self._manifest.gaps
            ),
            terms,
        )

    def _build_packet(
        self,
        query: str,
        query_terms: tuple[str, ...],
        route_matches: tuple[RouteMatch, ...],
        questions: tuple[AtlasQuestion, ...],
        hypotheses: tuple[AtlasHypothesis, ...],
        timeline: tuple[AtlasTimelineEvent, ...] = (),
        gaps: tuple[AtlasGap, ...] = (),
        relations: tuple[AtlasRelation, ...] = (),
    ) -> CompactMemoryPacket:
        evidence_ids: set[str] = set()
        for question in questions:
            evidence_ids.update(question.evidence_ids)
        for hypothesis in hypotheses:
            evidence_ids.update(hypothesis.evidence_ids)
            evidence_ids.update(hypothesis.counterevidence_ids)
        for event in timeline:
            evidence_ids.update(event.evidence_ids)
        for relation in relations:
            evidence_ids.update(relation.evidence_ids)
        evidence = tuple(
            item for item in self._manifest.evidence if item.evidence_id in evidence_ids
        )
        source_ids = {item.source_id for item in evidence}
        sources = tuple(item for item in self._manifest.sources if item.source_id in source_ids)
        status = RouteStatus.ROUTED if route_matches else RouteStatus.NO_MATCH
        payload: dict[str, object] = {
            "schema_id": CompactMemoryPacket.SCHEMA,
            "router_profile": self.PROFILE,
            "route_status": status.value,
            "query": query,
            "query_terms": list(query_terms),
            "atlas_id": self._manifest.atlas_id,
            "case_id": self._manifest.case_id,
            "manifest_hash": self._manifest_hash,
            "route_matches": [item.model_dump(mode="json") for item in route_matches],
            "sources": [item.model_dump(mode="json") for item in sources],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "questions": [item.model_dump(mode="json") for item in questions],
            "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
            "relations": [item.model_dump(mode="json") for item in relations],
            "timeline": [item.model_dump(mode="json") for item in timeline],
            "gaps": [item.model_dump(mode="json") for item in gaps],
        }
        return CompactMemoryPacket(
            schema_id=CompactMemoryPacket.SCHEMA,
            packet_id=canonical_content_id("memory_packet", payload),
            packet_hash=canonical_sha256_hex(payload),
            router_profile=self.PROFILE,
            route_status=status,
            query=query,
            query_terms=query_terms,
            atlas_id=self._manifest.atlas_id,
            case_id=self._manifest.case_id,
            manifest_hash=self._manifest_hash,
            route_matches=route_matches,
            sources=sources,
            evidence=evidence,
            questions=questions,
            hypotheses=hypotheses,
            relations=relations,
            timeline=timeline,
            gaps=gaps,
        )

    def _build_packet_v2(
        self,
        query: str,
        query_terms: tuple[str, ...],
        receipt: RouteCandidateReceipt,
        budget: RouteBudget,
        status: RouteStatus,
        selection: _BudgetedRouteSelection,
    ) -> CompactMemoryPacketV1_1:
        questions = tuple(selection.question_by_id[item_id] for item_id in selection.question_order)
        hypotheses = tuple(
            selection.hypothesis_by_id[item_id] for item_id in selection.hypothesis_order
        )
        relations = tuple(selection.relation_by_id[item_id] for item_id in selection.relation_order)
        timeline = tuple(selection.event_by_id[item_id] for item_id in selection.event_order)
        gaps = tuple(selection.gap_by_id[item_id] for item_id in selection.gap_order)
        evidence = tuple(
            item for item in self._manifest.evidence if item.evidence_id in selection.evidence_ids
        )
        sources = tuple(
            item for item in self._manifest.sources if item.source_id in selection.source_ids
        )
        payload: dict[str, object] = {
            "schema_id": CompactMemoryPacketV1_1.SCHEMA,
            "router_profile": self.PROFILE_V2,
            "route_status": status.value,
            "query": query,
            "query_terms": list(query_terms),
            "atlas_id": self._manifest.atlas_id,
            "case_id": self._manifest.case_id,
            "manifest_hash": self._manifest_hash,
            "route_matches": [],
            "sources": [item.model_dump(mode="json") for item in sources],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "questions": [item.model_dump(mode="json") for item in questions],
            "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
            "relations": [item.model_dump(mode="json") for item in relations],
            "timeline": [item.model_dump(mode="json") for item in timeline],
            "gaps": [item.model_dump(mode="json") for item in gaps],
            "route_receipt_id": receipt.receipt_id,
            "route_receipt_hash": receipt.receipt_hash,
            "route_budget": budget.model_dump(mode="json"),
            "route_trace": [item.model_dump(mode="json") for item in selection.trace],
            "omissions": [item.model_dump(mode="json") for item in selection.omissions],
        }
        return CompactMemoryPacketV1_1(
            schema_id=CompactMemoryPacketV1_1.SCHEMA,
            packet_id=canonical_content_id("memory_packet", payload),
            packet_hash=canonical_sha256_hex(payload),
            router_profile=self.PROFILE_V2,
            route_status=status,
            query=query,
            query_terms=query_terms,
            atlas_id=self._manifest.atlas_id,
            case_id=self._manifest.case_id,
            manifest_hash=self._manifest_hash,
            route_matches=(),
            sources=sources,
            evidence=evidence,
            questions=questions,
            hypotheses=hypotheses,
            relations=relations,
            timeline=timeline,
            gaps=gaps,
            route_receipt_id=receipt.receipt_id,
            route_receipt_hash=receipt.receipt_hash,
            route_budget=budget,
            route_trace=tuple(selection.trace),
            omissions=tuple(selection.omissions),
        )


class _BudgetedRouteSelection:
    """Mutable implementation detail for one deterministic v2 routing pass."""

    def __init__(self, manifest: ResearchAtlasManifest, budget: RouteBudget) -> None:
        self.manifest = manifest
        self.budget = budget
        self.question_by_id = {item.question_id: item for item in manifest.questions}
        self.hypothesis_by_id = {item.hypothesis_id: item for item in manifest.hypotheses}
        self.relation_by_id = {item.relation_id: item for item in manifest.relations}
        self.event_by_id = {item.event_id: item for item in manifest.timeline}
        self.gap_by_id = {item.gap_id: item for item in manifest.gaps}
        self.evidence_by_id = {item.evidence_id: item for item in manifest.evidence}

        self.seed_question_ids: set[str] = set()
        self.question_ids: set[str] = set()
        self.hypothesis_ids: set[str] = set()
        self.relation_ids: set[str] = set()
        self.event_ids: set[str] = set()
        self.gap_ids: set[str] = set()
        self.evidence_ids: set[str] = set()
        self.source_ids: set[str] = set()

        self.seed_question_order: list[str] = []
        self.question_order: list[str] = []
        self.hypothesis_order: list[str] = []
        self.relation_order: list[str] = []
        self.event_order: list[str] = []
        self.gap_order: list[str] = []
        self.trace: list[RouteTraceEntry] = []
        self.omissions: list[RouteOmission] = []

    def admit_seed(
        self,
        candidate: SemanticQuestionCandidate | LexicalQuestionCandidate,
    ) -> None:
        question_id = candidate.question_id
        evidence_ids, source_ids = self._resource_closure((question_id,), ())
        reason = self._capacity_reason(
            question_ids={question_id},
            evidence_ids=evidence_ids,
            source_ids=source_ids,
        )
        if reason is not None:
            self._omit(RouteObjectKind.QUESTION, question_id, reason)
            return

        self.question_ids.add(question_id)
        self.question_order.append(question_id)
        self.seed_question_ids.add(question_id)
        self.seed_question_order.append(question_id)
        self.evidence_ids.update(evidence_ids)
        self.source_ids.update(source_ids)
        if isinstance(candidate, SemanticQuestionCandidate):
            self.trace.append(
                RouteTraceEntry(
                    stage=RouteTraceStage.SEED,
                    object_kind=RouteObjectKind.QUESTION,
                    object_id=question_id,
                    channel=RouteCandidateChannel.SEMANTIC,
                    channel_rank=candidate.rank,
                    semantic_score_hex=candidate.score_hex,
                )
            )
        else:
            self.trace.append(
                RouteTraceEntry(
                    stage=RouteTraceStage.SEED,
                    object_kind=RouteObjectKind.QUESTION,
                    object_id=question_id,
                    channel=RouteCandidateChannel.LEXICAL,
                    channel_rank=candidate.rank,
                    lexical_score=candidate.score,
                )
            )

    def expand_graph(self) -> None:
        """Follow one declared expansion layer from the admitted seed closure."""

        for hypothesis in self.manifest.hypotheses:
            if self.seed_question_ids.intersection(hypothesis.question_ids):
                self._admit_hypothesis(hypothesis.hypothesis_id)

        linked_hypothesis_ids = set(self.hypothesis_ids)
        for event in self.manifest.timeline:
            if self.seed_question_ids.intersection(event.question_ids) or (
                linked_hypothesis_ids.intersection(event.hypothesis_ids)
            ):
                self._admit_event(event.event_id, linked_hypothesis_ids)

        for gap in self.manifest.gaps:
            if self.seed_question_ids.intersection(gap.related_question_ids) or (
                linked_hypothesis_ids.intersection(gap.related_hypothesis_ids)
            ):
                self._admit_gap(gap.gap_id, linked_hypothesis_ids)

        for relation in self.manifest.relations:
            endpoint_ids = {
                relation.source_hypothesis_id,
                relation.target_hypothesis_id,
            }
            if not endpoint_ids.intersection(self.hypothesis_ids):
                continue
            if not endpoint_ids.issubset(self.hypothesis_ids):
                self._omit(
                    RouteObjectKind.RELATION,
                    relation.relation_id,
                    RouteOmissionReason.REFERENCE_BUDGET,
                )
                continue
            self._admit_relation(relation.relation_id)

    def _admit_hypothesis(self, hypothesis_id: str) -> None:
        if hypothesis_id in self.hypothesis_ids:
            return
        hypothesis = self.hypothesis_by_id[hypothesis_id]
        required_questions = tuple(hypothesis.question_ids)
        evidence_ids, source_ids = self._resource_closure(
            required_questions,
            (hypothesis_id,),
        )
        reason = self._capacity_reason(
            question_ids=set(required_questions),
            hypothesis_ids={hypothesis_id},
            evidence_ids=evidence_ids,
            source_ids=source_ids,
        )
        if reason is not None:
            self._omit(RouteObjectKind.HYPOTHESIS, hypothesis_id, reason)
            return
        parent_id = next(
            item_id for item_id in self.seed_question_order if item_id in hypothesis.question_ids
        )
        parent = (RouteObjectKind.QUESTION, parent_id)
        self._commit_questions(required_questions, parent=parent)
        self.hypothesis_ids.add(hypothesis_id)
        self.hypothesis_order.append(hypothesis_id)
        self._clear_omission(RouteObjectKind.HYPOTHESIS, hypothesis_id)
        self.trace.append(self._expansion_trace(RouteObjectKind.HYPOTHESIS, hypothesis_id, parent))
        self.evidence_ids.update(evidence_ids)
        self.source_ids.update(source_ids)

    def _admit_event(self, event_id: str, linked_hypothesis_ids: set[str]) -> None:
        event = self.event_by_id[event_id]
        required_hypotheses = tuple(event.hypothesis_ids)
        required_questions = _stable_unique(
            (
                *event.question_ids,
                *(
                    question_id
                    for hypothesis_id in required_hypotheses
                    for question_id in self.hypothesis_by_id[hypothesis_id].question_ids
                ),
            )
        )
        evidence_ids, source_ids = self._resource_closure(
            required_questions,
            required_hypotheses,
            extra_evidence_ids=event.evidence_ids,
        )
        reason = self._capacity_reason(
            question_ids=set(required_questions),
            hypothesis_ids=set(required_hypotheses),
            event_ids={event_id},
            evidence_ids=evidence_ids,
            source_ids=source_ids,
        )
        if reason is not None:
            self._omit(RouteObjectKind.TIMELINE, event_id, reason)
            return
        parent = self._linked_parent(
            event.question_ids,
            event.hypothesis_ids,
            linked_hypothesis_ids,
        )
        self._commit_questions(required_questions, parent=parent)
        self._commit_hypotheses(required_hypotheses, parent=parent)
        self.event_ids.add(event_id)
        self.event_order.append(event_id)
        self._clear_omission(RouteObjectKind.TIMELINE, event_id)
        self.trace.append(self._expansion_trace(RouteObjectKind.TIMELINE, event_id, parent))
        self.evidence_ids.update(evidence_ids)
        self.source_ids.update(source_ids)

    def _admit_gap(self, gap_id: str, linked_hypothesis_ids: set[str]) -> None:
        gap = self.gap_by_id[gap_id]
        required_hypotheses = tuple(gap.related_hypothesis_ids)
        required_questions = _stable_unique(
            (
                *gap.related_question_ids,
                *(
                    question_id
                    for hypothesis_id in required_hypotheses
                    for question_id in self.hypothesis_by_id[hypothesis_id].question_ids
                ),
            )
        )
        evidence_ids, source_ids = self._resource_closure(
            required_questions,
            required_hypotheses,
        )
        reason = self._capacity_reason(
            question_ids=set(required_questions),
            hypothesis_ids=set(required_hypotheses),
            gap_ids={gap_id},
            evidence_ids=evidence_ids,
            source_ids=source_ids,
        )
        if reason is not None:
            self._omit(RouteObjectKind.GAP, gap_id, reason)
            return
        parent = self._linked_parent(
            gap.related_question_ids,
            gap.related_hypothesis_ids,
            linked_hypothesis_ids,
        )
        self._commit_questions(required_questions, parent=parent)
        self._commit_hypotheses(required_hypotheses, parent=parent)
        self.gap_ids.add(gap_id)
        self.gap_order.append(gap_id)
        self._clear_omission(RouteObjectKind.GAP, gap_id)
        self.trace.append(self._expansion_trace(RouteObjectKind.GAP, gap_id, parent))
        self.evidence_ids.update(evidence_ids)
        self.source_ids.update(source_ids)

    def _admit_relation(self, relation_id: str) -> None:
        relation = self.relation_by_id[relation_id]
        evidence_ids, source_ids = self._resource_closure(
            (),
            (),
            extra_evidence_ids=relation.evidence_ids,
        )
        reason = self._capacity_reason(
            relation_ids={relation_id},
            evidence_ids=evidence_ids,
            source_ids=source_ids,
        )
        if reason is not None:
            self._omit(RouteObjectKind.RELATION, relation_id, reason)
            return
        parent = (RouteObjectKind.HYPOTHESIS, relation.source_hypothesis_id)
        self.relation_ids.add(relation_id)
        self.relation_order.append(relation_id)
        self.trace.append(self._expansion_trace(RouteObjectKind.RELATION, relation_id, parent))
        self.evidence_ids.update(evidence_ids)
        self.source_ids.update(source_ids)

    def _commit_questions(
        self,
        question_ids: tuple[str, ...],
        *,
        parent: tuple[RouteObjectKind, str],
    ) -> None:
        for question_id in question_ids:
            if question_id in self.question_ids:
                continue
            self.question_ids.add(question_id)
            self.question_order.append(question_id)
            self._clear_omission(RouteObjectKind.QUESTION, question_id)
            self.trace.append(self._expansion_trace(RouteObjectKind.QUESTION, question_id, parent))

    def _commit_hypotheses(
        self,
        hypothesis_ids: tuple[str, ...],
        *,
        parent: tuple[RouteObjectKind, str],
    ) -> None:
        for hypothesis_id in hypothesis_ids:
            if hypothesis_id in self.hypothesis_ids:
                continue
            self.hypothesis_ids.add(hypothesis_id)
            self.hypothesis_order.append(hypothesis_id)
            self._clear_omission(RouteObjectKind.HYPOTHESIS, hypothesis_id)
            self.trace.append(
                self._expansion_trace(RouteObjectKind.HYPOTHESIS, hypothesis_id, parent)
            )

    def _linked_parent(
        self,
        question_ids: tuple[str, ...],
        hypothesis_ids: tuple[str, ...],
        linked_hypothesis_ids: set[str],
    ) -> tuple[RouteObjectKind, str]:
        for question_id in self.seed_question_order:
            if question_id in question_ids:
                return RouteObjectKind.QUESTION, question_id
        for hypothesis_id in self.hypothesis_order:
            if hypothesis_id in hypothesis_ids and hypothesis_id in linked_hypothesis_ids:
                return RouteObjectKind.HYPOTHESIS, hypothesis_id
        raise ValueError("graph expansion lacks an admitted seed-linked parent")

    def _resource_closure(
        self,
        question_ids: tuple[str, ...],
        hypothesis_ids: tuple[str, ...],
        *,
        extra_evidence_ids: tuple[str, ...] = (),
    ) -> tuple[set[str], set[str]]:
        evidence_ids = set(extra_evidence_ids)
        for question_id in question_ids:
            evidence_ids.update(self.question_by_id[question_id].evidence_ids)
        for hypothesis_id in hypothesis_ids:
            hypothesis = self.hypothesis_by_id[hypothesis_id]
            evidence_ids.update(hypothesis.evidence_ids)
            evidence_ids.update(hypothesis.counterevidence_ids)
        source_ids = {self.evidence_by_id[evidence_id].source_id for evidence_id in evidence_ids}
        return evidence_ids, source_ids

    def _capacity_reason(
        self,
        *,
        question_ids: set[str] | None = None,
        hypothesis_ids: set[str] | None = None,
        relation_ids: set[str] | None = None,
        event_ids: set[str] | None = None,
        gap_ids: set[str] | None = None,
        evidence_ids: set[str] | None = None,
        source_ids: set[str] | None = None,
    ) -> RouteOmissionReason | None:
        if len(self.question_ids.union(question_ids or set())) > self.budget.max_questions:
            return RouteOmissionReason.QUESTION_BUDGET
        if (
            len(self.hypothesis_ids.union(hypothesis_ids or set()))
            > self.budget.max_linked_hypotheses
        ):
            return RouteOmissionReason.HYPOTHESIS_BUDGET
        if len(self.relation_ids.union(relation_ids or set())) > self.budget.max_relations:
            return RouteOmissionReason.RELATION_BUDGET
        if len(self.event_ids.union(event_ids or set())) > self.budget.max_timeline_events:
            return RouteOmissionReason.TIMELINE_BUDGET
        if len(self.gap_ids.union(gap_ids or set())) > self.budget.max_gaps:
            return RouteOmissionReason.GAP_BUDGET
        if len(self.evidence_ids.union(evidence_ids or set())) > self.budget.max_evidence:
            return RouteOmissionReason.EVIDENCE_BUDGET
        if len(self.source_ids.union(source_ids or set())) > self.budget.max_sources:
            return RouteOmissionReason.SOURCE_BUDGET
        return None

    def _omit(
        self,
        kind: RouteObjectKind,
        object_id: str,
        reason: RouteOmissionReason,
    ) -> None:
        key = (kind, object_id)
        if any((item.object_kind, item.object_id) == key for item in self.omissions):
            return
        self.omissions.append(RouteOmission(object_kind=kind, object_id=object_id, reason=reason))

    def _clear_omission(self, kind: RouteObjectKind, object_id: str) -> None:
        self.omissions[:] = [
            item
            for item in self.omissions
            if (item.object_kind, item.object_id) != (kind, object_id)
        ]

    @staticmethod
    def _expansion_trace(
        kind: RouteObjectKind,
        object_id: str,
        parent: tuple[RouteObjectKind, str],
    ) -> RouteTraceEntry:
        return RouteTraceEntry(
            stage=RouteTraceStage.GRAPH_EXPANSION,
            object_kind=kind,
            object_id=object_id,
            parent_object_kind=parent[0],
            parent_object_id=parent[1],
        )


def _terms(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", value).casefold()
    terms = {
        token
        for token in _WORD_PATTERN.findall(normalized)
        if token not in _STOP_WORDS and (len(token) >= 3 or token.isdigit())
    }
    return tuple(sorted(terms))


def _stable_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _require_float_hex(value: str) -> None:
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError("semantic score must use canonical float.hex syntax") from exc
    if not math.isfinite(parsed):
        raise ValueError("semantic score must be finite")
    if parsed.hex() != value:
        raise ValueError("semantic score must use exact canonical float.hex representation")


def _rank(
    records: Iterable[tuple[str, tuple[tuple[str, int], ...]]],
    query_terms: tuple[str, ...],
) -> list[tuple[int, str, tuple[str, ...]]]:
    ranked: list[tuple[int, str, tuple[str, ...]]] = []
    query_set = set(query_terms)
    for object_id, fields in records:
        score = 0
        matched: set[str] = set()
        for value, weight in fields:
            overlap = query_set.intersection(_terms(value))
            score += len(overlap) * weight
            matched.update(overlap)
        if score:
            ranked.append((score, object_id, tuple(sorted(matched))))
    return sorted(ranked, key=lambda item: (-item[0], item[1]))


def _route_matches(
    questions: list[tuple[int, str, tuple[str, ...]]],
    hypotheses: list[tuple[int, str, tuple[str, ...]]],
    timeline: list[tuple[int, str, tuple[str, ...]]],
    gaps: list[tuple[int, str, tuple[str, ...]]],
) -> tuple[RouteMatch, ...]:
    rows: list[RouteMatch] = []
    for kind, ranked in (
        (RouteObjectKind.QUESTION, questions),
        (RouteObjectKind.HYPOTHESIS, hypotheses),
        (RouteObjectKind.TIMELINE, timeline),
        (RouteObjectKind.GAP, gaps),
    ):
        rows.extend(
            RouteMatch(object_kind=kind, object_id=object_id, score=score, matched_terms=terms)
            for score, object_id, terms in ranked
        )
    return tuple(
        sorted(rows, key=lambda item: (-item.score, item.object_kind.value, item.object_id))
    )


def _selected_scores(
    ranked: list[tuple[int, str, tuple[str, ...]]], selected_ids: set[str]
) -> list[tuple[int, str, tuple[str, ...]]]:
    return [item for item in ranked if item[1] in selected_ids]


def _capped_ids(primary: Iterable[str], linked: Iterable[str], limit: int) -> set[str]:
    if limit == 0:
        return set()
    selected = list(dict.fromkeys(primary))[:limit]
    for object_id in linked:
        if object_id not in selected and len(selected) < limit:
            selected.append(object_id)
    return set(selected)


def _ids(values: Iterable[str], label: str) -> frozenset[str]:
    items = tuple(values)
    if len(items) != len(set(items)):
        raise ValueError(f"compact packet {label} identifiers must be unique")
    return frozenset(items)


def _require_refs(values: Iterable[str], valid: frozenset[str], label: str) -> None:
    missing = sorted(set(values).difference(valid))
    if missing:
        raise ValueError(f"compact packet {label} references missing IDs: {', '.join(missing)}")
