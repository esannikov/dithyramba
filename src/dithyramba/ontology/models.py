"""Strict contracts for a scoped, rebuildable candidate ontology.

The ontology is an orientation view, never an accepted truth layer.  Every
concept and every visible co-occurrence keeps exact fragment support.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.contracts import canonical_json_bytes, sha256_hex

_ID = r"^[a-z][a-z0-9_-]{1,95}$"
_HASH = r"^[0-9a-f]{64}$"
_SCORE = r"^(?:0|[1-9][0-9]*)\.[0-9]{6}$"


class _OntologyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class OntologyRelationKind(StrEnum):
    """The only relation inferred without a generative semantic claim."""

    CO_OCCURS_WITH = "co_occurs_with"


class OntologyEvidence(_OntologyModel):
    evidence_id: str = Field(pattern=_ID)
    source_fragment_id: str = Field(pattern=r"^fragment_[a-z0-9_]+$")
    source_id: str = Field(pattern=r"^source_[a-z0-9_]+$")
    source_version_id: str = Field(pattern=r"^source_version_[a-z0-9_]+$")
    source_title: str = Field(min_length=1, max_length=1_000)
    source_creator: str = Field(min_length=1, max_length=500)
    source_kind: str = Field(min_length=1, max_length=160)
    locator: str = Field(min_length=1, max_length=500)
    text_sha256: str = Field(pattern=_HASH)
    address_hash: str = Field(pattern=_HASH)
    source_address: dict[str, object]
    excerpt: str = Field(min_length=1, max_length=2_000)

    @field_validator("source_address")
    @classmethod
    def address_is_explicit(cls, value: dict[str, object]) -> dict[str, object]:
        if not value or not isinstance(value.get("kind"), str):
            raise ValueError("ontology evidence requires an explicit source address")
        return value


class OntologyConcept(_OntologyModel):
    concept_id: str = Field(pattern=_ID)
    cluster_id: str = Field(pattern=_ID)
    label: str = Field(min_length=2, max_length=160)
    score: str = Field(pattern=_SCORE)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    source_count: int = Field(ge=1, le=16)
    origin: Literal["scope_anchor", "emergent"]
    state: Literal["candidate"] = "candidate"


class OntologyCluster(_OntologyModel):
    cluster_id: str = Field(pattern=_ID)
    label: str = Field(min_length=2, max_length=320)
    concept_ids: tuple[str, ...] = Field(min_length=2, max_length=24)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    source_count: int = Field(ge=1)
    state: Literal["candidate"] = "candidate"


class OntologyRelation(_OntologyModel):
    relation_id: str = Field(pattern=_ID)
    kind: OntologyRelationKind
    source_concept_id: str = Field(pattern=_ID)
    target_concept_id: str = Field(pattern=_ID)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    shared_fragment_count: int = Field(ge=1)
    source_count: int = Field(ge=1, le=16)
    state: Literal["candidate"] = "candidate"

    @model_validator(mode="after")
    def distinct_ordered_endpoints(self) -> Self:
        if self.source_concept_id >= self.target_concept_id:
            raise ValueError("ontology relation endpoints must be distinct and canonical")
        return self


class OntologyMetrics(_OntologyModel):
    admitted_fragment_count: int = Field(ge=1)
    selected_fragment_count: int = Field(ge=1)
    source_count: int = Field(ge=1)
    concept_count: int = Field(ge=2)
    scope_anchor_count: int = Field(ge=0)
    emergent_concept_count: int = Field(ge=0)
    cluster_count: int = Field(ge=1)
    relation_count: int = Field(ge=0)
    multi_source_relation_ratio: str = Field(pattern=_SCORE)


class CandidateOntologyManifest(_OntologyModel):
    """One immutable, local, candidate-only orientation projection."""

    SCHEMA: ClassVar[str] = "dithyramba.candidate_ontology/1.0"

    schema_id: str
    ontology_id: str = Field(pattern=_ID)
    scope_label: str = Field(min_length=2, max_length=300)
    original_question: str = Field(min_length=2, max_length=2_000)
    language: Literal["en"]
    generated_at: datetime
    mode: Literal["candidate_only"] = "candidate_only"
    model_role: Literal["none", "bounded_relevance_filter"] = "none"
    model_receipt_hash: str | None = Field(default=None, pattern=_HASH)
    query_cloud: tuple[str, ...] = Field(min_length=1, max_length=32)
    input_manifest_hash: str = Field(pattern=_HASH)
    extractor_profile: str = Field(pattern=_HASH)
    clusters: tuple[OntologyCluster, ...] = Field(min_length=1, max_length=16)
    concepts: tuple[OntologyConcept, ...] = Field(min_length=2, max_length=256)
    relations: tuple[OntologyRelation, ...] = Field(default=(), max_length=256)
    evidence: tuple[OntologyEvidence, ...] = Field(min_length=1, max_length=4_096)
    metrics: OntologyMetrics

    @model_validator(mode="after")
    def closed_candidate_projection(self) -> Self:
        if self.schema_id != self.SCHEMA:
            raise ValueError(f"schema_id must be {self.SCHEMA}")
        if (self.model_role == "none") != (self.model_receipt_hash is None):
            raise ValueError("model receipt must exist exactly when a relevance model was used")
        cluster_ids = _unique((item.cluster_id for item in self.clusters), "cluster")
        concepts = {item.concept_id: item for item in self.concepts}
        if len(concepts) != len(self.concepts):
            raise ValueError("concept identifiers must be unique")
        evidence = {item.evidence_id: item for item in self.evidence}
        if len(evidence) != len(self.evidence):
            raise ValueError("evidence identifiers must be unique")
        _unique((item.relation_id for item in self.relations), "relation")

        cluster_members: dict[str, set[str]] = {cluster_id: set() for cluster_id in cluster_ids}
        for concept in self.concepts:
            if concept.cluster_id not in cluster_ids:
                raise ValueError("concept references a missing ontology cluster")
            _require_subset(concept.evidence_ids, evidence, "concept evidence")
            actual_sources = {evidence[item].source_id for item in concept.evidence_ids}
            if concept.source_count != len(actual_sources):
                raise ValueError("concept source_count does not match exact evidence")
            cluster_members[concept.cluster_id].add(concept.concept_id)

        for cluster in self.clusters:
            if set(cluster.concept_ids) != cluster_members[cluster.cluster_id]:
                raise ValueError("cluster concept membership is not closed")
            _require_subset(cluster.evidence_ids, evidence, "cluster evidence")
            actual_sources = {evidence[item].source_id for item in cluster.evidence_ids}
            if cluster.source_count != len(actual_sources):
                raise ValueError("cluster source_count does not match exact evidence")

        for relation in self.relations:
            if (
                relation.source_concept_id not in concepts
                or relation.target_concept_id not in concepts
            ):
                raise ValueError("relation references a missing concept")
            _require_subset(relation.evidence_ids, evidence, "relation evidence")
            actual_sources = {evidence[item].source_id for item in relation.evidence_ids}
            if relation.source_count != len(actual_sources):
                raise ValueError("relation source_count does not match exact evidence")

        expected = (
            len(self.concepts),
            len(self.clusters),
            len(self.relations),
            len({item.source_id for item in self.evidence}),
        )
        actual = (
            self.metrics.concept_count,
            self.metrics.cluster_count,
            self.metrics.relation_count,
            self.metrics.source_count,
        )
        if actual != expected:
            raise ValueError("ontology metrics do not close over the projection")
        if (
            self.metrics.scope_anchor_count + self.metrics.emergent_concept_count
            != self.metrics.concept_count
        ):
            raise ValueError("ontology concept-origin metrics do not close")
        if self.metrics.selected_fragment_count > self.metrics.admitted_fragment_count:
            raise ValueError("selected fragments cannot exceed admitted fragments")
        return self


@dataclass(frozen=True, slots=True)
class LoadedCandidateOntology:
    manifest: CandidateOntologyManifest
    manifest_path: Path
    manifest_hash: str


def load_candidate_ontology(path: str | Path) -> LoadedCandidateOntology:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError("candidate ontology manifest must be a regular file")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = CandidateOntologyManifest.model_validate(payload)
    canonical = canonical_json_bytes(manifest.model_dump(mode="json"))
    return LoadedCandidateOntology(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_hash=sha256_hex(canonical),
    )


def _unique(values: Iterable[str], label: str) -> frozenset[str]:
    received = tuple(values)
    if len(received) != len(set(received)):
        raise ValueError(f"{label} identifiers must be unique")
    return frozenset(received)


def _require_subset(values: tuple[str, ...], valid: Mapping[str, object], label: str) -> None:
    missing = sorted(set(values).difference(valid))
    if missing:
        raise ValueError(f"{label} references missing IDs: {', '.join(missing)}")
