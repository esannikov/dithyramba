"""Automatic, local construction of one scoped candidate ontology."""

from __future__ import annotations

import importlib
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal, Protocol, cast

from dithyramba.contracts import canonical_content_id, canonical_sha256_hex, sha256_hex

from .errors import OntologyError
from .models import (
    CandidateOntologyManifest,
    OntologyCluster,
    OntologyConcept,
    OntologyEvidence,
    OntologyMetrics,
    OntologyRelation,
    OntologyRelationKind,
)

_WORD = re.compile(r"[A-Za-z][A-Za-z\u2019'-]{2,}")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_MARKUP_STOP_WORDS = frozenset(
    {
        "align",
        "author",
        "book",
        "chapter",
        "class",
        "copyright",
        "div",
        "don't",
        "don\u2019t",
        "edition",
        "figure",
        "font",
        "href",
        "html",
        "introduction",
        "it's",
        "it\u2019s",
        "italic",
        "page",
        "span",
        "style",
        "table",
        "title",
        "volume",
        "width",
    }
)
_LOW_INFORMATION_STOP_WORDS = frozenset(
    {
        "film",
        "good",
        "great",
        "make",
        "making",
        "people",
        "standard",
        "thing",
        "things",
        "time",
        "use",
        "used",
        "using",
        "want",
        "way",
        "work",
        "working",
    }
)


class _FeatureNames(Protocol):
    def __getitem__(self, index: int) -> str: ...


@dataclass(frozen=True, slots=True)
class OntologyFragment:
    """One already-authorized exact-text input, ranked before extraction."""

    source_fragment_id: str
    source_id: str
    source_version_id: str
    text: str = field(repr=False)
    text_sha256: str
    address_hash: str
    source_address: Mapping[str, object]
    source_title: str
    source_creator: str
    source_kind: str
    locator: str
    rank: int

    def __post_init__(self) -> None:
        _prefixed(self.source_fragment_id, "fragment_", "source_fragment_id")
        _prefixed(self.source_id, "source_", "source_id")
        _prefixed(self.source_version_id, "source_version_", "source_version_id")
        if type(self.rank) is not int or self.rank < 1 or self.rank > 500:
            raise OntologyError("ontology fragment rank must be between 1 and 500")
        if (
            type(self.text) is not str
            or not self.text.strip()
            or len(self.text) > 250_000
            or "\x00" in self.text
            or unicodedata.normalize("NFC", self.text) != self.text
        ):
            raise OntologyError("ontology fragment text must be bounded NFC text")
        _hash(self.text_sha256, "text_sha256")
        _hash(self.address_hash, "address_hash")
        if sha256_hex(self.text.encode("utf-8")) != self.text_sha256:
            raise OntologyError("ontology fragment text does not match text_sha256")
        if not self.source_address or not isinstance(self.source_address.get("kind"), str):
            raise OntologyError("ontology fragment requires an explicit source address")
        for label, value in (
            ("source_title", self.source_title),
            ("source_creator", self.source_creator),
            ("source_kind", self.source_kind),
            ("locator", self.locator),
        ):
            if type(value) is not str or not value.strip():
                raise OntologyError(f"ontology fragment {label} must be nonblank")

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "text_sha256": self.text_sha256,
            "address_hash": self.address_hash,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class OntologyConfig:
    SCHEMA: ClassVar[str] = "dithyramba.ontology_config/1.0"

    cluster_count: int = 7
    concepts_per_cluster: int = 6
    minimum_document_frequency: int = 3
    minimum_concept_sources: int = 2
    maximum_document_ratio: float = 0.55
    maximum_features: int = 6_000
    evidence_per_concept: int = 5
    maximum_relations: int = 30
    minimum_relation_sources: int = 2
    maximum_fragments: int = 500

    def __post_init__(self) -> None:
        _integer(self.cluster_count, 2, 16, "cluster_count")
        _integer(self.concepts_per_cluster, 2, 16, "concepts_per_cluster")
        _integer(self.minimum_document_frequency, 2, 64, "minimum_document_frequency")
        _integer(self.minimum_concept_sources, 1, 16, "minimum_concept_sources")
        if not 0.1 <= self.maximum_document_ratio <= 0.95:
            raise OntologyError("maximum_document_ratio must be within 0.1-0.95")
        _integer(self.maximum_features, 100, 50_000, "maximum_features")
        _integer(self.evidence_per_concept, 1, 16, "evidence_per_concept")
        _integer(self.maximum_relations, 0, 256, "maximum_relations")
        _integer(self.minimum_relation_sources, 1, 16, "minimum_relation_sources")
        _integer(self.maximum_fragments, 20, 500, "maximum_fragments")

    @property
    def profile_hash(self) -> str:
        return canonical_sha256_hex(
            {
                "schema": self.SCHEMA,
                "method": "scoped_tfidf_nmf_literal_closure_v2",
                "cluster_count": self.cluster_count,
                "concepts_per_cluster": self.concepts_per_cluster,
                "minimum_document_frequency": self.minimum_document_frequency,
                "minimum_concept_sources": self.minimum_concept_sources,
                "maximum_document_ratio": self.maximum_document_ratio.hex(),
                "maximum_features": self.maximum_features,
                "evidence_per_concept": self.evidence_per_concept,
                "maximum_relations": self.maximum_relations,
                "minimum_relation_sources": self.minimum_relation_sources,
                "maximum_fragments": self.maximum_fragments,
            }
        )


def build_candidate_ontology(
    *,
    scope_label: str,
    original_question: str,
    query_cloud: tuple[str, ...],
    admitted_fragment_count: int,
    fragments: Sequence[OntologyFragment],
    generated_at: datetime,
    config: OntologyConfig | None = None,
    model_role: Literal["none", "bounded_relevance_filter"] = "none",
    model_receipt_hash: str | None = None,
) -> CandidateOntologyManifest:
    """Extract concepts from one bounded neighbourhood without generative calls."""

    resolved = OntologyConfig() if config is None else config
    ordered = _validate_inputs(fragments, admitted_fragment_count, resolved)
    if not query_cloud or len(query_cloud) > 32 or any(not item.strip() for item in query_cloud):
        raise OntologyError("query cloud must contain 1-32 nonblank queries")
    if model_role == "none" and model_receipt_hash is not None:
        raise OntologyError("model receipt is invalid when no relevance model was used")
    if model_role != "none":
        _hash(model_receipt_hash, "model_receipt_hash")

    vectorizer_type, nmf_type, stop_words, runtime_hash = _load_runtime()
    query_frequencies: dict[str, int] = defaultdict(int)
    for query in query_cloud:
        for token in set(_WORD.findall(query.casefold())):
            query_frequencies[token] += 1
    query_terms = {
        token
        for token, count in query_frequencies.items()
        if count >= max(2, math.ceil(len(query_cloud) / 2))
    }
    protected_tokens = {
        token.casefold() for token in _WORD.findall(" ".join((original_question, *query_cloud)))
    }
    proper_names = _proper_name_stop_words(ordered, protected_tokens=protected_tokens)
    domain_stop_words = sorted(
        set(stop_words).union(
            query_terms,
            _MARKUP_STOP_WORDS,
            _LOW_INFORMATION_STOP_WORDS,
            proper_names,
        )
    )
    vectorizer = vectorizer_type(
        lowercase=True,
        stop_words=domain_stop_words,
        ngram_range=(1, 3),
        min_df=resolved.minimum_document_frequency,
        max_df=resolved.maximum_document_ratio,
        max_features=resolved.maximum_features,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z\u2019'-]{2,}\b",
    )
    try:
        matrix = vectorizer.fit_transform([item.text for item in ordered])
    except ValueError as error:
        raise OntologyError("scoped ontology has no usable concept vocabulary") from error
    rows, columns = matrix.shape
    component_count = min(resolved.cluster_count, rows - 1, columns)
    if component_count < 2:
        raise OntologyError("scoped ontology requires at least two extractable components")
    extractor = nmf_type(
        n_components=component_count,
        init="nndsvda",
        random_state=7,
        max_iter=800,
        l1_ratio=0.0,
        alpha_W=0.0,
        alpha_H=0.0,
    )
    loadings = extractor.fit_transform(matrix)
    components = extractor.components_
    features = cast("_FeatureNames", vectorizer.get_feature_names_out())

    evidence: dict[str, OntologyEvidence] = {}
    concepts: list[OntologyConcept] = []
    clusters: list[OntologyCluster] = []
    concept_occurrences: dict[str, frozenset[int]] = {}
    normalized_texts = tuple(_normalized_text(item.text) for item in ordered)
    anchors = _query_anchors(
        query_cloud,
        normalized_texts=normalized_texts,
        fragments=ordered,
        stop_words=frozenset(domain_stop_words),
        loadings=loadings,
        component_count=component_count,
        minimum_document_frequency=resolved.minimum_document_frequency,
        minimum_concept_sources=resolved.minimum_concept_sources,
    )
    claimed_labels: set[str] = set()

    for component_index in range(component_count):
        cluster_id = canonical_content_id(
            "ontology_cluster",
            {"scope": scope_label, "component": component_index, "profile": resolved.profile_hash},
        )
        emergent_labels = _component_labels(
            components[component_index],
            features=features,
            maximum=resolved.concepts_per_cluster,
        )
        anchor_limit = max(1, resolved.concepts_per_cluster - 2)
        anchor_labels = anchors.get(component_index, ())[:anchor_limit]
        labels = tuple(
            [
                (label, raw_score, "scope_anchor", occurrences)
                for label, raw_score, occurrences in anchor_labels
            ]
            + [(label, raw_score, "emergent", None) for label, raw_score in emergent_labels]
        )
        cluster_concepts: list[OntologyConcept] = []
        cluster_evidence_ids: set[str] = set()
        for label, raw_score, origin, anchor_occurrences in labels:
            if label in claimed_labels or any(
                label in existing or existing in label for existing in claimed_labels
            ):
                continue
            occurrences = (
                anchor_occurrences
                if anchor_occurrences is not None
                else frozenset(
                    index
                    for index, text in enumerate(normalized_texts)
                    if _literal_present(label, text)
                )
            )
            if len(occurrences) < resolved.minimum_document_frequency:
                continue
            if (
                len({ordered[index].source_id for index in occurrences})
                < resolved.minimum_concept_sources
            ):
                continue
            concept_id = canonical_content_id(
                "ontology_concept",
                {"scope": scope_label, "cluster_id": cluster_id, "label": label},
            )
            evidence_indexes = _diverse_evidence(
                occurrences,
                fragments=ordered,
                loadings=loadings,
                component_index=component_index,
                maximum=resolved.evidence_per_concept,
            )
            evidence_ids = tuple(
                _evidence(item=ordered[index], evidence=evidence).evidence_id
                for index in evidence_indexes
            )
            cluster_evidence_ids.update(evidence_ids)
            concept = OntologyConcept(
                concept_id=concept_id,
                cluster_id=cluster_id,
                label=label,
                score=_score(raw_score),
                evidence_ids=evidence_ids,
                source_count=len({evidence[item].source_id for item in evidence_ids}),
                origin=cast("Literal['scope_anchor', 'emergent']", origin),
            )
            concepts.append(concept)
            cluster_concepts.append(concept)
            concept_occurrences[concept_id] = occurrences
            claimed_labels.add(label)
            if len(cluster_concepts) >= resolved.concepts_per_cluster:
                break
        if len(cluster_concepts) < 2:
            continue
        label_candidates = sorted(
            cluster_concepts,
            key=lambda item: (item.origin != "scope_anchor", -float(item.score), item.label),
        )
        cluster_label = " · ".join(item.label for item in label_candidates[:2])
        clusters.append(
            OntologyCluster(
                cluster_id=cluster_id,
                label=cluster_label,
                concept_ids=tuple(item.concept_id for item in cluster_concepts),
                evidence_ids=tuple(sorted(cluster_evidence_ids)),
                source_count=len({evidence[item].source_id for item in cluster_evidence_ids}),
            )
        )

    valid_cluster_ids = {item.cluster_id for item in clusters}
    concepts = [item for item in concepts if item.cluster_id in valid_cluster_ids]
    if len(concepts) < 2 or not clusters:
        raise OntologyError("scoped ontology did not produce a closed concept cluster")
    relations = _relations(
        concepts,
        occurrences=concept_occurrences,
        fragments=ordered,
        evidence=evidence,
        config=resolved,
    )
    input_hash = canonical_sha256_hex(
        {
            "schema": "dithyramba.ontology_input/1.0",
            "items": [item.identity_payload() for item in ordered],
        }
    )
    ontology_id = canonical_content_id(
        "candidate_ontology",
        {
            "scope": scope_label,
            "question": original_question,
            "query_cloud": list(query_cloud),
            "input_manifest_hash": input_hash,
            "extractor": resolved.profile_hash,
            "runtime": runtime_hash,
        },
    )
    multi_source = sum(item.source_count >= 2 for item in relations)
    return CandidateOntologyManifest(
        schema_id=CandidateOntologyManifest.SCHEMA,
        ontology_id=ontology_id,
        scope_label=scope_label,
        original_question=original_question,
        language="en",
        generated_at=generated_at,
        model_role=model_role,
        model_receipt_hash=model_receipt_hash,
        query_cloud=query_cloud,
        input_manifest_hash=input_hash,
        extractor_profile=canonical_sha256_hex(
            {"config": resolved.profile_hash, "runtime": runtime_hash}
        ),
        clusters=tuple(clusters),
        concepts=tuple(concepts),
        relations=relations,
        evidence=tuple(sorted(evidence.values(), key=lambda item: item.evidence_id)),
        metrics=OntologyMetrics(
            admitted_fragment_count=admitted_fragment_count,
            selected_fragment_count=len(ordered),
            source_count=len({item.source_id for item in evidence.values()}),
            concept_count=len(concepts),
            scope_anchor_count=sum(item.origin == "scope_anchor" for item in concepts),
            emergent_concept_count=sum(item.origin == "emergent" for item in concepts),
            cluster_count=len(clusters),
            relation_count=len(relations),
            multi_source_relation_ratio=_score(multi_source / len(relations) if relations else 0.0),
        ),
    )


def _load_runtime() -> tuple[Any, Any, frozenset[str], str]:
    try:
        sklearn = importlib.import_module("sklearn")
        decomposition = importlib.import_module("sklearn.decomposition")
        text = importlib.import_module("sklearn.feature_extraction.text")
    except ImportError as error:
        raise OntologyError(
            "candidate ontology requires the optional 'ontology' runtime"
        ) from error
    runtime_hash = canonical_sha256_hex(
        {"schema": "dithyramba.ontology_runtime/1.0", "scikit_learn": sklearn.__version__}
    )
    return (
        text.TfidfVectorizer,
        decomposition.NMF,
        frozenset(text.ENGLISH_STOP_WORDS),
        runtime_hash,
    )


def _validate_inputs(
    fragments: Sequence[OntologyFragment],
    admitted_fragment_count: int,
    config: OntologyConfig,
) -> tuple[OntologyFragment, ...]:
    received = tuple(fragments)
    if any(type(item) is not OntologyFragment for item in received):
        raise OntologyError("ontology inputs must contain OntologyFragment values")
    if not 20 <= len(received) <= config.maximum_fragments:
        raise OntologyError("selected ontology fragment count is outside the configured bound")
    if type(admitted_fragment_count) is not int or admitted_fragment_count < len(received):
        raise OntologyError("admitted fragment count must cover the selected neighbourhood")
    identifiers = tuple(item.source_fragment_id for item in received)
    if len(set(identifiers)) != len(identifiers):
        raise OntologyError("ontology fragment IDs must be unique")
    ranks = tuple(item.rank for item in received)
    if len(set(ranks)) != len(ranks):
        raise OntologyError("ontology fragment ranks must be unique")
    return tuple(sorted(received, key=lambda item: (item.rank, item.source_fragment_id)))


def _component_labels(
    weights: Any,
    *,
    features: _FeatureNames,
    maximum: int,
) -> tuple[tuple[str, float], ...]:
    ranked = sorted(
        range(len(weights)),
        key=lambda index: (-float(weights[index]), features[index]),
    )
    selected: list[tuple[str, float]] = []
    for index in ranked:
        label = features[index].casefold().strip(" -'")
        if len(label) < 3 or any(
            label in existing or existing in label for existing, _score_value in selected
        ):
            continue
        selected.append((label, float(weights[index])))
        if len(selected) >= maximum:
            break
    return tuple(selected)


def _proper_name_stop_words(
    fragments: tuple[OntologyFragment, ...],
    *,
    protected_tokens: set[str],
) -> frozenset[str]:
    """Remove repeated proper names without suppressing words in the declared scope."""

    capitalized_sources: dict[str, set[str]] = defaultdict(set)
    lowercase_tokens: set[str] = set()
    for fragment in fragments:
        for token in _WORD.findall(fragment.text):
            normalized = token.casefold()
            if token[0].isupper() and not token.isupper():
                capitalized_sources[normalized].add(fragment.source_id)
            else:
                lowercase_tokens.add(normalized)
    return frozenset(
        token
        for token, sources in capitalized_sources.items()
        if len(sources) >= 2 and token not in lowercase_tokens and token not in protected_tokens
    )


def _query_anchors(
    query_cloud: tuple[str, ...],
    *,
    normalized_texts: tuple[str, ...],
    fragments: tuple[OntologyFragment, ...],
    stop_words: frozenset[str],
    loadings: Any,
    component_count: int,
    minimum_document_frequency: int,
    minimum_concept_sources: int,
) -> dict[int, tuple[tuple[str, float, frozenset[int]], ...]]:
    candidates: dict[str, frozenset[int]] = {}
    for query in query_cloud:
        tokens = [
            token.casefold() for token in _WORD.findall(query) if token.casefold() not in stop_words
        ]
        for size in (3, 2, 1):
            for start in range(0, len(tokens) - size + 1):
                label = " ".join(tokens[start : start + size])
                occurrences = frozenset(
                    index
                    for index, text in enumerate(normalized_texts)
                    if _literal_present(label, text)
                )
                if len(occurrences) < minimum_document_frequency:
                    continue
                source_count = len({fragments[index].source_id for index in occurrences})
                if source_count < minimum_concept_sources:
                    continue
                candidates[label] = occurrences

    selected: list[tuple[str, frozenset[int]]] = []
    for label, occurrences in sorted(
        candidates.items(),
        key=lambda item: (
            -len(item[0].split()),
            -len({fragments[index].source_id for index in item[1]}),
            -len(item[1]),
            item[0],
        ),
    ):
        if any(label in existing or existing in label for existing, _ in selected):
            continue
        selected.append((label, occurrences))

    by_component: dict[int, list[tuple[str, float, frozenset[int]]]] = defaultdict(list)
    for label, occurrences in selected:
        scores = [
            sum(float(loadings[index, component]) for index in occurrences) / len(occurrences)
            for component in range(component_count)
        ]
        component_index = max(range(component_count), key=lambda item: scores[item])
        by_component[component_index].append((label, scores[component_index], occurrences))
    return {
        component: tuple(
            sorted(
                values,
                key=lambda item: (
                    -len(item[0].split()),
                    -len({fragments[index].source_id for index in item[2]}),
                    -item[1],
                    item[0],
                ),
            )
        )
        for component, values in by_component.items()
    }


def _diverse_evidence(
    indexes: frozenset[int],
    *,
    fragments: tuple[OntologyFragment, ...],
    loadings: Any,
    component_index: int,
    maximum: int,
) -> tuple[int, ...]:
    ranked = sorted(
        indexes,
        key=lambda index: (-float(loadings[index, component_index]), fragments[index].rank),
    )
    selected: list[int] = []
    used_sources: set[str] = set()
    for index in ranked:
        if fragments[index].source_id in used_sources:
            continue
        selected.append(index)
        used_sources.add(fragments[index].source_id)
        if len(selected) >= maximum:
            return tuple(selected)
    for index in ranked:
        if index not in selected:
            selected.append(index)
        if len(selected) >= maximum:
            break
    return tuple(selected)


def _relations(
    concepts: list[OntologyConcept],
    *,
    occurrences: dict[str, frozenset[int]],
    fragments: tuple[OntologyFragment, ...],
    evidence: dict[str, OntologyEvidence],
    config: OntologyConfig,
) -> tuple[OntologyRelation, ...]:
    candidates: list[tuple[int, int, str, str, tuple[int, ...]]] = []
    for left_index, left in enumerate(concepts):
        for right in concepts[left_index + 1 :]:
            shared = tuple(sorted(occurrences[left.concept_id] & occurrences[right.concept_id]))
            if not shared:
                continue
            source_count = len({fragments[index].source_id for index in shared})
            if source_count < config.minimum_relation_sources:
                continue
            source_concept_id, target_concept_id = sorted((left.concept_id, right.concept_id))
            candidates.append(
                (
                    source_count,
                    len(shared),
                    source_concept_id,
                    target_concept_id,
                    shared,
                )
            )
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    relations: list[OntologyRelation] = []
    for source_count, shared_count, left_id, right_id, shared in candidates[
        : config.maximum_relations
    ]:
        evidence_indexes = _source_diverse_indexes(shared, fragments=fragments, maximum=8)
        evidence_ids = tuple(
            _evidence(item=fragments[index], evidence=evidence).evidence_id
            for index in evidence_indexes
        )
        relation_id = canonical_content_id(
            "ontology_relation",
            {
                "kind": OntologyRelationKind.CO_OCCURS_WITH.value,
                "source": left_id,
                "target": right_id,
                "evidence": list(evidence_ids),
            },
        )
        relations.append(
            OntologyRelation(
                relation_id=relation_id,
                kind=OntologyRelationKind.CO_OCCURS_WITH,
                source_concept_id=left_id,
                target_concept_id=right_id,
                evidence_ids=evidence_ids,
                shared_fragment_count=shared_count,
                source_count=source_count,
            )
        )
    return tuple(sorted(relations, key=lambda item: item.relation_id))


def _source_diverse_indexes(
    indexes: tuple[int, ...],
    *,
    fragments: tuple[OntologyFragment, ...],
    maximum: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    sources: set[str] = set()
    for index in sorted(indexes, key=lambda item: fragments[item].rank):
        source_id = fragments[index].source_id
        if source_id in sources:
            continue
        selected.append(index)
        sources.add(source_id)
        if len(selected) >= maximum:
            break
    return tuple(selected)


def _evidence(
    *,
    item: OntologyFragment,
    evidence: dict[str, OntologyEvidence],
) -> OntologyEvidence:
    evidence_id = canonical_content_id(
        "ontology_evidence",
        {
            "source_fragment_id": item.source_fragment_id,
            "text_sha256": item.text_sha256,
            "address_hash": item.address_hash,
        },
    )
    existing = evidence.get(evidence_id)
    if existing is not None:
        return existing
    excerpt = re.sub(r"\s+", " ", item.text).strip()
    if len(excerpt) > 900:
        excerpt = excerpt[:897].rstrip() + "..."
    created = OntologyEvidence(
        evidence_id=evidence_id,
        source_fragment_id=item.source_fragment_id,
        source_id=item.source_id,
        source_version_id=item.source_version_id,
        source_title=item.source_title,
        source_creator=item.source_creator,
        source_kind=item.source_kind,
        locator=item.locator,
        text_sha256=item.text_sha256,
        address_hash=item.address_hash,
        source_address=dict(item.source_address),
        excerpt=excerpt,
    )
    evidence[evidence_id] = created
    return created


def _normalized_text(value: str) -> str:
    return " ".join(token.casefold() for token in _WORD.findall(value))


def _literal_present(label: str, normalized_text: str) -> bool:
    return f" {label} " in f" {normalized_text} "


def _score(value: float) -> str:
    if not math.isfinite(value) or value < 0.0:
        raise OntologyError("ontology score must be finite and non-negative")
    return f"{value:.6f}"


def _hash(value: object, label: str) -> None:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise OntologyError(f"{label} must be lowercase SHA-256 hex")


def _prefixed(value: str, prefix: str, label: str) -> None:
    if type(value) is not str or not value.startswith(prefix) or len(value) <= len(prefix):
        raise OntologyError(f"{label} must use {prefix}<key>")


def _integer(value: int, minimum: int, maximum: int, label: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise OntologyError(f"{label} must be within {minimum}-{maximum}")
