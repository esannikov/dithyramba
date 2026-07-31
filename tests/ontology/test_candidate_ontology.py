"""Contract and construction tests for scoped candidate ontologies."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.ontology import (
    CandidateOntologyManifest,
    OntologyConfig,
    OntologyError,
    OntologyFragment,
    build_candidate_ontology,
    load_candidate_ontology,
)

STAMP = datetime(2026, 7, 30, tzinfo=UTC)


def _fragments() -> tuple[OntologyFragment, ...]:
    themes = (
        "Actor objective and subtext guide rehearsal choices in a dialogue scene.",
        "Camera blocking and eyeline shape the shot reverse shot dialogue scene.",
        "Listening reaction, silence, and pause create dialogue rhythm.",
        "Physical action, voice delivery, and tempo shape actor performance.",
    )
    result: list[OntologyFragment] = []
    rank = 1
    for source_index in range(1, 7):
        for theme_index, theme in enumerate(themes, start=1):
            text = f"{theme} Source example {source_index}; exercise {theme_index}."
            address = {"kind": "markdown", "line_start": rank, "line_end": rank}
            result.append(
                OntologyFragment(
                    source_fragment_id=f"fragment_{source_index}_{theme_index}",
                    source_id=f"source_{source_index}",
                    source_version_id=f"source_version_{source_index}",
                    text=text,
                    text_sha256=sha256_hex(text.encode()),
                    address_hash=canonical_sha256_hex(address),
                    source_address=address,
                    source_title=f"Directing Book {source_index}",
                    source_creator=f"Author {source_index}",
                    source_kind="book",
                    locator=f"line {rank}",
                    rank=rank,
                )
            )
            rank += 1
    return tuple(result)


def build_manifest() -> CandidateOntologyManifest:
    fragments = _fragments()
    return build_candidate_ontology(
        scope_label="Dialogue directing",
        original_question="How should a director shape a dialogue scene?",
        query_cloud=(
            "actor objective subtext",
            "camera blocking eyeline",
            "listening silence rhythm",
            "physical action voice tempo",
        ),
        admitted_fragment_count=len(fragments),
        fragments=fragments,
        generated_at=STAMP,
        config=OntologyConfig(
            cluster_count=4,
            concepts_per_cluster=4,
            minimum_document_frequency=3,
            minimum_concept_sources=2,
            maximum_document_ratio=0.9,
            maximum_features=500,
            evidence_per_concept=3,
            maximum_relations=12,
            minimum_relation_sources=2,
            maximum_fragments=100,
        ),
    )


def test_builder_is_deterministic_closed_and_source_grounded() -> None:
    first = build_manifest()
    second = build_manifest()

    assert canonical_json_bytes(first.model_dump(mode="json")) == canonical_json_bytes(
        second.model_dump(mode="json")
    )
    assert first.mode == "candidate_only"
    assert first.model_role == "none"
    assert first.metrics.cluster_count >= 2
    assert first.metrics.concept_count >= 4
    assert first.metrics.source_count >= 2
    assert all(concept.state == "candidate" for concept in first.concepts)
    assert all(relation.kind.value == "co_occurs_with" for relation in first.relations)
    assert all(relation.source_count >= 2 for relation in first.relations)
    assert {item.evidence_id for item in first.evidence} >= {
        evidence_id for concept in first.concepts for evidence_id in concept.evidence_ids
    }


def test_builder_accepts_bounded_relevance_receipt() -> None:
    fragments = _fragments()
    manifest = build_candidate_ontology(
        scope_label="Dialogue directing",
        original_question="How should a director shape a dialogue scene?",
        query_cloud=("subtext objective", "blocking camera"),
        admitted_fragment_count=len(fragments),
        fragments=fragments,
        generated_at=STAMP,
        config=OntologyConfig(
            cluster_count=3,
            concepts_per_cluster=3,
            minimum_document_frequency=3,
            minimum_concept_sources=2,
            maximum_document_ratio=0.9,
            maximum_features=300,
            maximum_fragments=100,
        ),
        model_role="bounded_relevance_filter",
        model_receipt_hash="a" * 64,
    )

    assert manifest.model_receipt_hash == "a" * 64
    assert manifest.model_role == "bounded_relevance_filter"


def test_loader_revalidates_manifest_and_hashes_canonical_bytes(tmp_path: Path) -> None:
    manifest = build_manifest()
    path = tmp_path / "ontology.json"
    path.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))

    loaded = load_candidate_ontology(path)

    assert loaded.manifest == manifest
    assert loaded.manifest_path == path
    assert len(loaded.manifest_hash) == 64
    with pytest.raises(ValueError, match="regular file"):
        load_candidate_ontology(tmp_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rank": 0}, "rank"),
        ({"source_fragment_id": "bad"}, "source_fragment_id"),
        ({"source_id": "bad"}, "source_id"),
        ({"source_version_id": "bad"}, "source_version_id"),
        ({"text": ""}, "bounded NFC text"),
        ({"text": "bad\x00text"}, "bounded NFC text"),
        ({"text_sha256": "bad"}, "text_sha256"),
        ({"address_hash": "bad"}, "address_hash"),
        ({"source_address": {}}, "explicit source address"),
        ({"source_title": ""}, "source_title"),
    ],
)
def test_fragment_rejects_invalid_identity_and_text(changes: dict[str, Any], message: str) -> None:
    base = _fragments()[0]
    with pytest.raises(OntologyError, match=message):
        replace(base, **changes)


def test_fragment_rejects_hash_mismatch_and_non_nfc() -> None:
    base = _fragments()[0]
    text = "Cafe\u0301 dialogue"
    with pytest.raises(OntologyError, match="bounded NFC"):
        replace(base, text=text, text_sha256=sha256_hex(text.encode()))
    with pytest.raises(OntologyError, match="does not match"):
        replace(base, text="different")


@pytest.mark.parametrize(
    "changes",
    [
        {"cluster_count": 1},
        {"concepts_per_cluster": 17},
        {"minimum_document_frequency": 1},
        {"minimum_concept_sources": 0},
        {"maximum_document_ratio": 0.01},
        {"maximum_features": 99},
        {"evidence_per_concept": 17},
        {"maximum_relations": -1},
        {"minimum_relation_sources": 17},
        {"maximum_fragments": 19},
    ],
)
def test_config_rejects_unbounded_values(changes: dict[str, Any]) -> None:
    with pytest.raises(OntologyError):
        OntologyConfig(**changes)


@pytest.mark.parametrize(
    ("fragments", "admitted", "cloud", "role", "receipt", "message"),
    [
        ((_fragments()[:19]), 19, ("valid",), "none", None, "outside"),
        ((_fragments()), 23, ("valid",), "none", None, "must cover"),
        ((_fragments()), 24, (), "none", None, "query cloud"),
        ((_fragments()), 24, ("valid",), "none", "a" * 64, "receipt"),
        ((_fragments()), 24, ("valid",), "bounded_relevance_filter", None, "SHA-256"),
    ],
)
def test_builder_rejects_invalid_scope_inputs(
    fragments: tuple[OntologyFragment, ...],
    admitted: int,
    cloud: tuple[str, ...],
    role: str,
    receipt: str | None,
    message: str,
) -> None:
    with pytest.raises(OntologyError, match=message):
        build_candidate_ontology(
            scope_label="Dialogue directing",
            original_question="How?",
            query_cloud=cloud,
            admitted_fragment_count=admitted,
            fragments=fragments,
            generated_at=STAMP,
            model_role=role,  # type: ignore[arg-type]
            model_receipt_hash=receipt,
        )


def test_builder_rejects_duplicate_identifiers_and_ranks() -> None:
    items = list(_fragments())
    items[1] = replace(items[1], source_fragment_id=items[0].source_fragment_id)
    with pytest.raises(OntologyError, match="IDs must be unique"):
        _build_default(tuple(items))
    items = list(_fragments())
    items[1] = replace(items[1], rank=items[0].rank)
    with pytest.raises(OntologyError, match="ranks must be unique"):
        _build_default(tuple(items))


def test_builder_rejects_wrong_item_type_and_empty_vocabulary() -> None:
    with pytest.raises(OntologyError, match="OntologyFragment"):
        _build_default(cast("tuple[OntologyFragment, ...]", tuple([object()] * 20)))
    items = tuple(
        replace(
            item,
            text="the and but for with from",
            text_sha256=sha256_hex(b"the and but for with from"),
        )
        for item in _fragments()
    )
    with pytest.raises(OntologyError, match="no usable concept vocabulary"):
        _build_default(items)


def test_manifest_rejects_tampered_closure() -> None:
    payload = build_manifest().model_dump(mode="json")
    payload["metrics"]["concept_count"] += 1
    with pytest.raises(ValidationError, match="metrics do not close"):
        CandidateOntologyManifest.model_validate(payload)

    payload = build_manifest().model_dump(mode="json")
    payload["concepts"][0]["evidence_ids"] = ["ontology_evidence_missing"]
    with pytest.raises(ValidationError, match="missing IDs"):
        CandidateOntologyManifest.model_validate(payload)

    payload = build_manifest().model_dump(mode="json")
    payload["schema_id"] = "wrong"
    with pytest.raises(ValidationError, match="schema_id"):
        CandidateOntologyManifest.model_validate(payload)


def test_manifest_rejects_tampered_internal_links_and_counts() -> None:
    def reject(payload: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            CandidateOntologyManifest.model_validate(payload)

    payload = build_manifest().model_dump(mode="json")
    payload["model_receipt_hash"] = "a" * 64
    reject(payload, "model receipt")

    payload = build_manifest().model_dump(mode="json")
    payload["evidence"][0]["source_address"] = {}
    reject(payload, "explicit source address")

    payload = build_manifest().model_dump(mode="json")
    payload["relations"][0]["target_concept_id"] = payload["relations"][0]["source_concept_id"]
    reject(payload, "distinct and canonical")

    payload = build_manifest().model_dump(mode="json")
    payload["clusters"][1]["cluster_id"] = payload["clusters"][0]["cluster_id"]
    reject(payload, "cluster identifiers must be unique")

    payload = build_manifest().model_dump(mode="json")
    payload["concepts"][1]["concept_id"] = payload["concepts"][0]["concept_id"]
    reject(payload, "concept identifiers must be unique")

    payload = build_manifest().model_dump(mode="json")
    payload["evidence"][1]["evidence_id"] = payload["evidence"][0]["evidence_id"]
    reject(payload, "evidence identifiers must be unique")

    payload = build_manifest().model_dump(mode="json")
    payload["relations"][1]["relation_id"] = payload["relations"][0]["relation_id"]
    reject(payload, "relation identifiers must be unique")

    payload = build_manifest().model_dump(mode="json")
    payload["concepts"][0]["cluster_id"] = "cluster_missing"
    reject(payload, "missing ontology cluster")

    payload = build_manifest().model_dump(mode="json")
    payload["concepts"][0]["source_count"] += 1
    reject(payload, "concept source_count")

    payload = build_manifest().model_dump(mode="json")
    payload["clusters"][0]["concept_ids"] = payload["clusters"][0]["concept_ids"][1:]
    reject(payload, "membership is not closed")

    payload = build_manifest().model_dump(mode="json")
    payload["clusters"][0]["source_count"] += 1
    reject(payload, "cluster source_count")

    payload = build_manifest().model_dump(mode="json")
    payload["relations"][0]["source_concept_id"] = "concept_missing"
    reject(payload, "missing concept")

    payload = build_manifest().model_dump(mode="json")
    payload["relations"][0]["source_count"] += 1
    reject(payload, "relation source_count")

    payload = build_manifest().model_dump(mode="json")
    payload["metrics"]["scope_anchor_count"] += 1
    reject(payload, "concept-origin metrics")

    payload = build_manifest().model_dump(mode="json")
    payload["metrics"]["selected_fragment_count"] = (
        payload["metrics"]["admitted_fragment_count"] + 1
    )
    reject(payload, "cannot exceed")


def test_profile_hash_changes_with_configuration() -> None:
    assert OntologyConfig().profile_hash != OntologyConfig(cluster_count=6).profile_hash


def test_builder_filters_repeated_proper_names_outside_declared_scope() -> None:
    fragments = tuple(
        replace(
            item,
            text=f"{item.text} George explains the example.",
            text_sha256=sha256_hex(f"{item.text} George explains the example.".encode()),
        )
        for item in _fragments()
    )

    manifest = _build_default(fragments)

    assert all("george" not in concept.label for concept in manifest.concepts)


def _build_default(fragments: tuple[OntologyFragment, ...]) -> CandidateOntologyManifest:
    return build_candidate_ontology(
        scope_label="Dialogue directing",
        original_question="How?",
        query_cloud=("subtext", "camera"),
        admitted_fragment_count=len(fragments),
        fragments=fragments,
        generated_at=STAMP,
        config=OntologyConfig(maximum_fragments=100),
    )
