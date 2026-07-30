"""Acceptance checks for the scoped candidate-ontology Lens."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dithyramba.api import create_concept_lens_app
from dithyramba.contracts import canonical_json_bytes
from dithyramba.ontology import CandidateOntologyManifest
from tests.ontology.test_candidate_ontology import build_manifest


def _client(tmp_path: Path) -> tuple[TestClient, CandidateOntologyManifest]:
    manifest = build_manifest()
    path = tmp_path / "candidate-ontology.json"
    path.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    app = create_concept_lens_app(
        projection_path=path,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )
    return TestClient(app), manifest


def _presentation_payload(manifest: CandidateOntologyManifest) -> dict[str, Any]:
    return {
        "schema": "dithyramba.concept_lens_presentation/1.0",
        "ontology_id": manifest.ontology_id,
        "title": "Режисура діалогу",
        "question": "Як перетворити розмову на сцену?",
        "lead": "Людський шар пояснює машинну карту.",
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "title": f"Область {index}",
                "question": f"Питання {index}?",
                "summary": f"Пояснення області {index}.",
                "entry_concept_id": (
                    None if index == len(manifest.clusters) else cluster.concept_ids[0]
                ),
            }
            for index, cluster in enumerate(manifest.clusters, start=1)
        ],
    }


def test_lens_renders_clusters_limits_and_exact_source(tmp_path: Path) -> None:
    client, manifest = _client(tmp_path)
    response = client.get("/")

    assert response.status_code == 200
    assert manifest.original_question in response.text
    assert "карта-кандидат" in response.text
    assert "Точний уривок" in response.text
    assert manifest.clusters[0].label in response.text
    assert "Directing Book" in response.text
    assert "<script" not in response.text
    assert response.headers["content-security-policy"].startswith("default-src 'self'")


def test_lens_selects_concept_and_evidence(tmp_path: Path) -> None:
    client, manifest = _client(tmp_path)
    concept = manifest.concepts[-1]
    evidence_id = concept.evidence_ids[-1]
    response = client.get(
        "/",
        params={
            "cluster": concept.cluster_id,
            "concept": concept.concept_id,
            "evidence": evidence_id,
        },
    )

    assert response.status_code == 200
    assert concept.label in response.text
    assert 'aria-current="true"' in response.text
    assert evidence_id in response.text


def test_lens_rejects_unknown_or_cross_cluster_selection(tmp_path: Path) -> None:
    client, manifest = _client(tmp_path)
    concept = manifest.concepts[-1]
    other = next(item for item in manifest.clusters if item.cluster_id != concept.cluster_id)

    assert client.get("/?cluster=cluster_unknown").status_code == 404
    assert client.get("/?concept=concept_unknown").status_code == 404
    response = client.get(
        "/",
        params={"cluster": other.cluster_id, "concept": concept.concept_id},
    )
    assert response.status_code == 404
    assert client.get("/?evidence=ontology_evidence_missing").status_code == 404


def test_lens_json_and_styles_are_local_and_responsive(tmp_path: Path) -> None:
    client, manifest = _client(tmp_path)
    projection = client.get("/ontology.json")
    styles = client.get("/concept-lens.css")

    assert projection.status_code == 200
    assert projection.json()["ontology_id"] == manifest.ontology_id
    assert "manifest_hash" in projection.json()
    assert styles.status_code == 200
    assert "@media (max-width: 58rem)" in styles.text
    assert "linear-gradient" not in styles.text
    assert client.get("/favicon.ico").status_code == 204


def test_lens_accepts_ontology_bound_human_presentation(tmp_path: Path) -> None:
    manifest = build_manifest()
    projection = tmp_path / "candidate-ontology.json"
    projection.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    presentation = tmp_path / "presentation.json"
    presentation.write_text(
        json.dumps(_presentation_payload(manifest)),
        encoding="utf-8",
    )
    app = create_concept_lens_app(
        projection_path=projection,
        presentation_path=presentation,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
    )

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Як перетворити розмову на сцену?" in response.text
    assert "Питання 1?" in response.text
    assert "Область 1" in response.text


def test_lens_rejects_presentation_for_another_ontology(tmp_path: Path) -> None:
    manifest = build_manifest()
    projection = tmp_path / "candidate-ontology.json"
    projection.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    presentation = tmp_path / "presentation.json"
    presentation.write_text(
        json.dumps(
            {
                "schema": "dithyramba.concept_lens_presentation/1.0",
                "ontology_id": "candidate_ontology_other",
                "title": "Title",
                "question": "Question",
                "lead": "Lead",
                "clusters": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        create_concept_lens_app(
            projection_path=projection,
            presentation_path=presentation,
            allowed_origin="http://testserver",
            test_only_allow_testserver=True,
        )


def test_lens_rejects_malformed_or_incomplete_presentation(tmp_path: Path) -> None:
    manifest = build_manifest()
    projection = tmp_path / "candidate-ontology.json"
    projection.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    presentation = tmp_path / "presentation.json"

    def reject(payload: object, message: str) -> None:
        presentation.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            create_concept_lens_app(
                projection_path=projection,
                presentation_path=presentation,
                allowed_origin="http://testserver",
                test_only_allow_testserver=True,
            )

    with pytest.raises(ValueError, match="bounded regular file"):
        create_concept_lens_app(
            projection_path=projection,
            presentation_path=tmp_path / "missing.json",
            allowed_origin="http://testserver",
            test_only_allow_testserver=True,
        )

    presentation.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        create_concept_lens_app(
            projection_path=projection,
            presentation_path=presentation,
            allowed_origin="http://testserver",
            test_only_allow_testserver=True,
        )

    reject([], "JSON object")
    payload = _presentation_payload(manifest)
    payload["schema"] = "wrong"
    reject(payload, "unsupported")

    payload = _presentation_payload(manifest)
    payload["title"] = " "
    reject(payload, "bounded text")

    payload = _presentation_payload(manifest)
    payload["clusters"] = {}
    reject(payload, "cluster list")

    payload = _presentation_payload(manifest)
    payload["clusters"] = [1]
    reject(payload, "must be an object")

    payload = _presentation_payload(manifest)
    clusters = deepcopy(payload["clusters"])
    assert isinstance(clusters, list)
    clusters.append(deepcopy(clusters[0]))
    payload["clusters"] = clusters
    reject(payload, "unique strings")

    payload = _presentation_payload(manifest)
    clusters = deepcopy(payload["clusters"])
    assert isinstance(clusters, list)
    payload["clusters"] = clusters[:-1]
    reject(payload, "cover every")

    payload = _presentation_payload(manifest)
    clusters = deepcopy(payload["clusters"])
    assert isinstance(clusters, list)
    assert isinstance(clusters[0], dict)
    clusters[0]["entry_concept_id"] = "bad id"
    payload["clusters"] = clusters
    reject(payload, "stable identifier")

    payload = _presentation_payload(manifest)
    clusters = deepcopy(payload["clusters"])
    assert isinstance(clusters, list)
    assert isinstance(clusters[0], dict)
    clusters[0]["entry_concept_id"] = manifest.clusters[1].concept_ids[0]
    payload["clusters"] = clusters
    reject(payload, "belong to its cluster")
