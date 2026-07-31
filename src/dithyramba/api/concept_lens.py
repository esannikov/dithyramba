"""Loopback, GET-only renderer for one scoped candidate ontology."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dithyramba.contracts import canonical_json_bytes
from dithyramba.ontology import (
    LoadedCandidateOntology,
    OntologyCluster,
    OntologyConcept,
    OntologyEvidence,
    OntologyRelation,
    load_candidate_ontology,
)

from .config import LoopbackApiConfig, build_config
from .security import LoopbackSecurityMiddleware, new_bearer_token

_TEMPLATES = Environment(
    loader=PackageLoader("dithyramba.api", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html",), default_for_string=True),
    undefined=StrictUndefined,
    enable_async=False,
)
_STYLESHEET = Path(__file__).with_name("static") / "concept_lens.css"
_LENS_LIBRARY_ID = "library_00000000000040008000000000000002"
_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"


@dataclass(frozen=True, slots=True)
class ConceptLensWebConfig:
    loopback: LoopbackApiConfig
    ontology: LoadedCandidateOntology
    presentation: _LensPresentation


@dataclass(frozen=True, slots=True)
class _ClusterPresentation:
    title: str
    question: str
    summary: str
    entry_concept_id: str | None


@dataclass(frozen=True, slots=True)
class _LensPresentation:
    title: str
    question: str
    lead: str
    clusters: dict[str, _ClusterPresentation]


@dataclass(frozen=True, slots=True)
class _ClusterView:
    cluster: OntologyCluster
    concepts: tuple[OntologyConcept, ...]
    entry_concept: OntologyConcept
    presentation: _ClusterPresentation
    selected: bool


@dataclass(frozen=True, slots=True)
class _RelationView:
    relation: OntologyRelation
    counterpart: OntologyConcept
    evidence: tuple[OntologyEvidence, ...]


def create_concept_lens_app(
    *,
    projection_path: str | Path,
    presentation_path: str | Path | None = None,
    allowed_origin: str,
    test_only_allow_testserver: bool = False,
) -> FastAPI:
    """Create an offline concept observatory from one validated projection."""

    ontology = load_candidate_ontology(projection_path)
    presentation = _load_presentation(presentation_path, ontology=ontology)
    loopback = build_config(
        library_id=_LENS_LIBRARY_ID,
        data_home=ontology.manifest_path.parent,
        allowed_origin=allowed_origin,
        test_only_allow_testserver=test_only_allow_testserver,
    )
    config = ConceptLensWebConfig(
        loopback=loopback,
        ontology=ontology,
        presentation=presentation,
    )
    app = FastAPI(debug=False, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.concept_lens_config = config
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(loopback.trusted_hosts),
        www_redirect=False,
    )
    app.add_middleware(
        LoopbackSecurityMiddleware,
        bearer_token=new_bearer_token(),
        allowed_origin=loopback.allowed_origin,
        trusted_hosts=loopback.trusted_hosts,
        max_request_body_bytes=loopback.max_request_body_bytes,
    )

    @app.get("/concept-lens.css", include_in_schema=False)
    async def stylesheet() -> Response:
        return Response(
            _STYLESHEET.read_text(encoding="utf-8"),
            media_type="text/css; charset=utf-8",
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/ontology.json", include_in_schema=False)
    async def ontology_json() -> Response:
        payload = ontology.manifest.model_dump(mode="json")
        payload["manifest_hash"] = ontology.manifest_hash
        return Response(
            canonical_json_bytes(payload),
            media_type="application/json",
            headers={"ETag": f'"{ontology.manifest_hash}"'},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(
        cluster: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        concept: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        evidence: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
    ) -> HTMLResponse:
        context = _template_context(
            ontology,
            presentation=presentation,
            selected_cluster_id=cluster,
            selected_concept_id=concept,
            selected_evidence_id=evidence,
        )
        return HTMLResponse(_TEMPLATES.get_template("concept_lens.html").render(**context))

    return app


def _template_context(
    loaded: LoadedCandidateOntology,
    *,
    presentation: _LensPresentation,
    selected_cluster_id: str | None,
    selected_concept_id: str | None,
    selected_evidence_id: str | None,
) -> dict[str, object]:
    manifest = loaded.manifest
    cluster_by_id = {item.cluster_id: item for item in manifest.clusters}
    concept_by_id = {item.concept_id: item for item in manifest.concepts}
    evidence_by_id = {item.evidence_id: item for item in manifest.evidence}

    def entry_concept(cluster_value: OntologyCluster) -> OntologyConcept:
        preferred = presentation.clusters[cluster_value.cluster_id].entry_concept_id
        if preferred is not None:
            return concept_by_id[preferred]
        return concept_by_id[cluster_value.concept_ids[0]]

    selected_concept = None
    if selected_concept_id is not None:
        selected_concept = concept_by_id.get(selected_concept_id)
        if selected_concept is None:
            raise HTTPException(status_code=404, detail="not found")
        if selected_cluster_id is not None and selected_concept.cluster_id != selected_cluster_id:
            raise HTTPException(status_code=404, detail="not found")
        selected_cluster_id = selected_concept.cluster_id

    if selected_cluster_id is None:
        selected_cluster_id = manifest.clusters[0].cluster_id
    selected_cluster = cluster_by_id.get(selected_cluster_id)
    if selected_cluster is None:
        raise HTTPException(status_code=404, detail="not found")
    if selected_concept is None:
        selected_concept = entry_concept(selected_cluster)

    allowed_evidence_ids = set(selected_concept.evidence_ids)
    relation_views: list[_RelationView] = []
    for relation in manifest.relations:
        endpoint_ids = {relation.source_concept_id, relation.target_concept_id}
        if selected_concept.concept_id not in endpoint_ids:
            continue
        counterpart_id = next(item for item in endpoint_ids if item != selected_concept.concept_id)
        allowed_evidence_ids.update(relation.evidence_ids)
        relation_views.append(
            _RelationView(
                relation=relation,
                counterpart=concept_by_id[counterpart_id],
                evidence=tuple(evidence_by_id[item] for item in relation.evidence_ids),
            )
        )
    relation_views.sort(
        key=lambda item: (
            -item.relation.source_count,
            -item.relation.shared_fragment_count,
            item.counterpart.label,
        )
    )

    if selected_evidence_id is None:
        selected_evidence_id = selected_concept.evidence_ids[0]
    if selected_evidence_id not in allowed_evidence_ids:
        raise HTTPException(status_code=404, detail="not found")
    selected_evidence = evidence_by_id[selected_evidence_id]

    cluster_views = tuple(
        _ClusterView(
            cluster=item,
            concepts=tuple(concept_by_id[concept_id] for concept_id in item.concept_ids),
            entry_concept=entry_concept(item),
            presentation=presentation.clusters[item.cluster_id],
            selected=item.cluster_id == selected_cluster.cluster_id,
        )
        for item in manifest.clusters
    )
    concept_evidence = tuple(evidence_by_id[item] for item in selected_concept.evidence_ids)
    source_count = len({item.source_id for item in manifest.evidence})
    return {
        "manifest": manifest,
        "presentation": presentation,
        "manifest_hash": loaded.manifest_hash,
        "cluster_views": cluster_views,
        "selected_cluster": selected_cluster,
        "selected_concept": selected_concept,
        "selected_evidence": selected_evidence,
        "concept_evidence": concept_evidence,
        "relation_views": tuple(relation_views),
        "source_count": source_count,
        "concept_url": _concept_url,
        "evidence_url": _evidence_url,
    }


def _load_presentation(
    path: str | Path | None,
    *,
    ontology: LoadedCandidateOntology,
) -> _LensPresentation:
    manifest = ontology.manifest
    if path is None:
        return _LensPresentation(
            title=manifest.scope_label,
            question=manifest.original_question,
            lead=(
                "Оберіть область, щоб перейти від загальної карти до понять, "
                "їхніх співпояв і точних джерел."  # noqa: RUF001
            ),
            clusters={
                item.cluster_id: _ClusterPresentation(
                    title=item.label,
                    question="Що повʼязує поняття цієї області?",
                    summary="Автоматично виділена локальна область корпусу.",
                    entry_concept_id=None,
                )
                for item in manifest.clusters
            },
        )

    resolved = Path(path)
    if not resolved.is_file() or resolved.stat().st_size > 256_000:
        raise ValueError("concept Lens presentation must be a bounded regular file")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("concept Lens presentation must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("concept Lens presentation must be a JSON object")
    if payload.get("schema") != "dithyramba.concept_lens_presentation/1.0":
        raise ValueError("unsupported concept Lens presentation schema")
    if payload.get("ontology_id") != manifest.ontology_id:
        raise ValueError("concept Lens presentation does not match the ontology")
    title = _presentation_text(payload, "title")
    question = _presentation_text(payload, "question")
    lead = _presentation_text(payload, "lead")
    raw_clusters = payload.get("clusters")
    if not isinstance(raw_clusters, list):
        raise ValueError("concept Lens presentation requires a cluster list")
    clusters: dict[str, _ClusterPresentation] = {}
    for raw in raw_clusters:
        if not isinstance(raw, dict):
            raise ValueError("concept Lens cluster presentation must be an object")
        cluster_id = raw.get("cluster_id")
        if not isinstance(cluster_id, str) or cluster_id in clusters:
            raise ValueError("concept Lens cluster IDs must be unique strings")
        clusters[cluster_id] = _ClusterPresentation(
            title=_presentation_text(raw, "title"),
            question=_presentation_text(raw, "question"),
            summary=_presentation_text(raw, "summary"),
            entry_concept_id=_optional_presentation_id(raw, "entry_concept_id"),
        )
    expected = {item.cluster_id for item in manifest.clusters}
    if set(clusters) != expected:
        raise ValueError("concept Lens presentation must cover every ontology cluster")
    concept_cluster = {item.concept_id: item.cluster_id for item in manifest.concepts}
    for cluster_id, item in clusters.items():
        if (
            item.entry_concept_id is not None
            and concept_cluster.get(item.entry_concept_id) != cluster_id
        ):
            raise ValueError("concept Lens entry concept must belong to its cluster")
    return _LensPresentation(
        title=title,
        question=question,
        lead=lead,
        clusters=clusters,
    )


def _presentation_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > 1_000:
        raise ValueError(f"concept Lens presentation {key} must be bounded text")
    return value.strip()


def _optional_presentation_id(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(_ID_PATTERN, value) is None:
        raise ValueError(f"concept Lens presentation {key} must be a stable identifier")
    return value


def _concept_url(concept: OntologyConcept) -> str:
    return "/?" + urlencode({"cluster": concept.cluster_id, "concept": concept.concept_id})


def _evidence_url(concept: OntologyConcept, evidence: OntologyEvidence) -> str:
    return "/?" + urlencode(
        {
            "cluster": concept.cluster_id,
            "concept": concept.concept_id,
            "evidence": evidence.evidence_id,
        }
    )


__all__ = ["ConceptLensWebConfig", "create_concept_lens_app"]
