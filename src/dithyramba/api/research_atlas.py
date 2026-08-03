"""Loopback, GET-only renderer for one immutable Research Atlas manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse, HTMLResponse, Response
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dithyramba.atlas import (
    AtlasEvidence,
    AtlasHypothesis,
    AtlasQuestion,
    AtlasSource,
    AtlasSourceAddress,
    AtlasTimelineEvent,
    AtlasTraceSpan,
    LoadedResearchAtlas,
    LoadedResearchProjection,
    ProjectionMaterial,
    ResearchAtlasManifest,
    load_research_atlas,
    load_research_projection,
)
from dithyramba.contracts import canonical_json_bytes, sha256_hex

from .config import LoopbackApiConfig, build_config
from .security import LoopbackSecurityMiddleware, new_bearer_token

_TEMPLATES = Environment(
    loader=PackageLoader("dithyramba.api", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html",), default_for_string=True),
    undefined=StrictUndefined,
    enable_async=False,
)
_STYLESHEET = Path(__file__).with_name("static") / "research_atlas.css"
_ATLAS_LIBRARY_ID = "library_00000000000040008000000000000000"
_ID_PATTERN = r"^[a-z][a-z0-9_-]{1,95}$"
_VIEW_PATTERN = r"^(overview|questions|hypotheses|timeline|materials|gaps|sources)$"
_TEXT_ARTIFACT_SUFFIXES = frozenset(
    {".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".xml", ".html", ".htm"}
)
_BINARY_ARTIFACT_MEDIA = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_MAX_ARTIFACT_BYTES = 25 * 1024 * 1024

_VIEW_LABELS = {
    "overview": "Огляд теми",
    "questions": "Питання й відповіді",
    "hypotheses": "Гіпотези",
    "timeline": "Часова лінія",
    "materials": "Матеріали памʼяті",
    "gaps": "Прогалини",
    "sources": "Корпус джерел",
}
_STATE_LABELS = {
    "confirmed": "підтверджено",
    "qualified": "із застереженням",
    "contested": "є суперечність",
    "refuted": "спростовано",
    "open": "відкрите питання",
    "established": "стійка гіпотеза",
    "working": "робоча гіпотеза",
}
_SOURCE_KIND_LABELS = {
    "letter": "лист",
    "diary": "щоденник",
    "contract": "контракт",
    "patent": "патент",
    "archival_record": "архівний запис",
    "artwork": "твір",
    "photograph": "фотографія",
    "newspaper": "преса",
    "scholarly_article": "наукова стаття",
    "book": "книга",
    "dataset": "датасет",
    "institutional_record": "інституційний запис",
    "other": "інше джерело",
}
_VOICE_LABELS = {
    "first_person": "власний голос",
    "witness": "свідчення",
    "institutional": "інституційний голос",
    "scholarly": "дослідницький голос",
    "journalistic": "журналістський голос",
    "curatorial": "кураторський голос",
    "unknown": "голос не визначено",
}
_EVIDENCE_ROLE_LABELS = {
    "supports": "підтримує",
    "qualifies": "уточнює",
    "refutes": "спростовує",
    "context": "дає контекст",
}
_TRACE_KIND_LABELS = {
    "fact": "перевірюване твердження",
    "synthesis": "синтез джерел",
    "hypothesis": "робоча гіпотеза",
    "question": "дослідницьке питання",
}
_RELATION_LABELS = {
    "supports": "підсилює",
    "qualifies": "уточнює",
    "refutes": "суперечить",
    "precedes": "передує",
    "influences": "впливає на",
    "parallels": "має паралель з",
}


@dataclass(frozen=True, slots=True)
class ResearchAtlasWebConfig:
    loopback: LoopbackApiConfig
    atlas: LoadedResearchAtlas
    projection: LoadedResearchProjection | None = None
    artifact_root: Path | None = None

    def __post_init__(self) -> None:
        root = self.artifact_root
        if root is not None and (not root.is_absolute() or not root.is_dir()):
            raise ValueError("artifact_root must be an existing absolute directory")


@dataclass(frozen=True, slots=True)
class _EvidenceView:
    evidence: AtlasEvidence
    source: AtlasSource
    chip_title: str
    chip_attribution: str


@dataclass(frozen=True, slots=True)
class _MaterialView:
    material: ProjectionMaterial
    source: AtlasSource


@dataclass(frozen=True, slots=True)
class _SourceBindingView:
    locator: str
    address: AtlasSourceAddress
    expected_sha256: str
    fragment_id: str | None
    full_url: str
    back_url: str
    label: str


@dataclass(frozen=True, slots=True)
class _TraceTextPart:
    text: str
    span_id: str | None = None
    kind: str | None = None
    evidence_ids: tuple[str, ...] = ()
    first_evidence_id: str | None = None
    is_selected: bool = False


def create_research_atlas_app(
    *,
    manifest_path: str | Path,
    allowed_origin: str,
    projection_path: str | Path | None = None,
    artifact_root: str | Path | None = None,
    test_only_allow_testserver: bool = False,
) -> FastAPI:
    """Create a frozen, script-free Atlas from one validated case manifest."""

    atlas = load_research_atlas(manifest_path)
    projection = (
        None if projection_path is None else load_research_projection(projection_path, atlas=atlas)
    )
    loopback = build_config(
        library_id=_ATLAS_LIBRARY_ID,
        data_home=atlas.manifest_path.parent,
        allowed_origin=allowed_origin,
        test_only_allow_testserver=test_only_allow_testserver,
    )
    resolved_artifact_root = (
        None if artifact_root is None else Path(artifact_root).expanduser().resolve(strict=True)
    )
    config = ResearchAtlasWebConfig(
        loopback=loopback,
        atlas=atlas,
        projection=projection,
        artifact_root=resolved_artifact_root,
    )
    app = FastAPI(
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.research_atlas_config = config
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

    @app.get("/research-atlas.css", include_in_schema=False)
    async def stylesheet() -> Response:
        return Response(
            _STYLESHEET.read_text(encoding="utf-8"),
            media_type="text/css; charset=utf-8",
        )

    @app.get("/manifest.json", include_in_schema=False)
    async def manifest_json() -> Response:
        payload = atlas.manifest.model_dump(mode="json")
        payload["manifest_hash"] = atlas.manifest_hash
        return Response(
            canonical_json_bytes(payload),
            media_type="application/json",
            headers={"ETag": f'"{atlas.manifest_hash}"'},
        )

    @app.get("/projection.json", include_in_schema=False)
    async def projection_json() -> Response:
        if projection is None:
            raise HTTPException(status_code=404, detail="not found")
        payload = projection.manifest.model_dump(mode="json")
        payload["projection_hash"] = projection.manifest_hash
        return Response(
            canonical_json_bytes(payload),
            media_type="application/json",
            headers={"ETag": f'"{projection.manifest_hash}"'},
        )

    @app.get(
        "/artifacts/{source_id}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def source_artifact(
        source_id: Annotated[str, ApiPath(pattern=_ID_PATTERN, max_length=96)],
        evidence: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        material: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        full: bool = False,
    ) -> Response:
        if evidence is not None and material is not None:
            raise HTTPException(status_code=422, detail="choose evidence or material")
        source, artifact = _authorized_artifact(config, source_id)
        selected_evidence = _authorized_artifact_evidence(
            config,
            source_id=source_id,
            evidence_id=evidence,
        )
        selected_material = _authorized_artifact_material(
            config,
            source_id=source_id,
            material_id=material,
        )
        binding = _source_binding_view(
            selected_evidence,
            selected_material,
            source_id=source_id,
        )
        suffix = artifact.suffix.casefold()
        if suffix in _BINARY_ARTIFACT_MEDIA:
            return FileResponse(
                artifact,
                media_type=_BINARY_ARTIFACT_MEDIA[suffix],
                filename=artifact.name,
                content_disposition_type="inline",
            )
        if suffix not in _TEXT_ARTIFACT_SUFFIXES:
            raise HTTPException(status_code=404, detail="not found")
        try:
            text = artifact.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=404, detail="not found") from exc
        lines = _artifact_lines(
            text,
            binding=binding,
            full=full,
        )
        template = _TEMPLATES.get_template("research_atlas_source.html")
        return HTMLResponse(
            template.render(
                atlas=config.atlas.manifest,
                source=source,
                binding=binding,
                artifact_path=source.artifact_path,
                lines=lines,
                full=full,
            )
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def atlas_view(
        view: Annotated[str, Query(pattern=_VIEW_PATTERN)] = "overview",
        selected: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        evidence: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        span: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        material: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        theme: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        period: Annotated[
            str | None,
            Query(pattern=_ID_PATTERN, max_length=96),
        ] = None,
        page: Annotated[int, Query(ge=1, le=10_000)] = 1,
    ) -> HTMLResponse:
        if view == "materials" and projection is None:
            raise HTTPException(status_code=404, detail="not found")
        template = _TEMPLATES.get_template("research_atlas.html")
        return HTMLResponse(
            template.render(
                **_template_context(
                    atlas,
                    projection=projection,
                    view=view,
                    selected=selected,
                    selected_evidence=evidence,
                    selected_span=span,
                    selected_material=material,
                    theme=theme,
                    period=period,
                    page=page,
                    artifact_root=config.artifact_root,
                )
            )
        )

    return app


def _template_context(
    loaded: LoadedResearchAtlas,
    *,
    projection: LoadedResearchProjection | None,
    view: str,
    selected: str | None,
    selected_evidence: str | None,
    selected_span: str | None,
    selected_material: str | None,
    theme: str | None,
    period: str | None,
    page: int,
    artifact_root: Path | None,
) -> dict[str, object]:
    manifest = loaded.manifest
    evidence_views = _evidence_views(manifest)
    selected_object = _selected_object(manifest, view=view, selected=selected)
    selected_trace_span = _selected_trace_span(
        selected_object,
        requested_span_id=selected_span,
    )
    explicit_evidence = evidence_views.get(selected_evidence or "")
    material_views = _material_views(projection, manifest=manifest)
    selected_material_view = material_views.get(selected_material or "")
    selected_evidence_view = None
    if selected_material_view is None:
        selected_evidence_view = (
            explicit_evidence
            or _first_span_evidence(
                selected_trace_span,
                evidence_views=evidence_views,
            )
            or (
                _first_evidence(
                    selected_object,
                    evidence_views=evidence_views,
                )
                if isinstance(selected_object, AtlasSource)
                else None
            )
        )
    evidence_by_id = {
        item.evidence_id: evidence_views[item.evidence_id] for item in manifest.evidence
    }
    hypothesis_by_id = {item.hypothesis_id: item for item in manifest.hypotheses}
    trace_parts_by_object = {
        **{
            item.question_id: _trace_text_parts(
                item.short_answer,
                item.trace_spans,
                selected_span_id=None
                if selected_trace_span is None
                else selected_trace_span.span_id,
            )
            for item in manifest.questions
        },
        **{
            item.hypothesis_id: _trace_text_parts(
                item.synthesis,
                item.trace_spans,
                selected_span_id=None
                if selected_trace_span is None
                else selected_trace_span.span_id,
            )
            for item in manifest.hypotheses
        },
    }
    projection_manifest = None if projection is None else projection.manifest
    projection_materials = tuple(material_views.values())
    active_theme = None
    active_period = None
    if projection_manifest is not None:
        active_theme = next(
            (item for item in projection_manifest.themes if item.theme_id == theme),
            None,
        )
        active_period = next(
            (item for item in projection_manifest.periods if item.period_id == period),
            None,
        )
        if theme is not None and active_theme is None:
            raise HTTPException(status_code=404, detail="not found")
        if period is not None and active_period is None:
            raise HTTPException(status_code=404, detail="not found")
        if active_theme is not None:
            projection_materials = tuple(
                item
                for item in projection_materials
                if active_theme.theme_id in item.material.theme_ids
            )
        if active_period is not None:
            projection_materials = tuple(
                item
                for item in projection_materials
                if item.material.period_id == active_period.period_id
            )

    page_size = 30
    page_count = max(1, (len(projection_materials) + page_size - 1) // page_size)
    source_page_count = max(1, (len(manifest.sources) + page_size - 1) // page_size)
    active_page_count = source_page_count if view == "sources" else page_count
    if page > active_page_count:
        raise HTTPException(status_code=404, detail="not found")
    page_start = (page - 1) * page_size
    paged_materials = projection_materials[page_start : page_start + page_size]
    paged_sources = manifest.sources[page_start : page_start + page_size]
    theme_metrics = _theme_metrics(projection, manifest=manifest)
    hypothesis_projection_metrics = _hypothesis_projection_metrics(
        projection,
        theme_metrics=theme_metrics,
    )
    metrics = {
        "source_count": len(manifest.sources),
        "family_count": len({item.independence_group for item in manifest.sources}),
        "verified_source_count": sum(
            item.verification_state.value == "verified" for item in manifest.sources
        ),
        "answered_question_count": sum(item.state.value != "open" for item in manifest.questions),
        "question_count": len(manifest.questions),
        "hypothesis_count": len(manifest.hypotheses),
        "timeline_count": len(manifest.timeline),
        "gap_count": len(manifest.gaps),
        "material_count": len(material_views),
        "review_priority_count": sum(
            item.material.review_priority for item in material_views.values()
        ),
    }
    view_labels = dict(_VIEW_LABELS)
    if projection is None:
        view_labels.pop("materials")
    return {
        "atlas": manifest,
        "manifest_hash": loaded.manifest_hash,
        "view": view,
        "view_labels": view_labels,
        "state_labels": _STATE_LABELS,
        "source_kind_labels": _SOURCE_KIND_LABELS,
        "voice_labels": _VOICE_LABELS,
        "evidence_role_labels": _EVIDENCE_ROLE_LABELS,
        "trace_kind_labels": _TRACE_KIND_LABELS,
        "relation_labels": _RELATION_LABELS,
        "evidence_by_id": evidence_by_id,
        "hypothesis_by_id": hypothesis_by_id,
        "selected": selected_object,
        "selected_evidence": selected_evidence_view,
        "selected_span": selected_trace_span,
        "active_span_evidence_ids": (
            () if selected_trace_span is None else selected_trace_span.evidence_ids
        ),
        "trace_parts_by_object": trace_parts_by_object,
        "selected_material": selected_material_view,
        "metrics": metrics,
        "projection": projection_manifest,
        "projection_hash": None if projection is None else projection.manifest_hash,
        "theme_metrics": theme_metrics,
        "hypothesis_projection_metrics": hypothesis_projection_metrics,
        "active_theme": active_theme,
        "active_period": active_period,
        "materials": paged_materials,
        "material_page": page,
        "material_page_count": page_count,
        "material_total": len(projection_materials),
        "sources": paged_sources,
        "source_page": page,
        "source_page_count": source_page_count,
        "source_total": len(manifest.sources),
        "artifact_links_enabled": artifact_root is not None,
    }


def _material_views(
    projection: LoadedResearchProjection | None,
    *,
    manifest: ResearchAtlasManifest,
) -> dict[str, _MaterialView]:
    if projection is None:
        return {}
    sources = {item.source_id: item for item in manifest.sources}
    return {
        item.material_id: _MaterialView(
            material=item,
            source=sources[item.source_id],
        )
        for item in projection.manifest.materials
    }


def _theme_metrics(
    projection: LoadedResearchProjection | None,
    *,
    manifest: ResearchAtlasManifest,
) -> dict[str, dict[str, int]]:
    if projection is None:
        return {}
    evidence_by_id = {item.evidence_id: item for item in manifest.evidence}
    source_by_id = {item.source_id: item for item in manifest.sources}
    hypothesis_by_id = {item.hypothesis_id: item for item in manifest.hypotheses}
    metrics: dict[str, dict[str, int]] = {}
    for theme in projection.manifest.themes:
        materials = [
            item for item in projection.manifest.materials if theme.theme_id in item.theme_ids
        ]
        evidence_ids = {
            evidence_id
            for hypothesis_id in theme.hypothesis_ids
            for evidence_id in hypothesis_by_id[hypothesis_id].evidence_ids
        }
        source_families = {
            source_by_id[evidence_by_id[evidence_id].source_id].independence_group
            for evidence_id in evidence_ids
        }
        counterevidence_count = sum(
            len(hypothesis_by_id[hypothesis_id].counterevidence_ids)
            for hypothesis_id in theme.hypothesis_ids
        )
        metrics[theme.theme_id] = {
            "evidence_count": len(evidence_ids),
            "family_count": len(source_families),
            "material_count": len(materials),
            "review_priority_count": sum(item.review_priority for item in materials),
            "counterevidence_count": counterevidence_count,
            "gap_count": len(theme.gap_ids),
        }
    return metrics


def _hypothesis_projection_metrics(
    projection: LoadedResearchProjection | None,
    *,
    theme_metrics: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    if projection is None:
        return {}
    result: dict[str, dict[str, int]] = {}
    for theme in projection.manifest.themes:
        for hypothesis_id in theme.hypothesis_ids:
            current = result.setdefault(
                hypothesis_id,
                {"material_count": 0, "review_priority_count": 0, "theme_count": 0},
            )
            current["material_count"] += theme_metrics[theme.theme_id]["material_count"]
            current["review_priority_count"] += theme_metrics[theme.theme_id][
                "review_priority_count"
            ]
            current["theme_count"] += 1
    return result


def _evidence_views(manifest: ResearchAtlasManifest) -> dict[str, _EvidenceView]:
    sources = {item.source_id: item for item in manifest.sources}
    return {
        item.evidence_id: _EvidenceView(
            evidence=item,
            source=sources[item.source_id],
            chip_title=sources[item.source_id].title,
            chip_attribution=_chip_attribution(sources[item.source_id]),
        )
        for item in manifest.evidence
    }


def _chip_attribution(source: AtlasSource) -> str:
    parts = [source.creator]
    if source.date_label and source.date_label.casefold() not in source.title.casefold():
        parts.append(source.date_label)
    return " · ".join(parts)


def _selected_object(
    manifest: ResearchAtlasManifest,
    *,
    view: str,
    selected: str | None,
) -> AtlasQuestion | AtlasHypothesis | AtlasTimelineEvent | AtlasSource | None:
    collections: dict[str, tuple[object, ...]] = {
        "overview": manifest.questions,
        "questions": manifest.questions,
        "hypotheses": manifest.hypotheses,
        "timeline": manifest.timeline,
        "materials": (),
        "sources": manifest.sources,
        "gaps": (),
    }
    values = collections[view]
    if selected is None:
        return None
    for item in values:
        identifier = _object_id(item)
        if selected is not None and identifier == selected:
            return item  # type: ignore[return-value]
    if values:
        return values[0]  # type: ignore[return-value]
    return None


def _object_id(value: object) -> str | None:
    for attribute in ("question_id", "hypothesis_id", "event_id", "source_id"):
        identifier = getattr(value, attribute, None)
        if isinstance(identifier, str):
            return identifier
    return None


def _first_evidence(
    selected: AtlasQuestion | AtlasHypothesis | AtlasTimelineEvent | AtlasSource | None,
    *,
    evidence_views: dict[str, _EvidenceView],
) -> _EvidenceView | None:
    if selected is None:
        return None
    evidence_ids = getattr(selected, "evidence_ids", ())
    if evidence_ids:
        return evidence_views.get(evidence_ids[0])
    source_id = getattr(selected, "source_id", None)
    if source_id is not None:
        return next(
            (item for item in evidence_views.values() if item.source.source_id == source_id),
            None,
        )
    return None


def _selected_trace_span(
    selected: AtlasQuestion | AtlasHypothesis | AtlasTimelineEvent | AtlasSource | None,
    *,
    requested_span_id: str | None,
) -> AtlasTraceSpan | None:
    if requested_span_id is None:
        return None
    spans = selected.trace_spans if isinstance(selected, (AtlasQuestion, AtlasHypothesis)) else ()
    for span in spans:
        if span.span_id == requested_span_id:
            return span
    raise HTTPException(status_code=404, detail="not found")


def _first_span_evidence(
    span: AtlasTraceSpan | None,
    *,
    evidence_views: dict[str, _EvidenceView],
) -> _EvidenceView | None:
    if span is None:
        return None
    return evidence_views.get(span.evidence_ids[0])


def _trace_text_parts(
    text: str,
    spans: tuple[AtlasTraceSpan, ...],
    *,
    selected_span_id: str | None,
) -> tuple[_TraceTextPart, ...]:
    if not spans:
        return (_TraceTextPart(text=text),)
    parts: list[_TraceTextPart] = []
    cursor = 0
    for span in spans:
        if cursor < span.start:
            parts.append(_TraceTextPart(text=text[cursor : span.start]))
        parts.append(
            _TraceTextPart(
                text=span.text,
                span_id=span.span_id,
                kind=span.kind.value,
                evidence_ids=span.evidence_ids,
                first_evidence_id=span.evidence_ids[0],
                is_selected=span.span_id == selected_span_id,
            )
        )
        cursor = span.end
    if cursor < len(text):
        parts.append(_TraceTextPart(text=text[cursor:]))
    return tuple(parts)


def _authorized_artifact(
    config: ResearchAtlasWebConfig,
    source_id: str,
) -> tuple[AtlasSource, Path]:
    root = config.artifact_root
    source = next(
        (item for item in config.atlas.manifest.sources if item.source_id == source_id),
        None,
    )
    if root is None or source is None or source.artifact_path is None:
        raise HTTPException(status_code=404, detail="not found")
    artifact = (root / source.artifact_path).resolve(strict=False)
    if not artifact.is_relative_to(root) or not artifact.is_file():
        raise HTTPException(status_code=404, detail="not found")
    if artifact.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise HTTPException(status_code=413, detail="artifact is too large for the local viewer")
    return source, artifact


def _authorized_artifact_evidence(
    config: ResearchAtlasWebConfig,
    *,
    source_id: str,
    evidence_id: str | None,
) -> AtlasEvidence | None:
    if evidence_id is None:
        return None
    evidence = next(
        (
            item
            for item in config.atlas.manifest.evidence
            if item.evidence_id == evidence_id and item.source_id == source_id
        ),
        None,
    )
    if evidence is None:
        raise HTTPException(status_code=404, detail="not found")
    return evidence


def _authorized_artifact_material(
    config: ResearchAtlasWebConfig,
    *,
    source_id: str,
    material_id: str | None,
) -> ProjectionMaterial | None:
    if material_id is None:
        return None
    projection = config.projection
    if projection is None:
        raise HTTPException(status_code=404, detail="not found")
    material = next(
        (
            item
            for item in projection.manifest.materials
            if item.material_id == material_id and item.source_id == source_id
        ),
        None,
    )
    if material is None:
        raise HTTPException(status_code=404, detail="not found")
    return material


def _source_binding_view(
    evidence: AtlasEvidence | None,
    material: ProjectionMaterial | None,
    *,
    source_id: str,
) -> _SourceBindingView | None:
    if evidence is not None:
        if evidence.source_address is None or evidence.fragment_text_sha256 is None:
            return None
        return _SourceBindingView(
            locator=evidence.locator,
            address=evidence.source_address,
            expected_sha256=evidence.fragment_text_sha256,
            fragment_id=evidence.source_fragment_id,
            full_url=(
                f"/artifacts/{source_id}?evidence={evidence.evidence_id}&full=true"
                f"#L{evidence.source_address.line_start}"
            ),
            back_url=f"/?view=sources&selected={source_id}",
            label="Точний доказ",
        )
    if material is not None:
        return _SourceBindingView(
            locator=material.locator,
            address=material.source_address,
            expected_sha256=material.fragment_text_sha256,
            fragment_id=material.material_id,
            full_url=(
                f"/artifacts/{source_id}?material={material.material_id}&full=true"
                f"#L{material.source_address.line_start}"
            ),
            back_url=f"/?view=materials&material={material.material_id}",
            label="Матеріал-кандидат",
        )
    return None


def _artifact_lines(
    text: str,
    *,
    binding: _SourceBindingView | None,
    full: bool,
) -> tuple[dict[str, object], ...]:
    source_lines = text.splitlines()
    selected_start: int | None = None
    selected_end: int | None = None
    if binding is not None:
        selected_start = binding.address.line_start
        selected_end = binding.address.line_end
        if selected_end > len(source_lines):
            raise HTTPException(status_code=409, detail="bound source fragment is out of range")
        selected_text = "\n".join(source_lines[selected_start - 1 : selected_end])
        if sha256_hex(selected_text.encode("utf-8")) != binding.expected_sha256:
            raise HTTPException(
                status_code=409,
                detail="bound source fragment no longer matches the manifest",
            )
    if full or selected_start is None or selected_end is None:
        first = 1
        last = len(source_lines)
    else:
        first = max(1, selected_start - 8)
        last = min(len(source_lines), selected_end + 8)
    return tuple(
        {
            "number": number,
            "text": source_lines[number - 1],
            "selected": (
                selected_start is not None
                and selected_end is not None
                and selected_start <= number <= selected_end
            ),
        }
        for number in range(first, last + 1)
    )
