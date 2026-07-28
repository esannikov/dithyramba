"""Secure-by-default FastAPI factory for one loopback Dithyramba Library."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

from fastapi import FastAPI, Query, Request, status
from fastapi import Path as ApiPath
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dithyramba.collections import CollectionError
from dithyramba.contracts import canonical_json_bytes
from dithyramba.ingest.service import IngestService
from dithyramba.persistence.errors import (
    AccessPolicyNotFoundError,
    CollectionNotFoundError,
    CorpusSnapshotNotFoundError,
    EvidencePacketNotFoundError,
    PersistenceConflictError,
    SourceNotFoundError,
    SourceVersionNotFoundError,
)
from dithyramba.persistence.recall import SQLiteRecallBackend
from dithyramba.persistence.repository import LibraryRepository, open_library
from dithyramba.recall import (
    QueryContractError,
    RecallDependencyMismatchError,
    RecallReplayMismatchError,
    RecallRequestError,
    RecallService,
    RecallSnapshotError,
    current_fts_runtime_profile,
)
from dithyramba.review import (
    ReviewConflictError,
    ReviewDecision,
    ReviewDecisionNotFoundError,
    ReviewIntegrityError,
    ReviewTargetNotFoundError,
    ReviewTargetType,
)
from dithyramba.review.service import ReviewService

from .config import LoopbackApiConfig, build_config
from .models import (
    CollectionCreateRequest,
    CollectionListResponse,
    CollectionResponse,
    CorpusSnapshotResponse,
    ErrorResponse,
    EvidenceFragmentResponse,
    EvidencePacketResponse,
    HealthResponse,
    IngestResultResponse,
    LibraryListResponse,
    LibraryResponse,
    PacketBackedSourceFragmentResponse,
    PacketExportRequest,
    PacketPresentationStatusResponse,
    ReadReceiptResponse,
    RecallRequest,
    RecallResultResponse,
    ReviewDecisionCreateRequest,
    ReviewDecisionListResponse,
    ReviewDecisionResponse,
    SourceChipListResponse,
    SourceIngestRequest,
    SourceResponse,
    SourceVersionListResponse,
    SourceVersionResponse,
)
from .security import LoopbackSecurityMiddleware, new_bearer_token
from .services import PacketProjection, PacketViewService, ViewerNotFoundError

_ID_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$"
_LIBRARY_ID_PATTERN = r"^library_[0-9a-f]{32}$"
_COLLECTION_ID_PATTERN = r"^collection_[0-9a-f]{32}$"
_SOURCE_ID_PATTERN = r"^source_[a-z0-9]+(?:_[a-z0-9]+)*$"
_SOURCE_VERSION_ID_PATTERN = r"^source_version_[a-z0-9]+(?:_[a-z0-9]+)*$"
_FRAGMENT_ID_PATTERN = r"^fragment_[a-z0-9]+(?:_[a-z0-9]+)*$"
_PACKET_ID_PATTERN = r"^packet_[a-z0-9]+(?:_[a-z0-9]+)*$"
_REVIEW_ID_PATTERN = r"^review_[a-z0-9]+(?:_[a-z0-9]+)*$"
_TEMPLATES = Environment(
    loader=PackageLoader("dithyramba.api", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html",), default_for_string=True),
    undefined=StrictUndefined,
    enable_async=False,
)


@dataclass(slots=True)
class _Services:
    repository: LibraryRepository
    recall_backend: SQLiteRecallBackend
    recall: RecallService
    ingest: IngestService
    reviews: ReviewService
    packets: PacketViewService


@dataclass(slots=True)
class _Runtime:
    config: LoopbackApiConfig
    bearer_token: str
    services: _Services | None = None

    def require_services(self) -> _Services:
        if self.services is None:
            raise RuntimeError("loopback API lifespan is not active")
        return self.services


def create_app(
    *,
    library_id: str,
    data_home: str | Path,
    allowed_origin: str,
    test_only_allow_testserver: bool = False,
    max_request_body_bytes: int = 64 * 1024,
) -> FastAPI:
    """Create one non-debug app pinned to an explicit physical Library."""

    config = build_config(
        library_id=library_id,
        data_home=data_home,
        allowed_origin=allowed_origin,
        test_only_allow_testserver=test_only_allow_testserver,
        max_request_body_bytes=max_request_body_bytes,
    )
    runtime = _Runtime(config=config, bearer_token=new_bearer_token())

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        with open_library(config.library_id, data_root=config.data_home) as repository:
            recall_backend = SQLiteRecallBackend(repository)
            runtime.services = _Services(
                repository=repository,
                recall_backend=recall_backend,
                recall=RecallService(
                    recall_backend,
                    profile_version=current_fts_runtime_profile().profile_version,
                ),
                ingest=IngestService(repository),
                reviews=ReviewService(repository),
                packets=PacketViewService(repository, recall_backend),
            )
            try:
                yield
            finally:
                runtime.services = None

    app = FastAPI(
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.loopback_bearer_token = runtime.bearer_token
    app.state.loopback_config = config
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(config.trusted_hosts),
        www_redirect=False,
    )
    app.add_middleware(
        LoopbackSecurityMiddleware,
        bearer_token=runtime.bearer_token,
        allowed_origin=config.allowed_origin,
        trusted_hosts=config.trusted_hosts,
        max_request_body_bytes=config.max_request_body_bytes,
    )
    _install_error_handlers(app)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        repository = runtime.require_services().repository
        repository.verify()
        return HealthResponse(
            library_id=repository.library_id,
            schema_version=repository.schema_version,
        )

    @app.get("/v1/libraries", response_model=LibraryListResponse)
    async def list_libraries() -> LibraryListResponse:
        """Return only the Library pinned into this process, never scan data_home."""

        record = runtime.require_services().repository.library
        return LibraryListResponse(libraries=(LibraryResponse.from_record(record),))

    @app.post(
        "/v1/libraries/{library_id}/collections",
        response_model=CollectionResponse,
        status_code=status.HTTP_201_CREATED,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    async def create_collection(
        library_id: Annotated[str, ApiPath(pattern=_LIBRARY_ID_PATTERN)],
        request: CollectionCreateRequest,
    ) -> CollectionResponse:
        _require_configured_library(config, library_id)
        try:
            collection = request.to_domain(library_id=library_id)
        except (CollectionError, TypeError, ValueError):
            raise StarletteHTTPException(status_code=422) from None
        record = runtime.require_services().repository.create_collection(collection)
        return CollectionResponse.from_record(record)

    @app.get(
        "/v1/libraries/{library_id}/collections",
        response_model=CollectionListResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def list_collections(
        library_id: Annotated[str, ApiPath(pattern=_LIBRARY_ID_PATTERN)],
    ) -> CollectionListResponse:
        _require_configured_library(config, library_id)
        records = runtime.require_services().repository.list_collections()
        return CollectionListResponse(
            collections=tuple(CollectionResponse.from_record(item) for item in records),
            library_id=library_id,
        )

    @app.post(
        "/v1/libraries/{library_id}/collections/{collection_id}/snapshots",
        response_model=CorpusSnapshotResponse,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
        },
    )
    async def create_snapshot(
        library_id: Annotated[str, ApiPath(pattern=_LIBRARY_ID_PATTERN)],
        collection_id: Annotated[str, ApiPath(pattern=_COLLECTION_ID_PATTERN)],
    ) -> CorpusSnapshotResponse:
        _require_configured_library(config, library_id)
        repository = runtime.require_services().repository
        repository.get_collection(collection_id)
        return CorpusSnapshotResponse.from_snapshot(repository.freeze_snapshot((collection_id,)))

    @app.post(
        "/v1/sources",
        response_model=IngestResultResponse,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    async def ingest_source(request: SourceIngestRequest) -> IngestResultResponse:
        _require_configured_library(config, request.library_id)
        services = runtime.require_services()
        collection = services.repository.get_collection(request.collection_id)
        _validate_collection_root(collection.collection_root_ids, request.collection_root_id)
        result = services.ingest.ingest_path(
            request.collection_id,
            request.relative_path,
            collection_root_id=request.collection_root_id,
        )
        return IngestResultResponse.from_result(
            result,
            library_id=config.library_id,
            collection_id=request.collection_id,
        )

    @app.get(
        "/v1/sources/{source_id}/versions",
        response_model=SourceVersionListResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def source_versions(
        source_id: Annotated[str, ApiPath(pattern=_SOURCE_ID_PATTERN)],
    ) -> SourceVersionListResponse:
        repository = runtime.require_services().repository
        source = repository.get_source(source_id)
        return SourceVersionListResponse(
            library_id=config.library_id,
            source=SourceResponse.from_record(source),
            versions=tuple(
                SourceVersionResponse.from_record(record)
                for record in repository.list_source_versions(source_id)
            ),
        )

    @app.get(
        "/v1/source-fragments/{source_fragment_id}",
        response_model=PacketBackedSourceFragmentResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def source_fragment(
        source_fragment_id: Annotated[str, ApiPath(pattern=_FRAGMENT_ID_PATTERN)],
        packet: Annotated[str, Query(pattern=_PACKET_ID_PATTERN)],
    ) -> PacketBackedSourceFragmentResponse:
        projection = runtime.require_services().packets.load_fragment(
            source_fragment_id,
            evidence_packet_id=packet,
        )
        return PacketBackedSourceFragmentResponse(
            evidence_packet_id=projection.packet.evidence_packet_id,
            packet_hash=projection.packet.packet_hash,
            source_id=projection.source_chip.source_id,
            fragment=EvidenceFragmentResponse.from_fragment(projection.fragment),
            source_chip=projection.source_chip,
        )

    @app.post(
        "/v1/recall",
        response_model=RecallResultResponse,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    async def recall(request: RecallRequest) -> RecallResultResponse:
        _require_configured_library(config, request.library_id)
        try:
            domain_request = request.to_domain()
        except (TypeError, ValueError):
            raise StarletteHTTPException(status_code=422) from None
        services = runtime.require_services()
        for collection_id in domain_request.collection_ids:
            services.repository.get_collection(collection_id)
        services.repository.get_corpus_snapshot(domain_request.corpus_snapshot_id)
        services.repository.get_access_policy(domain_request.access_policy_id)
        return RecallResultResponse.from_result(services.recall.recall(domain_request))

    @app.get(
        "/v1/evidence-packets/{evidence_packet_id}",
        response_model=EvidencePacketResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def evidence_packet(
        evidence_packet_id: Annotated[str, ApiPath(pattern=_PACKET_ID_PATTERN)],
    ) -> EvidencePacketResponse:
        packet = runtime.require_services().packets.load_packet(evidence_packet_id).packet
        return EvidencePacketResponse.from_packet(packet)

    @app.get(
        "/v1/evidence-packets/{evidence_packet_id}/source-chips",
        response_model=SourceChipListResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def packet_source_chips(
        evidence_packet_id: Annotated[str, ApiPath(pattern=_PACKET_ID_PATTERN)],
    ) -> SourceChipListResponse:
        projection = runtime.require_services().packets.load_packet(evidence_packet_id)
        return SourceChipListResponse(
            evidence_packet_id=projection.packet.evidence_packet_id,
            packet_hash=projection.packet.packet_hash,
            items=projection.source_chips,
        )

    @app.get(
        "/v1/evidence-packets/{evidence_packet_id}/read-receipt",
        response_model=ReadReceiptResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def packet_read_receipt(
        evidence_packet_id: Annotated[str, ApiPath(pattern=_PACKET_ID_PATTERN)],
    ) -> ReadReceiptResponse:
        packet = runtime.require_services().packets.load_packet(evidence_packet_id).packet
        return ReadReceiptResponse.from_receipt(packet.read_receipt)

    @app.post(
        "/v1/evidence-packets/{evidence_packet_id}/replay",
        response_model=RecallResultResponse,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    async def replay_packet(
        evidence_packet_id: Annotated[str, ApiPath(pattern=_PACKET_ID_PATTERN)],
    ) -> RecallResultResponse:
        services = runtime.require_services()
        services.packets.load_packet(evidence_packet_id)
        return RecallResultResponse.from_result(services.recall.replay(evidence_packet_id))

    @app.post(
        "/v1/evidence-packets/{evidence_packet_id}/exports",
        response_model=None,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
        },
    )
    async def export_packet(
        evidence_packet_id: Annotated[str, ApiPath(pattern=_PACKET_ID_PATTERN)],
        request: PacketExportRequest,
    ) -> Response:
        services = runtime.require_services()
        projection = services.packets.load_packet(evidence_packet_id)
        access_receipt = projection.packet.access_receipt
        policy = services.repository.get_access_policy(access_receipt.access_policy_id)
        if policy.snapshot.policy_hash != access_receipt.policy_hash:
            raise RuntimeError("packet export policy hash is inconsistent")
        if not policy.snapshot.allow_export:
            raise StarletteHTTPException(status_code=403)
        decisions = _packet_review_decisions(services, projection)
        presentation = _packet_presentation(decisions)
        extension = "json" if request.format == "json" else "md"
        headers = {
            "Content-Disposition": (
                f'attachment; filename="evidence-packet-{evidence_packet_id}.{extension}"'
            )
        }
        if request.format == "json":
            content = canonical_json_bytes(
                _packet_export_payload(
                    projection,
                    decisions,
                    presentation,
                    library_id=config.library_id,
                    access_policy_id=policy.snapshot.access_policy_id,
                    policy_hash=policy.snapshot.policy_hash,
                )
            )
            return Response(
                content=content,
                media_type="application/json",
                headers=headers,
            )
        return Response(
            content=_packet_markdown(projection, decisions, presentation),
            media_type="text/markdown",
            headers=headers,
        )

    @app.get(
        "/v1/review-decisions",
        response_model=ReviewDecisionListResponse,
    )
    async def list_review_decisions(
        target_type: ReviewTargetType | None = None,
        target_id: Annotated[str | None, Query(pattern=_ID_PATTERN)] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> ReviewDecisionListResponse:
        if (target_type is None) != (target_id is None):
            raise StarletteHTTPException(status_code=422)
        decisions = runtime.require_services().reviews.list(
            target_type=target_type,
            target_id=target_id,
            limit=limit,
        )
        return ReviewDecisionListResponse(
            library_id=config.library_id,
            items=tuple(ReviewDecisionResponse.from_decision(item) for item in decisions),
        )

    @app.get(
        "/v1/review-decisions/{review_decision_id}",
        response_model=ReviewDecisionResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def get_review_decision(
        review_decision_id: Annotated[str, ApiPath(pattern=_REVIEW_ID_PATTERN)],
    ) -> ReviewDecisionResponse:
        decision = runtime.require_services().reviews.get(review_decision_id)
        return ReviewDecisionResponse.from_decision(decision)

    @app.post(
        "/v1/review-decisions",
        response_model=ReviewDecisionResponse,
        status_code=status.HTTP_201_CREATED,
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
    )
    async def create_review_decision(
        request: ReviewDecisionCreateRequest,
    ) -> ReviewDecisionResponse:
        try:
            domain_request = request.to_domain()
        except (TypeError, ValueError):
            raise StarletteHTTPException(status_code=422) from None
        try:
            decision = runtime.require_services().reviews.decide(domain_request)
        except ReviewIntegrityError:
            raise StarletteHTTPException(status_code=409) from None
        return ReviewDecisionResponse.from_decision(decision)

    @app.get(
        "/libraries/{library_id}",
        response_class=HTMLResponse,
        response_model=None,
        responses={404: {"model": ErrorResponse}},
    )
    async def library_view(
        library_id: Annotated[str, ApiPath(pattern=_LIBRARY_ID_PATTERN)],
    ) -> HTMLResponse:
        _require_configured_library(config, library_id)
        services = runtime.require_services()
        template = _TEMPLATES.get_template("library_view.html")
        return HTMLResponse(
            template.render(
                library=LibraryResponse.from_record(services.repository.library),
                collections=tuple(
                    CollectionResponse.from_record(record)
                    for record in services.repository.list_collections()
                ),
            )
        )

    @app.get(
        "/evidence-packets/{evidence_packet_id}",
        response_class=HTMLResponse,
        response_model=None,
        responses={404: {"model": ErrorResponse}},
    )
    async def packet_view(
        evidence_packet_id: Annotated[str, ApiPath(pattern=_PACKET_ID_PATTERN)],
    ) -> HTMLResponse:
        services = runtime.require_services()
        projection = services.packets.load_packet(evidence_packet_id)
        decisions = _packet_review_decisions(services, projection)
        template = _TEMPLATES.get_template("packet_viewer.html")
        return HTMLResponse(
            template.render(
                packet=projection.packet,
                status=_packet_presentation(decisions),
                decisions=decisions,
                items=tuple(
                    {"fragment": fragment, "chip": chip}
                    for fragment, chip in zip(
                        projection.packet.source_fragments,
                        projection.source_chips,
                        strict=True,
                    )
                ),
            )
        )

    @app.get(
        "/sources/{source_id}/versions/{source_version_id}",
        response_class=HTMLResponse,
        response_model=None,
        responses={404: {"model": ErrorResponse}},
    )
    async def source_viewer(
        source_id: Annotated[str, ApiPath(pattern=_SOURCE_ID_PATTERN)],
        source_version_id: Annotated[
            str,
            ApiPath(pattern=_SOURCE_VERSION_ID_PATTERN),
        ],
        packet: Annotated[str, Query(pattern=_PACKET_ID_PATTERN)],
        fragment: Annotated[str, Query(pattern=_FRAGMENT_ID_PATTERN)],
    ) -> HTMLResponse:
        projection = runtime.require_services().packets.load_viewer(
            source_id=source_id,
            source_version_id=source_version_id,
            source_fragment_id=fragment,
            evidence_packet_id=packet,
        )
        template = _TEMPLATES.get_template("source_viewer.html")
        rendered = template.render(
            packet_id=projection.packet.evidence_packet_id,
            packet_hash=projection.packet.packet_hash,
            source_id=projection.source_chip.source_id,
            source_family_id=projection.source_chip.source_family_id,
            root_source_id=projection.source_chip.root_source_id,
            family_role=projection.source_chip.family_role,
            source_version_id=projection.source_chip.source_version_id,
            source_fragment_id=projection.source_chip.source_fragment_id,
            text=projection.fragment.text,
            text_sha256=projection.fragment.text_sha256,
            source_address=projection.source_address_json,
        )
        return HTMLResponse(rendered)

    return app


def bearer_token_for(app: FastAPI) -> str:
    """Return the per-instance token to the trusted local server launcher."""

    token = cast(object, app.state.loopback_bearer_token)
    if type(token) is not str:
        raise RuntimeError("loopback bearer token is unavailable")
    return token


def _require_configured_library(config: LoopbackApiConfig, library_id: str) -> None:
    if library_id != config.library_id:
        raise StarletteHTTPException(status_code=404)


def _validate_collection_root(
    root_ids: tuple[str, ...],
    requested_root_id: str | None,
) -> None:
    if not root_ids:
        raise StarletteHTTPException(status_code=422)
    if requested_root_id is None and len(root_ids) != 1:
        raise StarletteHTTPException(status_code=422)
    if requested_root_id is not None and requested_root_id not in root_ids:
        raise StarletteHTTPException(status_code=422)


def _packet_presentation(
    decisions: tuple[ReviewDecision, ...],
) -> PacketPresentationStatusResponse:
    return PacketPresentationStatusResponse(
        review_status="has_decisions" if decisions else "unreviewed",
        review_decision_ids=tuple(item.review_decision_id for item in decisions),
    )


def _packet_review_decisions(
    services: _Services,
    projection: PacketProjection,
) -> tuple[ReviewDecision, ...]:
    """Collect exact scoped decisions without collapsing them into global truth."""

    decisions: dict[str, ReviewDecision] = {}

    def add(items: tuple[ReviewDecision, ...]) -> None:
        for item in items:
            decisions[item.review_decision_id] = item

    packet_id = projection.packet.evidence_packet_id
    add(
        services.reviews.list(
            target_type=ReviewTargetType.EVIDENCE_PACKET,
            target_id=packet_id,
            limit=500,
        )
    )
    for target in services.reviews.packet_item_targets(packet_id):
        add(
            services.reviews.list(
                target_type=ReviewTargetType.PACKET_ITEM,
                target_id=target.target_id,
                limit=500,
            )
        )
    for fragment in projection.packet.source_fragments:
        add(
            services.reviews.list(
                target_type=ReviewTargetType.SOURCE_FRAGMENT,
                target_id=fragment.source_fragment_id,
                limit=500,
            )
        )
    return tuple(
        sorted(
            decisions.values(),
            key=lambda item: (item.created_at, item.review_decision_id),
        )
    )


def _packet_export_payload(
    projection: PacketProjection,
    decisions: tuple[ReviewDecision, ...],
    presentation: PacketPresentationStatusResponse,
    *,
    library_id: str,
    access_policy_id: str,
    policy_hash: str,
) -> dict[str, object]:
    return {
        "schema": "dithyramba.evidence_packet_export/1.0",
        "library_id": library_id,
        "presentation": presentation.model_dump(mode="json", by_alias=True),
        "export_authorization": {
            "access_policy_id": access_policy_id,
            "policy_hash": policy_hash,
            "allow_export": True,
        },
        "evidence_packet": projection.packet.payload(),
        "source_chips": [
            chip.model_dump(mode="json", by_alias=True) for chip in projection.source_chips
        ],
        "review_decisions": [decision.payload() for decision in decisions],
    }


def _packet_markdown(
    projection: PacketProjection,
    decisions: tuple[ReviewDecision, ...],
    presentation: PacketPresentationStatusResponse,
) -> str:
    packet = projection.packet
    lines = [
        f"# EvidencePacket {packet.evidence_packet_id}",
        "",
        f"- Packet hash: `{packet.packet_hash}`",
        f"- Result status: `{packet.result_status.value}`",
        f"- Candidate status: `{presentation.candidate_status}`",
        f"- Review status: `{presentation.review_status}`",
        "- Review semantics: decisions remain scoped by authority, use, and Collections.",
        "",
        "## Review decisions",
        "",
    ]
    if decisions:
        for decision in decisions:
            collections = ", ".join(decision.scope.collection_ids)
            lines.extend(
                (
                    f"- `{decision.review_decision_id}` — `{decision.action.value}`; "
                    f"authority `{decision.authority}`; use `{decision.scope.use}`; "
                    f"Collections `{collections}`",
                )
            )
    else:
        lines.append("- None recorded.")
    lines.extend(("", "## Evidence", ""))
    if not packet.source_fragments:
        lines.append("No evidence fragments were selected.")
    for fragment, chip in zip(
        packet.source_fragments,
        projection.source_chips,
        strict=True,
    ):
        address = canonical_json_bytes(fragment.source_address.payload()).decode("utf-8")
        lines.extend(
            (
                f"### {fragment.rank}. {fragment.source_fragment_id}",
                "",
                f"- SourceChip: `{chip.viewer_url}`",
                f"- Source: `{chip.source_id}`",
                f"- SourceVersion: `{chip.source_version_id}`",
                f"- SourceFamily: `{chip.source_family_id}`",
                f"- Root Source: `{chip.root_source_id}`",
                f"- Family role: `{chip.family_role}`",
                f"- Text SHA-256: `{fragment.text_sha256}`",
                f"- SourceAddress: `{address}`",
                "",
            )
        )
        lines.extend(f"    {line}" for line in fragment.text.split("\n"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _install_error_handlers(app: FastAPI) -> None:
    async def validation_error(
        _request: Request,
        _error: Exception,
    ) -> JSONResponse:
        return _error_response(422, "invalid request")

    async def not_found(_request: Request, _error: Exception) -> JSONResponse:
        return _error_response(404, "not found")

    async def conflict(_request: Request, _error: Exception) -> JSONResponse:
        return _error_response(409, "conflict")

    async def invalid(_request: Request, _error: Exception) -> JSONResponse:
        return _error_response(422, "invalid request")

    async def http_error(_request: Request, error: Exception) -> JSONResponse:
        status_code = cast(StarletteHTTPException, error).status_code
        labels = {
            400: "invalid request",
            404: "not found",
            405: "method not allowed",
            409: "conflict",
            413: "request body too large",
            422: "invalid request",
        }
        return _error_response(status_code, labels.get(status_code, "request rejected"))

    app.add_exception_handler(RequestValidationError, validation_error)
    for not_found_type in (
        AccessPolicyNotFoundError,
        CollectionNotFoundError,
        CorpusSnapshotNotFoundError,
        EvidencePacketNotFoundError,
        ReviewDecisionNotFoundError,
        ReviewTargetNotFoundError,
        SourceNotFoundError,
        SourceVersionNotFoundError,
        ViewerNotFoundError,
    ):
        app.add_exception_handler(not_found_type, not_found)
    for conflict_type in (
        PersistenceConflictError,
        RecallDependencyMismatchError,
        RecallReplayMismatchError,
        ReviewConflictError,
    ):
        app.add_exception_handler(conflict_type, conflict)
    for invalid_type in (
        CollectionError,
        QueryContractError,
        RecallRequestError,
        RecallSnapshotError,
    ):
        app.add_exception_handler(invalid_type, invalid)
    app.add_exception_handler(StarletteHTTPException, http_error)


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(detail=detail).model_dump(mode="json"),
    )
