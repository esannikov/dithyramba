# ruff: noqa: RUF001
"""Loopback, GET-only web surface for one access-scoped Reading Room."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi import Path as ApiPath
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dithyramba.access import RequestScope
from dithyramba.contracts import canonical_json_bytes
from dithyramba.persistence import SQLiteRecallBackend
from dithyramba.persistence.errors import (
    EvidencePacketNotFoundError,
    PersistenceIntegrityError,
)
from dithyramba.persistence.repository import LibraryRepository, open_library
from dithyramba.reading_room import (
    ReadingRoomEdge,
    ReadingRoomEdgeType,
    ReadingRoomError,
    ReadingRoomLimits,
    ReadingRoomNode,
    ReadingRoomProjection,
    ReadingRoomService,
)
from dithyramba.recall import EvidenceFragment, EvidencePacket, QueryRequest

from .config import LoopbackApiConfig, build_config
from .models import SourceChipResponse
from .security import LoopbackSecurityMiddleware, new_bearer_token
from .services import PacketViewService, ViewerNotFoundError

_TEMPLATES = Environment(
    loader=PackageLoader("dithyramba.api", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html",), default_for_string=True),
    undefined=StrictUndefined,
    enable_async=False,
)
_STYLESHEET = Path(__file__).with_name("static") / "reading_room.css"
_NODE_ID_PATTERN = r"^[a-z][a-z0-9_]+$"
_SOURCE_ID_PATTERN = r"^source_[a-z0-9_]+$"
_SOURCE_VERSION_ID_PATTERN = r"^source_version_[a-z0-9_]+$"
_PACKET_ID_PATTERN = r"^packet_[a-z0-9_]+$"
_FRAGMENT_ID_PATTERN = r"^fragment_[a-z0-9_]+$"
_PACKET_ID_RE = re.compile(_PACKET_ID_PATTERN)

_NODE_TYPE_LABELS = {
    "source_fragment": "Фрагмент джерела",
    "statement": "Твердження",
    "evidence_link": "Доказовий зв’язок",
    "voice": "Голос",
    "entity": "Сутність",
    "entity_mention": "Згадка сутності",
    "concept": "Поняття",
    "concept_mention": "Згадка поняття",
    "concept_meaning": "Значення поняття",
    "time_context": "Часовий контекст",
}
_REVIEW_STATE_LABELS = {
    "unreviewed": "Без рішення",
    "accept": "Прийнято",
    "reject": "Відхилено",
    "revise": "Замінено новою версією",
    "defer": "Відкладено",
    "supersede": "Рішення скасовано",
    "not_reviewable": "Точна згадка",
}
_TERMINAL_REVIEW_STATES = frozenset({"accept", "reject", "revise", "not_reviewable"})
_EDGE_VERBS = {
    ReadingRoomEdgeType.STATEMENT_VOICE: "має голос",
    ReadingRoomEdgeType.STATEMENT_EVIDENCE_LINK: "має доказ",
    ReadingRoomEdgeType.EVIDENCE_LINK_FRAGMENT: "посилається на",
    ReadingRoomEdgeType.ENTITY_MENTION_ENTITY: "позначає сутність",
    ReadingRoomEdgeType.ENTITY_MENTION_FRAGMENT: "знайдено у",
    ReadingRoomEdgeType.CONCEPT_MENTION_CONCEPT: "позначає поняття",
    ReadingRoomEdgeType.CONCEPT_MENTION_FRAGMENT: "знайдено у",
    ReadingRoomEdgeType.CONCEPT_MEANING_CONCEPT: "уточнює поняття",
    ReadingRoomEdgeType.CONCEPT_MEANING_VOICE: "належить голосу",
    ReadingRoomEdgeType.CONCEPT_MEANING_STATEMENT: "спирається на",
    ReadingRoomEdgeType.STATEMENT_TIME_CONTEXT: "діє у часі",
    ReadingRoomEdgeType.CONCEPT_MEANING_TIME_CONTEXT: "зафіксовано у часі",
}


@dataclass(frozen=True, slots=True)
class ReadingRoomWebConfig:
    """One immutable web process scope; the browser cannot widen it."""

    loopback: LoopbackApiConfig
    corpus_snapshot_id: str
    access_policy_id: str
    scope: RequestScope
    limits: ReadingRoomLimits
    evidence_board: EvidenceBoardWebConfig | None = None

    def __post_init__(self) -> None:
        if self.scope.library_id != self.loopback.library_id:
            raise ValueError("Reading Room scope belongs to another Library")
        if not self.corpus_snapshot_id.startswith("snapshot_"):
            raise ValueError("corpus_snapshot_id must be explicit")
        if not self.access_policy_id.startswith("policy_"):
            raise ValueError("access_policy_id must be explicit")
        if not isinstance(self.limits, ReadingRoomLimits):
            raise TypeError("limits must be ReadingRoomLimits")
        board = self.evidence_board
        if board is not None:
            if set(board.primary_collection_ids).intersection(board.research_collection_ids):
                raise ValueError("evidence-board Collection lanes must be disjoint")
            board_scope = set(board.primary_collection_ids).union(board.research_collection_ids)
            if board_scope != set(self.scope.collection_ids):
                raise ValueError("evidence-board lanes must exactly cover Reading Room scope")


@dataclass(frozen=True, slots=True)
class EvidenceBoardWebConfig:
    """Two explicit packet lanes for one read-only evidence comparison."""

    primary_packet_id: str
    research_packet_id: str
    primary_collection_ids: tuple[str, ...]
    research_collection_ids: tuple[str, ...]
    display_question: str
    open_gap: str | None = None

    def __post_init__(self) -> None:
        for value in (self.primary_packet_id, self.research_packet_id):
            if _PACKET_ID_RE.fullmatch(value) is None:
                raise ValueError("evidence-board packet ID is invalid")
        if self.primary_packet_id == self.research_packet_id:
            raise ValueError("evidence-board lanes require two distinct packets")
        for values, label in (
            (self.primary_collection_ids, "primary"),
            (self.research_collection_ids, "research"),
        ):
            if not values or len(values) > 16 or len(set(values)) != len(values):
                raise ValueError(f"evidence-board {label} Collections are invalid")
            if any(not item.startswith("collection_") for item in values):
                raise ValueError(f"evidence-board {label} Collection ID is invalid")
            object.__setattr__(self, f"{label}_collection_ids", tuple(sorted(values)))
        question = self.display_question.strip()
        if not question or len(question) > 2_000 or question != self.display_question:
            raise ValueError("evidence-board display question is invalid")
        if self.open_gap is not None:
            gap = self.open_gap.strip()
            if not gap or len(gap) > 2_000 or gap != self.open_gap:
                raise ValueError("evidence-board open gap is invalid")


@dataclass(frozen=True, slots=True)
class _ConnectionRow:
    edge: ReadingRoomEdge
    source: ReadingRoomNode
    target: ReadingRoomNode
    source_type_label: str
    target_type_label: str
    verb: str
    review_state: str


@dataclass(frozen=True, slots=True)
class _SelectedConnection:
    edge: ReadingRoomEdge
    source: ReadingRoomNode
    target: ReadingRoomNode
    verb: str
    selected_is_source: bool


@dataclass(frozen=True, slots=True)
class _EvidenceTarget:
    source_fragment_id: str
    source_id: str
    source_version_id: str
    evidence_packet_id: str
    viewer_url: str


@dataclass(frozen=True, slots=True)
class _EvidenceBoardItem:
    fragment: EvidenceFragment
    chip: SourceChipResponse
    source_title: str
    address_label: str
    excerpt: str
    selection_url: str


@dataclass(frozen=True, slots=True)
class _EvidenceBoardLane:
    lane_id: str
    label: str
    explanation: str
    packet: EvidencePacket
    request: QueryRequest
    collection_names: tuple[str, ...]
    items: tuple[_EvidenceBoardItem, ...]
    budget_omission_count: int


@dataclass(frozen=True, slots=True)
class _EvidenceBoard:
    display_question: str
    retrieval_query: str
    open_gap: str | None
    primary: _EvidenceBoardLane
    research: _EvidenceBoardLane
    selected_item: _EvidenceBoardItem | None
    selected_lane: _EvidenceBoardLane | None
    external_provider_allowed: bool


@dataclass(slots=True)
class _Runtime:
    config: ReadingRoomWebConfig
    repository: LibraryRepository | None = None
    recall_backend: SQLiteRecallBackend | None = None
    service: ReadingRoomService | None = None
    packets: PacketViewService | None = None

    def require_service(self) -> ReadingRoomService:
        if self.service is None:
            raise RuntimeError("Reading Room lifespan is not active")
        return self.service

    def require_packets(self) -> PacketViewService:
        if self.packets is None:
            raise RuntimeError("Reading Room lifespan is not active")
        return self.packets

    def require_recall_backend(self) -> SQLiteRecallBackend:
        if self.recall_backend is None:
            raise RuntimeError("Reading Room lifespan is not active")
        return self.recall_backend


def create_reading_room_app(
    *,
    library_id: str,
    data_home: str | Path,
    corpus_snapshot_id: str,
    access_policy_id: str,
    scope: RequestScope,
    allowed_origin: str,
    limits: ReadingRoomLimits | None = None,
    evidence_board: EvidenceBoardWebConfig | None = None,
    test_only_allow_testserver: bool = False,
) -> FastAPI:
    """Create a script-free loopback viewer pinned to one immutable access scope."""

    loopback = build_config(
        library_id=library_id,
        data_home=data_home,
        allowed_origin=allowed_origin,
        test_only_allow_testserver=test_only_allow_testserver,
    )
    config = ReadingRoomWebConfig(
        loopback=loopback,
        corpus_snapshot_id=corpus_snapshot_id,
        access_policy_id=access_policy_id,
        scope=scope,
        limits=limits or ReadingRoomLimits(),
        evidence_board=evidence_board,
    )
    runtime = _Runtime(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        with open_library(config.loopback.library_id, data_root=config.loopback.data_home) as repo:
            repo.verify()
            recall_backend = SQLiteRecallBackend(repo)
            runtime.repository = repo
            runtime.recall_backend = recall_backend
            runtime.service = ReadingRoomService(repo)
            runtime.packets = PacketViewService(repo, recall_backend)
            try:
                yield
            finally:
                runtime.packets = None
                runtime.service = None
                runtime.recall_backend = None
                runtime.repository = None

    app = FastAPI(
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.reading_room_config = config
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(loopback.trusted_hosts),
        www_redirect=False,
    )
    # There are no mutation routes. Reusing the hardened middleware still gives
    # strict Host checks, no-store responses, a script-free CSP, and bounded bodies.
    app.add_middleware(
        LoopbackSecurityMiddleware,
        bearer_token=new_bearer_token(),
        allowed_origin=loopback.allowed_origin,
        trusted_hosts=loopback.trusted_hosts,
        max_request_body_bytes=loopback.max_request_body_bytes,
    )

    @app.exception_handler(ReadingRoomError)
    async def reading_room_error(_request: Request, _error: ReadingRoomError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": (
                    "Reading Room could not build this permitted view. "
                    "Verify the Library, snapshot, policy, and scope locally."
                )
            },
        )

    @app.exception_handler(ViewerNotFoundError)
    @app.exception_handler(EvidencePacketNotFoundError)
    async def viewer_not_found(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "not found"})

    @app.exception_handler(PersistenceIntegrityError)
    async def viewer_integrity_error(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": "authorized evidence could not be verified"},
        )

    @app.get("/reading-room.css", include_in_schema=False)
    async def stylesheet() -> Response:
        return Response(
            _STYLESHEET.read_text(encoding="utf-8"),
            media_type="text/css; charset=utf-8",
        )

    @app.get("/projection.json", include_in_schema=False)
    async def projection_json() -> Response:
        projection = _project(runtime)
        return Response(
            canonical_json_bytes(projection.payload()),
            media_type="application/json",
            headers={"ETag": f'"{projection.projection_hash}"'},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def overview(
        selected: Annotated[
            str | None,
            Query(pattern=_NODE_ID_PATTERN, min_length=3, max_length=160),
        ] = None,
    ) -> HTMLResponse:
        projection = _project(runtime)
        template = _TEMPLATES.get_template("reading_room.html")
        context = _template_context(
            projection,
            selected=selected,
            scope=config.scope,
            evidence_targets=_evidence_targets(runtime, projection),
        )
        context["evidence_board_enabled"] = config.evidence_board is not None
        return HTMLResponse(template.render(**context))

    @app.get("/evidence-board", response_class=HTMLResponse, include_in_schema=False)
    async def evidence_board_view(
        selected_packet: Annotated[
            str | None,
            Query(pattern=_PACKET_ID_PATTERN),
        ] = None,
        selected_fragment: Annotated[
            str | None,
            Query(pattern=_FRAGMENT_ID_PATTERN),
        ] = None,
    ) -> HTMLResponse:
        if config.evidence_board is None:
            raise ViewerNotFoundError("evidence board is not configured")
        board = _evidence_board(
            runtime,
            selected_packet=selected_packet,
            selected_fragment=selected_fragment,
        )
        template = _TEMPLATES.get_template("evidence_board.html")
        return HTMLResponse(
            template.render(
                board=board,
                library=runtime.repository.library if runtime.repository is not None else None,
                policy=_project(runtime).access,
            )
        )

    @app.get(
        "/sources/{source_id}/versions/{source_version_id}",
        response_class=HTMLResponse,
        include_in_schema=False,
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
        # Rebuild the current access-scoped projection before any packet text is
        # loaded.  A packet identifier is not an authorization capability on its
        # own: it must be the exact target emitted for this pinned snapshot,
        # policy, exclusion set, permitted set, and visible graph fragment.
        current_projection = _project(runtime)
        authorized_target = _authorized_evidence_target(
            runtime,
            current_projection,
            packet_id=packet,
            fragment_id=fragment,
        )
        if (
            authorized_target is None
            or authorized_target.source_id != source_id
            or authorized_target.source_version_id != source_version_id
            or authorized_target.evidence_packet_id != packet
        ):
            raise ViewerNotFoundError("packet is outside the Reading Room scope")
        projection = runtime.require_packets().load_viewer(
            source_id=source_id,
            source_version_id=source_version_id,
            source_fragment_id=fragment,
            evidence_packet_id=packet,
        )
        board_target = _board_evidence_targets(runtime).get((packet, fragment))
        template = _TEMPLATES.get_template("reading_room_source_viewer.html")
        return HTMLResponse(
            template.render(
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
                back_url=(
                    f"/evidence-board?selected_packet={packet}&selected_fragment={fragment}"
                    if board_target is not None
                    else "/"
                ),
            )
        )

    return app


def _project(runtime: _Runtime) -> ReadingRoomProjection:
    config = runtime.config
    return runtime.require_service().project(
        corpus_snapshot_id=config.corpus_snapshot_id,
        access_policy_id=config.access_policy_id,
        scope=config.scope,
        limits=config.limits,
    )


def _template_context(
    projection: ReadingRoomProjection,
    *,
    selected: str | None,
    scope: RequestScope,
    evidence_targets: dict[str, _EvidenceTarget],
) -> dict[str, object]:
    nodes = {item.node_id: item for item in projection.graph.nodes}
    node_counts: dict[str, int] = {}
    for node in projection.graph.nodes:
        node_counts[node.node_type.value] = node_counts.get(node.node_type.value, 0) + 1

    rows = tuple(_connection_row(edge, nodes=nodes) for edge in projection.graph.edges)
    selected_node = nodes.get(selected) if selected is not None else _default_node(projection)
    selected_connections = (
        ()
        if selected_node is None
        else tuple(
            _selected_connection(edge, selected_node=selected_node, nodes=nodes)
            for edge in projection.graph.edges
            if selected_node.node_id in {edge.source_node_id, edge.target_node_id}
        )
    )
    selected_evidence = (
        ()
        if selected_node is None
        else tuple(
            evidence_targets[fragment_id]
            for fragment_id in selected_node.provenance_fragment_ids
            if fragment_id in evidence_targets
        )
    )
    memory_state_label, memory_state_explanation = _memory_state(projection)
    return {
        "projection": projection,
        "connection_rows": rows,
        "selected_node": selected_node,
        "selected_connections": selected_connections,
        "selected_evidence": selected_evidence,
        "scope": scope,
        "omitted_node_count": max(
            0,
            projection.graph.visible_node_count - len(projection.graph.nodes),
        ),
        "omitted_edge_count": max(
            0,
            projection.graph.visible_edge_count - len(projection.graph.edges),
        ),
        "node_counts": node_counts,
        "node_type_labels": _NODE_TYPE_LABELS,
        "review_state_labels": _REVIEW_STATE_LABELS,
        "memory_state_label": memory_state_label,
        "memory_state_explanation": memory_state_explanation,
    }


def _connection_row(
    edge: ReadingRoomEdge,
    *,
    nodes: dict[str, ReadingRoomNode],
) -> _ConnectionRow:
    source = nodes[edge.source_node_id]
    target = nodes[edge.target_node_id]
    states = {source.review_state, target.review_state}
    if "reject" in states:
        review_state = "rejected"
    elif any(state not in _TERMINAL_REVIEW_STATES for state in states):
        review_state = "unresolved"
    else:
        review_state = "reviewed"
    return _ConnectionRow(
        edge=edge,
        source=source,
        target=target,
        source_type_label=_NODE_TYPE_LABELS[source.node_type.value],
        target_type_label=_NODE_TYPE_LABELS[target.node_type.value],
        verb=_EDGE_VERBS[edge.edge_type],
        review_state=review_state,
    )


def _selected_connection(
    edge: ReadingRoomEdge,
    *,
    selected_node: ReadingRoomNode,
    nodes: dict[str, ReadingRoomNode],
) -> _SelectedConnection:
    return _SelectedConnection(
        edge=edge,
        source=nodes[edge.source_node_id],
        target=nodes[edge.target_node_id],
        verb=_EDGE_VERBS[edge.edge_type],
        selected_is_source=edge.source_node_id == selected_node.node_id,
    )


def _evidence_targets(
    runtime: _Runtime,
    projection: ReadingRoomProjection,
) -> dict[str, _EvidenceTarget]:
    repository = runtime.repository
    if repository is None:
        raise RuntimeError("Reading Room lifespan is not active")
    fragment_ids = tuple(
        sorted(
            {
                fragment_id
                for node in projection.graph.nodes
                for fragment_id in node.provenance_fragment_ids
            }
        )
    )
    if not fragment_ids:
        return {}
    placeholders = ",".join("?" for _item in fragment_ids)
    rows = repository._store.connection.execute(
        f"""
        SELECT pi.source_fragment_id, sv.source_id, sf.source_version_id,
               ep.evidence_packet_id, ep.created_at, pi.rank
        FROM packet_items AS pi
        JOIN evidence_packets AS ep
          ON ep.evidence_packet_id = pi.evidence_packet_id
        JOIN query_requests AS qr
          ON qr.query_request_id = ep.query_request_id
        JOIN recall_run_artifacts AS rra
          ON rra.evidence_packet_id = ep.evidence_packet_id
        JOIN access_receipts AS ar
          ON ar.access_receipt_id = rra.access_receipt_id
        JOIN source_fragments AS sf
          ON sf.source_fragment_id = pi.source_fragment_id
        JOIN source_versions AS sv
          ON sv.source_version_id = sf.source_version_id
        WHERE qr.library_id = ?
          AND ep.corpus_snapshot_id = ?
          AND qr.access_policy_id = ?
          AND ar.exclusion_hash = ?
          AND ar.permitted_set_hash = ?
          AND pi.source_fragment_id IN ({placeholders})
        ORDER BY ep.created_at DESC, pi.rank, ep.evidence_packet_id
        """,
        (
            projection.library.library_id,
            projection.access.corpus_snapshot_id,
            projection.access.access_policy_id,
            projection.access.exclusion_hash,
            projection.access.permitted_set_hash,
            *fragment_ids,
        ),
    ).fetchall()
    targets: dict[str, _EvidenceTarget] = {}
    for row in rows:
        fragment_id = str(row[0])
        if fragment_id in targets:
            continue
        source_id = str(row[1])
        version_id = str(row[2])
        packet_id = str(row[3])
        targets[fragment_id] = _EvidenceTarget(
            source_fragment_id=fragment_id,
            source_id=source_id,
            source_version_id=version_id,
            evidence_packet_id=packet_id,
            viewer_url=(
                f"/sources/{source_id}/versions/{version_id}"
                f"?packet={packet_id}&fragment={fragment_id}"
            ),
        )
    return targets


def _evidence_board(
    runtime: _Runtime,
    *,
    selected_packet: str | None,
    selected_fragment: str | None,
) -> _EvidenceBoard:
    config = runtime.config.evidence_board
    repository = runtime.repository
    if config is None or repository is None:
        raise ViewerNotFoundError("evidence board is not configured")
    packets = runtime.require_packets()
    recall_backend = runtime.require_recall_backend()
    primary_projection = packets.load_packet(config.primary_packet_id)
    research_projection = packets.load_packet(config.research_packet_id)
    primary_request = recall_backend.load_query_request(primary_projection.packet.query_request_id)
    research_request = recall_backend.load_query_request(
        research_projection.packet.query_request_id
    )
    _validate_board_requests(
        runtime,
        primary_request=primary_request,
        research_request=research_request,
    )
    primary = _board_lane(
        runtime,
        lane_id="primary",
        label="Первинні джерела",
        explanation="Що буквально зафіксовано в дозволених документах.",
        packet=primary_projection.packet,
        request=primary_request,
        chips=primary_projection.source_chips,
        collection_ids=config.primary_collection_ids,
    )
    research = _board_lane(
        runtime,
        lane_id="research",
        label="Дослідницькі тлумачення",
        explanation="Що вже припущено, зіставлено або залишено для перевірки.",
        packet=research_projection.packet,
        request=research_request,
        chips=research_projection.source_chips,
        collection_ids=config.research_collection_ids,
    )
    selected_item: _EvidenceBoardItem | None = None
    selected_lane: _EvidenceBoardLane | None = None
    if (selected_packet is None) != (selected_fragment is None):
        raise ViewerNotFoundError("evidence-board selection is incomplete")
    if selected_packet is not None and selected_fragment is not None:
        for lane in (primary, research):
            if lane.packet.evidence_packet_id != selected_packet:
                continue
            selected_item = next(
                (
                    item
                    for item in lane.items
                    if item.fragment.source_fragment_id == selected_fragment
                ),
                None,
            )
            if selected_item is not None:
                selected_lane = lane
                break
        if selected_item is None:
            raise ViewerNotFoundError("evidence-board selection is outside configured packets")
    else:
        for lane in (primary, research):
            if lane.items:
                selected_item = lane.items[0]
                selected_lane = lane
                break
    policy = repository.get_access_policy(runtime.config.access_policy_id).snapshot
    return _EvidenceBoard(
        display_question=config.display_question,
        retrieval_query=primary_request.question,
        open_gap=config.open_gap,
        primary=primary,
        research=research,
        selected_item=selected_item,
        selected_lane=selected_lane,
        external_provider_allowed=policy.allow_external_provider,
    )


def _validate_board_requests(
    runtime: _Runtime,
    *,
    primary_request: QueryRequest,
    research_request: QueryRequest,
) -> None:
    config = runtime.config.evidence_board
    if config is None:
        raise ViewerNotFoundError("evidence board is not configured")
    for request, expected_collections in (
        (primary_request, config.primary_collection_ids),
        (research_request, config.research_collection_ids),
    ):
        if (
            request.library_id != runtime.config.loopback.library_id
            or request.access_policy_id != runtime.config.access_policy_id
            or request.purpose != runtime.config.scope.purpose
            or request.collection_ids != expected_collections
        ):
            raise ReadingRoomError("evidence-board packet scope is inconsistent")
    if (
        primary_request.question != research_request.question
        or primary_request.exclusions != research_request.exclusions
        or primary_request.retrieval != research_request.retrieval
        or primary_request.result_contract != research_request.result_contract
    ):
        raise ReadingRoomError("evidence-board packets are not a comparable pair")


def _board_lane(
    runtime: _Runtime,
    *,
    lane_id: str,
    label: str,
    explanation: str,
    packet: EvidencePacket,
    request: QueryRequest,
    chips: tuple[SourceChipResponse, ...],
    collection_ids: tuple[str, ...],
) -> _EvidenceBoardLane:
    repository = runtime.repository
    if repository is None:
        raise RuntimeError("Reading Room lifespan is not active")
    items: list[_EvidenceBoardItem] = []
    for fragment, chip in zip(packet.source_fragments, chips, strict=True):
        source = repository.get_source(chip.source_id)
        items.append(
            _EvidenceBoardItem(
                fragment=fragment,
                chip=chip,
                source_title=source.title or source.canonical_uri,
                address_label=_address_label(fragment),
                excerpt=_query_excerpt(fragment.text, request.question),
                selection_url=(
                    "/evidence-board"
                    f"?selected_packet={packet.evidence_packet_id}"
                    f"&selected_fragment={fragment.source_fragment_id}"
                ),
            )
        )
    omission_count = sum(
        item.count or 0
        for item in packet.coverage_report.omissions
        if item.category.value == "budget"
    )
    collection_names = tuple(
        repository.get_collection(collection_id).config.name for collection_id in collection_ids
    )
    return _EvidenceBoardLane(
        lane_id=lane_id,
        label=label,
        explanation=explanation,
        packet=packet,
        request=request,
        collection_names=collection_names,
        items=tuple(items),
        budget_omission_count=omission_count,
    )


def _address_label(fragment: EvidenceFragment) -> str:
    payload = fragment.source_address.payload()
    headings = payload.get("heading_path")
    line_start = payload.get("line_start")
    line_end = payload.get("line_end")
    parts: list[str] = []
    if isinstance(headings, list) and headings:
        parts.append(" › ".join(str(item) for item in headings))
    if isinstance(line_start, int) and isinstance(line_end, int):
        parts.append(
            f"рядок {line_start}" if line_start == line_end else f"рядки {line_start}–{line_end}"
        )
    return " · ".join(parts) or fragment.source_fragment_id


def _query_excerpt(text: str, question: str, *, limit: int = 360) -> str:
    """Return a bounded display excerpt around the first meaningful query match."""

    folded = text.casefold()
    terms = {
        item
        for item in re.findall(r"[^\W_]+", question.casefold(), flags=re.UNICODE)
        if len(item) >= 4
    }
    offsets = tuple(offset for term in terms if (offset := folded.find(term)) >= 0)
    anchor = min(offsets, default=0)
    start = max(0, anchor - limit // 4)
    end = min(len(text), start + limit)
    if end - start < limit:
        start = max(0, end - limit)
    excerpt = text[start:end].strip()
    return f"{'…' if start else ''}{excerpt}{'…' if end < len(text) else ''}"


def _board_evidence_targets(
    runtime: _Runtime,
) -> dict[tuple[str, str], _EvidenceTarget]:
    if runtime.config.evidence_board is None:
        return {}
    board = _evidence_board(runtime, selected_packet=None, selected_fragment=None)
    targets: dict[tuple[str, str], _EvidenceTarget] = {}
    for lane in (board.primary, board.research):
        packet_id = lane.packet.evidence_packet_id
        for item in lane.items:
            fragment_id = item.fragment.source_fragment_id
            targets[(packet_id, fragment_id)] = _EvidenceTarget(
                source_fragment_id=fragment_id,
                source_id=item.chip.source_id,
                source_version_id=item.chip.source_version_id,
                evidence_packet_id=packet_id,
                viewer_url=item.chip.viewer_url,
            )
    return targets


def _authorized_evidence_target(
    runtime: _Runtime,
    projection: ReadingRoomProjection,
    *,
    packet_id: str,
    fragment_id: str,
) -> _EvidenceTarget | None:
    board_target = _board_evidence_targets(runtime).get((packet_id, fragment_id))
    if board_target is not None:
        return board_target
    projection_target = _evidence_targets(runtime, projection).get(fragment_id)
    if projection_target is None or projection_target.evidence_packet_id != packet_id:
        return None
    return projection_target


def _default_node(projection: ReadingRoomProjection) -> ReadingRoomNode | None:
    for node in projection.graph.nodes:
        if node.node_type.value == "statement":
            return node
    return projection.graph.nodes[0] if projection.graph.nodes else None


def _memory_state(projection: ReadingRoomProjection) -> tuple[str, str]:
    if projection.corpus.source_fragment_count == 0:
        return (
            "порожній дозволений зріз",
            "Політика успішно застосована, але не дозволила жодного фрагмента у цьому scope.",
        )
    if not projection.recent_runs:
        return (
            "корпус готовий",
            "Дозволені джерела доступні, але meaning extraction у цьому scope ще не зафіксовано.",
        )
    states = {item.coverage_state for item in projection.recent_runs}
    if "failed" in states or "refused" in states:
        return (
            "потребує уваги",
            "Принаймні один видимий ProcessingRun завершився відмовою або помилкою.",
        )
    if "partial" in states:
        return (
            "частково опрацьовано",
            "Покриття неповне; пропуски залишаються CoverageReport, а не доказовими прогалинами.",
        )
    if projection.review.unresolved_count:
        return (
            "кандидати очікують перевірки",
            "Meaning extraction завершено, але людські рішення ще не покривають "
            "усі видимі кандидати.",
        )
    return (
        "опрацьовано в поточному scope",
        "Усі видимі кандидати мають рішення; це не перетворює їх на універсальну істину.",
    )
