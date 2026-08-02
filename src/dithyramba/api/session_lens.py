"""GET-only human projection of one durable interactive research session."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlencode, urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.interactive.models import AgentEvidencePacket
from dithyramba.persistence import (
    LibraryRepository,
    ResearchSessionNotFoundError,
    SQLiteResearchSessionRepository,
    open_library,
)
from dithyramba.persistence.recall import SQLiteRecallBackend
from dithyramba.recall import EvidenceFragment
from dithyramba.sessions import (
    ResearchSession,
    ResearchSessionState,
    SessionArtifactKind,
    SessionEvent,
    SessionEventKind,
)

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
_STYLESHEET = Path(__file__).with_name("static") / "session_lens.css"
_PACKET_PATTERN = r"^packet_[a-z0-9]+(?:_[a-z0-9]+)*$"
_FRAGMENT_PATTERN = r"^fragment_[a-z0-9]+(?:_[a-z0-9]+)*$"


@dataclass(frozen=True, slots=True)
class SessionLensWebConfig:
    loopback: LoopbackApiConfig
    session_id: str


@dataclass(frozen=True, slots=True)
class _EvidenceView:
    packet_id: str
    fragment: EvidenceFragment
    chip: SourceChipResponse
    source_title: str
    source_locator: str
    source_role: str
    address_label: str
    selection_url: str


@dataclass(frozen=True, slots=True)
class _EventView:
    event: SessionEvent
    label: str
    explanation: str
    tone: str
    evidence: tuple[_EvidenceView, ...]


@dataclass(frozen=True, slots=True)
class _SessionProjection:
    session: ResearchSession
    state: ResearchSessionState
    events: tuple[_EventView, ...]
    selected: _EvidenceView | None
    projection_payload: dict[str, object]
    projection_hash: str


@dataclass(slots=True)
class _Runtime:
    config: SessionLensWebConfig
    repository: LibraryRepository | None = None
    sessions: SQLiteResearchSessionRepository | None = None
    packets: PacketViewService | None = None

    def require_repository(self) -> LibraryRepository:
        if self.repository is None:
            raise RuntimeError("Session Lens lifespan is not active")
        return self.repository

    def require_sessions(self) -> SQLiteResearchSessionRepository:
        if self.sessions is None:
            raise RuntimeError("Session Lens lifespan is not active")
        return self.sessions

    def require_packets(self) -> PacketViewService:
        if self.packets is None:
            raise RuntimeError("Session Lens lifespan is not active")
        return self.packets


def create_session_lens_app(
    *,
    library_id: str,
    data_home: str | Path,
    session_id: str,
    allowed_origin: str,
    test_only_allow_testserver: bool = False,
) -> FastAPI:
    """Create one offline session workbench with packet-backed source inspection."""

    loopback = build_config(
        library_id=library_id,
        data_home=data_home,
        allowed_origin=allowed_origin,
        test_only_allow_testserver=test_only_allow_testserver,
    )
    config = SessionLensWebConfig(loopback=loopback, session_id=session_id)
    runtime = _Runtime(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        with open_library(config.loopback.library_id, data_root=config.loopback.data_home) as repo:
            repo.verify()
            runtime.repository = repo
            runtime.sessions = SQLiteResearchSessionRepository(repo)
            runtime.packets = PacketViewService(repo, SQLiteRecallBackend(repo))
            try:
                runtime.sessions.get_session(config.session_id)
                yield
            finally:
                runtime.packets = None
                runtime.sessions = None
                runtime.repository = None

    app = FastAPI(
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.session_lens_config = config
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

    @app.get("/session-lens.css", include_in_schema=False)
    async def stylesheet() -> Response:
        return Response(
            _STYLESHEET.read_text(encoding="utf-8"),
            media_type="text/css; charset=utf-8",
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/projection.json", include_in_schema=False)
    async def projection_json() -> Response:
        projected = _project(runtime, selected_packet=None, selected_fragment=None)
        return Response(
            canonical_json_bytes(projected.projection_payload),
            media_type="application/json",
            headers={"ETag": f'"{projected.projection_hash}"'},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(
        packet: Annotated[
            str | None,
            Query(pattern=_PACKET_PATTERN, max_length=96),
        ] = None,
        fragment: Annotated[
            str | None,
            Query(pattern=_FRAGMENT_PATTERN, max_length=96),
        ] = None,
    ) -> HTMLResponse:
        try:
            projected = _project(
                runtime,
                selected_packet=packet,
                selected_fragment=fragment,
            )
        except (ResearchSessionNotFoundError, ViewerNotFoundError) as error:
            raise HTTPException(status_code=404, detail="not found") from error
        repository = runtime.require_repository()
        template = _TEMPLATES.get_template("session_lens.html")
        return HTMLResponse(
            template.render(
                library=repository.library,
                session=projected.session,
                state=projected.state,
                event_views=projected.events,
                selected=projected.selected,
                projection_hash=projected.projection_hash,
            )
        )

    return app


def _project(
    runtime: _Runtime,
    *,
    selected_packet: str | None,
    selected_fragment: str | None,
) -> _SessionProjection:
    if (selected_packet is None) != (selected_fragment is None):
        raise ViewerNotFoundError("session Lens selection is incomplete")
    repository = runtime.require_repository()
    sessions = runtime.require_sessions()
    packets = runtime.require_packets()
    session = sessions.get_session(runtime.config.session_id)
    state = sessions.get_state(session.session_id)
    raw_events = sessions.list_events(session.session_id)
    event_views: list[_EventView] = []
    all_evidence: list[_EvidenceView] = []
    for event in raw_events:
        evidence: list[_EvidenceView] = []
        if event.kind is SessionEventKind.EVIDENCE_ATTACHED:
            packet_refs = tuple(
                item
                for item in event.artifact_refs
                if item.artifact_kind is SessionArtifactKind.EVIDENCE_PACKET
            )
            if len(packet_refs) != 1:
                raise ViewerNotFoundError("session evidence event has no exact packet")
            packet_reference = packet_refs[0]
            completion = sessions.load_recall_completion_for_evidence_event(event.event_id)
            if completion is not None and completion.agent_evidence_json is not None:
                try:
                    agent_packet = AgentEvidencePacket.model_validate_json(
                        completion.agent_evidence_json
                    )
                except ValidationError as error:
                    raise ViewerNotFoundError(
                        "session agent evidence projection is invalid"
                    ) from error
                if (
                    completion.evidence_packet_id != packet_reference.artifact_id
                    or completion.evidence_packet_hash != packet_reference.artifact_hash
                    or agent_packet.evidence_packet_id != packet_reference.artifact_id
                    or agent_packet.evidence_packet_hash != packet_reference.artifact_hash
                ):
                    raise ViewerNotFoundError("session evidence event has a stale compact packet")
                agent_projection = packets.load_agent_packet(agent_packet)
                source_fragments = agent_projection.packet.source_fragments
                source_chips = agent_projection.source_chips
            else:
                # Pre-1.1 interactive events have no compact persisted projection.
                # Preserve exact replay for those legacy journals only.
                full_projection = packets.load_packet(packet_reference.artifact_id)
                source_fragments = full_projection.packet.source_fragments
                source_chips = full_projection.source_chips
            for fragment, chip in zip(
                source_fragments,
                source_chips,
                strict=True,
            ):
                source = repository.get_source(chip.source_id)
                item = _EvidenceView(
                    packet_id=packet_reference.artifact_id,
                    fragment=fragment,
                    chip=chip,
                    source_title=source.title or source.canonical_uri,
                    source_locator=_source_locator(source.canonical_uri),
                    source_role=_source_role(chip),
                    address_label=_address_label(fragment),
                    selection_url="/?"
                    + urlencode(
                        {
                            "packet": packet_reference.artifact_id,
                            "fragment": fragment.source_fragment_id,
                        }
                    ),
                )
                evidence.append(item)
                all_evidence.append(item)
        label, explanation, tone = _event_copy(event.kind)
        event_views.append(
            _EventView(
                event=event,
                label=label,
                explanation=explanation,
                tone=tone,
                evidence=tuple(evidence),
            )
        )

    selected: _EvidenceView | None = None
    if selected_packet is not None and selected_fragment is not None:
        selected = next(
            (
                item
                for item in all_evidence
                if item.packet_id == selected_packet
                and item.fragment.source_fragment_id == selected_fragment
            ),
            None,
        )
        if selected is None:
            raise ViewerNotFoundError("selected evidence is outside this session")
    elif all_evidence:
        selected = all_evidence[0]

    payload: dict[str, object] = {
        "schema": "dithyramba.session_lens_projection/1.0",
        "library_id": session.library_id,
        "session": session.model_dump(mode="json"),
        "state": state.model_dump(mode="json"),
        "journal": [
            {
                "event": item.event.model_dump(mode="json"),
                "evidence": [
                    {
                        "packet_id": evidence.packet_id,
                        "source_id": evidence.chip.source_id,
                        "source_version_id": evidence.chip.source_version_id,
                        "source_fragment_id": evidence.fragment.source_fragment_id,
                        "text_sha256": evidence.fragment.text_sha256,
                        "rank": evidence.fragment.rank,
                        "source_title": evidence.source_title,
                        "source_role": evidence.source_role,
                        "address": evidence.fragment.source_address.payload(),
                    }
                    for evidence in item.evidence
                ],
            }
            for item in event_views
        ],
        "boundary": "attached evidence is visible and replayable, not automatically accepted",
    }
    projection_hash = canonical_sha256_hex(payload)
    return _SessionProjection(
        session=session,
        state=state,
        events=tuple(event_views),
        selected=selected,
        projection_payload={**payload, "projection_hash": projection_hash},
        projection_hash=projection_hash,
    )


def _event_copy(kind: SessionEventKind) -> tuple[str, str, str]:
    return {
        SessionEventKind.QUESTION_ASKED: (
            "Питання",
            "Що дослідник чи агент перевіряє зараз.",
            "question",
        ),
        SessionEventKind.EVIDENCE_ATTACHED: (
            "Знайдені опори",
            "Точні уривки прикріплено до сесії; це ще не прийнятий висновок.",
            "evidence",
        ),
        SessionEventKind.ANSWER_DRAFTED: (
            "Робочий синтез",
            "Чернетка інтерпретації, яку ще слід перевірити.",
            "draft",
        ),
        SessionEventKind.GAP_RECORDED: (
            "Прогалина",
            "Назване місце, де доказів чи точності поки бракує.",
            "gap",
        ),
        SessionEventKind.PATH_REJECTED: (
            "Відхилений шлях",
            "Підхід, який не варто повторювати без нових підстав.",
            "rejected",
        ),
        SessionEventKind.CANDIDATE_LINKED: (
            "Матеріал-кандидат",
            "Поняття чи твердження для майбутньої людської перевірки.",
            "candidate",
        ),
        SessionEventKind.DECISION_RECORDED: (
            "Людське рішення",
            "Явно записане оператором рішення.",
            "decision",
        ),
        SessionEventKind.SESSION_CLOSED: (
            "Сесію завершено",
            "Подальші записи в цю сесію закрито.",
            "closed",
        ),
    }[kind]


def _source_role(chip: SourceChipResponse) -> str:
    return {
        "root": "оригінальна версія джерела",
        "derivative": "похідна версія",
        "duplicate": "дублікат перевіреного джерела",
    }.get(chip.family_role, chip.family_role)


def _source_locator(canonical_uri: str) -> str:
    """Return a readable exact locator without exposing a full local path."""

    parsed = urlparse(canonical_uri)
    if parsed.scheme and parsed.scheme not in {"file", "http", "https"}:
        return canonical_uri
    path_name = Path(unquote(parsed.path)).name
    if path_name:
        return path_name
    if parsed.netloc:
        return parsed.netloc
    return canonical_uri


def _address_label(fragment: EvidenceFragment) -> str:
    payload = fragment.source_address.payload()
    headings = payload.get("heading_path")
    line_start = payload.get("line_start")
    line_end = payload.get("line_end")
    parts: list[str] = []
    if isinstance(headings, list) and headings:
        parts.append(" / ".join(str(item) for item in headings))
    if isinstance(line_start, int) and isinstance(line_end, int):
        parts.append(
            f"рядок {line_start}" if line_start == line_end else f"рядки {line_start}-{line_end}"
        )
    return " · ".join(parts) or "точна адреса збережена в пакеті"


__all__ = ["SessionLensWebConfig", "create_session_lens_app"]
