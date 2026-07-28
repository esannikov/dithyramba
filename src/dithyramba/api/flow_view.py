"""Small GET-only explainer for the main Dithyramba memory flow."""

# ruff: noqa: RUF001

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dithyramba.contracts import canonical_json_bytes, sha256_hex

from .config import LoopbackApiConfig, build_config
from .security import LoopbackSecurityMiddleware, new_bearer_token

_TEMPLATES = Environment(
    loader=PackageLoader("dithyramba.api", "templates"),
    autoescape=select_autoescape(enabled_extensions=("html",), default_for_string=True),
    undefined=StrictUndefined,
    enable_async=False,
)
_STYLESHEET = Path(__file__).with_name("static") / "flow_view.css"
_FLOW_LIBRARY_ID = "library_00000000000040008000000000000001"
_NODE_PATTERN = r"^[a-z][a-z0-9_-]{1,31}$"


@dataclass(frozen=True, slots=True)
class FlowNode:
    node_id: str
    label_lines: tuple[str, ...]
    subtitle: str
    explanation: str
    receives: str
    passes: str
    x: int
    y: int
    width: int
    height: int
    kind: str

    @property
    def label(self) -> str:
        return " ".join(self.label_lines)


@dataclass(frozen=True, slots=True)
class FlowLink:
    link_id: str
    source: str
    target: str
    label: str
    path: str
    kind: str
    weight: str


@dataclass(frozen=True, slots=True)
class FlowViewWebConfig:
    loopback: LoopbackApiConfig
    version: str = "1.0"


_NODES = (
    FlowNode(
        node_id="sources",
        label_lines=("Джерела",),
        subtitle="файли · книги · дані",
        explanation=(
            "Дозволені матеріали входять у систему як джерела. Їхня наявність "
            "ще не робить жодне твердження доведеним."
        ),
        receives="матеріали дослідника",
        passes="точно адресований текст",
        x=38,
        y=82,
        width=188,
        height=92,
        kind="input",
    ),
    FlowNode(
        node_id="question",
        label_lines=("Питання",),
        subtitle="що потрібно зʼясувати",
        explanation=(
            "Дослідник формулює питання або гіпотезу. Воно задає напрям пошуку, "
            "але не змінює сам корпус."
        ),
        receives="намір дослідника",
        passes="чіткий запит",
        x=38,
        y=438,
        width=188,
        height=92,
        kind="human",
    ),
    FlowNode(
        node_id="library",
        label_lines=("Бібліотека",),
        subtitle="локальний точний текст",
        explanation=(
            "Dithyramba витягує читабельний текст, зберігає його точні адреси й "
            "фіксує склад корпусу для відтворюваної роботи."
        ),
        receives="дозволені джерела",
        passes="зафіксований корпус",
        x=310,
        y=82,
        width=202,
        height=92,
        kind="boundary",
    ),
    FlowNode(
        node_id="search",
        label_lines=("Пошук",),
        subtitle="локальний добір",
        explanation=(
            "Спочатку працює точний текстовий пошук. За потреби невеликий список "
            "кандидатів локально переранжовується."
        ),
        receives="корпус + питання",
        passes="релевантні кандидати",
        x=310,
        y=342,
        width=202,
        height=92,
        kind="search",
    ),
    FlowNode(
        node_id="evidence",
        label_lines=("Пакет", "доказів"),
        subtitle="точні уривки",
        explanation=(
            "Із кандидатів збирається невеликий пакет точних уривків із назвами "
            "джерел і стабільними адресами."
        ),
        receives="кандидати пошуку",
        passes="перевірні опори",
        x=604,
        y=236,
        width=202,
        height=108,
        kind="evidence",
    ),
    FlowNode(
        node_id="gate",
        label_lines=("Перевірка",),
        subtitle="чи доведені всі частини",
        explanation=(
            "Брама перевіряє, чи є опора для кожної потрібної частини висновку. "
            "Це перевірка повноти доказу, а не відсоток істини."
        ),
        receives="пакет доказів",
        passes="достатньо / бракує",
        x=888,
        y=236,
        width=218,
        height=108,
        kind="gate",
    ),
    FlowNode(
        node_id="memory",
        label_lines=("Перевірена", "памʼять"),
        subtitle="прийняте + застереження",
        explanation=(
            "До памʼяті потрапляють прийняті рішення, явні застереження та "
            "простежувані звʼязки. Попередня історія не стирається."
        ),
        receives="достатньо підтверджений висновок",
        passes="безпечний контекст",
        x=1198,
        y=112,
        width=218,
        height=108,
        kind="memory",
    ),
    FlowNode(
        node_id="lens",
        label_lines=("Lens",),
        subtitle="робочий стіл дослідника",
        explanation=(
            "Lens показує питання, гіпотези, час, прогалини й точні джерела. "
            "Остаточне рішення та наступна дія залишаються за людиною."
        ),
        receives="памʼять або явна прогалина",
        passes="рішення людини",
        x=1198,
        y=410,
        width=218,
        height=108,
        kind="human",
    ),
)

_LINKS = (
    FlowLink(
        "sources_library",
        "sources",
        "library",
        "витягує",
        "M226 128 C264 128 272 128 310 128",
        "source",
        "wide",
    ),
    FlowLink(
        "library_search",
        "library",
        "search",
        "фіксує корпус",
        "M411 174 C411 234 411 282 411 342",
        "boundary",
        "wide",
    ),
    FlowLink(
        "question_search",
        "question",
        "search",
        "спрямовує",
        "M226 484 C268 484 272 388 310 388",
        "human",
        "medium",
    ),
    FlowLink(
        "search_evidence",
        "search",
        "evidence",
        "знаходить",
        "M512 388 C554 388 562 290 604 290",
        "search",
        "wide",
    ),
    FlowLink(
        "evidence_gate",
        "evidence",
        "gate",
        "передає опори",
        "M806 290 C838 290 856 290 888 290",
        "evidence",
        "wide",
    ),
    FlowLink(
        "gate_memory",
        "gate",
        "memory",
        "приймає",
        "M1106 268 C1140 268 1154 166 1198 166",
        "memory",
        "wide",
    ),
    FlowLink(
        "gate_lens",
        "gate",
        "lens",
        "показує прогалину",
        "M1106 320 C1144 320 1158 464 1198 464",
        "gap",
        "narrow",
    ),
    FlowLink(
        "memory_lens",
        "memory",
        "lens",
        "повертає людині",
        "M1307 220 C1307 288 1307 342 1307 410",
        "memory",
        "medium",
    ),
)

_NODE_BY_ID = {node.node_id: node for node in _NODES}


def create_flow_view_app(
    *,
    allowed_origin: str,
    test_only_allow_testserver: bool = False,
) -> FastAPI:
    """Create a small, offline, script-free process map on loopback."""

    loopback = build_config(
        library_id=_FLOW_LIBRARY_ID,
        data_home=_STYLESHEET.parent.resolve(),
        allowed_origin=allowed_origin,
        test_only_allow_testserver=test_only_allow_testserver,
    )
    config = FlowViewWebConfig(loopback=loopback)
    app = FastAPI(debug=False, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.flow_view_config = config
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

    @app.get("/flow-view.css", include_in_schema=False)
    async def stylesheet() -> Response:
        return Response(
            _STYLESHEET.read_text(encoding="utf-8"),
            media_type="text/css; charset=utf-8",
        )

    @app.get("/flow.json", include_in_schema=False)
    async def flow_json() -> Response:
        payload = _flow_payload()
        payload_hash = sha256_hex(canonical_json_bytes(payload))
        return Response(
            canonical_json_bytes({**payload, "flow_hash": payload_hash}),
            media_type="application/json",
            headers={"ETag": f'"{payload_hash}"'},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(
        node: Annotated[str | None, Query(pattern=_NODE_PATTERN, max_length=32)] = None,
    ) -> HTMLResponse:
        if node is not None and node not in _NODE_BY_ID:
            raise HTTPException(status_code=404, detail="not found")
        selected = None if node is None else _NODE_BY_ID[node]
        active_link_ids = (
            frozenset()
            if selected is None
            else frozenset(
                link.link_id for link in _LINKS if selected.node_id in {link.source, link.target}
            )
        )
        html = _TEMPLATES.get_template("flow_view.html").render(
            nodes=_NODES,
            links=_LINKS,
            selected=selected,
            active_link_ids=active_link_ids,
        )
        return HTMLResponse(html)

    return app


def _flow_payload() -> dict[str, object]:
    return {
        "schema": "dithyramba.flow_view/1.0",
        "mode": "static_process_map_not_live_trace",
        "nodes": [
            {
                "node_id": node.node_id,
                "label": node.label,
                "subtitle": node.subtitle,
                "kind": node.kind,
            }
            for node in _NODES
        ],
        "links": [
            {
                "link_id": link.link_id,
                "source": link.source,
                "target": link.target,
                "label": link.label,
            }
            for link in _LINKS
        ],
    }


__all__ = ["FlowViewWebConfig", "create_flow_view_app"]
