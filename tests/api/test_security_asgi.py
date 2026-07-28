"""Low-level ASGI tests for body streaming and malformed header defenses."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

import pytest
from starlette.types import Message, Receive, Scope, Send

from dithyramba.api.security import LoopbackSecurityMiddleware

_SendableApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def _http_scope(headers: list[tuple[bytes, bytes]]) -> Scope:
    return cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 4312),
        },
    )


async def _invoke(
    *,
    scope: Scope,
    messages: list[Message],
    downstream: _SendableApp | None = None,
    body_limit: int = 4,
) -> list[Message]:
    sent: list[Message] = []
    pending = iter(messages)

    async def receive() -> Message:
        return next(pending)

    async def send(message: Message) -> None:
        sent.append(message)

    async def consume(_scope: Scope, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [
                    (b"content-security-policy", b"unsafe"),
                    (b"set-cookie", b"session=forbidden"),
                    (b"access-control-allow-origin", b"*"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    middleware = LoopbackSecurityMiddleware(
        downstream or consume,
        bearer_token="secret",
        allowed_origin="http://localhost:4312",
        trusted_hosts=("127.0.0.1", "localhost"),
        max_request_body_bytes=body_limit,
    )
    await middleware(scope, receive, send)
    return sent


def _status(messages: list[Message]) -> int:
    start = next(message for message in messages if message["type"] == "http.response.start")
    return int(start["status"])


def test_non_http_scope_is_delegated_without_http_guards() -> None:
    called = False

    async def downstream(_scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal called
        called = True

    scope = cast(Scope, {"type": "lifespan", "asgi": {"version": "3.0"}})
    asyncio.run(_invoke(scope=scope, messages=[], downstream=downstream))
    assert called is True


@pytest.mark.parametrize(
    "headers",
    [
        [(b"host", b"localhost"), (b"content-length", b"1"), (b"content-length", b"1")],
        [(b"host", b"localhost"), (b"content-length", b"\xff")],
        [(b"host", b"localhost"), (b"content-length", b"not-a-number")],
    ],
)
def test_malformed_content_length_is_rejected(headers: list[tuple[bytes, bytes]]) -> None:
    authenticated_headers = [
        *headers,
        (b"authorization", b"Bearer secret"),
        (b"origin", b"http://localhost:4312"),
    ]
    messages = asyncio.run(
        _invoke(
            scope=_http_scope(authenticated_headers),
            messages=[{"type": "http.request", "body": b"", "more_body": False}],
        )
    )
    assert _status(messages) == 400


@pytest.mark.parametrize(
    "host",
    [b"", b" localhost", b"localhost:nope", b"localhost:0", b"localhost:99999"],
)
def test_malformed_host_is_rejected(host: bytes) -> None:
    messages = asyncio.run(
        _invoke(
            scope=_http_scope([(b"host", host)]),
            messages=[{"type": "http.request", "body": b"", "more_body": False}],
        )
    )
    assert _status(messages) == 400


def test_chunked_body_is_bounded_without_content_length() -> None:
    headers = [
        (b"host", b"localhost:4312"),
        (b"authorization", b"Bearer secret"),
        (b"origin", b"http://localhost:4312"),
    ]
    messages = asyncio.run(
        _invoke(
            scope=_http_scope(headers),
            messages=[
                {"type": "http.request", "body": b"123", "more_body": True},
                {"type": "http.request", "body": b"45", "more_body": False},
            ],
        )
    )
    assert _status(messages) == 413


def test_security_headers_replace_downstream_values_once() -> None:
    headers = [
        (b"host", b"LOCALHOST:4312"),
        (b"authorization", b"bearer secret"),
        (b"origin", b"http://localhost:4312"),
        (b"content-length", b"0"),
    ]
    messages = asyncio.run(
        _invoke(
            scope=_http_scope(headers),
            messages=[{"type": "http.request", "body": b"", "more_body": False}],
        )
    )
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_headers = list(start.get("headers", []))
    csp_values = [value for name, value in response_headers if name == b"content-security-policy"]
    assert _status(messages) == 204
    assert csp_values == [
        b"default-src 'self'; script-src 'none'; object-src 'none'; "
        b"base-uri 'none'; frame-ancestors 'none'"
    ]
    assert all(name != b"set-cookie" for name, _value in response_headers)
    assert all(not name.startswith(b"access-control-") for name, _value in response_headers)


def test_oversized_body_after_response_start_is_not_double_started() -> None:
    async def starts_first(_scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await receive()

    headers = [
        (b"host", b"localhost"),
        (b"authorization", b"Bearer secret"),
        (b"origin", b"http://localhost:4312"),
    ]
    with pytest.raises(RuntimeError):
        asyncio.run(
            _invoke(
                scope=_http_scope(headers),
                messages=[{"type": "http.request", "body": b"12345", "more_body": False}],
                downstream=starts_first,
            )
        )
