"""Central loopback request guards and response hardening."""

from __future__ import annotations

import json
import secrets
from collections.abc import Awaitable, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_SECURITY_HEADERS = (
    (
        b"content-security-policy",
        b"default-src 'self'; script-src 'none'; object-src 'none'; "
        b"base-uri 'none'; frame-ancestors 'none'",
    ),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (
        b"permissions-policy",
        b"accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        b"magnetometer=(), microphone=(), payment=(), usb=()",
    ),
    (b"cache-control", b"no-store"),
)
_SECURITY_HEADER_NAMES = frozenset(name for name, _value in _SECURITY_HEADERS)
_FORBIDDEN_RESPONSE_HEADER_NAMES = frozenset({b"set-cookie"})
_FORBIDDEN_RESPONSE_HEADER_PREFIXES = (b"access-control-",)


class _BodyTooLargeError(RuntimeError):
    pass


class LoopbackSecurityMiddleware:
    """Authenticate every mutation, bound bodies, and harden every response."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        bearer_token: str,
        allowed_origin: str,
        trusted_hosts: tuple[str, ...],
        max_request_body_bytes: int,
    ) -> None:
        self._app = app
        self._bearer_token = bearer_token
        self._allowed_origin = allowed_origin
        self._trusted_hosts = frozenset(host.casefold() for host in trusted_hosts)
        self._max_request_body_bytes = max_request_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        response_started = False

        async def hardened_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = [
                    (name.lower(), value)
                    for name, value in message.get("headers", [])
                    if name.lower() not in _SECURITY_HEADER_NAMES
                    and name.lower() not in _FORBIDDEN_RESPONSE_HEADER_NAMES
                    and not name.lower().startswith(_FORBIDDEN_RESPONSE_HEADER_PREFIXES)
                ]
                headers.extend(_SECURITY_HEADERS)
                message["headers"] = headers
            await send(message)

        method = str(scope.get("method", "")).upper()
        if not _valid_host(_one_header(scope, b"host"), self._trusted_hosts):
            await _json_response(hardened_send, status=400, detail="invalid request")
            return
        if method in _MUTATING_METHODS:
            authorization = _one_header(scope, b"authorization")
            if not _valid_bearer(authorization, self._bearer_token):
                await _json_response(
                    hardened_send,
                    status=401,
                    detail="authentication required",
                    extra_headers=((b"www-authenticate", b"Bearer"),),
                )
                return
            origin = _one_header(scope, b"origin")
            if origin != self._allowed_origin:
                await _json_response(hardened_send, status=403, detail="request rejected")
                return

        content_lengths = _headers(scope, b"content-length")
        if len(content_lengths) > 1 or (content_lengths and not content_lengths[0].isascii()):
            await _json_response(hardened_send, status=400, detail="invalid request")
            return
        if content_lengths:
            raw_length = content_lengths[0]
            if not raw_length.isdigit():
                await _json_response(hardened_send, status=400, detail="invalid request")
                return
            if int(raw_length) > self._max_request_body_bytes:
                await _json_response(hardened_send, status=413, detail="request body too large")
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_request_body_bytes:
                    raise _BodyTooLargeError
            return message

        try:
            await self._app(scope, limited_receive, hardened_send)
        except _BodyTooLargeError:
            if response_started:
                raise
            await _json_response(hardened_send, status=413, detail="request body too large")
        except Exception:
            if response_started:
                raise
            await _json_response(hardened_send, status=500, detail="internal error")


def new_bearer_token() -> str:
    """Return a fresh 256-bit token for one app/server instance."""

    return secrets.token_urlsafe(32)


def _headers(scope: Scope, name: bytes) -> tuple[str, ...]:
    values: list[str] = []
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name.lower() != name:
            continue
        try:
            values.append(raw_value.decode("ascii"))
        except UnicodeDecodeError:
            values.append("")
    return tuple(values)


def _one_header(scope: Scope, name: bytes) -> str | None:
    values = _headers(scope, name)
    return values[0] if len(values) == 1 else None


def _valid_bearer(value: str | None, expected_token: str) -> bool:
    if value is None:
        return False
    scheme, separator, token = value.partition(" ")
    return (
        bool(separator)
        and scheme.casefold() == "bearer"
        and token == token.strip()
        and secrets.compare_digest(token, expected_token)
    )


def _valid_host(value: str | None, trusted_hosts: frozenset[str]) -> bool:
    if value is None or not value or value != value.strip() or value.count(":") > 1:
        return False
    host, separator, port = value.rpartition(":")
    if not separator:
        host = value
    elif not port.isdigit() or not 1 <= int(port) <= 65_535:
        return False
    return host.casefold() in trusted_hosts


async def _json_response(
    send: Callable[[Message], Awaitable[None]],
    *,
    status: int,
    detail: str,
    extra_headers: tuple[tuple[bytes, bytes], ...] = (),
) -> None:
    body = json.dumps({"detail": detail}, separators=(",", ":")).encode("utf-8")
    headers = (
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *extra_headers,
    )
    await send({"type": "http.response.start", "status": status, "headers": list(headers)})
    await send({"type": "http.response.body", "body": body})
