"""Explicit one-Library configuration for the loopback HTTP surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dithyramba.library import validate_library_id


@dataclass(frozen=True, slots=True)
class LoopbackApiConfig:
    """Immutable launch configuration with no ambient Library selection."""

    library_id: str
    data_home: Path
    allowed_origin: str
    test_only_allow_testserver: bool = False
    max_request_body_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        validate_library_id(self.library_id)
        if not isinstance(self.data_home, Path) or not self.data_home.is_absolute():
            raise ValueError("data_home must be an absolute Path")
        if type(self.test_only_allow_testserver) is not bool:
            raise TypeError("test_only_allow_testserver must be a bool")
        if (
            type(self.max_request_body_bytes) is not int
            or not 1_024 <= self.max_request_body_bytes <= 1024 * 1024
        ):
            raise ValueError("max_request_body_bytes must be between 1 KiB and 1 MiB")
        _validate_origin(
            self.allowed_origin,
            allow_testserver=self.test_only_allow_testserver,
        )

    @property
    def trusted_hosts(self) -> tuple[str, ...]:
        hosts = ("127.0.0.1", "localhost")
        if self.test_only_allow_testserver:
            return (*hosts, "testserver")
        return hosts


def build_config(
    *,
    library_id: str,
    data_home: str | Path,
    allowed_origin: str,
    test_only_allow_testserver: bool = False,
    max_request_body_bytes: int = 64 * 1024,
) -> LoopbackApiConfig:
    """Normalize caller-owned paths while retaining an explicit origin boundary."""

    return LoopbackApiConfig(
        library_id=library_id,
        data_home=Path(data_home).expanduser().resolve(strict=False),
        allowed_origin=allowed_origin,
        test_only_allow_testserver=test_only_allow_testserver,
        max_request_body_bytes=max_request_body_bytes,
    )


def _validate_origin(value: str, *, allow_testserver: bool) -> None:
    if type(value) is not str or not value:
        raise ValueError("allowed_origin must be a canonical loopback Origin")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("allowed_origin contains an invalid port") from exc
    allowed_hosts = {"127.0.0.1", "localhost"}
    if allow_testserver:
        allowed_hosts.add("testserver")
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("allowed_origin must be an exact loopback Origin")
    canonical_host = parsed.hostname
    canonical = f"{parsed.scheme}://{canonical_host}"
    if port is not None:
        canonical += f":{port}"
    if value != canonical:
        raise ValueError("allowed_origin must use canonical lowercase Origin syntax")
