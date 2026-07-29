"""Immutable contracts shared by safe readers, parsers, and ingest services."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex

SOURCE_ADDRESS_SCHEMA = "dithyramba.source_address/1.0"
PARSER_PROFILE = "index/1.0"
LARGE_DOCUMENT_PARSER_PROFILE = "index/large-document/1.0"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_POINT_PATTERN = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{3}$")
_FAILURE_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)


class MediaType(StrEnum):
    """Formats accepted by the frozen VS0 ingest boundary."""

    MARKDOWN = "text/markdown"
    PLAIN_TEXT = "text/plain"
    PDF = "application/pdf"


class FragmentKind(StrEnum):
    """Persisted SourceFragment kinds from schema v1."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    PAGE_TEXT = "page_text"


class ParseStatus(StrEnum):
    """Parser outcome persisted on one immutable SourceVersion."""

    PROCESSED = "processed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ParserLimits:
    """Hard limits applied before or inside any parser process."""

    max_file_bytes: int = 25 * 1024 * 1024
    max_pdf_pages: int = 500
    max_extracted_codepoints: int = 2_000_000
    timeout_seconds: int = 30
    max_rss_mib: int = 512

    def __post_init__(self) -> None:
        for name in (
            "max_file_bytes",
            "max_pdf_pages",
            "max_extracted_codepoints",
            "timeout_seconds",
            "max_rss_mib",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def parser_profile(name: str) -> tuple[str, ParserLimits]:
    """Return one explicit, bounded core-ingest parser profile."""

    if name == "default":
        return PARSER_PROFILE, ParserLimits()
    if name == "large-document":
        return (
            LARGE_DOCUMENT_PARSER_PROFILE,
            ParserLimits(
                max_file_bytes=512 * 1024 * 1024,
                max_pdf_pages=1_500,
                max_extracted_codepoints=20_000_000,
                timeout_seconds=180,
                max_rss_mib=1_024,
            ),
        )
    raise ValueError("unknown parser profile; expected default or large-document")


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Descriptor-derived identity used for before/after mutation detection."""

    device: int
    inode: int
    size: int
    modified_ns: int

    def __post_init__(self) -> None:
        for name in ("device", "inode", "size", "modified_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SourceBytes:
    """Exact bytes obtained from one descriptor-stable source read."""

    relative_path: str
    canonical_uri: str
    media_type: MediaType
    data: bytes
    content_sha256: str
    source_modified_at: str
    identity: FileIdentity

    def __post_init__(self) -> None:
        if type(self.relative_path) is not str:
            raise TypeError("relative_path must be str")
        raw_parts = self.relative_path.split("/")
        relative = PurePosixPath(self.relative_path)
        if (
            not self.relative_path.strip()
            or "\x00" in self.relative_path
            or "\\" in self.relative_path
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in raw_parts)
            or relative.as_posix() == "."
        ):
            raise ValueError("relative_path must be an unambiguous relative POSIX path")
        if type(self.canonical_uri) is not str:
            raise TypeError("canonical_uri must be str")
        parsed_uri = urlsplit(self.canonical_uri)
        if (
            parsed_uri.scheme != "file"
            or parsed_uri.netloc not in ("", "localhost")
            or not parsed_uri.path.startswith("/")
            or parsed_uri.path == "/"
            or parsed_uri.query
            or parsed_uri.fragment
        ):
            raise ValueError("canonical_uri must be an absolute file URI")
        if not isinstance(self.media_type, MediaType):
            raise TypeError("media_type must be MediaType")
        if type(self.data) is not bytes:
            raise TypeError("data must be immutable bytes")
        if not isinstance(self.identity, FileIdentity):
            raise TypeError("identity must be FileIdentity")
        if not _SHA256_PATTERN.fullmatch(self.content_sha256):
            raise ValueError("content_sha256 must be lowercase SHA-256 hex")
        if sha256_hex(self.data) != self.content_sha256:
            raise ValueError("content_sha256 does not match data")
        if len(self.data) != self.identity.size:
            raise ValueError("descriptor identity size does not match data")
        if type(self.source_modified_at) is not str or not _is_canonical_timestamp(
            self.source_modified_at
        ):
            raise ValueError("source_modified_at must be canonical UTC with microseconds")

    @property
    def byte_size(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class MarkdownSourceAddress:
    """Exact Markdown/TXT address in normalized full-document text."""

    heading_path: tuple[str, ...]
    line_start: int
    line_end: int
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if type(self.heading_path) is not tuple or any(
            type(part) is not str for part in self.heading_path
        ):
            raise TypeError("heading_path must be a tuple of strings")
        normalized_path = tuple(unicodedata.normalize("NFC", part) for part in self.heading_path)
        if any(not part.strip() for part in normalized_path):
            raise ValueError("heading_path entries must be non-empty")
        object.__setattr__(self, "heading_path", normalized_path)
        if type(self.line_start) is not int or self.line_start < 1:
            raise ValueError("line_start must be a positive integer")
        if type(self.line_end) is not int or self.line_end < self.line_start:
            raise ValueError("line_end must be at least line_start")
        _validate_offsets(self.char_start, self.char_end)

    def payload(self) -> dict[str, object]:
        return {
            "schema": SOURCE_ADDRESS_SCHEMA,
            "kind": "markdown",
            "heading_path": list(self.heading_path),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass(frozen=True, slots=True)
class PdfSourceAddress:
    """Exact PDF page/bbox address with canonical decimal-point strings."""

    page: int
    bbox: tuple[str, str, str, str]
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if type(self.page) is not int or self.page < 1:
            raise ValueError("page must be a positive integer")
        if type(self.bbox) is not tuple or len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly four canonical point strings")
        if any(
            type(point) is not str or not _POINT_PATTERN.fullmatch(point) for point in self.bbox
        ):
            raise ValueError("bbox points must use fixed-point strings with three decimals")
        x0, top, x1, bottom = (Decimal(point) for point in self.bbox)
        if any(point == "-0.000" for point in self.bbox) or any(
            abs(point) > Decimal("1000000000") for point in (x0, top, x1, bottom)
        ):
            raise ValueError("bbox points must be canonical and within the supported range")
        if x0 >= x1 or top >= bottom:
            raise ValueError("bbox must satisfy x0 < x1 and top < bottom")
        _validate_offsets(self.char_start, self.char_end)

    def payload(self) -> dict[str, object]:
        return {
            "schema": SOURCE_ADDRESS_SCHEMA,
            "kind": "pdf",
            "page": self.page,
            "bbox": list(self.bbox),
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


SourceAddress = MarkdownSourceAddress | PdfSourceAddress


@dataclass(frozen=True, slots=True)
class ParsedFragment:
    """One normalized, non-empty fragment and its exact address."""

    ordinal: int
    kind: FragmentKind
    text: str
    address: SourceAddress

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if not isinstance(self.kind, FragmentKind):
            raise TypeError("kind must be FragmentKind")
        if type(self.text) is not str:
            raise TypeError("text must be str")
        normalized = unicodedata.normalize("NFC", self.text)
        if not normalized.strip() or "\x00" in normalized:
            raise ValueError("fragment text must be non-empty and contain no NUL")
        object.__setattr__(self, "text", normalized)
        if self.kind in (FragmentKind.HEADING, FragmentKind.PARAGRAPH) and not isinstance(
            self.address, MarkdownSourceAddress
        ):
            raise ValueError("Markdown fragment kinds require MarkdownSourceAddress")
        if self.kind is FragmentKind.PAGE_TEXT and not isinstance(self.address, PdfSourceAddress):
            raise ValueError("page_text requires PdfSourceAddress")
        if self.address.char_end - self.address.char_start != len(self.text):
            raise ValueError("address character span must equal fragment text length")

    @property
    def text_sha256(self) -> str:
        return sha256_hex(self.text.encode("utf-8"))

    @property
    def address_payload(self) -> dict[str, object]:
        return self.address.payload()

    @property
    def address_json(self) -> str:
        return canonical_json_bytes(self.address_payload).decode("utf-8")

    @property
    def address_hash(self) -> str:
        return canonical_sha256_hex(self.address_payload)


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Validated parser result; failure and skip states contain no fragments."""

    status: ParseStatus
    fragments: tuple[ParsedFragment, ...]
    parser_revision: str
    normalized_codepoints: int
    failure_code: str | None = None
    parser_profile: str = PARSER_PROFILE

    def __post_init__(self) -> None:
        if not isinstance(self.status, ParseStatus):
            raise TypeError("status must be ParseStatus")
        if type(self.fragments) is not tuple or any(
            not isinstance(fragment, ParsedFragment) for fragment in self.fragments
        ):
            raise TypeError("fragments must be a tuple of ParsedFragment")
        if not self.parser_revision.strip() or not self.parser_profile.strip():
            raise ValueError("parser revision/profile must be non-empty")
        if type(self.normalized_codepoints) is not int or self.normalized_codepoints < 0:
            raise ValueError("normalized_codepoints must be a non-negative integer")
        if self.status is ParseStatus.PROCESSED:
            if not self.fragments or self.failure_code is not None:
                raise ValueError("processed result requires fragments and no failure code")
            if tuple(fragment.ordinal for fragment in self.fragments) != tuple(
                range(len(self.fragments))
            ):
                raise ValueError("fragment ordinals must be contiguous from zero")
        else:
            if self.fragments or self.failure_code is None:
                raise ValueError("skipped/failed result requires no fragments and a failure code")
            if not _FAILURE_CODE_PATTERN.fullmatch(self.failure_code):
                raise ValueError("failure_code must use stable snake_case grammar")

    @classmethod
    def processed(
        cls,
        fragments: tuple[ParsedFragment, ...],
        *,
        parser_revision: str,
        normalized_codepoints: int,
    ) -> ParseResult:
        return cls(
            status=ParseStatus.PROCESSED,
            fragments=fragments,
            parser_revision=parser_revision,
            normalized_codepoints=normalized_codepoints,
        )

    @classmethod
    def skipped(cls, code: str, *, parser_revision: str) -> ParseResult:
        return cls(ParseStatus.SKIPPED, (), parser_revision, 0, code)

    @classmethod
    def failed(cls, code: str, *, parser_revision: str) -> ParseResult:
        return cls(ParseStatus.FAILED, (), parser_revision, 0, code)


def format_pdf_point(value: Any) -> str:
    """Convert a finite parser coordinate to a canonical three-decimal string."""

    if type(value) is bool:
        raise ValueError("PDF point must be numeric")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("PDF point must be numeric") from exc
    if not decimal.is_finite() or abs(decimal) > Decimal("1000000000"):
        raise ValueError("PDF point must be finite and within the supported range")
    quantized = decimal.quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN)
    if quantized == 0:
        quantized = Decimal("0.000")
    return f"{quantized:.3f}"


def _validate_offsets(char_start: int, char_end: int) -> None:
    if type(char_start) is not int or char_start < 0:
        raise ValueError("char_start must be a non-negative integer")
    if type(char_end) is not int or char_end <= char_start:
        raise ValueError("char_end must be greater than char_start")


def _is_canonical_timestamp(value: str) -> bool:
    if _TIMESTAMP_PATTERN.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") == value
