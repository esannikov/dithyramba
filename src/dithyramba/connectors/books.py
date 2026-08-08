"""Safe, deterministic Markdown projections for immutable book bytes.

This connector deliberately stops before the ingest/persistence boundary.  It
does not add EPUB or FB2 to the core ``MediaType`` enum and it does not promote
projected text to evidence.  Callers receive Markdown plus a canonical JSON
sidecar that binds every projected unit to the exact source bytes and locator.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import stat
import sys
import tempfile
import unicodedata
import zipfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

from defusedxml import ElementTree as DefusedElementTree  # type: ignore[import-untyped]
from defusedxml.common import DefusedXmlException  # type: ignore[import-untyped]

from dithyramba.contracts import canonical_json_bytes, sha256_hex
from dithyramba.ingest.errors import IngestError
from dithyramba.ingest.models import (
    FileIdentity,
    MediaType,
    ParserLimits,
    PdfSourceAddress,
    SourceBytes,
)
from dithyramba.ingest.pdf import PDF_PARSER_REVISION, parse_pdf_source

BOOK_PROJECTION_SCHEMA = "dithyramba.book_projection/1.0"
BOOK_CONNECTOR_REVISION = "books/1.2"
_DOCLING_SIDECAR_CONTRACT = "dithyramba.book_docling_sidecar/1.0"
_EPUB_BLOCK_KINDS: Mapping[str, str] = MappingProxyType(
    {
        "h1": "heading",
        "h2": "heading",
        "h3": "heading",
        "h4": "heading",
        "h5": "heading",
        "h6": "heading",
        "p": "paragraph",
        "li": "list_item",
        "blockquote": "quotation",
        "pre": "preformatted",
        "table": "table",
        "dl": "definition_list",
        "aside": "note",
        "figcaption": "caption",
    }
)
_FB2_BLOCK_KINDS: Mapping[str, str] = MappingProxyType(
    {
        "p": "paragraph",
        "subtitle": "heading",
        "text-author": "attribution",
        "v": "verse",
        "table": "table",
    }
)
_XML_PROLOG_PREFIX = re.compile(
    rb"^(?:\xef\xbb\xbf)?[\x20\t\r\n]*(?:<\?xml[\x20\t\r\n]+[^?<>]*\?>[\x20\t\r\n]*)?$",
    re.ASCII,
)
_XHTML_10_STRICT_DOCTYPE = re.compile(
    rb"""<!DOCTYPE[\x20\t\r\n]+html[\x20\t\r\n]+PUBLIC[\x20\t\r\n]+
        ["']-//W3C//DTD[\x20\t\r\n]+XHTML[\x20\t\r\n]+1\.0[\x20\t\r\n]+Strict//EN["']
        [\x20\t\r\n]+["']http://www\.w3\.org/TR/xhtml1/DTD/xhtml1-strict\.dtd["']
        [\x20\t\r\n]*>""",
    re.ASCII | re.IGNORECASE | re.VERBOSE,
)


class BookMediaType(StrEnum):
    """Book formats supported outside the frozen core ingest media types."""

    PDF = "application/pdf"
    EPUB = "application/epub+zip"
    FB2 = "application/x-fictionbook+xml"


class BookProjectionStatus(StrEnum):
    """Completeness of one projection attempt."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class BookProjectionProfile:
    """All resource bounds used by a book projection."""

    profile_id: str
    max_file_bytes: int
    max_pdf_pages: int
    max_archive_members: int
    max_archive_member_bytes: int
    max_archive_uncompressed_bytes: int
    max_compression_ratio: int
    max_xml_elements: int
    max_units: int
    max_extracted_codepoints: int
    timeout_seconds: int
    max_rss_mib: int

    def __post_init__(self) -> None:
        if type(self.profile_id) is not str or not self.profile_id.strip():
            raise ValueError("profile_id must be non-empty text")
        for field_name in (
            "max_file_bytes",
            "max_pdf_pages",
            "max_archive_members",
            "max_archive_member_bytes",
            "max_archive_uncompressed_bytes",
            "max_compression_ratio",
            "max_xml_elements",
            "max_units",
            "max_extracted_codepoints",
            "timeout_seconds",
            "max_rss_mib",
        ):
            if type(getattr(self, field_name)) is not int or getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


DEFAULT_BOOK_PROFILE = BookProjectionProfile(
    profile_id="book/default-1.0",
    max_file_bytes=25 * 1024 * 1024,
    max_pdf_pages=500,
    max_archive_members=2_000,
    max_archive_member_bytes=8 * 1024 * 1024,
    max_archive_uncompressed_bytes=64 * 1024 * 1024,
    max_compression_ratio=200,
    max_xml_elements=100_000,
    max_units=20_000,
    max_extracted_codepoints=2_000_000,
    timeout_seconds=30,
    max_rss_mib=512,
)

LARGE_BOOK_PROFILE = BookProjectionProfile(
    profile_id="book/large-1.0",
    max_file_bytes=512 * 1024 * 1024,
    max_pdf_pages=1_500,
    max_archive_members=8_000,
    max_archive_member_bytes=24 * 1024 * 1024,
    max_archive_uncompressed_bytes=256 * 1024 * 1024,
    max_compression_ratio=200,
    max_xml_elements=400_000,
    max_units=80_000,
    max_extracted_codepoints=20_000_000,
    timeout_seconds=180,
    max_rss_mib=1_024,
)


@dataclass(frozen=True, slots=True)
class BookSourceMember:
    """One immutable member in a deterministic expanded-EPUB capture."""

    path: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_archive_path(self.path)
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("byte_size must be a non-negative integer")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sha256 must be lowercase SHA-256 hex")


@dataclass(frozen=True, slots=True)
class BookSource:
    """One already captured immutable source and non-canonical role metadata."""

    name: str
    media_type: BookMediaType
    data: bytes
    roles: tuple[str, ...] = ("private",)
    source_form: str = "file"
    members: tuple[BookSourceMember, ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip() or "\x00" in self.name:
            raise ValueError("name must be non-empty text without NUL")
        object.__setattr__(self, "name", unicodedata.normalize("NFC", self.name))
        if not isinstance(self.media_type, BookMediaType):
            raise TypeError("media_type must be BookMediaType")
        if type(self.data) is not bytes:
            raise TypeError("data must be immutable bytes")
        if type(self.roles) is not tuple or any(
            type(role) is not str or not role.strip() or "\x00" in role for role in self.roles
        ):
            raise ValueError("roles must be a tuple of non-empty strings")
        normalized_roles = tuple(unicodedata.normalize("NFC", role) for role in self.roles)
        if len(set(normalized_roles)) != len(normalized_roles):
            raise ValueError("roles must be unique")
        object.__setattr__(self, "roles", normalized_roles)
        if self.source_form not in ("file", "expanded_epub_repack"):
            raise ValueError("source_form is unsupported")
        if type(self.members) is not tuple or any(
            not isinstance(member, BookSourceMember) for member in self.members
        ):
            raise TypeError("members must be a tuple of BookSourceMember")
        if self.source_form == "file" and self.members:
            raise ValueError("file sources cannot declare expanded EPUB members")
        if self.source_form == "expanded_epub_repack":
            if self.media_type is not BookMediaType.EPUB:
                raise ValueError("expanded EPUB sources require EPUB media")
            paths = tuple(member.path for member in self.members)
            if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
                raise ValueError("expanded EPUB members must be uniquely path-sorted")


@dataclass(frozen=True, slots=True)
class BookProjection:
    """Deterministic connector artifacts; neither artifact is persisted here."""

    status: BookProjectionStatus
    markdown: str
    sidecar_json: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.status, BookProjectionStatus):
            raise TypeError("status must be BookProjectionStatus")
        if type(self.markdown) is not str:
            raise TypeError("markdown must be str")
        if type(self.sidecar_json) is not bytes:
            raise TypeError("sidecar_json must be immutable bytes")

    @property
    def markdown_sha256(self) -> str:
        return sha256_hex(self.markdown.encode("utf-8"))

    @property
    def sidecar_sha256(self) -> str:
        return sha256_hex(self.sidecar_json)


@dataclass(frozen=True, slots=True)
class CatalogProjectionSummary:
    """Deterministic count-only result from a catalog projection."""

    complete: int
    partial: int
    failure: int

    @property
    def total(self) -> int:
        return self.complete + self.partial + self.failure


@dataclass(frozen=True, slots=True)
class _Unit:
    kind: str
    label: str
    text: str
    locator: MappingProxyType[str, Any]


class _ProjectionFailure(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def project_book(
    source: BookSource,
    *,
    profile: BookProjectionProfile = DEFAULT_BOOK_PROFILE,
) -> BookProjection:
    """Project immutable book bytes without extending canonical ingest schemas."""

    if not isinstance(source, BookSource):
        raise TypeError("source must be BookSource")
    if not isinstance(profile, BookProjectionProfile):
        raise TypeError("profile must be BookProjectionProfile")
    source_hash = sha256_hex(source.data)
    if len(source.data) > profile.max_file_bytes:
        return _failure_projection(source, profile, source_hash, "file_size_limit_exceeded")

    warnings: list[str] = []
    parser_revision = BOOK_CONNECTOR_REVISION
    try:
        if source.media_type is BookMediaType.PDF:
            units, warnings = _project_pdf(source, profile)
            parser_revision = PDF_PARSER_REVISION
        elif source.media_type is BookMediaType.EPUB:
            units, warnings = _project_epub(source.data, profile)
        else:
            units, warnings = _project_fb2(source.data, profile)
        if not units:
            raise _ProjectionFailure("no_extractable_text", "book contains no extractable text")
    except _ProjectionFailure as error:
        return _failure_projection(source, profile, source_hash, error.code)
    except IngestError as error:
        return _failure_projection(source, profile, source_hash, error.code)

    markdown, sidecar_units = _render_markdown(source.name, units)
    status = BookProjectionStatus.PARTIAL if warnings else BookProjectionStatus.COMPLETE
    manifest = _base_manifest(source, profile, source_hash, status)
    manifest.update(
        {
            "failure": None,
            "markdown": {
                "codepoints": len(markdown),
                "sha256": sha256_hex(markdown.encode("utf-8")),
            },
            "parser_revision": parser_revision,
            "units": sidecar_units,
            "warnings": warnings,
        }
    )
    return BookProjection(
        status=status,
        markdown=markdown,
        sidecar_json=canonical_json_bytes(manifest),
    )


def project_book_path(
    path: str | Path,
    *,
    profile: BookProjectionProfile = DEFAULT_BOOK_PROFILE,
    roles: tuple[str, ...] = ("private",),
) -> BookProjection:
    """Capture one stable file or expanded ``.epub`` directory, then project it."""

    source_path = Path(path)
    if not source_path.is_absolute():
        raise ValueError("book source path must be absolute")
    try:
        source_stat = source_path.lstat()
    except OSError as error:
        raise ValueError("book source path is unavailable") from error
    if stat.S_ISLNK(source_stat.st_mode):
        raise ValueError("book source symlink rejected")
    suffix = source_path.suffix.lower()
    if stat.S_ISDIR(source_stat.st_mode):
        if suffix != ".epub":
            raise ValueError("only .epub book sources may be directories")
        source = _capture_expanded_epub(source_path, profile, roles)
    elif stat.S_ISREG(source_stat.st_mode):
        media_type = _media_type_for_suffix(suffix)
        data = _read_stable_file(source_path, profile.max_file_bytes)
        source = BookSource(source_path.name, media_type, data, roles=roles)
    else:
        raise ValueError("book source must be a regular file or expanded .epub directory")
    return project_book(source, profile=profile)


def project_book_catalog(
    input_directory: str | Path,
    output_directory: str | Path,
    *,
    profile: BookProjectionProfile = LARGE_BOOK_PROFILE,
    roles: tuple[str, ...] = ("private",),
) -> CatalogProjectionSummary:
    """Project a catalog tree, treating expanded ``.epub`` directories as leaves."""

    input_root = _validated_catalog_root(Path(input_directory))
    output_root = Path(output_directory)
    if not output_root.is_absolute():
        raise ValueError("book output directory must be absolute")
    input_resolved = input_root.resolve(strict=True)
    output_resolved = output_root.resolve(strict=False)
    if output_resolved == input_resolved or output_resolved.is_relative_to(input_resolved):
        raise ValueError("book output directory must be outside the input catalog")
    _ensure_private_directory(output_root)

    counts = {status: 0 for status in BookProjectionStatus}
    for source_path in _discover_book_paths(input_root):
        relative = source_path.relative_to(input_root)
        projection = project_book_path(source_path, profile=profile, roles=roles)
        destination = _ensure_private_subdirectory(output_root, relative.parent)
        _atomic_private_write(destination / f"{relative.name}.md", projection.markdown.encode())
        _atomic_private_write(
            destination / f"{relative.name}.book-projection.json",
            projection.sidecar_json,
        )
        counts[projection.status] += 1
    return CatalogProjectionSummary(
        complete=counts[BookProjectionStatus.COMPLETE],
        partial=counts[BookProjectionStatus.PARTIAL],
        failure=counts[BookProjectionStatus.FAILURE],
    )


def _media_type_for_suffix(suffix: str) -> BookMediaType:
    try:
        return {
            ".epub": BookMediaType.EPUB,
            ".fb2": BookMediaType.FB2,
            ".pdf": BookMediaType.PDF,
        }[suffix]
    except KeyError as error:
        raise ValueError(f"unsupported book source suffix: {suffix or '<none>'}") from error


def _read_stable_file(path: Path, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"could not open book source: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"book source is not a regular file: {path}")
        if before.st_size > maximum_bytes:
            raise ValueError(f"book source exceeds its byte limit: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError(f"book source exceeds its byte limit: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        entry_after = path.lstat()
    except OSError as error:
        raise ValueError(f"book source changed during capture: {path}") from error
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(before) != _stat_identity(
        entry_after
    ):
        raise ValueError(f"book source changed during capture: {path}")
    return b"".join(chunks)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_mode)


def _capture_expanded_epub(
    root: Path,
    profile: BookProjectionProfile,
    roles: tuple[str, ...],
) -> BookSource:
    root_before = root.lstat()
    entries = _scan_expanded_epub(root, profile)
    payloads: dict[str, bytes] = {}
    members: list[BookSourceMember] = []
    total = 0
    for member_path, file_path in entries:
        payload = _read_stable_file(file_path, profile.max_archive_member_bytes)
        total += len(payload)
        if total > profile.max_archive_uncompressed_bytes:
            raise ValueError("expanded EPUB exceeds its uncompressed byte limit")
        payloads[member_path] = payload
        members.append(BookSourceMember(member_path, len(payload), sha256_hex(payload)))
    if _stat_identity(root_before) != _stat_identity(root.lstat()):
        raise ValueError("expanded EPUB root changed during capture")
    if tuple(path for path, _file in _scan_expanded_epub(root, profile)) != tuple(payloads):
        raise ValueError("expanded EPUB member roster changed during capture")
    archive_bytes = _deterministic_epub_zip(payloads)
    if len(archive_bytes) > profile.max_file_bytes:
        raise ValueError("deterministic EPUB repack exceeds the file byte limit")
    return BookSource(
        name=root.name,
        media_type=BookMediaType.EPUB,
        data=archive_bytes,
        roles=roles,
        source_form="expanded_epub_repack",
        members=tuple(members),
    )


def _scan_expanded_epub(root: Path, profile: BookProjectionProfile) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def visit(directory: Path, raw_parts: tuple[str, ...]) -> None:
        try:
            children = sorted(
                os.scandir(directory),
                key=lambda item: unicodedata.normalize("NFC", item.name),
            )
        except OSError as error:
            raise ValueError(f"could not scan expanded EPUB directory: {directory}") from error
        for child in children:
            child_parts = (*raw_parts, child.name)
            relative = unicodedata.normalize("NFC", PurePosixPath(*child_parts).as_posix())
            if relative in seen:
                raise ValueError("expanded EPUB member names collide after NFC normalization")
            seen.add(relative)
            try:
                child_stat = child.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"could not inspect expanded EPUB member: {relative}") from error
            if stat.S_ISLNK(child_stat.st_mode):
                raise ValueError(f"expanded EPUB symlink rejected: {relative}")
            if stat.S_ISDIR(child_stat.st_mode):
                visit(Path(child.path), child_parts)
            elif stat.S_ISREG(child_stat.st_mode):
                _validate_archive_path(relative)
                entries.append((relative, Path(child.path)))
                if len(entries) > profile.max_archive_members:
                    raise ValueError("expanded EPUB has too many members")
            else:
                raise ValueError(f"expanded EPUB member is not regular: {relative}")

    visit(root, ())
    return sorted(entries, key=lambda item: item[0])


def _deterministic_epub_zip(payloads: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        if "mimetype" in payloads:
            _write_deterministic_zip_member(
                archive,
                "mimetype",
                payloads["mimetype"],
                compression=zipfile.ZIP_STORED,
            )
        for member_path in sorted(set(payloads).difference({"mimetype"})):
            _write_deterministic_zip_member(
                archive,
                member_path,
                payloads[member_path],
                compression=zipfile.ZIP_DEFLATED,
            )
    return output.getvalue()


def _write_deterministic_zip_member(
    archive: zipfile.ZipFile,
    member_path: str,
    payload: bytes,
    *,
    compression: int,
) -> None:
    info = zipfile.ZipInfo(member_path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, payload)


def _validated_catalog_root(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("book input directory must be absolute")
    try:
        root_stat = path.lstat()
    except OSError as error:
        raise ValueError("book input directory is unavailable") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("book input must be a real directory")
    return path


def _discover_book_paths(root: Path) -> list[Path]:
    discovered: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(
                os.scandir(directory),
                key=lambda item: unicodedata.normalize("NFC", item.name),
            )
        except OSError as error:
            raise ValueError(f"could not scan book catalog: {directory}") from error
        for entry in entries:
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"could not inspect catalog entry: {entry.path}") from error
            path = Path(entry.path)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ValueError(f"catalog symlink rejected: {path}")
            if stat.S_ISDIR(entry_stat.st_mode):
                if path.suffix.lower() == ".epub":
                    discovered.append(path)
                else:
                    visit(path)
            elif stat.S_ISREG(entry_stat.st_mode) and path.suffix.lower() in {
                ".epub",
                ".fb2",
                ".pdf",
            }:
                discovered.append(path)

    visit(root)
    return sorted(discovered, key=lambda path: path.relative_to(root).as_posix())


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"output path must be a real directory: {path}")
    os.chmod(path, 0o700)


def _ensure_private_subdirectory(root: Path, relative: Path) -> Path:
    current = root
    for component in relative.parts:
        if component in ("", ".", ".."):
            raise ValueError("output subdirectory path is invalid")
        current = current / component
        with suppress(FileExistsError):
            current.mkdir(mode=0o700)
        current_stat = current.lstat()
        if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(current_stat.st_mode):
            raise ValueError(f"output subdirectory must be a real directory: {current}")
        os.chmod(current, 0o700)
    return current


def _atomic_private_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _project_pdf(
    source: BookSource, profile: BookProjectionProfile
) -> tuple[list[_Unit], list[str]]:
    captured = SourceBytes(
        relative_path="source.pdf",
        canonical_uri="file:///dithyramba-book-projection/source.pdf",
        media_type=MediaType.PDF,
        data=source.data,
        content_sha256=sha256_hex(source.data),
        source_modified_at="1970-01-01T00:00:00.000000Z",
        identity=FileIdentity(device=0, inode=0, size=len(source.data), modified_ns=0),
    )
    result = parse_pdf_source(
        captured,
        ParserLimits(
            max_file_bytes=profile.max_file_bytes,
            max_pdf_pages=profile.max_pdf_pages,
            max_extracted_codepoints=profile.max_extracted_codepoints,
            timeout_seconds=profile.timeout_seconds,
            max_rss_mib=profile.max_rss_mib,
        ),
    )
    units: list[_Unit] = []
    for fragment in result.fragments:
        if not isinstance(fragment.address, PdfSourceAddress):
            raise _ProjectionFailure("invalid_pdf_locator", "PDF parser returned a non-PDF locator")
        units.append(
            _Unit(
                kind="page",
                label=f"Page {fragment.address.page}",
                text=fragment.text,
                locator=MappingProxyType(
                    {
                        "bbox": list(fragment.address.bbox),
                        "kind": "pdf",
                        "page": fragment.address.page,
                    }
                ),
            )
        )
    return units, []


def _project_epub(data: bytes, profile: BookProjectionProfile) -> tuple[list[_Unit], list[str]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise _ProjectionFailure("invalid_epub", "EPUB is not a valid ZIP archive") from error
    with archive:
        members = _validate_archive(archive, profile)
        if _read_member(archive, members, "mimetype", profile) != b"application/epub+zip":
            raise _ProjectionFailure("invalid_epub_mimetype", "EPUB mimetype is invalid")
        container = _parse_xml(
            _read_member(archive, members, "META-INF/container.xml", profile),
            profile.max_xml_elements,
        )
        rootfiles = [
            item
            for item in container.iter()
            if _local_name(item.tag) == "rootfile" and item.attrib.get("full-path")
        ]
        if len(rootfiles) != 1:
            raise _ProjectionFailure("invalid_epub_container", "EPUB requires one rootfile")
        opf_path = _resolve_archive_reference("", rootfiles[0].attrib["full-path"])
        opf_root = _parse_xml(
            _read_member(archive, members, opf_path, profile),
            profile.max_xml_elements,
        )

        manifest: dict[str, tuple[str, str]] = {}
        manifest_paths: set[str] = set()
        for item in opf_root.iter():
            if _local_name(item.tag) != "item":
                continue
            item_id = item.attrib.get("id")
            href = item.attrib.get("href")
            media_type = item.attrib.get("media-type")
            if not item_id or not href or not media_type or item_id in manifest:
                raise _ProjectionFailure("invalid_epub_manifest", "EPUB manifest is ambiguous")
            member_path = _resolve_archive_reference(opf_path, href)
            if member_path in manifest_paths:
                raise _ProjectionFailure(
                    "invalid_epub_manifest", "EPUB manifest paths must be unique"
                )
            manifest_paths.add(member_path)
            manifest[item_id] = (member_path, media_type)

        spine_ids: list[str] = []
        for item in opf_root.iter():
            if _local_name(item.tag) == "itemref":
                idref = item.attrib.get("idref")
                if not idref or idref in spine_ids:
                    raise _ProjectionFailure("invalid_epub_spine", "EPUB spine is ambiguous")
                spine_ids.append(idref)
        if not spine_ids:
            raise _ProjectionFailure("invalid_epub_spine", "EPUB spine is empty")

        units: list[_Unit] = []
        warnings: list[str] = []
        for spine_index, item_id in enumerate(spine_ids):
            try:
                member_path, media_type = manifest[item_id]
            except KeyError as error:
                raise _ProjectionFailure(
                    "invalid_epub_spine", "EPUB spine references an absent item"
                ) from error
            if media_type not in ("application/xhtml+xml", "text/html"):
                raise _ProjectionFailure(
                    "unsupported_epub_spine_media", "EPUB spine contains unsupported media"
                )
            document = _parse_xml(
                _read_member(archive, members, member_path, profile),
                profile.max_xml_elements,
                allow_standard_xhtml_doctype=True,
            )
            document_units, dropped_tags = _xml_text_units(
                document,
                locator_base={
                    "archive_path": member_path,
                    "kind": "epub",
                    "spine_index": spine_index,
                },
            )
            warnings.extend(f"dropped_block:{tag}:spine:{spine_index}" for tag in dropped_tags)
            if not document_units:
                warnings.append(f"empty_spine_item:{spine_index}")
            units.extend(document_units)
            _validate_unit_bounds(units, profile)
        return units, warnings


def _project_fb2(data: bytes, profile: BookProjectionProfile) -> tuple[list[_Unit], list[str]]:
    root = _parse_xml(data, profile.max_xml_elements)
    if _local_name(root.tag) != "FictionBook":
        raise _ProjectionFailure("invalid_fb2_root", "FB2 root must be FictionBook")
    units: list[_Unit] = []
    warnings: list[str] = []
    bodies = [element for element in root if _local_name(element.tag) == "body"]
    if not bodies:
        raise _ProjectionFailure("invalid_fb2_body", "FB2 contains no body")
    for body_index, body in enumerate(bodies):
        body_units, dropped_tags = _xml_text_units(
            body,
            locator_base={"body_index": body_index, "kind": "fb2"},
            block_kinds=_FB2_BLOCK_KINDS,
        )
        warnings.extend(f"dropped_block:{tag}:body:{body_index}" for tag in dropped_tags)
        if not body_units:
            warnings.append(f"empty_body:{body_index}")
        units.extend(body_units)
        _validate_unit_bounds(units, profile)
    return units, warnings


def _validate_archive(
    archive: zipfile.ZipFile, profile: BookProjectionProfile
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > profile.max_archive_members:
        raise _ProjectionFailure("archive_member_limit_exceeded", "archive has too many members")
    members: dict[str, zipfile.ZipInfo] = {}
    seen: set[str] = set()
    total = 0
    for info in infos:
        normalized = unicodedata.normalize("NFC", info.filename)
        if normalized in seen:
            raise _ProjectionFailure("duplicate_archive_member", "archive member names collide")
        seen.add(normalized)
        _validate_archive_path(normalized.removesuffix("/") if info.is_dir() else normalized)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise _ProjectionFailure("archive_symlink_rejected", "archive contains a symlink")
        if info.flag_bits & 0x1:
            raise _ProjectionFailure("encrypted_archive_member", "archive member is encrypted")
        if info.file_size > profile.max_archive_member_bytes:
            raise _ProjectionFailure(
                "archive_member_size_limit_exceeded", "archive member is too large"
            )
        total += info.file_size
        if total > profile.max_archive_uncompressed_bytes:
            raise _ProjectionFailure(
                "archive_uncompressed_limit_exceeded", "archive expands beyond its limit"
            )
        if info.file_size and (
            info.compress_size == 0
            or info.file_size > info.compress_size * profile.max_compression_ratio
        ):
            raise _ProjectionFailure(
                "archive_compression_ratio_exceeded", "archive ratio is unsafe"
            )
        if not info.is_dir():
            members[normalized] = info
    return members


def _validate_archive_path(value: str) -> None:
    if not value or "\x00" in value or "\\" in value:
        raise _ProjectionFailure("archive_path_traversal", "archive member path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise _ProjectionFailure("archive_path_traversal", "archive member path escapes root")


def _read_member(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    path: str,
    profile: BookProjectionProfile,
) -> bytes:
    normalized = unicodedata.normalize("NFC", path)
    try:
        info = members[normalized]
    except KeyError as error:
        raise _ProjectionFailure(
            "missing_archive_member", f"missing archive member: {path}"
        ) from error
    try:
        with archive.open(info, "r") as member:
            payload = member.read(profile.max_archive_member_bytes + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _ProjectionFailure("invalid_epub", "could not read EPUB member") from error
    if len(payload) > profile.max_archive_member_bytes or len(payload) != info.file_size:
        raise _ProjectionFailure(
            "archive_member_size_limit_exceeded", "archive member is too large"
        )
    return payload


def _resolve_archive_reference(base_path: str, reference: str) -> str:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or parsed.query:
        raise _ProjectionFailure("external_reference_rejected", "EPUB reference is not local")
    try:
        decoded = unquote(parsed.path, errors="strict")
    except UnicodeError as error:
        raise _ProjectionFailure(
            "invalid_epub_reference", "EPUB reference encoding is invalid"
        ) from error
    if not decoded or "\\" in decoded or "\x00" in decoded:
        raise _ProjectionFailure("invalid_epub_reference", "EPUB reference is invalid")
    base = PurePosixPath(base_path).parent if base_path else PurePosixPath()
    parts: list[str] = []
    for part in (*base.parts, *PurePosixPath(decoded).parts):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise _ProjectionFailure("archive_path_traversal", "EPUB reference escapes root")
            parts.pop()
        elif part == "/":
            raise _ProjectionFailure("archive_path_traversal", "EPUB reference is absolute")
        else:
            parts.append(part)
    resolved = PurePosixPath(*parts).as_posix()
    _validate_archive_path(resolved)
    return unicodedata.normalize("NFC", resolved)


def _parse_xml(
    payload: bytes,
    max_elements: int,
    *,
    allow_standard_xhtml_doctype: bool = False,
) -> Any:
    if type(payload) is not bytes:
        raise TypeError("XML payload must be immutable bytes")
    if allow_standard_xhtml_doctype:
        payload = _strip_standard_xhtml_doctype(payload)
    try:
        root = DefusedElementTree.fromstring(
            payload,
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
    except (DefusedXmlException, DefusedElementTree.ParseError, UnicodeError, ValueError) as error:
        raise _ProjectionFailure("unsafe_or_invalid_xml", "XML is unsafe or invalid") from error
    if sum(1 for _element in root.iter()) > max_elements:
        raise _ProjectionFailure("xml_element_limit_exceeded", "XML has too many elements")
    return root


def _strip_standard_xhtml_doctype(payload: bytes) -> bytes:
    """Strip one known-safe external XHTML declaration without resolving it."""

    lowered = payload.lower()
    marker = b"<!doctype"
    start = lowered.find(marker)
    if start < 0:
        return payload
    if lowered.find(marker, start + len(marker)) >= 0 or b"<!entity" in lowered:
        raise _ProjectionFailure("unsafe_or_invalid_xml", "XML DTD declaration is unsafe")
    if _XML_PROLOG_PREFIX.fullmatch(payload[:start]) is None:
        raise _ProjectionFailure("unsafe_or_invalid_xml", "XML DTD position is invalid")

    quote: int | None = None
    end = -1
    for index in range(start + len(marker), len(payload)):
        value = payload[index]
        if quote is not None:
            if value == quote:
                quote = None
            continue
        if value in (ord('"'), ord("'")):
            quote = value
        elif value == ord("["):
            raise _ProjectionFailure("unsafe_or_invalid_xml", "XML internal subset is forbidden")
        elif value == ord(">"):
            end = index + 1
            break
    if end < 0 or _XHTML_10_STRICT_DOCTYPE.fullmatch(payload[start:end]) is None:
        raise _ProjectionFailure("unsafe_or_invalid_xml", "XML DTD declaration is unsupported")
    return payload[:start] + payload[end:]


def _xml_text_units(
    root: Any,
    *,
    locator_base: dict[str, object],
    block_kinds: Mapping[str, str] = _EPUB_BLOCK_KINDS,
) -> tuple[list[_Unit], tuple[str, ...]]:
    allowed_blocks = frozenset(block_kinds)
    paths = _xml_paths(root)
    elements = list(root.iter())
    units: list[_Unit] = []
    selected_element_ids: set[int] = set()
    covered_element_ids: set[int] = set()
    for element in elements:
        local = _local_name(element.tag)
        if local not in allowed_blocks or id(element) in covered_element_ids:
            continue
        kind = block_kinds[local]
        text = _normalized_text(
            element,
            separate_descendants=kind in {"table", "definition_list"},
        )
        if not text:
            continue
        locator = dict(locator_base)
        locator["xml_path"] = paths[id(element)]
        element_id = element.attrib.get("id")
        if element_id:
            locator["element_id"] = unicodedata.normalize("NFC", element_id)
        units.append(
            _Unit(
                kind=kind,
                label=text if kind == "heading" else _locator_label(locator),
                text=text,
                locator=MappingProxyType(locator),
            )
        )
        selected_element_ids.add(id(element))
        covered_element_ids.update(id(descendant) for descendant in element.iter())
    dropped_tags = _dropped_text_tags(root, selected_element_ids)
    return units, dropped_tags


def _dropped_text_tags(root: Any, selected_element_ids: set[int]) -> tuple[str, ...]:
    """Report text nodes that are not covered by any projected unit."""

    dropped: set[str] = set()
    pending = [root]
    while pending:
        element = pending.pop()
        if id(element) in selected_element_ids:
            continue
        local = _local_name(element.tag)
        if element.text is not None and element.text.strip():
            dropped.add(local)
        children = list(element)
        pending.extend(reversed(children))
        for child in children:
            if child.tail is not None and child.tail.strip():
                dropped.add(local)
    return tuple(sorted(dropped, key=lambda value: value.encode("utf-8")))


def _xml_paths(root: Any) -> dict[int, str]:
    namespace, local = _split_tag(root.tag)
    paths = {id(root): f"/{_path_tag(namespace, local)}[1]"}
    pending = [root]
    while pending:
        element = pending.pop()
        current = paths[id(element)]
        counts: dict[tuple[str, str], int] = {}
        children: list[Any] = []
        for child in element:
            namespace, child_local = _split_tag(child.tag)
            key = (namespace, child_local)
            counts[key] = counts.get(key, 0) + 1
            child_path = f"{current}/{_path_tag(namespace, child_local)}[{counts[key]}]"
            paths[id(child)] = child_path
            children.append(child)
        pending.extend(reversed(children))
    return paths


def _split_tag(tag: Any) -> tuple[str, str]:
    if type(tag) is not str:
        raise _ProjectionFailure("invalid_xml_tag", "XML contains a non-text tag")
    if tag.startswith("{") and "}" in tag:
        namespace, local = tag[1:].split("}", 1)
        return namespace, local
    return "", tag


def _local_name(tag: Any) -> str:
    return _split_tag(tag)[1]


def _path_tag(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}" if namespace else local


def _normalized_text(element: Any, *, separate_descendants: bool = False) -> str:
    parts = tuple(element.itertext())
    text = " ".join((" ".join(parts) if separate_descendants else "".join(parts)).split())
    return unicodedata.normalize("NFC", text).strip()


def _locator_label(locator: dict[str, object]) -> str:
    if locator["kind"] == "epub":
        index = locator["spine_index"]
        if type(index) is not int:
            raise TypeError("spine_index must be int")
        return f"Spine {index + 1}"
    index = locator["body_index"]
    if type(index) is not int:
        raise TypeError("body_index must be int")
    return f"Body {index + 1}"


def _validate_unit_bounds(units: list[_Unit], profile: BookProjectionProfile) -> None:
    if len(units) > profile.max_units:
        raise _ProjectionFailure("unit_limit_exceeded", "book contains too many text units")
    if sum(len(unit.text) for unit in units) > profile.max_extracted_codepoints:
        raise _ProjectionFailure("extracted_text_limit_exceeded", "book text exceeds its limit")


def _render_markdown(source_name: str, units: list[_Unit]) -> tuple[str, list[dict[str, object]]]:
    safe_name = " ".join(source_name.split())
    chunks = [f"# {safe_name}\n\n"]
    offset = len(chunks[0])
    sidecar_units: list[dict[str, object]] = []
    for ordinal, unit in enumerate(units):
        prefix = "## " if unit.kind == "heading" else ""
        chunks.append(prefix)
        offset += len(prefix)
        text_start = offset
        chunks.append(unit.text)
        offset += len(unit.text)
        text_end = offset
        chunks.append("\n\n")
        offset += 2
        sidecar_units.append(
            {
                "kind": unit.kind,
                "markdown_char_end": text_end,
                "markdown_char_start": text_start,
                "ordinal": ordinal,
                "source_locator": dict(unit.locator),
                "source_text_sha256": sha256_hex(unit.text.encode("utf-8")),
            }
        )
    return "".join(chunks), sidecar_units


def _base_manifest(
    source: BookSource,
    profile: BookProjectionProfile,
    source_hash: str,
    status: BookProjectionStatus,
) -> dict[str, object]:
    source_payload: dict[str, object] = {
        "byte_size": len(source.data),
        "form": source.source_form,
        "media_type": source.media_type.value,
        "name": source.name,
        "sha256": source_hash,
    }
    if source.source_form == "expanded_epub_repack":
        source_payload["members"] = [
            {"byte_size": member.byte_size, "path": member.path, "sha256": member.sha256}
            for member in source.members
        ]
    return {
        "connector_revision": BOOK_CONNECTOR_REVISION,
        "metadata": {"roles": list(source.roles)},
        "optional_sidecars": {
            "docling": {"contract": _DOCLING_SIDECAR_CONTRACT, "status": "not_run"}
        },
        "profile": {"id": profile.profile_id},
        "schema": BOOK_PROJECTION_SCHEMA,
        "source": source_payload,
        "status": status.value,
    }


def _failure_projection(
    source: BookSource,
    profile: BookProjectionProfile,
    source_hash: str,
    code: str,
) -> BookProjection:
    manifest = _base_manifest(source, profile, source_hash, BookProjectionStatus.FAILURE)
    manifest.update(
        {
            "failure": {"code": code},
            "markdown": {"codepoints": 0, "sha256": sha256_hex(b"")},
            "parser_revision": BOOK_CONNECTOR_REVISION,
            "units": [],
            "warnings": [],
        }
    )
    return BookProjection(
        status=BookProjectionStatus.FAILURE,
        markdown="",
        sidecar_json=canonical_json_bytes(manifest),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the bounded catalog projection module."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("default", "large"), default="large")
    parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        help="manifest role; repeatable (default: private)",
    )
    arguments = parser.parse_args(argv)
    profile = LARGE_BOOK_PROFILE if arguments.profile == "large" else DEFAULT_BOOK_PROFILE
    roles = tuple(arguments.roles) if arguments.roles else ("private",)
    try:
        summary = project_book_catalog(
            arguments.input_dir,
            arguments.output_dir,
            profile=profile,
            roles=roles,
        )
    except (OSError, ValueError) as error:
        print(f"book projection failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "complete": summary.complete,
                "failure": summary.failure,
                "partial": summary.partial,
                "total": summary.total,
            }
        )
        + b"\n"
    )
    return 0 if summary.failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
