"""Descriptor-first discovery and immutable source-byte capture."""

from __future__ import annotations

import errno
import fnmatch
import os
import stat
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import NoReturn

from dithyramba.collections import CollectionRoot
from dithyramba.contracts import sha256_hex

from .errors import (
    FileSizeLimitExceededError,
    InvalidSourcePathError,
    SourceChangedDuringIngestError,
    SourceNotRegularError,
    SourceSymlinkError,
    SourceUnavailableError,
    UnsupportedFormatError,
)
from .models import FileIdentity, MediaType, ParserLimits, SourceBytes

_READ_CHUNK_BYTES = 1024 * 1024
_SUPPORTED_MEDIA_TYPES = {
    ".md": MediaType.MARKDOWN,
    ".markdown": MediaType.MARKDOWN,
    ".txt": MediaType.PLAIN_TEXT,
    ".pdf": MediaType.PDF,
}


def discover_source_paths(root: CollectionRoot) -> tuple[str, ...]:
    """Return deterministic, glob-filtered regular files below ``root``.

    Directory symlinks are never traversed and symlink/non-regular leaf entries
    are never returned. The subsequent byte reader repeats all checks against
    descriptors; discovery is deliberately not an authorization decision.
    """

    directory_chain = _open_configured_root_chain(root.path)
    try:
        discovered: list[str] = []
        _discover_directory(directory_chain[-1].descriptor, (), root, discovered)
        _revalidate_directory_chain(directory_chain)
        return tuple(sorted(discovered))
    finally:
        _close_directory_chain(directory_chain)


def read_source_bytes(
    root: CollectionRoot,
    relative_path: str,
    limits: ParserLimits,
) -> SourceBytes:
    """Capture one source using no-follow descriptor traversal and bounded I/O."""

    parts = _validate_relative_path(relative_path)
    media_type = _media_type(relative_path)
    directory_chain: list[_DirectoryHandle] = []
    file_fd: int | None = None
    try:
        directory_chain.extend(_open_configured_root_chain(root.path))
        for component in parts[:-1]:
            parent_fd = directory_chain[-1].descriptor
            child_fd = _open_child_directory(parent_fd, component)
            directory_chain.append(_capture_directory(child_fd, parent_fd, component))

        parent_fd = directory_chain[-1].descriptor
        final_name = parts[-1]
        file_fd = _open_regular_file(parent_fd, final_name)
        before_stat = os.fstat(file_fd)
        if not stat.S_ISREG(before_stat.st_mode):
            raise SourceNotRegularError(f"source is not a regular file: {relative_path}")
        before = _identity(before_stat)
        if before.size > limits.max_file_bytes:
            raise FileSizeLimitExceededError(
                f"source exceeds {limits.max_file_bytes} bytes: {relative_path}"
            )

        data = _bounded_read(file_fd, limits.max_file_bytes)
        after_descriptor = _identity(os.fstat(file_fd))
        after_entry = _entry_identity_after_read(parent_fd, final_name)
        if before != after_descriptor or before != after_entry or len(data) != before.size:
            raise SourceChangedDuringIngestError(
                f"source changed while it was being read: {relative_path}"
            )
        _revalidate_directory_chain(directory_chain)

        return SourceBytes(
            relative_path=relative_path,
            canonical_uri=(root.path / Path(*parts)).as_uri(),
            media_type=media_type,
            data=data,
            content_sha256=sha256_hex(data),
            source_modified_at=_format_modified_ns(before.modified_ns),
            identity=before,
        )
    finally:
        if file_fd is not None:
            os.close(file_fd)
        _close_directory_chain(directory_chain)


def validate_source_unchanged(
    root: CollectionRoot,
    captured: SourceBytes,
    limits: ParserLimits,
) -> None:
    """Fail if an exact post-parse reread differs from ``captured``.

    The reread receives the same descriptor-chain protection as the initial
    capture. Path disappearance, replacement, or a newly introduced symlink is
    reported as one retryable source-change failure at this validation stage.
    """

    try:
        current = read_source_bytes(root, captured.relative_path, limits)
    except SourceChangedDuringIngestError:
        raise
    except (
        FileSizeLimitExceededError,
        SourceNotRegularError,
        SourceSymlinkError,
        SourceUnavailableError,
    ) as error:
        raise SourceChangedDuringIngestError(
            f"source changed after capture: {captured.relative_path}"
        ) from error
    if (
        current.relative_path != captured.relative_path
        or current.canonical_uri != captured.canonical_uri
        or current.media_type is not captured.media_type
        or current.identity != captured.identity
        or current.content_sha256 != captured.content_sha256
        or current.data != captured.data
    ):
        raise SourceChangedDuringIngestError(
            f"source changed after capture: {captured.relative_path}"
        )


@dataclass(frozen=True, slots=True)
class _DirectoryHandle:
    """One opened directory plus its stable edge from the parent descriptor."""

    descriptor: int
    parent_descriptor: int | None
    component: str | None
    device: int
    inode: int


def _discover_directory(
    directory_fd: int,
    parent_parts: tuple[str, ...],
    root: CollectionRoot,
    discovered: list[str],
) -> None:
    try:
        with os.scandir(directory_fd) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
    except OSError as error:
        raise SourceUnavailableError("could not scan Collection root") from error

    for entry in ordered:
        relative_parts = (*parent_parts, entry.name)
        relative_path = PurePosixPath(*relative_parts).as_posix()
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise SourceUnavailableError(
                f"could not inspect source entry: {relative_path}"
            ) from error
        if stat.S_ISLNK(entry_stat.st_mode):
            continue
        if stat.S_ISDIR(entry_stat.st_mode):
            child_fd = _open_child_directory(directory_fd, entry.name)
            try:
                _discover_directory(child_fd, relative_parts, root, discovered)
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(entry_stat.st_mode):
            continue
        if _is_included(relative_path, root.include_globs, root.exclude_globs):
            discovered.append(relative_path)


def _is_included(
    relative_path: str,
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
) -> bool:
    return any(_glob_matches(relative_path, pattern) for pattern in include_globs) and not any(
        _glob_matches(relative_path, pattern) for pattern in exclude_globs
    )


def _glob_matches(relative_path: str, pattern: str) -> bool:
    """Match NFC-equivalent path globs with ``**`` spanning path segments.

    macOS commonly exposes filenames in decomposed Unicode even when persisted
    Collection configuration has been canonicalized to NFC.  Matching the raw
    code-point sequences would silently omit canonically equivalent sources.
    The original filesystem spelling is still returned for descriptor reads.
    """

    path_parts = tuple(unicodedata.normalize("NFC", relative_path).split("/"))
    pattern_parts = tuple(unicodedata.normalize("NFC", pattern).split("/"))

    def matches(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        current = pattern_parts[pattern_index]
        if current == "**":
            return matches(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and matches(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], current)
            and matches(path_index + 1, pattern_index + 1)
        )

    return matches(0, 0)


def _validate_relative_path(relative_path: str) -> tuple[str, ...]:
    if type(relative_path) is not str or not relative_path or "\x00" in relative_path:
        raise InvalidSourcePathError("source path must be non-empty text without NUL")
    if "\\" in relative_path:
        raise InvalidSourcePathError("source path must use POSIX separators")
    path = PurePosixPath(relative_path)
    parts = path.parts
    if path.is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise InvalidSourcePathError("source path must be normalized and relative")
    if path.as_posix() != relative_path:
        raise InvalidSourcePathError("source path must be normalized POSIX text")
    return parts


def _media_type(relative_path: str) -> MediaType:
    suffix = PurePosixPath(relative_path).suffix
    try:
        return _SUPPORTED_MEDIA_TYPES[suffix]
    except KeyError as error:
        raise UnsupportedFormatError(f"unsupported source format: {suffix or '<none>'}") from error


def _open_configured_root_chain(path: Path) -> list[_DirectoryHandle]:
    """Open an absolute root one no-follow component at a time from ``/``."""

    if not path.is_absolute() or ".." in path.parts:
        raise SourceUnavailableError("Collection root must be an absolute normalized path")
    chain: list[_DirectoryHandle] = []
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | _no_follow_flag()
    try:
        descriptor = os.open(os.sep, flags)
    except OSError as error:
        _raise_open_error(error, os.sep, directory=True)
    chain.append(_capture_directory(descriptor, None, None))
    try:
        for component in path.parts[1:]:
            parent_fd = chain[-1].descriptor
            child_fd = _open_child_directory(parent_fd, component)
            chain.append(_capture_directory(child_fd, parent_fd, component))
    except Exception:
        _close_directory_chain(chain)
        raise
    return chain


def _capture_directory(
    descriptor: int,
    parent_descriptor: int | None,
    component: str | None,
) -> _DirectoryHandle:
    try:
        directory_stat = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise SourceUnavailableError("could not inspect opened directory") from error
    if not stat.S_ISDIR(directory_stat.st_mode):
        os.close(descriptor)
        raise SourceNotRegularError("opened source path component is not a directory")
    return _DirectoryHandle(
        descriptor=descriptor,
        parent_descriptor=parent_descriptor,
        component=component,
        device=directory_stat.st_dev,
        inode=directory_stat.st_ino,
    )


def _revalidate_directory_chain(chain: list[_DirectoryHandle]) -> None:
    """Verify every held directory and every parent-to-child namespace edge."""

    for handle in chain:
        try:
            descriptor_stat = os.fstat(handle.descriptor)
        except OSError as error:
            raise SourceChangedDuringIngestError(
                "source directory chain changed during read"
            ) from error
        if not stat.S_ISDIR(descriptor_stat.st_mode) or (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ) != (handle.device, handle.inode):
            raise SourceChangedDuringIngestError("source directory chain changed during read")
        if handle.parent_descriptor is None or handle.component is None:
            continue
        try:
            entry_stat = os.stat(
                handle.component,
                dir_fd=handle.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise SourceChangedDuringIngestError(
                "source directory chain changed during read"
            ) from error
        if not stat.S_ISDIR(entry_stat.st_mode) or (entry_stat.st_dev, entry_stat.st_ino) != (
            handle.device,
            handle.inode,
        ):
            raise SourceChangedDuringIngestError("source directory chain changed during read")


def _close_directory_chain(chain: list[_DirectoryHandle]) -> None:
    for handle in reversed(chain):
        os.close(handle.descriptor)


def _open_child_directory(parent_fd: int, component: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | _no_follow_flag()
    try:
        descriptor = os.open(component, flags, dir_fd=parent_fd)
    except OSError as error:
        if _entry_is_symlink(parent_fd, component):
            raise SourceSymlinkError(f"symlink source is forbidden: {component}") from error
        _raise_open_error(error, component, directory=True)
    try:
        child_stat = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise SourceUnavailableError(f"could not inspect directory: {component}") from error
    if not stat.S_ISDIR(child_stat.st_mode):
        os.close(descriptor)
        raise SourceNotRegularError(f"source path component is not a directory: {component}")
    return descriptor


def _open_regular_file(parent_fd: int, final_name: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | _no_follow_flag()
    try:
        return os.open(final_name, flags, dir_fd=parent_fd)
    except OSError as error:
        if _entry_is_symlink(parent_fd, final_name):
            raise SourceSymlinkError(f"symlink source is forbidden: {final_name}") from error
        _raise_open_error(error, final_name, directory=False)


def _entry_is_symlink(parent_fd: int, name: str) -> bool:
    try:
        entry_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISLNK(entry_stat.st_mode)


def _raise_open_error(error: OSError, path: str, *, directory: bool) -> NoReturn:
    if error.errno in (errno.ELOOP, errno.EMLINK):
        raise SourceSymlinkError(f"symlink source is forbidden: {path}") from error
    if error.errno in (errno.ENOTDIR, errno.EISDIR):
        kind = "directory" if directory else "regular file"
        raise SourceNotRegularError(f"source is not the required {kind}: {path}") from error
    raise SourceUnavailableError(f"source is unavailable: {path}") from error


def _no_follow_flag() -> int:
    try:
        return os.O_NOFOLLOW
    except AttributeError as error:  # pragma: no cover - VS0 targets POSIX/macOS
        raise SourceUnavailableError("this platform lacks O_NOFOLLOW") from error


def _bounded_read(file_fd: int, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        request_size = min(_READ_CHUNK_BYTES, max_bytes - total + 1)
        try:
            chunk = os.read(file_fd, request_size)
        except OSError as error:
            raise SourceUnavailableError("source read failed") from error
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise FileSizeLimitExceededError(f"source exceeds {max_bytes} bytes during read")
    return b"".join(chunks)


def _entry_identity_after_read(parent_fd: int, final_name: str) -> FileIdentity:
    try:
        entry_stat = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise SourceChangedDuringIngestError(
            "source directory entry changed during read"
        ) from error
    if not stat.S_ISREG(entry_stat.st_mode):
        raise SourceChangedDuringIngestError("source directory entry changed during read")
    return _identity(entry_stat)


def _identity(file_stat: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size=file_stat.st_size,
        modified_ns=file_stat.st_mtime_ns,
    )


def _format_modified_ns(modified_ns: int) -> str:
    seconds, nanoseconds = divmod(modified_ns, 1_000_000_000)
    timestamp = datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{timestamp}.{nanoseconds // 1_000:06d}Z"
