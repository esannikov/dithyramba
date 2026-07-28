"""Private descriptor-safe content-addressed blob storage."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import re
import secrets
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Literal
from weakref import WeakValueDictionary

from dithyramba.contracts import sha256_hex
from dithyramba.library import LibraryPaths, validate_library_layout

from .errors import BlobIntegrityError, BlobStoreError
from .models import PromotedBlob

_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_READ_CHUNK = 1024 * 1024
_CONTROLLED_TEMP_PATTERN = re.compile(r"^\.blob-[0-9a-f]{32}\.tmp$")
_LOCK_NAMESPACES = frozenset({"blobs", "sources"})
_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: WeakValueDictionary[str, threading.Lock] = WeakValueDictionary()


class PrivateLockStore:
    """Persistent private advisory locks scoped to one Library installation."""

    def __init__(self, paths: LibraryPaths) -> None:
        if not isinstance(paths, LibraryPaths):
            raise TypeError("PrivateLockStore requires verified LibraryPaths")
        self._paths = paths

    @contextmanager
    def hold(
        self,
        namespace: Literal["blobs", "sources"],
        key: str,
    ) -> Iterator[None]:
        """Hold one cross-process lock until the protected operation completes."""

        if namespace not in _LOCK_NAMESPACES:
            raise ValueError("private lock namespace is unsupported")
        if type(key) is not str or not key:
            raise ValueError("private lock key must be non-empty text")
        validate_library_layout(self._paths)
        key_hash = sha256_hex(key.encode("utf-8"))
        process_lock = _process_lock(f"{self._paths.root}:{namespace}:{key_hash}")
        process_lock.acquire()
        cache_descriptor: int | None = None
        locks_descriptor: int | None = None
        namespace_descriptor: int | None = None
        lock_descriptor: int | None = None
        acquired = False
        try:
            cache_descriptor = _open_private_directory(self._paths.cache)
            locks_descriptor, locks_created = _open_or_create_private_directory_name(
                cache_descriptor,
                "locks",
            )
            if locks_created:
                os.fsync(cache_descriptor)
            namespace_descriptor, namespace_created = _open_or_create_private_directory_name(
                locks_descriptor,
                namespace,
            )
            if namespace_created:
                os.fsync(locks_descriptor)
            leaf = f"{key_hash}.lock"
            lock_descriptor, lock_created = _open_or_create_private_lock(
                namespace_descriptor,
                leaf,
            )
            if lock_created:
                os.fsync(namespace_descriptor)
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            acquired = True
            _verify_locked_name(namespace_descriptor, leaf, lock_descriptor)
            yield
        except (BlobStoreError, BlobIntegrityError, TypeError, ValueError):
            raise
        except OSError as exc:
            raise BlobStoreError("private Library lock operation failed") from exc
        finally:
            if lock_descriptor is not None:
                if acquired:
                    with suppress(OSError):
                        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            if namespace_descriptor is not None:
                os.close(namespace_descriptor)
            if locks_descriptor is not None:
                os.close(locks_descriptor)
            if cache_descriptor is not None:
                os.close(cache_descriptor)
            process_lock.release()


def _process_lock(identity: str) -> threading.Lock:
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(identity, threading.Lock())


class BlobStore:
    """Promote and verify immutable bytes below one Library's blob root."""

    def __init__(self, paths: LibraryPaths) -> None:
        if not isinstance(paths, LibraryPaths):
            raise TypeError("BlobStore requires verified LibraryPaths")
        self._paths = paths
        self._locks = PrivateLockStore(paths)

    def promote(self, data: bytes, content_sha256: str) -> PromotedBlob:
        """Atomically publish exact bytes without replacing an existing digest."""

        if type(data) is not bytes:
            raise TypeError("blob data must be immutable bytes")
        if sha256_hex(data) != content_sha256:
            raise BlobIntegrityError("blob bytes do not match content_sha256")
        validate_library_layout(self._paths)
        with self._locks.hold("blobs", content_sha256):
            return self._promote_locked(data, content_sha256)

    def _promote_locked(self, data: bytes, content_sha256: str) -> PromotedBlob:
        """Publish bytes while the digest-specific cross-process lock is held."""

        shard = content_sha256[:2]
        leaf = content_sha256[2:]
        relative_path = f"{shard}/{leaf}"
        root_descriptor: int | None = None
        shard_descriptor: int | None = None
        temporary_name: str | None = None
        created = False
        try:
            root_descriptor = _open_private_directory(self._paths.blobs)
            shard_descriptor, shard_created = _open_or_create_shard(root_descriptor, shard)
            if shard_created:
                os.fsync(root_descriptor)

            try:
                existing_descriptor = _open_blob(shard_descriptor, leaf)
            except FileNotFoundError:
                temporary_name = f".blob-{secrets.token_hex(16)}.tmp"
                temporary_descriptor = os.open(
                    temporary_name,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                    _FILE_MODE,
                    dir_fd=shard_descriptor,
                )
                try:
                    _write_all(temporary_descriptor, data)
                    os.fsync(temporary_descriptor)
                    _verify_blob_descriptor(
                        temporary_descriptor,
                        expected_sha256=content_sha256,
                        expected_size=len(data),
                    )
                    try:
                        os.link(
                            temporary_name,
                            leaf,
                            src_dir_fd=shard_descriptor,
                            dst_dir_fd=shard_descriptor,
                            follow_symlinks=False,
                        )
                        created = True
                    except FileExistsError:
                        created = False
                finally:
                    os.close(temporary_descriptor)
                    with suppress(FileNotFoundError):
                        os.unlink(temporary_name, dir_fd=shard_descriptor)
                    temporary_name = None
                os.fsync(shard_descriptor)
            else:
                os.close(existing_descriptor)

            final_descriptor = _open_blob(shard_descriptor, leaf)
            try:
                _verify_blob_descriptor(
                    final_descriptor,
                    expected_sha256=content_sha256,
                    expected_size=len(data),
                )
            finally:
                os.close(final_descriptor)
        except (BlobStoreError, TypeError):
            raise
        except OSError as exc:
            raise BlobStoreError("content-addressed blob promotion failed") from exc
        finally:
            if temporary_name is not None and shard_descriptor is not None:
                with suppress(OSError):
                    os.unlink(temporary_name, dir_fd=shard_descriptor)
            if shard_descriptor is not None:
                os.close(shard_descriptor)
            if root_descriptor is not None:
                os.close(root_descriptor)

        path = self._paths.blobs / shard / leaf
        return PromotedBlob(content_sha256, len(data), relative_path, path, created)

    def verify(self, blob: PromotedBlob) -> None:
        """Fail closed unless a promoted record resolves to exact private bytes."""

        if not isinstance(blob, PromotedBlob):
            raise TypeError("blob verification requires PromotedBlob")
        expected_path = self._paths.blobs / blob.relative_path
        if blob.path != expected_path:
            raise BlobIntegrityError("blob record escaped its Library blob root")
        validate_library_layout(self._paths)
        with self._locks.hold("blobs", blob.content_sha256):
            self._verify_locked(blob)

    def _verify_locked(self, blob: PromotedBlob) -> None:
        """Verify exact bytes while the digest lock excludes publication recovery."""

        root_descriptor = _open_private_directory(self._paths.blobs)
        try:
            shard_descriptor = _open_private_directory_name(
                root_descriptor, blob.content_sha256[:2]
            )
            try:
                descriptor = _open_blob(shard_descriptor, blob.content_sha256[2:])
                try:
                    _verify_blob_descriptor(
                        descriptor,
                        expected_sha256=blob.content_sha256,
                        expected_size=blob.byte_size,
                    )
                finally:
                    os.close(descriptor)
            finally:
                os.close(shard_descriptor)
        except FileNotFoundError as exc:
            raise BlobIntegrityError("content-addressed blob is missing") from exc
        except OSError as exc:
            raise BlobIntegrityError("content-addressed blob cannot be verified") from exc
        finally:
            os.close(root_descriptor)


def _open_or_create_shard(root_descriptor: int, shard: str) -> tuple[int, bool]:
    return _open_or_create_private_directory_name(root_descriptor, shard)


def _open_or_create_private_directory_name(
    root_descriptor: int,
    name: str,
) -> tuple[int, bool]:
    try:
        return _open_private_directory_name(root_descriptor, name), False
    except FileNotFoundError:
        try:
            os.mkdir(name, _DIRECTORY_MODE, dir_fd=root_descriptor)
            created = True
        except FileExistsError:
            created = False
        descriptor = _open_private_directory_name(root_descriptor, name)
        if created:
            os.fchmod(descriptor, _DIRECTORY_MODE)
            _require_private_directory_descriptor(descriptor)
        return descriptor, created


def _open_or_create_private_lock(parent_descriptor: int, name: str) -> tuple[int, bool]:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            name, flags | os.O_CREAT | os.O_EXCL, _FILE_MODE, dir_fd=parent_descriptor
        )
        created = True
    except FileExistsError:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        created = False
    try:
        if created:
            os.fchmod(descriptor, _FILE_MODE)
        _require_private_lock_descriptor(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, created


def _require_private_lock_descriptor(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise BlobIntegrityError("private lock must be one regular file")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
    ):
        raise BlobIntegrityError("private lock file is not private")


def _verify_locked_name(parent_descriptor: int, name: str, descriptor: int) -> None:
    _require_private_lock_descriptor(descriptor)
    descriptor_metadata = os.fstat(descriptor)
    try:
        name_metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise BlobIntegrityError("private lock name changed while acquiring it") from exc
    if (
        not stat.S_ISREG(name_metadata.st_mode)
        or name_metadata.st_dev != descriptor_metadata.st_dev
        or name_metadata.st_ino != descriptor_metadata.st_ino
    ):
        raise BlobIntegrityError("private lock name changed while acquiring it")


def _open_private_directory(path: Path) -> int:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        _require_private_directory_descriptor(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_private_directory_name(parent_descriptor: int, name: str) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    try:
        _require_private_directory_descriptor(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _require_private_directory_descriptor(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise BlobIntegrityError("blob directory is not a real directory")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != _DIRECTORY_MODE
    ):
        raise BlobIntegrityError("blob directory is not private")


def _open_blob(shard_descriptor: int, leaf: str) -> int:
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=shard_descriptor,
        )
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise FileNotFoundError(leaf) from exc
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BlobIntegrityError("blob must be one private regular file")
        if os.name == "posix" and (
            metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
        ):
            raise BlobIntegrityError("blob file is not private")
        _recover_controlled_stale_links(shard_descriptor, descriptor)
        if os.fstat(descriptor).st_nlink != 1:
            raise BlobIntegrityError("blob must be one private regular file")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _recover_controlled_stale_links(shard_descriptor: int, final_descriptor: int) -> None:
    """Remove only private controlled temp names that alias the locked final inode."""

    final_metadata = os.fstat(final_descriptor)
    if final_metadata.st_nlink == 1:
        return
    removed = False
    for name in os.listdir(shard_descriptor):
        if _CONTROLLED_TEMP_PATTERN.fullmatch(name) is None:
            continue
        try:
            candidate = os.stat(name, dir_fd=shard_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if candidate.st_dev != final_metadata.st_dev or candidate.st_ino != final_metadata.st_ino:
            continue
        if not stat.S_ISREG(candidate.st_mode) or (
            os.name == "posix"
            and (candidate.st_uid != os.geteuid() or stat.S_IMODE(candidate.st_mode) != _FILE_MODE)
        ):
            raise BlobIntegrityError("controlled blob temporary link is not private")
        os.unlink(name, dir_fd=shard_descriptor)
        removed = True
    if removed:
        os.fsync(shard_descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise BlobStoreError("short write while promoting blob")
        offset += written


def _verify_blob_descriptor(
    descriptor: int,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise BlobIntegrityError("blob must be one private regular file")
    if metadata.st_size != expected_size:
        raise BlobIntegrityError("blob byte size does not match its DB candidate")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != _FILE_MODE
    ):
        raise BlobIntegrityError("blob file is not private")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, _READ_CHUNK):
        digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise BlobIntegrityError("blob bytes do not match their content address")
