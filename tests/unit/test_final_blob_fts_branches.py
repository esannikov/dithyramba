from __future__ import annotations

import errno
import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from dithyramba.contracts import sha256_hex
from dithyramba.library import LibraryConfig
from dithyramba.persistence import initialize_library
from dithyramba.provenance import BlobIntegrityError, BlobStore, BlobStoreError
from dithyramba.provenance import blobs as blobs_module
from dithyramba.provenance.blobs import PrivateLockStore
from dithyramba.recall import (
    MAX_QUERY_TOKENS,
    FtsCandidate,
    FtsContractError,
    FtsFragment,
    FtsRuntimeProfile,
    FtsSearchResult,
    FtsUnavailableError,
    current_fts_runtime_profile,
    search_ephemeral_fts,
)
from dithyramba.recall import fts as fts_module

_VALID_HASH = "a" * 64


def _profile() -> FtsRuntimeProfile:
    return FtsRuntimeProfile(sqlite_version="3.45.0", compile_options_hash=_VALID_HASH)


def _result() -> FtsSearchResult:
    return FtsSearchResult(
        retrieval_corpus_hash=_VALID_HASH,
        query_tokens=("alpha",),
        max_candidates=10,
        candidate_count=0,
        trace=(),
        profile=_profile(),
    )


def test_fts_leaf_contracts_reject_wrong_types_and_noncanonical_values() -> None:
    with pytest.raises(FtsContractError, match="text must be str"):
        FtsFragment("fragment_a", cast(Any, b"alpha"), _VALID_HASH)

    with pytest.raises(FtsContractError, match="positive integer"):
        FtsCandidate("fragment_a", cast(Any, False), "0.000000")

    with pytest.raises(FtsContractError, match="numeric version"):
        FtsRuntimeProfile(cast(Any, 3), _VALID_HASH)

    with pytest.raises(FtsContractError, match="lowercase SHA-256"):
        FtsRuntimeProfile("3.45.0", "A" * 64)

    with pytest.raises(FtsContractError, match="six-decimal"):
        FtsCandidate("fragment_a", 1, "1")

    with pytest.raises(FtsContractError, match="negative zero"):
        FtsCandidate("fragment_a", 1, "-0.000000")


def test_fts_search_result_rejects_each_structural_corruption() -> None:
    base = _result()

    with pytest.raises(FtsContractError, match="query_tokens"):
        replace(base, query_tokens=cast(Any, ["alpha"]))

    too_many_tokens = tuple(f"token{index}" for index in range(MAX_QUERY_TOKENS + 1))
    with pytest.raises(FtsContractError, match="token count"):
        replace(base, query_tokens=too_many_tokens)

    with pytest.raises(FtsContractError, match="unique"):
        replace(base, query_tokens=("alpha", "alpha"))

    with pytest.raises(FtsContractError, match="tuple of FtsCandidate"):
        replace(base, trace=cast(Any, [FtsCandidate("fragment_a", 1, "0.000000")]))

    with pytest.raises(FtsContractError, match="contiguous"):
        replace(
            base,
            candidate_count=1,
            trace=(FtsCandidate("fragment_a", 2, "0.000000"),),
        )

    with pytest.raises(FtsContractError, match="FtsRuntimeProfile"):
        replace(base, profile=cast(Any, object()))


def test_fts_question_must_be_text_before_sqlite_is_open() -> None:
    with pytest.raises(FtsContractError, match="question must be str"):
        search_ephemeral_fts(
            question=cast(Any, b"alpha"),
            fragments=(),
            max_candidates=1,
        )


def test_fts_connection_and_profile_failures_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(_database: str) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database unavailable")

    with monkeypatch.context() as patch:
        patch.setattr("dithyramba.recall.fts.sqlite3.connect", fail_connect)
        with pytest.raises(FtsUnavailableError, match="ephemeral SQLite"):
            search_ephemeral_fts(question="alpha", fragments=(), max_candidates=1)
        with pytest.raises(FtsUnavailableError, match="profile probe"):
            current_fts_runtime_profile()

    def fail_create(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("FTS5 unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(fts_module, "_create_fts_tables", fail_create)
        with pytest.raises(FtsUnavailableError, match="FTS5 profile"):
            current_fts_runtime_profile()


def test_private_lock_constructor_namespace_and_key_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="LibraryPaths"):
        PrivateLockStore(cast(Any, object()))

    with initialize_library(
        LibraryConfig(name="Lock contract branches"), data_root=tmp_path / "data"
    ) as repository:
        locks = PrivateLockStore(repository.paths)
        with pytest.raises(ValueError, match="namespace"), locks.hold(cast(Any, "reviews"), "key"):
            raise AssertionError("invalid namespace entered the lock")
        with pytest.raises(ValueError, match="key"), locks.hold("blobs", cast(Any, b"key")):
            raise AssertionError("invalid key entered the lock")


def test_private_lock_releases_process_lock_when_cache_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialize_library(
        LibraryConfig(name="Lock cache failure"), data_root=tmp_path / "data"
    ) as repository:
        locks = PrivateLockStore(repository.paths)

        def fail_open(_path: Path) -> int:
            raise OSError(errno.EIO, "cache unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(blobs_module, "_open_private_directory", fail_open)
            with (
                pytest.raises(BlobStoreError, match="lock operation"),
                locks.hold("blobs", "same-key"),
            ):
                raise AssertionError("failed cache open entered the lock")

        with locks.hold("blobs", "same-key"):
            pass


def test_private_lock_closes_descriptors_when_flock_acquisition_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialize_library(
        LibraryConfig(name="Lock flock failure"), data_root=tmp_path / "data"
    ) as repository:
        locks = PrivateLockStore(repository.paths)

        def fail_flock(_descriptor: int, _operation: int) -> None:
            raise OSError(errno.EIO, "flock unavailable")

        with monkeypatch.context() as patch:
            patch.setattr("dithyramba.provenance.blobs.fcntl.flock", fail_flock)
            with (
                pytest.raises(BlobStoreError, match="lock operation"),
                locks.hold("sources", "same-key"),
            ):
                raise AssertionError("failed flock entered the lock")

        with locks.hold("sources", "same-key"):
            pass


def test_lock_descriptor_and_locked_name_reject_filesystem_substitution(tmp_path: Path) -> None:
    read_descriptor, write_descriptor = os.pipe()
    try:
        with pytest.raises(BlobIntegrityError, match="regular file"):
            blobs_module._require_private_lock_descriptor(read_descriptor)
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)

    public_lock = tmp_path / "public.lock"
    public_lock.write_bytes(b"")
    public_lock.chmod(0o644)
    descriptor = os.open(public_lock, os.O_RDONLY)
    try:
        with pytest.raises(BlobIntegrityError, match="not private"):
            blobs_module._require_private_lock_descriptor(descriptor)
    finally:
        os.close(descriptor)

    lock_directory = tmp_path / "locks"
    lock_directory.mkdir(mode=0o700)
    first = lock_directory / "first.lock"
    second = lock_directory / "second.lock"
    first.write_bytes(b"")
    second.write_bytes(b"")
    first.chmod(0o600)
    second.chmod(0o600)
    parent_descriptor = os.open(lock_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    first_descriptor = os.open(first, os.O_RDONLY)
    try:
        with pytest.raises(BlobIntegrityError, match="name changed"):
            blobs_module._verify_locked_name(
                parent_descriptor,
                second.name,
                first_descriptor,
            )
    finally:
        os.close(first_descriptor)
        os.close(parent_descriptor)


def test_open_blob_rejects_symlink_and_nonregular_leaf(tmp_path: Path) -> None:
    shard = tmp_path / "shard"
    shard.mkdir(mode=0o700)
    target = shard / "target"
    target.write_bytes(b"target")
    target.chmod(0o600)
    (shard / "link").symlink_to(target)
    (shard / "directory").mkdir(mode=0o700)
    shard_descriptor = os.open(shard, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(OSError) as symlink_error:
            blobs_module._open_blob(shard_descriptor, "link")
        assert symlink_error.value.errno != errno.ENOENT

        with pytest.raises(BlobIntegrityError, match="regular file"):
            blobs_module._open_blob(shard_descriptor, "directory")
    finally:
        os.close(shard_descriptor)


def test_stale_link_recovery_ignores_unrelated_controlled_name(tmp_path: Path) -> None:
    shard = tmp_path / "shard"
    shard.mkdir(mode=0o700)
    final = shard / "final"
    final.write_bytes(b"final")
    final.chmod(0o600)
    os.link(final, shard / "external-alias")
    unrelated = shard / f".blob-{'a' * 32}.tmp"
    unrelated.write_bytes(b"unrelated")
    unrelated.chmod(0o600)
    shard_descriptor = os.open(shard, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    final_descriptor = os.open(final, os.O_RDONLY)
    try:
        blobs_module._recover_controlled_stale_links(shard_descriptor, final_descriptor)
    finally:
        os.close(final_descriptor)
        os.close(shard_descriptor)

    assert unrelated.read_bytes() == b"unrelated"
    assert final.stat().st_nlink == 2


def test_stale_link_recovery_rejects_nonprivate_controlled_alias(tmp_path: Path) -> None:
    shard = tmp_path / "shard"
    shard.mkdir(mode=0o700)
    final = shard / "final"
    final.write_bytes(b"final")
    final.chmod(0o644)
    os.link(final, shard / f".blob-{'b' * 32}.tmp")
    shard_descriptor = os.open(shard, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    final_descriptor = os.open(final, os.O_RDONLY)
    try:
        with pytest.raises(BlobIntegrityError, match="temporary link is not private"):
            blobs_module._recover_controlled_stale_links(shard_descriptor, final_descriptor)
    finally:
        os.close(final_descriptor)
        os.close(shard_descriptor)


def test_blob_descriptor_verification_rejects_nonregular_and_wrong_size(tmp_path: Path) -> None:
    read_descriptor, write_descriptor = os.pipe()
    try:
        with pytest.raises(BlobIntegrityError, match="regular file"):
            blobs_module._verify_blob_descriptor(
                read_descriptor,
                expected_sha256=sha256_hex(b""),
                expected_size=0,
            )
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)

    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    blob.chmod(0o600)
    descriptor = os.open(blob, os.O_RDONLY)
    try:
        with pytest.raises(BlobIntegrityError, match="byte size"):
            blobs_module._verify_blob_descriptor(
                descriptor,
                expected_sha256=sha256_hex(b"x"),
                expected_size=2,
            )
    finally:
        os.close(descriptor)


def test_promotion_cleanup_runs_when_temporary_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with initialize_library(
        LibraryConfig(name="Blob temporary open failure"), data_root=tmp_path / "data"
    ) as repository:
        store = BlobStore(repository.paths)
        data = b"cleanup unopened temporary"
        digest = sha256_hex(data)
        real_open = os.open

        def fail_temporary_open(
            path: Any,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if isinstance(path, str) and blobs_module._CONTROLLED_TEMP_PATTERN.fullmatch(path):
                raise OSError(errno.EIO, "temporary file unavailable")
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr("dithyramba.provenance.blobs.os.open", fail_temporary_open)
        with pytest.raises(BlobStoreError, match="promotion failed"):
            store.promote(data, digest)

        shard = repository.paths.blobs / digest[:2]
        assert tuple(shard.iterdir()) == ()
