"""Explicit-network and exact-tree model provisioning contracts."""

from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.contracts import canonical_json_bytes
from dithyramba.recall import provisioning as provisioning_module
from dithyramba.recall.hybrid_models import EmbeddingModelProfile
from dithyramba.recall.provisioning import (
    HARRIER_OSS_V1_270M_PROFILE,
    MULTILINGUAL_E5_SMALL_PROFILE,
    ModelProvisioningError,
    ModelProvisioningReceipt,
    ProvisionedModelFile,
    provision_model,
    verify_provisioned_model,
)

REVISION = "a" * 40


def _profile(*, model_id: str = "intfloat/multilingual-e5-small") -> EmbeddingModelProfile:
    return EmbeddingModelProfile(
        model_id=model_id,
        revision=REVISION,
        license="MIT",
        dimensions=384,
        max_tokens=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )


class _Download:
    def __init__(self, *, result: int = 0, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def __call__(self, arguments: tuple[str, ...], *, timeout_seconds: int) -> int:
        self.calls.append((arguments, timeout_seconds))
        if self.error is not None:
            raise self.error
        if self.result == 0:
            target = Path(arguments[arguments.index("--local-dir") + 1])
            (target / "1_Pooling").mkdir()
            (target / "config.json").write_text('{"model":"test"}\n', encoding="utf-8")
            (target / "1_Pooling" / "config.json").write_text("{}\n", encoding="utf-8")
            cache = target / ".cache" / "huggingface"
            cache.mkdir(parents=True)
            (cache / "metadata").write_text("operational", encoding="utf-8")
        return self.result


class _PinnedDownload:
    def __init__(self, *, omit_last: bool = False, add_extra: bool = False) -> None:
        self.omit_last = omit_last
        self.add_extra = add_extra
        self.requested: tuple[str, ...] = ()

    def __call__(self, arguments: tuple[str, ...], *, timeout_seconds: int) -> int:
        assert timeout_seconds > 0
        revision_index = arguments.index("--revision")
        self.requested = arguments[3:revision_index]
        selected = self.requested[:-1] if self.omit_last else self.requested
        target = Path(arguments[arguments.index("--local-dir") + 1])
        for relative_path in selected:
            path = target / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative_path.encode("utf-8"))
        if self.add_extra:
            (target / "alternate.bin").write_bytes(b"unexpected")
        return 0


def _target(data_root: Path, profile: EmbeddingModelProfile | None = None) -> Path:
    selected = profile or _profile()
    return data_root / "models" / selected.profile_id


def test_harrier_profile_is_pinned_for_long_context_recall() -> None:
    assert HARRIER_OSS_V1_270M_PROFILE.model_id == "microsoft/harrier-oss-v1-270m"
    assert HARRIER_OSS_V1_270M_PROFILE.revision == ("31de22b673913c7d658c0f03f792d77c2dcf8ebd")
    assert HARRIER_OSS_V1_270M_PROFILE.license == "MIT"
    assert HARRIER_OSS_V1_270M_PROFILE.dimensions == 640
    assert HARRIER_OSS_V1_270M_PROFILE.max_tokens == 32_768
    assert HARRIER_OSS_V1_270M_PROFILE.query_prefix == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query: "
    )
    assert HARRIER_OSS_V1_270M_PROFILE.passage_prefix == ""


def test_harrier_profile_uses_exact_pinned_allowlist(tmp_path: Path) -> None:
    command = _PinnedDownload()

    provision_model(
        HARRIER_OSS_V1_270M_PROFILE,
        data_root=tmp_path,
        allow_network=True,
        command=command,
    )

    assert command.requested == (
        "1_Pooling/config.json",
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "mteb_v2_eval_prompts.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )


def test_provision_requires_explicit_network_before_running_command(tmp_path: Path) -> None:
    command = _Download()

    with pytest.raises(ModelProvisioningError, match="explicit allow_network"):
        provision_model(_profile(), data_root=tmp_path, command=command)

    assert command.calls == []
    assert not _target(tmp_path).exists()


def test_success_is_pinned_atomic_receipted_and_idempotent(tmp_path: Path) -> None:
    command = _Download()
    path, receipt = provision_model(
        _profile(),
        data_root=tmp_path,
        allow_network=True,
        timeout_seconds=123,
        command=command,
    )

    assert path == _target(tmp_path)
    assert path.is_dir()
    arguments, timeout = command.calls[0]
    assert arguments == (
        "hf",
        "download",
        "intfloat/multilingual-e5-small",
        "--revision",
        REVISION,
        "--local-dir",
        arguments[-1],
    )
    assert ".partial" in arguments[-1]
    assert Path(arguments[-1]).parent == tmp_path
    assert timeout == 123
    assert tuple(item.relative_path for item in receipt.files) == (
        "1_Pooling/config.json",
        "config.json",
    )
    assert receipt.receipt_id.startswith("model_provisioning_")
    assert len(receipt.receipt_hash) == 64
    assert verify_provisioned_model(path, expected_profile=_profile()) == receipt

    second = _Download(error=AssertionError("must not run"))
    assert provision_model(_profile(), data_root=tmp_path, command=second) == (path, receipt)
    assert second.calls == []


@pytest.mark.parametrize(
    "profile",
    [
        MULTILINGUAL_E5_SMALL_PROFILE,
        HARRIER_OSS_V1_270M_PROFILE,
    ],
)
def test_maintained_profile_requires_its_complete_minimal_allowlist(
    tmp_path: Path,
    profile: EmbeddingModelProfile,
) -> None:
    complete = _PinnedDownload()
    path, receipt = provision_model(
        profile,
        data_root=tmp_path / "complete",
        allow_network=True,
        command=complete,
    )
    assert tuple(item.relative_path for item in receipt.files) == complete.requested
    assert (
        verify_provisioned_model(
            path,
            expected_profile=profile,
        )
        == receipt
    )

    for label, command in (
        ("missing", _PinnedDownload(omit_last=True)),
        ("extra", _PinnedDownload(add_extra=True)),
    ):
        data_root = tmp_path / label
        with pytest.raises(ModelProvisioningError, match="pinned allowlist"):
            provision_model(
                profile,
                data_root=data_root,
                allow_network=True,
                command=command,
            )
        assert not _target(data_root, profile).exists()


@pytest.mark.parametrize("timeout", [False, 0, 7_201, cast(Any, "60")])
def test_timeout_is_bounded(tmp_path: Path, timeout: Any) -> None:
    with pytest.raises(ModelProvisioningError, match="timeout"):
        provision_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            timeout_seconds=timeout,
            command=_Download(),
        )


@pytest.mark.parametrize(
    "command",
    [
        _Download(result=9),
        _Download(error=OSError("hf missing token=secret")),
        _Download(error=subprocess.TimeoutExpired("hf", 1)),
    ],
)
def test_failed_command_leaves_no_partial_tree_or_provider_detail(
    tmp_path: Path,
    command: _Download,
) -> None:
    with pytest.raises(ModelProvisioningError) as caught:
        provision_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            command=command,
        )

    assert "secret" not in str(caught.value)
    assert not _target(tmp_path).exists()
    models = tmp_path / "models"
    assert not models.exists() or not tuple(models.glob("*.partial"))
    assert not tuple(tmp_path.glob(".*.partial"))


def test_command_must_return_an_exact_zero_integer(tmp_path: Path) -> None:
    class _BooleanCommand:
        def __call__(self, arguments: tuple[str, ...], *, timeout_seconds: int) -> bool:
            _Download()(arguments, timeout_seconds=timeout_seconds)
            return False

    with pytest.raises(ModelProvisioningError, match="failed"):
        provision_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            command=cast(Any, _BooleanCommand()),
        )


def test_verify_detects_changed_missing_extra_and_linked_files(tmp_path: Path) -> None:
    path, _receipt = provision_model(
        _profile(), data_root=tmp_path, allow_network=True, command=_Download()
    )
    config = path / "config.json"
    config.write_text("tampered", encoding="utf-8")
    with pytest.raises(ModelProvisioningError, match="differ"):
        verify_provisioned_model(path, expected_profile=_profile())

    config.write_text('{"model":"test"}\n', encoding="utf-8")
    (path / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ModelProvisioningError, match="differ"):
        verify_provisioned_model(path, expected_profile=_profile())

    (path / "extra.bin").unlink()
    config.unlink()
    with pytest.raises(ModelProvisioningError, match="differ"):
        verify_provisioned_model(path, expected_profile=_profile())

    config.symlink_to(path / "1_Pooling" / "config.json")
    with pytest.raises(ModelProvisioningError, match="symbolic"):
        verify_provisioned_model(path, expected_profile=_profile())


def test_verify_ignores_only_operational_cache(tmp_path: Path) -> None:
    path, receipt = provision_model(
        _profile(), data_root=tmp_path, allow_network=True, command=_Download()
    )
    cache_file = path / ".cache" / "huggingface" / "new-metadata"
    cache_file.write_text("changed", encoding="utf-8")
    assert verify_provisioned_model(path, expected_profile=_profile()) == receipt


def test_verify_rejects_wrong_profile_marker_and_path_shapes(tmp_path: Path) -> None:
    path, receipt = provision_model(
        _profile(), data_root=tmp_path, allow_network=True, command=_Download()
    )
    with pytest.raises(ModelProvisioningError, match="another profile"):
        verify_provisioned_model(path, expected_profile=_profile(model_id="owner/other"))
    with pytest.raises(ModelProvisioningError, match="absolute"):
        verify_provisioned_model(Path("relative"), expected_profile=_profile())

    marker = path / ".dithyramba-model.json"
    marker.write_text("not-json", encoding="utf-8")
    with pytest.raises(ModelProvisioningError, match="malformed"):
        verify_provisioned_model(path, expected_profile=_profile())
    marker.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    with pytest.raises(ModelProvisioningError, match="size"):
        verify_provisioned_model(path, expected_profile=_profile())

    marker.write_bytes(canonical_json_bytes(receipt.stored_payload()) + b"\n")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["receipt_hash"] = "0" * 64
    marker.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ModelProvisioningError, match="hash is inconsistent"):
        verify_provisioned_model(path, expected_profile=_profile())


def test_existing_symlink_target_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = _target(tmp_path)
    target.parent.mkdir()
    target.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ModelProvisioningError, match="real absolute"):
        provision_model(_profile(), data_root=tmp_path)


def test_verify_rejects_a_symlink_substitution_of_the_model_parent(tmp_path: Path) -> None:
    path, _receipt = provision_model(
        _profile(), data_root=tmp_path, allow_network=True, command=_Download()
    )
    models = path.parent
    moved = tmp_path / "models-real"
    models.rename(moved)
    models.symlink_to(moved, target_is_directory=True)

    with pytest.raises(ModelProvisioningError, match="real absolute"):
        verify_provisioned_model(path, expected_profile=_profile())


def test_verify_rechecks_files_read_earlier_in_the_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _receipt = provision_model(
        _profile(), data_root=tmp_path, allow_network=True, command=_Download()
    )
    original_read = provisioning_module._read_regular_entry
    regular_reads = 0

    def racing_read(*args: Any, **kwargs: Any) -> Any:
        nonlocal regular_reads
        result = original_read(*args, **kwargs)
        if kwargs.get("collect") is False:
            regular_reads += 1
            if regular_reads == 2:
                (path / "1_Pooling" / "config.json").write_text(
                    "tampered after its scan",
                    encoding="utf-8",
                )
        return result

    monkeypatch.setattr(provisioning_module, "_read_regular_entry", racing_read)
    with pytest.raises(ModelProvisioningError, match="changed during verification"):
        verify_provisioned_model(path, expected_profile=_profile())


@pytest.mark.parametrize(
    "file",
    [
        {"relative_path": "", "size_bytes": 1, "sha256": "a" * 64},
        {"relative_path": "bad\\path", "size_bytes": 1, "sha256": "a" * 64},
        {"relative_path": "bad\x00path", "size_bytes": 1, "sha256": "a" * 64},
        {"relative_path": "../escape", "size_bytes": 1, "sha256": "a" * 64},
        {"relative_path": "/absolute", "size_bytes": 1, "sha256": "a" * 64},
        {"relative_path": ".cache/item", "size_bytes": 1, "sha256": "a" * 64},
        {"relative_path": "file", "size_bytes": 1, "sha256": "a" * 63},
        {"relative_path": "file", "size_bytes": 1, "sha256": "A" * 64},
    ],
)
def test_manifest_file_contract_rejects_unsafe_values(file: dict[str, object]) -> None:
    with pytest.raises((ValidationError, ModelProvisioningError)):
        ProvisionedModelFile.model_validate(file)


def test_receipt_requires_sorted_unique_nonempty_files() -> None:
    alpha = ProvisionedModelFile(relative_path="a", size_bytes=1, sha256="a" * 64)
    beta = ProvisionedModelFile(relative_path="b", size_bytes=1, sha256="b" * 64)
    base = {
        "embedding_profile_id": _profile().profile_id,
        "embedding_profile_hash": _profile().profile_hash,
        "model_id": _profile().model_id,
        "revision": REVISION,
    }
    for files in ((), (beta, alpha), (alpha, alpha)):
        with pytest.raises((ValidationError, ModelProvisioningError)):
            ModelProvisioningReceipt(**base, files=files)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_profile_id", "wrong"),
        ("embedding_profile_hash", "a" * 63),
        ("embedding_profile_hash", "A" * 64),
        ("model_id", "missing-slash"),
        ("revision", "a" * 39),
        ("revision", "A" * 40),
    ],
)
def test_receipt_rejects_invalid_identity_fields(field: str, value: str) -> None:
    file = ProvisionedModelFile(relative_path="config.json", size_bytes=2, sha256="a" * 64)
    payload: dict[str, object] = {
        "embedding_profile_id": _profile().profile_id,
        "embedding_profile_hash": _profile().profile_hash,
        "model_id": _profile().model_id,
        "revision": REVISION,
        "files": (file,),
    }
    payload[field] = value

    with pytest.raises((ValidationError, ModelProvisioningError)):
        ModelProvisioningReceipt.model_validate(payload)


def test_public_entrypoints_require_the_exact_profile_type(tmp_path: Path) -> None:
    with pytest.raises(ModelProvisioningError, match="exact EmbeddingModelProfile"):
        provision_model(cast(Any, object()), data_root=tmp_path)
    with pytest.raises(ModelProvisioningError, match="exact EmbeddingModelProfile"):
        verify_provisioned_model(tmp_path, expected_profile=cast(Any, object()))


def test_model_root_must_not_be_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "models").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ModelProvisioningError, match="model root"):
        provision_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            command=_Download(),
        )


def test_verify_rejects_missing_malformed_and_unsupported_receipts(tmp_path: Path) -> None:
    path, receipt = provision_model(
        _profile(), data_root=tmp_path, allow_network=True, command=_Download()
    )
    marker = path / ".dithyramba-model.json"
    marker.unlink()
    with pytest.raises(ModelProvisioningError, match="receipt is missing"):
        verify_provisioned_model(path, expected_profile=_profile())

    marker.write_text("[]", encoding="utf-8")
    with pytest.raises(ModelProvisioningError, match="malformed"):
        verify_provisioned_model(path, expected_profile=_profile())

    payload = receipt.stored_payload()
    payload["schema"] = "unsupported"
    marker.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ModelProvisioningError, match="schema is unsupported"):
        verify_provisioned_model(path, expected_profile=_profile())

    malformed_files = receipt.stored_payload()
    malformed_files["files"] = "not-a-list"
    marker.write_bytes(canonical_json_bytes(malformed_files) + b"\n")
    with pytest.raises(ModelProvisioningError, match="malformed"):
        verify_provisioned_model(path, expected_profile=_profile())


def test_unsupported_file_type_is_rejected(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")

    class _FifoDownload(_Download):
        def __call__(self, arguments: tuple[str, ...], *, timeout_seconds: int) -> int:
            result = super().__call__(arguments, timeout_seconds=timeout_seconds)
            target = Path(arguments[arguments.index("--local-dir") + 1])
            os.mkfifo(target / "unsupported.pipe")
            return result

    with pytest.raises(ModelProvisioningError, match="unsupported file type"):
        provision_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            command=_FifoDownload(),
        )


def test_file_count_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provisioning_module, "_MAX_FILES", 1)

    with pytest.raises(ModelProvisioningError, match="too many files"):
        provision_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            command=_Download(),
        )


def test_marker_write_failure_is_sanitized_and_cleans_partial_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fsync(_descriptor: int) -> None:
        raise OSError("sensitive filesystem detail")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(ModelProvisioningError, match="could not be finalized") as caught:
        provision_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            command=_Download(),
        )

    assert "sensitive" not in str(caught.value)
    assert not _target(tmp_path).exists()
    assert not tuple((tmp_path / "models").glob("*.partial"))


def test_matching_concurrent_provision_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(tmp_path)

    def concurrent_promotion(source: Path, **_kwargs: object) -> None:
        shutil.copytree(source, target)
        raise FileExistsError(target)

    monkeypatch.setattr(provisioning_module, "_promote_staging", concurrent_promotion)
    path, receipt = provision_model(
        _profile(), data_root=tmp_path, allow_network=True, command=_Download()
    )

    assert path == _target(tmp_path)
    assert verify_provisioned_model(path, expected_profile=_profile()) == receipt


def test_conflicting_concurrent_provision_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(tmp_path)

    def concurrent_promotion(source: Path, **_kwargs: object) -> None:
        shutil.copytree(source, target)
        (target / "different.bin").write_bytes(b"different")
        other = provisioning_module._build_receipt(target, profile=_profile())
        (target / ".dithyramba-model.json").unlink()
        provisioning_module._write_marker(target, other)
        raise FileExistsError(target)

    monkeypatch.setattr(provisioning_module, "_promote_staging", concurrent_promotion)
    with pytest.raises(ModelProvisioningError, match=r"concurrent.*conflict"):
        provision_model(_profile(), data_root=tmp_path, allow_network=True, command=_Download())


def test_real_concurrent_provision_reuses_one_exact_tree(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)

    class _ConcurrentDownload(_Download):
        def __call__(self, arguments: tuple[str, ...], *, timeout_seconds: int) -> int:
            result = super().__call__(arguments, timeout_seconds=timeout_seconds)
            barrier.wait(timeout=5)
            return result

    def provision_once(_index: int) -> tuple[Path, ModelProvisioningReceipt]:
        return provision_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            command=_ConcurrentDownload(),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = tuple(executor.map(provision_once, range(2)))

    assert first == second
    assert first[0] == _target(tmp_path)
    assert verify_provisioned_model(first[0], expected_profile=_profile()) == first[1]
    assert not tuple((tmp_path / "models").glob("*.partial"))


def test_model_root_creation_race_and_promotion_failure_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    original_mkdir = os.mkdir

    def racing_mkdir(
        path: Any,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if path == "models" and dir_fd is not None:
            os.symlink(outside, path, dir_fd=dir_fd, target_is_directory=True)
            return
        original_mkdir(path, mode=mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", racing_mkdir)
    with pytest.raises(ModelProvisioningError, match="model root"):
        provision_model(_profile(), data_root=tmp_path, allow_network=True, command=_Download())

    monkeypatch.undo()
    (tmp_path / "models").unlink()

    def denied_promotion(_source: Path, **_kwargs: object) -> None:
        raise PermissionError(errno.EACCES, "private detail")

    monkeypatch.setattr(provisioning_module, "_promote_staging", denied_promotion)
    with pytest.raises(ModelProvisioningError, match="promotion failed") as caught:
        provision_model(_profile(), data_root=tmp_path, allow_network=True, command=_Download())
    assert "private detail" not in str(caught.value)


def test_model_root_swap_cannot_redirect_the_download_staging(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    detached = tmp_path / "models-detached"
    seen_staging: list[Path] = []

    class _SwapDuringDownload(_Download):
        def __call__(self, arguments: tuple[str, ...], *, timeout_seconds: int) -> int:
            models.rename(detached)
            models.symlink_to(outside, target_is_directory=True)
            staging = Path(arguments[arguments.index("--local-dir") + 1])
            seen_staging.append(staging)
            assert not staging.is_relative_to(outside)
            return super().__call__(arguments, timeout_seconds=timeout_seconds)

    with pytest.raises(ModelProvisioningError, match="changed during verification"):
        provision_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            command=_SwapDuringDownload(),
        )

    assert seen_staging
    assert not any(outside.iterdir())
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755


def test_fd_scanner_rejects_zero_marker_hardlinks_and_open_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, receipt = provision_model(
        _profile(), data_root=tmp_path, allow_network=True, command=_Download()
    )
    marker = path / ".dithyramba-model.json"
    marker.write_bytes(b"")
    with pytest.raises(ModelProvisioningError, match="receipt size"):
        verify_provisioned_model(path, expected_profile=_profile())

    marker.write_bytes(canonical_json_bytes(receipt.stored_payload()) + b"\n")
    os.link(path / "config.json", path / "hardlink.bin")
    with pytest.raises(ModelProvisioningError, match="privately linked"):
        verify_provisioned_model(path, expected_profile=_profile())
    (path / "hardlink.bin").unlink()

    original_scandir = os.scandir

    def denied_scandir(_descriptor: int) -> Any:
        raise OSError("denied")

    monkeypatch.setattr(os, "scandir", denied_scandir)
    with pytest.raises(ModelProvisioningError, match="could not be scanned"):
        verify_provisioned_model(path, expected_profile=_profile())
    monkeypatch.setattr(os, "scandir", original_scandir)

    original_open = os.open

    def denied_open(
        target: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if target == "1_Pooling":
            raise OSError("changed")
        return original_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", denied_open)
    with pytest.raises(ModelProvisioningError, match="changed during verification"):
        verify_provisioned_model(path, expected_profile=_profile())


def test_fd_scanner_rejects_unopenable_root_and_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModelProvisioningError, match="real absolute"):
        provisioning_module._scan_model_tree(tmp_path / "missing")

    path, _receipt = provision_model(
        _profile(), data_root=tmp_path, allow_network=True, command=_Download()
    )
    original_open = os.open

    def denied_open(
        target: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if target == "config.json":
            raise OSError("changed")
        return original_open(target, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", denied_open)
    with pytest.raises(ModelProvisioningError, match="changed during verification"):
        verify_provisioned_model(path, expected_profile=_profile())


def test_default_subprocess_is_shell_free_and_output_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(
        arguments: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(kwargs)
        target = Path(arguments[arguments.index("--local-dir") + 1])
        (target / "config.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(
        provisioning_module,
        "_resolve_hf_executable",
        lambda: str(tmp_path / "isolated-hf"),
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    path, _receipt = provision_model(_profile(), data_root=tmp_path, allow_network=True)

    assert path.is_dir()
    assert calls == [
        {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "timeout": 3_600,
            "check": False,
            "shell": False,
        }
    ]


def test_default_command_prefers_hf_next_to_the_active_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = tmp_path / "isolated-environment" / "bin"
    scripts.mkdir(parents=True)
    interpreter = scripts / "python"
    interpreter.write_text("", encoding="utf-8")
    hf = scripts / "hf"
    hf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hf.chmod(0o700)
    calls: list[tuple[str, ...]] = []

    def fake_run(
        arguments: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(arguments)
        target = Path(arguments[arguments.index("--local-dir") + 1])
        (target / "config.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(subprocess, "run", fake_run)

    provision_model(_profile(), data_root=tmp_path / "data", allow_network=True)

    assert calls[0][0] == str(hf.resolve(strict=True))


def test_default_command_falls_back_to_an_absolute_hf_from_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = tmp_path / "isolated-environment" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    hf = path_bin / "hf"
    hf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hf.chmod(0o700)
    calls: list[tuple[str, ...]] = []

    def fake_run(
        arguments: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(arguments)
        target = Path(arguments[arguments.index("--local-dir") + 1])
        (target / "config.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setenv("PATH", str(path_bin))
    monkeypatch.setattr(subprocess, "run", fake_run)

    provision_model(_profile(), data_root=tmp_path / "data", allow_network=True)

    assert calls[0][0] == str(hf.resolve(strict=True))


def test_default_command_fails_closed_when_hf_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = tmp_path / "isolated-environment" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    calls = 0

    def unexpected_run(
        _arguments: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        raise AssertionError("subprocess must not run without a resolved hf executable")

    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(subprocess, "run", unexpected_run)

    with pytest.raises(ModelProvisioningError, match="download command could not run"):
        provision_model(_profile(), data_root=tmp_path / "data", allow_network=True)

    assert calls == 0
    assert not _target(tmp_path / "data").exists()
    assert not tuple((tmp_path / "data" / "models").glob("*.partial"))
