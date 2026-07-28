"""Explicit-network and exact-tree provisioning for local P7 rerankers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.recall.provisioning import (
    ModelProvisioningError,
    RerankerProvisioningReceipt,
    provision_reranker_model,
    verify_provisioned_reranker_model,
)
from dithyramba.recall.rerank import (
    BGE_M3_RERANKER_PROFILE,
    MMARCO_MINILM_RERANKER_PROFILE,
    RerankerProfile,
)

REVISION = "b" * 40


def _profile(*, model_id: str = "cross-encoder/test-model") -> RerankerProfile:
    return RerankerProfile(
        model_id=model_id,
        revision=REVISION,
        license="Apache-2.0",
        max_tokens=512,
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
            (target / "config.json").write_text('{"model":"reranker"}\n', encoding="utf-8")
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


def _target(root: Path, profile: RerankerProfile | None = None) -> Path:
    selected = profile or _profile()
    return root / "models" / selected.profile_id


def test_reranker_provision_is_explicit_pinned_receipted_and_idempotent(
    tmp_path: Path,
) -> None:
    command = _Download()
    with pytest.raises(ModelProvisioningError, match="explicit allow_network"):
        provision_reranker_model(_profile(), data_root=tmp_path, command=command)
    assert command.calls == []

    path, receipt = provision_reranker_model(
        _profile(),
        data_root=tmp_path,
        allow_network=True,
        timeout_seconds=321,
        command=command,
    )
    arguments, timeout = command.calls[0]
    assert path == _target(tmp_path)
    assert arguments == (
        "hf",
        "download",
        _profile().model_id,
        "--revision",
        REVISION,
        "--local-dir",
        arguments[-1],
    )
    assert ".partial" in arguments[-1]
    assert timeout == 321
    assert receipt.reranker_profile_id == _profile().profile_id
    assert receipt.reranker_profile_hash == _profile().profile_hash
    assert receipt.receipt_id.startswith("reranker_provisioning_")
    assert verify_provisioned_reranker_model(path, expected_profile=_profile()) == receipt
    second = _Download(error=AssertionError("must not run"))
    assert provision_reranker_model(_profile(), data_root=tmp_path, command=second) == (
        path,
        receipt,
    )
    assert second.calls == []


@pytest.mark.parametrize(
    "profile",
    [MMARCO_MINILM_RERANKER_PROFILE, BGE_M3_RERANKER_PROFILE],
)
def test_maintained_reranker_requires_its_complete_minimal_allowlist(
    tmp_path: Path,
    profile: RerankerProfile,
) -> None:
    complete = _PinnedDownload()
    path, receipt = provision_reranker_model(
        profile,
        data_root=tmp_path / "complete",
        allow_network=True,
        command=complete,
    )
    assert tuple(item.relative_path for item in receipt.files) == complete.requested
    assert (
        verify_provisioned_reranker_model(
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
            provision_reranker_model(
                profile,
                data_root=data_root,
                allow_network=True,
                command=command,
            )
        assert not _target(data_root, profile).exists()


@pytest.mark.parametrize("timeout", [False, 0, 7_201, cast(Any, "60")])
def test_reranker_timeout_is_bounded(tmp_path: Path, timeout: Any) -> None:
    with pytest.raises(ModelProvisioningError, match="timeout"):
        provision_reranker_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            timeout_seconds=timeout,
            command=_Download(),
        )


@pytest.mark.parametrize(
    "command",
    [
        _Download(result=7),
        _Download(error=OSError("secret provider detail")),
        _Download(error=subprocess.TimeoutExpired("hf", 1)),
    ],
)
def test_failed_reranker_download_is_sanitized_and_leaves_no_target(
    tmp_path: Path,
    command: _Download,
) -> None:
    with pytest.raises(ModelProvisioningError) as caught:
        provision_reranker_model(
            _profile(),
            data_root=tmp_path,
            allow_network=True,
            command=command,
        )
    assert "secret" not in str(caught.value)
    assert not _target(tmp_path).exists()


def test_reranker_verify_rejects_tamper_wrong_profile_and_bad_path(tmp_path: Path) -> None:
    path, _receipt = provision_reranker_model(
        _profile(),
        data_root=tmp_path,
        allow_network=True,
        command=_Download(),
    )
    with pytest.raises(ModelProvisioningError, match="another profile"):
        verify_provisioned_reranker_model(
            path,
            expected_profile=_profile(model_id="cross-encoder/other"),
        )
    with pytest.raises(ModelProvisioningError, match="absolute"):
        verify_provisioned_reranker_model(
            Path("relative"),
            expected_profile=_profile(),
        )
    (path / "config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ModelProvisioningError, match="differ"):
        verify_provisioned_reranker_model(path, expected_profile=_profile())


def test_reranker_receipt_and_entrypoints_require_exact_identity(tmp_path: Path) -> None:
    with pytest.raises(ModelProvisioningError, match="exact RerankerProfile"):
        provision_reranker_model(cast(Any, object()), data_root=tmp_path)
    with pytest.raises(ModelProvisioningError, match="exact RerankerProfile"):
        verify_provisioned_reranker_model(
            tmp_path,
            expected_profile=cast(Any, object()),
        )
    with pytest.raises((ValidationError, ModelProvisioningError), match="profile ID"):
        RerankerProvisioningReceipt(
            reranker_profile_id="wrong",
            reranker_profile_hash="a" * 64,
            model_id="owner/model",
            revision="a" * 40,
            files=(),
        )
