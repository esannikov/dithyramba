"""Explicit, atomic and locally verifiable model provisioning."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dithyramba.contracts import canonical_content_id, canonical_json_bytes, canonical_sha256_hex
from dithyramba.library.paths import PathInput, application_data_root

from .hybrid_models import EmbeddingModelProfile, HybridContractError
from .rerank import BGE_M3_RERANKER_PROFILE, RerankerProfile

_MARKER_NAME = ".dithyramba-model.json"
_OPERATIONAL_DIRECTORY = ".cache"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_FILES = 100_000
_HASH_CHUNK_BYTES = 1024 * 1024

MULTILINGUAL_E5_SMALL_PROFILE = EmbeddingModelProfile(
    model_id="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    license="MIT",
    dimensions=384,
    max_tokens=512,
    query_prefix="query: ",
    passage_prefix="passage: ",
)

MULTILINGUAL_E5_BASE_PROFILE = EmbeddingModelProfile(
    model_id="intfloat/multilingual-e5-base",
    revision="d128750597153bb5987e10b1c3493a34e5a4502a",
    license="MIT",
    dimensions=768,
    max_tokens=512,
    query_prefix="query: ",
    passage_prefix="passage: ",
)

HARRIER_OSS_V1_270M_PROFILE = EmbeddingModelProfile(
    model_id="microsoft/harrier-oss-v1-270m",
    revision="31de22b673913c7d658c0f03f792d77c2dcf8ebd",
    license="MIT",
    dimensions=640,
    max_tokens=32_768,
    query_prefix=(
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query: "
    ),
    passage_prefix="",
)

_PINNED_MODEL_FILES: dict[tuple[str, str], tuple[str, ...]] = {
    (
        MULTILINGUAL_E5_SMALL_PROFILE.model_id,
        MULTILINGUAL_E5_SMALL_PROFILE.revision,
    ): (
        "1_Pooling/config.json",
        "config.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    (
        MULTILINGUAL_E5_BASE_PROFILE.model_id,
        MULTILINGUAL_E5_BASE_PROFILE.revision,
    ): (
        "1_Pooling/config.json",
        "config.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    (
        HARRIER_OSS_V1_270M_PROFILE.model_id,
        HARRIER_OSS_V1_270M_PROFILE.revision,
    ): (
        "1_Pooling/config.json",
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "mteb_v2_eval_prompts.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "1427fd652930e4ba29e8149678df786c240d8825",
    ): (
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    (
        BGE_M3_RERANKER_PROFILE.model_id,
        BGE_M3_RERANKER_PROFILE.revision,
    ): (
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
}


class ModelProvisioningError(HybridContractError):
    """Model weights are absent, incomplete, conflicting or locally modified."""


class ProvisioningCommand(Protocol):
    """Bounded command seam; production suppresses provider output and never uses a shell."""

    def __call__(self, arguments: tuple[str, ...], *, timeout_seconds: int) -> int: ...


@dataclass(frozen=True, slots=True)
class _DirectoryHandle:
    descriptor: int
    parent_descriptor: int | None
    component: str | None
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _TreeEntryFingerprint:
    parts: tuple[str, ...]
    identity: tuple[int, ...]
    is_directory: bool


class ProvisionedModelFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str

    @field_validator("relative_path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        if type(value) is not str or not value or "\x00" in value or "\\" in value:
            raise ModelProvisioningError("model manifest path is invalid")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ModelProvisioningError("model manifest path must be canonical and relative")
        if _OPERATIONAL_DIRECTORY in path.parts or value == _MARKER_NAME:
            raise ModelProvisioningError("operational model files cannot enter the manifest")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if type(value) is not str or len(value) != 64:
            raise ModelProvisioningError("model file hash must be lowercase SHA-256")
        if any(character not in "0123456789abcdef" for character in value):
            raise ModelProvisioningError("model file hash must be lowercase SHA-256")
        return value

    def payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


class ModelProvisioningReceipt(BaseModel):
    """Content-addressed inventory of one explicitly pinned local model tree."""

    SCHEMA: ClassVar[str] = "dithyramba.model_provisioning_receipt/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    embedding_profile_id: str
    embedding_profile_hash: str
    model_id: str
    revision: str
    files: tuple[ProvisionedModelFile, ...]

    @field_validator("embedding_profile_id")
    @classmethod
    def _profile_id(cls, value: str) -> str:
        if type(value) is not str or not value.startswith("embedding_profile_"):
            raise ModelProvisioningError("model receipt requires an embedding profile ID")
        return value

    @field_validator("embedding_profile_hash")
    @classmethod
    def _profile_hash(cls, value: str) -> str:
        if type(value) is not str or len(value) != 64:
            raise ModelProvisioningError("model receipt profile hash is invalid")
        if any(character not in "0123456789abcdef" for character in value):
            raise ModelProvisioningError("model receipt profile hash is invalid")
        return value

    @field_validator("model_id")
    @classmethod
    def _model_id(cls, value: str) -> str:
        if type(value) is not str or "/" not in value or len(value) > 256:
            raise ModelProvisioningError("model receipt repository ID is invalid")
        return value

    @field_validator("revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        if type(value) is not str or len(value) != 40:
            raise ModelProvisioningError("model receipt revision is invalid")
        if any(character not in "0123456789abcdef" for character in value):
            raise ModelProvisioningError("model receipt revision is invalid")
        return value

    @model_validator(mode="after")
    def _files_are_closed(self) -> ModelProvisioningReceipt:
        if not self.files or len(self.files) > _MAX_FILES:
            raise ModelProvisioningError("model receipt requires a bounded non-empty file set")
        ordered = tuple(sorted(self.files, key=lambda item: item.relative_path.encode("utf-8")))
        if ordered != self.files:
            raise ModelProvisioningError("model receipt files must use binary path order")
        if len({item.relative_path for item in self.files}) != len(self.files):
            raise ModelProvisioningError("model receipt paths must be unique")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "embedding_profile_id": self.embedding_profile_id,
            "embedding_profile_hash": self.embedding_profile_hash,
            "model_id": self.model_id,
            "revision": self.revision,
            "files": [item.payload() for item in self.files],
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def receipt_id(self) -> str:
        return canonical_content_id("model_provisioning", self.semantic_payload())

    def stored_payload(self) -> dict[str, object]:
        return {
            **self.semantic_payload(),
            "receipt_id": self.receipt_id,
            "receipt_hash": self.receipt_hash,
        }


class RerankerProvisioningReceipt(BaseModel):
    """Content-addressed inventory for one explicitly pinned local reranker."""

    SCHEMA: ClassVar[str] = "dithyramba.reranker_provisioning_receipt/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    reranker_profile_id: str
    reranker_profile_hash: str
    model_id: str
    revision: str
    files: tuple[ProvisionedModelFile, ...]

    @field_validator("reranker_profile_id")
    @classmethod
    def _profile_id(cls, value: str) -> str:
        if type(value) is not str or not value.startswith("reranker_profile_"):
            raise ModelProvisioningError("reranker receipt requires a reranker profile ID")
        return value

    @field_validator("reranker_profile_hash")
    @classmethod
    def _profile_hash(cls, value: str) -> str:
        if type(value) is not str or len(value) != 64:
            raise ModelProvisioningError("reranker receipt profile hash is invalid")
        if any(character not in "0123456789abcdef" for character in value):
            raise ModelProvisioningError("reranker receipt profile hash is invalid")
        return value

    @field_validator("model_id")
    @classmethod
    def _model_id(cls, value: str) -> str:
        if type(value) is not str or "/" not in value or len(value) > 256:
            raise ModelProvisioningError("reranker receipt repository ID is invalid")
        return value

    @field_validator("revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        if type(value) is not str or len(value) != 40:
            raise ModelProvisioningError("reranker receipt revision is invalid")
        if any(character not in "0123456789abcdef" for character in value):
            raise ModelProvisioningError("reranker receipt revision is invalid")
        return value

    @model_validator(mode="after")
    def _files_are_closed(self) -> RerankerProvisioningReceipt:
        if not self.files or len(self.files) > _MAX_FILES:
            raise ModelProvisioningError("reranker receipt requires a bounded non-empty file set")
        ordered = tuple(sorted(self.files, key=lambda item: item.relative_path.encode("utf-8")))
        if ordered != self.files:
            raise ModelProvisioningError("reranker receipt files must use binary path order")
        if len({item.relative_path for item in self.files}) != len(self.files):
            raise ModelProvisioningError("reranker receipt paths must be unique")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "reranker_profile_id": self.reranker_profile_id,
            "reranker_profile_hash": self.reranker_profile_hash,
            "model_id": self.model_id,
            "revision": self.revision,
            "files": [item.payload() for item in self.files],
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def receipt_id(self) -> str:
        return canonical_content_id("reranker_provisioning", self.semantic_payload())

    def stored_payload(self) -> dict[str, object]:
        return {
            **self.semantic_payload(),
            "receipt_id": self.receipt_id,
            "receipt_hash": self.receipt_hash,
        }


def provision_model(
    profile: EmbeddingModelProfile,
    *,
    data_root: PathInput | None = None,
    allow_network: bool = False,
    timeout_seconds: int = 3_600,
    command: ProvisioningCommand | None = None,
) -> tuple[Path, ModelProvisioningReceipt]:
    """Download one exact revision only when network authority is explicit."""

    _require_profile(profile)
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7_200:
        raise ModelProvisioningError("model provisioning timeout must be 1-7200 seconds")
    app_root = application_data_root(data_root)
    root = app_root / "models"
    target = root / profile.profile_id
    if (root.exists() or root.is_symlink()) and (root.is_symlink() or not root.is_dir()):
        raise ModelProvisioningError("model root must be a real local directory")
    if target.exists() or target.is_symlink():
        return target, verify_provisioned_model(target, expected_profile=profile)
    if allow_network is not True:
        raise ModelProvisioningError(
            "model weights are absent; repeat with explicit allow_network=True"
        )
    root_chain = _open_or_create_model_root_chain(app_root)
    root_descriptor = root_chain[-1].descriptor
    staging = Path()
    try:
        if _model_target_exists(root_descriptor, profile.profile_id):
            _revalidate_directory_chain(root_chain)
            return target, verify_provisioned_model(target, expected_profile=profile)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{profile.profile_id}.",
                suffix=".partial",
            )
        ).resolve(strict=True)
        runner = _run_hf if command is None else command
        arguments = (
            "hf",
            "download",
            profile.model_id,
            *_PINNED_MODEL_FILES.get((profile.model_id, profile.revision), ()),
            "--revision",
            profile.revision,
            "--local-dir",
            str(staging),
        )
        try:
            exit_code = runner(arguments, timeout_seconds=timeout_seconds)
        except (OSError, subprocess.SubprocessError) as error:
            raise ModelProvisioningError("model download command could not run") from error
        if type(exit_code) is not int or exit_code != 0:
            raise ModelProvisioningError("model download command failed")
        try:
            receipt = _build_receipt(staging, profile=profile)
            _write_marker(staging, receipt)
            verify_provisioned_model(staging, expected_profile=profile)
        except OSError as error:
            raise ModelProvisioningError("model files could not be finalized") from error
        staging_parent_chain = _open_absolute_directory_chain(staging.parent)
        try:
            _revalidate_directory_chain(root_chain)
            _promote_staging(
                staging,
                target_name=profile.profile_id,
                source_parent_descriptor=staging_parent_chain[-1].descriptor,
                model_root_descriptor=root_descriptor,
            )
        except OSError as error:
            if not isinstance(error, FileExistsError) and error.errno not in {
                errno.EEXIST,
                errno.ENOTEMPTY,
            }:
                raise ModelProvisioningError("model staging promotion failed") from error
            existing = verify_provisioned_model(target, expected_profile=profile)
            if existing.receipt_hash != receipt.receipt_hash:
                raise ModelProvisioningError(
                    "concurrent model provisioning produced a conflict"
                ) from error
            return target, existing
        finally:
            _close_directory_chain(staging_parent_chain)
        staging = Path()
        _revalidate_directory_chain(root_chain)
        return target, verify_provisioned_model(target, expected_profile=profile)
    finally:
        _close_directory_chain(root_chain)
        if staging != Path() and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def provision_reranker_model(
    profile: RerankerProfile,
    *,
    data_root: PathInput | None = None,
    allow_network: bool = False,
    timeout_seconds: int = 3_600,
    command: ProvisioningCommand | None = None,
) -> tuple[Path, RerankerProvisioningReceipt]:
    """Download one exact reranker revision only with explicit network authority."""

    _require_reranker_profile(profile)
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7_200:
        raise ModelProvisioningError("reranker provisioning timeout must be 1-7200 seconds")
    app_root = application_data_root(data_root)
    root = app_root / "models"
    target = root / profile.profile_id
    if (root.exists() or root.is_symlink()) and (root.is_symlink() or not root.is_dir()):
        raise ModelProvisioningError("model root must be a real local directory")
    if target.exists() or target.is_symlink():
        return target, verify_provisioned_reranker_model(target, expected_profile=profile)
    if allow_network is not True:
        raise ModelProvisioningError(
            "reranker weights are absent; repeat with explicit allow_network=True"
        )
    root_chain = _open_or_create_model_root_chain(app_root)
    root_descriptor = root_chain[-1].descriptor
    staging = Path()
    try:
        if _model_target_exists(root_descriptor, profile.profile_id):
            _revalidate_directory_chain(root_chain)
            return target, verify_provisioned_reranker_model(
                target,
                expected_profile=profile,
            )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{profile.profile_id}.",
                suffix=".partial",
            )
        ).resolve(strict=True)
        runner = _run_hf if command is None else command
        arguments = (
            "hf",
            "download",
            profile.model_id,
            *_PINNED_MODEL_FILES.get((profile.model_id, profile.revision), ()),
            "--revision",
            profile.revision,
            "--local-dir",
            str(staging),
        )
        try:
            exit_code = runner(arguments, timeout_seconds=timeout_seconds)
        except (OSError, subprocess.SubprocessError) as error:
            raise ModelProvisioningError("reranker download command could not run") from error
        if type(exit_code) is not int or exit_code != 0:
            raise ModelProvisioningError("reranker download command failed")
        try:
            receipt = _build_reranker_receipt(staging, profile=profile)
            _write_marker(staging, receipt)
            verify_provisioned_reranker_model(staging, expected_profile=profile)
        except OSError as error:
            raise ModelProvisioningError("reranker files could not be finalized") from error
        staging_parent_chain = _open_absolute_directory_chain(staging.parent)
        try:
            _revalidate_directory_chain(root_chain)
            _promote_staging(
                staging,
                target_name=profile.profile_id,
                source_parent_descriptor=staging_parent_chain[-1].descriptor,
                model_root_descriptor=root_descriptor,
            )
        except OSError as error:
            if not isinstance(error, FileExistsError) and error.errno not in {
                errno.EEXIST,
                errno.ENOTEMPTY,
            }:
                raise ModelProvisioningError("reranker staging promotion failed") from error
            existing = verify_provisioned_reranker_model(
                target,
                expected_profile=profile,
            )
            if existing.receipt_hash != receipt.receipt_hash:
                raise ModelProvisioningError(
                    "concurrent reranker provisioning produced a conflict"
                ) from error
            return target, existing
        finally:
            _close_directory_chain(staging_parent_chain)
        staging = Path()
        _revalidate_directory_chain(root_chain)
        return target, verify_provisioned_reranker_model(target, expected_profile=profile)
    finally:
        _close_directory_chain(root_chain)
        if staging != Path() and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def verify_provisioned_model(
    model_path: str | Path,
    *,
    expected_profile: EmbeddingModelProfile,
) -> ModelProvisioningReceipt:
    """Rehash an exact local tree and reject missing, extra, linked or changed files."""

    _require_profile(expected_profile)
    root = Path(model_path)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ModelProvisioningError("provisioned model must be a real absolute directory")
    files, marker_payload = _scan_model_tree(root)
    if marker_payload is None:
        raise ModelProvisioningError("provisioned model receipt is missing")
    try:
        raw = json.loads(marker_payload.decode("utf-8"))
        if type(raw) is not dict:
            raise ModelProvisioningError("provisioned model receipt is malformed")
        stored_id = raw.pop("receipt_id", None)
        stored_hash = raw.pop("receipt_hash", None)
        stored_schema = raw.pop("schema", None)
        if stored_schema != ModelProvisioningReceipt.SCHEMA:
            raise ModelProvisioningError("provisioned model receipt schema is unsupported")
        if type(raw.get("files")) is list:
            raw["files"] = tuple(raw["files"])
        receipt = ModelProvisioningReceipt.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ModelProvisioningError):
            raise
        raise ModelProvisioningError("provisioned model receipt is malformed") from error
    if stored_id != receipt.receipt_id or stored_hash != receipt.receipt_hash:
        raise ModelProvisioningError("provisioned model receipt hash is inconsistent")
    if (
        receipt.embedding_profile_id != expected_profile.profile_id
        or receipt.embedding_profile_hash != expected_profile.profile_hash
        or receipt.model_id != expected_profile.model_id
        or receipt.revision != expected_profile.revision
    ):
        raise ModelProvisioningError("provisioned model belongs to another profile")
    actual = _receipt_from_files(files, profile=expected_profile)
    if actual.receipt_hash != receipt.receipt_hash or actual != receipt:
        raise ModelProvisioningError("provisioned model files differ from their receipt")
    return receipt


def verify_provisioned_reranker_model(
    model_path: str | Path,
    *,
    expected_profile: RerankerProfile,
) -> RerankerProvisioningReceipt:
    """Rehash one local reranker tree and bind it to its exact frozen profile."""

    _require_reranker_profile(expected_profile)
    root = Path(model_path)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ModelProvisioningError("provisioned reranker must be a real absolute directory")
    files, marker_payload = _scan_model_tree(root)
    if marker_payload is None:
        raise ModelProvisioningError("provisioned reranker receipt is missing")
    try:
        raw = json.loads(marker_payload.decode("utf-8"))
        if type(raw) is not dict:
            raise ModelProvisioningError("provisioned reranker receipt is malformed")
        stored_id = raw.pop("receipt_id", None)
        stored_hash = raw.pop("receipt_hash", None)
        stored_schema = raw.pop("schema", None)
        if stored_schema != RerankerProvisioningReceipt.SCHEMA:
            raise ModelProvisioningError("provisioned reranker receipt schema is unsupported")
        if type(raw.get("files")) is list:
            raw["files"] = tuple(raw["files"])
        receipt = RerankerProvisioningReceipt.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ModelProvisioningError):
            raise
        raise ModelProvisioningError("provisioned reranker receipt is malformed") from error
    if stored_id != receipt.receipt_id or stored_hash != receipt.receipt_hash:
        raise ModelProvisioningError("provisioned reranker receipt hash is inconsistent")
    if (
        receipt.reranker_profile_id != expected_profile.profile_id
        or receipt.reranker_profile_hash != expected_profile.profile_hash
        or receipt.model_id != expected_profile.model_id
        or receipt.revision != expected_profile.revision
    ):
        raise ModelProvisioningError("provisioned reranker belongs to another profile")
    actual = _reranker_receipt_from_files(files, profile=expected_profile)
    if actual.receipt_hash != receipt.receipt_hash or actual != receipt:
        raise ModelProvisioningError("provisioned reranker files differ from their receipt")
    return receipt


def _materialize_verified_model_snapshot(
    model_path: Path,
    *,
    expected_profile: EmbeddingModelProfile,
) -> tuple[tempfile.TemporaryDirectory[str], Path, ModelProvisioningReceipt]:
    """Copy exact local bytes to a private stable tree before a model loader sees them."""

    source_receipt = verify_provisioned_model(
        model_path,
        expected_profile=expected_profile,
    )
    temporary = tempfile.TemporaryDirectory(prefix="dithyramba-model-load-")
    temporary_root = Path(temporary.name).resolve(strict=True)
    snapshot = temporary_root / "model"
    try:
        shutil.copytree(
            model_path,
            snapshot,
            symlinks=True,
            ignore=shutil.ignore_patterns(_OPERATIONAL_DIRECTORY),
        )
        snapshot_receipt = verify_provisioned_model(
            snapshot,
            expected_profile=expected_profile,
        )
    except (OSError, shutil.Error, ModelProvisioningError) as error:
        temporary.cleanup()
        if isinstance(error, ModelProvisioningError):
            raise ModelProvisioningError(
                "a stable verified model snapshot could not be created"
            ) from error
        raise ModelProvisioningError(
            "a stable verified model snapshot could not be created"
        ) from error
    if snapshot_receipt != source_receipt:
        temporary.cleanup()
        raise ModelProvisioningError("the model changed while creating its stable snapshot")
    return temporary, snapshot, snapshot_receipt


def _materialize_verified_reranker_snapshot(
    model_path: Path,
    *,
    expected_profile: RerankerProfile,
) -> tuple[tempfile.TemporaryDirectory[str], Path, RerankerProvisioningReceipt]:
    """Copy exact reranker bytes to a private stable tree before model loading."""

    source_receipt = verify_provisioned_reranker_model(
        model_path,
        expected_profile=expected_profile,
    )
    temporary = tempfile.TemporaryDirectory(prefix="dithyramba-reranker-load-")
    temporary_root = Path(temporary.name).resolve(strict=True)
    snapshot = temporary_root / "model"
    try:
        shutil.copytree(
            model_path,
            snapshot,
            symlinks=True,
            ignore=shutil.ignore_patterns(_OPERATIONAL_DIRECTORY),
        )
        snapshot_receipt = verify_provisioned_reranker_model(
            snapshot,
            expected_profile=expected_profile,
        )
    except (OSError, shutil.Error, ModelProvisioningError) as error:
        temporary.cleanup()
        if isinstance(error, ModelProvisioningError):
            raise ModelProvisioningError(
                "a stable verified reranker snapshot could not be created"
            ) from error
        raise ModelProvisioningError(
            "a stable verified reranker snapshot could not be created"
        ) from error
    if snapshot_receipt != source_receipt:
        temporary.cleanup()
        raise ModelProvisioningError("the reranker changed while creating its stable snapshot")
    return temporary, snapshot, snapshot_receipt


def _build_receipt(
    root: Path,
    *,
    profile: EmbeddingModelProfile,
) -> ModelProvisioningReceipt:
    files, _marker_payload = _scan_model_tree(root)
    return _receipt_from_files(files, profile=profile)


def _build_reranker_receipt(
    root: Path,
    *,
    profile: RerankerProfile,
) -> RerankerProvisioningReceipt:
    files, _marker_payload = _scan_model_tree(root)
    return _reranker_receipt_from_files(files, profile=profile)


def _receipt_from_files(
    files: tuple[ProvisionedModelFile, ...],
    *,
    profile: EmbeddingModelProfile,
) -> ModelProvisioningReceipt:
    _require_pinned_inventory(
        model_id=profile.model_id,
        revision=profile.revision,
        files=files,
    )
    return ModelProvisioningReceipt(
        embedding_profile_id=profile.profile_id,
        embedding_profile_hash=profile.profile_hash,
        model_id=profile.model_id,
        revision=profile.revision,
        files=files,
    )


def _reranker_receipt_from_files(
    files: tuple[ProvisionedModelFile, ...],
    *,
    profile: RerankerProfile,
) -> RerankerProvisioningReceipt:
    _require_pinned_inventory(
        model_id=profile.model_id,
        revision=profile.revision,
        files=files,
    )
    return RerankerProvisioningReceipt(
        reranker_profile_id=profile.profile_id,
        reranker_profile_hash=profile.profile_hash,
        model_id=profile.model_id,
        revision=profile.revision,
        files=files,
    )


def _require_pinned_inventory(
    *,
    model_id: str,
    revision: str,
    files: tuple[ProvisionedModelFile, ...],
) -> None:
    """Require the complete minimal tree for profiles maintained by Dithyramba.

    Exact-tree receipts prevent later mutation, while this allowlist also prevents
    a successful downloader (or a compromised cache) from finalizing an incomplete
    tree or silently adding alternate executable/weight formats.
    """

    expected = _PINNED_MODEL_FILES.get((model_id, revision))
    if expected is None:
        return
    actual = tuple(item.relative_path for item in files)
    if actual != expected:
        raise ModelProvisioningError(
            "provisioned model file inventory does not match its pinned allowlist"
        )


def _scan_model_tree(
    root: Path,
) -> tuple[tuple[ProvisionedModelFile, ...], bytes | None]:
    """Read one descriptor-bound tree and revalidate every namespace edge."""

    chain = _open_absolute_directory_chain(root)
    root_descriptor = chain[-1].descriptor
    files: list[ProvisionedModelFile] = []
    marker_payload: list[bytes] = []
    fingerprints: list[_TreeEntryFingerprint] = []
    try:
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ModelProvisioningError("provisioned model must be a real absolute directory")
        root_identity = _stat_identity(root_metadata)
        _scan_model_directory(
            root_descriptor,
            parts=(),
            files=files,
            marker_payload=marker_payload,
            fingerprints=fingerprints,
        )
        _revalidate_tree_entries(root_descriptor, fingerprints)
        if _stat_identity(os.fstat(root_descriptor)) != root_identity:
            raise ModelProvisioningError("provisioned model changed during verification")
        _revalidate_directory_chain(chain)
    finally:
        _close_directory_chain(chain)
    return (
        tuple(sorted(files, key=lambda item: item.relative_path.encode("utf-8"))),
        marker_payload[0] if marker_payload else None,
    )


def _scan_model_directory(
    descriptor: int,
    *,
    parts: tuple[str, ...],
    files: list[ProvisionedModelFile],
    marker_payload: list[bytes],
    fingerprints: list[_TreeEntryFingerprint],
) -> None:
    try:
        with os.scandir(descriptor) as scanner:
            entries = tuple(scanner)
    except OSError as error:
        raise ModelProvisioningError("provisioned model directory could not be scanned") from error
    for entry in sorted(entries, key=lambda item: item.name.encode("utf-8")):
        relative_parts = (*parts, entry.name)
        if _OPERATIONAL_DIRECTORY in relative_parts:
            continue
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise ModelProvisioningError("provisioned model changed during verification") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ModelProvisioningError("provisioned model cannot contain symbolic links")
        if stat.S_ISDIR(metadata.st_mode):
            fingerprints.append(
                _TreeEntryFingerprint(
                    parts=relative_parts,
                    identity=_stat_identity(metadata),
                    is_directory=True,
                )
            )
            child = _open_child_directory(descriptor, entry.name, expected=metadata)
            try:
                _scan_model_directory(
                    child,
                    parts=relative_parts,
                    files=files,
                    marker_payload=marker_payload,
                    fingerprints=fingerprints,
                )
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ModelProvisioningError("provisioned model contains an unsupported file type")
        payload, digest, final_metadata = _read_regular_entry(
            descriptor,
            entry.name,
            expected=metadata,
            collect=parts == () and entry.name == _MARKER_NAME,
        )
        fingerprints.append(
            _TreeEntryFingerprint(
                parts=relative_parts,
                identity=_stat_identity(final_metadata),
                is_directory=False,
            )
        )
        if parts == () and entry.name == _MARKER_NAME:
            if marker_payload:
                raise ModelProvisioningError("provisioned model contains duplicate receipts")
            if not 1 <= metadata.st_size <= _MAX_MANIFEST_BYTES:
                raise ModelProvisioningError("provisioned model receipt size is invalid")
            marker_payload.append(payload)
            continue
        files.append(
            ProvisionedModelFile(
                relative_path=PurePosixPath(*relative_parts).as_posix(),
                size_bytes=metadata.st_size,
                sha256=digest,
            )
        )
        if len(files) > _MAX_FILES:
            raise ModelProvisioningError("provisioned model contains too many files")


def _open_absolute_directory_chain(path: Path) -> list[_DirectoryHandle]:
    """Open an absolute path one no-follow component at a time from the root."""

    if not path.is_absolute() or ".." in path.parts or path == Path(os.sep):
        raise ModelProvisioningError("provisioned model must be a real absolute directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    chain: list[_DirectoryHandle] = []
    try:
        descriptor = os.open(os.sep, flags)
        chain.append(_directory_handle(descriptor, None, None))
        for component in path.parts[1:]:
            parent = chain[-1].descriptor
            descriptor = os.open(component, flags, dir_fd=parent)
            chain.append(_directory_handle(descriptor, parent, component))
    except (OSError, ModelProvisioningError) as error:
        _close_directory_chain(chain)
        if isinstance(error, ModelProvisioningError):
            raise
        raise ModelProvisioningError(
            "provisioned model must be a real absolute directory"
        ) from error
    return chain


def _open_or_create_model_root_chain(app_root: Path) -> list[_DirectoryHandle]:
    """Create app/model directories through held descriptors, never mutable pathnames."""

    chain = _open_or_create_absolute_directory_chain(app_root)
    parent = chain[-1].descriptor
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        try:
            descriptor = os.open("models", flags, dir_fd=parent)
        except FileNotFoundError:
            with suppress(FileExistsError):
                os.mkdir("models", 0o700, dir_fd=parent)
            descriptor = os.open("models", flags, dir_fd=parent)
        handle = _directory_handle(descriptor, parent, "models")
        chain.append(handle)
        os.fchmod(handle.descriptor, 0o700)
        metadata = os.fstat(handle.descriptor)
        if os.name == "posix" and (
            metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ModelProvisioningError("model root must be a private local directory")
        _revalidate_directory_chain(chain)
        return chain
    except (OSError, ModelProvisioningError) as error:
        _close_directory_chain(chain)
        if isinstance(error, ModelProvisioningError):
            raise
        raise ModelProvisioningError("model root must be a real local directory") from error


def _open_or_create_absolute_directory_chain(path: Path) -> list[_DirectoryHandle]:
    if not path.is_absolute() or ".." in path.parts or path == Path(os.sep):
        raise ModelProvisioningError("model data root must be an absolute directory")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    chain: list[_DirectoryHandle] = []
    try:
        descriptor = os.open(os.sep, flags)
        chain.append(_directory_handle(descriptor, None, None))
        for component in path.parts[1:]:
            parent = chain[-1].descriptor
            created = False
            try:
                descriptor = os.open(component, flags, dir_fd=parent)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=parent)
                    created = True
                except FileExistsError:
                    pass
                descriptor = os.open(component, flags, dir_fd=parent)
            handle = _directory_handle(descriptor, parent, component)
            if created:
                os.fchmod(handle.descriptor, 0o700)
            chain.append(handle)
    except (OSError, ModelProvisioningError) as error:
        _close_directory_chain(chain)
        if isinstance(error, ModelProvisioningError):
            raise
        raise ModelProvisioningError("model data root could not be created safely") from error
    return chain


def _model_target_exists(root_descriptor: int, target_name: str) -> bool:
    try:
        metadata = os.stat(target_name, dir_fd=root_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ModelProvisioningError("model target could not be inspected") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ModelProvisioningError("provisioned model must be a real absolute directory")
    return True


def _promote_staging(
    source: Path,
    *,
    target_name: str,
    source_parent_descriptor: int,
    model_root_descriptor: int,
) -> None:
    os.rename(
        source.name,
        target_name,
        src_dir_fd=source_parent_descriptor,
        dst_dir_fd=model_root_descriptor,
    )


def _directory_handle(
    descriptor: int,
    parent_descriptor: int | None,
    component: str | None,
) -> _DirectoryHandle:
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise ModelProvisioningError(
            "provisioned model directory could not be inspected"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ModelProvisioningError("provisioned model path contains a non-directory")
    return _DirectoryHandle(
        descriptor=descriptor,
        parent_descriptor=parent_descriptor,
        component=component,
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def _revalidate_directory_chain(chain: list[_DirectoryHandle]) -> None:
    for handle in chain:
        try:
            descriptor_metadata = os.fstat(handle.descriptor)
        except OSError as error:
            raise ModelProvisioningError("provisioned model changed during verification") from error
        if not stat.S_ISDIR(descriptor_metadata.st_mode) or (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
        ) != (handle.device, handle.inode):
            raise ModelProvisioningError("provisioned model changed during verification")
        if handle.parent_descriptor is None or handle.component is None:
            continue
        try:
            entry_metadata = os.stat(
                handle.component,
                dir_fd=handle.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ModelProvisioningError("provisioned model changed during verification") from error
        if not stat.S_ISDIR(entry_metadata.st_mode) or (
            entry_metadata.st_dev,
            entry_metadata.st_ino,
        ) != (handle.device, handle.inode):
            raise ModelProvisioningError("provisioned model changed during verification")


def _close_directory_chain(chain: list[_DirectoryHandle]) -> None:
    for handle in reversed(chain):
        os.close(handle.descriptor)


def _revalidate_tree_entries(
    root_descriptor: int,
    fingerprints: list[_TreeEntryFingerprint],
) -> None:
    expected_by_path = {item.parts: item for item in fingerprints}
    for fingerprint in fingerprints:
        descriptor = os.dup(root_descriptor)
        try:
            for index, component in enumerate(fingerprint.parts[:-1], start=1):
                expected = expected_by_path.get(fingerprint.parts[:index])
                if expected is None or not expected.is_directory:
                    raise ModelProvisioningError("provisioned model changed during verification")
                child_metadata = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(child_metadata.st_mode)
                    or _stat_identity(child_metadata) != expected.identity
                ):
                    raise ModelProvisioningError("provisioned model changed during verification")
                child = _open_child_directory(
                    descriptor,
                    component,
                    expected=child_metadata,
                )
                os.close(descriptor)
                descriptor = child
            metadata = os.stat(
                fingerprint.parts[-1],
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISDIR(metadata.st_mode) != fingerprint.is_directory
                or _stat_identity(metadata) != fingerprint.identity
            ):
                raise ModelProvisioningError("provisioned model changed during verification")
        except OSError as error:
            raise ModelProvisioningError("provisioned model changed during verification") from error
        finally:
            os.close(descriptor)


def _open_child_directory(parent_descriptor: int, name: str, *, expected: os.stat_result) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        actual = os.fstat(descriptor)
    except OSError as error:
        raise ModelProvisioningError("provisioned model changed during verification") from error
    if (
        not stat.S_ISDIR(actual.st_mode)
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
    ):
        os.close(descriptor)
        raise ModelProvisioningError("provisioned model changed during verification")
    return descriptor


def _read_regular_entry(
    parent_descriptor: int,
    name: str,
    *,
    expected: os.stat_result,
    collect: bool,
) -> tuple[bytes, str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        before = os.fstat(descriptor)
    except OSError as error:
        raise ModelProvisioningError("provisioned model changed during verification") from error
    try:
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != expected.st_dev
            or before.st_ino != expected.st_ino
            or before.st_nlink != 1
        ):
            raise ModelProvisioningError(
                "provisioned model files must be regular and privately linked"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
            total += len(chunk)
            digest.update(chunk)
            if collect:
                if total > _MAX_MANIFEST_BYTES:
                    raise ModelProvisioningError("provisioned model receipt size is invalid")
                chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after) or total != after.st_size:
            raise ModelProvisioningError("provisioned model changed during verification")
        return b"".join(chunks), digest.hexdigest(), after
    finally:
        os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _write_marker(
    root: Path,
    receipt: ModelProvisioningReceipt | RerankerProvisioningReceipt,
) -> None:
    marker = root / _MARKER_NAME
    temporary = root / f"{_MARKER_NAME}.partial"
    payload = canonical_json_bytes(receipt.stored_payload()) + b"\n"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(marker)


def _run_hf(arguments: tuple[str, ...], *, timeout_seconds: int) -> int:
    completed = subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )
    return completed.returncode


def _require_profile(profile: object) -> EmbeddingModelProfile:
    if type(profile) is not EmbeddingModelProfile:
        raise ModelProvisioningError("provisioning requires an exact EmbeddingModelProfile")
    return profile


def _require_reranker_profile(profile: object) -> RerankerProfile:
    if type(profile) is not RerankerProfile:
        raise ModelProvisioningError("provisioning requires an exact RerankerProfile")
    return profile
