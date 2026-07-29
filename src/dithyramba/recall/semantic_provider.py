"""Offline-only Sentence Transformers adapter behind a small typed seam."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

from .hybrid_models import EmbeddingModelProfile, HybridContractError, ModelRuntimeProfile
from .provisioning import (
    ModelProvisioningError,
    ModelProvisioningReceipt,
    _materialize_verified_model_snapshot,
    verify_provisioned_model,
)
from .semantic_spans import ContentTokenOffset
from .vector import PackedVector, pack_normalized_vector

_MAX_PASSAGES = 512
_MAX_PASSAGE_BATCH_CHARACTERS = 2_000_000
_MAX_PARENT_CHARACTERS = 2_000_000


class SemanticProviderError(HybridContractError):
    """A local semantic runtime cannot satisfy its receipted profile."""


class SemanticRuntimeUnavailableError(SemanticProviderError):
    """The optional local runtime or explicitly provisioned weights are absent."""


class SemanticProvider(Protocol):
    """Model-agnostic embedding seam used by hybrid orchestration."""

    @property
    def model_profile(self) -> EmbeddingModelProfile: ...

    @property
    def runtime_profile(self) -> ModelRuntimeProfile: ...

    def embed_query(self, question: str) -> PackedVector: ...

    def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]: ...


class _SingleTextTokenizer(Protocol):
    model_max_length: int

    def __call__(
        self,
        *,
        text: list[str],
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_length: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        verbose: bool,
    ) -> object: ...


class _OffsetTokenizer(Protocol):
    def __call__(
        self,
        *,
        text: list[str],
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_length: bool,
        return_offsets_mapping: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        verbose: bool,
    ) -> object: ...


class _SentenceTransformerModel(Protocol):
    max_seq_length: int

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        output_value: str,
        precision: str,
        convert_to_numpy: bool,
        convert_to_tensor: bool,
        normalize_embeddings: bool,
    ) -> object: ...


_ModelLoader = Callable[[Path, ModelRuntimeProfile], _SentenceTransformerModel]
_PackageVersion = Callable[[str], str]


class SentenceTransformersProvider:
    """Load one pre-provisioned model directory and never contact a model hub."""

    def __init__(
        self,
        *,
        model_profile: EmbeddingModelProfile,
        runtime_profile: ModelRuntimeProfile,
        model_path: str | Path,
        _loader: _ModelLoader | None = None,
        _package_version: _PackageVersion = importlib.metadata.version,
    ) -> None:
        if type(model_profile) is not EmbeddingModelProfile:
            raise SemanticProviderError("model_profile must be an exact EmbeddingModelProfile")
        if type(runtime_profile) is not ModelRuntimeProfile:
            raise SemanticProviderError("runtime_profile must be an exact ModelRuntimeProfile")
        path = Path(model_path).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise SemanticRuntimeUnavailableError(
                "semantic model weights must be an existing absolute local directory"
            )
        _verify_runtime_versions(runtime_profile, package_version=_package_version)
        try:
            model_snapshot, stable_model_path, provisioning_receipt = (
                _materialize_verified_model_snapshot(
                    path,
                    expected_profile=model_profile,
                )
            )
        except ModelProvisioningError as error:
            raise SemanticRuntimeUnavailableError(
                "semantic model weights are not an exact verified provisioning"
            ) from error
        loader = _default_loader if _loader is None else _loader
        try:
            model = loader(stable_model_path, runtime_profile)
        except SemanticProviderError:
            model_snapshot.cleanup()
            raise
        except Exception as error:
            model_snapshot.cleanup()
            raise SemanticRuntimeUnavailableError(
                "the local Sentence Transformers model could not be loaded"
            ) from error
        try:
            dimensions = _embedding_dimensions(model)
            if type(dimensions) is not int or dimensions != model_profile.dimensions:
                raise SemanticProviderError("loaded embedding dimensions differ from model profile")
            maximum = model.max_seq_length
            if type(maximum) is not int or maximum < model_profile.max_tokens:
                raise SemanticProviderError("loaded model token limit is below the model profile")
            tokenizer_object = getattr(model, "tokenizer", None)
            tokenizer_maximum = getattr(tokenizer_object, "model_max_length", None)
            if (
                not callable(tokenizer_object)
                or type(tokenizer_maximum) is not int
                or tokenizer_maximum < model_profile.max_tokens
            ):
                raise SemanticProviderError(
                    "loaded embedding tokenizer limit is below the model profile"
                )
            # Make truncation part of the exact runtime contract even when the local
            # model advertises a larger context window.  Input completeness is
            # independently checked with ``truncation=False`` before every encode.
            model.max_seq_length = model_profile.max_tokens
            receipt_after_load = verify_provisioned_model(
                stable_model_path,
                expected_profile=model_profile,
            )
        except SemanticProviderError:
            model_snapshot.cleanup()
            raise
        except ModelProvisioningError as error:
            model_snapshot.cleanup()
            raise SemanticRuntimeUnavailableError(
                "semantic model provisioning changed during model load"
            ) from error
        if receipt_after_load != provisioning_receipt:
            model_snapshot.cleanup()
            raise SemanticRuntimeUnavailableError(
                "semantic model provisioning changed during model load"
            )
        self._model_profile = model_profile
        self._runtime_profile = runtime_profile
        self._model_path = path
        self._model = model
        self._tokenizer = cast("_SingleTextTokenizer", tokenizer_object)
        self._provisioning_receipt = provisioning_receipt
        self._model_snapshot: tempfile.TemporaryDirectory[str] | None = model_snapshot

    @property
    def model_profile(self) -> EmbeddingModelProfile:
        return self._model_profile

    @property
    def runtime_profile(self) -> ModelRuntimeProfile:
        return self._runtime_profile

    @property
    def model_path(self) -> Path:
        """Return the local directory; it is operational state, not semantic identity."""

        return self._model_path

    @property
    def provisioning_receipt(self) -> ModelProvisioningReceipt:
        """Return the exact verified local-weight inventory bound to this provider."""

        return self._provisioning_receipt

    def close(self) -> None:
        """Release the private stable load tree after the provider is no longer used."""

        snapshot = self._model_snapshot
        if snapshot is not None:
            snapshot.cleanup()
            self._model_snapshot = None

    def embed_query(self, question: str) -> PackedVector:
        _validate_text(question, field="query", maximum=2_000, require_unpadded=True)
        return self._encode((self.model_profile.query_prefix + question,))[0]

    def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
        if type(texts) is not tuple or not texts or len(texts) > _MAX_PASSAGES:
            raise SemanticProviderError("passage embedding requires 1-512 texts")
        if sum(len(text) for text in texts if type(text) is str) > _MAX_PASSAGE_BATCH_CHARACTERS:
            raise SemanticProviderError("passage embedding batch exceeds its character budget")
        prepared: list[str] = []
        for text in texts:
            _validate_text(
                text,
                field="passage",
                maximum=_MAX_PARENT_CHARACTERS,
                require_unpadded=False,
            )
            prepared.append(self.model_profile.passage_prefix + text)
        return self._encode(tuple(prepared))

    def prepared_token_counts(self, texts: tuple[str, ...]) -> tuple[int, ...]:
        """Count complete already-prefixed inputs with the inference tokenizer.

        No truncation or padding is permitted.  This method is the audit seam
        for deciding whether an exact SourceFragment fits the frozen model
        profile before any vector is created.
        """

        if type(texts) is not tuple or not texts or len(texts) > _MAX_PASSAGES:
            raise SemanticProviderError("token audit requires 1-512 prepared texts")
        for text in texts:
            _validate_text(
                text,
                field="prepared embedding input",
                maximum=_MAX_PARENT_CHARACTERS + len(self.model_profile.passage_prefix),
                require_unpadded=False,
            )
        prepared_prefix_budget = len(self.model_profile.passage_prefix) * len(texts)
        if sum(len(text) for text in texts) > (
            _MAX_PASSAGE_BATCH_CHARACTERS + prepared_prefix_budget
        ):
            raise SemanticProviderError("token audit batch exceeds its character budget")
        return tuple(_exact_token_length(self._tokenizer, text) for text in texts)

    def count_prepared_tokens_no_truncation(self, prepared_text: str) -> int:
        """Implement the exact single-input seam used by semantic coverage audits."""

        return self.prepared_token_counts((prepared_text,))[0]

    def content_token_offsets_no_special_tokens(
        self,
        text: str,
    ) -> tuple[ContentTokenOffset, ...]:
        """Return parent-text offsets from the same verified inference tokenizer."""

        _validate_text(
            text,
            field="semantic span parent",
            maximum=_MAX_PARENT_CHARACTERS,
            require_unpadded=False,
        )
        return _exact_content_token_offsets(cast("_OffsetTokenizer", self._tokenizer), text)

    def _encode(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
        token_counts = self.prepared_token_counts(texts)
        for index, token_count in enumerate(token_counts):
            if token_count > self.model_profile.max_tokens:
                raise SemanticProviderError(
                    "embedding input at index "
                    f"{index} requires {token_count} tokens; profile limit is "
                    f"{self.model_profile.max_tokens}; silent truncation is forbidden"
                )
        try:
            raw = self._model.encode(
                list(texts),
                batch_size=self.runtime_profile.batch_size,
                show_progress_bar=False,
                output_value="sentence_embedding",
                precision="float32",
                convert_to_numpy=True,
                convert_to_tensor=False,
                normalize_embeddings=True,
            )
            rows = _matrix_rows(raw)
            if len(rows) != len(texts):
                raise SemanticProviderError("embedding provider changed the input cardinality")
            vectors = tuple(pack_normalized_vector(row) for row in rows)
        except SemanticProviderError:
            raise
        except HybridContractError as error:
            raise SemanticProviderError("embedding output violates the vector contract") from error
        except Exception as error:
            raise SemanticProviderError("embedding inference failed closed") from error
        if any(vector.dimensions != self.model_profile.dimensions for vector in vectors):
            raise SemanticProviderError("embedding output dimensions differ from model profile")
        return vectors


def current_model_runtime_profile(
    *,
    device: Literal["cpu", "mps", "cuda"] = "cpu",
    batch_size: int = 16,
    _package_version: _PackageVersion = importlib.metadata.version,
) -> ModelRuntimeProfile:
    """Capture exact installed versions without importing Torch at base-package import time."""

    try:
        sentence_transformers_version = _package_version("sentence-transformers")
        transformers_version = _package_version("transformers")
        torch_version = _package_version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise SemanticRuntimeUnavailableError(
            "install the 'semantic' extra before capturing a model runtime profile"
        ) from error
    return ModelRuntimeProfile(
        sentence_transformers_version=sentence_transformers_version,
        transformers_version=transformers_version,
        torch_version=torch_version,
        device=device,
        batch_size=batch_size,
    )


def _embedding_dimensions(model: _SentenceTransformerModel) -> object:
    """Use the current Sentence Transformers API with a legacy compatibility seam."""

    candidate: object = getattr(model, "get_embedding_dimension", None)
    if not callable(candidate):
        candidate = getattr(model, "get_sentence_embedding_dimension", None)
    if not callable(candidate):
        raise SemanticProviderError("loaded model does not expose embedding dimensions")
    getter = cast("Callable[[], object]", candidate)
    return getter()


def _default_loader(
    path: Path,
    runtime: ModelRuntimeProfile,
) -> _SentenceTransformerModel:
    try:
        module = importlib.import_module("sentence_transformers")
        sentence_transformer = cast("Callable[..., object]", module.SentenceTransformer)
    except ImportError as error:
        raise SemanticRuntimeUnavailableError(
            "install the 'semantic' extra before loading an embedding model"
        ) from error
    return cast(
        "_SentenceTransformerModel",
        sentence_transformer(
            str(path),
            device=runtime.device,
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs={"dtype": runtime.dtype},
        ),
    )


def _verify_runtime_versions(
    runtime: ModelRuntimeProfile,
    *,
    package_version: _PackageVersion,
) -> None:
    expected = {
        "sentence-transformers": runtime.sentence_transformers_version,
        "transformers": runtime.transformers_version,
        "torch": runtime.torch_version,
    }
    for package, version in expected.items():
        try:
            actual = package_version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise SemanticRuntimeUnavailableError(
                f"required semantic runtime package is missing: {package}"
            ) from error
        if actual != version:
            raise SemanticProviderError(
                f"installed {package} version differs from the runtime profile"
            )


def _matrix_rows(value: object) -> tuple[tuple[float, ...], ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise SemanticProviderError("embedding provider did not return a numeric matrix")
    rows: list[tuple[float, ...]] = []
    for raw_row in value:
        if isinstance(raw_row, (str, bytes, bytearray)) or not isinstance(raw_row, Sequence):
            raise SemanticProviderError("embedding provider returned a malformed matrix row")
        row: list[float] = []
        for raw_item in raw_row:
            if isinstance(raw_item, bool):
                raise SemanticProviderError("embedding provider returned a non-float value")
            try:
                item = float(raw_item)
            except (TypeError, ValueError, OverflowError) as error:
                raise SemanticProviderError(
                    "embedding provider returned a non-float value"
                ) from error
            if not math.isfinite(item):
                raise SemanticProviderError("embedding provider returned a non-finite value")
            row.append(item)
        rows.append(tuple(row))
    return tuple(rows)


def _validate_text(
    value: object,
    *,
    field: str,
    maximum: int,
    require_unpadded: bool,
) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        raise SemanticProviderError(f"{field} text is invalid")
    if unicodedata.normalize("NFC", value) != value:
        raise SemanticProviderError(f"{field} text must already be NFC")
    if require_unpadded and value != value.strip():
        raise SemanticProviderError(f"{field} text must be unpadded")
    return value


def _exact_token_length(tokenizer: _SingleTextTokenizer, text: str) -> int:
    try:
        encoded = tokenizer(
            text=[text],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
    except Exception as error:
        raise SemanticProviderError("embedding token audit failed closed") from error
    if not isinstance(encoded, Mapping) or "length" not in encoded:
        raise SemanticProviderError("embedding tokenizer omitted the exact input length")
    raw_lengths = encoded["length"]
    if hasattr(raw_lengths, "tolist"):
        raw_lengths = raw_lengths.tolist()
    if (
        isinstance(raw_lengths, (str, bytes, bytearray))
        or not isinstance(raw_lengths, Sequence)
        or len(raw_lengths) != 1
        or type(raw_lengths[0]) is not int
        or raw_lengths[0] < 1
    ):
        raise SemanticProviderError("embedding tokenizer returned an invalid input length")
    return raw_lengths[0]


def _exact_content_token_offsets(
    tokenizer: _OffsetTokenizer,
    text: str,
) -> tuple[ContentTokenOffset, ...]:
    try:
        encoded = tokenizer(
            text=[text],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
    except Exception as error:
        raise SemanticProviderError("embedding offset mapping failed closed") from error
    if not isinstance(encoded, Mapping) or "offset_mapping" not in encoded:
        raise SemanticProviderError("embedding tokenizer omitted the exact offset mapping")
    raw_batches = encoded["offset_mapping"]
    if hasattr(raw_batches, "tolist"):
        raw_batches = raw_batches.tolist()
    if (
        isinstance(raw_batches, (str, bytes, bytearray))
        or not isinstance(raw_batches, Sequence)
        or len(raw_batches) != 1
    ):
        raise SemanticProviderError("embedding tokenizer returned an invalid offset batch")
    raw_offsets = raw_batches[0]
    if isinstance(raw_offsets, (str, bytes, bytearray)) or not isinstance(raw_offsets, Sequence):
        raise SemanticProviderError("embedding tokenizer returned invalid content offsets")
    result: list[ContentTokenOffset] = []
    for raw_offset in raw_offsets:
        if (
            isinstance(raw_offset, (str, bytes, bytearray))
            or not isinstance(raw_offset, Sequence)
            or len(raw_offset) != 2
            or type(raw_offset[0]) is not int
            or type(raw_offset[1]) is not int
        ):
            raise SemanticProviderError("embedding tokenizer returned invalid content offsets")
        try:
            result.append(ContentTokenOffset(raw_offset[0], raw_offset[1]))
        except HybridContractError as error:
            raise SemanticProviderError(
                "embedding tokenizer returned invalid content offsets"
            ) from error
    return tuple(result)
