"""Offline-only Sentence Transformers CrossEncoder reranker adapter."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from .hybrid_models import HybridContractError, ModelRuntimeProfile
from .provisioning import (
    ModelProvisioningError,
    RerankerProvisioningReceipt,
    _materialize_verified_reranker_snapshot,
    verify_provisioned_reranker_model,
)
from .rerank import RerankCandidate, RerankerProfile, RerankerScore

_MAX_CANDIDATES = 500
_MAX_BATCH_CHARACTERS = 5_000_000


class RerankerProviderError(HybridContractError):
    """A local reranker runtime cannot satisfy its receipted profile."""


class RerankerRuntimeUnavailableError(RerankerProviderError):
    """The optional local runtime or explicitly provisioned weights are absent."""


class _PairTokenizer(Protocol):
    model_max_length: int

    def __call__(
        self,
        *,
        text: list[str],
        text_pair: list[str],
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_length: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
        verbose: bool,
    ) -> object: ...


class _CrossEncoderModel(Protocol):
    max_seq_length: int

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        apply_softmax: bool,
        convert_to_numpy: bool,
        convert_to_tensor: bool,
        device: str,
    ) -> object: ...


_ModelLoader = Callable[[Path, RerankerProfile, ModelRuntimeProfile], _CrossEncoderModel]
_PackageVersion = Callable[[str], str]


class SentenceTransformersCrossEncoderProvider:
    """Load one pre-provisioned CrossEncoder and score only supplied candidates."""

    def __init__(
        self,
        *,
        reranker_profile: RerankerProfile,
        runtime_profile: ModelRuntimeProfile,
        model_path: str | Path,
        _loader: _ModelLoader | None = None,
        _package_version: _PackageVersion = importlib.metadata.version,
    ) -> None:
        if type(reranker_profile) is not RerankerProfile:
            raise RerankerProviderError("reranker_profile must be an exact RerankerProfile")
        if type(runtime_profile) is not ModelRuntimeProfile:
            raise RerankerProviderError("runtime_profile must be an exact ModelRuntimeProfile")
        path = Path(model_path).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise RerankerRuntimeUnavailableError(
                "reranker weights must be an existing absolute local directory"
            )
        _verify_runtime_versions(runtime_profile, package_version=_package_version)
        try:
            model_snapshot, stable_model_path, provisioning_receipt = (
                _materialize_verified_reranker_snapshot(
                    path,
                    expected_profile=reranker_profile,
                )
            )
        except ModelProvisioningError as error:
            raise RerankerRuntimeUnavailableError(
                "reranker weights are not an exact verified provisioning"
            ) from error
        loader = _default_loader if _loader is None else _loader
        try:
            model = loader(stable_model_path, reranker_profile, runtime_profile)
        except RerankerProviderError:
            model_snapshot.cleanup()
            raise
        except Exception as error:
            model_snapshot.cleanup()
            raise RerankerRuntimeUnavailableError(
                "the local Sentence Transformers reranker could not be loaded"
            ) from error
        try:
            maximum = getattr(model, "max_seq_length", None)
            tokenizer_object = getattr(model, "tokenizer", None)
            tokenizer_maximum = getattr(tokenizer_object, "model_max_length", None)
            if type(maximum) is not int or maximum != reranker_profile.max_tokens:
                raise RerankerProviderError("loaded reranker token limit differs from its profile")
            if (
                not callable(tokenizer_object)
                or type(tokenizer_maximum) is not int
                or tokenizer_maximum != reranker_profile.max_tokens
            ):
                raise RerankerProviderError(
                    "loaded reranker tokenizer limit differs from its profile"
                )
            receipt_after_load = verify_provisioned_reranker_model(
                stable_model_path,
                expected_profile=reranker_profile,
            )
        except RerankerProviderError:
            model_snapshot.cleanup()
            raise
        except ModelProvisioningError as error:
            model_snapshot.cleanup()
            raise RerankerRuntimeUnavailableError(
                "reranker provisioning changed during model load"
            ) from error
        if receipt_after_load != provisioning_receipt:
            model_snapshot.cleanup()
            raise RerankerRuntimeUnavailableError("reranker provisioning changed during model load")
        self._reranker_profile = reranker_profile
        self._runtime_profile = runtime_profile
        self._model_path = path
        self._model = model
        self._tokenizer = cast("_PairTokenizer", tokenizer_object)
        self._provisioning_receipt = provisioning_receipt
        self._model_snapshot: tempfile.TemporaryDirectory[str] | None = model_snapshot

    @property
    def reranker_profile(self) -> RerankerProfile:
        return self._reranker_profile

    @property
    def runtime_profile(self) -> ModelRuntimeProfile:
        return self._runtime_profile

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def provisioning_receipt(self) -> RerankerProvisioningReceipt:
        return self._provisioning_receipt

    def close(self) -> None:
        snapshot = self._model_snapshot
        if snapshot is not None:
            snapshot.cleanup()
            self._model_snapshot = None

    def score(
        self,
        question: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankerScore, ...]:
        _validate_question(question)
        if (
            type(candidates) is not tuple
            or not candidates
            or len(candidates) > _MAX_CANDIDATES
            or any(type(item) is not RerankCandidate for item in candidates)
        ):
            raise RerankerProviderError("reranker requires 1-500 exact candidates")
        if sum(len(item.text) for item in candidates) > _MAX_BATCH_CHARACTERS:
            raise RerankerProviderError("reranker candidate batch exceeds its character budget")
        texts = tuple(item.text for item in candidates)
        for candidate, token_count in zip(
            candidates,
            self.pair_token_lengths(question, texts),
            strict=True,
        ):
            if token_count > self.reranker_profile.max_tokens:
                raise RerankerProviderError(
                    f"reranker candidate {candidate.source_fragment_id} requires "
                    f"{token_count} tokens; profile limit is "
                    f"{self.reranker_profile.max_tokens}; silent truncation is forbidden"
                )
        # ``score_text_pairs`` deliberately repeats the exact tokenizer check:
        # the provider remains safe even when this fragment-specific wrapper is
        # bypassed by a later versioned candidate contract.
        values = self.score_text_pairs(question, texts)
        if len(values) != len(candidates):
            raise RerankerProviderError("reranker changed the input cardinality")
        return tuple(
            RerankerScore.from_float(candidate.source_fragment_id, score)
            for candidate, score in zip(candidates, values, strict=True)
        )

    def pair_token_lengths(
        self,
        question: str,
        candidate_texts: tuple[str, ...],
    ) -> tuple[int, ...]:
        """Count each complete pair with the exact inference tokenizer, without truncation."""

        _validate_question(question)
        _validate_candidate_texts(candidate_texts)
        return tuple(
            _exact_pair_token_length(self._tokenizer, question, candidate_text)
            for candidate_text in candidate_texts
        )

    def score_text_pairs(
        self,
        question: str,
        candidate_texts: tuple[str, ...],
    ) -> tuple[float, ...]:
        """Score complete text pairs for versioned non-fragment retrieval contracts.

        The public preflight is repeated here so a caller cannot bypass the
        no-truncation boundary. Candidate identity and no-add/no-drop checks
        remain the responsibility of the calling versioned contract.
        """

        lengths = self.pair_token_lengths(question, candidate_texts)
        for index, token_count in enumerate(lengths):
            if token_count > self.reranker_profile.max_tokens:
                raise RerankerProviderError(
                    "reranker pair at candidate index "
                    f"{index} requires {token_count} tokens; "
                    f"profile limit is {self.reranker_profile.max_tokens}; "
                    "silent truncation is forbidden"
                )
        pairs = [(question, text) for text in candidate_texts]
        try:
            raw = self._model.predict(
                pairs,
                batch_size=self.runtime_profile.batch_size,
                show_progress_bar=False,
                apply_softmax=False,
                convert_to_numpy=True,
                convert_to_tensor=False,
                device=self.runtime_profile.device,
            )
            values = _score_values(raw)
        except RerankerProviderError:
            raise
        except Exception as error:
            raise RerankerProviderError("reranker inference failed closed") from error
        if len(values) != len(candidate_texts):
            raise RerankerProviderError("reranker changed the input cardinality")
        return values


def _default_loader(
    path: Path,
    profile: RerankerProfile,
    runtime: ModelRuntimeProfile,
) -> _CrossEncoderModel:
    try:
        module = importlib.import_module("sentence_transformers")
        cross_encoder = cast("Callable[..., object]", module.CrossEncoder)
    except ImportError as error:
        raise RerankerRuntimeUnavailableError(
            "install the 'semantic' extra before loading a reranker"
        ) from error
    return cast(
        "_CrossEncoderModel",
        cross_encoder(
            str(path),
            device=runtime.device,
            local_files_only=True,
            trust_remote_code=False,
            max_length=profile.max_tokens,
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
            raise RerankerRuntimeUnavailableError(
                f"required reranker runtime package is missing: {package}"
            ) from error
        if actual != version:
            raise RerankerProviderError(
                f"installed {package} version differs from the runtime profile"
            )


def _score_values(value: object) -> tuple[float, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RerankerProviderError("reranker did not return a numeric vector")
    result: list[float] = []
    for raw in value:
        if hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, (bool, Sequence)):
            raise RerankerProviderError("reranker returned a non-scalar score")
        try:
            item = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise RerankerProviderError("reranker returned a non-float score") from error
        if not math.isfinite(item):
            raise RerankerProviderError("reranker returned a non-finite score")
        result.append(item)
    return tuple(result)


def _validate_question(question: object) -> str:
    if (
        type(question) is not str
        or not question.strip()
        or question != question.strip()
        or len(question) > 2_000
        or "\x00" in question
        or unicodedata.normalize("NFC", question) != question
    ):
        raise RerankerProviderError("reranker question is invalid")
    return question


def _validate_candidate_texts(candidate_texts: object) -> tuple[str, ...]:
    if (
        type(candidate_texts) is not tuple
        or not candidate_texts
        or len(candidate_texts) > _MAX_CANDIDATES
        or any(
            type(item) is not str
            or not item
            or len(item) > 100_000
            or "\x00" in item
            or unicodedata.normalize("NFC", item) != item
            for item in candidate_texts
        )
    ):
        raise RerankerProviderError("reranker requires 1-500 exact candidate texts")
    if sum(len(item) for item in candidate_texts) > _MAX_BATCH_CHARACTERS:
        raise RerankerProviderError("reranker candidate batch exceeds its character budget")
    return candidate_texts


def _exact_pair_token_length(
    tokenizer: _PairTokenizer,
    question: str,
    candidate_text: str,
) -> int:
    try:
        encoded = tokenizer(
            text=[question],
            text_pair=[candidate_text],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
    except Exception as error:
        raise RerankerProviderError("reranker token preflight failed closed") from error
    if not isinstance(encoded, Mapping) or "length" not in encoded:
        raise RerankerProviderError("reranker tokenizer omitted the exact pair length")
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
        raise RerankerProviderError("reranker tokenizer returned an invalid pair length")
    return raw_lengths[0]
