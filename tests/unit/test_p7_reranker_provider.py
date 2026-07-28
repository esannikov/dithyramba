"""Offline CrossEncoder provider tests without heavyweight inference."""

from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path
from typing import Any, cast

import pytest

from dithyramba.recall.hybrid_models import ModelRuntimeProfile
from dithyramba.recall.provisioning import provision_reranker_model
from dithyramba.recall.rerank import RerankCandidate, RerankerProfile
from dithyramba.recall.reranker_provider import (
    RerankerProviderError,
    RerankerRuntimeUnavailableError,
    SentenceTransformersCrossEncoderProvider,
)

REVISION = "b" * 40


def _profile(*, max_tokens: int = 64) -> RerankerProfile:
    return RerankerProfile(
        model_id="cross-encoder/test-model",
        revision=REVISION,
        license="Apache-2.0",
        max_tokens=max_tokens,
    )


def _runtime() -> ModelRuntimeProfile:
    return ModelRuntimeProfile(
        sentence_transformers_version="5.6.0",
        transformers_version="4.99.0",
        torch_version="2.9.0",
        batch_size=7,
    )


def _versions(name: str) -> str:
    return {
        "sentence-transformers": "5.6.0",
        "transformers": "4.99.0",
        "torch": "2.9.0",
    }[name]


class _Scores:
    def __init__(self, values: object) -> None:
        self.values = values

    def tolist(self) -> object:
        return self.values


class _Tokenizer:
    def __init__(
        self,
        *,
        model_max_length: int = 64,
        lengths: dict[str, object] | None = None,
        output: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.model_max_length = model_max_length
        self.lengths = lengths or {}
        self.output = output
        self.error = error
        self.calls: list[dict[str, object]] = []

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
    ) -> object:
        kwargs: dict[str, object] = {
            "text": text,
            "text_pair": text_pair,
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "return_length": return_length,
            "return_attention_mask": return_attention_mask,
            "return_token_type_ids": return_token_type_ids,
            "verbose": verbose,
        }
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return self.output
        question = text[0]
        candidate = text_pair[0]
        length = self.lengths.get(
            candidate,
            len(question.split()) + len(candidate.split()) + 4,
        )
        return {"length": [length]}


class _Model:
    def __init__(
        self,
        *,
        max_seq_length: int = 64,
        tokenizer: _Tokenizer | None = None,
        output: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer or _Tokenizer(model_max_length=max_seq_length)
        self.output = output
        self.error = error
        self.calls: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> object:
        self.calls.append((inputs, kwargs))
        if self.error is not None:
            raise self.error
        return self.output if self.output is not None else _Scores([0.5] * len(inputs))


def _provisioned_path(tmp_path: Path, profile: RerankerProfile) -> Path:
    def download(arguments: tuple[str, ...], *, timeout_seconds: int) -> int:
        assert timeout_seconds > 0
        target = Path(arguments[arguments.index("--local-dir") + 1])
        (target / "config.json").write_text("{}\n", encoding="utf-8")
        return 0

    return provision_reranker_model(
        profile,
        data_root=tmp_path,
        allow_network=True,
        command=download,
    )[0]


def _provider(
    tmp_path: Path,
    *,
    model: _Model | None = None,
    profile: RerankerProfile | None = None,
) -> tuple[SentenceTransformersCrossEncoderProvider, _Model]:
    selected = profile or _profile()
    loaded = model or _Model(max_seq_length=selected.max_tokens)
    path = _provisioned_path(tmp_path, selected)
    provider = SentenceTransformersCrossEncoderProvider(
        reranker_profile=selected,
        runtime_profile=_runtime(),
        model_path=path,
        _loader=lambda _path, _profile, _runtime_profile: loaded,
        _package_version=_versions,
    )
    return provider, loaded


def _candidate(fragment_id: str, text: str = "Evidence") -> RerankCandidate:
    return RerankCandidate(
        source_fragment_id=fragment_id,
        source_id="source_one",
        text=text,
    )


def test_provider_is_offline_profile_bound_and_scores_exact_pairs(tmp_path: Path) -> None:
    provider, model = _provider(tmp_path, model=_Model(output=_Scores([0.25, -0.5])))
    candidates = (_candidate("fragment_one"), _candidate("fragment_two", "Other"))

    scores = provider.score("Питання?", candidates)

    assert provider.reranker_profile == _profile()
    assert provider.runtime_profile == _runtime()
    assert provider.model_path == tmp_path / "models" / _profile().profile_id
    assert provider.provisioning_receipt.reranker_profile_hash == _profile().profile_hash
    assert [item.score for item in scores] == [0.25, -0.5]
    assert [item.source_fragment_id for item in scores] == ["fragment_one", "fragment_two"]
    assert model.calls == [
        (
            [("Питання?", "Evidence"), ("Питання?", "Other")],
            {
                "batch_size": 7,
                "show_progress_bar": False,
                "apply_softmax": False,
                "convert_to_numpy": True,
                "convert_to_tensor": False,
                "device": "cpu",
            },
        )
    ]
    provider.close()


def test_provider_requires_exact_profiles_versions_path_and_token_limit(
    tmp_path: Path,
) -> None:
    loader = lambda _path, _profile, _runtime: _Model()  # noqa: E731
    with pytest.raises(RerankerRuntimeUnavailableError, match="absolute local"):
        SentenceTransformersCrossEncoderProvider(
            reranker_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=Path("relative"),
            _loader=loader,
            _package_version=_versions,
        )
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(RerankerProviderError, match="reranker_profile"):
        SentenceTransformersCrossEncoderProvider(
            reranker_profile=cast(Any, object()),
            runtime_profile=_runtime(),
            model_path=raw,
            _loader=loader,
            _package_version=_versions,
        )
    with pytest.raises(RerankerProviderError, match="runtime_profile"):
        SentenceTransformersCrossEncoderProvider(
            reranker_profile=_profile(),
            runtime_profile=cast(Any, object()),
            model_path=raw,
            _loader=loader,
            _package_version=_versions,
        )

    path = _provisioned_path(tmp_path, _profile())
    with pytest.raises(RerankerProviderError, match="version differs"):
        SentenceTransformersCrossEncoderProvider(
            reranker_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=path,
            _loader=loader,
            _package_version=lambda name: "wrong" if name == "torch" else _versions(name),
        )
    with pytest.raises(RerankerProviderError, match="token limit"):
        _provider(tmp_path, model=_Model(max_seq_length=63))
    with pytest.raises(RerankerProviderError, match="tokenizer limit"):
        _provider(
            tmp_path,
            model=_Model(tokenizer=_Tokenizer(model_max_length=63)),
        )


def test_provider_fails_closed_for_missing_runtime_loader_and_unverified_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _provisioned_path(tmp_path, _profile())

    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("missing")

    with pytest.raises(RerankerRuntimeUnavailableError, match="package is missing"):
        SentenceTransformersCrossEncoderProvider(
            reranker_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=path,
            _loader=lambda _path, _profile, _runtime: _Model(),
            _package_version=missing,
        )
    with pytest.raises(RerankerRuntimeUnavailableError, match="could not be loaded"):
        SentenceTransformersCrossEncoderProvider(
            reranker_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=path,
            _loader=lambda _path, _profile, _runtime: (_ for _ in ()).throw(RuntimeError("bad")),
            _package_version=_versions,
        )

    def missing_extra(_name: str) -> object:
        raise ImportError("semantic extra missing")

    monkeypatch.setattr(
        "dithyramba.recall.reranker_provider.importlib.import_module",
        missing_extra,
    )
    with pytest.raises(RerankerRuntimeUnavailableError, match=r"semantic.*reranker"):
        SentenceTransformersCrossEncoderProvider(
            reranker_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=path,
            _package_version=_versions,
        )

    unverified = tmp_path / "unverified"
    unverified.mkdir()
    with pytest.raises(RerankerRuntimeUnavailableError, match="exact verified"):
        SentenceTransformersCrossEncoderProvider(
            reranker_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=unverified,
            _loader=lambda _path, _profile, _runtime: _Model(),
            _package_version=_versions,
        )


@pytest.mark.parametrize(
    ("output", "match"),
    [
        (_Scores("bad"), "numeric vector"),
        (_Scores([[0.1]]), "non-scalar"),
        (_Scores([True]), "non-scalar"),
        (_Scores([object()]), "non-float"),
        (_Scores([math.inf]), "non-finite"),
        (_Scores([]), "cardinality"),
    ],
)
def test_provider_validates_every_score_before_use(
    tmp_path: Path,
    output: object,
    match: str,
) -> None:
    provider, _model = _provider(tmp_path, model=_Model(output=output))
    with pytest.raises(RerankerProviderError, match=match):
        provider.score("Question?", (_candidate("fragment_one"),))


def test_provider_bounds_inputs_and_wraps_inference(tmp_path: Path) -> None:
    provider, _model = _provider(tmp_path)
    for question in ("", cast(Any, 7), "x" * 2_001):
        with pytest.raises(RerankerProviderError, match="question"):
            provider.score(question, (_candidate("fragment_one"),))
    for candidates in ((), cast(Any, []), (_candidate("fragment_one"),) * 501):
        with pytest.raises(RerankerProviderError, match="1-500"):
            provider.score("Question?", candidates)
    huge = tuple(_candidate(f"fragment_{index}", "x" * 100_000) for index in range(51))
    with pytest.raises(RerankerProviderError, match="character budget"):
        provider.score("Question?", huge)

    failing, _model = _provider(tmp_path, model=_Model(error=RuntimeError("detail")))
    with pytest.raises(RerankerProviderError, match="failed closed"):
        failing.score("Question?", (_candidate("fragment_one"),))


def test_provider_preflights_exact_pairs_without_truncation(tmp_path: Path) -> None:
    tokenizer = _Tokenizer(
        model_max_length=64,
        lengths={"fits": 64, "too long": 65},
    )
    provider, model = _provider(tmp_path, model=_Model(tokenizer=tokenizer))

    assert provider.pair_token_lengths("Питання?", ("fits", "too long")) == (64, 65)
    scores = provider.score("Питання?", (_candidate("fragment_one", "fits"),))
    assert [item.score for item in scores] == [0.5]
    assert len(model.calls) == 1

    with pytest.raises(RerankerProviderError, match=r"65 tokens.*limit is 64.*truncation"):
        provider.score("Питання?", (_candidate("fragment_two", "too long"),))
    assert len(model.calls) == 1
    assert tokenizer.calls[0] == {
        "text": ["Питання?"],
        "text_pair": ["fits"],
        "add_special_tokens": True,
        "padding": False,
        "truncation": False,
        "return_length": True,
        "return_attention_mask": False,
        "return_token_type_ids": False,
        "verbose": False,
    }


@pytest.mark.parametrize(
    ("tokenizer", "match"),
    [
        (_Tokenizer(output={}), "omitted"),
        (_Tokenizer(output={"length": []}), "invalid"),
        (_Tokenizer(output={"length": [True]}), "invalid"),
        (_Tokenizer(output={"length": [1.5]}), "invalid"),
        (_Tokenizer(output={"length": [1, 2]}), "invalid"),
        (_Tokenizer(error=RuntimeError("detail")), "failed closed"),
    ],
)
def test_provider_fails_closed_for_invalid_token_preflight(
    tmp_path: Path,
    tokenizer: _Tokenizer,
    match: str,
) -> None:
    provider, model = _provider(tmp_path, model=_Model(tokenizer=tokenizer))
    with pytest.raises(RerankerProviderError, match=match):
        provider.score("Question?", (_candidate("fragment_one"),))
    assert model.calls == []


def test_provider_validates_generic_text_pairs_before_tokenization(tmp_path: Path) -> None:
    provider, _model = _provider(tmp_path)
    for question in (" padded ", "e\u0301", "\x00"):
        with pytest.raises(RerankerProviderError, match="question"):
            provider.pair_token_lengths(question, ("Evidence",))
    for texts in (
        (),
        cast(Any, []),
        ("e\u0301",),
        ("Evidence\x00",),
        (cast(Any, 7),),
    ):
        with pytest.raises(RerankerProviderError, match="candidate texts"):
            provider.pair_token_lengths("Question?", texts)


def test_provider_loads_from_private_stable_snapshot(tmp_path: Path) -> None:
    profile = _profile()
    original_path = _provisioned_path(tmp_path, profile)
    original_file = original_path / "config.json"
    original_bytes = original_file.read_bytes()
    stable_paths: list[Path] = []
    observed: list[bytes] = []

    def loader(
        stable_path: Path,
        _profile: RerankerProfile,
        _runtime: ModelRuntimeProfile,
    ) -> _Model:
        stable_paths.append(stable_path)
        original_file.write_bytes(b"transient tamper")
        observed.append((stable_path / "config.json").read_bytes())
        original_file.write_bytes(original_bytes)
        return _Model()

    provider = SentenceTransformersCrossEncoderProvider(
        reranker_profile=profile,
        runtime_profile=_runtime(),
        model_path=original_path,
        _loader=loader,
        _package_version=_versions,
    )
    assert stable_paths[0] != original_path
    assert observed == [original_bytes]
    assert stable_paths[0].is_dir()
    provider.close()
    assert not stable_paths[0].exists()
    provider.close()
