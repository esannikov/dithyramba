"""Offline semantic provider tests with no heavyweight optional dependency."""

from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dithyramba.recall import semantic_provider
from dithyramba.recall.hybrid_models import EmbeddingModelProfile, ModelRuntimeProfile
from dithyramba.recall.provisioning import provision_model
from dithyramba.recall.semantic_provider import (
    SemanticProviderError,
    SemanticRuntimeUnavailableError,
    SentenceTransformersProvider,
    current_model_runtime_profile,
)
from dithyramba.recall.semantic_spans import ContentTokenOffset

REVISION = "a" * 40


def _profile(*, dimensions: int = 2, max_tokens: int = 64) -> EmbeddingModelProfile:
    return EmbeddingModelProfile(
        model_id="intfloat/multilingual-e5-small",
        revision=REVISION,
        license="MIT",
        dimensions=dimensions,
        max_tokens=max_tokens,
        query_prefix="query: ",
        passage_prefix="passage: ",
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


class _Matrix:
    def __init__(self, rows: object) -> None:
        self._rows = rows

    def tolist(self) -> object:
        return self._rows


class _Tokenizer:
    def __init__(
        self,
        *,
        model_max_length: int = 512,
        lengths: dict[str, object] | None = None,
        offsets: dict[str, object] | None = None,
        output: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.model_max_length = model_max_length
        self.lengths = lengths or {}
        self.offsets = offsets or {}
        self.output = output
        self.error = error
        self.calls: list[dict[str, object]] = []

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
        return_offsets_mapping: bool = False,
    ) -> object:
        call: dict[str, object] = {
            "text": text,
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "return_length": return_length,
            "return_attention_mask": return_attention_mask,
            "return_token_type_ids": return_token_type_ids,
            "verbose": verbose,
        }
        self.calls.append(call)
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return self.output
        value = text[0]
        if return_offsets_mapping:
            return {"offset_mapping": [self.offsets.get(value, [])]}
        return {"length": [self.lengths.get(value, len(value.split()) + 2)]}


class _Model:
    def __init__(
        self,
        *,
        dimensions: int = 2,
        max_seq_length: int = 512,
        tokenizer: _Tokenizer | None = None,
        output: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer or _Tokenizer(model_max_length=max_seq_length)
        self.output = output
        self.error = error
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimensions

    def encode(self, sentences: list[str], **kwargs: object) -> object:
        self.calls.append((sentences, kwargs))
        if self.error is not None:
            raise self.error
        if self.output is not None:
            return self.output
        return _Matrix([[1.0, 0.0] for _item in sentences])


class _ModernModel(_Model):
    def __init__(
        self,
        *,
        dimensions: int = 2,
        max_seq_length: int = 512,
        output: object | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__(
            dimensions=dimensions,
            max_seq_length=max_seq_length,
            output=output,
            error=error,
        )
        self.modern_dimension_calls = 0

    def get_embedding_dimension(self) -> int:
        self.modern_dimension_calls += 1
        return self.dimensions


def _provisioned_path(tmp_path: Path, profile: EmbeddingModelProfile) -> Path:
    def download(arguments: tuple[str, ...], *, timeout_seconds: int) -> int:
        assert timeout_seconds > 0
        target = Path(arguments[arguments.index("--local-dir") + 1])
        (target / "config.json").write_text("{}\n", encoding="utf-8")
        return 0

    return provision_model(
        profile,
        data_root=tmp_path,
        allow_network=True,
        command=download,
    )[0]


def _provider(
    tmp_path: Path,
    *,
    model: _Model | None = None,
    profile: EmbeddingModelProfile | None = None,
) -> tuple[SentenceTransformersProvider, _Model]:
    selected_profile = profile or _profile()
    model_path = _provisioned_path(tmp_path, selected_profile)
    loaded = model or _Model()
    provider = SentenceTransformersProvider(
        model_profile=selected_profile,
        runtime_profile=_runtime(),
        model_path=model_path,
        _loader=lambda _path, _runtime_profile: loaded,
        _package_version=_versions,
    )
    return provider, loaded


def test_provider_is_offline_profile_bound_and_prefix_exact(tmp_path: Path) -> None:
    provider, model = _provider(tmp_path)

    query = provider.embed_query("Що таке agency?")
    passages = provider.embed_passages(("Перше", "Друге"))

    assert provider.model_profile == _profile()
    assert provider.runtime_profile == _runtime()
    assert provider.model_path == tmp_path / "models" / _profile().profile_id
    assert provider.provisioning_receipt.embedding_profile_hash == _profile().profile_hash
    assert model.max_seq_length == 64
    assert query.values == (1.0, 0.0)
    assert len(passages) == 2
    assert model.calls[0][0] == ["query: Що таке agency?"]
    assert model.calls[1][0] == ["passage: Перше", "passage: Друге"]
    assert model.calls[0][1] == {
        "batch_size": 7,
        "show_progress_bar": False,
        "output_value": "sentence_embedding",
        "precision": "float32",
        "convert_to_numpy": True,
        "convert_to_tensor": False,
        "normalize_embeddings": True,
    }


def test_provider_prefers_the_current_dimension_api(tmp_path: Path) -> None:
    model = _ModernModel()
    provider, loaded = _provider(tmp_path, model=model)

    assert loaded is model
    assert model.modern_dimension_calls == 1
    provider.close()


def test_provider_requires_existing_absolute_path_and_exact_profiles(tmp_path: Path) -> None:
    loader = lambda _path, _runtime_profile: _Model()  # noqa: E731
    with pytest.raises(SemanticRuntimeUnavailableError, match="absolute local"):
        SentenceTransformersProvider(
            model_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=Path("relative"),
            _loader=loader,
            _package_version=_versions,
        )
    with pytest.raises(SemanticRuntimeUnavailableError, match="absolute local"):
        SentenceTransformersProvider(
            model_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=tmp_path / "missing",
            _loader=loader,
            _package_version=_versions,
        )

    model_path = tmp_path / "model"
    model_path.mkdir()
    with pytest.raises(SemanticProviderError, match="model_profile"):
        SentenceTransformersProvider(
            model_profile=cast(Any, object()),
            runtime_profile=_runtime(),
            model_path=model_path,
            _loader=loader,
            _package_version=_versions,
        )
    with pytest.raises(SemanticProviderError, match="runtime_profile"):
        SentenceTransformersProvider(
            model_profile=_profile(),
            runtime_profile=cast(Any, object()),
            model_path=model_path,
            _loader=loader,
            _package_version=_versions,
        )


def test_provider_rejects_version_model_shape_and_loader_drift(tmp_path: Path) -> None:
    model_path = _provisioned_path(tmp_path, _profile())
    with pytest.raises(SemanticProviderError, match="version differs"):
        SentenceTransformersProvider(
            model_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=model_path,
            _loader=lambda _path, _runtime_profile: _Model(),
            _package_version=lambda name: "wrong" if name == "torch" else _versions(name),
        )

    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("missing")

    with pytest.raises(SemanticRuntimeUnavailableError, match="package is missing"):
        SentenceTransformersProvider(
            model_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=model_path,
            _loader=lambda _path, _runtime_profile: _Model(),
            _package_version=missing,
        )
    with pytest.raises(SemanticRuntimeUnavailableError, match="could not be loaded"):
        SentenceTransformersProvider(
            model_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=model_path,
            _loader=lambda _path, _runtime_profile: (_ for _ in ()).throw(RuntimeError("bad")),
            _package_version=_versions,
        )
    with pytest.raises(SemanticProviderError, match="dimensions"):
        _provider(tmp_path, model=_Model(dimensions=3))
    with pytest.raises(SemanticProviderError, match="token limit"):
        _provider(tmp_path, model=_Model(max_seq_length=32))
    with pytest.raises(SemanticProviderError, match="tokenizer limit"):
        _provider(tmp_path, model=_Model(tokenizer=_Tokenizer(model_max_length=32)))


def test_current_runtime_profile_is_exact_and_missing_extra_is_clear() -> None:
    profile = current_model_runtime_profile(
        device="mps",
        batch_size=9,
        _package_version=_versions,
    )
    assert profile.device == "mps"
    assert profile.batch_size == 9
    assert profile.sentence_transformers_version == "5.6.0"

    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("missing")

    with pytest.raises(SemanticRuntimeUnavailableError, match=r"semantic.*extra"):
        current_model_runtime_profile(_package_version=missing)


def test_default_loader_enforces_the_receipted_float32_weight_dtype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    model = _Model()

    def sentence_transformer(*args: object, **kwargs: object) -> _Model:
        calls.append((args, kwargs))
        return model

    monkeypatch.setattr(
        "dithyramba.recall.semantic_provider.importlib.import_module",
        lambda _name: SimpleNamespace(SentenceTransformer=sentence_transformer),
    )

    loaded = semantic_provider._default_loader(tmp_path, _runtime())

    assert loaded is model
    assert calls == [
        (
            (str(tmp_path),),
            {
                "device": "cpu",
                "local_files_only": True,
                "trust_remote_code": False,
                "model_kwargs": {"dtype": "float32"},
            },
        )
    ]


@pytest.mark.parametrize(
    "question",
    ["", " padded", "bad\x00", "e\u0301", cast(Any, 7)],
)
def test_query_text_is_strict(tmp_path: Path, question: object) -> None:
    provider, _model = _provider(tmp_path)
    with pytest.raises(SemanticProviderError, match="query text"):
        provider.embed_query(cast(Any, question))


def test_passage_batch_is_bounded_and_text_is_exact(tmp_path: Path) -> None:
    provider, _model = _provider(tmp_path)
    for value in ((), cast(Any, ["text"]), tuple("x" for _item in range(513))):
        with pytest.raises(SemanticProviderError, match="1-512"):
            provider.embed_passages(value)
    for value in (("",), ("bad\x00",), ("e\u0301",), (cast(Any, 7),)):
        with pytest.raises(SemanticProviderError, match="passage text"):
            provider.embed_passages(value)
    with pytest.raises(SemanticProviderError, match="character budget"):
        provider.embed_passages(tuple("x" * 4_000 for _item in range(501)))
    with pytest.raises(SemanticProviderError, match="character budget"):
        provider.embed_passages(("x" * 2_000_001,))


def test_provider_encodes_character_long_passage_when_exact_token_count_fits(
    tmp_path: Path,
) -> None:
    text = "x" * 40_000
    prepared = f"passage: {text}"
    tokenizer = _Tokenizer(model_max_length=64, lengths={prepared: 64})
    provider, model = _provider(tmp_path, model=_Model(tokenizer=tokenizer))

    vectors = provider.embed_passages((text,))

    assert len(vectors) == 1
    assert model.calls[0][0] == [prepared]
    assert tokenizer.calls[0]["text"] == [prepared]
    assert tokenizer.calls[0]["truncation"] is False


def test_prepared_token_audit_accommodates_full_parent_plus_passage_prefix(
    tmp_path: Path,
) -> None:
    profile = _profile()
    prepared = f"{profile.passage_prefix}{'x' * 2_000_000}"
    tokenizer = _Tokenizer(model_max_length=64, lengths={prepared: 64})
    provider, model = _provider(
        tmp_path,
        profile=profile,
        model=_Model(tokenizer=tokenizer),
    )

    assert provider.prepared_token_counts((prepared,)) == (64,)
    assert model.calls == []
    with pytest.raises(SemanticProviderError, match="prepared embedding input text"):
        provider.prepared_token_counts((f"{prepared}x",))

    over_batch = f"{profile.passage_prefix}{'x' * 1_000_001}"
    with pytest.raises(SemanticProviderError, match="character budget"):
        provider.prepared_token_counts((over_batch, over_batch))


def test_provider_audits_complete_inputs_and_forbids_silent_truncation(
    tmp_path: Path,
) -> None:
    tokenizer = _Tokenizer(
        model_max_length=64,
        lengths={"passage: fits": 64, "passage: too long": 65},
    )
    provider, model = _provider(tmp_path, model=_Model(tokenizer=tokenizer))

    assert provider.prepared_token_counts(("passage: fits", "passage: too long")) == (64, 65)
    assert provider.count_prepared_tokens_no_truncation("passage: fits") == 64
    assert len(provider.embed_passages(("fits",))) == 1
    assert len(model.calls) == 1

    with pytest.raises(SemanticProviderError, match=r"65 tokens.*limit is 64.*truncation"):
        provider.embed_passages(("too long",))
    assert len(model.calls) == 1
    assert tokenizer.calls[0] == {
        "text": ["passage: fits"],
        "add_special_tokens": True,
        "padding": False,
        "truncation": False,
        "return_length": True,
        "return_attention_mask": False,
        "return_token_type_ids": False,
        "verbose": False,
    }


def test_provider_exposes_exact_no_special_token_offsets_for_span_planning(
    tmp_path: Path,
) -> None:
    text = "Привіт світе"
    tokenizer = _Tokenizer(offsets={text: [(0, 6), (7, 12)]})
    provider, model = _provider(tmp_path, model=_Model(tokenizer=tokenizer))

    assert provider.content_token_offsets_no_special_tokens(text) == (
        ContentTokenOffset(0, 6),
        ContentTokenOffset(7, 12),
    )
    assert model.calls == []
    assert tokenizer.calls == [
        {
            "text": [text],
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
            "return_length": False,
            "return_attention_mask": False,
            "return_token_type_ids": False,
            "verbose": False,
        }
    ]


@pytest.mark.parametrize(
    ("output", "match"),
    [
        ({}, "omitted"),
        ({"offset_mapping": []}, "offset batch"),
        ({"offset_mapping": [[(0, 0)]]}, "content offsets"),
        ({"offset_mapping": [[(0, 1, 2)]]}, "content offsets"),
        ({"offset_mapping": [[(False, 1)]]}, "content offsets"),
        ({"offset_mapping": "bad"}, "offset batch"),
    ],
)
def test_provider_offset_mapping_fails_closed(
    tmp_path: Path,
    output: object,
    match: str,
) -> None:
    tokenizer = _Tokenizer(output=output)
    provider, model = _provider(tmp_path, model=_Model(tokenizer=tokenizer))

    with pytest.raises(SemanticProviderError, match=match):
        provider.content_token_offsets_no_special_tokens("a")
    assert model.calls == []


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
def test_provider_token_audit_fails_closed(
    tmp_path: Path,
    tokenizer: _Tokenizer,
    match: str,
) -> None:
    provider, model = _provider(tmp_path, model=_Model(tokenizer=tokenizer))
    with pytest.raises(SemanticProviderError, match=match):
        provider.embed_query("query")
    assert model.calls == []


@pytest.mark.parametrize(
    ("output", "match"),
    [
        (_Matrix("not-a-matrix"), "numeric matrix"),
        (_Matrix(["not-a-row"]), "matrix row"),
        (_Matrix([[True, 0.0]]), "non-float"),
        (_Matrix([[object(), 0.0]]), "non-float"),
        (_Matrix([[math.inf, 0.0]]), "non-finite"),
        (_Matrix([]), "cardinality"),
        (_Matrix([[1.0, 0.0, 0.0]]), "dimensions"),
        (_Matrix([[0.5, 0.5]]), "vector contract"),
    ],
)
def test_provider_validates_every_output_before_use(
    tmp_path: Path,
    output: object,
    match: str,
) -> None:
    provider, _model = _provider(tmp_path, model=_Model(output=output))
    with pytest.raises(SemanticProviderError, match=match):
        provider.embed_query("query")


def test_provider_wraps_inference_failure_and_default_missing_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _model = _provider(tmp_path, model=_Model(error=RuntimeError("provider detail")))
    with pytest.raises(SemanticProviderError, match="failed closed"):
        provider.embed_query("query")

    model_path = _provisioned_path(tmp_path, _profile())

    def missing_semantic_extra(_name: str) -> object:
        raise ImportError("semantic extra unavailable")

    monkeypatch.setattr(
        "dithyramba.recall.semantic_provider.importlib.import_module", missing_semantic_extra
    )
    with pytest.raises(SemanticRuntimeUnavailableError, match=r"semantic.*extra"):
        SentenceTransformersProvider(
            model_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=model_path,
            _package_version=_versions,
        )


def test_provider_rejects_tampered_or_unreceipted_weights_before_loader(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    called = False

    def loader(_path: Path, _runtime_profile: ModelRuntimeProfile) -> _Model:
        nonlocal called
        called = True
        return _Model()

    with pytest.raises(SemanticRuntimeUnavailableError, match="exact verified"):
        SentenceTransformersProvider(
            model_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=raw,
            _loader=loader,
            _package_version=_versions,
        )
    assert called is False

    path = _provisioned_path(tmp_path, _profile())
    (path / "config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(SemanticRuntimeUnavailableError, match="exact verified"):
        SentenceTransformersProvider(
            model_profile=_profile(),
            runtime_profile=_runtime(),
            model_path=path,
            _loader=loader,
            _package_version=_versions,
        )
    assert called is False


def test_provider_loads_from_a_private_stable_snapshot(
    tmp_path: Path,
) -> None:
    profile = _profile()
    original_path = _provisioned_path(tmp_path, profile)
    original_file = original_path / "config.json"
    original_bytes = original_file.read_bytes()
    observed: list[bytes] = []
    stable_paths: list[Path] = []

    def loader(stable_path: Path, _runtime_profile: ModelRuntimeProfile) -> _Model:
        stable_paths.append(stable_path)
        original_file.write_bytes(b"transient tamper")
        observed.append((stable_path / "config.json").read_bytes())
        original_file.write_bytes(original_bytes)
        return _Model()

    provider = SentenceTransformersProvider(
        model_profile=profile,
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
