"""Offline, deterministic, no-add/no-drop contracts for the P7 reranker."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.recall.rerank import (
    BGE_M3_RERANKER_PROFILE,
    RerankCandidate,
    RerankedCandidate,
    RerankerContractError,
    RerankerProfile,
    RerankerScore,
    rerank_candidates,
    run_reranker,
)

REVISION = "a" * 40


def _profile() -> RerankerProfile:
    return RerankerProfile(
        model_id="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        revision=REVISION,
        license="Apache-2.0",
        max_tokens=512,
    )


def _candidate(
    fragment: str,
    *,
    source: str = "source_primary",
    text: str = "Нормалізований фрагмент.",
) -> RerankCandidate:
    return RerankCandidate(
        source_fragment_id=fragment,
        source_id=source,
        text=text,
    )


class RecordingProvider:
    def __init__(self, scores: object) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[RerankCandidate, ...]]] = []

    def score(
        self,
        question: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankerScore, ...]:
        self.calls.append((question, candidates))
        return cast(tuple[RerankerScore, ...], self.scores)


def test_profile_is_frozen_content_addressed_and_strictly_offline() -> None:
    profile = _profile()

    assert profile.semantic_payload() == {
        "schema": "dithyramba.reranker_profile/1.0",
        "provider": "sentence_transformers_cross_encoder",
        "model_id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "revision": REVISION,
        "license": "Apache-2.0",
        "max_tokens": 512,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert profile.profile_id.startswith("reranker_profile_")
    assert len(profile.profile_hash) == 64
    assert profile.profile_hash == _profile().profile_hash
    with pytest.raises(ValidationError):
        cast(Any, profile).model_id = "owner/other"

    assert BGE_M3_RERANKER_PROFILE.semantic_payload() == {
        "schema": "dithyramba.reranker_profile/1.0",
        "provider": "sentence_transformers_cross_encoder",
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "license": "Apache-2.0",
        "max_tokens": 8_192,
        "local_files_only": True,
        "trust_remote_code": False,
    }


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"provider": "remote_api"}, "unsupported reranker provider"),
        ({"model_id": "missing-slash"}, "owner/repository"),
        ({"model_id": "/missing-owner"}, "owner/repository"),
        ({"revision": "main"}, "40-character commit"),
        ({"revision": "A" * 40}, "40-character commit"),
        ({"license": ""}, "bounded ASCII label"),
        ({"license": "not a license"}, "bounded ASCII label"),
        ({"license": "a" * 129}, "bounded ASCII label"),
        ({"local_files_only": False}, "must be local_files_only"),
        ({"trust_remote_code": True}, "disable remote code"),
    ],
)
def test_profile_rejects_unpinned_or_online_configuration(
    update: dict[str, object],
    match: str,
) -> None:
    payload = _profile().model_dump()
    payload.update(update)

    with pytest.raises(ValidationError, match=match):
        RerankerProfile(**payload)


@pytest.mark.parametrize("max_tokens", [0, 32_769, cast(Any, 1.0)])
def test_profile_rejects_invalid_token_limits(max_tokens: Any) -> None:
    payload = _profile().model_dump()
    payload["max_tokens"] = max_tokens

    with pytest.raises(ValidationError):
        RerankerProfile(**payload)


def test_candidate_preserves_authorized_text_and_exact_identifiers() -> None:
    candidate = _candidate("fragment_alpha")

    assert candidate.source_fragment_id == "fragment_alpha"
    assert candidate.source_id == "source_primary"
    assert candidate.text == "Нормалізований фрагмент."


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: _candidate(cast(Any, 1)),
        lambda: _candidate("fragment_ok", source="wrong_source"),
        lambda: _candidate("fragment_"),
        lambda: _candidate("fragment_UPPER"),
        lambda: _candidate("fragment_ok", text=cast(Any, 1)),
        lambda: _candidate("fragment_ok", text=""),
        lambda: _candidate("fragment_ok", text="x" * 100_001),
        lambda: _candidate("fragment_ok", text="has\x00nul"),
        lambda: _candidate("fragment_ok", text="e\u0301"),
    ],
)
def test_candidate_rejects_invalid_or_noncanonical_input(constructor: Any) -> None:
    with pytest.raises(RerankerContractError):
        constructor()


def test_score_round_trips_canonical_python_float_hex() -> None:
    score = RerankerScore.from_float("fragment_alpha", 0.125)

    assert score.score_hex == "0x1.0000000000000p-3"
    assert score.score == 0.125


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: RerankerScore(source_fragment_id="fragment_", score_hex="0x1p+0"),
        lambda: RerankerScore(
            source_fragment_id="fragment_alpha",
            score_hex=cast(Any, 1.0),
        ),
        lambda: RerankerScore(source_fragment_id="fragment_alpha", score_hex="garbage"),
        lambda: RerankerScore(source_fragment_id="fragment_alpha", score_hex="nan"),
        lambda: RerankerScore(source_fragment_id="fragment_alpha", score_hex="inf"),
    ],
)
def test_score_rejects_invalid_identifier_text_or_nonfinite_value(
    constructor: Any,
) -> None:
    with pytest.raises(RerankerContractError):
        constructor()


@pytest.mark.parametrize("score_hex", ["1.0", "0x1p+0", "+0x1.0000000000000p+0"])
def test_score_rejects_noncanonical_float_hex_spellings(score_hex: str) -> None:
    with pytest.raises(RerankerContractError, match=r"canonical float\.hex"):
        RerankerScore(source_fragment_id="fragment_alpha", score_hex=score_hex)


@pytest.mark.parametrize("score", [cast(Any, 1), cast(Any, True), float("nan"), float("inf")])
def test_from_float_requires_a_finite_exact_python_float(score: Any) -> None:
    with pytest.raises(RerankerContractError, match="finite Python float"):
        RerankerScore.from_float("fragment_alpha", score)


def test_rerank_is_score_descending_with_binary_id_ties_and_no_set_change() -> None:
    beta = _candidate("fragment_beta")
    alpha = _candidate("fragment_alpha")
    gamma = _candidate("fragment_gamma")
    tied = RerankerScore.from_float("fragment_beta", 0.5)
    result = rerank_candidates(
        (beta, alpha, gamma),
        (
            tied,
            RerankerScore.from_float("fragment_gamma", 0.75),
            RerankerScore.from_float("fragment_alpha", 0.5),
        ),
    )

    assert all(isinstance(item, RerankedCandidate) for item in result)
    assert tuple(item.candidate.source_fragment_id for item in result) == (
        "fragment_gamma",
        "fragment_alpha",
        "fragment_beta",
    )
    assert tuple(item.rank for item in result) == (1, 2, 3)
    assert result[2].candidate is beta
    assert result[2].score_hex == tied.score_hex


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        ((), "exactly the input candidate IDs"),
        (
            (
                RerankerScore.from_float("fragment_alpha", 0.5),
                RerankerScore.from_float("fragment_extra", 0.4),
            ),
            "exactly the input candidate IDs",
        ),
        (
            (
                RerankerScore.from_float("fragment_alpha", 0.5),
                RerankerScore.from_float("fragment_alpha", 0.4),
            ),
            "duplicate IDs",
        ),
        ((cast(Any, object()),), "invalid score"),
    ],
)
def test_rerank_rejects_drop_add_duplicate_and_untyped_scores(
    scores: tuple[RerankerScore, ...],
    match: str,
) -> None:
    with pytest.raises(RerankerContractError, match=match):
        rerank_candidates((_candidate("fragment_alpha"),), scores)


@pytest.mark.parametrize(
    ("candidates", "match"),
    [
        ((), "1-500 candidates"),
        ((_candidate("fragment_repeated"),) * 501, "1-500 candidates"),
        (
            (_candidate("fragment_duplicate"), _candidate("fragment_duplicate")),
            "must be unique",
        ),
        ((cast(Any, object()),), "invalid candidate"),
    ],
)
def test_rerank_rejects_invalid_candidate_pools(
    candidates: tuple[RerankCandidate, ...],
    match: str,
) -> None:
    with pytest.raises(RerankerContractError, match=match):
        rerank_candidates(candidates, ())


def test_run_invokes_provider_once_with_exact_authorized_pool() -> None:
    candidates = (_candidate("fragment_beta"), _candidate("fragment_alpha"))
    provider = RecordingProvider(
        (
            RerankerScore.from_float("fragment_alpha", 0.5),
            RerankerScore.from_float("fragment_beta", 0.5),
        )
    )

    result = run_reranker(provider, question="Який зв'язок?", candidates=candidates)

    assert provider.calls == [("Який зв'язок?", candidates)]
    assert tuple(item.candidate.source_fragment_id for item in result) == (
        "fragment_alpha",
        "fragment_beta",
    )


def test_run_rejects_provider_without_score_method() -> None:
    with pytest.raises(RerankerContractError, match=r"expose score\(\)"):
        run_reranker(
            cast(Any, object()),
            question="Питання?",
            candidates=(_candidate("fragment_alpha"),),
        )


@pytest.mark.parametrize(
    "question",
    [
        cast(Any, 1),
        "",
        "   ",
        " padded",
        "padded ",
        "e\u0301",
        "has\x00nul",
        "x" * 2_001,
    ],
)
def test_run_rejects_invalid_question_before_provider_call(question: Any) -> None:
    provider = RecordingProvider((RerankerScore.from_float("fragment_alpha", 0.5),))

    with pytest.raises(RerankerContractError, match="question"):
        run_reranker(
            provider,
            question=question,
            candidates=(_candidate("fragment_alpha"),),
        )
    assert provider.calls == []


def test_run_validates_candidate_pool_before_provider_call() -> None:
    provider = RecordingProvider(())

    with pytest.raises(RerankerContractError, match="1-500 candidates"):
        run_reranker(provider, question="Питання?", candidates=())
    assert provider.calls == []


def test_run_requires_provider_to_return_tuple_after_single_call() -> None:
    provider = RecordingProvider([RerankerScore.from_float("fragment_alpha", 0.5)])

    with pytest.raises(RerankerContractError, match="return a tuple"):
        run_reranker(
            provider,
            question="Питання?",
            candidates=(_candidate("fragment_alpha"),),
        )
    assert len(provider.calls) == 1


def test_run_enforces_no_add_no_drop_after_single_provider_call() -> None:
    provider = RecordingProvider((RerankerScore.from_float("fragment_extra", 0.5),))

    with pytest.raises(RerankerContractError, match="exactly the input candidate IDs"):
        run_reranker(
            provider,
            question="Питання?",
            candidates=(_candidate("fragment_alpha"),),
        )
    assert len(provider.calls) == 1
