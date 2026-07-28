"""Deterministic and fail-closed contracts for P7 reciprocal-rank fusion."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import pytest

from dithyramba.recall.fusion import (
    FusedCandidate,
    FusionCandidate,
    FusionContractError,
    reciprocal_rank_fusion,
    select_layer_aware,
)
from dithyramba.recall.hybrid_models import CorpusLayer, FusionProfile


def _candidate(
    fragment: str,
    *,
    source: str = "source_primary",
    family: str = "family_primary",
    layer: CorpusLayer = CorpusLayer.PRIMARY,
) -> FusionCandidate:
    return FusionCandidate(
        source_fragment_id=fragment,
        source_id=source,
        source_family_id=family,
        layer=layer,
    )


def _fused(candidate: FusionCandidate) -> FusedCandidate:
    return FusedCandidate(
        candidate=candidate,
        fts_rank=1,
        dense_rank=None,
        score_numerator=1,
        score_denominator=61,
    )


def test_rrf_uses_exact_k60_scores_and_binary_id_ties() -> None:
    alpha = _candidate("fragment_alpha")
    beta = _candidate("fragment_beta")
    charlie = _candidate("fragment_charlie")
    delta = _candidate("fragment_delta")

    result = reciprocal_rank_fusion(
        (alpha, beta, charlie),
        (beta, alpha, delta),
        profile=FusionProfile(),
        limit=4,
    )

    assert tuple(item.candidate.source_fragment_id for item in result) == (
        "fragment_alpha",
        "fragment_beta",
        "fragment_charlie",
        "fragment_delta",
    )
    assert result[0].exact_score == Fraction(1, 61) + Fraction(1, 62)
    assert result[1].exact_score == result[0].exact_score
    assert (result[0].fts_rank, result[0].dense_rank) == (1, 2)
    assert (result[1].fts_rank, result[1].dense_rank) == (2, 1)
    assert result[2].exact_score == Fraction(1, 63)
    assert result[3].exact_score == result[2].exact_score
    assert result[2].score_repr == "1/63"
    assert result[0].score_numerator == 123
    assert result[0].score_denominator == 3782
    assert result[0].payload(rank=1) == {
        "rank": 1,
        "source_fragment_id": "fragment_alpha",
        "source_id": "source_primary",
        "source_family_id": "family_primary",
        "layer": "primary",
        "fts_rank": 1,
        "dense_rank": 2,
        "score_numerator": 123,
        "score_denominator": 3782,
    }


def test_rrf_handles_empty_and_single_channel_inputs_and_honors_limit() -> None:
    alpha = _candidate("fragment_alpha")
    beta = _candidate("fragment_beta")

    assert reciprocal_rank_fusion((), (), profile=FusionProfile(), limit=1) == ()
    result = reciprocal_rank_fusion(
        (beta, alpha),
        (),
        profile=FusionProfile(),
        limit=1,
    )

    assert len(result) == 1
    assert result[0].candidate is beta
    assert (result[0].fts_rank, result[0].dense_rank) == (1, None)


def test_rrf_rejects_cross_channel_provenance_disagreement() -> None:
    fts = _candidate("fragment_same", source="source_one")
    dense = _candidate("fragment_same", source="source_other")

    with pytest.raises(FusionContractError, match="provenance metadata"):
        reciprocal_rank_fusion((fts,), (dense,), profile=FusionProfile(), limit=1)


@pytest.mark.parametrize("channel", ["fts", "dense"])
def test_rrf_rejects_duplicate_fragment_ids_within_each_channel(channel: str) -> None:
    candidate = _candidate("fragment_duplicate")
    kwargs = {
        "fts": (candidate, candidate) if channel == "fts" else (),
        "dense": (candidate, candidate) if channel == "dense" else (),
    }

    with pytest.raises(FusionContractError, match="duplicate fragment IDs"):
        reciprocal_rank_fusion(**kwargs, profile=FusionProfile(), limit=1)


@pytest.mark.parametrize("channel", ["fts", "dense"])
def test_rrf_rejects_more_than_500_candidates_per_channel(channel: str) -> None:
    candidate = _candidate("fragment_repeated")
    over_limit = (candidate,) * 501
    kwargs = {
        "fts": over_limit if channel == "fts" else (),
        "dense": over_limit if channel == "dense" else (),
    }

    with pytest.raises(FusionContractError, match="exceeds 500"):
        reciprocal_rank_fusion(**kwargs, profile=FusionProfile(), limit=1)


@pytest.mark.parametrize("channel", ["fts", "dense"])
def test_rrf_rejects_non_candidate_channel_members(channel: str) -> None:
    invalid = (cast(Any, object()),)
    kwargs = {
        "fts": invalid if channel == "fts" else (),
        "dense": invalid if channel == "dense" else (),
    }

    with pytest.raises(FusionContractError, match="invalid candidate"):
        reciprocal_rank_fusion(**kwargs, profile=FusionProfile(), limit=1)


@pytest.mark.parametrize("limit", [False, 0, 501, cast(Any, "1")])
def test_rrf_rejects_invalid_limits(limit: Any) -> None:
    with pytest.raises(FusionContractError, match="fusion limit"):
        reciprocal_rank_fusion((), (), profile=FusionProfile(), limit=limit)


def test_rrf_requires_the_exact_fusion_profile_type() -> None:
    with pytest.raises(FusionContractError, match="exact FusionProfile"):
        reciprocal_rank_fusion((), (), profile=cast(Any, object()), limit=1)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: _candidate(cast(Any, 1)),
        lambda: _candidate("fragment_wrong", source="wrong_source"),
        lambda: _candidate("fragment_wrong", family="wrong_family"),
        lambda: _candidate("fragment_"),
        lambda: _candidate("fragment_UPPER"),
        lambda: _candidate("fragment_ok", layer=cast(Any, "primary")),
    ],
)
def test_fusion_candidate_rejects_invalid_identifiers_and_layer(
    constructor: Any,
) -> None:
    with pytest.raises(FusionContractError):
        constructor()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate": cast(Any, object()), "fts_rank": 1, "dense_rank": None},
        {"candidate": _candidate("fragment_ok"), "fts_rank": None, "dense_rank": None},
        {"candidate": _candidate("fragment_ok"), "fts_rank": False, "dense_rank": None},
        {"candidate": _candidate("fragment_ok"), "fts_rank": 0, "dense_rank": None},
        {"candidate": _candidate("fragment_ok"), "fts_rank": 501, "dense_rank": None},
        {"candidate": _candidate("fragment_ok"), "fts_rank": 1, "dense_rank": 501},
    ],
)
def test_fused_candidate_rejects_invalid_metadata(kwargs: dict[str, Any]) -> None:
    with pytest.raises(FusionContractError):
        FusedCandidate(**kwargs, score_numerator=1, score_denominator=61)


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [(False, 61), (1, False), (0, 61), (-1, 61), (1, 0), (1, -1)],
)
def test_fused_candidate_requires_a_positive_exact_fraction(
    numerator: Any,
    denominator: Any,
) -> None:
    with pytest.raises(FusionContractError, match="positive exact fraction"):
        FusedCandidate(
            candidate=_candidate("fragment_ok"),
            fts_rank=1,
            dense_rank=None,
            score_numerator=numerator,
            score_denominator=denominator,
        )


@pytest.mark.parametrize("rank", [False, 0, -1, cast(Any, "1")])
def test_fused_payload_requires_positive_integer_rank(rank: Any) -> None:
    with pytest.raises(FusionContractError, match="positive integer"):
        _fused(_candidate("fragment_ok")).payload(rank=rank)


def test_layer_aware_cap_applies_to_top_five_then_backfills_in_input_order() -> None:
    candidates = tuple(
        _fused(candidate)
        for candidate in (
            _candidate(
                "fragment_a",
                source="source_derived_one",
                family="family_derived_one",
                layer=CorpusLayer.DERIVED,
            ),
            _candidate(
                "fragment_b",
                source="source_derived_one",
                family="family_derived_one",
                layer=CorpusLayer.DERIVED,
            ),
            _candidate(
                "fragment_c",
                source="source_derived_one",
                family="family_derived_one",
                layer=CorpusLayer.DERIVED,
            ),
            _candidate("fragment_d"),
            _candidate(
                "fragment_e",
                source="source_derived_two",
                family="family_derived_two",
                layer=CorpusLayer.DERIVED,
            ),
            _candidate("fragment_f", layer=CorpusLayer.SYNTHESIS),
        )
    )

    result = select_layer_aware(candidates, limit=6, profile=FusionProfile())

    assert tuple(item.candidate.source_fragment_id for item in result) == (
        "fragment_a",
        "fragment_b",
        "fragment_d",
        "fragment_e",
        "fragment_f",
        "fragment_c",
    )
    assert sum(item.candidate.source_id == "source_derived_one" for item in result[:5]) == 2


def test_layer_aware_returns_short_result_when_cap_exhausts_the_pool() -> None:
    candidates = tuple(
        _fused(
            _candidate(
                f"fragment_{suffix}",
                source="source_derived",
                family="family_derived",
                layer=CorpusLayer.DERIVED,
            )
        )
        for suffix in ("one", "two", "three")
    )

    result = select_layer_aware(candidates, limit=5, profile=FusionProfile())

    assert tuple(item.candidate.source_fragment_id for item in result) == (
        "fragment_one",
        "fragment_two",
    )


def test_layer_aware_stops_both_passes_when_top_five_already_fills_limit() -> None:
    candidates = tuple(
        _fused(_candidate(f"fragment_{suffix}")) for suffix in ("a", "b", "c", "d", "e", "f")
    )

    result = select_layer_aware(candidates, limit=5, profile=FusionProfile())

    assert result == candidates[:5]


def test_layer_aware_allows_unclassified_non_derived_items_without_relabeling() -> None:
    candidate = _fused(_candidate("fragment_unclassified", layer=CorpusLayer.UNCLASSIFIED))

    assert select_layer_aware((candidate,), limit=1, profile=FusionProfile()) == (candidate,)


def test_layer_aware_rejects_duplicate_fragment_ids() -> None:
    candidate = _fused(_candidate("fragment_duplicate"))

    with pytest.raises(FusionContractError, match="must be unique"):
        select_layer_aware((candidate, candidate), limit=2, profile=FusionProfile())


def test_layer_aware_rejects_non_fused_items_fail_closed() -> None:
    with pytest.raises(FusionContractError, match="must be FusedCandidate"):
        select_layer_aware(
            (cast(Any, object()),),
            limit=1,
            profile=FusionProfile(),
        )


@pytest.mark.parametrize("limit", [False, 0, 101, cast(Any, "1")])
def test_layer_aware_rejects_invalid_limits(limit: Any) -> None:
    with pytest.raises(FusionContractError, match="selection limit"):
        select_layer_aware((), limit=limit, profile=FusionProfile())


def test_layer_aware_requires_the_exact_fusion_profile_type() -> None:
    with pytest.raises(FusionContractError, match="exact FusionProfile"):
        select_layer_aware((), limit=1, profile=cast(Any, object()))
