"""Pure P7 hybrid artifact and hash-domain contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.access import QueryExclusions
from dithyramba.access.errors import InvalidAccessScopeError
from dithyramba.recall.hybrid_models import (
    CorpusLayer,
    CorpusLayerItem,
    CorpusLayerProfile,
    EmbeddingModelProfile,
    FusionProfile,
    HybridContractError,
    HybridEvidencePacket,
    HybridPacketItem,
    HybridQueryRequest,
    HybridRankTraceItem,
    HybridRetrievalBudget,
    HybridRetrievalReceipt,
    HybridRunObservation,
    ModelRuntimeProfile,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
REVISION = "c" * 40


def _model() -> EmbeddingModelProfile:
    return EmbeddingModelProfile(
        model_id="intfloat/multilingual-e5-small",
        revision=REVISION,
        license="MIT",
        dimensions=384,
        max_tokens=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )


def _runtime() -> ModelRuntimeProfile:
    return ModelRuntimeProfile(
        sentence_transformers_version="5.1.2",
        transformers_version="4.53.2",
        torch_version="2.7.1",
    )


def _layer_profile() -> CorpusLayerProfile:
    return CorpusLayerProfile(
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        exclusion_hash=HASH,
        permitted_set_hash=OTHER_HASH,
        items=(
            CorpusLayerItem(source_id="source_two", layer=CorpusLayer.DERIVED),
            CorpusLayerItem(source_id="source_one", layer=CorpusLayer.PRIMARY),
        ),
    )


def _request() -> HybridQueryRequest:
    return HybridQueryRequest(
        question="Що означає agency?",
        library_id="library_one",
        collection_ids=("collection_two", "collection_one"),
        corpus_snapshot_id="snapshot_one",
        access_policy_id="policy_one",
        purpose="research",
        model_profile_hash=_model().profile_hash,
        runtime_profile_hash=_runtime().profile_hash,
        layer_profile_hash=_layer_profile().profile_hash,
        fusion_profile_hash=FusionProfile().profile_hash,
    )


def _trace(fragment: str, rank: int) -> HybridRankTraceItem:
    return HybridRankTraceItem(
        source_fragment_id=fragment,
        source_id="source_one",
        rank=rank,
        score_repr=f"score:{rank}",
    )


def _receipt(*, reranker: bool = False) -> HybridRetrievalReceipt:
    request = _request()
    return HybridRetrievalReceipt(
        request_id=request.request_id,
        request_hash=request.request_hash,
        exclusion_hash=HASH,
        permitted_set_hash=OTHER_HASH,
        model_profile_hash=request.model_profile_hash,
        runtime_profile_hash=request.runtime_profile_hash,
        layer_profile_hash=request.layer_profile_hash,
        fusion_profile_hash=request.fusion_profile_hash,
        vector_generation_hash=HASH,
        fts_profile_version="fts_v1.test",
        fts_result_hash=OTHER_HASH,
        query_vector_hash=HASH,
        reranker_profile_hash=OTHER_HASH if reranker else None,
        fts=(_trace("fragment_one", 1),),
        dense=(_trace("fragment_two", 1),),
        fused=(_trace("fragment_one", 1), _trace("fragment_two", 2)),
        reranked=((_trace("fragment_two", 1), _trace("fragment_one", 2)) if reranker else ()),
    )


def test_model_runtime_layer_and_request_are_content_addressed_and_sorted() -> None:
    model = _model()
    runtime = _runtime()
    layers = _layer_profile()
    request = _request()

    assert model.profile_id.startswith("embedding_profile_")
    assert len(model.profile_hash) == 64
    assert runtime.profile_id.startswith("runtime_profile_")
    assert layers.profile_id.startswith("layer_profile_")
    assert tuple(item.source_id for item in layers.items) == ("source_one", "source_two")
    layers.require_exact_sources(("source_two", "source_one"), benchmark=True)
    assert request.collection_ids == ("collection_one", "collection_two")
    assert request.request_id.startswith("hybrid_query_")
    assert request.canonical_bytes.startswith(b'{"access_policy_id"')
    assert request.semantic_payload()["result_format"] == "hybrid_evidence_packet"
    assert HybridRetrievalBudget().payload()["max_dense_candidates"] == 100


@pytest.mark.parametrize(
    ("constructor", "match"),
    [
        (
            lambda: EmbeddingModelProfile(
                model_id="missing-slash",
                revision=REVISION,
                license="MIT",
                dimensions=384,
                max_tokens=512,
            ),
            "model_id",
        ),
        (
            lambda: EmbeddingModelProfile(
                model_id="owner/model",
                revision="main",
                license="MIT",
                dimensions=384,
                max_tokens=512,
            ),
            "revision",
        ),
        (
            lambda: EmbeddingModelProfile(
                model_id="owner/model",
                revision=REVISION,
                license="bad license",
                dimensions=384,
                max_tokens=512,
            ),
            "license",
        ),
        (
            lambda: EmbeddingModelProfile(
                model_id="owner/model",
                revision=REVISION,
                license="MIT",
                dimensions=384,
                max_tokens=512,
                query_prefix="e\u0301",
            ),
            "NFC",
        ),
        (
            lambda: ModelRuntimeProfile(
                sentence_transformers_version="bad version",
                transformers_version="1",
                torch_version="1",
            ),
            "runtime version",
        ),
    ],
)
def test_model_profile_rejects_ambient_or_mutable_identity(
    constructor: Any,
    match: str,
) -> None:
    with pytest.raises((ValidationError, HybridContractError), match=match):
        constructor()


def test_layer_profile_requires_exact_sources_and_classified_benchmark() -> None:
    layers = _layer_profile()
    with pytest.raises(HybridContractError, match="exact permitted"):
        layers.require_exact_sources(("source_one",))
    with pytest.raises(HybridContractError, match="exact permitted"):
        layers.require_exact_sources(())
    with pytest.raises(HybridContractError, match="unique"):
        layers.require_exact_sources(("source_one", "source_one"))

    unclassified = CorpusLayerProfile(
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        exclusion_hash=HASH,
        permitted_set_hash=HASH,
        items=(CorpusLayerItem(source_id="source_one", layer=CorpusLayer.UNCLASSIFIED),),
    )
    with pytest.raises(HybridContractError, match="unclassified"):
        unclassified.require_exact_sources(("source_one",), benchmark=True)
    empty = CorpusLayerProfile(
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        exclusion_hash=HASH,
        permitted_set_hash=HASH,
        items=(),
    )
    empty.require_exact_sources(())
    with pytest.raises(HybridContractError, match="cannot be empty"):
        empty.require_exact_sources((), benchmark=True)
    with pytest.raises((ValidationError, HybridContractError), match="unique"):
        CorpusLayerProfile(
            library_id="library_one",
            corpus_snapshot_id="snapshot_one",
            exclusion_hash=HASH,
            permitted_set_hash=HASH,
            items=(
                CorpusLayerItem(source_id="source_one", layer=CorpusLayer.PRIMARY),
                CorpusLayerItem(source_id="source_one", layer=CorpusLayer.DERIVED),
            ),
        )


@pytest.mark.parametrize(
    "update",
    [
        {"question": " padded"},
        {"question": "e\u0301"},
        {"question": "bad\x00"},
        {"purpose": "Research"},
        {"library_id": "wrong"},
        {"corpus_snapshot_id": "wrong"},
        {"access_policy_id": "wrong"},
        {"collection_ids": ()},
        {"collection_ids": ("collection_one", "collection_one")},
        {"model_profile_hash": "bad"},
        {"reranker_profile_hash": "bad"},
        {"exclusions": cast(Any, {})},
        {"retrieval": cast(Any, {})},
    ],
)
def test_hybrid_request_rejects_implicit_or_malformed_scope(update: dict[str, object]) -> None:
    with pytest.raises((ValidationError, HybridContractError)):
        HybridQueryRequest(**{**_request().model_dump(), **update})


def test_request_validates_exclusion_identifiers_and_budget() -> None:
    request = _request()
    with pytest.raises(InvalidAccessScopeError, match="identifier"):
        HybridQueryRequest(
            **{
                **request.model_dump(),
                "exclusions": QueryExclusions(source_ids=("wrong",)),
            }
        )
    with pytest.raises((ValidationError, HybridContractError), match="fused"):
        HybridRetrievalBudget(max_fused_candidates=2, max_source_fragments=3)


def test_retrieval_receipt_closes_channels_and_excludes_observation() -> None:
    receipt = _receipt(reranker=True)
    observation = HybridRunObservation(
        processing_run_id="run_one",
        request_id=receipt.request_id,
        cold_model_load_ns=10,
        passage_embedding_ns=20,
        query_embedding_ns=30,
        retrieval_ns=40,
        reranker_ns=50,
        peak_rss_bytes=1_000,
        cache_bytes=2_000,
        provider_calls=2,
        observed_at="2026-07-21T12:00:00Z",
    )

    assert receipt.receipt_id.startswith("hybrid_retrieval_")
    assert "cold_model_load_ns" not in receipt.semantic_payload()
    assert observation.payload()["cold_model_load_ns"] == 10
    assert observation.observation_hash != receipt.receipt_hash


def test_receipt_fails_closed_on_rank_or_candidate_set_drift() -> None:
    base = _receipt()
    with pytest.raises((ValidationError, HybridContractError), match="contiguous"):
        HybridRetrievalReceipt(**{**base.model_dump(), "fts": (_trace("fragment_one", 2),)})
    with pytest.raises((ValidationError, HybridContractError), match="duplicate"):
        HybridRetrievalReceipt(
            **{
                **base.model_dump(),
                "fts": (_trace("fragment_one", 1), _trace("fragment_one", 2)),
            }
        )
    with pytest.raises((ValidationError, HybridContractError), match="only FTS or dense"):
        HybridRetrievalReceipt(**{**base.model_dump(), "fused": (_trace("fragment_hidden", 1),)})
    with pytest.raises((ValidationError, HybridContractError), match="requires"):
        HybridRetrievalReceipt(**{**base.model_dump(), "reranked": (_trace("fragment_one", 1),)})
    with pytest.raises((ValidationError, HybridContractError), match="exact fused"):
        HybridRetrievalReceipt(
            **{
                **base.model_dump(),
                "reranker_profile_hash": HASH,
                "reranked": (_trace("fragment_one", 1),),
            }
        )


def test_hybrid_packet_is_text_free_ranked_and_status_bound() -> None:
    request = _request()
    item = HybridPacketItem(
        rank=1,
        source_fragment_id="fragment_one",
        source_id="source_one",
        source_family_id="family_one",
        layer=CorpusLayer.PRIMARY,
    )
    packet = HybridEvidencePacket(
        request_id=request.request_id,
        request_hash=request.request_hash,
        corpus_snapshot_id=request.corpus_snapshot_id,
        access_policy_id=request.access_policy_id,
        read_receipt_hash=HASH,
        access_receipt_hash=HASH,
        coverage_report_hash=HASH,
        retrieval_receipt_hash=_receipt().receipt_hash,
        result_status="evidence_found",
        items=(item,),
    )
    assert packet.packet_id.startswith("hybrid_packet_")
    payload_items = cast(list[dict[str, object]], packet.semantic_payload()["items"])
    assert "text" not in payload_items[0]

    with pytest.raises((ValidationError, HybridContractError), match="result_status"):
        HybridEvidencePacket(**{**packet.model_dump(), "result_status": "no_evidence"})
    with pytest.raises((ValidationError, HybridContractError), match="contiguous"):
        HybridEvidencePacket(
            **{
                **packet.model_dump(),
                "items": (item.model_copy(update={"rank": 2}),),
            }
        )
    with pytest.raises((ValidationError, HybridContractError), match="unique"):
        HybridEvidencePacket(
            **{
                **packet.model_dump(),
                "items": (item, item.model_copy(update={"rank": 2})),
            }
        )


def test_observation_timestamp_and_trace_score_are_strict() -> None:
    with pytest.raises((ValidationError, HybridContractError), match="UTC"):
        HybridRunObservation(
            processing_run_id="run_one",
            request_id=_request().request_id,
            cold_model_load_ns=0,
            passage_embedding_ns=0,
            query_embedding_ns=0,
            retrieval_ns=0,
            reranker_ns=0,
            peak_rss_bytes=0,
            cache_bytes=0,
            provider_calls=0,
            observed_at="2026-07-21T12:00:00+03:00",
        )
    with pytest.raises((ValidationError, HybridContractError), match="score"):
        HybridRankTraceItem(
            source_fragment_id="fragment_one",
            source_id="source_one",
            rank=1,
            score_repr="",
        )
