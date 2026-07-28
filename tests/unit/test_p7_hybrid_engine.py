"""Exact-closure orchestration tests for the P7 hybrid retrieval engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

import dithyramba.recall.hybrid_engine as hybrid_engine
from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    CompiledAccess,
    MembershipState,
    PermittedManifest,
    PermittedManifestItem,
    PermittedToken,
    PolicyEffect,
    PublicAccessResult,
    QueryExclusions,
    RequestScope,
)
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.persistence.models import SourceFragmentText
from dithyramba.provenance import SourceFamilyRole
from dithyramba.recall.fts import (
    FtsCandidate,
    FtsFragment,
    FtsRuntimeProfile,
    FtsSearchResult,
)
from dithyramba.recall.hybrid_engine import (
    HybridExecutionArtifacts,
    HybridExecutionError,
    HybridFtsSearch,
    execute_hybrid_retrieval,
)
from dithyramba.recall.hybrid_models import (
    CorpusLayer,
    CorpusLayerItem,
    CorpusLayerProfile,
    EmbeddingModelProfile,
    FusionProfile,
    HybridQueryRequest,
    HybridRetrievalBudget,
    ModelRuntimeProfile,
)
from dithyramba.recall.rerank import (
    RerankCandidate,
    RerankerProfile,
    RerankerProvider,
    RerankerScore,
)
from dithyramba.recall.semantic_provider import SemanticProvider
from dithyramba.recall.vector import (
    CorpusVectorGeneration,
    DenseHit,
    DenseVectorItem,
    PackedVector,
    VectorManifestItem,
    pack_normalized_vector,
)
from dithyramba.snapshots import CorpusSnapshot, SnapshotMember

HASH_A = "a" * 64
HASH_B = "b" * 64
MODEL_REVISION = "c" * 40
RERANKER_REVISION = "d" * 40
COLLECTION_ID = "collection_one"
POLICY_ID = "policy_one"
QUESTION = "Which evidence supports the claim?"


class RecordingSemanticProvider:
    def __init__(
        self,
        *,
        model_profile: EmbeddingModelProfile,
        runtime_profile: ModelRuntimeProfile,
        query_vector: object,
    ) -> None:
        self.model_profile = model_profile
        self.runtime_profile = runtime_profile
        self.query_vector = query_vector
        self.query_calls: list[str] = []

    def embed_query(self, question: str) -> PackedVector:
        self.query_calls.append(question)
        return cast(PackedVector, self.query_vector)

    def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
        raise AssertionError(f"query execution must not embed passages: {texts!r}")


class RecordingFts:
    def __init__(self, trace_ids: tuple[str, ...]) -> None:
        self.trace_ids = trace_ids
        self.calls: list[tuple[str, tuple[FtsFragment, ...], int]] = []

    def __call__(
        self,
        *,
        question: str,
        fragments: tuple[FtsFragment, ...],
        max_candidates: int,
    ) -> FtsSearchResult:
        self.calls.append((question, fragments, max_candidates))
        return _fts_result(
            fragments,
            max_candidates=max_candidates,
            trace_ids=self.trace_ids,
        )


class FaultyFts:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(
        self,
        *,
        question: str,
        fragments: tuple[FtsFragment, ...],
        max_candidates: int,
    ) -> FtsSearchResult:
        del question
        if self.mode == "wrong_type":
            return cast(FtsSearchResult, object())
        if self.mode == "wrong_corpus":
            return _fts_result(
                fragments,
                max_candidates=max_candidates,
                trace_ids=("fragment_alpha",),
                corpus_hash=HASH_A,
            )
        if self.mode == "outside_permitted":
            return _fts_result(
                fragments,
                max_candidates=max_candidates,
                trace_ids=("fragment_outside",),
            )
        if self.mode == "max_candidates":
            return _fts_result(
                fragments,
                max_candidates=max_candidates - 1,
                trace_ids=("fragment_alpha",),
            )
        if self.mode == "candidate_count":
            return _fts_result(
                fragments,
                max_candidates=max_candidates,
                trace_ids=("fragment_alpha",),
                candidate_count=len(fragments) + 1,
            )
        raise AssertionError(f"unknown faulty FTS mode: {self.mode}")


class RecordingReranker:
    def __init__(self, order: tuple[str, ...]) -> None:
        self.order = order
        self.calls: list[tuple[str, tuple[RerankCandidate, ...]]] = []

    def score(
        self,
        question: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankerScore, ...]:
        self.calls.append((question, candidates))
        assert set(self.order) == {item.source_fragment_id for item in candidates}
        return tuple(
            RerankerScore.from_float(fragment_id, float(len(self.order) - index))
            for index, fragment_id in enumerate(self.order)
        )


@dataclass(frozen=True, slots=True)
class EngineCase:
    request: HybridQueryRequest
    snapshot: CorpusSnapshot
    compiled_access: CompiledAccess
    fragments: tuple[SourceFragmentText, ...]
    layer_profile: CorpusLayerProfile
    fusion_profile: FusionProfile
    vector_generation: CorpusVectorGeneration
    vectors: tuple[DenseVectorItem, ...]
    semantic_provider: SemanticProvider
    fts_search: HybridFtsSearch
    reranker_profile: RerankerProfile | None = None
    reranker_provider: RerankerProvider | None = None


def _model_profile() -> EmbeddingModelProfile:
    return EmbeddingModelProfile(
        model_id="intfloat/multilingual-e5-small",
        revision=MODEL_REVISION,
        license="MIT",
        dimensions=2,
        max_tokens=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )


def _runtime_profile() -> ModelRuntimeProfile:
    return ModelRuntimeProfile(
        sentence_transformers_version="5.6.0",
        transformers_version="4.53.2",
        torch_version="2.7.1",
    )


def _reranker_profile() -> RerankerProfile:
    return RerankerProfile(
        model_id="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        revision=RERANKER_REVISION,
        license="Apache-2.0",
        max_tokens=512,
    )


def _retrieval_corpus_hash(fragments: tuple[FtsFragment, ...]) -> str:
    return canonical_sha256_hex(
        {
            "schema": "dithyramba.fts_corpus/1.0",
            "fragments": [
                fragment.identity_payload()
                for fragment in sorted(fragments, key=lambda item: item.source_fragment_id)
            ],
        }
    )


def _fts_result(
    fragments: tuple[FtsFragment, ...],
    *,
    max_candidates: int,
    trace_ids: tuple[str, ...],
    corpus_hash: str | None = None,
    candidate_count: int | None = None,
) -> FtsSearchResult:
    trace_length = len(trace_ids)
    return FtsSearchResult(
        retrieval_corpus_hash=(
            _retrieval_corpus_hash(fragments) if corpus_hash is None else corpus_hash
        ),
        query_tokens=("evidence",),
        max_candidates=max_candidates,
        candidate_count=trace_length if candidate_count is None else candidate_count,
        trace=tuple(
            FtsCandidate(
                source_fragment_id=fragment_id,
                rank=rank,
                score=f"{rank - trace_length - 1}.000000",
            )
            for rank, fragment_id in enumerate(trace_ids, start=1)
        ),
        profile=FtsRuntimeProfile(sqlite_version="3.45.0", compile_options_hash=HASH_A),
    )


def _make_case(
    *,
    empty: bool = False,
    require_family: bool = True,
    with_reranker: bool = False,
) -> EngineCase:
    library_id = "library_one"
    suffixes = () if empty else ("alpha", "beta", "gamma")
    text_by_suffix = {
        "alpha": "Alpha evidence supports the claim.",
        "beta": "Beta evidence supplies another source.",
        "gamma": "Gamma context qualifies the claim.",
    }
    layer_by_suffix = {
        "alpha": CorpusLayer.PRIMARY,
        "beta": CorpusLayer.DERIVED,
        "gamma": CorpusLayer.SYNTHESIS,
    }
    vector_by_suffix = {
        "alpha": pack_normalized_vector((0.8, 0.6)),
        "beta": pack_normalized_vector((1.0, 0.0)),
        "gamma": pack_normalized_vector((0.0, 1.0)),
    }
    snapshot = CorpusSnapshot(
        library_id=library_id,
        collection_ids=(COLLECTION_ID,),
        members=tuple(
            SnapshotMember(
                collection_id=COLLECTION_ID,
                source_id=f"source_{suffix}",
                source_version_id=f"source_version_{suffix}",
                content_sha256=sha256_hex(text_by_suffix[suffix].encode()),
                membership_state=MembershipState.ACTIVE,
                source_family_id=f"family_{suffix}",
                root_source_id=f"source_{suffix}",
                family_role=SourceFamilyRole.ROOT,
            )
            for suffix in suffixes
        ),
    )
    exclusions = QueryExclusions()
    scope = RequestScope(
        library_id=library_id,
        snapshot_hash=snapshot.manifest_hash,
        purpose="research",
        collection_ids=(COLLECTION_ID,),
        exclusions=exclusions,
    )
    policy = AccessPolicySnapshot(
        access_policy_id=POLICY_ID,
        library_id=library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(COLLECTION_ID, PolicyEffect.ALLOW),),
    )
    manifest = PermittedManifest(
        library_id=library_id,
        snapshot_hash=snapshot.manifest_hash,
        items=tuple(
            PermittedManifestItem(
                source_fragment_id=f"fragment_{suffix}",
                source_version_id=f"source_version_{suffix}",
                source_id=f"source_{suffix}",
                source_family_id=f"family_{suffix}" if require_family else None,
                collection_ids=(COLLECTION_ID,),
            )
            for suffix in suffixes
        ),
    )
    token = PermittedToken(
        library_id=library_id,
        snapshot_hash=snapshot.manifest_hash,
        policy_hash=policy.policy_hash,
        exclusion_hash=scope.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
    )
    compiled_access = CompiledAccess(
        manifest=manifest,
        token=token,
        public_result=PublicAccessResult(policy_omission_present=False),
    )
    fragments = tuple(
        SourceFragmentText(
            source_fragment_id=f"fragment_{suffix}",
            source_version_id=f"source_version_{suffix}",
            source_id=f"source_{suffix}",
            ordinal=0,
            fragment_kind="paragraph",
            text=text_by_suffix[suffix],
            text_sha256=sha256_hex(text_by_suffix[suffix].encode()),
            source_address_json='{"paragraph":1}',
        )
        for suffix in suffixes
    )
    model_profile = _model_profile()
    runtime_profile = _runtime_profile()
    fusion_profile = FusionProfile()
    layer_profile = CorpusLayerProfile(
        library_id=library_id,
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        exclusion_hash=scope.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
        items=tuple(
            CorpusLayerItem(
                source_id=f"source_{suffix}",
                layer=layer_by_suffix[suffix],
            )
            for suffix in suffixes
        ),
    )
    vectors = tuple(
        DenseVectorItem(
            source_fragment_id=f"fragment_{suffix}",
            source_id=f"source_{suffix}",
            vector=vector_by_suffix[suffix],
        )
        for suffix in suffixes
    )
    vector_generation = CorpusVectorGeneration(
        library_id=library_id,
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        snapshot_hash=snapshot.manifest_hash,
        access_policy_id=POLICY_ID,
        policy_hash=policy.policy_hash,
        exclusion_hash=scope.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
        model_profile_hash=model_profile.profile_hash,
        runtime_profile_hash=runtime_profile.profile_hash,
        dimensions=2,
        items=tuple(
            VectorManifestItem(
                source_fragment_id=f"fragment_{suffix}",
                text_sha256=sha256_hex(text_by_suffix[suffix].encode()),
                vector_sha256=vector_by_suffix[suffix].vector_hash,
            )
            for suffix in suffixes
        ),
    )
    reranker_profile = _reranker_profile() if with_reranker else None
    request = HybridQueryRequest(
        question=QUESTION,
        library_id=library_id,
        collection_ids=(COLLECTION_ID,),
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        access_policy_id=POLICY_ID,
        purpose="research",
        model_profile_hash=model_profile.profile_hash,
        runtime_profile_hash=runtime_profile.profile_hash,
        layer_profile_hash=layer_profile.profile_hash,
        fusion_profile_hash=fusion_profile.profile_hash,
        reranker_profile_hash=(None if reranker_profile is None else reranker_profile.profile_hash),
        exclusions=exclusions,
        retrieval=HybridRetrievalBudget(
            max_fts_candidates=3,
            max_dense_candidates=3,
            max_fused_candidates=3,
            max_source_fragments=3,
        ),
    )
    provider = RecordingSemanticProvider(
        model_profile=model_profile,
        runtime_profile=runtime_profile,
        query_vector=pack_normalized_vector((1.0, 0.0)),
    )
    fts_search = RecordingFts(() if empty else ("fragment_alpha", "fragment_beta"))
    return EngineCase(
        request=request,
        snapshot=snapshot,
        compiled_access=compiled_access,
        fragments=fragments,
        layer_profile=layer_profile,
        fusion_profile=fusion_profile,
        vector_generation=vector_generation,
        vectors=vectors,
        semantic_provider=provider,
        fts_search=fts_search,
        reranker_profile=reranker_profile,
    )


def _run(case: EngineCase) -> HybridExecutionArtifacts:
    return execute_hybrid_retrieval(
        request=case.request,
        snapshot=case.snapshot,
        compiled_access=case.compiled_access,
        fragments=case.fragments,
        layer_profile=case.layer_profile,
        fusion_profile=case.fusion_profile,
        vector_generation=case.vector_generation,
        vectors=case.vectors,
        semantic_provider=case.semantic_provider,
        reranker_profile=case.reranker_profile,
        reranker_provider=case.reranker_provider,
        fts_search=case.fts_search,
    )


def test_execute_fuses_lexical_and_dense_results_and_closes_all_artifacts() -> None:
    case = _make_case()

    artifacts = _run(case)

    assert tuple(item.source_fragment_id for item in artifacts.retrieval_receipt.fts) == (
        "fragment_alpha",
        "fragment_beta",
    )
    assert tuple(item.source_fragment_id for item in artifacts.retrieval_receipt.dense) == (
        "fragment_beta",
        "fragment_alpha",
        "fragment_gamma",
    )
    assert tuple(item.source_fragment_id for item in artifacts.retrieval_receipt.fused) == (
        "fragment_alpha",
        "fragment_beta",
        "fragment_gamma",
    )
    assert tuple(item.source_fragment_id for item in artifacts.packet.items) == (
        "fragment_alpha",
        "fragment_beta",
        "fragment_gamma",
    )
    assert tuple(item.source_family_id for item in artifacts.packet.items) == (
        "family_alpha",
        "family_beta",
        "family_gamma",
    )
    assert artifacts.packet.result_status == "evidence_found"
    assert artifacts.packet.request_id == case.request.request_id
    assert artifacts.packet.request_hash == case.request.request_hash
    assert artifacts.packet.read_receipt_hash == artifacts.read_receipt.receipt_hash
    assert artifacts.packet.access_receipt_hash == artifacts.access_receipt.receipt_hash
    assert artifacts.packet.coverage_report_hash == artifacts.coverage_report.report_hash
    assert artifacts.packet.retrieval_receipt_hash == artifacts.retrieval_receipt.receipt_hash
    assert artifacts.retrieval_receipt.vector_generation_hash == (
        case.vector_generation.generation_hash
    )
    assert artifacts.access_receipt.snapshot_hash == case.snapshot.manifest_hash
    assert artifacts.access_receipt.policy_hash == case.compiled_access.token.policy_hash
    assert artifacts.access_receipt.exclusion_hash == case.compiled_access.token.exclusion_hash
    assert artifacts.access_receipt.permitted_set_hash == (
        case.compiled_access.manifest.permitted_set_hash
    )
    assert artifacts.coverage_report.processed_count == len(case.fragments)
    assert tuple(item.text_sha256 for item in artifacts.read_receipt.items) == tuple(
        item.text_sha256 for item in case.fragments
    )

    provider = cast(RecordingSemanticProvider, case.semantic_provider)
    fts_search = cast(RecordingFts, case.fts_search)
    assert provider.query_calls == [QUESTION]
    assert len(fts_search.calls) == 1
    fts_question, fts_fragments, fts_limit = fts_search.calls[0]
    assert fts_question == QUESTION
    assert fts_limit == case.request.retrieval.max_fts_candidates
    assert tuple(item.source_fragment_id for item in fts_fragments) == tuple(
        item.source_fragment_id for item in case.fragments
    )
    assert tuple(item.text for item in fts_fragments) == tuple(item.text for item in case.fragments)


def test_execute_is_byte_deterministic_for_the_same_exact_inputs() -> None:
    case = _make_case()

    first = _run(case)
    second = _run(case)

    assert first == second
    assert first.read_receipt.canonical_bytes == second.read_receipt.canonical_bytes
    assert first.access_receipt.canonical_bytes == second.access_receipt.canonical_bytes
    assert first.coverage_report.canonical_bytes == second.coverage_report.canonical_bytes
    assert first.retrieval_receipt.receipt_hash == second.retrieval_receipt.receipt_hash
    assert first.packet.packet_hash == second.packet.packet_hash


def test_empty_authorized_set_returns_closed_no_evidence_packet() -> None:
    case = _make_case(empty=True)

    artifacts = _run(case)

    assert artifacts.packet.result_status == "no_evidence"
    assert artifacts.packet.items == ()
    assert artifacts.retrieval_receipt.fts == ()
    assert artifacts.retrieval_receipt.dense == ()
    assert artifacts.retrieval_receipt.fused == ()
    assert artifacts.read_receipt.items == ()
    assert artifacts.coverage_report.processed_count == 0
    assert cast(RecordingSemanticProvider, case.semantic_provider).query_calls == [QUESTION]


def test_reranker_receives_only_fused_authorized_text_and_controls_selection_order() -> None:
    case = _make_case(with_reranker=True)
    reranker = RecordingReranker(("fragment_gamma", "fragment_beta", "fragment_alpha"))
    case = replace(case, reranker_provider=reranker)

    artifacts = _run(case)

    assert tuple(item.source_fragment_id for item in artifacts.retrieval_receipt.fused) == (
        "fragment_alpha",
        "fragment_beta",
        "fragment_gamma",
    )
    assert tuple(item.source_fragment_id for item in artifacts.retrieval_receipt.reranked) == (
        "fragment_gamma",
        "fragment_beta",
        "fragment_alpha",
    )
    assert tuple(item.source_fragment_id for item in artifacts.packet.items) == (
        "fragment_gamma",
        "fragment_beta",
        "fragment_alpha",
    )
    assert artifacts.retrieval_receipt.reranker_profile_hash == (
        cast(RerankerProfile, case.reranker_profile).profile_hash
    )
    assert len(reranker.calls) == 1
    question, candidates = reranker.calls[0]
    assert question == QUESTION
    assert tuple(item.source_fragment_id for item in candidates) == (
        "fragment_alpha",
        "fragment_beta",
        "fragment_gamma",
    )
    assert {item.text for item in candidates} == {item.text for item in case.fragments}


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"library_id": "library_other"}, "request, snapshot"),
        ({"corpus_snapshot_id": "snapshot_other"}, "request, snapshot"),
        ({"collection_ids": ("collection_other",)}, "request, snapshot"),
        ({"purpose": "other"}, "exclusions differ"),
        (
            {"exclusions": QueryExclusions(source_ids=("source_alpha",))},
            "exclusions differ",
        ),
        ({"access_policy_id": "policy_other"}, "vector generation differs"),
    ],
)
def test_request_scope_must_close_over_snapshot_access_and_generation(
    updates: dict[str, object],
    match: str,
) -> None:
    case = _make_case()
    request = case.request.model_copy(update=updates)

    with pytest.raises(HybridExecutionError, match=match):
        _run(replace(case, request=request))


def test_snapshot_and_compiled_access_must_match_the_request_exactly() -> None:
    case = _make_case()
    wrong_snapshot = case.snapshot.model_copy(update={"library_id": "library_other"})
    with pytest.raises(HybridExecutionError, match="request, snapshot"):
        _run(replace(case, snapshot=wrong_snapshot))

    wrong_manifest = PermittedManifest(
        library_id="library_other",
        snapshot_hash=case.snapshot.manifest_hash,
        items=case.compiled_access.manifest.items,
    )
    wrong_library_access = CompiledAccess(
        manifest=wrong_manifest,
        token=PermittedToken(
            library_id="library_other",
            snapshot_hash=case.snapshot.manifest_hash,
            policy_hash=case.compiled_access.token.policy_hash,
            exclusion_hash=case.compiled_access.token.exclusion_hash,
            permitted_set_hash=wrong_manifest.permitted_set_hash,
        ),
        public_result=case.compiled_access.public_result,
    )
    with pytest.raises(HybridExecutionError, match="request, snapshot"):
        _run(replace(case, compiled_access=wrong_library_access))

    wrong_exclusion_access = replace(
        case.compiled_access,
        token=replace(case.compiled_access.token, exclusion_hash=HASH_B),
    )
    with pytest.raises(HybridExecutionError, match="exclusions differ"):
        _run(replace(case, compiled_access=wrong_exclusion_access))


def test_request_snapshot_and_access_require_exact_contract_types() -> None:
    class DerivedRequest(HybridQueryRequest):
        pass

    class DerivedSnapshot(CorpusSnapshot):
        pass

    class DerivedAccess(CompiledAccess):
        pass

    case = _make_case()
    derived_request = DerivedRequest.model_validate(dict(case.request))
    derived_snapshot = DerivedSnapshot.model_validate(dict(case.snapshot))
    derived_access = DerivedAccess(
        manifest=case.compiled_access.manifest,
        token=case.compiled_access.token,
        public_result=case.compiled_access.public_result,
    )

    with pytest.raises(HybridExecutionError, match="exact HybridQueryRequest"):
        _run(replace(case, request=cast(HybridQueryRequest, derived_request)))
    with pytest.raises(HybridExecutionError, match="exact CorpusSnapshot"):
        _run(replace(case, snapshot=cast(CorpusSnapshot, derived_snapshot)))
    with pytest.raises(HybridExecutionError, match=r"compiled_access.*exact"):
        _run(replace(case, compiled_access=cast(CompiledAccess, derived_access)))


def test_profiles_fragments_generation_and_vectors_require_exact_contract_types() -> None:
    class DerivedLayerProfile(CorpusLayerProfile):
        pass

    class DerivedFusionProfile(FusionProfile):
        pass

    class DerivedVectorGeneration(CorpusVectorGeneration):
        pass

    class DerivedFragment(SourceFragmentText):
        pass

    class DerivedDenseVector(DenseVectorItem):
        pass

    case = _make_case()
    derived_layer = DerivedLayerProfile.model_validate(dict(case.layer_profile))
    derived_fusion = DerivedFusionProfile.model_validate(dict(case.fusion_profile))
    derived_generation = DerivedVectorGeneration.model_validate(dict(case.vector_generation))
    first_fragment = case.fragments[0]
    derived_fragment = DerivedFragment(
        source_fragment_id=first_fragment.source_fragment_id,
        source_version_id=first_fragment.source_version_id,
        source_id=first_fragment.source_id,
        ordinal=first_fragment.ordinal,
        fragment_kind=first_fragment.fragment_kind,
        text=first_fragment.text,
        text_sha256=first_fragment.text_sha256,
        source_address_json=first_fragment.source_address_json,
    )
    first_vector = case.vectors[0]
    derived_vector = DerivedDenseVector(
        source_fragment_id=first_vector.source_fragment_id,
        source_id=first_vector.source_id,
        vector=first_vector.vector,
    )

    with pytest.raises(HybridExecutionError, match=r"layer_profile.*exact"):
        _run(replace(case, layer_profile=derived_layer))
    with pytest.raises(HybridExecutionError, match="fusion profile differs"):
        _run(replace(case, fusion_profile=derived_fusion))
    with pytest.raises(HybridExecutionError, match="authorized SourceFragmentText"):
        _run(
            replace(
                case,
                fragments=(derived_fragment, *case.fragments[1:]),
            )
        )
    with pytest.raises(HybridExecutionError, match=r"generation.*exact"):
        _run(
            replace(
                case,
                vector_generation=derived_generation,
            )
        )
    with pytest.raises(HybridExecutionError, match="exact DenseVectorItem"):
        _run(
            replace(
                case,
                vectors=(derived_vector, *case.vectors[1:]),
            )
        )


def test_layer_fusion_and_provider_profiles_are_exactly_request_bound() -> None:
    case = _make_case()
    wrong_layer = case.layer_profile.model_copy(update={"library_id": "library_other"})
    with pytest.raises(HybridExecutionError, match="layer profile differs"):
        _run(replace(case, layer_profile=wrong_layer))

    incomplete_layer = CorpusLayerProfile(
        library_id=case.request.library_id,
        corpus_snapshot_id=case.request.corpus_snapshot_id,
        exclusion_hash=case.compiled_access.token.exclusion_hash,
        permitted_set_hash=case.compiled_access.manifest.permitted_set_hash,
        items=case.layer_profile.items[:-1],
    )
    incomplete_request = case.request.model_copy(
        update={"layer_profile_hash": incomplete_layer.profile_hash}
    )
    with pytest.raises(ValueError, match="exact permitted Sources"):
        _run(
            replace(
                case,
                request=incomplete_request,
                layer_profile=incomplete_layer,
            )
        )

    wrong_fusion_request = case.request.model_copy(update={"fusion_profile_hash": HASH_B})
    with pytest.raises(HybridExecutionError, match="fusion profile differs"):
        _run(replace(case, request=wrong_fusion_request))

    provider = cast(RecordingSemanticProvider, case.semantic_provider)
    wrong_model = _model_profile().model_copy(update={"revision": "e" * 40})
    wrong_model_provider = RecordingSemanticProvider(
        model_profile=wrong_model,
        runtime_profile=provider.runtime_profile,
        query_vector=provider.query_vector,
    )
    with pytest.raises(HybridExecutionError, match="model differs"):
        _run(replace(case, semantic_provider=wrong_model_provider))

    wrong_runtime = _runtime_profile().model_copy(update={"sentence_transformers_version": "5.6.1"})
    wrong_runtime_provider = RecordingSemanticProvider(
        model_profile=provider.model_profile,
        runtime_profile=wrong_runtime,
        query_vector=provider.query_vector,
    )
    with pytest.raises(HybridExecutionError, match="runtime differs"):
        _run(replace(case, semantic_provider=wrong_runtime_provider))


def test_provider_must_expose_exact_profile_contracts() -> None:
    class OpaqueProvider:
        def embed_query(self, question: str) -> PackedVector:
            raise AssertionError(question)

        def embed_passages(self, texts: tuple[str, ...]) -> tuple[PackedVector, ...]:
            raise AssertionError(texts)

    case = _make_case()
    with pytest.raises(HybridExecutionError, match="does not expose exact profiles"):
        _run(replace(case, semantic_provider=cast(SemanticProvider, OpaqueProvider())))

    provider = cast(RecordingSemanticProvider, case.semantic_provider)
    wrong_type_provider = RecordingSemanticProvider(
        model_profile=provider.model_profile,
        runtime_profile=provider.runtime_profile,
        query_vector=provider.query_vector,
    )
    wrong_type_provider.model_profile = cast(EmbeddingModelProfile, object())
    with pytest.raises(HybridExecutionError, match="profiles must use the exact contracts"):
        _run(replace(case, semantic_provider=wrong_type_provider))


def test_reranker_authorization_profile_and_provider_must_close() -> None:
    case = _make_case()
    unexpected_profile = _reranker_profile()
    unexpected_provider = RecordingReranker(("fragment_alpha", "fragment_beta", "fragment_gamma"))
    with pytest.raises(HybridExecutionError, match="did not authorize"):
        _run(
            replace(
                case,
                reranker_profile=unexpected_profile,
                reranker_provider=unexpected_provider,
            )
        )

    authorized = _make_case(with_reranker=True)
    with pytest.raises(HybridExecutionError, match="provider is unavailable"):
        _run(authorized)

    authorized_profile = cast(RerankerProfile, authorized.reranker_profile)
    wrong_profile = authorized_profile.model_copy(update={"revision": "e" * 40})
    with pytest.raises(HybridExecutionError, match="profile differs"):
        _run(
            replace(
                authorized,
                reranker_profile=wrong_profile,
                reranker_provider=unexpected_provider,
            )
        )


def test_fragment_closure_rejects_missing_untyped_and_provenance_drift() -> None:
    case = _make_case()
    with pytest.raises(HybridExecutionError, match="exact permitted manifest"):
        _run(replace(case, fragments=case.fragments[:-1]))
    with pytest.raises(HybridExecutionError, match="authorized SourceFragmentText"):
        _run(replace(case, fragments=cast(Any, list(case.fragments))))

    wrong_version = replace(
        case.fragments[0],
        source_version_id="source_version_other",
    )
    with pytest.raises(HybridExecutionError, match="text or provenance"):
        _run(replace(case, fragments=(wrong_version, *case.fragments[1:])))

    wrong_source = replace(case.fragments[0], source_id="source_other")
    with pytest.raises(HybridExecutionError, match="text or provenance"):
        _run(replace(case, fragments=(wrong_source, *case.fragments[1:])))


def test_text_hash_tamper_and_vector_generation_text_drift_fail_closed() -> None:
    case = _make_case()
    tampered_fragment = replace(
        case.fragments[0],
        text=case.fragments[0].text + " tampered",
    )
    with pytest.raises(HybridExecutionError, match="text or provenance"):
        _run(replace(case, fragments=(tampered_fragment, *case.fragments[1:])))

    tampered_items = (
        case.vector_generation.items[0].model_copy(update={"text_sha256": HASH_B}),
        *case.vector_generation.items[1:],
    )
    tampered_generation = case.vector_generation.model_copy(update={"items": tampered_items})
    with pytest.raises(HybridExecutionError, match="vector bytes or provenance"):
        _run(replace(case, vector_generation=tampered_generation))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("library_id", "library_other"),
        ("corpus_snapshot_id", "snapshot_other"),
        ("snapshot_hash", HASH_B),
        ("access_policy_id", "policy_other"),
        ("policy_hash", HASH_B),
        ("exclusion_hash", HASH_B),
        ("permitted_set_hash", HASH_B),
        ("model_profile_hash", HASH_B),
        ("runtime_profile_hash", HASH_B),
    ],
)
def test_vector_generation_must_close_every_request_and_access_field(
    field: str,
    value: str,
) -> None:
    case = _make_case()
    generation = case.vector_generation.model_copy(update={field: value})

    with pytest.raises(HybridExecutionError, match="generation differs"):
        _run(replace(case, vector_generation=generation))


def test_vector_manifest_and_bytes_require_exact_cardinality_source_dimension_and_hash() -> None:
    case = _make_case()
    with pytest.raises(HybridExecutionError, match="exact generation manifest"):
        _run(replace(case, vectors=case.vectors[:-1]))

    wrong_source = replace(case.vectors[0], source_id="source_other")
    with pytest.raises(HybridExecutionError, match="vector bytes or provenance"):
        _run(replace(case, vectors=(wrong_source, *case.vectors[1:])))

    three_dimensions = pack_normalized_vector((1.0, 0.0, 0.0))
    wrong_dimension_vector = replace(case.vectors[0], vector=three_dimensions)
    dimension_items = (
        case.vector_generation.items[0].model_copy(
            update={"vector_sha256": three_dimensions.vector_hash}
        ),
        *case.vector_generation.items[1:],
    )
    dimension_generation = case.vector_generation.model_copy(update={"items": dimension_items})
    with pytest.raises(HybridExecutionError, match="vector bytes or provenance"):
        _run(
            replace(
                case,
                vectors=(wrong_dimension_vector, *case.vectors[1:]),
                vector_generation=dimension_generation,
            )
        )

    wrong_hash_items = (
        case.vector_generation.items[0].model_copy(update={"vector_sha256": HASH_B}),
        *case.vector_generation.items[1:],
    )
    wrong_hash_generation = case.vector_generation.model_copy(update={"items": wrong_hash_items})
    with pytest.raises(HybridExecutionError, match="vector bytes or provenance"):
        _run(replace(case, vector_generation=wrong_hash_generation))


@pytest.mark.parametrize("query_vector", [object(), pack_normalized_vector((1.0, 0.0, 0.0))])
def test_query_vector_requires_exact_type_and_generation_dimensions(query_vector: object) -> None:
    case = _make_case()
    provider = cast(RecordingSemanticProvider, case.semantic_provider)
    mismatched_provider = RecordingSemanticProvider(
        model_profile=provider.model_profile,
        runtime_profile=provider.runtime_profile,
        query_vector=query_vector,
    )

    with pytest.raises(HybridExecutionError, match="query vector"):
        _run(replace(case, semantic_provider=mismatched_provider))


def test_dense_scan_result_cannot_escape_the_permitted_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case()

    def escaping_scan(
        query: PackedVector,
        candidates: object,
        *,
        limit: int,
    ) -> tuple[DenseHit, ...]:
        del query, candidates, limit
        return (
            DenseHit(
                source_fragment_id="fragment_outside",
                source_id="source_outside",
                rank=1,
                score_hex=(1.0).hex(),
            ),
        )

    monkeypatch.setattr(hybrid_engine, "exact_cosine_scan", escaping_scan)

    with pytest.raises(HybridExecutionError, match=r"dense retrieval.*outside"):
        _run(case)


@pytest.mark.parametrize(
    "mode",
    ["wrong_type", "wrong_corpus", "outside_permitted", "max_candidates", "candidate_count"],
)
def test_injected_fts_must_return_exact_bounded_permitted_result(mode: str) -> None:
    case = _make_case()

    with pytest.raises(HybridExecutionError):
        _run(replace(case, fts_search=FaultyFts(mode)))

    assert cast(RecordingSemanticProvider, case.semantic_provider).query_calls == []


def test_fts_dependency_must_be_callable() -> None:
    case = _make_case()
    with pytest.raises(HybridExecutionError, match="must be callable"):
        _run(replace(case, fts_search=cast(HybridFtsSearch, object())))


def test_source_family_lineage_is_required_before_any_retrieval_call() -> None:
    case = _make_case(require_family=False)

    with pytest.raises(HybridExecutionError, match="SourceFamily lineage"):
        _run(case)

    assert cast(RecordingFts, case.fts_search).calls == []
    assert cast(RecordingSemanticProvider, case.semantic_provider).query_calls == []


def test_artifact_payloads_are_text_and_source_address_free() -> None:
    case = _make_case(with_reranker=True)
    case = replace(
        case,
        reranker_provider=RecordingReranker(("fragment_gamma", "fragment_beta", "fragment_alpha")),
    )
    artifacts = _run(case)
    encoded = canonical_json_bytes(
        {
            "read": artifacts.read_receipt.semantic_payload(),
            "access": artifacts.access_receipt.semantic_payload(),
            "coverage": artifacts.coverage_report.semantic_payload(),
            "retrieval": artifacts.retrieval_receipt.semantic_payload(),
            "packet": artifacts.packet.semantic_payload(),
        }
    )

    for fragment in case.fragments:
        assert fragment.text.encode() not in encoded
        assert fragment.source_address_json.encode() not in encoded
    assert QUESTION.encode() not in encoded
    assert b"text_sha256" in encoded
    assert b"source_fragment_id" in encoded


@pytest.mark.parametrize(
    "packet_hash_field",
    [
        "read_receipt_hash",
        "access_receipt_hash",
        "coverage_report_hash",
        "retrieval_receipt_hash",
    ],
)
def test_execution_artifacts_reject_packet_receipt_hash_drift(packet_hash_field: str) -> None:
    artifacts = _run(_make_case())
    packet = artifacts.packet.model_copy(update={packet_hash_field: HASH_B})

    with pytest.raises(HybridExecutionError, match="closure is inconsistent"):
        HybridExecutionArtifacts(
            read_receipt=artifacts.read_receipt,
            access_receipt=artifacts.access_receipt,
            coverage_report=artifacts.coverage_report,
            retrieval_receipt=artifacts.retrieval_receipt,
            packet=packet,
        )


def test_policy_omission_presence_propagates_without_denied_identity_or_count() -> None:
    case = _make_case()
    compiled_access = replace(
        case.compiled_access,
        public_result=PublicAccessResult(policy_omission_present=True),
    )

    artifacts = _run(replace(case, compiled_access=compiled_access))

    assert artifacts.access_receipt.policy_omission_present is True
    assert artifacts.coverage_report.policy_omission_present is True
    assert artifacts.coverage_report.semantic_payload()["policy_omission"] == {
        "present": True,
        "count": None,
    }
