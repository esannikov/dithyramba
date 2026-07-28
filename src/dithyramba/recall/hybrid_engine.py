"""Pure, deterministic P7 hybrid retrieval over an already-authorized corpus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from dithyramba.access import CompiledAccess, RequestScope
from dithyramba.contracts import canonical_sha256_hex, sha256_hex
from dithyramba.persistence.models import SourceFragmentText
from dithyramba.snapshots import CorpusSnapshot

from .fts import FtsFragment, FtsSearchResult, search_ephemeral_fts
from .fusion import (
    FusedCandidate,
    FusionCandidate,
    reciprocal_rank_fusion,
    select_layer_aware,
)
from .hybrid_artifacts import HybridAccessReceipt, HybridCoverageReport, HybridReadReceipt
from .hybrid_artifacts import HybridReadReceiptItem as HybridReadItem
from .hybrid_models import (
    CorpusLayerProfile,
    EmbeddingModelProfile,
    FusionProfile,
    HybridContractError,
    HybridEvidencePacket,
    HybridPacketItem,
    HybridQueryRequest,
    HybridRankTraceItem,
    HybridRetrievalReceipt,
    ModelRuntimeProfile,
)
from .rerank import RerankCandidate, RerankerProfile, RerankerProvider, run_reranker
from .semantic_provider import SemanticProvider
from .vector import CorpusVectorGeneration, DenseVectorItem, PackedVector, exact_cosine_scan


class HybridExecutionError(HybridContractError):
    """Authorized hybrid inputs or provider outputs fail exact closure."""


class HybridFtsSearch(Protocol):
    def __call__(
        self,
        *,
        question: str,
        fragments: tuple[FtsFragment, ...],
        max_candidates: int,
    ) -> FtsSearchResult: ...


@dataclass(frozen=True, slots=True)
class HybridExecutionArtifacts:
    """Complete semantic output of one successful graph-off hybrid execution."""

    read_receipt: HybridReadReceipt
    access_receipt: HybridAccessReceipt
    coverage_report: HybridCoverageReport
    retrieval_receipt: HybridRetrievalReceipt
    packet: HybridEvidencePacket

    def __post_init__(self) -> None:
        if self.packet.read_receipt_hash != self.read_receipt.receipt_hash:
            raise HybridExecutionError("packet/read receipt closure is inconsistent")
        if self.packet.access_receipt_hash != self.access_receipt.receipt_hash:
            raise HybridExecutionError("packet/access receipt closure is inconsistent")
        if self.packet.coverage_report_hash != self.coverage_report.report_hash:
            raise HybridExecutionError("packet/coverage closure is inconsistent")
        if self.packet.retrieval_receipt_hash != self.retrieval_receipt.receipt_hash:
            raise HybridExecutionError("packet/retrieval closure is inconsistent")


def execute_hybrid_retrieval(
    *,
    request: HybridQueryRequest,
    snapshot: CorpusSnapshot,
    compiled_access: CompiledAccess,
    fragments: tuple[SourceFragmentText, ...],
    layer_profile: CorpusLayerProfile,
    fusion_profile: FusionProfile,
    vector_generation: CorpusVectorGeneration,
    vectors: tuple[DenseVectorItem, ...],
    semantic_provider: SemanticProvider,
    reranker_profile: RerankerProfile | None = None,
    reranker_provider: RerankerProvider | None = None,
    fts_search: HybridFtsSearch = search_ephemeral_fts,
) -> HybridExecutionArtifacts:
    """Fuse lexical and dense rankings without expanding the permitted set."""

    _require_exact_inputs(
        request=request,
        snapshot=snapshot,
        compiled_access=compiled_access,
        fragments=fragments,
        layer_profile=layer_profile,
        fusion_profile=fusion_profile,
        vector_generation=vector_generation,
        vectors=vectors,
        semantic_provider=semantic_provider,
        reranker_profile=reranker_profile,
        reranker_provider=reranker_provider,
        fts_search=fts_search,
    )
    fragment_by_id = {item.source_fragment_id: item for item in fragments}
    manifest_by_id = {item.source_fragment_id: item for item in compiled_access.manifest.items}
    layer_by_source = {item.source_id: item.layer for item in layer_profile.items}
    metadata = {
        fragment_id: FusionCandidate(
            source_fragment_id=fragment_id,
            source_id=fragment.source_id,
            source_family_id=_required_family_id(manifest_by_id[fragment_id].source_family_id),
            layer=layer_by_source[fragment.source_id],
        )
        for fragment_id, fragment in fragment_by_id.items()
    }

    fts_fragments = tuple(
        FtsFragment(
            source_fragment_id=item.source_fragment_id,
            text=item.text,
            text_sha256=item.text_sha256,
        )
        for item in fragments
    )
    fts_result = fts_search(
        question=request.question,
        fragments=fts_fragments,
        max_candidates=request.retrieval.max_fts_candidates,
    )
    if type(fts_result) is not FtsSearchResult:
        raise HybridExecutionError("FTS search must return the exact result contract")
    expected_corpus_hash = _retrieval_corpus_hash(fts_fragments)
    if fts_result.retrieval_corpus_hash != expected_corpus_hash:
        raise HybridExecutionError("FTS result belongs to another permitted corpus")
    if (
        fts_result.max_candidates != request.retrieval.max_fts_candidates
        or fts_result.candidate_count > len(fragments)
    ):
        raise HybridExecutionError("FTS result differs from the requested bounded search")
    if any(item.source_fragment_id not in metadata for item in fts_result.trace):
        raise HybridExecutionError("FTS returned a fragment outside permitted access")

    query_vector = semantic_provider.embed_query(request.question)
    if type(query_vector) is not PackedVector:
        raise HybridExecutionError("semantic provider returned an invalid query vector")
    if query_vector.dimensions != vector_generation.dimensions:
        raise HybridExecutionError("query vector dimensions differ from the generation")
    dense_hits = exact_cosine_scan(
        query_vector,
        vectors,
        limit=request.retrieval.max_dense_candidates,
    )
    if any(item.source_fragment_id not in metadata for item in dense_hits):
        raise HybridExecutionError("dense retrieval returned a fragment outside permitted access")

    fts_candidates = tuple(metadata[item.source_fragment_id] for item in fts_result.trace)
    dense_candidates = tuple(metadata[item.source_fragment_id] for item in dense_hits)
    fused = reciprocal_rank_fusion(
        fts_candidates,
        dense_candidates,
        profile=fusion_profile,
        limit=request.retrieval.max_fused_candidates,
    )
    reranked_trace: tuple[HybridRankTraceItem, ...] = ()
    selection_pool = fused
    if reranker_profile is not None and fused:
        reranked = run_reranker(
            cast(RerankerProvider, reranker_provider),
            question=request.question,
            candidates=tuple(
                RerankCandidate(
                    source_fragment_id=item.candidate.source_fragment_id,
                    source_id=item.candidate.source_id,
                    text=fragment_by_id[item.candidate.source_fragment_id].text,
                )
                for item in fused
            ),
        )
        fused_by_id = {item.candidate.source_fragment_id: item for item in fused}
        selection_pool = tuple(fused_by_id[item.candidate.source_fragment_id] for item in reranked)
        reranked_trace = tuple(
            HybridRankTraceItem(
                source_fragment_id=item.candidate.source_fragment_id,
                source_id=item.candidate.source_id,
                rank=item.rank,
                score_repr=item.score_hex,
            )
            for item in reranked
        )
    selected = select_layer_aware(
        selection_pool,
        limit=request.retrieval.max_source_fragments,
        profile=fusion_profile,
    )

    retrieval = HybridRetrievalReceipt(
        request_id=request.request_id,
        request_hash=request.request_hash,
        exclusion_hash=compiled_access.token.exclusion_hash,
        permitted_set_hash=compiled_access.manifest.permitted_set_hash,
        model_profile_hash=request.model_profile_hash,
        runtime_profile_hash=request.runtime_profile_hash,
        layer_profile_hash=request.layer_profile_hash,
        fusion_profile_hash=request.fusion_profile_hash,
        vector_generation_hash=vector_generation.generation_hash,
        fts_profile_version=fts_result.profile_version,
        fts_result_hash=fts_result.result_hash,
        query_vector_hash=query_vector.vector_hash,
        reranker_profile_hash=request.reranker_profile_hash,
        fts=tuple(
            HybridRankTraceItem(
                source_fragment_id=item.source_fragment_id,
                source_id=fragment_by_id[item.source_fragment_id].source_id,
                rank=item.rank,
                score_repr=item.score,
            )
            for item in fts_result.trace
        ),
        dense=tuple(
            HybridRankTraceItem(
                source_fragment_id=item.source_fragment_id,
                source_id=item.source_id,
                rank=item.rank,
                score_repr=item.score_hex,
            )
            for item in dense_hits
        ),
        fused=_fused_trace(fused),
        reranked=reranked_trace,
    )
    read = HybridReadReceipt(
        request_id=request.request_id,
        request_hash=request.request_hash,
        retrieval_corpus_hash=expected_corpus_hash,
        items=tuple(
            HybridReadItem(
                source_fragment_id=item.source_fragment_id,
                source_version_id=item.source_version_id,
                read_order=order,
                text_sha256=item.text_sha256,
            )
            for order, item in enumerate(fragments)
        ),
    )
    access = HybridAccessReceipt(
        library_id=request.library_id,
        request_id=request.request_id,
        request_hash=request.request_hash,
        access_policy_id=request.access_policy_id,
        policy_hash=compiled_access.token.policy_hash,
        corpus_snapshot_id=request.corpus_snapshot_id,
        snapshot_hash=snapshot.manifest_hash,
        exclusion_hash=compiled_access.token.exclusion_hash,
        permitted_set_hash=compiled_access.manifest.permitted_set_hash,
        retrieval_corpus_hash=expected_corpus_hash,
        policy_omission_present=compiled_access.public_result.policy_omission_present,
    )
    coverage = HybridCoverageReport(
        request_id=request.request_id,
        request_hash=request.request_hash,
        processed_count=len(fragments),
        policy_omission_present=compiled_access.public_result.policy_omission_present,
    )
    packet_items = tuple(
        HybridPacketItem(
            rank=rank,
            source_fragment_id=item.candidate.source_fragment_id,
            source_id=item.candidate.source_id,
            source_family_id=item.candidate.source_family_id,
            layer=item.candidate.layer,
        )
        for rank, item in enumerate(selected, start=1)
    )
    packet = HybridEvidencePacket(
        request_id=request.request_id,
        request_hash=request.request_hash,
        corpus_snapshot_id=request.corpus_snapshot_id,
        access_policy_id=request.access_policy_id,
        read_receipt_hash=read.receipt_hash,
        access_receipt_hash=access.receipt_hash,
        coverage_report_hash=coverage.report_hash,
        retrieval_receipt_hash=retrieval.receipt_hash,
        result_status="evidence_found" if packet_items else "no_evidence",
        items=packet_items,
    )
    return HybridExecutionArtifacts(
        read_receipt=read,
        access_receipt=access,
        coverage_report=coverage,
        retrieval_receipt=retrieval,
        packet=packet,
    )


def _require_exact_inputs(
    *,
    request: HybridQueryRequest,
    snapshot: CorpusSnapshot,
    compiled_access: CompiledAccess,
    fragments: tuple[SourceFragmentText, ...],
    layer_profile: CorpusLayerProfile,
    fusion_profile: FusionProfile,
    vector_generation: CorpusVectorGeneration,
    vectors: tuple[DenseVectorItem, ...],
    semantic_provider: SemanticProvider,
    reranker_profile: RerankerProfile | None,
    reranker_provider: RerankerProvider | None,
    fts_search: HybridFtsSearch,
) -> None:
    if type(request) is not HybridQueryRequest:
        raise HybridExecutionError("request must be an exact HybridQueryRequest")
    if type(snapshot) is not CorpusSnapshot:
        raise HybridExecutionError("snapshot must be an exact CorpusSnapshot")
    if type(compiled_access) is not CompiledAccess:
        raise HybridExecutionError("compiled_access must use the exact contract")
    if not callable(fts_search):
        raise HybridExecutionError("fts_search must be callable")
    if (
        snapshot.library_id != request.library_id
        or snapshot.corpus_snapshot_id != request.corpus_snapshot_id
        or not set(request.collection_ids).issubset(snapshot.collection_ids)
        or compiled_access.manifest.library_id != request.library_id
        or compiled_access.manifest.snapshot_hash != snapshot.manifest_hash
        or compiled_access.token.snapshot_hash != snapshot.manifest_hash
    ):
        raise HybridExecutionError("request, snapshot, and compiled access do not match")
    request_scope = RequestScope(
        library_id=request.library_id,
        snapshot_hash=snapshot.manifest_hash,
        purpose=request.purpose,
        collection_ids=request.collection_ids,
        exclusions=request.exclusions,
    )
    if compiled_access.token.exclusion_hash != request_scope.exclusion_hash:
        raise HybridExecutionError("compiled access exclusions differ from the request")
    if type(layer_profile) is not CorpusLayerProfile:
        raise HybridExecutionError("layer_profile must use the exact contract")
    if (
        layer_profile.library_id != request.library_id
        or layer_profile.corpus_snapshot_id != request.corpus_snapshot_id
        or layer_profile.exclusion_hash != compiled_access.token.exclusion_hash
        or layer_profile.permitted_set_hash != compiled_access.manifest.permitted_set_hash
        or layer_profile.profile_hash != request.layer_profile_hash
    ):
        raise HybridExecutionError("layer profile differs from the authorized request")
    permitted_sources = tuple(
        sorted(
            {item.source_id for item in compiled_access.manifest.items},
            key=lambda value: value.encode("ascii"),
        )
    )
    layer_profile.require_exact_sources(permitted_sources)
    if (
        type(fusion_profile) is not FusionProfile
        or fusion_profile.profile_hash != request.fusion_profile_hash
    ):
        raise HybridExecutionError("fusion profile differs from the request")
    _require_provider(request, semantic_provider)
    _require_reranker(request, reranker_profile, reranker_provider)
    _require_fragment_closure(compiled_access, fragments)
    _require_vector_closure(
        request=request,
        snapshot=snapshot,
        compiled_access=compiled_access,
        fragments=fragments,
        vector_generation=vector_generation,
        vectors=vectors,
    )


def _require_provider(request: HybridQueryRequest, provider: SemanticProvider) -> None:
    try:
        model = provider.model_profile
        runtime = provider.runtime_profile
    except Exception as error:
        raise HybridExecutionError("semantic provider does not expose exact profiles") from error
    if type(model) is not EmbeddingModelProfile or type(runtime) is not ModelRuntimeProfile:
        raise HybridExecutionError("semantic provider profiles must use the exact contracts")
    if model.profile_hash != request.model_profile_hash:
        raise HybridExecutionError("semantic provider model differs from the request")
    if runtime.profile_hash != request.runtime_profile_hash:
        raise HybridExecutionError("semantic provider runtime differs from the request")


def _require_reranker(
    request: HybridQueryRequest,
    profile: RerankerProfile | None,
    provider: RerankerProvider | None,
) -> None:
    if request.reranker_profile_hash is None:
        if profile is not None or provider is not None:
            raise HybridExecutionError("request did not authorize a reranker")
        return
    if (
        type(profile) is not RerankerProfile
        or profile.profile_hash != request.reranker_profile_hash
    ):
        raise HybridExecutionError("reranker profile differs from the request")
    if provider is None or not callable(getattr(provider, "score", None)):
        raise HybridExecutionError("authorized reranker provider is unavailable")


def _require_fragment_closure(
    compiled: CompiledAccess,
    fragments: tuple[SourceFragmentText, ...],
) -> None:
    if type(fragments) is not tuple or any(
        type(item) is not SourceFragmentText for item in fragments
    ):
        raise HybridExecutionError("fragments must be authorized SourceFragmentText values")
    manifest = {item.source_fragment_id: item for item in compiled.manifest.items}
    if tuple(item.source_fragment_id for item in fragments) != tuple(sorted(manifest)):
        raise HybridExecutionError("fragment texts must equal the exact permitted manifest")
    for item in fragments:
        expected = manifest[item.source_fragment_id]
        if (
            item.source_version_id != expected.source_version_id
            or item.source_id != expected.source_id
            or sha256_hex(item.text.encode("utf-8")) != item.text_sha256
        ):
            raise HybridExecutionError("fragment text or provenance differs from permitted access")


def _require_vector_closure(
    *,
    request: HybridQueryRequest,
    snapshot: CorpusSnapshot,
    compiled_access: CompiledAccess,
    fragments: tuple[SourceFragmentText, ...],
    vector_generation: CorpusVectorGeneration,
    vectors: tuple[DenseVectorItem, ...],
) -> None:
    if type(vector_generation) is not CorpusVectorGeneration:
        raise HybridExecutionError("vector generation must use the exact contract")
    if (
        vector_generation.library_id != request.library_id
        or vector_generation.corpus_snapshot_id != request.corpus_snapshot_id
        or vector_generation.snapshot_hash != snapshot.manifest_hash
        or vector_generation.access_policy_id != request.access_policy_id
        or vector_generation.policy_hash != compiled_access.token.policy_hash
        or vector_generation.exclusion_hash != compiled_access.token.exclusion_hash
        or vector_generation.permitted_set_hash != compiled_access.manifest.permitted_set_hash
        or vector_generation.model_profile_hash != request.model_profile_hash
        or vector_generation.runtime_profile_hash != request.runtime_profile_hash
    ):
        raise HybridExecutionError("vector generation differs from the authorized request")
    if type(vectors) is not tuple or any(type(item) is not DenseVectorItem for item in vectors):
        raise HybridExecutionError("vectors must be exact DenseVectorItem values")
    manifest = {item.source_fragment_id: item for item in vector_generation.items}
    if tuple(item.source_fragment_id for item in vectors) != tuple(sorted(manifest)):
        raise HybridExecutionError("vectors must equal the exact generation manifest")
    permitted = {item.source_fragment_id: item.source_id for item in compiled_access.manifest.items}
    text_hashes = {item.source_fragment_id: item.text_sha256 for item in fragments}
    for item in vectors:
        expected = manifest[item.source_fragment_id]
        if (
            permitted.get(item.source_fragment_id) != item.source_id
            or text_hashes.get(item.source_fragment_id) != expected.text_sha256
            or item.vector.dimensions != vector_generation.dimensions
            or item.vector.vector_hash != expected.vector_sha256
        ):
            raise HybridExecutionError("vector bytes or provenance differ from their generation")


def _fused_trace(candidates: tuple[FusedCandidate, ...]) -> tuple[HybridRankTraceItem, ...]:
    return tuple(
        HybridRankTraceItem(
            source_fragment_id=item.candidate.source_fragment_id,
            source_id=item.candidate.source_id,
            rank=rank,
            score_repr=item.score_repr,
        )
        for rank, item in enumerate(candidates, start=1)
    )


def _retrieval_corpus_hash(fragments: tuple[FtsFragment, ...]) -> str:
    return canonical_sha256_hex(
        {
            "schema": "dithyramba.fts_corpus/1.0",
            "fragments": [
                item.identity_payload()
                for item in sorted(fragments, key=lambda value: value.source_fragment_id)
            ],
        }
    )


def _required_family_id(value: str | None) -> str:
    if value is None:
        raise HybridExecutionError("hybrid retrieval requires explicit SourceFamily lineage")
    return value
