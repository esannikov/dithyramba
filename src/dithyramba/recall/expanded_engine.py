"""Pure quality-first structure expansion over one frozen hybrid recall run.

The executor is deliberately source-text-free at its public boundary. Complete
source and StructureUnit text is admitted only as an authorized read closure,
passed to the reranker without truncation, and then discarded from returned
artifacts. The exact query is retained because it binds the frozen base request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from dithyramba.access import (
    CompiledAccess,
    PermittedManifest,
    PermittedManifestItem,
    PermittedToken,
    PublicAccessResult,
    QueryExclusions,
    RequestScope,
)
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex, sha256_hex
from dithyramba.persistence.models import SourceFragmentText
from dithyramba.structure import (
    StructureGeneration,
    StructureReadReceipt,
    StructureReadReceiptItem,
    StructureUnit,
    StructureUnitMember,
    StructureUnitText,
)

from .expanded_models import (
    ExpandedCandidateKind,
    ExpandedCandidateTrace,
    ExpandedContractError,
    ExpandedEvidencePacket,
    ExpandedPacketItem,
    ExpandedQueryRequest,
    ExpansionOmission,
    ExpansionOmissionReason,
    ExpansionPath,
    ExpansionReceipt,
    ExpansionSeedRank,
)
from .fts import FtsFragment
from .fusion import FusionCandidate, reciprocal_rank_fusion, select_layer_aware_fragment_ids
from .hybrid_artifacts import (
    HybridAccessReceipt,
    HybridCoverageReport,
    HybridReadReceipt,
    HybridReadReceiptItem,
)
from .hybrid_engine import HybridExecutionArtifacts
from .hybrid_models import (
    CorpusLayer,
    CorpusLayerProfile,
    FusionProfile,
    HybridEvidencePacket,
    HybridPacketItem,
    HybridQueryRequest,
    HybridRankTraceItem,
    HybridRetrievalBudget,
    HybridRetrievalReceipt,
)
from .rerank import RerankerProfile


class ExpandedExecutionError(ExpandedContractError):
    """Expanded inputs, provider output, or artifact closure failed closed."""


class ExpandedReranker(Protocol):
    """Minimal no-truncation reranker boundary used by expanded candidates."""

    @property
    def reranker_profile(self) -> RerankerProfile: ...

    def pair_token_lengths(
        self,
        question: str,
        candidate_texts: tuple[str, ...],
    ) -> tuple[int, ...]: ...

    def score_text_pairs(
        self,
        question: str,
        candidate_texts: tuple[str, ...],
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class _ValidatedExpandedSnapshot:
    expansion_paths: tuple[ExpansionPath, ...]
    receipt: ExpansionReceipt
    packet: ExpandedEvidencePacket
    request: ExpandedQueryRequest
    base_request: HybridQueryRequest
    base_artifacts: HybridExecutionArtifacts
    permitted_manifest: PermittedManifest
    layer_profile: CorpusLayerProfile
    fusion_profile: FusionProfile


@dataclass(frozen=True, slots=True)
class ExpandedExecutionArtifacts:
    """Complete source-text-free result owning validated nested copies.

    Python can bypass frozen dataclasses with ``object.__setattr__``.  Consumers
    that receive an object across an untrusted in-process boundary can call
    :meth:`validate_closure` immediately before use; ordinary construction
    snapshots every nested model so later caller mutation cannot affect it.
    """

    expansion_paths: tuple[ExpansionPath, ...]
    receipt: ExpansionReceipt
    packet: ExpandedEvidencePacket
    request: ExpandedQueryRequest
    base_request: HybridQueryRequest
    base_artifacts: HybridExecutionArtifacts
    permitted_manifest: PermittedManifest
    layer_profile: CorpusLayerProfile
    fusion_profile: FusionProfile

    def __post_init__(self) -> None:
        if type(self.expansion_paths) is not tuple or any(
            type(item) is not ExpansionPath for item in self.expansion_paths
        ):
            raise ExpandedExecutionError("expansion_paths must use the exact contract")
        if type(self.receipt) is not ExpansionReceipt:
            raise ExpandedExecutionError("receipt must use the exact ExpansionReceipt")
        if type(self.packet) is not ExpandedEvidencePacket:
            raise ExpandedExecutionError("packet must use the exact ExpandedEvidencePacket")
        if type(self.request) is not ExpandedQueryRequest:
            raise ExpandedExecutionError("request must use the exact ExpandedQueryRequest")
        if type(self.base_request) is not HybridQueryRequest:
            raise ExpandedExecutionError("base_request must use the exact HybridQueryRequest")
        if type(self.base_artifacts) is not HybridExecutionArtifacts:
            raise ExpandedExecutionError("base_artifacts must use the exact contract")
        if type(self.permitted_manifest) is not PermittedManifest:
            raise ExpandedExecutionError("permitted_manifest must use the exact contract")
        if type(self.layer_profile) is not CorpusLayerProfile:
            raise ExpandedExecutionError("layer_profile must use the exact contract")
        if type(self.fusion_profile) is not FusionProfile:
            raise ExpandedExecutionError("fusion_profile must use the exact FusionProfile")
        snapshot = _validated_execution_snapshot(
            expansion_paths=self.expansion_paths,
            receipt=self.receipt,
            packet=self.packet,
            request=self.request,
            base_request=self.base_request,
            base_artifacts=self.base_artifacts,
            permitted_manifest=self.permitted_manifest,
            layer_profile=self.layer_profile,
            fusion_profile=self.fusion_profile,
        )
        _require_execution_artifact_closure(
            request=snapshot.request,
            base_request=snapshot.base_request,
            base_artifacts=snapshot.base_artifacts,
            permitted_manifest=snapshot.permitted_manifest,
            layer_profile=snapshot.layer_profile,
            fusion_profile=snapshot.fusion_profile,
            receipt=snapshot.receipt,
            packet=snapshot.packet,
        )
        object.__setattr__(self, "expansion_paths", snapshot.expansion_paths)
        object.__setattr__(self, "receipt", snapshot.receipt)
        object.__setattr__(self, "packet", snapshot.packet)
        object.__setattr__(self, "request", snapshot.request)
        object.__setattr__(self, "base_request", snapshot.base_request)
        object.__setattr__(self, "base_artifacts", snapshot.base_artifacts)
        object.__setattr__(self, "permitted_manifest", snapshot.permitted_manifest)
        object.__setattr__(self, "layer_profile", snapshot.layer_profile)
        object.__setattr__(self, "fusion_profile", snapshot.fusion_profile)

    def validate_closure(self) -> None:
        """Revalidate the complete aggregate after an untrusted hand-off."""

        snapshot = _validated_execution_snapshot(
            expansion_paths=self.expansion_paths,
            receipt=self.receipt,
            packet=self.packet,
            request=self.request,
            base_request=self.base_request,
            base_artifacts=self.base_artifacts,
            permitted_manifest=self.permitted_manifest,
            layer_profile=self.layer_profile,
            fusion_profile=self.fusion_profile,
        )
        _require_execution_artifact_closure(
            request=snapshot.request,
            base_request=snapshot.base_request,
            base_artifacts=snapshot.base_artifacts,
            permitted_manifest=snapshot.permitted_manifest,
            layer_profile=snapshot.layer_profile,
            fusion_profile=snapshot.fusion_profile,
            receipt=snapshot.receipt,
            packet=snapshot.packet,
        )


@dataclass(frozen=True, slots=True)
class _ProjectedCandidate:
    path: ExpansionPath
    text: str
    exact_base_score: Fraction

    @property
    def key(self) -> tuple[str, str, str]:
        return self.path.candidate_key

    @property
    def best_anchor_rank(self) -> int:
        return min(self.path.anchor_atomic_ranks)


def execute_expanded_retrieval(
    *,
    request: ExpandedQueryRequest,
    base_request: HybridQueryRequest,
    base_artifacts: HybridExecutionArtifacts,
    compiled_access: CompiledAccess,
    fragments: tuple[SourceFragmentText, ...],
    layer_profile: CorpusLayerProfile,
    fusion_profile: FusionProfile,
    reranker_profile: RerankerProfile,
    reranker_provider: ExpandedReranker,
    structure_generation: StructureGeneration | None = None,
    structure_reads: tuple[StructureUnitText, ...] = (),
) -> ExpandedExecutionArtifacts:
    """Expand exact fused seeds into atomic or complete natural units and rerank once.

    The eligible pre-score is the highest exact reciprocal-rank-fusion score of
    the candidate's atomic anchors, converted once with ``float(...).hex()``.
    Complete texts are never included in returned paths, receipts, or packets.
    """

    _require_exact_public_inputs(
        request=request,
        base_request=base_request,
        base_artifacts=base_artifacts,
        compiled_access=compiled_access,
        fragments=fragments,
        layer_profile=layer_profile,
        fusion_profile=fusion_profile,
        reranker_profile=reranker_profile,
        reranker_provider=reranker_provider,
    )
    fragment_by_id = {item.source_fragment_id: item for item in fragments}
    manifest_by_id = {item.source_fragment_id: item for item in compiled_access.manifest.items}
    layer_by_source = {item.source_id: item.layer for item in layer_profile.items}
    fused = base_artifacts.retrieval_receipt.fused
    seeds = tuple(
        ExpansionSeedRank(
            source_fragment_id=item.source_fragment_id,
            atomic_rank=item.rank,
        )
        for item in fused
    )
    seed_score = {item.source_fragment_id: _exact_fused_score(item.score_repr) for item in fused}
    seed_rank = {item.source_fragment_id: item.rank for item in fused}

    projected = _project_candidates(
        request=request,
        base_request=base_request,
        compiled_access=compiled_access,
        fragment_by_id=fragment_by_id,
        manifest_by_id=manifest_by_id,
        layer_by_source=layer_by_source,
        seed_score=seed_score,
        seed_rank=seed_rank,
        structure_generation=structure_generation,
        structure_reads=structure_reads,
    )
    projected = tuple(sorted(projected, key=lambda item: (item.best_anchor_rank, item.key)))

    texts = tuple(item.text for item in projected)
    token_counts = _pair_token_lengths(
        reranker_provider,
        reranker_profile,
        base_request.question,
        texts,
    )
    eligible_candidates: list[_ProjectedCandidate] = []
    eligible_counts: list[int] = []
    omissions: list[ExpansionOmission] = []
    for candidate, token_count in zip(projected, token_counts, strict=True):
        if token_count > reranker_profile.max_tokens:
            omissions.append(
                ExpansionOmission(
                    candidate_kind=candidate.path.candidate_kind,
                    candidate_id=candidate.path.candidate_id,
                    relation_node_type=candidate.path.relation_node_type,
                    reason=ExpansionOmissionReason.TOKEN_BUDGET_EXCEEDED,
                    observed_token_count=token_count,
                    expansion_path_hash=candidate.path.path_hash,
                )
            )
            continue
        eligible_candidates.append(candidate)
        eligible_counts.append(token_count)

    eligible = tuple(
        _trace(
            candidate,
            rank=rank,
            pair_token_count=token_count,
            score_hex=_fraction_float_hex(candidate.exact_base_score),
        )
        for rank, (candidate, token_count) in enumerate(
            zip(eligible_candidates, eligible_counts, strict=True),
            start=1,
        )
    )

    reranked: tuple[ExpandedCandidateTrace, ...]
    if eligible_candidates:
        fit_texts = tuple(item.text for item in eligible_candidates)
        scores = _score_text_pairs(
            reranker_provider,
            reranker_profile,
            base_request.question,
            fit_texts,
        )
        scored = tuple(zip(eligible_candidates, eligible_counts, scores, strict=True))
        ordered = tuple(sorted(scored, key=lambda item: (-item[2], item[0].key)))
        reranked = tuple(
            _trace(
                candidate,
                rank=rank,
                pair_token_count=token_count,
                score_hex=score.hex(),
            )
            for rank, (candidate, token_count, score) in enumerate(ordered, start=1)
        )
    else:
        reranked = ()

    all_paths = tuple(
        sorted((item.path for item in projected), key=lambda item: item.path_hash.encode("ascii"))
    )
    receipt = ExpansionReceipt(
        expanded_request_id=request.request_id,
        expanded_request_hash=request.request_hash,
        base_request_id=base_request.request_id,
        base_request_hash=base_request.request_hash,
        base_retrieval_receipt_hash=base_artifacts.retrieval_receipt.receipt_hash,
        reranker_profile_hash=reranker_profile.profile_hash,
        seed_ranks=seeds,
        eligible=eligible,
        reranked=reranked,
        omissions=tuple(omissions),
        expansion_path_hashes=tuple(item.path_hash for item in all_paths),
        expansion_paths=all_paths,
        structure_read_receipt_hashes=tuple(
            sorted(
                {
                    item.structure_read_receipt_hash
                    for item in all_paths
                    if item.structure_read_receipt_hash is not None
                },
                key=str.encode,
            )
        ),
        relation_path_receipt_hashes=(),
    )
    selected = _select_final(reranked, request.max_final, fusion_profile)
    packet_items = tuple(
        ExpandedPacketItem(**(item.model_dump(exclude={"rank"}) | {"rank": rank}))
        for rank, item in enumerate(selected, start=1)
    )
    packet = ExpandedEvidencePacket(
        expanded_request_id=request.request_id,
        expanded_request_hash=request.request_hash,
        base_packet_id=base_artifacts.packet.packet_id,
        base_packet_hash=base_artifacts.packet.packet_hash,
        expansion_receipt_hash=receipt.receipt_hash,
        result_status="evidence_found" if packet_items else "no_evidence",
        items=packet_items,
    )
    return ExpandedExecutionArtifacts(
        expansion_paths=all_paths,
        receipt=receipt,
        packet=packet,
        request=request,
        base_request=base_request,
        base_artifacts=base_artifacts,
        permitted_manifest=compiled_access.manifest,
        layer_profile=layer_profile,
        fusion_profile=fusion_profile,
    )


def _validated_execution_snapshot(
    *,
    expansion_paths: tuple[ExpansionPath, ...],
    receipt: ExpansionReceipt,
    packet: ExpandedEvidencePacket,
    request: ExpandedQueryRequest,
    base_request: HybridQueryRequest,
    base_artifacts: HybridExecutionArtifacts,
    permitted_manifest: PermittedManifest,
    layer_profile: CorpusLayerProfile,
    fusion_profile: FusionProfile,
) -> _ValidatedExpandedSnapshot:
    """Rebuild every nested public model and return an independent snapshot."""

    if type(expansion_paths) is not tuple or any(
        type(item) is not ExpansionPath for item in expansion_paths
    ):
        raise ExpandedExecutionError("expansion_paths must use the exact contract")
    if type(receipt) is not ExpansionReceipt:
        raise ExpandedExecutionError("receipt must use the exact ExpansionReceipt")
    if type(packet) is not ExpandedEvidencePacket:
        raise ExpandedExecutionError("packet must use the exact ExpandedEvidencePacket")
    if type(request) is not ExpandedQueryRequest:
        raise ExpandedExecutionError("request must use the exact ExpandedQueryRequest")
    if type(base_request) is not HybridQueryRequest:
        raise ExpandedExecutionError("base_request must use the exact HybridQueryRequest")
    if type(base_artifacts) is not HybridExecutionArtifacts:
        raise ExpandedExecutionError("base_artifacts must use the exact contract")
    if type(permitted_manifest) is not PermittedManifest or any(
        type(item) is not PermittedManifestItem for item in permitted_manifest.items
    ):
        raise ExpandedExecutionError("permitted_manifest must use the exact contract")
    if type(layer_profile) is not CorpusLayerProfile:
        raise ExpandedExecutionError("layer_profile must use the exact contract")
    if type(fusion_profile) is not FusionProfile:
        raise ExpandedExecutionError("fusion_profile must use the exact FusionProfile")
    base_values = (
        base_artifacts.read_receipt,
        base_artifacts.access_receipt,
        base_artifacts.coverage_report,
        base_artifacts.retrieval_receipt,
        base_artifacts.packet,
    )
    if (
        type(base_values[0]) is not HybridReadReceipt
        or type(base_values[1]) is not HybridAccessReceipt
        or type(base_values[2]) is not HybridCoverageReport
        or type(base_values[3]) is not HybridRetrievalReceipt
        or type(base_values[4]) is not HybridEvidencePacket
    ):
        raise ExpandedExecutionError("base artifacts contain an inexact inner contract")

    try:
        validated_paths = tuple(
            ExpansionPath.model_validate(item.model_dump(mode="python")) for item in expansion_paths
        )
        validated_receipt = ExpansionReceipt.model_validate(receipt.model_dump(mode="python"))
        validated_packet = ExpandedEvidencePacket.model_validate(packet.model_dump(mode="python"))
        validated_request = ExpandedQueryRequest.model_validate(request.model_dump(mode="python"))
        validated_base_request = HybridQueryRequest(
            question=base_request.question,
            library_id=base_request.library_id,
            collection_ids=tuple(base_request.collection_ids),
            corpus_snapshot_id=base_request.corpus_snapshot_id,
            access_policy_id=base_request.access_policy_id,
            purpose=base_request.purpose,
            model_profile_hash=base_request.model_profile_hash,
            runtime_profile_hash=base_request.runtime_profile_hash,
            layer_profile_hash=base_request.layer_profile_hash,
            fusion_profile_hash=base_request.fusion_profile_hash,
            reranker_profile_hash=base_request.reranker_profile_hash,
            exclusions=QueryExclusions(
                source_ids=tuple(base_request.exclusions.source_ids),
                source_family_ids=tuple(base_request.exclusions.source_family_ids),
                source_fragment_ids=tuple(base_request.exclusions.source_fragment_ids),
            ),
            retrieval=HybridRetrievalBudget(
                max_fts_candidates=base_request.retrieval.max_fts_candidates,
                max_dense_candidates=base_request.retrieval.max_dense_candidates,
                max_fused_candidates=base_request.retrieval.max_fused_candidates,
                max_source_fragments=base_request.retrieval.max_source_fragments,
            ),
            result_format=base_request.result_format,
        )
        validated_manifest = PermittedManifest(
            library_id=permitted_manifest.library_id,
            snapshot_hash=permitted_manifest.snapshot_hash,
            items=tuple(
                PermittedManifestItem(
                    source_fragment_id=item.source_fragment_id,
                    source_version_id=item.source_version_id,
                    source_id=item.source_id,
                    source_family_id=item.source_family_id,
                    collection_ids=tuple(item.collection_ids),
                )
                for item in permitted_manifest.items
            ),
        )
        validated_layer_profile = CorpusLayerProfile.model_validate(
            layer_profile.model_dump(mode="python")
        )
        validated_profile = FusionProfile.model_validate(fusion_profile.model_dump(mode="python"))
        validated_read = HybridReadReceipt.model_validate(base_values[0].model_dump(mode="python"))
        validated_access = HybridAccessReceipt.model_validate(
            base_values[1].model_dump(mode="python")
        )
        validated_coverage = HybridCoverageReport.model_validate(
            base_values[2].model_dump(mode="python")
        )
        validated_retrieval = HybridRetrievalReceipt.model_validate(
            base_values[3].model_dump(mode="python")
        )
        validated_base_packet = HybridEvidencePacket.model_validate(
            base_values[4].model_dump(mode="python")
        )
        validated_base = HybridExecutionArtifacts(
            validated_read,
            validated_access,
            validated_coverage,
            validated_retrieval,
            validated_base_packet,
        )
    except Exception as error:
        raise ExpandedExecutionError(
            "expanded artifacts contain an invalid frozen contract"
        ) from error

    if (
        validated_paths != expansion_paths
        or validated_receipt != receipt
        or validated_packet != packet
        or validated_request != request
        or validated_base_request != base_request
        or validated_manifest != permitted_manifest
        or validated_layer_profile != layer_profile
        or validated_profile != fusion_profile
        or validated_base != base_artifacts
    ):
        raise ExpandedExecutionError("expanded artifacts changed after contract validation")
    if validated_paths != validated_receipt.expansion_paths:
        raise ExpandedExecutionError("artifact paths differ from the receipt closure")
    return _ValidatedExpandedSnapshot(
        expansion_paths=validated_receipt.expansion_paths,
        receipt=validated_receipt,
        packet=validated_packet,
        request=validated_request,
        base_request=validated_base_request,
        base_artifacts=validated_base,
        permitted_manifest=validated_manifest,
        layer_profile=validated_layer_profile,
        fusion_profile=validated_profile,
    )


def _require_execution_artifact_closure(
    *,
    request: ExpandedQueryRequest,
    base_request: HybridQueryRequest,
    base_artifacts: HybridExecutionArtifacts,
    permitted_manifest: PermittedManifest,
    layer_profile: CorpusLayerProfile,
    fusion_profile: FusionProfile,
    receipt: ExpansionReceipt,
    packet: ExpandedEvidencePacket,
) -> None:
    """Reassert the complete source-text-free closure, including selection."""

    base_read = base_artifacts.read_receipt
    base_access = base_artifacts.access_receipt
    base_coverage = base_artifacts.coverage_report
    base_retrieval = base_artifacts.retrieval_receipt
    base_packet = base_artifacts.packet
    if (
        type(base_read) is not HybridReadReceipt
        or type(base_access) is not HybridAccessReceipt
        or type(base_coverage) is not HybridCoverageReport
        or type(base_retrieval) is not HybridRetrievalReceipt
        or type(base_packet) is not HybridEvidencePacket
    ):
        raise ExpandedExecutionError("base artifacts contain an inexact inner contract")
    if (
        base_packet.read_receipt_hash != base_read.receipt_hash
        or base_packet.access_receipt_hash != base_access.receipt_hash
        or base_packet.coverage_report_hash != base_coverage.report_hash
        or base_packet.retrieval_receipt_hash != base_retrieval.receipt_hash
    ):
        raise ExpandedExecutionError("base artifact wrapper closure is inconsistent")
    if (
        request.base_request_id != base_request.request_id
        or request.base_request_hash != base_request.request_hash
    ):
        raise ExpandedExecutionError("expanded request differs from its frozen base request")
    base_binding = (base_request.request_id, base_request.request_hash)
    if any(
        (item.request_id, item.request_hash) != base_binding
        for item in (base_read, base_access, base_coverage, base_retrieval, base_packet)
    ):
        raise ExpandedExecutionError(
            "every base artifact must bind the exact expanded base request"
        )
    derived_retrieval_corpus_hash = _read_receipt_corpus_hash(base_read.items)
    base_scope = RequestScope(
        library_id=base_request.library_id,
        snapshot_hash=base_access.snapshot_hash,
        purpose=base_request.purpose,
        collection_ids=base_request.collection_ids,
        exclusions=base_request.exclusions,
    )
    if (
        permitted_manifest.permitted_set_hash != canonical_sha256_hex(permitted_manifest.payload())
        or permitted_manifest.library_id != base_request.library_id
        or permitted_manifest.snapshot_hash != base_access.snapshot_hash
        or permitted_manifest.permitted_set_hash != base_access.permitted_set_hash
        or base_access.library_id != base_request.library_id
        or base_access.corpus_snapshot_id != base_request.corpus_snapshot_id
        or base_access.access_policy_id != base_request.access_policy_id
        or base_access.exclusion_hash != base_scope.exclusion_hash
        or layer_profile.library_id != base_request.library_id
        or layer_profile.corpus_snapshot_id != base_request.corpus_snapshot_id
        or layer_profile.exclusion_hash != base_access.exclusion_hash
        or layer_profile.permitted_set_hash != permitted_manifest.permitted_set_hash
        or layer_profile.profile_hash != base_request.layer_profile_hash
        or base_read.retrieval_corpus_hash != base_access.retrieval_corpus_hash
        or base_coverage.processed_count != len(base_read.items)
        or base_coverage.policy_omission_present != base_access.policy_omission_present
        or base_retrieval.exclusion_hash != base_access.exclusion_hash
        or base_retrieval.permitted_set_hash != base_access.permitted_set_hash
        or base_retrieval.fusion_profile_hash != fusion_profile.profile_hash
        or base_retrieval.model_profile_hash != base_request.model_profile_hash
        or base_retrieval.runtime_profile_hash != base_request.runtime_profile_hash
        or base_retrieval.layer_profile_hash != layer_profile.profile_hash
        or base_retrieval.fusion_profile_hash != base_request.fusion_profile_hash
        or base_request.reranker_profile_hash is not None
        or base_retrieval.reranker_profile_hash is not None
        or base_retrieval.reranked
        or base_read.retrieval_corpus_hash != derived_retrieval_corpus_hash
        or base_packet.corpus_snapshot_id != base_access.corpus_snapshot_id
        or base_packet.access_policy_id != base_access.access_policy_id
    ):
        raise ExpandedExecutionError("base artifacts disagree on their exact execution scope")
    expected_read_bindings = tuple(
        (item.source_fragment_id, item.source_version_id, order)
        for order, item in enumerate(permitted_manifest.items)
    )
    actual_read_bindings = tuple(
        (item.source_fragment_id, item.source_version_id, item.read_order)
        for item in base_read.items
    )
    if actual_read_bindings != expected_read_bindings:
        raise ExpandedExecutionError(
            "base read receipt must equal the exact permitted manifest order"
        )
    sources = tuple(sorted({item.source_id for item in permitted_manifest.items}, key=str.encode))
    try:
        layer_profile.require_exact_sources(sources)
    except Exception as error:
        raise ExpandedExecutionError(
            "layer profile is not the exact permitted Source closure"
        ) from error
    if (
        len(base_retrieval.fts) > base_request.retrieval.max_fts_candidates
        or len(base_retrieval.dense) > base_request.retrieval.max_dense_candidates
        or len(base_retrieval.fused) > base_request.retrieval.max_fused_candidates
    ):
        raise ExpandedExecutionError("base retrieval exceeds its frozen request budget")
    expected_base_items = _derive_exact_base_selection(
        retrieval=base_retrieval,
        packet_limit=base_request.retrieval.max_source_fragments,
        max_fused_candidates=base_request.retrieval.max_fused_candidates,
        permitted_manifest=permitted_manifest,
        layer_profile=layer_profile,
        fusion_profile=fusion_profile,
    )
    expected_base_status = "evidence_found" if expected_base_items else "no_evidence"
    if (
        base_packet.items != expected_base_items
        or base_packet.result_status != expected_base_status
    ):
        raise ExpandedExecutionError("base packet is not the exact fused selection")
    if (
        receipt.expanded_request_id != request.request_id
        or receipt.expanded_request_hash != request.request_hash
        or receipt.base_request_id != request.base_request_id
        or receipt.base_request_hash != request.base_request_hash
        or receipt.reranker_profile_hash != request.reranker_profile_hash
        or base_packet.request_id != request.base_request_id
        or base_packet.request_hash != request.base_request_hash
        or base_retrieval.request_id != request.base_request_id
        or base_retrieval.request_hash != request.base_request_hash
        or receipt.base_retrieval_receipt_hash != base_retrieval.receipt_hash
    ):
        raise ExpandedExecutionError("request/receipt/base artifact closure is inconsistent")
    expected_seeds = tuple(
        ExpansionSeedRank(
            source_fragment_id=item.source_fragment_id,
            atomic_rank=item.rank,
        )
        for item in base_retrieval.fused
    )
    if receipt.seed_ranks != expected_seeds:
        raise ExpandedExecutionError("expansion seed ranks must equal the exact base fused trace")
    _require_seed_projection_closure(
        receipt,
        permitted_manifest,
        layer_profile,
        base_read.items,
    )
    _require_expansion_order_closure(receipt)
    _require_eligible_score_closure(receipt, base_retrieval)
    if (
        packet.expanded_request_id != request.request_id
        or packet.expanded_request_hash != request.request_hash
        or packet.base_packet_id != base_packet.packet_id
        or packet.base_packet_hash != base_packet.packet_hash
        or packet.expansion_receipt_hash != receipt.receipt_hash
    ):
        raise ExpandedExecutionError("packet/request/base/receipt closure is inconsistent")

    selected = _select_final(receipt.reranked, request.max_final, fusion_profile)
    expected_items = tuple(
        ExpandedPacketItem(**(item.model_dump(exclude={"rank"}) | {"rank": rank}))
        for rank, item in enumerate(selected, start=1)
    )
    expected_status = "evidence_found" if expected_items else "no_evidence"
    if packet.items != expected_items or packet.result_status != expected_status:
        raise ExpandedExecutionError("packet items must equal the exact capped reranked selection")


def _require_seed_projection_closure(
    receipt: ExpansionReceipt,
    permitted_manifest: PermittedManifest,
    layer_profile: CorpusLayerProfile,
    read_items: tuple[HybridReadReceiptItem, ...],
) -> None:
    """Require every exact seed to resolve once and only once."""

    seed_rank_by_fragment = {
        item.source_fragment_id: item.atomic_rank for item in receipt.seed_ranks
    }
    manifest_by_fragment = {item.source_fragment_id: item for item in permitted_manifest.items}
    read_hash_by_fragment = {item.source_fragment_id: item.text_sha256 for item in read_items}
    layer_by_source = {item.source_id: item.layer for item in layer_profile.items}
    resolution_count = {fragment_id: 0 for fragment_id in seed_rank_by_fragment}
    for path in receipt.expansion_paths:
        try:
            grounding = tuple(
                manifest_by_fragment[fragment_id] for fragment_id in path.source_fragment_ids
            )
            expected_sources = tuple(sorted({item.source_id for item in grounding}, key=str.encode))
            expected_families = tuple(
                sorted(
                    {
                        item.source_family_id
                        for item in grounding
                        if item.source_family_id is not None
                    },
                    key=str.encode,
                )
            )
            expected_layers = tuple(
                sorted(
                    {layer_by_source[source_id] for source_id in expected_sources},
                    key=lambda item: item.value.encode("ascii"),
                )
            )
        except KeyError as error:
            raise ExpandedExecutionError(
                "ExpansionPath contains foreign permitted provenance"
            ) from error
        if (
            len(expected_families) != len({item.source_family_id for item in grounding})
            or path.source_ids != expected_sources
            or path.source_family_ids != expected_families
            or path.corpus_layers != expected_layers
        ):
            raise ExpandedExecutionError(
                "ExpansionPath provenance must equal its exact permitted grounding"
            )
        if (
            path.candidate_kind is ExpandedCandidateKind.SOURCE_FRAGMENT
            and read_hash_by_fragment.get(path.candidate_id) != path.text_sha256
        ):
            raise ExpandedExecutionError(
                "atomic ExpansionPath text hash must equal its base read item"
            )
        anchored_fragments = tuple(
            fragment_id
            for fragment_id in path.source_fragment_ids
            if fragment_id in seed_rank_by_fragment
        )
        expected_anchor_ranks = tuple(
            sorted(seed_rank_by_fragment[fragment_id] for fragment_id in anchored_fragments)
        )
        if not anchored_fragments or path.anchor_atomic_ranks != expected_anchor_ranks:
            raise ExpandedExecutionError(
                "ExpansionPath anchors must identify its exact seed fragments"
            )
        for fragment_id in anchored_fragments:
            resolution_count[fragment_id] += 1
    for omission in receipt.omissions:
        if omission.expansion_path_hash is not None:
            continue
        if (
            omission.candidate_kind is not ExpandedCandidateKind.SOURCE_FRAGMENT
            or omission.candidate_id not in seed_rank_by_fragment
        ):
            raise ExpandedExecutionError("pathless omission must identify one exact atomic seed")
        resolution_count[omission.candidate_id] += 1
    if any(count != 1 for count in resolution_count.values()):
        raise ExpandedExecutionError(
            "every atomic seed must resolve to exactly one path or explicit omission"
        )


def _require_eligible_score_closure(
    receipt: ExpansionReceipt,
    retrieval: HybridRetrievalReceipt,
) -> None:
    """Re-derive every deterministic eligible pre-score from fused anchors."""

    score_by_rank = {item.rank: _exact_fused_score(item.score_repr) for item in retrieval.fused}
    for candidate in receipt.eligible:
        try:
            expected_score = max(score_by_rank[rank] for rank in candidate.anchor_atomic_ranks)
        except (KeyError, ValueError) as error:
            raise ExpandedExecutionError(
                "eligible candidate references an unavailable fused anchor"
            ) from error
        if candidate.score_hex != _fraction_float_hex(expected_score):
            raise ExpandedExecutionError(
                "eligible pre-score must equal its strongest exact fused anchor"
            )


def _require_expansion_order_closure(receipt: ExpansionReceipt) -> None:
    """Re-derive deterministic eligible and reranked ordering."""

    omitted_keys = {item.candidate_key for item in receipt.omissions}
    expected_eligible_keys = tuple(
        item.candidate_key
        for item in sorted(
            (path for path in receipt.expansion_paths if path.candidate_key not in omitted_keys),
            key=lambda path: (min(path.anchor_atomic_ranks), path.candidate_key),
        )
    )
    if tuple(item.candidate_key for item in receipt.eligible) != expected_eligible_keys:
        raise ExpandedExecutionError(
            "eligible order must equal the exact deterministic path projection"
        )
    expected_reranked_keys = tuple(
        item.candidate_key
        for item in sorted(
            receipt.reranked,
            key=lambda item: (-float.fromhex(item.score_hex), item.candidate_key),
        )
    )
    if tuple(item.candidate_key for item in receipt.reranked) != expected_reranked_keys:
        raise ExpandedExecutionError(
            "reranked order must equal deterministic score and identity ordering"
        )


def _require_exact_public_inputs(
    *,
    request: ExpandedQueryRequest,
    base_request: HybridQueryRequest,
    base_artifacts: HybridExecutionArtifacts,
    compiled_access: CompiledAccess,
    fragments: tuple[SourceFragmentText, ...],
    layer_profile: CorpusLayerProfile,
    fusion_profile: FusionProfile,
    reranker_profile: RerankerProfile,
    reranker_provider: ExpandedReranker,
) -> None:
    if type(request) is not ExpandedQueryRequest:
        raise ExpandedExecutionError("request must be an exact ExpandedQueryRequest")
    if type(base_request) is not HybridQueryRequest:
        raise ExpandedExecutionError("base_request must be an exact HybridQueryRequest")
    if type(base_artifacts) is not HybridExecutionArtifacts:
        raise ExpandedExecutionError("base_artifacts must use the exact contract")
    if type(compiled_access) is not CompiledAccess:
        raise ExpandedExecutionError("compiled_access must use the exact contract")
    if request.graph_enabled:
        raise ExpandedExecutionError("graph expansion is unsupported by this executor")
    if base_request.reranker_profile_hash is not None:
        raise ExpandedExecutionError("base request must not rerank before final expansion")
    if (
        request.base_request_id != base_request.request_id
        or request.base_request_hash != base_request.request_hash
    ):
        raise ExpandedExecutionError("expanded request targets another base request")
    if type(reranker_profile) is not RerankerProfile:
        raise ExpandedExecutionError("reranker_profile must use the exact contract")
    if request.reranker_profile_hash != reranker_profile.profile_hash:
        raise ExpandedExecutionError("expanded request targets another reranker profile")
    _require_provider_profile(reranker_provider, reranker_profile)
    _require_access_closure(base_request, base_artifacts, compiled_access, fragments)
    _require_profile_closure(
        base_request,
        base_artifacts,
        compiled_access,
        layer_profile,
        fusion_profile,
    )


def _require_access_closure(
    request: HybridQueryRequest,
    artifacts: HybridExecutionArtifacts,
    compiled: CompiledAccess,
    fragments: tuple[SourceFragmentText, ...],
) -> None:
    if (
        type(compiled.manifest) is not PermittedManifest
        or type(compiled.token) is not PermittedToken
        or type(compiled.public_result) is not PublicAccessResult
        or any(type(item) is not PermittedManifestItem for item in compiled.manifest.items)
    ):
        raise ExpandedExecutionError("compiled access contains an inexact contract")
    if compiled.manifest.permitted_set_hash != canonical_sha256_hex(
        compiled.manifest.payload()
    ) or compiled.token.token_hash != canonical_sha256_hex(compiled.token.binding_payload()):
        raise ExpandedExecutionError("compiled access content hash changed")
    expected_snapshot_id = f"snapshot_{compiled.token.snapshot_hash[:32]}"
    if request.corpus_snapshot_id != expected_snapshot_id:
        raise ExpandedExecutionError(
            "corpus snapshot ID must be derived from the exact access snapshot hash"
        )
    scope = RequestScope(
        library_id=request.library_id,
        snapshot_hash=compiled.token.snapshot_hash,
        purpose=request.purpose,
        collection_ids=request.collection_ids,
        exclusions=request.exclusions,
    )
    if (
        compiled.manifest.library_id != request.library_id
        or compiled.token.library_id != request.library_id
        or compiled.manifest.snapshot_hash != compiled.token.snapshot_hash
        or compiled.token.exclusion_hash != scope.exclusion_hash
        or compiled.manifest.permitted_set_hash != compiled.token.permitted_set_hash
    ):
        raise ExpandedExecutionError("compiled access differs from the base request")

    if type(fragments) is not tuple or any(
        type(item) is not SourceFragmentText for item in fragments
    ):
        raise ExpandedExecutionError("fragments must be exact SourceFragmentText values")
    manifest_ids = tuple(item.source_fragment_id for item in compiled.manifest.items)
    if tuple(item.source_fragment_id for item in fragments) != manifest_ids:
        raise ExpandedExecutionError("fragment texts must equal the full permitted manifest")
    manifest_by_id = {item.source_fragment_id: item for item in compiled.manifest.items}
    for fragment in fragments:
        manifest = manifest_by_id[fragment.source_fragment_id]
        if (
            fragment.source_version_id != manifest.source_version_id
            or fragment.source_id != manifest.source_id
            or sha256_hex(fragment.text.encode("utf-8")) != fragment.text_sha256
        ):
            raise ExpandedExecutionError("fragment text or provenance changed after access")

    if (
        type(artifacts.read_receipt) is not HybridReadReceipt
        or type(artifacts.access_receipt) is not HybridAccessReceipt
        or type(artifacts.coverage_report) is not HybridCoverageReport
        or type(artifacts.retrieval_receipt) is not HybridRetrievalReceipt
        or type(artifacts.packet) is not HybridEvidencePacket
    ):
        raise ExpandedExecutionError("base artifacts contain an inexact contract")
    expected_read_items = tuple(
        HybridReadReceiptItem(
            source_fragment_id=item.source_fragment_id,
            source_version_id=item.source_version_id,
            read_order=order,
            text_sha256=item.text_sha256,
        )
        for order, item in enumerate(fragments)
    )
    retrieval_corpus_hash = _retrieval_corpus_hash(fragments)
    read = artifacts.read_receipt
    access = artifacts.access_receipt
    coverage = artifacts.coverage_report
    retrieval = artifacts.retrieval_receipt
    packet = artifacts.packet
    if (
        read.request_id != request.request_id
        or read.request_hash != request.request_hash
        or read.retrieval_corpus_hash != retrieval_corpus_hash
        or read.items != expected_read_items
    ):
        raise ExpandedExecutionError("base read receipt is not the authorized text closure")
    if (
        access.library_id != request.library_id
        or access.request_id != request.request_id
        or access.request_hash != request.request_hash
        or access.access_policy_id != request.access_policy_id
        or access.corpus_snapshot_id != request.corpus_snapshot_id
        or access.snapshot_hash != compiled.token.snapshot_hash
        or access.policy_hash != compiled.token.policy_hash
        or access.exclusion_hash != compiled.token.exclusion_hash
        or access.permitted_set_hash != compiled.token.permitted_set_hash
        or access.retrieval_corpus_hash != retrieval_corpus_hash
        or access.policy_omission_present != compiled.public_result.policy_omission_present
    ):
        raise ExpandedExecutionError("base access receipt differs from snapshot/policy access")
    if (
        coverage.request_id != request.request_id
        or coverage.request_hash != request.request_hash
        or coverage.processed_count != len(fragments)
        or coverage.skipped_count != 0
        or coverage.failed_count != 0
        or coverage.policy_omission_present != compiled.public_result.policy_omission_present
    ):
        raise ExpandedExecutionError("base coverage report is not complete")
    if (
        packet.request_id != request.request_id
        or packet.request_hash != request.request_hash
        or packet.corpus_snapshot_id != request.corpus_snapshot_id
        or packet.access_policy_id != request.access_policy_id
        or packet.read_receipt_hash != read.receipt_hash
        or packet.access_receipt_hash != access.receipt_hash
        or packet.coverage_report_hash != coverage.report_hash
        or packet.retrieval_receipt_hash != retrieval.receipt_hash
    ):
        raise ExpandedExecutionError("base packet does not close over its exact artifacts")


def _require_profile_closure(
    request: HybridQueryRequest,
    artifacts: HybridExecutionArtifacts,
    compiled: CompiledAccess,
    layer_profile: CorpusLayerProfile,
    fusion_profile: FusionProfile,
) -> None:
    if type(layer_profile) is not CorpusLayerProfile:
        raise ExpandedExecutionError("layer_profile must use the exact contract")
    if type(fusion_profile) is not FusionProfile:
        raise ExpandedExecutionError("fusion_profile must use the exact contract")
    if (
        layer_profile.library_id != request.library_id
        or layer_profile.corpus_snapshot_id != request.corpus_snapshot_id
        or layer_profile.exclusion_hash != compiled.token.exclusion_hash
        or layer_profile.permitted_set_hash != compiled.token.permitted_set_hash
        or layer_profile.profile_hash != request.layer_profile_hash
        or fusion_profile.profile_hash != request.fusion_profile_hash
    ):
        raise ExpandedExecutionError("layer/fusion profile differs from the base request")
    sources = tuple(sorted({item.source_id for item in compiled.manifest.items}, key=str.encode))
    try:
        layer_profile.require_exact_sources(sources)
    except Exception as error:
        raise ExpandedExecutionError("layer profile is not the exact source closure") from error

    retrieval = artifacts.retrieval_receipt
    if (
        retrieval.request_id != request.request_id
        or retrieval.request_hash != request.request_hash
        or retrieval.exclusion_hash != compiled.token.exclusion_hash
        or retrieval.permitted_set_hash != compiled.token.permitted_set_hash
        or retrieval.model_profile_hash != request.model_profile_hash
        or retrieval.runtime_profile_hash != request.runtime_profile_hash
        or retrieval.layer_profile_hash != request.layer_profile_hash
        or retrieval.fusion_profile_hash != request.fusion_profile_hash
        or retrieval.reranker_profile_hash is not None
        or retrieval.reranked
    ):
        raise ExpandedExecutionError("base retrieval receipt differs from the request")
    if (
        len(retrieval.fts) > request.retrieval.max_fts_candidates
        or len(retrieval.dense) > request.retrieval.max_dense_candidates
        or len(retrieval.fused) > request.retrieval.max_fused_candidates
    ):
        raise ExpandedExecutionError("base retrieval trace exceeds the requested budget")

    expected_items = _derive_exact_base_selection(
        retrieval=retrieval,
        packet_limit=request.retrieval.max_source_fragments,
        max_fused_candidates=request.retrieval.max_fused_candidates,
        permitted_manifest=compiled.manifest,
        layer_profile=layer_profile,
        fusion_profile=fusion_profile,
    )
    expected_status = "evidence_found" if expected_items else "no_evidence"
    if (
        artifacts.packet.items != expected_items
        or artifacts.packet.result_status != expected_status
    ):
        raise ExpandedExecutionError("base packet is not the exact fused selection")


def _derive_exact_base_selection(
    *,
    retrieval: HybridRetrievalReceipt,
    packet_limit: int,
    max_fused_candidates: int,
    permitted_manifest: PermittedManifest,
    layer_profile: CorpusLayerProfile,
    fusion_profile: FusionProfile,
) -> tuple[HybridPacketItem, ...]:
    """Re-derive exact RRF scores/order and the family/layer-aware base packet."""

    manifest_by_id = {item.source_fragment_id: item for item in permitted_manifest.items}
    layer_by_source = {item.source_id: item.layer for item in layer_profile.items}
    for trace_name, trace in (("FTS", retrieval.fts), ("dense", retrieval.dense)):
        for item in trace:
            manifest = manifest_by_id.get(item.source_fragment_id)
            if manifest is None or manifest.source_id != item.source_id:
                raise ExpandedExecutionError(
                    f"base {trace_name} trace contains unauthorized provenance"
                )
    fts_candidates = tuple(
        _fusion_candidate(item, manifest_by_id, layer_by_source) for item in retrieval.fts
    )
    dense_candidates = tuple(
        _fusion_candidate(item, manifest_by_id, layer_by_source) for item in retrieval.dense
    )
    expected_fused = reciprocal_rank_fusion(
        fts_candidates,
        dense_candidates,
        profile=fusion_profile,
        limit=max_fused_candidates,
    )
    expected_trace = tuple(
        HybridRankTraceItem(
            source_fragment_id=item.candidate.source_fragment_id,
            source_id=item.candidate.source_id,
            rank=rank,
            score_repr=item.score_repr,
        )
        for rank, item in enumerate(expected_fused, start=1)
    )
    if retrieval.fused != expected_trace:
        raise ExpandedExecutionError("base fused trace is not exact reciprocal-rank fusion")
    selected_ids = select_layer_aware_fragment_ids(
        tuple(
            (
                item.candidate.source_fragment_id,
                item.candidate.source_id,
                item.candidate.layer,
            )
            for item in expected_fused
        ),
        limit=packet_limit,
        profile=fusion_profile,
    )
    fused_by_id = {item.candidate.source_fragment_id: item.candidate for item in expected_fused}
    return tuple(
        HybridPacketItem(
            rank=rank,
            source_fragment_id=fragment_id,
            source_id=fused_by_id[fragment_id].source_id,
            source_family_id=fused_by_id[fragment_id].source_family_id,
            layer=fused_by_id[fragment_id].layer,
        )
        for rank, fragment_id in enumerate(selected_ids, start=1)
    )


def _fusion_candidate(
    item: HybridRankTraceItem,
    manifest_by_id: dict[str, PermittedManifestItem],
    layer_by_source: dict[str, CorpusLayer],
) -> FusionCandidate:
    manifest = manifest_by_id[item.source_fragment_id]
    family_id = manifest.source_family_id
    if family_id is None:
        raise ExpandedExecutionError("expanded recall requires SourceFamily provenance")
    try:
        layer = layer_by_source[item.source_id]
    except KeyError as error:
        raise ExpandedExecutionError("trace Source lacks a corpus layer") from error
    return FusionCandidate(
        source_fragment_id=item.source_fragment_id,
        source_id=item.source_id,
        source_family_id=family_id,
        layer=layer,
    )


def _project_candidates(
    *,
    request: ExpandedQueryRequest,
    base_request: HybridQueryRequest,
    compiled_access: CompiledAccess,
    fragment_by_id: dict[str, SourceFragmentText],
    manifest_by_id: dict[str, PermittedManifestItem],
    layer_by_source: dict[str, CorpusLayer],
    seed_score: dict[str, Fraction],
    seed_rank: dict[str, int],
    structure_generation: StructureGeneration | None,
    structure_reads: tuple[StructureUnitText, ...],
) -> tuple[_ProjectedCandidate, ...]:
    if not request.structure_enabled:
        if structure_generation is not None or structure_reads != ():
            raise ExpandedExecutionError("structure-off execution cannot accept structure data")
        return tuple(
            _atomic_candidate(
                fragment_by_id[fragment_id],
                manifest_by_id[fragment_id],
                layer_by_source,
                seed_rank[fragment_id],
                score,
            )
            for fragment_id, score in seed_score.items()
        )
    if type(structure_generation) is not StructureGeneration:
        raise ExpandedExecutionError("structure-enabled request requires exact generation")
    if type(structure_reads) is not tuple or any(
        type(item) is not StructureUnitText for item in structure_reads
    ):
        raise ExpandedExecutionError("structure_reads must use exact StructureUnitText values")
    generation = structure_generation
    _require_generation_closure(
        request,
        base_request,
        compiled_access,
        generation,
        fragment_by_id,
        manifest_by_id,
    )
    anchored = tuple(
        unit
        for unit in generation.units
        if any(member.source_fragment_id in seed_rank for member in unit.members)
    )
    _require_non_overlapping_anchored_units(anchored)
    expected_ids = tuple(item.structure_unit_id for item in anchored)
    if tuple(item.structure_unit.structure_unit_id for item in structure_reads) != expected_ids:
        raise ExpandedExecutionError("structure reads must equal the exact anchored-unit set")
    reads_by_id = {item.structure_unit.structure_unit_id: item for item in structure_reads}
    covered: set[str] = set()
    result: list[_ProjectedCandidate] = []
    for unit in anchored:
        read = reads_by_id[unit.structure_unit_id]
        _require_structure_read(
            base_request,
            compiled_access,
            unit,
            read,
            fragment_by_id,
            manifest_by_id,
        )
        member_ids = tuple(item.source_fragment_id for item in unit.members)
        anchors = tuple(sorted(seed_rank[item] for item in member_ids if item in seed_rank))
        scores = tuple(seed_score[item] for item in member_ids if item in seed_score)
        manifest = manifest_by_id[member_ids[0]]
        family_id = manifest.source_family_id
        if family_id is None:
            raise ExpandedExecutionError("StructureUnit requires SourceFamily provenance")
        path = ExpansionPath(
            candidate_kind=ExpandedCandidateKind.STRUCTURE_UNIT,
            candidate_id=unit.structure_unit_id,
            source_fragment_ids=member_ids,
            source_ids=(unit.source_id,),
            source_family_ids=(family_id,),
            corpus_layers=(layer_by_source[unit.source_id],),
            anchor_atomic_ranks=anchors,
            text_sha256=read.receipt.rendered_text_sha256,
            structure_read_receipt_id=read.receipt.receipt_id,
            structure_read_receipt_hash=read.receipt.receipt_hash,
        )
        result.append(_ProjectedCandidate(path, read.text, max(scores)))
        covered.update(item for item in member_ids if item in seed_rank)
    for fragment_id, score in seed_score.items():
        if fragment_id not in covered:
            result.append(
                _atomic_candidate(
                    fragment_by_id[fragment_id],
                    manifest_by_id[fragment_id],
                    layer_by_source,
                    seed_rank[fragment_id],
                    score,
                )
            )
    if len({item.key for item in result}) != len(result):
        raise ExpandedExecutionError("candidate projection produced duplicate identities")
    return tuple(result)


def _atomic_candidate(
    fragment: SourceFragmentText,
    manifest: PermittedManifestItem,
    layer_by_source: dict[str, CorpusLayer],
    rank: int,
    score: Fraction,
) -> _ProjectedCandidate:
    family_id = manifest.source_family_id
    if family_id is None:
        raise ExpandedExecutionError("atomic candidate requires SourceFamily provenance")
    path = ExpansionPath(
        candidate_kind=ExpandedCandidateKind.SOURCE_FRAGMENT,
        candidate_id=fragment.source_fragment_id,
        source_fragment_ids=(fragment.source_fragment_id,),
        source_ids=(fragment.source_id,),
        source_family_ids=(family_id,),
        corpus_layers=(layer_by_source[fragment.source_id],),
        anchor_atomic_ranks=(rank,),
        text_sha256=fragment.text_sha256,
    )
    return _ProjectedCandidate(path, fragment.text, score)


def _require_generation_closure(
    request: ExpandedQueryRequest,
    base_request: HybridQueryRequest,
    compiled: CompiledAccess,
    generation: StructureGeneration,
    fragment_by_id: dict[str, SourceFragmentText],
    manifest_by_id: dict[str, PermittedManifestItem],
) -> None:
    generation_hash = canonical_sha256_hex(generation.payload())
    expected_id = canonical_content_id(
        "structure_generation",
        {
            "schema": "dithyramba.structure_generation_identity/1.0",
            "generation_hash": generation_hash,
        },
    )
    if (
        generation.generation_hash != generation_hash
        or generation.generation_id != expected_id
        or request.structure_generation_id != generation.generation_id
        or request.structure_generation_hash != generation.generation_hash
        or generation.library_id != base_request.library_id
        or generation.corpus_snapshot_id != base_request.corpus_snapshot_id
        or generation.access_policy_id != base_request.access_policy_id
        or generation.policy_hash != compiled.token.policy_hash
        or generation.scope_hash != compiled.token.exclusion_hash
        or generation.permitted_set_hash != compiled.token.permitted_set_hash
    ):
        raise ExpandedExecutionError("StructureGeneration differs from request/access closure")
    if type(generation.units) is not tuple or any(
        type(item) is not StructureUnit for item in generation.units
    ):
        raise ExpandedExecutionError("StructureGeneration units must use the exact contract")
    for unit in generation.units:
        _require_unit_content(unit)
        if unit.generation_id != generation.generation_id:
            raise ExpandedExecutionError(
                "StructureUnit generation ID differs from its exact generation"
            )
        unit_families: set[str | None] = set()
        for member in unit.members:
            manifest = manifest_by_id.get(member.source_fragment_id)
            fragment = fragment_by_id.get(member.source_fragment_id)
            if manifest is None or fragment is None:
                raise ExpandedExecutionError(
                    "StructureGeneration contains a member outside permitted access"
                )
            if (
                manifest.source_id != unit.source_id
                or manifest.source_version_id != unit.source_version_id
                or fragment.source_id != unit.source_id
                or fragment.source_version_id != unit.source_version_id
                or fragment.ordinal != member.ordinal
                or fragment.text_sha256 != member.text_sha256
            ):
                raise ExpandedExecutionError(
                    "StructureGeneration contains changed or foreign unit provenance"
                )
            unit_families.add(manifest.source_family_id)
        if len(unit_families) != 1 or None in unit_families:
            raise ExpandedExecutionError(
                "StructureGeneration unit lacks one exact SourceFamily closure"
            )


def _require_unit_content(unit: StructureUnit) -> None:
    expected_content_hash = canonical_sha256_hex(unit.semantic_payload())
    expected_unit_id = canonical_content_id("structure_unit", unit.semantic_payload())
    member_ids = [item.source_fragment_id for item in unit.members]
    boundary = {
        "schema": "dithyramba.structure_boundary/1.0",
        "source_version_id": unit.source_version_id,
        "first_ordinal": unit.first_ordinal,
        "last_ordinal": unit.last_ordinal,
        "source_fragment_ids": member_ids,
    }
    if (
        unit.content_hash != expected_content_hash
        or unit.structure_unit_id != expected_unit_id
        or unit.boundary_hash != canonical_sha256_hex(boundary)
    ):
        raise ExpandedExecutionError("StructureUnit content identity changed")
    if any(type(item) is not StructureUnitMember for item in unit.members):
        raise ExpandedExecutionError("StructureUnit members must use the exact contract")


def _require_non_overlapping_anchored_units(units: tuple[StructureUnit, ...]) -> None:
    owner: dict[str, str] = {}
    for unit in units:
        for member in unit.members:
            prior = owner.setdefault(member.source_fragment_id, unit.structure_unit_id)
            if prior != unit.structure_unit_id:
                raise ExpandedExecutionError("anchored StructureUnits overlap ambiguously")


def _require_structure_read(
    request: HybridQueryRequest,
    compiled: CompiledAccess,
    unit: StructureUnit,
    read: StructureUnitText,
    fragment_by_id: dict[str, SourceFragmentText],
    manifest_by_id: dict[str, PermittedManifestItem],
) -> None:
    if read.structure_unit != unit or type(read.receipt) is not StructureReadReceipt:
        raise ExpandedExecutionError("StructureUnit read differs from its generation")
    receipt = read.receipt
    if receipt.receipt_hash != canonical_sha256_hex(receipt.payload()):
        raise ExpandedExecutionError("StructureReadReceipt content hash changed")
    if receipt.receipt_id != f"structure_read_{receipt.receipt_hash[:32]}":
        raise ExpandedExecutionError("StructureReadReceipt ID is not content-derived")
    if (
        receipt.structure_unit_id != unit.structure_unit_id
        or receipt.library_id != request.library_id
        or receipt.corpus_snapshot_id != request.corpus_snapshot_id
        or receipt.access_policy_id != request.access_policy_id
        or receipt.policy_hash != compiled.token.policy_hash
        or receipt.scope_hash != compiled.token.exclusion_hash
        or receipt.permitted_set_hash != compiled.token.permitted_set_hash
    ):
        raise ExpandedExecutionError("StructureReadReceipt differs from access closure")
    expected_parts: list[str] = []
    expected_items: list[StructureReadReceiptItem] = []
    cursor = 0
    families: set[str | None] = set()
    for order, member in enumerate(unit.members):
        fragment = fragment_by_id.get(member.source_fragment_id)
        manifest = manifest_by_id.get(member.source_fragment_id)
        if fragment is None or manifest is None:
            raise ExpandedExecutionError("StructureUnit member is outside authorized access")
        if (
            fragment.source_id != unit.source_id
            or fragment.source_version_id != unit.source_version_id
            or fragment.ordinal != member.ordinal
            or fragment.text_sha256 != member.text_sha256
            or manifest.source_id != unit.source_id
            or manifest.source_version_id != unit.source_version_id
        ):
            raise ExpandedExecutionError("StructureUnit contains mixed or changed provenance")
        families.add(manifest.source_family_id)
        if order:
            cursor += len(receipt.delimiter)
        start = cursor
        cursor += len(fragment.text)
        expected_items.append(
            StructureReadReceiptItem(
                source_fragment_id=fragment.source_fragment_id,
                member_order=order,
                text_sha256=fragment.text_sha256,
                rendered_start=start,
                rendered_end=cursor,
            )
        )
        expected_parts.append(fragment.text)
    rendered = receipt.delimiter.join(expected_parts)
    if len(families) != 1 or None in families:
        raise ExpandedExecutionError("StructureUnit contains mixed SourceFamily provenance")
    if (
        receipt.items != tuple(expected_items)
        or read.text != rendered
        or receipt.rendered_text_sha256 != sha256_hex(rendered.encode("utf-8"))
    ):
        raise ExpandedExecutionError("StructureUnit rendered text/span closure changed")


def _pair_token_lengths(
    provider: ExpandedReranker,
    profile: RerankerProfile,
    question: str,
    texts: tuple[str, ...],
) -> tuple[int, ...]:
    if not texts:
        return ()
    _require_provider_profile(provider, profile)
    try:
        counts = provider.pair_token_lengths(question, texts)
    except Exception as error:
        raise ExpandedExecutionError("reranker token preflight failed closed") from error
    _require_provider_profile(provider, profile)
    if (
        type(counts) is not tuple
        or len(counts) != len(texts)
        or any(type(item) is not int or not 1 <= item <= 1_000_000 for item in counts)
    ):
        raise ExpandedExecutionError("reranker changed token-count cardinality or type")
    return counts


def _score_text_pairs(
    provider: ExpandedReranker,
    profile: RerankerProfile,
    question: str,
    texts: tuple[str, ...],
) -> tuple[float, ...]:
    _require_provider_profile(provider, profile)
    try:
        scores = provider.score_text_pairs(question, texts)
    except Exception as error:
        raise ExpandedExecutionError("expanded reranker failed closed") from error
    _require_provider_profile(provider, profile)
    if (
        type(scores) is not tuple
        or len(scores) != len(texts)
        or any(type(item) is not float or not math.isfinite(item) for item in scores)
    ):
        raise ExpandedExecutionError("reranker changed cardinality or emitted non-finite scores")
    return scores


def _require_provider_profile(
    provider: ExpandedReranker,
    profile: RerankerProfile,
) -> None:
    if not callable(getattr(provider, "pair_token_lengths", None)) or not callable(
        getattr(provider, "score_text_pairs", None)
    ):
        raise ExpandedExecutionError("expanded reranker methods are unavailable")
    try:
        actual = provider.reranker_profile
    except Exception as error:
        raise ExpandedExecutionError("expanded reranker profile is unavailable") from error
    if type(actual) is not RerankerProfile or actual != profile:
        raise ExpandedExecutionError("expanded reranker profile changed")


def _trace(
    candidate: _ProjectedCandidate,
    *,
    rank: int,
    pair_token_count: int,
    score_hex: str,
) -> ExpandedCandidateTrace:
    path = candidate.path
    return ExpandedCandidateTrace(
        candidate_kind=path.candidate_kind,
        candidate_id=path.candidate_id,
        relation_node_type=path.relation_node_type,
        rank=rank,
        source_fragment_ids=path.source_fragment_ids,
        source_ids=path.source_ids,
        source_family_ids=path.source_family_ids,
        corpus_layers=path.corpus_layers,
        anchor_atomic_ranks=path.anchor_atomic_ranks,
        text_sha256=path.text_sha256,
        pair_token_count=pair_token_count,
        score_hex=score_hex,
    )


def _select_final(
    reranked: tuple[ExpandedCandidateTrace, ...],
    limit: int,
    profile: FusionProfile,
) -> tuple[ExpandedCandidateTrace, ...]:
    """Select exact candidates while capping dependent derived source families.

    The original hybrid selector is Source-shaped.  Expanded candidates can be
    StructureUnits or future relation nodes, so this boundary counts the exact
    ``source_family_ids`` carried by each candidate.  A multi-family derived
    candidate consumes one slot in every family it depends on.
    """

    if type(reranked) is not tuple or any(
        type(item) is not ExpandedCandidateTrace for item in reranked
    ):
        raise ExpandedExecutionError("final selection requires exact reranked candidates")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ExpandedExecutionError("final selection limit must be between 1 and 100")
    if type(profile) is not FusionProfile:
        raise ExpandedExecutionError("final selection requires the exact FusionProfile")

    selected: list[ExpandedCandidateTrace] = []
    selected_keys: set[tuple[str, str, str]] = set()
    derived_top_five: dict[str, int] = {}
    top_five_target = min(5, limit)
    for item in reranked:
        if len(selected) >= top_five_target:
            break
        has_derived = CorpusLayer.DERIVED in item.corpus_layers
        if has_derived and any(layer is not CorpusLayer.DERIVED for layer in item.corpus_layers):
            raise ExpandedExecutionError(
                "mixed-layer derived candidate cannot bind families to layers exactly"
            )
        if has_derived:
            if any(
                derived_top_five.get(family_id, 0) >= profile.max_per_derived_source_top_five
                for family_id in item.source_family_ids
            ):
                continue
            for family_id in item.source_family_ids:
                derived_top_five[family_id] = derived_top_five.get(family_id, 0) + 1
        selected.append(item)
        selected_keys.add(item.candidate_key)
    if len(selected) < top_five_target:
        return tuple(selected)
    for item in reranked:
        if len(selected) >= limit:
            break
        if item.candidate_key not in selected_keys:
            selected.append(item)
            selected_keys.add(item.candidate_key)
    return tuple(selected)


def _exact_fused_score(value: str) -> Fraction:
    try:
        score = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ExpandedExecutionError("base fused score is not an exact fraction") from error
    if score <= 0 or f"{score.numerator}/{score.denominator}" != value:
        raise ExpandedExecutionError("base fused score is not a canonical positive fraction")
    return score


def _fraction_float_hex(score: Fraction) -> str:
    try:
        value = float(score)
    except OverflowError as error:
        raise ExpandedExecutionError("base fused score cannot become a finite float") from error
    if not math.isfinite(value):
        raise ExpandedExecutionError("base fused score cannot become a finite float")
    return value.hex()


def _retrieval_corpus_hash(fragments: tuple[SourceFragmentText, ...]) -> str:
    values = tuple(
        FtsFragment(
            source_fragment_id=item.source_fragment_id,
            text=item.text,
            text_sha256=item.text_sha256,
        )
        for item in fragments
    )
    return canonical_sha256_hex(
        {
            "schema": "dithyramba.fts_corpus/1.0",
            "fragments": [
                item.identity_payload()
                for item in sorted(values, key=lambda value: value.source_fragment_id)
            ],
        }
    )


def _read_receipt_corpus_hash(
    items: tuple[HybridReadReceiptItem, ...],
) -> str:
    """Re-derive the FTS corpus identity from a text-free read receipt."""

    return canonical_sha256_hex(
        {
            "schema": "dithyramba.fts_corpus/1.0",
            "fragments": [
                {
                    "source_fragment_id": item.source_fragment_id,
                    "text_sha256": item.text_sha256,
                }
                for item in sorted(items, key=lambda value: value.source_fragment_id)
            ],
        }
    )
