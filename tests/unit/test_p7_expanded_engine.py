"""Quality, integrity and determinism tests for structure-expanded recall."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, cast

import pytest

import dithyramba.recall.expanded_engine as expanded_engine
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
from dithyramba.recall.expanded_engine import (
    ExpandedExecutionArtifacts,
    ExpandedExecutionError,
    execute_expanded_retrieval,
)
from dithyramba.recall.expanded_models import (
    ExpandedCandidateKind,
    ExpandedPacketItem,
    ExpandedQueryRequest,
    ExpansionOmission,
    ExpansionOmissionReason,
    ExpansionSeedRank,
    GraphExpansionConfig,
)
from dithyramba.recall.fts import FtsFragment
from dithyramba.recall.fusion import (
    FusionCandidate,
    reciprocal_rank_fusion,
    select_layer_aware_fragment_ids,
)
from dithyramba.recall.hybrid_artifacts import (
    HybridAccessReceipt,
    HybridCoverageReport,
    HybridReadReceipt,
    HybridReadReceiptItem,
)
from dithyramba.recall.hybrid_engine import HybridExecutionArtifacts
from dithyramba.recall.hybrid_models import (
    CorpusLayer,
    CorpusLayerItem,
    CorpusLayerProfile,
    FusionProfile,
    HybridEvidencePacket,
    HybridPacketItem,
    HybridQueryRequest,
    HybridRankTraceItem,
    HybridRetrievalBudget,
    HybridRetrievalReceipt,
)
from dithyramba.recall.rerank import RerankerProfile
from dithyramba.relations import RelationType
from dithyramba.structure import (
    StructureGeneration,
    StructureReadReceipt,
    StructureReadReceiptItem,
    StructureUnit,
    StructureUnitKind,
    StructureUnitMember,
    StructureUnitText,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
QUESTION = "Which complete evidence best supports the claim?"


@dataclass(frozen=True, slots=True)
class ExpandedCase:
    base_request: HybridQueryRequest
    base_artifacts: HybridExecutionArtifacts
    compiled_access: CompiledAccess
    fragments: tuple[SourceFragmentText, ...]
    layer_profile: CorpusLayerProfile
    fusion_profile: FusionProfile
    reranker_profile: RerankerProfile


@dataclass(frozen=True, slots=True)
class FragmentSpec:
    suffix: str
    source_suffix: str
    layer: CorpusLayer
    ordinal: int
    text: str
    family_suffix: str | None = None


DEFAULT_SPECS = (
    FragmentSpec("alpha", "document", CorpusLayer.PRIMARY, 0, "Alpha opening evidence."),
    FragmentSpec("beta", "document", CorpusLayer.PRIMARY, 1, "Beta completes the thought."),
    FragmentSpec("epsilon", "document", CorpusLayer.PRIMARY, 2, "Epsilon closes the section."),
    FragmentSpec("gamma", "other", CorpusLayer.SYNTHESIS, 0, "Gamma independent context."),
)


class RecordingExpandedReranker:
    def __init__(
        self,
        profile: RerankerProfile,
        *,
        counts: tuple[int, ...] | None = None,
        scores: tuple[float, ...] | None = None,
    ) -> None:
        self.reranker_profile = profile
        self.counts = counts
        self.scores = scores
        self.token_calls: list[tuple[str, tuple[str, ...]]] = []
        self.score_calls: list[tuple[str, tuple[str, ...]]] = []

    def pair_token_lengths(
        self, question: str, candidate_texts: tuple[str, ...]
    ) -> tuple[int, ...]:
        self.token_calls.append((question, candidate_texts))
        return self.counts or tuple(10 + index for index, _ in enumerate(candidate_texts))

    def score_text_pairs(
        self, question: str, candidate_texts: tuple[str, ...]
    ) -> tuple[float, ...]:
        self.score_calls.append((question, candidate_texts))
        return self.scores or tuple(
            float(len(candidate_texts) - index) for index in range(len(candidate_texts))
        )


def _profile(*, max_tokens: int = 512) -> RerankerProfile:
    return RerankerProfile(
        model_id="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        revision="d" * 40,
        license="Apache-2.0",
        max_tokens=max_tokens,
    )


def _make_case(
    specs: tuple[FragmentSpec, ...] = DEFAULT_SPECS,
    *,
    base_packet_limit: int = 2,
    final_token_limit: int = 512,
    seed_suffixes: tuple[str, ...] | None = None,
    familyless_suffixes: tuple[str, ...] = (),
) -> ExpandedCase:
    ordered_specs = tuple(sorted(specs, key=lambda item: item.suffix.encode("ascii")))
    snapshot_hash = HASH_A
    snapshot_id = f"snapshot_{snapshot_hash[:32]}"
    library_id = "library_one"
    collection_id = "collection_one"
    policy_id = "policy_one"
    exclusions = QueryExclusions()
    scope = RequestScope(
        library_id=library_id,
        snapshot_hash=snapshot_hash,
        purpose="research",
        collection_ids=(collection_id,),
        exclusions=exclusions,
    )
    manifest = PermittedManifest(
        library_id=library_id,
        snapshot_hash=snapshot_hash,
        items=tuple(
            PermittedManifestItem(
                source_fragment_id=f"fragment_{item.suffix}",
                source_version_id=f"source_version_{item.source_suffix}",
                source_id=f"source_{item.source_suffix}",
                source_family_id=(
                    None
                    if item.suffix in familyless_suffixes
                    else f"family_{item.family_suffix or item.source_suffix}"
                ),
                collection_ids=(collection_id,),
            )
            for item in ordered_specs
        ),
    )
    token = PermittedToken(
        library_id=library_id,
        snapshot_hash=snapshot_hash,
        policy_hash=HASH_B,
        exclusion_hash=scope.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
    )
    compiled = CompiledAccess(manifest, token, PublicAccessResult(False))
    fragments = tuple(
        SourceFragmentText(
            source_fragment_id=f"fragment_{item.suffix}",
            source_version_id=f"source_version_{item.source_suffix}",
            source_id=f"source_{item.source_suffix}",
            ordinal=item.ordinal,
            fragment_kind="paragraph",
            text=item.text,
            text_sha256=sha256_hex(item.text.encode("utf-8")),
            source_address_json='{"paragraph":1}',
        )
        for item in ordered_specs
    )
    layer_by_source = {f"source_{item.source_suffix}": item.layer for item in ordered_specs}
    layer_profile = CorpusLayerProfile(
        library_id=library_id,
        corpus_snapshot_id=snapshot_id,
        exclusion_hash=token.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
        items=tuple(
            CorpusLayerItem(source_id=source_id, layer=layer)
            for source_id, layer in layer_by_source.items()
        ),
    )
    fusion_profile = FusionProfile()
    request = HybridQueryRequest(
        question=QUESTION,
        library_id=library_id,
        collection_ids=(collection_id,),
        corpus_snapshot_id=snapshot_id,
        access_policy_id=policy_id,
        purpose="research",
        model_profile_hash=HASH_A,
        runtime_profile_hash=HASH_B,
        layer_profile_hash=layer_profile.profile_hash,
        fusion_profile_hash=fusion_profile.profile_hash,
        exclusions=exclusions,
        retrieval=HybridRetrievalBudget(
            max_fts_candidates=max(1, len(fragments)),
            max_dense_candidates=max(1, len(fragments)),
            max_fused_candidates=max(1, len(fragments)),
            max_source_fragments=base_packet_limit,
        ),
    )
    seed_ids = (
        {f"fragment_{item}" for item in seed_suffixes}
        if seed_suffixes is not None
        else {item.source_fragment_id for item in fragments}
    )
    if not seed_ids.issubset({item.source_fragment_id for item in fragments}):
        raise AssertionError("test seed suffixes must identify permitted fragments")
    seeded_fragments = tuple(item for item in fragments if item.source_fragment_id in seed_ids)
    fts = tuple(
        HybridRankTraceItem(
            source_fragment_id=item.source_fragment_id,
            source_id=item.source_id,
            rank=rank,
            score_repr=f"{len(seeded_fragments) - rank + 1}.0",
        )
        for rank, item in enumerate(seeded_fragments, start=1)
    )
    metadata = tuple(
        FusionCandidate(
            source_fragment_id=item.source_fragment_id,
            source_id=item.source_id,
            source_family_id=manifest.items[index].source_family_id or "family_placeholder",
            layer=layer_by_source[item.source_id],
        )
        for index, item in enumerate(fragments)
        if item.source_fragment_id in seed_ids
    )
    fused_values = reciprocal_rank_fusion(
        metadata,
        (),
        profile=fusion_profile,
        limit=max(1, len(seeded_fragments)),
    )
    fused = tuple(
        HybridRankTraceItem(
            source_fragment_id=item.candidate.source_fragment_id,
            source_id=item.candidate.source_id,
            rank=rank,
            score_repr=item.score_repr,
        )
        for rank, item in enumerate(fused_values, start=1)
    )
    retrieval_corpus_hash = _corpus_hash(fragments)
    retrieval = HybridRetrievalReceipt(
        request_id=request.request_id,
        request_hash=request.request_hash,
        exclusion_hash=token.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
        model_profile_hash=request.model_profile_hash,
        runtime_profile_hash=request.runtime_profile_hash,
        layer_profile_hash=request.layer_profile_hash,
        fusion_profile_hash=request.fusion_profile_hash,
        vector_generation_hash=HASH_C,
        fts_profile_version="fts5-v1",
        fts_result_hash=HASH_A,
        query_vector_hash=HASH_B,
        fts=fts,
        fused=fused,
    )
    read = HybridReadReceipt(
        request_id=request.request_id,
        request_hash=request.request_hash,
        retrieval_corpus_hash=retrieval_corpus_hash,
        items=tuple(
            HybridReadReceiptItem(
                source_fragment_id=item.source_fragment_id,
                source_version_id=item.source_version_id,
                read_order=order,
                text_sha256=item.text_sha256,
            )
            for order, item in enumerate(fragments)
        ),
    )
    access = HybridAccessReceipt(
        library_id=library_id,
        request_id=request.request_id,
        request_hash=request.request_hash,
        access_policy_id=policy_id,
        policy_hash=token.policy_hash,
        corpus_snapshot_id=snapshot_id,
        snapshot_hash=snapshot_hash,
        exclusion_hash=token.exclusion_hash,
        permitted_set_hash=manifest.permitted_set_hash,
        retrieval_corpus_hash=retrieval_corpus_hash,
    )
    coverage = HybridCoverageReport(
        request_id=request.request_id,
        request_hash=request.request_hash,
        processed_count=len(fragments),
    )
    selected_ids = select_layer_aware_fragment_ids(
        tuple(
            (
                item.candidate.source_fragment_id,
                item.candidate.source_id,
                item.candidate.layer,
            )
            for item in fused_values
        ),
        limit=base_packet_limit,
        profile=fusion_profile,
    )
    by_id = {item.candidate.source_fragment_id: item.candidate for item in fused_values}
    packet_items = tuple(
        HybridPacketItem(
            rank=rank,
            source_fragment_id=fragment_id,
            source_id=by_id[fragment_id].source_id,
            source_family_id=by_id[fragment_id].source_family_id,
            layer=by_id[fragment_id].layer,
        )
        for rank, fragment_id in enumerate(selected_ids, start=1)
    )
    packet = HybridEvidencePacket(
        request_id=request.request_id,
        request_hash=request.request_hash,
        corpus_snapshot_id=snapshot_id,
        access_policy_id=policy_id,
        read_receipt_hash=read.receipt_hash,
        access_receipt_hash=access.receipt_hash,
        coverage_report_hash=coverage.report_hash,
        retrieval_receipt_hash=retrieval.receipt_hash,
        result_status="evidence_found" if packet_items else "no_evidence",
        items=packet_items,
    )
    return ExpandedCase(
        request,
        HybridExecutionArtifacts(read, access, coverage, retrieval, packet),
        compiled,
        fragments,
        layer_profile,
        fusion_profile,
        _profile(max_tokens=final_token_limit),
    )


def _corpus_hash(fragments: tuple[SourceFragmentText, ...]) -> str:
    values = tuple(
        FtsFragment(item.source_fragment_id, item.text, item.text_sha256) for item in fragments
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


def _expanded_request(
    case: ExpandedCase,
    generation: StructureGeneration | None = None,
    *,
    max_final: int = 20,
    graph: GraphExpansionConfig | None = None,
) -> ExpandedQueryRequest:
    return ExpandedQueryRequest(
        base_request_id=case.base_request.request_id,
        base_request_hash=case.base_request.request_hash,
        structure_generation_id=None if generation is None else generation.generation_id,
        structure_generation_hash=None if generation is None else generation.generation_hash,
        reranker_profile_hash=case.reranker_profile.profile_hash,
        graph=graph,
        max_final=max_final,
    )


def _generation(
    case: ExpandedCase,
    definitions: tuple[tuple[StructureUnitKind, tuple[str, ...], str | None], ...],
) -> tuple[StructureGeneration, tuple[StructureUnitText, ...]]:
    fragments = {item.source_fragment_id: item for item in case.fragments}
    prepared: list[tuple[str, dict[str, object], str, tuple[StructureUnitMember, ...]]] = []
    for kind, fragment_ids, label in definitions:
        first = fragments[fragment_ids[0]]
        members = tuple(
            StructureUnitMember(
                source_fragment_id=fragment_id,
                member_order=order,
                ordinal=fragments[fragment_id].ordinal,
                text_sha256=fragments[fragment_id].text_sha256,
            )
            for order, fragment_id in enumerate(fragment_ids)
        )
        boundary_hash = canonical_sha256_hex(
            {
                "schema": "dithyramba.structure_boundary/1.0",
                "source_version_id": first.source_version_id,
                "first_ordinal": members[0].ordinal,
                "last_ordinal": members[-1].ordinal,
                "source_fragment_ids": list(fragment_ids),
            }
        )
        semantic: dict[str, object] = {
            "schema": "dithyramba.structure_unit/1.0",
            "source_id": first.source_id,
            "source_version_id": first.source_version_id,
            "kind": kind.value,
            "label": label,
            "members": [item.payload() for item in members],
            "boundary_hash": boundary_hash,
        }
        prepared.append(
            (
                canonical_content_id("structure_unit", semantic),
                semantic,
                canonical_sha256_hex(semantic),
                members,
            )
        )
    prepared.sort(key=lambda item: item[0])
    generation_payload = {
        "schema": "dithyramba.structure_generation/1.0",
        "library_id": case.base_request.library_id,
        "corpus_snapshot_id": case.base_request.corpus_snapshot_id,
        "access_policy_id": case.base_request.access_policy_id,
        "policy_hash": case.compiled_access.token.policy_hash,
        "scope_hash": case.compiled_access.token.exclusion_hash,
        "permitted_set_hash": case.compiled_access.token.permitted_set_hash,
        "profile_id": "natural-units",
        "profile_version": "1.0",
        "units": [
            semantic | {"structure_unit_id": unit_id, "content_hash": content_hash}
            for unit_id, semantic, content_hash, _members in prepared
        ],
    }
    generation_hash = canonical_sha256_hex(generation_payload)
    generation_id = canonical_content_id(
        "structure_generation",
        {
            "schema": "dithyramba.structure_generation_identity/1.0",
            "generation_hash": generation_hash,
        },
    )
    units = tuple(
        StructureUnit(
            structure_unit_id=unit_id,
            generation_id=generation_id,
            source_id=cast(str, semantic["source_id"]),
            source_version_id=cast(str, semantic["source_version_id"]),
            kind=StructureUnitKind(cast(str, semantic["kind"])),
            label=cast(str | None, semantic["label"]),
            members=members,
            boundary_hash=cast(str, semantic["boundary_hash"]),
            content_hash=content_hash,
        )
        for unit_id, semantic, content_hash, members in prepared
    )
    generation = StructureGeneration(
        generation_id=generation_id,
        library_id=case.base_request.library_id,
        corpus_snapshot_id=case.base_request.corpus_snapshot_id,
        access_policy_id=case.base_request.access_policy_id,
        policy_hash=case.compiled_access.token.policy_hash,
        scope_hash=case.compiled_access.token.exclusion_hash,
        permitted_set_hash=case.compiled_access.token.permitted_set_hash,
        profile_id="natural-units",
        profile_version="1.0",
        units=units,
        generation_hash=generation_hash,
    )
    reads = tuple(_read(case, unit) for unit in units)
    return generation, reads


def _rebuild_generation(
    case: ExpandedCase,
    units: tuple[StructureUnit, ...],
) -> StructureGeneration:
    ordered = tuple(sorted(units, key=lambda item: item.structure_unit_id.encode("ascii")))
    payload = {
        "schema": "dithyramba.structure_generation/1.0",
        "library_id": case.base_request.library_id,
        "corpus_snapshot_id": case.base_request.corpus_snapshot_id,
        "access_policy_id": case.base_request.access_policy_id,
        "policy_hash": case.compiled_access.token.policy_hash,
        "scope_hash": case.compiled_access.token.exclusion_hash,
        "permitted_set_hash": case.compiled_access.token.permitted_set_hash,
        "profile_id": "natural-units",
        "profile_version": "1.0",
        "units": [
            item.semantic_payload()
            | {
                "structure_unit_id": item.structure_unit_id,
                "content_hash": item.content_hash,
            }
            for item in ordered
        ],
    }
    generation_hash = canonical_sha256_hex(payload)
    generation_id = canonical_content_id(
        "structure_generation",
        {
            "schema": "dithyramba.structure_generation_identity/1.0",
            "generation_hash": generation_hash,
        },
    )
    rebound = tuple(replace(item, generation_id=generation_id) for item in ordered)
    return StructureGeneration(
        generation_id=generation_id,
        library_id=case.base_request.library_id,
        corpus_snapshot_id=case.base_request.corpus_snapshot_id,
        access_policy_id=case.base_request.access_policy_id,
        policy_hash=case.compiled_access.token.policy_hash,
        scope_hash=case.compiled_access.token.exclusion_hash,
        permitted_set_hash=case.compiled_access.token.permitted_set_hash,
        profile_id="natural-units",
        profile_version="1.0",
        units=rebound,
        generation_hash=generation_hash,
    )


def _read(case: ExpandedCase, unit: StructureUnit) -> StructureUnitText:
    fragments = {item.source_fragment_id: item for item in case.fragments}
    parts: list[str] = []
    items: list[StructureReadReceiptItem] = []
    cursor = 0
    for order, member in enumerate(unit.members):
        fragment = fragments[member.source_fragment_id]
        if order:
            cursor += 2
        start = cursor
        cursor += len(fragment.text)
        parts.append(fragment.text)
        items.append(
            StructureReadReceiptItem(
                fragment.source_fragment_id,
                order,
                fragment.text_sha256,
                start,
                cursor,
            )
        )
    text = "\n\n".join(parts)
    receipt_payload = {
        "schema": "dithyramba.structure_read_receipt/1.0",
        "structure_unit_id": unit.structure_unit_id,
        "library_id": case.base_request.library_id,
        "corpus_snapshot_id": case.base_request.corpus_snapshot_id,
        "access_policy_id": case.base_request.access_policy_id,
        "policy_hash": case.compiled_access.token.policy_hash,
        "scope_hash": case.compiled_access.token.exclusion_hash,
        "permitted_set_hash": case.compiled_access.token.permitted_set_hash,
        "delimiter": "\n\n",
        "rendered_text_sha256": sha256_hex(text.encode("utf-8")),
        "items": [item.payload() for item in items],
    }
    receipt = StructureReadReceipt(
        receipt_id=canonical_content_id("structure_read", receipt_payload),
        structure_unit_id=unit.structure_unit_id,
        library_id=case.base_request.library_id,
        corpus_snapshot_id=case.base_request.corpus_snapshot_id,
        access_policy_id=case.base_request.access_policy_id,
        policy_hash=case.compiled_access.token.policy_hash,
        scope_hash=case.compiled_access.token.exclusion_hash,
        permitted_set_hash=case.compiled_access.token.permitted_set_hash,
        delimiter="\n\n",
        rendered_text_sha256=sha256_hex(text.encode("utf-8")),
        items=tuple(items),
    )
    return StructureUnitText(unit, text, receipt)


def _execute(
    case: ExpandedCase,
    provider: RecordingExpandedReranker,
    *,
    request: ExpandedQueryRequest | None = None,
    generation: StructureGeneration | None = None,
    reads: tuple[StructureUnitText, ...] = (),
) -> Any:
    return execute_expanded_retrieval(
        request=request or _expanded_request(case, generation),
        base_request=case.base_request,
        base_artifacts=case.base_artifacts,
        compiled_access=case.compiled_access,
        fragments=case.fragments,
        layer_profile=case.layer_profile,
        fusion_profile=case.fusion_profile,
        reranker_profile=case.reranker_profile,
        reranker_provider=provider,
        structure_generation=generation,
        structure_reads=reads,
    )


def _execute_with_overrides(
    case: ExpandedCase,
    provider: Any,
    **overrides: Any,
) -> Any:
    values: dict[str, Any] = {
        "request": _expanded_request(case),
        "base_request": case.base_request,
        "base_artifacts": case.base_artifacts,
        "compiled_access": case.compiled_access,
        "fragments": case.fragments,
        "layer_profile": case.layer_profile,
        "fusion_profile": case.fusion_profile,
        "reranker_profile": case.reranker_profile,
        "reranker_provider": provider,
        "structure_generation": None,
        "structure_reads": (),
    }
    values.update(overrides)
    return cast(Any, execute_expanded_retrieval)(**values)


def _artifact_bindings(result: ExpandedExecutionArtifacts) -> tuple[Any, ...]:
    return (
        result.request,
        result.base_request,
        result.base_artifacts,
        result.permitted_manifest,
        result.layer_profile,
        result.fusion_profile,
    )


def test_structure_off_uses_full_fused_seed_closure_and_one_final_rerank() -> None:
    case = _make_case(base_packet_limit=1)
    provider = RecordingExpandedReranker(
        case.reranker_profile,
        scores=(0.1, 0.4, 0.3, 0.2),
    )

    result = _execute(case, provider)

    assert len(case.base_artifacts.packet.items) == 1
    assert len(result.receipt.seed_ranks) == len(result.receipt.expansion_paths) == 4
    assert {item.candidate_kind for item in result.receipt.expansion_paths} == {
        ExpandedCandidateKind.SOURCE_FRAGMENT
    }
    assert len(provider.token_calls) == len(provider.score_calls) == 1
    assert provider.score_calls[0][1] == provider.token_calls[0][1]
    assert tuple(item.rank for item in result.receipt.reranked) == (1, 2, 3, 4)
    assert all("text" not in item.semantic_payload() for item in result.expansion_paths)


def test_returned_artifact_dataclass_enforces_exact_closure() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    bindings = _artifact_bindings(result)

    with pytest.raises(ExpandedExecutionError, match="expansion_paths"):
        ExpandedExecutionArtifacts(cast(Any, []), result.receipt, result.packet, *bindings)
    with pytest.raises(ExpandedExecutionError, match="receipt must"):
        ExpandedExecutionArtifacts(
            result.expansion_paths, cast(Any, object()), result.packet, *bindings
        )
    with pytest.raises(ExpandedExecutionError, match="packet must"):
        ExpandedExecutionArtifacts(
            result.expansion_paths, result.receipt, cast(Any, object()), *bindings
        )
    with pytest.raises(ExpandedExecutionError, match="paths differ"):
        ExpandedExecutionArtifacts((), result.receipt, result.packet, *bindings)
    wrong_packet = result.packet.model_copy(update={"expansion_receipt_hash": HASH_C})
    with pytest.raises(ExpandedExecutionError, match="packet/request/base/receipt"):
        ExpandedExecutionArtifacts(result.expansion_paths, result.receipt, wrong_packet, *bindings)


@pytest.mark.parametrize("mode", ["missing_selected", "changed_score"])
def test_artifact_wrapper_rejects_divergent_packet_with_correct_receipt_hash(
    mode: str,
) -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    if mode == "missing_selected":
        items = result.packet.items[:-1]
    else:
        first = result.packet.items[0].model_copy(update={"score_hex": (123.0).hex()})
        items = (first, *result.packet.items[1:])
    forged = result.packet.model_copy(update={"items": items})

    assert forged.expansion_receipt_hash == result.receipt.receipt_hash
    with pytest.raises(ExpandedExecutionError, match="exact capped reranked selection"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            result.receipt,
            forged,
            *_artifact_bindings(result),
        )


@pytest.mark.parametrize("mode", ["request", "base_packet"])
def test_artifact_wrapper_rejects_foreign_packet_bindings(mode: str) -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    update = (
        {"expanded_request_hash": HASH_C} if mode == "request" else {"base_packet_hash": HASH_C}
    )
    forged = result.packet.model_copy(update=update)

    assert forged.expansion_receipt_hash == result.receipt.receipt_hash
    with pytest.raises(ExpandedExecutionError, match="packet/request/base/receipt"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            result.receipt,
            forged,
            *_artifact_bindings(result),
        )


@pytest.mark.parametrize(
    "field",
    [
        "request",
        "base_request",
        "base_artifacts",
        "permitted_manifest",
        "layer_profile",
        "fusion_profile",
    ],
)
def test_artifact_wrapper_requires_exact_text_free_binding_types(field: str) -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    values: dict[str, Any] = {
        "expansion_paths": result.expansion_paths,
        "receipt": result.receipt,
        "packet": result.packet,
        "request": result.request,
        "base_request": result.base_request,
        "base_artifacts": result.base_artifacts,
        "permitted_manifest": result.permitted_manifest,
        "layer_profile": result.layer_profile,
        "fusion_profile": result.fusion_profile,
    }
    values[field] = object()
    with pytest.raises(ExpandedExecutionError):
        ExpandedExecutionArtifacts(**values)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("invalid_request", "invalid frozen contract"),
        ("noncanonical_receipt", "changed after contract validation"),
        ("inexact_base_packet", "inexact inner contract"),
        ("broken_base_wrapper", "invalid frozen contract"),
        ("foreign_receipt_base", "request/receipt/base artifact closure"),
    ],
)
def test_artifact_wrapper_revalidates_mutation_resistant_closure(
    mode: str,
    message: str,
) -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    if mode == "invalid_request":
        object.__setattr__(result.request, "max_final", 0)
    elif mode == "noncanonical_receipt":
        object.__setattr__(result.receipt, "seed_ranks", tuple(reversed(result.receipt.seed_ranks)))
    elif mode == "inexact_base_packet":
        object.__setattr__(result.base_artifacts, "packet", object())
    elif mode == "broken_base_wrapper":
        object.__setattr__(result.base_artifacts.packet, "read_receipt_hash", HASH_C)
    else:
        object.__setattr__(result.receipt, "base_request_hash", HASH_C)

    with pytest.raises(ExpandedExecutionError, match=message):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            result.receipt,
            result.packet,
            *_artifact_bindings(result),
        )


def test_artifact_wrapper_owns_independent_validated_nested_snapshots() -> None:
    case = _make_case()
    request = _expanded_request(case)
    result = _execute(
        case,
        RecordingExpandedReranker(case.reranker_profile),
        request=request,
    )

    assert result.request is not request
    assert result.base_request is not case.base_request
    assert result.base_artifacts is not case.base_artifacts
    assert result.base_artifacts.packet is not case.base_artifacts.packet
    assert result.base_artifacts.retrieval_receipt is not (case.base_artifacts.retrieval_receipt)
    assert result.permitted_manifest is not case.compiled_access.manifest
    assert result.permitted_manifest.items[0] is not case.compiled_access.manifest.items[0]
    assert result.layer_profile is not case.layer_profile
    assert result.fusion_profile is not case.fusion_profile

    copied = ExpandedExecutionArtifacts(
        result.expansion_paths,
        result.receipt,
        result.packet,
        *_artifact_bindings(result),
    )
    assert copied.receipt is not result.receipt
    assert copied.packet is not result.packet
    assert copied.expansion_paths[0] is not result.expansion_paths[0]
    assert copied.receipt.expansion_paths[0] is not result.receipt.expansion_paths[0]
    object.__setattr__(result.request, "max_final", 0)
    copied.validate_closure()
    assert copied.request.max_final > 0


def test_explicit_revalidation_rejects_deliberate_nested_mutation() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    object.__setattr__(result.packet, "base_packet_hash", HASH_C)

    with pytest.raises(ExpandedExecutionError, match="packet/request/base/receipt"):
        result.validate_closure()


def test_artifact_wrapper_rederives_corpus_hash_from_read_item_identities() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    original_read = result.base_artifacts.read_receipt
    changed_first = original_read.items[0].model_copy(update={"text_sha256": HASH_C})
    changed_read = original_read.model_copy(
        update={"items": (changed_first, *original_read.items[1:])}
    )
    changed_base_packet = result.base_artifacts.packet.model_copy(
        update={"read_receipt_hash": changed_read.receipt_hash}
    )
    changed_base = HybridExecutionArtifacts(
        changed_read,
        result.base_artifacts.access_receipt,
        result.base_artifacts.coverage_report,
        result.base_artifacts.retrieval_receipt,
        changed_base_packet,
    )

    with pytest.raises(ExpandedExecutionError, match="execution scope"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            result.receipt,
            result.packet,
            result.request,
            result.base_request,
            changed_base,
            result.permitted_manifest,
            result.layer_profile,
            result.fusion_profile,
        )


def test_artifact_wrapper_binds_atomic_path_text_hash_to_base_read_item() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    atomic = next(
        item
        for item in result.expansion_paths
        if item.candidate_kind is ExpandedCandidateKind.SOURCE_FRAGMENT
    )
    changed_path = atomic.model_copy(update={"text_sha256": HASH_C})
    changed_paths = tuple(
        sorted(
            (
                changed_path if item.candidate_key == atomic.candidate_key else item
                for item in result.expansion_paths
            ),
            key=lambda item: item.path_hash.encode("ascii"),
        )
    )

    def changed_trace_item(item: Any) -> Any:
        return (
            item.model_copy(update={"text_sha256": HASH_C})
            if item.candidate_key == atomic.candidate_key
            else item
        )

    receipt_values = result.receipt.model_dump(mode="python")
    receipt_values.update(
        {
            "eligible": tuple(changed_trace_item(item) for item in result.receipt.eligible),
            "reranked": tuple(changed_trace_item(item) for item in result.receipt.reranked),
            "expansion_paths": changed_paths,
            "expansion_path_hashes": tuple(item.path_hash for item in changed_paths),
        }
    )
    changed_receipt = type(result.receipt).model_validate(receipt_values)
    packet_values = result.packet.model_dump(mode="python")
    packet_values.update(
        {
            "expansion_receipt_hash": changed_receipt.receipt_hash,
            "items": tuple(changed_trace_item(item) for item in result.packet.items),
        }
    )
    changed_packet = type(result.packet).model_validate(packet_values)

    with pytest.raises(ExpandedExecutionError, match="atomic ExpansionPath text hash"):
        ExpandedExecutionArtifacts(
            changed_paths,
            changed_receipt,
            changed_packet,
            *_artifact_bindings(result),
        )


@pytest.mark.parametrize(
    "field",
    [
        "expansion_paths",
        "receipt",
        "packet",
        "request",
        "base_request",
        "base_artifacts",
        "permitted_manifest",
        "layer_profile",
        "fusion_profile",
    ],
)
def test_explicit_revalidation_rejects_replaced_top_level_contract(field: str) -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    replacement: Any = [] if field == "expansion_paths" else object()
    object.__setattr__(result, field, replacement)

    with pytest.raises(ExpandedExecutionError):
        result.validate_closure()


@pytest.mark.parametrize("model", ["read", "access", "coverage", "retrieval", "packet"])
def test_wrapper_reruns_every_base_model_validator(model: str) -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    base = result.base_artifacts
    if model == "read":
        object.__setattr__(base.read_receipt, "items", base.read_receipt.items[1:])
    elif model == "access":
        object.__setattr__(base.access_receipt, "corpus_snapshot_id", "snapshot_foreign")
    elif model == "coverage":
        object.__setattr__(base.coverage_report, "skipped_count", 1)
    elif model == "retrieval":
        object.__setattr__(
            base.retrieval_receipt,
            "fused",
            tuple(reversed(base.retrieval_receipt.fused)),
        )
    else:
        object.__setattr__(base.packet, "result_status", "no_evidence")

    with pytest.raises(ExpandedExecutionError, match="invalid frozen contract"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            result.receipt,
            result.packet,
            result.request,
            result.base_request,
            base,
            result.permitted_manifest,
            result.layer_profile,
            result.fusion_profile,
        )


def test_coherently_rebuilt_base_fused_subset_cannot_violate_frozen_rrf_budget() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    base = result.base_artifacts
    retrieval = base.retrieval_receipt.model_copy(
        update={"fused": base.retrieval_receipt.fused[:2]}
    )
    base_packet = base.packet.model_copy(update={"retrieval_receipt_hash": retrieval.receipt_hash})
    rebuilt_base = HybridExecutionArtifacts(
        base.read_receipt,
        base.access_receipt,
        base.coverage_report,
        retrieval,
        base_packet,
    )
    receipt = result.receipt.model_copy(
        update={"base_retrieval_receipt_hash": retrieval.receipt_hash}
    )
    packet = result.packet.model_copy(
        update={
            "base_packet_id": base_packet.packet_id,
            "base_packet_hash": base_packet.packet_hash,
            "expansion_receipt_hash": receipt.receipt_hash,
        }
    )

    assert len(receipt.seed_ranks) > len(retrieval.fused)
    with pytest.raises(ExpandedExecutionError, match="exact reciprocal-rank fusion"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            receipt,
            packet,
            result.request,
            result.base_request,
            rebuilt_base,
            result.permitted_manifest,
            result.layer_profile,
            result.fusion_profile,
        )


def test_seed_cannot_survive_coherent_removal_of_its_entire_projection() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    removed = result.receipt.eligible[-1]
    eligible = result.receipt.eligible[:-1]
    reranked_values = tuple(
        item for item in result.receipt.reranked if item.candidate_key != removed.candidate_key
    )
    reranked = tuple(
        item.model_copy(update={"rank": rank}) for rank, item in enumerate(reranked_values, start=1)
    )
    paths = tuple(
        item for item in result.expansion_paths if item.candidate_key != removed.candidate_key
    )
    receipt = result.receipt.model_copy(
        update={
            "eligible": eligible,
            "reranked": reranked,
            "expansion_paths": paths,
            "expansion_path_hashes": tuple(item.path_hash for item in paths),
        }
    )
    selected = expanded_engine._select_final(
        reranked,
        result.request.max_final,
        result.fusion_profile,
    )
    packet_items = tuple(
        ExpandedPacketItem(**(item.model_dump(exclude={"rank"}) | {"rank": rank}))
        for rank, item in enumerate(selected, start=1)
    )
    packet = result.packet.model_copy(
        update={
            "expansion_receipt_hash": receipt.receipt_hash,
            "items": packet_items,
            "result_status": "evidence_found" if packet_items else "no_evidence",
        }
    )

    assert any(
        item.source_fragment_id == removed.source_fragment_ids[0] for item in receipt.seed_ranks
    )
    with pytest.raises(ExpandedExecutionError, match="exactly one path or explicit omission"):
        ExpandedExecutionArtifacts(
            paths,
            receipt,
            packet,
            *_artifact_bindings(result),
        )


def test_foreign_extra_seed_and_pathless_omission_fail_projection_closure() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    extra_seed = ExpansionSeedRank(
        source_fragment_id="fragment_foreign",
        atomic_rank=len(result.receipt.seed_ranks) + 1,
    )
    receipt = result.receipt.model_copy(
        update={"seed_ranks": (*result.receipt.seed_ranks, extra_seed)}
    )
    packet = result.packet.model_copy(update={"expansion_receipt_hash": receipt.receipt_hash})
    with pytest.raises(ExpandedExecutionError, match="exact base fused trace"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            receipt,
            packet,
            *_artifact_bindings(result),
        )

    foreign_omission = ExpansionOmission(
        candidate_kind=ExpandedCandidateKind.SOURCE_FRAGMENT,
        candidate_id="fragment_foreign",
        reason=ExpansionOmissionReason.UNGROUNDED,
    )
    receipt = result.receipt.model_copy(
        update={"omissions": (*result.receipt.omissions, foreign_omission)}
    )
    packet = result.packet.model_copy(update={"expansion_receipt_hash": receipt.receipt_hash})
    with pytest.raises(ExpandedExecutionError, match="pathless omission"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            receipt,
            packet,
            *_artifact_bindings(result),
        )


def test_eligible_pre_score_is_rederived_from_exact_fused_anchors() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    changed = result.receipt.eligible[0].model_copy(update={"score_hex": (123.0).hex()})
    receipt = result.receipt.model_copy(
        update={"eligible": (changed, *result.receipt.eligible[1:])}
    )
    packet = result.packet.model_copy(update={"expansion_receipt_hash": receipt.receipt_hash})

    with pytest.raises(ExpandedExecutionError, match="eligible pre-score"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            receipt,
            packet,
            *_artifact_bindings(result),
        )


def test_reranked_order_is_rederived_from_scores_before_packet_projection() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    swapped_values = (
        result.receipt.reranked[1],
        result.receipt.reranked[0],
        *result.receipt.reranked[2:],
    )
    reranked = tuple(
        item.model_copy(update={"rank": rank}) for rank, item in enumerate(swapped_values, start=1)
    )
    receipt = result.receipt.model_copy(update={"reranked": reranked})
    selected = expanded_engine._select_final(
        reranked,
        result.request.max_final,
        result.fusion_profile,
    )
    packet_items = tuple(
        ExpandedPacketItem(**(item.model_dump(exclude={"rank"}) | {"rank": rank}))
        for rank, item in enumerate(selected, start=1)
    )
    packet = result.packet.model_copy(
        update={
            "expansion_receipt_hash": receipt.receipt_hash,
            "items": packet_items,
        }
    )

    with pytest.raises(ExpandedExecutionError, match="reranked order"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            receipt,
            packet,
            *_artifact_bindings(result),
        )


def test_coherently_rehashed_forged_base_fused_score_fails_exact_rrf() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    base = result.base_artifacts
    first = base.retrieval_receipt.fused[0].model_copy(update={"score_repr": "1/1"})
    retrieval = base.retrieval_receipt.model_copy(
        update={"fused": (first, *base.retrieval_receipt.fused[1:])}
    )
    base_packet = base.packet.model_copy(update={"retrieval_receipt_hash": retrieval.receipt_hash})
    rebuilt_base = HybridExecutionArtifacts(
        base.read_receipt,
        base.access_receipt,
        base.coverage_report,
        retrieval,
        base_packet,
    )
    receipt = result.receipt.model_copy(
        update={"base_retrieval_receipt_hash": retrieval.receipt_hash}
    )
    packet = result.packet.model_copy(
        update={
            "base_packet_id": base_packet.packet_id,
            "base_packet_hash": base_packet.packet_hash,
            "expansion_receipt_hash": receipt.receipt_hash,
        }
    )

    with pytest.raises(ExpandedExecutionError, match="exact reciprocal-rank fusion"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            receipt,
            packet,
            result.request,
            result.base_request,
            rebuilt_base,
            result.permitted_manifest,
            result.layer_profile,
            result.fusion_profile,
        )


def test_coherently_rehashed_forged_base_packet_fragment_fails_selection() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    base = result.base_artifacts
    forged_first = HybridPacketItem(
        rank=1,
        source_fragment_id="fragment_gamma",
        source_id="source_other",
        source_family_id="family_other",
        layer=CorpusLayer.SYNTHESIS,
    )
    base_packet = base.packet.model_copy(update={"items": (forged_first, *base.packet.items[1:])})
    rebuilt_base = HybridExecutionArtifacts(
        base.read_receipt,
        base.access_receipt,
        base.coverage_report,
        base.retrieval_receipt,
        base_packet,
    )
    packet = result.packet.model_copy(
        update={
            "base_packet_id": base_packet.packet_id,
            "base_packet_hash": base_packet.packet_hash,
        }
    )

    with pytest.raises(ExpandedExecutionError, match="base packet is not the exact"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            result.receipt,
            packet,
            result.request,
            result.base_request,
            rebuilt_base,
            result.permitted_manifest,
            result.layer_profile,
            result.fusion_profile,
        )


def test_coherently_rehashed_base_read_order_must_equal_permitted_manifest() -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    base = result.base_artifacts
    first, second, *rest = base.read_receipt.items
    changed_first = first.model_copy(
        update={
            "source_fragment_id": second.source_fragment_id,
            "source_version_id": second.source_version_id,
            "text_sha256": second.text_sha256,
        }
    )
    changed_second = second.model_copy(
        update={
            "source_fragment_id": first.source_fragment_id,
            "source_version_id": first.source_version_id,
            "text_sha256": first.text_sha256,
        }
    )
    read = base.read_receipt.model_copy(update={"items": (changed_first, changed_second, *rest)})
    base_packet = base.packet.model_copy(update={"read_receipt_hash": read.receipt_hash})
    rebuilt_base = HybridExecutionArtifacts(
        read,
        base.access_receipt,
        base.coverage_report,
        base.retrieval_receipt,
        base_packet,
    )
    packet = result.packet.model_copy(
        update={
            "base_packet_id": base_packet.packet_id,
            "base_packet_hash": base_packet.packet_hash,
        }
    )

    with pytest.raises(ExpandedExecutionError, match="permitted manifest order"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            result.receipt,
            packet,
            result.request,
            result.base_request,
            rebuilt_base,
            result.permitted_manifest,
            result.layer_profile,
            result.fusion_profile,
        )


@pytest.mark.parametrize("model", ["read", "access", "coverage", "retrieval", "packet"])
def test_every_coherently_rehashed_base_model_must_share_the_base_request(
    model: str,
) -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    base = result.base_artifacts
    read = base.read_receipt
    access = base.access_receipt
    coverage = base.coverage_report
    retrieval = base.retrieval_receipt
    base_packet = base.packet
    if model == "read":
        read = read.model_copy(update={"request_hash": HASH_C})
    elif model == "access":
        access = access.model_copy(update={"request_hash": HASH_C})
    elif model == "coverage":
        coverage = coverage.model_copy(update={"request_hash": HASH_C})
    elif model == "retrieval":
        retrieval = retrieval.model_copy(update={"request_hash": HASH_C})
    else:
        base_packet = base_packet.model_copy(update={"request_hash": HASH_C})
    base_packet = base_packet.model_copy(
        update={
            "read_receipt_hash": read.receipt_hash,
            "access_receipt_hash": access.receipt_hash,
            "coverage_report_hash": coverage.report_hash,
            "retrieval_receipt_hash": retrieval.receipt_hash,
        }
    )
    rebuilt_base = HybridExecutionArtifacts(
        read,
        access,
        coverage,
        retrieval,
        base_packet,
    )
    receipt = result.receipt.model_copy(
        update={"base_retrieval_receipt_hash": retrieval.receipt_hash}
    )
    packet = result.packet.model_copy(
        update={
            "base_packet_id": base_packet.packet_id,
            "base_packet_hash": base_packet.packet_hash,
            "expansion_receipt_hash": receipt.receipt_hash,
        }
    )

    with pytest.raises(ExpandedExecutionError, match="every base artifact"):
        ExpandedExecutionArtifacts(
            result.expansion_paths,
            receipt,
            packet,
            result.request,
            result.base_request,
            rebuilt_base,
            result.permitted_manifest,
            result.layer_profile,
            result.fusion_profile,
        )


@pytest.mark.parametrize(
    "mode",
    [
        "request_type",
        "base_type",
        "artifacts_type",
        "compiled_type",
        "base_binding",
        "profile_type",
        "profile_binding",
    ],
)
def test_public_input_types_and_hash_bindings_fail_closed(mode: str) -> None:
    case = _make_case()
    provider = RecordingExpandedReranker(case.reranker_profile)
    overrides: dict[str, Any] = {}
    if mode == "request_type":
        overrides["request"] = object()
    elif mode == "base_type":
        overrides["base_request"] = object()
    elif mode == "artifacts_type":
        overrides["base_artifacts"] = object()
    elif mode == "compiled_type":
        overrides["compiled_access"] = object()
    elif mode == "base_binding":
        overrides["request"] = ExpandedQueryRequest(
            base_request_id=case.base_request.request_id,
            base_request_hash=HASH_C,
            reranker_profile_hash=case.reranker_profile.profile_hash,
        )
    elif mode == "profile_type":
        overrides["reranker_profile"] = object()
    else:
        overrides["request"] = ExpandedQueryRequest(
            base_request_id=case.base_request.request_id,
            base_request_hash=case.base_request.request_hash,
            reranker_profile_hash=HASH_C,
        )

    with pytest.raises(ExpandedExecutionError):
        _execute_with_overrides(case, provider, **overrides)


@pytest.mark.parametrize(
    "mode",
    [
        "compiled_inner_type",
        "compiled_binding",
        "fragments_type",
        "fragment_set",
        "fragment_text",
        "artifact_inner_type",
        "read_receipt",
        "access_receipt",
        "coverage",
        "packet",
    ],
)
def test_authorized_base_artifact_mutations_fail_closed(mode: str) -> None:
    case = _make_case()
    overrides: dict[str, Any] = {}
    if mode == "compiled_inner_type":
        object.__setattr__(case.compiled_access, "manifest", object())
    elif mode == "compiled_binding":
        object.__setattr__(case.compiled_access.token, "exclusion_hash", HASH_C)
        object.__setattr__(
            case.compiled_access.token,
            "token_hash",
            canonical_sha256_hex(case.compiled_access.token.binding_payload()),
        )
    elif mode == "fragments_type":
        overrides["fragments"] = list(case.fragments)
    elif mode == "fragment_set":
        overrides["fragments"] = tuple(reversed(case.fragments))
    elif mode == "fragment_text":
        overrides["fragments"] = (
            replace(case.fragments[0], text="Changed but unhashed text"),
            *case.fragments[1:],
        )
    elif mode == "artifact_inner_type":
        object.__setattr__(case.base_artifacts, "read_receipt", object())
    elif mode == "read_receipt":
        object.__setattr__(case.base_artifacts.read_receipt, "retrieval_corpus_hash", HASH_C)
    elif mode == "access_receipt":
        object.__setattr__(case.base_artifacts.access_receipt, "policy_hash", HASH_C)
    elif mode == "coverage":
        object.__setattr__(case.base_artifacts.coverage_report, "processed_count", 0)
    else:
        object.__setattr__(case.base_artifacts.packet, "request_hash", HASH_C)

    with pytest.raises(ExpandedExecutionError):
        _execute_with_overrides(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            **overrides,
        )


def test_coherent_foreign_snapshot_id_cannot_reuse_another_snapshot_hash() -> None:
    case = _make_case()
    foreign_snapshot_id = f"snapshot_{HASH_C[:32]}"
    foreign_layer = case.layer_profile.model_copy(
        update={"corpus_snapshot_id": foreign_snapshot_id}
    )
    foreign_request = case.base_request.model_copy(
        update={
            "corpus_snapshot_id": foreign_snapshot_id,
            "layer_profile_hash": foreign_layer.profile_hash,
        }
    )
    request_id = foreign_request.request_id
    request_hash = foreign_request.request_hash
    read = case.base_artifacts.read_receipt.model_copy(
        update={"request_id": request_id, "request_hash": request_hash}
    )
    access = case.base_artifacts.access_receipt.model_copy(
        update={
            "request_id": request_id,
            "request_hash": request_hash,
            "corpus_snapshot_id": foreign_snapshot_id,
        }
    )
    coverage = case.base_artifacts.coverage_report.model_copy(
        update={"request_id": request_id, "request_hash": request_hash}
    )
    retrieval = case.base_artifacts.retrieval_receipt.model_copy(
        update={
            "request_id": request_id,
            "request_hash": request_hash,
            "layer_profile_hash": foreign_layer.profile_hash,
        }
    )
    packet = case.base_artifacts.packet.model_copy(
        update={
            "request_id": request_id,
            "request_hash": request_hash,
            "corpus_snapshot_id": foreign_snapshot_id,
            "read_receipt_hash": read.receipt_hash,
            "access_receipt_hash": access.receipt_hash,
            "coverage_report_hash": coverage.report_hash,
            "retrieval_receipt_hash": retrieval.receipt_hash,
        }
    )
    coherent_base = HybridExecutionArtifacts(read, access, coverage, retrieval, packet)
    expanded_request = ExpandedQueryRequest(
        base_request_id=request_id,
        base_request_hash=request_hash,
        reranker_profile_hash=case.reranker_profile.profile_hash,
    )

    with pytest.raises(ExpandedExecutionError, match="snapshot ID must be derived"):
        _execute_with_overrides(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            request=expanded_request,
            base_request=foreign_request,
            base_artifacts=coherent_base,
            layer_profile=foreign_layer,
        )


@pytest.mark.parametrize(
    "mode",
    [
        "layer_type",
        "fusion_type",
        "profile_binding",
        "source_closure",
        "retrieval_binding",
        "trace_budget",
        "trace_provenance",
        "fused_trace",
        "base_packet",
    ],
)
def test_base_profile_and_retrieval_mutations_fail_closed(mode: str) -> None:
    case = _make_case()
    overrides: dict[str, Any] = {}
    if mode == "layer_type":
        overrides["layer_profile"] = object()
    elif mode == "fusion_type":
        overrides["fusion_profile"] = object()
    elif mode == "profile_binding":
        object.__setattr__(case.layer_profile, "library_id", "library_other")
    elif mode == "source_closure":
        object.__setattr__(case.layer_profile, "items", case.layer_profile.items[:-1])
    elif mode == "retrieval_binding":
        object.__setattr__(case.base_artifacts.retrieval_receipt, "model_profile_hash", HASH_C)
    elif mode == "trace_budget":
        extra = HybridRankTraceItem(
            source_fragment_id=case.fragments[0].source_fragment_id,
            source_id=case.fragments[0].source_id,
            rank=5,
            score_repr="0.1",
        )
        object.__setattr__(
            case.base_artifacts.retrieval_receipt,
            "fts",
            (*case.base_artifacts.retrieval_receipt.fts, extra),
        )
    elif mode == "trace_provenance":
        outside = HybridRankTraceItem(
            source_fragment_id="fragment_outside",
            source_id="source_outside",
            rank=1,
            score_repr="1.0",
        )
        object.__setattr__(case.base_artifacts.retrieval_receipt, "fts", (outside,))
    elif mode == "fused_trace":
        object.__setattr__(
            case.base_artifacts.retrieval_receipt,
            "fused",
            tuple(reversed(case.base_artifacts.retrieval_receipt.fused)),
        )
    else:
        object.__setattr__(case.base_artifacts.packet, "items", ())
        object.__setattr__(case.base_artifacts.packet, "result_status", "no_evidence")

    if mode in {"retrieval_binding", "trace_budget", "trace_provenance", "fused_trace"}:
        object.__setattr__(
            case.base_artifacts.packet,
            "retrieval_receipt_hash",
            case.base_artifacts.retrieval_receipt.receipt_hash,
        )

    with pytest.raises(ExpandedExecutionError):
        _execute_with_overrides(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            **overrides,
        )


def test_structure_on_replaces_covered_atomics_with_complete_unit_without_label_text() -> None:
    case = _make_case(base_packet_limit=1, seed_suffixes=("alpha", "gamma"))
    generation, reads = _generation(
        case,
        ((StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), "SECRET LABEL"),),
    )
    provider = RecordingExpandedReranker(case.reranker_profile)

    result = _execute(case, provider, generation=generation, reads=reads)

    paths = {item.candidate_id: item for item in result.expansion_paths}
    assert generation.units[0].structure_unit_id in paths
    unit_path = paths[generation.units[0].structure_unit_id]
    assert unit_path.source_fragment_ids == ("fragment_alpha", "fragment_beta")
    assert unit_path.anchor_atomic_ranks == (1,)
    assert "fragment_alpha" not in paths and "fragment_beta" not in paths
    assert "fragment_beta" not in {item.source_fragment_id for item in result.receipt.seed_ranks}
    assert reads[0].text in provider.token_calls[0][1]
    assert all("SECRET LABEL" not in text for text in provider.token_calls[0][1])
    assert unit_path.structure_read_receipt_hash == reads[0].receipt.receipt_hash


@pytest.mark.parametrize("mode", ["missing", "extra", "wrong_receipt", "wrong_generation"])
def test_structure_closure_mismatch_fails_closed(mode: str) -> None:
    case = _make_case()
    generation, reads = _generation(
        case,
        ((StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), None),),
    )
    request = _expanded_request(case, generation)
    supplied_generation = generation
    supplied_reads = reads
    if mode == "missing":
        supplied_reads = ()
    elif mode == "extra":
        supplied_reads = reads + reads
    elif mode == "wrong_receipt":
        object.__setattr__(reads[0].receipt, "receipt_id", "structure_read_wrong")
    else:
        supplied_generation = replace(generation, generation_hash=HASH_C)

    with pytest.raises(ExpandedExecutionError):
        _execute(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            request=request,
            generation=supplied_generation,
            reads=supplied_reads,
        )


def test_structure_mode_boundary_and_read_integrity_fail_closed() -> None:
    case = _make_case()
    generation, reads = _generation(
        case,
        ((StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), None),),
    )
    provider = RecordingExpandedReranker(case.reranker_profile)
    with pytest.raises(ExpandedExecutionError, match="structure-off"):
        _execute_with_overrides(
            case,
            provider,
            structure_generation=generation,
            structure_reads=reads,
        )
    with pytest.raises(ExpandedExecutionError, match="requires exact generation"):
        _execute_with_overrides(
            case,
            provider,
            request=_expanded_request(case, generation),
        )
    with pytest.raises(ExpandedExecutionError, match="structure_reads"):
        _execute_with_overrides(
            case,
            provider,
            request=_expanded_request(case, generation),
            structure_generation=generation,
            structure_reads=list(reads),
        )

    changed_unit = replace(reads[0].structure_unit, label="Changed label")
    object.__setattr__(reads[0], "structure_unit", changed_unit)
    with pytest.raises(ExpandedExecutionError, match="differs from its generation"):
        _execute(
            case,
            provider,
            generation=generation,
            reads=reads,
        )


@pytest.mark.parametrize("mode", ["access_receipt", "content_hash", "rendered_text"])
def test_structure_receipt_policy_and_rendered_span_mutations_fail_closed(mode: str) -> None:
    case = _make_case()
    generation, reads = _generation(
        case,
        ((StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), None),),
    )
    if mode == "access_receipt":
        receipt = reads[0].receipt
        payload = receipt.payload() | {"policy_hash": HASH_C}
        changed = StructureReadReceipt(
            receipt_id=canonical_content_id("structure_read", payload),
            structure_unit_id=receipt.structure_unit_id,
            library_id=receipt.library_id,
            corpus_snapshot_id=receipt.corpus_snapshot_id,
            access_policy_id=receipt.access_policy_id,
            policy_hash=HASH_C,
            scope_hash=receipt.scope_hash,
            permitted_set_hash=receipt.permitted_set_hash,
            delimiter=receipt.delimiter,
            rendered_text_sha256=receipt.rendered_text_sha256,
            items=receipt.items,
        )
        object.__setattr__(reads[0], "receipt", changed)
    elif mode == "content_hash":
        object.__setattr__(reads[0].receipt, "policy_hash", HASH_C)
    else:
        object.__setattr__(reads[0], "text", "Text not represented by receipt spans")
    with pytest.raises(ExpandedExecutionError):
        _execute(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            generation=generation,
            reads=reads,
        )


def test_structure_unit_with_familyless_nonseed_fails_closed() -> None:
    case = _make_case(
        base_packet_limit=1,
        seed_suffixes=("alpha",),
        familyless_suffixes=("beta",),
    )
    generation, reads = _generation(
        case,
        ((StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), None),),
    )
    with pytest.raises(ExpandedExecutionError, match="SourceFamily"):
        _execute(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            generation=generation,
            reads=reads,
        )


def test_unanchored_foreign_structure_unit_fails_full_generation_closure() -> None:
    case = _make_case(seed_suffixes=("alpha",))
    generation, _reads = _generation(
        case,
        (
            (StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), None),
            (StructureUnitKind.PARAGRAPH, ("fragment_gamma",), None),
        ),
    )
    unanchored = next(
        item for item in generation.units if item.members[0].source_fragment_id == "fragment_gamma"
    )
    boundary_hash = canonical_sha256_hex(
        {
            "schema": "dithyramba.structure_boundary/1.0",
            "source_version_id": "source_version_foreign",
            "first_ordinal": unanchored.first_ordinal,
            "last_ordinal": unanchored.last_ordinal,
            "source_fragment_ids": [member.source_fragment_id for member in unanchored.members],
        }
    )
    semantic = {
        "schema": "dithyramba.structure_unit/1.0",
        "source_id": "source_foreign",
        "source_version_id": "source_version_foreign",
        "kind": unanchored.kind.value,
        "label": unanchored.label,
        "members": [member.payload() for member in unanchored.members],
        "boundary_hash": boundary_hash,
    }
    foreign_unit = StructureUnit(
        structure_unit_id=canonical_content_id("structure_unit", semantic),
        generation_id=generation.generation_id,
        source_id="source_foreign",
        source_version_id="source_version_foreign",
        kind=unanchored.kind,
        label=unanchored.label,
        members=unanchored.members,
        boundary_hash=boundary_hash,
        content_hash=canonical_sha256_hex(semantic),
    )
    coherent = _rebuild_generation(
        case,
        tuple(foreign_unit if item is unanchored else item for item in generation.units),
    )

    with pytest.raises(ExpandedExecutionError, match="changed or foreign"):
        _execute(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            request=_expanded_request(case, coherent),
            generation=coherent,
            reads=(),
        )


def test_mutated_unit_generation_id_fails_even_when_generation_hash_is_unchanged() -> None:
    case = _make_case(seed_suffixes=("alpha",))
    generation, reads = _generation(
        case,
        ((StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), None),),
    )
    object.__setattr__(
        generation.units[0],
        "generation_id",
        "structure_generation_foreign",
    )

    with pytest.raises(ExpandedExecutionError, match="generation ID differs"):
        _execute(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            request=_expanded_request(case, generation),
            generation=generation,
            reads=reads,
        )


def test_familyless_atomic_seed_fails_before_candidate_projection() -> None:
    case = _make_case(
        base_packet_limit=1,
        seed_suffixes=("alpha",),
        familyless_suffixes=("alpha",),
    )
    with pytest.raises(ExpandedExecutionError, match="SourceFamily"):
        _execute(case, RecordingExpandedReranker(case.reranker_profile))


def test_internal_structure_guards_reject_changed_unit_and_outside_member() -> None:
    case = _make_case()
    generation, reads = _generation(
        case,
        ((StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), None),),
    )
    unit = generation.units[0]
    with pytest.raises(ExpandedExecutionError, match="content identity"):
        expanded_engine._require_unit_content(replace(unit, content_hash=HASH_C))
    with pytest.raises(ExpandedExecutionError, match="outside authorized"):
        expanded_engine._require_structure_read(
            case.base_request,
            case.compiled_access,
            unit,
            reads[0],
            {},
            {},
        )


@pytest.mark.parametrize("mode", ["unit_container", "outside_member", "familyless"])
def test_full_generation_internal_closure_rejects_inexact_units(mode: str) -> None:
    case = _make_case(seed_suffixes=("alpha",))
    generation, _reads = _generation(
        case,
        ((StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), None),),
    )
    fragment_by_id = {item.source_fragment_id: item for item in case.fragments}
    manifest_by_id = {item.source_fragment_id: item for item in case.compiled_access.manifest.items}
    if mode == "unit_container":
        object.__setattr__(generation, "units", list(generation.units))
    elif mode == "outside_member":
        fragment_by_id = {}
        manifest_by_id = {}
    else:
        first_id = generation.units[0].members[0].source_fragment_id
        manifest_by_id[first_id] = replace(
            manifest_by_id[first_id],
            source_family_id=None,
        )

    with pytest.raises(ExpandedExecutionError):
        expanded_engine._require_generation_closure(
            _expanded_request(case, generation),
            case.base_request,
            case.compiled_access,
            generation,
            fragment_by_id,
            manifest_by_id,
        )


def test_overlapping_anchored_units_and_mixed_provenance_fail_closed() -> None:
    case = _make_case()
    overlap, overlap_reads = _generation(
        case,
        (
            (StructureUnitKind.SECTION, ("fragment_alpha", "fragment_beta"), None),
            (StructureUnitKind.BLOCK, ("fragment_beta", "fragment_epsilon"), None),
        ),
    )
    with pytest.raises(ExpandedExecutionError, match="overlap"):
        _execute(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            generation=overlap,
            reads=overlap_reads,
        )

    mixed_specs = (
        FragmentSpec("alpha", "document", CorpusLayer.PRIMARY, 0, "Alpha."),
        FragmentSpec("gamma", "other", CorpusLayer.SYNTHESIS, 1, "Gamma."),
    )
    mixed_case = _make_case(mixed_specs, base_packet_limit=2)
    mixed, mixed_reads = _generation(
        mixed_case,
        ((StructureUnitKind.SECTION, ("fragment_alpha", "fragment_gamma"), None),),
    )
    with pytest.raises(ExpandedExecutionError, match="changed or foreign"):
        _execute(
            mixed_case,
            RecordingExpandedReranker(mixed_case.reranker_profile),
            generation=mixed,
            reads=mixed_reads,
        )


def test_double_rerank_graph_and_policy_hash_mismatch_fail_closed() -> None:
    case = _make_case()
    reranked_base = case.base_request.model_copy(
        update={"reranker_profile_hash": case.reranker_profile.profile_hash}
    )
    with pytest.raises(ExpandedExecutionError, match="must not rerank"):
        execute_expanded_retrieval(
            request=ExpandedQueryRequest(
                base_request_id=reranked_base.request_id,
                base_request_hash=reranked_base.request_hash,
                reranker_profile_hash=case.reranker_profile.profile_hash,
            ),
            base_request=reranked_base,
            base_artifacts=case.base_artifacts,
            compiled_access=case.compiled_access,
            fragments=case.fragments,
            layer_profile=case.layer_profile,
            fusion_profile=case.fusion_profile,
            reranker_profile=case.reranker_profile,
            reranker_provider=RecordingExpandedReranker(case.reranker_profile),
        )
    graph_request = _expanded_request(
        case,
        graph=GraphExpansionConfig(
            relation_types=(RelationType.SUPPORTS,),
            max_hops=1,
        ),
    )
    with pytest.raises(ExpandedExecutionError, match="unsupported"):
        _execute(
            case,
            RecordingExpandedReranker(case.reranker_profile),
            request=graph_request,
        )

    object.__setattr__(case.compiled_access.token, "policy_hash", HASH_C)
    with pytest.raises(ExpandedExecutionError, match="content hash"):
        _execute(case, RecordingExpandedReranker(case.reranker_profile))


def test_one_long_candidate_is_omitted_but_fit_candidates_are_scored_once() -> None:
    case = _make_case(final_token_limit=12)
    provider = RecordingExpandedReranker(
        case.reranker_profile,
        counts=(10, 99, 11, 12),
        scores=(0.1, 0.9, 0.3),
    )

    result = _execute(case, provider)

    assert len(result.receipt.eligible) == len(result.receipt.reranked) == 3
    assert len(result.receipt.omissions) == 1
    omission = result.receipt.omissions[0]
    assert omission.reason is ExpansionOmissionReason.TOKEN_BUDGET_EXCEEDED
    assert omission.observed_token_count == 99
    assert omission.expansion_path_hash in result.receipt.expansion_path_hashes
    assert provider.score_calls[0][1] == tuple(
        text for index, text in enumerate(provider.token_calls[0][1]) if index != 1
    )
    assert len(provider.score_calls) == 1


@pytest.mark.parametrize(
    ("counts", "scores"),
    [
        ((10,), None),
        (None, (1.0,)),
        (None, (1.0, 2.0, float("nan"), 4.0)),
        (None, (1.0, 2.0, float("inf"), 4.0)),
    ],
)
def test_provider_cardinality_and_nonfinite_outputs_fail_closed(
    counts: tuple[int, ...] | None,
    scores: tuple[float, ...] | None,
) -> None:
    case = _make_case()
    with pytest.raises(ExpandedExecutionError):
        _execute(
            case,
            RecordingExpandedReranker(case.reranker_profile, counts=counts, scores=scores),
        )


def test_provider_profile_mutation_fails_closed() -> None:
    case = _make_case()

    class MutatingProvider(RecordingExpandedReranker):
        def pair_token_lengths(
            self, question: str, candidate_texts: tuple[str, ...]
        ) -> tuple[int, ...]:
            result = super().pair_token_lengths(question, candidate_texts)
            self.reranker_profile = _profile(max_tokens=511)
            return result

    with pytest.raises(ExpandedExecutionError, match="profile changed"):
        _execute(case, MutatingProvider(case.reranker_profile))


@pytest.mark.parametrize("mode", ["missing_methods", "profile_error", "token_error", "score_error"])
def test_provider_interface_and_runtime_errors_fail_closed(mode: str) -> None:
    case = _make_case()

    class ProfileError:
        @property
        def reranker_profile(self) -> RerankerProfile:
            raise RuntimeError("profile unavailable")

        def pair_token_lengths(self, question: str, texts: tuple[str, ...]) -> tuple[int, ...]:
            raise AssertionError((question, texts))

        def score_text_pairs(self, question: str, texts: tuple[str, ...]) -> tuple[float, ...]:
            raise AssertionError((question, texts))

    class RuntimeErrorProvider(RecordingExpandedReranker):
        def pair_token_lengths(
            self, question: str, candidate_texts: tuple[str, ...]
        ) -> tuple[int, ...]:
            if mode == "token_error":
                raise RuntimeError("tokenizer failed")
            return super().pair_token_lengths(question, candidate_texts)

        def score_text_pairs(
            self, question: str, candidate_texts: tuple[str, ...]
        ) -> tuple[float, ...]:
            if mode == "score_error":
                raise RuntimeError("model failed")
            return super().score_text_pairs(question, candidate_texts)

    if mode == "missing_methods":
        provider: Any = object()
    elif mode == "profile_error":
        provider = ProfileError()
    else:
        provider = RuntimeErrorProvider(case.reranker_profile)
    with pytest.raises(ExpandedExecutionError):
        _execute_with_overrides(case, provider)


@pytest.mark.parametrize(
    "counts",
    [
        (0, 10, 10, 10),
        (-1, 10, 10, 10),
        (True, 10, 10, 10),
        (1_000_001, 10, 10, 10),
    ],
)
def test_invalid_token_count_values_fail_closed(counts: tuple[Any, ...]) -> None:
    case = _make_case()
    provider = RecordingExpandedReranker(case.reranker_profile)
    provider.counts = cast(tuple[int, ...], counts)
    with pytest.raises(ExpandedExecutionError, match="token-count"):
        _execute(case, provider)


def test_non_float_reranker_scores_fail_closed() -> None:
    case = _make_case()
    provider = RecordingExpandedReranker(case.reranker_profile)
    provider.scores = cast(tuple[float, ...], (1, 2.0, 3.0, 4.0))
    with pytest.raises(ExpandedExecutionError, match="cardinality or emitted"):
        _execute(case, provider)


def test_same_derived_source_top_five_cap_is_applied_before_final_limit() -> None:
    specs = tuple(
        FragmentSpec(
            f"item{index}",
            "derived",
            CorpusLayer.DERIVED,
            index,
            f"Derived evidence {index}.",
        )
        for index in range(6)
    )
    case = _make_case(specs, base_packet_limit=2)
    result = _execute(
        case,
        RecordingExpandedReranker(case.reranker_profile),
        request=_expanded_request(case, max_final=6),
    )

    assert len(result.receipt.reranked) == 6
    assert len(result.packet.items) == case.fusion_profile.max_per_derived_source_top_five


def test_distinct_derived_sources_in_one_family_share_the_top_five_cap() -> None:
    specs = tuple(
        FragmentSpec(
            f"item{index}",
            f"derived{index}",
            CorpusLayer.DERIVED,
            index,
            f"Derived evidence {index}.",
            "shared" if index < 3 else f"independent{index}",
        )
        for index in range(6)
    )
    case = _make_case(specs, base_packet_limit=2)
    result = _execute(
        case,
        RecordingExpandedReranker(case.reranker_profile),
        request=_expanded_request(case, max_final=6),
    )

    assert len({item.source_ids[0] for item in result.packet.items}) == 6
    assert len(result.packet.items) == 6
    top_five_shared = sum(
        item.source_family_ids == ("family_shared",) for item in result.packet.items[:5]
    )
    assert top_five_shared == case.fusion_profile.max_per_derived_source_top_five
    assert result.packet.items[5].source_family_ids == ("family_shared",)


@pytest.mark.parametrize("mode", ["trace_type", "limit", "profile", "mixed_layer"])
def test_family_aware_final_selector_fails_closed_on_ambiguous_inputs(mode: str) -> None:
    case = _make_case()
    result = _execute(case, RecordingExpandedReranker(case.reranker_profile))
    reranked: Any = result.receipt.reranked
    limit: Any = 5
    profile: Any = case.fusion_profile
    if mode == "trace_type":
        reranked = list(reranked)
    elif mode == "limit":
        limit = 0
    elif mode == "profile":
        profile = object()
    else:
        mixed = reranked[0].model_copy(
            update={"corpus_layers": (CorpusLayer.DERIVED, CorpusLayer.PRIMARY)}
        )
        reranked = (mixed,)

    with pytest.raises(ExpandedExecutionError):
        expanded_engine._select_final(reranked, limit, profile)


def test_all_omitted_empty_and_repeated_runs_are_honest_and_deterministic() -> None:
    case = _make_case(final_token_limit=1)
    provider = RecordingExpandedReranker(case.reranker_profile, counts=(2, 3, 4, 5))
    first = _execute(case, provider)
    second = _execute(
        case,
        RecordingExpandedReranker(case.reranker_profile, counts=(2, 3, 4, 5)),
    )

    assert first.packet.result_status == "no_evidence"
    assert not first.packet.items and not first.receipt.reranked
    assert len(first.receipt.omissions) == 4
    assert not provider.score_calls
    assert first.receipt.receipt_hash == second.receipt.receipt_hash
    assert first.packet.packet_hash == second.packet.packet_hash

    empty = _make_case((), base_packet_limit=1)
    empty_provider = RecordingExpandedReranker(empty.reranker_profile)
    empty_result = _execute(empty, empty_provider)
    assert empty_result.packet.result_status == "no_evidence"
    assert not empty_result.receipt.seed_ranks
    assert not empty_provider.token_calls and not empty_provider.score_calls


def test_internal_profile_source_closure_and_score_parsing_fail_closed() -> None:
    case = _make_case()
    object.__setattr__(case.layer_profile, "items", case.layer_profile.items[:-1])
    object.__setattr__(
        case.base_request,
        "layer_profile_hash",
        case.layer_profile.profile_hash,
    )
    with pytest.raises(ExpandedExecutionError, match="exact source closure"):
        expanded_engine._require_profile_closure(
            case.base_request,
            case.base_artifacts,
            case.compiled_access,
            case.layer_profile,
            case.fusion_profile,
        )

    for score in ("not-a-fraction", "0/1", "2/4"):
        with pytest.raises(ExpandedExecutionError):
            expanded_engine._exact_fused_score(score)
    with pytest.raises(ExpandedExecutionError, match="finite float"):
        expanded_engine._fraction_float_hex(Fraction(10**10_000, 1))
