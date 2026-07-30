"""Deterministic, zero-generative-token AreaMap construction."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import Protocol, cast

from dithyramba.contracts import canonical_sha256_hex
from dithyramba.recall.vector import PackedVector, pack_normalized_vector

from .errors import CartographyError
from .models import (
    AreaMap,
    AreaMember,
    AreaProjection,
    CartographyConfig,
    CartographyFragment,
    CartographyResult,
    InquirySignal,
    InquirySignalKind,
)


class _LabelArray(Protocol):
    def tolist(self) -> list[int]: ...


class _ProbabilityArray(Protocol):
    def __getitem__(self, index: int) -> float: ...


class _Clusterer(Protocol):
    probabilities_: _ProbabilityArray

    def fit_predict(self, matrix: object) -> _LabelArray: ...


class _HdbscanFactory(Protocol):
    def __call__(
        self,
        *,
        min_cluster_size: int,
        min_samples: int,
        metric: str,
        cluster_selection_method: str,
        allow_single_cluster: bool,
        copy: bool,
    ) -> _Clusterer: ...


class _NumpyModule(Protocol):
    float32: object

    def asarray(self, values: object, *, dtype: object) -> object: ...


def build_cartography(
    *,
    library_id: str,
    corpus_snapshot_id: str,
    access_policy_id: str,
    permitted_set_hash: str,
    vector_generation_hash: str,
    fragments: Sequence[CartographyFragment],
    config: CartographyConfig | None = None,
) -> CartographyResult:
    """Cluster one bounded exact vector view and emit text-free inquiry signals."""

    resolved_config = CartographyConfig() if config is None else config
    ordered = _validate_inputs(fragments, config=resolved_config)
    np, hdbscan, runtime_hash = _load_runtime()
    matrix = np.asarray([item.vector.values for item in ordered], dtype=np.float32)
    clusterer = hdbscan(
        min_cluster_size=resolved_config.min_cluster_size,
        min_samples=resolved_config.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        copy=True,
    )
    labels = clusterer.fit_predict(matrix)
    probabilities = clusterer.probabilities_

    grouped: dict[int, list[int]] = {}
    unassigned: list[str] = []
    for index, raw_label in enumerate(labels.tolist()):
        label = int(raw_label)
        if label < 0:
            unassigned.append(ordered[index].source_fragment_id)
        else:
            grouped.setdefault(label, []).append(index)

    areas_with_centroids: list[tuple[AreaProjection, tuple[float, ...]]] = []
    for indexes in grouped.values():
        member_fragments = tuple(ordered[index] for index in indexes)
        centroid = _centroid(tuple(item.vector for item in member_fragments))
        representatives = _representatives(
            member_fragments,
            centroid,
            maximum=resolved_config.representative_count,
        )
        members = tuple(
            sorted(
                (
                    AreaMember(
                        source_fragment_id=ordered[index].source_fragment_id,
                        source_id=ordered[index].source_id,
                        source_version_id=ordered[index].source_version_id,
                        text_sha256=ordered[index].text_sha256,
                        address_hash=ordered[index].address_hash,
                        vector_sha256=ordered[index].vector.vector_hash,
                        membership_hex=float(probabilities[index]).hex(),
                    )
                    for index in indexes
                ),
                key=lambda item: item.source_fragment_id,
            )
        )
        boundary_hash = canonical_sha256_hex(
            {
                "schema": "dithyramba.area_boundary/1.0",
                "cartography_profile_hash": resolved_config.profile_hash,
                "members": [
                    {
                        "source_fragment_id": item.source_fragment_id,
                        "text_sha256": item.text_sha256,
                        "address_hash": item.address_hash,
                        "vector_sha256": item.vector_sha256,
                    }
                    for item in members
                ],
            }
        )
        area = AreaProjection(
            area_id=f"area_{boundary_hash[:32]}",
            boundary_hash=boundary_hash,
            centroid_vector_sha256=centroid.vector_hash,
            members=members,
            representative_fragment_ids=tuple(item.source_fragment_id for item in representatives),
        )
        areas_with_centroids.append((area, centroid.values))

    areas_with_centroids.sort(key=lambda item: item[0].area_id)
    areas = tuple(item[0] for item in areas_with_centroids)
    input_manifest_hash = canonical_sha256_hex(
        {
            "schema": "dithyramba.cartography_input_manifest/1.0",
            "items": [item.text_free_payload() for item in ordered],
        }
    )
    area_map = AreaMap(
        library_id=library_id,
        corpus_snapshot_id=corpus_snapshot_id,
        access_policy_id=access_policy_id,
        permitted_set_hash=permitted_set_hash,
        vector_generation_hash=vector_generation_hash,
        cartography_profile_hash=resolved_config.profile_hash,
        cartography_runtime_hash=runtime_hash,
        input_manifest_hash=input_manifest_hash,
        input_fragment_count=len(ordered),
        areas=areas,
        unassigned_fragment_ids=tuple(sorted(unassigned)),
    )
    signals = _signals(
        area_map,
        centroids={area.area_id: centroid for area, centroid in areas_with_centroids},
        config=resolved_config,
    )
    return CartographyResult(area_map=area_map, signals=signals)


def _validate_inputs(
    fragments: Sequence[CartographyFragment],
    *,
    config: CartographyConfig,
) -> tuple[CartographyFragment, ...]:
    received = tuple(fragments)
    if any(type(item) is not CartographyFragment for item in received):
        raise CartographyError("cartography inputs must contain CartographyFragment values")
    ordered = tuple(sorted(received, key=lambda item: item.source_fragment_id))
    if not config.min_cluster_size <= len(ordered) <= config.max_units:
        raise CartographyError("cartography fragment count is outside the configured bound")
    identifiers = tuple(item.source_fragment_id for item in ordered)
    if len(set(identifiers)) != len(identifiers):
        raise CartographyError("cartography fragment IDs must be unique")
    dimensions = {item.vector.dimensions for item in ordered}
    if len(dimensions) != 1:
        raise CartographyError("cartography vectors must use one exact dimension")
    return ordered


def _load_runtime() -> tuple[_NumpyModule, _HdbscanFactory, str]:
    try:
        np = importlib.import_module("numpy")
        sklearn = importlib.import_module("sklearn")
        cluster = importlib.import_module("sklearn.cluster")
        hdbscan = cluster.HDBSCAN
        numpy_version = str(np.__version__)
        sklearn_version = str(sklearn.__version__)
    except (AttributeError, ImportError) as error:
        raise CartographyError("cartography requires the optional 'cartography' runtime") from error
    runtime_hash = canonical_sha256_hex(
        {
            "schema": "dithyramba.cartography_runtime/1.0",
            "numpy": numpy_version,
            "scikit_learn": sklearn_version,
        }
    )
    return cast("_NumpyModule", np), cast("_HdbscanFactory", hdbscan), runtime_hash


def _centroid(vectors: tuple[PackedVector, ...]) -> PackedVector:
    dimensions = vectors[0].dimensions
    totals = [0.0] * dimensions
    for vector in vectors:
        for index, value in enumerate(vector.values):
            totals[index] += value
    norm = math.sqrt(sum(value * value for value in totals))
    if norm == 0.0:
        raise CartographyError("area centroid is undefined for cancelling vectors")
    return pack_normalized_vector(tuple(float(value / norm) for value in totals))


def _representatives(
    fragments: tuple[CartographyFragment, ...],
    centroid: PackedVector,
    *,
    maximum: int,
) -> tuple[CartographyFragment, ...]:
    centroid_values = centroid.values
    ranked = sorted(
        fragments,
        key=lambda item: (
            -sum(
                left * right
                for left, right in zip(item.vector.values, centroid_values, strict=True)
            ),
            item.source_fragment_id,
        ),
    )
    return tuple(ranked[: min(maximum, len(ranked))])


def _signals(
    area_map: AreaMap,
    *,
    centroids: dict[str, tuple[float, ...]],
    config: CartographyConfig,
) -> tuple[InquirySignal, ...]:
    signals: list[InquirySignal] = []
    areas_by_id = {area.area_id: area for area in area_map.areas}
    bridge_threshold = float.fromhex(config.bridge_similarity_hex)
    area_ids = tuple(sorted(centroids))
    for left_index, left_id in enumerate(area_ids):
        for right_id in area_ids[left_index + 1 :]:
            similarity = sum(
                left * right
                for left, right in zip(centroids[left_id], centroids[right_id], strict=True)
            )
            similarity = _stable_unit_score(similarity)
            if similarity < bridge_threshold:
                continue
            trigger_ids = tuple(
                sorted(
                    set(areas_by_id[left_id].representative_fragment_ids[:2]).union(
                        areas_by_id[right_id].representative_fragment_ids[:2]
                    )
                )
            )
            signals.append(
                _signal(
                    InquirySignalKind.AREA_BRIDGE,
                    area_ids=(left_id, right_id),
                    trigger_fragment_ids=trigger_ids,
                    score=similarity,
                )
            )
    for area in area_map.areas:
        source_count = len({member.source_id for member in area.members})
        if source_count >= 2 or len(area.members) < config.min_cluster_size:
            continue
        concentration = 1.0 - (source_count / len(area.members))
        signals.append(
            _signal(
                InquirySignalKind.SOURCE_CONCENTRATION,
                area_ids=(area.area_id,),
                trigger_fragment_ids=tuple(sorted(area.representative_fragment_ids)),
                score=concentration,
            )
        )
    total = sum(len(area.members) for area in area_map.areas) + len(
        area_map.unassigned_fragment_ids
    )
    noise_ratio = len(area_map.unassigned_fragment_ids) / total
    if noise_ratio >= float.fromhex(config.noise_signal_ratio_hex):
        signals.append(
            _signal(
                InquirySignalKind.UNMAPPED_MASS,
                area_ids=(),
                trigger_fragment_ids=area_map.unassigned_fragment_ids[:32],
                score=noise_ratio,
            )
        )
    return tuple(sorted(signals, key=lambda item: item.signal_id))


def _stable_unit_score(value: float) -> float:
    """Quantize a derived score before thresholding and content addressing."""

    return round(min(1.0, max(0.0, float(value))), 10)


def _signal(
    kind: InquirySignalKind,
    *,
    area_ids: tuple[str, ...],
    trigger_fragment_ids: tuple[str, ...],
    score: float,
) -> InquirySignal:
    payload = {
        "schema": "dithyramba.inquiry_signal/1.0",
        "kind": kind.value,
        "area_ids": list(area_ids),
        "trigger_fragment_ids": list(trigger_fragment_ids),
        "score_hex": float(score).hex(),
    }
    digest = canonical_sha256_hex(payload)
    return InquirySignal(
        signal_id=f"inquiry_signal_{digest[:32]}",
        kind=kind,
        area_ids=area_ids,
        trigger_fragment_ids=trigger_fragment_ids,
        score_hex=float(score).hex(),
    )
