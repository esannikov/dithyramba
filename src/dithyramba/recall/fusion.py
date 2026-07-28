"""Exact reciprocal-rank fusion and source-layer-aware selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from .hybrid_models import CorpusLayer, FusionProfile, HybridContractError

_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class FusionContractError(HybridContractError):
    """One channel ranking or fused selection violates the P7 contract."""


@dataclass(frozen=True, slots=True)
class FusionCandidate:
    source_fragment_id: str
    source_id: str
    source_family_id: str
    layer: CorpusLayer

    def __post_init__(self) -> None:
        _require_id(self.source_fragment_id, "fragment")
        _require_id(self.source_id, "source")
        _require_id(self.source_family_id, "family")
        if not isinstance(self.layer, CorpusLayer):
            raise FusionContractError("candidate layer must be CorpusLayer")


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    candidate: FusionCandidate
    fts_rank: int | None
    dense_rank: int | None
    score_numerator: int
    score_denominator: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, FusionCandidate):
            raise FusionContractError("fused candidate metadata is invalid")
        if self.fts_rank is None and self.dense_rank is None:
            raise FusionContractError("fused candidate requires at least one channel rank")
        for rank in (self.fts_rank, self.dense_rank):
            if rank is not None and (type(rank) is not int or not 1 <= rank <= 500):
                raise FusionContractError("channel rank must be between 1 and 500")
        if (
            type(self.score_numerator) is not int
            or type(self.score_denominator) is not int
            or self.score_numerator <= 0
            or self.score_denominator <= 0
        ):
            raise FusionContractError("fused score must be a positive exact fraction")

    @property
    def exact_score(self) -> Fraction:
        return Fraction(self.score_numerator, self.score_denominator)

    @property
    def score_repr(self) -> str:
        return f"{self.score_numerator}/{self.score_denominator}"

    def payload(self, *, rank: int) -> dict[str, object]:
        if type(rank) is not int or rank < 1:
            raise FusionContractError("fused output rank must be a positive integer")
        return {
            "rank": rank,
            "source_fragment_id": self.candidate.source_fragment_id,
            "source_id": self.candidate.source_id,
            "source_family_id": self.candidate.source_family_id,
            "layer": self.candidate.layer.value,
            "fts_rank": self.fts_rank,
            "dense_rank": self.dense_rank,
            "score_numerator": self.score_numerator,
            "score_denominator": self.score_denominator,
        }


def reciprocal_rank_fusion(
    fts: tuple[FusionCandidate, ...],
    dense: tuple[FusionCandidate, ...],
    *,
    profile: FusionProfile,
    limit: int,
) -> tuple[FusedCandidate, ...]:
    """Fuse independent ranked lists without normalizing incomparable raw scores."""

    if not isinstance(profile, FusionProfile):
        raise FusionContractError("profile must be an exact FusionProfile")
    if type(limit) is not int or not 1 <= limit <= 500:
        raise FusionContractError("fusion limit must be between 1 and 500")
    fts_by_id = _ranked_channel(fts, "FTS")
    dense_by_id = _ranked_channel(dense, "dense")
    all_ids = set(fts_by_id) | set(dense_by_id)
    fused: list[FusedCandidate] = []
    for fragment_id in all_ids:
        fts_item = fts_by_id.get(fragment_id)
        dense_item = dense_by_id.get(fragment_id)
        if fts_item is not None:
            candidate = fts_item[1]
        elif dense_item is not None:
            candidate = dense_item[1]
        else:  # pragma: no cover - all_ids is the exact union above.
            raise FusionContractError("fused candidate disappeared from both channels")
        if fts_item is not None and dense_item is not None and fts_item[1] != dense_item[1]:
            raise FusionContractError("channels disagree on candidate provenance metadata")
        fts_rank = None if fts_item is None else fts_item[0]
        dense_rank = None if dense_item is None else dense_item[0]
        score = Fraction(0, 1)
        if fts_rank is not None:
            score += Fraction(profile.fts_weight, profile.rrf_k + fts_rank)
        if dense_rank is not None:
            score += Fraction(profile.dense_weight, profile.rrf_k + dense_rank)
        fused.append(
            FusedCandidate(
                candidate=candidate,
                fts_rank=fts_rank,
                dense_rank=dense_rank,
                score_numerator=score.numerator,
                score_denominator=score.denominator,
            )
        )
    return tuple(
        sorted(
            fused,
            key=lambda item: (
                -item.exact_score,
                item.candidate.source_fragment_id.encode("ascii"),
            ),
        )[:limit]
    )


def select_layer_aware(
    candidates: tuple[FusedCandidate, ...],
    *,
    limit: int,
    profile: FusionProfile,
) -> tuple[FusedCandidate, ...]:
    """Enforce the preregistered derived-Source cap without relabeling evidence."""

    if any(not isinstance(item, FusedCandidate) for item in candidates):
        raise FusionContractError("layer-aware candidates must be FusedCandidate values")
    selected_ids = select_layer_aware_fragment_ids(
        tuple(
            (
                item.candidate.source_fragment_id,
                item.candidate.source_id,
                item.candidate.layer,
            )
            for item in candidates
        ),
        limit=limit,
        profile=profile,
    )
    by_id = {item.candidate.source_fragment_id: item for item in candidates}
    return tuple(by_id[fragment_id] for fragment_id in selected_ids)


def select_layer_aware_fragment_ids(
    candidates: tuple[tuple[str, str, CorpusLayer], ...],
    *,
    limit: int,
    profile: FusionProfile,
) -> tuple[str, ...]:
    """Return the exact selected IDs for execution and persistence validation."""

    if not isinstance(profile, FusionProfile):
        raise FusionContractError("profile must be an exact FusionProfile")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise FusionContractError("selection limit must be between 1 and 100")
    normalized: list[tuple[str, str, CorpusLayer]] = []
    for item in candidates:
        if type(item) is not tuple or len(item) != 3:
            raise FusionContractError("layer-aware candidate keys are invalid")
        fragment_id, source_id, layer = item
        _require_id(fragment_id, "fragment")
        _require_id(source_id, "source")
        if not isinstance(layer, CorpusLayer):
            raise FusionContractError("layer-aware candidate layer is invalid")
        normalized.append((fragment_id, source_id, layer))
    ids = tuple(item[0] for item in normalized)
    if len(set(ids)) != len(ids):
        raise FusionContractError("layer-aware candidates must be unique")
    selected: list[str] = []
    selected_ids: set[str] = set()
    derived_top_five: dict[str, int] = {}
    top_five_target = min(5, limit)
    for fragment_id, source_id, layer in normalized:
        if len(selected) >= top_five_target:
            break
        if layer is CorpusLayer.DERIVED:
            count = derived_top_five.get(source_id, 0)
            if count >= profile.max_per_derived_source_top_five:
                continue
            derived_top_five[source_id] = count + 1
        selected.append(fragment_id)
        selected_ids.add(fragment_id)
    if len(selected) < top_five_target:
        return tuple(selected)
    for fragment_id, _source_id, _layer in normalized:
        if len(selected) >= limit:
            break
        if fragment_id not in selected_ids:
            selected.append(fragment_id)
            selected_ids.add(fragment_id)
    return tuple(selected)


def _ranked_channel(
    candidates: tuple[FusionCandidate, ...],
    label: str,
) -> dict[str, tuple[int, FusionCandidate]]:
    if len(candidates) > 500:
        raise FusionContractError(f"{label} channel exceeds 500 candidates")
    result: dict[str, tuple[int, FusionCandidate]] = {}
    for rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, FusionCandidate):
            raise FusionContractError(f"{label} channel contains an invalid candidate")
        if candidate.source_fragment_id in result:
            raise FusionContractError(f"{label} channel contains duplicate fragment IDs")
        result[candidate.source_fragment_id] = (rank, candidate)
    return result


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise FusionContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise FusionContractError(f"identifier must use a canonical {expected} suffix")
    return value
