"""Minimal contracts for rebuildable areas, inquiries, and bounded traces.

Cartography is a derived view over authoritative source memory.  These
contracts deliberately keep source text out of durable payloads and never
promote an embedding cluster into a fact.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from dithyramba.contracts import canonical_content_id, canonical_sha256_hex, sha256_hex
from dithyramba.recall.vector import PackedVector

from .errors import CartographyError

_HASH = re.compile(r"^[0-9a-f]{64}$")


class InquirySignalKind(StrEnum):
    """Observable reason to formulate a question; never a question itself."""

    AREA_BRIDGE = "area_bridge"
    SOURCE_CONCENTRATION = "source_concentration"
    UNMAPPED_MASS = "unmapped_mass"


class InquiryState(StrEnum):
    CANDIDATE = "candidate"
    ANSWERABLE = "answerable"
    PARTIAL = "partial"
    CONTESTED = "contested"
    OPEN_GAP = "open_gap"


class ReviewState(StrEnum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REVISE = "revise"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class CartographyFragment:
    """Ephemeral exact-text input used only while building a projection."""

    source_fragment_id: str
    source_id: str
    source_version_id: str
    text: str = field(repr=False)
    text_sha256: str
    address_hash: str
    vector: PackedVector = field(repr=False)

    def __post_init__(self) -> None:
        _identifier(self.source_fragment_id, "fragment", "source_fragment_id")
        _identifier(self.source_id, "source", "source_id")
        _identifier(self.source_version_id, "source_version", "source_version_id")
        if type(self.text) is not str or not self.text or len(self.text) > 250_000:
            raise CartographyError("cartography fragment text is empty or exceeds its bound")
        if "\x00" in self.text or unicodedata.normalize("NFC", self.text) != self.text:
            raise CartographyError("cartography fragment text must be NFC without NUL")
        _hash(self.text_sha256, "text_sha256")
        _hash(self.address_hash, "address_hash")
        if sha256_hex(self.text.encode("utf-8")) != self.text_sha256:
            raise CartographyError("cartography fragment text does not match text_sha256")
        if type(self.vector) is not PackedVector:
            raise CartographyError("cartography fragment vector must be PackedVector")

    def text_free_payload(self) -> dict[str, str]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "text_sha256": self.text_sha256,
            "address_hash": self.address_hash,
            "vector_sha256": self.vector.vector_hash,
        }


@dataclass(frozen=True, slots=True)
class CartographyConfig:
    """Frozen, domain-neutral clustering and signal thresholds."""

    SCHEMA: ClassVar[str] = "dithyramba.cartography_config/1.0"

    min_cluster_size: int = 3
    min_samples: int = 2
    representative_count: int = 5
    bridge_similarity_hex: str = (0.58).hex()
    noise_signal_ratio_hex: str = (0.25).hex()
    max_units: int = 5_000

    def __post_init__(self) -> None:
        _integer(self.min_cluster_size, 2, 1_000, "min_cluster_size")
        _integer(self.min_samples, 1, self.min_cluster_size, "min_samples")
        _integer(self.representative_count, 1, 32, "representative_count")
        _integer(self.max_units, self.min_cluster_size, 50_000, "max_units")
        _bounded_score(self.bridge_similarity_hex, 0.0, 1.0, "bridge_similarity_hex")
        _bounded_score(self.noise_signal_ratio_hex, 0.0, 1.0, "noise_signal_ratio_hex")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "method": "hdbscan_l2_over_unit_vectors_v1",
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "representative_count": self.representative_count,
            "bridge_similarity_hex": self.bridge_similarity_hex,
            "noise_signal_ratio_hex": self.noise_signal_ratio_hex,
            "max_units": self.max_units,
        }

    @property
    def profile_hash(self) -> str:
        return canonical_sha256_hex(self.payload())


@dataclass(frozen=True, slots=True)
class AreaMember:
    source_fragment_id: str
    source_id: str
    source_version_id: str
    text_sha256: str
    address_hash: str
    vector_sha256: str
    membership_hex: str

    def __post_init__(self) -> None:
        _identifier(self.source_fragment_id, "fragment", "source_fragment_id")
        _identifier(self.source_id, "source", "source_id")
        _identifier(self.source_version_id, "source_version", "source_version_id")
        _hash(self.text_sha256, "text_sha256")
        _hash(self.address_hash, "address_hash")
        _hash(self.vector_sha256, "vector_sha256")
        _bounded_score(self.membership_hex, 0.0, 1.0, "membership_hex")

    def payload(self) -> dict[str, str]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "source_id": self.source_id,
            "source_version_id": self.source_version_id,
            "text_sha256": self.text_sha256,
            "address_hash": self.address_hash,
            "vector_sha256": self.vector_sha256,
            "membership_hex": self.membership_hex,
        }


@dataclass(frozen=True, slots=True)
class AreaProjection:
    """One content-addressed, rebuildable semantic neighbourhood."""

    area_id: str
    boundary_hash: str
    centroid_vector_sha256: str
    members: tuple[AreaMember, ...]
    representative_fragment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.area_id, "area", "area_id")
        _hash(self.boundary_hash, "boundary_hash")
        _hash(self.centroid_vector_sha256, "centroid_vector_sha256")
        if self.area_id != f"area_{self.boundary_hash[:32]}":
            raise CartographyError("area ID does not match boundary hash")
        if type(self.members) is not tuple or not self.members:
            raise CartographyError("area requires at least one member")
        ordered = tuple(sorted(self.members, key=lambda item: item.source_fragment_id))
        if ordered != self.members:
            raise CartographyError("area members must use canonical fragment order")
        member_ids = tuple(item.source_fragment_id for item in self.members)
        if len(set(member_ids)) != len(member_ids):
            raise CartographyError("area member IDs must be unique")
        if not self.representative_fragment_ids:
            raise CartographyError("area requires at least one representative")
        if not set(self.representative_fragment_ids).issubset(member_ids):
            raise CartographyError("area representative is not an area member")
        if len(set(self.representative_fragment_ids)) != len(self.representative_fragment_ids):
            raise CartographyError("area representatives must be unique")

    def payload(self) -> dict[str, object]:
        return {
            "area_id": self.area_id,
            "boundary_hash": self.boundary_hash,
            "centroid_vector_sha256": self.centroid_vector_sha256,
            "members": [item.payload() for item in self.members],
            "representative_fragment_ids": list(self.representative_fragment_ids),
        }


@dataclass(frozen=True, slots=True)
class InquirySignal:
    """A deterministic trigger for later human/LLM question formulation."""

    signal_id: str
    kind: InquirySignalKind
    area_ids: tuple[str, ...]
    trigger_fragment_ids: tuple[str, ...]
    score_hex: str

    def __post_init__(self) -> None:
        _identifier(self.signal_id, "inquiry_signal", "signal_id")
        if not isinstance(self.kind, InquirySignalKind):
            raise CartographyError("inquiry signal kind is invalid")
        if not self.area_ids and self.kind is not InquirySignalKind.UNMAPPED_MASS:
            raise CartographyError("area-bound inquiry signal requires an area")
        for area_id in self.area_ids:
            _identifier(area_id, "area", "area_id")
        for fragment_id in self.trigger_fragment_ids:
            _identifier(fragment_id, "fragment", "source_fragment_id")
        if tuple(sorted(set(self.area_ids))) != self.area_ids:
            raise CartographyError("inquiry signal area IDs must be canonical and unique")
        if tuple(sorted(set(self.trigger_fragment_ids))) != self.trigger_fragment_ids:
            raise CartographyError("inquiry trigger IDs must be canonical and unique")
        _bounded_score(self.score_hex, 0.0, 1.0, "score_hex")
        digest = canonical_sha256_hex(
            {"schema": "dithyramba.inquiry_signal/1.0", **self.semantic_payload()}
        )
        if self.signal_id != f"inquiry_signal_{digest[:32]}":
            raise CartographyError("inquiry signal ID does not match its payload")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "area_ids": list(self.area_ids),
            "trigger_fragment_ids": list(self.trigger_fragment_ids),
            "score_hex": self.score_hex,
        }


@dataclass(frozen=True, slots=True)
class AreaMap:
    """Text-free map bound to one exact authorized corpus and vector view."""

    SCHEMA: ClassVar[str] = "dithyramba.area_map/1.0"

    library_id: str
    corpus_snapshot_id: str
    access_policy_id: str
    permitted_set_hash: str
    vector_generation_hash: str
    cartography_profile_hash: str
    cartography_runtime_hash: str
    input_manifest_hash: str
    input_fragment_count: int
    areas: tuple[AreaProjection, ...]
    unassigned_fragment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.library_id, "library", "library_id")
        _identifier(self.corpus_snapshot_id, "snapshot", "corpus_snapshot_id")
        _identifier(self.access_policy_id, "policy", "access_policy_id")
        for value in (
            self.permitted_set_hash,
            self.vector_generation_hash,
            self.cartography_profile_hash,
            self.cartography_runtime_hash,
            self.input_manifest_hash,
        ):
            _hash(value, "area map hash")
        _integer(self.input_fragment_count, 1, 50_000, "input_fragment_count")
        if tuple(sorted(self.areas, key=lambda item: item.area_id)) != self.areas:
            raise CartographyError("areas must use canonical ID order")
        if len({item.area_id for item in self.areas}) != len(self.areas):
            raise CartographyError("area IDs must be unique")
        for area in self.areas:
            expected_boundary_hash = canonical_sha256_hex(
                {
                    "schema": "dithyramba.area_boundary/1.0",
                    "cartography_profile_hash": self.cartography_profile_hash,
                    "members": [
                        {
                            "source_fragment_id": member.source_fragment_id,
                            "text_sha256": member.text_sha256,
                            "address_hash": member.address_hash,
                            "vector_sha256": member.vector_sha256,
                        }
                        for member in area.members
                    ],
                }
            )
            if area.boundary_hash != expected_boundary_hash:
                raise CartographyError("area boundary hash does not match its members")
        assigned = {member.source_fragment_id for area in self.areas for member in area.members}
        if len(assigned) != sum(len(area.members) for area in self.areas):
            raise CartographyError("one fragment cannot belong to two hard-cluster areas")
        if tuple(sorted(set(self.unassigned_fragment_ids))) != self.unassigned_fragment_ids:
            raise CartographyError("unassigned fragment IDs must be canonical and unique")
        if assigned.intersection(self.unassigned_fragment_ids):
            raise CartographyError("assigned and unassigned fragments overlap")
        if len(assigned) + len(self.unassigned_fragment_ids) != self.input_fragment_count:
            raise CartographyError("area map does not account for every input fragment")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "library_id": self.library_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "access_policy_id": self.access_policy_id,
            "permitted_set_hash": self.permitted_set_hash,
            "vector_generation_hash": self.vector_generation_hash,
            "cartography_profile_hash": self.cartography_profile_hash,
            "cartography_runtime_hash": self.cartography_runtime_hash,
            "input_manifest_hash": self.input_manifest_hash,
            "input_fragment_count": self.input_fragment_count,
            "areas": [item.payload() for item in self.areas],
            "unassigned_fragment_ids": list(self.unassigned_fragment_ids),
        }

    @property
    def map_hash(self) -> str:
        return canonical_sha256_hex(self.payload())

    @property
    def map_id(self) -> str:
        return canonical_content_id("area_map", self.payload())


@dataclass(frozen=True, slots=True)
class Inquiry:
    """Human-readable question candidate bound to observable map signals."""

    inquiry_id: str
    question: str
    state: InquiryState
    signal_ids: tuple[str, ...]
    evidence_requirement_ids: tuple[str, ...]
    formulation_profile_hash: str

    def __post_init__(self) -> None:
        _identifier(self.inquiry_id, "inquiry", "inquiry_id")
        _text(self.question, "question", maximum=2_000)
        if not isinstance(self.state, InquiryState):
            raise CartographyError("inquiry state is invalid")
        _canonical_ids(self.signal_ids, "inquiry_signal", "signal_ids")
        _canonical_ids(
            self.evidence_requirement_ids,
            "requirement",
            "evidence_requirement_ids",
        )
        _hash(self.formulation_profile_hash, "formulation_profile_hash")


@dataclass(frozen=True, slots=True)
class BoundedTrace:
    """Reviewable synthesis whose claims remain closed over exact fragments."""

    trace_id: str
    inquiry_id: str
    statement: str
    supporting_fragment_ids: tuple[str, ...]
    counterevidence_fragment_ids: tuple[str, ...]
    falsifier: str
    review_state: ReviewState = ReviewState.CANDIDATE

    def __post_init__(self) -> None:
        _identifier(self.trace_id, "trace", "trace_id")
        _identifier(self.inquiry_id, "inquiry", "inquiry_id")
        _text(self.statement, "statement", maximum=8_000)
        _canonical_ids(
            self.supporting_fragment_ids,
            "fragment",
            "supporting_fragment_ids",
        )
        _canonical_ids(
            self.counterevidence_fragment_ids,
            "fragment",
            "counterevidence_fragment_ids",
            allow_empty=True,
        )
        if set(self.supporting_fragment_ids).intersection(self.counterevidence_fragment_ids):
            raise CartographyError("one fragment cannot support and counter one trace")
        _text(self.falsifier, "falsifier", maximum=4_000)
        if not isinstance(self.review_state, ReviewState):
            raise CartographyError("bounded trace review state is invalid")


@dataclass(frozen=True, slots=True)
class CartographyResult:
    area_map: AreaMap
    signals: tuple[InquirySignal, ...]

    def __post_init__(self) -> None:
        if type(self.area_map) is not AreaMap:
            raise CartographyError("cartography result requires an AreaMap")
        if type(self.signals) is not tuple or any(
            type(item) is not InquirySignal for item in self.signals
        ):
            raise CartographyError("cartography signals must be an exact tuple")
        if tuple(sorted(self.signals, key=lambda item: item.signal_id)) != self.signals:
            raise CartographyError("cartography signals must use canonical ID order")


def _identifier(value: str, prefix: str, label: str) -> str:
    if type(value) is not str or not value.startswith(prefix + "_"):
        raise CartographyError(f"{label} must start with {prefix}_")
    suffix = value[len(prefix) + 1 :]
    if (
        not suffix
        or len(value) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in suffix)
    ):
        raise CartographyError(f"{label} is not canonical")
    return value


def _canonical_ids(
    values: tuple[str, ...],
    prefix: str,
    label: str,
    *,
    allow_empty: bool = False,
) -> None:
    if type(values) is not tuple or (not values and not allow_empty):
        raise CartographyError(f"{label} must be a canonical tuple")
    for value in values:
        _identifier(value, prefix, label)
    if tuple(sorted(set(values))) != values:
        raise CartographyError(f"{label} must be sorted and unique")


def _hash(value: str, label: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise CartographyError(f"{label} must be lowercase SHA-256")
    return value


def _integer(value: int, minimum: int, maximum: int, label: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise CartographyError(f"{label} is outside its bound")


def _bounded_score(value: str, minimum: float, maximum: float, label: str) -> float:
    if type(value) is not str:
        raise CartographyError(f"{label} must use canonical float.hex text")
    try:
        decoded = float.fromhex(value)
    except ValueError as error:
        raise CartographyError(f"{label} must use canonical float.hex text") from error
    if not math.isfinite(decoded) or decoded.hex() != value:
        raise CartographyError(f"{label} must use canonical finite float.hex text")
    if not minimum <= decoded <= maximum:
        raise CartographyError(f"{label} is outside its bound")
    return decoded


def _text(value: str, label: str, *, maximum: int) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise CartographyError(f"{label} must be bounded, stripped NFC text")
    return value
