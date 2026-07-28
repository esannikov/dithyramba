"""Float32 storage, generation binding, and exact vector-scan contracts."""

from __future__ import annotations

import math
import struct
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dithyramba.contracts import sha256_hex
from dithyramba.recall.vector import (
    CorpusVectorGeneration,
    DenseHit,
    DenseVectorItem,
    PackedVector,
    VectorContractError,
    VectorManifestItem,
    exact_cosine_scan,
    load_packed_vector,
    pack_normalized_vector,
)

HASH = "a" * 64


def _vector(x: float, y: float) -> PackedVector:
    return pack_normalized_vector((x, y))


def _generation() -> CorpusVectorGeneration:
    return CorpusVectorGeneration(
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        snapshot_hash=HASH,
        access_policy_id="policy_one",
        policy_hash=HASH,
        exclusion_hash=HASH,
        permitted_set_hash=HASH,
        model_profile_hash=HASH,
        runtime_profile_hash=HASH,
        dimensions=2,
        items=(
            VectorManifestItem(
                source_fragment_id="fragment_two",
                text_sha256="b" * 64,
                vector_sha256="c" * 64,
            ),
            VectorManifestItem(
                source_fragment_id="fragment_one",
                text_sha256="d" * 64,
                vector_sha256="e" * 64,
            ),
        ),
    )


def test_normalized_float32_round_trip_is_little_endian_and_hashed() -> None:
    vector = _vector(1.0, 0.0)

    assert vector.dimensions == 2
    assert vector.blob == struct.pack("<ff", 1.0, 0.0)
    assert vector.vector_hash == sha256_hex(vector.blob)
    assert vector.values == (1.0, 0.0)
    assert (
        load_packed_vector(
            vector.blob,
            dimensions=2,
            expected_hash=vector.vector_hash,
        )
        == vector
    )


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ((), "dimensions"),
        ((0.0, 0.0), "zero"),
        ((0.5, 0.5), "normalized"),
        ((math.nan, 0.0), "finite"),
        ((math.inf, 0.0), "finite"),
        ((cast(Any, 1), 0.0), "Python floats"),
        ((1e100, 0.0), "float32"),
    ],
)
def test_vector_pack_rejects_invalid_provider_output(
    values: tuple[Any, ...],
    match: str,
) -> None:
    with pytest.raises(VectorContractError, match=match):
        pack_normalized_vector(values)


def test_vector_pack_rejects_text_and_non_sequence() -> None:
    with pytest.raises(VectorContractError, match="numeric sequence"):
        pack_normalized_vector(cast(Any, "1,0"))
    with pytest.raises(VectorContractError, match="finite sequence"):
        pack_normalized_vector(cast(Any, 4))


def test_persisted_vector_rejects_hash_length_and_float_drift() -> None:
    vector = _vector(1.0, 0.0)
    with pytest.raises(VectorContractError, match="vector_hash"):
        PackedVector(dimensions=2, blob=vector.blob, vector_hash="bad")
    with pytest.raises(VectorContractError, match="does not match"):
        PackedVector(dimensions=2, blob=vector.blob, vector_hash="0" * 64)
    with pytest.raises(VectorContractError, match="length"):
        PackedVector(dimensions=3, blob=vector.blob, vector_hash=vector.vector_hash)
    nan_blob = struct.pack("<ff", math.nan, 0.0)
    with pytest.raises(VectorContractError, match="invalid"):
        PackedVector(dimensions=2, blob=nan_blob, vector_hash=sha256_hex(nan_blob))
    with pytest.raises(VectorContractError, match="immutable"):
        PackedVector(dimensions=2, blob=cast(Any, bytearray(vector.blob)), vector_hash=HASH)


def test_exact_cosine_scan_ranks_descending_and_breaks_ties_by_binary_id() -> None:
    query = _vector(1.0, 0.0)
    candidates = (
        DenseVectorItem("fragment_z", "source_z", _vector(1.0, 0.0)),
        DenseVectorItem("fragment_a", "source_a", _vector(1.0, 0.0)),
        DenseVectorItem("fragment_mid", "source_mid", _vector(0.0, 1.0)),
        DenseVectorItem("fragment_negative", "source_negative", _vector(-1.0, 0.0)),
    )

    hits = exact_cosine_scan(query, candidates, limit=3)

    assert tuple(item.source_fragment_id for item in hits) == (
        "fragment_a",
        "fragment_z",
        "fragment_mid",
    )
    assert tuple(item.rank for item in hits) == (1, 2, 3)
    assert hits[0].score == 1.0
    assert hits[2].score == 0.0


def test_exact_scan_fails_closed_on_duplicates_dimensions_and_invalid_inputs() -> None:
    query = _vector(1.0, 0.0)
    item = DenseVectorItem("fragment_one", "source_one", query)
    with pytest.raises(VectorContractError, match="unique"):
        exact_cosine_scan(query, (item, item), limit=2)
    with pytest.raises(VectorContractError, match="dimensions differ"):
        exact_cosine_scan(
            query,
            (DenseVectorItem("fragment_two", "source_two", pack_normalized_vector((1.0,))),),
            limit=1,
        )
    with pytest.raises(VectorContractError, match="limit"):
        exact_cosine_scan(query, (), limit=0)
    with pytest.raises(VectorContractError, match="PackedVector"):
        exact_cosine_scan(cast(Any, object()), (), limit=1)
    with pytest.raises(VectorContractError, match="DenseVectorItem"):
        exact_cosine_scan(query, cast(Any, (object(),)), limit=1)
    with pytest.raises(VectorContractError, match="fragment_"):
        DenseVectorItem("wrong", "source_one", query)
    with pytest.raises(VectorContractError, match="PackedVector"):
        DenseVectorItem("fragment_one", "source_one", cast(Any, object()))


def test_dense_hit_validates_exact_float_hex_on_read() -> None:
    assert DenseHit("fragment_one", "source_one", 1, (0.5).hex()).score == 0.5
    with pytest.raises(VectorContractError, match=r"float\.hex"):
        _ = DenseHit("fragment_one", "source_one", 1, "not-a-score").score
    with pytest.raises(VectorContractError, match="finite"):
        _ = DenseHit("fragment_one", "source_one", 1, "inf").score


def test_vector_generation_is_sorted_content_addressed_and_scope_bound() -> None:
    generation = _generation()

    assert tuple(item.source_fragment_id for item in generation.items) == (
        "fragment_one",
        "fragment_two",
    )
    assert generation.generation_id.startswith("vector_generation_")
    assert len(generation.generation_hash) == 64
    assert generation.semantic_payload()["dimensions"] == 2

    empty = CorpusVectorGeneration(**{**generation.model_dump(), "items": ()})
    assert empty.items == ()
    with pytest.raises((ValidationError, VectorContractError), match="unique"):
        CorpusVectorGeneration(
            **{**generation.model_dump(), "items": (generation.items[0], generation.items[0])}
        )
    with pytest.raises((ValidationError, VectorContractError), match="hash"):
        VectorManifestItem(
            source_fragment_id="fragment_one",
            text_sha256="bad",
            vector_sha256=HASH,
        )
