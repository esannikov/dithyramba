from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import pytest

from dithyramba.cartography import (
    AreaMap,
    BoundedTrace,
    CartographyConfig,
    CartographyFragment,
    CartographyResult,
    Inquiry,
    InquirySignal,
    InquirySignalKind,
    InquiryState,
    ReviewState,
    build_cartography,
)
from dithyramba.cartography import engine as cartography_engine
from dithyramba.cartography.errors import CartographyError
from dithyramba.contracts import canonical_sha256_hex, sha256_hex
from dithyramba.recall.vector import PackedVector, pack_normalized_vector

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _unit(*values: float) -> PackedVector:
    norm = math.sqrt(sum(value * value for value in values))
    return pack_normalized_vector(tuple(float(value / norm) for value in values))


def _fragment(
    index: int,
    vector: tuple[float, ...],
    *,
    source_group: int | None = None,
) -> CartographyFragment:
    text = f"camera rhythm movement cluster token {index}"
    group = index if source_group is None else source_group
    return CartographyFragment(
        source_fragment_id=f"fragment_f{index:03d}",
        source_id=f"source_s{group:03d}",
        source_version_id=f"source_version_v{group:03d}",
        text=text,
        text_sha256=sha256_hex(text.encode()),
        address_hash=HASH_A,
        vector=_unit(*vector),
    )


def _inputs() -> tuple[CartographyFragment, ...]:
    return (
        _fragment(1, (1.0, 0.01, 0.0), source_group=1),
        _fragment(2, (1.0, 0.02, 0.0), source_group=1),
        _fragment(3, (1.0, 0.03, 0.0), source_group=1),
        _fragment(4, (0.70, 0.70, 0.01)),
        _fragment(5, (0.71, 0.69, 0.01)),
        _fragment(6, (0.69, 0.71, 0.01)),
        _fragment(7, (-1.0, 0.0, 0.0)),
    )


def _build(fragments: tuple[CartographyFragment, ...]) -> CartographyResult:
    return build_cartography(
        library_id="library_test",
        corpus_snapshot_id="snapshot_test",
        access_policy_id="policy_test",
        permitted_set_hash=HASH_A,
        vector_generation_hash=HASH_B,
        fragments=fragments,
        config=CartographyConfig(
            min_cluster_size=3,
            min_samples=2,
            representative_count=2,
            bridge_similarity_hex=(0.55).hex(),
            noise_signal_ratio_hex=(0.10).hex(),
        ),
    )


def test_area_map_is_deterministic_and_text_free() -> None:
    forward = _build(_inputs())
    reverse = _build(tuple(reversed(_inputs())))

    assert forward.area_map.payload() == reverse.area_map.payload()
    assert forward.area_map.map_hash == reverse.area_map.map_hash
    assert forward.area_map.map_id.startswith("area_map_")
    assert len(forward.area_map.areas) == 2
    assert forward.area_map.unassigned_fragment_ids == ("fragment_f007",)
    assert "camera rhythm" not in str(forward.area_map.payload())


def test_cartography_emits_observable_signals_without_questions() -> None:
    result = _build(_inputs())

    kinds = {item.kind for item in result.signals}
    assert InquirySignalKind.AREA_BRIDGE in kinds
    assert InquirySignalKind.SOURCE_CONCENTRATION in kinds
    assert InquirySignalKind.UNMAPPED_MASS in kinds
    assert all(not hasattr(item, "question") for item in result.signals)


def test_cartography_rejects_text_hash_mismatch() -> None:
    with pytest.raises(CartographyError, match="does not match"):
        CartographyFragment(
            source_fragment_id="fragment_bad",
            source_id="source_bad",
            source_version_id="source_version_bad",
            text="exact text",
            text_sha256=HASH_C,
            address_hash=HASH_A,
            vector=_unit(1.0, 0.0),
        )


def test_inquiry_and_bounded_trace_remain_review_candidates() -> None:
    inquiry = Inquiry(
        inquiry_id="inquiry_test",
        question="How do these two areas define the same operation differently?",
        state=InquiryState.CANDIDATE,
        signal_ids=("inquiry_signal_test",),
        evidence_requirement_ids=("requirement_difference",),
        formulation_profile_hash=HASH_A,
    )
    trace = BoundedTrace(
        trace_id="trace_test",
        inquiry_id=inquiry.inquiry_id,
        statement="The sources propose two incompatible definitions within the selected scope.",
        supporting_fragment_ids=("fragment_f001",),
        counterevidence_fragment_ids=("fragment_f004",),
        falsifier="Reject the trace if the exact fragments use compatible definitions.",
        review_state=ReviewState.CANDIDATE,
    )

    assert inquiry.state is InquiryState.CANDIDATE
    assert trace.review_state is ReviewState.CANDIDATE


def test_trace_cannot_use_one_fragment_as_support_and_counterevidence() -> None:
    with pytest.raises(CartographyError, match="support and counter"):
        BoundedTrace(
            trace_id="trace_bad",
            inquiry_id="inquiry_test",
            statement="A bounded statement.",
            supporting_fragment_ids=("fragment_same",),
            counterevidence_fragment_ids=("fragment_same",),
            falsifier="A concrete condition that would reject the statement.",
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"text": ""}, "empty or exceeds"),
        ({"text": "bad\x00text"}, "NFC without NUL"),
        ({"address_hash": "bad"}, "lowercase SHA-256"),
        ({"vector": cast(PackedVector, object())}, "must be PackedVector"),
    ),
)
def test_fragment_contract_fails_closed(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(CartographyError, match=message):
        replace(_inputs()[0], **changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"min_cluster_size": 1},
        {"min_samples": 4},
        {"representative_count": 0},
        {"bridge_similarity_hex": "0.5"},
        {"noise_signal_ratio_hex": "nan"},
        {"max_units": 2},
    ),
)
def test_cartography_config_rejects_noncanonical_bounds(
    changes: dict[str, Any],
) -> None:
    values: dict[str, Any] = {"min_cluster_size": 3}
    values.update(changes)
    with pytest.raises(CartographyError):
        CartographyConfig(**values)


def test_area_and_signal_contracts_reject_tampering() -> None:
    result = _build(_inputs())
    area = result.area_map.areas[0]
    member = area.members[0]
    signal = next(item for item in result.signals if item.kind is InquirySignalKind.AREA_BRIDGE)

    with pytest.raises(CartographyError, match="does not match"):
        replace(area, area_id="area_wrong")
    with pytest.raises(CartographyError, match="requires at least one member"):
        replace(area, members=())
    with pytest.raises(CartographyError, match="canonical fragment order"):
        replace(area, members=tuple(reversed(area.members)))
    with pytest.raises(CartographyError, match="member IDs must be unique"):
        replace(area, members=(member, member))
    with pytest.raises(CartographyError, match="at least one representative"):
        replace(area, representative_fragment_ids=())
    with pytest.raises(CartographyError, match="not an area member"):
        replace(area, representative_fragment_ids=("fragment_missing",))
    with pytest.raises(CartographyError, match="representatives must be unique"):
        replace(
            area,
            representative_fragment_ids=(
                area.representative_fragment_ids[0],
                area.representative_fragment_ids[0],
            ),
        )
    with pytest.raises(CartographyError, match="kind is invalid"):
        replace(signal, kind=cast(InquirySignalKind, "bad"))
    with pytest.raises(CartographyError, match="requires an area"):
        InquirySignal(
            signal_id="inquiry_signal_missing_area",
            kind=InquirySignalKind.AREA_BRIDGE,
            area_ids=(),
            trigger_fragment_ids=(),
            score_hex=(0.5).hex(),
        )
    with pytest.raises(CartographyError, match="canonical and unique"):
        replace(signal, area_ids=tuple(reversed(signal.area_ids)))
    with pytest.raises(CartographyError, match="canonical and unique"):
        replace(signal, trigger_fragment_ids=("fragment_z", "fragment_a"))
    with pytest.raises(CartographyError, match="does not match its payload"):
        replace(signal, signal_id="inquiry_signal_tampered")


def test_area_map_inquiry_trace_and_result_reject_invalid_state() -> None:
    result = _build(_inputs())
    area_map = result.area_map
    areas = area_map.areas
    duplicated_members = tuple(
        sorted(
            (areas[0].members[0], *areas[1].members),
            key=lambda item: item.source_fragment_id,
        )
    )
    duplicated_boundary_hash = canonical_sha256_hex(
        {
            "schema": "dithyramba.area_boundary/1.0",
            "cartography_profile_hash": area_map.cartography_profile_hash,
            "members": [
                {
                    "source_fragment_id": member.source_fragment_id,
                    "text_sha256": member.text_sha256,
                    "address_hash": member.address_hash,
                    "vector_sha256": member.vector_sha256,
                }
                for member in duplicated_members
            ],
        }
    )
    duplicated_member_area = replace(
        areas[1],
        boundary_hash=duplicated_boundary_hash,
        area_id=f"area_{duplicated_boundary_hash[:32]}",
        members=duplicated_members,
    )

    with pytest.raises(CartographyError, match="canonical ID order"):
        replace(area_map, areas=tuple(reversed(areas)))
    with pytest.raises(CartographyError, match="area IDs must be unique"):
        replace(area_map, areas=(areas[0], areas[0]))
    with pytest.raises(CartographyError, match="hard-cluster"):
        replace(
            area_map,
            areas=tuple(sorted((areas[0], duplicated_member_area), key=lambda item: item.area_id)),
        )
    with pytest.raises(CartographyError, match="canonical and unique"):
        replace(area_map, unassigned_fragment_ids=("fragment_z", "fragment_a"))
    with pytest.raises(CartographyError, match="overlap"):
        replace(
            area_map,
            unassigned_fragment_ids=(areas[0].members[0].source_fragment_id,),
        )
    with pytest.raises(CartographyError, match="does not account"):
        replace(area_map, input_fragment_count=area_map.input_fragment_count + 1)
    with pytest.raises(CartographyError, match="boundary hash"):
        replace(
            area_map,
            areas=tuple(
                sorted(
                    (
                        replace(areas[0], boundary_hash=HASH_C, area_id=f"area_{HASH_C[:32]}"),
                        areas[1],
                    ),
                    key=lambda item: item.area_id,
                )
            ),
        )

    with pytest.raises(CartographyError, match="state is invalid"):
        Inquiry(
            inquiry_id="inquiry_bad_state",
            question="A valid bounded question?",
            state=cast(InquiryState, "bad"),
            signal_ids=("inquiry_signal_test",),
            evidence_requirement_ids=("requirement_test",),
            formulation_profile_hash=HASH_A,
        )
    with pytest.raises(CartographyError, match="bounded, stripped NFC text"):
        replace(
            BoundedTrace(
                trace_id="trace_valid",
                inquiry_id="inquiry_valid",
                statement="A valid statement.",
                supporting_fragment_ids=("fragment_a",),
                counterevidence_fragment_ids=(),
                falsifier="A valid falsifier.",
            ),
            statement=" padded ",
        )
    with pytest.raises(CartographyError, match="review state is invalid"):
        BoundedTrace(
            trace_id="trace_bad_state",
            inquiry_id="inquiry_valid",
            statement="A valid statement.",
            supporting_fragment_ids=("fragment_a",),
            counterevidence_fragment_ids=(),
            falsifier="A valid falsifier.",
            review_state=cast(ReviewState, "bad"),
        )
    with pytest.raises(CartographyError, match="requires an AreaMap"):
        CartographyResult(
            area_map=cast(AreaMap, object()),
            signals=(),
        )
    with pytest.raises(CartographyError, match="exact tuple"):
        CartographyResult(
            area_map=area_map,
            signals=cast(tuple[InquirySignal, ...], []),
        )
    with pytest.raises(CartographyError, match="canonical ID order"):
        CartographyResult(
            area_map=area_map,
            signals=tuple(reversed(result.signals)),
        )


def test_build_cartography_rejects_invalid_input_sets() -> None:
    inputs = _inputs()
    with pytest.raises(CartographyError, match="outside the configured bound"):
        _build(inputs[:2])
    with pytest.raises(CartographyError, match="must be unique"):
        _build((*inputs[:3], inputs[0]))
    with pytest.raises(CartographyError, match="one exact dimension"):
        _build((*inputs[:3], _fragment(9, (1.0, 0.0))))
    with pytest.raises(CartographyError, match="must contain"):
        build_cartography(
            library_id="library_test",
            corpus_snapshot_id="snapshot_test",
            access_policy_id="policy_test",
            permitted_set_hash=HASH_A,
            vector_generation_hash=HASH_B,
            fragments=cast(Sequence[CartographyFragment], (object(), object(), object())),
        )


def test_runtime_centroid_and_no_noise_defences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(CartographyError, match="centroid is undefined"):
        cartography_engine._centroid((_unit(1.0, 0.0), _unit(-1.0, 0.0)))

    result = build_cartography(
        library_id="library_test",
        corpus_snapshot_id="snapshot_test",
        access_policy_id="policy_test",
        permitted_set_hash=HASH_A,
        vector_generation_hash=HASH_B,
        fragments=_inputs(),
        config=CartographyConfig(
            min_cluster_size=3,
            min_samples=2,
            noise_signal_ratio_hex=(1.0).hex(),
        ),
    )
    assert InquirySignalKind.UNMAPPED_MASS not in {item.kind for item in result.signals}

    def _missing_runtime(_: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(
        "dithyramba.cartography.engine.importlib.import_module",
        _missing_runtime,
    )
    with pytest.raises(CartographyError, match="optional 'cartography' runtime"):
        _build(_inputs())
