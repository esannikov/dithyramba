"""Causal tests for adaptive compatibility receipts and proof metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pytest
from pydantic import ValidationError

from dithyramba.recall.compatibility import (
    CompatibilityContractError,
    ExternalReferenceMap,
    ExternalReferenceMapEntry,
    FragmentTextProjectionReceipt,
    ProofMetadataEntry,
    ProofMetadataManifest,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _projection(suffix: str, *, library_id: str = "library_main") -> FragmentTextProjectionReceipt:
    raw = f"  fragment {suffix}  "
    return FragmentTextProjectionReceipt.from_texts(
        library_id=library_id,
        source_version_id=f"source_version_{suffix}",
        source_fragment_id=f"fragment_{suffix}",
        raw_text=raw,
        raw_trim_start=2,
        raw_trim_end=len(raw) - 2,
    )


def _reference_entry(
    suffix: str,
    *,
    library_id: str = "library_main",
    alias: str | None = None,
    canonical_kind: Literal["collection", "source", "source_version", "fragment"] = "fragment",
    canonical_id: str | None = None,
) -> ExternalReferenceMapEntry:
    return ExternalReferenceMapEntry(
        library_id=library_id,
        external_id=f"external-{suffix}",
        original_alias=alias or f"Original {suffix}",
        canonical_kind=canonical_kind,
        canonical_id=canonical_id or f"fragment_{suffix}",
    )


def _reference_map(
    suffixes: Iterable[str],
    *,
    library_id: str = "library_main",
    entries: tuple[ExternalReferenceMapEntry, ...] | None = None,
) -> ExternalReferenceMap:
    values = tuple(suffixes)
    actual_entries = entries or tuple(
        _reference_entry(item, library_id=library_id) for item in values
    )
    return ExternalReferenceMap(
        library_id=library_id,
        namespace="maulstick_import",
        expected_external_ids=tuple(f"external-{item}" for item in values),
        entries=actual_entries,
    )


def _metadata(
    suffix: str,
    projection: FragmentTextProjectionReceipt,
    *,
    library_id: str = "library_main",
    external_reference_id: str | None = None,
    proof: bool = True,
) -> ProofMetadataEntry:
    return ProofMetadataEntry(
        library_id=library_id,
        source_id=f"source_{suffix}",
        source_version_id=f"source_version_{suffix}",
        source_fragment_id=f"fragment_{suffix}",
        projection_receipt_hash=projection.receipt_hash,
        external_reference_id=external_reference_id,
        fragment_kind="paragraph",
        source_kind="archival record",
        source_family="archive",
        authority="primary",
        independence_group=f"source_{suffix}",
        evidence_tags=("root_source", "dated"),
        anchor_keys=("decisive_anchor",),
        direction_keys=("actor_to_record",),
        body_proof_eligible=proof,
    )


def _manifest(
    suffixes: tuple[str, ...] = ("alpha", "beta"),
    *,
    library_id: str = "library_main",
    projections: tuple[FragmentTextProjectionReceipt, ...] | None = None,
    entries: tuple[ProofMetadataEntry, ...] | None = None,
    references: ExternalReferenceMap | None = None,
) -> ProofMetadataManifest:
    actual_projections = projections or tuple(
        _projection(suffix, library_id=library_id) for suffix in suffixes
    )
    projection_by_suffix = {
        item.source_fragment_id.removeprefix("fragment_"): item for item in actual_projections
    }
    actual_entries = entries or tuple(
        _metadata(
            suffix,
            projection_by_suffix[suffix],
            library_id=library_id,
            external_reference_id=f"external-{suffix}",
        )
        for suffix in suffixes
    )
    return ProofMetadataManifest(
        library_id=library_id,
        retrieval_corpus_hash=HASH_A,
        expected_fragment_ids=tuple(f"fragment_{suffix}" for suffix in suffixes),
        text_projections=actual_projections,
        external_references=references or _reference_map(suffixes, library_id=library_id),
        entries=actual_entries,
    )


def test_fragment_projection_replays_trim_and_nfc_without_storing_text() -> None:
    raw = "  Cafe\u0301  "
    receipt = FragmentTextProjectionReceipt.from_texts(
        library_id="library_main",
        source_version_id="source_version_alpha",
        source_fragment_id="fragment_alpha",
        raw_text=raw,
        raw_trim_start=2,
        raw_trim_end=len(raw) - 2,
    )

    receipt.verify_texts(raw_text=raw, projected_text="Café")
    assert receipt.raw_character_count == len(raw)
    assert receipt.projected_character_count == len("Café")
    assert "Cafe" not in receipt.canonical_bytes.decode("utf-8")
    assert len(receipt.receipt_hash) == 64


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 4), (0, 99), (2, 2), (0, 4)],
)
def test_fragment_projection_rejects_invalid_or_non_whitespace_trim(
    start: int,
    end: int,
) -> None:
    with pytest.raises((CompatibilityContractError, ValidationError)):
        FragmentTextProjectionReceipt.from_texts(
            library_id="library_main",
            source_version_id="source_version_alpha",
            source_fragment_id="fragment_alpha",
            raw_text="xxproofyy",
            raw_trim_start=start,
            raw_trim_end=end,
        )


def test_fragment_projection_detects_text_tampering() -> None:
    receipt = _projection("alpha")

    with pytest.raises(CompatibilityContractError, match="does not replay"):
        receipt.verify_texts(raw_text="  fragment alpha  ", projected_text="fragment beta")
    with pytest.raises(CompatibilityContractError, match="raw text hash"):
        receipt.verify_texts(raw_text="\t fragment alpha  ", projected_text="fragment alpha")


def test_fragment_projection_rejects_fake_identity_projection() -> None:
    with pytest.raises(ValidationError, match="identity projection"):
        FragmentTextProjectionReceipt(
            library_id="library_main",
            source_version_id="source_version_alpha",
            source_fragment_id="fragment_alpha",
            raw_text_sha256=HASH_A,
            projected_text_sha256=HASH_A,
            raw_character_count=10,
            projected_character_count=8,
            raw_trim_start=1,
            raw_trim_end=9,
        )
    with pytest.raises(ValidationError, match="cannot exceed"):
        FragmentTextProjectionReceipt(
            library_id="library_main",
            source_version_id="source_version_alpha",
            source_fragment_id="fragment_alpha",
            raw_text_sha256=HASH_A,
            projected_text_sha256=HASH_B,
            raw_character_count=10,
            projected_character_count=9,
            raw_trim_start=1,
            raw_trim_end=9,
        )


def test_external_reference_map_is_canonical_and_resolves_explicit_scope() -> None:
    alpha = _reference_entry("alpha")
    beta = _reference_entry("beta")
    forward = _reference_map(("alpha", "beta"), entries=(alpha, beta))
    reverse = _reference_map(("beta", "alpha"), entries=(beta, alpha))

    assert forward.map_hash == reverse.map_hash
    assert forward.canonical_bytes == reverse.canonical_bytes
    assert forward.resolve("external-alpha", library_id="library_main").canonical_id == (
        "fragment_alpha"
    )
    assert forward.resolve_alias("original alpha", library_id="library_main").external_id == (
        "external-alpha"
    )


def test_external_reference_map_rejects_missing_extra_and_duplicate_entries() -> None:
    alpha = _reference_entry("alpha")
    beta = _reference_entry("beta")

    with pytest.raises(ValidationError, match="coverage mismatch"):
        _reference_map(("alpha", "beta"), entries=(alpha,))
    with pytest.raises(ValidationError, match="coverage mismatch"):
        _reference_map(("alpha",), entries=(alpha, beta))
    with pytest.raises(ValidationError, match="external identifiers must be unique"):
        _reference_map(("alpha",), entries=(alpha, alpha))


def test_external_reference_map_rejects_normalized_alias_collisions() -> None:
    alpha = _reference_entry("alpha", alias="Évidence")
    beta = _reference_entry("beta", alias="éVIDENCE")

    with pytest.raises(ValidationError, match="aliases collide"):
        _reference_map(("alpha", "beta"), entries=(alpha, beta))


def test_external_reference_map_rejects_undeclared_many_to_one_targets() -> None:
    alpha = _reference_entry("alpha", canonical_id="fragment_shared")
    beta = _reference_entry("beta", canonical_id="fragment_shared")

    with pytest.raises(ValidationError, match="many-to-one"):
        _reference_map(("alpha", "beta"), entries=(alpha, beta))


def test_external_reference_lookup_rejects_cross_library_and_unknown_reference() -> None:
    references = _reference_map(("alpha",))

    with pytest.raises(CompatibilityContractError, match="cross-Library"):
        references.resolve("external-alpha", library_id="library_other")
    with pytest.raises(CompatibilityContractError, match="unknown external reference"):
        references.resolve("external-unknown", library_id="library_main")


def test_external_reference_map_rejects_wrong_id_kind_and_cross_library_entry() -> None:
    with pytest.raises(ValidationError, match="source_version_ prefix"):
        _reference_entry(
            "alpha",
            canonical_kind="source_version",
            canonical_id="fragment_alpha",
        )
    foreign = _reference_entry("alpha", library_id="library_other")
    with pytest.raises(ValidationError, match="cannot cross"):
        _reference_map(("alpha",), entries=(foreign,))


def test_proof_manifest_has_deterministic_order_hash_and_explicit_false() -> None:
    alpha_projection = _projection("alpha")
    beta_projection = _projection("beta")
    alpha = _metadata(
        "alpha",
        alpha_projection,
        external_reference_id="external-alpha",
    )
    beta = _metadata(
        "beta",
        beta_projection,
        external_reference_id="external-beta",
        proof=False,
    )
    references = _reference_map(("alpha", "beta"))
    forward = _manifest(
        projections=(alpha_projection, beta_projection),
        entries=(alpha, beta),
        references=references,
    )
    reverse = _manifest(
        suffixes=("beta", "alpha"),
        projections=(beta_projection, alpha_projection),
        entries=(beta, alpha),
        references=_reference_map(("beta", "alpha")),
    )

    assert forward.manifest_hash == reverse.manifest_hash
    assert tuple(item.source_fragment_id for item in forward.entries) == (
        "fragment_alpha",
        "fragment_beta",
    )
    assert forward.entries[1].body_proof_eligible is False


@pytest.mark.parametrize("missing_field", ["body_proof_eligible", "projection_receipt_hash"])
def test_proof_metadata_has_no_implicit_critical_fields(missing_field: str) -> None:
    projection = _projection("alpha")
    payload = _metadata("alpha", projection).model_dump(mode="python")
    payload.pop(missing_field)

    with pytest.raises(ValidationError, match="Field required"):
        ProofMetadataEntry.model_validate(payload)


def test_proof_manifest_rejects_missing_extra_and_duplicate_coverage() -> None:
    alpha_projection = _projection("alpha")
    beta_projection = _projection("beta")
    alpha = _metadata("alpha", alpha_projection)
    empty_references = _reference_map(())

    with pytest.raises(ValidationError, match="proof metadata coverage mismatch"):
        _manifest(
            projections=(alpha_projection, beta_projection),
            entries=(alpha,),
            references=empty_references,
        )
    with pytest.raises(ValidationError, match="text projection coverage mismatch"):
        _manifest(
            suffixes=("alpha",),
            projections=(alpha_projection, beta_projection),
            entries=(alpha,),
            references=empty_references,
        )
    with pytest.raises(ValidationError, match="duplicate fragments"):
        _manifest(
            suffixes=("alpha",),
            projections=(alpha_projection, alpha_projection),
            entries=(alpha,),
            references=empty_references,
        )
    with pytest.raises(ValidationError, match="duplicate fragments"):
        _manifest(
            suffixes=("alpha",),
            projections=(alpha_projection,),
            entries=(alpha, alpha),
            references=empty_references,
        )


def test_proof_manifest_rejects_cross_library_objects() -> None:
    projection = _projection("alpha")
    foreign_projection = _projection("alpha", library_id="library_other")
    entry = _metadata("alpha", projection)
    foreign_entry = _metadata("alpha", projection, library_id="library_other")
    empty_references = _reference_map(())

    with pytest.raises(ValidationError, match="text projections cannot cross"):
        _manifest(
            suffixes=("alpha",),
            projections=(foreign_projection,),
            entries=(
                _metadata(
                    "alpha",
                    foreign_projection,
                    library_id="library_main",
                ),
            ),
            references=empty_references,
        )
    with pytest.raises(ValidationError, match="proof metadata cannot cross"):
        _manifest(
            suffixes=("alpha",),
            projections=(projection,),
            entries=(foreign_entry,),
            references=empty_references,
        )
    with pytest.raises(ValidationError, match="reference map cannot cross"):
        _manifest(
            suffixes=("alpha",),
            projections=(projection,),
            entries=(entry,),
            references=_reference_map((), library_id="library_other"),
        )


def test_proof_manifest_rejects_unknown_or_misdirected_external_reference() -> None:
    projection = _projection("alpha")
    unknown = _metadata(
        "alpha",
        projection,
        external_reference_id="external-unknown",
    )
    misdirected_reference = _reference_entry(
        "alpha",
        canonical_id="fragment_beta",
    )

    with pytest.raises(ValidationError, match="unknown external reference"):
        _manifest(
            suffixes=("alpha",),
            projections=(projection,),
            entries=(unknown,),
            references=_reference_map(("alpha",)),
        )
    with pytest.raises(ValidationError, match="resolves to another fragment"):
        _manifest(
            suffixes=("alpha",),
            projections=(projection,),
            entries=(
                _metadata(
                    "alpha",
                    projection,
                    external_reference_id="external-alpha",
                ),
            ),
            references=_reference_map(("alpha",), entries=(misdirected_reference,)),
        )


def test_proof_manifest_rejects_projection_version_and_hash_tampering() -> None:
    projection = _projection("alpha")
    entry = _metadata("alpha", projection)
    empty_references = _reference_map(())
    wrong_version = projection.model_copy(update={"source_version_id": "source_version_other"})
    tampered_projection = projection.model_copy(update={"projected_text_sha256": HASH_B})

    with pytest.raises(ValidationError, match="source version differs"):
        _manifest(
            suffixes=("alpha",),
            projections=(wrong_version,),
            entries=(entry,),
            references=empty_references,
        )
    with pytest.raises(ValidationError, match="receipt hash does not match"):
        _manifest(
            suffixes=("alpha",),
            projections=(tampered_projection,),
            entries=(entry,),
            references=empty_references,
        )


def test_compatibility_models_are_frozen_and_forbid_unknown_fields() -> None:
    manifest = _manifest()

    with pytest.raises(ValidationError, match="frozen"):
        manifest.library_id = "library_other"
    payload = manifest.model_dump(mode="python")
    payload["implicit_proof"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProofMetadataManifest.model_validate(payload)
