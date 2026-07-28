from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dithyramba.access import MembershipState
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex
from dithyramba.provenance import SourceFamilyRole
from dithyramba.snapshots import (
    CorpusSnapshot,
    SnapshotContractError,
    SnapshotMember,
    build_corpus_snapshot,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _member(
    collection_id: str,
    source_version_id: str,
    *,
    source_id: str = "source_root",
    content_sha256: str = HASH_A,
    membership_state: MembershipState = MembershipState.ACTIVE,
    source_family_id: str = "family_root",
    root_source_id: str = "source_root",
    family_role: SourceFamilyRole = SourceFamilyRole.ROOT,
) -> SnapshotMember:
    return SnapshotMember(
        collection_id=collection_id,
        source_id=source_id,
        source_version_id=source_version_id,
        content_sha256=content_sha256,
        membership_state=membership_state,
        source_family_id=source_family_id,
        root_source_id=root_source_id,
        family_role=family_role,
    )


def test_snapshot_canonicalizes_scope_and_member_order() -> None:
    snapshot = build_corpus_snapshot(
        library_id="library_main",
        collection_ids=("collection_z", "collection_a"),
        members=(
            _member("collection_z", "source_version_2"),
            _member("collection_a", "source_version_2"),
            _member("collection_a", "source_version_1", content_sha256=HASH_B),
        ),
    )

    assert snapshot.collection_ids == ("collection_a", "collection_z")
    assert [(item.collection_id, item.source_version_id) for item in snapshot.members] == [
        ("collection_a", "source_version_1"),
        ("collection_a", "source_version_2"),
        ("collection_z", "source_version_2"),
    ]
    assert snapshot.manifest_hash == canonical_sha256_hex(snapshot.semantic_payload())
    assert snapshot.corpus_snapshot_id == canonical_content_id(
        "snapshot", snapshot.semantic_payload()
    )


def test_semantically_identical_snapshots_have_identical_bytes_hash_and_id() -> None:
    first_member = _member("collection_a", "source_version_1")
    second_member = _member("collection_b", "source_version_2", content_sha256=HASH_B)
    first = build_corpus_snapshot(
        library_id="library_main",
        collection_ids=("collection_b", "collection_a"),
        members=(second_member, first_member),
    )
    second = build_corpus_snapshot(
        library_id="library_main",
        collection_ids=("collection_a", "collection_b"),
        members=(first_member, second_member),
    )
    assert first.canonical_bytes == second.canonical_bytes
    assert first.manifest_hash == second.manifest_hash
    assert first.corpus_snapshot_id == second.corpus_snapshot_id


def test_empty_collections_remain_explicitly_represented_in_snapshot_scope() -> None:
    snapshot = build_corpus_snapshot(
        library_id="library_main",
        collection_ids=("collection_empty",),
    )
    assert snapshot.members == ()
    assert snapshot.semantic_payload()["scope"] == {
        "schema": "dithyramba.corpus_snapshot_scope/1.0",
        "library_id": "library_main",
        "collection_ids": ["collection_empty"],
    }
    assert snapshot.semantic_payload()["schema"] == ("dithyramba.corpus_snapshot_manifest/1.0")
    assert b'"members":[]' in snapshot.canonical_bytes


def test_snapshot_hash_domain_covers_library_scope_membership_content_and_lineage() -> None:
    base = build_corpus_snapshot(
        library_id="library_main",
        collection_ids=("collection_a",),
        members=(_member("collection_a", "source_version_1"),),
    )
    variants = (
        build_corpus_snapshot(
            library_id="library_other",
            collection_ids=("collection_a",),
            members=(_member("collection_a", "source_version_1"),),
        ),
        build_corpus_snapshot(
            library_id="library_main",
            collection_ids=("collection_a", "collection_empty"),
            members=(_member("collection_a", "source_version_1"),),
        ),
        build_corpus_snapshot(
            library_id="library_main",
            collection_ids=("collection_a",),
            members=(
                _member(
                    "collection_a",
                    "source_version_1",
                    membership_state=MembershipState.HOLDOUT,
                ),
            ),
        ),
        build_corpus_snapshot(
            library_id="library_main",
            collection_ids=("collection_a",),
            members=(_member("collection_a", "source_version_1", content_sha256=HASH_B),),
        ),
        build_corpus_snapshot(
            library_id="library_main",
            collection_ids=("collection_a",),
            members=(
                _member(
                    "collection_a",
                    "source_version_1",
                    source_id="source_derivative",
                    root_source_id="source_root",
                    family_role=SourceFamilyRole.DERIVATIVE,
                ),
            ),
        ),
    )
    assert len({base.manifest_hash, *(item.manifest_hash for item in variants)}) == 6


@pytest.mark.parametrize(
    "values",
    [
        {"collection_ids": ()},
        {"collection_ids": ("collection_a", "collection_a")},
        {"collection_ids": tuple(f"collection_{index}" for index in range(17))},
        {"library_id": "wrong_library"},
    ],
)
def test_snapshot_rejects_invalid_scope(values: dict[str, Any]) -> None:
    payload: dict[str, Any] = {
        "library_id": "library_main",
        "collection_ids": ("collection_a",),
        "members": (),
        **values,
    }
    with pytest.raises((ValidationError, SnapshotContractError)):
        CorpusSnapshot.model_validate(payload)


def test_snapshot_rejects_duplicate_out_of_scope_and_inconsistent_members() -> None:
    item = _member("collection_a", "source_version_1")
    with pytest.raises(ValidationError, match="unique"):
        CorpusSnapshot(
            library_id="library_main",
            collection_ids=("collection_a",),
            members=(item, item),
        )
    with pytest.raises(ValidationError, match="outside scope"):
        CorpusSnapshot(
            library_id="library_main",
            collection_ids=("collection_a",),
            members=(_member("collection_b", "source_version_1"),),
        )
    with pytest.raises(ValidationError, match="inconsistent"):
        CorpusSnapshot(
            library_id="library_main",
            collection_ids=("collection_a", "collection_b"),
            members=(
                _member("collection_a", "source_version_1"),
                _member("collection_b", "source_version_1", content_sha256=HASH_B),
            ),
        )


def test_snapshot_member_rejects_invalid_root_lineage_and_hash() -> None:
    with pytest.raises(ValidationError, match="root lineage"):
        _member("collection_a", "source_version_1", root_source_id="source_other")
    with pytest.raises(ValidationError, match="non-root"):
        _member(
            "collection_a",
            "source_version_1",
            family_role=SourceFamilyRole.DERIVATIVE,
        )
    with pytest.raises(ValidationError, match="SHA-256"):
        _member("collection_a", "source_version_1", content_sha256="A" * 64)


def test_snapshot_prefixed_ids_require_canonical_nonempty_suffixes() -> None:
    with pytest.raises(ValidationError, match="canonical non-empty suffix"):
        CorpusSnapshot(library_id="library_", collection_ids=("collection_a",))
    invalid_members = (
        {"collection_id": "collection_"},
        {"source_id": "source_"},
        {"source_version_id": "source_version_"},
        {"source_family_id": "family_"},
        {"root_source_id": "source_"},
        {"source_id": "source_a__b", "root_source_id": "source_a__b"},
    )
    for changes in invalid_members:
        values: dict[str, Any] = {
            "collection_id": "collection_a",
            "source_id": "source_root",
            "source_version_id": "source_version_1",
            "source_family_id": "family_root",
            "root_source_id": "source_root",
            **changes,
        }
        with pytest.raises(ValidationError, match="canonical non-empty suffix"):
            _member(
                values["collection_id"],
                values["source_version_id"],
                source_id=values["source_id"],
                source_family_id=values["source_family_id"],
                root_source_id=values["root_source_id"],
            )


def test_snapshot_rejects_member_subclasses_with_overridable_payloads() -> None:
    class DerivedSnapshotMember(SnapshotMember):
        pass

    derived = DerivedSnapshotMember.model_validate(
        _member("collection_a", "source_version_1").model_dump()
    )
    with pytest.raises(ValidationError, match="SnapshotMember"):
        CorpusSnapshot(
            library_id="library_main",
            collection_ids=("collection_a",),
            members=(derived,),
        )


def test_snapshot_models_are_strict_frozen_and_forbid_run_time_or_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        CorpusSnapshot.model_validate(
            {
                "library_id": "library_main",
                "collection_ids": ("collection_a",),
                "members": (),
                "processing_run_id": "run_forbidden",
                "created_at": "2026-07-20T12:00:00.000000Z",
            }
        )
    snapshot = CorpusSnapshot(
        library_id="library_main",
        collection_ids=("collection_a",),
    )
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.library_id = "library_other"
    assert "run_" not in snapshot.canonical_bytes.decode()
    assert "created_at" not in snapshot.canonical_bytes.decode()
