"""Shared source-closed fixtures for StructureUnit integration tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    RequestScope,
)
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.library import LibraryConfig
from dithyramba.persistence import SQLiteStructureRepository
from dithyramba.persistence.repository import LibraryRepository, initialize_library
from dithyramba.structure import StructureGenerationRequest

NOW = datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-21T14:00:00.000000Z"


def _clock() -> datetime:
    return NOW


def structure_context(
    tmp_path: Path,
    *,
    fragment_count: int = 3,
) -> tuple[
    LibraryRepository,
    SQLiteStructureRepository,
    tuple[str, ...],
    str,
    AccessPolicySnapshot,
    RequestScope,
]:
    """Create one authorized Library with exact synthetic fragments."""

    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    source_root.mkdir()
    repository = initialize_library(
        LibraryConfig(name="Structure"), data_root=data_root, clock=_clock
    )
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="Corpus",
            kind=CollectionKind.CORPUS,
            roots=(build_collection_root(source_root, data_root=data_root),),
        )
    )
    fragment_ids = _seed_source(
        repository,
        collection.config.collection_id,
        fragment_count=fragment_count,
    )
    snapshot = repository.freeze_snapshot((collection.config.collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id="policy_structure",
        library_id=repository.library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),),
    )
    repository.persist_access_policy(name="Research", snapshot=policy)
    scope = RequestScope(
        library_id=repository.library_id,
        snapshot_hash=snapshot.manifest_hash,
        purpose="research",
        collection_ids=(collection.config.collection_id,),
    )
    return (
        repository,
        SQLiteStructureRepository(repository),
        fragment_ids,
        snapshot.corpus_snapshot_id,
        policy,
        scope,
    )


def _seed_source(
    repository: LibraryRepository,
    collection_id: str,
    *,
    fragment_count: int = 3,
) -> tuple[str, ...]:
    connection = repository._store.connection
    texts = (
        ("Heading", "Primary evidence", "Its qualification")
        if fragment_count == 3
        else tuple(f"Evidence {ordinal}" for ordinal in range(fragment_count))
    )
    source_bytes = "\n".join(texts).encode()
    content_hash = sha256_hex(source_bytes)
    connection.execute(
        "INSERT INTO blobs VALUES (?, ?, 'aa/blob', ?)",
        (content_hash, len(source_bytes), NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO sources VALUES ('source_structure', ?, 'file:///source.md', "
        "'text/markdown', 'Source', ?)",
        (repository.library_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_versions VALUES ('source_version_structure', "
        "'source_structure', 1, ?, ?, ?, NULL, 'markdown_v1', 'processed', NULL)",
        (content_hash, len(source_bytes), NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_heads VALUES ('source_structure', 'source_version_structure', ?, NULL)",
        (NOW_TEXT,),
    )
    connection.execute(
        "INSERT INTO source_families VALUES ('family_structure', ?, 'Family', ?)",
        (repository.library_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_family_members VALUES "
        "('family_structure', 'source_structure', 'root', NULL, ?)",
        (NOW_TEXT,),
    )
    fragment_ids: list[str] = []
    for ordinal, text in enumerate(texts):
        fragment_id = f"fragment_structure_{ordinal}"
        address = {"kind": "markdown", "heading_path": [], "paragraph_index": ordinal}
        connection.execute(
            "INSERT INTO source_fragments VALUES (?, 'source_version_structure', ?, "
            "'paragraph', ?, ?, ?, ?)",
            (
                fragment_id,
                ordinal,
                text,
                sha256_hex(text.encode()),
                canonical_json_bytes(address).decode(),
                canonical_sha256_hex(address),
            ),
        )
        fragment_ids.append(fragment_id)
    connection.execute(
        "INSERT INTO collection_memberships VALUES (?, 'source_structure', 'active', ?)",
        (collection_id, NOW_TEXT),
    )
    return tuple(fragment_ids)


def structure_request(
    snapshot_id: str,
    policy: AccessPolicySnapshot,
    scope: RequestScope,
) -> StructureGenerationRequest:
    """Build the canonical synthetic StructureGenerationRequest."""

    return StructureGenerationRequest(
        snapshot_id,
        policy.access_policy_id,
        scope,
        "transcript_section",
        "1.0",
    )
