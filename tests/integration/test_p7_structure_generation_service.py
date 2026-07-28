"""Policy-safe metadata-only Markdown StructureUnit generation tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    QueryExclusions,
    RequestScope,
)
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.ingest import (
    FragmentKind,
    MarkdownSourceAddress,
    PdfSourceAddress,
)
from dithyramba.library import LibraryConfig
from dithyramba.persistence import SQLiteStructureRepository
from dithyramba.persistence.repository import LibraryRepository, initialize_library
from dithyramba.structure import (
    MarkdownStructureProfile,
    StructureAuthorizationError,
    StructureGenerationBuildStatus,
    StructureGenerationRequest,
    StructureInputExclusionReason,
    StructureIntegrityError,
    StructureProposalOmissionReason,
)

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-21T16:00:00.000000Z"


def _clock() -> datetime:
    return NOW


def _context(
    tmp_path: Path,
    *,
    markdown_address_override: dict[int, str] | None = None,
) -> tuple[
    LibraryRepository,
    SQLiteStructureRepository,
    tuple[str, ...],
    tuple[str, ...],
    str,
    AccessPolicySnapshot,
    RequestScope,
]:
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    source_root.mkdir()
    repository = initialize_library(
        LibraryConfig(name="Structure generation"), data_root=data_root, clock=_clock
    )
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="Corpus",
            kind=CollectionKind.CORPUS,
            roots=(build_collection_root(source_root, data_root=data_root),),
        )
    )
    markdown_ids = _seed_markdown_source(
        repository,
        collection.config.collection_id,
        address_override=markdown_address_override or {},
    )
    pdf_ids = _seed_pdf_source(repository, collection.config.collection_id)
    snapshot = repository.freeze_snapshot((collection.config.collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id="policy_structure_generation",
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
        markdown_ids,
        pdf_ids,
        snapshot.corpus_snapshot_id,
        policy,
        scope,
    )


def _seed_markdown_source(
    repository: LibraryRepository,
    collection_id: str,
    *,
    address_override: dict[int, str],
) -> tuple[str, ...]:
    specifications = (
        (FragmentKind.HEADING, ("Introduction",)),
        (FragmentKind.PARAGRAPH, ("Introduction",)),
        (FragmentKind.PARAGRAPH, ("Introduction",)),
        (FragmentKind.HEADING, ("Introduction", "Question")),
        (FragmentKind.PARAGRAPH, ("Introduction", "Question")),
        (FragmentKind.HEADING, ("Single",)),
    )
    texts = tuple(f"PRIVATE MARKDOWN {ordinal}" for ordinal in range(len(specifications)))
    source_id = "source_structure_markdown"
    version_id = "source_version_structure_markdown"
    _seed_source_shell(
        repository,
        collection_id,
        source_id=source_id,
        source_version_id=version_id,
        family_id="family_structure_markdown",
        media_type="text/markdown",
        texts=texts,
    )
    connection = repository._store.connection
    fragment_ids: list[str] = []
    for ordinal, ((fragment_kind, heading_path), text) in enumerate(
        zip(specifications, texts, strict=True)
    ):
        address = MarkdownSourceAddress(
            heading_path=heading_path,
            line_start=ordinal + 1,
            line_end=ordinal + 1,
            char_start=ordinal * 10,
            char_end=ordinal * 10 + 5,
        ).payload()
        address_json = address_override.get(ordinal, canonical_json_bytes(address).decode("utf-8"))
        fragment_id = f"fragment_structure_markdown_{ordinal}"
        connection.execute(
            """
            INSERT INTO source_fragments(
                source_fragment_id, source_version_id, ordinal, fragment_kind,
                text, text_sha256, source_address_json, address_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fragment_id,
                version_id,
                ordinal,
                fragment_kind.value,
                text,
                sha256_hex(text.encode("utf-8")),
                address_json,
                canonical_sha256_hex(address),
            ),
        )
        fragment_ids.append(fragment_id)
    return tuple(fragment_ids)


def _seed_pdf_source(
    repository: LibraryRepository,
    collection_id: str,
) -> tuple[str, ...]:
    texts = ("PRIVATE PDF 0", "PRIVATE PDF 1")
    source_id = "source_structure_pdf"
    version_id = "source_version_structure_pdf"
    _seed_source_shell(
        repository,
        collection_id,
        source_id=source_id,
        source_version_id=version_id,
        family_id="family_structure_pdf",
        media_type="application/pdf",
        texts=texts,
    )
    connection = repository._store.connection
    fragment_ids: list[str] = []
    for ordinal, text in enumerate(texts):
        address = PdfSourceAddress(
            page=ordinal + 1,
            bbox=("0.000", "0.000", "10.000", "10.000"),
            char_start=0,
            char_end=len(text),
        ).payload()
        fragment_id = f"fragment_structure_pdf_{ordinal}"
        connection.execute(
            """
            INSERT INTO source_fragments(
                source_fragment_id, source_version_id, ordinal, fragment_kind,
                text, text_sha256, source_address_json, address_hash
            ) VALUES (?, ?, ?, 'page_text', ?, ?, ?, ?)
            """,
            (
                fragment_id,
                version_id,
                ordinal,
                text,
                sha256_hex(text.encode("utf-8")),
                canonical_json_bytes(address).decode("utf-8"),
                canonical_sha256_hex(address),
            ),
        )
        fragment_ids.append(fragment_id)
    return tuple(fragment_ids)


def _seed_source_shell(
    repository: LibraryRepository,
    collection_id: str,
    *,
    source_id: str,
    source_version_id: str,
    family_id: str,
    media_type: str,
    texts: tuple[str, ...],
) -> None:
    connection = repository._store.connection
    source_bytes = "\n".join(texts).encode("utf-8")
    content_hash = sha256_hex(source_bytes)
    connection.execute(
        "INSERT OR IGNORE INTO blobs VALUES (?, ?, ?, ?)",
        (content_hash, len(source_bytes), f"aa/{source_id}", NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
        (
            source_id,
            repository.library_id,
            f"file:///{source_id}",
            media_type,
            source_id,
            NOW_TEXT,
        ),
    )
    connection.execute(
        "INSERT INTO source_versions VALUES (?, ?, 1, ?, ?, ?, NULL, "
        "'parser_v1', 'processed', NULL)",
        (source_version_id, source_id, content_hash, len(source_bytes), NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_heads VALUES (?, ?, ?, NULL)",
        (source_id, source_version_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_families VALUES (?, ?, ?, ?)",
        (family_id, repository.library_id, source_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO source_family_members VALUES (?, ?, 'root', NULL, ?)",
        (family_id, source_id, NOW_TEXT),
    )
    connection.execute(
        "INSERT INTO collection_memberships VALUES (?, ?, 'active', ?)",
        (collection_id, source_id, NOW_TEXT),
    )


def _request(
    snapshot_id: str,
    policy: AccessPolicySnapshot,
    scope: RequestScope,
) -> StructureGenerationRequest:
    profile = MarkdownStructureProfile()
    return StructureGenerationRequest(
        corpus_snapshot_id=snapshot_id,
        access_policy_id=policy.access_policy_id,
        scope=scope,
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
    )


def test_metadata_only_generation_groups_natural_markdown_and_excludes_pdf(
    tmp_path: Path,
) -> None:
    repository, structures, markdown_ids, pdf_ids, snapshot_id, policy, scope = _context(tmp_path)
    statements: list[str] = []
    repository._store.connection.set_trace_callback(statements.append)
    try:
        request = _request(snapshot_id, policy, scope)
        prepared_build, prepared_exclusions = structures.prepare_markdown_structure(
            request,
            MarkdownStructureProfile(),
        )
        assert len(prepared_build.proposals) == 2
        assert tuple(item.source_fragment_id for item in prepared_exclusions) == pdf_ids
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM structure_unit_generations"
            ).fetchone()[0]
            == 0
        )
        first = structures.generate_markdown_structure(request, MarkdownStructureProfile())
        repeated = structures.generate_markdown_structure(request, MarkdownStructureProfile())

        assert first == repeated
        assert first.status is StructureGenerationBuildStatus.PERSISTED
        assert first.generation is not None
        assert {unit.label for unit in first.generation.units} == {"Introduction", "Question"}
        assert [proposal.source_fragment_ids for proposal in first.build.proposals] == [
            markdown_ids[:3],
            markdown_ids[3:5],
        ]
        assert tuple(item.reason for item in first.build.omissions) == (
            StructureProposalOmissionReason.BELOW_MINIMUM_MEMBERS,
        )
        assert first.build.omissions[0].source_fragment_ids == (markdown_ids[5],)
        assert tuple(item.source_fragment_id for item in first.exclusions) == pdf_ids
        assert all(
            item.reason is StructureInputExclusionReason.UNSUPPORTED_SOURCE_ADDRESS
            for item in first.exclusions
        )
        assert first.result_hash == repeated.result_hash
        metadata_queries = [
            statement for statement in statements if "fragment.source_address_json" in statement
        ]
        assert metadata_queries
        assert all("fragment.text," not in statement for statement in metadata_queries)
        assert repository._fragment_text_read_depth == 0
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM structure_unit_generations"
            ).fetchone()[0]
            == 1
        )
    finally:
        repository._store.connection.set_trace_callback(None)
        repository.close()


def test_fragment_denial_refuses_partial_markdown_structure_and_returns_no_proposals(
    tmp_path: Path,
) -> None:
    repository, structures, markdown_ids, pdf_ids, snapshot_id, policy, scope = _context(tmp_path)
    try:
        denied_scope = replace(
            scope,
            exclusions=QueryExclusions(source_fragment_ids=(markdown_ids[0],)),
        )
        result = structures.generate_markdown_structure(
            _request(snapshot_id, policy, denied_scope),
            MarkdownStructureProfile(),
        )

        assert result.status is StructureGenerationBuildStatus.NO_PROPOSALS
        assert result.generation is None
        assert result.build.proposals == ()
        assert result.build.omissions == ()
        fragment_filtered = tuple(
            item
            for item in result.exclusions
            if item.reason is StructureInputExclusionReason.FRAGMENT_LEVEL_EXCLUSION_PRESENT
        )
        assert {item.source_fragment_id for item in fragment_filtered} == set(markdown_ids) - {
            markdown_ids[0]
        }
        assert markdown_ids[0] not in {item.source_fragment_id for item in result.exclusions}
        assert {
            item.source_fragment_id
            for item in result.exclusions
            if item.reason is StructureInputExclusionReason.UNSUPPORTED_SOURCE_ADDRESS
        } == set(pdf_ids)
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM structure_unit_generations"
            ).fetchone()[0]
            == 0
        )
    finally:
        repository.close()


def test_empty_permitted_manifest_returns_typed_no_proposals_without_writes(
    tmp_path: Path,
) -> None:
    repository, structures, _markdown_ids, _pdf_ids, snapshot_id, policy, scope = _context(tmp_path)
    try:
        empty_scope = replace(
            scope,
            exclusions=QueryExclusions(
                source_ids=("source_structure_markdown", "source_structure_pdf")
            ),
        )
        result = structures.generate_markdown_structure(
            _request(snapshot_id, policy, empty_scope), MarkdownStructureProfile()
        )

        assert result.status is StructureGenerationBuildStatus.NO_PROPOSALS
        assert result.generation is None
        assert result.build.proposals == ()
        assert result.build.omissions == ()
        assert result.exclusions == ()
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM structure_unit_generations"
            ).fetchone()[0]
            == 0
        )
    finally:
        repository.close()


def test_generation_service_requires_exact_typed_request_and_profile(tmp_path: Path) -> None:
    repository, structures, _markdown_ids, _pdf_ids, snapshot_id, policy, scope = _context(tmp_path)
    try:
        with pytest.raises(TypeError, match="request"):
            structures.generate_markdown_structure(
                cast(StructureGenerationRequest, object()), MarkdownStructureProfile()
            )
        with pytest.raises(TypeError, match="profile"):
            structures.generate_markdown_structure(
                _request(snapshot_id, policy, scope),
                cast(MarkdownStructureProfile, object()),
            )
    finally:
        repository.close()


@pytest.mark.parametrize(
    "override",
    [
        '{ "kind": "markdown" }',
        canonical_json_bytes(
            {
                "schema": "dithyramba.source_address/1.0",
                "kind": "markdown",
                "heading_path": ["Introduction"],
                "line_start": 2,
                "line_end": 2,
                "char_start": 10,
                "char_end": 15,
                "extra": True,
            }
        ).decode("utf-8"),
    ],
)
def test_tampered_or_noncanonical_markdown_address_fails_before_persistence(
    tmp_path: Path,
    override: str,
) -> None:
    repository, structures, _markdown_ids, _pdf_ids, snapshot_id, policy, scope = _context(
        tmp_path,
        markdown_address_override={1: override},
    )
    try:
        with pytest.raises(StructureIntegrityError, match="SourceAddress"):
            structures.generate_markdown_structure(
                _request(snapshot_id, policy, scope), MarkdownStructureProfile()
            )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM structure_unit_generations"
            ).fetchone()[0]
            == 0
        )
    finally:
        repository.close()


def test_request_snapshot_policy_scope_and_profile_mismatches_fail_closed(
    tmp_path: Path,
) -> None:
    repository, structures, _markdown_ids, _pdf_ids, snapshot_id, policy, scope = _context(tmp_path)
    try:
        request = _request(snapshot_id, policy, scope)
        wrong_profile_request = replace(request, profile_id="another_profile")
        with pytest.raises(StructureAuthorizationError, match="profile"):
            structures.generate_markdown_structure(
                wrong_profile_request, MarkdownStructureProfile()
            )

        wrong_snapshot_scope = replace(scope, snapshot_hash="f" * 64)
        with pytest.raises(StructureAuthorizationError, match="ID/hash"):
            structures.generate_markdown_structure(
                _request(snapshot_id, policy, wrong_snapshot_scope),
                MarkdownStructureProfile(),
            )

        with pytest.raises(StructureAuthorizationError):
            structures.generate_markdown_structure(
                replace(request, access_policy_id="policy_missing"),
                MarkdownStructureProfile(),
            )

        wrong_library_scope = replace(scope, library_id="library_other")
        with pytest.raises(StructureAuthorizationError, match="another Library"):
            structures.generate_markdown_structure(
                replace(request, scope=wrong_library_scope),
                MarkdownStructureProfile(),
            )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM structure_unit_generations"
            ).fetchone()[0]
            == 0
        )
    finally:
        repository.close()
