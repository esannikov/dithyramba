"""Pure Markdown StructureUnit proposal generation tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Literal, cast

import pytest

from dithyramba.access import RequestScope
from dithyramba.contracts import canonical_sha256_hex
from dithyramba.ingest import FragmentKind
from dithyramba.structure import (
    MarkdownFragmentDescriptor,
    MarkdownStructureGenerationResult,
    MarkdownStructureProfile,
    MarkdownStructureProposalBuild,
    StructureContractError,
    StructureGeneration,
    StructureGenerationBuildStatus,
    StructureGenerationRequest,
    StructureInputExclusion,
    StructureInputExclusionReason,
    StructureProposalOmission,
    StructureProposalOmissionReason,
    StructureUnit,
    StructureUnitKind,
    StructureUnitMember,
    StructureUnitProposal,
    build_markdown_structure_proposals,
)


def _fragment(
    ordinal: int,
    heading_path: tuple[str, ...],
    *,
    source: str = "source_alpha",
    version: str = "source_version_alpha",
    fragment_id: str | None = None,
    kind: FragmentKind = FragmentKind.PARAGRAPH,
) -> MarkdownFragmentDescriptor:
    return MarkdownFragmentDescriptor(
        source_id=source,
        source_version_id=version,
        source_fragment_id=fragment_id or f"fragment_{version}_{ordinal}",
        ordinal=ordinal,
        fragment_kind=kind,
        heading_path=heading_path,
    )


def test_profile_is_versioned_hashed_and_strictly_bounded() -> None:
    profile = MarkdownStructureProfile(maximum_members=7)

    assert profile.payload() == {
        "schema": "dithyramba.markdown_structure_profile/1.0",
        "profile_id": "markdown_exact_heading_path",
        "profile_version": "1.0",
        "minimum_members": 2,
        "maximum_members": 7,
    }
    assert len(profile.profile_hash) == 64
    with pytest.raises(StructureContractError):
        replace(
            profile,
            profile_id=cast(Literal["markdown_exact_heading_path"], "other"),
        )
    with pytest.raises(StructureContractError):
        replace(profile, profile_version=cast(Literal["1.0"], "2.0"))
    with pytest.raises(StructureContractError):
        replace(profile, minimum_members=cast(Literal[2], 1))
    with pytest.raises(StructureContractError, match="maximum_members"):
        MarkdownStructureProfile(maximum_members=129)
    with pytest.raises(StructureContractError, match="maximum_members"):
        MarkdownStructureProfile(maximum_members=cast(int, True))


def test_builder_groups_exact_natural_paths_and_uses_metadata_labels_only() -> None:
    fragments = (
        _fragment(0, ()),
        _fragment(1, ()),
        _fragment(2, ("Part",), kind=FragmentKind.HEADING),
        _fragment(3, ("Part",)),
        _fragment(4, ("Part", "Question"), kind=FragmentKind.HEADING),
        _fragment(5, ("Part", "Question")),
    )

    result = build_markdown_structure_proposals(fragments)

    assert [(item.kind, item.label) for item in result.proposals] == [
        (StructureUnitKind.DOCUMENT, None),
        (StructureUnitKind.SECTION, "Part"),
        (StructureUnitKind.SECTION, "Question"),
    ]
    assert result.proposals[2].source_fragment_ids == (
        "fragment_source_version_alpha_4",
        "fragment_source_version_alpha_5",
    )
    assert result.omissions == ()
    assert "text" not in str(result.payload())
    assert fragments[0].payload()["heading_path"] == []
    assert result.build_hash == build_markdown_structure_proposals(fragments).build_hash


def test_builder_canonicalizes_input_and_separates_source_versions() -> None:
    fragments = (
        _fragment(1, ("Z",), source="source_z", version="source_version_z"),
        _fragment(0, ("A",), source="source_a", version="source_version_a"),
        _fragment(
            0,
            ("Z",),
            source="source_z",
            version="source_version_z",
            kind=FragmentKind.HEADING,
        ),
        _fragment(1, ("A",), source="source_a", version="source_version_a"),
    )

    first = build_markdown_structure_proposals(fragments)
    second = build_markdown_structure_proposals(tuple(reversed(fragments)))

    assert first == second
    assert first.proposals[0].label == "A"
    assert first.proposals[1].label == "Z"
    assert first.payload() == second.payload()


def test_builder_keeps_every_natural_unit_beyond_legacy_five_thousand_limit() -> None:
    fragments = tuple(
        _fragment(
            ordinal,
            (f"Section {source_index}",),
            source=f"source_scale_{source_index:04d}",
            version=f"source_version_scale_{source_index:04d}",
            kind=FragmentKind.HEADING if ordinal == 0 else FragmentKind.PARAGRAPH,
        )
        for source_index in range(5_001)
        for ordinal in range(2)
    )

    result = build_markdown_structure_proposals(fragments)

    assert len(result.proposals) == 5_001
    assert result.omissions == ()
    assert sum(len(item.source_fragment_ids) for item in result.proposals) == 10_002


def test_small_and_oversized_natural_sections_are_explicit_and_never_split() -> None:
    fragments = (
        _fragment(0, ("Single",), kind=FragmentKind.HEADING),
        _fragment(1, ("Large",), kind=FragmentKind.HEADING),
        _fragment(2, ("Large",)),
        _fragment(3, ("Large",)),
    )

    result = build_markdown_structure_proposals(
        fragments, profile=MarkdownStructureProfile(maximum_members=2)
    )

    assert result.proposals == ()
    assert [item.reason for item in result.omissions] == [
        StructureProposalOmissionReason.BELOW_MINIMUM_MEMBERS,
        StructureProposalOmissionReason.OVERSIZED_NATURAL_SECTION,
    ]
    oversized = result.omissions[1]
    assert oversized.source_fragment_ids == (
        "fragment_source_version_alpha_1",
        "fragment_source_version_alpha_2",
        "fragment_source_version_alpha_3",
    )
    assert oversized.member_count == 3
    assert oversized.label == "Large"
    assert oversized.first_ordinal == 1
    assert oversized.last_ordinal == 3


@pytest.mark.parametrize(
    ("fragments", "message"),
    [
        ((), "non-empty tuple"),
        ((_fragment(0, ()), _fragment(2, ())), "ordinal-contiguous"),
        (
            (_fragment(0, ()), _fragment(1, (), fragment_id="fragment_source_version_alpha_0")),
            "duplicate source_fragment_id",
        ),
        (
            (_fragment(0, ()), _fragment(0, (), fragment_id="fragment_distinct")),
            "duplicate ordinal",
        ),
        (
            (
                _fragment(0, (), source="source_a", version="source_version_shared"),
                _fragment(1, (), source="source_b", version="source_version_shared"),
            ),
            "multiple Sources",
        ),
    ],
)
def test_builder_rejects_duplicate_or_noncontiguous_input(
    fragments: tuple[MarkdownFragmentDescriptor, ...], message: str
) -> None:
    with pytest.raises(StructureContractError, match=message):
        build_markdown_structure_proposals(fragments)


@pytest.mark.parametrize(
    "fragment",
    [
        lambda: _fragment(-1, (), fragment_id="fragment_negative"),
        lambda: MarkdownFragmentDescriptor(
            "source_a",
            "source_version_a",
            "fragment_a",
            0,
            cast(FragmentKind, "paragraph"),
            (),
        ),
        lambda: MarkdownFragmentDescriptor(
            "source_a",
            "source_version_a",
            "fragment_a",
            0,
            FragmentKind.PARAGRAPH,
            cast(tuple[str, ...], ["A"]),
        ),
        lambda: _fragment(0, tuple("A" for _ in range(65))),
        lambda: _fragment(0, ("",)),
        lambda: _fragment(0, (" Cafe\u0301",)),
        lambda: _fragment(0, ("Cafe\u0301",)),
        lambda: _fragment(0, ("A\x00B",)),
        lambda: _fragment(0, ("A" * 501,)),
        lambda: MarkdownFragmentDescriptor(
            "source_a",
            "source_version_a",
            "fragment_a",
            0,
            FragmentKind.PARAGRAPH,
            cast(tuple[str, ...], (7,)),
        ),
        lambda: _fragment(0, (), kind=FragmentKind.HEADING),
        lambda: _fragment(0, (), kind=FragmentKind.PAGE_TEXT),
        lambda: MarkdownFragmentDescriptor(
            "bad", "source_version_a", "fragment_a", 0, FragmentKind.PARAGRAPH, ()
        ),
        lambda: MarkdownFragmentDescriptor(
            "source_a", "source-version-a", "fragment_a", 0, FragmentKind.PARAGRAPH, ()
        ),
    ],
)
def test_fragment_descriptor_rejects_malformed_markdown_metadata(
    fragment: Callable[[], object],
) -> None:
    with pytest.raises(StructureContractError):
        fragment()


def test_output_contracts_reject_inconsistent_manual_construction() -> None:
    proposal = StructureUnitProposal(StructureUnitKind.DOCUMENT, ("fragment_a", "fragment_b"), None)
    omission = StructureProposalOmission(
        reason=StructureProposalOmissionReason.OVERSIZED_NATURAL_SECTION,
        source_id="source_a",
        source_version_id="source_version_a",
        kind=StructureUnitKind.SECTION,
        heading_path=("A",),
        source_fragment_ids=("fragment_b", "fragment_c"),
        first_ordinal=1,
        last_ordinal=2,
    )
    with pytest.raises(StructureContractError, match="reuse"):
        MarkdownStructureProposalBuild(MarkdownStructureProfile(), (proposal,), (omission,))
    with pytest.raises(StructureContractError, match="contiguous"):
        replace(omission, last_ordinal=4)


def _valid_omission() -> StructureProposalOmission:
    return StructureProposalOmission(
        reason=StructureProposalOmissionReason.BELOW_MINIMUM_MEMBERS,
        source_id="source_a",
        source_version_id="source_version_a",
        kind=StructureUnitKind.DOCUMENT,
        heading_path=(),
        source_fragment_ids=("fragment_a",),
        first_ordinal=0,
        last_ordinal=0,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(_valid_omission(), reason=cast(StructureProposalOmissionReason, "other")),
        lambda: replace(_valid_omission(), kind=cast(StructureUnitKind, "document")),
        lambda: replace(_valid_omission(), heading_path=cast(tuple[str, ...], ["A"])),
        lambda: replace(_valid_omission(), heading_path=("A",)),
        lambda: replace(_valid_omission(), kind=StructureUnitKind.SECTION, heading_path=()),
        lambda: replace(_valid_omission(), source_fragment_ids=()),
        lambda: replace(
            _valid_omission(), source_fragment_ids=("fragment_a", "fragment_a"), last_ordinal=1
        ),
        lambda: replace(_valid_omission(), first_ordinal=-1),
        lambda: replace(_valid_omission(), last_ordinal=-1),
    ],
)
def test_omission_contract_rejects_inconsistent_metadata(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(StructureContractError):
        factory()


def test_document_omission_payload_has_no_fabricated_label() -> None:
    omission = _valid_omission()

    assert omission.label is None
    assert omission.payload()["label"] is None


def test_build_and_builder_contracts_reject_untyped_values() -> None:
    proposal = StructureUnitProposal(StructureUnitKind.DOCUMENT, ("fragment_a", "fragment_b"), None)
    omission = _valid_omission()
    profile = MarkdownStructureProfile()

    with pytest.raises(StructureContractError, match="profile"):
        MarkdownStructureProposalBuild(
            cast(MarkdownStructureProfile, object()), (proposal,), (omission,)
        )
    with pytest.raises(StructureContractError, match="proposals"):
        MarkdownStructureProposalBuild(
            profile,
            cast(tuple[StructureUnitProposal, ...], [proposal]),
            (omission,),
        )
    with pytest.raises(StructureContractError, match="omissions"):
        MarkdownStructureProposalBuild(
            profile,
            (proposal,),
            cast(tuple[StructureProposalOmission, ...], [omission]),
        )
    with pytest.raises(StructureContractError, match="profile"):
        build_markdown_structure_proposals(
            (_fragment(0, ()),), profile=cast(MarkdownStructureProfile, object())
        )
    with pytest.raises(StructureContractError, match="non-empty tuple"):
        build_markdown_structure_proposals(cast(tuple[MarkdownFragmentDescriptor, ...], []))
    with pytest.raises(StructureContractError, match="MarkdownFragmentDescriptor"):
        build_markdown_structure_proposals(
            cast(tuple[MarkdownFragmentDescriptor, ...], (object(),))
        )


def _request() -> StructureGenerationRequest:
    profile = MarkdownStructureProfile()
    return StructureGenerationRequest(
        "snapshot_one",
        "policy_one",
        RequestScope(
            library_id="library_one",
            snapshot_hash="a" * 64,
            purpose="research",
            collection_ids=("collection_one",),
        ),
        profile.profile_id,
        profile.profile_version,
    )


def _generation() -> StructureGeneration:
    request = _request()
    members = (
        StructureUnitMember("fragment_one", 0, 0, "b" * 64),
        StructureUnitMember("fragment_two", 1, 1, "c" * 64),
    )
    semantic = {
        "schema": "dithyramba.structure_unit/1.0",
        "source_id": "source_one",
        "source_version_id": "source_version_one",
        "kind": "section",
        "label": "Section",
        "members": [item.payload() for item in members],
        "boundary_hash": "d" * 64,
    }
    unit = StructureUnit(
        structure_unit_id="structure_unit_one",
        generation_id="structure_generation_one",
        source_id="source_one",
        source_version_id="source_version_one",
        kind=StructureUnitKind.SECTION,
        label="Section",
        members=members,
        boundary_hash="d" * 64,
        content_hash=canonical_sha256_hex(semantic),
    )
    return StructureGeneration(
        generation_id="structure_generation_one",
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        access_policy_id="policy_one",
        policy_hash="e" * 64,
        scope_hash=request.scope_hash,
        permitted_set_hash="f" * 64,
        profile_id=request.profile_id,
        profile_version=request.profile_version,
        units=(unit,),
        generation_hash="1" * 64,
    )


def _exclusion(
    fragment_id: str = "fragment_excluded",
    *,
    source: str = "source_a",
) -> StructureInputExclusion:
    return StructureInputExclusion(
        source_id=source,
        source_version_id=f"source_version_{source}",
        source_fragment_id=fragment_id,
        ordinal=0,
        fragment_kind=FragmentKind.PAGE_TEXT,
        address_kind="pdf",
        reason=StructureInputExclusionReason.UNSUPPORTED_SOURCE_ADDRESS,
    )


def test_generation_result_binds_exact_build_generation_and_exclusions() -> None:
    build = MarkdownStructureProposalBuild(
        MarkdownStructureProfile(),
        (
            StructureUnitProposal(
                StructureUnitKind.SECTION,
                ("fragment_one", "fragment_two"),
                "Section",
            ),
        ),
        (),
    )
    result = MarkdownStructureGenerationResult(_request(), build, _generation(), (_exclusion(),))

    assert result.status is StructureGenerationBuildStatus.PERSISTED
    assert result.payload()["generation"] == {
        "generation_id": "structure_generation_one",
        "generation_hash": "1" * 64,
    }
    assert result.payload()["build_hash"] == build.build_hash
    assert len(result.result_hash) == 64

    empty = MarkdownStructureGenerationResult(
        _request(), MarkdownStructureProposalBuild(MarkdownStructureProfile(), (), ()), None
    )
    assert empty.status is StructureGenerationBuildStatus.NO_PROPOSALS
    assert empty.payload()["generation"] is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StructureInputExclusion(
            "source_a",
            "source_version_a",
            "fragment_a",
            -1,
            FragmentKind.PAGE_TEXT,
            "pdf",
            StructureInputExclusionReason.UNSUPPORTED_SOURCE_ADDRESS,
        ),
        lambda: replace(_exclusion(), fragment_kind=cast(FragmentKind, "page_text")),
        lambda: replace(_exclusion(), address_kind=" "),
        lambda: replace(_exclusion(), reason=cast(StructureInputExclusionReason, "unsupported")),
    ],
)
def test_input_exclusion_contract_rejects_malformed_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(StructureContractError):
        factory()


def test_generation_result_contract_rejects_unbound_artifacts() -> None:
    profile = MarkdownStructureProfile()
    empty = MarkdownStructureProposalBuild(profile, (), ())
    proposal_build = MarkdownStructureProposalBuild(
        profile,
        (
            StructureUnitProposal(
                StructureUnitKind.SECTION,
                ("fragment_one", "fragment_two"),
                "Section",
            ),
        ),
        (),
    )

    with pytest.raises(StructureContractError, match="request"):
        MarkdownStructureGenerationResult(cast(StructureGenerationRequest, object()), empty, None)
    with pytest.raises(StructureContractError, match="build"):
        MarkdownStructureGenerationResult(
            _request(), cast(MarkdownStructureProposalBuild, object()), None
        )
    with pytest.raises(StructureContractError, match="profile identities"):
        MarkdownStructureGenerationResult(
            replace(_request(), profile_id="other_profile"), empty, None
        )
    with pytest.raises(StructureContractError, match="exclusions"):
        MarkdownStructureGenerationResult(
            _request(), empty, None, cast(tuple[StructureInputExclusion, ...], [_exclusion()])
        )
    with pytest.raises(StructureContractError, match="canonical order"):
        MarkdownStructureGenerationResult(
            _request(),
            empty,
            None,
            (
                _exclusion("fragment_z", source="source_z"),
                _exclusion("fragment_a", source="source_a"),
            ),
        )
    with pytest.raises(StructureContractError, match="unique"):
        MarkdownStructureGenerationResult(_request(), empty, None, (_exclusion(), _exclusion()))
    with pytest.raises(StructureContractError, match="excluded fragments"):
        MarkdownStructureGenerationResult(
            _request(), proposal_build, _generation(), (_exclusion("fragment_one"),)
        )
    with pytest.raises(StructureContractError, match="persisted"):
        MarkdownStructureGenerationResult(_request(), proposal_build, None)
    with pytest.raises(StructureContractError, match="empty proposal"):
        MarkdownStructureGenerationResult(_request(), empty, _generation())
    with pytest.raises(StructureContractError, match="differs"):
        MarkdownStructureGenerationResult(
            _request(), proposal_build, replace(_generation(), corpus_snapshot_id="snapshot_two")
        )
