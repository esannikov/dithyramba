"""Contract tests for honest broader StructureUnit context."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dithyramba.access import QueryExclusions, RequestScope
from dithyramba.contracts import canonical_sha256_hex, sha256_hex
from dithyramba.structure import (
    StructureContractError,
    StructureGenerationRequest,
    StructureReadReceipt,
    StructureReadReceiptItem,
    StructureUnit,
    StructureUnitKind,
    StructureUnitMember,
    StructureUnitProposal,
    StructureUnitText,
)

HASH = "a" * 64


def _scope() -> RequestScope:
    return RequestScope(
        library_id="library_one",
        snapshot_hash=HASH,
        purpose="research",
        collection_ids=("collection_one",),
        exclusions=QueryExclusions(),
    )


def _unit() -> StructureUnit:
    members = (
        StructureUnitMember("fragment_one", 0, 4, "b" * 64),
        StructureUnitMember("fragment_two", 1, 5, "c" * 64),
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
    return StructureUnit(
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


def test_request_and_proposal_preserve_explicit_member_order() -> None:
    request = StructureGenerationRequest(
        "snapshot_one", "policy_one", _scope(), "transcript_window", "1.0"
    )
    proposal = StructureUnitProposal(
        StructureUnitKind.SECTION,
        ("fragment_two", "fragment_one"),
        "Named section",
    )

    assert request.scope_hash == request.scope.exclusion_hash
    assert proposal.source_fragment_ids == ("fragment_two", "fragment_one")
    assert proposal.payload()["kind"] == "section"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StructureUnitProposal(StructureUnitKind.SECTION, ()),
        lambda: StructureUnitProposal(StructureUnitKind.SECTION, ("fragment_one", "fragment_one")),
        lambda: StructureGenerationRequest(
            "snapshot_one", "policy_one", _scope(), "Bad Profile", "1.0"
        ),
        lambda: StructureUnitMember("fragment_one", True, 0, HASH),
        lambda: replace(_unit(), members=tuple(reversed(_unit().members))),
        lambda: replace(
            _unit(),
            members=(
                StructureUnitMember("fragment_one", 0, 4, "b" * 64),
                StructureUnitMember("fragment_two", 1, 6, "c" * 64),
            ),
        ),
    ],
)
def test_malformed_structure_contracts_fail(factory: object) -> None:
    with pytest.raises(StructureContractError):
        factory()  # type: ignore[operator]


def test_rendered_text_is_bound_to_exact_member_spans() -> None:
    text = "Alpha\n\nBeta"
    items = (
        StructureReadReceiptItem("fragment_one", 0, "b" * 64, 0, 5),
        StructureReadReceiptItem("fragment_two", 1, "c" * 64, 7, 11),
    )
    receipt = StructureReadReceipt(
        receipt_id="structure_read_one",
        structure_unit_id=_unit().structure_unit_id,
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        access_policy_id="policy_one",
        policy_hash="e" * 64,
        scope_hash="f" * 64,
        permitted_set_hash="1" * 64,
        delimiter="\n\n",
        rendered_text_sha256=sha256_hex(text.encode()),
        items=items,
    )

    rendered = StructureUnitText(_unit(), text, receipt)

    assert rendered.text[items[0].rendered_start : items[0].rendered_end] == "Alpha"
    assert rendered.text[items[1].rendered_start : items[1].rendered_end] == "Beta"
    with pytest.raises(StructureContractError):
        StructureUnitText(_unit(), text + "!", receipt)
