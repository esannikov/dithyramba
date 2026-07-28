"""Boundary tests for the public P7 Structure and Relation contracts.

These cases protect the rejection paths that keep stored context explicit,
ordered, source-bound, and reproducible.  They deliberately use only public
contracts: persistence callers must not be able to construct ambiguous values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from dithyramba.access import QueryExclusions, RequestScope
from dithyramba.contracts import canonical_sha256_hex, sha256_hex
from dithyramba.relations import (
    Relation,
    RelationContractError,
    RelationDirection,
    RelationImportReceipt,
    RelationImportRequest,
    RelationImportResult,
    RelationNodeGrounding,
    RelationNodeRef,
    RelationNodeType,
    RelationPath,
    RelationPathReceipt,
    RelationPathRequest,
    RelationPathStep,
    RelationProposal,
    RelationType,
)
from dithyramba.structure import (
    StructureContractError,
    StructureGeneration,
    StructureGenerationRequest,
    StructureReadReceipt,
    StructureReadReceiptItem,
    StructureUnit,
    StructureUnitKind,
    StructureUnitMember,
    StructureUnitProposal,
    StructureUnitText,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _scope() -> RequestScope:
    return RequestScope(
        library_id="library_one",
        snapshot_hash=HASH_A,
        purpose="research",
        collection_ids=("collection_one",),
        exclusions=QueryExclusions(),
    )


def _structure_unit(
    *,
    structure_unit_id: str = "structure_unit_one",
    generation_id: str = "structure_generation_one",
    ordinal: int = 3,
) -> StructureUnit:
    member = StructureUnitMember("fragment_one", 0, ordinal, HASH_B)
    semantic = {
        "schema": "dithyramba.structure_unit/1.0",
        "source_id": "source_one",
        "source_version_id": "source_version_one",
        "kind": "section",
        "label": "Section",
        "members": [member.payload()],
        "boundary_hash": HASH_C,
    }
    return StructureUnit(
        structure_unit_id=structure_unit_id,
        generation_id=generation_id,
        source_id="source_one",
        source_version_id="source_version_one",
        kind=StructureUnitKind.SECTION,
        label="Section",
        members=(member,),
        boundary_hash=HASH_C,
        content_hash=canonical_sha256_hex(semantic),
    )


def _structure_receipt(
    *,
    structure_unit_id: str = "structure_unit_one",
    text: str = "Evidence",
) -> StructureReadReceipt:
    return StructureReadReceipt(
        receipt_id="structure_read_one",
        structure_unit_id=structure_unit_id,
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        access_policy_id="policy_one",
        policy_hash=HASH_A,
        scope_hash=HASH_B,
        permitted_set_hash=HASH_C,
        delimiter="\n\n",
        rendered_text_sha256=sha256_hex(text.encode()),
        items=(StructureReadReceiptItem("fragment_one", 0, HASH_B, 0, len(text)),),
    )


def _structure_generation(*units: StructureUnit) -> StructureGeneration:
    return StructureGeneration(
        generation_id="structure_generation_one",
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        access_policy_id="policy_one",
        policy_hash=HASH_A,
        scope_hash=HASH_B,
        permitted_set_hash=HASH_C,
        profile_id="section_profile",
        profile_version="1.0",
        units=tuple(units) or (_structure_unit(),),
        generation_hash=HASH_D,
    )


@pytest.mark.parametrize(
    "factory, message",
    [
        (
            lambda: StructureGenerationRequest(
                "snapshot_one",
                "policy_one",
                cast(Any, object()),
                "section_profile",
                "1.0",
            ),
            "scope must be",
        ),
        (
            lambda: StructureGenerationRequest(
                "snapshot_one", "policy_one", _scope(), "section_profile", " 1.0"
            ),
            "profile_version",
        ),
        (
            lambda: StructureUnitProposal(cast(Any, "section"), ("fragment_one",), "Section"),
            "kind",
        ),
        (
            lambda: StructureUnitProposal(
                StructureUnitKind.SECTION, cast(Any, ["fragment_one"]), "Section"
            ),
            "tuple",
        ),
        (
            lambda: StructureUnitProposal(
                StructureUnitKind.SECTION, tuple(f"fragment_{index}" for index in range(129))
            ),
            "at most 128",
        ),
        (
            lambda: StructureUnitProposal(
                StructureUnitKind.SECTION, ("fragment_one",), " Section "
            ),
            "label",
        ),
        (
            lambda: replace(_structure_unit(), kind=cast(Any, "section")),
            "kind",
        ),
        (
            lambda: replace(_structure_unit(), label=""),
            "label",
        ),
        (
            lambda: replace(_structure_unit(), members=cast(Any, [])),
            "non-empty tuple",
        ),
        (
            lambda: replace(_structure_unit(), members=(cast(Any, "fragment_one"),)),
            "StructureUnitMember",
        ),
        (
            lambda: StructureReadReceiptItem("fragment_one", 0, HASH_A, 5, 5),
            "greater than",
        ),
        (
            lambda: replace(_structure_receipt(), delimiter="\n"),
            "frozen",
        ),
        (
            lambda: replace(_structure_receipt(), items=cast(Any, [])),
            "member items",
        ),
        (
            lambda: replace(_structure_receipt(), items=(cast(Any, "not-a-receipt-item"),)),
            "StructureReadReceiptItem",
        ),
        (
            lambda: replace(
                _structure_receipt(),
                items=(StructureReadReceiptItem("fragment_one", 1, HASH_A, 0, 1),),
            ),
            "member order",
        ),
        (
            lambda: StructureUnitText(cast(Any, object()), "Evidence", _structure_receipt()),
            "structure_unit",
        ),
        (
            lambda: StructureUnitText(_structure_unit(), "", _structure_receipt()),
            "non-empty",
        ),
        (
            lambda: StructureUnitText(_structure_unit(), "Evidence", cast(Any, object())),
            "receipt must be",
        ),
        (
            lambda: StructureUnitText(
                _structure_unit(),
                "Evidence",
                _structure_receipt(structure_unit_id="structure_unit_other"),
            ),
            "IDs differ",
        ),
    ],
)
def test_structure_contracts_reject_ambiguous_or_unverifiable_values(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises(StructureContractError, match=message):
        factory()


def test_structure_generation_requires_owned_canonical_units() -> None:
    valid = _structure_unit()
    other_generation = _structure_unit(generation_id="structure_generation_other")
    second = _structure_unit(structure_unit_id="structure_unit_alpha", ordinal=4)

    with pytest.raises(StructureContractError, match="between 1 and 5,000"):
        replace(_structure_generation(valid), units=())

    invalid_cases: tuple[tuple[object, str], ...] = (
        ((cast(Any, "not-a-unit"),), "StructureUnit values"),
        ((other_generation,), "generation IDs"),
        ((valid, second), "canonical ID order"),
    )
    for units, message in invalid_cases:
        with pytest.raises(StructureContractError, match=message):
            _structure_generation(*cast(Any, units))

    with pytest.raises(StructureContractError, match="profile_id"):
        replace(_structure_generation(valid), profile_id="Not Canonical")


def _nodes() -> tuple[RelationNodeRef, RelationNodeRef, RelationNodeRef]:
    return (
        RelationNodeRef(RelationNodeType.STATEMENT, "statement_one"),
        RelationNodeRef(RelationNodeType.CONCEPT, "concept_one"),
        RelationNodeRef(RelationNodeType.STATEMENT, "statement_two"),
    )


def _proposal() -> RelationProposal:
    first, second, _third = _nodes()
    return RelationProposal(
        RelationType.DEFINES,
        first,
        second,
        ("evidence_one",),
        "The source statement defines the concept.",
        "human_import/1.0",
    )


def _relation() -> Relation:
    proposal = _proposal()
    return Relation(
        relation_id="relation_one",
        import_run_id="relation_run_one",
        library_id="library_one",
        relation_type=proposal.relation_type,
        subject=proposal.subject,
        object_node=proposal.object_node,
        evidence_link_ids=proposal.evidence_link_ids,
        reason_text=proposal.reason_text,
        extraction_method=proposal.extraction_method,
        valid_start=proposal.valid_start,
        valid_end=proposal.valid_end,
        content_hash=proposal.content_hash,
    )


def _import_receipt(*relation_ids: str) -> RelationImportReceipt:
    return RelationImportReceipt(
        import_run_id="relation_run_one",
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        access_policy_id="policy_one",
        policy_hash=HASH_A,
        scope_hash=HASH_B,
        permitted_set_hash=HASH_C,
        code_version="p7",
        profile_version="relations/1.0",
        input_hash=HASH_A,
        output_hash=HASH_B,
        relation_ids=tuple(relation_ids) or ("relation_one",),
    )


def _path_request(*, max_paths: int = 10) -> RelationPathRequest:
    return RelationPathRequest(
        "snapshot_one",
        "policy_one",
        _scope(),
        (_nodes()[0],),
        (RelationType.DEFINES,),
        RelationDirection.OUTGOING,
        2,
        max_paths,
    )


def _path(*, relation_id: str = "relation_one") -> RelationPath:
    first, second, _third = _nodes()
    return RelationPath(
        0,
        first,
        (
            RelationPathStep(
                relation_id,
                RelationType.DEFINES,
                first,
                second,
                ("evidence_one",),
            ),
        ),
    )


def _path_receipt(*paths: RelationPath) -> RelationPathReceipt:
    return RelationPathReceipt(
        receipt_id="relation_path_one",
        library_id="library_one",
        corpus_snapshot_id="snapshot_one",
        access_policy_id="policy_one",
        policy_hash=HASH_A,
        scope_hash=HASH_B,
        permitted_set_hash=HASH_C,
        request=_path_request(),
        paths=tuple(paths) or (_path(),),
    )


def test_every_relation_node_type_enforces_its_explicit_id_namespace() -> None:
    expected = {
        RelationNodeType.STATEMENT: "statement_",
        RelationNodeType.CONCEPT: "concept_",
        RelationNodeType.CONCEPT_MEANING: "concept_meaning_",
        RelationNodeType.ENTITY: "entity_",
        RelationNodeType.VOICE: "voice_",
        RelationNodeType.TIME_CONTEXT: "time_context_",
        RelationNodeType.STRUCTURE_UNIT: "structure_unit_",
    }

    for node_type, prefix in expected.items():
        node = RelationNodeRef(node_type, f"{prefix}one")
        assert node.payload() == {"node_type": node_type.value, "node_id": f"{prefix}one"}


def test_relation_node_grounding_is_sorted_exact_and_non_empty() -> None:
    node = RelationNodeRef(RelationNodeType.STATEMENT, "statement_one")
    grounding = RelationNodeGrounding(
        node,
        ("fragment_two", "fragment_one"),
    )

    assert grounding.source_fragment_ids == ("fragment_one", "fragment_two")
    assert grounding.payload() == {
        "node": node.payload(),
        "source_fragment_ids": ["fragment_one", "fragment_two"],
    }

    invalid_cases = (
        (cast(Any, object()), ("fragment_one",), "node must be"),
        (node, (), "cannot be empty"),
        (node, cast(Any, ["fragment_one"]), "tuple"),
        (node, ("fragment_one", "fragment_one"), "unique"),
        (node, ("source_one",), "fragment_"),
    )
    for invalid_node, fragment_ids, message in invalid_cases:
        with pytest.raises(RelationContractError, match=message):
            RelationNodeGrounding(invalid_node, fragment_ids)


@pytest.mark.parametrize(
    "factory, message",
    [
        (
            lambda: RelationNodeRef(cast(Any, "statement"), "statement_one"),
            "node_type",
        ),
        (
            lambda: RelationImportRequest(
                "snapshot_one",
                "policy_one",
                cast(Any, object()),
                "p7",
                "relations/1.0",
            ),
            "scope must be",
        ),
        (
            lambda: RelationImportRequest(
                "snapshot_one", "policy_one", _scope(), "p7", "relations/1.0", 0
            ),
            "max_relations",
        ),
        (
            lambda: replace(_proposal(), relation_type=cast(Any, "defines")),
            "relation_type",
        ),
        (
            lambda: replace(_proposal(), subject=cast(Any, object())),
            "endpoints",
        ),
        (
            lambda: replace(_proposal(), valid_start=" 2026"),
            "valid_start",
        ),
        (
            lambda: replace(_relation(), content_hash=HASH_A),
            "content_hash does not match",
        ),
        (
            lambda: RelationImportReceipt(
                "relation_run_one",
                "library_one",
                "snapshot_one",
                "policy_one",
                HASH_A,
                HASH_B,
                HASH_C,
                "p7",
                "relations/1.0",
                HASH_A,
                HASH_B,
                (),
            ),
            "cannot be empty",
        ),
        (
            lambda: RelationPathRequest(
                "snapshot_one",
                "policy_one",
                cast(Any, object()),
                (_nodes()[0],),
                (RelationType.DEFINES,),
            ),
            "scope must be",
        ),
        (
            lambda: RelationPathRequest(
                "snapshot_one",
                "policy_one",
                _scope(),
                cast(Any, []),
                (RelationType.DEFINES,),
            ),
            "seeds must contain",
        ),
        (
            lambda: RelationPathRequest(
                "snapshot_one",
                "policy_one",
                _scope(),
                (cast(Any, "statement_one"),),
                (RelationType.DEFINES,),
            ),
            "RelationNodeRef",
        ),
        (
            lambda: RelationPathRequest(
                "snapshot_one", "policy_one", _scope(), (_nodes()[0],), cast(Any, ())
            ),
            "non-empty tuple",
        ),
        (
            lambda: RelationPathRequest(
                "snapshot_one",
                "policy_one",
                _scope(),
                (_nodes()[0],),
                (cast(Any, "defines"),),
            ),
            "RelationType",
        ),
        (
            lambda: RelationPathRequest(
                "snapshot_one",
                "policy_one",
                _scope(),
                (_nodes()[0],),
                (RelationType.DEFINES,),
                cast(Any, "outgoing"),
            ),
            "direction",
        ),
        (
            lambda: RelationPathStep(
                "relation_one",
                cast(Any, "defines"),
                _nodes()[0],
                _nodes()[1],
                ("evidence_one",),
            ),
            "relation_type",
        ),
        (
            lambda: RelationPathStep(
                "relation_one",
                RelationType.DEFINES,
                cast(Any, object()),
                _nodes()[1],
                ("evidence_one",),
            ),
            "path endpoints",
        ),
        (
            lambda: RelationPathStep(
                "relation_one",
                RelationType.DEFINES,
                _nodes()[0],
                _nodes()[0],
                ("evidence_one",),
            ),
            "self-loop",
        ),
        (
            lambda: RelationPath(0, cast(Any, object()), (_path().steps[0],)),
            "seed must be",
        ),
        (
            lambda: RelationPath(0, _nodes()[0], cast(Any, [])),
            "1 to 3",
        ),
        (
            lambda: RelationPath(0, _nodes()[0], (cast(Any, "step"),)),
            "RelationPathStep",
        ),
        (
            lambda: RelationPath(
                0,
                _nodes()[0],
                (
                    RelationPathStep(
                        "relation_one",
                        RelationType.DEFINES,
                        _nodes()[1],
                        _nodes()[2],
                        ("evidence_one",),
                    ),
                ),
            ),
            "not contiguous",
        ),
        (
            lambda: replace(_path_receipt(), request=cast(Any, object())),
            "request must be",
        ),
        (
            lambda: replace(_path_receipt(), paths=cast(Any, [_path()])),
            "max_paths",
        ),
        (
            lambda: replace(_path_receipt(), paths=(cast(Any, "path"),)),
            "RelationPath values",
        ),
    ],
)
def test_relation_contracts_reject_ambiguous_or_unverifiable_values(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises(RelationContractError, match=message):
        factory()


def test_relation_results_and_receipts_remain_canonically_ordered() -> None:
    relation = _relation()
    receipt = _import_receipt(relation.relation_id)
    assert RelationImportResult((relation,), receipt).receipt.receipt_hash == receipt.receipt_hash

    for relations, bad_receipt, message in (
        ((), receipt, "cannot be empty"),
        ((cast(Any, "relation"),), receipt, "Relation values"),
        ((relation,), _import_receipt("relation_other"), "differs"),
    ):
        with pytest.raises(RelationContractError, match=message):
            RelationImportResult(cast(Any, relations), bad_receipt)

    first = _path(relation_id="relation_zed")
    second = _path(relation_id="relation_alpha")
    assert first.path_hash != second.path_hash
    unsorted = tuple(sorted((first, second), key=lambda item: item.path_hash, reverse=True))
    with pytest.raises(RelationContractError, match="canonical hash order"):
        replace(_path_receipt(), paths=unsorted)

    bounded = replace(_path_receipt(), request=_path_request(max_paths=1))
    with pytest.raises(RelationContractError, match="max_paths"):
        replace(bounded, paths=tuple(sorted((first, second), key=lambda item: item.path_hash)))
