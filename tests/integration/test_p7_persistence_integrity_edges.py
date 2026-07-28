"""Integrity and authorization edges for P7 persistence entry points."""

from __future__ import annotations

from typing import Any, cast

import pytest

from dithyramba.access import QueryExclusions, RequestScope
from dithyramba.persistence import SQLiteRelationRepository, SQLiteStructureRepository
from dithyramba.relations import (
    RelationAuthorizationError,
    RelationDirection,
    RelationImportRequest,
    RelationIntegrityError,
    RelationNodeRef,
    RelationNodeType,
    RelationNotFoundError,
    RelationPathRequest,
    RelationProposal,
    RelationType,
)
from dithyramba.structure import (
    StructureAuthorizationError,
    StructureGenerationRequest,
    StructureIntegrityError,
    StructureNotFoundError,
    StructureUnitKind,
    StructureUnitProposal,
)
from tests.integration._structure_fixtures import structure_context, structure_request
from tests.integration.test_p7_relations import (
    _context as relation_context,
)
from tests.integration.test_p7_relations import (
    _import_request,
    _proposals,
)


def test_structure_repository_rejects_invalid_entrypoints_and_duplicate_content(
    tmp_path: Any,
) -> None:
    with pytest.raises(TypeError, match="LibraryRepository"):
        SQLiteStructureRepository(cast(Any, object()))

    repository, structures, fragment_ids, snapshot_id, policy, scope = structure_context(tmp_path)
    proposal = StructureUnitProposal(
        StructureUnitKind.SECTION,
        fragment_ids[1:],
        "Evidence with qualification",
    )
    try:
        with pytest.raises(TypeError, match="request must be"):
            structures.persist_generation(cast(Any, object()), (proposal,))
        for proposals in ((), cast(Any, [proposal]), (cast(Any, "proposal"),)):
            with pytest.raises(TypeError, match="1 to 5,000"):
                structures.persist_generation(
                    structure_request(snapshot_id, policy, scope), cast(Any, proposals)
                )

        other_library_scope = RequestScope(
            library_id="library_other",
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
        )
        with pytest.raises(StructureAuthorizationError, match="another Library"):
            structures.persist_generation(
                StructureGenerationRequest(
                    snapshot_id,
                    policy.access_policy_id,
                    other_library_scope,
                    "transcript_section",
                    "1.0",
                ),
                (proposal,),
            )

        with pytest.raises(StructureAuthorizationError, match="does not exist"):
            structures.persist_generation(
                structure_request("snapshot_missing", policy, scope), (proposal,)
            )

        denied_scope = RequestScope(
            library_id=scope.library_id,
            snapshot_hash=scope.snapshot_hash,
            purpose="publishing",
            collection_ids=scope.collection_ids,
        )
        with pytest.raises(StructureAuthorizationError, match="not authorized"):
            structures.persist_generation(
                structure_request(snapshot_id, policy, denied_scope), (proposal,)
            )

        with pytest.raises(StructureIntegrityError, match="duplicate content"):
            structures.persist_generation(
                structure_request(snapshot_id, policy, scope), (proposal, proposal)
            )
        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM structure_unit_generations"
            ).fetchone()[0]
            == 0
        )
    finally:
        repository.close()


def test_structure_read_rejects_unknown_or_mismatched_context(tmp_path: Any) -> None:
    repository, structures, fragment_ids, snapshot_id, policy, scope = structure_context(tmp_path)
    try:
        generation = structures.persist_generation(
            structure_request(snapshot_id, policy, scope),
            (
                StructureUnitProposal(
                    StructureUnitKind.SECTION,
                    fragment_ids[1:],
                    "Evidence with qualification",
                ),
            ),
        )
        unit_id = generation.units[0].structure_unit_id

        with pytest.raises(StructureNotFoundError, match="generation"):
            structures.load_generation("structure_generation_missing")
        with pytest.raises(StructureNotFoundError, match="does not exist"):
            structures.load_unit("structure_unit_missing")
        with pytest.raises(StructureNotFoundError, match="receipt"):
            structures.load_read_receipt("structure_read_missing")
        with pytest.raises(TypeError, match="RequestScope"):
            structures.read_unit(
                structure_unit_id=unit_id,
                corpus_snapshot_id=snapshot_id,
                access_policy_id=policy.access_policy_id,
                scope=object(),
            )

        other_library_scope = RequestScope(
            library_id="library_other",
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
        )
        with pytest.raises(StructureAuthorizationError, match="another Library"):
            structures.read_unit(
                structure_unit_id=unit_id,
                corpus_snapshot_id=snapshot_id,
                access_policy_id=policy.access_policy_id,
                scope=other_library_scope,
            )
        with pytest.raises(StructureAuthorizationError, match="generation CorpusSnapshot"):
            structures.read_unit(
                structure_unit_id=unit_id,
                corpus_snapshot_id="snapshot_other",
                access_policy_id=policy.access_policy_id,
                scope=scope,
            )
        with pytest.raises(StructureAuthorizationError, match="does not exist"):
            structures.read_unit(
                structure_unit_id=unit_id,
                corpus_snapshot_id=snapshot_id,
                access_policy_id="policy_missing",
                scope=scope,
            )
    finally:
        repository.close()


def test_structure_resolution_is_anchor_exact_and_generation_scope_bound(tmp_path: Any) -> None:
    repository, structures, fragment_ids, snapshot_id, policy, scope = structure_context(tmp_path)
    try:
        generation = structures.persist_generation(
            structure_request(snapshot_id, policy, scope),
            (
                StructureUnitProposal(
                    StructureUnitKind.SECTION,
                    fragment_ids[1:],
                    "Evidence with qualification",
                ),
            ),
        )
        expected = generation.units

        assert (
            structures.resolve_permitted_units(
                generation_id=generation.generation_id,
                access_policy_id=policy.access_policy_id,
                scope=scope,
                anchor_fragment_ids=(fragment_ids[2],),
            )
            == expected
        )
        assert (
            structures.resolve_permitted_units(
                generation_id=generation.generation_id,
                access_policy_id=policy.access_policy_id,
                scope=scope,
                anchor_fragment_ids=(fragment_ids[0],),
            )
            == ()
        )

        invalid_anchors = (
            (),
            cast(Any, [fragment_ids[1]]),
            (fragment_ids[1], fragment_ids[1]),
            ("source_wrong_namespace",),
        )
        for anchors in invalid_anchors:
            with pytest.raises(TypeError, match="anchor_fragment_ids"):
                structures.resolve_permitted_units(
                    generation_id=generation.generation_id,
                    access_policy_id=policy.access_policy_id,
                    scope=scope,
                    anchor_fragment_ids=cast(Any, anchors),
                )
        with pytest.raises(TypeError, match="RequestScope"):
            structures.resolve_permitted_units(
                generation_id=generation.generation_id,
                access_policy_id=policy.access_policy_id,
                scope=object(),
                anchor_fragment_ids=(fragment_ids[1],),
            )

        other_library_scope = RequestScope(
            library_id="library_other",
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
        )
        with pytest.raises(StructureAuthorizationError, match="another Library"):
            structures.resolve_permitted_units(
                generation_id=generation.generation_id,
                access_policy_id=policy.access_policy_id,
                scope=other_library_scope,
                anchor_fragment_ids=(fragment_ids[1],),
            )
        with pytest.raises(StructureAuthorizationError, match="does not exist"):
            structures.resolve_permitted_units(
                generation_id=generation.generation_id,
                access_policy_id="policy_missing",
                scope=scope,
                anchor_fragment_ids=(fragment_ids[1],),
            )
        with pytest.raises(StructureAuthorizationError, match="outside"):
            structures.resolve_permitted_units(
                generation_id=generation.generation_id,
                access_policy_id=policy.access_policy_id,
                scope=scope,
                anchor_fragment_ids=("fragment_absent",),
            )

        changed_scope = RequestScope(
            library_id=scope.library_id,
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
            exclusions=QueryExclusions(source_fragment_ids=(fragment_ids[0],)),
        )
        with pytest.raises(StructureAuthorizationError, match="differs"):
            structures.resolve_permitted_units(
                generation_id=generation.generation_id,
                access_policy_id=policy.access_policy_id,
                scope=changed_scope,
                anchor_fragment_ids=(fragment_ids[1],),
            )
    finally:
        repository.close()


def test_relation_repository_rejects_invalid_imports_before_writing(tmp_path: Any) -> None:
    with pytest.raises(TypeError, match="LibraryRepository"):
        SQLiteRelationRepository(cast(Any, object()))

    repository, relations, snapshot_id, policy, scope, statements, evidence = relation_context(
        tmp_path
    )
    proposals = _proposals(statements, evidence)
    request = _import_request(snapshot_id, policy, scope)
    try:
        with pytest.raises(TypeError, match="request must be"):
            relations.import_relations(cast(Any, object()), proposals)
        for invalid in ((), cast(Any, [proposals[0]]), (cast(Any, "proposal"),)):
            with pytest.raises(TypeError, match="non-empty bounded tuple"):
                relations.import_relations(request, cast(Any, invalid))
        bounded = RelationImportRequest(
            snapshot_id,
            policy.access_policy_id,
            scope,
            "p7",
            "relations/1.0",
            max_relations=1,
        )
        with pytest.raises(TypeError, match="bounded"):
            relations.import_relations(bounded, proposals)

        other_library_scope = RequestScope(
            library_id="library_other",
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
        )
        with pytest.raises(RelationAuthorizationError, match="another Library"):
            relations.import_relations(
                RelationImportRequest(
                    snapshot_id,
                    policy.access_policy_id,
                    other_library_scope,
                    "p7",
                    "relations/1.0",
                ),
                proposals,
            )

        with pytest.raises(RelationAuthorizationError, match="does not exist"):
            relations.import_relations(
                _import_request("snapshot_missing", policy, scope), proposals
            )
        denied_scope = RequestScope(
            library_id=scope.library_id,
            snapshot_hash=scope.snapshot_hash,
            purpose="publishing",
            collection_ids=scope.collection_ids,
        )
        with pytest.raises(RelationAuthorizationError, match="not authorized"):
            relations.import_relations(
                _import_request(snapshot_id, policy, denied_scope), proposals
            )
        with pytest.raises(RelationIntegrityError, match="duplicate content"):
            relations.import_relations(request, (proposals[0], proposals[0]))

        missing_evidence = RelationProposal(
            RelationType.SUPPORTS,
            RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),
            RelationNodeRef(RelationNodeType.STATEMENT, statements[1]),
            ("evidence_missing",),
            "Unverifiable evidence must fail before persistence.",
            "human_import/1.0",
        )
        with pytest.raises(RelationIntegrityError, match="EvidenceLink set is incomplete"):
            relations.import_relations(request, (missing_evidence,))

        assert (
            repository._store.connection.execute(
                "SELECT COUNT(*) FROM relation_import_runs"
            ).fetchone()[0]
            == 0
        )
    finally:
        repository.close()


def test_relation_load_and_traversal_validate_nodes_policy_and_direction(tmp_path: Any) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = relation_context(
        tmp_path
    )
    try:
        result = relations.import_relations(
            _import_request(snapshot_id, policy, scope), _proposals(statements, evidence)
        )
        with pytest.raises(RelationNotFoundError, match="import run"):
            relations.load_import_result("relation_run_missing")
        with pytest.raises(RelationNotFoundError, match="does not exist"):
            relations.load_relation("relation_missing")
        with pytest.raises(RelationNotFoundError, match="path receipt"):
            relations.load_path_receipt("relation_path_missing")
        with pytest.raises(TypeError, match="RelationPathRequest"):
            relations.traverse(cast(Any, object()))

        other_library_scope = RequestScope(
            library_id="library_other",
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
        )
        with pytest.raises(RelationAuthorizationError, match="another Library"):
            relations.traverse(
                RelationPathRequest(
                    snapshot_id,
                    policy.access_policy_id,
                    other_library_scope,
                    (RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),),
                    (RelationType.SUPPORTS,),
                )
            )

        with pytest.raises(RelationAuthorizationError, match="does not exist"):
            relations.traverse(
                RelationPathRequest(
                    snapshot_id,
                    "policy_missing",
                    scope,
                    (RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),),
                    (RelationType.SUPPORTS,),
                )
            )

        with pytest.raises(RelationIntegrityError, match="does not exist"):
            relations.traverse(
                RelationPathRequest(
                    snapshot_id,
                    policy.access_policy_id,
                    scope,
                    (RelationNodeRef(RelationNodeType.STATEMENT, "statement_missing"),),
                    (RelationType.SUPPORTS,),
                )
            )

        incoming = relations.traverse(
            RelationPathRequest(
                snapshot_id,
                policy.access_policy_id,
                scope,
                (RelationNodeRef(RelationNodeType.STATEMENT, statements[1]),),
                (RelationType.SUPPORTS,),
                RelationDirection.INCOMING,
                1,
                10,
            )
        )
        assert len(incoming.paths) == 1
        assert incoming.paths[0].terminal.node_id == statements[0]

        both = relations.traverse(
            RelationPathRequest(
                snapshot_id,
                policy.access_policy_id,
                scope,
                (RelationNodeRef(RelationNodeType.STATEMENT, statements[1]),),
                (RelationType.SUPPORTS, RelationType.CONTRADICTS),
                RelationDirection.BOTH,
                1,
                10,
            )
        )
        assert {path.terminal.node_id for path in both.paths} == {
            statements[0],
            statements[2],
        }
        assert all(
            relation.relation_id in {step.relation_id for path in both.paths for step in path.steps}
            for relation in result.relations
        )
    finally:
        repository.close()


def test_relation_path_receipt_round_trips_all_exclusion_namespaces(tmp_path: Any) -> None:
    repository, relations, snapshot_id, policy, scope, statements, evidence = relation_context(
        tmp_path
    )
    try:
        relations.import_relations(
            _import_request(snapshot_id, policy, scope), _proposals(statements, evidence)
        )
        scoped = RequestScope(
            library_id=scope.library_id,
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
            exclusions=QueryExclusions(
                source_ids=("source_absent",),
                source_family_ids=("family_absent",),
                source_fragment_ids=("fragment_absent",),
            ),
        )
        receipt = relations.traverse(
            RelationPathRequest(
                snapshot_id,
                policy.access_policy_id,
                scoped,
                (RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),),
                (RelationType.OPENS_GAP,),
                RelationDirection.OUTGOING,
                1,
                10,
            )
        )

        assert receipt.paths == ()
        loaded = relations.load_path_receipt(receipt.receipt_id)
        assert loaded == receipt
        assert loaded.request.scope.exclusions == scoped.exclusions
    finally:
        repository.close()


def test_relation_seed_resolution_and_grounding_are_policy_complete(tmp_path: Any) -> None:
    repository, relations, snapshot_id, policy, scope, statements, _evidence = relation_context(
        tmp_path
    )
    try:
        seeds = relations.resolve_permitted_seeds(
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy.access_policy_id,
            scope=scope,
            source_fragment_ids=("fragment_relations_1", "fragment_relations_0"),
        )
        assert seeds == (
            RelationNodeRef(RelationNodeType.STATEMENT, statements[0]),
            RelationNodeRef(RelationNodeType.STATEMENT, statements[1]),
        )

        grounding = relations.ground_permitted_nodes(
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy.access_policy_id,
            scope=scope,
            nodes=tuple(reversed(seeds)),
        )
        assert tuple(item.node for item in grounding) == seeds
        assert tuple(item.source_fragment_ids for item in grounding) == (
            ("fragment_relations_0",),
            ("fragment_relations_1",),
        )

        voice = RelationNodeRef(RelationNodeType.VOICE, "voice_relations")
        voice_grounding = relations.ground_permitted_nodes(
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy.access_policy_id,
            scope=scope,
            nodes=(voice,),
        )
        assert voice_grounding[0].source_fragment_ids == (
            "fragment_relations_0",
            "fragment_relations_1",
            "fragment_relations_2",
        )

        for invalid_fragments in (
            (),
            cast(Any, ["fragment_relations_0"]),
            ("fragment_relations_0", "fragment_relations_0"),
            ("source_relations",),
        ):
            with pytest.raises(TypeError, match="source_fragment_ids"):
                relations.resolve_permitted_seeds(
                    corpus_snapshot_id=snapshot_id,
                    access_policy_id=policy.access_policy_id,
                    scope=scope,
                    source_fragment_ids=cast(Any, invalid_fragments),
                )
        with pytest.raises(RelationAuthorizationError, match="outside"):
            relations.resolve_permitted_seeds(
                corpus_snapshot_id=snapshot_id,
                access_policy_id=policy.access_policy_id,
                scope=scope,
                source_fragment_ids=("fragment_absent",),
            )

        for invalid_nodes in (
            (),
            cast(Any, [seeds[0]]),
            (seeds[0], seeds[0]),
            (cast(Any, "statement_relations_0"),),
        ):
            with pytest.raises(TypeError, match="unique tuple"):
                relations.ground_permitted_nodes(
                    corpus_snapshot_id=snapshot_id,
                    access_policy_id=policy.access_policy_id,
                    scope=scope,
                    nodes=cast(Any, invalid_nodes),
                )
        with pytest.raises(RelationAuthorizationError, match="lacks complete"):
            relations.ground_permitted_nodes(
                corpus_snapshot_id=snapshot_id,
                access_policy_id=policy.access_policy_id,
                scope=scope,
                nodes=(RelationNodeRef(RelationNodeType.STATEMENT, "statement_missing"),),
            )

        restricted = RequestScope(
            library_id=scope.library_id,
            snapshot_hash=scope.snapshot_hash,
            purpose=scope.purpose,
            collection_ids=scope.collection_ids,
            exclusions=QueryExclusions(
                source_fragment_ids=("fragment_relations_2",),
            ),
        )
        with pytest.raises(RelationAuthorizationError, match="lacks complete"):
            relations.ground_permitted_nodes(
                corpus_snapshot_id=snapshot_id,
                access_policy_id=policy.access_policy_id,
                scope=restricted,
                nodes=(voice,),
            )
    finally:
        repository.close()
