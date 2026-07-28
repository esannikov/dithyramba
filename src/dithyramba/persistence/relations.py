"""Policy-before-traversal persistence for typed P7 Relations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from dithyramba.access import AccessContractError, QueryExclusions, RequestScope
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex
from dithyramba.relations import (
    Relation,
    RelationAuthorizationError,
    RelationDirection,
    RelationImportReceipt,
    RelationImportRequest,
    RelationImportResult,
    RelationIntegrityError,
    RelationNodeGrounding,
    RelationNodeRef,
    RelationNodeType,
    RelationNotFoundError,
    RelationPath,
    RelationPathReceipt,
    RelationPathRequest,
    RelationPathStep,
    RelationProposal,
    RelationType,
)

from .errors import (
    AccessPolicyNotFoundError,
    AuthorizationError,
    CollectionNotFoundError,
    CorpusSnapshotNotFoundError,
)
from .models import AuthorizedRead
from .repository import LibraryRepository, _timestamp

_TEMP_PERMITTED = "temp_dithyramba_relation_permitted_fragments"
_TEMP_AUTHORIZED_NODES = "temp_dithyramba_relation_authorized_nodes"
_TEMP_AUTHORIZED = "temp_dithyramba_authorized_relations"
_TEMP_EDGES = "temp_dithyramba_relation_edges"


class SQLiteRelationRepository:
    """Import exact-evidence Relations and traverse only policy-compiled edges."""

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteRelationRepository requires a LibraryRepository")
        self._repository = repository

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def import_relations(
        self,
        request: RelationImportRequest,
        proposals: tuple[RelationProposal, ...],
    ) -> RelationImportResult:
        """Validate and atomically append a bounded set of candidate Relations."""

        if not isinstance(request, RelationImportRequest):
            raise TypeError("request must be a RelationImportRequest")
        if (
            type(proposals) is not tuple
            or not proposals
            or len(proposals) > request.max_relations
            or any(not isinstance(item, RelationProposal) for item in proposals)
        ):
            raise TypeError("proposals must be a non-empty bounded tuple of RelationProposal")
        authorization = self._authorize_scope(
            corpus_snapshot_id=request.corpus_snapshot_id,
            access_policy_id=request.access_policy_id,
            scope=request.scope,
        )
        token = authorization.compiled.token
        permitted_ids = {item.source_fragment_id for item in authorization.compiled.manifest.items}
        ordered = tuple(sorted(proposals, key=lambda item: item.content_hash))
        if len({item.content_hash for item in ordered}) != len(ordered):
            raise RelationIntegrityError("Relation proposals contain duplicate content")
        evidence_ids = tuple(
            sorted({evidence_id for item in ordered for evidence_id in item.evidence_link_ids})
        )

        input_hash = canonical_sha256_hex(
            {
                "schema": "dithyramba.relation_import_input/1.0",
                "request": request.payload(),
                "proposals": [item.payload() for item in ordered],
            }
        )
        import_run_id = canonical_content_id(
            "relation_run",
            {
                "schema": "dithyramba.relation_import_identity/1.0",
                "library_id": self.library_id,
                "input_hash": input_hash,
                "policy_hash": token.policy_hash,
                "scope_hash": token.exclusion_hash,
                "permitted_set_hash": token.permitted_set_hash,
                "profile_version": request.profile_version,
            },
        )
        relations = tuple(
            sorted(
                (
                    Relation(
                        relation_id=canonical_content_id("relation", proposal.payload()),
                        import_run_id=import_run_id,
                        library_id=self.library_id,
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
                    for proposal in ordered
                ),
                key=lambda item: item.relation_id,
            )
        )
        output_hash = canonical_sha256_hex(
            {
                "schema": "dithyramba.relation_import_output/1.0",
                "relations": [
                    {
                        "relation_id": item.relation_id,
                        "content_hash": item.content_hash,
                    }
                    for item in relations
                ],
            }
        )
        receipt = RelationImportReceipt(
            import_run_id=import_run_id,
            library_id=self.library_id,
            corpus_snapshot_id=request.corpus_snapshot_id,
            access_policy_id=request.access_policy_id,
            policy_hash=token.policy_hash,
            scope_hash=token.exclusion_hash,
            permitted_set_hash=token.permitted_set_hash,
            code_version=request.code_version,
            profile_version=request.profile_version,
            input_hash=input_hash,
            output_hash=output_hash,
            relation_ids=tuple(item.relation_id for item in relations),
        )
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                evidence = self._evidence_metadata(evidence_ids)
                for proposal in ordered:
                    self._validate_proposal(proposal, evidence, permitted_ids)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO relation_import_runs(
                        relation_import_run_id, library_id, corpus_snapshot_id,
                        access_policy_id, policy_hash, scope_hash, permitted_set_hash,
                        code_version, profile_version, input_hash, relation_count,
                        output_hash, receipt_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.import_run_id,
                        receipt.library_id,
                        receipt.corpus_snapshot_id,
                        receipt.access_policy_id,
                        receipt.policy_hash,
                        receipt.scope_hash,
                        receipt.permitted_set_hash,
                        receipt.code_version,
                        receipt.profile_version,
                        receipt.input_hash,
                        len(receipt.relation_ids),
                        receipt.output_hash,
                        receipt.receipt_hash,
                        created_at,
                    ),
                )
                if cursor.rowcount == 1:
                    self._insert_relations(connection, relations, created_at)
                try:
                    persisted = self._load_import_result(connection, import_run_id)
                except RelationNotFoundError as exc:
                    raise RelationIntegrityError(
                        "persisted Relation import run is missing after insert"
                    ) from exc
                if persisted.receipt != receipt or tuple(
                    item.as_proposal() for item in persisted.relations
                ) != tuple(item.as_proposal() for item in relations):
                    raise RelationIntegrityError(
                        "persisted Relation import conflicts with its content address"
                    )
        except sqlite3.IntegrityError as exc:
            raise RelationIntegrityError("Relation import insert conflicted") from exc
        return persisted

    def load_import_result(self, import_run_id: str) -> RelationImportResult:
        with self._repository._store.transaction() as connection:
            return self._load_import_result(connection, import_run_id)

    def load_relation(self, relation_id: str) -> Relation:
        with self._repository._store.transaction() as connection:
            return self._load_relation(connection, relation_id)

    def resolve_permitted_seeds(
        self,
        *,
        corpus_snapshot_id: str,
        access_policy_id: str,
        scope: RequestScope,
        source_fragment_ids: tuple[str, ...],
    ) -> tuple[RelationNodeRef, ...]:
        """Resolve retrieved fragments to grounded graph seeds without text reads.

        The first retrieval slice intentionally exposes only Statement,
        ConceptMeaning and StructureUnit seeds. Other node families remain
        traversable when explicitly requested, but cannot enter recall until
        their grounding oracle is separately benchmarked.
        """

        identifiers = _fragment_ids(source_fragment_ids)
        authorization = self._authorize_scope(
            corpus_snapshot_id=corpus_snapshot_id,
            access_policy_id=access_policy_id,
            scope=scope,
        )
        permitted_ids = {item.source_fragment_id for item in authorization.compiled.manifest.items}
        if not set(identifiers).issubset(permitted_ids):
            raise RelationAuthorizationError(
                "Relation seed fragment is outside the permitted manifest"
            )
        connection = self._repository._store.connection
        nodes: set[RelationNodeRef] = set()
        try:
            with self._repository._store.transaction() as tx:
                self._prepare_permitted_fragments(tx, tuple(sorted(permitted_ids)))
                self._prepare_authorized_nodes(tx)
                for offset in range(0, len(identifiers), 500):
                    chunk = identifiers[offset : offset + 500]
                    placeholders = ", ".join("?" for _ in chunk)
                    rows = tx.execute(
                        f"""
                        SELECT 'statement', evidence.statement_id
                        FROM evidence_links AS evidence
                        JOIN {_TEMP_AUTHORIZED_NODES} AS node
                          ON node.node_type = 'statement'
                         AND node.node_id = evidence.statement_id
                        WHERE evidence.source_fragment_id IN ({placeholders})
                        UNION
                        SELECT 'concept_meaning', support.concept_meaning_id
                        FROM concept_meaning_statements AS support
                        JOIN evidence_links AS evidence
                          ON evidence.statement_id = support.statement_id
                        JOIN {_TEMP_AUTHORIZED_NODES} AS node
                          ON node.node_type = 'concept_meaning'
                         AND node.node_id = support.concept_meaning_id
                        WHERE evidence.source_fragment_id IN ({placeholders})
                        UNION
                        SELECT 'structure_unit', member.structure_unit_id
                        FROM structure_unit_members AS member
                        JOIN {_TEMP_AUTHORIZED_NODES} AS node
                          ON node.node_type = 'structure_unit'
                         AND node.node_id = member.structure_unit_id
                        WHERE member.source_fragment_id IN ({placeholders})
                        """,
                        (*chunk, *chunk, *chunk),
                    ).fetchall()
                    nodes.update(
                        RelationNodeRef(RelationNodeType(str(row[0])), str(row[1])) for row in rows
                    )
        finally:
            self._drop_temp_tables(connection)
        return tuple(sorted(nodes))

    def ground_permitted_nodes(
        self,
        *,
        corpus_snapshot_id: str,
        access_policy_id: str,
        scope: RequestScope,
        nodes: tuple[RelationNodeRef, ...],
    ) -> tuple[RelationNodeGrounding, ...]:
        """Return complete, policy-safe SourceFragment grounding for exact nodes."""

        if (
            type(nodes) is not tuple
            or not 1 <= len(nodes) <= 100
            or any(not isinstance(item, RelationNodeRef) for item in nodes)
            or len(set(nodes)) != len(nodes)
        ):
            raise TypeError("nodes must be a unique tuple of 1 to 100 RelationNodeRef values")
        authorization = self._authorize_scope(
            corpus_snapshot_id=corpus_snapshot_id,
            access_policy_id=access_policy_id,
            scope=scope,
        )
        permitted_ids = {item.source_fragment_id for item in authorization.compiled.manifest.items}
        connection = self._repository._store.connection
        grounding: list[RelationNodeGrounding] = []
        try:
            with self._repository._store.transaction() as tx:
                self._prepare_permitted_fragments(tx, tuple(sorted(permitted_ids)))
                self._prepare_authorized_nodes(tx)
                for node in sorted(nodes):
                    if (
                        tx.execute(
                            f"SELECT 1 FROM {_TEMP_AUTHORIZED_NODES} "
                            "WHERE node_type = ? AND node_id = ?",
                            (node.node_type.value, node.node_id),
                        ).fetchone()
                        is None
                    ):
                        raise RelationAuthorizationError(
                            "Relation node lacks complete permitted grounding"
                        )
                    fragment_ids = self._node_grounding_fragments(tx, node)
                    if not fragment_ids or not set(fragment_ids).issubset(permitted_ids):
                        raise RelationIntegrityError(
                            "authorized Relation node grounding is incomplete"
                        )
                    grounding.append(RelationNodeGrounding(node, fragment_ids))
        finally:
            self._drop_temp_tables(connection)
        return tuple(grounding)

    def _authorize_scope(
        self,
        *,
        corpus_snapshot_id: str,
        access_policy_id: str,
        scope: RequestScope,
    ) -> AuthorizedRead:
        if not isinstance(scope, RequestScope):
            raise TypeError("scope must be a RequestScope")
        if scope.library_id != self.library_id:
            raise RelationAuthorizationError("Relation operation targets another Library")
        self._validate_snapshot_scope(corpus_snapshot_id, scope.snapshot_hash)
        try:
            authorization = self._repository.authorize_read(
                access_policy_id=access_policy_id,
                scope=scope,
            )
        except (
            AccessContractError,
            AccessPolicyNotFoundError,
            AuthorizationError,
            CollectionNotFoundError,
        ) as exc:
            raise RelationAuthorizationError(str(exc)) from exc
        if not isinstance(authorization, AuthorizedRead):
            raise RelationIntegrityError("Relation authorization contract changed")
        return authorization

    def traverse(self, request: RelationPathRequest) -> RelationPathReceipt:
        """Run a deterministic recursive CTE over policy-authorized Relations only."""

        if not isinstance(request, RelationPathRequest):
            raise TypeError("request must be a RelationPathRequest")
        authorization = self._authorize_scope(
            corpus_snapshot_id=request.corpus_snapshot_id,
            access_policy_id=request.access_policy_id,
            scope=request.scope,
        )
        token = authorization.compiled.token
        for seed in request.seeds:
            self._require_node(seed)

        connection = self._repository._store.connection
        try:
            with self._repository._store.transaction() as tx:
                self._prepare_authorized_edges(
                    tx,
                    permitted_fragment_ids=tuple(
                        item.source_fragment_id for item in authorization.compiled.manifest.items
                    ),
                    relation_types=request.relation_types,
                    direction=request.direction,
                )
                rows = self._walk(tx, request)
                relation_ids = tuple(
                    sorted(
                        {
                            relation_id
                            for row in rows
                            for relation_id in str(row[4]).split(",")
                            if relation_id
                        }
                    )
                )
                relation_map = {
                    relation_id: self._load_relation(tx, relation_id)
                    for relation_id in relation_ids
                }
                paths = self._paths_from_rows(request, rows, relation_map)
        finally:
            self._drop_temp_tables(connection)

        receipt_payload: dict[str, object] = {
            "schema": "dithyramba.relation_path_receipt/1.0",
            "library_id": self.library_id,
            "corpus_snapshot_id": request.corpus_snapshot_id,
            "access_policy_id": request.access_policy_id,
            "policy_hash": token.policy_hash,
            "scope_hash": token.exclusion_hash,
            "permitted_set_hash": token.permitted_set_hash,
            "request": request.payload(),
            "paths": [item.payload() | {"path_hash": item.path_hash} for item in paths],
        }
        receipt = RelationPathReceipt(
            receipt_id=canonical_content_id("relation_path", receipt_payload),
            library_id=self.library_id,
            corpus_snapshot_id=request.corpus_snapshot_id,
            access_policy_id=request.access_policy_id,
            policy_hash=token.policy_hash,
            scope_hash=token.exclusion_hash,
            permitted_set_hash=token.permitted_set_hash,
            request=request,
            paths=paths,
        )
        self._persist_path_receipt(receipt)
        return receipt

    def _validate_snapshot_scope(
        self,
        corpus_snapshot_id: str,
        snapshot_hash: str,
    ) -> None:
        try:
            snapshot = self._repository.get_corpus_snapshot(corpus_snapshot_id)
        except CorpusSnapshotNotFoundError as exc:
            raise RelationAuthorizationError(
                "Relation CorpusSnapshot does not exist in this Library"
            ) from exc
        if snapshot.manifest_hash != snapshot_hash:
            raise RelationAuthorizationError("Relation CorpusSnapshot ID/hash are inconsistent")

    def load_path_receipt(self, receipt_id: str) -> RelationPathReceipt:
        """Load and independently rehash a complete normalized path receipt."""

        with self._repository._store.transaction() as connection:
            row = connection.execute(
                """
                SELECT corpus_snapshot_id, access_policy_id, snapshot_hash,
                       policy_hash, scope_hash, permitted_set_hash, request_hash,
                       purpose, direction, max_hops, max_paths, seed_count,
                       path_count, receipt_hash
                FROM relation_path_receipts
                WHERE relation_path_receipt_id = ? AND library_id = ?
                """,
                (receipt_id, self.library_id),
            ).fetchone()
            if row is None:
                raise RelationNotFoundError("Relation path receipt does not exist")
            collections = tuple(
                str(item[0])
                for item in connection.execute(
                    "SELECT collection_id FROM relation_path_collections "
                    "WHERE relation_path_receipt_id = ? ORDER BY collection_id",
                    (receipt_id,),
                ).fetchall()
            )
            exclusions = connection.execute(
                "SELECT target_type, target_id FROM relation_path_exclusions "
                "WHERE relation_path_receipt_id = ? ORDER BY target_type, target_id",
                (receipt_id,),
            ).fetchall()
            relation_types = tuple(
                RelationType(str(item[0]))
                for item in connection.execute(
                    "SELECT relation_type FROM relation_path_relation_types "
                    "WHERE relation_path_receipt_id = ? ORDER BY relation_type",
                    (receipt_id,),
                ).fetchall()
            )
            seeds = tuple(
                RelationNodeRef(RelationNodeType(str(item[1])), str(item[2]))
                for item in connection.execute(
                    "SELECT seed_order, node_type, node_id FROM relation_path_seeds "
                    "WHERE relation_path_receipt_id = ? ORDER BY seed_order",
                    (receipt_id,),
                ).fetchall()
            )
            path_rows = connection.execute(
                "SELECT path_order, seed_order, path_hash FROM relation_paths "
                "WHERE relation_path_receipt_id = ? ORDER BY path_order",
                (receipt_id,),
            ).fetchall()
            paths: list[RelationPath] = []
            for path_row in path_rows:
                step_rows = connection.execute(
                    """
                    SELECT relation_id, from_type, from_id, to_type, to_id
                    FROM relation_path_steps
                    WHERE relation_path_receipt_id = ? AND path_order = ?
                    ORDER BY hop_order
                    """,
                    (receipt_id, int(path_row[0])),
                ).fetchall()
                steps = tuple(self._step_from_row(connection, step_row) for step_row in step_rows)
                path = RelationPath(int(path_row[1]), seeds[int(path_row[1])], steps)
                if path.path_hash != str(path_row[2]):
                    raise RelationIntegrityError("persisted Relation path hash mismatch")
                paths.append(path)

        request = RelationPathRequest(
            corpus_snapshot_id=str(row[0]),
            access_policy_id=str(row[1]),
            scope=RequestScope(
                library_id=self.library_id,
                snapshot_hash=str(row[2]),
                purpose=str(row[7]),
                collection_ids=collections,
                exclusions=_query_exclusions(exclusions),
            ),
            seeds=seeds,
            relation_types=relation_types,
            direction=RelationDirection(str(row[8])),
            max_hops=int(row[9]),
            max_paths=int(row[10]),
        )
        if request.request_hash != str(row[6]) or len(seeds) != int(row[11]):
            raise RelationIntegrityError("persisted Relation path request mismatch")
        canonical_paths = tuple(sorted(paths, key=lambda item: item.path_hash))
        if len(canonical_paths) != int(row[12]):
            raise RelationIntegrityError("persisted Relation path count mismatch")
        receipt = RelationPathReceipt(
            receipt_id=receipt_id,
            library_id=self.library_id,
            corpus_snapshot_id=str(row[0]),
            access_policy_id=str(row[1]),
            policy_hash=str(row[3]),
            scope_hash=str(row[4]),
            permitted_set_hash=str(row[5]),
            request=request,
            paths=canonical_paths,
        )
        if receipt.receipt_hash != str(row[13]):
            raise RelationIntegrityError("persisted Relation path receipt hash mismatch")
        if canonical_content_id("relation_path", receipt.payload()) != receipt.receipt_id:
            raise RelationIntegrityError("persisted Relation path receipt ID mismatch")
        return receipt

    def _validate_proposal(
        self,
        proposal: RelationProposal,
        evidence: dict[str, tuple[str, str]],
        permitted_fragment_ids: set[str],
    ) -> None:
        self._require_node(proposal.subject)
        self._require_node(proposal.object_node)
        evidence_rows = tuple(evidence[item] for item in proposal.evidence_link_ids)
        if not {item[1] for item in evidence_rows}.issubset(permitted_fragment_ids):
            raise RelationAuthorizationError(
                "Relation evidence contains a SourceFragment outside the permitted manifest"
            )
        statement_ids = {item[0] for item in evidence_rows}
        for endpoint in (proposal.subject, proposal.object_node):
            grounding_fragment_ids = self._node_grounding_fragments(
                self._repository._store.connection,
                endpoint,
            )
            if not grounding_fragment_ids:
                raise RelationIntegrityError(
                    "Relation endpoint requires complete SourceFragment grounding"
                )
            if not set(grounding_fragment_ids).issubset(permitted_fragment_ids):
                raise RelationAuthorizationError(
                    "Relation endpoint grounding is outside the permitted manifest"
                )
            if (
                endpoint.node_type is RelationNodeType.STATEMENT
                and endpoint.node_id not in statement_ids
            ):
                raise RelationIntegrityError(
                    "Statement Relation endpoints require their own exact EvidenceLink"
                )
            if endpoint.node_type is RelationNodeType.CONCEPT_MEANING:
                linked = {
                    str(item[0])
                    for item in self._repository._store.connection.execute(
                        "SELECT statement_id FROM concept_meaning_statements "
                        "WHERE concept_meaning_id = ?",
                        (endpoint.node_id,),
                    ).fetchall()
                }
                if not linked.intersection(statement_ids):
                    raise RelationIntegrityError(
                        "ConceptMeaning Relation endpoints require linked Statement evidence"
                    )

    def _evidence_metadata(self, evidence_link_ids: tuple[str, ...]) -> dict[str, tuple[str, str]]:
        rows: list[sqlite3.Row] = []
        connection = self._repository._store.connection
        for offset in range(0, len(evidence_link_ids), 500):
            chunk = evidence_link_ids[offset : offset + 500]
            placeholders = ", ".join("?" for _ in chunk)
            rows.extend(
                connection.execute(
                    f"""
                    SELECT evidence.evidence_link_id, evidence.statement_id,
                           evidence.source_fragment_id
                    FROM evidence_links AS evidence
                    JOIN statements AS statement
                      ON statement.statement_id = evidence.statement_id
                    WHERE statement.library_id = ?
                      AND evidence.evidence_link_id IN ({placeholders})
                    """,
                    (self.library_id, *chunk),
                ).fetchall()
            )
        if {str(row[0]) for row in rows} != set(evidence_link_ids):
            raise RelationIntegrityError("Relation EvidenceLink set is incomplete")
        return {str(row[0]): (str(row[1]), str(row[2])) for row in rows}

    def _require_node(self, node: RelationNodeRef) -> None:
        table, id_column, library_join = _node_query(node.node_type)
        row = self._repository._store.connection.execute(
            f"SELECT 1 FROM {table} {library_join} WHERE {id_column} = ? AND node_library_id = ?",
            (node.node_id, self.library_id),
        ).fetchone()
        if row is None:
            raise RelationIntegrityError(
                f"Relation node does not exist in this Library: {node.node_type.value}"
            )

    def _insert_relations(
        self,
        connection: sqlite3.Connection,
        relations: tuple[Relation, ...],
        created_at: str,
    ) -> None:
        for output_order, relation in enumerate(relations):
            cursor = connection.execute(
                """
                INSERT INTO relations(
                    relation_id, relation_import_run_id, library_id, relation_type,
                    subject_type, subject_id, object_type, object_id, reason_text,
                    extraction_method, valid_start, valid_end, evidence_count,
                    status, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?)
                ON CONFLICT(relation_id) DO NOTHING
                """,
                (
                    relation.relation_id,
                    relation.import_run_id,
                    relation.library_id,
                    relation.relation_type.value,
                    relation.subject.node_type.value,
                    relation.subject.node_id,
                    relation.object_node.node_type.value,
                    relation.object_node.node_id,
                    relation.reason_text,
                    relation.extraction_method,
                    relation.valid_start,
                    relation.valid_end,
                    len(relation.evidence_link_ids),
                    relation.content_hash,
                    created_at,
                ),
            )
            if cursor.rowcount == 1:
                connection.executemany(
                    "INSERT INTO relation_evidence_links "
                    "(relation_id, evidence_link_id, evidence_order) VALUES (?, ?, ?)",
                    (
                        (relation.relation_id, evidence_id, order)
                        for order, evidence_id in enumerate(relation.evidence_link_ids)
                    ),
                )
            elif cursor.rowcount == 0:
                try:
                    existing = self._load_relation(connection, relation.relation_id)
                except RelationNotFoundError as exc:
                    raise RelationIntegrityError(
                        "canonical Relation ID belongs to another Library"
                    ) from exc
                if existing.as_proposal() != relation.as_proposal():
                    raise RelationIntegrityError(
                        "canonical Relation conflicts with its content address"
                    )
            else:  # pragma: no cover - sqlite rowcount is 0 or 1 for this statement
                raise RelationIntegrityError("Relation insert returned an invalid row count")
            connection.execute(
                """
                INSERT INTO relation_import_run_items(
                    relation_import_run_id, relation_id, output_order
                ) VALUES (?, ?, ?)
                """,
                (relation.import_run_id, relation.relation_id, output_order),
            )

    def _load_import_result(
        self, connection: sqlite3.Connection, import_run_id: str
    ) -> RelationImportResult:
        row = connection.execute(
            """
            SELECT library_id, corpus_snapshot_id, access_policy_id, policy_hash,
                   scope_hash, permitted_set_hash, code_version, profile_version,
                   input_hash, relation_count, output_hash, receipt_hash
            FROM relation_import_runs
            WHERE relation_import_run_id = ? AND library_id = ?
            """,
            (import_run_id, self.library_id),
        ).fetchone()
        if row is None:
            raise RelationNotFoundError("Relation import run does not exist")
        relation_rows = connection.execute(
            """
            SELECT item.relation_id, item.output_order
            FROM relation_import_run_items AS item
            JOIN relations AS relation ON relation.relation_id = item.relation_id
            WHERE item.relation_import_run_id = ?
              AND relation.library_id = ?
            ORDER BY item.output_order
            """,
            (import_run_id, self.library_id),
        ).fetchall()
        if len(relation_rows) != int(row[9]):
            raise RelationIntegrityError("Relation import count mismatch")
        output_orders = tuple(int(item[1]) for item in relation_rows)
        if output_orders != tuple(range(int(row[9]))):
            raise RelationIntegrityError("Relation import output order mismatch")
        relation_ids = tuple(str(item[0]) for item in relation_rows)
        if relation_ids != tuple(sorted(relation_ids)):
            raise RelationIntegrityError("Relation import output order is not canonical")
        relations = tuple(
            self._load_relation(connection, relation_id) for relation_id in relation_ids
        )
        output_hash = canonical_sha256_hex(
            {
                "schema": "dithyramba.relation_import_output/1.0",
                "relations": [
                    {
                        "relation_id": item.relation_id,
                        "content_hash": item.content_hash,
                    }
                    for item in relations
                ],
            }
        )
        if output_hash != str(row[10]):
            raise RelationIntegrityError("Relation import output hash mismatch")
        receipt = RelationImportReceipt(
            import_run_id=import_run_id,
            library_id=str(row[0]),
            corpus_snapshot_id=str(row[1]),
            access_policy_id=str(row[2]),
            policy_hash=str(row[3]),
            scope_hash=str(row[4]),
            permitted_set_hash=str(row[5]),
            code_version=str(row[6]),
            profile_version=str(row[7]),
            input_hash=str(row[8]),
            output_hash=output_hash,
            relation_ids=tuple(item.relation_id for item in relations),
        )
        if receipt.receipt_hash != str(row[11]):
            raise RelationIntegrityError("Relation import receipt hash mismatch")
        return RelationImportResult(relations, receipt)

    def _load_relation(self, connection: sqlite3.Connection, relation_id: str) -> Relation:
        row = connection.execute(
            """
            SELECT relation_import_run_id, library_id, relation_type,
                   subject_type, subject_id, object_type, object_id, reason_text,
                   extraction_method, valid_start, valid_end, evidence_count,
                   status, content_hash
            FROM relations WHERE relation_id = ? AND library_id = ?
            """,
            (relation_id, self.library_id),
        ).fetchone()
        if row is None:
            raise RelationNotFoundError("Relation does not exist")
        if str(row[12]) != "candidate":
            raise RelationIntegrityError("Relation status is not candidate")
        evidence_rows = connection.execute(
            "SELECT evidence_link_id FROM relation_evidence_links "
            "WHERE relation_id = ? ORDER BY evidence_order",
            (relation_id,),
        ).fetchall()
        if len(evidence_rows) != int(row[11]):
            raise RelationIntegrityError("Relation evidence count mismatch")
        relation = Relation(
            relation_id=relation_id,
            import_run_id=str(row[0]),
            library_id=str(row[1]),
            relation_type=RelationType(str(row[2])),
            subject=RelationNodeRef(RelationNodeType(str(row[3])), str(row[4])),
            object_node=RelationNodeRef(RelationNodeType(str(row[5])), str(row[6])),
            evidence_link_ids=tuple(str(item[0]) for item in evidence_rows),
            reason_text=str(row[7]),
            extraction_method=str(row[8]),
            valid_start=None if row[9] is None else str(row[9]),
            valid_end=None if row[10] is None else str(row[10]),
            content_hash=str(row[13]),
        )
        if canonical_content_id("relation", relation.as_proposal().payload()) != relation_id:
            raise RelationIntegrityError("Relation content ID mismatch")
        return relation

    def _prepare_authorized_edges(
        self,
        connection: sqlite3.Connection,
        *,
        permitted_fragment_ids: tuple[str, ...],
        relation_types: tuple[RelationType, ...],
        direction: RelationDirection,
    ) -> None:
        self._prepare_permitted_fragments(connection, permitted_fragment_ids)
        self._prepare_authorized_nodes(connection)
        connection.execute(
            f"CREATE TEMP TABLE {_TEMP_AUTHORIZED} (relation_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        placeholders = ", ".join("?" for _ in relation_types)
        connection.execute(
            f"""
            INSERT INTO {_TEMP_AUTHORIZED}(relation_id)
            SELECT relation.relation_id
            FROM relations AS relation
            JOIN {_TEMP_AUTHORIZED_NODES} AS subject
              ON subject.node_type = relation.subject_type
             AND subject.node_id = relation.subject_id
            JOIN {_TEMP_AUTHORIZED_NODES} AS object_node
              ON object_node.node_type = relation.object_type
             AND object_node.node_id = relation.object_id
            WHERE relation.library_id = ?
              AND relation.relation_type IN ({placeholders})
              AND relation.evidence_count = (
                  SELECT COUNT(*) FROM relation_evidence_links AS link
                  WHERE link.relation_id = relation.relation_id
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM relation_evidence_links AS link
                  JOIN evidence_links AS evidence
                    ON evidence.evidence_link_id = link.evidence_link_id
                  LEFT JOIN {_TEMP_PERMITTED} AS permitted
                    ON permitted.source_fragment_id = evidence.source_fragment_id
                  WHERE link.relation_id = relation.relation_id
                    AND permitted.source_fragment_id IS NULL
              )
            """,
            (self.library_id, *(item.value for item in relation_types)),
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE {_TEMP_EDGES}(
                relation_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                from_type TEXT NOT NULL,
                from_id TEXT NOT NULL,
                to_type TEXT NOT NULL,
                to_id TEXT NOT NULL,
                PRIMARY KEY (relation_id, from_type, from_id, to_type, to_id)
            ) WITHOUT ROWID
            """
        )
        if direction in (RelationDirection.OUTGOING, RelationDirection.BOTH):
            connection.execute(
                f"""
                INSERT INTO {_TEMP_EDGES}
                SELECT relation.relation_id, relation.relation_type,
                       relation.subject_type, relation.subject_id,
                       relation.object_type, relation.object_id
                FROM relations AS relation
                JOIN {_TEMP_AUTHORIZED} AS authorized
                  ON authorized.relation_id = relation.relation_id
                """
            )
        if direction in (RelationDirection.INCOMING, RelationDirection.BOTH):
            connection.execute(
                f"""
                INSERT INTO {_TEMP_EDGES}
                SELECT relation.relation_id, relation.relation_type,
                       relation.object_type, relation.object_id,
                       relation.subject_type, relation.subject_id
                FROM relations AS relation
                JOIN {_TEMP_AUTHORIZED} AS authorized
                  ON authorized.relation_id = relation.relation_id
                """
            )

    def _prepare_permitted_fragments(
        self,
        connection: sqlite3.Connection,
        permitted_fragment_ids: tuple[str, ...],
    ) -> None:
        self._drop_temp_tables(connection)
        connection.execute(
            f"CREATE TEMP TABLE {_TEMP_PERMITTED} "
            "(source_fragment_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.executemany(
            f"INSERT INTO {_TEMP_PERMITTED}(source_fragment_id) VALUES (?)",
            ((item,) for item in permitted_fragment_ids),
        )

    def _prepare_authorized_nodes(self, connection: sqlite3.Connection) -> None:
        """Materialize only nodes whose complete source grounding is permitted."""

        connection.execute(
            f"""
            CREATE TEMP TABLE {_TEMP_AUTHORIZED_NODES}(
                node_type TEXT NOT NULL,
                node_id TEXT NOT NULL,
                PRIMARY KEY (node_type, node_id)
            ) WITHOUT ROWID
            """
        )
        statement_grounding = f"""
            EXISTS (
                SELECT 1 FROM evidence_links AS evidence
                WHERE evidence.statement_id = statement.statement_id
            )
            AND NOT EXISTS (
                SELECT 1
                FROM evidence_links AS evidence
                LEFT JOIN {_TEMP_PERMITTED} AS permitted
                  ON permitted.source_fragment_id = evidence.source_fragment_id
                WHERE evidence.statement_id = statement.statement_id
                  AND permitted.source_fragment_id IS NULL
            )
        """
        connection.execute(
            f"""
            INSERT INTO {_TEMP_AUTHORIZED_NODES}(node_type, node_id)
            SELECT 'statement', statement.statement_id
            FROM statements AS statement
            WHERE statement.library_id = ? AND {statement_grounding}
            """,
            (self.library_id,),
        )

        for node_type, table, mention_table, node_column in (
            ("entity", "entities", "entity_mentions", "entity_id"),
            ("concept", "concepts", "concept_mentions", "concept_id"),
        ):
            connection.execute(
                f"""
                INSERT INTO {_TEMP_AUTHORIZED_NODES}(node_type, node_id)
                SELECT ?, node.{node_column}
                FROM {table} AS node
                WHERE node.library_id = ?
                  AND EXISTS (
                      SELECT 1 FROM {mention_table} AS mention
                      WHERE mention.{node_column} = node.{node_column}
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM {mention_table} AS mention
                      LEFT JOIN {_TEMP_PERMITTED} AS permitted
                        ON permitted.source_fragment_id = mention.source_fragment_id
                      WHERE mention.{node_column} = node.{node_column}
                        AND permitted.source_fragment_id IS NULL
                  )
                """,
                (node_type, self.library_id),
            )
        connection.execute(
            f"""
            INSERT INTO {_TEMP_AUTHORIZED_NODES}(node_type, node_id)
            SELECT 'concept_meaning', meaning.concept_meaning_id
            FROM concept_meanings AS meaning
            JOIN concepts AS concept ON concept.concept_id = meaning.concept_id
            WHERE concept.library_id = ?
              AND EXISTS (
                  SELECT 1
                  FROM concept_meaning_statements AS support
                  JOIN evidence_links AS evidence
                    ON evidence.statement_id = support.statement_id
                  WHERE support.concept_meaning_id = meaning.concept_meaning_id
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM concept_meaning_statements AS support
                  WHERE support.concept_meaning_id = meaning.concept_meaning_id
                    AND NOT EXISTS (
                        SELECT 1 FROM evidence_links AS evidence
                        WHERE evidence.statement_id = support.statement_id
                    )
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM concept_meaning_statements AS support
                  JOIN evidence_links AS evidence
                    ON evidence.statement_id = support.statement_id
                  LEFT JOIN {_TEMP_PERMITTED} AS permitted
                    ON permitted.source_fragment_id = evidence.source_fragment_id
                  WHERE support.concept_meaning_id = meaning.concept_meaning_id
                    AND permitted.source_fragment_id IS NULL
              )
            """,
            (self.library_id,),
        )
        for node_type, table, id_column, statement_predicate in (
            ("voice", "voices", "voice_id", "statement.voice_id = node.voice_id"),
            (
                "time_context",
                "time_contexts",
                "time_context_id",
                "statement.time_context_id = node.time_context_id",
            ),
        ):
            connection.execute(
                f"""
                INSERT INTO {_TEMP_AUTHORIZED_NODES}(node_type, node_id)
                SELECT ?, node.{id_column}
                FROM {table} AS node
                WHERE node.library_id = ?
                  AND EXISTS (
                      SELECT 1
                      FROM statements AS statement
                      JOIN evidence_links AS evidence
                        ON evidence.statement_id = statement.statement_id
                      WHERE {statement_predicate}
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM statements AS statement
                      WHERE {statement_predicate}
                        AND NOT EXISTS (
                            SELECT 1 FROM evidence_links AS evidence
                            WHERE evidence.statement_id = statement.statement_id
                        )
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM statements AS statement
                      JOIN evidence_links AS evidence
                        ON evidence.statement_id = statement.statement_id
                      LEFT JOIN {_TEMP_PERMITTED} AS permitted
                        ON permitted.source_fragment_id = evidence.source_fragment_id
                      WHERE {statement_predicate}
                        AND permitted.source_fragment_id IS NULL
                  )
                """,
                (node_type, self.library_id),
            )
        connection.execute(
            f"""
            INSERT INTO {_TEMP_AUTHORIZED_NODES}(node_type, node_id)
            SELECT 'structure_unit', unit.structure_unit_id
            FROM structure_units AS unit
            JOIN structure_unit_generations AS generation
              ON generation.structure_unit_generation_id = unit.structure_unit_generation_id
            WHERE generation.library_id = ?
              AND EXISTS (
                  SELECT 1 FROM structure_unit_members AS member
                  WHERE member.structure_unit_id = unit.structure_unit_id
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM structure_unit_members AS member
                  LEFT JOIN {_TEMP_PERMITTED} AS permitted
                    ON permitted.source_fragment_id = member.source_fragment_id
                  WHERE member.structure_unit_id = unit.structure_unit_id
                    AND permitted.source_fragment_id IS NULL
              )
            """,
            (self.library_id,),
        )

    @staticmethod
    def _node_grounding_fragments(
        connection: sqlite3.Connection,
        node: RelationNodeRef,
    ) -> tuple[str, ...]:
        query = {
            RelationNodeType.STATEMENT: (
                "SELECT source_fragment_id FROM evidence_links "
                "WHERE statement_id = ? ORDER BY source_fragment_id"
            ),
            RelationNodeType.CONCEPT_MEANING: (
                "SELECT DISTINCT evidence.source_fragment_id "
                "FROM concept_meaning_statements AS support "
                "JOIN evidence_links AS evidence "
                "ON evidence.statement_id = support.statement_id "
                "WHERE support.concept_meaning_id = ? "
                "ORDER BY evidence.source_fragment_id"
            ),
            RelationNodeType.ENTITY: (
                "SELECT source_fragment_id FROM entity_mentions "
                "WHERE entity_id = ? ORDER BY source_fragment_id"
            ),
            RelationNodeType.CONCEPT: (
                "SELECT source_fragment_id FROM concept_mentions "
                "WHERE concept_id = ? ORDER BY source_fragment_id"
            ),
            RelationNodeType.VOICE: (
                "SELECT DISTINCT evidence.source_fragment_id "
                "FROM statements AS statement JOIN evidence_links AS evidence "
                "ON evidence.statement_id = statement.statement_id "
                "WHERE statement.voice_id = ? ORDER BY evidence.source_fragment_id"
            ),
            RelationNodeType.TIME_CONTEXT: (
                "SELECT DISTINCT evidence.source_fragment_id "
                "FROM statements AS statement JOIN evidence_links AS evidence "
                "ON evidence.statement_id = statement.statement_id "
                "WHERE statement.time_context_id = ? ORDER BY evidence.source_fragment_id"
            ),
            RelationNodeType.STRUCTURE_UNIT: (
                "SELECT source_fragment_id FROM structure_unit_members "
                "WHERE structure_unit_id = ? ORDER BY member_order"
            ),
        }[node.node_type]
        return tuple(str(row[0]) for row in connection.execute(query, (node.node_id,)).fetchall())

    def _walk(
        self, connection: sqlite3.Connection, request: RelationPathRequest
    ) -> tuple[sqlite3.Row, ...]:
        values = ", ".join("(?, ?, ?)" for _ in request.seeds)
        seed_parameters: list[object] = []
        for order, seed in enumerate(request.seeds):
            seed_parameters.extend((order, seed.node_type.value, seed.node_id))
        rows = connection.execute(
            f"""
            WITH RECURSIVE
            seeds(seed_order, node_type, node_id) AS (VALUES {values}),
            walk(seed_order, current_type, current_id, depth, relation_path, node_path) AS (
                SELECT seed_order, node_type, node_id, 0, '',
                       '|' || node_type || ':' || node_id || '|'
                FROM seeds
                UNION ALL
                SELECT walk.seed_order, edge.to_type, edge.to_id, walk.depth + 1,
                       CASE WHEN walk.relation_path = '' THEN edge.relation_id
                            ELSE walk.relation_path || ',' || edge.relation_id END,
                       walk.node_path || edge.to_type || ':' || edge.to_id || '|'
                FROM walk
                JOIN {_TEMP_EDGES} AS edge
                  ON edge.from_type = walk.current_type
                 AND edge.from_id = walk.current_id
                WHERE walk.depth < ?
                  AND instr(
                      walk.node_path,
                      '|' || edge.to_type || ':' || edge.to_id || '|'
                  ) = 0
            )
            SELECT seed_order, current_type, current_id, depth, relation_path
            FROM walk
            WHERE depth > 0
            ORDER BY depth, relation_path, seed_order, current_type, current_id
            LIMIT ?
            """,
            (*seed_parameters, request.max_hops, request.max_paths),
        ).fetchall()
        return tuple(rows)

    def _paths_from_rows(
        self,
        request: RelationPathRequest,
        rows: tuple[sqlite3.Row, ...],
        relation_map: dict[str, Relation],
    ) -> tuple[RelationPath, ...]:
        paths: list[RelationPath] = []
        for row in rows:
            seed_order = int(row[0])
            seed = request.seeds[seed_order]
            current = seed
            steps: list[RelationPathStep] = []
            for relation_id in str(row[4]).split(","):
                relation = relation_map[relation_id]
                if relation.subject == current:
                    target = relation.object_node
                elif relation.object_node == current:
                    target = relation.subject
                else:
                    raise RelationIntegrityError("recursive Relation path is discontinuous")
                steps.append(
                    RelationPathStep(
                        relation_id=relation.relation_id,
                        relation_type=relation.relation_type,
                        from_node=current,
                        to_node=target,
                        evidence_link_ids=relation.evidence_link_ids,
                    )
                )
                current = target
            path = RelationPath(seed_order, seed, tuple(steps))
            if current.node_type.value != str(row[1]) or current.node_id != str(row[2]):
                raise RelationIntegrityError("recursive Relation terminal node mismatch")
            paths.append(path)
        return tuple(sorted(paths, key=lambda item: item.path_hash))

    def _persist_path_receipt(self, receipt: RelationPathReceipt) -> None:
        created_at = _timestamp(self._repository._clock)
        exclusions = _exclusion_rows(receipt.request.scope.exclusions)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO relation_path_receipts(
                        relation_path_receipt_id, library_id, corpus_snapshot_id,
                        access_policy_id, snapshot_hash, policy_hash, scope_hash,
                        permitted_set_hash, request_hash, purpose, direction, max_hops,
                        max_paths, seed_count, path_count, receipt_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.receipt_id,
                        receipt.library_id,
                        receipt.corpus_snapshot_id,
                        receipt.access_policy_id,
                        receipt.request.scope.snapshot_hash,
                        receipt.policy_hash,
                        receipt.scope_hash,
                        receipt.permitted_set_hash,
                        receipt.request.request_hash,
                        receipt.request.scope.purpose,
                        receipt.request.direction.value,
                        receipt.request.max_hops,
                        receipt.request.max_paths,
                        len(receipt.request.seeds),
                        len(receipt.paths),
                        receipt.receipt_hash,
                        created_at,
                    ),
                )
                if cursor.rowcount == 1:
                    connection.executemany(
                        "INSERT INTO relation_path_collections VALUES (?, ?)",
                        (
                            (receipt.receipt_id, collection_id)
                            for collection_id in receipt.request.scope.collection_ids
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO relation_path_exclusions VALUES (?, ?, ?)",
                        (
                            (receipt.receipt_id, target_type, target_id)
                            for target_type, target_id in exclusions
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO relation_path_relation_types VALUES (?, ?)",
                        (
                            (receipt.receipt_id, relation_type.value)
                            for relation_type in receipt.request.relation_types
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO relation_path_seeds VALUES (?, ?, ?, ?)",
                        (
                            (
                                receipt.receipt_id,
                                order,
                                seed.node_type.value,
                                seed.node_id,
                            )
                            for order, seed in enumerate(receipt.request.seeds)
                        ),
                    )
                    for path_order, path in enumerate(receipt.paths):
                        connection.execute(
                            "INSERT INTO relation_paths VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                receipt.receipt_id,
                                path_order,
                                path.seed_order,
                                path.terminal.node_type.value,
                                path.terminal.node_id,
                                len(path.steps),
                                path.path_hash,
                            ),
                        )
                        connection.executemany(
                            "INSERT INTO relation_path_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                (
                                    receipt.receipt_id,
                                    path_order,
                                    hop_order,
                                    step.relation_id,
                                    step.from_node.node_type.value,
                                    step.from_node.node_id,
                                    step.to_node.node_type.value,
                                    step.to_node.node_id,
                                )
                                for hop_order, step in enumerate(path.steps)
                            ),
                        )
        except sqlite3.IntegrityError as exc:
            raise RelationIntegrityError("Relation path receipt insert conflicted") from exc
        if self.load_path_receipt(receipt.receipt_id) != receipt:
            raise RelationIntegrityError(
                "persisted Relation path receipt conflicts with its content address"
            )

    def _step_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> RelationPathStep:
        relation = self._load_relation(connection, str(row[0]))
        return RelationPathStep(
            relation_id=relation.relation_id,
            relation_type=relation.relation_type,
            from_node=RelationNodeRef(RelationNodeType(str(row[1])), str(row[2])),
            to_node=RelationNodeRef(RelationNodeType(str(row[3])), str(row[4])),
            evidence_link_ids=relation.evidence_link_ids,
        )

    @staticmethod
    def _drop_temp_tables(connection: sqlite3.Connection) -> None:
        for table in (
            _TEMP_EDGES,
            _TEMP_AUTHORIZED,
            _TEMP_AUTHORIZED_NODES,
            _TEMP_PERMITTED,
        ):
            connection.execute(f"DROP TABLE IF EXISTS {table}")


def _node_query(node_type: RelationNodeType) -> tuple[str, str, str]:
    return {
        RelationNodeType.STATEMENT: (
            "(SELECT statement_id, library_id AS node_library_id FROM statements)",
            "statement_id",
            "",
        ),
        RelationNodeType.CONCEPT: (
            "(SELECT concept_id, library_id AS node_library_id FROM concepts)",
            "concept_id",
            "",
        ),
        RelationNodeType.CONCEPT_MEANING: (
            "(SELECT meaning.concept_meaning_id, concept.library_id AS node_library_id "
            "FROM concept_meanings AS meaning JOIN concepts AS concept "
            "ON concept.concept_id = meaning.concept_id)",
            "concept_meaning_id",
            "",
        ),
        RelationNodeType.ENTITY: (
            "(SELECT entity_id, library_id AS node_library_id FROM entities)",
            "entity_id",
            "",
        ),
        RelationNodeType.VOICE: (
            "(SELECT voice_id, library_id AS node_library_id FROM voices)",
            "voice_id",
            "",
        ),
        RelationNodeType.TIME_CONTEXT: (
            "(SELECT time_context_id, library_id AS node_library_id FROM time_contexts)",
            "time_context_id",
            "",
        ),
        RelationNodeType.STRUCTURE_UNIT: (
            "(SELECT unit.structure_unit_id, generation.library_id AS node_library_id "
            "FROM structure_units AS unit JOIN structure_unit_generations AS generation "
            "ON generation.structure_unit_generation_id = unit.structure_unit_generation_id)",
            "structure_unit_id",
            "",
        ),
    }[node_type]


def _exclusion_rows(exclusions: QueryExclusions) -> tuple[tuple[str, str], ...]:
    rows = [*(("source", item) for item in exclusions.source_ids)]
    rows.extend(("source_family", item) for item in exclusions.source_family_ids)
    rows.extend(("source_fragment", item) for item in exclusions.source_fragment_ids)
    return tuple(sorted(rows))


def _fragment_ids(values: object) -> tuple[str, ...]:
    if (
        type(values) is not tuple
        or not 1 <= len(values) <= 500
        or any(type(item) is not str or not item.startswith("fragment_") for item in values)
        or len(set(values)) != len(values)
    ):
        raise TypeError("source_fragment_ids must be a unique tuple of 1 to 500 SourceFragment IDs")
    return tuple(str(item) for item in values)


def _query_exclusions(rows: Iterable[sqlite3.Row]) -> QueryExclusions:
    source_ids: list[str] = []
    family_ids: list[str] = []
    fragment_ids: list[str] = []
    for row in rows:
        target_type, target_id = str(row[0]), str(row[1])
        if target_type == "source":
            source_ids.append(target_id)
        elif target_type == "source_family":
            family_ids.append(target_id)
        elif target_type == "source_fragment":
            fragment_ids.append(target_id)
        else:
            raise RelationIntegrityError("persisted Relation path exclusion type is invalid")
    return QueryExclusions(tuple(source_ids), tuple(family_ids), tuple(fragment_ids))
