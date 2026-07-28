"""Policy-first construction of the metadata-only Reading Room projection."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import cast

from dithyramba.access import PermittedManifestItem, RequestScope, verify_permitted_token
from dithyramba.contracts import canonical_content_id, canonical_sha256_hex
from dithyramba.meaning import (
    CoverageState,
    MeaningOutputManifest,
    MeaningReviewAction,
    MeaningRunResult,
    MeaningRunStatus,
)
from dithyramba.persistence import LibraryRepository, SQLiteMeaningRepository

from .errors import ReadingRoomAuthorizationError, ReadingRoomIntegrityError
from .models import (
    ReadingRoomAccess,
    ReadingRoomCorpus,
    ReadingRoomEdge,
    ReadingRoomEdgeType,
    ReadingRoomFragmentRef,
    ReadingRoomGraph,
    ReadingRoomLibrary,
    ReadingRoomLimits,
    ReadingRoomMetadata,
    ReadingRoomNode,
    ReadingRoomNodeType,
    ReadingRoomOmissions,
    ReadingRoomProjection,
    ReadingRoomReview,
    ReadingRoomReviewItem,
    ReadingRoomReviewStatusCount,
    ReadingRoomRun,
)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_REVIEW_STATES = frozenset(
    {
        MeaningReviewAction.ACCEPT.value,
        MeaningReviewAction.REJECT.value,
        MeaningReviewAction.REVISE.value,
    }
)
_REVIEWABLE_NODE_TYPES = frozenset(
    {
        ReadingRoomNodeType.STATEMENT,
        ReadingRoomNodeType.EVIDENCE_LINK,
        ReadingRoomNodeType.VOICE,
        ReadingRoomNodeType.ENTITY,
        ReadingRoomNodeType.CONCEPT,
        ReadingRoomNodeType.CONCEPT_MEANING,
        ReadingRoomNodeType.TIME_CONTEXT,
    }
)


@dataclass(frozen=True, slots=True)
class _FragmentMetadata:
    source_fragment_id: str
    source_version_id: str
    source_id: str
    source_family_id: str | None
    collection_ids: tuple[str, ...]
    ordinal: int
    fragment_kind: str
    text_sha256: str
    address_hash: str
    version_number: int
    content_sha256: str
    observed_at: str
    source_modified_at: str | None
    parser_profile: str
    parse_status: str


@dataclass(frozen=True, slots=True)
class _GraphBuild:
    nodes: tuple[ReadingRoomNode, ...]
    edges: tuple[ReadingRoomEdge, ...]
    closure_omission_present: bool


@dataclass(frozen=True, slots=True)
class _DecisionState:
    decision_id: str
    target_type: str
    target_id: str
    target_hash: str
    action: str
    replacement_target_id: str | None
    supersedes_decision_id: str | None
    created_at: str


class ReadingRoomService:
    """Build an immutable dashboard projection without reading source text.

    The service deliberately accepts no fragment/source/version selector. One
    request is always compiled from a persisted snapshot, policy and complete
    ``RequestScope``. SQL is limited to static metadata queries on the already
    open repository connection, whose SourceFragment text authorizer remains
    installed for the entire operation.
    """

    def __init__(
        self,
        repository: LibraryRepository,
        meaning_repository: SQLiteMeaningRepository | None = None,
    ) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("ReadingRoomService requires a LibraryRepository")
        if meaning_repository is not None and not isinstance(
            meaning_repository, SQLiteMeaningRepository
        ):
            raise TypeError("meaning_repository must be SQLiteMeaningRepository")
        if (
            meaning_repository is not None
            and meaning_repository.library_id != repository.library_id
        ):
            raise ValueError("Reading Room repositories belong to different Libraries")
        self._repository = repository
        self._meaning = meaning_repository or SQLiteMeaningRepository(repository)

    def project(
        self,
        *,
        corpus_snapshot_id: str,
        access_policy_id: str,
        scope: RequestScope,
        limits: ReadingRoomLimits | None = None,
    ) -> ReadingRoomProjection:
        """Compile authorization, prove closure, and return bounded metadata."""

        selected_limits = limits or ReadingRoomLimits()
        if not isinstance(selected_limits, ReadingRoomLimits):
            raise TypeError("limits must be ReadingRoomLimits")
        if not isinstance(scope, RequestScope):
            raise ReadingRoomAuthorizationError("reading room request is not authorized")
        if self._repository.schema_version != ReadingRoomProjection.STORAGE_SCHEMA_VERSION:
            raise ReadingRoomIntegrityError("reading room requires the current storage schema")

        try:
            authorization = self._repository.authorize_read(
                access_policy_id=access_policy_id,
                scope=scope,
            )
            compiled = authorization.compiled
            verify_permitted_token(
                compiled.token,
                library_id=self._repository.library_id,
                snapshot_hash=scope.snapshot_hash,
                policy_hash=compiled.token.policy_hash,
                exclusion_hash=scope.exclusion_hash,
                permitted_set_hash=compiled.manifest.permitted_set_hash,
            )
        except Exception:
            # Do not echo identifiers or distinguish absent from denied state.
            raise ReadingRoomAuthorizationError("reading room request is not authorized") from None

        try:
            # A single read transaction keeps all metadata and semantic
            # relations on one SQLite snapshot. It deliberately reuses the
            # repository connection so its SourceFragment text authorizer stays
            # active; no permission context or authorizer is replaced.
            with self._repository._store.transaction() as connection:
                snapshot = self._repository.get_corpus_snapshot(corpus_snapshot_id)
                policy_record = self._repository.get_access_policy(access_policy_id)
                if (
                    snapshot.library_id != self._repository.library_id
                    or snapshot.manifest_hash != scope.snapshot_hash
                    or snapshot.corpus_snapshot_id != corpus_snapshot_id
                    or policy_record.snapshot.policy_hash != compiled.token.policy_hash
                    or compiled.manifest.library_id != self._repository.library_id
                    or compiled.manifest.snapshot_hash != snapshot.manifest_hash
                ):
                    raise ReadingRoomAuthorizationError("reading room request is not authorized")

                snapshot_created_at = self._snapshot_created_at(
                    connection,
                    corpus_snapshot_id=corpus_snapshot_id,
                    manifest_hash=snapshot.manifest_hash,
                )
                fragments = self._load_fragment_metadata(connection, compiled.manifest.items)
                permitted_collections = tuple(
                    sorted(
                        {
                            collection_id
                            for item in compiled.manifest.items
                            for collection_id in item.collection_ids
                        }
                    )
                )
                allowed_pairs = frozenset(
                    (item.source_fragment_id, collection_id)
                    for item in compiled.manifest.items
                    for collection_id in item.collection_ids
                )
                corpus = self._corpus_projection(
                    fragments,
                    permitted_collections,
                    fragment_limit=selected_limits.corpus_fragment_limit,
                )

                matching_run_ids, matching_run_count = self._matching_run_ids(
                    connection,
                    corpus_snapshot_id=corpus_snapshot_id,
                    access_policy_id=access_policy_id,
                    scope_hash=scope.exclusion_hash,
                    permitted_set_hash=compiled.manifest.permitted_set_hash,
                    scan_limit=max(selected_limits.run_limit, selected_limits.provenance_run_limit),
                )
                validated_runs = tuple(
                    self._load_validated_run(run_id, compiled.manifest.items, fragments)
                    for run_id in matching_run_ids
                )
                recent_runs = tuple(
                    self._run_projection(item)
                    for item in validated_runs[: selected_limits.run_limit]
                )

                eligible_runs = tuple(
                    item
                    for item in validated_runs[: selected_limits.provenance_run_limit]
                    if item.run.status is MeaningRunStatus.SUCCEEDED
                    and item.coverage_report.state is CoverageState.COMPLETE
                    and self._receipt_is_complete(item, compiled.manifest.items)
                )
                graph_builds = tuple(
                    self._graph_for_run(
                        connection,
                        result=result,
                        fragments=fragments,
                        allowed_pairs=allowed_pairs,
                    )
                    for result in eligible_runs
                    if result.run.output.candidate_count > 0
                )
                merged = self._merge_graphs(graph_builds)
                reviewed_nodes, review, decision_omission = self._apply_reviews(
                    connection,
                    nodes=merged.nodes,
                    permitted_collections=frozenset(permitted_collections),
                    purpose=scope.purpose,
                    review_limit=selected_limits.review_limit,
                )
                bounded_graph = self._bounded_graph(
                    nodes=reviewed_nodes,
                    edges=merged.edges,
                    limits=selected_limits,
                )

                library = ReadingRoomLibrary(
                    library_id=self._repository.library_id,
                    name=self._repository.library.config.name,
                    logical_identity_hash=self._repository.library.logical_identity_hash,
                    created_at=self._repository.library.created_at,
                    storage_schema_version=self._repository.schema_version,
                    storage_schema_fingerprint=self._repository.schema_fingerprint,
                )
                access = ReadingRoomAccess(
                    corpus_snapshot_id=corpus_snapshot_id,
                    snapshot_hash=snapshot.manifest_hash,
                    snapshot_created_at=snapshot_created_at,
                    access_policy_id=access_policy_id,
                    access_policy_name=policy_record.name,
                    policy_hash=policy_record.snapshot.policy_hash,
                    policy_created_at=policy_record.created_at,
                    purpose=scope.purpose,
                    permitted_collection_ids=permitted_collections,
                    exclusion_hash=scope.exclusion_hash,
                    permitted_set_hash=compiled.manifest.permitted_set_hash,
                    permitted_token_hash=compiled.token.token_hash,
                    policy_omission_present=compiled.public_result.policy_omission_present,
                )
                omissions = ReadingRoomOmissions(
                    limits=selected_limits,
                    provenance_run_scan_truncated=(
                        matching_run_count > selected_limits.provenance_run_limit
                    ),
                    graph_nodes_truncated=(
                        bounded_graph.visible_node_count > len(bounded_graph.nodes)
                    ),
                    graph_edges_truncated=(
                        bounded_graph.visible_edge_count > len(bounded_graph.edges)
                    ),
                    recent_runs_truncated=matching_run_count > len(recent_runs),
                    review_queue_truncated=review.unresolved_count > len(review.queue),
                    corpus_fragments_truncated=(
                        corpus.source_fragment_count > len(corpus.fragments)
                    ),
                    closure_omission_present=(merged.closure_omission_present or decision_omission),
                )
                return ReadingRoomProjection(
                    library=library,
                    access=access,
                    corpus=corpus,
                    recent_runs=recent_runs,
                    review=review,
                    graph=bounded_graph,
                    omissions=omissions,
                )
        except ReadingRoomAuthorizationError:
            raise
        except ReadingRoomIntegrityError:
            raise
        except Exception:
            # SQL and validation failures remain deliberately identifier-free.
            raise ReadingRoomIntegrityError("reading room metadata is inconsistent") from None

    def build_projection(
        self,
        *,
        corpus_snapshot_id: str,
        access_policy_id: str,
        scope: RequestScope,
        limits: ReadingRoomLimits | None = None,
    ) -> ReadingRoomProjection:
        """Named alias for integrations that prefer an explicit builder verb."""

        return self.project(
            corpus_snapshot_id=corpus_snapshot_id,
            access_policy_id=access_policy_id,
            scope=scope,
            limits=limits,
        )

    def _snapshot_created_at(
        self,
        connection: sqlite3.Connection,
        *,
        corpus_snapshot_id: str,
        manifest_hash: str,
    ) -> str:
        row = connection.execute(
            """
            SELECT created_at
            FROM corpus_snapshots
            WHERE corpus_snapshot_id = ? AND library_id = ? AND manifest_hash = ?
            """,
            (corpus_snapshot_id, self._repository.library_id, manifest_hash),
        ).fetchone()
        if row is None:
            raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
        return str(row[0])

    def _load_fragment_metadata(
        self,
        connection: sqlite3.Connection,
        manifest_items: tuple[PermittedManifestItem, ...],
    ) -> dict[str, _FragmentMetadata]:
        # ``manifest_items`` is intentionally accepted structurally here so
        # this adapter never manufactures or broadens a permitted set.
        if not manifest_items:
            return {}
        by_id = {item.source_fragment_id: item for item in manifest_items}
        rows = connection.execute(
            """
            WITH requested(source_fragment_id) AS (
                SELECT value FROM json_each(?)
            )
            SELECT sf.source_fragment_id, sf.source_version_id, sv.source_id,
                   sf.ordinal, sf.fragment_kind, sf.text_sha256, sf.address_hash,
                   sv.version_number, sv.content_sha256, sv.observed_at,
                   sv.source_modified_at, sv.parser_profile, sv.parse_status
            FROM requested AS requested
            JOIN source_fragments AS sf
              ON sf.source_fragment_id = requested.source_fragment_id
            JOIN source_versions AS sv ON sv.source_version_id = sf.source_version_id
            JOIN sources AS s ON s.source_id = sv.source_id
            WHERE s.library_id = ?
            ORDER BY sf.source_fragment_id
            """,
            (_json_array(tuple(sorted(by_id))), self._repository.library_id),
        ).fetchall()
        if len(rows) != len(by_id):
            raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
        result: dict[str, _FragmentMetadata] = {}
        for row in rows:
            fragment_id = str(row[0])
            item = by_id.get(fragment_id)
            if item is None:
                raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
            source_version_id = item.source_version_id
            source_id = item.source_id
            if str(row[1]) != source_version_id or str(row[2]) != source_id:
                raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
            text_hash = _hash(str(row[5]))
            address_hash = _hash(str(row[6]))
            content_hash = _hash(str(row[8]))
            result[fragment_id] = _FragmentMetadata(
                source_fragment_id=fragment_id,
                source_version_id=source_version_id,
                source_id=source_id,
                source_family_id=item.source_family_id,
                collection_ids=item.collection_ids,
                ordinal=int(row[3]),
                fragment_kind=str(row[4]),
                text_sha256=text_hash,
                address_hash=address_hash,
                version_number=int(row[7]),
                content_sha256=content_hash,
                observed_at=str(row[9]),
                source_modified_at=None if row[10] is None else str(row[10]),
                parser_profile=str(row[11]),
                parse_status=str(row[12]),
            )
        if set(result) != set(by_id):
            raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
        return result

    def _corpus_projection(
        self,
        fragments: dict[str, _FragmentMetadata],
        permitted_collections: tuple[str, ...],
        *,
        fragment_limit: int,
    ) -> ReadingRoomCorpus:
        source_ids = {item.source_id for item in fragments.values()}
        version_ids = {item.source_version_id for item in fragments.values()}
        observed = tuple(item.observed_at for item in fragments.values())
        modified = tuple(
            item.source_modified_at
            for item in fragments.values()
            if item.source_modified_at is not None
        )
        return ReadingRoomCorpus(
            collection_count=len(permitted_collections),
            source_count=len(source_ids),
            source_version_count=len(version_ids),
            source_fragment_count=len(fragments),
            oldest_source_observed_at=min(observed) if observed else None,
            newest_source_observed_at=max(observed) if observed else None,
            newest_source_modified_at=max(modified) if modified else None,
            parser_profiles=tuple(item.parser_profile for item in fragments.values()),
            fragments=tuple(
                ReadingRoomFragmentRef(
                    source_fragment_id=item.source_fragment_id,
                    source_version_id=item.source_version_id,
                    source_id=item.source_id,
                    source_family_id=item.source_family_id,
                    collection_ids=item.collection_ids,
                    text_sha256=item.text_sha256,
                    address_hash=item.address_hash,
                    ordinal=item.ordinal,
                    fragment_kind=item.fragment_kind,
                    version_number=item.version_number,
                    content_sha256=item.content_sha256,
                    observed_at=item.observed_at,
                    source_modified_at=item.source_modified_at,
                    parser_profile=item.parser_profile,
                    parse_status=item.parse_status,
                )
                for item in sorted(fragments.values(), key=lambda value: value.source_fragment_id)[
                    :fragment_limit
                ]
            ),
        )

    def _matching_run_ids(
        self,
        connection: sqlite3.Connection,
        *,
        corpus_snapshot_id: str,
        access_policy_id: str,
        scope_hash: str,
        permitted_set_hash: str,
        scan_limit: int,
    ) -> tuple[tuple[str, ...], int]:
        parameters = (
            self._repository.library_id,
            corpus_snapshot_id,
            access_policy_id,
            scope_hash,
            permitted_set_hash,
        )
        total = int(
            connection.execute(
                """
                SELECT count(*)
                FROM meaning_processing_runs
                WHERE library_id = ? AND corpus_snapshot_id = ?
                  AND access_policy_id = ? AND scope_hash = ?
                  AND permitted_set_hash = ?
                """,
                parameters,
            ).fetchone()[0]
        )
        rows = connection.execute(
            """
            SELECT processing_run_id
            FROM meaning_processing_runs
            WHERE library_id = ? AND corpus_snapshot_id = ?
              AND access_policy_id = ? AND scope_hash = ?
              AND permitted_set_hash = ?
            ORDER BY finished_at DESC, processing_run_id DESC
            LIMIT ?
            """,
            (*parameters, scan_limit),
        ).fetchall()
        return tuple(str(row[0]) for row in rows), total

    def _load_validated_run(
        self,
        processing_run_id: str,
        manifest_items: tuple[PermittedManifestItem, ...],
        fragments: dict[str, _FragmentMetadata],
    ) -> MeaningRunResult:
        try:
            result = self._meaning.get_run(processing_run_id)
        except Exception:
            raise ReadingRoomIntegrityError("reading room metadata is inconsistent") from None

        expected = tuple(
            (
                item.source_fragment_id,
                item.source_version_id,
                item.source_id,
                item.collection_ids,
            )
            for item in manifest_items
        )
        actual_items = result.read_receipt.items
        if actual_items:
            if len(actual_items) != len(expected):
                raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
            input_fragments: list[dict[str, object]] = []
            receipt_items: list[dict[str, object]] = []
            pairs = zip(actual_items, expected, strict=True)
            for index, (actual, expected_item) in enumerate(pairs):
                fragment_id, version_id, source_id, collection_ids = expected_item
                fragment = fragments.get(fragment_id)
                if (
                    fragment is None
                    or actual.source_fragment_id != fragment_id
                    or actual.source_version_id != version_id
                    or actual.source_id != source_id
                    or actual.collection_ids != collection_ids
                    or actual.read_order != index
                    or actual.text_sha256 != fragment.text_sha256
                ):
                    raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
                input_fragments.append(actual.input_payload())
                receipt_items.append(actual.receipt_payload())
        else:
            input_fragments = []
            receipt_items = []

        input_hash = canonical_sha256_hex(
            {"schema": "dithyramba.meaning_input/2.0", "fragments": input_fragments}
        )
        receipt_payload: dict[str, object] = {
            "schema": "dithyramba.meaning_read_receipt/2.0",
            "processing_run_id": result.run.processing_run_id,
            "input_hash": input_hash,
            "items": receipt_items,
        }
        if (
            result.run.input_hash != input_hash
            or result.read_receipt.input_hash != input_hash
            or result.read_receipt.receipt_hash != canonical_sha256_hex(receipt_payload)
            or result.read_receipt.read_receipt_id != canonical_content_id("read", receipt_payload)
            or result.coverage_report.read_fragment_count != len(actual_items)
        ):
            raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
        coverage_payload: dict[str, object] = {
            "schema": "dithyramba.meaning_coverage_report/2.0",
            "processing_run_id": result.run.processing_run_id,
            "profile": result.coverage_report.profile.value,
            "state": result.coverage_report.state.value,
            "read_fragment_count": result.coverage_report.read_fragment_count,
            "candidate_count": result.coverage_report.candidate_count,
            "omissions": [item.payload() for item in result.coverage_report.omissions],
        }
        if result.coverage_report.report_hash != canonical_sha256_hex(
            coverage_payload
        ) or result.coverage_report.coverage_report_id != canonical_content_id(
            "coverage", coverage_payload
        ):
            raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
        return result

    def _receipt_is_complete(
        self,
        result: MeaningRunResult,
        manifest_items: tuple[PermittedManifestItem, ...],
    ) -> bool:
        return tuple(item.source_fragment_id for item in result.read_receipt.items) == tuple(
            item.source_fragment_id for item in manifest_items
        )

    def _run_projection(self, result: MeaningRunResult) -> ReadingRoomRun:
        return ReadingRoomRun(
            processing_run_id=result.run.processing_run_id,
            kind=result.run.kind.value,
            status=result.run.status.value,
            extraction_profile=result.run.profile.value,
            code_version=result.run.code_version,
            generator_hash=result.run.generator_hash,
            review_budget_hash=result.run.review_budget_hash,
            input_hash=result.run.input_hash,
            proposal_hash=result.run.proposal_hash,
            output_hash=result.run.output.output_hash,
            run_receipt_hash=result.run.receipt_hash,
            started_at=result.run.started_at,
            finished_at=result.run.finished_at,
            coverage_report_id=result.coverage_report.coverage_report_id,
            coverage_state=result.coverage_report.state.value,
            coverage_report_hash=result.coverage_report.report_hash,
            read_receipt_id=result.read_receipt.read_receipt_id,
            read_receipt_hash=result.read_receipt.receipt_hash,
            read_fragment_count=result.coverage_report.read_fragment_count,
            candidate_count=result.coverage_report.candidate_count,
            omission_count=sum(item.count for item in result.coverage_report.omissions),
            backlog_warning=result.run.backlog_warning,
            error_code=result.run.error_code,
        )

    def _graph_for_run(
        self,
        connection: sqlite3.Connection,
        *,
        result: MeaningRunResult,
        fragments: dict[str, _FragmentMetadata],
        allowed_pairs: frozenset[tuple[str, str]],
    ) -> _GraphBuild:
        output = result.run.output
        voices = self._voice_rows(connection, output)
        entities = self._entity_rows(connection, output)
        entity_mentions = self._entity_mention_rows(connection, output)
        concepts = self._concept_rows(connection, output)
        concept_mentions = self._concept_mention_rows(connection, output)
        time_contexts = self._time_context_rows(connection, output)
        statements = self._statement_rows(connection, output)
        evidence_links = self._evidence_rows(connection, output)
        meanings = self._concept_meaning_rows(connection, output)
        support = self._concept_meaning_support_rows(connection, output)

        closure_omission = False
        evidence_by_statement: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in evidence_links.values():
            evidence_by_statement[str(row[1])].append(row)

        visible_statements: dict[str, sqlite3.Row] = {}
        visible_evidence: dict[str, sqlite3.Row] = {}
        for statement_id, statement in sorted(statements.items()):
            collection_id = str(statement[1])
            voice_id = str(statement[4])
            time_context_id = None if statement[5] is None else str(statement[5])
            evidence_rows = evidence_by_statement.get(statement_id, [])
            valid = (
                voice_id in voices
                and (time_context_id is None or time_context_id in time_contexts)
                and bool(evidence_rows)
                and all(
                    str(item[2]) == voice_id
                    and (str(item[3]), collection_id) in allowed_pairs
                    and str(item[3]) in fragments
                    for item in evidence_rows
                )
            )
            if not valid:
                closure_omission = True
                continue
            visible_statements[statement_id] = statement
            visible_evidence.update((str(item[0]), item) for item in evidence_rows)

        # Output evidence not attached to an output-visible Statement is an
        # incomplete per-run closure and is omitted without probing its target.
        if len(visible_evidence) != len(evidence_links):
            closure_omission = True

        visible_entity_mentions = {
            mention_id: row
            for mention_id, row in entity_mentions.items()
            if str(row[1]) in entities
            and (str(row[3]), str(row[2])) in allowed_pairs
            and str(row[3]) in fragments
        }
        if len(visible_entity_mentions) != len(entity_mentions):
            closure_omission = True

        visible_concept_mentions = {
            mention_id: row
            for mention_id, row in concept_mentions.items()
            if str(row[1]) in concepts
            and (str(row[3]), str(row[2])) in allowed_pairs
            and str(row[3]) in fragments
        }
        if len(visible_concept_mentions) != len(concept_mentions):
            closure_omission = True

        support_by_meaning: dict[str, list[str]] = defaultdict(list)
        for row in support:
            support_by_meaning[str(row[0])].append(str(row[1]))
        visible_meanings: dict[str, sqlite3.Row] = {}
        for meaning_id, row in sorted(meanings.items()):
            concept_id = str(row[1])
            voice_id = str(row[2])
            time_context_id = None if row[3] is None else str(row[3])
            statement_ids = support_by_meaning.get(meaning_id, [])
            valid = (
                concept_id in concepts
                and voice_id in voices
                and (time_context_id is None or time_context_id in time_contexts)
                and bool(statement_ids)
                and all(
                    statement_id in visible_statements
                    and str(visible_statements[statement_id][4]) == voice_id
                    for statement_id in statement_ids
                )
            )
            if valid:
                visible_meanings[meaning_id] = row
            else:
                closure_omission = True

        used_entities = {str(row[1]) for row in visible_entity_mentions.values()}
        used_concepts = {str(row[1]) for row in visible_concept_mentions.values()} | {
            str(row[1]) for row in visible_meanings.values()
        }
        used_voices = {str(row[4]) for row in visible_statements.values()} | {
            str(row[2]) for row in visible_meanings.values()
        }
        used_times = {str(row[5]) for row in visible_statements.values() if row[5] is not None} | {
            str(row[3]) for row in visible_meanings.values() if row[3] is not None
        }
        if set(entities) != used_entities or set(concepts) != used_concepts:
            closure_omission = True
        if set(voices) != used_voices or set(time_contexts) != used_times:
            closure_omission = True

        entities = {key: value for key, value in entities.items() if key in used_entities}
        concepts = {key: value for key, value in concepts.items() if key in used_concepts}
        voices = {key: value for key, value in voices.items() if key in used_voices}
        time_contexts = {key: value for key, value in time_contexts.items() if key in used_times}

        direct_fragments = (
            {str(row[3]) for row in visible_entity_mentions.values()}
            | {str(row[3]) for row in visible_concept_mentions.values()}
            | {str(row[3]) for row in visible_evidence.values()}
        )
        provenance_run_id = result.run.processing_run_id
        nodes: dict[str, ReadingRoomNode] = {}
        for fragment_id in sorted(direct_fragments):
            fragment = fragments[fragment_id]
            nodes[fragment_id] = ReadingRoomNode(
                node_id=fragment_id,
                node_type=ReadingRoomNodeType.SOURCE_FRAGMENT,
                kind=fragment.fragment_kind,
                label=None,
                candidate_state="permitted",
                review_state="not_reviewable",
                latest_review_decision_id=None,
                content_hash=fragment.text_sha256,
                provenance_processing_run_id=None,
                provenance_fragment_ids=(fragment_id,),
                collection_ids=fragment.collection_ids,
                created_at=None,
                metadata=(
                    ReadingRoomMetadata("address_hash", fragment.address_hash),
                    ReadingRoomMetadata("content_sha256", fragment.content_sha256),
                    ReadingRoomMetadata("ordinal", fragment.ordinal),
                    ReadingRoomMetadata("parse_status", fragment.parse_status),
                    ReadingRoomMetadata("parser_profile", fragment.parser_profile),
                    ReadingRoomMetadata("source_id", fragment.source_id),
                    ReadingRoomMetadata("source_version_id", fragment.source_version_id),
                    ReadingRoomMetadata("version_number", fragment.version_number),
                ),
            )

        statement_fragments: dict[str, tuple[str, ...]] = {
            statement_id: tuple(
                sorted({str(item[3]) for item in evidence_by_statement[statement_id]})
            )
            for statement_id in visible_statements
        }
        for statement_id, row in visible_statements.items():
            nodes[statement_id] = self._semantic_node(
                node_id=statement_id,
                node_type=ReadingRoomNodeType.STATEMENT,
                kind=str(row[2]),
                label=str(row[3]),
                content_hash=str(row[8]),
                status=str(row[6]),
                provenance_run_id=provenance_run_id,
                provenance_fragments=statement_fragments[statement_id],
                collection_ids=(str(row[1]),),
                created_at=str(row[10]),
                metadata=(ReadingRoomMetadata("lifecycle", str(row[7])),),
            )
        for evidence_id, row in visible_evidence.items():
            statement_id = str(row[1])
            nodes[evidence_id] = self._semantic_node(
                node_id=evidence_id,
                node_type=ReadingRoomNodeType.EVIDENCE_LINK,
                kind=str(row[4]),
                label=None,
                content_hash=str(row[11]),
                status=str(row[10]),
                provenance_run_id=provenance_run_id,
                provenance_fragments=(str(row[3]),),
                collection_ids=(str(visible_statements[statement_id][1]),),
                created_at=str(row[13]),
                metadata=(
                    ReadingRoomMetadata("alignment", str(row[5])),
                    ReadingRoomMetadata("attributed_voice_id", str(row[2])),
                    ReadingRoomMetadata("extraction_method", str(row[9])),
                    ReadingRoomMetadata("quote_end", int(row[8])),
                    ReadingRoomMetadata("quote_sha256", str(row[6])),
                    ReadingRoomMetadata("quote_start", int(row[7])),
                ),
            )

        entity_collections: dict[str, set[str]] = defaultdict(set)
        entity_fragments: dict[str, set[str]] = defaultdict(set)
        for mention_id, row in visible_entity_mentions.items():
            entity_id, collection_id, fragment_id = str(row[1]), str(row[2]), str(row[3])
            entity_collections[entity_id].add(collection_id)
            entity_fragments[entity_id].add(fragment_id)
            nodes[mention_id] = self._semantic_node(
                node_id=mention_id,
                node_type=ReadingRoomNodeType.ENTITY_MENTION,
                kind=None,
                label=None,
                content_hash=str(row[7]),
                status="candidate",
                provenance_run_id=provenance_run_id,
                provenance_fragments=(fragment_id,),
                collection_ids=(collection_id,),
                created_at=result.run.finished_at,
                metadata=(
                    ReadingRoomMetadata("quote_end", int(row[5])),
                    ReadingRoomMetadata("quote_sha256", str(row[6])),
                    ReadingRoomMetadata("quote_start", int(row[4])),
                ),
            )
        for entity_id, row in entities.items():
            nodes[entity_id] = self._semantic_node(
                node_id=entity_id,
                node_type=ReadingRoomNodeType.ENTITY,
                kind=str(row[1]),
                label=str(row[2]),
                content_hash=str(row[4]),
                status=str(row[3]),
                provenance_run_id=provenance_run_id,
                provenance_fragments=tuple(entity_fragments[entity_id]),
                collection_ids=tuple(entity_collections[entity_id]),
                created_at=str(row[6]),
            )

        concept_collections: dict[str, set[str]] = defaultdict(set)
        concept_fragments: dict[str, set[str]] = defaultdict(set)
        for mention_id, row in visible_concept_mentions.items():
            concept_id, collection_id, fragment_id = str(row[1]), str(row[2]), str(row[3])
            concept_collections[concept_id].add(collection_id)
            concept_fragments[concept_id].add(fragment_id)
            nodes[mention_id] = self._semantic_node(
                node_id=mention_id,
                node_type=ReadingRoomNodeType.CONCEPT_MENTION,
                kind=None,
                label=None,
                content_hash=str(row[7]),
                status="candidate",
                provenance_run_id=provenance_run_id,
                provenance_fragments=(fragment_id,),
                collection_ids=(collection_id,),
                created_at=result.run.finished_at,
                metadata=(
                    ReadingRoomMetadata("quote_end", int(row[5])),
                    ReadingRoomMetadata("quote_sha256", str(row[6])),
                    ReadingRoomMetadata("quote_start", int(row[4])),
                ),
            )

        meaning_fragments: dict[str, set[str]] = defaultdict(set)
        meaning_collections: dict[str, set[str]] = defaultdict(set)
        for meaning_id in visible_meanings:
            for statement_id in support_by_meaning[meaning_id]:
                meaning_fragments[meaning_id].update(statement_fragments[statement_id])
                meaning_collections[meaning_id].add(str(visible_statements[statement_id][1]))
        for concept_id, row in concepts.items():
            for meaning_id, meaning in visible_meanings.items():
                if str(meaning[1]) == concept_id:
                    concept_fragments[concept_id].update(meaning_fragments[meaning_id])
                    concept_collections[concept_id].update(meaning_collections[meaning_id])
            nodes[concept_id] = self._semantic_node(
                node_id=concept_id,
                node_type=ReadingRoomNodeType.CONCEPT,
                kind=None,
                label=str(row[1]),
                content_hash=str(row[3]),
                status=str(row[2]),
                provenance_run_id=provenance_run_id,
                provenance_fragments=tuple(concept_fragments[concept_id]),
                collection_ids=tuple(concept_collections[concept_id]),
                created_at=str(row[5]),
            )

        voice_fragments: dict[str, set[str]] = defaultdict(set)
        voice_collections: dict[str, set[str]] = defaultdict(set)
        time_fragments: dict[str, set[str]] = defaultdict(set)
        time_collections: dict[str, set[str]] = defaultdict(set)
        for statement_id, statement in visible_statements.items():
            voice_id = str(statement[4])
            collection_id = str(statement[1])
            voice_fragments[voice_id].update(statement_fragments[statement_id])
            voice_collections[voice_id].add(collection_id)
            if statement[5] is not None:
                time_id = str(statement[5])
                time_fragments[time_id].update(statement_fragments[statement_id])
                time_collections[time_id].add(collection_id)
        for meaning_id, meaning in visible_meanings.items():
            voice_id = str(meaning[2])
            voice_fragments[voice_id].update(meaning_fragments[meaning_id])
            voice_collections[voice_id].update(meaning_collections[meaning_id])
            if meaning[3] is not None:
                time_id = str(meaning[3])
                time_fragments[time_id].update(meaning_fragments[meaning_id])
                time_collections[time_id].update(meaning_collections[meaning_id])
            nodes[meaning_id] = self._semantic_node(
                node_id=meaning_id,
                node_type=ReadingRoomNodeType.CONCEPT_MEANING,
                kind=None,
                label=str(meaning[4]),
                content_hash=str(meaning[6]),
                status=str(meaning[5]),
                provenance_run_id=provenance_run_id,
                provenance_fragments=tuple(meaning_fragments[meaning_id]),
                collection_ids=tuple(meaning_collections[meaning_id]),
                created_at=str(meaning[8]),
            )
        for voice_id, row in voices.items():
            nodes[voice_id] = self._semantic_node(
                node_id=voice_id,
                node_type=ReadingRoomNodeType.VOICE,
                kind=str(row[1]),
                label=str(row[2]),
                content_hash=str(row[4]),
                status=str(row[3]),
                provenance_run_id=provenance_run_id,
                provenance_fragments=tuple(voice_fragments[voice_id]),
                collection_ids=tuple(voice_collections[voice_id]),
                created_at=str(row[6]),
            )
        for time_id, row in time_contexts.items():
            metadata = tuple(
                ReadingRoomMetadata(key, str(row[index]))
                for index, key in enumerate(
                    (
                        "event_valid_start",
                        "event_valid_end",
                        "recorded_at",
                        "available_at",
                        "revealed_at",
                        "reviewed_at",
                    ),
                    start=1,
                )
                if row[index] is not None
            )
            nodes[time_id] = self._semantic_node(
                node_id=time_id,
                node_type=ReadingRoomNodeType.TIME_CONTEXT,
                kind=None,
                label=None,
                content_hash=str(row[8]),
                status=str(row[7]),
                provenance_run_id=provenance_run_id,
                provenance_fragments=tuple(time_fragments[time_id]),
                collection_ids=tuple(time_collections[time_id]),
                created_at=str(row[10]),
                metadata=metadata,
            )

        edges: dict[str, ReadingRoomEdge] = {}
        for statement_id, statement in visible_statements.items():
            self._add_edge(
                edges,
                ReadingRoomEdgeType.STATEMENT_VOICE,
                statement_id,
                str(statement[4]),
            )
            if statement[5] is not None:
                self._add_edge(
                    edges,
                    ReadingRoomEdgeType.STATEMENT_TIME_CONTEXT,
                    statement_id,
                    str(statement[5]),
                )
        for evidence_id, evidence in visible_evidence.items():
            self._add_edge(
                edges,
                ReadingRoomEdgeType.STATEMENT_EVIDENCE_LINK,
                str(evidence[1]),
                evidence_id,
            )
            self._add_edge(
                edges,
                ReadingRoomEdgeType.EVIDENCE_LINK_FRAGMENT,
                evidence_id,
                str(evidence[3]),
            )
        for mention_id, mention in visible_entity_mentions.items():
            self._add_edge(
                edges,
                ReadingRoomEdgeType.ENTITY_MENTION_ENTITY,
                mention_id,
                str(mention[1]),
            )
            self._add_edge(
                edges,
                ReadingRoomEdgeType.ENTITY_MENTION_FRAGMENT,
                mention_id,
                str(mention[3]),
            )
        for mention_id, mention in visible_concept_mentions.items():
            self._add_edge(
                edges,
                ReadingRoomEdgeType.CONCEPT_MENTION_CONCEPT,
                mention_id,
                str(mention[1]),
            )
            self._add_edge(
                edges,
                ReadingRoomEdgeType.CONCEPT_MENTION_FRAGMENT,
                mention_id,
                str(mention[3]),
            )
        for meaning_id, meaning in visible_meanings.items():
            self._add_edge(
                edges,
                ReadingRoomEdgeType.CONCEPT_MEANING_CONCEPT,
                meaning_id,
                str(meaning[1]),
            )
            self._add_edge(
                edges,
                ReadingRoomEdgeType.CONCEPT_MEANING_VOICE,
                meaning_id,
                str(meaning[2]),
            )
            if meaning[3] is not None:
                self._add_edge(
                    edges,
                    ReadingRoomEdgeType.CONCEPT_MEANING_TIME_CONTEXT,
                    meaning_id,
                    str(meaning[3]),
                )
            for statement_id in support_by_meaning[meaning_id]:
                self._add_edge(
                    edges,
                    ReadingRoomEdgeType.CONCEPT_MEANING_STATEMENT,
                    meaning_id,
                    statement_id,
                )

        return _GraphBuild(tuple(nodes.values()), tuple(edges.values()), closure_omission)

    def _semantic_node(
        self,
        *,
        node_id: str,
        node_type: ReadingRoomNodeType,
        kind: str | None,
        label: str | None,
        content_hash: str,
        status: str,
        provenance_run_id: str,
        provenance_fragments: tuple[str, ...],
        collection_ids: tuple[str, ...],
        created_at: str,
        metadata: tuple[ReadingRoomMetadata, ...] = (),
    ) -> ReadingRoomNode:
        return ReadingRoomNode(
            node_id=node_id,
            node_type=node_type,
            kind=kind,
            label=label,
            candidate_state=status,
            review_state="unreviewed",
            latest_review_decision_id=None,
            content_hash=_hash(content_hash),
            provenance_processing_run_id=provenance_run_id,
            provenance_fragment_ids=provenance_fragments,
            collection_ids=collection_ids,
            created_at=created_at,
            metadata=metadata,
        )

    def _add_edge(
        self,
        edges: dict[str, ReadingRoomEdge],
        edge_type: ReadingRoomEdgeType,
        source_node_id: str,
        target_node_id: str,
    ) -> None:
        payload = {
            "schema": "dithyramba.reading_room_relation/1.0",
            "edge_type": edge_type.value,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        relation_hash = canonical_sha256_hex(payload)
        edge = ReadingRoomEdge(
            edge_id=canonical_content_id("rr_edge", payload),
            edge_type=edge_type,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_hash=relation_hash,
        )
        previous = edges.setdefault(edge.edge_id, edge)
        if previous != edge:
            raise ReadingRoomIntegrityError("reading room metadata is inconsistent")

    def _merge_graphs(self, builds: tuple[_GraphBuild, ...]) -> _GraphBuild:
        nodes: dict[str, ReadingRoomNode] = {}
        edges: dict[str, ReadingRoomEdge] = {}
        closure_omission = False
        for build in builds:
            closure_omission = closure_omission or build.closure_omission_present
            for node in build.nodes:
                previous = nodes.get(node.node_id)
                nodes[node.node_id] = node if previous is None else _merged_node(previous, node)
            for edge in build.edges:
                previous_edge = edges.setdefault(edge.edge_id, edge)
                if previous_edge != edge:
                    raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
        return _GraphBuild(tuple(nodes.values()), tuple(edges.values()), closure_omission)

    def _apply_reviews(
        self,
        connection: sqlite3.Connection,
        *,
        nodes: tuple[ReadingRoomNode, ...],
        permitted_collections: frozenset[str],
        purpose: str,
        review_limit: int,
    ) -> tuple[tuple[ReadingRoomNode, ...], ReadingRoomReview, bool]:
        candidates = {
            node.node_id: node for node in nodes if node.node_type in _REVIEWABLE_NODE_TYPES
        }
        if not candidates:
            return (
                nodes,
                ReadingRoomReview(0, 0, (), ()),
                False,
            )
        rows = connection.execute(
            """
            WITH visible(target_id) AS (SELECT value FROM json_each(?))
            SELECT d.review_decision_id, d.target_type, d.target_id,
                   d.target_hash, d.action, d.scope_json,
                   d.replacement_target_id, d.supersedes_review_decision_id,
                   d.created_at
            FROM meaning_review_decisions AS d
            JOIN visible AS v ON v.target_id = d.target_id
            WHERE d.library_id = ?
            ORDER BY d.created_at, d.review_decision_id
            """,
            (_json_array(tuple(sorted(candidates))), self._repository.library_id),
        ).fetchall()
        accepted: dict[str, _DecisionState] = {}
        omission = False
        for row in rows:
            target_id = str(row[2])
            node = candidates.get(target_id)
            try:
                scope_payload = json.loads(str(row[5]))
                scope_collections = tuple(scope_payload["collection_ids"])
                scope_use = str(scope_payload["use"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                omission = True
                continue
            target_type = str(row[1])
            expected_target_type = node.node_type.value if node is not None else ""
            replacement = None if row[6] is None else str(row[6])
            supersedes = None if row[7] is None else str(row[7])
            valid = (
                node is not None
                and target_type == expected_target_type
                and str(row[3]) == node.content_hash
                and bool(scope_collections)
                and set(scope_collections).issubset(permitted_collections)
                and set(scope_collections).issubset(node.collection_ids)
                and scope_use == purpose
                and str(row[4]) in {item.value for item in MeaningReviewAction}
            )
            if replacement is not None and replacement not in candidates:
                valid = False
            if not valid:
                omission = True
                continue
            decision = _DecisionState(
                decision_id=str(row[0]),
                target_type=target_type,
                target_id=target_id,
                target_hash=str(row[3]),
                action=str(row[4]),
                replacement_target_id=replacement,
                supersedes_decision_id=supersedes,
                created_at=str(row[8]),
            )
            accepted[decision.decision_id] = decision

        # Supersede chains are only visible if their referenced prior decision
        # passed the exact same access filters.
        accepted = {
            decision_id: decision
            for decision_id, decision in accepted.items()
            if decision.supersedes_decision_id is None
            or decision.supersedes_decision_id in accepted
        }
        latest: dict[str, _DecisionState] = {}
        for decision in sorted(
            accepted.values(), key=lambda item: (item.created_at, item.decision_id)
        ):
            latest[decision.target_id] = decision

        reviewed_nodes: list[ReadingRoomNode] = []
        for node in nodes:
            latest_decision = latest.get(node.node_id)
            if latest_decision is None or node.node_type not in _REVIEWABLE_NODE_TYPES:
                reviewed_nodes.append(node)
            else:
                reviewed_nodes.append(
                    replace(
                        node,
                        review_state=latest_decision.action,
                        latest_review_decision_id=latest_decision.decision_id,
                    )
                )
        reviewable_nodes = tuple(
            node for node in reviewed_nodes if node.node_type in _REVIEWABLE_NODE_TYPES
        )
        counts = Counter(node.review_state for node in reviewable_nodes)
        unresolved = tuple(
            node for node in reviewable_nodes if node.review_state not in _TERMINAL_REVIEW_STATES
        )
        queue = tuple(
            ReadingRoomReviewItem(
                target_type=node.node_type.value,
                target_id=node.node_id,
                target_hash=node.content_hash,
                collection_ids=node.collection_ids,
                created_at=cast(str, node.created_at),
                review_state=node.review_state,
                latest_review_decision_id=node.latest_review_decision_id,
            )
            for node in sorted(
                unresolved,
                key=lambda item: (cast(str, item.created_at), item.node_type.value, item.node_id),
            )[:review_limit]
        )
        review = ReadingRoomReview(
            visible_candidate_count=len(reviewable_nodes),
            unresolved_count=len(unresolved),
            status_counts=tuple(
                ReadingRoomReviewStatusCount(state, count)
                for state, count in sorted(counts.items())
            ),
            queue=queue,
        )
        return tuple(reviewed_nodes), review, omission

    def _bounded_graph(
        self,
        *,
        nodes: tuple[ReadingRoomNode, ...],
        edges: tuple[ReadingRoomEdge, ...],
        limits: ReadingRoomLimits,
    ) -> ReadingRoomGraph:
        node_by_id = {node.node_id: node for node in nodes}
        sorted_edges = tuple(
            sorted(
                edges,
                key=lambda item: (
                    item.edge_type.value,
                    item.source_node_id,
                    item.target_node_id,
                    item.edge_id,
                ),
            )
        )
        selected_ids: set[str] = set()
        selected_edges: list[ReadingRoomEdge] = []
        for edge in sorted_edges:
            if len(selected_edges) >= limits.edge_limit:
                break
            needed = {edge.source_node_id, edge.target_node_id} - selected_ids
            if len(selected_ids) + len(needed) > limits.node_limit:
                continue
            selected_ids.update(needed)
            selected_edges.append(edge)
        for node in sorted(nodes, key=lambda item: (item.node_type.value, item.node_id)):
            if len(selected_ids) >= limits.node_limit:
                break
            selected_ids.add(node.node_id)
        selected_nodes = tuple(node_by_id[node_id] for node_id in sorted(selected_ids))
        return ReadingRoomGraph(
            nodes=selected_nodes,
            edges=tuple(selected_edges),
            visible_node_count=len(nodes),
            visible_edge_count=len(edges),
        )

    def _voice_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> dict[str, sqlite3.Row]:
        return self._rows_by_ids(
            connection,
            ids=output.voice_ids,
            query="""
                WITH requested(id) AS (SELECT value FROM json_each(?))
                SELECT v.voice_id, v.kind, v.label, v.status, v.content_hash,
                       v.first_processing_run_id, v.created_at
                FROM voices AS v JOIN requested AS q ON q.id = v.voice_id
                WHERE v.library_id = ? ORDER BY v.voice_id
            """,
        )

    def _entity_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> dict[str, sqlite3.Row]:
        return self._rows_by_ids(
            connection,
            ids=output.entity_ids,
            query="""
                WITH requested(id) AS (SELECT value FROM json_each(?))
                SELECT e.entity_id, e.kind, e.label, e.status, e.content_hash,
                       e.first_processing_run_id, e.created_at
                FROM entities AS e JOIN requested AS q ON q.id = e.entity_id
                WHERE e.library_id = ? ORDER BY e.entity_id
            """,
        )

    def _entity_mention_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> dict[str, sqlite3.Row]:
        return self._rows_by_ids(
            connection,
            ids=output.entity_mention_ids,
            query="""
                WITH requested(id) AS (SELECT value FROM json_each(?))
                SELECT m.entity_mention_id, m.entity_id, m.collection_id,
                       m.source_fragment_id, m.quote_start, m.quote_end,
                       m.quote_sha256, m.content_hash
                FROM entity_mentions AS m
                JOIN requested AS q ON q.id = m.entity_mention_id
                JOIN entities AS e ON e.entity_id = m.entity_id
                WHERE e.library_id = ? ORDER BY m.entity_mention_id
            """,
        )

    def _concept_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> dict[str, sqlite3.Row]:
        return self._rows_by_ids(
            connection,
            ids=output.concept_ids,
            query="""
                WITH requested(id) AS (SELECT value FROM json_each(?))
                SELECT c.concept_id, c.label, c.status, c.content_hash,
                       c.first_processing_run_id, c.created_at
                FROM concepts AS c JOIN requested AS q ON q.id = c.concept_id
                WHERE c.library_id = ? ORDER BY c.concept_id
            """,
        )

    def _concept_mention_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> dict[str, sqlite3.Row]:
        return self._rows_by_ids(
            connection,
            ids=output.concept_mention_ids,
            query="""
                WITH requested(id) AS (SELECT value FROM json_each(?))
                SELECT m.concept_mention_id, m.concept_id, m.collection_id,
                       m.source_fragment_id, m.quote_start, m.quote_end,
                       m.quote_sha256, m.content_hash
                FROM concept_mentions AS m
                JOIN requested AS q ON q.id = m.concept_mention_id
                JOIN concepts AS c ON c.concept_id = m.concept_id
                WHERE c.library_id = ? ORDER BY m.concept_mention_id
            """,
        )

    def _time_context_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> dict[str, sqlite3.Row]:
        return self._rows_by_ids(
            connection,
            ids=output.time_context_ids,
            query="""
                WITH requested(id) AS (SELECT value FROM json_each(?))
                SELECT t.time_context_id, t.event_valid_start, t.event_valid_end,
                       t.recorded_at, t.available_at, t.revealed_at, t.reviewed_at,
                       t.status, t.content_hash, t.first_processing_run_id, t.created_at
                FROM time_contexts AS t JOIN requested AS q ON q.id = t.time_context_id
                WHERE t.library_id = ? ORDER BY t.time_context_id
            """,
        )

    def _statement_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> dict[str, sqlite3.Row]:
        return self._rows_by_ids(
            connection,
            ids=output.statement_ids,
            query="""
                WITH requested(id) AS (SELECT value FROM json_each(?))
                SELECT s.statement_id, s.collection_id, s.kind, s.statement_text,
                       s.voice_id, s.time_context_id, s.status, s.lifecycle,
                       s.content_hash, s.first_processing_run_id, s.created_at
                FROM statements AS s JOIN requested AS q ON q.id = s.statement_id
                WHERE s.library_id = ? ORDER BY s.statement_id
            """,
        )

    def _evidence_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> dict[str, sqlite3.Row]:
        return self._rows_by_ids(
            connection,
            ids=output.evidence_link_ids,
            query="""
                WITH requested(id) AS (SELECT value FROM json_each(?))
                SELECT e.evidence_link_id, e.statement_id, e.attributed_voice_id,
                       e.source_fragment_id, e.polarity, e.alignment,
                       e.quote_sha256, e.quote_start, e.quote_end,
                       e.extraction_method, e.status, e.content_hash,
                       e.first_processing_run_id, e.created_at
                FROM evidence_links AS e
                JOIN requested AS q ON q.id = e.evidence_link_id
                JOIN statements AS s ON s.statement_id = e.statement_id
                WHERE s.library_id = ? ORDER BY e.evidence_link_id
            """,
        )

    def _concept_meaning_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> dict[str, sqlite3.Row]:
        return self._rows_by_ids(
            connection,
            ids=output.concept_meaning_ids,
            query="""
                WITH requested(id) AS (SELECT value FROM json_each(?))
                SELECT m.concept_meaning_id, m.concept_id, m.voice_id,
                       m.time_context_id, m.meaning_text, m.status, m.content_hash,
                       m.first_processing_run_id, m.created_at
                FROM concept_meanings AS m
                JOIN requested AS q ON q.id = m.concept_meaning_id
                JOIN concepts AS c ON c.concept_id = m.concept_id
                WHERE c.library_id = ? ORDER BY m.concept_meaning_id
            """,
        )

    def _concept_meaning_support_rows(
        self, connection: sqlite3.Connection, output: MeaningOutputManifest
    ) -> tuple[sqlite3.Row, ...]:
        if not output.concept_meaning_ids:
            return ()
        rows = connection.execute(
            """
            WITH requested(id) AS (SELECT value FROM json_each(?))
            SELECT r.concept_meaning_id, r.statement_id
            FROM concept_meaning_statements AS r
            JOIN requested AS q ON q.id = r.concept_meaning_id
            JOIN concept_meanings AS m ON m.concept_meaning_id = r.concept_meaning_id
            JOIN concepts AS c ON c.concept_id = m.concept_id
            WHERE c.library_id = ?
            ORDER BY r.concept_meaning_id, r.statement_id
            """,
            (
                _json_array(output.concept_meaning_ids),
                self._repository.library_id,
            ),
        ).fetchall()
        return tuple(rows)

    def _rows_by_ids(
        self,
        connection: sqlite3.Connection,
        *,
        ids: tuple[str, ...],
        query: str,
    ) -> dict[str, sqlite3.Row]:
        if not ids:
            return {}
        rows = connection.execute(
            query,
            (_json_array(ids), self._repository.library_id),
        ).fetchall()
        result = {str(row[0]): row for row in rows}
        if len(result) != len(rows) or set(result) != set(ids):
            raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
        return result


def _json_array(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _hash(value: str) -> str:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
    return value


def _merged_node(left: ReadingRoomNode, right: ReadingRoomNode) -> ReadingRoomNode:
    comparable_left = (
        left.node_type,
        left.kind,
        left.label,
        left.candidate_state,
        left.content_hash,
        left.metadata,
    )
    comparable_right = (
        right.node_type,
        right.kind,
        right.label,
        right.candidate_state,
        right.content_hash,
        right.metadata,
    )
    if comparable_left != comparable_right:
        raise ReadingRoomIntegrityError("reading room metadata is inconsistent")
    provenance_candidates = tuple(
        value
        for value in (
            left.provenance_processing_run_id,
            right.provenance_processing_run_id,
        )
        if value is not None
    )
    created_candidates = tuple(
        value for value in (left.created_at, right.created_at) if value is not None
    )
    return replace(
        left,
        provenance_processing_run_id=(
            min(provenance_candidates) if provenance_candidates else None
        ),
        provenance_fragment_ids=tuple(
            sorted(set(left.provenance_fragment_ids) | set(right.provenance_fragment_ids))
        ),
        collection_ids=tuple(sorted(set(left.collection_ids) | set(right.collection_ids))),
        created_at=min(created_candidates) if created_candidates else None,
    )
