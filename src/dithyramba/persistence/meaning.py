"""Policy-first SQLite repository for the P6 typed meaning core."""

from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from dithyramba.contracts import (
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_hex,
    new_id,
)
from dithyramba.meaning.errors import (
    MeaningBudgetError,
    MeaningContractError,
    MeaningNotFoundError,
    MeaningPersistenceError,
    MeaningReviewError,
)
from dithyramba.meaning.models import (
    CoverageState,
    ExtractionProfile,
    GeneratorProfile,
    MeaningCoverageReport,
    MeaningGenerator,
    MeaningInputFragment,
    MeaningOmission,
    MeaningOutputManifest,
    MeaningProcessingRun,
    MeaningProposal,
    MeaningReadReceipt,
    MeaningReadReceiptItem,
    MeaningReviewAction,
    MeaningReviewDecision,
    MeaningReviewQueueItem,
    MeaningReviewRequest,
    MeaningReviewScope,
    MeaningReviewSession,
    MeaningReviewTargetType,
    MeaningRunKind,
    MeaningRunRequest,
    MeaningRunResult,
    MeaningRunStatus,
    OmissionCategory,
    RelationAdmission,
    ReviewBudget,
)
from dithyramba.meaning.validation import (
    PreparedMeaningGraph,
    meaning_input_hash,
    prepare_meaning_graph,
)
from dithyramba.persistence.models import AuthorizedRead
from dithyramba.relations import RelationNodeType

from .repository import LibraryRepository, _timestamp

_BACKLOG_WARNING = 200
_BACKLOG_HARD_LIMIT = 1_000
_TERMINAL_REVIEW_ACTIONS = frozenset(
    {
        MeaningReviewAction.ACCEPT,
        MeaningReviewAction.REJECT,
        MeaningReviewAction.REVISE,
    }
)


@dataclass(frozen=True, slots=True)
class _TargetInfo:
    target_type: MeaningReviewTargetType
    target_id: str
    target_hash: str
    collection_ids: tuple[str, ...]
    created_at: str


class SQLiteMeaningRepository:
    """Run, persist, review, and inspect one Library's P6 candidate layer."""

    def __init__(self, repository: LibraryRepository) -> None:
        if not isinstance(repository, LibraryRepository):
            raise TypeError("SQLiteMeaningRepository requires a LibraryRepository")
        self._repository = repository

    @property
    def library_id(self) -> str:
        return self._repository.library_id

    def import_proposal(
        self,
        *,
        request: MeaningRunRequest,
        proposal: MeaningProposal,
    ) -> MeaningRunResult:
        """Idempotently import a proposal after compiling policy and reading exact evidence."""

        self._validate_request(request)
        if not isinstance(proposal, MeaningProposal):
            raise TypeError("import_proposal requires a MeaningProposal")
        authorization = self._authorize(request)
        import_key = self._import_key(
            kind=MeaningRunKind.IMPORT,
            request=request,
            permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
            proposal_hash=proposal.proposal_hash,
            terminal_marker=None,
        )
        existing = self._find_run_by_import_key(import_key)
        if existing is not None:
            return self._load_result(existing, deduplicated=True)
        preflight = self._preflight_refusal(
            kind=MeaningRunKind.IMPORT,
            request=request,
            authorization=authorization,
            proposal_hash=proposal.proposal_hash,
            import_key=import_key,
        )
        if preflight is not None:
            return preflight
        fragments = self._read_inputs(authorization)
        return self._complete_proposal(
            kind=MeaningRunKind.IMPORT,
            request=request,
            authorization=authorization,
            fragments=fragments,
            proposal=proposal,
            import_key=import_key,
        )

    def extract(
        self,
        *,
        request: MeaningRunRequest,
        generator: MeaningGenerator,
    ) -> MeaningRunResult:
        """Call a generator only with policy-permitted fragments, then import its proposal."""

        self._validate_request(request)
        if not callable(generator):
            raise TypeError("extract requires a callable MeaningGenerator")
        authorization = self._authorize(request)
        preflight_key = self._import_key(
            kind=MeaningRunKind.EXTRACT,
            request=request,
            permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
            proposal_hash=None,
            terminal_marker="preflight",
        )
        preflight = self._preflight_refusal(
            kind=MeaningRunKind.EXTRACT,
            request=request,
            authorization=authorization,
            proposal_hash=None,
            import_key=preflight_key,
        )
        if preflight is not None:
            return preflight
        fragments = self._read_inputs(authorization)
        try:
            generated: object = generator(fragments, request)
        except Exception:
            failure_key = self._import_key(
                kind=MeaningRunKind.EXTRACT,
                request=request,
                permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
                proposal_hash=None,
                terminal_marker="generator_failed",
            )
            existing = self._find_run_by_import_key(failure_key)
            if existing is not None:
                return self._load_result(existing, deduplicated=True)
            return self._persist_result(
                kind=MeaningRunKind.EXTRACT,
                request=request,
                authorization=authorization,
                fragments=fragments,
                proposal_hash=None,
                import_key=failure_key,
                status=MeaningRunStatus.FAILED,
                coverage_state=CoverageState.FAILED,
                omissions=(MeaningOmission(OmissionCategory.FAILURE, "generator_failed", 1),),
                error_code="generator_failed",
                graph=None,
                unresolved_before=self.unresolved_candidate_count(),
            )
        if not isinstance(generated, MeaningProposal):
            invalid_key = self._import_key(
                kind=MeaningRunKind.EXTRACT,
                request=request,
                permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
                proposal_hash=None,
                terminal_marker="invalid_generator_output",
            )
            return self._persist_result(
                kind=MeaningRunKind.EXTRACT,
                request=request,
                authorization=authorization,
                fragments=fragments,
                proposal_hash=None,
                import_key=invalid_key,
                status=MeaningRunStatus.FAILED,
                coverage_state=CoverageState.FAILED,
                omissions=(
                    MeaningOmission(
                        OmissionCategory.FAILURE,
                        "invalid_generator_output",
                        1,
                    ),
                ),
                error_code="invalid_generator_output",
                graph=None,
                unresolved_before=self.unresolved_candidate_count(),
            )
        proposal = generated
        import_key = self._import_key(
            kind=MeaningRunKind.EXTRACT,
            request=request,
            permitted_set_hash=authorization.compiled.manifest.permitted_set_hash,
            proposal_hash=proposal.proposal_hash,
            terminal_marker=None,
        )
        existing = self._find_run_by_import_key(import_key)
        if existing is not None:
            return self._load_result(existing, deduplicated=True)
        return self._complete_proposal(
            kind=MeaningRunKind.EXTRACT,
            request=request,
            authorization=authorization,
            fragments=fragments,
            proposal=proposal,
            import_key=import_key,
        )

    def get_run(self, processing_run_id: str) -> MeaningRunResult:
        """Load one complete semantic ProcessingRun artifact closure."""

        return self._load_result(processing_run_id, deduplicated=False)

    def unresolved_candidate_count(self) -> int:
        """Return unresolved typed review candidates; exact mentions are not review targets."""

        return self._unresolved_candidate_count(self._repository._store.connection)

    def _unresolved_candidate_count(self, connection: sqlite3.Connection) -> int:
        resolved = self._resolved_target_keys(connection=connection)
        return sum(
            (target_type, target_id) not in resolved
            for target_type, target_id, _target_hash, _created_at in self._candidate_stubs(
                connection=connection
            )
        )

    def list_review_queue(self, *, limit: int = 100) -> tuple[MeaningReviewQueueItem, ...]:
        """Return a stable read-only semantic review queue."""

        if type(limit) is not int or not 1 <= limit <= 500:
            raise MeaningReviewError("review queue limit must be between 1 and 500")
        connection = self._repository._store.connection
        resolved = self._resolved_target_keys(connection=connection)
        items: list[MeaningReviewQueueItem] = []
        for target_type, target_id, _target_hash, _created_at in self._candidate_stubs(
            connection=connection
        ):
            if (target_type, target_id) in resolved:
                continue
            target = self._target_info(target_type, target_id)
            items.append(
                MeaningReviewQueueItem(
                    target.target_type,
                    target.target_id,
                    target.target_hash,
                    target.collection_ids,
                    target.created_at,
                )
            )
            if len(items) == limit:
                break
        return tuple(items)

    def start_review_session(
        self, review_budget: ReviewBudget | None = None
    ) -> MeaningReviewSession:
        """Create an append-only human review session with an explicit decision cap."""

        budget = review_budget or ReviewBudget()
        if not isinstance(budget, ReviewBudget):
            raise MeaningReviewError("review_budget must be a ReviewBudget")
        review_session_id = new_id("review_session")
        created_at = _timestamp(self._repository._clock)
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO meaning_review_sessions(
                        review_session_id, library_id, review_budget_json,
                        review_budget_hash, decision_limit, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_session_id,
                        self.library_id,
                        _canonical_text(budget.payload()),
                        budget.budget_hash,
                        budget.review_decision_limit,
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise MeaningPersistenceError("semantic review session insert conflicted") from exc
        return MeaningReviewSession(review_session_id, self.library_id, budget, created_at)

    def get_review_session(self, review_session_id: str) -> MeaningReviewSession:
        row = self._repository._store.connection.execute(
            """
            SELECT review_session_id, library_id, review_budget_json,
                   review_budget_hash, decision_limit, created_at
            FROM meaning_review_sessions
            WHERE review_session_id = ? AND library_id = ?
            """,
            (review_session_id, self.library_id),
        ).fetchone()
        if row is None:
            raise MeaningNotFoundError("semantic review session does not exist")
        budget_payload = _load_object(str(row[2]), "ReviewBudget")
        budget = _review_budget_from_payload(budget_payload)
        if budget.budget_hash != str(row[3]) or budget.review_decision_limit != int(row[4]):
            raise MeaningPersistenceError("persisted semantic ReviewBudget is inconsistent")
        return MeaningReviewSession(str(row[0]), str(row[1]), budget, str(row[5]))

    def create_review_decision(
        self,
        request: MeaningReviewRequest,
    ) -> MeaningReviewDecision:
        """Append one scoped decision without mutating a candidate or Source."""

        if not isinstance(request, MeaningReviewRequest):
            raise TypeError("create_review_decision requires a MeaningReviewRequest")
        session = self.get_review_session(request.review_session_id)
        target = self._target_info(request.target_type, request.target_id)
        if request.target_hash != target.target_hash:
            raise MeaningReviewError("semantic review target hash is stale")
        if (
            request.target_type is MeaningReviewTargetType.RELATION
            and request.scope.collection_ids != target.collection_ids
        ):
            raise MeaningReviewError("Relation ReviewScope must equal its exact Collection closure")
        if not set(request.scope.collection_ids).issubset(target.collection_ids):
            raise MeaningReviewError("semantic ReviewScope exceeds target Collections")
        if request.replacement_target_id is not None:
            replacement = self._target_info(request.target_type, request.replacement_target_id)
            if replacement.target_id == target.target_id:
                raise MeaningReviewError("revise requires a distinct replacement candidate")
            if (
                request.target_type is MeaningReviewTargetType.RELATION
                and request.scope.collection_ids != replacement.collection_ids
            ):
                raise MeaningReviewError(
                    "replacement Relation must have the same exact Collection closure"
                )
            if not set(request.scope.collection_ids).issubset(replacement.collection_ids):
                raise MeaningReviewError("replacement candidate is outside ReviewScope")

        relation_target = request.target_type is MeaningReviewTargetType.RELATION
        decision_table = (
            "relation_review_decisions" if relation_target else "meaning_review_decisions"
        )
        review_decision_id = new_id("review_relation" if relation_target else "review")
        created_at = _timestamp(self._repository._clock)
        scope_json = _canonical_text(request.scope.payload())
        try:
            with self._repository._store.transaction(immediate=True) as connection:
                count = int(
                    connection.execute(
                        """
                        SELECT
                            (SELECT count(*) FROM meaning_review_decisions
                             WHERE review_session_id = ?)
                          + (SELECT count(*) FROM relation_review_decisions
                             WHERE review_session_id = ?)
                        """,
                        (session.review_session_id, session.review_session_id),
                    ).fetchone()[0]
                )
                if count >= session.review_budget.review_decision_limit:
                    raise MeaningBudgetError("semantic review session decision budget is exhausted")
                if request.action is not MeaningReviewAction.SUPERSEDE:
                    if relation_target:
                        active_decisions = self._load_active_relation_decisions(
                            target_id=request.target_id,
                            target_hash=request.target_hash,
                            connection=connection,
                        )
                        if any(decision.scope == request.scope for decision in active_decisions):
                            raise MeaningReviewError(
                                "exact ReviewScope already has an active decision; "
                                "supersede it explicitly"
                            )
                    elif (
                        connection.execute(
                            f"""
                            SELECT 1
                            FROM {decision_table} AS active
                            WHERE active.library_id = ?
                              AND active.target_type = ?
                              AND active.target_id = ?
                              AND active.target_hash = ?
                              AND active.scope_json = ?
                              AND active.action <> 'supersede'
                              AND NOT EXISTS (
                                  SELECT 1 FROM {decision_table} AS later
                                  WHERE later.supersedes_review_decision_id =
                                        active.review_decision_id
                              )
                            LIMIT 1
                            """,
                            (
                                self.library_id,
                                request.target_type.value,
                                request.target_id,
                                request.target_hash,
                                scope_json,
                            ),
                        ).fetchone()
                        is not None
                    ):
                        raise MeaningReviewError(
                            "exact ReviewScope already has an active decision; "
                            "supersede it explicitly"
                        )
                if request.supersedes_review_decision_id is not None:
                    prior = self._load_decision_row(
                        request.supersedes_review_decision_id,
                        connection=connection,
                    )
                    if (
                        prior.target_type is not request.target_type
                        or prior.target_id != request.target_id
                        or prior.target_hash != request.target_hash
                        or prior.scope != request.scope
                    ):
                        raise MeaningReviewError(
                            "supersede must retain target, hash, and ReviewScope"
                        )
                    if (
                        connection.execute(
                            f"""
                        SELECT 1 FROM {decision_table}
                        WHERE supersedes_review_decision_id = ? LIMIT 1
                        """,
                            (prior.review_decision_id,),
                        ).fetchone()
                        is not None
                    ):
                        raise MeaningReviewError("semantic ReviewDecision is already superseded")
                connection.execute(
                    f"""
                    INSERT INTO {decision_table}(
                        review_decision_id, review_session_id, library_id,
                        target_type, target_id, target_hash, action, reason,
                        authority, scope_json, replacement_target_id,
                        supersedes_review_decision_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_decision_id,
                        request.review_session_id,
                        self.library_id,
                        request.target_type.value,
                        request.target_id,
                        request.target_hash,
                        request.action.value,
                        request.reason,
                        request.authority,
                        scope_json,
                        request.replacement_target_id,
                        request.supersedes_review_decision_id,
                        created_at,
                    ),
                )
        except MeaningReviewError:
            raise
        except sqlite3.IntegrityError as exc:
            raise MeaningPersistenceError("semantic ReviewDecision insert conflicted") from exc
        return self.get_review_decision(review_decision_id)

    def get_review_decision(self, review_decision_id: str) -> MeaningReviewDecision:
        return self._load_decision_row(
            review_decision_id,
            connection=self._repository._store.connection,
        )

    def resolve_relation_admission(
        self,
        relation_id: str,
        *,
        use: str,
        request_collection_ids: tuple[str, ...],
    ) -> RelationAdmission | None:
        """Return the sole active exact-scope acceptance for one Relation."""

        target = self._target_info(MeaningReviewTargetType.RELATION, relation_id)
        request_scope = MeaningReviewScope(request_collection_ids, use)
        if not set(target.collection_ids).issubset(request_scope.collection_ids):
            return None
        target_scope = MeaningReviewScope(target.collection_ids, request_scope.use)
        connection = self._repository._store.connection
        active_decisions = self._load_active_relation_decisions(
            target_id=relation_id,
            target_hash=target.target_hash,
            connection=connection,
        )
        matching_decisions = tuple(
            decision for decision in active_decisions if decision.scope == target_scope
        )
        if len(matching_decisions) > 1:
            raise MeaningPersistenceError(
                "Relation admission has multiple active exact-scope decisions"
            )
        if not matching_decisions:
            return None
        decision = matching_decisions[0]
        if decision.action is not MeaningReviewAction.ACCEPT:
            return None
        return RelationAdmission(
            relation_id=target.target_id,
            target_hash=target.target_hash,
            review_decision_id=decision.review_decision_id,
            scope=decision.scope,
        )

    def _validate_request(self, request: MeaningRunRequest) -> None:
        if not isinstance(request, MeaningRunRequest):
            raise TypeError("semantic operation requires a MeaningRunRequest")
        if request.scope.library_id != self.library_id:
            raise MeaningContractError("MeaningRunRequest belongs to another Library")
        snapshot = self._repository.get_corpus_snapshot(request.corpus_snapshot_id)
        if snapshot.manifest_hash != request.scope.snapshot_hash:
            raise MeaningContractError("MeaningRunRequest snapshot ID/hash are inconsistent")

    def _authorize(self, request: MeaningRunRequest) -> AuthorizedRead:
        return self._repository.authorize_read(
            access_policy_id=request.access_policy_id,
            scope=request.scope,
        )

    def _preflight_refusal(
        self,
        *,
        kind: MeaningRunKind,
        request: MeaningRunRequest,
        authorization: AuthorizedRead,
        proposal_hash: str | None,
        import_key: str,
    ) -> MeaningRunResult | None:
        unresolved = self.unresolved_candidate_count()
        if request.generator.external_provider:
            policy = self._repository.get_access_policy(request.access_policy_id)
            if not policy.snapshot.allow_external_provider:
                return self._persist_result(
                    kind=kind,
                    request=request,
                    authorization=authorization,
                    fragments=(),
                    proposal_hash=proposal_hash,
                    import_key=import_key,
                    status=MeaningRunStatus.REFUSED,
                    coverage_state=CoverageState.REFUSED,
                    omissions=(
                        MeaningOmission(
                            OmissionCategory.FAILURE,
                            "provider_egress_denied",
                            1,
                        ),
                    ),
                    error_code="provider_egress_denied",
                    graph=None,
                    unresolved_before=unresolved,
                )
        if unresolved >= request.review_budget.max_unresolved_candidates or (
            unresolved >= _BACKLOG_HARD_LIMIT
            and not request.review_budget.has_explicit_backlog_override
        ):
            return self._persist_result(
                kind=kind,
                request=request,
                authorization=authorization,
                fragments=(),
                proposal_hash=proposal_hash,
                import_key=import_key,
                status=MeaningRunStatus.REFUSED,
                coverage_state=CoverageState.REFUSED,
                omissions=(MeaningOmission(OmissionCategory.BUDGET, "review_backlog_limit", 1),),
                error_code="review_backlog_limit",
                graph=None,
                unresolved_before=unresolved,
            )
        permitted_count = len(authorization.compiled.manifest.items)
        if permitted_count > request.review_budget.max_read_fragments:
            return self._persist_result(
                kind=kind,
                request=request,
                authorization=authorization,
                fragments=(),
                proposal_hash=proposal_hash,
                import_key=import_key,
                status=MeaningRunStatus.REFUSED,
                coverage_state=CoverageState.REFUSED,
                omissions=(MeaningOmission(OmissionCategory.BUDGET, "read_fragment_limit", 1),),
                error_code="read_fragment_limit",
                graph=None,
                unresolved_before=unresolved,
            )
        return None

    def _read_inputs(self, authorization: AuthorizedRead) -> tuple[MeaningInputFragment, ...]:
        text_fragments = self._repository.read_permitted_fragments(authorization)
        by_id = {item.source_fragment_id: item for item in text_fragments}
        inputs: list[MeaningInputFragment] = []
        for manifest_item in authorization.compiled.manifest.items:
            try:
                fragment = by_id[manifest_item.source_fragment_id]
            except KeyError as exc:
                raise MeaningPersistenceError(
                    "permitted semantic manifest did not resolve exactly"
                ) from exc
            inputs.append(
                MeaningInputFragment(
                    source_fragment_id=fragment.source_fragment_id,
                    source_version_id=fragment.source_version_id,
                    source_id=fragment.source_id,
                    collection_ids=manifest_item.collection_ids,
                    text=fragment.text,
                    text_sha256=fragment.text_sha256,
                )
            )
        if len(by_id) != len(inputs):
            raise MeaningPersistenceError("semantic read returned fragments outside manifest")
        return tuple(inputs)

    def _complete_proposal(
        self,
        *,
        kind: MeaningRunKind,
        request: MeaningRunRequest,
        authorization: AuthorizedRead,
        fragments: tuple[MeaningInputFragment, ...],
        proposal: MeaningProposal,
        import_key: str,
    ) -> MeaningRunResult:
        unresolved = self.unresolved_candidate_count()
        if proposal.state is not CoverageState.COMPLETE:
            status = {
                CoverageState.PARTIAL: MeaningRunStatus.PARTIAL,
                CoverageState.FAILED: MeaningRunStatus.FAILED,
                CoverageState.REFUSED: MeaningRunStatus.REFUSED,
            }[proposal.state]
            return self._persist_result(
                kind=kind,
                request=request,
                authorization=authorization,
                fragments=fragments,
                proposal_hash=proposal.proposal_hash,
                import_key=import_key,
                status=status,
                coverage_state=proposal.state,
                omissions=proposal.omissions,
                error_code=proposal.failure_code,
                graph=None,
                unresolved_before=unresolved,
            )
        try:
            graph = prepare_meaning_graph(
                library_id=self.library_id,
                profile=request.profile,
                review_budget=request.review_budget,
                proposal=proposal,
                fragments=fragments,
            )
        except MeaningBudgetError:
            return self._persist_result(
                kind=kind,
                request=request,
                authorization=authorization,
                fragments=fragments,
                proposal_hash=proposal.proposal_hash,
                import_key=import_key,
                status=MeaningRunStatus.REFUSED,
                coverage_state=CoverageState.REFUSED,
                omissions=(MeaningOmission(OmissionCategory.BUDGET, "candidate_budget_limit", 1),),
                error_code="candidate_budget_limit",
                graph=None,
                unresolved_before=unresolved,
            )
        except MeaningContractError:
            return self._persist_result(
                kind=kind,
                request=request,
                authorization=authorization,
                fragments=fragments,
                proposal_hash=proposal.proposal_hash,
                import_key=import_key,
                status=MeaningRunStatus.FAILED,
                coverage_state=CoverageState.FAILED,
                omissions=(MeaningOmission(OmissionCategory.FAILURE, "invalid_proposal", 1),),
                error_code="invalid_proposal",
                graph=None,
                unresolved_before=unresolved,
            )
        return self._persist_result(
            kind=kind,
            request=request,
            authorization=authorization,
            fragments=fragments,
            proposal_hash=proposal.proposal_hash,
            import_key=import_key,
            status=MeaningRunStatus.SUCCEEDED,
            coverage_state=CoverageState.COMPLETE,
            omissions=proposal.omissions,
            error_code=None,
            graph=graph,
            unresolved_before=unresolved,
        )

    def _persist_result(
        self,
        *,
        kind: MeaningRunKind,
        request: MeaningRunRequest,
        authorization: AuthorizedRead,
        fragments: tuple[MeaningInputFragment, ...],
        proposal_hash: str | None,
        import_key: str,
        status: MeaningRunStatus,
        coverage_state: CoverageState,
        omissions: tuple[MeaningOmission, ...],
        error_code: str | None,
        graph: PreparedMeaningGraph | None,
        unresolved_before: int,
    ) -> MeaningRunResult:
        existing = self._find_run_by_import_key(import_key)
        if existing is not None:
            return self._load_result(existing, deduplicated=True)
        input_hash = meaning_input_hash(fragments)
        processing_run_id = canonical_content_id(
            "run",
            {"schema": "dithyramba.meaning_run_identity/2.0", "import_key": import_key},
        )
        started_at = _timestamp(self._repository._clock)
        finished_at = _timestamp(self._repository._clock)
        try:
            text_guard = (
                self._repository._permit_fragment_text_read()
                if graph is not None
                else nullcontext()
            )
            with text_guard, self._repository._store.transaction(immediate=True) as connection:
                effective_graph = graph
                effective_status = status
                effective_coverage_state = coverage_state
                effective_omissions = omissions
                effective_error_code = error_code
                effective_unresolved_before = unresolved_before
                if effective_graph is not None:
                    effective_unresolved_before = self._unresolved_candidate_count(connection)
                    new_review_candidates = self._new_review_candidate_count(
                        effective_graph.output,
                        connection=connection,
                    )
                    if (
                        effective_unresolved_before + new_review_candidates
                        > request.review_budget.max_unresolved_candidates
                    ):
                        effective_graph = None
                        effective_status = MeaningRunStatus.REFUSED
                        effective_coverage_state = CoverageState.REFUSED
                        effective_omissions = (
                            MeaningOmission(
                                OmissionCategory.BUDGET,
                                "review_backlog_limit",
                                1,
                            ),
                        )
                        effective_error_code = "review_backlog_limit"
                output = (
                    MeaningOutputManifest() if effective_graph is None else effective_graph.output
                )
                backlog_warning = effective_unresolved_before >= _BACKLOG_WARNING or (
                    effective_graph is not None
                    and effective_unresolved_before
                    + self._new_review_candidate_count(
                        effective_graph.output,
                        connection=connection,
                    )
                    >= _BACKLOG_WARNING
                )
                coverage_payload: dict[str, object] = {
                    "schema": "dithyramba.meaning_coverage_report/2.0",
                    "processing_run_id": processing_run_id,
                    "profile": request.profile.value,
                    "state": effective_coverage_state.value,
                    "read_fragment_count": len(fragments),
                    "candidate_count": output.candidate_count,
                    "omissions": [item.payload() for item in effective_omissions],
                }
                coverage_hash = canonical_sha256_hex(coverage_payload)
                coverage_id = canonical_content_id("coverage", coverage_payload)
                read_items_payload = [
                    {
                        "source_fragment_id": fragment.source_fragment_id,
                        "source_version_id": fragment.source_version_id,
                        "source_id": fragment.source_id,
                        "collection_ids": list(fragment.collection_ids),
                        "read_order": index,
                        "text_sha256": fragment.text_sha256,
                    }
                    for index, fragment in enumerate(fragments)
                ]
                read_payload: dict[str, object] = {
                    "schema": "dithyramba.meaning_read_receipt/2.0",
                    "processing_run_id": processing_run_id,
                    "input_hash": input_hash,
                    "items": read_items_payload,
                }
                read_hash = canonical_sha256_hex(read_payload)
                read_id = canonical_content_id("read", read_payload)
                run_payload = _run_receipt_payload(
                    processing_run_id=processing_run_id,
                    kind=kind.value,
                    corpus_snapshot_id=request.corpus_snapshot_id,
                    access_policy_id=request.access_policy_id,
                    scope_hash=request.scope.exclusion_hash,
                    permitted_set_hash=(authorization.compiled.manifest.permitted_set_hash),
                    extraction_profile=request.profile.value,
                    code_version=request.code_version,
                    generator_hash=request.generator.generator_hash,
                    review_budget_hash=request.review_budget.budget_hash,
                    input_hash=input_hash,
                    proposal_hash=proposal_hash,
                    import_key=import_key,
                    status=effective_status.value,
                    candidate_count=output.candidate_count,
                    unresolved_before=effective_unresolved_before,
                    backlog_warning=backlog_warning,
                    error_code=effective_error_code,
                    output_hash=output.output_hash,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                run_receipt_hash = canonical_sha256_hex(run_payload)
                connection.execute(
                    """
                    INSERT INTO meaning_processing_runs(
                        processing_run_id, library_id, kind, corpus_snapshot_id,
                        access_policy_id, scope_hash, permitted_set_hash,
                        extraction_profile, code_version, generator_json, generator_hash,
                        review_budget_json, review_budget_hash, input_hash,
                        proposal_hash, import_key, status, candidate_count,
                        unresolved_before, backlog_warning, error_code,
                        output_manifest_json, output_hash, receipt_hash,
                        started_at, finished_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        processing_run_id,
                        self.library_id,
                        kind.value,
                        request.corpus_snapshot_id,
                        request.access_policy_id,
                        request.scope.exclusion_hash,
                        authorization.compiled.manifest.permitted_set_hash,
                        request.profile.value,
                        request.code_version,
                        _canonical_text(request.generator.payload()),
                        request.generator.generator_hash,
                        _canonical_text(request.review_budget.payload()),
                        request.review_budget.budget_hash,
                        input_hash,
                        proposal_hash,
                        import_key,
                        effective_status.value,
                        output.candidate_count,
                        effective_unresolved_before,
                        int(backlog_warning),
                        effective_error_code,
                        _canonical_text(output.payload()),
                        output.output_hash,
                        run_receipt_hash,
                        started_at,
                        finished_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO meaning_coverage_reports(
                        coverage_report_id, processing_run_id, extraction_profile,
                        state, read_fragment_count, candidate_count,
                        omissions_json, report_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        coverage_id,
                        processing_run_id,
                        request.profile.value,
                        effective_coverage_state.value,
                        len(fragments),
                        output.candidate_count,
                        _canonical_text([item.payload() for item in effective_omissions]),
                        coverage_hash,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO meaning_read_receipts(
                        read_receipt_id, processing_run_id, input_hash, receipt_hash
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (read_id, processing_run_id, input_hash, read_hash),
                )
                connection.executemany(
                    """
                    INSERT INTO meaning_read_receipt_items(
                        read_receipt_id, source_fragment_id, source_version_id,
                        source_id, collection_ids_json, read_order, text_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            read_id,
                            fragment.source_fragment_id,
                            fragment.source_version_id,
                            fragment.source_id,
                            _canonical_text(list(fragment.collection_ids)),
                            index,
                            fragment.text_sha256,
                        )
                        for index, fragment in enumerate(fragments)
                    ),
                )
                if effective_graph is not None:
                    self._insert_graph(
                        connection,
                        processing_run_id=processing_run_id,
                        created_at=finished_at,
                        graph=effective_graph,
                    )
                    self._verify_statement_evidence_closure(
                        connection,
                        effective_graph.output,
                    )
        except sqlite3.IntegrityError as exc:
            raced = self._find_run_by_import_key(import_key)
            if raced is not None:
                return self._load_result(raced, deduplicated=True)
            raise MeaningPersistenceError("semantic run transaction conflicted") from exc
        return self._load_result(processing_run_id, deduplicated=False)

    def _insert_graph(
        self,
        connection: sqlite3.Connection,
        *,
        processing_run_id: str,
        created_at: str,
        graph: PreparedMeaningGraph,
    ) -> None:
        for voice in graph.voices:
            connection.execute(
                """
                INSERT OR IGNORE INTO voices(
                    voice_id, library_id, kind, label, status, content_hash,
                    first_processing_run_id, created_at
                ) VALUES (?, ?, ?, ?, 'candidate', ?, ?, ?)
                """,
                (
                    voice.voice_id,
                    self.library_id,
                    voice.kind.value,
                    voice.label,
                    voice.content_hash,
                    processing_run_id,
                    created_at,
                ),
            )
            _require_content_row(
                connection,
                "voices",
                "voice_id",
                voice.voice_id,
                voice.content_hash,
            )
        for entity in graph.entities:
            connection.execute(
                """
                INSERT OR IGNORE INTO entities(
                    entity_id, library_id, kind, label, status, content_hash,
                    first_processing_run_id, created_at
                ) VALUES (?, ?, ?, ?, 'candidate', ?, ?, ?)
                """,
                (
                    entity.entity_id,
                    self.library_id,
                    entity.kind.value,
                    entity.label,
                    entity.content_hash,
                    processing_run_id,
                    created_at,
                ),
            )
            _require_content_row(
                connection,
                "entities",
                "entity_id",
                entity.entity_id,
                entity.content_hash,
            )
            connection.executemany(
                "INSERT OR IGNORE INTO entity_aliases(entity_id, alias) VALUES (?, ?)",
                ((entity.entity_id, alias) for alias in entity.aliases),
            )
        for concept in graph.concepts:
            connection.execute(
                """
                INSERT OR IGNORE INTO concepts(
                    concept_id, library_id, label, status, content_hash,
                    first_processing_run_id, created_at
                ) VALUES (?, ?, ?, 'candidate', ?, ?, ?)
                """,
                (
                    concept.concept_id,
                    self.library_id,
                    concept.label,
                    concept.content_hash,
                    processing_run_id,
                    created_at,
                ),
            )
            _require_content_row(
                connection,
                "concepts",
                "concept_id",
                concept.concept_id,
                concept.content_hash,
            )
            connection.executemany(
                "INSERT OR IGNORE INTO concept_aliases(concept_id, alias) VALUES (?, ?)",
                ((concept.concept_id, alias) for alias in concept.aliases),
            )
        for time_context in graph.time_contexts:
            connection.execute(
                """
                INSERT OR IGNORE INTO time_contexts(
                    time_context_id, library_id, event_valid_start, event_valid_end,
                    recorded_at, available_at, revealed_at, reviewed_at, status,
                    content_hash, first_processing_run_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?)
                """,
                (
                    time_context.time_context_id,
                    self.library_id,
                    time_context.event_valid_start,
                    time_context.event_valid_end,
                    time_context.recorded_at,
                    time_context.available_at,
                    time_context.revealed_at,
                    time_context.reviewed_at,
                    time_context.content_hash,
                    processing_run_id,
                    created_at,
                ),
            )
            _require_content_row(
                connection,
                "time_contexts",
                "time_context_id",
                time_context.time_context_id,
                time_context.content_hash,
            )
        for statement in graph.statements:
            connection.execute(
                """
                INSERT OR IGNORE INTO statements(
                    statement_id, library_id, collection_id, kind, statement_text,
                    voice_id, time_context_id, status, lifecycle, content_hash,
                    first_processing_run_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'candidate', 'active', ?, ?, ?)
                """,
                (
                    statement.statement_id,
                    self.library_id,
                    statement.collection_id,
                    statement.kind.value,
                    statement.statement_text,
                    statement.voice_id,
                    statement.time_context_id,
                    statement.content_hash,
                    processing_run_id,
                    created_at,
                ),
            )
            _require_content_row(
                connection,
                "statements",
                "statement_id",
                statement.statement_id,
                statement.content_hash,
            )
        for evidence in graph.evidence_links:
            connection.execute(
                """
                INSERT OR IGNORE INTO evidence_links(
                    evidence_link_id, statement_id, source_fragment_id,
                    attributed_voice_id, polarity, alignment, limits_text,
                    quote_start, quote_end, quote_text, quote_sha256,
                    extraction_method, status, content_hash,
                    first_processing_run_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?)
                """,
                (
                    evidence.evidence_link_id,
                    evidence.statement_id,
                    evidence.source_fragment_id,
                    evidence.attributed_voice_id,
                    evidence.polarity.value,
                    evidence.alignment.value,
                    evidence.limits_text,
                    evidence.quote_start,
                    evidence.quote_end,
                    evidence.quote_text,
                    evidence.quote_sha256,
                    evidence.extraction_method,
                    evidence.content_hash,
                    processing_run_id,
                    created_at,
                ),
            )
            _require_content_row(
                connection,
                "evidence_links",
                "evidence_link_id",
                evidence.evidence_link_id,
                evidence.content_hash,
            )
        for meaning in graph.concept_meanings:
            connection.execute(
                """
                INSERT OR IGNORE INTO concept_meanings(
                    concept_meaning_id, concept_id, voice_id, time_context_id,
                    meaning_text, status, content_hash,
                    first_processing_run_id, created_at
                ) VALUES (?, ?, ?, ?, ?, 'candidate', ?, ?, ?)
                """,
                (
                    meaning.concept_meaning_id,
                    meaning.concept_id,
                    meaning.voice_id,
                    meaning.time_context_id,
                    meaning.meaning_text,
                    meaning.content_hash,
                    processing_run_id,
                    created_at,
                ),
            )
            _require_content_row(
                connection,
                "concept_meanings",
                "concept_meaning_id",
                meaning.concept_meaning_id,
                meaning.content_hash,
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO concept_meaning_statements(
                    concept_meaning_id, statement_id
                ) VALUES (?, ?)
                """,
                (
                    (meaning.concept_meaning_id, statement_id)
                    for statement_id in meaning.statement_ids
                ),
            )
        for entity_mention in graph.entity_mentions:
            connection.execute(
                """
                INSERT OR IGNORE INTO entity_mentions(
                    entity_mention_id, entity_id, collection_id,
                    source_fragment_id, quote_start, quote_end, quote_text,
                    quote_sha256, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_mention.mention_id,
                    entity_mention.target_id,
                    entity_mention.collection_id,
                    entity_mention.source_fragment_id,
                    entity_mention.quote_start,
                    entity_mention.quote_end,
                    entity_mention.quote_text,
                    entity_mention.quote_sha256,
                    entity_mention.content_hash,
                ),
            )
            _require_content_row(
                connection,
                "entity_mentions",
                "entity_mention_id",
                entity_mention.mention_id,
                entity_mention.content_hash,
            )
        for concept_mention in graph.concept_mentions:
            connection.execute(
                """
                INSERT OR IGNORE INTO concept_mentions(
                    concept_mention_id, concept_id, collection_id,
                    source_fragment_id, quote_start, quote_end, quote_text,
                    quote_sha256, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    concept_mention.mention_id,
                    concept_mention.target_id,
                    concept_mention.collection_id,
                    concept_mention.source_fragment_id,
                    concept_mention.quote_start,
                    concept_mention.quote_end,
                    concept_mention.quote_text,
                    concept_mention.quote_sha256,
                    concept_mention.content_hash,
                ),
            )
            _require_content_row(
                connection,
                "concept_mentions",
                "concept_mention_id",
                concept_mention.mention_id,
                concept_mention.content_hash,
            )

    def _verify_statement_evidence_closure(
        self,
        connection: sqlite3.Connection,
        output: MeaningOutputManifest,
    ) -> None:
        if not output.statement_ids:
            return
        placeholders = ", ".join("?" for _ in output.statement_ids)
        row = connection.execute(
            f"""
            SELECT s.statement_id
            FROM statements AS s
            LEFT JOIN evidence_links AS e ON e.statement_id = s.statement_id
            WHERE s.statement_id IN ({placeholders})
            GROUP BY s.statement_id
            HAVING count(e.evidence_link_id) = 0
            LIMIT 1
            """,
            output.statement_ids,
        ).fetchone()
        if row is not None:
            raise MeaningPersistenceError("persisted Statement lacks an EvidenceLink")

    def _new_review_candidate_count(
        self,
        output: MeaningOutputManifest,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> int:
        selected_connection = connection or self._repository._store.connection
        typed_ids: tuple[tuple[str, str, tuple[str, ...]], ...] = (
            ("voices", "voice_id", output.voice_ids),
            ("entities", "entity_id", output.entity_ids),
            ("concepts", "concept_id", output.concept_ids),
            ("concept_meanings", "concept_meaning_id", output.concept_meaning_ids),
            ("time_contexts", "time_context_id", output.time_context_ids),
            ("statements", "statement_id", output.statement_ids),
            ("evidence_links", "evidence_link_id", output.evidence_link_ids),
        )
        count = 0
        for table, column, identifiers in typed_ids:
            if not identifiers:
                continue
            placeholders = ", ".join("?" for _ in identifiers)
            existing = int(
                selected_connection.execute(
                    f'SELECT count(*) FROM "{table}" WHERE "{column}" IN ({placeholders})',
                    identifiers,
                ).fetchone()[0]
            )
            count += len(identifiers) - existing
        return count

    def _find_run_by_import_key(self, import_key: str) -> str | None:
        row = self._repository._store.connection.execute(
            """
            SELECT processing_run_id FROM meaning_processing_runs
            WHERE library_id = ? AND import_key = ?
            """,
            (self.library_id, import_key),
        ).fetchone()
        return None if row is None else str(row[0])

    def _import_key(
        self,
        *,
        kind: MeaningRunKind,
        request: MeaningRunRequest,
        permitted_set_hash: str,
        proposal_hash: str | None,
        terminal_marker: str | None,
    ) -> str:
        return canonical_sha256_hex(
            {
                "schema": "dithyramba.meaning_import_identity/2.0",
                "kind": kind.value,
                "request": request.payload(),
                "permitted_set_hash": permitted_set_hash,
                "proposal_hash": proposal_hash,
                "terminal_marker": terminal_marker,
            }
        )

    def _load_result(self, processing_run_id: str, *, deduplicated: bool) -> MeaningRunResult:
        row = self._repository._store.connection.execute(
            """
            SELECT processing_run_id, kind, corpus_snapshot_id, access_policy_id,
                   scope_hash, permitted_set_hash, extraction_profile,
                   code_version, generator_json, generator_hash,
                   review_budget_json, review_budget_hash, input_hash,
                   proposal_hash, import_key, status, candidate_count, unresolved_before,
                   backlog_warning, error_code, output_manifest_json,
                   output_hash, receipt_hash, started_at, finished_at
            FROM meaning_processing_runs
            WHERE processing_run_id = ? AND library_id = ?
            """,
            (processing_run_id, self.library_id),
        ).fetchone()
        if row is None:
            raise MeaningNotFoundError("semantic ProcessingRun does not exist")
        persisted_run_id = str(row["processing_run_id"])
        import_key = _strict_hash(row["import_key"], "semantic import_key")
        expected_run_id = canonical_content_id(
            "run",
            {"schema": "dithyramba.meaning_run_identity/2.0", "import_key": import_key},
        )
        if persisted_run_id != expected_run_id:
            raise MeaningPersistenceError("semantic ProcessingRun identity is inconsistent")
        generator_payload = _load_object(str(row["generator_json"]), "generator receipt")
        generator = _generator_from_payload(generator_payload)
        generator_hash = _strict_hash(row["generator_hash"], "semantic generator hash")
        if generator.generator_hash != generator_hash:
            raise MeaningPersistenceError("semantic generator receipt hash is inconsistent")
        budget_payload = _load_object(str(row["review_budget_json"]), "ReviewBudget")
        budget = _review_budget_from_payload(budget_payload)
        review_budget_hash = _strict_hash(
            row["review_budget_hash"],
            "semantic ReviewBudget hash",
        )
        if budget.budget_hash != review_budget_hash:
            raise MeaningPersistenceError("semantic ReviewBudget hash is inconsistent")
        output = _output_from_payload(
            _load_object(str(row["output_manifest_json"]), "output manifest")
        )
        output_hash = _strict_hash(row["output_hash"], "semantic output hash")
        if output.output_hash != output_hash or output.candidate_count != int(
            row["candidate_count"]
        ):
            raise MeaningPersistenceError("semantic output manifest is inconsistent")
        try:
            kind = MeaningRunKind(str(row["kind"]))
            profile = ExtractionProfile(str(row["extraction_profile"]))
            status = MeaningRunStatus(str(row["status"]))
        except ValueError as exc:
            raise MeaningPersistenceError("semantic ProcessingRun enum is invalid") from exc
        scope_hash = _strict_hash(row["scope_hash"], "semantic scope hash")
        permitted_set_hash = _strict_hash(
            row["permitted_set_hash"],
            "semantic permitted-set hash",
        )
        input_hash = _strict_hash(row["input_hash"], "semantic input hash")
        code_version = _strict_string(row["code_version"])
        if not code_version.strip():
            raise MeaningPersistenceError("semantic code version is empty")
        proposal_hash = (
            None
            if row["proposal_hash"] is None
            else _strict_hash(row["proposal_hash"], "semantic proposal hash")
        )
        backlog_warning = _strict_bool(row["backlog_warning"])
        error_code = None if row["error_code"] is None else _strict_string(row["error_code"])
        if (status is MeaningRunStatus.SUCCEEDED) != (error_code is None):
            raise MeaningPersistenceError("semantic ProcessingRun status/error are inconsistent")
        if status is not MeaningRunStatus.SUCCEEDED and output.candidate_count != 0:
            raise MeaningPersistenceError("non-success semantic run persisted candidates")
        started_at = _strict_timestamp(row["started_at"], "semantic run started_at")
        finished_at = _strict_timestamp(row["finished_at"], "semantic run finished_at")
        if _parse_timestamp(finished_at) < _parse_timestamp(started_at):
            raise MeaningPersistenceError("semantic ProcessingRun finishes before it starts")
        run_payload = _run_receipt_payload(
            processing_run_id=persisted_run_id,
            kind=kind.value,
            corpus_snapshot_id=str(row["corpus_snapshot_id"]),
            access_policy_id=str(row["access_policy_id"]),
            scope_hash=scope_hash,
            permitted_set_hash=permitted_set_hash,
            extraction_profile=profile.value,
            code_version=code_version,
            generator_hash=generator_hash,
            review_budget_hash=review_budget_hash,
            input_hash=input_hash,
            proposal_hash=proposal_hash,
            import_key=import_key,
            status=status.value,
            candidate_count=int(row["candidate_count"]),
            unresolved_before=int(row["unresolved_before"]),
            backlog_warning=backlog_warning,
            error_code=error_code,
            output_hash=output_hash,
            started_at=started_at,
            finished_at=finished_at,
        )
        receipt_hash = _strict_hash(row["receipt_hash"], "semantic run receipt hash")
        if canonical_sha256_hex(run_payload) != receipt_hash:
            raise MeaningPersistenceError("semantic ProcessingRun receipt hash is inconsistent")
        run = MeaningProcessingRun(
            processing_run_id=persisted_run_id,
            kind=kind,
            corpus_snapshot_id=str(row["corpus_snapshot_id"]),
            access_policy_id=str(row["access_policy_id"]),
            scope_hash=scope_hash,
            permitted_set_hash=permitted_set_hash,
            profile=profile,
            code_version=code_version,
            generator_hash=generator_hash,
            review_budget_hash=review_budget_hash,
            input_hash=input_hash,
            proposal_hash=proposal_hash,
            status=status,
            unresolved_before=int(row["unresolved_before"]),
            backlog_warning=backlog_warning,
            error_code=error_code,
            output=output,
            receipt_hash=receipt_hash,
            started_at=started_at,
            finished_at=finished_at,
        )
        coverage = self._load_coverage(run)
        receipt = self._load_read_receipt(run)
        expected_coverage_state = {
            MeaningRunStatus.SUCCEEDED: CoverageState.COMPLETE,
            MeaningRunStatus.PARTIAL: CoverageState.PARTIAL,
            MeaningRunStatus.FAILED: CoverageState.FAILED,
            MeaningRunStatus.REFUSED: CoverageState.REFUSED,
        }[run.status]
        if coverage.profile is not run.profile or coverage.state is not expected_coverage_state:
            raise MeaningPersistenceError("semantic run and CoverageReport states are inconsistent")
        if coverage.candidate_count != output.candidate_count:
            raise MeaningPersistenceError("semantic CoverageReport candidate count differs")
        if receipt.input_hash != run.input_hash:
            raise MeaningPersistenceError("semantic ReadReceipt input hash differs")
        if coverage.read_fragment_count != len(receipt.items):
            raise MeaningPersistenceError("semantic CoverageReport read count differs")
        return MeaningRunResult(run, coverage, receipt, deduplicated)

    def _load_coverage(self, run: MeaningProcessingRun) -> MeaningCoverageReport:
        row = self._repository._store.connection.execute(
            """
            SELECT coverage_report_id, extraction_profile, state,
                   read_fragment_count, candidate_count, omissions_json, report_hash
            FROM meaning_coverage_reports WHERE processing_run_id = ?
            """,
            (run.processing_run_id,),
        ).fetchone()
        if row is None:
            raise MeaningPersistenceError("semantic ProcessingRun lacks CoverageReport")
        raw_omissions = _load_array(str(row[5]), "meaning omissions")
        omissions = tuple(_omission_from_payload(item) for item in raw_omissions)
        payload: dict[str, object] = {
            "schema": "dithyramba.meaning_coverage_report/2.0",
            "processing_run_id": run.processing_run_id,
            "profile": str(row[1]),
            "state": str(row[2]),
            "read_fragment_count": int(row[3]),
            "candidate_count": int(row[4]),
            "omissions": [item.payload() for item in omissions],
        }
        if canonical_sha256_hex(payload) != str(row[6]):
            raise MeaningPersistenceError("semantic CoverageReport hash is inconsistent")
        if str(row[0]) != canonical_content_id("coverage", payload):
            raise MeaningPersistenceError("semantic CoverageReport identity is inconsistent")
        try:
            profile = ExtractionProfile(str(row[1]))
            state = CoverageState(str(row[2]))
        except ValueError as exc:
            raise MeaningPersistenceError("semantic CoverageReport enum is invalid") from exc
        return MeaningCoverageReport(
            coverage_report_id=str(row[0]),
            processing_run_id=run.processing_run_id,
            profile=profile,
            state=state,
            read_fragment_count=int(row[3]),
            candidate_count=int(row[4]),
            omissions=omissions,
            report_hash=str(row[6]),
        )

    def _load_read_receipt(self, run: MeaningProcessingRun) -> MeaningReadReceipt:
        row = self._repository._store.connection.execute(
            """
            SELECT read_receipt_id, input_hash, receipt_hash
            FROM meaning_read_receipts WHERE processing_run_id = ?
            """,
            (run.processing_run_id,),
        ).fetchone()
        if row is None:
            raise MeaningPersistenceError("semantic ProcessingRun lacks ReadReceipt")
        item_rows = self._repository._store.connection.execute(
            """
            SELECT source_fragment_id, source_version_id, source_id,
                   collection_ids_json, read_order, text_sha256
            FROM meaning_read_receipt_items
            WHERE read_receipt_id = ? ORDER BY read_order
            """,
            (str(row[0]),),
        ).fetchall()
        items_list: list[MeaningReadReceiptItem] = []
        try:
            for item in item_rows:
                raw_collections = _load_array(str(item[3]), "read item Collections")
                if any(type(value) is not str for value in raw_collections):
                    raise MeaningPersistenceError("persisted read item Collections are invalid")
                items_list.append(
                    MeaningReadReceiptItem(
                        source_fragment_id=str(item[0]),
                        source_version_id=str(item[1]),
                        source_id=str(item[2]),
                        collection_ids=tuple(cast(list[str], raw_collections)),
                        read_order=int(item[4]),
                        text_sha256=str(item[5]),
                    )
                )
        except MeaningContractError as exc:
            raise MeaningPersistenceError("persisted semantic read item is invalid") from exc
        items = tuple(items_list)
        if tuple(item.read_order for item in items) != tuple(range(len(items))):
            raise MeaningPersistenceError("semantic ReadReceipt order is not contiguous")
        persisted_input_hash = _strict_hash(row[1], "semantic ReadReceipt input hash")
        reconstructed_input_hash = canonical_sha256_hex(
            {
                "schema": "dithyramba.meaning_input/2.0",
                "fragments": [item.input_payload() for item in items],
            }
        )
        if reconstructed_input_hash != persisted_input_hash:
            raise MeaningPersistenceError("semantic ReadReceipt does not reconstruct input hash")
        payload: dict[str, object] = {
            "schema": "dithyramba.meaning_read_receipt/2.0",
            "processing_run_id": run.processing_run_id,
            "input_hash": persisted_input_hash,
            "items": [item.receipt_payload() for item in items],
        }
        if canonical_sha256_hex(payload) != str(row[2]):
            raise MeaningPersistenceError("semantic ReadReceipt hash is inconsistent")
        if str(row[0]) != canonical_content_id("read", payload):
            raise MeaningPersistenceError("semantic ReadReceipt identity is inconsistent")
        return MeaningReadReceipt(
            str(row[0]), run.processing_run_id, persisted_input_hash, items, str(row[2])
        )

    def _candidate_stubs(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[tuple[MeaningReviewTargetType, str, str, str], ...]:
        selected_connection = connection or self._repository._store.connection
        queries: tuple[tuple[MeaningReviewTargetType, str], ...] = (
            (
                MeaningReviewTargetType.STATEMENT,
                """SELECT statement_id, content_hash, created_at
                   FROM statements WHERE library_id = ?""",
            ),
            (
                MeaningReviewTargetType.VOICE,
                "SELECT voice_id, content_hash, created_at FROM voices WHERE library_id = ?",
            ),
            (
                MeaningReviewTargetType.ENTITY,
                "SELECT entity_id, content_hash, created_at FROM entities WHERE library_id = ?",
            ),
            (
                MeaningReviewTargetType.CONCEPT,
                "SELECT concept_id, content_hash, created_at FROM concepts WHERE library_id = ?",
            ),
            (
                MeaningReviewTargetType.TIME_CONTEXT,
                """SELECT time_context_id, content_hash, created_at
                   FROM time_contexts WHERE library_id = ?""",
            ),
            (
                MeaningReviewTargetType.EVIDENCE_LINK,
                """SELECT e.evidence_link_id, e.content_hash, e.created_at
                   FROM evidence_links AS e JOIN statements AS s ON s.statement_id = e.statement_id
                   WHERE s.library_id = ?""",
            ),
            (
                MeaningReviewTargetType.CONCEPT_MEANING,
                """SELECT cm.concept_meaning_id, cm.content_hash, cm.created_at
                   FROM concept_meanings AS cm JOIN concepts AS c ON c.concept_id = cm.concept_id
                   WHERE c.library_id = ?""",
            ),
            (
                MeaningReviewTargetType.RELATION,
                """SELECT relation_id, content_hash, created_at
                   FROM relations WHERE library_id = ?""",
            ),
        )
        stubs: list[tuple[MeaningReviewTargetType, str, str, str]] = []
        for target_type, query in queries:
            rows = selected_connection.execute(query, (self.library_id,)).fetchall()
            stubs.extend((target_type, str(row[0]), str(row[1]), str(row[2])) for row in rows)
        return tuple(sorted(stubs, key=lambda item: (item[3], item[0].value, item[1])))

    def _resolved_target_keys(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> set[tuple[MeaningReviewTargetType, str]]:
        selected_connection = connection or self._repository._store.connection
        rows = selected_connection.execute(
            """
            SELECT review_decision_id, target_type, target_id, target_hash,
                   action, scope_json, supersedes_review_decision_id
            FROM meaning_review_decisions
            WHERE library_id = ?
            UNION ALL
            SELECT review_decision_id, target_type, target_id, target_hash,
                   action, scope_json, supersedes_review_decision_id
            FROM relation_review_decisions
            WHERE library_id = ?
            ORDER BY review_decision_id
            """,
            (self.library_id, self.library_id),
        ).fetchall()
        rows_by_id = {str(row[0]): row for row in rows}
        superseded_ids: set[str] = set()
        for row in rows:
            if row[6] is None:
                continue
            prior_id = str(row[6])
            prior = rows_by_id.get(prior_id)
            if prior is None or tuple(str(row[index]) for index in (1, 2, 3, 5)) != tuple(
                str(prior[index]) for index in (1, 2, 3, 5)
            ):
                raise MeaningPersistenceError(
                    "semantic ReviewDecision supersede scope is inconsistent"
                )
            superseded_ids.add(prior_id)
        target_hashes = {
            (target_type, target_id): target_hash
            for target_type, target_id, target_hash, _created_at in self._candidate_stubs(
                connection=selected_connection
            )
        }
        coverage: dict[
            tuple[MeaningReviewTargetType, str, str],
            set[str],
        ] = {}
        for row in rows:
            if str(row[0]) in superseded_ids:
                continue
            target_type = MeaningReviewTargetType(str(row[1]))
            target_id = str(row[2])
            target_key = (target_type, target_id)
            if target_hashes.get(target_key) != str(row[3]):
                raise MeaningPersistenceError(
                    "active semantic ReviewDecision target hash is inconsistent"
                )
            action = MeaningReviewAction(str(row[4]))
            if action not in _TERMINAL_REVIEW_ACTIONS:
                continue
            scope = _scope_from_payload(_load_object(str(row[5]), "semantic ReviewScope"))
            coverage.setdefault((target_type, target_id, scope.use), set()).update(
                scope.collection_ids
            )
        resolved: set[tuple[MeaningReviewTargetType, str]] = set()
        for (target_type, target_id, _use), collection_ids in coverage.items():
            target_collections = set(
                self._target_collections(
                    target_type,
                    target_id,
                    connection=selected_connection,
                )
            )
            if target_collections and target_collections.issubset(collection_ids):
                resolved.add((target_type, target_id))
        return resolved

    def _target_info(
        self,
        target_type: MeaningReviewTargetType,
        target_id: str,
    ) -> _TargetInfo:
        table_info = {
            MeaningReviewTargetType.STATEMENT: ("statements", "statement_id"),
            MeaningReviewTargetType.VOICE: ("voices", "voice_id"),
            MeaningReviewTargetType.ENTITY: ("entities", "entity_id"),
            MeaningReviewTargetType.CONCEPT: ("concepts", "concept_id"),
            MeaningReviewTargetType.TIME_CONTEXT: ("time_contexts", "time_context_id"),
            MeaningReviewTargetType.RELATION: ("relations", "relation_id"),
        }
        if target_type in table_info:
            table, column = table_info[target_type]
            row = self._repository._store.connection.execute(
                f"""SELECT content_hash, created_at FROM "{table}"
                    WHERE "{column}" = ? AND library_id = ?""",
                (target_id, self.library_id),
            ).fetchone()
        elif target_type is MeaningReviewTargetType.EVIDENCE_LINK:
            row = self._repository._store.connection.execute(
                """SELECT e.content_hash, e.created_at FROM evidence_links AS e
                   JOIN statements AS s ON s.statement_id = e.statement_id
                   WHERE e.evidence_link_id = ? AND s.library_id = ?""",
                (target_id, self.library_id),
            ).fetchone()
        else:
            row = self._repository._store.connection.execute(
                """SELECT cm.content_hash, cm.created_at FROM concept_meanings AS cm
                   JOIN concepts AS c ON c.concept_id = cm.concept_id
                   WHERE cm.concept_meaning_id = ? AND c.library_id = ?""",
                (target_id, self.library_id),
            ).fetchone()
        if row is None:
            raise MeaningNotFoundError("semantic review target does not exist in this Library")
        collections = self._target_collections(target_type, target_id)
        if not collections:
            raise MeaningPersistenceError("semantic review target has no Collection closure")
        return _TargetInfo(target_type, target_id, str(row[0]), collections, str(row[1]))

    def _target_collections(
        self,
        target_type: MeaningReviewTargetType,
        target_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[str, ...]:
        selected_connection = connection or self._repository._store.connection
        if target_type is MeaningReviewTargetType.RELATION:
            return self._relation_target_collections(
                target_id,
                connection=selected_connection,
            )
        queries = {
            MeaningReviewTargetType.STATEMENT: (
                "SELECT collection_id FROM statements WHERE statement_id = ?"
            ),
            MeaningReviewTargetType.EVIDENCE_LINK: (
                """SELECT s.collection_id FROM evidence_links AS e
                   JOIN statements AS s ON s.statement_id = e.statement_id
                   WHERE e.evidence_link_id = ?"""
            ),
            MeaningReviewTargetType.ENTITY: (
                "SELECT collection_id FROM entity_mentions WHERE entity_id = ?"
            ),
            MeaningReviewTargetType.VOICE: (
                """SELECT collection_id FROM statements WHERE voice_id = ?
                   UNION SELECT s.collection_id FROM concept_meanings AS cm
                   JOIN concept_meaning_statements AS cms
                     ON cms.concept_meaning_id = cm.concept_meaning_id
                   JOIN statements AS s ON s.statement_id = cms.statement_id
                   WHERE cm.voice_id = ?"""
            ),
            MeaningReviewTargetType.CONCEPT: (
                """SELECT collection_id FROM concept_mentions WHERE concept_id = ?
                   UNION SELECT s.collection_id FROM concept_meanings AS cm
                   JOIN concept_meaning_statements AS cms
                     ON cms.concept_meaning_id = cm.concept_meaning_id
                   JOIN statements AS s ON s.statement_id = cms.statement_id
                   WHERE cm.concept_id = ?"""
            ),
            MeaningReviewTargetType.CONCEPT_MEANING: (
                """SELECT s.collection_id FROM concept_meaning_statements AS cms
                   JOIN statements AS s ON s.statement_id = cms.statement_id
                   WHERE cms.concept_meaning_id = ?"""
            ),
            MeaningReviewTargetType.TIME_CONTEXT: (
                """SELECT collection_id FROM statements WHERE time_context_id = ?
                   UNION SELECT s.collection_id FROM concept_meanings AS cm
                   JOIN concept_meaning_statements AS cms
                     ON cms.concept_meaning_id = cm.concept_meaning_id
                   JOIN statements AS s ON s.statement_id = cms.statement_id
                   WHERE cm.time_context_id = ?"""
            ),
        }
        query = queries[target_type]
        parameter_count = query.count("?")
        rows = selected_connection.execute(
            query,
            tuple(target_id for _ in range(parameter_count)),
        ).fetchall()
        return tuple(sorted({str(row[0]) for row in rows}))

    def _relation_target_collections(
        self,
        relation_id: str,
        *,
        connection: sqlite3.Connection,
    ) -> tuple[str, ...]:
        row = connection.execute(
            """
            SELECT subject_type, subject_id, object_type, object_id
            FROM relations
            WHERE relation_id = ? AND library_id = ?
            """,
            (relation_id, self.library_id),
        ).fetchone()
        if row is None:
            return ()
        collections = {
            str(item[0])
            for item in connection.execute(
                """
                SELECT DISTINCT statement.collection_id
                FROM relation_evidence_links AS relation_evidence
                JOIN evidence_links AS evidence
                  ON evidence.evidence_link_id = relation_evidence.evidence_link_id
                JOIN statements AS statement
                  ON statement.statement_id = evidence.statement_id
                WHERE relation_evidence.relation_id = ?
                """,
                (relation_id,),
            ).fetchall()
        }
        for raw_type, raw_id in ((row[0], row[1]), (row[2], row[3])):
            node_type = RelationNodeType(str(raw_type))
            node_id = str(raw_id)
            if node_type is RelationNodeType.STRUCTURE_UNIT:
                endpoint_rows = connection.execute(
                    """
                    SELECT DISTINCT member.collection_id
                    FROM structure_units AS unit
                    JOIN structure_unit_generations AS generation
                      ON generation.structure_unit_generation_id =
                         unit.structure_unit_generation_id
                    JOIN snapshot_members AS member
                      ON member.corpus_snapshot_id = generation.corpus_snapshot_id
                     AND member.source_version_id = unit.source_version_id
                     AND member.membership_state = 'active'
                    WHERE unit.structure_unit_id = ?
                      AND generation.library_id = ?
                    """,
                    (node_id, self.library_id),
                ).fetchall()
                collections.update(str(item[0]) for item in endpoint_rows)
                continue
            collections.update(
                self._target_collections(
                    MeaningReviewTargetType(node_type.value),
                    node_id,
                    connection=connection,
                )
            )
        return tuple(sorted(collections))

    def _load_decision_row(
        self,
        review_decision_id: str,
        *,
        connection: sqlite3.Connection,
    ) -> MeaningReviewDecision:
        decision_table = (
            "relation_review_decisions"
            if review_decision_id.startswith("review_relation_")
            else "meaning_review_decisions"
        )
        row = connection.execute(
            f"""
            SELECT review_decision_id, review_session_id, target_type, target_id,
                   target_hash, action, reason, authority, scope_json,
                   replacement_target_id, supersedes_review_decision_id, created_at
            FROM {decision_table}
            WHERE review_decision_id = ? AND library_id = ?
            """,
            (review_decision_id, self.library_id),
        ).fetchone()
        if row is None:
            raise MeaningNotFoundError("semantic ReviewDecision does not exist")
        scope = _scope_from_payload(_load_object(str(row[8]), "semantic ReviewScope"))
        decision = MeaningReviewDecision(
            review_decision_id=str(row[0]),
            review_session_id=str(row[1]),
            target_type=MeaningReviewTargetType(str(row[2])),
            target_id=str(row[3]),
            target_hash=str(row[4]),
            action=MeaningReviewAction(str(row[5])),
            reason=str(row[6]),
            authority=str(row[7]),
            scope=scope,
            replacement_target_id=None if row[9] is None else str(row[9]),
            supersedes_review_decision_id=None if row[10] is None else str(row[10]),
            created_at=str(row[11]),
        )
        target = self._target_info(decision.target_type, decision.target_id)
        if decision.target_hash != target.target_hash:
            raise MeaningPersistenceError("persisted semantic ReviewDecision hash is stale")
        return decision

    def _load_active_relation_decisions(
        self,
        *,
        target_id: str,
        target_hash: str,
        connection: sqlite3.Connection,
    ) -> tuple[MeaningReviewDecision, ...]:
        """Load every active decision before comparing ReviewScope semantically.

        ``scope_json`` is deliberately not part of the SQL predicate. SQLite can
        store schema-valid JSON with noncanonical whitespace or key order, and a
        byte comparison would let two representations of the same ReviewScope
        evade the uniqueness check. Loading every active row also makes malformed
        persisted state fail closed instead of being silently ignored.
        """

        rows = connection.execute(
            """
            SELECT active.review_decision_id
            FROM relation_review_decisions AS active
            WHERE active.library_id = ?
              AND active.target_type = 'relation'
              AND active.target_id = ?
              AND active.target_hash = ?
              AND active.action <> 'supersede'
              AND NOT EXISTS (
                  SELECT 1 FROM relation_review_decisions AS later
                  WHERE later.supersedes_review_decision_id = active.review_decision_id
              )
            ORDER BY active.created_at, active.review_decision_id
            """,
            (self.library_id, target_id, target_hash),
        ).fetchall()
        return tuple(self._load_decision_row(str(row[0]), connection=connection) for row in rows)


def _canonical_text(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _run_receipt_payload(
    *,
    processing_run_id: str,
    kind: str,
    corpus_snapshot_id: str,
    access_policy_id: str,
    scope_hash: str,
    permitted_set_hash: str,
    extraction_profile: str,
    code_version: str,
    generator_hash: str,
    review_budget_hash: str,
    input_hash: str,
    proposal_hash: str | None,
    import_key: str,
    status: str,
    candidate_count: int,
    unresolved_before: int,
    backlog_warning: bool,
    error_code: str | None,
    output_hash: str,
    started_at: str,
    finished_at: str,
) -> dict[str, object]:
    return {
        "schema": "dithyramba.meaning_processing_run/2.0",
        "processing_run_id": processing_run_id,
        "kind": kind,
        "corpus_snapshot_id": corpus_snapshot_id,
        "access_policy_id": access_policy_id,
        "scope_hash": scope_hash,
        "permitted_set_hash": permitted_set_hash,
        "extraction_profile": extraction_profile,
        "code_version": code_version,
        "generator_hash": generator_hash,
        "review_budget_hash": review_budget_hash,
        "input_hash": input_hash,
        "proposal_hash": proposal_hash,
        "import_key": import_key,
        "status": status,
        "candidate_count": candidate_count,
        "unresolved_before": unresolved_before,
        "backlog_warning": backlog_warning,
        "error_code": error_code,
        "output_hash": output_hash,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def _load_object(value: str, label: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise MeaningPersistenceError(f"persisted {label} is invalid JSON") from exc
    if type(loaded) is not dict:
        raise MeaningPersistenceError(f"persisted {label} must be an object")
    result = cast(dict[str, object], loaded)
    if _canonical_text(result) != value:
        raise MeaningPersistenceError(f"persisted {label} is not canonical JSON")
    return result


def _load_array(value: str, label: str) -> list[object]:
    try:
        loaded = json.loads(value)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise MeaningPersistenceError(f"persisted {label} is invalid JSON") from exc
    if type(loaded) is not list:
        raise MeaningPersistenceError(f"persisted {label} must be an array")
    result = cast(list[object], loaded)
    if _canonical_text(result) != value:
        raise MeaningPersistenceError(f"persisted {label} is not canonical JSON")
    return result


def _output_from_payload(payload: dict[str, object]) -> MeaningOutputManifest:
    expected = {
        "schema",
        "voice_ids",
        "entity_ids",
        "entity_mention_ids",
        "concept_ids",
        "concept_mention_ids",
        "concept_meaning_ids",
        "time_context_ids",
        "statement_ids",
        "evidence_link_ids",
    }
    if (
        set(payload) != expected
        or payload.get("schema") != "dithyramba.meaning_output_manifest/2.0"
    ):
        raise MeaningPersistenceError("persisted semantic output manifest shape is invalid")

    def string_tuple(name: str) -> tuple[str, ...]:
        value = payload[name]
        if type(value) is not list or any(type(item) is not str for item in value):
            raise MeaningPersistenceError("persisted semantic output IDs are invalid")
        return tuple(cast(list[str], value))

    try:
        return MeaningOutputManifest(
            voice_ids=string_tuple("voice_ids"),
            entity_ids=string_tuple("entity_ids"),
            entity_mention_ids=string_tuple("entity_mention_ids"),
            concept_ids=string_tuple("concept_ids"),
            concept_mention_ids=string_tuple("concept_mention_ids"),
            concept_meaning_ids=string_tuple("concept_meaning_ids"),
            time_context_ids=string_tuple("time_context_ids"),
            statement_ids=string_tuple("statement_ids"),
            evidence_link_ids=string_tuple("evidence_link_ids"),
        )
    except MeaningContractError as exc:
        raise MeaningPersistenceError("persisted semantic output manifest is invalid") from exc


def _generator_from_payload(payload: dict[str, object]) -> GeneratorProfile:
    expected = {
        "schema",
        "generator_id",
        "revision",
        "license",
        "prompt_version",
        "dimensions",
        "external_provider",
    }
    if set(payload) != expected or payload.get("schema") != "dithyramba.meaning_generator/1.0":
        raise MeaningPersistenceError("persisted generator receipt shape is invalid")
    try:
        return GeneratorProfile(
            generator_id=_strict_string(payload["generator_id"]),
            revision=_strict_string(payload["revision"]),
            license=_strict_string(payload["license"]),
            prompt_version=_strict_string(payload["prompt_version"]),
            dimensions=(
                None if payload["dimensions"] is None else _strict_int(payload["dimensions"])
            ),
            external_provider=_strict_bool(payload["external_provider"]),
        )
    except MeaningContractError as exc:
        raise MeaningPersistenceError("persisted generator receipt is invalid") from exc


def _review_budget_from_payload(payload: dict[str, object]) -> ReviewBudget:
    expected = {
        "schema",
        "max_candidates",
        "max_read_fragments",
        "max_unresolved_candidates",
        "review_decision_limit",
    }
    if set(payload) != expected or payload.get("schema") != "dithyramba.review_budget/1.0":
        raise MeaningPersistenceError("persisted ReviewBudget shape is invalid")
    try:
        return ReviewBudget(
            max_candidates=_strict_int(payload["max_candidates"]),
            max_read_fragments=_strict_int(payload["max_read_fragments"]),
            max_unresolved_candidates=_strict_int(payload["max_unresolved_candidates"]),
            review_decision_limit=_strict_int(payload["review_decision_limit"]),
        )
    except MeaningContractError as exc:
        raise MeaningPersistenceError("persisted ReviewBudget is invalid") from exc


def _omission_from_payload(payload: object) -> MeaningOmission:
    if type(payload) is not dict:
        raise MeaningPersistenceError("persisted meaning omission must be an object")
    values = cast(dict[str, object], payload)
    if set(values) != {"category", "reason_code", "count"}:
        raise MeaningPersistenceError("persisted meaning omission shape is invalid")
    try:
        return MeaningOmission(
            OmissionCategory(_strict_string(values["category"])),
            _strict_string(values["reason_code"]),
            _strict_int(values["count"]),
        )
    except (MeaningContractError, ValueError) as exc:
        raise MeaningPersistenceError("persisted meaning omission is invalid") from exc


def _scope_from_payload(payload: dict[str, object]) -> MeaningReviewScope:
    if (
        set(payload) != {"schema", "collection_ids", "use"}
        or payload.get("schema") != "dithyramba.meaning_review_scope/1.0"
    ):
        raise MeaningPersistenceError("persisted semantic ReviewScope shape is invalid")
    collections = payload["collection_ids"]
    if type(collections) is not list or any(type(item) is not str for item in collections):
        raise MeaningPersistenceError("persisted semantic ReviewScope Collections are invalid")
    try:
        return MeaningReviewScope(
            tuple(cast(list[str], collections)),
            _strict_string(payload["use"]),
        )
    except MeaningContractError as exc:
        raise MeaningPersistenceError("persisted semantic ReviewScope is invalid") from exc


def _require_content_row(
    connection: sqlite3.Connection,
    table: str,
    id_column: str,
    identifier: str,
    content_hash: str,
) -> None:
    allowed = {
        ("voices", "voice_id"),
        ("entities", "entity_id"),
        ("concepts", "concept_id"),
        ("time_contexts", "time_context_id"),
        ("statements", "statement_id"),
        ("evidence_links", "evidence_link_id"),
        ("concept_meanings", "concept_meaning_id"),
        ("entity_mentions", "entity_mention_id"),
        ("concept_mentions", "concept_mention_id"),
    }
    if (table, id_column) not in allowed:
        raise MeaningPersistenceError("invalid typed semantic table verification")
    row = connection.execute(
        f'SELECT content_hash FROM "{table}" WHERE "{id_column}" = ?',
        (identifier,),
    ).fetchone()
    if row is None or str(row[0]) != content_hash:
        raise MeaningPersistenceError("content-addressed semantic row is inconsistent")


def _strict_string(value: object) -> str:
    if type(value) is not str:
        raise MeaningPersistenceError("persisted semantic value must be text")
    return value


def _strict_int(value: object) -> int:
    if type(value) is not int:
        raise MeaningPersistenceError("persisted semantic value must be an integer")
    return value


def _strict_bool(value: object) -> bool:
    if value not in (0, 1, False, True) or type(value) not in (int, bool):
        raise MeaningPersistenceError("persisted semantic value must be boolean")
    return bool(value)


def _strict_hash(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MeaningPersistenceError(f"persisted {label} must be lowercase SHA-256")
    return value


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def _strict_timestamp(value: object, label: str) -> str:
    text = _strict_string(value)
    try:
        parsed = _parse_timestamp(text)
    except ValueError as exc:
        raise MeaningPersistenceError(f"persisted {label} is not canonical UTC") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != text:
        raise MeaningPersistenceError(f"persisted {label} is not canonical UTC")
    return text
