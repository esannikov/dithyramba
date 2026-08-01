"""End-to-end least-context agent facade tests over a cold Library reopen."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    QueryExclusions,
)
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.ingest.service import IngestService
from dithyramba.interactive import AgentResearchFacade, SessionContextBudget
from dithyramba.library import LibraryConfig
from dithyramba.persistence import initialize_library, open_library
from dithyramba.recall import RetrievalBudget, current_fts_runtime_profile
from dithyramba.sessions import ResearchSessionBrief, SessionStatus


class _TickingClock:
    def __init__(self) -> None:
        self._next = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self._next
        self._next += timedelta(microseconds=1)
        return current


def _prepare_library(repository: object, root: Path) -> tuple[str, str, str, str]:
    from dithyramba.persistence import LibraryRepository

    assert isinstance(repository, LibraryRepository)
    collection = repository.create_collection(
        CollectionConfig(
            library_id=repository.library_id,
            name="Interactive corpus",
            kind=CollectionKind.CORPUS,
            roots=(
                build_collection_root(
                    root,
                    data_root=repository.paths.application_data_root,
                ),
            ),
        )
    )
    collection_id = collection.config.collection_id
    ingested = IngestService(repository).ingest_path(collection_id, "craft.md")
    source_id = ingested.outcomes[0].source_id
    assert source_id is not None
    snapshot = repository.freeze_snapshot((collection_id,))
    policy = AccessPolicySnapshot(
        access_policy_id="policy_interactive_research",
        library_id=repository.library_id,
        allowed_purposes=("research",),
        collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
    )
    repository.persist_access_policy(name="Interactive research", snapshot=policy)
    return collection_id, snapshot.corpus_snapshot_id, policy.access_policy_id, source_id


def test_agent_turn_reopens_with_same_compact_context_and_exact_packet(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "craft.md").write_text(
        "# Dialogue evidence\n\nBlocking changes the power relation inside a dialogue scene.\n",
        encoding="utf-8",
    )
    data_root = tmp_path / "data"
    budget = SessionContextBudget(
        recent_questions=1,
        recent_drafts=1,
        recent_gaps=1,
        recent_rejected_paths=1,
        recent_decisions=1,
        evidence_references=10,
        candidate_references=10,
    )
    profile = current_fts_runtime_profile().profile_version
    with initialize_library(
        LibraryConfig(name="Interactive research"), data_root=data_root
    ) as repository:
        library_id = repository.library_id
        collection_id, snapshot_id, policy_id, _source_id = _prepare_library(repository, root)
        facade = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=profile,
            clock=_TickingClock(),
            context_budget=budget,
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="How should dialogue staging reveal power?",
                intended_use="directorial decision",
                success_criteria=("find exact craft support", "retain open gaps"),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
        )
        assert opened.status is SessionStatus.OPEN
        turn = facade.recall(
            opened.session_id,
            "dialogue blocking power relation",
            retrieval=RetrievalBudget(max_candidates=10, max_source_fragments=3),
        )
        assert turn.evidence_packet.source_fragments
        assert turn.context.questions[-1].text == "dialogue blocking power relation"
        assert {item.artifact_kind.value for item in turn.context.evidence_references} == {
            "evidence_packet",
            "source_fragment",
        }
        facade.record_draft(
            opened.session_id,
            "Stage the change in status through movement; this remains a draft.",
        )
        facade.record_gap(opened.session_id, "Need a counterexample with static blocking.")
        facade.record_gap(opened.session_id, "Need evidence about eyelines.")
        facade.record_gap(opened.session_id, "Need a scene-specific production constraint.")
        final = facade.reject_path(
            opened.session_id,
            "Do not treat generic dialogue advice as evidence for this scene.",
        )
        assert final.omissions.gaps == 2
        assert final.gaps[0].text == "Need a scene-specific production constraint."
        assert final.drafts[0].text.endswith("this remains a draft.")
        assert final.evidence_references == turn.context.evidence_references
        final_hash = final.context_hash
        assert not hasattr(facade, "accept_candidate")
        assert not hasattr(facade, "record_decision")
        assert not hasattr(facade, "close")

    with open_library(library_id, data_root=data_root) as repository:
        reopened = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=profile,
            clock=_TickingClock(),
            context_budget=budget,
        ).context(opened.session_id)
        assert reopened.context_hash == final_hash
        assert reopened.state_hash == final.state_hash
        assert reopened.evidence_references == final.evidence_references


def test_agent_facade_preserves_pre_read_source_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "craft.md").write_text(
        "# Evidence\n\nThis sentence would match if its Source were permitted.\n",
        encoding="utf-8",
    )
    with initialize_library(
        LibraryConfig(name="Excluded interactive research"),
        data_root=tmp_path / "data",
    ) as repository:
        collection_id, snapshot_id, policy_id, source_id = _prepare_library(repository, root)
        facade = AgentResearchFacade(
            repository,
            agent_id="agent:test",
            profile_version=current_fts_runtime_profile().profile_version,
            clock=_TickingClock(),
        )
        opened = facade.open(
            brief=ResearchSessionBrief.create(
                question="Does excluded material remain unread?",
                intended_use="access-boundary verification",
                success_criteria=("preserve exclusion",),
            ),
            corpus_snapshot_id=snapshot_id,
            access_policy_id=policy_id,
            purpose="research",
            collection_ids=(collection_id,),
            exclusions=QueryExclusions(source_ids=(source_id,)),
        )
        turn = facade.recall(opened.session_id, "sentence Source permitted")

        assert turn.evidence_packet.source_fragments == ()
        assert turn.context.evidence_references[0].artifact_kind.value == "evidence_packet"
        assert len(turn.context.evidence_references) == 1
