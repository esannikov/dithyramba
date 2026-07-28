"""Public fixtures through the complete protected SQLite recall path."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

import pytest

from dithyramba.access import (
    AccessPolicySnapshot,
    CollectionRule,
    PolicyEffect,
    QueryExclusions,
)
from dithyramba.collections import (
    CollectionConfig,
    CollectionKind,
    build_collection_root,
)
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import (
    LibraryRepository,
    ProcessingRunStatus,
    SQLiteRecallBackend,
    initialize_library,
)
from dithyramba.provenance import SourceFamilyRole
from dithyramba.recall import (
    EvidencePacket,
    OmissionCategory,
    QueryRequest,
    RecallService,
    RecallSnapshotError,
    RetrievalBudget,
    current_fts_runtime_profile,
)
from dithyramba.snapshots import CorpusSnapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"
PROFILE_VERSION = current_fts_runtime_profile().profile_version
FROZEN_FIXTURE_MANIFEST_HASHES = {
    "caillebotte_regression": ("19c8b49a5cf282602ddf2cd6ea682589fbf1da6771f32672fb24353f3bb8779e"),
    "public_multilingual": ("cb3a5965b0cad98fd789ca9473bc98d2e9bccf1042aa168158fca6987047fcae"),
    "synthetic_isolation": ("a54a2bc80efa687a0f1fe810d46e3f86ad825ff0b084dc88f9d949eeb173f047"),
}


@dataclass(frozen=True, slots=True)
class _IsolationRuntime:
    fixture: dict[str, Any]
    repository: LibraryRepository
    service: RecallService
    snapshot: CorpusSnapshot
    policy_id: str
    collection_ids: tuple[str, ...]
    actual_fragment_by_declared: dict[str, str]
    text_by_declared: dict[str, str]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate fixture key: {key}")
        value[key] = item
    return value


def _load_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    decoded: object = json.loads(raw, object_pairs_hook=_unique_object)
    if type(decoded) is not dict:
        raise ValueError(f"fixture root must be an object: {path}")
    value = cast(dict[str, Any], decoded)
    if raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"fixture is not canonical JSON plus LF: {path}")
    return value


def _fixture_artifacts(name: str) -> dict[str, dict[str, Any]]:
    manifest_path = FIXTURES / name / "manifest.json"
    manifest = _load_canonical(manifest_path)
    assert canonical_sha256_hex(manifest) == FROZEN_FIXTURE_MANIFEST_HASHES[name]
    loaded: dict[str, dict[str, Any]] = {}
    rows = _list_of_dicts(manifest, "artifacts")
    for row in rows:
        relative = _string(row, "path")
        assert Path(relative).name == relative
        payload = _load_canonical(manifest_path.parent / relative)
        assert canonical_sha256_hex(payload) == _string(row, "canonical_sha256")
        loaded[relative] = payload
    return loaded


def _list_of_dicts(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = row.get(key)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"{key} must be a JSON object array")
    return cast(list[dict[str, Any]], value)


def _string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if type(value) is not str or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _strings(row: dict[str, Any], key: str) -> tuple[str, ...]:
    value = row.get(key)
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise ValueError(f"{key} must be a non-empty-string array")
    return tuple(cast(list[str], value))


def _integer(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _collection(
    repository: LibraryRepository,
    *,
    root: Path,
    name: str,
    kind: CollectionKind,
    collection_id: str | None = None,
) -> str:
    root.mkdir(parents=True)
    collection_root = build_collection_root(
        root,
        data_root=repository.paths.application_data_root,
    )
    config = (
        CollectionConfig(
            collection_id=collection_id,
            library_id=repository.library_id,
            name=name,
            kind=kind,
            roots=(collection_root,),
        )
        if collection_id is not None
        else CollectionConfig(
            library_id=repository.library_id,
            name=name,
            kind=kind,
            roots=(collection_root,),
        )
    )
    return repository.create_collection(config).config.collection_id


def _ingest_collection(repository: LibraryRepository, collection_id: str) -> None:
    result = IngestService(repository).ingest_collection(collection_id)
    assert result.run.status is ProcessingRunStatus.SUCCEEDED
    assert all(outcome.failure_code is None for outcome in result.outcomes)


def _service(repository: LibraryRepository) -> RecallService:
    return RecallService(
        SQLiteRecallBackend(repository),
        profile_version=PROFILE_VERSION,
    )


def _request(
    *,
    question: str,
    repository: LibraryRepository,
    collection_ids: tuple[str, ...],
    snapshot: CorpusSnapshot,
    policy_id: str,
    exclusions: QueryExclusions | None = None,
) -> QueryRequest:
    return QueryRequest(
        question=question,
        library_id=repository.library_id,
        collection_ids=collection_ids,
        corpus_snapshot_id=snapshot.corpus_snapshot_id,
        access_policy_id=policy_id,
        purpose="research",
        exclusions=exclusions or QueryExclusions(),
        retrieval=RetrievalBudget(max_candidates=10, max_source_fragments=10),
    )


def _packet_fragment_ids(packet: EvidencePacket) -> set[str]:
    return {
        *(item.source_fragment_id for item in packet.source_fragments),
        *(item.source_fragment_id for item in packet.read_receipt.items),
        *(item.source_fragment_id for item in packet.retrieval_receipt.trace),
    }


def _source_fixture_name(canonical_uri: str) -> str:
    return Path(unquote(urlsplit(canonical_uri).path)).stem


def test_public_multilingual_macro_recall_at_10_through_real_library_pipeline(
    tmp_path: Path,
) -> None:
    artifacts = _fixture_artifacts("public_multilingual")
    corpus = artifacts["corpus.json"]
    query_set = artifacts["queries.json"]
    fragment_rows = _list_of_dicts(corpus, "fragments")
    query_rows = _list_of_dicts(query_set, "queries")
    assert len(fragment_rows) == 12
    assert len(query_rows) == 12
    source_ref_by_text_hash = {
        _string(row, "text_sha256"): _string(row, "source_ref") for row in fragment_rows
    }

    repository = initialize_library(
        LibraryConfig(name="P3 public multilingual acceptance"),
        data_root=tmp_path / "data",
    )
    try:
        root = tmp_path / "sources"
        collection_id = _collection(
            repository,
            root=root,
            name="Public multilingual",
            kind=CollectionKind.CORPUS,
        )
        for row in fragment_rows:
            (root / f"{_string(row, 'source_fragment_id')}.txt").write_text(
                _string(row, "text"),
                encoding="utf-8",
            )
        _ingest_collection(repository, collection_id)
        assert len(repository.list_sources(collection_id=collection_id)) == 12
        snapshot = repository.freeze_snapshot((collection_id,))
        policy = AccessPolicySnapshot(
            access_policy_id="policy_public_multilingual",
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
        )
        repository.persist_access_policy(name="Public fixture", snapshot=policy)
        service = _service(repository)

        scores: dict[str, list[Fraction]] = defaultdict(list)
        for row in query_rows:
            language = _string(row, "language")
            expected_refs = set(_strings(row, "expected_source_refs"))
            packet = service.recall(
                _request(
                    question=_string(row, "text"),
                    repository=repository,
                    collection_ids=(collection_id,),
                    snapshot=snapshot,
                    policy_id=policy.access_policy_id,
                )
            ).packet
            assert len(packet.read_receipt.items) == 12
            assert all(item.source_address is not None for item in packet.source_fragments)
            retrieved_refs = {
                source_ref_by_text_hash[item.text_sha256] for item in packet.source_fragments
            }
            scores[language].append(
                Fraction(len(retrieved_refs.intersection(expected_refs)), len(expected_refs))
            )

        assert {language: len(values) for language, values in scores.items()} == {
            "en": 6,
            "uk": 6,
        }
        language_scores = {
            language: sum(values, start=Fraction()) / len(values)
            for language, values in scores.items()
        }
        macro = sum(
            (score for values in scores.values() for score in values),
            start=Fraction(),
        ) / len(query_rows)
        metric = cast(dict[str, Any], query_set["metric"])
        assert macro >= Fraction(_integer(metric, "minimum_macro_recall_micros"), 1_000_000)
        language_gate = Fraction(
            _integer(metric, "minimum_language_recall_micros"),
            1_000_000,
        )
        assert language_scores["uk"] >= language_gate
        assert language_scores["en"] >= language_gate
    finally:
        repository.close()


def test_caillebotte_duplicate_ai_ledger_independence_and_family_exclusion(
    tmp_path: Path,
) -> None:
    fixture = _fixture_artifacts("caillebotte_regression")["fixture.json"]
    sources = _list_of_dicts(fixture, "sources")
    ledger = _list_of_dicts(fixture, "ledger_entries")
    expected = cast(dict[str, Any], fixture["expected_accounting"])
    repository = initialize_library(
        LibraryConfig(name="P3 Caillebotte regression acceptance"),
        data_root=tmp_path / "data",
    )
    try:
        root = tmp_path / "sources"
        collection_id = _collection(
            repository,
            root=root,
            name="Regression corpus",
            kind=CollectionKind.CORPUS,
        )
        for row in sources:
            (root / f"{_string(row, 'source_id')}.txt").write_text(
                _string(row, "text"),
                encoding="utf-8",
            )
        _ingest_collection(repository, collection_id)
        actual_sources = {
            _source_fixture_name(source.canonical_uri): source
            for source in repository.list_sources(collection_id=collection_id)
        }
        assert set(actual_sources) == {_string(row, "source_id") for row in sources}
        roots = tuple(
            source
            for source in actual_sources.values()
            if source.family_role is SourceFamilyRole.ROOT
        )
        duplicates = tuple(
            source
            for source in actual_sources.values()
            if source.family_role is SourceFamilyRole.DUPLICATE
        )
        assert len(roots) == _integer(expected, "independent_unit_count") == 2
        assert len(duplicates) == 1
        secondary = actual_sources["source_caillebotte_secondary"]
        same_bytes = (
            actual_sources["source_caillebotte_primary"],
            actual_sources["source_caillebotte_exact_duplicate"],
        )
        family_root = next(
            source for source in same_bytes if source.family_role is SourceFamilyRole.ROOT
        )
        family_duplicate = next(
            source for source in same_bytes if source.family_role is SourceFamilyRole.DUPLICATE
        )
        assert family_duplicate.source_family_id == family_root.source_family_id
        assert family_duplicate.root_source_id == family_root.source_id
        assert secondary.source_family_id != family_root.source_family_id

        fragments_by_fixture = {
            fixture_id: repository.list_source_fragments(source.current_source_version_id)
            for fixture_id, source in actual_sources.items()
        }
        snapshot = repository.freeze_snapshot((collection_id,))
        policy = AccessPolicySnapshot(
            access_policy_id="policy_caillebotte_regression",
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
        )
        repository.persist_access_policy(name="Regression fixture", snapshot=policy)
        service = _service(repository)
        base_request = _request(
            question="orchard-17 review slot blue",
            repository=repository,
            collection_ids=(collection_id,),
            snapshot=snapshot,
            policy_id=policy.access_policy_id,
        )
        packet = service.recall(base_request).packet
        independent_families = {item.source_family_id for item in packet.source_fragments}
        assert None not in independent_families
        assert len(independent_families) == 2
        packet_text = packet.canonical_bytes.decode("utf-8")
        for entry in ledger:
            assert _string(entry, "entry_id") not in packet_text
            assert _string(entry, "text") not in packet_text

        excluded_packet = service.recall(
            _request(
                question="orchard-17 review slot blue",
                repository=repository,
                collection_ids=(collection_id,),
                snapshot=snapshot,
                policy_id=policy.access_policy_id,
                exclusions=QueryExclusions(
                    source_family_ids=(family_root.source_family_id,),
                ),
            )
        ).packet
        excluded_ids = {
            item.source_fragment_id
            for fixture_id in (
                "source_caillebotte_primary",
                "source_caillebotte_exact_duplicate",
            )
            for item in fragments_by_fixture[fixture_id]
        }
        assert _packet_fragment_ids(excluded_packet).isdisjoint(excluded_ids)
        assert {item.source_family_id for item in excluded_packet.source_fragments} == {
            secondary.source_family_id
        }
        explicit = tuple(
            item
            for item in excluded_packet.coverage_report.omissions
            if item.category is OmissionCategory.EXPLICIT_EXCLUSION
        )
        assert len(explicit) == 1 and explicit[0].count == 1
    finally:
        repository.close()


def _isolation_runtime(
    tmp_path: Path,
    library: dict[str, Any],
) -> _IsolationRuntime:
    library_id = _string(library, "library_id")
    repository = initialize_library(
        LibraryConfig(library_id=library_id, name=_string(library, "label")),
        data_root=tmp_path / "data",
    )
    collections = _list_of_dicts(library, "collections")
    collection_ids: list[str] = []
    actual_fragment_by_declared: dict[str, str] = {}
    text_by_declared: dict[str, str] = {}
    rules: list[CollectionRule] = []
    for row in collections:
        collection_id = _string(row, "collection_id")
        state = _string(row, "membership_state")
        kind = {
            "corpus": CollectionKind.CORPUS,
            "branch": CollectionKind.BRANCH,
            "holdout": CollectionKind.HOLDOUT,
        }[_string(row, "kind")]
        root = tmp_path / "sources" / library_id / collection_id
        created_id = _collection(
            repository,
            root=root,
            name=f"Fixture {state} {collection_id[-4:]}",
            kind=kind,
            collection_id=collection_id,
        )
        fragments = _list_of_dicts(row, "fragments")
        assert len(fragments) == 1
        fixture_fragment = fragments[0]
        declared_id = _string(fixture_fragment, "source_fragment_id")
        text = _string(fixture_fragment, "text")
        filename = f"{declared_id}.txt"
        (root / filename).write_text(text, encoding="utf-8")
        _ingest_collection(repository, created_id)
        source = repository.list_sources(collection_id=created_id)[0]
        actual_fragments = repository.list_source_fragments(source.current_source_version_id)
        assert len(actual_fragments) == 1
        actual_fragment_by_declared[declared_id] = actual_fragments[0].source_fragment_id
        text_by_declared[declared_id] = text
        collection_ids.append(created_id)
        rules.append(
            CollectionRule(
                created_id,
                PolicyEffect.DENY if state == "excluded" else PolicyEffect.ALLOW,
            )
        )

    snapshot = repository.freeze_snapshot(tuple(collection_ids))
    policy_id = "policy_isolation_" + ("public" if len(collection_ids) > 1 else "private")
    policy = AccessPolicySnapshot(
        access_policy_id=policy_id,
        library_id=library_id,
        allowed_purposes=("research",),
        collection_rules=tuple(rules),
    )
    repository.persist_access_policy(name=f"Isolation {library_id[-4:]}", snapshot=policy)
    return _IsolationRuntime(
        fixture=library,
        repository=repository,
        service=_service(repository),
        snapshot=snapshot,
        policy_id=policy_id,
        collection_ids=tuple(collection_ids),
        actual_fragment_by_declared=actual_fragment_by_declared,
        text_by_declared=text_by_declared,
    )


def test_two_library_canaries_denied_and_holdout_content_never_cross_boundaries(
    tmp_path: Path,
) -> None:
    fixture = _fixture_artifacts("synthetic_isolation")["libraries.json"]
    libraries = _list_of_dicts(fixture, "libraries")
    expectations = _list_of_dicts(fixture, "visibility_expectations")
    canaries = cast(dict[str, str], fixture["canaries"])
    runtimes = {
        _string(library, "library_id"): _isolation_runtime(tmp_path, library)
        for library in libraries
    }
    all_actual = {
        declared: actual
        for runtime in runtimes.values()
        for declared, actual in runtime.actual_fragment_by_declared.items()
    }
    try:
        for expectation in expectations:
            library_id = _string(expectation, "library_id")
            runtime = runtimes[library_id]
            expected_declared = set(_strings(expectation, "expected_source_fragment_ids"))
            forbidden_canaries = _strings(expectation, "forbidden_canaries")
            packet = runtime.service.recall(
                _request(
                    question=" ".join(canaries.values()),
                    repository=runtime.repository,
                    collection_ids=runtime.collection_ids,
                    snapshot=runtime.snapshot,
                    policy_id=runtime.policy_id,
                )
            ).packet
            packet_ids = _packet_fragment_ids(packet)
            assert packet_ids == {all_actual[item] for item in expected_declared}
            packet_text = packet.canonical_bytes.decode("utf-8")
            for canary in forbidden_canaries:
                assert canary not in packet_text
            forbidden_declared = set(all_actual).difference(expected_declared)
            forbidden_actual = {all_actual[item] for item in forbidden_declared}
            assert packet_ids.isdisjoint(forbidden_actual)
            assert all(item not in packet_text for item in forbidden_declared)

        public = runtimes["library_11111111111141118111111111111111"]
        private = runtimes["library_22222222222242228222222222222222"]
        public_packet = public.service.recall(
            _request(
                question=" ".join(canaries.values()),
                repository=public.repository,
                collection_ids=public.collection_ids,
                snapshot=public.snapshot,
                policy_id=public.policy_id,
            )
        ).packet
        assert public_packet.access_receipt.policy_omission_present is True
        assert any(
            item.category is OmissionCategory.POLICY and item.count is None
            for item in public_packet.coverage_report.omissions
        )
        denied_or_holdout = {
            all_actual["fragment_isolation_excluded"],
            all_actual["fragment_isolation_holdout"],
        }
        assert _packet_fragment_ids(public_packet).isdisjoint(denied_or_holdout)
        public_bytes = public_packet.canonical_bytes.decode("utf-8")
        assert canaries["excluded"] not in public_bytes
        assert canaries["holdout"] not in public_bytes

        cross_private_request = _request(
            question=canaries["private"],
            repository=private.repository,
            collection_ids=private.collection_ids,
            snapshot=private.snapshot,
            policy_id=private.policy_id,
        )
        with pytest.raises(RecallSnapshotError):
            public.service.recall(cross_private_request)
        cross_public_request = _request(
            question=canaries["public"],
            repository=public.repository,
            collection_ids=public.collection_ids,
            snapshot=public.snapshot,
            policy_id=public.policy_id,
        )
        with pytest.raises(RecallSnapshotError):
            private.service.recall(cross_public_request)
    finally:
        for runtime in runtimes.values():
            runtime.repository.close()
