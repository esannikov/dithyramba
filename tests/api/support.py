"""Build small, real persisted Libraries for loopback API tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.api import bearer_token_for, create_app
from dithyramba.collections import (
    CollectionConfig,
    CollectionKind,
    build_collection_root,
)
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import SQLiteRecallBackend, initialize_library
from dithyramba.recall import QueryRequest, RecallService, search_ephemeral_fts


@dataclass(frozen=True, slots=True)
class PreparedLibrary:
    library_id: str
    library_name: str
    collection_id: str
    collection_root_id: str
    corpus_snapshot_id: str
    access_policy_id: str
    access_policy_hash: str
    query_payload: dict[str, object]
    evidence_packet_id: str
    packet_hash: str
    packet_payload: dict[str, object]
    read_receipt_payload: dict[str, object]
    denied_evidence_packet_id: str
    denied_packet_hash: str
    source_id: str
    source_family_id: str
    root_source_id: str
    family_role: str
    source_version_id: str
    source_fragment_id: str
    text: str
    text_sha256: str
    source_address: dict[str, object]


@dataclass(frozen=True, slots=True)
class ApiWorld:
    data_home: Path
    primary: PreparedLibrary
    foreign: PreparedLibrary


def build_api_world(tmp_path: Path) -> ApiWorld:
    data_home = tmp_path / "data"
    primary = build_prepared_library(
        data_home=data_home,
        source_root=tmp_path / "primary-source",
        name="Primary API library",
        text=(
            "# Oracular evidence\n\n"
            "The oracular fragment is packet-backed. "
            "<script>window.pwned = true</script> "
            "<img src=x onerror=window.pwned=true>\n"
        ),
    )
    foreign = build_prepared_library(
        data_home=data_home,
        source_root=tmp_path / "foreign-source",
        name="Foreign API library",
        text="# Foreign evidence\n\nA foreign oracular fragment must remain undisclosed.\n",
    )
    return ApiWorld(data_home=data_home, primary=primary, foreign=foreign)


def build_prepared_library(
    *,
    data_home: Path,
    source_root: Path,
    name: str,
    text: str,
) -> PreparedLibrary:
    source_root.mkdir(parents=True)
    (source_root / "evidence.md").write_text(text, encoding="utf-8")
    repository = initialize_library(LibraryConfig(name=name), data_root=data_home)
    try:
        collection = repository.create_collection(
            CollectionConfig(
                library_id=repository.library_id,
                name="Evidence corpus",
                kind=CollectionKind.CORPUS,
                roots=(
                    build_collection_root(
                        source_root,
                        data_root=data_home,
                    ),
                ),
            )
        )
        outcome = (
            IngestService(repository)
            .ingest_path(
                collection.config.collection_id,
                "evidence.md",
            )
            .outcomes[0]
        )
        if (
            outcome.source_id is None
            or outcome.source_version_id is None
            or outcome.source_family_id is None
            or outcome.root_source_id is None
            or outcome.disposition is None
        ):
            raise AssertionError("test source did not ingest")
        snapshot = repository.freeze_snapshot((collection.config.collection_id,))
        policy = AccessPolicySnapshot(
            access_policy_id="policy_research",
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),),
            allow_export=True,
        )
        repository.persist_access_policy(name="Research", snapshot=policy)
        request = QueryRequest(
            question="oracular",
            library_id=repository.library_id,
            collection_ids=(collection.config.collection_id,),
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            purpose="research",
        )
        profile_version = search_ephemeral_fts(
            question="profile",
            fragments=(),
            max_candidates=1,
        ).profile_version
        recall = RecallService(
            SQLiteRecallBackend(repository),
            profile_version=profile_version,
        )
        result = recall.recall(request)
        if not result.packet.source_fragments:
            raise AssertionError("test recall must return at least one source fragment")
        fragment = max(result.packet.source_fragments, key=lambda item: len(item.text))

        denied_policy = AccessPolicySnapshot(
            access_policy_id="policy_no_export",
            library_id=repository.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),),
            allow_export=False,
        )
        repository.persist_access_policy(name="No export", snapshot=denied_policy)
        denied = recall.recall(
            request.model_copy(update={"access_policy_id": denied_policy.access_policy_id})
        )
        read_receipt = result.packet.read_receipt
        return PreparedLibrary(
            library_id=repository.library_id,
            library_name=name,
            collection_id=collection.config.collection_id,
            collection_root_id=collection.collection_root_ids[0],
            corpus_snapshot_id=snapshot.corpus_snapshot_id,
            access_policy_id=policy.access_policy_id,
            access_policy_hash=policy.policy_hash,
            query_payload=request.semantic_payload(),
            evidence_packet_id=result.packet.evidence_packet_id,
            packet_hash=result.packet.packet_hash,
            packet_payload=result.packet.payload(),
            read_receipt_payload={
                **read_receipt.semantic_payload(),
                "read_receipt_id": read_receipt.read_receipt_id,
                "receipt_hash": read_receipt.receipt_hash,
            },
            denied_evidence_packet_id=denied.packet.evidence_packet_id,
            denied_packet_hash=denied.packet.packet_hash,
            source_id=outcome.source_id,
            source_family_id=outcome.source_family_id,
            root_source_id=outcome.root_source_id,
            family_role="root",
            source_version_id=fragment.source_version_id,
            source_fragment_id=fragment.source_fragment_id,
            text=fragment.text,
            text_sha256=fragment.text_sha256,
            source_address=fragment.source_address.payload(),
        )
    finally:
        repository.close()


def make_test_app(
    world: ApiWorld,
    *,
    max_request_body_bytes: int = 64 * 1024,
) -> tuple[FastAPI, str]:
    app = create_app(
        library_id=world.primary.library_id,
        data_home=world.data_home,
        allowed_origin="http://testserver",
        test_only_allow_testserver=True,
        max_request_body_bytes=max_request_body_bytes,
    )
    return app, bearer_token_for(app)


def mutation_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Origin": "http://testserver",
    }
