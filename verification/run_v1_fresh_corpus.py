#!/usr/bin/env python3
"""Exercise the v1 path on a newly generated 400-source corpus and 30 questions."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import canonical_json_bytes
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import ProcessingRunStatus, SQLiteRecallBackend, initialize_library
from dithyramba.provenance import IngestDisposition
from dithyramba.recall import (
    QueryRequest,
    RecallService,
    RetrievalBudget,
    current_fts_runtime_profile,
)

SOURCE_COUNT = 400
QUESTION_COUNT = 30
LIBRARY_ID = "library_aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa"
COLLECTION_ID = "collection_bbbbbbbbbbbb4bbb8bbbbbbbbbbbbbbb"
POLICY_ID = "policy_fresh_v1_acceptance"


def _source_text(index: int) -> str:
    key = f"lenskey{index:03d}"
    return (
        f"# Fresh research record {index:03d}\n\n"
        "## Observation\n\n"
        f"Record {key} links the archive threshold to observation cycle {index:03d}. "
        f"Its distinctive finding is prism value {index * 17 + 11}.\n\n"
        "## Context\n\n"
        "This synthetic note shares archive, threshold, observation, and evidence vocabulary "
        "with every other record, so the unique record key must disambiguate it.\n\n"
        "## Qualification\n\n"
        "The note is generated verification material, not a claim about the world.\n"
    )


def _question_rows() -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            f"Which observation connects the archive threshold in record lenskey{index:03d}?",
            index,
        )
        for index in ((question * 13 + 7) % SOURCE_COUNT for question in range(QUESTION_COUNT))
    )


def run_acceptance() -> dict[str, object]:
    """Return a compact, machine-readable acceptance receipt."""

    with tempfile.TemporaryDirectory(prefix="dithyramba-v1-fresh-") as temporary:
        root = Path(temporary)
        source_root = root / "sources"
        source_root.mkdir()
        for index in range(SOURCE_COUNT):
            (source_root / f"record_{index:03d}.md").write_text(
                _source_text(index), encoding="utf-8"
            )

        repository = initialize_library(
            LibraryConfig(library_id=LIBRARY_ID, name="Fresh v1 acceptance"),
            data_root=root / "data",
        )
        try:
            collection = repository.create_collection(
                CollectionConfig(
                    collection_id=COLLECTION_ID,
                    library_id=LIBRARY_ID,
                    name="Fresh synthetic corpus",
                    kind=CollectionKind.CORPUS,
                    roots=(
                        build_collection_root(
                            source_root,
                            data_root=repository.paths.application_data_root,
                        ),
                    ),
                )
            )
            ingest_service = IngestService(repository)
            cold_started = perf_counter()
            cold_ingest = ingest_service.ingest_collection(collection.config.collection_id)
            cold_seconds = perf_counter() - cold_started
            if cold_ingest.run.status is not ProcessingRunStatus.SUCCEEDED:
                raise RuntimeError("fresh-corpus ingest failed")
            if len(cold_ingest.outcomes) != SOURCE_COUNT or any(
                outcome.disposition is not IngestDisposition.ADDED
                for outcome in cold_ingest.outcomes
            ):
                raise RuntimeError("fresh-corpus ingest did not add every source")

            warm_started = perf_counter()
            warm_ingest = ingest_service.ingest_collection(collection.config.collection_id)
            warm_seconds = perf_counter() - warm_started
            if warm_ingest.run.status is not ProcessingRunStatus.SUCCEEDED or any(
                outcome.disposition is not IngestDisposition.UNCHANGED
                for outcome in warm_ingest.outcomes
            ):
                raise RuntimeError("unchanged ingest did not reuse every source")

            versions_by_index = {
                int(Path(outcome.relative_path).stem.removeprefix("record_")): (
                    outcome.source_version_id
                )
                for outcome in cold_ingest.outcomes
                if outcome.relative_path is not None and outcome.source_version_id is not None
            }
            sources = repository.list_sources(collection_id=collection.config.collection_id)
            if len(sources) != SOURCE_COUNT:
                raise RuntimeError("fresh-corpus source count is inconsistent")
            if len(versions_by_index) != SOURCE_COUNT:
                raise RuntimeError("fresh-corpus version map is inconsistent")
            fragment_count = sum(
                len(repository.list_source_fragments(source.current_source_version_id))
                for source in sources
            )
            snapshot = repository.freeze_snapshot((collection.config.collection_id,))
            policy = AccessPolicySnapshot(
                access_policy_id=POLICY_ID,
                library_id=LIBRARY_ID,
                allowed_purposes=("verification",),
                collection_rules=(
                    CollectionRule(collection.config.collection_id, PolicyEffect.ALLOW),
                ),
            )
            repository.persist_access_policy(name="Fresh v1 acceptance", snapshot=policy)
            service = RecallService(
                SQLiteRecallBackend(repository),
                profile_version=current_fts_runtime_profile().profile_version,
            )

            top1_hits = 0
            selected_hits = 0
            replay_hits = 0
            address_hits = 0
            latencies: list[float] = []
            packet_ids: list[str] = []
            for question, expected_index in _question_rows():
                request = QueryRequest(
                    question=question,
                    library_id=LIBRARY_ID,
                    collection_ids=(collection.config.collection_id,),
                    corpus_snapshot_id=snapshot.corpus_snapshot_id,
                    access_policy_id=POLICY_ID,
                    purpose="verification",
                    retrieval=RetrievalBudget(max_candidates=20, max_source_fragments=10),
                )
                started = perf_counter()
                packet = service.recall(request).packet
                latencies.append(perf_counter() - started)
                expected_version = versions_by_index[expected_index]
                selected_versions = tuple(
                    fragment.source_version_id for fragment in packet.source_fragments
                )
                if selected_versions and selected_versions[0] == expected_version:
                    top1_hits += 1
                if expected_version in selected_versions:
                    selected_hits += 1
                if any(
                    getattr(fragment.source_address, "line_start", 0) >= 1
                    for fragment in packet.source_fragments
                    if fragment.source_version_id == expected_version
                ):
                    address_hits += 1
                replayed = service.verify_replay(packet.evidence_packet_id)
                if replayed.canonical_bytes == packet.canonical_bytes:
                    replay_hits += 1
                packet_ids.append(packet.evidence_packet_id)

            if (top1_hits, selected_hits, replay_hits, address_hits) != (
                QUESTION_COUNT,
                QUESTION_COUNT,
                QUESTION_COUNT,
                QUESTION_COUNT,
            ):
                raise RuntimeError("fresh-corpus evidence or replay acceptance failed")
            database_bytes = repository.paths.database.stat().st_size
        finally:
            repository.close()

    return {
        "schema": "dithyramba.v1_fresh_corpus_acceptance/1.0",
        "workload": {
            "source_count": SOURCE_COUNT,
            "fragment_count": fragment_count,
            "question_count": QUESTION_COUNT,
            "corpus_origin": "repository-authored synthetic data",
            "model_calls": 0,
            "model_tokens": 0,
        },
        "ingest": {
            "cold_microseconds": round(cold_seconds * 1_000_000),
            "unchanged_microseconds": round(warm_seconds * 1_000_000),
            "unchanged_source_count": SOURCE_COUNT,
        },
        "recall": {
            "top1_document_hits": top1_hits,
            "selected_document_hits": selected_hits,
            "source_address_hits": address_hits,
            "exact_replay_hits": replay_hits,
            "median_microseconds": round(median(latencies) * 1_000_000),
            "total_microseconds": round(sum(latencies) * 1_000_000),
            "packet_count": len(set(packet_ids)),
        },
        "storage": {"database_bytes": database_bytes},
        "claim_boundary": (
            "This acceptance proves deterministic lexical retrieval, source coordinates, "
            "unchanged-source reuse, and exact replay on fresh synthetic inputs. It does not "
            "prove semantic research quality or truth of external sources."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_acceptance()
    encoded = canonical_json_bytes(payload) + b"\n"
    if args.output is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
