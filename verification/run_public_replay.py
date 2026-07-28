#!/usr/bin/env python3
"""Run the deterministic 1,000-fragment public verification replay."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter_ns
from typing import Any, cast

from dithyramba._version import __version__
from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex, sha256_hex
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import ProcessingRunStatus, SQLiteRecallBackend, initialize_library
from dithyramba.recall import (
    QueryRequest,
    RecallService,
    RetrievalBudget,
    current_fts_runtime_profile,
)
from verification.generate_synthetic_1000 import build_payload, check_manifest, payload_hash

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFESTS = (
    "caillebotte_regression/manifest.json",
    "public_multilingual/manifest.json",
    "synthetic_isolation/manifest.json",
)
LIBRARY_ID = "library_33333333333343338333333333333333"
COLLECTION_ID = "collection_44444444444444448444444444444444"
POLICY_ID = "policy_public_replay"
QUERY_ID = "query_verify_0500"


def _load_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value: object = json.loads(raw)
    if type(value) is not dict:
        raise ValueError(f"canonical verification input must be an object: {path}")
    result = cast(dict[str, Any], value)
    if raw != canonical_json_bytes(result) + b"\n":
        raise ValueError(f"verification input is not canonical JSON plus LF: {path}")
    return result


def _fixture_hashes() -> dict[str, str]:
    return {
        relative: canonical_sha256_hex(_load_canonical(ROOT / "fixtures" / relative))
        for relative in FIXTURE_MANIFESTS
    }


def _cpu_name() -> str:
    try:
        result = subprocess.run(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return platform.processor() or "unknown"
    value = result.stdout.strip()
    return value or platform.processor() or "unknown"


def _physical_memory_bytes() -> int | None:
    """Return host RAM without adding a runtime dependency."""

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ("sysctl", "-n", "hw.memsize"),
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            value = int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        return value if value > 0 else None
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):
        return None
    value = pages * page_size
    return value if value > 0 else None


def _power_source() -> str:
    """Record the coarse macOS power source used for the performance run."""

    if platform.system() != "Darwin":
        return "unknown"
    try:
        result = subprocess.run(
            ("pmset", "-g", "batt"),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    first = next((line.casefold() for line in result.stdout.splitlines() if line.strip()), "")
    if "ac power" in first:
        return "ac"
    if "battery power" in first:
        return "battery"
    return "unknown"


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the BSD-compatible Python docs use KiB
    # on the other supported local development hosts.
    return value if platform.system() == "Darwin" else value * 1024


def _nearest_rank(values: list[int], numerator: int, denominator: int) -> int:
    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(values)
    rank = (len(ordered) * numerator + denominator - 1) // denominator
    return ordered[max(rank - 1, 0)]


def _microseconds(nanoseconds: int) -> int:
    return (nanoseconds + 500) // 1_000


def _required_rows(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    value = payload.get(key)
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise ValueError(f"generated {key} must be an object array")
    return cast(list[dict[str, object]], value)


def _required_text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if type(value) is not str or not value:
        raise ValueError(f"generated {key} must be non-empty text")
    return value


def run_verification(
    *,
    warmups: int = 20,
    measured: int = 100,
) -> dict[str, object]:
    """Build the complete local pipeline and return one canonical result payload."""

    if type(warmups) is not int or warmups < 1:
        raise ValueError("warmups must be a positive integer")
    if type(measured) is not int or measured < 1:
        raise ValueError("measured must be a positive integer")
    cpu = _cpu_name()

    payload = build_payload()
    generated_hash = payload_hash(payload)
    if check_manifest(payload) != generated_hash:
        raise RuntimeError("generated verification payload failed its frozen manifest")
    fragments = _required_rows(payload, "fragments")
    queries = _required_rows(payload, "queries")
    if len(fragments) != 1_000:
        raise RuntimeError("public replay requires exactly 1,000 generated fragments")
    query_row = next(row for row in queries if _required_text(row, "query_id") == QUERY_ID)
    expected_fragment = fragments[500]
    expected_text_hash = _required_text(expected_fragment, "text_sha256")
    profile = current_fts_runtime_profile()

    setup_started = perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="dithyramba-public-replay-") as temporary:
        work = Path(temporary)
        source_root = work / "sources"
        source_root.mkdir()
        for index, fragment in enumerate(fragments):
            (source_root / f"fragment_{index:04d}.txt").write_text(
                _required_text(fragment, "text"),
                encoding="utf-8",
            )

        repository = initialize_library(
            LibraryConfig(library_id=LIBRARY_ID, name="Public 1000-fragment replay"),
            data_root=work / "data",
        )
        try:
            collection = repository.create_collection(
                CollectionConfig(
                    collection_id=COLLECTION_ID,
                    library_id=LIBRARY_ID,
                    name="Synthetic verification corpus",
                    kind=CollectionKind.CORPUS,
                    roots=(
                        build_collection_root(
                            source_root,
                            data_root=repository.paths.application_data_root,
                        ),
                    ),
                )
            )
            ingest = IngestService(repository).ingest_collection(collection.config.collection_id)
            if ingest.run.status is not ProcessingRunStatus.SUCCEEDED:
                raise RuntimeError("verification ingest did not succeed")
            if len(ingest.outcomes) != 1_000 or any(
                outcome.failure_code is not None for outcome in ingest.outcomes
            ):
                raise RuntimeError("verification ingest did not process all 1,000 inputs")
            source_records = repository.list_sources(collection_id=collection.config.collection_id)
            stored_fragment_count = sum(
                len(repository.list_source_fragments(source.current_source_version_id))
                for source in source_records
            )
            if stored_fragment_count != 1_000:
                raise RuntimeError(
                    f"verification stored {stored_fragment_count} fragments instead of 1,000"
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
            repository.persist_access_policy(name="Public verification", snapshot=policy)
            request = QueryRequest(
                question=_required_text(query_row, "text"),
                library_id=LIBRARY_ID,
                collection_ids=(collection.config.collection_id,),
                corpus_snapshot_id=snapshot.corpus_snapshot_id,
                access_policy_id=POLICY_ID,
                purpose="verification",
                retrieval=RetrievalBudget(max_candidates=10, max_source_fragments=10),
            )
            service = RecallService(
                SQLiteRecallBackend(repository),
                profile_version=profile.profile_version,
            )
            setup_elapsed_ns = perf_counter_ns() - setup_started

            packet_hashes: set[str] = set()

            def recall_once() -> None:
                packet = service.recall(request).packet
                if len(packet.read_receipt.items) != 1_000:
                    raise RuntimeError("warm recall did not read all 1,000 permitted fragments")
                if packet.retrieval_receipt.candidate_count != 1:
                    raise RuntimeError("unique verification token did not produce one candidate")
                if (
                    len(packet.source_fragments) != 1
                    or packet.source_fragments[0].text_sha256 != expected_text_hash
                ):
                    raise RuntimeError("verification query selected the wrong evidence fragment")
                packet_hashes.add(packet.packet_hash)

            for _ in range(warmups):
                recall_once()

            samples_ns: list[int] = []
            for _ in range(measured):
                started = perf_counter_ns()
                recall_once()
                samples_ns.append(perf_counter_ns() - started)

            if len(packet_hashes) != 1:
                raise RuntimeError("identical warm recalls produced different EvidencePackets")
            connection = repository._store.connection
            persisted = {
                "evidence_packets": int(
                    connection.execute("SELECT COUNT(*) FROM evidence_packets").fetchone()[0]
                ),
                "query_requests": int(
                    connection.execute("SELECT COUNT(*) FROM query_requests").fetchone()[0]
                ),
                "recall_run_artifacts": int(
                    connection.execute("SELECT COUNT(*) FROM recall_run_artifacts").fetchone()[0]
                ),
                "recall_runs": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM processing_runs WHERE kind = 'recall'"
                    ).fetchone()[0]
                ),
            }
            compile_options = sorted(
                str(row[0]) for row in connection.execute("PRAGMA compile_options").fetchall()
            )
            database_path = repository.paths.database
            wal_path = database_path.with_name(f"{database_path.name}-wal")
            storage_bytes = {
                "database": database_path.stat().st_size,
                "wal": wal_path.stat().st_size if wal_path.exists() else 0,
            }
        finally:
            repository.close()

    verification_manifest = _load_canonical(ROOT / "verification" / "synthetic_1000_manifest.json")
    return {
        "schema": "dithyramba.public_replay_result/1.0",
        "verification_id": "public_replay_1000_v1",
        "measurement_scope": (
            "QueryRequest persistence and snapshot load; policy compilation; protected "
            "1,000-fragment read; fresh ephemeral FTS; receipts and EvidencePacket; "
            "atomic artifact persistence or reuse"
        ),
        "workload": {
            "fragment_count": 1_000,
            "measured_iterations": measured,
            "query_id": QUERY_ID,
            "warmup_iterations": warmups,
        },
        "latency_microseconds": {
            "max": _microseconds(max(samples_ns)),
            "min": _microseconds(min(samples_ns)),
            "p50_nearest_rank": _microseconds(_nearest_rank(samples_ns, 50, 100)),
            "p95_nearest_rank": _microseconds(_nearest_rank(samples_ns, 95, 100)),
            "samples_sha256": canonical_sha256_hex(
                {
                    "schema": "dithyramba.public_replay_samples/1.0",
                    "samples_nanoseconds": samples_ns,
                }
            ),
            "samples_nanoseconds": samples_ns,
        },
        "setup_microseconds": _microseconds(setup_elapsed_ns),
        "result": {
            "candidate_count": 1,
            "packet_hash_count": len(packet_hashes),
            "packet_hash": next(iter(packet_hashes)),
            "provider_call_count": 0,
            "persisted_counts": persisted,
            "selected_text_sha256": expected_text_hash,
            "storage_bytes": storage_bytes,
        },
        "frozen_hashes": {
            "verification_generator_manifest_canonical_sha256": canonical_sha256_hex(
                verification_manifest
            ),
            "verification_generator_script_sha256": sha256_hex(
                (ROOT / "verification" / "generate_synthetic_1000.py").read_bytes()
            ),
            "verification_payload_canonical_sha256": generated_hash,
            "dependency_lock_sha256": sha256_hex((ROOT / "uv.lock").read_bytes()),
            "fixture_manifest_canonical_sha256": _fixture_hashes(),
            "runner_script_sha256": sha256_hex(Path(__file__).read_bytes()),
        },
        "runtime": {
            "cpu": cpu,
            "dithyramba_version": __version__,
            "fts_profile_version": profile.profile_version,
            "machine": platform.machine(),
            "os": platform.system(),
            "os_version": platform.release(),
            "peak_rss_bytes": _peak_rss_bytes(),
            "physical_memory_bytes": _physical_memory_bytes(),
            "power_source": _power_source(),
            "python_version": platform.python_version(),
            "sqlite_compile_options": compile_options,
            "sqlite_version": sqlite3.sqlite_version,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_verification(
        warmups=args.warmups,
        measured=args.measured,
    )
    encoded = canonical_json_bytes(result) + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
