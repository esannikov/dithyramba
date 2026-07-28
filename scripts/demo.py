"""Build and replay one tiny, local Dithyramba memory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
CORPUS = (ROOT / "examples" / "demo-corpus").resolve()


def _command_json(*arguments: str) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-m", "dithyramba", *arguments, "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"Dithyramba command failed: {detail}")
    return cast(dict[str, Any], json.loads(process.stdout))


def _identifier(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Expected non-empty {key} in command result")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create, query, inspect, and replay a small synthetic Library."
    )
    parser.add_argument(
        "--data-home",
        type=Path,
        help="Optional empty application-data root. Defaults to a new temporary directory.",
    )
    args = parser.parse_args()

    data_home = (
        args.data_home.expanduser().resolve()
        if args.data_home is not None
        else Path(tempfile.mkdtemp(prefix="dithyramba-demo-")).resolve()
    )
    data_home.mkdir(parents=True, exist_ok=True)

    print("Dithyramba demo")
    print(f"  corpus:    {CORPUS}")
    print(f"  data home: {data_home}")

    library = _command_json(
        "library",
        "init",
        "Dithyramba demo",
        "--data-home",
        str(data_home),
        "--source-root",
        str(CORPUS),
    )
    library_id = _identifier(library, "library_id")
    print("✓ Library created")

    doctor = _command_json(
        "doctor",
        "--library",
        library_id,
        "--data-home",
        str(data_home),
    )
    if doctor.get("status") != "ok" or doctor.get("external_services_contacted") is not False:
        raise RuntimeError("Library doctor did not confirm a local healthy runtime")
    print("✓ SQLite, schema, and local-only runtime verified")

    collection = _command_json(
        "collection",
        "add",
        "Synthetic field notes",
        "--library",
        library_id,
        "--root",
        str(CORPUS),
        "--kind",
        "corpus",
        "--data-home",
        str(data_home),
    )
    collection_id = _identifier(collection, "collection_id")
    print("✓ Collection scoped")

    policy = _command_json(
        "access-policy",
        "create",
        "Local demo research",
        "--library",
        library_id,
        "--purpose",
        "research",
        "--allow-collection",
        collection_id,
        "--data-home",
        str(data_home),
    )
    policy_id = _identifier(policy, "access_policy_id")
    print("✓ Default-deny policy compiled")

    indexed = _command_json(
        "index",
        "--library",
        library_id,
        "--collection",
        collection_id,
        "--data-home",
        str(data_home),
    )
    coverage = indexed.get("coverage")
    processed = coverage.get("processed_count") if isinstance(coverage, dict) else None
    print(f"✓ Sources indexed ({processed} processed)")

    snapshot = _command_json(
        "collection",
        "freeze",
        "--library",
        library_id,
        "--collection",
        collection_id,
        "--data-home",
        str(data_home),
    )
    snapshot_id = _identifier(snapshot, "corpus_snapshot_id")
    print("✓ Immutable snapshot frozen")

    recalled = _command_json(
        "recall",
        "Which note records the observation time and authorship uncertainty?",
        "--library",
        library_id,
        "--collection",
        collection_id,
        "--snapshot",
        snapshot_id,
        "--access-policy",
        policy_id,
        "--purpose",
        "research",
        "--max-candidates",
        "10",
        "--max-source-fragments",
        "5",
        "--data-home",
        str(data_home),
    )
    packet = recalled.get("packet")
    if not isinstance(packet, dict):
        raise RuntimeError("Recall result did not contain a packet")
    packet_id = _identifier(packet, "evidence_packet_id")
    packet_hash = _identifier(packet, "packet_hash")
    fragments = packet.get("source_fragments")
    fragment_count = len(fragments) if isinstance(fragments, list) else 0
    print(f"✓ Evidence packet persisted ({fragment_count} fragment(s))")

    inspected = _command_json(
        "packet",
        "inspect",
        packet_id,
        "--library",
        library_id,
        "--data-home",
        str(data_home),
    )
    if inspected != packet:
        raise RuntimeError("Inspected packet differs from the recall result")
    print("✓ Packet inspected and revalidated")

    replayed = _command_json(
        "packet",
        "replay",
        packet_id,
        "--library",
        library_id,
        "--data-home",
        str(data_home),
    )
    replay_packet = replayed.get("packet")
    if not isinstance(replay_packet, dict) or replay_packet.get("packet_hash") != packet_hash:
        raise RuntimeError("Cold replay did not reproduce the packet hash")
    print("✓ Exact packet hash reproduced")

    backup_root = data_home.parent / f"{data_home.name}-backups"
    restored_home = data_home.parent / f"{data_home.name}-restored"
    backup_root.mkdir(parents=True, exist_ok=False)
    backup_root.chmod(0o700)
    backup = _command_json(
        "backup",
        "--library",
        library_id,
        "--data-home",
        str(data_home),
        "--output-directory",
        str(backup_root),
    )
    bundle_path = _identifier(backup, "path")
    restored = _command_json(
        "restore",
        bundle_path,
        "--data-home",
        str(restored_home),
        "--source-root",
        str(CORPUS),
    )
    if restored.get("status") != "verified" or restored.get("library_id") != library_id:
        raise RuntimeError("Backup restore did not verify the original Library")
    restored_doctor = _command_json(
        "doctor",
        "--library",
        library_id,
        "--data-home",
        str(restored_home),
    )
    if restored_doctor.get("status") != "ok":
        raise RuntimeError("Restored Library doctor did not report a healthy runtime")
    print("✓ Backup restored into a separate verified data root")

    print("\nDemo result")
    print(f"  library_id:         {library_id}")
    print(f"  collection_id:      {collection_id}")
    print(f"  corpus_snapshot_id: {snapshot_id}")
    print(f"  access_policy_id:   {policy_id}")
    print(f"  evidence_packet_id: {packet_id}")
    print(f"  packet_hash:        {packet_hash}")
    print(f"  data_home:          {data_home}")
    print(f"  restored_data_home: {restored_home}")
    print("\nThe temporary Library is intentionally retained for inspection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
