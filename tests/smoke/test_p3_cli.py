"""P3 snapshot, recall, packet inspection, and replay CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from typer.testing import CliRunner, Result

from dithyramba.cli import app

runner = CliRunner()


def _json(arguments: list[str], *, exit_code: int = 0) -> tuple[dict[str, object], Result]:
    result = runner.invoke(app, [*arguments, "--json"])
    assert result.exit_code == exit_code, result.output
    payload = cast(dict[str, object], json.loads(result.stdout))
    assert result.stdout.strip() == json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload, result


def test_cli_recall_packet_inspect_receipt_and_replay_are_canonical(tmp_path: Path) -> None:
    data_home = tmp_path / "data"
    source_root = tmp_path / "corpus"
    source_root.mkdir()
    source_text = "# Memory\n\nDithyramba preserves exact cited evidence.\n"
    (source_root / "note.md").write_text(source_text, encoding="utf-8")

    library, _ = _json(
        [
            "library",
            "init",
            "P3 CLI",
            "--data-home",
            str(data_home),
            "--source-root",
            str(source_root),
        ]
    )
    library_id = cast(str, library["library_id"])
    collection, _ = _json(
        [
            "collection",
            "add",
            "Corpus",
            "--library",
            library_id,
            "--root",
            str(source_root),
            "--kind",
            "corpus",
            "--data-home",
            str(data_home),
        ]
    )
    collection_id = cast(str, collection["collection_id"])
    policy, _ = _json(
        [
            "access-policy",
            "create",
            "Local research",
            "--library",
            library_id,
            "--purpose",
            "research",
            "--allow-collection",
            collection_id,
            "--data-home",
            str(data_home),
        ]
    )
    policy_id = cast(str, policy["access_policy_id"])
    indexed, _ = _json(
        [
            "index",
            "--library",
            library_id,
            "--collection",
            collection_id,
            "--data-home",
            str(data_home),
        ]
    )
    assert cast(dict[str, object], indexed["coverage"])["processed_count"] == 1
    snapshot, _ = _json(
        [
            "collection",
            "freeze",
            "--library",
            library_id,
            "--collection",
            collection_id,
            "--data-home",
            str(data_home),
        ]
    )
    snapshot_id = cast(str, snapshot["corpus_snapshot_id"])

    recalled, _ = _json(
        [
            "recall",
            "What preserves exact cited evidence?",
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
        ]
    )
    assert recalled["schema"] == "dithyramba.recall_result/1.0"
    run = cast(dict[str, object], recalled["run"])
    packet = cast(dict[str, object], recalled["packet"])
    assert run["status"] == "succeeded"
    assert run["output_hash"] == packet["packet_hash"]
    assert packet["result_status"] == "evidence_found"
    packet_id = cast(str, packet["evidence_packet_id"])
    packet_hash = cast(str, packet["packet_hash"])
    fragments = cast(list[dict[str, object]], packet["source_fragments"])
    assert cast(str, fragments[0]["text"]) in source_text

    inspected, _ = _json(
        [
            "packet",
            "inspect",
            packet_id,
            "--library",
            library_id,
            "--data-home",
            str(data_home),
        ]
    )
    assert inspected == packet

    receipt, _ = _json(
        [
            "packet",
            "read-receipt",
            packet_id,
            "--library",
            library_id,
            "--data-home",
            str(data_home),
        ]
    )
    assert receipt["schema"] == "dithyramba.read_receipt/1.0"
    assert len(cast(list[object], receipt["items"])) == 2

    replayed, _ = _json(
        [
            "packet",
            "replay",
            packet_id,
            "--library",
            library_id,
            "--data-home",
            str(data_home),
        ]
    )
    replay_packet = cast(dict[str, object], replayed["packet"])
    assert replay_packet == packet
    assert replay_packet["packet_hash"] == packet_hash
    assert cast(dict[str, object], replayed["run"])["processing_run_id"] != run["processing_run_id"]


def test_cli_separate_primary_and_research_scopes_stay_distinct_and_replayable(
    tmp_path: Path,
) -> None:
    data_home = tmp_path / "data"
    primary_root = tmp_path / "primary"
    research_root = tmp_path / "research"
    primary_root.mkdir()
    research_root.mkdir()
    (primary_root / "catalogue.md").write_text(
        "# Catalogue\n\nPrimary address evidence: Singerstrasse 849.\n",
        encoding="utf-8",
    )
    (research_root / "dossier.md").write_text(
        "# Dossier\n\nResearch interpretation: address evidence may identify one complex.\n",
        encoding="utf-8",
    )

    library, _ = _json(
        [
            "library",
            "init",
            "Separated evidence scopes",
            "--data-home",
            str(data_home),
            "--source-root",
            str(primary_root),
            "--source-root",
            str(research_root),
        ]
    )
    library_id = cast(str, library["library_id"])

    collection_ids: list[str] = []
    snapshot_ids: list[str] = []
    for name, root in (
        ("Primary sources", primary_root),
        ("Research context", research_root),
    ):
        collection, _ = _json(
            [
                "collection",
                "add",
                name,
                "--library",
                library_id,
                "--root",
                str(root),
                "--kind",
                "corpus",
                "--data-home",
                str(data_home),
            ]
        )
        collection_id = cast(str, collection["collection_id"])
        collection_ids.append(collection_id)
        indexed, _ = _json(
            [
                "index",
                "--library",
                library_id,
                "--collection",
                collection_id,
                "--data-home",
                str(data_home),
            ]
        )
        assert cast(dict[str, object], indexed["coverage"])["processed_count"] == 1
        snapshot, _ = _json(
            [
                "collection",
                "freeze",
                "--library",
                library_id,
                "--collection",
                collection_id,
                "--data-home",
                str(data_home),
            ]
        )
        snapshot_ids.append(cast(str, snapshot["corpus_snapshot_id"]))

    policy, _ = _json(
        [
            "access-policy",
            "create",
            "Local paired research",
            "--library",
            library_id,
            "--purpose",
            "research",
            "--allow-collection",
            collection_ids[0],
            "--allow-collection",
            collection_ids[1],
            "--data-home",
            str(data_home),
        ]
    )
    policy_id = cast(str, policy["access_policy_id"])

    packets: list[dict[str, object]] = []
    for collection_id, snapshot_id in zip(collection_ids, snapshot_ids, strict=True):
        recalled, _ = _json(
            [
                "recall",
                "What does the address evidence support?",
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
            ]
        )
        packet = cast(dict[str, object], recalled["packet"])
        packets.append(packet)
        assert packet["corpus_snapshot_id"] == snapshot_id
        assert packet["result_status"] == "evidence_found"

    primary_fragments = cast(list[dict[str, object]], packets[0]["source_fragments"])
    research_fragments = cast(list[dict[str, object]], packets[1]["source_fragments"])
    assert primary_fragments
    assert research_fragments
    assert all(
        "Research interpretation" not in cast(str, item["text"]) for item in primary_fragments
    )
    assert all(
        "Primary address evidence" not in cast(str, item["text"]) for item in research_fragments
    )
    assert packets[0]["evidence_packet_id"] != packets[1]["evidence_packet_id"]
    assert packets[0]["packet_hash"] != packets[1]["packet_hash"]
    assert packets[0]["query_request_hash"] != packets[1]["query_request_hash"]

    for packet in packets:
        replayed, _ = _json(
            [
                "packet",
                "replay",
                cast(str, packet["evidence_packet_id"]),
                "--library",
                library_id,
                "--data-home",
                str(data_home),
            ]
        )
        assert replayed["packet"] == packet

    mismatch = runner.invoke(
        app,
        [
            "recall",
            "What does the address evidence support?",
            "--library",
            library_id,
            "--collection",
            collection_ids[0],
            "--snapshot",
            snapshot_ids[1],
            "--access-policy",
            policy_id,
            "--purpose",
            "research",
            "--data-home",
            str(data_home),
        ],
    )
    assert mismatch.exit_code == 1
    assert "CorpusSnapshot Collection scope differs from QueryRequest" in mismatch.output


def test_cli_recall_requires_every_explicit_scope_component() -> None:
    result = runner.invoke(app, ["recall", "question"])

    assert result.exit_code == 2
    assert "Missing option" in result.output
