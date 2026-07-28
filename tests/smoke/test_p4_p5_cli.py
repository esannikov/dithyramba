"""P4 review and P5 complete BackupBundle CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from typer.testing import CliRunner, Result

from dithyramba.access import AccessPolicySnapshot, CollectionRule, PolicyEffect
from dithyramba.cli import app
from dithyramba.collections import CollectionConfig, CollectionKind, build_collection_root
from dithyramba.ingest.service import IngestService
from dithyramba.library import LibraryConfig
from dithyramba.persistence import SQLiteRecallBackend, initialize_library, open_library
from dithyramba.recall import QueryRequest, RecallService, current_fts_runtime_profile
from dithyramba.review import ReviewTargetType
from dithyramba.review.service import ReviewService

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


def _packet_and_target(
    tmp_path: Path,
) -> tuple[Path, str, str, str, str, str]:
    data_home = tmp_path / "data"
    source_root = tmp_path / "corpus"
    source_root.mkdir()
    (source_root / "paper.md").write_text(
        "# Evidence\n\nA review keeps evidence immutable and scoped.\n",
        encoding="utf-8",
    )
    library = LibraryConfig(name="Review and backup CLI")
    with initialize_library(
        library,
        data_root=data_home,
        declared_source_roots=(source_root,),
    ) as repository:
        collection = repository.create_collection(
            CollectionConfig(
                library_id=library.library_id,
                name="Corpus",
                kind=CollectionKind.CORPUS,
                roots=(
                    build_collection_root(
                        source_root,
                        data_root=repository.paths.application_data_root,
                    ),
                ),
            )
        )
        collection_id = collection.config.collection_id
        IngestService(repository).ingest_collection(collection_id)
        snapshot = repository.freeze_snapshot((collection_id,))
        policy = AccessPolicySnapshot(
            access_policy_id="policy_cli_review_backup",
            library_id=library.library_id,
            allowed_purposes=("research",),
            collection_rules=(CollectionRule(collection_id, PolicyEffect.ALLOW),),
        )
        repository.persist_access_policy(name="Research", snapshot=policy)
        profile = current_fts_runtime_profile()
        recalled = RecallService(
            SQLiteRecallBackend(repository),
            profile_version=profile.profile_version,
        ).recall(
            QueryRequest(
                question="What keeps evidence immutable and scoped?",
                library_id=library.library_id,
                collection_ids=(collection_id,),
                corpus_snapshot_id=snapshot.corpus_snapshot_id,
                access_policy_id=policy.access_policy_id,
                purpose="research",
            )
        )
        fragment = recalled.packet.source_fragments[0]
        target = ReviewService(repository).target(
            ReviewTargetType.SOURCE_FRAGMENT,
            fragment.source_fragment_id,
        )
        return (
            data_home,
            library.library_id,
            collection_id,
            target.target_id,
            target.target_hash,
            recalled.packet.evidence_packet_id,
        )


def test_review_decide_and_queue_emit_the_same_canonical_decision(tmp_path: Path) -> None:
    data_home, library_id, collection_id, target_id, target_hash, _packet_id = _packet_and_target(
        tmp_path
    )
    decision, _ = _json(
        [
            "review",
            "decide",
            target_id,
            "--library",
            library_id,
            "--target-type",
            "source_fragment",
            "--target-hash",
            target_hash,
            "--action",
            "accept",
            "--reason",
            "Checked against the exact source and intended use.",
            "--authority",
            "researcher:eugene",
            "--collection",
            collection_id,
            "--use",
            "research",
            "--data-home",
            str(data_home),
        ]
    )
    assert decision["schema"] == "dithyramba.review_decision/1.0"
    assert decision["target_hash"] == target_hash

    queue, _ = _json(
        [
            "review",
            "queue",
            "--library",
            library_id,
            "--target-type",
            "source_fragment",
            "--target-id",
            target_id,
            "--data-home",
            str(data_home),
        ]
    )
    assert queue["schema"] == "dithyramba.review_decision_list/1.0"
    assert cast(list[dict[str, object]], queue["items"]) == [decision]


def test_complete_backup_and_restore_cli_preserve_library_and_packet(tmp_path: Path) -> None:
    data_home, library_id, _collection_id, _target_id, _target_hash, packet_id = _packet_and_target(
        tmp_path
    )
    with open_library(library_id, data_root=data_home) as repository:
        original_hash = SQLiteRecallBackend(repository).load_evidence_packet(packet_id).packet_hash

    backup, _ = _json(
        [
            "backup",
            "--library",
            library_id,
            "--data-home",
            str(data_home),
        ]
    )
    bundle = Path(cast(str, backup["path"]))
    assert backup["schema"] == "dithyramba.backup_bundle/1.0"
    assert bundle.is_dir()
    assert len(cast(str, backup["manifest_hash"])) == 64

    restored_home = tmp_path / "restored"
    receipt, _ = _json(
        [
            "restore",
            str(bundle),
            "--data-home",
            str(restored_home),
        ]
    )
    assert receipt["schema"] == "dithyramba.restore_receipt/1.0"
    assert receipt["library_id"] == library_id
    assert receipt["status"] == "verified"
    assert receipt["backup_manifest_hash"] == backup["manifest_hash"]

    with open_library(library_id, data_root=restored_home) as repository:
        restored_packet = SQLiteRecallBackend(repository).load_evidence_packet(packet_id)
        assert restored_packet.packet_hash == original_hash


def test_review_and_restore_require_explicit_scope() -> None:
    for arguments in (
        ["review", "queue"],
        ["review", "decide", "fragment_missing"],
        ["restore", "/tmp/missing-bundle"],
    ):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 2
        assert "Missing option" in result.output
