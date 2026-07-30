"""CLI smoke coverage for the explicit backup-first Library migration command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

from typer.testing import CliRunner

from dithyramba.cli import app
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.library import LibraryConfig, create_library_layout
from dithyramba.persistence import open_library
from dithyramba.persistence.repository import library_logical_identity_hash
from dithyramba.store.database import Store

_NOW = "2026-07-21T12:00:00.000000Z"
runner = CliRunner()


def _v3_library(tmp_path: Path) -> tuple[LibraryConfig, Path]:
    data_home = tmp_path / "data"
    config = LibraryConfig(name="Migration CLI fixture")
    paths = create_library_layout(config, data_root=data_home)
    prefix = tmp_path / "v3-migrations"
    prefix.mkdir()
    repository_root = Path(__file__).resolve().parents[2]
    for name in ("0001_core.sql", "0002_source_heads.sql", "0003_recall_run_artifacts.sql"):
        shutil.copyfile(repository_root / "migrations" / name, prefix / name)

    with Store.open_library(paths, migration_source=prefix) as store:
        identity_hash = library_logical_identity_hash(config)
        store.connection.execute(
            "INSERT INTO libraries VALUES (?, ?, ?, ?)",
            (config.library_id, config.name, identity_hash, _NOW),
        )
        event_payload = {
            "schema": "dithyramba.library_created/1.0",
            "library_id": config.library_id,
            "name": config.name,
            "logical_identity_hash": identity_hash,
        }
        store.connection.execute(
            "INSERT INTO event_outbox VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                "event_migration_cli",
                "library.created",
                "library",
                config.library_id,
                canonical_json_bytes(event_payload).decode("utf-8"),
                canonical_sha256_hex(event_payload),
                _NOW,
            ),
        )
    return config, data_home


def test_library_migrate_cli_emits_canonical_verified_receipt(tmp_path: Path) -> None:
    config, data_home = _v3_library(tmp_path)

    result = runner.invoke(
        app,
        [
            "library",
            "migrate",
            "--library",
            config.library_id,
            "--data-home",
            str(data_home),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = cast(dict[str, object], json.loads(result.stdout))
    assert payload["schema"] == "dithyramba.library_migration_receipt/1.0"
    assert payload["library_id"] == config.library_id
    assert payload["old_schema_version"] == 3
    assert payload["new_schema_version"] == 11
    assert payload["status"] == "verified"
    assert len(cast(str, payload["backup_manifest_hash"])) == 64
    assert len(cast(str, payload["receipt_hash"])) == 64
    assert Path(cast(str, payload["backup_path"])).is_dir()
    assert result.stdout.encode("utf-8") == canonical_json_bytes(payload) + b"\n"

    with open_library(config.library_id, data_root=data_home) as repository:
        assert repository.schema_version == 11

    repeated = runner.invoke(
        app,
        [
            "library",
            "migrate",
            "--library",
            config.library_id,
            "--data-home",
            str(data_home),
        ],
    )
    assert repeated.exit_code == 1
    assert "already at schema head version 11" in repeated.output
