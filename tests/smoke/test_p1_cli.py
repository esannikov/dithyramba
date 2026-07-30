"""P1 smoke coverage for the explicit Library-scoped CLI."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from dithyramba.cli import _fts5_available, app

runner = CliRunner()


@dataclass(frozen=True, slots=True)
class CliState:
    data_home: Path
    corpus_root: Path
    sync_root: Path
    library_id: str


def _invoke_json(arguments: list[str]) -> dict[str, object]:
    result = runner.invoke(app, [*arguments, "--json"])
    assert result.exit_code == 0, result.output
    encoded = result.stdout.strip()
    decoded = cast(dict[str, object], json.loads(encoded))
    assert encoded == json.dumps(
        decoded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return decoded


def _bootstrap(tmp_path: Path) -> CliState:
    data_home = tmp_path / "app-data"
    corpus_root = tmp_path / "corpus"
    sync_root = tmp_path / "sync"
    corpus_root.mkdir(parents=True)
    sync_root.mkdir()
    payload = _invoke_json(
        [
            "library",
            "init",
            "Research Library",
            "--data-home",
            str(data_home),
            "--source-root",
            str(corpus_root),
            "--sync-root",
            str(sync_root),
        ]
    )
    return CliState(
        data_home=data_home,
        corpus_root=corpus_root,
        sync_root=sync_root,
        library_id=cast(str, payload["library_id"]),
    )


def _add_collection(
    state: CliState,
    *,
    name: str,
    root: Path,
    kind: str = "corpus",
    custom_globs: bool = False,
) -> dict[str, object]:
    arguments = [
        "collection",
        "add",
        name,
        "--library",
        state.library_id,
        "--root",
        str(root),
        "--kind",
        kind,
        "--data-home",
        str(state.data_home),
    ]
    if custom_globs:
        arguments.extend(["--include", "**/*.md", "--exclude", "private/**"])
    return _invoke_json(arguments)


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (["--help"], "access-policy"),
        (["library", "--help"], "doctor"),
        (["collection", "--help"], "add"),
        (["access-policy", "--help"], "check"),
    ],
)
def test_help_surfaces_are_discoverable(arguments: list[str], expected: str) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert expected in result.stdout


def test_library_init_list_and_both_doctor_surfaces(tmp_path: Path) -> None:
    empty_home = tmp_path / "empty-home"
    empty = runner.invoke(app, ["library", "list", "--data-home", str(empty_home)])
    assert empty.exit_code == 0
    assert empty.stdout.strip() == "No Libraries."

    state = _bootstrap(tmp_path)
    listed = _invoke_json(["library", "list", "--data-home", str(state.data_home)])
    libraries = cast(list[dict[str, object]], listed["libraries"])
    assert [item["library_id"] for item in libraries] == [state.library_id]
    assert libraries[0]["path"] == str(state.data_home / "libraries" / state.library_id)

    library_doctor = _invoke_json(
        [
            "library",
            "doctor",
            "--library",
            state.library_id,
            "--data-home",
            str(state.data_home),
        ]
    )
    assert library_doctor["status"] == "ok"
    assert library_doctor["schema_version"] == 11
    assert isinstance(library_doctor["schema_fingerprint"], str)
    assert library_doctor["external_services_contacted"] is False

    root_doctor = runner.invoke(
        app,
        [
            "doctor",
            "--library",
            state.library_id,
            "--data-home",
            str(state.data_home),
        ],
    )
    assert root_doctor.exit_code == 0
    assert "status: ok" in root_doctor.stdout


def test_collection_add_and_list_preserve_roots_and_globs(tmp_path: Path) -> None:
    state = _bootstrap(tmp_path)
    empty = runner.invoke(
        app,
        [
            "collection",
            "list",
            "--library",
            state.library_id,
            "--data-home",
            str(state.data_home),
        ],
    )
    assert empty.exit_code == 0
    assert empty.stdout.strip() == "No Collections."

    first = _add_collection(
        state,
        name="PhD corpus",
        root=state.corpus_root,
        custom_globs=True,
    )
    second_root = state.corpus_root.parent / "case"
    second_root.mkdir()
    second = _add_collection(state, name="Case", root=second_root, kind="case")

    assert first["kind"] == "corpus"
    first_roots = cast(list[dict[str, object]], first["roots"])
    assert first_roots[0]["include_globs"] == ["**/*.md"]
    assert first_roots[0]["exclude_globs"] == ["private/**"]
    second_roots = cast(list[dict[str, object]], second["roots"])
    assert second_roots[0]["include_globs"] == [
        "**/*.md",
        "**/*.markdown",
        "**/*.txt",
        "**/*.pdf",
    ]

    listed = _invoke_json(
        [
            "collection",
            "list",
            "--library",
            state.library_id,
            "--data-home",
            str(state.data_home),
        ]
    )
    records = cast(list[dict[str, object]], listed["collections"])
    assert {item["collection_id"] for item in records} == {
        first["collection_id"],
        second["collection_id"],
    }


def test_access_policy_create_show_and_metadata_checks(tmp_path: Path) -> None:
    state = _bootstrap(tmp_path)
    allowed = _add_collection(state, name="Allowed", root=state.corpus_root)
    (state.corpus_root / "explicit-allow.md").write_text("Allowed source.\n", encoding="utf-8")
    (state.corpus_root / "explicit-deny.md").write_text("Denied source.\n", encoding="utf-8")
    allowed_source_result = _invoke_json(
        [
            "source",
            "add",
            "explicit-allow.md",
            "--library",
            state.library_id,
            "--collection",
            cast(str, allowed["collection_id"]),
            "--data-home",
            str(state.data_home),
        ]
    )
    denied_source_result = _invoke_json(
        [
            "source",
            "add",
            "explicit-deny.md",
            "--library",
            state.library_id,
            "--collection",
            cast(str, allowed["collection_id"]),
            "--data-home",
            str(state.data_home),
        ]
    )
    allowed_source_id = cast(
        str,
        cast(list[dict[str, object]], allowed_source_result["outcomes"])[0]["source_id"],
    )
    denied_source_id = cast(
        str,
        cast(list[dict[str, object]], denied_source_result["outcomes"])[0]["source_id"],
    )
    denied_root = tmp_path / "denied"
    denied_root.mkdir()
    denied = _add_collection(state, name="Denied", root=denied_root, kind="holdout")
    unlisted_root = tmp_path / "unlisted"
    unlisted_root.mkdir()
    unlisted = _add_collection(state, name="Unlisted", root=unlisted_root, kind="branch")

    policy = _invoke_json(
        [
            "access-policy",
            "create",
            "Research policy",
            "--library",
            state.library_id,
            "--purpose",
            "research",
            "--purpose",
            "writing",
            "--allow-collection",
            cast(str, allowed["collection_id"]),
            "--deny-collection",
            cast(str, denied["collection_id"]),
            "--allow-source",
            allowed_source_id,
            "--deny-source",
            denied_source_id,
            "--allow-export",
            "--allow-external-provider",
            "--data-home",
            str(state.data_home),
        ]
    )
    policy_id = cast(str, policy["access_policy_id"])
    assert policy["allow_export"] is True
    assert policy["allow_external_provider"] is True
    assert policy["allowed_purposes"] == ["research", "writing"]
    source_rules = cast(list[dict[str, str]], policy["source_rules"])
    assert {item["source_id"]: item["effect"] for item in source_rules} == {
        allowed_source_id: "allow",
        denied_source_id: "deny",
    }

    shown = _invoke_json(
        [
            "access-policy",
            "show",
            policy_id,
            "--library",
            state.library_id,
            "--data-home",
            str(state.data_home),
        ]
    )
    assert shown == policy

    allowed_check = _invoke_json(
        [
            "access-policy",
            "check",
            policy_id,
            "--library",
            state.library_id,
            "--purpose",
            "research",
            "--collection",
            cast(str, allowed["collection_id"]),
            "--data-home",
            str(state.data_home),
        ]
    )
    assert allowed_check["request_allowed"] is True

    denied_check = _invoke_json(
        [
            "access-policy",
            "check",
            policy_id,
            "--library",
            state.library_id,
            "--purpose",
            "other",
            "--collection",
            cast(str, denied["collection_id"]),
            "--collection",
            cast(str, unlisted["collection_id"]),
            "--data-home",
            str(state.data_home),
        ]
    )
    assert denied_check["purpose_allowed"] is False
    assert denied_check["request_allowed"] is False
    decisions = cast(list[dict[str, str]], denied_check["collection_decisions"])
    assert {item["basis"] for item in decisions} == {"explicit_deny", "default_deny"}
    assert "path" not in json.dumps(denied_check)


def test_backup_is_verified_and_promoted(tmp_path: Path) -> None:
    state = _bootstrap(tmp_path)
    payload = _invoke_json(
        [
            "backup",
            "--library",
            state.library_id,
            "--data-home",
            str(state.data_home),
        ]
    )

    backup_path = Path(cast(str, payload["path"]))
    assert backup_path.is_dir()
    assert backup_path.parent == state.data_home / "libraries" / state.library_id / "backups"
    assert payload["schema_version"] == 11
    assert len(cast(str, payload["manifest_hash"])) == 64
    assert (backup_path / "manifest.json").is_file()
    assert (backup_path / "memory.sqlite3").is_file()
    assert (backup_path / "events.jsonl").is_file()


@pytest.mark.parametrize(
    "arguments",
    [
        ["doctor"],
        ["library", "doctor"],
        ["collection", "list", "--data-home", "/tmp"],
        ["access-policy", "show", "policy_missing", "--data-home", "/tmp"],
        ["backup", "--data-home", "/tmp"],
    ],
)
def test_commands_require_explicit_library_scope(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert "Missing option" in result.output


def test_relative_and_overlapping_paths_fail_concisely(tmp_path: Path) -> None:
    relative = runner.invoke(
        app,
        ["library", "init", "Relative", "--data-home", "relative-home"],
    )
    assert relative.exit_code == 1
    assert relative.stderr.startswith("error:")
    assert "Traceback" not in relative.output

    overlap_home = tmp_path / "overlap"
    overlap = runner.invoke(
        app,
        [
            "library",
            "init",
            "Overlap",
            "--data-home",
            str(overlap_home),
            "--source-root",
            str(overlap_home),
        ],
    )
    assert overlap.exit_code == 1
    assert "overlaps" in overlap.stderr
    assert not overlap_home.exists()

    state = _bootstrap(tmp_path / "collection-case")
    collection_overlap = runner.invoke(
        app,
        [
            "collection",
            "add",
            "Unsafe",
            "--library",
            state.library_id,
            "--root",
            str(state.data_home),
            "--kind",
            "corpus",
            "--data-home",
            str(state.data_home),
        ],
    )
    assert collection_overlap.exit_code == 1
    assert "overlaps application data" in collection_overlap.stderr
    assert "Traceback" not in collection_overlap.output


def test_policy_check_rejects_ambiguous_scope_without_snapshot_fabrication(tmp_path: Path) -> None:
    state = _bootstrap(tmp_path)
    collection = _add_collection(state, name="Corpus", root=state.corpus_root)
    collection_id = cast(str, collection["collection_id"])
    policy = _invoke_json(
        [
            "access-policy",
            "create",
            "Policy",
            "--library",
            state.library_id,
            "--purpose",
            "research",
            "--allow-collection",
            collection_id,
            "--data-home",
            str(state.data_home),
        ]
    )
    policy_id = cast(str, policy["access_policy_id"])

    duplicate = runner.invoke(
        app,
        [
            "access-policy",
            "check",
            policy_id,
            "--library",
            state.library_id,
            "--purpose",
            "research",
            "--collection",
            collection_id,
            "--collection",
            collection_id,
            "--data-home",
            str(state.data_home),
        ],
    )
    assert duplicate.exit_code == 1
    assert "unique IDs" in duplicate.stderr

    malformed_purpose = runner.invoke(
        app,
        [
            "access-policy",
            "check",
            policy_id,
            "--library",
            state.library_id,
            "--purpose",
            "Not Valid",
            "--collection",
            collection_id,
            "--data-home",
            str(state.data_home),
        ],
    )
    assert malformed_purpose.exit_code == 1
    assert "purpose must match" in malformed_purpose.stderr


def test_no_destructive_command_is_exposed() -> None:
    for arguments in (
        ["delete"],
        ["library", "delete"],
        ["collection", "delete"],
        ["access-policy", "delete"],
    ):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 2
        assert "No such command" in result.output


def test_fts5_probe_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoFtsConnection:
        def __init__(self) -> None:
            self.closed = False

        def execute(self, _statement: str) -> None:
            raise sqlite3.DatabaseError("fts5 omitted")

        def close(self) -> None:
            self.closed = True

    connection = NoFtsConnection()
    monkeypatch.setattr("dithyramba.cli.sqlite3.connect", lambda _database: connection)

    assert _fts5_available() is False
    assert connection.closed is True
