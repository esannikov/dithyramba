"""P2 CLI smoke and adversarial contract coverage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from typer.main import get_command
from typer.testing import CliRunner, Result

from dithyramba.cli import app
from dithyramba.ingest.models import LARGE_DOCUMENT_PARSER_PROFILE, PARSER_PROFILE, ParserLimits

runner = CliRunner()


@dataclass(frozen=True, slots=True)
class P2CliState:
    data_home: Path
    roots: tuple[Path, ...]
    library_id: str
    collection_id: str
    collection_root_ids: tuple[str, ...]


def _invoke_json(
    arguments: list[str], *, expected_exit: int = 0
) -> tuple[dict[str, object], Result]:
    result = runner.invoke(app, [*arguments, "--json"])
    assert result.exit_code == expected_exit, result.output
    encoded = result.stdout.strip()
    decoded = cast(dict[str, object], json.loads(encoded))
    assert encoded == json.dumps(
        decoded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return decoded, result


def _bootstrap(tmp_path: Path, *, root_count: int = 1) -> P2CliState:
    data_home = tmp_path / "data"
    roots = tuple(tmp_path / f"corpus-{index}" for index in range(root_count))
    for root in roots:
        root.mkdir(parents=True)

    library_arguments = [
        "library",
        "init",
        "P2 CLI",
        "--data-home",
        str(data_home),
    ]
    for root in roots:
        library_arguments.extend(["--source-root", str(root)])
    library, _ = _invoke_json(library_arguments)
    library_id = cast(str, library["library_id"])

    collection_arguments = [
        "collection",
        "add",
        "Corpus",
        "--library",
        library_id,
        "--kind",
        "corpus",
        "--data-home",
        str(data_home),
    ]
    for root in roots:
        collection_arguments.extend(["--root", str(root)])
    collection, _ = _invoke_json(collection_arguments)
    root_payloads = cast(list[dict[str, object]], collection["roots"])
    return P2CliState(
        data_home=data_home,
        roots=roots,
        library_id=library_id,
        collection_id=cast(str, collection["collection_id"]),
        collection_root_ids=tuple(cast(str, item["collection_root_id"]) for item in root_payloads),
    )


def _scope(state: P2CliState) -> list[str]:
    return [
        "--library",
        state.library_id,
        "--collection",
        state.collection_id,
        "--data-home",
        str(state.data_home),
    ]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--help"], "index"),
        (["--help"], "source"),
        (["source", "--help"], "versions"),
    ],
)
def test_p2_help_surfaces_are_discoverable(arguments: list[str], expected: str) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 0
    assert expected in result.stdout


def test_source_add_exposes_collection_root_option() -> None:
    root = cast(Any, get_command(app))
    source = root.get_command(None, "source")
    assert source is not None
    add = source.get_command(None, "add")
    assert add is not None
    assert any("--collection-root" in parameter.opts for parameter in add.params)


def test_index_emits_full_canonical_receipt_and_unchanged_replay(tmp_path: Path) -> None:
    state = _bootstrap(tmp_path)
    source_text = "# Canary\n\nCLI_SOURCE_TEXT_MUST_NOT_LEAK.\n"
    (state.roots[0] / "note.md").write_text(source_text, encoding="utf-8")

    added, _ = _invoke_json(["index", *_scope(state)])

    assert added["schema"] == "dithyramba.ingest_result/1.0"
    assert added["library_id"] == state.library_id
    assert added["collection_id"] == state.collection_id
    run = cast(dict[str, object], added["run"])
    coverage = cast(dict[str, object], added["coverage"])
    outcomes = cast(list[dict[str, object]], added["outcomes"])
    assert run["status"] == "succeeded"
    assert run["profile_version"] == PARSER_PROFILE
    assert run["processing_run_id"] == coverage["processing_run_id"]
    assert run["output_hash"] == coverage["report_hash"]
    assert coverage["processed_count"] == 1
    assert coverage["skipped_count"] == coverage["failed_count"] == 0
    assert outcomes[0]["relative_path"] == "note.md"
    assert outcomes[0]["terminal_outcome"] == "processed"
    assert outcomes[0]["disposition"] == "added"

    serialized = json.dumps(added, ensure_ascii=False)
    assert source_text.strip() not in serialized
    assert "CLI_SOURCE_TEXT_MUST_NOT_LEAK" not in serialized
    assert str(state.roots[0]) not in serialized
    assert str(state.data_home) not in serialized
    assert "canonical_uri" not in serialized

    unchanged, _ = _invoke_json(["index", *_scope(state)])
    repeated = cast(list[dict[str, object]], unchanged["outcomes"])
    assert repeated[0]["disposition"] == "unchanged"
    assert repeated[0]["source_id"] == outcomes[0]["source_id"]
    assert repeated[0]["source_version_id"] == outcomes[0]["source_version_id"]


def test_cli_parser_profiles_keep_default_bounded_and_admit_opted_in_large_text(
    tmp_path: Path,
) -> None:
    state = _bootstrap(tmp_path)
    defaults = ParserLimits()
    assert (defaults.max_file_bytes, defaults.max_extracted_codepoints) == (
        25 * 1024 * 1024,
        2_000_000,
    )

    payload = "\n\n".join("é" * (1024 * 1024) for _ in range(13)) + "\n"
    path = state.roots[0] / "large.txt"
    path.write_text(payload, encoding="utf-8")
    assert path.stat().st_size > defaults.max_file_bytes

    receipt, _ = _invoke_json(["index", *_scope(state), "--parser-profile", "large-document"])
    run = cast(dict[str, object], receipt["run"])
    outcome = cast(list[dict[str, object]], receipt["outcomes"])[0]
    assert run["profile_version"] == LARGE_DOCUMENT_PARSER_PROFILE
    assert outcome["terminal_outcome"] == "processed"


@pytest.mark.parametrize("command", ("index", "source"))
def test_cli_rejects_unknown_parser_profile(tmp_path: Path, command: str) -> None:
    state = _bootstrap(tmp_path)
    arguments = (
        ["index", *_scope(state)]
        if command == "index"
        else ["source", "add", "absent.md", *_scope(state)]
    )
    result = runner.invoke(app, [*arguments, "--parser-profile", "unsafe"])

    assert result.exit_code == 1
    assert "unknown parser profile" in result.stderr


def test_explicit_source_add_list_and_versions_hide_physical_locations(tmp_path: Path) -> None:
    state = _bootstrap(tmp_path)
    nested = state.roots[0] / "notes"
    nested.mkdir()
    path = nested / "paper.txt"
    path.write_text("First grounded observation.", encoding="utf-8")

    added, _ = _invoke_json(["source", "add", "notes/paper.txt", *_scope(state)])
    first_outcome = cast(list[dict[str, object]], added["outcomes"])[0]
    source_id = cast(str, first_outcome["source_id"])

    listed, _ = _invoke_json(
        [
            "source",
            "list",
            "--library",
            state.library_id,
            "--collection",
            state.collection_id,
            "--data-home",
            str(state.data_home),
        ]
    )
    sources = cast(list[dict[str, object]], listed["sources"])
    assert len(sources) == 1
    assert sources[0]["source_id"] == source_id
    assert sources[0]["title"] == "paper.txt"
    assert sources[0]["media_type"] == "text/plain"

    path.write_text("Second grounded observation.", encoding="utf-8")
    changed, _ = _invoke_json(["source", "add", "notes/paper.txt", *_scope(state)])
    assert cast(list[dict[str, object]], changed["outcomes"])[0]["disposition"] == "changed"

    versions, _ = _invoke_json(
        [
            "source",
            "versions",
            source_id,
            "--library",
            state.library_id,
            "--data-home",
            str(state.data_home),
        ]
    )
    records = cast(list[dict[str, object]], versions["versions"])
    assert [record["version_number"] for record in records] == [1, 2]
    assert [record["is_current"] for record in records] == [False, True]
    assert all(record["parse_status"] == "processed" for record in records)
    assert all(len(cast(str, record["content_sha256"])) == 64 for record in records)

    serialized = json.dumps({"listed": listed, "versions": versions})
    assert str(state.roots[0]) not in serialized
    assert str(state.data_home) not in serialized
    assert "file://" not in serialized
    assert "First grounded observation" not in serialized
    assert "Second grounded observation" not in serialized


def test_parser_skip_failure_and_invalid_path_are_successful_receipts(tmp_path: Path) -> None:
    state = _bootstrap(tmp_path)
    (state.roots[0] / "empty.md").write_text("\n\n", encoding="utf-8")
    (state.roots[0] / "invalid.txt").write_bytes(b"\xff")

    receipt, result = _invoke_json(["index", *_scope(state)])

    assert result.exit_code == 0
    run = cast(dict[str, object], receipt["run"])
    coverage = cast(dict[str, object], receipt["coverage"])
    outcomes = cast(list[dict[str, object]], receipt["outcomes"])
    assert run["status"] == "succeeded"
    assert coverage["processed_count"] == 0
    assert coverage["skipped_count"] == 1
    assert coverage["failed_count"] == 1
    assert {item["failure_code"] for item in outcomes} == {
        "invalid_utf8",
        "no_extractable_text",
    }

    traversal, traversal_result = _invoke_json(["source", "add", "../outside.md", *_scope(state)])
    assert traversal_result.exit_code == 0
    traversal_outcome = cast(list[dict[str, object]], traversal["outcomes"])[0]
    assert traversal_outcome["terminal_outcome"] == "failed"
    assert traversal_outcome["failure_code"] == "invalid_source_path"

    listed, _ = _invoke_json(
        [
            "source",
            "list",
            "--library",
            state.library_id,
            "--data-home",
            str(state.data_home),
        ]
    )
    assert listed["sources"] == []


def test_multi_root_source_add_requires_and_honors_explicit_root(tmp_path: Path) -> None:
    state = _bootstrap(tmp_path, root_count=2)
    (state.roots[1] / "second.md").write_text("# Selected\n", encoding="utf-8")

    missing = runner.invoke(app, ["source", "add", "second.md", *_scope(state)])
    assert missing.exit_code == 1
    assert "--collection-root is required" in missing.stderr
    assert "Traceback" not in missing.output

    selected, _ = _invoke_json(
        [
            "source",
            "add",
            "second.md",
            *_scope(state),
            "--collection-root",
            state.collection_root_ids[1],
        ]
    )
    outcome = cast(list[dict[str, object]], selected["outcomes"])[0]
    assert outcome["collection_root_id"] == state.collection_root_ids[1]
    assert outcome["disposition"] == "added"

    absent = runner.invoke(
        app,
        [
            "source",
            "add",
            "second.md",
            *_scope(state),
            "--collection-root",
            "collection_root_absent",
        ],
    )
    assert absent.exit_code == 1
    assert "absent from the Collection" in absent.stderr


def test_human_source_output_escapes_terminal_control_characters(tmp_path: Path) -> None:
    state = _bootstrap(tmp_path)
    relative_path = "bad\x1b[31m\nname.txt"
    (state.roots[0] / relative_path).write_text("Safe body", encoding="utf-8")

    added = runner.invoke(app, ["source", "add", relative_path, *_scope(state)])
    assert added.exit_code == 0
    assert "\x1b" not in added.stdout
    assert "\\u001b" in added.stdout
    assert "\\nname.txt" in added.stdout

    listed = runner.invoke(
        app,
        [
            "source",
            "list",
            "--library",
            state.library_id,
            "--data-home",
            str(state.data_home),
        ],
    )
    assert listed.exit_code == 0
    assert "\x1b" not in listed.stdout
    assert "\\u001b" in listed.stdout
    assert "\\nname.txt" in listed.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["index", "--library", "library_missing"],
        ["index", "--collection", "collection_missing"],
        ["source", "add", "note.md", "--library", "library_missing"],
        ["source", "list"],
        ["source", "versions", "source_missing"],
    ],
)
def test_p2_commands_require_explicit_scope(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 2
    assert "Missing option" in result.output
