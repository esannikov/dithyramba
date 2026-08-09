"""Generate one commit-bound receipt from executable public contracts."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

from typer.main import get_command

from dithyramba import __version__
from dithyramba.cli import app
from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.mcp_stdio import mcp_contract_manifest
from dithyramba.store.migrations import verify_migrations_smoke

RECEIPT_SCHEMA = "dithyramba.release_contract_receipt/1.0"
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_IGNORED_PARTS = frozenset(
    {
        ".agent",
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        "artifacts",
        "build",
        "dist",
        "htmlcov",
    }
)


def build_contract_receipt(*, source_root: Path, commit_sha: str) -> dict[str, object]:
    """Build a deterministic release receipt or fail on broken local docs links."""

    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError("source root must be an existing directory")
    normalized_commit = commit_sha.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", normalized_commit) is None:
        raise ValueError("commit SHA must contain exactly 40 lowercase hexadecimal characters")

    docs = _documentation_link_receipt(root)
    broken_links = docs["broken_local_links"]
    if not isinstance(broken_links, list):
        raise ValueError("documentation receipt returned an invalid link list")
    if broken_links:
        broken = ", ".join(str(item) for item in broken_links)
        raise ValueError(f"broken local documentation links: {broken}")
    semantic: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "package_version": __version__,
        "commit_sha": normalized_commit,
        "cli_commands": list(_cli_command_paths()),
        "mcp": mcp_contract_manifest(),
        "sqlite": verify_migrations_smoke(),
        "documentation": docs,
    }
    return {**semantic, "receipt_hash": canonical_sha256_hex(semantic)}


def _cli_command_paths() -> tuple[str, ...]:
    root = get_command(app)
    paths: list[str] = []

    def visit(command: object, prefix: tuple[str, ...]) -> None:
        children = getattr(command, "commands", None)
        if not isinstance(children, dict):
            return
        for name, child in sorted(children.items()):
            path = (*prefix, str(name))
            paths.append(" ".join(path))
            visit(child, path)

    visit(root, ("dithyramba",))
    return tuple(paths)


def _documentation_link_receipt(root: Path) -> dict[str, object]:
    markdown_files = tuple(
        sorted(
            (
                path
                for path in root.rglob("*.md")
                if not _IGNORED_PARTS.intersection(path.relative_to(root).parts)
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    checked_links = 0
    broken: list[str] = []
    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for raw_target in _MARKDOWN_LINK.findall(text):
            target = _local_link_target(raw_target)
            if target is None:
                continue
            checked_links += 1
            linked = (
                root / target.lstrip("/") if target.startswith("/") else document.parent / target
            ).resolve()
            if not linked.is_relative_to(root) or not linked.exists():
                broken.append(f"{document.relative_to(root).as_posix()} -> {target}")
    return {
        "checked_markdown_files": len(markdown_files),
        "checked_local_links": checked_links,
        "broken_local_links": broken,
    }


def _local_link_target(raw_target: str) -> str | None:
    value = raw_target.strip()
    if value.startswith("<") and ">" in value:
        value = value[1 : value.index(">")]
    elif " " in value:
        value = value.split(" ", 1)[0]
    lowered = value.casefold()
    if (
        not value
        or value.startswith("#")
        or lowered.startswith(("data:", "http://", "https://", "mailto:"))
    ):
        return None
    return unquote(value.split("#", 1)[0].split("?", 1)[0]) or None


def _git_head(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--commit-sha")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        receipt = build_contract_receipt(
            source_root=arguments.source_root,
            commit_sha=arguments.commit_sha or _git_head(arguments.source_root),
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError) as error:
        print(f"contract receipt failed: {error}", file=sys.stderr)
        return 1
    encoded = canonical_json_bytes(receipt)
    if arguments.output is None:
        sys.stdout.buffer.write(encoded + b"\n")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(encoded + b"\n")
        print(
            f"✓ contract receipt {receipt['receipt_hash']} -> {arguments.output}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
