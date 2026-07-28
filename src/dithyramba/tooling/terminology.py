"""Deterministic linting for Dithyramba's deprecated public identifiers.

The linter deliberately matches complete ASCII identifier tokens. A deprecated
standalone identifier is therefore reported on its own, while a canonical
compound that merely contains the same letters is not.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

# dithyramba:deprecated-registry:start
DEPRECATED_REPLACEMENTS: dict[str, str] = {
    "MemoryProject": "Library",
    "MemorySpace": "Collection",
    "PolicyLabel": "AccessPolicy",
    "FederationGrant": "CrossLibraryGrant",
    "PacketProfile": "OutputProfile",
    "Adapter": "Connector",
    "Span": "SourceFragment",
    "Locator": "SourceAddress",
    "StructureNode": "StructureUnit",
    "StructureMap": "StructureAlignment",
    "EpistemicRecord": "Statement",
    "Claim": "Statement",
    "ConceptSense": "ConceptMeaning",
    "Position": "Viewpoint",
    "TemporalEnvelope": "TimeContext",
    "SupportProfile": "EvidenceProfile",
    "Gap": "EvidenceGap",
    "ResearchJob": "ResearchTask",
    "CoverageReceipt": "CoverageReport",
    "QuerySpec": "QueryRequest",
    "ActualReadSet": "ReadReceipt",
    "PolicyReceipt": "AccessReceipt",
    "DerivationRun": "ProcessingRun",
    "DatasetRelease": "ReleaseBundle",
    "Projection": "View",
    "Decision": "ReviewDecision",
    "Attestation": "ReviewCertificate",
    "LifecycleRecord": "ChangeRecord",
    "RetentionClass": "StorageClass",
    "ReviewDebt": "ReviewBacklog",
}

DEPRECATED_IDENTIFIERS: tuple[str, ...] = tuple(DEPRECATED_REPLACEMENTS)
# dithyramba:deprecated-registry:end

EXTERNAL_QUOTE_START = "<!-- dithyramba:external-quote:start -->"
EXTERNAL_QUOTE_END = "<!-- dithyramba:external-quote:end -->"
DEPRECATED_REGISTRY_START = "dithyramba:deprecated-registry:start"
DEPRECATED_REGISTRY_END = "dithyramba:deprecated-registry:end"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_REPLACEMENT_HEADER = "замінює"
_DEPRECATED_GATE_HEADING = "deprecated-term gate"


@dataclass(frozen=True, slots=True)
class TerminologyFinding:
    """One stable, source-addressed deprecated-identifier finding."""

    path: str
    line: int
    column: int
    identifier: str
    replacement: str
    text: str
    rule: str = "deprecated-identifier"

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable mapping with a stable key order."""

        return {
            "rule": self.rule,
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "identifier": self.identifier,
            "replacement": self.replacement,
            "text": self.text,
        }


def _tagged_intervals(text: str, start_marker: str, end_marker: str) -> tuple[tuple[int, int], ...]:
    """Return only closed tagged intervals; an unmatched opener grants no exemption."""

    intervals: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(start_marker, cursor)
        if start < 0:
            break
        end = text.find(end_marker, start + len(start_marker))
        if end < 0:
            break
        interval_end = end + len(end_marker)
        intervals.append((start, interval_end))
        cursor = interval_end
    return tuple(intervals)


def _inside_intervals(offset: int, intervals: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= offset < end for start, end in intervals)


def _markdown_cells(line: str) -> tuple[tuple[int, int, str], ...]:
    """Return Markdown table cells as zero-based source ranges."""

    if not line.lstrip().startswith("|"):
        return ()
    separators = [
        index
        for index, character in enumerate(line)
        if character == "|" and (index == 0 or line[index - 1] != "\\")
    ]
    if len(separators) < 2:
        return ()
    return tuple((left + 1, right, line[left + 1 : right]) for left, right in pairwise(separators))


def _normalized_cell(value: str) -> str:
    return value.strip().strip("`").strip().casefold()


def _is_migration_history_fixture(path: str) -> bool:
    normalized = path.replace("\\", "/").casefold()
    parts = tuple(part for part in normalized.split("/") if part)
    if "fixtures" not in parts:
        return False
    return any("migration_history" in part or "migration-history" in part for part in parts)


def _deprecated_registry_lines(lines: tuple[str, ...], path: str) -> frozenset[int]:
    """Allow the canonical registry code block in TERMINOLOGY.md to name its entries."""

    if Path(path).name.casefold() != "terminology.md":
        return frozenset()

    in_gate = False
    in_fence = False
    allowed: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        heading = _MARKDOWN_HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip().casefold()
            if level == 2:
                in_gate = title == _DEPRECATED_GATE_HEADING
                in_fence = False
            elif level < 2:
                in_gate = False
                in_fence = False
        if in_gate and line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_gate and in_fence:
            allowed.add(line_number)
    return frozenset(allowed)


def scan_text(text: str, *, path: str | Path = "<memory>") -> tuple[TerminologyFinding, ...]:
    """Find deprecated identifier tokens in text.

    Three narrowly scoped historical-data exemptions are recognized:

    * cells under a Markdown ``Замінює`` column;
    * files under a ``fixtures/**/migration_history*`` path;
    * closed blocks marked with :data:`EXTERNAL_QUOTE_START` and
      :data:`EXTERNAL_QUOTE_END`.

    The deprecated registry fenced block in the canonical ``TERMINOLOGY.md`` is
    also exempt so that the source of this lint rule can lint itself.
    """

    display_path = str(path).replace("\\", "/")
    if _is_migration_history_fixture(display_path):
        return ()

    quote_intervals = (
        *_tagged_intervals(text, EXTERNAL_QUOTE_START, EXTERNAL_QUOTE_END),
        *_tagged_intervals(text, DEPRECATED_REGISTRY_START, DEPRECATED_REGISTRY_END),
    )
    physical_lines = tuple(text.splitlines(keepends=True))
    plain_lines = tuple(line.rstrip("\r\n") for line in physical_lines)
    registry_lines = _deprecated_registry_lines(plain_lines, display_path)

    findings: list[TerminologyFinding] = []
    replacement_column: int | None = None
    absolute_line_start = 0

    for line_number, (physical_line, line) in enumerate(
        zip(physical_lines, plain_lines, strict=True),
        start=1,
    ):
        cells = _markdown_cells(line)
        replacement_range: tuple[int, int] | None = None
        if cells:
            normalized_cells = tuple(_normalized_cell(cell[2]) for cell in cells)
            if _REPLACEMENT_HEADER in normalized_cells:
                replacement_column = normalized_cells.index(_REPLACEMENT_HEADER)
            elif replacement_column is not None and replacement_column < len(cells):
                replacement_range = cells[replacement_column][:2]
        else:
            replacement_column = None

        if line_number not in registry_lines:
            for token in _IDENTIFIER.finditer(line):
                identifier = token.group(0)
                replacement = DEPRECATED_REPLACEMENTS.get(identifier)
                if replacement is None:
                    continue
                if replacement_range is not None:
                    start, end = replacement_range
                    if start <= token.start() < end:
                        continue
                if _inside_intervals(absolute_line_start + token.start(), quote_intervals):
                    continue
                findings.append(
                    TerminologyFinding(
                        path=display_path,
                        line=line_number,
                        column=token.start() + 1,
                        identifier=identifier,
                        replacement=replacement,
                        text=line,
                    )
                )
        absolute_line_start += len(physical_line)

    return tuple(findings)


def scan_file(path: str | Path) -> tuple[TerminologyFinding, ...]:
    """Scan one UTF-8 text file."""

    source = Path(path)
    return scan_text(source.read_text(encoding="utf-8"), path=source)


def findings_as_jsonable(
    findings: Iterable[TerminologyFinding],
) -> list[dict[str, str | int]]:
    """Return findings in deterministic source order for JSON serialization."""

    ordered = sorted(
        findings,
        key=lambda item: (item.path, item.line, item.column, item.identifier),
    )
    return [finding.to_dict() for finding in ordered]


def exit_code_for_findings(findings: Iterable[TerminologyFinding]) -> int:
    """Return the conventional lint exit code: zero when clean, one otherwise."""

    return 1 if any(True for _ in findings) else 0


_TEXT_SUFFIXES = frozenset(
    {".json", ".jsonl", ".md", ".py", ".sql", ".toml", ".txt", ".yaml", ".yml"}
)
_IGNORED_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "artifacts", "build", "dist"}
)
_SELF_TEST_PATH = "tests/unit/test_terminology.py"


def iter_text_files(roots: Iterable[str | Path]) -> tuple[Path, ...]:
    """Return a stable, de-duplicated list of repository text files."""

    discovered: dict[str, Path] = {}
    cwd = Path.cwd().resolve()
    for raw_root in roots:
        root = Path(raw_root)
        candidates = (root,) if root.is_file() else root.rglob("*")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.casefold() not in _TEXT_SUFFIXES:
                continue
            resolved = candidate.resolve()
            try:
                relative = resolved.relative_to(cwd).as_posix()
            except ValueError:
                relative = resolved.as_posix()
            if any(part in _IGNORED_DIRECTORIES for part in Path(relative).parts):
                continue
            if relative == _SELF_TEST_PATH:
                continue
            discovered[relative] = candidate
    return tuple(discovered[key] for key in sorted(discovered))


def scan_paths(roots: Iterable[str | Path]) -> tuple[TerminologyFinding, ...]:
    """Scan repository paths in stable source order."""

    findings: list[TerminologyFinding] = []
    cwd = Path.cwd().resolve()
    for path in iter_text_files(roots):
        resolved = path.resolve()
        try:
            display_path = resolved.relative_to(cwd)
        except ValueError:
            display_path = resolved
        findings.extend(scan_text(path.read_text(encoding="utf-8"), path=display_path))
    return tuple(
        sorted(findings, key=lambda item: (item.path, item.line, item.column, item.identifier))
    )


def main(argv: list[str] | None = None) -> int:
    """Run the repository terminology gate and emit deterministic JSON."""

    parser = argparse.ArgumentParser(description="Reject deprecated Dithyramba identifiers.")
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan.")
    namespace = parser.parse_args(argv)
    findings = scan_paths(namespace.paths)
    json.dump(
        {
            "rule": "deprecated-identifiers",
            "count": len(findings),
            "findings": findings_as_jsonable(findings),
        },
        sys.stdout,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")
    return exit_code_for_findings(findings)


if __name__ == "__main__":
    raise SystemExit(main())
