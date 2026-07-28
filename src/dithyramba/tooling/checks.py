"""Deterministic aggregate quality checks for local development."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Check:
    """One required project check."""

    name: str
    command: tuple[str, ...]


CHECKS: tuple[Check, ...] = (
    Check("terminology", (sys.executable, "-m", "dithyramba.tooling.terminology", ".")),
    Check("format", ("ruff", "format", "--check", ".")),
    Check("lint", ("ruff", "check", ".")),
    Check("types", ("mypy", "--strict", "src", "tests", "verification")),
    Check("tests", ("pytest",)),
    Check("dependencies", ("pip-audit",)),
)


def run_checks(checks: Sequence[Check] = CHECKS) -> int:
    """Run checks in a stable order and stop at the first failure."""

    for check in checks:
        print(f"[check:{check.name}] {' '.join(check.command)}", flush=True)
        completed = subprocess.run(check.command, check=False)
        if completed.returncode != 0:
            print(
                f"[check:{check.name}] failed with exit code {completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode
    print("[check] all required checks passed")
    return 0


def main() -> None:
    """Console entrypoint."""

    raise SystemExit(run_checks())
