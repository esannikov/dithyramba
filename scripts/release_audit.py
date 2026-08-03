"""Deterministic pre-release audit for metadata, secrets, and local state."""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
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
    "__pycache__",
}
FORBIDDEN_STATE_PARTS = IGNORED_PARTS - {".git"}
FORBIDDEN_STATE_FILES = {".coverage"}
FORBIDDEN_NAMES = {
    ".env",
    "auth.json",
    "credentials.json",
    "memory.sqlite3",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".db",
    ".gguf",
    ".key",
    ".onnx",
    ".pem",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".wal",
}
EXPECTED_URLS = {
    "Homepage": "https://github.com/esannikov/dithyramba",
    "Issues": "https://github.com/esannikov/dithyramba/issues",
    "Repository": "https://github.com/esannikov/dithyramba",
}
EXPECTED_LICENSE = "Apache-2.0"
EXPECTED_LICENSE_SHA256 = "a8ad31b1c3f40dca5a84119351b8fa8ddc868edd77fad8a8ebf6d8f2d16fa4ae"
EXPECTED_SDIST_EXCLUDES = {
    "/.agent",
    "/.git",
    "/.github",
    "/.hypothesis",
    "/.venv",
    "/artifacts",
    "/build",
    "/dist",
}
REQUIRED_PUBLIC_PATHS = {
    "docs/REPOSITORY_GUIDE.md",
    "skills/dithyramba/SKILL.md",
    "skills/dithyramba/agents/openai.yaml",
    "skills/dithyramba/references/evidence-contract.md",
    "skills/dithyramba/references/lifecycle.md",
    "skills/dithyramba/references/mcp-tools.md",
    "verification/README.md",
    "verification/__init__.py",
    "verification/generate_synthetic_1000.py",
    "verification/run_public_replay.py",
    "verification/run_v1_fresh_corpus.py",
    "verification/synthetic_1000_manifest.json",
}
RETIRED_PUBLIC_PATHS = {
    "benchmarks",
    "src/dithyramba/evaluation",
    "src/dithyramba/tooling/acceptance.py",
}

# Assemble sensitive literals from fragments so this scanner can audit its own
# source without its policy definitions becoming findings.
_USER_HOME_PREFIX = b"/" + b"Users" + b"/"
_PRIVATE_VAULT = bytes.fromhex("5375706572") + b"[Vv]" + bytes.fromhex("61756c74323f")
_PRIVATE_ART_PATH = bytes.fromhex("30325068442f417274486973746f72792f41727469737473")
TEXT_PATTERNS = {
    "absolute macOS user path": re.compile(re.escape(_USER_HOME_PREFIX) + rb"[A-Za-z0-9._-]+/"),
    "private vault label": re.compile(_PRIVATE_VAULT + b"|" + re.escape(_PRIVATE_ART_PATH)),
    "private key header": re.compile(rb"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "OpenAI-style secret": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
}


def _entries() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def _audit_generated_state(findings: list[str]) -> None:
    """Reject local/generated state instead of silently hiding it from the audit."""

    for directory, names, filenames in os.walk(ROOT):
        current = Path(directory)
        relative = current.relative_to(ROOT)
        if ".git" in relative.parts:
            names[:] = []
            continue
        retained: list[str] = []
        for name in names:
            path = current / name
            if name == ".git":
                continue
            if name in FORBIDDEN_STATE_PARTS:
                findings.append(f"forbidden generated directory: {path.relative_to(ROOT)}")
                continue
            retained.append(name)
        names[:] = retained
        for name in filenames:
            if name in FORBIDDEN_STATE_FILES or name.startswith(".coverage."):
                findings.append(f"forbidden generated file: {(current / name).relative_to(ROOT)}")


def _audit_metadata(findings: list[str]) -> None:
    for relative in sorted(REQUIRED_PUBLIC_PATHS):
        if not (ROOT / relative).is_file():
            findings.append(f"required public release file is missing: {relative}")
    for relative in sorted(RETIRED_PUBLIC_PATHS):
        if (ROOT / relative).exists():
            findings.append(f"retired experimental release path returned: {relative}")

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_system = config.get("build-system", {})
    if build_system.get("requires") != ["hatchling==1.27.0"]:
        findings.append("pyproject build backend is not pinned to hatchling==1.27.0")
    if build_system.get("build-backend") != "hatchling.build":
        findings.append("pyproject build backend entry point differs from hatchling.build")

    project = config.get("project", {})
    if project.get("urls") != EXPECTED_URLS:
        findings.append("pyproject public URLs differ from the release contract")
    if project.get("license") != EXPECTED_LICENSE:
        findings.append("pyproject license expression is not Apache-2.0")
    if project.get("license-files") != ["LICENSE"]:
        findings.append("pyproject license-files must contain only LICENSE")
    classifiers = project.get("classifiers", [])
    if any(str(item).startswith("License ::") for item in classifiers):
        findings.append("pyproject must use the SPDX license expression, not a license classifier")

    tool = config.get("tool", {})
    if tool.get("uv", {}).get("required-version") != "==0.6.14":
        findings.append("pyproject does not enforce uv==0.6.14")
    hatch = tool.get("hatch", {}).get("build", {})
    sdist = hatch.get("targets", {}).get("sdist", {})
    excludes = set(sdist.get("exclude", []))
    missing_excludes = EXPECTED_SDIST_EXCLUDES - excludes
    if hatch.get("ignore-vcs") is not True:
        findings.append("sdist selection is still dependent on ambient VCS state")
    if hatch.get("skip-excluded-dirs") is not True:
        findings.append("sdist build still traverses explicitly excluded local-state directories")
    if missing_excludes:
        findings.append(f"sdist exclusions are incomplete: {sorted(missing_excludes)!r}")

    citation_lines = (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines()
    citation_fields = {
        line.partition(":")[0].strip().casefold(): line.partition(":")[2].strip().strip('"')
        for line in citation_lines
        if line and not line[0].isspace() and ":" in line
    }
    if citation_fields.get("version") != project.get("version"):
        findings.append("CITATION.cff version differs from pyproject.toml")
    if citation_fields.get("repository-code") != EXPECTED_URLS["Repository"]:
        findings.append("CITATION.cff repository differs from the release contract")
    if "date-released" in citation_fields:
        findings.append("CITATION.cff declares a release date before the release exists")
    if citation_fields.get("license") != EXPECTED_LICENSE:
        findings.append("CITATION.cff license differs from Apache-2.0")

    license_path = ROOT / "LICENSE"
    if not license_path.is_file():
        findings.append("LICENSE is missing")
    elif hashlib.sha256(license_path.read_bytes()).hexdigest() != EXPECTED_LICENSE_SHA256:
        findings.append("LICENSE differs from the approved Apache-2.0 text")

    for filename in ("COPYING", "LICENSE.md", "LICENSE.txt"):
        if (ROOT / filename).exists():
            findings.append(f"unexpected competing license file: {filename}")


def main() -> int:
    findings: list[str] = []
    _audit_generated_state(findings)
    files_scanned = 0
    for path in _entries():
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            findings.append(f"symbolic link is forbidden in the release tree: {relative}")
            continue
        if not path.is_file():
            continue
        files_scanned += 1
        if any(part.endswith(".egg-info") for part in relative.parts):
            findings.append(f"forbidden package-build state: {relative}")
            continue
        forbidden_environment = path.name.startswith(".env.") and path.name != ".env.example"
        if (
            path.name in FORBIDDEN_NAMES
            or forbidden_environment
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            findings.append(f"forbidden local-state filename: {relative}")
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            findings.append(f"file exceeds 5 MiB release limit: {relative}")
            continue
        data = path.read_bytes()
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(data):
                findings.append(f"{label}: {relative}")

    _audit_metadata(findings)
    if findings:
        print("Release audit failed:")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print(f"✓ release audit passed for {files_scanned} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
