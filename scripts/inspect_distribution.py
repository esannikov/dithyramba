"""Verify that one sdist-to-wheel build exactly closes over the package source."""

from __future__ import annotations

import argparse
import hashlib
import stat
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = tuple(f"{index:04d}_" for index in range(1, 13))
FORBIDDEN_PARTS = {
    ".agent",
    ".git",
    ".github",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "htmlcov",
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
    ".pyc",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".wal",
}
REQUIRED_PROJECT_URLS = {
    "Homepage": "https://github.com/esannikov/dithyramba",
    "Issues": "https://github.com/esannikov/dithyramba/issues",
    "Repository": "https://github.com/esannikov/dithyramba",
}
REQUIRED_LICENSE_EXPRESSION = "Apache-2.0"


@dataclass(frozen=True, slots=True)
class Distribution:
    """One validated archive and its regular-file payloads."""

    path: Path
    kind: str
    files: dict[str, bytes]


def _safe_member_name(name: str, *, archive: Path) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise RuntimeError(f"{archive.name} contains an invalid member name: {name!r}")
    normalized = PurePosixPath(name)
    if normalized.is_absolute() or any(part in {"", ".", ".."} for part in normalized.parts):
        raise RuntimeError(f"{archive.name} contains an unsafe member path: {name!r}")
    return normalized


def _zip_files(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            _safe_member_name(member.filename, archive=path)
            unix_mode = member.external_attr >> 16
            if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
                raise RuntimeError(f"{path.name} contains a symbolic link: {member.filename}")
            if member.is_dir():
                continue
            if member.filename in files:
                raise RuntimeError(f"{path.name} contains a duplicate member: {member.filename}")
            files[member.filename] = archive.read(member)
    return files


def _tar_files(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            _safe_member_name(member.name, archive=path)
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise RuntimeError(f"{path.name} contains a non-regular link/device: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"{path.name} contains an unsupported member: {member.name}")
            if member.name in files:
                raise RuntimeError(f"{path.name} contains a duplicate member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"{path.name} member could not be read: {member.name}")
            files[member.name] = stream.read()
    return files


def _load(path: Path) -> Distribution:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.suffix == ".whl":
        return Distribution(resolved, "wheel", _zip_files(resolved))
    if resolved.name.endswith(".tar.gz"):
        return Distribution(resolved, "sdist", _tar_files(resolved))
    raise ValueError(f"unsupported distribution: {resolved}")


def _assert_clean(distribution: Distribution) -> None:
    bad: list[str] = []
    for name in distribution.files:
        normalized = PurePosixPath(name)
        if (
            FORBIDDEN_PARTS.intersection(normalized.parts)
            or any(part.endswith(".egg-info") for part in normalized.parts)
            or normalized.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            bad.append(name)
    if bad:
        preview = "\n".join(f"  - {name}" for name in bad[:20])
        raise RuntimeError(
            f"{distribution.path.name} contains forbidden local/release state:\n{preview}"
        )


def _source_package(source_root: Path) -> dict[str, bytes]:
    package_root = source_root / "src" / "dithyramba"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    return {
        path.relative_to(source_root / "src").as_posix(): path.read_bytes()
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _wheel_package(distribution: Distribution) -> dict[str, bytes]:
    return {
        name: payload
        for name, payload in distribution.files.items()
        if PurePosixPath(name).parts[0] == "dithyramba"
    }


def _sdist_root(distribution: Distribution, *, project_name: str, version: str) -> str:
    roots = {PurePosixPath(name).parts[0] for name in distribution.files}
    expected = f"{project_name.replace('-', '_')}-{version}"
    if roots != {expected}:
        raise RuntimeError(
            f"{distribution.path.name} must have the single root {expected!r}, "
            f"found {sorted(roots)!r}"
        )
    return expected


def _sdist_package(distribution: Distribution, *, archive_root: str) -> dict[str, bytes]:
    prefix = f"{archive_root}/src/"
    return {
        name.removeprefix(prefix): payload
        for name, payload in distribution.files.items()
        if name.startswith(f"{prefix}dithyramba/")
    }


def _preview(values: set[str]) -> str:
    return ", ".join(sorted(values)[:12]) or "none"


def _assert_exact_payload(
    expected: dict[str, bytes],
    actual: dict[str, bytes],
    *,
    label: str,
) -> None:
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    changed = {name for name in expected.keys() & actual.keys() if expected[name] != actual[name]}
    if missing or extra or changed:
        raise RuntimeError(
            f"{label} differs from src/dithyramba: "
            f"missing=[{_preview(missing)}] extra=[{_preview(extra)}] "
            f"byte_changed=[{_preview(changed)}]"
        )


def _message(payload: bytes) -> Message:
    return BytesParser(policy=policy.compat32).parsebytes(payload)


def _single_file(distribution: Distribution, suffix: str) -> bytes:
    matches = [payload for name, payload in distribution.files.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"{distribution.path.name} must contain exactly one {suffix}, found {len(matches)}"
        )
    return matches[0]


def _assert_metadata(wheel: Distribution, sdist: Distribution, project: dict[str, object]) -> None:
    name = str(project["name"])
    version = str(project["version"])
    wheel_metadata = _message(_single_file(wheel, ".dist-info/METADATA"))
    sdist_metadata = _message(_single_file(sdist, "/PKG-INFO"))

    for label, metadata in (("wheel", wheel_metadata), ("sdist", sdist_metadata)):
        if metadata.get("Name") != name or metadata.get("Version") != version:
            raise RuntimeError(f"{label} Name/Version metadata differs from pyproject.toml")
        if metadata.get("Requires-Python") != project["requires-python"]:
            raise RuntimeError(f"{label} Requires-Python metadata differs from pyproject.toml")
        if metadata.get("License"):
            raise RuntimeError(f"{label} contains a deprecated free-text License field")
        if metadata.get("License-Expression") != REQUIRED_LICENSE_EXPRESSION:
            raise RuntimeError(f"{label} license expression is not Apache-2.0")
        if metadata.get_all("License-File", []) != ["LICENSE"]:
            raise RuntimeError(f"{label} license-file metadata differs from LICENSE")

        urls: dict[str, str] = {}
        for item in metadata.get_all("Project-URL", []):
            key, separator, value = item.partition(",")
            if not separator:
                raise RuntimeError(f"{label} contains malformed Project-URL metadata: {item!r}")
            urls[key.strip()] = value.strip()
        if urls != REQUIRED_PROJECT_URLS:
            raise RuntimeError(f"{label} project URLs differ from the public release contract")

        requirements = metadata.get_all("Requires-Dist", [])
        if not any(item.startswith("defusedxml==0.7.1") for item in requirements):
            raise RuntimeError(f"{label} is missing the defusedxml runtime dependency")
        if "semantic" not in metadata.get_all("Provides-Extra", []):
            raise RuntimeError(f"{label} is missing the semantic extra")
        if not any(
            "sentence-transformers==5.6.0" in item and "semantic" in item for item in requirements
        ):
            raise RuntimeError(f"{label} is missing the pinned semantic dependency")

    wheel_headers = _message(_single_file(wheel, ".dist-info/WHEEL"))
    if wheel_headers.get("Generator") != "hatchling 1.27.0":
        raise RuntimeError("wheel was not generated by the pinned hatchling 1.27.0 backend")
    if wheel_headers.get("Root-Is-Purelib") != "true":
        raise RuntimeError("wheel is not marked as a pure-Python distribution")
    if wheel_headers.get_all("Tag", []) != ["py3-none-any"]:
        raise RuntimeError("wheel tag is not exactly py3-none-any")


def _assert_runtime_closure(package: dict[str, bytes]) -> None:
    required = {
        "dithyramba/py.typed",
        "dithyramba/api/templates/reading_room.html",
        "dithyramba/api/static/reading_room.css",
        "dithyramba/api/templates/research_atlas.html",
        "dithyramba/api/static/research_atlas.css",
        "dithyramba/store/sql/__init__.py",
        "dithyramba/persistence/answer_projection.py",
        "dithyramba/recall/compatibility.py",
        "dithyramba/connectors/books.py",
    }
    missing = required - set(package)
    for prefix in MIGRATIONS:
        if not any(
            PurePosixPath(name).parent.as_posix() == "dithyramba/store/sql"
            and PurePosixPath(name).name.startswith(prefix)
            for name in package
        ):
            missing.add(f"dithyramba/store/sql/{prefix}*.sql")
    if missing:
        formatted = "\n".join(f"  - {name}" for name in sorted(missing))
        raise RuntimeError(f"distribution misses required runtime files:\n{formatted}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_acceptance_script_executable(sdist: Distribution, *, archive_root: str) -> None:
    expected = f"{archive_root}/scripts/acceptance.sh"
    with tarfile.open(sdist.path, "r:gz") as archive:
        member = archive.getmember(expected)
    if member.mode & 0o111 == 0:
        raise RuntimeError("sdist did not preserve the executable acceptance entry point")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("distributions", nargs=2, type=Path)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve(strict=True)
    config = tomllib.loads((source_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    distributions = [_load(path) for path in args.distributions]
    by_kind = {distribution.kind: distribution for distribution in distributions}
    if set(by_kind) != {"wheel", "sdist"}:
        raise RuntimeError("provide exactly one wheel and one .tar.gz source distribution")
    wheel = by_kind["wheel"]
    sdist = by_kind["sdist"]

    for distribution in distributions:
        _assert_clean(distribution)

    source_package = _source_package(source_root)
    wheel_package = _wheel_package(wheel)
    archive_root = _sdist_root(
        sdist,
        project_name=str(project["name"]),
        version=str(project["version"]),
    )
    sdist_package = _sdist_package(sdist, archive_root=archive_root)
    _assert_exact_payload(source_package, sdist_package, label="sdist package payload")
    _assert_exact_payload(source_package, wheel_package, label="wheel package payload")
    _assert_runtime_closure(wheel_package)
    _assert_metadata(wheel, sdist, project)

    source_license = (source_root / "LICENSE").read_bytes()
    sdist_license_name = f"{archive_root}/LICENSE"
    if sdist.files.get(sdist_license_name) != source_license:
        raise RuntimeError("sdist does not contain the exact approved LICENSE")
    wheel_license_matches = [
        payload
        for name, payload in wheel.files.items()
        if name.endswith(".dist-info/licenses/LICENSE")
    ]
    if wheel_license_matches != [source_license]:
        raise RuntimeError("wheel does not contain the exact approved LICENSE")

    required_sdist_files = {
        "CITATION.cff",
        "docs/REPOSITORY_GUIDE.md",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "scripts/acceptance.sh",
        "scripts/demo.py",
        "scripts/inspect_distribution.py",
        "scripts/release_audit.py",
        "uv.lock",
        "verification/README.md",
        "verification/__init__.py",
        "verification/generate_synthetic_1000.py",
        "verification/run_public_replay.py",
        "verification/synthetic_1000_manifest.json",
    }
    required_sdist_members = {f"{archive_root}/{relative}" for relative in required_sdist_files}
    missing_sdist_members = required_sdist_members - set(sdist.files)
    if missing_sdist_members:
        raise RuntimeError(f"sdist misses release metadata: {sorted(missing_sdist_members)!r}")
    changed_sdist_members = {
        relative
        for relative in required_sdist_files
        if sdist.files[f"{archive_root}/{relative}"] != (source_root / relative).read_bytes()
    }
    if changed_sdist_members:
        raise RuntimeError(f"sdist changed release inputs: {sorted(changed_sdist_members)!r}")
    _assert_acceptance_script_executable(sdist, archive_root=archive_root)

    for distribution in distributions:
        print(
            f"✓ {distribution.path.name}: {len(distribution.files)} files, "
            f"sha256={_sha256(distribution.path)}"
        )
    print(f"✓ exact source/sdist/wheel package closure: {len(source_package)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
