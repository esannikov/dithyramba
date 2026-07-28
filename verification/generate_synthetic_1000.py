#!/usr/bin/env python3
"""Build the rights-safe deterministic 1,000-fragment verification payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

FRAGMENT_COUNT = 1_000
QUERY_STRIDE = 50
PROFILE = "dithyramba.synthetic_verification_1000/v1"
ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "synthetic_1000_manifest.json"


def canonical_bytes(value: Any) -> bytes:
    """Serialize the generator's JSON-only payload with the canonical profile."""

    normalized = _normalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_payload() -> dict[str, object]:
    """Return the complete deterministic verification payload."""

    fragments: list[dict[str, object]] = []
    for index in range(FRAGMENT_COUNT):
        suffix = f"{index:04d}"
        cohort = f"{index % 25:02d}"
        permuted = f"{(index * 37) % FRAGMENT_COUNT:04d}"
        text = (
            f"Synthetic verification fragment {suffix} carries unique token shard{suffix}, "
            f"cohort token cohort{cohort}, and deterministic payload token payload{permuted}."
        )
        fragments.append(
            {
                "source_fragment_id": f"fragment_verify_{suffix}",
                "source_id": f"source_verify_{suffix}",
                "source_ref": f"fixture://verification_1000/source/{suffix}",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    queries = [
        {
            "expected_source_refs": [f"fixture://verification_1000/source/{index:04d}"],
            "query_id": f"query_verify_{index:04d}",
            "text": f"shard{index:04d}",
        }
        for index in range(0, FRAGMENT_COUNT, QUERY_STRIDE)
    ]
    return {
        "authorship": "repository-authored synthetic data",
        "fragment_count": FRAGMENT_COUNT,
        "fragments": fragments,
        "generator_profile": PROFILE,
        "license": "CC0-1.0",
        "queries": queries,
        "schema": "dithyramba.synthetic_verification_corpus/1.0",
    }


def payload_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def check_manifest(payload: dict[str, object]) -> str:
    """Fail closed unless the generated payload matches the frozen manifest."""

    manifest = _load_canonical(MANIFEST_PATH)
    expected = manifest["expected"]
    actual_hash = payload_hash(payload)
    fragments = payload["fragments"]
    queries = payload["queries"]
    if not isinstance(fragments, list) or not isinstance(queries, list):
        raise ValueError("generated verification lists are malformed")
    checks = {
        "fragment_count": len(fragments),
        "first_fragment_id": fragments[0]["source_fragment_id"],
        "last_fragment_id": fragments[-1]["source_fragment_id"],
        "payload_canonical_sha256": actual_hash,
        "query_count": len(queries),
    }
    if checks != expected:
        raise ValueError(f"verification manifest mismatch: expected {expected!r}, got {checks!r}")
    return actual_hash


def _load_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_bytes(value) + b"\n":
        raise ValueError(f"manifest is not canonical JSON plus LF: {path}")
    return value


def _normalize(value: Any) -> Any:
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        return unicodedata.normalize("NFC", value)
    if type(value) is list:
        return [_normalize(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {unicodedata.normalize("NFC", key): _normalize(item) for key, item in value.items()}
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the frozen hash only")
    parser.add_argument("--output", type=Path, help="write canonical JSON to this path")
    args = parser.parse_args()
    if args.check and args.output is not None:
        parser.error("--check and --output are mutually exclusive")

    payload = build_payload()
    if args.check:
        digest = check_manifest(payload)
        print(f"synthetic_1000 ok fragments={FRAGMENT_COUNT} queries=20 sha256={digest}")
        return 0

    encoded = canonical_bytes(payload) + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(encoded)
    else:
        args.output.write_bytes(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
