#!/usr/bin/env python3
"""Deterministically validate every committed rights-safe fixture."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from dithyramba.contracts import canonical_json_bytes, canonical_sha256_hex
from dithyramba.recall import FtsFragment, search_ephemeral_fts

ROOT = Path(__file__).resolve().parent
FROZEN_MANIFEST_HASHES = {
    "caillebotte_regression/manifest.json": (
        "19c8b49a5cf282602ddf2cd6ea682589fbf1da6771f32672fb24353f3bb8779e"
    ),
    "public_multilingual/manifest.json": (
        "cb3a5965b0cad98fd789ca9473bc98d2e9bccf1042aa168158fca6987047fcae"
    ),
    "synthetic_isolation/manifest.json": (
        "a54a2bc80efa687a0f1fe810d46e3f86ad825ff0b084dc88f9d949eeb173f047"
    ),
}


class FixtureValidationError(ValueError):
    """A committed fixture no longer satisfies its frozen contract."""


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_pairs_without_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FixtureValidationError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise FixtureValidationError(f"fixture root must be an object: {path}")
    if raw != canonical_json_bytes(value) + b"\n":
        raise FixtureValidationError(f"not canonical JSON plus one LF: {path}")
    return value


def _validate_manifest(relative_path: str) -> dict[str, dict[str, Any]]:
    manifest_path = ROOT / relative_path
    manifest = _load_canonical(manifest_path)
    actual_manifest_hash = canonical_sha256_hex(manifest)
    if actual_manifest_hash != FROZEN_MANIFEST_HASHES[relative_path]:
        raise FixtureValidationError(f"frozen manifest hash changed: {relative_path}")

    loaded: dict[str, dict[str, Any]] = {}
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list) or not artifact_rows:
        raise FixtureValidationError(f"manifest has no artifacts: {relative_path}")
    for artifact in artifact_rows:
        if not isinstance(artifact, dict):
            raise FixtureValidationError(f"invalid artifact row: {relative_path}")
        artifact_name = artifact.get("path")
        expected_hash = artifact.get("canonical_sha256")
        if (
            not isinstance(artifact_name, str)
            or not artifact_name
            or Path(artifact_name).name != artifact_name
        ):
            raise FixtureValidationError(f"unsafe artifact path: {artifact_name!r}")
        artifact_path = manifest_path.parent / artifact_name
        payload = _load_canonical(artifact_path)
        if canonical_sha256_hex(payload) != expected_hash:
            raise FixtureValidationError(f"artifact hash mismatch: {artifact_path}")
        loaded[artifact_name] = payload
    return loaded


def _validate_public_multilingual() -> tuple[Fraction, dict[str, Fraction]]:
    artifacts = _validate_manifest("public_multilingual/manifest.json")
    corpus = artifacts["corpus.json"]
    query_set = artifacts["queries.json"]
    fragments = corpus.get("fragments")
    queries = query_set.get("queries")
    if not isinstance(fragments, list) or len(fragments) != 12:
        raise FixtureValidationError("public_multilingual must contain 12 fragments")
    if not isinstance(queries, list) or len(queries) != 12:
        raise FixtureValidationError("public_multilingual must contain 12 queries")

    language_counts = Counter(query.get("language") for query in queries)
    if language_counts != Counter({"en": 6, "uk": 6}):
        raise FixtureValidationError("queries must contain exactly six en and six uk rows")
    if any(query.get("preregistered") is not True for query in queries):
        raise FixtureValidationError("every lexical query must be preregistered")

    fragment_ids: set[str] = set()
    source_ids: set[str] = set()
    source_refs: set[str] = set()
    ref_by_fragment: dict[str, str] = {}
    fts_fragments: list[FtsFragment] = []
    for fragment in fragments:
        if not isinstance(fragment, dict):
            raise FixtureValidationError("corpus fragment rows must be objects")
        fragment_id = _required_string(fragment, "source_fragment_id")
        source_id = _required_string(fragment, "source_id")
        source_ref = _required_string(fragment, "source_ref")
        text = _required_string(fragment, "text")
        text_hash = _required_string(fragment, "text_sha256")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != text_hash:
            raise FixtureValidationError(f"text hash mismatch: {fragment_id}")
        if fragment_id in fragment_ids or source_id in source_ids or source_ref in source_refs:
            raise FixtureValidationError("public corpus IDs and source refs must be unique")
        fragment_ids.add(fragment_id)
        source_ids.add(source_id)
        source_refs.add(source_ref)
        ref_by_fragment[fragment_id] = source_ref
        fts_fragments.append(FtsFragment(fragment_id, text, text_hash))

    query_ids: set[str] = set()
    scores: dict[str, list[Fraction]] = {"en": [], "uk": []}
    for query in queries:
        query_id = _required_string(query, "query_id")
        language = _required_string(query, "language")
        question = _required_string(query, "text")
        expected_refs = query.get("expected_source_refs")
        if query_id in query_ids:
            raise FixtureValidationError(f"duplicate query ID: {query_id}")
        query_ids.add(query_id)
        if (
            not isinstance(expected_refs, list)
            or not expected_refs
            or any(not isinstance(ref, str) or ref not in source_refs for ref in expected_refs)
            or len(set(expected_refs)) != len(expected_refs)
        ):
            raise FixtureValidationError(f"invalid expected source refs: {query_id}")
        result = search_ephemeral_fts(
            question=question,
            fragments=tuple(fts_fragments),
            max_candidates=10,
        )
        retrieved_refs = {
            ref_by_fragment[candidate.source_fragment_id] for candidate in result.trace
        }
        score = Fraction(len(retrieved_refs.intersection(expected_refs)), len(expected_refs))
        scores[language].append(score)

    language_scores = {
        language: sum(values, start=Fraction()) / len(values) for language, values in scores.items()
    }
    macro = sum(
        (score for values in scores.values() for score in values),
        start=Fraction(),
    ) / len(queries)
    metric = query_set.get("metric")
    if not isinstance(metric, dict) or metric.get("cutoff") != 10:
        raise FixtureValidationError("public metric must be recall at 10")
    language_gate = Fraction(_required_integer(metric, "minimum_language_recall_micros"), 1_000_000)
    macro_gate = Fraction(_required_integer(metric, "minimum_macro_recall_micros"), 1_000_000)
    if macro < macro_gate or any(score < language_gate for score in language_scores.values()):
        raise FixtureValidationError(
            f"recall gates failed: macro={macro}, languages={language_scores}"
        )
    return macro, language_scores


def _validate_caillebotte_regression() -> None:
    artifacts = _validate_manifest("caillebotte_regression/manifest.json")
    fixture = artifacts["fixture.json"]
    sources = fixture.get("sources")
    ledger_entries = fixture.get("ledger_entries")
    expected = fixture.get("expected_accounting")
    if not isinstance(sources, list) or len(sources) != 3:
        raise FixtureValidationError("caillebotte fixture must contain three source records")
    if not isinstance(ledger_entries, list) or len(ledger_entries) != 1:
        raise FixtureValidationError("caillebotte fixture must contain one AI ledger record")
    if not isinstance(expected, dict):
        raise FixtureValidationError("caillebotte expected accounting is missing")

    by_source_id: dict[str, dict[str, Any]] = {}
    root_families: set[str] = set()
    duplicates: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise FixtureValidationError("caillebotte source rows must be objects")
        source_id = _required_string(source, "source_id")
        text = _required_string(source, "text")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != source.get("content_sha256"):
            raise FixtureValidationError(f"source content hash mismatch: {source_id}")
        if source_id in by_source_id:
            raise FixtureValidationError(f"duplicate Source ID: {source_id}")
        by_source_id[source_id] = source
        if source.get("family_role") == "root":
            if source.get("root_source_id") != source_id:
                raise FixtureValidationError(f"root lineage mismatch: {source_id}")
            root_families.add(_required_string(source, "source_family_id"))
        elif source.get("family_role") == "duplicate":
            duplicates.add(source_id)
        else:
            raise FixtureValidationError(f"unexpected family role: {source_id}")

    for source_id in duplicates:
        duplicate = by_source_id[source_id]
        root_id = _required_string(duplicate, "root_source_id")
        root = by_source_id.get(root_id)
        if root is None or root.get("family_role") != "root":
            raise FixtureValidationError(f"duplicate has no declared root: {source_id}")
        if duplicate.get("source_family_id") != root.get("source_family_id") or duplicate.get(
            "content_sha256"
        ) != root.get("content_sha256"):
            raise FixtureValidationError(f"duplicate changed family or bytes: {source_id}")

    ledger_ids: set[str] = set()
    for entry in ledger_entries:
        if not isinstance(entry, dict):
            raise FixtureValidationError("AI ledger rows must be objects")
        entry_id = _required_string(entry, "entry_id")
        text = _required_string(entry, "text")
        if (
            entry.get("entry_kind") != "ai_generated_candidate"
            or entry.get("evidence_status") != "non_evidence"
        ):
            raise FixtureValidationError(f"AI ledger entry became evidence: {entry_id}")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != entry.get("text_sha256"):
            raise FixtureValidationError(f"AI ledger text hash mismatch: {entry_id}")
        ledger_ids.add(entry_id)

    expected_families = expected.get("eligible_independent_source_family_ids")
    expected_count = expected.get("independent_unit_count")
    if expected_families != sorted(root_families) or expected_count != len(root_families):
        raise FixtureValidationError("independent corroboration units are not root-family based")
    expected_non_independent = {
        row.get("id")
        for row in expected.get("non_independent_records", [])
        if isinstance(row, dict)
    }
    if expected_non_independent != duplicates.union(ledger_ids):
        raise FixtureValidationError("duplicate/AI non-independent accounting changed")


def _validate_synthetic_isolation() -> None:
    artifacts = _validate_manifest("synthetic_isolation/manifest.json")
    fixture = artifacts["libraries.json"]
    libraries = fixture.get("libraries")
    expectations = fixture.get("visibility_expectations")
    canaries = fixture.get("canaries")
    if not isinstance(libraries, list) or len(libraries) != 2:
        raise FixtureValidationError("synthetic isolation requires exactly two Libraries")
    if not isinstance(expectations, list) or len(expectations) != 2:
        raise FixtureValidationError("synthetic isolation requires two visibility scopes")
    if not isinstance(canaries, dict) or set(canaries) != {
        "excluded",
        "holdout",
        "private",
        "public",
    }:
        raise FixtureValidationError("synthetic isolation requires four named canaries")

    library_ids: set[str] = set()
    collection_ids: set[str] = set()
    fragment_ids: set[str] = set()
    rows: list[tuple[str, str, str, str, str]] = []
    observed_canaries: list[str] = []
    for library in libraries:
        if not isinstance(library, dict):
            raise FixtureValidationError("Library rows must be objects")
        library_id = _required_string(library, "library_id")
        if library_id in library_ids:
            raise FixtureValidationError(f"duplicate Library ID: {library_id}")
        library_ids.add(library_id)
        collections = library.get("collections")
        if not isinstance(collections, list):
            raise FixtureValidationError(f"Library collections are missing: {library_id}")
        for collection in collections:
            if not isinstance(collection, dict):
                raise FixtureValidationError("Collection rows must be objects")
            collection_id = _required_string(collection, "collection_id")
            state = _required_string(collection, "membership_state")
            if collection_id in collection_ids or state not in {"active", "excluded", "holdout"}:
                raise FixtureValidationError(f"invalid Collection metadata: {collection_id}")
            collection_ids.add(collection_id)
            fragments = collection.get("fragments")
            if not isinstance(fragments, list) or len(fragments) != 1:
                raise FixtureValidationError(
                    f"fixture Collection needs one fragment: {collection_id}"
                )
            fragment = fragments[0]
            if not isinstance(fragment, dict):
                raise FixtureValidationError("isolation fragment must be an object")
            fragment_id = _required_string(fragment, "source_fragment_id")
            canary = _required_string(fragment, "canary")
            text = _required_string(fragment, "text")
            if fragment_id in fragment_ids:
                raise FixtureValidationError(f"duplicate fragment ID: {fragment_id}")
            if canary not in text or hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest() != fragment.get("text_sha256"):
                raise FixtureValidationError(f"canary or hash mismatch: {fragment_id}")
            fragment_ids.add(fragment_id)
            observed_canaries.append(canary)
            rows.append((library_id, collection_id, state, fragment_id, canary))

    if sorted(observed_canaries) != sorted(canaries.values()):
        raise FixtureValidationError("named and embedded isolation canaries differ")
    if len(collection_ids) != 4 or len(fragment_ids) != 4:
        raise FixtureValidationError("isolation fixture counts changed")

    for expectation in expectations:
        if not isinstance(expectation, dict):
            raise FixtureValidationError("visibility expectation must be an object")
        library_id = _required_string(expectation, "library_id")
        requested = expectation.get("requested_collection_ids")
        expected_ids = expectation.get("expected_source_fragment_ids")
        forbidden = expectation.get("forbidden_canaries")
        if not isinstance(requested, list) or not isinstance(expected_ids, list):
            raise FixtureValidationError("visibility scope lists are missing")
        permitted_rows = [
            row
            for row in rows
            if row[0] == library_id and row[1] in requested and row[2] == "active"
        ]
        if sorted(row[3] for row in permitted_rows) != expected_ids:
            raise FixtureValidationError(
                f"visibility expectation mismatch: {expectation.get('scope_id')}"
            )
        if not isinstance(forbidden, list) or any(row[4] in forbidden for row in permitted_rows):
            raise FixtureValidationError(
                f"forbidden canary became visible: {expectation.get('scope_id')}"
            )


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise FixtureValidationError(f"{key} must be non-empty text")
    return value


def _required_integer(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise FixtureValidationError(f"{key} must be an integer")
    return value


def main() -> int:
    macro, languages = _validate_public_multilingual()
    _validate_caillebotte_regression()
    _validate_synthetic_isolation()
    print(
        "fixtures ok "
        f"public_multilingual recall@10 macro={float(macro):.3f} "
        f"uk={float(languages['uk']):.3f} en={float(languages['en']):.3f}; "
        "caillebotte independent_units=2; isolation libraries=2 canaries=4"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
