from __future__ import annotations

import sqlite3

import pytest

from dithyramba.meaning import (
    GeneratorProfile,
    MeaningOutputManifest,
    MeaningPersistenceError,
    MeaningReviewScope,
    OmissionCategory,
    ReviewBudget,
)
from dithyramba.persistence.meaning import (
    SQLiteMeaningRepository,
    _canonical_text,
    _generator_from_payload,
    _load_array,
    _load_object,
    _omission_from_payload,
    _output_from_payload,
    _require_content_row,
    _review_budget_from_payload,
    _scope_from_payload,
    _strict_bool,
    _strict_int,
    _strict_string,
)


def test_repository_requires_library_repository() -> None:
    with pytest.raises(TypeError, match="LibraryRepository"):
        SQLiteMeaningRepository(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("loader", "value", "message"),
    [
        (_load_object, "{", "invalid JSON"),
        (_load_object, "[]", "must be an object"),
        (_load_object, '{"b":2, "a":1}', "not canonical JSON"),
        (_load_array, "[", "invalid JSON"),
        (_load_array, "{}", "must be an array"),
        (_load_array, "[1, 2]", "not canonical JSON"),
    ],
)
def test_canonical_json_loaders_reject_corruption(
    loader: object,
    value: str,
    message: str,
) -> None:
    with pytest.raises(MeaningPersistenceError, match=message):
        if loader is _load_object:
            _load_object(value, "test")
        else:
            _load_array(value, "test")


def test_canonical_json_loaders_accept_exact_encoding() -> None:
    assert _load_object('{"a":1}', "test") == {"a": 1}
    assert _load_array('[1,"a"]', "test") == [1, "a"]
    assert _canonical_text({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_output_payload_round_trip_and_rejects_shape_types_and_ids() -> None:
    valid = MeaningOutputManifest(
        voice_ids=("voice_one",),
        entity_ids=("entity_one",),
        entity_mention_ids=("entity_mention_one",),
        concept_ids=("concept_one",),
        concept_mention_ids=("concept_mention_one",),
        concept_meaning_ids=("concept_meaning_one",),
        time_context_ids=("time_context_one",),
        statement_ids=("statement_one",),
        evidence_link_ids=("evidence_one",),
    )
    assert _output_from_payload(valid.payload()) == valid

    with pytest.raises(MeaningPersistenceError, match="shape"):
        _output_from_payload({})

    wrong_type = valid.payload()
    wrong_type["voice_ids"] = [1]
    with pytest.raises(MeaningPersistenceError, match="IDs"):
        _output_from_payload(wrong_type)

    wrong_prefix = valid.payload()
    wrong_prefix["voice_ids"] = ["statement_one"]
    with pytest.raises(MeaningPersistenceError, match="manifest is invalid"):
        _output_from_payload(wrong_prefix)


def test_generator_payload_round_trip_and_rejects_shape_types_and_contract() -> None:
    without_dimensions = GeneratorProfile("local", "1", "internal", "p1")
    with_dimensions = GeneratorProfile(
        "remote",
        "2",
        "licensed",
        "p2",
        dimensions=8,
        external_provider=True,
    )
    assert _generator_from_payload(without_dimensions.payload()) == without_dimensions
    assert _generator_from_payload(with_dimensions.payload()) == with_dimensions

    with pytest.raises(MeaningPersistenceError, match="shape"):
        _generator_from_payload({})

    bad_type = with_dimensions.payload()
    bad_type["generator_id"] = 1
    with pytest.raises(MeaningPersistenceError, match="must be text"):
        _generator_from_payload(bad_type)

    bad_contract = with_dimensions.payload()
    bad_contract["generator_id"] = " "
    with pytest.raises(MeaningPersistenceError, match="receipt is invalid"):
        _generator_from_payload(bad_contract)


def test_review_budget_payload_round_trip_and_rejects_shape_types_and_contract() -> None:
    budget = ReviewBudget(
        max_candidates=10,
        max_read_fragments=20,
        max_unresolved_candidates=30,
        review_decision_limit=40,
    )
    assert _review_budget_from_payload(budget.payload()) == budget

    with pytest.raises(MeaningPersistenceError, match="shape"):
        _review_budget_from_payload({})

    bad_type = budget.payload()
    bad_type["max_candidates"] = True
    with pytest.raises(MeaningPersistenceError, match="integer"):
        _review_budget_from_payload(bad_type)

    bad_contract = budget.payload()
    bad_contract["max_candidates"] = 0
    with pytest.raises(MeaningPersistenceError, match="ReviewBudget is invalid"):
        _review_budget_from_payload(bad_contract)


def test_omission_payload_round_trip_and_rejects_corruption() -> None:
    omission = _omission_from_payload({"category": "budget", "reason_code": "bounded", "count": 1})
    assert omission.category is OmissionCategory.BUDGET
    assert omission.reason_code == "bounded"

    with pytest.raises(MeaningPersistenceError, match="must be an object"):
        _omission_from_payload([])
    with pytest.raises(MeaningPersistenceError, match="shape"):
        _omission_from_payload({})
    with pytest.raises(MeaningPersistenceError, match="omission is invalid"):
        _omission_from_payload({"category": "unknown", "reason_code": "bounded", "count": 1})


def test_review_scope_payload_round_trip_and_rejects_corruption() -> None:
    scope = MeaningReviewScope(("collection_one",), "research")
    assert _scope_from_payload(scope.payload()) == scope

    with pytest.raises(MeaningPersistenceError, match="shape"):
        _scope_from_payload({})

    wrong_collections = scope.payload()
    wrong_collections["collection_ids"] = [1]
    with pytest.raises(MeaningPersistenceError, match="Collections"):
        _scope_from_payload(wrong_collections)

    empty = scope.payload()
    empty["collection_ids"] = []
    with pytest.raises(MeaningPersistenceError, match="ReviewScope is invalid"):
        _scope_from_payload(empty)


def test_strict_persisted_scalar_parsers() -> None:
    assert _strict_string("value") == "value"
    assert _strict_int(1) == 1
    assert _strict_bool(1) is True
    assert _strict_bool(False) is False
    with pytest.raises(MeaningPersistenceError, match="text"):
        _strict_string(1)
    with pytest.raises(MeaningPersistenceError, match="integer"):
        _strict_int(True)
    with pytest.raises(MeaningPersistenceError, match="boolean"):
        _strict_bool(2)


def test_content_addressed_row_verification() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE voices(voice_id TEXT, content_hash TEXT)")
    connection.execute("INSERT INTO voices VALUES ('voice_one', 'expected')")
    _require_content_row(connection, "voices", "voice_id", "voice_one", "expected")

    with pytest.raises(MeaningPersistenceError, match="invalid typed"):
        _require_content_row(connection, "unsafe", "id", "voice_one", "expected")
    with pytest.raises(MeaningPersistenceError, match="inconsistent"):
        _require_content_row(connection, "voices", "voice_id", "voice_missing", "expected")
    with pytest.raises(MeaningPersistenceError, match="inconsistent"):
        _require_content_row(connection, "voices", "voice_id", "voice_one", "wrong")
