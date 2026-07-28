from __future__ import annotations

import re
import uuid

import pytest

from dithyramba.contracts import (
    CanonicalizationError,
    ContractError,
    InvalidCanonicalKeyError,
    InvalidIdentifierPrefixError,
    InvalidPayloadError,
    NormalizationCollisionError,
    UnsupportedCanonicalTypeError,
    canonical_content_id,
    canonical_json_bytes,
    canonical_sha256_digest,
    canonical_sha256_hex,
    content_id,
    new_id,
    random_id,
    sha256_digest,
    sha256_hex,
)


def test_canonical_json_is_utf8_nfc_sorted_and_compact() -> None:
    decomposed = "e\u0301"
    value = {decomposed: ["Cafe\u0301", {"z": 2, "a": 1}], "b": True}

    encoded = canonical_json_bytes(value)

    assert encoded == '{"b":true,"é":["Café",{"a":1,"z":2}]}'.encode()
    assert encoded.decode("utf-8") == '{"b":true,"é":["Café",{"a":1,"z":2}]}'


def test_canonical_json_does_not_mutate_input() -> None:
    value = {"e\u0301": ["Cafe\u0301"]}

    canonical_json_bytes(value)

    assert list(value) == ["e\u0301"]
    assert value["e\u0301"] == ["Cafe\u0301"]


@pytest.mark.parametrize("value", [1.0, {"nested": [0.25]}, float("nan")])
def test_canonical_json_rejects_every_float(value: object) -> None:
    with pytest.raises(UnsupportedCanonicalTypeError, match="floats are forbidden"):
        canonical_json_bytes(value)


def test_canonical_json_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(InvalidCanonicalKeyError, match="must be str"):
        canonical_json_bytes({1: "one"})


def test_canonical_json_rejects_key_collision_after_nfc() -> None:
    with pytest.raises(NormalizationCollisionError, match="normalize"):
        canonical_json_bytes({"é": 1, "e\u0301": 2})


@pytest.mark.parametrize("value", [(1, 2), {1, 2}, b"bytes", object()])
def test_canonical_json_rejects_types_outside_strict_subset(value: object) -> None:
    with pytest.raises(UnsupportedCanonicalTypeError):
        canonical_json_bytes(value)


def test_canonical_json_wraps_cycles_in_typed_error() -> None:
    value: list[object] = []
    value.append(value)

    with pytest.raises(CanonicalizationError):
        canonical_json_bytes(value)


def test_canonical_json_rejects_cyclic_mapping() -> None:
    value: dict[str, object] = {}
    value["self"] = value

    with pytest.raises(CanonicalizationError, match="cyclic"):
        canonical_json_bytes(value)


def test_canonical_json_wraps_invalid_unicode_scalar() -> None:
    with pytest.raises(CanonicalizationError, match="serialized canonically"):
        canonical_json_bytes("\ud800")


def test_sha256_helpers_have_stable_known_output() -> None:
    payload = b'{"a":1}'
    expected = "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"

    assert sha256_hex(payload) == expected
    assert sha256_digest(payload) == f"sha256:{expected}"
    assert canonical_sha256_hex({"a": 1}) == expected
    assert canonical_sha256_digest({"a": 1}) == f"sha256:{expected}"


def test_canonical_hash_is_stable_across_order_and_unicode_form() -> None:
    left = {"z": "Cafe\u0301", "a": [2, 1]}
    right = {"a": [2, 1], "z": "Café"}

    assert canonical_sha256_digest(left) == canonical_sha256_digest(right)


def test_sha256_rejects_text_instead_of_implicitly_encoding_it() -> None:
    with pytest.raises(InvalidPayloadError, match="bytes-like"):
        sha256_hex("ambiguous text")  # type: ignore[arg-type]


def test_random_id_contains_valid_uuid4_and_prefix() -> None:
    identifier = random_id("source_version")
    prefix, uuid_hex = identifier.rsplit("_", 1)

    parsed = uuid.UUID(hex=uuid_hex)
    assert prefix == "source_version"
    assert parsed.version == 4
    assert parsed.variant == uuid.RFC_4122
    assert re.fullmatch(r"[0-9a-f]{32}", uuid_hex)


def test_new_id_uses_the_same_uuid4_contract() -> None:
    identifier = new_id("library")

    assert uuid.UUID(hex=identifier.removeprefix("library_")).version == 4


def test_content_id_uses_first_32_sha256_hex_characters() -> None:
    digest = sha256_hex(b"stable content")

    assert content_id("src", b"stable content") == f"src_{digest[:32]}"


def test_canonical_content_id_is_stable() -> None:
    left = canonical_content_id("packet", {"z": "e\u0301", "a": 1})
    right = canonical_content_id("packet", {"a": 1, "z": "é"})

    assert left == right
    assert re.fullmatch(r"packet_[0-9a-f]{32}", left)


@pytest.mark.parametrize(
    "prefix",
    ["", "Upper", "two--parts", "trailing_", "double__part", "9starts", "a" * 33, 1],
)
def test_identifier_helpers_reject_invalid_prefixes(prefix: object) -> None:
    with pytest.raises(InvalidIdentifierPrefixError):
        random_id(prefix)  # type: ignore[arg-type]


def test_all_contract_failures_share_typed_base() -> None:
    assert issubclass(CanonicalizationError, ContractError)
    assert issubclass(InvalidCanonicalKeyError, ContractError)
    assert issubclass(InvalidIdentifierPrefixError, ContractError)
    assert issubclass(InvalidPayloadError, ContractError)
