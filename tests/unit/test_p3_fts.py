from __future__ import annotations

import json
import re
import socket
import sqlite3
import urllib.request
from dataclasses import replace
from decimal import Decimal
from typing import Any, NoReturn, cast

import pytest

from dithyramba.contracts import sha256_hex
from dithyramba.recall import (
    FTS_SCORE_CONTRACT,
    FTS_TOKENIZER,
    MAX_QUERY_TOKENS,
    FtsCandidate,
    FtsContractError,
    FtsFragment,
    FtsSearchResult,
    FtsUnavailableError,
    PermittedFtsSession,
    current_fts_runtime_profile,
    format_fts_score,
    search_ephemeral_fts,
)


def _fragment(identifier: str, text: str) -> FtsFragment:
    return FtsFragment(
        source_fragment_id=identifier,
        text=text,
        text_sha256=sha256_hex(text.encode("utf-8")),
    )


def _search(
    question: str,
    fragments: tuple[FtsFragment, ...],
    *,
    max_candidates: int = 10,
) -> FtsSearchResult:
    return search_ephemeral_fts(
        question=question,
        fragments=fragments,
        max_candidates=max_candidates,
    )


def _serialized_result(result: FtsSearchResult) -> bytes:
    return json.dumps(
        result.semantic_payload(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_ua_en_and_remove_diacritics_use_the_frozen_sqlite_tokenizer() -> None:
    result = _search(
        "українське CAFÉ archive",
        (
            _fragment("fragment_en", "A visual archive of performance."),
            _fragment("fragment_fr", "A cafe research notebook."),
            _fragment("fragment_ua", "Українське мистецтво і пам'ять."),  # noqa: RUF001
        ),
    )

    assert result.query_tokens == ("українське", "cafe", "archive")
    assert result.candidate_count == 3
    assert {candidate.source_fragment_id for candidate in result.trace} == {
        "fragment_en",
        "fragment_fr",
        "fragment_ua",
    }
    assert result.profile.tokenizer == FTS_TOKENIZER
    assert result.profile.score_contract == FTS_SCORE_CONTRACT


def test_operator_like_question_is_reduced_to_literal_tokens() -> None:
    result = _search(
        '"alpha" OR beta* NOT gamma NEAR(delta)',
        (
            _fragment("fragment_alpha", "alpha only"),
            _fragment("fragment_beta", "beta only"),
            _fragment("fragment_unrelated", "epsilon only"),
        ),
    )

    assert result.query_tokens == ("alpha", "or", "beta", "not", "gamma", "near", "delta")
    assert {candidate.source_fragment_id for candidate in result.trace} == {
        "fragment_alpha",
        "fragment_beta",
    }


def test_punctuation_only_question_has_no_match_tokens_or_candidates() -> None:
    result = _search('?! -- () [] {} ""', (_fragment("fragment_a", "anything"),))

    assert result.query_tokens == ()
    assert result.candidate_count == 0
    assert result.trace == ()


def test_query_token_count_is_capped_before_match_construction() -> None:
    question = " ".join(f"token{index}" for index in range(70))
    result = _search(
        question,
        (_fragment("fragment_a", "token63 token64"),),
    )

    assert len(result.query_tokens) == MAX_QUERY_TOKENS
    assert result.query_tokens[-1] == "token63"
    assert "token64" not in result.query_tokens
    assert result.candidate_count == 1


def test_query_token_cap_is_applied_after_first_occurrence_deduplication() -> None:
    repeated_prefix = " ".join("repeat" for _ in range(MAX_QUERY_TOKENS + 10))
    result = _search(
        f"{repeated_prefix} late",
        (_fragment("fragment_late", "late evidence"),),
    )

    assert result.query_tokens == ("repeat", "late")
    assert result.candidate_count == 1


def test_raw_bm25_ties_use_binary_fragment_id_order() -> None:
    result = _search(
        "shared",
        (
            _fragment("fragment_z", "shared exact text"),
            _fragment("fragment_a", "shared exact text"),
        ),
    )

    assert [candidate.source_fragment_id for candidate in result.trace] == [
        "fragment_a",
        "fragment_z",
    ]
    assert result.trace[0].score == result.trace[1].score


def test_distinct_raw_scores_that_quantize_equal_preserve_sql_order() -> None:
    result = _search(
        "alpha",
        (
            _fragment("fragment_z", "alpha alpha"),
            _fragment("fragment_a", "alpha"),
        ),
    )

    assert [candidate.source_fragment_id for candidate in result.trace] == [
        "fragment_z",
        "fragment_a",
    ]
    assert result.trace[0].score == result.trace[1].score


def test_scores_are_half_even_fixed_six_decimals_without_negative_zero() -> None:
    assert format_fts_score(Decimal("1.2345665")) == "1.234566"
    assert format_fts_score(Decimal("1.2345675")) == "1.234568"
    assert format_fts_score(Decimal("-0.0000004")) == "0.000000"
    assert format_fts_score(Decimal("-0.0000005")) == "0.000000"
    assert format_fts_score(Decimal("-0.0000006")) == "-0.000001"

    result = _search("alpha", (_fragment("fragment_a", "alpha"),))
    assert re.fullmatch(r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}", result.trace[0].score)
    assert result.trace[0].score != "-0.000000"


@pytest.mark.parametrize("value", [True, "1.0", float("nan"), float("inf")])
def test_score_formatter_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(FtsContractError):
        format_fts_score(value)


@pytest.mark.parametrize("value", [Decimal("1e1000"), Decimal("1e1000000")])
def test_score_formatter_wraps_quantize_range_failures(value: Decimal) -> None:
    with pytest.raises(FtsContractError, match="supported decimal range"):
        format_fts_score(value)


def test_search_result_requires_monotonic_serialized_scores() -> None:
    base = _search("alpha", (_fragment("fragment_base", "alpha"),))

    with pytest.raises(FtsContractError, match="monotonic"):
        replace(
            base,
            candidate_count=2,
            trace=(
                FtsCandidate("fragment_a", 1, "-1.000000"),
                FtsCandidate("fragment_b", 2, "-2.000000"),
            ),
        )


def test_search_result_allows_equal_quantized_scores_without_id_tie_break() -> None:
    base = _search("alpha", (_fragment("fragment_base", "alpha"),))

    result = replace(
        base,
        candidate_count=2,
        trace=(
            FtsCandidate("fragment_z", 1, "-1.000000"),
            FtsCandidate("fragment_a", 2, "-1.000000"),
        ),
    )

    assert [candidate.source_fragment_id for candidate in result.trace] == [
        "fragment_z",
        "fragment_a",
    ]


def test_search_result_rejects_duplicate_ids_and_inconsistent_counts() -> None:
    base = _search("alpha", (_fragment("fragment_base", "alpha"),))
    duplicate_trace = (
        FtsCandidate("fragment_duplicate", 1, "-2.000000"),
        FtsCandidate("fragment_duplicate", 2, "-1.000000"),
    )

    with pytest.raises(FtsContractError, match="IDs must be unique"):
        replace(base, candidate_count=2, trace=duplicate_trace)
    with pytest.raises(FtsContractError, match="candidate counts"):
        replace(
            base,
            candidate_count=1,
            trace=(
                FtsCandidate("fragment_a", 1, "-2.000000"),
                FtsCandidate("fragment_b", 2, "-1.000000"),
            ),
        )
    with pytest.raises(FtsContractError, match="non-negative integer"):
        replace(base, candidate_count=True)


def test_fragment_contract_rejects_invalid_ids_text_and_hashes() -> None:
    with pytest.raises(FtsContractError, match="fragment_"):
        _fragment("fragment_A", "text")
    with pytest.raises(FtsContractError, match="NFC"):
        _fragment("fragment_a", "e\u0301")
    with pytest.raises(FtsContractError, match="does not match"):
        FtsFragment("fragment_a", "text", "a" * 64)


def test_search_rejects_duplicate_or_untyped_fragments_and_invalid_limits() -> None:
    fragment = _fragment("fragment_a", "alpha")
    with pytest.raises(FtsContractError, match="unique"):
        _search("alpha", (fragment, fragment))
    with pytest.raises(FtsContractError, match="tuple"):
        search_ephemeral_fts(
            question="alpha",
            fragments=cast(Any, [fragment]),
            max_candidates=10,
        )
    for limit in (0, 501, True, 1.0):
        with pytest.raises(FtsContractError, match="max_candidates"):
            search_ephemeral_fts(
                question="alpha",
                fragments=(fragment,),
                max_candidates=cast(Any, limit),
            )


def test_question_requires_the_query_request_nfc_boundary() -> None:
    fragment = _fragment("fragment_a", "é")
    for question in (" e", "e ", "e\x00x", "e\u0301", "x" * 2_001):
        with pytest.raises(FtsContractError, match="question"):
            _search(question, (fragment,))


def test_rebuilding_from_the_same_manifest_is_content_identical() -> None:
    first_fragment = _fragment("fragment_b", "memory archive")
    second_fragment = _fragment("fragment_a", "archive method")
    first = _search("archive", (first_fragment, second_fragment))
    second = _search("archive", (second_fragment, first_fragment))

    assert first.retrieval_corpus_hash == second.retrieval_corpus_hash
    assert first.profile_version == second.profile_version
    assert first.trace == second.trace
    assert first.semantic_payload() == second.semantic_payload()
    assert first.result_hash == second.result_hash


def test_permitted_session_matches_the_one_shot_result_byte_for_byte() -> None:
    fragments = (
        _fragment("fragment_beta", "beta archive"),
        _fragment("fragment_alpha", "alpha beta archive"),
    )
    one_shot = _search("alpha archive", fragments, max_candidates=2)

    with PermittedFtsSession(fragments=fragments) as session:
        reused = session.search(question="alpha archive", max_candidates=2)

    assert _serialized_result(reused) == _serialized_result(one_shot)
    assert reused.result_hash == one_shot.result_hash


def test_permitted_session_reuses_one_index_without_query_token_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    def connect(database: str) -> sqlite3.Connection:
        connection = real_connect(database)
        connections.append(connection)
        return connection

    monkeypatch.setattr("dithyramba.recall.fts.sqlite3.connect", connect)
    fragments = (
        _fragment("fragment_alpha", "alpha evidence"),
        _fragment("fragment_beta", "beta evidence"),
        _fragment("fragment_gamma", "gamma evidence"),
    )

    with PermittedFtsSession(fragments=fragments) as session:
        q0 = session.search(question="alpha alpha", max_candidates=3)
        q1 = session.search(question="beta", max_candidates=3)
        q2 = session.search(question="gamma", max_candidates=3)

        assert len(connections) == 1
        assert q0.query_tokens == ("alpha",)
        assert q1.query_tokens == ("beta",)
        assert q2.query_tokens == ("gamma",)
        assert [candidate.source_fragment_id for candidate in q0.trace] == ["fragment_alpha"]
        assert [candidate.source_fragment_id for candidate in q1.trace] == ["fragment_beta"]
        assert [candidate.source_fragment_id for candidate in q2.trace] == ["fragment_gamma"]
        assert q0.retrieval_corpus_hash == q1.retrieval_corpus_hash == q2.retrieval_corpus_hash
        assert q0.profile == q1.profile == q2.profile

    assert session.closed is True


def test_permitted_session_explicit_close_is_idempotent_and_blocks_reuse() -> None:
    session = PermittedFtsSession(fragments=(_fragment("fragment_alpha", "alpha evidence"),))
    initially_closed = session.closed
    assert initially_closed is False

    session.close()
    session.close()

    assert session.closed is True
    with pytest.raises(FtsContractError, match="closed"):
        session.search(question="alpha", max_candidates=1)
    with pytest.raises(FtsContractError, match="cannot be reopened"):
        session.__enter__()


def test_permitted_session_context_closes_when_the_body_raises() -> None:
    session = PermittedFtsSession(fragments=(_fragment("fragment_alpha", "alpha evidence"),))

    with pytest.raises(RuntimeError, match="body failed"), session:
        raise RuntimeError("body failed")

    assert session.closed is True


def test_candidate_count_covers_all_matches_while_trace_is_bounded() -> None:
    fragments = tuple(
        _fragment(f"fragment_{index}", f"shared document {index}") for index in range(7)
    )
    result = _search("shared", fragments, max_candidates=2)

    assert result.candidate_count == 7
    assert len(result.trace) == 2


def test_omitted_forbidden_fragment_never_affects_index_trace_or_result_payload() -> None:
    permitted = (_fragment("fragment_permitted", "archive methods"),)
    forbidden_id = "fragment_forbidden"
    forbidden_canary = "FORBIDDEN_CANARY_MUST_NOT_APPEAR"

    result = _search("archive", permitted)
    serialized = json.dumps(result.semantic_payload(), sort_keys=True)

    assert result.candidate_count == 1
    assert forbidden_id not in serialized
    assert forbidden_canary not in serialized


def test_search_has_no_socket_or_provider_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden_call(*_args: object, **_kwargs: object) -> NoReturn:
        calls.append("egress")
        raise AssertionError("ephemeral FTS attempted external egress")

    monkeypatch.setattr(socket, "create_connection", forbidden_call)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_call)

    result = _search("local", (_fragment("fragment_local", "local only"),))

    assert result.candidate_count == 1
    assert calls == []


def test_runtime_profile_records_sqlite_and_full_compile_options_hash() -> None:
    result = _search("alpha", (_fragment("fragment_a", "alpha"),))

    assert result.profile.sqlite_version == sqlite3.sqlite_version
    assert re.fullmatch(r"[0-9a-f]{64}", result.profile.compile_options_hash)
    assert FTS_TOKENIZER in result.profile.payload().values()
    assert FTS_SCORE_CONTRACT in result.profile.payload().values()
    assert result.profile.compile_options_hash in result.profile_version
    assert result.profile.sqlite_version in result.profile_version
    assert len(result.profile_version) <= 128


def test_runtime_profile_probe_matches_search_without_corpus_materialization() -> None:
    probed = current_fts_runtime_profile()
    searched = _search("alpha", (_fragment("fragment_a", "alpha"),)).profile

    assert probed == searched
    assert probed.profile_version == searched.profile_version


class _ConnectionSpy:
    def __init__(
        self,
        wrapped: sqlite3.Connection,
        *,
        fail_create: bool = False,
        fail_query: bool = False,
    ) -> None:
        self.wrapped = wrapped
        self.fail_create = fail_create
        self.fail_query = fail_query
        self.closed = False

    def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
        if self.fail_create and "CREATE VIRTUAL TABLE permitted_fts" in sql:
            raise sqlite3.OperationalError("simulated FTS failure")
        if self.fail_query and "SELECT count(*) FROM permitted_fts" in sql:
            raise sqlite3.OperationalError("simulated query failure")
        return self.wrapped.execute(sql, cast(Any, parameters))

    def executemany(self, sql: str, parameters: object) -> sqlite3.Cursor:
        return self.wrapped.executemany(sql, cast(Any, parameters))

    def close(self) -> None:
        self.closed = True
        self.wrapped.close()


@pytest.mark.parametrize("fail_create", [False, True])
def test_ephemeral_connection_is_closed_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    fail_create: bool,
) -> None:
    real_connect = sqlite3.connect
    spies: list[_ConnectionSpy] = []

    def connect(database: str) -> Any:
        assert database == ":memory:"
        spy = _ConnectionSpy(real_connect(database), fail_create=fail_create)
        spies.append(spy)
        return spy

    monkeypatch.setattr("dithyramba.recall.fts.sqlite3.connect", connect)
    if fail_create:
        with pytest.raises(FtsUnavailableError):
            _search("alpha", (_fragment("fragment_a", "alpha"),))
    else:
        assert _search("alpha", (_fragment("fragment_a", "alpha"),)).candidate_count == 1
    assert len(spies) == 1
    assert spies[0].closed is True


def test_permitted_session_runtime_failure_closes_and_blocks_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_connect = sqlite3.connect
    spies: list[_ConnectionSpy] = []

    def connect(database: str) -> Any:
        spy = _ConnectionSpy(real_connect(database), fail_query=True)
        spies.append(spy)
        return spy

    monkeypatch.setattr("dithyramba.recall.fts.sqlite3.connect", connect)
    session = PermittedFtsSession(fragments=(_fragment("fragment_alpha", "alpha evidence"),))

    with pytest.raises(FtsUnavailableError, match="failed closed"):
        session.search(question="alpha", max_candidates=1)

    assert session.closed is True
    assert spies[0].closed is True
    with pytest.raises(FtsContractError, match="closed"):
        session.search(question="alpha", max_candidates=1)
