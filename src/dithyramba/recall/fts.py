"""Fresh permitted-only SQLite FTS5 recall.

``PermittedFtsSession`` owns one request-scoped ``:memory:`` connection so an
expanded recall route can issue the original question plus a bounded number of
repair queries without rebuilding the same authorized index.  The one-shot
``search_ephemeral_fts`` wrapper preserves the original public contract.  No
index survives the session and no provider or network path exists here.
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, Overflow, localcontext
from typing import Any, ClassVar

from dithyramba.contracts import canonical_sha256_hex, sha256_hex

FTS_TOKENIZER = "unicode61 remove_diacritics 2"
FTS_SCORE_CONTRACT = "bm25_asc_round_half_even_6"
MAX_QUERY_TOKENS = 64
MAX_CANDIDATES = 500

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FRAGMENT_ID_PATTERN = re.compile(r"^fragment_[a-z0-9_]+$")
_SQLITE_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_SCORE_PATTERN = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{6}$")
_SCORE_QUANTUM = Decimal("0.000001")


class FtsError(RuntimeError):
    """An ephemeral FTS index could not be built or queried."""


class FtsUnavailableError(FtsError):
    """The local SQLite runtime cannot satisfy the frozen FTS5 contract."""


class FtsContractError(ValueError):
    """An input or result violates the frozen ephemeral-FTS contract."""


@dataclass(frozen=True, slots=True)
class FtsFragment:
    """One already-authorized text fragment supplied to ephemeral recall."""

    source_fragment_id: str
    text: str
    text_sha256: str

    def __post_init__(self) -> None:
        _require_fragment_id(self.source_fragment_id)
        if type(self.text) is not str:
            raise FtsContractError("FTS fragment text must be str")
        if (
            not self.text.strip()
            or "\x00" in self.text
            or unicodedata.normalize("NFC", self.text) != self.text
        ):
            raise FtsContractError("FTS fragment text must be nonblank NFC text without NUL")
        _require_hash(self.text_sha256, "text_sha256")
        if sha256_hex(self.text.encode("utf-8")) != self.text_sha256:
            raise FtsContractError("FTS fragment text_sha256 does not match text")

    def identity_payload(self) -> dict[str, str]:
        """Return the content-only corpus identity projection."""

        return {
            "source_fragment_id": self.source_fragment_id,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True, slots=True)
class FtsCandidate:
    """One bounded candidate in raw-BM25 order."""

    source_fragment_id: str
    rank: int
    score: str

    def __post_init__(self) -> None:
        _require_fragment_id(self.source_fragment_id)
        if type(self.rank) is not int or self.rank < 1:
            raise FtsContractError("FTS candidate rank must be a positive integer")
        _require_score(self.score)

    def payload(self) -> dict[str, object]:
        return {
            "source_fragment_id": self.source_fragment_id,
            "rank": self.rank,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class FtsRuntimeProfile:
    """SQLite/tokenizer/score identity required for reproducible replay."""

    sqlite_version: str
    compile_options_hash: str
    tokenizer: str = field(default=FTS_TOKENIZER, init=False)
    score_contract: str = field(default=FTS_SCORE_CONTRACT, init=False)

    def __post_init__(self) -> None:
        if (
            type(self.sqlite_version) is not str
            or _SQLITE_VERSION_PATTERN.fullmatch(self.sqlite_version) is None
        ):
            raise FtsContractError("SQLite version must be a canonical numeric version")
        _require_hash(self.compile_options_hash, "compile_options_hash")

    @property
    def profile_version(self) -> str:
        """Return the compact persisted identity of the complete FTS contract."""

        return f"fts_v1.u61rd2.bm25a_he6.sqlite{self.sqlite_version}.co{self.compile_options_hash}"

    def payload(self) -> dict[str, str]:
        return {
            "profile_version": self.profile_version,
            "tokenizer": self.tokenizer,
            "score_contract": self.score_contract,
            "sqlite_version": self.sqlite_version,
            "compile_options_hash": self.compile_options_hash,
        }


@dataclass(frozen=True, slots=True)
class FtsSearchResult:
    """Complete deterministic result of one fresh permitted-only FTS build."""

    SCHEMA: ClassVar[str] = "dithyramba.fts_search_result/1.0"

    retrieval_corpus_hash: str
    query_tokens: tuple[str, ...]
    max_candidates: int
    candidate_count: int
    trace: tuple[FtsCandidate, ...]
    profile: FtsRuntimeProfile

    def __post_init__(self) -> None:
        _require_hash(self.retrieval_corpus_hash, "retrieval_corpus_hash")
        if type(self.query_tokens) is not tuple or any(
            type(token) is not str
            or not token
            or "\x00" in token
            or any(character.isspace() for character in token)
            or unicodedata.normalize("NFC", token) != token
            for token in self.query_tokens
        ):
            raise FtsContractError("query_tokens must be canonical non-empty token strings")
        if len(self.query_tokens) > MAX_QUERY_TOKENS:
            raise FtsContractError("query token count exceeds the frozen limit")
        if len(set(self.query_tokens)) != len(self.query_tokens):
            raise FtsContractError("query_tokens must be unique in first-occurrence order")
        _require_max_candidates(self.max_candidates)
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise FtsContractError("candidate_count must be a non-negative integer")
        if type(self.trace) is not tuple or any(
            not isinstance(candidate, FtsCandidate) for candidate in self.trace
        ):
            raise FtsContractError("trace must be a tuple of FtsCandidate values")
        if len(self.trace) > self.max_candidates or len(self.trace) > self.candidate_count:
            raise FtsContractError("bounded trace is inconsistent with candidate counts")
        if tuple(candidate.rank for candidate in self.trace) != tuple(
            range(1, len(self.trace) + 1)
        ):
            raise FtsContractError("FTS candidate ranks must be contiguous from one")
        identifiers = tuple(candidate.source_fragment_id for candidate in self.trace)
        if len(set(identifiers)) != len(identifiers):
            raise FtsContractError("FTS candidate IDs must be unique")
        serialized_scores = tuple(Decimal(candidate.score) for candidate in self.trace)
        if serialized_scores != tuple(sorted(serialized_scores)):
            raise FtsContractError("FTS serialized scores must be monotonic ascending")
        if not isinstance(self.profile, FtsRuntimeProfile):
            raise FtsContractError("profile must be an FtsRuntimeProfile")

    @property
    def profile_version(self) -> str:
        return self.profile.profile_version

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "retrieval_corpus_hash": self.retrieval_corpus_hash,
            "query_tokens": list(self.query_tokens),
            "max_candidates": self.max_candidates,
            "candidate_count": self.candidate_count,
            "trace": [candidate.payload() for candidate in self.trace],
            "profile": self.profile.payload(),
        }

    @property
    def result_hash(self) -> str:
        """Return the rebuild/replay identity of this exact FTS result."""

        return canonical_sha256_hex(self.semantic_payload())


def search_ephemeral_fts(
    *,
    question: str,
    fragments: tuple[FtsFragment, ...],
    max_candidates: int,
) -> FtsSearchResult:
    """Build, query, and close one permitted-only in-memory FTS5 index."""

    # Keep the one-shot boundary compatible with the original implementation:
    # reject an invalid request before allocating even an ephemeral database.
    # ``PermittedFtsSession.search`` repeats these checks because it is also a
    # public entry point.
    _validated_question(question)
    _require_max_candidates(max_candidates)
    with PermittedFtsSession(fragments=fragments) as session:
        return session.search(question=question, max_candidates=max_candidates)


class PermittedFtsSession:
    """One authorized, reusable and explicitly closed in-memory FTS index.

    A session is deliberately narrow: the permitted fragment set is immutable,
    every query keeps the existing ``FtsSearchResult/1.0`` contract, and closing
    the session destroys the only SQLite connection.  It is therefore safe to
    reuse for ``q0`` plus a small QueryCloud without creating persistent recall
    state.
    """

    def __init__(self, *, fragments: tuple[FtsFragment, ...]) -> None:
        ordered_fragments = _validated_fragments(fragments)
        retrieval_corpus_hash = _retrieval_corpus_hash(ordered_fragments)
        try:
            connection = sqlite3.connect(":memory:")
        except sqlite3.Error as error:
            raise FtsUnavailableError("could not create the ephemeral SQLite database") from error
        try:
            connection.execute("PRAGMA temp_store = MEMORY")
            _create_fts_tables(connection)
            connection.executemany(
                "INSERT INTO permitted_fts(source_fragment_id, text) VALUES (?, ?)",
                ((fragment.source_fragment_id, fragment.text) for fragment in ordered_fragments),
            )
            profile = _runtime_profile(connection)
        except (sqlite3.Error, TypeError, ValueError, IndexError) as error:
            connection.close()
            raise FtsUnavailableError("ephemeral FTS5 recall failed closed") from error
        self._connection: sqlite3.Connection | None = connection
        self._retrieval_corpus_hash = retrieval_corpus_hash
        self._profile = profile

    @property
    def closed(self) -> bool:
        return self._connection is None

    @property
    def retrieval_corpus_hash(self) -> str:
        return self._retrieval_corpus_hash

    @property
    def profile(self) -> FtsRuntimeProfile:
        return self._profile

    def __enter__(self) -> PermittedFtsSession:
        if self.closed:
            raise FtsContractError("closed permitted FTS session cannot be reopened")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        connection = self._connection
        # Mark the capability closed before asking SQLite to release it.  Even
        # if SQLite reports a close failure, callers can never query this
        # session again.
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error as error:
                raise FtsUnavailableError("could not close the permitted FTS session") from error

    def search(self, *, question: str, max_candidates: int) -> FtsSearchResult:
        """Query the immutable permitted index and return a frozen raw result."""

        normalized_question = _validated_question(question)
        _require_max_candidates(max_candidates)
        connection = self._connection
        if connection is None:
            raise FtsContractError("permitted FTS session is closed")
        try:
            # Tokenization state is per query.  Clear it so offsets and first-
            # occurrence order cannot leak across q0/q1/q2.
            connection.execute("DELETE FROM question_fts")
            query_tokens = _derive_query_tokens(connection, normalized_question)
            if not query_tokens:
                return FtsSearchResult(
                    retrieval_corpus_hash=self._retrieval_corpus_hash,
                    query_tokens=(),
                    max_candidates=max_candidates,
                    candidate_count=0,
                    trace=(),
                    profile=self._profile,
                )

            match_expression = " OR ".join(_quoted_match_token(token) for token in query_tokens)
            candidate_count = int(
                connection.execute(
                    "SELECT count(*) FROM permitted_fts WHERE permitted_fts MATCH ?",
                    (match_expression,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                """
                SELECT source_fragment_id, bm25(permitted_fts)
                FROM permitted_fts
                WHERE permitted_fts MATCH ?
                ORDER BY bm25(permitted_fts) ASC,
                         source_fragment_id COLLATE BINARY ASC
                LIMIT ?
                """,
                (match_expression, max_candidates),
            ).fetchall()
            trace = tuple(
                FtsCandidate(
                    source_fragment_id=str(row[0]),
                    rank=rank,
                    score=format_fts_score(row[1]),
                )
                for rank, row in enumerate(rows, start=1)
            )
            return FtsSearchResult(
                retrieval_corpus_hash=self._retrieval_corpus_hash,
                query_tokens=query_tokens,
                max_candidates=max_candidates,
                candidate_count=candidate_count,
                trace=trace,
                profile=self._profile,
            )
        except FtsContractError:
            # A contract failure after SQLite work began means the result path
            # itself is no longer trustworthy (for example, a malformed score
            # returned by a substituted runtime).  Destroy the session rather
            # than allowing a partially failed request to be retried in place.
            self.close()
            raise
        except (sqlite3.Error, TypeError, ValueError, IndexError) as error:
            with suppress(FtsUnavailableError):
                self.close()
            raise FtsUnavailableError("ephemeral FTS5 recall failed closed") from error


def current_fts_runtime_profile() -> FtsRuntimeProfile:
    """Probe and return the complete local FTS5 replay identity."""

    try:
        connection = sqlite3.connect(":memory:")
    except sqlite3.Error as error:
        raise FtsUnavailableError("could not create the FTS profile probe") from error
    try:
        _create_fts_tables(connection)
        return _runtime_profile(connection)
    except sqlite3.Error as error:
        raise FtsUnavailableError("local SQLite does not satisfy the FTS5 profile") from error
    finally:
        connection.close()


def format_fts_score(value: Any) -> str:
    """Format one finite SQLite BM25 score with the frozen decimal contract."""

    if type(value) is bool or not isinstance(value, (int, float, Decimal)):
        raise FtsContractError("FTS score must be numeric")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise FtsContractError("FTS score must be numeric") from error
    if not decimal.is_finite() or (isinstance(value, float) and not math.isfinite(value)):
        raise FtsContractError("FTS score must be finite")
    try:
        with localcontext() as context:
            context.prec = 80
            quantized = decimal.quantize(_SCORE_QUANTUM, rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, Overflow, OverflowError) as error:
        raise FtsContractError("FTS score is outside the supported decimal range") from error
    if quantized == 0:
        quantized = Decimal("0.000000")
    formatted = f"{quantized:.6f}"
    _require_score(formatted)
    return formatted


def _create_fts_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE VIRTUAL TABLE permitted_fts USING fts5(
            source_fragment_id UNINDEXED,
            text,
            tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )
    connection.execute(
        """
        CREATE VIRTUAL TABLE question_fts USING fts5(
            text,
            tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )
    connection.execute(
        "CREATE VIRTUAL TABLE question_vocab USING fts5vocab(question_fts, 'instance')"
    )


def _derive_query_tokens(
    connection: sqlite3.Connection,
    normalized_question: str,
) -> tuple[str, ...]:
    connection.execute("INSERT INTO question_fts(text) VALUES (?)", (normalized_question,))
    rows = connection.execute(
        "SELECT term FROM question_vocab ORDER BY doc ASC, col ASC, offset ASC"
    ).fetchall()
    unique: list[str] = []
    seen: set[str] = set()
    for row in rows:
        token = str(row[0])
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)
        if len(unique) == MAX_QUERY_TOKENS:
            break
    return tuple(unique)


def _runtime_profile(connection: sqlite3.Connection) -> FtsRuntimeProfile:
    sqlite_version = str(connection.execute("SELECT sqlite_version()").fetchone()[0])
    compile_options = sorted(
        str(row[0]) for row in connection.execute("PRAGMA compile_options").fetchall()
    )
    compile_options_hash = canonical_sha256_hex(
        {
            "schema": "dithyramba.sqlite_compile_options/1.0",
            "options": compile_options,
        }
    )
    return FtsRuntimeProfile(
        sqlite_version=sqlite_version,
        compile_options_hash=compile_options_hash,
    )


def _validated_question(question: str) -> str:
    if type(question) is not str:
        raise FtsContractError("question must be str")
    normalized = unicodedata.normalize("NFC", question)
    if (
        not normalized.strip()
        or normalized != normalized.strip()
        or normalized != question
        or "\x00" in normalized
        or len(normalized) > 2_000
    ):
        raise FtsContractError(
            "question must be nonblank, unpadded NFC text without NUL and at most 2,000 code points"
        )
    return normalized


def _validated_fragments(fragments: tuple[FtsFragment, ...]) -> tuple[FtsFragment, ...]:
    if type(fragments) is not tuple or any(
        not isinstance(fragment, FtsFragment) for fragment in fragments
    ):
        raise FtsContractError("fragments must be a tuple of FtsFragment values")
    identifiers = tuple(fragment.source_fragment_id for fragment in fragments)
    if len(set(identifiers)) != len(identifiers):
        raise FtsContractError("permitted FTS fragment IDs must be unique")
    return tuple(sorted(fragments, key=lambda fragment: fragment.source_fragment_id))


def _retrieval_corpus_hash(fragments: tuple[FtsFragment, ...]) -> str:
    return canonical_sha256_hex(
        {
            "schema": "dithyramba.fts_corpus/1.0",
            "fragments": [fragment.identity_payload() for fragment in fragments],
        }
    )


def _quoted_match_token(token: str) -> str:
    return f'"{token.replace(chr(34), chr(34) * 2)}"'


def _require_fragment_id(value: str) -> str:
    if type(value) is not str or _FRAGMENT_ID_PATTERN.fullmatch(value) is None:
        raise FtsContractError("source_fragment_id must use the fragment_ identifier prefix")
    return value


def _require_hash(value: str, label: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise FtsContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _require_score(value: str) -> str:
    if type(value) is not str or _SCORE_PATTERN.fullmatch(value) is None:
        raise FtsContractError("FTS score must be a fixed six-decimal string")
    if Decimal(value) == 0 and value != "0.000000":
        raise FtsContractError("FTS score must canonicalize negative zero")
    return value


def _require_max_candidates(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_CANDIDATES:
        raise FtsContractError("max_candidates must be an integer from 1 through 500")
    return value
