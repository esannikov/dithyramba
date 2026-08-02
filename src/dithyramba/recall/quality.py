"""Deterministic candidate hygiene before evidence admission.

Raw FTS remains an immutable retrieval trace.  This module makes a narrower
decision for the answer route: obvious reference matter, indexes, tabular
layout, and fragments with no meaningful query overlap are not offered to an
evidence gate.  The rules are deliberately conservative and provider-free.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_YEAR_PATTERN = re.compile(r"\b(?:1[5-9]|20)\d{2}[a-z]?\b", re.IGNORECASE)
_PAGE_RUN_PATTERN = re.compile(r"(?:^|[,;]\s*)\d{1,4}(?:[-\u2013]\d{1,4})?(?=\s*(?:[,;]|$))")
_MARKDOWN_TABLE_RULE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|){2,}", re.MULTILINE)

_REFERENCE_HEADINGS = frozenset(
    {
        "bibliography",
        "references",
        "reference list",
        "works cited",
        "literature cited",
        "sources",
        "бібліографія",
        "список літератури",
        "список використаних джерел",
        "література",
    }
)
_INDEX_HEADINGS = frozenset(
    {
        "index",
        "name index",
        "subject index",
        "general index",
        "table of contents",
        "contents",
        "покажчик",
        "іменний покажчик",
        "предметний покажчик",
        "зміст",
    }
)
_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "and",
        "are",
        "been",
        "before",
        "between",
        "could",
        "does",
        "from",
        "have",
        "how",
        "into",
        "is",
        "it",
        "that",
        "their",
        "this",
        "through",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "для",
        "його",
        "коли",
        "між",
        "після",
        "перед",
        "про",
        "також",
        "через",
        "чого",
        "який",
        "яким",
        "які",
        "як",
        "що",
    }
)


class CandidateNoiseReason(StrEnum):
    """Stable reasons why a fragment is withheld from answer admission."""

    BIBLIOGRAPHY = "bibliography"
    INDEX = "index"
    TABLE = "table"
    TOPIC_DRIFT = "topic_drift"


class CandidateQualityAssessment(BaseModel):
    """Auditable, content-addressable decision for one retrieved fragment."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_fragment_id: str = Field(pattern=r"^fragment_[a-z0-9_]+$")
    admitted: bool
    reasons: tuple[CandidateNoiseReason, ...] = ()

    @field_validator("reasons")
    @classmethod
    def _unique_reasons(
        cls,
        value: tuple[CandidateNoiseReason, ...],
    ) -> tuple[CandidateNoiseReason, ...]:
        ordered = tuple(sorted(value, key=lambda item: item.value))
        if len(set(ordered)) != len(ordered):
            raise ValueError("candidate noise reasons must be unique")
        return ordered


def assess_candidate_quality(
    *,
    source_fragment_id: str,
    question: str,
    text: str,
    fragment_kind: str,
    source_address: dict[str, object],
) -> CandidateQualityAssessment:
    """Apply strict layout and topic guards without semantic inference."""

    headings = _heading_labels(source_address)
    normalized_text = _normalized_text(text)
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    reasons: set[CandidateNoiseReason] = set()

    if headings.intersection(_REFERENCE_HEADINGS) or _looks_like_reference_list(lines):
        reasons.add(CandidateNoiseReason.BIBLIOGRAPHY)
    if headings.intersection(_INDEX_HEADINGS) or _looks_like_index(lines):
        reasons.add(CandidateNoiseReason.INDEX)
    if _looks_like_table(fragment_kind=fragment_kind, text=text, lines=lines):
        reasons.add(CandidateNoiseReason.TABLE)
    if _has_topic_drift(question=question, normalized_text=normalized_text):
        reasons.add(CandidateNoiseReason.TOPIC_DRIFT)

    ordered = tuple(sorted(reasons, key=lambda item: item.value))
    return CandidateQualityAssessment(
        source_fragment_id=source_fragment_id,
        admitted=not ordered,
        reasons=ordered,
    )


def meaningful_query_tokens(question: str) -> tuple[str, ...]:
    """Return conservative lexical anchors used only for drift detection."""

    tokens: list[str] = []
    seen: set[str] = set()
    for token in _tokens(question):
        if len(token) < 3 or token in _QUERY_STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


def _heading_labels(source_address: dict[str, object]) -> frozenset[str]:
    raw = source_address.get("heading_path")
    if type(raw) is not list or any(type(item) is not str for item in raw):
        return frozenset()
    return frozenset(_normalized_text(item) for item in raw)


def _looks_like_reference_list(lines: tuple[str, ...]) -> bool:
    if len(lines) < 3:
        return False
    reference_like = 0
    for line in lines:
        normalized = _normalized_text(line)
        has_year = _YEAR_PATTERN.search(normalized) is not None
        has_identifier = "doi.org/" in normalized or "isbn" in normalized
        has_citation_punctuation = normalized.count(".") >= 2 and "," in normalized
        if has_identifier or (has_year and has_citation_punctuation):
            reference_like += 1
    return reference_like >= 3 and reference_like * 4 >= len(lines) * 3


def _looks_like_index(lines: tuple[str, ...]) -> bool:
    if len(lines) < 4:
        return False
    index_like = 0
    for line in lines:
        if len(_PAGE_RUN_PATTERN.findall(line)) >= 2 and len(line) <= 180:
            index_like += 1
    return index_like >= 4 and index_like * 4 >= len(lines) * 3


def _looks_like_table(
    *,
    fragment_kind: str,
    text: str,
    lines: tuple[str, ...],
) -> bool:
    normalized_kind = _normalized_text(fragment_kind).replace("-", "_")
    if normalized_kind in {"table", "table_cell", "table_row"}:
        return True
    if _MARKDOWN_TABLE_RULE.search(text) is not None and text.count("|") >= 6:
        return True
    tabular_lines = sum(1 for line in lines if line.count("\t") >= 2)
    return len(lines) >= 4 and tabular_lines * 4 >= len(lines) * 3


def _has_topic_drift(*, question: str, normalized_text: str) -> bool:
    query_tokens = meaningful_query_tokens(question)
    if not query_tokens:
        return False
    text_tokens = set(_tokens(normalized_text))
    return not text_tokens.intersection(query_tokens)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(_normalized_text(value)))


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().replace("\u00ad", " ").strip()
