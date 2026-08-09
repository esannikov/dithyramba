"""Shared deterministic folding for lexical evidence checks.

SQLite FTS5 uses ``unicode61 remove_diacritics 2`` for discovery.  The
post-retrieval quality and evidence gates must not reject the same Latin text
merely because one spelling carries combining marks and the other does not.
This module supplies that shared, provider-free comparison boundary; it does
not alter persisted source text or claim semantic equivalence.
"""

from __future__ import annotations

import re
import unicodedata

LEXICAL_FOLD_PROFILE = "nfkd_latin_marks_casefold_v2"

_SPACE_PATTERN = re.compile(r"\s+")


def fold_lexical_text(value: str) -> str:
    """Return a stable Latin-diacritic-insensitive comparison projection."""

    if type(value) is not str:
        raise TypeError("lexical text must be str")
    decomposed = unicodedata.normalize("NFKD", value).casefold().replace("\u00ad", "")
    folded: list[str] = []
    latin_cluster = False
    for character in decomposed:
        if unicodedata.category(character).startswith("M"):
            if not latin_cluster:
                folded.append(character)
            continue
        latin_cluster = _is_latin_letter(character)
        folded.append(character)
    normalized = unicodedata.normalize("NFC", "".join(folded))
    return _SPACE_PATTERN.sub(" ", normalized).strip()


def folded_literal_present(text: str, literal: str) -> bool:
    """Match one folded literal without accepting substrings inside words."""

    folded_text = fold_lexical_text(text)
    return folded_literal_present_in_folded_text(folded_text, literal)


def folded_literal_present_in_folded_text(folded_text: str, literal: str) -> bool:
    """Match a literal against text that was already folded by this module."""

    folded_literal = fold_lexical_text(literal)
    if not folded_literal:
        return False
    start = 0
    while True:
        index = folded_text.find(folded_literal, start)
        if index < 0:
            return False
        end = index + len(folded_literal)
        before_is_word = index > 0 and _is_word_character(folded_text[index - 1])
        after_is_word = end < len(folded_text) and _is_word_character(folded_text[end])
        literal_starts_word = _is_word_character(folded_literal[0])
        literal_ends_word = _is_word_character(folded_literal[-1])
        if not (literal_starts_word and before_is_word) and not (
            literal_ends_word and after_is_word
        ):
            return True
        start = index + 1


def normalize_strict_literal_text(value: str) -> str:
    """Return a case-insensitive projection that preserves letter distinctions.

    The strict evidence route keeps Latin and Cyrillic diacritics and avoids
    compatibility folding such as German ``ß`` to ``ss``. Soft hyphens remain
    ignorable layout characters, matching the existing evidence profiles.
    """

    if type(value) is not str:
        raise TypeError("literal text must be str")
    normalized = unicodedata.normalize("NFC", value).lower().replace("\u00ad", "")
    return _SPACE_PATTERN.sub(" ", normalized).strip()


def strict_literal_present_in_normalized_text(normalized_text: str, literal: str) -> bool:
    """Match one strict literal with Unicode letter/number/mark boundaries."""

    normalized_literal = normalize_strict_literal_text(literal)
    if not normalized_literal:
        return False
    start = 0
    while True:
        index = normalized_text.find(normalized_literal, start)
        if index < 0:
            return False
        end = index + len(normalized_literal)
        before_is_word = index > 0 and _is_strict_word_character(normalized_text[index - 1])
        after_is_word = end < len(normalized_text) and _is_strict_word_character(
            normalized_text[end]
        )
        literal_starts_word = _is_strict_word_character(normalized_literal[0])
        literal_ends_word = _is_strict_word_character(normalized_literal[-1])
        if not (literal_starts_word and before_is_word) and not (
            literal_ends_word and after_is_word
        ):
            return True
        start = index + 1


def _is_word_character(value: str) -> bool:
    return value == "_" or value.isalnum()


def _is_strict_word_character(value: str) -> bool:
    return value == "_" or unicodedata.category(value)[0] in {"L", "M", "N"}


def _is_latin_letter(value: str) -> bool:
    return unicodedata.category(value).startswith("L") and unicodedata.name(value, "").startswith(
        "LATIN "
    )
