"""Shared lexical-fold regression tests."""

import pytest

from dithyramba.lexical import (
    fold_lexical_text,
    folded_literal_present,
    normalize_strict_literal_text,
    strict_literal_present_in_normalized_text,
)


def test_fold_is_diacritic_insensitive_and_collapses_layout() -> None:
    assert fold_lexical_text("  CÔTÉ\nMädchen  ") == "cote madchen"


def test_fold_preserves_meaning_bearing_non_latin_marks() -> None:
    assert fold_lexical_text("Й Ї Ё Київ") == "й ї ё київ"
    assert fold_lexical_text("कि") == "कि"
    assert folded_literal_present("й", "и") is False
    assert folded_literal_present("ї", "\u0456") is False


def test_literal_match_respects_unicode_word_boundaries() -> None:
    assert folded_literal_present("Kant and art", "kant") is True
    assert folded_literal_present("Kantian artist", "kant") is False
    assert folded_literal_present("Kantian artist", "art") is False
    assert folded_literal_present("(art)", "art") is True
    assert folded_literal_present("subject", "") is False


def test_fold_rejects_non_text_input() -> None:
    with pytest.raises(TypeError, match="must be str"):
        fold_lexical_text(1)  # type: ignore[arg-type]


def test_strict_literal_projection_preserves_latin_and_cyrillic_marks() -> None:
    normalized = normalize_strict_literal_text("SÍ · AÑO · Maße · до́ма")

    assert strict_literal_present_in_normalized_text(normalized, "sí") is True
    assert strict_literal_present_in_normalized_text(normalized, "si") is False
    assert strict_literal_present_in_normalized_text(normalized, "año") is True
    assert strict_literal_present_in_normalized_text(normalized, "ano") is False
    assert strict_literal_present_in_normalized_text(normalized, "Maße") is True
    assert strict_literal_present_in_normalized_text(normalized, "masse") is False
    assert strict_literal_present_in_normalized_text(normalized, "до́") is False


def test_strict_literal_projection_rejects_non_text_input() -> None:
    with pytest.raises(TypeError, match="must be str"):
        normalize_strict_literal_text(1)  # type: ignore[arg-type]
