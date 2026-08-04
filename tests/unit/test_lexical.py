"""Shared lexical-fold regression tests."""

import pytest

from dithyramba.lexical import fold_lexical_text, folded_literal_present


def test_fold_is_diacritic_insensitive_and_collapses_layout() -> None:
    assert fold_lexical_text("  CÔTÉ\nMädchen  ") == "cote madchen"


def test_literal_match_respects_unicode_word_boundaries() -> None:
    assert folded_literal_present("Kant and art", "kant") is True
    assert folded_literal_present("Kantian artist", "kant") is False
    assert folded_literal_present("Kantian artist", "art") is False
    assert folded_literal_present("(art)", "art") is True
    assert folded_literal_present("subject", "") is False


def test_fold_rejects_non_text_input() -> None:
    with pytest.raises(TypeError, match="must be str"):
        fold_lexical_text(1)  # type: ignore[arg-type]
