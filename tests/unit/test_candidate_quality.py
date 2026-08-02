"""Conservative answer-route hygiene over raw FTS candidates."""

import pytest
from pydantic import ValidationError

from dithyramba.recall import (
    CandidateNoiseReason,
    CandidateQualityAssessment,
    assess_candidate_quality,
)


def _assessment(
    text: str,
    *,
    question: str = "How does dialogue blocking reveal power?",
    fragment_kind: str = "paragraph",
    headings: list[str] | None = None,
) -> CandidateQualityAssessment:
    return assess_candidate_quality(
        source_fragment_id="fragment_quality_001",
        question=question,
        text=text,
        fragment_kind=fragment_kind,
        source_address={"heading_path": headings or []},
    )


def test_reference_heading_is_withheld_even_when_query_terms_match() -> None:
    result = _assessment(
        "Smith, J. Dialogue Blocking and Power. London, 2020.",
        headings=["Bibliography"],
    )

    assert result.admitted is False
    assert CandidateNoiseReason.BIBLIOGRAPHY in result.reasons


def test_index_layout_is_withheld() -> None:
    result = _assessment(
        "\n".join(
            (
                "blocking, 12, 18",
                "dialogue, 24, 91",
                "power, 16, 44",
                "staging, 31, 70",
            )
        )
    )

    assert result.admitted is False
    assert CandidateNoiseReason.INDEX in result.reasons


def test_structural_table_fragment_is_withheld() -> None:
    result = _assessment(
        "character\tblocking\tpower",
        fragment_kind="table_row",
    )

    assert result.admitted is False
    assert CandidateNoiseReason.TABLE in result.reasons


def test_fragment_without_meaningful_query_overlap_is_topic_drift() -> None:
    result = _assessment("A chapter about costume colour and textile restoration.")

    assert result.admitted is False
    assert result.reasons == (CandidateNoiseReason.TOPIC_DRIFT,)


def test_substantive_prose_with_query_anchors_is_admitted() -> None:
    result = _assessment(
        "Dialogue blocking externalizes the power relation between two characters."
    )

    assert result.admitted is True
    assert result.reasons == ()


def test_duplicate_noise_reasons_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        CandidateQualityAssessment(
            source_fragment_id="fragment_quality_001",
            admitted=False,
            reasons=(
                CandidateNoiseReason.TABLE,
                CandidateNoiseReason.TABLE,
            ),
        )


def test_reference_list_is_detected_without_a_heading() -> None:
    result = _assessment(
        "\n".join(
            (
                "Smith, J. Dialogue. Journal. 2020, doi.org/10.1/example",
                "Jones, A. Blocking. Review. 2019, doi.org/10.2/example",
                "Lee, B. Power. Quarterly. 2018, doi.org/10.3/example",
            )
        )
    )

    assert CandidateNoiseReason.BIBLIOGRAPHY in result.reasons


def test_wrapped_numbered_bibliography_page_is_withheld() -> None:
    result = _assessment(
        "\n".join(
            (
                "81. Carruthers G. The sense of agency // Cognition. 2012. Vol. 21.",
                "Pp. 30-45.",
                "82. Carruthers P. Conscious Thought. Cambridge : MIT Press, 2005.",
                "Pp. 134-156.",
                "83. Chalmers D. The Conscious Mind. Oxford : OUP, 1996.",
                "432 p.",
                "84. Chambon V. Human agency // NeuroReport. 2002. Vol. 13.",
                "Pp. 1975-1978.",
            )
        ),
        question="model agency consciousness authorship",
    )

    assert CandidateNoiseReason.BIBLIOGRAPHY in result.reasons


def test_pdf_contents_page_is_withheld_without_heading_metadata() -> None:
    result = _assessment(
        "\n".join(
            (
                "Contents xv",
                "7.6.3.1 Exploratory data analysis 211",
                "7.6.3.2 Experiments 211",
                "7.6.3.3 Error analysis and explainability 212",
                "Bibliography 221",
                "Index 255",
            )
        ),
        question="data analysis explainability",
    )

    assert CandidateNoiseReason.INDEX in result.reasons


def test_index_heading_in_raw_pdf_text_is_sufficient_to_withhold_page() -> None:
    result = _assessment(
        "\n".join(
            (
                "Index Page numbers followed by f and t indicate figures and tables.",
                "Actor-Network Theory, 92",
                "Advanced imaging, 147",
                "Artistic Assessment, 131",
            )
        ),
        question="art provenance expert validation",
    )

    assert CandidateNoiseReason.INDEX in result.reasons


@pytest.mark.parametrize(
    ("text", "fragment_kind"),
    [
        ("| term | value |\n| --- | --- |\n| dialogue | power |", "paragraph"),
        ("term\tvalue\tnote\na\tb\tc\nd\te\tf\ng\th\ti", "paragraph"),
    ],
)
def test_markdown_and_tab_separated_tables_are_withheld(
    text: str,
    fragment_kind: str,
) -> None:
    result = _assessment(text, fragment_kind=fragment_kind)

    assert CandidateNoiseReason.TABLE in result.reasons


def test_malformed_heading_metadata_is_ignored_safely() -> None:
    result = assess_candidate_quality(
        source_fragment_id="fragment_quality_001",
        question="dialogue power",
        text="Dialogue makes power visible.",
        fragment_kind="paragraph",
        source_address={"heading_path": ["Evidence", 7]},
    )

    assert result.admitted is True


def test_question_without_meaningful_tokens_does_not_create_topic_drift() -> None:
    result = _assessment(
        "Substantive prose remains available for the explicit Gate.",
        question="What is it and how?",
    )

    assert CandidateNoiseReason.TOPIC_DRIFT not in result.reasons
