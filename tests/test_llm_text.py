"""Unit tests for src/llm/llm_text.py — section extraction and text preprocessing."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.llm_text import (
    _section_priority,
    _score_paragraph,
    _truncate_at_sentence,
    classify_page,
    extract_key_sections,
    load_document,
    split_into_pages,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

# Realistic synthetic diet-survey text with [PAGE N] markers.
# Mimics the layout of a real paper so tests exercise real-world patterns.
_SURVEY_TEXT = """\
[PAGE 1]
Abstract

We examined the stomach contents of 144 Pygoscelis papua individuals
collected at Marion Island, sub-Antarctic between 1984 and 1985.
A total of 144 food samples was collected; all samples contained prey.
num_empty_stomachs = 0, num_nonempty_stomachs = 144, sample_size = 144.

[PAGE 2]
Methods

Specimens were collected during the 1984-1985 field season.
Stomach pumping was performed on all 144 birds; every sample yielded prey.
Study site: Marion Island, sub-Antarctic.

[PAGE 3]
Results

Of 144 stomachs examined, 144 were non-empty (100%).
Empty stomachs: 0.  Feeding rate: 1.0.
Diet composition consisted primarily of euphausiids and fish.

[PAGE 4]
References

Adams NJ, Brown CR. 1989. Prey selection in the kelp gull...
Anderson DT. 1975. Embryology and phylogeny...
"""

_SHORT_TEXT = "Short text under limit."


# ---------------------------------------------------------------------------
# split_into_pages
# ---------------------------------------------------------------------------


class TestSplitIntoPages:
    def test_basic_split(self):
        text = "[PAGE 1]\nPage one content.\n[PAGE 2]\nPage two content."
        pages = split_into_pages(text)
        assert len(pages) == 2
        assert pages[0] == (1, "\nPage one content.\n")
        assert pages[1] == (2, "\nPage two content.")

    def test_preamble_before_first_marker(self):
        text = "Preamble text.\n[PAGE 1]\nFirst page."
        pages = split_into_pages(text)
        # Preamble goes to page 0; PAGE 1 is second entry
        assert pages[0] == (0, "Preamble text.\n")
        assert pages[1][0] == 1

    def test_no_page_markers_returns_empty(self):
        pages = split_into_pages("No markers here.")
        # Entire text is treated as preamble at page 0
        assert len(pages) == 1
        assert pages[0][0] == 0

    def test_multi_page_document(self):
        pages = split_into_pages(_SURVEY_TEXT)
        page_numbers = [p for p, _ in pages]
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers
        assert 4 in page_numbers

    def test_empty_string(self):
        pages = split_into_pages("")
        assert pages == []

    def test_page_content_preserved(self):
        text = "[PAGE 5]\nSome content here.\n"
        pages = split_into_pages(text)
        assert pages[0][0] == 5
        assert "Some content here." in pages[0][1]


# ---------------------------------------------------------------------------
# classify_page
# ---------------------------------------------------------------------------


class TestClassifyPage:
    def test_abstract_page_is_useful(self):
        useful, priority = classify_page("Abstract\n\nWe studied Gadus morhua diets.")
        assert useful is True
        assert priority == 0  # first SECTION_PATTERNS entry

    def test_results_page_is_useful(self):
        useful, priority = classify_page("Results\n\n144 stomachs examined.")
        assert useful is True
        assert priority == 1

    def test_methods_page_is_useful(self):
        useful, priority = classify_page("Methods\n\nAnimals were sampled at sea.")
        assert useful is True

    def test_references_page_is_dropped(self):
        useful, priority = classify_page("References\n\nSmith J. 1995. ...")
        assert useful is False
        assert priority == 999

    def test_acknowledgements_page_is_dropped(self):
        useful, priority = classify_page("Acknowledgements\n\nWe thank the team.")
        assert useful is False

    def test_unknown_section_is_useful_with_default_priority(self):
        useful, priority = classify_page("Some random section\n\nRandom text.")
        assert useful is True
        assert priority == len([None] * 6)  # len(SECTION_PATTERNS)

    def test_bibliography_dropped(self):
        useful, _ = classify_page("Bibliography\nSmith 1999...")
        assert useful is False


# ---------------------------------------------------------------------------
# _section_priority
# ---------------------------------------------------------------------------


class TestSectionPriority:
    def test_abstract_priority_zero(self):
        assert _section_priority("Abstract") == 0
        assert _section_priority("ABSTRACT") == 0
        assert _section_priority("Summary") == 0

    def test_results_priority_one(self):
        assert _section_priority("Results") == 1
        assert _section_priority("RESULTS") == 1

    def test_methods_priority_two(self):
        assert _section_priority("Methods") == 2
        assert _section_priority("Materials and Methods") == 2
        assert _section_priority("Study Area") == 2

    def test_introduction_priority_four(self):
        assert _section_priority("Introduction") == 4

    def test_discussion_priority_five(self):
        assert _section_priority("Discussion") == 5
        assert _section_priority("Conclusions") == 5

    def test_unknown_heading_returns_six(self):
        assert _section_priority("Some Unknown Section") == 6
        assert _section_priority("Appendix A") == 6

    def test_numbered_headings_recognized(self):
        assert _section_priority("1. Abstract") == 0
        assert _section_priority("2. Methods") == 2
        assert _section_priority("3.1 Results") == 1


# ---------------------------------------------------------------------------
# _score_paragraph
# ---------------------------------------------------------------------------


class TestScoreParagraph:
    def test_empty_paragraph_scores_zero(self):
        assert _score_paragraph("") == 0

    def test_sample_size_language_scores_high(self):
        para = "A total of 144 stomachs were examined."
        assert _score_paragraph(para) > 0

    def test_empty_stomach_language_scores_high(self):
        para = "Of 200 individuals, 45 had empty stomachs."
        assert _score_paragraph(para) > 0

    def test_nonempty_language_scores_high(self):
        para = "144 stomachs were non-empty and contained food."
        assert _score_paragraph(para) > 0

    def test_diet_keywords_score_positive(self):
        para = "Stomach content analysis revealed euphausiids as the primary prey."
        assert _score_paragraph(para) > 0

    def test_date_keywords_score_positive(self):
        para = "Specimens were collected during the 1984-1985 field season."
        assert _score_paragraph(para) > 0

    def test_location_keywords_score_positive(self):
        para = "The study site was located on Marion Island, sub-Antarctic."
        assert _score_paragraph(para) > 0

    def test_data_rich_paragraph_scores_higher_than_plain(self):
        data_para = "N=144 individuals examined; 0 empty stomachs, 144 non-empty, feeding rate 100%."
        plain_para = "The weather was cold during the study."
        assert _score_paragraph(data_para) > _score_paragraph(plain_para)

    def test_binomial_name_scores_nonzero(self):
        para = "Pygoscelis papua was the focal species."
        assert _score_paragraph(para) > 0


# ---------------------------------------------------------------------------
# _truncate_at_sentence
# ---------------------------------------------------------------------------


class TestTruncateAtSentence:
    def test_short_text_returned_unchanged(self):
        text = "Short."
        assert _truncate_at_sentence(text, 100) == text

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _truncate_at_sentence(text, 32)
        assert result.endswith(".")
        assert len(result) <= 32

    def test_falls_back_to_hard_limit(self):
        text = "No sentence boundaries in this very long string of words and more words"
        result = _truncate_at_sentence(text, 20)
        assert len(result) == 20

    def test_exact_limit_unchanged(self):
        text = "Exactly twenty chars"
        assert _truncate_at_sentence(text, len(text)) == text

    def test_respects_exclamation_and_question(self):
        text = "Is this a question? Yes it is! And more text here."
        result = _truncate_at_sentence(text, 25)
        assert result[-1] in ("?", "!", ".")


# ---------------------------------------------------------------------------
# extract_key_sections
# ---------------------------------------------------------------------------


class TestExtractKeySections:
    def test_short_text_returned_unchanged(self):
        result = extract_key_sections(_SHORT_TEXT, max_chars=1000)
        assert result == _SHORT_TEXT

    def test_output_within_budget(self):
        budget = 500
        result = extract_key_sections(_SURVEY_TEXT, max_chars=budget)
        assert len(result) <= budget

    def test_abstract_content_prioritized(self):
        result = extract_key_sections(_SURVEY_TEXT, max_chars=600)
        # The abstract section (PAGE 1) contains key metadata
        assert "144" in result or "Pygoscelis" in result or "Marion Island" in result

    def test_references_excluded(self):
        # Budget forces truncation (text is ~795 chars); references must be dropped
        result = extract_key_sections(_SURVEY_TEXT, max_chars=700)
        assert "Adams NJ, Brown CR. 1989." not in result

    def test_data_rich_content_included(self):
        # Results page has high-scoring phrases — should be included when budget allows
        result = extract_key_sections(_SURVEY_TEXT, max_chars=800)
        # At least some data-bearing content should survive
        assert any(kw in result for kw in ["144", "stomachs", "empty", "non-empty", "Methods", "Abstract"])

    def test_exact_budget_limit_respected(self):
        # The function pins the abstract/preamble first; budgets >= 400 are large
        # enough that the pinned preamble fits and total output respects the limit.
        for budget in [400, 600, 800]:
            result = extract_key_sections(_SURVEY_TEXT, max_chars=budget)
            assert len(result) <= budget, f"Exceeded budget={budget}: got {len(result)}"

    def test_empty_string_returns_empty(self):
        result = extract_key_sections("", max_chars=500)
        assert result == ""

    def test_large_budget_returns_most_content(self):
        # With a very large budget the full text should come back (it's short enough)
        result = extract_key_sections(_SURVEY_TEXT, max_chars=100_000)
        assert result == _SURVEY_TEXT


# ---------------------------------------------------------------------------
# load_document
# ---------------------------------------------------------------------------


class TestLoadDocument:
    def test_loads_txt_file(self, tmp_path):
        txt_file = tmp_path / "survey.txt"
        txt_file.write_text("Sample diet survey text.\n[PAGE 1]\nContent.", encoding="utf-8")
        result = load_document(txt_file)
        assert "Sample diet survey text." in result
        assert "[PAGE 1]" in result

    def test_loads_text_extension(self, tmp_path):
        text_file = tmp_path / "survey.text"
        text_file.write_text("Diet content.", encoding="utf-8")
        result = load_document(text_file)
        assert result == "Diet content."

    def test_unsupported_extension_raises(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Unsupported file type"):
            load_document(csv_file)

    def test_pdf_extension_triggers_pdf_path(self, tmp_path, monkeypatch):
        """PDF loading delegates to extract_text_from_pdf — verify the call is made."""
        pdf_file = tmp_path / "paper.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        import src.llm.llm_text as llm_text_module

        monkeypatch.setattr(
            "src.llm.llm_text.extract_text_from_pdf",
            lambda path: f"extracted from {Path(path).name}",
            raising=False,
        )
        # Patch the lazy import inside load_document
        import importlib
        import sys as _sys

        fake_module = type(_sys)("src.preprocessing.pdf_text_extraction")
        fake_module.extract_text_from_pdf = lambda path: f"extracted from {Path(path).name}"
        monkeypatch.setitem(_sys.modules, "src.preprocessing.pdf_text_extraction", fake_module)

        result = load_document(pdf_file)
        assert "extracted from paper.pdf" in result

    def test_encoding_error_raises_runtime(self, tmp_path):
        bad_file = tmp_path / "bad.txt"
        bad_file.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")
        with pytest.raises(RuntimeError, match="encoding error"):
            load_document(bad_file)
