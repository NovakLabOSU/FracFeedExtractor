"""Unit tests for src/preprocessing/text_cleaner.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.text_cleaner import clean_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(n: int, body: str) -> str:
    return f"[PAGE {n}]\n{body}"


# ---------------------------------------------------------------------------
# Reference / bibliography section removal
# ---------------------------------------------------------------------------


class TestReferenceSectionRemoval:
    def test_drops_references_header_and_trailing_content(self):
        text = "Methods\nWe examined 50 stomachs from Canis lupus.\n\n" "References\nSmith J. 2001. J Ecol 10:1-5.\nDoe A. 2002. Nature 400:1.\n"
        result = clean_text(text)
        assert "Smith J." not in result
        assert "Doe A." not in result

    def test_keeps_content_before_references(self):
        text = "Results\nSample size was 30.\n\n" "References\n1. Author A. 2000. Title. Journal.\n"
        result = clean_text(text)
        assert "Sample size was 30" in result

    def test_drops_literature_cited(self):
        text = "Discussion\nSee below.\n\nLiterature Cited\nFoo B. 1999.\n"
        result = clean_text(text)
        assert "Foo B." not in result

    def test_drops_bibliography(self):
        text = "Results\nN=42.\n\nBibliography\nBar C. 2005.\n"
        result = clean_text(text)
        assert "Bar C." not in result

    def test_references_on_separate_page_doesnt_poison_next_page(self):
        text = "[PAGE 3]\nResults\nSample size = 30.\n" "[PAGE 4]\nReferences\n1. Smith 2001.\n" "[PAGE 5]\nDiscussion\nThis study found important results.\n"
        result = clean_text(text)
        assert "Sample size = 30" in result
        assert "Smith 2001" not in result
        assert "This study found important results" in result


# ---------------------------------------------------------------------------
# Acknowledgement / funding section removal
# ---------------------------------------------------------------------------


class TestAcknowledgementRemoval:
    def test_drops_acknowledgements(self):
        text = "Methods\nWe sampled fish.\n\nAcknowledgements\nWe thank our funders.\n"
        result = clean_text(text)
        assert "We thank our funders" not in result

    def test_drops_acknowledgments_us_spelling(self):
        text = "Results\nN=10.\n\nAcknowledgments\nFunded by NSF.\n"
        result = clean_text(text)
        assert "Funded by NSF" not in result

    def test_drops_funding_section(self):
        text = "Abstract\nWe studied diets.\n\nFunding\nGrant XYZ-123.\n"
        result = clean_text(text)
        assert "Grant XYZ-123" not in result

    def test_keeps_content_before_acknowledgements(self):
        text = "Results\n42 stomachs examined.\n\nAcknowledgements\nThanks.\n"
        result = clean_text(text)
        assert "42 stomachs examined" in result


# ---------------------------------------------------------------------------
# Line-level noise removal
# ---------------------------------------------------------------------------


class TestPageNumberRemoval:
    def test_removes_standalone_page_number(self):
        text = "Results\n\n42\n\nMore text here.\n"
        result = clean_text(text)
        assert "\n42\n" not in result

    def test_keeps_number_inside_sentence(self):
        text = "We examined 42 stomachs.\n"
        result = clean_text(text)
        assert "We examined 42 stomachs" in result


class TestReferenceEntryRemoval:
    def test_removes_bracketed_ref_entry(self):
        result = clean_text("[1] Smith J. 2001. Nature.\n")
        assert "[1] Smith" not in result

    def test_removes_numbered_ref_entry(self):
        result = clean_text("1. Jones A. 1999. J Ecol.\n")
        assert "Jones A." not in result

    def test_keeps_regular_sentences_starting_with_number(self):
        # Sentences like "40 specimens were examined" should stay
        text = "A total of 40 specimens were examined in 2005.\n"
        result = clean_text(text)
        assert "40 specimens" in result


class TestUrlAndDoiRemoval:
    def test_removes_http_url(self):
        result = clean_text("See https://www.example.com/paper for details.\n")
        assert "https://" not in result

    def test_removes_doi(self):
        result = clean_text("doi.org/10.1016/j.biocon.2001.01.001\n")
        assert "doi.org" not in result


class TestEmailRemoval:
    def test_removes_email_address(self):
        result = clean_text("Contact: author@university.edu for more info.\n")
        assert "@university.edu" not in result


class TestCopyrightRemoval:
    def test_removes_copyright_symbol(self):
        result = clean_text("© 2021 Elsevier Ltd. All rights reserved.\n")
        assert "Elsevier" not in result

    def test_removes_copyright_word(self):
        result = clean_text("Copyright 2020 The Authors.\n")
        assert "Copyright 2020" not in result


class TestFigureCaptionRemoval:
    def test_removes_figure_caption(self):
        result = clean_text("Figure 1. Map of study area showing sampling sites.\n")
        assert "Map of study area" not in result

    def test_removes_table_caption(self):
        result = clean_text("Table 2. Diet composition of Vulpes vulpes.\n")
        assert "Diet composition" not in result

    def test_removes_fig_abbreviation(self):
        result = clean_text("Fig. 3. Distribution of stomach contents.\n")
        assert "Distribution of stomach" not in result


class TestAffiliationRemoval:
    def test_removes_department_line(self):
        result = clean_text("Department of Ecology, University of Oslo, Norway.\n")
        assert "Department of Ecology" not in result

    def test_removes_university_of_line(self):
        result = clean_text("University of Cambridge, Cambridge CB2 1TN, UK.\n")
        assert "University of Cambridge" not in result


class TestReceivedAcceptedRemoval:
    def test_removes_received_line(self):
        result = clean_text("Received: 12 March 2023; Accepted: 5 June 2023\n")
        assert "12 March 2023" not in result


class TestKeywordsRemoval:
    def test_removes_keywords_line(self):
        result = clean_text("Keywords: predator, diet, stomach contents, ecology\n")
        assert "predator, diet" not in result


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------


class TestWhitespaceNormalisation:
    def test_collapses_multiple_blank_lines(self):
        text = "Line one.\n\n\n\n\nLine two.\n"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_strips_leading_and_trailing_whitespace(self):
        text = "   \n\nSome content.\n\n   "
        result = clean_text(text)
        assert result == result.strip()


# ---------------------------------------------------------------------------
# Pass-through for clean text
# ---------------------------------------------------------------------------


class TestCleanPassThrough:
    def test_clean_text_passes_through_unaffected_content(self):
        text = (
            "Abstract\nThis study examined stomach contents of Canis lupus.\n\n"
            "Methods\nWe collected 50 specimens from Yellowstone, USA.\n\n"
            "Results\nOf 50 stomachs, 8 were empty and 42 contained prey.\n"
        )
        result = clean_text(text)
        for fragment in ["Canis lupus", "50 specimens", "Yellowstone", "8 were empty"]:
            assert fragment in result, f"Expected '{fragment}' to be preserved"

    def test_page_markers_preserved(self):
        text = "[PAGE 1]\nAbstract\nDiet study.\n[PAGE 2]\nMethods\nWe sampled.\n"
        result = clean_text(text)
        assert "[PAGE 1]" in result
        assert "[PAGE 2]" in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_text("   \n\n   ") == ""
