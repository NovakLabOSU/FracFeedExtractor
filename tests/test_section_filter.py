"""Unit tests for src/preprocessing/section_filter.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.section_filter import (
    filter_relevant_sections,
    _has_positive_signal,
    _has_negative_signal,
    _should_keep,
    _split_into_blocks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(n: int, body: str) -> str:
    return f"[PAGE {n}]\n{body}"


# ---------------------------------------------------------------------------
# Structural preservation
# ---------------------------------------------------------------------------


class TestStructuralPreservation:
    """Page markers and section headings must never be removed."""

    def test_preserves_page_markers(self):
        text = "[PAGE 1]\nSome content.\n\n[PAGE 2]\nMore content."
        result = filter_relevant_sections(text)
        assert "[PAGE 1]" in result
        assert "[PAGE 2]" in result

    def test_preserves_section_headings(self):
        text = (
            "Abstract\n\nWe studied diets.\n\n"
            "Methods\n\nWe sampled fish.\n\n"
            "Results\n\nN=42 stomachs examined."
        )
        result = filter_relevant_sections(text)
        assert "Abstract" in result
        assert "Methods" in result
        assert "Results" in result

    def test_preserves_numbered_section_headings(self):
        text = "1. Introduction\n\nBackground text.\n\n2. Methods\n\nSampling."
        result = filter_relevant_sections(text)
        assert "1. Introduction" in result
        assert "2. Methods" in result

    def test_empty_input_returns_empty(self):
        assert filter_relevant_sections("") == ""
        assert filter_relevant_sections("   ") == "   "
        assert filter_relevant_sections(None) is None

    def test_short_text_preserved_fully(self):
        text = "A brief note about Canis lupus diet."
        result = filter_relevant_sections(text)
        assert result == text


# ---------------------------------------------------------------------------
# Positive-signal paragraphs are always kept
# ---------------------------------------------------------------------------


class TestPositiveSignalKept:
    """Paragraphs with target-metric language must always be retained."""

    def test_keeps_sample_size_paragraph(self):
        para = "A total of 144 stomach samples were collected from Canis lupus."
        text = f"Results\n\n{para}"
        result = filter_relevant_sections(text)
        assert "144 stomach samples" in result

    def test_keeps_empty_stomach_paragraph(self):
        para = "Of the 100 stomachs examined, 23 were empty."
        text = f"Results\n\n{para}"
        result = filter_relevant_sections(text)
        assert "23 were empty" in result

    def test_keeps_nonempty_paragraph(self):
        para = "58% of stomachs contained food items."
        text = f"Results\n\n{para}"
        result = filter_relevant_sections(text)
        assert "58%" in result

    def test_keeps_feeding_rate(self):
        para = "The feeding rate was 72% across all seasons."
        text = f"Results\n\n{para}"
        result = filter_relevant_sections(text)
        assert "feeding rate" in result

    def test_keeps_collection_date(self):
        para = "Specimens were collected between 2005 and 2010."
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "collected between 2005 and 2010" in result

    def test_keeps_study_location(self):
        para = "The study area was located in the Okavango Delta, Botswana."
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "Okavango Delta" in result

    def test_keeps_diet_composition(self):
        para = "Diet composition of Panthera leo was dominated by ungulates."
        text = f"Results\n\n{para}"
        result = filter_relevant_sections(text)
        assert "Diet composition" in result

    def test_keeps_percentage_data(self):
        para = "Mammals constituted 45% of the total prey items."
        text = f"Results\n\n{para}"
        result = filter_relevant_sections(text)
        assert "45%" in result

    def test_keeps_n_equals(self):
        para = "We analysed gut contents (n = 87) from adult specimens."
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "n = 87" in result

    def test_keeps_coordinates(self):
        para = "The sampling site (34°15'S, 18°29'E) was coastal."
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "34°15'S" in result


# ---------------------------------------------------------------------------
# Negative-signal paragraphs are dropped (when no positive signal)
# ---------------------------------------------------------------------------


class TestNegativeSignalDropped:
    """Paragraphs with only negative signals should be removed."""

    def test_drops_phylogenetic_paragraph(self):
        para = (
            "Bayesian inference of the phylogenetic relationships among "
            "the sister group taxa revealed strong bootstrap support for "
            "the monophyletic clade."
        )
        text = f"Discussion\n\n{para}"
        result = filter_relevant_sections(text)
        assert "Bayesian inference" not in result

    def test_drops_habitat_description(self):
        para = (
            "Vegetation type was classified as tropical dry forest with "
            "canopy cover dense and understory dominated by shrubs."
        )
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "canopy cover" not in result

    def test_drops_literature_review(self):
        para = (
            "Several studies have reported similar findings. "
            "Previously reported by Smith (2001) and consistent with "
            "the findings of Jones et al. (2005)."
        )
        text = f"Discussion\n\n{para}"
        result = filter_relevant_sections(text)
        assert "Several studies have reported" not in result

    def test_drops_prey_id_methodology(self):
        para = (
            "Prey were identified to the lowest taxonomic level using "
            "a stereomicroscope and reference collection of diagnostic bones."
        )
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "stereomicroscope" not in result

    def test_drops_morphometric_paragraph(self):
        para = (
            "Snout-vent length was measured to the nearest 0.1 mm. "
            "Total length was measured for each individual using calipers."
        )
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "Snout-vent length" not in result

    def test_drops_statistical_methods(self):
        para = (
            "Differences were tested using Kruskal-Wallis tests with "
            "Bonferroni correction for multiple comparisons. A generalized "
            "linear model was used to assess trends."
        )
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "Kruskal-Wallis" not in result

    def test_drops_conservation_policy(self):
        para = (
            "Conservation implications suggest the species should be "
            "upgraded to a higher IUCN Red List category given the "
            "observed population decline."
        )
        text = f"Discussion\n\n{para}"
        result = filter_relevant_sections(text)
        assert "IUCN Red List" not in result

    def test_drops_genetic_methods(self):
        para = (
            "DNA was extracted using a commercial kit. PCR amplification "
            "was performed using primers targeting the mitochondrial gene "
            "region. Products were separated by gel electrophoresis."
        )
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "PCR amplification" not in result


# ---------------------------------------------------------------------------
# Conservative behaviour — borderline paragraphs kept
# ---------------------------------------------------------------------------


class TestConservativeBehaviour:
    """Paragraphs with no signals should be kept (borderline = safe)."""

    def test_keeps_neutral_paragraph(self):
        para = "The region experiences a temperate maritime climate."
        text = f"Introduction\n\n{para}"
        result = filter_relevant_sections(text)
        assert "temperate maritime climate" in result

    def test_keeps_paragraph_with_mixed_signals(self):
        """Positive signal overrides co-occurring negative signal."""
        para = (
            "A total of 50 stomachs were examined. Prey were identified "
            "using a stereomicroscope and diagnostic bones from a "
            "reference collection."
        )
        text = f"Methods\n\n{para}"
        result = filter_relevant_sections(text)
        assert "50 stomachs" in result
        assert "stereomicroscope" in result

    def test_keeps_species_diet_sentence(self):
        para = "Feeding ecology of Vulpes vulpes in agricultural landscapes."
        text = f"Abstract\n\n{para}"
        result = filter_relevant_sections(text)
        assert "Vulpes vulpes" in result


# ---------------------------------------------------------------------------
# Integration-style tests with multi-section documents
# ---------------------------------------------------------------------------


class TestFullDocumentFiltering:
    """Test the filter on realistic multi-section document text."""

    def test_realistic_document(self):
        text = (
            "[PAGE 1]\n"
            "Abstract\n\n"
            "We examined the diet of Canis lupus in Yellowstone.\n\n"
            "[PAGE 2]\n"
            "Methods\n\n"
            "Specimens were collected between 2010 and 2015 from Yellowstone "
            "National Park (44°36'N, 110°30'W).\n\n"
            "Snout-vent length was measured to the nearest millimetre "
            "using digital calipers.\n\n"
            "DNA was extracted from tissue samples. PCR amplification of "
            "mitochondrial DNA markers was performed.\n\n"
            "[PAGE 3]\n"
            "Results\n\n"
            "A total of 200 stomachs were examined, of which 42 were empty.\n\n"
            "Diet composition included 65% ungulates and 20% lagomorphs.\n\n"
            "[PAGE 4]\n"
            "Discussion\n\n"
            "Previous work has reported broadly concordant results. Previously "
            "reported by Mech (1970) and consistent with findings of "
            "Black et al. (2003).\n\n"
            "Conservation implications suggest continued monitoring of wolf "
            "population viability in the Greater Yellowstone area.\n"
        )
        result = filter_relevant_sections(text)

        # Structural markers preserved
        assert "[PAGE 1]" in result
        assert "[PAGE 2]" in result
        assert "[PAGE 3]" in result
        assert "[PAGE 4]" in result
        assert "Abstract" in result
        assert "Methods" in result
        assert "Results" in result
        assert "Discussion" in result

        # Positive-signal paragraphs kept
        assert "diet of Canis lupus" in result
        assert "200 stomachs" in result
        assert "42 were empty" in result
        assert "65%" in result
        assert "collected between 2010 and 2015" in result

        # Negative-only paragraphs dropped
        assert "Snout-vent length" not in result
        assert "PCR amplification" not in result
        assert "Previous work has reported" not in result
        assert "population viability" not in result

    def test_does_not_over_filter_short_doc(self):
        """A short document with mostly positive content should lose very little."""
        text = (
            "Abstract\n\n"
            "We examined stomach contents of 50 Accipiter nisus.\n\n"
            "Results\n\n"
            "Sample size was n=50. Of these, 12 stomachs were empty.\n"
            "Birds comprised 78% of prey items.\n"
        )
        result = filter_relevant_sections(text)
        # Everything here is positive — nothing should be lost
        assert "50 Accipiter nisus" in result
        assert "n=50" in result
        assert "12 stomachs were empty" in result
        assert "78%" in result


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Test the internal scoring/splitting functions."""

    def test_has_positive_signal_true(self):
        assert _has_positive_signal("A total of 30 stomachs were examined.")
        assert _has_positive_signal("Specimens were collected during 2018.")
        assert _has_positive_signal("The study area was in Kenya.")
        assert _has_positive_signal("Empty stomachs accounted for 15%.")

    def test_has_positive_signal_false(self):
        assert not _has_positive_signal("The weather was mild that year.")
        assert not _has_positive_signal(
            "Bayesian inference supported the monophyletic clade."
        )

    def test_has_negative_signal_true(self):
        assert _has_negative_signal(
            "Phylogenetic analysis using maximum likelihood trees."
        )
        assert _has_negative_signal(
            "Habitat type was classified as open grassland."
        )
        assert _has_negative_signal(
            "Mann-Whitney tests were used for comparisons."
        )

    def test_has_negative_signal_false(self):
        assert not _has_negative_signal("We counted 30 prey items in the gut.")
        assert not _has_negative_signal("The study was conducted in Brazil.")

    def test_should_keep_positive_overrides_negative(self):
        text = (
            "A total of 100 stomachs were examined using a stereomicroscope "
            "and reference collection."
        )
        assert _should_keep(text) is True

    def test_should_keep_neutral_kept(self):
        assert _should_keep("The area has a subtropical climate.") is True

    def test_should_keep_negative_only_dropped(self):
        assert _should_keep(
            "Bayesian inference of phylogenetic relationships."
        ) is False

    def test_split_into_blocks_basic(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        blocks = _split_into_blocks(text)
        assert len(blocks) == 3
        assert "Para one." in blocks[0]
        assert "Para two." in blocks[1]
        assert "Para three." in blocks[2]

    def test_split_into_blocks_page_markers(self):
        text = "[PAGE 1]\nContent one.\n\n[PAGE 2]\nContent two."
        blocks = _split_into_blocks(text)
        page_blocks = [b for b in blocks if "[PAGE" in b]
        assert len(page_blocks) >= 2
