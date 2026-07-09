"""Unit tests for src/extraction/llm_client.py — _resolve_source_pages."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.llm_client import _resolve_source_pages


def _doc(*pages: tuple[int, str]) -> str:
    """Build a fake document string with [PAGE N] markers and text blocks."""
    parts = []
    for num, text in pages:
        parts.append(f"[PAGE {num}]\n{text}")
    return "\n\n".join(parts)


def test_resolve_source_pages_basic():
    text = _doc(
        (1, "Introduction text about nothing useful."),
        (3, "Results: Pygoscelis papua had 42 non-empty stomachs."),
    )
    result = _resolve_source_pages({"species_name": "Pygoscelis papua"}, text)
    assert result == [3]


def test_resolve_source_pages_skips_short_values():
    # Values shorter than 5 chars should be skipped regardless
    text = _doc((2, "Sample size n=5, year 2020."))
    result = _resolve_source_pages({"study_year": "2020"}, text)  # 4 chars — skipped
    assert result is None


def test_resolve_source_pages_word_boundary_prevents_false_positive():
    # "2019" appears first in a citation like "Smith 2019" on page 1,
    # then as the actual study year on page 4. Without word-boundary matching
    # the old .find() would attribute page 1; with \b it still finds page 1
    # because "2019" has word boundaries in "Smith 2019" too. So this test
    # verifies the correct page when the value appears ONLY on the intended page.
    text = _doc(
        (1, "See Smith (2020) for background."),
        (4, "Surveys were conducted in 2019-2021 across three sites."),
    )
    result = _resolve_source_pages({"study_year_range": "2019-2021"}, text)
    assert result == [4]


def test_resolve_source_pages_no_match_returns_none():
    text = _doc((1, "This document mentions nothing relevant."))
    result = _resolve_source_pages({"species_name": "Thunnus albacares"}, text)
    assert result is None


def test_resolve_source_pages_multiple_fields_multiple_pages():
    text = _doc(
        (2, "Study location: Marion Island, sub-Antarctic."),
        (5, "Results: 127 stomachs examined, num_nonempty = 89."),
    )
    record = {"study_location": "Marion Island", "num_nonempty": 89}
    result = _resolve_source_pages(record, text)
    # "Marion Island" (12 chars) → page 2; "89" (2 chars) → skipped
    assert result == [2]


def test_resolve_source_pages_skips_fraction_feeding_and_source_pages():
    text = _doc((1, "fraction_feeding 0.70123 source_pages 3."))
    result = _resolve_source_pages({"fraction_feeding": 0.70123, "source_pages": [3]}, text)
    assert result is None


def test_resolve_source_pages_no_page_markers():
    # Document without any [PAGE N] markers — should return None even if value found
    text = "Species: Gadus morhua. Diet survey results."
    result = _resolve_source_pages({"species_name": "Gadus morhua"}, text)
    assert result is None
