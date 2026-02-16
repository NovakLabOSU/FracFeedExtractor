"""Text preprocessing and section extraction utilities.

This module handles intelligent extraction of key sections from scientific papers,
prioritizing the most informative content when text must be truncated to fit
within LLM context windows.
"""

import re
from typing import List, Tuple


# Section headers commonly found in scientific diet / stomach-content papers.
# Order matters: earlier entries are higher priority when budget is tight.
SECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?i)^\s*(?:abstract|summary)"),
    re.compile(r"(?i)^\s*(?:results?)\b"),
    re.compile(r"(?i)^\s*(?:methods?|materials?\s*(?:and|&)\s*methods?|study\s*(?:area|site))"),
    re.compile(r"(?i)^\s*(?:table)\s*\d"),
    re.compile(r"(?i)^\s*(?:introduction|background)"),
    re.compile(r"(?i)^\s*(?:discussion)"),
]

# Sections that are almost never useful for metric extraction.
SKIP_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?i)^\s*(?:acknowledge|literature\s*cited|references|bibliography|appendix|supplementary)"),
]


def split_into_pages(text: str) -> List[Tuple[int, str]]:
    """Split text on ``[PAGE N]`` markers.

    Args:
        text: Input text with [PAGE N] markers

    Returns:
        List of (page_number, page_text) tuples
    """
    parts = re.split(r"\[PAGE\s+(\d+)\]", text)
    # parts: [before_first_marker, page_num, page_text, page_num, page_text, ...]
    pages: List[Tuple[int, str]] = []
    if parts[0].strip():
        pages.append((0, parts[0]))
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_text = parts[i + 1] if i + 1 < len(parts) else ""
        pages.append((page_num, page_text))
    return pages


def classify_page(page_text: str) -> Tuple[bool, int]:
    """Determine if a page is useful and assign a priority score.

    Args:
        page_text: Text content of the page

    Returns:
        Tuple of (is_useful, priority) where lower priority number means higher importance.
        Pages matching skip patterns return (False, 999).
        Pages with no recognized header get default mid-priority.
    """
    for pat in SKIP_PATTERNS:
        if pat.search(page_text):
            return False, 999
    for idx, pat in enumerate(SECTION_PATTERNS):
        if pat.search(page_text):
            return True, idx
    # No recognised header — still potentially useful (e.g. tables without
    # a "Table" header, continuation of Results, etc.)
    return True, len(SECTION_PATTERNS)


def extract_key_sections(text: str, max_chars: int) -> str:
    """Return the most informative portion of text within the character budget.

    Strategy:
    1. Split the paper into pages using [PAGE N] markers
    2. Drop pages belonging to References/Acknowledgements/Appendix
    3. Rank remaining pages by section priority:
       Abstract > Results > Methods > Tables > Introduction > Discussion > other
    4. Greedily pack pages in priority order until the budget is spent
    5. Re-order selected pages by their original page number so the LLM
       sees them in reading order

    Args:
        text: Full text of the document
        max_chars: Maximum character budget for the output

    Returns:
        Extracted text containing the most relevant sections within the budget.
        If the full text fits within max_chars, it is returned as-is.
    """
    if len(text) <= max_chars:
        return text

    pages = split_into_pages(text)
    scored: List[Tuple[int, int, str]] = []  # (priority, page_num, page_text)
    for page_num, page_text in pages:
        useful, priority = classify_page(page_text)
        if useful:
            scored.append((priority, page_num, page_text))

    # Sort by priority (ascending = most important first)
    scored.sort(key=lambda t: t[0])

    selected: List[Tuple[int, str]] = []
    budget = max_chars
    for _priority, page_num, page_text in scored:
        page_with_marker = f"[PAGE {page_num}]\n{page_text}"
        if len(page_with_marker) <= budget:
            selected.append((page_num, page_with_marker))
            budget -= len(page_with_marker)
        elif budget > 200:
            # Partially include the page up to the remaining budget
            selected.append((page_num, page_with_marker[:budget]))
            budget = 0
            break

    # Re-sort by page number so the LLM sees content in reading order
    selected.sort(key=lambda t: t[0])
    return "\n".join(chunk for _, chunk in selected)