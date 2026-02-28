"""Text preprocessing and section extraction utilities.

This module handles intelligent extraction of key sections from scientific papers,
prioritizing the most informative content when text must be truncated to fit
within LLM context windows.
"""

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.pdf_text_extraction import extract_text_from_pdf

# ---------------------------------------------------------------------------
# Section-boundary splitting helpers
# ---------------------------------------------------------------------------

# Optional numeric prefix shared by all section patterns, e.g. "1.", "2.1.", "3.2.1 "
_NUM_PREFIX = r"(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?"

# (pattern, priority)  — lower number = kept first when budget is tight
_SECTION_PRIORITIES: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(abstract|summary)\s*[:\.]?\s*$"), 0),
    (re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(results?|findings?)\s*[:\.]?\s*$"), 1),
    (re.compile(
        r"(?i)^\s*" + _NUM_PREFIX +
        r"(materials?\s*(?:and|&)\s*methods?|methods?|methodology"
        r"|study\s*(?:area|site|design|region|period))\s*[:\.]?\s*$"
    ), 2),
    (re.compile(r"(?i)^\s*table\s*\d"), 3),
    (re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(introduction|background)\s*[:\.]?\s*$"), 4),
    (re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(discussion|conclusions?|summary\s+and\s+discussion)\s*[:\.]?\s*$"), 5),
]

_DROP_SECTION_RE: re.Pattern = re.compile(
    r"(?i)^\s*"
    r"(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?"   # optional numeric prefix
    r"("
    r"acknowledge?ments?"
    r"|literature\s+cited"
    r"|references?\s+cited"
    r"|references?"
    r"|bibliography"
    r"|appendix\b"
    r"|supplementary\s+(data|material|information)"
    r"|supporting\s+information"
    r"|conflict\s+of\s+interest"
    r"|competing\s+interests?"
    r"|author\s+contributions?"
    r"|funding(?:\s+(?:sources?|information))?"
    r"|data\s+availability"
    r"|ethics\s+(statement|declaration)"
    r")\s*[:\.]?\s*$"
)


def _section_priority(heading: str) -> int:
    """Return the priority integer for a section heading (lower = more important).
    Unknown / un-labelled sections get priority 6.
    Drop sections return 999 and should be excluded before calling this.
    """
    for pat, pri in _SECTION_PRIORITIES:
        if pat.match(heading.strip()):
            return pri
    return 6


# ---------------------------------------------------------------------------
# Legacy page-split helpers (kept for source-page resolution in llm_client.py)
# ---------------------------------------------------------------------------

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
    1. Scan the cleaned text for section headings (Abstract, Results, Methods …)
       regardless of [PAGE N] markers, giving section-level rather than
       page-level granularity.
    2. Drop Reference / Acknowledgement / Appendix sections entirely.
    3. Rank remaining sections by content priority:
       Abstract > Results > Methods > Tables > Introduction > Discussion > other
    4. Greedily pack sections in priority order until the budget is spent.
    5. Re-order selected sections in their original reading order so the LLM
       receives coherent, in-document-order text.

    Falls back to simple character truncation if no section headings are found
    (e.g. very short files or files with no structural markers).

    Args:
        text: Cleaned text of the document (may contain [PAGE N] markers).
        max_chars: Maximum character budget for the output.

    Returns:
        Extracted text containing the most relevant sections within the budget.
        If the full text fits within max_chars, it is returned as-is.
    """
    if len(text) <= max_chars:
        return text

    lines = text.split("\n")

    # ── Build section list ─────────────────────────────────────────────────
    # Each entry: (original_line_index, heading_str, content_str)
    sections: List[Tuple[int, str, str]] = []
    current_heading: str = "[PREAMBLE]"
    current_start: int = 0
    current_lines: List[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_drop = bool(_DROP_SECTION_RE.match(stripped)) if stripped else False
        is_known = any(pat.match(stripped) for pat, _ in _SECTION_PRIORITIES) if stripped else False

        if is_drop or is_known:
            # Flush the in-progress section
            sections.append((current_start, current_heading, "\n".join(current_lines)))
            current_heading = stripped
            current_start = i
            current_lines = []
        else:
            current_lines.append(line)

    # Flush the final section
    sections.append((current_start, current_heading, "\n".join(current_lines)))

    # Fall back to simple truncation when no headings were detected
    meaningful = [s for s in sections if s[1] != "[PREAMBLE]"]
    if not meaningful:
        return text[:max_chars]

    # ── Score and filter ───────────────────────────────────────────────────
    scored: List[Tuple[int, int, str, str]] = []  # (priority, orig_idx, heading, content)
    for orig_idx, (start, heading, content) in enumerate(sections):
        if _DROP_SECTION_RE.match(heading.strip()) if heading.strip() else False:
            continue  # hard-drop references, acknowledgements, appendix, …
        if heading == "[PREAMBLE]":
            # The preamble (everything before the first section heading)
            # almost always contains the abstract.  Treat it as priority 0
            # so it is packed first, ahead of all other sections.
            priority = 0
        else:
            priority = _section_priority(heading)
        scored.append((priority, orig_idx, heading, content))

    # Sort by priority: most important sections first
    scored.sort(key=lambda t: t[0])

    # ── Greedily fill budget ───────────────────────────────────────────────
    selected: List[Tuple[int, str]] = []  # (orig_idx, chunk)
    budget = max_chars
    for _pri, orig_idx, heading, content in scored:
        chunk = (f"{heading}\n{content}").strip() if heading != "[PREAMBLE]" else content.strip()
        if not chunk:
            continue
        if len(chunk) <= budget:
            selected.append((orig_idx, chunk))
            budget -= len(chunk)
        elif budget > 200:
            selected.append((orig_idx, chunk[:budget]))
            budget = 0
            break

    # Re-sort by original index so the LLM reads content in document order
    selected.sort(key=lambda t: t[0])
    return "\n\n".join(chunk for _, chunk in selected)


def load_document(file_path: Path) -> str:
    """Load document from PDF or text file.

    Args:
        file_path: Path to the input file (.pdf or .txt)

    Returns:
        Extracted text content with [PAGE N] markers

    Raises:
        RuntimeError: If file reading fails
    """
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        print(f"[INFO] Reading PDF file...", file=sys.stderr)
        return extract_text_from_pdf(str(file_path))
    elif suffix in ['.txt', '.text']:
        print(f"[INFO] Reading text file...", file=sys.stderr)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError as e:
            raise RuntimeError(f"Text file encoding error: {e}")
    else:
        raise RuntimeError(f"Unsupported file type: {suffix}. Use .pdf or .txt files.")
