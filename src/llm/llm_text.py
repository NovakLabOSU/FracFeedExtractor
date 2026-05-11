"""Text preprocessing and section extraction utilities.

This module handles intelligent extraction of key sections from scientific papers,
prioritizing the most informative content when text must be truncated to fit
within LLM context windows.
"""

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# NOTE: pdf_text_extraction is imported lazily inside read_file_text() to
# avoid pulling in heavy PDF dependencies (camelot, fitz) when only the
# text-based pipeline is used.

# ---------------------------------------------------------------------------
# Section-boundary splitting helpers
# ---------------------------------------------------------------------------

# Optional numeric prefix shared by all section patterns, e.g. "1.", "2.1.", "3.2.1 "
_NUM_PREFIX = r"(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?"

# (pattern, priority)  — lower number = kept first when budget is tight
_SECTION_PRIORITIES: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(abstract|summary)\s*[:\.]?\s*$"), 0),
    (re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(results?|findings?)\s*[:\.]?\s*$"), 1),
    (re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(materials?\s*(?:and|&)\s*methods?|methods?|methodology" r"|study\s*(?:area|site|design|region|period))\s*[:\.]?\s*$"), 2),
    (re.compile(r"(?i)^\s*table\s*\d"), 3),
    (re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(introduction|background)\s*[:\.]?\s*$"), 4),
    (re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(discussion|conclusions?|summary\s+and\s+discussion)\s*[:\.]?\s*$"), 5),
]

_DROP_SECTION_RE: re.Pattern = re.compile(
    r"(?i)^\s*"
    r"(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?"  # optional numeric prefix
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
# Paragraph-level keyword scoring
# ---------------------------------------------------------------------------
# Each tuple is (compiled pattern, score weight).  A paragraph’s total score
# is the sum of weights for every pattern that matches anywhere in it.
# Higher-scoring paragraphs are packed into the LLM budget first.

_FIELD_PATTERNS: List[Tuple[re.Pattern, int]] = [
    # sample_size — explicit counts of stomachs / specimens / individuals
    (
        re.compile(
            r"(?i)(\bn\s*=\s*\d+"
            r"|total\s+of\s+\d+"
            r"|\d+\s+stomachs?"
            r"|\d+\s+specimens?"
            r"|\d+\s+individuals?"
            r"|\d+\s+birds?"
            r"|\d+\s+fish"
            r"|\d+\s+samples?"
            r"|sample\s+size\s+(of\s+)?\d+"
            r"|examined\s+\d+"
            r"|\d+\s+(were|was)\s+(examined|collected|analysed|analyzed|sampled))"
        ),
        4,
    ),
    # num_empty_stomachs — explicit empty-stomach language
    (
        re.compile(
            r"(?i)(empty\s+stomachs?"
            r"|stomachs?\s+(were\s+)?empty"
            r"|had\s+empty"
            r"|without\s+food"
            r"|without\s+(stomach\s+)?contents?"
            r"|zero\s+prey"
            r"|no\s+food\s+(items?|remains?)"
            r"|vacuous|vacant\s+stomachs?)"
        ),
        5,
    ),
    # num_nonempty_stomachs / fraction_feeding
    (
        re.compile(
            r"(?i)(non.?empty"
            r"|contained\s+(food|prey|items?)"
            r"|with\s+food"
            r"|with\s+(stomach\s+)?contents?"
            r"|had\s+(food|prey)"
            r"|feeding\s+rate"
            r"|proportion\s+(feeding|with\s+food)"
            r"|percent\s+(feeding|with\s+food)"
            r"|\d+\s*%\s+of\s+(stomachs?|individuals?|birds?|fish|specimens?))"
        ),
        5,
    ),
    # frequency of occurrence — standard diet-study metric
    (
        re.compile(r"(?i)(frequency\s+of\s+occurrence" r"|occurrence\s+frequency" r"|%\s*fo\b" r"|\bfo\s*=\s*\d" r"|index\s+of\s+relative\s+importance" r"|\biri\b)"),
        3,
    ),
    # stomach content / diet composition — general relevance signal
    (
        re.compile(r"(?i)(stomach\s+content" r"|gut\s+content" r"|diet\s+composition" r"|food\s+habits?" r"|dietary\s+(analysis|study|data))"),
        2,
    ),
    # general percentage / fraction near gut/stomach context
    (re.compile(r"(?i)(\d+\.?\d*\s*%|\d+\s+percent" r"|\d+\s+of\s+\d+\s+(were|had|contained)" r"|proportion\s+of\s+\d+)"), 2),
    # study date — collection period
    (
        re.compile(
            r"(?i)(collected\s+(in|during|between)"
            r"|sampled\s+(in|during|between)"
            r"|field\s+season"
            r"|study\s+period"
            r"|between\s+\d{4}\s+and\s+\d{4}"
            r"|\d{4}[\-\u2013]\d{4}"
            r"|sampling\s+period)"
        ),
        2,
    ),
    # study location
    (re.compile(r"(?i)(study\s+(area|site|region)" r"|specimens?\s+(were\s+)?(obtained|collected|caught)\s+(from|at|in)" r"|sampling\s+(location|site|area))"), 2),
    # any binomial species name (used as a weak relevance signal)
    (re.compile(r"\b[A-Z][a-z]+\s+[a-z]{3,}\b"), 1),
]

# Maximum characters reserved for the pinned abstract/preamble.  Any
# remaining budget is filled by keyword-scored paragraphs.
_ABSTRACT_CAP: int = 2000


def _truncate_at_sentence(text: str, limit: int) -> str:
    """Truncate *text* to at most *limit* chars, preferring a sentence boundary.

    Searches backwards from *limit* for the last ``.``, ``!``, or ``?``.  If
    one is found in the second half of the allowed window it is used; otherwise
    the hard character limit is applied.
    """
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    for punct in (".", "!", "?"):
        pos = candidate.rfind(punct)
        if pos > limit // 2:
            return candidate[: pos + 1]
    return candidate


def _score_paragraph(para: str) -> int:
    """Return a keyword-relevance score for a single paragraph of text.

    Paragraphs that mention many extraction targets (stomach counts, empty /
    non-empty language, sample sizes, dates, locations) score higher and are
    preferentially included in the LLM prompt.
    """
    score = 0
    for pat, weight in _FIELD_PATTERNS:
        if pat.search(para):
            score += weight
    return score


# ---------------------------------------------------------------------------
# Legacy page-split helpers (kept for source-page resolution in llm_client.py)
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

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
    return True, len(SECTION_PATTERNS)


def extract_key_sections(text: str, max_chars: int) -> str:
    """Return the most informative portion of text within the character budget.

    Two-phase strategy
    ------------------
    Phase 1 — Abstract pin
        Always include the preamble (everything before the first section
        heading), which almost always contains the abstract, title, and key
        study metadata.  Capped at ``_ABSTRACT_CAP`` characters so it cannot
        crowd out the data-rich content below.

    Phase 2 — Keyword-scored paragraph mining
        Split all remaining non-dropped section text into blank-line-separated
        paragraphs.  Score each paragraph by how many extraction-relevant
        keywords it contains (sample counts, empty/non-empty stomach language,
        percentages, dates, locations, species names).  Pack the
        highest-scoring paragraphs first until the remaining budget is full.
        Re-order selected paragraphs to their original document position so
        the LLM receives coherent, in-order text.

    This approach guarantees that sentences like
        “A total of 144 stomach samples… 58% contained food”
    are always included regardless of which named section they fall in.

    Args:
        text: Cleaned text of the document (may contain [PAGE N] markers).
        max_chars: Maximum character budget for the output.

    Returns:
        Extracted text fitting within *max_chars*.
        If the full text already fits, it is returned unchanged.
    """
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")

    # ── Phase 1: Split document into named sections ────────────────────────
    # Each entry: (start_line_idx, heading, content)
    sections: List[Tuple[int, str, str]] = []
    current_heading: str = "[PREAMBLE]"
    current_start: int = 0
    current_lines: List[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_drop = bool(_DROP_SECTION_RE.match(stripped)) if stripped else False
        is_known = any(pat.match(stripped) for pat, _ in _SECTION_PRIORITIES) if stripped else False
        if is_drop or is_known:
            sections.append((current_start, current_heading, "\n".join(current_lines)))
            current_heading = stripped
            current_start = i
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_start, current_heading, "\n".join(current_lines)))

    # ── Phase 1 result: pin the abstract/preamble ──────────────────────────
    # Sections whose content is always pinned (counted against _ABSTRACT_CAP):
    # the implicit preamble and any explicitly-named Abstract/Summary section.
    _ABSTRACT_HEADING_RE = re.compile(r"(?i)^\s*" + _NUM_PREFIX + r"(abstract|summary)\s*[:\.]?\s*$")

    preamble_parts: List[str] = []
    body_sections: List[Tuple[int, str, str]] = []  # (start, heading, content)
    for start, heading, content in sections:
        if _DROP_SECTION_RE.match(heading.strip()) if heading.strip() else False:
            continue  # hard-drop references / acknowledgements / appendix
        if heading == "[PREAMBLE]" or _ABSTRACT_HEADING_RE.match(heading.strip()):
            preamble_parts.append(content.strip())
        else:
            body_sections.append((start, heading, content))

    preamble_text = "\n\n".join(p for p in preamble_parts if p)[:_ABSTRACT_CAP]

    budget = max_chars - len(preamble_text)

    # ── Phase 2: keyword-scored paragraph mining ───────────────────────────
    # Collect every paragraph from ALL body sections (blank-line separated).
    # Each paragraph remembers its position so we can restore reading order.
    # (start_line, paragraph_text, keyword_score)
    raw_paragraphs: List[Tuple[int, str, int]] = []

    for sec_start, heading, content in body_sections:
        # Prepend the section heading so the LLM sees which section it’s in.
        block = (f"{heading}\n{content}").strip() if heading else content.strip()
        if not block:
            continue
        # Split on blank lines into paragraphs
        para_lines: List[str] = []
        para_start = sec_start
        for j, ln in enumerate(block.split("\n")):
            if ln.strip():
                para_lines.append(ln)
            else:
                if para_lines:
                    para_text = "\n".join(para_lines).strip()
                    raw_paragraphs.append((sec_start + j, para_text, _score_paragraph(para_text)))
                    para_lines = []
                    para_start = sec_start + j + 1
        if para_lines:
            para_text = "\n".join(para_lines).strip()
            raw_paragraphs.append((para_start, para_text, _score_paragraph(para_text)))

    # Sort by score descending; use original position as tiebreaker (earlier first)
    raw_paragraphs.sort(key=lambda t: (-t[2], t[0]))

    # Greedily fill budget with highest-scoring paragraphs
    selected_paras: List[Tuple[int, str]] = []  # (orig_pos, text)
    for pos, para_text, score in raw_paragraphs:
        if budget <= 0:
            break
        if len(para_text) <= budget:
            selected_paras.append((pos, para_text))
            budget -= len(para_text)
        elif budget > 200:
            selected_paras.append((pos, _truncate_at_sentence(para_text, budget)))
            budget = 0

    # Re-sort to original document order so the LLM reads coherent text
    selected_paras.sort(key=lambda t: t[0])

    parts: List[str] = []
    if preamble_text:
        parts.append(preamble_text)
    parts.extend(p for _, p in selected_paras)
    return "\n\n".join(parts)


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
        print("[INFO] Reading PDF file...", file=sys.stderr)
        from src.preprocessing.pdf_text_extraction import extract_text_from_pdf

        return extract_text_from_pdf(str(file_path))
    elif suffix in ['.txt', '.text']:
        print("[INFO] Reading text file...", file=sys.stderr)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError as e:
            log.error("Text file encoding error for %s: %s", file_path, e)
            raise RuntimeError(f"Text file encoding error: {e}")
    else:
        log.error("Unsupported file type attempted: %s", file_path.suffix)
        raise RuntimeError(f"Unsupported file type: {suffix}. Use .pdf or .txt files.")
