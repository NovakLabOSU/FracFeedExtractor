"""Noise removal for raw .txt files extracted from scientific PDFs.

Strips content that adds no value to structured data extraction:
  - Reference / bibliography sections (everything from the header onward)
  - Acknowledgement and funding sections
  - Author affiliation blocks (department names, university lines, emails)
  - Figure and table captions
  - Copyright and licence lines
  - Standalone page-number lines
  - DOI / URL lines
  - Journal metadata lines (volume, issue, ISSN, received/accepted dates)
  - Excessive blank lines and leading/trailing whitespace

Exposes one primary function::

    clean_text(text: str) -> str

The function is safe to call on text that already has ``[PAGE N]`` markers;
those markers are preserved so the downstream ``extract_key_sections()``
function can still do page-priority ranking.

Usage (standalone)::

    python text_cleaner.py path/to/file.txt
    python text_cleaner.py path/to/file.txt --max-chars 5000
"""

import re
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Section-level drop patterns
# When a line matches one of these, the entire remainder of that page/section
# block is discarded (until the next [PAGE N] marker or end of text).
# ---------------------------------------------------------------------------
_SECTION_DROP_HEADERS: re.Pattern = re.compile(
    r"(?i)^\s*("
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

# Structured-header block patterns common in two-column journal PDFs.
# When a line matches, subsequent short lines (< 60 chars) that look like
# keyword / metadata values are dropped until a long content line resumes.
# NOTE: requires actual whitespace between letters to avoid matching "Abstract".
_STRUCTURED_HEADER_START: re.Pattern = re.compile(
    r"(?i)^\s*("
    r"a\s+b\s+s\s+t\s+r\s+a\s+c\s+t"  # spaced "A B S T R A C T"
    r"|a\s+r\s+t\s+i\s+c\s+l\s+e\s+i\s+n\s+f\s+o"  # spaced "A R T I C L E  I N F O"
    r")\s*$"
)

# ---------------------------------------------------------------------------
# Line-level drop patterns
# Lines that individually match one of these are removed regardless of where
# they appear in the document.
# ---------------------------------------------------------------------------
_LINE_DROP_PATTERNS: List[re.Pattern] = [
    # Standalone page numbers (digit-only line, 1–4 digits, optional spaces)
    re.compile(r"^\s*\d{1,4}\s*$"),
    # Reference list entries: "[1] Smith ...", "1. Smith ...", "1) Smith ..."
    re.compile(r"^\s*\[\d+\]\s+[A-Z]"),
    re.compile(r"^\s*\d{1,3}[.)]\s{1,4}[A-Z][a-z]"),
    # DOI and bare URLs (doi.org/, bare doi:, https://, www.)
    re.compile(r"(?i)(https?://|doi\.org/|\bdoi:\s*10\.|www\.)\S*"),
    # Email addresses
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # Copyright / licence lines
    re.compile(r"(?i)(©|\(c\)\s*\d{4}|copyright\s+\d{4}|all\s+rights\s+reserved" r"|published\s+by\s+elsevier|creative\s+commons|open\s+access" r"|this\s+(article|paper|is)\s+(is\s+)?published)"),
    # Journal metadata: volume, issue, ISSN, page range
    re.compile(r"(?i)^\s*(vol(ume)?\.?\s*\d|issue\s*\d|pp\.\s*\d|issn\s*[\d\-]" r"|journal\s+of|proceedings\s+of)"),
    # Received / accepted / revised / available-online timestamps
    # Match with or without trailing colon/semicolon
    re.compile(r"(?i)^\s*(received|accepted|revised|available\s+online|" r"published\s+online|handling\s+editor)" r"(\s*[:;]|\s+\d|\s+in)"),
    # Article history block header
    re.compile(r"(?i)^\s*article\s+history\s*[:\.]?\s*$"),
    # Keywords header line AND single-word keyword-style lines that follow it
    re.compile(r"(?i)^\s*key\s*-?\s*words?\s*[:\-]"),
    # Journal / publisher metadata lines
    re.compile(r"(?i)^\s*(contents?\s+lists?\s+available|journal\s+homepage|" r"elsevier\.com|sciencedirect\.com|springer\.|wiley\.com)"),
    # Figure / table / plate captions
    re.compile(r"(?i)^\s*(fig(ure)?\.?\s*\d|table\s*\d|plate\s*\d|" r"fig\.\s*s\d|supplemental?\s+(table|figure)\s*\d)" r"[\s.\-–—:]"),
    # Author affiliation lines (institution / department / lab names)
    # Allow for leading special characters (e.g. ⁎, *, †)
    re.compile(
        r"(?i)^[\s\*⁎†‡#]*"
        r"(department\s+of|faculty\s+of|institute\s+(of|for)|"
        r"division\s+of|school\s+of|laboratory\s+of|lab\s+of|"
        r"centre?\s+(for|of)|program(me)?\s+(in|of)|"
        r"universidad|universit[éy]|université|universidade|"
        r"university\s+of|college\s+of)"
    ),
    # Running page headers / footers: short all-caps lines that are NOT
    # known section headings (those are whitelisted in _drop_line below).
    re.compile(r"^[A-Z\s\d\.\-–—:,]{5,60}$"),
]

# ---------------------------------------------------------------------------
# Section-header whitelist
# Lines matching this pattern are structural anchors the LLM needs; they must
# never be removed even if they look like all-caps noise.
# ---------------------------------------------------------------------------
_SECTION_HEADER_WHITELIST: re.Pattern = re.compile(
    r"(?i)^\s*"
    # optional section number prefix  e.g. "1.", "2.1.", "3.2.1 "
    r"(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?"
    r"("
    r"abstract"
    r"|summary"
    r"|introduction"
    r"|background"
    r"|methods?"
    r"|materials?\s*(?:and|&)\s*methods?"
    r"|methodology"
    r"|study\s*(?:area|site|design|region|period)"
    r"|results?"
    r"|findings?"
    r"|discussion"
    r"|conclusions?"
    r"|summary\s+and\s+discussion"
    r"|acknowledge?ments?"
    r")\s*[:\.\-]?\s*$"
)

# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
# Line ending with a soft hyphen or mid-word break (next line starts lowercase
# or with punctuation that continues the word).
_SOFT_HYPHEN_END = re.compile(r"-$")
_CONTINUATION_LINE = re.compile(r"^[a-z,\.\)\];:!\?]")


def _rejoin_broken_lines(text: str) -> str:
    """Rejoin lines that were broken mid-sentence or mid-word by column wrapping.

    Two cases are handled:
    1. Hard hyphenation: line ends with ``-`` and next line starts with a
       lowercase letter → join without any space (remove the hyphen).
    2. Soft wrap: line ends without sentence-terminating punctuation and next
       line starts with lowercase → join with a single space.

    ``[PAGE N]`` marker lines are never joined.
    """
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Never merge PAGE markers
        if re.match(r"\[PAGE\s+\d+\]", line.strip()):
            out.append(line)
            i += 1
            continue

        rstripped = line.rstrip()
        if i + 1 < len(lines):
            next_line = lines[i + 1].lstrip()
            next_is_page = re.match(r"\[PAGE\s+\d+\]", next_line)
            if not next_is_page and next_line and _CONTINUATION_LINE.match(next_line):
                # Hard hyphen: remove hyphen and join directly
                if _SOFT_HYPHEN_END.search(rstripped):
                    out.append(rstripped[:-1] + next_line)
                    i += 2
                    continue
                # Soft wrap: line doesn't end with sentence-ending punctuation
                if rstripped and rstripped[-1] not in ".!?:;)]":
                    out.append(rstripped + " " + next_line)
                    i += 2
                    continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _drop_line(line: str) -> bool:
    """Return True if this line should be removed from the output."""
    stripped = line.strip()
    if not stripped:
        return False  # preserve blank lines for now; collapsed later
    # Never drop structural section headings — the LLM and section-ranker
    # both depend on them to orient in the document.
    if _SECTION_HEADER_WHITELIST.match(stripped):
        return False
    for pat in _LINE_DROP_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def clean_text(text: str) -> str:
    """Strip noise from raw extracted text.

    Processes the text in two passes:

    1. **Section-level pass** — scans for section headers that mark the start
       of non-useful content (References, Acknowledgements, etc.) and discards
       everything from that header to the next ``[PAGE N]`` marker (or end of
       text).  This preserves other sections on subsequent pages.

    2. **Line-level pass** — removes individual noisy lines (page numbers,
       DOIs, email addresses, figure captions, affiliation lines, copyright
       notices, reference-list entries, journal metadata).

    Finally, excess blank lines are collapsed to a single blank line and
    leading/trailing whitespace is stripped.

    Args:
        text: Raw text content, optionally containing ``[PAGE N]`` markers.

    Returns:
        Cleaned text with noise removed.
    """
    # ── Pass 1: section-level drop ──────────────────────────────────────────
    # Split on [PAGE N] markers and process each page block independently.
    # Within each block, once a drop-section header is found, discard the
    # rest of that block.
    page_split = re.split(r"(\[PAGE\s+\d+\])", text)
    # page_split alternates: [pre-marker-text, marker, text, marker, text, ...]

    cleaned_parts: List[str] = []
    for segment in page_split:
        if re.match(r"\[PAGE\s+\d+\]", segment):
            # Keep the marker itself
            cleaned_parts.append(segment)
            continue

        lines = segment.split("\n")
        kept: List[str] = []
        in_drop_section = False
        for line in lines:
            if in_drop_section:
                # Skip until end of this page block
                continue
            if _SECTION_DROP_HEADERS.match(line):
                in_drop_section = True
                continue
            kept.append(line)
        cleaned_parts.append("\n".join(kept))

    after_section_pass = "".join(cleaned_parts)

    # ── Pass 1b: strip structured-header blocks (spaced-letter headers +
    # short keyword/metadata lines that follow them in two-column PDFs) ──────
    structured_lines: List[str] = []
    in_structured_block = False
    for line in after_section_pass.split("\n"):
        stripped = line.strip()
        if _STRUCTURED_HEADER_START.match(stripped):
            in_structured_block = True
            continue  # drop the header itself
        if in_structured_block:
            # Exit block once a long line (real content) appears
            if len(stripped) > 60:
                in_structured_block = False
                structured_lines.append(line)
            # Drop short keyword/metadata values
            continue
        structured_lines.append(line)
    after_section_pass = "\n".join(structured_lines)

    # ── Pass 2: line-level drop ──────────────────────────────────────────
    output_lines: List[str] = []
    for line in after_section_pass.split("\n"):
        if _drop_line(line):
            continue
        output_lines.append(line)

    after_line_pass = "\n".join(output_lines)

    # ── Normalise whitespace ────────────────────────────────────────────────
    after_line_pass = _TRAILING_SPACES.sub("\n", after_line_pass)
    after_line_pass = _rejoin_broken_lines(after_line_pass)
    after_line_pass = _MULTI_BLANK.sub("\n\n", after_line_pass)
    return after_line_pass.strip()


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Strip noise from a raw .txt file extracted from a scientific PDF.")
    parser.add_argument("input", type=str, help="Path to the .txt file to clean.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="If set, truncate the cleaned output to this many characters.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the cleaned text. Defaults to stdout.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    raw = input_path.read_text(encoding="utf-8")
    cleaned = clean_text(raw)

    if args.max_chars and len(cleaned) > args.max_chars:
        cleaned = cleaned[: args.max_chars]

    if args.output:
        Path(args.output).write_text(cleaned, encoding="utf-8")
        print(f"[INFO] Cleaned text written to {args.output} ({len(cleaned)} chars)", file=sys.stderr)
    else:
        print(cleaned)


if __name__ == "__main__":
    main()
