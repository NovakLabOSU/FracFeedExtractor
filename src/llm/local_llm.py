"""LLM-based metric extraction from preprocessed text files.

Usage:
    python extract_metrics.py path/to/text_file.txt
    python extract_metrics.py path/to/text_file.txt --model llama3.1:8b
    python extract_metrics.py path/to/text_file.txt --output-dir results/

This script uses Ollama to extract structured data from preprocessed predator diet
surveys, including species name, study date, location, and stomach content data.
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Annotated, Optional

from ollama import chat
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, computed_field, constr, model_validator


class PredatorDietMetrics(BaseModel):
    """Structured schema for extracted predator diet survey metrics.

    All count fields are non-negative integers.  When both
    ``num_empty_stomachs`` and ``num_nonempty_stomachs`` are present the
    model guarantees that ``sample_size`` equals their sum.
    ``fraction_feeding`` is derived automatically from the validated counts.
    """

    model_config = ConfigDict(
        strict=True,
        validate_default=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    species_name: Optional[
        Annotated[
            str,
            constr(min_length=3, max_length=200, pattern=r"^[A-Z][a-z]+(\s[a-z]+)*$"),
        ]
    ] = Field(
        default=None,
        description="Binomial scientific name of the primary predator species studied (e.g. 'Canis lupus').",
        examples=["Canis lupus", "Vulpes vulpes"],
    )

    study_location: Optional[Annotated[str, constr(min_length=1, max_length=500)]] = Field(
        default=None,
        description="Geographic location where the study was conducted.",
        examples=["Yellowstone National Park, Wyoming, USA"],
    )

    study_date: Optional[
        Annotated[
            str,
            constr(
                min_length=4,
                max_length=30,
                pattern=r"^\d{4}([\-–]\d{4})?$",
            ),
        ]
    ] = Field(
        default=None,
        description="Year (YYYY) or year range (YYYY-YYYY) when the study was conducted.",
        examples=["2019", "2019-2021"],
    )

    num_empty_stomachs: Optional[NonNegativeInt] = Field(
        default=None,
        description="Number of predators with empty stomachs. Must be >= 0.",
    )

    num_nonempty_stomachs: Optional[NonNegativeInt] = Field(
        default=None,
        description="Number of predators with non-empty (food-containing) stomachs. Must be >= 0.",
    )

    sample_size: Optional[Annotated[int, Field(gt=0)]] = Field(
        default=None,
        description="Total number of organisms, specimens, or individuals examined or included in the study group. "
        "Must be a positive integer (> 0). This represents the sample size regardless of organism role (e.g., predator, "
        "prey, or experimental subject). Extract the largest explicitly stated numeric count of organisms from explicit "
        "counts of individuals, group sizes, or described cohort totals.",
    )

    @model_validator(mode="after")
    def _reconcile_sample_size(self) -> "PredatorDietMetrics":
        """Ensure sample_size == num_empty + num_nonempty when both counts are present."""
        empty = self.num_empty_stomachs
        nonempty = self.num_nonempty_stomachs
        if empty is not None and nonempty is not None:
            calculated = empty + nonempty
            if self.sample_size is None or self.sample_size != calculated:
                self.sample_size = calculated
        return self

    @computed_field(  # type: ignore[misc]
        description="Fraction of predators that had food in their stomachs (0.0–1.0).",
    )
    @property
    def fraction_feeding(self) -> Optional[float]:
        if self.num_nonempty_stomachs is not None and self.sample_size is not None and self.sample_size > 0:
            return round(self.num_nonempty_stomachs / self.sample_size, 4)
        return None


# ---------------------------------------------------------------------------
# Smart section extraction
# ---------------------------------------------------------------------------

# Section headers commonly found in scientific diet / stomach-content papers.
# Order matters: earlier entries are higher priority when budget is tight.
_SECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)^\s*(?:abstract|summary)"),
    re.compile(r"(?i)^\s*(?:results?)\b"),
    re.compile(r"(?i)^\s*(?:methods?|materials?\s*(?:and|&)\s*methods?|study\s*(?:area|site))"),
    re.compile(r"(?i)^\s*(?:table)\s*\d"),
    re.compile(r"(?i)^\s*(?:introduction|background)"),
    re.compile(r"(?i)^\s*(?:discussion)"),
]

# Sections that are almost never useful for metric extraction.
_SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)^\s*(?:acknowledg|literature\s*cited|references|bibliography|appendix|supplementar)"),
]


def _split_into_pages(text: str) -> list[tuple[int, str]]:
    """Split text on ``[PAGE N]`` markers.

    Returns a list of ``(page_number, page_text)`` tuples.
    """
    parts = re.split(r"\[PAGE\s+(\d+)\]", text)
    # parts: [before_first_marker, page_num, page_text, page_num, page_text, ...]
    pages: list[tuple[int, str]] = []
    if parts[0].strip():
        pages.append((0, parts[0]))
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_text = parts[i + 1] if i + 1 < len(parts) else ""
        pages.append((page_num, page_text))
    return pages


def _classify_page(page_text: str) -> tuple[bool, int]:
    """Return ``(is_useful, priority)`` for a page.

    Lower priority number == more important.
    A page that matches a skip pattern is marked not-useful.
    A page with no recognised header gets a default mid-priority.
    """
    for pat in _SKIP_PATTERNS:
        if pat.search(page_text):
            return False, 999
    for idx, pat in enumerate(_SECTION_PATTERNS):
        if pat.search(page_text):
            return True, idx
    # No recognised header – still potentially useful (e.g. tables without
    # a "Table" header, continuation of Results, etc.)
    return True, len(_SECTION_PATTERNS)


def extract_key_sections(text: str, max_chars: int) -> str:
    """Return the most informative portion of *text* within *max_chars*.

    Strategy:
    1. Split the paper into pages using ``[PAGE N]`` markers.
    2. Drop pages belonging to References / Acknowledgements / Appendix.
    3. Rank remaining pages by section priority (Abstract > Results >
       Methods > Tables > Introduction > Discussion > other).
    4. Greedily pack pages in priority order until the budget is spent.
    5. Re-order selected pages by their original page number so the LLM
       sees them in reading order.

    If the full text already fits within *max_chars* it is returned as-is.
    """
    if len(text) <= max_chars:
        return text

    pages = _split_into_pages(text)
    scored: list[tuple[int, int, str]] = []  # (priority, page_num, page_text)
    for page_num, page_text in pages:
        useful, priority = _classify_page(page_text)
        if useful:
            scored.append((priority, page_num, page_text))

    # Sort by priority (ascending = most important first)
    scored.sort(key=lambda t: t[0])

    selected: list[tuple[int, str]] = []
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


def extract_metrics_from_text(text: str, model: str = "llama3.1:8b", num_ctx: int = 4096) -> PredatorDietMetrics:
    """Extract structured metrics from text using Ollama.

    Args:
        text: Preprocessed text content from a scientific publication
        model: Name of the Ollama model to use
        num_ctx: Context window size to request from Ollama (lower = less memory)

    Returns:
        PredatorDietMetrics object with extracted data
    """
    prompt = f"""You are a scientific data extraction assistant specializing in predator diet surveys.

Extract specific metrics from the text below. Focus on stomach content data where:
- EMPTY stomachs = no food/prey
- NON-EMPTY stomachs = contained food/prey
- SAMPLE SIZE = total number of predators examined

KEY INFORMATION TO FIND:
- Species names are in Latin format (Genus species)
- Look in tables, methods, and results sections
- Empty stomachs: "empty", "vacant", "no prey"
- Non-empty stomachs: "with prey", "fed", "containing food"
- Page markers appear as [PAGE N] in the text

EXTRACT:
- species_name: Scientific name of PRIMARY predator studied (not prey)
- study_location: Geographic location of sampling
- study_date: Year or date range of collection
- num_empty_stomachs: Number with empty stomachs
- num_nonempty_stomachs: Number with food in stomachs
- sample_size: Total number examined


TEXT:
{text}
"""
    # Ollama call with structured schema output
    response = chat(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model=model,
        format=PredatorDietMetrics.model_json_schema(),
    )

    metrics = PredatorDietMetrics.model_validate_json(response.message.content)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Extract predator diet metrics from preprocessed text using LLM")
    parser.add_argument("text_file", type=str, help="Path to the preprocessed text file")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama model to use (default: llama3.1:8b)")
    parser.add_argument("--output-dir", type=str, default="data/results", help="Output directory for JSON results (default: data/results)")
    parser.add_argument("--max-chars", type=int, default=12000, help="Maximum characters of text to send to the model (default: 12000). Reduce if you hit CUDA/OOM errors.")
    parser.add_argument("--num-ctx", type=int, default=4096, help="Context window size for the model (default: 4096). Lower values use less memory.")

    args = parser.parse_args()

    # Load text file
    text_path = Path(args.text_file)
    if not text_path.exists():
        print(f"[ERROR] File not found: {text_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract metrics
    print(f"Extracting metrics from {text_path.name}...", file=sys.stderr)

    # Store original text for page extraction
    original_text = text
    print(f"[INFO] Text size: {len(text)} chars", file=sys.stderr)

    if len(text) > args.max_chars:
        text = extract_key_sections(text, args.max_chars)
        print(f"[INFO] Extracted key sections: {len(text)} chars (budget {args.max_chars})", file=sys.stderr)

    try:
        metrics = extract_metrics_from_text(text, model=args.model, num_ctx=args.num_ctx)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    # model_validator reconciles sample_size; computed_field provides fraction_feeding
    metrics_dict = metrics.model_dump()

    # Extract page numbers programmatically from where data was found
    source_pages: set[int] = set()
    _skip_fields = {"fraction_feeding", "source_pages"}
    for field_name, value in metrics_dict.items():
        if value is not None and field_name not in _skip_fields:
            value_str = str(value)
            if value_str in original_text:
                pos = original_text.find(value_str)
                page_markers = re.findall(r'\[PAGE (\d+)\]', original_text[:pos])
                if page_markers:
                    source_pages.add(int(page_markers[-1]))

    metrics_dict["source_pages"] = sorted(source_pages) if source_pages else None

    # Prepare output
    result = {"source_file": text_path.name, "metrics": metrics_dict}

    # Generate output filename: input_name_results.json
    output_filename = text_path.stem + "_results.json"
    output_path = Path(args.output_dir) / output_filename

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
