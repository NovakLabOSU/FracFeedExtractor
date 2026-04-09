"""LLM-based metric extraction from scientific publications.

Exposes two reusable functions for use by other modules:
    extract_metrics_from_text()   – call Ollama and return a PredatorDietMetrics object
    save_extraction_result()      – resolve source pages and write results to JSON

Usage (standalone):
    python llm_client.py path/to/file.pdf
    python llm_client.py path/to/file.txt
    python llm_client.py path/to/file.pdf --model llama3.1:8b
    python llm_client.py path/to/file.txt --output-dir results/
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from ollama import chat

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm.models import PredatorDietMetrics
from src.llm.llm_text import extract_key_sections, load_document
from src.utils.logger import setup_logging

log = logging.getLogger(__name__)


def extract_metrics_from_text(
    text: str,
    # model: str = "llama3.1:8b",
    model: str = "qwen2.5:7b",
    num_ctx: int = 8192,
    _retry: bool = False,
) -> PredatorDietMetrics:
    """Extract structured metrics from text using Ollama.

    On the first call, if any fields come back null the function automatically
    retries once with a focused follow-up prompt that gives method-specific
    hints for finding the missing data.

    Args:
        text: Preprocessed text content from a scientific publication.
        model: Name of the Ollama model to use.
        num_ctx: Context window size to request from Ollama (lower = less memory).
        _retry: Internal flag — True when this is the automatic retry attempt.

    Returns:
        PredatorDietMetrics object with extracted data.
    """
    prompt = f"""You are a scientific data extraction assistant. Your task is to read a predator diet study and return a single flat JSON object with exactly these fields:

  species_name          - string or null
  study_location        - string or null
  study_date            - string or null
  num_empty_stomachs    - integer (>= 0) or null
  num_nonempty_stomachs - integer (>= 0) or null
  sample_size           - integer (> 0) or null

Use null ONLY when the value truly cannot be determined from any part of the text.

FIELD DEFINITIONS

species_name: Binomial Latin name (Genus species) of the PRIMARY PREDATOR whose diet is studied. This is the animal being studied, not its prey. Return exactly one species. If multiple predators appear, choose the one with the most samples. Capitalize genus, lowercase epithet (e.g., "Pygoscelis papua").

study_location: Geographic area where specimens were collected. Include site, region, and country if available (e.g., "Marion Island, sub-Antarctic"). Check Methods, Study Area, Study Site, and Abstract.

study_date: Year or year-range of specimen collection, NOT publication year. Format "YYYY" or "YYYY-YYYY".
  Where to look:
  - "specimens collected in", "sampling period", "field season", "between [year] and [year]"
  - "from March 1984 to March 1985" → "1984-1985"
  - "Received 23 November 2007" in article info suggests collection was ~2005-2007
  - If the only dates are "Received" or "Accepted" submission dates and no collection dates are stated, estimate the collection period as 1-2 years before submission.

num_empty_stomachs: Number of predators with NO food in their digestive tract. Apply broadly across study methods:
  - Stomach dissection: "empty", "vacant", "without food", "zero prey items"
  - Stomach pumping / lavage: "yielded no food", "no contents obtained", "produced no material"
  - Scat / fecal analysis: "scats with no identifiable prey", "empty scats"
  - Regurgitation: "failed to regurgitate", "no pellet produced"
  - Immunoassay / molecular: "tested negative for all prey", "no prey detected"
  If the study uses stomach pumping and ALL samples contained food, set this to 0.

num_nonempty_stomachs: Number of predators with food in their digestive tract. Same method mapping as above:
  - "non-empty", "with food", "containing prey", "with contents", "fed"
  - Stomach pumping: "food samples collected", "samples containing prey"
  - Scat: "scats with identifiable prey remains"
  - Immunoassay: "tested positive for prey", "positive reactions"
  If study says "a total of N food samples was collected" and implies ALL had food, set num_nonempty_stomachs = N.

sample_size: Total number of predator individuals examined. Equals num_empty + num_nonempty when both are known.
  - "N stomachs examined", "N individuals", "N specimens", "n=N", "a total of N"
  - "N food samples" when all sampled animals contributed one sample
  - "two groups of 225" → sample_size = 450
  - Check Abstract, Methods, and Results.

RULES
- Do not invent data; use null only if truly ambiguous or missing.
- Return a single JSON object; do not return arrays.
- Ignore page markers [PAGE N].
- Prioritize Abstract, Methods, and Results sections.
- Carefully distinguish collection dates from publication/submission dates.
- If ALL samples had food (e.g., stomach pumping where every sample yielded prey), set num_empty_stomachs = 0 and num_nonempty_stomachs = sample_size.

EXAMPLES

1. Traditional stomach dissection:
{{"species_name": "Canis lupus", "study_location": "Yellowstone National Park, Wyoming, USA", "study_date": "2019", "num_empty_stomachs": 5, "num_nonempty_stomachs": 47, "sample_size": 52}}

2. Stomach pumping (all samples had food):
{{"species_name": "Pygoscelis papua", "study_location": "Marion Island, sub-Antarctic", "study_date": "1984-1985", "num_empty_stomachs": 0, "num_nonempty_stomachs": 144, "sample_size": 144}}

3. Immunoassay / molecular detection:
{{"species_name": "Nucella lapillus", "study_location": "Swans Island, Maine, USA", "study_date": "2005-2007", "num_empty_stomachs": null, "num_nonempty_stomachs": null, "sample_size": 450}}

4. Scat / fecal analysis:
{{"species_name": "Vulpes vulpes", "study_location": "Bristol, UK", "study_date": "2015-2018", "num_empty_stomachs": 12, "num_nonempty_stomachs": 88, "sample_size": 100}}

5. Minimal data:
{{"species_name": "Ursus arctos", "study_location": null, "study_date": "2020", "num_empty_stomachs": null, "num_nonempty_stomachs": null, "sample_size": 23}}

TEXT
{text}
"""
    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        format=PredatorDietMetrics.model_json_schema(),
        options={"num_ctx": num_ctx},
    )

    metrics = PredatorDietMetrics.model_validate_json(response.message.content)

    # ── Retry once if any extractable fields are null ───────────────────────
    _retryable = [
        "species_name",
        "study_location",
        "study_date",
        "num_empty_stomachs",
        "num_nonempty_stomachs",
        "sample_size",
    ]
    missing = [f for f in _retryable if getattr(metrics, f) is None]

    if not _retry and missing:
        print(
            file=sys.stderr,
        )

        # Build targeted hints for each missing field
        _hints = {
            "species_name": ("- species_name: Look for the first binomial Latin name (Genus species) " "in the title or abstract. This is the PREDATOR, not its prey.\n"),
            "study_location": ("- study_location: Check Methods or Study Area sections for place names, " "islands, countries, or coordinates.\n"),
            "study_date": (
                "- study_date: Look for phrases like 'collected in', 'sampled during', "
                "'field season', 'from [month] [year] to [month] [year]'. "
                "If no collection date is explicit, infer from 'Received [date]' — "
                "collection is typically 1-2 years before manuscript submission.\n"
            ),
            "num_empty_stomachs": (
                "- num_empty_stomachs: Look for 'empty', 'no food', 'no contents', "
                "'negative for prey'. If ALL samples had food (e.g., stomach pumping "
                "where every sample produced material), return 0.\n"
            ),
            "num_nonempty_stomachs": (
                "- num_nonempty_stomachs: Look for 'contained food', 'with prey', " "'non-empty', 'food samples collected'. If ALL samples had food, " "this equals sample_size.\n"
            ),
            "sample_size": ("- sample_size: Look for 'N stomachs', 'N specimens', 'a total of N', " "'n=N', 'N individuals examined', 'two groups of N'. Check Abstract, " "Methods, and Results.\n"),
        }

        retry_prompt = (
            "The following fields were returned as null. Please re-read the text "
            "carefully — especially the Abstract, Methods, and Results sections — "
            "and try harder to find values for them. Think about different study "
            "methods (stomach pumping, scat analysis, immunoassays, etc.).\n\n"
            f"Missing fields: {', '.join(missing)}\n\n"
            "Hints:\n"
        )
        for field in missing:
            retry_prompt += _hints.get(field, "")
        retry_prompt += f"\nTEXT\n{text}"

        retry_response = chat(
            messages=[{"role": "user", "content": retry_prompt}],
            model=model,
            format=PredatorDietMetrics.model_json_schema(),
            options={"num_ctx": num_ctx},
        )
        retry_metrics = PredatorDietMetrics.model_validate_json(retry_response.message.content)

        # Merge: prefer retry values for fields that were null, keep originals otherwise
        merged = metrics.model_dump()
        retry_dict = retry_metrics.model_dump()
        for field in _retryable:
            if merged.get(field) is None and retry_dict.get(field) is not None:
                merged[field] = retry_dict[field]

        metrics = PredatorDietMetrics.model_validate(merged)

    return metrics


def save_extraction_result(
    metrics: PredatorDietMetrics,
    source_file: Path,
    original_text: str,
    output_dir: Path,
) -> dict:
    """Resolve source page numbers and save extraction results to JSON.

    Args:
        metrics: Populated PredatorDietMetrics object.
        source_file: Original PDF/text path.
        original_text: Full un-truncated extracted text (with [PAGE N] markers).
        output_dir: Directory where the JSON result file will be written.

    Returns:
        The complete result dict written to disk.
    """
    metrics_dict = metrics.model_dump()

    _skip_fields = {"fraction_feeding", "source_pages"}
    source_pages: set[int] = set()
    for field_name, value in metrics_dict.items():
        if value is not None and field_name not in _skip_fields:
            value_str = str(value)
            if value_str in original_text:
                pos = original_text.find(value_str)
                page_markers = re.findall(r'\[PAGE (\d+)\]', original_text[:pos])
                if page_markers:
                    source_pages.add(int(page_markers[-1]))

    metrics_dict["source_pages"] = sorted(source_pages) if source_pages else None

    result = {
        "source_file": source_file.name,
        "file_type": source_file.suffix.lower(),
        "metrics": metrics_dict,
    }

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / (source_file.stem + "_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[SUCCESS] Results saved to {output_path}", file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract predator diet metrics from PDFs or text files using LLM")
    parser.add_argument("input_file", type=str, help="Path to the input file (.pdf or .txt)")
    # parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama model to use (default: llama3.1:8b)")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Ollama model to use (default: qwen2.5:7b)")
    parser.add_argument("--output-dir", type=str, default="data/results", help="Output directory for JSON results (default: data/results/metrics)")
    parser.add_argument("--max-chars", type=int, default=12000, help="Maximum characters of text to send to the model (default: 12000). Reduce if you hit CUDA/OOM errors.")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Context window size for the model (default: 8192). Lower values use less memory.")

    args = parser.parse_args()

    setup_logging()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}", file=sys.stderr)
        log.error("File not found: %s", input_path)
        sys.exit(1)

    print(f"Processing {input_path.name}...", file=sys.stderr)
    try:
        original_text = load_document(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}", file=sys.stderr)
        log.error("Failed to load file %s: %s", input_path, e)
        sys.exit(1)

    print(f"[INFO] Text size: {len(original_text)} chars", file=sys.stderr)

    text = original_text
    if len(text) > args.max_chars:
        text = extract_key_sections(text, args.max_chars)
        print(f"[INFO] Extracted key sections: {len(text)} chars (budget {args.max_chars})", file=sys.stderr)

    print(f"[INFO] Extracting metrics with {args.model}...", file=sys.stderr)
    try:
        metrics = extract_metrics_from_text(text, model=args.model, num_ctx=args.num_ctx)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}", file=sys.stderr)
        log.error("Metric extraction failed for %s: %s", input_path.name, e)
        sys.exit(1)

    result = save_extraction_result(
        metrics=metrics,
        source_file=input_path,
        original_text=original_text,
        output_dir=Path(args.output_dir),
    )

    metrics_dict = result["metrics"]
    print("\n=== Extraction Summary ===", file=sys.stderr)
    print(f"Species         : {metrics_dict.get('species_name', 'N/A')}", file=sys.stderr)
    print(f"Location        : {metrics_dict.get('study_location', 'N/A')}", file=sys.stderr)
    print(f"Date            : {metrics_dict.get('study_date', 'N/A')}", file=sys.stderr)
    print(f"Sample size     : {metrics_dict.get('sample_size', 'N/A')}", file=sys.stderr)
    print(f"Empty stomachs  : {metrics_dict.get('num_empty_stomachs', 'N/A')}", file=sys.stderr)
    print(f"Non-empty       : {metrics_dict.get('num_nonempty_stomachs', 'N/A')}", file=sys.stderr)
    print(f"Fraction feeding: {metrics_dict.get('fraction_feeding', 'N/A')}", file=sys.stderr)
    print(f"Source pages    : {metrics_dict.get('source_pages', 'N/A')}", file=sys.stderr)


if __name__ == "__main__":
    main()
