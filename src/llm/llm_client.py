"""LLM-based metric extraction from preprocessed text files.

Usage:
    python local_llm.py path/to/text_file.txt
    python local_llm.py path/to/text_file.txt --model llama3.1:8b
    python local_llm.py path/to/text_file.txt --output-dir results/

This script uses Ollama to extract structured data from preprocessed predator diet
surveys, including species name, study date, location, and stomach content data.
"""

import argparse
import json
import sys
import re
from pathlib import Path

from ollama import chat

from models import PredatorDietMetrics
from llm_text import extract_key_sections


def extract_metrics_from_text(text: str, model: str = "llama3.1:8b", num_ctx: int = 4096) -> PredatorDietMetrics:
    """Extract structured metrics from text using Ollama.

    Args:
        text: Preprocessed text content from a scientific publication
        model: Name of the Ollama model to use
        num_ctx: Context window size to request from Ollama (lower = less memory)

    Returns:
        PredatorDietMetrics object with extracted data
    """
    prompt = f"""You are a scientific data extraction assistant. Your task is to read a predator diet survey publication and return a single flat JSON object with exactly these fields:

  species_name          - string or null
  study_location        - string or null
  study_date            - string or null
  num_empty_stomachs    - integer (>= 0) or null
  num_nonempty_stomachs - integer (>= 0) or null
  sample_size           - integer (> 0) or null

Use null for any field whose value cannot be confidently determined from the text.

FIELD DEFINITIONS

species_name: Binomial Latin name (Genus species) of the PRIMARY PREDATOR whose diet is studied. This is the animal whose stomachs/guts were examined, not its prey. Return exactly one species. If multiple predators are studied, choose the one with the most stomach samples. Capitalize the genus, lowercase the specific epithet (e.g., "Pygoscelis papua").

study_location: Geographic area where predator specimens were collected. Include site, region, and country if available (e.g., "Marion Island, sub-Antarctic"). Check Methods, Study Area, or Study Site sections.

study_date: Year or year-range of specimen collection, NOT publication year. Format "YYYY" or "YYYY-YYYY". Look for phrases like "specimens collected in", "sampling period", "field season", "between [year] and [year]". Return null if only publication year is visible.

num_empty_stomachs: Number of predators with stomachs containing no food. Synonyms: "empty", "vacant", "without food", "zero prey items", "stomachs with no contents", "N individuals had empty stomachs".

num_nonempty_stomachs: Number of predators with stomachs containing food. Synonyms: "non-empty", "with food", "containing prey", "with contents", "fed", "N contained food", "N had prey items".

sample_size: Total number of predator individuals examined. When both num_empty_stomachs and num_nonempty_stomachs are available, sample_size equals their sum. Look for phrases like "N stomachs were examined", "a total of N individuals", "N specimens", "n=", "sample size of N".


RULES
- Do not invent data; use null if ambiguous or missing.
- Return a single JSON object; do not return arrays.
- Ignore page markers [PAGE N].
- Prioritize Abstract, Methods, and Results sections.
- Be especially careful to distinguish collection dates from publication dates.

EXAMPLES

1. Simple complete case:
{{"species_name": "Pygoscelis papua", "study_location": "Marion Island, sub-Antarctic", "study_date": "1984-1985", "num_empty_stomachs": 5, "num_nonempty_stomachs": 15, "sample_size": 20}}

2. Missing empty stomach data (can infer from sample_size):
{{"species_name": "Canis lupus", "study_location": "Yellowstone National Park, Wyoming, USA", "study_date": "2019", "num_empty_stomachs": null, "num_nonempty_stomachs": 47, "sample_size": 52}}

3. Multi-year study:
{{"species_name": "Vulpes vulpes", "study_location": "Bristol, UK", "study_date": "2015-2018", "num_empty_stomachs": 12, "num_nonempty_stomachs": 88, "sample_size": 100}}

4. Minimal data available:
{{"species_name": "Ursus arctos", "study_location": null, "study_date": "2020", "num_empty_stomachs": null, "num_nonempty_stomachs": null, "sample_size": 23}}

5. Only some fields extractable:
{{"species_name": "Zalophus californianus", "study_location": "California coast", "study_date": null, "num_empty_stomachs": 8, "num_nonempty_stomachs": 34, "sample_size": 42}}

TEXT
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