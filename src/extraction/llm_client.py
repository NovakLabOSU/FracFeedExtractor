"""LLM-based metric extraction from scientific publications.

Exposes two reusable functions for use by other modules:
    extract_metrics_from_text()   – call the LLM and return a PredatorDietMetrics object
    save_extraction_result()      – resolve source pages and write results to JSON

Usage (standalone):
    python llm_client.py path/to/file.pdf
    python llm_client.py path/to/file.txt
    python llm_client.py path/to/file.pdf --model qwen3:30b
    python llm_client.py path/to/file.txt --output-dir results/
"""

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

from src.config import DEFAULT_LLM_MODEL, FIELDS, build_prompt
from src.extraction.models import PredatorDietMetrics
from src.extraction.llm_text import extract_key_sections, load_document
from src.io.summary_csv import metrics_to_row, write_summary_csv
from src.utils.logger import setup_logging

log = logging.getLogger(__name__)

_LLM_TIMEOUT = 120  # seconds


def _strip_patterns(schema):
    """Recursively remove 'pattern' keys from a JSON schema before sending to Ollama.

    Pydantic's pattern constraints (regex) crash the llama.cpp GBNF grammar
    compiler when passed via format=. Stripping them here lets Ollama enforce
    field names and types via grammar while Pydantic still validates the regex
    on the Python side when model_validate_json() parses the response.
    """
    if isinstance(schema, dict):
        return {k: _strip_patterns(v) for k, v in schema.items() if k != "pattern"}
    if isinstance(schema, list):
        return [_strip_patterns(item) for item in schema]
    return schema


_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds
_ANTHROPIC_MAX_TOKENS = 512  # JSON extraction output is ~100-200 tokens across all fields
_FULL_SCHEMA = PredatorDietMetrics.model_json_schema()
_STRIPPED_SCHEMA = _strip_patterns(_FULL_SCHEMA)  # patterns removed for Ollama GBNF compiler


# Per-field hints and retryable list derived from FIELDS registry.
_hints = {spec.name: spec.hint for spec in FIELDS if spec.hint}

_FIELD_NAMES = {spec.name for spec in FIELDS}
_geocoder = None  # lazy singleton


def _get_geocoder():
    global _geocoder
    if _geocoder is None:
        from src.config import GEOCODER_USER_AGENT, GEOCODER_CACHE_PATH
        from src.extraction.geocoder import NominatimGeocoder
        _geocoder = NominatimGeocoder(GEOCODER_USER_AGENT, GEOCODER_CACHE_PATH)
    return _geocoder


# ---------------------------------------------------------------------------
# Provider detection and backend dispatch
# ---------------------------------------------------------------------------


def _detect_provider(model: str) -> str:
    return "anthropic" if model.startswith("claude-") else "ollama"


def _call_ollama(model: str, messages: list, schema: dict, options: dict) -> str:
    """Call Ollama with timeout and exponential backoff on transient failures."""
    transient_errors = (ConnectionError, OSError, TimeoutError, FuturesTimeoutError)

    if "qwen3" in model.lower():
        options = {**options, "think": False}

    for attempt in range(_MAX_RETRIES):
        try:
            from ollama import chat
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(chat, messages=messages, model=model, format=schema, options=options)
                response = future.result(timeout=_LLM_TIMEOUT)
                return response.message.content
        except FuturesTimeoutError:
            log.warning("LLM call timed out (attempt %d/%d)", attempt + 1, _MAX_RETRIES)
        except transient_errors as e:
            log.warning("Transient LLM error (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, e)

        if attempt < _MAX_RETRIES - 1:
            wait = _BACKOFF_BASE ** attempt
            log.info("Retrying in %ds...", wait)
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {_MAX_RETRIES} attempts")


def _call_anthropic(model: str, messages: list, schema: dict) -> str:
    """Call Anthropic API using tool-use to enforce structured JSON output."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=_ANTHROPIC_MAX_TOKENS,
        tools=[{
            "name": "extract_metrics",
            "description": "Extract predator diet metrics from text.",
            "input_schema": schema,
        }],
        tool_choice={"type": "tool", "name": "extract_metrics"},
        messages=messages,
    )
    if response.stop_reason != "tool_use" or not response.content:
        raise ValueError(
            f"Anthropic did not return structured output "
            f"(stop_reason={response.stop_reason!r})"
        )
    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise ValueError("Anthropic response contained no tool_use block")
    return json.dumps(tool_block.input)


def _call_llm_with_retry(model: str, messages: list, options: dict) -> str:
    """Dispatch to the correct LLM backend, selecting the appropriate schema per provider."""
    provider = _detect_provider(model)
    if provider == "anthropic":
        return _call_anthropic(model, messages, _FULL_SCHEMA)
    return _call_ollama(model, messages, _STRIPPED_SCHEMA, options)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_metrics_from_text(
    text: str,
    model: str = DEFAULT_LLM_MODEL,
    num_ctx: int = 8192,
    _retry: bool = False,
) -> PredatorDietMetrics:
    """Extract structured metrics from text using an LLM.

    On the first call, if any fields come back null the function automatically
    retries once with a focused follow-up prompt that gives method-specific
    hints for finding the missing data.

    Args:
        text: Preprocessed text content from a scientific publication.
        model: Name of the LLM model to use (Ollama or Anthropic).
        num_ctx: Context window size (passed to Ollama; ignored for Anthropic).
        _retry: Internal flag — True when this is the automatic retry attempt.

    Returns:
        PredatorDietMetrics object with extracted data.
    """
    prompt = build_prompt(text)

    content = _call_llm_with_retry(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": num_ctx},
    )

    if not content:
        raise ValueError("LLM returned empty content on initial call")
    metrics = PredatorDietMetrics.model_validate_json(content)

    # ── Retry once if any extractable fields are null ───────────────────────
    _retryable = [spec.name for spec in FIELDS if spec.retryable]
    missing = [f for f in _retryable if getattr(metrics, f) is None]

    if not _retry and missing:
        print(
            f"  [INFO] Retry: {', '.join(missing)} came back null — re-prompting",
            file=sys.stderr,
        )

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

        retry_content = _call_llm_with_retry(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            options={"num_ctx": num_ctx},
        )
        if not retry_content:
            raise ValueError("LLM returned empty content on retry")
        retry_metrics = PredatorDietMetrics.model_validate_json(retry_content)

        # Merge: prefer retry values for fields that were null, keep originals otherwise
        merged = metrics.model_dump()
        retry_dict = retry_metrics.model_dump()
        for field in _retryable:
            if merged.get(field) is None and retry_dict.get(field) is not None:
                merged[field] = retry_dict[field]

        metrics = PredatorDietMetrics.model_validate(merged)

    # ── Geocode if lat/lon are registered fields but came back null ──────────
    if (
        "latitude" in _FIELD_NAMES
        and "longitude" in _FIELD_NAMES
        and metrics.latitude is None
        and metrics.longitude is None
        and metrics.study_location
    ):
        try:
            geo = _get_geocoder().geocode(metrics.study_location)
            if geo is not None:
                metrics = metrics.model_copy(update={"latitude": geo.lat, "longitude": geo.lon})
                if geo.confidence < 0.4:
                    log.warning(
                        "Low-confidence geocode (%.2f) for '%s' → %s",
                        geo.confidence, metrics.study_location, geo.display_name,
                    )
        except Exception as e:
            log.warning("Geocoding failed for '%s': %s", metrics.study_location, e)

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
            if len(value_str) < 5:
                continue
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
    parser = argparse.ArgumentParser(description="Extract predator diet metrics from PDFs or text files using an LLM")
    parser.add_argument("input_file", type=str, help="Path to the input file (.pdf or .txt)")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help=f"LLM model to use (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--output-dir", type=str, default="data/results", help="Output directory for JSON results (default: data/results/metrics)")
    parser.add_argument("--max-chars", type=int, default=12000, help="Maximum characters of text to send to the model (default: 12000). Reduce if you hit CUDA/OOM errors.")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Context window size for Ollama (default: 8192). Ignored for Anthropic models.")

    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
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

    # Also emit a one-row summary CSV so the single-file path produces the
    # same tabular output as the batch pipeline (same columns; classifier
    # columns are left blank since this path has no classification step).
    csv_row = metrics_to_row(
        filename=input_path.name,
        metrics=result["metrics"],
        extraction_status="success",
    )
    csv_path = write_summary_csv(
        [csv_row],
        Path(args.output_dir),
        filename=input_path.stem + "_results.csv",
    )
    print(f"[SUCCESS] Summary CSV saved to {csv_path}", file=sys.stderr)

    metrics_dict = result["metrics"]
    print("\n=== Extraction Summary ===", file=sys.stderr)
    print(f"Species         : {metrics_dict.get('species_name', 'N/A')}", file=sys.stderr)
    print(f"Location        : {metrics_dict.get('study_location', 'N/A')}", file=sys.stderr)
    print(f"Date            : {metrics_dict.get('study_year', 'N/A')}", file=sys.stderr)
    print(f"Sample size     : {metrics_dict.get('num_sampled', 'N/A')}", file=sys.stderr)
    print(f"Empty stomachs  : {metrics_dict.get('num_empty', 'N/A')}", file=sys.stderr)
    print(f"Non-empty       : {metrics_dict.get('num_nonempty', 'N/A')}", file=sys.stderr)
    print(f"Fraction feeding: {metrics_dict.get('fraction_feeding', 'N/A')}", file=sys.stderr)
    print(f"Source pages    : {metrics_dict.get('source_pages', 'N/A')}", file=sys.stderr)


if __name__ == "__main__":
    main()
