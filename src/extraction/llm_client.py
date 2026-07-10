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
from src.extraction.models import ExtractionResult, PredatorDietMetrics
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
_ANTHROPIC_MAX_TOKENS = 4096  # multi-record output can easily exceed 512 tokens
_FULL_SCHEMA = ExtractionResult.model_json_schema()
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
            wait = _BACKOFF_BASE**attempt
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
        tools=[
            {
                "name": "extract_metrics",
                "description": "Extract predator diet metrics from text.",
                "input_schema": schema,
            }
        ],
        tool_choice={"type": "tool", "name": "extract_metrics"},
        messages=messages,
    )
    if response.stop_reason != "tool_use" or not response.content:
        raise ValueError(f"Anthropic did not return structured output " f"(stop_reason={response.stop_reason!r})")
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
) -> list[PredatorDietMetrics]:
    """Extract structured metrics from text using an LLM.

    Returns one PredatorDietMetrics per (species, survey) pair found in the
    text.  On the first call, if any record has retryable fields that came back
    null, the function automatically retries once with a focused follow-up
    prompt that provides the current partial extraction as context.

    Args:
        text: Preprocessed text content from a scientific publication.
        model: Name of the LLM model to use (Ollama or Anthropic).
        num_ctx: Context window size (passed to Ollama; ignored for Anthropic).
        _retry: Internal flag — True when this is the automatic retry attempt.

    Returns:
        List of PredatorDietMetrics objects, one per (species, survey) pair.
    """
    prompt = build_prompt(text)

    content = _call_llm_with_retry(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": num_ctx},
    )

    if not content:
        raise ValueError("LLM returned empty content on initial call")
    records = ExtractionResult.model_validate_json(content).records

    # ── Retry once if any record has retryable null fields ──────────────────
    _retryable = [spec.name for spec in FIELDS if spec.retryable]

    # Collect missing fields per record index: {index: [field, ...]}
    missing_per_record: dict[int, list[str]] = {}
    for i, rec in enumerate(records):
        missing = [f for f in _retryable if getattr(rec, f) is None]
        if missing:
            missing_per_record[i] = missing

    if not _retry and missing_per_record:
        missing_summary = "; ".join(
            f"record {i} ({records[i].species_name or 'unknown'}, " f"{records[i].study_location or 'unknown location'}, " f"{records[i].study_year_range or 'unknown year'}): " + ", ".join(fields)
            for i, fields in missing_per_record.items()
        )
        print(
            f"  [INFO] Retry: null fields in {len(missing_per_record)} record(s) — re-prompting",
            file=sys.stderr,
        )

        # Build per-record hint lines for only the fields that are null
        all_missing_fields: list[str] = sorted({f for fields in missing_per_record.values() for f in fields})
        hint_text = "".join(_hints.get(f, "") for f in all_missing_fields)

        current_json = ExtractionResult(records=records).model_dump_json(indent=2)
        retry_prompt = (
            "Some fields in the extraction below are null. Re-read the text carefully "
            "— especially the Abstract, Methods, and Results sections — and return a "
            "corrected JSON with the same records array and all missing fields filled in.\n\n"
            f"Current extraction:\n{current_json}\n\n"
            f"Missing fields per record:\n  {missing_summary}\n\n"
            f"Hints:\n{hint_text}"
            f"\nTEXT\n{text}"
        )

        retry_content = _call_llm_with_retry(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            options={"num_ctx": num_ctx},
        )
        if not retry_content:
            raise ValueError("LLM returned empty content on retry")
        retry_records = ExtractionResult.model_validate_json(retry_content).records

        # Merge by species_name key; fall back to positional when key is absent.
        # This handles both record-count mismatches and reordered retry results.
        if len(retry_records) != len(records):
            log.warning(
                "Retry returned %d records vs original %d — merging by key with positional fallback",
                len(retry_records),
                len(records),
            )
        retry_by_species: dict[str, PredatorDietMetrics] = {}
        for r in retry_records:
            retry_by_species[(r.species_name or "").strip().lower()] = r

        merged_records = []
        for i, orig in enumerate(records):
            orig_key = (orig.species_name or "").strip().lower()
            retry_rec = retry_by_species.get(orig_key)
            if retry_rec is None and i < len(retry_records):
                retry_rec = retry_records[i]
            if retry_rec is not None:
                orig_dict = orig.model_dump()
                retry_dict = retry_rec.model_dump()
                for field in _retryable:
                    if orig_dict.get(field) is None and retry_dict.get(field) is not None:
                        orig_dict[field] = retry_dict[field]
                merged_records.append(PredatorDietMetrics.model_validate(orig_dict))
            else:
                merged_records.append(orig)
        records = merged_records

    # ── Geocode each record that lacks lat/lon but has a study_location ──────
    if "latitude" in _FIELD_NAMES and "longitude" in _FIELD_NAMES:
        geocoded: list[PredatorDietMetrics] = []
        for rec in records:
            if rec.latitude is None and rec.longitude is None and rec.study_location:
                try:
                    geo = _get_geocoder().geocode(rec.study_location)
                    if geo is not None:
                        rec = rec.model_copy(update={"latitude": geo.lat, "longitude": geo.lon})
                        if geo.confidence < 0.4:
                            log.warning(
                                "Low-confidence geocode (%.2f) for '%s' → %s",
                                geo.confidence,
                                rec.study_location,
                                geo.display_name,
                            )
                except Exception as e:
                    log.warning("Geocoding failed for '%s': %s", rec.study_location, e)
            geocoded.append(rec)
        records = geocoded

    return records


def _resolve_source_pages(metrics_dict: dict, original_text: str) -> list[int] | None:
    """Return sorted page numbers where field values from metrics_dict appear in original_text."""
    _skip_fields = {"fraction_feeding", "source_pages"}
    source_pages: set[int] = set()
    for field_name, value in metrics_dict.items():
        if value is not None and field_name not in _skip_fields:
            value_str = str(value)
            if len(value_str) < 5:
                continue
            match = re.search(r'\b' + re.escape(value_str) + r'\b', original_text)
            if match:
                pos = match.start()
                page_markers = re.findall(r'\[PAGE (\d+)\]', original_text[:pos])
                if page_markers:
                    source_pages.add(int(page_markers[-1]))
    return sorted(source_pages) if source_pages else None


def save_extraction_result(
    records: list[PredatorDietMetrics],
    source_file: Path,
    original_text: str,
    output_dir: Path,
) -> dict:
    """Resolve source page numbers and save extraction results to JSON.

    Args:
        records: List of PredatorDietMetrics objects, one per (species, survey).
        source_file: Original PDF/text path.
        original_text: Full un-truncated extracted text (with [PAGE N] markers).
        output_dir: Directory where the JSON result file will be written.

    Returns:
        The complete result dict written to disk.
    """
    serialized_records = []
    for rec in records:
        rec_dict = rec.model_dump()
        rec_dict["source_pages"] = _resolve_source_pages(rec_dict, original_text)
        serialized_records.append(rec_dict)

    result = {
        "source_file": source_file.name,
        "file_type": source_file.suffix.lower(),
        "records": serialized_records,
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
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for JSON results (default: results/metrics)")
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
        records = extract_metrics_from_text(text, model=args.model, num_ctx=args.num_ctx)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}", file=sys.stderr)
        log.error("Metric extraction failed for %s: %s", input_path.name, e)
        sys.exit(1)

    result = save_extraction_result(
        records=records,
        source_file=input_path,
        original_text=original_text,
        output_dir=Path(args.output_dir),
    )

    # Emit one CSV row per (species, survey) record.
    csv_rows = [
        metrics_to_row(
            filename=input_path.name,
            metrics=rec_dict,
            extraction_status="success",
        )
        for rec_dict in result["records"]
    ]
    csv_path = write_summary_csv(
        csv_rows,
        Path(args.output_dir),
        filename=input_path.stem + "_results.csv",
    )
    print(f"[SUCCESS] Summary CSV saved to {csv_path}", file=sys.stderr)

    print(f"\n=== Extraction Summary ({len(result['records'])} record(s)) ===", file=sys.stderr)
    for i, rec_dict in enumerate(result["records"], start=1):
        print(f"\n  Record {i}:", file=sys.stderr)
        print(f"    Species         : {rec_dict.get('species_name', 'N/A')}", file=sys.stderr)
        print(f"    Location        : {rec_dict.get('study_location', 'N/A')}", file=sys.stderr)
        print(f"    Date            : {rec_dict.get('study_year', 'N/A')}", file=sys.stderr)
        print(f"    Sample size     : {rec_dict.get('num_sampled', 'N/A')}", file=sys.stderr)
        print(f"    Empty stomachs  : {rec_dict.get('num_empty', 'N/A')}", file=sys.stderr)
        print(f"    Non-empty       : {rec_dict.get('num_nonempty', 'N/A')}", file=sys.stderr)
        print(f"    Fraction feeding: {rec_dict.get('fraction_feeding', 'N/A')}", file=sys.stderr)
        print(f"    Source pages    : {rec_dict.get('source_pages', 'N/A')}", file=sys.stderr)


if __name__ == "__main__":
    main()
