"""Classify-and-Extract Pipeline

Accepts a single PDF or a folder of PDFs, classifies each using the trained
XGBoost classifier, and passes only the files classified as "useful" to the
LLM for structured data extraction.

Usage:
    # Single PDF
    python src/pipeline/classify_extract.py path/to/file.pdf

    # Folder of PDFs
    python src/pipeline/classify_extract.py path/to/folder/

    # Custom options
    python src/pipeline/classify_extract.py path/to/folder/ \\
        --model-dir src/classifier/models \\
        --llm-model qwen3:30b \\
        --output-dir results/ \\
        --confidence-threshold 0.70 \\
        --max-chars 12000 \\
        --num-ctx 4096

Output:
    - One JSON file per useful PDF (in --output-dir) containing extracted metrics.
    - A summary CSV (pipeline_summary.csv) in --output-dir listing every PDF,
      its classification, confidence, and extraction status.
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.config import DEFAULT_LLM_MODEL
from src.io.pdf_text_extraction import extract_text_from_pdf
from src.io.text_cleaner import clean_text
from src.io.section_filter import filter_relevant_sections
from src.io.summary_csv import blank_row, metrics_to_row, write_summary_csv
from src.classifier.pdf_classifier import load_classifier, classify_text, explain_classification
from src.extraction.llm_text import extract_key_sections
from src.extraction.llm_client import extract_metrics_from_text, save_extraction_result
from src.utils.logger import setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _process_single_pdf(
    pdf_path: Path,
    llm_model: str,
    output_dir: Path,
    confidence_threshold: float,
    max_chars: int,
    num_ctx: int,
    clf_model,
    vectorizer,
    encoder,
    skip_classifier: bool = False,
    explain: bool = False,
) -> list[dict]:
    """Classify one PDF and return a list of summary row dicts (one per record).

    Returns a single-element list on failure or when the paper is not useful,
    and one element per extracted (species, survey) record on success.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_row = blank_row(pdf_path.name)

    # ── Step 1: Extract text ──────────────────────────────────────────
    try:
        original_text = extract_text_from_pdf(str(pdf_path))
    except Exception as e:
        print(f"  [ERROR] Text extraction failed ({pdf_path.name}): {e}", file=sys.stderr)
        log.error("Text extraction failed for %s: %s", pdf_path.name, e)
        base_row["extraction_status"] = "text_extraction_failed"
        return [base_row]

    if not original_text.strip():
        print(f"  [WARN] No text extracted from {pdf_path.name}. Skipping.", file=sys.stderr)
        log.warning("No text extracted from %s — skipping.", pdf_path.name)
        base_row["extraction_status"] = "empty_text"
        return [base_row]

    print(f"  [INFO] {pdf_path.name}: {len(original_text)} chars", file=sys.stderr)

    # ── Step 2: Classify ──────────────────────────────────────────────
    if skip_classifier:
        label, confidence, pred_prob = "useful", 1.0, 1.0
        print(f"  [CLASSIFIER] {pdf_path.name} → classifier skipped (--skip-classifier)", file=sys.stderr)
    else:
        label, confidence, pred_prob = classify_text(
            text=original_text,
            model=clf_model,
            vectorizer=vectorizer,
            encoder=encoder,
            threshold=confidence_threshold,
        )
        print(f"  [CLASSIFIER] {pdf_path.name} → {label} ({confidence:.2%})", file=sys.stderr)

    base_row["classification"] = label
    base_row["confidence"] = f"{confidence:.4f}"
    base_row["pred_prob"] = f"{pred_prob:.4f}"

    # ── Step 2b: Explain (optional) ───────────────────────────────────
    if label == "useful" and explain and not skip_classifier:
        top_terms = explain_classification(original_text, clf_model, vectorizer, top_n=10)
        if top_terms:
            print(f"  [EXPLAIN] Top classifier keywords for {pdf_path.name}:", file=sys.stderr)
            for term, score, pages in top_terms:
                page_str = f" (p. {pages[0]})" if pages else ""
                print(f"    {term:<30} score={score:.2f}{page_str}", file=sys.stderr)
        base_row["top_keywords"] = "; ".join(t for t, _, _ in top_terms)

    # ── Step 3: Extract ───────────────────────────────────────────────
    if label == "useful":
        print(f"  [INFO] {pdf_path.name}: Running LLM extraction...", file=sys.stderr)

        # Apply the same preprocessing chain as the txt pipeline:
        # clean noise → filter irrelevant sections → trim to LLM budget
        text_for_llm = clean_text(original_text)
        text_for_llm = filter_relevant_sections(text_for_llm)
        if len(text_for_llm) > max_chars:
            text_for_llm = extract_key_sections(text_for_llm, max_chars)
            print(f"  [INFO] {pdf_path.name}: trimmed to {len(text_for_llm)} chars (budget {max_chars})", file=sys.stderr)

        try:
            records = extract_metrics_from_text(
                text=text_for_llm,
                model=llm_model,
                num_ctx=num_ctx,
            )
            result = save_extraction_result(
                records=records,
                source_file=pdf_path,
                original_text=original_text,
                output_dir=output_dir,
            )

            rows = [
                metrics_to_row(
                    filename=pdf_path.name,
                    metrics=rec_dict,
                    classification=base_row["classification"],
                    confidence=base_row["confidence"],
                    pred_prob=base_row["pred_prob"],
                    extraction_status="success",
                )
                for rec_dict in result["records"]
            ]
            return rows if rows else [dict(base_row, extraction_status="no_records")]

        except Exception as e:
            print(f"  [ERROR] LLM extraction failed ({pdf_path.name}): {e}", file=sys.stderr)
            log.error("LLM extraction failed for %s: %s", pdf_path.name, e)
            base_row["extraction_status"] = "extraction_failed"

    else:
        print(f"  [INFO] {pdf_path.name}: Not useful — skipping LLM extraction.", file=sys.stderr)
        base_row["extraction_status"] = "skipped_not_useful"

    return [base_row]


def run_pipeline(
    input_path: Path,
    model_dir: str,
    llm_model: str,
    output_dir: Path,
    confidence_threshold: float,
    max_chars: int,
    num_ctx: int,
    workers: int = 1,
    skip_classifier: bool = False,
    explain: bool = False,
):
    """Run classify → extract pipeline on one or more PDFs.

    For each PDF:
      1. Extract text via PyMuPDF / OCR (pdf_text_extraction.py)
      2. Classify with XGBoost (pdf_classifier.py) — skipped when skip_classifier=True
      3. If 'useful': clean noise, filter sections, trim to budget, run LLM extraction,
         and save result JSON
      4. Append a row to the summary CSV regardless of classification outcome

    Args:
        input_path: Path to a single PDF or a directory of PDFs.
        model_dir: Directory containing classifier model artifacts.
        llm_model: LLM model name for extraction.
        output_dir: Where to write JSON results and the summary CSV.
        confidence_threshold: Classifier probability threshold for 'useful'.
        max_chars: Max characters to send to the LLM.
        num_ctx: Context window size (passed to Ollama; ignored for Anthropic).
        workers: Number of parallel worker processes (default: 1 = sequential).
        skip_classifier: When True, treat every PDF as useful and skip classification.
    """
    # ── Collect PDF paths ─────────────────────────────────────────────────
    if input_path.is_dir():
        pdf_paths = sorted(input_path.glob("*.pdf"))
        if not pdf_paths:
            print(f"[ERROR] No PDF files found in directory: {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Found {len(pdf_paths)} PDF(s) in {input_path}", file=sys.stderr)
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_paths = [input_path]
    else:
        print(f"[ERROR] Input must be a .pdf file or a directory of PDFs: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_classifier:
        print("[INFO] --skip-classifier: treating all PDFs as useful, skipping classification.", file=sys.stderr)
        clf_model = vectorizer = encoder = None
    else:
        print("[INFO] Loading classifier...", file=sys.stderr)
        try:
            clf_model, vectorizer, encoder = load_classifier(model_dir)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            log.critical("Classifier artifacts not found: %s", e)
            sys.exit(1)
        print("[INFO] Classifier loaded.", file=sys.stderr)

    # Per-PDF rows lists; each inner list has ≥1 row (one per record on success).
    per_pdf_rows: list[list[dict]] = []

    if workers > 1 and len(pdf_paths) > 1:
        print(f"[INFO] Using {workers} worker processes.", file=sys.stderr)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_single_pdf,
                    pdf_path,
                    llm_model,
                    output_dir,
                    confidence_threshold,
                    max_chars,
                    num_ctx,
                    clf_model,
                    vectorizer,
                    encoder,
                    skip_classifier,
                    explain,
                ): pdf_path
                for pdf_path in pdf_paths
            }
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:
                    print(f"  [ERROR] Worker failed for {pdf_path.name}: {exc}", file=sys.stderr)
                    row = blank_row(pdf_path.name)
                    row["extraction_status"] = "worker_failed"
                    rows = [row]
                per_pdf_rows.append(rows)
    else:
        for idx, pdf_path in enumerate(pdf_paths, start=1):
            print(f"\n[{idx}/{len(pdf_paths)}] Processing: {pdf_path.name}", file=sys.stderr)
            rows = _process_single_pdf(
                pdf_path,
                llm_model,
                output_dir,
                confidence_threshold,
                max_chars,
                num_ctx,
                clf_model,
                vectorizer,
                encoder,
                skip_classifier,
                explain,
            )
            per_pdf_rows.append(rows)

    # Flatten: one CSV row per (paper × record)
    summary_rows = [row for rows in per_pdf_rows for row in rows]

    # ── Write summary CSV ─────────────────────────────────────────────────
    summary_path = write_summary_csv(summary_rows, output_dir)

    # ── Final summary (stats counted per PDF, not per row) ────────────────
    total = len(per_pdf_rows)
    # Use first row of each PDF for classification/status (all rows from one PDF share these)
    first_rows = [rows[0] for rows in per_pdf_rows]
    useful_count = sum(1 for r in first_rows if r["classification"] == "useful")
    not_useful_count = sum(1 for r in first_rows if r["classification"] == "not useful")
    extracted_count = sum(1 for r in first_rows if r["extraction_status"] == "success")
    error_count = sum(1 for r in first_rows if r["extraction_status"] in ("text_extraction_failed", "empty_text", "extraction_failed", "worker_failed", "no_records"))

    print("\n" + "=" * 50, file=sys.stderr)
    print("PIPELINE COMPLETE", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"  Total PDFs processed  : {total}", file=sys.stderr)
    print(f"  Useful                : {useful_count}", file=sys.stderr)
    print(f"  Not useful            : {not_useful_count}", file=sys.stderr)
    print(f"  Successfully extracted: {extracted_count}", file=sys.stderr)
    print(f"  Errors                : {error_count}", file=sys.stderr)
    print(f"  Summary CSV           : {summary_path}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    if error_count > 0:
        log.warning("Pipeline finished with %d error(s). See logs/fracfeed.log for details.", error_count)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=("Classify PDFs as useful/not-useful, then extract structured diet " "metrics from useful ones using an LLM."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single PDF:
    python src/pipeline/classify_extract.py paper.pdf

  Folder of PDFs:
    python src/pipeline/classify_extract.py data/pdfs/

  Custom options:
    python src/pipeline/classify_extract.py data/pdfs/ \\
        --model-dir src/classifier/models \\
        --output-dir results/ \\
        --llm-model qwen3:30b \\
        --confidence-threshold 0.70
        """,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a single PDF file or a directory containing PDF files.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="src/classifier/models",
        help="Directory containing classifier model artifacts (default: src/classifier/models).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model to use for extraction (default: {DEFAULT_LLM_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory for JSON results and summary CSV (default: data/results).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.70,
        help="Classifier probability threshold for 'useful' (default: 0.70).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Max characters of text to send to the LLM (default: 12000).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Context window size for Ollama (default: 8192). Ignored for Anthropic models.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1 = sequential).",
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        default=False,
        help="Skip the XGBoost classifier and treat every PDF as useful. "
             "Useful when PDFs are already known to be relevant.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="For each useful PDF, print the top classifier keywords and the page "
             "numbers where they appear. Adds 'top_keywords' to the summary CSV.",
    )

    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    # Configure persistent logging for this process — one call covers all modules
    setup_logging()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
        log.error("Input path not found: %s", input_path)
        sys.exit(1)

    run_pipeline(
        input_path=input_path,
        model_dir=args.model_dir,
        llm_model=args.llm_model,
        output_dir=Path(args.output_dir),
        confidence_threshold=args.confidence_threshold,
        max_chars=args.max_chars,
        num_ctx=args.num_ctx,
        workers=args.workers,
        skip_classifier=args.skip_classifier,
        explain=args.explain,
    )


if __name__ == "__main__":
    main()
