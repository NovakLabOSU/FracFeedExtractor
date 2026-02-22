"""Classify-and-Extract Pipeline

Accepts a single PDF or a folder of PDFs, classifies each using the trained
XGBoost classifier, and passes only the files classified as "useful" to the
LLM for structured data extraction.

Usage:
    # Single PDF
    python classify-extract.py path/to/file.pdf

    # Folder of PDFs
    python classify-extract.py path/to/folder/

    # Custom options
    python classify-extract.py path/to/folder/ \\
        --model-dir src/model/models \\
        --llm-model llama3.1:8b \\
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
import csv
import sys
from pathlib import Path

from src.preprocessing.pdf_text_extraction import extract_text_from_pdf
from src.model.pdf_classifier import load_classifier, classify_text
from src.llm.llm_text import extract_key_sections
from src.llm.llm_client import extract_metrics_from_text, save_extraction_result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    model_dir: str,
    llm_model: str,
    output_dir: Path,
    confidence_threshold: float,
    max_chars: int,
    num_ctx: int,
):
    """Run classify → extract pipeline on one or more PDFs.

    For each PDF:
      1. Extract text via PyMuPDF / OCR (pdf_text_extraction.py)
      2. Classify with XGBoost (pdf_classifier.py)
      3. If 'useful': trim text to budget (llm_text.py), run LLM extraction
         (llm_client.py), and save result JSON (llm_client.py)
      4. Append a row to the summary CSV regardless of classification outcome

    Args:
        input_path: Path to a single PDF or a directory of PDFs.
        model_dir: Directory containing classifier model artifacts.
        llm_model: Ollama model name for extraction.
        output_dir: Where to write JSON results and the summary CSV.
        confidence_threshold: Classifier probability threshold for 'useful'.
        max_chars: Max characters to send to the LLM.
        num_ctx: Context window size for Ollama.
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

    # ── Load classifier once (avoid re-reading model artifacts per file) ──
    print("[INFO] Loading classifier...", file=sys.stderr)
    try:
        clf_model, vectorizer, encoder = load_classifier(model_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    print("[INFO] Classifier loaded.", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        print(f"\n[{idx}/{len(pdf_paths)}] Processing: {pdf_path.name}", file=sys.stderr)

        row = {
            "filename": pdf_path.name,
            "classification": "",
            "confidence": "",
            "pred_prob": "",
            "extraction_status": "",
            "species_name": "",
            "study_location": "",
            "study_date": "",
            "sample_size": "",
            "num_empty_stomachs": "",
            "num_nonempty_stomachs": "",
            "fraction_feeding": "",
        }

        # ── Step 1: Extract text ─────────────────
        try:
            original_text = extract_text_from_pdf(str(pdf_path))
        except Exception as e:
            print(f"  [ERROR] Text extraction failed: {e}", file=sys.stderr)
            row["extraction_status"] = "text_extraction_failed"
            summary_rows.append(row)
            continue

        if not original_text.strip():
            print(f"  [WARN] No text extracted from {pdf_path.name}. Skipping.", file=sys.stderr)
            row["extraction_status"] = "empty_text"
            summary_rows.append(row)
            continue

        print(f"  [INFO] Text size: {len(original_text)} chars", file=sys.stderr)

        # ── Step 2: Classify ──────────────────────────
        label, confidence, pred_prob = classify_text(
            text=original_text,
            model=clf_model,
            vectorizer=vectorizer,
            encoder=encoder,
            threshold=confidence_threshold,
        )
        print(f"  [CLASSIFIER] → {label} ({confidence:.2%} confidence)", file=sys.stderr)

        row["classification"] = label
        row["confidence"] = f"{confidence:.4f}"
        row["pred_prob"] = f"{pred_prob:.4f}"

        # ── Step 3: Extract ─────────────────
        if label == "useful":
            print(f"  [INFO] Running LLM extraction...", file=sys.stderr)

            # Trim text to budget using section-priority logic (llm_text.py)
            text_for_llm = original_text
            if len(text_for_llm) > max_chars:
                text_for_llm = extract_key_sections(text_for_llm, max_chars)
                print(f"  [INFO] Text trimmed to {len(text_for_llm)} chars (budget {max_chars})", file=sys.stderr)

            try:
                # LLM call (llm_client.py)
                metrics = extract_metrics_from_text(
                    text=text_for_llm,
                    model=llm_model,
                    num_ctx=num_ctx,
                )

                # Resolve source pages + save JSON (llm_client.py)
                result = save_extraction_result(
                    metrics=metrics,
                    source_file=pdf_path,
                    original_text=original_text,
                    output_dir=output_dir,
                )

                m = result["metrics"]
                row["extraction_status"] = "success"
                row["species_name"] = m.get("species_name") or ""
                row["study_location"] = m.get("study_location") or ""
                row["study_date"] = m.get("study_date") or ""
                row["sample_size"] = "" if m.get("sample_size") is None else m["sample_size"]
                row["num_empty_stomachs"] = "" if m.get("num_empty_stomachs") is None else m["num_empty_stomachs"]
                row["num_nonempty_stomachs"] = "" if m.get("num_nonempty_stomachs") is None else m["num_nonempty_stomachs"]
                row["fraction_feeding"] = "" if m.get("fraction_feeding") is None else m["fraction_feeding"]

            except Exception as e:
                print(f"  [ERROR] LLM extraction failed: {e}", file=sys.stderr)
                row["extraction_status"] = "extraction_failed"

        else:
            print(f"  [INFO] Not useful — skipping LLM extraction.", file=sys.stderr)
            row["extraction_status"] = "skipped_not_useful"

        summary_rows.append(row)

    # ── Write summary CSV ─────────────────────────────────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / f"pipeline_summary_{timestamp}.csv"
    
    fieldnames = [
        "filename", "classification", "confidence", "pred_prob",
        "extraction_status", "species_name", "study_location", "study_date",
        "sample_size", "num_empty_stomachs", "num_nonempty_stomachs", "fraction_feeding",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── Final summary ─────────────────────────────────────────────────────
    total = len(summary_rows)
    useful_count = sum(1 for r in summary_rows if r["classification"] == "useful")
    not_useful_count = sum(1 for r in summary_rows if r["classification"] == "not useful")
    extracted_count = sum(1 for r in summary_rows if r["extraction_status"] == "success")
    error_count = sum(1 for r in summary_rows if r["extraction_status"] in ("text_extraction_failed", "empty_text", "extraction_failed"))

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify PDFs as useful/not-useful, then extract structured diet "
            "metrics from useful ones using an LLM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single PDF:
    python classify-extract.py paper.pdf

  Folder of PDFs:
    python classify-extract.py data/pdfs/

  Custom options:
    python classify-extract.py data/pdfs/ \\
        --model-dir src/model/models \\
        --output-dir results/ \\
        --llm-model llama3.1:8b \\
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
        default="src/model/models",
        help="Directory containing classifier model artifacts (default: src/model/models).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model to use for extraction (default: llama3.1:8b).",
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
        default=4096,
        help="Context window size for Ollama (default: 4096).",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        input_path=input_path,
        model_dir=args.model_dir,
        llm_model=args.llm_model,
        output_dir=Path(args.output_dir),
        confidence_threshold=args.confidence_threshold,
        max_chars=args.max_chars,
        num_ctx=args.num_ctx,
    )


if __name__ == "__main__":
    main()