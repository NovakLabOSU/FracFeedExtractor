"""Classify-and-Extract Pipeline

Single entry point for the full pipeline: PDF classify → extract, or
pre-classified .txt extract (with ``--skip-classifier``).

Usage:
    # PDF folder (classify then extract useful papers)
    python src/pipeline/classify_extract.py path/to/folder/

    # Single PDF
    python src/pipeline/classify_extract.py path/to/file.pdf

    # Pre-classified .txt files (no XGBoost step)
    python src/pipeline/classify_extract.py path/to/txts/ --skip-classifier

    # Custom options
    python src/pipeline/classify_extract.py path/to/folder/ \\
        --model-dir src/classifier/models \\
        --llm-model qwen2.5:7b \\
        --output-dir results/ \\
        --confidence-threshold 0.70 \\
        --max-chars 12000 \\
        --num-ctx 4096

Output:
    - One JSON file per useful PDF or processed .txt (in --output-dir).
    - A summary CSV in --output-dir/summaries/ listing each file and status.
"""

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.io.pdf_text_extraction import extract_text_from_pdf
from src.io.text_cleaner import clean_text
from src.io.section_filter import filter_relevant_sections
from src.classifier.pdf_classifier import load_classifier, classify_text
from src.extraction.llm_text import extract_key_sections
from src.extraction.llm_client import extract_metrics_from_text, save_extraction_result
from src.extraction.chunked_extraction import extract_with_chunking
from src.utils.logger import setup_logging

log = logging.getLogger(__name__)

METRIC_FIELDNAMES = [
    "species_name",
    "study_location",
    "study_date",
    "sample_size",
    "num_empty_stomachs",
    "num_nonempty_stomachs",
    "fraction_feeding",
]
CLASSIFIER_FIELDNAMES = ["classification", "confidence", "pred_prob"]
PREPROCESS_FIELDNAMES = ["raw_chars", "cleaned_chars", "filtered_chars", "trimmed_chars"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _summary_fieldnames(*, include_classifier: bool, include_preprocess: bool) -> list[str]:
    fields = ["filename"]
    if include_classifier:
        fields.extend(CLASSIFIER_FIELDNAMES)
    if include_preprocess:
        fields.extend(PREPROCESS_FIELDNAMES)
    fields.append("extraction_status")
    fields.extend(METRIC_FIELDNAMES)
    return fields


def _new_summary_row(filename: str, *, include_classifier: bool, include_preprocess: bool) -> dict:
    row = {"filename": filename, "extraction_status": ""}
    if include_classifier:
        row.update({k: "" for k in CLASSIFIER_FIELDNAMES})
    if include_preprocess:
        row.update({k: "" for k in PREPROCESS_FIELDNAMES})
    row.update({k: "" for k in METRIC_FIELDNAMES})
    return row


def _populate_metrics_row(row: dict, metrics: dict) -> None:
    row["species_name"] = metrics.get("species_name") or ""
    row["study_location"] = metrics.get("study_location") or ""
    row["study_date"] = metrics.get("study_date") or ""
    row["sample_size"] = "" if metrics.get("sample_size") is None else metrics["sample_size"]
    row["num_empty_stomachs"] = "" if metrics.get("num_empty_stomachs") is None else metrics["num_empty_stomachs"]
    row["num_nonempty_stomachs"] = "" if metrics.get("num_nonempty_stomachs") is None else metrics["num_nonempty_stomachs"]
    row["fraction_feeding"] = "" if metrics.get("fraction_feeding") is None else metrics["fraction_feeding"]


def _load_useful_stems(labels_path: Path) -> set[str]:
    if not labels_path.exists():
        print(f"[ERROR] Labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    useful = {k for k, v in labels.items() if v == "useful"}
    print(f"[INFO] Labels filter: {len(useful)} useful papers", file=sys.stderr)
    return useful


def _collect_pdf_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        pdf_paths = sorted(input_path.glob("*.pdf"))
        if not pdf_paths:
            print(f"[ERROR] No PDF files found in directory: {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Found {len(pdf_paths)} PDF(s) in {input_path}", file=sys.stderr)
        return pdf_paths
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    print(f"[ERROR] Input must be a .pdf file or a directory of PDFs: {input_path}", file=sys.stderr)
    sys.exit(1)


def _collect_txt_paths(
    input_dir: Path,
    *,
    single_file: Path | None = None,
    useful_stems: set | None = None,
) -> list[Path]:
    if single_file is not None:
        return [single_file]
    txt_paths = sorted(input_dir.glob("*.txt"))
    if useful_stems is not None:
        txt_paths = [p for p in txt_paths if p.stem in useful_stems or p.name in useful_stems]
    if not txt_paths:
        print(f"[ERROR] No .txt files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(txt_paths)} .txt file(s) to process", file=sys.stderr)
    return txt_paths


def _cleaned_text_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    base = output_dir.parent / "cleaned-text"
    cleaner_dir = base / "text_cleaner"
    filter_dir = base / "section_filter"
    llm_dir = base / "llm_text"
    for d in (cleaner_dir, filter_dir, llm_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cleaner_dir, filter_dir, llm_dir


def _preprocess_text(
    raw_text: str,
    max_chars: int,
    *,
    stem: str,
    save_intermediates: bool,
    cleaner_dir: Path | None,
    filter_dir: Path | None,
    llm_dir: Path | None,
) -> tuple[str | None, str | None, dict, str | None]:
    counts: dict = {"raw_chars": len(raw_text)}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not raw_text.strip():
        return None, None, counts, "empty_text"

    cleaned = clean_text(raw_text)
    counts["cleaned_chars"] = len(cleaned)
    print(f"  [INFO] After clean: {len(cleaned):,} chars", file=sys.stderr)

    if not cleaned.strip():
        return None, None, counts, "empty_after_clean"

    if save_intermediates and cleaner_dir is not None:
        _write_intermediate(cleaner_dir / f"{stem}_{ts}.txt", cleaned, "Cleaner text")

    filtered = filter_relevant_sections(cleaned)
    counts["filtered_chars"] = len(filtered)
    print(f"  [INFO] After filter: {len(filtered):,} chars", file=sys.stderr)

    if save_intermediates and filter_dir is not None:
        _write_intermediate(filter_dir / f"{stem}_{ts}.txt", filtered, "Filter text")

    if len(filtered) > max_chars:
        trimmed = extract_key_sections(filtered, max_chars)
        print(
            f"  [INFO] After trim : {len(trimmed):,} chars (budget {max_chars:,})",
            file=sys.stderr,
        )
    else:
        trimmed = filtered

    counts["trimmed_chars"] = len(trimmed)

    if save_intermediates and llm_dir is not None:
        _write_intermediate(llm_dir / f"{stem}_{ts}.txt", trimmed, "LLM text")

    return filtered, trimmed, counts, None


def _write_intermediate(path: Path, text: str, label: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
        print(f"  [INFO] {label:<13}: {path.name}", file=sys.stderr)
    except Exception as exc:
        print(f"  [WARN] Could not save {label.lower()}: {exc}", file=sys.stderr)


def _run_llm_extraction(
    *,
    filtered_text: str,
    trimmed_text: str,
    source_file: Path,
    original_text: str,
    output_dir: Path,
    llm_model: str,
    num_ctx: int,
    chunked: bool,
    model_dir: str,
    top_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[dict | None, str | None]:
    if chunked:
        print(f"  [INFO] Chunked extraction (top {top_chunks} chunks)…", file=sys.stderr)
        try:
            merged = extract_with_chunking(
                text=filtered_text,
                model_dir=model_dir,
                llm_model=llm_model,
                num_ctx=num_ctx,
                top_n=top_chunks,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )
        except Exception as exc:
            print(f"  [ERROR] Chunked extraction failed: {exc}", file=sys.stderr)
            log.error("Chunked extraction failed for %s: %s", source_file.name, exc)
            return None, "extraction_failed"

        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_path = metrics_dir / f"{source_file.stem}_results.json"
        result_obj = {
            "source_file": source_file.name,
            "file_type": source_file.suffix.lower(),
            "extraction_mode": "chunked",
            "metrics": merged,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_obj, f, indent=2)
        print(f"  [SUCCESS] Results saved to {out_path}", file=sys.stderr)
        return merged, None

    print(f"  [INFO] Calling Ollama ({llm_model})…", file=sys.stderr)
    try:
        metrics = extract_metrics_from_text(
            text=trimmed_text,
            model=llm_model,
            num_ctx=num_ctx,
        )
        result = save_extraction_result(
            metrics=metrics,
            source_file=source_file,
            original_text=original_text,
            output_dir=output_dir,
        )
        return result["metrics"], None
    except Exception as exc:
        print(f"  [ERROR] LLM extraction failed: {exc}", file=sys.stderr)
        log.error("LLM extraction failed for %s: %s", source_file.name, exc)
        return None, "extraction_failed"


def _write_summary_csv(
    summary_rows: list[dict],
    output_dir: Path,
    *,
    include_classifier: bool,
    include_preprocess: bool,
    prefix: str = "pipeline",
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / f"{prefix}_summary_{timestamp}.csv"
    fieldnames = _summary_fieldnames(
        include_classifier=include_classifier,
        include_preprocess=include_preprocess,
    )
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def _print_classify_summary(summary_rows: list[dict], summary_path: Path) -> None:
    total = len(summary_rows)
    useful_count = sum(1 for r in summary_rows if r.get("classification") == "useful")
    not_useful_count = sum(1 for r in summary_rows if r.get("classification") == "not useful")
    extracted_count = sum(1 for r in summary_rows if r.get("extraction_status") == "success")
    error_statuses = {
        "text_extraction_failed",
        "empty_text",
        "empty_after_clean",
        "extraction_failed",
        "worker_failed",
    }
    error_count = sum(1 for r in summary_rows if r.get("extraction_status") in error_statuses)

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


def _print_txt_summary(summary_rows: list[dict], summary_path: Path) -> None:
    total = len(summary_rows)
    succeeded = sum(1 for r in summary_rows if r.get("extraction_status") == "success")
    failed = total - succeeded

    print("\n" + "=" * 55, file=sys.stderr)
    print("TXT EXTRACTION PIPELINE COMPLETE", file=sys.stderr)
    print("=" * 55, file=sys.stderr)
    print(f"  Files processed   : {total}", file=sys.stderr)
    print(f"  Successful        : {succeeded}", file=sys.stderr)
    print(f"  Failed / skipped  : {failed}", file=sys.stderr)
    print(f"  Summary CSV       : {summary_path}", file=sys.stderr)
    print("=" * 55, file=sys.stderr)


def _apply_char_counts(row: dict, counts: dict) -> None:
    for key in PREPROCESS_FIELDNAMES:
        if key in row and key in counts:
            row[key] = counts[key]


# ---------------------------------------------------------------------------
# Per-file processing
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
    *,
    chunked: bool,
    model_dir: str,
    top_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
):
    """Classify one PDF and return a summary row dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    row = _new_summary_row(pdf_path.name, include_classifier=True, include_preprocess=False)

    try:
        original_text = extract_text_from_pdf(str(pdf_path))
    except Exception as e:
        print(f"  [ERROR] Text extraction failed ({pdf_path.name}): {e}", file=sys.stderr)
        log.error("Text extraction failed for %s: %s", pdf_path.name, e)
        row["extraction_status"] = "text_extraction_failed"
        return row

    if not original_text.strip():
        print(f"  [WARN] No text extracted from {pdf_path.name}. Skipping.", file=sys.stderr)
        log.warning("No text extracted from %s — skipping.", pdf_path.name)
        row["extraction_status"] = "empty_text"
        return row

    print(f"  [INFO] {pdf_path.name}: {len(original_text)} chars", file=sys.stderr)

    label, confidence, pred_prob = classify_text(
        text=original_text,
        model=clf_model,
        vectorizer=vectorizer,
        encoder=encoder,
        threshold=confidence_threshold,
    )
    print(f"  [CLASSIFIER] {pdf_path.name} → {label} ({confidence:.2%})", file=sys.stderr)

    row["classification"] = label
    row["confidence"] = f"{confidence:.4f}"
    row["pred_prob"] = f"{pred_prob:.4f}"

    if label != "useful":
        print(f"  [INFO] {pdf_path.name}: Not useful — skipping LLM extraction.", file=sys.stderr)
        row["extraction_status"] = "skipped_not_useful"
        return row

    print(f"  [INFO] {pdf_path.name}: Running LLM extraction...", file=sys.stderr)
    row = _run_extraction_on_text(
        row,
        raw_text=original_text,
        source_file=pdf_path,
        original_text=original_text,
        output_dir=output_dir,
        llm_model=llm_model,
        max_chars=max_chars,
        num_ctx=num_ctx,
        save_intermediates=False,
        cleaner_dir=None,
        filter_dir=None,
        llm_dir=None,
        chunked=chunked,
        model_dir=model_dir,
        top_chunks=top_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return row


def _process_single_txt(
    txt_path: Path,
    llm_model: str,
    output_dir: Path,
    max_chars: int,
    num_ctx: int,
    cleaner_dir: Path,
    filter_dir: Path,
    llm_dir: Path,
    *,
    chunked: bool,
    model_dir: str,
    top_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
):
    """Process one .txt file (skip-classifier mode) and return a summary row dict."""
    row = _new_summary_row(txt_path.name, include_classifier=False, include_preprocess=True)

    try:
        raw_text = txt_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"  [ERROR] Could not read file: {exc}", file=sys.stderr)
        row["extraction_status"] = "read_failed"
        return row

    row["raw_chars"] = len(raw_text)
    print(f"  [INFO] Raw size   : {len(raw_text):,} chars", file=sys.stderr)

    if not raw_text.strip():
        print(f"  [WARN] File is empty — skipping.", file=sys.stderr)
        row["extraction_status"] = "empty_file"
        return row

    return _run_extraction_on_text(
        row,
        raw_text=raw_text,
        source_file=txt_path,
        original_text=raw_text,
        output_dir=output_dir,
        llm_model=llm_model,
        max_chars=max_chars,
        num_ctx=num_ctx,
        save_intermediates=True,
        cleaner_dir=cleaner_dir,
        filter_dir=filter_dir,
        llm_dir=llm_dir,
        chunked=chunked,
        model_dir=model_dir,
        top_chunks=top_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _run_extraction_on_text(
    row: dict,
    *,
    raw_text: str,
    source_file: Path,
    original_text: str,
    output_dir: Path,
    llm_model: str,
    max_chars: int,
    num_ctx: int,
    save_intermediates: bool,
    cleaner_dir: Path | None,
    filter_dir: Path | None,
    llm_dir: Path | None,
    chunked: bool,
    model_dir: str,
    top_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """Clean, filter, optionally trim, then run LLM extraction."""
    filtered, trimmed, counts, err = _preprocess_text(
        raw_text,
        max_chars,
        stem=source_file.stem,
        save_intermediates=save_intermediates,
        cleaner_dir=cleaner_dir,
        filter_dir=filter_dir,
        llm_dir=llm_dir if not chunked else None,
    )
    _apply_char_counts(row, counts)

    if err == "empty_text":
        row["extraction_status"] = "empty_text" if "classification" in row else "empty_file"
        return row
    if err == "empty_after_clean":
        row["extraction_status"] = "empty_after_clean"
        return row

    if chunked:
        row["trimmed_chars"] = ""

    metrics, extract_err = _run_llm_extraction(
        filtered_text=filtered,
        trimmed_text=trimmed if trimmed is not None else filtered,
        source_file=source_file,
        original_text=original_text,
        output_dir=output_dir,
        llm_model=llm_model,
        num_ctx=num_ctx,
        chunked=chunked,
        model_dir=model_dir,
        top_chunks=top_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if extract_err:
        row["extraction_status"] = extract_err
        return row

    row["extraction_status"] = "success"
    _populate_metrics_row(row, metrics)
    print(
        f"  [OK] species={metrics.get('species_name')}  " f"n={metrics.get('sample_size')}  " f"date={metrics.get('study_date')}",
        file=sys.stderr,
    )
    return row


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------


def run_pdf_pipeline(
    input_path: Path,
    model_dir: str,
    llm_model: str,
    output_dir: Path,
    confidence_threshold: float,
    max_chars: int,
    num_ctx: int,
    workers: int,
    useful_stems: set | None,
    *,
    chunked: bool,
    top_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
):
    """Classify PDFs and extract metrics from those labelled useful."""
    pdf_paths = _collect_pdf_paths(input_path)
    if useful_stems is not None:
        pdf_paths = [p for p in pdf_paths if p.stem in useful_stems or p.name in useful_stems]
        if not pdf_paths:
            print("[ERROR] No PDF files match the labels filter.", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] After labels filter: {len(pdf_paths)} PDF(s)", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading classifier...", file=sys.stderr)
    try:
        clf_model, vectorizer, encoder = load_classifier(model_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        log.critical("Classifier artifacts not found: %s", e)
        sys.exit(1)
    print("[INFO] Classifier loaded.", file=sys.stderr)

    summary_rows = []
    extract_kw = dict(
        chunked=chunked,
        model_dir=model_dir,
        top_chunks=top_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

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
                    **extract_kw,
                ): pdf_path
                for pdf_path in pdf_paths
            }
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    print(f"  [ERROR] Worker failed for {pdf_path.name}: {exc}", file=sys.stderr)
                    row = _new_summary_row(pdf_path.name, include_classifier=True, include_preprocess=False)
                    row["extraction_status"] = "worker_failed"
                summary_rows.append(row)
    else:
        for idx, pdf_path in enumerate(pdf_paths, start=1):
            print(f"\n[{idx}/{len(pdf_paths)}] Processing: {pdf_path.name}", file=sys.stderr)
            row = _process_single_pdf(
                pdf_path,
                llm_model,
                output_dir,
                confidence_threshold,
                max_chars,
                num_ctx,
                clf_model,
                vectorizer,
                encoder,
                **extract_kw,
            )
            summary_rows.append(row)

    summary_path = _write_summary_csv(
        summary_rows,
        output_dir,
        include_classifier=True,
        include_preprocess=False,
        prefix="pipeline",
    )
    _print_classify_summary(summary_rows, summary_path)


def run_txt_pipeline(
    input_path: Path,
    output_dir: Path,
    llm_model: str,
    max_chars: int,
    num_ctx: int,
    useful_stems: set | None,
    *,
    chunked: bool,
    model_dir: str,
    top_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
):
    """Process .txt files without classification (all inputs treated as useful)."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".txt":
            print(f"[ERROR] With --skip-classifier, input must be a .txt file: {input_path}", file=sys.stderr)
            sys.exit(1)
        single_file = input_path
        input_dir = input_path.parent
    elif input_path.is_dir():
        single_file = None
        input_dir = input_path
    else:
        print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    txt_paths = _collect_txt_paths(input_dir, single_file=single_file, useful_stems=useful_stems)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaner_dir, filter_dir, llm_dir = _cleaned_text_dirs(output_dir)
    summary_rows = []

    for idx, txt_path in enumerate(txt_paths, start=1):
        print(f"\n[{idx}/{len(txt_paths)}] {txt_path.name}", file=sys.stderr)
        row = _process_single_txt(
            txt_path,
            llm_model,
            output_dir,
            max_chars,
            num_ctx,
            cleaner_dir,
            filter_dir,
            llm_dir,
            chunked=chunked,
            model_dir=model_dir,
            top_chunks=top_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        summary_rows.append(row)

    summary_path = _write_summary_csv(
        summary_rows,
        output_dir,
        include_classifier=False,
        include_preprocess=True,
        prefix="txt_pipeline",
    )
    _print_txt_summary(summary_rows, summary_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=("Classify PDFs as useful/not-useful and extract structured diet metrics, " "or extract from pre-classified .txt files with --skip-classifier."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  PDF folder:
    python src/pipeline/classify_extract.py data/pdfs/

  Pre-classified .txt folder:
    python src/pipeline/classify_extract.py data/processed-text/ --skip-classifier

  Chunked .txt extraction:
    python src/pipeline/classify_extract.py data/processed-text/ --skip-classifier --chunked

  Labels filter:
    python src/pipeline/classify_extract.py data/processed-text/ --skip-classifier --labels labels.json
        """,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a PDF or .txt file, or a directory of PDFs or .txt files.",
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Skip XGBoost classification; process .txt files as useful (extract-from-txt mode).",
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
        default="qwen2.5:7b",
        help="Ollama model to use for extraction (default: qwen2.5:7b).",
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
        help="Classifier probability threshold for 'useful' (default: 0.70). Ignored with --skip-classifier.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Max characters of text to send to the LLM after cleaning (default: 12000).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Context window size for Ollama (default: 4096).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for PDF mode only (default: 1 = sequential).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to labels.json; only files labelled 'useful' are processed.",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        default=False,
        help=("Chunked extraction: split text, score chunks with XGBoost, " "extract from top-N chunks, merge via majority voting."),
    )
    parser.add_argument(
        "--top-chunks",
        type=int,
        default=3,
        help="Number of top-scoring chunks to extract from (default: 3). Only used with --chunked.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4000,
        help="Character size per chunk (default: 4000). Only used with --chunked.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=500,
        help="Overlap between consecutive chunks (default: 500). Only used with --chunked.",
    )

    args = parser.parse_args()
    setup_logging()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
        log.error("Input path not found: %s", input_path)
        sys.exit(1)

    useful_stems = _load_useful_stems(Path(args.labels)) if args.labels else None

    extract_kw = dict(
        chunked=args.chunked,
        model_dir=args.model_dir,
        top_chunks=args.top_chunks,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.skip_classifier:
        run_txt_pipeline(
            input_path=input_path,
            output_dir=Path(args.output_dir),
            llm_model=args.llm_model,
            max_chars=args.max_chars,
            num_ctx=args.num_ctx,
            useful_stems=useful_stems,
            **extract_kw,
        )
    else:
        if input_path.suffix.lower() == ".txt":
            print(
                "[ERROR] .txt input requires --skip-classifier. " "Use: python src/pipeline/classify_extract.py <path> --skip-classifier",
                file=sys.stderr,
            )
            sys.exit(1)
        if input_path.is_dir() and not list(input_path.glob("*.pdf")) and list(input_path.glob("*.txt")):
            print(
                "[ERROR] Directory contains .txt files only; use --skip-classifier for .txt extraction.",
                file=sys.stderr,
            )
            sys.exit(1)
        run_pdf_pipeline(
            input_path=input_path,
            model_dir=args.model_dir,
            llm_model=args.llm_model,
            output_dir=Path(args.output_dir),
            confidence_threshold=args.confidence_threshold,
            max_chars=args.max_chars,
            num_ctx=args.num_ctx,
            workers=args.workers,
            useful_stems=useful_stems,
            **extract_kw,
        )


if __name__ == "__main__":
    main()
