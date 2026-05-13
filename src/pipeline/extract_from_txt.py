"""Extract-from-TXT Pipeline

Processes pre-classified useful .txt files through noise cleaning, section
filtering, text trimming, and LLM extraction — bypassing the XGBoost
classifier entirely.

Every .txt file fed to this script is assumed to have already been confirmed
as useful (e.g. by the classifier in src/pipeline/classify_extract.py or by manual review).
The pipeline:

  1. Read raw .txt file
  2. Strip noise (references, acknowledgements, affiliations, captions, …)
     via src/io/text_cleaner.py
  3. Drop irrelevant paragraphs (taxonomy, morphometrics, stats methods, …)
     via src/io/section_filter.py
  4. Trim to the character budget using section-priority ranking
     via src/extraction/llm_text.py::extract_key_sections()
  5. Call Ollama for structured extraction via src/extraction/llm_client.py
  6. Save result JSON per file and a summary CSV

Usage::

    # Process the default directory (data/processed-text/)
    python src/pipeline/extract_from_txt.py

    # Custom input directory
    python src/pipeline/extract_from_txt.py --input-dir path/to/txt_files/

    # Full options
    python src/pipeline/extract_from_txt.py \\
        --input-dir  data/processed-text/ \\
        --output-dir data/results/ \\
        --llm-model  qwen2.5:7b \\
        --max-chars  10000 \\
        --num-ctx    8192

Output:
    - data/cleaned-text/text_cleaner/<stem>_<YYYYMMDD_HHMMSS>.txt  noise-stripped text
    - data/cleaned-text/section_filter/<stem>_<YYYYMMDD_HHMMSS>.txt  section-filtered text
    - data/cleaned-text/llm_text/<stem>_<YYYYMMDD_HHMMSS>.txt  trimmed text passed to Ollama
    - data/results/metrics/<stem>_results.json  per file
    - data/results/summaries/txt_pipeline_summary_<timestamp>.csv  overall
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from src.io.text_cleaner import clean_text
from src.io.section_filter import filter_relevant_sections
from src.extraction.llm_text import extract_key_sections
from src.extraction.llm_client import extract_metrics_from_text, save_extraction_result
from src.extraction.chunked_extraction import extract_with_chunking


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------


def run_txt_pipeline(
    input_dir: Path,
    output_dir: Path,
    llm_model: str,
    max_chars: int,
    num_ctx: int,
    single_file: Path = None,
    useful_stems: set = None,
    chunked: bool = False,
    top_chunks: int = 3,
    chunk_size: int = 4000,
    chunk_overlap: int = 500,
    model_dir: str = "src/classifier/models",
) -> None:
    """Process every .txt file in *input_dir* through clean → filter → trim → extract.

    Args:
        input_dir:   Directory containing pre-classified useful .txt files.
                     Ignored when *single_file* is provided.
        output_dir:  Root output directory for JSON results and summary CSV.
        llm_model:   Ollama model name (e.g. ``"qwen2.5:7b"``).
        max_chars:   Character budget for the text sent to Ollama.
        num_ctx:     Context window size requested from Ollama.
        single_file: If set, process only this one .txt file.
    """
    if single_file is not None:
        txt_paths = [single_file]
    else:
        txt_paths = sorted(input_dir.glob("*.txt"))
        if useful_stems is not None:
            txt_paths = [p for p in txt_paths if p.stem in useful_stems or p.name in useful_stems]
        if not txt_paths:
            print(f"[ERROR] No .txt files found in: {input_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Found {len(txt_paths)} .txt file(s) to process", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaner_text_dir = output_dir.parent / "cleaned-text" / "text_cleaner"
    filter_text_dir = output_dir.parent / "cleaned-text" / "section_filter"
    llm_text_dir = output_dir.parent / "cleaned-text" / "llm_text"
    cleaner_text_dir.mkdir(parents=True, exist_ok=True)
    filter_text_dir.mkdir(parents=True, exist_ok=True)
    llm_text_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for idx, txt_path in enumerate(txt_paths, start=1):
        print(f"\n[{idx}/{len(txt_paths)}] {txt_path.name}", file=sys.stderr)

        row: dict = {
            "filename": txt_path.name,
            "raw_chars": "",
            "cleaned_chars": "",
            "filtered_chars": "",
            "trimmed_chars": "",
            "extraction_status": "",
            "species_name": "",
            "study_location": "",
            "study_date": "",
            "sample_size": "",
            "num_empty_stomachs": "",
            "num_nonempty_stomachs": "",
            "fraction_feeding": "",
        }

        # ── Step 1: Read ────────────────────────────────────────────────────
        try:
            raw_text = txt_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"  [ERROR] Could not read file: {exc}", file=sys.stderr)
            row["extraction_status"] = "read_failed"
            summary_rows.append(row)
            continue

        row["raw_chars"] = len(raw_text)
        print(f"  [INFO] Raw size   : {len(raw_text):,} chars", file=sys.stderr)

        if not raw_text.strip():
            print(f"  [WARN] File is empty — skipping.", file=sys.stderr)
            row["extraction_status"] = "empty_file"
            summary_rows.append(row)
            continue

        # ── Step 2: Clean ───────────────────────────────────────────────────
        cleaned = clean_text(raw_text)
        row["cleaned_chars"] = len(cleaned)
        print(f"  [INFO] After clean: {len(cleaned):,} chars", file=sys.stderr)

        if not cleaned.strip():
            print(f"  [WARN] Nothing left after cleaning — skipping.", file=sys.stderr)
            row["extraction_status"] = "empty_after_clean"
            summary_rows.append(row)
            continue

        # ── Step 3: Save text_cleaner output ────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaner_path = cleaner_text_dir / f"{txt_path.stem}_{ts}.txt"
        try:
            cleaner_path.write_text(cleaned, encoding="utf-8")
            print(f"  [INFO] Cleaner text : {cleaner_path.name}", file=sys.stderr)
        except Exception as exc:
            print(f"  [WARN] Could not save cleaner text: {exc}", file=sys.stderr)

        # ── Step 4: Section filter ──────────────────────────────────────────
        filtered = filter_relevant_sections(cleaned)
        row["filtered_chars"] = len(filtered)
        print(f"  [INFO] After filter: {len(filtered):,} chars", file=sys.stderr)

        # ── Step 4b: Save section_filter output ─────────────────────────────
        filter_path = filter_text_dir / f"{txt_path.stem}_{ts}.txt"
        try:
            filter_path.write_text(filtered, encoding="utf-8")
            print(f"  [INFO] Filter text  : {filter_path.name}", file=sys.stderr)
        except Exception as exc:
            print(f"  [WARN] Could not save filter text: {exc}", file=sys.stderr)

        # ── Step 5+6+7: Extract (chunked or single-pass) ────────────────────
        if chunked:
            print(f"  [INFO] Chunked extraction (top {top_chunks} chunks)…", file=sys.stderr)
            row["trimmed_chars"] = ""
            try:
                merged = extract_with_chunking(
                    text=filtered,
                    model_dir=model_dir,
                    llm_model=llm_model,
                    num_ctx=num_ctx,
                    top_n=top_chunks,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                )
            except Exception as exc:
                print(f"  [ERROR] Chunked extraction failed: {exc}", file=sys.stderr)
                row["extraction_status"] = "extraction_failed"
                summary_rows.append(row)
                continue

            # save JSON
            import json as _json

            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            out_path = metrics_dir / (txt_path.stem + "_results.json")
            result_obj = {
                "source_file": txt_path.name,
                "file_type": txt_path.suffix.lower(),
                "extraction_mode": "chunked",
                "metrics": merged,
            }
            with open(out_path, "w", encoding="utf-8") as _f:
                _json.dump(result_obj, _f, indent=2)
            print(f"[SUCCESS] Results saved to {out_path}", file=sys.stderr)

            m = merged
        else:
            # ── Step 5: Trim to LLM budget ──────────────────────────────────
            if len(filtered) > max_chars:
                trimmed = extract_key_sections(filtered, max_chars)
                print(
                    f"  [INFO] After trim : {len(trimmed):,} chars " f"(budget {max_chars:,})",
                    file=sys.stderr,
                )
            else:
                trimmed = filtered

            row["trimmed_chars"] = len(trimmed)

            # ── Step 5b: Save llm_text output ───────────────────────────────
            llm_path = llm_text_dir / f"{txt_path.stem}_{ts}.txt"
            try:
                llm_path.write_text(trimmed, encoding="utf-8")
                print(f"  [INFO] LLM text     : {llm_path.name}", file=sys.stderr)
            except Exception as exc:
                print(f"  [WARN] Could not save LLM text: {exc}", file=sys.stderr)

            # ── Step 6: LLM extraction ──────────────────────────────────────
            print(f"  [INFO] Calling Ollama ({llm_model})…", file=sys.stderr)
            try:
                metrics = extract_metrics_from_text(
                    text=trimmed,
                    model=llm_model,
                    num_ctx=num_ctx,
                )
            except Exception as exc:
                print(f"  [ERROR] Ollama extraction failed: {exc}", file=sys.stderr)
                row["extraction_status"] = "extraction_failed"
                summary_rows.append(row)
                continue

            # ── Step 7: Save JSON ───────────────────────────────────────────
            try:
                result = save_extraction_result(
                    metrics=metrics,
                    source_file=txt_path,
                    original_text=raw_text,
                    output_dir=output_dir,
                )
            except Exception as exc:
                print(f"  [ERROR] Could not save result: {exc}", file=sys.stderr)
                row["extraction_status"] = "save_failed"
                summary_rows.append(row)
                continue

            m = result["metrics"]

        row["extraction_status"] = "success"
        row["species_name"] = m.get("species_name") or ""
        row["study_location"] = m.get("study_location") or ""
        row["study_date"] = m.get("study_date") or ""
        row["sample_size"] = "" if m.get("sample_size") is None else m["sample_size"]
        row["num_empty_stomachs"] = "" if m.get("num_empty_stomachs") is None else m["num_empty_stomachs"]
        row["num_nonempty_stomachs"] = "" if m.get("num_nonempty_stomachs") is None else m["num_nonempty_stomachs"]
        row["fraction_feeding"] = "" if m.get("fraction_feeding") is None else m["fraction_feeding"]

        print(
            f"  [OK] species={m.get('species_name')}  " f"n={m.get('sample_size')}  " f"date={m.get('study_date')}",
            file=sys.stderr,
        )

        summary_rows.append(row)

    # ── Write summary CSV ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / f"txt_pipeline_summary_{timestamp}.csv"

    fieldnames = [
        "filename",
        "raw_chars",
        "cleaned_chars",
        "filtered_chars",
        "trimmed_chars",
        "extraction_status",
        "species_name",
        "study_location",
        "study_date",
        "sample_size",
        "num_empty_stomachs",
        "num_nonempty_stomachs",
        "fraction_feeding",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── Final report ────────────────────────────────────────────────────────
    total = len(summary_rows)
    succeeded = sum(1 for r in summary_rows if r["extraction_status"] == "success")
    failed = total - succeeded

    print("\n" + "=" * 55, file=sys.stderr)
    print("TXT EXTRACTION PIPELINE COMPLETE", file=sys.stderr)
    print("=" * 55, file=sys.stderr)
    print(f"  Files processed   : {total}", file=sys.stderr)
    print(f"  Successful        : {succeeded}", file=sys.stderr)
    print(f"  Failed / skipped  : {failed}", file=sys.stderr)
    print(f"  Summary CSV       : {summary_path}", file=sys.stderr)
    print("=" * 55, file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Extract structured predator-diet metrics from pre-classified " "useful .txt files using Ollama."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Default (data/processed-text/ → data/results/):
    python src/pipeline/extract_from_txt.py

  Custom directories:
    python src/pipeline/extract_from_txt.py --input-dir data/useful-txt/ --output-dir out/

  Different model / tighter budget:
    python src/pipeline/extract_from_txt.py --llm-model mistral:7b --max-chars 4500
        """,
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single .txt file to process. Overrides --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed-text",
        help="Directory of .txt files to process (default: data/processed-text).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Root output directory for JSON results and CSV summary " "(default: data/results).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        # default="qwen2.5:7b",
        default="qwen2.5:7b",
        help="Ollama model name (default: qwen2.5:7b).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=10000,
        help="Maximum characters to send to Ollama after cleaning (default: 10000).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Ollama context window size (default: 8192).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to labels.json. When provided, only files labelled 'useful' are processed.",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        default=False,
        help="Use chunked extraction: split text, score chunks with XGBoost, " "extract from top-N chunks, merge via majority voting.",
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
    parser.add_argument(
        "--model-dir",
        type=str,
        default="src/classifier/models",
        help="Directory containing XGBoost model artifacts (default: src/classifier/models). Only used with --chunked.",
    )

    args = parser.parse_args()

    # ── Load label filter ───────────────────────────────────────────────
    useful_stems = None
    if args.labels:
        import json

        labels_path = Path(args.labels)
        if not labels_path.exists():
            print(f"[ERROR] Labels file not found: {labels_path}", file=sys.stderr)
            sys.exit(1)
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)
        useful_stems = {k for k, v in labels.items() if v == "useful"}
        print(f"[INFO] Labels filter: {len(useful_stems)} useful papers", file=sys.stderr)

    single_file = None
    if args.file:
        single_file = Path(args.file)
        if not single_file.exists():
            print(f"[ERROR] File not found: {single_file}", file=sys.stderr)
            sys.exit(1)
        if single_file.suffix.lower() != ".txt":
            print(f"[ERROR] --file must point to a .txt file: {single_file}", file=sys.stderr)
            sys.exit(1)
        input_dir = single_file.parent
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)
        if not input_dir.is_dir():
            print(f"[ERROR] --input-dir must be a directory: {input_dir}", file=sys.stderr)
            sys.exit(1)

    run_txt_pipeline(
        input_dir=input_dir,
        output_dir=Path(args.output_dir),
        llm_model=args.llm_model,
        max_chars=args.max_chars,
        num_ctx=args.num_ctx,
        single_file=single_file,
        useful_stems=useful_stems,
        chunked=args.chunked,
        top_chunks=args.top_chunks,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
