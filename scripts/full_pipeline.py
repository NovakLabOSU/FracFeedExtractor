"""Full training pipeline: stream ALL PDFs from Drive, extract text, label, train model.

Modes:
 - API mode: Download PDFs from Google Drive and process them
 - Local mode: Use PDFs already downloaded locally

API Mode Environment variables:
 - GOOGLE_SERVICE_ACCOUNT_JSON (service account JSON string)
 - GOOGLE_DRIVE_ROOT_FOLDER_ID (root folder containing 'useful' and 'not-useful')
 - GOOGLE_DRIVE_USE_SHARED_DRIVE=true (if using shared drives / shared folders)

Usage:
 - API mode: python full_pipeline.py --api
 - Local mode: python full_pipeline.py --local <path_to_pdfs>

Behavior:
 - API mode: Streams every PDF (no local PDF persistence) and writes extracted text to data/processed-text.
 - Local mode: Processes PDFs from specified local directory (expects 'useful' and 'not-useful' subfolders).
 - Generates labels.json based on folder origin.
 - Trains model with src/model/train_model.py.
"""

from __future__ import annotations

import os
import json
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Dict, Tuple
import subprocess
import sys

import sys
from pathlib import Path as _Path2

sys.path.append(str(_Path2(__file__).resolve().parents[1]))  # add repo root to sys.path

from scripts.env_loader import load_env

load_env()  # Load .env file if present (for local dev)

from scripts.google_drive.drive_io import (
    get_drive_service,
    find_child_folder_id,
    list_pdfs_in_folder,
    download_file_bytes,
    sanitize_filename,
)
from src.preprocessing.pdf_text_extraction import extract_text_from_pdf_bytes


# Module-level flag set once per worker process via initializer
_worker_skip_ocr: bool = False


def _init_worker(skip_ocr: bool) -> None:
    """Called once per worker process to set shared config."""
    global _worker_skip_ocr
    _worker_skip_ocr = skip_ocr


def _extract_local_pdf(args: Tuple[Path, str]) -> Tuple[str, str, str | None]:
    """Worker: read a local PDF and return (txt_name, label, text | None)."""
    pdf_path, label = args
    print(f"  [STARTED] {pdf_path.name}", flush=True)
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        text = extract_text_from_pdf_bytes(pdf_bytes, skip_ocr=_worker_skip_ocr)
        return (f"{pdf_path.stem}.txt", label, text)
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return (f"{pdf_path.stem}.txt", label, None)


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def write_labels(labels: Dict[str, str], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def process_api_mode():
    """Download PDFs from Google Drive and process them."""
    root_id = os.environ.get("GOOGLE_DRIVE_ROOT_FOLDER_ID")
    if not root_id:
        raise RuntimeError("Missing GOOGLE_DRIVE_ROOT_FOLDER_ID environment variable")

    service = get_drive_service()
    useful_id = find_child_folder_id(service, root_id, "useful")
    not_useful_id = find_child_folder_id(service, root_id, "not-useful")
    if not useful_id:
        raise RuntimeError(f"Could not find 'useful' subfolder under root folder {root_id}")
    if not not_useful_id:
        raise RuntimeError(f"Could not find 'not-useful' subfolder under root folder {root_id}")

    out_dir = Path("data/processed-text")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels: Dict[str, str] = {}
    count = 1
    for folder_id, label in [(useful_id, "useful"), (not_useful_id, "not-useful")]:
        files = list_pdfs_in_folder(service, folder_id, max_files=None)
        print(f"Found {len(files)} PDFs in folder label '{label}'")
        for f in files:
            pdf_bytes = download_file_bytes(service, f["id"])
            text = extract_text_from_pdf_bytes(pdf_bytes)
            stem = sanitize_filename(f.get("name", f.get("id", "file")))
            txt_name = f"{stem}.txt"
            (out_dir / txt_name).write_text(text, encoding="utf-8")
            labels[txt_name] = label
            print(f"{count} Processed {f['name']}")
            count += 1

    write_labels(labels, Path("data/labels.json"))
    print(f"Wrote {len(labels)} labeled text files.")


def process_local_mode(data_path: Path, workers: int = 1, skip_ocr: bool = False, timeout: int = 120):
    """Process PDFs from local directory."""
    if not data_path.exists():
        raise RuntimeError(f"Data path does not exist: {data_path}")

    useful_dir = data_path / "useful"
    not_useful_dir = data_path / "not-useful"

    if not useful_dir.exists():
        raise RuntimeError(f"'useful' subfolder not found in {data_path}")
    if not not_useful_dir.exists():
        raise RuntimeError(f"'not-useful' subfolder not found in {data_path}")

    out_dir = Path("data/processed-text")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: load existing labels and skip already-processed files
    labels_file = Path("data/labels.json")
    if labels_file.exists():
        with labels_file.open("r", encoding="utf-8") as f:
            labels: Dict[str, str] = json.load(f)
        already_done = {name for name in labels if (out_dir / name).exists()}
        print(f"[INFO] Resuming — {len(already_done)} files already processed, skipping them.")
    else:
        labels = {}
        already_done = set()

    # Build work items: (pdf_path, label)
    work_items = []
    for folder, label in [(useful_dir, "useful"), (not_useful_dir, "not-useful")]:
        pdf_files = list(folder.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDFs in local folder '{label}'")
        for pdf_path in pdf_files:
            txt_name = f"{pdf_path.stem}.txt"
            if txt_name in already_done:
                continue
            work_items.append((pdf_path, label))

    if not work_items:
        print("[INFO] All files already processed. Nothing to do.")
        write_labels(labels, labels_file)
        return

    print(f"[INFO] {len(work_items)} PDFs to process.")
    if skip_ocr:
        print("[INFO] OCR disabled — using embedded text only (fast mode).")

    total = len(work_items)
    done = 0
    failed = 0
    t0 = time.time()

    if workers > 1 and len(work_items) > 1:
        print(f"[INFO] Using {workers} worker processes for PDF extraction.")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(skip_ocr,),
        ) as executor:
            futures = {executor.submit(_extract_local_pdf, item): item for item in work_items}
            for future in as_completed(futures):
                pdf_path, label = futures[future]
                try:
                    txt_name, label, text = future.result(timeout=timeout)
                except TimeoutError:
                    print(f"  [TIMEOUT] {pdf_path.name} exceeded {timeout}s — skipped")
                    failed += 1
                    continue
                except Exception as exc:
                    print(f"  [ERROR] {pdf_path.name}: {exc}")
                    failed += 1
                    continue
                done += 1
                if text is not None:
                    (out_dir / txt_name).write_text(text, encoding="utf-8")
                    labels[txt_name] = label
                elapsed = time.time() - t0
                print(f"  [{done + failed}/{total}] Processed {txt_name}  ({elapsed:.0f}s elapsed)")
                # Checkpoint labels every 50 files
                if done % 50 == 0:
                    write_labels(labels, labels_file)
    else:
        for item in work_items:
            txt_name, label, text = _extract_local_pdf(item)
            done += 1
            if text is not None:
                (out_dir / txt_name).write_text(text, encoding="utf-8")
                labels[txt_name] = label
            elapsed = time.time() - t0
            print(f"  [{done}/{total}] Processed {txt_name}  ({elapsed:.0f}s elapsed)")

    write_labels(labels, labels_file)
    print(f"Wrote {len(labels)} labeled text files. ({failed} timed out / failed)")


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: extract text from PDFs and train model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  API mode:   python full_pipeline.py --api
  Local mode: python full_pipeline.py --local ./data/pdfs
        """,
    )

    # Create mutually exclusive group for --api and --local
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--api", action="store_true", help="Use API mode to download PDFs from Google Drive")
    group.add_argument("--local", type=Path, metavar="PATH", help="Use local mode with PDFs from specified directory (should contain 'useful' and 'not-useful' subfolders)")

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes for PDF extraction (default: 0 = auto-detect CPU count).",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip Tesseract OCR fallback — use only embedded text (much faster, recommended for first pass).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-PDF timeout in seconds (default: 120). PDFs exceeding this are skipped.",
    )

    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else os.cpu_count() or 4
    if args.local:
        print(f"Running in LOCAL mode with data path: {args.local}")
        process_local_mode(args.local, workers=workers, skip_ocr=args.skip_ocr, timeout=args.timeout)
    else:  # args.api
        print("Running in API mode (Google Drive)")
        process_api_mode()

    print("Beginning model training...")
    run([sys.executable, "src/model/train_model.py"])
    print("Training complete.")


if __name__ == "__main__":
    main()
