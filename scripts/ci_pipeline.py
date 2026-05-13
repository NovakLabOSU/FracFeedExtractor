"""CI pipeline: stream a small, fixed sample from Google Drive and run extraction.

Requirements:
 - Env GOOGLE_SERVICE_ACCOUNT_JSON set to Service Account JSON string.
 - Env GOOGLE_DRIVE_ROOT_FOLDER_ID set to the Drive folder containing 'useful' and 'not-useful'.
 - Optional env CI_FILES_PER_CLASS (default 1).

This script DOES NOT save PDFs locally. It streams bytes and writes extracted text
into data/processed-text/*.txt, and writes data/labels.json. No training.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List
import subprocess

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))  # add repo root to sys.path

from scripts.env_loader import load_env

load_env()  # Load .env file if present (for local dev)

from scripts.google_drive.drive_io import (
    get_drive_service,
    find_child_folder_id,
    list_pdfs_in_folder,
    download_file_bytes,
    sanitize_filename,
)
from src.io.pdf_text_extraction import extract_text_from_pdf_bytes


def write_labels(labels: Dict[str, str], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def main():
    root_id = os.environ.get("GOOGLE_DRIVE_ROOT_FOLDER_ID")
    if not root_id:
        raise RuntimeError("Missing GOOGLE_DRIVE_ROOT_FOLDER_ID environment variable")

    per_class = 5  # Fixed set size for CI

    print("Connecting to Google Drive...")
    service = get_drive_service()
    print("Connected to Google Drive")

    print(f"Looking for 'useful' and 'not-useful' folders under root {root_id}...")
    useful_id = find_child_folder_id(service, root_id, "useful")
    not_useful_id = find_child_folder_id(service, root_id, "not-useful")
    if not useful_id:
        raise RuntimeError(f"Could not find 'useful' subfolder under root folder {root_id}")
    if not not_useful_id:
        raise RuntimeError(f"Could not find 'not-useful' subfolder under root folder {root_id}")
    print(f"Found both folders (useful: {useful_id}, not-useful: {not_useful_id})")

    # Prepare output directory
    out_dir = Path("data/processed-text")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {out_dir}")

    labels: Dict[str, str] = {}

    for folder_id, label in [(useful_id, "useful"), (not_useful_id, "not useful")]:
        print(f"\nProcessing '{label}' folder (max {per_class} PDFs)...")
        files = list_pdfs_in_folder(service, folder_id, max_files=per_class)
        print(f"Found {len(files)} PDFs in '{label}' folder")
        for idx, f in enumerate(files, 1):
            pdf_name = f.get("name", f.get("id", "file"))
            print(f"[{idx}/{len(files)}] Processing: {pdf_name}")
            pdf_bytes = download_file_bytes(service, f["id"])
            print(f"Downloaded {len(pdf_bytes)} bytes")
            text = extract_text_from_pdf_bytes(pdf_bytes)
            stem = sanitize_filename(pdf_name)
            txt_name = f"{stem}.txt"
            (out_dir / txt_name).write_text(text, encoding="utf-8")
            labels[txt_name] = label
            print(f"Extracted {len(text)} chars to {txt_name}")

    write_labels(labels, Path("data/labels.json"))
    print(f"\nWrote {len(labels)} labels to data/labels.json")
    print(f"Extracted {len(labels)} text files to {out_dir}")

    # Train model on the CI sample
    print("\nStarting model training on CI sample...")
    r = subprocess.run([sys.executable, "src/classifier/train_model.py"], env={**os.environ, "CI_TRAIN": "1"})
    if r.returncode != 0:
        print("Model training failed")
        raise SystemExit(r.returncode)
    print("Model training complete")


if __name__ == "__main__":
    main()
