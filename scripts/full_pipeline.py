"""Full training pipeline: stream ALL PDFs from Drive, extract text, label, train model.

Modes:
 - API mode: Stream PDFs from Google Drive and process them (no local PDF persistence)
 - Local mode: Use PDFs already downloaded locally (DEFAULT)

API Mode Environment variables:
 - GOOGLE_SERVICE_ACCOUNT_JSON (service account JSON string)
 - GOOGLE_DRIVE_ROOT_FOLDER_ID (root folder containing 'useful' and 'not-useful')
 - GOOGLE_DRIVE_USE_SHARED_DRIVE=true (if using shared drives / shared folders)

Usage:
 - Default (local): python full_pipeline.py
 - API mode: python full_pipeline.py --api
 - Custom path: python full_pipeline.py --local C:\\path\\to\\data

Behavior:
 - API mode: Streams every PDF (no local PDF persistence) and writes extracted text to data/processed-text.
 - Local mode: Processes PDFs from data/useful and data/not-useful folders.
 - Generates labels.json based on folder origin.
 - Trains model with src/model/train_model.py.
 - Automatically generates classification results (CSV & JSON).
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict
import subprocess
import sys

# Project setup - MUST be before other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment (for API mode)
try:
    from scripts.env_loader import load_env

    load_env()
except Exception as e:
    print(f"[WARNING] Could not load environment: {e}")

# Import Google Drive modules (only needed for API mode)
try:
    from scripts.google_drive.drive_io import (
        get_drive_service,
        find_child_folder_id,
        list_pdfs_in_folder,
        download_file_bytes,
        sanitize_filename,
    )

    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

from src.preprocessing.pdf_text_extraction import extract_text_from_pdf_bytes


def run(cmd):
    """Run a subprocess command and exit if it fails."""
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def write_labels(labels: Dict[str, str], output_file: Path):
    """Write label dictionary to a JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def process_api_mode():
    """Download PDFs from Google Drive and process them."""
    if not GOOGLE_DRIVE_AVAILABLE:
        raise RuntimeError("Google Drive modules not available. " "Please install: pip install google-auth google-api-python-client")

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

    out_dir = PROJECT_ROOT / "data" / "processed-text"
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

    write_labels(labels, PROJECT_ROOT / "data" / "labels.json")
    print(f"Wrote {len(labels)} labeled text files.")
    return True


def process_local_mode(data_path: Path):
    """Process PDFs from local directory."""
    if not data_path.exists():
        raise RuntimeError(f"Data path does not exist: {data_path}")

    useful_dir = data_path / "useful"
    not_useful_dir = data_path / "not-useful"

    if not useful_dir.exists():
        raise RuntimeError(f"'useful' subfolder not found in {data_path}")
    if not not_useful_dir.exists():
        raise RuntimeError(f"'not-useful' subfolder not found in {data_path}")

    # Validate sufficient PDFs
    useful_pdfs = list(useful_dir.glob("*.pdf"))
    not_useful_pdfs = list(not_useful_dir.glob("*.pdf"))

    print(f"Found {len(useful_pdfs)} PDFs in 'useful' folder")
    print(f"Found {len(not_useful_pdfs)} PDFs in 'not-useful' folder")

    if len(useful_pdfs) < 2 or len(not_useful_pdfs) < 2:
        print("ERROR: Not enough PDF files!")
        print(f"Please add PDF files to:")
        print(f"  - {useful_dir}")
        print(f"  - {not_useful_dir}")
        print("You need at least 2 PDFs in each folder.")
        sys.exit(1)

    out_dir = PROJECT_ROOT / "data" / "processed-text"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels: Dict[str, str] = {}

    for folder, label in [(useful_dir, "useful"), (not_useful_dir, "not-useful")]:
        pdf_files = list(folder.glob("*.pdf"))
        print(f"Processing {len(pdf_files)} PDFs in local folder '{label}'")

        for pdf_path in pdf_files:
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                text = extract_text_from_pdf_bytes(pdf_bytes)
                stem = pdf_path.stem
                txt_name = f"{stem}.txt"
                (out_dir / txt_name).write_text(text, encoding="utf-8")
                labels[txt_name] = label
                print(f"Processed {pdf_path.name}")
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
                continue

    write_labels(labels, PROJECT_ROOT / "data" / "labels.json")
    print(f"Wrote {len(labels)} labeled text files.")
    return True


def generate_results():
    """Generate classification results CSV and JSON using subprocess for reliability."""
    print("\n" + "=" * 50)
    print("Generating classification results (CSV & JSON)...")
    print("=" * 50)

    useful_folder = PROJECT_ROOT / "data" / "useful"
    output_dir = PROJECT_ROOT / "data" / "results"
    model_dir = PROJECT_ROOT / "src" / "model" / "models"
    classifier_script = PROJECT_ROOT / "src" / "model" / "pdf_classifier.py"

    # Create results directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if useful_folder.exists() and list(useful_folder.glob("*.pdf")):
        print(f"\nClassifying PDFs in: {useful_folder}")

        # Use subprocess for more reliable imports across different systems
        result = subprocess.run([sys.executable, str(classifier_script), "--folder", str(useful_folder), "--model_dir", str(model_dir), "--output_dir", str(output_dir)], cwd=str(PROJECT_ROOT))

        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("OUTPUT FILES CREATED:")
            print("=" * 50)
            csv_file = output_dir / "classifications.csv"
            json_file = output_dir / "classifications.json"
            if csv_file.exists():
                print(f"  - {csv_file}")
            if json_file.exists():
                print(f"  - {json_file}")
            print("\nYou can open the CSV file!")
        else:
            print("\n[WARNING] Classification had some issues. Check output above.")
    else:
        print("\nNo PDFs found in useful folder to classify.")


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: extract text from PDFs and train model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Default (local): python full_pipeline.py
  API mode:        python full_pipeline.py --api
  Custom path:     python full_pipeline.py --local C:\\path\\to\\data
        """,
    )

    # Mutually exclusive: can't use both --api and --local
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--api", action="store_true", help="Use API mode to download PDFs from Google Drive")
    group.add_argument("--local", type=Path, metavar="PATH", default=None, help="Use local mode with PDFs from specified directory (default: data/)")

    args = parser.parse_args()

    print("=" * 50)
    print("FracFeedExtractor - Full Training Pipeline")
    print("=" * 50)
    print(f"Project folder: {PROJECT_ROOT}")

    if args.api:
        print("\nRunning in API mode (Google Drive)")
        try:
            process_api_mode()
        except RuntimeError as e:
            print(f"[ERROR] API mode failed: {e}")
            sys.exit(1)
    else:
        data_path = args.local if args.local else PROJECT_ROOT / "data"
        print(f"\nRunning in LOCAL mode")
        print(f"Data path: {data_path}")
        process_local_mode(data_path)

    print("\n" + "=" * 50)
    print("Beginning model training...")
    print("=" * 50)
    train_script = PROJECT_ROOT / "src" / "model" / "train_model.py"
    run([sys.executable, str(train_script)])

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Model saved to: {PROJECT_ROOT / 'src' / 'model' / 'models'}")

    # Generate CSV/JSON results
    generate_results()

    print("\n" + "=" * 50)
    print("All Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
