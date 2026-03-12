"""PDF Text Extraction Script

Usage:
    python pdf_to_text.py path/to/input.pdf

This script uses PyMuPDF for accurate and efficient text extraction from
scientific PDFs. It preserves reading order, handles multi-column text, and
automatically applies OCR when a page contains only images (e.g., scanned documents).
Tables are automatically detected and extracted using PyMuPDF first, with camelot-py as fallback.
"""

import fitz
import pytesseract
from PIL import Image
import io
import argparse
from pathlib import Path
import sys
import json
from typing import List, Dict
import camelot
import cv2
import numpy as np
from spellchecker import SpellChecker

Image.MAX_IMAGE_PIXELS = None
fitz.TOOLS.mupdf_display_errors(False)

# Maximum allowed ratio of misspelled words to total words in a pdf
MAX_SPELLING_ERROR_RATE = 0.05

# Module-level singleton — avoids reloading the dictionary for every PDF
_spell_checker = SpellChecker()


def check_spelling(text: str) -> float:
    """
    Returns ratio of mispelled words to total words

    Returns 1 if no words detected in input string
    """
    words = _spell_checker.split_words(text)
    if len(words) == 0:
        return 1
    misspelled = _spell_checker.unknown(words)
    return len(misspelled) / len(words)


def extract_tables_with_camelot(pdf_path: str) -> List[Dict]:
    """Extract tables using camelot-py (fallback method).

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries containing table data and metadata
    """

    tables_data = []

    try:
        # Try stream method first with edge detection (better for tables without borders)
        tables = camelot.read_pdf(
            str(pdf_path), pages='all', flavor='stream', edge_tol=500, row_tol=10, column_tol=5  # Tolerance for detecting table edges  # Ttolerance for row detection  # Tolerance for column detection
        )

        # If still no tables, try lattice method (for bordered tables)
        if len(tables) == 0:
            tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice', line_scale=40)

        for idx, table in enumerate(tables, start=1):
            # Convert to list of lists
            table_cells = table.df.values.tolist()

            # Add header row (pandas columns)
            header = table.df.columns.tolist()
            table_cells.insert(0, header)

            # Skip tables with very few cells (likely detection errors)
            if len(table_cells) < 3 or (len(table_cells[0]) if table_cells else 0) < 2:
                continue

            table_info = {
                "table_id": f"Table_P{table.page}_T{idx}",
                "page_number": table.page,
                "table_index": idx,
                "bbox": {
                    "x0": table._bbox[0] if hasattr(table, '_bbox') else 0,
                    "y0": table._bbox[1] if hasattr(table, '_bbox') else 0,
                    "x1": table._bbox[2] if hasattr(table, '_bbox') else 0,
                    "y1": table._bbox[3] if hasattr(table, '_bbox') else 0,
                },
                "num_rows": len(table_cells),
                "num_cols": len(table_cells[0]) if table_cells else 0,
                "cells": table_cells,
                "accuracy": float(table.accuracy) if hasattr(table, 'accuracy') else 0.0,
                "extraction_method": "camelot",
            }

            tables_data.append(table_info)

    except Exception as e:
        print(f"[ERROR] Camelot extraction failed: {e}", file=sys.stderr)

    return tables_data


def extract_tables_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract tables from PDF using PyMuPDF first, then camelot-py as fallback.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries containing table data and metadata
    """
    tables_data = []

    # Try PyMuPDF first
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                tabs = page.find_tables()

                if not tabs.tables:
                    continue

                for table_idx, tab in enumerate(tabs.tables, start=1):
                    try:
                        table_cells = tab.extract()

                        if not table_cells or len(table_cells) == 0:
                            continue

                        bbox = tab.bbox

                        table_info = {
                            "table_id": f"Table_P{page_num}_T{table_idx}",
                            "page_number": page_num,
                            "table_index": table_idx,
                            "bbox": {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1},
                            "num_rows": len(table_cells),
                            "num_cols": len(table_cells[0]) if table_cells else 0,
                            "cells": table_cells,
                            "extraction_method": "pymupdf",
                        }

                        tables_data.append(table_info)

                    except Exception as e:
                        tables_data.append({"table_id": f"Table_P{page_num}_T{table_idx}", "page_number": page_num, "table_index": table_idx, "error": str(e), "extraction_method": "pymupdf"})

    except Exception as e:
        print(f"[ERROR] PyMuPDF table extraction failed: {e}", file=sys.stderr)

    # If PyMuPDF found no tables, try camelot
    if len(tables_data) == 0:
        tables_data = extract_tables_with_camelot(pdf_path)

    return tables_data


def parse_page_ocr(page: fitz.Page, page_num: int) -> str:
    """Extracts text from page using OCR"""
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.fastNlMeansDenoising(img_array, h=10, templateWindowSize=7, searchWindowSize=21)
    page_text = pytesseract.image_to_string(img_array)
    return f"[PAGE {page_num}]\n{page_text}"


def parse_page_embedded(page: fitz.Page, page_num: int) -> str:
    """Extracts text embedded in a PDF page"""
    page_text = page.get_text("text")
    if "\x00" in page_text:
        page_text = page_text.replace("\x00", "")
    return f"[PAGE {page_num}]\n{page_text}"


def extract_text_from_pdf(pdf_path: str, skip_ocr: bool = False) -> str:
    text = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text.append(parse_page_embedded(page, page_num))  # updated
            text = "\n".join(text)
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}", file=sys.stderr)
        return ""

    if not skip_ocr and check_spelling(text) > MAX_SPELLING_ERROR_RATE:
        print(f"[INFO] High misspelling rate in {Path(pdf_path).name} — falling back to OCR", file=sys.stderr)
        text = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text.append(parse_page_ocr(page, page_num))  # updated
                text = "\n".join(text)
        except Exception as e:
            print(f"[ERROR] Failed to extract text from {pdf_path}: {e}", file=sys.stderr)
            return ""
    return text


def extract_text_from_pdf_bytes(data: bytes, skip_ocr: bool = False) -> str:
    """Extract text from an in-memory PDF without writing the PDF to disk."""
    text = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text.append(parse_page_embedded(page, page_num))  # updated
            text = "\n".join(text)
    except Exception as e:
        print(f"[ERROR] Failed to extract text from PDF bytes: {e}", file=sys.stderr)
        return ""

    if not skip_ocr and check_spelling(text) > MAX_SPELLING_ERROR_RATE:
        print(f"[INFO] High misspelling rate — falling back to OCR", file=sys.stderr)
        text = []
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page_num, page in enumerate(doc, start=1):
                    text.append(parse_page_ocr(page, page_num))  # updated
                text = "\n".join(text)
        except Exception as e:
            print(f"[ERROR] Failed to extract text from PDF bytes: {e}", file=sys.stderr)
            return ""
    return text


def save_to_file(text: str, output_path: str):
    """Save extracted text to a file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[ERROR] Could not save text to {output_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Extract text and tables from PDF using PyMuPDF and camelot-py.")
    parser.add_argument("pdf", type=str, help="Path to the input PDF file.")
    parser.add_argument("--output-dir", type=str, default="data/processed-text", help="Output directory for extracted text (default: data/processed-text)")

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Extract text
    text = extract_text_from_pdf(str(pdf_path))

    # Extract tables
    tables_data = extract_tables_from_pdf(str(pdf_path))

    # If tables were found, save JSON
    if tables_data:

        # Save tables JSON
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables_json_path = output_dir / (pdf_path.stem + "_tables.json")
        with open(tables_json_path, "w", encoding="utf-8") as f:
            json.dump(tables_data, f, indent=2)

    # Save combined text
    output_path = Path(args.output_dir) / pdf_path.with_suffix(".txt").name
    save_to_file(text, str(output_path))


if __name__ == "__main__":
    main()
