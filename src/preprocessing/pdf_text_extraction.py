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

Image.MAX_IMAGE_PIXELS = None
fitz.TOOLS.mupdf_display_errors(False)

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
            str(pdf_path), 
            pages='all', 
            flavor='stream',
            edge_tol=500,  # Tolerance for detecting table edges
            row_tol=10,    # Ttolerance for row detection
            column_tol=5   # Tolerance for column detection
        )
        
        # If still no tables, try lattice method (for bordered tables)
        if len(tables) == 0:
            tables = camelot.read_pdf(
                str(pdf_path), 
                pages='all', 
                flavor='lattice',
                line_scale=40
            )
        
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
                    "y1": table._bbox[3] if hasattr(table, '_bbox') else 0
                },
                "num_rows": len(table_cells),
                "num_cols": len(table_cells[0]) if table_cells else 0,
                "cells": table_cells,
                "accuracy": float(table.accuracy) if hasattr(table, 'accuracy') else 0.0,
                "extraction_method": "camelot"
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
                            "bbox": {
                                "x0": bbox.x0,
                                "y0": bbox.y0,
                                "x1": bbox.x1,
                                "y1": bbox.y1
                            },
                            "num_rows": len(table_cells),
                            "num_cols": len(table_cells[0]) if table_cells else 0,
                            "cells": table_cells,
                            "extraction_method": "pymupdf"
                        }
                        
                        tables_data.append(table_info)
                        
                    except Exception as e:
                        tables_data.append({
                            "table_id": f"Table_P{page_num}_T{table_idx}",
                            "page_number": page_num,
                            "table_index": table_idx,
                            "error": str(e),
                            "extraction_method": "pymupdf"
                        })
                        
    except Exception as e:
        print(f"[ERROR] PyMuPDF table extraction failed: {e}", file=sys.stderr)
    
    # If PyMuPDF found no tables, try camelot
    if len(tables_data) == 0:
        tables_data = extract_tables_with_camelot(pdf_path)
    
    return tables_data


def extract_text_from_pdf(pdf_path: str) -> str:
    text = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                # Extract text from the page using PyMuPDF
                page_text = page.get_text("text")

                # Clean out null bytes or UTF-16 artifacts
                if "\x00" in page_text:
                    page_text = page_text.replace("\x00", "")

                # If the page is mostly empty, treat as image and use OCR
                if not page_text.strip():
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_text = pytesseract.image_to_string(img)

                text.append(page_text)
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}", file=sys.stderr)
        return ""
    
    return "\n".join(text)


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from an in-memory PDF without writing the PDF to disk."""
    text = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                if "\x00" in page_text:
                    page_text = page_text.replace("\x00", "")
                if not page_text.strip():
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_text = pytesseract.image_to_string(img)
                text.append(page_text)
    except Exception as e:
        print(f"[ERROR] Failed to extract text from PDF bytes: {e}", file=sys.stderr)
        return ""
    return "\n".join(text)


def save_to_file(text: str, output_path: str):
    """Save extracted text to a file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[ERROR] Could not save text to {output_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Extract text and tables from PDF using PyMuPDF and camelot-py."
    )
    parser.add_argument("pdf", type=str, help="Path to the input PDF file.")
    parser.add_argument("--output-dir", type=str, default="data/processed-text",
                        help="Output directory for extracted text (default: data/processed-text)")
    
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