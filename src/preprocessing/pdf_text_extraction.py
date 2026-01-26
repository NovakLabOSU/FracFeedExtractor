"""PDF Text Extraction Script

Usage:
    python pdf_to_text.py path/to/input.pdf

This script uses PyMuPDF for accurate and efficient text extraction from
scientific PDFs. It preserves reading order, handles multi-column text, and
automatically applies OCR when a page contains only images (e.g., scanned documents).
"""

# Extract all text from a PDF using PyMuPDF
import fitz
import pytesseract
from PIL import Image
import io
import argparse
from pathlib import Path
import sys

Image.MAX_IMAGE_PIXELS = None
fitz.TOOLS.mupdf_display_errors(False)


def extract_text_from_pdf(pdf_path: str) -> str:
    text = []
    print(f"Extracting text from {pdf_path}.")
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
    # Join all pages into a single string separated by newlines
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


# Save extracted text to a file.
def save_to_file(text: str, output_path: str):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[INFO] Text successfully saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Could not save text to {output_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF using PyMuPDF.")
    parser.add_argument("pdf", type=str, help="Path to the input PDF file.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Perform extraction
    text = extract_text_from_pdf(str(pdf_path))

    output_path = Path("data/processed-text") / pdf_path.with_suffix(".txt").name

    save_to_file(text, str(output_path))


if __name__ == "__main__":
    main()
