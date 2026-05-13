import pytest
import subprocess
import fitz
from pathlib import Path
import sys
from src.io.pdf_text_extraction import extract_text_from_pdf, save_to_file, main


def test_extract_text_exists():
    pdf_path = Path("tests/test.pdf")
    text = extract_text_from_pdf(str(pdf_path))
    assert text is not None
    assert len(text) > 0


def test_save_to_file_creates_output(tmp_path):
    output_path = tmp_path / "output.txt"
    text = "Sample text"
    save_to_file(text, output_path)
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == text


def test_main_handles_missing_pdf(monkeypatch, tmp_path):
    fake_pdf = tmp_path / "nonexistent.pdf"
    test_args = ["pdf_to_text.py", str(fake_pdf)]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit):
        main()


def test_extract_text_invalid_path(tmp_path):
    invalid_pdf = tmp_path / "does_not_exist.pdf"
    result = extract_text_from_pdf(str(invalid_pdf))
    assert result == ""


def test_extract_text_with_ocr(tmp_path):
    pdf_path = tmp_path / "image_only.pdf"

    # Create image-only PDF
    doc = fitz.open()
    page = doc.new_page()
    # Draw white rectangle (no text layer)
    rect = fitz.Rect(0, 0, 100, 100)
    page.insert_textbox(rect, "")  # empty text
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100))
    page.insert_image(rect, pixmap=pix)
    doc.save(pdf_path)
    doc.close()

    text = extract_text_from_pdf(str(pdf_path))
    assert isinstance(text, str)
    # Empty string is acceptable if OCR yields nothing, but must not crash
    assert text is not None


def test_save_to_file_permission_error(monkeypatch, tmp_path):
    import io

    output_file = tmp_path / "readonly.txt"
    output_file.touch()
    output_file.chmod(0o400)

    monkeypatch.setattr("sys.stderr", io.StringIO())

    # Should not raise exception even if write fails
    save_to_file("data", output_file)


def test_main_cli(tmp_path):
    input_pdf = "tests/test.pdf"
    output_dir = Path("data/processed-text")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test.txt"

    result = subprocess.run(
        ["python", "src/io/pdf_text_extraction.py", input_pdf],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert output_file.exists()
    assert output_file.stat().st_size > 0
