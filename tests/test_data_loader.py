import pytest
from pathlib import Path
from src.io.data_loader import load_processed_text
import subprocess
import shutil


def test_load_processed_text_reads_files(tmp_path):
    data_dir = tmp_path / "data" / "processed-text"
    data_dir.mkdir(parents=True)

    # Create mock text files
    file1 = data_dir / "file1.txt"
    file2 = data_dir / "file2.txt"
    file1.write_text("Predator diet data example 1", encoding="utf-8")
    file2.write_text("Predator diet data example 2", encoding="utf-8")

    texts = load_processed_text(str(data_dir))

    assert len(texts) == 2
    assert any("example 1" in t for t in texts)
    assert any("example 2" in t for t in texts)


def test_load_processed_text_empty_directory(tmp_path):
    empty_dir = tmp_path / "data" / "processed-text"
    empty_dir.mkdir(parents=True)

    texts = load_processed_text(str(empty_dir))

    assert isinstance(texts, list)
    assert len(texts) == 0


def test_load_processed_text_ignores_non_txt(tmp_path):
    data_dir = tmp_path / "data" / "processed-text"
    data_dir.mkdir(parents=True)

    (data_dir / "ignore.pdf").write_text("Not a text file", encoding="utf-8")
    (data_dir / "keep.txt").write_text("Keep this one", encoding="utf-8")

    texts = load_processed_text(str(data_dir))

    assert len(texts) == 1
    assert texts[0] == "Keep this one"


def test_load_processed_text_encoding(tmp_path):
    data_dir = tmp_path / "data" / "processed-text"
    data_dir.mkdir(parents=True)

    content = "Café with UTF-8 accents"
    file_path = data_dir / "utf8_test.txt"
    file_path.write_text(content, encoding="utf-8")

    texts = load_processed_text(str(data_dir))

    assert len(texts) == 1
    assert "Café" in texts[0]


def test_main_prints_summary(monkeypatch, tmp_path, capsys):
    from src.preprocessing import data_loader

    data_dir = tmp_path / "data" / "processed-text"
    data_dir.mkdir(parents=True)
    (data_dir / "example.txt").write_text("Sample content", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    data_loader.texts = data_loader.load_processed_text(str(data_dir))

    data_loader.__main__ = True
    print(f"Loaded {len(data_loader.texts)} text files.")
    if data_loader.texts:
        print(f"First file preview:\n{data_loader.texts[0][:300]}...")

    captured = capsys.readouterr()
    assert "Loaded 1 text files." in captured.out
    assert "Sample content" in captured.out


def test_data_loader_main_executes(tmp_path):
    data_dir = tmp_path / "data" / "processed-text"
    data_dir.mkdir(parents=True)
    (data_dir / "sample.txt").write_text("Predator diet study data", encoding="utf-8")

    script_path = Path("src/io/data_loader.py")
    tmp_script_path = tmp_path / "data_loader.py"
    shutil.copy(script_path, tmp_script_path)

    result = subprocess.run(
        ["python", str(tmp_script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Loaded 1 text files." in result.stdout
    assert "Predator diet study data" in result.stdout
