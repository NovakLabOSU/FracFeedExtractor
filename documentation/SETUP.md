# Setup Guide

Step-by-step instructions for getting FracFeedExtractor running on a new machine.

---

## 1. System Dependencies

Install the following before touching Python. These are system-level tools that `pip` cannot install.

### Ollama (required — runs the LLM locally)

Download and install from [ollama.com](https://ollama.com). Minimum 8 GB RAM; 16 GB recommended for `qwen2.5:7b`.

After installing, pull the default extraction model:

```bash
ollama pull qwen2.5:7b   # ~5 GB download
ollama list              # verify it appears
```

Ollama must be running in the background whenever you use the pipeline. On macOS and Windows it starts automatically after install. On Linux, run `ollama serve` in a separate terminal.

### Tesseract OCR (required for scanned PDFs)

| OS | Command |
|---|---|
| Windows | Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), or `choco install tesseract` |
| macOS | `brew install tesseract` |
| Ubuntu/Debian | `sudo apt install tesseract-ocr` |

After installing, verify it is on your PATH:

```bash
tesseract --version
```

### Ghostscript (optional — improves table extraction)

The pipeline extracts tables using PyMuPDF first, then camelot's stream mode as a fallback — neither requires Ghostscript. Ghostscript is only used by camelot's lattice mode, which is a last-resort fallback for bordered tables that the other two methods missed. The pipeline will run without it.

| OS | Command |
|---|---|
| Windows | Download from [ghostscript.com](https://www.ghostscript.com/releases/gsdnld.html), or `choco install ghostscript` |
| macOS | `brew install ghostscript` |
| Ubuntu/Debian | `sudo apt install ghostscript` |

---

## 2. Clone the Repository

```bash
git clone https://github.com/NovakLabOSU/FracFeedExtractor.git
cd FracFeedExtractor
```

---

## 3. Python Environment

Requires Python 3.10 or higher. Check your version with `python --version` or `python3 --version`.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Windows PowerShell

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

> If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

The `.[dev]` install includes the core pipeline dependencies plus `pytest`, `coverage`, `black`, and `flake8`.

---

## 4. Verify the Installation

Run the test suite to confirm everything is wired up correctly:

```bash
pytest tests/
```

All tests should pass. If you see import errors, check that your virtual environment is activated and that `pip install -e ".[dev]"` completed without errors.

---

## 5. Run the Pipeline

The classifier artifacts are already committed to the repository (`src/classifier/models/`), so you can run the pipeline immediately after setup without retraining.

```bash
# Classify and extract from a single PDF
python src/pipeline/classify_extract.py path/to/file.pdf

# Classify and extract from a folder of PDFs
python src/pipeline/classify_extract.py path/to/pdfs/
```

Results are written to:
- `data/results/metrics/` — one JSON file per paper classified as useful
- `data/results/summaries/` — a pipeline summary CSV covering all processed files

### Full CLI reference

```bash
python src/pipeline/classify_extract.py path/to/pdfs/ \
    --model-dir src/classifier/models \
    --llm-model qwen2.5:7b \
    --output-dir data/results \
    --confidence-threshold 0.70 \
    --max-chars 12000 \
    --num-ctx 4096 \
    --workers 1
```

| Flag | Default | Description |
|---|---|---|
| `--model-dir` | `src/classifier/models` | Directory containing classifier artifacts |
| `--llm-model` | `qwen2.5:7b` | Ollama model used for extraction |
| `--output-dir` | `data/results` | Destination for JSON results and summary CSV |
| `--confidence-threshold` | `0.70` | Minimum classifier confidence to route a paper to the LLM |
| `--max-chars` | `12000` | Maximum characters sent to the LLM per paper |
| `--num-ctx` | `4096` | Ollama context window size in tokens |
| `--workers` | `1` | Number of parallel worker processes (`1` = sequential) |

> **Windows + multiple workers:** Tesseract sometimes does not work reliably with Python multiprocessing on Windows. If workers hang or crash on a batch that includes scanned PDFs, reduce `--workers` back to `1`.

---

## 6. Environment Variables (optional — only needed for API mode)

If you intend to use `scripts/full_pipeline.py --api` to fetch the training dataset from Google Drive, you need a `.env` file in the project root. Copy the example and fill in the values:

```bash
# Linux / macOS
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

Then open `.env` and add the credentials. Contact the project partner (Mark Novak) or a returning team member to obtain the Google service account key and Drive folder ID. The `.env` file is excluded from version control — never commit it.

---

## 7. Retraining the Classifier (optional)

The committed classifier artifacts are ready to use. Retraining is only necessary if you add new labeled data.

1. Add extracted text files to `data/processed-text/` and update `data/labels.json` with `"filename.txt": "useful"` or `"filename.txt": "not useful"` entries.
2. Run the trainer:

```bash
python -m src.classifier.train_model
```

This overwrites the three artifacts in `src/classifier/models/`. See `documentation/CONTRIBUTING.md` for tunable hyperparameters.

---

## 8. Sample Output

A paper classified as useful produces a JSON file like this in `data/results/metrics/`:

```json
{
  "source_file": "Smith_2002.pdf",
  "extracted_at": "2026-04-24T14:32:00",
  "metrics": {
    "species_name": "Esox lucius",
    "study_location": "Lake Windermere, UK",
    "study_date": "1998-2000",
    "num_empty_stomachs": 42,
    "num_nonempty_stomachs": 158,
    "sample_size": 200,
    "fraction_feeding": 0.79
  }
}
```

---

## Troubleshooting

**`tesseract` not found at runtime**
Tesseract is installed but not on PATH. On Windows, add the Tesseract install directory (e.g., `C:\Program Files\Tesseract-OCR`) to your system PATH and restart your terminal.

**`camelot` import error or Ghostscript not found**
Ghostscript is not installed or not on PATH. Re-check step 1. On Windows, the Ghostscript installer does not always add itself to PATH automatically — check the install directory and add it manually if needed.

**Ollama connection refused**
Ollama is not running. Start it with `ollama serve` (Linux) or open the Ollama app (macOS/Windows), then retry.

**`ModuleNotFoundError` for any `src.*` import**
The package is not installed in editable mode. Run `pip install -e ".[dev]"` from the project root with your virtual environment activated.

**PowerShell blocks `Activate.ps1`**
Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell as your user (not as Administrator), then activate again.
