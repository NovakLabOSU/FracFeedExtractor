"""End-to-end integration tests for src/pipeline/classify_extract.py.

Uses real PDFs from data/useful/ and data/not-useful/ with the committed
classifier artifacts. The LLM call is mocked so no Ollama server is needed.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.models import PredatorDietMetrics
from src.pipeline.classify_extract import run_pipeline

# Paths to the committed test assets
REPO_ROOT = Path(__file__).resolve().parents[1]
USEFUL_PDF = REPO_ROOT / "data" / "useful" / "Fisher_2008.pdf"
NOT_USEFUL_PDF = REPO_ROOT / "data" / "not-useful" / "AlejoPlata_2019.pdf"
MODEL_DIR = str(REPO_ROOT / "src" / "classifier" / "models")

# Minimal valid extraction result for mocking
_MOCK_RECORD = PredatorDietMetrics(
    species_name="Gadus morhua",
    study_location="North Sea",
    num_empty=10,
    num_nonempty=90,
    num_sampled=100,
)


@pytest.fixture()
def output_dir(tmp_path):
    return tmp_path / "results"


@pytest.mark.skipif(not USEFUL_PDF.exists(), reason="data/useful/Fisher_2008.pdf not found")
def test_integration_useful_pdf_runs_llm_and_writes_json(output_dir):
    """A useful PDF is classified, the LLM is called, and a result JSON is written."""
    with patch(
        "src.pipeline.classify_extract.extract_metrics_from_text",
        return_value=[_MOCK_RECORD],
    ) as mock_llm:
        run_pipeline(
            input_path=USEFUL_PDF,
            model_dir=MODEL_DIR,
            llm_model="test-model",
            output_dir=output_dir,
            confidence_threshold=0.70,
            max_chars=12000,
            num_ctx=4096,
        )

    # LLM must have been called
    mock_llm.assert_called_once()

    # At least one JSON result file must be written
    metrics_dir = output_dir / "metrics"
    json_files = list(metrics_dir.glob("*.json"))
    assert json_files, "No result JSON written for a useful PDF"

    result = json.loads(json_files[0].read_text())
    assert "records" in result
    assert len(result["records"]) >= 1

    # Each record must pass schema validation
    for rec in result["records"]:
        validated = PredatorDietMetrics(**{k: v for k, v in rec.items() if k != "source_pages"})
        assert validated.species_name == "Gadus morhua"


@pytest.mark.skipif(not NOT_USEFUL_PDF.exists(), reason="data/not-useful/AlejoPlata_2019.pdf not found")
def test_integration_not_useful_pdf_skips_llm(output_dir):
    """A not-useful PDF is rejected by the classifier; the LLM is never called."""
    with patch(
        "src.pipeline.classify_extract.extract_metrics_from_text",
    ) as mock_llm:
        run_pipeline(
            input_path=NOT_USEFUL_PDF,
            model_dir=MODEL_DIR,
            llm_model="test-model",
            output_dir=output_dir,
            confidence_threshold=0.70,
            max_chars=12000,
            num_ctx=4096,
        )

    # LLM must NOT be called for a not-useful paper
    mock_llm.assert_not_called()

    # No extraction JSON should exist
    metrics_dir = output_dir / "metrics"
    json_files = list(metrics_dir.glob("*.json")) if metrics_dir.exists() else []
    assert not json_files, "Result JSON was written for a not-useful PDF"

    # A summary CSV should exist (the pipeline always writes one)
    summaries_dir = output_dir / "summaries"
    csv_files = list(summaries_dir.glob("*.csv")) if summaries_dir.exists() else []
    assert csv_files, "No summary CSV written"
