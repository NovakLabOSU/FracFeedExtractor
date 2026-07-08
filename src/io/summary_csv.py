"""Shared schema and writer for the extraction summary CSV.

Provides a single source of truth for the summary-CSV columns and the writer,
so both run paths emit the same table and cannot drift apart:

  - the batch pipeline   (src/pipeline/classify_extract.py)
  - the single-file tool (src/extraction/llm_client.py)

The batch path fills the classifier columns (classification / confidence /
pred_prob); the single-file path leaves them blank since it has no
classification step.  Every other column is shared.

Rows are stored internally with machine-friendly keys (species_name, etc.),
but the CSV is written with reader-friendly headers (see COLUMN_LABELS) and a
reader-friendly column order (the extracted data first, classifier metadata
last).  The writer is tolerant of partial rows (e.g. a failed worker that only
has a filename and an extraction_status); missing columns are written blank.

Exposes three helpers::

    metrics_to_row(filename, metrics, ...) -> dict
    write_summary_csv(rows, output_dir) -> Path
    blank_row(filename) -> dict
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.config import FIELDS

# Canonical column order for every summary CSV the project writes.  The
# extracted data is listed first, with classifier metadata trailing at the end.
SUMMARY_FIELDNAMES: List[str] = (
    ["filename"]
    + [spec.name for spec in FIELDS]
    + ["fraction_feeding", "classification", "confidence", "pred_prob", "extraction_status"]
)

# Reader-friendly header shown in the CSV, one per key in SUMMARY_FIELDNAMES.
COLUMN_LABELS: Dict[str, str] = {
    "filename": "File",
    **{spec.name: spec.csv_label for spec in FIELDS},
    "fraction_feeding": "Fraction Feeding",
    "classification": "Classification",
    "confidence": "Classifier Confidence",
    "pred_prob": "Raw P(useful)",
    "extraction_status": "Status",
}

# Metric fields copied straight out of an extracted metrics dict.
_METRIC_FIELDS = tuple(spec.name for spec in FIELDS) + ("fraction_feeding",)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def blank_row(filename: str) -> Dict[str, str]:
    """Return a summary row with every column present and empty except filename."""
    row = {key: "" for key in SUMMARY_FIELDNAMES}
    row["filename"] = filename
    return row


def metrics_to_row(
    filename: str,
    metrics: Dict,
    classification: str = "",
    confidence: str = "",
    pred_prob: str = "",
    extraction_status: str = "success",
) -> Dict[str, str]:
    """Flatten an extracted metrics dict into one summary CSV row.

    Args:
        filename: Source file name.
        metrics: Extracted metrics dict (e.g. PredatorDietMetrics.model_dump()).
        classification: Classifier label, if any (batch path only).
        confidence: Classifier confidence string, if any.
        pred_prob: Raw classifier probability string, if any.
        extraction_status: Outcome of the extraction step.

    Returns:
        A dict keyed by SUMMARY_FIELDNAMES, with None values rendered as "".
    """
    row = blank_row(filename)
    row["classification"] = classification
    row["confidence"] = confidence
    row["pred_prob"] = pred_prob
    row["extraction_status"] = extraction_status
    for field in _METRIC_FIELDS:
        value = metrics.get(field)
        row[field] = "" if value is None else value
    return row


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_summary_csv(
    rows: List[Dict],
    output_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """Write summary rows to ``output_dir/summaries/<filename>``.

    The header row uses the reader-friendly labels in COLUMN_LABELS, and values
    are written in SUMMARY_FIELDNAMES order.  Missing keys are written blank, so
    partial rows (e.g. a failed worker with only filename + extraction_status)
    are safe to pass.

    Args:
        rows: One dict per file, keyed by SUMMARY_FIELDNAMES.
        output_dir: Results directory; the CSV is written under its
            ``summaries`` subfolder.
        filename: Output file name.  When None, a timestamped
            ``pipeline_summary_<YYYYmmdd_HHMMSS>.csv`` name is generated.

    Returns:
        Path to the CSV that was written.
    """
    summaries_dir = Path(output_dir) / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_summary_{timestamp}.csv"

    summary_path = summaries_dir / filename
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([COLUMN_LABELS[key] for key in SUMMARY_FIELDNAMES])
        for row in rows:
            writer.writerow([row.get(key, "") for key in SUMMARY_FIELDNAMES])

    return summary_path
