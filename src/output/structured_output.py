"""Structured Output Module

This module handles the export of classification and data extraction results
to JSON and CSV formats with clear provenance and uncertainty tracking.

Usage:
    from src.output import OutputManager, ClassificationResult

    # Create a classification result
    result = ClassificationResult(
        filename="Adams_1989.pdf",
        classification="useful",
        confidence=0.92,
        model_version="1.0.0"
    )

    # Export results
    manager = OutputManager(output_dir="data/results")
    manager.add_classification(result)
    manager.export_all()
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ClassificationResult:
    """Stores the result of classifying a single PDF."""

    filename: str
    classification: str
    confidence: float
    model_version: str = "1.0.0"
    processing_time_seconds: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    text_length: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ExtractionResult:
    """Stores extracted data from a useful PDF."""
    filename: str
    predator_species: Optional[str] = None
    predator_common_name: Optional[str] = None
    survey_location: Optional[str] = None
    survey_latitude: Optional[float] = None
    survey_longitude: Optional[float] = None
    survey_year: Optional[int] = None
    survey_month: Optional[int] = None
    total_stomachs_examined: Optional[int] = None
    empty_stomachs: Optional[int] = None
    non_empty_stomachs: Optional[int] = None
    fraction_feeding: Optional[float] = None
    sample_size_confidence: Optional[float] = None
    extraction_confidence: Optional[float] = None
    extraction_notes: Optional[str] = None
    source_text_snippet: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    extractor_version: str = "1.0.0"
    error: Optional[str] = None

    def __post_init__(self):
        """Calculate fraction_feeding if stomach counts are available."""
        if self.fraction_feeding is None and self.total_stomachs_examined is not None and self.non_empty_stomachs is not None and self.total_stomachs_examined > 0:
            self.fraction_feeding = self.non_empty_stomachs / self.total_stomachs_examined

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PipelineResult:
    """Combined result from the full pipeline (classification and extraction)."""

    filename: str
    classification: ClassificationResult
    extraction: Optional[ExtractionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "filename": self.filename,
            "classification": self.classification.to_dict(),
        }
        if self.extraction:
            result["extraction"] = self.extraction.to_dict()
        return result


class OutputManager:
    """Manages collection and export of pipeline results."""

    def __init__(self, output_dir: str = "data/results"):
        """Initialize the OutputManager."""

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classifications: List[ClassificationResult] = []
        self.extractions: List[ExtractionResult] = []
        self.pipeline_results: List[PipelineResult] = []

    def add_classification(self, result: ClassificationResult) -> None:
        """Add a classification result to the collection."""
        self.classifications.append(result)

    def add_extraction(self, result: ExtractionResult) -> None:
        """Add an extraction result to the collection."""
        self.extractions.append(result)

    def add_pipeline_result(self, result: PipelineResult) -> None:
        """Add a complete pipeline result to the collection."""
        self.pipeline_results.append(result)
        self.classifications.append(result.classification)
        if result.extraction:
            self.extractions.append(result.extraction)

    def export_classifications_json(self, filename: str = "classifications.json") -> Path:
        """Export classification results to JSON."""
        output_path = self.output_dir / filename
        data = {
            "metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_files": len(self.classifications),
                "useful_count": sum(1 for c in self.classifications if c.classification == "useful"),
                "not_useful_count": sum(1 for c in self.classifications if c.classification == "not-useful"),
            },
            "results": [c.to_dict() for c in self.classifications],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Classifications exported to {output_path}")
        return output_path

    def export_classifications_csv(self, filename: str = "classifications.csv") -> Path:
        """Export classification results to CSV."""
        output_path = self.output_dir / filename
        if not self.classifications:
            print("[WARN] No classifications to export.")
            return output_path

        fieldnames = list(self.classifications[0].to_dict().keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.classifications:
                writer.writerow(result.to_dict())
        print(f"[INFO] Classifications exported to {output_path}")
        return output_path

    def export_extractions_json(self, filename: str = "extractions.json") -> Path:
        """Export extraction results to JSON."""
        output_path = self.output_dir / filename
        data = {
            "metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_extractions": len(self.extractions),
                "successful_extractions": sum(1 for e in self.extractions if e.error is None),
            },
            "results": [e.to_dict() for e in self.extractions],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Extractions exported to {output_path}")
        return output_path

    def export_extractions_csv(self, filename: str = "extractions.csv") -> Path:
        """Export extraction results to CSV."""
        output_path = self.output_dir / filename
        if not self.extractions:
            print("[WARN] No extractions to export.")
            return output_path

        fieldnames = list(self.extractions[0].to_dict().keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.extractions:
                writer.writerow(result.to_dict())
        print(f"[INFO] Extractions exported to {output_path}")
        return output_path

    def export_all(self, prefix: str = "") -> Dict[str, Path]:
        """Export all results to both JSON and CSV formats."""

        paths = {}
        if self.classifications:
            paths["classifications_json"] = self.export_classifications_json(f"{prefix}classifications.json" if prefix else "classifications.json")
            paths["classifications_csv"] = self.export_classifications_csv(f"{prefix}classifications.csv" if prefix else "classifications.csv")
        if self.extractions:
            paths["extractions_json"] = self.export_extractions_json(f"{prefix}extractions.json" if prefix else "extractions.json")
            paths["extractions_csv"] = self.export_extractions_csv(f"{prefix}extractions.csv" if prefix else "extractions.csv")
        return paths

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected results."""
        return {
            "total_classifications": len(self.classifications),
            "useful_count": sum(1 for c in self.classifications if c.classification == "useful"),
            "not_useful_count": sum(1 for c in self.classifications if c.classification == "not-useful"),
            "total_extractions": len(self.extractions),
            "successful_extractions": sum(1 for e in self.extractions if e.error is None),
            "average_classification_confidence": (sum(c.confidence for c in self.classifications) / len(self.classifications) if self.classifications else 0.0),
        }


# Convenience functions for simple use cases
def export_to_json(results: List[Dict[str, Any]], output_path: str) -> Path:
    """Export a list of result dictionaries to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path


def export_to_csv(results: List[Dict[str, Any]], output_path: str) -> Path:
    """Export a list of result dictionaries to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        # Create empty file
        path.touch()
        return path

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return path


if __name__ == "__main__":
    # Example usage demonstration
    print("=== FracFeedExtractor Structured Output ===\n")

    # Create sample classification results
    class_results = [
        ClassificationResult(
            filename="Adams_1989.pdf",
            classification="useful",
            confidence=0.92,
            text_length=45000,
        ),
        ClassificationResult(
            filename="Rosalino_2009.pdf",
            classification="not-useful",
            confidence=0.85,
            text_length=32000,
        ),
    ]

    # Create sample extraction result
    extraction = ExtractionResult(
        filename="Adams_1989.pdf",
        predator_species="Pygoscelis papua",
        predator_common_name="Gentoo Penguin",
        survey_location="Marion Island, sub-Antarctic",
        survey_latitude=-46.88,
        survey_longitude=37.90,
        survey_year=1984,
        total_stomachs_examined=144,
        empty_stomachs=12,
        non_empty_stomachs=132,
        extraction_confidence=0.88,
        source_text_snippet="A total of 144 stomach samples was collected...",
    )

    # Use OutputManager
    manager = OutputManager(output_dir="data/results")

    for result in class_results:
        manager.add_classification(result)

    manager.add_extraction(extraction)

    # Export all results
    paths = manager.export_all()

    print("\nExported files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("\nSummary:")
    summary = manager.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")