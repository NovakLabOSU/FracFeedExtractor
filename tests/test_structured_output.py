"""Tests for the structured output module."""

import pytest
import json
import csv
from pathlib import Path
from src.output.structured_output import (
    ClassificationResult,
    ExtractionResult,
    PipelineResult,
    OutputManager,
    export_to_json,
    export_to_csv,
)


class TestClassificationResult:
    # Tests for ClassificationResult dataclass

    def test_create_basic_result(self):
        result = ClassificationResult(
            filename="test.pdf",
            classification="useful",
            confidence=0.85,
        )
        assert result.filename == "test.pdf"
        assert result.classification == "useful"
        assert result.confidence == 0.85
        assert result.model_version == "1.0.0"

    def test_to_dict(self):
        result = ClassificationResult(
            filename="test.pdf",
            classification="useful",
            confidence=0.85,
        )
        d = result.to_dict()
        assert d["filename"] == "test.pdf"
        assert d["classification"] == "useful"
        assert d["confidence"] == 0.85
        assert "timestamp" in d

    def test_with_error(self):
        result = ClassificationResult(
            filename="bad.pdf",
            classification="unknown",
            confidence=0.0,
            error="Failed to extract text",
        )
        assert result.error == "Failed to extract text"


class TestExtractionResult:
    # Tests for ExtractionResult dataclass

    def test_create_basic_result(self):
        result = ExtractionResult(
            filename="test.pdf",
            predator_species="Canis lupus",
            survey_year=2020,
        )
        assert result.filename == "test.pdf"
        assert result.predator_species == "Canis lupus"
        assert result.survey_year == 2020

    def test_fraction_feeding_auto_calculation(self):
        result = ExtractionResult(
            filename="test.pdf",
            total_stomachs_examined=100,
            non_empty_stomachs=75,
        )
        assert result.fraction_feeding == 0.75

    def test_fraction_feeding_not_calculated_when_missing_data(self):
        result = ExtractionResult(
            filename="test.pdf",
            total_stomachs_examined=100,
            # non_empty_stomachs not provided
        )
        assert result.fraction_feeding is None

    def test_fraction_feeding_not_overwritten(self):
        result = ExtractionResult(
            filename="test.pdf",
            total_stomachs_examined=100,
            non_empty_stomachs=75,
            fraction_feeding=0.80,  # Manually set
        )
        assert result.fraction_feeding == 0.80  # Should not be overwritten

    def test_to_dict(self):
        result = ExtractionResult(
            filename="test.pdf",
            predator_species="Canis lupus",
            survey_location="Yellowstone",
            survey_year=2020,
            total_stomachs_examined=50,
            empty_stomachs=10,
            non_empty_stomachs=40,
        )
        d = result.to_dict()
        assert d["predator_species"] == "Canis lupus"
        assert d["survey_location"] == "Yellowstone"
        assert d["fraction_feeding"] == 0.80


class TestPipelineResult:
    # Tests for PipelineResult dataclass

    def test_create_useful_result(self):
        classification = ClassificationResult(
            filename="test.pdf",
            classification="useful",
            confidence=0.9,
        )
        extraction = ExtractionResult(
            filename="test.pdf",
            predator_species="Canis lupus",
        )
        pipeline = PipelineResult(
            filename="test.pdf",
            classification=classification,
            extraction=extraction,
        )
        assert pipeline.extraction is not None

    def test_create_not_useful_result(self):
        classification = ClassificationResult(
            filename="test.pdf",
            classification="not-useful",
            confidence=0.85,
        )
        pipeline = PipelineResult(
            filename="test.pdf",
            classification=classification,
            extraction=None,
        )
        assert pipeline.extraction is None

    def test_to_dict(self):
        classification = ClassificationResult(
            filename="test.pdf",
            classification="useful",
            confidence=0.9,
        )
        pipeline = PipelineResult(
            filename="test.pdf",
            classification=classification,
        )
        d = pipeline.to_dict()
        assert "classification" in d
        assert d["classification"]["confidence"] == 0.9


class TestOutputManager:
    # Tests for OutputManager class

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "results"

    @pytest.fixture
    def manager(self, output_dir):
        return OutputManager(output_dir=str(output_dir))

    @pytest.fixture
    def sample_classifications(self):
        return [
            ClassificationResult(filename="a.pdf", classification="useful", confidence=0.9),
            ClassificationResult(filename="b.pdf", classification="not-useful", confidence=0.8),
            ClassificationResult(filename="c.pdf", classification="useful", confidence=0.95),
        ]

    @pytest.fixture
    def sample_extractions(self):
        return [
            ExtractionResult(
                filename="a.pdf",
                predator_species="Species A",
                total_stomachs_examined=100,
                non_empty_stomachs=80,
            ),
            ExtractionResult(
                filename="c.pdf",
                predator_species="Species C",
                total_stomachs_examined=50,
                non_empty_stomachs=40,
            ),
        ]

    def test_creates_output_directory(self, output_dir):
        OutputManager(output_dir=str(output_dir))
        assert output_dir.exists()

    def test_add_classification(self, manager, sample_classifications):
        for c in sample_classifications:
            manager.add_classification(c)
        assert len(manager.classifications) == 3

    def test_add_extraction(self, manager, sample_extractions):
        for e in sample_extractions:
            manager.add_extraction(e)
        assert len(manager.extractions) == 2

    def test_export_classifications_json(self, manager, sample_classifications, output_dir):
        for c in sample_classifications:
            manager.add_classification(c)

        path = manager.export_classifications_json()
        assert path.exists()

        with open(path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert data["metadata"]["total_files"] == 3
        assert data["metadata"]["useful_count"] == 2
        assert len(data["results"]) == 3

    def test_export_classifications_csv(self, manager, sample_classifications, output_dir):
        for c in sample_classifications:
            manager.add_classification(c)

        path = manager.export_classifications_csv()
        assert path.exists()

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["filename"] == "a.pdf"

    def test_export_extractions_json(self, manager, sample_extractions, output_dir):
        for e in sample_extractions:
            manager.add_extraction(e)

        path = manager.export_extractions_json()
        assert path.exists()

        with open(path) as f:
            data = json.load(f)

        assert len(data["results"]) == 2
        assert data["results"][0]["predator_species"] == "Species A"

    def test_export_extractions_csv(self, manager, sample_extractions, output_dir):
        for e in sample_extractions:
            manager.add_extraction(e)

        path = manager.export_extractions_csv()
        assert path.exists()

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

    def test_export_all(self, manager, sample_classifications, sample_extractions, output_dir):
        for c in sample_classifications:
            manager.add_classification(c)
        for e in sample_extractions:
            manager.add_extraction(e)

        paths = manager.export_all()

        assert "classifications_json" in paths
        assert "classifications_csv" in paths
        assert "extractions_json" in paths
        assert "extractions_csv" in paths

        for path in paths.values():
            assert path.exists()

    def test_get_summary(self, manager, sample_classifications, sample_extractions):
        for c in sample_classifications:
            manager.add_classification(c)
        for e in sample_extractions:
            manager.add_extraction(e)

        summary = manager.get_summary()

        assert summary["total_classifications"] == 3
        assert summary["useful_count"] == 2
        assert summary["not_useful_count"] == 1
        assert summary["total_extractions"] == 2


class TestConvenienceFunctions:
    # Tests for standalone export functions.

    def test_export_to_json(self, tmp_path):
        results = [{"name": "test", "value": 123}]
        path = export_to_json(results, str(tmp_path / "test.json"))

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data == results

    def test_export_to_csv(self, tmp_path):
        results = [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
        ]
        path = export_to_csv(results, str(tmp_path / "test.csv"))

        assert path.exists()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["name"] == "a"

    def test_export_empty_csv(self, tmp_path):
        path = export_to_csv([], str(tmp_path / "empty.csv"))
        assert path.exists()
