"""LLM-based metric extraction from preprocessed text files.

Usage:
    python local_llm.py path/to/text_file.txt
    python local_llm.py path/to/text_file.txt --model llama3.1:8b
    python local_llm.py path/to/text_file.txt --output-dir results/

This script uses Ollama to extract structured data from preprocessed predator diet
surveys, including species name, study date, location, and stomach content data.
Uses few-shot prompting for improved accuracy and Pydantic validation to catch
bad or inconsistent extractions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ollama import chat
from pydantic import BaseModel, Field, model_validator


class PredatorDietMetrics(BaseModel):
    """Structured schema for extracted predator diet survey metrics."""

    species_name: Optional[str] = Field(None, description="Scientific name of the predator species studied")
    study_location: Optional[str] = Field(None, description="Geographic location where the study was conducted")
    study_date: Optional[str] = Field(None, description="Year or date range when the study was conducted")
    num_empty_stomachs: Optional[int] = Field(None, description="Number of predators with empty stomachs")
    num_nonempty_stomachs: Optional[int] = Field(None, description="Number of predators with non-empty stomachs")
    sample_size: Optional[int] = Field(None, description="Total number of predators surveyed")

    @model_validator(mode="after")
    def validate_stomach_counts(self):
        """Ensure stomach counts are consistent and non-negative."""
        empty = self.num_empty_stomachs
        nonempty = self.num_nonempty_stomachs
        sample = self.sample_size

        # Check for negative values
        if empty is not None and empty < 0:
            self.num_empty_stomachs = None
        if nonempty is not None and nonempty < 0:
            self.num_nonempty_stomachs = None
        if sample is not None and sample < 0:
            self.sample_size = None

        # Fix sample size if it doesn't match the sum
        if self.num_empty_stomachs is not None and self.num_nonempty_stomachs is not None:
            calculated = self.num_empty_stomachs + self.num_nonempty_stomachs
            if self.sample_size is None:
                self.sample_size = calculated
            elif self.sample_size != calculated:
                self.sample_size = calculated

        # Check that parts don't exceed the whole
        if self.sample_size is not None:
            if self.num_empty_stomachs is not None and self.num_empty_stomachs > self.sample_size:
                self.num_empty_stomachs = None
            if self.num_nonempty_stomachs is not None and self.num_nonempty_stomachs > self.sample_size:
                self.num_nonempty_stomachs = None

        return self


# Few-shot examples that teach the LLM what good extraction looks like
FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Text: "A total of 342 Atlantic cod (Gadus morhua) were collected from the North Sea
between March and October 2019. Stomach contents were analyzed and 89 individuals
had empty stomachs while 253 contained prey items."

Extracted:
{
  "species_name": "Gadus morhua",
  "study_location": "North Sea",
  "study_date": "2019",
  "num_empty_stomachs": 89,
  "num_nonempty_stomachs": 253,
  "sample_size": 342
}

EXAMPLE 2:
Text: "Between 1984 and 1986, we examined stomach contents of 144 gentoo penguins
(Pygoscelis papua) collected at Marion Island in the sub-Antarctic. Twelve stomachs
were empty."

Extracted:
{
  "species_name": "Pygoscelis papua",
  "study_location": "Marion Island, sub-Antarctic",
  "study_date": "1984-1986",
  "num_empty_stomachs": 12,
  "num_nonempty_stomachs": 132,
  "sample_size": 144
}

EXAMPLE 3:
Text: "Diet composition of largemouth bass (Micropterus salmoides) was studied in
Lake Erie. Fish were sampled monthly from June to September 2015. Of 200 bass
examined, 45 had empty alimentary tracts and 155 had consumed prey."

Extracted:
{
  "species_name": "Micropterus salmoides",
  "study_location": "Lake Erie",
  "study_date": "2015",
  "num_empty_stomachs": 45,
  "num_nonempty_stomachs": 155,
  "sample_size": 200
}
"""


def extract_metrics_from_text(text: str, model: str = "llama3.1:8b") -> PredatorDietMetrics:
    """Extract structured metrics from text using Ollama with few-shot prompting.

    Args:
        text: Preprocessed text content from a scientific publication
        model: Name of the Ollama model to use

    Returns:
        PredatorDietMetrics object with extracted and validated data
    """
    prompt = f"""You are a scientific data extraction assistant specializing in predator diet surveys.

Your task is to extract specific metrics from a scientific paper. Study the examples below carefully, then extract from the actual text.

{FEW_SHOT_EXAMPLES}

RULES:
- species_name: Extract the scientific name (Genus species) of the PRIMARY predator, not prey
- study_location: Geographic location where sampling occurred
- study_date: Year or date range of specimen collection
- num_empty_stomachs: Count of predators with empty stomachs (look for "empty", "vacant", "no prey", "vacuity")
- num_nonempty_stomachs: Count of predators with food (look for "with prey", "fed", "containing food", "non-empty")
- sample_size: Total number of predators examined (should equal empty + non-empty)
- If a value is not clearly stated in the text, use null
- Do NOT guess or infer values that are not in the text
- Look carefully in tables, methods, and results sections

NOW EXTRACT FROM THIS TEXT:
{text}
"""
    response = chat(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        format=PredatorDietMetrics.model_json_schema(),
    )

    metrics = PredatorDietMetrics.model_validate_json(response.message.content)
    return metrics


def calculate_fraction_feeding(metrics_dict: dict) -> dict:
    """Calculate the fraction of feeding predators from validated metrics.

    Args:
        metrics_dict: Dictionary of validated metrics from Pydantic model

    Returns:
        Dictionary with added fraction_feeding value
    """
    nonempty = metrics_dict.get("num_nonempty_stomachs")
    sample = metrics_dict.get("sample_size")

    fraction_feeding = None
    if nonempty is not None and sample is not None and sample > 0:
        fraction_feeding = round(nonempty / sample, 4)

    metrics_dict["fraction_feeding"] = fraction_feeding
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Extract predator diet metrics from preprocessed text using LLM")
    parser.add_argument("text_file", type=str, help="Path to the preprocessed text file")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model to use (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory for JSON results (default: data/results)",
    )

    args = parser.parse_args()

    # Load text file
    text_path = Path(args.text_file)
    if not text_path.exists():
        print(f"[ERROR] File not found: {text_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract metrics
    print(f"Extracting metrics from {text_path.name}...", file=sys.stderr)
    try:
        metrics = extract_metrics_from_text(text, model=args.model)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate derived metrics
    metrics_dict = metrics.model_dump()
    metrics_dict = calculate_fraction_feeding(metrics_dict)

    # Prepare output
    result = {"source_file": text_path.name, "metrics": metrics_dict}

    # Generate output filename
    output_filename = text_path.stem + "_results.json"
    output_path = Path(args.output_dir) / output_filename

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
