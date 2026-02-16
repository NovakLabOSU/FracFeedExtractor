"""BioMistral extraction - uses model trained on PubMed papers."""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional

from ollama import chat
from pydantic import BaseModel, Field, model_validator


class PredatorDietMetrics(BaseModel):
    """Schema for extracted diet survey metrics."""

    species_name: Optional[str] = Field(None)
    study_location: Optional[str] = Field(None)
    study_date: Optional[str] = Field(None)
    num_empty_stomachs: Optional[int] = Field(None)
    num_nonempty_stomachs: Optional[int] = Field(None)
    sample_size: Optional[int] = Field(None)

    @model_validator(mode="after")
    def validate_stomach_counts(self):
        """Make sure stomach counts add up."""
        empty = self.num_empty_stomachs
        nonempty = self.num_nonempty_stomachs
        sample = self.sample_size

        if empty is not None and empty < 0:
            self.num_empty_stomachs = None
        if nonempty is not None and nonempty < 0:
            self.num_nonempty_stomachs = None
        if sample is not None and sample < 0:
            self.sample_size = None

        # Auto-calculate sample size if we have both counts
        if self.num_empty_stomachs is not None and self.num_nonempty_stomachs is not None:
            self.sample_size = self.num_empty_stomachs + self.num_nonempty_stomachs

        # Sanity check
        if self.sample_size is not None:
            if self.num_empty_stomachs is not None and self.num_empty_stomachs > self.sample_size:
                self.num_empty_stomachs = None
            if self.num_nonempty_stomachs is not None and self.num_nonempty_stomachs > self.sample_size:
                self.num_nonempty_stomachs = None

        return self


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


def extract_metrics_from_text(text: str) -> PredatorDietMetrics:
    """Send text to BioMistral and get structured output."""
    
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
        messages=[{"role": "user", "content": prompt}],
        model="cniongolo/biomistral",
        format=PredatorDietMetrics.model_json_schema(),
    )

    return PredatorDietMetrics.model_validate_json(response.message.content)  # parse response


def calculate_fraction_feeding(metrics_dict):
    """Add fraction_feeding to the results."""
    nonempty = metrics_dict.get("num_nonempty_stomachs")
    sample = metrics_dict.get("sample_size")

    if nonempty and sample and sample > 0:
        metrics_dict["fraction_feeding"] = round(nonempty / sample, 4)
    else:
        metrics_dict["fraction_feeding"] = None
    return metrics_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", help="Path to preprocessed text file")
    parser.add_argument("--output-dir", default="data/results")
    args = parser.parse_args()

    text_path = Path(args.text_file)
    if not text_path.exists():
        print(f"File not found: {text_path}")
        sys.exit(1)

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Extracting from {text_path.name} using BioMistral...")
    
    metrics = extract_metrics_from_text(text)
    metrics_dict = calculate_fraction_feeding(metrics.model_dump())

    result = {"source_file": text_path.name, "metrics": metrics_dict}

    output_path = Path(args.output_dir) / f"{text_path.stem}_biomistral_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()