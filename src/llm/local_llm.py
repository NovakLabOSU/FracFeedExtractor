"""LLM-based metric extraction from preprocessed text files.

Usage:
    python extract_metrics.py path/to/text_file.txt
    python extract_metrics.py path/to/text_file.txt --model llama3.1:8b
    python extract_metrics.py path/to/text_file.txt --output-dir results/

This script uses Ollama to extract structured data from preprocessed predator diet 
surveys, including species name, study date, location, and stomach content data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ollama import chat
from pydantic import BaseModel, Field


class PredatorDietMetrics(BaseModel):
    """Structured schema for extracted predator diet survey metrics."""
    
    species_name: Optional[str] = Field(
        None,
        description="Scientific name of the predator species studied"
    )
    study_location: Optional[str] = Field(
        None,
        description="Geographic location where the study was conducted"
    )
    study_date: Optional[str] = Field(
        None,
        description="Year or date range when the study was conducted"
    )
    num_empty_stomachs: Optional[int] = Field(
        None,
        description="Number of predators with empty stomachs"
    )
    num_nonempty_stomachs: Optional[int] = Field(
        None,
        description="Number of predators with non-empty stomachs"
    )
    sample_size: Optional[int] = Field(
        None,
        description="Total number of predators surveyed"
    )


def extract_metrics_from_text(text: str, model: str = "llama3.1:8b") -> PredatorDietMetrics:
    """Extract structured metrics from text using Ollama.
    
    Args:
        text: Preprocessed text content from a scientific publication
        model: Name of the Ollama model to use
        
    Returns:
        PredatorDietMetrics object with extracted data
    """
    prompt = f"""You are a scientific data extraction assistant specializing in predator diet surveys.

Extract specific metrics from the text below. Focus on stomach content data where:
- EMPTY stomachs = no food/prey
- NON-EMPTY stomachs = contained food/prey
- SAMPLE SIZE = total number of predators examined

KEY INFORMATION TO FIND:
- Species names are in Latin format (Genus species)
- Look in tables, methods, and results sections
- Empty stomachs: "empty", "vacant", "no prey"
- Non-empty stomachs: "with prey", "fed", "containing food"

EXTRACT:
- species_name: Scientific name of PRIMARY predator studied (not prey)
- study_location: Geographic location of sampling
- study_date: Year or date range of collection
- num_empty_stomachs: Number with empty stomachs
- num_nonempty_stomachs: Number with food in stomachs
- sample_size: Total number examined


TEXT:
{text}
"""
    # Ollama call with structured schema output
    response = chat(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model=model,
        format=PredatorDietMetrics.model_json_schema(),
    )
    
    metrics = PredatorDietMetrics.model_validate_json(response.message.content)
    return metrics


def validate_and_calculate(metrics: dict) -> dict:
    """Validate extracted metrics and calculate derived values.
    
    Args:
        metrics: Dictionary of extracted metrics
        
    Returns:
        Dictionary with validated metrics and calculated fraction_feeding
    """
    empty = metrics.get("num_empty_stomachs")
    nonempty = metrics.get("num_nonempty_stomachs")
    sample = metrics.get("sample_size")
    
    # Validate and fix sample size if needed
    if empty is not None and nonempty is not None:
        calculated_sample = empty + nonempty
        if sample is None:
            metrics["sample_size"] = calculated_sample
            sample = calculated_sample
        elif sample != calculated_sample:
            # LLM made an error, use calculated value
            metrics["sample_size"] = calculated_sample
            sample = calculated_sample
    
    # Calculate fraction of feeding predators
    fraction_feeding = None
    if nonempty is not None and sample is not None and sample > 0:
        fraction_feeding = round(nonempty / sample, 4)
    
    metrics["fraction_feeding"] = fraction_feeding
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Extract predator diet metrics from preprocessed text using LLM"
    )
    parser.add_argument(
        "text_file",
        type=str,
        help="Path to the preprocessed text file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model to use (default: llama3.1:8b)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory for JSON results (default: data/results)"
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
    
    # Validate and calculate derived metrics
    metrics_dict = metrics.model_dump()
    metrics_dict = validate_and_calculate(metrics_dict)
    
    # Prepare output
    result = {
        "source_file": text_path.name,
        "metrics": metrics_dict
    }
    
    # Generate output filename: input_name_results.json
    output_filename = text_path.stem + "_results.json"
    output_path = Path(args.output_dir) / output_filename
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()