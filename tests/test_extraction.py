"""Test script to compare original vs improved LLM extraction.

Usage:
    python test_comparison.py path/to/text_file.txt path/to/original.pdf

This script runs both the original and improved extraction methods and
compares the results side-by-side.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def load_json_results(filepath: str) -> Dict:
    """Load JSON results from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load {filepath}: {e}")
        return {}


def count_null_fields(metrics: Dict) -> int:
    """Count how many fields are null."""
    null_count = 0
    fields_to_check = ['species_name', 'study_location', 'study_date', 'num_empty_stomachs', 'num_nonempty_stomachs', 'sample_size']
    for field in fields_to_check:
        if metrics.get(field) is None:
            null_count += 1
    return null_count


def print_comparison_table(original_metrics: Dict, improved_metrics: Dict):
    """Print a side-by-side comparison of extraction results."""

    print("\n" + "=" * 80)
    print("EXTRACTION COMPARISON: Original vs Improved")
    print("=" * 80)

    fields = [
        ('species_name', 'Species Name'),
        ('study_location', 'Study Location'),
        ('study_date', 'Study Date'),
        ('num_empty_stomachs', 'Empty Stomachs'),
        ('num_nonempty_stomachs', 'Non-Empty Stomachs'),
        ('sample_size', 'Sample Size'),
        ('fraction_feeding', 'Fraction Feeding'),
    ]

    print(f"\n{'Field':<25} {'Original':<30} {'Improved':<30}")
    print("-" * 80)

    for field_key, field_name in fields:
        orig_val = original_metrics.get(field_key)
        impr_val = improved_metrics.get(field_key)

        orig_str = str(orig_val) if orig_val is not None else "NULL"
        impr_str = str(impr_val) if impr_val is not None else "NULL"

        # Highlight improvements
        marker = ""
        if orig_val is None and impr_val is not None:
            marker = " ✓ IMPROVED"
        elif orig_val != impr_val and impr_val is not None:
            marker = " ✓ CHANGED"

        print(f"{field_name:<25} {orig_str:<30} {impr_str:<30}{marker}")

    print("-" * 80)

    # Summary statistics
    orig_null = count_null_fields(original_metrics)
    impr_null = count_null_fields(improved_metrics)

    print(f"\nNull Fields:              {orig_null}/6                        {impr_null}/6")
    print(f"Completeness:             {((6 - orig_null) / 6) * 100:.1f}%                       {((6 - impr_null) / 6) * 100:.1f}%")

    if impr_null < orig_null:
        print(f"\n✓ IMPROVEMENT: {orig_null - impr_null} fewer null fields!")
    elif impr_null > orig_null:
        print(f"\n✗ REGRESSION: {impr_null - orig_null} more null fields")
    else:
        print("\n→ No change in null field count")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare original vs improved LLM extraction")
    parser.add_argument("text_file", type=str, help="Path to preprocessed text file")
    parser.add_argument("pdf_file", type=str, help="Path to original PDF file")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Model to use for both extractions")
    parser.add_argument("--output-dir", type=str, default="data/test_comparison", help="Output directory for comparison results")

    args = parser.parse_args()

    text_path = Path(args.text_file)
    pdf_path = Path(args.pdf_file)

    if not text_path.exists():
        print(f"[ERROR] Text file not found: {text_path}")
        sys.exit(1)

    if not pdf_path.exists():
        print(f"[ERROR] PDF file not found: {pdf_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("RUNNING EXTRACTION COMPARISON TEST")
    print("=" * 80)
    print(f"Text file: {text_path.name}")
    print(f"PDF file: {pdf_path.name}")
    print(f"Model: {args.model}")
    print("=" * 80 + "\n")

    # Run original extraction
    print("[1/2] Running ORIGINAL extraction...")
    print("-" * 80)

    import subprocess

    original_output = output_dir / f"{text_path.stem}_original_results.json"

    try:
        result = subprocess.run([sys.executable, "local_llm.py", str(text_path), "--model", args.model, "--output-dir", str(output_dir)], capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print("[ERROR] Original extraction failed:")
            print(result.stderr)
            sys.exit(1)

        # Rename output to distinguish it
        default_output = output_dir / f"{text_path.stem}_results.json"
        if default_output.exists():
            default_output.rename(original_output)

    except subprocess.TimeoutExpired:
        print("[ERROR] Original extraction timed out")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Original extraction failed: {e}")
        sys.exit(1)

    print("✓ Original extraction complete\n")

    # Run improved extraction
    print("[2/2] Running IMPROVED extraction...")
    print("-" * 80)

    improved_output = output_dir / f"{text_path.stem}_improved_results.json"

    try:
        result = subprocess.run(
            [sys.executable, "local_llm_improved.py", str(text_path), "--pdf", str(pdf_path), "--model", args.model, "--output-dir", str(output_dir)], capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            print("[ERROR] Improved extraction failed:")
            print(result.stderr)
            sys.exit(1)

        # Rename output to distinguish it
        default_output = output_dir / f"{text_path.stem}_results.json"
        if default_output.exists():
            default_output.rename(improved_output)

    except subprocess.TimeoutExpired:
        print("[ERROR] Improved extraction timed out")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Improved extraction failed: {e}")
        sys.exit(1)

    print("✓ Improved extraction complete\n")

    # Load and compare results
    original_data = load_json_results(str(original_output))
    improved_data = load_json_results(str(improved_output))

    if not original_data or not improved_data:
        print("[ERROR] Could not load results for comparison")
        sys.exit(1)

    original_metrics = original_data.get('metrics', {})
    improved_metrics = improved_data.get('metrics', {})

    # Print comparison
    print_comparison_table(original_metrics, improved_metrics)

    # Save comparison report
    comparison_report = {
        'text_file': text_path.name,
        'pdf_file': pdf_path.name,
        'model': args.model,
        'original': {'metrics': original_metrics, 'null_count': count_null_fields(original_metrics), 'completeness': f"{((6 - count_null_fields(original_metrics)) / 6) * 100:.1f}%"},
        'improved': {'metrics': improved_metrics, 'null_count': count_null_fields(improved_metrics), 'completeness': f"{((6 - count_null_fields(improved_metrics)) / 6) * 100:.1f}%"},
        'improvement': {
            'fields_gained': count_null_fields(original_metrics) - count_null_fields(improved_metrics),
            'is_better': count_null_fields(improved_metrics) < count_null_fields(original_metrics),
        },
    }

    report_path = output_dir / f"{text_path.stem}_comparison_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, indent=2)

    print(f"Detailed comparison report saved to: {report_path}\n")


if __name__ == "__main__":
    main()
