"""Fixed LLM extraction with anti-hallucination measures.

This version:
1. Fixes section extraction regex to actually capture content
2. Searches full text when sections fail
3. Adds explicit anti-hallucination instructions
4. Validates LLM output against input text
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List

from ollama import chat
from pydantic import BaseModel, Field


class PredatorDietMetrics(BaseModel):
    """Structured schema for extracted predator diet survey metrics."""

    species_name: Optional[str] = Field(None, description="Scientific name of the predator species studied")
    study_location: Optional[str] = Field(None, description="Geographic location where the study was conducted")
    study_date: Optional[str] = Field(None, description="Year or date range when the study was conducted")
    num_empty_stomachs: Optional[int] = Field(None, description="Number of predators with empty stomachs")
    num_nonempty_stomachs: Optional[int] = Field(None, description="Number of predators with non-empty stomachs")
    sample_size: Optional[int] = Field(None, description="Total number of predators surveyed")


def extract_stomach_counts_from_text(text: str) -> Dict[str, Optional[int]]:
    """
    Try to extract stomach counts using regex patterns as backup.
    Only applies if not found from tables.
    """
    result = {"num_empty_stomachs": None, "num_nonempty_stomachs": None, "sample_size": None}

    # Pattern 1: "X stomachs were found empty"
    empty_pattern = r'(\d+)\s+stomachs?\s+(?:were\s+)?(?:found\s+)?empty'
    empty_match = re.search(empty_pattern, text, re.IGNORECASE)
    if empty_match:
        result["num_empty_stomachs"] = int(empty_match.group(1))

    # Pattern 2: "total of X stomachs" or "X stomachs"
    total_pattern = r'(?:total\s+of\s+|^|\.\s+)(\d+)\s+stomachs'
    total_match = re.search(total_pattern, text, re.IGNORECASE | re.MULTILINE)
    if total_match:
        result["sample_size"] = int(total_match.group(1))

    # Calculate non-empty if we have both
    if result["num_empty_stomachs"] and result["sample_size"]:
        result["num_nonempty_stomachs"] = result["sample_size"] - result["num_empty_stomachs"]

    return result

def format_tables_for_llm(tables: List[Dict]) -> str:
    """Format extracted tables into readable text for LLM."""
    if not tables:
        return ""
    
    formatted = []
    for table in tables:
        if "error" in table:
            continue
            
        cells = table.get("cells", [])
        if not cells:
            continue
        
        # Format as markdown-style table
        table_text = f"\n--- {table['table_id']} (Page {table['page_number']}) ---\n"
        
        for row in cells:
            # Clean and join cells with | separator
            row_text = " | ".join(str(cell).strip() if cell else "" for cell in row)
            table_text += row_text + "\n"
        
        formatted.append(table_text)
    
    return "\n".join(formatted)

def extract_metadata_with_search(text: str, tables_text: str, model: str) -> Dict[str, Optional[str]]:
    """Extract metadata in full text."""

    # Get first 12000 chars which should include title, abstract, intro, methods
    context = text[:12000]

    # Add tables if available
    if tables_text:
        context = f"TABLES:\n{tables_text}\n\nTEXT:\n{context}"

    prompt = f"""Extract metadata from this scientific paper about predator diet.

**CRITICAL: You MUST extract information ONLY from the text provided below. If you cannot find something in the text, return null.**

Find these 3 fields:

1. species_name: The Latin name (Genus species) of the PREDATOR being studied
   - Look in the title or first paragraph
   - Format: "Genus species" (e.g., "Martes foina", not "stone marten")
   
2. study_location: Where specimens were collected
   - Country, region, or coordinates
   - Example: "Central Greece" or "38°44'N, 22°02'E"
   
3. study_date: When specimens were collected  
   - Year or range: "2005" or "2003-2006"
   - Look for phrases like "collected between", "during", "from...to"

TEXT TO ANALYZE:
{context}

Remember: Extract ONLY from this text. If not clearly stated, return null.
"""

    try:
        response = chat(
            messages=[{'role': 'user', 'content': prompt}],
            model=model,
            format={
                "type": "object",
                "properties": {"species_name": {"type": ["string", "null"]}, "study_location": {"type": ["string", "null"]}, "study_date": {"type": ["string", "null"]}},
                "required": ["species_name", "study_location", "study_date"],
            },
        )

        return json.loads(response.message.content)
    except Exception as e:
        print(f"[ERROR] Metadata extraction failed: {e}", file=sys.stderr)
        return {"species_name": None, "study_location": None, "study_date": None}


def extract_stomach_data_with_search(text: str, tables_text: str, model: str) -> Dict[str, Optional[int]]:
    """Extract stomach content counts with targeted search."""

    # First try regex extraction as truth
    regex_result = extract_stomach_counts_from_text(text)
    print(f"[INFO] Regex found: {regex_result}", file=sys.stderr)

    # Search for sections with stomach/empty keywords
    stomach_pattern = r'.{0,800}(?:stomachs?|empty|sample\s+size|n\s*=).{0,800}'
    matches = re.findall(stomach_pattern, text, re.IGNORECASE)
    context = '\n\n---\n\n'.join(matches[:15])  # Up to 15 relevant passages

    if not context:
        context = text[:15000]  # Fallback to first part

    # Prepend tables if available
    if tables_text:
        context = f"TABLES:\n{tables_text}\n\nTEXT:\n{context}"

    prompt = f"""Extract stomach content counts from this predator diet study.

**CRITICAL: Extract ONLY numbers that appear in the text below. DO NOT invent numbers.**

Find these 3 numbers:

1. num_empty_stomachs: How many predators had EMPTY stomachs
   - Keywords: "empty", "vacant", "no prey", "unfed"
   
2. num_nonempty_stomachs: How many had NON-EMPTY stomachs (contained food)
   - May need to calculate: total - empty = non-empty
   
3. sample_size: Total number of predators examined
   - Keywords: "total", "n =", "sample size"

**VALIDATION**: 
- empty + non-empty should equal sample_size
- If text says "14 stomachs were found empty" and "106 stomachs", then:
  num_empty_stomachs = 14
  num_nonempty_stomachs = 92 (calculated: 106 - 14)
  sample_size = 106

TEXT TO ANALYZE:
{context}

Extract the numbers. If a number is not stated or calculable, return null for that field.
"""

    try:
        response = chat(
            messages=[{'role': 'user', 'content': prompt}],
            model=model,
            format={
                "type": "object",
                "properties": {"num_empty_stomachs": {"type": ["integer", "null"]}, "num_nonempty_stomachs": {"type": ["integer", "null"]}, "sample_size": {"type": ["integer", "null"]}},
                "required": ["num_empty_stomachs", "num_nonempty_stomachs", "sample_size"],
            },
        )

        llm_result = json.loads(response.message.content)

        # Validate: prefer regex if it found values and they differ from LLM
        if regex_result["num_empty_stomachs"] and llm_result.get("num_empty_stomachs") != regex_result["num_empty_stomachs"]:
            llm_result["num_empty_stomachs"] = regex_result["num_empty_stomachs"]

        if regex_result["sample_size"] and llm_result.get("sample_size") != regex_result["sample_size"]:
            llm_result["sample_size"] = regex_result["sample_size"]

        return llm_result

    except Exception as e:
        print(f"[ERROR] Stomach data extraction failed: {e}", file=sys.stderr)
        # Fall back to regex result if LLM fails
        return regex_result if any(regex_result.values()) else {"num_empty_stomachs": None, "num_nonempty_stomachs": None, "sample_size": None}


def validate_and_calculate(metrics: dict) -> dict:
    """Validate extracted metrics and calculate derived values."""
    empty = metrics.get("num_empty_stomachs")
    nonempty = metrics.get("num_nonempty_stomachs")
    sample = metrics.get("sample_size")

    # Validate and fix sample size if needed
    if empty is not None and nonempty is not None:
        calculated_sample = empty + nonempty
        if sample is None:
            print(f"[INFO] Calculated sample_size: {calculated_sample}", file=sys.stderr)
            metrics["sample_size"] = calculated_sample
            sample = calculated_sample
        elif sample != calculated_sample:
            print(f"[WARN] Sample size mismatch: stated={sample}, calculated={calculated_sample}. Using calculated.", file=sys.stderr)
            metrics["sample_size"] = calculated_sample
            sample = calculated_sample

    # Calculate fraction of feeding predators
    fraction_feeding = None
    if nonempty is not None and sample is not None and sample > 0:
        fraction_feeding = round(nonempty / sample, 4)

    metrics["fraction_feeding"] = fraction_feeding

    # Report completeness
    null_count = sum(1 for k, v in metrics.items() if v is None and k != "fraction_feeding")
    total_fields = 6
    print(f"[INFO] Completeness: {total_fields - null_count}/{total_fields} fields filled", file=sys.stderr)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Extract predator diet metrics from PDF (fixed version)")
    parser.add_argument("pdf", type=str, help="Path to the PDF file")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama model to use")
    parser.add_argument("--output-dir", type=str, default="data/results", help="Output directory")

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Extract text from PDF
    print(f"[INFO] Extracting text from: {pdf_path.name}", file=sys.stderr)
    try:
        src_path = Path(__file__).resolve().parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from preprocessing.pdf_text_extraction import extract_text_from_pdf, extract_tables_from_pdf

        text = extract_text_from_pdf(str(pdf_path))

        if not text.strip():
            print("[ERROR] No text extracted", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] Extracted {len(text)} characters", file=sys.stderr)

        # Extract tables
        tables = extract_tables_from_pdf(str(pdf_path))
        print(f"[INFO] Extracted {len(tables)} tables", file=sys.stderr)
        
        # Format tables for LLM
        tables_text = format_tables_for_llm(tables)

    except Exception as e:
        print(f"[ERROR] Text extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract metrics
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Extracting from: {pdf_path.name}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    print("[1/2] Extracting metadata...", file=sys.stderr)
    metadata = extract_metadata_with_search(text, tables_text, args.model)
    print(f"      Species: {metadata.get('species_name')}", file=sys.stderr)
    print(f"      Location: {metadata.get('study_location')}", file=sys.stderr)
    print(f"      Date: {metadata.get('study_date')}", file=sys.stderr)

    print("\n[2/2] Extracting stomach data...", file=sys.stderr)
    stomach_data = extract_stomach_data_with_search(text, tables_text, args.model)
    print(f"      Empty: {stomach_data.get('num_empty_stomachs')}", file=sys.stderr)
    print(f"      Non-empty: {stomach_data.get('num_nonempty_stomachs')}", file=sys.stderr)
    print(f"      Total: {stomach_data.get('sample_size')}", file=sys.stderr)

    # Combine and validate
    metrics_dict = {**metadata, **stomach_data}
    metrics_dict = validate_and_calculate(metrics_dict)

    # Save results
    result = {
        "source_file": pdf_path.name,
        "model_used": args.model,
        "metrics": metrics_dict,
        "tables_found": len(tables)
    }

    output_path = Path(args.output_dir) / f"{pdf_path.stem}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Results saved to: {output_path}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
