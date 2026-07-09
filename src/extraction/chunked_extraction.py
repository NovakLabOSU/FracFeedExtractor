"""Chunked extraction - uses existing llm_client for extraction."""

import sys
import gc
import json
import xgboost as xgb
from pathlib import Path
from collections import Counter
from typing import Any, Optional

from src.config import DEFAULT_LLM_MODEL
from src.extraction.llm_client import extract_metrics_from_text
from src.extraction.models import PredatorDietMetrics
from src.classifier.pdf_classifier import load_classifier


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 300) -> list[str]:
    """Split text into overlapping chunks."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + chunk_size // 2:
                end = para_break
            else:
                sent_break = text.rfind('. ', start, end)
                if sent_break > start + chunk_size // 2:
                    end = sent_break + 1

        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def score_chunk(chunk: str, model: xgb.Booster, vectorizer: Any) -> float:
    """Score a chunk using XGBoost classifier."""
    X_vec = vectorizer.transform([chunk])
    dtest = xgb.DMatrix(X_vec)
    score = model.predict(dtest)[0]
    return float(score)


def _record_key(rec: dict[str, Any], idx: int | None = None) -> tuple:
    """Grouping key for a (species, survey) record: species × location × year_range.

    When all three fields are absent, falls back to a unique index-based key so
    distinct all-null records are never merged together.
    """
    key = (
        (rec.get("species_name") or "").lower().strip(),
        (rec.get("study_location") or "").lower().strip(),
        (rec.get("study_year_range") or "").strip(),
    )
    if key == ("", "", "") and idx is not None:
        return ("__null__", str(idx))
    return key


def _merge_record_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Majority-vote merge of records that share the same (species, location, year) key."""
    fields = list(PredatorDietMetrics.model_fields.keys())
    merged: dict[str, Any] = {}
    for field in fields:
        values = [r.get(field) for r in group if r.get(field) is not None]
        if not values:
            merged[field] = None
        else:
            merged[field] = Counter(values).most_common(1)[0][0]

    merged["source_pages"] = None
    return PredatorDietMetrics.model_validate(merged).model_dump()


def merge_results(chunk_results: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge multi-record extraction results from multiple chunks.

    Each element of chunk_results is the list of (species, survey) records
    returned by extract_metrics_from_text() for one chunk.  Records that share
    the same (species_name, study_location, study_year_range) key are merged
    by majority vote.  Records with distinct keys become separate output rows.

    Args:
        chunk_results: One list of record dicts per chunk.

    Returns:
        Deduplicated list of merged record dicts.
    """
    # Flatten all records from every chunk
    all_records = [rec for chunk in chunk_results for rec in chunk if rec is not None]

    if not all_records:
        return []

    # Group by natural (species, survey) key
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for i, rec in enumerate(all_records):
        key = _record_key(rec, i)
        groups.setdefault(key, []).append(rec)

    return [_merge_record_group(group) for group in groups.values()]


def extract_with_chunking(
    text: str,
    model_dir: str = "src/classifier/models",
    llm_model: str = DEFAULT_LLM_MODEL,
    num_ctx: int = 8192,
    top_n: int = 3,
    chunk_size: int = 3000,
    overlap: int = 300,
) -> list[dict[str, Any]]:
    """Main extraction with chunking pipeline.

    Returns a list of merged record dicts, one per unique (species, survey) pair
    found across all selected chunks.
    """

    print("  [CHUNK] Loading classifier...", file=sys.stderr)
    model, vectorizer, _encoder = load_classifier(model_dir)

    chunks = chunk_text(text, chunk_size, overlap)
    print(f"  [CHUNK] Split into {len(chunks)} chunks", file=sys.stderr)

    if not chunks:
        return []

    # Score all chunks
    scored = [(chunk, score_chunk(chunk, model, vectorizer)) for chunk in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Always include first chunk (abstract has species/location/date)
    first_chunk = chunks[0]
    first_score = next((s for c, s in scored if c == first_chunk), 0.5)

    # Get top scoring chunks that aren't the first chunk
    top_scored = [(c, s) for c, s in scored if c != first_chunk][: top_n - 1]

    # Combine: first chunk + top scoring chunks
    top_chunks = [(first_chunk, first_score)] + top_scored

    print(f"  [CHUNK] Using {len(top_chunks)} chunks (first + top {len(top_scored)} scored)", file=sys.stderr)
    print(f"  [CHUNK] Scores: {[round(s, 3) for _, s in top_chunks]}", file=sys.stderr)

    # Each element is a list of record dicts from one chunk
    chunk_results: list[list[dict[str, Any]]] = []
    successful_chunks = 0
    for i, (chunk, score) in enumerate(top_chunks):
        print(f"  [CHUNK] Extracting chunk {i + 1}/{len(top_chunks)}...", file=sys.stderr)

        try:
            records = extract_metrics_from_text(
                text=chunk,
                model=llm_model,
                num_ctx=num_ctx,
            )
            record_dicts = [r.model_dump() for r in records]
            chunk_results.append(record_dicts)
            successful_chunks += 1
            for rd in record_dicts:
                print(
                    f"    species={rd.get('species_name')}, "
                    f"location={rd.get('study_location')}, "
                    f"sample={rd.get('num_sampled')}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"    [ERROR] {e}", file=sys.stderr)

        gc.collect()

    if not chunk_results:
        raise RuntimeError(f"All {len(top_chunks)} LLM extraction attempts failed")

    merged = merge_results(chunk_results)
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", help="Path to text file")
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--top-chunks", type=int, default=3)
    parser.add_argument("--llm-model", default="cniongolo/biomistral")
    args = parser.parse_args()

    text_path = Path(args.text_file)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Processing {text_path.name}...")

    records = extract_with_chunking(text, top_n=args.top_chunks, llm_model=args.llm_model)

    result = {"source_file": text_path.name, "records": records}

    output_path = Path(args.output_dir) / f"{text_path.stem}_chunked_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_path}")
    print(json.dumps(records, indent=2))
