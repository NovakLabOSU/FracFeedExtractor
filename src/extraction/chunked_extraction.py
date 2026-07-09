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


def merge_results(results: list[Optional[dict[str, Any]]]) -> dict[str, Any]:
    """Merge extraction results from multiple chunks using voting."""
    results = [r for r in results if r is not None]

    if not results:
        return {}

    merged: dict[str, Any] = {}
    fields = list(PredatorDietMetrics.model_fields.keys())

    for field in fields:
        values = [r.get(field) for r in results if r.get(field) is not None]

        if not values:
            merged[field] = None
        else:
            counter = Counter(values)
            merged[field] = counter.most_common(1)[0][0]

    nonempty = merged.get("num_nonempty")
    sample = merged.get("num_sampled")
    if nonempty is not None and sample and sample > 0:
        merged["fraction_feeding"] = round(nonempty / sample, 4)
    else:
        merged["fraction_feeding"] = None

    merged["source_pages"] = None

    return merged


def extract_with_chunking(
    text: str,
    model_dir: str = "src/classifier/models",
    llm_model: str = DEFAULT_LLM_MODEL,
    num_ctx: int = 8192,
    top_n: int = 3,
    chunk_size: int = 3000,
    overlap: int = 300,
) -> dict[str, Any]:
    """Main extraction with chunking pipeline."""

    print("  [CHUNK] Loading classifier...", file=sys.stderr)
    model, vectorizer, _encoder = load_classifier(model_dir)

    chunks = chunk_text(text, chunk_size, overlap)
    print(f"  [CHUNK] Split into {len(chunks)} chunks", file=sys.stderr)

    if not chunks:
        return {}

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

    results = []
    for i, (chunk, score) in enumerate(top_chunks):
        print(f"  [CHUNK] Extracting chunk {i + 1}/{len(top_chunks)}...", file=sys.stderr)

        try:
            metrics = extract_metrics_from_text(
                text=chunk,
                model=llm_model,
                num_ctx=num_ctx,
            )
            result_dict = metrics.model_dump()
            results.append(result_dict)
            print(f"    species={result_dict.get('species_name')}, location={result_dict.get('study_location')}, sample={result_dict.get('num_sampled')}", file=sys.stderr)
        except Exception as e:
            print(f"    [ERROR] {e}", file=sys.stderr)

        gc.collect()

    if not results:
        raise RuntimeError(f"All {len(top_chunks)} LLM extraction attempts failed")

    merged = merge_results(results)
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

    metrics = extract_with_chunking(text, top_n=args.top_chunks, llm_model=args.llm_model)

    result = {"source_file": text_path.name, "metrics": metrics}

    output_path = Path(args.output_dir) / f"{text_path.stem}_chunked_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_path}")
    print(json.dumps(metrics, indent=2))
