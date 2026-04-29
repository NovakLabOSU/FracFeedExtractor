"""Chunked extraction - uses existing llm_client for extraction."""

import sys
import gc
import json
import xgboost as xgb
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm.llm_client import extract_metrics_from_text
from src.model.pdf_classifier import load_classifier


def chunk_text(text, chunk_size=3000, overlap=300):
    """Split text into overlapping chunks."""
    chunks = []
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


def score_chunk(chunk, model, vectorizer):
    """Score a chunk using XGBoost classifier."""
    X_vec = vectorizer.transform([chunk])
    dtest = xgb.DMatrix(X_vec)
    score = model.predict(dtest)[0]
    return float(score)


def merge_results(results):
    """Merge extraction results from multiple chunks using voting."""
    results = [r for r in results if r is not None]

    if not results:
        return {}

    merged = {}
    fields = ['species_name', 'study_location', 'study_date', 'num_empty_stomachs', 'num_nonempty_stomachs', 'sample_size']

    for field in fields:
        values = [r.get(field) for r in results if r.get(field) is not None]

        if not values:
            merged[field] = None
            merged[f"{field}_confidence"] = 0.0
        else:
            counter = Counter(values)
            most_common = counter.most_common(1)[0]
            merged[field] = most_common[0]
            merged[f"{field}_confidence"] = round(most_common[1] / len(values), 2)

    nonempty = merged.get("num_nonempty_stomachs")
    sample = merged.get("sample_size")
    if nonempty and sample and sample > 0:
        merged["fraction_feeding"] = round(nonempty / sample, 4)
    else:
        merged["fraction_feeding"] = None

    return merged


def extract_with_chunking(
    text,
    model_dir="src/model/models",
    llm_model="qwen2.5:7b",  # Changed from biomistral
    num_ctx=8192,
    top_n=3,
    chunk_size=3000,
    overlap=300,
):
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
            print(f"    species={result_dict.get('species_name')}, location={result_dict.get('study_location')}, sample={result_dict.get('sample_size')}", file=sys.stderr)
        except Exception as e:
            print(f"    [ERROR] {e}", file=sys.stderr)

        gc.collect()

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
