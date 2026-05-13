"""Chunked extraction pipeline — split papers, score chunks, extract from top-N, merge.

Instead of sending one big trimmed blob to the LLM, this module:
  1. Splits the document into overlapping character-level chunks
  2. Scores each chunk with the XGBoost classifier (higher = more likely "useful" content)
  3. Sends the top-N scoring chunks through the LLM independently
  4. Merges the per-chunk results via majority voting

This improves recall on long papers where the single-pass trim can miss
data-rich paragraphs buried in the middle of the document.
"""

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import xgboost as xgb

from src.extraction.llm_client import extract_metrics_from_text
from src.extraction.models import PredatorDietMetrics
from src.classifier.pdf_classifier import load_classifier

log = logging.getLogger(__name__)

# fields we try to merge across chunks
_MERGE_FIELDS = [
    "species_name",
    "study_location",
    "study_date",
    "num_empty_stomachs",
    "num_nonempty_stomachs",
    "sample_size",
]


def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 500,
    min_chunk_len: int = 100,
) -> List[str]:
    """Split *text* into overlapping character-level chunks.

    Tries to break at paragraph boundaries first, then sentence boundaries,
    so chunks don't start/end mid-word.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # try to snap to a paragraph break
        if end < len(text):
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break
            else:
                sent_break = text.rfind(". ", start, end)
                if sent_break > start + chunk_size // 2:
                    end = sent_break + 1

        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_len:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def score_chunks(
    chunks: List[str],
    model_dir: str = "src/classifier/models",
) -> List[Tuple[str, float]]:
    """Score each chunk using the XGBoost classifier.

    Returns a list of (chunk, score) sorted by score descending.
    Higher score = chunk looks more like a "useful" paper section.
    """
    model, vectorizer, _encoder = load_classifier(model_dir)

    scored = []
    for chunk in chunks:
        X_vec = vectorizer.transform([chunk])
        dtest = xgb.DMatrix(X_vec)
        score = float(model.predict(dtest)[0])
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def merge_results(results: List[dict]) -> dict:
    """Merge extraction dicts from multiple chunks via majority voting.

    For each field, the value that appears most often across chunks wins.
    A confidence score (votes / total) is stored alongside each field.
    """
    if not results:
        return {}

    merged = {}
    for field in _MERGE_FIELDS:
        values = [r.get(field) for r in results if r.get(field) is not None]

        if not values:
            merged[field] = None
            merged[f"{field}_confidence"] = 0.0
        else:
            counter = Counter(values)
            most_common_val, most_common_count = counter.most_common(1)[0]
            merged[field] = most_common_val
            merged[f"{field}_confidence"] = round(most_common_count / len(values), 2)

    return merged


def extract_with_chunking(
    text: str,
    model_dir: str = "src/classifier/models",
    llm_model: str = "qwen2.5:7b",
    num_ctx: int = 8192,
    top_n: int = 3,
    chunk_size: int = 4000,
    overlap: int = 500,
) -> dict:
    """Full chunked extraction pipeline.

    Args:
        text:       Full document text.
        model_dir:  Path to XGBoost model artifacts.
        llm_model:  Ollama model name for extraction.
        num_ctx:    Context window size for Ollama.
        top_n:      Number of highest-scoring chunks to extract from.
        chunk_size: Character size per chunk.
        overlap:    Overlap between consecutive chunks.

    Returns:
        Merged metrics dict with per-field confidence scores and
        fraction_feeding computed from the merged counts.
    """
    # chunk
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"  [CHUNK] Split into {len(chunks)} chunks", file=sys.stderr)

    if not chunks:
        log.warning("No chunks produced from text of length %d", len(text))
        return {}

    # score
    scored = score_chunks(chunks, model_dir=model_dir)
    top_chunks = scored[:top_n]
    scores_str = ", ".join(f"{s:.3f}" for _, s in top_chunks)
    print(f"  [CHUNK] Top {len(top_chunks)} chunk scores: [{scores_str}]", file=sys.stderr)

    # extract from each chunk
    results = []
    for i, (chunk, score) in enumerate(top_chunks):
        print(f"  [CHUNK] Extracting from chunk {i + 1}/{len(top_chunks)} (score={score:.3f})...", file=sys.stderr)
        try:
            metrics = extract_metrics_from_text(
                text=chunk,
                model=llm_model,
                num_ctx=num_ctx,
            )
            result = metrics.model_dump()
            results.append(result)
            print(
                f"    Got: species={result.get('species_name')}, " f"n={result.get('sample_size')}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"    Failed: {e}", file=sys.stderr)
            log.error("Chunk %d extraction failed: %s", i + 1, e)

    if not results:
        log.warning("All chunk extractions failed")
        return {}

    # merge via voting
    merged = merge_results(results)

    # compute fraction_feeding from merged counts
    nonempty = merged.get("num_nonempty_stomachs")
    sample = merged.get("sample_size")
    if nonempty is not None and sample is not None and sample > 0:
        merged["fraction_feeding"] = round(nonempty / sample, 4)
    else:
        merged["fraction_feeding"] = None

    return merged
