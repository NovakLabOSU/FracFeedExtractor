"""PDF classifier: predict whether a PDF is useful for predator diet data extraction.

Exposes three reusable functions for use by other modules:
    load_classifier()   – load model artifacts from disk once
    classify_text()     – score already-extracted text (no I/O)
    classify_pdf()      – convenience wrapper: extract text + classify + print

Usage (standalone):
    python pdf_classifier.py --pdf-path path/to/file.pdf
    python pdf_classifier.py --pdf-path path/to/file.pdf --model-dir src/classifier/models
    python pdf_classifier.py --pdf-path path/to/file.pdf --threshold 0.80
"""

import argparse
import functools
import logging
import re
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import xgboost as xgb

from src.io.pdf_text_extraction import extract_text_from_pdf
from src.utils.logger import setup_logging

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

log = logging.getLogger(__name__)


def load_classifier(model_dir="src/classifier/models") -> Tuple:
    """Load the trained classifier artifacts from disk, cached by directory path.

    Args:
        model_dir: Directory containing:
            - ``pdf_classifier.json``   (XGBoost model)
            - ``tfidf_vectorizer.pkl``  (fitted TF-IDF vectorizer)
            - ``label_encoder.pkl``     (fitted LabelEncoder)

    Returns:
        Tuple of ``(model, vectorizer, encoder)``.

    Raises:
        FileNotFoundError: If any of the three artifacts are missing.
    """
    return _load_classifier_cached(str(model_dir))


@functools.lru_cache(maxsize=None)
def _load_classifier_cached(model_dir: str) -> Tuple:
    model_path = Path(model_dir) / "pdf_classifier.json"
    vectorizer_path = Path(model_dir) / "tfidf_vectorizer.pkl"
    encoder_path = Path(model_dir) / "label_encoder.pkl"

    missing = [p for p in [model_path, vectorizer_path, encoder_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts in '{model_dir}': " + ", ".join(p.name for p in missing))

    model = xgb.Booster()
    model.load_model(str(model_path))
    vectorizer = joblib.load(vectorizer_path)
    encoder = joblib.load(encoder_path)
    return model, vectorizer, encoder


def classify_text(
    text: str,
    model,
    vectorizer,
    encoder,
    threshold: float = 0.70,
) -> Tuple[str, float, float]:
    """Classify already-extracted text as 'useful' or 'not useful'.

    Args:
        text: Full text extracted from a PDF.
        model: Loaded XGBoost Booster (from :func:`load_classifier`).
        vectorizer: Fitted TF-IDF vectorizer (from :func:`load_classifier`).
        encoder: Fitted LabelEncoder (from :func:`load_classifier`).
        threshold: Probability threshold above which the PDF is 'useful'.

    Returns:
        Tuple of ``(label, confidence, pred_prob)``.
    """
    X_vec = vectorizer.transform([text])
    dtest = xgb.DMatrix(X_vec)
    pred_prob = float(model.predict(dtest)[0])
    pred_class = 1 if pred_prob >= threshold else 0
    label = encoder.inverse_transform([pred_class])[0]
    if label == "not-useful":
        label = "not useful"
    confidence = pred_prob if pred_class == 1 else (1.0 - pred_prob)
    return label, confidence, pred_prob


def explain_classification(
    text: str,
    model,
    vectorizer,
    top_n: int = 10,
) -> list[tuple[str, float, list[int]]]:
    """Return the top-N terms driving a classification and the pages they appear on.

    Combines the TF-IDF weight of each term in *text* with the XGBoost global
    feature importance (gain) to produce a document-specific relevance score.
    Terms that appear in the document AND are important to the classifier rank
    highest. This is an approximation — for exact per-prediction attribution,
    SHAP values would be needed.

    Args:
        text: Full extracted document text (may contain [PAGE N] markers).
        model: Loaded XGBoost Booster.
        vectorizer: Fitted TF-IDF vectorizer.
        top_n: Number of top terms to return.

    Returns:
        List of ``(term, combined_score, page_numbers)`` tuples, sorted by
        combined_score descending. ``page_numbers`` lists the pages where the
        term first appears, or ``[]`` if no [PAGE N] markers are in the text.
    """
    # TF-IDF weights for this document
    tfidf_vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    # Global feature importances from XGBoost (gain = average loss reduction)
    raw_scores = model.get_score(importance_type="gain")  # {'f123': 4.2, ...}
    importance_array = np.zeros(len(feature_names))
    for key, val in raw_scores.items():
        idx = int(key[1:])  # strip leading 'f'
        if idx < len(importance_array):
            importance_array[idx] = val

    # Combined score: TF-IDF weight × classifier gain — document-specific
    tfidf_dense = np.asarray(tfidf_vec.todense()).flatten()
    combined = tfidf_dense * importance_array

    # Top-N indices by combined score (only terms that appear in the document)
    present = np.where(tfidf_dense > 0)[0]
    if len(present) == 0:
        return []
    top_indices = present[np.argsort(combined[present])[::-1][:top_n]]

    # Resolve page numbers for each top term using [PAGE N] markers
    results = []
    for idx in top_indices:
        term = feature_names[idx]
        score = float(combined[idx])
        pages: list[int] = []
        for match in re.finditer(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
            page_markers = re.findall(r'\[PAGE (\d+)\]', text[:match.start()])
            if page_markers:
                pages.append(int(page_markers[-1]))
                break  # first occurrence only
        results.append((term, score, sorted(set(pages))))

    return results


def classify_pdf(
    pdf_path: str,
    model_dir: str = "src/classifier/models",
    threshold: float = 0.70,
) -> Tuple[str, float, float]:
    """Convenience wrapper: extract text from a PDF and classify it."""
    try:
        model, vectorizer, encoder = load_classifier(model_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        log.error("Could not load classifier: %s", e)
        return None, None, None

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"[ERROR] No text extracted from {pdf_path}. Skipping classification.", file=sys.stderr)
        log.error("No text extracted from %s — skipping classification.", pdf_path)
        return None, None, None

    label, confidence, pred_prob = classify_text(text, model, vectorizer, encoder, threshold)

    print("\n=== PDF Classification Result ===")
    print(f" File      : {Path(pdf_path).name}")
    print(f" Prediction: {label} ({confidence:.2%} confidence)")
    print(f" Raw prob  : {pred_prob:.4f}  (threshold: {threshold})")
    print("=================================\n")

    return label, confidence, pred_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a PDF as useful or not useful.")
    parser.add_argument("--pdf-path", type=str, required=True, help="Path to the PDF file to classify.")
    parser.add_argument("--model-dir", type=str, default="src/classifier/models", help="Directory containing the trained model artifacts (default: src/classifier/models).")
    parser.add_argument("--threshold", type=float, default=0.70, help="Probability threshold for the 'useful' class (default: 0.70).")
    args = parser.parse_args()

    setup_logging()
    classify_pdf(args.pdf_path, model_dir=args.model_dir, threshold=args.threshold)
