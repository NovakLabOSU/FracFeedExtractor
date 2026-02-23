"""PDF classifier: predict whether a PDF is useful for predator diet data extraction.

Exposes three reusable functions for use by other modules:
    load_classifier()   – load model artifacts from disk once
    classify_text()     – score already-extracted text (no I/O)
    classify_pdf()      – convenience wrapper: extract text + classify + print

Usage (standalone):
    python pdf_classifier.py --pdf-path path/to/file.pdf
    python pdf_classifier.py --pdf-path path/to/file.pdf --model-dir src/model/models
    python pdf_classifier.py --pdf-path path/to/file.pdf --threshold 0.80
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import joblib
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocessing.pdf_text_extraction import extract_text_from_pdf


def load_classifier(model_dir: str = "src/model/models") -> Tuple:
    """Load the trained classifier artifacts from disk.

    Loads the XGBoost model, TF-IDF vectorizer, and label encoder from
    ``model_dir``. Call this once and reuse the returned objects across
    multiple calls to :func:`classify_text` to avoid repeated disk I/O.

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

    This function performs no file I/O — pass the artifacts returned by
    :func:`load_classifier` and the text returned by ``extract_text_from_pdf``.

    Args:
        text: Full text extracted from a PDF.
        model: Loaded XGBoost Booster (from :func:`load_classifier`).
        vectorizer: Fitted TF-IDF vectorizer (from :func:`load_classifier`).
        encoder: Fitted LabelEncoder (from :func:`load_classifier`).
        threshold: Probability threshold above which the PDF is 'useful'.
            Defaults to 0.70.

    Returns:
        Tuple of ``(label, confidence, pred_prob)`` where:
            - ``label`` is the decoded class name (e.g. ``'useful'``).
            - ``confidence`` is the probability of the *predicted* class (0–1).
            - ``pred_prob`` is the raw model probability for class 1 (0–1).
    """
    X_vec = vectorizer.transform([text])
    dtest = xgb.DMatrix(X_vec)
    pred_prob = float(model.predict(dtest)[0])
    pred_class = 1 if pred_prob >= threshold else 0
    label = encoder.inverse_transform([pred_class])[0]
    confidence = pred_prob if pred_class == 1 else (1.0 - pred_prob)
    return label, confidence, pred_prob


def classify_pdf(
    pdf_path: str,
    model_dir: str = "src/model/models",
    threshold: float = 0.70,
) -> Tuple[str, float, float]:
    """Convenience wrapper: extract text from a PDF and classify it.

    Loads the classifier artifacts, extracts text from the PDF, classifies
    it, prints a human-readable result, and returns the classification details.

    Args:
        pdf_path: Path to the PDF file to classify.
        model_dir: Directory containing the trained model artifacts.
        threshold: Probability threshold for the 'useful' class.

    Returns:
        Tuple of ``(label, confidence, pred_prob)``, or
        ``(None, None, None)`` if text extraction or model loading fails.
    """
    # Load model artifacts
    try:
        model, vectorizer, encoder = load_classifier(model_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return None, None, None

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"[ERROR] No text extracted from {pdf_path}. Skipping classification.", file=sys.stderr)
        return None, None, None

    # Classify
    label, confidence, pred_prob = classify_text(text, model, vectorizer, encoder, threshold)

    # Print result
    print("\n=== PDF Classification Result ===")
    print(f" File      : {Path(pdf_path).name}")
    print(f" Prediction: {label} ({confidence:.2%} confidence)")
    print(f" Raw prob  : {pred_prob:.4f}  (threshold: {threshold})")
    print("=================================\n")

    return label, confidence, pred_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a PDF as useful or not useful.")
    parser.add_argument("--pdf-path", type=str, required=True, help="Path to the PDF file to classify.")
    parser.add_argument("--model-dir", type=str, default="src/model/models", help="Directory containing the trained model artifacts (default: src/model/models).")
    parser.add_argument("--threshold", type=float, default=0.70, help="Probability threshold for the 'useful' class (default: 0.70).")
    args = parser.parse_args()

    classify_pdf(args.pdf_path, model_dir=args.model_dir, threshold=args.threshold)
