import argparse
import joblib
from pathlib import Path
import xgboost as xgb
import sys
import os
import time

# Setup project root path, must be before other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.preprocessing.pdf_text_extraction import extract_text_from_pdf

# Try to import structured output module
try:
    from src.output.structured_output import ClassificationResult, OutputManager

    STRUCTURED_OUTPUT_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Could not import structured_output: {e}")
    STRUCTURED_OUTPUT_AVAILABLE = False
    ClassificationResult = None
    OutputManager = None


def classify_pdf(pdf_path, model_dir="src/model/models", return_result=False):
    """Classify a single PDF as useful or not useful."""

    start_time = time.time()
    model_path = Path(model_dir) / "pdf_classifier.json"
    vectorizer_path = Path(model_dir) / "tfidf_vectorizer.pkl"
    encoder_path = Path(model_dir) / "label_encoder.pkl"
    filename = Path(pdf_path).name

    if not model_path.exists() or not vectorizer_path.exists() or not encoder_path.exists():
        print(f"[ERROR] Missing model, encoder, or vectorizer in {model_dir}")
        if return_result and STRUCTURED_OUTPUT_AVAILABLE:
            return ClassificationResult(
                filename=filename,
                classification="unknown",
                confidence=0.0,
                error=f"Missing model files in {model_dir}",
            )
        return None

    # Load model, encoder, and TF-IDF vectorizer
    model = xgb.Booster()
    model.load_model(str(model_path))
    vectorizer = joblib.load(vectorizer_path)
    encoder = joblib.load(encoder_path)

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"[ERROR] No text extracted from {pdf_path}. Skipping classification.")
        if return_result and STRUCTURED_OUTPUT_AVAILABLE:
            return ClassificationResult(
                filename=filename,
                classification="unknown",
                confidence=0.0,
                error="No text extracted from PDF",
            )
        return None

    # Transform text into vectorized TF-IDF format
    X_vec = vectorizer.transform([text])

    # Wrap in DMatrix for XGBoost prediction
    dtest = xgb.DMatrix(X_vec)
    pred_prob = float(model.predict(dtest)[0])
    pred_class = 1 if pred_prob >= 0.70 else 0

    # Convert numeric class back into original label name
    pred_label = encoder.inverse_transform([pred_class])[0]

    if pred_class == 0:
        confidence = 1 - pred_prob
    else:
        confidence = pred_prob

    processing_time = time.time() - start_time

    print("\n=== PDF Classification Result ===")
    print(f" File: {filename}")
    print(f" Prediction: {pred_label} ({confidence:.2%} confidence)")
    print("=================================\n")

    if return_result and STRUCTURED_OUTPUT_AVAILABLE:
        return ClassificationResult(
            filename=filename,
            classification=pred_label,
            confidence=float(confidence),
            processing_time_seconds=processing_time,
            text_length=len(text),
        )
    return None


def classify_folder(folder_path, model_dir="src/model/models", output_dir="data/results"):
    """Classify all PDFs in a folder and export results."""

    if not STRUCTURED_OUTPUT_AVAILABLE:
        print("[ERROR] Structured output module not available.")
        print("Make sure src/output/structured_output.py exists.")
        return {}

    folder = Path(folder_path)
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder_path}")
        return {}

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] No PDF files found in {folder_path}")
        return {}

    print(f"Found {len(pdf_files)} PDF files to classify.")

    manager = OutputManager(output_dir=output_dir)

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        result = classify_pdf(str(pdf_path), model_dir=model_dir, return_result=True)
        if result:
            manager.add_classification(result)

    # Export results
    paths = manager.export_all()

    # Print summary
    print("\n=== Classification Summary ===")
    summary = manager.get_summary()
    print(f" Total files: {summary['total_classifications']}")
    print(f" Useful: {summary['useful_count']}")
    print(f" Not useful: {summary['not_useful_count']}")
    print(f" Avg confidence: {summary['average_classification_confidence']:.2%}")
    print("==============================\n")

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify PDFs as useful or not useful.")
    parser.add_argument("--pdf-path", type=str, help="Path to a single PDF file to classify.")
    parser.add_argument("--folder", type=str, help="Path to a folder of PDFs to classify.")
    parser.add_argument("--model_dir", type=str, default="src/model/models", help="Directory containing the trained model and TF-IDF vectorizer.")
    parser.add_argument("--output_dir", type=str, default="data/results", help="Directory for output files (JSON/CSV).")
    args = parser.parse_args()

    if args.folder:
        paths = classify_folder(args.folder, args.model_dir, args.output_dir)
        if paths:
            print("Exported files:")
            for name, path in paths.items():
                print(f"  {name}: {path}")
    elif args.pdf_path:
        classify_pdf(args.pdf_path, args.model_dir)
    else:
        parser.print_help()
