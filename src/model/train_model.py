"""
PDF FracFeedExtractor Classifier Model Training
-------------------------

This module trains a PDF classifier using TF-IDF vectorization + XGBoost
to determine whether a PDF is useful for predator diet data analysis.
"""

import logging
import sys
from collections import Counter
from pathlib import Path

import joblib
import json
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import setup_logging

log = logging.getLogger(__name__)


def load_labeled_data(data_dir="data/processed-text", labels_file="data/labels.json"):
    data_dir = Path(data_dir)

    with open(labels_file, "r", encoding="utf-8") as f:
        labels_map = json.load(f)

    texts, labels, filenames = [], [], []
    for txt_file in data_dir.glob("*.txt"):
        fname = txt_file.name
        if fname in labels_map:
            with open(txt_file, "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(labels_map[fname])
                filenames.append(fname)
        else:
            print(f"[WARN] No label found for {fname}, skipping.")
            log.warning("No label found for %s — skipping.", fname)

    return texts, labels, filenames


def train_pdf_classifier(texts, labels, output_dir="src/model/models"):

    if not texts or not labels:
        print("[ERROR] No training samples found.")
        log.error("No training samples found — cannot train classifier.")
        return None

    class_counts = Counter(labels)
    if len(class_counts) < 2:
        print("[ERROR] Need at least two classes.")
        log.error("Need at least two classes to train — found: %s", list(class_counts.keys()))
        return None

    if any(c < 2 for c in class_counts.values()):
        print("[ERROR] Each class needs at least 2 samples.")
        log.error("Each class needs at least 2 samples. Current counts: %s", dict(class_counts))
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    except ValueError as e:
        print(f"[ERROR] train_test_split failed: {e}")
        log.error("train_test_split failed: %s", e)
        return None

    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_test = enc.transform(y_test)

    num_pos = sum(y_train)
    num_neg = len(y_train) - num_pos
    scale_pos_weight = num_neg / max(num_pos, 1)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 3),
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    dtrain = xgb.DMatrix(X_train_vec, label=y_train)
    dtest = xgb.DMatrix(X_test_vec, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "alpha": 1.0,
        "lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=20,
        verbose_eval=True,
    )

    y_pred_prob = model.predict(dtest)
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]

    acc = accuracy_score(y_test, y_pred)
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, digits=2))
    print("========================\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_model(str(Path(output_dir) / "pdf_classifier.json"))
    joblib.dump(vectorizer, Path(output_dir) / "tfidf_vectorizer.pkl")
    joblib.dump(enc, Path(output_dir) / "label_encoder.pkl")

    return {"accuracy": acc}


if __name__ == "__main__":
    setup_logging()

    texts, labels, _ = load_labeled_data()
    result = train_pdf_classifier(texts, labels, "src/model/models")
    if result is None:
        sys.exit(1)
    print(f"Model trained successfully! Accuracy: {result['accuracy']:.2f}")
