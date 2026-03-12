"""
PDF FracFeedExtractor Classifier Model Training
-------------------------

This module trains a PDF classifier using TF-IDF vectorization + XGBoost
to determine whether a PDF is useful for predator diet data analysis.
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import xgboost as xgb
import joblib
import json
import sys
from collections import Counter


# Load training texts and their labels
def load_labeled_data(data_dir="data/processed-text", labels_file="data/labels.json"):
    data_dir = Path(data_dir)

    # Load label dictionary
    with open(labels_file, "r", encoding="utf-8") as f:
        labels_map = json.load(f)

    # Iterate through processed text files
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

    return texts, labels, filenames


# Train an XGBoost text classifier with TF-IDF features.
def train_pdf_classifier(texts, labels, output_dir="src/model/models"):

    # Ensure dataset is not empty
    if not texts or not labels:
        print("[ERROR] No training samples found.")
        return None

    # Need at least 2 classes to learn a classifier
    class_counts = Counter(labels)
    if len(class_counts) < 2:
        print("[ERROR] Need at least two classes.")
        return None

    if any(c < 2 for c in class_counts.values()):
        print("[ERROR] Each class needs at least 2 samples.")
        return None

    try:
        # Stratified splitter ensures class balance in train/test sets
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    except ValueError as e:
        print(f"[ERROR] train_test_split failed: {e}")
        return None

    # Convert target labels into numeric form
    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_test = enc.transform(y_test)

    # Calculate scale_pos_weight for imbalanced classes
    num_pos = sum(y_train)
    num_neg = len(y_train) - num_pos
    scale_pos_weight = num_neg / max(num_pos, 1)

    # Build TF-IDF vectorizer for extracting text features
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 3),
    )

    # Fit TF-IDF transformer on training data and apply to test data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_vec, label=y_train)
    dtest = xgb.DMatrix(X_test_vec, label=y_test)

    # Use GPU if available (e.g. HPC gpu nodes), fall back to CPU
    try:
        _probe = xgb.DMatrix(X_train_vec[:1], label=y_train[:1])
        xgb.train({"device": "cuda", "tree_method": "hist"}, _probe, num_boost_round=1)
        device, tree_method = "cuda", "hist"
        print("[INFO] GPU detected — training with CUDA.")
    except xgb.core.XGBoostError:
        device, tree_method = "cpu", "hist"
        print("[INFO] No GPU available — training on CPU.")

    # XGBoost parameters
    params = {
        "device": device,
        "tree_method": tree_method,
        "objective": "binary:logistic",  # binary classification
        "eval_metric": "logloss",  # log loss metric
        "eta": 0.05,  # learning rate
        "max_depth": 6,
        "subsample": 0.8,  # use 80% of data per boosting round
        "alpha": 1.0,  # L1 regularization
        "lambda": 1.0,  # L2 regularization
        "scale_pos_weight": scale_pos_weight,
    }

    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dtrain, "train"), (dtest, "eval")], early_stopping_rounds=20, verbose_eval=True)  # stop if no improvement for 20 rounds

    # Predict on test set and convert probabilities to labels
    y_pred_prob = model.predict(dtest)
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]

    # Evaluate accuracy
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, digits=2))
    print("========================\n")

    # Save artifacts
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_model(str(Path(output_dir) / "pdf_classifier.json"))
    joblib.dump(vectorizer, Path(output_dir) / "tfidf_vectorizer.pkl")
    joblib.dump(enc, Path(output_dir) / "label_encoder.pkl")

    return {"accuracy": acc}


if __name__ == "__main__":
    texts, labels, _ = load_labeled_data()
    result = train_pdf_classifier(texts, labels, "src/model/models")
    if result is None:
        sys.exit(1)
    print(f"Model trained successfully! Accuracy: {result['accuracy']:.2f}")
