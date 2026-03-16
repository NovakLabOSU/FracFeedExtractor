import pytest
import joblib
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from unittest.mock import patch
from src.model.pdf_classifier import classify_pdf


@pytest.fixture
def model_dir_with_mock_model(tmp_path):
    """Prepare a minimal XGBoost model, vectorizer, and label encoder on disk."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)

    # Fake small training dataset
    texts = ["predator stomach content", "fish prey analysis", "rock study geology", "mineral content paper"]
    labels = ["useful", "useful", "not useful", "not useful"]

    # Encode labels
    enc = LabelEncoder()
    y = enc.fit_transform(labels)

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=6)
    X = vectorizer.fit_transform(texts)

    # Train tiny XGBoost model
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    model = xgb.train(params, dtrain, num_boost_round=3)

    # Save artifacts
    model.save_model(str(model_dir / "pdf_classifier.json"))
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.pkl")
    joblib.dump(enc, model_dir / "label_encoder.pkl")

    return model_dir


@patch("src.model.pdf_classifier.extract_text_from_pdf", return_value="predator stomach content analysis")
def test_classify_pdf_valid_case(mock_extract, model_dir_with_mock_model, capsys):
    test_pdf = Path("tests/test.pdf")
    classify_pdf(test_pdf, model_dir_with_mock_model)

    output = capsys.readouterr().out
    assert "PDF Classification Result" in output
    assert "useful" in output.lower()
    assert "%" in output


def test_classify_pdf_missing_model(capsys, tmp_path):
    """Verifier missing model directory gracefully errors."""
    model_dir = tmp_path / "empty"
    model_dir.mkdir()

    classify_pdf(Path("tests/test.pdf"), model_dir)

    output = capsys.readouterr().err
    assert "[ERROR]" in output
    assert "Missing model" in output


@patch("src.model.pdf_classifier.extract_text_from_pdf", return_value="")
def test_classify_pdf_no_text(mock_extract, model_dir_with_mock_model, capsys):
    classify_pdf(Path("tests/empty.pdf"), model_dir_with_mock_model)

    output = capsys.readouterr().err
    assert "[ERROR]" in output
    assert "No text extracted" in output


@patch("src.model.pdf_classifier.extract_text_from_pdf", return_value="diet prey stomach analysis")
def test_classify_pdf_prediction_output(mock_extract, model_dir_with_mock_model, capsys):
    classify_pdf(Path("tests/valid.pdf"), model_dir_with_mock_model)

    output = capsys.readouterr().out
    assert "Prediction:" in output
    assert "%" in output


def test_classify_pdf_handles_nonexistent_dir(tmp_path, capsys):
    classify_pdf(Path("tests/missing.pdf"), tmp_path / "nope")

    output = capsys.readouterr().err
    assert "[ERROR]" in output
    assert "Missing model" in output
