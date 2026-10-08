import pickle

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.models.fraud_detector import FraudDetector


@pytest.fixture
def detector(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 30))
    y = (X[:, 1] > 0).astype(int)
    model = LogisticRegression(max_iter=500).fit(X, y)

    model_path = tmp_path / "model.pkl"
    thresh_path = tmp_path / "threshold.txt"
    model_path.write_bytes(pickle.dumps(model))
    thresh_path.write_text("0.5")

    return FraudDetector(str(model_path), str(thresh_path))


def test_predict_one_structure(detector):
    result = detector.predict_one([0.0] * 30)
    assert set(result) == {"prediction", "probability", "label"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in {"fraud", "normal"}


def test_label_matches_prediction(detector):
    features = [0.0] * 30
    features[1] = 5.0
    result = detector.predict_one(features)
    assert result["prediction"] == 1
    assert result["label"] == "fraud"


def test_missing_threshold_falls_back_to_default(tmp_path, detector):
    model_path = tmp_path / "m.pkl"
    model_path.write_bytes(pickle.dumps(detector.model))
    fresh = FraudDetector(str(model_path), str(tmp_path / "missing.txt"))
    assert fresh.threshold == 0.5
