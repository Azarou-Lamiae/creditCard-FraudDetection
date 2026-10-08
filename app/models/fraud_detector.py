import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "saved_model.pkl")
DEFAULT_THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.txt")


class FraudDetector:
    """Wraps the trained classifier and the decision threshold."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, threshold_path=DEFAULT_THRESHOLD_PATH):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        try:
            with open(threshold_path, "r") as f:
                self.threshold = float(f.read())
        except (OSError, ValueError):
            self.threshold = 0.5

        logger.info("Model loaded with threshold: %s", self.threshold)

    def predict_one(self, features):
        X = np.array(features, dtype=float).reshape(1, -1)
        proba = self.model.predict_proba(X)[0][1]
        pred = int(proba > self.threshold)

        return {
            "prediction": pred,
            "probability": float(proba),
            "label": "fraud" if pred == 1 else "normal",
        }

    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        preds = (proba > self.threshold).astype(int)

        return [
            {
                "prediction": int(p),
                "probability": float(pr),
                "label": "fraud" if p == 1 else "normal",
            }
            for p, pr in zip(preds, proba)
        ]
