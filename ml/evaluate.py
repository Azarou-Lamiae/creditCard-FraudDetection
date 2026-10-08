"""Evaluate the saved model on the held-out test split and write reports/metrics.json.

Usage:
    python -m ml.evaluate
"""
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from ml.preprocess import load_data

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "app", "models", "saved_model.pkl")
THRESH_PATH = os.path.join(ROOT_DIR, "app", "models", "threshold.txt")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
METRICS_PATH = os.path.join(ROOT_DIR, "reports", "metrics.json")


def main():
    _, X_test, _, y_test = load_data()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    try:
        with open(THRESH_PATH, "r") as f:
            threshold = float(f.read())
    except (OSError, ValueError):
        threshold = 0.5

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    print("AUC:", auc)
    print()
    print(classification_report(y_test, y_pred))

    os.makedirs(DOCS_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Confusion matrix (test split)")
    plt.savefig(os.path.join(DOCS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (test split)")
    plt.savefig(os.path.join(DOCS_DIR, "precision_recall_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    metrics = {
        "auc": float(auc),
        "threshold": float(threshold),
        "features": int(X_test.shape[1]),
        "model": "FraudDetector v1",
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Figures saved in docs/, metrics saved to reports/metrics.json")


if __name__ == "__main__":
    main()
