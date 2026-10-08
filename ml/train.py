"""Train the XGBoost fraud classifier and save it to app/models/.

Usage:
    python -m ml.train
"""
import os
import pickle

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from ml.preprocess import load_data

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "app", "models")


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Decision threshold maximising F1 on the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    precision, recall = precision[:-1], recall[:-1]
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print("Best threshold:", best_threshold)

    # Cross-validated AUC on the training split
    cv_scores = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=3)
    print("CV AUC mean (train split):", cv_scores.mean())

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "saved_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, "threshold.txt"), "w") as f:
        f.write(str(best_threshold))

    print("Model and threshold saved.")


if __name__ == "__main__":
    main()
