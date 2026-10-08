import json
import logging
import sqlite3

from flask import Blueprint, jsonify, render_template, request
from markupsafe import escape

from app.config import METRICS_PATH, get_db_path
from app.models.fraud_detector import FraudDetector

logger = logging.getLogger(__name__)

main = Blueprint("main", __name__)

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

_detector = None


def get_detector():
    """Load the model on first use (keeps imports cheap and testable)."""
    global _detector
    if _detector is None:
        _detector = FraudDetector()
    return _detector


def _connect():
    return sqlite3.connect(get_db_path())


def init_db():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            probability REAL,
            label TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(probability, label):
    init_db()
    conn = _connect()
    conn.execute(
        "INSERT INTO predictions (probability, label) VALUES (?, ?)",
        (probability, label.lower()),
    )
    conn.commit()
    conn.close()


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/result", methods=["POST"])
def result():
    try:
        features = [float(request.form.get(name, 0)) for name in FEATURE_NAMES]
    except ValueError as e:
        return f"Invalid input: {escape(e)}", 400

    prediction = get_detector().predict_one(features)
    save_prediction(prediction["probability"], prediction["label"])

    return render_template("result.html", result=prediction)


@main.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    try:
        features = [float(data.get(name, 0)) for name in FEATURE_NAMES]
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    prediction = get_detector().predict_one(features)
    save_prediction(prediction["probability"], prediction["label"])

    return jsonify(prediction)


@main.route("/history")
def history():
    init_db()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()

    return jsonify(rows)


@main.route("/dashboard")
def dashboard():
    init_db()
    conn = _connect()
    rows = conn.execute("SELECT label FROM predictions").fetchall()
    conn.close()

    labels = [r[0].lower() for r in rows]
    total = len(labels)
    frauds = sum(1 for label in labels if label == "fraud")
    normal = total - frauds

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    return render_template(
        "dashboard.html",
        total=total,
        frauds=frauds,
        normal=normal,
        auc=metrics["auc"],
        threshold=metrics["threshold"],
        features=metrics["features"],
        model=metrics["model"],
    )
