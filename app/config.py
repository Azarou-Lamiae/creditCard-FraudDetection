"""Central place for file paths used by the web app."""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(ROOT_DIR, "reports", "metrics.json")


def get_db_path():
    """SQLite file storing the prediction history (not versioned).

    Override with the FRAUD_DB_PATH environment variable.
    """
    default = os.path.join(ROOT_DIR, "instance", "history.db")
    path = os.environ.get("FRAUD_DB_PATH", default)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
