import pytest

from app import create_app
from app import routes


class FakeDetector:
    def predict_one(self, features):
        assert len(features) == 30
        return {"prediction": 1, "probability": 0.97, "label": "fraud"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(routes, "_detector", FakeDetector())
    return create_app().test_client()


def test_home_renders(client):
    assert client.get("/").status_code == 200


def test_predict_json(client):
    response = client.post("/predict", json={"Amount": 120.5, "V1": -1.2})
    assert response.status_code == 200
    assert response.get_json()["label"] == "fraud"


def test_predict_rejects_non_json(client):
    assert client.post("/predict", data="nope").status_code == 400


def test_predict_rejects_non_numeric(client):
    assert client.post("/predict", json={"Amount": "abc"}).status_code == 400


def test_result_form_and_xss_is_escaped(client):
    ok = client.post("/result", data={"Amount": "10"})
    assert ok.status_code == 200

    bad = client.post("/result", data={"Amount": "<script>alert(1)</script>"})
    assert bad.status_code == 400
    assert b"<script>" not in bad.data


def test_history_and_dashboard_use_saved_predictions(client):
    client.post("/predict", json={"Amount": 1})
    client.post("/predict", json={"Amount": 2})

    history = client.get("/history").get_json()
    assert len(history) == 2

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
