# test/test_api.py

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_predict():
    r = client.post("/predict", json={"email": "Need help with my invoice"})
    assert r.status_code == 200
    data = r.json()
    assert data["category"] in {"support", "sales", "partnership", "spam"}
    assert len(data["probabilities"]) == 4
