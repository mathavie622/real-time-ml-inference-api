
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    response = client.post(
        "/predict",
        json={
            "text": "I really love this service"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "text" in data
    assert "prediction" in data
    assert "probability" in data
    assert "confidence" in data

    assert data["prediction"] in ["positive", "negative"]
    assert 0 <= data["probability"] <= 1
    assert 0 <= data["confidence"] <= 1


def test_invalid_request():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422
