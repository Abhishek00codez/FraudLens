import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()

def test_predict_single():
    payload = {
        "timestamp": "2023-01-01T12:00:00Z",
        "user_id": 123,
        "merchant_id": 456,
        "amount": 100.50
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "threshold_used" in data
    assert "inference_time_ms" in data

def test_predict_batch():
    payload = {
        "transactions": [
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "user_id": 123,
                "merchant_id": 456,
                "amount": 100.50
            },
            {
                "timestamp": "2023-01-01T12:05:00Z",
                "user_id": 123,
                "merchant_id": 789,
                "amount": 5000.00
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total_inference_time_ms" in data
    assert len(data["results"]) == 2
    assert "fraud_probability" in data["results"][0]

def test_predict_invalid_payload():
    payload = {
        "timestamp": "2023-01-01T12:00:00Z",
        "user_id": 123,
        # missing merchant_id
        "amount": -10.0 # invalid negative amount
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity validation error
