"""Unit and integration tests for FastAPI monitoring service."""

import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test that the root endpoint returns 200 OK and expected metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "FastAPI lets other applications use your trained model" in data["description"]


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data


def test_predict_single_endpoint(client):
    """Test single transaction prediction."""
    payload = {
        "Sender_account": "ACC-9901",
        "Receiver_account": "ACC-9902",
        "Amount": 15000.0,
        "Payment_currency": "US Dollar",
        "Received_currency": "Euro",
        "Sender_bank_location": "USA",
        "Receiver_bank_location": "Switzerland",
        "Payment_type": "Cross-border",
        "Date": "2023-10-15",
        "Time": "14:20:00"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_suspicious" in data
    assert "risk_score" in data
    assert "risk_level" in data
    assert data["transaction_id"] == "ACC-9901"


def test_predict_batch_endpoint(client):
    """Test batch transaction prediction."""
    payload = {
        "transactions": [
            {"Amount": 200.0, "Payment_type": "Credit card"},
            {"Amount": 250000.0, "Payment_type": "Cross-border"}
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2


def test_model_reload_endpoint(client):
    """Test hot reload of model artifacts."""
    response = client.post("/model/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
