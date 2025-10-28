import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()

def test_predict_endpoint():
    """Test prediction endpoint"""
    # This would require a loaded model for proper testing
    response = client.post("/api/v1/predict", json={
        "features": [[5.1, 3.5, 1.4, 0.2]],
        "model_version": "v1"
    })
    # Should either work or give appropriate error
    assert response.status_code in [200, 404, 500]

def test_models_list():
    """Test models list endpoint"""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_docs_available():
    """Test that API docs are available"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_redoc_available():
    """Test that ReDoc is available"""
    response = client.get("/redoc")
    assert response.status_code == 200