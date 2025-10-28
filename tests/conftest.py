import pytest
import asyncio
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def test_client():
    """Create a test client for the FastAPI app"""
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="module")
def test_settings():
    """Override settings for testing"""
    original_settings = settings.copy()
    settings.ENVIRONMENT = "testing"
    settings.MODEL_STORAGE_TYPE = "memory"
    yield settings
    # Restore original settings
    for key, value in original_settings.dict().items():
        setattr(settings, key, value)