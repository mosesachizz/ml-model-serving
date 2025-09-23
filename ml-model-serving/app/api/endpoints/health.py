from fastapi import APIRouter, Depends
from datetime import datetime
import psutil
import os

from app.models.schemas import HealthCheck
from app.core.config import settings

router = APIRouter()

@router.get("", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "process_memory": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        },
        "service": {
            "environment": settings.ENVIRONMENT,
            "model_storage": settings.MODEL_STORAGE_TYPE,
            "default_model": settings.DEFAULT_MODEL_VERSION
        }
    }