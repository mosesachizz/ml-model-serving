"""
Main endpoints router configuration.
"""

from fastapi import APIRouter
from app.api.endpoints import predictions, models, monitoring, health

# Create main router
router = APIRouter()

# Include all endpoint routers
router.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
router.include_router(models.router, prefix="/api/v1", tags=["models"])
router.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])
router.include_router(health.router, prefix="/health", tags=["health"])