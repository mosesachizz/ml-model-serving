from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any

from app.ml.model_manager import ModelManager
from app.ml.monitoring import ModelMonitor
from app.utils.logger import logger

router = APIRouter()

@router.get("/monitoring/models/{version}/stats")
async def get_model_stats(version: str, model_manager: ModelManager = Depends()):
    """Get monitoring statistics for a specific model version"""
    try:
        stats = await model_manager.get_model_stats(version)
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"No statistics available for model version {version}"
            )
        return stats
    except Exception as e:
        logger.error(f"Failed to get model stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get model statistics"
        )

@router.get("/monitoring/models/{version}/drift")
async def check_data_drift(version: str, window_size: int = 100):
    """Check for data drift in recent predictions"""
    try:
        # This would be implemented with your actual drift detection
        return {
            "drift_detected": False,
            "confidence": 0.0,
            "message": "Drift detection not implemented in this example"
        }
    except Exception as e:
        logger.error(f"Failed to check data drift: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to check data drift"
        )

@router.get("/monitoring/overview")
async def get_monitoring_overview(model_manager: ModelManager = Depends()):
    """Get overview of all model monitoring data"""
    try:
        models = await model_manager.list_models()
        overview = {}
        
        for model in models:
            stats = await model_manager.get_model_stats(model['version'])
            overview[model['version']] = stats
        
        return overview
    except Exception as e:
        logger.error(f"Failed to get monitoring overview: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get monitoring overview"
        )