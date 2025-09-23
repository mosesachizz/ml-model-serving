from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import List, Optional, Dict, Any
import uuid

from app.core.config import settings
from app.ml.model_manager import ModelManager
from app.models.schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest
from app.utils.logger import logger

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends()
):
    """Make a prediction using the specified model version"""
    try:
        request_id = str(uuid.uuid4())
        
        result = await model_manager.predict(
            version=request.model_version or settings.DEFAULT_MODEL_VERSION,
            features=request.features,
            request_id=request_id
        )
        
        return PredictionResponse(
            request_id=request_id,
            predictions=result["predictions"],
            model_version=result["model_version"],
            inference_time=result["inference_time"],
            metadata=result["metadata"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )

@router.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends()
):
    """Make batch predictions"""
    try:
        if len(request.requests) > settings.MAX_PREDICTION_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size exceeds maximum of {settings.MAX_PREDICTION_BATCH_SIZE}"
            )
        
        results = []
        for pred_request in request.requests:
            request_id = str(uuid.uuid4())
            
            result = await model_manager.predict(
                version=pred_request.model_version or settings.DEFAULT_MODEL_VERSION,
                features=pred_request.features,
                request_id=request_id
            )
            
            results.append(PredictionResponse(
                request_id=request_id,
                predictions=result["predictions"],
                model_version=result["model_version"],
                inference_time=result["inference_time"],
                metadata=result["metadata"]
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )

@router.get("/predict/versions")
async def get_model_versions(model_manager: ModelManager = Depends()):
    """Get available model versions"""
    try:
        versions = await model_manager.list_models()
        return {"versions": versions}
    except Exception as e:
        logger.error(f"Failed to get model versions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model versions"
        )