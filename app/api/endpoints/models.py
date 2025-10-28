from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional, Dict, Any
import json

from app.ml.model_manager import ModelManager
from app.models.schemas import ModelInfo, ModelUpdateRequest
from app.utils.logger import logger

router = APIRouter()

@router.get("/models", response_model=List[ModelInfo])
async def list_models(model_manager: ModelManager = Depends()):
    """List all available models with their information"""
    try:
        models = await model_manager.list_models()
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list models"
        )

@router.get("/models/{version}", response_model=ModelInfo)
async def get_model_info(version: str, model_manager: ModelManager = Depends()):
    """Get information about a specific model version"""
    try:
        model_info = await model_manager.get_model_info(version)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version {version} not found"
            )
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )

@router.post("/models/{version}", status_code=status.HTTP_201_CREATED)
async def update_model(
    version: str,
    model_file: UploadFile = File(...),
    metadata: str = Form(...),
    model_manager: ModelManager = Depends()
):
    """Update or add a new model version"""
    try:
        # Parse metadata
        try:
            model_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid metadata JSON"
            )
        
        # Read model file
        model_data = await model_file.read()
        
        # Update model
        await model_manager.update_model(version, model_data, model_metadata)
        
        return {"message": f"Model version {version} updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update model"
        )

@router.delete("/models/{version}", status_code=status.HTTP_200_OK)
async def delete_model(version: str, model_manager: ModelManager = Depends()):
    """Delete a model version"""
    try:
        await model_manager.unload_model(version)
        # Note: This only unloads from memory. For permanent deletion, 
        # you'd need to implement storage deletion logic.
        return {"message": f"Model version {version} unloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete model"
        )