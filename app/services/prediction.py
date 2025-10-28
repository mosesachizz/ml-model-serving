import logging
from typing import List, Dict, Any, Optional
from app.ml.model_manager import ModelManager

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    async def predict(
        self, 
        features: List[Any], 
        model_version: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Business logic for prediction requests"""
        try:
            # Validate input
            if not features:
                raise ValueError("Features cannot be empty")
            
            # Get default version if not specified
            version = model_version or "v1"
            
            # Make prediction
            result = await self.model_manager.predict(version, features, request_id)
            
            return {
                "success": True,
                "data": result,
                "request_id": request_id
            }
            
        except Exception as e:
            logger.error(f"Prediction service error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id
            }
    
    async def batch_predict(
        self, 
        requests: List[Dict[str, Any]],
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Business logic for batch predictions"""
        try:
            if not requests:
                raise ValueError("Batch requests cannot be empty")
            
            results = []
            for i, req in enumerate(requests):
                features = req.get('features', [])
                model_version = req.get('model_version')
                
                result = await self.predict(features, model_version, f"{request_id}_{i}")
                results.append(result)
            
            return {
                "success": True,
                "results": results,
                "total_requests": len(requests)
            }
            
        except Exception as e:
            logger.error(f"Batch prediction service error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id
            }