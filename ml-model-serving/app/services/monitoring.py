import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MonitoringService:
    def __init__(self):
        self.predictions_history = []
        self.errors_history = []
    
    async def record_prediction(
        self, 
        model_version: str, 
        features: List[Any], 
        prediction: Any,
        inference_time: float,
        request_id: str
    ) -> None:
        """Record prediction for monitoring"""
        record = {
            "timestamp": datetime.now(),
            "model_version": model_version,
            "features": features,
            "prediction": prediction,
            "inference_time": inference_time,
            "request_id": request_id
        }
        self.predictions_history.append(record)
    
    async def record_error(
        self, 
        model_version: str, 
        error_message: str,
        request_data: Any = None
    ) -> None:
        """Record error for monitoring"""
        record = {
            "timestamp": datetime.now(),
            "model_version": model_version,
            "error_message": error_message,
            "request_data": request_data
        }
        self.errors_history.append(record)
    
    async def get_performance_metrics(self, model_version: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        now = datetime.now()
        time_threshold = now - timedelta(hours=hours)
        
        # Filter recent predictions
        recent_predictions = [
            p for p in self.predictions_history 
            if p['model_version'] == model_version and p['timestamp'] >= time_threshold
        ]
        
        if not recent_predictions:
            return {}
        
        # Calculate metrics
        inference_times = [p['inference_time'] for p in recent_predictions]
        
        return {
            "total_predictions": len(recent_predictions),
            "avg_inference_time": np.mean(inference_times),
            "max_inference_time": np.max(inference_times),
            "min_inference_time": np.min(inference_times),
            "throughput": len(recent_predictions) / hours
        }
    
    async def check_data_drift(self, model_version: str, reference_data: List[Any]) -> Dict[str, Any]:
        """Check for data drift compared to reference data"""
        # Simple drift detection - in production, use specialized libraries
        recent_features = []
        for pred in self.predictions_history:
            if pred['model_version'] == model_version:
                recent_features.extend(pred['features'])
        
        if not recent_features or not reference_data:
            return {"drift_detected": False, "confidence": 0.0}
        
        # Simple statistical test (would use KS-test or similar in production)
        recent_mean = np.mean(recent_features, axis=0)
        reference_mean = np.mean(reference_data, axis=0)
        
        drift_score = np.mean(np.abs(recent_mean - reference_mean) / (reference_mean + 1e-10))
        
        return {
            "drift_detected": drift_score > 0.1,
            "confidence": float(drift_score),
            "recent_data_stats": {
                "mean": recent_mean.tolist(),
                "std": np.std(recent_features, axis=0).tolist()
            },
            "reference_data_stats": {
                "mean": reference_mean.tolist(),
                "std": np.std(reference_data, axis=0).tolist()
            }
        }