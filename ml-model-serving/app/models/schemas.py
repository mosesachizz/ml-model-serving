from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    features: List[Any]
    model_version: Optional[str] = None

class PredictionResponse(BaseModel):
    request_id: str
    predictions: List[Any]
    model_version: str
    inference_time: float
    metadata: Optional[Dict[str, Any]] = None

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

class ModelInfo(BaseModel):
    version: str
    metadata: Dict[str, Any]
    loaded: bool
    loaded_at: Optional[str] = None

class ModelUpdateRequest(BaseModel):
    model_data: bytes
    metadata: Dict[str, Any]

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

class MonitoringStats(BaseModel):
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    average_latency: float
    throughput: float
    recent_errors: List[Dict[str, Any]]