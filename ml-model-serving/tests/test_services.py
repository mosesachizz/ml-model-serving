import pytest
from app.services.prediction import PredictionService
from app.services.training import TrainingService
from app.services.monitoring import MonitoringService

@pytest.mark.asyncio
async def test_prediction_service():
    """Test prediction service"""
    service = PredictionService(None)  # Mock model manager
    
    # Test validation
    result = await service.predict([], "v1")
    assert result["success"] == False
    assert "error" in result

@pytest.mark.asyncio
async def test_training_service():
    """Test training service"""
    service = TrainingService()
    
    # Test with sample data
    import pandas as pd
    import numpy as np
    
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    result = await service.train_model(data, 'target')
    assert result["success"] == True
    assert "model" in result
    assert "metadata" in result

@pytest.mark.asyncio
async def test_monitoring_service():
    """Test monitoring service"""
    service = MonitoringService()
    
    # Test recording predictions
    await service.record_prediction("v1", [1, 2, 3], 0, 0.1, "test-123")
    assert len(service.predictions_history) == 1
    
    # Test recording errors
    await service.record_error("v1", "Test error", {"data": "test"})
    assert len(service.errors_history) == 1
    
    # Test performance metrics
    metrics = await service.get_performance_metrics("v1", 1)
    assert isinstance(metrics, dict)
    
    # Test drift detection (will return empty without enough data)
    drift = await service.check_data_drift("v1", [[1, 2, 3], [4, 5, 6]])
    assert isinstance(drift, dict)