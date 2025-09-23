import pytest
import numpy as np
from app.ml.model_loader import ModelLoader
from app.ml.preprocessor import DataPreprocessor

@pytest.mark.asyncio
async def test_model_loader():
    """Test model loader functionality"""
    loader = ModelLoader()
    
    # Test format support detection
    assert loader.supports_format('.pkl') == True
    assert loader.supports_format('.joblib') == True
    assert loader.supports_format('.invalid') == False

@pytest.mark.asyncio
async def test_preprocessor():
    """Test data preprocessor"""
    preprocessor = DataPreprocessor()
    
    # Test basic preprocessing
    features = [[1, 2, 3], [4, 5, 6]]
    processed = await preprocessor.process(features, {})
    
    assert len(processed) == len(features)
    assert len(processed[0]) == len(features[0])

def test_preprocessor_fit():
    """Test preprocessor fitting"""
    preprocessor = DataPreprocessor()
    
    # Test fitting
    data = [[1, 2], [3, 4], [5, 6]]
    config = {'normalization': 'standard'}
    
    # This would normally be async but we're testing the sync interface
    import asyncio
    asyncio.run(preprocessor.fit_preprocessor(data, config))
    
    assert 'default' in preprocessor.scalers

def test_feature_validation():
    """Test feature validation"""
    from app.utils.helpers import validate_features
    
    features_2d = [[1, 2, 3], [4, 5, 6]]
    features_1d = [1, 2, 3]
    
    assert validate_features(features_2d, (2, 3)) == True
    assert validate_features(features_1d, (3,)) == True
    assert validate_features(features_2d, (3, 2)) == False  # Wrong shape