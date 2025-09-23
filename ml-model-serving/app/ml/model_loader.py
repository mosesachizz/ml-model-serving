import logging
import pickle
import asyncio
from typing import Any, List, Optional
import numpy as np
from pathlib import Path

from app.utils.logger import logger

class ModelLoader:
    def __init__(self):
        self.supported_formats = ['.pkl', '.joblib', '.h5', '.onnx']
    
    async def load_model(self, model_path: str) -> Any:
        """Load a model from the given path"""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load based on file format
            if model_path.suffix == '.pkl':
                return await self._load_pickle_model(model_path)
            elif model_path.suffix == '.joblib':
                return await self._load_joblib_model(model_path)
            elif model_path.suffix == '.h5':
                return await self._load_keras_model(model_path)
            elif model_path.suffix == '.onnx':
                return await self._load_onnx_model(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")
                
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    
    async def predict(self, model: Any, features: List[Any]) -> List[Any]:
        """Make predictions using the loaded model"""
        try:
            # Convert to numpy array if needed
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Handle different model types
            if hasattr(model, 'predict'):
                predictions = model.predict(features)
            elif hasattr(model, 'forward'):
                # PyTorch model
                import torch
                with torch.no_grad():
                    features_tensor = torch.from_numpy(features).float()
                    predictions = model(features_tensor).numpy()
            else:
                raise ValueError("Model does not have predict method")
            
            # Convert predictions to list
            if hasattr(predictions, 'tolist'):
                return predictions.tolist()
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    async def _load_pickle_model(self, model_path: Path) -> Any:
        """Load a pickle model"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    async def _load_joblib_model(self, model_path: Path) -> Any:
        """Load a joblib model"""
        try:
            import joblib
            return joblib.load(model_path)
        except ImportError:
            logger.warning("joblib not installed, falling back to pickle")
            return await self._load_pickle_model(model_path)
    
    async def _load_keras_model(self, model_path: Path) -> Any:
        """Load a Keras/TensorFlow model"""
        try:
            from tensorflow.keras.models import load_model
            return load_model(model_path)
        except ImportError:
            raise ImportError("TensorFlow is required to load .h5 models")
    
    async def _load_onnx_model(self, model_path: Path) -> Any:
        """Load an ONNX model"""
        try:
            import onnxruntime as ort
            return ort.InferenceSession(model_path)
        except ImportError:
            raise ImportError("ONNX Runtime is required to load .onnx models")
    
    def supports_format(self, file_format: str) -> bool:
        """Check if a file format is supported"""
        return file_format.lower() in self.supported_formats