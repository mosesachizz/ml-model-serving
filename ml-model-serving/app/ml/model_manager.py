import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiofiles
from pathlib import Path

from app.core.config import settings
from app.ml.model_loader import ModelLoader
from app.ml.preprocessor import DataPreprocessor
from app.ml.monitoring import ModelMonitor
from app.utils.storage import ModelStorage
from app.utils.logger import logger

class ModelManager:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.model_loader = ModelLoader()
        self.preprocessor = DataPreprocessor()
        self.monitor = ModelMonitor()
        self.storage = ModelStorage()
        
    async def load_models(self) -> None:
        """Load all available models from storage"""
        try:
            model_versions = await self.storage.list_models()
            
            for version in model_versions:
                try:
                    await self.load_model(version)
                    logger.info(f"Successfully loaded model version {version}")
                except Exception as e:
                    logger.error(f"Failed to load model version {version}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
    
    async def load_model(self, version: str) -> None:
        """Load a specific model version"""
        try:
            # Load model artifact
            model_path = await self.storage.get_model_path(version)
            model = await self.model_loader.load_model(model_path)
            
            # Load metadata
            metadata_path = await self.storage.get_metadata_path(version)
            metadata = await self.load_metadata(metadata_path)
            
            # Store model and metadata
            self.models[version] = model
            self.model_metadata[version] = metadata
            
            logger.info(f"Loaded model {version} with metadata: {metadata}")
            
        except Exception as e:
            logger.error(f"Failed to load model {version}: {str(e)}")
            raise
    
    async def unload_model(self, version: str) -> None:
        """Unload a specific model version"""
        if version in self.models:
            del self.models[version]
        if version in self.model_metadata:
            del self.model_metadata[version]
        logger.info(f"Unloaded model version {version}")
    
    async def predict(
        self, 
        version: str, 
        features: List[Any], 
        request_id: Optional[str] = None
    ) -> Dict:
        """Make predictions using the specified model version"""
        if version not in self.models:
            raise ValueError(f"Model version {version} not loaded")
        
        model = self.models[version]
        metadata = self.model_metadata[version]
        
        try:
            # Preprocess features
            processed_features = await self.preprocessor.process(
                features, 
                metadata.get("preprocessing", {})
            )
            
            # Make prediction
            start_time = datetime.now()
            predictions = await self.model_loader.predict(model, processed_features)
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Monitor prediction
            await self.monitor.record_prediction(
                version=version,
                features=features,
                predictions=predictions,
                inference_time=inference_time,
                request_id=request_id
            )
            
            return {
                "predictions": predictions,
                "model_version": version,
                "inference_time": inference_time,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for model {version}: {str(e)}")
            await self.monitor.record_error(version, str(e))
            raise
    
    async def get_model_info(self, version: str) -> Optional[Dict]:
        """Get information about a specific model version"""
        if version not in self.model_metadata:
            return None
        
        return {
            "version": version,
            "metadata": self.model_metadata[version],
            "loaded": version in self.models,
            "loaded_at": datetime.now().isoformat() if version in self.models else None
        }
    
    async def list_models(self) -> List[Dict]:
        """List all available models"""
        models_info = []
        
        for version in self.model_metadata.keys():
            info = await self.get_model_info(version)
            if info:
                models_info.append(info)
        
        return models_info
    
    async def load_metadata(self, metadata_path: str) -> Dict:
        """Load model metadata from file"""
        try:
            async with aiofiles.open(metadata_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {str(e)}")
            return {}
    
    async def update_model(self, version: str, model_data: bytes, metadata: Dict) -> None:
        """Update or add a new model version"""
        try:
            # Save model and metadata
            await self.storage.save_model(version, model_data, metadata)
            
            # Load the new model
            await self.load_model(version)
            
            logger.info(f"Successfully updated model version {version}")
            
        except Exception as e:
            logger.error(f"Failed to update model {version}: {str(e)}")
            raise
    
    async def get_model_stats(self, version: str) -> Dict:
        """Get statistics for a model version"""
        stats = await self.monitor.get_model_stats(version)
        return {
            "version": version,
            "stats": stats,
            "loaded": version in self.models
        }