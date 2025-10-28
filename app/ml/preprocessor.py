import logging
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

from app.utils.logger import logger

class DataPreprocessor:
    def __init__(self):
        self.scalers: Dict[str, Any] = {}
    
    async def process(self, features: List[Any], preprocessing_config: Dict) -> List[Any]:
        """Preprocess features based on configuration"""
        try:
            if not preprocessing_config:
                return features
            
            processed_features = features
            
            # Handle different preprocessing steps
            if preprocessing_config.get('normalization') == 'standard':
                processed_features = await self._standard_scale(processed_features, preprocessing_config)
            elif preprocessing_config.get('normalization') == 'minmax':
                processed_features = await self._minmax_scale(processed_features, preprocessing_config)
            
            if preprocessing_config.get('encoding') == 'onehot':
                processed_features = await self._onehot_encode(processed_features, preprocessing_config)
            
            if preprocessing_config.get('imputation') == 'mean':
                processed_features = await self._impute_missing(processed_features, preprocessing_config)
            
            return processed_features
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    async def _standard_scale(self, features: List[Any], config: Dict) -> List[Any]:
        """Apply standard scaling"""
        try:
            features_array = np.array(features)
            scaler_key = config.get('scaler_key', 'default')
            
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                # Fit on the first batch
                if config.get('fit_on_first_batch', True):
                    self.scalers[scaler_key].fit(features_array)
            
            return self.scalers[scaler_key].transform(features_array).tolist()
            
        except Exception as e:
            logger.error(f"Standard scaling failed: {str(e)}")
            return features
    
    async def _minmax_scale(self, features: List[Any], config: Dict) -> List[Any]:
        """Apply min-max scaling"""
        try:
            features_array = np.array(features)
            scaler_key = config.get('scaler_key', 'default')
            
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = MinMaxScaler(
                    feature_range=config.get('feature_range', (0, 1))
                )
                if config.get('fit_on_first_batch', True):
                    self.scalers[scaler_key].fit(features_array)
            
            return self.scalers[scaler_key].transform(features_array).tolist()
            
        except Exception as e:
            logger.error(f"MinMax scaling failed: {str(e)}")
            return features
    
    async def _onehot_encode(self, features: List[Any], config: Dict) -> List[Any]:
        """Apply one-hot encoding"""
        # This would be implemented based on specific categorical features
        # For now, return features as-is
        return features
    
    async def _impute_missing(self, features: List[Any], config: Dict) -> List[Any]:
        """Impute missing values"""
        try:
            features_array = np.array(features)
            
            # Replace NaN with mean
            if np.isnan(features_array).any():
                col_mean = np.nanmean(features_array, axis=0)
                inds = np.where(np.isnan(features_array))
                features_array[inds] = np.take(col_mean, inds[1])
            
            return features_array.tolist()
            
        except Exception as e:
            logger.error(f"Imputation failed: {str(e)}")
            return features
    
    async def fit_preprocessor(self, data: List[Any], config: Dict) -> None:
        """Fit preprocessor on training data"""
        try:
            data_array = np.array(data)
            
            if config.get('normalization') == 'standard':
                scaler_key = config.get('scaler_key', 'default')
                self.scalers[scaler_key] = StandardScaler()
                self.scalers[scaler_key].fit(data_array)
            
            elif config.get('normalization') == 'minmax':
                scaler_key = config.get('scaler_key', 'default')
                self.scalers[scaler_key] = MinMaxScaler(
                    feature_range=config.get('feature_range', (0, 1))
                )
                self.scalers[scaler_key].fit(data_array)
                
        except Exception as e:
            logger.error(f"Failed to fit preprocessor: {str(e)}")
            raise