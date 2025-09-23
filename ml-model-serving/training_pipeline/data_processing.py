import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None
    
    async def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess data"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    async def preprocess_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Preprocess the data"""
        try:
            # Handle missing values
            df = df.dropna(subset=[target_column])
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Impute missing values
            X_imputed = self.imputer.fit_transform(X)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_imputed)
            
            # Create processed DataFrame
            processed_df = pd.DataFrame(X_scaled, columns=self.feature_columns)
            processed_df[target_column] = y.reset_index(drop=True)
            
            logger.info(f"Processed data shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    def get_preprocessing_config(self) -> dict:
        """Get preprocessing configuration for model metadata"""
        return {
            "imputer_strategy": "mean",
            "scaler_type": "StandardScaler",
            "feature_columns": self.feature_columns
        }